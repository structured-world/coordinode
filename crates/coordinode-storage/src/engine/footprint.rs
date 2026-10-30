//! The total owned footprint of an endpoint, held to its hard limit.
//!
//! Every byte the database's files take on an endpoint with a hard limit is
//! charged here before it is written: journal segments, tables, value files,
//! manifests, text indexes, captures and scratch files. A file's charge is
//! the space the file system allocates to it, in whole allocation units, so
//! the account bounds what `st_blocks` reports, not only file lengths. One
//! file reached under several names (a hard link) is charged once.
//!
//! Two kinds of growth meet the limit differently:
//!
//! - A file growing is checked and charged in one atomic step: it lands only
//!   if the charged total stays within the limit, whoever writes it.
//! - A client write is admitted before it is journaled, against the limit
//!   less the recovery reserve: what it adds to the journal, and what its
//!   data will take when its memtable is flushed, must fit beside what is
//!   charged, what other admitted writes hold and what the memtables already
//!   owe. So a write is refused at the door rather than acknowledged and
//!   left with no room to flush.
//!
//! The recovery reserve is the part of the limit client writes never take:
//! flushes, compactions that reclaim space, and the store's own recovery
//! records run in it. It is inside the limit, not beside it.
//!
//! Bytes other processes write to the same file system are outside this
//! account; the free-space reserve ([`crate::engine::space`]) guards the
//! volume as a whole.

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use crate::engine::config::EndpointConfig;
use crate::error::StorageError;

/// The allocation unit a file system without a reported one is taken to use.
const DEFAULT_UNIT: u64 = 4096;

/// Whose growth an admission is: it decides how much of the limit it may use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admit {
    /// A client write: the limit less the recovery reserve.
    Client,
    /// The store's own recovery records: the whole limit.
    Recovery,
}

impl Admit {
    /// The class a write asked for as `self` is admitted as on this thread: a
    /// client write made inside a [`ReclaimScope`] is the store's own work
    /// that gives room back, admitted as a recovery record.
    #[must_use]
    pub fn on_this_thread(self) -> Self {
        if RECLAIMING.get() {
            Self::Recovery
        } else {
            self
        }
    }
}

/// What the memtables owe the footprint: the bytes their flushes will write.
pub type FlushDebt = Box<dyn Fn() -> u64 + Send + Sync>;

/// Room a flush of any size is held to need past its data: the table's
/// index, filter, footer and the manifest version it installs, each rounded
/// up to whole allocation units.
pub const FLUSH_ALLOWANCE: u64 = 128 * 1024;

/// The room a flush of `unflushed` memtable bytes takes while it runs,
/// bounded from above: the data, an eighth more for block and filter
/// overhead, the fixed [`FLUSH_ALLOWANCE`], and what the file system may
/// allocate ahead of the table being written ([`allocation_ahead`]). Nothing
/// for empty memtables. Held for every admitted write until its memtable is
/// flushed, so a flush always finds the room it needs and reclamation, which
/// flushes before it removes covered journal segments, never stalls.
#[must_use]
pub fn flush_debt(unflushed: u64) -> u64 {
    if unflushed == 0 {
        return 0;
    }
    // Memtable bytes are bounded by memory: no overflow.
    let table = unflushed + unflushed / 8 + FLUSH_ALLOWANCE;
    table + allocation_ahead(table)
}

/// The smallest journal segment a hard limit brings the configured size down
/// to: below it the per-segment header, footer and file cost outweigh the
/// finer reclamation.
const MIN_LIMITED_SEGMENT: u64 = 1024 * 1024;

/// The journal segment size on an endpoint with hard limit `limit`: the
/// configured size, or a sixteenth of the limit when that is smaller (never
/// below 1 MiB). Space is reclaimed a whole sealed segment at a time and the
/// segment being appended to holds its allowance ahead of the writer, so a
/// segment large beside the limit holds space the store could otherwise use.
#[must_use]
pub fn journal_segment_bytes(limit: u64, configured: u64) -> u64 {
    match limit {
        0 => configured,
        limit => configured.min((limit / 16).max(MIN_LIMITED_SEGMENT)),
    }
}

/// How far past what an endpoint holds the store's own work may grow while
/// the store opens, or while it holds more than its limit: the files an
/// empty store creates (a manifest, version and edit files per tree, the
/// format marker, the first journal segment) and what replay at open
/// writes, or the flushes and compactions that bring a store over its limit
/// back under it. An empty store was measured at under 300 KiB. Client
/// writes never use it.
pub const OVER_LIMIT_ALLOWANCE: u64 = 1024 * 1024;

std::thread_local! {
    /// Whether this thread runs maintenance ([`MaintenanceScope`]).
    static MAINTENANCE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Whether this thread runs work that gives room back ([`ReclaimScope`]).
    static RECLAIMING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Held while the thread runs work whose commits give room back, the orphan
/// blob collector's: they are admitted as the store's own recovery records
/// ([`Admit::on_this_thread`]), so a disk below its free-space reserve or an
/// endpoint at its limit, which refuse client writes, still lets the
/// collection that frees them finish. They take the upper half of the reserve
/// only, as every recovery record does; the lower half stays for the flushes
/// and compactions that complete them.
#[must_use = "the scope lasts until it drops"]
pub struct ReclaimScope {
    was: bool,
}

impl ReclaimScope {
    /// Enter reclamation on this thread until the scope drops.
    pub fn enter() -> Self {
        Self {
            was: RECLAIMING.replace(true),
        }
    }
}

impl Drop for ReclaimScope {
    fn drop(&mut self) {
        RECLAIMING.set(self.was);
    }
}

/// Held while the thread runs a compaction: the files it grows may take only
/// the room admitted writes and the memtables' flushes leave
/// ([`Footprint::room`]). A compaction's output is estimated before it
/// starts, and the file system allocates ahead of what it writes; without
/// the scope it could take the room a flush was promised, and a flush that
/// fails holds its journal, so nothing would come back. A compaction refused
/// this way leaves its inputs in place and is tried again.
#[must_use = "the scope lasts until it drops"]
pub struct MaintenanceScope {
    was: bool,
}

impl MaintenanceScope {
    /// Enter maintenance on this thread until the scope drops.
    pub fn enter() -> Self {
        Self {
            was: MAINTENANCE.replace(true),
        }
    }
}

impl Drop for MaintenanceScope {
    fn drop(&mut self) {
        MAINTENANCE.set(self.was);
    }
}

/// The owned footprint of one endpoint with a hard limit.
pub struct Footprint {
    id: String,
    root: PathBuf,
    /// The hard limit; `0` for none, which charges without refusing.
    limit: AtomicU64,
    /// The ceiling the store's own work is held to instead of the limit,
    /// `0` for none. While the store opens: what it held when the open began
    /// and [`OVER_LIMIT_ALLOWANCE`]. While the endpoint holds or was promised
    /// more than its limit (the limit was lowered below it, or the store
    /// opened under a smaller one): what it held and was promised then and
    /// [`Footprint::over_limit_room`], lowered as the promises land and space
    /// is freed, until it meets the limit.
    pending: AtomicU64,
    /// Open phases in progress ([`Footprints::open_phase`]).
    opening: AtomicU32,
    /// The part of the limit client writes leave to recovery work.
    reserve: AtomicU64,
    /// The file system's allocation unit under `root`.
    unit: u64,
    /// The bytes the endpoint's files hold, as charged.
    charged: AtomicU64,
    /// What admitted client writes hold until they land.
    admitted: AtomicU64,
    /// See [`FlushDebt`]; none until the engine's trees are open.
    debt: parking_lot::RwLock<Option<FlushDebt>>,
    /// The charge of every file under `root`, by every name it has.
    files: parking_lot::Mutex<HashMap<PathBuf, Arc<FileCharge>>>,
    /// Handed to file charges, which give their bytes back when they go.
    this: Weak<Footprint>,
}

impl std::fmt::Debug for Footprint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Footprint")
            .field("id", &self.id)
            .field("root", &self.root)
            .field("limit", &self.limit())
            .field("charged", &self.charged())
            .field("admitted", &self.admitted())
            .finish_non_exhaustive()
    }
}

impl Footprint {
    /// The footprint of `endpoint`, charged with the files already under its
    /// root. A store already past the limit opens: nothing is deleted to fit
    /// it, and growth is refused until it fits again.
    ///
    /// # Errors
    ///
    /// The root cannot be listed.
    pub fn open(endpoint: &EndpointConfig) -> io::Result<Arc<Self>> {
        let limit = endpoint.hard_limit_bytes;
        let unit = allocation_unit(&endpoint.path);
        let footprint = Arc::new_cyclic(|this| Self {
            id: endpoint.id.clone(),
            root: endpoint.path.clone(),
            limit: AtomicU64::new(limit),
            pending: AtomicU64::new(0),
            opening: AtomicU32::new(0),
            reserve: AtomicU64::new(endpoint.recovery_reserve()),
            unit,
            charged: AtomicU64::new(0),
            admitted: AtomicU64::new(0),
            debt: parking_lot::RwLock::new(None),
            files: parking_lot::Mutex::new(HashMap::new()),
            this: this.clone(),
        });
        footprint.charge_existing()?;
        footprint.pend_if_over();
        Ok(footprint)
    }

    /// The limit the endpoint's own growth is held to: the hard limit, or
    /// while the endpoint holds more than it, the pending ceiling. `0` for
    /// none.
    pub fn enforced_limit(&self) -> u64 {
        match self.limit() {
            0 => 0,
            limit => limit.max(self.pending.load(Ordering::Acquire)),
        }
    }

    /// The ceiling the store's own work is held to instead of the limit
    /// while the store opens or holds more than its limit; `None` otherwise.
    pub fn pending_limit(&self) -> Option<u64> {
        match self.pending.load(Ordering::Acquire) {
            0 => None,
            pending => Some(pending),
        }
    }

    /// Begin an open phase: the store's own work may grow up to
    /// [`OVER_LIMIT_ALLOWANCE`] past what the endpoint holds now, whatever
    /// the limit, so a store that does not fit its limit still opens.
    fn begin_open(&self) {
        self.opening.fetch_add(1, Ordering::AcqRel);
        if self.limit() == 0 {
            return;
        }
        // Bounded by what the endpoint can hold.
        let ceiling = self.charged() + OVER_LIMIT_ALLOWANCE;
        self.pending.fetch_max(ceiling, Ordering::AcqRel);
    }

    /// How far past what it holds the store's own work may grow while the
    /// endpoint holds more than its limit: the reserve, or
    /// [`OVER_LIMIT_ALLOWANCE`] when the reserve of a small limit is less.
    pub fn over_limit_room(&self) -> u64 {
        self.reserve().max(OVER_LIMIT_ALLOWANCE)
    }

    /// End an open phase; once none is left, the endpoint is held to its
    /// limit, or pending when it holds more.
    fn end_open(&self) {
        if self.opening.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.pend_if_over();
        }
    }

    /// Hold the store's own growth to what the endpoint holds and was
    /// promised now and [`Self::over_limit_room`], when that is more than the
    /// limit: the limit is then pending. Nothing is removed to meet it; client writes are
    /// refused until the footprint fits, and the ceiling follows the space
    /// freed down to it. Within an open phase the open ceiling stands until
    /// the phase ends.
    fn pend_if_over(&self) {
        if self.opening.load(Ordering::Acquire) > 0 {
            return;
        }
        let limit = self.limit();
        // What the flushes owe and admitted writes hold is promised already:
        // the ceiling leaves it room to land, so a flush still gives back
        // its journal.
        let committed = self.committed();
        if limit == 0 || committed <= limit {
            self.pending.store(0, Ordering::Release);
            return;
        }
        // Bounded by what the endpoint can hold.
        let ceiling = committed + self.over_limit_room();
        self.pending.store(ceiling, Ordering::Release);
        tracing::warn!(
            endpoint = %self.id,
            charged_bytes = self.charged(),
            committed_bytes = committed,
            hard_limit_bytes = limit,
            pending_ceiling_bytes = ceiling,
            "the endpoint holds more than its hard limit: client writes are refused \
             until reclamation brings it under the limit; nothing is deleted to meet it"
        );
    }

    /// Lower the pending ceiling to what the endpoint holds now and
    /// [`Self::over_limit_room`], ending it once the endpoint fits its limit.
    fn follow_pending(&self) {
        let pending = self.pending.load(Ordering::Acquire);
        if pending == 0 || self.opening.load(Ordering::Acquire) > 0 {
            return;
        }
        let limit = self.limit();
        let committed = self.committed();
        if committed <= limit {
            if self
                .pending
                .compare_exchange(pending, 0, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                tracing::info!(endpoint = %self.id, hard_limit_bytes = limit, "the endpoint fits its hard limit again");
            }
            return;
        }
        // Bounded by what the endpoint can hold.
        self.pending
            .fetch_min(committed + self.over_limit_room(), Ordering::AcqRel);
    }

    /// The endpoint's id.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The endpoint's root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The hard limit; `0` for none.
    pub fn limit(&self) -> u64 {
        self.limit.load(Ordering::Acquire)
    }

    /// The recovery reserve inside the limit.
    pub fn reserve(&self) -> u64 {
        self.reserve.load(Ordering::Acquire)
    }

    /// The part of the limit client writes may take: the limit less the
    /// recovery reserve. A reserve read past a limit lowered at the same
    /// moment leaves client writes no room until the next read: none, not
    /// an error.
    pub fn client_ceiling(&self) -> u64 {
        // Floored at zero on purpose: no room is the answer, see above.
        self.limit().saturating_sub(self.reserve())
    }

    /// The bytes the endpoint's files hold, as charged.
    pub fn charged(&self) -> u64 {
        self.charged.load(Ordering::Acquire)
    }

    /// What admitted client writes hold until they land.
    pub fn admitted(&self) -> u64 {
        self.admitted.load(Ordering::Acquire)
    }

    /// What the memtables will write when they are flushed.
    pub fn debt(&self) -> u64 {
        self.debt.read().as_ref().map_or(0, |debt| debt())
    }

    /// Everything the endpoint holds or has promised: charged, admitted and
    /// owed to flushes.
    pub fn committed(&self) -> u64 {
        self.charged() + self.admitted() + self.debt()
    }

    /// Change the limit and the reserve without a restart. A limit below what
    /// the endpoint already holds deletes nothing: growth is refused until
    /// reclamation brings the footprint under it.
    pub fn set_limit(&self, limit: u64, reserve: u64) {
        self.limit.store(limit, Ordering::Release);
        self.reserve.store(reserve.min(limit), Ordering::Release);
        self.pend_if_over();
    }

    /// Install what the memtables owe, once the trees are open.
    pub fn set_debt(&self, debt: FlushDebt) {
        *self.debt.write() = Some(debt);
    }

    /// The room growth that keeps the flush debt and admitted writes covered
    /// may still take: what compactions and other maintenance see as free.
    /// `u64::MAX` without a limit.
    pub fn room(&self) -> u64 {
        match self.enforced_limit() {
            0 => u64::MAX,
            // A footprint past its limit has no room: none, not an error.
            limit => limit.saturating_sub(self.committed()),
        }
    }

    /// The space a file of `len` bytes takes: whole allocation units.
    pub fn allocation(&self, len: u64) -> u64 {
        len.div_ceil(self.unit) * self.unit
    }

    /// Admit a write that adds about `bytes` to the footprint, or refuse it
    /// while it does not fit beside what is charged, admitted and owed. The
    /// admission holds its bytes until it is dropped, by when what it wrote
    /// is charged to its files and owed by its memtable.
    ///
    /// # Errors
    ///
    /// [`StorageError::CapacityExhausted`] when the write does not fit.
    pub fn admit(&self, bytes: u64, class: Admit) -> Result<Admission, StorageError> {
        let limit = self.limit();
        if limit == 0 {
            return Ok(self.admission(0));
        }
        let ceiling = match class {
            Admit::Client => self.client_ceiling(),
            Admit::Recovery => self.enforced_limit(),
        };
        let debt = self.debt();
        let mut admitted = self.admitted.load(Ordering::Acquire);
        loop {
            // Each term is bounded by the bytes the endpoint can hold.
            let need = self.charged() + admitted + debt + bytes;
            if need > ceiling {
                metrics::counter!("coordinode_footprint_refusals_total").increment(1);
                return Err(StorageError::CapacityExhausted {
                    endpoint_id: self.id.clone(),
                    used_bytes: need - bytes,
                    hard_limit_bytes: limit,
                });
            }
            match self.admitted.compare_exchange_weak(
                admitted,
                admitted + bytes,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(self.admission(bytes)),
                Err(now) => admitted = now,
            }
        }
    }

    /// Whether a write of `class` would find any room now: what is charged,
    /// admitted and owed is below the ceiling the class may use. Always
    /// without a limit.
    pub fn has_room(&self, class: Admit) -> bool {
        if self.limit() == 0 {
            return true;
        }
        let ceiling = match class {
            Admit::Client => self.client_ceiling(),
            Admit::Recovery => self.enforced_limit(),
        };
        self.committed() < ceiling
    }

    /// The refusal of a write that found no room.
    pub fn refusal(&self) -> StorageError {
        StorageError::CapacityExhausted {
            endpoint_id: self.id.clone(),
            used_bytes: self.committed(),
            hard_limit_bytes: self.limit(),
        }
    }

    fn admission(&self, bytes: u64) -> Admission {
        Admission {
            footprint: self.this.clone(),
            bytes,
        }
    }

    /// Charge `bytes` of growth, or refuse it when the charged total would
    /// pass the limit. Checked and charged in one step, so concurrent growth
    /// cannot pass the limit together. Growth inside a [`MaintenanceScope`]
    /// is held to [`Self::room`] instead: what admitted writes and the
    /// memtables' flushes were promised stays theirs.
    fn take(&self, bytes: u64) -> io::Result<()> {
        if self.pending.load(Ordering::Acquire) != 0 {
            // Flushes lower what is promised without giving bytes back.
            self.follow_pending();
        }
        let limit = self.enforced_limit();
        if limit == 0 {
            self.charged.fetch_add(bytes, Ordering::AcqRel);
            return Ok(());
        }
        let promised = if MAINTENANCE.get() {
            // Bounded by the bytes the endpoint can hold.
            self.admitted() + self.debt()
        } else {
            0
        };
        let mut charged = self.charged.load(Ordering::Acquire);
        loop {
            // Bounded by the bytes the endpoint can hold.
            let next = charged + bytes;
            if next + promised > limit {
                metrics::counter!("coordinode_footprint_growth_refusals_total").increment(1);
                return Err(io::Error::new(
                    io::ErrorKind::StorageFull,
                    format!(
                        "endpoint {:?} footprint limit reached: {charged} charged, \
                         {promised} promised, {bytes} more asked, limit {limit}",
                        self.id
                    ),
                ));
            }
            match self.charged.compare_exchange_weak(
                charged,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(()),
                Err(now) => charged = now,
            }
        }
    }

    /// Record growth that already happened (the file system allocated past
    /// what was asked), unchecked.
    fn record(&self, bytes: u64) {
        self.charged.fetch_add(bytes, Ordering::AcqRel);
    }

    /// Give back `bytes` a file no longer takes.
    fn give(&self, bytes: u64) {
        let before = self.charged.fetch_sub(bytes, Ordering::AcqRel);
        debug_assert!(
            before >= bytes,
            "a footprint gives back only what it charged"
        );
        self.follow_pending();
    }

    /// The charge of the file at `path`, made at none when the file is new.
    pub fn file(&self, path: &Path) -> Arc<FileCharge> {
        let mut files = self.files.lock();
        Arc::clone(
            files
                .entry(path.to_path_buf())
                .or_insert_with(|| self.new_charge(0)),
        )
    }

    fn new_charge(&self, bytes: u64) -> Arc<FileCharge> {
        Arc::new(FileCharge {
            footprint: self.this.clone(),
            bytes: AtomicU64::new(bytes),
        })
    }

    /// The name `path` is gone. Its bytes are given back once no other name
    /// and no open handle reaches the file.
    pub fn removed(&self, path: &Path) {
        let gone = self.files.lock().remove(path);
        drop(gone);
    }

    /// Every name under `dir` is gone.
    pub fn removed_tree(&self, dir: &Path) {
        let gone: Vec<Arc<FileCharge>> = {
            let mut files = self.files.lock();
            let names: Vec<PathBuf> = files
                .keys()
                .filter(|name| name.starts_with(dir))
                .cloned()
                .collect();
            names
                .into_iter()
                .filter_map(|name| files.remove(&name))
                .collect()
        };
        drop(gone);
    }

    /// The file `from` is now named `to`, replacing whatever `to` named.
    pub fn renamed(&self, from: &Path, to: &Path) {
        let replaced = {
            let mut files = self.files.lock();
            match files.remove(from) {
                Some(charge) => files.insert(to.to_path_buf(), charge),
                None => files.remove(to),
            }
        };
        drop(replaced);
    }

    /// `dst` is a second name of the file `src`: the same bytes, charged once.
    pub fn linked(&self, src: &Path, dst: &Path) {
        let replaced = {
            let mut files = self.files.lock();
            let charge = files
                .entry(src.to_path_buf())
                .or_insert_with(|| self.new_charge(0))
                .clone();
            files.insert(dst.to_path_buf(), charge)
        };
        drop(replaced);
    }

    /// Charge every file already under the root with the space the file
    /// system allocated to it, each file once however many names it has.
    fn charge_existing(&self) -> io::Result<()> {
        let mut files = self.files.lock();
        #[cfg(unix)]
        let mut inodes: HashMap<(u64, u64), Arc<FileCharge>> = HashMap::new();
        let mut pending = vec![self.root.clone()];
        while let Some(dir) = pending.pop() {
            let entries = match std::fs::read_dir(&dir) {
                Ok(entries) => entries,
                // A root not created yet holds nothing.
                Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
                Err(e) => return Err(e),
            };
            for entry in entries {
                let entry = entry?;
                let meta = match std::fs::symlink_metadata(entry.path()) {
                    Ok(meta) => meta,
                    // Removed between the listing and the look.
                    Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
                    Err(e) => return Err(e),
                };
                if meta.is_dir() {
                    pending.push(entry.path());
                    continue;
                }
                if !meta.is_file() {
                    continue;
                }
                let bytes = allocated(&meta, self.unit);
                #[cfg(unix)]
                let charge = {
                    use std::os::unix::fs::MetadataExt;
                    match inodes.entry((meta.dev(), meta.ino())) {
                        std::collections::hash_map::Entry::Occupied(seen) => Arc::clone(seen.get()),
                        std::collections::hash_map::Entry::Vacant(slot) => {
                            self.record(bytes);
                            Arc::clone(slot.insert(self.new_charge(bytes)))
                        }
                    }
                };
                #[cfg(not(unix))]
                let charge = {
                    self.record(bytes);
                    self.new_charge(bytes)
                };
                files.insert(entry.path(), charge);
            }
        }
        Ok(())
    }
}

/// Bytes the file system allocated to the file `meta` describes, for a file
/// written outside the engine's file system and measured after.
#[must_use]
pub fn allocated_bytes(meta: &std::fs::Metadata) -> u64 {
    allocated(meta, DEFAULT_UNIT)
}

/// Bytes the file system allocated to the file `meta` describes.
#[cfg(unix)]
fn allocated(meta: &std::fs::Metadata, _unit: u64) -> u64 {
    use std::os::unix::fs::MetadataExt;
    // st_blocks counts 512-byte units on every unix (POSIX <sys/stat.h>).
    meta.blocks() * 512
}

/// Bytes the file system allocated to the file `meta` describes: its length
/// in whole allocation units, where no block count is reported.
#[cfg(not(unix))]
fn allocated(meta: &std::fs::Metadata, unit: u64) -> u64 {
    meta.len().div_ceil(unit) * unit
}

/// The allocation unit of the file system holding `path`, or of the nearest
/// existing ancestor.
fn allocation_unit(path: &Path) -> u64 {
    let mut at = Some(path);
    while let Some(dir) = at {
        if let Ok(unit) = fs4::allocation_granularity(dir)
            && unit > 0
        {
            return unit;
        }
        at = dir.parent();
    }
    DEFAULT_UNIT
}

/// The least a file being appended to is charged past its length.
const MIN_AHEAD: u64 = 64 * 1024;

/// The most a file being appended to is charged past its length.
const MAX_AHEAD: u64 = 4 * 1024 * 1024;

/// Space a file system may allocate ahead of a writer appending to a file of
/// `len` bytes, beyond its length in whole units: as much as the file holds,
/// at least 64 KiB and at most 4 MiB. APFS was measured allocating a quarter
/// to a third past a journal segment's length, and rounding a 1.45 MB
/// compaction output up to 2 MiB, both kept after the file was closed; XFS
/// preallocates speculatively past the end of a growing file in proportion
/// to its size. The charge comes down to what the file system allocated when
/// the file is synced or closed.
#[must_use]
pub fn allocation_ahead(len: u64) -> u64 {
    len.clamp(MIN_AHEAD, MAX_AHEAD)
}

/// What the file system allocated to the file at `path` now, or `None` when
/// it cannot tell (a file system in memory, a file already gone).
#[must_use]
pub fn allocated_at(path: &Path) -> Option<u64> {
    std::fs::metadata(path)
        .ok()
        .filter(std::fs::Metadata::is_file)
        .map(|meta| allocated_bytes(&meta))
}

/// What one file is charged: the space allocated to it, shared by its names
/// and its open handles, given back when the last of them goes.
pub struct FileCharge {
    footprint: Weak<Footprint>,
    bytes: AtomicU64,
}

impl std::fmt::Debug for FileCharge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileCharge")
            .field("bytes", &self.bytes())
            .finish_non_exhaustive()
    }
}

impl FileCharge {
    /// The bytes charged to the file.
    pub fn bytes(&self) -> u64 {
        self.bytes.load(Ordering::Acquire)
    }

    /// Charge the file up to `len` bytes of content before they are written,
    /// or refuse when the footprint has no room for them. A file already
    /// charged that far is not charged again.
    ///
    /// # Errors
    ///
    /// [`io::ErrorKind::StorageFull`] when the limit would be passed.
    pub fn grow_to(&self, len: u64) -> io::Result<()> {
        let Some(footprint) = self.footprint.upgrade() else {
            return Ok(());
        };
        self.reserve(&footprint, footprint.allocation(len))
    }

    /// Charge a file being appended to up to `end` bytes of content and the
    /// space a file system may allocate past them ([`allocation_ahead`]),
    /// before the bytes are written. What the file system actually allocated
    /// settles the charge at the next [`Self::settle`].
    ///
    /// # Errors
    ///
    /// [`io::ErrorKind::StorageFull`] when the limit would be passed.
    pub fn grow_appending(&self, end: u64) -> io::Result<()> {
        let Some(footprint) = self.footprint.upgrade() else {
            return Ok(());
        };
        // Both bounded by what the endpoint can hold.
        self.reserve(
            &footprint,
            footprint.allocation(end) + allocation_ahead(end),
        )
    }

    /// [`Self::grow_appending`], refused as the endpoint's capacity being
    /// exhausted.
    ///
    /// # Errors
    ///
    /// [`StorageError::CapacityExhausted`] when the limit would be passed.
    pub fn charge_appending(&self, end: u64) -> Result<(), StorageError> {
        self.grow_appending(end).map_err(|_| self.exhausted())
    }

    /// Charge the file `want` bytes in all, checked against the limit.
    fn reserve(&self, footprint: &Footprint, want: u64) -> io::Result<()> {
        let mut have = self.bytes();
        while want > have {
            footprint.take(want - have)?;
            match self
                .bytes
                .compare_exchange(have, want, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => return Ok(()),
                Err(now) => {
                    // Another handle charged the file meanwhile.
                    footprint.give(want - have);
                    have = now;
                }
            }
        }
        Ok(())
    }

    /// [`Self::grow_to`], refused as the endpoint's capacity being exhausted.
    ///
    /// # Errors
    ///
    /// [`StorageError::CapacityExhausted`] when the limit would be passed.
    pub fn charge_to(&self, len: u64) -> Result<(), StorageError> {
        self.grow_to(len).map_err(|_| self.exhausted())
    }

    /// The refusal of growth on this file's endpoint.
    fn exhausted(&self) -> StorageError {
        match self.footprint.upgrade() {
            Some(footprint) => StorageError::CapacityExhausted {
                endpoint_id: footprint.id.clone(),
                used_bytes: footprint.charged(),
                hard_limit_bytes: footprint.limit(),
            },
            None => StorageError::Io("the footprint of a closed store".into()),
        }
    }

    /// Raise the charge to `allocated` when the file system reports the file
    /// takes more than it was charged: an allocation ahead of the writer past
    /// the allowance. Recorded, since it already happened; growth after it is
    /// checked against the footprint that counts it.
    pub fn observe(&self, allocated: u64) {
        let Some(footprint) = self.footprint.upgrade() else {
            return;
        };
        let mut have = self.bytes();
        while allocated > have {
            match self
                .bytes
                .compare_exchange(have, allocated, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => {
                    footprint.record(allocated - have);
                    return;
                }
                Err(now) => have = now,
            }
        }
    }

    /// Set the charge to `allocated`, what the file system reports the file
    /// takes now: after a truncation, a hole punched, or once a writer is done
    /// and what it was charged ahead is no longer needed. Growth recorded this
    /// way already happened.
    pub fn settle(&self, allocated: u64) {
        let Some(footprint) = self.footprint.upgrade() else {
            return;
        };
        let had = self.bytes.swap(allocated, Ordering::AcqRel);
        if allocated > had {
            footprint.record(allocated - had);
        } else {
            footprint.give(had - allocated);
        }
    }

    /// Set the charge to `len` bytes of content, the file's new length after
    /// a truncation.
    pub fn truncate_to(&self, len: u64) {
        let Some(footprint) = self.footprint.upgrade() else {
            return;
        };
        let want = footprint.allocation(len);
        if want < self.bytes() {
            self.settle(want);
        }
    }
}

impl Drop for FileCharge {
    fn drop(&mut self) {
        if let Some(footprint) = self.footprint.upgrade() {
            footprint.give(*self.bytes.get_mut());
        }
    }
}

/// An admitted write's hold on the footprint, given back when it drops.
#[must_use = "an admission holds its bytes until it drops"]
pub struct Admission {
    footprint: Weak<Footprint>,
    bytes: u64,
}

impl std::fmt::Debug for Admission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Admission")
            .field("bytes", &self.bytes)
            .finish_non_exhaustive()
    }
}

impl Drop for Admission {
    fn drop(&mut self) {
        if self.bytes == 0 {
            return;
        }
        if let Some(footprint) = self.footprint.upgrade() {
            footprint.admitted.fetch_sub(self.bytes, Ordering::AcqRel);
        }
    }
}

/// Bytes a mutation whose effects are decided at apply (a metadata command,
/// a derived index change, an operation record) is taken to write.
const DECIDED_MUTATION_BYTES: u64 = 1024;

/// Bytes each mutation adds past its key and value: journal framing and the
/// sequence number, type and node overhead it takes in a memtable.
const MUTATION_OVERHEAD: u64 = 64;

/// The room a client write is admitted for, bounded from above: its journal
/// entry, and its data's share of the flush that writes its memtable out.
#[must_use]
pub fn write_footprint(mutations: &[coordinode_core::txn::proposal::Mutation]) -> u64 {
    use coordinode_core::txn::proposal::Mutation;
    // Bounded by the bytes the mutations hold in memory: no overflow.
    let data: u64 = mutations
        .iter()
        .map(|mutation| match mutation {
            Mutation::Put { key, value, .. } => (key.len() + value.len()) as u64,
            Mutation::Delete { key, .. } => key.len() as u64,
            Mutation::Merge { key, operand, .. } => (key.len() + operand.len()) as u64,
            Mutation::RemoveRange { start, end, .. } => (start.len() + end.len()) as u64,
            Mutation::Command(_) | Mutation::Derive(_) | Mutation::Operation(_) => {
                DECIDED_MUTATION_BYTES
            }
        })
        .sum::<u64>();
    footprint_of(data, mutations.len())
}

/// The room `count` writes of `data` key and value bytes in all are admitted
/// for, bounded from above: each into the journal, then into a table an
/// eighth larger than what it holds.
#[must_use]
pub fn footprint_of(data: u64, count: usize) -> u64 {
    // Bounded by the bytes the writes hold in memory: no overflow.
    let data = data + count as u64 * MUTATION_OVERHEAD;
    data + data + data / 8
}

/// A write's admissions on every capped endpoint, given back when it drops:
/// held across the journal append and the apply, by when what the write
/// added is charged to the journal and owed by the memtables.
#[derive(Debug, Default)]
#[must_use = "an admission holds its bytes until it drops"]
pub struct Admitted {
    /// Held for their drop, which gives the bytes back.
    held: Vec<Admission>,
}

impl Admitted {
    /// Whether the admission holds nothing: no endpoint has a limit.
    fn is_empty(&self) -> bool {
        self.held.iter().all(|admission| admission.bytes == 0)
    }
}

/// A store opening under its footprints; see [`Footprints::open_phase`].
#[derive(Debug)]
#[must_use = "the open phase ends when it drops"]
pub struct OpenPhase {
    footprints: Footprints,
}

impl Drop for OpenPhase {
    fn drop(&mut self) {
        for footprint in self.footprints.endpoints.iter() {
            footprint.end_open();
        }
    }
}

/// The footprints of every endpoint with a hard limit, found by path.
#[derive(Debug, Default, Clone)]
pub struct Footprints {
    /// Deepest root first, so a path finds the endpoint that holds it.
    endpoints: Arc<Vec<Arc<Footprint>>>,
    /// Asks the flush monitor to flush every memtable holding data.
    squeeze: Arc<Squeeze>,
    /// Room held for consensus log entries this member took until they are
    /// applied.
    unapplied: Arc<parking_lot::Mutex<Unapplied>>,
}

/// See [`Footprints::hold_until_applied`].
#[derive(Debug, Default)]
struct Unapplied {
    /// The highest index the log was applied through.
    applied: u64,
    /// Each hold by the index of the last entry it covers.
    held: Vec<(u64, Admitted)>,
}

/// A request, raised by a write a hard limit refused, that the memtables be
/// flushed now rather than when their size or age makes them due: what they
/// owe is reserved at a bound above what their tables take, so a flush hands
/// room back. Whoever met the limit (a client write, a member taking its
/// leader's entries) raises it, so room comes back without that caller
/// owning a flush.
#[derive(Debug, Default)]
struct Squeeze {
    requested: AtomicBool,
    wake: parking_lot::RwLock<Option<Arc<coordinode_core::txn::wake::Wake>>>,
}

impl Footprints {
    /// The footprints of `endpoints` that carry a hard limit, each charged
    /// with what its root already holds.
    ///
    /// # Errors
    ///
    /// A root cannot be listed.
    pub fn open(endpoints: &[EndpointConfig]) -> io::Result<Self> {
        let mut capped = Vec::new();
        for endpoint in endpoints {
            if endpoint.hard_limit_bytes > 0 {
                capped.push(Footprint::open(endpoint)?);
            }
        }
        capped.sort_by_key(|footprint| std::cmp::Reverse(footprint.root.components().count()));
        Ok(Self {
            endpoints: Arc::new(capped),
            squeeze: Arc::default(),
            unapplied: Arc::default(),
        })
    }

    /// Keep `room` until the consensus log is applied through `index`: what
    /// entries a member took will be owed by its memtables only once they
    /// are applied, and the room stays theirs until then.
    pub fn hold_until_applied(&self, index: u64, room: Admitted) {
        if room.is_empty() {
            return;
        }
        let mut unapplied = self.unapplied.lock();
        // Under the lock [`Self::applied`] moves it: an apply that ran ahead
        // of the hold has already passed it.
        if index > unapplied.applied {
            unapplied.held.push((index, room));
        }
    }

    /// The consensus log is applied through `index` (or a snapshot installed
    /// at it): the room held for the entries up to it goes back.
    pub fn applied(&self, index: u64) {
        let mut unapplied = self.unapplied.lock();
        unapplied.applied = unapplied.applied.max(index);
        if !unapplied.held.is_empty() {
            unapplied.held.retain(|(last, _)| *last > index);
        }
    }

    /// The flush monitor's wake, which a squeeze notifies.
    pub(crate) fn set_flush_wake(&self, wake: Arc<coordinode_core::txn::wake::Wake>) {
        *self.squeeze.wake.write() = Some(wake);
    }

    /// Ask for every memtable holding data to be flushed, when some endpoint
    /// owes flushes: a write met a hard limit, and the room the flushes owe
    /// beyond their tables comes back once they run.
    pub fn squeeze(&self) {
        if self.endpoints.iter().all(|f| f.debt() == 0) {
            return;
        }
        self.squeeze.requested.store(true, Ordering::Release);
        if let Some(wake) = self.squeeze.wake.read().as_ref() {
            wake.notify();
        }
    }

    /// Whether a squeeze was asked for since the last call, clearing it.
    pub(crate) fn take_squeeze(&self) -> bool {
        self.squeeze.requested.swap(false, Ordering::AcqRel)
    }

    /// Whether any endpoint has a hard limit.
    pub fn is_empty(&self) -> bool {
        self.endpoints.is_empty()
    }

    /// Open the store under its footprints until the returned phase drops:
    /// meanwhile the store's own work on each endpoint may grow up to
    /// [`OVER_LIMIT_ALLOWANCE`] past what the endpoint held when the phase began,
    /// whatever its limit. Client writes stay held to the limit. When the
    /// last phase ends, an endpoint holding more than its limit is pending
    /// ([`Footprint::pending_limit`]).
    pub fn open_phase(&self) -> OpenPhase {
        for footprint in self.endpoints.iter() {
            footprint.begin_open();
        }
        OpenPhase {
            footprints: self.clone(),
        }
    }

    /// The footprint of the endpoint holding `path`, if it has a hard limit.
    pub fn for_path(&self, path: &Path) -> Option<&Arc<Footprint>> {
        self.endpoints
            .iter()
            .find(|footprint| path.starts_with(&footprint.root))
    }

    /// The footprint of the endpoint `id`, if it has a hard limit.
    pub fn get(&self, id: &str) -> Option<&Arc<Footprint>> {
        self.endpoints.iter().find(|footprint| footprint.id == id)
    }

    /// Every footprint.
    pub fn iter(&self) -> impl Iterator<Item = &Arc<Footprint>> {
        self.endpoints.iter()
    }

    /// Install what the memtables owe on every footprint.
    pub fn set_debt(&self, debt: impl Fn() -> u64 + Send + Sync + Clone + 'static) {
        for footprint in self.endpoints.iter() {
            footprint.set_debt(Box::new(debt.clone()));
        }
    }

    /// Admit a write of about `bytes` on every capped endpoint it may grow:
    /// all of them, since its journal entry and its tables may land on any.
    ///
    /// # Errors
    ///
    /// [`StorageError::CapacityExhausted`] from the first endpoint it does not
    /// fit on.
    pub fn admit(&self, bytes: u64, class: Admit) -> Result<Admitted, StorageError> {
        let admitted = self
            .endpoints
            .iter()
            .map(|footprint| footprint.admit(bytes, class))
            .collect::<Result<Vec<_>, _>>()
            .map(|held| Admitted { held });
        if admitted.is_err() {
            self.squeeze();
        }
        admitted
    }

    /// Refuse a write of `class` that some endpoint's hard limit has no room
    /// for, asking for the flushes that give room back.
    ///
    /// # Errors
    ///
    /// [`StorageError::CapacityExhausted`] for the first endpoint without
    /// room.
    pub fn has_room(&self, class: Admit) -> Result<(), StorageError> {
        match self.endpoints.iter().find(|f| !f.has_room(class)) {
            Some(full) => {
                self.squeeze();
                Err(full.refusal())
            }
            None => Ok(()),
        }
    }

    /// The charge of the file at `path`, or `None` on an endpoint without a
    /// hard limit.
    pub fn file(&self, path: &Path) -> Option<Arc<FileCharge>> {
        self.for_path(path).map(|footprint| footprint.file(path))
    }

    /// The charge of a file without a name in the folder `dir` (a scratch
    /// file unlinked at creation), or `None` on an endpoint without a hard
    /// limit. Its bytes are given back when the charge drops, with the file.
    pub fn unnamed(&self, dir: &Path) -> Option<Arc<FileCharge>> {
        self.for_path(dir).map(|footprint| footprint.new_charge(0))
    }

    /// See [`Footprint::removed`].
    pub fn removed(&self, path: &Path) {
        if let Some(footprint) = self.for_path(path) {
            footprint.removed(path);
        }
    }

    /// See [`Footprint::removed_tree`]: on every endpoint, since `dir` may
    /// hold another endpoint's root.
    pub fn removed_tree(&self, dir: &Path) {
        for footprint in self.endpoints.iter() {
            if footprint.root.starts_with(dir) || dir.starts_with(&footprint.root) {
                footprint.removed_tree(dir);
            }
        }
    }

    /// See [`Footprint::renamed`]. A move across endpoints gives the bytes
    /// back on one and charges them on the other.
    ///
    /// # Errors
    ///
    /// The destination endpoint has no room for the file.
    pub fn renamed(&self, from: &Path, to: &Path) -> io::Result<()> {
        match (self.for_path(from), self.for_path(to)) {
            (Some(a), Some(b)) if Arc::ptr_eq(a, b) => {
                a.renamed(from, to);
                Ok(())
            }
            (source, target) => {
                let bytes = source.map_or(0, |a| a.file(from).bytes());
                if let Some(b) = target {
                    b.removed(to);
                    b.file(to).grow_to(bytes)?;
                }
                if let Some(a) = source {
                    a.removed(from);
                }
                Ok(())
            }
        }
    }

    /// See [`Footprint::linked`].
    ///
    /// # Errors
    ///
    /// A link across endpoints is a copy there, refused without room.
    pub fn linked(&self, src: &Path, dst: &Path) -> io::Result<()> {
        match (self.for_path(src), self.for_path(dst)) {
            (Some(a), Some(b)) if Arc::ptr_eq(a, b) => {
                a.linked(src, dst);
                Ok(())
            }
            (source, Some(b)) => {
                let bytes = source.map_or(0, |a| a.file(src).bytes());
                b.file(dst).grow_to(bytes)
            }
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
