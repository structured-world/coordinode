//! Write admission by free disk space.
//!
//! A write that reaches the disk with no room left fails inside an fsync, and
//! a failed fsync of the consensus log stops the node for good. So writes
//! are refused before anything is written, while the filesystems under the
//! durable endpoints still hold a reserve: below `min_free_bytes` free on any
//! of them, every new write is refused with [`StorageError::OutOfSpace`];
//! reads go on. Writes are admitted again once every filesystem has
//! `resume_free_bytes` free, so a write that frees a little does not flip
//! the state back and forth.
//!
//! Free space is read from the filesystem (statvfs), which counts everything
//! on it, the database's own files and anything else. It is read again when
//! the last reading is older than [`RECHECK`], so the hot path costs an
//! atomic load almost always and a syscall at most a few times a second.
//!
//! The guard also answers for the endpoints' hard limits
//! ([`Footprints`]): a write that finds no room under one is refused with
//! [`StorageError::CapacityExhausted`], at the same points and with the same
//! consequences as one below the free-space reserve. That is what keeps a
//! consensus log append from meeting a limit inside its write.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::engine::config::{Durability, StorageConfig};
use crate::engine::footprint::{Admit, Admitted, Footprints};
use crate::error::{StorageError, StorageResult};

/// How old a free-space reading may be before a write reads it again.
pub const RECHECK: Duration = Duration::from_millis(250);

/// Default free space below which writes are refused.
pub const DEFAULT_MIN_FREE_BYTES: u64 = 1 << 30;

/// Default free space above which refused writes are admitted again.
pub const DEFAULT_RESUME_FREE_BYTES: u64 = 2 << 30;

/// Free space as a write sees it, and whether writes are paused.
pub struct SpaceGuard {
    /// Roots of the durable endpoints: the filesystems a write lands on.
    paths: Vec<PathBuf>,
    min_free: AtomicU64,
    resume_free: AtomicU64,
    /// The least free space over `paths` at the last reading.
    available: AtomicU64,
    /// The path that had the least free space at the last reading.
    tightest: parking_lot::Mutex<Option<PathBuf>>,
    paused: AtomicBool,
    /// When the last reading was taken, in nanoseconds since `epoch`; 0
    /// before the first.
    read_at: AtomicU64,
    epoch: Instant,
    /// The endpoints' hard limits, answered for with the free space.
    footprints: Footprints,
}

impl SpaceGuard {
    /// A guard over the durable endpoints of `config`, with its reserve.
    pub fn new(config: &StorageConfig) -> Self {
        let paths = config
            .endpoints
            .iter()
            .filter(|e| e.durability != Durability::Volatile)
            .map(|e| e.path.clone())
            .collect();
        let guard = Self {
            paths,
            min_free: AtomicU64::new(config.min_free_bytes),
            resume_free: AtomicU64::new(config.resume_free_bytes.max(config.min_free_bytes)),
            available: AtomicU64::new(u64::MAX),
            tightest: parking_lot::Mutex::new(None),
            paused: AtomicBool::new(false),
            read_at: AtomicU64::new(0),
            epoch: Instant::now(),
            footprints: Footprints::default(),
        };
        guard.refresh();
        guard
    }

    /// Answer for the hard limits of `footprints` as well.
    #[must_use]
    pub fn with_footprints(mut self, footprints: Footprints) -> Self {
        self.footprints = footprints;
        self
    }

    /// The endpoints' hard limits the guard answers for.
    pub fn footprints(&self) -> &Footprints {
        &self.footprints
    }

    /// Admit a write, or refuse it while free space is below the reserve or
    /// an endpoint's hard limit leaves client writes no room.
    ///
    /// # Errors
    ///
    /// [`StorageError::OutOfSpace`] while writes are paused;
    /// [`StorageError::CapacityExhausted`] for an endpoint without room.
    pub fn admit(&self) -> StorageResult<()> {
        self.admit_free_space()?;
        self.has_room(Admit::Client)
    }

    /// [`Self::admit`] by free space alone, for a write the engine then
    /// admits on the footprint itself (reclaiming before it refuses).
    ///
    /// # Errors
    ///
    /// [`StorageError::OutOfSpace`] while writes are paused.
    pub(crate) fn admit_free_space(&self) -> StorageResult<()> {
        if self.is_stale() {
            self.refresh();
        }
        if self.paused.load(Ordering::Acquire) {
            metrics::counter!("coordinode_storage_out_of_space_refusals_total").increment(1);
            return Err(self.refusal());
        }
        Ok(())
    }

    /// [`Self::admit`], then reserve `bytes` of `class` on every endpoint
    /// with a hard limit, held until the returned admission drops: a write
    /// that cannot be checked by size later (a consensus log append) holds
    /// its room from before it is proposed until it is in the log.
    ///
    /// # Errors
    ///
    /// As [`Self::admit`], and [`StorageError::CapacityExhausted`] when the
    /// write does not fit.
    pub fn reserve(&self, bytes: u64, class: Admit) -> StorageResult<Admitted> {
        let class = class.on_this_thread();
        match class {
            Admit::Client => self.admit()?,
            Admit::Recovery => self.admit_recovery()?,
        }
        self.footprints.admit(bytes, class)
    }

    /// Refuse a write of `class` that no endpoint's hard limit has room for.
    fn has_room(&self, class: Admit) -> StorageResult<()> {
        let room = self.footprints.has_room(class);
        if room.is_err() {
            metrics::counter!("coordinode_footprint_refusals_total").increment(1);
        }
        room
    }

    /// Admit a write of the store's own recovery records (index use guards,
    /// the consensus vote and log bounds), which go on while client writes
    /// are refused: out of the upper half of the reserve only. The lower half
    /// is left to the flushes and compactions that retire those records from
    /// their journal, so recovery never takes the room its own completion
    /// needs.
    ///
    /// # Errors
    ///
    /// [`StorageError::OutOfSpace`] while free space is below half the
    /// reserve; [`StorageError::CapacityExhausted`] for an endpoint whose
    /// hard limit leaves even the store's own work no room.
    pub fn admit_recovery(&self) -> StorageResult<()> {
        self.admit_free_space_recovery()?;
        self.has_room(Admit::Recovery)
    }

    /// [`Self::admit_recovery`] by free space alone, for a write the engine
    /// then admits on the footprint itself.
    ///
    /// # Errors
    ///
    /// [`StorageError::OutOfSpace`] while free space is below half the
    /// reserve.
    pub(crate) fn admit_free_space_recovery(&self) -> StorageResult<()> {
        if self.is_stale() {
            self.refresh();
        }
        if self.available_bytes() < self.min_free_bytes() / 2 {
            metrics::counter!("coordinode_storage_out_of_space_refusals_total").increment(1);
            return Err(self.refusal());
        }
        Ok(())
    }

    /// Whether client writes are paused: free space below the reserve
    /// (reading it again if the last reading is stale), or an endpoint whose
    /// hard limit leaves them no room.
    pub fn is_paused(&self) -> bool {
        if self.is_stale() {
            self.refresh();
        }
        self.paused.load(Ordering::Acquire) || self.has_room(Admit::Client).is_err()
    }

    /// Whether free space alone pauses writes, whatever the hard limits say:
    /// for the store's own work that frees room under a limit (a snapshot
    /// that lets the consensus log be compacted) yet must not run on a disk
    /// below its reserve.
    pub fn is_paused_by_free_space(&self) -> bool {
        if self.is_stale() {
            self.refresh();
        }
        self.paused.load(Ordering::Acquire)
    }

    /// The least free space over the durable endpoints at the last reading.
    pub fn available_bytes(&self) -> u64 {
        self.available.load(Ordering::Acquire)
    }

    /// The reserve: below this, writes are refused.
    pub fn min_free_bytes(&self) -> u64 {
        self.min_free.load(Ordering::Acquire)
    }

    /// Above this, refused writes are admitted again.
    pub fn resume_free_bytes(&self) -> u64 {
        self.resume_free.load(Ordering::Acquire)
    }

    /// Change the reserve without a restart. `resume_free_bytes` below
    /// `min_free_bytes` is taken as `min_free_bytes`. Takes effect at once.
    pub fn set_reserve(&self, min_free_bytes: u64, resume_free_bytes: u64) {
        self.min_free.store(min_free_bytes, Ordering::Release);
        self.resume_free
            .store(resume_free_bytes.max(min_free_bytes), Ordering::Release);
        self.refresh();
    }

    /// Read free space now and update the paused state.
    pub fn refresh(&self) {
        let mut least: Option<(u64, &PathBuf)> = None;
        for path in &self.paths {
            match fs4::available_space(path) {
                Ok(free) if least.is_none_or(|(l, _)| free < l) => least = Some((free, path)),
                Ok(_) => {}
                // A directory not created yet holds nothing to measure; an
                // unreadable one cannot prove room, so it does not either.
                Err(e) => tracing::debug!(path = %path.display(), %e, "free space unreadable"),
            }
        }
        let available = least.map_or(u64::MAX, |(free, _)| free);
        self.available.store(available, Ordering::Release);
        *self.tightest.lock() = least.map(|(_, p)| p.clone());
        let was = self.paused.load(Ordering::Acquire);
        let paused = if was {
            available < self.resume_free.load(Ordering::Acquire)
        } else {
            available < self.min_free.load(Ordering::Acquire)
        };
        if paused != was {
            self.paused.store(paused, Ordering::Release);
            if paused {
                tracing::error!(
                    available_bytes = available,
                    min_free_bytes = self.min_free_bytes(),
                    path = ?self.tightest.lock().as_deref(),
                    "free disk space below the reserve: writes are refused, reads go on"
                );
            } else {
                tracing::warn!(
                    available_bytes = available,
                    "free disk space back above the resume level: writes are admitted"
                );
            }
        }
        metrics::gauge!("coordinode_storage_free_bytes").set(available as f64);
        metrics::gauge!("coordinode_storage_writes_paused").set(if paused { 1.0 } else { 0.0 });
        let now = u64::try_from(self.epoch.elapsed().as_nanos()).unwrap_or(u64::MAX);
        // Never 0, which means "not read yet".
        self.read_at.store(now.max(1), Ordering::Release);
    }

    fn is_stale(&self) -> bool {
        // Loaded before the clock is read, and the clock is monotonic: the
        // reading is never later than `now`.
        let at = self.read_at.load(Ordering::Acquire);
        let now = u64::try_from(self.epoch.elapsed().as_nanos()).unwrap_or(u64::MAX);
        // RECHECK is a fraction of a second, well inside u64 nanoseconds.
        at == 0 || now - at >= RECHECK.as_nanos() as u64
    }

    fn refusal(&self) -> StorageError {
        StorageError::OutOfSpace {
            path: self
                .tightest
                .lock()
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default(),
            available_bytes: self.available_bytes(),
            min_free_bytes: self.min_free_bytes(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
