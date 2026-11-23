//! Decoded journal entries shared by every change-feed reader of one log.
//!
//! Readers of a log, embedded feeds and server streams alike, at its head or
//! catching up, read the same entries. Decoding each entry's units once per
//! reader multiplies the work by the number of readers; here an entry is
//! decoded once and handed to every reader as a shared value, each reader
//! projecting it for its own interest and keeping its own place.
//!
//! A segment is the unit of identity and eviction. Only the frames a read
//! needs are decoded, with a bounded read-ahead; readers of the same frames
//! wait for one fill rather than decode them in parallel. What a segment held
//! before the log was cut back is not taken for what the segment under the
//! same name holds after. Decoded entries are charged to one memory budget
//! from before they are allocated until the last reader lets go of them; when
//! room is needed the oldest segments, in the log's own order, stop being
//! held, and an entry that cannot be held is still shared while some reader
//! holds it.
//!
//! Readers that keep reading one interest follow it ([`ChangeFollower`]):
//! what the interest selects of each unit, the events it takes and how each
//! reaches it, is made once for all of them and held beside the entry's
//! segment, within the same budget and let go with the segment. It lives
//! while some follower of the interest does; each reader delivers from it at
//! its own place and within its own budget.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

use parking_lot::{Condvar, Mutex};

use crate::engine::change_feed::ChangeUnit;
use crate::engine::core::{Deployment, EntryOutcome};
use crate::engine::fixed_cut::EntryPool;
use crate::error::{StorageError, StorageResult};
use crate::oplog::manager::RetainedFloor;
use crate::oplog::segment::{FrameCursor, HEADER_SIZE, Peeked};

/// Entries a fill decodes past what its reader asked for, for the readers
/// that follow it.
const READ_AHEAD: usize = 64;

/// Default target of the decoded entries a store holds for its change feeds:
/// 64 MiB, the server's `cdc_buffer_bytes` default.
pub const DEFAULT_DECODED_BYTES: usize = 64 << 20;

/// The memory decoded entries of every log of a store take: the target held
/// for later readers and what is held now, readers' borrows included.
#[derive(Debug)]
pub(crate) struct DecodedMemory {
    target: AtomicUsize,
    held: AtomicUsize,
}

impl DecodedMemory {
    /// A budget holding at most `target` bytes for later readers.
    pub(crate) fn new(target: usize) -> Arc<Self> {
        Arc::new(Self {
            target: AtomicUsize::new(target),
            held: AtomicUsize::new(0),
        })
    }

    /// Change the target; what is held above it is let go as room is needed.
    pub(crate) fn set_target(&self, bytes: usize) {
        self.target.store(bytes, Ordering::Release);
    }

    /// The target.
    pub(crate) fn target(&self) -> usize {
        self.target.load(Ordering::Acquire)
    }

    /// What decoded entries take now, those only readers still hold included.
    pub(crate) fn held(&self) -> usize {
        self.held.load(Ordering::Acquire)
    }

    /// Room left under the target. Bytes held above it, borrowed by readers
    /// or not yet let go, leave none: they cannot fund another entry.
    fn room(&self) -> usize {
        self.target().saturating_sub(self.held())
    }

    fn take(&self, bytes: usize) {
        self.held.fetch_add(bytes, Ordering::AcqRel);
    }

    fn give(&self, bytes: usize) {
        self.held.fetch_sub(bytes, Ordering::AcqRel);
    }
}

/// Memory an entry takes, given back when the last holder of the entry lets
/// go of it.
#[derive(Debug)]
struct Charge {
    bytes: usize,
    memory: Arc<DecodedMemory>,
}

impl Drop for Charge {
    fn drop(&mut self) {
        self.memory.give(self.bytes);
    }
}

/// One log entry decoded for change feeds: its units, refused ones marked,
/// as every reader projects them.
#[derive(Debug)]
pub struct DecodedEntry {
    index: u64,
    units: Vec<ChangeUnit>,
    /// Where the next frame of its segment starts.
    end: u64,
    /// The segment it was decoded from: its first index, its file
    /// ([`path_id`]) and the log's cut count then.
    segment: (u64, u64, u64),
    /// Its charge to the store's budget; `None` for an entry decoded within
    /// a reader's own budget, which that reader's charge holds.
    _charge: Option<Charge>,
}

/// A read's own budget and the charge its share of the read holds: an entry
/// the store's budget cannot hold is decoded within it.
pub(crate) struct Reader<'r, 'b> {
    pub(crate) budget: &'b coordinode_core::budget::QueryBudget,
    pub(crate) charge: &'r mut coordinode_core::budget::MemoryCharge<'b>,
}

impl Reader<'_, '_> {
    /// What the reader's budget has left.
    fn room(&self) -> u64 {
        debug_assert!(self.budget.memory_used() <= self.budget.memory_limit());
        // A budget refuses every reservation past its limit.
        self.budget.memory_limit() - self.budget.memory_used()
    }

    /// Room in `out` for `more` entries, charged to the reader before the
    /// list grows: the old and the new buffer while the entries move, then
    /// the new one only. `false`, nothing grown, when its budget refuses.
    fn reserve<T>(&mut self, out: &mut Vec<T>, more: usize) -> bool {
        if out.capacity() - out.len() >= more {
            return true;
        }
        let slot = core::mem::size_of::<T>() as u64;
        let old = out.capacity() as u64;
        let new = (out.len() + more) as u64;
        if self.charge.grow(new * slot).is_err() {
            return false;
        }
        out.reserve_exact(more);
        self.charge.shrink(old * slot);
        true
    }
}

/// Entries a read makes room for at a time in its list.
const PAGE_STEP: usize = 64;

/// Bytes a decoded entry takes besides its units: the entry, and the counts
/// of the `Arc` that shares it.
const ENTRY_BYTES: usize = core::mem::size_of::<DecodedEntry>() + 2 * core::mem::size_of::<usize>();

impl DecodedEntry {
    /// The entry's log index.
    pub fn index(&self) -> u64 {
        self.index
    }

    /// The entry's units, in log order.
    pub fn units(&self) -> &[ChangeUnit] {
        &self.units
    }
}

/// A decoded entry as a segment keeps it.
#[derive(Debug)]
enum Slot {
    /// Held for later readers.
    Held(Arc<DecodedEntry>),
    /// Not held: shared while some reader still holds it.
    Lent(Weak<DecodedEntry>),
}

impl Slot {
    fn get(&self) -> Option<Arc<DecodedEntry>> {
        match self {
            Self::Held(entry) => Some(Arc::clone(entry)),
            Self::Lent(entry) => entry.upgrade(),
        }
    }
}

/// Which file a segment's entries were decoded from, by a hash of its path:
/// two directories of one log can hold a segment of the same first index.
/// No path is kept, so a read holds no copy of one.
fn path_id(path: &Path) -> u64 {
    use core::hash::{Hash, Hasher};
    let mut hasher = rustc_hash::FxHasher::default();
    path.hash(&mut hasher);
    hasher.finish()
}

/// What one followed interest selected of a segment's entries.
#[derive(Debug, Default)]
struct Selections {
    /// By entry index.
    entries: BTreeMap<u64, Arc<EntrySelection>>,
    /// A fill of this interest's selections of the segment is running.
    filling: bool,
}

/// What is decoded of one segment.
#[derive(Debug)]
struct Segment {
    /// The file its entries were decoded from ([`path_id`]).
    file: u64,
    /// The log's cut count its entries were decoded under.
    cuts: u64,
    slots: BTreeMap<u64, Slot>,
    /// A fill of this segment is running; readers needing it wait for it.
    filling: bool,
    /// What each followed interest selected of its entries, by the
    /// interest's [`Variant`] id.
    selections: rustc_hash::FxHashMap<u64, Selections>,
}

impl Segment {
    fn new(file: u64, cuts: u64) -> Self {
        Self {
            file,
            cuts,
            slots: BTreeMap::new(),
            filling: false,
            selections: rustc_hash::FxHashMap::default(),
        }
    }

    /// Whether a fill of the segment or of a selection of it runs.
    fn busy(&self) -> bool {
        self.filling || self.selections.values().any(|s| s.filling)
    }

    /// Whether anything is left to look up in it.
    fn kept(&self) -> bool {
        self.busy() || !self.slots.is_empty() || !self.selections.is_empty()
    }

    /// Where the frame of entry `index` starts, when the entry before it is
    /// decoded; the first frame otherwise, the fill passing the frames
    /// before by their headers.
    fn start_of(&self, index: u64) -> u64 {
        index
            .checked_sub(1)
            .and_then(|before| self.slots.get(&before))
            .and_then(Slot::get)
            .map_or(HEADER_SIZE, |entry| entry.end)
    }

    /// Stop holding its entries and the selections of them: what readers
    /// still hold stays shared, and charged until they let go.
    fn lend(&mut self) {
        for slot in self.slots.values_mut() {
            if let Slot::Held(entry) = slot {
                *slot = Slot::Lent(Arc::downgrade(entry));
            }
        }
        self.slots.retain(|_, slot| slot.get().is_some());
        for selections in self.selections.values_mut() {
            selections.entries.clear();
        }
        self.selections.retain(|_, s| s.filling);
    }

    fn holds_any(&self) -> bool {
        self.slots
            .values()
            .any(|slot| matches!(slot, Slot::Held(_)))
            || self.selections.values().any(|s| !s.entries.is_empty())
    }
}

/// What one followed interest selected of one entry's units, made once for
/// every reader of the interest, charged to the store's budget until the
/// last of them lets go of it.
#[derive(Debug)]
pub(crate) struct EntrySelection {
    units: Vec<crate::engine::change_feed::UnitSelection>,
    _charge: Charge,
}

impl EntrySelection {
    /// The selection of each unit of the entry, in the entry's order.
    pub(crate) fn units(&self) -> &[crate::engine::change_feed::UnitSelection] {
        &self.units
    }
}

/// One interest followed on one log: the readers of it share what it
/// selects of each unit. Lives while some [`ChangeFollower`] of it does.
#[derive(Debug)]
struct Variant {
    /// Its key in every segment's selections; never given to another.
    id: u64,
    interest: crate::engine::change_feed::Interest,
    log: Weak<DecodedLog>,
}

impl Drop for Variant {
    /// The last follower left: nothing of what the interest selected stays
    /// to be looked up. A fill finishing later cannot keep it, as a fill
    /// runs for a follower that keeps the variant alive until it is done.
    fn drop(&mut self) {
        if let Some(log) = self.log.upgrade() {
            log.retire(self.id, &self.interest);
        }
    }
}

/// A reader following one interest of one log. While a follower of an
/// interest lives, its reads share each unit's selection with every other
/// follower of the same interest: the units are selected once, and each
/// reader delivers from them at its own place, within its own budget.
/// Dropping the last follower of an interest drops what it selected.
#[derive(Debug, Clone)]
pub struct ChangeFollower {
    log: Arc<DecodedLog>,
    variant: Arc<Variant>,
}

impl ChangeFollower {
    /// The interest it follows.
    pub fn interest(&self) -> &crate::engine::change_feed::Interest {
        &self.variant.interest
    }

    /// Whether it follows `log`.
    pub(crate) fn follows_log(&self, log: &Arc<DecodedLog>) -> bool {
        Arc::ptr_eq(&self.log, log)
    }
}

/// A reader's claim on the fill of a segment, or of one interest's
/// selections of it. The fill ends it with [`Self::done`] before publishing;
/// a fill that unwinds lets it go here, so the readers waiting on it look
/// again rather than wait for good.
struct Claim<'l> {
    log: &'l DecodedLog,
    first: u64,
    cuts: u64,
    /// The variant whose selections are filled; `None` for the entries.
    variant: Option<u64>,
    armed: bool,
}

impl<'l> Claim<'l> {
    fn new(log: &'l DecodedLog, first: u64, cuts: u64, variant: Option<u64>) -> Self {
        Self {
            log,
            first,
            cuts,
            variant,
            armed: true,
        }
    }

    /// The fill ran to its end; its publish lets the claim go.
    fn done(mut self) {
        self.armed = false;
    }
}

impl Drop for Claim<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let mut state = self.log.state.lock();
        if let Some(segment) = state.segments.get_mut(&self.first)
            && segment.cuts == self.cuts
        {
            match self.variant {
                None => segment.filling = false,
                Some(id) => {
                    if let Some(selections) = segment.selections.get_mut(&id) {
                        selections.filling = false;
                    }
                }
            }
        }
        drop(state);
        self.log.filled.notify_all();
    }
}

/// The store's budget as a projection's room: what it takes counts as held
/// at once, the oldest segments let go first when it is short.
struct SharedRoom<'l> {
    log: &'l DecodedLog,
    /// The segment being selected, which stays.
    keep: u64,
    taken: u64,
}

impl crate::engine::change_feed::Room for SharedRoom<'_> {
    fn take(&mut self, bytes: u64) -> bool {
        let Ok(wanted) = usize::try_from(bytes) else {
            return false;
        };
        self.log.make_room(wanted, self.keep);
        if self.log.memory.room() < wanted {
            return false;
        }
        self.log.memory.take(wanted);
        self.taken += bytes;
        true
    }

    fn give(&mut self, bytes: u64) {
        // At most what was taken: the projection gives back its own.
        self.log.memory.give(bytes as usize);
        self.taken -= bytes;
    }
}

#[derive(Debug, Default)]
struct State {
    /// By first index: the log's own order.
    segments: BTreeMap<u64, Segment>,
}

/// The decoded entries of one log, shared by its change-feed readers.
#[derive(Debug)]
pub struct DecodedLog {
    dirs: Vec<PathBuf>,
    /// Where the log starts and how often it was cut back, as its writer
    /// publishes them; `None` for a log whose writer this store does not
    /// know, read with the retention its segments show and no cut count.
    floor: Option<RetainedFloor>,
    memory: Arc<DecodedMemory>,
    state: Mutex<State>,
    filled: Condvar,
    /// The interests some reader follows, each with its live variant.
    variants: Mutex<rustc_hash::FxHashMap<crate::engine::change_feed::Interest, Weak<Variant>>>,
    /// The id the next variant takes.
    next_variant: core::sync::atomic::AtomicU64,
    /// Frames decoded, for tests counting the work readers share.
    #[cfg(any(test, feature = "test-faults"))]
    decodes: AtomicUsize,
    /// Units selected for a followed interest, for the same tests.
    #[cfg(any(test, feature = "test-faults"))]
    selects: AtomicUsize,
    /// The next fill of entries, or of a selection, panics: a reader that
    /// fails while it fills.
    #[cfg(test)]
    panic_next: Mutex<Option<bool>>,
    /// Run once a fill of entries ended and before it publishes: what
    /// happens to the log while a fill is late.
    #[cfg(test)]
    before_publish: LateHook,
}

/// What a test runs while a fill is late.
#[cfg(test)]
#[derive(Default)]
struct LateHook(Mutex<Option<Box<dyn FnOnce() + Send>>>);

#[cfg(test)]
impl core::fmt::Debug for LateHook {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("LateHook")
    }
}

impl DecodedLog {
    pub(crate) fn new(
        dirs: Vec<PathBuf>,
        floor: Option<RetainedFloor>,
        memory: Arc<DecodedMemory>,
    ) -> Self {
        Self {
            dirs,
            floor,
            memory,
            state: Mutex::new(State::default()),
            filled: Condvar::new(),
            variants: Mutex::new(rustc_hash::FxHashMap::default()),
            next_variant: core::sync::atomic::AtomicU64::new(0),
            #[cfg(any(test, feature = "test-faults"))]
            decodes: AtomicUsize::new(0),
            #[cfg(any(test, feature = "test-faults"))]
            selects: AtomicUsize::new(0),
            #[cfg(test)]
            panic_next: Mutex::new(None),
            #[cfg(test)]
            before_publish: LateHook::default(),
        }
    }

    /// Run `late` once the next fill of entries ended, before it publishes.
    #[cfg(test)]
    fn before_next_publish(&self, late: impl FnOnce() + Send + 'static) {
        *self.before_publish.0.lock() = Some(Box::new(late));
    }

    /// Make the next fill panic: of a selection when `selection`, of
    /// entries otherwise.
    #[cfg(test)]
    fn panic_next_fill(&self, selection: bool) {
        *self.panic_next.lock() = Some(selection);
    }

    /// Panic here when the next fill of this kind is to.
    #[cfg(test)]
    #[expect(clippy::panic, reason = "a test makes a reader fail inside a fill")]
    fn maybe_panic(&self, selection: bool) {
        let mut next = self.panic_next.lock();
        if *next == Some(selection) {
            *next = None;
            drop(next);
            panic!("a reader fails while it fills");
        }
    }

    /// Frames decoded so far.
    #[cfg(any(test, feature = "test-faults"))]
    pub fn decodes(&self) -> usize {
        self.decodes.load(Ordering::Acquire)
    }

    /// Units selected for a followed interest so far.
    #[cfg(any(test, feature = "test-faults"))]
    pub fn selects(&self) -> usize {
        self.selects.load(Ordering::Acquire)
    }

    /// How many interests have something selected held for them.
    #[cfg(any(test, feature = "test-faults"))]
    pub fn held_selections(&self) -> usize {
        let state = self.state.lock();
        let mut ids: Vec<u64> = state
            .segments
            .values()
            .flat_map(|segment| {
                segment
                    .selections
                    .iter()
                    .filter(|(_, s)| !s.entries.is_empty())
                    .map(|(id, _)| *id)
            })
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids.len()
    }

    /// A follower of `interest` on this log: the variant other followers of
    /// the same interest share, or a new one.
    pub(crate) fn follow(
        self: &Arc<Self>,
        interest: &crate::engine::change_feed::Interest,
    ) -> ChangeFollower {
        let mut variants = self.variants.lock();
        if let Some(variant) = variants.get(interest).and_then(Weak::upgrade) {
            return ChangeFollower {
                log: Arc::clone(self),
                variant,
            };
        }
        let variant = Arc::new(Variant {
            id: self.next_variant.fetch_add(1, Ordering::AcqRel),
            interest: interest.clone(),
            log: Arc::downgrade(self),
        });
        variants.insert(interest.clone(), Arc::downgrade(&variant));
        ChangeFollower {
            log: Arc::clone(self),
            variant,
        }
    }

    /// The variant `id` of `interest` lost its last follower: forget it and
    /// what it selected. What readers still hold of it stays charged until
    /// they let go.
    fn retire(&self, id: u64, interest: &crate::engine::change_feed::Interest) {
        {
            let mut variants = self.variants.lock();
            // A follower may have made a new variant of the interest since.
            if variants
                .get(interest)
                .is_some_and(|v| v.strong_count() == 0)
            {
                variants.remove(interest);
            }
        }
        let mut state = self.state.lock();
        for segment in state.segments.values_mut() {
            segment.selections.remove(&id);
        }
        state.segments.retain(|_, segment| segment.kept());
    }

    /// The selection `follower`'s interest makes of each of `entries`, made
    /// once for every follower of it: held for later readers while the
    /// store's budget has room, beside the entry's segment, and lent while
    /// some reader holds it otherwise. `None` for an entry whose selection
    /// does not fit the store's budget: its reader selects it within its
    /// own. Readers of the same entry wait for one selection of it.
    ///
    /// # Errors
    ///
    /// A unit that does not project.
    pub(crate) fn select(
        &self,
        follower: &ChangeFollower,
        entries: &[Arc<DecodedEntry>],
        reader: &mut Reader<'_, '_>,
    ) -> StorageResult<Vec<Option<Arc<EntrySelection>>>> {
        let variant = &follower.variant;
        let mut out: Vec<Option<Arc<EntrySelection>>> = Vec::new();
        if !reader.reserve(&mut out, entries.len()) {
            return Err(StorageError::Budget(
                coordinode_core::budget::BudgetStop::Memory {
                    requested: (entries.len() * core::mem::size_of::<Option<Arc<EntrySelection>>>())
                        as u64,
                    used: reader.budget.memory_used(),
                    limit: reader.budget.memory_limit(),
                },
            ));
        }
        let mut n = 0;
        while n < entries.len() {
            let entry = &entries[n];
            let (first, file, cuts) = entry.segment;
            {
                let mut state = self.state.lock();
                // An entry decoded before a cut is of a segment the cut may
                // have replaced: selected for this read only, nothing kept.
                if self.cuts() != cuts {
                    drop(state);
                    out.push(
                        self.select_entry(entry, &variant.interest, first)?
                            .map(Arc::new),
                    );
                    n += 1;
                    continue;
                }
                let segment = state
                    .segments
                    .entry(first)
                    .or_insert_with(|| Segment::new(file, cuts));
                if segment.cuts != cuts || segment.file != file {
                    *segment = Segment::new(file, cuts);
                }
                let selections = segment.selections.entry(variant.id).or_default();
                if let Some(selection) = selections.entries.get(&entry.index) {
                    out.push(Some(Arc::clone(selection)));
                    n += 1;
                    continue;
                }
                if selections.filling {
                    // One selection of the segment at a time: wait for it,
                    // then look again.
                    self.filled.wait(&mut state);
                    continue;
                }
                selections.filling = true;
            }
            let claim = Claim::new(self, first, cuts, Some(variant.id));
            let selected = self.select_entry(entry, &variant.interest, first);
            let selected = match selected {
                Ok(selection) => Ok(selection.map(Arc::new)),
                Err(e) => Err(e),
            };
            claim.done();
            {
                let mut state = self.state.lock();
                let current = self.cuts() == cuts;
                if let Some(segment) = state
                    .segments
                    .get_mut(&first)
                    .filter(|s| s.cuts == cuts && s.file == file)
                    && let Some(selections) = segment.selections.get_mut(&variant.id)
                {
                    selections.filling = false;
                    // Kept for later readers unless the log was cut back
                    // meanwhile; a failed selection keeps nothing and the
                    // next reader selects again.
                    if current && let Ok(Some(selection)) = &selected {
                        selections
                            .entries
                            .insert(entry.index, Arc::clone(selection));
                    }
                }
            }
            self.filled.notify_all();
            out.push(selected?);
            n += 1;
        }
        Ok(out)
    }

    /// What `interest` selects of `entry`'s units, taken from the store's
    /// budget before it is allocated, the oldest segments but `keep` let go
    /// first when the budget is short. `None` when it does not fit.
    ///
    /// # Errors
    ///
    /// A unit that does not project.
    fn select_entry(
        &self,
        entry: &DecodedEntry,
        interest: &crate::engine::change_feed::Interest,
        keep: u64,
    ) -> StorageResult<Option<EntrySelection>> {
        use crate::engine::change_feed::Room as _;
        #[cfg(test)]
        self.maybe_panic(true);
        let mut room = SharedRoom {
            log: self,
            keep,
            taken: 0,
        };
        let own = (core::mem::size_of::<EntrySelection>()
            + 2 * core::mem::size_of::<usize>()
            + entry.units.len() * core::mem::size_of::<crate::engine::change_feed::UnitSelection>())
            as u64;
        if !room.take(own) {
            return Ok(None);
        }
        let mut units = Vec::with_capacity(entry.units.len());
        for unit in &entry.units {
            #[cfg(any(test, feature = "test-faults"))]
            self.selects.fetch_add(1, Ordering::AcqRel);
            match crate::engine::change_feed::select_unit(unit, interest, &mut room) {
                Ok(Some(selection)) => units.push(selection),
                Ok(None) => {
                    drop(units);
                    self.memory.give(room.taken as usize);
                    return Ok(None);
                }
                Err(e) => {
                    drop(units);
                    self.memory.give(room.taken as usize);
                    return Err(e);
                }
            }
        }
        Ok(Some(EntrySelection {
            units,
            _charge: Charge {
                bytes: room.taken as usize,
                memory: Arc::clone(&self.memory),
            },
        }))
    }

    /// The first index of every segment some entry of which is held.
    #[cfg(test)]
    fn held_segments(&self) -> Vec<u64> {
        self.state
            .lock()
            .segments
            .iter()
            .filter(|(_, segment)| segment.holds_any())
            .map(|(first, _)| *first)
            .collect()
    }

    fn cuts(&self) -> u64 {
        self.floor.as_ref().map_or(0, RetainedFloor::cuts)
    }

    /// Up to `max` decoded entries from `from` below `until`, and the
    /// position after the last of them. Entries decoded for an earlier read
    /// are handed over as they are; the others are decoded once for every
    /// reader waiting on them. An entry the store's budget cannot hold is
    /// decoded within what `reader`'s own budget has left and charged to it,
    /// as a read the store shares nothing of would be.
    ///
    /// # Errors
    ///
    /// [`StorageError::RetentionLost`] when the log no longer holds `from`,
    /// held entries notwithstanding; [`StorageError::Budget`] when the first
    /// entry fits neither the store's budget nor the reader's; an entry below
    /// `until` the log holds but cannot read, or a unit the feed cannot read.
    pub(crate) fn read(
        &self,
        engine: &Deployment,
        from: u64,
        until: u64,
        max: usize,
        reader: &mut Reader<'_, '_>,
    ) -> StorageResult<(Vec<Arc<DecodedEntry>>, u64)> {
        let mut out: Vec<Arc<DecodedEntry>> = Vec::new();
        let mut next = from;
        while next < until && out.len() < max {
            // Held bytes cannot make a position readable the log let go of.
            if let Some(first) = self.floor.as_ref().and_then(RetainedFloor::get)
                && next < first
            {
                return Err(StorageError::RetentionLost {
                    requested: next,
                    first_retained: first,
                });
            }
            let (held_from, holding) = crate::oplog::tailer::segment_holding(&self.dirs, next)?;
            let Some((first, path)) = holding else {
                return Err(StorageError::RetentionLost {
                    requested: next,
                    first_retained: held_from.unwrap_or(until),
                });
            };
            // The reader's list holds the entries it takes: room for the next
            // of them first, from its budget.
            let more = (max - out.len()).min(PAGE_STEP);
            if !reader.reserve(&mut out, more) {
                if out.is_empty() {
                    return Err(StorageError::Budget(
                        coordinode_core::budget::BudgetStop::Memory {
                            requested: core::mem::size_of::<Arc<DecodedEntry>>() as u64,
                            used: reader.budget.memory_used(),
                            limit: reader.budget.memory_limit(),
                        },
                    ));
                }
                break;
            }
            let room = out.capacity().min(max);
            let cuts = self.cuts();
            let Some(start) = self.claim(first, &path, cuts, &mut out, &mut next, until, room)
            else {
                continue;
            };
            let claim = Claim::new(self, first, cuts, None);
            // A cap: past the address space it bounds nothing.
            let want = (room - out.len()).saturating_add(READ_AHEAD);
            let filled = self.fill(
                engine,
                (first, &path, cuts),
                start,
                next,
                until,
                want,
                reader,
            );
            #[cfg(test)]
            if let Some(late) = self.before_publish.0.lock().take() {
                late();
            }
            claim.done();
            let Some(decoded) = self.publish(first, cuts, filled)? else {
                // Read across a cut of the log: what it decoded may be of the
                // segment the cut replaced. Read again.
                continue;
            };
            if decoded.is_empty() {
                // The segment that should hold the next entry ends before it:
                // purged meanwhile, or unreadable.
                let (held_from, holding) = crate::oplog::tailer::segment_holding(&self.dirs, next)?;
                return Err(if holding.is_some() {
                    StorageError::Io(format!(
                        "the log holds entry {next}, below its confirmed end {until}, but it \
                         cannot be read"
                    ))
                } else {
                    StorageError::RetentionLost {
                        requested: next,
                        first_retained: held_from.unwrap_or(until),
                    }
                });
            }
            for entry in decoded {
                if out.len() >= room || entry.index != next {
                    break;
                }
                next = entry.index + 1;
                out.push(entry);
            }
        }
        Ok((out, next))
    }

    /// Hand over the entries of segment `first` already decoded from `next`
    /// on; when the read needs more of this segment, take its fill and return
    /// where it starts. `None` when the read is done, moved on (its next entry
    /// may be in a later segment), or another fill ended while waiting: the
    /// caller looks again.
    #[allow(clippy::too_many_arguments)]
    fn claim(
        &self,
        first: u64,
        path: &Path,
        cuts: u64,
        out: &mut Vec<Arc<DecodedEntry>>,
        next: &mut u64,
        until: u64,
        max: usize,
    ) -> Option<u64> {
        let file = path_id(path);
        let mut state = self.state.lock();
        let segment = state
            .segments
            .entry(first)
            .or_insert_with(|| Segment::new(file, cuts));
        // Decoded before a cut, or of another file under the name: not this
        // segment's.
        if segment.cuts != cuts || segment.file != file {
            *segment = Segment::new(file, cuts);
        }
        let from = *next;
        while *next < until && out.len() < max {
            match segment.slots.get(next).and_then(Slot::get) {
                Some(entry) => {
                    *next = entry.index + 1;
                    out.push(entry);
                }
                None => break,
            }
        }
        if *next != from || out.len() >= max {
            return None;
        }
        if segment.filling {
            // One fill at a time per segment: wait for it, then look again.
            self.filled.wait(&mut state);
            return None;
        }
        segment.filling = true;
        Some(segment.start_of(*next))
    }

    /// End segment `first`'s fill: keep what it decoded for later readers
    /// unless the log was cut back meanwhile, wake its waiters, and hand
    /// what it decoded to the reader that ran it; `None` when a cut overlapped
    /// the fill, which is then read again.
    fn publish(
        &self,
        first: u64,
        cuts: u64,
        filled: StorageResult<Vec<(Arc<DecodedEntry>, bool)>>,
    ) -> StorageResult<Option<Vec<Arc<DecodedEntry>>>> {
        let mut state = self.state.lock();
        let current = self.cuts() == cuts;
        if let Some(segment) = state.segments.get_mut(&first)
            && segment.cuts == cuts
        {
            segment.filling = false;
            if current && let Ok(decoded) = &filled {
                for (entry, held) in decoded {
                    let slot = if *held {
                        Slot::Held(Arc::clone(entry))
                    } else {
                        Slot::Lent(Arc::downgrade(entry))
                    };
                    segment.slots.insert(entry.index, slot);
                }
            }
        }
        drop(state);
        self.filled.notify_all();
        let decoded = filled?;
        if !current {
            return Ok(None);
        }
        Ok(Some(decoded.into_iter().map(|(entry, _)| entry).collect()))
    }

    /// Let go of the oldest segments, in the log's order, until `needed`
    /// bytes fit under the target or no other segment is held. The segment
    /// `keep`, by its first index, is the one being read and stays.
    fn make_room(&self, needed: usize, keep: u64) {
        if self.memory.room() >= needed {
            return;
        }
        let mut state = self.state.lock();
        for (first, segment) in &mut state.segments {
            if self.memory.room() >= needed {
                break;
            }
            if *first != keep && segment.holds_any() {
                segment.lend();
            }
        }
        state.segments.retain(|_, segment| segment.kept());
    }

    /// Let go of segments until what is held fits the target: a shrunk target
    /// takes effect for what is held now, not only for later entries.
    pub(crate) fn trim(&self) {
        let mut state = self.state.lock();
        for segment in state.segments.values_mut() {
            if self.memory.held() <= self.memory.target() {
                break;
            }
            segment.lend();
        }
        state.segments.retain(|_, segment| segment.kept());
    }

    /// Decode up to `want` entries of the segment at `path` from `next`, its
    /// frame starting at or after byte `start`, below `until`. Each entry is
    /// held for later readers when the budget has room for it, after the
    /// oldest segments are let go if needed; the first one the budget cannot
    /// hold is decoded within `reader_room` and only lent. Returns each
    /// entry with whether it is held.
    #[allow(clippy::too_many_arguments)]
    fn fill(
        &self,
        engine: &Deployment,
        (first, path, cuts): (u64, &Path, u64),
        start: u64,
        next: u64,
        until: u64,
        want: usize,
        reader: &mut Reader<'_, '_>,
    ) -> StorageResult<Vec<(Arc<DecodedEntry>, bool)>> {
        #[cfg(test)]
        self.maybe_panic(false);
        let mut frames = match FrameCursor::open_at(path, start) {
            Ok(frames) => frames,
            // Purged since it was found: the caller reads the log again.
            Err(_) if !path.exists() => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };
        let mut decoded: Vec<(Arc<DecodedEntry>, bool)> = Vec::new();
        let mut expect = next;
        loop {
            let (len, index, ops) = match frames.peek()? {
                Peeked::End => break,
                Peeked::Frame { len, index, ops } => (len, index, ops),
            };
            if index < expect {
                frames.skip();
                continue;
            }
            if index > expect || index >= until || decoded.len() >= want {
                break;
            }
            // The fill's list of what it decoded, grown from the reader's
            // budget before it grows.
            let more = PAGE_STEP.min(want - decoded.len());
            if decoded.len() == decoded.capacity() && !reader.reserve(&mut decoded, more) {
                break;
            }
            // The entry's payload and its list of operations, before anything
            // of it is read.
            let Some(envelope) = ops
                .checked_mul(core::mem::size_of::<crate::oplog::OplogOpRef<'_>>() as u64)
                .and_then(|bytes| bytes.checked_add(len))
                .and_then(|bytes| usize::try_from(bytes).ok())
            else {
                break;
            };
            self.make_room(envelope, first);
            let segment = (first, path_id(path), cuts);
            let shared = self.memory.room();
            if envelope > shared && !decoded.is_empty() {
                // Past the store's budget: only the first entry of a fill
                // goes through the reader's own.
                break;
            }
            let mut payload = None;
            if envelope <= shared {
                // The payload, its list of operations and the decoding's
                // working room count against the store's budget while the
                // entry is decoded, so concurrent fills see them taken.
                self.memory.take(shared);
                let bytes = match frames.read_payload() {
                    Ok(Some(bytes)) => bytes,
                    // A checksum that fails ends the complete frames.
                    Ok(None) => {
                        self.memory.give(shared);
                        break;
                    }
                    Err(e) => {
                        self.memory.give(shared);
                        return Err(e);
                    }
                };
                // Room for the entry itself is kept out of what its units may take.
                let units_room = shared - envelope - ENTRY_BYTES.min(shared - envelope);
                let held = self.decode(engine, &bytes, index, units_room as u64);
                self.memory.give(shared);
                match held {
                    Ok((units, kept)) => {
                        let bytes = kept + ENTRY_BYTES;
                        self.memory.take(bytes);
                        decoded.push((
                            self.entry(index, units, frames.at(), segment, Some(bytes)),
                            true,
                        ));
                        expect = index + 1;
                        continue;
                    }
                    // Its units do not fit what the store has left: the first
                    // entry of a fill goes through the reader's budget.
                    Err(EntryOutcome::NoRoom) if decoded.is_empty() => payload = Some(bytes),
                    Err(EntryOutcome::NoRoom | EntryOutcome::Unreadable) => break,
                    Err(EntryOutcome::Failed(e)) => return Err(e),
                }
            }
            // Decoded within the reader's own budget, charged to it before
            // anything of the entry is allocated, as a read the store shares
            // nothing of is; lent to other readers while it lives.
            let room = reader.room();
            if envelope as u64 > room {
                return Err(StorageError::Budget(
                    coordinode_core::budget::BudgetStop::Memory {
                        requested: envelope as u64,
                        used: reader.budget.memory_used(),
                        limit: reader.budget.memory_limit(),
                    },
                ));
            }
            reader.charge.grow(room).map_err(StorageError::Budget)?;
            let bytes = match payload {
                Some(bytes) => Ok(Some(bytes)),
                None => frames.read_payload(),
            };
            // Room for the entry itself is kept out of what its units may take.
            let units_room = room - envelope as u64;
            let units_room = units_room - (ENTRY_BYTES as u64).min(units_room);
            let outcome = match bytes {
                Ok(Some(bytes)) => self.decode(engine, &bytes, index, units_room),
                Ok(None) => Err(EntryOutcome::Unreadable),
                Err(e) => Err(EntryOutcome::Failed(e)),
            };
            match outcome {
                Ok((units, kept)) => {
                    let bytes = (kept + ENTRY_BYTES) as u64;
                    // What the entry does not keep returns to the reader.
                    reader.charge.shrink(room - bytes.min(room));
                    decoded.push((self.entry(index, units, frames.at(), segment, None), false));
                }
                Err(failed) => {
                    reader.charge.shrink(room);
                    match failed {
                        EntryOutcome::NoRoom => {
                            return Err(StorageError::Budget(
                                coordinode_core::budget::BudgetStop::Memory {
                                    requested: envelope as u64,
                                    used: reader.budget.memory_used(),
                                    limit: reader.budget.memory_limit(),
                                },
                            ));
                        }
                        EntryOutcome::Unreadable => {}
                        EntryOutcome::Failed(e) => return Err(e),
                    }
                }
            }
            // One entry past the store's budget at a time.
            break;
        }
        Ok(decoded)
    }

    /// A decoded entry, charged to the store's budget for `held` bytes, or
    /// to the reader that decoded it when `None`.
    fn entry(
        &self,
        index: u64,
        units: Vec<ChangeUnit>,
        end: u64,
        segment: (u64, u64, u64),
        held: Option<usize>,
    ) -> Arc<DecodedEntry> {
        #[cfg(any(test, feature = "test-faults"))]
        self.decodes.fetch_add(1, Ordering::AcqRel);
        Arc::new(DecodedEntry {
            index,
            units,
            end,
            segment,
            _charge: held.map(|bytes| Charge {
                bytes,
                memory: Arc::clone(&self.memory),
            }),
        })
    }

    /// Decode the units of entry `index`, its frame's `payload`, within
    /// `room` bytes. Returns them with the bytes they keep.
    fn decode(
        &self,
        engine: &Deployment,
        payload: &[u8],
        index: u64,
        room: u64,
    ) -> Result<(Vec<ChangeUnit>, usize), EntryOutcome> {
        let entry = match crate::oplog::OplogEntryRef::decode(payload) {
            Ok(entry) if entry.index == index => entry,
            _ => return Err(EntryOutcome::Unreadable),
        };
        let mut pool = EntryPool::new(room);
        let mut units = Vec::new();
        engine.change_units_of_entry(&entry, &mut pool, &mut units)?;
        let kept = usize::try_from(pool.kept()).unwrap_or(usize::MAX);
        Ok((units, kept))
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests;
