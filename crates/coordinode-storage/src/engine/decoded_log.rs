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
}

impl Segment {
    fn new(file: u64, cuts: u64) -> Self {
        Self {
            file,
            cuts,
            slots: BTreeMap::new(),
            filling: false,
        }
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

    /// Stop holding its entries: what readers still hold stays shared.
    fn lend(&mut self) {
        for slot in self.slots.values_mut() {
            if let Slot::Held(entry) = slot {
                *slot = Slot::Lent(Arc::downgrade(entry));
            }
        }
        self.slots.retain(|_, slot| slot.get().is_some());
    }

    fn holds_any(&self) -> bool {
        self.slots
            .values()
            .any(|slot| matches!(slot, Slot::Held(_)))
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
    /// Frames decoded, for tests counting the work readers share.
    #[cfg(any(test, feature = "test-faults"))]
    decodes: AtomicUsize,
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
            #[cfg(any(test, feature = "test-faults"))]
            decodes: AtomicUsize::new(0),
        }
    }

    /// Frames decoded so far.
    #[cfg(any(test, feature = "test-faults"))]
    pub fn decodes(&self) -> usize {
        self.decodes.load(Ordering::Acquire)
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
            // A cap: past the address space it bounds nothing.
            let want = (room - out.len()).saturating_add(READ_AHEAD);
            let filled = self.fill(engine, &path, start, next, until, want, reader);
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
    /// `filling` is the one being read and stays.
    fn make_room(&self, needed: usize, filling: &Path) {
        if self.memory.room() >= needed {
            return;
        }
        let filling = path_id(filling);
        let mut state = self.state.lock();
        for segment in state.segments.values_mut() {
            if self.memory.room() >= needed {
                break;
            }
            if segment.file != filling && segment.holds_any() {
                segment.lend();
            }
        }
        state
            .segments
            .retain(|_, segment| segment.filling || !segment.slots.is_empty());
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
        state
            .segments
            .retain(|_, segment| segment.filling || !segment.slots.is_empty());
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
        path: &Path,
        start: u64,
        next: u64,
        until: u64,
        want: usize,
        reader: &mut Reader<'_, '_>,
    ) -> StorageResult<Vec<(Arc<DecodedEntry>, bool)>> {
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
            self.make_room(envelope, path);
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
                        decoded.push((self.entry(index, units, frames.at(), Some(bytes)), true));
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
                    decoded.push((self.entry(index, units, frames.at(), None), false));
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
        held: Option<usize>,
    ) -> Arc<DecodedEntry> {
        #[cfg(any(test, feature = "test-faults"))]
        self.decodes.fetch_add(1, Ordering::AcqRel);
        Arc::new(DecodedEntry {
            index,
            units,
            end,
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
