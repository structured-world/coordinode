//! How far the Raft log applied here is closed, and the commit ceiling of
//! each of its prefixes.
//!
//! A read bounded by a log position must hold every commit of the log up to
//! that position: the prefix's commit ceiling, the running maximum of its
//! commit timestamps. The ceiling rises at some positions, its breakpoints;
//! the ceiling of a position is the one of the last breakpoint at or before
//! it. The ledger keeps every breakpoint exactly, as one record per
//! breakpoint beside the compact head record, so a prefix closed before a
//! later one is never made to need the later one's commits, however many
//! breakpoints pile up while the closed bound stalls. The newest breakpoints
//! are cached; an older position is answered by a predecessor seek.
//!
//! A breakpoint is retired once no view the store still serves can need it:
//! its ceiling is below the closed bound and below the oldest snapshot the
//! store serves reads at. Every read the store can still answer is at a
//! timestamp at or above that snapshot, so the retired breakpoint's ceiling
//! (the anchor) answers every position at or before it for all of them.
//! Retirement runs in bounded batches, one per entry.
//!
//! Every change is planned against the current ledger ([`ClosureLedger::plan`]),
//! written with the apply coverage of the entry it describes, and installed
//! only once those writes landed ([`ClosureLedger::install`]): no reader is
//! answered from a breakpoint the store does not hold.

// no-std: alloc::collections::VecDeque
use std::collections::VecDeque;

use crate::error::{StorageError, StorageResult};

/// The newest breakpoints a ledger keeps in memory.
pub(crate) const CACHED_BREAKPOINTS: usize = 256;

/// The most breakpoints one entry retires.
pub(crate) const RETIRE_BATCH: usize = 64;

/// The head record's key, in the Schema tree's engine-reserved namespace:
/// protocol metadata of the engine, not an application row. Application
/// scans, change enumeration and range deletion never reach it.
pub(crate) const HEAD_KEY: &[u8] = b"\x00clh";

/// The key prefix of the breakpoint records, beside the head record in the
/// reserved namespace; the index follows big-endian, so keys sort by
/// position.
pub(crate) const BREAKPOINT_PREFIX: &[u8] = b"\x00clb";

/// Where an earlier layout kept the head and the breakpoints, among the
/// application's Schema rows. A store holding either is of a layout this
/// build does not read: it is refused, never read as having no ledger.
pub(crate) const REFUSED_HEAD_KEY: &[u8] = b"closure:below";
pub(crate) const REFUSED_BREAKPOINT_PREFIX: &[u8] = b"closure:bp:";

/// The head record's leading byte.
const FORMAT: u8 = 2;

/// The head record's length: the format byte and six fields.
pub(crate) const HEAD_LEN: usize = 1 + 6 * 8;

/// The room the removal of one breakpoint record takes once written: a
/// tombstone for its key. Owed from the record's write until its removal.
pub(crate) const BREAKPOINT_KEY_LEN: usize = BREAKPOINT_PREFIX.len() + 8;

/// A log position where the commit ceiling rose, and the ceiling there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Breakpoint {
    pub(crate) index: u64,
    pub(crate) ceiling: u64,
}

impl Breakpoint {
    /// The record key.
    pub(crate) fn key(self) -> Vec<u8> {
        breakpoint_key(self.index)
    }

    /// The record value.
    pub(crate) fn value(self) -> [u8; 8] {
        self.ceiling.to_be_bytes()
    }

    /// Read a record back.
    ///
    /// # Errors
    ///
    /// The record is not one a breakpoint wrote.
    pub(crate) fn decode(key: &[u8], value: &[u8]) -> StorageResult<Self> {
        let malformed = || StorageError::Serialization("a closure breakpoint record".into());
        let index = key
            .strip_prefix(BREAKPOINT_PREFIX)
            .and_then(|rest| <[u8; 8]>::try_from(rest).ok())
            .ok_or_else(malformed)?;
        let ceiling = <[u8; 8]>::try_from(value).map_err(|_| malformed())?;
        Ok(Self {
            index: u64::from_be_bytes(index),
            ceiling: u64::from_be_bytes(ceiling),
        })
    }
}

/// The record key of the breakpoint at `index`.
pub(crate) fn breakpoint_key(index: u64) -> Vec<u8> {
    let mut key = Vec::with_capacity(BREAKPOINT_PREFIX.len() + 8);
    key.extend_from_slice(BREAKPOINT_PREFIX);
    key.extend_from_slice(&index.to_be_bytes());
    key
}

/// The breakpoint records as the store holds them.
pub(crate) trait Breakpoints {
    /// The breakpoint with the highest index in `from..=through`.
    fn predecessor(&self, from: u64, through: u64) -> StorageResult<Option<Breakpoint>>;
    /// Up to `limit` breakpoints with an index above `after`, in order.
    fn following(&self, after: u64, limit: usize) -> StorageResult<Vec<Breakpoint>>;
}

/// What an entry changes in the ledger: planned before its writes, installed
/// after them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ClosurePlan {
    /// The last entry accounted for once it is in.
    last_index: u64,
    /// The closed bound once it is in.
    below: u64,
    /// Breakpoints its commits add, in order; a raise of the newest one
    /// at the same index replaces it.
    added: Vec<Breakpoint>,
    /// Breakpoints it retires, oldest first.
    retired: Vec<Breakpoint>,
    /// The first breakpoint held after the retired ones, when it retires any.
    next: Option<Breakpoint>,
}

impl ClosurePlan {
    /// The breakpoint records to write.
    pub(crate) fn added(&self) -> &[Breakpoint] {
        &self.added
    }

    /// The breakpoint records to delete.
    pub(crate) fn retired(&self) -> &[Breakpoint] {
        &self.retired
    }

    /// Whether the entry writes no breakpoint record.
    pub(crate) fn writes_no_breakpoint(&self) -> bool {
        self.added.is_empty() && self.retired.is_empty()
    }
}

/// The closed bound of the log applied here and the commit ceilings of its
/// prefixes; see the module documentation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ClosureLedger {
    /// The closed bound: every commit below it is in an applied entry.
    below: u64,
    /// The last entry the ledger accounts for; an entry at or before it is
    /// already in, as when a restarted member applies its log again.
    last_index: u64,
    /// The last retired breakpoint: every position at or before it is
    /// answered with its ceiling.
    anchor_through: u64,
    anchor_ceiling: u64,
    /// The commit ceiling of the whole log applied here.
    applied_ceiling: u64,
    /// How many breakpoint records the store holds: each owes its removal.
    held: u64,
    /// The newest breakpoints after the anchor, a contiguous suffix of them,
    /// in log order with ceilings rising.
    cache: VecDeque<Breakpoint>,
    /// The first breakpoint after the anchor, the next one to retire: while
    /// its ceiling is not below the retirement threshold no entry reads the
    /// records to retire any.
    oldest: Option<Breakpoint>,
}

impl ClosureLedger {
    /// Whether the cache holds every breakpoint after the anchor.
    fn cache_complete(&self) -> bool {
        match (self.oldest, self.cache.front()) {
            (None, _) => true,
            (Some(oldest), Some(front)) => oldest.index >= front.index,
            (Some(_), None) => false,
        }
    }

    /// The closed bound.
    pub(crate) fn below(&self) -> u64 {
        self.below
    }

    /// The commit ceiling of the whole log applied here.
    pub(crate) fn applied_ceiling(&self) -> u64 {
        self.applied_ceiling
    }

    /// How many breakpoint records the store holds.
    pub(crate) fn held(&self) -> u64 {
        self.held
    }

    /// Plan entry `index`: its commits (`(index, commit_ts)` in apply order),
    /// the closed bound `below` it carries (`0` for none), and the
    /// breakpoints it retires, which needs `retained`, the oldest snapshot the
    /// store serves reads at.
    ///
    /// # Errors
    ///
    /// The breakpoint records could not be read.
    pub(crate) fn plan(
        &self,
        commits: &[(u64, u64)],
        below: u64,
        index: u64,
        retained: u64,
        held: &dyn Breakpoints,
    ) -> StorageResult<ClosurePlan> {
        let mut added: Vec<Breakpoint> = Vec::new();
        let mut last_index = self.last_index;
        let mut ceiling = self.applied_ceiling;
        for &(at, commit_ts) in commits {
            // Entry 0 is the group's initial membership: the prefix through
            // it is empty, its ceiling 0 (the anchor's), and a commit there
            // would have no position the ledger could answer it at.
            if at == 0 {
                return Err(StorageError::Io(
                    "a commit at log position 0, which holds the group's initial membership".into(),
                ));
            }
            if at < last_index {
                continue;
            }
            last_index = at;
            if commit_ts <= ceiling {
                continue;
            }
            ceiling = commit_ts;
            let raised = Breakpoint {
                index: at,
                ceiling: commit_ts,
            };
            match added.last_mut() {
                Some(last) if last.index == at => *last = raised,
                _ => added.push(raised),
            }
        }
        // A bound carried by an entry already accounted for, or no higher
        // than the one held, changes nothing.
        let closed = if index < last_index || below <= self.below {
            self.below
        } else {
            last_index = index;
            below
        };
        // Retire the oldest breakpoints no served view needs, in order. The
        // store holds every breakpoint, the cache only the newest.
        let threshold = closed.min(retained);
        let mut retired = Vec::new();
        let mut next = self.oldest;
        if self.oldest.is_some_and(|b| b.ceiling < threshold) {
            let candidates: Vec<Breakpoint> = if self.cache_complete() {
                self.cache.iter().copied().take(RETIRE_BATCH + 1).collect()
            } else {
                held.following(self.anchor_through, RETIRE_BATCH + 1)?
            };
            next = None;
            for b in candidates {
                if retired.len() < RETIRE_BATCH && b.ceiling < threshold {
                    retired.push(b);
                } else {
                    next = Some(b);
                    break;
                }
            }
        }
        Ok(ClosurePlan {
            last_index,
            below: closed,
            added,
            retired,
            next,
        })
    }

    /// Install a plan whose records landed. Whether the closed bound rose.
    pub(crate) fn install(&mut self, plan: &ClosurePlan) -> bool {
        // Each breakpoint is a record of its own: one per entry at most, never
        // one the store already holds (an entry is accounted for once).
        // Bounded by the entries applied: no overflow.
        self.held = self.held + plan.added.len() as u64 - plan.retired.len() as u64;
        // Retired first: they are all older than what the entry adds.
        if let Some(last) = plan.retired.last() {
            self.anchor_through = last.index;
            self.anchor_ceiling = last.ceiling;
            self.oldest = plan.next;
            while self
                .cache
                .front()
                .is_some_and(|b| b.index <= self.anchor_through)
            {
                self.cache.pop_front();
            }
        }
        for &b in &plan.added {
            match self.cache.back_mut() {
                Some(last) if last.index == b.index => *last = b,
                _ => {
                    self.cache.push_back(b);
                    if self.cache.len() > CACHED_BREAKPOINTS {
                        self.cache.pop_front();
                    }
                }
            }
            match &mut self.oldest {
                Some(oldest) if oldest.index == b.index => *oldest = b,
                Some(_) => {}
                None => self.oldest = Some(b),
            }
            self.applied_ceiling = self.applied_ceiling.max(b.ceiling);
        }
        self.last_index = self.last_index.max(plan.last_index);
        if plan.below > self.below {
            self.below = plan.below;
            return true;
        }
        false
    }

    /// The commit timestamp a read must reach to hold every commit of the
    /// log through `index`, a position this member applied. Position 0 holds
    /// the group's initial membership and no commit ([`Self::plan`] refuses
    /// one there), so its prefix is empty and its ceiling 0, however far
    /// retirement moved the anchor past it.
    ///
    /// # Errors
    ///
    /// The breakpoint records could not be read.
    pub(crate) fn ceiling_through(&self, index: u64, held: &dyn Breakpoints) -> StorageResult<u64> {
        if index == 0 {
            return Ok(0);
        }
        if index <= self.anchor_through {
            return Ok(self.anchor_ceiling);
        }
        if let Some(front) = self.cache.front()
            && front.index <= index
        {
            return Ok(self
                .cache
                .iter()
                .rev()
                .find(|b| b.index <= index)
                .map_or(self.anchor_ceiling, |b| b.ceiling));
        }
        if self.cache_complete() {
            return Ok(self.anchor_ceiling);
        }
        Ok(held
            .predecessor(self.anchor_through + 1, index)?
            .map_or(self.anchor_ceiling, |b| b.ceiling))
    }

    /// The head record.
    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEAD_LEN);
        out.push(FORMAT);
        for field in [
            self.below,
            self.last_index,
            self.anchor_through,
            self.anchor_ceiling,
            self.applied_ceiling,
            self.held,
        ] {
            out.extend_from_slice(&field.to_be_bytes());
        }
        out
    }

    /// The head record as `plan` leaves it; the cache is not copied.
    pub(crate) fn encode_after(&self, plan: &ClosurePlan) -> Vec<u8> {
        let mut head = Self {
            below: self.below,
            last_index: self.last_index,
            anchor_through: self.anchor_through,
            anchor_ceiling: self.anchor_ceiling,
            applied_ceiling: self.applied_ceiling,
            held: self.held,
            cache: VecDeque::new(),
            oldest: self.oldest,
        };
        head.install(plan);
        head.encode()
    }

    /// Read the head record back, with the first breakpoint after its anchor
    /// (`held.following`) and the newest ones cached from `newest`, in
    /// descending order: never the whole window.
    ///
    /// # Errors
    ///
    /// The record is not one [`Self::encode`] wrote, or the breakpoint
    /// records could not be read.
    pub(crate) fn decode(
        bytes: &[u8],
        newest: impl Iterator<Item = StorageResult<Breakpoint>>,
        held: &dyn Breakpoints,
    ) -> StorageResult<Self> {
        let field = head_fields(bytes)?;
        let mut ledger = Self {
            below: field(0),
            last_index: field(1),
            anchor_through: field(2),
            anchor_ceiling: field(3),
            applied_ceiling: field(4),
            held: field(5),
            cache: VecDeque::new(),
            oldest: None,
        };
        let anchor = ledger.anchor_through;
        for b in newest.take(CACHED_BREAKPOINTS) {
            let b = b?;
            if b.index <= anchor {
                break;
            }
            ledger.cache.push_front(b);
        }
        ledger.oldest = held.following(anchor, 1)?.first().copied();
        Ok(ledger)
    }

    /// How many breakpoint records the head record says are held, read from
    /// it alone: what the ledger owes before it is loaded.
    ///
    /// # Errors
    ///
    /// The record is not one [`Self::encode`] wrote.
    pub(crate) fn stored_held(bytes: &[u8]) -> StorageResult<u64> {
        Ok(head_fields(bytes)?(5))
    }
}

/// The fields of a head record, by position, once its format and length
/// are checked.
fn head_fields(bytes: &[u8]) -> StorageResult<impl Fn(usize) -> u64 + '_> {
    fn malformed(why: &str) -> StorageError {
        StorageError::Serialization(format!("the closed bound record {why}"))
    }
    let (&format, rest) = bytes.split_first().ok_or_else(|| malformed("is empty"))?;
    if format != FORMAT {
        return Err(malformed(&format!("has format {format}, not {FORMAT}")));
    }
    if rest.len() != HEAD_LEN - 1 {
        return Err(malformed("has the wrong length"));
    }
    Ok(move |n: usize| {
        let mut word = [0u8; 8];
        word.copy_from_slice(&rest[n * 8..n * 8 + 8]);
        u64::from_be_bytes(word)
    })
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
