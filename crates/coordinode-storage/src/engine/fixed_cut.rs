//! Fixed-cut source handoff for one history: a pinned read view below an
//! exclusive bound `B`, the first journal position `L` to replay, and a
//! finite join boundary `H`.
//!
//! Every committed unit before `L` has its commit timestamp below `B`, and
//! every committed unit below `B` is before `H`. The view holds exactly the
//! state below `B`. A consumer copies the view, then reconciles every unit
//! of `[L, H)`: one below `B` is already in the view, one at or above it is
//! applied once. From `H` on it follows the journal as any reader does. A
//! physical journal entry may carry units on both sides of `B`, so the
//! decision is per unit, by the unit's own commit timestamp.

use coordinode_core::budget::{BudgetStop, MemoryCharge, QueryBudget};
use coordinode_core::txn::operation::LogPosition;
use coordinode_core::txn::proposal::{Mutation, PartitionId};

use crate::engine::coordinator::SnapshotPin;
use crate::engine::partition::Partition;
use crate::error::{StorageError, StorageResult};

/// The rows a cut hands over: the keys of one partition under a prefix. A
/// cut of rows recovers rows only; it does not stand for the whole store, a
/// columnar table or an index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CutScope {
    partition: Partition,
    prefix: Vec<u8>,
    /// The least key past every key under the prefix; `None` when the scope
    /// reaches the end of the key space.
    end: Option<Vec<u8>>,
}

impl CutScope {
    /// The rows of `partition` whose keys start with `prefix`.
    ///
    /// # Errors
    ///
    /// [`StorageError::CutUnsupported`] for the index and schema partitions,
    /// whose state follows from interpretations a row copy does not carry
    /// (index installations, metadata decisions), and for the Raft
    /// partition, which holds no rows of the history.
    pub fn rows(partition: Partition, prefix: Vec<u8>) -> StorageResult<Self> {
        if matches!(
            partition,
            Partition::Idx | Partition::Schema | Partition::Raft
        ) {
            return Err(StorageError::CutUnsupported {
                reason: format!("the {partition:?} partition is not a row scope"),
            });
        }
        Ok(Self {
            partition,
            end: prefix_successor(&prefix),
            prefix,
        })
    }

    /// The partition.
    #[must_use]
    pub fn partition(&self) -> Partition {
        self.partition
    }

    /// The key prefix.
    #[must_use]
    pub fn prefix(&self) -> &[u8] {
        &self.prefix
    }

    /// Whether `partition` is this scope's.
    pub(crate) fn holds_partition(&self, partition: PartitionId) -> bool {
        Partition::from(partition) == self.partition
    }

    /// Whether `key` of this scope's partition is in it.
    pub(crate) fn holds_key(&self, key: &[u8]) -> bool {
        key.starts_with(&self.prefix)
    }

    /// Whether the rows of this scope hold what `target` names: a node's
    /// record in the node partition, a relationship's properties or instance
    /// in the edge property partition, its presence in its forward adjacency
    /// list. A change of what a type implies moves the membership of node
    /// records, so every node scope takes it.
    pub(crate) fn holds_target(&self, target: &crate::engine::change_feed::Target<'_>) -> bool {
        use crate::engine::change_feed::Target;
        match target {
            Target::Node(key) => self.partition == Partition::Node && self.holds_key(key),
            Target::RelationshipProperties(key) | Target::RelationshipInstance(key) => {
                self.partition == Partition::EdgeProp && self.holds_key(key)
            }
            Target::Relationship {
                edge_type, source, ..
            } => {
                if self.partition != Partition::Adj {
                    return false;
                }
                let mut key = Vec::with_capacity(b"adj:".len() + edge_type.len() + 13);
                key.extend_from_slice(b"adj:");
                key.extend_from_slice(edge_type);
                key.extend_from_slice(b":out:");
                key.extend_from_slice(&source.to_be_bytes());
                self.holds_key(&key)
            }
            Target::Scope { .. } | Target::Label(_) => self.partition == Partition::Node,
            Target::BlobOwner(key) => self.partition == Partition::BlobRef && self.holds_key(key),
        }
    }

    /// Whether some key of `[start, end)` is in it.
    pub(crate) fn meets_range(&self, start: &[u8], end: &[u8]) -> bool {
        self.clip_range(start, end).is_some()
    }

    /// The part of `[start, end)` inside the scope, `None` when nothing of
    /// it is: bounds borrowed from the range or the scope, copied by whoever
    /// keeps them. Keys under the prefix sort from the prefix up to its
    /// successor, or to the end of the key space when it has none.
    pub(crate) fn clip_range<'s>(
        &'s self,
        start: &'s [u8],
        end: &'s [u8],
    ) -> Option<(&'s [u8], &'s [u8])> {
        let lo = start.max(self.prefix.as_slice());
        let hi = match &self.end {
            Some(after) => end.min(after.as_slice()),
            None => end,
        };
        (lo < hi).then_some((lo, hi))
    }

    /// The least key past every key under the prefix; `None` when the scope
    /// reaches the end of the key space.
    pub(crate) fn end(&self) -> Option<&[u8]> {
        self.end.as_deref()
    }
}

/// The least key sorting after every key starting with `prefix`; `None`
/// when no key does (an empty or all-`0xFF` prefix).
pub(crate) fn prefix_successor(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut end = prefix.to_vec();
    while let Some(last) = end.pop() {
        if last < 0xFF {
            end.push(last + 1);
            return Some(end);
        }
    }
    None
}

/// Bytes of a block a page's rows are copied into, unless one row is longer.
const ROW_BLOCK: usize = 64 * 1024;

/// Keys and values of a cut's view, in key order, copied into blocks that
/// stay charged to the read's budget while they are owned. The budget is
/// asked when a block or the row index grows, not for every row.
#[derive(Debug, Default)]
pub struct CutRows {
    /// Each holds rows back to back, never past its first capacity: a block
    /// is not moved once rows are in it.
    blocks: Vec<Vec<u8>>,
    rows: Vec<RowAt>,
}

/// Where a row lies: its key, then its value, in one block.
#[derive(Debug, Clone, Copy)]
struct RowAt {
    block: usize,
    start: usize,
    key: usize,
    value: usize,
}

impl CutRows {
    /// The number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the page holds no row.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    fn row(&self, at: &RowAt) -> (&[u8], &[u8]) {
        let block = &self.blocks[at.block];
        let value = at.start + at.key;
        (&block[at.start..value], &block[value..value + at.value])
    }

    /// The key and value of row `i`.
    #[must_use]
    pub fn get(&self, i: usize) -> Option<(&[u8], &[u8])> {
        self.rows.get(i).map(|at| self.row(at))
    }

    /// The rows, in key order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&[u8], &[u8])> + '_ {
        self.rows.iter().map(|at| self.row(at))
    }

    /// The last key: where the next page goes on from.
    #[must_use]
    pub fn last_key(&self) -> Option<&[u8]> {
        self.rows.last().map(|at| self.row(at).0)
    }

    /// Bytes the page's blocks and row index hold, all charged to the
    /// budget it was read within.
    #[must_use]
    pub fn held(&self) -> usize {
        self.blocks.iter().map(Vec::capacity).sum::<usize>()
            + self.blocks.capacity() * core::mem::size_of::<Vec<u8>>()
            + self.rows.capacity() * core::mem::size_of::<RowAt>()
    }

    /// Copy a row in, growing a block or the index first under `charge`, a
    /// charge on `budget`, when they are full.
    ///
    /// # Errors
    ///
    /// The budget refused the growth; the rows held stay as they were.
    pub(crate) fn push(
        &mut self,
        key: &[u8],
        value: &[u8],
        charge: &mut MemoryCharge<'_>,
        budget: &QueryBudget,
    ) -> Result<(), BudgetStop> {
        let len = key.len() + value.len();
        if self.rows.len() == self.rows.capacity() {
            grow_exact(&mut self.rows, charge)?;
        }
        let fits = self
            .blocks
            .last()
            .is_some_and(|block| block.capacity() - block.len() >= len);
        if !fits {
            if self.blocks.len() == self.blocks.capacity() {
                grow_exact(&mut self.blocks, charge)?;
            }
            // A whole block, or the budget's last room when a block does not
            // fit: one more block, not one per row, near the limit.
            // A budget refuses every reservation past its limit.
            let room = budget.memory_limit() - budget.memory_used();
            // Slices of memory that exist: their lengths fit in u64.
            let size = (len.max(ROW_BLOCK) as u64).min(room).max(len as u64);
            charge.grow(size)?;
            self.blocks.push(Vec::with_capacity(size as usize));
        }
        let block = self.blocks.len() - 1;
        let bytes = &mut self.blocks[block];
        let start = bytes.len();
        // Within the capacity checked above: the block does not move.
        bytes.extend_from_slice(key);
        bytes.extend_from_slice(value);
        self.rows.push(RowAt {
            block,
            start,
            key: key.len(),
            value: value.len(),
        });
        Ok(())
    }
}

/// Double `v`'s capacity under `charge`, which holds the old and the new
/// buffer while the elements move and then only the new one.
fn grow_exact<T>(v: &mut Vec<T>, charge: &mut MemoryCharge<'_>) -> Result<(), BudgetStop> {
    let slot = core::mem::size_of::<T>() as u64;
    let old = v.capacity();
    let new = old.max(2) * 2;
    charge.grow(new as u64 * slot)?;
    v.reserve_exact(new - v.len());
    charge.shrink(old as u64 * slot);
    Ok(())
}

/// The working memory one log entry's reading was given, already charged
/// to the read's budget: the units decoding takes are spent from it, and
/// what it holds at the end is what the entry keeps.
#[derive(Debug)]
pub(crate) struct EntryPool {
    /// Not yet spent.
    left: u64,
    /// Held by the entry's decoded units.
    retained: u64,
    /// Added to the batch's capacity for units.
    grown: u64,
}

impl EntryPool {
    /// A pool of `left` bytes.
    pub(crate) fn new(left: u64) -> Self {
        Self {
            left,
            retained: 0,
            grown: 0,
        }
    }

    /// Bounds for decoding one unit within what is left: its operations
    /// and whatever the decoder holds per operation take at most half, the
    /// bytes it expands to, kept once in the unit and once more at most
    /// while decoding, the other half.
    pub(crate) fn decode_limits(&self) -> coordinode_core::txn::frame::DecodeLimits {
        // A pool past the address space bounds nothing more.
        let left = usize::try_from(self.left).unwrap_or(usize::MAX);
        coordinode_core::txn::frame::DecodeLimits {
            max_frame_bytes: coordinode_core::txn::frame::DecodeLimits::DEFAULT.max_frame_bytes,
            max_ops: left / 2 / (core::mem::size_of::<Mutation>() + DECODER_PER_OP),
            max_expanded_bytes: left / 4,
        }
    }

    /// Keep `bytes` for a decoded unit; `false` when they do not fit.
    pub(crate) fn retain(&mut self, bytes: u64) -> bool {
        let fits = self.spend(bytes);
        if fits {
            self.retained += bytes;
        }
        fits
    }

    /// Free `bytes` kept for the entry's units, once their memory is freed:
    /// a list that grew into a larger buffer.
    pub(crate) fn unretain(&mut self, bytes: u64) {
        self.retained -= bytes;
        self.left += bytes;
    }

    /// Grow the batch's units by `bytes`; `false` when they do not fit.
    pub(crate) fn grow(&mut self, bytes: u64) -> bool {
        let fits = self.spend(bytes);
        if fits {
            self.grown += bytes;
        }
        fits
    }

    fn spend(&mut self, bytes: u64) -> bool {
        match self.left.checked_sub(bytes) {
            Some(left) => {
                self.left = left;
                true
            }
            None => false,
        }
    }

    /// What the entry keeps: its units and the batch's growth for them.
    pub(crate) fn kept(&self) -> u64 {
        self.retained + self.grown
    }

    /// The batch's growth alone.
    pub(crate) fn grown(&self) -> u64 {
        self.grown
    }
}

/// Working state a unit's projection takes while it runs and gives back.
impl crate::engine::change_feed::Room for EntryPool {
    fn take(&mut self, bytes: u64) -> bool {
        self.spend(bytes)
    }

    fn give(&mut self, bytes: u64) {
        self.left += bytes;
    }
}

/// What the frame decoder holds per operation besides the operation: its
/// literal references, with room for their list to grow.
const DECODER_PER_OP: usize = 2 * core::mem::size_of::<&[u8]>();

/// The log a cut's positions count entries of: a store's own journal, named
/// by the deployment it founded, or one group's consensus log, named by its
/// history. Positions of different histories are not comparable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CutHistory {
    /// The journal of a store of this deployment.
    Journal(coordinode_core::group::DeploymentId),
    /// The log of this group of this deployment.
    Group(coordinode_core::txn::read_concern::HistoryId),
}

/// Where ordinary change reading resumes after a cut's join: a position in
/// one history's log, for one scope. Read back with
/// [`crate::engine::core::StorageEngine::resume_position`], which refuses it
/// in another history or for another scope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CutResume {
    pub(crate) history: CutHistory,
    pub(crate) scope: CutScope,
    pub(crate) next: u64,
}

impl CutResume {
    /// The history the position counts entries of.
    #[must_use]
    pub fn history(&self) -> CutHistory {
        self.history
    }

    /// The scope the position was reached for.
    #[must_use]
    pub fn scope(&self) -> &CutScope {
        &self.scope
    }
}

/// A pinned view of one history below `bound`, and where its journal
/// continuation starts and joins. Dropping it releases the view.
pub struct FixedCut {
    pub(crate) first: u64,
    pub(crate) bound: u64,
    pub(crate) join_end: u64,
    pub(crate) history: CutHistory,
    pub(crate) scope: CutScope,
    pub(crate) _pin: SnapshotPin,
}

impl std::fmt::Debug for FixedCut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FixedCut")
            .field("first", &self.first)
            .field("bound", &self.bound)
            .field("join_end", &self.join_end)
            .field("scope", &self.scope)
            .finish_non_exhaustive()
    }
}

impl FixedCut {
    /// The rows the cut hands over.
    #[must_use]
    pub fn scope(&self) -> &CutScope {
        &self.scope
    }

    /// `L`: the first journal position the continuation replays.
    #[must_use]
    pub fn first(&self) -> u64 {
        self.first
    }

    /// `B`: the exclusive commit timestamp bound of the view. Reads at this
    /// snapshot see every commit below it and none at or above it.
    #[must_use]
    pub fn bound(&self) -> u64 {
        self.bound
    }

    /// `H`: the end of the join. Units before it are reconciled against the
    /// view; no unit below the bound appears at or after it.
    #[must_use]
    pub fn join_end(&self) -> u64 {
        self.join_end
    }

    /// The history the cut belongs to: a continuation read from another
    /// history is refused.
    #[must_use]
    pub fn history(&self) -> CutHistory {
        self.history
    }

    /// Where ordinary reading resumes once the continuation is read to
    /// `next`; `None` before the join is complete.
    #[must_use]
    pub fn resume_at(&self, next: u64) -> Option<CutResume> {
        (next >= self.join_end).then(|| CutResume {
            history: self.history,
            scope: self.scope.clone(),
            next,
        })
    }

    /// Whether the unit committed at `commit_ts`, found at `position`, is
    /// already represented by the view.
    #[must_use]
    pub fn in_view(&self, position: LogPosition, commit_ts: u64) -> bool {
        position.index < self.join_end && commit_ts < self.bound
    }
}

/// One committed unit of the journal, as it was recorded: its position, its
/// own commit timestamp and its data changes.
#[derive(Debug, Clone, PartialEq)]
pub struct CutUnit {
    /// Where the unit is: the journal entry and its place in the entry.
    pub position: LogPosition,
    /// The unit's own commit timestamp, not its entry's.
    pub commit_ts: u64,
    /// The unit's data changes; empty when its operation binding refused it.
    pub mutations: Vec<Mutation>,
    /// Whether its operation binding refused it: it took no effect.
    pub refused: bool,
    /// Whether the cut's view already holds it.
    pub in_view: bool,
    /// The unit's logical events whose targets are in the cut's scope,
    /// numbered in the unit's whole projection before the scope chose them;
    /// `None` for a unit recorded without the context the change feed reads
    /// it by, which only its rows can be applied from.
    pub events: Option<CutEvents>,
}

/// The logical events of one unit inside a cut's scope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CutEvents {
    /// The events, in ordinal order, each delivered as applying to the scope.
    pub events: Vec<crate::engine::change_feed::ChangeEvent>,
    /// How many events the unit's whole projection holds, in the scope or
    /// not.
    pub projected: u32,
}

/// A read of a cut's continuation: its units, and the position the next
/// read starts at.
#[derive(Debug, Clone, PartialEq)]
pub struct CutRead {
    /// The units of the entries read, in log order.
    pub units: Vec<CutUnit>,
    /// The position after the last entry read; where it started when the
    /// journal held nothing more yet.
    pub next: u64,
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests;
