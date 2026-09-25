//! Physical apply coverage: which journal entries each partition tree holds.
//!
//! A commit's timestamp is its MVCC version, not evidence of where it
//! physically is. A commit that reserved its timestamp early can reach the
//! memtable after a later one was flushed, so "the tree's highest persisted
//! seqno is at least the entry's ts" says nothing about whether the entry
//! itself survived a crash. Coverage answers that question exactly, per tree,
//! in the source's own index space (the embedded journal index).
//!
//! The record lives in the tree it describes, under a reserved key namespace
//! (every key whose first byte is `0x00`), so the tree's own memtable install
//! and flush publish it together with the data it covers:
//!
//! - every apply writes a marker key `0x00 "cov:" be_u64(index)` in the same
//!   lsm batch as the entry's point writes, after the entry's range
//!   tombstones: a flush persists memtables as a prefix in seal order, so a
//!   persisted marker implies every effect of that entry in this tree is
//!   persisted too;
//! - a fold writes `0x00 "cov-base"` = `next` (every index below `next` is
//!   covered) and then one range tombstone over the markers below `next`, at
//!   a seqno above every marker it removes. The base lands first, so a crash
//!   between the two only leaves redundant markers behind.
//!
//! An index is covered in a tree iff it is below the tree's base or its
//! marker exists.

use core::sync::atomic::{AtomicU64, Ordering};
use std::collections::{BTreeSet, HashMap};

use lsm_tree::{AbstractTree, AnyTree, Guard, SeqNo};

use crate::engine::StorageIter;
use crate::error::{StorageError, StorageResult};

/// First key past the engine-reserved namespace. Every user key sorts at or
/// above it; every coverage key sorts below it.
pub(crate) const USER_KEYSPACE_START: &[u8] = &[0x01];

/// How many applied indices accumulate as markers before the commit path
/// folds them into the base.
pub(crate) const FOLD_EVERY: u64 = 4096;

/// The index space a coverage record is kept in. Each source of applied
/// entries numbers them its own way, so each keeps its own record: a store
/// that moved from one to the other must never read one's indices as the
/// other's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Domain {
    /// The embedded journal's entry index.
    Journal,
    /// The Raft log index; `sub` is the proposal's position in its entry.
    Raft,
}

impl Domain {
    const fn tag(self) -> u8 {
        match self {
            Self::Journal => b'j',
            Self::Raft => b'r',
        }
    }

    /// The fold record: every index below the stored value is covered.
    pub(crate) const fn base_key(self) -> [u8; 4] {
        [0x00, b'c', self.tag(), b'b']
    }

    /// The marker for `(index, sub)`; `sub` orders the several applies one
    /// source entry can carry and is 0 when an entry is one apply.
    pub(crate) fn marker_key(self, index: u64, sub: u32) -> [u8; MARKER_LEN] {
        let mut key = [0u8; MARKER_LEN];
        key[..4].copy_from_slice(&[0x00, b'c', self.tag(), b'm']);
        key[4..12].copy_from_slice(&index.to_be_bytes());
        key[12..].copy_from_slice(&sub.to_be_bytes());
        key
    }

    /// Exclusive end of this domain's marker range.
    pub(crate) const fn marker_end(self) -> [u8; 4] {
        [0x00, b'c', self.tag(), b'n']
    }
}

const MARKER_LEN: usize = 16;

/// The coverage marker one apply writes into every tree it touches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Mark {
    pub(crate) domain: Domain,
    pub(crate) index: u64,
    pub(crate) sub: u32,
}

impl Mark {
    pub(crate) fn key(self) -> [u8; MARKER_LEN] {
        self.domain.marker_key(self.index, self.sub)
    }
}

/// The `(index, sub)` a marker key names, or `None` for a malformed key.
fn decode_marker(key: &[u8]) -> Option<(u64, u32)> {
    if key.len() != MARKER_LEN {
        return None;
    }
    let (index, sub) = key.get(4..)?.split_first_chunk::<8>()?;
    let sub: [u8; 4] = sub.try_into().ok()?;
    Some((u64::from_be_bytes(*index), u32::from_be_bytes(sub)))
}

/// The base value covering every index below `next`, followed by the
/// source's own description of its last covered entry (empty when the source
/// needs none).
pub(crate) fn encode_base(next: u64, payload: &[u8]) -> Vec<u8> {
    let mut value = Vec::with_capacity(8 + payload.len());
    value.extend_from_slice(&next.to_be_bytes());
    value.extend_from_slice(payload);
    value
}

fn decode_base(value: &[u8]) -> StorageResult<(u64, &[u8])> {
    let (next, payload) = value.split_first_chunk::<8>().ok_or_else(|| {
        StorageError::Serialization(format!(
            "coverage base holds {} bytes, expected at least 8",
            value.len()
        ))
    })?;
    Ok((u64::from_be_bytes(*next), payload))
}

/// Whether `key` belongs to the engine-reserved namespace.
#[inline]
pub(crate) fn is_reserved(key: &[u8]) -> bool {
    key.first() == Some(&0x00)
}

/// A user-data range tombstone never starts in the reserved namespace: a
/// delete of "everything from the empty key" means everything the user wrote.
pub(crate) fn clamp_user_start(start: &[u8]) -> &[u8] {
    if start < USER_KEYSPACE_START {
        USER_KEYSPACE_START
    } else {
        start
    }
}

/// Scan `prefix` of `tree` at `seqno`, never yielding a reserved key. An
/// empty prefix means the user keyspace, not the whole tree.
pub(crate) fn user_prefix(tree: &AnyTree, prefix: &[u8], seqno: SeqNo) -> StorageIter {
    match prefix.first() {
        None => Box::new(tree.range(USER_KEYSPACE_START.., seqno, None)),
        Some(0x00) => Box::new(std::iter::empty()),
        Some(_) => Box::new(tree.prefix(prefix, seqno, None)),
    }
}

/// The durable coverage of one tree in one domain, read when it opens.
#[derive(Debug, Default)]
pub(crate) struct TreeCoverage {
    /// `None` when the tree carries no coverage record at all.
    base: Option<u64>,
    /// The source's description of the last entry the base covers.
    base_payload: Vec<u8>,
    /// Marked `(index, sub)` pairs at or above the base.
    sparse: BTreeSet<(u64, u32)>,
}

impl TreeCoverage {
    /// Read the base and the markers above it, at the latest version.
    pub(crate) fn read(tree: &AnyTree, domain: Domain) -> StorageResult<Self> {
        let (base, base_payload) = match tree.get(domain.base_key(), SeqNo::MAX)? {
            Some(value) => {
                let (next, payload) = decode_base(&value)?;
                (Some(next), payload.to_vec())
            }
            None => (None, Vec::new()),
        };
        let from = domain.marker_key(base.unwrap_or(0), 0);
        let end = domain.marker_end();
        let mut sparse = BTreeSet::new();
        for guard in tree.range(from.as_slice()..end.as_slice(), SeqNo::MAX, None) {
            let (key, _) = guard.into_inner()?;
            let marker = decode_marker(&key).ok_or_else(|| {
                StorageError::Serialization(format!(
                    "coverage marker key holds {} bytes, expected {MARKER_LEN}",
                    key.len()
                ))
            })?;
            sparse.insert(marker);
        }
        Ok(Self {
            base,
            base_payload,
            sparse,
        })
    }

    /// The base (every index below it covered) and its payload.
    pub(crate) fn base(&self) -> Option<(u64, &[u8])> {
        self.base.map(|next| (next, self.base_payload.as_slice()))
    }

    /// Whether the tree carries a coverage record (a base).
    pub(crate) fn has_record(&self) -> bool {
        self.base.is_some()
    }

    /// The indices this tree holds markers for above its base.
    pub(crate) fn marked_indices(&self) -> impl Iterator<Item = u64> + '_ {
        self.sparse.iter().map(|&(index, _)| index)
    }

    /// One past the highest index this tree records as covered.
    pub(crate) fn next_uncovered(&self) -> u64 {
        let above = self.sparse.last().map_or(0, |&(i, _)| i + 1);
        self.base.unwrap_or(0).max(above)
    }

    /// Whether apply `sub` of source entry `index` is physically in this tree.
    pub(crate) fn contains(&self, index: u64, sub: u32) -> bool {
        index < self.base.unwrap_or(0) || self.sparse.contains(&(index, sub))
    }
}

/// The journal indices applied in this process: every index below `next`,
/// plus the ones above it that finished out of order.
#[derive(Debug)]
pub(crate) struct AppliedSet {
    next: u64,
    above: BTreeSet<u64>,
}

impl AppliedSet {
    /// Every index below `next` is applied.
    pub(crate) fn starting_at(next: u64) -> Self {
        Self {
            next,
            above: BTreeSet::new(),
        }
    }

    /// Record that `index` is applied in every tree it touches.
    pub(crate) fn mark(&mut self, index: u64) {
        if index < self.next {
            return;
        }
        if index > self.next {
            self.above.insert(index);
            return;
        }
        self.next += 1;
        while self.above.remove(&self.next) {
            self.next += 1;
        }
    }

    /// The contiguous applied prefix: every index below it is applied.
    pub(crate) fn next(&self) -> u64 {
        self.next
    }

    /// Indices above the prefix that are applied.
    pub(crate) fn above(&self) -> impl Iterator<Item = u64> + '_ {
        self.above.iter().copied()
    }
}

/// The engine's coverage state: what was applied, and what was folded.
#[derive(Debug)]
pub(crate) struct Coverage {
    applied: parking_lot::Mutex<AppliedSet>,
    /// The base last written to every tree. Held for the whole fold so two
    /// folds never interleave their base and tombstone writes.
    folded: parking_lot::Mutex<u64>,
    /// Lock-free copy of `folded` for the commit path's "is a fold due" test.
    folded_hint: AtomicU64,
    /// Per `STORAGE COLUMNAR` table, the indices marked in its tree and not
    /// folded yet: a table fold removes them by name.
    table_markers: parking_lot::Mutex<HashMap<String, Vec<u64>>>,
}

impl Coverage {
    /// Coverage after an open that left every index below `next` applied.
    pub(crate) fn new(next: u64) -> Self {
        Self {
            applied: parking_lot::Mutex::new(AppliedSet::starting_at(next)),
            // The trees may still hold markers from before the open, so the
            // first fold removes markers from index 0.
            folded: parking_lot::Mutex::new(0),
            folded_hint: AtomicU64::new(0),
            table_markers: parking_lot::Mutex::new(HashMap::new()),
        }
    }

    /// Record that `table`'s tree now holds the marker for `index`.
    pub(crate) fn note_table_marker(&self, table: &str, index: u64) {
        let mut markers = self.table_markers.lock();
        match markers.get_mut(table) {
            Some(indices) => indices.push(index),
            None => {
                markers.insert(table.to_owned(), vec![index]);
            }
        }
    }

    /// Take, per table, the unfolded markers below `next`; the ones at or
    /// above it stay for a later fold.
    pub(crate) fn take_table_markers_below(&self, next: u64) -> Vec<(String, Vec<u64>)> {
        let mut markers = self.table_markers.lock();
        let mut taken = Vec::new();
        markers.retain(|table, indices| {
            let (below, above): (Vec<u64>, Vec<u64>) = indices.iter().partition(|&&i| i < next);
            if !below.is_empty() {
                taken.push((table.clone(), below));
            }
            *indices = above;
            !indices.is_empty()
        });
        taken
    }

    /// Record `index` as applied; returns whether a fold is due.
    pub(crate) fn mark_applied(&self, index: u64) -> bool {
        let next = {
            let mut applied = self.applied.lock();
            applied.mark(index);
            applied.next()
        };
        // `folded` is only ever set to an applied prefix, and the prefix never
        // shrinks, so it cannot exceed `next`.
        let folded = self.folded_hint.load(Ordering::Relaxed);
        debug_assert!(
            folded <= next,
            "folded coverage ahead of the applied prefix"
        );
        next - folded >= FOLD_EVERY
    }

    pub(crate) fn applied_prefix(&self) -> u64 {
        self.applied.lock().next()
    }

    /// The applied prefix and the applied indices above it, read together.
    pub(crate) fn applied_snapshot(&self) -> (u64, Vec<u64>) {
        let applied = self.applied.lock();
        (applied.next(), applied.above().collect())
    }

    /// Serialize folds: the guard holds the base last written to every tree.
    pub(crate) fn lock_folded(&self) -> FoldGuard<'_> {
        FoldGuard {
            guard: self.folded.lock(),
            hint: &self.folded_hint,
        }
    }

    /// Like [`Self::lock_folded`], but gives up when a fold is running.
    pub(crate) fn try_lock_folded(&self) -> Option<FoldGuard<'_>> {
        self.folded.try_lock().map(|guard| FoldGuard {
            guard,
            hint: &self.folded_hint,
        })
    }
}

/// Exclusive access to the folded base while a fold writes it.
pub(crate) struct FoldGuard<'a> {
    guard: parking_lot::MutexGuard<'a, u64>,
    hint: &'a AtomicU64,
}

impl FoldGuard<'_> {
    /// The base last written to every tree.
    pub(crate) fn folded(&self) -> u64 {
        *self.guard
    }

    /// Record that every tree now holds base `next`.
    pub(crate) fn set_folded(&mut self, next: u64) {
        *self.guard = next;
        self.hint.store(next, Ordering::Relaxed);
    }
}

/// Write the fold of every index below `next` into a partition tree: the base
/// first, then the tombstone over the markers from `from`. `seqno` must
/// exceed the seqno of every marker below `next`.
pub(crate) fn write_fold(
    tree: &AnyTree,
    domain: Domain,
    from: u64,
    next: u64,
    payload: &[u8],
    seqno: SeqNo,
) {
    tree.insert(domain.base_key(), encode_base(next, payload), seqno);
    if from < next {
        tree.remove_range(
            domain.marker_key(from, 0).to_vec(),
            domain.marker_key(next, 0).to_vec(),
            seqno,
        );
    }
}

/// [`write_fold`] for a `STORAGE COLUMNAR` table tree. A columnar table is
/// written only by its own rows, so its markers are few and known, and each
/// is removed by a point tombstone rather than a range one: `markers` are the
/// indices below `next` marked in this table since its last fold.
pub(crate) fn write_table_fold(
    tree: &AnyTree,
    domain: Domain,
    next: u64,
    markers: &[u64],
    seqno: SeqNo,
) {
    tree.insert(domain.base_key(), encode_base(next, &[]), seqno);
    for &index in markers {
        tree.remove(domain.marker_key(index, 0), seqno);
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
