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
use std::collections::BTreeSet;

use lsm_tree::{AbstractTree, AnyTree, Guard, SeqNo};

use crate::engine::StorageIter;
use crate::error::{StorageError, StorageResult};

/// First key past the engine-reserved namespace. Every user key sorts at or
/// above it; every coverage key sorts below it.
pub(crate) const USER_KEYSPACE_START: &[u8] = &[0x01];

/// The fold record: every index below the stored value is covered.
pub(crate) const BASE_KEY: &[u8] = b"\x00cov-base";

/// Prefix of the per-index markers.
const MARKER_PREFIX: &[u8] = b"\x00cov:";

/// Exclusive end of the marker range (`:` + 1).
const MARKER_END: &[u8] = b"\x00cov;";

/// How many applied indices accumulate as markers before the commit path
/// folds them into the base.
pub(crate) const FOLD_EVERY: u64 = 4096;

/// The marker key for journal `index`.
pub(crate) fn marker_key(index: u64) -> [u8; 13] {
    let mut key = [0u8; 13];
    key[..5].copy_from_slice(MARKER_PREFIX);
    key[5..].copy_from_slice(&index.to_be_bytes());
    key
}

/// The base value covering every index below `next`.
pub(crate) fn encode_base(next: u64) -> [u8; 8] {
    next.to_be_bytes()
}

fn decode_base(value: &[u8]) -> StorageResult<u64> {
    let bytes: [u8; 8] = value.try_into().map_err(|_| {
        StorageError::Serialization(format!(
            "coverage base holds {} bytes, expected 8",
            value.len()
        ))
    })?;
    Ok(u64::from_be_bytes(bytes))
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

/// The durable coverage of one tree, read when the engine opens.
#[derive(Debug, Default)]
pub(crate) struct TreeCoverage {
    /// `None` when the tree carries no coverage record at all.
    base: Option<u64>,
    /// Marker indices at or above the base.
    sparse: BTreeSet<u64>,
}

impl TreeCoverage {
    /// Read the base and the markers above it, at the latest version.
    pub(crate) fn read(tree: &AnyTree) -> StorageResult<Self> {
        let base = tree
            .get(BASE_KEY, SeqNo::MAX)?
            .map(|v| decode_base(&v))
            .transpose()?;
        let from = marker_key(base.unwrap_or(0));
        let mut sparse = BTreeSet::new();
        for guard in tree.range(from.as_slice()..MARKER_END, SeqNo::MAX, None) {
            let (key, _) = guard.into_inner()?;
            let index: [u8; 8] = key[MARKER_PREFIX.len()..].try_into().map_err(|_| {
                StorageError::Serialization(format!(
                    "coverage marker key holds {} bytes, expected 13",
                    key.len()
                ))
            })?;
            sparse.insert(u64::from_be_bytes(index));
        }
        Ok(Self { base, sparse })
    }

    /// Whether the tree carries a coverage record (a base).
    pub(crate) fn has_record(&self) -> bool {
        self.base.is_some()
    }

    /// One past the highest index this tree records as covered.
    pub(crate) fn next_uncovered(&self) -> u64 {
        let above = self.sparse.last().map_or(0, |&i| i + 1);
        self.base.unwrap_or(0).max(above)
    }

    /// Whether journal entry `index` is physically in this tree.
    pub(crate) fn contains(&self, index: u64) -> bool {
        index < self.base.unwrap_or(0) || self.sparse.contains(&index)
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
        }
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

/// Write the fold of every index below `next` into `tree`: the base first,
/// then the tombstone over the markers from `from`. `seqno` must exceed the
/// seqno of every marker below `next`.
pub(crate) fn write_fold(tree: &AnyTree, from: u64, next: u64, seqno: SeqNo) {
    tree.insert(BASE_KEY, encode_base(next), seqno);
    if from < next {
        tree.remove_range(marker_key(from).to_vec(), marker_key(next).to_vec(), seqno);
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
