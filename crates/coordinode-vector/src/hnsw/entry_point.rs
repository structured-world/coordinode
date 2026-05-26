//! Lock-free packed HNSW entry-point.
//!
//! Stores `Option<(level, idx)>` in a single `AtomicU64`. The classical
//! HNSW entry-point is the highest-layer node; concurrent inserts that
//! happen to land on novel max-layers race to promote it. Per
//! `arch/search/vector-parallel-insert.md` §"Layer-promotion race" the
//! correct primitive is a CAS-loop on a single atomic.
//!
//! # Wire format
//!
//! ```text
//!   u64:  [ 8 bits level ][        56 bits idx        ]
//! ```
//!
//! `level` fits 0..=254 (we reserve 0xFF combined with idx=0x00ff_ffff_ffff_ffff
//! as `u64::MAX` to mean "no entry yet"). HNSW practical max layer is
//! ~16 for any realistic dataset (log_M(N) with M=16, N=1B ≈ 7.5),
//! so 8 bits is comfortable.
//!
//! `idx` fits 0..=2^56 − 1 — same scale as the inline neighbour-list
//! capacity per node, comfortably above any single-shard target. The
//! coordinode shard model caps single-shard HNSW well below 2^40
//! nodes (1T), so the encoding has decades of headroom.
//!
//! # no-std
//!
//! `core::sync::atomic::AtomicU64` is core-level — this type does not
//! tighten the crate's std tier. Loom interleaving tests pull in
//! `loom::sync::atomic::AtomicU64` under `#[cfg(loom)]`; both share
//! the same surface API so the production code is unchanged.

// no-std: core::sync::atomic — already no-std-ready. loom variant is
// test-only and gated by --cfg loom.
#[cfg(not(loom))]
use core::sync::atomic::{AtomicU64, Ordering};
#[cfg(loom)]
use loom::sync::atomic::{AtomicU64, Ordering};

/// Sentinel: "no entry point assigned yet". All 1-bits is unreachable
/// in a valid `pack(level, idx)` encoding because we cap `level ≤ 254`.
const SENTINEL: u64 = u64::MAX;

/// Width of the `idx` field. The remaining `64 - IDX_BITS = 8` bits
/// hold the `level`.
const IDX_BITS: u32 = 56;

/// Mask covering the `idx` bits — used for unpacking and the level
/// shift in [`pack`].
const IDX_MASK: u64 = (1u64 << IDX_BITS) - 1;

/// Cap on the encodable layer. Layer 0xFF is reserved as part of the
/// `SENTINEL` value, so the highest usable level is one less.
const MAX_LEVEL: u8 = 254;

/// Pack a `(level, idx)` pair into the storage word.
///
/// Panics in debug if `level > MAX_LEVEL` or `idx` overflows
/// `IDX_BITS` — both are programming errors that would otherwise
/// silently fold a valid level into the sentinel.
#[inline]
fn pack(level: u8, idx: u64) -> u64 {
    debug_assert!(level <= MAX_LEVEL, "level {level} exceeds cap {MAX_LEVEL}");
    debug_assert!(idx & !IDX_MASK == 0, "idx {idx} overflows {IDX_BITS} bits");
    ((level as u64) << IDX_BITS) | (idx & IDX_MASK)
}

/// Inverse of [`pack`]. Returns `(level, idx)`.
#[inline]
fn unpack(word: u64) -> (u8, u64) {
    let level = (word >> IDX_BITS) as u8;
    let idx = word & IDX_MASK;
    (level, idx)
}

/// Atomic, packed entry-point for an HNSW index. Multiple concurrent
/// inserts that land on novel max-layers race through [`try_promote`]
/// — the highest layer wins, the rest no-op.
#[derive(Debug)]
pub struct EntryPoint {
    /// Packed `(level, idx)` (or `SENTINEL` when no node has been
    /// inserted yet).
    inner: AtomicU64,
}

impl EntryPoint {
    /// Create an empty entry-point. [`load`](Self::load) returns
    /// `None` until the first successful [`try_promote`].
    pub fn new() -> Self {
        Self {
            inner: AtomicU64::new(SENTINEL),
        }
    }

    /// Read the current entry-point. Returns `None` while the index
    /// has no nodes; `Some((level, idx))` afterwards.
    #[inline]
    pub fn load(&self) -> Option<(u8, u64)> {
        let word = self.inner.load(Ordering::SeqCst);
        if word == SENTINEL {
            None
        } else {
            Some(unpack(word))
        }
    }

    /// Convenience for search-path callers: returns the starting
    /// `(idx, top_level)` pair as `usize` (HNSW search code works in
    /// `usize` throughout for Vec indexing + layer counting), or
    /// `None` on an empty index.
    ///
    /// Important: this is a **single** atomic load, so the two
    /// fields come from a consistent snapshot. The previous
    /// idiom of calling `idx_or_zero()` and `max_level()`
    /// separately did TWO loads, and a concurrent
    /// [`try_promote`] between them could leave the caller with
    /// `idx` from snapshot S1 and `level` from snapshot S2.
    /// Searchers tolerated the inconsistency (top-level layers
    /// just iterate over a slightly stale start node) but it's
    /// noise we can elide cheaply.
    #[inline]
    pub fn for_search(&self) -> Option<(usize, usize)> {
        self.load()
            .map(|(level, idx)| (idx as usize, level as usize))
    }

    /// Outcome of [`try_promote`] — communicates whether the caller's
    /// insert actually owns the entry-point now, or another insert
    /// already had a higher (or equal) level.
    ///
    /// Either result means the entry-point invariant holds after the
    /// call: caller doesn't need to retry from outside.
    pub fn try_promote(&self, new_level: u8, new_idx: u64) -> PromoteOutcome {
        let new_word = pack(new_level, new_idx);
        loop {
            let cur = self.inner.load(Ordering::SeqCst);
            if cur == SENTINEL {
                // First insert wins unopposed if no one else CAS'd
                // in between.
                match self.inner.compare_exchange(
                    SENTINEL,
                    new_word,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => return PromoteOutcome::Installed,
                    Err(_) => continue, // someone else won the seed race
                }
            }
            let (cur_level, _cur_idx) = unpack(cur);
            if new_level <= cur_level {
                // Either we tied or another writer already promoted
                // past us. Either way the invariant ("entry-point is
                // at the global max layer") holds without our help.
                return PromoteOutcome::NotNeeded {
                    current_level: cur_level,
                };
            }
            match self
                .inner
                .compare_exchange(cur, new_word, Ordering::SeqCst, Ordering::SeqCst)
            {
                Ok(_) => return PromoteOutcome::Installed,
                Err(_) => continue, // another writer raced us; re-load and retry
            }
        }
    }
}

impl Default for EntryPoint {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of [`EntryPoint::try_promote`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromoteOutcome {
    /// The caller's `(level, idx)` is now the entry-point.
    Installed,
    /// The current entry-point already has a level `≥` the caller's
    /// requested level — caller's node should NOT be the entry-point.
    /// `current_level` is the value the caller observed.
    NotNeeded { current_level: u8 },
}

#[cfg(test)]
#[cfg(not(loom))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_roundtrip() {
        for level in [0u8, 1, 7, 16, 100, MAX_LEVEL] {
            for idx in [0u64, 1, 42, 1 << 30, IDX_MASK] {
                let (l, i) = unpack(pack(level, idx));
                assert_eq!(l, level, "level roundtrip");
                assert_eq!(i, idx, "idx roundtrip");
            }
        }
    }

    #[test]
    fn new_is_empty() {
        let ep = EntryPoint::new();
        assert_eq!(ep.load(), None);
        assert_eq!(ep.for_search(), None);
    }

    #[test]
    fn first_promote_installs() {
        let ep = EntryPoint::new();
        let out = ep.try_promote(3, 42);
        assert_eq!(out, PromoteOutcome::Installed);
        assert_eq!(ep.load(), Some((3, 42)));
    }

    #[test]
    fn second_promote_with_lower_level_is_noop() {
        let ep = EntryPoint::new();
        ep.try_promote(5, 100);
        let out = ep.try_promote(3, 99);
        assert_eq!(out, PromoteOutcome::NotNeeded { current_level: 5 });
        assert_eq!(ep.load(), Some((5, 100)));
    }

    #[test]
    fn second_promote_with_equal_level_is_noop() {
        let ep = EntryPoint::new();
        ep.try_promote(5, 100);
        // Two nodes hit the same novel max-level: linearisation rule
        // is "first wins" — the second is a no-op.
        let out = ep.try_promote(5, 200);
        assert_eq!(out, PromoteOutcome::NotNeeded { current_level: 5 });
        assert_eq!(ep.load(), Some((5, 100)));
    }

    #[test]
    fn promote_with_higher_level_replaces() {
        let ep = EntryPoint::new();
        ep.try_promote(3, 42);
        let out = ep.try_promote(7, 99);
        assert_eq!(out, PromoteOutcome::Installed);
        assert_eq!(ep.load(), Some((7, 99)));
    }

    #[test]
    fn for_search_tracks_promote() {
        let ep = EntryPoint::new();
        assert_eq!(ep.for_search(), None);
        ep.try_promote(3, 0);
        assert_eq!(ep.for_search(), Some((0, 3)));
        ep.try_promote(7, 1);
        assert_eq!(ep.for_search(), Some((1, 7)));
        // Lower-level promote attempt does not lower the snapshot.
        ep.try_promote(2, 2);
        assert_eq!(ep.for_search(), Some((1, 7)));
    }
}
