//! Run-length coalescing of a delete set into point + range deletes (G096).
//!
//! Bulk deletes (drop a shard's node keyspace, drop an edge type, drop an index,
//! "delete all relationships between these nodes") often contain dense runs of
//! consecutive keys. Recording each as its own point tombstone bloats the oplog
//! / Raft payload / PITR log and the LSM tombstone count. This module folds the
//! sorted delete set into a mix of point deletes and **range deletes**, where a
//! range covers a maximal run of bytewise-consecutive, equal-length keys.
//!
//! ## Safety — only dense contiguous runs, never across a gap
//!
//! A [`CoalescedDelete::Range`] is emitted ONLY for a run where each key is the
//! immediate byte successor of the previous one AND all keys share the same
//! length. Such a run has no other equal-length key between its members, so the
//! range `[start, end)` covers exactly the run — never a surviving key in a gap.
//! Singletons and short runs stay point deletes.
//!
//! **Caller precondition:** within one partition, the supplied keys must come
//! from a single fixed-width key family (e.g. all `adj:‹type›:out:‹src be64›`,
//! or all `node:‹shard›:‹id be64›`). CoordiNode key families are
//! prefix-disjoint and fixed-width per family, so no key of a *different* length
//! sharing a run member's prefix exists in storage — which is what makes a
//! same-length consecutive run a safe range. Mixing families in one partition's
//! input could place a longer key (e.g. `a‖0x00`) lexicographically inside a
//! run; do not do that.

/// One delete after coalescing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoalescedDelete {
    /// Delete a single key.
    Point {
        /// Partition discriminant.
        partition: u8,
        /// Key to delete.
        key: Vec<u8>,
    },
    /// Delete every key in the half-open range `[start, end)`.
    Range {
        /// Partition discriminant.
        partition: u8,
        /// Inclusive start (first key of the run).
        start: Vec<u8>,
        /// Exclusive end (immediate successor of the last key of the run).
        end: Vec<u8>,
    },
}

/// The immediate byte successor of an equal-length key: `key` interpreted as a
/// big-endian integer plus one, keeping the same length. `None` when every byte
/// is `0xFF` (the successor would need an extra byte — the run cannot be a
/// same-length range and is broken at that point).
fn same_len_successor(key: &[u8]) -> Option<Vec<u8>> {
    let mut out = key.to_vec();
    for byte in out.iter_mut().rev() {
        if *byte == 0xFF {
            *byte = 0x00;
        } else {
            *byte += 1;
            return Some(out);
        }
    }
    None
}

/// Are `b` the immediate same-length successor of `a`? (Equal length and
/// `successor(a) == b`.)
fn is_adjacent(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && same_len_successor(a).is_some_and(|s| s == b)
}

/// Run-length coalesce a delete set. Input keys are grouped by partition and
/// sorted; within a partition, maximal runs of `>= min_run` bytewise-consecutive
/// equal-length keys become a [`CoalescedDelete::Range`], everything else a
/// [`CoalescedDelete::Point`]. Output is ordered by `(partition, key)`.
///
/// `min_run` is the threshold below which a run stays point deletes (a 2-3 key
/// "run" is not worth a range tombstone's read cost). `min_run <= 1` is treated
/// as 2 (a single key is never a range).
///
/// See the module docs for the caller precondition (one fixed-width key family
/// per partition).
pub fn coalesce_deletes(mut deletes: Vec<(u8, Vec<u8>)>, min_run: usize) -> Vec<CoalescedDelete> {
    let min_run = min_run.max(2);
    deletes.sort();
    deletes.dedup();

    let mut out = Vec::new();
    let mut i = 0;
    while i < deletes.len() {
        let (partition, _) = (deletes[i].0, &deletes[i].1);
        // Extend the run while same partition + adjacent to the previous key.
        let mut j = i + 1;
        while j < deletes.len()
            && deletes[j].0 == partition
            && is_adjacent(&deletes[j - 1].1, &deletes[j].1)
        {
            j += 1;
        }
        let run_len = j - i;
        if run_len >= min_run {
            let start = deletes[i].1.clone();
            // end = immediate successor of the run's last key (run is same-len
            // consecutive, so the last key has a same-length successor unless it
            // is all-0xFF — but then it could not have been reached by adjacency
            // as a non-final member, and as the final member we still need an
            // end; fall back to point deletes for that degenerate tail).
            match same_len_successor(&deletes[j - 1].1) {
                Some(end) => out.push(CoalescedDelete::Range {
                    partition,
                    start,
                    end,
                }),
                None => {
                    for d in &deletes[i..j] {
                        out.push(CoalescedDelete::Point {
                            partition,
                            key: d.1.clone(),
                        });
                    }
                }
            }
        } else {
            for d in &deletes[i..j] {
                out.push(CoalescedDelete::Point {
                    partition,
                    key: d.1.clone(),
                });
            }
        }
        i = j;
    }
    out
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests;
