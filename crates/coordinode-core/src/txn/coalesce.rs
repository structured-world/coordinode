//! Run-length coalescing of delete mutations into range deletes (G096).
//!
//! A bulk delete ("delete all relationships between these nodes", drop an edge
//! type / index / shard keyspace) produces many [`Mutation::Delete`] of
//! consecutive keys. Recording each as its own point tombstone bloats the Raft
//! payload / oplog / PITR log and the LSM tombstone count. This pass folds a
//! proposal's deletes into a mix of point deletes and [`Mutation::RemoveRange`],
//! where a range covers a maximal run of bytewise-consecutive, equal-length keys
//! in one partition.
//!
//! Runs ONLY within a dense contiguous prefix — a range is emitted solely for a
//! run where each key is the immediate byte successor of the previous one and
//! all keys share the same length, so it never spans a gap holding a surviving
//! key. Each CoordiNode partition holds a single fixed-width key family, so a
//! per-partition run has no intermediate key of a different length: the range is
//! an exact encoding of "delete exactly these keys". Non-delete mutations and
//! short runs pass through unchanged, preserving order.

use super::proposal::Mutation;

/// Default minimum run length to coalesce into a range. Shorter runs stay point
/// deletes (a 2-3 key range tombstone is not worth its read cost).
pub const DEFAULT_MIN_RUN: usize = 4;

/// The immediate same-length byte successor of `key` (big-endian + 1, same
/// length). `None` when every byte is `0xFF` (successor needs an extra byte).
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

/// Is `b` the immediate same-length successor of `a`?
fn is_adjacent(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && same_len_successor(a).is_some_and(|s| s == b)
}

/// Coalesce a proposal's mutations: dense runs of point deletes within a
/// partition become [`Mutation::RemoveRange`]; everything else is preserved in
/// order. `min_run <= 1` is treated as 2 (a single key is never a range).
///
/// Non-delete mutations (`Put` / `Merge` / existing `RemoveRange`) act as run
/// boundaries and pass through untouched, so deletes are only ever coalesced
/// among themselves and the proposal's apply order is preserved.
pub fn coalesce_delete_mutations(mutations: Vec<Mutation>, min_run: usize) -> Vec<Mutation> {
    let min_run = min_run.max(2);
    let mut out = Vec::with_capacity(mutations.len());
    let mut i = 0;
    while i < mutations.len() {
        // Only a Delete can start a run.
        let Mutation::Delete { partition, key } = &mutations[i] else {
            out.push(mutations[i].clone());
            i += 1;
            continue;
        };
        let part = *partition;

        // Extend while the next mutation is a Delete in the same partition whose
        // key is the immediate successor of the previous key.
        let mut j = i + 1;
        let mut prev_key = key.as_slice();
        while j < mutations.len() {
            let Mutation::Delete {
                partition: np,
                key: nk,
            } = &mutations[j]
            else {
                break;
            };
            if *np != part || !is_adjacent(prev_key, nk) {
                break;
            }
            prev_key = nk;
            j += 1;
        }

        let run_len = j - i;
        let last_key = match &mutations[j - 1] {
            Mutation::Delete { key, .. } => key.clone(),
            _ => unreachable!("run members are all Delete"),
        };
        if run_len >= min_run {
            match same_len_successor(&last_key) {
                Some(end) => {
                    let start = match &mutations[i] {
                        Mutation::Delete { key, .. } => key.clone(),
                        _ => unreachable!("run head is Delete"),
                    };
                    out.push(Mutation::RemoveRange {
                        partition: part,
                        start,
                        end,
                    });
                }
                // All-0xFF tail: no same-length end bound — keep as point deletes.
                None => out.extend(mutations[i..j].iter().cloned()),
            }
        } else {
            out.extend(mutations[i..j].iter().cloned());
        }
        i = j;
    }
    out
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests;
