//! A summary of per-key charges ordered by the key's newest change, read in
//! a bounded number of steps.
//!
//! A coalesced state-delta consumer at continuation `s` needs every key whose
//! newest change at the accounting cut is at or after `s`, each with the
//! charge of what reconstructs it there. Summing those keys one by one would
//! walk them all. The summary holds the charges in buckets of the newest
//! change instead, at [`LEVELS`] levels of [`FANOUT`] children each, under
//! the scope of the tree they describe in the accounting tree:
//! `head 0x00 "as" level be_u64(n >> (BITS * level))` = the bytes of every key
//! whose newest change `n` falls in that bucket. A sum from `s` reads one
//! contiguous run of at most `FANOUT - 1` buckets per level.
//!
//! Buckets are plain values, rewritten whole by the one writer that moves
//! charges, so a read never resolves a chain of operands.

use std::collections::BTreeMap;

use lsm_tree::{AbstractTree, Guard, SeqNo};

use super::{HEAD_LEN, Head, Scope};
use crate::error::{StorageError, StorageResult};

/// Bucket key prefix.
const PREFIX: [u8; 3] = [0x00, b'a', b's'];
/// Bits of the newest change each level consumes.
const BITS: u32 = 8;
/// Children per bucket.
const FANOUT: u64 = 1 << BITS;
/// Levels: together they cover a `u64` seqno.
const LEVELS: u8 = (u64::BITS / BITS) as u8;
const KEY_LEN: usize = PREFIX.len() + 1 + 8;

/// A bucket's key past the scope's head.
pub(crate) type BucketKey = [u8; KEY_LEN];

/// `bucket` under `head`.
fn scoped(head: Head, bucket: &BucketKey) -> [u8; HEAD_LEN + KEY_LEN] {
    let mut out = [0u8; HEAD_LEN + KEY_LEN];
    out[..HEAD_LEN].copy_from_slice(head.bytes());
    out[HEAD_LEN..].copy_from_slice(bucket);
    out
}
/// The buckets one charge's move touches at most: its old and its new
/// bucket at every level.
pub(crate) const MOVES_PER_SHIFT: usize = 2 * LEVELS as usize;

/// The bucket at `level` holding the newest change `n`, or of index `index`
/// when `n` is already shifted.
fn bucket(level: u8, index: u64) -> [u8; KEY_LEN] {
    let mut key = [0u8; KEY_LEN];
    key[..PREFIX.len()].copy_from_slice(&PREFIX);
    key[PREFIX.len()] = level;
    key[PREFIX.len() + 1..].copy_from_slice(&index.to_be_bytes());
    key
}

/// `n`'s bucket index at `level`.
fn index_at(level: u8, n: SeqNo) -> u64 {
    // level < LEVELS, so the shift is below 64.
    n >> (BITS * u32::from(level))
}

/// Charges moved between buckets, summed before they are written.
#[derive(Debug, Default)]
pub(crate) struct Moves(BTreeMap<BucketKey, i128>);

impl Moves {
    /// Move one key's charge: `from` its previous newest change and charge,
    /// `to` its new ones (`None`: none).
    pub(crate) fn shift(&mut self, from: Option<(SeqNo, u64)>, to: Option<(SeqNo, u64)>) {
        if from == to {
            return;
        }
        for level in 0..LEVELS {
            if let Some((n, bytes)) = from {
                *self.0.entry(bucket(level, index_at(level, n))).or_insert(0) -= i128::from(bytes);
            }
            if let Some((n, bytes)) = to {
                *self.0.entry(bucket(level, index_at(level, n))).or_insert(0) += i128::from(bytes);
            }
        }
    }

    /// The buckets moved so far.
    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    /// Add the moves to `scope`'s buckets into `batch`. The buckets are read
    /// as they stand: the caller is their one writer. Returns the key and
    /// value bytes written.
    ///
    /// # Errors
    ///
    /// A bucket could not be read, is malformed, or would go below zero or
    /// past `u64::MAX`: the charges it was built from are not what moved.
    pub(crate) fn write(
        self,
        scope: &Scope<'_>,
        batch: &mut lsm_tree::WriteBatch,
    ) -> StorageResult<u64> {
        // Bounded by the buckets one batch moves.
        let mut bytes = 0u64;
        for (bucket, delta) in self.0 {
            if delta == 0 {
                continue;
            }
            let key = scoped(scope.head, &bucket);
            let held = match scope.tree.get(key.as_slice(), SeqNo::MAX)? {
                Some(value) => decode(&key, &value)?,
                None => 0,
            };
            let next = i128::from(held)
                .checked_add(delta)
                .and_then(|n| u64::try_from(n).ok())
                .ok_or_else(|| {
                    StorageError::Serialization(format!(
                        "material summary bucket {key:02x?} holds {held}, moved by {delta}"
                    ))
                })?;
            bytes += (HEAD_LEN + KEY_LEN + 8) as u64;
            if next == 0 {
                batch.remove(key.as_slice());
            } else {
                batch.insert(key.as_slice(), next.to_le_bytes().as_slice());
            }
        }
        Ok(bytes)
    }
}

/// The bytes of every key whose newest change is at or after `s`, in
/// `scope`'s summary visible at `snapshot`.
///
/// # Errors
///
/// A bucket could not be read or is malformed.
pub(crate) fn charge_from(scope: &Scope<'_>, s: SeqNo, snapshot: SeqNo) -> StorageResult<u64> {
    let mut total = 0u64;
    let mut below = 0u64;
    let top = LEVELS - 1;
    // The top level's buckets together hold every charge.
    for (_, bytes) in run(scope, top, 0, FANOUT, snapshot)? {
        total = add(total, bytes)?;
    }
    for level in 0..LEVELS {
        // The siblings before `s`'s bucket under the same parent.
        let index = index_at(level, s);
        let first = index & !(FANOUT - 1);
        for (_, bytes) in run(scope, level, first, index, snapshot)? {
            below = add(below, bytes)?;
        }
    }
    total.checked_sub(below).ok_or_else(|| {
        StorageError::Serialization(format!(
            "material summary: {below} bytes below {s} of {total} in all"
        ))
    })
}

/// The buckets of `level` with index in `[first, end)`.
fn run(
    scope: &Scope<'_>,
    level: u8,
    first: u64,
    end: u64,
    snapshot: SeqNo,
) -> StorageResult<Vec<(u64, u64)>> {
    if first >= end {
        return Ok(Vec::new());
    }
    let start = scoped(scope.head, &bucket(level, first));
    let stop = scoped(scope.head, &bucket(level, end));
    let mut out = Vec::new();
    for guard in scope
        .tree
        .range(start.as_slice()..stop.as_slice(), snapshot, None)
    {
        let (k, v) = guard.into_inner()?;
        let index = k
            .get(HEAD_LEN + PREFIX.len() + 1..)
            .and_then(|s| <[u8; 8]>::try_from(s).ok())
            .map(u64::from_be_bytes)
            .ok_or_else(|| malformed(&k, &v))?;
        out.push((index, decode(&k, &v)?));
    }
    Ok(out)
}

fn add(sum: u64, bytes: u64) -> StorageResult<u64> {
    sum.checked_add(bytes)
        .ok_or_else(|| StorageError::Serialization("material summary overflows u64".to_string()))
}

fn decode(key: &[u8], value: &[u8]) -> StorageResult<u64> {
    if key.len() != HEAD_LEN + KEY_LEN {
        return Err(malformed(key, value));
    }
    <[u8; 8]>::try_from(value)
        .map(u64::from_le_bytes)
        .map_err(|_| malformed(key, value))
}

fn malformed(key: &[u8], value: &[u8]) -> StorageError {
    StorageError::Serialization(format!(
        "material summary bucket {key:02x?} holds {} bytes",
        value.len()
    ))
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests;
