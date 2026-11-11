//! The logical material each commit adds to a tree, for consumers that read
//! the MVCC store from a checkpoint on.
//!
//! A consumer reading changes from seqno `s` keeps every version at or above
//! `s` from collection. What that costs it is the logical size of those
//! versions, before compression: key, value or operand, deletion and range
//! bounds. Nothing per version records that, and summing it from the versions
//! themselves would walk every key. So every write to a tree also puts one
//! record into the same tree, in the same lsm batch, under the
//! engine-reserved namespace: `0x00 "ax" be_u64(seqno)` = the logical bytes
//! that seqno wrote there. Landing in the batch with the data, it reaches the
//! memtable, a flush and a replay together with what it describes; a put at
//! the same seqno is idempotent, so a replay charges nothing twice.
//!
//! A seqno is one commit, so the first record at or after `s` is the oldest
//! work a consumer at `s` has not seen, and the records from `s` on sum to
//! the material it requires.

use lsm_tree::{AbstractTree, AnyTree, Guard, SeqNo};

use crate::engine::coverage;
use crate::engine::partition::Partition;
use crate::error::{StorageError, StorageResult};

/// The record key prefix.
const PREFIX: [u8; 3] = [0x00, b'a', b'x'];
/// Exclusive end of the record range.
const END: [u8; 3] = [0x00, b'a', b'y'];
const KEY_LEN: usize = PREFIX.len() + 8;

/// The record for `seqno`.
pub(crate) fn key(seqno: SeqNo) -> [u8; KEY_LEN] {
    let mut key = [0u8; KEY_LEN];
    key[..PREFIX.len()].copy_from_slice(&PREFIX);
    key[PREFIX.len()..].copy_from_slice(&seqno.to_be_bytes());
    key
}

/// Whether writes to `part` are source material a consumer reads. The
/// consensus partition holds this member's own log, and the registry holds
/// the consumers' own records: a consumer acknowledging must not owe itself
/// work.
pub(crate) fn accounted(part: Partition) -> bool {
    !matches!(part, Partition::Raft | Partition::Registry)
}

/// The logical bytes one write unit adds: its key and what it writes (a
/// value, an operand, nothing for a deletion), or a range deletion's bounds.
/// Engine-reserved keys are bookkeeping, not material.
pub(crate) fn unit_bytes(key: &[u8], written: &[u8]) -> u64 {
    if coverage::is_reserved(key) {
        return 0;
    }
    // Both lengths are of in-memory slices, far below u64::MAX together.
    (key.len() + written.len()) as u64
}

/// Sums the units of one tree's batch and puts the record for them.
#[derive(Debug, Default)]
pub(crate) struct Tally(u64);

impl Tally {
    /// Count one unit ([`unit_bytes`]).
    pub(crate) fn unit(&mut self, key: &[u8], written: &[u8]) {
        // A batch is bounded by memory, so its sum is far below u64::MAX.
        self.0 += unit_bytes(key, written);
    }

    /// Put the record for what was counted into `batch` at the batch's
    /// seqno, when `part` is accounted and anything was.
    pub(crate) fn record(self, part: Partition, seqno: SeqNo, batch: &mut lsm_tree::WriteBatch) {
        if self.0 > 0 && accounted(part) {
            batch.insert(key(seqno).as_slice(), encode(self.0).as_slice());
        }
    }
}

/// The records of a tree rebuilt from a copy's history, counted from the
/// versions actually reinstalled. The history comes in key order, so one
/// seqno's units arrive scattered: they are summed in a bounded map and
/// added to the tree's records a page at a time.
#[derive(Debug, Default)]
pub(crate) struct Reinstalled {
    pending: std::collections::BTreeMap<SeqNo, u64>,
}

/// Seqnos summed in memory before they are added to the tree.
const REINSTALL_PAGE: usize = 4096;

impl Reinstalled {
    /// Count `bytes` reinstalled at `seqno`.
    pub(crate) fn add(&mut self, seqno: SeqNo, bytes: u64) {
        if bytes > 0 {
            // A tree's versions at one seqno are bounded by one batch.
            *self.pending.entry(seqno).or_insert(0) += bytes;
        }
    }

    /// Whether a page is due ([`Self::spill`]).
    pub(crate) fn full(&self) -> bool {
        self.pending.len() >= REINSTALL_PAGE
    }

    /// Add what was counted to `tree`'s records. Each sum is written at
    /// `at`, a seqno above every earlier page's, so the newest sum is the one
    /// read whichever of them a flush separated.
    ///
    /// # Errors
    ///
    /// A record could not be read.
    pub(crate) fn spill(
        &mut self,
        part: Partition,
        tree: &AnyTree,
        at: SeqNo,
    ) -> StorageResult<()> {
        let pending = core::mem::take(&mut self.pending);
        if !accounted(part) {
            return Ok(());
        }
        for (seqno, bytes) in pending {
            let k = key(seqno);
            let held = match tree.get(k.as_slice(), SeqNo::MAX)? {
                Some(value) => decode(&k, &value)?.1,
                None => 0,
            };
            // Both are sums of one seqno's units in one tree.
            tree.insert(k.as_slice(), encode(held + bytes).as_slice(), at);
        }
        Ok(())
    }
}

/// Key and value bytes of one record.
pub(crate) const RECORD_BYTES: u64 = (KEY_LEN + 8) as u64;

/// The room the records of one apply of `units` writes take, bounded from
/// above: one record per tree it touches, and it touches at most one tree
/// per write and no more trees than there are.
pub(crate) fn record_allowance(units: usize) -> u64 {
    let trees = Partition::all().len();
    // At most a dozen trees: far below u64::MAX.
    crate::engine::footprint::footprint_of(RECORD_BYTES, 1) * units.min(trees) as u64
}

/// The material from a seqno on, over every accounted tree.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Material {
    /// The first seqno with material, if any.
    pub first: Option<SeqNo>,
    /// Logical bytes of the material, each seqno's once.
    pub bytes: u64,
}

/// The material from a checkpoint on as one consistent observation, or why
/// there is none.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Observation {
    /// Read at one complete cut: every commit at or below `through`, and no
    /// write of a commit above it, is counted, so neither the age nor the
    /// bytes describe a commit applied in part.
    Observed {
        /// The complete cut the material was read at.
        through: SeqNo,
        /// The source generation it was read in: unchanged from before the
        /// read to after it. A decision taken on this observation is valid
        /// while the store's generation is still this one.
        generation: u64,
        /// The material from the checkpoint through the cut.
        material: Material,
    },
    /// No commit is known to be complete yet: nothing can be claimed.
    NoCut,
    /// A tree is being rebuilt from a checkpoint, so its records are not
    /// all there: the evidence is incomplete, not empty.
    Rebuilding,
    /// The store no longer serves reads from the checkpoint: records it
    /// needs may already be retired.
    Lost,
}

impl Material {
    fn add(&mut self, seqno: SeqNo, bytes: u64) -> StorageResult<()> {
        self.first = Some(self.first.map_or(seqno, |f| f.min(seqno)));
        self.bytes = self.bytes.checked_add(bytes).ok_or_else(|| {
            StorageError::Serialization("material bytes overflow u64".to_string())
        })?;
        Ok(())
    }
}

/// Add the records of `tree` in `[from, to)` visible at `snapshot` to
/// `material`.
pub(crate) fn read_into(
    tree: &AnyTree,
    from: SeqNo,
    to: Option<SeqNo>,
    snapshot: SeqNo,
    material: &mut Material,
) -> StorageResult<()> {
    let start = key(from);
    let end: Vec<u8> = match to {
        Some(to) => key(to).to_vec(),
        None => END.to_vec(),
    };
    if start.as_slice() >= end.as_slice() {
        return Ok(());
    }
    for guard in tree.range(start.as_slice()..end.as_slice(), snapshot, None) {
        let (k, v) = guard.into_inner()?;
        let (seqno, bytes) = decode(&k, &v)?;
        material.add(seqno, bytes)?;
    }
    Ok(())
}

/// How far the readable floor moves before the records below it are dropped
/// (seqnos; with an oracle, microseconds: about a minute). Each retirement
/// is one range deletion per tree.
pub(crate) const RETIRE_WINDOW: SeqNo = 1 << 26;

/// Drop `tree`'s records below `floor`, at seqno `at`, above every record
/// they cover. A tree with none there takes no write: a deletion would make
/// it flush a table for nothing. Returns whether it wrote.
///
/// # Errors
///
/// A record could not be read or is malformed.
pub(crate) fn retire_below(tree: &AnyTree, floor: SeqNo, at: SeqNo) -> StorageResult<bool> {
    if first_from(tree, 0, SeqNo::MAX)?.is_none_or(|first| first >= floor) {
        return Ok(false);
    }
    let end = key(floor);
    tree.remove_range(PREFIX.as_slice(), end.as_slice(), at);
    Ok(true)
}

/// The first seqno at or after `from` with a record in `tree` visible at
/// `snapshot`.
pub(crate) fn first_from(
    tree: &AnyTree,
    from: SeqNo,
    snapshot: SeqNo,
) -> StorageResult<Option<SeqNo>> {
    let start = key(from);
    match tree
        .range(start.as_slice()..END.as_slice(), snapshot, None)
        .next()
    {
        Some(guard) => {
            let (k, v) = guard.into_inner()?;
            Ok(Some(decode(&k, &v)?.0))
        }
        None => Ok(None),
    }
}

/// Encode the bytes a record holds.
pub(crate) fn encode(bytes: u64) -> [u8; 8] {
    bytes.to_le_bytes()
}

fn decode(key: &[u8], value: &[u8]) -> StorageResult<(SeqNo, u64)> {
    let seqno = key
        .strip_prefix(PREFIX.as_slice())
        .and_then(|s| <[u8; 8]>::try_from(s).ok())
        .map(u64::from_be_bytes);
    let bytes = <[u8; 8]>::try_from(value).ok().map(u64::from_le_bytes);
    match (seqno, bytes) {
        (Some(seqno), Some(bytes)) => Ok((seqno, bytes)),
        _ => Err(StorageError::Serialization(format!(
            "material record {key:02x?} holds {} bytes",
            value.len()
        ))),
    }
}

#[cfg(test)]
mod tests;
