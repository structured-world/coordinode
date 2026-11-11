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
//! work a consumer at `s` has not seen.
//!
//! Beside it, every unit of the batch gets one record of the unit index,
//! `0x00 "au" be_u64(seqno) be_u32(ordinal)` = its kind, encoded length and
//! key (and a range's end), no payload: what the fold ([`fold`]) reads, in
//! commit order, to keep each key's reconstruction closure. The ordinal is
//! the unit's place in its batch, so two equal operands at one seqno stay two
//! units and a replay of the batch rewrites the same records.

use lsm_tree::{AbstractTree, AnyTree, Guard, SeqNo};

use crate::engine::coverage;
use crate::engine::partition::Partition;
use crate::error::{StorageError, StorageResult};

pub(crate) mod fold;
pub(crate) mod summary;

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

/// What a unit does to its key's reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UnitKind {
    /// A value: the key's closure is this unit alone.
    Value,
    /// A deletion: evidence of the change and what a later operand starts
    /// from.
    Delete,
    /// An operand merged onto the closure so far.
    Operand,
    /// A range deletion of `[key, end)`.
    Range,
}

impl UnitKind {
    fn tag(self) -> u8 {
        match self {
            Self::Value => 0,
            Self::Delete => 1,
            Self::Operand => 2,
            Self::Range => 3,
        }
    }

    fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(Self::Value),
            1 => Some(Self::Delete),
            2 => Some(Self::Operand),
            3 => Some(Self::Range),
            _ => None,
        }
    }
}

/// One unit of a batch, borrowed from the write it describes.
#[derive(Debug, Clone, Copy)]
struct Counted<'a> {
    kind: UnitKind,
    key: &'a [u8],
    end: &'a [u8],
    bytes: u64,
}

/// Bytes a tree batch's tally holds per unit it counts.
pub(crate) const COUNTED_SLOT: u64 = core::mem::size_of::<Counted<'static>>() as u64;

/// Bytes of a unit record beyond the key it carries: its key and the kind,
/// length and key-length fields of its value ([`unit_value`]).
pub(crate) const UNIT_RECORD_FIXED: u64 = UNIT_KEY_LEN as u64 + 1 + 8 + 4;

/// Sums the units of one tree's batch and puts their records.
#[derive(Debug, Default)]
pub(crate) struct Tally<'a> {
    bytes: u64,
    units: Vec<Counted<'a>>,
}

impl<'a> Tally<'a> {
    /// A value written under `key`.
    pub(crate) fn put(&mut self, key: &'a [u8], value: &[u8]) {
        self.count(UnitKind::Value, key, &[], unit_bytes(key, value));
    }

    /// A deletion of `key`.
    pub(crate) fn delete(&mut self, key: &'a [u8]) {
        self.count(UnitKind::Delete, key, &[], unit_bytes(key, &[]));
    }

    /// An operand merged onto `key`.
    pub(crate) fn merge(&mut self, key: &'a [u8], operand: &[u8]) {
        self.count(UnitKind::Operand, key, &[], unit_bytes(key, operand));
    }

    /// A deletion of `[start, end)`.
    pub(crate) fn range(&mut self, start: &'a [u8], end: &'a [u8]) {
        self.count(UnitKind::Range, start, end, unit_bytes(start, end));
    }

    fn count(&mut self, kind: UnitKind, key: &'a [u8], end: &'a [u8], bytes: u64) {
        // Engine bookkeeping is no unit.
        if bytes == 0 {
            return;
        }
        // A batch is bounded by memory, so its sum is far below u64::MAX.
        self.bytes += bytes;
        self.units.push(Counted {
            kind,
            key,
            end,
            bytes,
        });
    }

    /// Put the record of what was counted and the record of every unit into
    /// `batch` at the batch's seqno, when `part` is accounted and anything
    /// was.
    pub(crate) fn record(self, part: Partition, seqno: SeqNo, batch: &mut lsm_tree::WriteBatch) {
        if self.bytes == 0 || !accounted(part) {
            return;
        }
        batch.insert(key(seqno).as_slice(), encode(self.bytes).as_slice());
        let mut value = Vec::new();
        for (ordinal, unit) in self.units.iter().enumerate() {
            unit_value(unit.kind, unit.bytes, unit.key, unit.end, &mut value);
            // A batch holds far fewer than u32::MAX units.
            batch.insert(unit_key(seqno, ordinal as u32).as_slice(), value.as_slice());
        }
    }
}

/// Encode a unit record's value into `out`.
pub(crate) fn unit_value(kind: UnitKind, bytes: u64, key: &[u8], end: &[u8], out: &mut Vec<u8>) {
    out.clear();
    out.push(kind.tag());
    out.extend_from_slice(&bytes.to_le_bytes());
    // Keys are bounded far below u32::MAX by the store's key limit.
    out.extend_from_slice(&(key.len() as u32).to_le_bytes());
    out.extend_from_slice(key);
    out.extend_from_slice(end);
}

/// The ordinal after the last unit record `tree` holds at `seqno`.
fn next_ordinal(tree: &AnyTree, seqno: SeqNo) -> StorageResult<u32> {
    let start = unit_key(seqno, 0);
    let end: Vec<u8> = match seqno.checked_add(1) {
        Some(next) => unit_key(next, 0).to_vec(),
        None => [0x00, b'a', b'v'].to_vec(),
    };
    match tree
        .range(start.as_slice()..end.as_slice(), SeqNo::MAX, None)
        .next_back()
    {
        Some(guard) => {
            let (k, v) = guard.into_inner()?;
            decode_unit(&k, &v)?.ordinal.checked_add(1).ok_or_else(|| {
                StorageError::Serialization(format!("material units at {seqno} exhaust u32"))
            })
        }
        None => Ok(0),
    }
}

/// A unit put back without a batch of its own: kind, key, range end (empty
/// for a point) and encoded length.
pub(crate) type OwnedUnit = (UnitKind, Vec<u8>, Vec<u8>, u64);

/// Write unit records for `units` at `seqno`, after the ones `tree` already
/// holds there, written at `at`: units a rebuild or an installation puts
/// back without a batch of their own.
///
/// # Errors
///
/// A record could not be read.
pub(crate) fn add_units(
    tree: &AnyTree,
    seqno: SeqNo,
    units: &[OwnedUnit],
    at: SeqNo,
) -> StorageResult<()> {
    let mut ordinal = next_ordinal(tree, seqno)?;
    let mut value = Vec::new();
    for (kind, key, end, bytes) in units {
        unit_value(*kind, *bytes, key, end, &mut value);
        tree.insert(unit_key(seqno, ordinal).as_slice(), value.as_slice(), at);
        ordinal = ordinal.checked_add(1).ok_or_else(|| {
            StorageError::Serialization(format!("material units at {seqno} exhaust u32"))
        })?;
    }
    Ok(())
}

/// The unit index record prefix.
const UNIT_PREFIX: [u8; 3] = [0x00, b'a', b'u'];
const UNIT_KEY_LEN: usize = UNIT_PREFIX.len() + 8 + 4;
/// A unit record's value past its key and end: kind, length, key length.
const UNIT_HEADER: usize = 1 + 8 + 4;

/// The unit index record of the `ordinal`-th unit at `seqno`.
pub(crate) fn unit_key(seqno: SeqNo, ordinal: u32) -> [u8; UNIT_KEY_LEN] {
    let mut out = [0u8; UNIT_KEY_LEN];
    out[..UNIT_PREFIX.len()].copy_from_slice(&UNIT_PREFIX);
    out[UNIT_PREFIX.len()..UNIT_PREFIX.len() + 8].copy_from_slice(&seqno.to_be_bytes());
    out[UNIT_PREFIX.len() + 8..].copy_from_slice(&ordinal.to_be_bytes());
    out
}

/// One unit read back from the unit index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Unit {
    pub(crate) seqno: SeqNo,
    pub(crate) ordinal: u32,
    pub(crate) kind: UnitKind,
    pub(crate) bytes: u64,
    pub(crate) key: Vec<u8>,
    /// A range's exclusive end; empty for a point unit.
    pub(crate) end: Vec<u8>,
}

fn decode_unit(key: &[u8], value: &[u8]) -> StorageResult<Unit> {
    let bad = || {
        StorageError::Serialization(format!(
            "material unit record {key:02x?} holds {} bytes",
            value.len()
        ))
    };
    let id = key.strip_prefix(UNIT_PREFIX.as_slice()).ok_or_else(bad)?;
    let seqno = id
        .get(..8)
        .and_then(|s| <[u8; 8]>::try_from(s).ok())
        .map(u64::from_be_bytes)
        .ok_or_else(bad)?;
    let ordinal = id
        .get(8..12)
        .and_then(|s| <[u8; 4]>::try_from(s).ok())
        .map(u32::from_be_bytes)
        .ok_or_else(bad)?;
    let kind = value
        .first()
        .copied()
        .and_then(UnitKind::from_tag)
        .ok_or_else(bad)?;
    let bytes = value
        .get(1..9)
        .and_then(|s| <[u8; 8]>::try_from(s).ok())
        .map(u64::from_le_bytes)
        .ok_or_else(bad)?;
    let key_len = value
        .get(9..13)
        .and_then(|s| <[u8; 4]>::try_from(s).ok())
        .map(|b| u32::from_le_bytes(b) as usize)
        .ok_or_else(bad)?;
    let unit_key = value
        .get(UNIT_HEADER..UNIT_HEADER + key_len)
        .ok_or_else(bad)?;
    let end = &value[UNIT_HEADER + key_len..];
    if kind != UnitKind::Range && !end.is_empty() {
        return Err(bad());
    }
    Ok(Unit {
        seqno,
        ordinal,
        kind,
        bytes,
        key: unit_key.to_vec(),
        end: end.to_vec(),
    })
}

/// Up to `limit` units of `tree` from `(seqno, ordinal)` on, below `to`
/// (exclusive), visible at `snapshot`, in commit order. With `held`, each
/// unit's copy is reserved there before it is made, and the units stop at
/// the first that does not fit; the flag says they stopped so.
///
/// # Errors
///
/// A record could not be read or is malformed.
pub(crate) fn units_from(
    tree: &AnyTree,
    from: (SeqNo, u32),
    to: SeqNo,
    snapshot: SeqNo,
    limit: usize,
    mut held: Option<&mut coordinode_core::budget::MemoryCharge<'_>>,
) -> StorageResult<(Vec<Unit>, bool)> {
    let start = unit_key(from.0, from.1);
    let end = unit_key(to, 0);
    if start.as_slice() >= end.as_slice() {
        return Ok((Vec::new(), false));
    }
    let mut out = Vec::new();
    for guard in tree.range(start.as_slice()..end.as_slice(), snapshot, None) {
        if out.len() >= limit {
            break;
        }
        let (k, v) = guard.into_inner()?;
        // The unit's key and range end, and its place in the list.
        let copy = (v.len() + core::mem::size_of::<Unit>()) as u64;
        if let Some(held) = held.as_deref_mut()
            && held.grow(copy).is_err()
        {
            return Ok((out, true));
        }
        out.push(decode_unit(&k, &v)?);
    }
    Ok((out, false))
}

/// The bytes of the range deletion [`retire_units_below`] writes: its bounds.
pub(crate) const RETIRE_RECORD_BYTES: u64 = (UNIT_PREFIX.len() + UNIT_KEY_LEN) as u64;

/// Drop `tree`'s unit records below `seqno`, at seqno `at`.
pub(crate) fn retire_units_below(tree: &AnyTree, seqno: SeqNo, at: SeqNo) {
    let end = unit_key(seqno, 0);
    tree.remove_range(UNIT_PREFIX.as_slice(), end.as_slice(), at);
}

/// The records of a tree rebuilt from a copy's history, counted from the
/// versions actually reinstalled. The history comes in key order, so one
/// seqno's units arrive scattered: they are summed in a bounded map and
/// added to the tree's records a page at a time.
#[derive(Debug, Default)]
pub(crate) struct Reinstalled {
    pending: std::collections::BTreeMap<SeqNo, u64>,
    /// The units reinstalled, by seqno, for the unit index.
    units: std::collections::BTreeMap<SeqNo, Vec<OwnedUnit>>,
    /// Units held in `units`.
    held: usize,
    /// The bytes of the keys and range ends held in `units`.
    held_bytes: usize,
}

/// Seqnos (and units) summed in memory before they are added to the tree.
const REINSTALL_PAGE: usize = 4096;
/// Key bytes held in memory before they are added to the tree.
const REINSTALL_BYTES: usize = 8 << 20;

impl Reinstalled {
    /// Count a unit of `kind` of `key` (and a range's `end`) reinstalled at
    /// `seqno`.
    pub(crate) fn add(&mut self, seqno: SeqNo, kind: UnitKind, key: &[u8], end: &[u8], bytes: u64) {
        if bytes > 0 {
            // A tree's versions at one seqno are bounded by one batch.
            *self.pending.entry(seqno).or_insert(0) += bytes;
            self.units
                .entry(seqno)
                .or_default()
                .push((kind, key.to_vec(), end.to_vec(), bytes));
            self.held += 1;
            self.held_bytes += key.len() + end.len();
        }
    }

    /// Whether a page is due ([`Self::spill`]): one more unit holds at most
    /// one key beyond these bounds.
    pub(crate) fn full(&self) -> bool {
        self.pending.len() >= REINSTALL_PAGE
            || self.held >= REINSTALL_PAGE
            || self.held_bytes >= REINSTALL_BYTES
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
        let units = core::mem::take(&mut self.units);
        self.held = 0;
        self.held_bytes = 0;
        if !accounted(part) {
            return Ok(());
        }
        for (seqno, units) in &units {
            add_units(tree, *seqno, units, at)?;
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
/// above: one record per tree it touches (it touches at most one tree per
/// write and no more trees than there are), and one unit record per unit,
/// `keys` being the bytes of the units' keys and range ends together.
pub(crate) fn record_allowance(units: usize, keys: u64) -> u64 {
    let trees = Partition::all().len();
    // At most a dozen trees, and units bounded by the apply's memory: far
    // below u64::MAX.
    crate::engine::footprint::footprint_of(RECORD_BYTES, 1) * units.min(trees) as u64
        + crate::engine::footprint::footprint_of(
            ((UNIT_KEY_LEN + UNIT_HEADER) * units) as u64 + keys,
            units,
        )
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
