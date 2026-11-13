//! The logical material each commit adds to a tree, for consumers that read
//! the MVCC store from a checkpoint on.
//!
//! A consumer reading changes from seqno `s` keeps every version at or above
//! `s` from collection. What that costs it is the logical size of those
//! versions, before compression: key, value or operand, deletion and range
//! bounds. Nothing per version records that, and summing it from the versions
//! themselves would walk every key. So every write to an accounted tree also
//! puts records into the accounting tree ([`Partition::Accounting`]), at the
//! write's seqno, under the scope of the tree it describes ([`Head`]: the
//! source's tag and the incarnation of its accounting):
//! `head "ax" be_u64(seqno)` = the logical bytes that seqno wrote there. A put
//! at the same seqno is idempotent, so a replay charges nothing twice.
//!
//! The records live apart from the data they describe: commit-ordered keys
//! in a data tree would give every table of it a key range from the reserved
//! namespace to its user keys, overlapping every table below it and turning
//! each compaction of it into a merge. A common seqno is not atomicity across
//! two trees, so a commit's accounting is one batch of the accounting tree,
//! with its own coverage marker on a journaled path, applied after the data;
//! replay derives it from the whole entry, whichever trees already hold
//! their part.
//!
//! A seqno is one commit, so the first record at or after `s` is the oldest
//! work a consumer at `s` has not seen.
//!
//! Beside it, every unit gets one record of the unit index,
//! `head "au" be_u64(seqno) be_u32(ordinal)` = its kind, encoded length and
//! key (and a range's end), no payload: what the material fold reads, in
//! commit order, to keep each key's reconstruction closure. The ordinal is
//! the unit's place in its source tree's part of the batch, so two equal
//! operands at one seqno stay two units and a replay of the batch rewrites
//! the same records.

use lsm_tree::{AbstractTree, AnyTree, Guard, SeqNo};

use crate::engine::coverage;
use crate::engine::partition::Partition;
use crate::error::{StorageError, StorageResult};

pub(crate) mod fold;
pub(crate) mod summary;

/// Bytes of the head every scoped key of the accounting tree starts with.
pub(crate) const HEAD_LEN: usize = 5;

/// The head of one source scope's keys in the accounting tree: the source
/// partition's tag (never `0x00`, the coverage records' namespace, nor the
/// `0xFE`/`0xFF` of the tree's own records), then the incarnation of its
/// accounting, big-endian. A clear of the source renews the incarnation, so
/// what is rebuilt never meets a record of the accounting it replaced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Head([u8; HEAD_LEN]);

impl Head {
    fn of(part: Partition, incarnation: u32) -> Self {
        let mut head = [0u8; HEAD_LEN];
        head[0] = tag(part);
        head[1..].copy_from_slice(&incarnation.to_be_bytes());
        Self(head)
    }

    /// The head's bytes.
    pub(crate) fn bytes(&self) -> &[u8; HEAD_LEN] {
        &self.0
    }

    /// `rest` under this head.
    pub(crate) fn key(&self, rest: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEAD_LEN + rest.len());
        out.extend_from_slice(&self.0);
        out.extend_from_slice(rest);
        out
    }

    /// The exclusive end of every key under this head: the next
    /// incarnation's head, or past the tag at the last one.
    fn end(&self) -> Vec<u8> {
        let incarnation = u32::from_be_bytes([self.0[1], self.0[2], self.0[3], self.0[4]]);
        match incarnation.checked_add(1) {
            Some(next) => {
                let mut out = self.0;
                out[1..].copy_from_slice(&next.to_be_bytes());
                out.to_vec()
            }
            // Tags stop below 0xFE.
            None => vec![self.0[0] + 1],
        }
    }
}

/// The tag of `part`'s scope: its position from one on.
fn tag(part: Partition) -> u8 {
    // At most a dozen partitions: far below 0xFE.
    part.position() as u8 + 1
}

/// The first byte of the record of a source's accounting incarnation,
/// `0xFF tag`.
const INCARNATION: u8 = 0xFF;

/// The incarnation of every source's accounting, as the accounting tree
/// records it (zero for a source never cleared).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Heads([u32; Partition::COUNT]);

impl Heads {
    /// The incarnations `acct` records.
    ///
    /// # Errors
    ///
    /// A record could not be read or is malformed.
    pub(crate) fn read(acct: &AnyTree) -> StorageResult<Self> {
        let mut heads = Self::default();
        for &part in Partition::all() {
            let key = [INCARNATION, tag(part)];
            if let Some(value) = acct.get(key.as_slice(), SeqNo::MAX)? {
                let bytes = <[u8; 4]>::try_from(value.as_ref()).map_err(|_| {
                    StorageError::Serialization(format!(
                        "accounting incarnation of {} holds {} bytes",
                        part.name(),
                        value.len()
                    ))
                })?;
                heads.0[part.position()] = u32::from_be_bytes(bytes);
            }
        }
        Ok(heads)
    }

    /// `part`'s head.
    pub(crate) fn head(&self, part: Partition) -> Head {
        Head::of(part, self.0[part.position()])
    }

    /// Renew `part`'s accounting in `acct` at `seqno`: the next incarnation
    /// is recorded and every record of the one before is deleted, so what
    /// follows is written, and read, under the new head.
    ///
    /// # Errors
    ///
    /// The incarnations are exhausted.
    pub(crate) fn renew(
        &mut self,
        acct: &AnyTree,
        part: Partition,
        seqno: SeqNo,
    ) -> StorageResult<()> {
        let old = self.head(part);
        let next = self.0[part.position()].checked_add(1).ok_or_else(|| {
            StorageError::Serialization(format!(
                "accounting incarnations of {} exhausted",
                part.name()
            ))
        })?;
        acct.insert(
            [INCARNATION, tag(part)].as_slice(),
            next.to_be_bytes().as_slice(),
            seqno,
        );
        acct.remove_range(old.bytes().as_slice(), old.end().as_slice(), seqno);
        self.0[part.position()] = next;
        Ok(())
    }
}

/// One source's scope in the accounting tree.
#[derive(Clone, Copy)]
pub(crate) struct Scope<'t> {
    /// The accounting tree.
    pub(crate) tree: &'t AnyTree,
    /// The source's head.
    pub(crate) head: Head,
}

impl core::fmt::Debug for Scope<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Scope").field("head", &self.head).finish()
    }
}

impl<'t> Scope<'t> {
    /// `part`'s scope in `acct` under `heads`.
    pub(crate) fn new(acct: &'t AnyTree, heads: &Heads, part: Partition) -> Self {
        Self {
            tree: acct,
            head: heads.head(part),
        }
    }

    /// The runs a read of the accounting tree goes through as it stands.
    pub(crate) fn runs(&self) -> usize {
        self.tree
            .current_version()
            .iter_levels()
            .map(|level| level.run_count())
            .sum()
    }
}

/// Whether writes to `part` are source material a consumer reads. The
/// consensus partition holds this member's own log, the registry holds the
/// consumers' own records (a consumer acknowledging must not owe itself
/// work), and the accounting holds the records describing the rest.
pub(crate) fn accounted(part: Partition) -> bool {
    !matches!(
        part,
        Partition::Raft | Partition::Registry | Partition::Accounting
    )
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

/// Bytes a tally holds per unit it counts.
pub(crate) const COUNTED_SLOT: u64 = core::mem::size_of::<Counted<'static>>() as u64;

/// Bytes of a unit record beyond the key it carries: its key and the kind,
/// length and key-length fields of its value ([`unit_value`]).
pub(crate) const UNIT_RECORD_FIXED: u64 = UNIT_KEY_LEN as u64 + 1 + 8 + 4;

/// Sums the units of one source tree's part of a batch and puts their
/// records.
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

    /// Whether this tally puts no record for source `part`.
    pub(crate) fn writes_nothing(&self, part: Partition) -> bool {
        self.bytes == 0 || !accounted(part)
    }

    /// Put the record of what was counted in source `part` and the record of
    /// every unit into `batch`, a batch of the accounting tree applied at
    /// `seqno`, under `part`'s head in `heads`, when `part` is accounted and
    /// anything was counted.
    pub(crate) fn record(
        &self,
        part: Partition,
        heads: &Heads,
        seqno: SeqNo,
        batch: &mut lsm_tree::WriteBatch,
    ) {
        if self.writes_nothing(part) {
            return;
        }
        let head = heads.head(part);
        batch.insert(key(head, seqno).as_slice(), encode(self.bytes).as_slice());
        let mut value = Vec::new();
        for (ordinal, unit) in self.units.iter().enumerate() {
            unit_value(unit.kind, unit.bytes, unit.key, unit.end, &mut value);
            // A batch holds far fewer than u32::MAX units.
            batch.insert(
                unit_key(head, seqno, ordinal as u32).as_slice(),
                value.as_slice(),
            );
        }
    }
}

/// The accounting batch of one commit at `seqno`: every source's tally under
/// `heads`, and `marker` last when the commit is journaled. `None` when no
/// tally writes anything: the commit touches the accounting tree nowhere.
pub(crate) fn batch_of<'a>(
    tallies: impl IntoIterator<Item = (Partition, &'a Tally<'a>)>,
    heads: &Heads,
    seqno: SeqNo,
    marker: Option<&[u8]>,
) -> Option<lsm_tree::WriteBatch> {
    let mut batch: Option<lsm_tree::WriteBatch> = None;
    for (part, tally) in tallies {
        if tally.writes_nothing(part) {
            continue;
        }
        let batch =
            batch.get_or_insert_with(|| lsm_tree::WriteBatch::with_capacity(tally.units.len() + 2));
        tally.record(part, heads, seqno, batch);
    }
    if let (Some(batch), Some(marker)) = (batch.as_mut(), marker) {
        batch.insert(marker, &[][..]);
    }
    batch
}

/// Where a commit's accounting comes from.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Accounting<'a> {
    /// The writes the commit applies.
    OwnWrites,
    /// Nothing: the accounting tree already holds the commit's.
    Skip,
    /// The writes the commit applies and these resolved mutations of the
    /// same unit, whose data trees already hold them: a replay applying only
    /// the data a tree lacks still accounts for all of the unit. A source is
    /// either applied or held whole, so no source takes units from both.
    OwnWritesAnd(&'a [coordinode_core::txn::proposal::Mutation]),
}

/// The tallies of `mutations` (resolved, decided), one per source partition,
/// each in the order the mutations name it: the order its live apply counts
/// them in, so the ordinals match.
pub(crate) fn tally_mutations(
    mutations: &[coordinode_core::txn::proposal::Mutation],
) -> Vec<(Partition, Tally<'_>)> {
    use coordinode_core::txn::proposal::Mutation;
    let mut tallies: Vec<(Partition, Tally<'_>)> = Vec::new();
    for mutation in mutations {
        let partition = match mutation {
            Mutation::Put { partition, .. }
            | Mutation::Delete { partition, .. }
            | Mutation::Merge { partition, .. }
            | Mutation::RemoveRange { partition, .. } => Partition::from(*partition),
            // Decided and resolved before an apply: nothing of the kind is
            // written.
            Mutation::Command(_) | Mutation::Operation(_) | Mutation::Derive(_) => continue,
        };
        if !accounted(partition) {
            continue;
        }
        let at = match tallies.iter().position(|(p, _)| *p == partition) {
            Some(at) => at,
            None => {
                tallies.push((partition, Tally::default()));
                tallies.len() - 1
            }
        };
        let tally = &mut tallies[at].1;
        match mutation {
            Mutation::Put { key, value, .. } => tally.put(key, value),
            Mutation::Delete { key, .. } => tally.delete(key),
            Mutation::Merge { key, operand, .. } => tally.merge(key, operand),
            Mutation::RemoveRange { start, end, .. } => {
                tally.range(coverage::clamp_user_start(start), end);
            }
            Mutation::Command(_) | Mutation::Operation(_) | Mutation::Derive(_) => {}
        }
    }
    tallies
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

/// The ordinal after the last unit record `scope` holds at `seqno`.
fn next_ordinal(scope: &Scope<'_>, seqno: SeqNo) -> StorageResult<u32> {
    let start = unit_key(scope.head, seqno, 0);
    let end: Vec<u8> = match seqno.checked_add(1) {
        Some(next) => unit_key(scope.head, next, 0).to_vec(),
        None => scope.head.key(&[0x00, b'a', b'v']),
    };
    match scope
        .tree
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

/// Write unit records for `units` at `seqno` into `scope`, after the ones it
/// already holds there, written at `at`: units a rebuild or an installation
/// puts back without a batch of their own.
///
/// # Errors
///
/// A record could not be read.
pub(crate) fn add_units(
    scope: &Scope<'_>,
    seqno: SeqNo,
    units: &[OwnedUnit],
    at: SeqNo,
) -> StorageResult<()> {
    let mut ordinal = next_ordinal(scope, seqno)?;
    let mut value = Vec::new();
    for (kind, key, end, bytes) in units {
        unit_value(*kind, *bytes, key, end, &mut value);
        scope.tree.insert(
            unit_key(scope.head, seqno, ordinal).as_slice(),
            value.as_slice(),
            at,
        );
        ordinal = ordinal.checked_add(1).ok_or_else(|| {
            StorageError::Serialization(format!("material units at {seqno} exhaust u32"))
        })?;
    }
    Ok(())
}

/// The unit index record prefix.
const UNIT_PREFIX: [u8; 3] = [0x00, b'a', b'u'];
const UNIT_KEY_LEN: usize = HEAD_LEN + UNIT_PREFIX.len() + 8 + 4;
/// A unit record's value past its key and end: kind, length, key length.
const UNIT_HEADER: usize = 1 + 8 + 4;

/// The unit index record of the `ordinal`-th unit at `seqno` under `head`.
pub(crate) fn unit_key(head: Head, seqno: SeqNo, ordinal: u32) -> [u8; UNIT_KEY_LEN] {
    let mut out = [0u8; UNIT_KEY_LEN];
    out[..HEAD_LEN].copy_from_slice(head.bytes());
    let at = HEAD_LEN + UNIT_PREFIX.len();
    out[HEAD_LEN..at].copy_from_slice(&UNIT_PREFIX);
    out[at..at + 8].copy_from_slice(&seqno.to_be_bytes());
    out[at + 8..].copy_from_slice(&ordinal.to_be_bytes());
    out
}

/// The unit index prefix under `head`.
fn unit_prefix(head: Head) -> [u8; HEAD_LEN + UNIT_PREFIX.len()] {
    let mut out = [0u8; HEAD_LEN + UNIT_PREFIX.len()];
    out[..HEAD_LEN].copy_from_slice(head.bytes());
    out[HEAD_LEN..].copy_from_slice(&UNIT_PREFIX);
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
    let id = key
        .get(HEAD_LEN..)
        .and_then(|k| k.strip_prefix(UNIT_PREFIX.as_slice()))
        .ok_or_else(bad)?;
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

/// Up to `limit` units of `scope` from `(seqno, ordinal)` on, below `to`
/// (exclusive), visible at `snapshot`, in commit order. With `held`, each
/// unit's copy is reserved there before it is made, and the units stop at
/// the first that does not fit; the flag says they stopped so.
///
/// # Errors
///
/// A record could not be read or is malformed.
pub(crate) fn units_from(
    scope: &Scope<'_>,
    from: (SeqNo, u32),
    to: SeqNo,
    snapshot: SeqNo,
    limit: usize,
    mut held: Option<&mut coordinode_core::budget::MemoryCharge<'_>>,
) -> StorageResult<(Vec<Unit>, bool)> {
    let start = unit_key(scope.head, from.0, from.1);
    let end = unit_key(scope.head, to, 0);
    if start.as_slice() >= end.as_slice() {
        return Ok((Vec::new(), false));
    }
    let mut out = Vec::new();
    for guard in scope
        .tree
        .range(start.as_slice()..end.as_slice(), snapshot, None)
    {
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
pub(crate) const RETIRE_RECORD_BYTES: u64 = (HEAD_LEN + UNIT_PREFIX.len() + UNIT_KEY_LEN) as u64;

/// Drop `scope`'s unit records below `seqno`, at seqno `at`.
pub(crate) fn retire_units_below(scope: &Scope<'_>, seqno: SeqNo, at: SeqNo) {
    let end = unit_key(scope.head, seqno, 0);
    scope
        .tree
        .remove_range(unit_prefix(scope.head).as_slice(), end.as_slice(), at);
}

/// The records of a tree rebuilt from a copy's history, counted from the
/// versions actually reinstalled. The history comes in key order, so one
/// seqno's units arrive scattered: they are summed in a bounded map and
/// added to the scope's records a page at a time.
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

    /// Add what was counted for source `part` to its `scope`'s records.
    /// Each sum is written at `at`, a seqno above every earlier page's, so
    /// the newest sum is the one read whichever of them a flush separated.
    ///
    /// # Errors
    ///
    /// A record could not be read.
    pub(crate) fn spill(
        &mut self,
        part: Partition,
        scope: &Scope<'_>,
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
            add_units(scope, *seqno, units, at)?;
        }
        for (seqno, bytes) in pending {
            let k = key(scope.head, seqno);
            let held = match scope.tree.get(k.as_slice(), SeqNo::MAX)? {
                Some(value) => decode(&k, &value)?.1,
                None => 0,
            };
            // Both are sums of one seqno's units in one tree.
            scope
                .tree
                .insert(k.as_slice(), encode(held + bytes).as_slice(), at);
        }
        Ok(())
    }
}

/// The record key prefix.
const PREFIX: [u8; 3] = [0x00, b'a', b'x'];
/// Exclusive end of the record range.
const END: [u8; 3] = [0x00, b'a', b'y'];
const KEY_LEN: usize = HEAD_LEN + PREFIX.len() + 8;

/// The record for `seqno` under `head`.
pub(crate) fn key(head: Head, seqno: SeqNo) -> [u8; KEY_LEN] {
    let mut out = [0u8; KEY_LEN];
    out[..HEAD_LEN].copy_from_slice(head.bytes());
    out[HEAD_LEN..HEAD_LEN + PREFIX.len()].copy_from_slice(&PREFIX);
    out[HEAD_LEN + PREFIX.len()..].copy_from_slice(&seqno.to_be_bytes());
    out
}

/// Key and value bytes of one record.
pub(crate) const RECORD_BYTES: u64 = (KEY_LEN + 8) as u64;

/// The room the accounting of one apply of `units` writes takes, bounded
/// from above: one record per source tree it touches (it touches at most one
/// tree per write and no more trees than there are), one unit record per
/// unit, `keys` being the bytes of the units' keys and range ends together,
/// and the coverage marker the accounting batch carries.
pub(crate) fn record_allowance(units: usize, keys: u64) -> u64 {
    let trees = Partition::all().len();
    // At most a dozen trees, and units bounded by the apply's memory: far
    // below u64::MAX.
    crate::engine::footprint::footprint_of(RECORD_BYTES, 1) * units.min(trees) as u64
        + crate::engine::footprint::footprint_of(
            ((UNIT_KEY_LEN + UNIT_HEADER) * units) as u64 + keys,
            units,
        )
        + crate::engine::footprint::footprint_of(coverage::MARKER_LEN as u64, 1)
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

/// Add the records of `scope` in `[from, to)` visible at `snapshot` to
/// `material`.
pub(crate) fn read_into(
    scope: &Scope<'_>,
    from: SeqNo,
    to: Option<SeqNo>,
    snapshot: SeqNo,
    material: &mut Material,
) -> StorageResult<()> {
    let start = key(scope.head, from);
    let end: Vec<u8> = match to {
        Some(to) => key(scope.head, to).to_vec(),
        None => scope.head.key(&END),
    };
    if start.as_slice() >= end.as_slice() {
        return Ok(());
    }
    for guard in scope
        .tree
        .range(start.as_slice()..end.as_slice(), snapshot, None)
    {
        let (k, v) = guard.into_inner()?;
        let (seqno, bytes) = decode(&k, &v)?;
        material.add(seqno, bytes)?;
    }
    Ok(())
}

/// How far the readable floor moves before the records below it are dropped
/// (seqnos; with an oracle, microseconds: about a minute). Each retirement
/// is one range deletion per accounted source.
pub(crate) const RETIRE_WINDOW: SeqNo = 1 << 26;

/// Drop `scope`'s records below `floor`, at seqno `at`, above every record
/// they cover. A scope with none there takes no write: a deletion would make
/// the tree flush a table for nothing. Returns whether it wrote.
///
/// # Errors
///
/// A record could not be read or is malformed.
pub(crate) fn retire_below(scope: &Scope<'_>, floor: SeqNo, at: SeqNo) -> StorageResult<bool> {
    if first_from(scope, 0, SeqNo::MAX)?.is_none_or(|first| first >= floor) {
        return Ok(false);
    }
    let start = scope.head.key(&PREFIX);
    let end = key(scope.head, floor);
    scope
        .tree
        .remove_range(start.as_slice(), end.as_slice(), at);
    Ok(true)
}

/// The first seqno at or after `from` with a record in `scope` visible at
/// `snapshot`.
pub(crate) fn first_from(
    scope: &Scope<'_>,
    from: SeqNo,
    snapshot: SeqNo,
) -> StorageResult<Option<SeqNo>> {
    let start = key(scope.head, from);
    let end = scope.head.key(&END);
    match scope
        .tree
        .range(start.as_slice()..end.as_slice(), snapshot, None)
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
        .get(HEAD_LEN..)
        .and_then(|k| k.strip_prefix(PREFIX.as_slice()))
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
