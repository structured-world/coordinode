//! Folding committed material units into per-key descriptors, in commit
//! order, for coalesced state-delta consumers.
//!
//! Such a consumer at continuation `s` reading at cut `r` needs, for every key
//! changed in `[s, r)`, the units that reconstruct the key at `r`: the last
//! value or deletion of the key and every operand merged onto it since.
//! Overwritten payloads are not needed. Each key's descriptor
//! (`head 0x00 "ak" key`, in the source's scope of the accounting tree)
//! holds its newest change and the encoded length of the units of that
//! closure it owns alone; the summary ([`super::summary`]) holds the charges
//! by newest change.
//!
//! A range deletion is one unit shared by every key it deleted. It is
//! charged once (`head 0x00 "ar" be(seqno) be(ordinal)`), at the newest
//! change of the keys still resting on it: keys untouched since it are
//! counted, keys with operands merged onto it are indexed by their newest
//! change (`head 0x00 "ad" tombstone be(newest) key`), so the newest one is one step
//! from the end of that index. A key written or deleted again leaves it;
//! with no key left on it, it is needed by no reader and charged nothing.
//!
//! Units are folded strictly in commit order below a complete cut, so no
//! unit lands inside a closure already folded: a commit with an older
//! timestamp is folded before a newer one whatever order they were applied
//! in. A closure is the real one: a retention floor or a resolved length
//! does not replace operands by a base.

use std::collections::BTreeMap;

use lsm_tree::{AbstractTree, Guard, SeqNo};

use coordinode_core::budget::{MemoryCharge, QueryBudget};

use super::summary::Moves;
use super::{HEAD_LEN, Head, Scope, UnitKind};
use crate::error::{StorageError, StorageResult};

/// Descriptor key prefix.
const PREFIX: [u8; 3] = [0x00, b'a', b'k'];
/// Range tombstone record prefix.
const TOMB_PREFIX: [u8; 3] = [0x00, b'a', b'r'];
/// Prefix of the index of keys resting on a range tombstone.
const DEPENDENT_PREFIX: [u8; 3] = [0x00, b'a', b'd'];

/// A range deletion: the unit at `(seqno, ordinal)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Tomb {
    pub(crate) seqno: SeqNo,
    pub(crate) ordinal: u32,
}

impl Tomb {
    fn id(self) -> [u8; 12] {
        let mut out = [0u8; 12];
        out[..8].copy_from_slice(&self.seqno.to_be_bytes());
        out[8..].copy_from_slice(&self.ordinal.to_be_bytes());
        out
    }

    fn record_key(self, head: Head) -> Vec<u8> {
        let mut out = head.key(&TOMB_PREFIX);
        out.extend_from_slice(&self.id());
        out
    }

    fn dependents_prefix(self, head: Head) -> Vec<u8> {
        let mut out = head.key(&DEPENDENT_PREFIX);
        out.extend_from_slice(&self.id());
        out
    }

    /// The dependents index entry of the key stored under `suffix`
    /// ([`stored_suffix`]).
    fn dependent_key(self, head: Head, newest: SeqNo, suffix: &[u8]) -> Vec<u8> {
        let mut out = self.dependents_prefix(head);
        out.extend_from_slice(&newest.to_be_bytes());
        out.extend_from_slice(suffix);
        out
    }
}

/// The longest key an LSM key can hold.
const KEY_LIMIT: usize = u16::MAX as usize;
/// What the longest metadata key puts before a key: the scope's head and the
/// dependents index.
const DEPENDENT_HEAD: usize = HEAD_LEN + DEPENDENT_PREFIX.len() + 12 + 8;
/// The longest key stored as it is in the metadata keys.
const DIRECT_MAX: usize = KEY_LIMIT - DEPENDENT_HEAD - 1;
/// A longer key's hash and probe.
const HASHED: usize = 16 + 2;
/// The bytes of a longer key kept in its stored form, so that form is one
/// byte longer than any key stored as it is and sorts by the key's head.
const CUT: usize = DIRECT_MAX + 1 - HASHED;

/// How `key` is stored in the metadata keys at `probe`: as it is when it
/// fits beside the longest metadata prefix; otherwise its head, its hash and
/// the probe, the full key kept in the descriptor's value. A stored form of
/// another key at the same probe moves this one to the next.
fn stored_suffix(key: &[u8], probe: u16) -> Vec<u8> {
    if key.len() <= DIRECT_MAX {
        return key.to_vec();
    }
    let mut out = Vec::with_capacity(DIRECT_MAX + 1);
    out.extend_from_slice(&key[..CUT]);
    out.extend_from_slice(&xxhash_rust::xxh3::xxh3_128(key).to_be_bytes());
    out.extend_from_slice(&probe.to_be_bytes());
    out
}

/// The descriptor key under `head` of the key stored under `suffix`.
fn descriptor_key(head: Head, suffix: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEAD_LEN + PREFIX.len() + suffix.len());
    out.extend_from_slice(head.bytes());
    out.extend_from_slice(&PREFIX);
    out.extend_from_slice(suffix);
    out
}

/// Where `key`'s descriptor is stored in `scope` and what it holds, skipping
/// stored forms `taken` by other keys in a fold not written yet.
fn locate(
    scope: &Scope<'_>,
    key: &[u8],
    taken: &BTreeMap<Vec<u8>, Vec<u8>>,
) -> StorageResult<(Vec<u8>, Option<Descriptor>)> {
    let tree = scope.tree;
    for probe in 0..=u16::MAX {
        let suffix = stored_suffix(key, probe);
        if key.len() <= DIRECT_MAX {
            let k = descriptor_key(scope.head, &suffix);
            let held = match tree.get(k.as_slice(), SeqNo::MAX)? {
                Some(v) => Some(decode(&k, &v)?.1),
                None => None,
            };
            return Ok((suffix, held));
        }
        if taken
            .get(&suffix)
            .is_some_and(|owner| owner.as_slice() != key)
        {
            continue;
        }
        let k = descriptor_key(scope.head, &suffix);
        match tree.get(k.as_slice(), SeqNo::MAX)? {
            None => return Ok((suffix, None)),
            Some(v) => {
                let (full, d) = decode(&k, &v)?;
                if full.as_deref() == Some(key) {
                    return Ok((suffix, Some(d)));
                }
            }
        }
    }
    Err(StorageError::Serialization(format!(
        "material fold: every stored form of a {}-byte key is taken",
        key.len()
    )))
}

/// One key's newest folded change and the encoded length of the closure
/// units it owns alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Descriptor {
    pub(crate) newest: SeqNo,
    pub(crate) charge: u64,
    /// The range deletion the key's closure starts from, if any.
    pub(crate) rests_on: Option<Tomb>,
    /// Whether operands were merged onto it since.
    pub(crate) touched: bool,
    /// Whether the key has a value to read: a deletion (point or range)
    /// with no operand after it leaves none.
    pub(crate) visible: bool,
}

impl Descriptor {
    /// What a fixed snapshot reading the key needs of it: its closure when
    /// it has a value, nothing when it is deleted.
    fn visible_charge(&self) -> u64 {
        if self.visible { self.charge } else { 0 }
    }
}

/// What a range tombstone is charged: its length, at the newest change of
/// the keys resting on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TombState {
    bytes: u64,
    /// Keys resting on it with no change since it.
    untouched: u64,
    /// Keys resting on it with operands merged onto it since: they have a
    /// value, and it is their base.
    touched: u64,
    /// The newest change of a key resting on it.
    newest: SeqNo,
}

impl TombState {
    /// What a fixed snapshot needs of it: its length while it is the base of
    /// a key with a value.
    fn visible_charge(&self) -> u64 {
        if self.touched > 0 { self.bytes } else { 0 }
    }
}

type Change<T> = (Option<T>, Option<T>);

/// Units folded into one source's descriptors and summary, written together.
pub(crate) struct Fold<'t> {
    scope: Scope<'t>,
    /// Descriptors touched, by key: as read, and as they stand now.
    touched: BTreeMap<Vec<u8>, Change<Descriptor>>,
    /// The stored form of each touched key ([`stored_suffix`]).
    stored: BTreeMap<Vec<u8>, Vec<u8>>,
    /// The key each stored form in `stored` belongs to.
    owners: BTreeMap<Vec<u8>, Vec<u8>>,
    /// Range tombstones touched: as read, and as they stand now.
    tombs: BTreeMap<Tomb, Change<TombState>>,
    /// Entries of the dependents index added (`true`) or removed.
    dependents: BTreeMap<Vec<u8>, bool>,
    /// The last unit's commit, to keep the order.
    last: Option<SeqNo>,
    /// How much the bytes a fixed snapshot needs moved by what was folded.
    visible: i128,
    /// `visible` as it stood once each commit's units were all folded, in
    /// commit order.
    marks: Vec<(SeqNo, i128)>,
    /// The memory the fold holds, reserved before it grows (`None`: not
    /// metered).
    held: Option<MemoryCharge<'t>>,
    /// Whether `held` covers what the fold takes once (`base`) yet.
    reads_held: bool,
    /// What the fold takes once besides its reads through the runs:
    /// [`FOLD_BASE`] and the reads' fixed part.
    base: u64,
    /// The runs whose reads `held` covers: the most the tree had at any of
    /// the fold's reservations, and one more a flush may add before a read
    /// opens.
    runs_held: usize,
    /// The summary moves of every change folded, as it is folded.
    moves: Moves,
}

/// The charge moves one point unit makes at most: its key's, and a range
/// tombstone's as the key leaves it or an operand lands on it and as it is
/// settled.
const UNIT_SHIFTS: usize = 4;
/// The moves of a range page's own tombstone at most: as it is first taken
/// in, as its keys join it, and as it is settled.
const RANGE_PAGE_SHIFTS: usize = 3;
/// What one summary bucket moved takes: its map entry and its record.
const MOVE_COST: usize = map_entry(MOVE) + BATCH_SLOT;

/// What one entry of an ordered map of `kv`-byte entries can take beyond
/// the bytes its key and value own on the heap: its slot in a leaf of
/// eleven, at least five of which are in use once the map has more than
/// one leaf, and its share of the internal nodes above, each over at least
/// six leaves. The first leaf and internal node of a map are paid with the
/// fold ([`FOLD_BASE`]).
const fn map_entry(kv: usize) -> usize {
    let leaf = 16 + 11 * kv;
    let internal = leaf + 12 * size_of::<usize>();
    (leaf + internal.div_ceil(6)).div_ceil(5)
}

/// What a map of `kv`-byte entries takes before its second leaf.
const fn map_first_nodes(kv: usize) -> usize {
    let leaf = 16 + 11 * kv;
    2 * leaf + 12 * size_of::<usize>()
}

/// An entry of each of the fold's maps.
const TOUCHED: usize = size_of::<(Vec<u8>, Change<Descriptor>)>();
const FORMS: usize = size_of::<(Vec<u8>, Vec<u8>)>();
const DEPENDENT: usize = size_of::<(Vec<u8>, bool)>();
const TOMBS: usize = size_of::<(Tomb, Change<TombState>)>();
const MOVE: usize = size_of::<(super::summary::BucketKey, i128)>();

/// A record's slot in a write batch, which doubles as it grows: the old
/// block and the new one twice its size are held together while it is
/// copied.
const BATCH_SLOT: usize = 3 * (2 * size_of::<lsm_tree::UserKey>() + size_of::<usize>());
/// What a batch's copy of a key or value holds beyond its bytes.
const SLICE_HEAD: usize = 2 * size_of::<usize>();
/// A snapshot mark: its place in a list that doubles, and its copy as the
/// total written.
const MARK: usize = 3 * size_of::<(SeqNo, i128)>() + size_of::<(SeqNo, u64)>();

/// The memory folding one unit of a `key_len`-byte key can take, bounded
/// from above, from its taking in to its batch being built:
/// - the key as the fold holds it, and for a key not stored as it is, its
///   stored form and owner, and the full key in its descriptor's value;
/// - two dependents entries, and their records in the batch;
/// - the range tombstone it rests on, and its record;
/// - the summary moves of its charge and of that tombstone's, at every
///   level, each a record: reserved at their most, and what the unit did
///   not move given back once it is folded;
/// - its descriptor key and value, as built and as the batch copies them;
/// - its snapshot mark.
const fn unit_cost(key_len: usize) -> u64 {
    let stored = if key_len > DIRECT_MAX {
        DIRECT_MAX + 1
    } else {
        key_len
    };
    let touched = key_len + map_entry(TOUCHED);
    let long = if key_len > DIRECT_MAX {
        2 * (key_len + stored + map_entry(FORMS)) + 2 * key_len + SLICE_HEAD
    } else {
        0
    };
    let dependent = DEPENDENT_HEAD + stored;
    let dependents = 2 * (dependent + map_entry(DEPENDENT) + dependent + SLICE_HEAD + BATCH_SLOT);
    let tomb = map_entry(TOMBS)
        + 2 * (HEAD_LEN + TOMB_PREFIX.len() + 12 + TOMB_LEN + SLICE_HEAD)
        + BATCH_SLOT;
    let moves = UNIT_SHIFTS * super::summary::MOVES_PER_SHIFT * MOVE_COST;
    let descriptor =
        2 * (HEAD_LEN + PREFIX.len() + stored + DESCRIPTOR_LEN + SLICE_HEAD) + BATCH_SLOT;
    // A key is at most u16::MAX bytes: far below u64::MAX.
    (touched + long + dependents + tomb + moves + descriptor + MARK) as u64
}

/// What a fold's reads hold beside the runs they go through: the memtables'
/// iterators and the read's own state.
const READS_FIXED: usize = 16 << 10;

/// The trees' data block size: the LSM default, which the engine keeps.
const DATA_BLOCK: usize = 4096;

/// What one table read holds: its iterator, and the block it stands in,
/// which holds the data block size and at most one record past it, at most
/// the longest of the fold's records.
const RUN_READ: usize = 1024 + DATA_BLOCK + RANGE_RECORD_MAX;

/// The table reads one read holds in each run: a range read opens a run's
/// first table as a source of its own and the rest as a second (one more
/// table, or a run reader, which going one way holds one table at a time).
const READS_PER_RUN: usize = 2;

/// The runs a read of `tree` goes through as it stands: each of its first
/// level's, and the one of each deeper level. Their tables do not count: a
/// read holds a run's tables one at a time past its first.
fn runs(tree: &lsm_tree::AnyTree) -> usize {
    tree.current_version()
        .iter_levels()
        .map(|level| level.run_count())
        .sum()
}

/// What a fold's reads hold through `runs` runs: two reads at once (a range
/// iterator and a point read or a second iterator beside it). A tree with
/// more runs than a round's memory holds reads for folds nothing until
/// compaction merges them.
const fn reads_cost(runs: usize) -> u64 {
    // Runs are bounded by the tree's levels and its first level's runs: far
    // below u64::MAX.
    (2 * READS_PER_RUN * runs * RUN_READ) as u64
}

/// The cursor record written with a fold, without a key to resume from:
/// what it is encoded to, and the batch's copy.
const CURSOR_RECORD: usize = 2 * (64 + SLICE_HEAD) + BATCH_SLOT;

/// What a fold takes once, whatever it folds, besides its reads: the first
/// nodes of each of its maps, the batch's first block, and the cursor
/// written with it.
const FOLD_BASE: u64 = (map_first_nodes(TOUCHED)
    + 2 * map_first_nodes(FORMS)
    + map_first_nodes(DEPENDENT)
    + map_first_nodes(TOMBS)
    + map_first_nodes(MOVE)
    + BATCH_SLOT
    + CURSOR_RECORD) as u64;

/// The longest descriptor record a range page reads: a stored form at its
/// longest under the head and a value carrying the longest key.
const RANGE_RECORD_MAX: usize =
    HEAD_LEN + PREFIX.len() + DIRECT_MAX + 1 + DESCRIPTOR_LEN + KEY_LIMIT;

/// The memory a range page takes for one key whose descriptor record is
/// `record` bytes long: what folding it takes, and its key and stored form
/// as the page lists them before they are taken in.
const fn range_key_cost(record: usize) -> u64 {
    // Its place in the list, which doubles as it grows: the old block and
    // the new one twice its size are held together while it is copied.
    let place = 3 * size_of::<(Vec<u8>, Vec<u8>, Descriptor)>();
    unit_cost(record) + (2 * record + place) as u64
}

/// The memory a range page takes before its first key, for bounds of
/// `start` and `end` bytes and a resume key of `from` bytes (`0`: none): the
/// descriptor keys it reads between and their heads, its tombstone's entry
/// and record with the record's copy in the batch, its mark.
const fn range_page_cost(start: usize, end: usize, from: usize) -> u64 {
    let start = if start < CUT { start } else { CUT };
    let end = if end < CUT { end } else { CUT };
    let first = if from > 0 {
        from
    } else {
        HEAD_LEN + PREFIX.len() + start
    };
    let stop = HEAD_LEN + PREFIX.len() + end + HASHED + 1;
    let tomb = map_entry(TOMBS)
        + 2 * (HEAD_LEN + TOMB_PREFIX.len() + 12 + TOMB_LEN + SLICE_HEAD)
        + BATCH_SLOT;
    let moves = RANGE_PAGE_SHIFTS * super::summary::MOVES_PER_SHIFT * MOVE_COST;
    // The key the next page resumes from, in the cursor written with this
    // one: copied out of the tree's read, encoded, and copied by the batch.
    let resume = 3 * (HEAD_LEN + PREFIX.len() + DIRECT_MAX + 1 + SLICE_HEAD);
    // Bounded by a few keys: far below u64::MAX.
    (first + stop + start + end + tomb + moves + resume + MARK) as u64
}

/// What one scan of `tree` holds besides the records it hands out, when no
/// record in it is longer than `longest_record` key and value bytes: the
/// read's own state and memtable iterators, and in each run (its memtables
/// counted as one more) the table reads it opens, each standing in a data
/// block that holds at most one record past the block size.
pub(crate) fn scan_holds(tree: &lsm_tree::AnyTree, longest_record: u64) -> u64 {
    let run_read = (RUN_READ - RANGE_RECORD_MAX) as u64 + longest_record;
    // Runs are bounded by the tree's levels and its first level's runs, a
    // record by the engine's key and value limits: far below u64::MAX.
    READS_FIXED as u64 + (READS_PER_RUN * (runs(tree) + 1)) as u64 * run_read
}

/// The least memory with which a fold of a tree of `runs` runs always goes
/// on: its reads, and a range page between the longest bounds with its
/// longest key, or the longest point unit.
pub(crate) const fn least_memory(runs: usize) -> u64 {
    let range = range_page_cost(
        KEY_LIMIT,
        KEY_LIMIT,
        HEAD_LEN + PREFIX.len() + DIRECT_MAX + 1,
    ) + range_key_cost(RANGE_RECORD_MAX);
    let point = unit_cost(KEY_LIMIT);
    let reads = READS_FIXED as u64 + reads_cost(runs + 1);
    FOLD_BASE + reads + if range > point { range } else { point }
}

/// A fold's output ([`Fold::write`]).
pub(crate) struct Written<'t> {
    /// Its records and the cursor's, to apply as one batch.
    pub(crate) batch: lsm_tree::WriteBatch,
    /// The key and value bytes in `batch`.
    pub(crate) bytes: u64,
    /// The bytes a fixed snapshot needs after each commit it completed.
    pub(crate) marks: Vec<(SeqNo, u64)>,
    /// The memory the fold reserved, held until `batch` is applied.
    pub(crate) held: Option<MemoryCharge<'t>>,
}

/// How a page of a range deletion ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RangePage {
    /// Every key it deleted is folded; the page went through this many.
    Done(usize),
    /// Keys from this descriptor key on are still to fold, in a later
    /// fold over what this one wrote; the page went through this many
    /// (fewer than a page when the fold's memory ran out). The key is the
    /// one the tree read, shared, not a copy.
    From(lsm_tree::UserKey, usize),
    /// Nothing was taken in: the fold's memory does not hold even the
    /// page's bounds and its first key. The range stays where it was.
    NoRoom,
}

impl<'t> Fold<'t> {
    /// A fold over `scope`.
    pub(crate) fn new(scope: Scope<'t>) -> Self {
        Self {
            scope,
            touched: BTreeMap::new(),
            stored: BTreeMap::new(),
            owners: BTreeMap::new(),
            tombs: BTreeMap::new(),
            dependents: BTreeMap::new(),
            last: None,
            visible: 0,
            marks: Vec::new(),
            held: None,
            reads_held: false,
            base: 0,
            runs_held: 0,
            moves: Moves::default(),
        }
    }

    /// Give back the moves reserved for `shifts` charge moves beyond those
    /// they made: the summary held `before` buckets when they began.
    fn settle_moves(&mut self, shifts: usize, before: usize) -> StorageResult<()> {
        let Some(held) = self.held.as_mut() else {
            return Ok(());
        };
        let made = self.moves.len() - before;
        let unused = (shifts * super::summary::MOVES_PER_SHIFT)
            .checked_sub(made)
            .ok_or_else(|| {
                StorageError::Serialization(format!(
                    "material fold: {shifts} charge moves reserved, {made} buckets moved"
                ))
            })?;
        // Bounded by the buckets reserved: far below u64::MAX.
        held.shrink((unused * MOVE_COST) as u64);
        Ok(())
    }

    /// A fold over `scope` whose memory is reserved from `budget` before it
    /// is taken: a unit that does not fit is not folded.
    pub(crate) fn metered(scope: Scope<'t>, budget: &'t QueryBudget) -> Self {
        Self {
            held: Some(budget.empty_charge()),
            base: FOLD_BASE + READS_FIXED as u64,
            ..Self::new(scope)
        }
    }

    /// This fold continuing one before it at `last`, its commit not yet
    /// known complete: a unit of a later commit marks it.
    #[must_use]
    pub(crate) fn after(mut self, last: Option<SeqNo>) -> Self {
        self.last = last;
        self
    }

    /// Reserve `bytes` before the fold takes them; `false` when they do not
    /// fit, the fold left as it was.
    fn room(&mut self, bytes: u64) -> bool {
        if self.held.is_none() {
            return true;
        }
        // Every reservation comes before the reads it is for: the runs are
        // counted then, so a flush or a compaction between two reservations
        // is covered at the next one.
        let runs = self.scope.runs() + 1;
        let grown = if runs > self.runs_held {
            reads_cost(runs - self.runs_held)
        } else {
            0
        };
        // The first reservation also covers what the fold takes once.
        let once = if self.reads_held { 0 } else { self.base };
        let Some(held) = self.held.as_mut() else {
            return true;
        };
        let fits = held.grow(bytes + grown + once).is_ok();
        if fits {
            self.reads_held = true;
            self.runs_held = self.runs_held.max(runs);
        }
        fits
    }

    fn order(&mut self, seqno: SeqNo) -> StorageResult<()> {
        match self.last {
            Some(last) if seqno < last => {
                return Err(StorageError::Serialization(format!(
                    "material fold: unit at {seqno} after one at {last}"
                )));
            }
            // A later commit: every unit of the one before is folded.
            Some(last) if seqno > last => self.marks.push((last, self.visible)),
            _ => {}
        }
        self.last = Some(seqno);
        Ok(())
    }

    /// Mark the last commit folded as complete when `next`, the next unit's
    /// commit, is a later one. `false`, marking nothing, without the memory
    /// for the mark.
    pub(crate) fn close_before(&mut self, next: SeqNo) -> bool {
        if self.last.is_some_and(|last| last < next) {
            return self.close();
        }
        true
    }

    /// Mark the last commit folded as complete: no unit of it is left.
    /// `false`, marking nothing, without the memory for the mark.
    pub(crate) fn close(&mut self) -> bool {
        if let Some(last) = self.last
            && self.marks.last().is_none_or(|(c, _)| *c != last)
        {
            if !self.room(MARK as u64) {
                return false;
            }
            self.marks.push((last, self.visible));
        }
        true
    }

    /// Reserve what writing the fold takes whatever it folded: its reads,
    /// first nodes, batch block and cursor record. `false` without the room:
    /// a fold that folded nothing, only completing a commit or writing its
    /// cursor, is not written then.
    pub(crate) fn ready(&mut self) -> bool {
        self.room(0)
    }

    /// Whether the fold took in nothing and completed no commit: writing it
    /// would only rewrite the cursor where it is.
    pub(crate) fn is_empty(&self) -> bool {
        self.touched.is_empty()
            && self.tombs.is_empty()
            && self.dependents.is_empty()
            && self.marks.is_empty()
    }

    fn descriptor(&mut self, key: &[u8]) -> StorageResult<Option<Descriptor>> {
        if let Some((_, now)) = self.touched.get(key) {
            return Ok(*now);
        }
        let (suffix, held) = locate(&self.scope, key, &self.owners)?;
        self.hold(key, suffix, held);
        Ok(held)
    }

    /// Take `key`'s descriptor as read, stored under `suffix`, into the fold.
    /// A key stored as it is needs no record of its form.
    fn hold(&mut self, key: &[u8], suffix: Vec<u8>, held: Option<Descriptor>) {
        self.touched.insert(key.to_vec(), (held, held));
        if suffix.as_slice() != key {
            self.owners.insert(suffix.clone(), key.to_vec());
            self.stored.insert(key.to_vec(), suffix);
        }
    }

    /// The stored form of a key the fold holds.
    fn suffix_of(&self, key: &[u8]) -> Vec<u8> {
        self.stored
            .get(key)
            .cloned()
            .unwrap_or_else(|| key.to_vec())
    }

    fn set_descriptor(&mut self, key: &[u8], next: Option<Descriptor>) {
        if let Some(entry) = self.touched.get_mut(key) {
            let charge = |d: Option<Descriptor>| i128::from(d.map_or(0, |d| d.visible_charge()));
            self.visible += charge(next) - charge(entry.1);
            let summary = |d: Option<Descriptor>| d.map(|d| (d.newest, d.charge));
            self.moves.shift(summary(entry.1), summary(next));
            entry.1 = next;
        }
    }

    fn tomb(&mut self, tomb: Tomb) -> StorageResult<Option<TombState>> {
        if let Some((_, now)) = self.tombs.get(&tomb) {
            return Ok(*now);
        }
        let k = tomb.record_key(self.scope.head);
        let held = match self.scope.tree.get(k.as_slice(), SeqNo::MAX)? {
            Some(value) => Some(decode_tomb(&k, &value)?),
            None => None,
        };
        self.tombs.insert(tomb, (held, held));
        Ok(held)
    }

    fn set_tomb(&mut self, tomb: Tomb, next: Option<TombState>) {
        if let Some(entry) = self.tombs.get_mut(&tomb) {
            let charge = |t: Option<TombState>| i128::from(t.map_or(0, |t| t.visible_charge()));
            self.visible += charge(next) - charge(entry.1);
            let summary = |t: Option<TombState>| t.map(|t| (t.newest, t.bytes));
            self.moves.shift(summary(entry.1), summary(next));
            entry.1 = next;
        }
    }

    /// The newest change among the touched keys resting on `tomb`.
    fn newest_dependent(&self, tomb: Tomb) -> StorageResult<Option<SeqNo>> {
        let prefix = tomb.dependents_prefix(self.scope.head);
        let newest_of = |key: &[u8]| {
            key.get(prefix.len()..prefix.len() + 8)
                .and_then(|s| <[u8; 8]>::try_from(s).ok())
                .map(u64::from_be_bytes)
        };
        let mut end = prefix.clone();
        end.extend_from_slice(&[0xff; 9]);
        // Added in this fold.
        let mut newest = self
            .dependents
            .range(prefix.clone()..end.clone())
            .filter(|(_, added)| **added)
            .filter_map(|(k, _)| newest_of(k))
            .max();
        // Held, skipping what this fold removed: the first one left from
        // the end is the newest.
        for guard in self
            .scope
            .tree
            .range(prefix.as_slice()..end.as_slice(), SeqNo::MAX, None)
            .rev()
        {
            let k = guard.key()?;
            if self.dependents.get(k.as_ref()) == Some(&false) {
                continue;
            }
            let held = newest_of(&k).ok_or_else(|| {
                StorageError::Serialization(format!("material dependent record {k:02x?}"))
            })?;
            newest = Some(newest.map_or(held, |n| n.max(held)));
            break;
        }
        Ok(newest)
    }

    /// Recompute what `tomb` is charged after its keys changed.
    fn settle(&mut self, tomb: Tomb) -> StorageResult<()> {
        let Some(state) = self.tomb(tomb)? else {
            return Err(StorageError::Serialization(format!(
                "material fold: a key rests on a missing range tombstone {tomb:?}"
            )));
        };
        let touched = self.newest_dependent(tomb)?;
        let untouched = (state.untouched > 0).then_some(tomb.seqno);
        let next = match (untouched, touched) {
            (None, None) => None,
            (a, b) => Some(TombState {
                newest: a.max(b).unwrap_or(tomb.seqno),
                ..state
            }),
        };
        self.set_tomb(tomb, next);
        Ok(())
    }

    /// `key`, resting on `tomb` as `d` says, leaves it.
    fn leave(&mut self, key: &[u8], d: Descriptor, tomb: Tomb) -> StorageResult<()> {
        let mut state = self.tomb(tomb)?.ok_or_else(|| {
            StorageError::Serialization(format!(
                "material fold: {key:02x?} rests on a missing range tombstone"
            ))
        })?;
        let suffix = self.suffix_of(key);
        let head = self.scope.head;
        let count = if d.touched {
            self.dependents
                .insert(tomb.dependent_key(head, d.newest, &suffix), false);
            &mut state.touched
        } else {
            &mut state.untouched
        };
        *count = count.checked_sub(1).ok_or_else(|| {
            StorageError::Serialization(format!(
                "material fold: range tombstone {tomb:?} counts fewer keys than rest on it"
            ))
        })?;
        self.set_tomb(tomb, Some(state));
        self.settle(tomb)
    }

    /// Fold one point unit of `key` committed at `seqno`, of encoded length
    /// `bytes`.
    ///
    /// # Errors
    ///
    /// Returns `false`, folding nothing, when the memory the unit can take
    /// does not fit the fold's budget.
    ///
    /// # Errors
    ///
    /// The unit comes before one already folded, is a range deletion, a
    /// record could not be read or is malformed, or a closure would pass
    /// `u64::MAX`.
    pub(crate) fn unit(
        &mut self,
        seqno: SeqNo,
        key: &[u8],
        kind: UnitKind,
        bytes: u64,
    ) -> StorageResult<bool> {
        if !self.room(unit_cost(key.len())) {
            return Ok(false);
        }
        let moved = self.moves.len();
        self.order(seqno)?;
        let current = self.descriptor(key)?;
        let next = match kind {
            UnitKind::Value | UnitKind::Delete => {
                if let Some(d) = current
                    && let Some(tomb) = d.rests_on
                {
                    self.leave(key, d, tomb)?;
                }
                Descriptor {
                    newest: seqno,
                    charge: bytes,
                    rests_on: None,
                    touched: false,
                    visible: kind == UnitKind::Value,
                }
            }
            UnitKind::Operand => {
                let charge = current
                    .map_or(0, |d| d.charge)
                    .checked_add(bytes)
                    .ok_or_else(|| {
                        StorageError::Serialization(format!(
                            "material fold: closure of {key:02x?} passes u64::MAX"
                        ))
                    })?;
                let rests_on = current.and_then(|d| d.rests_on);
                if let (Some(d), Some(tomb)) = (current, rests_on) {
                    let suffix = self.suffix_of(key);
                    let head = self.scope.head;
                    if d.touched {
                        self.dependents
                            .insert(tomb.dependent_key(head, d.newest, &suffix), false);
                    } else if let Some(mut state) = self.tomb(tomb)? {
                        // Its first operand: the key now has a value resting
                        // on the range deletion.
                        state.untouched = state.untouched.checked_sub(1).ok_or_else(|| {
                            StorageError::Serialization(format!(
                                "material fold: range tombstone {tomb:?} has no untouched key left"
                            ))
                        })?;
                        // Bounded by the keys of one tree.
                        state.touched += 1;
                        self.set_tomb(tomb, Some(state));
                    }
                    self.dependents
                        .insert(tomb.dependent_key(head, seqno, &suffix), true);
                    self.settle(tomb)?;
                }
                Descriptor {
                    newest: seqno,
                    charge,
                    rests_on,
                    touched: rests_on.is_some(),
                    // An operand makes a value, also over a deletion.
                    visible: true,
                }
            }
            UnitKind::Range => {
                return Err(StorageError::Serialization(
                    "material fold: a range deletion is folded with range_page".to_string(),
                ));
            }
        };
        self.set_descriptor(key, Some(next));
        self.settle_moves(UNIT_SHIFTS, moved)?;
        Ok(true)
    }

    /// Fold up to `page` of the keys the range deletion `tomb` of
    /// `[start, end)`, of encoded length `bytes`, deleted, from descriptor
    /// key `from` on (`None`: from the start). The fold must hold nothing
    /// written yet that the range covers: the keys are read from the tree.
    ///
    /// # Errors
    ///
    /// The unit comes before one already folded or a record could not be
    /// read or is malformed.
    pub(crate) fn range_page(
        &mut self,
        tomb: Tomb,
        start: &[u8],
        end: &[u8],
        bytes: u64,
        from: Option<&[u8]>,
        page: usize,
    ) -> StorageResult<RangePage> {
        // Before anything is taken in: the page's bounds and resume key, its
        // list of keys, its tombstone's entry and record, the reads that
        // find them, and its first key at the longest.
        if !self.room(range_page_cost(
            start.len(),
            end.len(),
            from.map_or(0, <[u8]>::len),
        )) {
            return Ok(RangePage::NoRoom);
        }
        let moved = self.moves.len();
        self.order(tomb.seqno)?;
        if self.tomb(tomb)?.is_none() {
            self.set_tomb(
                tomb,
                Some(TombState {
                    bytes,
                    untouched: 0,
                    touched: 0,
                    newest: tomb.seqno,
                }),
            );
        }
        // Stored forms sort by the key's head: every key of the range lies
        // between the heads of its bounds, and the full key of each stored
        // form read is checked against them.
        let lead = |key: &[u8]| key[..key.len().min(CUT)].to_vec();
        let scope_head = self.scope.head;
        let first = match from {
            Some(from) => from.to_vec(),
            None => descriptor_key(scope_head, &lead(start)),
        };
        let mut stop = descriptor_key(scope_head, &lead(end));
        stop.extend_from_slice(&[0xff; HASHED + 1]);
        if first.as_slice() >= stop.as_slice() {
            let ended = self.close_range(tomb, 0)?;
            self.settle_moves(RANGE_PAGE_SHIFTS, moved)?;
            return Ok(ended);
        }
        // Each key reserves its place in the list as it is taken in.
        let mut keys = Vec::new();
        let mut resume = None;
        let mut walked = 0;
        for guard in self
            .scope
            .tree
            .range(first.as_slice()..stop.as_slice(), SeqNo::MAX, None)
        {
            let (k, v) = guard.into_inner()?;
            // The key's copies, its descriptor taken in and its move, held
            // before they are made; a key that does not fit starts the next
            // page, and a first key that does not leaves the range where it
            // was, with nothing copied.
            if walked >= page {
                resume = Some(k);
                break;
            }
            if !self.room(range_key_cost(k.len() + v.len())) {
                if walked == 0 {
                    return Ok(RangePage::NoRoom);
                }
                resume = Some(k);
                break;
            }
            walked += 1;
            let (full, d) = decode(&k, &v)?;
            let suffix = k[HEAD_LEN + PREFIX.len()..].to_vec();
            let key = full.unwrap_or_else(|| suffix.clone());
            if key.as_slice() >= start && key.as_slice() < end {
                keys.push((key, suffix, d));
            }
        }
        let mut joined = 0u64;
        for (key, suffix, d) in keys {
            let key = key.as_slice();
            if !self.touched.contains_key(key) {
                self.hold(key, suffix, Some(d));
            }
            if let Some(previous) = d.rests_on
                && previous != tomb
            {
                self.leave(key, d, previous)?;
            } else if d.rests_on == Some(tomb) {
                // Already folded by an earlier page.
                continue;
            }
            self.set_descriptor(
                key,
                Some(Descriptor {
                    newest: tomb.seqno,
                    charge: 0,
                    rests_on: Some(tomb),
                    touched: false,
                    visible: false,
                }),
            );
            joined += 1;
        }
        if let Some(mut state) = self.tomb(tomb)? {
            // Bounded by the keys of one tree.
            state.untouched += joined;
            self.set_tomb(tomb, Some(state));
        }
        let ended = match resume {
            Some(k) => RangePage::From(k, walked),
            None => self.close_range(tomb, walked)?,
        };
        // Each key walked reserved a unit's moves.
        self.settle_moves(RANGE_PAGE_SHIFTS + UNIT_SHIFTS * walked, moved)?;
        Ok(ended)
    }

    fn close_range(&mut self, tomb: Tomb, walked: usize) -> StorageResult<RangePage> {
        self.settle(tomb)?;
        Ok(RangePage::Done(walked))
    }

    /// Build the fold's output: the descriptors, tombstones, dependents and
    /// summary moves, and the cursor record `cursor_key` = `cursor(visible)`
    /// after them, `visible` being the bytes a fixed snapshot needs after all
    /// of it given `base` before; the same after each commit it completed.
    /// Everything built is within what the fold reserved, which the output
    /// holds until it is applied.
    ///
    /// # Errors
    ///
    /// A summary bucket could not be read or would go out of range, or the
    /// snapshot bytes would leave `u64`.
    pub(crate) fn write(
        self,
        base: u64,
        cursor_key: &[u8],
        cursor: impl FnOnce(u64) -> Vec<u8>,
    ) -> StorageResult<Written<'t>> {
        let absolute = |delta: i128| {
            u64::try_from(i128::from(base) + delta).map_err(|_| {
                StorageError::Serialization(format!(
                    "material fold: {base} snapshot bytes moved by {delta}"
                ))
            })
        };
        // Reserved with each mark.
        let mut marks = Vec::with_capacity(self.marks.len());
        for &(commit, delta) in &self.marks {
            marks.push((commit, absolute(delta)?));
        }
        let visible = absolute(self.visible)?;
        let mut batch = lsm_tree::WriteBatch::new();
        let held = self.held;
        // Every record here is bounded by one key: the sum stays far below
        // u64::MAX.
        let mut bytes = 0u64;
        for (key, (before, after)) in self.touched {
            if before != after {
                let suffix = self.stored.get(&key).unwrap_or(&key);
                let k = descriptor_key(self.scope.head, suffix);
                match after {
                    Some(d) => {
                        let mut value = encode(d).to_vec();
                        // A key not stored as it is keeps itself in the value.
                        if suffix.as_slice() != key.as_slice() {
                            value.extend_from_slice(&key);
                        }
                        bytes += (k.len() + value.len()) as u64;
                        batch.insert(k.as_slice(), value.as_slice());
                    }
                    None => {
                        bytes += k.len() as u64;
                        batch.remove(k.as_slice());
                    }
                }
            }
        }
        for (tomb, (before, after)) in self.tombs {
            if before != after {
                let k = tomb.record_key(self.scope.head);
                bytes += (k.len() + TOMB_LEN) as u64;
                match after {
                    Some(t) => batch.insert(k.as_slice(), encode_tomb(t).as_slice()),
                    None => batch.remove(k.as_slice()),
                }
            }
        }
        for (k, added) in self.dependents {
            bytes += k.len() as u64;
            if added {
                batch.insert(k.as_slice(), &[][..]);
            } else {
                batch.remove(k.as_slice());
            }
        }
        bytes += self.moves.write(&self.scope, &mut batch)?;
        let record = cursor(visible);
        bytes += (cursor_key.len() + record.len()) as u64;
        batch.insert(cursor_key, record.as_slice());
        Ok(Written {
            batch,
            bytes,
            marks,
            held,
        })
    }
}

/// `key`'s descriptor as `scope` holds it.
#[cfg(test)]
pub(crate) fn read(scope: &Scope<'_>, key: &[u8]) -> StorageResult<Option<Descriptor>> {
    Ok(locate(scope, key, &BTreeMap::new())?.1)
}

/// newest, charge, then the tombstone a key rests on (a flag, seqno,
/// ordinal), whether it was touched since, and whether it has a value.
const DESCRIPTOR_LEN: usize = 8 + 8 + 1 + 8 + 4 + 1 + 1;

fn encode(d: Descriptor) -> [u8; DESCRIPTOR_LEN] {
    let mut out = [0u8; DESCRIPTOR_LEN];
    out[..8].copy_from_slice(&d.newest.to_le_bytes());
    out[8..16].copy_from_slice(&d.charge.to_le_bytes());
    if let Some(tomb) = d.rests_on {
        out[16] = 1;
        out[17..25].copy_from_slice(&tomb.seqno.to_le_bytes());
        out[25..29].copy_from_slice(&tomb.ordinal.to_le_bytes());
    }
    out[29] = u8::from(d.touched);
    out[30] = u8::from(d.visible);
    out
}

/// A descriptor, with the full key a key not stored as it is keeps after it.
fn decode(key: &[u8], value: &[u8]) -> StorageResult<(Option<Vec<u8>>, Descriptor)> {
    let bad = || {
        StorageError::Serialization(format!(
            "material descriptor {:02x?} holds {} bytes",
            &key[..key.len().min(32)],
            value.len()
        ))
    };
    let (fixed, full) = value.split_at_checked(DESCRIPTOR_LEN).ok_or_else(bad)?;
    let full = (!full.is_empty()).then(|| full.to_vec());
    let v = <[u8; DESCRIPTOR_LEN]>::try_from(fixed).map_err(|_| bad())?;
    let u64_at = |at: usize| {
        let mut b = [0u8; 8];
        b.copy_from_slice(&v[at..at + 8]);
        u64::from_le_bytes(b)
    };
    let rests_on = match v[16] {
        0 => None,
        1 => {
            let mut ordinal = [0u8; 4];
            ordinal.copy_from_slice(&v[25..29]);
            Some(Tomb {
                seqno: u64_at(17),
                ordinal: u32::from_le_bytes(ordinal),
            })
        }
        _ => return Err(bad()),
    };
    let flag = |at: usize| match v[at] {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(bad()),
    };
    Ok((
        full,
        Descriptor {
            newest: u64_at(0),
            charge: u64_at(8),
            rests_on,
            touched: flag(29)?,
            visible: flag(30)?,
        },
    ))
}

const TOMB_LEN: usize = 32;

fn encode_tomb(t: TombState) -> [u8; TOMB_LEN] {
    let mut out = [0u8; TOMB_LEN];
    out[..8].copy_from_slice(&t.bytes.to_le_bytes());
    out[8..16].copy_from_slice(&t.untouched.to_le_bytes());
    out[16..24].copy_from_slice(&t.touched.to_le_bytes());
    out[24..].copy_from_slice(&t.newest.to_le_bytes());
    out
}

fn decode_tomb(key: &[u8], value: &[u8]) -> StorageResult<TombState> {
    let v = <[u8; TOMB_LEN]>::try_from(value).map_err(|_| {
        StorageError::Serialization(format!(
            "material range tombstone {key:02x?} holds {} bytes",
            value.len()
        ))
    })?;
    let u64_at = |at: usize| {
        let mut b = [0u8; 8];
        b.copy_from_slice(&v[at..at + 8]);
        u64::from_le_bytes(b)
    };
    Ok(TombState {
        bytes: u64_at(0),
        untouched: u64_at(8),
        touched: u64_at(16),
        newest: u64_at(24),
    })
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests;
