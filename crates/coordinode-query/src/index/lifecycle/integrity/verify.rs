//! One pass over a shard's records checks several B-tree index generations
//! at once, both ways, at one pinned snapshot.
//!
//! Every record is read once. Its labels are read before anything else, and
//! only a record of a label some checked generation indexes is decoded; its
//! expected entries go to each such generation's run. Each run is then
//! sorted, spilled to scratch files when it outgrows its memory budget, and
//! merged against the generation's stored entries read in key order, so
//! neither side needs a point read per entry: an expected entry missing from
//! the index is missing, a stored one no record expects is extra.
//!
//! The records and the entries are read at the same snapshot, held by a pin
//! rather than an open transaction, so builds and DDL waiting for older
//! transactions do not wait for a check.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use coordinode_core::graph::node::NodeRecord;
use coordinode_core::index::derive::{
    EntryOwner, IndexInterpretation, append_entry_key_of, tuples,
};
use coordinode_core::index::encoding::{decode_entry_parts, decode_unique_entry_tuple};
use coordinode_modality::{LocalIndexStore, LocalNodeStore, Mismatch};
use coordinode_storage::engine::core::StorageEngine;
use xxhash_rust::xxh3::Xxh3;

use crate::index::definition::IndexDefinition;

/// A generation one pass checks: its definition and the interpretation its
/// entries are derived under.
pub(super) struct Target<'a> {
    pub def: &'a IndexDefinition,
    pub interpretation: &'a IndexInterpretation,
}

/// What a pass found for one generation.
#[derive(Debug, Default)]
pub(super) struct Findings {
    /// Records of the generation's label read, and stored entries read.
    pub checked: u64,
    /// Every disagreement, in the shape the repairs take.
    pub mismatches: Vec<Mismatch>,
}

/// What one pass read and did: the work it cost, not only what matched.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VerifyStats {
    /// Rows of the shard read, every label.
    pub source_rows: u64,
    /// Bytes of those rows as stored.
    pub source_bytes: u64,
    /// Rows decoded whole: of a label some checked generation indexes.
    pub decoded_rows: u64,
    /// Bytes of the rows decoded.
    pub decoded_bytes: u64,
    /// Entries the records give the checked generations.
    pub expected_entries: u64,
    /// Entries the checked generations store.
    pub stored_entries: u64,
    /// Sorted runs written to scratch files.
    pub spilled_runs: u64,
    /// Bytes written to scratch files.
    pub spilled_bytes: u64,
    /// Time reading and decoding the records.
    pub scan: Duration,
    /// Time sorting runs, spills included.
    pub sort: Duration,
    /// Time merging expected entries against stored ones.
    pub compare: Duration,
}

/// What a pass may hold: the memory of its runs and where it spills them.
pub(super) struct Budget {
    /// Bytes of expected entries all runs hold in memory together.
    pub memory_bytes: usize,
    /// Where sorted runs are written once memory is full; `None` refuses a
    /// pass that would need it.
    pub scratch: Option<PathBuf>,
}

/// Why a pass could not finish.
#[derive(Debug, thiserror::Error)]
pub(super) enum VerifyError {
    /// The store could not be read.
    #[error("read the store: {0}")]
    Store(String),
    /// A record could not be decoded.
    #[error("decode a record: {0}")]
    Decode(String),
    /// The expected entries outgrew memory and there is nowhere to spill.
    #[error(
        "the expected entries outgrew the {limit}-byte memory budget and this store has no \
         scratch directory to spill them to"
    )]
    NoScratch {
        /// The budget.
        limit: usize,
    },
    /// A scratch file could not be written or read back.
    #[error("scratch file: {0}")]
    Scratch(#[from] std::io::Error),
}

/// One expected entry: its key and the node holding it, with the version
/// that holds it for a temporal node.
#[derive(Clone, Copy)]
struct Item {
    start: u32,
    len: u32,
    node: u64,
    valid_from: Option<i64>,
}

/// The expected entries of one generation not yet spilled.
#[derive(Default)]
struct Run {
    bytes: Vec<u8>,
    items: Vec<Item>,
}

impl Run {
    fn held(&self) -> usize {
        self.bytes.len() + self.items.len() * core::mem::size_of::<Item>()
    }

    /// The entry of `owner` under `tuple` in `def`'s generation, its key
    /// written straight into the run.
    fn push_entry(&mut self, def: &IndexDefinition, tuple: &[u8], owner: EntryOwner) {
        // Keys are at most a record's values long, and a run is spilled
        // long before its bytes reach u32::MAX (the budget is far below).
        let start = self.bytes.len() as u32;
        append_entry_key_of(&mut self.bytes, def.generation, def.unique, tuple, owner);
        self.items.push(Item {
            start,
            len: self.bytes.len() as u32 - start,
            node: owner.node_id,
            valid_from: owner.valid_from,
        });
    }

    fn key(&self, item: &Item) -> &[u8] {
        &self.bytes[item.start as usize..(item.start + item.len) as usize]
    }

    /// Sort by key, then node, and drop repeats: a node whose versions hold
    /// one unique value expects one entry for it.
    fn sort(&mut self) {
        let bytes = &self.bytes;
        let key = |i: &Item| &bytes[i.start as usize..(i.start + i.len) as usize];
        self.items
            .sort_unstable_by(|a, b| key(a).cmp(key(b)).then(a.node.cmp(&b.node)));
        self.items
            .dedup_by(|a, b| a.node == b.node && key(a) == key(b));
    }

    fn clear(&mut self) {
        self.bytes.clear();
        self.items.clear();
    }
}

/// An expected entry read back from a run, owned.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Owned {
    key: Vec<u8>,
    node: u64,
    valid_from: Option<i64>,
}

impl Ord for Owned {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.cmp(&other.key).then(self.node.cmp(&other.node))
    }
}

impl PartialOrd for Owned {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// The length field that opens a run's trailer instead of an entry: no key
/// is this long.
const RUN_END: u32 = u32::MAX;

/// A sorted run spilled to a scratch file: `[len u32][key][node u64][vf
/// flag u8][vf i64?]` per entry, then `[RUN_END][count u64][xxh3 u64]` over
/// every entry byte, so a run cut short or damaged fails to read instead of
/// reading as a shorter one.
///
/// The run is charged to the owned footprint of `dir`'s endpoint before it
/// is written; the charge goes with the file, which has no name.
fn spill(
    run: &Run,
    dir: &Path,
    footprints: &coordinode_storage::engine::footprint::Footprints,
) -> Result<(std::fs::File, u64, Option<Charge>), VerifyError> {
    std::fs::create_dir_all(dir)?;
    let charge = footprints.unnamed(dir);
    if let Some(charge) = &charge {
        // Each entry's fixed fields and key, then the trailer.
        let size: u64 = run
            .items
            .iter()
            .map(|item| 4 + u64::from(item.len) + 8 + if item.valid_from.is_some() { 9 } else { 1 })
            .sum::<u64>()
            + 4
            + 8
            + 8;
        charge.grow_to(size)?;
    }
    let file = tempfile::tempfile_in(dir)?;
    let mut out = BufWriter::new(file);
    let mut hash = Xxh3::new();
    let mut put = |bytes: &[u8], out: &mut BufWriter<std::fs::File>| {
        hash.update(bytes);
        out.write_all(bytes)
    };
    let mut written = 0u64;
    for item in &run.items {
        let key = run.key(item);
        put(&item.len.to_be_bytes(), &mut out)?;
        put(key, &mut out)?;
        put(&item.node.to_be_bytes(), &mut out)?;
        match item.valid_from {
            Some(vf) => {
                put(&[1], &mut out)?;
                put(&vf.to_be_bytes(), &mut out)?;
                written += 9;
            }
            None => {
                put(&[0], &mut out)?;
                written += 1;
            }
        }
        // A key's length plus the fixed fields, far below u64::MAX in total.
        written += 4 + u64::from(item.len) + 8;
    }
    out.write_all(&RUN_END.to_be_bytes())?;
    out.write_all(&(run.items.len() as u64).to_be_bytes())?;
    out.write_all(&hash.digest().to_be_bytes())?;
    written += 4 + 8 + 8;
    let mut file = out.into_inner().map_err(|e| e.into_error())?;
    use std::io::Seek as _;
    file.seek(std::io::SeekFrom::Start(0))?;
    Ok((file, written, charge))
}

/// What a scratch run is charged on its endpoint's footprint.
type Charge = std::sync::Arc<coordinode_storage::engine::footprint::FileCharge>;

/// A spilled run read back in order, checked against its trailer.
struct Spilled {
    reader: BufReader<std::fs::File>,
    hash: Xxh3,
    read: u64,
    /// Given back with the file, when the run is dropped.
    _charge: Option<Charge>,
}

impl Spilled {
    fn new(file: std::fs::File, charge: Option<Charge>) -> Self {
        Self {
            reader: BufReader::new(file),
            hash: Xxh3::new(),
            read: 0,
            _charge: charge,
        }
    }

    fn take(&mut self, buf: &mut [u8]) -> Result<(), VerifyError> {
        match self.reader.read_exact(buf) {
            Ok(()) => {
                self.hash.update(buf);
                Ok(())
            }
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                Err(damaged("it ends before its trailer"))
            }
            Err(e) => Err(e.into()),
        }
    }

    fn next(&mut self) -> Result<Option<Owned>, VerifyError> {
        let mut len = [0u8; 4];
        if let Err(e) = self.reader.read_exact(&mut len) {
            return Err(if e.kind() == std::io::ErrorKind::UnexpectedEof {
                damaged("it ends before its trailer")
            } else {
                e.into()
            });
        }
        if u32::from_be_bytes(len) == RUN_END {
            let mut count = [0u8; 8];
            let mut digest = [0u8; 8];
            self.reader.read_exact(&mut count)?;
            self.reader.read_exact(&mut digest)?;
            if u64::from_be_bytes(count) != self.read {
                return Err(damaged("its entry count disagrees with its trailer"));
            }
            if u64::from_be_bytes(digest) != self.hash.digest() {
                return Err(damaged("its checksum disagrees with its trailer"));
            }
            return Ok(None);
        }
        self.hash.update(&len);
        let mut key = vec![0u8; u32::from_be_bytes(len) as usize];
        self.take(&mut key)?;
        let mut node = [0u8; 8];
        self.take(&mut node)?;
        let mut flag = [0u8; 1];
        self.take(&mut flag)?;
        let valid_from = match flag[0] {
            0 => None,
            1 => {
                let mut vf = [0u8; 8];
                self.take(&mut vf)?;
                Some(i64::from_be_bytes(vf))
            }
            other => return Err(damaged(&format!("an entry has version flag {other}"))),
        };
        self.read += 1;
        Ok(Some(Owned {
            key,
            node: u64::from_be_bytes(node),
            valid_from,
        }))
    }
}

/// A scratch run that does not read back as written.
fn damaged(why: &str) -> VerifyError {
    VerifyError::Scratch(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("a scratch run is damaged: {why}"),
    ))
}

/// The expected entries of one generation: the runs it spilled and the one
/// still in memory, merged into one sorted stream without repeats.
struct Expected {
    spilled: Vec<Spilled>,
    memory: Run,
    at: usize,
    heap: BinaryHeap<core::cmp::Reverse<(Owned, usize)>>,
    /// Whether an entry was given out, which `next_into`'s buffer then holds.
    given: bool,
    primed: bool,
}

/// The source index [`Expected::heap`] uses for the run still in memory.
const MEMORY_RUN: usize = usize::MAX;

impl Expected {
    fn new(spilled: Vec<Spilled>, memory: Run) -> Self {
        Self {
            spilled,
            memory,
            at: 0,
            heap: BinaryHeap::new(),
            given: false,
            primed: false,
        }
    }

    fn pull(&mut self, source: usize) -> Result<Option<Owned>, VerifyError> {
        if source == MEMORY_RUN {
            let Some(item) = self.memory.items.get(self.at).copied() else {
                return Ok(None);
            };
            self.at += 1;
            return Ok(Some(Owned {
                key: self.memory.key(&item).to_vec(),
                node: item.node,
                valid_from: item.valid_from,
            }));
        }
        self.spilled[source].next()
    }

    /// The next entry, written into `out` (whose key buffer is reused);
    /// `false` once every entry was given out. A generation that never
    /// spilled reads its one sorted run in place, allocating nothing.
    fn next_into(&mut self, out: &mut Owned) -> Result<bool, VerifyError> {
        if self.spilled.is_empty() {
            // One run, sorted without repeats.
            let Some(item) = self.memory.items.get(self.at).copied() else {
                return Ok(false);
            };
            self.at += 1;
            out.key.clear();
            out.key.extend_from_slice(self.memory.key(&item));
            out.node = item.node;
            out.valid_from = item.valid_from;
            return Ok(true);
        }
        if !self.primed {
            self.primed = true;
            for source in (0..self.spilled.len()).chain(core::iter::once(MEMORY_RUN)) {
                if let Some(first) = self.pull(source)? {
                    self.heap.push(core::cmp::Reverse((first, source)));
                }
            }
        }
        loop {
            let Some(core::cmp::Reverse((item, source))) = self.heap.pop() else {
                return Ok(false);
            };
            if let Some(next) = self.pull(source)? {
                self.heap.push(core::cmp::Reverse((next, source)));
            }
            // Runs spilled apart may each hold one node's entry; `out` still
            // holds the entry given out last.
            if self.given && *out == item {
                continue;
            }
            self.given = true;
            *out = item;
            return Ok(true);
        }
    }
}

/// The checked generations of one label and what of its records they read.
struct Projection<'t> {
    label: &'t str,
    /// The generations, by their position in the pass's targets.
    slots: Vec<usize>,
    /// The declared properties they read, by field id.
    fields: Vec<u32>,
    /// The undeclared properties they read, by name.
    names: Vec<&'t str>,
}

impl<'t> Projection<'t> {
    /// One projection per label `targets` index.
    fn of(targets: &'t [Target<'_>]) -> Vec<Self> {
        let mut out: Vec<Self> = Vec::new();
        for (i, target) in targets.iter().enumerate() {
            let label = target.def.label.as_str();
            let at = match out.iter().position(|p| p.label == label) {
                Some(at) => at,
                None => {
                    out.push(Self {
                        label,
                        slots: Vec::new(),
                        fields: Vec::new(),
                        names: Vec::new(),
                    });
                    out.len() - 1
                }
            };
            let projection = &mut out[at];
            projection.slots.push(i);
            let interpretation = target.interpretation;
            let read = interpretation
                .properties
                .iter()
                .chain(interpretation.filter.as_ref().map(|f| f.property()));
            for property in read {
                // A property is read by id and, missing there, by name.
                if let Some(field) = property.field {
                    if !projection.fields.contains(&field) {
                        projection.fields.push(field);
                    }
                }
                if !projection.names.contains(&property.name.as_str()) {
                    projection.names.push(property.name.as_str());
                }
            }
        }
        out
    }
}

/// A generation being checked: its target, its runs and its findings.
struct Slot<'t, 'a> {
    target: &'t Target<'a>,
    memory: Run,
    spilled: Vec<Spilled>,
    findings: Findings,
}

/// Check `targets` against the records of shard `shard_id`, both ways, at
/// one snapshot. `Ok(None)` when `stop` is set before the pass ends.
///
/// # Errors
///
/// The store could not be read, a record not decoded, or the expected
/// entries could not be held or spilled.
pub(super) fn verify(
    engine: &StorageEngine,
    shard_id: u16,
    targets: &[Target<'_>],
    budget: &Budget,
    stop: &AtomicBool,
) -> Result<Option<(Vec<Findings>, VerifyStats)>, VerifyError> {
    let mut stats = VerifyStats::default();
    let (snapshot, _pin) = engine.pin_snapshot();
    let mut slots: Vec<Slot<'_, '_>> = targets
        .iter()
        .map(|target| Slot {
            target,
            memory: Run::default(),
            spilled: Vec::new(),
            findings: Findings::default(),
        })
        .collect();

    // Per label, the checked generations indexing it and the properties
    // they read together: a record decodes only those, every other value
    // skipped unread however large.
    let projections = Projection::of(targets);
    // What each type implies, as the pinned snapshot's catalog says.
    let mut catalog = crate::index::membership::CatalogMembership::at(engine, snapshot);

    // The records, once.
    let scanning = Instant::now();
    let rows = LocalNodeStore::encoded_rows_at(engine, snapshot, shard_id)
        .map_err(|e| VerifyError::Store(e.to_string()))?;
    for (read, row) in rows.enumerate() {
        if read % 1024 == 0 && stop.load(core::sync::atomic::Ordering::Acquire) {
            return Ok(None);
        }
        let (owner, value) = row.map_err(|e| VerifyError::Store(e.to_string()))?;
        stats.source_rows += 1;
        stats.source_bytes += value.len() as u64;
        // The labels alone first: a record no checked generation indexes is
        // never decoded, however large it is.
        let labels = NodeRecord::labels_from_msgpack(&value)
            .map_err(|e| VerifyError::Decode(e.to_string()))?;
        // An index on a label covers every node carrying it, as its type,
        // implied or auxiliary.
        let membership = catalog
            .of_stored(&labels)
            .map_err(|e| VerifyError::Store(e.to_string()))?;
        let carries = |p: &&Projection<'_>| membership.contains(p.label);
        let mut carried = projections.iter().filter(carries);
        let Some(first) = carried.next() else {
            continue;
        };
        // One checked label reads its projection; several read the record
        // whole rather than merging their projections per row.
        let record = match carried.next() {
            None => NodeRecord::project_from_msgpack(&value, &first.fields, &first.names),
            Some(_) => NodeRecord::from_msgpack(&value),
        }
        .map_err(|e| VerifyError::Decode(e.to_string()))?;
        stats.decoded_rows += 1;
        stats.decoded_bytes += value.len() as u64;
        for &i in projections.iter().filter(carries).flat_map(|p| &p.slots) {
            let slot = &mut slots[i];
            slot.findings.checked += 1;
            let Some(values) = slot.target.interpretation.record_membership(&record) else {
                continue;
            };
            for tuple in tuples(&values) {
                slot.memory.push_entry(slot.target.def, &tuple, owner);
                stats.expected_entries += 1;
            }
        }
        let held: usize = slots.iter().map(|s| s.memory.held()).sum();
        if held > budget.memory_bytes {
            let dir = budget.scratch.as_deref().ok_or(VerifyError::NoScratch {
                limit: budget.memory_bytes,
            })?;
            let sorting = Instant::now();
            // The largest run goes first: it frees the most.
            if let Some(slot) = slots.iter_mut().max_by_key(|s| s.memory.held()) {
                slot.memory.sort();
                let (file, written, charge) = spill(&slot.memory, dir, engine.footprints())?;
                slot.memory.clear();
                slot.spilled.push(Spilled::new(file, charge));
                stats.spilled_runs += 1;
                stats.spilled_bytes += written;
            }
            stats.sort += sorting.elapsed();
        }
    }
    // The spills' time was spent inside the scan's interval, so it is never
    // more than the interval.
    stats.scan = scanning.elapsed() - stats.sort;

    // Each generation's entries against what its records expect.
    let mut out = Vec::with_capacity(slots.len());
    for slot in slots {
        if stop.load(core::sync::atomic::Ordering::Acquire) {
            return Ok(None);
        }
        let Slot {
            target,
            mut memory,
            spilled,
            mut findings,
        } = slot;
        let sorting = Instant::now();
        memory.sort();
        stats.sort += sorting.elapsed();
        let comparing = Instant::now();
        let mut expected = Expected::new(spilled, memory);
        compare(
            engine,
            snapshot,
            target,
            &mut expected,
            &mut findings,
            &mut stats,
        )?;
        stats.compare += comparing.elapsed();
        out.push(findings);
    }
    Ok(Some((out, stats)))
}

/// Merge `expected` against `target`'s stored entries at `snapshot`.
fn compare(
    engine: &StorageEngine,
    snapshot: u64,
    target: &Target<'_>,
    expected: &mut Expected,
    findings: &mut Findings,
    stats: &mut VerifyStats,
) -> Result<(), VerifyError> {
    let def = target.def;
    let generation = def.generation;
    // Shared slices of the stored pairs, read in key order.
    let mut stored = LocalIndexStore::new(engine)
        .stored_entries_at(snapshot, def)
        .map_err(|e| VerifyError::Store(e.to_string()))?
        .map(|pair| pair.map_err(|e| VerifyError::Store(e.to_string())));
    if def.unique {
        compare_unique(generation, &mut stored, expected, findings, stats)
    } else {
        compare_plain(generation, &mut stored, expected, findings, stats)
    }
}

/// A non-unique generation: one entry per value and owner, the key is the
/// whole membership.
fn compare_plain<I, K, V>(
    generation: coordinode_core::index::identity::GenerationId,
    stored: &mut I,
    expected: &mut Expected,
    findings: &mut Findings,
    stats: &mut VerifyStats,
) -> Result<(), VerifyError>
where
    I: Iterator<Item = Result<(K, V), VerifyError>>,
    K: core::ops::Deref<Target = [u8]>,
{
    let next_stored = |stored: &mut I| stored.next().transpose();
    let mut want = Want::first(expected)?;
    let mut have = next_stored(stored)?;
    loop {
        // A stored key that is no entry of this generation's shape names no
        // owner a repair could act on: it is skipped, as it is skipped when
        // read entry by entry.
        if let Some((key, _)) = &have {
            if decode_entry_parts(generation, key).is_none() {
                have = next_stored(stored)?;
                continue;
            }
        }
        match (want.get(), &have) {
            (None, None) => return Ok(()),
            (Some(w), Some((key, _))) if w.key.as_slice() == &**key => {
                stats.stored_entries += 1;
                findings.checked += 1;
                want.advance(expected)?;
                have = next_stored(stored)?;
            }
            (Some(w), Some((key, _))) if w.key.as_slice() < &**key => {
                push_missing(generation, w, findings);
                want.advance(expected)?;
            }
            (Some(w), None) => {
                push_missing(generation, w, findings);
                want.advance(expected)?;
            }
            (_, Some((key, _))) => {
                stats.stored_entries += 1;
                findings.checked += 1;
                if let Some((tuple, node, valid_from)) = decode_entry_parts(generation, key) {
                    findings.mismatches.push(Mismatch::Extra {
                        node,
                        valid_from,
                        tuple: tuple.to_vec(),
                    });
                }
                have = next_stored(stored)?;
            }
        }
    }
}

/// The expected entry a merge is at, in a buffer reused from entry to entry.
struct Want {
    entry: Owned,
    some: bool,
}

impl Want {
    fn first(expected: &mut Expected) -> Result<Self, VerifyError> {
        let mut want = Self {
            entry: Owned::default(),
            some: false,
        };
        want.advance(expected)?;
        Ok(want)
    }

    fn get(&self) -> Option<&Owned> {
        self.some.then_some(&self.entry)
    }

    fn advance(&mut self, expected: &mut Expected) -> Result<(), VerifyError> {
        self.some = expected.next_into(&mut self.entry)?;
        Ok(())
    }
}

fn push_missing(
    generation: coordinode_core::index::identity::GenerationId,
    want: &Owned,
    findings: &mut Findings,
) {
    if let Some((tuple, node, valid_from)) = decode_entry_parts(generation, &want.key) {
        findings.mismatches.push(Mismatch::Missing {
            node,
            valid_from,
            tuple: tuple.to_vec(),
        });
    }
}

/// A unique generation: one entry per value, naming its holder. Every node
/// whose record holds a value expects to be its holder; two such nodes are
/// records breaking the constraint, never an entry's damage.
fn compare_unique<I, K, V>(
    generation: coordinode_core::index::identity::GenerationId,
    stored: &mut I,
    expected: &mut Expected,
    findings: &mut Findings,
    stats: &mut VerifyStats,
) -> Result<(), VerifyError>
where
    I: Iterator<Item = Result<(K, V), VerifyError>>,
    K: core::ops::Deref<Target = [u8]>,
    V: core::ops::Deref<Target = [u8]>,
{
    let next_stored = |stored: &mut I| stored.next().transpose();
    let mut want = Want::first(expected)?;
    let mut have = next_stored(stored)?;
    // The value both sides are at, reused from value to value.
    let mut key: Vec<u8> = Vec::new();
    let mut holders: Vec<(u64, Option<i64>)> = Vec::new();
    loop {
        // The smaller of the two keys.
        let at: &[u8] = match (want.get(), &have) {
            (None, None) => return Ok(()),
            (Some(w), None) => &w.key,
            (None, Some((k, _))) => k,
            (Some(w), Some((k, _))) => {
                if w.key.as_slice() <= &**k {
                    &w.key
                } else {
                    k
                }
            }
        };
        key.clear();
        key.extend_from_slice(at);
        // Copied out only for a disagreement, which a repair names it in.
        let Some(tuple) = decode_unique_entry_tuple(generation, &key) else {
            // No value of this generation's shape: nothing a repair can name.
            if want.get().is_some_and(|w| w.key == key) {
                want.advance(expected)?;
            }
            if have.as_ref().is_some_and(|(k, _)| **k == *key) {
                have = next_stored(stored)?;
            }
            continue;
        };
        // Every node expecting this value, with the version that holds it.
        holders.clear();
        while let Some(w) = want.get().filter(|w| w.key == key) {
            holders.push((w.node, w.valid_from));
            want.advance(expected)?;
        }
        // The stored holder: `Some(None)` for one that does not decode.
        let stored_holder = match have.as_ref().filter(|(k, _)| **k == *key) {
            Some((_, value)) => {
                stats.stored_entries += 1;
                findings.checked += 1;
                let holder = <[u8; 8]>::try_from(&**value).ok().map(u64::from_be_bytes);
                have = next_stored(stored)?;
                Some(holder)
            }
            None => None,
        };
        let holds = |node: u64| holders.iter().any(|(n, _)| *n == node);
        for &(node, valid_from) in &holders {
            match stored_holder {
                Some(Some(h)) if h == node => {}
                Some(Some(h)) if holds(h) => findings.mismatches.push(Mismatch::SourceDuplicate {
                    nodes: [h.min(node), h.max(node)],
                    tuple: tuple.to_vec(),
                }),
                _ => findings.mismatches.push(Mismatch::Missing {
                    node,
                    valid_from,
                    tuple: tuple.to_vec(),
                }),
            }
        }
        match stored_holder {
            Some(None) => findings.mismatches.push(Mismatch::Extra {
                node: u64::MAX,
                valid_from: None,
                tuple: tuple.to_vec(),
            }),
            Some(Some(h)) if !holds(h) => findings.mismatches.push(Mismatch::Extra {
                node: h,
                valid_from: None,
                tuple: tuple.to_vec(),
            }),
            _ => {}
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
