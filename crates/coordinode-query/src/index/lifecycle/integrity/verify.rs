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
use coordinode_core::index::derive::{IndexInterpretation, entry, tuples};
use coordinode_core::index::encoding::{decode_entry_parts, decode_unique_entry_tuple};
use coordinode_modality::{LocalIndexStore, LocalNodeStore, Mismatch};
use coordinode_storage::engine::core::StorageEngine;

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

    fn push(&mut self, key: &[u8], node: u64, valid_from: Option<i64>) {
        // Keys are at most a record's values long, and a run is spilled
        // long before its bytes reach u32::MAX (the budget is far below).
        let start = self.bytes.len() as u32;
        self.bytes.extend_from_slice(key);
        self.items.push(Item {
            start,
            len: key.len() as u32,
            node,
            valid_from,
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
#[derive(Debug, Clone, PartialEq, Eq)]
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

/// A sorted run spilled to a scratch file: `[len u32][key][node u64][vf
/// flag u8][vf i64?]` per entry.
fn spill(run: &Run, dir: &Path) -> Result<(std::fs::File, u64), VerifyError> {
    std::fs::create_dir_all(dir)?;
    let file = tempfile::tempfile_in(dir)?;
    let mut out = BufWriter::new(file);
    let mut written = 0u64;
    for item in &run.items {
        let key = run.key(item);
        out.write_all(&item.len.to_be_bytes())?;
        out.write_all(key)?;
        out.write_all(&item.node.to_be_bytes())?;
        match item.valid_from {
            Some(vf) => {
                out.write_all(&[1])?;
                out.write_all(&vf.to_be_bytes())?;
                written += 9;
            }
            None => {
                out.write_all(&[0])?;
                written += 1;
            }
        }
        // A key's length plus the fixed fields, far below u64::MAX in total.
        written += 4 + u64::from(item.len) + 8;
    }
    let mut file = out.into_inner().map_err(|e| e.into_error())?;
    use std::io::Seek as _;
    file.seek(std::io::SeekFrom::Start(0))?;
    Ok((file, written))
}

/// A spilled run read back in order.
struct Spilled {
    reader: BufReader<std::fs::File>,
}

impl Spilled {
    fn next(&mut self) -> Result<Option<Owned>, VerifyError> {
        let mut len = [0u8; 4];
        match self.reader.read_exact(&mut len) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e.into()),
        }
        let mut key = vec![0u8; u32::from_be_bytes(len) as usize];
        self.reader.read_exact(&mut key)?;
        let mut node = [0u8; 8];
        self.reader.read_exact(&mut node)?;
        let mut flag = [0u8; 1];
        self.reader.read_exact(&mut flag)?;
        let valid_from = if flag[0] == 1 {
            let mut vf = [0u8; 8];
            self.reader.read_exact(&mut vf)?;
            Some(i64::from_be_bytes(vf))
        } else {
            None
        };
        Ok(Some(Owned {
            key,
            node: u64::from_be_bytes(node),
            valid_from,
        }))
    }
}

/// The expected entries of one generation: the runs it spilled and the one
/// still in memory, merged into one sorted stream without repeats.
struct Expected {
    spilled: Vec<Spilled>,
    memory: Run,
    at: usize,
    heap: BinaryHeap<core::cmp::Reverse<(Owned, usize)>>,
    last: Option<Owned>,
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
            last: None,
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

    fn next(&mut self) -> Result<Option<Owned>, VerifyError> {
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
                return Ok(None);
            };
            if let Some(next) = self.pull(source)? {
                self.heap.push(core::cmp::Reverse((next, source)));
            }
            // Runs spilled apart may each hold one node's entry.
            if self.last.as_ref() == Some(&item) {
                continue;
            }
            self.last = Some(item.clone());
            return Ok(Some(item));
        }
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

    // The records, once.
    let scanning = Instant::now();
    let rows = LocalNodeStore::encoded_rows_at(engine, snapshot, shard_id)
        .map_err(|e| VerifyError::Store(e.to_string()))?;
    let mut matching: Vec<usize> = Vec::with_capacity(slots.len());
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
        let primary = labels.first().map_or("", String::as_str);
        matching.clear();
        matching.extend(
            slots
                .iter()
                .enumerate()
                .filter(|(_, slot)| slot.target.def.label == primary)
                .map(|(i, _)| i),
        );
        if matching.is_empty() {
            continue;
        }
        let record =
            NodeRecord::from_msgpack(&value).map_err(|e| VerifyError::Decode(e.to_string()))?;
        stats.decoded_rows += 1;
        stats.decoded_bytes += value.len() as u64;
        for &i in &matching {
            let slot = &mut slots[i];
            slot.findings.checked += 1;
            let Some(values) = slot.target.interpretation.record_membership(&record) else {
                continue;
            };
            let def = slot.target.def;
            for tuple in tuples(&values) {
                let key = entry(def.generation, def.unique, &tuple, owner).0;
                slot.memory.push(&key, owner.node_id, owner.valid_from);
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
                let (file, written) = spill(&slot.memory, dir)?;
                slot.memory.clear();
                slot.spilled.push(Spilled {
                    reader: BufReader::new(file),
                });
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
    let mut want = expected.next()?;
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
        match (&want, &have) {
            (None, None) => return Ok(()),
            (Some(w), Some((key, _))) if w.key.as_slice() == &**key => {
                stats.stored_entries += 1;
                findings.checked += 1;
                want = expected.next()?;
                have = next_stored(stored)?;
            }
            (Some(w), Some((key, _))) if w.key.as_slice() < &**key => {
                push_missing(generation, w, findings);
                want = expected.next()?;
            }
            (Some(w), None) => {
                push_missing(generation, w, findings);
                want = expected.next()?;
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
    let mut want = expected.next()?;
    let mut have = next_stored(stored)?;
    // The value both sides are at, reused from value to value.
    let mut key: Vec<u8> = Vec::new();
    let mut holders: Vec<(u64, Option<i64>)> = Vec::new();
    loop {
        // The smaller of the two keys.
        let at: &[u8] = match (&want, &have) {
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
        let Some(tuple) = decode_unique_entry_tuple(generation, &key).map(<[u8]>::to_vec) else {
            // No value of this generation's shape: nothing a repair can name.
            if want.as_ref().is_some_and(|w| w.key == key) {
                want = expected.next()?;
            }
            if have.as_ref().is_some_and(|(k, _)| **k == *key) {
                have = next_stored(stored)?;
            }
            continue;
        };
        // Every node expecting this value, with the version that holds it.
        holders.clear();
        while let Some(w) = want.as_ref().filter(|w| w.key == key) {
            holders.push((w.node, w.valid_from));
            want = expected.next()?;
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
                    tuple: tuple.clone(),
                }),
                _ => findings.mismatches.push(Mismatch::Missing {
                    node,
                    valid_from,
                    tuple: tuple.clone(),
                }),
            }
        }
        match stored_holder {
            Some(None) => findings.mismatches.push(Mismatch::Extra {
                node: u64::MAX,
                valid_from: None,
                tuple,
            }),
            Some(Some(h)) if !holds(h) => findings.mismatches.push(Mismatch::Extra {
                node: h,
                valid_from: None,
                tuple,
            }),
            _ => {}
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
