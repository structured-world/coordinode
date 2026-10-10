//! Member-local installations of index generations in the index partition.
//!
//! Replicated effects and every caller above the engine address an index
//! entry by its logical key, `tag / GenerationId:u64_BE / …`. In this
//! member's index tree the entry lives at `tag / InstallationId:u64_BE / …`:
//! the local installation takes the generation's slot, so the prefix keeps
//! its nine bytes. The engine translates at its boundary, so nothing above it
//! sees an installation, and one member can give a generation a new local
//! copy without a new global generation.
//!
//! The catalog lives in the same tree under the reserved namespace, so a
//! binding persists with the entries written under it and never behind them:
//! it is inserted before the first of them, and a flush persists memtables as
//! a prefix in seal order.
//!
//! - `0x00 'i' 'n'` holds the next unallocated installation (u64 BE).
//!   Installations are never reused: a whole-partition rebuild keeps the
//!   catalog.
//! - `0x00 'i' 'b' <GenerationId u64 BE>` holds the generation's published
//!   installation and the lowest seqno its history covers (u64 BE each).
//! - `0x00 'i' 's' <GenerationId u64 BE>` holds a replacement being prepared:
//!   its installation, the history it will cover and whether its imported
//!   contents are complete (see the `replace` module).
//! - `0x00 'i' 'r' <InstallationId u64 BE>` names a retired installation and
//!   its generation: its entries are deleted, and nothing reads it again.
//!
//! Only keys whose first byte is a generation tag ([`GENERATION_TAGS`]) are
//! translated; every other family of the partition is stored as written.

use std::ops::Bound;
use std::sync::Arc;

use coordinode_core::index::encoding::{GENERATION_PREFIX_LEN, GENERATION_TAGS};
use lsm_tree::{AbstractTree, AnyTree, Guard, SeqNo};
use rustc_hash::FxHashMap;

use crate::engine::{StorageGuard, StorageIter};
use crate::error::{StorageError, StorageResult};

/// The catalog's reserved namespace: `0x00 'i' …`.
const CATALOG_PREFIX: [u8; 2] = [0x00, b'i'];
/// The record of the next unallocated installation.
const NEXT_KEY: [u8; 3] = [0x00, b'i', b'n'];
/// Start of the binding records, `0x00 'i' 'b' <generation>`.
const BINDING_PREFIX: [u8; 3] = [0x00, b'i', b'b'];
/// Exclusive end of the binding records.
const BINDING_END: [u8; 3] = [0x00, b'i', b'c'];
/// Start of the retired-installation records, `0x00 'i' 'r' <installation>`.
const RETIRED_PREFIX: [u8; 3] = [0x00, b'i', b'r'];
/// Exclusive end of the retired-installation records.
const RETIRED_END: [u8; 3] = [0x00, b'i', b's'];
/// Start of the replacement records, `0x00 'i' 's' <generation>`.
const STAGING_PREFIX: [u8; 3] = [0x00, b'i', b's'];
/// Exclusive end of the replacement records.
const STAGING_END: [u8; 3] = [0x00, b'i', b't'];
/// The first installation handed out; 0 is never one.
const FIRST_INSTALLATION: u64 = 1;

mod replace;
pub use replace::{GenerationHistory, HistoryEntry};

/// Whether `key` is one of the installation catalog's own records. They
/// belong to this member and never travel with the partition's rows.
#[must_use]
pub fn is_catalog_key(key: &[u8]) -> bool {
    key.starts_with(&CATALOG_PREFIX)
}

/// A catalog key: one of the record prefixes and a u64.
fn record_key(prefix: [u8; 3], id: u64) -> [u8; 11] {
    let mut key = [0u8; 11];
    key[..3].copy_from_slice(&prefix);
    key[3..].copy_from_slice(&id.to_be_bytes());
    key
}

fn binding_key(generation: u64) -> [u8; 11] {
    record_key(BINDING_PREFIX, generation)
}

/// A binding's value: the installation and the lowest seqno its history
/// covers.
fn encode_binding(installation: u64, history_from: SeqNo) -> [u8; 16] {
    let mut value = [0u8; 16];
    value[..8].copy_from_slice(&installation.to_be_bytes());
    value[8..].copy_from_slice(&history_from.to_be_bytes());
    value
}

fn decode_pair(what: &str, value: &[u8]) -> StorageResult<(u64, u64)> {
    let (a, b) = value
        .split_first_chunk::<8>()
        .filter(|(_, rest)| rest.len() == 8)
        .ok_or_else(|| {
            StorageError::InstallationCatalog(format!(
                "{what} holds {} bytes, expected 16",
                value.len()
            ))
        })?;
    Ok((u64::from_be_bytes(*a), decode_u64(what, b)?))
}

/// The tag of a generation-domain key, `None` for any other key.
#[inline]
fn domain_tag(key: &[u8]) -> Option<u8> {
    let tag = *key.first()?;
    GENERATION_TAGS.contains(&tag).then_some(tag)
}

/// The id in the slot after the tag of a domain key at least a prefix long.
#[inline]
fn slot(key: &[u8]) -> Option<u64> {
    let bytes: [u8; 8] = key.get(1..GENERATION_PREFIX_LEN)?.try_into().ok()?;
    Some(u64::from_be_bytes(bytes))
}

/// `key` with its slot replaced by `id`, written into `out`.
#[inline]
fn with_slot(key: &[u8], id: u64, out: &mut Vec<u8>) {
    out.clear();
    out.extend_from_slice(key);
    out[1..GENERATION_PREFIX_LEN].copy_from_slice(&id.to_be_bytes());
}

/// Write `generation` into the slot of the generation key `key` read from
/// its installation, giving the key back its logical form.
#[inline]
pub(crate) fn put_generation(key: &mut [u8], generation: u64) {
    if let Some(slot) = key.get_mut(1..GENERATION_PREFIX_LEN) {
        slot.copy_from_slice(&generation.to_be_bytes());
    }
}

/// Scan the logical range `(lo, hi)` of the index tree at `seqno`: each
/// piece from its own physical range, in logical order, every key given
/// back in logical form.
pub(crate) fn scan(
    tree: &AnyTree,
    bindings: &Bindings,
    lo: Bound<&[u8]>,
    hi: Bound<&[u8]>,
    seqno: SeqNo,
) -> StorageIter {
    let tree = tree.clone();
    Box::new(
        bindings
            .pieces(lo, hi)
            .into_iter()
            .flat_map(move |piece| -> StorageIter {
                match piece {
                    Piece::Raw { lo, hi } => {
                        Box::new(tree.range((lo, hi), seqno, None).map(StorageGuard::raw))
                    }
                    Piece::Domain { generation, lo, hi } => Box::new(
                        tree.range((lo, hi), seqno, None)
                            .map(move |guard| StorageGuard::of_generation(guard, generation)),
                    ),
                }
            }),
    )
}

/// Scan the keys of the index tree starting with the logical `prefix` at
/// `seqno`. A prefix inside one generation is one physical prefix scan (its
/// prefix filter applies); a shorter generation prefix, or the empty prefix
/// (the whole user keyspace), comes apart per generation.
pub(crate) fn scan_prefix(
    tree: &AnyTree,
    bindings: &Bindings,
    prefix: &[u8],
    seqno: SeqNo,
) -> StorageIter {
    match prefix.first() {
        None => scan(
            tree,
            bindings,
            Bound::Included(crate::engine::coverage::USER_KEYSPACE_START),
            Bound::Unbounded,
            seqno,
        ),
        // The reserved namespace is never part of a user scan.
        Some(0x00) => Box::new(std::iter::empty()),
        Some(_) if domain_tag(prefix).is_none() => {
            Box::new(tree.prefix(prefix, seqno, None).map(StorageGuard::raw))
        }
        Some(_) if prefix.len() >= GENERATION_PREFIX_LEN => {
            let mut buf = Vec::with_capacity(prefix.len());
            let generation = slot(prefix);
            match (bindings.locate(prefix, &mut buf), generation) {
                (Located::Bound(physical), Some(generation)) => Box::new(
                    tree.prefix(physical, seqno, None)
                        .map(move |guard| StorageGuard::of_generation(guard, generation)),
                ),
                _ => Box::new(std::iter::empty()),
            }
        }
        Some(_) => {
            let end = successor(prefix);
            scan(
                tree,
                bindings,
                Bound::Included(prefix),
                end.as_deref().map_or(Bound::Unbounded, Bound::Excluded),
                seqno,
            )
        }
    }
}

/// The first key past every key starting with `prefix`; `None` when every
/// byte is `0xFF`.
fn successor(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut end = prefix.to_vec();
    while let Some(last) = end.pop() {
        if last < 0xFF {
            end.push(last + 1);
            return Some(end);
        }
    }
    None
}

/// The nine-byte prefix `tag / id`.
fn prefix(tag: u8, id: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(GENERATION_PREFIX_LEN);
    out.push(tag);
    out.extend_from_slice(&id.to_be_bytes());
    out
}

/// The first key past every key starting with `tag / id`.
fn prefix_end(tag: u8, id: u64) -> Vec<u8> {
    match id.checked_add(1) {
        Some(next) => prefix(tag, next),
        None => vec![tag + 1],
    }
}

fn decode_u64(what: &str, value: &[u8]) -> StorageResult<u64> {
    let bytes: [u8; 8] = value.try_into().map_err(|_| {
        StorageError::InstallationCatalog(format!("{what} holds {} bytes, expected 8", value.len()))
    })?;
    Ok(u64::from_be_bytes(bytes))
}

/// Where a key lives in this member's tree.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Located<'a> {
    /// Not a generation key: stored as written.
    Raw(&'a [u8]),
    /// A generation key, at this physical address.
    Bound(&'a [u8]),
    /// A key of a generation this member holds no installation of: nothing
    /// is stored under it.
    Unbound,
}

/// One piece of a logical key range, as it is stored: either a stretch of
/// untranslated keys, or the part of one generation's range held by its
/// installation. Pieces come in logical key order.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Piece {
    /// Keys stored as written, between these bounds.
    Raw {
        lo: Bound<Vec<u8>>,
        hi: Bound<Vec<u8>>,
    },
    /// Keys of `generation`, stored between these physical bounds.
    Domain {
        generation: u64,
        lo: Bound<Vec<u8>>,
        hi: Bound<Vec<u8>>,
    },
}

/// The validated bindings of one moment: immutable, so an access resolves
/// them once and keeps them for its whole run.
#[derive(Debug, Default, Clone)]
pub(crate) struct Bindings {
    /// Generation to its published installation.
    installed: FxHashMap<u64, u64>,
    /// Installation to the generation it holds: published, being prepared
    /// or retired.
    owner: FxHashMap<u64, u64>,
    /// Bound generations, ascending: the order of their logical keys.
    generations: Vec<u64>,
    /// Generation to the replacement being prepared for it, which every
    /// write of the generation also reaches.
    staging: FxHashMap<u64, u64>,
    /// Installation to the lowest seqno its history covers; absent means 0.
    history_from: FxHashMap<u64, SeqNo>,
}

impl Bindings {
    /// The published installation of `generation`.
    #[inline]
    pub(crate) fn installation(&self, generation: u64) -> Option<u64> {
        self.installed.get(&generation).copied()
    }

    /// The replacement being prepared for `generation`.
    #[inline]
    pub(crate) fn staging(&self, generation: u64) -> Option<u64> {
        self.staging.get(&generation).copied()
    }

    /// Whether any replacement is being prepared: the write path's one test
    /// before it looks for a second address.
    #[inline]
    pub(crate) fn any_staging(&self) -> bool {
        !self.staging.is_empty()
    }

    /// The lowest seqno the history of `installation` covers.
    #[inline]
    pub(crate) fn history_from(&self, installation: u64) -> SeqNo {
        self.history_from.get(&installation).copied().unwrap_or(0)
    }

    /// Refuse a read of the generations in `(lo, hi)` at `seqno` below the
    /// history their installations hold: an installation built from current
    /// contents does not answer for the past.
    ///
    /// # Errors
    ///
    /// [`StorageError::SnapshotOutsideRetention`] naming the first floor the
    /// read is below.
    pub(crate) fn check_history(
        &self,
        lo: Bound<&[u8]>,
        hi: Bound<&[u8]>,
        seqno: SeqNo,
    ) -> StorageResult<()> {
        if self.history_from.is_empty() {
            return Ok(());
        }
        for split in self.split(lo, hi) {
            if let Some(installation) = split.installation {
                let floor = self.history_from(installation);
                if seqno < floor {
                    return Err(StorageError::SnapshotOutsideRetention {
                        snapshot: seqno,
                        watermark: floor,
                    });
                }
            }
        }
        Ok(())
    }

    /// [`Self::check_history`] for the keys starting with `prefix`.
    pub(crate) fn check_prefix_history(&self, prefix: &[u8], seqno: SeqNo) -> StorageResult<()> {
        if self.history_from.is_empty() {
            return Ok(());
        }
        let end = successor(prefix);
        self.check_history(
            Bound::Included(prefix),
            end.as_deref().map_or(Bound::Unbounded, Bound::Excluded),
            seqno,
        )
    }

    /// [`Self::check_history`] for the one key `key`.
    pub(crate) fn check_key_history(&self, key: &[u8], seqno: SeqNo) -> StorageResult<()> {
        if self.history_from.is_empty() || domain_tag(key).is_none() {
            return Ok(());
        }
        let Some(installation) = slot(key).and_then(|g| self.installation(g)) else {
            return Ok(());
        };
        let floor = self.history_from(installation);
        if seqno < floor {
            return Err(StorageError::SnapshotOutsideRetention {
                snapshot: seqno,
                watermark: floor,
            });
        }
        Ok(())
    }

    /// The replacements being prepared, as if each were its generation's
    /// published installation: what a write also reaches.
    fn staging_view(&self) -> Self {
        let mut generations: Vec<u64> = self.staging.keys().copied().collect();
        generations.sort_unstable();
        Self {
            installed: self.staging.clone(),
            owner: self
                .staging
                .iter()
                .map(|(&generation, &installation)| (installation, generation))
                .collect(),
            generations,
            staging: FxHashMap::default(),
            history_from: FxHashMap::default(),
        }
    }

    /// The bindings with `generation` published at `installation`.
    fn with_published(&self, generation: u64, installation: u64, history_from: SeqNo) -> Self {
        let mut next = self.clone();
        // A previous installation stays in `owner`: retired, never reused.
        if next.installed.insert(generation, installation).is_none() {
            let at = next
                .generations
                .binary_search(&generation)
                .unwrap_or_else(|at| at);
            next.generations.insert(at, generation);
        }
        next.owner.insert(installation, generation);
        if history_from == 0 {
            next.history_from.remove(&installation);
        } else {
            next.history_from.insert(installation, history_from);
        }
        next
    }

    /// The generation `installation` holds.
    #[inline]
    pub(crate) fn generation(&self, installation: u64) -> Option<u64> {
        self.owner.get(&installation).copied()
    }

    /// Where the logical `key` is stored. `buf` receives a translated key.
    #[inline]
    pub(crate) fn locate<'a>(&self, key: &'a [u8], buf: &'a mut Vec<u8>) -> Located<'a> {
        if domain_tag(key).is_none() {
            return Located::Raw(key);
        }
        match slot(key).and_then(|generation| self.installation(generation)) {
            Some(installation) => {
                with_slot(key, installation, buf);
                Located::Bound(buf)
            }
            None => Located::Unbound,
        }
    }

    /// The logical form of the physical `key`, read from this member's tree.
    ///
    /// # Errors
    ///
    /// [`StorageError::InstallationCatalog`] for a generation key under an
    /// installation the catalog does not name.
    pub(crate) fn logical(&self, key: &[u8]) -> StorageResult<Option<u64>> {
        if domain_tag(key).is_none() {
            return Ok(None);
        }
        let installation = slot(key).ok_or_else(|| {
            StorageError::InstallationCatalog(format!(
                "an index entry key of {} bytes is shorter than its prefix",
                key.len()
            ))
        })?;
        self.generation(installation).map(Some).ok_or_else(|| {
            StorageError::InstallationCatalog(format!(
                "entries are stored under installation {installation}, which no generation owns"
            ))
        })
    }

    /// The logical ranges, `[start, end)`, of the keys stored in the
    /// physical range `[min, max]` (inclusive): what a range of the tree
    /// itself, such as one a structural repair lost, holds in the keys the
    /// application addresses. Reserved keys are left out.
    pub(crate) fn logical_ranges(&self, min: &[u8], max: &[u8]) -> Vec<(Vec<u8>, Vec<u8>)> {
        let mut past_max = max.to_vec();
        past_max.push(0x00);
        let mut out = Vec::new();
        let first_tag = [GENERATION_TAGS[0]];
        let past_tags = [GENERATION_TAGS[GENERATION_TAGS.len() - 1] + 1];
        // User keys below the generation tags are stored as written.
        let lo = min.max(crate::engine::coverage::USER_KEYSPACE_START);
        let hi = past_max.as_slice().min(&first_tag[..]);
        if lo < hi {
            out.push((lo.to_vec(), hi.to_vec()));
        }
        let mut installations: Vec<(u64, u64)> = self
            .owner
            .iter()
            .map(|(&installation, &generation)| (installation, generation))
            .collect();
        installations.sort_unstable();
        for &tag in &GENERATION_TAGS {
            for &(installation, generation) in &installations {
                let start = prefix(tag, installation);
                let end = prefix_end(tag, installation);
                let lo = min.max(start.as_slice());
                let hi = past_max.as_slice().min(end.as_slice());
                if lo >= hi {
                    continue;
                }
                let logical_lo = if lo == start.as_slice() {
                    prefix(tag, generation)
                } else {
                    let mut out = Vec::with_capacity(lo.len());
                    with_slot(lo, generation, &mut out);
                    out
                };
                let logical_hi = if hi == end.as_slice() {
                    prefix_end(tag, generation)
                } else {
                    let mut out = Vec::with_capacity(hi.len());
                    with_slot(hi, generation, &mut out);
                    out
                };
                out.push((logical_lo, logical_hi));
            }
        }
        let lo = min.max(&past_tags[..]);
        if lo < past_max.as_slice() {
            out.push((lo.to_vec(), past_max));
        }
        out
    }

    /// The pieces of the logical range `(lo, hi)`, in logical order.
    pub(crate) fn pieces(&self, lo: Bound<&[u8]>, hi: Bound<&[u8]>) -> Vec<Piece> {
        self.split(lo, hi)
            .into_iter()
            .map(|split| split.piece)
            .collect()
    }

    /// [`Self::pieces`], each with the logical range it covers and the
    /// installation it is read from.
    fn split(&self, lo: Bound<&[u8]>, hi: Bound<&[u8]>) -> Vec<Split> {
        let mut out = Vec::new();
        let first_tag = [GENERATION_TAGS[0]];
        let past_tags = [GENERATION_TAGS[GENERATION_TAGS.len() - 1] + 1];
        // Below the generation keys.
        let below_hi = upper_min(hi, Bound::Excluded(&first_tag[..]));
        if !is_empty(lo, below_hi) {
            out.push(Split::raw(lo, below_hi));
        }
        for &tag in &GENERATION_TAGS {
            for &generation in &self.generations {
                let start = prefix(tag, generation);
                let end = prefix_end(tag, generation);
                let dlo = lower_max(lo, Bound::Included(start.as_slice()));
                let dhi = upper_min(hi, Bound::Excluded(end.as_slice()));
                if is_empty(dlo, dhi) {
                    continue;
                }
                let Some(installation) = self.installation(generation) else {
                    continue;
                };
                out.push(Split {
                    piece: Piece::Domain {
                        generation,
                        lo: physical_lower(dlo, &start, tag, installation),
                        hi: physical_upper(dhi, &end, tag, installation),
                    },
                    logical_lo: owned(dlo),
                    logical_hi: owned(dhi),
                    installation: Some(installation),
                });
            }
        }
        // Above the generation keys.
        let above_lo = lower_max(lo, Bound::Included(&past_tags[..]));
        if !is_empty(above_lo, hi) {
            out.push(Split::raw(above_lo, hi));
        }
        out
    }
}

/// One piece of a logical range, with what it covers logically.
struct Split {
    piece: Piece,
    logical_lo: Bound<Vec<u8>>,
    logical_hi: Bound<Vec<u8>>,
    /// The installation a generation's piece is read from.
    installation: Option<u64>,
}

impl Split {
    fn raw(lo: Bound<&[u8]>, hi: Bound<&[u8]>) -> Self {
        Self {
            piece: Piece::Raw {
                lo: owned(lo),
                hi: owned(hi),
            },
            logical_lo: owned(lo),
            logical_hi: owned(hi),
            installation: None,
        }
    }
}

fn owned(bound: Bound<&[u8]>) -> Bound<Vec<u8>> {
    match bound {
        Bound::Included(k) => Bound::Included(k.to_vec()),
        Bound::Excluded(k) => Bound::Excluded(k.to_vec()),
        Bound::Unbounded => Bound::Unbounded,
    }
}

/// The physical form of a lower bound inside one generation's range: the
/// installation's own start when the bound is the range's start, else the
/// bound key with its slot replaced (a key past the start carries the
/// generation's whole prefix).
fn physical_lower(bound: Bound<&[u8]>, start: &[u8], tag: u8, installation: u64) -> Bound<Vec<u8>> {
    let translate = |key: &[u8]| {
        if key.len() < GENERATION_PREFIX_LEN || key <= start {
            prefix(tag, installation)
        } else {
            let mut out = Vec::with_capacity(key.len());
            with_slot(key, installation, &mut out);
            out
        }
    };
    match bound {
        Bound::Included(k) => Bound::Included(translate(k)),
        Bound::Excluded(k) if k < start => Bound::Included(prefix(tag, installation)),
        Bound::Excluded(k) => Bound::Excluded(translate(k)),
        Bound::Unbounded => Bound::Included(prefix(tag, installation)),
    }
}

/// The physical form of an upper bound inside one generation's range: the
/// installation's own end when the bound is the range's end, else the bound
/// key with its slot replaced.
fn physical_upper(bound: Bound<&[u8]>, end: &[u8], tag: u8, installation: u64) -> Bound<Vec<u8>> {
    let translate = |key: &[u8]| {
        let mut out = Vec::with_capacity(key.len());
        with_slot(key, installation, &mut out);
        out
    };
    match bound {
        Bound::Excluded(k) if k >= end => Bound::Excluded(prefix_end(tag, installation)),
        Bound::Excluded(k) => Bound::Excluded(translate(k)),
        Bound::Included(k) => Bound::Included(translate(k)),
        Bound::Unbounded => Bound::Excluded(prefix_end(tag, installation)),
    }
}

/// The greater of two lower bounds.
fn lower_max<'a>(a: Bound<&'a [u8]>, b: Bound<&'a [u8]>) -> Bound<&'a [u8]> {
    match (a, b) {
        (Bound::Unbounded, x) | (x, Bound::Unbounded) => x,
        (Bound::Included(x), Bound::Included(y)) => Bound::Included(x.max(y)),
        (Bound::Excluded(x), Bound::Excluded(y)) => Bound::Excluded(x.max(y)),
        (Bound::Included(i), Bound::Excluded(e)) | (Bound::Excluded(e), Bound::Included(i)) => {
            if i > e {
                Bound::Included(i)
            } else {
                Bound::Excluded(e)
            }
        }
    }
}

/// The smaller of two upper bounds.
fn upper_min<'a>(a: Bound<&'a [u8]>, b: Bound<&'a [u8]>) -> Bound<&'a [u8]> {
    match (a, b) {
        (Bound::Unbounded, x) | (x, Bound::Unbounded) => x,
        (Bound::Included(x), Bound::Included(y)) => Bound::Included(x.min(y)),
        (Bound::Excluded(x), Bound::Excluded(y)) => Bound::Excluded(x.min(y)),
        (Bound::Included(i), Bound::Excluded(e)) | (Bound::Excluded(e), Bound::Included(i)) => {
            if i < e {
                Bound::Included(i)
            } else {
                Bound::Excluded(e)
            }
        }
    }
}

/// Whether no key lies between `lo` and `hi`.
fn is_empty(lo: Bound<&[u8]>, hi: Bound<&[u8]>) -> bool {
    match (lo, hi) {
        (Bound::Unbounded, _) | (_, Bound::Unbounded) => false,
        (Bound::Included(a), Bound::Included(b)) => a > b,
        (Bound::Included(a) | Bound::Excluded(a), Bound::Excluded(b))
        | (Bound::Excluded(a), Bound::Included(b)) => a >= b,
    }
}

/// The installation catalog of one index tree: the validated bindings and
/// the allocator of new installations.
#[derive(Debug)]
pub(crate) struct Installations {
    current: parking_lot::RwLock<Arc<Bindings>>,
    /// Held while a binding is written, so two writers of one new generation
    /// bind it once.
    alloc: parking_lot::Mutex<Allocator>,
    /// Every batch applied to the index tree holds it shared; registering or
    /// publishing a replacement holds it exclusively, for the time of one
    /// catalog write, so a batch reaches either both copies or the old one.
    fence: parking_lot::RwLock<()>,
    /// Replacements found unfinished at open, cleared once the store's seqno
    /// is restored.
    unfinished: parking_lot::Mutex<Vec<(u64, u64)>>,
    /// Generations whose replacement is recorded complete: the ones that may
    /// be published.
    imported: parking_lot::Mutex<rustc_hash::FxHashSet<u64>>,
}

#[derive(Debug)]
struct Allocator {
    /// The next unallocated installation.
    next: u64,
    /// The seqno the catalog was last written at. Each write goes above it:
    /// entries replayed from a journal carry their old commit timestamps, and
    /// an allocator record written below an earlier one would read back as
    /// the earlier value, handing an installation out twice.
    written_at: SeqNo,
}

impl Installations {
    /// Read and validate the catalog of `tree`.
    ///
    /// # Errors
    ///
    /// [`StorageError::InstallationCatalog`] when a record is malformed, two
    /// generations claim one installation, an installation is at or past
    /// the allocator, or entries sit under an installation no generation
    /// owns; a read failure.
    pub(crate) fn load(tree: &AnyTree) -> StorageResult<Self> {
        let next = match tree.get(NEXT_KEY, SeqNo::MAX)? {
            Some(value) => decode_u64("the installation allocator", &value)?,
            None => FIRST_INSTALLATION,
        };
        let allocated = |what: &str, installation: u64| {
            if installation < FIRST_INSTALLATION || installation >= next {
                return Err(StorageError::InstallationCatalog(format!(
                    "{what} names installation {installation}, outside the allocated \
                     {FIRST_INSTALLATION}..{next}"
                )));
            }
            Ok(())
        };
        let mut bindings = Bindings::default();
        let own = |bindings: &mut Bindings, installation: u64, generation: u64| {
            if let Some(other) = bindings.owner.insert(installation, generation) {
                return Err(StorageError::InstallationCatalog(format!(
                    "installation {installation} is claimed by generations {other} and \
                     {generation}"
                )));
            }
            Ok(())
        };
        for (generation, value) in read_records(tree, BINDING_PREFIX, BINDING_END)? {
            let (installation, history_from) = decode_pair("a binding", &value)?;
            allocated("a binding", installation)?;
            own(&mut bindings, installation, generation)?;
            bindings.installed.insert(generation, installation);
            bindings.generations.push(generation);
            if history_from != 0 {
                bindings.history_from.insert(installation, history_from);
            }
        }
        // Binding keys sort by generation, so the list is ascending.
        let mut unfinished = Vec::new();
        for (generation, value) in read_records(tree, STAGING_PREFIX, STAGING_END)? {
            let staged = replace::Staged::decode(&value)?;
            allocated("a replacement", staged.installation)?;
            own(&mut bindings, staged.installation, generation)?;
            if staged.imported {
                bindings.staging.insert(generation, staged.installation);
                if staged.history_from != 0 {
                    bindings
                        .history_from
                        .insert(staged.installation, staged.history_from);
                }
            } else {
                // Its contents may be partial: it is cleared, and its
                // generation keeps the installation it had.
                unfinished.push((generation, staged.installation));
            }
        }
        for (installation, value) in read_records(tree, RETIRED_PREFIX, RETIRED_END)? {
            allocated("a retired installation", installation)?;
            own(
                &mut bindings,
                installation,
                decode_u64("a retirement", &value)?,
            )?;
        }
        verify_no_orphans(tree, &bindings)?;
        Ok(Self {
            alloc: parking_lot::Mutex::new(Allocator {
                next,
                written_at: tree.get_highest_seqno().unwrap_or(0),
            }),
            imported: parking_lot::Mutex::new(bindings.staging.keys().copied().collect()),
            current: parking_lot::RwLock::new(Arc::new(bindings)),
            fence: parking_lot::RwLock::new(()),
            unfinished: parking_lot::Mutex::new(unfinished),
        })
    }

    /// The seqno the next catalog write goes at: above the catalog's last
    /// write and at least `seqno`.
    fn next_write(alloc: &Allocator, seqno: SeqNo) -> StorageResult<SeqNo> {
        Ok(alloc
            .written_at
            .checked_add(1)
            .ok_or_else(|| {
                StorageError::InstallationCatalog("the catalog's seqnos are exhausted".into())
            })?
            .max(seqno))
    }

    /// Write the allocator and every binding back into `tree` after it was
    /// cleared: the installations stay allocated, so none is handed out
    /// again, and entries written after the clear land under the bindings
    /// already in use. A replacement in preparation lost its contents with
    /// the clear and is dropped.
    ///
    /// # Errors
    ///
    /// A write failure.
    pub(crate) fn rewrite(&self, tree: &AnyTree, seqno: SeqNo) -> StorageResult<()> {
        let mut alloc = self.alloc.lock();
        let at = Self::next_write(&alloc, seqno)?;
        let mut current = self.current.write();
        let mut bindings = Bindings::clone(&current);
        bindings.staging.clear();
        let mut batch = lsm_tree::WriteBatch::with_capacity(bindings.owner.len() + 1);
        batch.insert(NEXT_KEY.as_slice(), alloc.next.to_be_bytes().as_slice());
        for (&generation, &installation) in &bindings.installed {
            batch.insert(
                binding_key(generation).as_slice(),
                encode_binding(installation, bindings.history_from(installation)).as_slice(),
            );
        }
        for (&installation, &generation) in &bindings.owner {
            if bindings.installed.get(&generation) != Some(&installation) {
                batch.insert(
                    record_key(RETIRED_PREFIX, installation).as_slice(),
                    generation.to_be_bytes().as_slice(),
                );
            }
        }
        tree.apply_batch(batch, at)?;
        alloc.written_at = at;
        *current = Arc::new(bindings);
        Ok(())
    }

    /// The bindings now, to resolve an access or batch with.
    #[inline]
    pub(crate) fn current(&self) -> Arc<Bindings> {
        Arc::clone(&self.current.read())
    }

    /// The installation of `generation`, binding a fresh one when it has
    /// none. A new binding and the allocator are written to `tree`, at
    /// `seqno` or above the catalog's last write, before this returns, so
    /// entries written under it after the call follow it in seal order.
    ///
    /// # Errors
    ///
    /// [`StorageError::InstallationCatalog`] when the installations or the
    /// seqnos are exhausted; a write failure.
    pub(crate) fn bind(&self, tree: &AnyTree, generation: u64, seqno: SeqNo) -> StorageResult<u64> {
        if let Some(installation) = self.current.read().installation(generation) {
            return Ok(installation);
        }
        let mut alloc = self.alloc.lock();
        if let Some(installation) = self.current.read().installation(generation) {
            return Ok(installation);
        }
        let (installation, after) = Self::allocate(&alloc)?;
        let at = Self::next_write(&alloc, seqno)?;
        let mut batch = lsm_tree::WriteBatch::with_capacity(2);
        batch.insert(NEXT_KEY.as_slice(), after.to_be_bytes().as_slice());
        batch.insert(
            binding_key(generation).as_slice(),
            encode_binding(installation, 0).as_slice(),
        );
        tree.apply_batch(batch, at)?;
        alloc.next = after;
        alloc.written_at = at;
        let mut current = self.current.write();
        *current = Arc::new(current.with_published(generation, installation, 0));
        Ok(installation)
    }

    /// The installation to hand out next, and the allocator's value after.
    fn allocate(alloc: &Allocator) -> StorageResult<(u64, u64)> {
        let installation = alloc.next;
        let after = installation.checked_add(1).ok_or_else(|| {
            StorageError::InstallationCatalog("every installation id has been allocated".into())
        })?;
        Ok((installation, after))
    }
}

/// The records under `[prefix, end)`, each with the u64 after the prefix.
fn read_records(
    tree: &AnyTree,
    prefix: [u8; 3],
    end: [u8; 3],
) -> StorageResult<Vec<(u64, lsm_tree::UserValue)>> {
    let mut out = Vec::new();
    for guard in tree.range(prefix.as_slice()..end.as_slice(), SeqNo::MAX, None) {
        let (key, value) = guard.into_inner()?;
        let id = decode_u64("a catalog key", key.get(prefix.len()..).unwrap_or(&[]))?;
        out.push((id, value));
    }
    Ok(out)
}

/// One piece of a seekable scan: its own tree iterator, and how its keys
/// map.
struct SeekPiece {
    inner: Box<dyn lsm_tree::SeekableGuardIter>,
    /// Bounds as stored, to reopen the piece fresh.
    lo: Bound<Vec<u8>>,
    hi: Bound<Vec<u8>>,
    /// Bounds as addressed, to tell which piece a key falls in.
    logical_lo: Bound<Vec<u8>>,
    logical_hi: Bound<Vec<u8>>,
    /// `(generation, installation)` of a generation's piece.
    domain: Option<(u64, u64)>,
}

/// A seekable scan over a logical range: one tree iterator per piece, in
/// logical order. A seek repositions the piece holding the key and reopens
/// the pieces it passes over, so every key is read once whichever way the
/// scan moves.
pub(crate) struct Seekable {
    tree: AnyTree,
    seqno: SeqNo,
    pieces: Vec<SeekPiece>,
    /// The first piece the forward end has not finished.
    front: usize,
    /// One past the last piece the backward end has not finished.
    back: usize,
}

impl Seekable {
    /// A seekable scan of the logical range `(lo, hi)` of `tree` at
    /// `seqno`; `bindings` translates when `tree` is the index tree.
    pub(crate) fn open(
        tree: &AnyTree,
        bindings: Option<&Bindings>,
        lo: Bound<&[u8]>,
        hi: Bound<&[u8]>,
        seqno: SeqNo,
    ) -> Self {
        let splits = match bindings {
            Some(bindings) => bindings.split(lo, hi),
            None if is_empty(lo, hi) => Vec::new(),
            None => vec![Split::raw(lo, hi)],
        };
        let pieces: Vec<SeekPiece> = splits
            .into_iter()
            .map(|split| {
                let (lo, hi, domain) = match split.piece {
                    Piece::Raw { lo, hi } => (lo, hi, None),
                    Piece::Domain { generation, lo, hi } => (
                        lo,
                        hi,
                        split
                            .installation
                            .map(|installation| (generation, installation)),
                    ),
                };
                SeekPiece {
                    inner: tree.range_seekable((lo.clone(), hi.clone()), seqno, None),
                    lo,
                    hi,
                    logical_lo: split.logical_lo,
                    logical_hi: split.logical_hi,
                    domain,
                }
            })
            .collect();
        Self {
            tree: tree.clone(),
            seqno,
            back: pieces.len(),
            pieces,
            front: 0,
        }
    }

    fn reopen(&mut self, at: usize) {
        if let Some(piece) = self.pieces.get_mut(at) {
            piece.inner =
                self.tree
                    .range_seekable((piece.lo.clone(), piece.hi.clone()), self.seqno, None);
        }
    }

    fn wrap(piece: &SeekPiece, guard: lsm_tree::IterGuardImpl) -> StorageGuard {
        match piece.domain {
            Some((generation, _)) => StorageGuard::of_generation(guard, generation),
            None => StorageGuard::raw(guard),
        }
    }

    /// `key` as stored in `piece`, which holds it.
    fn stored(piece: &SeekPiece, key: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(key.len());
        match piece.domain {
            Some((_, installation)) if key.len() >= GENERATION_PREFIX_LEN => {
                with_slot(key, installation, &mut out);
            }
            _ => out.extend_from_slice(key),
        }
        out
    }

    /// Reposition so the forward end next yields the first key at or above
    /// the logical `key`.
    pub(crate) fn seek_to(&mut self, key: &[u8]) {
        let Some(at) = self
            .pieces
            .iter()
            .position(|piece| upper_admits(&piece.logical_hi, key))
        else {
            self.front = self.pieces.len();
            return;
        };
        // Pieces after it are read from their start again.
        for later in at + 1..self.pieces.len() {
            self.reopen(later);
        }
        self.back = self.pieces.len();
        self.front = at;
        if lower_admits(&self.pieces[at].logical_lo, key) {
            let stored = Self::stored(&self.pieces[at], key);
            self.pieces[at].inner.seek_to(&stored);
        } else {
            self.reopen(at);
        }
    }

    /// Reposition so the backward end next yields the last key at or below
    /// the logical `key`.
    pub(crate) fn seek_to_for_prev(&mut self, key: &[u8]) {
        let Some(at) = self
            .pieces
            .iter()
            .rposition(|piece| lower_admits(&piece.logical_lo, key))
        else {
            self.back = 0;
            return;
        };
        // Pieces before it are read from their end again.
        for earlier in 0..at {
            self.reopen(earlier);
        }
        self.front = 0;
        self.back = at + 1;
        if upper_admits(&self.pieces[at].logical_hi, key) {
            let stored = Self::stored(&self.pieces[at], key);
            self.pieces[at].inner.seek_to_for_prev(&stored);
        } else {
            self.reopen(at);
        }
    }

    /// The logical key the forward end yields next, without consuming it.
    pub(crate) fn peek_key(&mut self) -> Option<lsm_tree::Result<lsm_tree::UserKey>> {
        while self.front < self.back {
            let piece = &mut self.pieces[self.front];
            match piece.inner.peek_key() {
                Some(Ok(key)) => {
                    return Some(Ok(match piece.domain {
                        Some((generation, _)) => {
                            let mut out = key.to_vec();
                            put_generation(&mut out, generation);
                            lsm_tree::UserKey::from(out)
                        }
                        None => key,
                    }));
                }
                Some(Err(e)) => return Some(Err(e)),
                None => self.front += 1,
            }
        }
        None
    }
}

impl Iterator for Seekable {
    type Item = StorageGuard;

    fn next(&mut self) -> Option<StorageGuard> {
        while self.front < self.back {
            let piece = &mut self.pieces[self.front];
            if let Some(guard) = piece.inner.next() {
                return Some(Self::wrap(piece, guard));
            }
            self.front += 1;
        }
        None
    }
}

impl DoubleEndedIterator for Seekable {
    fn next_back(&mut self) -> Option<StorageGuard> {
        while self.back > self.front {
            let piece = &mut self.pieces[self.back - 1];
            if let Some(guard) = piece.inner.next_back() {
                return Some(Self::wrap(piece, guard));
            }
            self.back -= 1;
        }
        None
    }
}

/// Whether `key` is at or below the upper bound.
fn upper_admits(bound: &Bound<Vec<u8>>, key: &[u8]) -> bool {
    match bound {
        Bound::Included(hi) => key <= hi.as_slice(),
        Bound::Excluded(hi) => key < hi.as_slice(),
        Bound::Unbounded => true,
    }
}

/// Whether `key` is at or above the lower bound.
fn lower_admits(bound: &Bound<Vec<u8>>, key: &[u8]) -> bool {
    match bound {
        Bound::Included(lo) => key >= lo.as_slice(),
        Bound::Excluded(lo) => key > lo.as_slice(),
        Bound::Unbounded => true,
    }
}

/// How one tree batch addresses its keys: through the index tree's
/// installations, binding a generation on its first write, or as written for
/// every other tree. The bindings are resolved once for the batch and again
/// only after a binding is made. For the index tree it holds the catalog's
/// apply side for the batch's lifetime, so a replacement is registered or
/// published between batches, never inside one.
pub(crate) struct Addressing<'a> {
    installations: Option<&'a Installations>,
    _applying: Option<parking_lot::RwLockReadGuard<'a, ()>>,
    tree: &'a AnyTree,
    seqno: SeqNo,
    bindings: Option<Arc<Bindings>>,
    buf: Vec<u8>,
    staging_buf: Vec<u8>,
}

impl<'a> Addressing<'a> {
    /// Addressing for a batch written to `tree` at `seqno`; `installations`
    /// is the index tree's catalog when `tree` is the index tree.
    pub(crate) fn new(
        installations: Option<&'a Installations>,
        tree: &'a AnyTree,
        seqno: SeqNo,
    ) -> Self {
        // Taken before the bindings are read, so they stay current for the
        // whole batch. Recursive: a batch may run inside another engine write.
        let applying = installations.map(|i| i.fence.read_recursive());
        Self {
            bindings: installations.map(Installations::current),
            _applying: applying,
            installations,
            tree,
            seqno,
            buf: Vec::new(),
            staging_buf: Vec::new(),
        }
    }

    /// Where the replacement being prepared for the generation of `key`
    /// holds the same write, if one is: the write lands there too, in the
    /// same tree batch, so the tree's coverage of the batch covers both.
    pub(crate) fn staging_point<'k>(&'k mut self, key: &'k [u8]) -> Option<&'k [u8]> {
        let bindings = self.bindings.as_ref()?;
        if !bindings.any_staging() || domain_tag(key).is_none() {
            return None;
        }
        let installation = bindings.staging(slot(key)?)?;
        with_slot(key, installation, &mut self.staging_buf);
        Some(&self.staging_buf)
    }

    /// The stored ranges, `[start, end)`, of the replacements being prepared
    /// that the range delete of `[start, end)` covers.
    pub(crate) fn staging_range(&self, start: &[u8], end: &[u8]) -> Vec<(Vec<u8>, Vec<u8>)> {
        let Some(bindings) = self.bindings.as_ref().filter(|b| b.any_staging()) else {
            return Vec::new();
        };
        let view = bindings.staging_view();
        view.pieces(Bound::Included(start), Bound::Excluded(end))
            .into_iter()
            .filter_map(|piece| match piece {
                Piece::Domain { lo, hi, .. } => Some((exclusive_start(lo), exclusive_end(hi)?)),
                Piece::Raw { .. } => None,
            })
            .collect()
    }

    /// Where the point write of `key` lands. A generation without an
    /// installation is bound to a fresh one first.
    ///
    /// # Errors
    ///
    /// A generation key too short to name its generation; the errors of
    /// [`Installations::bind`].
    pub(crate) fn point<'k>(&'k mut self, key: &'k [u8]) -> StorageResult<&'k [u8]> {
        let Some(installations) = self.installations else {
            return Ok(key);
        };
        if domain_tag(key).is_none() {
            return Ok(key);
        }
        let generation = slot(key).ok_or_else(|| {
            StorageError::InstallationCatalog(format!(
                "an index entry key of {} bytes is shorter than its prefix",
                key.len()
            ))
        })?;
        let known = self
            .bindings
            .as_ref()
            .and_then(|bindings| bindings.installation(generation));
        let installation = match known {
            Some(installation) => installation,
            None => {
                let installation = installations.bind(self.tree, generation, self.seqno)?;
                self.bindings = Some(installations.current());
                installation
            }
        };
        with_slot(key, installation, &mut self.buf);
        Ok(&self.buf)
    }

    /// The stored ranges, `[start, end)`, the range delete of `[start, end)`
    /// covers: the untranslated stretches and each bound generation's part,
    /// within its installation. A generation without an installation holds
    /// nothing to delete.
    pub(crate) fn range(&self, start: &[u8], end: &[u8]) -> Vec<(Vec<u8>, Vec<u8>)> {
        let Some(bindings) = &self.bindings else {
            return vec![(start.to_vec(), end.to_vec())];
        };
        bindings
            .pieces(Bound::Included(start), Bound::Excluded(end))
            .into_iter()
            .filter_map(|piece| {
                let (Piece::Raw { lo, hi } | Piece::Domain { lo, hi, .. }) = piece;
                Some((exclusive_start(lo), exclusive_end(hi)?))
            })
            .collect()
    }
}

/// The inclusive start key of a lower bound.
fn exclusive_start(bound: Bound<Vec<u8>>) -> Vec<u8> {
    match bound {
        Bound::Included(key) => key,
        // The smallest key past `key`.
        Bound::Excluded(mut key) => {
            key.push(0x00);
            key
        }
        Bound::Unbounded => Vec::new(),
    }
}

/// The exclusive end key of an upper bound; `None` when unbounded.
fn exclusive_end(bound: Bound<Vec<u8>>) -> Option<Vec<u8>> {
    match bound {
        Bound::Excluded(key) => Some(key),
        // The smallest key past `key`.
        Bound::Included(mut key) => {
            key.push(0x00);
            Some(key)
        }
        Bound::Unbounded => None,
    }
}

/// Fail when entries sit under an installation the catalog does not name:
/// one seek per stored installation, at open.
fn verify_no_orphans(tree: &AnyTree, bindings: &Bindings) -> StorageResult<()> {
    for &tag in &GENERATION_TAGS {
        let mut from = vec![tag];
        let end = vec![tag + 1];
        loop {
            let Some(guard) = tree
                .range(from.clone()..end.clone(), SeqNo::MAX, None)
                .next()
            else {
                break;
            };
            let key = guard.key()?;
            let installation = slot(&key).ok_or_else(|| {
                StorageError::InstallationCatalog(format!(
                    "an index entry key of {} bytes is shorter than its prefix",
                    key.len()
                ))
            })?;
            if bindings.generation(installation).is_none() {
                return Err(StorageError::InstallationCatalog(format!(
                    "entries are stored under installation {installation}, which no generation owns"
                )));
            }
            match installation.checked_add(1) {
                Some(after) => from = prefix(tag, after),
                None => break,
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
