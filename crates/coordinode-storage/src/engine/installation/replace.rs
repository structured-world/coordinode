//! Replacing one generation's local installation.
//!
//! A replacement is prepared beside the published installation and swapped
//! in by one catalog write:
//!
//! 1. [`Installations::stage`] allocates a fresh installation and registers
//!    it, while no batch applies. From then on every write of the generation
//!    reaches both installations in the same tree batch, so the tree's own
//!    coverage of that batch covers the replacement too, and nothing applied
//!    after the registration can be missing from it.
//! 2. [`Installations::import`] writes the generation's history into the
//!    replacement at each version's own seqno, outside any fence. A version
//!    the replacement already holds from the writes above is written again
//!    identically: the generation's entries are puts, deletes and range
//!    deletes, which a repeat at the same seqno does not change.
//! 3. [`Installations::mark_imported`] records that the replacement's
//!    contents are complete, once its history reaches the registration.
//!    Written to the tree after the imported versions, the record persists
//!    no earlier than they do: a flush persists the tree's writes as a prefix.
//! 4. [`Installations::publish`] makes the replacement the published
//!    installation and retires the old one in one catalog write, again while
//!    no batch applies, then deletes the old entries with MVCC range
//!    tombstones: a reader already holding the old binding at an earlier
//!    snapshot still sees them.
//!
//! A crash before step 3's record reaches disk leaves a replacement whose
//! contents may be partial; the next open clears it and the generation keeps
//! its installation
//! ([`Installations::settle`]). A crash after step 4's catalog write and
//! before its tombstones leaves a retired installation with entries; the next
//! open deletes them.

use std::sync::Arc;

use coordinode_core::index::encoding::GENERATION_TAGS;
use lsm_tree::{AbstractTree, AnyTree, SeqNo};

use super::{
    Bindings, Installations, RETIRED_PREFIX, STAGING_PREFIX, domain_tag, encode_binding, prefix,
    prefix_end, record_key, slot, with_slot,
};
use crate::engine::coverage::{Domain, TreeCoverage};
use crate::error::{StorageError, StorageResult};

/// One version of an index generation's history, addressed by its logical
/// key, with the seqno it was written at. A sequence of them is the portable
/// form of a generation's contents: it carries the original versions, so a
/// copy made from it answers reads at earlier snapshots as the source did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HistoryEntry {
    /// `key` was written with `value` at `seqno`.
    Put {
        /// The logical key.
        key: Vec<u8>,
        /// The value written.
        value: Vec<u8>,
        /// The seqno it was written at.
        seqno: SeqNo,
    },
    /// `key` was deleted at `seqno`.
    Delete {
        /// The logical key.
        key: Vec<u8>,
        /// The seqno it was deleted at.
        seqno: SeqNo,
    },
    /// The keys in `[start, end)` were deleted at `seqno`.
    RemoveRange {
        /// Inclusive start, a logical key.
        start: Vec<u8>,
        /// Exclusive end, a logical key.
        end: Vec<u8>,
        /// The seqno they were deleted at.
        seqno: SeqNo,
    },
}

impl HistoryEntry {
    /// The seqno this version was written at.
    #[must_use]
    pub fn seqno(&self) -> SeqNo {
        match self {
            Self::Put { seqno, .. }
            | Self::Delete { seqno, .. }
            | Self::RemoveRange { seqno, .. } => *seqno,
        }
    }
}

/// A generation's history in portable form, and the source position it is
/// complete up to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationHistory {
    /// Every effect the source applied below this position of its log is in
    /// `entries`. A replacement filled from it is complete only when this
    /// reaches the position the replacement was registered at.
    pub covers_through: u64,
    /// The oldest snapshot the history answers: versions below it may have
    /// been folded at the source, so a copy filled from it refuses reads at
    /// earlier snapshots.
    pub history_from: SeqNo,
    /// The versions, in seqno order.
    pub entries: Vec<HistoryEntry>,
}

/// Length of an encoded replacement record.
const STAGED_LEN: usize = 25;

/// A replacement record: its installation, the lowest seqno its history will
/// cover, the log position it was registered at, and whether its imported
/// contents are complete.
pub(super) struct Staged {
    pub(super) installation: u64,
    pub(super) history_from: SeqNo,
    /// Every effect applied below this position reached only the published
    /// copy; every one applied later reaches both.
    pub(super) cut: u64,
    pub(super) imported: bool,
}

impl Staged {
    fn encode(&self) -> [u8; STAGED_LEN] {
        let mut value = [0u8; STAGED_LEN];
        value[..16].copy_from_slice(&encode_binding(self.installation, self.history_from));
        value[16..24].copy_from_slice(&self.cut.to_be_bytes());
        value[24] = u8::from(self.imported);
        value
    }

    pub(super) fn decode(value: &[u8]) -> StorageResult<Self> {
        let (flag, fields) = value
            .split_last()
            .filter(|_| value.len() == STAGED_LEN)
            .ok_or_else(|| {
                StorageError::InstallationCatalog(format!(
                    "a replacement record holds {} bytes, expected {STAGED_LEN}",
                    value.len()
                ))
            })?;
        let (pair, cut) = fields.split_at(16);
        let (installation, history_from) = super::decode_pair("a replacement record", pair)?;
        let cut = super::decode_u64("a replacement's position", cut)?;
        let imported = match flag {
            0 => false,
            1 => true,
            other => {
                return Err(StorageError::InstallationCatalog(format!(
                    "a replacement record's state is {other}, expected 0 or 1"
                )));
            }
        };
        Ok(Self {
            installation,
            history_from,
            cut,
            imported,
        })
    }
}

/// The stored ranges, `[start, end)`, of every entry of `installation`.
fn installation_ranges(installation: u64) -> [(Vec<u8>, Vec<u8>); 2] {
    GENERATION_TAGS.map(|tag| (prefix(tag, installation), prefix_end(tag, installation)))
}

/// Delete every entry of `installation` with range tombstones at `seqno`,
/// when any is left.
fn clear_installation(tree: &AnyTree, installation: u64, seqno: SeqNo) {
    for (start, end) in installation_ranges(installation) {
        let holds = tree
            .range(start.clone()..end.clone(), SeqNo::MAX, None)
            .next()
            .is_some();
        if holds {
            tree.remove_range(start, end, seqno);
        }
    }
}

impl Installations {
    /// Register a replacement of `generation`'s installation, whose history
    /// will cover seqnos from `history_from` on, and return it. From this
    /// call on, every write of the generation also reaches the replacement.
    ///
    /// # Errors
    ///
    /// [`StorageError::InstallationCatalog`] when the generation has no
    /// installation to replace, already has a replacement in preparation, or
    /// the installations are exhausted; a write failure.
    pub(crate) fn stage(
        &self,
        tree: &AnyTree,
        domain: Domain,
        generation: u64,
        seqno: SeqNo,
    ) -> StorageResult<u64> {
        let _quiet = self.fence.write();
        let mut alloc = self.alloc.lock();
        let mut current = self.current.write();
        if current.installation(generation).is_none() {
            return Err(StorageError::InstallationCatalog(format!(
                "generation {generation} has no installation here to replace"
            )));
        }
        if current.staging(generation).is_some() {
            return Err(StorageError::InstallationCatalog(format!(
                "generation {generation} already has a replacement in preparation"
            )));
        }
        let (installation, after) = Self::allocate(&alloc)?;
        let at = Self::next_write(&alloc, seqno)?;
        // No batch applies while the fence is held, so every effect marked
        // here reached the published copy alone, and every later one reaches
        // both. The highest marker bounds them from above, gaps included.
        let cut = TreeCoverage::read(tree, domain)?.next_uncovered();
        // The history's floor is known once it is imported.
        let staged = Staged {
            installation,
            history_from: 0,
            cut,
            imported: false,
        };
        let mut batch = lsm_tree::WriteBatch::with_capacity(2);
        batch.insert(super::NEXT_KEY.as_slice(), after.to_be_bytes().as_slice());
        batch.insert(
            record_key(STAGING_PREFIX, generation).as_slice(),
            staged.encode().as_slice(),
        );
        tree.apply_batch(batch, at)?;
        alloc.next = after;
        alloc.written_at = at;
        let mut bindings = Bindings::clone(&current);
        bindings.owner.insert(installation, generation);
        bindings.staging.insert(generation, installation);
        *current = Arc::new(bindings);
        Ok(installation)
    }

    /// Write `entries`, versions of `generation`'s history, into its
    /// replacement at their own seqnos. Returns how many were written.
    ///
    /// # Errors
    ///
    /// [`StorageError::InstallationCatalog`] when the generation has no
    /// replacement in preparation, or an entry's key lies outside the
    /// generation.
    pub(crate) fn import(
        &self,
        tree: &AnyTree,
        generation: u64,
        entries: &[HistoryEntry],
    ) -> StorageResult<usize> {
        let installation = self.current().staging(generation).ok_or_else(|| {
            StorageError::InstallationCatalog(format!(
                "generation {generation} has no replacement in preparation"
            ))
        })?;
        let within = |key: &[u8]| domain_tag(key).is_some() && slot(key) == Some(generation);
        let refuse = |key: &[u8]| {
            StorageError::InstallationCatalog(format!(
                "a history entry of {} bytes is not a key of generation {generation}",
                key.len()
            ))
        };
        let mut buf = Vec::new();
        for entry in entries {
            match entry {
                HistoryEntry::Put { key, value, seqno } => {
                    if !within(key) {
                        return Err(refuse(key));
                    }
                    with_slot(key, installation, &mut buf);
                    tree.insert(buf.as_slice(), value.as_slice(), *seqno);
                }
                HistoryEntry::Delete { key, seqno } => {
                    if !within(key) {
                        return Err(refuse(key));
                    }
                    with_slot(key, installation, &mut buf);
                    tree.remove(buf.as_slice(), *seqno);
                }
                HistoryEntry::RemoveRange { start, end, seqno } => {
                    let range =
                        range_within(start, end, generation).ok_or_else(|| refuse(start))?;
                    let (start, end) = translate_range(range, generation, installation);
                    if start < end {
                        tree.remove_range(start, end, *seqno);
                    }
                }
            }
        }
        Ok(entries.len())
    }

    /// Record that `generation`'s replacement holds its complete imported
    /// contents, from a history complete below `covers_through` and answering
    /// snapshots from `history_from` on. Written after the imported versions,
    /// it persists no earlier than they do.
    ///
    /// # Errors
    ///
    /// [`StorageError::InstallationCatalog`] when the generation has no
    /// replacement in preparation; [`StorageError::PositionBehind`] when the
    /// history ends before the position the replacement was registered at, so
    /// effects applied in between reached the published copy only; a read or
    /// write failure.
    pub(crate) fn mark_imported(
        &self,
        tree: &AnyTree,
        generation: u64,
        covers_through: u64,
        history_from: SeqNo,
        seqno: SeqNo,
    ) -> StorageResult<()> {
        let mut alloc = self.alloc.lock();
        let mut current = self.current.write();
        let Some(installation) = current.staging(generation) else {
            return Err(StorageError::InstallationCatalog(format!(
                "generation {generation} has no replacement in preparation"
            )));
        };
        let record = record_key(STAGING_PREFIX, generation);
        let value = tree.get(record, SeqNo::MAX)?.ok_or_else(|| {
            StorageError::InstallationCatalog(format!(
                "the replacement of generation {generation} has no record"
            ))
        })?;
        let mut staged = Staged::decode(&value)?;
        if covers_through < staged.cut {
            return Err(StorageError::PositionBehind {
                partition: format!("index generation {generation}"),
                source_next: covers_through,
                local_next: staged.cut,
            });
        }
        staged.imported = true;
        staged.history_from = history_from;
        let at = Self::next_write(&alloc, seqno)?;
        tree.insert(
            record_key(STAGING_PREFIX, generation).as_slice(),
            staged.encode().as_slice(),
            at,
        );
        alloc.written_at = at;
        if history_from != 0 {
            let mut bindings = Bindings::clone(&current);
            bindings.history_from.insert(installation, history_from);
            *current = Arc::new(bindings);
        }
        self.imported.lock().insert(generation);
        Ok(())
    }

    /// Publish `generation`'s replacement in place of its installation, and
    /// delete the old installation's entries with range tombstones at
    /// `clear_at`, which must lie above every seqno the old entries carry.
    /// Returns the retired installation.
    ///
    /// # Errors
    ///
    /// [`StorageError::InstallationCatalog`] when the generation has no
    /// replacement, or its contents are not recorded complete; a write
    /// failure.
    pub(crate) fn publish(
        &self,
        tree: &AnyTree,
        generation: u64,
        seqno: SeqNo,
        clear_at: SeqNo,
    ) -> StorageResult<u64> {
        let quiet = self.fence.write();
        let mut alloc = self.alloc.lock();
        let mut current = self.current.write();
        let (Some(old), Some(new)) = (
            current.installation(generation),
            current.staging(generation),
        ) else {
            return Err(StorageError::InstallationCatalog(format!(
                "generation {generation} has no replacement in preparation"
            )));
        };
        if !self.imported.lock().contains(&generation) {
            return Err(StorageError::InstallationCatalog(format!(
                "the replacement of generation {generation} is not recorded complete"
            )));
        }
        let history_from = current.history_from(new);
        let at = Self::next_write(&alloc, seqno)?;
        let mut batch = lsm_tree::WriteBatch::with_capacity(3);
        batch.insert(
            super::binding_key(generation).as_slice(),
            encode_binding(new, history_from).as_slice(),
        );
        batch.insert(
            record_key(RETIRED_PREFIX, old).as_slice(),
            generation.to_be_bytes().as_slice(),
        );
        batch.remove(record_key(STAGING_PREFIX, generation).as_slice());
        tree.apply_batch(batch, at)?;
        alloc.written_at = at;
        let mut bindings = current.with_published(generation, new, history_from);
        bindings.staging.remove(&generation);
        *current = Arc::new(bindings);
        self.imported.lock().remove(&generation);
        // Applies resume once the binding is swapped: nothing writes to the
        // retired copy any more, so clearing it needs no pause.
        drop((current, alloc, quiet));
        clear_installation(tree, old, clear_at.max(above(at)?));
        Ok(old)
    }

    /// Give up `generation`'s replacement: it is retired and its entries are
    /// deleted at `clear_at`; the generation keeps its installation.
    ///
    /// # Errors
    ///
    /// [`StorageError::InstallationCatalog`] when the generation has no
    /// replacement in preparation; a write failure.
    pub(crate) fn abandon(
        &self,
        tree: &AnyTree,
        generation: u64,
        seqno: SeqNo,
        clear_at: SeqNo,
    ) -> StorageResult<()> {
        let quiet = self.fence.write();
        let mut alloc = self.alloc.lock();
        let mut current = self.current.write();
        let staging = current.staging(generation).ok_or_else(|| {
            StorageError::InstallationCatalog(format!(
                "generation {generation} has no replacement in preparation"
            ))
        })?;
        let at = Self::next_write(&alloc, seqno)?;
        retire(tree, generation, staging, at)?;
        alloc.written_at = at;
        let mut bindings = Bindings::clone(&current);
        bindings.staging.remove(&generation);
        *current = Arc::new(bindings);
        self.imported.lock().remove(&generation);
        drop((current, alloc, quiet));
        clear_installation(tree, staging, clear_at.max(above(at)?));
        Ok(())
    }

    /// Finish what an interrupted replacement left at open: a replacement
    /// whose contents were never recorded complete is retired and cleared,
    /// and entries a retired installation still holds are deleted. `seqno`
    /// lies above every seqno the store holds.
    ///
    /// # Errors
    ///
    /// A write failure.
    pub(crate) fn settle(&self, tree: &AnyTree, seqno: SeqNo) -> StorageResult<()> {
        let unfinished = std::mem::take(&mut *self.unfinished.lock());
        let mut alloc = self.alloc.lock();
        let at = Self::next_write(&alloc, seqno)?;
        for &(generation, installation) in &unfinished {
            retire(tree, generation, installation, at)?;
        }
        alloc.written_at = at;
        let current = self.current();
        let clear_at = above(at)?;
        for (&installation, &generation) in &current.owner {
            let published = current.installation(generation) == Some(installation);
            let staging = current.staging(generation) == Some(installation);
            if !published && !staging {
                clear_installation(tree, installation, clear_at);
            }
        }
        Ok(())
    }

    /// The history of `generation`'s published installation, in logical
    /// keys, as far as `tree` still holds it, in seqno order. `position`
    /// reads, with no batch applying, the log position every effect below
    /// which the tree holds.
    ///
    /// # Errors
    ///
    /// [`StorageError::InstallationCatalog`] when the generation has no
    /// installation here, or its history holds a version a history entry
    /// cannot carry (a merge operand, a single delete); a read failure.
    pub(crate) fn export(
        &self,
        tree: &AnyTree,
        position: impl FnOnce(&AnyTree) -> StorageResult<u64>,
        generation: u64,
        oldest_readable: SeqNo,
    ) -> StorageResult<GenerationHistory> {
        use lsm_tree::ScanSinceEvent;
        let current = self.current();
        let installation = current.installation(generation).ok_or_else(|| {
            StorageError::InstallationCatalog(format!(
                "generation {generation} has no installation here"
            ))
        })?;
        let history_from = oldest_readable.max(current.history_from(installation));
        drop(current);
        // Read with no batch applying, so every effect below the position is
        // in the tree in full when the scan starts.
        let covers_through = {
            let _quiet = self.fence.write();
            position(tree)?
        };
        let ranges = installation_ranges(installation);
        let inside = |key: &[u8]| {
            ranges
                .iter()
                .any(|(s, e)| key >= s.as_slice() && key < e.as_slice())
        };
        let logical = |key: &[u8]| {
            let mut out = Vec::with_capacity(key.len());
            with_slot(key, generation, &mut out);
            out
        };
        let events: Vec<ScanSinceEvent> = match tree {
            AnyTree::Standard(t) => t.scan_since_seqno(0)?.collect(),
            AnyTree::Blob(t) => t.scan_since_seqno(0)?.collect(),
        };
        let mut out = Vec::new();
        for event in events {
            match event {
                ScanSinceEvent::Insert { key, value, seqno } if inside(&key) => {
                    out.push(HistoryEntry::Put {
                        key: logical(&key),
                        value: value.to_vec(),
                        seqno,
                    });
                }
                ScanSinceEvent::PointTombstone { key, seqno } if inside(&key) => {
                    out.push(HistoryEntry::Delete {
                        key: logical(&key),
                        seqno,
                    });
                }
                ScanSinceEvent::MergeOperand { key, .. }
                | ScanSinceEvent::WeakTombstone { key, .. }
                    if inside(&key) =>
                {
                    return Err(StorageError::InstallationCatalog(format!(
                        "generation {generation} holds a merge operand or single delete, which \
                         its history cannot carry"
                    )));
                }
                ScanSinceEvent::RangeTombstone {
                    start_key,
                    end_key,
                    seqno,
                } => {
                    // The part of the range inside each of the installation's
                    // stretches, as a logical range of the generation.
                    for (s, e) in &ranges {
                        let start = start_key.as_ref().max(s.as_slice());
                        let end = end_key.as_ref().min(e.as_slice());
                        if start < end {
                            let tag = s[0];
                            let logical_start = if start == s.as_slice() {
                                prefix(tag, generation)
                            } else {
                                logical(start)
                            };
                            let logical_end = if end == e.as_slice() {
                                prefix_end(tag, generation)
                            } else {
                                logical(end)
                            };
                            out.push(HistoryEntry::RemoveRange {
                                start: logical_start,
                                end: logical_end,
                                seqno,
                            });
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(GenerationHistory {
            covers_through,
            history_from,
            entries: out,
        })
    }
}

/// Retire `installation` of `generation`: drop its replacement record and
/// record it retired, in one batch at `at`.
fn retire(tree: &AnyTree, generation: u64, installation: u64, at: SeqNo) -> StorageResult<()> {
    let mut batch = lsm_tree::WriteBatch::with_capacity(2);
    batch.remove(record_key(STAGING_PREFIX, generation).as_slice());
    batch.insert(
        record_key(RETIRED_PREFIX, installation).as_slice(),
        generation.to_be_bytes().as_slice(),
    );
    tree.apply_batch(batch, at)?;
    Ok(())
}

/// `[start, end)` when it lies within one stretch of `generation`'s keys:
/// the tag, and the bounds.
fn range_within<'k>(
    start: &'k [u8],
    end: &'k [u8],
    generation: u64,
) -> Option<(u8, &'k [u8], &'k [u8])> {
    let tag = domain_tag(start)?;
    let lo = prefix(tag, generation);
    let hi = prefix_end(tag, generation);
    (start >= lo.as_slice() && end <= hi.as_slice() && start < end).then_some((tag, start, end))
}

/// A range of one stretch of `generation`'s keys, as stored under
/// `installation`. Both bounds lie within the stretch, so each carries the
/// generation's prefix, but for an end at the stretch's end.
fn translate_range(
    (tag, start, end): (u8, &[u8], &[u8]),
    generation: u64,
    installation: u64,
) -> (Vec<u8>, Vec<u8>) {
    let mut lo = Vec::with_capacity(start.len());
    with_slot(start, installation, &mut lo);
    let hi = if end == prefix_end(tag, generation).as_slice() {
        prefix_end(tag, installation)
    } else {
        let mut hi = Vec::with_capacity(end.len());
        with_slot(end, installation, &mut hi);
        hi
    };
    (lo, hi)
}

/// The seqno just above `at`, where a clear after a catalog write at `at`
/// goes.
fn above(at: SeqNo) -> StorageResult<SeqNo> {
    at.checked_add(1).ok_or_else(|| {
        StorageError::InstallationCatalog("the catalog's seqnos are exhausted".into())
    })
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
