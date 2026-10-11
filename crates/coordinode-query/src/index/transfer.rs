//! The index side of a transfer of the index partition: the inventory of
//! the B-tree copies a capture holds, and what a receiving member keeps of
//! it.
//!
//! Every B-tree generation in the captured catalog is listed, an empty one
//! too, as qualified (with the proof this member kept of its copy),
//! excluded (found disagreeing here, stored or not) or unknown (no proof).
//! An entry the inventory lacks is unknown. The receiver keeps the proof of
//! a qualified copy, rebound to its own installation, and marks every other
//! copy unfit, so it answers from the records until a check here proves it.

use std::sync::Arc;

use coordinode_modality::{CopyQualification, IndexStore as _, LocalCopy, LocalIndexStore};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::index_copies::IndexCopyQualifier;
use coordinode_storage::error::{StorageError, StorageResult};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use super::definition::{GenerationId, IndexDefinition, IndexState, IndexType};
use super::registry::IndexRegistry;

/// What a transfer says of one copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum CopyState {
    /// A check, build or qualified import proved the copy, and nothing found
    /// against it since.
    Qualified(CopyQualification),
    /// Found disagreeing with its records on the sending member.
    Excluded,
    /// No proof of the copy.
    Unknown,
}

/// One copy of the inventory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct Entry {
    generation: u64,
    state: CopyState,
}

/// The inventory as a capture keeps it until a transfer sends it: with the
/// revision of every finding this member had at the capture, to seal
/// against.
#[derive(Debug, Serialize, Deserialize)]
struct Captured {
    entries: Vec<Entry>,
    findings: Vec<(u64, u64)>,
}

/// Takes part in the engine's transfers of the index partition for the
/// member whose findings `registry` holds.
pub struct CopyQualifier {
    registry: Arc<IndexRegistry>,
}

impl CopyQualifier {
    /// The qualifier of the member whose findings `registry` holds.
    pub fn new(registry: Arc<IndexRegistry>) -> Self {
        Self { registry }
    }
}

impl IndexCopyQualifier for CopyQualifier {
    fn capture(&self, engine: &StorageEngine) -> StorageResult<Vec<u8>> {
        let store = LocalIndexStore::new(engine);
        let marked: Vec<GenerationId> = store
            .list_unfit_here()
            .map_err(storage)?
            .into_iter()
            .map(|copy| copy.generation)
            .collect();
        let proved: FxHashMap<GenerationId, CopyQualification> = store
            .list_qualified_here()
            .map_err(storage)?
            .into_iter()
            .collect();
        let findings = self.registry.findings();
        let entries = btree_generations(engine)?
            .into_iter()
            .map(|def| {
                let generation = def.generation;
                let state = if marked.contains(&generation)
                    || findings.iter().any(|(g, _)| *g == generation)
                {
                    CopyState::Excluded
                } else {
                    match proved.get(&generation) {
                        Some(proof)
                            if def.state == IndexState::Ready
                                && proof.interprets(&def)
                                && engine.installation_of(generation)
                                    == Some(proof.installation) =>
                        {
                            CopyState::Qualified(*proof)
                        }
                        _ => CopyState::Unknown,
                    }
                };
                Entry {
                    generation: generation.as_raw(),
                    state,
                }
            })
            .collect();
        let captured = Captured {
            entries,
            findings: findings
                .into_iter()
                .map(|(g, revision)| (g.as_raw(), revision))
                .collect(),
        };
        rmp_serde::to_vec(&captured).map_err(|e| encoding("encode", e))
    }

    fn seal(&self, _engine: &StorageEngine, captured: &[u8]) -> StorageResult<Vec<u8>> {
        let captured: Captured =
            rmp_serde::from_slice(captured).map_err(|e| encoding("decode", e))?;
        let then: FxHashMap<u64, u64> = captured.findings.into_iter().collect();
        let now: FxHashMap<u64, u64> = self
            .registry
            .findings()
            .into_iter()
            .map(|(g, revision)| (g.as_raw(), revision))
            .collect();
        let entries: Vec<Entry> = captured
            .entries
            .into_iter()
            .map(|mut entry| {
                // A finding made after the capture may concern what the
                // capture holds: nothing shows its damage is outside it.
                if matches!(entry.state, CopyState::Qualified(_))
                    && now.get(&entry.generation) != then.get(&entry.generation)
                {
                    entry.state = CopyState::Excluded;
                }
                entry
            })
            .collect();
        rmp_serde::to_vec(&entries).map_err(|e| encoding("encode", e))
    }

    fn installed(&self, engine: &StorageEngine, inventory: Option<&[u8]>) -> StorageResult<()> {
        take_inventory(engine, inventory)
    }

    fn copy_proof(
        &self,
        engine: &StorageEngine,
        generation: u64,
    ) -> StorageResult<Option<(Vec<u8>, u64)>> {
        let generation = GenerationId::from_raw(generation);
        let revision = self.registry.local_revision(generation);
        if revision != 0 {
            return Ok(None);
        }
        let store = LocalIndexStore::new(engine);
        if store
            .list_unfit_here()
            .map_err(storage)?
            .iter()
            .any(|copy| copy.generation == generation)
        {
            return Ok(None);
        }
        let Some(def) = btree_generations(engine)?
            .into_iter()
            .find(|d| d.generation == generation && d.state == IndexState::Ready)
        else {
            return Ok(None);
        };
        let proved = store
            .list_qualified_here()
            .map_err(storage)?
            .into_iter()
            .find(|(g, _)| *g == generation)
            .map(|(_, proof)| proof)
            .filter(|proof| {
                proof.interprets(&def)
                    && engine.installation_of(generation) == Some(proof.installation)
            });
        let Some(proof) = proved else {
            return Ok(None);
        };
        let carried = ImportProof {
            layout: proof.layout,
            maintenance: proof.maintenance,
        };
        let bytes = rmp_serde::to_vec(&carried).map_err(|e| encoding("encode", e))?;
        Ok(Some((bytes, revision)))
    }
}

/// What a history export carries of the sender's proof of its copy: the
/// interpretation the copy was proved under, alike on every member that
/// proved the same generation, so pieces from several of them still match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct ImportProof {
    layout: u32,
    maintenance: coordinode_modality::IndexMaintenance,
}

/// Whether `proof`, as a history export carried it, proves a copy of `def`:
/// the sender proved its copy under `def`'s interpretation, with nothing
/// found against it during the export.
pub fn import_proves(proof: &[u8], def: &IndexDefinition) -> bool {
    rmp_serde::from_slice::<ImportProof>(proof)
        .is_ok_and(|p| p.layout == def.layout && p.maintenance == def.maintenance)
}

/// A transfer replaced `engine`'s index copies, carrying `inventory`
/// (`None`: none, or one this build cannot read): keep the proof of every
/// copy it qualifies under the interpretation installed here, rebound to
/// this member's installation, and mark every other built copy unfit, in
/// one flush that also clears the transfer record. A generation still being
/// built is left to its build.
///
/// # Errors
///
/// The catalog could not be read, or the records could not be written.
pub(crate) fn take_inventory(
    engine: &StorageEngine,
    inventory: Option<&[u8]>,
) -> StorageResult<()> {
    let sent: FxHashMap<u64, CopyState> = match inventory {
        Some(bytes) => match rmp_serde::from_slice::<Vec<Entry>>(bytes) {
            Ok(entries) => entries
                .into_iter()
                .map(|e| (e.generation, e.state))
                .collect(),
            Err(e) => {
                tracing::warn!(error = %e, "an index inventory this build cannot read; \
                    every copy it came with is checked here");
                FxHashMap::default()
            }
        },
        None => FxHashMap::default(),
    };
    let store = LocalIndexStore::new(engine);
    let mut qualified = Vec::new();
    let mut unfit: Vec<LocalCopy> = Vec::new();
    for def in btree_generations(engine)? {
        if def.state != IndexState::Ready {
            continue;
        }
        let copy = store.bind_local_copy(def.generation).map_err(storage)?;
        match sent.get(&def.generation.as_raw()) {
            Some(CopyState::Qualified(proof)) if proof.interprets(&def) => qualified.push((
                def.generation,
                CopyQualification {
                    installation: copy.installation,
                    ..*proof
                },
            )),
            _ => unfit.push(copy),
        }
    }
    if !unfit.is_empty() {
        tracing::info!(
            copies = unfit.len(),
            qualified = qualified.len(),
            "index copies a transfer brought without proof answer from the records until \
             checked here"
        );
    }
    store.record_transfer(&qualified, &unfit).map_err(storage)
}

/// The B-tree index definitions `engine`'s catalog holds.
fn btree_generations(engine: &StorageEngine) -> StorageResult<Vec<IndexDefinition>> {
    Ok(super::ops::list_index_definitions(engine)?
        .into_iter()
        .filter(|def| def.index_type == IndexType::BTree)
        .collect())
}

fn storage(e: coordinode_modality::StoreError) -> StorageError {
    match e {
        coordinode_modality::StoreError::Storage(e) => e,
        other => StorageError::Serialization(other.to_string()),
    }
}

fn encoding(what: &str, e: impl core::fmt::Display) -> StorageError {
    StorageError::Serialization(format!("{what} an index inventory: {e}"))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
