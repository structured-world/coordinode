//! Reconciliation of the label lookup and the declared indexes for a change
//! of what a type implies.
//!
//! The change is published with a gained and a retired part for every label
//! whose carriers it moves, in the lookup and in every B-tree index on such
//! a label. From that commit every writer places what it writes by both
//! interpretations; a writer still holding an index as it was is refused at
//! its commit, since the publication moved the definition it is bound to.
//! What is left is the nodes stored before: each is read a page at a time
//! and its postings and entries placed as a writer would, a carrier the
//! change removes moving from the index's own entries to the retired part
//! and one it adds written into the gained part. Placement is idempotent, so
//! a node a writer already placed is placed again unchanged.
//!
//! A page commits only while its rows are the rows it read and every index
//! definition it placed under is the one it read, so a node changed or a
//! change concluded meanwhile is read again or ends the pass. A unique value
//! two carriers hold under the candidate ends it too: the candidate cannot
//! be put in force over them.

use coordinode_core::graph::intern::FieldInterner;
use coordinode_core::graph::node::{NodeId, NodeRecord, decode_node_key, decode_temporal_node_key};
use coordinode_core::index::derive::EntryOwner;
use coordinode_core::schema::label_id::LabelId;
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_modality::{
    GenerationId, IndexDefinition, IndexStore as _, IndexType, LabelMembership, LocalIndexStore,
    LocalNodeStore, NodeStore, PartState, StoreError,
};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::transaction::{CommitError, Transaction};

use super::build::BackfillError;
use super::membership::CatalogMembership;
use super::registry::{Carried, place_entry, record_lookup};

/// Nodes read per page.
const PAGE: usize = 512;

/// Conflicts one page may meet in a row before the pass gives up.
const MAX_PAGE_CONFLICTS: u32 = 64;

/// Where a reconciliation reads and how it opens its transactions.
pub(crate) struct Reconcile<'a> {
    /// The storage engine.
    pub engine: &'a StorageEngine,
    /// The timestamp oracle; `None` for a direct-mode engine.
    pub oracle: Option<&'a TimestampOracle>,
    /// Resolves property names to the field ids records are keyed by.
    pub interner: &'a FieldInterner,
    /// The shard whose nodes are placed.
    pub shard_id: u16,
    /// The certification whose parts are filled.
    pub operation: GenerationId,
    /// Set when the process is closing.
    pub stop: Option<&'a core::sync::atomic::AtomicBool>,
}

/// An index with parts of the certification, as one page reads it.
struct Placing {
    def: IndexDefinition,
    /// The label its parts are kept for.
    label: LabelId,
}

impl Reconcile<'_> {
    /// Place the postings and entries of every stored node of the shard
    /// whose labels the two interpretations disagree on, committing each
    /// page through `commit`. Returns the nodes placed.
    ///
    /// # Errors
    ///
    /// [`BackfillError::Superseded`] once no index carries the
    /// certification's parts as proposed (concluded or cancelled
    /// meanwhile); [`BackfillError::Duplicate`] for a unique value two
    /// carriers hold under the candidate; otherwise as a backfill.
    pub(crate) fn run(
        &self,
        commit: &mut dyn FnMut(&mut Transaction<'_>) -> Result<(), CommitError>,
    ) -> Result<u64, BackfillError> {
        let store = LocalIndexStore::new(self.engine);
        let nodes = LocalNodeStore;
        let prefix = nodes.shard_scan_prefix(self.shard_id);
        let mut in_force = CatalogMembership::current(self.engine);
        let mut candidate = CatalogMembership::candidates(self.engine);
        let field_of = |name: &str| self.interner.lookup(name);
        let mut start_after: Option<Vec<u8>> = None;
        let mut placed = 0u64;
        let mut conflicts = 0u32;
        loop {
            if self
                .stop
                .is_some_and(|s| s.load(core::sync::atomic::Ordering::Acquire))
            {
                return Err(BackfillError::Stopped);
            }
            let mut txn = match self.oracle {
                Some(oracle) => Transaction::begin(self.engine, Some(oracle), oracle.next()),
                None => Transaction::new(self.engine, None, Timestamp::ZERO, None),
            };
            // Read for each page, each bound to the record as read: one the
            // conclusion moved refuses the page.
            let mut placing: Vec<Placing> = Vec::new();
            let mut moved = false;
            for def in store.list_definitions()? {
                let Some(part) = def
                    .selection
                    .iter()
                    .find(|p| p.operation == self.operation && p.state == PartState::Proposed)
                else {
                    continue;
                };
                let label = part.label;
                let Some(version) = store.definition_version_as(&def)? else {
                    moved = true;
                    break;
                };
                txn.bind_index_definition(&def.schema_key(), Some(version))
                    .map_err(StoreError::from)?;
                placing.push(Placing { def, label });
            }
            if moved {
                continue;
            }
            if placing.is_empty() {
                return Err(BackfillError::Superseded);
            }
            let page =
                nodes.prefix_scan_paged_tracked(&mut txn, &prefix, start_after.as_deref(), PAGE)?;
            let mut read = Vec::with_capacity(page.rows.len());
            let mut page_placed = 0u64;
            for (key, bytes) in &page.rows {
                let Some(owner) = stored_owner(key) else {
                    continue;
                };
                let node = NodeId::from_raw(owner.node_id);
                let stored =
                    NodeRecord::labels_from_msgpack(bytes).map_err(|e| StoreError::Decode {
                        kind: "node labels",
                        message: format!("node {}: {e}", node.as_raw()),
                    })?;
                let now: smallvec::SmallVec<[LabelId; 8]> =
                    in_force.of_stored(&stored)?.iter().collect();
                let then: smallvec::SmallVec<[LabelId; 8]> =
                    candidate.of_stored(&stored)?.iter().collect();
                if now.len() == then.len() && now.iter().all(|l| then.contains(l)) {
                    continue;
                }
                let mut record = None;
                for Placing { def, label } in &placing {
                    let carried = Carried {
                        in_force: now.contains(label),
                        candidate: then.contains(label),
                    };
                    if carried.in_force == carried.candidate {
                        continue;
                    }
                    match def.index_type {
                        IndexType::Labels => store.stage_label_postings_txn(
                            &mut txn,
                            def,
                            self.shard_id,
                            node,
                            LabelMembership::settled(&now),
                            LabelMembership {
                                in_force: &now,
                                candidate: &then,
                            },
                        )?,
                        IndexType::BTree => {
                            if record.is_none() {
                                record = Some(NodeRecord::from_msgpack(bytes).map_err(|e| {
                                    StoreError::Decode {
                                        kind: "node record",
                                        message: format!("node {}: {e}", node.as_raw()),
                                    }
                                })?);
                            }
                            let Some(record) = record.as_ref() else {
                                continue;
                            };
                            let value_of = record_lookup(record, self.interner);
                            place_entry(
                                self.engine,
                                &mut txn,
                                self.shard_id,
                                def,
                                carried,
                                owner,
                                &value_of,
                                &field_of,
                            )?;
                        }
                        // Publication refuses a change moving the carriers
                        // of any other kind of index.
                        _ => {}
                    }
                }
                page_placed += 1;
                read.push(key.clone());
            }
            let unchanged = nodes.condition_unchanged(&mut txn, &read)?;
            let committed = unchanged
                && match commit(&mut txn) {
                    Ok(()) => true,
                    Err(CommitError::Conflict(_) | CommitError::RevisionMismatch { .. }) => false,
                    Err(e) => return Err(BackfillError::Commit(e)),
                };
            if !committed {
                conflicts += 1;
                if conflicts > MAX_PAGE_CONFLICTS {
                    return Err(BackfillError::Contended);
                }
                continue;
            }
            conflicts = 0;
            placed += page_placed;
            if page.exhausted {
                return Ok(placed);
            }
            start_after = page.last_key;
        }
    }
}

/// The owner of the entries of the stored node row under `key`: the node,
/// or for a temporal node the version the key names. `None` for a key that
/// is not a node row.
fn stored_owner(key: &[u8]) -> Option<EntryOwner> {
    match decode_node_key(key) {
        Some((_, node_id)) => Some(EntryOwner::node(node_id.as_raw())),
        None => decode_temporal_node_key(key)
            .map(|(_, node_id, valid_from)| EntryOwner::version(node_id.as_raw(), valid_from)),
    }
}

/// Register every index with parts of the certification `operation` with
/// this member's writers as its record now stands: after a commit that
/// proposed or concluded them.
///
/// # Errors
///
/// A catalog read failure.
pub(crate) fn register_parts_of(
    engine: &StorageEngine,
    registry: &super::IndexRegistry,
    operation: GenerationId,
) -> Result<(), StoreError> {
    for def in LocalIndexStore::new(engine).list_definitions()? {
        if def.selection.iter().any(|p| p.operation == operation) {
            registry
                .register_published(engine, def)
                .map_err(StoreError::from)?;
        }
    }
    Ok(())
}
