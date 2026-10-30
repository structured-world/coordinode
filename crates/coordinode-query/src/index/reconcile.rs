//! Reconciliation of the label lookup for a change of what a type implies.
//!
//! The change is published with a gained and a retired part for every label
//! whose carriers it moves. From that commit every writer places the
//! postings it writes by both interpretations; a writer still holding the
//! lookup as it was is refused at its commit, since the publication moved
//! the definition it is bound to. What is left is the nodes stored before:
//! each is read a page at a time and its postings placed as a writer would,
//! a carrier the change removes moving from the lookup's own postings to the
//! retired part and one it adds written into the gained part. Placement is
//! idempotent, so a node a writer already placed is placed again unchanged.
//!
//! A page commits only while its rows are the rows it read and the lookup
//! definition is the one it read, so a node changed or a change concluded
//! meanwhile is read again or ends the pass.

use coordinode_core::graph::node::{NodeId, NodeRecord, decode_node_key, decode_temporal_node_key};
use coordinode_core::schema::label_id::LabelId;
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_modality::{
    GenerationId, IndexStore as _, LabelMembership, LocalIndexStore, LocalNodeStore, NodeStore,
    StoreError,
};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::transaction::{CommitError, Transaction};

use super::build::BackfillError;
use super::membership::CatalogMembership;

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
    /// The shard whose nodes are placed.
    pub shard_id: u16,
    /// The certification whose parts are filled.
    pub operation: GenerationId,
    /// Set when the process is closing.
    pub stop: Option<&'a core::sync::atomic::AtomicBool>,
}

impl Reconcile<'_> {
    /// Place the postings of every stored node of the shard whose labels
    /// the two interpretations disagree on, committing each page through
    /// `commit`. Returns the nodes placed.
    ///
    /// # Errors
    ///
    /// [`BackfillError::Superseded`] once the lookup no longer carries the
    /// certification's parts as proposed (concluded or cancelled
    /// meanwhile); otherwise as a backfill.
    pub(crate) fn run(
        &self,
        commit: &mut dyn FnMut(&mut Transaction<'_>) -> Result<(), CommitError>,
    ) -> Result<u64, BackfillError> {
        let store = LocalIndexStore::new(self.engine);
        let nodes = LocalNodeStore;
        let prefix = nodes.shard_scan_prefix(self.shard_id);
        let mut in_force = CatalogMembership::current(self.engine);
        let mut candidate = CatalogMembership::candidates(self.engine);
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
            let Some(lookup) = store.label_lookup()? else {
                return Ok(placed);
            };
            if !lookup.selection.iter().any(|p| {
                p.operation == self.operation && p.state == coordinode_modality::PartState::Proposed
            }) {
                return Err(BackfillError::Superseded);
            }
            let Some(version) = store.definition_version_as(&lookup)? else {
                // Moved between the two reads: read it again.
                continue;
            };
            let mut txn = match self.oracle {
                Some(oracle) => Transaction::begin(self.engine, Some(oracle), oracle.next()),
                None => Transaction::new(self.engine, None, Timestamp::ZERO, None),
            };
            txn.bind_index_definition(&lookup.schema_key(), Some(version))
                .map_err(StoreError::from)?;
            let page =
                nodes.prefix_scan_paged_tracked(&mut txn, &prefix, start_after.as_deref(), PAGE)?;
            let mut read = Vec::with_capacity(page.rows.len());
            let mut page_placed = 0u64;
            for (key, bytes) in &page.rows {
                let Some(node) = decode_node_key(key)
                    .map(|(_, id)| id)
                    .or_else(|| decode_temporal_node_key(key).map(|(_, id, _)| id))
                else {
                    continue;
                };
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
                place(&store, &mut txn, &lookup, self.shard_id, node, &now, &then)?;
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

/// Stage node `node`'s postings as the change places them: from the
/// lookup's own postings of every label it carries now (`now`) to the
/// placement both interpretations give (`now` and `then`).
fn place(
    store: &LocalIndexStore<'_>,
    txn: &mut Transaction<'_>,
    lookup: &coordinode_modality::IndexDefinition,
    shard: u16,
    node: NodeId,
    now: &[LabelId],
    then: &[LabelId],
) -> Result<(), StoreError> {
    store.stage_label_postings_txn(
        txn,
        lookup,
        shard,
        node,
        LabelMembership::settled(now),
        LabelMembership {
            in_force: now,
            candidate: then,
        },
    )
}
