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
    /// Stored node rows one commit reads (at least one).
    pub page: usize,
    /// Bytes one commit may hold: the rows it reads, the writes it stages
    /// and the label scratch it grows.
    pub page_bytes: u64,
    /// Set when the process is closing.
    pub stop: Option<&'a core::sync::atomic::AtomicBool>,
}

/// The size of a page that did not fit: half of it, or `None` when it was a
/// single row already, which nothing smaller can hold.
fn halved(page: usize) -> Option<usize> {
    (page > 1).then_some(page / 2)
}

/// Make room in `labels` for `wanted` labels, passing `charge` the bytes its
/// capacity is about to grow by, before it grows, and adding them to `held`.
/// Emptied first, so the room asked for is `wanted` itself.
fn reserve_labels(
    labels: &mut Vec<LabelId>,
    wanted: usize,
    charge: &dyn Fn(usize) -> Result<(), coordinode_core::budget::BudgetStop>,
    held: &mut usize,
) -> Result<(), coordinode_core::budget::BudgetStop> {
    labels.clear();
    if let Some(more) = wanted.checked_sub(labels.capacity()).filter(|m| *m > 0) {
        let bytes = more * core::mem::size_of::<LabelId>();
        charge(bytes)?;
        labels.reserve_exact(wanted);
        *held += bytes;
    }
    Ok(())
}

/// An index with parts of the certification, as one page reads it.
struct Placing {
    def: IndexDefinition,
    /// The label a B-tree index's parts are kept for: its own. The label
    /// lookup keeps parts for every label the certification changes, and
    /// places a node's postings for all of them at once.
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
        let field_of = |name: &str| self.interner.lookup(name);
        let mut start_after: Option<Vec<u8>> = None;
        let mut placed = 0u64;
        let mut conflicts = 0u32;
        // Rows one page reads: halved while a page does not fit its bytes,
        // grown back towards the configured size once one does.
        let mut size = self.page.max(1);
        // The labels a node carries under each interpretation, reused from
        // node to node: they grow to the widest closure once and allocate
        // nothing after. Their bytes count against every page they outlive,
        // charged into each page's budget before the page takes anything,
        // and every growth is charged before it happens.
        let mut scratch_held = 0usize;
        let mut now: Vec<LabelId> = Vec::new();
        let mut then: Vec<LabelId> = Vec::new();
        loop {
            if self
                .stop
                .is_some_and(|s| s.load(core::sync::atomic::Ordering::Acquire))
            {
                return Err(BackfillError::Stopped);
            }
            // Both arms read at a pinned snapshot: the page's nodes and the
            // catalog it judges them by are read at it.
            let mut txn = match self.oracle {
                Some(oracle) => Transaction::begin(self.engine, Some(oracle), oracle.next()),
                None => Transaction::begin(self.engine, None, Timestamp::ZERO),
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
            // Both interpretations of the catalog as the page's own snapshot
            // reads it, the one its nodes are read at: every fact a node is
            // judged by is of one committed state, never a mix.
            let view = txn.snapshot().ok_or_else(|| {
                StoreError::Invariant("a reconciliation page reads at no snapshot".into())
            })?;
            // One budget per page, of the configured bytes: the label scratch
            // the pass still holds first, then what the page reads, keeps,
            // decodes and stages, and the catalog facts it learns, each
            // before the memory is taken.
            let page_budget =
                std::sync::Arc::new(coordinode_core::budget::QueryBudget::new(self.page_bytes));
            txn.charge_writes_to(Some(std::sync::Arc::clone(&page_budget)));
            let charge = |bytes: usize| -> Result<(), coordinode_core::budget::BudgetStop> {
                // Bytes of memory about to exist: fits in u64.
                page_budget.reserve(bytes as u64)?.keep_until_query_ends();
                Ok(())
            };
            let mut staged = || -> Result<_, BackfillError> {
                charge(scratch_held)?;
                let mut in_force =
                    CatalogMembership::at(self.engine, view).charged_to(&page_budget);
                let mut candidate =
                    CatalogMembership::candidates_at(self.engine, view).charged_to(&page_budget);
                // The properties a B-tree index the page places reads: only
                // they are decoded from a record.
                let wanted = placing
                    .iter()
                    .filter(|p| p.def.index_type == IndexType::BTree)
                    .map(|p| p.def.properties.len() + usize::from(p.def.filter.is_some()))
                    .sum::<usize>();
                charge(wanted * (core::mem::size_of::<&str>() + core::mem::size_of::<u32>()))?;
                let mut names: Vec<&str> = Vec::with_capacity(wanted);
                for Placing { def, .. } in &placing {
                    if def.index_type != IndexType::BTree {
                        continue;
                    }
                    names.extend(def.properties.iter().map(String::as_str));
                    names.extend(def.filter.as_ref().map(|f| f.property()));
                }
                let mut fields: Vec<u32> = Vec::with_capacity(wanted);
                fields.extend(names.iter().filter_map(|n| self.interner.lookup(n)));

                let page = nodes.prefix_scan_paged_charged(
                    &mut txn,
                    &prefix,
                    start_after.as_deref(),
                    size,
                    &page_budget,
                )?;
                charge(page.rows.len() * core::mem::size_of::<Vec<u8>>())?;
                let mut read = Vec::with_capacity(page.rows.len());
                let mut page_placed = 0u64;
                for (key, bytes) in &page.rows {
                    let Some(owner) = stored_owner(key) else {
                        continue;
                    };
                    let node = NodeId::from_raw(owner.node_id);
                    let decode = |e: rmp_serde::decode::Error| StoreError::Decode {
                        kind: "node record",
                        message: format!("node {}: {e}", node.as_raw()),
                    };
                    // The decoded labels live for this row only.
                    let labels_len = NodeRecord::labels_len_from_msgpack(bytes).map_err(decode)?;
                    let _labels_held = page_budget
                        .reserve((labels_len * core::mem::size_of::<LabelId>()) as u64)?;
                    let stored = NodeRecord::labels_from_msgpack(bytes).map_err(decode)?;
                    let carried_now = in_force.of_stored(&stored)?;
                    reserve_labels(&mut now, carried_now.len(), &charge, &mut scratch_held)?;
                    now.extend(carried_now.iter());
                    let carried_then = candidate.of_stored(&stored)?;
                    reserve_labels(&mut then, carried_then.len(), &charge, &mut scratch_held)?;
                    then.extend(carried_then.iter());
                    if now.len() == then.len() && now.iter().all(|l| then.contains(l)) {
                        continue;
                    }
                    // The projected record, the values an entry clones from it
                    // and the key it encodes them into live for this row
                    // only: each is at most the projection's bound.
                    let mut record_held = None;
                    let mut record = None;
                    for Placing { def, label } in &placing {
                        let carried = Carried {
                            in_force: now.contains(label),
                            candidate: then.contains(label),
                        };
                        // The lookup moves the postings of every label the two
                        // interpretations disagree on; a B-tree index only its
                        // own label's entry.
                        if def.index_type != IndexType::Labels
                            && carried.in_force == carried.candidate
                        {
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
                                    let bound =
                                        NodeRecord::projection_bound(bytes, &fields, &names)
                                            .map_err(decode)?;
                                    record_held = Some(page_budget.reserve(3 * bound as u64)?);
                                    record = Some(
                                        NodeRecord::project_from_msgpack(bytes, &fields, &names)
                                            .map_err(decode)?,
                                    );
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
                    drop(record);
                    drop(record_held);
                    page_placed += 1;
                    charge(key.len())?;
                    read.push(key.clone());
                }
                Ok((read, page_placed, page.exhausted, page.last_key))
            };
            let (read, page_placed, exhausted, last_key) = match staged() {
                Ok(page) => page,
                Err(error) => match error.budget_stop() {
                    // A page that did not fit commits nothing: half of it is
                    // tried from the same row; a single row that does not
                    // fit stops the pass.
                    Some(stop) => match halved(size) {
                        Some(half) => {
                            size = half;
                            continue;
                        }
                        None => return Err(BackfillError::Budget(stop)),
                    },
                    None => return Err(error),
                },
            };
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
            if exhausted {
                return Ok(placed);
            }
            start_after = last_key;
            size = size.saturating_mul(2).min(self.page.max(1));
        }
    }
}

/// Folding of the parts a concluded change left: put in force, the gained
/// entries move into the index's own and the retired ones are removed;
/// withdrawn, the retired entries move back and the gained ones are
/// removed. Each part is dropped from its definition once empty, what the
/// change left pending for the label's counter settled in that same
/// commit, so readers go back to one scope and the counter.
pub(crate) struct Fold<'a> {
    /// The storage engine.
    pub engine: &'a StorageEngine,
    /// The timestamp oracle; `None` for a direct-mode engine.
    pub oracle: Option<&'a TimestampOracle>,
    /// Set when the process is closing.
    pub stop: Option<&'a core::sync::atomic::AtomicBool>,
    /// The writers' registry, told each definition a step changed as soon
    /// as it commits: a writer still bound to the definition before the
    /// step is refused at its commit.
    pub registry: Option<&'a super::IndexRegistry>,
    /// Entries one fold transaction moves (at least one).
    pub page: usize,
    /// Bytes one fold transaction may stage, scratch included.
    pub page_bytes: u64,
}

impl Fold<'_> {
    /// Fold every concluded part of every index, committing each step
    /// through `commit`. Returns the parts dropped.
    ///
    /// # Errors
    ///
    /// [`BackfillError::Stopped`] once the process is closing; a storage or
    /// commit failure; [`BackfillError::Contended`] when writers kept moving
    /// what a step read.
    pub(crate) fn run(
        &self,
        commit: &mut dyn FnMut(&mut Transaction<'_>) -> Result<(), CommitError>,
    ) -> Result<u64, BackfillError> {
        let store = LocalIndexStore::new(self.engine);
        let mut dropped = 0u64;
        let mut conflicts = 0u32;
        // Entries one step moves: halved while a step does not fit its
        // bytes, grown back towards the configured size once one does.
        let mut size = self.page.max(1);
        loop {
            if self
                .stop
                .is_some_and(|s| s.load(core::sync::atomic::Ordering::Acquire))
            {
                return Err(BackfillError::Stopped);
            }
            let next = store.list_definitions()?.into_iter().find_map(|def| {
                let part = *def
                    .selection
                    .iter()
                    .find(|p| p.state != PartState::Proposed)?;
                Some((def, part))
            });
            let Some((def, part)) = next else {
                return Ok(dropped);
            };
            let Some(version) = store.definition_version_as(&def)? else {
                continue;
            };
            let mut txn = match self.oracle {
                Some(oracle) => Transaction::begin(self.engine, Some(oracle), oracle.next()),
                None => Transaction::new(self.engine, None, Timestamp::ZERO, None),
            };
            txn.bind_index_definition(&def.schema_key(), Some(version))
                .map_err(StoreError::from)?;
            // What the step stages is charged to its own bytes, each write
            // before it is buffered.
            txn.charge_writes_to(Some(std::sync::Arc::new(
                coordinode_core::budget::QueryBudget::new(self.page_bytes),
            )));
            // Gained in force and retired withdrawn hold members: moved into
            // the index's own entries. The others are read by nobody now.
            let keep = matches!(
                (part.state, part.role),
                (PartState::InForce, coordinode_modality::PartRole::Gained)
                    | (PartState::Withdrawn, coordinode_modality::PartRole::Retired)
            );
            let empty = if keep {
                match store.move_scope_page_txn(&mut txn, part.part, def.generation, size) {
                    Ok(empty) => empty,
                    Err(error) => {
                        let failure = BackfillError::from(error);
                        match failure.budget_stop() {
                            // A step that did not fit commits nothing: half
                            // of it is tried; a single entry that does not
                            // fit stops.
                            Some(stop) => match halved(size) {
                                Some(half) => {
                                    size = half;
                                    continue;
                                }
                                None => return Err(BackfillError::Budget(stop)),
                            },
                            None => return Err(failure),
                        }
                    }
                }
            } else {
                store.clear_txn(&mut txn, part.part)?;
                true
            };
            let mut next_def = None;
            if empty {
                let mut folded = def.clone();
                folded.drop_part(part.part);
                if def.index_type == IndexType::Labels && !folded.has_parts_for(part.label) {
                    settle_count(self.engine, &mut txn, part)?;
                }
                store.put_definition_txn(&mut txn, &folded)?;
                next_def = Some(folded);
            }
            match commit(&mut txn) {
                Ok(()) => {
                    conflicts = 0;
                    dropped += u64::from(empty);
                    size = size.saturating_mul(2).min(self.page.max(1));
                    if let (Some(registry), Some(folded)) = (self.registry, next_def) {
                        registry
                            .register_published(self.engine, folded)
                            .map_err(StoreError::from)?;
                    }
                    #[cfg(any(test, feature = "test-hooks"))]
                    if super::lifecycle::fold_stops_after(empty) {
                        return Err(BackfillError::Stopped);
                    }
                }
                Err(CommitError::Conflict(_) | CommitError::RevisionMismatch { .. }) => {
                    conflicts += 1;
                    if conflicts > MAX_PAGE_CONFLICTS {
                        return Err(BackfillError::Contended);
                    }
                }
                Err(e) => return Err(BackfillError::Commit(e)),
            }
        }
    }
}

/// Settle what the change `part` belongs to left pending for the count of
/// its label, in the commit dropping the label's last part: added to the
/// counter when it was put in force, dropped when withdrawn. No step reads
/// the label's postings, so the cost does not grow with the label.
fn settle_count(
    engine: &StorageEngine,
    txn: &mut Transaction<'_>,
    part: coordinode_modality::SelectionPart,
) -> Result<(), StoreError> {
    use coordinode_modality::{
        LocalSchemaStore, LocalStatsStore, SchemaStore as _, StatsStore as _,
    };
    let Some(name) = LocalSchemaStore::new(engine).label_name(part.label)? else {
        return Ok(());
    };
    LocalStatsStore.settle_pending(
        txn,
        (part.operation.as_raw(), part.label, &name),
        part.state == PartState::InForce,
    )
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

/// Stage, in the commit publishing the candidates of certification
/// `operation`, a gained and a retired part for each label of `affected`
/// (with its name) in the label lookup and in every B-tree index on it,
/// each conditioned on its record as read: from that commit every writer
/// keeps both interpretations, and one still holding an index as it was is
/// refused at its commit.
///
/// # Errors
///
/// [`StoreError::Invariant`] for a label another change is certified for,
/// for an index on an affected label of a kind that does not take moved
/// carriers, or for an index record that moved meanwhile; a storage
/// failure.
pub fn propose_parts_txn(
    engine: &StorageEngine,
    txn: &mut Transaction<'_>,
    operation: GenerationId,
    affected: &[(LabelId, String)],
) -> Result<(), StoreError> {
    if affected.is_empty() {
        return Ok(());
    }
    let indexes = LocalIndexStore::new(engine);
    for mut def in indexes.list_definitions()? {
        let labels: Vec<&(LabelId, String)> = match def.index_type {
            IndexType::Labels => affected.iter().collect(),
            _ => affected
                .iter()
                .filter(|(_, name)| *name == def.label)
                .collect(),
        };
        let Some((_, name)) = labels.first().copied() else {
            continue;
        };
        match def.index_type {
            IndexType::Labels | IndexType::BTree => {}
            // A text or vector index keeps the members of both
            // interpretations apart itself and learns where the change
            // stands from the catalog: it keeps no parts of its own.
            IndexType::Text | IndexType::Hnsw => continue,
            other => {
                return Err(StoreError::Invariant(format!(
                    "the change moves the carriers of :{name}, which the {other:?} index {} \
                     covers; an index of that kind does not take moved carriers",
                    def.name.as_deref().unwrap_or("(unnamed)")
                )));
            }
        }
        let version = indexes.definition_version_as(&def)?.ok_or_else(|| {
            StoreError::Invariant(format!(
                "an index on :{name} changed while the change was being published; retry it"
            ))
        })?;
        indexes.expect_definition_txn(txn, def.id, Some(version))?;
        for &(label, ref name) in labels {
            // An earlier change keeps parts until its fold brings them back
            // to the index's own entries, whether it is still certified or
            // already concluded: a change intersecting it waits for that.
            if def.has_parts_for(label) {
                return Err(StoreError::Invariant(format!(
                    "an earlier change moving the carriers of :{name} is not folded yet; retry \
                     once it is"
                )));
            }
            let gained = indexes.allocate_generation_txn(txn)?;
            let retired = indexes.allocate_generation_txn(txn)?;
            def.propose_parts(label, operation, gained, retired);
        }
        indexes.put_definition_txn(txn, &def)?;
    }
    Ok(())
}

/// Register every index with parts of the certification `operation` with
/// this member's writers as its record now stands: after a commit that
/// proposed or concluded them.
///
/// # Errors
///
/// A catalog read failure.
pub fn register_parts_of(
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
