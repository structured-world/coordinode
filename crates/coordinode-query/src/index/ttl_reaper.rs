//! COMPUTED TTL background reaper: deletes expired nodes/fields/subtrees.
//!
//! Scans all label schemas for `PropertyType::Computed(ComputedSpec::Ttl {...})`
//! properties, finds nodes where `anchor_field + duration_secs < now`, and
//! deletes according to `TtlScope`:
//!
//! - **Node**: delete entire node + all edges (DETACH DELETE)
//! - **Field**: remove just the anchor property from the node
//! - **Subtree**: remove the DOCUMENT property (nested content)
//!
//! The reaper is rate-limited to `batch_size` (default 1000) deletions per
//! pass to avoid write stalls. Runs every `interval_secs` (default 60).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use coordinode_core::graph::edge::{encode_adj_key_forward, encode_adj_key_reverse};
use coordinode_core::graph::node::{NodeId, NodeRecord};
use coordinode_core::graph::types::Value;
use coordinode_core::schema::computed::{ComputedSpec, TtlScope};
#[cfg(test)]
use coordinode_core::schema::definition::LabelSchema;
use coordinode_core::schema::definition::PropertyType;
use coordinode_core::txn::proposal::{
    Mutation, PartitionId, ProposalIdGenerator, ProposalPipeline, RaftProposal,
};
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_modality::{
    EdgeStore as _, LocalEdgeStore, LocalNodeStore, LocalSchemaStore, NodeStore as _,
    SchemaStore as _,
};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::merge::encode_remove;

/// Configuration for the COMPUTED TTL background reaper.
#[derive(Debug, Clone)]
pub struct TtlReaperConfig {
    /// Reaper scan interval in seconds. Default: 60.
    pub interval_secs: u64,
    /// Maximum deletions per reaper pass. Default: 1000.
    pub batch_size: usize,
    /// Whether the reaper is enabled. Default: true.
    pub enabled: bool,
}

impl Default for TtlReaperConfig {
    fn default() -> Self {
        Self {
            interval_secs: 60,
            batch_size: 1000,
            enabled: true,
        }
    }
}

/// Result of a single COMPUTED TTL reap pass.
#[derive(Debug, Default)]
pub struct ComputedTtlReapResult {
    /// Labels scanned.
    pub labels_scanned: usize,
    /// Total nodes checked.
    pub nodes_checked: usize,
    /// Nodes deleted (scope: Node).
    pub nodes_deleted: usize,
    /// Fields removed (scope: Field).
    pub fields_removed: usize,
    /// Subtrees removed (scope: Subtree).
    pub subtrees_removed: usize,
    /// Non-fatal errors encountered.
    pub errors: Vec<String>,
}

impl ComputedTtlReapResult {
    /// Total number of deletions performed.
    pub fn total_deletions(&self) -> usize {
        self.nodes_deleted + self.fields_removed + self.subtrees_removed
    }
}

/// A COMPUTED TTL property found in a label schema.
struct TtlTarget {
    label: String,
    _property_name: String,
    duration_secs: u64,
    anchor_field: String,
    /// Resolved field ID for the anchor field (from interner).
    /// `None` = interner didn't have this name (fall back to heuristic scan).
    anchor_field_id: Option<u32>,
    scope: TtlScope,
    /// For `scope = Subtree`: the DOCUMENT property to delete on expiry.
    /// If `None`, falls back to deleting `anchor_field` (same as `Field` scope).
    target_field: Option<String>,
    /// Resolved field ID for `target_field` (from interner).
    /// `None` when no interner or field not yet interned.
    target_field_id: Option<u32>,
}

/// Run a single COMPUTED TTL reap pass (no interner, no pipeline — direct engine writes).
///
/// Convenience wrapper for tests and simple embedded use. For production,
/// prefer `reap_computed_ttl_via_pipeline`.
pub fn reap_computed_ttl(
    engine: &StorageEngine,
    shard_id: u16,
    batch_size: usize,
) -> ComputedTtlReapResult {
    reap_computed_ttl_inner(engine, shard_id, batch_size, None, None, None)
}

/// Run a single COMPUTED TTL reap pass with interner (no pipeline — direct engine writes).
pub fn reap_computed_ttl_with_interner(
    engine: &StorageEngine,
    shard_id: u16,
    batch_size: usize,
    interner: &coordinode_core::graph::intern::FieldInterner,
) -> ComputedTtlReapResult {
    reap_computed_ttl_inner(engine, shard_id, batch_size, Some(interner), None, None)
}

/// Run a single COMPUTED TTL reap pass through ProposalPipeline.
///
/// All mutations are batched into `RaftProposal`s and submitted via the
/// pipeline. In embedded mode this is equivalent to direct writes; in
/// cluster mode mutations are replicated via Raft.
pub fn reap_computed_ttl_via_pipeline(
    engine: &StorageEngine,
    shard_id: u16,
    batch_size: usize,
    interner: &coordinode_core::graph::intern::FieldInterner,
    pipeline: &dyn ProposalPipeline,
    id_gen: &ProposalIdGenerator,
) -> ComputedTtlReapResult {
    reap_computed_ttl_inner(
        engine,
        shard_id,
        batch_size,
        Some(interner),
        Some(pipeline),
        Some(id_gen),
    )
}

/// Shared implementation.
fn reap_computed_ttl_inner(
    engine: &StorageEngine,
    shard_id: u16,
    batch_size: usize,
    interner: Option<&coordinode_core::graph::intern::FieldInterner>,
    pipeline: Option<&dyn ProposalPipeline>,
    id_gen: Option<&ProposalIdGenerator>,
) -> ComputedTtlReapResult {
    let mut result = ComputedTtlReapResult::default();

    let targets = match discover_ttl_targets(engine, interner) {
        Ok(t) => t,
        Err(e) => {
            result.errors.push(format!("schema scan error: {e}"));
            return result;
        }
    };

    if targets.is_empty() {
        return result;
    }

    let now_us = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0);

    let edge_types = list_edge_types(engine);
    let mut total_deletions = 0usize;

    for target in &targets {
        result.labels_scanned += 1;
        if total_deletions >= batch_size {
            break;
        }
        let remaining = batch_size - total_deletions;
        let pass_result = reap_label(
            engine,
            shard_id,
            target,
            now_us,
            remaining,
            &edge_types,
            pipeline,
            id_gen,
        );
        result.nodes_checked += pass_result.nodes_checked;
        result.nodes_deleted += pass_result.nodes_deleted;
        result.fields_removed += pass_result.fields_removed;
        result.subtrees_removed += pass_result.subtrees_removed;
        total_deletions += pass_result.total_deletions();
        result.errors.extend(pass_result.errors);
    }

    result
}

/// Scan label schemas and find all COMPUTED TTL properties.
/// When `interner` is provided, resolves anchor field names to field IDs
/// for correct multi-Timestamp resolution.
///
/// Uses the typed [`SchemaStore::list_labels`] API — surfaces every
/// declared label at its **current** revision (the prior raw-prefix
/// scan also visited historical revisions, which was wasteful and
/// could mis-report TTLs from superseded schemas).
fn discover_ttl_targets(
    engine: &StorageEngine,
    interner: Option<&coordinode_core::graph::intern::FieldInterner>,
) -> Result<Vec<TtlTarget>, String> {
    let schemas = LocalSchemaStore::new(engine)
        .list_labels()
        .map_err(|e| e.to_string())?;

    let mut targets = Vec::new();
    for schema in schemas {
        let label_name = schema.name.clone();
        for (prop_name, prop_def) in &schema.properties {
            if let PropertyType::Computed(ComputedSpec::Ttl {
                duration_secs,
                anchor_field,
                scope,
                target_field,
            }) = &prop_def.property_type
            {
                let anchor_field_id = interner.and_then(|int| int.lookup(anchor_field));
                let target_field_id =
                    interner.and_then(|int| target_field.as_deref().and_then(|tf| int.lookup(tf)));
                targets.push(TtlTarget {
                    label: label_name.clone(),
                    _property_name: prop_name.clone(),
                    duration_secs: *duration_secs,
                    anchor_field: anchor_field.clone(),
                    anchor_field_id,
                    scope: *scope,
                    target_field: target_field.clone(),
                    target_field_id,
                });
            }
        }
    }

    Ok(targets)
}

/// List all registered edge type names via the schema store. Versioned keys
/// (`schema:edge_type:<name>:<version>`) dedup to one entry per name.
fn list_edge_types(engine: &StorageEngine) -> Vec<String> {
    LocalSchemaStore::new(engine)
        .list_edge_type_names_engine()
        .unwrap_or_default()
}

/// Run reaper for one label+TTL target. Collects mutations and submits
/// via pipeline (if provided) or applies directly to engine.
#[allow(clippy::too_many_arguments)]
fn reap_label(
    engine: &StorageEngine,
    shard_id: u16,
    target: &TtlTarget,
    now_us: i64,
    max_deletions: usize,
    edge_types: &[String],
    pipeline: Option<&dyn ProposalPipeline>,
    id_gen: Option<&ProposalIdGenerator>,
) -> ComputedTtlReapResult {
    let mut result = ComputedTtlReapResult::default();
    let mut pending: Vec<Mutation> = Vec::new();

    // Guard: if Subtree scope specifies a target_field that was not in the
    // interner snapshot at startup, skip the entire label scan and record a
    // single diagnostic.  Continuing into the node loop would emit one
    // identical error per expired node and scan the entire shard uselessly —
    // target_field_id is fixed per TtlTarget and cannot become Some mid-pass.
    if let (TtlScope::Subtree, Some(tf), None) = (
        target.scope,
        target.target_field.as_deref(),
        target.target_field_id,
    ) {
        result.errors.push(format!(
            "label {}: target_field '{tf}' not in interner — Subtree deletion skipped",
            target.label,
        ));
        return result;
    }

    let cutoff_us = now_us - (target.duration_secs as i64 * 1_000_000);

    // Walk the shard through the node store (it owns the key encoding and the
    // partition); the reaper only turns each expired record into mutations.
    let scan = LocalNodeStore.for_each_in_shard_at_snapshot(
        engine,
        None,
        shard_id,
        &mut |node_id, key, record| {
            if result.total_deletions() >= max_deletions {
                return Ok(std::ops::ControlFlow::Break(()));
            }
            if !record.labels.contains(&target.label) {
                return Ok(std::ops::ControlFlow::Continue(()));
            }
            result.nodes_checked += 1;

            let anchor_us =
                match resolve_anchor(record, &target.anchor_field, target.anchor_field_id) {
                    Some(ts) => ts,
                    None => return Ok(std::ops::ControlFlow::Continue(())),
                };
            if anchor_us >= cutoff_us {
                return Ok(std::ops::ControlFlow::Continue(()));
            }

            let nid_raw = node_id.as_raw();
            match target.scope {
                TtlScope::Node => {
                    match collect_node_deletion_mutations(engine, nid_raw, key, edge_types) {
                        Ok(mutations) => {
                            pending.extend(mutations);
                            result.nodes_deleted += 1;
                        }
                        Err(e) => result.errors.push(format!(
                            "collect node {nid_raw} (label {}): {e}",
                            target.label
                        )),
                    }
                }
                TtlScope::Field => {
                    match collect_property_removal_mutations(
                        engine,
                        key,
                        target.anchor_field_id,
                        &target.anchor_field,
                    ) {
                        Ok(mutations) => {
                            pending.extend(mutations);
                            result.fields_removed += 1;
                        }
                        Err(e) => result.errors.push(format!(
                            "collect remove {} from node {nid_raw}: {e}",
                            target.anchor_field
                        )),
                    }
                }
                TtlScope::Subtree => {
                    // Subtree: delete `target_field` if specified and resolved,
                    // otherwise fall back to the anchor field (same as Field scope).
                    //
                    // When `target_field` is Some but `target_field_id` is None the
                    // field name was not present in the interner snapshot given to the
                    // reaper at startup — this can happen in production when a schema
                    // with a new `target_field` is added after the database is opened
                    // and no nodes carrying that field have been written yet.
                    // Skipping is safer than letting the timestamp heuristic in
                    // `collect_property_removal_mutations` remove the wrong field.
                    // The skip is recorded in `result.errors` so operators can detect
                    // the stale interner condition.
                    // Guard at reap_label entry already returns early when
                    // target_field is Some but target_field_id is None, so the
                    // (Some(_), None) arm is unreachable here.
                    let to_delete: Option<(Option<u32>, &str)> =
                        match (&target.target_field, target.target_field_id) {
                            (Some(tf), Some(field_id)) => {
                                // Skip if the target field is already absent — anchor_field
                                // is preserved by design, so the node stays visible to the
                                // reaper on every pass.  Without this check, subtrees_removed
                                // would be incremented and a (no-op) merge mutation submitted
                                // on every reap cycle after the first deletion.
                                if record.props.contains_key(&field_id) {
                                    Some((target.target_field_id, tf.as_str()))
                                } else {
                                    None
                                }
                            }
                            (Some(_), None) => unreachable!(
                                "unresolved target_field should have been caught by early return"
                            ),
                            (None, _) => {
                                Some((target.anchor_field_id, target.anchor_field.as_str()))
                            }
                        };

                    if let Some((del_field_id, del_field_name)) = to_delete {
                        match collect_property_removal_mutations(
                            engine,
                            key,
                            del_field_id,
                            del_field_name,
                        ) {
                            Ok(mutations) => {
                                pending.extend(mutations);
                                result.subtrees_removed += 1;
                            }
                            Err(e) => result.errors.push(format!(
                                "collect remove {del_field_name} from node {nid_raw}: {e}",
                            )),
                        }
                    }
                }
            }

            // Flush batch when full.
            if pending.len() >= max_deletions {
                if let Err(e) = submit_mutations(&mut pending, engine, pipeline, id_gen) {
                    result.errors.push(format!("submit batch: {e}"));
                }
            }
            Ok(std::ops::ControlFlow::Continue(()))
        },
    );
    if let Err(e) = scan {
        result.errors.push(format!("node scan error: {e}"));
    }

    // Flush remaining.
    if !pending.is_empty() {
        if let Err(e) = submit_mutations(&mut pending, engine, pipeline, id_gen) {
            result.errors.push(format!("submit final batch: {e}"));
        }
    }

    result
}

/// Submit collected mutations: via pipeline if available, else direct engine writes.
fn submit_mutations(
    mutations: &mut Vec<Mutation>,
    engine: &StorageEngine,
    pipeline: Option<&dyn ProposalPipeline>,
    id_gen: Option<&ProposalIdGenerator>,
) -> Result<(), String> {
    if mutations.is_empty() {
        return Ok(());
    }
    let batch = std::mem::take(mutations);

    if let (Some(pipe), Some(gen)) = (pipeline, id_gen) {
        let proposal = RaftProposal {
            id: gen.next(),
            mutations: batch,
            commit_ts: Timestamp::from_raw(0),
            start_ts: Timestamp::from_raw(0),
            bypass_rate_limiter: true, // background maintenance
        };
        pipe.propose_and_wait(&proposal)
            .map_err(|e| e.to_string())?;
    } else {
        // No pipeline (single-node / tests): apply each mutation straight to
        // the engine. The storage layer owns the PartitionId -> Partition
        // bridge, so this background path never names a partition.
        for mutation in &batch {
            engine.apply_mutation(mutation).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

/// Resolve anchor timestamp: uses interner when available, falls back
/// to heuristic scan (first Timestamp found).
/// Resolve anchor timestamp using pre-resolved field_id when available,
/// falling back to heuristic scan.
fn resolve_anchor(
    record: &NodeRecord,
    anchor_name: &str,
    anchor_field_id: Option<u32>,
) -> Option<i64> {
    // Fast path: use pre-resolved field ID from discover_ttl_targets.
    if let Some(field_id) = anchor_field_id {
        if let Some(value) = record.props.get(&field_id) {
            match value {
                Value::Timestamp(ts) => return Some(*ts),
                Value::Int(ts) => return Some(*ts),
                _ => {}
            }
        }
        // Field exists in interner but not in this node → check extra map.
        if let Some(extra) = &record.extra {
            if let Some(val) = extra.get(anchor_name) {
                match val {
                    Value::Timestamp(ts) => return Some(*ts),
                    Value::Int(ts) => return Some(*ts),
                    _ => {}
                }
            }
        }
        return None;
    }

    // Slow path: no interner → heuristic scan.
    find_anchor_timestamp(record, anchor_name)
}

/// Find a Timestamp value in a NodeRecord by scanning all properties.
///
/// Since we don't have the FieldInterner context, we cannot resolve field IDs
/// to names. Instead, we check all Timestamp values — the anchor field will
/// be among them. For nodes with multiple Timestamp fields, this is an
/// approximation. However, the interner-aware path is used when available.
///
/// Also checks the `extra` overflow map (VALIDATED schema mode).
fn find_anchor_timestamp(record: &NodeRecord, _anchor_name: &str) -> Option<i64> {
    // Check regular props — field IDs are opaque without interner,
    // so we take the FIRST Timestamp value found. This is correct for
    // schemas with exactly one Timestamp field (the common case for TTL).
    //
    // For schemas with multiple Timestamp fields, we'd need the interner
    // to resolve names. See find_anchor_timestamp_with_interner() below.
    for value in record.props.values() {
        if let Value::Timestamp(ts) = value {
            return Some(*ts);
        }
        if let Value::Int(ts) = value {
            // Cypher literals create Int, not Timestamp. Accept both.
            return Some(*ts);
        }
    }

    // Check extra overflow map (VALIDATED mode) — string-keyed.
    if let Some(extra) = &record.extra {
        for (key, val) in extra {
            if key == _anchor_name {
                match val {
                    Value::Timestamp(ts) => return Some(*ts),
                    Value::Int(ts) => return Some(*ts),
                    _ => {}
                }
            }
        }
    }

    None
}

/// Find anchor timestamp using interner for correct field name resolution.
pub fn find_anchor_timestamp_with_interner(
    record: &NodeRecord,
    anchor_name: &str,
    interner: &coordinode_core::graph::intern::FieldInterner,
) -> Option<i64> {
    // Resolve anchor field name → field ID.
    let field_id = interner.lookup(anchor_name)?;

    if let Some(value) = record.props.get(&field_id) {
        match value {
            Value::Timestamp(ts) => return Some(*ts),
            Value::Int(ts) => return Some(*ts),
            _ => {}
        }
    }

    // Check extra overflow map.
    if let Some(extra) = &record.extra {
        if let Some(val) = extra.get(anchor_name) {
            match val {
                Value::Timestamp(ts) => return Some(*ts),
                Value::Int(ts) => return Some(*ts),
                _ => {}
            }
        }
    }

    None
}

/// Collect mutations for DETACH DELETE of a node (node + all edges).
/// Reads are from engine; writes are returned as `Vec<Mutation>`.
fn collect_node_deletion_mutations(
    engine: &StorageEngine,
    node_id: u64,
    node_key: &[u8],
    edge_types: &[String],
) -> Result<Vec<Mutation>, String> {
    let mut mutations = Vec::new();
    let nid = NodeId::from_raw(node_id);

    for edge_type in edge_types {
        for (adj_key, is_outgoing) in [
            (encode_adj_key_forward(edge_type, nid), true),
            (encode_adj_key_reverse(edge_type, nid), false),
        ] {
            if let Ok(Some(plist)) = LocalEdgeStore.posting_at_snapshot(engine, None, &adj_key) {
                for peer_uid in plist.iter() {
                    let peer_id = NodeId::from_raw(peer_uid);
                    let counterpart = if is_outgoing {
                        encode_adj_key_reverse(edge_type, peer_id)
                    } else {
                        encode_adj_key_forward(edge_type, peer_id)
                    };

                    // Merge-remove this node from peer's posting list.
                    mutations.push(Mutation::Merge {
                        partition: PartitionId::Adj,
                        key: counterpart,
                        operand: encode_remove(node_id),
                    });

                    // Delete edge properties.
                    let (src, tgt) = if is_outgoing {
                        (nid, peer_id)
                    } else {
                        (peer_id, nid)
                    };
                    mutations.push(Mutation::delete_edge_props(edge_type, src, tgt));
                }

                // Delete the adj key.
                mutations.push(Mutation::Delete {
                    partition: PartitionId::Adj,
                    key: adj_key,
                });
            }
        }
    }

    // Delete the node record.
    mutations.push(Mutation::Delete {
        partition: PartitionId::Node,
        key: node_key.to_vec(),
    });

    Ok(mutations)
}

/// Collect a Merge mutation that removes a property via `DocDelta::RemoveProperty`.
fn collect_property_removal_mutations(
    engine: &StorageEngine,
    node_key: &[u8],
    anchor_field_id: Option<u32>,
    anchor_field_name: &str,
) -> Result<Vec<Mutation>, String> {
    use coordinode_core::graph::doc_delta::{DocDelta, PathTarget};

    let delta = if let Some(field_id) = anchor_field_id {
        DocDelta::RemoveProperty {
            target: PathTarget::PropField(field_id),
            key: None,
        }
    } else {
        // No interner — read node to find field_id, then emit merge operand.
        let bytes = LocalNodeStore
            .read_raw_at_snapshot(engine, None, node_key)
            .map_err(|e| e.to_string())?;
        match bytes {
            Some(data) => {
                let record = NodeRecord::from_msgpack(&data).map_err(|e| e.to_string())?;
                let ts_field_id = record
                    .props
                    .iter()
                    .find(|(_, v)| matches!(v, Value::Timestamp(_) | Value::Int(_)))
                    .map(|(k, _)| *k);

                if let Some(fid) = ts_field_id {
                    DocDelta::RemoveProperty {
                        target: PathTarget::PropField(fid),
                        key: None,
                    }
                } else {
                    DocDelta::RemoveProperty {
                        target: PathTarget::Extra,
                        key: Some(anchor_field_name.to_string()),
                    }
                }
            }
            None => return Ok(Vec::new()), // Node already deleted.
        }
    };

    let operand = delta.encode().map_err(|e| e.to_string())?;
    Ok(vec![Mutation::Merge {
        partition: PartitionId::Node,
        key: node_key.to_vec(),
        operand,
    }])
}

/// Background reaper handle. Spawns a thread that periodically runs
/// `reap_computed_ttl`. Stopped on drop (graceful shutdown).
pub struct TtlReaperHandle {
    shutdown: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl TtlReaperHandle {
    /// Start the background COMPUTED TTL reaper thread.
    ///
    /// The thread runs until `shutdown()` is called or the handle is dropped.
    /// `interner`: cloned snapshot of the `FieldInterner` at Database open time.
    /// `pipeline`: proposal pipeline for cluster-replicated writes.
    /// `id_gen`: proposal ID generator (shared with other pipeline users).
    pub fn start(
        engine: Arc<StorageEngine>,
        shard_id: u16,
        config: TtlReaperConfig,
        interner: coordinode_core::graph::intern::FieldInterner,
        pipeline: Arc<dyn ProposalPipeline>,
        id_gen: Arc<ProposalIdGenerator>,
    ) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = Arc::clone(&shutdown);

        let thread = match std::thread::Builder::new()
            .name("coordinode-ttl-reaper".into())
            .spawn(move || {
                reaper_loop(
                    &engine,
                    shard_id,
                    &config,
                    &shutdown_clone,
                    &interner,
                    pipeline.as_ref(),
                    &id_gen,
                );
            }) {
            Ok(t) => Some(t),
            Err(e) => {
                tracing::error!("ttl_reaper: failed to spawn thread: {e}");
                None
            }
        };

        Self { shutdown, thread }
    }

    /// Signal the reaper thread to stop.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
    }
}

impl Drop for TtlReaperHandle {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Main reaper loop: sleep → scan → delete → repeat.
fn reaper_loop(
    engine: &StorageEngine,
    shard_id: u16,
    config: &TtlReaperConfig,
    shutdown: &AtomicBool,
    interner: &coordinode_core::graph::intern::FieldInterner,
    pipeline: &dyn ProposalPipeline,
    id_gen: &ProposalIdGenerator,
) {
    let interval = Duration::from_secs(config.interval_secs);
    tracing::info!(
        "ttl_reaper: started (interval={}s, batch_size={})",
        config.interval_secs,
        config.batch_size,
    );

    loop {
        // Sleep in small increments to check shutdown flag.
        let sleep_step = Duration::from_millis(500);
        let mut slept = Duration::ZERO;
        while slept < interval {
            if shutdown.load(Ordering::Acquire) {
                tracing::info!("ttl_reaper: shutdown");
                return;
            }
            std::thread::sleep(sleep_step.min(interval - slept));
            slept += sleep_step;
        }

        if shutdown.load(Ordering::Acquire) {
            tracing::info!("ttl_reaper: shutdown");
            return;
        }

        let result = reap_computed_ttl_via_pipeline(
            engine,
            shard_id,
            config.batch_size,
            interner,
            pipeline,
            id_gen,
        );

        if result.total_deletions() > 0 || !result.errors.is_empty() {
            tracing::info!(
                "ttl_reaper: pass complete — checked={}, deleted_nodes={}, \
                 removed_fields={}, removed_subtrees={}, errors={}",
                result.nodes_checked,
                result.nodes_deleted,
                result.fields_removed,
                result.subtrees_removed,
                result.errors.len(),
            );
        }

        for err in &result.errors {
            tracing::warn!("ttl_reaper: {err}");
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
// Tests plant raw fixtures (nodes, adjacency, edge-type markers) via the
// storage partition.
#[allow(clippy::disallowed_types)]
mod tests;
