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

use coordinode_core::graph::edge::{encode_adj_key_forward, encode_adj_key_reverse, PostingList};
use coordinode_core::graph::node::{NodeId, NodeRecord};
use coordinode_core::graph::types::Value;
use coordinode_core::schema::computed::{ComputedSpec, TtlScope};
use coordinode_core::schema::definition::{LabelSchema, PropertyType};
use coordinode_core::txn::proposal::{
    Mutation, PartitionId, ProposalIdGenerator, ProposalPipeline, RaftProposal,
};
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::merge::encode_remove;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::Guard;

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
fn discover_ttl_targets(
    engine: &StorageEngine,
    interner: Option<&coordinode_core::graph::intern::FieldInterner>,
) -> Result<Vec<TtlTarget>, String> {
    let prefix = b"schema:label:";
    let iter = engine
        .prefix_scan(Partition::Schema, prefix)
        .map_err(|e| e.to_string())?;

    let mut targets = Vec::new();

    for guard in iter {
        let (key, value) = match guard.into_inner() {
            Ok(kv) => kv,
            Err(e) => {
                tracing::warn!("ttl_reaper: failed to read schema entry: {e}");
                continue;
            }
        };

        let label_name = match std::str::from_utf8(&key[prefix.len()..]) {
            Ok(s) => s.to_string(),
            Err(_) => continue,
        };

        let schema: LabelSchema = match LabelSchema::from_msgpack(&value) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("ttl_reaper: failed to parse schema for {label_name}: {e}");
                continue;
            }
        };

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

/// List all registered edge types from schema partition.
fn list_edge_types(engine: &StorageEngine) -> Vec<String> {
    let prefix = b"schema:edge_type:";
    let iter = match engine.prefix_scan(Partition::Schema, prefix) {
        Ok(it) => it,
        Err(_) => return Vec::new(),
    };

    let mut types = Vec::new();
    for guard in iter {
        if let Ok((key, _)) = guard.into_inner() {
            if let Ok(name) = std::str::from_utf8(&key[prefix.len()..]) {
                types.push(name.to_string());
            }
        }
    }
    types
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

    let node_prefix = {
        let mut p = Vec::with_capacity(8);
        p.extend_from_slice(b"node:");
        p.extend_from_slice(&shard_id.to_be_bytes());
        p.push(b':');
        p
    };

    let iter = match engine.prefix_scan(Partition::Node, &node_prefix) {
        Ok(it) => it,
        Err(e) => {
            result.errors.push(format!("node scan error: {e}"));
            return result;
        }
    };

    for guard in iter {
        if result.total_deletions() >= max_deletions {
            break;
        }

        let (key, value) = match guard.into_inner() {
            Ok(kv) => kv,
            Err(e) => {
                result.errors.push(format!("node read error: {e}"));
                continue;
            }
        };

        let record = match NodeRecord::from_msgpack(&value) {
            Ok(r) => r,
            Err(_) => continue,
        };

        if !record.labels.contains(&target.label) {
            continue;
        }

        result.nodes_checked += 1;

        let anchor_us = match resolve_anchor(&record, &target.anchor_field, target.anchor_field_id)
        {
            Some(ts) => ts,
            None => continue,
        };

        if anchor_us >= cutoff_us {
            continue;
        }

        let node_id = decode_node_id_from_key(&key);

        match target.scope {
            TtlScope::Node => {
                match collect_node_deletion_mutations(engine, node_id, &key, edge_types) {
                    Ok(mutations) => {
                        pending.extend(mutations);
                        result.nodes_deleted += 1;
                    }
                    Err(e) => {
                        result.errors.push(format!(
                            "collect node {node_id} (label {}): {e}",
                            target.label
                        ));
                    }
                }
            }
            TtlScope::Field => {
                match collect_property_removal_mutations(
                    engine,
                    &key,
                    target.anchor_field_id,
                    &target.anchor_field,
                ) {
                    Ok(mutations) => {
                        pending.extend(mutations);
                        result.fields_removed += 1;
                    }
                    Err(e) => {
                        result.errors.push(format!(
                            "collect remove {} from node {node_id}: {e}",
                            target.anchor_field
                        ));
                    }
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
                        (None, _) => Some((target.anchor_field_id, target.anchor_field.as_str())),
                    };

                if let Some((del_field_id, del_field_name)) = to_delete {
                    match collect_property_removal_mutations(
                        engine,
                        &key,
                        del_field_id,
                        del_field_name,
                    ) {
                        Ok(mutations) => {
                            pending.extend(mutations);
                            result.subtrees_removed += 1;
                        }
                        Err(e) => {
                            result.errors.push(format!(
                                "collect remove {del_field_name} from node {node_id}: {e}",
                            ));
                        }
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
        for mutation in &batch {
            apply_mutation_directly(engine, mutation)?;
        }
    }
    Ok(())
}

/// Apply a single mutation directly to the storage engine (no pipeline).
fn apply_mutation_directly(engine: &StorageEngine, mutation: &Mutation) -> Result<(), String> {
    match mutation {
        Mutation::Put {
            partition,
            key,
            value,
        } => engine
            .put(to_partition(*partition), key, value)
            .map_err(|e| e.to_string()),
        Mutation::Delete { partition, key } => engine
            .delete(to_partition(*partition), key)
            .map_err(|e| e.to_string()),
        Mutation::Merge {
            partition,
            key,
            operand,
        } => engine
            .merge(to_partition(*partition), key, operand)
            .map_err(|e| e.to_string()),
    }
}

fn to_partition(id: PartitionId) -> Partition {
    match id {
        PartitionId::Node => Partition::Node,
        PartitionId::Adj => Partition::Adj,
        PartitionId::EdgeProp => Partition::EdgeProp,
        PartitionId::Blob => Partition::Blob,
        PartitionId::BlobRef => Partition::BlobRef,
        PartitionId::Schema => Partition::Schema,
        PartitionId::Idx => Partition::Idx,
        PartitionId::Counter => Partition::Counter,
    }
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
            if let Ok(Some(bytes)) = engine.get(Partition::Adj, &adj_key) {
                if let Ok(plist) = PostingList::from_bytes(&bytes) {
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
                        let ep_key =
                            coordinode_core::graph::edge::encode_edgeprop_key(edge_type, src, tgt);
                        mutations.push(Mutation::Delete {
                            partition: PartitionId::EdgeProp,
                            key: ep_key,
                        });
                    }
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
        let bytes = engine
            .get(Partition::Node, node_key)
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

/// Extract node ID from a node: key.
/// Key format: `node:` (5) + shard_id BE (2) + `:` (1) + node_id BE (8)
fn decode_node_id_from_key(key: &[u8]) -> u64 {
    if key.len() >= 16 {
        let id_bytes = &key[8..16];
        u64::from_be_bytes([
            id_bytes[0],
            id_bytes[1],
            id_bytes[2],
            id_bytes[3],
            id_bytes[4],
            id_bytes[5],
            id_bytes[6],
            id_bytes[7],
        ])
    } else {
        0
    }
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
mod tests {
    use super::*;
    use coordinode_core::graph::intern::FieldInterner;
    use coordinode_core::graph::node::encode_node_key;
    use coordinode_core::schema::definition::PropertyDef;
    use coordinode_storage::engine::config::StorageConfig;

    fn test_engine(dir: &std::path::Path) -> StorageEngine {
        let config = StorageConfig::new(dir);
        StorageEngine::open(&config).expect("open engine")
    }

    fn persist_schema(engine: &StorageEngine, schema: &LabelSchema) {
        let key = coordinode_core::schema::definition::encode_label_schema_key(&schema.name);
        let bytes = schema.to_msgpack().expect("serialize schema");
        engine
            .put(Partition::Schema, &key, &bytes)
            .expect("persist schema");
    }

    fn insert_node(
        engine: &StorageEngine,
        shard_id: u16,
        node_id: u64,
        label: &str,
        timestamp_us: i64,
        interner: &mut FieldInterner,
    ) {
        let mut record = NodeRecord::new(label);
        let ts_field = interner.intern("created_at");
        record.set(ts_field, Value::Timestamp(timestamp_us));
        let key = encode_node_key(shard_id, NodeId::from_raw(node_id));
        let bytes = record.to_msgpack().expect("serialize");
        engine.put(Partition::Node, &key, &bytes).expect("put node");
    }

    fn node_exists(engine: &StorageEngine, shard_id: u16, node_id: u64) -> bool {
        let key = encode_node_key(shard_id, NodeId::from_raw(node_id));
        engine.get(Partition::Node, &key).expect("get").is_some()
    }

    fn make_ttl_schema(label: &str, duration_secs: u64, scope: TtlScope) -> LabelSchema {
        let mut schema = LabelSchema::new(label);
        schema.add_property(PropertyDef::new("content", PropertyType::String));
        schema.add_property(PropertyDef::new("created_at", PropertyType::Timestamp));
        schema.add_property(PropertyDef::computed(
            "_ttl",
            ComputedSpec::Ttl {
                duration_secs,
                anchor_field: "created_at".into(),
                scope,
                target_field: None,
            },
        ));
        schema
    }

    fn now_us() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as i64
    }

    // ── discover_ttl_targets ─────────────────────────────────────────

    #[test]
    fn discover_finds_ttl_properties() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());

        let schema = make_ttl_schema("Session", 3600, TtlScope::Node);
        persist_schema(&engine, &schema);

        let targets = discover_ttl_targets(&engine, None).expect("discover");
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].label, "Session");
        assert_eq!(targets[0].duration_secs, 3600);
        assert_eq!(targets[0].scope, TtlScope::Node);
    }

    #[test]
    fn discover_ignores_non_ttl_labels() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());

        // Schema without COMPUTED TTL.
        let mut schema = LabelSchema::new("User");
        schema.add_property(PropertyDef::new("name", PropertyType::String));
        persist_schema(&engine, &schema);

        let targets = discover_ttl_targets(&engine, None).expect("discover");
        assert!(targets.is_empty());
    }

    // ── reap_computed_ttl: scope Node ────────────────────────────────

    #[test]
    fn reap_deletes_expired_node() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let mut interner = FieldInterner::new();

        let schema = make_ttl_schema("Session", 3600, TtlScope::Node);
        persist_schema(&engine, &schema);

        let now = now_us();

        // Node 1: created 2 hours ago (expired, TTL = 1h).
        insert_node(
            &engine,
            1,
            1,
            "Session",
            now - 2 * 3600 * 1_000_000,
            &mut interner,
        );
        // Node 2: created 30 min ago (NOT expired).
        insert_node(
            &engine,
            1,
            2,
            "Session",
            now - 30 * 60 * 1_000_000,
            &mut interner,
        );

        let result = reap_computed_ttl(&engine, 1, 1000);
        assert_eq!(result.nodes_deleted, 1);
        assert_eq!(result.nodes_checked, 2);

        assert!(
            !node_exists(&engine, 1, 1),
            "expired node should be deleted"
        );
        assert!(node_exists(&engine, 1, 2), "fresh node should remain");
    }

    #[test]
    fn reap_respects_batch_size_limit() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let mut interner = FieldInterner::new();

        let schema = make_ttl_schema("Session", 3600, TtlScope::Node);
        persist_schema(&engine, &schema);

        let old_ts = now_us() - 2 * 3600 * 1_000_000;

        // Create 5 expired nodes.
        for i in 1..=5 {
            insert_node(&engine, 1, i, "Session", old_ts, &mut interner);
        }

        // Batch size = 2 → only 2 deleted.
        let result = reap_computed_ttl(&engine, 1, 2);
        assert_eq!(result.nodes_deleted, 2);

        // Run again → 2 more.
        let result2 = reap_computed_ttl(&engine, 1, 2);
        assert_eq!(result2.nodes_deleted, 2);

        // Run again → last 1.
        let result3 = reap_computed_ttl(&engine, 1, 2);
        assert_eq!(result3.nodes_deleted, 1);
    }

    // ── reap_computed_ttl: scope Field ───────────────────────────────

    #[test]
    fn reap_removes_field_on_expiry() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let mut interner = FieldInterner::new();

        let schema = make_ttl_schema("CacheEntry", 60, TtlScope::Field);
        persist_schema(&engine, &schema);

        let old_ts = now_us() - 120 * 1_000_000; // 2 min ago, TTL = 60s
        insert_node(&engine, 1, 10, "CacheEntry", old_ts, &mut interner);

        let result = reap_computed_ttl(&engine, 1, 1000);
        assert_eq!(result.fields_removed, 1);

        // Node should still exist but without the timestamp field.
        assert!(
            node_exists(&engine, 1, 10),
            "node should survive field removal"
        );

        let key = encode_node_key(1, NodeId::from_raw(10));
        let bytes = engine.get(Partition::Node, &key).unwrap().unwrap();
        let record = NodeRecord::from_msgpack(&bytes).unwrap();
        // The timestamp field should be removed.
        let has_timestamp = record
            .props
            .values()
            .any(|v| matches!(v, Value::Timestamp(_)));
        assert!(
            !has_timestamp,
            "timestamp field should be removed after TTL expiry"
        );
    }

    // ── reap_computed_ttl: scope Subtree ─────────────────────────────

    /// When `target_field` is specified, Subtree scope must delete the target
    /// DOCUMENT field, NOT the anchor TIMESTAMP field that triggered expiry.
    ///
    /// Regression test for G068: previously Subtree behaved identically to Field
    /// (always deleted anchor_field regardless of target_field).
    #[test]
    fn reap_subtree_with_target_field_deletes_target_not_anchor() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let mut interner = FieldInterner::new();

        // Schema: anchor = created_at (TIMESTAMP), target = profile_data (String).
        let mut schema = LabelSchema::new("Profile");
        schema.add_property(PropertyDef::new("created_at", PropertyType::Timestamp));
        schema.add_property(PropertyDef::new("profile_data", PropertyType::String));
        schema.add_property(PropertyDef::computed(
            "_ttl",
            ComputedSpec::Ttl {
                duration_secs: 60,
                anchor_field: "created_at".into(),
                scope: TtlScope::Subtree,
                target_field: Some("profile_data".into()),
            },
        ));
        persist_schema(&engine, &schema);

        // Insert node with expired anchor (created 2 minutes ago) + profile_data content.
        let old_ts = now_us() - 120 * 1_000_000;
        let mut record = NodeRecord::new("Profile");
        let ts_field = interner.intern("created_at");
        let pd_field = interner.intern("profile_data");
        record.set(ts_field, Value::Timestamp(old_ts));
        record.set(pd_field, Value::String("sensitive content".into()));
        let key = encode_node_key(1, NodeId::from_raw(30));
        engine
            .put(Partition::Node, &key, &record.to_msgpack().expect("ser"))
            .expect("put");

        // Use the same interner so the reaper can resolve target_field_id = pd_field.
        // Without an interner, the reaper has no way to map "profile_data" → u32 field_id
        // (props is keyed by u32, not by name).
        let result = reap_computed_ttl_with_interner(&engine, 1, 1000, &interner);
        assert_eq!(
            result.subtrees_removed, 1,
            "subtree removal should be counted"
        );
        assert!(
            node_exists(&engine, 1, 30),
            "node must survive subtree removal"
        );

        // Reload node and verify: profile_data deleted, created_at preserved.
        let raw = engine
            .get(Partition::Node, &key)
            .expect("get")
            .expect("node exists");
        let updated = NodeRecord::from_msgpack(&raw).expect("decode");
        assert!(
            !updated.props.contains_key(&pd_field),
            "profile_data must be removed by subtree TTL"
        );
        assert!(
            updated.props.contains_key(&ts_field),
            "created_at (anchor) must NOT be removed — only the target field is deleted"
        );
    }

    /// When `target_field` is specified but the field name is NOT in the interner
    /// (e.g., schema added after database open, no nodes with that field yet),
    /// the reaper must skip the deletion AND surface an error in `result.errors`.
    ///
    /// This is NOT a silent no-op — operators must be able to detect the condition.
    #[test]
    fn reap_subtree_unresolved_target_field_records_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        // Empty interner — "payload" is not interned.
        let interner = FieldInterner::new();

        // Schema with target_field = "payload", but "payload" is not in the interner.
        let mut schema = LabelSchema::new("Cache");
        schema.add_property(PropertyDef::new("cached_at", PropertyType::Timestamp));
        schema.add_property(PropertyDef::new("payload", PropertyType::String));
        schema.add_property(PropertyDef::computed(
            "_ttl",
            ComputedSpec::Ttl {
                duration_secs: 60,
                anchor_field: "cached_at".into(),
                scope: TtlScope::Subtree,
                target_field: Some("payload".into()),
            },
        ));
        persist_schema(&engine, &schema);

        // Insert expired node using a separate interner (simulates data written after startup).
        let mut write_interner = FieldInterner::new();
        let old_ts = now_us() - 120 * 1_000_000;
        let ts_field = write_interner.intern("cached_at");
        let payload_field = write_interner.intern("payload");
        let mut record = NodeRecord::new("Cache");
        record.set(ts_field, Value::Timestamp(old_ts));
        record.set(payload_field, Value::String("stale data".into()));
        let key = encode_node_key(1, NodeId::from_raw(99));
        engine
            .put(Partition::Node, &key, &record.to_msgpack().expect("ser"))
            .expect("put");

        // Reap with the EMPTY interner — target_field_id will be None.
        let result = reap_computed_ttl_with_interner(&engine, 1, 1000, &interner);

        // No deletion should happen (safe no-op), but an error must be recorded.
        assert_eq!(
            result.subtrees_removed, 0,
            "no deletion when target unresolved"
        );
        assert!(
            !result.errors.is_empty(),
            "must record error for unresolved target_field"
        );
        assert!(
            result.errors[0].contains("payload"),
            "error must mention the unresolved field name, got: {:?}",
            result.errors[0]
        );

        // The node must be untouched.
        let raw = engine
            .get(Partition::Node, &key)
            .expect("get")
            .expect("node exists");
        let updated = NodeRecord::from_msgpack(&raw).expect("decode");
        assert!(
            updated.props.contains_key(&payload_field),
            "payload must NOT be removed when target_field_id is unresolved"
        );
    }

    /// When `target_field` is specified and the first reap deletes it, the node
    /// stays alive (anchor_field preserved by design).  On the second pass,
    /// `resolve_anchor()` still finds the expired anchor — but the target field
    /// is already absent, so NO mutation should be submitted and `subtrees_removed`
    /// must NOT be incremented again.
    ///
    /// Without the `record.props.contains_key` guard this would emit a no-op
    /// merge mutation and increment the counter on every subsequent pass.
    #[test]
    fn reap_subtree_second_pass_is_idempotent() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let mut interner = FieldInterner::new();

        let mut schema = LabelSchema::new("Profile");
        schema.add_property(PropertyDef::new("created_at", PropertyType::Timestamp));
        schema.add_property(PropertyDef::new("bio", PropertyType::String));
        schema.add_property(PropertyDef::computed(
            "_ttl",
            ComputedSpec::Ttl {
                duration_secs: 60,
                anchor_field: "created_at".into(),
                scope: TtlScope::Subtree,
                target_field: Some("bio".into()),
            },
        ));
        persist_schema(&engine, &schema);

        let old_ts = now_us() - 120 * 1_000_000;
        let ts_field = interner.intern("created_at");
        let bio_field = interner.intern("bio");
        let mut record = NodeRecord::new("Profile");
        record.set(ts_field, Value::Timestamp(old_ts));
        record.set(bio_field, Value::String("hello".into()));
        let key = encode_node_key(1, NodeId::from_raw(77));
        engine
            .put(Partition::Node, &key, &record.to_msgpack().expect("ser"))
            .expect("put");

        // First reap: bio must be removed, node survives, subtrees_removed = 1.
        let r1 = reap_computed_ttl_with_interner(&engine, 1, 1000, &interner);
        assert_eq!(r1.subtrees_removed, 1, "first pass must remove bio");
        assert!(node_exists(&engine, 1, 77), "node must survive subtree TTL");

        // Verify bio is gone.
        let raw = engine
            .get(Partition::Node, &key)
            .expect("get")
            .expect("node exists");
        let after_r1 = NodeRecord::from_msgpack(&raw).expect("decode");
        assert!(
            !after_r1.props.contains_key(&bio_field),
            "bio removed after first pass"
        );
        assert!(after_r1.props.contains_key(&ts_field), "anchor preserved");

        // Second reap: bio already absent — subtrees_removed must be 0 (idempotent).
        let r2 = reap_computed_ttl_with_interner(&engine, 1, 1000, &interner);
        assert_eq!(
            r2.subtrees_removed, 0,
            "second pass must not count already-absent target as removed"
        );
        assert!(r2.errors.is_empty(), "no errors on idempotent second pass");
    }

    #[test]
    fn reap_removes_subtree_on_expiry() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let mut interner = FieldInterner::new();

        let schema = make_ttl_schema("TempDoc", 60, TtlScope::Subtree);
        persist_schema(&engine, &schema);

        let old_ts = now_us() - 120 * 1_000_000;
        insert_node(&engine, 1, 20, "TempDoc", old_ts, &mut interner);

        let result = reap_computed_ttl(&engine, 1, 1000);
        assert_eq!(result.subtrees_removed, 1);
        assert!(
            node_exists(&engine, 1, 20),
            "node should survive subtree removal"
        );
    }

    // ── edge cleanup on Node scope ───────────────────────────────────

    #[test]
    fn reap_node_scope_cleans_edges() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let mut interner = FieldInterner::new();

        let schema = make_ttl_schema("Session", 3600, TtlScope::Node);
        persist_schema(&engine, &schema);

        // Register edge type.
        let et_key = coordinode_core::schema::definition::encode_edge_type_schema_key("OWNS");
        engine
            .put(Partition::Schema, &et_key, &[])
            .expect("put edge type");

        let old_ts = now_us() - 2 * 3600 * 1_000_000;

        // Node 1 (expired) and node 100 (not TTL-managed, just a peer).
        insert_node(&engine, 1, 1, "Session", old_ts, &mut interner);

        let mut peer_record = NodeRecord::new("User");
        let name_field = interner.intern("name");
        peer_record.set(name_field, Value::String("alice".into()));
        let peer_key = encode_node_key(1, NodeId::from_raw(100));
        let peer_bytes = peer_record.to_msgpack().unwrap();
        engine.put(Partition::Node, &peer_key, &peer_bytes).unwrap();

        // Create edge: node 1 -[OWNS]-> node 100
        let fwd_key = encode_adj_key_forward("OWNS", NodeId::from_raw(1));
        let rev_key = encode_adj_key_reverse("OWNS", NodeId::from_raw(100));
        let fwd_plist = PostingList::from_sorted(vec![100]);
        let rev_plist = PostingList::from_sorted(vec![1]);
        engine
            .put(Partition::Adj, &fwd_key, &fwd_plist.to_bytes().unwrap())
            .unwrap();
        engine
            .put(Partition::Adj, &rev_key, &rev_plist.to_bytes().unwrap())
            .unwrap();

        let result = reap_computed_ttl(&engine, 1, 1000);
        assert_eq!(result.nodes_deleted, 1);

        // Forward adj key for deleted node should be gone.
        assert!(engine.get(Partition::Adj, &fwd_key).unwrap().is_none());

        // Peer node should still exist.
        assert!(node_exists(&engine, 1, 100));
    }

    // ── no TTL schemas → no-op ───────────────────────────────────────

    #[test]
    fn reap_no_schemas_is_noop() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());

        let result = reap_computed_ttl(&engine, 1, 1000);
        assert_eq!(result.labels_scanned, 0);
        assert_eq!(result.nodes_checked, 0);
        assert_eq!(result.total_deletions(), 0);
    }

    // ── regression: multi-Timestamp field anchor resolution ────────

    /// BUG: find_anchor_timestamp without interner picks FIRST Timestamp,
    /// which may be `updated_at` (fresh) instead of `created_at` (expired).
    /// Uses find_anchor_timestamp_with_interner to verify correct resolution.
    #[test]
    fn find_anchor_without_interner_may_pick_wrong_field() {
        let mut interner = FieldInterner::new();
        let mut record = NodeRecord::new("Session");

        // Intern 5 fields to make HashMap order unpredictable.
        let _f1 = interner.intern("field_a");
        let _f2 = interner.intern("field_b");
        let created_field = interner.intern("created_at");
        let _f3 = interner.intern("field_c");
        let updated_field = interner.intern("updated_at");

        let old_ts = 1000i64; // "expired" timestamp
        let new_ts = 9_999_999_999i64; // "fresh" timestamp

        record.set(created_field, Value::Timestamp(old_ts));
        record.set(updated_field, Value::Timestamp(new_ts));

        // Interner-aware: always finds the correct field.
        let correct = find_anchor_timestamp_with_interner(&record, "created_at", &interner);
        assert_eq!(correct, Some(old_ts), "interner-aware must find created_at");

        // Without interner: finds SOME Timestamp — may or may not be correct.
        let heuristic = find_anchor_timestamp(&record, "created_at");
        assert!(heuristic.is_some(), "should find at least one timestamp");
        // The heuristic might return old_ts or new_ts depending on HashMap order.
        // This is the bug — it's nondeterministic.

        // Verify interner-aware is always correct for both fields.
        let updated = find_anchor_timestamp_with_interner(&record, "updated_at", &interner);
        assert_eq!(updated, Some(new_ts));
    }

    /// Regression test: reaper with interner resolves correct anchor for
    /// multi-Timestamp nodes. Node with created_at=expired + updated_at=fresh
    /// MUST be deleted (anchor is created_at).
    #[test]
    fn reap_multi_timestamp_uses_interner_for_correct_anchor() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let mut interner = FieldInterner::new();

        let schema = make_ttl_schema("Session", 3600, TtlScope::Node);
        persist_schema(&engine, &schema);

        let now = now_us();
        let two_hours_ago = now - 2 * 3600 * 1_000_000;

        // Node with TWO Timestamp fields.
        let mut record = NodeRecord::new("Session");
        let created_field = interner.intern("created_at");
        let updated_field = interner.intern("updated_at");
        record.set(created_field, Value::Timestamp(two_hours_ago));
        record.set(updated_field, Value::Timestamp(now));

        let key = encode_node_key(1, NodeId::from_raw(42));
        let bytes = record.to_msgpack().expect("serialize");
        engine.put(Partition::Node, &key, &bytes).expect("put");

        // Use interner-aware reaper function directly.
        let result = reap_computed_ttl_with_interner(&engine, 1, 1000, &interner);

        assert_eq!(
            result.nodes_deleted, 1,
            "node with expired created_at should be deleted even when updated_at is fresh"
        );
        assert!(
            !node_exists(&engine, 1, 42),
            "expired node should not exist after reap"
        );
    }

    // ── interner-aware anchor lookup ─────────────────────────────────

    #[test]
    fn find_anchor_with_interner_resolves_correct_field() {
        let mut interner = FieldInterner::new();
        let mut record = NodeRecord::new("Session");

        let created_field = interner.intern("created_at");
        let updated_field = interner.intern("updated_at");

        record.set(created_field, Value::Timestamp(1000));
        record.set(updated_field, Value::Timestamp(2000));

        // Should find created_at (1000), not updated_at (2000).
        let ts = find_anchor_timestamp_with_interner(&record, "created_at", &interner);
        assert_eq!(ts, Some(1000));

        let ts2 = find_anchor_timestamp_with_interner(&record, "updated_at", &interner);
        assert_eq!(ts2, Some(2000));

        // Non-existent field → None.
        let ts3 = find_anchor_timestamp_with_interner(&record, "missing", &interner);
        assert_eq!(ts3, None);
    }
}
