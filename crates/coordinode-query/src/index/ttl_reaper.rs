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

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use coordinode_core::graph::node::{NodeId, NodeRecord, decode_node_key};
use coordinode_core::graph::types::Value;
use coordinode_core::schema::computed::{ComputedSpec, TtlScope};
#[cfg(test)]
use coordinode_core::schema::definition::LabelSchema;
use coordinode_core::schema::definition::PropertyType;
use coordinode_core::txn::proposal::{ProposalIdGenerator, ProposalPipeline};
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_core::txn::wake::Wake;
use coordinode_modality::{
    EdgeStore as _, LocalEdgeStore, LocalNodeStore, LocalSchemaStore, LocalTableKeyStore,
    NodeStore as _, SchemaStore as _, TableKeyStore as _,
};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::transaction::{CommitContext, CommitError, Transaction};
use rustc_hash::FxHashMap;

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
    /// Node records read from the shard, by their labels alone.
    pub records_scanned: usize,
    /// Node records decoded in full: those of a label with a TTL.
    pub records_decoded: usize,
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
    /// How a deleted row frees its table key.
    key: RowKeyRelease,
    /// The B-tree indexes a reaped row's entries leave with it.
    indexes: Arc<BtreeIndexes>,
}

/// The B-tree indexes in force for a reap pass, and how to read the values
/// they index from a stored record.
struct BtreeIndexes {
    registry: super::IndexRegistry,
    /// Field ids of the indexed property names the dictionary knows; a name
    /// it does not know can only be stored by name, in the record's overflow.
    fields: std::collections::HashMap<String, u32>,
    /// Whether values can be read at all: a pass without the dictionary
    /// cannot tell what a record holds under an id.
    readable: bool,
}

impl BtreeIndexes {
    fn load(
        engine: &StorageEngine,
        interner: Option<&coordinode_core::graph::intern::FieldInterner>,
    ) -> Result<Self, String> {
        let registry = super::IndexRegistry::new();
        registry.load_all(engine).map_err(|e| e.to_string())?;
        let mut fields = std::collections::HashMap::new();
        if let Some(interner) = interner {
            for index in registry.all() {
                let filter = index.filter.as_ref().map(|f| f.property().to_string());
                for name in index.properties.iter().cloned().chain(filter) {
                    if let Some(id) = interner.lookup(&name) {
                        fields.insert(name, id);
                    }
                }
            }
        }
        Ok(Self {
            readable: interner.is_some() || registry.is_empty(),
            registry,
            fields,
        })
    }

    /// A ready B-tree index holding exactly the `anchor` property of every
    /// node of `label`, which can supply the TTL candidates of the label in
    /// anchor order. An index still building is not evidence that a node it
    /// lacks has not expired, and a partial index leaves out nodes its filter
    /// rejects, so neither qualifies; a sparse one only leaves out nodes
    /// without the anchor, which never expire.
    fn anchor_index(&self, label: &str, anchor: &str) -> Option<super::IndexDefinition> {
        self.registry
            .indexes_for_property(label, anchor)
            .into_iter()
            .find(|def| {
                def.index_type == super::IndexType::BTree
                    && def.properties.len() == 1
                    && def.filter.is_none()
                    && !def.multikey
                    && def.state == super::IndexState::Ready
            })
    }

    /// Whether a reaped node's entries can be named: a label with B-tree
    /// indexes needs the values they index, which a pass without the field
    /// dictionary cannot read.
    fn can_remove(&self, record: &NodeRecord) -> bool {
        self.readable || !self.registry.has_btree_for(record.primary_label())
    }

    /// Stage the removal of a reaped node's entries in `txn`, the same way a
    /// statement deleting the node stages it. Call only when
    /// [`Self::can_remove`] holds.
    fn stage_delete(
        &self,
        engine: &StorageEngine,
        txn: &mut Transaction<'_>,
        node_id: NodeId,
        record: &NodeRecord,
    ) -> Result<(), coordinode_modality::StoreError> {
        let label = record.primary_label();
        if !self.registry.has_btree_for(label) {
            return Ok(());
        }
        let lookup = |name: &str| {
            self.fields
                .get(name)
                .and_then(|id| record.props.get(id).cloned())
                .or_else(|| record.get_extra(name).cloned())
        };
        let field_of = |name: &str| self.fields.get(name).copied();
        // The reaper reads only node keys that are not temporal versions.
        let node = super::registry::NodeState {
            node_id,
            valid_from: None,
            label,
            value_of: &lookup,
        };
        self.registry.on_node_deleted(engine, txn, &node, &field_of)
    }
}

/// What deleting a row of a TTL target does to its table's key index.
enum RowKeyRelease {
    /// Not a table with declared key columns: nothing to free.
    None,
    /// The key columns, as field ids and names in key order.
    Columns(Vec<(u32, String)>),
    /// A keyed table whose key columns the dictionary could not resolve: its
    /// rows are not deleted, because a deleted row would leave its key held.
    Unresolved,
}

/// Run a single COMPUTED TTL reap pass on a direct-mode engine (no oracle,
/// writes applied as they are staged). For tests and tools.
pub fn reap_computed_ttl(
    engine: &StorageEngine,
    shard_id: u16,
    batch_size: usize,
) -> ComputedTtlReapResult {
    reap_computed_ttl_inner(engine, shard_id, batch_size, None, None, &mut direct_commit)
}

/// Run a single COMPUTED TTL reap pass on a direct-mode engine, resolving
/// field names through `interner`.
pub fn reap_computed_ttl_with_interner(
    engine: &StorageEngine,
    shard_id: u16,
    batch_size: usize,
    interner: &coordinode_core::graph::intern::FieldInterner,
) -> ComputedTtlReapResult {
    reap_computed_ttl_inner(
        engine,
        shard_id,
        batch_size,
        Some(interner),
        None,
        &mut direct_commit,
    )
}

/// Run a single COMPUTED TTL reap pass as transactions of `oracle`,
/// committing each through `commit` (the database's commit path, so the
/// pass replicates like any write and passes the commit guard).
///
/// A cleanup is a write that depends on a condition, the expiry, evaluated
/// against one version of the record. Each page of the pass is one
/// transaction that states that version for every record it changes and,
/// for every node it deletes, that the node's identity is being destroyed: a
/// renewal committed after the page read, or an edge attached meanwhile,
/// refuses the page, which is then read again.
pub fn reap_computed_ttl_committed<'a>(
    engine: &'a StorageEngine,
    shard_id: u16,
    batch_size: usize,
    interner: &coordinode_core::graph::intern::FieldInterner,
    oracle: &'a TimestampOracle,
    commit: &mut dyn FnMut(&mut Transaction<'a>) -> Result<(), CommitError>,
) -> ComputedTtlReapResult {
    reap_computed_ttl_inner(
        engine,
        shard_id,
        batch_size,
        Some(interner),
        Some(oracle),
        commit,
    )
}

/// Run one COMPUTED TTL reap pass the way the background reaper does: each
/// page is a transaction of `oracle` committed through `pipeline`, so the
/// deletions replicate and reach every follower of the applied commits.
///
/// # Errors
///
/// The field dictionary could not be read; nothing was reaped.
#[allow(clippy::too_many_arguments)]
pub fn reap_pass(
    engine: &StorageEngine,
    shard_id: u16,
    batch_size: usize,
    fields: &dyn coordinode_core::graph::intern::FieldRegistrar,
    oracle: &TimestampOracle,
    pipeline: &dyn ProposalPipeline,
    id_gen: &ProposalIdGenerator,
) -> Result<ComputedTtlReapResult, coordinode_core::graph::intern::DictionaryError> {
    // Each pass reads the dictionary as it stands, so a TTL property
    // registered since the last pass is reaped too.
    let interner = fields.view()?;
    // Majority, as every replicated write is by default: a cleanup a
    // failover could forget would be redone, but one acknowledged and then
    // lost would hand a renewed record back to a reaper that already decided.
    let write_concern = coordinode_core::txn::write_concern::WriteConcern::default();
    let commit_ctx = CommitContext {
        write_concern: &write_concern,
        pipeline: Some(pipeline),
        id_gen: Some(id_gen),
        drain_buffer: None,
        nvme_write_buffer: None,
    };
    Ok(reap_computed_ttl_committed(
        engine,
        shard_id,
        batch_size,
        &interner,
        oracle,
        &mut |txn| txn.commit(&commit_ctx).map(|_| ()),
    ))
}

/// The commit of a direct-mode transaction: its writes were applied as they
/// were staged, and this flushes what it buffered.
fn direct_commit(txn: &mut Transaction<'_>) -> Result<(), CommitError> {
    let write_concern = coordinode_core::txn::write_concern::WriteConcern::default();
    let ctx = CommitContext {
        write_concern: &write_concern,
        pipeline: None,
        id_gen: None,
        drain_buffer: None,
        nvme_write_buffer: None,
    };
    txn.commit(&ctx).map(|_| ())
}

/// Shared implementation.
fn reap_computed_ttl_inner<'a>(
    engine: &'a StorageEngine,
    shard_id: u16,
    batch_size: usize,
    interner: Option<&coordinode_core::graph::intern::FieldInterner>,
    oracle: Option<&'a TimestampOracle>,
    commit: &mut dyn FnMut(&mut Transaction<'a>) -> Result<(), CommitError>,
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

    result.labels_scanned = targets.len();
    reap_shard(
        engine,
        shard_id,
        &targets,
        now_us,
        batch_size,
        oracle,
        commit,
        &mut result,
    );
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
    let schemas: Vec<_> = LocalSchemaStore::new(engine)
        .list_labels()
        .map_err(|e| e.to_string())?
        .into_iter()
        .filter(|schema| {
            schema.properties.values().any(|def| {
                matches!(
                    def.property_type,
                    PropertyType::Computed(ComputedSpec::Ttl { .. })
                )
            })
        })
        .collect();
    // The index catalog is read only when there is TTL work to do.
    if schemas.is_empty() {
        return Ok(Vec::new());
    }
    let indexes = Arc::new(BtreeIndexes::load(engine, interner)?);

    let mut targets = Vec::new();
    for schema in schemas {
        let label_name = schema.name.clone();
        let key = match schema.key_columns() {
            [] => RowKeyRelease::None,
            columns => columns
                .iter()
                .map(|c| {
                    interner
                        .and_then(|int| int.lookup(c))
                        .map(|id| (id, c.clone()))
                })
                .collect::<Option<Vec<_>>>()
                .map_or(RowKeyRelease::Unresolved, RowKeyRelease::Columns),
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
                    key: match &key {
                        RowKeyRelease::None => RowKeyRelease::None,
                        RowKeyRelease::Columns(c) => RowKeyRelease::Columns(c.clone()),
                        RowKeyRelease::Unresolved => RowKeyRelease::Unresolved,
                    },
                    indexes: Arc::clone(&indexes),
                });
            }
        }
    }

    Ok(targets)
}

/// List all registered edge type names via the schema store. Versioned keys
/// (`schema:edge_type:<name>:<version>`) dedup to one entry per name.
fn list_edge_types(engine: &StorageEngine) -> Result<Vec<String>, String> {
    LocalSchemaStore::new(engine)
        .list_edge_type_names_engine()
        .map_err(|e| e.to_string())
}

/// Reap every target in one read of the shard. Collects mutations and
/// submits them via pipeline (if provided) or applies them directly.
///
/// A record is first read by its labels alone; only one of a label with a
/// target is decoded in full, so the cost of the records of other labels
/// does not grow with their size or with the number of targets.
#[allow(clippy::too_many_arguments)]
fn reap_shard<'a>(
    engine: &'a StorageEngine,
    shard_id: u16,
    targets: &[TtlTarget],
    now_us: i64,
    max_deletions: usize,
    oracle: Option<&'a TimestampOracle>,
    commit: &mut dyn FnMut(&mut Transaction<'a>) -> Result<(), CommitError>,
    result: &mut ComputedTtlReapResult,
) {
    // The targets of each label, with the instant before which an anchor has
    // expired. A Subtree target whose target field the dictionary does not
    // know is left out with one diagnostic: it cannot become known during the
    // pass, and every expired node would otherwise repeat the same error.
    let mut by_label: FxHashMap<&str, Vec<(&TtlTarget, i64)>> = FxHashMap::default();
    for target in targets {
        if let (TtlScope::Subtree, Some(tf), None) = (
            target.scope,
            target.target_field.as_deref(),
            target.target_field_id,
        ) {
            result.errors.push(format!(
                "label {}: target_field '{tf}' not in interner — Subtree deletion skipped",
                target.label,
            ));
            continue;
        }
        let cutoff_us = now_us - (target.duration_secs as i64 * 1_000_000);
        // A target with a ready index on its anchor reads its candidates
        // there and leaves the shard alone.
        if let Some(index) = target
            .indexes
            .anchor_index(&target.label, &target.anchor_field)
        {
            reap_through_index(
                engine,
                shard_id,
                target,
                &index,
                cutoff_us,
                max_deletions,
                oracle,
                commit,
                result,
            );
            continue;
        }
        by_label
            .entry(target.label.as_str())
            .or_default()
            .push((target, cutoff_us));
    }
    if by_label.is_empty() {
        return;
    }

    // The shard is read a page at a time, each page in a transaction of its
    // own. A page that meets a concurrent change (a renewal of a record it
    // reaps, an edge attached to a node it deletes) is refused at commit and
    // read again on a fresh view, where the renewed record is no longer
    // expired.
    let nodes = LocalNodeStore;
    let prefix = nodes.shard_scan_prefix(shard_id);
    let mut start_after: Option<Vec<u8>> = None;
    let mut retries = 0u32;
    'pages: while result.total_deletions() < max_deletions {
        let mut txn = match oracle {
            Some(oracle) => Transaction::begin(engine, Some(oracle), oracle.next()),
            None => Transaction::new(engine, None, Timestamp::ZERO, None),
        };
        let page = match nodes.prefix_scan_paged_tracked(
            &mut txn,
            &prefix,
            start_after.as_deref(),
            PAGE,
        ) {
            Ok(page) => page,
            Err(e) => {
                result.errors.push(format!("node scan error: {e}"));
                break;
            }
        };

        // Listed per page, after the page's view was taken: an edge of a type
        // created since the pass began is one this page must find. Read at or
        // after the view, the list holds every type the view can hold.
        let edge_types = match list_edge_types(engine) {
            Ok(types) => types,
            Err(e) => {
                result.errors.push(format!("edge type scan error: {e}"));
                break;
            }
        };
        let budget = max_deletions - result.total_deletions();
        let mut page_result = ComputedTtlReapResult::default();
        let mut changed: Vec<Vec<u8>> = Vec::new();
        for (key, bytes) in &page.rows {
            if page_result.total_deletions() >= budget {
                break;
            }
            // Temporal versions are keyed apart, and a record that does not
            // decode is skipped rather than taking the pass down.
            let Some((_, node_id)) = decode_node_key(key) else {
                continue;
            };
            page_result.records_scanned += 1;
            let Ok(labels) = NodeRecord::labels_from_msgpack(bytes) else {
                continue;
            };
            if !labels.iter().any(|l| by_label.contains_key(l.as_str())) {
                continue;
            }
            let Ok(record) = NodeRecord::from_msgpack(bytes) else {
                continue;
            };
            page_result.records_decoded += 1;
            let mut staged = false;
            'targets: for label in &labels {
                for (target, cutoff_us) in by_label.get(label.as_str()).into_iter().flatten() {
                    let deleted_before = page_result.nodes_deleted;
                    match stage_expired(
                        engine,
                        &mut txn,
                        shard_id,
                        target,
                        node_id,
                        key,
                        &record,
                        *cutoff_us,
                        &edge_types,
                        &mut page_result,
                    ) {
                        Ok(true) => staged = true,
                        Ok(false) => {}
                        Err(e) => {
                            // The staged work of the page is no longer one
                            // decision; nothing of it is committed.
                            result.errors.push(format!(
                                "label {}: node {}: {e}",
                                target.label,
                                node_id.as_raw()
                            ));
                            break 'pages;
                        }
                    }
                    // A deleted node has nothing left for another target.
                    if page_result.nodes_deleted > deleted_before {
                        break 'targets;
                    }
                }
            }
            if staged {
                changed.push(key.clone());
            }
        }

        match commit_page(&mut txn, &changed, commit, &mut result.errors) {
            PageCommit::Done => {}
            PageCommit::Retry => {
                retries += 1;
                if retries <= MAX_PAGE_RETRIES {
                    continue;
                }
                page_result = left_for_next_pass(page_result);
            }
            PageCommit::Abort => break,
        }
        retries = 0;
        absorb(result, page_result);
        if page.exhausted {
            break;
        }
        start_after = page.last_key;
    }
}

/// Reap `target` from the candidates its ready anchor `index` holds below
/// `cutoff_us`, in anchor order, a page at a time. The index only proposes:
/// each candidate is read and decided on its record as the page's view holds
/// it, and the page commits conditioned on those versions, as a shard page
/// does. Anchors are kept as timestamps or as integers, so both ranges are
/// read.
#[allow(clippy::too_many_arguments)]
fn reap_through_index<'a>(
    engine: &'a StorageEngine,
    shard_id: u16,
    target: &TtlTarget,
    index: &super::IndexDefinition,
    cutoff_us: i64,
    max_deletions: usize,
    oracle: Option<&'a TimestampOracle>,
    commit: &mut dyn FnMut(&mut Transaction<'a>) -> Result<(), CommitError>,
    result: &mut ComputedTtlReapResult,
) {
    use coordinode_modality::{IndexStore as _, LocalIndexStore};

    let store = LocalIndexStore::new(engine);
    let nodes = LocalNodeStore;
    let ranges = [
        (Value::Int(i64::MIN), Value::Int(cutoff_us)),
        (Value::Timestamp(i64::MIN), Value::Timestamp(cutoff_us)),
    ];
    for (from, to) in &ranges {
        let mut after: Option<Vec<u8>> = None;
        let mut retries = 0u32;
        while result.total_deletions() < max_deletions {
            let mut txn = match oracle {
                Some(oracle) => Transaction::begin(engine, Some(oracle), oracle.next()),
                None => Transaction::new(engine, None, Timestamp::ZERO, None),
            };
            let page =
                match store.scan_entries_in(&mut txn, index, from, to, after.as_deref(), PAGE) {
                    Ok(Some(page)) => page,
                    Ok(None) => break,
                    Err(e) => {
                        result
                            .errors
                            .push(format!("label {}: index scan: {e}", target.label));
                        return;
                    }
                };
            // The reaper reaps nodes, not single versions of temporal ones.
            let ids: Vec<NodeId> = page
                .entries
                .iter()
                .filter(|(_, valid_from)| valid_from.is_none())
                .map(|(id, _)| *id)
                .collect();
            let edge_types = match list_edge_types(engine) {
                Ok(types) => types,
                Err(e) => {
                    result.errors.push(format!("edge type scan error: {e}"));
                    return;
                }
            };
            let records = match nodes.get_many(&txn, shard_id, &ids) {
                Ok(records) => records,
                Err(e) => {
                    result
                        .errors
                        .push(format!("label {}: candidate read: {e}", target.label));
                    return;
                }
            };
            let budget = max_deletions - result.total_deletions();
            let mut page_result = ComputedTtlReapResult::default();
            let mut changed: Vec<Vec<u8>> = Vec::new();
            for (node_id, record) in ids.iter().zip(records) {
                if page_result.total_deletions() >= budget {
                    break;
                }
                // Gone since the entry was read: nothing to reap.
                let Some(record) = record else {
                    continue;
                };
                page_result.records_decoded += 1;
                let key = coordinode_core::graph::node::encode_node_key(shard_id, *node_id);
                match stage_expired(
                    engine,
                    &mut txn,
                    shard_id,
                    target,
                    *node_id,
                    &key,
                    &record,
                    cutoff_us,
                    &edge_types,
                    &mut page_result,
                ) {
                    Ok(true) => changed.push(key),
                    Ok(false) => {}
                    Err(e) => {
                        result.errors.push(format!(
                            "label {}: node {}: {e}",
                            target.label,
                            node_id.as_raw()
                        ));
                        return;
                    }
                }
            }
            match commit_page(&mut txn, &changed, commit, &mut result.errors) {
                PageCommit::Done => {}
                PageCommit::Retry => {
                    retries += 1;
                    if retries <= MAX_PAGE_RETRIES {
                        continue;
                    }
                    page_result = left_for_next_pass(page_result);
                }
                PageCommit::Abort => return,
            }
            retries = 0;
            absorb(result, page_result);
            if page.exhausted {
                break;
            }
            after = page.resume;
        }
    }
}

/// What became of a page's staged work at its commit.
enum PageCommit {
    /// Committed, or nothing to commit.
    Done,
    /// Refused for a concurrent change: read the page again.
    Retry,
    /// Failed for another reason, recorded; the pass stops.
    Abort,
}

/// Commit a page's staged work, conditioned on the version of every record
/// it changes: the expiry was decided against that version, and a renewal
/// is a later one.
fn commit_page<'a>(
    txn: &mut Transaction<'a>,
    changed: &[Vec<u8>],
    commit: &mut dyn FnMut(&mut Transaction<'a>) -> Result<(), CommitError>,
    errors: &mut Vec<String>,
) -> PageCommit {
    if changed.is_empty() {
        return PageCommit::Done;
    }
    match LocalNodeStore.condition_unchanged(txn, changed) {
        Ok(true) => match commit(txn) {
            Ok(()) => PageCommit::Done,
            Err(
                CommitError::Conflict(_)
                | CommitError::RevisionMismatch { .. }
                | CommitError::InvariantRefused { .. },
            ) => PageCommit::Retry,
            Err(e) => {
                errors.push(format!("commit: {e}"));
                PageCommit::Abort
            }
        },
        Ok(false) => PageCommit::Retry,
        Err(e) => {
            errors.push(format!("condition: {e}"));
            PageCommit::Abort
        }
    }
}

/// A page refused more times than it is retried: what it read is counted,
/// what it would have changed is not, and the next pass reads it again.
fn left_for_next_pass(page: ComputedTtlReapResult) -> ComputedTtlReapResult {
    let mut errors = page.errors;
    errors.push("a page kept changing under the reaper; left for the next pass".to_string());
    ComputedTtlReapResult {
        nodes_checked: page.nodes_checked,
        records_scanned: page.records_scanned,
        records_decoded: page.records_decoded,
        errors,
        ..ComputedTtlReapResult::default()
    }
}

/// Add a page's outcome to the pass's.
fn absorb(result: &mut ComputedTtlReapResult, page: ComputedTtlReapResult) {
    result.nodes_checked += page.nodes_checked;
    result.records_scanned += page.records_scanned;
    result.records_decoded += page.records_decoded;
    result.nodes_deleted += page.nodes_deleted;
    result.fields_removed += page.fields_removed;
    result.subtrees_removed += page.subtrees_removed;
    result.errors.extend(page.errors);
}

/// Records read per page of a reap pass, and so the most one page's
/// transaction changes.
const PAGE: usize = 512;

/// Times a page refused for a concurrent change is read again before it is
/// left for the next pass.
const MAX_PAGE_RETRIES: u32 = 3;

/// Stage the cleanup of one record if it is expired, in `txn`. Returns
/// whether anything was staged; a record kept for a reason the operator
/// should see adds that reason to `result.errors`.
#[allow(clippy::too_many_arguments)]
fn stage_expired(
    engine: &StorageEngine,
    txn: &mut Transaction<'_>,
    shard_id: u16,
    target: &TtlTarget,
    node_id: NodeId,
    key: &[u8],
    record: &NodeRecord,
    cutoff_us: i64,
    edge_types: &[String],
    result: &mut ComputedTtlReapResult,
) -> Result<bool, String> {
    if !record.labels.contains(&target.label) {
        return Ok(false);
    }
    result.nodes_checked += 1;
    let Some(anchor_us) = resolve_anchor(record, &target.anchor_field, target.anchor_field_id)
    else {
        return Ok(false);
    };
    if anchor_us >= cutoff_us {
        return Ok(false);
    }

    let nid_raw = node_id.as_raw();
    match target.scope {
        TtlScope::Node => {
            // Everything that can keep the row is decided before anything is
            // staged, so a kept row leaves nothing of itself in the page.
            let key_values = match &target.key {
                RowKeyRelease::None => None,
                RowKeyRelease::Unresolved => {
                    result.errors.push(format!(
                        "label {}: key columns not in the field dictionary; \
                         row {nid_raw} kept so its key is not left held",
                        target.label
                    ));
                    return Ok(false);
                }
                RowKeyRelease::Columns(columns) => {
                    let values: Option<Vec<Value>> = columns
                        .iter()
                        .map(|(id, _)| record.props.get(id).cloned())
                        .collect();
                    let Some(values) = values else {
                        result.errors.push(format!(
                            "label {}: row {nid_raw} has no usable key; kept",
                            target.label
                        ));
                        return Ok(false);
                    };
                    Some(values)
                }
            };
            if !target.indexes.can_remove(record) {
                result.errors.push(format!(
                    "label {}: indexed values unreadable without the field \
                     dictionary; row {nid_raw} kept so its index entries are not left",
                    target.label
                ));
                return Ok(false);
            }

            if let Some(values) = key_values {
                LocalTableKeyStore
                    .release(txn, &target.label, &values)
                    .map_err(|e| e.to_string())?;
            }
            target
                .indexes
                .stage_delete(engine, txn, node_id, record)
                .map_err(|e| e.to_string())?;
            stage_detach(txn, node_id, edge_types)?;
            // Deleting through the node store states that the identity is
            // being destroyed, which excludes an edge attached to it before
            // this commit lands.
            LocalNodeStore
                .delete(txn, shard_id, node_id)
                .map_err(|e| e.to_string())?;
            use coordinode_modality::stats as stat_keys;
            txn.push_counter_delta(stat_keys::NODES_TOTAL_KEY, -1);
            for label in &record.labels {
                txn.push_counter_delta(&stat_keys::label_count_key(label), -1);
            }
            result.nodes_deleted += 1;
            Ok(true)
        }
        TtlScope::Field => {
            if property_is_indexed(target, record, &target.anchor_field, result) {
                return Ok(false);
            }
            let operand = removal_operand(record, target.anchor_field_id, &target.anchor_field)?;
            txn.push_node_delta(key.to_vec(), operand);
            result.fields_removed += 1;
            Ok(true)
        }
        TtlScope::Subtree => {
            // Subtree: delete `target_field` if specified and resolved,
            // otherwise fall back to the anchor field (same as Field scope).
            // An unresolved `target_field` returned early at reap_label entry.
            let (del_field_id, del_field_name) =
                match (&target.target_field, target.target_field_id) {
                    (Some(tf), Some(field_id)) => {
                        // The anchor is preserved by design, so the node stays
                        // visible to the reaper on every pass; a target field that
                        // is already gone has nothing left to remove.
                        if !record.props.contains_key(&field_id) {
                            return Ok(false);
                        }
                        (target.target_field_id, tf.as_str())
                    }
                    (Some(_), None) => return Ok(false),
                    (None, _) => (target.anchor_field_id, target.anchor_field.as_str()),
                };
            if property_is_indexed(target, record, del_field_name, result) {
                return Ok(false);
            }
            let operand = removal_operand(record, del_field_id, del_field_name)?;
            txn.push_node_delta(key.to_vec(), operand);
            result.subtrees_removed += 1;
            Ok(true)
        }
    }
}

/// Stage the removal of every edge of `node`, as the node's own lists and
/// every peer's entry for it, together with every instance of each pair
/// (the pair's own key and every version or discriminator beneath it).
fn stage_detach(
    txn: &mut Transaction<'_>,
    node: NodeId,
    edge_types: &[String],
) -> Result<(), String> {
    let edges = LocalEdgeStore;
    for edge_type in edge_types {
        for forward in [true, false] {
            let posting = if forward {
                edges.posting_fwd(txn, edge_type, node)
            } else {
                edges.posting_rev(txn, edge_type, node)
            }
            .map_err(|e| e.to_string())?;
            let Some(posting) = posting else {
                continue;
            };
            for peer_uid in posting.iter() {
                let peer = NodeId::from_raw(peer_uid);
                let (source, target) = if forward {
                    edges.merge_remove_rev(txn, edge_type, peer, node.as_raw());
                    (node, peer)
                } else {
                    edges.merge_remove_fwd(txn, edge_type, peer, node.as_raw());
                    (peer, node)
                };
                edges
                    .delete_pair_instances(txn, edge_type, source, target)
                    .map_err(|e| e.to_string())?;
            }
            edges
                .purge_adj(txn, edge_type, node, forward)
                .map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

/// Whether expiring `property` of `record` would move an index entry: a
/// B-tree entry, or the row's table key. Such a property is kept, with a
/// diagnostic: removing it by mutation would leave the entry of the old value
/// behind, and a key column cannot change at all.
fn property_is_indexed(
    target: &TtlTarget,
    record: &NodeRecord,
    property: &str,
    result: &mut ComputedTtlReapResult,
) -> bool {
    let key_column = match &target.key {
        RowKeyRelease::None => false,
        RowKeyRelease::Columns(columns) => columns.iter().any(|(_, name)| name == property),
        // The key columns are unknown, so this may be one of them.
        RowKeyRelease::Unresolved => true,
    };
    let indexed = key_column
        || target
            .indexes
            .registry
            .reads_property(record.primary_label(), property);
    if indexed {
        result.errors.push(format!(
            "label {}: TTL would remove indexed property '{property}'; kept",
            target.label
        ));
    }
    indexed
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

/// The node-delta operand removing a property from `record`.
///
/// Without a field id (no dictionary for the pass) the property is found by
/// the same heuristic the anchor lookup uses: the first timestamp or integer
/// field, else the named entry of the record's overflow.
fn removal_operand(
    record: &NodeRecord,
    field_id: Option<u32>,
    field_name: &str,
) -> Result<Vec<u8>, String> {
    use coordinode_core::graph::doc_delta::{DocDelta, PathTarget};

    let field_id = field_id.or_else(|| {
        record
            .props
            .iter()
            .find(|(_, v)| matches!(v, Value::Timestamp(_) | Value::Int(_)))
            .map(|(k, _)| *k)
    });
    let delta = match field_id {
        Some(id) => DocDelta::RemoveProperty {
            target: PathTarget::PropField(id),
            key: None,
        },
        None => DocDelta::RemoveProperty {
            target: PathTarget::Extra,
            key: Some(field_name.to_string()),
        },
    };
    delta.encode().map_err(|e| e.to_string())
}

/// Background reaper handle. Spawns a thread that periodically runs
/// `reap_computed_ttl`. Stopped on drop (graceful shutdown).
pub struct TtlReaperHandle {
    shutdown: Arc<AtomicBool>,
    /// The thread sleeps on this between passes; a stop interrupts it.
    wake: Arc<Wake>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl TtlReaperHandle {
    /// Start the background COMPUTED TTL reaper thread.
    ///
    /// The thread runs until `shutdown()` is called or the handle is dropped.
    /// `fields`: the database's field dictionary, read afresh each pass.
    /// `oracle`: the database's timestamp oracle; each page of a pass is one
    /// of its transactions.
    /// `pipeline`: proposal pipeline for cluster-replicated writes.
    /// `id_gen`: proposal ID generator (shared with other pipeline users).
    pub fn start(
        engine: Arc<StorageEngine>,
        shard_id: u16,
        config: TtlReaperConfig,
        fields: Arc<dyn coordinode_core::graph::intern::FieldRegistrar>,
        oracle: Arc<TimestampOracle>,
        pipeline: Arc<dyn ProposalPipeline>,
        id_gen: Arc<ProposalIdGenerator>,
    ) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = Arc::clone(&shutdown);
        let wake = Arc::new(Wake::default());
        let wake_clone = Arc::clone(&wake);

        let thread = match std::thread::Builder::new()
            .name("coordinode-ttl-reaper".into())
            .spawn(move || {
                reaper_loop(
                    &engine,
                    shard_id,
                    &config,
                    &shutdown_clone,
                    &wake_clone,
                    fields.as_ref(),
                    &oracle,
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

        Self {
            shutdown,
            wake,
            thread,
        }
    }

    /// Signal the reaper thread to stop.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
        self.wake.interrupt();
    }
}

impl Drop for TtlReaperHandle {
    fn drop(&mut self) {
        self.shutdown();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Main reaper loop: sleep → scan → delete → repeat.
#[allow(clippy::too_many_arguments)]
fn reaper_loop(
    engine: &StorageEngine,
    shard_id: u16,
    config: &TtlReaperConfig,
    shutdown: &AtomicBool,
    wake: &Wake,
    fields: &dyn coordinode_core::graph::intern::FieldRegistrar,
    oracle: &TimestampOracle,
    pipeline: &dyn ProposalPipeline,
    id_gen: &ProposalIdGenerator,
) {
    let interval = Duration::from_secs(config.interval_secs);
    tracing::info!(
        "ttl_reaper: started (interval={}s, batch_size={})",
        config.interval_secs,
        config.batch_size,
    );

    wake.bind();
    loop {
        // Asleep for the whole interval; a stop interrupts it.
        let deadline = std::time::Instant::now() + interval;
        loop {
            if shutdown.load(Ordering::Acquire) {
                tracing::info!("ttl_reaper: shutdown");
                return;
            }
            let now = std::time::Instant::now();
            if now >= deadline {
                break;
            }
            wake.wait(Some(deadline - now));
        }

        if shutdown.load(Ordering::Acquire) {
            tracing::info!("ttl_reaper: shutdown");
            return;
        }

        let result = match reap_pass(
            engine,
            shard_id,
            config.batch_size,
            fields,
            oracle,
            pipeline,
            id_gen,
        ) {
            Ok(result) => result,
            Err(e) => {
                tracing::error!("ttl_reaper: field dictionary unavailable, pass skipped: {e}");
                continue;
            }
        };

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
