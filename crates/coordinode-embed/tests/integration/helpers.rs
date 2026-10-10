//! Shared test fixtures for integration tests.
//!
//! Reduces ExecutionContext boilerplate from ~20 lines to 1 function call.
//! Three variants: legacy (no MVCC), MVCC (oracle + snapshot), MVCC + pipeline.

#![allow(dead_code)]

use std::collections::HashMap;

use coordinode_core::graph::intern::FieldInterner;
use coordinode_core::graph::node::NodeIdAllocator;
use coordinode_core::graph::types::VectorConsistencyMode;
use coordinode_core::txn::proposal::{ProposalIdGenerator, ProposalPipeline};
use coordinode_core::txn::read_concern::ReadConcernLevel;
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_query::executor::runner::{AdaptiveConfig, ExecutionContext, WriteStats};
use coordinode_storage::engine::StorageSnapshot;
use coordinode_storage::engine::core::StorageEngine;

/// Every stored version of every temporal node labelled `label`, in key order
/// (by node, then by `valid_from`): the history a bare MATCH, which projects
/// each node's state valid now, does not show. Each row holds `id`,
/// `valid_from` and the version's properties by name, `valid_to` and
/// `__deleted__` included when the version carries them.
#[allow(clippy::expect_used)]
pub fn temporal_versions(
    db: &coordinode_embed::Database,
    label: &str,
) -> Vec<std::collections::BTreeMap<String, coordinode_core::graph::types::Value>> {
    use coordinode_core::graph::node::{NodeRecord, decode_temporal_node_key};
    use coordinode_core::graph::types::Value;
    use coordinode_modality::{LocalNodeStore, NodeStore as _};

    // The shard an embedded database keeps its nodes in.
    const SHARD: u16 = 1;
    let interner = db.interner().expect("interner");
    let mut txn = coordinode_storage::engine::transaction::Transaction::new(
        db.engine(),
        None,
        Timestamp::ZERO,
        None,
    );
    let scanned = LocalNodeStore
        .prefix_scan_tracked(&mut txn, &LocalNodeStore.shard_scan_prefix(SHARD))
        .expect("scan nodes");
    let mut out = Vec::new();
    for (key, bytes) in scanned {
        let Some((_, id, valid_from)) = decode_temporal_node_key(&key) else {
            continue;
        };
        let record = NodeRecord::from_msgpack(&bytes).expect("decode version");
        if !record.has_label(label) {
            continue;
        }
        let mut row = std::collections::BTreeMap::new();
        for (field, value) in &record.props {
            let name = interner.resolve(*field).expect("registered field");
            row.insert(name.to_string(), value.clone());
        }
        for (name, value) in record.extra.iter().flatten() {
            row.insert(name.clone(), value.clone());
        }
        row.insert("id".to_string(), Value::Int(id.as_raw() as i64));
        row.insert("valid_from".to_string(), Value::Int(valid_from));
        out.push(row);
    }
    out
}

/// The stored definition of the index named `name`, resolved through the
/// catalog's name binding, or `None` when no index holds the name.
#[allow(clippy::expect_used)]
pub fn index_named(
    engine: &StorageEngine,
    name: &str,
) -> Option<coordinode_modality::IndexDefinition> {
    use coordinode_modality::{IndexStore as _, LocalIndexStore};
    let store = LocalIndexStore::new(engine);
    let id = store.resolve_name(name).expect("resolve name")?;
    store.load_definition(id).expect("load definition")
}

/// Publish `descriptor` through the catalog of `engine` in a direct-mode
/// transaction, as a database opened over it afterwards will find it.
#[allow(clippy::expect_used)]
pub fn publish_index(
    engine: &StorageEngine,
    descriptor: coordinode_modality::IndexDescriptor,
) -> coordinode_modality::IndexDefinition {
    use coordinode_modality::{IndexStore as _, LocalIndexStore};
    LocalIndexStore::new(engine)
        .publish_definition_txn(
            &mut coordinode_storage::engine::transaction::Transaction::new(
                engine,
                None,
                Timestamp::ZERO,
                None,
            ),
            descriptor,
        )
        .expect("publish index")
}

/// Publish `descriptor` as building with its build admitted, as a CREATE
/// whose statement never returned leaves it: the next open of a database
/// over `engine` takes the build up.
#[allow(clippy::expect_used)]
pub fn admit_index(
    engine: &StorageEngine,
    mut descriptor: coordinode_modality::IndexDescriptor,
) -> coordinode_modality::IndexDefinition {
    use coordinode_modality::{
        BuildFailure, IndexBuildRecord, IndexState, IndexStore as _, LocalIndexStore,
    };
    descriptor.state = IndexState::Building {
        written: 0,
        estimated_total: 0,
    };
    let store = LocalIndexStore::new(engine);
    let mut txn = coordinode_storage::engine::transaction::Transaction::new(
        engine,
        None,
        Timestamp::ZERO,
        None,
    );
    let def = store
        .publish_definition_txn(&mut txn, descriptor)
        .expect("publish index");
    store
        .put_build_txn(
            &mut txn,
            &IndexBuildRecord::accepted(def.id, def.generation, BuildFailure::Withdraw),
            None,
        )
        .expect("admit build");
    def
}

/// The budget a hand-built test context runs under: the default memory
/// limit, no deadline.
fn test_budget() -> std::sync::Arc<coordinode_core::budget::QueryBudget> {
    std::sync::Arc::new(coordinode_core::budget::QueryBudget::new(
        coordinode_core::budget::DEFAULT_QUERY_MEMORY_LIMIT,
    ))
}

/// Build an ExecutionContext in legacy mode (no MVCC, no oracle).
///
/// Used by tests that write directly to engine without MVCC versioning.
pub fn make_ctx_legacy<'a>(
    engine: &'a StorageEngine,
    interner: &'a mut FieldInterner,
    allocator: &'a NodeIdAllocator,
) -> ExecutionContext<'a> {
    ExecutionContext {
        engine,
        interner,
        field_registrar: None,
        id_allocator: allocator,
        shard_id: 1,
        scan_paging: None,
        operations: None,
        adaptive: AdaptiveConfig::default(),
        dedup_varlen_targets: false,
        snapshot_ts: None,
        valid_now: coordinode_query::executor::runner::wall_clock_us(),
        temporal_instants: Vec::new(),
        snapshot_pin: None,
        warnings: Vec::new(),
        write_stats: WriteStats::default(),
        key_claims: Default::default(),
        text_index: None,
        text_index_registry: None,
        vector_indexes: None,
        btree_index_registry: None,
        index_builds: None,
        extensions: None,
        vector_loader: None,
        mvcc_oracle: None,
        mvcc_read_ts: Timestamp::ZERO,
        txn: coordinode_storage::engine::transaction::Transaction::new(
            engine,
            None,
            Timestamp::ZERO,
            None,
        ),
        procedures: None,
        advisor: None,
        budget: test_budget(),
        unaccounted_operator: None,
        vector_consistency: VectorConsistencyMode::default(),
        vector_overfetch_factor: 1.2,
        vector_mvcc_stats: None,
        proposal_pipeline: None,
        proposal_id_gen: None,
        read_concern: ReadConcernLevel::Local,
        write_concern: WriteConcern::majority(),
        drain_buffer: None,
        nvme_write_buffer: None,
        mvcc_snapshot: None,
        cascade_depth: 0,
        cascade_depth_limit: 10,
        cascade_fire_counts: std::collections::HashMap::new(),
        cascade_fanout_limit: 100,
        cascade_chain: Vec::new(),
        after_commit_generation: 0,
        correlated_row: None,
        foreach_scope: None,
        feedback_cache: None,
        schema_label_cache: HashMap::new(),
        label_schema_cache: HashMap::new(),
        applied_watermark: None,
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        read_timeout: std::time::Duration::from_millis(2000),
        params: HashMap::new(),
    }
}

/// Build an ExecutionContext with MVCC enabled (oracle + read_ts).
///
/// `snapshot` is optional — if None, `execute()` will create one lazily.
pub fn make_ctx_mvcc<'a>(
    engine: &'a StorageEngine,
    oracle: &'a TimestampOracle,
    read_ts: Timestamp,
    snapshot: Option<StorageSnapshot>,
    interner: &'a mut FieldInterner,
    allocator: &'a NodeIdAllocator,
) -> ExecutionContext<'a> {
    ExecutionContext {
        engine,
        interner,
        field_registrar: None,
        id_allocator: allocator,
        shard_id: 0,
        scan_paging: None,
        operations: None,
        adaptive: AdaptiveConfig::default(),
        dedup_varlen_targets: false,
        snapshot_ts: None,
        valid_now: coordinode_query::executor::runner::wall_clock_us(),
        temporal_instants: Vec::new(),
        snapshot_pin: None,
        warnings: Vec::new(),
        write_stats: WriteStats::default(),
        key_claims: Default::default(),
        text_index: None,
        text_index_registry: None,
        vector_indexes: None,
        btree_index_registry: None,
        index_builds: None,
        extensions: None,
        vector_loader: None,
        mvcc_oracle: Some(oracle),
        mvcc_read_ts: read_ts,
        txn: coordinode_storage::engine::transaction::Transaction::new(
            engine,
            None,
            Timestamp::ZERO,
            None,
        ),
        procedures: None,
        advisor: None,
        budget: test_budget(),
        unaccounted_operator: None,
        vector_consistency: VectorConsistencyMode::default(),
        vector_overfetch_factor: 1.2,
        vector_mvcc_stats: None,
        proposal_pipeline: None,
        proposal_id_gen: None,
        read_concern: ReadConcernLevel::Local,
        write_concern: WriteConcern::majority(),
        drain_buffer: None,
        nvme_write_buffer: None,
        mvcc_snapshot: snapshot,
        cascade_depth: 0,
        cascade_depth_limit: 10,
        cascade_fire_counts: std::collections::HashMap::new(),
        cascade_fanout_limit: 100,
        cascade_chain: Vec::new(),
        after_commit_generation: 0,
        correlated_row: None,
        foreach_scope: None,
        feedback_cache: None,
        schema_label_cache: HashMap::new(),
        label_schema_cache: HashMap::new(),
        applied_watermark: None,
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        read_timeout: std::time::Duration::from_millis(2000),
        params: HashMap::new(),
    }
}

/// Build an ExecutionContext with MVCC + proposal pipeline.
#[allow(clippy::too_many_arguments)]
pub fn make_ctx_with_pipeline<'a>(
    engine: &'a StorageEngine,
    oracle: &'a TimestampOracle,
    read_ts: Timestamp,
    snapshot: Option<StorageSnapshot>,
    pipeline: &'a dyn ProposalPipeline,
    id_gen: &'a ProposalIdGenerator,
    interner: &'a mut FieldInterner,
    allocator: &'a NodeIdAllocator,
) -> ExecutionContext<'a> {
    ExecutionContext {
        engine,
        interner,
        field_registrar: None,
        id_allocator: allocator,
        shard_id: 0,
        scan_paging: None,
        operations: None,
        adaptive: AdaptiveConfig::default(),
        dedup_varlen_targets: false,
        snapshot_ts: None,
        valid_now: coordinode_query::executor::runner::wall_clock_us(),
        temporal_instants: Vec::new(),
        snapshot_pin: None,
        warnings: Vec::new(),
        write_stats: WriteStats::default(),
        key_claims: Default::default(),
        text_index: None,
        text_index_registry: None,
        vector_indexes: None,
        btree_index_registry: None,
        index_builds: None,
        extensions: None,
        vector_loader: None,
        mvcc_oracle: Some(oracle),
        mvcc_read_ts: read_ts,
        txn: coordinode_storage::engine::transaction::Transaction::new(
            engine,
            None,
            Timestamp::ZERO,
            None,
        ),
        procedures: None,
        advisor: None,
        budget: test_budget(),
        unaccounted_operator: None,
        vector_consistency: VectorConsistencyMode::default(),
        vector_overfetch_factor: 1.2,
        vector_mvcc_stats: None,
        proposal_pipeline: Some(pipeline),
        proposal_id_gen: Some(id_gen),
        read_concern: ReadConcernLevel::Local,
        write_concern: WriteConcern::majority(),
        drain_buffer: None,
        nvme_write_buffer: None,
        mvcc_snapshot: snapshot,
        cascade_depth: 0,
        cascade_depth_limit: 10,
        cascade_fire_counts: std::collections::HashMap::new(),
        cascade_fanout_limit: 100,
        cascade_chain: Vec::new(),
        after_commit_generation: 0,
        correlated_row: None,
        foreach_scope: None,
        feedback_cache: None,
        schema_label_cache: HashMap::new(),
        label_schema_cache: HashMap::new(),
        applied_watermark: None,
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        read_timeout: std::time::Duration::from_millis(2000),
        params: HashMap::new(),
    }
}
