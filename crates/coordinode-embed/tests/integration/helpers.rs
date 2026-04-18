//! Shared test fixtures for integration tests (G056).
//!
//! Reduces ExecutionContext boilerplate from ~20 lines to 1 function call.
//! Three variants: legacy (no MVCC), MVCC (oracle + snapshot), MVCC + pipeline.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use coordinode_core::graph::intern::FieldInterner;
use coordinode_core::graph::node::NodeIdAllocator;
use coordinode_core::graph::types::VectorConsistencyMode;
use coordinode_core::txn::proposal::{ProposalIdGenerator, ProposalPipeline};
use coordinode_core::txn::read_concern::ReadConcernLevel;
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_query::executor::runner::{AdaptiveConfig, ExecutionContext, WriteStats};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::StorageSnapshot;

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
        id_allocator: allocator,
        shard_id: 1,
        adaptive: AdaptiveConfig::default(),
        snapshot_ts: None,
        retention_window_us: 7 * 24 * 3600 * 1_000_000,
        warnings: Vec::new(),
        write_stats: WriteStats::default(),
        text_index: None,
        text_index_registry: None,
        vector_index_registry: None,
        btree_index_registry: None,
        vector_loader: None,
        mvcc_oracle: None,
        mvcc_read_ts: Timestamp::ZERO,
        mvcc_write_buffer: HashMap::new(),
        procedure_ctx: None,
        mvcc_read_set: HashSet::new(),
        vector_consistency: VectorConsistencyMode::default(),
        vector_overfetch_factor: 1.2,
        vector_mvcc_stats: None,
        proposal_pipeline: None,
        proposal_id_gen: None,
        read_concern: ReadConcernLevel::Local,
        write_concern: WriteConcern::majority(),
        drain_buffer: None,
        nvme_write_buffer: None,
        merge_adj_adds: HashMap::new(),
        merge_adj_removes: HashMap::new(),
        mvcc_snapshot: None,
        adj_snapshot: None,
        merge_node_deltas: Vec::new(),
        correlated_row: None,
        feedback_cache: None,
        schema_label_cache: HashMap::new(),
        applied_watermark: None,
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
        id_allocator: allocator,
        shard_id: 0,
        adaptive: AdaptiveConfig::default(),
        snapshot_ts: None,
        retention_window_us: 7 * 24 * 3600 * 1_000_000,
        warnings: Vec::new(),
        write_stats: WriteStats::default(),
        text_index: None,
        text_index_registry: None,
        vector_index_registry: None,
        btree_index_registry: None,
        vector_loader: None,
        mvcc_oracle: Some(oracle),
        mvcc_read_ts: read_ts,
        mvcc_write_buffer: HashMap::new(),
        procedure_ctx: None,
        mvcc_read_set: HashSet::new(),
        vector_consistency: VectorConsistencyMode::default(),
        vector_overfetch_factor: 1.2,
        vector_mvcc_stats: None,
        proposal_pipeline: None,
        proposal_id_gen: None,
        read_concern: ReadConcernLevel::Local,
        write_concern: WriteConcern::majority(),
        drain_buffer: None,
        nvme_write_buffer: None,
        merge_adj_adds: HashMap::new(),
        merge_adj_removes: HashMap::new(),
        mvcc_snapshot: snapshot,
        adj_snapshot: None,
        merge_node_deltas: Vec::new(),
        correlated_row: None,
        feedback_cache: None,
        schema_label_cache: HashMap::new(),
        applied_watermark: None,
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
        id_allocator: allocator,
        shard_id: 0,
        adaptive: AdaptiveConfig::default(),
        snapshot_ts: None,
        retention_window_us: 7 * 24 * 3600 * 1_000_000,
        warnings: Vec::new(),
        write_stats: WriteStats::default(),
        text_index: None,
        text_index_registry: None,
        vector_index_registry: None,
        btree_index_registry: None,
        vector_loader: None,
        mvcc_oracle: Some(oracle),
        mvcc_read_ts: read_ts,
        mvcc_write_buffer: HashMap::new(),
        procedure_ctx: None,
        mvcc_read_set: HashSet::new(),
        vector_consistency: VectorConsistencyMode::default(),
        vector_overfetch_factor: 1.2,
        vector_mvcc_stats: None,
        proposal_pipeline: Some(pipeline),
        proposal_id_gen: Some(id_gen),
        read_concern: ReadConcernLevel::Local,
        write_concern: WriteConcern::majority(),
        drain_buffer: None,
        nvme_write_buffer: None,
        merge_adj_adds: HashMap::new(),
        merge_adj_removes: HashMap::new(),
        mvcc_snapshot: snapshot,
        adj_snapshot: None,
        merge_node_deltas: Vec::new(),
        correlated_row: None,
        feedback_cache: None,
        schema_label_cache: HashMap::new(),
        applied_watermark: None,
        params: HashMap::new(),
    }
}
