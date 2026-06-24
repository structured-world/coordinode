//! Prometheus metric catalog: defines all CE metric families.
//!
//! Metrics are registered once at startup. Actual recording happens
//! in the respective modules (storage, query, graph).
//!
//! # Cluster-ready notes
//! - Metrics are per-node (Prometheus scrapes each node independently).
//! - No shared metric state between nodes.

/// Register all CE metric families with the Prometheus recorder.
///
/// Call once at startup, after the Prometheus exporter is installed.
pub fn register_all_metrics() {
    register_storage_metrics();
    register_query_metrics();
    register_graph_metrics();
    register_replication_metrics();
    register_network_metrics();
}

fn register_storage_metrics() {
    // LSM tree
    metrics::describe_gauge!("coordinode_storage_lsm_levels", "Number of LSM levels");
    metrics::describe_gauge!("coordinode_storage_lsm_bytes", "Bytes per LSM level");
    metrics::describe_counter!(
        "coordinode_storage_compaction_total",
        "Compactions completed"
    );
    metrics::describe_histogram!(
        "coordinode_storage_compaction_duration_seconds",
        "Compaction duration"
    );

    // Block cache
    metrics::describe_counter!("coordinode_cache_hit_total", "Cache hits");
    metrics::describe_counter!("coordinode_cache_miss_total", "Cache misses");
    metrics::describe_counter!("coordinode_cache_eviction_total", "Cache evictions");
    metrics::describe_gauge!("coordinode_cache_bytes", "Current cache size in bytes");

    // WAL
    metrics::describe_counter!(
        "coordinode_wal_bytes_written_total",
        "Total WAL bytes written"
    );
    metrics::describe_histogram!("coordinode_wal_sync_duration_seconds", "WAL fsync duration");
    metrics::describe_counter!("coordinode_wal_entries_total", "WAL entries written");

    // BlobStore
    metrics::describe_gauge!("coordinode_blobstore_chunks_total", "Total chunks stored");
    metrics::describe_gauge!("coordinode_blobstore_bytes_total", "Total blob bytes");

    // Page integrity
    metrics::describe_counter!("coordinode_page_checksum_verified_total", "Pages verified");
    metrics::describe_counter!(
        "coordinode_page_checksum_failed_total",
        "Pages with checksum failure"
    );
    metrics::describe_counter!(
        "coordinode_scrub_pages_scanned_total",
        "Background scrub progress"
    );
    metrics::describe_counter!(
        "coordinode_scrub_errors_total",
        "Corrupt blocks found by the background scrub"
    );
    metrics::describe_gauge!(
        "coordinode_scrub_blocks_checked",
        "Blocks verified in the last completed scrub cycle"
    );
    metrics::describe_gauge!(
        "coordinode_scrub_last_timestamp_seconds",
        "Unix time of the last completed scrub cycle"
    );
    metrics::describe_gauge!(
        "coordinode_scrub_duration_seconds",
        "Wall-clock duration of the last completed scrub cycle"
    );
    metrics::describe_counter!(
        "coordinode_scrub_repairs_total",
        "Partitions repaired from healthy peers after the scrub found corruption"
    );
}

fn register_query_metrics() {
    // Cypher
    metrics::describe_histogram!(
        "coordinode_query_duration_seconds",
        "Query duration by type and operation"
    );
    metrics::describe_counter!("coordinode_query_total", "Queries executed by type");
    metrics::describe_counter!("coordinode_query_errors_total", "Query errors by type");
    metrics::describe_gauge!("coordinode_query_active", "Currently executing queries");

    // Vector search
    metrics::describe_histogram!(
        "coordinode_vector_candidates_scanned",
        "HNSW candidates scanned per search"
    );
    metrics::describe_gauge!("coordinode_vector_index_size", "Vectors in HNSW index");
    metrics::describe_gauge!(
        "coordinode_vector_index_state",
        "Vector index serving state per {label, property}: 0=ready, 1=rebuilding, 2=offline"
    );
    metrics::describe_gauge!(
        "coordinode_vector_index_lag_hlc",
        "Vector index freshness lag per {label, property}: committed HLC minus indexed HLC (microseconds)"
    );

    // Full-text search
    metrics::describe_gauge!(
        "coordinode_fulltext_index_docs",
        "Documents in tantivy index"
    );

    // GraphQL subscriptions
    metrics::describe_gauge!(
        "coordinode_subscription_active",
        "Active GraphQL subscriptions"
    );
}

fn register_graph_metrics() {
    metrics::describe_gauge!("coordinode_graph_nodes_total", "Node count per label");
    metrics::describe_gauge!("coordinode_graph_edges_total", "Edge count per type");
    metrics::describe_gauge!(
        "coordinode_graph_properties_total",
        "Total properties stored"
    );
    metrics::describe_gauge!("coordinode_graph_index_count", "Indexes per type");
    metrics::describe_histogram!(
        "coordinode_graph_traversal_hops_total",
        "Traversal hops per query"
    );
}

fn register_replication_metrics() {
    // Raft (populated in distributed mode)
    metrics::describe_gauge!("coordinode_raft_term", "Current Raft term");
    metrics::describe_gauge!("coordinode_raft_commit_index", "Committed log index");
    metrics::describe_gauge!("coordinode_raft_applied_index", "Applied log index");
    metrics::describe_gauge!(
        "coordinode_raft_leader",
        "1 if this node is leader, 0 otherwise"
    );
    metrics::describe_counter!(
        "coordinode_raft_election_total",
        "Leader elections observed"
    );
    metrics::describe_counter!("coordinode_raft_snapshot_total", "Snapshots taken");
    metrics::describe_histogram!(
        "coordinode_raft_snapshot_duration_seconds",
        "Snapshot duration"
    );

    // Proposal pipeline (G036)
    metrics::describe_counter!(
        "coordinode_raft_proposals_total",
        "Proposals by outcome: ok, timeout, not_leader, error"
    );
    metrics::describe_counter!(
        "coordinode_raft_proposal_retries_total",
        "Proposal timeout retries (each retry increments once)"
    );
    metrics::describe_counter!(
        "coordinode_raft_proposals_bypassed_total",
        "Proposals that bypassed rate limiter (delta/membership)"
    );
    metrics::describe_histogram!(
        "coordinode_raft_proposal_duration_seconds",
        "Proposal end-to-end latency (from first attempt to final outcome)"
    );
    metrics::describe_gauge!(
        "coordinode_raft_rate_limiter_pending",
        "Available rate limiter permits (lower = more backpressure)"
    );
}

fn register_network_metrics() {
    metrics::describe_counter!(
        "coordinode_grpc_requests_total",
        "Total gRPC requests by method"
    );
    metrics::describe_histogram!(
        "coordinode_grpc_duration_seconds",
        "gRPC request duration by method"
    );
    metrics::describe_counter!("coordinode_grpc_errors_total", "gRPC errors by code");
    metrics::describe_gauge!(
        "coordinode_grpc_connections_active",
        "Active gRPC connections"
    );
}
