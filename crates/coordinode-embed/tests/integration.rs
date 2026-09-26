//! CoordiNode integration test suite.
//!
//! Tests the full pipeline: parse → plan → execute over real CoordiNode storage.
//! Each test creates an isolated temp directory for crash safety.

mod integration {
    mod adaptive;
    mod advisor;
    mod alter_label;
    mod background_workers;
    mod backup_restore;
    mod btree_index;
    mod capacity_tracking;
    mod compound_queries;
    mod computed;
    mod concurrent;
    mod crash;
    mod create_table;
    mod cross_match;
    mod crud;
    mod cypher;
    mod document;
    mod drain;
    mod encrypted_search;
    mod from_engine;
    mod helpers;
    mod historical_index;
    mod hnsw;
    mod interactive_txn;
    mod merge_stress;
    mod multi_endpoint;
    mod mvcc;
    mod mvcc_snapshots;
    mod oplog_placement;
    mod oplog_segments;
    mod page_checksum;
    mod per_level_routing;
    mod retention;
    mod schema;
    mod shared_engine;
    mod text_index;
    mod tiered_cache;
    mod validated_extra;
    mod vector_index_build_lifecycle;
    mod vector_index_ddl;
}
