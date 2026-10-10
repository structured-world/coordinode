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
    mod constraints;
    mod crash;
    mod create_table;
    mod cross_match;
    mod crud;
    mod cypher;
    mod document;
    mod drain;
    mod duplicate_repair;
    mod encrypted_search;
    mod field_dictionary;
    mod from_engine;
    mod helpers;
    mod historical_index;
    mod hnsw;
    mod index_builds;
    mod index_profiles;
    mod index_repair;
    mod interactive_txn;
    mod label_count;
    mod merge_stress;
    mod multi_endpoint;
    mod multi_label;
    mod mvcc;
    mod mvcc_snapshots;
    mod node_id_lookup;
    mod oplog_placement;
    mod oplog_segments;
    mod page_checksum;
    mod per_level_routing;
    mod procedures;
    mod retention;
    mod schema;
    mod schema_publication;
    mod shared_engine;
    mod temporal_node_reads;
    mod temporal_strict_schema;
    mod temporal_traversal;
    mod text_index;
    mod tiered_cache;
    mod unique_admission;
    mod validated_extra;
    mod vector_index_build_lifecycle;
    mod vector_index_ddl;
}
