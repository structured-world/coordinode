//! CoordiNode integration test suite.
//!
//! Tests the full pipeline: parse → plan → execute over real CoordiNode storage.
//! Each test creates an isolated temp directory for crash safety.

mod integration {
    mod adaptive;
    mod advisor;
    mod alter_label;
    mod compound_queries;
    mod computed;
    mod concurrent;
    mod crash;
    mod cross_match;
    mod crud;
    mod cypher;
    mod document;
    mod drain;
    mod helpers;
    mod hnsw;
    mod merge_stress;
    mod mvcc;
    mod r069_mvcc_integration;
    mod r072_r073_background_workers;
    mod r074_oplog;
    mod r075_oplog_raft;
    mod schema;
    mod shared_engine;
    mod text_index;
    mod tiered_cache;
    mod validated_extra;
}
