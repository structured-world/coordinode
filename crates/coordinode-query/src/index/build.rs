//! Online index build: scan existing nodes and create index entries.
//!
//! Three-phase protocol (adapted from MongoDB's hybrid builder):
//! 1. Setup: validate index definition
//! 2. InProgress: scan all nodes of the target label, create index entries
//! 3. Commit: mark index as ready (or Abort on errors)
//!
//! Current phase scope: direct storage writes (single-node).
//! CE product (3-node HA): index build runs on Raft leader only.
//! Commit is a Raft log entry replicated to all nodes.
//! Side-writes interception for concurrent writers added with Raft (distributed mode).

use coordinode_core::graph::intern::FieldInterner;
use coordinode_core::graph::node::NodeId;
use coordinode_core::graph::types::Value;
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_modality::{IndexStore as _, LocalIndexStore, LocalNodeStore, NodeStore};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::transaction::Transaction;

use super::definition::IndexDefinition;
use super::ops::{create_index_entry, save_index_definition};

/// Index build state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexBuildState {
    /// Initial setup: definition validated.
    Setup,
    /// Collection scan in progress.
    InProgress,
    /// Build completed successfully, index is usable.
    Committed,
    /// Build failed, index entries cleaned up.
    Aborted,
}

/// Progress and result of an index build.
#[derive(Debug, Default)]
pub struct IndexBuildResult {
    /// Current state of the build.
    pub state: Option<IndexBuildState>,
    /// Number of nodes scanned.
    pub scanned: usize,
    /// Number of index entries created.
    pub indexed: usize,
    /// Number of nodes skipped (no matching property, filtered out).
    pub skipped: usize,
    /// Unique constraint violations detected.
    pub violations: Vec<String>,
    /// Non-fatal errors.
    pub errors: Vec<String>,
}

/// Build an index online by scanning existing nodes.
///
/// Scans all nodes with the target label in the given shard,
/// creates index entries for each matching node, and saves
/// the index definition on success.
///
/// For unique indexes, violations are collected during the scan.
/// If any remain at commit time, the build is aborted.
///
/// # Cluster-ready notes
/// - This function runs on the Raft leader only in CE 3-node HA mode.
/// - Distributed mode adds side-writes interception for concurrent writers
///   and Raft log commit for the index definition.
/// - The scan-drain-commit protocol is compatible with distributed replay.
pub fn build_index(
    engine: &StorageEngine,
    index: &IndexDefinition,
    interner: &FieldInterner,
    shard_id: u16,
) -> IndexBuildResult {
    let mut result = IndexBuildResult {
        state: Some(IndexBuildState::Setup),
        ..Default::default()
    };

    // Phase 1: Setup — validate
    if index.properties.is_empty() {
        result
            .errors
            .push("index must have at least one property".into());
        result.state = Some(IndexBuildState::Aborted);
        return result;
    }

    // Phase 2: InProgress — scan nodes
    result.state = Some(IndexBuildState::InProgress);

    let nodes = LocalNodeStore;
    // Index build is a bulk background scan — cheap direct-mode transaction
    // (no snapshot/OCC); reads the latest committed state, like the engine
    // prefix scan it replaces.
    let scan_txn = Transaction::new(engine, None, Timestamp::ZERO, None);
    let scanned = match nodes.scan_shard(&scan_txn, shard_id) {
        Ok(v) => v,
        Err(e) => {
            result.errors.push(format!("scan error: {e}"));
            result.state = Some(IndexBuildState::Aborted);
            return result;
        }
    };

    // Resolve property field IDs
    let field_ids: Vec<Option<u32>> = index
        .properties
        .iter()
        .map(|p| interner.lookup(p))
        .collect();

    for (node_id_typed, record) in scanned {
        result.scanned += 1;
        let node_id = node_id_typed.as_raw();

        // Label filter: only index nodes matching the target label
        if !record.has_label(&index.label) {
            result.skipped += 1;
            continue;
        }

        // Check partial filter
        if index.filter.is_some() {
            let props: Vec<(String, Value)> = record
                .props
                .iter()
                .filter_map(|(fid, val)| {
                    interner
                        .resolve(*fid)
                        .map(|name| (name.to_string(), val.clone()))
                })
                .collect();

            if !index.matches_filter(&props) {
                result.skipped += 1;
                continue;
            }
        }

        // For single-field index: get the property value
        if !index.is_compound() {
            let value = field_ids[0]
                .and_then(|fid| record.get(fid))
                .cloned()
                .unwrap_or(Value::Null);

            // Sparse check
            if index.sparse && value.is_null() {
                result.skipped += 1;
                continue;
            }

            match create_index_entry(engine, index, NodeId::from_raw(node_id), &value) {
                Ok(()) => result.indexed += 1,
                Err(coordinode_storage::error::StorageError::Conflict) => {
                    result
                        .violations
                        .push(format!("duplicate key for node {node_id}: {value:?}"));
                }
                Err(e) => {
                    result.errors.push(format!("index write error: {e}"));
                }
            }
        } else {
            // Compound: collect all property values
            let values: Vec<Value> = field_ids
                .iter()
                .map(|fid| {
                    fid.and_then(|id| record.get(id))
                        .cloned()
                        .unwrap_or(Value::Null)
                })
                .collect();

            if index.sparse && values.iter().any(|v| v.is_null()) {
                result.skipped += 1;
                continue;
            }

            // Use compound index entry (Layer-4 store hides the key encoding).
            match LocalIndexStore::new(engine).put_entry(
                &index.name,
                &values,
                NodeId::from_raw(node_id),
            ) {
                Ok(()) => result.indexed += 1,
                Err(e) => {
                    result.errors.push(format!("compound index write: {e}"));
                }
            }
        }
    }

    // Phase 3: Commit or Abort
    if !result.violations.is_empty() {
        // Unique constraint violations — abort
        // Clean up: drop all created index entries
        let _ = LocalIndexStore::new(engine).clear(&index.name);
        result.state = Some(IndexBuildState::Aborted);
    } else if !result.errors.is_empty() && result.indexed == 0 {
        result.state = Some(IndexBuildState::Aborted);
    } else {
        // Commit: save index definition
        match save_index_definition(engine, index) {
            Ok(()) => result.state = Some(IndexBuildState::Committed),
            Err(e) => {
                result.errors.push(format!("definition save error: {e}"));
                result.state = Some(IndexBuildState::Aborted);
            }
        }
    }

    result
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use coordinode_core::graph::node::NodeRecord;
    use coordinode_storage::engine::config::{
        Durability, EndpointConfig, Media, StorageConfig, Tier,
    };

    fn test_engine(dir: &std::path::Path) -> StorageEngine {
        let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            dir,
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        StorageEngine::open(&config).expect("open engine")
    }

    fn insert_node(
        engine: &StorageEngine,
        shard_id: u16,
        node_id: u64,
        label: &str,
        props: &[(&str, Value)],
        interner: &mut FieldInterner,
    ) {
        use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
        use coordinode_core::txn::write_concern::WriteConcern;
        use coordinode_modality::{LocalNodeStore, NodeStore as _};
        use coordinode_storage::engine::transaction::{CommitContext, Transaction};
        let mut record = NodeRecord::new(label);
        for (name, value) in props {
            let field_id = interner.intern(name);
            record.set(field_id, value.clone());
        }
        let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
        let read_ts = oracle.next();
        let mut txn = Transaction::new(engine, Some(&oracle), read_ts, Some(engine.snapshot()));
        LocalNodeStore
            .put(&mut txn, shard_id, NodeId::from_raw(node_id), &record)
            .expect("put");
        let wc = WriteConcern::majority();
        let ctx = CommitContext {
            write_concern: &wc,
            pipeline: None,
            id_gen: None,
            drain_buffer: None,
            nvme_write_buffer: None,
        };
        txn.commit(&ctx).expect("commit");
    }

    #[test]
    fn build_index_on_existing_nodes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let mut interner = FieldInterner::new();

        // Create some nodes
        insert_node(
            &engine,
            1,
            1,
            "User",
            &[("email", Value::String("alice@test.com".into()))],
            &mut interner,
        );
        insert_node(
            &engine,
            1,
            2,
            "User",
            &[("email", Value::String("bob@test.com".into()))],
            &mut interner,
        );
        insert_node(
            &engine,
            1,
            3,
            "Movie",
            &[("title", Value::String("Matrix".into()))],
            &mut interner,
        );

        // Build index on User.email
        let idx = IndexDefinition::btree("user_email", "User", "email");
        let result = build_index(&engine, &idx, &interner, 1);

        assert_eq!(result.state, Some(IndexBuildState::Committed));
        assert_eq!(result.scanned, 3); // All nodes scanned
        assert_eq!(result.indexed, 2); // Only User nodes indexed
        assert_eq!(result.skipped, 1); // Movie skipped
        assert!(result.violations.is_empty());

        // Verify index entries exist
        let alice = super::super::ops::index_scan_exact(
            &engine,
            "user_email",
            &Value::String("alice@test.com".into()),
        )
        .expect("scan");
        assert_eq!(alice, vec![1]);
    }

    #[test]
    fn build_unique_index_aborts_on_duplicates() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let mut interner = FieldInterner::new();

        // Two nodes with same email
        insert_node(
            &engine,
            1,
            1,
            "User",
            &[("email", Value::String("same@test.com".into()))],
            &mut interner,
        );
        insert_node(
            &engine,
            1,
            2,
            "User",
            &[("email", Value::String("same@test.com".into()))],
            &mut interner,
        );

        let idx = IndexDefinition::btree("user_email", "User", "email").unique();
        let result = build_index(&engine, &idx, &interner, 1);

        assert_eq!(result.state, Some(IndexBuildState::Aborted));
        assert!(!result.violations.is_empty());

        // Index entries should be cleaned up
        let entries = super::super::ops::index_scan(&engine, &idx).expect("scan");
        assert!(entries.is_empty());
    }

    #[test]
    fn build_sparse_index_skips_nulls() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let mut interner = FieldInterner::new();

        insert_node(
            &engine,
            1,
            1,
            "User",
            &[("bio", Value::String("Developer".into()))],
            &mut interner,
        );
        insert_node(
            &engine,
            1,
            2,
            "User",
            &[("name", Value::String("Bob".into()))],
            &mut interner,
        ); // No bio

        let idx = IndexDefinition::btree("user_bio", "User", "bio").sparse();
        let result = build_index(&engine, &idx, &interner, 1);

        assert_eq!(result.state, Some(IndexBuildState::Committed));
        assert_eq!(result.indexed, 1); // Only node 1 has bio
    }

    #[test]
    fn build_index_saves_definition() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let mut interner = FieldInterner::new();

        insert_node(
            &engine,
            1,
            1,
            "User",
            &[("name", Value::String("Alice".into()))],
            &mut interner,
        );

        let idx = IndexDefinition::btree("user_name", "User", "name");
        let result = build_index(&engine, &idx, &interner, 1);
        assert_eq!(result.state, Some(IndexBuildState::Committed));

        // Definition should be loadable
        let loaded = super::super::ops::load_index_definition(&engine, "user_name")
            .expect("load")
            .expect("should exist");
        assert_eq!(loaded.name, "user_name");
    }

    #[test]
    fn build_empty_label_succeeds() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let interner = FieldInterner::new();

        // No nodes exist
        let idx = IndexDefinition::btree("user_email", "User", "email");
        let result = build_index(&engine, &idx, &interner, 1);

        assert_eq!(result.state, Some(IndexBuildState::Committed));
        assert_eq!(result.scanned, 0);
        assert_eq!(result.indexed, 0);
    }

    #[test]
    fn build_index_with_partial_filter() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let mut interner = FieldInterner::new();

        insert_node(
            &engine,
            1,
            1,
            "User",
            &[
                ("email", Value::String("alice@test.com".into())),
                ("status", Value::String("active".into())),
            ],
            &mut interner,
        );
        insert_node(
            &engine,
            1,
            2,
            "User",
            &[
                ("email", Value::String("bob@test.com".into())),
                ("status", Value::String("inactive".into())),
            ],
            &mut interner,
        );

        let idx = IndexDefinition::btree("active_email", "User", "email").with_filter(
            super::super::definition::PartialFilter::PropertyEquals {
                property: "status".into(),
                value: "active".into(),
            },
        );

        let result = build_index(&engine, &idx, &interner, 1);
        assert_eq!(result.state, Some(IndexBuildState::Committed));
        assert_eq!(result.indexed, 1); // Only active user
        assert_eq!(result.skipped, 1); // Inactive filtered out
    }
}
