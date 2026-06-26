//! Integration tests for Database::from_engine() (G063).
//!
//! Verifies that Database works correctly when initialized with an
//! externally-provided StorageEngine, TimestampOracle, and ProposalPipeline.
//! This is the cluster-mode path where server creates RaftProposalPipeline
//! and shares the engine with RaftNode.

#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_core::txn::write_concern::WriteConcernLevel;
use coordinode_embed::Database;
use coordinode_raft::proposal::OwnedLocalProposalPipeline;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use tempfile::tempdir;

/// Database::from_engine() opens successfully and executes queries.
#[test]
fn from_engine_basic_query() {
    let dir = tempdir().unwrap();
    let oracle = Arc::new(TimestampOracle::new());
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open_with_oracle(&config, oracle.clone()).unwrap();
    let engine = Arc::new(engine);
    let pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> =
        Arc::new(OwnedLocalProposalPipeline::new(&engine));

    let mut db = Database::from_engine(dir.path(), engine, oracle, pipeline).unwrap();

    db.execute_cypher("CREATE (n:User {name: 'alice'}) RETURN n")
        .unwrap();

    let results = db.execute_cypher("MATCH (n:User) RETURN n.name").unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].get("n.name").unwrap().as_str(), Some("alice"));
}

/// Database::from_engine() with volatile writes drains through the provided pipeline.
#[test]
fn from_engine_volatile_write_drains() {
    let dir = tempdir().unwrap();
    let oracle = Arc::new(TimestampOracle::new());
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open_with_oracle(&config, oracle.clone()).unwrap();
    let engine = Arc::new(engine);
    let pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> =
        Arc::new(OwnedLocalProposalPipeline::new(&engine));

    let mut db = Database::from_engine(dir.path(), engine, oracle, pipeline).unwrap();
    db.set_write_concern(WriteConcernLevel::Memory);

    // w:memory write — locally visible immediately
    db.execute_cypher("CREATE (n:Sensor {id: 's1'}) RETURN n")
        .unwrap();
    let results = db.execute_cypher("MATCH (n:Sensor) RETURN n.id").unwrap();
    assert_eq!(results.len(), 1);

    // Wait for drain thread to process (default 100ms interval)
    std::thread::sleep(std::time::Duration::from_millis(250));

    // Data still readable after drain
    let results = db.execute_cypher("MATCH (n:Sensor) RETURN n.id").unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].get("n.id").unwrap().as_str(), Some("s1"));
}

/// Database::from_engine() shares engine — external reads see written data.
#[test]
fn from_engine_shared_engine_visibility() {
    let dir = tempdir().unwrap();
    let oracle = Arc::new(TimestampOracle::new());
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open_with_oracle(&config, oracle.clone()).unwrap();
    let engine = Arc::new(engine);
    let pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> =
        Arc::new(OwnedLocalProposalPipeline::new(&engine));

    // Keep a reference to the engine for external reads
    let engine_ref = Arc::clone(&engine);

    let mut db = Database::from_engine(dir.path(), engine, oracle, pipeline).unwrap();

    // Write through Database
    db.execute_cypher("CREATE (n:Log {msg: 'hello'})").unwrap();

    // External engine reference can see the data (same Arc<StorageEngine>)
    use coordinode_storage::engine::partition::Partition;
    let iter = engine_ref.prefix_scan(Partition::Node, b"").unwrap();
    let count = iter.count();
    assert!(
        count >= 1,
        "expected ≥1 node via shared engine, got {count}"
    );
}

/// In cluster mode (`from_engine`), AFTER COMMIT triggers are NOT drained
/// inline on the write path — the leader-gated background worker owns the
/// drain. The event sits queued until the worker's `dispatch_after_commit_triggers`
/// call (simulated here) executes the body. This is the cluster counterpart of
/// the embedded inline-drain behaviour.
#[test]
fn from_engine_after_commit_trigger_drains_via_explicit_dispatch() {
    use coordinode_core::graph::types::Value;
    let dir = tempdir().unwrap();
    let oracle = Arc::new(TimestampOracle::new());
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open_with_oracle(&config, oracle.clone()).unwrap();
    let engine = Arc::new(engine);
    let pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> =
        Arc::new(OwnedLocalProposalPipeline::new(&engine));

    let mut db = Database::from_engine(dir.path(), engine, oracle, pipeline).unwrap();

    db.execute_cypher(
        "CREATE TRIGGER audit ON :User CREATE AFTER COMMIT \
         EXECUTE CREATE (e:AuditEntry {action: $event})",
    )
    .unwrap();
    db.execute_cypher("CREATE (u:User {id: 1})").unwrap();

    // Cluster mode: the write committed and the event is queued, but the inline
    // path did NOT drain it (that is the leader worker's job).
    assert_eq!(
        db.after_commit_pending_count(),
        1,
        "cluster mode must not drain AFTER COMMIT inline"
    );
    let audit = db.execute_cypher("MATCH (e:AuditEntry) RETURN e").unwrap();
    assert_eq!(audit.len(), 0, "body must not have run yet in cluster mode");

    // The leader worker's call drains the queue and fires the body.
    let report = db.dispatch_after_commit_triggers();
    assert_eq!(report.fired, 1);
    assert_eq!(db.after_commit_pending_count(), 0);
    let audit = db
        .execute_cypher("MATCH (e:AuditEntry) RETURN e.action AS act")
        .unwrap();
    assert_eq!(audit.len(), 1, "body fires once dispatched");
    assert_eq!(audit[0].get("act"), Some(&Value::String("CREATE".into())));
}

/// Database::from_engine() graceful shutdown flushes DrainBuffer.
#[test]
fn from_engine_drop_flushes_drain() {
    let dir = tempdir().unwrap();
    let oracle = Arc::new(TimestampOracle::new());
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open_with_oracle(&config, oracle.clone()).unwrap();
    let engine = Arc::new(engine);
    let pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> =
        Arc::new(OwnedLocalProposalPipeline::new(&engine));

    {
        let mut db = Database::from_engine(dir.path(), engine, oracle, pipeline).unwrap();
        db.set_write_concern(WriteConcernLevel::Memory);

        // Write with w:memory — buffered in DrainBuffer
        db.execute_cypher("CREATE (n:Temp {x: 1})").unwrap();
        db.execute_cypher("CREATE (n:Temp {x: 2})").unwrap();

        // Drop db — should flush DrainBuffer
    }

    // Re-open and verify data survived the graceful shutdown
    let mut db2 = Database::open(dir.path()).unwrap();
    let results = db2
        .execute_cypher("MATCH (n:Temp) RETURN n.x ORDER BY n.x")
        .unwrap();
    assert_eq!(
        results.len(),
        2,
        "expected 2 nodes after graceful shutdown flush"
    );
}
