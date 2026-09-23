//! Integration tests: Crash safety and data integrity.
//!
//! Tests that data persists correctly across close/reopen cycles
//! and that the storage engine maintains consistency.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use coordinode_embed::Database;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

// ── Process kill ────────────────────────────────────────────────────

/// Set by the parent test; the child test does nothing without it.
const CRASH_CHILD_DIR: &str = "COORDINODE_CRASH_CHILD_DIR";

/// Child half of `acknowledged_write_survives_process_kill`: one acknowledged
/// write, then the process dies without running a single destructor, as it
/// does under SIGKILL. Run on its own it has no directory and returns.
#[test]
fn crash_child_writes_then_aborts() {
    let Some(dir) = std::env::var_os(CRASH_CHILD_DIR) else {
        return;
    };
    let mut db = Database::open(std::path::Path::new(&dir)).expect("open in child");
    db.execute_cypher("CREATE (:Survivor {k: 1})")
        .expect("write in child");
    std::process::abort();
}

/// A write acknowledged before the process was killed survives it, and the
/// database opens. A kill leaves the journal's active segment without the
/// footer a clean close writes; reopening treated it as sealed and failed on
/// the missing footer every time, so the installation could not be opened
/// until someone deleted the journal and with it the acknowledged write.
#[test]
fn acknowledged_write_survives_process_kill() {
    let dir = tempfile::tempdir().expect("tempdir");
    let status = std::process::Command::new(std::env::current_exe().expect("test binary"))
        .args([
            "--exact",
            "integration::crash::crash_child_writes_then_aborts",
            "--nocapture",
        ])
        .env(CRASH_CHILD_DIR, dir.path())
        .status()
        .expect("run the child");
    // The child must have reached its abort, not failed before the write.
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        assert_eq!(
            status.signal(),
            Some(6),
            "the child must die by abort after its write, got {status:?}"
        );
    }
    #[cfg(not(unix))]
    assert!(!status.success(), "the child must die, got {status:?}");

    let mut db = Database::open(dir.path()).expect("reopen after the kill");
    let rows = db
        .execute_cypher("MATCH (n:Survivor) RETURN n.k AS k")
        .expect("read after the kill");
    assert_eq!(rows.len(), 1, "the acknowledged write was lost");
}

// ── Close/Reopen persistence ────────────────────────────────────────

#[test]
fn nodes_persist_across_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");

    {
        let mut db = Database::open(dir.path()).expect("open");
        db.execute_cypher("CREATE (n:User {name: 'Alice', age: 30})")
            .expect("create");
        db.execute_cypher("CREATE (n:User {name: 'Bob', age: 25})")
            .expect("create");
    }

    {
        let mut db = Database::open(dir.path()).expect("reopen");
        let rows = db
            .execute_cypher("MATCH (n:User) RETURN n.name ORDER BY n.name")
            .expect("match");
        assert_eq!(rows.len(), 2);
        // Verify property values survive reopen (G027 fix: interner persistence)
        let mut names: Vec<String> = rows
            .iter()
            .filter_map(|r| {
                r.get("n.name")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            })
            .collect();
        names.sort();
        assert_eq!(
            names,
            vec!["Alice", "Bob"],
            "property values must survive reopen"
        );
    }
}

#[test]
fn relationship_creation_succeeds() {
    let (mut db, _dir) = open_db();
    // Verify relationship creation pattern doesn't error
    let result =
        db.execute_cypher("CREATE (a:Person {name: 'Alice'})-[:KNOWS]->(b:Person {name: 'Bob'})");
    assert!(result.is_ok(), "relationship creation should not error");
}

#[test]
fn deletes_persist_across_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");

    {
        let mut db = Database::open(dir.path()).expect("open");
        db.execute_cypher("CREATE (n:Temp {val: 1})")
            .expect("create");
        db.execute_cypher("MATCH (n:Temp) DELETE n")
            .expect("delete");
    }

    {
        let mut db = Database::open(dir.path()).expect("reopen");
        let rows = db.execute_cypher("MATCH (n:Temp) RETURN n").expect("match");
        assert!(rows.is_empty(), "deleted nodes should not reappear");
    }
}

#[test]
fn set_property_within_session() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (n:Config {key: 'timeout', value: '30'})")
        .expect("create");
    db.execute_cypher("MATCH (n:Config {key: 'timeout'}) SET n.value = '60'")
        .expect("update");
    let rows = db
        .execute_cypher("MATCH (n:Config {key: 'timeout'}) RETURN n.value")
        .expect("match");
    assert_eq!(rows.len(), 1);
}

// ── Storage engine integrity ────────────────────────────────────────

#[test]
fn storage_partitions_are_isolated() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("open");

    // Write to Node partition
    engine
        .put(Partition::Node, b"test:1", b"node_data")
        .expect("put node");

    // Same key in Adj partition should be independent
    engine
        .put(Partition::Adj, b"test:1", b"adj_data")
        .expect("put adj");

    let node_val = engine.get(Partition::Node, b"test:1").expect("get node");
    let adj_val = engine.get(Partition::Adj, b"test:1").expect("get adj");

    assert_eq!(node_val.as_deref(), Some(b"node_data".as_slice()));
    assert_eq!(adj_val.as_deref(), Some(b"adj_data".as_slice()));
}

#[test]
fn storage_survives_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");

    {
        let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            dir.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        let engine = StorageEngine::open(&config).expect("open");
        engine
            .put(Partition::Node, b"persist:key", b"persist:value")
            .expect("put");
    }

    {
        let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            dir.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        let engine = StorageEngine::open(&config).expect("reopen");
        let val = engine.get(Partition::Node, b"persist:key").expect("get");
        assert_eq!(val.as_deref(), Some(b"persist:value".as_slice()));
    }
}

#[test]
fn batch_insert_within_session() {
    let (mut db, _dir) = open_db();

    for i in 0..5 {
        db.execute_cypher(&format!("CREATE (n:Cycle {{round: {i}}})",))
            .expect("create");
    }

    let rows = db
        .execute_cypher("MATCH (n:Cycle) RETURN n")
        .expect("match all");
    assert_eq!(rows.len(), 5);
}

// ── Large data ──────────────────────────────────────────────────────

#[test]
fn large_batch_insert() {
    let (mut db, _dir) = open_db();

    for i in 0..100 {
        db.execute_cypher(&format!("CREATE (n:Batch {{id: {i}}})",))
            .expect("create");
    }

    let rows = db
        .execute_cypher("MATCH (n:Batch) RETURN count(n)")
        .expect("count");
    assert_eq!(rows.len(), 1);
}

fn open_db() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open db");
    (db, dir)
}
