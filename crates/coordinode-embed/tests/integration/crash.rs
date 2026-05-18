//! Integration tests: Crash safety and data integrity.
//!
//! Tests that data persists correctly across close/reopen cycles
//! and that the storage engine maintains consistency.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use coordinode_embed::Database;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

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
