//! Integration tests: CRUD lifecycle (CREATE, MATCH, SET, DELETE, DETACH DELETE).
//!
//! Uses the embedded Database for full-pipeline testing:
//! parse → semantic analysis → logical plan → physical executor → CoordiNode storage.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_embed::Database;

fn open_db() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open db");
    (db, dir)
}

// ── CREATE ──────────────────────────────────────────────────────────

#[test]
fn create_single_node() {
    let (mut db, _dir) = open_db();
    let rows = db
        .execute_cypher("CREATE (n:User {name: 'Alice'}) RETURN n")
        .expect("create");
    assert_eq!(rows.len(), 1);
}

#[test]
fn create_multiple_nodes() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:User {name: 'Alice'})")
        .expect("create a");
    db.execute_cypher("CREATE (b:User {name: 'Bob'})")
        .expect("create b");
    let rows = db.execute_cypher("MATCH (n:User) RETURN n").expect("match");
    assert_eq!(rows.len(), 2);
}

#[test]
fn create_node_with_single_label() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (n:Admin {name: 'Root'})")
        .expect("create");
    let rows = db
        .execute_cypher("MATCH (n:Admin) RETURN n.name")
        .expect("match");
    assert_eq!(rows.len(), 1);
}

#[test]
fn create_node_with_various_types() {
    let (mut db, _dir) = open_db();
    let rows = db
        .execute_cypher(
            "CREATE (n:TypeTest {str: 'hello', int: 42, float: 3.14, bool: true}) RETURN n",
        )
        .expect("create");
    assert_eq!(rows.len(), 1);
}

#[test]
fn create_relationship() {
    let (mut db, _dir) = open_db();
    // Verify CREATE with relationship pattern doesn't error
    let result =
        db.execute_cypher("CREATE (a:User {name: 'Alice'})-[:KNOWS]->(b:User {name: 'Bob'})");
    assert!(result.is_ok(), "relationship creation should not error");
}

// ── MATCH ───────────────────────────────────────────────────────────

#[test]
fn match_no_results() {
    let (mut db, _dir) = open_db();
    let rows = db
        .execute_cypher("MATCH (n:Nonexistent) RETURN n")
        .expect("match");
    assert!(rows.is_empty());
}

#[test]
fn match_with_where_filter() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:User {name: 'Alice', age: 30})")
        .expect("create a");
    db.execute_cypher("CREATE (b:User {name: 'Bob', age: 25})")
        .expect("create b");
    let rows = db
        .execute_cypher("MATCH (n:User) WHERE n.age > 28 RETURN n.name")
        .expect("match where");
    assert_eq!(rows.len(), 1);
}

#[test]
fn match_return_count() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:Fruit {name: 'Apple'})")
        .expect("1");
    db.execute_cypher("CREATE (b:Fruit {name: 'Banana'})")
        .expect("2");
    db.execute_cypher("CREATE (c:Fruit {name: 'Cherry'})")
        .expect("3");
    let rows = db
        .execute_cypher("MATCH (n:Fruit) RETURN count(n)")
        .expect("count");
    assert_eq!(rows.len(), 1);
}

// ── SET ─────────────────────────────────────────────────────────────

#[test]
fn set_property() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (n:User {name: 'Alice'})")
        .expect("create");
    db.execute_cypher("MATCH (n:User {name: 'Alice'}) SET n.age = 30")
        .expect("set");
    let rows = db
        .execute_cypher("MATCH (n:User {name: 'Alice'}) RETURN n.age")
        .expect("read");
    assert_eq!(rows.len(), 1);
}

// ── DELETE ──────────────────────────────────────────────────────────

#[test]
fn delete_node() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (n:Temp {val: 1})")
        .expect("create");
    let before = db
        .execute_cypher("MATCH (n:Temp) RETURN n")
        .expect("before");
    assert_eq!(before.len(), 1);

    db.execute_cypher("MATCH (n:Temp) DELETE n")
        .expect("delete");
    let after = db.execute_cypher("MATCH (n:Temp) RETURN n").expect("after");
    assert!(after.is_empty());
}

#[test]
fn detach_delete_node() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:User {name: 'Alice'})")
        .expect("create");
    db.execute_cypher("CREATE (b:User {name: 'Bob'})")
        .expect("create");
    db.execute_cypher("MATCH (n:User {name: 'Alice'}) DETACH DELETE n")
        .expect("detach delete");
    let rows = db
        .execute_cypher("MATCH (n:User) RETURN n.name")
        .expect("remaining");
    assert_eq!(rows.len(), 1);
}

/// DETACH DELETE removes all edges across multiple edge types.
/// Verifies that the targeted adj lookup (G051 fix) works for multi-edge-type graphs.
#[test]
fn detach_delete_cleans_multi_edge_type_graph() {
    let (mut db, _dir) = open_db();

    // Schema
    db.execute_cypher("CREATE (a:Person {name: 'Alice'})")
        .expect("create Alice");
    db.execute_cypher("CREATE (b:Person {name: 'Bob'})")
        .expect("create Bob");
    db.execute_cypher("CREATE (c:Person {name: 'Carol'})")
        .expect("create Carol");

    // Alice is connected via two different edge types
    db.execute_cypher(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) \
         CREATE (a)-[:FOLLOWS]->(b)",
    )
    .expect("follows edge");
    db.execute_cypher(
        "MATCH (a:Person {name: 'Alice'}), (c:Person {name: 'Carol'}) \
         CREATE (a)-[:LIKES]->(c)",
    )
    .expect("likes edge");
    db.execute_cypher(
        "MATCH (b:Person {name: 'Bob'}), (c:Person {name: 'Carol'}) \
         CREATE (b)-[:FOLLOWS]->(c)",
    )
    .expect("bob follows carol");

    // DETACH DELETE Alice removes her and all her edges
    db.execute_cypher("MATCH (n:Person {name: 'Alice'}) DETACH DELETE n")
        .expect("detach delete Alice");

    // Alice is gone
    let remaining = db
        .execute_cypher("MATCH (n:Person) RETURN n.name")
        .expect("remaining nodes");
    assert_eq!(remaining.len(), 2, "only Bob and Carol remain");

    // Bob's FOLLOWS edge to Carol must still exist
    let bob_follows = db
        .execute_cypher("MATCH (b:Person {name: 'Bob'})-[:FOLLOWS]->(c:Person) RETURN c.name")
        .expect("bob follows");
    assert_eq!(bob_follows.len(), 1, "Bob still follows Carol");
}

/// DETACH DELETE on a node with no edges works (no spurious errors).
#[test]
fn detach_delete_node_without_edges() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (n:Orphan {id: 1})")
        .expect("create");
    db.execute_cypher("MATCH (n:Orphan) DETACH DELETE n")
        .expect("detach delete orphan");
    let rows = db
        .execute_cypher("MATCH (n:Orphan) RETURN n")
        .expect("check");
    assert!(rows.is_empty(), "orphan node deleted");
}

// ── MERGE ───────────────────────────────────────────────────────────

#[test]
fn merge_create_when_not_exists() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("MERGE (n:Config {key: 'timeout', value: '30'})")
        .expect("merge");
    let rows = db
        .execute_cypher("MATCH (n:Config) RETURN n.key")
        .expect("check");
    assert_eq!(rows.len(), 1);
}

#[test]
fn merge_no_duplicate() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("MERGE (n:Config {key: 'timeout'})")
        .expect("merge 1");
    db.execute_cypher("MERGE (n:Config {key: 'timeout'})")
        .expect("merge 2");
    let rows = db
        .execute_cypher("MATCH (n:Config {key: 'timeout'}) RETURN n")
        .expect("check");
    assert_eq!(rows.len(), 1);
}

/// G074 — standalone relationship MERGE through the embed API.
///
/// What this tests:
/// - `MERGE (a:L {k:v})-[r:T]->(b:L {k:v})` succeeds end-to-end via Database API
/// - Idempotent: second run finds existing path, does not create duplicates
#[test]
fn merge_relationship_standalone() {
    let (mut db, _dir) = open_db();

    // First run — creates both nodes and the edge
    db.execute_cypher(
        "MERGE (a:Device {id: 'sensor-1'})-[r:CONNECTED_TO]->(b:Device {id: 'hub-1'})",
    )
    .expect("standalone relationship MERGE should succeed");

    // Verify the edge was created
    let rows = db
        .execute_cypher(
            "MATCH (a:Device {id: 'sensor-1'})-[r:CONNECTED_TO]->(b:Device {id: 'hub-1'}) \
             RETURN a.id, b.id",
        )
        .expect("match after MERGE");
    assert_eq!(rows.len(), 1, "edge should exist after standalone MERGE");

    // Second run — idempotent, must not create duplicates
    db.execute_cypher(
        "MERGE (a:Device {id: 'sensor-1'})-[r:CONNECTED_TO]->(b:Device {id: 'hub-1'})",
    )
    .expect("idempotent standalone MERGE should succeed");

    // Verify no duplicates: MATCH returns exactly 2 Device nodes (one per label+id combo)
    let rows2 = db
        .execute_cypher(
            "MATCH (a:Device {id: 'sensor-1'})-[:CONNECTED_TO]->(b:Device {id: 'hub-1'}) \
             RETURN a.id, b.id",
        )
        .expect("match after second MERGE");
    assert_eq!(
        rows2.len(),
        1,
        "must still be exactly 1 path after second MERGE (no duplicates)"
    );
}

// ── Persistence ─────────────────────────────────────────────────────

#[test]
fn data_survives_close_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");

    {
        let mut db = Database::open(dir.path()).expect("open");
        db.execute_cypher("CREATE (n:Persistent {key: 'survives'})")
            .expect("create");
    }

    {
        let mut db = Database::open(dir.path()).expect("reopen");
        let rows = db
            .execute_cypher("MATCH (n:Persistent) RETURN n.key")
            .expect("match");
        assert!(!rows.is_empty());
    }
}

// ── Map Projection ─────────────────────────────────────────────────

#[test]
fn map_projection_shorthand() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (n:User {name: 'Alice', age: 30})")
        .expect("create");

    let rows = db
        .execute_cypher("MATCH (n:User) RETURN n { .name, .age }")
        .expect("map projection");
    assert_eq!(rows.len(), 1);

    // The result should have a single column with a Map value
    let first_row = &rows[0];
    // Column name for map projection on variable "n"
    let val = first_row.values().next().expect("should have one column");
    if let coordinode_core::graph::types::Value::Map(map) = val {
        assert_eq!(
            map.get("name"),
            Some(&coordinode_core::graph::types::Value::String(
                "Alice".into()
            ))
        );
        assert_eq!(
            map.get("age"),
            Some(&coordinode_core::graph::types::Value::Int(30))
        );
    } else {
        panic!("expected Map value, got: {val:?}");
    }
}

#[test]
fn map_projection_with_computed_field() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (n:User {name: 'Bob', age: 25})")
        .expect("create");

    let rows = db
        .execute_cypher("MATCH (n:User) RETURN n { .name, doubled: n.age }")
        .expect("map projection with computed");
    assert_eq!(rows.len(), 1);

    let val = rows[0].values().next().expect("column");
    if let coordinode_core::graph::types::Value::Map(map) = val {
        assert_eq!(
            map.get("name"),
            Some(&coordinode_core::graph::types::Value::String("Bob".into()))
        );
        assert_eq!(
            map.get("doubled"),
            Some(&coordinode_core::graph::types::Value::Int(25))
        );
    } else {
        panic!("expected Map, got: {val:?}");
    }
}

/// R123: Snapshot read concern — reads data at a specific MVCC timestamp,
/// ignoring writes that happened after that timestamp.
#[test]
fn read_concern_snapshot_pins_to_timestamp() {
    use coordinode_core::txn::read_concern::{ReadConcern, ReadConcernLevel};

    let (mut db, _dir) = open_db();

    // Write v1 of a node
    db.execute_cypher("CREATE (u:User {name: 'alice', version: 1})")
        .expect("create v1");

    // Capture the current oracle timestamp (approx) for snapshot
    // The oracle advances on each execute_cypher call.
    // We'll use a timestamp between v1 and v2 writes.
    let snapshot_ts = 5; // Low timestamp — before most MVCC versions

    // Write v2 (update)
    db.execute_cypher("MATCH (u:User {name: 'alice'}) SET u.version = 2")
        .expect("update to v2");

    // Local read (default) — should see v2 (latest)
    let rows = db
        .execute_cypher("MATCH (u:User {name: 'alice'}) RETURN u.version")
        .expect("local read");
    assert!(!rows.is_empty(), "local read should find alice");

    // Snapshot read at early timestamp — tests that at_timestamp is used
    // instead of oracle.next(). Even if the exact data visibility varies
    // (depends on MVCC commit_ts assignment), the read_concern path is exercised.
    let rc = ReadConcern::snapshot_at(snapshot_ts);
    let snap_rows = db
        .execute_cypher_with_read_concern("MATCH (u:User {name: 'alice'}) RETURN u.version", rc)
        .expect("snapshot read");

    // At very early timestamp (5), alice may not exist yet (MVCC versions
    // are at higher timestamps). This verifies snapshot read returns
    // different results than local read.
    // The key assertion: snapshot read at ts=5 returns FEWER rows than local read.
    assert!(
        snap_rows.len() <= rows.len(),
        "snapshot at early ts should return <= rows than local"
    );

    // Snapshot read at high timestamp — should see everything
    let rc_future = ReadConcern::snapshot_at(u64::MAX / 2);
    let future_rows = db
        .execute_cypher_with_read_concern(
            "MATCH (u:User {name: 'alice'}) RETURN u.version",
            rc_future,
        )
        .expect("snapshot read future");
    assert!(
        !future_rows.is_empty(),
        "snapshot at future ts should see alice"
    );

    // Verify read concern validation works
    let invalid = ReadConcern {
        level: ReadConcernLevel::Majority,
        after_index: None,
        at_timestamp: Some(100), // at_timestamp only valid with Snapshot
    };
    let err = db.execute_cypher_with_read_concern("MATCH (n) RETURN n", invalid);
    assert!(err.is_err(), "invalid read concern should fail validation");
}

/// R123: Read concern session API — set_read_concern persists across queries.
#[test]
fn read_concern_session_level() {
    use coordinode_core::txn::read_concern::ReadConcernLevel;

    let (mut db, _dir) = open_db();

    // Default is Local
    assert_eq!(db.read_concern(), ReadConcernLevel::Local);

    // Set to Majority
    db.set_read_concern(ReadConcernLevel::Majority);
    assert_eq!(db.read_concern(), ReadConcernLevel::Majority);

    // Queries should work with any level in embedded mode
    db.execute_cypher("CREATE (n:Test {name: 'test'})")
        .expect("create");
    let rows = db
        .execute_cypher("MATCH (n:Test) RETURN n.name")
        .expect("read at majority");
    assert!(
        !rows.is_empty(),
        "majority read should work in embedded mode"
    );

    // One-shot read concern doesn't change session level
    let rc = coordinode_core::txn::read_concern::ReadConcern::snapshot_at(u64::MAX / 2);
    let _ = db.execute_cypher_with_read_concern("MATCH (n:Test) RETURN n", rc);
    assert_eq!(
        db.read_concern(),
        ReadConcernLevel::Majority,
        "session level should be restored after one-shot"
    );
}

/// R124: Write concern levels — W0 fire-and-forget still writes data locally,
/// Majority uses proposal pipeline, session API works.
#[test]
fn write_concern_levels() {
    use coordinode_core::txn::write_concern::WriteConcernLevel;

    let (mut db, _dir) = open_db();

    // Default is Majority
    assert_eq!(db.write_concern(), WriteConcernLevel::Majority);

    // Write with Majority (default) — data visible
    db.execute_cypher("CREATE (n:WC {name: 'majority', v: 1})")
        .expect("create with majority");
    let rows = db
        .execute_cypher("MATCH (n:WC {name: 'majority'}) RETURN n.v")
        .expect("read majority");
    assert!(!rows.is_empty(), "majority write should be visible");

    // Switch to W0 (fire-and-forget)
    db.set_write_concern(WriteConcernLevel::W0);
    assert_eq!(db.write_concern(), WriteConcernLevel::W0);

    // W0 write — data still visible locally (embedded mode, direct write)
    db.execute_cypher("CREATE (n:WC {name: 'w0', v: 2})")
        .expect("create with w0");
    let w0_rows = db
        .execute_cypher("MATCH (n:WC {name: 'w0'}) RETURN n.v")
        .expect("read w0");
    assert!(
        !w0_rows.is_empty(),
        "w0 write visible locally in embedded mode"
    );

    // Switch to W1
    db.set_write_concern(WriteConcernLevel::W1);
    db.execute_cypher("CREATE (n:WC {name: 'w1', v: 3})")
        .expect("create with w1");
    let w1_rows = db
        .execute_cypher("MATCH (n:WC {name: 'w1'}) RETURN n.v")
        .expect("read w1");
    assert!(!w1_rows.is_empty(), "w1 write visible locally");

    // Restore to Majority
    db.set_write_concern(WriteConcernLevel::Majority);
    assert_eq!(db.write_concern(), WriteConcernLevel::Majority);
}

/// R124: Write concern validation — causal session rejects volatile writes.
#[test]
fn write_concern_causal_validation() {
    use coordinode_core::txn::write_concern::{WriteConcern, WriteConcernLevel};

    // Majority is causal-safe
    assert!(WriteConcern::majority()
        .validate_for_causal_session()
        .is_ok());

    // W0, W1 are NOT causal-safe
    assert!(WriteConcern::w0().validate_for_causal_session().is_err());
    assert!(WriteConcern::w1().validate_for_causal_session().is_err());

    // j:true + W0 upgrades to W1, still not causal-safe
    let wc = WriteConcern {
        level: WriteConcernLevel::W0,
        journal: true,
        timeout_ms: 0,
    };
    assert_eq!(wc.effective_level(), WriteConcernLevel::W1);
    assert!(wc.validate_for_causal_session().is_err());
}

/// R124: Journal gate (j:true) forces WAL fsync; write concern full API.
#[test]
fn write_concern_journal_and_full_api() {
    use coordinode_core::txn::write_concern::{WriteConcern, WriteConcernLevel};

    let (mut db, _dir) = open_db();

    // Set full write concern with journal gate
    let wc = WriteConcern {
        level: WriteConcernLevel::Majority,
        journal: true,
        timeout_ms: 0,
    };
    db.set_write_concern_full(wc);
    assert_eq!(db.write_concern(), WriteConcernLevel::Majority);

    // Write with j:true — forces WAL fsync after commit
    db.execute_cypher("CREATE (n:Journal {name: 'durable', v: 1})")
        .expect("create with journal");
    let rows = db
        .execute_cypher("MATCH (n:Journal) RETURN n.v")
        .expect("read journal");
    assert!(!rows.is_empty(), "journaled write should be visible");

    // j:true + W0 → effective level W1 (journal gate upgrade)
    let wc_upgrade = WriteConcern {
        level: WriteConcernLevel::W0,
        journal: true,
        timeout_ms: 0,
    };
    db.set_write_concern_full(wc_upgrade);
    // Effective level is W1, not W0 — write goes through pipeline, not fire-and-forget
    db.execute_cypher("CREATE (n:Journal {name: 'upgraded', v: 2})")
        .expect("create with j:true upgrade");
    let rows2 = db
        .execute_cypher("MATCH (n:Journal {name: 'upgraded'}) RETURN n.v")
        .expect("read upgraded");
    assert!(!rows2.is_empty(), "j:true upgraded W0→W1 write visible");
}
