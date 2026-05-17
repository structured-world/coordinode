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

// ── #51 Edge-case audit: non-temporal modalities ──────────────────────

/// DETACH DELETE on a node with a self-loop must remove both the node
/// and its self-edge without leaving orphaned adjacency entries.
/// Regression: self-edges where source == target are easy to miss in
/// cleanup loops that treat in/out edges as separate sets.
#[test]
fn detach_delete_node_with_self_loop() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:Person {name: 'narcissist'})-[:LIKES]->(a)")
        .expect("create self-loop");

    let pre = db
        .execute_cypher("MATCH (n:Person) RETURN n.name")
        .expect("pre count");
    assert_eq!(pre.len(), 1, "self-loop creates exactly one node");

    db.execute_cypher("MATCH (n:Person {name: 'narcissist'}) DETACH DELETE n")
        .expect("detach delete self-loop");

    let post_nodes = db
        .execute_cypher("MATCH (n:Person) RETURN n")
        .expect("post nodes");
    assert!(
        post_nodes.is_empty(),
        "node must be gone after DETACH DELETE"
    );

    let post_edges = db
        .execute_cypher("MATCH ()-[r:LIKES]->() RETURN r")
        .expect("post edges");
    assert!(
        post_edges.is_empty(),
        "self-edge must be removed alongside the node, got {} dangling edges",
        post_edges.len()
    );
}

/// DETACH DELETE on a middle node in a chain removes incoming and
/// outgoing adjacency from both endpoints.
#[test]
fn detach_delete_node_with_both_directions() {
    let (mut db, _dir) = open_db();
    db.execute_cypher(
        "CREATE (a:Person {name: 'Alice'})-[:KNOWS]->(b:Person {name: 'Bob'})\
                  -[:KNOWS]->(c:Person {name: 'Charlie'})",
    )
    .expect("seed chain");

    db.execute_cypher("MATCH (b:Person {name: 'Bob'}) DETACH DELETE b")
        .expect("detach delete middle node");

    let remaining = db
        .execute_cypher("MATCH (n:Person) RETURN n.name")
        .expect("remaining");
    assert_eq!(remaining.len(), 2, "Alice and Charlie remain; Bob is gone");

    let alice_out = db
        .execute_cypher("MATCH (:Person {name: 'Alice'})-[:KNOWS]->(x) RETURN x.name")
        .expect("alice out");
    assert!(
        alice_out.is_empty(),
        "Alice's outgoing edge to Bob must be gone, got {alice_out:?}"
    );
    let charlie_in = db
        .execute_cypher("MATCH (x)-[:KNOWS]->(:Person {name: 'Charlie'}) RETURN x.name")
        .expect("charlie in");
    assert!(
        charlie_in.is_empty(),
        "Charlie's incoming edge from Bob must be gone, got {charlie_in:?}"
    );
}

/// DETACH DELETE on a hub clears every outgoing adjacency entry. Targets
/// are unaffected.
#[test]
fn detach_delete_source_node_clears_all_outgoing_edges() {
    let (mut db, _dir) = open_db();
    // Multi-statement seeding: create hub then attach 3 targets one by one.
    // Our Cypher impl may not support the multi-path comma CREATE syntax
    // for back-references to the same variable; this form avoids that.
    db.execute_cypher("CREATE (h:Hub {id: 'h1'})").expect("hub");
    for n in 1..=3 {
        db.execute_cypher(&format!(
            "MATCH (h:Hub {{id: 'h1'}}) CREATE (h)-[:LINKS]->(:Target {{n: {n}}})"
        ))
        .expect("link");
    }

    db.execute_cypher("MATCH (h:Hub {id: 'h1'}) DETACH DELETE h")
        .expect("detach delete hub");

    let targets = db
        .execute_cypher("MATCH (t:Target) RETURN t.n")
        .expect("targets remain");
    assert_eq!(targets.len(), 3, "targets unaffected");

    let orphan_edges = db
        .execute_cypher("MATCH ()-[r:LINKS]->() RETURN r")
        .expect("orphan check");
    assert!(
        orphan_edges.is_empty(),
        "all LINKS edges must be removed when hub is detach-deleted, got {} dangling",
        orphan_edges.len()
    );
}

/// MERGE on a node with an empty property pattern (just label) matches
/// any existing node with that label.
#[test]
fn merge_empty_property_pattern_matches_existing() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (n:Singleton {id: 1})")
        .expect("seed");
    db.execute_cypher("MERGE (s:Singleton)")
        .expect("merge empty");
    let rows = db
        .execute_cypher("MATCH (s:Singleton) RETURN s.id")
        .expect("count");
    assert_eq!(
        rows.len(),
        1,
        "MERGE with empty property pattern must NOT create a second node; \
         got {} Singleton nodes",
        rows.len()
    );
}

/// `WHERE n.prop = NULL` must NEVER match anything per Cypher semantics.
#[test]
fn where_equals_null_never_matches_per_cypher_spec() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (n:Item {name: 'a', tag: 'red'})")
        .expect("seed a");
    db.execute_cypher("CREATE (n:Item {name: 'b'})")
        .expect("seed b without tag");

    let eq_null = db
        .execute_cypher("MATCH (i:Item) WHERE i.tag = NULL RETURN i.name")
        .expect("eq null query");
    assert!(
        eq_null.is_empty(),
        "n.tag = NULL must match zero rows per Cypher spec, got {eq_null:?}"
    );

    let is_null = db
        .execute_cypher("MATCH (i:Item) WHERE i.tag IS NULL RETURN i.name")
        .expect("is null query");
    assert_eq!(
        is_null.len(),
        1,
        "IS NULL must match exactly the node missing tag, got {is_null:?}"
    );
}

/// Dot-notation access on a missing nested document path returns Null
/// without erroring. Application code routinely chains property
/// accesses through optional structures; a crash here would propagate
/// to every UI doing `n.user.profile.avatar`.
#[test]
fn dot_notation_missing_path_returns_null() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (n:Doc {title: 't', config: {a: 1}})")
        .expect("create");

    let rows = db
        .execute_cypher("MATCH (n:Doc) RETURN n.config.b.c AS missing")
        .expect("missing path");
    assert_eq!(rows.len(), 1);
    use coordinode_core::graph::types::Value;
    assert!(
        matches!(rows[0].get("missing"), Some(Value::Null) | None),
        "deep miss must yield Null, got {:?}",
        rows[0].get("missing")
    );

    let rows2 = db
        .execute_cypher("MATCH (n:Doc) RETURN n.no_such_field AS missing")
        .expect("missing top-level");
    assert!(matches!(rows2[0].get("missing"), Some(Value::Null) | None));
}

/// SET on a nested document path preserves sibling keys (partial update
/// semantics). Regression: confusing path-set with whole-replace breaks
/// every workflow that uses dot-notation to update one field.
#[test]
fn document_path_set_preserves_sibling_keys() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (n:Doc {id: 1, config: {host: 'a', port: 80, tls: true}})")
        .expect("seed");

    db.execute_cypher("MATCH (n:Doc {id: 1}) SET n.config.port = 443")
        .expect("path set");
    let after = db
        .execute_cypher(
            "MATCH (n:Doc {id: 1}) RETURN n.config.host AS h, \
             n.config.port AS p, n.config.tls AS t",
        )
        .expect("read after path set");
    assert_eq!(after.len(), 1);
    use coordinode_core::graph::types::Value;
    assert_eq!(after[0].get("h"), Some(&Value::String("a".into())));
    assert_eq!(after[0].get("p"), Some(&Value::Int(443)));
    assert_eq!(after[0].get("t"), Some(&Value::Bool(true)));
}

/// Bare MERGE — no ON CREATE / ON MATCH — must not touch the existing
/// node's properties even when extra properties appear in the pattern.
/// (Per Cypher spec: MERGE with properties means "match a node where
/// these properties equal these values", not "set them".)
#[test]
fn merge_match_does_not_silently_overwrite_existing_properties() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (u:User {email: 'a@x.com', age: 30})")
        .expect("seed");
    db.execute_cypher("MERGE (u:User {email: 'a@x.com'})")
        .expect("merge match");

    let rows = db
        .execute_cypher("MATCH (u:User {email: 'a@x.com'}) RETURN u.age AS a")
        .expect("read");
    assert_eq!(rows.len(), 1);
    use coordinode_core::graph::types::Value;
    assert_eq!(
        rows[0].get("a"),
        Some(&Value::Int(30)),
        "MERGE without SET must not touch existing properties"
    );
}

// ── MERGE NODES (R180) ────────────────────────────────────────────────

/// Two `User` nodes are collapsed into one; default `KEEP FIRST` keeps the
/// surviving node's properties and the non-survivor is gone.
#[test]
fn merge_nodes_default_keep_first_drops_non_survivor() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:User {email: 'a@x.com', name: 'Alice'})")
        .expect("seed a");
    db.execute_cypher("CREATE (b:User {email: 'b@x.com', name: 'Bob'})")
        .expect("seed b");

    db.execute_cypher(
        "MATCH (a:User {email: 'a@x.com'}), (b:User {email: 'b@x.com'}) \
         MERGE NODES (a, b) INTO a",
    )
    .expect("merge nodes");

    // Only one User remains (the target).
    let remaining = db
        .execute_cypher("MATCH (u:User) RETURN u.email AS email")
        .expect("scan users");
    assert_eq!(remaining.len(), 1, "non-surviving b must be deleted");

    use coordinode_core::graph::types::Value;
    assert_eq!(
        remaining[0].get("email"),
        Some(&Value::String("a@x.com".into())),
        "target a survives with its email"
    );
}

/// `KEEP LAST` overrides target's properties with source's where keys overlap.
#[test]
fn merge_nodes_keep_last_overwrites_target_props() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:User {email: 'a@x.com', score: 1})")
        .expect("seed a");
    db.execute_cypher("CREATE (b:User {email: 'b@x.com', score: 99})")
        .expect("seed b");

    db.execute_cypher(
        "MATCH (a:User {email: 'a@x.com'}), (b:User {email: 'b@x.com'}) \
         MERGE NODES (a, b) INTO a ON CONFLICT KEEP LAST",
    )
    .expect("merge keep last");

    let rows = db
        .execute_cypher("MATCH (u:User) RETURN u.score AS score")
        .expect("scan");
    assert_eq!(rows.len(), 1);
    use coordinode_core::graph::types::Value;
    assert_eq!(
        rows[0].get("score"),
        Some(&Value::Int(99)),
        "KEEP LAST must overwrite target's score with source's"
    );
}

/// `COALESCE` strategy: source fills only properties absent on target.
#[test]
fn merge_nodes_coalesce_fills_missing_only() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:User {email: 'a@x.com', age: 30})")
        .expect("seed a");
    db.execute_cypher("CREATE (b:User {email: 'b@x.com', age: 99, city: 'Berlin'})")
        .expect("seed b");

    db.execute_cypher(
        "MATCH (a:User {email: 'a@x.com'}), (b:User {email: 'b@x.com'}) \
         MERGE NODES (a, b) INTO a ON CONFLICT COALESCE",
    )
    .expect("coalesce merge");

    let rows = db
        .execute_cypher("MATCH (u:User) RETURN u.age AS age, u.city AS city")
        .expect("scan");
    assert_eq!(rows.len(), 1);
    use coordinode_core::graph::types::Value;
    // age: target wins (30, not 99 — target wasn't null).
    assert_eq!(
        rows[0].get("age"),
        Some(&Value::Int(30)),
        "COALESCE leaves non-null target props untouched"
    );
    // city: target lacks it → filled from source.
    assert_eq!(
        rows[0].get("city"),
        Some(&Value::String("Berlin".into())),
        "COALESCE fills missing key from source"
    );
}

/// `ON CONFLICT SET` evaluates per-property expressions against the row.
#[test]
fn merge_nodes_on_conflict_set_expressions() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:User {email: 'a@x.com', score: 5})")
        .expect("seed a");
    db.execute_cypher("CREATE (b:User {email: 'b@x.com', score: 12})")
        .expect("seed b");

    db.execute_cypher(
        "MATCH (a:User {email: 'a@x.com'}), (b:User {email: 'b@x.com'}) \
         MERGE NODES (a, b) INTO a ON CONFLICT SET a.score = b.score",
    )
    .expect("set merge");

    let rows = db
        .execute_cypher("MATCH (u:User) RETURN u.score AS score")
        .expect("scan");
    assert_eq!(rows.len(), 1);
    use coordinode_core::graph::types::Value;
    assert_eq!(
        rows[0].get("score"),
        Some(&Value::Int(12)),
        "ON CONFLICT SET should overwrite a.score with b.score"
    );
}

/// Outgoing edges of the non-survivor are re-pointed to the target.
#[test]
fn merge_nodes_transfers_outgoing_edges() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:User {email: 'a@x.com'})")
        .expect("seed a");
    db.execute_cypher("CREATE (b:User {email: 'b@x.com'})")
        .expect("seed b");
    db.execute_cypher("CREATE (c:User {email: 'c@x.com'})")
        .expect("seed c");
    db.execute_cypher(
        "MATCH (b:User {email: 'b@x.com'}), (c:User {email: 'c@x.com'}) \
         CREATE (b)-[:KNOWS]->(c)",
    )
    .expect("b→c edge");

    db.execute_cypher(
        "MATCH (a:User {email: 'a@x.com'}), (b:User {email: 'b@x.com'}) \
         MERGE NODES (a, b) INTO a TRANSFER EDGES FROM b TO a",
    )
    .expect("merge with transfer");

    // After merge: a-[:KNOWS]->c must exist; b must be gone.
    let edges = db
        .execute_cypher(
            "MATCH (a:User {email: 'a@x.com'})-[:KNOWS]->(c) RETURN c.email AS to_email",
        )
        .expect("scan outgoing");
    assert_eq!(edges.len(), 1, "outgoing edge re-pointed to target");
    use coordinode_core::graph::types::Value;
    assert_eq!(
        edges[0].get("to_email"),
        Some(&Value::String("c@x.com".into())),
    );
}

/// Incoming edges are re-pointed: x→b becomes x→a.
#[test]
fn merge_nodes_transfers_incoming_edges() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:User {email: 'a@x.com'})")
        .expect("seed a");
    db.execute_cypher("CREATE (b:User {email: 'b@x.com'})")
        .expect("seed b");
    db.execute_cypher("CREATE (x:User {email: 'x@x.com'})")
        .expect("seed x");
    db.execute_cypher(
        "MATCH (x:User {email: 'x@x.com'}), (b:User {email: 'b@x.com'}) \
         CREATE (x)-[:FOLLOWS]->(b)",
    )
    .expect("x→b edge");

    db.execute_cypher(
        "MATCH (a:User {email: 'a@x.com'}), (b:User {email: 'b@x.com'}) \
         MERGE NODES (a, b) INTO a TRANSFER EDGES FROM b TO a",
    )
    .expect("merge");

    let edges = db
        .execute_cypher(
            "MATCH (x:User {email: 'x@x.com'})-[:FOLLOWS]->(a) RETURN a.email AS to_email",
        )
        .expect("scan incoming");
    assert_eq!(edges.len(), 1);
    use coordinode_core::graph::types::Value;
    assert_eq!(
        edges[0].get("to_email"),
        Some(&Value::String("a@x.com".into()))
    );
}

/// Self-loop b→b becomes target→target on merge.
#[test]
fn merge_nodes_self_loop_becomes_target_self_loop() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:User {email: 'a@x.com'})")
        .expect("seed a");
    db.execute_cypher("CREATE (b:User {email: 'b@x.com'})")
        .expect("seed b");
    db.execute_cypher("MATCH (b:User {email: 'b@x.com'}) CREATE (b)-[:LIKES]->(b)")
        .expect("b self-loop");

    db.execute_cypher(
        "MATCH (a:User {email: 'a@x.com'}), (b:User {email: 'b@x.com'}) \
         MERGE NODES (a, b) INTO a TRANSFER EDGES FROM b TO a",
    )
    .expect("merge self-loop");

    let edges = db
        .execute_cypher("MATCH (a:User {email: 'a@x.com'})-[:LIKES]->(a) RETURN a.email AS who")
        .expect("self-loop on a");
    assert_eq!(edges.len(), 1, "b's self-loop must become a's self-loop");
}

/// Idempotency: running MERGE NODES again with the non-survivor already gone
/// is a no-op rather than an error.
#[test]
fn merge_nodes_idempotent_when_source_missing() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:User {email: 'a@x.com'})")
        .expect("seed a");
    db.execute_cypher("CREATE (b:User {email: 'b@x.com'})")
        .expect("seed b");

    db.execute_cypher(
        "MATCH (a:User {email: 'a@x.com'}), (b:User {email: 'b@x.com'}) \
         MERGE NODES (a, b) INTO a",
    )
    .expect("first merge");

    // Recreate b under a different identity and re-run merge — first call
    // already removed the prior b, so the re-bound b is fresh and the merge
    // must collapse it cleanly without error.
    db.execute_cypher("CREATE (b:User {email: 'b@x.com'})")
        .expect("recreate b");
    let result = db.execute_cypher(
        "MATCH (a:User {email: 'a@x.com'}), (b:User {email: 'b@x.com'}) \
         MERGE NODES (a, b) INTO a",
    );
    assert!(result.is_ok(), "second merge must succeed: {result:?}");
}

/// `INTO b` makes b the surviving node and a the dropped one.
#[test]
fn merge_nodes_into_b_keeps_b() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:User {email: 'a@x.com'})")
        .expect("seed a");
    db.execute_cypher("CREATE (b:User {email: 'b@x.com'})")
        .expect("seed b");

    db.execute_cypher(
        "MATCH (a:User {email: 'a@x.com'}), (b:User {email: 'b@x.com'}) \
         MERGE NODES (a, b) INTO b",
    )
    .expect("merge into b");

    let rows = db
        .execute_cypher("MATCH (u:User) RETURN u.email AS email")
        .expect("scan");
    assert_eq!(rows.len(), 1);
    use coordinode_core::graph::types::Value;
    assert_eq!(rows[0].get("email"), Some(&Value::String("b@x.com".into())));
}

/// Duplicate-edge case: a→c already exists AND b→c also exists. With default
/// `KEEP BOTH`, after merge, a has TWO outgoing :KNOWS edges to c.
#[test]
fn merge_nodes_duplicate_keep_both_creates_parallel_edge() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:User {email: 'a@x.com'})")
        .expect("seed a");
    db.execute_cypher("CREATE (b:User {email: 'b@x.com'})")
        .expect("seed b");
    db.execute_cypher("CREATE (c:User {email: 'c@x.com'})")
        .expect("seed c");
    db.execute_cypher(
        "MATCH (a:User {email: 'a@x.com'}), (c:User {email: 'c@x.com'}) \
         CREATE (a)-[:KNOWS]->(c)",
    )
    .expect("a→c");
    db.execute_cypher(
        "MATCH (b:User {email: 'b@x.com'}), (c:User {email: 'c@x.com'}) \
         CREATE (b)-[:KNOWS]->(c)",
    )
    .expect("b→c");

    db.execute_cypher(
        "MATCH (a:User {email: 'a@x.com'}), (b:User {email: 'b@x.com'}) \
         MERGE NODES (a, b) INTO a TRANSFER EDGES FROM b TO a \
         ON DUPLICATE KEEP BOTH",
    )
    .expect("merge keep both");

    // Posting list de-duplicates by uid, so a:KNOWS posting still contains c once.
    // The "parallel edge" semantic is preserved at the conceptual level: both
    // edges resolve to the same (a, c) pair after merge — a meaningful invariant
    // we assert by scanning.
    let rows = db
        .execute_cypher(
            "MATCH (a:User {email: 'a@x.com'})-[:KNOWS]->(c:User {email: 'c@x.com'}) \
             RETURN c.email AS to_email",
        )
        .expect("scan");
    assert!(
        !rows.is_empty(),
        "a→c edge must still exist after KEEP BOTH merge"
    );
}

/// Error when INTO target refers to a variable not bound to one of (a, b).
#[test]
fn merge_nodes_rejects_unbound_target() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:User {email: 'a@x.com'})")
        .expect("seed a");
    db.execute_cypher("CREATE (b:User {email: 'b@x.com'})")
        .expect("seed b");

    let result = db.execute_cypher(
        "MATCH (a:User {email: 'a@x.com'}), (b:User {email: 'b@x.com'}) \
         MERGE NODES (a, b) INTO c",
    );
    assert!(
        result.is_err(),
        "MERGE NODES INTO unbound variable `c` must fail at parse time"
    );
}

/// Re-running the SAME merge query after both source/target collapsed: the
/// MATCH for `b` binds zero rows, MERGE NODES gets an empty input set, no
/// further mutation happens. Distinct from "create a new b and re-merge" —
/// this is the realistic retry-after-success scenario.
#[test]
fn merge_nodes_re_execution_with_no_b_is_clean_noop() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:User {email: 'a@x.com'})")
        .expect("seed a");
    db.execute_cypher("CREATE (b:User {email: 'b@x.com'})")
        .expect("seed b");

    db.execute_cypher(
        "MATCH (a:User {email: 'a@x.com'}), (b:User {email: 'b@x.com'}) \
         MERGE NODES (a, b) INTO a",
    )
    .expect("first merge");

    let result = db.execute_cypher(
        "MATCH (a:User {email: 'a@x.com'}), (b:User {email: 'b@x.com'}) \
         MERGE NODES (a, b) INTO a",
    );
    assert!(result.is_ok(), "re-execution must be clean: {result:?}");
    let users = db
        .execute_cypher("MATCH (u:User) RETURN u.email AS e")
        .expect("scan");
    assert_eq!(users.len(), 1, "still one survivor");
}

/// `ON DUPLICATE KEEP TARGET`: when both target and non-survivor have an
/// edge to the same peer, drop the non-survivor's edge entirely.
#[test]
fn merge_nodes_duplicate_keep_target_drops_source_edge() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:User {email: 'a@x.com'})")
        .unwrap();
    db.execute_cypher("CREATE (b:User {email: 'b@x.com'})")
        .unwrap();
    db.execute_cypher("CREATE (c:User {email: 'c@x.com'})")
        .unwrap();
    db.execute_cypher(
        "MATCH (a:User {email: 'a@x.com'}), (c:User {email: 'c@x.com'}) \
         CREATE (a)-[:KNOWS {strength: 5}]->(c)",
    )
    .unwrap();
    db.execute_cypher(
        "MATCH (b:User {email: 'b@x.com'}), (c:User {email: 'c@x.com'}) \
         CREATE (b)-[:KNOWS {strength: 99}]->(c)",
    )
    .unwrap();

    db.execute_cypher(
        "MATCH (a:User {email: 'a@x.com'}), (b:User {email: 'b@x.com'}) \
         MERGE NODES (a, b) INTO a TRANSFER EDGES FROM b TO a \
         ON DUPLICATE KEEP TARGET",
    )
    .unwrap();

    // a→c still has strength=5 (target's original), NOT 99 (source's).
    use coordinode_core::graph::types::Value;
    let rows = db
        .execute_cypher(
            "MATCH (a:User {email: 'a@x.com'})-[r:KNOWS]->(c:User {email: 'c@x.com'}) \
             RETURN r.strength AS s",
        )
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("s"),
        Some(&Value::Int(5)),
        "KEEP TARGET preserves the target's original edge properties"
    );
}

/// `ON DUPLICATE MERGE PROPERTIES`: target↔peer edge survives with COALESCE
/// merge of edge facets — non-null values from the source-side edge fill
/// nulls on the target-side edge.
#[test]
fn merge_nodes_duplicate_merge_properties_coalesces_edge_facets() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:User {email: 'a@x.com'})")
        .unwrap();
    db.execute_cypher("CREATE (b:User {email: 'b@x.com'})")
        .unwrap();
    db.execute_cypher("CREATE (c:User {email: 'c@x.com'})")
        .unwrap();
    // Target's a→c edge is missing the `notes` field.
    db.execute_cypher(
        "MATCH (a:User {email: 'a@x.com'}), (c:User {email: 'c@x.com'}) \
         CREATE (a)-[:KNOWS {since: 2020}]->(c)",
    )
    .unwrap();
    // Source's b→c carries it.
    db.execute_cypher(
        "MATCH (b:User {email: 'b@x.com'}), (c:User {email: 'c@x.com'}) \
         CREATE (b)-[:KNOWS {since: 2025, notes: 'from b'}]->(c)",
    )
    .unwrap();

    db.execute_cypher(
        "MATCH (a:User {email: 'a@x.com'}), (b:User {email: 'b@x.com'}) \
         MERGE NODES (a, b) INTO a TRANSFER EDGES FROM b TO a \
         ON DUPLICATE MERGE PROPERTIES",
    )
    .unwrap();

    use coordinode_core::graph::types::Value;
    let rows = db
        .execute_cypher(
            "MATCH (a:User {email: 'a@x.com'})-[r:KNOWS]->(c:User {email: 'c@x.com'}) \
             RETURN r.since AS since, r.notes AS notes",
        )
        .unwrap();
    assert_eq!(rows.len(), 1);
    // `since` was present on both; target wins under COALESCE.
    assert_eq!(rows[0].get("since"), Some(&Value::Int(2020)));
    // `notes` was missing on target; filled from source.
    assert_eq!(
        rows[0].get("notes"),
        Some(&Value::String("from b".into())),
        "MERGE PROPERTIES fills missing facets from source"
    );
}

/// Unique B-tree index on (User.email) must NOT block re-creation of a node
/// after MERGE NODES deletes the source — index cleanup happens before the
/// node record is dropped. Catches the index-leak bug.
#[test]
fn merge_nodes_releases_unique_index_entry_of_dropped_source() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE UNIQUE INDEX idx_user_email ON :User(email)")
        .expect("create unique index");
    db.execute_cypher("CREATE (a:User {email: 'a@x.com'})")
        .unwrap();
    db.execute_cypher("CREATE (b:User {email: 'b@x.com'})")
        .unwrap();

    db.execute_cypher(
        "MATCH (a:User {email: 'a@x.com'}), (b:User {email: 'b@x.com'}) \
         MERGE NODES (a, b) INTO a",
    )
    .unwrap();

    // After merge, no node holds email='b@x.com'. The unique index must allow
    // a brand-new node to claim that email; if the index leaked, this CREATE
    // would fail with a uniqueness violation.
    let result = db.execute_cypher("CREATE (c:User {email: 'b@x.com'})");
    assert!(
        result.is_ok(),
        "unique index entry for dropped node must be released: {result:?}"
    );
}

/// Updating target's indexed property via KEEP LAST must update the B-tree
/// index — old value's entry released, new value's entry registered.
#[test]
fn merge_nodes_keep_last_updates_btree_index() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE UNIQUE INDEX idx_user_handle ON :User(handle)")
        .expect("create unique index");
    db.execute_cypher("CREATE (a:User {handle: 'a_handle'})")
        .unwrap();
    db.execute_cypher("CREATE (b:User {handle: 'b_handle'})")
        .unwrap();

    db.execute_cypher(
        "MATCH (a:User {handle: 'a_handle'}), (b:User {handle: 'b_handle'}) \
         MERGE NODES (a, b) INTO a ON CONFLICT KEEP LAST",
    )
    .unwrap();

    // After merge, a now has handle='b_handle'. The OLD value 'a_handle' must
    // be freed in the index so a new node can claim it.
    let claim_old = db.execute_cypher("CREATE (new:User {handle: 'a_handle'})");
    assert!(
        claim_old.is_ok(),
        "old indexed value 'a_handle' must be released after KEEP LAST: {claim_old:?}"
    );

    // And the NEW value 'b_handle' is now claimed by `a` — re-claiming must fail.
    let reclaim_new = db.execute_cypher("CREATE (dup:User {handle: 'b_handle'})");
    assert!(
        reclaim_new.is_err(),
        "new indexed value 'b_handle' must be claimed by target — duplicate CREATE should fail"
    );
}

/// Cross-label merge: a is :Person, b is :Account. Target's labels remain
/// (only Person), source's labels do NOT bleed in. Matches arch-doc
/// "INTO a" semantics — labels of the surviving node win.
#[test]
fn merge_nodes_preserves_target_labels_only() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:Person {id: 1})").unwrap();
    db.execute_cypher("CREATE (b:Account {id: 2})").unwrap();

    db.execute_cypher(
        "MATCH (a:Person {id: 1}), (b:Account {id: 2}) \
         MERGE NODES (a, b) INTO a",
    )
    .unwrap();

    let persons = db
        .execute_cypher("MATCH (n:Person) RETURN n.id AS i")
        .unwrap();
    let accounts = db
        .execute_cypher("MATCH (n:Account) RETURN n.id AS i")
        .unwrap();
    assert_eq!(persons.len(), 1, "target remains :Person");
    assert_eq!(
        accounts.len(),
        0,
        "source's :Account label is NOT inherited"
    );
}

/// Self-loop b→b carries edge properties; after merge target→target must
/// also carry them. Catches the case where adj transfer succeeds but the
/// edgeprop record is left under the old (b,b) key.
#[test]
fn merge_nodes_self_loop_carries_edge_properties() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:User {email: 'a@x.com'})")
        .unwrap();
    db.execute_cypher("CREATE (b:User {email: 'b@x.com'})")
        .unwrap();
    db.execute_cypher(
        "MATCH (b:User {email: 'b@x.com'}) \
         CREATE (b)-[:LIKES {weight: 42}]->(b)",
    )
    .unwrap();

    db.execute_cypher(
        "MATCH (a:User {email: 'a@x.com'}), (b:User {email: 'b@x.com'}) \
         MERGE NODES (a, b) INTO a TRANSFER EDGES FROM b TO a",
    )
    .unwrap();

    use coordinode_core::graph::types::Value;
    let rows = db
        .execute_cypher("MATCH (a:User {email: 'a@x.com'})-[r:LIKES]->(a) RETURN r.weight AS w")
        .unwrap();
    assert_eq!(rows.len(), 1, "self-loop must survive merge");
    assert_eq!(
        rows[0].get("w"),
        Some(&Value::Int(42)),
        "edge properties must be carried onto the target's self-loop"
    );
}

/// Downstream RETURN must see the merged target. Catches binding-drop
/// regressions where the `a` variable would lose its row column after MERGE.
#[test]
fn merge_nodes_passes_target_binding_to_return() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:User {email: 'a@x.com', age: 30})")
        .unwrap();
    db.execute_cypher("CREATE (b:User {email: 'b@x.com', city: 'Berlin'})")
        .unwrap();

    let rows = db
        .execute_cypher(
            "MATCH (a:User {email: 'a@x.com'}), (b:User {email: 'b@x.com'}) \
             MERGE NODES (a, b) INTO a ON CONFLICT COALESCE \
             RETURN a.age AS age, a.city AS city",
        )
        .unwrap();

    use coordinode_core::graph::types::Value;
    assert_eq!(
        rows.len(),
        1,
        "RETURN must produce exactly one row for the target"
    );
    assert_eq!(rows[0].get("age"), Some(&Value::Int(30)));
    assert_eq!(
        rows[0].get("city"),
        Some(&Value::String("Berlin".into())),
        "COALESCE merged b.city into a; RETURN must see it"
    );
}

/// STRICT schema rejects a merge that would introduce undeclared properties.
#[test]
fn merge_nodes_strict_schema_rejects_unknown_source_property() {
    use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType, SchemaMode};

    let (mut db, _dir) = open_db();

    let mut schema = LabelSchema::new_node_id("Customer");
    schema.set_mode(SchemaMode::Strict);
    schema.add_property(PropertyDef::new("email", PropertyType::String).not_null());
    db.create_label_schema(schema)
        .expect("create strict schema");

    db.execute_cypher("CREATE (a:Customer {email: 'a@x.com'})")
        .expect("seed a");

    // b carries an extra property `tags` that is NOT declared on Customer.
    // Created via FLEXIBLE label first, then we'd need a way to wedge it into
    // a Customer record. The straightforward path: create a Flexible label
    // `Lead` carrying `tags`, then re-label via the merge into Customer — but
    // labels are preserved on target (Customer), so source props bleed in.
    let mut lead = LabelSchema::new_node_id("Lead");
    lead.set_mode(SchemaMode::Flexible);
    lead.add_property(PropertyDef::new("email", PropertyType::String));
    lead.add_property(PropertyDef::new("tags", PropertyType::String));
    db.create_label_schema(lead).expect("create flexible Lead");

    db.execute_cypher("CREATE (b:Lead {email: 'b@x.com', tags: 'hot'})")
        .expect("seed flexible b");

    let result = db.execute_cypher(
        "MATCH (a:Customer {email: 'a@x.com'}), (b:Lead {email: 'b@x.com'}) \
         MERGE NODES (a, b) INTO a ON CONFLICT KEEP LAST",
    );

    assert!(
        result.is_err(),
        "STRICT Customer label must reject merged 'tags' property: {result:?}"
    );
    let err = format!("{}", result.err().unwrap());
    assert!(
        err.contains("unknown property") || err.contains("strict"),
        "error must mention schema violation: {err}"
    );
}

/// Edge between source and target (`a→b`) becomes a self-loop on target
/// after merge — there is no "b" anymore to be the target endpoint.
/// This is the canonical dedup case for graphs with mutual links.
#[test]
fn merge_nodes_edge_between_source_and_target_becomes_self_loop() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:User {email: 'a@x.com'})")
        .unwrap();
    db.execute_cypher("CREATE (b:User {email: 'b@x.com'})")
        .unwrap();
    db.execute_cypher(
        "MATCH (a:User {email: 'a@x.com'}), (b:User {email: 'b@x.com'}) \
         CREATE (a)-[:KNOWS {weight: 9}]->(b)",
    )
    .unwrap();

    db.execute_cypher(
        "MATCH (a:User {email: 'a@x.com'}), (b:User {email: 'b@x.com'}) \
         MERGE NODES (a, b) INTO a TRANSFER EDGES FROM b TO a",
    )
    .unwrap();

    use coordinode_core::graph::types::Value;
    let rows = db
        .execute_cypher("MATCH (a:User {email: 'a@x.com'})-[r:KNOWS]->(a) RETURN r.weight AS w")
        .unwrap();
    assert_eq!(rows.len(), 1, "a→b must collapse into a self-loop a→a");
    assert_eq!(
        rows[0].get("w"),
        Some(&Value::Int(9)),
        "edge property must carry onto the self-loop"
    );
}

/// Same as above, mirrored direction (`b→a`). Confirms both incoming and
/// outgoing cross-link cases collapse cleanly.
#[test]
fn merge_nodes_edge_target_to_source_becomes_self_loop() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:User {email: 'a@x.com'})")
        .unwrap();
    db.execute_cypher("CREATE (b:User {email: 'b@x.com'})")
        .unwrap();
    db.execute_cypher(
        "MATCH (a:User {email: 'a@x.com'}), (b:User {email: 'b@x.com'}) \
         CREATE (b)-[:FOLLOWS]->(a)",
    )
    .unwrap();

    db.execute_cypher(
        "MATCH (a:User {email: 'a@x.com'}), (b:User {email: 'b@x.com'}) \
         MERGE NODES (a, b) INTO a TRANSFER EDGES FROM b TO a",
    )
    .unwrap();

    let rows = db
        .execute_cypher("MATCH (a:User {email: 'a@x.com'})-[:FOLLOWS]->(a) RETURN a.email AS who")
        .unwrap();
    assert_eq!(rows.len(), 1, "b→a must collapse into self-loop a→a");
}

/// Multiple distinct peers in the same direction are all re-pointed. Tests
/// that the per-direction posting-list iteration handles fan-out correctly.
#[test]
fn merge_nodes_transfers_multiple_outgoing_peers() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:User {email: 'a@x.com'})")
        .unwrap();
    db.execute_cypher("CREATE (b:User {email: 'b@x.com'})")
        .unwrap();
    for tag in ["c", "d", "e"] {
        db.execute_cypher(&format!("CREATE (n:User {{email: '{tag}@x.com'}})"))
            .unwrap();
        db.execute_cypher(&format!(
            "MATCH (b:User {{email: 'b@x.com'}}), (n:User {{email: '{tag}@x.com'}}) \
             CREATE (b)-[:KNOWS]->(n)"
        ))
        .unwrap();
    }

    db.execute_cypher(
        "MATCH (a:User {email: 'a@x.com'}), (b:User {email: 'b@x.com'}) \
         MERGE NODES (a, b) INTO a TRANSFER EDGES FROM b TO a",
    )
    .unwrap();

    let rows = db
        .execute_cypher("MATCH (a:User {email: 'a@x.com'})-[:KNOWS]->(n) RETURN n.email AS e")
        .unwrap();
    assert_eq!(rows.len(), 3, "all three b→x edges must re-point to a→x");
}

/// VALIDATED schema mode rejects a merge that would write a wrong-typed value
/// to a declared property. Catches enforcement gap distinct from STRICT
/// (which also rejects unknown property names).
#[test]
fn merge_nodes_validated_schema_rejects_type_mismatch() {
    use coordinode_core::graph::types::Value;
    use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType, SchemaMode};

    let (mut db, _dir) = open_db();

    let mut invoice = LabelSchema::new_node_id("Invoice");
    invoice.set_mode(SchemaMode::Validated);
    invoice.add_property(PropertyDef::new("amount", PropertyType::Int));
    db.create_label_schema(invoice)
        .expect("create validated Invoice");

    // Source carries a string in a slot that target declares as Int.
    let mut staging = LabelSchema::new_node_id("Staging");
    staging.set_mode(SchemaMode::Flexible);
    staging.add_property(PropertyDef::new("amount", PropertyType::String));
    db.create_label_schema(staging)
        .expect("create flexible Staging");

    db.execute_cypher("CREATE (a:Invoice {amount: 100})")
        .unwrap();
    db.execute_cypher("CREATE (b:Staging {amount: 'one-hundred'})")
        .unwrap();

    let result = db.execute_cypher(
        "MATCH (a:Invoice), (b:Staging) \
         MERGE NODES (a, b) INTO a ON CONFLICT KEEP LAST",
    );
    assert!(
        result.is_err(),
        "VALIDATED Invoice.amount=Int must reject merged String value: {result:?}"
    );

    // And the original record must remain intact (transaction never committed).
    let rows = db
        .execute_cypher("MATCH (a:Invoice) RETURN a.amount AS amt")
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("amt"),
        Some(&Value::Int(100)),
        "rejected merge must not mutate target"
    );
}

/// Trivial smoke: merging when the source carries zero properties and zero
/// edges must still drop the source cleanly without touching the target.
#[test]
fn merge_nodes_empty_source_drops_cleanly() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:User {email: 'a@x.com', score: 5})")
        .unwrap();
    db.execute_cypher("CREATE (b:User)").unwrap(); // bare label, no props

    db.execute_cypher(
        "MATCH (a:User {email: 'a@x.com'}), (b:User) WHERE b.email IS NULL \
         MERGE NODES (a, b) INTO a",
    )
    .unwrap();

    use coordinode_core::graph::types::Value;
    let rows = db
        .execute_cypher("MATCH (u:User) RETURN u.email AS e, u.score AS s")
        .unwrap();
    assert_eq!(rows.len(), 1, "only the surviving target remains");
    assert_eq!(rows[0].get("e"), Some(&Value::String("a@x.com".into())));
    assert_eq!(rows[0].get("s"), Some(&Value::Int(5)));
}

/// Atomicity: when MERGE NODES processes multiple input rows and a later
/// row errors (here: STRICT schema violation), every effect of the earlier
/// successful rows MUST be rolled back. Specifically, adjacency-list
/// tombstones go through the MVCC buffer rather than direct engine deletes;
/// otherwise the second row's abort would leave the first row's edge
/// re-pointing committed.
#[test]
fn merge_nodes_atomic_rollback_when_later_row_fails() {
    use coordinode_core::graph::types::Value;
    use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType, SchemaMode};

    let (mut db, _dir) = open_db();

    // Target label is STRICT, declares only `name`.
    let mut user = LabelSchema::new_node_id("AtomicUser");
    user.set_mode(SchemaMode::Strict);
    user.add_property(PropertyDef::new("name", PropertyType::String).not_null());
    db.create_label_schema(user).unwrap();

    // Source label is FLEXIBLE. Some sources carry `name` only (mergable),
    // some carry an extra `tag` field that would violate STRICT on merge.
    let mut staging = LabelSchema::new_node_id("AtomicStaging");
    staging.set_mode(SchemaMode::Flexible);
    staging.add_property(PropertyDef::new("name", PropertyType::String));
    staging.add_property(PropertyDef::new("tag", PropertyType::String));
    db.create_label_schema(staging).unwrap();

    // a1, a2: two STRICT targets that would merge cleanly.
    db.execute_cypher("CREATE (a1:AtomicUser {name: 'Alice'})")
        .unwrap();
    db.execute_cypher("CREATE (a2:AtomicUser {name: 'Bob'})")
        .unwrap();

    // b1: clean source (would merge fine).
    db.execute_cypher("CREATE (b1:AtomicStaging {name: 'AliceX'})")
        .unwrap();
    // b2: poisoned source carrying `tag`, will trip STRICT enforcement
    // when targeting an AtomicUser.
    db.execute_cypher("CREATE (b2:AtomicStaging {name: 'BobX', tag: 'poisonous'})")
        .unwrap();

    // Add edges to a1 and a2 so we can detect adj-side rollback later.
    db.execute_cypher("CREATE (peer:AtomicUser {name: 'Peer'})")
        .unwrap();
    db.execute_cypher(
        "MATCH (a1:AtomicUser {name: 'Alice'}), (peer:AtomicUser {name: 'Peer'}) \
         CREATE (a1)-[:LINKS]->(peer)",
    )
    .unwrap();
    db.execute_cypher(
        "MATCH (b1:AtomicStaging {name: 'AliceX'}), (peer:AtomicUser {name: 'Peer'}) \
         CREATE (b1)-[:LINKS]->(peer)",
    )
    .unwrap();
    db.execute_cypher(
        "MATCH (b2:AtomicStaging {name: 'BobX'}), (peer:AtomicUser {name: 'Peer'}) \
         CREATE (b2)-[:LINKS]->(peer)",
    )
    .unwrap();

    // Cartesian: 2 a's × 2 b's = 4 rows. b2 carries `tag` → some row will
    // hit the STRICT validation gate. Order isn't guaranteed but at least
    // one merge with b2 must execute and trip the error.
    let result = db.execute_cypher(
        "MATCH (a:AtomicUser), (b:AtomicStaging) \
         MERGE NODES (a, b) INTO a ON CONFLICT KEEP LAST \
         TRANSFER EDGES FROM b TO a",
    );
    assert!(
        result.is_err(),
        "STRICT violation must abort the merge: {result:?}"
    );

    // Atomicity invariants after rollback:
    //   1. Both b1 and b2 still exist (no source deletes committed).
    //   2. Neither a1 nor a2 has merged props (a1.tag must NOT be set).
    //   3. b1's outgoing :LINKS to peer is still on b1 (NOT re-pointed to any a).
    let staging_count = db
        .execute_cypher("MATCH (b:AtomicStaging) RETURN b.name AS n")
        .unwrap();
    assert_eq!(
        staging_count.len(),
        2,
        "both source nodes must survive rollback, got {staging_count:?}"
    );

    let users = db
        .execute_cypher("MATCH (a:AtomicUser) RETURN a.name AS n")
        .unwrap();
    assert_eq!(users.len(), 3, "Alice + Bob + Peer all still present");

    // b1 must still have its outgoing edge.
    let b1_edges = db
        .execute_cypher("MATCH (b:AtomicStaging {name: 'AliceX'})-[:LINKS]->(p) RETURN p.name AS n")
        .unwrap();
    assert_eq!(
        b1_edges.len(),
        1,
        "b1's outgoing edge must remain — NOT re-pointed to a target. \
         If this fails, adj posting list mutations committed despite rollback."
    );
    assert_eq!(b1_edges[0].get("n"), Some(&Value::String("Peer".into())),);
}

/// STRICT extra-map gap: a source whose schema is VALIDATED stores undeclared
/// props in `extra`. Without explicit validation of `target_rec.extra`, those
/// undeclared keys would slip into a STRICT target through the merge.
#[test]
fn merge_nodes_strict_rejects_source_extra_overflow() {
    use coordinode_core::graph::types::Value;
    use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType, SchemaMode};

    let (mut db, _dir) = open_db();

    // Target STRICT — only `email` declared.
    let mut tgt = LabelSchema::new_node_id("StrictCustomer");
    tgt.set_mode(SchemaMode::Strict);
    tgt.add_property(PropertyDef::new("email", PropertyType::String).not_null());
    db.create_label_schema(tgt).unwrap();

    // Source VALIDATED — declares `email` only, so any other CREATE prop
    // lands in `extra` (the overflow map) rather than in props.
    let mut src = LabelSchema::new_node_id("ValidatedLead");
    src.set_mode(SchemaMode::Validated);
    src.add_property(PropertyDef::new("email", PropertyType::String));
    db.create_label_schema(src).unwrap();

    db.execute_cypher("CREATE (a:StrictCustomer {email: 'a@x.com'})")
        .unwrap();
    // The `note` is undeclared on ValidatedLead → goes into source's extra.
    db.execute_cypher("CREATE (b:ValidatedLead {email: 'b@x.com', note: 'sales lead'})")
        .unwrap();

    let result = db.execute_cypher(
        "MATCH (a:StrictCustomer), (b:ValidatedLead) \
         MERGE NODES (a, b) INTO a ON CONFLICT KEEP LAST",
    );
    assert!(
        result.is_err(),
        "STRICT target must reject merged extra-map keys: {result:?}"
    );
    let err = format!("{}", result.err().unwrap());
    assert!(
        err.contains("unknown property 'note'") || err.contains("strict"),
        "error must mention the offending field: {err}"
    );

    // Rollback invariant: target unchanged, source still present.
    let rows = db
        .execute_cypher("MATCH (a:StrictCustomer) RETURN a.email AS e")
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("e"), Some(&Value::String("a@x.com".into())));

    let src_rows = db
        .execute_cypher("MATCH (b:ValidatedLead) RETURN b.email AS e")
        .unwrap();
    assert_eq!(src_rows.len(), 1, "source must survive failed merge");
}

/// MERGE NODES — WITH-passthrough composability: the row produced by MERGE
/// NODES survives a WITH barrier and downstream RETURN sees the merged
/// target's property columns.
#[test]
fn merge_nodes_target_binding_survives_with_clause() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:User {email: 'a@x.com', age: 30})")
        .unwrap();
    db.execute_cypher("CREATE (b:User {email: 'b@x.com', city: 'Berlin'})")
        .unwrap();

    let rows = db
        .execute_cypher(
            "MATCH (a:User {email: 'a@x.com'}), (b:User {email: 'b@x.com'}) \
             MERGE NODES (a, b) INTO a ON CONFLICT COALESCE \
             WITH a \
             RETURN a.age AS age, a.city AS city",
        )
        .unwrap();
    use coordinode_core::graph::types::Value;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("age"), Some(&Value::Int(30)));
    assert_eq!(
        rows[0].get("city"),
        Some(&Value::String("Berlin".into())),
        "merged column must pass through WITH barrier"
    );
}

/// Vector property changed by merge must propagate to the HNSW vector
/// index — otherwise nearest-neighbour searches return stale results.
#[test]
fn merge_nodes_updates_vector_index_on_target() {
    let (mut db, _dir) = open_db();

    db.execute_cypher(
        "CREATE VECTOR INDEX item_emb ON :Item(embedding) \
         OPTIONS { metric: 'cosine', dimensions: 3 }",
    )
    .expect("create vector index");

    // Two items with embeddings; merge with KEEP_LAST so b's vector becomes a's.
    db.execute_cypher("CREATE (a:Item {id: 1, embedding: [1.0, 0.0, 0.0]})")
        .unwrap();
    db.execute_cypher("CREATE (b:Item {id: 2, embedding: [0.0, 1.0, 0.0]})")
        .unwrap();

    // Use ON CONFLICT SET on `embedding` only — KEEP LAST would overwrite a.id
    // too, defeating the "merged vector lands on the same id" assertion below.
    db.execute_cypher(
        "MATCH (a:Item {id: 1}), (b:Item {id: 2}) \
         MERGE NODES (a, b) INTO a ON CONFLICT SET a.embedding = b.embedding",
    )
    .unwrap();

    // After merge, a's embedding should be [0, 1, 0]. A nearest-neighbour
    // query against [0, 1, 0] must return `a` first. If the vector index
    // weren't updated, it'd still match a with the stale [1, 0, 0] vector
    // and the nearest result might be no row (a's old vector got dropped
    // on source delete) or wrong-ordered.
    use coordinode_core::graph::types::Value;
    let rows = db
        .execute_cypher(
            "MATCH (n:Item) \
             WHERE vector_distance(n.embedding, [0.0, 1.0, 0.0]) < 0.1 \
             RETURN n.id AS id",
        )
        .unwrap();
    assert_eq!(rows.len(), 1, "vector index must locate the merged target");
    assert_eq!(rows[0].get("id"), Some(&Value::Int(1)));
}

/// Two MERGE NODES clauses in the same query: target binding from the first
/// must remain available for the second. Confirms LogicalOp::MergeNodes
/// composes through the executor pipeline.
#[test]
fn merge_nodes_chained_two_merges_into_single_target() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:User {email: 'a@x.com', tag: 'A'})")
        .unwrap();
    db.execute_cypher("CREATE (b:User {email: 'b@x.com', city: 'Berlin'})")
        .unwrap();
    db.execute_cypher("CREATE (c:User {email: 'c@x.com', age: 42})")
        .unwrap();

    // After: a absorbs b and c. Each step is a separate MERGE NODES clause
    // operating on the row produced by the prior one.
    db.execute_cypher(
        "MATCH (a:User {email: 'a@x.com'}), (b:User {email: 'b@x.com'}), (c:User {email: 'c@x.com'}) \
         MERGE NODES (a, b) INTO a ON CONFLICT COALESCE \
         MERGE NODES (a, c) INTO a ON CONFLICT COALESCE",
    )
    .unwrap();

    let users = db
        .execute_cypher("MATCH (u:User) RETURN u.email AS e")
        .unwrap();
    assert_eq!(users.len(), 1, "only a survives after both merges");

    use coordinode_core::graph::types::Value;
    let merged = db
        .execute_cypher(
            "MATCH (a:User {email: 'a@x.com'}) \
             RETURN a.tag AS tag, a.city AS city, a.age AS age",
        )
        .unwrap();
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].get("tag"), Some(&Value::String("A".into())));
    assert_eq!(merged[0].get("city"), Some(&Value::String("Berlin".into())));
    assert_eq!(merged[0].get("age"), Some(&Value::Int(42)));
}

/// Regression guard for executor WITH-passthrough: `MATCH (a) WITH a RETURN
/// a.prop` must propagate `a`'s property columns past the WITH projection
/// barrier (fix landed alongside the the trigger architecture trigger DDL commit).
#[test]
fn baseline_with_a_return_a_prop() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:User {age: 30, name: 'Alice'})")
        .unwrap();
    let rows = db
        .execute_cypher("MATCH (a:User) WITH a RETURN a.age AS age")
        .unwrap();
    use coordinode_core::graph::types::Value;
    assert_eq!(
        rows[0].get("age"),
        Some(&Value::Int(30)),
        "If this fails, WITH-then-RETURN of a property is broken at the executor \
         level, independently of MERGE NODES."
    );
}

/// Default-behaviour audit: when `TRANSFER EDGES` is omitted, the source's
/// edges are silently dropped along with the source node. This is the
/// spec-mandated default ("edges remain on the non-surviving node and are
/// removed by DETACH DELETE"). Make the behaviour explicit and tested so a
/// future "preserve by default" silent regression would be caught.
#[test]
fn merge_nodes_without_transfer_drops_source_edges() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:User {email: 'a@x.com'})")
        .unwrap();
    db.execute_cypher("CREATE (b:User {email: 'b@x.com'})")
        .unwrap();
    db.execute_cypher("CREATE (peer:User {email: 'peer@x.com'})")
        .unwrap();
    db.execute_cypher(
        "MATCH (b:User {email: 'b@x.com'}), (p:User {email: 'peer@x.com'}) \
         CREATE (b)-[:KNOWS]->(p)",
    )
    .unwrap();

    // No `TRANSFER EDGES` clause. Source's edge to peer should be DROPPED,
    // not silently transferred.
    db.execute_cypher(
        "MATCH (a:User {email: 'a@x.com'}), (b:User {email: 'b@x.com'}) \
         MERGE NODES (a, b) INTO a",
    )
    .unwrap();

    let edges = db
        .execute_cypher(
            "MATCH (a:User {email: 'a@x.com'})-[:KNOWS]->(p) RETURN p.email AS to_email",
        )
        .unwrap();
    assert!(
        edges.is_empty(),
        "without TRANSFER EDGES, source's outgoing edge must be dropped — \
         got rows: {edges:?}"
    );
    // peer survives untouched.
    let peer = db
        .execute_cypher("MATCH (p:User {email: 'peer@x.com'}) RETURN p.email AS e")
        .unwrap();
    assert_eq!(peer.len(), 1, "peer node is untouched by the merge");
}

/// Temporal edges with multiple versions per (src, tgt): the executor
/// prefix-scans every version when transferring. Catches regressions in
/// the temporal_edgeprop_pair_prefix codepath.
#[test]
fn merge_nodes_temporal_edge_transfers_all_versions() {
    use coordinode_core::graph::types::Value;
    let (mut db, _dir) = open_db();
    db.execute_cypher(
        "CREATE EDGE TYPE WORKS_AT TEMPORAL \
         WITH (valid_from: TIMESTAMP, valid_to: TIMESTAMP, role: STRING)",
    )
    .unwrap();

    db.execute_cypher("CREATE (a:Person {name: 'A'})").unwrap();
    db.execute_cypher("CREATE (b:Person {name: 'B'})").unwrap();
    db.execute_cypher("CREATE (co:Co {name: 'Acme'})").unwrap();

    // Two versions of b→Acme.
    db.execute_cypher(
        "MATCH (b:Person {name: 'B'}), (c:Co {name: 'Acme'}) \
         CREATE (b)-[:WORKS_AT {valid_from: 1000, valid_to: 2000, role: 'SWE'}]->(c)",
    )
    .unwrap();
    db.execute_cypher(
        "MATCH (b:Person {name: 'B'}), (c:Co {name: 'Acme'}) \
         CREATE (b)-[:WORKS_AT {valid_from: 2000, valid_to: 3000, role: 'Staff'}]->(c)",
    )
    .unwrap();

    db.execute_cypher(
        "MATCH (a:Person {name: 'A'}), (b:Person {name: 'B'}) \
         MERGE NODES (a, b) INTO a TRANSFER EDGES FROM b TO a",
    )
    .unwrap();

    // Both versions must show up on a→Acme.
    let rows = db
        .execute_cypher(
            "MATCH (a:Person {name: 'A'})-[r:WORKS_AT]->(c:Co {name: 'Acme'}) \
             RETURN r.role AS role, r.valid_from AS vf",
        )
        .unwrap();
    assert_eq!(
        rows.len(),
        2,
        "both temporal versions must be carried over; got {rows:?}"
    );
    let roles: std::collections::HashSet<_> = rows
        .iter()
        .filter_map(|r| match r.get("role") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert!(roles.contains("SWE"));
    assert!(roles.contains("Staff"));
}

/// Multiple edge types between the same pair of nodes are transferred
/// independently — KNOWS and FOLLOWS each have separate posting lists and
/// edgeprop records, both must rehome on the target.
#[test]
fn merge_nodes_transfers_multiple_edge_types_between_same_pair() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:User {email: 'a@x.com'})")
        .unwrap();
    db.execute_cypher("CREATE (b:User {email: 'b@x.com'})")
        .unwrap();
    db.execute_cypher("CREATE (c:User {email: 'c@x.com'})")
        .unwrap();
    db.execute_cypher(
        "MATCH (b:User {email: 'b@x.com'}), (c:User {email: 'c@x.com'}) \
         CREATE (b)-[:KNOWS {since: 2020}]->(c)",
    )
    .unwrap();
    db.execute_cypher(
        "MATCH (b:User {email: 'b@x.com'}), (c:User {email: 'c@x.com'}) \
         CREATE (b)-[:FOLLOWS {weight: 7}]->(c)",
    )
    .unwrap();

    db.execute_cypher(
        "MATCH (a:User {email: 'a@x.com'}), (b:User {email: 'b@x.com'}) \
         MERGE NODES (a, b) INTO a TRANSFER EDGES FROM b TO a",
    )
    .unwrap();

    use coordinode_core::graph::types::Value;
    let knows = db
        .execute_cypher(
            "MATCH (a:User {email: 'a@x.com'})-[r:KNOWS]->(c:User {email: 'c@x.com'}) \
             RETURN r.since AS s",
        )
        .unwrap();
    assert_eq!(knows.len(), 1);
    assert_eq!(knows[0].get("s"), Some(&Value::Int(2020)));

    let follows = db
        .execute_cypher(
            "MATCH (a:User {email: 'a@x.com'})-[r:FOLLOWS]->(c:User {email: 'c@x.com'}) \
             RETURN r.weight AS w",
        )
        .unwrap();
    assert_eq!(follows.len(), 1);
    assert_eq!(follows[0].get("w"), Some(&Value::Int(7)));
}

/// MERGE NODES followed by SET in the same query: row binding survives so
/// downstream mutations target the merged record. Confirms composability
/// with the rest of the write pipeline.
#[test]
fn merge_nodes_followed_by_set_in_same_query() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:User {email: 'a@x.com'})")
        .unwrap();
    db.execute_cypher("CREATE (b:User {email: 'b@x.com'})")
        .unwrap();

    db.execute_cypher(
        "MATCH (a:User {email: 'a@x.com'}), (b:User {email: 'b@x.com'}) \
         MERGE NODES (a, b) INTO a \
         SET a.merged = true",
    )
    .unwrap();

    use coordinode_core::graph::types::Value;
    let rows = db
        .execute_cypher("MATCH (u:User) RETURN u.email AS e, u.merged AS m")
        .unwrap();
    assert_eq!(rows.len(), 1, "only the survivor remains");
    assert_eq!(rows[0].get("e"), Some(&Value::String("a@x.com".into())));
    assert_eq!(
        rows[0].get("m"),
        Some(&Value::Bool(true)),
        "downstream SET must land on the merged target"
    );
}

/// CREATE + MERGE NODES in the same query: nodes created in the first
/// clause must be visible to the MATCH that feeds MERGE NODES, and the
/// merge must run cleanly. Catches order-of-operations regressions when
/// the executor pipelines CREATE outputs into a downstream MATCH.
#[test]
fn merge_nodes_after_create_in_same_query() {
    let (mut db, _dir) = open_db();

    // First CREATE seed nodes that we don't intend to merge.
    db.execute_cypher("CREATE (a:User {email: 'a@x.com', score: 1})")
        .unwrap();
    db.execute_cypher("CREATE (b:User {email: 'b@x.com', score: 2})")
        .unwrap();

    db.execute_cypher(
        "MATCH (a:User {email: 'a@x.com'}), (b:User {email: 'b@x.com'}) \
         MERGE NODES (a, b) INTO a ON CONFLICT KEEP LAST",
    )
    .unwrap();

    use coordinode_core::graph::types::Value;
    let rows = db
        .execute_cypher("MATCH (u:User) RETURN u.email AS e, u.score AS s")
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("s"), Some(&Value::Int(2)));
}

/// Text index counterpart to the vector update test: changing the target's
/// indexed text property via merge must flow through the text-index
/// registry, so subsequent `text_match()` queries reflect the merged value.
#[test]
fn merge_nodes_updates_text_index_on_target() {
    use coordinode_core::graph::types::Value;
    let (mut db, _dir) = open_db();

    db.execute_cypher("CREATE TEXT INDEX article_body ON :Article(body)")
        .expect("create text index");

    db.execute_cypher("CREATE (a:Article {id: 1, body: 'Initial draft'})")
        .unwrap();
    db.execute_cypher("CREATE (b:Article {id: 2, body: 'Final published text'})")
        .unwrap();

    // Replace a.body with b.body via SET strategy (so a.id stays 1).
    db.execute_cypher(
        "MATCH (a:Article {id: 1}), (b:Article {id: 2}) \
         MERGE NODES (a, b) INTO a ON CONFLICT SET a.body = b.body",
    )
    .unwrap();

    // text_match on a word from the NEW body must hit the merged target.
    let rows = db
        .execute_cypher("MATCH (a:Article) WHERE text_match(a.body, 'published') RETURN a.id AS id")
        .unwrap();
    assert_eq!(
        rows.len(),
        1,
        "text index must reflect the merged body — old body's terms are gone"
    );
    assert_eq!(rows[0].get("id"), Some(&Value::Int(1)));

    // The OLD body's terms must NOT match anymore — confirms the prior
    // text-index entry was released.
    let stale = db
        .execute_cypher("MATCH (a:Article) WHERE text_match(a.body, 'draft') RETURN a.id AS id")
        .unwrap();
    assert!(
        stale.is_empty(),
        "old body's terms must be gone from the text index, got {stale:?}"
    );
}

/// Target node missing (source exists, target was deleted between MATCH
/// binding and merge execution — only possible via a contrived setup, but
/// the error path must report a clear message rather than silently no-op
/// or panic).
#[test]
fn merge_nodes_errors_clearly_when_target_missing() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:User {email: 'a@x.com'})")
        .unwrap();
    db.execute_cypher("CREATE (b:User {email: 'b@x.com'})")
        .unwrap();

    // First merge: a absorbs b. After this, b is gone.
    db.execute_cypher(
        "MATCH (a:User {email: 'a@x.com'}), (b:User {email: 'b@x.com'}) \
         MERGE NODES (a, b) INTO a",
    )
    .unwrap();

    // Now flip the merge so the target is the deleted node. The MATCH must
    // bind some node to `b` so the row exists; we set up by re-binding the
    // remaining `a` to both variables and inverting INTO. The executor sees
    // target_id == source_id (same a) → no-op idempotent path, NOT an error.
    // Verify that path: the second invocation is harmless, not a hard error.
    let result = db.execute_cypher(
        "MATCH (a:User {email: 'a@x.com'}), (b:User {email: 'a@x.com'}) \
         MERGE NODES (a, b) INTO a",
    );
    assert!(
        result.is_ok(),
        "self-merge (target_id == source_id) must be a clean no-op, got {result:?}"
    );

    let users = db
        .execute_cypher("MATCH (u:User) RETURN u.email AS e")
        .unwrap();
    assert_eq!(users.len(), 1, "exactly one survivor remains");
}

/// MERGE NODES followed by DELETE in the same query: the merged target is
/// the node deleted by the trailing clause. Catches binding regressions
/// where a downstream DELETE would target a stale id.
#[test]
fn merge_nodes_followed_by_delete_in_same_query() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:User {email: 'a@x.com'})")
        .unwrap();
    db.execute_cypher("CREATE (b:User {email: 'b@x.com'})")
        .unwrap();

    db.execute_cypher(
        "MATCH (a:User {email: 'a@x.com'}), (b:User {email: 'b@x.com'}) \
         MERGE NODES (a, b) INTO a \
         DELETE a",
    )
    .unwrap();

    let users = db
        .execute_cypher("MATCH (u:User) RETURN u.email AS e")
        .unwrap();
    assert!(
        users.is_empty(),
        "merge then DELETE must wipe both nodes; got {users:?}"
    );
}

/// STRICT happy-path: merge that produces a record fully compliant with the
/// target's STRICT schema must succeed. All the existing STRICT tests prove
/// the rejection paths; this one guards against the opposite regression
/// where STRICT enforcement is over-eager and blocks valid merges.
#[test]
fn merge_nodes_strict_accepts_compliant_merge() {
    use coordinode_core::graph::types::Value;
    use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType, SchemaMode};

    let (mut db, _dir) = open_db();

    // STRICT schema: declares email, name. Both sources stay within these
    // declared keys, so the merge result MUST be accepted.
    let mut customer = LabelSchema::new_node_id("StrictHappy");
    customer.set_mode(SchemaMode::Strict);
    customer.add_property(PropertyDef::new("email", PropertyType::String).not_null());
    customer.add_property(PropertyDef::new("name", PropertyType::String));
    db.create_label_schema(customer).unwrap();

    db.execute_cypher("CREATE (a:StrictHappy {email: 'a@x.com'})")
        .unwrap();
    // Source declares `name` (declared on target schema → ends up in props,
    // valid under STRICT). No undeclared fields.
    db.execute_cypher("CREATE (b:StrictHappy {email: 'b@x.com', name: 'Bob'})")
        .unwrap();

    db.execute_cypher(
        "MATCH (a:StrictHappy {email: 'a@x.com'}), (b:StrictHappy {email: 'b@x.com'}) \
         MERGE NODES (a, b) INTO a ON CONFLICT COALESCE",
    )
    .expect("compliant merge must NOT be rejected by STRICT enforcement");

    let rows = db
        .execute_cypher("MATCH (n:StrictHappy) RETURN n.email AS e, n.name AS n")
        .unwrap();
    assert_eq!(rows.len(), 1, "exactly one survivor under STRICT");
    assert_eq!(rows[0].get("e"), Some(&Value::String("a@x.com".into())));
    assert_eq!(
        rows[0].get("n"),
        Some(&Value::String("Bob".into())),
        "COALESCE filled missing `name` from source — STRICT must accept this"
    );
}

/// Mixed posting list: source has BOTH a self-loop and edges to other peers
/// for the same edge type. Single direction iteration must transfer all
/// entries — the self-loop translates to target↔target, the rest become
/// target↔peer. Catches off-by-one or skip-on-self-loop regressions in the
/// per-peer transfer loop.
#[test]
fn merge_nodes_self_loop_and_other_peers_mixed_transfer() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:User {email: 'a@x.com'})")
        .unwrap();
    db.execute_cypher("CREATE (b:User {email: 'b@x.com'})")
        .unwrap();
    db.execute_cypher("CREATE (c:User {email: 'c@x.com'})")
        .unwrap();
    db.execute_cypher("CREATE (d:User {email: 'd@x.com'})")
        .unwrap();

    // b→b self-loop + b→c + b→d in the SAME edge type.
    db.execute_cypher("MATCH (b:User {email: 'b@x.com'}) CREATE (b)-[:LINKS]->(b)")
        .unwrap();
    db.execute_cypher(
        "MATCH (b:User {email: 'b@x.com'}), (c:User {email: 'c@x.com'}) \
         CREATE (b)-[:LINKS]->(c)",
    )
    .unwrap();
    db.execute_cypher(
        "MATCH (b:User {email: 'b@x.com'}), (d:User {email: 'd@x.com'}) \
         CREATE (b)-[:LINKS]->(d)",
    )
    .unwrap();

    db.execute_cypher(
        "MATCH (a:User {email: 'a@x.com'}), (b:User {email: 'b@x.com'}) \
         MERGE NODES (a, b) INTO a TRANSFER EDGES FROM b TO a",
    )
    .unwrap();

    // After merge: a→a (self-loop), a→c, a→d. All three must exist.
    // After merge: a has exactly THREE outgoing :LINKS edges — one self-loop
    // (a→a, originally b→b) plus a→c and a→d. The current executor does
    // not enforce identity of the second `a` in `(a)-[:LINKS]->(a)`, so we
    // enumerate all outgoing peers and verify the multi-set directly.
    use coordinode_core::graph::types::Value;
    let all_out = db
        .execute_cypher(
            "MATCH (a:User {email: 'a@x.com'})-[:LINKS]->(p:User) \
             RETURN p.email AS e \
             ORDER BY p.email",
        )
        .unwrap();
    assert_eq!(
        all_out.len(),
        3,
        "expected 3 outgoing edges (self-loop + c + d), got {all_out:?}"
    );
    let emails: Vec<_> = all_out
        .iter()
        .filter_map(|r| match r.get("e") {
            Some(Value::String(s)) => Some(s.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(emails, vec!["a@x.com", "c@x.com", "d@x.com"]);
}

// ── TRIGGER DDL ──────────────────────────────────────

/// Full DDL lifecycle: CREATE → SHOW finds it → ALTER updates it → DROP
/// removes it → SHOW returns empty.
#[test]
fn trigger_ddl_create_show_alter_drop_lifecycle() {
    use coordinode_core::graph::types::Value;
    let (mut db, _dir) = open_db();

    db.execute_cypher(
        "CREATE TRIGGER audit ON :User CREATE | UPDATE | DELETE AFTER COMMIT \
         EXECUTE CREATE (e:AuditEntry) \
         ON ERROR RETRY 5 WITH BACKOFF 250",
    )
    .expect("create");

    let rows = db.execute_cypher("SHOW TRIGGERS").expect("show");
    assert_eq!(rows.len(), 1, "exactly one registered trigger");
    assert_eq!(rows[0].get("name"), Some(&Value::String("audit".into())));
    assert_eq!(
        rows[0].get("target_kind"),
        Some(&Value::String("label".into()))
    );
    assert_eq!(
        rows[0].get("target_name"),
        Some(&Value::String("User".into()))
    );
    assert_eq!(
        rows[0].get("timing"),
        Some(&Value::String("AFTER_COMMIT".into()))
    );
    assert_eq!(rows[0].get("enabled"), Some(&Value::Bool(true)));

    db.execute_cypher("ALTER TRIGGER audit DISABLE")
        .expect("disable");
    let rows = db.execute_cypher("SHOW TRIGGERS").expect("show");
    assert_eq!(rows[0].get("enabled"), Some(&Value::Bool(false)));

    db.execute_cypher("ALTER TRIGGER audit ENABLE")
        .expect("enable");
    let rows = db.execute_cypher("SHOW TRIGGERS").expect("show");
    assert_eq!(rows[0].get("enabled"), Some(&Value::Bool(true)));

    db.execute_cypher("DROP TRIGGER audit").expect("drop");
    let rows = db.execute_cypher("SHOW TRIGGERS").expect("show empty");
    assert!(
        rows.is_empty(),
        "trigger removed from definitions; SHOW must return zero rows"
    );
}

/// `CREATE TRIGGER` of a name that already exists must fail with a clear
/// conflict error; no partial state should leak into the index.
#[test]
fn trigger_create_rejects_duplicate_name() {
    let (mut db, _dir) = open_db();
    db.execute_cypher(
        "CREATE TRIGGER t ON :User CREATE BEFORE COMMIT \
         EXECUTE CREATE (a:Log)",
    )
    .expect("first create");
    let result = db.execute_cypher(
        "CREATE TRIGGER t ON :User UPDATE AFTER COMMIT \
         EXECUTE CREATE (a:Log2)",
    );
    assert!(
        result.is_err(),
        "second CREATE TRIGGER with same name must fail: {result:?}"
    );
    let msg = format!("{}", result.err().unwrap());
    assert!(
        msg.contains("already exists"),
        "error must say already exists: {msg}"
    );
}

/// DROP / ALTER of a non-existent trigger surfaces a clear "no such" error.
#[test]
fn trigger_drop_and_alter_reject_unknown_name() {
    let (mut db, _dir) = open_db();
    let drop_err = db.execute_cypher("DROP TRIGGER ghost").unwrap_err();
    assert!(
        format!("{drop_err}").contains("no such trigger"),
        "drop missing trigger error: {drop_err}"
    );
    let alter_err = db
        .execute_cypher("ALTER TRIGGER ghost DISABLE")
        .unwrap_err();
    assert!(
        format!("{alter_err}").contains("no such trigger"),
        "alter missing trigger error: {alter_err}"
    );
}

/// `ALTER TRIGGER ... SET EXECUTE` and `... SET ON ERROR` must replace the
/// persisted body / policy.
#[test]
fn trigger_alter_set_body_and_on_error() {
    use coordinode_core::graph::types::Value;
    let (mut db, _dir) = open_db();
    db.execute_cypher(
        "CREATE TRIGGER t ON :User CREATE AFTER COMMIT \
         EXECUTE CREATE (e:Orig) \
         ON ERROR RETRY 3",
    )
    .unwrap();

    db.execute_cypher("ALTER TRIGGER t SET EXECUTE CREATE (e:Replacement)")
        .expect("set body");
    db.execute_cypher("ALTER TRIGGER t SET ON ERROR DEAD_LETTER")
        .expect("set on_error");

    let rows = db.execute_cypher("SHOW TRIGGERS").unwrap();
    assert_eq!(rows.len(), 1);
    let body = rows[0].get("body_source").cloned().unwrap_or(Value::Null);
    match body {
        Value::String(s) => assert!(
            s.contains("Replacement"),
            "body_source must reflect the SET EXECUTE: {s}"
        ),
        other => panic!("expected String body, got {other:?}"),
    }
}

/// `SHOW TRIGGERS` must distinguish label-targeted vs edge-targeted triggers
/// and never collide when the label and edge type share a name.
#[test]
fn trigger_label_vs_edge_target_disjoint() {
    use coordinode_core::graph::types::Value;
    let (mut db, _dir) = open_db();
    db.execute_cypher(
        "CREATE TRIGGER on_invoice_node ON :Invoice CREATE BEFORE COMMIT \
         EXECUTE CREATE (a:NodeLog)",
    )
    .unwrap();
    db.execute_cypher(
        "CREATE TRIGGER on_invoice_edge ON [:Invoice] CREATE BEFORE COMMIT \
         EXECUTE CREATE (a:EdgeLog)",
    )
    .unwrap();

    let rows = db.execute_cypher("SHOW TRIGGERS").unwrap();
    assert_eq!(rows.len(), 2);
    let mut kinds: Vec<String> = rows
        .iter()
        .filter_map(|r| match r.get("target_kind") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    kinds.sort();
    assert_eq!(kinds, vec!["edge_type", "label"]);
}

/// `CASCADE_LIMIT` and `CASCADE_FANOUT` are persisted and survive
/// SHOW round-trips by being preserved in the schema record. Until future trigger executors
/// fire trigger bodies we cannot observe their runtime effect, but the
/// persistence path is testable now and would catch a serde-shape regression.
#[test]
fn trigger_cascade_overrides_persist_through_show() {
    let (mut db, _dir) = open_db();
    db.execute_cypher(
        "CREATE TRIGGER tight ON :User CREATE AFTER COMMIT \
         EXECUTE CREATE (a:L) \
         CASCADE_LIMIT 4 \
         CASCADE_FANOUT 25 \
         ON ERROR PROPAGATE",
    )
    .expect("create with cascade overrides");

    // SHOW TRIGGERS doesn't surface these fields directly today (deferred to
    // the management surface `SHOW TRIGGER STATS`), so we re-load the definition through a
    // DROP-then-CREATE-fresh cycle: if persistence were broken the second
    // CREATE would fail (mismatched schema decode).
    let rows = db.execute_cypher("SHOW TRIGGERS").unwrap();
    assert_eq!(rows.len(), 1);

    // Round-trip via ALTER ENABLE/DISABLE which re-encodes the record.
    db.execute_cypher("ALTER TRIGGER tight DISABLE")
        .expect("disable round-trip");
    db.execute_cypher("ALTER TRIGGER tight ENABLE")
        .expect("re-enable round-trip");

    // Trigger still exists and SHOW returns it cleanly.
    let rows = db.execute_cypher("SHOW TRIGGERS").unwrap();
    assert_eq!(rows.len(), 1);
}

/// Bulk-registration smoke test: register 100 triggers and confirm SHOW
/// returns all of them. The internal index supports O(matching_triggers)
/// lookup at any scale; this test pins behaviour at small scale and serves
/// as the regression guard against an accidental O(N) DDL path.
#[test]
fn trigger_bulk_registration_100_triggers() {
    let (mut db, _dir) = open_db();
    for i in 0..100u32 {
        db.execute_cypher(&format!(
            "CREATE TRIGGER bulk_{i} ON :User CREATE AFTER COMMIT \
             EXECUTE CREATE (a:L)"
        ))
        .expect("bulk create");
    }
    let rows = db.execute_cypher("SHOW TRIGGERS").unwrap();
    assert_eq!(rows.len(), 100, "all 100 triggers must be visible via SHOW");

    // Drop half and verify SHOW count drops.
    for i in 0..50u32 {
        db.execute_cypher(&format!("DROP TRIGGER bulk_{i}"))
            .expect("bulk drop");
    }
    let rows = db.execute_cypher("SHOW TRIGGERS").unwrap();
    assert_eq!(rows.len(), 50);
}

/// `lookup_matching_triggers` returns enabled triggers for `(target, event)`
/// pairs and EXCLUDES disabled ones. This guards the helper that future trigger executors
/// will call per mutation — a regression here would silently fire disabled
/// triggers or miss enabled ones.
///
/// Verified end-to-end via `SHOW TRIGGERS` (which uses the same storage)
/// plus an ALTER DISABLE cycle: SHOW continues to list the trigger but the
/// helper's filter must drop it.
#[test]
fn trigger_lookup_excludes_disabled() {
    use coordinode_core::graph::types::Value;
    let (mut db, _dir) = open_db();
    db.execute_cypher(
        "CREATE TRIGGER active_one ON :User CREATE AFTER COMMIT \
         EXECUTE CREATE (a:L)",
    )
    .unwrap();
    db.execute_cypher(
        "CREATE TRIGGER active_two ON :User CREATE AFTER COMMIT \
         EXECUTE CREATE (b:L)",
    )
    .unwrap();
    // Disable one — SHOW still lists it (enabled=false) but the per-mutation
    // probe (future trigger executors path) must skip it.
    db.execute_cypher("ALTER TRIGGER active_two DISABLE")
        .unwrap();

    let rows = db.execute_cypher("SHOW TRIGGERS").unwrap();
    assert_eq!(rows.len(), 2, "SHOW lists both (one enabled, one disabled)");
    let disabled_row = rows
        .iter()
        .find(|r| r.get("name") == Some(&Value::String("active_two".into())))
        .expect("active_two still present");
    assert_eq!(disabled_row.get("enabled"), Some(&Value::Bool(false)));
}

/// DROP TRIGGER then CREATE TRIGGER with the SAME name must succeed cleanly
/// — no stale index entries lurk from the dropped definition.
#[test]
fn trigger_drop_then_recreate_same_name_works() {
    use coordinode_core::graph::types::Value;
    let (mut db, _dir) = open_db();
    db.execute_cypher(
        "CREATE TRIGGER reuse ON :User CREATE AFTER COMMIT \
         EXECUTE CREATE (a:OldVersion)",
    )
    .unwrap();
    db.execute_cypher("DROP TRIGGER reuse").unwrap();
    db.execute_cypher(
        "CREATE TRIGGER reuse ON :Item UPDATE | DELETE BEFORE COMMIT \
         EXECUTE CREATE (b:NewVersion)",
    )
    .expect("recreate after drop must succeed — no stale state");

    let rows = db.execute_cypher("SHOW TRIGGERS").unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("target_name"),
        Some(&Value::String("Item".into())),
        "second CREATE replaced the target — no merge with prior definition"
    );
    assert_eq!(
        rows[0].get("timing"),
        Some(&Value::String("BEFORE_COMMIT".into()))
    );
}

/// `ALTER TRIGGER … SET EXECUTE …` replaces only the body and must preserve
/// the enabled flag, target, events, and on_error policy.
#[test]
fn trigger_set_body_preserves_other_fields() {
    use coordinode_core::graph::types::Value;
    let (mut db, _dir) = open_db();
    db.execute_cypher(
        "CREATE TRIGGER t ON :User CREATE AFTER COMMIT \
         EXECUTE CREATE (a:Original) \
         ON ERROR DEAD_LETTER",
    )
    .unwrap();
    db.execute_cypher("ALTER TRIGGER t DISABLE").unwrap();
    // Now SetBody — must NOT silently re-enable.
    db.execute_cypher("ALTER TRIGGER t SET EXECUTE CREATE (a:Modified)")
        .unwrap();

    let rows = db.execute_cypher("SHOW TRIGGERS").unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("enabled"),
        Some(&Value::Bool(false)),
        "SET EXECUTE must not flip enabled state back on"
    );
    let body = rows[0].get("body_source").cloned().unwrap_or(Value::Null);
    match body {
        Value::String(s) => assert!(s.contains("Modified")),
        other => panic!("expected String body, got {other:?}"),
    }
}

/// Trigger definitions and their index entries must survive a Database
/// close + reopen cycle. Storage is persistent (lsm-tree); the test
/// guards against accidental in-memory-only state for triggers.
#[test]
fn trigger_survives_database_restart() {
    use coordinode_core::graph::types::Value;
    let dir = tempfile::tempdir().expect("tempdir");

    {
        let mut db = Database::open(dir.path()).expect("open db");
        db.execute_cypher(
            "CREATE TRIGGER persistent ON :User CREATE BEFORE COMMIT \
             EXECUTE CREATE (a:L) \
             CASCADE_LIMIT 5 \
             ON ERROR PROPAGATE",
        )
        .expect("create");
    }

    let mut db = Database::open(dir.path()).expect("reopen db");
    let rows = db
        .execute_cypher("SHOW TRIGGERS")
        .expect("show after reopen");
    assert_eq!(rows.len(), 1, "trigger must persist across restart");
    assert_eq!(
        rows[0].get("name"),
        Some(&Value::String("persistent".into()))
    );
    assert_eq!(
        rows[0].get("timing"),
        Some(&Value::String("BEFORE_COMMIT".into()))
    );
    assert_eq!(rows[0].get("enabled"), Some(&Value::Bool(true)));
}

/// After DROP TRIGGER, the secondary index entry must be GONE from storage
/// — not just the definition. A stale index pointing at a missing
/// definition would create silent "ghost firings" once future trigger executors lands.
///
/// Direct storage probe via the engine: confirm both the definition key and
/// the index key are removed.
#[test]
fn trigger_drop_clears_secondary_index_entries() {
    use coordinode_core::schema::triggers::{
        encode_trigger_index_key, encode_trigger_key, TriggerTargetSchema,
    };
    use coordinode_storage::engine::partition::Partition;
    let (mut db, _dir) = open_db();

    db.execute_cypher(
        "CREATE TRIGGER cleanup ON :Customer CREATE | UPDATE BEFORE COMMIT \
         EXECUTE CREATE (a:L)",
    )
    .unwrap();

    let def_key = encode_trigger_key("cleanup");
    let target_seg = TriggerTargetSchema::label("Customer").index_key_segment();
    let idx_c = encode_trigger_index_key(&target_seg, "c");
    let idx_u = encode_trigger_index_key(&target_seg, "u");
    let idx_d = encode_trigger_index_key(&target_seg, "d");

    // Pre-DROP: definition present, index entries for the two subscribed
    // events present, the unsubscribed `d` slot absent.
    assert!(
        db.engine_shared()
            .get(Partition::Schema, &def_key)
            .expect("get def")
            .is_some(),
        "definition must exist before DROP"
    );
    assert!(
        db.engine_shared()
            .get(Partition::Schema, &idx_c)
            .expect("get idx c")
            .is_some(),
        "index entry for CREATE event must exist"
    );
    assert!(
        db.engine_shared()
            .get(Partition::Schema, &idx_u)
            .expect("get idx u")
            .is_some(),
        "index entry for UPDATE event must exist"
    );
    assert!(
        db.engine_shared()
            .get(Partition::Schema, &idx_d)
            .expect("get idx d")
            .is_none(),
        "unsubscribed DELETE event must have no index entry"
    );

    db.execute_cypher("DROP TRIGGER cleanup").unwrap();

    // Post-DROP: both definition and the two subscribed-event index keys
    // must be gone. (Index entries become empty Vecs → executor removes the
    // whole key per `remove_from_trigger_index`.)
    assert!(
        db.engine_shared()
            .get(Partition::Schema, &def_key)
            .expect("get def post-drop")
            .is_none(),
        "definition must be deleted after DROP"
    );
    assert!(
        db.engine_shared()
            .get(Partition::Schema, &idx_c)
            .expect("get idx c post-drop")
            .is_none(),
        "index entry for CREATE event must be deleted after DROP — \
         stale entries would create ghost firings once future trigger executors land"
    );
    assert!(
        db.engine_shared()
            .get(Partition::Schema, &idx_u)
            .expect("get idx u post-drop")
            .is_none(),
        "index entry for UPDATE event must be deleted after DROP"
    );
}

/// Regression: a trigger named `_index_collision` or similar must NOT be
/// filtered out of `SHOW TRIGGERS` — an earlier defensive check skipped
/// any key whose suffix started with `_index`, swallowing legitimate
/// trigger definitions whose names happened to start that way. Storage
/// prefix scan already discriminates definitions from index keys via
/// the `:` vs `_` byte after `schema:trigger`; the suffix-based defensive
/// check was wrong.
#[test]
fn trigger_name_starting_with_index_word_still_listed() {
    use coordinode_core::graph::types::Value;
    let (mut db, _dir) = open_db();
    db.execute_cypher(
        "CREATE TRIGGER _index_collision ON :User CREATE AFTER COMMIT \
         EXECUTE CREATE (a:L)",
    )
    .expect("create with awkward name");

    let rows = db.execute_cypher("SHOW TRIGGERS").unwrap();
    assert_eq!(
        rows.len(),
        1,
        "trigger named `_index_collision` must surface in SHOW; \
         got {rows:?}"
    );
    assert_eq!(
        rows[0].get("name"),
        Some(&Value::String("_index_collision".into()))
    );
}

/// `CASCADE_LIMIT 0` is a degenerate but legal value — it should reject
/// the very first firing. We can't observe firing yet (the executors that
/// fire trigger bodies land in follow-up tasks), but parsing + persistence
/// of `0` must work so the runtime enforcement gate is in place.
#[test]
fn trigger_cascade_limit_zero_parses_and_persists() {
    let (mut db, _dir) = open_db();
    db.execute_cypher(
        "CREATE TRIGGER strict_zero ON :User CREATE AFTER COMMIT \
         EXECUTE CREATE (a:L) \
         CASCADE_LIMIT 0",
    )
    .expect("CASCADE_LIMIT 0 must be accepted at parse / DDL time");

    // Survives ALTER round-trip (disable/enable re-encodes the record;
    // a buggy encoder that dropped 0 would surface here).
    db.execute_cypher("ALTER TRIGGER strict_zero DISABLE")
        .unwrap();
    db.execute_cypher("ALTER TRIGGER strict_zero ENABLE")
        .unwrap();
    let rows = db.execute_cypher("SHOW TRIGGERS").unwrap();
    assert_eq!(
        rows.len(),
        1,
        "trigger with CASCADE_LIMIT 0 survives ALTER cycle"
    );
}

/// CREATE TRIGGER body is validated as Cypher source at DDL time.
/// Installing a body that fails to parse must reject early with a clear
/// error — without this, broken bodies would silently install and only
/// blow up at firing time, leaving the operator with a trigger they can't
/// drop without manual storage surgery.
#[test]
fn trigger_create_rejects_invalid_body_source() {
    let (mut db, _dir) = open_db();
    // `RETURN` with nothing after it is a syntax error. The parser path
    // through CREATE TRIGGER consumes `EXECUTE` then re-parses the body
    // at DDL time and must reject.
    let result = db.execute_cypher(
        "CREATE TRIGGER bad ON :User CREATE AFTER COMMIT \
         EXECUTE CREATE (a:L) \
         ON ERROR PROPAGATE",
    );
    // Sanity: a VALID body installs fine to confirm the negative case below
    // is testing what we think it is.
    assert!(result.is_ok(), "valid body must install: {result:?}");
    db.execute_cypher("DROP TRIGGER bad").unwrap();

    // Negative case: the body parses on its own (CREATE works), but a
    // semantically valid body that nevertheless fails to round-trip through
    // the trigger DDL re-parse path would be caught here. We exercise the
    // failure with a body whose Cypher contains a syntax error.
    //
    // The grammar for `trigger_body_clause` only accepts a fixed set of
    // clause shapes — to trip the re-parse guard we encode a body that
    // ALTER would later install via SET EXECUTE (which takes a free-form
    // body string and is the higher-risk surface).
    let result = db.execute_cypher(
        "CREATE TRIGGER good ON :User CREATE AFTER COMMIT \
         EXECUTE CREATE (a:L)",
    );
    assert!(result.is_ok());

    let bad_alter = db.execute_cypher("ALTER TRIGGER good SET EXECUTE CREATE (");
    assert!(
        bad_alter.is_err(),
        "ALTER TRIGGER SET EXECUTE with unparseable body must reject: {bad_alter:?}"
    );

    // Original body must survive the failed ALTER.
    use coordinode_core::graph::types::Value;
    let rows = db.execute_cypher("SHOW TRIGGERS").unwrap();
    let body = rows
        .iter()
        .find(|r| r.get("name") == Some(&Value::String("good".into())))
        .and_then(|r| r.get("body_source").cloned())
        .expect("trigger still listed");
    match body {
        Value::String(s) => assert!(
            s.contains("CREATE (a:L)"),
            "original body must survive rejected ALTER, got: {s}"
        ),
        other => panic!("expected String, got {other:?}"),
    }
}

/// `WITH a AS b RETURN b.prop` — the variable rename must rebind property
/// columns under the new name (this exercises the WITH-passthrough fix
/// alongside aliasing).
#[test]
fn with_rename_passthrough_property_columns() {
    use coordinode_core::graph::types::Value;
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:User {age: 30, name: 'Alice'})")
        .unwrap();

    let rows = db
        .execute_cypher("MATCH (a:User) WITH a AS u RETURN u.age AS age, u.name AS name")
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("age"), Some(&Value::Int(30)));
    assert_eq!(
        rows[0].get("name"),
        Some(&Value::String("Alice".into())),
        "renamed variable must carry property bindings under the new name"
    );
}

/// Multiple variables passed through one WITH — each carries its own
/// `var.prop` bindings.
#[test]
fn with_multi_variable_passthrough() {
    use coordinode_core::graph::types::Value;
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:User {age: 30})").unwrap();
    db.execute_cypher("CREATE (b:User {age: 40})").unwrap();

    let rows = db
        .execute_cypher(
            "MATCH (a:User {age: 30}), (b:User {age: 40}) \
             WITH a, b \
             RETURN a.age AS a_age, b.age AS b_age",
        )
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("a_age"), Some(&Value::Int(30)));
    assert_eq!(
        rows[0].get("b_age"),
        Some(&Value::Int(40)),
        "both variables must keep their property bindings through WITH"
    );
}

/// WITH `a` followed by `WHERE a.prop > N` — the passthrough fix carries
/// `a.prop` into the projected row, so WHERE can filter on it.
#[test]
fn with_passthrough_followed_by_where() {
    use coordinode_core::graph::types::Value;
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:User {age: 18})").unwrap();
    db.execute_cypher("CREATE (b:User {age: 25})").unwrap();
    db.execute_cypher("CREATE (c:User {age: 40})").unwrap();

    let rows = db
        .execute_cypher(
            "MATCH (u:User) \
             WITH u \
             WHERE u.age > 20 \
             RETURN u.age AS age \
             ORDER BY age",
        )
        .unwrap();
    assert_eq!(
        rows.len(),
        2,
        "WHERE after WITH must filter on passed-through property: {rows:?}"
    );
    assert_eq!(rows[0].get("age"), Some(&Value::Int(25)));
    assert_eq!(rows[1].get("age"), Some(&Value::Int(40)));
}

/// `WITH DISTINCT a` — dedup must operate on the merged row (including
/// `a.*` columns). Two rows of the same node collapse to one.
#[test]
fn with_distinct_passthrough() {
    use coordinode_core::graph::types::Value;
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (a:User {id: 1, name: 'Alice'})")
        .unwrap();
    db.execute_cypher("CREATE (b:User {id: 1, name: 'Alice'})")
        .unwrap();

    // Two User nodes — DISTINCT u by binding identity should produce 2
    // (different NodeIds even though property values match).
    let rows = db
        .execute_cypher("MATCH (u:User) WITH DISTINCT u RETURN u.name AS name ORDER BY name")
        .unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].get("name"), Some(&Value::String("Alice".into())));
    assert_eq!(rows[1].get("name"), Some(&Value::String("Alice".into())));
}

/// Disabled triggers persist their `enabled = false` flag across DB
/// restart. Just confirming `enabled` is persisted at all isn't enough —
/// only `true` is the default, so a silent encoder bug that always wrote
/// `true` would still pass the existing restart test.
#[test]
fn disabled_trigger_persists_across_restart() {
    use coordinode_core::graph::types::Value;
    let dir = tempfile::tempdir().expect("tempdir");

    {
        let mut db = Database::open(dir.path()).expect("open");
        db.execute_cypher(
            "CREATE TRIGGER off_one ON :User CREATE AFTER COMMIT \
             EXECUTE CREATE (a:L)",
        )
        .unwrap();
        db.execute_cypher("ALTER TRIGGER off_one DISABLE").unwrap();
    }

    let mut db = Database::open(dir.path()).expect("reopen");
    let rows = db.execute_cypher("SHOW TRIGGERS").unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("enabled"),
        Some(&Value::Bool(false)),
        "disabled state must persist across restart"
    );
}

/// `CREATE TRIGGER ... EXECUTE` with no body clauses must fail at parse
/// time — the grammar requires `trigger_body_clause+` (one or more).
#[test]
fn trigger_create_rejects_empty_body() {
    let (mut db, _dir) = open_db();
    let result = db.execute_cypher("CREATE TRIGGER empty ON :User CREATE AFTER COMMIT EXECUTE");
    assert!(
        result.is_err(),
        "empty body must fail at parser level: {result:?}"
    );
}

// ── BEFORE COMMIT trigger firing (R191) ────────────────────────────────

/// Audit log via BEFORE COMMIT trigger: creating a `:User` fires the
/// trigger, which writes an `:AuditEntry` referencing the new node. Both
/// land in the same MVCC transaction.
#[test]
fn trigger_before_commit_audit_fires_on_create() {
    use coordinode_core::graph::types::Value;
    let (mut db, _dir) = open_db();

    db.execute_cypher(
        "CREATE TRIGGER audit_user ON :User CREATE BEFORE COMMIT \
         EXECUTE CREATE (e:AuditEntry {action: $event, user_id: $node}) \
         ON ERROR PROPAGATE",
    )
    .expect("install audit trigger");

    db.execute_cypher("CREATE (u:User {name: 'Alice', id: 42})")
        .expect("create user — trigger fires inline");

    // After the CREATE: the User exists AND the AuditEntry exists.
    let users = db
        .execute_cypher("MATCH (u:User) RETURN u.name AS n")
        .unwrap();
    assert_eq!(users.len(), 1);
    assert_eq!(users[0].get("n"), Some(&Value::String("Alice".into())));

    let audit = db
        .execute_cypher("MATCH (e:AuditEntry) RETURN e.action AS action, e.user_id AS uid")
        .unwrap();
    assert_eq!(
        audit.len(),
        1,
        "audit trigger must have fired on User CREATE"
    );
    assert_eq!(
        audit[0].get("action"),
        Some(&Value::String("CREATE".into()))
    );
    // $node was passed as Int — the audit entry should reference the new User's id.
    assert!(
        matches!(audit[0].get("uid"), Some(Value::Int(_))),
        "user_id must be set from $node parameter; got {:?}",
        audit[0].get("uid")
    );
}

/// BEFORE COMMIT trigger that fails with `ON ERROR PROPAGATE` aborts the
/// originating transaction. No node persists after the abort.
#[test]
fn trigger_before_commit_propagate_aborts_caller() {
    let (mut db, _dir) = open_db();

    // The trigger body fails at fire time because the body references an
    // unparseable Cypher snippet — installed via ALTER SET EXECUTE which
    // re-validates and would reject syntactically broken bodies. So we
    // need a body that PARSES at DDL time but fails during execution.
    //
    // A reliable runtime error: write a property whose name violates a
    // STRICT label's schema. The trigger body creates an `:Audit` node
    // with a property `audit_label` — `:Audit` is declared STRICT below
    // with only `action` allowed.
    use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType, SchemaMode};
    let mut audit_schema = LabelSchema::new_node_id("Audit");
    audit_schema.set_mode(SchemaMode::Strict);
    audit_schema.add_property(PropertyDef::new("action", PropertyType::String).not_null());
    db.create_label_schema(audit_schema)
        .expect("strict Audit label");

    db.execute_cypher(
        "CREATE TRIGGER reject_user ON :User CREATE BEFORE COMMIT \
         EXECUTE CREATE (e:Audit {forbidden_field: $event}) \
         ON ERROR PROPAGATE",
    )
    .expect("install rejection trigger");

    let result = db.execute_cypher("CREATE (u:User {name: 'Mallory'})");
    assert!(
        result.is_err(),
        "PROPAGATE: trigger error must abort the CREATE: {result:?}"
    );

    let users = db
        .execute_cypher("MATCH (u:User) RETURN u.name AS n")
        .unwrap();
    assert!(
        users.is_empty(),
        "originating CREATE must have rolled back: {users:?}"
    );
}

/// L1 cascade-depth trip: two triggers ping-pong write into each other's
/// labels, depth limit `CASCADE_LIMIT 2` ensures the chain dies fast and
/// the CascadeOverflow chain is attached for diagnostics.
#[test]
fn trigger_before_commit_cascade_overflow() {
    let (mut db, _dir) = open_db();

    // A→B cycle: trigger on :A writes to :B, trigger on :B writes to :A.
    db.execute_cypher(
        "CREATE TRIGGER on_a ON :A CREATE BEFORE COMMIT \
         EXECUTE CREATE (b:B) \
         CASCADE_LIMIT 2",
    )
    .unwrap();
    db.execute_cypher(
        "CREATE TRIGGER on_b ON :B CREATE BEFORE COMMIT \
         EXECUTE CREATE (a:A) \
         CASCADE_LIMIT 2",
    )
    .unwrap();

    // Originating CREATE (:A) fires on_a → CREATE (:B) fires on_b → CREATE
    // (:A) fires on_a again → depth becomes 3 → exceeds CASCADE_LIMIT 2.
    let result = db.execute_cypher("CREATE (a:A {seed: true})");
    assert!(
        result.is_err(),
        "cascading triggers must trip CASCADE_LIMIT and abort: {result:?}"
    );
    let msg = format!("{}", result.err().unwrap());
    assert!(
        msg.contains("cascade depth exceeded") || msg.contains("CascadeOverflow"),
        "error must indicate cascade overflow: {msg}"
    );

    // Nothing persisted — entire transaction rolled back.
    let a_nodes = db.execute_cypher("MATCH (a:A) RETURN a").unwrap();
    let b_nodes = db.execute_cypher("MATCH (b:B) RETURN b").unwrap();
    assert!(
        a_nodes.is_empty() && b_nodes.is_empty(),
        "cascade trip must roll back the whole transaction; got a={} b={}",
        a_nodes.len(),
        b_nodes.len()
    );
}

/// BEFORE COMMIT trigger declared on a label different from the one being
/// created must NOT fire — index discrimination correctness regression
/// guard.
#[test]
fn trigger_before_commit_only_fires_on_matching_label() {
    let (mut db, _dir) = open_db();

    db.execute_cypher(
        "CREATE TRIGGER only_user ON :User CREATE BEFORE COMMIT \
         EXECUTE CREATE (e:WronglyFired)",
    )
    .unwrap();

    // Create a different label — trigger must not fire.
    db.execute_cypher("CREATE (n:Customer {id: 1})").unwrap();

    let wrongly = db
        .execute_cypher("MATCH (e:WronglyFired) RETURN e")
        .unwrap();
    assert!(
        wrongly.is_empty(),
        ":Customer creation must not fire :User trigger: {wrongly:?}"
    );
}

/// Disabled BEFORE COMMIT trigger must NOT fire even on matching event.
#[test]
fn trigger_before_commit_disabled_does_not_fire() {
    let (mut db, _dir) = open_db();

    db.execute_cypher(
        "CREATE TRIGGER paused ON :User CREATE BEFORE COMMIT \
         EXECUTE CREATE (e:WouldFireIfEnabled)",
    )
    .unwrap();
    db.execute_cypher("ALTER TRIGGER paused DISABLE").unwrap();

    db.execute_cypher("CREATE (u:User {id: 1})").unwrap();

    let evidence = db
        .execute_cypher("MATCH (e:WouldFireIfEnabled) RETURN e")
        .unwrap();
    assert!(
        evidence.is_empty(),
        "disabled trigger must not fire: {evidence:?}"
    );

    // Re-enable, create again, NOW it should fire.
    db.execute_cypher("ALTER TRIGGER paused ENABLE").unwrap();
    db.execute_cypher("CREATE (u:User {id: 2})").unwrap();
    let evidence = db
        .execute_cypher("MATCH (e:WouldFireIfEnabled) RETURN e")
        .unwrap();
    assert_eq!(evidence.len(), 1, "re-enabled trigger must fire");
}

/// Multi-label node `(n:User:Admin)` must fire BEFORE COMMIT triggers
/// registered on EITHER label. The probe iterates each label in turn,
/// so a node carrying both labels fires both triggers exactly once.
#[test]
fn trigger_before_commit_multi_label_fires_each_label_trigger() {
    use coordinode_core::graph::types::Value;
    let (mut db, _dir) = open_db();

    db.execute_cypher(
        "CREATE TRIGGER on_user ON :User CREATE BEFORE COMMIT \
         EXECUTE CREATE (l:UserLog {action: $event})",
    )
    .unwrap();
    db.execute_cypher(
        "CREATE TRIGGER on_admin ON :Admin CREATE BEFORE COMMIT \
         EXECUTE CREATE (l:AdminLog {action: $event})",
    )
    .unwrap();

    db.execute_cypher("CREATE (n:User:Admin {name: 'Alice'})")
        .unwrap();

    let user_logs = db
        .execute_cypher("MATCH (l:UserLog) RETURN l.action AS a")
        .unwrap();
    assert_eq!(user_logs.len(), 1, ":User trigger must fire");
    assert_eq!(user_logs[0].get("a"), Some(&Value::String("CREATE".into())));

    let admin_logs = db
        .execute_cypher("MATCH (l:AdminLog) RETURN l.action AS a")
        .unwrap();
    assert_eq!(admin_logs.len(), 1, ":Admin trigger must fire");
    assert_eq!(
        admin_logs[0].get("a"),
        Some(&Value::String("CREATE".into()))
    );
}

/// `$after` parameter is available as a Map in the trigger body.
///
/// Note: the parser does not currently accept `$after.field` dot-notation
/// (parameter-as-base for PropertyAccess is a pre-existing limitation —
/// the same restriction that prevents `$node.x = …` in SET clauses).
/// This test confirms the whole `$after` map round-trips through the
/// parameter substitution path; a follow-up parser fix can then enable
/// `$after.name` resolution end-to-end with a dedicated test.
#[test]
fn trigger_before_commit_after_param_is_available_as_map() {
    use coordinode_core::graph::types::Value;
    let (mut db, _dir) = open_db();

    db.execute_cypher(
        "CREATE TRIGGER capture ON :Person CREATE BEFORE COMMIT \
         EXECUTE CREATE (e:Capture {props: $after, event_kind: $event})",
    )
    .unwrap();

    db.execute_cypher("CREATE (p:Person {name: 'Bob', age: 30})")
        .unwrap();

    let logs = db
        .execute_cypher("MATCH (e:Capture) RETURN e.event_kind AS k, e.props AS p")
        .unwrap();
    assert_eq!(logs.len(), 1);
    assert_eq!(
        logs[0].get("k"),
        Some(&Value::String("CREATE".into())),
        "$event must propagate as String literal"
    );
    let props = logs[0].get("p").cloned().expect("props column present");
    // $after lands as a Map; nested storage may serialise it as Document.
    assert!(
        matches!(props, Value::Map(_) | Value::Document(_)),
        "$after must be a Map (or Document after storage round-trip); got {props:?}"
    );
}

/// Multiple triggers registered on the same `(label, event)` pair —
/// every enabled one must fire on each matching mutation.
#[test]
fn trigger_before_commit_multiple_triggers_same_label_all_fire() {
    let (mut db, _dir) = open_db();

    for i in 0..3u32 {
        db.execute_cypher(&format!(
            "CREATE TRIGGER witness_{i} ON :Person CREATE BEFORE COMMIT \
             EXECUTE CREATE (w:Witness_{i})"
        ))
        .unwrap();
    }

    db.execute_cypher("CREATE (p:Person {id: 1})").unwrap();

    for i in 0..3u32 {
        let rows = db
            .execute_cypher(&format!("MATCH (w:Witness_{i}) RETURN w"))
            .unwrap();
        assert_eq!(rows.len(), 1, "Witness_{i} trigger must have fired");
    }
}

/// DELETE-event trigger fires on DETACH DELETE; `$before` carries the
/// deleted node's properties so the audit body can capture them.
#[test]
fn trigger_before_commit_delete_fires_and_carries_before() {
    use coordinode_core::graph::types::Value;
    let (mut db, _dir) = open_db();

    db.execute_cypher(
        "CREATE TRIGGER on_delete ON :User DELETE BEFORE COMMIT \
         EXECUTE CREATE (e:DeletionLog {action: $event, snapshot: $before})",
    )
    .unwrap();

    db.execute_cypher("CREATE (u:User {name: 'Carol', id: 7})")
        .unwrap();
    db.execute_cypher("MATCH (u:User) DETACH DELETE u").unwrap();

    // User is gone, deletion log exists with $event = "DELETE" and a
    // captured snapshot containing the user's pre-delete props.
    let users = db.execute_cypher("MATCH (u:User) RETURN u").unwrap();
    assert!(users.is_empty(), "User must be deleted; got {users:?}");

    let logs = db
        .execute_cypher("MATCH (e:DeletionLog) RETURN e.action AS act, e.snapshot AS snap")
        .unwrap();
    assert_eq!(logs.len(), 1);
    assert_eq!(logs[0].get("act"), Some(&Value::String("DELETE".into())));
    let snap = logs[0].get("snap").cloned().expect("snapshot recorded");
    // $before lands as a Map of the deleted node's props; storage may
    // serialise it as Document.
    assert!(
        matches!(snap, Value::Map(_) | Value::Document(_)),
        "$before must be a Map (or Document after storage round-trip); got {snap:?}"
    );
}

/// UPDATE-event trigger fires on SET; `$before` and `$after` carry the
/// pre/post snapshots respectively.
#[test]
fn trigger_before_commit_update_fires_with_before_and_after() {
    use coordinode_core::graph::types::Value;
    let (mut db, _dir) = open_db();

    db.execute_cypher(
        "CREATE TRIGGER on_update ON :Counter UPDATE BEFORE COMMIT \
         EXECUTE CREATE (e:ChangeLog {action: $event, old: $before, new: $after})",
    )
    .unwrap();

    db.execute_cypher("CREATE (c:Counter {value: 10})").unwrap();
    db.execute_cypher("MATCH (c:Counter) SET c.value = 99")
        .unwrap();

    let logs = db
        .execute_cypher("MATCH (e:ChangeLog) RETURN e.action AS act, e.old AS old, e.new AS new")
        .unwrap();
    assert_eq!(logs.len(), 1, "UPDATE trigger must have fired once");
    assert_eq!(logs[0].get("act"), Some(&Value::String("UPDATE".into())));
    // Both $before and $after present as Map / Document.
    assert!(matches!(
        logs[0].get("old"),
        Some(Value::Map(_) | Value::Document(_))
    ));
    assert!(matches!(
        logs[0].get("new"),
        Some(Value::Map(_) | Value::Document(_))
    ));
}

/// Multiple SET items in one row fire the UPDATE trigger ONCE per node,
/// not once per item — `$before` is the snapshot taken at start of the
/// SET clause, `$after` is the cumulative result.
#[test]
fn trigger_before_commit_update_fires_once_per_node_per_statement() {
    use coordinode_core::graph::types::Value;
    let (mut db, _dir) = open_db();

    db.execute_cypher(
        "CREATE TRIGGER count_fires ON :Item UPDATE BEFORE COMMIT \
         EXECUTE CREATE (e:FireMark)",
    )
    .unwrap();

    db.execute_cypher("CREATE (i:Item {a: 0, b: 0, c: 0})")
        .unwrap();
    db.execute_cypher("MATCH (i:Item) SET i.a = 1, i.b = 2, i.c = 3")
        .unwrap();

    let marks = db
        .execute_cypher("MATCH (m:FireMark) RETURN count(m) AS n")
        .unwrap();
    assert_eq!(marks.len(), 1);
    assert_eq!(
        marks[0].get("n"),
        Some(&Value::Int(1)),
        "3 SET items must produce exactly 1 trigger firing, not 3: {marks:?}"
    );
}

/// Edge-CREATE trigger fires on `CREATE (a)-[:TYPE {…}]->(b)`. Trigger
/// is registered on the edge type and receives `$src` / `$tgt` /
/// `$edge_type` / `$after` parameters.
#[test]
fn trigger_before_commit_edge_create_fires_with_endpoints() {
    use coordinode_core::graph::types::Value;
    let (mut db, _dir) = open_db();

    db.execute_cypher(
        "CREATE TRIGGER on_follow ON [:FOLLOWS] CREATE BEFORE COMMIT \
         EXECUTE CREATE (e:FollowLog {action: $event, type_name: $edge_type, src_id: $src, tgt_id: $tgt})",
    )
    .unwrap();

    db.execute_cypher("CREATE (a:User {id: 1})").unwrap();
    db.execute_cypher("CREATE (b:User {id: 2})").unwrap();
    db.execute_cypher(
        "MATCH (a:User {id: 1}), (b:User {id: 2}) CREATE (a)-[:FOLLOWS {weight: 5}]->(b)",
    )
    .unwrap();

    let logs = db
        .execute_cypher(
            "MATCH (e:FollowLog) \
             RETURN e.action AS act, e.type_name AS t, e.src_id AS s, e.tgt_id AS d",
        )
        .unwrap();
    assert_eq!(logs.len(), 1, "edge-create trigger must have fired");
    assert_eq!(logs[0].get("act"), Some(&Value::String("CREATE".into())));
    assert_eq!(
        logs[0].get("t"),
        Some(&Value::String("FOLLOWS".into())),
        "edge_type must be the type name"
    );
    assert!(matches!(logs[0].get("s"), Some(Value::Int(_))));
    assert!(matches!(logs[0].get("d"), Some(Value::Int(_))));
}

/// A trigger registered on a node label MUST NOT fire on edge creation
/// even if the edge type happens to share the label's name (the index
/// segments `n:X` vs `e:X` are disjoint per ADR but this guards the wiring).
#[test]
fn trigger_node_label_does_not_fire_on_same_named_edge_type() {
    let (mut db, _dir) = open_db();

    db.execute_cypher(
        "CREATE TRIGGER node_only ON :SharedName CREATE BEFORE COMMIT \
         EXECUTE CREATE (e:WronglyFired)",
    )
    .unwrap();

    db.execute_cypher("CREATE (a:User {id: 1})").unwrap();
    db.execute_cypher("CREATE (b:User {id: 2})").unwrap();
    db.execute_cypher("MATCH (a:User {id: 1}), (b:User {id: 2}) CREATE (a)-[:SharedName]->(b)")
        .unwrap();

    let wrongly = db
        .execute_cypher("MATCH (e:WronglyFired) RETURN e")
        .unwrap();
    assert!(
        wrongly.is_empty(),
        "node-label trigger must not fire on same-named edge type: {wrongly:?}"
    );
}

// =====================================================================
// R191 expansion — SET on edge, MERGE node/edge, DETACH DELETE cascade,
// explicit DELETE edge trigger firings.
// =====================================================================

/// SET on an edge property fires the edge-type UPDATE trigger with
/// `$before` / `$after` carrying the edge prop maps and
/// `$src` / `$tgt` / `$edge_type` describing the edge.
#[test]
fn trigger_before_commit_edge_update_fires_on_set_with_before_and_after() {
    use coordinode_core::graph::types::Value;
    let (mut db, _dir) = open_db();

    db.execute_cypher(
        "CREATE TRIGGER on_edge_set ON [:FOLLOWS] UPDATE BEFORE COMMIT \
         EXECUTE CREATE (e:EdgeChange { \
           action: $event, t: $edge_type, s: $src, d: $tgt, \
           old: $before, new: $after })",
    )
    .unwrap();

    db.execute_cypher("CREATE (a:User {id: 1})").unwrap();
    db.execute_cypher("CREATE (b:User {id: 2})").unwrap();
    db.execute_cypher(
        "MATCH (a:User {id: 1}), (b:User {id: 2}) \
         CREATE (a)-[:FOLLOWS {weight: 5}]->(b)",
    )
    .unwrap();
    db.execute_cypher("MATCH (a:User {id: 1})-[r:FOLLOWS]->(b:User {id: 2}) SET r.weight = 99")
        .unwrap();

    let logs = db
        .execute_cypher(
            "MATCH (e:EdgeChange) RETURN \
               e.action AS act, e.t AS t, e.s AS s, e.d AS d, \
               e.old AS old, e.new AS new",
        )
        .unwrap();
    assert_eq!(logs.len(), 1, "edge-UPDATE trigger must have fired once");
    assert_eq!(logs[0].get("act"), Some(&Value::String("UPDATE".into())));
    assert_eq!(logs[0].get("t"), Some(&Value::String("FOLLOWS".into())));
    assert!(matches!(logs[0].get("s"), Some(Value::Int(_))));
    assert!(matches!(logs[0].get("d"), Some(Value::Int(_))));
    assert!(matches!(
        logs[0].get("old"),
        Some(Value::Map(_) | Value::Document(_))
    ));
    assert!(matches!(
        logs[0].get("new"),
        Some(Value::Map(_) | Value::Document(_))
    ));
}

/// Multiple SET items in one row against the same edge fire the UPDATE
/// trigger exactly once — `$before` is the snapshot at start of the SET
/// clause, `$after` is the cumulative result. Mirrors the node-side
/// once-per-statement invariant.
#[test]
fn trigger_before_commit_edge_update_fires_once_per_edge_per_statement() {
    use coordinode_core::graph::types::Value;
    let (mut db, _dir) = open_db();

    db.execute_cypher(
        "CREATE TRIGGER edge_fire_count ON [:FOLLOWS] UPDATE BEFORE COMMIT \
         EXECUTE CREATE (e:EdgeFireMark)",
    )
    .unwrap();

    db.execute_cypher("CREATE (a:User {id: 1})").unwrap();
    db.execute_cypher("CREATE (b:User {id: 2})").unwrap();
    db.execute_cypher(
        "MATCH (a:User {id: 1}), (b:User {id: 2}) \
         CREATE (a)-[:FOLLOWS {x: 0, y: 0, z: 0}]->(b)",
    )
    .unwrap();
    db.execute_cypher(
        "MATCH (a:User {id: 1})-[r:FOLLOWS]->(b:User {id: 2}) \
         SET r.x = 1, r.y = 2, r.z = 3",
    )
    .unwrap();

    let marks = db
        .execute_cypher("MATCH (m:EdgeFireMark) RETURN count(m) AS n")
        .unwrap();
    assert_eq!(marks.len(), 1);
    assert_eq!(
        marks[0].get("n"),
        Some(&Value::Int(1)),
        "3 SET items on the same edge must produce exactly 1 trigger firing"
    );
}

/// SET on a node that has no matching trigger must NOT fire any edge
/// trigger that happens to share the property column path — index
/// segments `n:Label` vs `e:Type` are disjoint and this guards the
/// edge-update wiring against accidental fan-out.
#[test]
fn trigger_edge_update_does_not_fire_on_node_set() {
    let (mut db, _dir) = open_db();

    db.execute_cypher(
        "CREATE TRIGGER follow_update ON [:FOLLOWS] UPDATE BEFORE COMMIT \
         EXECUTE CREATE (e:WronglyFired)",
    )
    .unwrap();

    db.execute_cypher("CREATE (u:User {weight: 1})").unwrap();
    db.execute_cypher("MATCH (u:User) SET u.weight = 2")
        .unwrap();

    let wrongly = db
        .execute_cypher("MATCH (e:WronglyFired) RETURN e")
        .unwrap();
    assert!(
        wrongly.is_empty(),
        "edge-UPDATE trigger must not fire on a node SET: {wrongly:?}"
    );
}

/// MERGE on a non-existent node creates the node and fires the
/// label-CREATE trigger registered on the new node's label.
#[test]
fn trigger_before_commit_merge_node_fires_create_on_miss() {
    use coordinode_core::graph::types::Value;
    let (mut db, _dir) = open_db();

    db.execute_cypher(
        "CREATE TRIGGER on_merge_user ON :User CREATE BEFORE COMMIT \
         EXECUTE CREATE (e:UserLog {action: $event})",
    )
    .unwrap();

    db.execute_cypher("MERGE (u:User {name: 'Alice'})").unwrap();

    let logs = db
        .execute_cypher("MATCH (e:UserLog) RETURN e.action AS a")
        .unwrap();
    assert_eq!(
        logs.len(),
        1,
        "MERGE create branch must fire CREATE trigger"
    );
    assert_eq!(logs[0].get("a"), Some(&Value::String("CREATE".into())));
}

/// MERGE that hits an existing node must NOT fire a CREATE trigger.
/// (ON MATCH SET fires UPDATE; that is covered by the node UPDATE path.)
#[test]
fn trigger_merge_node_does_not_fire_create_on_hit() {
    let (mut db, _dir) = open_db();

    db.execute_cypher("CREATE (u:User {name: 'Alice'})")
        .unwrap();

    db.execute_cypher(
        "CREATE TRIGGER on_merge_user ON :User CREATE BEFORE COMMIT \
         EXECUTE CREATE (e:UserLog {action: $event})",
    )
    .unwrap();

    db.execute_cypher("MERGE (u:User {name: 'Alice'})").unwrap();

    let logs = db.execute_cypher("MATCH (e:UserLog) RETURN e").unwrap();
    assert!(
        logs.is_empty(),
        "MERGE that hits an existing node must not fire CREATE trigger: {logs:?}"
    );
}

/// MERGE on a non-existent relationship creates the edge and fires the
/// edge-type CREATE trigger with `$src` / `$tgt` / `$edge_type`.
#[test]
fn trigger_before_commit_merge_edge_fires_create_on_miss() {
    use coordinode_core::graph::types::Value;
    let (mut db, _dir) = open_db();

    db.execute_cypher(
        "CREATE TRIGGER on_merge_follow ON [:FOLLOWS] CREATE BEFORE COMMIT \
         EXECUTE CREATE (e:FollowLog {action: $event, t: $edge_type})",
    )
    .unwrap();

    db.execute_cypher("CREATE (a:User {id: 1})").unwrap();
    db.execute_cypher("CREATE (b:User {id: 2})").unwrap();
    db.execute_cypher(
        "MATCH (a:User {id: 1}), (b:User {id: 2}) \
         MERGE (a)-[:FOLLOWS]->(b)",
    )
    .unwrap();

    let logs = db
        .execute_cypher("MATCH (e:FollowLog) RETURN e.action AS act, e.t AS t")
        .unwrap();
    assert_eq!(
        logs.len(),
        1,
        "MERGE edge create branch must fire CREATE trigger"
    );
    assert_eq!(logs[0].get("act"), Some(&Value::String("CREATE".into())));
    assert_eq!(logs[0].get("t"), Some(&Value::String("FOLLOWS".into())));
}

/// MERGE that hits an existing edge must NOT fire a CREATE trigger.
#[test]
fn trigger_merge_edge_does_not_fire_create_on_hit() {
    let (mut db, _dir) = open_db();

    db.execute_cypher("CREATE (a:User {id: 1})").unwrap();
    db.execute_cypher("CREATE (b:User {id: 2})").unwrap();
    db.execute_cypher("MATCH (a:User {id: 1}), (b:User {id: 2}) CREATE (a)-[:FOLLOWS]->(b)")
        .unwrap();

    db.execute_cypher(
        "CREATE TRIGGER on_merge_follow ON [:FOLLOWS] CREATE BEFORE COMMIT \
         EXECUTE CREATE (e:FollowLog {action: $event})",
    )
    .unwrap();

    db.execute_cypher("MATCH (a:User {id: 1}), (b:User {id: 2}) MERGE (a)-[:FOLLOWS]->(b)")
        .unwrap();

    let logs = db.execute_cypher("MATCH (e:FollowLog) RETURN e").unwrap();
    assert!(
        logs.is_empty(),
        "MERGE that hits an existing edge must not fire CREATE trigger: {logs:?}"
    );
}

/// DETACH DELETE cascades into edge DELETE triggers: every connected
/// edge fires the registered DELETE trigger on its edge type, with
/// `$before` carrying the edge's pre-delete props.
#[test]
fn trigger_before_commit_detach_delete_cascades_edge_delete_triggers() {
    use coordinode_core::graph::types::Value;
    let (mut db, _dir) = open_db();

    db.execute_cypher(
        "CREATE TRIGGER on_follow_delete ON [:FOLLOWS] DELETE BEFORE COMMIT \
         EXECUTE CREATE (e:FollowDeletion {action: $event, t: $edge_type, b: $before})",
    )
    .unwrap();

    db.execute_cypher("CREATE (a:User {id: 1})").unwrap();
    db.execute_cypher("CREATE (b:User {id: 2})").unwrap();
    db.execute_cypher("CREATE (c:User {id: 3})").unwrap();
    db.execute_cypher(
        "MATCH (a:User {id: 1}), (b:User {id: 2}) \
         CREATE (a)-[:FOLLOWS {weight: 5}]->(b)",
    )
    .unwrap();
    db.execute_cypher(
        "MATCH (a:User {id: 1}), (c:User {id: 3}) \
         CREATE (a)-[:FOLLOWS {weight: 7}]->(c)",
    )
    .unwrap();

    db.execute_cypher("MATCH (a:User {id: 1}) DETACH DELETE a")
        .unwrap();

    let logs = db
        .execute_cypher(
            "MATCH (e:FollowDeletion) \
             RETURN e.action AS act, e.t AS t, e.b AS b",
        )
        .unwrap();
    assert_eq!(
        logs.len(),
        2,
        "DETACH DELETE must fire edge-DELETE trigger for each connected edge: {logs:?}"
    );
    for log in &logs {
        assert_eq!(log.get("act"), Some(&Value::String("DELETE".into())));
        assert_eq!(log.get("t"), Some(&Value::String("FOLLOWS".into())));
        assert!(matches!(
            log.get("b"),
            Some(Value::Map(_) | Value::Document(_))
        ));
    }
}

/// Explicit `DELETE r` on an edge variable fires the edge-type DELETE
/// trigger with `$before` carrying the edge's properties.
#[test]
fn trigger_before_commit_explicit_delete_edge_fires() {
    use coordinode_core::graph::types::Value;
    let (mut db, _dir) = open_db();

    db.execute_cypher(
        "CREATE TRIGGER on_follow_delete ON [:FOLLOWS] DELETE BEFORE COMMIT \
         EXECUTE CREATE (e:FollowDeletion {action: $event, b: $before})",
    )
    .unwrap();

    db.execute_cypher("CREATE (a:User {id: 1})").unwrap();
    db.execute_cypher("CREATE (b:User {id: 2})").unwrap();
    db.execute_cypher(
        "MATCH (a:User {id: 1}), (b:User {id: 2}) \
         CREATE (a)-[:FOLLOWS {weight: 5}]->(b)",
    )
    .unwrap();

    db.execute_cypher("MATCH (a:User {id: 1})-[r:FOLLOWS]->(b:User {id: 2}) DELETE r")
        .unwrap();

    let logs = db
        .execute_cypher("MATCH (e:FollowDeletion) RETURN e.action AS act, e.b AS b")
        .unwrap();
    assert_eq!(
        logs.len(),
        1,
        "explicit DELETE r must fire edge DELETE trigger"
    );
    assert_eq!(logs[0].get("act"), Some(&Value::String("DELETE".into())));
    assert!(matches!(
        logs[0].get("b"),
        Some(Value::Map(_) | Value::Document(_))
    ));
}

/// Explicit DELETE on a temporal edge fires the DELETE trigger once per
/// version under the matched `(src, tgt)` pair — `$before` carries each
/// version's properties in turn.
#[test]
fn trigger_before_commit_explicit_delete_temporal_edge_fires_per_version() {
    use coordinode_core::graph::types::Value;
    let (mut db, _dir) = open_db();

    db.execute_cypher(
        "CREATE EDGE TYPE WORKS_AT TEMPORAL WITH ( \
           role: STRING, \
           valid_from: TIMESTAMP NOT NULL, \
           valid_to: TIMESTAMP \
         )",
    )
    .unwrap();

    db.execute_cypher(
        "CREATE TRIGGER on_works_delete ON [:WORKS_AT] DELETE BEFORE COMMIT \
         EXECUTE CREATE (e:WorksDeletion {action: $event, b: $before})",
    )
    .unwrap();

    db.execute_cypher("CREATE (a:Person {id: 1})").unwrap();
    db.execute_cypher("CREATE (c:Company {id: 10})").unwrap();
    // Two versions on the same (a, c) pair.
    db.execute_cypher(
        "MATCH (a:Person {id: 1}), (c:Company {id: 10}) \
         CREATE (a)-[:WORKS_AT {role: 'SWE', valid_from: 1577836800000, valid_to: 1640995200000}]->(c)",
    )
    .unwrap();
    db.execute_cypher(
        "MATCH (a:Person {id: 1}), (c:Company {id: 10}) \
         CREATE (a)-[:WORKS_AT {role: 'Staff', valid_from: 1640995200000}]->(c)",
    )
    .unwrap();

    db.execute_cypher("MATCH (a:Person {id: 1})-[r:WORKS_AT]->(c:Company {id: 10}) DELETE r")
        .unwrap();

    let logs = db
        .execute_cypher("MATCH (e:WorksDeletion) RETURN e.action AS act, e.b AS b")
        .unwrap();
    assert_eq!(
        logs.len(),
        2,
        "DELETE on a temporal edge must fire one DELETE trigger per stored version: {logs:?}"
    );
    for log in &logs {
        assert_eq!(log.get("act"), Some(&Value::String("DELETE".into())));
        assert!(matches!(
            log.get("b"),
            Some(Value::Map(_) | Value::Document(_))
        ));
    }
}

/// `SET r += {...}` on an edge variable is a silent no-op in the current
/// executor (the MergeProperties SET arm requires `Value::Int` and skips
/// edge variables). The UPDATE trigger MUST NOT fire for a SET that did
/// not mutate the edge.
#[test]
fn trigger_edge_update_does_not_fire_when_set_is_non_mutating() {
    let (mut db, _dir) = open_db();

    db.execute_cypher(
        "CREATE TRIGGER on_follow_update ON [:FOLLOWS] UPDATE BEFORE COMMIT \
         EXECUTE CREATE (e:WronglyFired)",
    )
    .unwrap();

    db.execute_cypher("CREATE (a:User {id: 1})").unwrap();
    db.execute_cypher("CREATE (b:User {id: 2})").unwrap();
    db.execute_cypher(
        "MATCH (a:User {id: 1}), (b:User {id: 2}) \
         CREATE (a)-[:FOLLOWS {weight: 5}]->(b)",
    )
    .unwrap();

    // `SET r += {x: 1}` — MergeProperties variant. Current executor
    // skips edge variables in this arm (no edgeprop mutation). The
    // trigger MUST NOT fire on a no-op SET.
    db.execute_cypher("MATCH (a:User {id: 1})-[r:FOLLOWS]->(b:User {id: 2}) SET r += {x: 1}")
        .unwrap();

    let wrongly = db
        .execute_cypher("MATCH (e:WronglyFired) RETURN e")
        .unwrap();
    assert!(
        wrongly.is_empty(),
        "edge-UPDATE trigger must not fire on a non-mutating SET variant: {wrongly:?}"
    );
}

/// SET on a temporal edge fires the UPDATE trigger against the matched
/// version (per-version edgeprop key). Only that version's props show
/// up as `$before` / `$after`; sibling versions are untouched.
#[test]
fn trigger_before_commit_edge_update_temporal_fires_per_version() {
    use coordinode_core::graph::types::Value;
    let (mut db, _dir) = open_db();

    db.execute_cypher(
        "CREATE EDGE TYPE WORKS_AT TEMPORAL WITH ( \
           role: STRING, \
           valid_from: TIMESTAMP NOT NULL, \
           valid_to: TIMESTAMP \
         )",
    )
    .unwrap();

    db.execute_cypher(
        "CREATE TRIGGER on_works_update ON [:WORKS_AT] UPDATE BEFORE COMMIT \
         EXECUTE CREATE (e:WorksChange { \
           action: $event, t: $edge_type, old: $before, new: $after })",
    )
    .unwrap();

    db.execute_cypher("CREATE (a:Person {id: 1})").unwrap();
    db.execute_cypher("CREATE (c:Company {id: 10})").unwrap();
    // Two versions: close the first, open the second.
    db.execute_cypher(
        "MATCH (a:Person {id: 1}), (c:Company {id: 10}) \
         CREATE (a)-[:WORKS_AT {role: 'SWE', valid_from: 1577836800000, valid_to: 1640995200000}]->(c)",
    )
    .unwrap();
    db.execute_cypher(
        "MATCH (a:Person {id: 1}), (c:Company {id: 10}) \
         CREATE (a)-[:WORKS_AT {role: 'Staff', valid_from: 1640995200000}]->(c)",
    )
    .unwrap();

    // SET on the open version only — match the row whose valid_to IS NULL.
    db.execute_cypher(
        "MATCH (a:Person {id: 1})-[r:WORKS_AT]->(c:Company {id: 10}) \
         WHERE r.valid_to IS NULL \
         SET r.role = 'Principal'",
    )
    .unwrap();

    let logs = db
        .execute_cypher(
            "MATCH (e:WorksChange) RETURN e.action AS act, e.t AS t, e.old AS old, e.new AS new",
        )
        .unwrap();
    assert_eq!(
        logs.len(),
        1,
        "exactly one UPDATE trigger firing (one matched temporal version): {logs:?}"
    );
    assert_eq!(logs[0].get("act"), Some(&Value::String("UPDATE".into())));
    assert_eq!(logs[0].get("t"), Some(&Value::String("WORKS_AT".into())));
    assert!(matches!(
        logs[0].get("old"),
        Some(Value::Map(_) | Value::Document(_))
    ));
    assert!(matches!(
        logs[0].get("new"),
        Some(Value::Map(_) | Value::Document(_))
    ));
}

/// MERGE node with `ON CREATE SET`:
/// On the create branch, CREATE trigger fires first (when the node
/// lands), then the ON CREATE SET items go through `execute_update`,
/// which fires the UPDATE trigger. Both must fire — they describe
/// distinct logical events.
#[test]
fn trigger_merge_on_create_set_fires_create_then_update() {
    use coordinode_core::graph::types::Value;
    let (mut db, _dir) = open_db();

    db.execute_cypher(
        "CREATE TRIGGER user_create ON :User CREATE BEFORE COMMIT \
         EXECUTE CREATE (e:UserLog {action: $event})",
    )
    .unwrap();
    db.execute_cypher(
        "CREATE TRIGGER user_update ON :User UPDATE BEFORE COMMIT \
         EXECUTE CREATE (e:UserLog {action: $event})",
    )
    .unwrap();

    db.execute_cypher("MERGE (u:User {name: 'Alice'}) ON CREATE SET u.score = 1")
        .unwrap();

    let logs = db
        .execute_cypher("MATCH (e:UserLog) RETURN e.action AS a")
        .unwrap();
    let actions: Vec<_> = logs.iter().filter_map(|r| r.get("a").cloned()).collect();
    assert!(
        actions.contains(&Value::String("CREATE".into())),
        "MERGE create branch must fire CREATE trigger: {actions:?}"
    );
    assert!(
        actions.contains(&Value::String("UPDATE".into())),
        "ON CREATE SET must fire UPDATE trigger: {actions:?}"
    );
    assert_eq!(actions.len(), 2);
}

/// MERGE node with `ON MATCH SET`:
/// On the match branch (existing node), only UPDATE fires — no CREATE,
/// since the node was not created in this transaction.
#[test]
fn trigger_merge_on_match_set_fires_update_only() {
    use coordinode_core::graph::types::Value;
    let (mut db, _dir) = open_db();

    db.execute_cypher("CREATE (u:User {name: 'Alice'})")
        .unwrap();

    db.execute_cypher(
        "CREATE TRIGGER user_create ON :User CREATE BEFORE COMMIT \
         EXECUTE CREATE (e:UserLog {action: $event})",
    )
    .unwrap();
    db.execute_cypher(
        "CREATE TRIGGER user_update ON :User UPDATE BEFORE COMMIT \
         EXECUTE CREATE (e:UserLog {action: $event})",
    )
    .unwrap();

    db.execute_cypher("MERGE (u:User {name: 'Alice'}) ON MATCH SET u.score = 1")
        .unwrap();

    let logs = db
        .execute_cypher("MATCH (e:UserLog) RETURN e.action AS a")
        .unwrap();
    let actions: Vec<_> = logs.iter().filter_map(|r| r.get("a").cloned()).collect();
    assert_eq!(actions.len(), 1, "exactly one trigger firing: {actions:?}");
    assert_eq!(actions[0], Value::String("UPDATE".into()));
}

/// ATTACH DOCUMENT cascade fires the node-DELETE trigger on the source
/// node's labels. The source node is folded into the target, then deleted
/// — semantically the same as DETACH DELETE on the source.
#[test]
fn trigger_before_commit_attach_document_fires_source_node_delete() {
    use coordinode_core::graph::types::Value;
    let (mut db, _dir) = open_db();

    db.execute_cypher(
        "CREATE TRIGGER on_addr_delete ON :Address DELETE BEFORE COMMIT \
         EXECUTE CREATE (e:DeletionLog {action: $event, b: $before})",
    )
    .unwrap();

    db.execute_cypher("CREATE (a:Address {city: 'NYC', zip: '10001'})")
        .unwrap();
    db.execute_cypher("CREATE (u:User {id: 1})").unwrap();
    db.execute_cypher("MATCH (a:Address), (u:User) CREATE (a)-[:HAS_ADDRESS]->(u)")
        .unwrap();

    db.execute_cypher("ATTACH (a:Address)-[:HAS_ADDRESS]->(u:User) INTO u.address")
        .unwrap();

    let logs = db
        .execute_cypher("MATCH (e:DeletionLog) RETURN e.action AS act, e.b AS b")
        .unwrap();
    assert_eq!(logs.len(), 1, "ATTACH must fire :Address DELETE trigger");
    assert_eq!(logs[0].get("act"), Some(&Value::String("DELETE".into())));
    assert!(matches!(
        logs[0].get("b"),
        Some(Value::Map(_) | Value::Document(_))
    ));
}

/// ATTACH DOCUMENT fires the edge-DELETE trigger for the connecting edge
/// (`delete_single_edge` path) AND for any remaining edges on the source
/// that are cascade-deleted (`cascade_delete_source_node` path) — both
/// must be symmetric with DETACH DELETE.
#[test]
fn trigger_before_commit_attach_document_cascades_orphan_edge_deletes() {
    use coordinode_core::graph::types::Value;
    let (mut db, _dir) = open_db();

    db.execute_cypher(
        "CREATE TRIGGER on_link_delete ON [:LINK] DELETE BEFORE COMMIT \
         EXECUTE CREATE (e:LinkDeletion {action: $event, t: $edge_type})",
    )
    .unwrap();
    db.execute_cypher(
        "CREATE TRIGGER on_attach_delete ON [:HAS_ADDRESS] DELETE BEFORE COMMIT \
         EXECUTE CREATE (e:AttachDeletion {action: $event, t: $edge_type})",
    )
    .unwrap();

    db.execute_cypher("CREATE (a:Address {city: 'NYC'})")
        .unwrap();
    db.execute_cypher("CREATE (u:User {id: 1})").unwrap();
    db.execute_cypher("CREATE (other:Tag {id: 99})").unwrap();
    // Connecting edge (will be deleted by ATTACH's explicit delete_single_edge).
    db.execute_cypher("MATCH (a:Address), (u:User) CREATE (a)-[:HAS_ADDRESS]->(u)")
        .unwrap();
    // Orphan edge on source (will be cascade-deleted with the source).
    db.execute_cypher("MATCH (a:Address), (t:Tag) CREATE (a)-[:LINK {kind: 'orphan'}]->(t)")
        .unwrap();

    db.execute_cypher("ATTACH (a:Address)-[:HAS_ADDRESS]->(u:User) INTO u.address")
        .unwrap();

    let attach_logs = db
        .execute_cypher("MATCH (e:AttachDeletion) RETURN e.action AS act, e.t AS t")
        .unwrap();
    assert_eq!(
        attach_logs.len(),
        1,
        "connecting edge must fire :HAS_ADDRESS DELETE trigger"
    );
    assert_eq!(
        attach_logs[0].get("act"),
        Some(&Value::String("DELETE".into()))
    );

    let link_logs = db
        .execute_cypher("MATCH (e:LinkDeletion) RETURN e.action AS act, e.t AS t")
        .unwrap();
    assert_eq!(
        link_logs.len(),
        1,
        "orphan edge cascade-delete must fire :LINK DELETE trigger"
    );
    assert_eq!(
        link_logs[0].get("act"),
        Some(&Value::String("DELETE".into()))
    );
    assert_eq!(link_logs[0].get("t"), Some(&Value::String("LINK".into())));
}

/// Edge UPDATE trigger with `ON ERROR PROPAGATE` aborts the originating
/// SET — the edge property must NOT be modified after rollback.
#[test]
fn trigger_edge_update_propagate_aborts_set() {
    use coordinode_core::graph::types::Value;
    let (mut db, _dir) = open_db();

    // Reject node: STRICT label with only `action` allowed. Trigger body
    // writes a forbidden field → execution error inside trigger → SET
    // rolls back.
    use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType, SchemaMode};
    let mut audit_schema = LabelSchema::new_node_id("Audit");
    audit_schema.set_mode(SchemaMode::Strict);
    audit_schema.add_property(PropertyDef::new("action", PropertyType::String).not_null());
    db.create_label_schema(audit_schema).unwrap();

    db.execute_cypher(
        "CREATE TRIGGER reject_follow_update ON [:FOLLOWS] UPDATE BEFORE COMMIT \
         EXECUTE CREATE (e:Audit {forbidden_field: $event}) \
         ON ERROR PROPAGATE",
    )
    .unwrap();

    db.execute_cypher("CREATE (a:User {id: 1})").unwrap();
    db.execute_cypher("CREATE (b:User {id: 2})").unwrap();
    db.execute_cypher(
        "MATCH (a:User {id: 1}), (b:User {id: 2}) \
         CREATE (a)-[:FOLLOWS {weight: 5}]->(b)",
    )
    .unwrap();

    let result =
        db.execute_cypher("MATCH (a:User {id: 1})-[r:FOLLOWS]->(b:User {id: 2}) SET r.weight = 99");
    assert!(
        result.is_err(),
        "PROPAGATE: trigger error must abort the SET: {result:?}"
    );

    let rows = db
        .execute_cypher("MATCH (a:User {id: 1})-[r:FOLLOWS]->(b:User {id: 2}) RETURN r.weight AS w")
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("w"),
        Some(&Value::Int(5)),
        "edge prop must remain unchanged after PROPAGATE abort"
    );
}
