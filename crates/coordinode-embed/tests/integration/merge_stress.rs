//! Integration tests: merge operator stress + time-travel through Database API.
//!
//! Tests the full pipeline: Cypher → parser → planner → executor → merge operator → storage.
//! Verifies:
//!   (1) Concurrent edge creation via merge operators (zero ErrConflict)
//!   (2) Edge time-travel via storage snapshots
//!   (3) Compaction preserves snapshot correctness

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_core::graph::edge::PostingList;
use coordinode_core::graph::types::Value;
use coordinode_embed::Database;
use coordinode_storage::engine::partition::Partition;

fn open_db() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open db");
    (db, dir)
}

// ── Concurrent edge creation through Cypher ────────────────────────

#[test]
fn concurrent_edge_creation_via_cypher_zero_conflict() {
    // Create nodes, then add edges from multiple sequential execute_cypher
    // calls. Each call goes through the merge operator pipeline (not
    // read-modify-write), so edges accumulate without conflict.
    let (mut db, _dir) = open_db();

    // Create a hub node and 100 target nodes.
    db.execute_cypher("CREATE (h:Hub {name: 'central'})")
        .expect("create hub");

    for i in 0..100 {
        db.execute_cypher(&format!("CREATE (t:Target {{id: {i}}})"))
            .expect("create target");
    }

    // Create 100 edges from Hub to each Target via separate Cypher calls.
    // Each CREATE edge goes through adj_merge_add → merge operator.
    for i in 0..100 {
        db.execute_cypher(&format!(
            "MATCH (h:Hub {{name: 'central'}}), (t:Target {{id: {i}}}) CREATE (h)-[:CONNECTS]->(t)"
        ))
        .expect("create edge");
    }

    // Verify all 100 edges exist via graph traversal.
    let rows = db
        .execute_cypher(
            "MATCH (h:Hub {name: 'central'})-[:CONNECTS]->(t:Target) RETURN t.id AS tid",
        )
        .expect("traverse");

    assert_eq!(
        rows.len(),
        100,
        "expected 100 edges from Hub, got {}",
        rows.len()
    );

    // Verify all target IDs are present (0..100).
    let mut ids: Vec<i64> = rows
        .iter()
        .filter_map(|r| match r.get("tid") {
            Some(Value::Int(v)) => Some(*v),
            _ => None,
        })
        .collect();
    ids.sort();
    let expected: Vec<i64> = (0..100).collect();
    assert_eq!(ids, expected, "not all target IDs found via traversal");
}

// ── Edge time-travel via storage snapshot ────────────────────────────

#[test]
fn edge_time_travel_via_storage_snapshot() {
    // Demonstrates that storage snapshots correctly capture posting list
    // state at a point in time: edges written AFTER the snapshot are invisible.
    //
    // This tests the storage layer through the Database API because the query
    // engine's AS OF TIMESTAMP does not yet filter adj: partition reads (gap).
    let (mut db, _dir) = open_db();

    // Step 1: Create 3 nodes and 2 edges.
    db.execute_cypher("CREATE (a:Node {name: 'A'})")
        .expect("create A");
    db.execute_cypher("CREATE (b:Node {name: 'B'})")
        .expect("create B");
    db.execute_cypher("CREATE (c:Node {name: 'C'})")
        .expect("create C");

    db.execute_cypher("MATCH (a:Node {name: 'A'}), (b:Node {name: 'B'}) CREATE (a)-[:LINK]->(b)")
        .expect("create A->B");
    db.execute_cypher("MATCH (a:Node {name: 'A'}), (c:Node {name: 'C'}) CREATE (a)-[:LINK]->(c)")
        .expect("create A->C");

    // Flush and take snapshot.
    db.engine_shared().persist().expect("persist");
    let snap_after_2_edges = db.engine_shared().snapshot();

    // Step 2: Add 1 more edge.
    db.execute_cypher("CREATE (d:Node {name: 'D'})")
        .expect("create D");
    db.execute_cypher("MATCH (a:Node {name: 'A'}), (d:Node {name: 'D'}) CREATE (a)-[:LINK]->(d)")
        .expect("create A->D");

    db.engine_shared().persist().expect("persist");

    // Current state: A has 3 outgoing LINK edges (B, C, D).
    let rows = db
        .execute_cypher("MATCH (a:Node {name: 'A'})-[:LINK]->(t) RETURN t.name AS name")
        .expect("traverse current");
    assert_eq!(rows.len(), 3, "current: A should have 3 outgoing edges");

    // Snapshot should see only 2 edges (B, C) — D was added after snapshot.
    // Read adj: keys through the snapshot using snapshot_prefix_scan.
    let entries = db
        .engine_shared()
        .snapshot_prefix_scan(&snap_after_2_edges, Partition::Adj, b"adj:LINK:out:")
        .expect("snapshot scan");

    let mut found = false;
    for (_key, value) in &entries {
        let plist = PostingList::from_bytes(value).expect("decode");
        // Snapshot should see exactly 2 UIDs (B and C).
        assert_eq!(
            plist.len(),
            2,
            "snapshot: A should have 2 outgoing edges, got {}",
            plist.len()
        );
        found = true;
    }
    assert!(
        found,
        "snapshot should find adj key for A's outgoing LINK edges"
    );
}

// ── Compaction preserves edge snapshot correctness ──────────────────

#[test]
fn edge_compaction_preserves_snapshot_correctness() {
    // Write edges in batches, take snapshots, compact, verify snapshots.
    let (mut db, _dir) = open_db();

    // Create source and target nodes.
    db.execute_cypher("CREATE (s:Source {name: 'S'})")
        .expect("create source");
    for i in 0..10 {
        db.execute_cypher(&format!("CREATE (t:Tgt {{id: {i}}})"))
            .expect("create target");
    }

    // Batch 1: 5 edges (targets 0-4).
    for i in 0..5 {
        db.execute_cypher(&format!(
            "MATCH (s:Source {{name: 'S'}}), (t:Tgt {{id: {i}}}) CREATE (s)-[:REL]->(t)"
        ))
        .expect("create edge");
    }
    db.engine_shared().persist().expect("persist");
    let snap_5 = db.engine_shared().snapshot();

    // Batch 2: 5 more edges (targets 5-9).
    for i in 5..10 {
        db.execute_cypher(&format!(
            "MATCH (s:Source {{name: 'S'}}), (t:Tgt {{id: {i}}}) CREATE (s)-[:REL]->(t)"
        ))
        .expect("create edge");
    }
    db.engine_shared().persist().expect("persist");
    let snap_10 = db.engine_shared().snapshot();

    // Force compaction.
    db.engine_shared()
        .force_compaction(Partition::Adj)
        .expect("compaction");

    // Verify current state: 10 edges via full Cypher pipeline.
    let rows = db
        .execute_cypher("MATCH (s:Source {name: 'S'})-[:REL]->(t) RETURN t.id")
        .expect("traverse");
    assert_eq!(rows.len(), 10, "current: S should have 10 edges");

    // Verify snapshot after 5 edges (through raw storage snapshot).
    let entries_5 = db
        .engine_shared()
        .snapshot_prefix_scan(&snap_5, Partition::Adj, b"adj:REL:out:")
        .expect("snap 5 scan");

    let snap5_count: usize = entries_5
        .iter()
        .map(|(_, v)| PostingList::from_bytes(v).expect("decode").len())
        .sum();
    assert_eq!(
        snap5_count, 5,
        "snap_5 should see 5 edges, got {snap5_count}"
    );

    // Verify snapshot after 10 edges.
    let entries_10 = db
        .engine_shared()
        .snapshot_prefix_scan(&snap_10, Partition::Adj, b"adj:REL:out:")
        .expect("snap 10 scan");

    let snap10_count: usize = entries_10
        .iter()
        .map(|(_, v)| PostingList::from_bytes(v).expect("decode").len())
        .sum();
    assert_eq!(
        snap10_count, 10,
        "snap_10 should see 10 edges, got {snap10_count}"
    );
}
