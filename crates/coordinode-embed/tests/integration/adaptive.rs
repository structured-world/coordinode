//! Integration tests: adaptive parallel traversal (G010).
//!
//! Tests that the executor switches to rayon parallel processing when
//! fan-out exceeds `parallel_threshold`, producing the same results as
//! sequential execution with no data loss (no truncation).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_embed::Database;

fn open_db() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open db");
    (db, dir)
}

/// Create a hub node with N outgoing edges to N target nodes.
fn create_star_graph(db: &mut Database, hub_name: &str, target_count: usize) {
    db.execute_cypher(&format!("CREATE (h:Hub {{name: '{hub_name}'}})"))
        .expect("create hub");

    for i in 0..target_count {
        db.execute_cypher(&format!(
            "CREATE (t:Target {{name: 'target_{i}', idx: {i}}})"
        ))
        .expect("create target");
    }

    // Create edges from hub to all targets
    for i in 0..target_count {
        db.execute_cypher(&format!(
            "MATCH (h:Hub {{name: '{hub_name}'}}), (t:Target {{idx: {i}}}) \
             CREATE (h)-[:LINKS]->(t)"
        ))
        .expect("create edge");
    }
}

/// Parallel traversal returns ALL edges (no truncation) when fan-out exceeds
/// the parallel threshold.
#[test]
fn parallel_traversal_returns_all_targets() {
    let (mut db, _dir) = open_db();

    // Create hub with 30 targets — above default parallel_threshold in tests
    create_star_graph(&mut db, "BigHub", 30);

    // Set parallel threshold low (5) to force parallel path
    db.set_adaptive_parallel_threshold(5);

    let rows = db
        .execute_cypher("MATCH (h:Hub {name: 'BigHub'})-[:LINKS]->(t:Target) RETURN t.name AS name")
        .expect("parallel traversal");

    // ALL 30 targets returned — no truncation
    assert_eq!(
        rows.len(),
        30,
        "expected 30 targets (no truncation), got {}",
        rows.len()
    );
}

/// Parallel and sequential traversal produce the same results.
#[test]
fn parallel_matches_sequential_results() {
    let (mut db, _dir) = open_db();
    create_star_graph(&mut db, "TestHub", 20);

    // Query with parallel (threshold=5 → 20 edges triggers parallel)
    db.set_adaptive_parallel_threshold(5);
    let parallel_rows = db
        .execute_cypher(
            "MATCH (h:Hub {name: 'TestHub'})-[:LINKS]->(t:Target) \
             RETURN t.idx AS idx ORDER BY idx",
        )
        .expect("parallel query");

    // Query with sequential (threshold=1000000 → never triggers parallel)
    db.set_adaptive_parallel_threshold(1_000_000);
    let sequential_rows = db
        .execute_cypher(
            "MATCH (h:Hub {name: 'TestHub'})-[:LINKS]->(t:Target) \
             RETURN t.idx AS idx ORDER BY idx",
        )
        .expect("sequential query");

    assert_eq!(parallel_rows.len(), sequential_rows.len());
    assert_eq!(parallel_rows.len(), 20);

    // Same values (after ORDER BY, should be identical)
    for (p, s) in parallel_rows.iter().zip(sequential_rows.iter()) {
        assert_eq!(p.get("idx"), s.get("idx"));
    }
}

/// Small fan-out stays sequential (no parallel overhead).
#[test]
fn small_fanout_stays_sequential() {
    let (mut db, _dir) = open_db();
    create_star_graph(&mut db, "SmallHub", 3);

    // Default threshold (1000) — 3 edges won't trigger parallel
    let rows = db
        .execute_cypher(
            "MATCH (h:Hub {name: 'SmallHub'})-[:LINKS]->(t:Target) RETURN t.name AS name",
        )
        .expect("sequential traversal");

    assert_eq!(rows.len(), 3);
}

/// Edge properties work correctly in parallel single-hop traversal.
#[test]
fn parallel_single_hop_edge_properties() {
    let (mut db, _dir) = open_db();

    db.execute_cypher("CREATE (h:Person {name: 'Hub'})")
        .expect("create hub");
    for i in 0..10 {
        db.execute_cypher(&format!("CREATE (t:Person {{name: 'F{i}'}})"))
            .expect("create friend");
        db.execute_cypher(&format!(
            "MATCH (h:Person {{name: 'Hub'}}), (t:Person {{name: 'F{i}'}}) \
             CREATE (h)-[:FRIEND {{year: {}, tag: 'y{}'}}]->(t)",
            2020 + i,
            i
        ))
        .expect("create edge with props");
    }

    // Force parallel (threshold=3, 10 edges > 3)
    db.set_adaptive_parallel_threshold(3);

    let rows = db
        .execute_cypher(
            "MATCH (h:Person {name: 'Hub'})-[r:FRIEND]->(f:Person) \
             RETURN f.name AS name, r.year AS year, r.tag AS tag ORDER BY year",
        )
        .expect("parallel with edge props");

    assert_eq!(rows.len(), 10, "all 10 friends returned");

    // Verify edge properties are present (not Null)
    for row in &rows {
        let year = row.get("year");
        assert!(
            matches!(year, Some(coordinode_core::graph::types::Value::Int(_))),
            "edge property 'year' should be Int, got {year:?}"
        );
        let tag = row.get("tag");
        assert!(
            matches!(tag, Some(coordinode_core::graph::types::Value::String(_))),
            "edge property 'tag' should be String, got {tag:?}"
        );
    }

    // First row (ordered by year) should be year=2020
    assert_eq!(
        rows[0].get("year"),
        Some(&coordinode_core::graph::types::Value::Int(2020))
    );
}

/// COMPUTED properties are injected in parallel traversal path.
#[test]
fn parallel_traversal_injects_computed_properties() {
    use coordinode_core::schema::computed::{ComputedSpec, DecayFormula};
    use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType};

    let (mut db, _dir) = open_db();

    // Create schema with COMPUTED Decay property
    let mut schema = LabelSchema::new("Memory");
    schema.add_property(PropertyDef::new("content", PropertyType::String));
    schema.add_property(PropertyDef::new("created_at", PropertyType::Timestamp));
    schema.add_property(PropertyDef::computed(
        "freshness",
        ComputedSpec::Decay {
            formula: DecayFormula::Linear,
            initial: 1.0,
            target: 0.0,
            duration_secs: 604800, // 7 days
            anchor_field: "created_at".into(),
        },
    ));

    // Persist schema
    let schema_key = coordinode_core::schema::definition::encode_label_schema_key("Memory");
    let bytes = schema.to_msgpack().expect("serialize schema");
    db.engine_shared()
        .put(
            coordinode_storage::engine::partition::Partition::Schema,
            &schema_key,
            &bytes,
        )
        .expect("persist schema");

    // Create hub node
    db.execute_cypher("CREATE (h:Hub {name: 'source'})")
        .expect("create hub");

    // Create 10 Memory nodes with timestamps and edges
    let now_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64;

    for i in 0..10 {
        db.execute_cypher(&format!(
            "CREATE (m:Memory {{content: 'note_{i}', created_at: {now_us}}})"
        ))
        .expect("create memory node");
        db.execute_cypher(&format!(
            "MATCH (h:Hub {{name: 'source'}}), (m:Memory {{content: 'note_{i}'}}) \
             CREATE (h)-[:RECALLS]->(m)"
        ))
        .expect("create edge");
    }

    // Force parallel (threshold=3, 10 edges > 3)
    db.set_adaptive_parallel_threshold(3);

    let rows = db
        .execute_cypher(
            "MATCH (h:Hub {name: 'source'})-[:RECALLS]->(m:Memory) \
             RETURN m.content AS content, m.freshness AS freshness",
        )
        .expect("parallel with computed");

    assert_eq!(rows.len(), 10, "all 10 memories returned");

    // Verify COMPUTED freshness is injected (should be ~1.0 for just-created nodes)
    for row in &rows {
        let freshness = row.get("freshness");
        match freshness {
            Some(coordinode_core::graph::types::Value::Float(f)) => {
                assert!(
                    *f > 0.99,
                    "freshly created node should have freshness ≈ 1.0, got {f}"
                );
            }
            other => panic!("expected Float for computed freshness, got {other:?}"),
        }
    }
}

// ── G066 regression: varlen edge properties at depth > 1 ──────────────

/// REGRESSION (G066): Edge properties in variable-length traversal at depth > 1
/// should reflect the actual edge at that depth, not the original source edge.
///
/// Graph: A --[KNOWS {since: 2020}]--> B --[KNOWS {since: 2023}]--> C
///
/// Query: MATCH (a {name: "Alice"})-[r:KNOWS*1..2]->(c) RETURN c.name, r.since
///
/// At depth=1, edge A→B has since=2020.
/// At depth=2, edge B→C has since=2023.
/// Bug: depth=2 lookup uses A as source (gets A→C edge which doesn't exist → NULL).
#[test]
fn varlen_edge_properties_at_depth_2() {
    let (mut db, _dir) = open_db();

    // Create chain: Alice → Bob → Charlie with edge properties
    db.execute_cypher("CREATE (a:Person {name: 'Alice'})")
        .expect("create Alice");
    db.execute_cypher("CREATE (b:Person {name: 'Bob'})")
        .expect("create Bob");
    db.execute_cypher("CREATE (c:Person {name: 'Charlie'})")
        .expect("create Charlie");

    db.execute_cypher(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) \
         CREATE (a)-[:KNOWS {since: 2020}]->(b)",
    )
    .expect("create Alice→Bob edge");

    db.execute_cypher(
        "MATCH (b:Person {name: 'Bob'}), (c:Person {name: 'Charlie'}) \
         CREATE (b)-[:KNOWS {since: 2023}]->(c)",
    )
    .expect("create Bob→Charlie edge");

    // Varlen query: depth 1 (Alice→Bob) and depth 2 (Alice→Bob→Charlie)
    let rows = db
        .execute_cypher(
            "MATCH (a:Person {name: 'Alice'})-[r:KNOWS*1..2]->(c:Person) \
             RETURN c.name AS name, r.since AS since",
        )
        .expect("varlen traverse with edge props");

    // Should get 2 rows: Bob (depth 1) and Charlie (depth 2)
    assert_eq!(rows.len(), 2, "expected 2 results, got {}", rows.len());

    // Find the Charlie row (depth 2)
    let charlie_row = rows
        .iter()
        .find(|r| {
            r.get("name")
                == Some(&coordinode_core::graph::types::Value::String(
                    "Charlie".into(),
                ))
        })
        .expect("Charlie row should exist");

    // G066 BUG: At depth=2, r.since should be 2023 (Bob→Charlie edge),
    // but the pre-existing bug looks up Alice→Charlie which doesn't exist → Null.
    // This test documents the expected CORRECT behavior.
    let since = charlie_row.get("since");
    assert_eq!(
        since,
        Some(&coordinode_core::graph::types::Value::Int(2023)),
        "depth-2 edge property should be 2023 (Bob→Charlie), got {since:?} \
         — G066: varlen edge source ID bug"
    );
}

// ── Varlen traversal with parallel ─────────────────────────────────

/// Variable-length traversal triggers parallel when depth has high fan-out.
#[test]
fn varlen_parallel_high_fanout_at_depth_1() {
    let (mut db, _dir) = open_db();

    // Create root → 15 intermediate → 1 leaf each
    // At depth 1: 15 edges (triggers parallel at threshold=5)
    db.execute_cypher("CREATE (r:Root {name: 'root'})")
        .expect("create root");
    for i in 0..15 {
        db.execute_cypher(&format!("CREATE (m:Mid {{name: 'mid_{i}', idx: {i}}})"))
            .expect("create mid");
        db.execute_cypher(&format!(
            "MATCH (r:Root {{name: 'root'}}), (m:Mid {{idx: {i}}}) CREATE (r)-[:STEP]->(m)"
        ))
        .expect("root→mid edge");

        db.execute_cypher(&format!("CREATE (l:Leaf {{name: 'leaf_{i}'}})"))
            .expect("create leaf");
        db.execute_cypher(&format!(
            "MATCH (m:Mid {{idx: {i}}}), (l:Leaf {{name: 'leaf_{i}'}}) CREATE (m)-[:STEP]->(l)"
        ))
        .expect("mid→leaf edge");
    }

    db.set_adaptive_parallel_threshold(5);

    let rows = db
        .execute_cypher("MATCH (r:Root {name: 'root'})-[:STEP*1..2]->(n) RETURN n.name AS name")
        .expect("varlen parallel");

    // 15 mid nodes (depth 1) + 15 leaf nodes (depth 2) = 30
    assert_eq!(
        rows.len(),
        30,
        "expected 30 (15 mid + 15 leaf), got {}",
        rows.len()
    );
}

// ── Feedback cache persistence across queries ──────────────────────

/// Feedback cache records super-node degree and persists across queries.
#[test]
fn feedback_cache_persists_across_queries() {
    let (mut db, _dir) = open_db();

    // Create hub with 20 targets
    create_star_graph(&mut db, "CacheHub", 20);

    // First query with low threshold → triggers parallel → records in cache
    db.set_adaptive_parallel_threshold(5);
    let rows1 = db
        .execute_cypher(
            "MATCH (h:Hub {name: 'CacheHub'})-[:LINKS]->(t:Target) RETURN t.name AS name",
        )
        .expect("first query");
    assert_eq!(rows1.len(), 20);

    // Second query — feedback cache should already know about this super-node.
    // We can't directly observe the cache hit, but we verify the query still
    // produces correct results (parallel path reused).
    let rows2 = db
        .execute_cypher(
            "MATCH (h:Hub {name: 'CacheHub'})-[:LINKS]->(t:Target) RETURN t.name AS name",
        )
        .expect("second query (cache hit)");
    assert_eq!(rows2.len(), 20);
}

/// Parallel traversal works with incoming direction.
#[test]
fn parallel_incoming_direction() {
    let (mut db, _dir) = open_db();

    // Create 15 sources pointing to one target
    db.execute_cypher("CREATE (t:Sink {name: 'sink'})")
        .expect("create sink");
    for i in 0..15 {
        db.execute_cypher(&format!("CREATE (s:Source {{name: 'src_{i}', idx: {i}}})"))
            .expect("create source");
        db.execute_cypher(&format!(
            "MATCH (s:Source {{idx: {i}}}), (t:Sink {{name: 'sink'}}) CREATE (s)-[:FEEDS]->(t)"
        ))
        .expect("create edge");
    }

    db.set_adaptive_parallel_threshold(5);

    // Query INCOMING: who feeds the sink?
    let rows = db
        .execute_cypher("MATCH (t:Sink {name: 'sink'})<-[:FEEDS]-(s:Source) RETURN s.name AS name")
        .expect("parallel incoming");

    assert_eq!(
        rows.len(),
        15,
        "all 15 sources returned via incoming, got {}",
        rows.len()
    );
}
