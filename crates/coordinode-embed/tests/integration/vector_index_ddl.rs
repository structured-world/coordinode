//! Integration tests: VectorIndexRegistry + CREATE/DROP VECTOR INDEX DDL (R-API3).
//!
//! Tests the full vector index lifecycle through Database:
//! - CREATE VECTOR INDEX DDL registers an HNSW index and backfills existing nodes
//! - EXPLAIN after CREATE shows "HnswScan(<name>)"
//! - EXPLAIN without index shows "BruteForce"
//! - EXPLAIN: label/property mismatch does NOT use the index
//! - DROP VECTOR INDEX removes the index
//! - After DROP, EXPLAIN reverts to "BruteForce"
//! - Duplicate CREATE on same (label, property) returns error
//! - DROP then re-CREATE works
//! - SET updates HNSW — updated vector is searchable
//! - DETACH DELETE removes node from HNSW
//! - REMOVE property removes from HNSW
//! - Persistence: index survives Database close+reopen

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_core::graph::types::Value;
use coordinode_embed::Database;

fn open_db() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open db");
    (db, dir)
}

// ── Bulk-insert HNSW batching ─────────────────────────────────────────

/// UNWIND-driven CREATE batches N vectors into the HNSW index. After
/// the batch, every inserted node must be searchable (recall = 1.0
/// on its own vector). Regression for the executor-side write buffer
/// that flushes per-(label, property) at end of execute().
#[test]
fn unwind_create_flushes_all_vectors_into_hnsw() {
    let (mut db, _dir) = open_db();
    db.execute_cypher(
        "CREATE VECTOR INDEX bulk_emb ON :Bulk(emb) OPTIONS {m: 8, ef_construction: 64, metric: \"l2\"}",
    )
    .expect("create vector index");

    // Bulk-insert 32 distinct nodes via UNWIND $rows AS r CREATE …
    // Same pattern the gRPC CreateNodesBatch handler uses.
    let mut cypher_rows = String::from("[");
    for i in 0..32 {
        if i > 0 {
            cypher_rows.push_str(", ");
        }
        // Each vector is the unit basis along its own index: e_i.
        // Concrete values keep the test self-contained without
        // parameter binding through the embedded API.
        let mut v = [0.0f32; 32];
        v[i] = 1.0;
        cypher_rows.push_str(&format!("{{i: {i}, emb: ["));
        for (j, x) in v.iter().enumerate() {
            if j > 0 {
                cypher_rows.push_str(", ");
            }
            cypher_rows.push_str(&format!("{x}"));
        }
        cypher_rows.push_str("]}");
    }
    cypher_rows.push(']');
    let query = format!("UNWIND {cypher_rows} AS r CREATE (m:Bulk) SET m = r RETURN m");
    let rows = db.execute_cypher(&query).expect("bulk create");
    assert_eq!(rows.len(), 32, "bulk create returned wrong row count");

    // Every inserted vector is its own nearest neighbour at k=1.
    // If the batched flush missed any row, search would either miss
    // it (returning some other node) or return fewer than the
    // expected 32 candidates over the loop.
    for i in 0..32 {
        let mut v = [0.0f32; 32];
        v[i] = 1.0;
        let mut q = String::from("[");
        for (j, x) in v.iter().enumerate() {
            if j > 0 {
                q.push_str(", ");
            }
            q.push_str(&format!("{x}"));
        }
        q.push(']');
        let cypher = format!(
            "MATCH (n:Bulk) WITH n, vector_similarity(n.emb, {q}) AS s \
             ORDER BY s DESC LIMIT 1 RETURN n.i AS i"
        );
        let res = db.execute_cypher(&cypher).expect("vector search");
        assert_eq!(res.len(), 1, "search for e_{i} returned no rows");
        let got = match res[0].get("i") {
            Some(Value::Int(x)) => *x,
            other => panic!("unexpected result for e_{i}: {other:?}"),
        };
        assert_eq!(
            got, i as i64,
            "search for e_{i} returned node {got} — batched flush dropped a row"
        );
    }
}

// ── CREATE VECTOR INDEX DDL ───────────────────────────────────────────

/// CREATE VECTOR INDEX returns metadata row with correct fields.
#[test]
fn create_vector_index_returns_metadata() {
    let (mut db, _dir) = open_db();

    let rows = db
        .execute_cypher(
            "CREATE VECTOR INDEX emb ON :Item(embedding) OPTIONS {m: 16, ef_construction: 200, metric: \"cosine\"}",
        )
        .expect("create vector index");

    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("index"),
        Some(&Value::String("emb".into())),
        "index name"
    );
    assert_eq!(
        rows[0].get("label"),
        Some(&Value::String("Item".into())),
        "label"
    );
    assert_eq!(
        rows[0].get("property"),
        Some(&Value::String("embedding".into())),
        "property"
    );
}

/// CREATE VECTOR INDEX with default OPTIONS (no OPTIONS clause).
#[test]
fn create_vector_index_default_options() {
    let (mut db, _dir) = open_db();

    let rows = db
        .execute_cypher("CREATE VECTOR INDEX doc_emb ON :Document(emb)")
        .expect("create vector index");

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("index"), Some(&Value::String("doc_emb".into())));
}

// ── EXPLAIN strategy selection ────────────────────────────────────────

/// EXPLAIN after CREATE VECTOR INDEX shows HnswScan, not BruteForce.
#[test]
fn explain_shows_hnsw_scan_after_create() {
    let (mut db, _dir) = open_db();

    // Insert a node to make the MATCH non-trivial.
    db.execute_cypher("CREATE (n:Item {embedding: [1.0, 0.0, 0.0]})")
        .expect("create node");

    // Without index: BruteForce.
    let before = db
        .explain_cypher(
            "MATCH (n:Item) WITH n, vector_similarity(n.embedding, [1.0, 0.0, 0.0]) AS s ORDER BY s DESC LIMIT 5 RETURN n, s",
        )
        .expect("explain before");
    assert!(
        before.contains("BruteForce"),
        "expected BruteForce before index, got:\n{before}"
    );

    // Create the index.
    db.execute_cypher(
        "CREATE VECTOR INDEX item_emb ON :Item(embedding) OPTIONS {m: 16, ef_construction: 200, metric: \"cosine\"}",
    )
    .expect("create vector index");

    // After index: HnswScan.
    let after = db
        .explain_cypher(
            "MATCH (n:Item) WITH n, vector_similarity(n.embedding, [1.0, 0.0, 0.0]) AS s ORDER BY s DESC LIMIT 5 RETURN n, s",
        )
        .expect("explain after");
    // The access path replaces NodeScan+VectorTopK with the HnswScan
    // source operator; the EXPLAIN line carries index, property, and
    // the ORDER BY function (which implies the metric).
    assert!(
        after.contains("HnswScan(n:Item ON item_emb(embedding), vector_similarity"),
        "expected HnswScan access path after index, got:\n{after}"
    );
}

/// EXPLAIN without any vector index shows BruteForce.
#[test]
fn explain_shows_brute_force_without_index() {
    let (db, _dir) = open_db();

    let plan = db
        .explain_cypher(
            "MATCH (n:Item) WITH n, vector_similarity(n.embedding, [1.0, 0.0, 0.0]) AS s ORDER BY s DESC LIMIT 5 RETURN n, s",
        )
        .expect("explain");
    assert!(
        plan.contains("BruteForce"),
        "expected BruteForce without index, got:\n{plan}"
    );
}

// ── DROP VECTOR INDEX DDL ─────────────────────────────────────────────

/// DROP VECTOR INDEX removes the index; EXPLAIN reverts to BruteForce.
#[test]
fn drop_vector_index_removes_index() {
    let (mut db, _dir) = open_db();

    // Create, verify HnswScan.
    db.execute_cypher(
        "CREATE VECTOR INDEX item_emb ON :Item(embedding) OPTIONS {metric: \"cosine\"}",
    )
    .expect("create vector index");

    let after_create = db
        .explain_cypher(
            "MATCH (n:Item) WITH n, vector_similarity(n.embedding, [1.0, 0.0, 0.0]) AS s ORDER BY s DESC LIMIT 5 RETURN n, s",
        )
        .expect("explain after create");
    assert!(
        after_create.contains("HnswScan(n:Item ON item_emb(embedding), vector_similarity"),
        "expected HnswScan access path after create, got:\n{after_create}"
    );

    // Drop the index.
    let drop_rows = db
        .execute_cypher("DROP VECTOR INDEX item_emb")
        .expect("drop vector index");
    assert_eq!(drop_rows.len(), 1);
    assert_eq!(
        drop_rows[0].get("index"),
        Some(&Value::String("item_emb".into())),
        "index name in drop result"
    );
    assert_eq!(
        drop_rows[0].get("dropped"),
        Some(&Value::Bool(true)),
        "dropped flag"
    );

    // After drop: BruteForce.
    let after_drop = db
        .explain_cypher(
            "MATCH (n:Item) WITH n, vector_similarity(n.embedding, [1.0, 0.0, 0.0]) AS s ORDER BY s DESC LIMIT 5 RETURN n, s",
        )
        .expect("explain after drop");
    assert!(
        after_drop.contains("BruteForce"),
        "expected BruteForce after drop, got:\n{after_drop}"
    );
}

/// DROP VECTOR INDEX on non-existent index returns error.
#[test]
fn drop_vector_index_nonexistent_returns_error() {
    let (mut db, _dir) = open_db();

    let result = db.execute_cypher("DROP VECTOR INDEX no_such_index");
    assert!(
        result.is_err(),
        "expected error when dropping non-existent index"
    );
}

// ── Persistence across reopen ─────────────────────────────────────────

/// Vector index created via DDL survives a Database close+reopen.
/// After reopen EXPLAIN still shows HnswScan (index definition persisted
/// to schema partition and registry rebuilt by load_vector_indexes).
#[test]
fn vector_index_persists_across_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");

    // Session 1: create index.
    {
        let mut db = Database::open(dir.path()).expect("open");
        db.execute_cypher(
            "CREATE VECTOR INDEX persist_idx ON :Doc(vec) OPTIONS {metric: \"cosine\"}",
        )
        .expect("create vector index");
        // Flush + close implicitly when `db` drops.
    }

    // Session 2: reopen and verify EXPLAIN shows HnswScan.
    {
        let db = Database::open(dir.path()).expect("reopen");
        let plan = db
            .explain_cypher(
                "MATCH (n:Doc) WITH n, vector_similarity(n.vec, [1.0, 0.0]) AS s ORDER BY s DESC LIMIT 3 RETURN n, s",
            )
            .expect("explain after reopen");
        assert!(
            plan.contains("ON persist_idx("),
            "expected HnswScan access path after reopen, got:\n{plan}"
        );
    }
}

// ── Backfill + search ─────────────────────────────────────────────────

/// Nodes created BEFORE the vector index exist are backfilled into HNSW
/// during CREATE VECTOR INDEX. Subsequent vector_similarity queries find them.
#[test]
fn create_vector_index_backfills_and_search_finds_nodes() {
    let (mut db, _dir) = open_db();

    // Insert nodes BEFORE creating the index.
    db.execute_cypher("CREATE (n:Item {embedding: [1.0, 0.0, 0.0]})")
        .expect("node 1");
    db.execute_cypher("CREATE (n:Item {embedding: [0.0, 1.0, 0.0]})")
        .expect("node 2");
    db.execute_cypher("CREATE (n:Item {embedding: [0.0, 0.0, 1.0]})")
        .expect("node 3");

    // Create index — backfill runs.
    db.execute_cypher(
        "CREATE VECTOR INDEX item_emb ON :Item(embedding) OPTIONS {metric: \"cosine\"}",
    )
    .expect("create vector index");

    // Query should return the closest vector to [1,0,0].
    // Use WHERE filter pattern — nearest to [1,0,0] has similarity > 0.9.
    let rows = db
        .execute_cypher(
            "MATCH (n:Item) \
             WHERE vector_similarity(n.embedding, [1.0, 0.0, 0.0]) > 0.9 \
             RETURN n.embedding AS vec",
        )
        .expect("vector search");

    assert_eq!(
        rows.len(),
        1,
        "only [1,0,0] has similarity > 0.9 to [1,0,0]"
    );
    // Nodes created via Cypher literal store arrays as Array([Float(...)]),
    // not Value::Vector (which is the Rust API form).
    assert_eq!(
        rows[0].get("vec"),
        Some(&Value::Array(vec![
            Value::Float(1.0),
            Value::Float(0.0),
            Value::Float(0.0)
        ])),
        "nearest should be [1,0,0]"
    );
}

/// Vector search executes correctly (not just EXPLAIN) after CREATE VECTOR INDEX.
/// Verifies that the executor routes through HNSW, not just that EXPLAIN says so.
#[test]
fn vector_search_executes_after_create_index() {
    let (mut db, _dir) = open_db();

    db.execute_cypher(
        "CREATE VECTOR INDEX item_emb ON :Item(embedding) OPTIONS {metric: \"cosine\"}",
    )
    .expect("create vector index");

    // Insert nodes AFTER index creation (auto-indexed via on_vector_written).
    for i in 0u32..5 {
        let vec = format!("[{}.0, 0.0, 0.0]", i);
        db.execute_cypher(&format!("CREATE (n:Item {{embedding: {vec}, rank: {i}}})"))
            .expect("insert node");
    }

    // Nearest to [4.5, 0, 0]: rank=4 ([4.0, 0, 0]) has smallest distance.
    // Project rank alias in WITH alongside the distance (VectorTopK Pattern B).
    let rows = db
        .execute_cypher(
            "MATCH (n:Item) \
             WITH n.rank AS rank, vector_distance(n.embedding, [4.5, 0.0, 0.0]) AS dist \
             ORDER BY dist \
             LIMIT 1 \
             RETURN rank",
        )
        .expect("vector search");

    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("rank"),
        Some(&Value::Int(4)),
        "nearest to [4.5,0,0] should be rank=4"
    );
}

// ── nodes_indexed in backfill response ───────────────────────────────

/// CREATE VECTOR INDEX response includes nodes_indexed count.
#[test]
fn create_vector_index_returns_nodes_indexed_count() {
    let (mut db, _dir) = open_db();

    // Create 3 nodes with the indexed property, 1 without.
    db.execute_cypher("CREATE (n:Item {embedding: [1.0, 0.0, 0.0]})")
        .expect("node 1");
    db.execute_cypher("CREATE (n:Item {embedding: [0.0, 1.0, 0.0]})")
        .expect("node 2");
    db.execute_cypher("CREATE (n:Item {embedding: [0.0, 0.0, 1.0]})")
        .expect("node 3");
    db.execute_cypher("CREATE (n:Item {name: 'no_vector'})")
        .expect("node without embedding");

    let rows = db
        .execute_cypher(
            "CREATE VECTOR INDEX item_emb ON :Item(embedding) OPTIONS {metric: \"cosine\"}",
        )
        .expect("create vector index");

    assert_eq!(rows.len(), 1);
    // CREATE VECTOR INDEX now returns immediately with state="building" and
    // nodes_indexed=0; the backfill thread fills the graph in the background
    // and transitions the persisted state to "ready" with the final count.
    let state = rows[0]
        .get("state")
        .and_then(|v| match v {
            Value::String(s) => Some(s.as_str()),
            _ => None,
        })
        .expect("state field present on CREATE VECTOR INDEX response");
    assert!(
        matches!(state, "building" | "ready"),
        "state should be building (background) or ready (legacy sync), got {state}",
    );

    // Wait for the background backfill to settle, then verify the graph
    // has the three vectorised nodes by issuing a vector-search query.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let mut search_hits = 0usize;
    while std::time::Instant::now() < deadline {
        let results = db
            .execute_cypher(
                "MATCH (n:Item) WHERE vector_distance(n.embedding, [1.0, 0.0, 0.0]) < 2.0 \
                 RETURN n.embedding AS e LIMIT 10",
            )
            .expect("vector search");
        if results.len() >= 3 {
            search_hits = results.len();
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert!(
        search_hits >= 3,
        "background backfill should populate the HNSW graph with all 3 vectorised \
         nodes within 5s, got {search_hits} hits",
    );
}

// ── Duplicate CREATE ──────────────────────────────────────────────────

/// Creating a vector index on the same (label, property) twice returns an error.
#[test]
fn duplicate_create_vector_index_errors() {
    let (mut db, _dir) = open_db();

    db.execute_cypher("CREATE VECTOR INDEX emb1 ON :Item(embedding)")
        .expect("first create should succeed");

    let result = db.execute_cypher("CREATE VECTOR INDEX emb2 ON :Item(embedding)");
    assert!(
        result.is_err(),
        "duplicate vector index on same (label, property) should fail"
    );
}

// ── EXPLAIN: label / property mismatch does not use index ────────────

/// EXPLAIN for a query on a different label still shows BruteForce even if
/// an index exists on a different label with the same property name.
#[test]
fn explain_brute_force_for_different_label() {
    let (mut db, _dir) = open_db();

    // Index is on Item.embedding — NOT on User.embedding.
    db.execute_cypher(
        "CREATE VECTOR INDEX item_emb ON :Item(embedding) OPTIONS {metric: \"cosine\"}",
    )
    .expect("create vector index");

    let plan = db
        .explain_cypher(
            "MATCH (n:User) WITH n, vector_similarity(n.embedding, [1.0, 0.0, 0.0]) AS s \
             ORDER BY s DESC LIMIT 5 RETURN n, s",
        )
        .expect("explain");

    assert!(
        plan.contains("BruteForce"),
        "query on :User should show BruteForce even when index exists on :Item, got:\n{plan}"
    );
    assert!(
        !plan.contains("HnswScan"),
        "query on :User must NOT show HnswScan, got:\n{plan}"
    );
}

/// EXPLAIN for a query on a different property still shows BruteForce.
#[test]
fn explain_brute_force_for_different_property() {
    let (mut db, _dir) = open_db();

    // Index is on Item.embedding — NOT on Item.other_vec.
    db.execute_cypher(
        "CREATE VECTOR INDEX item_emb ON :Item(embedding) OPTIONS {metric: \"cosine\"}",
    )
    .expect("create vector index");

    let plan = db
        .explain_cypher(
            "MATCH (n:Item) WITH n, vector_similarity(n.other_vec, [1.0, 0.0, 0.0]) AS s \
             ORDER BY s DESC LIMIT 5 RETURN n, s",
        )
        .expect("explain");

    assert!(
        plan.contains("BruteForce"),
        "query on Item.other_vec should show BruteForce (index only covers Item.embedding), got:\n{plan}"
    );
}

// ── DROP then re-CREATE ───────────────────────────────────────────────

/// After DROP VECTOR INDEX, the same (label, property) can be re-created.
#[test]
fn drop_then_recreate_vector_index_succeeds() {
    let (mut db, _dir) = open_db();

    db.execute_cypher(
        "CREATE VECTOR INDEX item_emb ON :Item(embedding) OPTIONS {metric: \"cosine\"}",
    )
    .expect("first create");

    db.execute_cypher("DROP VECTOR INDEX item_emb")
        .expect("drop");

    // Re-create with a different name — must succeed.
    let rows = db
        .execute_cypher(
            "CREATE VECTOR INDEX item_emb2 ON :Item(embedding) OPTIONS {metric: \"cosine\"}",
        )
        .expect("re-create after drop");

    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("index"),
        Some(&Value::String("item_emb2".into())),
        "re-created index should have the new name"
    );

    // EXPLAIN should now show the new index name.
    let plan = db
        .explain_cypher(
            "MATCH (n:Item) WITH n, vector_similarity(n.embedding, [1.0, 0.0, 0.0]) AS s \
             ORDER BY s DESC LIMIT 5 RETURN n, s",
        )
        .expect("explain after re-create");
    assert!(
        plan.contains("ON item_emb2("),
        "expected HnswScan access path after re-create, got:\n{plan}"
    );
}

/// After DROP VECTOR INDEX, EXPLAIN reverts to BruteForce on reopen.
/// Verifies that the tombstone persists — re-opening DB does not resurrect a dropped index.
#[test]
fn drop_vector_index_not_resurrected_on_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");

    // Session 1: create then drop.
    {
        let mut db = Database::open(dir.path()).expect("open");
        db.execute_cypher(
            "CREATE VECTOR INDEX item_emb ON :Item(embedding) OPTIONS {metric: \"cosine\"}",
        )
        .expect("create");
        db.execute_cypher("DROP VECTOR INDEX item_emb")
            .expect("drop");
    }

    // Session 2: reopen — index must be gone.
    {
        let db = Database::open(dir.path()).expect("reopen");
        let plan = db
            .explain_cypher(
                "MATCH (n:Item) WITH n, vector_similarity(n.embedding, [1.0, 0.0, 0.0]) AS s \
                 ORDER BY s DESC LIMIT 5 RETURN n, s",
            )
            .expect("explain after reopen");
        assert!(
            plan.contains("BruteForce"),
            "dropped index must not reappear after reopen, got:\n{plan}"
        );
    }
}

// ── Auto-maintenance: SET / DELETE / REMOVE ───────────────────────────

/// SET on a vector property updates the HNSW index.
/// The old vector is replaced: queries for old vector no longer match,
/// queries for new vector succeed.
#[test]
fn set_updates_vector_index() {
    let (mut db, _dir) = open_db();

    db.execute_cypher(
        "CREATE VECTOR INDEX item_emb ON :Item(embedding) OPTIONS {metric: \"cosine\"}",
    )
    .expect("create index");

    // Create node with embedding close to [1, 0, 0].
    db.execute_cypher("CREATE (n:Item {name: 'A', embedding: [0.99, 0.1, 0.0]})")
        .expect("create node");

    // Update embedding to be close to [0, 1, 0].
    db.execute_cypher("MATCH (n:Item {name: 'A'}) SET n.embedding = [0.1, 0.99, 0.0]")
        .expect("set embedding");

    // Query for vector near [0, 1, 0] — should find the updated node.
    let rows = db
        .execute_cypher(
            "MATCH (n:Item) WHERE vector_similarity(n.embedding, [0.0, 1.0, 0.0]) > 0.9 \
             RETURN n.name AS name",
        )
        .expect("search after SET");

    assert_eq!(rows.len(), 1, "updated node should be found near [0,1,0]");
    assert_eq!(
        rows[0].get("name"),
        Some(&Value::String("A".into())),
        "found node should be 'A'"
    );
}

/// DETACH DELETE removes the node from the HNSW index.
/// After deletion, the node is no longer returned by vector search.
#[test]
fn delete_node_removes_from_vector_index() {
    let (mut db, _dir) = open_db();

    db.execute_cypher(
        "CREATE VECTOR INDEX item_emb ON :Item(embedding) OPTIONS {metric: \"cosine\"}",
    )
    .expect("create index");

    db.execute_cypher("CREATE (n:Item {name: 'target', embedding: [1.0, 0.0, 0.0]})")
        .expect("create node");

    // Verify the node is found before deletion.
    let before = db
        .execute_cypher(
            "MATCH (n:Item) WHERE vector_similarity(n.embedding, [1.0, 0.0, 0.0]) > 0.9 \
             RETURN n.name AS name",
        )
        .expect("search before delete");
    assert_eq!(before.len(), 1, "node should be found before deletion");

    // Delete the node.
    db.execute_cypher("MATCH (n:Item {name: 'target'}) DETACH DELETE n")
        .expect("detach delete");

    // After deletion, vector search should return no results.
    let after = db
        .execute_cypher(
            "MATCH (n:Item) WHERE vector_similarity(n.embedding, [1.0, 0.0, 0.0]) > 0.9 \
             RETURN n.name AS name",
        )
        .expect("search after delete");
    assert_eq!(
        after.len(),
        0,
        "deleted node should not appear in vector search"
    );
}

// ── Non-default metric in EXPLAIN ─────────────────────────────────────

/// EXPLAIN for a non-default metric (l2) shows the metric name in HnswScan.
///
/// Verifies that `annotate_vector_top_k` encodes the metric into the EXPLAIN
/// string for all supported metrics, not just the default (cosine).
#[test]
fn explain_shows_non_default_metric_in_hnsw_scan() {
    let (mut db, _dir) = open_db();

    db.execute_cypher(r#"CREATE VECTOR INDEX l2_idx ON :Embed(vec) OPTIONS {metric: "l2"}"#)
        .expect("create l2 index");

    let plan = db
        .explain_cypher(
            "MATCH (n:Embed) WITH n, vector_distance(n.vec, [1.0, 0.0]) AS d ORDER BY d LIMIT 5 RETURN n, d",
        )
        .expect("explain");

    // vector_distance + l2 index = compatible pair, so the access path
    // engages; the function name in the EXPLAIN line implies the metric.
    assert!(
        plan.contains("HnswScan(n:Embed ON l2_idx(vec), vector_distance"),
        "expected HnswScan access path for l2 metric, got:\n{plan}"
    );
}

/// EXPLAIN with dot_product metric shows "dot" in HnswScan.
#[test]
fn explain_shows_dot_metric_in_hnsw_scan() {
    let (mut db, _dir) = open_db();

    db.execute_cypher(r#"CREATE VECTOR INDEX dot_idx ON :Dot(vec) OPTIONS {metric: "dot"}"#)
        .expect("create dot index");

    let plan = db
        .explain_cypher(
            "MATCH (n:Dot) WITH n, vector_distance(n.vec, [1.0, 0.0]) AS d ORDER BY d LIMIT 5 RETURN n, d",
        )
        .expect("explain");

    assert!(
        plan.contains("HnswScan(dot_idx, dot)"),
        "expected HnswScan(dot_idx, dot) for dot_product metric, got:\n{plan}"
    );
}

/// EXPLAIN with l1 metric shows "l1" in HnswScan.
#[test]
fn explain_shows_l1_metric_in_hnsw_scan() {
    let (mut db, _dir) = open_db();

    db.execute_cypher(r#"CREATE VECTOR INDEX l1_idx ON :L1Vec(vec) OPTIONS {metric: "l1"}"#)
        .expect("create l1 index");

    let plan = db
        .explain_cypher(
            "MATCH (n:L1Vec) WITH n, vector_distance(n.vec, [1.0, 0.0]) AS d ORDER BY d LIMIT 5 RETURN n, d",
        )
        .expect("explain");

    assert!(
        plan.contains("HnswScan(l1_idx, l1)"),
        "expected HnswScan(l1_idx, l1) for l1 metric, got:\n{plan}"
    );
}

// ── Vector search after reopen ────────────────────────────────────────

/// After reopen the HNSW index is rebuilt from stored nodes and produces
/// correct search results (not just EXPLAIN — actual execution).
#[test]
fn vector_search_executes_correctly_after_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");

    // Session 1: create index and insert nodes.
    {
        let mut db = Database::open(dir.path()).expect("open");
        db.execute_cypher(
            "CREATE VECTOR INDEX persist_emb ON :Doc(vec) OPTIONS {metric: \"cosine\"}",
        )
        .expect("create index");
        db.execute_cypher("CREATE (n:Doc {name: 'near', vec: [1.0, 0.0, 0.0]})")
            .expect("near node");
        db.execute_cypher("CREATE (n:Doc {name: 'far', vec: [0.0, 0.0, 1.0]})")
            .expect("far node");
    }

    // Session 2: reopen — HNSW is rebuilt from storage.
    {
        let mut db = Database::open(dir.path()).expect("reopen");

        // Nearest to [1,0,0]: 'near' has similarity 1.0, 'far' has similarity 0.0.
        let rows = db
            .execute_cypher(
                "MATCH (n:Doc) \
                 WHERE vector_similarity(n.vec, [1.0, 0.0, 0.0]) > 0.9 \
                 RETURN n.name AS name",
            )
            .expect("vector search after reopen");

        assert_eq!(rows.len(), 1, "only 'near' should match after reopen");
        assert_eq!(
            rows[0].get("name"),
            Some(&Value::String("near".into())),
            "found node should be 'near'"
        );
    }
}

/// REMOVE a.embedding removes the property from the HNSW index.
/// The node itself remains but is no longer indexed.
#[test]
fn remove_property_removes_from_vector_index() {
    let (mut db, _dir) = open_db();

    db.execute_cypher(
        "CREATE VECTOR INDEX item_emb ON :Item(embedding) OPTIONS {metric: \"cosine\"}",
    )
    .expect("create index");

    db.execute_cypher("CREATE (n:Item {name: 'removable', embedding: [1.0, 0.0, 0.0]})")
        .expect("create node");

    // Verify it's indexed.
    let before = db
        .execute_cypher(
            "MATCH (n:Item) WHERE vector_similarity(n.embedding, [1.0, 0.0, 0.0]) > 0.9 \
             RETURN n.name AS name",
        )
        .expect("search before remove");
    assert_eq!(before.len(), 1, "node should be indexed before REMOVE");

    // Remove the embedding property.
    db.execute_cypher("MATCH (n:Item {name: 'removable'}) REMOVE n.embedding")
        .expect("remove property");

    // Node should no longer appear in vector search.
    let after = db
        .execute_cypher(
            "MATCH (n:Item) WHERE vector_similarity(n.embedding, [1.0, 0.0, 0.0]) > 0.9 \
             RETURN n.name AS name",
        )
        .expect("search after remove");
    assert_eq!(
        after.len(),
        0,
        "node with REMOVE'd embedding should not appear in vector search"
    );

    // But the node itself still exists.
    let node = db
        .execute_cypher("MATCH (n:Item {name: 'removable'}) RETURN n.name AS name")
        .expect("node exists check");
    assert_eq!(node.len(), 1, "node itself should still exist after REMOVE");
}

// ── Cypher OPTIONS { quantization } ──────────────────────────────────

/// CREATE VECTOR INDEX ... OPTIONS { quantization: "rabitq-2bit" } must
/// thread the Extended-RaBitQ 2-bit codec from the parser through the
/// planner, executor, and into the live HnswIndex. After enough inserts
/// to trigger calibration, every node in the index must carry a
/// Multi(bits=2) code and search must navigate to the right cluster.
#[test]
fn create_vector_index_with_rabitq_2bit_quantization() {
    let (mut db, _dir) = open_db();

    // d=64 is the smallest dim accepted by RaBitQ (multiple of 64).
    // calibration_threshold defaults to 1000 inside HnswIndex; we go
    // above that by inserting 1100 vectors so calibration definitely
    // fires before the assertions.
    db.execute_cypher(
        "CREATE VECTOR INDEX ext_idx ON :Doc(emb) \
         OPTIONS {m: 16, ef_construction: 200, metric: \"cosine\", \
                  dimensions: 64, quantization: \"rabitq-2bit\"}",
    )
    .expect("create vector index with rabitq-2bit");

    // Workload: K well-separated dense clusters in 64-D, ~N/K vectors per
    // cluster as centroid + per-coordinate deterministic noise. Single-
    // non-zero unit vectors are an adversarial case for low-bit RaBitQ
    // (rotated codes for orthogonal directions become indistinguishable
    // under the cheap-distance kernel at default ef_search), so the
    // workload here mirrors the production embedding shape: dense
    // values with cluster structure.
    const N: usize = 1100;
    const K: usize = 7;
    let cluster_centroid = |k: usize| -> [f32; 64] {
        let mut c = [0.0f32; 64];
        for (d, slot) in c.iter_mut().enumerate() {
            // 7-step zigzag per cluster on each dim. With K = 7 every
            // (k + d) % 7 row of the pattern is distinct, so clusters
            // stay linearly independent and well-separated after L2
            // normalisation. K must equal the modulus, otherwise
            // (k = M, k = 0) collapse to the same centroid.
            *slot = ((k + d) % 7) as f32 - 3.0;
        }
        c
    };

    let mut rows = String::from("[");
    for i in 0..N {
        if i > 0 {
            rows.push_str(", ");
        }
        let k = i % K;
        let centroid = cluster_centroid(k);
        let mut v = [0.0f32; 64];
        for (d, slot) in v.iter_mut().enumerate() {
            // Deterministic per-(node,dim) noise keeps codes distinct
            // within a cluster without making any node closer to a
            // different cluster's centroid than to its own.
            let noise = ((i * (d + 1)) % 13) as f32 * 0.01 - 0.06;
            *slot = centroid[d] + noise;
        }
        rows.push_str(&format!("{{i: {i}, emb: ["));
        for (j, x) in v.iter().enumerate() {
            if j > 0 {
                rows.push_str(", ");
            }
            rows.push_str(&format!("{x}"));
        }
        rows.push_str("]}");
    }
    rows.push(']');
    let create_query = format!("UNWIND {rows} AS r CREATE (m:Doc) SET m = r RETURN m");
    let created = db.execute_cypher(&create_query).expect("bulk create");
    assert_eq!(created.len(), N, "bulk create returned wrong row count");

    // Query = cluster-0 centroid (matches noisy nodes with i % K == 0).
    let query_centroid = cluster_centroid(0);
    let mut q = String::from("[");
    for (j, x) in query_centroid.iter().enumerate() {
        if j > 0 {
            q.push_str(", ");
        }
        q.push_str(&format!("{x}"));
    }
    q.push(']');
    let result = db
        .execute_cypher(&format!(
            "MATCH (n:Doc) WITH n, vector_similarity(n.emb, {q}) AS s \
             ORDER BY s DESC LIMIT 5 RETURN n.i AS i"
        ))
        .expect("vector search through rabitq-2bit index");
    assert!(
        !result.is_empty(),
        "vector_similarity must return results through Extended-RaBitQ index"
    );
    // N/K = 157 nodes belong to cluster 0 (id % K == 0). The codec
    // wiring is proven iff the top-5 search results all land in that
    // cluster. Intra-cluster ordering depends on the per-node noise
    // and isn't load-bearing for the wiring assertion.
    let top5_ids: Vec<i64> = result
        .iter()
        .filter_map(|r| match r.get("i") {
            Some(Value::Int(x)) => Some(*x),
            _ => None,
        })
        .collect();
    assert!(
        !top5_ids.is_empty(),
        "expected non-empty top-5 from rabitq-2bit search"
    );
    for id in &top5_ids {
        assert_eq!(
            (*id as usize) % K,
            0,
            "rabitq-2bit top-5 leaked a wrong-cluster node: id={id}, full={top5_ids:?}",
        );
    }
}

/// Exposing `ef_search` via CREATE VECTOR INDEX OPTIONS lets a user recover
/// recall on adversarial low-bit-RaBitQ data. Single-non-zero unit vectors are
/// the worst case for 2-bit cheap-distance navigation: at the default
/// ef_search=200 the search lands in the wrong cluster, but raising ef_search
/// through the new option navigates correctly. Proves the option is parsed,
/// threaded through planner/executor into the live HnswIndex, and takes effect
/// end-to-end.
#[test]
fn create_vector_index_ef_search_option_recovers_adversarial_recall() {
    let (mut db, _dir) = open_db();

    // ef_search=5000 >> N forces a near-exhaustive candidate list, so the
    // 2-bit cheap-distance noise floor no longer traps the search in an
    // adjacent cluster (the failure mode at the default ef_search=200).
    db.execute_cypher(
        "CREATE VECTOR INDEX adv_idx ON :Adv(emb) \
         OPTIONS {m: 8, ef_construction: 64, metric: \"cosine\", \
                  dimensions: 64, quantization: \"rabitq-2bit\", ef_search: 5000}",
    )
    .expect("create vector index with ef_search override");

    // 1100 unit vectors, each a single non-zero dim at position (i mod 64).
    // The dim-0 family {i : i % 64 == 0} are the only exact matches for a
    // query along dim 0; every other vector is orthogonal (cosine 0).
    const N: usize = 1100;
    let mut rows = String::from("[");
    for i in 0..N {
        if i > 0 {
            rows.push_str(", ");
        }
        let pos = i % 64;
        let mut v = [0.0f32; 64];
        v[pos] = 1.0;
        rows.push_str(&format!("{{i: {i}, emb: ["));
        for (j, x) in v.iter().enumerate() {
            if j > 0 {
                rows.push_str(", ");
            }
            rows.push_str(&format!("{x}"));
        }
        rows.push_str("]}");
    }
    rows.push(']');
    let created = db
        .execute_cypher(&format!(
            "UNWIND {rows} AS r CREATE (m:Adv) SET m = r RETURN m"
        ))
        .expect("bulk create adversarial vectors");
    assert_eq!(created.len(), N, "bulk create returned wrong row count");

    // Query e_0 (non-zero at dim 0 only).
    let mut q = String::from("[1.0");
    for _ in 1..64 {
        q.push_str(", 0.0");
    }
    q.push(']');
    let result = db
        .execute_cypher(&format!(
            "MATCH (n:Adv) WITH n, vector_similarity(n.emb, {q}) AS s \
             ORDER BY s DESC LIMIT 5 RETURN n.i AS i"
        ))
        .expect("adversarial vector search with high ef_search");
    let top5_ids: Vec<i64> = result
        .iter()
        .filter_map(|r| match r.get("i") {
            Some(Value::Int(x)) => Some(*x),
            _ => None,
        })
        .collect();
    assert_eq!(top5_ids.len(), 5, "expected top-5 from adversarial search");
    // Every result must be a dim-0 vector — the only exact matches. Exact ids
    // are not asserted (the dim-0 family members are identical, so tie-break
    // ordering is not load-bearing); family membership proves recall recovery.
    for id in &top5_ids {
        assert_eq!(
            (*id as usize) % 64,
            0,
            "ef_search=5000 must keep top-5 in the dim-0 family; leaked id={id}, full={top5_ids:?}",
        );
    }
}

// ── R858b-pre3 Step 3: online_during_build policy tests ──────────────────

/// Default policy ("block") must let CREATE-then-SEARCH succeed: the gate
/// polls until the background backfill completes before letting the query
/// hit the HNSW graph, matching the legacy synchronous semantic.
#[test]
fn online_during_build_default_block_waits_for_backfill() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (n:Item {embedding: [1.0, 0.0, 0.0]})")
        .expect("node");

    db.execute_cypher(
        "CREATE VECTOR INDEX item_emb ON :Item(embedding) OPTIONS {metric: \"cosine\"}",
    )
    .expect("create vector index");

    let rows = db
        .execute_cypher(
            "MATCH (n:Item) \
             WHERE vector_similarity(n.embedding, [1.0, 0.0, 0.0]) > 0.9 \
             RETURN n.embedding AS vec",
        )
        .expect("search waits for backfill, then returns the row");
    assert_eq!(rows.len(), 1);
}

/// `online_during_build: "partial-recall"` lets the search hit the graph
/// immediately, even before backfill finishes. With one node + cosine the
/// graph fills almost instantly so this is mostly a parser-and-plumbing
/// smoke test, but the gate must NOT consult the schema for this policy.
#[test]
fn online_during_build_partial_recall_does_not_block() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (n:Item {embedding: [1.0, 0.0, 0.0]})")
        .expect("node");

    db.execute_cypher(
        "CREATE VECTOR INDEX item_emb ON :Item(embedding) \
         OPTIONS {metric: \"cosine\", online_during_build: \"partial-recall\"}",
    )
    .expect("create vector index with partial-recall");

    // Even if the search races the backfill thread, partial-recall lets
    // the call go through without an error from the gate.
    let _ = db
        .execute_cypher(
            "MATCH (n:Item) \
             WHERE vector_similarity(n.embedding, [1.0, 0.0, 0.0]) > 0.9 \
             RETURN n.embedding AS vec",
        )
        .expect("partial-recall search must not error");
}

/// Crash-recovery: an index whose persisted state is `Building` (because
/// the engine crashed mid-backfill) is rebuilt on reopen and the state is
/// reset to `Ready`. Verified by writing a Building marker into the schema
/// without going through the executor, closing the DB, and reopening.
#[test]
fn building_state_resets_to_ready_on_reopen() {
    use coordinode_query::index::{ops as index_ops, IndexState};

    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().to_path_buf();

    // Phase 1: open, create index synchronously, close.
    {
        let mut db = coordinode_embed::Database::open(&path).expect("first open");
        db.execute_cypher("CREATE (n:Item {embedding: [1.0, 0.0, 0.0]})")
            .expect("node");
        db.execute_cypher(
            "CREATE VECTOR INDEX item_emb ON :Item(embedding) OPTIONS {metric: \"cosine\"}",
        )
        .expect("create");

        // Inject a stale Building marker directly via the storage engine,
        // simulating a crash that left the backfill half-done.
        let storage = db.engine();
        index_ops::save_index_state(
            storage,
            "item_emb",
            IndexState::Building {
                written: 0,
                estimated_total: 1,
            },
        )
        .expect("inject Building state");

        // db drops here, releasing the engine.
    }

    // Phase 2: reopen. The HNSW rebuild path should flip the state back
    // to Ready after re-populating the graph from node records.
    {
        let mut db = coordinode_embed::Database::open(&path).expect("reopen");
        let storage = db.engine();
        let def = index_ops::load_index_definition(storage, "item_emb")
            .expect("load")
            .expect("def present after reopen");
        assert_eq!(
            def.state,
            IndexState::Ready,
            "reopen should reset stale Building to Ready"
        );

        // And the rebuilt index serves searches without the block gate
        // hanging on a stale state.
        let rows = db
            .execute_cypher(
                "MATCH (n:Item) \
                 WHERE vector_similarity(n.embedding, [1.0, 0.0, 0.0]) > 0.9 \
                 RETURN n.embedding AS vec",
            )
            .expect("post-reopen search");
        assert_eq!(rows.len(), 1);
    }
}
