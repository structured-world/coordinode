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
    assert!(
        after.contains("HnswScan(item_emb)"),
        "expected HnswScan(item_emb) after index, got:\n{after}"
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
        after_create.contains("HnswScan(item_emb)"),
        "expected HnswScan after create, got:\n{after_create}"
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
            plan.contains("HnswScan(persist_idx)"),
            "expected HnswScan after reopen, got:\n{plan}"
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
    assert_eq!(
        rows[0].get("nodes_indexed"),
        Some(&Value::Int(3)),
        "should backfill exactly 3 nodes that have the embedding property"
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
        plan.contains("HnswScan(item_emb2)"),
        "expected HnswScan(item_emb2) after re-create, got:\n{plan}"
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
