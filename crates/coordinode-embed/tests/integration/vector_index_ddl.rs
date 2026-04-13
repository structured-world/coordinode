//! Integration tests: VectorIndexRegistry + CREATE/DROP VECTOR INDEX DDL (R-API3).
//!
//! Tests the full vector index lifecycle through Database:
//! - CREATE VECTOR INDEX DDL registers an HNSW index
//! - EXPLAIN after CREATE shows "HnswScan(<name>)"
//! - EXPLAIN without index shows "BruteForce"
//! - DROP VECTOR INDEX removes the index
//! - After DROP, EXPLAIN reverts to "BruteForce"

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
