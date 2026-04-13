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
