//! CREATE TABLE end-to-end: parse -> plan -> execute over real storage (R901).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_embed::db::Database;

#[test]
fn create_columnar_table_creates_tree_and_persists_schema() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).expect("open db");

    let rows = db
        .execute_cypher(
            "CREATE TABLE Trade (trade_id BIGINT PRIMARY KEY, symbol STRING NOT NULL, qty INT) \
             STORAGE COLUMNAR",
        )
        .expect("create columnar table");
    assert_eq!(rows.len(), 1);

    // The per-table columnar tree is opened at CREATE TABLE time.
    assert!(db.engine().columnar_table_tree("Trade").is_some());

    // Re-creating the same table is rejected (schema persisted).
    assert!(db
        .execute_cypher("CREATE TABLE Trade (trade_id BIGINT PRIMARY KEY) STORAGE COLUMNAR")
        .is_err());
}

#[test]
fn create_row_table_persists_without_columnar_tree() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).expect("open db");

    db.execute_cypher("CREATE TABLE Account (id BIGINT PRIMARY KEY, name STRING)")
        .expect("create row table");

    // A ROW table stays on the node path; no columnar tree is created.
    assert!(db.engine().columnar_table_tree("Account").is_none());
}

#[test]
fn create_table_requires_primary_key() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).expect("open db");

    // No PRIMARY KEY column -> rejected.
    assert!(db
        .execute_cypher("CREATE TABLE NoPk (a INT, b STRING)")
        .is_err());
}
