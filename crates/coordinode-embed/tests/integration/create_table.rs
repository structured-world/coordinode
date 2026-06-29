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
fn drop_columnar_table_removes_tree_and_allows_recreate() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).expect("open db");

    db.execute_cypher("CREATE TABLE Trade (id BIGINT PRIMARY KEY, qty INT) STORAGE COLUMNAR")
        .expect("create");
    assert!(db.engine().columnar_table_tree("Trade").is_some());

    let rows = db.execute_cypher("DROP TABLE Trade").expect("drop");
    assert_eq!(rows.len(), 1);
    // Tree gone, schema tombstoned.
    assert!(db.engine().columnar_table_tree("Trade").is_none());

    // The name is free again: re-create succeeds.
    db.execute_cypher("CREATE TABLE Trade (id BIGINT PRIMARY KEY) STORAGE COLUMNAR")
        .expect("recreate");
    assert!(db.engine().columnar_table_tree("Trade").is_some());
}

#[test]
fn drop_unknown_table_errors() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).expect("open db");
    assert!(db.execute_cypher("DROP TABLE Nope").is_err());
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

#[test]
fn create_table_rejects_unknown_column_type() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).expect("open db");

    // QUATERNION is not a supported column type.
    assert!(db
        .execute_cypher("CREATE TABLE Bad (id BIGINT PRIMARY KEY, q QUATERNION)")
        .is_err());
    // The failed CREATE left no table behind.
    assert!(db.engine().columnar_table_tree("Bad").is_none());
}

#[test]
fn drop_table_rejects_non_table_label() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).expect("open db");

    // A plain graph label is not a table.
    db.execute_cypher("CREATE (n:Person {name: 'Alice'})")
        .expect("create node");
    assert!(db.execute_cypher("DROP TABLE Person").is_err());
}

#[test]
fn columnar_table_survives_database_reopen() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut db = Database::open(dir.path()).expect("open db");
        db.execute_cypher(
            "CREATE TABLE Trade (id BIGINT PRIMARY KEY, sym STRING NOT NULL) STORAGE COLUMNAR",
        )
        .expect("create");
        assert!(db.engine().columnar_table_tree("Trade").is_some());
    }

    // Reopen the same directory: the columnar tree is recovered AND the schema
    // is still registered (a duplicate CREATE is rejected).
    let mut db = Database::open(dir.path()).expect("reopen db");
    assert!(
        db.engine().columnar_table_tree("Trade").is_some(),
        "columnar tree must survive reopen"
    );
    assert!(
        db.execute_cypher("CREATE TABLE Trade (id BIGINT PRIMARY KEY) STORAGE COLUMNAR")
            .is_err(),
        "table schema must survive reopen (duplicate rejected)"
    );
}

#[test]
fn dropped_table_stays_dropped_after_reopen() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut db = Database::open(dir.path()).expect("open db");
        db.execute_cypher("CREATE TABLE T (id BIGINT PRIMARY KEY) STORAGE COLUMNAR")
            .expect("create");
        db.execute_cypher("DROP TABLE T").expect("drop");
    }

    // After reopen the drop persists: the name is free to re-create.
    let mut db = Database::open(dir.path()).expect("reopen db");
    assert!(db.engine().columnar_table_tree("T").is_none());
    db.execute_cypher("CREATE TABLE T (id BIGINT PRIMARY KEY) STORAGE COLUMNAR")
        .expect("recreate after reopen");
}
