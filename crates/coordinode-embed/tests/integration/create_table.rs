//! CREATE TABLE end-to-end: parse -> plan -> execute over real storage (R901).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_core::graph::types::Value;
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
fn row_table_insert_and_match_by_primary_key() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).expect("open db");
    db.execute_cypher("CREATE TABLE Account (id BIGINT PRIMARY KEY, name STRING)")
        .expect("create");
    db.execute_cypher("CREATE (a:Account {id: 1, name: 'Alice'})")
        .expect("insert");

    let rows = db
        .execute_cypher("MATCH (a:Account {id: 1}) RETURN a.name AS name")
        .expect("match");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("name"), Some(&Value::String("Alice".into())));
}

#[test]
fn row_table_primary_key_is_stable_identity() {
    // The same primary key maps to the same node (identity / upsert-by-key),
    // not two separate rows.
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).expect("open db");
    db.execute_cypher("CREATE TABLE Account (id BIGINT PRIMARY KEY, name STRING)")
        .expect("create");
    db.execute_cypher("CREATE (a:Account {id: 7, name: 'First'})")
        .expect("insert 1");
    db.execute_cypher("CREATE (a:Account {id: 7, name: 'Second'})")
        .expect("insert 2");

    let rows = db
        .execute_cypher("MATCH (a:Account {id: 7}) RETURN a.name AS name")
        .expect("match");
    assert_eq!(rows.len(), 1, "same PK must resolve to one node, not two");
    assert_eq!(rows[0].get("name"), Some(&Value::String("Second".into())));
}

#[test]
fn row_table_data_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut db = Database::open(dir.path()).expect("open db");
        db.execute_cypher("CREATE TABLE Account (id BIGINT PRIMARY KEY, name STRING)")
            .expect("create");
        db.execute_cypher("CREATE (a:Account {id: 1, name: 'Alice'})")
            .expect("insert");
    }
    let mut db = Database::open(dir.path()).expect("reopen db");
    let rows = db
        .execute_cypher("MATCH (a:Account {id: 1}) RETURN a.name AS name")
        .expect("match after reopen");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("name"), Some(&Value::String("Alice".into())));
}

#[test]
fn columnar_table_insert_and_match_by_primary_key() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).expect("open db");
    db.execute_cypher(
        "CREATE TABLE Trade (id BIGINT PRIMARY KEY, sym STRING NOT NULL) STORAGE COLUMNAR",
    )
    .expect("create");
    db.execute_cypher("CREATE (t:Trade {id: 1, sym: 'AAPL'})")
        .expect("insert 1");
    db.execute_cypher("CREATE (t:Trade {id: 2, sym: 'MSFT'})")
        .expect("insert 2");

    let rows = db
        .execute_cypher("MATCH (t:Trade {id: 1}) RETURN t.sym AS sym")
        .expect("match");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("sym"), Some(&Value::String("AAPL".into())));

    let all = db
        .execute_cypher("MATCH (t:Trade) RETURN t.sym AS sym")
        .expect("scan all");
    assert_eq!(all.len(), 2, "both columnar rows must be scanned");
}

#[test]
fn columnar_table_data_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut db = Database::open(dir.path()).expect("open db");
        db.execute_cypher(
            "CREATE TABLE Trade (id BIGINT PRIMARY KEY, sym STRING) STORAGE COLUMNAR",
        )
        .expect("create");
        db.execute_cypher("CREATE (t:Trade {id: 1, sym: 'AAPL'})")
            .expect("insert");
    }
    let mut db = Database::open(dir.path()).expect("reopen db");
    let rows = db
        .execute_cypher("MATCH (t:Trade {id: 1}) RETURN t.sym AS sym")
        .expect("match after reopen");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("sym"), Some(&Value::String("AAPL".into())));
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

#[test]
fn sql_insert_and_select_on_row_table() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).expect("open db");
    // DDL via cypher; DML via SQL — both over the one engine + IR.
    db.execute_cypher("CREATE TABLE Account (id BIGINT PRIMARY KEY, name STRING)")
        .expect("create table");
    db.execute_sql("INSERT INTO Account (id, name) VALUES (1, 'Alice')")
        .expect("sql insert");
    db.execute_sql("INSERT INTO Account (id, name) VALUES (2, 'Bob')")
        .expect("sql insert 2");

    let rows = db
        .execute_sql("SELECT name FROM Account WHERE id = 1")
        .expect("sql select");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("name"), Some(&Value::String("Alice".into())));

    let all = db
        .execute_sql("SELECT id FROM Account")
        .expect("sql select all");
    assert_eq!(all.len(), 2);
}

#[test]
fn sql_select_reads_rows_written_by_cypher() {
    // SQL and Cypher are two dialects over one store: a Cypher-created row is
    // visible to a SQL SELECT on the same table.
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).expect("open db");
    db.execute_cypher("CREATE TABLE Account (id BIGINT PRIMARY KEY, name STRING)")
        .expect("create");
    db.execute_cypher("CREATE (a:Account {id: 5, name: 'Carol'})")
        .expect("cypher insert");

    let rows = db
        .execute_sql("SELECT name FROM Account WHERE id = 5")
        .expect("sql select");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("name"), Some(&Value::String("Carol".into())));
}
