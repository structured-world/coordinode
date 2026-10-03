//! CREATE TABLE end-to-end: parse -> plan -> execute over real storage.

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
    assert!(
        db.execute_cypher("CREATE TABLE Trade (trade_id BIGINT PRIMARY KEY) STORAGE COLUMNAR")
            .is_err()
    );
}

/// A table with a constraint is not dropped from under it: the constraint
/// would keep its name and its index with nothing to constrain. Once the
/// constraint is dropped, the table drops and the name is free again.
#[test]
fn a_table_with_a_constraint_is_not_dropped_from_under_it() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).expect("open db");
    db.execute_cypher("CREATE TABLE Acct (id BIGINT PRIMARY KEY, email STRING)")
        .expect("create table");
    db.execute_cypher("CREATE CONSTRAINT acct_email FOR (a:Acct) REQUIRE a.email IS UNIQUE")
        .expect("constraint");

    let err = db
        .execute_cypher("DROP TABLE Acct")
        .expect_err("the constraint depends on the table");
    assert!(err.to_string().contains("acct_email"), "{err}");
    assert_eq!(db.constraints().expect("constraints").len(), 1);

    db.execute_cypher("DROP CONSTRAINT acct_email")
        .expect("drop constraint");
    db.execute_cypher("DROP TABLE Acct").expect("drop table");
    db.execute_cypher("CREATE TABLE Acct (id BIGINT PRIMARY KEY, email STRING)")
        .expect("the name is free");
    db.execute_cypher("CREATE CONSTRAINT acct_email FOR (a:Acct) REQUIRE a.email IS UNIQUE")
        .expect("so is the constraint name");
}

/// UNIQUE on a column is refused before anything of the table is written, in
/// either layout: no schema, no index, no columnar tree, nothing after a
/// reopen, and the same table without the clause can still be created.
#[test]
fn a_unique_column_is_refused_and_leaves_nothing() {
    for storage in ["ROW", "COLUMNAR"] {
        let dir = tempfile::tempdir().unwrap();
        {
            let mut db = Database::open(dir.path()).expect("open db");
            let err = db
                .execute_cypher(&format!(
                    "CREATE TABLE Acct (id BIGINT PRIMARY KEY, email STRING UNIQUE) \
                     STORAGE {storage}"
                ))
                .expect_err("UNIQUE on a column is refused");
            assert!(err.to_string().contains("UNIQUE"), "{storage}: {err}");
            assert!(
                db.engine().columnar_table_tree("Acct").is_none(),
                "{storage}"
            );
            assert!(db.label_schemas().expect("labels").is_empty(), "{storage}");
            assert!(
                db.constraints().expect("constraints").is_empty(),
                "{storage}"
            );
        }
        let mut db = Database::open(dir.path()).expect("reopen");
        assert!(
            db.engine().columnar_table_tree("Acct").is_none(),
            "{storage}"
        );
        assert!(db.label_schemas().expect("labels").is_empty(), "{storage}");
        db.execute_cypher(&format!(
            "CREATE TABLE Acct (id BIGINT PRIMARY KEY, email STRING) STORAGE {storage}"
        ))
        .unwrap_or_else(|e| panic!("{storage}: the name stays free: {e}"));
    }
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

/// A table without a declared key is keyed by its row id, the way a
/// collection without a custom `_id` is: every insert is a new row, even with
/// identical values, and each row is addressed by its own id.
#[test]
fn a_table_without_a_declared_key_is_keyed_by_its_row_id() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).expect("open db");
    db.execute_cypher("CREATE TABLE Event (kind STRING, n INT)")
        .expect("a table needs no declared key");
    db.execute_cypher("CREATE (:Event {kind: 'click', n: 1})")
        .expect("insert 1");
    db.execute_cypher("CREATE (:Event {kind: 'click', n: 1})")
        .expect("an identical row is a second row");

    let rows = db
        .execute_cypher("MATCH (e:Event) RETURN elementId(e) AS id")
        .expect("scan");
    assert_eq!(rows.len(), 2);
    let ids: Vec<&Value> = rows.iter().filter_map(|r| r.get("id")).collect();
    assert_ne!(ids[0], ids[1], "each row has its own id");

    let Value::String(first) = ids[0] else {
        panic!("elementId is a string, got {:?}", ids[0]);
    };
    let by_id = db
        .execute_cypher(&format!(
            "MATCH (e:Event) WHERE elementId(e) = '{first}' RETURN e.n AS n"
        ))
        .expect("lookup by id");
    assert_eq!(by_id.len(), 1, "the row id addresses exactly one row");
}

/// Two keys whose hashes collide are two rows. A key derived from a hash of
/// the key would give both the same row id, and the second insert would
/// silently overwrite the first; the pair below was found by search.
#[test]
fn keys_that_share_a_hash_are_two_rows() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).expect("open db");
    db.execute_cypher("CREATE TABLE T (k STRING PRIMARY KEY, v INT)")
        .expect("create");
    db.execute_cypher("CREATE (:T {k: '322c0144ad2e1418', v: 1})")
        .expect("insert first");
    db.execute_cypher("CREATE (:T {k: '1140bd0208eee55b', v: 2})")
        .expect("insert second");

    let rows = db
        .execute_cypher("MATCH (t:T) RETURN t.k AS k, t.v AS v ORDER BY v")
        .expect("scan");
    assert_eq!(rows.len(), 2, "both rows exist");
    assert_eq!(
        rows[0].get("k"),
        Some(&Value::String("322c0144ad2e1418".into()))
    );
    assert_eq!(
        rows[1].get("k"),
        Some(&Value::String("1140bd0208eee55b".into()))
    );
}

/// Inserting a key that already exists fails and leaves the stored row as it
/// was, through either dialect. The error names what went wrong.
#[test]
fn inserting_an_existing_key_fails_and_keeps_the_row() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).expect("open db");
    db.execute_cypher("CREATE TABLE Account (id BIGINT PRIMARY KEY, name STRING)")
        .expect("create");
    db.execute_cypher("CREATE (:Account {id: 7, name: 'First'})")
        .expect("insert");

    let err = db
        .execute_cypher("CREATE (:Account {id: 7, name: 'Second'})")
        .expect_err("a second row with the same key");
    assert!(
        err.to_string().contains("already exists"),
        "the error says the key exists: {err}"
    );
    let err = db
        .execute_sql("INSERT INTO Account (id, name) VALUES (7, 'Third')")
        .expect_err("the same through SQL");
    assert!(err.to_string().contains("already exists"), "{err}");

    let rows = db
        .execute_cypher("MATCH (a:Account {id: 7}) RETURN a.name AS name")
        .expect("match");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("name"), Some(&Value::String("First".into())));
}

/// One statement inserting the same key twice fails as a whole: neither row
/// is written.
#[test]
fn a_statement_repeating_a_key_writes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).expect("open db");
    db.execute_cypher("CREATE TABLE Account (id BIGINT PRIMARY KEY)")
        .expect("create");
    let err = db
        .execute_cypher("UNWIND [1, 2, 1] AS i CREATE (:Account {id: i})")
        .expect_err("the key repeats");
    assert!(err.to_string().contains("already exists"), "{err}");
    let rows = db
        .execute_cypher("MATCH (a:Account) RETURN a.id AS id")
        .expect("scan");
    assert!(
        rows.is_empty(),
        "the failed statement wrote nothing: {rows:?}"
    );
}

/// Writers racing on one key: exactly one row results, and every loser is
/// told the key exists.
#[test]
fn concurrent_inserts_of_one_key_leave_one_row() {
    const WRITERS: usize = 8;
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).expect("open db");
    db.execute_cypher("CREATE TABLE Account (id BIGINT PRIMARY KEY, w INT)")
        .expect("create");
    let db = &db;
    let outcomes: Vec<Result<(), String>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..WRITERS)
            .map(|w| {
                s.spawn(move || {
                    db.execute_cypher_shared(
                        &format!("CREATE (:Account {{id: 1, w: {w}}})"),
                        None,
                        None,
                        None,
                        None,
                    )
                    .map(|_| ())
                    .map_err(|e| e.to_string())
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert_eq!(
        outcomes.iter().filter(|o| o.is_ok()).count(),
        1,
        "exactly one writer inserts: {outcomes:?}"
    );
    for err in outcomes.iter().filter_map(|o| o.as_ref().err()) {
        assert!(err.contains("already exists"), "a loser is told why: {err}");
    }
}

/// A deleted row's key is free again; dropping the table frees all of them.
#[test]
fn deleting_a_row_or_dropping_the_table_frees_its_keys() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).expect("open db");
    db.execute_cypher("CREATE TABLE Account (id BIGINT PRIMARY KEY, name STRING)")
        .expect("create");
    db.execute_cypher("CREATE (:Account {id: 1, name: 'a'})")
        .expect("insert");
    db.execute_cypher("MATCH (a:Account {id: 1}) DELETE a")
        .expect("delete");
    db.execute_cypher("CREATE (:Account {id: 1, name: 'b'})")
        .expect("the key is free after its row is deleted");

    db.execute_cypher("DROP TABLE Account").expect("drop");
    db.execute_cypher("CREATE TABLE Account (id BIGINT PRIMARY KEY, name STRING)")
        .expect("recreate");
    assert!(
        db.execute_cypher("MATCH (a:Account) RETURN a")
            .expect("scan")
            .is_empty(),
        "the dropped table's rows went with it"
    );
    db.execute_cypher("CREATE (:Account {id: 1, name: 'c'})")
        .expect("the key is free after the table is dropped");
}

/// A lookup by key finds the row the key index names and still applies the
/// statement's other conditions, through inline filters and WHERE alike. A
/// value of another type than the key column answers as a scan would.
#[test]
fn a_key_lookup_answers_as_the_scan_would() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).expect("open db");
    db.execute_cypher("CREATE TABLE Account (id BIGINT PRIMARY KEY, name STRING)")
        .expect("create");
    db.execute_cypher(
        "UNWIND range(1, 50) AS i CREATE (:Account {id: i, name: 'n' + toString(i)})",
    )
    .expect("insert");

    let count = |db: &mut Database, q: &str| db.execute_cypher(q).expect(q).len();
    assert_eq!(count(&mut db, "MATCH (a:Account {id: 7}) RETURN a"), 1);
    assert_eq!(
        count(&mut db, "MATCH (a:Account {id: 7, name: 'n7'}) RETURN a"),
        1
    );
    assert_eq!(
        count(&mut db, "MATCH (a:Account {id: 7, name: 'other'}) RETURN a"),
        0,
        "a non-key filter still applies"
    );
    assert_eq!(count(&mut db, "MATCH (a:Account {id: 51}) RETURN a"), 0);
    assert_eq!(
        count(&mut db, "MATCH (a:Account) WHERE a.id = 7 RETURN a"),
        1
    );
    assert_eq!(
        count(
            &mut db,
            "MATCH (a:Account) WHERE a.id = 7 AND a.name = 'other' RETURN a"
        ),
        0
    );
    assert_eq!(
        count(
            &mut db,
            "MATCH (a:Account) WHERE a.id = 7 OR a.id = 8 RETURN a"
        ),
        2,
        "a disjunction is not a key lookup"
    );
    let rows = db
        .execute_sql("SELECT name FROM Account WHERE id = 42")
        .expect("sql");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("name"), Some(&Value::String("n42".into())));

    for q in [
        "MATCH (a:Account) WHERE a.id = 7.0 RETURN a",
        "MATCH (a:Account {id: 7.0}) RETURN a",
    ] {
        let by_type = count(&mut db, q);
        let scanned = count(
            &mut db,
            &q.replace("MATCH (a:Account", "MATCH (a:Account) WITH a MATCH (a"),
        );
        assert_eq!(by_type, scanned, "{q}");
    }
}

/// A row's key is its identity and does not change, the way `_id` does not:
/// setting or removing a key column is refused.
#[test]
fn a_key_column_cannot_be_changed() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).expect("open db");
    db.execute_cypher("CREATE TABLE Account (id BIGINT PRIMARY KEY, name STRING)")
        .expect("create");
    db.execute_cypher("CREATE (:Account {id: 1, name: 'a'})")
        .expect("insert");
    assert!(
        db.execute_cypher("MATCH (a:Account {id: 1}) SET a.id = 2")
            .is_err(),
        "a key column cannot be set"
    );
    assert!(
        db.execute_sql("UPDATE Account SET id = 2 WHERE id = 1")
            .is_err(),
        "nor through SQL"
    );
    assert!(
        db.execute_cypher("MATCH (a:Account {id: 1}) REMOVE a.id")
            .is_err(),
        "nor removed"
    );
    let rows = db
        .execute_cypher("MATCH (a:Account {id: 1}) RETURN a.name AS name")
        .expect("match");
    assert_eq!(rows.len(), 1, "the row keeps its key");
}

/// Key uniqueness holds across a reopen.
#[test]
fn an_existing_key_is_refused_after_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut db = Database::open(dir.path()).expect("open db");
        db.execute_cypher("CREATE TABLE Account (id BIGINT PRIMARY KEY)")
            .expect("create");
        db.execute_cypher("CREATE (:Account {id: 1})")
            .expect("insert");
    }
    let mut db = Database::open(dir.path()).expect("reopen db");
    let err = db
        .execute_cypher("CREATE (:Account {id: 1})")
        .expect_err("the key still exists");
    assert!(err.to_string().contains("already exists"), "{err}");
}

/// A columnar table enforces its key the same way.
#[test]
fn a_columnar_table_refuses_an_existing_key() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).expect("open db");
    db.execute_cypher("CREATE TABLE Trade (id BIGINT PRIMARY KEY, sym STRING) STORAGE COLUMNAR")
        .expect("create");
    db.execute_cypher("CREATE (:Trade {id: 1, sym: 'AAPL'})")
        .expect("insert");
    let err = db
        .execute_cypher("CREATE (:Trade {id: 1, sym: 'MSFT'})")
        .expect_err("the key exists");
    assert!(err.to_string().contains("already exists"), "{err}");
    let rows = db
        .execute_cypher("MATCH (t:Trade {id: 1}) RETURN t.sym AS sym")
        .expect("match");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("sym"), Some(&Value::String("AAPL".into())));
}

#[test]
fn create_table_rejects_unknown_column_type() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).expect("open db");

    // QUATERNION is not a supported column type.
    assert!(
        db.execute_cypher("CREATE TABLE Bad (id BIGINT PRIMARY KEY, q QUATERNION)")
            .is_err()
    );
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

#[test]
fn sql_update_modifies_matching_row() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).expect("open db");
    db.execute_cypher("CREATE TABLE Account (id BIGINT PRIMARY KEY, name STRING)")
        .expect("create");
    db.execute_sql("INSERT INTO Account (id, name) VALUES (1, 'Alice')")
        .expect("insert");

    db.execute_sql("UPDATE Account SET name = 'Alicia' WHERE id = 1")
        .expect("update");

    let rows = db
        .execute_sql("SELECT name FROM Account WHERE id = 1")
        .expect("select");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("name"), Some(&Value::String("Alicia".into())));
}

#[test]
fn sql_delete_removes_matching_row() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).expect("open db");
    db.execute_cypher("CREATE TABLE Account (id BIGINT PRIMARY KEY, name STRING)")
        .expect("create");
    db.execute_sql("INSERT INTO Account (id, name) VALUES (1, 'Alice')")
        .expect("insert 1");
    db.execute_sql("INSERT INTO Account (id, name) VALUES (2, 'Bob')")
        .expect("insert 2");

    db.execute_sql("DELETE FROM Account WHERE id = 1")
        .expect("delete");

    let gone = db
        .execute_sql("SELECT name FROM Account WHERE id = 1")
        .expect("select gone");
    assert_eq!(gone.len(), 0, "deleted row must not be found");
    let remaining = db
        .execute_sql("SELECT id FROM Account")
        .expect("select all");
    assert_eq!(remaining.len(), 1, "the other row remains");
}
