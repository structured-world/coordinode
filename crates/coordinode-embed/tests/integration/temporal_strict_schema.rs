//! Integration tests: a STRICT temporal label holds its schema on every
//! version a write creates, and a definition published over stored versions
//! is checked against the properties users wrote, not the metadata the
//! engine keeps beside them.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_core::graph::types::Value;
use coordinode_core::schema::definition::LabelSchema;
use coordinode_embed::Database;

use super::helpers::temporal_versions;

fn now_us() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock after epoch")
        .as_micros() as i64
}

/// A database over `dir` holding STRICT temporal label `Emp`.
fn open_strict(dir: &std::path::Path) -> Database {
    let mut db = Database::open(dir).expect("open");
    db.execute_cypher(
        "CREATE NODE TYPE Emp TEMPORAL WITH (name: STRING, valid_from: INT, valid_to: INT)",
    )
    .expect("temporal label");
    db.execute_cypher("ALTER LABEL Emp SET SCHEMA STRICT")
        .expect("strict");
    db
}

/// The definition of `Emp` as published.
fn emp_schema(db: &Database) -> LabelSchema {
    db.label_schemas()
        .expect("label schemas")
        .into_iter()
        .find(|s| s.name == "Emp")
        .expect("Emp is defined")
}

/// Stored versions of `Emp`: two of one node (a create and a revision) and a
/// deleted node's tombstone, all carrying the engine's own metadata.
fn seed(db: &mut Database) {
    let t = now_us();
    db.execute_cypher(&format!(
        "CREATE (:Emp {{name: 'a', valid_from: {t}}}), (:Emp {{name: 'gone', valid_from: {t}}})"
    ))
    .expect("create");
    db.execute_cypher("MATCH (n:Emp {name: 'a'}) SET n.name = 'b'")
        .expect("revise");
    db.execute_cypher("MATCH (n:Emp {name: 'gone'}) DELETE n")
        .expect("delete");
}

/// Republishing the definition a STRICT temporal label already has, over
/// versions the engine stamped with its own metadata, is accepted, before
/// and after a reopen: that metadata is not a property the definition could
/// declare.
#[test]
fn an_identical_definition_republishes_over_stored_versions() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = open_strict(dir.path());
    seed(&mut db);
    let schema = emp_schema(&db);
    db.create_label_schema(schema.clone())
        .expect("the same definition republishes");
    drop(db);

    let db = Database::open(dir.path()).expect("reopen");
    db.create_label_schema(emp_schema(&db))
        .expect("the same definition republishes after a reopen");
}

/// A definition that a stored version genuinely breaks is still refused:
/// dropping a property the versions carry leaves them with an undeclared one.
#[test]
fn a_definition_a_stored_version_breaks_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = open_strict(dir.path());
    seed(&mut db);
    let mut schema = emp_schema(&db);
    schema.properties.remove("name");
    let err = db
        .create_label_schema(schema)
        .expect_err("a stored version carries `name`");
    assert!(err.to_string().contains("name"), "{err}");
}

/// The engine's metadata stays out of users' reach: a SET of it is refused.
#[test]
fn a_user_write_of_the_ingestion_timestamp_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = open_strict(dir.path());
    seed(&mut db);
    let err = db
        .execute_cypher("MATCH (n:Emp {name: 'b'}) SET n.__ingestion_ts__ = 1")
        .expect_err("reserved");
    assert!(err.to_string().contains("__ingestion_ts__"), "{err}");
}

/// Inside an interactive transaction, a valid SET followed by a refused one
/// leaves the transaction unable to commit, and neither the commit attempt
/// nor a rollback leaves a new or partly closed version, also after a reopen.
#[test]
fn a_refused_set_in_a_transaction_leaves_no_version() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = open_strict(dir.path());
    let t = now_us();
    db.execute_cypher(&format!("CREATE (:Emp {{name: 'a', valid_from: {t}}})"))
        .expect("create");
    let before = temporal_versions(&db, "Emp");

    for commit in [true, false] {
        let tx = db.begin_transaction();
        db.execute_in_transaction(tx, "MATCH (n:Emp {name: 'a'}) SET n.name = 'b'", None)
            .expect("a declared property");
        db.execute_in_transaction(tx, "MATCH (n:Emp) SET n.unknown = 1", None)
            .expect_err("`unknown` is not declared on a STRICT label");
        // A statement error aborts the transaction: its state is dropped and
        // both a commit and a rollback find no transaction left.
        let ended = if commit {
            db.commit_transaction(tx).map(drop)
        } else {
            db.rollback_transaction(tx)
        };
        assert!(
            matches!(ended, Err(coordinode_embed::DatabaseError::UnknownTransaction(id)) if id == tx),
            "commit={commit}: {ended:?}"
        );
        assert_eq!(temporal_versions(&db, "Emp"), before, "commit={commit}");
    }
    drop(db);

    let db = Database::open(dir.path()).expect("reopen");
    assert_eq!(temporal_versions(&db, "Emp"), before, "after a reopen");
}

/// The engine's temporal fields are refused as user input on a temporal
/// label of every schema mode, in a CREATE and in a programmatic definition.
#[test]
fn the_engine_temporal_fields_are_refused_as_user_input() {
    for mode in ["FLEXIBLE", "VALIDATED", "STRICT"] {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut db = Database::open(dir.path()).expect("open");
        db.execute_cypher(
            "CREATE NODE TYPE Emp TEMPORAL WITH (name: STRING, valid_from: INT, valid_to: INT)",
        )
        .expect("temporal label");
        db.execute_cypher(&format!("ALTER LABEL Emp SET SCHEMA {mode}"))
            .expect("mode");
        let t = now_us();
        for field in ["__ingestion_ts__", "__deleted__"] {
            let err = db
                .execute_cypher(&format!(
                    "CREATE (:Emp {{name: 'a', valid_from: {t}, {field}: true}})"
                ))
                .expect_err(&format!("{mode}: {field} in CREATE"));
            assert!(err.to_string().contains(field), "{mode}: {err}");
        }
        assert!(temporal_versions(&db, "Emp").is_empty(), "{mode}");

        for field in ["__ingestion_ts__", "__deleted__"] {
            let mut schema = emp_schema(&db);
            schema.add_property(coordinode_core::schema::definition::PropertyDef::new(
                field,
                coordinode_core::schema::definition::PropertyType::Bool,
            ));
            let err = db
                .create_label_schema(schema)
                .expect_err(&format!("{mode}: {field} declared"));
            assert!(err.to_string().contains(field), "{mode}: {err}");
        }
    }
}

/// A map SET carrying an undeclared key or the engine's metadata is refused
/// on a STRICT temporal label, and writes no version.
#[test]
fn a_map_set_with_an_undeclared_key_writes_no_version() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = open_strict(dir.path());
    let t = now_us();
    db.execute_cypher(&format!("CREATE (:Emp {{name: 'a', valid_from: {t}}})"))
        .expect("create");
    let before = temporal_versions(&db, "Emp");
    for statement in [
        "MATCH (n:Emp {name: 'a'}) SET n += {name: 'b', unknown: 1}",
        "MATCH (n:Emp {name: 'a'}) SET n = {name: 'b', valid_from: 1, unknown: 1}",
        "MATCH (n:Emp {name: 'a'}) SET n += {__deleted__: true}",
    ] {
        db.execute_cypher(statement)
            .expect_err(&format!("refused: {statement}"));
        assert_eq!(temporal_versions(&db, "Emp"), before, "{statement}");
    }
}

/// A SET of a property the STRICT label does not declare is refused at the
/// statement, and the node keeps exactly the versions it had, also after a
/// reopen; a SET of a declared property still opens the next version.
#[test]
fn an_undeclared_set_on_a_strict_temporal_label_writes_no_version() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = open_strict(dir.path());
    let t = now_us();
    db.execute_cypher(&format!("CREATE (:Emp {{name: 'a', valid_from: {t}}})"))
        .expect("create");
    let before = temporal_versions(&db, "Emp");
    assert_eq!(before.len(), 1);

    let err = db
        .execute_cypher("MATCH (n:Emp {name: 'a'}) SET n.unknown = 1")
        .expect_err("`unknown` is not declared on a STRICT label");
    assert!(err.to_string().contains("unknown"), "{err}");
    assert_eq!(temporal_versions(&db, "Emp"), before, "no version written");
    drop(db);

    let mut db = Database::open(dir.path()).expect("reopen");
    assert_eq!(
        temporal_versions(&db, "Emp"),
        before,
        "no version after a reopen either"
    );
    db.execute_cypher("MATCH (n:Emp {name: 'a'}) SET n.name = 'b'")
        .expect("a declared property");
    let after = temporal_versions(&db, "Emp");
    assert_eq!(after.len(), 2, "the next version");
    assert!(
        after.iter().all(|v| !v.contains_key("unknown")),
        "{after:?}"
    );
    assert!(
        after
            .iter()
            .any(|v| v.get("name") == Some(&Value::String("b".into()))),
        "{after:?}"
    );
}
