//! `ON DUPLICATE RENAME`: a unique index or uniqueness constraint whose
//! build may end a stored duplicate by renaming one holder's string value.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_core::graph::types::Value;
use coordinode_embed::Database;

fn open_db() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open db");
    (db, dir)
}

/// Every email `:User` nodes hold, sorted.
fn emails(db: &mut Database) -> Vec<String> {
    let mut out: Vec<String> = db
        .execute_cypher("MATCH (u:User) RETURN u.email AS e")
        .expect("match")
        .iter()
        .map(|r| match r.get("e") {
            Some(Value::String(s)) => s.clone(),
            other => panic!("email: {other:?}"),
        })
        .collect();
    out.sort();
    out
}

/// The operation a CREATE statement's row names.
fn operation(rows: &[coordinode_query::executor::row::Row]) -> i64 {
    match rows[0].get("operation") {
        Some(Value::Int(op)) => *op,
        other => panic!("operation: {other:?}"),
    }
}

/// CREATE UNIQUE INDEX ... ON DUPLICATE RENAME over two holders of one
/// value: the index is ready, every node keeps a distinct value (one of them
/// the old value with a suffix), and the repair is listed through its
/// operation. The clause is the build's: a later duplicate write is refused
/// as always.
#[test]
fn a_unique_index_renames_a_stored_duplicate_and_enforces_after() {
    let (mut db, _dir) = open_db();
    db.execute_cypher(
        "CREATE (:User {email: 'same@x'}), (:User {email: 'same@x'}), (:User {email: 'b@x'})",
    )
    .expect("seed");

    let rows = db
        .execute_cypher("CREATE UNIQUE INDEX user_email ON :User(email) ON DUPLICATE RENAME email")
        .expect("the build repairs the duplicate");
    assert_eq!(rows[0].get("state"), Some(&Value::String("READY".into())));
    let op = operation(&rows);

    let all = emails(&mut db);
    assert_eq!(all.len(), 3);
    assert!(all.contains(&"same@x".to_string()));
    assert!(all.contains(&"b@x".to_string()));
    let renamed = all
        .iter()
        .find(|e| e.starts_with("same@x_"))
        .expect("one holder renamed")
        .clone();

    let repairs = db
        .execute_cypher(&format!(
            "CALL db.indexBuildRepairs({op}) YIELD property, oldValue, newValue \
             RETURN property, oldValue, newValue"
        ))
        .expect("repairs");
    assert_eq!(repairs.len(), 1);
    assert_eq!(
        repairs[0].get("oldValue"),
        Some(&Value::String("same@x".into()))
    );
    assert_eq!(repairs[0].get("newValue"), Some(&Value::String(renamed)));
    let build = db
        .execute_cypher(&format!(
            "CALL db.indexBuild({op}) YIELD repaired, renameProperty RETURN repaired, renameProperty"
        ))
        .expect("build");
    assert_eq!(build[0].get("repaired"), Some(&Value::Int(1)));
    assert_eq!(
        build[0].get("renameProperty"),
        Some(&Value::String("email".into()))
    );

    let refused = db
        .execute_cypher("CREATE (:User {email: 'b@x'})")
        .expect_err("a later duplicate is refused, not renamed");
    assert!(
        refused.to_string().contains("unique constraint violated"),
        "{refused}"
    );
}

/// The constraint form repairs every extra holder of a value: of three
/// nodes sharing one, two are renamed, each to a value of its own, and the
/// constraint ends active.
#[test]
fn a_uniqueness_constraint_renames_every_extra_holder() {
    let (mut db, _dir) = open_db();
    db.execute_cypher(
        "CREATE (:User {email: 'same@x'}), (:User {email: 'same@x'}), (:User {email: 'same@x'})",
    )
    .expect("seed");

    let rows = db
        .execute_cypher(
            "CREATE CONSTRAINT user_email FOR (u:User) REQUIRE u.email IS UNIQUE \
             ON DUPLICATE RENAME u.email",
        )
        .expect("the build repairs both duplicates");
    assert_eq!(rows[0].get("state"), Some(&Value::String("ACTIVE".into())));

    let mut all = emails(&mut db);
    all.dedup();
    assert_eq!(all.len(), 3, "every value distinct: {all:?}");
    assert_eq!(all.iter().filter(|e| *e == "same@x").count(), 1);
    let listed = db.constraints().expect("constraints");
    let op = listed[0].operation.expect("its build");
    assert_eq!(db.index_build_repairs(op).expect("repairs").len(), 2);
}

/// What the clause cannot do is refused before anything is published: a
/// non-unique index, a property the uniqueness does not cover, a constraint
/// kind with no index, a temporal label, and a property declared other than
/// a string.
#[test]
fn a_rename_the_build_cannot_make_is_refused_up_front() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (:User {email: 'a@x', age: 1})")
        .expect("seed");
    for (statement, why) in [
        (
            "CREATE INDEX user_email ON :User(email) ON DUPLICATE RENAME email",
            "not unique",
        ),
        (
            "CREATE CONSTRAINT c FOR (u:User) REQUIRE u.email IS UNIQUE ON DUPLICATE RENAME u.age",
            "must be one the uniqueness covers",
        ),
        (
            "CREATE CONSTRAINT c FOR (u:User) REQUIRE u.email IS NOT NULL \
             ON DUPLICATE RENAME u.email",
            "UNIQUE or NODE KEY",
        ),
    ] {
        let error = db.execute_cypher(statement).expect_err(statement);
        assert!(error.to_string().contains(why), "{statement}: {error}");
    }

    db.execute_cypher(
        "CREATE NODE TYPE Emp TEMPORAL WITH (name: STRING, valid_from: INT, valid_to: INT)",
    )
    .expect("temporal label");
    let temporal = db
        .execute_cypher(
            "CREATE CONSTRAINT emp_name FOR (e:Emp) REQUIRE e.name IS UNIQUE \
             ON DUPLICATE RENAME e.name",
        )
        .expect_err("temporal");
    assert!(temporal.to_string().contains("temporal"), "{temporal}");

    db.execute_cypher("CREATE NODE TYPE Acct WITH (code: INT)")
        .expect("typed label");
    let typed = db
        .execute_cypher(
            "CREATE CONSTRAINT acct_code FOR (a:Acct) REQUIRE a.code IS UNIQUE \
             ON DUPLICATE RENAME a.code",
        )
        .expect_err("not a string");
    assert!(typed.to_string().contains("declared"), "{typed}");

    assert!(db.constraints().expect("constraints").is_empty());
    assert!(
        db.index_build_status().expect("builds").is_empty(),
        "no build was admitted"
    );
}
