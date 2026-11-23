//! A store released 0.10.1 wrote is brought to this build and opens as an
//! application opens it, with every record it held.
//!
//! The fixture is the directory the 0.10.1 embedded quickstart left after
//! one run: an account with a unique email and its wallet, written in one
//! transaction, and a run counter node.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::collections::HashMap;
use std::path::Path;

use coordinode_core::graph::types::Value;
use coordinode_embed::{Database, DatabaseError};

/// Copy the directory `from` to `to`, recursively.
fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).expect("dir");
    for entry in std::fs::read_dir(from).expect("list") {
        let entry = entry.expect("entry");
        let target = to.join(entry.file_name());
        if entry.file_type().expect("type").is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), &target).expect("copy");
        }
    }
}

fn int(db: &mut Database, query: &str) -> i64 {
    let rows = db.execute_cypher(query).expect("query");
    match rows.first().and_then(|row| row.values().next()) {
        Some(Value::Int(n)) => *n,
        other => panic!("{query}: {other:?}"),
    }
}

/// Migrated, the 0.10.1 store opens with its run, its account and wallet,
/// and its unique constraint holding; nodes created now take identifiers
/// past the ones it issued, so none of its nodes is written over; and a
/// second run finds nothing to rewrite.
#[test]
fn a_store_0_10_1_wrote_opens_whole() {
    let dir = tempfile::tempdir().expect("tempdir");
    let data = dir.path().join("data");
    copy_dir(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/embedded-0.10.1"),
        &data,
    );
    assert!(
        Database::open(&data).is_err(),
        "this build refuses the store as 0.10.1 left it"
    );

    let found = coordinode_migrate::apply(&data, "run1").expect("migrate");
    assert!(
        found.iter().all(|(_, f)| !f.is_empty()),
        "every step applies"
    );

    let mut db = Database::open(&data).expect("open the migrated store");
    assert_eq!(
        int(&mut db, "MATCH (r:QuickstartRun) RETURN count(r) AS n"),
        1
    );
    assert_eq!(
        int(
            &mut db,
            "MATCH (a:Account {email: 'alice-1@example.com'})-[:OWNS]->(w:Wallet) \
             RETURN w.balance AS b"
        ),
        100
    );
    let mut params = HashMap::new();
    params.insert(
        "email".to_string(),
        Value::String("alice-1@example.com".into()),
    );
    assert!(matches!(
        db.execute_cypher_with_params("CREATE (:Account {email: $email})", params),
        Err(DatabaseError::UniqueViolation { .. })
    ));
    for i in 0..50 {
        db.execute_cypher(&format!("CREATE (:Probe {{i: {i}}})"))
            .expect("create");
    }
    assert_eq!(int(&mut db, "MATCH (p:Probe) RETURN count(p) AS n"), 50);
    assert_eq!(
        int(&mut db, "MATCH (r:QuickstartRun) RETURN count(r) AS n"),
        1
    );
    assert_eq!(
        int(
            &mut db,
            "MATCH (a:Account)-[:OWNS]->(w:Wallet) RETURN count(w) AS n"
        ),
        1,
        "no node 0.10.1 wrote was written over"
    );
    drop(db);

    let again = coordinode_migrate::survey(&data).expect("survey");
    assert!(again.iter().all(|(_, f)| f.is_empty()));
}
