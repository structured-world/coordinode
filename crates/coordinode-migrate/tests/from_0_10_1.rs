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

/// The commit of every version the journal still holds is kept: the node
/// 0.10.1 wrote keeps the version of its write, and a conditional write on
/// it lands.
#[test]
fn a_version_the_journal_proves_is_kept() {
    let dir = tempfile::tempdir().expect("tempdir");
    let data = dir.path().join("data");
    copy_dir(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/embedded-0.10.1"),
        &data,
    );
    coordinode_migrate::apply(&data, "run1").expect("migrate");
    let mut db = Database::open(&data).expect("open the migrated store");
    let id = int(&mut db, "MATCH (r:QuickstartRun) RETURN id(r) AS n");
    let node = coordinode_core::graph::node::NodeId::from_raw(id as u64);
    let version = db
        .node_version(node)
        .expect("the journal proves the node's write")
        .expect("the node is there");
    assert!(version > 0);
    let tx = db.begin_transaction().expect("begin");
    db.execute_in_transaction(tx, "MATCH (r:QuickstartRun) SET r.again = true", None)
        .expect("statement");
    db.expect_node_version(tx, node, Some(version))
        .expect("state the condition");
    db.commit_transaction(tx)
        .expect("the kept version is current");
    let written = db.node_version(node).expect("version").expect("there");
    assert!(written > version, "the new write has its own version");
    drop(db);

    // Reopened: the version stands, the old one is stale, the index serves.
    let mut db = Database::open(&data).expect("reopen");
    assert_eq!(db.node_version(node).expect("version"), Some(written));
    let tx = db.begin_transaction().expect("begin");
    db.execute_in_transaction(tx, "MATCH (r:QuickstartRun) SET r.stale = true", None)
        .expect("statement");
    db.expect_node_version(tx, node, Some(version))
        .expect("state the condition");
    assert!(matches!(
        db.commit_transaction(tx),
        Err(DatabaseError::RevisionMismatch { current, .. }) if current == Some(written)
    ));
    db.execute_cypher("CREATE (:Account {email: 'bob@example.com'})")
        .expect("a new value");
    assert_eq!(int(&mut db, "MATCH (a:Account) RETURN count(a) AS n"), 2);
}

/// With the journal past what wrote them, no version 0.10.1 wrote is known:
/// the store still opens and reads every record as it was, a conditional
/// write on a record of unknown version is refused without writing, and a
/// write replacing it gives it a version of its own.
#[test]
fn a_version_nothing_proves_is_unknown() {
    let dir = tempfile::tempdir().expect("tempdir");
    let data = dir.path().join("data");
    copy_dir(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/embedded-0.10.1"),
        &data,
    );
    std::fs::remove_dir_all(data.join("oplog")).expect("lose the journal");
    coordinode_migrate::apply(&data, "run1").expect("migrate");
    let mut db = Database::open(&data).expect("open the migrated store");
    assert_eq!(
        int(
            &mut db,
            "MATCH (a:Account {email: 'alice-1@example.com'})-[:OWNS]->(w:Wallet) \
             RETURN w.balance AS b"
        ),
        100
    );
    let id = int(&mut db, "MATCH (r:QuickstartRun) RETURN id(r) AS n");
    let node = coordinode_core::graph::node::NodeId::from_raw(id as u64);
    assert!(matches!(
        db.node_version(node),
        Err(DatabaseError::RevisionUnavailable { .. })
    ));
    let tx = db.begin_transaction().expect("begin");
    db.execute_in_transaction(tx, "MATCH (r:QuickstartRun) SET r.refused = true", None)
        .expect("statement");
    db.execute_in_transaction(tx, "CREATE (:SideEffect)", None)
        .expect("statement");
    db.expect_node_version(tx, node, Some(1))
        .expect("state the condition");
    assert!(matches!(
        db.commit_transaction(tx),
        Err(DatabaseError::RevisionUnavailable { .. })
    ));
    assert_eq!(int(&mut db, "MATCH (s:SideEffect) RETURN count(s) AS n"), 0);

    db.execute_cypher("MATCH (r:QuickstartRun) SET r.replaced = true")
        .expect("a write replaces the record");
    assert!(db.node_version(node).expect("version").is_some());

    // The unique index whose definition the open wrote again serves
    // writes: a new value is taken, a held one refused.
    db.execute_cypher("CREATE (:Account {email: 'bob@example.com'})")
        .expect("a new value");
    let mut params = HashMap::new();
    params.insert(
        "email".to_string(),
        Value::String("alice-1@example.com".into()),
    );
    assert!(matches!(
        db.execute_cypher_with_params("CREATE (:Account {email: $email})", params),
        Err(DatabaseError::UniqueViolation { .. })
    ));
    assert_eq!(int(&mut db, "MATCH (a:Account) RETURN count(a) AS n"), 2);
}
