//! A store released 0.10.1 wrote is brought to this build and opens as an
//! application opens it, with every record it held.
//!
//! The fixtures are directories the released 0.10.1 engine wrote:
//! `embedded-0.10.1`, the embedded quickstart after one run (an account with
//! a unique email and its wallet, written in one transaction, and a run
//! counter node); `embedded-0.10.1-receipts`, a unique account, two probe
//! nodes committed one after the other and a refused write, the journal
//! holding every write; `embedded-0.10.1-settled`, the same compacted with
//! no retention window, every version settled, the journal still whole. The
//! last two carry a file `receipts`: each probe's key, node id and the
//! commit its write returned.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use coordinode_core::graph::node::NodeId;
use coordinode_core::graph::types::Value;
use coordinode_embed::db::{CommitOptions, ExpectedVersion};
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

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn int(db: &mut Database, query: &str) -> i64 {
    let rows = db.execute_cypher(query).expect("query");
    match rows.first().and_then(|row| row.values().next()) {
        Some(Value::Int(n)) => *n,
        other => panic!("{query}: {other:?}"),
    }
}

/// A fixture with receipts migrated into a fresh directory: the data
/// directory and each probe's node and commit, by key.
fn migrated(name: &str, dir: &Path) -> (PathBuf, HashMap<String, (NodeId, u64)>) {
    let data = dir.join("data");
    copy_dir(&fixture(name).join("data"), &data);
    let receipts = std::fs::read_to_string(fixture(name).join("receipts")).expect("receipts");
    let receipts = receipts
        .lines()
        .map(|line| {
            let mut fields = line.split(' ');
            let key = fields.next().expect("key").to_owned();
            let id: u64 = fields.next().expect("id").parse().expect("id");
            let commit: u64 = fields.next().expect("commit").parse().expect("commit");
            (key, (NodeId::from_raw(id), commit))
        })
        .collect();
    coordinode_migrate::apply(&data, "run1").expect("migrate");
    (data, receipts)
}

/// The unique email of the fixtures holds: a held value is refused, a new
/// one taken.
fn unique_holds(db: &mut Database, held: &str, new: &str) {
    let mut params = HashMap::new();
    params.insert("email".to_string(), Value::String(held.into()));
    assert!(matches!(
        db.execute_cypher_with_params("CREATE (:Account {email: $email})", params),
        Err(DatabaseError::UniqueViolation { .. })
    ));
    let mut params = HashMap::new();
    params.insert("email".to_string(), Value::String(new.into()));
    db.execute_cypher_with_params("CREATE (:Account {email: $email})", params)
        .expect("a new value");
}

/// Migrated, the 0.10.1 store opens with its run, its account and wallet,
/// and its unique constraint holding; nodes created now take identifiers
/// past the ones it issued, so none of its nodes is written over; and a
/// second run finds nothing to rewrite.
#[test]
fn a_store_0_10_1_wrote_opens_whole() {
    let dir = tempfile::tempdir().expect("tempdir");
    let data = dir.path().join("data");
    copy_dir(&fixture("embedded-0.10.1"), &data);
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

/// Every version the journal proves keeps exactly the commit its write
/// returned: two probes written one after the other read back their own
/// receipts. A conditional write on one lands; the new write has its own
/// version, which stands after a reopen, where the old one is stale.
#[test]
fn a_version_the_journal_proves_keeps_its_receipt() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (data, receipts) = migrated("embedded-0.10.1-receipts", dir.path());
    let (a, c1) = receipts["a"];
    let (b, c2) = receipts["b"];
    assert!(c1 < c2);

    let db = Database::open(&data).expect("open the migrated store");
    assert_eq!(db.node_version(a).expect("version"), Some(c1));
    assert_eq!(db.node_version(b).expect("version"), Some(c2));
    let tx = db.begin_transaction().expect("begin");
    db.execute_in_transaction(tx, "MATCH (p:Probe {k: 'a'}) SET p.v = 10", None)
        .expect("statement");
    db.expect_node_version(tx, a, Some(c1))
        .expect("state the condition");
    db.commit_transaction(tx)
        .expect("the kept version is current");
    let written = db.node_version(a).expect("version").expect("there");
    assert!(written > c2, "the new write has its own version");
    assert_eq!(db.node_version(b).expect("version"), Some(c2));
    drop(db);

    let mut db = Database::open(&data).expect("reopen");
    assert_eq!(db.node_version(a).expect("version"), Some(written));
    let tx = db.begin_transaction().expect("begin");
    db.execute_in_transaction(tx, "MATCH (p:Probe {k: 'a'}) SET p.v = 11", None)
        .expect("statement");
    db.expect_node_version(tx, a, Some(c1))
        .expect("state the condition");
    assert!(matches!(
        db.commit_transaction(tx),
        Err(DatabaseError::RevisionMismatch { current, .. }) if current == Some(written)
    ));
    unique_holds(&mut db, "alice@example.com", "bob@example.com");
}

/// A store whose every version a compaction settled, its journal still
/// whole, proves no commit: each record reads as it was, a conditional write
/// on one is refused with nothing of its transaction applied, a write
/// replacing it gives it a version of its own, and the unique index serves
/// writes, before and after a reopen.
#[test]
fn a_settled_version_is_unknown() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (data, receipts) = migrated("embedded-0.10.1-settled", dir.path());
    let (a, _) = receipts["a"];
    let (b, _) = receipts["b"];

    let mut db = Database::open(&data).expect("open the migrated store");
    assert_eq!(
        int(&mut db, "MATCH (p:Probe {k: 'b'}) RETURN p.v AS v"),
        2,
        "the record reads as it was"
    );
    for node in [a, b] {
        assert!(matches!(
            db.node_version(node),
            Err(DatabaseError::RevisionUnavailable { .. })
        ));
    }
    // No version a caller names gets past the lost one: zero and absence
    // included, stated on the transaction or carried by the commit.
    for (expected, on_commit) in [
        (Some(1), false),
        (Some(0), false),
        (None, false),
        (Some(0), true),
    ] {
        let tx = db.begin_transaction().expect("begin");
        db.execute_in_transaction(tx, "MATCH (p:Probe {k: 'a'}) SET p.v = 10", None)
            .expect("statement");
        db.execute_in_transaction(tx, "CREATE (:SideEffect)", None)
            .expect("statement");
        let condition = [ExpectedVersion {
            node: a,
            version: expected,
        }];
        let outcome = if on_commit {
            db.commit_transaction_with(
                tx,
                &CommitOptions {
                    expect: &condition,
                    ..CommitOptions::default()
                },
            )
        } else {
            db.expect_node_version(tx, a, expected)
                .expect("state the condition");
            db.commit_transaction(tx)
        };
        assert!(
            matches!(outcome, Err(DatabaseError::RevisionUnavailable { .. })),
            "{expected:?} on the commit: {on_commit}"
        );
        assert_eq!(int(&mut db, "MATCH (s:SideEffect) RETURN count(s) AS n"), 0);
        assert_eq!(int(&mut db, "MATCH (p:Probe {k: 'a'}) RETURN p.v AS v"), 1);
    }

    db.execute_cypher("MATCH (p:Probe {k: 'a'}) SET p.v = 12")
        .expect("a write replaces the record");
    let written = db.node_version(a).expect("version").expect("there");
    unique_holds(&mut db, "alice@example.com", "bob@example.com");
    drop(db);

    let mut db = Database::open(&data).expect("reopen");
    assert_eq!(db.node_version(a).expect("version"), Some(written));
    assert!(matches!(
        db.node_version(b),
        Err(DatabaseError::RevisionUnavailable { .. })
    ));
    unique_holds(&mut db, "bob@example.com", "carol@example.com");
}
