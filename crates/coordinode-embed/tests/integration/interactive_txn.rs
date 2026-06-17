//! Integration tests: interactive multi-statement transactions (ADR-042).
//!
//! Exercises the `Database` interactive transaction API end to end through
//! the full pipeline: `begin_transaction` → N `execute_in_transaction` →
//! `commit_transaction` / `rollback_transaction`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_embed::Database;
use std::time::Duration;

fn open_db() -> Database {
    Database::open_in_memory().expect("open db")
}

#[test]
fn commit_makes_multi_statement_writes_visible() {
    let mut db = open_db();
    let tx = db.begin_transaction();
    db.execute_in_transaction(tx, "CREATE (n:User {name: 'Alice'})", None)
        .expect("stmt 1");
    db.execute_in_transaction(tx, "CREATE (n:User {name: 'Bob'})", None)
        .expect("stmt 2");

    // Before commit, a separate auto-commit read does NOT see the buffered,
    // uncommitted writes (they live in the transaction, not the engine).
    let before = db
        .execute_cypher("MATCH (n:User) RETURN n")
        .expect("read before commit");
    assert_eq!(
        before.len(),
        0,
        "uncommitted writes are invisible elsewhere"
    );

    db.commit_transaction(tx).expect("commit");

    // After commit, both statements' writes are visible atomically.
    let after = db
        .execute_cypher("MATCH (n:User) RETURN n")
        .expect("read after commit");
    assert_eq!(after.len(), 2, "both committed nodes are visible");
}

#[test]
fn rollback_discards_writes_and_consumes_handle() {
    let mut db = open_db();
    let tx = db.begin_transaction();
    db.execute_in_transaction(tx, "CREATE (n:User {name: 'Carol'})", None)
        .expect("stmt");
    db.rollback_transaction(tx).expect("rollback");

    let rows = db
        .execute_cypher("MATCH (n:User) RETURN n")
        .expect("read after rollback");
    assert_eq!(rows.len(), 0, "rolled-back writes are discarded");

    // The handle is consumed by rollback — commit on it now fails.
    assert!(
        db.commit_transaction(tx).is_err(),
        "rolled-back transaction id is no longer known"
    );
}

#[test]
fn read_your_own_writes_within_transaction() {
    let db = open_db();
    let tx = db.begin_transaction();
    db.execute_in_transaction(tx, "CREATE (n:User {name: 'Dave'})", None)
        .expect("create");
    // A read in the SAME transaction sees its own uncommitted write.
    let rows = db
        .execute_in_transaction(tx, "MATCH (n:User) RETURN n.name", None)
        .expect("read own write");
    assert_eq!(rows.len(), 1, "read-your-own-writes within the transaction");
    db.commit_transaction(tx).expect("commit");
}

#[test]
fn repeatable_read_across_statements() {
    let mut db = open_db();
    db.execute_cypher("CREATE (n:User {name: 'Seed'})")
        .expect("seed");

    let tx = db.begin_transaction();
    let first = db
        .execute_in_transaction(tx, "MATCH (n:User) RETURN n", None)
        .expect("read 1");
    assert_eq!(first.len(), 1);

    // A concurrent auto-commit insert commits AFTER the transaction's pinned
    // snapshot timestamp.
    db.execute_cypher("CREATE (n:User {name: 'Later'})")
        .expect("concurrent insert");

    // The transaction re-reads at its pinned snapshot → the later insert is
    // invisible (repeatable read across statements).
    let second = db
        .execute_in_transaction(tx, "MATCH (n:User) RETURN n", None)
        .expect("read 2");
    assert_eq!(
        second.len(),
        1,
        "pinned snapshot hides writes committed after begin",
    );
    db.rollback_transaction(tx).expect("rollback");

    // After rollback, a fresh auto-commit read sees both committed nodes.
    let now = db
        .execute_cypher("MATCH (n:User) RETURN n")
        .expect("read now");
    assert_eq!(now.len(), 2);
}

#[test]
fn unknown_transaction_id_errors() {
    let db = open_db();
    assert!(db
        .execute_in_transaction(999, "MATCH (n) RETURN n", None)
        .is_err());
    assert!(db.commit_transaction(999).is_err());
    assert!(db.rollback_transaction(999).is_err());
}

#[test]
fn idle_transaction_is_reaped() {
    let db = open_db();
    let tx = db.begin_transaction();
    // Zero timeout → the just-opened transaction is immediately idle-expired.
    db.reap_idle_transactions(Duration::from_secs(0));
    assert!(
        db.commit_transaction(tx).is_err(),
        "reaped transaction handle is gone",
    );
}

#[test]
fn concurrent_transactions_are_independent() {
    let mut db = open_db();
    let tx_a = db.begin_transaction();
    let tx_b = db.begin_transaction();
    assert_ne!(tx_a, tx_b, "each begin allocates a distinct id");

    db.execute_in_transaction(tx_a, "CREATE (n:User {name: 'A'})", None)
        .expect("a stmt");
    db.execute_in_transaction(tx_b, "CREATE (n:User {name: 'B'})", None)
        .expect("b stmt");

    // Commit A, roll back B → only A's write survives.
    db.commit_transaction(tx_a).expect("commit a");
    db.rollback_transaction(tx_b).expect("rollback b");

    let rows = db
        .execute_cypher("MATCH (n:User) RETURN n.name")
        .expect("read");
    assert_eq!(
        rows.len(),
        1,
        "only the committed transaction's write survives"
    );
}
