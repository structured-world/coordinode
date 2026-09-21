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
    assert!(
        db.execute_in_transaction(999, "MATCH (n) RETURN n", None)
            .is_err()
    );
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
fn max_buffered_bytes_aborts_transaction() {
    let mut db = open_db();
    // Tiny ceiling so a single CREATE's buffered write exceeds it.
    db.set_max_interactive_txn_bytes(8);
    let tx = db.begin_transaction();
    let err = db
        .execute_in_transaction(
            tx,
            "CREATE (n:Big {payload: 'this exceeds eight bytes'})",
            None,
        )
        .expect_err("statement must abort over the byte ceiling");
    // The breach is a typed variant carrying the numbers, not a sentence to
    // grep. This asserted on the message text until the variant existed.
    match err {
        coordinode_embed::DatabaseError::TransactionTooLarge {
            id,
            buffered,
            limit,
        } => {
            assert_eq!(id, tx);
            assert_eq!(limit, 8);
            assert!(buffered > limit, "breach must exceed the ceiling");
        }
        other => panic!("expected TransactionTooLarge, got {other}"),
    }
    // Aborted → handle consumed: commit fails.
    assert!(db.commit_transaction(tx).is_err());
    // Nothing committed.
    let rows = db.execute_cypher("MATCH (n:Big) RETURN n").expect("read");
    assert_eq!(rows.len(), 0);
}

#[test]
fn configured_idle_timeout_reaps_on_begin() {
    let mut db = open_db();
    // Zero timeout: any prior open transaction is idle on the next begin.
    db.set_interactive_idle_timeout(Duration::from_secs(0));
    let stale = db.begin_transaction();
    // A second begin runs the reaper with the configured (zero) timeout.
    let _fresh = db.begin_transaction();
    assert!(
        db.commit_transaction(stale).is_err(),
        "stale transaction reaped by the configured idle timeout on begin",
    );
}

#[test]
fn commit_conflict_aborts_second_transaction() {
    let mut db = open_db();
    db.execute_cypher("CREATE (n:Acct {id: 1, bal: 100})")
        .expect("seed");

    // Two interactive transactions both read-modify-write the same node.
    let tx_a = db.begin_transaction();
    let tx_b = db.begin_transaction();
    db.execute_in_transaction(tx_a, "MATCH (n:Acct {id: 1}) SET n.bal = n.bal + 10", None)
        .expect("a read-modify-write");
    db.execute_in_transaction(tx_b, "MATCH (n:Acct {id: 1}) SET n.bal = n.bal + 20", None)
        .expect("b read-modify-write");

    // A commits first → succeeds.
    db.commit_transaction(tx_a).expect("commit a");
    // B read the node at its snapshot; A committed a write to it after B began
    // → OCC conflict at commit, B is rejected (lost update prevented).
    assert!(
        db.commit_transaction(tx_b).is_err(),
        "B's commit conflicts with A's concurrent write to the same node",
    );

    // Exactly A's update is durable.
    let rows = db
        .execute_cypher("MATCH (n:Acct {id: 1}) RETURN n.bal AS bal")
        .expect("read");
    assert_eq!(rows.len(), 1);
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

#[test]
fn reading_what_another_transaction_writes_is_not_a_conflict() {
    // Snapshot isolation, first-committer-wins on WRITES only: the mainstream
    // default. In PostgreSQL (read committed and repeatable read alike), Oracle
    // and MongoDB, a transaction that merely READ data someone else changed
    // commits fine; only two writers on the same row/document collide. Failing
    // this commit would make every read-heavy transaction race with every
    // writer, which is the strictness of opt-in SERIALIZABLE, not of a default.
    let mut db = open_db();
    db.execute_cypher("CREATE (n:Acct {id: 1, bal: 100})")
        .expect("seed");
    db.execute_cypher("CREATE (n:Audit {id: 9, seen: 0})")
        .expect("seed audit");

    let reader = db.begin_transaction();
    let writer = db.begin_transaction();

    // The reader READS the account and writes somewhere else entirely.
    db.execute_in_transaction(
        reader,
        "MATCH (a:Acct {id: 1}) MATCH (u:Audit {id: 9}) SET u.seen = a.bal",
        None,
    )
    .expect("reader statement");
    // The writer modifies the account the reader looked at.
    db.execute_in_transaction(writer, "MATCH (a:Acct {id: 1}) SET a.bal = 200", None)
        .expect("writer statement");

    db.commit_transaction(writer).expect("writer commits first");
    db.commit_transaction(reader)
        .expect("a stale READ must not abort a commit whose writes touch nobody");

    // The reader captured the value as of ITS snapshot: that is what snapshot
    // isolation promises, and asserting it guards against `seen` silently
    // picking up the writer's 200.
    let rows = db
        .execute_cypher("MATCH (u:Audit {id: 9}) RETURN u.seen AS seen")
        .expect("read back");
    assert_eq!(rows.len(), 1);
}

/// Count the `Anchor` nodes visible at snapshot `ts` (`AS OF TIMESTAMP` takes
/// the raw HLC value directly: commit timestamps are storage seqnos).
fn anchors_as_of(db: &mut Database, ts: u64) -> usize {
    db.execute_cypher(&format!("MATCH (n:Anchor) RETURN n AS OF TIMESTAMP {ts}"))
        .expect("as-of read")
        .len()
}

#[test]
fn commit_receipt_is_the_snapshot_boundary_of_the_write() {
    // The receipt's commit_ts must be exactly the seqno the mutations landed
    // at: a snapshot AT commit_ts sees the write, a snapshot ONE TICK BEFORE
    // does not. Both directions are asserted so an off-by-one in either the
    // receipt or the snapshot inclusivity fails loudly instead of passing as
    // "some timestamp near the commit".
    let mut db = open_db();
    let tx = db.begin_transaction();
    db.execute_in_transaction(tx, "CREATE (n:Anchor {id: 1})", None)
        .expect("create");
    let receipt = db.commit_transaction(tx).expect("commit");
    let ts = receipt.commit_ts.as_raw();

    assert!(ts > 0, "commit_ts is a real HLC value, never zero");
    assert_eq!(
        anchors_as_of(&mut db, ts),
        1,
        "snapshot AT commit_ts sees the write"
    );
    assert_eq!(
        anchors_as_of(&mut db, ts - 1),
        0,
        "snapshot one tick before commit_ts does not see the write"
    );
}

#[test]
fn as_of_timestamp_rejects_a_negative_literal() {
    // The snapshot seqno is unsigned; a negative literal would wrap to a huge
    // value and silently read "everything". It must be rejected as an error,
    // and the rejection must not disturb committed data.
    let mut db = open_db();
    let tx = db.begin_transaction();
    db.execute_in_transaction(tx, "CREATE (n:Anchor {id: 11})", None)
        .expect("create");
    db.commit_transaction(tx).expect("commit");

    let err = db
        .execute_cypher("MATCH (n:Anchor) RETURN n AS OF TIMESTAMP -1")
        .expect_err("a negative AS OF TIMESTAMP is out of range");
    assert!(
        err.to_string().contains("out of range"),
        "unexpected error: {err}"
    );
    assert_eq!(
        db.execute_cypher("MATCH (n:Anchor) RETURN n")
            .expect("read")
            .len(),
        1,
        "the rejected read leaves committed data untouched"
    );
}

#[test]
fn commit_receipt_has_no_raft_index_in_embedded_mode() {
    // Embedded mode has no Raft log, so the receipt says so with `None`
    // rather than a fake 0 the host could mistake for a real index.
    let db = open_db();
    let tx = db.begin_transaction();
    db.execute_in_transaction(tx, "CREATE (n:Anchor {id: 2})", None)
        .expect("create");
    let receipt = db.commit_transaction(tx).expect("commit");
    assert_eq!(receipt.applied_index, None);
}

#[test]
fn commit_receipt_anchors_a_changed_keys_cursor() {
    use coordinode_storage::engine::partition::Partition;

    // A change consumer positions itself with the receipt: a changed-keys
    // scan from commit_ts includes this commit (the scan is inclusive, pairing
    // with snapshots that see strictly below their seqno), and a consumer
    // that has processed it resumes from commit_ts + 1, which yields nothing.
    // This is the contract a polling CDC dispatcher builds on.
    let db = open_db();
    let tx = db.begin_transaction();
    db.execute_in_transaction(tx, "CREATE (n:Anchor {id: 3})", None)
        .expect("create");
    let receipt = db.commit_transaction(tx).expect("commit");
    let ts = receipt.commit_ts.as_raw();

    let including = db
        .engine()
        .changed_keys_since(Partition::Node, ts)
        .expect("scan from commit_ts");
    assert!(
        !including.is_empty(),
        "a scan from commit_ts replays the committed node"
    );
    let after = db
        .engine()
        .changed_keys_since(Partition::Node, ts + 1)
        .expect("scan after");
    assert!(
        after.is_empty(),
        "nothing in the node partition changed after the commit"
    );
}

#[test]
fn commit_receipts_are_monotonic_across_commits() {
    // Two sequential commits get strictly increasing commit timestamps: a
    // cursor built from the later receipt never replays the earlier commit.
    let mut db = open_db();
    let first = db.begin_transaction();
    db.execute_in_transaction(first, "CREATE (n:Anchor {id: 4})", None)
        .expect("create 1");
    let r1 = db.commit_transaction(first).expect("commit 1");

    let second = db.begin_transaction();
    db.execute_in_transaction(second, "CREATE (n:Anchor {id: 5})", None)
        .expect("create 2");
    let r2 = db.commit_transaction(second).expect("commit 2");

    assert!(r2.commit_ts > r1.commit_ts, "later commit, later timestamp");
    assert_eq!(anchors_as_of(&mut db, r1.commit_ts.as_raw()), 1);
    assert_eq!(anchors_as_of(&mut db, r2.commit_ts.as_raw()), 2);
}

#[test]
fn read_only_commit_reports_its_pinned_snapshot() {
    // A transaction that wrote nothing still returns a receipt: its pinned
    // read timestamp, so a host can use it as an "as of this point" anchor.
    // It must not be later than the engine's current seqno (it was allocated
    // at begin) and must not be zero.
    let mut db = open_db();
    db.execute_cypher("CREATE (n:Anchor {id: 6})")
        .expect("seed");
    let tx = db.begin_transaction();
    let rows = db
        .execute_in_transaction(tx, "MATCH (n:Anchor) RETURN n", None)
        .expect("read");
    assert_eq!(rows.len(), 1);
    let receipt = db.commit_transaction(tx).expect("read-only commit");
    let ts = receipt.commit_ts.as_raw();
    assert!(ts > 0);
    assert!(
        ts <= db.engine().current_seqno(),
        "pinned at begin, never in the future"
    );
    assert_eq!(receipt.applied_index, None, "nothing was proposed");
}

#[test]
fn rollback_yields_no_receipt_and_no_change_to_scan_from() {
    // Rollback returns unit, not a receipt: there is no timestamp to anchor
    // on because nothing became durable. A cursor taken before the rolled-back
    // transaction stays empty for the node partition.
    use coordinode_storage::engine::partition::Partition;

    let db = open_db();
    let before = db.engine().current_seqno();
    let tx = db.begin_transaction();
    db.execute_in_transaction(tx, "CREATE (n:Anchor {id: 7})", None)
        .expect("create");
    db.rollback_transaction(tx).expect("rollback");

    let changed = db
        .engine()
        .changed_keys_since(Partition::Node, before)
        .expect("scan");
    assert!(
        changed.is_empty(),
        "a rolled-back write never reaches storage"
    );
    assert!(
        db.commit_transaction(tx).is_err(),
        "handle consumed by rollback"
    );
}

/// Read-modify-write against the version the caller read: it commits while
/// the node is still there, and is refused with the version that replaced it
/// once somebody else has written.
///
/// This is the loop a caller used to hand-roll out of a read, a write and a
/// hope. The point of stating the version is that the refusal is a fact about
/// the record rather than a guess about the race.
#[test]
fn a_transaction_commits_only_at_the_node_version_it_read() {
    use coordinode_core::graph::node::NodeId;
    use coordinode_embed::DatabaseError;

    let mut db = open_db();
    db.execute_cypher("CREATE (n:Account {balance: 100})")
        .expect("create");

    // The first node of a fresh database. Asserting the version is there also
    // asserts that assumption, loudly, rather than silently testing nothing.
    let account = NodeId::from_raw(1);
    let read_version = db
        .node_version(account)
        .expect("read version")
        .expect("the account exists");

    // Somebody else changes it between the read and the write.
    db.execute_cypher("MATCH (n:Account) SET n.balance = 250")
        .expect("concurrent write");
    let moved_to = db
        .node_version(account)
        .expect("read version")
        .expect("still there");
    assert_ne!(moved_to, read_version, "the write moved the version");

    // The stale writer's transaction is refused, and told what is there now.
    let stale = db.begin_transaction();
    db.execute_in_transaction(stale, "MATCH (n:Account) SET n.balance = 500", None)
        .expect("statement");
    db.expect_node_version(stale, account, Some(read_version))
        .expect("state the condition");
    match db.commit_transaction(stale) {
        Err(DatabaseError::RevisionMismatch {
            expected, current, ..
        }) => {
            assert_eq!(expected, Some(read_version));
            assert_eq!(current, Some(moved_to));
        }
        other => panic!("expected a version mismatch, got {other:?}"),
    }

    let after = db
        .execute_cypher("MATCH (n:Account) RETURN n.balance AS balance")
        .expect("read");
    assert_eq!(
        after[0].get("balance").and_then(|v| v.as_int()),
        Some(250),
        "the refused transaction applied nothing"
    );

    // Retried against what is actually there, it lands.
    let fresh = db.begin_transaction();
    db.execute_in_transaction(fresh, "MATCH (n:Account) SET n.balance = 500", None)
        .expect("statement");
    db.expect_node_version(fresh, account, Some(moved_to))
        .expect("state the condition");
    db.commit_transaction(fresh).expect("the version matches");

    let after = db
        .execute_cypher("MATCH (n:Account) RETURN n.balance AS balance")
        .expect("read");
    assert_eq!(after[0].get("balance").and_then(|v| v.as_int()), Some(500));
}

/// Two claimers of one record on the embedded surface: exactly one takes it,
/// and the other is told which version is there.
///
/// The same race the wire transport runs. Both are tested because a contract
/// that holds on one surface and not the other is not a contract: they share
/// the engine, but not the code that states the condition or renders the
/// refusal, and that code is where a guarantee gets lost.
#[test]
fn two_claimers_of_one_record_produce_one_winner() {
    use coordinode_core::graph::node::NodeId;
    use coordinode_embed::DatabaseError;

    let mut db = open_db();
    db.execute_cypher("CREATE (c:Lease {holder: 'nobody'})")
        .expect("create");

    let lease = NodeId::from_raw(1);
    let free_at = db
        .node_version(lease)
        .expect("read")
        .expect("the lease record exists");

    // Both stake the same version before either commits.
    let mut claims = Vec::new();
    for holder in ["first", "second"] {
        let tx = db.begin_transaction();
        db.execute_in_transaction(
            tx,
            &format!("MATCH (c:Lease) SET c.holder = '{holder}'"),
            None,
        )
        .expect("statement");
        db.expect_node_version(tx, lease, Some(free_at))
            .expect("each believes the lease is free");
        claims.push(tx);
    }

    db.commit_transaction(claims[0])
        .expect("the first claimer takes it");
    let refusal = db
        .commit_transaction(claims[1])
        .expect_err("the lease is no longer free");
    let DatabaseError::RevisionMismatch { current, .. } = refusal else {
        panic!("expected a version mismatch, got {refusal:?}");
    };
    assert_eq!(
        current,
        db.node_version(lease).expect("read"),
        "the loser is told the version that is actually there"
    );

    let holder = db
        .execute_cypher("MATCH (c:Lease) RETURN c.holder AS holder")
        .expect("read");
    assert_eq!(
        holder[0].get("holder").and_then(|v| v.as_str()),
        Some("first"),
        "and the winner's write is the one that stands"
    );
}

/// A fenced claim is the same primitive, not a second one.
///
/// The claim is a record. Holding it means having written it; keeping it
/// means every protected write states the version it was written at. When
/// somebody takes the claim over, the old holder's next write is refused, and
/// refused for the claim rather than for the data it was about to change,
/// which is what makes the fence a fence: the work it would have done never
/// reaches the records it protects.
#[test]
fn a_claim_fences_the_writes_of_the_holder_it_replaced() {
    use coordinode_core::graph::node::NodeId;
    use coordinode_embed::DatabaseError;

    let mut db = open_db();
    db.execute_cypher("CREATE (c:Claim {holder: 'first'})")
        .expect("take the claim");
    db.execute_cypher("CREATE (d:Data {value: 0})")
        .expect("the record the claim protects");

    let claim = NodeId::from_raw(1);
    let held_at = db
        .node_version(claim)
        .expect("read")
        .expect("the claim exists");

    // A protected write does two things, and needs both. It states the
    // claim's version, which is what catches a takeover that has already
    // committed; and it writes the claim record itself, which puts the claim
    // in its write set, so a takeover racing this very commit loses to
    // first-committer-wins instead of interleaving with it.
    let protected = db.begin_transaction();
    db.execute_in_transaction(protected, "MATCH (d:Data) SET d.value = 1", None)
        .expect("statement");
    db.execute_in_transaction(protected, "MATCH (c:Claim) SET c.touched = 1", None)
        .expect("write the claim record too");
    db.expect_node_version(protected, claim, Some(held_at))
        .expect("state the claim");

    // Somebody takes the claim over first.
    db.execute_cypher("MATCH (c:Claim) SET c.holder = 'second'")
        .expect("take over");

    let refusal = db
        .commit_transaction(protected)
        .expect_err("the claim is no longer held by this writer");
    assert!(
        matches!(refusal, DatabaseError::RevisionMismatch { expected, .. } if expected == Some(held_at)),
        "the refusal is about the claim it named, got {refusal:?}"
    );

    let data = db
        .execute_cypher("MATCH (d:Data) RETURN d.value AS value")
        .expect("read");
    assert_eq!(
        data[0].get("value").and_then(|v| v.as_int()),
        Some(0),
        "the fenced writer changed nothing it was about to change"
    );
}

/// After an outcome the caller never learned, the same condition tells it
/// which way the first attempt went.
///
/// The dangerous retry is the blind one: the client did not hear back, so it
/// runs the write again and applies it twice. Retrying with the version it
/// originally read answers the question instead. If the first attempt landed,
/// the version moved and the retry is refused, which is the client learning
/// that it succeeded; if it did not, the version is unchanged and the retry
/// does the work.
#[test]
fn a_retry_after_an_unheard_outcome_learns_which_way_it_went() {
    use coordinode_core::graph::node::NodeId;
    use coordinode_embed::DatabaseError;

    let mut db = open_db();
    db.execute_cypher("CREATE (n:Counter {value: 0})")
        .expect("create");
    let counter = NodeId::from_raw(1);
    let before = db.node_version(counter).expect("read").expect("exists");

    // The first attempt commits, but the caller never hears the receipt.
    let attempt = db.begin_transaction();
    db.execute_in_transaction(attempt, "MATCH (n:Counter) SET n.value = 1", None)
        .expect("statement");
    db.expect_node_version(attempt, counter, Some(before))
        .expect("state the condition");
    db.commit_transaction(attempt).expect("it did land");

    // Not knowing that, the caller retries the same work against the same
    // version it read at the start.
    let retry = db.begin_transaction();
    db.execute_in_transaction(retry, "MATCH (n:Counter) SET n.value = 1", None)
        .expect("statement");
    db.expect_node_version(retry, counter, Some(before))
        .expect("state the same condition");
    let outcome = db.commit_transaction(retry);

    assert!(
        matches!(outcome, Err(DatabaseError::RevisionMismatch { .. })),
        "the refusal is how the caller learns the first attempt landed, got {outcome:?}"
    );
    let value = db
        .execute_cypher("MATCH (n:Counter) RETURN n.value AS value")
        .expect("read");
    assert_eq!(
        value[0].get("value").and_then(|v| v.as_int()),
        Some(1),
        "and the work happened exactly once"
    );
}

/// The condition can only be stated on a transaction that exists, and a node
/// that does not exist has no version rather than a zero one.
#[test]
fn a_version_condition_refuses_what_it_cannot_be_stated_on() {
    use coordinode_core::graph::node::NodeId;
    use coordinode_embed::DatabaseError;

    let db = open_db();

    assert_eq!(
        db.node_version(NodeId::from_raw(404)).expect("read"),
        None,
        "a node that was never written has no version"
    );

    let err = db
        .expect_node_version(4242, NodeId::from_raw(1), None)
        .expect_err("there is no such transaction");
    assert!(
        matches!(err, DatabaseError::UnknownTransaction(4242)),
        "expected the unknown-transaction error, got {err:?}"
    );

    // An absent node with the create-if-absent condition commits, and the
    // same condition afterwards does not.
    let tx = db.begin_transaction();
    db.execute_in_transaction(tx, "CREATE (n:Account {balance: 1})", None)
        .expect("statement");
    db.expect_node_version(tx, NodeId::from_raw(1), None)
        .expect("state the condition");
    db.commit_transaction(tx).expect("nothing was there");

    let again = db.begin_transaction();
    db.execute_in_transaction(again, "MATCH (n:Account) SET n.balance = 2", None)
        .expect("statement");
    db.expect_node_version(again, NodeId::from_raw(1), None)
        .expect("state the condition");
    assert!(
        matches!(
            db.commit_transaction(again),
            Err(DatabaseError::RevisionMismatch { expected: None, .. })
        ),
        "the node exists now, so requiring its absence must refuse"
    );
}
