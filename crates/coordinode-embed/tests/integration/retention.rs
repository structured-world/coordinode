//! Integration tests: the MVCC retention window in library mode.
//!
//! `AS OF TIMESTAMP` and `ReadConcern.at_timestamp` stay exact inside the
//! configured window across compaction, are refused below the horizon, and
//! the window is tunable at runtime with an observable effect.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_core::graph::types::Value;
use coordinode_core::txn::read_concern::{ReadConcern, ReadConcernLevel};
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_embed::{Database, DatabaseError};
use coordinode_query::executor::runner::ExecutionError;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::partition::Partition;
use std::time::Duration;

/// A durable database in a temp dir (compaction needs real tables).
fn open_db(dir: &std::path::Path) -> Database {
    Database::open(dir).expect("open db")
}

/// Count the `Anchor` nodes visible at snapshot `ts`.
fn anchors_as_of(db: &mut Database, ts: u64) -> Result<usize, DatabaseError> {
    db.execute_cypher(&format!("MATCH (n:Anchor) RETURN n AS OF TIMESTAMP {ts}"))
        .map(|rows| rows.len())
}

fn anchors_at(db: &mut Database, ts: u64) -> Result<usize, DatabaseError> {
    db.execute_cypher_full(
        "MATCH (n:Anchor) RETURN n",
        None,
        None,
        Some(ReadConcern {
            level: ReadConcernLevel::Snapshot,
            at_timestamp: Some(ts),
            ..ReadConcern::default()
        }),
        None,
    )
    .map(|result| result.rows.len())
}

fn commit_anchor(db: &mut Database, id: u64) -> u64 {
    let tx = db.begin_transaction();
    db.execute_in_transaction(tx, &format!("CREATE (n:Anchor {{id: {id}}})"), None)
        .expect("create");
    db.commit_transaction(tx)
        .expect("commit")
        .commit_ts
        .as_raw()
}

/// Move the database clock forward by `by`: the oracle is the seqno source,
/// so every later commit and the retention floor follow it.
fn advance_clock(db: &Database, by: Duration) {
    let oracle = db.engine().oracle().expect("embedded engine has an oracle");
    let now = oracle.current().as_raw();
    oracle.advance_to(Timestamp::from_raw(
        now + u64::try_from(by.as_micros()).expect("fits"),
    ));
}

/// Flush and compact the node partition so version GC actually runs.
fn compact(db: &Database) {
    db.engine()
        .force_compaction(Partition::Node)
        .expect("compact");
}

/// Inside the window, a time-travel read between two commits keeps seeing
/// exactly the first commit after a compaction, in both read APIs.
#[test]
fn as_of_inside_the_window_is_exact_across_compaction() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = open_db(dir.path());
    assert_eq!(db.retention_window(), Duration::from_secs(7 * 24 * 3600));

    let first = commit_anchor(&mut db, 1);
    advance_clock(&db, Duration::from_secs(60));
    let second = commit_anchor(&mut db, 2);
    compact(&db);

    assert_eq!(anchors_as_of(&mut db, first).expect("as of first"), 1);
    assert_eq!(anchors_as_of(&mut db, second).expect("as of second"), 2);
    assert_eq!(anchors_as_of(&mut db, first - 1).expect("before first"), 0);
    assert_eq!(anchors_at(&mut db, first).expect("at first"), 1);
    assert_eq!(anchors_at(&mut db, second).expect("at second"), 2);
    assert!(
        db.oldest_readable_timestamp().as_raw() < first,
        "the default window keeps both commits readable"
    );
}

/// Once the window has elapsed (clock driven through the oracle) and
/// compaction ran, a read below the horizon is refused by both APIs, a
/// read at the horizon and above is served, and committed data is intact.
#[test]
fn as_of_below_the_horizon_is_refused_after_the_window_elapses() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = open_db(dir.path());
    db.set_retention_window(Duration::from_secs(2));

    let first = commit_anchor(&mut db, 1);
    advance_clock(&db, Duration::from_secs(10));
    let second = commit_anchor(&mut db, 2);
    compact(&db);

    let horizon = db.oldest_readable_timestamp().as_raw();
    assert!(
        first < horizon && horizon < second,
        "horizon {horizon} must lie between the old commit {first} and the recent one {second}"
    );

    // Every statement allocates a timestamp and nudges the horizon forward
    // by a few microseconds, so the error reports a horizon at or slightly
    // past the one sampled above, still well before the recent commit.
    match anchors_as_of(&mut db, first) {
        Err(DatabaseError::Execution(ExecutionError::OutsideRetention {
            requested,
            oldest_readable,
        })) => {
            assert_eq!(requested, i64::try_from(first).expect("fits"));
            assert!(horizon <= oldest_readable && oldest_readable < second);
        }
        other => panic!("expected OutsideRetention, got {other:?}"),
    }
    match anchors_at(&mut db, first) {
        Err(DatabaseError::OutsideRetention {
            requested,
            oldest_readable,
        }) => {
            assert_eq!(requested, first);
            assert!(horizon <= oldest_readable && oldest_readable < second);
        }
        other => panic!("expected OutsideRetention, got {other:?}"),
    }

    // Above the horizon: served and exact. Every statement allocates a
    // timestamp, which nudges the horizon forward by a few microseconds, so
    // probe a millisecond past it rather than exactly at it.
    let probe = horizon + 1_000;
    assert!(probe < second);
    assert_eq!(anchors_as_of(&mut db, probe).expect("above horizon"), 1);
    assert_eq!(anchors_at(&mut db, probe).expect("above horizon"), 1);
    assert_eq!(anchors_as_of(&mut db, second).expect("as of second"), 2);
    assert_eq!(
        db.execute_cypher("MATCH (n:Anchor) RETURN n")
            .expect("current read")
            .len(),
        2,
        "committed data is untouched by the refusal"
    );
}

/// Widening the window at runtime moves the horizon back at once, so a
/// read that was refused a moment ago is served again while its history is
/// still on disk; narrowing it moves the horizon forward.
#[test]
fn runtime_window_change_moves_the_horizon() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = open_db(dir.path());
    db.set_retention_window(Duration::from_secs(2));

    let first = commit_anchor(&mut db, 1);
    advance_clock(&db, Duration::from_secs(10));
    commit_anchor(&mut db, 2);
    assert!(matches!(
        anchors_as_of(&mut db, first),
        Err(DatabaseError::Execution(
            ExecutionError::OutsideRetention { .. }
        ))
    ));

    // No compaction ran since the window closed, so the history is still
    // there: a wider window makes it readable again.
    db.set_retention_window(Duration::from_secs(3_600));
    assert_eq!(db.retention_window(), Duration::from_secs(3_600));
    assert!(db.oldest_readable_timestamp().as_raw() < first);
    assert_eq!(anchors_as_of(&mut db, first).expect("readable again"), 1);

    db.set_retention_window(Duration::from_secs(1));
    assert!(db.oldest_readable_timestamp().as_raw() > first);
    assert!(matches!(
        anchors_as_of(&mut db, first),
        Err(DatabaseError::Execution(
            ExecutionError::OutsideRetention { .. }
        ))
    ));
}

/// The window configured through `StorageConfig` is the one the database
/// opens with.
#[test]
fn window_comes_from_storage_config() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    config.retention_window_secs = 42;
    let db = Database::open_with_config(config).expect("open");
    assert_eq!(db.retention_window(), Duration::from_secs(42));
    // Republish against the current clock: opening allocated timestamps
    // after the last recompute.
    db.engine().advance_gc_watermark();
    let now = db.engine().snapshot();
    assert_eq!(
        db.oldest_readable_timestamp().as_raw(),
        now - 42_000_000 - 1
    );
}

/// A live snapshot pin (an open interactive transaction reading at its
/// start timestamp) keeps its history readable even as the window closes
/// around it: the watermark is the minimum of the window floor and every
/// live pin, so narrowing the window never invalidates an open transaction's
/// snapshot.
#[test]
fn open_transaction_snapshot_survives_window_narrowing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = open_db(dir.path());
    db.set_retention_window(Duration::from_secs(3_600));

    commit_anchor(&mut db, 1);
    let tx = db.begin_transaction();
    let seen_at_begin = db
        .execute_in_transaction(tx, "MATCH (n:Anchor) RETURN n", None)
        .expect("read in txn")
        .len();
    assert_eq!(seen_at_begin, 1);

    advance_clock(&db, Duration::from_secs(10));
    commit_anchor(&mut db, 2);
    db.set_retention_window(Duration::from_secs(1));
    compact(&db);

    // The transaction still reads its own snapshot: one anchor.
    assert_eq!(
        db.execute_in_transaction(tx, "MATCH (n:Anchor) RETURN n", None)
            .expect("repeatable read")
            .len(),
        1
    );
    db.rollback_transaction(tx).expect("rollback");
}

/// With no time-travel window, only live pins hold the GC watermark back, so
/// a statement's snapshot has to be pinned the moment it is chosen. Pinned
/// later, commits landing meanwhile let the watermark pass it and the
/// statement is refused as outside retention although it asked for the
/// latest state. No statement here may fail that way, through either the
/// auto-commit or the interactive path, and every counter ends exact.
///
/// A write conflict is a different and legitimate answer: a snapshot that
/// waited its bounded time for a commit still in flight steps behind it, and
/// can then miss the writer's own previous commit. The attempt is refused and
/// retried, which is what a client does.
#[test]
fn a_zero_window_refuses_no_statement_under_concurrent_writes() {
    const WRITERS: u64 = 4;
    const ROUNDS: u64 = 150;
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = open_db(dir.path());
    db.set_retention_window(Duration::ZERO);
    for w in 0..WRITERS {
        db.execute_cypher(&format!("CREATE (:Tally {{id: {w}, v: 0}})"))
            .expect("seed");
    }

    let failures = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for w in 0..WRITERS {
            let (db, failures) = (&db, &failures);
            s.spawn(move || {
                let bump = format!("MATCH (n:Tally {{id: {w}}}) SET n.v = n.v + 1");
                let retryable = |e: &DatabaseError| {
                    matches!(
                        e,
                        DatabaseError::TransactionConflict { .. }
                            | DatabaseError::Execution(ExecutionError::Conflict(_))
                    )
                };
                for _ in 0..ROUNDS {
                    loop {
                        match db.execute_cypher_shared(&bump, None, None, None, None) {
                            Ok(_) => break,
                            Err(e) if retryable(&e) => {}
                            Err(e) => {
                                failures
                                    .lock()
                                    .expect("lock")
                                    .push(format!("auto-commit: {e}"));
                                break;
                            }
                        }
                    }
                    loop {
                        let tx = db.begin_transaction();
                        let outcome = db
                            .execute_in_transaction(tx, &bump, None)
                            .and_then(|_| db.commit_transaction(tx).map(|_| ()));
                        match outcome {
                            Ok(()) => break,
                            Err(e) if retryable(&e) => {}
                            Err(e) => {
                                failures
                                    .lock()
                                    .expect("lock")
                                    .push(format!("interactive: {e}"));
                                break;
                            }
                        }
                    }
                }
            });
        }
    });

    let failures = failures.into_inner().expect("lock");
    assert!(
        failures.is_empty(),
        "{} of {} statements refused, first: {:?}",
        failures.len(),
        WRITERS * ROUNDS * 2,
        failures.first()
    );
    for w in 0..WRITERS {
        let rows = db
            .execute_cypher(&format!("MATCH (n:Tally {{id: {w}}}) RETURN n.v AS v"))
            .expect("read tally");
        assert_eq!(
            rows[0].get("v"),
            Some(&Value::Int(i64::try_from(ROUNDS * 2).expect("fits"))),
            "tally {w}"
        );
    }
}

fn commit(db: &mut Database, cypher: &str) -> u64 {
    let tx = db.begin_transaction();
    db.execute_in_transaction(tx, cypher, None)
        .expect("statement");
    db.commit_transaction(tx)
        .expect("commit")
        .commit_ts
        .as_raw()
}

/// The `w` property of every `LINK` edge matched as of `ts`, one entry per
/// row, so "no edge at that snapshot" and "an edge without the property"
/// read differently.
fn link_weight_as_of(db: &mut Database, ts: u64) -> Result<Vec<Option<Value>>, DatabaseError> {
    db.execute_cypher(&format!(
        "MATCH (:Anchor {{id: 1}})-[r:LINK]->(:Anchor {{id: 2}}) \
         RETURN r.w AS w AS OF TIMESTAMP {ts}"
    ))
    .map(|rows| rows.iter().map(|row| row.get("w").cloned()).collect())
}

/// An edge property rewritten on both sides of the horizon keeps the version
/// a snapshot inside the window resolves to, across a compaction of every
/// partition and across a restart: the read is neither refused nor answered
/// with the newer value or with nothing. Edge properties live in a partition
/// compacted without a merge operator, the path node reads do not take.
#[test]
fn edge_property_history_survives_compaction_and_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (probe, newest) = {
        let mut db = open_db(dir.path());
        db.set_retention_window(Duration::from_secs(2));

        commit(
            &mut db,
            "CREATE (a:Anchor {id: 1})-[:LINK {w: 1}]->(b:Anchor {id: 2})",
        );
        let second = commit(
            &mut db,
            "MATCH (:Anchor {id: 1})-[r:LINK]->(:Anchor {id: 2}) SET r.w = 2",
        );
        advance_clock(&db, Duration::from_secs(10));
        let newest = commit(
            &mut db,
            "MATCH (:Anchor {id: 1})-[r:LINK]->(:Anchor {id: 2}) SET r.w = 3",
        );
        // Both old versions sit below the horizon and the newest above it; a
        // snapshot just above the horizon resolves to the newer of the two.
        // Every statement nudges the horizon forward by a few microseconds,
        // so probe a millisecond past it.
        let horizon = db.oldest_readable_timestamp().as_raw();
        assert!(second < horizon && horizon < newest);
        let probe = horizon + 1_000;
        assert!(probe < newest);
        assert_eq!(
            link_weight_as_of(&mut db, probe).expect("before compaction"),
            vec![Some(Value::Int(2))]
        );
        assert_eq!(
            link_weight_as_of(&mut db, newest).expect("before compaction"),
            vec![Some(Value::Int(3))]
        );

        for part in Partition::all() {
            if *part != Partition::Raft {
                db.engine().force_compaction(*part).expect("compact");
            }
        }
        assert_eq!(
            link_weight_as_of(&mut db, probe).expect("inside the window"),
            vec![Some(Value::Int(2))]
        );
        assert!(
            matches!(
                link_weight_as_of(&mut db, second - 1),
                Err(DatabaseError::Execution(
                    ExecutionError::OutsideRetention { .. }
                ))
            ),
            "the collected version is refused, not answered"
        );
        (probe, newest)
    };

    // A fresh process. The default window puts the policy horizon far in the
    // past, so what bounds the read is the floor the compaction recorded.
    let mut db = open_db(dir.path());
    assert!(
        db.oldest_readable_timestamp().as_raw() <= probe,
        "the recorded floor still admits a snapshot the window covered"
    );
    assert_eq!(
        link_weight_as_of(&mut db, probe).expect("inside the window after a reopen"),
        vec![Some(Value::Int(2))]
    );
    assert_eq!(
        link_weight_as_of(&mut db, newest).expect("newest"),
        vec![Some(Value::Int(3))]
    );
}
