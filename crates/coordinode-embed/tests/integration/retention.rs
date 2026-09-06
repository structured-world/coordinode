//! Integration tests: the MVCC retention window in library mode.
//!
//! `AS OF TIMESTAMP` and `ReadConcern.at_timestamp` stay exact inside the
//! configured window across compaction, are refused below the horizon, and
//! the window is tunable at runtime with an observable effect.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

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
