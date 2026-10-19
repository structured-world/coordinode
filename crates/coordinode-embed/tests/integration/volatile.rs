//! Integration tests for the volatile write concerns.
//!
//! `j: memory` and `j: cache` commit, apply and become visible exactly like a
//! journaled write; only the answer stops waiting for the journal's sync.
//! Closing the database syncs the journal, so a clean reopen finds every
//! write; what a crash before the sync loses is tested at the engine, where
//! the crash can be simulated.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_core::graph::types::Value;
use coordinode_core::txn::read_concern::{ReadConcern, ReadConcernLevel};
use coordinode_core::txn::write_concern::{Journal, WriteAck, WriteConcern};
use coordinode_embed::Database;
use coordinode_embed::db::DatabaseError;
use tempfile::tempdir;

/// A write under `concern` reads back the moment it returns, and once.
fn visible_at_once(concern: WriteConcern) {
    let dir = tempdir().unwrap();
    {
        let mut db = Database::open(dir.path()).unwrap();
        db.set_write_concern(concern);
        db.execute_cypher("CREATE (n:User {name: 'alice'})")
            .unwrap();
        let read = db.execute_cypher("MATCH (n:User) RETURN n.name").unwrap();
        assert_eq!(read.len(), 1, "{concern}: {read:?}");
        assert_eq!(read[0].get("n.name").unwrap().as_str(), Some("alice"));
    }
    let mut db = Database::open(dir.path()).unwrap();
    let read = db.execute_cypher("MATCH (n:User) RETURN n.name").unwrap();
    assert_eq!(read.len(), 1, "{concern} after a clean reopen: {read:?}");
}

#[test]
fn a_memory_write_is_visible_at_once() {
    visible_at_once(WriteConcern::memory());
}

#[test]
fn a_cache_write_is_visible_at_once() {
    visible_at_once(WriteConcern::cache());
}

/// Volatile and journaled writes interleave on one handle, each visible as
/// soon as it returns, and a journaled write of a key a volatile one just
/// wrote replaces it like any later write: the volatile one is committed.
#[test]
fn volatile_and_journaled_writes_interleave() {
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    db.set_write_concern(WriteConcern::memory());
    db.execute_cypher("CREATE (n:Slot {k: 1, v: 1})").unwrap();
    db.set_write_concern(WriteConcern::majority());
    db.execute_cypher("MATCH (n:Slot {k: 1}) SET n.v = 2")
        .unwrap();
    db.set_write_concern(WriteConcern::cache());
    db.execute_cypher("CREATE (n:Slot {k: 2, v: 3})").unwrap();
    let read = db
        .execute_cypher("MATCH (n:Slot) RETURN n.k, n.v ORDER BY n.k")
        .unwrap();
    assert_eq!(read.len(), 2, "{read:?}");
    assert_eq!(read[0].get("n.v"), Some(&Value::Int(2)));
    assert_eq!(read[1].get("n.v"), Some(&Value::Int(3)));
}

/// A volatile write is applied once: the label counter its node adds to is
/// an additive merge operand, so a second application would count it twice.
#[test]
fn a_memory_write_counts_once() {
    let dir = tempdir().unwrap();
    {
        let mut db = Database::open(dir.path()).unwrap();
        db.set_write_concern(WriteConcern::memory());
        db.execute_cypher("CREATE (n:Tally {k: 1})").unwrap();
        let counts = db.label_counts().unwrap();
        assert_eq!(counts.get("Tally").copied(), Some(1), "{counts:?}");
    }
    let db = Database::open(dir.path()).unwrap();
    let counts = db.label_counts().unwrap();
    assert_eq!(counts.get("Tally").copied(), Some(1), "{counts:?}");
}

/// Edges, written as merge operands, land once under a volatile concern.
#[test]
fn a_memory_write_with_edges() {
    let dir = tempdir().unwrap();
    {
        let mut db = Database::open(dir.path()).unwrap();
        db.set_write_concern(WriteConcern::memory());
        db.execute_cypher("CREATE (a:Person {name: 'alice'})-[:KNOWS]->(b:Person {name: 'bob'})")
            .unwrap();
    }
    let mut db = Database::open(dir.path()).unwrap();
    let results = db
        .execute_cypher("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name")
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].get("a.name").unwrap().as_str(), Some("alice"));
    assert_eq!(results[0].get("b.name").unwrap().as_str(), Some("bob"));
}

/// A schema change takes the one write path too: under a volatile concern it
/// is published at once and kept by a clean reopen.
#[test]
fn a_schema_change_under_a_memory_concern() {
    let dir = tempdir().unwrap();
    {
        let mut db = Database::open(dir.path()).unwrap();
        db.set_write_concern(WriteConcern::memory());
        db.execute_cypher("CREATE INDEX user_email ON :User(email)")
            .unwrap();
        db.execute_cypher("CREATE (n:User {email: 'a@x'})").unwrap();
        let read = db
            .execute_cypher("MATCH (n:User {email: 'a@x'}) RETURN n.email")
            .unwrap();
        assert_eq!(read.len(), 1, "{read:?}");
    }
    let mut db = Database::open(dir.path()).unwrap();
    let read = db
        .execute_cypher("MATCH (n:User {email: 'a@x'}) RETURN n.email")
        .unwrap();
    assert_eq!(read.len(), 1, "{read:?}");
}

/// A store without a journal (in memory) takes volatile writes like any.
#[test]
fn a_memory_write_on_an_in_memory_store() {
    let mut db = Database::open_in_memory().unwrap();
    db.set_write_concern(WriteConcern::memory());
    db.execute_cypher("CREATE (n:Ephemeral {k: 1})").unwrap();
    let read = db.execute_cypher("MATCH (n:Ephemeral) RETURN n.k").unwrap();
    assert_eq!(read.len(), 1);
}

/// A store whose journal flusher stays asleep for the test: a volatile write
/// stays unsynced until something else syncs the journal.
fn open_flusher_held(dir: &std::path::Path) -> Database {
    use coordinode_storage::engine::config::{
        Durability, EndpointConfig, Media, StorageConfig, Tier,
    };
    let mut config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.join("data"),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    config.volatile_flush_interval_ms = 600_000;
    Database::open_with_config(config).unwrap()
}

fn read_at(level: ReadConcernLevel, at_timestamp: Option<u64>) -> ReadConcern {
    ReadConcern {
        level,
        at_timestamp,
        ..ReadConcern::default()
    }
}

/// A volatile write under `db`'s write concern, and its commit timestamp.
fn write_volatile(db: &mut Database, query: &str) -> u64 {
    db.execute_cypher_full(query, None, None, None, None)
        .unwrap()
        .commit_ts()
        .unwrap()
}

/// Whether the journal has synced the commit at `ts`. Asked after a read, it
/// says whether the read may have seen that commit: the bound only rises
/// while nothing new is written. Anything that syncs the journal (a table
/// flush, a journaled commit of a background service) may do so before the
/// read, which makes the write legitimately durable; a durable read seeing
/// it while this is false is the defect the tests look for.
fn synced(db: &Database, ts: u64) -> bool {
    db.engine().pending_commits().durable_ceiling() > ts
}

/// Before its sync a volatile write is seen by a `local` read and not by a
/// `majority`, `linearizable` or `snapshot` read, on a standalone store too;
/// the durable reads neither wait nor sync. A journaled write syncs the
/// journal and brings both into every view.
#[test]
fn durable_reads_skip_an_unsynced_write_until_it_syncs() {
    let dir = tempdir().unwrap();
    let mut db = open_flusher_held(dir.path());
    db.set_write_concern(WriteConcern::memory());
    let written = write_volatile(&mut db, "CREATE (n:Tick {k: 1})");

    let count = |db: &mut Database, level| {
        db.execute_cypher_with_read_concern("MATCH (n:Tick) RETURN n.k", read_at(level, None))
            .unwrap()
            .len()
    };
    assert_eq!(count(&mut db, ReadConcernLevel::Local), 1);
    for level in [
        ReadConcernLevel::Majority,
        ReadConcernLevel::Linearizable,
        ReadConcernLevel::Snapshot,
    ] {
        let seen = count(&mut db, level);
        assert!(
            seen == 0 || synced(&db, written),
            "{level:?} saw an unsynced write"
        );
    }

    db.set_write_concern(WriteConcern::majority());
    db.execute_cypher("CREATE (n:Tick {k: 2})").unwrap();
    for level in [ReadConcernLevel::Local, ReadConcernLevel::Majority] {
        assert_eq!(count(&mut db, level), 2, "{level:?}");
    }
}

/// A snapshot read at a timestamp covering an unsynced write is refused as
/// not ready at once, naming what is readable now, instead of waiting; a
/// read at the timestamp before it is answered, and once the journal syncs
/// the first read succeeds.
#[test]
fn a_read_at_an_unsynced_timestamp_is_not_ready() {
    let dir = tempdir().unwrap();
    let mut db = open_flusher_held(dir.path());
    db.set_write_concern(WriteConcern::memory());
    let written = write_volatile(&mut db, "CREATE (n:Tick {k: 1})");

    let started = std::time::Instant::now();
    let read = db.execute_cypher_with_read_concern(
        "MATCH (n:Tick) RETURN n.k",
        read_at(ReadConcernLevel::Snapshot, Some(written)),
    );
    assert!(
        started.elapsed() < std::time::Duration::from_secs(1),
        "the read waited"
    );
    match read {
        // Answered only when something synced the journal first.
        Ok(rows) => {
            assert!(synced(&db, written), "answered over an unsynced write");
            assert_eq!(rows.len(), 1, "{rows:?}");
        }
        Err(DatabaseError::ReadNotReady {
            requested,
            ready_through,
        }) => {
            assert_eq!(requested, written);
            let ready = ready_through.expect("an earlier timestamp is readable");
            assert!(ready < written);
            let before = db
                .execute_cypher_with_read_concern(
                    "MATCH (n:Tick) RETURN n.k",
                    read_at(ReadConcernLevel::Snapshot, Some(ready)),
                )
                .unwrap();
            assert!(before.is_empty(), "{before:?}");
        }
        Err(other) => panic!("not a not-ready refusal: {other:?}"),
    }

    db.set_write_concern(WriteConcern::majority());
    db.execute_cypher("CREATE (n:Other {k: 1})").unwrap();
    let after = db
        .execute_cypher_with_read_concern(
            "MATCH (n:Tick) RETURN n.k",
            read_at(ReadConcernLevel::Snapshot, Some(written)),
        )
        .unwrap();
    assert_eq!(after.len(), 1, "{after:?}");
}

/// A cursor's first page chooses an eligible view, the latest durable one,
/// rather than naming a fresh timestamp: an unsynced write does not refuse
/// it, and every page reads that one view.
#[test]
fn a_cursor_starts_at_the_durable_view() {
    let dir = tempdir().unwrap();
    let mut db = open_flusher_held(dir.path());
    db.execute_cypher("CREATE (:Tick {k: 0})").unwrap();
    db.set_write_concern(WriteConcern::memory());
    let written = write_volatile(&mut db, "CREATE (n:Tick {k: 1})");
    let page = db
        .execute_cypher_paged("MATCH (n:Tick) RETURN n.k", None, None, None, 10)
        .expect("an unsynced write does not refuse a new cursor");
    assert!(
        page.rows.len() == 1 || synced(&db, written),
        "the cursor saw an unsynced write: {:?}",
        page.rows
    );
}

/// An interactive transaction under a durable read concern reads the durable
/// view for its whole life.
#[test]
fn a_durable_transaction_starts_below_an_unsynced_write() {
    let dir = tempdir().unwrap();
    let mut db = open_flusher_held(dir.path());
    db.set_write_concern(WriteConcern::memory());
    let written = write_volatile(&mut db, "CREATE (n:Tick {k: 1})");
    db.set_read_concern(ReadConcernLevel::Majority);
    let txn = db.begin_transaction().expect("begin");
    let read = db
        .execute_in_transaction(txn, "MATCH (n:Tick) RETURN n.k", None)
        .unwrap();
    assert!(
        read.is_empty() || synced(&db, written),
        "saw an unsynced write: {read:?}"
    );
    db.rollback_transaction(txn).unwrap();
}

/// A volatile journal level is honoured for the leader alone: with `w` above
/// one the concern is refused instead of being silently rewritten on either
/// axis, and `w:0` is answered before anything is held anywhere.
#[test]
fn volatile_journal_is_leader_only() {
    for journal in [Journal::Memory, Journal::Cache] {
        let leader_only = WriteConcern {
            w: WriteAck::LEADER,
            journal,
            timeout_ms: 0,
        };
        assert!(leader_only.validate(None).is_ok());
        assert!(leader_only.is_volatile());

        let two_members = WriteConcern {
            w: WriteAck::Acks(2),
            journal,
            timeout_ms: 0,
        };
        assert!(two_members.validate(None).is_err());

        let majority = WriteConcern {
            w: WriteAck::Majority,
            journal,
            timeout_ms: 0,
        };
        assert!(majority.validate(None).is_err());

        let fire_and_forget = WriteConcern {
            w: WriteAck::NONE,
            journal,
            timeout_ms: 0,
        };
        assert!(fire_and_forget.validate(None).is_ok());
        assert!(!fire_and_forget.is_volatile());
    }
}

/// A volatile concern cannot back a causal session: its position may be
/// lost with the unsynced tail.
#[test]
fn volatile_concerns_reject_causal_sessions() {
    assert!(
        WriteConcern::memory()
            .validate_for_causal_session()
            .is_err()
    );
    assert!(WriteConcern::cache().validate_for_causal_session().is_err());
}
