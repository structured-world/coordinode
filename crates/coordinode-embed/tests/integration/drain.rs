//! Integration tests for volatile write drain.
//!
//! A volatile write is accepted, not committed: no committed read sees it
//! until the drain hands it to the log, after which it is there exactly once.
//! Closing the database is the drain barrier these tests use: the drain thread
//! makes its final pass and is joined before the handle is gone. Also tests
//! the volatile journal levels' membership rule and causal session rejection.

#![allow(clippy::unwrap_used)]

use coordinode_core::txn::write_concern::{Journal, WriteAck, WriteConcern};
use coordinode_embed::Database;
use tempfile::tempdir;

/// A volatile write under `concern` is not seen by a committed read before
/// the drain, and is seen once after it.
fn invisible_until_drained(concern: WriteConcern) {
    let dir = tempdir().unwrap();
    {
        let mut db = Database::open(dir.path()).unwrap();
        db.set_write_concern(concern);
        let results = db
            .execute_cypher("CREATE (n:User {name: 'alice'}) RETURN n.name")
            .unwrap();
        assert_eq!(results[0].get("n.name").unwrap().as_str(), Some("alice"));
        // Nothing drains a write accepted this instant: the drain thread
        // waits out its batching window first.
        let read = db.execute_cypher("MATCH (n:User) RETURN n.name").unwrap();
        assert!(read.is_empty(), "visible before the drain: {read:?}");
    }
    let mut db = Database::open(dir.path()).unwrap();
    let read = db.execute_cypher("MATCH (n:User) RETURN n.name").unwrap();
    assert_eq!(read.len(), 1);
    assert_eq!(read[0].get("n.name").unwrap().as_str(), Some("alice"));
}

/// j:memory: not committed until drained, then there once.
#[test]
fn a_memory_write_is_invisible_until_drained() {
    invisible_until_drained(WriteConcern::memory());
}

/// j:cache: not committed until drained, then there once.
#[test]
fn a_cache_write_is_invisible_until_drained() {
    invisible_until_drained(WriteConcern::cache());
}

/// Several j:memory writes drain into the log, each once.
#[test]
fn memory_write_drains_to_pipeline() {
    let dir = tempdir().unwrap();
    {
        let mut db = Database::open(dir.path()).unwrap();
        db.set_write_concern(WriteConcern::memory());
        db.execute_cypher("CREATE (n:Person {name: 'bob'})")
            .unwrap();
        db.execute_cypher("CREATE (n:Person {name: 'carol'})")
            .unwrap();
    }
    let mut db = Database::open(dir.path()).unwrap();
    let results = db
        .execute_cypher("MATCH (n:Person) RETURN n.name ORDER BY n.name")
        .unwrap();
    assert_eq!(results.len(), 2);
}

/// Switching write concern mid-session: the journaled write is committed at
/// once, the j:memory one only once drained.
#[test]
fn switch_write_concern_mid_session() {
    let dir = tempdir().unwrap();
    {
        let mut db = Database::open(dir.path()).unwrap();
        db.set_write_concern(WriteConcern::memory());
        db.execute_cypher("CREATE (n:Sensor {id: 1})").unwrap();
        db.set_write_concern(WriteConcern::majority());
        db.execute_cypher("CREATE (n:Sensor {id: 2})").unwrap();
        let results = db.execute_cypher("MATCH (n:Sensor) RETURN n.id").unwrap();
        assert_eq!(results.len(), 1, "only the journaled write: {results:?}");
    }
    let mut db = Database::open(dir.path()).unwrap();
    let results = db.execute_cypher("MATCH (n:Sensor) RETURN n.id").unwrap();
    assert_eq!(results.len(), 2);
}

/// A journaled write of a key a j:memory write is waiting to drain is refused
/// rather than overwritten when the drain lands: the volatile write was
/// validated against the state before it, and the drain must not replace a
/// commit it never saw.
#[test]
fn a_journaled_write_of_a_key_waiting_to_drain_is_refused() {
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    db.execute_cypher("CREATE (n:Slot {k: 1, v: 0})").unwrap();
    db.set_write_concern(WriteConcern::memory());
    db.execute_cypher("MATCH (n:Slot {k: 1}) SET n.v = 1")
        .unwrap();
    db.set_write_concern(WriteConcern::majority());
    let refused = db.execute_cypher("MATCH (n:Slot {k: 1}) SET n.v = 2");
    assert!(refused.is_err(), "{refused:?}");
}

/// A volatile journal level is honoured for the leader alone: with `w` above
/// one the concern is refused instead of being silently rewritten on either
/// axis, and `w:0` bypasses the overlay because nobody waits for it.
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

/// j:memory rejects causal session validation.
#[test]
fn memory_rejects_causal_session() {
    let wc = WriteConcern::memory();
    assert!(wc.validate_for_causal_session().is_err());
}

/// j:cache rejects causal session validation.
#[test]
fn cache_rejects_causal_session() {
    let wc = WriteConcern::cache();
    assert!(wc.validate_for_causal_session().is_err());
}

/// Graceful shutdown flushes drain buffer.
#[test]
fn graceful_shutdown_flushes_drain() {
    let dir = tempdir().unwrap();
    {
        let mut db = Database::open(dir.path()).unwrap();
        db.set_write_concern(WriteConcern::memory());

        // Write data with j:memory.
        db.execute_cypher("CREATE (n:Log {msg: 'event1'})").unwrap();
        db.execute_cypher("CREATE (n:Log {msg: 'event2'})").unwrap();

        // Drop db — should flush drain buffer.
    }

    // Reopen and verify data survived (drained before shutdown).
    let mut db = Database::open(dir.path()).unwrap();
    let results = db.execute_cypher("MATCH (n:Log) RETURN n.msg").unwrap();
    assert_eq!(results.len(), 2);
}

/// A j:memory write lands once: the label counter its node adds to is an
/// additive merge operand, so applying the write when it is accepted and again
/// when it is drained counts the node twice.
#[test]
fn memory_write_counts_once_after_drain() {
    let dir = tempdir().unwrap();
    {
        let mut db = Database::open(dir.path()).unwrap();
        db.set_write_concern(WriteConcern::memory());
        db.execute_cypher("CREATE (n:Tally {k: 1})").unwrap();
        // Closing the database is the drain barrier: the drain thread makes
        // its final pass and is joined before the handle is gone.
    }

    let db = Database::open(dir.path()).unwrap();
    let counts = db.label_counts().unwrap();
    assert_eq!(counts.get("Tally").copied(), Some(1), "{counts:?}");
}

/// j:memory with edges: edge create uses merge operators, which drain once.
#[test]
fn memory_write_with_edges() {
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
