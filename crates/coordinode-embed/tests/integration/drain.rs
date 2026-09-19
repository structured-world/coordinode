//! Integration tests for volatile write drain (R077a).
//!
//! Tests the full flow: Database with j:memory → write data → verify local
//! visibility → drain → verify data persisted. Also tests the volatile
//! journal levels' membership rule and causal session rejection.

#![allow(clippy::unwrap_used)]

use coordinode_core::txn::write_concern::{Journal, WriteAck, WriteConcern};
use coordinode_embed::Database;
use tempfile::tempdir;

/// j:memory write is visible immediately to local reader (same Database).
#[test]
fn memory_write_locally_visible() {
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    db.set_write_concern(WriteConcern::memory());

    // CREATE a node with w:memory.
    let results = db
        .execute_cypher("CREATE (n:User {name: 'alice'}) RETURN n.name")
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].get("n.name").unwrap().as_str(), Some("alice"));

    // READ should see the node — w:memory applies locally before ACK.
    let read = db.execute_cypher("MATCH (n:User) RETURN n.name").unwrap();
    assert_eq!(read.len(), 1);
    assert_eq!(read[0].get("n.name").unwrap().as_str(), Some("alice"));
}

/// j:memory writes are drained to pipeline after drain interval.
#[test]
fn memory_write_drains_to_pipeline() {
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    db.set_write_concern(WriteConcern::memory());

    // Write multiple nodes with w:memory.
    db.execute_cypher("CREATE (n:Person {name: 'bob'})")
        .unwrap();
    db.execute_cypher("CREATE (n:Person {name: 'carol'})")
        .unwrap();

    // Drain buffer should have entries (or already drained).
    // Wait for drain thread to process (default 100ms interval).
    std::thread::sleep(std::time::Duration::from_millis(250));

    // After drain, data should still be readable.
    let results = db
        .execute_cypher("MATCH (n:Person) RETURN n.name ORDER BY n.name")
        .unwrap();
    assert_eq!(results.len(), 2);
}

/// j:cache write is visible immediately and survives drain cycle.
#[test]
fn cache_write_locally_visible() {
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    db.set_write_concern(WriteConcern::cache());

    let results = db
        .execute_cypher("CREATE (n:Device {mac: 'aa:bb:cc'}) RETURN n.mac")
        .unwrap();
    assert_eq!(results.len(), 1);

    let read = db.execute_cypher("MATCH (n:Device) RETURN n.mac").unwrap();
    assert_eq!(read.len(), 1);
    assert_eq!(read[0].get("n.mac").unwrap().as_str(), Some("aa:bb:cc"));
}

/// Switching write concern mid-session: j:memory then w:majority.
#[test]
fn switch_write_concern_mid_session() {
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();

    // Write with j:memory.
    db.set_write_concern(WriteConcern::memory());
    db.execute_cypher("CREATE (n:Sensor {id: 1})").unwrap();

    // Switch to w:majority.
    db.set_write_concern(WriteConcern::majority());
    db.execute_cypher("CREATE (n:Sensor {id: 2})").unwrap();

    // Both should be visible.
    let results = db.execute_cypher("MATCH (n:Sensor) RETURN n.id").unwrap();
    assert_eq!(results.len(), 2);
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

/// j:memory with edges: edge create uses merge operators, should drain correctly.
#[test]
fn memory_write_with_edges() {
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    db.set_write_concern(WriteConcern::memory());

    db.execute_cypher("CREATE (a:Person {name: 'alice'})-[:KNOWS]->(b:Person {name: 'bob'})")
        .unwrap();

    // Read back edge.
    let results = db
        .execute_cypher("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name")
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].get("a.name").unwrap().as_str(), Some("alice"));
    assert_eq!(results[0].get("b.name").unwrap().as_str(), Some("bob"));
}
