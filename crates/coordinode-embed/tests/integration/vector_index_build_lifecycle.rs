//! The lifecycle of an asynchronous vector-index build: who owns it, when it
//! stops, and what it is allowed to write.
//!
//! A build outliving the statement that started it used to write the index
//! definition from a detached thread — progress checkpoints every thousand
//! nodes, then a terminal state. Any later statement touching that index drew
//! its read snapshot before those writes landed, so conflict detection read
//! them as a concurrent transaction and rejected the statement. Both tests
//! here exercise that boundary from the two directions it can be crossed.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_embed::Database;

/// Back-to-back CREATE / DROP of the same index must never collide with its
/// own build. The loop is what makes it a test: the failure is a race, and a
/// single pass hits it only when the machine is loaded enough for the build's
/// write to slip past the next statement's snapshot.
#[test]
fn create_drop_cycle_never_conflicts() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open db");

    for i in 0..500 {
        let name = format!("idx_{i}");
        db.execute_cypher(&format!(
            "CREATE VECTOR INDEX {name} ON :Item(embedding) OPTIONS {{metric: \"cosine\"}}"
        ))
        .unwrap_or_else(|e| panic!("create #{i}: {e:?}"));
        db.execute_cypher(&format!("DROP VECTOR INDEX {name}"))
            .unwrap_or_else(|e| panic!("drop #{i}: {e:?}"));
    }
}

/// Dropping an index whose build is still running must cancel that build, and
/// the cancelled build must not resurrect the definition it was working on:
/// reopening the database has to find the index gone.
#[test]
fn drop_cancels_a_running_build_and_it_stays_dropped() {
    let dir = tempfile::tempdir().expect("tempdir");
    {
        let mut db = Database::open(dir.path()).expect("open db");

        // Enough vectors that the backfill is still scanning when the DROP
        // arrives — a build that finished first would prove nothing.
        db.execute_cypher(
            "UNWIND range(1, 4000) AS i \
             CREATE (:Item {embedding: [toFloat(i), 1.0, 2.0]})",
        )
        .expect("seed vectors");

        db.execute_cypher(
            "CREATE VECTOR INDEX live_build ON :Item(embedding) OPTIONS {metric: \"cosine\"}",
        )
        .expect("create vector index");

        assert!(
            !db.index_builds().is_empty(),
            "the build should still be running for this test to mean anything"
        );

        db.execute_cypher("DROP VECTOR INDEX live_build")
            .expect("drop while building");

        assert!(
            db.index_builds().is_empty(),
            "DROP returned while its index's build was still running"
        );

        // A cancelled build that was going to write anyway would do it within
        // this window; the reopen below is what catches it if it did.
        std::thread::sleep(std::time::Duration::from_millis(300));
    }

    let mut reopened = Database::open(dir.path()).expect("reopen db");
    // Re-creating under the same name proves the old definition is really
    // gone: a resurrected one would still occupy (label, property).
    reopened
        .execute_cypher(
            "CREATE VECTOR INDEX live_build ON :Item(embedding) OPTIONS {metric: \"cosine\"}",
        )
        .expect("re-create after the cancelled build");
}
