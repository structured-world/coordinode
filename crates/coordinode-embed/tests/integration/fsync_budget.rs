//! How many device flushes common work costs. A flush is owed only where a
//! crash would otherwise break a promise: a commit no answer depends on rides
//! the next one that does.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use coordinode_embed::Database;
use tempfile::tempdir;

/// A store whose journal flusher stays asleep, so every counted sync is one a
/// statement asked for.
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

fn syncs(db: &Database) -> u64 {
    db.engine().journal_syncs().expect("a store with a journal")
}

/// Let the background work the setup started finish and make it durable, so
/// the syncs counted afterwards are the statement's own: the label lookup's
/// build, and the spare NodeId lease the first CREATE asks for (a durable
/// grant, synced when it lands) would otherwise add one when the machine is
/// slow.
fn settle(db: &Database) {
    // The first CREATE takes the lease (0, 10000] and asks for the spare
    // (10000, 20000] in the background.
    const SPARE_LEASE_CEILING: u64 = 20_000;
    use coordinode_modality::{IndexStore as _, LocalIndexStore};
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        let lookup = LocalIndexStore::new(db.engine())
            .label_lookup()
            .expect("read")
            .expect("every store has its label lookup");
        let spare = coordinode_storage::engine::metadata::node_lease_ceiling(db.engine())
            .expect("read the lease records");
        if lookup.state == coordinode_modality::IndexState::Ready && spare >= SPARE_LEASE_CEILING {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the setup's background work never finished: label lookup {:?}, lease ceiling {spare}",
            lookup.state
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    db.engine().sync_applied().expect("sync");
}

/// A journaled write that registers a new property name syncs the journal
/// once: the registration is logged ahead of the data that uses it, and the
/// data's sync covers both.
#[test]
fn a_write_with_a_new_property_syncs_the_journal_once() {
    let dir = tempdir().unwrap();
    let mut db = open_flusher_held(dir.path());
    db.execute_cypher("CREATE (:Seed {k: 1})").unwrap();
    settle(&db);
    let before = syncs(&db);
    db.execute_cypher("CREATE (:Seed {fresh_name: 1})").unwrap();
    assert_eq!(syncs(&db) - before, 1);
}

/// Creating a B-tree index over existing data syncs the journal once, for
/// the statement's own answer: the build's pages and bookkeeping promise
/// nothing on their own.
#[test]
fn creating_an_index_syncs_the_journal_once() {
    let dir = tempdir().unwrap();
    let mut db = open_flusher_held(dir.path());
    db.execute_cypher("CREATE (:User {email: 'a@x'}), (:User {email: 'b@x'})")
        .unwrap();
    settle(&db);
    let before = syncs(&db);
    db.execute_cypher("CREATE INDEX user_email ON :User(email)")
        .unwrap();
    assert_eq!(syncs(&db) - before, 1);
}

/// A repeated CREATE that finds the object answers from durable records:
/// a client retrying after a lost reply reads `exists` (or the IF NOT EXISTS
/// row) as "created", so that answer waits for the sync a weaker creating
/// statement skipped, as a power cut would otherwise take the object back.
#[test]
fn an_answer_that_an_object_exists_is_durable() {
    use coordinode_core::txn::write_concern::WriteConcern;
    let statements = [
        (
            "CREATE INDEX user_email ON :User(email)",
            "CREATE INDEX user_email ON :User(email)",
        ),
        (
            "CREATE CONSTRAINT user_key FOR (u:User) REQUIRE u.email IS UNIQUE",
            "CREATE CONSTRAINT user_key IF NOT EXISTS FOR (u:User) REQUIRE u.email IS UNIQUE",
        ),
        (
            "CREATE CONSTRAINT user_key FOR (u:User) REQUIRE u.email IS UNIQUE",
            "CREATE CONSTRAINT user_key FOR (u:User) REQUIRE u.email IS UNIQUE",
        ),
    ];
    for (create, repeat) in statements {
        let dir = tempdir().unwrap();
        let mut db = open_flusher_held(dir.path());
        db.execute_cypher("CREATE (:User {email: 'a@x'})").unwrap();
        db.set_write_concern(WriteConcern::cache());
        db.execute_cypher(create).unwrap();
        let now = db.engine().current_seqno();
        assert!(
            db.engine().durable_view(now) < now,
            "setup: `{create}` under a cache concern synced"
        );
        db.set_write_concern(WriteConcern::majority());
        let _answer = db.execute_cypher(repeat);
        let now = db.engine().current_seqno();
        assert_eq!(
            db.engine().durable_view(now),
            now,
            "`{repeat}` answered about an object a power cut could still take"
        );
    }
}

/// A statement that answers an index ready answers from a durable
/// publication: when CREATE INDEX returns, nothing it wrote, the build's
/// published record included, waits unsynced in the journal, so a power cut
/// right after cannot take the index back to building.
#[test]
fn an_index_answered_ready_is_published_durably() {
    let dir = tempdir().unwrap();
    let mut db = open_flusher_held(dir.path());
    db.execute_cypher("CREATE (:User {email: 'a@x'}), (:User {email: 'b@x'})")
        .unwrap();
    db.execute_cypher("CREATE INDEX user_email ON :User(email)")
        .unwrap();
    let engine = db.engine();
    let now = engine.current_seqno();
    assert_eq!(
        engine.durable_view(now),
        now,
        "the answer went out with part of the publication unsynced"
    );
}
