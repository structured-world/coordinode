use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, Tier};

use super::*;

/// The CA the members of these tests are issued their identities by.
static PKI: std::sync::LazyLock<coordinode_test_fixtures::TestPki> =
    std::sync::LazyLock::new(coordinode_test_fixtures::TestPki::new);

fn storage(dir: &std::path::Path) -> StorageConfig {
    StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )])
}

/// A member alone needs no node certificate: it is its group's only voter
/// from its first open, leads, takes writes through its replicated pipeline,
/// and serves no network surface.
#[tokio::test(flavor = "multi_thread")]
async fn a_member_alone_leads_and_takes_writes() {
    let dir = tempfile::tempdir().unwrap();
    let db = ReplicatedDatabase::open(&storage(dir.path()), MemberOptions::new(1), None)
        .await
        .unwrap();
    assert_eq!(db.leader(), Some(1));
    assert!(db.local_addr().is_none(), "no listener without peers");
    assert!(db.identity().is_none(), "no node identity without peers");

    db.database()
        .write()
        .execute_cypher("CREATE (:Thing {k: 1})")
        .unwrap();
    let rows = db
        .database()
        .write()
        .execute_cypher("MATCH (t:Thing) RETURN t.k")
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert!(db.version_report().read_only.is_none(), "it serves writes");
    db.shutdown().await.unwrap();
}

/// A member with peers and no address to serve them on is refused before
/// its store is opened: nothing would answer the group.
#[tokio::test(flavor = "multi_thread")]
async fn a_member_with_peers_and_no_listener_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let mut options = MemberOptions::new(1);
    options.peering = Some(Peering {
        security: PKI.context(1),
        advertise: "127.0.0.1:1".to_string(),
        founding: Founding::Form { deployment: None },
        timeouts: PeerTimeouts::default(),
    });
    let refused = ReplicatedDatabase::open(&storage(dir.path()), options, None).await;
    assert!(
        matches!(refused, Err(ReplicationError::Listener { .. })),
        "refused for want of a listener, got {:?}",
        refused.as_ref().err()
    );
    // The store was never opened: the same directory opens at once.
    let reopened = ReplicatedDatabase::open(&storage(dir.path()), MemberOptions::new(1), None)
        .await
        .unwrap();
    reopened.shutdown().await.unwrap();
}

/// An open, non-replicated database becomes a member of a group on the same
/// store while the host runs: what it committed, through its journal and not
/// yet flushed, is all there under the replicated handle, which leads and
/// takes the next write through the log.
#[tokio::test(flavor = "multi_thread")]
async fn a_running_database_becomes_a_member_on_the_same_store() {
    let dir = tempfile::tempdir().unwrap();
    let mut alone = Database::open(dir.path()).unwrap();
    alone
        .execute_cypher("CREATE (:Before {k: 1}), (:Before {k: 2})")
        .unwrap();

    let db =
        ReplicatedDatabase::from_database(alone, &storage(dir.path()), MemberOptions::new(1), None)
            .await
            .unwrap();
    assert_eq!(db.leader(), Some(1));
    let before = db
        .database()
        .write()
        .execute_cypher("MATCH (n:Before) RETURN n")
        .unwrap();
    assert_eq!(before.len(), 2, "what the database held is all there");
    db.database()
        .write()
        .execute_cypher("CREATE (:After {k: 1})")
        .unwrap();
    db.shutdown().await.unwrap();

    // Reopened as a member, the store holds both, and its journal is gone:
    // the consensus log is what it opens from now.
    let db = ReplicatedDatabase::open(&storage(dir.path()), MemberOptions::new(1), None)
        .await
        .unwrap();
    for label in ["Before", "After"] {
        let rows = db
            .database()
            .write()
            .execute_cypher(&format!("MATCH (n:{label}) RETURN n"))
            .unwrap();
        assert!(!rows.is_empty(), "{label} survives the reopen");
    }
    db.shutdown().await.unwrap();
}

/// A member that has shut down has let go of its store: the host reopens the
/// same directory at once, every time, and finds what it wrote.
#[tokio::test(flavor = "multi_thread")]
async fn a_member_that_shut_down_has_released_its_store() {
    let dir = tempfile::tempdir().unwrap();
    for round in 0..8 {
        let db = ReplicatedDatabase::open(&storage(dir.path()), MemberOptions::new(1), None)
            .await
            .unwrap_or_else(|e| panic!("round {round}: the store is still held: {e}"));
        db.database()
            .write()
            .execute_cypher(&format!("CREATE (:Round {{k: {round}}})"))
            .unwrap();
        let rows = db
            .database()
            .write()
            .execute_cypher("MATCH (r:Round) RETURN r")
            .unwrap();
        assert_eq!(rows.len(), round + 1, "round {round}");
        db.shutdown().await.unwrap();
    }
}

/// A store written without replication, its journal kept in segments of two
/// entries, holding five nodes written one statement each.
fn standalone_store(dir: &std::path::Path) -> StorageConfig {
    let mut config = storage(dir);
    config.oplog_segment_max_entries = 2;
    let mut alone = Database::open_with_config(config.clone()).unwrap();
    for k in 0..5 {
        alone
            .execute_cypher(&format!("CREATE (:Before {{k: {k}}})"))
            .unwrap();
    }
    config
}

/// How many `Before` nodes `db` holds, by a scan and by its label count: an
/// entry applied twice shows in the count.
fn befores(db: &Database) -> (usize, u64) {
    let rows = db
        .execute_cypher_shared("MATCH (n:Before) RETURN n", None, None, None, None)
        .unwrap()
        .rows
        .len();
    let counted = db
        .label_counts()
        .unwrap()
        .get("Before")
        .copied()
        .unwrap_or(0);
    (rows, counted)
}

/// A retirement that finished before the host first opened the store as a
/// member leaves a store outside any group still: the host may go back to
/// opening it without replication, keep writing, and later grow it.
#[tokio::test(flavor = "multi_thread")]
async fn a_store_retired_but_never_opened_as_a_member_is_still_its_own() {
    let dir = tempfile::tempdir().unwrap();
    let config = standalone_store(dir.path());
    assert!(StorageEngine::retire_embedded_journal(&config).unwrap());

    let reopened = config.clone();
    tokio::task::spawn_blocking(move || {
        let mut alone = Database::open_with_config(reopened).unwrap();
        assert_eq!(befores(&alone), (5, 5));
        alone.execute_cypher("CREATE (:Before {k: 5})").unwrap();
    })
    .await
    .unwrap();

    let db = ReplicatedDatabase::open(&config, MemberOptions::new(1), None)
        .await
        .unwrap();
    assert_eq!(befores(&db.database().read()), (6, 6));
    db.shutdown().await.unwrap();
}

/// What the old path answered is kept across the transition and what it did
/// not commit is not: a j:memory write already answered is in the member's
/// store once, a transaction still open on the handle is discarded with it,
/// and the member takes writes through its log at once.
#[tokio::test(flavor = "multi_thread")]
async fn the_transition_keeps_answered_writes_and_drops_open_transactions() {
    use coordinode_core::txn::write_concern::WriteConcern;

    let dir = tempfile::tempdir().unwrap();
    let mut alone = Database::open(dir.path()).unwrap();
    alone.set_write_concern(WriteConcern::memory());
    alone.execute_cypher("CREATE (:Before {k: 0})").unwrap();
    alone.set_write_concern(WriteConcern::default());
    let open = alone.begin_transaction();
    alone
        .execute_in_transaction(open, "CREATE (:Unsubmitted {k: 1})", None)
        .unwrap();

    let db =
        ReplicatedDatabase::from_database(alone, &storage(dir.path()), MemberOptions::new(1), None)
            .await
            .unwrap();
    assert_eq!(befores(&db.database().read()), (1, 1));
    let unsubmitted = db
        .database()
        .write()
        .execute_cypher("MATCH (n:Unsubmitted) RETURN n")
        .unwrap();
    assert!(
        unsubmitted.is_empty(),
        "an uncommitted transaction survived"
    );
    db.database()
        .write()
        .execute_cypher("CREATE (:Before {k: 1})")
        .unwrap();
    db.shutdown().await.unwrap();

    let db = ReplicatedDatabase::open(&storage(dir.path()), MemberOptions::new(1), None)
        .await
        .unwrap();
    assert_eq!(befores(&db.database().read()), (2, 2));
    db.shutdown().await.unwrap();
}

/// The timing a member was opened with is readable from the handle.
#[tokio::test(flavor = "multi_thread")]
async fn the_timing_profile_is_visible_on_the_handle() {
    let dir = tempfile::tempdir().unwrap();
    let mut options = MemberOptions::new(1);
    options.timing = ElectionTiming::PERSONAL_FLEET;
    let db = ReplicatedDatabase::open(&storage(dir.path()), options, None)
        .await
        .unwrap();
    assert_eq!(db.timing(), ElectionTiming::PERSONAL_FLEET);
    db.shutdown().await.unwrap();
}
