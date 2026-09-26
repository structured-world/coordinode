use super::*;

/// A transaction the client walked away from must be rolled back by the
/// reaper alone, with no later `begin` to trigger a sweep: it pins a snapshot
/// and buffered writes for as long as it stays open.
#[tokio::test(flavor = "multi_thread")]
async fn an_abandoned_transaction_is_rolled_back_without_further_traffic() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");
    let tx = db.begin_transaction();
    db.execute_in_transaction(tx, "CREATE (n:Abandoned {k: 1})", None)
        .expect("buffered write");
    let database = Arc::new(RwLock::new(db));

    let idle = Duration::from_millis(50);
    let registry = Arc::new(SessionRegistry::new(idle));
    let reaper = spawn(
        Arc::clone(&database),
        registry,
        idle,
        Duration::from_millis(20),
    );

    tokio::time::sleep(Duration::from_millis(500)).await;
    reaper.abort();

    assert!(
        database.read().commit_transaction(tx).is_err(),
        "the idle transaction is still open after the reaper ran"
    );
}
