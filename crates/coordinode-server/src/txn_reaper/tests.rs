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
        Arc::new(Notify::new()),
        Duration::from_millis(20),
    );

    tokio::time::sleep(Duration::from_millis(500)).await;
    reaper.abort();

    assert!(
        database.read().commit_transaction(tx).is_err(),
        "the idle transaction is still open after the reaper ran"
    );
}

/// With no transaction open the reaper sleeps until one begins; the begin hook
/// wakes it, and the transaction opened then is still reaped once idle.
#[tokio::test(flavor = "multi_thread")]
async fn a_transaction_begun_while_the_reaper_sleeps_is_reaped() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    let begun = Arc::new(Notify::new());
    let hook = Arc::clone(&begun);
    db.set_interactive_begun_hook(Arc::new(move || hook.notify_one()));
    let database = Arc::new(RwLock::new(db));

    let idle = Duration::from_millis(50);
    let reaper = spawn(
        Arc::clone(&database),
        Arc::new(SessionRegistry::new(idle)),
        idle,
        begun,
        Duration::from_millis(20),
    );
    // The reaper has found nothing open and is waiting for a begin.
    tokio::time::sleep(Duration::from_millis(100)).await;
    let tx = database.read().begin_transaction();

    tokio::time::sleep(Duration::from_millis(500)).await;
    reaper.abort();

    assert!(
        database.read().commit_transaction(tx).is_err(),
        "the transaction begun while the reaper slept is still open"
    );
}
