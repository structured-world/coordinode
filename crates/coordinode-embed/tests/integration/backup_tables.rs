//! What a physical backup holds of `STORAGE COLUMNAR` tables changed while
//! it is taken: a table's rows, its creation and its drop each land in steps
//! no single commit gathers, and a backup holds each whole or not at all,
//! or is refused when the store itself was left half changed.
//!
//! The races run on a store without a journal of its own (a group member's
//! storage is one): with a journal, the backup's copy of it replays what
//! the trees taken lack, and a step left out of the trees comes back.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::Path;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_embed::Database;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::local_pipeline::OwnedLocalProposalPipeline;
use coordinode_storage::error::StorageError;

/// How long a paused step waits for the backup it races to finish: the
/// backup taken in the window finishes well inside it; one that waits for
/// the step lets it go on once it passes.
const PAUSE: Duration = Duration::from_secs(1);

/// A database at `dir` over a store with no journal of its own.
fn journal_less(dir: &Path) -> Arc<Database> {
    let oracle = Arc::new(TimestampOracle::new());
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = Arc::new(
        StorageEngine::open_with_oracle(&config, Arc::clone(&oracle)).expect("open the store"),
    );
    assert!(!engine.has_journal(), "a store with no journal of its own");
    let pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> =
        Arc::new(OwnedLocalProposalPipeline::new(&engine));
    Arc::new(Database::from_engine(dir, engine, oracle, pipeline).expect("open"))
}

/// Run `query` on `db` from any thread.
fn run(db: &Database, query: &str) -> Result<(), coordinode_embed::DatabaseError> {
    db.execute_cypher_shared(query, None, None, None, None)
        .map(drop)
}

/// The tables the catalog of the database at `dir` declares and the table
/// trees it holds agree: no table without its tree, no tree without its
/// table. Returns the rows of `table`, none when it is not there.
fn open_consistent(dir: &Path, table: &str) -> usize {
    let mut db = Database::open(dir).expect("the backup opens");
    let declared = db
        .label_schemas()
        .expect("catalog")
        .into_iter()
        .any(|schema| schema.name == table && schema.is_columnar());
    let tree = db.engine().columnar_table_tree(table).is_some();
    assert_eq!(
        declared, tree,
        "the backup holds {table} declared {declared}, its tree {tree}"
    );
    if !declared {
        return 0;
    }
    db.execute_cypher(&format!("MATCH (t:{table}) RETURN t.id AS id"))
        .expect("read the table")
        .len()
}

/// Take a backup of `db` into `target` on its own thread once `started`
/// tells a step is paused, and tell `finished` when it is done.
fn backup_racing(
    db: &Arc<Database>,
    target: std::path::PathBuf,
    started: mpsc::Receiver<()>,
    finished: mpsc::Sender<()>,
) -> std::thread::JoinHandle<Result<std::path::PathBuf, coordinode_embed::DatabaseError>> {
    let db = Arc::clone(db);
    std::thread::spawn(move || {
        started
            .recv_timeout(Duration::from_secs(10))
            .expect("the step paused");
        let taken = db.physical_backup(&target).map(|_| target);
        // The paused step may have gone on already.
        let _told = finished.send(()).is_ok();
        taken
    })
}

/// A backup taken while a drop has removed the table's tree and not yet
/// committed its catalog change waits for the drop: it holds the table with
/// its rows or not at all. Before, it held the table declared without a
/// tree.
#[test]
fn a_backup_waits_for_a_drop_between_its_tree_and_its_catalog() {
    let dir = tempfile::tempdir().expect("tempdir");
    let backups = tempfile::tempdir().expect("tempdir");
    let db = journal_less(dir.path());
    run(
        &db,
        "CREATE TABLE Trade (id BIGINT PRIMARY KEY, qty INT) STORAGE COLUMNAR",
    )
    .expect("create");
    run(&db, "CREATE (:Trade {id: 1, qty: 10})").expect("row");

    let (paused, started) = mpsc::channel();
    let (finished, backup_done) = mpsc::channel::<()>();
    db.engine().set_table_dropped_hook(Some(Box::new(move || {
        let _told = paused.send(()).is_ok();
        let _done = backup_done.recv_timeout(PAUSE).is_ok();
    })));
    let backup = backup_racing(&db, backups.path().join("b"), started, finished);
    run(&db, "DROP TABLE Trade").expect("dropped");
    let target = backup.join().expect("backup").expect("taken");
    db.engine().set_table_dropped_hook(None);

    open_consistent(&target, "Trade");
}

/// A backup taken while a statement writing several rows of a table is
/// between them waits for the statement: it holds all of its rows or none.
/// Before, it held the rows written so far.
#[test]
fn a_backup_holds_a_statement_writing_rows_whole() {
    let dir = tempfile::tempdir().expect("tempdir");
    let backups = tempfile::tempdir().expect("tempdir");
    let db = journal_less(dir.path());
    run(
        &db,
        "CREATE TABLE Trade (id BIGINT PRIMARY KEY, qty INT) STORAGE COLUMNAR",
    )
    .expect("create");

    let (paused, started) = mpsc::channel();
    let (finished, backup_done) = mpsc::channel::<()>();
    let mut first = true;
    db.engine().set_columnar_row_hook(Some(Box::new(move || {
        if !std::mem::replace(&mut first, false) {
            return;
        }
        let _told = paused.send(()).is_ok();
        let _done = backup_done.recv_timeout(PAUSE).is_ok();
    })));
    let backup = backup_racing(&db, backups.path().join("b"), started, finished);
    run(
        &db,
        "UNWIND [1, 2, 3] AS i CREATE (:Trade {id: i, qty: i * 10})",
    )
    .expect("written");
    let target = backup.join().expect("backup").expect("taken");
    db.engine().set_columnar_row_hook(None);

    let rows = open_consistent(&target, "Trade");
    assert!(
        rows == 0 || rows == 3,
        "the statement is in the backup whole or not at all: {rows} rows"
    );
}

/// A statement that starts writing a table's rows after a backup pinned its
/// trees is not in the backup, though its first row reached the journal
/// before the backup copied the journal: the journal copy ends where the
/// trees were taken. Every row the backup holds also answers by its key.
/// Before, the copy of the journal ran on past the pins and replayed the
/// statement's first row alone.
#[test]
fn a_backup_journal_ends_at_its_pins() {
    let dir = tempfile::tempdir().expect("tempdir");
    let backups = tempfile::tempdir().expect("tempdir");
    let db = Arc::new(Database::open(dir.path()).expect("open"));
    assert!(db.engine().has_journal(), "a store with its own journal");
    run(
        &db,
        "CREATE TABLE Trade (id BIGINT PRIMARY KEY, qty INT) STORAGE COLUMNAR",
    )
    .expect("create");

    let (pinned, trees_pinned) = mpsc::channel();
    let (landed, first_landed) = mpsc::channel();
    let (finished, backup_done) = mpsc::channel::<()>();
    db.engine().set_trees_pinned_hook(Some(Box::new(move || {
        let _told = pinned.send(()).is_ok();
        // The statement's first row lands before the backup goes on.
        let _landed = first_landed.recv_timeout(PAUSE).is_ok();
    })));
    let mut first = true;
    db.engine().set_columnar_row_hook(Some(Box::new(move || {
        if !std::mem::replace(&mut first, false) {
            return;
        }
        let _told = landed.send(()).is_ok();
        let _done = backup_done.recv_timeout(PAUSE).is_ok();
    })));
    let statement = {
        let db = Arc::clone(&db);
        std::thread::spawn(move || {
            trees_pinned
                .recv_timeout(Duration::from_secs(10))
                .expect("the backup pinned its trees");
            run(
                &db,
                "UNWIND [1, 2, 3] AS i CREATE (:Trade {id: i, qty: i * 10})",
            )
        })
    };
    let target = backups.path().join("b");
    db.physical_backup(&target).expect("taken");
    let _told = finished.send(()).is_ok();
    statement.join().expect("statement").expect("written");
    db.engine().set_trees_pinned_hook(None);
    db.engine().set_columnar_row_hook(None);

    let rows = open_consistent(&target, "Trade");
    assert_eq!(rows, 0, "the statement started after the pins");
    let mut copy = Database::open(&target).expect("the backup opens");
    for id in 1..=3 {
        let by_key = copy
            .execute_cypher(&format!("MATCH (t:Trade {{id: {id}}}) RETURN t.id AS id"))
            .expect("read by key")
            .len();
        assert_eq!(by_key, 0, "row {id} is not in the backup by its key");
    }
}

/// A drop whose catalog change never commits (its transaction rolled back)
/// leaves the table declared without its tree. A backup of that store is
/// refused, the catalog and the trees named; a row written to the table is
/// refused too rather than written to a new tree under its name. Before, the
/// backup was taken.
#[test]
fn a_backup_of_a_table_a_rolled_back_drop_left_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let backups = tempfile::tempdir().expect("tempdir");
    let db = journal_less(dir.path());
    run(
        &db,
        "CREATE TABLE Trade (id BIGINT PRIMARY KEY, qty INT) STORAGE COLUMNAR",
    )
    .expect("create");
    run(&db, "CREATE (:Trade {id: 1, qty: 10})").expect("row");

    let txn = db.begin_transaction().expect("begin");
    db.execute_in_transaction(txn, "DROP TABLE Trade", None)
        .expect("drop in the transaction");
    db.rollback_transaction(txn).expect("rollback");

    let refused = db
        .physical_backup(&backups.path().join("b"))
        .expect_err("a store whose catalog and trees disagree is not backed up");
    assert!(
        matches!(
            refused.storage_cause(),
            Some(StorageError::CatalogMismatch { declared_only, trees_only })
                if declared_only == &["Trade".to_owned()] && trees_only.is_empty()
        ),
        "refused naming the table declared without its tree: {refused:?}"
    );
    assert!(
        !backups.path().join("b").exists(),
        "the refused backup left nothing"
    );
    let row = run(&db, "CREATE (:Trade {id: 2, qty: 20})")
        .expect_err("no new tree stands in for the dropped one");
    assert!(
        matches!(
            row.storage_cause(),
            Some(StorageError::TableTreeMissing { table }) if table == "Trade"
        ),
        "{row:?}"
    );
}

/// A transaction that read a table before it was dropped and created again
/// under its name cannot write the old table's row into the new one: the
/// row is refused before it is written, and the new table and a backup of
/// it hold only the new table's rows. Before, the row landed in the new
/// table's tree, and stayed there when the transaction's commit failed.
#[test]
fn a_transaction_never_writes_into_a_table_created_after_it_read() {
    let dir = tempfile::tempdir().expect("tempdir");
    let backups = tempfile::tempdir().expect("tempdir");
    let db = journal_less(dir.path());
    run(
        &db,
        "CREATE TABLE Trade (id BIGINT PRIMARY KEY, qty INT) STORAGE COLUMNAR",
    )
    .expect("create");

    let old = db.begin_transaction().expect("begin");
    db.execute_in_transaction(old, "MATCH (t:Trade) RETURN t.id AS id", None)
        .expect("the old transaction reads the table");
    run(&db, "DROP TABLE Trade").expect("drop");
    run(
        &db,
        "CREATE TABLE Trade (id BIGINT PRIMARY KEY, qty INT) STORAGE COLUMNAR",
    )
    .expect("create again");
    run(&db, "CREATE (:Trade {id: 7, qty: 70})").expect("row of the new table");

    let stale = db
        .execute_in_transaction(old, "CREATE (:Trade {id: 1, qty: 10})", None)
        .expect_err("the old transaction's row is refused");
    assert!(
        matches!(
            stale.storage_cause(),
            Some(StorageError::TableCreatedSinceRead { table }) if table == "Trade"
        ),
        "{stale:?}"
    );
    // The statement's failure ended the transaction; nothing to roll back.
    let _ended = db.rollback_transaction(old).is_ok();

    let rows = db
        .engine()
        .columnar_scan("Trade", u64::MAX)
        .expect("scan the new table");
    assert_eq!(rows.len(), 1, "the new table holds its own row only");
    let target = backups.path().join("b");
    db.physical_backup(&target).expect("backup");
    drop(db);
    assert_eq!(open_consistent(&target, "Trade"), 1);
}

/// A transaction's statement writes rows into a table the transaction
/// created itself, though the table's tree is newer than its snapshot.
#[test]
fn a_transaction_writes_into_a_table_it_created() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = journal_less(dir.path());
    let txn = db.begin_transaction().expect("begin");
    db.execute_in_transaction(
        txn,
        "CREATE TABLE Trade (id BIGINT PRIMARY KEY, qty INT) STORAGE COLUMNAR",
        None,
    )
    .expect("create in the transaction");
    db.execute_in_transaction(txn, "CREATE (:Trade {id: 1, qty: 10})", None)
        .expect("a row of the table it created");
    assert_eq!(
        db.engine()
            .columnar_scan("Trade", u64::MAX)
            .expect("scan")
            .len(),
        1,
        "the row is in the table it created"
    );
    let _ended = db.rollback_transaction(txn).is_ok();
}

/// A table dropped and created again under its name between two backups is
/// in the second one as the table created last, with its own rows only.
#[test]
fn a_backup_holds_a_table_created_again_as_the_new_one() {
    let dir = tempfile::tempdir().expect("tempdir");
    let backups = tempfile::tempdir().expect("tempdir");
    let db = journal_less(dir.path());
    run(
        &db,
        "CREATE TABLE Trade (id BIGINT PRIMARY KEY, qty INT) STORAGE COLUMNAR",
    )
    .expect("create");
    run(&db, "CREATE (:Trade {id: 1, qty: 10})").expect("row");
    let first = backups.path().join("first");
    db.physical_backup(&first).expect("first backup");
    run(&db, "DROP TABLE Trade").expect("drop");
    run(
        &db,
        "CREATE TABLE Trade (id BIGINT PRIMARY KEY, qty INT) STORAGE COLUMNAR",
    )
    .expect("create again");
    run(&db, "CREATE (:Trade {id: 7, qty: 70})").expect("row of the new table");
    run(&db, "CREATE (:Trade {id: 8, qty: 80})").expect("row of the new table");
    let second = backups.path().join("second");
    db.physical_backup(&second).expect("second backup");
    drop(db);

    assert_eq!(open_consistent(&first, "Trade"), 1);
    assert_eq!(open_consistent(&second, "Trade"), 2, "the new table's rows");
}

/// A statement writing many rows of a table holds the set of trees once, not
/// once a row, and a read holds it never.
#[test]
fn a_table_statement_holds_the_set_of_trees_once() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = journal_less(dir.path());
    run(
        &db,
        "CREATE TABLE Trade (id BIGINT PRIMARY KEY, qty INT) STORAGE COLUMNAR",
    )
    .expect("create");
    let before = db.engine().tree_set_holds_granted();
    run(
        &db,
        "UNWIND range(1, 200) AS i CREATE (:Trade {id: i, qty: i})",
    )
    .expect("rows");
    assert_eq!(
        db.engine().tree_set_holds_granted() - before,
        1,
        "one hold for the statement's 200 rows"
    );
    let before = db.engine().tree_set_holds_granted();
    let read = db
        .execute_cypher_shared("MATCH (t:Trade) RETURN t.id AS id", None, None, None, None)
        .expect("read");
    assert_eq!(read.rows.len(), 200);
    assert_eq!(
        db.engine().tree_set_holds_granted(),
        before,
        "a read holds nothing"
    );
}
