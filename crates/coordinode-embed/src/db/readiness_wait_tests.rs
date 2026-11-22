//! Readiness waits of the native calls: a causal read, schema readiness and
//! a rebuild. Zero checks at once, an earlier deadline bounds the wait, and a
//! wait whose end the monotonic clock cannot represent is refused, typed,
//! before anything waits.

use std::time::{Duration, Instant};

use coordinode_core::txn::read_concern::ReadConcern;

use super::{Database, DatabaseError, StatementOptions};

fn open() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");
    (db, dir)
}

/// The commit timestamp of one written node.
fn write(db: &Database) -> u64 {
    db.execute_cypher_shared("CREATE (:Waited {k: 1})", None, None, None, None)
        .expect("write")
        .commit_ts()
        .expect("a commit timestamp")
}

fn read_after(
    db: &Database,
    after_ts: u64,
    wait: Duration,
    deadline: Option<Instant>,
) -> Result<usize, DatabaseError> {
    db.execute_cypher_shared_with(
        "MATCH (n:Waited) RETURN n",
        None,
        None,
        &StatementOptions {
            read_concern: Some(ReadConcern {
                after_ts: Some(after_ts),
                wait: Some(wait),
                ..ReadConcern::majority()
            }),
            deadline,
            ..StatementOptions::default()
        },
    )
    .map(|result| result.rows.len())
}

/// A causal read whose wait no clock can end is refused as a setting, even
/// with its bound already met; under a statement deadline the same wait is
/// a finite one and the read runs. Zero reads at once.
#[test]
fn a_causal_read_refuses_a_wait_the_clock_cannot_end() {
    let (db, _dir) = open();
    let ts = write(&db);
    let refused = read_after(&db, ts, Duration::MAX, None);
    assert!(
        matches!(refused, Err(DatabaseError::SettingsRefused(_))),
        "{refused:?}"
    );
    let deadline = Instant::now() + Duration::from_secs(30);
    assert_eq!(
        read_after(&db, ts, Duration::MAX, Some(deadline)).expect("bounded by the deadline"),
        1
    );
    assert_eq!(
        read_after(&db, ts, Duration::ZERO, None).expect("met at once"),
        1
    );
}

/// Schema readiness refuses a wait the clock cannot end before it waits,
/// for an empty set of operations as for any; zero answers at once.
#[test]
fn schema_readiness_refuses_a_wait_the_clock_cannot_end() {
    let (db, _dir) = open();
    let refused = db.await_operations(&[], Duration::MAX);
    assert!(
        matches!(refused, Err(DatabaseError::SettingsRefused(_))),
        "{refused:?}"
    );
    assert!(
        db.await_operations(&[], Duration::ZERO)
            .expect("nothing to wait for")
            .is_empty()
    );
}

/// A rebuild asked to wait without an end the clock can represent is
/// refused before the rebuild starts: no build is published for it.
#[test]
fn a_rebuild_refuses_a_wait_the_clock_cannot_end_before_it_starts() {
    use coordinode_modality::IndexStore as _;
    let (db, _dir) = open();
    db.execute_cypher_shared(
        "CREATE INDEX waited_k ON :Waited(k)",
        None,
        None,
        None,
        None,
    )
    .expect("an index");
    let store = coordinode_modality::LocalIndexStore::new(&db.engine);
    let before = store.list_builds().expect("builds").len();
    let refused = db.reindex(
        &coordinode_query::index::IndexSelector::Name("waited_k".into()),
        Duration::MAX,
    );
    assert!(
        matches!(refused, Err(DatabaseError::SettingsRefused(_))),
        "{refused:?}"
    );
    assert_eq!(
        store.list_builds().expect("builds").len(),
        before,
        "no rebuild was started"
    );
}
