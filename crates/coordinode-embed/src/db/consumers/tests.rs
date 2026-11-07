use std::time::{Duration, Instant};

use coordinode_replicate::{
    ConsumerKind, ConsumerRetentionPolicy, InitialSeqno, RegistrationState, RegistryError,
    TerminalReason, TopologyScope, ValidatedRetentionBounds,
};

use super::*;

/// A log consumer registration starting at `initial_seqno`.
fn log_consumer(
    id: &str,
    initial_seqno: InitialSeqno,
    retention: ConsumerRetentionPolicy,
) -> ConsumerRegistration {
    ConsumerRegistration {
        consumer_id: id.to_string(),
        kind: ConsumerKind::OplogEvents,
        scope: TopologyScope::Shard(0),
        initial_seqno,
        retention,
    }
}

/// Write `n` nodes, each its own journal entry.
fn write(db: &mut Database, n: u32, tag: &str) {
    for i in 0..n {
        db.execute_cypher(&format!("CREATE (:Event {{tag: '{tag}', i: {i}}})"))
            .expect("write");
    }
}

/// Persist every write into tables and reclaim the journal they cover, as
/// a store at its footprint limit does: everything not kept for someone goes.
fn reclaim_journal(db: &Database) {
    db.persist().expect("persist");
    db.engine().reclaim_footprint();
}

/// The first index the journal still holds.
fn journal_start(db: &Database) -> u64 {
    db.engine()
        .oplog_read_since(0)
        .expect("read")
        .expect("journal")
        .first()
        .map_or(u64::MAX, |e| e.index)
}

/// A STRICT log consumer keeps the journal from its position through
/// reclamation, and its acknowledgement moves that position. (The journal is
/// released by whole segments; releasing acknowledged segments is checked
/// where segments are small, in the registry's own tests.)
#[test]
fn a_strict_consumer_keeps_the_journal_until_it_acknowledges() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    write(&mut db, 3, "before");
    let h = db
        .register_consumer(log_consumer(
            "outbox",
            InitialSeqno::FromNow,
            ConsumerRetentionPolicy::Strict,
        ))
        .expect("register");
    let from = db.consumer_position(&h).expect("position");
    write(&mut db, 5, "after");
    reclaim_journal(&db);
    assert!(
        journal_start(&db) <= from,
        "reclamation removed the journal from {from} the consumer had not acknowledged"
    );
    let head = db
        .engine()
        .journal_next_index()
        .expect("head")
        .expect("journal");
    db.acknowledge_consumer(&h, head).expect("acknowledge");
    assert_eq!(db.consumer_position(&h).expect("position"), head);
    // A lower acknowledgement, a stale retry, changes nothing.
    db.acknowledge_consumer(&h, from)
        .expect("stale acknowledge");
    assert_eq!(db.consumer_position(&h).expect("position"), head);
}

/// A registration and the journal it keeps survive a restart: the reopened
/// database takes the consumer up again by its incarnation, before any
/// reclamation can pass its position.
#[test]
fn a_registration_holds_the_journal_across_a_restart() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (from, incarnation) = {
        let mut db = Database::open(dir.path()).expect("open");
        let h = db
            .register_consumer(log_consumer(
                "sink",
                InitialSeqno::FromNow,
                ConsumerRetentionPolicy::Strict,
            ))
            .expect("register");
        write(&mut db, 4, "x");
        (db.consumer_position(&h).expect("position"), h.incarnation())
    };
    let db = Database::open(dir.path()).expect("reopen");
    reclaim_journal(&db);
    assert!(journal_start(&db) <= from, "the restart lost the hold");
    let h = db.resume_consumer("sink", incarnation).expect("resume");
    assert_eq!(db.consumer_position(&h).expect("position"), from);
}

/// Registering at a position the journal no longer holds is refused with
/// RetentionLost, naming where the journal starts now; nothing is
/// registered.
#[test]
fn registering_below_the_journal_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    write(&mut db, 4, "x");
    reclaim_journal(&db);
    let start = journal_start(&db);
    assert!(start > 0, "nothing was reclaimed");
    let refused = db.register_consumer(log_consumer(
        "late",
        InitialSeqno::At(0),
        ConsumerRetentionPolicy::Strict,
    ));
    assert!(
        matches!(
            refused,
            Err(DatabaseError::Consumer(RegistryError::RetentionLost {
                checkpoint: 0,
                ..
            }))
        ),
        "got {refused:?}"
    );
    assert!(db.consumers_list().expect("list").is_empty());
}

/// A cancelled registration ends with its reason, refuses its handle, and no
/// longer keeps the journal; the same id registers again as a new
/// incarnation.
#[test]
fn a_cancelled_consumer_releases_the_journal_and_refuses_its_handle() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    let h = db
        .register_consumer(log_consumer(
            "c",
            InitialSeqno::FromNow,
            ConsumerRetentionPolicy::Strict,
        ))
        .expect("register");
    let from = db.consumer_position(&h).expect("position");
    write(&mut db, 4, "x");
    db.cancel_consumer(h.clone()).expect("cancel");
    assert!(matches!(
        db.acknowledge_consumer(&h, from + 1),
        Err(DatabaseError::Consumer(RegistryError::Terminated {
            reason: TerminalReason::Cancelled,
            ..
        }))
    ));
    let until = Instant::now() + Duration::from_secs(10);
    loop {
        reclaim_journal(&db);
        if journal_start(&db) > from || Instant::now() > until {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(
        journal_start(&db) > from,
        "the cancelled consumer kept the journal"
    );
    let again = db
        .register_consumer(log_consumer(
            "c",
            InitialSeqno::FromNow,
            ConsumerRetentionPolicy::Strict,
        ))
        .expect("register again");
    assert_eq!(again.incarnation(), h.incarnation() + 1);
}

/// A BOUNDED consumer that stops heartbeating past its declared liveness
/// timeout is ended by the background service, with the reason, and its
/// journal is released; heartbeats keep a live one.
#[test]
fn a_silent_bounded_consumer_is_ended_by_the_service() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");
    let bounds = ValidatedRetentionBounds::new(u64::MAX, u64::MAX, Some(300)).expect("bounds");
    db.register_consumer(log_consumer(
        "silent",
        InitialSeqno::FromNow,
        ConsumerRetentionPolicy::Bounded(bounds),
    ))
    .expect("register silent");
    let alive = db
        .register_consumer(log_consumer(
            "alive",
            InitialSeqno::FromNow,
            ConsumerRetentionPolicy::Bounded(bounds),
        ))
        .expect("register alive");
    let state = |id: &str| {
        db.consumers_list()
            .expect("list")
            .into_iter()
            .find(|c| c.consumer_id == id)
            .map(|c| c.state)
    };
    let until = Instant::now() + Duration::from_secs(10);
    while matches!(state("silent"), Some(RegistrationState::Live)) && Instant::now() < until {
        db.consumer_heartbeat(&alive).expect("heartbeat");
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(
        matches!(
            state("silent"),
            Some(RegistrationState::Terminated {
                reason: TerminalReason::LivenessExpired,
                ..
            })
        ),
        "got {:?}",
        state("silent")
    );
    assert_eq!(state("alive"), Some(RegistrationState::Live));
}

/// An MVCC consumer holds collection of the history it reads at its
/// position.
#[test]
fn an_mvcc_consumer_holds_collection_at_its_position() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");
    let at = db.engine().gc_watermark();
    let h = db
        .register_consumer(ConsumerRegistration {
            kind: ConsumerKind::LsmStateDelta,
            ..log_consumer(
                "state",
                InitialSeqno::At(at),
                ConsumerRetentionPolicy::Strict,
            )
        })
        .expect("register");
    db.set_retention_window(Duration::from_millis(1));
    assert!(
        db.engine().gc_watermark() <= at,
        "collection passed the consumer's position"
    );
    assert_eq!(db.consumer_position(&h).expect("held"), at);
}

/// A physical backup carries the registrations with their incarnations and
/// positions: the database opened from it keeps the same history for them
/// and takes them up again, and an ended one stays ended there.
#[test]
fn a_backup_carries_the_registrations() {
    let dir = tempfile::tempdir().expect("tempdir");
    let target = tempfile::tempdir().expect("backup dir");
    let backup = target.path().join("copy");
    let (from, incarnation) = {
        let mut db = Database::open(dir.path()).expect("open");
        let h = db
            .register_consumer(log_consumer(
                "kept",
                InitialSeqno::FromNow,
                ConsumerRetentionPolicy::Strict,
            ))
            .expect("register kept");
        let gone = db
            .register_consumer(log_consumer(
                "gone",
                InitialSeqno::FromNow,
                ConsumerRetentionPolicy::Strict,
            ))
            .expect("register gone");
        db.cancel_consumer(gone).expect("cancel gone");
        write(&mut db, 3, "x");
        db.physical_backup(&backup).expect("backup");
        (db.consumer_position(&h).expect("position"), h.incarnation())
    };
    let restored = Database::open(&backup).expect("open the backup");
    let h = restored
        .resume_consumer("kept", incarnation)
        .expect("resume in the backup");
    assert_eq!(restored.consumer_position(&h).expect("position"), from);
    reclaim_journal(&restored);
    assert!(
        journal_start(&restored) <= from,
        "the backup did not keep the registration's history"
    );
    assert!(matches!(
        restored.resume_consumer("gone", 1),
        Err(DatabaseError::Consumer(RegistryError::Terminated { .. }))
    ));
}

/// Releasing a consumer releases only what it held: the journal a rebuild
/// from the latest checkpoint replays is still kept, whatever consumers do.
#[test]
fn a_cancelled_consumer_does_not_release_what_recovery_needs() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    write(&mut db, 2, "base");
    let ckpt = db.checkpoint(1).expect("checkpoint");
    let replay_from = StorageEngine::checkpoint_replay_floor(&ckpt).expect("replay floor");
    let h = db
        .register_consumer(log_consumer(
            "c",
            InitialSeqno::FromNow,
            ConsumerRetentionPolicy::Strict,
        ))
        .expect("register");
    write(&mut db, 4, "tail");
    db.cancel_consumer(h).expect("cancel");
    reclaim_journal(&db);
    assert!(
        journal_start(&db) <= replay_from,
        "the journal a checkpoint rebuild replays from {replay_from} went with the consumer"
    );
}

/// The database reports, per space, the lowest live position and the
/// registration holding it.
#[test]
fn the_database_reports_who_holds_its_history() {
    use coordinode_replicate::RetentionSpace;

    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    let h = db
        .register_consumer(log_consumer(
            "holder",
            InitialSeqno::FromNow,
            ConsumerRetentionPolicy::Strict,
        ))
        .expect("register");
    let from = db.consumer_position(&h).expect("position");
    write(&mut db, 3, "x");
    let summary = db.consumer_retention().expect("retention");
    let journal = summary
        .iter()
        .find(|v| v.space == RetentionSpace::Log)
        .expect("a journal view");
    assert_eq!(journal.floor, Some(from));
    assert_eq!(journal.held_by, vec!["holder".to_string()]);
    assert!(journal.retained_bytes.is_some_and(|b| b > 0));
    assert!(journal.admits_new);
    let mvcc = summary
        .iter()
        .find(|v| v.space == RetentionSpace::Mvcc)
        .expect("an MVCC view");
    assert_eq!(mvcc.floor, None, "no MVCC consumer holds anything");
}

/// A database that keeps no journal of its own holds no history for
/// consumers and says so, rather than registering protection it cannot give.
#[test]
fn a_database_without_a_journal_refuses_consumers() {
    let db = Database::open_in_memory().expect("open in memory");
    assert!(matches!(
        db.register_consumer(log_consumer(
            "none",
            InitialSeqno::FromNow,
            ConsumerRetentionPolicy::Strict,
        )),
        Err(DatabaseError::Unsupported(_))
    ));
}
