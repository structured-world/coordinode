use std::time::Duration;

use openraft::entry::RaftEntry;
use openraft::type_config::TypeConfigExt;

use super::*;
use crate::storage::{CommittedLeaderId, LogReach};

/// A writer thread over a log in a fresh directory.
struct Fixture {
    _dir: tempfile::TempDir,
    oplog: Arc<std::sync::Mutex<OplogManager>>,
    tail: Arc<Mutex<Tail>>,
    notifier: Arc<AppendNotifier>,
    sync: Arc<LogSync>,
}

fn fixture() -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    let oplog = Arc::new(std::sync::Mutex::new(
        OplogManager::open(dir.path(), 0, 64 * 1024 * 1024, 50_000, 3600).expect("open"),
    ));
    let tail = Arc::new(Mutex::new(Tail::default()));
    let notifier = Arc::new(AppendNotifier::default());
    let sync = Arc::new(
        LogSync::start(
            Arc::clone(&notifier),
            Arc::clone(&oplog),
            Arc::clone(&tail),
            None,
        )
        .expect("start"),
    );
    Fixture {
        _dir: dir,
        oplog,
        tail,
        notifier,
        sync,
    }
}

fn entry(index: u64) -> Arc<Entry> {
    Arc::new(Entry::new_blank(openraft::LogId::new(
        CommittedLeaderId {
            term: 1,
            node_id: 1,
        },
        index,
    )))
}

/// An append of the entries at `indices`, carrying `proposals`, whose answer
/// arrives on the returned receiver.
fn append(
    indices: std::ops::RangeInclusive<u64>,
    proposals: Vec<(ProposalId, u64)>,
) -> (
    Append,
    openraft::type_config::alias::OneshotReceiverOf<TypeConfig, Result<(), io::Error>>,
) {
    let (tx, rx) = TypeConfig::oneshot();
    (
        Append {
            callback: IOFlushed::signal(tx),
            proposals,
            entries: indices.map(entry).collect(),
        },
        rx,
    )
}

async fn answer(
    rx: openraft::type_config::alias::OneshotReceiverOf<TypeConfig, Result<(), io::Error>>,
) -> Result<(), io::Error> {
    tokio::time::timeout(Duration::from_secs(10), rx)
        .await
        .expect("answered within the timeout")
        .expect("the callback was not dropped")
}

/// Appends are written to the segments and answered once durable, each with
/// its own result, and the writers of their proposals learn the index.
#[tokio::test]
async fn appends_are_written_and_answered_once_durable() {
    let f = fixture();
    let first = f
        .notifier
        .subscribe(ProposalId::from_raw(1), LogReach::Durable);
    let second = f
        .notifier
        .subscribe(ProposalId::from_raw(2), LogReach::Durable);

    let (a, a_rx) = append(1..=2, vec![(ProposalId::from_raw(1), 2)]);
    let (b, b_rx) = append(3..=3, vec![(ProposalId::from_raw(2), 3)]);
    f.sync.submit(a);
    f.sync.submit(b);

    answer(a_rx).await.expect("first append durable");
    answer(b_rx).await.expect("second append durable");
    assert_eq!(first.await.expect("first writer told"), 2);
    assert_eq!(second.await.expect("second writer told"), 3);
    let held: Vec<u64> = f
        .oplog
        .lock()
        .unwrap()
        .read_range(1, 4)
        .unwrap()
        .iter()
        .map(|e| e.index)
        .collect();
    assert_eq!(held, vec![1, 2, 3], "every entry is in the segments");
    assert_eq!(*f.sync.written().borrow(), Some(3));
}

/// Submitting never touches the segments: with them held by someone else
/// (a write stuck behind a full sync of the same file, as macOS holds it),
/// a submit still returns at once and the entry is readable from the tail.
/// The Raft core submits; if it wrote the segment itself it would wait out
/// every sync and hand over one batch per sync.
#[tokio::test(flavor = "multi_thread")]
async fn a_submit_never_waits_on_the_segments() {
    let f = fixture();
    let held = f.oplog.lock().unwrap();

    let (a, _rx) = append(1..=1, Vec::new());
    f.tail.lock().push(&a.entries);
    let started = std::time::Instant::now();
    f.sync.submit(a);
    assert!(
        started.elapsed() < Duration::from_millis(100),
        "the submit waited on the segments"
    );
    assert_eq!(
        f.tail.lock().range(1, 2).len(),
        1,
        "the entry is readable before it is written"
    );
    drop(held);
}

/// A write that fails answers every append of its group with the error and
/// tells no writer its proposal is durable: openraft then stops on a storage
/// error instead of counting entries that are not on disk. Every waiter for
/// the written mark is released.
#[cfg(feature = "test-faults")]
#[tokio::test]
async fn a_failed_write_answers_every_queued_append_with_the_error() {
    let f = fixture();
    // An active segment to break.
    let (warm, warm_rx) = append(1..=1, Vec::new());
    f.sync.submit(warm);
    answer(warm_rx).await.expect("first append durable");
    f.oplog.lock().unwrap().fail_writes();

    let mut told = f
        .notifier
        .subscribe(ProposalId::from_raw(9), LogReach::Durable);
    let (a, a_rx) = append(2..=2, vec![(ProposalId::from_raw(9), 2)]);
    f.sync.submit(a);
    assert!(answer(a_rx).await.is_err(), "the append is refused");
    assert!(told.try_recv().is_err(), "no writer hears durable");

    let (b, b_rx) = append(3..=3, Vec::new());
    f.sync.submit(b);
    assert!(
        answer(b_rx).await.is_err(),
        "an append after the failure is not acknowledged"
    );
    assert_eq!(
        *f.sync.written().borrow(),
        Some(u64::MAX),
        "waiters are released"
    );
}

/// A writer under a volatile `j` is answered once the entry is written, and
/// a durable waiter once the sync made it durable.
#[tokio::test]
async fn written_and_durable_waiters_are_both_answered() {
    let f = fixture();
    let written = f
        .notifier
        .subscribe(ProposalId::from_raw(1), LogReach::Written);
    let durable = f
        .notifier
        .subscribe(ProposalId::from_raw(2), LogReach::Durable);

    let (a, a_rx) = append(
        7..=8,
        vec![(ProposalId::from_raw(1), 7), (ProposalId::from_raw(2), 8)],
    );
    f.sync.submit(a);
    answer(a_rx).await.expect("durable");
    assert_eq!(written.await.expect("answered at the write"), 7);
    assert_eq!(durable.await.expect("answered at the sync"), 8);
}

/// `wait_idle` returns only once every queued append has been answered, so a
/// change that rewrites the log never overtakes an answer.
#[tokio::test(flavor = "multi_thread")]
async fn wait_idle_returns_once_the_queue_is_answered() {
    let f = fixture();
    let mut answers = Vec::new();
    for i in 1..=8 {
        let (a, rx) = append(i..=i, Vec::new());
        f.sync.submit(a);
        answers.push(rx);
    }
    let waiter = Arc::clone(&f.sync);
    tokio::task::spawn_blocking(move || waiter.wait_idle())
        .await
        .expect("wait_idle");
    for rx in answers {
        // A zero timeout still polls once: an answer already sent is ready.
        let ready = tokio::time::timeout(Duration::ZERO, rx).await;
        assert!(
            matches!(ready, Ok(Ok(Ok(())))),
            "answered before wait_idle returned"
        );
    }
}

/// `wait_written` returns once the entry is in the segments, so a reader of
/// the segments outside openraft finds it.
#[tokio::test(flavor = "multi_thread")]
async fn wait_written_returns_once_the_entry_is_in_the_segments() {
    let f = fixture();
    let (a, _rx) = append(1..=5, Vec::new());
    f.sync.submit(a);
    let waiter = Arc::clone(&f.sync);
    tokio::task::spawn_blocking(move || waiter.wait_written(5))
        .await
        .expect("wait_written");
    assert_eq!(f.oplog.lock().unwrap().read_range(5, 6).unwrap().len(), 1);
}

/// A cut log brings the written mark down, so an entry appended again at a
/// cut index is not taken for written before it is.
#[tokio::test]
async fn a_cut_brings_the_written_mark_down() {
    let f = fixture();
    let (a, rx) = append(1..=5, Vec::new());
    f.sync.submit(a);
    answer(rx).await.expect("durable");
    f.sync.truncated(3);
    assert_eq!(*f.sync.written().borrow(), Some(2));
}

/// Dropping the log's writer answers what is still queued before its thread
/// stops: an append taken is never left without an answer.
#[tokio::test(flavor = "multi_thread")]
async fn dropping_answers_what_is_queued() {
    let f = fixture();
    let mut answers = Vec::new();
    for i in 1..=8 {
        let (a, rx) = append(i..=i, Vec::new());
        f.sync.submit(a);
        answers.push(rx);
    }
    let sync = f.sync;
    tokio::task::spawn_blocking(move || drop(sync))
        .await
        .expect("drop");
    for rx in answers {
        answer(rx).await.expect("answered on drop");
    }
}

/// The tail holds a contiguous run: an entry at a held index replaces it and
/// what follows, and a cut drops the end.
#[test]
fn the_tail_overwrites_and_cuts() {
    let mut tail = Tail::default();
    tail.push(&(1..=5).map(entry).collect::<Vec<_>>());
    tail.push(&(4..=6).map(entry).collect::<Vec<_>>());
    let held: Vec<u64> = tail.range(0, 100).iter().map(|e| e.log_id.index).collect();
    assert_eq!(held, vec![1, 2, 3, 4, 5, 6]);
    tail.truncate_from(3);
    let held: Vec<u64> = tail.range(0, 100).iter().map(|e| e.log_id.index).collect();
    assert_eq!(held, vec![1, 2]);
    assert_eq!(tail.range(2, 3).len(), 1);
    // A run that does not follow the held one starts the tail over.
    tail.push(&(10..=11).map(entry).collect::<Vec<_>>());
    assert_eq!(tail.first(), Some(10));
}

/// Written entries beyond the kept count leave the tail, unwritten ones never:
/// a read below the tail goes to the segments, which must hold it.
#[test]
fn the_tail_keeps_every_unwritten_entry() {
    let mut tail = Tail::default();
    let last = TAIL_WRITTEN_ENTRIES as u64 + 100;
    tail.push(&(1..=last).map(entry).collect::<Vec<_>>());
    tail.written(50);
    assert_eq!(
        tail.first(),
        Some(1),
        "unwritten entries and the kept written ones stay"
    );
    tail.written(last);
    assert_eq!(tail.first(), Some(last - TAIL_WRITTEN_ENTRIES as u64 + 1));
}
