use std::time::Duration;

use coordinode_storage::engine::config::SyncMethod;
use openraft::type_config::TypeConfigExt;

use super::*;

/// A handle that syncs a real file in `dir`.
fn file_handle(dir: &std::path::Path) -> SyncHandle {
    let file = std::fs::File::create(dir.join("segment")).expect("create");
    SyncHandle::new(file, SyncMethod::Full)
}

/// An append of the proposals `proposals`, whose answer arrives on the
/// returned receiver.
fn append(
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

/// Appends are answered once durable, each with its own result, and the
/// writers of their proposals learn the index.
#[tokio::test]
async fn appends_are_answered_once_durable() {
    let dir = tempfile::tempdir().expect("tempdir");
    let notifier = Arc::new(AppendNotifier::default());
    let sync = LogSync::start(Arc::clone(&notifier)).expect("start");
    let first = notifier.subscribe(ProposalId::from_raw(1));
    let second = notifier.subscribe(ProposalId::from_raw(2));

    let handle = file_handle(dir.path());
    let (a, a_rx) = append(vec![(ProposalId::from_raw(1), 4)]);
    let (b, b_rx) = append(vec![(ProposalId::from_raw(2), 5)]);
    sync.submit(Some(handle.clone()), a);
    sync.submit(Some(handle), b);

    answer(a_rx).await.expect("first append durable");
    answer(b_rx).await.expect("second append durable");
    assert_eq!(first.await.expect("first writer told"), 4);
    assert_eq!(second.await.expect("second writer told"), 5);
}

/// A sync that fails answers every append of its group with the error and
/// tells no writer its proposal is durable: openraft then stops on a storage
/// error instead of counting entries that are not on disk.
#[cfg(unix)]
#[tokio::test]
async fn a_failed_sync_answers_every_queued_append_with_the_error() {
    use std::os::fd::OwnedFd;

    let notifier = Arc::new(AppendNotifier::default());
    let sync = LogSync::start(Arc::clone(&notifier)).expect("start");
    let mut writer_told = notifier.subscribe(ProposalId::from_raw(9));

    // A pipe cannot be synced, so the sync fails.
    let (_reader, writer) = std::io::pipe().expect("pipe");
    let handle = SyncHandle::new(std::fs::File::from(OwnedFd::from(writer)), SyncMethod::Full);
    let (a, a_rx) = append(vec![(ProposalId::from_raw(9), 1)]);
    let (b, b_rx) = append(Vec::new());
    sync.submit(Some(handle.clone()), a);
    sync.submit(Some(handle), b);

    assert!(answer(a_rx).await.is_err(), "the first append is refused");
    assert!(answer(b_rx).await.is_err(), "the second append is refused");
    assert!(
        writer_told.try_recv().is_err(),
        "no writer hears its proposal is durable"
    );
}

/// `wait_idle` returns only once every queued append has been answered, so a
/// change that rewrites the log never overtakes an answer.
#[tokio::test(flavor = "multi_thread")]
async fn wait_idle_returns_once_the_queue_is_answered() {
    let dir = tempfile::tempdir().expect("tempdir");
    let sync = Arc::new(LogSync::start(Arc::new(AppendNotifier::default())).expect("start"));
    let handle = file_handle(dir.path());
    let mut answers = Vec::new();
    for _ in 0..8 {
        let (a, rx) = append(Vec::new());
        sync.submit(Some(handle.clone()), a);
        answers.push(rx);
    }
    let waiter = Arc::clone(&sync);
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

/// Dropping the log's sync answers what is still queued before its thread
/// stops: an append taken is never left without an answer.
#[tokio::test(flavor = "multi_thread")]
async fn dropping_answers_what_is_queued() {
    let dir = tempfile::tempdir().expect("tempdir");
    let sync = LogSync::start(Arc::new(AppendNotifier::default())).expect("start");
    let handle = file_handle(dir.path());
    let mut answers = Vec::new();
    for _ in 0..8 {
        let (a, rx) = append(Vec::new());
        sync.submit(Some(handle.clone()), a);
        answers.push(rx);
    }
    tokio::task::spawn_blocking(move || drop(sync))
        .await
        .expect("drop");
    for rx in answers {
        answer(rx).await.expect("answered on drop");
    }
}
