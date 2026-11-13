#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! A causal read at a log position older than the commit ceilings the store
//! keeps in memory reads the ceiling from its stored record. A disk that
//! stops answering that read holds up neither the async runtime the read
//! runs on, nor the read's own deadline or cancellation, nor the applies and
//! the lookups of other positions; the answer, once the disk answers, is the
//! prefix's exact ceiling, and reads given up meanwhile leave a bounded
//! number of reads behind.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use coordinode_core::txn::proposal::{
    Mutation, PartitionId, ProposalIdGenerator, ProposalPipeline, RaftProposal,
};
use coordinode_core::txn::read_concern::{HistoryId, HistoryPosition};
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_raft::cluster::RaftNode;
use coordinode_raft::engine_hold::engine_holders;
use coordinode_raft::read_fence::STORED_CEILING_READS;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::coordinator::SnapshotPin;
use coordinode_storage::engine::core::{CommitCeiling, StorageEngine, breakpoint_read_gate};

/// The read [`member`] holds from before its commits.
type Reader = Option<SnapshotPin>;

/// More positions than the ledger keeps in memory, each raising the commit
/// ceiling, so the first ones are answered from their stored records.
const POSITIONS: u64 = 300;

/// The most positions written while looking for one the ledger answers from
/// its stored record: a machine whose commits raise the ceiling less often
/// needs more of them before the first leaves memory.
const MAX_POSITIONS: u64 = 4096;

/// The stalled read, released after this: long against the read's own
/// deadline, short against the test's.
const STALL: Duration = Duration::from_secs(3);

/// The stall the applies and the other reads are checked under.
const LONG_STALL: Duration = Duration::from_secs(8);

/// The wall clock in microseconds, where commit timestamps live.
fn now_us() -> u64 {
    u64::try_from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_micros(),
    )
    .expect("microseconds fit")
}

/// A single-node member that applied at least `POSITIONS` commits: the
/// coldest position the ledger answers from its stored record, then the
/// newest, each with its commit ceiling as the engine answered it from
/// memory right after the apply.
async fn member() -> (
    tempfile::TempDir,
    Arc<StorageEngine>,
    RaftNode,
    Vec<(u64, u64)>,
    Reader,
) {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = Arc::new(
        StorageEngine::open(&StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            dir.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]))
        .expect("open"),
    );
    let node = RaftNode::single_node(Arc::clone(&engine))
        .await
        .expect("bootstrap");
    for _ in 0..150 {
        if node.is_leader().await {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(node.is_leader().await, "a single node never became leader");
    // A read the store still serves from before the commits: while it is
    // held, the ledger keeps every prefix it can tell apart, so the first
    // positions stay stored breakpoints however soon maintenance moves the
    // collection floor (otherwise it may fold them into its anchor first).
    let reader = engine.pin_latest_snapshot().1;
    let ids = ProposalIdGenerator::new();
    let base = now_us();
    let mut written = Vec::new();
    let mut i = 0;
    loop {
        if i >= POSITIONS
            && written.first().is_some_and(|&(first, _)| {
                matches!(
                    engine.commit_ceiling_at(first),
                    Ok(CommitCeiling::Stored(_))
                )
            })
        {
            break;
        }
        assert!(
            i < MAX_POSITIONS,
            "no position left memory after {i} commits; {:?} records held",
            engine.closure_records_held()
        );
        i += 1;
        let ts = base + i * 10;
        let pipeline = node.pipeline();
        let proposal = RaftProposal {
            id: ids.next(),
            commit_ts: Timestamp::from_raw(ts),
            start_ts: Timestamp::from_raw(ts - 1),
            bypass_rate_limiter: false,
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: format!("node:00:cold{i:04}").into_bytes(),
                value: b"v".to_vec(),
            }],
        };
        tokio::task::spawn_blocking(move || pipeline.propose_and_wait(&proposal))
            .await
            .expect("join")
            .expect("propose");
        // Answered from memory now: the newest breakpoints are cached.
        let position = node.applied_index();
        let ceiling = engine
            .commit_ceiling_through(position)
            .expect("the ceiling of a position just applied");
        written.push((position, ceiling));
    }
    engine.persist().expect("persist the ledger");
    // The coldest position read from its stored record, and the newest.
    let cold = written[0];
    let hot = *written.last().expect("a position");
    (dir, engine, node, vec![cold, hot], reader)
}

/// `count` causal reads of position `index`, each given up after 20 ms.
async fn give_up_reads(
    fence: &mut coordinode_raft::read_fence::ReadFence,
    history: HistoryId,
    index: u64,
    count: usize,
) {
    for _ in 0..count {
        let gave_up = tokio::time::timeout(
            Duration::from_millis(20),
            fence.await_causal(None, Some(HistoryPosition { history, index }), None),
        )
        .await;
        assert!(gave_up.is_err(), "a stalled read answered: {gave_up:?}");
    }
}

/// The reads stopped on the stalled record once `expected` of them reached
/// it, or after a second: the reads given a place reach the record on
/// blocking threads, a moment after their fences gave up.
async fn stalled_reads(expected: usize) -> usize {
    let started = Instant::now();
    while breakpoint_read_gate::waiting() < expected && started.elapsed() < Duration::from_secs(1) {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    // A read past the bound would arrive in the same moment.
    tokio::time::sleep(Duration::from_millis(50)).await;
    breakpoint_read_gate::waiting()
}

/// The disk stopped answering stored ceiling reads until this is dropped: a
/// failed assertion lets the stalled reads go too, so the test runtime's
/// shutdown, which waits for its blocking threads, fails instead of hanging.
struct Stall;

impl Stall {
    fn hold() -> Self {
        breakpoint_read_gate::hold();
        Self
    }
}

impl Drop for Stall {
    fn drop(&mut self) {
        breakpoint_read_gate::release();
    }
}

/// The stored ceiling reads still holding the engine.
fn ceiling_holds(engine: &Arc<StorageEngine>) -> usize {
    engine_holders(engine)
        .into_iter()
        .filter(|name| *name == "stored ceiling read")
        .count()
}

/// A fence cancelled outright (its task aborted, no deadline) while its
/// read is stalled ends at once; the read it started runs on, holding its
/// place, so the reads after it get the places left, and none past them.
#[tokio::test(flavor = "multi_thread")]
async fn a_cancelled_cold_ceiling_read_ends_at_once_and_keeps_its_place() {
    let (_dir, _engine, node, written, _reader) = member().await;
    let (cold, ceiling) = written[0];
    let history = node.read_fence().history().expect("history");

    let stall = Stall::hold();
    let reader = tokio::spawn({
        let mut fence = node.read_fence();
        async move {
            fence
                .await_causal(
                    None,
                    Some(HistoryPosition {
                        history,
                        index: cold,
                    }),
                    None,
                )
                .await
        }
    });
    assert_eq!(
        stalled_reads(1).await,
        1,
        "the read never reached the record"
    );
    let cancelled = Instant::now();
    reader.abort();
    let ended = tokio::time::timeout(Duration::from_secs(1), reader).await;
    assert!(
        matches!(&ended, Ok(Err(e)) if e.is_cancelled()),
        "the cancelled read did not end: {ended:?}"
    );
    assert!(
        cancelled.elapsed() < Duration::from_millis(500),
        "the cancellation waited on the stalled record: {:?}",
        cancelled.elapsed()
    );
    assert_eq!(
        breakpoint_read_gate::waiting(),
        1,
        "the cancelled fence's read stopped with it"
    );

    let mut fence = node.read_fence();
    give_up_reads(&mut fence, history, cold, 16).await;
    assert_eq!(
        stalled_reads(STORED_CEILING_READS).await,
        STORED_CEILING_READS,
        "the cancelled read's place was given to another"
    );

    drop(stall);
    let bound = fence
        .await_causal(
            None,
            Some(HistoryPosition {
                history,
                index: cold,
            }),
            None,
        )
        .await
        .expect("met once the record answers");
    assert_eq!(bound, Some(ceiling), "the prefix's exact commit");
}

/// A member shut down while its disk stops answering stored ceiling reads:
/// the shutdown is not held up past its drain, and each read stalled keeps
/// its place, its engine work and its hold of the engine until the disk
/// answers; then all of them let go.
#[tokio::test(flavor = "multi_thread")]
async fn a_shutdown_waits_out_stalled_ceiling_reads_only_to_its_drain() {
    let (_dir, engine, node, written, _reader) = member().await;
    let (cold, _) = written[0];
    let history = node.read_fence().history().expect("history");

    let stall = Stall::hold();
    {
        let mut fence = node.read_fence();
        give_up_reads(&mut fence, history, cold, 16).await;
    }
    assert_eq!(
        stalled_reads(STORED_CEILING_READS).await,
        STORED_CEILING_READS
    );
    assert_eq!(ceiling_holds(&engine), STORED_CEILING_READS);

    let drain = Duration::from_millis(300);
    node.set_work_drain(drain);
    let started = Instant::now();
    tokio::time::timeout(Duration::from_secs(10), node.shutdown())
        .await
        .expect("the shutdown waited on the stalled reads")
        .expect("shutdown");
    assert!(
        started.elapsed() >= drain,
        "the shutdown did not wait for the reads' work: {:?}",
        started.elapsed()
    );
    assert_eq!(
        breakpoint_read_gate::waiting(),
        STORED_CEILING_READS,
        "a stalled read was cut off"
    );
    assert_eq!(
        ceiling_holds(&engine),
        STORED_CEILING_READS,
        "a stalled read let go of the engine before its read ended"
    );

    drop(stall);
    tokio::time::timeout(Duration::from_secs(5), node.consensus_work_released())
        .await
        .expect("the reads' work outlived the disk's answer");
    assert_eq!(breakpoint_read_gate::waiting(), 0);
    assert_eq!(
        ceiling_holds(&engine),
        0,
        "a read held the engine past its end"
    );
}

/// The causal read of the first position, cold, stalls on its stored record;
/// the runtime (one thread) keeps running a ticker, the read gives up at its
/// deadline, and once the record answers the same read gets the prefix's
/// exact commit.
#[tokio::test(flavor = "current_thread")]
async fn a_stalled_cold_ceiling_holds_up_neither_the_runtime_nor_the_deadline() {
    let (_dir, engine, node, written, _reader) = member().await;
    let (position, ceiling) = written[0];
    assert_eq!(
        engine.commit_ceiling_through(position).expect("ceiling"),
        ceiling,
        "the cold position's ceiling"
    );

    let ticks = Arc::new(AtomicU64::new(0));
    let ticker = tokio::spawn({
        let ticks = Arc::clone(&ticks);
        async move {
            loop {
                tokio::time::sleep(Duration::from_millis(10)).await;
                ticks.fetch_add(1, Ordering::Relaxed);
            }
        }
    });
    breakpoint_read_gate::hold();
    let released = std::thread::spawn(|| {
        std::thread::sleep(STALL);
        breakpoint_read_gate::release();
    });

    let mut fence = node.read_fence();
    let at = HistoryPosition {
        history: fence.history().expect("history"),
        index: position,
    };
    let started = Instant::now();
    let given_up = tokio::time::timeout(
        Duration::from_millis(300),
        fence.await_causal(None, Some(at), None),
    )
    .await;
    let waited = started.elapsed();
    let ticked = ticks.load(Ordering::Relaxed);
    assert!(given_up.is_err(), "the stalled read answered: {given_up:?}");
    assert!(
        waited < STALL / 2,
        "the read's deadline waited out the stalled record: {waited:?}"
    );
    assert!(
        ticked > 5,
        "the runtime stalled with the read: {ticked} ticks"
    );

    released.join().expect("release");
    let bound = fence
        .await_causal(None, Some(at), None)
        .await
        .expect("met once the record answers");
    assert_eq!(bound, Some(ceiling), "the prefix's exact commit");
    ticker.abort();
}

/// While a cold read is stalled, the applies go on (a new commit raises
/// the ledger), a hot position is answered, and reads of the cold position
/// given up again and again leave a bounded number of reads behind.
#[tokio::test(flavor = "multi_thread")]
async fn a_stalled_cold_ceiling_holds_up_no_apply_and_leaves_bounded_reads() {
    let (_dir, _engine, node, written, _reader) = member().await;
    let (cold, ceiling) = written[0];
    let (hot, hot_ceiling) = *written.last().expect("a hot position");
    let mut fence = node.read_fence();
    let history = fence.history().expect("history");

    breakpoint_read_gate::hold();
    // The disk answers again after a while, whatever the reads above did.
    let released = std::thread::spawn(|| {
        std::thread::sleep(LONG_STALL);
        breakpoint_read_gate::release();
    });
    let started = Instant::now();
    give_up_reads(&mut fence, history, cold, 64).await;
    assert!(
        started.elapsed() < LONG_STALL / 2,
        "the reads given up waited out the stall: {:?}",
        started.elapsed()
    );
    // The reads that got a place reach the record off the runtime, a moment
    // after their fences gave up; then exactly as many as there are places
    // stay behind, and more fences giving up add none.
    assert_eq!(
        stalled_reads(STORED_CEILING_READS).await,
        STORED_CEILING_READS,
        "the reads of the stalled record left behind"
    );
    give_up_reads(&mut fence, history, cold, 64).await;
    assert_eq!(
        stalled_reads(STORED_CEILING_READS).await,
        STORED_CEILING_READS,
        "more fences giving up left more reads behind"
    );

    let applied = tokio::time::timeout(Duration::from_secs(5), async {
        let pipeline = node.pipeline();
        let ts = now_us() + 1_000;
        let proposal = RaftProposal {
            id: ProposalIdGenerator::new().next(),
            commit_ts: Timestamp::from_raw(ts),
            start_ts: Timestamp::from_raw(ts - 1),
            bypass_rate_limiter: false,
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:00:during".to_vec(),
                value: b"v".to_vec(),
            }],
        };
        tokio::task::spawn_blocking(move || pipeline.propose_and_wait(&proposal))
            .await
            .expect("join")
    })
    .await;
    assert!(
        matches!(applied, Ok(Ok(_))),
        "an apply waited on the stalled read: {applied:?}"
    );
    let bound = tokio::time::timeout(
        Duration::from_secs(1),
        fence.await_causal(
            None,
            Some(HistoryPosition {
                history,
                index: hot,
            }),
            None,
        ),
    )
    .await
    .expect("a hot position waited on the stalled read")
    .expect("met");
    assert_eq!(bound, Some(hot_ceiling));
    assert!(
        breakpoint_read_gate::waiting() > 0,
        "the stall ended before the apply and the hot read were checked"
    );

    released.join().expect("release");
    let bound = fence
        .await_causal(
            None,
            Some(HistoryPosition {
                history,
                index: cold,
            }),
            None,
        )
        .await
        .expect("met once the record answers");
    assert_eq!(bound, Some(ceiling), "the prefix's exact commit");
    let drained = Instant::now();
    while breakpoint_read_gate::waiting() > 0 && drained.elapsed() < Duration::from_secs(5) {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(
        breakpoint_read_gate::waiting(),
        0,
        "reads left after release"
    );
}
