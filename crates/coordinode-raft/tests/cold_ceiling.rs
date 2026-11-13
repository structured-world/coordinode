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
use coordinode_core::txn::read_concern::HistoryPosition;
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_raft::cluster::RaftNode;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::{StorageEngine, breakpoint_read_gate};

/// More positions than the ledger keeps in memory, each raising the commit
/// ceiling, so the first ones are answered from their stored records.
const POSITIONS: u64 = 300;

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

/// A single-node member that applied `POSITIONS` commits, each raising the
/// ceiling: the position and commit of each.
async fn member() -> (
    tempfile::TempDir,
    Arc<StorageEngine>,
    RaftNode,
    Vec<(u64, u64)>,
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
    let ids = ProposalIdGenerator::new();
    let base = now_us();
    let mut written = Vec::new();
    for i in 1..=POSITIONS {
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
        written.push((node.applied_index(), ts));
    }
    engine.persist().expect("persist the ledger");
    (dir, engine, node, written)
}

/// The causal read of the first position, cold, stalls on its stored record;
/// the runtime (one thread) keeps running a ticker, the read gives up at its
/// deadline, and once the record answers the same read gets the prefix's
/// exact commit.
#[tokio::test(flavor = "current_thread")]
async fn a_stalled_cold_ceiling_holds_up_neither_the_runtime_nor_the_deadline() {
    let (_dir, engine, node, written) = member().await;
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
    let (_dir, _engine, node, written) = member().await;
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
    for _ in 0..64 {
        let gave_up = tokio::time::timeout(
            Duration::from_millis(20),
            fence.await_causal(
                None,
                Some(HistoryPosition {
                    history,
                    index: cold,
                }),
                None,
            ),
        )
        .await;
        assert!(gave_up.is_err(), "a stalled read answered: {gave_up:?}");
    }
    assert!(
        started.elapsed() < LONG_STALL / 2,
        "the reads given up waited out the stall: {:?}",
        started.elapsed()
    );
    let stalled = breakpoint_read_gate::waiting();
    assert!(
        (1..=8).contains(&stalled),
        "{stalled} reads of the stalled record are left behind"
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
