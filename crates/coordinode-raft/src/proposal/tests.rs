use super::*;
use crate::cluster::RaftNode;
use coordinode_core::txn::proposal::{Mutation, ProposalIdGenerator};
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use tempfile::TempDir;

fn test_engine() -> (TempDir, Arc<StorageEngine>) {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = Arc::new(StorageEngine::open(&config).expect("open"));
    (dir, engine)
}

/// Wait until `node`'s own startup writes are done, so no timestamp of
/// theirs holds a later entry's bound: the group's version pair (the last of
/// them) is recorded, and a sealed write then closes with its own entry,
/// which it does only with no lower timestamp held elsewhere.
async fn settle(
    node: &RaftNode,
    engine: &StorageEngine,
    oracle: &coordinode_core::txn::timestamp::TimestampOracle,
) {
    for _ in 0..300 {
        if node.version().group_pair().is_some() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    // Apart from the ids a test draws from 1, which a warm-up sharing one
    // would turn into a duplicate the log drops.
    let ids = ProposalIdGenerator::with_base(1 << 62);
    drop(closed_by_its_own_entry(node, engine, oracle, &ids, b"node:1:warm").await);
}

/// Write `key` sealed into its proposal, again until the entry carrying the
/// write closes the bound past it: a member-local record written at that
/// moment (its own timestamp held while it is written) keeps an entry's bound
/// below a later commit, and such a write proves nothing about sealing. The
/// proposal and the hold on its timestamp, still standing.
async fn closed_by_its_own_entry(
    node: &RaftNode,
    engine: &StorageEngine,
    oracle: &coordinode_core::txn::timestamp::TimestampOracle,
    ids: &ProposalIdGenerator,
    key: &[u8],
) -> (RaftProposal, coordinode_storage::engine::pending::Unlogged) {
    for _ in 0..20 {
        let (ts, held) = engine.pending_commits().obligate(|| oracle.next().as_raw());
        let proposal = RaftProposal {
            id: ids.next(),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: key.to_vec(),
                value: b"v".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(ts),
            start_ts: Timestamp::from_raw(ts),
            bypass_rate_limiter: false,
        };
        held.seal(proposal.id);
        node.pipeline()
            .propose_async(&proposal)
            .await
            .expect("commit");
        if engine.closure_frontier() > ts {
            return (proposal, held);
        }
        drop(held);
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    unreachable!("no write was closed by its own entry in 20 attempts")
}

#[test]
fn partition_roundtrip() {
    // Verify to_partition / to_partition_id are inverse.
    // Partition::Raft is internal-only and has no PartitionId mapping.
    for p in Partition::all() {
        if *p == Partition::Raft {
            continue;
        }
        let id = to_partition_id(*p);
        let back = to_partition(id);
        assert_eq!(*p, back);
    }
}

// ── RateLimiter tests ──────────────────────────────────────────

#[tokio::test]
async fn rate_limiter_acquire_releases() {
    // Verify that permits are acquired and released correctly
    let limiter = RateLimiter::new(4);

    // Acquire 1 permit (retry=0, weight=1)
    let permit1 = limiter.acquire(0).await.expect("acquire 1");
    // Acquire 2 permits (retry=1, weight=2)
    let permit2 = limiter.acquire(1).await.expect("acquire 2");
    // 3 permits used, 1 remaining — can acquire 1 more
    let permit3 = limiter.acquire(0).await.expect("acquire 3");
    // All 4 permits used

    // Drop permits to release
    drop(permit1);
    drop(permit2);
    drop(permit3);

    // Should be able to acquire again after release
    let _permit4 = limiter.acquire(0).await.expect("acquire after release");
}

#[tokio::test]
async fn rate_limiter_exponential_weight() {
    // Verify exponential weight: retry 0 = 1, retry 1 = 2, retry 2 = 4
    let limiter = RateLimiter::new(8);

    // retry=2 → weight=4, should succeed with 8 permits
    let p1 = limiter.acquire(2).await.expect("weight 4");
    // 4 remaining, retry=2 → weight=4 again, should succeed
    let p2 = limiter.acquire(2).await.expect("weight 4 again");
    // 0 remaining

    drop(p1);
    drop(p2);
}

#[tokio::test]
async fn rate_limiter_timeout_on_exhaustion() {
    // When all permits taken, acquire should block (we test with timeout)
    let limiter = RateLimiter::new(1);

    // Take the only permit
    let _permit = limiter.acquire(0).await.expect("acquire");

    // Trying to acquire another should block — verify with timeout
    let result = tokio::time::timeout(Duration::from_millis(50), limiter.acquire(0)).await;

    // Should timeout (not resolve), proving backpressure works
    assert!(result.is_err(), "should timeout when all permits taken");
}

/// bypass_rate_limiter=true skips semaphore acquisition: with the only
/// permit held, a normal proposal waits while a bypass proposal commits.
#[tokio::test(flavor = "multi_thread")]
async fn rate_limiter_bypass_succeeds_when_exhausted() {
    let (_dir, engine) = test_engine();
    let node = RaftNode::single_node(engine).await.expect("bootstrap");

    tokio::time::sleep(Duration::from_millis(500)).await;

    let pipeline = RaftProposalPipeline::with_max_pending(Arc::clone(node.raft()), 1);
    let id_gen = ProposalIdGenerator::new();

    // Exhaust the limiter: hold its only permit for the rest of the test.
    let _held = pipeline
        .rate_limiter
        .acquire(0)
        .await
        .expect("take the permit");

    let normal = RaftProposal {
        id: id_gen.next(),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: b"node:1:normal".to_vec(),
            value: b"data".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(100),
        start_ts: Timestamp::from_raw(99),
        bypass_rate_limiter: false,
    };
    let blocked =
        tokio::time::timeout(Duration::from_millis(200), pipeline.propose_async(&normal)).await;
    assert!(
        blocked.is_err(),
        "a normal proposal must wait while the limiter is exhausted"
    );

    let bypass = RaftProposal {
        id: id_gen.next(),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: b"node:1:bypass".to_vec(),
            value: b"delta".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(200),
        start_ts: Timestamp::from_raw(199),
        bypass_rate_limiter: true,
    };
    tokio::time::timeout(Duration::from_secs(5), pipeline.propose_async(&bypass))
        .await
        .expect("a bypass proposal must not wait on the exhausted limiter")
        .expect("bypass proposal commits");

    node.shutdown().await.expect("shutdown");
}

/// The bypass flag belongs to the proposal, so it holds for every write
/// concern: `w:0` and `w:1` must not wait on an exhausted limiter either.
#[tokio::test(flavor = "multi_thread")]
async fn rate_limiter_bypass_holds_below_majority() {
    let (_dir, engine) = test_engine();
    let node = RaftNode::single_node(engine).await.expect("bootstrap");

    tokio::time::sleep(Duration::from_millis(500)).await;

    let notifier = Arc::clone(node.append_notifier());
    let pipeline = RaftProposalPipeline {
        rate_limiter: RateLimiter::new(1),
        ..RaftProposalPipeline::with_append_notifier(Arc::clone(node.raft()), Arc::clone(&notifier))
    };
    let id_gen = ProposalIdGenerator::new();
    let _held = pipeline
        .rate_limiter
        .acquire(0)
        .await
        .expect("take the permit");

    let bypass = |key: &[u8], ts: u64| RaftProposal {
        id: id_gen.next(),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: key.to_vec(),
            value: b"delta".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(ts),
        start_ts: Timestamp::from_raw(ts - 1),
        bypass_rate_limiter: true,
    };

    let unacked = bypass(b"node:1:w0", 100);
    tokio::time::timeout(Duration::from_secs(5), pipeline.propose_unacked(&unacked))
        .await
        .expect("a w:0 bypass proposal must not wait on the exhausted limiter")
        .expect("w:0 bypass proposal is accepted");

    let acked = bypass(b"node:1:w1", 200);
    tokio::time::timeout(
        Duration::from_secs(5),
        pipeline.propose_acks(&acked, 1, LogReach::Durable, &notifier),
    )
    .await
    .expect("a w:1 bypass proposal must not wait on the exhausted limiter")
    .expect("w:1 bypass proposal is held by the leader");

    node.shutdown().await.expect("shutdown");
}

// ── Node-scoped ProposalIdGenerator tests ──────────────────────

#[test]
fn node_scoped_id_generator_uniqueness() {
    // Two generators with different node bases produce non-overlapping IDs
    let gen_node1 = ProposalIdGenerator::with_base(1u64 << 48);
    let gen_node2 = ProposalIdGenerator::with_base(2u64 << 48);

    let ids1: Vec<u64> = (0..100).map(|_| gen_node1.next().as_raw()).collect();
    let ids2: Vec<u64> = (0..100).map(|_| gen_node2.next().as_raw()).collect();

    // No overlaps between node1 and node2 ID spaces
    for id1 in &ids1 {
        assert!(
            !ids2.contains(id1),
            "node1 ID {} should not appear in node2 IDs",
            id1
        );
    }

    // IDs within each node are monotonically increasing
    for w in ids1.windows(2) {
        assert!(w[1] > w[0], "IDs should be monotonically increasing");
    }
}

#[test]
fn node_scoped_id_high_bits_preserved() {
    // Verify that node_id is encoded in the high 16 bits
    let node_id: u64 = 42;
    let ids = ProposalIdGenerator::with_base(node_id << 48);

    let id = ids.next();
    // High 16 bits should contain node_id
    assert_eq!(id.as_raw() >> 48, node_id);
}

// ── propose_with_timeout tests ───────────────────────────────

/// RaftProposalPipeline: propose_with_timeout with generous timeout succeeds.
#[tokio::test(flavor = "multi_thread")]
async fn raft_pipeline_propose_with_timeout_succeeds() {
    let (_dir, engine) = test_engine();
    let node = RaftNode::single_node(engine.clone())
        .await
        .expect("bootstrap");
    tokio::time::sleep(Duration::from_millis(500)).await;

    let pipeline = RaftProposalPipeline::new(Arc::clone(node.raft()));
    let id_gen = ProposalIdGenerator::new();

    let proposal = RaftProposal {
        id: id_gen.next(),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: b"node:1:timeout-raft".to_vec(),
            value: b"raft-val".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(100),
        start_ts: Timestamp::from_raw(99),
        bypass_rate_limiter: false,
    };

    // 10 seconds is more than enough for single-node
    pipeline
        .propose_with_timeout(&proposal, Duration::from_secs(10))
        .expect("propose_with_timeout should succeed with generous timeout");

    let val = engine
        .get(Partition::Node, b"node:1:timeout-raft")
        .expect("read");
    assert_eq!(val.as_deref(), Some(b"raft-val".as_slice()));

    node.shutdown().await.expect("shutdown");
}

/// operationTime regression: a Raft-replicated proposal RETURNS its committed
/// log index (the pipeline used to discard it, forcing the gRPC layer
/// to sample the node's current applied index instead — not this
/// write's index). The index must be present and strictly increase
/// across successive writes.
#[tokio::test(flavor = "multi_thread")]
async fn raft_pipeline_returns_committed_index() {
    let (_dir, engine) = test_engine();
    let node = RaftNode::single_node(engine.clone())
        .await
        .expect("bootstrap");
    tokio::time::sleep(Duration::from_millis(500)).await;

    let pipeline = RaftProposalPipeline::new(Arc::clone(node.raft()));
    let id_gen = ProposalIdGenerator::new();

    let mk = |key: &[u8], ts: u64| RaftProposal {
        id: id_gen.next(),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: key.to_vec(),
            value: b"v".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(ts),
        start_ts: Timestamp::from_raw(ts - 1),
        bypass_rate_limiter: false,
    };

    let first = pipeline
        .propose_and_wait(&mk(b"node:1:idx-a", 100))
        .expect("first propose");
    let second = pipeline
        .propose_and_wait(&mk(b"node:1:idx-b", 200))
        .expect("second propose");

    let i1 = first
        .applied_index
        .expect("replicated write must carry a committed index");
    let i2 = second
        .applied_index
        .expect("replicated write must carry a committed index");
    assert!(i1 > 0, "committed index must be non-zero, got {i1}");
    assert!(
        i2 > i1,
        "committed index must strictly increase: {i1} then {i2}"
    );

    // propose_with_timeout surfaces the same committed index.
    let third = pipeline
        .propose_with_timeout(&mk(b"node:1:idx-c", 300), Duration::from_secs(10))
        .expect("third propose");
    assert!(
        third.applied_index.expect("committed index") > i2,
        "propose_with_timeout must also return an advancing index"
    );

    node.shutdown().await.expect("shutdown");
}

/// WriteConcernTimeout error has correct format.
#[test]
fn write_concern_timeout_error_display() {
    let err = ProposalError::WriteConcernTimeout { timeout_ms: 5000 };
    let msg = format!("{err}");
    assert!(
        msg.contains("5000ms"),
        "error should contain timeout_ms: {msg}"
    );
    assert!(
        msg.contains("write concern timeout"),
        "error should mention write concern: {msg}"
    );
}

/// Background workers (the TTL reaper, the NodeId lease worker) propose from
/// a plain `std::thread`, where there is no current tokio runtime. Reaching
/// openraft from there used to panic with "there is no reactor running",
/// killing the worker.
#[tokio::test(flavor = "multi_thread")]
async fn propose_and_wait_works_from_a_plain_thread() {
    let (_dir, engine) = test_engine();
    let node = RaftNode::open(1, Arc::clone(&engine))
        .await
        .expect("open raft node");
    // Built inside the runtime, so the pipeline captures a handle to it.
    let pipeline = Arc::new(RaftProposalPipeline::new(Arc::clone(node.raft())));

    let proposal = RaftProposal {
        id: ProposalIdGenerator::new().next(),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: b"node:1:9001".to_vec(),
            value: b"drain-thread".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(100),
        start_ts: Timestamp::from_raw(99),
        bypass_rate_limiter: false,
    };

    let outcome = std::thread::spawn(move || pipeline.propose_and_wait(&proposal))
        .join()
        .expect("drain-style thread must not panic");

    assert!(outcome.is_ok(), "proposal failed: {outcome:?}");
}

/// A retry hands over a proposal an earlier attempt already handed to the
/// log, where it may have committed. A closed bound past its timestamp then
/// says nothing about that attempt, so the retry answers with an unknown
/// outcome, never with the refusal that tells the caller nothing was written.
#[tokio::test(flavor = "multi_thread")]
async fn a_retry_below_the_closed_bound_is_not_a_refusal() {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let oracle = Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new());
    let engine =
        Arc::new(StorageEngine::open_with_oracle(&config, Arc::clone(&oracle)).expect("open"));
    let node = RaftNode::open_with_oracle(1, Arc::clone(&engine), Some(Arc::clone(&oracle)))
        .await
        .expect("open raft node");
    let pipeline = node.pipeline();
    let (ts, held) = engine.pending_commits().obligate(|| oracle.next().as_raw());
    let proposal = RaftProposal {
        id: ProposalIdGenerator::new().next(),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: b"node:1:retried".to_vec(),
            value: b"v".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(ts),
        start_ts: Timestamp::from_raw(ts),
        bypass_rate_limiter: false,
    };
    pipeline.propose_async(&proposal).await.expect("commit");
    drop(held);
    for _ in 0..100 {
        if engine.closure_frontier() > ts {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(
        engine.closure_frontier() > ts,
        "the bound passed the commit"
    );

    let first = pipeline.entry(&proposal, 0).await.map(|_| ());
    assert!(
        matches!(first, Err(ProposalError::BelowClosedBound { .. })),
        "a first hand-over below the bound is refused: {first:?}"
    );
    let retry = pipeline.entry(&proposal, 1).await.map(|_| ());
    assert!(
        matches!(retry, Err(ProposalError::Timeout { retries: 1 })),
        "a retry below the bound has an unknown outcome: {retry:?}"
    );
    node.shutdown().await.expect("shutdown");
}

/// A commit sealed into its proposal is closed by the entry carrying it, as
/// soon as that entry is applied: no later write and no closing entry is
/// needed. Its timestamp is still registered when the bound is checked, so
/// no closing entry can have passed it. A lower timestamp held for anything
/// else keeps that entry's bound below it, which is also why the first part
/// writes again when a member-local record's timestamp held it.
#[tokio::test(flavor = "multi_thread")]
async fn a_sealed_commit_is_closed_by_its_own_entry() {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let oracle = Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new());
    let engine =
        Arc::new(StorageEngine::open_with_oracle(&config, Arc::clone(&oracle)).expect("open"));
    let node = RaftNode::open_with_oracle(1, Arc::clone(&engine), Some(Arc::clone(&oracle)))
        .await
        .expect("open raft node");
    settle(&node, &engine, &oracle).await;
    let pipeline = node.pipeline();
    let ids = ProposalIdGenerator::new();
    let put = |ts: u64, key: &[u8]| RaftProposal {
        id: ids.next(),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: key.to_vec(),
            value: b"v".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(ts),
        start_ts: Timestamp::from_raw(ts),
        bypass_rate_limiter: false,
    };

    drop(closed_by_its_own_entry(&node, &engine, &oracle, &ids, b"node:1:alone").await);

    // A lower timestamp held for another write, in flight beside this one.
    let (low, low_held) = engine.pending_commits().obligate(|| oracle.next().as_raw());
    let (high, high_held) = engine.pending_commits().obligate(|| oracle.next().as_raw());
    let later = put(high, b"node:1:later");
    high_held.seal(later.id);
    pipeline.propose_async(&later).await.expect("commit");
    assert!(
        engine.closure_frontier() <= low,
        "the bound passed {low}, held for another write: frontier {}",
        engine.closure_frontier()
    );
    drop(high_held);
    drop(low_held);
    node.shutdown().await.expect("shutdown");
}

/// A retry of a sealed commit whose earlier hand-over already closed the bound
/// past it is handed over again, not refused: the bound passed it only
/// through an entry carrying this very proposal, so the proposal is in the
/// log and the retry settles to the same commit. Refusing it reported an
/// unknown outcome for a write that had landed.
#[tokio::test(flavor = "multi_thread")]
async fn a_retry_of_a_sealed_commit_past_its_own_bound_is_handed_over() {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let oracle = Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new());
    let engine =
        Arc::new(StorageEngine::open_with_oracle(&config, Arc::clone(&oracle)).expect("open"));
    let node = RaftNode::open_with_oracle(1, Arc::clone(&engine), Some(Arc::clone(&oracle)))
        .await
        .expect("open raft node");
    settle(&node, &engine, &oracle).await;
    let pipeline = node.pipeline();
    let ids = ProposalIdGenerator::new();
    let (proposal, held) =
        closed_by_its_own_entry(&node, &engine, &oracle, &ids, b"node:1:sealed-retry").await;

    let retry = pipeline.entry(&proposal, 1).await.map(|_| ());
    assert!(retry.is_ok(), "the retry is handed over: {retry:?}");
    drop(held);
    node.shutdown().await.expect("shutdown");
}

/// A sealed write whose first hand-over gets no answer in time lands once.
/// The log's sync stalls past the first attempt's timeout, so the caller
/// does not know whether that attempt landed; it does, and its own entry
/// closes the bound past the write before the retry is stamped. The retry,
/// through the pipeline's own loop, is handed over and answers success, and
/// the write's effect applies once: the counter it increments reads one.
#[tokio::test(flavor = "multi_thread")]
async fn a_sealed_write_retried_after_an_unknown_outcome_lands_once() {
    use coordinode_storage::oplog::segment::SYNC_STALL_MS;
    use std::sync::atomic::Ordering;
    // Past the first attempt's timeout, short of the second's.
    const STALL: Duration = Duration::from_millis(5_000);

    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let oracle = Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new());
    let engine =
        Arc::new(StorageEngine::open_with_oracle(&config, Arc::clone(&oracle)).expect("open"));
    let node = RaftNode::open_with_oracle(1, Arc::clone(&engine), Some(Arc::clone(&oracle)))
        .await
        .expect("open raft node");
    settle(&node, &engine, &oracle).await;
    let ids = ProposalIdGenerator::new();

    // Two permits: the first attempt takes one, the retry two, so the retry
    // waits on the one the test holds.
    let pipeline = Arc::new(
        RaftProposalPipeline::with_max_pending(Arc::clone(node.raft()), 2)
            .with_closure(Arc::clone(&engine)),
    );
    let gate = pipeline.rate_limiter.acquire(0).await.expect("permit");
    let key = b"counter:once".to_vec();
    let (ts, held) = engine.pending_commits().obligate(|| oracle.next().as_raw());
    let proposal = RaftProposal {
        id: ids.next(),
        mutations: vec![Mutation::Merge {
            partition: PartitionId::Counter,
            key: key.clone(),
            operand: coordinode_storage::engine::merge::encode_counter_delta(1),
        }],
        commit_ts: Timestamp::from_raw(ts),
        start_ts: Timestamp::from_raw(ts),
        bypass_rate_limiter: false,
    };
    held.seal(proposal.id);
    // Whatever was allocated before the write (a member-local record being
    // written) ends before the write is stamped; nothing later can hold below
    // it, so its own entry is what closes the bound past it.
    until(Duration::from_secs(10), "earlier holds end", || {
        !engine.pending_commits().held_below(ts)
    })
    .await;
    let logged_before = last_log_index(&node);

    // The log's sync stalls past the first attempt's timeout: it is handed
    // over and gets no answer in time.
    SYNC_STALL_MS.store(
        u64::try_from(STALL.as_millis()).expect("small"),
        Ordering::Relaxed,
    );
    let write = {
        let pipeline = Arc::clone(&pipeline);
        let proposal = proposal.clone();
        tokio::spawn(async move { pipeline.propose_async(&proposal).await })
    };
    tokio::time::timeout(STALL, crate::proposal::ATTEMPT_TIMED_OUT.notified())
        .await
        .expect("the first attempt times out");
    SYNC_STALL_MS.store(0, Ordering::Relaxed);
    assert!(
        last_log_index(&node) > logged_before,
        "the first attempt was handed over: its entry is in the log"
    );

    // The retry waits for its permits until the first attempt, unanswered,
    // is applied and its own entry closed the bound past it.
    until(Duration::from_secs(30), "the first attempt lands", || {
        engine
            .get(Partition::Counter, &key)
            .ok()
            .flatten()
            .is_some()
    })
    .await;
    until(
        Duration::from_secs(10),
        "its own entry closes the bound past it",
        || engine.closure_frontier() > ts,
    )
    .await;
    assert!(!write.is_finished(), "the retry waits for its permits");

    drop(gate);
    let outcome = write.await.expect("write task");
    assert!(
        outcome.is_ok(),
        "the retry of a write that landed answers success: {outcome:?}"
    );
    let stored = engine
        .get(Partition::Counter, &key)
        .expect("read the counter")
        .expect("the counter exists");
    let count = i64::from_le_bytes(stored[..].try_into().expect("an i64"));
    assert_eq!(count, 1, "the write's effect applied once");
    drop(held);
    node.shutdown().await.expect("shutdown");
}

/// A commit allocated earlier and still held keeps the bound below a later
/// write's own entry: the entry lands, the bound stays below it until the
/// earlier hold ends, then a closing entry covers it.
#[tokio::test(flavor = "multi_thread")]
async fn an_earlier_hold_keeps_the_bound_below_a_later_write() {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let oracle = Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new());
    let engine =
        Arc::new(StorageEngine::open_with_oracle(&config, Arc::clone(&oracle)).expect("open"));
    let node = RaftNode::open_with_oracle(1, Arc::clone(&engine), Some(Arc::clone(&oracle)))
        .await
        .expect("open raft node");
    settle(&node, &engine, &oracle).await;
    let pipeline = RaftProposalPipeline::with_max_pending(Arc::clone(node.raft()), 2)
        .with_closure(Arc::clone(&engine));
    let (earlier, earlier_held) = engine.pending_commits().obligate(|| oracle.next().as_raw());
    let (ts, held) = engine.pending_commits().obligate(|| oracle.next().as_raw());
    let proposal = RaftProposal {
        id: ProposalIdGenerator::new().next(),
        mutations: vec![Mutation::Merge {
            partition: PartitionId::Counter,
            key: b"counter:held-below".to_vec(),
            operand: coordinode_storage::engine::merge::encode_counter_delta(1),
        }],
        commit_ts: Timestamp::from_raw(ts),
        start_ts: Timestamp::from_raw(ts),
        bypass_rate_limiter: false,
    };
    held.seal(proposal.id);
    pipeline.propose_async(&proposal).await.expect("committed");
    assert!(
        engine.closure_frontier() <= earlier,
        "the earlier hold keeps the bound below the write"
    );
    drop(earlier_held);
    drop(held);
    until(
        Duration::from_secs(10),
        "a closing entry covers the write once the hold ends",
        || engine.closure_frontier() > ts,
    )
    .await;
    node.shutdown().await.expect("shutdown");
}

/// The last log index `node` holds.
fn last_log_index(node: &RaftNode) -> u64 {
    use openraft::async_runtime::watch::WatchReceiver;
    node.raft()
        .metrics()
        .borrow_watched()
        .last_log_index
        .unwrap_or(0)
}

/// Wait until `cond` holds, for at most `within`; fail naming `what` if it
/// never does.
async fn until(within: Duration, what: &str, cond: impl Fn() -> bool) {
    let deadline = tokio::time::Instant::now() + within;
    while !cond() {
        assert!(tokio::time::Instant::now() < deadline, "{what}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// An early answer proves where the entry is, not that it committed: the
/// leader's copy written but not synced counts toward `w`, never toward the
/// majority that commits it, so such an answer carries no committed index a
/// caller could wait on or hand out as a causal position.
#[test]
fn an_unsynced_leader_copy_does_not_make_a_committed_answer() {
    use crate::storage::LogReach;

    // One member: the written copy answers w:1, and is not a commit.
    let alone = acks_outcome(7, 0, 1, 1, LogReach::Written).expect("w:1 is held");
    assert_eq!(alone.applied_index, None);
    // Three members, one follower synced: two hold it, but only one durably.
    let two = acks_outcome(7, 1, 2, 2, LogReach::Written).expect("w:2 is held");
    assert_eq!(two.applied_index, None);
    // Two synced followers are a majority without the leader.
    let followers = acks_outcome(7, 2, 1, 2, LogReach::Written).expect("held");
    assert_eq!(followers.applied_index, Some(7));
    // A durable leader copy counts toward the majority as before.
    let durable = acks_outcome(7, 1, 2, 2, LogReach::Durable).expect("held");
    assert_eq!(durable.applied_index, Some(7));
    // Fewer members than w hold it: no answer yet.
    assert!(acks_outcome(7, 0, 2, 2, LogReach::Durable).is_none());
}
