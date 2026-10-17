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
    let gen = ProposalIdGenerator::with_base(node_id << 48);

    let id = gen.next();
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
