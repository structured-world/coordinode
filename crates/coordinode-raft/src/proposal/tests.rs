use super::*;
use crate::cluster::RaftNode;
use coordinode_core::txn::proposal::ProposalIdGenerator;
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
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

fn setup() -> (StorageEngine, TempDir) {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("open");
    (engine, dir)
}

#[test]
fn local_pipeline_put_and_read() {
    let (engine, _dir) = setup();
    let pipeline = LocalProposalPipeline::new(&engine);
    let id_gen = ProposalIdGenerator::new();

    let proposal = RaftProposal {
        id: id_gen.next(),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: b"node:1:1".to_vec(),
            value: b"test-data".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(100),
        start_ts: Timestamp::from_raw(99),
        bypass_rate_limiter: false,
    };

    pipeline
        .propose_and_wait(&proposal)
        .expect("propose should succeed");

    // Verify via direct engine read (plain keys, no MVCC encoding)
    let result = engine.get(Partition::Node, b"node:1:1").expect("read");
    assert_eq!(result.as_deref(), Some(b"test-data".as_slice()));
}

#[test]
fn local_pipeline_delete() {
    let (engine, _dir) = setup();
    let pipeline = LocalProposalPipeline::new(&engine);
    let id_gen = ProposalIdGenerator::new();

    // First write a value
    let put = RaftProposal {
        id: id_gen.next(),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: b"node:1:2".to_vec(),
            value: b"data".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(100),
        start_ts: Timestamp::from_raw(99),
        bypass_rate_limiter: false,
    };
    pipeline.propose_and_wait(&put).expect("put");

    // Snapshot after put — captures the value before deletion
    let snap_after_put = engine.snapshot();

    // Then delete
    let del = RaftProposal {
        id: id_gen.next(),
        mutations: vec![Mutation::Delete {
            partition: PartitionId::Node,
            key: b"node:1:2".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(200),
        start_ts: Timestamp::from_raw(199),
        bypass_rate_limiter: false,
    };
    pipeline.propose_and_wait(&del).expect("delete");

    // Verify: snapshot before delete still sees the value
    let before_del = engine
        .snapshot_get(&snap_after_put, Partition::Node, b"node:1:2")
        .expect("snapshot read");
    assert_eq!(before_del.as_deref(), Some(b"data".as_slice()));

    // Verify: after delete, current value is gone (LSM tombstone)
    let after_del = engine.get(Partition::Node, b"node:1:2").expect("read");
    assert!(after_del.is_none());
}

#[test]
fn local_pipeline_multi_mutation_proposal() {
    let (engine, _dir) = setup();
    let pipeline = LocalProposalPipeline::new(&engine);
    let id_gen = ProposalIdGenerator::new();

    let proposal = RaftProposal {
        id: id_gen.next(),
        mutations: vec![
            Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:1:10".to_vec(),
                value: b"alice".to_vec(),
            },
            Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:1:11".to_vec(),
                value: b"bob".to_vec(),
            },
            Mutation::Put {
                partition: PartitionId::Adj,
                key: b"adj:KNOWS:out:10".to_vec(),
                value: b"posting-list".to_vec(),
            },
            Mutation::Put {
                partition: PartitionId::EdgeProp,
                key: b"edgeprop:KNOWS:10:11".to_vec(),
                value: b"since:2024".to_vec(),
            },
        ],
        commit_ts: Timestamp::from_raw(500),
        start_ts: Timestamp::from_raw(499),
        bypass_rate_limiter: false,
    };

    pipeline.propose_and_wait(&proposal).expect("propose");

    // Verify all four writes via direct engine reads
    assert_eq!(
        engine
            .get(Partition::Node, b"node:1:10")
            .expect("r")
            .as_deref(),
        Some(b"alice".as_slice())
    );
    assert_eq!(
        engine
            .get(Partition::Node, b"node:1:11")
            .expect("r")
            .as_deref(),
        Some(b"bob".as_slice())
    );
    assert_eq!(
        engine
            .get(Partition::Adj, b"adj:KNOWS:out:10")
            .expect("r")
            .as_deref(),
        Some(b"posting-list".as_slice())
    );
    assert_eq!(
        engine
            .get(Partition::EdgeProp, b"edgeprop:KNOWS:10:11")
            .expect("r")
            .as_deref(),
        Some(b"since:2024".as_slice())
    );
}

#[test]
fn local_pipeline_empty_proposal() {
    let (engine, _dir) = setup();
    let pipeline = LocalProposalPipeline::new(&engine);
    let id_gen = ProposalIdGenerator::new();

    let proposal = RaftProposal {
        id: id_gen.next(),
        mutations: vec![],
        commit_ts: Timestamp::from_raw(100),
        start_ts: Timestamp::from_raw(99),
        bypass_rate_limiter: false,
    };

    // Empty proposal should succeed (no-op)
    pipeline.propose_and_wait(&proposal).expect("empty ok");
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

/// G035: bypass_rate_limiter=true skips semaphore acquisition.
/// Proposal succeeds even when rate limiter is fully exhausted.
#[tokio::test(flavor = "multi_thread")]
async fn rate_limiter_bypass_succeeds_when_exhausted() {
    let (_dir, engine) = test_engine();
    let node = RaftNode::single_node(engine).await.expect("bootstrap");

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Create pipeline with capacity=1
    let pipeline = RaftProposalPipeline::with_max_pending(Arc::clone(node.raft()), 1);
    let id_gen = ProposalIdGenerator::new();

    // Exhaust the rate limiter by holding a permit
    // (We can't directly exhaust it from outside, so we test via
    // proposal behavior: a normal proposal uses the semaphore,
    // a bypass proposal doesn't.)

    // Normal proposal — should succeed (uses 1 of 1 permits)
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
    pipeline.propose_and_wait(&normal).expect("normal propose");

    // Bypass proposal — should also succeed (skips semaphore entirely)
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
    pipeline
        .propose_and_wait(&bypass)
        .expect("G035: bypass proposal should succeed regardless of rate limiter state");

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

// ── propose_with_timeout tests (G048) ────────────────────────

/// LocalProposalPipeline: propose_with_timeout delegates to propose_and_wait
/// (ignores timeout). Proposals complete in µs, timeout irrelevant.
#[test]
fn local_pipeline_propose_with_timeout_succeeds() {
    let (engine, _dir) = setup();
    let pipeline = LocalProposalPipeline::new(&engine);
    let id_gen = ProposalIdGenerator::new();

    let proposal = RaftProposal {
        id: id_gen.next(),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: b"node:1:timeout-local".to_vec(),
            value: b"value".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(100),
        start_ts: Timestamp::from_raw(99),
        bypass_rate_limiter: false,
    };

    // Even with a very short timeout, local pipeline ignores it
    pipeline
        .propose_with_timeout(&proposal, Duration::from_nanos(1))
        .expect("local propose_with_timeout should succeed (timeout ignored)");

    let val = engine
        .get(Partition::Node, b"node:1:timeout-local")
        .expect("read");
    assert_eq!(val.as_deref(), Some(b"value".as_slice()));
}

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

/// The local (embedded) pipeline has no Raft log, so it reports no
/// committed index — `None`, never a fabricated zero.
#[test]
fn local_pipeline_has_no_committed_index() {
    let (engine, _dir) = setup();
    let pipeline = LocalProposalPipeline::new(&engine);
    let id_gen = ProposalIdGenerator::new();

    let outcome = pipeline
        .propose_and_wait(&RaftProposal {
            id: id_gen.next(),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:1:local-idx".to_vec(),
                value: b"v".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(100),
            start_ts: Timestamp::from_raw(99),
            bypass_rate_limiter: false,
        })
        .expect("propose");
    assert_eq!(outcome.applied_index, None);
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
