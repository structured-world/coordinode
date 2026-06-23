use super::*;
use coordinode_core::txn::proposal::{
    Mutation, PartitionId, ProposalIdGenerator, ProposalPipeline, RaftProposal,
};
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::partition::Partition;

fn test_engine() -> (tempfile::TempDir, Arc<StorageEngine>) {
    let dir = tempfile::tempdir().expect("tempdir");
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

#[tokio::test]
async fn single_node_bootstrap_becomes_leader() {
    let (_dir, engine) = test_engine();
    let node = RaftNode::single_node(engine).await.expect("bootstrap");

    // Single-node cluster should become leader almost immediately
    // Give it a moment for the election
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    assert!(node.is_leader().await, "single node should be leader");
    assert_eq!(node.node_id(), 1);

    node.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread")]
async fn single_node_propose_and_read() {
    let (_dir, engine) = test_engine();
    let engine_clone = Arc::clone(&engine);
    let node = RaftNode::single_node(engine).await.expect("bootstrap");

    // Wait for leadership
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    let pipeline = node.pipeline();
    let id_gen = ProposalIdGenerator::new();

    let proposal = RaftProposal {
        id: id_gen.next(),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: b"node:1:42".to_vec(),
            value: b"hello-raft".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(100),
        start_ts: Timestamp::from_raw(99),
        bypass_rate_limiter: false,
    };

    // Propose through Raft pipeline
    pipeline.propose_and_wait(&proposal).expect("propose");

    // Verify data was applied to storage (ADR-016: plain keys, no versioned encoding)
    let result = engine_clone
        .get(Partition::Node, b"node:1:42")
        .expect("read");
    assert_eq!(result.as_deref(), Some(b"hello-raft".as_slice()));

    node.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread")]
async fn applied_watermark_advances() {
    let (_dir, engine) = test_engine();
    let mut node = RaftNode::single_node(engine).await.expect("bootstrap");

    // Wait for leadership
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Initial watermark may be >0 due to membership entry
    let initial = node.applied_index();

    let pipeline = node.pipeline();
    let id_gen = ProposalIdGenerator::new();

    // Submit a proposal
    let proposal = RaftProposal {
        id: id_gen.next(),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: b"node:1:1".to_vec(),
            value: b"data".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(100),
        start_ts: Timestamp::from_raw(99),
        bypass_rate_limiter: false,
    };
    pipeline.propose_and_wait(&proposal).expect("propose");

    // Watermark should have advanced
    let after = node.applied_index();
    assert!(
        after > initial,
        "watermark should advance: initial={initial}, after={after}"
    );

    // wait_for_applied should return immediately for already-applied index
    let result = node
        .wait_for_applied(after, std::time::Duration::from_secs(5))
        .await
        .expect("should not timeout for already-applied index");
    assert_eq!(result, after);

    node.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread")]
async fn multiple_proposals_sequential() {
    let (_dir, engine) = test_engine();
    let engine_clone = Arc::clone(&engine);
    let node = RaftNode::single_node(engine).await.expect("bootstrap");

    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    let pipeline = node.pipeline();
    let id_gen = ProposalIdGenerator::new();

    // Submit 5 proposals
    for i in 1..=5u64 {
        let proposal = RaftProposal {
            id: id_gen.next(),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: format!("node:1:{i}").into_bytes(),
                value: format!("value-{i}").into_bytes(),
            }],
            commit_ts: Timestamp::from_raw(100 + i),
            start_ts: Timestamp::from_raw(99 + i),
            bypass_rate_limiter: false,
        };
        pipeline.propose_and_wait(&proposal).expect("propose");
    }

    // Verify all 5 writes (ADR-016: plain keys, direct engine reads)
    for i in 1..=5u64 {
        let result = engine_clone
            .get(Partition::Node, format!("node:1:{i}").as_bytes())
            .expect("read");
        assert_eq!(
            result.as_deref(),
            Some(format!("value-{i}").as_bytes()),
            "mismatch at i={i}"
        );
    }

    node.shutdown().await.expect("shutdown");
}

#[tokio::test]
async fn graceful_shutdown() {
    let (_dir, engine) = test_engine();
    let node = RaftNode::single_node(engine).await.expect("bootstrap");

    tokio::time::sleep(std::time::Duration::from_millis(300)).await;

    // Shutdown should succeed cleanly
    node.shutdown().await.expect("shutdown");
}
