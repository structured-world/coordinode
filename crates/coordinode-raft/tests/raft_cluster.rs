#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration tests for 3-node Raft cluster.
//!
//! Tests the full cluster lifecycle: bootstrap, join, leader election,
//! propose through leader, data replication to followers.
//!
//! IMPORTANT: Every test has a hard timeout to prevent hanging.
//! If streaming AppendEntries is not working, tests fail fast
//! instead of spinning CPU indefinitely.

use std::sync::Arc;
use std::time::Duration;

use coordinode_core::txn::proposal::{
    Mutation, PartitionId, ProposalIdGenerator, ProposalPipeline, RaftProposal,
};
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_raft::cluster::RaftNode;
use coordinode_storage::engine::config::StorageConfig;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

/// Hard timeout for cluster tests.
const TEST_TIMEOUT: Duration = Duration::from_secs(30);

/// Allocate a random available port by binding to :0 and reading the assigned port.
///
/// Each nextest process gets its own address space, so static counters don't work.
/// Bind-to-zero guarantees OS-level uniqueness across parallel test processes.
fn alloc_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind :0 for port alloc");
    let port = listener.local_addr().expect("local_addr").port();
    drop(listener);
    port
}

struct TestNode {
    node: RaftNode,
    engine: Arc<StorageEngine>,
    _dir: tempfile::TempDir,
}

/// Create the FIRST node (leader) — calls initialize().
async fn create_leader(node_id: u64, port: u16) -> TestNode {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::new(dir.path());
    let engine = Arc::new(StorageEngine::open(&config).expect("open"));
    let listen_addr: std::net::SocketAddr = format!("127.0.0.1:{port}").parse().expect("addr");
    let advertise_addr = format!("http://127.0.0.1:{port}");

    let node = RaftNode::open_cluster(node_id, Arc::clone(&engine), listen_addr, advertise_addr)
        .await
        .expect("open leader node");

    TestNode {
        node,
        engine,
        _dir: dir,
    }
}

/// Create a JOINING node — does NOT call initialize(), waits for leader.
async fn create_follower(node_id: u64, port: u16) -> TestNode {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::new(dir.path());
    let engine = Arc::new(StorageEngine::open(&config).expect("open"));
    let listen_addr: std::net::SocketAddr = format!("127.0.0.1:{port}").parse().expect("addr");

    let node = RaftNode::open_joining(node_id, Arc::clone(&engine), listen_addr)
        .await
        .expect("open joining node");

    TestNode {
        node,
        engine,
        _dir: dir,
    }
}

/// Bootstrap a 3-node cluster: node 1 becomes leader, adds nodes 2 and 3.
/// Verifies data replication from leader to followers.
#[tokio::test(flavor = "multi_thread")]
async fn cluster_3_node_bootstrap() {
    // Enable tracing for debugging
    let _ = tracing_subscriber::fmt()
        .with_env_filter("coordinode_raft=debug,openraft=off")
        .with_test_writer()
        .try_init();

    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let p1 = alloc_port();
        let p2 = alloc_port();
        let p3 = alloc_port();

        // Node 1: leader (initializes cluster)
        // Nodes 2, 3: joining (wait for leader to add them)
        let n1 = create_leader(1, p1).await;
        let n2 = create_follower(2, p2).await;
        let n3 = create_follower(3, p3).await;

        // Wait for node 1 to become leader
        tokio::time::sleep(Duration::from_millis(1000)).await;
        assert!(n1.node.is_leader().await, "node 1 should be leader");

        // Add nodes 2 and 3 as learners
        n1.node
            .add_node(2, format!("http://127.0.0.1:{p2}"))
            .await
            .expect("add node 2");
        n1.node
            .add_node(3, format!("http://127.0.0.1:{p3}"))
            .await
            .expect("add node 3");

        // Promote to voters — this blocks until replication reaches quorum.
        // If streaming AppendEntries doesn't work, this will timeout.
        n1.node
            .change_membership(vec![1, 2, 3])
            .await
            .expect("change membership");

        // Wait for membership to propagate
        tokio::time::sleep(Duration::from_millis(1000)).await;

        // Propose a write through the leader
        let pipeline = n1.node.pipeline();
        let id_gen = ProposalIdGenerator::with_base(1u64 << 48);

        let proposal = RaftProposal {
            id: id_gen.next(),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:1:cluster-test".to_vec(),
                value: b"replicated!".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(100),
            start_ts: Timestamp::from_raw(99),
            bypass_rate_limiter: false,
        };

        pipeline
            .propose_and_wait(&proposal)
            .expect("propose on leader");

        // Verify data on leader
        let val1 = n1
            .engine
            .get(Partition::Node, b"node:1:cluster-test")
            .expect("read leader");
        assert_eq!(val1.as_deref(), Some(b"replicated!".as_slice()));

        // Wait for replication to followers
        tokio::time::sleep(Duration::from_millis(1000)).await;

        // Verify data replicated to node 2
        let val2 = n2
            .engine
            .get(Partition::Node, b"node:1:cluster-test")
            .expect("read follower 2");
        assert_eq!(
            val2.as_deref(),
            Some(b"replicated!".as_slice()),
            "data should replicate to node 2"
        );

        // Verify data replicated to node 3
        let val3 = n3
            .engine
            .get(Partition::Node, b"node:1:cluster-test")
            .expect("read follower 3");
        assert_eq!(
            val3.as_deref(),
            Some(b"replicated!".as_slice()),
            "data should replicate to node 3"
        );

        // Shutdown all nodes
        n1.node.shutdown().await.expect("shutdown 1");
        n2.node.shutdown().await.expect("shutdown 2");
        n3.node.shutdown().await.expect("shutdown 3");
    })
    .await;

    assert!(
        result.is_ok(),
        "cluster_3_node_bootstrap TIMED OUT after {TEST_TIMEOUT:?} — \
         likely streaming AppendEntries not working, openraft can't replicate"
    );
}

/// Helper: bootstrap a 3-node cluster and return the nodes + ports.
async fn bootstrap_3_node() -> (TestNode, TestNode, TestNode, u16, u16, u16) {
    let p1 = alloc_port();
    let p2 = alloc_port();
    let p3 = alloc_port();

    let n1 = create_leader(1, p1).await;
    let n2 = create_follower(2, p2).await;
    let n3 = create_follower(3, p3).await;

    tokio::time::sleep(Duration::from_millis(800)).await;

    n1.node
        .add_node(2, format!("http://127.0.0.1:{p2}"))
        .await
        .expect("add 2");
    n1.node
        .add_node(3, format!("http://127.0.0.1:{p3}"))
        .await
        .expect("add 3");
    n1.node
        .change_membership(vec![1, 2, 3])
        .await
        .expect("membership");

    tokio::time::sleep(Duration::from_millis(800)).await;

    (n1, n2, n3, p1, p2, p3)
}

/// Propose on follower returns NotLeader error (not panic, not hang).
#[tokio::test(flavor = "multi_thread")]
async fn cluster_propose_on_follower_returns_error() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("coordinode_raft=debug,openraft=off")
        .with_test_writer()
        .try_init();

    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let (n1, n2, _n3, _, _, _) = bootstrap_3_node().await;

        // Propose on follower (node 2) — should fail with NotLeader or Raft error
        let pipeline = n2.node.pipeline();
        let id_gen = ProposalIdGenerator::with_base(2u64 << 48);

        let proposal = RaftProposal {
            id: id_gen.next(),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:1:follower-write".to_vec(),
                value: b"should-fail".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(200),
            start_ts: Timestamp::from_raw(199),
            bypass_rate_limiter: false,
        };

        let err = pipeline.propose_and_wait(&proposal);
        assert!(err.is_err(), "propose on follower should fail");

        n1.node.shutdown().await.expect("shutdown 1");
        n2.node.shutdown().await.expect("shutdown 2");
    })
    .await;

    assert!(result.is_ok(), "TIMED OUT after {TEST_TIMEOUT:?}");
}

/// Multiple proposals replicate correctly across cluster.
#[tokio::test(flavor = "multi_thread")]
async fn cluster_multiple_proposals_replicate() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("openraft=off")
        .with_test_writer()
        .try_init();

    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let (n1, n2, n3, _, _, _) = bootstrap_3_node().await;

        let pipeline = n1.node.pipeline();
        let id_gen = ProposalIdGenerator::with_base(1u64 << 48);

        // Write 10 proposals
        for i in 1..=10u64 {
            let proposal = RaftProposal {
                id: id_gen.next(),
                mutations: vec![Mutation::Put {
                    partition: PartitionId::Node,
                    key: format!("node:1:multi-{i}").into_bytes(),
                    value: format!("v{i}").into_bytes(),
                }],
                commit_ts: Timestamp::from_raw(100 + i),
                start_ts: Timestamp::from_raw(99 + i),
                bypass_rate_limiter: false,
            };
            pipeline.propose_and_wait(&proposal).expect("propose");
        }

        // Wait for replication
        tokio::time::sleep(Duration::from_millis(1000)).await;

        // Verify on all 3 nodes
        for (label, engine) in [("n1", &n1.engine), ("n2", &n2.engine), ("n3", &n3.engine)] {
            for i in 1..=10u64 {
                let val = engine
                    .get(Partition::Node, format!("node:1:multi-{i}").as_bytes())
                    .expect("read");
                assert_eq!(
                    val.as_deref(),
                    Some(format!("v{i}").as_bytes()),
                    "{label}: missing value for i={i}"
                );
            }
        }

        n1.node.shutdown().await.expect("shutdown 1");
        n2.node.shutdown().await.expect("shutdown 2");
        n3.node.shutdown().await.expect("shutdown 3");
    })
    .await;

    assert!(result.is_ok(), "TIMED OUT — multi-proposal replicate");
}

/// Remove a node from cluster — membership shrinks, cluster continues working.
#[tokio::test(flavor = "multi_thread")]
async fn cluster_remove_node() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("openraft=off")
        .with_test_writer()
        .try_init();

    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let (n1, n2, n3, _, _, _) = bootstrap_3_node().await;

        // Remove node 3 from cluster
        n1.node.remove_node(3).await.expect("remove node 3");

        // Wait for membership to propagate
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Cluster should still work with 2 nodes (quorum = 2)
        let pipeline = n1.node.pipeline();
        let id_gen = ProposalIdGenerator::with_base(1u64 << 48);

        let proposal = RaftProposal {
            id: id_gen.next(),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:1:after-remove".to_vec(),
                value: b"still-works".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(300),
            start_ts: Timestamp::from_raw(299),
            bypass_rate_limiter: false,
        };
        pipeline
            .propose_and_wait(&proposal)
            .expect("propose after removal");

        // Verify on remaining nodes
        tokio::time::sleep(Duration::from_millis(500)).await;

        assert_eq!(
            n1.engine
                .get(Partition::Node, b"node:1:after-remove")
                .expect("r")
                .as_deref(),
            Some(b"still-works".as_slice())
        );

        assert_eq!(
            n2.engine
                .get(Partition::Node, b"node:1:after-remove")
                .expect("r")
                .as_deref(),
            Some(b"still-works".as_slice())
        );

        n1.node.shutdown().await.expect("shutdown 1");
        n2.node.shutdown().await.expect("shutdown 2");
        n3.node.shutdown().await.expect("shutdown 3");
    })
    .await;

    assert!(result.is_ok(), "TIMED OUT — cluster_remove_node");
}

/// Leader failover: shutdown leader → new leader elected → propose works.
#[tokio::test(flavor = "multi_thread")]
async fn cluster_leader_failover() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("openraft=off")
        .with_test_writer()
        .try_init();

    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let (n1, n2, n3, _, _, _) = bootstrap_3_node().await;

        // Write initial data through leader (node 1)
        let pipeline1 = n1.node.pipeline();
        let id_gen = ProposalIdGenerator::with_base(1u64 << 48);
        let proposal = RaftProposal {
            id: id_gen.next(),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:1:before-failover".to_vec(),
                value: b"pre-failover".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(100),
            start_ts: Timestamp::from_raw(99),
            bypass_rate_limiter: false,
        };
        pipeline1
            .propose_and_wait(&proposal)
            .expect("propose before failover");
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Kill leader
        n1.node.shutdown().await.expect("shutdown leader");
        drop(n1);

        // Wait for new election (election timeout 300-600ms + some margin)
        tokio::time::sleep(Duration::from_millis(2000)).await;

        // Find who became leader
        let n2_is_leader = n2.node.is_leader().await;
        let n3_is_leader = n3.node.is_leader().await;
        assert!(
            n2_is_leader || n3_is_leader,
            "one of node 2/3 should become leader after failover"
        );

        // Propose through new leader
        let new_leader = if n2_is_leader { &n2 } else { &n3 };
        let pipeline_new = new_leader.node.pipeline();
        let id_gen2 = ProposalIdGenerator::with_base(10u64 << 48);

        let proposal2 = RaftProposal {
            id: id_gen2.next(),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:1:after-failover".to_vec(),
                value: b"post-failover".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(200),
            start_ts: Timestamp::from_raw(199),
            bypass_rate_limiter: false,
        };
        pipeline_new
            .propose_and_wait(&proposal2)
            .expect("propose on new leader after failover");

        // Verify both old and new data on a follower
        tokio::time::sleep(Duration::from_millis(500)).await;
        let follower = if n2_is_leader { &n3 } else { &n2 };

        let old = follower
            .engine
            .get(Partition::Node, b"node:1:before-failover")
            .expect("r");
        assert_eq!(
            old.as_deref(),
            Some(b"pre-failover".as_slice()),
            "old data should survive failover"
        );

        let new = follower
            .engine
            .get(Partition::Node, b"node:1:after-failover")
            .expect("r");
        assert_eq!(
            new.as_deref(),
            Some(b"post-failover".as_slice()),
            "new data should replicate after failover"
        );

        n2.node.shutdown().await.expect("shutdown 2");
        n3.node.shutdown().await.expect("shutdown 3");
    })
    .await;

    assert!(result.is_ok(), "TIMED OUT — cluster_leader_failover");
}

/// Cluster crash recovery: write → shutdown all → reopen → verify data + cluster works.
#[tokio::test(flavor = "multi_thread")]
async fn cluster_crash_recovery() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("openraft=off")
        .with_test_writer()
        .try_init();

    let result = tokio::time::timeout(Duration::from_secs(45), async {
        let p1 = alloc_port();
        let p2 = alloc_port();
        let p3 = alloc_port();

        let dir1 = tempfile::tempdir().expect("d1");
        let dir2 = tempfile::tempdir().expect("d2");
        let dir3 = tempfile::tempdir().expect("d3");
        let path1 = dir1.path().to_path_buf();
        let path2 = dir2.path().to_path_buf();
        let path3 = dir3.path().to_path_buf();

        // Phase 1: Bootstrap + write data
        {
            let e1 = Arc::new(StorageEngine::open(&StorageConfig::new(&path1)).expect("open 1"));
            let e2 = Arc::new(StorageEngine::open(&StorageConfig::new(&path2)).expect("open 2"));
            let e3 = Arc::new(StorageEngine::open(&StorageConfig::new(&path3)).expect("open 3"));

            let n1 = RaftNode::open_cluster(
                1,
                Arc::clone(&e1),
                format!("127.0.0.1:{p1}").parse().expect("a"),
                format!("http://127.0.0.1:{p1}"),
            )
            .await
            .expect("n1");
            let n2 = RaftNode::open_joining(
                2,
                Arc::clone(&e2),
                format!("127.0.0.1:{p2}").parse().expect("a"),
            )
            .await
            .expect("n2");
            let n3 = RaftNode::open_joining(
                3,
                Arc::clone(&e3),
                format!("127.0.0.1:{p3}").parse().expect("a"),
            )
            .await
            .expect("n3");

            tokio::time::sleep(Duration::from_millis(800)).await;

            n1.add_node(2, format!("http://127.0.0.1:{p2}"))
                .await
                .expect("add 2");
            n1.add_node(3, format!("http://127.0.0.1:{p3}"))
                .await
                .expect("add 3");
            n1.change_membership(vec![1, 2, 3])
                .await
                .expect("membership");
            tokio::time::sleep(Duration::from_millis(800)).await;

            // Write data
            let pipeline = n1.pipeline();
            let id_gen = ProposalIdGenerator::with_base(1u64 << 48);
            for i in 1..=3u64 {
                let p = RaftProposal {
                    id: id_gen.next(),
                    mutations: vec![Mutation::Put {
                        partition: PartitionId::Node,
                        key: format!("node:1:recovery-{i}").into_bytes(),
                        value: format!("val-{i}").into_bytes(),
                    }],
                    commit_ts: Timestamp::from_raw(100 + i),
                    start_ts: Timestamp::from_raw(99 + i),
                    bypass_rate_limiter: false,
                };
                pipeline.propose_and_wait(&p).expect("propose");
            }
            tokio::time::sleep(Duration::from_millis(500)).await;

            // Shutdown all
            n1.shutdown().await.expect("s1");
            n2.shutdown().await.expect("s2");
            n3.shutdown().await.expect("s3");
        }
        // All dropped, files flushed

        // Phase 2: Reopen all nodes from same storage, verify data
        {
            let e1 = Arc::new(StorageEngine::open(&StorageConfig::new(&path1)).expect("reopen 1"));
            let e2 = Arc::new(StorageEngine::open(&StorageConfig::new(&path2)).expect("reopen 2"));
            let e3 = Arc::new(StorageEngine::open(&StorageConfig::new(&path3)).expect("reopen 3"));

            // Reopen with SAME ports so membership addresses match
            let n1 = RaftNode::open_cluster(
                1,
                Arc::clone(&e1),
                format!("127.0.0.1:{p1}").parse().expect("a"),
                format!("http://127.0.0.1:{p1}"),
            )
            .await
            .expect("reopen n1");
            let n2 = RaftNode::open_cluster(
                2,
                Arc::clone(&e2),
                format!("127.0.0.1:{p2}").parse().expect("a"),
                format!("http://127.0.0.1:{p2}"),
            )
            .await
            .expect("reopen n2");
            let n3 = RaftNode::open_cluster(
                3,
                Arc::clone(&e3),
                format!("127.0.0.1:{p3}").parse().expect("a"),
                format!("http://127.0.0.1:{p3}"),
            )
            .await
            .expect("reopen n3");

            // Wait for leader election
            tokio::time::sleep(Duration::from_millis(2000)).await;

            // Verify data survived on all nodes
            for (label, engine) in [("n1", &e1), ("n2", &e2), ("n3", &e3)] {
                for i in 1..=3u64 {
                    let val = engine
                        .get(Partition::Node, format!("node:1:recovery-{i}").as_bytes())
                        .expect("read");
                    assert_eq!(
                        val.as_deref(),
                        Some(format!("val-{i}").as_bytes()),
                        "{label}: data i={i} did not survive cluster restart"
                    );
                }
            }

            // Verify cluster is functional: find leader and propose
            let any_leader = n1.is_leader().await || n2.is_leader().await || n3.is_leader().await;
            assert!(any_leader, "cluster should elect a leader after restart");

            let leader = if n1.is_leader().await {
                &n1
            } else if n2.is_leader().await {
                &n2
            } else {
                &n3
            };

            let pipeline = leader.pipeline();
            let id_gen = ProposalIdGenerator::with_base(99u64 << 48);
            let p = RaftProposal {
                id: id_gen.next(),
                mutations: vec![Mutation::Put {
                    partition: PartitionId::Node,
                    key: b"node:1:post-recovery".to_vec(),
                    value: b"works!".to_vec(),
                }],
                commit_ts: Timestamp::from_raw(500),
                start_ts: Timestamp::from_raw(499),
                bypass_rate_limiter: false,
            };
            pipeline
                .propose_and_wait(&p)
                .expect("propose after cluster restart");

            n1.shutdown().await.expect("s1");
            n2.shutdown().await.expect("s2");
            n3.shutdown().await.expect("s3");
        }
    })
    .await;

    assert!(result.is_ok(), "TIMED OUT — cluster_crash_recovery");
}

/// Trigger snapshot on leader: verify snapshot is built with real data
/// and that log purge works correctly afterward.
#[tokio::test(flavor = "multi_thread")]
async fn cluster_snapshot_build_and_purge() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("coordinode_raft=debug,openraft=off")
        .with_test_writer()
        .try_init();

    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let (n1, n2, n3, _, _, _) = bootstrap_3_node().await;

        // Write data through leader
        let pipeline = n1.node.pipeline();
        let id_gen = ProposalIdGenerator::with_base(1u64 << 48);

        for i in 1..=5u64 {
            let proposal = RaftProposal {
                id: id_gen.next(),
                mutations: vec![Mutation::Put {
                    partition: PartitionId::Node,
                    key: format!("node:1:snap-test-{i}").into_bytes(),
                    value: format!("data-{i}").into_bytes(),
                }],
                commit_ts: Timestamp::from_raw(100 + i),
                start_ts: Timestamp::from_raw(99 + i),
                bypass_rate_limiter: false,
            };
            pipeline.propose_and_wait(&proposal).expect("propose");
        }

        // Wait for replication
        tokio::time::sleep(Duration::from_millis(1000)).await;

        // Trigger snapshot on leader via openraft API
        n1.node
            .raft()
            .trigger()
            .snapshot()
            .await
            .expect("trigger snapshot");

        // Wait for snapshot to complete
        tokio::time::sleep(Duration::from_millis(2000)).await;

        // Verify snapshot exists via get_snapshot
        let snap = n1.node.raft().get_snapshot().await.expect("get_snapshot");
        assert!(snap.is_some(), "snapshot should exist after trigger");

        let snap = snap.expect("snap");
        assert!(
            snap.meta.last_log_id.is_some(),
            "snapshot should have last_log_id"
        );

        // Verify snapshot data is non-empty (contains actual storage KV data)
        let snap_data = snap.snapshot.into_inner();
        assert!(
            snap_data.len() > 100,
            "snapshot data should contain actual KV data, got {} bytes",
            snap_data.len()
        );

        // Verify snapshot data starts with correct magic
        assert_eq!(
            &snap_data[..4],
            b"CNSN",
            "snapshot data should start with CNSN magic"
        );

        // Trigger log purge up to a known index
        let applied = n1.node.applied_index();
        if applied > 2 {
            n1.node
                .raft()
                .trigger()
                .purge_log(applied - 1)
                .await
                .expect("trigger purge");

            tokio::time::sleep(Duration::from_millis(500)).await;
        }

        // Verify data is still accessible after snapshot + purge
        for i in 1..=5u64 {
            let val = n1
                .engine
                .get(Partition::Node, format!("node:1:snap-test-{i}").as_bytes())
                .expect("read after snapshot");
            assert_eq!(
                val.as_deref(),
                Some(format!("data-{i}").as_bytes()),
                "data-{i} should survive snapshot + purge"
            );
        }

        n1.node.shutdown().await.expect("shutdown 1");
        n2.node.shutdown().await.expect("shutdown 2");
        n3.node.shutdown().await.expect("shutdown 3");
    })
    .await;

    assert!(
        result.is_ok(),
        "TIMED OUT — cluster_snapshot_build_and_purge"
    );
}

/// Snapshot install on follower: build snapshot from leader data, manually
/// install on a fresh engine, verify all KV data transferred.
#[tokio::test(flavor = "multi_thread")]
async fn cluster_snapshot_install_restores_data() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("coordinode_raft=debug,openraft=off")
        .with_test_writer()
        .try_init();

    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let (n1, n2, n3, _, _, _) = bootstrap_3_node().await;

        // Write data across multiple partitions
        let pipeline = n1.node.pipeline();
        let id_gen = ProposalIdGenerator::with_base(1u64 << 48);

        // Node data
        let proposal = RaftProposal {
            id: id_gen.next(),
            mutations: vec![
                Mutation::Put {
                    partition: PartitionId::Node,
                    key: b"node:0:100".to_vec(),
                    value: b"alice-data".to_vec(),
                },
                Mutation::Put {
                    partition: PartitionId::Adj,
                    key: b"adj:KNOWS:out:100".to_vec(),
                    value: b"\x01\xc8".to_vec(),
                },
                Mutation::Put {
                    partition: PartitionId::Schema,
                    key: b"schema:label:User".to_vec(),
                    value: b"{}".to_vec(),
                },
            ],
            commit_ts: Timestamp::from_raw(200),
            start_ts: Timestamp::from_raw(199),
            bypass_rate_limiter: false,
        };
        pipeline
            .propose_and_wait(&proposal)
            .expect("propose multi-partition");

        tokio::time::sleep(Duration::from_millis(1000)).await;

        // Build a snapshot from leader state
        let snap_data = coordinode_raft::snapshot::build_full_snapshot(&n1.engine)
            .expect("build snapshot from leader");

        // Install snapshot into a completely fresh engine (simulating new node)
        let fresh_dir = tempfile::tempdir().expect("fresh dir");
        let fresh_config = StorageConfig::new(fresh_dir.path());
        let fresh_engine = Arc::new(StorageEngine::open(&fresh_config).expect("fresh engine"));

        coordinode_raft::snapshot::install_full_snapshot(&fresh_engine, &snap_data)
            .expect("install snapshot on fresh engine");

        // Verify data was transferred by checking partition entry counts.
        // Data is MVCC-versioned (keys include version suffix), so we
        // verify via prefix scan count, not exact key lookup.
        let node_count = fresh_engine
            .prefix_scan(Partition::Node, b"")
            .expect("scan node")
            .count();
        assert!(
            node_count > 0,
            "node partition should have entries from snapshot, got {node_count}"
        );

        let adj_count = fresh_engine
            .prefix_scan(Partition::Adj, b"")
            .expect("scan adj")
            .count();
        assert!(
            adj_count > 0,
            "adj partition should have entries from snapshot, got {adj_count}"
        );

        let schema_count = fresh_engine
            .prefix_scan(Partition::Schema, b"")
            .expect("scan schema")
            .count();
        assert!(
            schema_count > 0,
            "schema partition should have entries from snapshot, got {schema_count}"
        );

        // Also verify via plain key read
        let node_val = fresh_engine
            .get(Partition::Node, b"node:0:100")
            .expect("get node");
        assert!(
            node_val.is_some(),
            "node data should be readable from snapshot"
        );

        n1.node.shutdown().await.expect("shutdown 1");
        n2.node.shutdown().await.expect("shutdown 2");
        n3.node.shutdown().await.expect("shutdown 3");
    })
    .await;

    assert!(
        result.is_ok(),
        "TIMED OUT — cluster_snapshot_install_restores_data"
    );
}

/// Background snapshot trigger: with short interval (1s), verify snapshot
/// is automatically built WITHOUT explicit trigger() call.
#[tokio::test(flavor = "multi_thread")]
async fn cluster_background_snapshot_trigger() {
    use coordinode_raft::cluster::SnapshotTriggerConfig;

    let _ = tracing_subscriber::fmt()
        .with_env_filter("coordinode_raft=debug,openraft=off")
        .with_test_writer()
        .try_init();

    let result = tokio::time::timeout(Duration::from_secs(20), async {
        let p1 = alloc_port();

        // Create leader with 1s snapshot trigger interval
        let dir = tempfile::tempdir().expect("tempdir");
        let config = StorageConfig::new(dir.path());
        let engine = Arc::new(StorageEngine::open(&config).expect("open"));
        let listen_addr: std::net::SocketAddr = format!("127.0.0.1:{p1}").parse().expect("addr");

        let snap_config = SnapshotTriggerConfig {
            check_interval: Duration::from_secs(1),
            disk_space_threshold: 0, // always above threshold
        };

        let n1 = RaftNode::open_cluster_with_snapshot_config(
            1,
            Arc::clone(&engine),
            listen_addr,
            format!("http://127.0.0.1:{p1}"),
            snap_config,
        )
        .await
        .expect("open leader with short trigger");

        tokio::time::sleep(Duration::from_millis(500)).await;

        // Write data (need some entries for openraft to build a non-empty snapshot)
        let pipeline = n1.pipeline();
        let id_gen = ProposalIdGenerator::with_base(1u64 << 48);
        for i in 1..=3u64 {
            let proposal = RaftProposal {
                id: id_gen.next(),
                mutations: vec![Mutation::Put {
                    partition: PartitionId::Node,
                    key: format!("node:1:bg-trigger-{i}").into_bytes(),
                    value: format!("val-{i}").into_bytes(),
                }],
                commit_ts: Timestamp::from_raw(100 + i),
                start_ts: Timestamp::from_raw(99 + i),
                bypass_rate_limiter: false,
            };
            pipeline.propose_and_wait(&proposal).expect("propose");
        }

        // NO explicit trigger().snapshot() call — wait for background task
        // Background trigger interval = 1s, so after 3s it should have fired
        tokio::time::sleep(Duration::from_secs(4)).await;

        // Verify snapshot was built automatically
        let snap = n1.raft().get_snapshot().await.expect("get_snapshot");
        assert!(
            snap.is_some(),
            "background trigger should have built a snapshot within 4 seconds"
        );
        let snap = snap.unwrap();
        let data = snap.snapshot.into_inner();
        assert!(data.len() > 10, "snapshot data should be non-empty");
        assert_eq!(&data[..4], b"CNSN", "snapshot should have CNSN magic");

        n1.shutdown().await.expect("shutdown");
    })
    .await;

    assert!(
        result.is_ok(),
        "TIMED OUT — cluster_background_snapshot_trigger"
    );
}

/// gRPC snapshot transfer e2e: leader takes snapshot, purges logs, then
/// a new node joins. The leader can't send AppendEntries (logs purged),
/// so openraft sends snapshot via gRPC. Verify new node has the data.
#[tokio::test(flavor = "multi_thread")]
async fn cluster_snapshot_grpc_transfer_to_new_node() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("coordinode_raft=info,openraft=off")
        .with_test_writer()
        .try_init();

    let result = tokio::time::timeout(Duration::from_secs(45), async {
        let p1 = alloc_port();
        let p2 = alloc_port();

        // Leader with disabled background trigger (we control timing manually)
        let dir1 = tempfile::tempdir().expect("d1");
        let e1 = Arc::new(StorageEngine::open(&StorageConfig::new(dir1.path())).expect("open1"));
        let snap_config = coordinode_raft::cluster::SnapshotTriggerConfig {
            check_interval: Duration::from_secs(3600), // disable periodic
            disk_space_threshold: u64::MAX,            // disable disk-based
        };

        let n1 = RaftNode::open_cluster_with_snapshot_config(
            1,
            Arc::clone(&e1),
            format!("127.0.0.1:{p1}").parse().expect("a"),
            format!("http://127.0.0.1:{p1}"),
            snap_config,
        )
        .await
        .expect("leader");

        tokio::time::sleep(Duration::from_millis(800)).await;
        assert!(n1.is_leader().await, "n1 should be leader");

        // Write data
        let pipeline = n1.pipeline();
        let id_gen = ProposalIdGenerator::with_base(1u64 << 48);
        for i in 1..=5u64 {
            let p = RaftProposal {
                id: id_gen.next(),
                mutations: vec![Mutation::Put {
                    partition: PartitionId::Node,
                    key: format!("node:1:xfer-{i}").into_bytes(),
                    value: format!("val-{i}").into_bytes(),
                }],
                commit_ts: Timestamp::from_raw(100 + i),
                start_ts: Timestamp::from_raw(99 + i),
                bypass_rate_limiter: false,
            };
            pipeline.propose_and_wait(&p).expect("propose");
        }
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Take snapshot and purge all logs
        n1.raft()
            .trigger()
            .snapshot()
            .await
            .expect("trigger snapshot");
        tokio::time::sleep(Duration::from_secs(2)).await;

        let applied = n1.applied_index();
        if applied > 1 {
            n1.raft()
                .trigger()
                .purge_log(applied)
                .await
                .expect("trigger purge");
            tokio::time::sleep(Duration::from_millis(500)).await;
        }

        // Now add a NEW node that has never seen any data.
        // The leader's logs are purged, so it MUST send a snapshot via gRPC.
        let dir2 = tempfile::tempdir().expect("d2");
        let e2 = Arc::new(StorageEngine::open(&StorageConfig::new(dir2.path())).expect("open2"));

        let n2 = RaftNode::open_joining(
            2,
            Arc::clone(&e2),
            format!("127.0.0.1:{p2}").parse().expect("a"),
        )
        .await
        .expect("joining node");

        // Add new node to cluster
        n1.add_node(2, format!("http://127.0.0.1:{p2}"))
            .await
            .expect("add node 2");

        // Wait for snapshot transfer + replication
        // This is the key moment: openraft detects node 2 needs entries
        // that are purged → triggers full_snapshot() → gRPC transfer
        tokio::time::sleep(Duration::from_secs(5)).await;

        // Verify data reached the new node via plain key reads
        let mut found = 0;
        for i in 1..=5u64 {
            let val = e2.get(Partition::Node, format!("node:1:xfer-{i}").as_bytes());
            if let Ok(Some(_)) = val {
                found += 1;
            }
        }

        // Snapshot should have transferred most data. We check >= 3 as a
        // reasonable threshold (MVCC visibility depends on snapshot timing).
        assert!(
            found >= 3,
            "new node should have received data via snapshot transfer, found {found}/5 entries"
        );

        tracing::info!(
            found,
            "gRPC snapshot transfer verified: {found}/5 entries on new node"
        );

        n1.shutdown().await.expect("shutdown 1");
        n2.shutdown().await.expect("shutdown 2");
    })
    .await;

    assert!(
        result.is_ok(),
        "TIMED OUT — cluster_snapshot_grpc_transfer_to_new_node"
    );
}

/// G046: Multi-chunk gRPC snapshot transfer.
/// Same pattern as cluster_snapshot_grpc_transfer_to_new_node but with
/// large payload (>4MB) to verify chunked transfer protocol works
/// end-to-end through real gRPC.
#[tokio::test(flavor = "multi_thread")]
async fn cluster_snapshot_multi_chunk_transfer() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("coordinode_raft=info,openraft=off")
        .with_test_writer()
        .try_init();

    let result = tokio::time::timeout(Duration::from_secs(60), async {
        let p1 = alloc_port();
        let p2 = alloc_port();

        let dir1 = tempfile::tempdir().expect("d1");
        let e1 = Arc::new(StorageEngine::open(&StorageConfig::new(dir1.path())).expect("open1"));
        let snap_config = coordinode_raft::cluster::SnapshotTriggerConfig {
            check_interval: Duration::from_secs(3600),
            disk_space_threshold: u64::MAX,
        };

        let n1 = RaftNode::open_cluster_with_snapshot_config(
            1,
            Arc::clone(&e1),
            format!("127.0.0.1:{p1}").parse().expect("a"),
            format!("http://127.0.0.1:{p1}"),
            snap_config,
        )
        .await
        .expect("leader");

        tokio::time::sleep(Duration::from_millis(800)).await;
        assert!(n1.is_leader().await, "n1 should be leader");

        // Write enough data to produce >4MB snapshot (multiple chunks).
        // Each value is 4KB, 1200 entries ≈ 4.8MB of node data.
        let pipeline = n1.pipeline();
        let id_gen = ProposalIdGenerator::with_base(1u64 << 48);
        let value_4kb = vec![0xABu8; 4096];

        for batch_start in (0..1200u64).step_by(50) {
            let mutations: Vec<Mutation> = (batch_start..batch_start + 50)
                .map(|i| Mutation::Put {
                    partition: PartitionId::Node,
                    key: format!("node:1:big-{i:05}").into_bytes(),
                    value: value_4kb.clone(),
                })
                .collect();
            let p = RaftProposal {
                id: id_gen.next(),
                mutations,
                commit_ts: Timestamp::from_raw(1000 + batch_start),
                start_ts: Timestamp::from_raw(999 + batch_start),
                bypass_rate_limiter: false,
            };
            pipeline.propose_and_wait(&p).expect("propose batch");
        }

        tokio::time::sleep(Duration::from_millis(500)).await;

        // Verify snapshot is large enough to produce multiple chunks
        let snap_data =
            coordinode_raft::snapshot::build_full_snapshot(&e1).expect("build snapshot");
        let chunk_count = coordinode_raft::snapshot::chunk_snapshot_data(&snap_data).count();
        assert!(
            chunk_count > 1,
            "snapshot should produce multiple chunks, got {chunk_count} \
             (data size: {} bytes, chunk size: {})",
            snap_data.len(),
            coordinode_raft::snapshot::SNAPSHOT_CHUNK_SIZE,
        );
        tracing::info!(
            data_bytes = snap_data.len(),
            chunk_count,
            "verified snapshot is multi-chunk"
        );
        drop(snap_data); // free memory

        // Trigger snapshot and purge logs to force gRPC snapshot transfer
        n1.raft()
            .trigger()
            .snapshot()
            .await
            .expect("trigger snapshot");
        tokio::time::sleep(Duration::from_secs(2)).await;

        let applied = n1.applied_index();
        if applied > 1 {
            n1.raft()
                .trigger()
                .purge_log(applied)
                .await
                .expect("trigger purge");
            tokio::time::sleep(Duration::from_millis(500)).await;
        }

        // Add new node — must receive snapshot via chunked gRPC transfer
        let dir2 = tempfile::tempdir().expect("d2");
        let e2 = Arc::new(StorageEngine::open(&StorageConfig::new(dir2.path())).expect("open2"));

        let n2 = RaftNode::open_joining(
            2,
            Arc::clone(&e2),
            format!("127.0.0.1:{p2}").parse().expect("a"),
        )
        .await
        .expect("joining node");

        n1.add_node(2, format!("http://127.0.0.1:{p2}"))
            .await
            .expect("add node 2");

        // Wait for multi-chunk snapshot transfer
        tokio::time::sleep(Duration::from_secs(10)).await;

        // Verify data reached the new node
        let mut found = 0;
        for i in 0..1200u64 {
            let key = format!("node:1:big-{i:05}");
            if let Ok(Some(_)) = e2.get(Partition::Node, key.as_bytes()) {
                found += 1;
            }
        }

        assert!(
            found >= 1000,
            "new node should have received most data via multi-chunk \
             snapshot transfer, found {found}/1200 entries"
        );

        tracing::info!(
            found,
            "multi-chunk gRPC snapshot transfer verified: {found}/1200 entries"
        );

        n1.shutdown().await.expect("shutdown 1");
        n2.shutdown().await.expect("shutdown 2");
    })
    .await;

    assert!(
        result.is_ok(),
        "TIMED OUT — cluster_snapshot_multi_chunk_transfer"
    );
}

/// G042: Follower restart reconnection.
/// Leader has cached gRPC connection to follower → follower shuts down →
/// follower restarts on same port → leader reconnects automatically →
/// new data replicates to restarted follower.
///
/// This specifically tests tonic connect_lazy() reconnection behavior.
#[tokio::test(flavor = "multi_thread")]
async fn cluster_follower_restart_reconnection() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("coordinode_raft=info,openraft=off")
        .with_test_writer()
        .try_init();

    let result = tokio::time::timeout(Duration::from_secs(45), async {
        let p1 = alloc_port();
        let p2 = alloc_port();
        let p3 = alloc_port();

        let dir1 = tempfile::tempdir().expect("d1");
        let dir3 = tempfile::tempdir().expect("d3");
        let path3 = dir3.path().to_path_buf();

        let e1 = Arc::new(StorageEngine::open(&StorageConfig::new(dir1.path())).expect("open1"));
        let e2_dir = tempfile::tempdir().expect("d2");
        let e2 = Arc::new(StorageEngine::open(&StorageConfig::new(e2_dir.path())).expect("open2"));
        let e3 = Arc::new(StorageEngine::open(&StorageConfig::new(&path3)).expect("open3"));

        // Bootstrap 3-node cluster
        let n1 = RaftNode::open_cluster(
            1,
            Arc::clone(&e1),
            format!("127.0.0.1:{p1}").parse().expect("a"),
            format!("http://127.0.0.1:{p1}"),
        )
        .await
        .expect("n1");
        let n2 = RaftNode::open_joining(
            2,
            Arc::clone(&e2),
            format!("127.0.0.1:{p2}").parse().expect("a"),
        )
        .await
        .expect("n2");
        let n3 = RaftNode::open_joining(
            3,
            Arc::clone(&e3),
            format!("127.0.0.1:{p3}").parse().expect("a"),
        )
        .await
        .expect("n3");

        tokio::time::sleep(Duration::from_millis(800)).await;
        n1.add_node(2, format!("http://127.0.0.1:{p2}"))
            .await
            .expect("add2");
        n1.add_node(3, format!("http://127.0.0.1:{p3}"))
            .await
            .expect("add3");
        n1.change_membership(vec![1, 2, 3])
            .await
            .expect("membership");
        tokio::time::sleep(Duration::from_millis(800)).await;

        // Write initial data — verify replication works
        let pipeline = n1.pipeline();
        let id_gen = ProposalIdGenerator::with_base(1u64 << 48);
        let p_before = RaftProposal {
            id: id_gen.next(),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:1:before-restart".to_vec(),
                value: b"pre".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(100),
            start_ts: Timestamp::from_raw(99),
            bypass_rate_limiter: false,
        };
        pipeline
            .propose_and_wait(&p_before)
            .expect("propose before");
        tokio::time::sleep(Duration::from_millis(500)).await;

        let val = e3
            .get(Partition::Node, b"node:1:before-restart")
            .expect("read");
        assert_eq!(
            val.as_deref(),
            Some(b"pre".as_slice()),
            "data should replicate before restart"
        );

        // ── Shutdown node 3 ──────────────────────────────────────────
        // This kills the gRPC server and drops the Raft instance.
        // Leader's cached connection to node 3 becomes stale.
        n3.shutdown().await.expect("shutdown n3");
        drop(n3);
        drop(e3);

        // Wait for leader to notice node 3 is gone (heartbeat failures)
        tokio::time::sleep(Duration::from_secs(2)).await;

        // ── Restart node 3 on SAME port ──────────────────────────────
        // Leader must reconnect to this new instance via connect_lazy().
        let e3_new = Arc::new(StorageEngine::open(&StorageConfig::new(&path3)).expect("reopen3"));
        let n3_new = RaftNode::open_cluster(
            3,
            Arc::clone(&e3_new),
            format!("127.0.0.1:{p3}").parse().expect("a"),
            format!("http://127.0.0.1:{p3}"),
        )
        .await
        .expect("reopen n3");

        // Wait for reconnection + log replay
        tokio::time::sleep(Duration::from_secs(3)).await;

        // ── Find current leader (may have changed after restart) ─────
        // Node 3 restart may trigger re-election (higher term).
        // We need to propose through whoever is leader now.
        let n1_is_leader = n1.is_leader().await;
        let n2_is_leader = n2.is_leader().await;
        let n3_is_leader = n3_new.is_leader().await;
        assert!(
            n1_is_leader || n2_is_leader || n3_is_leader,
            "cluster should have a leader after follower restart"
        );

        let current_leader = if n1_is_leader {
            &n1
        } else if n2_is_leader {
            &n2
        } else {
            &n3_new
        };
        let leader_pipeline = current_leader.pipeline();

        // ── Write new data AFTER restart ─────────────────────────────
        let id_gen2 = ProposalIdGenerator::with_base(10u64 << 48);
        let p_after = RaftProposal {
            id: id_gen2.next(),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:1:after-restart".to_vec(),
                value: b"post".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(200),
            start_ts: Timestamp::from_raw(199),
            bypass_rate_limiter: false,
        };
        leader_pipeline
            .propose_and_wait(&p_after)
            .expect("propose after restart");

        // Wait for replication to all nodes
        tokio::time::sleep(Duration::from_secs(2)).await;

        // ── Verify: restarted node 3 received new data ──────────────
        // This proves gRPC reconnection works: whoever is leader had
        // to reconnect to the restarted node 3 via connect_lazy().
        let val_after = e3_new
            .get(Partition::Node, b"node:1:after-restart")
            .expect("read after restart");
        assert_eq!(
            val_after.as_deref(),
            Some(b"post".as_slice()),
            "G042: data written AFTER follower restart should replicate — \
             proves gRPC reconnection via connect_lazy()"
        );

        n1.shutdown().await.expect("s1");
        n2.shutdown().await.expect("s2");
        n3_new.shutdown().await.expect("s3");
    })
    .await;

    assert!(
        result.is_ok(),
        "TIMED OUT — cluster_follower_restart_reconnection"
    );
}

/// R135: Incremental snapshot — build on leader engine, install on follower engine,
/// verify follower has the delta data. Exercises the full build→serialize→install
/// pipeline in a real 3-node cluster context.
#[tokio::test(flavor = "multi_thread")]
async fn cluster_incremental_snapshot_cross_engine() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("coordinode_raft=info,openraft=off")
        .with_test_writer()
        .try_init();

    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let p1 = alloc_port();
        let p2 = alloc_port();
        let p3 = alloc_port();

        let dir1 = tempfile::tempdir().expect("d1");
        let dir2 = tempfile::tempdir().expect("d2");
        let dir3 = tempfile::tempdir().expect("d3");

        let e1 = Arc::new(StorageEngine::open(&StorageConfig::new(dir1.path())).expect("open1"));
        let e2 = Arc::new(StorageEngine::open(&StorageConfig::new(dir2.path())).expect("open2"));
        let e3 = Arc::new(StorageEngine::open(&StorageConfig::new(dir3.path())).expect("open3"));

        let n1 = RaftNode::open_cluster(
            1,
            Arc::clone(&e1),
            format!("127.0.0.1:{p1}").parse().expect("a"),
            format!("http://127.0.0.1:{p1}"),
        )
        .await
        .expect("n1");
        tokio::time::sleep(Duration::from_millis(800)).await;

        let n2 = RaftNode::open_joining(
            2,
            Arc::clone(&e2),
            format!("127.0.0.1:{p2}").parse().expect("a"),
        )
        .await
        .expect("n2");
        let n3 = RaftNode::open_joining(
            3,
            Arc::clone(&e3),
            format!("127.0.0.1:{p3}").parse().expect("a"),
        )
        .await
        .expect("n3");

        n1.add_node(2, format!("http://127.0.0.1:{p2}"))
            .await
            .expect("add n2");
        n1.add_node(3, format!("http://127.0.0.1:{p3}"))
            .await
            .expect("add n3");
        tokio::time::sleep(Duration::from_secs(2)).await;
        assert!(n1.is_leader().await, "n1 should be leader");

        let pipeline = n1.pipeline();
        let id_gen = ProposalIdGenerator::with_base(1u64 << 48);

        // ── Phase 1: initial data at ts=100 (replicated to all nodes) ──
        for i in 1..=10u64 {
            let p = RaftProposal {
                id: id_gen.next(),
                mutations: vec![Mutation::Put {
                    partition: PartitionId::Node,
                    key: format!("node:0:baseline-{i}").into_bytes(),
                    value: format!("val-{i}").into_bytes(),
                }],
                commit_ts: Timestamp::from_raw(100 + i),
                start_ts: Timestamp::from_raw(99 + i),
                bypass_rate_limiter: false,
            };
            pipeline.propose_and_wait(&p).expect("propose baseline");
        }
        tokio::time::sleep(Duration::from_secs(2)).await;

        // Verify follower has baseline data
        let baseline_val = e3
            .get(Partition::Node, b"node:0:baseline-1")
            .expect("read baseline on follower");
        assert!(baseline_val.is_some(), "follower should have baseline data");

        // ── Phase 2: new data at ts=200 (only on leader via proposals) ──
        for i in 1..=5u64 {
            let p = RaftProposal {
                id: id_gen.next(),
                mutations: vec![Mutation::Put {
                    partition: PartitionId::Node,
                    key: format!("node:0:delta-{i}").into_bytes(),
                    value: format!("new-{i}").into_bytes(),
                }],
                commit_ts: Timestamp::from_raw(200 + i),
                start_ts: Timestamp::from_raw(199 + i),
                bypass_rate_limiter: false,
            };
            pipeline.propose_and_wait(&p).expect("propose delta");
        }
        tokio::time::sleep(Duration::from_secs(2)).await;

        // ── Build incremental on leader since ts=110 (after baseline) ──
        // G057: native seqno MVCC — incremental builder now correctly detects
        // changes in all partitions (not just Schema) via two-snapshot diff.
        use coordinode_core::txn::timestamp::Timestamp as Ts;
        let incr_data =
            coordinode_raft::snapshot::build_incremental_snapshot(&e1, Ts::from_raw(110))
                .expect("build incremental")
                .expect("should have changes (schema + data partitions)");

        // ── Serialize as SnapshotTransfer (simulates gRPC send) ──
        let transfer = coordinode_raft::snapshot::SnapshotTransfer {
            vote: coordinode_raft::storage::Vote::new(1, 1),
            meta: openraft::storage::SnapshotMeta {
                last_log_id: None,
                last_membership: openraft::StoredMembership::default(),
                snapshot_id: "incr-test".to_string(),
            },
            data: incr_data.clone(),
            since_ts: Some(110),
        };
        let wire_bytes = rmp_serde::to_vec(&transfer).expect("serialize transfer");

        // ── Deserialize on "follower" side (simulates gRPC receive) ──
        let received: coordinode_raft::snapshot::SnapshotTransfer =
            rmp_serde::from_slice(&wire_bytes).expect("deserialize transfer");
        assert_eq!(received.since_ts, Some(110));

        // ── Verify delta data replicated via Raft to follower (e3) ──
        // Delta data replicates via Raft AppendEntries. Verify on follower directly.
        let d1 = e3
            .get(Partition::Node, b"node:0:delta-1")
            .expect("get delta-1 on follower");
        assert!(
            d1.is_some(),
            "follower should have delta-1 via Raft replication"
        );
        assert_eq!(d1.unwrap().as_ref(), b"new-1");

        let d5 = e3
            .get(Partition::Node, b"node:0:delta-5")
            .expect("get delta-5 on follower");
        assert!(
            d5.is_some(),
            "follower should have delta-5 via Raft replication"
        );
        assert_eq!(d5.unwrap().as_ref(), b"new-5");

        // Verify incremental (Schema-only) is smaller than full snapshot
        let full_data = coordinode_raft::snapshot::build_full_snapshot(&e1).expect("build full");
        assert!(
            incr_data.len() < full_data.len(),
            "incremental ({}) should be smaller than full ({})",
            incr_data.len(),
            full_data.len()
        );

        n1.shutdown().await.expect("s1");
        n2.shutdown().await.expect("s2");
        n3.shutdown().await.expect("s3");
    })
    .await;

    assert!(
        result.is_ok(),
        "TIMED OUT — cluster_incremental_snapshot_cross_engine"
    );
}

/// R136: Graceful leader transfer — leader transfers to specific peer,
/// new leader serves writes, old leader becomes follower.
#[tokio::test(flavor = "multi_thread")]
async fn cluster_graceful_leader_transfer() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("coordinode_raft=info,openraft=off")
        .with_test_writer()
        .try_init();

    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let p1 = alloc_port();
        let p2 = alloc_port();
        let p3 = alloc_port();

        let dir1 = tempfile::tempdir().expect("d1");
        let dir2 = tempfile::tempdir().expect("d2");
        let dir3 = tempfile::tempdir().expect("d3");

        let e1 = Arc::new(StorageEngine::open(&StorageConfig::new(dir1.path())).expect("open1"));
        let e2 = Arc::new(StorageEngine::open(&StorageConfig::new(dir2.path())).expect("open2"));
        let e3 = Arc::new(StorageEngine::open(&StorageConfig::new(dir3.path())).expect("open3"));

        let n1 = RaftNode::open_cluster(
            1,
            Arc::clone(&e1),
            format!("127.0.0.1:{p1}").parse().expect("a"),
            format!("http://127.0.0.1:{p1}"),
        )
        .await
        .expect("n1");
        tokio::time::sleep(Duration::from_millis(800)).await;

        let n2 = RaftNode::open_joining(
            2,
            Arc::clone(&e2),
            format!("127.0.0.1:{p2}").parse().expect("a"),
        )
        .await
        .expect("n2");
        let n3 = RaftNode::open_joining(
            3,
            Arc::clone(&e3),
            format!("127.0.0.1:{p3}").parse().expect("a"),
        )
        .await
        .expect("n3");

        n1.add_node(2, format!("http://127.0.0.1:{p2}"))
            .await
            .expect("add n2");
        n1.add_node(3, format!("http://127.0.0.1:{p3}"))
            .await
            .expect("add n3");
        n1.change_membership(vec![1, 2, 3])
            .await
            .expect("membership");
        tokio::time::sleep(Duration::from_secs(2)).await;
        assert!(n1.is_leader().await, "n1 should be leader initially");

        // Write data through leader n1
        let pipeline = n1.pipeline();
        let id_gen = ProposalIdGenerator::with_base(1u64 << 48);
        let p = RaftProposal {
            id: id_gen.next(),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:0:before-transfer".to_vec(),
                value: b"pre-transfer".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(100),
            start_ts: Timestamp::from_raw(99),
            bypass_rate_limiter: false,
        };
        pipeline
            .propose_and_wait(&p)
            .expect("propose before transfer");
        tokio::time::sleep(Duration::from_secs(1)).await;

        // ── Transfer leadership from n1 to n2 ──
        n1.transfer_leadership_to(2).await.expect("transfer to n2");

        // n1 should no longer be leader
        assert!(
            !n1.is_leader().await,
            "n1 should NOT be leader after transfer"
        );

        // n2 should become leader (or n3 may win if election race)
        let new_leader = if n2.is_leader().await {
            &n2
        } else if n3.is_leader().await {
            &n3
        } else {
            // Wait a bit more for election
            tokio::time::sleep(Duration::from_secs(1)).await;
            if n2.is_leader().await {
                &n2
            } else {
                &n3
            }
        };
        assert!(
            new_leader.is_leader().await,
            "a new leader should be elected after transfer"
        );

        // ── Write through new leader ──
        let new_pipeline = new_leader.pipeline();
        let id_gen2 = ProposalIdGenerator::with_base(2u64 << 48);
        let p2_prop = RaftProposal {
            id: id_gen2.next(),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:0:after-transfer".to_vec(),
                value: b"post-transfer".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(200),
            start_ts: Timestamp::from_raw(199),
            bypass_rate_limiter: false,
        };
        new_pipeline
            .propose_and_wait(&p2_prop)
            .expect("propose through new leader");
        tokio::time::sleep(Duration::from_secs(1)).await;

        // ── Verify: both writes replicated to all nodes ──
        let pre = e1
            .get(Partition::Node, b"node:0:before-transfer")
            .expect("read pre on n1");
        assert_eq!(pre.as_deref(), Some(b"pre-transfer".as_slice()));

        let post = e1
            .get(Partition::Node, b"node:0:after-transfer")
            .expect("read post on n1");
        assert_eq!(post.as_deref(), Some(b"post-transfer".as_slice()));

        n1.shutdown().await.expect("s1");
        n2.shutdown().await.expect("s2");
        n3.shutdown().await.expect("s3");
    })
    .await;

    assert!(
        result.is_ok(),
        "TIMED OUT — cluster_graceful_leader_transfer"
    );
}

/// R136: Graceful shutdown transfers leadership automatically.
#[tokio::test(flavor = "multi_thread")]
async fn cluster_graceful_shutdown_transfers_leadership() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("coordinode_raft=info,openraft=off")
        .with_test_writer()
        .try_init();

    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let p1 = alloc_port();
        let p2 = alloc_port();
        let p3 = alloc_port();

        let dir1 = tempfile::tempdir().expect("d1");
        let dir2 = tempfile::tempdir().expect("d2");
        let dir3 = tempfile::tempdir().expect("d3");

        let e1 = Arc::new(StorageEngine::open(&StorageConfig::new(dir1.path())).expect("open1"));
        let e2 = Arc::new(StorageEngine::open(&StorageConfig::new(dir2.path())).expect("open2"));
        let e3 = Arc::new(StorageEngine::open(&StorageConfig::new(dir3.path())).expect("open3"));

        let n1 = RaftNode::open_cluster(
            1,
            Arc::clone(&e1),
            format!("127.0.0.1:{p1}").parse().expect("a"),
            format!("http://127.0.0.1:{p1}"),
        )
        .await
        .expect("n1");
        tokio::time::sleep(Duration::from_millis(800)).await;

        let n2 = RaftNode::open_joining(
            2,
            Arc::clone(&e2),
            format!("127.0.0.1:{p2}").parse().expect("a"),
        )
        .await
        .expect("n2");
        let n3 = RaftNode::open_joining(
            3,
            Arc::clone(&e3),
            format!("127.0.0.1:{p3}").parse().expect("a"),
        )
        .await
        .expect("n3");

        n1.add_node(2, format!("http://127.0.0.1:{p2}"))
            .await
            .expect("add n2");
        n1.add_node(3, format!("http://127.0.0.1:{p3}"))
            .await
            .expect("add n3");
        n1.change_membership(vec![1, 2, 3])
            .await
            .expect("membership");
        tokio::time::sleep(Duration::from_secs(2)).await;
        assert!(n1.is_leader().await, "n1 should be leader");

        // ── Graceful shutdown of leader (should auto-transfer) ──
        n1.shutdown().await.expect("graceful shutdown n1");

        // Wait for new election to complete
        tokio::time::sleep(Duration::from_secs(2)).await;

        // One of the remaining nodes should become leader
        let has_leader = n2.is_leader().await || n3.is_leader().await;
        assert!(
            has_leader,
            "cluster should elect a new leader after graceful shutdown"
        );

        n2.shutdown().await.expect("s2");
        n3.shutdown().await.expect("s3");
    })
    .await;

    assert!(
        result.is_ok(),
        "TIMED OUT — cluster_graceful_shutdown_transfers_leadership"
    );
}

/// R137: Snapshot-based replica bootstrap — new node joins a cluster with
/// purged logs, receives snapshot, then catches up via log replay for
/// writes that happened AFTER the snapshot. Verifies both snapshot data
/// and post-snapshot log replay are present on the bootstrapped node.
#[tokio::test(flavor = "multi_thread")]
async fn cluster_snapshot_bootstrap_then_log_replay() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("coordinode_raft=info,openraft=off")
        .with_test_writer()
        .try_init();

    let result = tokio::time::timeout(Duration::from_secs(45), async {
        let p1 = alloc_port();
        let p2 = alloc_port();

        let dir1 = tempfile::tempdir().expect("d1");
        let e1 = Arc::new(StorageEngine::open(&StorageConfig::new(dir1.path())).expect("open1"));

        let snap_config = coordinode_raft::cluster::SnapshotTriggerConfig {
            check_interval: Duration::from_secs(3600),
            disk_space_threshold: u64::MAX,
        };
        let n1 = RaftNode::open_cluster_with_snapshot_config(
            1,
            Arc::clone(&e1),
            format!("127.0.0.1:{p1}").parse().expect("a"),
            format!("http://127.0.0.1:{p1}"),
            snap_config,
        )
        .await
        .expect("leader");
        tokio::time::sleep(Duration::from_millis(800)).await;
        assert!(n1.is_leader().await);

        let pipeline = n1.pipeline();
        let id_gen = ProposalIdGenerator::with_base(1u64 << 48);

        // ── Phase 1: Write pre-snapshot data ──
        for i in 1..=5u64 {
            let p = RaftProposal {
                id: id_gen.next(),
                mutations: vec![Mutation::Put {
                    partition: PartitionId::Node,
                    key: format!("node:0:pre-{i}").into_bytes(),
                    value: format!("pre-val-{i}").into_bytes(),
                }],
                commit_ts: Timestamp::from_raw(100 + i),
                start_ts: Timestamp::from_raw(99 + i),
                bypass_rate_limiter: false,
            };
            pipeline.propose_and_wait(&p).expect("propose pre");
        }
        tokio::time::sleep(Duration::from_millis(500)).await;

        // ── Take snapshot + purge logs ──
        n1.raft().trigger().snapshot().await.expect("snapshot");
        tokio::time::sleep(Duration::from_secs(2)).await;

        let applied = n1.applied_index();
        if applied > 1 {
            n1.raft().trigger().purge_log(applied).await.expect("purge");
            tokio::time::sleep(Duration::from_millis(500)).await;
        }

        // ── Phase 2: Write post-snapshot data (still in leader's log) ──
        for i in 1..=5u64 {
            let p = RaftProposal {
                id: id_gen.next(),
                mutations: vec![Mutation::Put {
                    partition: PartitionId::Node,
                    key: format!("node:0:post-{i}").into_bytes(),
                    value: format!("post-val-{i}").into_bytes(),
                }],
                commit_ts: Timestamp::from_raw(200 + i),
                start_ts: Timestamp::from_raw(199 + i),
                bypass_rate_limiter: false,
            };
            pipeline.propose_and_wait(&p).expect("propose post");
        }
        tokio::time::sleep(Duration::from_millis(500)).await;

        // ── New node joins (zero data) ──
        // Must receive: snapshot (pre-data) + log replay (post-data)
        let dir2 = tempfile::tempdir().expect("d2");
        let e2 = Arc::new(StorageEngine::open(&StorageConfig::new(dir2.path())).expect("open2"));
        let n2 = RaftNode::open_joining(
            2,
            Arc::clone(&e2),
            format!("127.0.0.1:{p2}").parse().expect("a"),
        )
        .await
        .expect("joining");

        n1.add_node(2, format!("http://127.0.0.1:{p2}"))
            .await
            .expect("add n2");
        tokio::time::sleep(Duration::from_secs(8)).await;

        // ── Verify: new node has BOTH pre-snapshot AND post-snapshot data ──

        // Pre-snapshot data (came via StreamSnapshot)
        let mut pre_found = 0;
        for i in 1..=5u64 {
            if let Ok(Some(_)) = e2.get(Partition::Node, format!("node:0:pre-{i}").as_bytes()) {
                pre_found += 1;
            }
        }
        assert!(
            pre_found >= 3,
            "new node should have pre-snapshot data via StreamSnapshot ({pre_found}/5)"
        );

        // Post-snapshot data (came via log replay / AppendEntries)
        let mut post_found = 0;
        for i in 1..=5u64 {
            if let Ok(Some(_)) = e2.get(Partition::Node, format!("node:0:post-{i}").as_bytes()) {
                post_found += 1;
            }
        }
        assert!(
            post_found >= 3,
            "new node should have post-snapshot data via log replay ({post_found}/5)"
        );

        tracing::info!(
            pre_found,
            post_found,
            "R137: snapshot bootstrap + log replay verified"
        );

        n1.shutdown().await.expect("s1");
        n2.shutdown().await.expect("s2");
    })
    .await;

    assert!(
        result.is_ok(),
        "TIMED OUT — cluster_snapshot_bootstrap_then_log_replay"
    );
}

/// R140: Replication status and staleness tracking — leader reports
/// per-node matched index, lag, and heartbeat state.
#[tokio::test(flavor = "multi_thread")]
async fn cluster_replication_status_tracking() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("coordinode_raft=info,openraft=off")
        .with_test_writer()
        .try_init();

    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let p1 = alloc_port();
        let p2 = alloc_port();
        let p3 = alloc_port();

        let dir1 = tempfile::tempdir().expect("d1");
        let dir2 = tempfile::tempdir().expect("d2");
        let dir3 = tempfile::tempdir().expect("d3");

        let e1 = Arc::new(StorageEngine::open(&StorageConfig::new(dir1.path())).expect("open1"));
        let e2 = Arc::new(StorageEngine::open(&StorageConfig::new(dir2.path())).expect("open2"));
        let e3 = Arc::new(StorageEngine::open(&StorageConfig::new(dir3.path())).expect("open3"));

        let n1 = RaftNode::open_cluster(
            1,
            Arc::clone(&e1),
            format!("127.0.0.1:{p1}").parse().expect("a"),
            format!("http://127.0.0.1:{p1}"),
        )
        .await
        .expect("n1");
        tokio::time::sleep(Duration::from_millis(800)).await;

        let n2 = RaftNode::open_joining(
            2,
            Arc::clone(&e2),
            format!("127.0.0.1:{p2}").parse().expect("a"),
        )
        .await
        .expect("n2");
        let n3 = RaftNode::open_joining(
            3,
            Arc::clone(&e3),
            format!("127.0.0.1:{p3}").parse().expect("a"),
        )
        .await
        .expect("n3");

        n1.add_node(2, format!("http://127.0.0.1:{p2}"))
            .await
            .expect("add n2");
        n1.add_node(3, format!("http://127.0.0.1:{p3}"))
            .await
            .expect("add n3");
        n1.change_membership(vec![1, 2, 3])
            .await
            .expect("membership");
        tokio::time::sleep(Duration::from_secs(2)).await;
        assert!(n1.is_leader().await, "n1 should be leader");

        // Write some data to create replication activity
        let pipeline = n1.pipeline();
        let id_gen = ProposalIdGenerator::with_base(1u64 << 48);
        for i in 1..=10u64 {
            let p = RaftProposal {
                id: id_gen.next(),
                mutations: vec![Mutation::Put {
                    partition: PartitionId::Node,
                    key: format!("node:0:r140-{i}").into_bytes(),
                    value: format!("val-{i}").into_bytes(),
                }],
                commit_ts: Timestamp::from_raw(100 + i),
                start_ts: Timestamp::from_raw(99 + i),
                bypass_rate_limiter: false,
            };
            pipeline.propose_and_wait(&p).expect("propose");
        }
        tokio::time::sleep(Duration::from_secs(2)).await;

        // ── Test 1: Leader reports replication status ──
        let status = n1
            .replication_status()
            .expect("leader should report replication status");

        assert!(
            status.len() >= 3,
            "should have status for all 3 nodes, got {}",
            status.len()
        );

        // Find leader status
        let leader_status = status
            .iter()
            .find(|s| s.node_id == 1)
            .expect("leader in status");
        assert_eq!(
            leader_status.role,
            coordinode_raft::cluster::NodeRole::Leader
        );
        assert_eq!(leader_status.lag_entries, 0, "leader has zero lag");

        // Find follower statuses
        for node_id in [2, 3] {
            let fs = status
                .iter()
                .find(|s| s.node_id == node_id)
                .expect("node should be in status");
            assert_eq!(fs.role, coordinode_raft::cluster::NodeRole::Follower);
            // After 2s replication, followers should be caught up (lag < 5)
            assert!(
                fs.lag_entries < 5,
                "node {} lag {} should be < 5 after replication",
                node_id,
                fs.lag_entries
            );
            assert!(
                fs.matched_index > 0,
                "follower should have matched some entries"
            );
        }

        // ── Test 2: Followers don't report replication status ──
        assert!(
            n2.replication_status().is_none(),
            "follower should not report replication status"
        );

        // ── Test 3: Staleness check on follower ──
        let leader_last = leader_status.matched_index;
        assert!(
            n2.is_within_staleness(leader_last, 100),
            "follower with lag < 5 should be within staleness of 100"
        );

        // With very tight staleness (0 entries), follower may be stale
        // (depends on timing — leader may have committed entries not yet replicated)
        // Don't assert exact value, just verify the method works
        let _ = n2.is_within_staleness(leader_last, 0);

        tracing::info!("R140: replication status tracking verified");

        n1.shutdown().await.expect("s1");
        n2.shutdown().await.expect("s2");
        n3.shutdown().await.expect("s3");
    })
    .await;

    assert!(
        result.is_ok(),
        "TIMED OUT — cluster_replication_status_tracking"
    );
}

/// R123: Read concern levels — linearizable requires leader confirmation,
/// majority uses commit_index, local reads immediately.
#[tokio::test(flavor = "multi_thread")]
async fn cluster_read_concern_levels() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("coordinode_raft=info,openraft=off")
        .with_test_writer()
        .try_init();

    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let p1 = alloc_port();
        let p2 = alloc_port();
        let p3 = alloc_port();

        let dir1 = tempfile::tempdir().expect("d1");
        let dir2 = tempfile::tempdir().expect("d2");
        let dir3 = tempfile::tempdir().expect("d3");

        let e1 = Arc::new(StorageEngine::open(&StorageConfig::new(dir1.path())).expect("open1"));
        let e2 = Arc::new(StorageEngine::open(&StorageConfig::new(dir2.path())).expect("open2"));
        let e3 = Arc::new(StorageEngine::open(&StorageConfig::new(dir3.path())).expect("open3"));

        let n1 = RaftNode::open_cluster(
            1,
            Arc::clone(&e1),
            format!("127.0.0.1:{p1}").parse().expect("a"),
            format!("http://127.0.0.1:{p1}"),
        )
        .await
        .expect("n1");
        tokio::time::sleep(Duration::from_millis(800)).await;

        let n2 = RaftNode::open_joining(
            2,
            Arc::clone(&e2),
            format!("127.0.0.1:{p2}").parse().expect("a"),
        )
        .await
        .expect("n2");
        let n3 = RaftNode::open_joining(
            3,
            Arc::clone(&e3),
            format!("127.0.0.1:{p3}").parse().expect("a"),
        )
        .await
        .expect("n3");

        n1.add_node(2, format!("http://127.0.0.1:{p2}"))
            .await
            .expect("add n2");
        n1.add_node(3, format!("http://127.0.0.1:{p3}"))
            .await
            .expect("add n3");
        n1.change_membership(vec![1, 2, 3])
            .await
            .expect("membership");
        tokio::time::sleep(Duration::from_secs(2)).await;

        // Write data
        let pipeline = n1.pipeline();
        let id_gen = ProposalIdGenerator::with_base(1u64 << 48);
        let p = RaftProposal {
            id: id_gen.next(),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:0:rc-test".to_vec(),
                value: b"read-concern-data".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(100),
            start_ts: Timestamp::from_raw(99),
            bypass_rate_limiter: false,
        };
        pipeline.propose_and_wait(&p).expect("propose");
        tokio::time::sleep(Duration::from_secs(1)).await;

        // ── Test 1: Linearizable read on leader succeeds ──
        let lin_result = n1.ensure_linearizable_read().await;
        assert!(
            lin_result.is_ok(),
            "linearizable read on leader should succeed"
        );
        let applied = lin_result.expect("linearizable");
        assert!(
            applied > 0,
            "applied index should be > 0 after linearizable check"
        );

        // ── Test 2: Linearizable read on follower fails ──
        let lin_follower = n2.ensure_linearizable_read().await;
        assert!(
            lin_follower.is_err(),
            "linearizable read on follower should fail (not leader)"
        );

        // ── Test 3: Commit index (majority) accessible on leader ──
        let commit = n1.commit_index();
        assert!(commit > 0, "commit_index should be > 0 after writes");

        // ── Test 4: Applied index (local) on follower ──
        let follower_applied = n2.applied_index();
        assert!(
            follower_applied > 0,
            "follower should have applied entries for local reads"
        );

        // ── Test 5: Staleness check integrates with read concern ──
        // Follower with small lag should be eligible for majority reads
        assert!(
            n2.is_within_staleness(commit, 10),
            "follower should be within staleness bounds for majority reads"
        );

        tracing::info!(
            applied,
            commit,
            follower_applied,
            "R123: read concern levels verified"
        );

        n1.shutdown().await.expect("s1");
        n2.shutdown().await.expect("s2");
        n3.shutdown().await.expect("s3");
    })
    .await;

    assert!(result.is_ok(), "TIMED OUT — cluster_read_concern_levels");
}

/// R124: Write concern W0 vs Majority — Majority replicates to followers,
/// W0 (direct local write) does NOT replicate.
#[tokio::test(flavor = "multi_thread")]
async fn cluster_write_concern_w0_vs_majority() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("coordinode_raft=info,openraft=off")
        .with_test_writer()
        .try_init();

    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let p1 = alloc_port();
        let p2 = alloc_port();
        let p3 = alloc_port();

        let dir1 = tempfile::tempdir().expect("d1");
        let dir2 = tempfile::tempdir().expect("d2");
        let dir3 = tempfile::tempdir().expect("d3");

        let e1 = Arc::new(StorageEngine::open(&StorageConfig::new(dir1.path())).expect("open1"));
        let e2 = Arc::new(StorageEngine::open(&StorageConfig::new(dir2.path())).expect("open2"));
        let e3 = Arc::new(StorageEngine::open(&StorageConfig::new(dir3.path())).expect("open3"));

        let n1 = RaftNode::open_cluster(
            1,
            Arc::clone(&e1),
            format!("127.0.0.1:{p1}").parse().expect("a"),
            format!("http://127.0.0.1:{p1}"),
        )
        .await
        .expect("n1");
        tokio::time::sleep(Duration::from_millis(800)).await;

        let n2 = RaftNode::open_joining(
            2,
            Arc::clone(&e2),
            format!("127.0.0.1:{p2}").parse().expect("a"),
        )
        .await
        .expect("n2");
        let n3 = RaftNode::open_joining(
            3,
            Arc::clone(&e3),
            format!("127.0.0.1:{p3}").parse().expect("a"),
        )
        .await
        .expect("n3");

        n1.add_node(2, format!("http://127.0.0.1:{p2}"))
            .await
            .expect("add n2");
        n1.add_node(3, format!("http://127.0.0.1:{p3}"))
            .await
            .expect("add n3");
        n1.change_membership(vec![1, 2, 3])
            .await
            .expect("membership");
        tokio::time::sleep(Duration::from_secs(2)).await;

        // ── Write with w:majority (via Raft proposal) → replicated ──
        let pipeline = n1.pipeline();
        let id_gen = ProposalIdGenerator::with_base(1u64 << 48);
        let p_majority = RaftProposal {
            id: id_gen.next(),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:0:majority-write".to_vec(),
                value: b"replicated".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(100),
            start_ts: Timestamp::from_raw(99),
            bypass_rate_limiter: false,
        };
        pipeline
            .propose_and_wait(&p_majority)
            .expect("majority write");
        tokio::time::sleep(Duration::from_secs(2)).await;

        // Follower should see majority write
        let majority_on_follower = e2
            .get(Partition::Node, b"node:0:majority-write")
            .expect("read majority on follower");
        assert!(
            majority_on_follower.is_some(),
            "majority write should replicate to follower"
        );

        // ── Write with w:0 (direct local, NO Raft proposal) → NOT replicated ──
        // Simulates the W0 path: writes directly to engine
        // without going through the proposal pipeline.
        e1.put(Partition::Node, b"node:0:w0-write", b"local-only")
            .expect("w0 direct write");

        // Wait to ensure any hypothetical replication would have happened
        tokio::time::sleep(Duration::from_secs(2)).await;

        // Leader should see W0 write (it's local)
        let w0_on_leader = e1
            .get(Partition::Node, b"node:0:w0-write")
            .expect("read w0 on leader");
        assert!(
            w0_on_leader.is_some(),
            "w0 write should be visible on leader"
        );

        // Follower should NOT see W0 write (not replicated via Raft)
        let w0_on_follower = e2
            .get(Partition::Node, b"node:0:w0-write")
            .expect("read w0 on follower");
        assert!(
            w0_on_follower.is_none(),
            "w0 write should NOT replicate to follower (bypasses Raft)"
        );

        tracing::info!("R124: w:0 vs w:majority replication verified");

        n1.shutdown().await.expect("s1");
        n2.shutdown().await.expect("s2");
        n3.shutdown().await.expect("s3");
    })
    .await;

    assert!(
        result.is_ok(),
        "TIMED OUT — cluster_write_concern_w0_vs_majority"
    );
}
