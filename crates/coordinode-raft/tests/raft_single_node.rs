#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration tests for single-node Raft cluster.
//!
//! Tests the full propose → replicate → apply → read cycle through
//! the RaftNode orchestrator. Verifies that data written through the
//! Raft pipeline is durably stored and readable from CoordiNode storage.

use std::sync::Arc;

use coordinode_core::txn::proposal::{
    Mutation, PartitionId, ProposalIdGenerator, ProposalPipeline, RaftProposal,
};
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_raft::cluster::RaftNode;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

fn init_test_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("coordinode_raft=info,openraft=off")
        .with_test_writer()
        .try_init();
}

fn test_engine() -> (tempfile::TempDir, Arc<StorageEngine>) {
    init_test_tracing();
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

/// Full E2E: bootstrap → propose mutations → verify in storage.
#[tokio::test(flavor = "multi_thread")]
async fn e2e_propose_multi_partition_and_read() {
    init_test_tracing();
    let (_dir, engine) = test_engine();
    let engine_read = Arc::clone(&engine);
    let node = RaftNode::single_node(engine).await.expect("bootstrap");

    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    let pipeline = node.pipeline();
    let id_gen = ProposalIdGenerator::new();

    // Create a proposal with mutations across multiple partitions
    let proposal = RaftProposal {
        id: id_gen.next(),
        mutations: vec![
            Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:1:100".to_vec(),
                value: b"alice".to_vec(),
            },
            Mutation::Put {
                partition: PartitionId::Adj,
                key: b"adj:KNOWS:out:100".to_vec(),
                value: b"posting-list-data".to_vec(),
            },
            Mutation::Put {
                partition: PartitionId::EdgeProp,
                key: b"edgeprop:KNOWS:100:200".to_vec(),
                value: b"since:2024".to_vec(),
            },
        ],
        commit_ts: Timestamp::from_raw(1000),
        start_ts: Timestamp::from_raw(999),
        bypass_rate_limiter: false,
    };

    pipeline.propose_and_wait(&proposal).expect("propose");

    // Verify all mutations were applied via plain key reads
    let node_val = engine_read
        .get(Partition::Node, b"node:1:100")
        .expect("read node");
    assert_eq!(node_val.as_deref(), Some(b"alice".as_slice()));

    let adj_val = engine_read
        .get(Partition::Adj, b"adj:KNOWS:out:100")
        .expect("read adj");
    assert_eq!(adj_val.as_deref(), Some(b"posting-list-data".as_slice()));

    let ep_val = engine_read
        .get(Partition::EdgeProp, b"edgeprop:KNOWS:100:200")
        .expect("read edgeprop");
    assert_eq!(ep_val.as_deref(), Some(b"since:2024".as_slice()));

    node.shutdown().await.expect("shutdown");
}

/// Watermark advances with each proposal and wait_for_applied works.
#[tokio::test(flavor = "multi_thread")]
async fn e2e_applied_watermark_tracking() {
    init_test_tracing();
    let (_dir, engine) = test_engine();
    let mut node = RaftNode::single_node(engine).await.expect("bootstrap");

    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    let pipeline = node.pipeline();
    let id_gen = ProposalIdGenerator::new();

    // Initial watermark
    let wm0 = node.applied_index();

    // Submit 3 proposals
    for i in 1..=3u64 {
        let proposal = RaftProposal {
            id: id_gen.next(),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: format!("node:1:{i}").into_bytes(),
                value: format!("val-{i}").into_bytes(),
            }],
            commit_ts: Timestamp::from_raw(100 + i),
            start_ts: Timestamp::from_raw(99 + i),
            bypass_rate_limiter: false,
        };
        pipeline.propose_and_wait(&proposal).expect("propose");
    }

    // Watermark should have advanced by at least 3
    let wm3 = node.applied_index();
    assert!(
        wm3 >= wm0 + 3,
        "watermark should advance by >=3: wm0={wm0}, wm3={wm3}"
    );

    // wait_for_applied should return immediately for past index
    let result = node
        .wait_for_applied(wm3, std::time::Duration::from_secs(5))
        .await
        .expect("should not timeout");
    assert_eq!(result, wm3);

    node.shutdown().await.expect("shutdown");
}

/// Delete mutations work correctly through Raft.
#[tokio::test(flavor = "multi_thread")]
async fn e2e_delete_through_raft() {
    let (_dir, engine) = test_engine();
    let engine_read = Arc::clone(&engine);
    let node = RaftNode::single_node(engine).await.expect("bootstrap");

    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    let pipeline = node.pipeline();
    let id_gen = ProposalIdGenerator::new();

    // Put
    let put = RaftProposal {
        id: id_gen.next(),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: b"node:1:50".to_vec(),
            value: b"to-delete".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(100),
        start_ts: Timestamp::from_raw(99),
        bypass_rate_limiter: false,
    };
    pipeline.propose_and_wait(&put).expect("put");

    // Delete
    let del = RaftProposal {
        id: id_gen.next(),
        mutations: vec![Mutation::Delete {
            partition: PartitionId::Node,
            key: b"node:1:50".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(200),
        start_ts: Timestamp::from_raw(199),
        bypass_rate_limiter: false,
    };
    pipeline.propose_and_wait(&del).expect("delete");

    // After delete, the key should be gone (plain key reads return latest state)
    let after = engine_read
        .get(Partition::Node, b"node:1:50")
        .expect("read");
    assert!(
        after.is_none(),
        "should be deleted after Raft delete mutation"
    );

    node.shutdown().await.expect("shutdown");
}

/// Empty proposal (no mutations) goes through Raft without error.
#[tokio::test(flavor = "multi_thread")]
async fn e2e_empty_proposal_noop() {
    let (_dir, engine) = test_engine();
    let node = RaftNode::single_node(engine).await.expect("bootstrap");

    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    let pipeline = node.pipeline();
    let id_gen = ProposalIdGenerator::new();

    let proposal = RaftProposal {
        id: id_gen.next(),
        mutations: vec![],
        commit_ts: Timestamp::from_raw(100),
        start_ts: Timestamp::from_raw(99),
        bypass_rate_limiter: false,
    };

    // Should succeed as no-op
    pipeline
        .propose_and_wait(&proposal)
        .expect("empty proposal");

    node.shutdown().await.expect("shutdown");
}

/// Dedup: submit two proposals with same ID and same size but DIFFERENT values.
/// Dedup matches on (proposal_id, size_estimate). If size is the same, the second
/// apply is skipped. We verify the FIRST value persists (not overwritten by second).
///
/// To get same size: both values must be same length so size_estimate matches.
#[tokio::test(flavor = "multi_thread")]
async fn e2e_dedup_same_id_same_size_different_value() {
    let (_dir, engine) = test_engine();
    let engine_read = Arc::clone(&engine);
    let node = RaftNode::single_node(engine).await.expect("bootstrap");

    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    let pipeline = node.pipeline();

    let fixed_id = coordinode_core::txn::proposal::ProposalId::from_raw(999);

    // First proposal: value = "AAAA" (4 bytes)
    let proposal1 = RaftProposal {
        id: fixed_id,
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: b"node:1:dedup".to_vec(),
            value: b"AAAA".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(100),
        start_ts: Timestamp::from_raw(99),
        bypass_rate_limiter: false,
    };

    // Second proposal: same id, same key length, same value length → same size_estimate
    // But value = "BBBB" (different content)
    let proposal2 = RaftProposal {
        id: fixed_id,
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: b"node:1:dedup".to_vec(),
            value: b"BBBB".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(100),
        start_ts: Timestamp::from_raw(99),
        bypass_rate_limiter: false,
    };

    // Verify size estimates match (dedup uses this for matching)
    assert_eq!(
        proposal1.size_estimate(),
        proposal2.size_estimate(),
        "size estimates must match for dedup to trigger"
    );

    // First submit → applies, writes "AAAA"
    pipeline
        .propose_and_wait(&proposal1)
        .expect("first propose");

    // Second submit → dedup detects same id+size, SKIPS apply
    pipeline
        .propose_and_wait(&proposal2)
        .expect("second propose (dedup)");

    // Verify: value should be "AAAA" from first apply, NOT "BBBB"
    let val = engine_read
        .get(Partition::Node, b"node:1:dedup")
        .expect("read");
    assert_eq!(
        val.as_deref(),
        Some(b"AAAA".as_slice()),
        "dedup should have skipped second apply — value should be AAAA not BBBB"
    );

    node.shutdown().await.expect("shutdown");
}

/// All 7 partitions are writable through Raft.
#[tokio::test(flavor = "multi_thread")]
async fn e2e_all_partitions() {
    let (_dir, engine) = test_engine();
    let engine_read = Arc::clone(&engine);
    let node = RaftNode::single_node(engine).await.expect("bootstrap");

    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    let pipeline = node.pipeline();
    let id_gen = ProposalIdGenerator::new();
    let ts = Timestamp::from_raw(500);

    // Write to ALL 7 partitions in a single proposal
    let proposal = RaftProposal {
        id: id_gen.next(),
        mutations: vec![
            Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:1:1".to_vec(),
                value: b"node-data".to_vec(),
            },
            Mutation::Put {
                partition: PartitionId::Adj,
                key: b"adj:KNOWS:out:1".to_vec(),
                value: b"adj-data".to_vec(),
            },
            Mutation::Put {
                partition: PartitionId::EdgeProp,
                key: b"edgeprop:KNOWS:1:2".to_vec(),
                value: b"ep-data".to_vec(),
            },
            Mutation::Put {
                partition: PartitionId::Blob,
                key: b"blob:sha256:abc".to_vec(),
                value: b"blob-chunk".to_vec(),
            },
            Mutation::Put {
                partition: PartitionId::BlobRef,
                key: b"blobref:1:photo".to_vec(),
                value: b"sha256:abc".to_vec(),
            },
            Mutation::Put {
                partition: PartitionId::Schema,
                key: b"schema:label:User".to_vec(),
                value: b"schema-data".to_vec(),
            },
            Mutation::Put {
                partition: PartitionId::Idx,
                key: b"idx:User:name:alice".to_vec(),
                value: b"idx-data".to_vec(),
            },
        ],
        commit_ts: ts,
        start_ts: Timestamp::from_raw(499),
        bypass_rate_limiter: false,
    };

    pipeline
        .propose_and_wait(&proposal)
        .expect("all partitions");

    // Verify all 7 partitions via plain key reads
    assert_eq!(
        engine_read
            .get(Partition::Node, b"node:1:1")
            .expect("r")
            .as_deref(),
        Some(b"node-data".as_slice())
    );
    assert_eq!(
        engine_read
            .get(Partition::Adj, b"adj:KNOWS:out:1")
            .expect("r")
            .as_deref(),
        Some(b"adj-data".as_slice())
    );
    assert_eq!(
        engine_read
            .get(Partition::EdgeProp, b"edgeprop:KNOWS:1:2")
            .expect("r")
            .as_deref(),
        Some(b"ep-data".as_slice())
    );
    assert_eq!(
        engine_read
            .get(Partition::Blob, b"blob:sha256:abc")
            .expect("r")
            .as_deref(),
        Some(b"blob-chunk".as_slice())
    );
    assert_eq!(
        engine_read
            .get(Partition::BlobRef, b"blobref:1:photo")
            .expect("r")
            .as_deref(),
        Some(b"sha256:abc".as_slice())
    );
    assert_eq!(
        engine_read
            .get(Partition::Schema, b"schema:label:User")
            .expect("r")
            .as_deref(),
        Some(b"schema-data".as_slice())
    );
    assert_eq!(
        engine_read
            .get(Partition::Idx, b"idx:User:name:alice")
            .expect("r")
            .as_deref(),
        Some(b"idx-data".as_slice())
    );

    node.shutdown().await.expect("shutdown");
}

/// Schema partition user data doesn't collide with raft:* metadata keys.
/// Raft state machine stores its metadata in Schema partition under raft:* prefix.
/// User schema writes (schema:label:*, schema:meta:*) must coexist without corruption.
#[tokio::test(flavor = "multi_thread")]
async fn e2e_schema_partition_no_raft_collision() {
    let (_dir, engine) = test_engine();
    let engine_read = Arc::clone(&engine);
    let node = RaftNode::single_node(engine).await.expect("bootstrap");

    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    let pipeline = node.pipeline();
    let id_gen = ProposalIdGenerator::new();
    let ts = Timestamp::from_raw(100);

    // Write user data to Schema partition
    let proposal = RaftProposal {
        id: id_gen.next(),
        mutations: vec![
            Mutation::Put {
                partition: PartitionId::Schema,
                key: b"schema:label:User".to_vec(),
                value: b"user-schema".to_vec(),
            },
            Mutation::Put {
                partition: PartitionId::Schema,
                key: b"schema:meta:field_interner".to_vec(),
                value: b"interner-data".to_vec(),
            },
        ],
        commit_ts: ts,
        start_ts: Timestamp::from_raw(99),
        bypass_rate_limiter: false,
    };

    pipeline.propose_and_wait(&proposal).expect("schema write");

    // Verify user data is readable
    assert_eq!(
        engine_read
            .get(Partition::Schema, b"schema:label:User")
            .expect("r")
            .as_deref(),
        Some(b"user-schema".as_slice())
    );

    // Verify raft metadata is NOT corrupted — the node should still be functional
    // Submit another proposal to prove the raft state machine still works
    let proposal2 = RaftProposal {
        id: id_gen.next(),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: b"node:1:after-schema".to_vec(),
            value: b"still-working".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(200),
        start_ts: Timestamp::from_raw(199),
        bypass_rate_limiter: false,
    };

    pipeline
        .propose_and_wait(&proposal2)
        .expect("post-schema propose");

    let val = engine_read
        .get(Partition::Node, b"node:1:after-schema")
        .expect("r");
    assert_eq!(val.as_deref(), Some(b"still-working".as_slice()));

    node.shutdown().await.expect("shutdown");
}

/// wait_for_applied on a FUTURE index: blocks until proposal is applied.
#[tokio::test(flavor = "multi_thread")]
async fn e2e_wait_for_future_applied() {
    let (_dir, engine) = test_engine();
    let mut node = RaftNode::single_node(engine).await.expect("bootstrap");

    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    let current = node.applied_index();
    let target = current + 1; // One more than currently applied

    let pipeline = Arc::new(node.pipeline());
    let id_gen = Arc::new(ProposalIdGenerator::new());

    // Spawn a delayed proposal in background
    let pipeline_bg = Arc::clone(&pipeline);
    let id_gen_bg = Arc::clone(&id_gen);
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        let proposal = RaftProposal {
            id: id_gen_bg.next(),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:1:future".to_vec(),
                value: b"arrived".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(300),
            start_ts: Timestamp::from_raw(299),
            bypass_rate_limiter: false,
        };
        // This runs in spawn_blocking because propose_and_wait uses block_in_place
        tokio::task::spawn_blocking(move || {
            pipeline_bg.propose_and_wait(&proposal).expect("bg propose");
        })
        .await
        .expect("join");
    });

    // wait_for_applied should block until the background proposal arrives
    let result = node
        .wait_for_applied(target, std::time::Duration::from_secs(5))
        .await
        .expect("should not timeout — background proposal should arrive");

    assert!(result >= target, "applied index should reach target");

    node.shutdown().await.expect("shutdown");
}

/// wait_for_applied returns Err on timeout when target is unreachable.
#[tokio::test(flavor = "multi_thread")]
async fn e2e_wait_for_applied_timeout() {
    let (_dir, engine) = test_engine();
    let mut node = RaftNode::single_node(engine).await.expect("bootstrap");

    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Ask to wait for an index far in the future with a short timeout
    let result = node
        .wait_for_applied(999_999, std::time::Duration::from_millis(100))
        .await;

    assert!(result.is_err(), "should timeout for unreachable index");

    node.shutdown().await.expect("shutdown");
}

/// Propose after shutdown returns an error, not a panic.
#[tokio::test(flavor = "multi_thread")]
async fn e2e_propose_after_shutdown_returns_error() {
    let (_dir, engine) = test_engine();
    let node = RaftNode::single_node(engine).await.expect("bootstrap");

    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    let pipeline = node.pipeline();
    let id_gen = ProposalIdGenerator::new();

    // Shutdown first
    node.shutdown().await.expect("shutdown");

    // Now try to propose — should return error, NOT panic
    let proposal = RaftProposal {
        id: id_gen.next(),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: b"node:1:after-shutdown".to_vec(),
            value: b"should-fail".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(100),
        start_ts: Timestamp::from_raw(99),
        bypass_rate_limiter: false,
    };

    let result = pipeline.propose_and_wait(&proposal);
    assert!(
        result.is_err(),
        "propose after shutdown should return error"
    );
}

/// CRASH RECOVERY: write data → shutdown → reopen from same storage → verify data survives.
/// This is THE most important test for a consensus engine. If data doesn't survive
/// restart, the entire Raft implementation is broken.
#[tokio::test(flavor = "multi_thread")]
async fn e2e_crash_recovery_data_survives_restart() {
    init_test_tracing();
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().to_path_buf();

    // Phase 1: Write data
    {
        let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            &path,
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        let engine = Arc::new(StorageEngine::open(&config).expect("open"));
        let engine_read = Arc::clone(&engine);
        let node = RaftNode::single_node(engine).await.expect("bootstrap");

        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        let pipeline = node.pipeline();
        let id_gen = ProposalIdGenerator::new();

        // Write 3 proposals across different partitions
        for i in 1..=3u64 {
            let proposal = RaftProposal {
                id: id_gen.next(),
                mutations: vec![
                    Mutation::Put {
                        partition: PartitionId::Node,
                        key: format!("node:1:{i}").into_bytes(),
                        value: format!("value-{i}").into_bytes(),
                    },
                    Mutation::Put {
                        partition: PartitionId::Adj,
                        key: format!("adj:KNOWS:out:{i}").into_bytes(),
                        value: format!("adj-{i}").into_bytes(),
                    },
                ],
                commit_ts: Timestamp::from_raw(100 + i),
                start_ts: Timestamp::from_raw(99 + i),
                bypass_rate_limiter: false,
            };
            pipeline.propose_and_wait(&proposal).expect("propose");
        }

        // Verify data is there BEFORE shutdown
        let val = engine_read.get(Partition::Node, b"node:1:2").expect("read");
        assert_eq!(val.as_deref(), Some(b"value-2".as_slice()));

        // Graceful shutdown
        node.shutdown().await.expect("shutdown");
    }
    // All Arc<StorageEngine> dropped, files flushed

    // Phase 2: Reopen from same storage directory and verify
    {
        let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            &path,
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        let engine = Arc::new(StorageEngine::open(&config).expect("reopen"));
        let engine_read = Arc::clone(&engine);

        // Open (not single_node!) — should detect existing state and resume
        let node = RaftNode::open(1, engine).await.expect("resume");

        tokio::time::sleep(std::time::Duration::from_millis(3000)).await;

        // Verify ALL data survived the restart via plain key reads
        for i in 1..=3u64 {
            let node_val = engine_read
                .get(Partition::Node, format!("node:1:{i}").as_bytes())
                .expect("read node");
            assert_eq!(
                node_val.as_deref(),
                Some(format!("value-{i}").as_bytes()),
                "node data for i={i} did not survive restart"
            );

            let adj_val = engine_read
                .get(Partition::Adj, format!("adj:KNOWS:out:{i}").as_bytes())
                .expect("read adj");
            assert_eq!(
                adj_val.as_deref(),
                Some(format!("adj-{i}").as_bytes()),
                "adj data for i={i} did not survive restart"
            );
        }

        // Phase 3: Verify the node can accept NEW proposals after restart
        let pipeline = node.pipeline();
        let id_gen = ProposalIdGenerator::with_base(1000); // different base to avoid dedup

        let proposal = RaftProposal {
            id: id_gen.next(),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:1:after-restart".to_vec(),
                value: b"new-data".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(500),
            start_ts: Timestamp::from_raw(499),
            bypass_rate_limiter: false,
        };
        pipeline
            .propose_and_wait(&proposal)
            .expect("propose after restart");

        let new_val = engine_read
            .get(Partition::Node, b"node:1:after-restart")
            .expect("read new");
        assert_eq!(new_val.as_deref(), Some(b"new-data".as_slice()));

        // Applied watermark should reflect both pre-restart and post-restart entries
        assert!(
            node.applied_index() > 0,
            "applied watermark should be > 0 after restart"
        );

        node.shutdown().await.expect("shutdown");
    }
}

/// Restart preserves applied watermark — it resumes from where it left off.
#[tokio::test(flavor = "multi_thread")]
async fn e2e_watermark_survives_restart() {
    init_test_tracing();
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().to_path_buf();

    let watermark_before;

    // Phase 1: Write and record watermark
    {
        let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            &path,
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        let engine = Arc::new(StorageEngine::open(&config).expect("open"));
        let node = RaftNode::single_node(engine).await.expect("bootstrap");
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        let pipeline = node.pipeline();
        let id_gen = ProposalIdGenerator::new();
        for i in 1..=5u64 {
            let proposal = RaftProposal {
                id: id_gen.next(),
                mutations: vec![Mutation::Put {
                    partition: PartitionId::Node,
                    key: format!("node:1:{i}").into_bytes(),
                    value: b"x".to_vec(),
                }],
                commit_ts: Timestamp::from_raw(i),
                start_ts: Timestamp::from_raw(i),
                bypass_rate_limiter: false,
            };
            pipeline.propose_and_wait(&proposal).expect("propose");
        }

        watermark_before = node.applied_index();
        assert!(watermark_before > 0, "watermark should be > 0");
        node.shutdown().await.expect("shutdown");
    }

    // Phase 2: Reopen and verify watermark
    {
        let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            &path,
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        let engine = Arc::new(StorageEngine::open(&config).expect("reopen"));
        let node = RaftNode::open(1, engine).await.expect("resume");
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        let watermark_after = node.applied_index();
        assert!(
            watermark_after >= watermark_before,
            "watermark after restart ({watermark_after}) should be >= before ({watermark_before})"
        );

        node.shutdown().await.expect("shutdown");
    }
}

/// Concurrent proposals from multiple threads all succeed.
#[tokio::test(flavor = "multi_thread")]
async fn e2e_concurrent_proposals() {
    let (_dir, engine) = test_engine();
    let engine_read = Arc::clone(&engine);
    let node = RaftNode::single_node(engine).await.expect("bootstrap");

    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    let pipeline = Arc::new(node.pipeline());
    let id_gen = Arc::new(ProposalIdGenerator::new());
    let count = 20u64;

    // Spawn concurrent proposals
    let mut handles = Vec::new();
    for i in 0..count {
        let pipeline = Arc::clone(&pipeline);
        let id_gen = Arc::clone(&id_gen);
        handles.push(tokio::task::spawn_blocking(move || {
            let proposal = RaftProposal {
                id: id_gen.next(),
                mutations: vec![Mutation::Put {
                    partition: PartitionId::Node,
                    key: format!("node:1:c{i}").into_bytes(),
                    value: format!("concurrent-{i}").into_bytes(),
                }],
                commit_ts: Timestamp::from_raw(1000 + i),
                start_ts: Timestamp::from_raw(999 + i),
                bypass_rate_limiter: false,
            };
            pipeline
                .propose_and_wait(&proposal)
                .expect("concurrent propose");
        }));
    }

    // Wait for all
    for h in handles {
        h.await.expect("task join");
    }

    // Verify all writes landed
    for i in 0..count {
        let val = engine_read
            .get(Partition::Node, format!("node:1:c{i}").as_bytes())
            .expect("read");
        assert_eq!(
            val.as_deref(),
            Some(format!("concurrent-{i}").as_bytes()),
            "missing write for i={i}"
        );
    }

    node.shutdown().await.expect("shutdown");
}
