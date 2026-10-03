#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! A node whose disk falls below its free-space reserve pauses writes
//! instead of failing an fsync on a full disk: every write is refused with
//! "no space" before anything reaches the log, reads go on, consensus stays
//! alive, and writes resume once space is freed. A follower in that state
//! takes no new entries but keeps following its leader, and catches up
//! afterwards.
//!
//! The reserve is raised above the free space to put a node in that state
//! without filling a real disk; the engine reads free space the same way in
//! both cases.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use coordinode_core::txn::proposal::{
    Mutation, PartitionId, ProposalError, ProposalIdGenerator, ProposalPipeline, RaftProposal,
};
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_raft::cluster::RaftNode;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_test_fixtures::alloc_port;
use openraft::rt::watch::WatchReceiver;

const TEST_TIMEOUT: Duration = Duration::from_secs(90);

fn open_engine(path: &Path) -> Arc<StorageEngine> {
    Arc::new(
        StorageEngine::open(&StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            path,
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]))
        .expect("open engine"),
    )
}

fn write(
    node: &RaftNode,
    ids: &ProposalIdGenerator,
    key: &str,
    commit_ts: u64,
) -> Result<(), ProposalError> {
    let proposal = RaftProposal {
        id: ids.next(),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: key.as_bytes().to_vec(),
            value: b"v".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(commit_ts),
        start_ts: Timestamp::from_raw(commit_ts - 1),
        bypass_rate_limiter: false,
    };
    node.pipeline().propose_and_wait(&proposal).map(|_| ())
}

fn holds(engine: &StorageEngine, key: &str) -> bool {
    engine
        .get(Partition::Node, key.as_bytes())
        .expect("read")
        .is_some()
}

async fn eventually(what: &str, mut check: impl FnMut() -> bool) {
    for _ in 0..300 {
        if check() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("never: {what}");
}

/// Consensus is still running: nothing stopped it on a storage error.
fn consensus_running(node: &RaftNode) -> bool {
    node.raft().metrics().borrow_watched().running_state.is_ok()
}

fn fill(engine: &StorageEngine) {
    engine.space().set_reserve(u64::MAX, u64::MAX);
}

fn free(engine: &StorageEngine) {
    engine.space().set_reserve(0, 0);
}

/// A single node below its reserve refuses writes with "no space", keeps
/// serving what it holds, keeps its consensus running, and takes writes
/// again once space is freed.
#[tokio::test(flavor = "multi_thread")]
async fn a_node_out_of_space_refuses_writes_and_keeps_reading() {
    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let dir = tempfile::tempdir().expect("dir");
        let engine = open_engine(dir.path());
        let node = RaftNode::open(1, Arc::clone(&engine)).await.expect("open");
        let ids = ProposalIdGenerator::with_base(1u64 << 48);
        write(&node, &ids, "node:before", 100).expect("write with room");

        fill(&engine);
        for (i, key) in ["node:full-1", "node:full-2"].into_iter().enumerate() {
            match write(&node, &ids, key, 110 + i as u64) {
                Err(ProposalError::OutOfSpace { min_free_bytes, .. }) => {
                    assert_eq!(min_free_bytes, u64::MAX);
                }
                other => panic!("expected no space, got {other:?}"),
            }
        }
        assert!(holds(&engine, "node:before"), "reads go on");
        assert!(
            !holds(&engine, "node:full-1"),
            "a refused write never lands"
        );
        assert!(
            consensus_running(&node),
            "no storage error reached consensus"
        );

        free(&engine);
        write(&node, &ids, "node:after", 130).expect("writes resume once space is freed");
        assert!(holds(&engine, "node:after"));
        node.shutdown().await.expect("shutdown");
    })
    .await;
    assert!(result.is_ok(), "TIMED OUT");
}

/// In a group of three, a follower below its reserve takes no new entries
/// and stays a follower: the leader keeps its leadership and commits with
/// the other follower. Once space is freed the follower catches up, and no
/// member's consensus stopped on the way.
#[tokio::test(flavor = "multi_thread")]
async fn a_follower_out_of_space_pauses_and_catches_up() {
    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let dirs = [
            tempfile::tempdir().expect("d1"),
            tempfile::tempdir().expect("d2"),
            tempfile::tempdir().expect("d3"),
        ];
        let ports = [alloc_port(), alloc_port(), alloc_port()];
        let engines: Vec<_> = dirs.iter().map(|d| open_engine(d.path())).collect();
        let n1 = RaftNode::open_cluster(
            1,
            Arc::clone(&engines[0]),
            format!("127.0.0.1:{}", ports[0]).parse().expect("addr"),
            format!("http://127.0.0.1:{}", ports[0]),
        )
        .await
        .expect("open 1");
        let n2 = RaftNode::open_joining(
            2,
            Arc::clone(&engines[1]),
            format!("127.0.0.1:{}", ports[1]).parse().expect("addr"),
        )
        .await
        .expect("open 2");
        let n3 = RaftNode::open_joining(
            3,
            Arc::clone(&engines[2]),
            format!("127.0.0.1:{}", ports[2]).parse().expect("addr"),
        )
        .await
        .expect("open 3");
        eventually("member 1 leads", || {
            n1.raft().metrics().borrow_watched().state.is_leader()
        })
        .await;
        n1.add_node(2, format!("http://127.0.0.1:{}", ports[1]))
            .await
            .expect("add 2");
        n1.add_node(3, format!("http://127.0.0.1:{}", ports[2]))
            .await
            .expect("add 3");
        n1.change_membership(vec![1, 2, 3])
            .await
            .expect("three voters");

        let ids = ProposalIdGenerator::with_base(1u64 << 48);
        write(&n1, &ids, "node:before", 100).expect("write with room");
        eventually("member 3 has it", || holds(&engines[2], "node:before")).await;

        fill(&engines[2]);
        for i in 0..20u64 {
            write(&n1, &ids, &format!("node:during-{i}"), 110 + i)
                .expect("the leader commits with the follower that has room");
        }
        // Several election timeouts: a follower that stopped hearing its
        // leader would have stood for election by now.
        tokio::time::sleep(Duration::from_secs(2)).await;
        assert!(
            n1.raft().metrics().borrow_watched().state.is_leader(),
            "the paused follower kept following; leadership never moved"
        );
        assert!(holds(&engines[1], "node:during-19"));
        assert!(
            !holds(&engines[2], "node:during-19"),
            "the paused follower took no entries"
        );
        assert!(
            consensus_running(&n3),
            "the paused follower's consensus runs"
        );
        assert!(
            holds(&engines[2], "node:before"),
            "it still serves what it holds"
        );

        free(&engines[2]);
        eventually("member 3 catches up", || {
            holds(&engines[2], "node:during-19")
        })
        .await;
        for node in [&n1, &n2, &n3] {
            assert!(consensus_running(node), "node {} stopped", node.node_id());
        }
        for node in [n3, n2, n1] {
            node.shutdown().await.expect("shutdown");
        }
    })
    .await;
    assert!(result.is_ok(), "TIMED OUT");
}

/// A leader below its reserve refuses writes before they reach its log, and
/// its followers are unaffected; it writes again once space is freed.
#[tokio::test(flavor = "multi_thread")]
async fn a_leader_out_of_space_refuses_writes_without_stopping() {
    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let dirs = [
            tempfile::tempdir().expect("d1"),
            tempfile::tempdir().expect("d2"),
        ];
        let ports = [alloc_port(), alloc_port()];
        let engines: Vec<_> = dirs.iter().map(|d| open_engine(d.path())).collect();
        let n1 = RaftNode::open_cluster(
            1,
            Arc::clone(&engines[0]),
            format!("127.0.0.1:{}", ports[0]).parse().expect("addr"),
            format!("http://127.0.0.1:{}", ports[0]),
        )
        .await
        .expect("open 1");
        let n2 = RaftNode::open_joining(
            2,
            Arc::clone(&engines[1]),
            format!("127.0.0.1:{}", ports[1]).parse().expect("addr"),
        )
        .await
        .expect("open 2");
        eventually("member 1 leads", || {
            n1.raft().metrics().borrow_watched().state.is_leader()
        })
        .await;
        n1.add_node(2, format!("http://127.0.0.1:{}", ports[1]))
            .await
            .expect("add 2");
        let ids = ProposalIdGenerator::with_base(1u64 << 48);
        write(&n1, &ids, "node:before", 100).expect("write with room");

        fill(&engines[0]);
        assert!(matches!(
            write(&n1, &ids, "node:full", 110),
            Err(ProposalError::OutOfSpace { .. })
        ));
        assert!(holds(&engines[0], "node:before"));
        assert!(consensus_running(&n1) && consensus_running(&n2));

        free(&engines[0]);
        write(&n1, &ids, "node:after", 120).expect("writes resume");
        eventually("the follower has it", || holds(&engines[1], "node:after")).await;
        for node in [n2, n1] {
            node.shutdown().await.expect("shutdown");
        }
    })
    .await;
    assert!(result.is_ok(), "TIMED OUT");
}
