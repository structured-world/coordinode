#![allow(clippy::unwrap_used, clippy::expect_used)]
//! A data directory that started life under a standalone node (single-member
//! Raft, no network) must be able to grow into a three-voter cluster and shrink
//! back to one voter without losing what it held, because that is the only way
//! a single-machine deployment ever becomes a replicated one.
//!
//! Every test has a hard timeout so a stalled election fails instead of hanging.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use coordinode_core::txn::proposal::{
    Mutation, PartitionId, ProposalIdGenerator, ProposalPipeline, RaftProposal,
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

async fn await_leadership(node: &RaftNode) {
    for _ in 0..150 {
        if node.is_leader().await {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(
        node.is_leader().await,
        "node {} never became leader",
        node.node_id()
    );
}

async fn await_value(engine: &StorageEngine, key: &[u8], expected: &[u8]) -> bool {
    for _ in 0..150 {
        if let Ok(Some(v)) = engine.get(Partition::Node, key) {
            if v.as_ref() == expected {
                return true;
            }
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    false
}

/// Proposal ids are the state machine's dedup key, so every write of a test
/// draws from one generator: a second generator on the same base would repeat
/// ids and the writes would be dropped as replays.
fn put(node: &RaftNode, ids: &ProposalIdGenerator, key: &str, value: &str, commit_ts: u64) {
    let proposal = RaftProposal {
        id: ids.next(),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: key.as_bytes().to_vec(),
            value: value.as_bytes().to_vec(),
        }],
        commit_ts: Timestamp::from_raw(commit_ts),
        start_ts: Timestamp::from_raw(commit_ts - 1),
        bypass_rate_limiter: false,
    };
    node.pipeline()
        .propose_and_wait(&proposal)
        .expect("propose");
}

/// The address the cluster's membership holds for `node_id`, as this node sees it.
fn member_addr(node: &RaftNode, node_id: u64) -> Option<String> {
    let rx = node.raft().metrics();
    let metrics = rx.borrow_watched();
    let addr = metrics
        .membership_config
        .membership()
        .nodes()
        .find(|(id, _)| **id == node_id)
        .map(|(_, info)| info.addr.clone());
    addr
}

fn voters(node: &RaftNode) -> Vec<u64> {
    let rx = node.raft().metrics();
    let metrics = rx.borrow_watched();
    let mut ids: Vec<u64> = metrics.membership_config.membership().voter_ids().collect();
    ids.sort_unstable();
    ids
}

/// Reopen a cluster node on the port its peers already dial. The port was free
/// for the downtime window, so a concurrently running test may hold it briefly.
async fn reopen_on_same_port(node_id: u64, engine: &Arc<StorageEngine>, port: u16) -> RaftNode {
    let mut last_err = None;
    let mut reopened = None;
    for _ in 0..20 {
        match RaftNode::open_cluster(
            node_id,
            Arc::clone(engine),
            format!("127.0.0.1:{port}").parse().expect("addr"),
            format!("http://127.0.0.1:{port}"),
        )
        .await
        {
            Ok(node) => {
                reopened = Some(node);
                break;
            }
            Err(e) => {
                last_err = Some(e);
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        }
    }
    assert!(
        reopened.is_some(),
        "reopen node {node_id} on port {port}: {last_err:?}"
    );
    reopened.expect("asserted above")
}

/// Write under a standalone node, then bring the same directory up as the first
/// member of a cluster and add two fresh nodes. Returns the three running nodes.
struct Grown {
    n1: RaftNode,
    n2: RaftNode,
    n3: RaftNode,
    e1: Arc<StorageEngine>,
    e2: Arc<StorageEngine>,
    e3: Arc<StorageEngine>,
    p1: u16,
}

async fn grow_standalone_directory(
    ids: &ProposalIdGenerator,
    path1: &Path,
    path2: &Path,
    path3: &Path,
) -> Grown {
    {
        let engine = open_engine(path1);
        let standalone = RaftNode::open(1, Arc::clone(&engine))
            .await
            .expect("open standalone");
        await_leadership(&standalone).await;
        for i in 1..=3u64 {
            put(
                &standalone,
                ids,
                &format!("node:1:alone-{i}"),
                &format!("alone-{i}"),
                100 + i,
            );
        }
        standalone.shutdown().await.expect("shutdown standalone");
    }

    let p1 = alloc_port();
    let p2 = alloc_port();
    let p3 = alloc_port();
    let e1 = open_engine(path1);
    let e2 = open_engine(path2);
    let e3 = open_engine(path3);

    let n1 = RaftNode::open_cluster(
        1,
        Arc::clone(&e1),
        format!("127.0.0.1:{p1}").parse().expect("addr"),
        format!("http://127.0.0.1:{p1}"),
    )
    .await
    .expect("reopen the standalone directory in cluster mode");
    let n2 = RaftNode::open_joining(
        2,
        Arc::clone(&e2),
        format!("127.0.0.1:{p2}").parse().expect("addr"),
    )
    .await
    .expect("open node 2");
    let n3 = RaftNode::open_joining(
        3,
        Arc::clone(&e3),
        format!("127.0.0.1:{p3}").parse().expect("addr"),
    )
    .await
    .expect("open node 3");

    await_leadership(&n1).await;
    n1.add_node(2, format!("http://127.0.0.1:{p2}"))
        .await
        .expect("add node 2");
    n1.add_node(3, format!("http://127.0.0.1:{p3}"))
        .await
        .expect("add node 3");
    n1.change_membership(vec![1, 2, 3])
        .await
        .expect("three voters");

    Grown {
        n1,
        n2,
        n3,
        e1,
        e2,
        e3,
        p1,
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn standalone_directory_grows_to_three_voters_and_back() {
    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let dir1 = tempfile::tempdir().expect("d1");
        let dir2 = tempfile::tempdir().expect("d2");
        let dir3 = tempfile::tempdir().expect("d3");

        let ids = ProposalIdGenerator::with_base(1u64 << 48);
        let g = grow_standalone_directory(&ids, dir1.path(), dir2.path(), dir3.path()).await;
        assert_eq!(voters(&g.n1), vec![1, 2, 3]);

        // What the directory held before it had peers reaches both new voters.
        for (label, engine) in [("node 2", &g.e2), ("node 3", &g.e3)] {
            for i in 1..=3u64 {
                assert!(
                    await_value(
                        engine,
                        format!("node:1:alone-{i}").as_bytes(),
                        format!("alone-{i}").as_bytes()
                    )
                    .await,
                    "{label} never received standalone-era key {i}"
                );
            }
        }

        // A write made as a cluster replicates as well.
        put(&g.n1, &ids, "node:1:grown", "grown", 200);
        assert!(await_value(&g.e2, b"node:1:grown", b"grown").await);
        assert!(await_value(&g.e3, b"node:1:grown", b"grown").await);

        // Shrink back to the original single voter.
        g.n1.remove_node(3).await.expect("remove node 3");
        g.n1.remove_node(2).await.expect("remove node 2");
        assert_eq!(voters(&g.n1), vec![1]);
        put(&g.n1, &ids, "node:1:shrunk", "shrunk", 300);

        g.n3.shutdown().await.expect("shutdown 3");
        g.n2.shutdown().await.expect("shutdown 2");
        g.n1.shutdown().await.expect("shutdown 1");
        drop(g);

        // The directory serves alone again, with everything it accumulated.
        let engine = open_engine(dir1.path());
        let alone = RaftNode::open(1, Arc::clone(&engine))
            .await
            .expect("reopen standalone after shrinking");
        await_leadership(&alone).await;
        for (key, value) in [
            ("node:1:alone-1", "alone-1"),
            ("node:1:grown", "grown"),
            ("node:1:shrunk", "shrunk"),
        ] {
            let got = engine.get(Partition::Node, key.as_bytes()).expect("read");
            assert_eq!(got.as_deref(), Some(value.as_bytes()), "lost {key}");
        }
        put(&alone, &ids, "node:1:alone-again", "alone-again", 400);
        assert!(await_value(&engine, b"node:1:alone-again", b"alone-again").await);
        alone.shutdown().await.expect("shutdown");
    })
    .await;
    assert!(result.is_ok(), "TIMED OUT — standalone grow/shrink");
}

/// After growing, the original node must stay reachable as a follower: when it
/// goes down and another voter takes over, the new leader has to dial it back.
/// That only works if the membership carries its real address.
#[tokio::test(flavor = "multi_thread")]
async fn first_node_of_a_grown_directory_rejoins_after_failover() {
    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let dir1 = tempfile::tempdir().expect("d1");
        let dir2 = tempfile::tempdir().expect("d2");
        let dir3 = tempfile::tempdir().expect("d3");

        let ids = ProposalIdGenerator::with_base(2u64 << 48);
        let g = grow_standalone_directory(&ids, dir1.path(), dir2.path(), dir3.path()).await;
        put(&g.n1, &ids, "node:1:before-failover", "before", 200);
        assert!(await_value(&g.e2, b"node:1:before-failover", b"before").await);
        assert!(await_value(&g.e3, b"node:1:before-failover", b"before").await);
        let known_addr = member_addr(&g.n2, 1);

        // The stopped node still holds its engine, and the engine its directory lock.
        let Grown {
            n1, n2, n3, e1, p1, ..
        } = g;
        n1.shutdown().await.expect("shutdown node 1");
        drop(n1);
        drop(e1);

        let mut new_leader = None;
        for _ in 0..150 {
            if n2.is_leader().await {
                new_leader = Some(&n2);
                break;
            }
            if n3.is_leader().await {
                new_leader = Some(&n3);
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        let new_leader = new_leader.expect("nodes 2 and 3 elect a leader without node 1");
        put(new_leader, &ids, "node:1:during-outage", "outage", 500);

        let e1 = open_engine(dir1.path());
        let n1 = reopen_on_same_port(1, &e1, p1).await;
        assert!(
            await_value(&e1, b"node:1:during-outage", b"outage").await,
            "node 1 never caught up after rejoining; the cluster holds {known_addr:?} as its \
             address, it listens on http://127.0.0.1:{p1}"
        );
        assert_eq!(
            known_addr.as_deref(),
            Some(format!("http://127.0.0.1:{p1}").as_str()),
            "the cluster must know where node 1 listens"
        );

        n1.shutdown().await.expect("s1");
        n2.shutdown().await.expect("s2");
        n3.shutdown().await.expect("s3");
    })
    .await;
    assert!(result.is_ok(), "TIMED OUT — rejoin after failover");
}

/// Only the member a group is formed around brings data into it.
///
/// A node that already holds data of its own is refused when it is opened to
/// join an existing group: joining means receiving the group's state, so its
/// own would be replaced, and a silent replacement is how a single-machine
/// deployment loses everything it had. Growing that same directory into a
/// group, where it is the first member, stays the supported path (the tests
/// above), and an empty node joins as before.
#[tokio::test(flavor = "multi_thread")]
async fn a_node_holding_data_is_refused_when_it_joins() {
    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let dir = tempfile::tempdir().expect("tempdir");
        let ids = ProposalIdGenerator::with_base(9u64 << 48);

        // The same directory a single-machine deployment accumulates.
        {
            let engine = open_engine(dir.path());
            let standalone = RaftNode::open(1, Arc::clone(&engine))
                .await
                .expect("open standalone");
            await_leadership(&standalone).await;
            put(&standalone, &ids, "node:1:mine", "mine", 101);
            standalone.shutdown().await.expect("shutdown standalone");
        }

        let engine = open_engine(dir.path());
        assert!(
            engine.holds_user_data().expect("read the store"),
            "the standalone run must have left data behind for this test to mean anything"
        );
        let refused = RaftNode::open_joining(
            2,
            Arc::clone(&engine),
            format!("127.0.0.1:{}", alloc_port()).parse().expect("addr"),
        )
        .await;
        assert!(
            matches!(
                refused,
                Err(coordinode_raft::cluster::RaftNodeError::JoinWithLocalData)
            ),
            "a node holding data must be refused when it joins, got {:?}",
            refused.map(|_| "a running joining node")
        );
        assert_eq!(
            engine
                .get(Partition::Node, b"node:1:mine")
                .expect("read")
                .map(|v| v.to_vec()),
            Some(b"mine".to_vec()),
            "the refusal must leave the data where it was"
        );

        // An empty directory is what a joining node is supposed to have.
        let empty_dir = tempfile::tempdir().expect("tempdir");
        let empty = open_engine(empty_dir.path());
        assert!(!empty.holds_user_data().expect("read the empty store"));
        let joining = RaftNode::open_joining(
            3,
            empty,
            format!("127.0.0.1:{}", alloc_port()).parse().expect("addr"),
        )
        .await
        .expect("an empty node joins");
        joining.shutdown().await.expect("shutdown joining");
    })
    .await;
    assert!(
        result.is_ok(),
        "TIMED OUT: a node holding data is refused when it joins"
    );
}

/// A directory written by the non-replicated embedded build opens under the
/// replicated one, as the member a group is formed around.
///
/// That build has no consensus at all: its writes go straight to the engine
/// and its own journal, so nothing in the directory ever passed through a
/// log. Opening it as the first member of a group therefore starts from a
/// state machine that is already populated and a log that is empty, and the
/// question is whether what it holds reaches a member added afterwards. It
/// does, because a member catches up from a snapshot of the state machine
/// rather than by replaying history it never had. The same directory as a
/// joiner is refused, which is the other half of the rule.
#[tokio::test(flavor = "multi_thread")]
async fn an_embedded_directory_opens_as_the_first_member_of_a_group() {
    use coordinode_core::txn::timestamp::TimestampOracle;

    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let dir1 = tempfile::tempdir().expect("tempdir");
        let dir2 = tempfile::tempdir().expect("tempdir");
        let config = |path: &Path| {
            StorageConfig::with_endpoints(vec![EndpointConfig::new(
                "default",
                path,
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            )])
        };

        // What the non-replicated embedded build leaves behind: data applied
        // to the engine directly, with no Raft anywhere in the directory.
        let base = {
            let oracle = Arc::new(TimestampOracle::new());
            let engine = StorageEngine::open_embedded(&config(dir1.path()), Arc::clone(&oracle))
                .expect("open embedded");
            let base = oracle.next().as_raw() + 1_000;
            for i in 1..=3u64 {
                engine
                    .apply_proposal_at(
                        &[Mutation::Put {
                            partition: PartitionId::Node,
                            key: format!("node:1:embedded-{i}").into_bytes(),
                            value: format!("embedded-{i}").into_bytes(),
                        }],
                        base + i,
                    )
                    .expect("apply");
            }
            engine.persist().expect("persist");
            base
        };
        let _ = base;

        // The same directory, now the first member of a group.
        let p1 = alloc_port();
        let p2 = alloc_port();
        let e1 = Arc::new(
            StorageEngine::open_with_oracle(&config(dir1.path()), Arc::new(TimestampOracle::new()))
                .expect("reopen the embedded directory under the replicated build"),
        );
        assert!(
            e1.holds_user_data().expect("read"),
            "the embedded run must have left data behind"
        );
        let n1 = RaftNode::open_cluster(
            1,
            Arc::clone(&e1),
            format!("127.0.0.1:{p1}").parse().expect("addr"),
            format!("http://127.0.0.1:{p1}"),
        )
        .await
        .expect("open the embedded directory as the first member");
        await_leadership(&n1).await;

        let e2 = Arc::new(
            StorageEngine::open_with_oracle(&config(dir2.path()), Arc::new(TimestampOracle::new()))
                .expect("open the empty member"),
        );
        let n2 = RaftNode::open_joining(
            2,
            Arc::clone(&e2),
            format!("127.0.0.1:{p2}").parse().expect("addr"),
        )
        .await
        .expect("an empty member joins");
        n1.add_node(2, format!("http://127.0.0.1:{p2}"))
            .await
            .expect("add node 2");
        n1.change_membership(vec![1, 2]).await.expect("membership");

        for i in 1..=3u64 {
            let key = format!("node:1:embedded-{i}");
            let value = format!("embedded-{i}");
            assert!(
                await_value(&e2, key.as_bytes(), value.as_bytes()).await,
                "{key} never reached the member added after the group was formed: data written \
                 before there was any log has to travel in the snapshot"
            );
        }

        n1.shutdown().await.expect("s1");
        n2.shutdown().await.expect("s2");
    })
    .await;
    assert!(
        result.is_ok(),
        "TIMED OUT: an embedded directory opens as the first member of a group"
    );
}
