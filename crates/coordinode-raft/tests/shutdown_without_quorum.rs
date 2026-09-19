#![allow(clippy::unwrap_used, clippy::expect_used)]
//! A node must always be able to stop, whatever state the rest of the group is
//! in. The last machine of a group to be switched off is a leader without a
//! quorum: nothing it asks of the group can be answered, so nothing in its
//! shutdown may wait for the group.

use std::sync::Arc;
use std::time::Duration;

use coordinode_raft::cluster::RaftNode;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_test_fixtures::alloc_port;

fn open_engine(path: &std::path::Path) -> Arc<StorageEngine> {
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

#[tokio::test(flavor = "multi_thread")]
async fn the_last_leader_of_a_group_shuts_down_without_a_quorum() {
    let result = tokio::time::timeout(Duration::from_secs(90), async {
        let p1 = alloc_port();
        let p2 = alloc_port();
        let p3 = alloc_port();
        let d1 = tempfile::tempdir().expect("d1");
        let d2 = tempfile::tempdir().expect("d2");
        let d3 = tempfile::tempdir().expect("d3");

        let n1 = RaftNode::open_cluster(
            1,
            open_engine(d1.path()),
            format!("127.0.0.1:{p1}").parse().expect("addr"),
            format!("http://127.0.0.1:{p1}"),
        )
        .await
        .expect("n1");
        let n2 = RaftNode::open_joining(
            2,
            open_engine(d2.path()),
            format!("127.0.0.1:{p2}").parse().expect("addr"),
        )
        .await
        .expect("n2");
        let n3 = RaftNode::open_joining(
            3,
            open_engine(d3.path()),
            format!("127.0.0.1:{p3}").parse().expect("addr"),
        )
        .await
        .expect("n3");

        await_leadership(&n1).await;
        n1.add_node(2, format!("http://127.0.0.1:{p2}"))
            .await
            .expect("add 2");
        n1.add_node(3, format!("http://127.0.0.1:{p3}"))
            .await
            .expect("add 3");
        n1.change_membership(vec![1, 2, 3])
            .await
            .expect("three voters");

        // Make node 3 the leader, then take the other two away from it: node 3
        // is now a leader that can reach no quorum.
        n1.transfer_leadership_to(3).await.expect("transfer to 3");
        await_leadership(&n3).await;
        n1.shutdown().await.expect("shutdown 1");
        n2.shutdown().await.expect("shutdown 2");

        let stopped = tokio::time::timeout(Duration::from_secs(15), n3.shutdown()).await;
        assert!(
            stopped.is_ok(),
            "the last leader waited for a quorum that no longer exists and never stopped"
        );
        stopped.expect("bounded").expect("shutdown 3");
    })
    .await;
    assert!(result.is_ok(), "TIMED OUT — shutdown without a quorum");
}

/// The same holds for the question itself: asking a node whether it leads must
/// come back, with `false`, when the group cannot confirm it.
#[tokio::test(flavor = "multi_thread")]
async fn asking_a_quorumless_leader_whether_it_leads_returns() {
    let result = tokio::time::timeout(Duration::from_secs(90), async {
        let p1 = alloc_port();
        let p2 = alloc_port();
        let d1 = tempfile::tempdir().expect("d1");
        let d2 = tempfile::tempdir().expect("d2");

        let n1 = RaftNode::open_cluster(
            1,
            open_engine(d1.path()),
            format!("127.0.0.1:{p1}").parse().expect("addr"),
            format!("http://127.0.0.1:{p1}"),
        )
        .await
        .expect("n1");
        let n2 = RaftNode::open_joining(
            2,
            open_engine(d2.path()),
            format!("127.0.0.1:{p2}").parse().expect("addr"),
        )
        .await
        .expect("n2");

        await_leadership(&n1).await;
        n1.add_node(2, format!("http://127.0.0.1:{p2}"))
            .await
            .expect("add 2");
        n1.change_membership(vec![1, 2]).await.expect("two voters");

        // Node 1 leads a two-voter group; without node 2 it has no quorum.
        n2.shutdown().await.expect("shutdown 2");
        // Long enough for the leader lease to lapse.
        tokio::time::sleep(Duration::from_secs(4)).await;

        let answer = tokio::time::timeout(Duration::from_secs(15), n1.is_leader()).await;
        assert_eq!(
            answer,
            Ok(false),
            "leadership that the group cannot confirm is not leadership"
        );

        let stopped = tokio::time::timeout(Duration::from_secs(15), n1.shutdown()).await;
        assert!(stopped.is_ok(), "node 1 never stopped");
    })
    .await;
    assert!(result.is_ok(), "TIMED OUT — quorumless leadership question");
}
