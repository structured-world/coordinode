//! Cluster administration integration tests.
//!
//! These tests exercise the ClusterService gRPC endpoints against real
//! `coordinode` binaries — both the single-node bootstrap and a multi-node
//! cluster formed from three separate processes (the binary supports
//! `--node-id` + `--peers` multi-node bootstrap). Raft-library-level cluster
//! mechanics are covered in `crates/coordinode-raft/tests/raft_cluster.rs`;
//! these tests cover the *service-layer* paths that the library tests bypass.
//!
//! What we cover here:
//!
//! | Test | Endpoint | Scenario |
//! |------|----------|---------|
//! | `decommission_last_voter_rejected_with_failed_precondition` | DecommissionNode | Quorum gate blocks removing the only voter |
//! | `get_cluster_status_standalone_reports_single_leader` | GetClusterStatus | Standalone node self-reports as leader |
//! | `self_decommission_transfers_leadership_off_the_leader` | DecommissionNode | Leader decommissions itself: leadership transfers to a peer + membership shrinks (the `decommission_self` service path) |
//! | `a_standalone_server_with_data_grows_into_a_cluster` | JoinNode | A machine that already holds data restarts with `--peers`, a second machine is added, and the pre-cluster data is on it |
//! | `a_server_that_still_holds_data_is_refused_as_a_joiner` | serve | The same machine started as a joiner refuses at startup and names the empty directory as the fix |
//!
//! ## Running
//!
//! ```bash
//! cargo build -p coordinode-server
//! cargo nextest run -p coordinode-integration --test cluster
//! ```

// Test infrastructure: expect/unwrap panics are intentional — infrastructure
// failures should abort with a clear message, not be silently swallowed.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::time::{Duration, Instant};

use coordinode_integration::harness::{
    CoordinodeProcess, free_port, start_cluster_member_expecting_refusal,
};
use coordinode_integration::proto::admin::{
    DecommissionNodeRequest, GetClusterStatusRequest, JoinNodeRequest, NodeRole,
    cluster_service_client::ClusterServiceClient,
};
use coordinode_integration::proto::query::ExecuteCypherRequest;
use coordinode_integration::proto::replication::ReadConcern;
use tonic::transport::Channel;

/// Decommissioning the only voter in a single-node cluster must fail
/// with FAILED_PRECONDITION — the quorum gate must block it.
///
/// This exercises the full production path:
///   standalone binary → gRPC → ClusterService::decommission_node()
///     → RaftNode::decommission_node(1, false, false)
///       → Phase 0: quorum check → FAILED_PRECONDITION
///
/// A 1-node cluster with node_id=1 cannot remove its only voter — that
/// would leave zero voters and permanently lose the cluster.
#[tokio::test]
async fn decommission_last_voter_rejected_with_failed_precondition() {
    let server = CoordinodeProcess::start().await;
    let mut client = server.cluster_client().await;

    let resp = client
        .decommission_node(DecommissionNodeRequest {
            node_id: 1,
            pruning: false,
            force: false,
            skip_confirmation: false,
        })
        .await;

    let status = resp.expect_err("decommission of last voter must fail");
    assert_eq!(
        status.code(),
        tonic::Code::FailedPrecondition,
        "expected FAILED_PRECONDITION, got {:?}: {}",
        status.code(),
        status.message()
    );
    assert!(
        status.message().contains("quorum")
            || status.message().contains("last voter")
            || status.message().contains("voter"),
        "error message must mention quorum/voter, got: {}",
        status.message()
    );
}

/// A standalone node self-reports as the single leader with term ≥ 1.
///
/// This verifies that:
/// - ClusterService is registered and responding
/// - The embedded single-node Raft has elected itself leader
/// - GetClusterStatus returns a non-empty node list
#[tokio::test]
async fn get_cluster_status_standalone_reports_single_leader() {
    let server = CoordinodeProcess::start().await;
    let mut client = server.cluster_client().await;

    let resp = client
        .get_cluster_status(GetClusterStatusRequest {})
        .await
        .expect("GetClusterStatus must succeed on standalone node");

    let status = resp.into_inner();

    // Single-node cluster: exactly one node, which is the leader.
    assert_eq!(
        status.nodes.len(),
        1,
        "standalone must report exactly 1 node, got {:?}",
        status.nodes
    );
    assert!(
        !status.leader_id.is_empty(),
        "standalone node must report itself as leader"
    );
    assert!(
        status.raft_term >= 1,
        "raft_term must be ≥ 1 after initial election, got {}",
        status.raft_term
    );
}

/// A leader that is asked to decommission *itself* must hand leadership to a
/// peer and then have that new leader remove it from the voter set.
///
/// This is the `ClusterServiceImpl::decommission_self` path — the one the
/// in-process Raft tests bypass by calling `RaftNode::decommission_node()`
/// directly. The full production chain exercised here:
///
///   DecommissionNode(self) on the leader
///     → find_transfer_target() → transfer_leadership_to(peer)
///       → gRPC forward DecommissionNode to the new leader
///         → Phase 0 quorum gate + Phase 2 change_membership (remove node 1)
///
/// Needs a real 3-node cluster across three processes, since the forward is a
/// genuine gRPC call to the peer's advertised address.
#[tokio::test(flavor = "multi_thread")]
async fn self_decommission_transfers_leadership_off_the_leader() {
    // Pre-allocate every member's gRPC port: each member's `--peers` must name
    // the others up front.
    let p1 = free_port();
    let p2 = free_port();
    let p3 = free_port();

    // node 1 bootstraps as the single-voter leader; 2 and 3 start in
    // joining-wait until the leader adds them.
    let n1 = CoordinodeProcess::start_cluster_member(1, p1, &[p2, p3]).await;
    let n2 = CoordinodeProcess::start_cluster_member(2, p2, &[p1, p3]).await;
    let n3 = CoordinodeProcess::start_cluster_member(3, p3, &[p1, p2]).await;

    let mut leader = n1.cluster_client().await;

    // Grow the cluster one member at a time. openraft permits only one
    // membership change in flight, and each JoinNode is two changes
    // (add-learner + background promote-to-voter), so adding 2 and 3
    // concurrently makes the second collide with "configuration change already
    // in progress". Add node 2, wait until it is a voter, then add node 3.
    leader
        .join_node(JoinNodeRequest {
            node_id: 2,
            address: n2.endpoint(),
            pre_seeded: false,
        })
        .await
        .expect("JoinNode(2) must be accepted");
    wait_for_voters(&mut leader, 2, Duration::from_secs(40)).await;

    leader
        .join_node(JoinNodeRequest {
            node_id: 3,
            address: n3.endpoint(),
            pre_seeded: false,
        })
        .await
        .expect("JoinNode(3) must be accepted");
    wait_for_voters(&mut leader, 3, Duration::from_secs(40)).await;

    // Decommission node 1 — the current leader. Drives decommission_self.
    let resp = leader
        .decommission_node(DecommissionNodeRequest {
            node_id: 1,
            pruning: false,
            force: false,
            skip_confirmation: false,
        })
        .await
        .expect("self-decommission of the leader must succeed")
        .into_inner();
    assert!(
        !resp.message.is_empty(),
        "decommission response must carry a status message"
    );

    // Leadership must have moved off node 1 and node 1 must have left the
    // voter set. The surviving leader reports the shrunk membership.
    let (new_leader, members) =
        wait_for_post_decommission_state(&n2, &n3, Duration::from_secs(40)).await;
    assert!(
        matches!(new_leader.as_str(), "2" | "3"),
        "leadership must transfer to a surviving node, got leader_id={new_leader}"
    );
    assert_eq!(
        members.len(),
        2,
        "membership must shrink to the two survivors, got {members:?}"
    );
    assert!(
        !members.contains(&"1".to_string()),
        "node 1 must be removed from the voter set, got {members:?}"
    );
}

/// A single machine that already holds data becomes a replicated one, driven
/// through the binary the way an operator drives it.
///
/// The library tests grow a directory in process; this one takes the whole
/// path an operator takes: a standalone server accumulates data, stops, comes
/// back up with `--node-id` and `--peers`, and a second machine is added with
/// `admin node join`. What the first machine held before the cluster existed
/// has to be on the second one afterwards, read from that member itself
/// rather than from the leader.
#[tokio::test(flavor = "multi_thread")]
async fn a_standalone_server_with_data_grows_into_a_cluster() {
    let n1 = CoordinodeProcess::start().await;
    n1.wait_for_leader(Duration::from_secs(15)).await;
    {
        let mut cypher = n1.cypher_client().await;
        cypher
            .execute_cypher(ExecuteCypherRequest {
                query: "CREATE (n:BeforeTheCluster {id: 1, name: 'alone'}) RETURN n.id".to_string(),
                parameters: std::collections::HashMap::new(),
                read_preference: 0,
                read_concern: None,
                write_concern: None,
                transaction_id: 0,
            })
            .await
            .expect("the standalone server accepts the write");
    }

    // The same machine, now the first member of a cluster, and an empty
    // machine to add to it.
    let p2 = free_port();
    let n1 = n1.restart_as_cluster_member(1, &[p2]).await;
    n1.wait_for_leader(Duration::from_secs(20)).await;
    let n2 = CoordinodeProcess::start_cluster_member(2, p2, &[n1.port]).await;

    let mut leader = n1.cluster_client().await;
    leader
        .join_node(JoinNodeRequest {
            node_id: 2,
            address: n2.endpoint(),
            pre_seeded: false,
        })
        .await
        .expect("JoinNode(2) must be accepted");
    wait_for_voters(&mut leader, 2, Duration::from_secs(40)).await;

    // Read the added member's own copy: SECONDARY preference with a local
    // read concern answers from where the request landed, so an answer here
    // is that member's state and not the leader's.
    let mut follower = n2.cypher_client().await;
    let deadline = Instant::now() + Duration::from_secs(40);
    loop {
        let rows = follower
            .execute_cypher(ExecuteCypherRequest {
                query: "MATCH (n:BeforeTheCluster) RETURN n.name".to_string(),
                parameters: std::collections::HashMap::new(),
                read_preference: 3, // SECONDARY
                read_concern: Some(ReadConcern {
                    level: 1, // LOCAL
                    after_index: 0,
                    at_timestamp: 0,
                }),
                write_concern: None,
                transaction_id: 0,
            })
            .await
            .map(|r| r.into_inner().rows)
            .unwrap_or_default();
        if !rows.is_empty() {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the data the machine held before the cluster existed never reached the member \
             added to it"
        );
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
}

/// A machine that still holds data is refused when it is started as a joiner,
/// and the refusal tells the operator what to do about it.
///
/// This is the other direction of the same rule: data enters a group only
/// through the member the group is formed around. The refusal has to happen
/// where an operator sees it, at startup, naming the empty directory as the
/// fix, rather than as a silent replacement once replication begins.
#[tokio::test(flavor = "multi_thread")]
async fn a_server_that_still_holds_data_is_refused_as_a_joiner() {
    let n1 = CoordinodeProcess::start().await;
    n1.wait_for_leader(Duration::from_secs(15)).await;
    {
        let mut cypher = n1.cypher_client().await;
        cypher
            .execute_cypher(ExecuteCypherRequest {
                query: "CREATE (n:StillMine {id: 1}) RETURN n.id".to_string(),
                parameters: std::collections::HashMap::new(),
                read_preference: 0,
                read_concern: None,
                write_concern: None,
                transaction_id: 0,
            })
            .await
            .expect("the standalone server accepts the write");
    }
    let data_dir = n1.stop_keeping_data().await;

    // The same directory, started as a member joining someone else's group.
    let (status, printed) =
        start_cluster_member_expecting_refusal(2, free_port(), &[free_port()], data_dir.path())
            .await;

    assert!(
        !status.success(),
        "a machine that still holds data must not come up as a joiner"
    );
    assert!(
        printed.contains("cannot join an existing group"),
        "the refusal must say what happened, got: {printed}"
    );
    assert!(
        printed.contains("empty data directory"),
        "the refusal must name the fix, got: {printed}"
    );
}

/// Poll `GetClusterStatus` on `client` until it reports exactly `expected`
/// members, all of them voters (no `Learner`).
async fn wait_for_voters(
    client: &mut ClusterServiceClient<Channel>,
    expected: usize,
    timeout: Duration,
) {
    let deadline = Instant::now() + timeout;
    loop {
        if let Ok(resp) = client.get_cluster_status(GetClusterStatusRequest {}).await {
            let s = resp.into_inner();
            let voters = s
                .nodes
                .iter()
                .filter(|n| n.role != NodeRole::Learner as i32)
                .count();
            if s.nodes.len() == expected && voters == expected {
                return;
            }
        }
        if Instant::now() >= deadline {
            panic!("cluster did not reach {expected} voters within {timeout:?}");
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
}

/// Poll the two surviving members until one of them reports, as leader, a
/// membership that no longer contains node 1. Returns `(leader_id, member_ids)`.
///
/// Only the leader returns a populated node list (`replication_status` is
/// leader-only), so a non-empty list identifies the leader.
async fn wait_for_post_decommission_state(
    a: &CoordinodeProcess,
    b: &CoordinodeProcess,
    timeout: Duration,
) -> (String, Vec<String>) {
    let deadline = Instant::now() + timeout;
    loop {
        for node in [a, b] {
            let mut client = node.cluster_client().await;
            if let Ok(resp) = client.get_cluster_status(GetClusterStatusRequest {}).await {
                let s = resp.into_inner();
                let members: Vec<String> = s.nodes.iter().map(|n| n.node_id.clone()).collect();
                // Leader reports a populated list; wait until node 1 has been
                // removed and a survivor holds leadership.
                if !members.is_empty()
                    && !members.contains(&"1".to_string())
                    && matches!(s.leader_id.as_str(), "2" | "3")
                {
                    return (s.leader_id, members);
                }
            }
        }
        if Instant::now() >= deadline {
            panic!("surviving leader did not report a node-1-free membership within {timeout:?}");
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
}
