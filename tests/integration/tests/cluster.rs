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
//! | `a_machine_with_data_grows_to_three_and_shrinks_to_the_quorum_floor` | JoinNode, DecommissionNode | A machine with data grows to three and back to two with its data on both members; the step to one is refused naming the rule |
//! | `a_new_leader_never_reissues_a_node_id` | ExecuteCypher | After the leader goes away the member that takes over creates a node beside the old ones, never over one |
//! | `a_text_index_created_on_the_leader_serves_searches_on_a_follower` | ExecuteCypher | A text index defined on the leader registers on the follower when it applies there and answers searches from the follower's own index |
//! | `constraints_held_before_the_cluster_bind_every_member_after_a_leader_change` | ExecuteCypher, ListConstraints | Constraints that reached the members through the base snapshot are listed active on the new leader and refuse the writes that break them |
//! | `a_transaction_runs_through_any_member` | ExecuteCypher, CommitTransaction | A transaction started on a follower under the client's ID opens at the leader, takes statements through two members and commits through a follower; follower answers name the leader |
//! | `a_write_through_a_follower_runs_at_the_leader` | ExecuteCypher | An autonomous write on a follower runs at the leader, one hop, naming the leader; the leader adds no hint |
//! | `a_session_on_a_follower_works_on_the_leaders_transaction` | Session | A session on a follower fills the leader's transaction, and a session on another member continues and commits it |
//! | `a_member_without_its_vote_is_refused_and_replaced` | serve, ReplaceNode | A member that lost every copy of its vote does not start; the leader replaces it with a new node, a repeated request changes nothing |
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
///       → quorum gate → FAILED_PRECONDITION
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
///         → quorum gate + change_membership (remove node 1)
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
            address: n2.member_addr(),
            pre_seeded: false,
            node_identity: n2.member_identity(),
        })
        .await
        .expect("JoinNode(2) must be accepted");
    wait_for_voters(&mut leader, 2, Duration::from_secs(40)).await;

    leader
        .join_node(JoinNodeRequest {
            node_id: 3,
            address: n3.member_addr(),
            pre_seeded: false,
            node_identity: n3.member_identity(),
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
                ..Default::default()
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
            address: n2.member_addr(),
            pre_seeded: false,
            node_identity: n2.member_identity(),
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
                    ..Default::default()
                }),
                write_concern: None,
                ..Default::default()
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

/// The round trip an operator makes through the binary: one machine with data
/// becomes three, is shrunk back to the two a group needs, and what it held
/// at the start is still there. The step below two is refused, where the
/// operator sees it, naming the rule.
///
/// Growing is only half the promise. The way back is where data is easiest to
/// lose: the member that leaves held a copy, and the ones that stay have to be
/// complete members rather than survivors with whatever replication happened
/// to deliver. A CE group keeps at least two voters, so shrinking to one is
/// not a step the cluster takes, and asking for it must change nothing.
#[tokio::test(flavor = "multi_thread")]
async fn a_machine_with_data_grows_to_three_and_shrinks_to_the_quorum_floor() {
    let n1 = CoordinodeProcess::start().await;
    n1.wait_for_leader(Duration::from_secs(15)).await;
    {
        let mut cypher = n1.cypher_client().await;
        cypher
            .execute_cypher(ExecuteCypherRequest {
                query: "CREATE (n:Durable {id: 1, name: 'written alone'}) RETURN n.id".to_string(),
                parameters: std::collections::HashMap::new(),
                read_preference: 0,
                read_concern: None,
                write_concern: None,
                ..Default::default()
            })
            .await
            .expect("the standalone server accepts the write");
    }

    let p2 = free_port();
    let p3 = free_port();
    let n1 = n1.restart_as_cluster_member(1, &[p2, p3]).await;
    n1.wait_for_leader(Duration::from_secs(20)).await;
    let n2 = CoordinodeProcess::start_cluster_member(2, p2, &[n1.port, p3]).await;
    let n3 = CoordinodeProcess::start_cluster_member(3, p3, &[n1.port, p2]).await;

    // One membership change at a time: a JoinNode is two of them, so adding
    // both at once collides with the change already in flight.
    let mut leader = n1.cluster_client().await;
    for (id, member) in [(2u64, &n2), (3u64, &n3)] {
        leader
            .join_node(JoinNodeRequest {
                node_id: id,
                address: member.member_addr(),
                pre_seeded: false,
                node_identity: member.member_identity(),
            })
            .await
            .unwrap_or_else(|e| panic!("JoinNode({id}) must be accepted: {e}"));
        // Adding member `id` brings the voter count to `id`: node 1 was
        // already there, and the members are added in order.
        wait_for_voters(&mut leader, id as usize, Duration::from_secs(40)).await;
    }

    // Back down to two, removing the last member added. Node 1 keeps the
    // leadership it has, so this is the shrink an operator performs when a
    // machine goes away rather than a leadership handover.
    let decommission = |node_id: u64| DecommissionNodeRequest {
        node_id,
        pruning: false,
        force: false,
        skip_confirmation: false,
    };
    leader
        .decommission_node(decommission(3))
        .await
        .unwrap_or_else(|e| panic!("decommission of 3 must succeed: {e}"));
    wait_for_voters(&mut leader, 2, Duration::from_secs(40)).await;

    // One more would leave a single voter. The refusal comes back as a
    // precondition, names the rule, and leaves the membership as it was.
    let refused = leader
        .decommission_node(decommission(2))
        .await
        .expect_err("a two-voter group must refuse to shrink to one");
    assert_eq!(
        refused.code(),
        tonic::Code::FailedPrecondition,
        "{refused:?}"
    );
    assert!(
        refused.message().contains("minimum 2 required"),
        "the refusal must name the rule, got: {}",
        refused.message()
    );
    wait_for_voters(&mut leader, 2, Duration::from_secs(10)).await;

    // Both members that stayed still answer with what node 1 held before any
    // of this, each read from itself: the leader as PRIMARY, the follower as
    // SECONDARY with a local read concern, which answers from where the
    // request landed.
    for (id, node, read_preference) in [(1u64, &n1, 0), (2u64, &n2, 3)] {
        let mut cypher = node.cypher_client().await;
        let rows = cypher
            .execute_cypher(ExecuteCypherRequest {
                query: "MATCH (n:Durable) RETURN n.name".to_string(),
                parameters: std::collections::HashMap::new(),
                read_preference,
                read_concern: Some(ReadConcern {
                    level: 1, // LOCAL
                    ..Default::default()
                }),
                write_concern: None,
                ..Default::default()
            })
            .await
            .unwrap_or_else(|e| panic!("member {id} answers: {e}"))
            .into_inner()
            .rows;
        assert_eq!(
            rows.len(),
            1,
            "member {id}: the data the machine held before the cluster existed must survive"
        );
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
                ..Default::default()
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

/// The refusal holds after a crash as well: a machine killed right after an
/// acknowledged write still holds that write, and still may not join.
///
/// A graceful stop flushes storage before exiting, so the clean-stop test
/// above cannot tell a store that holds the write in its files from one that
/// holds it only in the log it replays on open.
#[tokio::test(flavor = "multi_thread")]
async fn a_server_killed_after_a_write_is_refused_as_a_joiner() {
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
                ..Default::default()
            })
            .await
            .expect("the standalone server accepts the write");
    }
    let data_dir = n1.kill_keeping_data();

    let (status, printed) =
        start_cluster_member_expecting_refusal(2, free_port(), &[free_port()], data_dir.path())
            .await;

    assert!(
        !status.success(),
        "a machine that holds an acknowledged write must not come up as a joiner"
    );
    assert!(
        printed.contains("cannot join an existing group"),
        "the refusal must say what happened, got: {printed}"
    );
}

/// A member of a group that holds the group's data comes back as that member
/// when its process restarts, whether it was stopped or killed: holding data
/// refuses only a node that is not a member yet.
#[tokio::test(flavor = "multi_thread")]
async fn a_member_restarts_into_its_group_with_its_data() {
    let (p1, p2, p3) = (free_port(), free_port(), free_port());
    let n1 = CoordinodeProcess::start_cluster_member(1, p1, &[p2, p3]).await;
    let n2 = CoordinodeProcess::start_cluster_member(2, p2, &[p1, p3]).await;
    let n3 = CoordinodeProcess::start_cluster_member(3, p3, &[p1, p2]).await;
    let mut leader = n1.cluster_client().await;
    for (id, member) in [(2, &n2), (3, &n3)] {
        leader
            .join_node(JoinNodeRequest {
                node_id: id,
                address: member.member_addr(),
                pre_seeded: false,
                node_identity: member.member_identity(),
            })
            .await
            .expect("JoinNode must be accepted");
        wait_for_voters(
            &mut leader,
            usize::try_from(id).expect("small"),
            Duration::from_secs(40),
        )
        .await;
    }
    cypher_on(&n1, "CREATE (:Restart {v: 'before'})")
        .await
        .expect("the leader writes");

    let n2 = n2.restart_member(2, &[p1, p3], &[], false).await;
    let n3 = n3.restart_member(3, &[p1, p2], &[], true).await;
    // Leadership may have moved while members were away: whichever member
    // leads takes the write.
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let mut wrote = false;
        for member in [&n1, &n2, &n3] {
            if cypher_on(member, "CREATE (:Restart {v: 'after'})")
                .await
                .is_ok()
            {
                wrote = true;
                break;
            }
        }
        if wrote {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the group never wrote with its members back"
        );
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
    for member in [&n2, &n3] {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            let rows = member
                .cypher_client()
                .await
                .execute_cypher(ExecuteCypherRequest {
                    query: "MATCH (n:Restart) RETURN n.v".to_string(),
                    parameters: std::collections::HashMap::new(),
                    read_preference: 5, // NEAREST
                    read_concern: Some(ReadConcern {
                        level: 1, // LOCAL
                        ..Default::default()
                    }),
                    write_concern: None,
                    ..Default::default()
                })
                .await
                .map(|r| r.into_inner().rows.len());
            if rows.as_ref().is_ok_and(|n| *n == 2) {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "member on {} never caught up: {rows:?}",
                member.port
            );
            tokio::time::sleep(Duration::from_millis(300)).await;
        }
    }
    // Leadership may have moved during the restarts, and only the leader
    // reports the membership, so every member is asked.
    let deadline = Instant::now() + Duration::from_secs(10);
    'members: loop {
        for member in [&n1, &n2, &n3] {
            let mut client = member.cluster_client().await;
            if let Ok(resp) = client.get_cluster_status(GetClusterStatusRequest {}).await {
                let s = resp.into_inner();
                let voters = s
                    .nodes
                    .iter()
                    .filter(|n| n.role != NodeRole::Learner as i32)
                    .count();
                if s.nodes.len() == 3 && voters == 3 {
                    break 'members;
                }
            }
        }
        assert!(
            Instant::now() < deadline,
            "no member reported 3 voters after the restarts"
        );
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
}

/// Run `query` on `node` as a primary read or write.
/// `query` answered by `node` itself, from what it has applied.
async fn local_read(
    node: &CoordinodeProcess,
    query: &str,
) -> Result<Vec<coordinode_integration::proto::query::Row>, tonic::Status> {
    node.cypher_client()
        .await
        .execute_cypher(ExecuteCypherRequest {
            query: query.to_string(),
            parameters: std::collections::HashMap::new(),
            read_preference: 5, // NEAREST
            read_concern: Some(ReadConcern {
                level: 1, // LOCAL
                ..Default::default()
            }),
            write_concern: None,
            ..Default::default()
        })
        .await
        .map(|r| r.into_inner().rows)
}

/// A follower killed while its index copy is in use comes back with that
/// copy unfit, and checks it itself: the shared check runs where writes are
/// taken and proves only that member's copy, so the leader records none.
/// The follower answers lookups from its records meanwhile, and its copy
/// serves again once its own check ends.
#[tokio::test(flavor = "multi_thread")]
async fn a_follower_checks_its_own_index_copy_after_a_crash() {
    use coordinode_integration::proto::common::property_value::Value as Pv;
    let (p1, p2, p3) = (free_port(), free_port(), free_port());
    let n1 = CoordinodeProcess::start_cluster_member(1, p1, &[p2, p3]).await;
    let n2 = CoordinodeProcess::start_cluster_member(2, p2, &[p1, p3]).await;
    let n3 = CoordinodeProcess::start_cluster_member(3, p3, &[p1, p2]).await;
    let mut leader = n1.cluster_client().await;
    for (id, member) in [(2, &n2), (3, &n3)] {
        leader
            .join_node(JoinNodeRequest {
                node_id: id,
                address: member.member_addr(),
                pre_seeded: false,
                node_identity: member.member_identity(),
            })
            .await
            .expect("JoinNode must be accepted");
        wait_for_voters(
            &mut leader,
            usize::try_from(id).expect("small"),
            Duration::from_secs(40),
        )
        .await;
    }
    cypher_on(&n1, "CREATE INDEX g_code ON :G(code)")
        .await
        .expect("index");
    cypher_on(&n1, "CREATE (:G {code: 'a'})")
        .await
        .expect("node");
    let lookup = "MATCH (g:G {code: 'a'}) RETURN g.code";
    let deadline = Instant::now() + Duration::from_secs(30);
    while !local_read(&n2, lookup)
        .await
        .is_ok_and(|rows| rows.len() == 1)
    {
        assert!(
            Instant::now() < deadline,
            "the follower never applied the node"
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    // A graceful restart puts the follower's copy on disk, as a copy that
    // has served for a while is; a copy only in memory is rebuilt whole from
    // the log by a crash, and nothing of it is left to check.
    let n2 = n2.restart_member(2, &[p1, p3], &[], false).await;
    let flag = |rows: &[coordinode_integration::proto::query::Row]| {
        rows.iter().any(|row| {
            row.values
                .first()
                .and_then(|v| v.value.as_ref())
                .is_some_and(|v| matches!(v, Pv::BoolValue(true)))
        })
    };
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let read = local_read(&n2, lookup).await;
        let in_use = local_read(&n2, "CALL db.indexCopies() YIELD inUse RETURN inUse")
            .await
            .is_ok_and(|rows| flag(&rows));
        if read.is_ok_and(|rows| rows.len() == 1) && in_use {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the follower's copy never came into use"
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    // Killed with its copy in use: its guard stays open.
    let n2 = n2.restart_member(2, &[p1, p3], &[], true).await;
    // A local read serves what this member has applied, and a member killed
    // applies its log again after it starts: until it has, it holds neither
    // the record nor its index entry. Wait for the record, read by a scan
    // that uses no index, before asking what the lookup answers.
    let deadline = Instant::now() + Duration::from_secs(30);
    while !local_read(&n2, "MATCH (g:G) RETURN g.code")
        .await
        .is_ok_and(|rows| rows.len() == 1)
    {
        assert!(
            Instant::now() < deadline,
            "the follower never applied its log again"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        assert_eq!(
            local_read(&n2, lookup)
                .await
                .expect("the follower answers")
                .len(),
            1,
            "answered while its copy is checked"
        );
        let copies = local_read(&n2, "CALL db.indexCopies() YIELD unfit RETURN unfit")
            .await
            .expect("the follower's copies");
        if !copies.is_empty() && !flag(&copies) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the follower's copy never served again: {copies:?}"
        );
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
    // Read where the checks are replicated to rather than at the leader:
    // the follower's kill may have moved leadership off n1.
    let checks = local_read(&n1, "CALL db.indexChecks()")
        .await
        .expect("the recorded checks");
    assert!(
        checks.is_empty(),
        "the follower checked its own copy; the leader recorded no check: {checks:?}"
    );
    drop((n1, n2, n3));
}

async fn cypher_on(
    node: &CoordinodeProcess,
    query: &str,
) -> Result<Vec<coordinode_integration::proto::query::Row>, tonic::Status> {
    node.cypher_client()
        .await
        .execute_cypher(ExecuteCypherRequest {
            query: query.to_string(),
            parameters: std::collections::HashMap::new(),
            read_preference: 0,
            read_concern: None,
            write_concern: None,
            ..Default::default()
        })
        .await
        .map(|r| r.into_inner().rows)
}

/// [`cypher_on`] for a write: its receipt's commit time.
async fn cypher_write_on(node: &CoordinodeProcess, query: &str) -> Result<u64, tonic::Status> {
    node.cypher_client()
        .await
        .execute_cypher(ExecuteCypherRequest {
            query: query.to_string(),
            ..Default::default()
        })
        .await
        .map(|r| {
            r.into_inner()
                .stats
                .and_then(|s| s.receipt)
                .map_or(0, |receipt| receipt.operation_time)
        })
}

/// [`cypher_on`] for a read that depends on the commit at `after_ts`, waited
/// for up to ten seconds.
async fn cypher_after_on(
    node: &CoordinodeProcess,
    query: &str,
    after_ts: u64,
) -> Result<Vec<coordinode_integration::proto::query::Row>, tonic::Status> {
    node.cypher_client()
        .await
        .execute_cypher(ExecuteCypherRequest {
            query: query.to_string(),
            read_concern: Some(coordinode_integration::proto::replication::ReadConcern {
                level: 2, // MAJORITY
                after_ts,
                wait_ms: 10_000,
                ..Default::default()
            }),
            ..Default::default()
        })
        .await
        .map(|r| r.into_inner().rows)
}

/// A new leader never hands out a NodeId the old one already used. The
/// identifiers a leader issues come from ranges the log granted it; a member
/// that takes over has seen those grants and draws above them, so the node it
/// creates is added beside the old ones instead of overwriting one.
#[tokio::test(flavor = "multi_thread")]
async fn a_new_leader_never_reissues_a_node_id() {
    let (p1, p2, p3) = (free_port(), free_port(), free_port());
    let n1 = CoordinodeProcess::start_cluster_member(1, p1, &[p2, p3]).await;
    let n2 = CoordinodeProcess::start_cluster_member(2, p2, &[p1, p3]).await;
    let n3 = CoordinodeProcess::start_cluster_member(3, p3, &[p1, p2]).await;

    let mut leader = n1.cluster_client().await;
    for (id, member) in [(2, &n2), (3, &n3)] {
        leader
            .join_node(JoinNodeRequest {
                node_id: id,
                address: member.member_addr(),
                pre_seeded: false,
                node_identity: member.member_identity(),
            })
            .await
            .expect("JoinNode must be accepted");
        wait_for_voters(
            &mut leader,
            usize::try_from(id).expect("small"),
            Duration::from_secs(40),
        )
        .await;
    }
    for i in 0..5 {
        cypher_on(&n1, &format!("CREATE (:Handover {{v: 'before-{i}'}})"))
            .await
            .expect("the first leader accepts the write");
    }

    // The first leader goes away; one of the others takes over.
    drop(n1);
    let deadline = Instant::now() + Duration::from_secs(40);
    let (new_leader, written) = loop {
        let mut led = None;
        for member in [&n2, &n3] {
            if let Ok(written) = cypher_write_on(member, "CREATE (:Handover {v: 'after'})").await {
                led = Some((member, written));
                break;
            }
        }
        if let Some(led) = led {
            break led;
        }
        assert!(Instant::now() < deadline, "no member took over");
        tokio::time::sleep(Duration::from_millis(300)).await;
    };

    // The read depends on the write it checks for, so it carries it.
    let rows = cypher_after_on(new_leader, "MATCH (n:Handover) RETURN id(n), n.v", written)
        .await
        .expect("read back");
    let mut ids = Vec::new();
    let mut values = Vec::new();
    for row in rows {
        use coordinode_integration::proto::common::property_value::Value as Pv;
        match (&row.values[0].value, &row.values[1].value) {
            (Some(Pv::IntValue(id)), Some(Pv::StringValue(v))) => {
                ids.push(*id);
                values.push(v.clone());
            }
            other => panic!("unexpected row {other:?}"),
        }
    }
    values.sort();
    assert_eq!(
        values,
        [
            "after", "before-0", "before-1", "before-2", "before-3", "before-4"
        ],
        "every node the first leader wrote survives and the new one is added"
    );
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), 6, "six distinct ids");
}

/// A unique index outlives the leader that created it: its definition and
/// its entries travel in the log, so the member that takes over refuses a
/// duplicate of a value the first leader indexed and finds that value
/// through the index.
#[tokio::test(flavor = "multi_thread")]
async fn a_unique_index_holds_across_a_leader_change() {
    let (p1, p2, p3) = (free_port(), free_port(), free_port());
    let n1 = CoordinodeProcess::start_cluster_member(1, p1, &[p2, p3]).await;
    let n2 = CoordinodeProcess::start_cluster_member(2, p2, &[p1, p3]).await;
    let n3 = CoordinodeProcess::start_cluster_member(3, p3, &[p1, p2]).await;

    let mut leader = n1.cluster_client().await;
    for (id, member) in [(2, &n2), (3, &n3)] {
        leader
            .join_node(JoinNodeRequest {
                node_id: id,
                address: member.member_addr(),
                pre_seeded: false,
                node_identity: member.member_identity(),
            })
            .await
            .expect("JoinNode must be accepted");
        wait_for_voters(
            &mut leader,
            usize::try_from(id).expect("small"),
            Duration::from_secs(40),
        )
        .await;
    }
    cypher_on(&n1, "CREATE UNIQUE INDEX account_email ON :Account(email)")
        .await
        .expect("the first leader creates the index");
    cypher_on(&n1, "CREATE (:Account {email: 'a@x'})")
        .await
        .expect("the first leader accepts the write");

    drop(n1);
    let deadline = Instant::now() + Duration::from_secs(40);
    let new_leader = loop {
        let mut led = None;
        for member in [&n2, &n3] {
            if cypher_on(member, "CREATE (:Account {email: 'b@x'})")
                .await
                .is_ok()
            {
                led = Some(member);
                break;
            }
        }
        if let Some(member) = led {
            break member;
        }
        assert!(Instant::now() < deadline, "no member took over");
        tokio::time::sleep(Duration::from_millis(300)).await;
    };

    let refused = cypher_on(new_leader, "CREATE (:Account {email: 'a@x'})")
        .await
        .expect_err("a duplicate of a value the first leader indexed");
    assert!(
        refused.message().contains("unique constraint"),
        "got: {}",
        refused.message()
    );
    let rows = cypher_on(
        new_leader,
        "MATCH (a:Account) WHERE a.email = 'a@x' RETURN count(a)",
    )
    .await
    .expect("read back");
    use coordinode_integration::proto::common::property_value::Value as Pv;
    assert!(
        matches!(rows[0].values[0].value, Some(Pv::IntValue(1))),
        "the indexed row is found once: {rows:?}"
    );
}

/// A text index created on the leader is registered on a follower when the
/// definition applies there, and that follower answers `text_match()` from its
/// own index with the documents written before and after the definition.
#[tokio::test(flavor = "multi_thread")]
async fn a_text_index_created_on_the_leader_serves_searches_on_a_follower() {
    let (p1, p2) = (free_port(), free_port());
    let n1 = CoordinodeProcess::start_cluster_member(1, p1, &[p2]).await;
    let n2 = CoordinodeProcess::start_cluster_member(2, p2, &[p1]).await;
    let mut leader = n1.cluster_client().await;
    leader
        .join_node(JoinNodeRequest {
            node_id: 2,
            address: n2.member_addr(),
            pre_seeded: false,
            node_identity: n2.member_identity(),
        })
        .await
        .expect("JoinNode(2) must be accepted");
    wait_for_voters(&mut leader, 2, Duration::from_secs(40)).await;

    for statement in [
        "CREATE (:Article {body: 'rust before the index'})",
        "CREATE TEXT INDEX article_body ON :Article(body)",
        "CREATE (:Article {body: 'rust after the index'})",
    ] {
        cypher_on(&n1, statement)
            .await
            .unwrap_or_else(|e| panic!("the leader runs {statement:?}: {e}"));
    }

    let mut follower = n2.cypher_client().await;
    let deadline = Instant::now() + Duration::from_secs(40);
    loop {
        let found = follower
            .execute_cypher(ExecuteCypherRequest {
                query: "MATCH (a:Article) WHERE text_match(a.body, 'rust') RETURN a.body"
                    .to_string(),
                parameters: std::collections::HashMap::new(),
                read_preference: 3, // SECONDARY
                read_concern: Some(ReadConcern {
                    level: 1, // LOCAL
                    ..Default::default()
                }),
                write_concern: None,
                ..Default::default()
            })
            .await
            .map(|r| r.into_inner().rows.len());
        if matches!(found, Ok(2)) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the follower never served the replicated text index: {found:?}"
        );
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
}

/// Constraints a standalone machine held before it became a cluster reach the
/// members added to it, which learn the pre-cluster state from a snapshot
/// rather than from the log: the member that takes over after the first
/// leader goes away lists them active with the index they own, and refuses
/// the writes that break them.
#[tokio::test(flavor = "multi_thread")]
async fn constraints_held_before_the_cluster_bind_every_member_after_a_leader_change() {
    use coordinode_integration::proto::v2::graph::{ConstraintState, ListConstraintsRequest};

    let n1 = CoordinodeProcess::start().await;
    n1.wait_for_leader(Duration::from_secs(15)).await;
    for statement in [
        "CREATE CONSTRAINT account_email FOR (a:Account) REQUIRE a.email IS UNIQUE",
        "CREATE CONSTRAINT account_name FOR (a:Account) REQUIRE a.name IS NOT NULL",
        "CREATE (:Account {email: 'a@x', name: 'first'})",
    ] {
        cypher_on(&n1, statement)
            .await
            .unwrap_or_else(|e| panic!("the standalone server runs {statement:?}: {e}"));
    }

    let (p2, p3) = (free_port(), free_port());
    let n1 = n1.restart_as_cluster_member(1, &[p2, p3]).await;
    n1.wait_for_leader(Duration::from_secs(20)).await;
    let n2 = CoordinodeProcess::start_cluster_member(2, p2, &[n1.port, p3]).await;
    let n3 = CoordinodeProcess::start_cluster_member(3, p3, &[n1.port, p2]).await;
    let mut leader = n1.cluster_client().await;
    for (id, member) in [(2u64, &n2), (3u64, &n3)] {
        leader
            .join_node(JoinNodeRequest {
                node_id: id,
                address: member.member_addr(),
                pre_seeded: false,
                node_identity: member.member_identity(),
            })
            .await
            .unwrap_or_else(|e| panic!("JoinNode({id}) must be accepted: {e}"));
        wait_for_voters(&mut leader, id as usize, Duration::from_secs(40)).await;
    }

    drop(n1);
    let deadline = Instant::now() + Duration::from_secs(40);
    let new_leader = loop {
        let mut led = None;
        for member in [&n2, &n3] {
            if cypher_on(member, "CREATE (:Account {email: 'b@x', name: 'second'})")
                .await
                .is_ok()
            {
                led = Some(member);
                break;
            }
        }
        if let Some(member) = led {
            break member;
        }
        assert!(Instant::now() < deadline, "no member took over");
        tokio::time::sleep(Duration::from_millis(300)).await;
    };

    let constraints = new_leader
        .schema_client()
        .await
        .list_constraints(ListConstraintsRequest {})
        .await
        .expect("list constraints")
        .into_inner()
        .constraints;
    let summary: Vec<(String, i32, String)> = constraints
        .into_iter()
        .map(|c| (c.name, c.state, c.backing_index))
        .collect();
    assert_eq!(
        summary,
        [
            (
                "account_email".to_string(),
                ConstraintState::Active as i32,
                "account_email".to_string()
            ),
            (
                "account_name".to_string(),
                ConstraintState::Active as i32,
                String::new()
            ),
        ]
    );

    let duplicate = cypher_on(new_leader, "CREATE (:Account {email: 'a@x', name: 'dup'})")
        .await
        .expect_err("a duplicate of a value written before the cluster");
    assert_eq!(
        duplicate.code(),
        tonic::Code::AlreadyExists,
        "{duplicate:?}"
    );
    let unnamed = cypher_on(new_leader, "CREATE (:Account {email: 'c@x'})")
        .await
        .expect_err("a node without the required name");
    assert_eq!(
        unnamed.code(),
        tonic::Code::FailedPrecondition,
        "{unnamed:?}"
    );
}

/// A group of three members formed around the first, which leads.
async fn three_member_group() -> (CoordinodeProcess, CoordinodeProcess, CoordinodeProcess) {
    three_member_group_with(&[], &[], "").await
}

/// [`three_member_group`] with `first_env` set on node 1, the first leader,
/// and `env` set on node 3 with `settings` added to its config file.
async fn three_member_group_with(
    first_env: &[(&str, &str)],
    env: &[(&str, &str)],
    settings: &str,
) -> (CoordinodeProcess, CoordinodeProcess, CoordinodeProcess) {
    let (p1, p2, p3) = (free_port(), free_port(), free_port());
    let n1 = CoordinodeProcess::start_cluster_member_with(1, p1, &[p2, p3], first_env, "").await;
    let n2 = CoordinodeProcess::start_cluster_member(2, p2, &[p1, p3]).await;
    let n3 = CoordinodeProcess::start_cluster_member_with(3, p3, &[p1, p2], env, settings).await;
    let mut leader = n1.cluster_client().await;
    for (id, member) in [(2, &n2), (3, &n3)] {
        leader
            .join_node(JoinNodeRequest {
                node_id: id,
                address: member.member_addr(),
                pre_seeded: false,
                node_identity: member.member_identity(),
            })
            .await
            .expect("JoinNode must be accepted");
        wait_for_voters(
            &mut leader,
            usize::try_from(id).expect("small"),
            Duration::from_secs(40),
        )
        .await;
    }
    (n1, n2, n3)
}

/// A member that lost every copy of its vote does not start again: the
/// process exits naming the refusal. The leader replaces it through
/// ReplaceNode with a new node on its own directory; the same request sent
/// again changes nothing, a request naming the old node id as the new one is
/// refused, and the group keeps three voters with the new member in place
/// of the old.
#[tokio::test(flavor = "multi_thread")]
async fn a_member_without_its_vote_is_refused_and_replaced() {
    use coordinode_integration::proto::admin::ReplaceNodeRequest;

    let (n1, n2, n3) = three_member_group().await;
    cypher_on(&n1, "CREATE (:Kept {v: 1})")
        .await
        .expect("the leader writes");
    let (p1, p2, p3) = (n1.port, n2.port, n3.port);
    let data_dir = n3.stop_keeping_data().await;
    let dir = data_dir.path();
    // The consensus log's tree, in the deployment's own set.
    std::fs::remove_dir_all(dir.join("system").join("raft")).expect("lose the Raft tree");
    std::fs::remove_dir_all(dir.join("local-journal")).expect("lose the journal");
    for entry in std::fs::read_dir(dir).expect("list the store") {
        let path = entry.expect("an entry").path();
        if path
            .file_name()
            .is_some_and(|name| name.to_string_lossy().starts_with("consensus-vote"))
        {
            std::fs::remove_file(path).expect("lose a copy slot");
        }
    }
    let (status, printed) = start_cluster_member_expecting_refusal(3, p3, &[p1, p2], dir).await;
    assert!(
        !status.success(),
        "a member whose vote no copy proves must not start"
    );
    assert!(
        printed.contains("vote cannot be proved"),
        "the refusal must say what happened, got: {printed}"
    );

    let p4 = free_port();
    let n4 = CoordinodeProcess::start_cluster_member(4, p4, &[p1, p2]).await;
    let mut leader = n1.cluster_client().await;
    let request = ReplaceNodeRequest {
        old_node_id: 3,
        new_node_id: 4,
        address: n4.member_addr(),
        node_identity: n4.member_identity(),
    };
    let own = leader
        .replace_node(ReplaceNodeRequest {
            new_node_id: 3,
            ..request.clone()
        })
        .await
        .expect_err("a member is never replaced by itself");
    assert_eq!(own.code(), tonic::Code::InvalidArgument, "{own:?}");
    let done = leader
        .replace_node(request.clone())
        .await
        .expect("the replacement completes")
        .into_inner();
    assert_eq!((done.old_node_id, done.new_node_id), (3, 4));
    leader
        .replace_node(request)
        .await
        .expect("the same request again changes nothing");
    wait_for_voters(&mut leader, 3, Duration::from_secs(40)).await;
    let members: Vec<String> = leader
        .get_cluster_status(GetClusterStatusRequest {})
        .await
        .expect("status")
        .into_inner()
        .nodes
        .into_iter()
        .map(|n| n.node_id)
        .collect();
    assert!(
        members.contains(&"4".to_string()) && !members.contains(&"3".to_string()),
        "member 4 stands where 3 did: {members:?}"
    );
    let deadline = Instant::now() + Duration::from_secs(30);
    while local_read(&n4, "MATCH (n:Kept) RETURN n.v")
        .await
        .map_or(true, |rows| rows.len() != 1)
    {
        assert!(
            Instant::now() < deadline,
            "member 4 never received the history"
        );
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
}

/// A transaction ID as a client draws it: a random UUID.
fn draw_transaction_id() -> Vec<u8> {
    uuid::Uuid::new_v4().as_bytes().to_vec()
}

/// An autonomous write sent to a follower runs at the leader and answers as
/// the follower's own, counted as one hop and naming the leader; the leader's
/// own answers carry no hint.
#[tokio::test(flavor = "multi_thread")]
async fn a_write_through_a_follower_runs_at_the_leader() {
    let (n1, n2, _n3) = three_member_group().await;
    let answered = n2
        .cypher_client()
        .await
        .execute_cypher(ExecuteCypherRequest {
            query: "CREATE (n:ViaFollower {k: 1}) RETURN n.k AS k".to_string(),
            ..Default::default()
        })
        .await
        .expect("the follower passes the write on");
    let metadata = answered.metadata().clone();
    let answer = answered.into_inner();
    assert_eq!(answer.columns, vec!["k".to_string()]);
    assert_eq!(answer.rows.len(), 1);
    let stats = answer.stats.expect("stats");
    assert_eq!(stats.nodes_created, 1);
    assert!(stats.served_by_leader);
    let receipt = stats.receipt.expect("the write's receipt comes back");
    assert!(receipt.operation_time > 0, "its commit time: {receipt:?}");
    let position = receipt.position.expect("its own entry");
    assert!(position.index > 0, "{position:?}");
    assert_eq!(
        position.history.as_ref().map(|h| h.deployment.len()),
        Some(16),
        "qualified by the deployment it counts entries of"
    );

    // Read-your-writes on the follower that passed the write on: a read
    // bounded by the write's commit time sees it there, and one bounded by
    // its position too.
    for (after_ts, after_position) in [(receipt.operation_time, None), (0, Some(position.clone()))]
    {
        let read = n2
            .cypher_client()
            .await
            .execute_cypher(ExecuteCypherRequest {
                query: "MATCH (n:ViaFollower) RETURN n.k AS k".to_string(),
                read_preference: 3, // SECONDARY
                read_concern: Some(ReadConcern {
                    level: 2, // MAJORITY
                    after_ts,
                    after_position,
                    wait_ms: 10_000,
                    ..Default::default()
                }),
                ..Default::default()
            })
            .await
            .expect("the follower meets the bound");
        let read = read.into_inner();
        assert_eq!(read.rows.len(), 1, "the follower read sees the write");
        assert!(
            !read.stats.expect("stats").served_by_leader,
            "served by the follower"
        );
    }

    // A position of another history is refused, however small its index.
    let mut foreign = position.clone();
    if let Some(history) = foreign.history.as_mut() {
        history.deployment = vec![0xEE; 16];
    }
    let refused = n2
        .cypher_client()
        .await
        .execute_cypher(ExecuteCypherRequest {
            query: "MATCH (n:ViaFollower) RETURN n.k AS k".to_string(),
            read_preference: 3, // SECONDARY
            read_concern: Some(ReadConcern {
                level: 2,
                after_position: Some(foreign),
                ..Default::default()
            }),
            ..Default::default()
        })
        .await
        .expect_err("a position of another history");
    assert_eq!(
        refused.code(),
        tonic::Code::FailedPrecondition,
        "{refused:?}"
    );
    assert_eq!(
        metadata
            .get("x-coordinode-hops")
            .map(|v| v.to_str().expect("ascii")),
        Some("1")
    );
    assert_eq!(
        metadata
            .get("x-coordinode-leader")
            .map(|v| v.to_str().expect("ascii")),
        Some("1")
    );

    let on_leader = n1
        .cypher_client()
        .await
        .execute_cypher(ExecuteCypherRequest {
            query: "MATCH (n:ViaFollower) RETURN n.k AS k".to_string(),
            ..Default::default()
        })
        .await
        .expect("read at the leader");
    assert!(on_leader.metadata().get("x-coordinode-leader").is_none());
    assert_eq!(on_leader.into_inner().rows.len(), 1);
}

/// Every member of a group stopped at once exits on its own. Each waited for
/// the connections other members held to it to close, while those members
/// kept them open until they stopped themselves: all three waited forever,
/// with the leader still appending entries, until something killed them.
#[tokio::test(flavor = "multi_thread")]
async fn a_group_stopped_at_once_exits() {
    let (mut n1, mut n2, mut n3) = three_member_group().await;
    // Peers' streams in use: a write a follower passes on to the leader.
    n2.cypher_client()
        .await
        .execute_cypher(ExecuteCypherRequest {
            query: "CREATE (:Stopping {k: 1})".to_string(),
            ..Default::default()
        })
        .await
        .expect("a write through a follower");

    for member in [&n1, &n2, &n3] {
        member.request_stop();
    }
    // The drain window, then the close and the store's flush.
    let within = Duration::from_secs(40);
    let (a, b, c) = tokio::join!(
        n1.exits_within(within),
        n2.exits_within(within),
        n3.exits_within(within)
    );
    assert!(a && b && c, "exited: n1 {a}, n2 {b}, n3 {c}");
}

/// A member asked to stop leaves consensus first, then closes its
/// connections. A leader hands its leadership over while the connections its
/// peers and a client hold to it stay open: another member leads well inside
/// the drain window (15 s), writes go on there, what the old leader
/// acknowledged is read there, and the old leader exits on its own within
/// 8 s of the request while its client's connection is still open. Before,
/// its consensus ran on, leading, until every connection had closed or the
/// process was killed.
#[tokio::test(flavor = "multi_thread")]
async fn a_stopped_leader_hands_over_before_the_drain_window() {
    let (mut n1, n2, n3) = three_member_group().await;
    // Peers' streams in use: a write a follower passes on to the leader.
    let acknowledged = cypher_write_on(&n2, "CREATE (:Handed {k: 1})")
        .await
        .expect("a write through a follower");
    // A client connection held open on the leader across its stop.
    let mut client = n1.cypher_client().await;
    client
        .execute_cypher(ExecuteCypherRequest {
            query: "MATCH (n:Handed) RETURN n.k".to_string(),
            ..Default::default()
        })
        .await
        .expect("a read on the leader");

    let asked = std::time::Instant::now();
    n1.request_stop();
    let mut status = n2.cluster_client().await;
    let handed_over = loop {
        let leader = status
            .get_cluster_status(GetClusterStatusRequest {})
            .await
            .map(|s| s.into_inner().leader_id)
            .unwrap_or_default();
        if leader == "2" || leader == "3" {
            break asked.elapsed();
        }
        assert!(
            asked.elapsed() < Duration::from_secs(5),
            "node 1 still led {:?} after it was asked to stop (leader {leader:?})",
            asked.elapsed()
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    };
    eprintln!("another member led {handed_over:?} after the stop request");

    let after = cypher_write_on(&n2, "CREATE (:AfterHandOver {k: 1})")
        .await
        .expect("writes go on after the hand-over");
    let rows = cypher_after_on(&n3, "MATCH (n:Handed) RETURN n.k", after.max(acknowledged))
        .await
        .expect("the acknowledged write is read after the hand-over");
    assert_eq!(rows.len(), 1, "the acknowledged write after the hand-over");
    // With its client's connection still open, the old leader exits on its
    // own before a supervisor's 10 s grace would kill it.
    assert!(
        n1.exits_within(Duration::from_secs(8).saturating_sub(asked.elapsed()))
            .await,
        "the old leader was still running {:?} after it was asked to stop",
        asked.elapsed()
    );
    eprintln!(
        "the old leader exited {:?} after the stop request",
        asked.elapsed()
    );
    drop(client);
}

/// A follower asked to stop leaves consensus first as well: it exits well
/// inside the drain window while the leader's streams to it stay open, and
/// the group goes on writing with the two members left.
#[tokio::test(flavor = "multi_thread")]
async fn a_stopped_follower_exits_before_the_drain_window() {
    let (n1, n2, mut n3) = three_member_group().await;
    cypher_write_on(&n1, "CREATE (:Before {k: 1})")
        .await
        .expect("the leader writes");

    let asked = std::time::Instant::now();
    n3.request_stop();
    assert!(
        n3.exits_within(Duration::from_secs(8)).await,
        "the follower was still running {:?} after it was asked to stop",
        asked.elapsed()
    );
    let after = cypher_write_on(&n1, "CREATE (:After {k: 1})")
        .await
        .expect("the group writes with two members");
    let rows = cypher_after_on(&n2, "MATCH (n:After) RETURN n.k", after)
        .await
        .expect("the other follower reads it");
    assert_eq!(rows.len(), 1);
}

/// A member whose consensus stop outlasts the drain window (1 s here, the
/// stop held 3 s) stops its consensus once: the serving is cut off at the
/// window, the stop already under way runs to its end, and only then does
/// the member close and exit. Before, cutting off the serving also cut off
/// the stop, and the member started it again from the beginning.
#[cfg(feature = "test-faults")]
#[tokio::test(flavor = "multi_thread")]
async fn a_consensus_stop_past_the_drain_window_runs_once() {
    let marks = tempfile::NamedTempFile::new().expect("marks file");
    let marks_path = marks.path().to_str().expect("utf-8 path").to_owned();
    let (n1, n2, mut n3) = three_member_group_with(
        &[],
        &[
            ("COORDINODE_TEST_CONSENSUS_STOP_DELAY_MS", "3000"),
            ("COORDINODE_TEST_CONSENSUS_STOP_MARKS", &marks_path),
        ],
        "shutdown_grace_secs: 1\n",
    )
    .await;
    cypher_write_on(&n1, "CREATE (:Before {k: 1})")
        .await
        .expect("the leader writes");

    let asked = std::time::Instant::now();
    n3.request_stop();
    assert!(
        n3.exits_within(Duration::from_secs(9)).await,
        "the member was still running {:?} after it was asked to stop",
        asked.elapsed()
    );
    let exited = asked.elapsed();
    assert!(
        exited >= Duration::from_secs(3),
        "the member closed after {exited:?}, before its held consensus stop ended"
    );
    let stops = std::fs::read_to_string(marks.path()).expect("the marks");
    assert_eq!(stops, "start\nend\n", "one consensus stop, run to its end");

    let after = cypher_write_on(&n1, "CREATE (:After {k: 1})")
        .await
        .expect("the group writes with two members");
    let rows = cypher_after_on(&n2, "MATCH (n:After) RETURN n.k", after)
        .await
        .expect("the other follower reads it");
    assert_eq!(rows.len(), 1);
}

/// A client works through whichever member it reached: a session on a
/// follower opens and fills a transaction held by the leader, a second
/// session on another member continues the transaction the first one began
/// and commits it, and a commit repeated after that is told the transaction
/// is gone.
#[tokio::test(flavor = "multi_thread")]
async fn a_session_on_a_follower_works_on_the_leaders_transaction() {
    use coordinode_integration::proto::session::server_frame::Event;
    use coordinode_integration::proto::session::session_service_client::SessionServiceClient;
    use coordinode_integration::proto::session::{
        Begin, ClientFrame, Commit, Execute, ServerFrame, client_frame,
    };
    use tokio_stream::wrappers::ReceiverStream;

    async fn open(
        node: &CoordinodeProcess,
    ) -> (
        tokio::sync::mpsc::Sender<ClientFrame>,
        tonic::Streaming<ServerFrame>,
    ) {
        let channel = tonic::transport::Endpoint::from_shared(node.endpoint())
            .expect("endpoint")
            .connect()
            .await
            .expect("connect");
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        let inbound = SessionServiceClient::new(channel)
            .session(ReceiverStream::new(rx))
            .await
            .expect("open session")
            .into_inner();
        (tx, inbound)
    }
    async fn terminal(inbound: &mut tonic::Streaming<ServerFrame>, want: u64) -> Event {
        loop {
            let f = inbound.message().await.expect("frame").expect("not end");
            if f.request_id == want {
                match f.event {
                    Some(e @ (Event::CursorEnd(_) | Event::Committed(_) | Event::Error(_))) => {
                        return e;
                    }
                    _ => continue,
                }
            }
        }
    }
    fn execute(request_id: u64, query: &str, transaction: &[u8]) -> ClientFrame {
        ClientFrame {
            request_id,
            op: Some(client_frame::Op::Execute(Execute {
                query: query.to_string(),
                txid: transaction.to_vec(),
                ..Default::default()
            })),
        }
    }

    let (n1, n2, n3) = three_member_group().await;
    let (first, mut first_in) = open(&n2).await;
    let transaction = draw_transaction_id();
    // The Begin is not answered; the write sent right after it runs once the
    // transaction is open.
    first
        .send(ClientFrame {
            request_id: 1,
            op: Some(client_frame::Op::Begin(Begin {
                txid: transaction.clone(),
                operation: None,
            })),
        })
        .await
        .expect("begin");
    first
        .send(execute(2, "CREATE (:SessionRouted {k: 1})", &transaction))
        .await
        .expect("write");
    match terminal(&mut first_in, 2).await {
        Event::CursorEnd(_) => {}
        other => panic!("the write runs at the leader, got {other:?}"),
    }
    drop(first);
    while first_in.message().await.expect("frame").is_some() {}

    let (second, mut second_in) = open(&n3).await;
    second
        .send(execute(1, "CREATE (:SessionRouted {k: 2})", &transaction))
        .await
        .expect("write");
    match terminal(&mut second_in, 1).await {
        Event::CursorEnd(_) => {}
        other => panic!("another member continues it, got {other:?}"),
    }
    let commit = |request_id| ClientFrame {
        request_id,
        op: Some(client_frame::Op::Commit(Commit {
            txid: transaction.clone(),
            operation: None,
            expect: Vec::new(),
        })),
    };
    second.send(commit(2)).await.expect("commit");
    match terminal(&mut second_in, 2).await {
        Event::Committed(_) => {}
        other => panic!("the commit lands, got {other:?}"),
    }
    // The transaction is gone at the leader: committing it again is told so.
    second.send(commit(3)).await.expect("commit again");
    match terminal(&mut second_in, 3).await {
        Event::Error(e) => assert_eq!(e.code, tonic::Code::NotFound as u32, "{e:?}"),
        other => panic!("already committed, got {other:?}"),
    }
    drop(second);
    while second_in.message().await.expect("frame").is_some() {}

    assert_eq!(
        cypher_on(&n1, "MATCH (n:SessionRouted) RETURN n")
            .await
            .expect("read")
            .len(),
        2
    );
}

/// Commits of interactive transactions held by a leader that is stopped
/// while they arrive, sent through a follower and on the leader itself, are
/// told the truth: each transaction's operation, resolved afterwards with
/// the exclusion on a connection opened again, is committed exactly when its
/// write is there, a commit answered Committed is there, and a commit
/// answered with a definite refusal (aborted, not found, a failed
/// precondition) is not. A commit cut off by the stop is answered with an
/// error, or not at all, and the operation then resolves it; never a false
/// success or a false rollback.
#[tokio::test(flavor = "multi_thread")]
async fn interactive_commits_across_a_leader_stop_resolve_truthfully() {
    commits_across_a_leader_stop(false).await;
}

/// [`interactive_commits_across_a_leader_stop_resolve_truthfully`] with each
/// of the leader's Raft log syncs held 1.5 s from just before its stop: the
/// commits it proposed are still on their way when it stops, and are
/// answered with an outcome to resolve, or committed through the other
/// members; never a false success or a false rollback.
#[cfg(all(unix, feature = "test-faults"))]
#[tokio::test(flavor = "multi_thread")]
async fn interactive_commits_in_flight_at_a_leader_stop_resolve_truthfully() {
    commits_across_a_leader_stop(true).await;
}

/// A COMMIT that committed and whose Committed answer was lost, on the
/// leader itself and through a follower, then the leader stopped: the client
/// got no answer stating an outcome, the operation resolves at a member still
/// running as committed with the commit time the lost answer carried, the
/// transaction's one node is there exactly once, and the same holds after
/// the stopped leader reopens its stored data.
#[cfg(feature = "test-faults")]
#[tokio::test(flavor = "multi_thread")]
async fn a_lost_committed_answer_across_a_leader_stop_resolves_committed() {
    use coordinode_integration::proto::operation::{
        self, operation_service_client::OperationServiceClient,
    };
    use coordinode_integration::proto::session::server_frame::Event;
    use coordinode_integration::proto::session::session_service_client::SessionServiceClient;
    use coordinode_integration::proto::session::{
        Begin, ClientFrame, Commit, Execute, client_frame,
    };
    use tokio_stream::wrappers::ReceiverStream;

    // The request ids whose Committed answers node 1 (the leader, the direct
    // case) and node 3 (a follower, the routed case) hold back.
    const DIRECT: u64 = 7001;
    const ROUTED: u64 = 7002;

    let direct_marks = tempfile::NamedTempFile::new().expect("marks file");
    let routed_marks = tempfile::NamedTempFile::new().expect("marks file");
    let direct_path = direct_marks.path().to_str().expect("utf-8").to_owned();
    let routed_path = routed_marks.path().to_str().expect("utf-8").to_owned();
    let direct_id = DIRECT.to_string();
    let routed_id = ROUTED.to_string();
    let (mut n1, n2, n3) = three_member_group_with(
        &[
            ("COORDINODE_TEST_WITHHOLD_COMMIT", &direct_id),
            ("COORDINODE_TEST_WITHHOLD_MARKS", &direct_path),
        ],
        &[
            ("COORDINODE_TEST_WITHHOLD_COMMIT", &routed_id),
            ("COORDINODE_TEST_WITHHOLD_MARKS", &routed_path),
        ],
        "",
    )
    .await;
    let deadline = Instant::now() + Duration::from_secs(90);

    let mut operations = OperationServiceClient::new(
        tonic::transport::Endpoint::from_shared(n2.endpoint())
            .expect("endpoint")
            .connect()
            .await
            .expect("connect"),
    );
    let issuer = draw_transaction_id();
    let enrolled = operations
        .enroll_issuer(operation::EnrollIssuerRequest {
            issuer: issuer.clone(),
        })
        .await
        .expect("enrolled")
        .into_inner();
    let reference = |k: u64| operation::OperationRef {
        issuer: issuer.clone(),
        incarnation: enrolled.incarnation,
        sequence: enrolled.next_sequence + k,
        deployment: enrolled.deployment.clone(),
    };

    // Each case: a transaction opened, its node written, its COMMIT sent,
    // and the member's mark that it committed and held the answer back.
    let mut cases = Vec::new();
    for (k, node, request_id, marks) in [
        (0u64, &n1, DIRECT, direct_marks.path()),
        (1, &n3, ROUTED, routed_marks.path()),
    ] {
        let channel = tonic::transport::Endpoint::from_shared(node.endpoint())
            .expect("endpoint")
            .connect()
            .await
            .expect("connect");
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        let mut inbound = SessionServiceClient::new(channel)
            .session(ReceiverStream::new(rx))
            .await
            .expect("open session")
            .into_inner();
        let txid = draw_transaction_id();
        tx.send(ClientFrame {
            request_id: 1,
            op: Some(client_frame::Op::Begin(Begin {
                txid: txid.clone(),
                operation: Some(reference(k)),
            })),
        })
        .await
        .expect("begin");
        tx.send(ClientFrame {
            request_id: 2,
            op: Some(client_frame::Op::Execute(Execute {
                query: format!("CREATE (:LostReply {{k: {k}}})"),
                txid: txid.clone(),
                ..Default::default()
            })),
        })
        .await
        .expect("write");
        loop {
            let frame = inbound.message().await.expect("frame").expect("not end");
            match (frame.request_id, frame.event) {
                (2, Some(Event::CursorEnd(_))) => break,
                (_, Some(Event::Error(e))) => panic!("the write failed: {e:?}"),
                _ => {}
            }
        }
        tx.send(ClientFrame {
            request_id,
            op: Some(client_frame::Op::Commit(Commit {
                txid,
                operation: None,
                expect: Vec::new(),
            })),
        })
        .await
        .expect("commit");
        let withheld_ts = loop {
            let marked = std::fs::read_to_string(marks).unwrap_or_default();
            if let Some(ts) = marked
                .lines()
                .find_map(|line| line.strip_prefix(&format!("withheld {request_id} ")))
            {
                break ts.trim().parse::<u64>().expect("a commit time");
            }
            assert!(
                Instant::now() < deadline,
                "case {k}: the member never marked its answer held back"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        };
        assert_ne!(withheld_ts, 0, "case {k}: the commit had a time");
        cases.push((k, request_id, withheld_ts, tx, inbound));
    }

    // Neither client got an answer stating the outcome, before the stop or
    // after it.
    let asked = Instant::now();
    n1.request_stop();
    // The direct session is a client stream left open, which the leader
    // waits for through its drain window (15 s) before it closes it.
    assert!(
        n1.exits_within(Duration::from_secs(25)).await,
        "the leader was still running {:?} after it was asked to stop",
        asked.elapsed()
    );
    for (k, request_id, _, _, inbound) in &mut cases {
        loop {
            match tokio::time::timeout(Duration::from_millis(500), inbound.message()).await {
                Err(_) | Ok(Ok(None) | Err(_)) => break,
                Ok(Ok(Some(frame))) => {
                    if frame.request_id != *request_id {
                        continue;
                    }
                    match frame.event {
                        Some(Event::Committed(_)) => {
                            panic!("case {k}: the held-back answer arrived")
                        }
                        Some(Event::Error(e)) => assert_eq!(
                            stated_outcome(&e),
                            None,
                            "case {k}: the client was told an outcome: {e:?}"
                        ),
                        _ => {}
                    }
                }
            }
        }
    }

    // Resolved at a member still running.
    let fence = write_after_a_leader_change(&n2, deadline).await;
    for (k, _, withheld_ts, _, _) in &cases {
        let resolved = resolve_excluded(&mut operations, reference(*k), deadline).await;
        let label =
            check_commit_outcome(&n3, "LostReply", *k, &Answer::None, &resolved, fence).await;
        assert_eq!(
            resolved.state(),
            operation::OperationState::Committed,
            "case {k}: {label}"
        );
        assert_eq!(
            resolved.commit_ts, *withheld_ts,
            "case {k}: the commit time the lost answer carried"
        );
    }

    // The same after the stopped leader reopens its stored data: resolved
    // through it and read there.
    let peers = [n2.port, n3.port];
    let n1 = n1.restart_member(1, &peers, &[], false).await;
    let mut reopened = OperationServiceClient::new(
        tonic::transport::Endpoint::from_shared(n1.endpoint())
            .expect("endpoint")
            .connect()
            .await
            .expect("connect to the reopened member"),
    );
    for (k, _, withheld_ts, _, _) in &cases {
        let resolved = resolve_excluded(&mut reopened, reference(*k), deadline).await;
        check_commit_outcome(&n1, "LostReply", *k, &Answer::None, &resolved, fence).await;
        assert_eq!(
            resolved.commit_ts, *withheld_ts,
            "case {k}: the same commit after the reopen"
        );
    }
}

/// The commits of transactions held by node 1, the leader, sent through
/// node 2 while node 1 is asked to stop; with `stall_leader_log`, node 1's
/// own log syncs are held from just before the stop.
async fn commits_across_a_leader_stop(stall_leader_log: bool) {
    use coordinode_integration::proto::operation::{
        self, operation_service_client::OperationServiceClient,
    };
    use coordinode_integration::proto::session::server_frame::Event;
    use coordinode_integration::proto::session::session_service_client::SessionServiceClient;
    use coordinode_integration::proto::session::{
        Begin, ClientFrame, Commit, Execute, client_frame,
    };
    use tokio_stream::wrappers::ReceiverStream;

    const TRANSACTIONS: u64 = 16;
    const SPACING: Duration = Duration::from_millis(25);

    let first_env: &[(&str, &str)] = if stall_leader_log {
        &[("COORDINODE_TEST_DISK_STALL_OWN_MS", "1500")]
    } else {
        &[]
    };
    let (n1, n2, n3) = three_member_group_with(first_env, &[], "").await;
    let channel = tonic::transport::Endpoint::from_shared(n2.endpoint())
        .expect("endpoint")
        .connect()
        .await
        .expect("connect");
    let mut operations = OperationServiceClient::new(channel.clone());
    let issuer = draw_transaction_id();
    let enrolled = operations
        .enroll_issuer(operation::EnrollIssuerRequest {
            issuer: issuer.clone(),
        })
        .await
        .expect("enrolled through the follower")
        .into_inner();
    let reference = |i: u64| operation::OperationRef {
        issuer: issuer.clone(),
        incarnation: enrolled.incarnation,
        sequence: enrolled.next_sequence + i,
        deployment: enrolled.deployment.clone(),
    };

    // Two sessions: one through node 2, whose commits node 2 passes on to the
    // leader, and one on the leader itself, whose connection the stop ends.
    // Transactions 0..TRANSACTIONS go through the first, the next
    // TRANSACTIONS through the second; every one open at the leader, its
    // write run, before the stop.
    let direct = tonic::transport::Endpoint::from_shared(n1.endpoint())
        .expect("endpoint")
        .connect()
        .await
        .expect("connect");
    let mut sessions = Vec::new();
    for (side, channel) in [channel, direct].into_iter().enumerate() {
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let mut inbound = SessionServiceClient::new(channel)
            .session(ReceiverStream::new(rx))
            .await
            .expect("open session")
            .into_inner();
        let first = side as u64 * TRANSACTIONS;
        let mut commits = Vec::new();
        for i in first..first + TRANSACTIONS {
            let txid = draw_transaction_id();
            tx.send(ClientFrame {
                request_id: 2 * i + 1,
                op: Some(client_frame::Op::Begin(Begin {
                    txid: txid.clone(),
                    operation: Some(reference(i)),
                })),
            })
            .await
            .expect("begin");
            tx.send(ClientFrame {
                request_id: 2 * i + 2,
                op: Some(client_frame::Op::Execute(Execute {
                    query: format!("CREATE (:AtStop {{k: {i}}})"),
                    txid: txid.clone(),
                    ..Default::default()
                })),
            })
            .await
            .expect("write");
            commits.push(ClientFrame {
                request_id: 1000 + i,
                op: Some(client_frame::Op::Commit(Commit {
                    txid,
                    operation: None,
                    expect: Vec::new(),
                })),
            });
        }
        let mut writes_done = 0;
        while writes_done < TRANSACTIONS {
            let frame = inbound.message().await.expect("frame").expect("not end");
            match frame.event {
                Some(Event::CursorEnd(_)) => writes_done += 1,
                Some(Event::Error(e)) => panic!("a write before the stop failed: {e:?}"),
                _ => {}
            }
        }
        sessions.push((tx, inbound, commits));
    }

    // The commits of both sessions, spaced out; the leader is asked to stop
    // a quarter in.
    let mut senders = Vec::new();
    let mut streams = Vec::new();
    for (tx, inbound, commits) in sessions {
        streams.push(inbound);
        senders.push(tokio::spawn(async move {
            for commit in commits {
                // A send into a session whose connection ended is a commit
                // never sent; it is resolved like one whose answer was lost.
                if tx.send(commit).await.is_err() {
                    break;
                }
                tokio::time::sleep(SPACING).await;
            }
        }));
    }
    tokio::time::sleep(SPACING * u32::try_from(TRANSACTIONS / 4).expect("small")).await;
    #[cfg(all(unix, feature = "test-faults"))]
    if stall_leader_log {
        // Held from here on: the stop finds commits proposed and unsynced.
        n1.toggle_disk_stall();
        tokio::time::sleep(SPACING * 2).await;
    }
    n1.request_stop();

    // Each commit's answer: Committed, an error, or none when its session's
    // connection ended first. The session through node 2 answers every one.
    let mut replies = std::collections::HashMap::<u64, Option<_>>::new();
    let deadline = Instant::now() + Duration::from_secs(60);
    let [mut routed, mut on_leader]: [_; 2] = streams.try_into().expect("two sessions");
    let mut leader_open = true;
    let routed_done = |replies: &std::collections::HashMap<u64, Option<_>>| {
        (0..TRANSACTIONS).all(|i| replies.contains_key(&i))
    };
    while replies.len() < 2 * TRANSACTIONS as usize && (leader_open || !routed_done(&replies)) {
        let frame = tokio::select! {
            f = routed.message() => f.expect("frame").expect("not end"),
            f = on_leader.message(), if leader_open => match f {
                Ok(Some(f)) => f,
                Ok(None) | Err(_) => {
                    leader_open = false;
                    continue;
                }
            },
            () = tokio::time::sleep_until(deadline.into()) => {
                panic!("commits unanswered after 60 s: {} of {}", replies.len(), 2 * TRANSACTIONS)
            }
        };
        if !(1000..1000 + 2 * TRANSACTIONS).contains(&frame.request_id) {
            continue;
        }
        match frame.event {
            Some(Event::Committed(_)) => {
                replies.insert(frame.request_id - 1000, None);
            }
            Some(Event::Error(e)) => {
                replies.insert(frame.request_id - 1000, Some(e));
            }
            _ => {}
        }
    }
    for sender in senders {
        sender.await.expect("the commits were sent");
    }

    // After the hand-over, on a connection opened again to node 2: each
    // operation resolved, excluded if it never committed, and its write
    // looked for at another member.
    let mut operations = OperationServiceClient::new(
        tonic::transport::Endpoint::from_shared(n2.endpoint())
            .expect("endpoint")
            .connect()
            .await
            .expect("connect again"),
    );
    let fence = write_after_a_leader_change(&n2, deadline).await;
    let mut outcomes = std::collections::BTreeMap::<String, u32>::new();
    for i in 0..2 * TRANSACTIONS {
        let side = if i < TRANSACTIONS {
            "routed"
        } else {
            "on the leader"
        };
        let answer = match replies.get(&i) {
            None => Answer::None,
            Some(None) => Answer::Committed,
            Some(Some(e)) => Answer::Error(e.clone()),
        };
        let resolved = resolve_excluded(&mut operations, reference(i), deadline).await;
        let label = check_commit_outcome(&n3, "AtStop", i, &answer, &resolved, fence).await;
        *outcomes.entry(format!("{side}: {label}")).or_default() += 1;
    }
    eprintln!("commit outcomes across the stop: {outcomes:?}");
}

/// The answer a client got to a COMMIT.
#[derive(Debug, Clone)]
enum Answer {
    /// Committed.
    Committed,
    /// An error, with the typed details it carried.
    Error(coordinode_integration::proto::session::SessionError),
    /// None: the connection ended, or the answer was lost, before it came.
    None,
}

/// The outcome an error answer to a COMMIT states, from its typed details:
/// a settled operation's outcome, a transaction that ended committed or
/// aborted, or a refusal that aborts the whole transaction (a conflict, a
/// refused invariant, a revision mismatch). `None` for an error that states
/// no outcome: a transaction not found where a COMMIT may have been sent, a
/// refusal before anything ran, an outcome left to resolve.
fn stated_outcome(
    error: &coordinode_integration::proto::session::SessionError,
) -> Option<coordinode_integration::proto::operation::OperationState> {
    use coordinode_integration::proto::operation::OperationState;
    use prost::Message as _;

    let info = error
        .status
        .as_ref()?
        .details
        .iter()
        .find(|any| any.type_url.ends_with("google.rpc.ErrorInfo"))
        .and_then(|any| tonic_types::pb::ErrorInfo::decode(any.value.as_slice()).ok())?;
    let outcome = |name: Option<&String>| match name.map(String::as_str) {
        Some("COMMITTED") => Some(OperationState::Committed),
        Some("ABORTED") => Some(OperationState::Aborted),
        _ => None,
    };
    match info.reason.as_str() {
        "OPERATION_SETTLED" => outcome(info.metadata.get("outcome")),
        "TRANSACTION_CONFLICT" | "INVARIANT_REFUSED" | "REVISION_MISMATCH" => {
            Some(OperationState::Aborted)
        }
        "TRANSACTION_ABORTED" => match info.metadata.get("end").map(String::as_str) {
            Some("IDLE_COMMITTED") => Some(OperationState::Committed),
            Some("IDLE_COMMIT_UNKNOWN") | None => None,
            Some(_) => Some(OperationState::Aborted),
        },
        _ => None,
    }
}

/// A write through `node` once the group has a leader again, retried until
/// `deadline`: its commit time, a bound later reads wait for.
async fn write_after_a_leader_change(node: &CoordinodeProcess, deadline: Instant) -> u64 {
    loop {
        match cypher_write_on(node, "CREATE (:AfterStop {k: 1})").await {
            Ok(at) => break at,
            Err(e) if Instant::now() < deadline => {
                eprintln!("waiting for a new leader: {e}");
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
            Err(e) => panic!("no new leader: {e}"),
        }
    }
}

/// `operation` resolved through `operations` with the exclusion, retried
/// while no leader answers, until `deadline`.
async fn resolve_excluded(
    operations: &mut coordinode_integration::proto::operation::operation_service_client::OperationServiceClient<Channel>,
    operation: coordinode_integration::proto::operation::OperationRef,
    deadline: Instant,
) -> coordinode_integration::proto::operation::ResolveOperationResponse {
    loop {
        match operations
            .resolve_operation(
                coordinode_integration::proto::operation::ResolveOperationRequest {
                    operation: Some(operation.clone()),
                    exclude: true,
                },
            )
            .await
        {
            Ok(r) => break r.into_inner(),
            Err(e) if Instant::now() < deadline => {
                eprintln!("resolving operation {}: {e}", operation.sequence);
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
            Err(e) => panic!("operation {} not resolved: {e}", operation.sequence),
        }
    }
}

/// Check a transaction that wrote one `label` node `{k: k}` against how its
/// operation resolved and what its COMMIT was answered: the resolution is
/// terminal; committed means a commit time and exactly that one node at
/// `reader` (read after the commit and after `fence`), aborted means none;
/// an answer of Committed, or an error stating an outcome, agrees with the
/// resolution. Returns a label of the case for the test's summary.
async fn check_commit_outcome(
    reader: &CoordinodeProcess,
    label: &str,
    k: u64,
    answer: &Answer,
    resolved: &coordinode_integration::proto::operation::ResolveOperationResponse,
    fence: u64,
) -> String {
    let rows = cypher_after_on(
        reader,
        &format!("MATCH (n:{label} {{k: {k}}}) RETURN n.k"),
        fence.max(resolved.commit_ts),
    )
    .await
    .expect("read at another member");
    let found: Vec<_> = rows
        .iter()
        .map(|row| row.values.first().and_then(|v| v.value.clone()))
        .collect();
    judge_commit_outcome(k, answer, resolved, &found)
}

/// [`check_commit_outcome`] over the values `found` of the transaction's
/// nodes read back.
fn judge_commit_outcome(
    k: u64,
    answer: &Answer,
    resolved: &coordinode_integration::proto::operation::ResolveOperationResponse,
    found: &[Option<coordinode_integration::proto::common::property_value::Value>],
) -> String {
    use coordinode_integration::proto::common::property_value::Value as Pv;
    use coordinode_integration::proto::operation::OperationState;

    let state = resolved.state();
    assert!(
        matches!(state, OperationState::Committed | OperationState::Aborted),
        "transaction {k}: resolved to {state:?}, not an outcome; answer {answer:?}"
    );
    if state == OperationState::Committed {
        assert_ne!(
            resolved.commit_ts, 0,
            "transaction {k}: committed with no commit time"
        );
        assert_eq!(
            found,
            [Some(Pv::IntValue(i64::try_from(k).expect("small")))],
            "transaction {k}: committed must have written exactly its one node; answer {answer:?}"
        );
    } else {
        assert!(
            found.is_empty(),
            "transaction {k}: aborted but wrote {found:?}; answer {answer:?}"
        );
    }
    match answer {
        Answer::Committed => {
            assert_eq!(
                state,
                OperationState::Committed,
                "transaction {k}: answered Committed but resolved {state:?}"
            );
            format!("Committed, resolved {state:?}")
        }
        Answer::Error(e) => {
            let stated = stated_outcome(e);
            if let Some(stated) = stated {
                assert_eq!(
                    stated, state,
                    "transaction {k}: the answer states {stated:?}, resolved {state:?}: {e:?}"
                );
            }
            let reason = e
                .status
                .as_ref()
                .and_then(|s| s.details.iter().find(|a| a.type_url.ends_with("ErrorInfo")))
                .and_then(|a| {
                    use prost::Message as _;
                    tonic_types::pb::ErrorInfo::decode(a.value.as_slice()).ok()
                })
                .map(|i| i.reason)
                .unwrap_or_default();
            format!(
                "error {} {reason} (states {stated:?}), resolved {state:?}",
                e.code
            )
        }
        Answer::None => format!("no answer, resolved {state:?}"),
    }
}

/// The commit-outcome check refuses what it exists to catch: an answer
/// stating an abort for a transaction that committed, an outcome left
/// unknown, a commit applied twice, a committed write missing, a commit
/// time missing, and an answer of Committed for an aborted transaction.
#[test]
fn the_commit_outcome_check_refuses_false_outcomes() {
    use coordinode_integration::proto::common::property_value::Value as Pv;
    use coordinode_integration::proto::operation::{OperationState, ResolveOperationResponse};
    use coordinode_integration::proto::session::SessionError;
    use prost::Message as _;

    let resolved = |state: OperationState, commit_ts: u64| {
        let mut r = ResolveOperationResponse {
            commit_ts,
            ..Default::default()
        };
        r.set_state(state);
        r
    };
    let refused_as = |reason: &str| {
        let info = tonic_types::pb::ErrorInfo {
            reason: reason.to_owned(),
            domain: "coordinode".to_owned(),
            metadata: std::collections::HashMap::new(),
        };
        Answer::Error(SessionError {
            code: tonic::Code::Aborted as u32,
            message: String::new(),
            status: Some(tonic_types::pb::Status {
                code: tonic::Code::Aborted as i32,
                message: String::new(),
                details: vec![prost_types::Any {
                    type_url: "type.googleapis.com/google.rpc.ErrorInfo".to_owned(),
                    value: info.encode_to_vec(),
                }],
            }),
        })
    };
    let one = [Some(Pv::IntValue(3))];
    let refuses = |answer: Answer, r: ResolveOperationResponse, found: &[Option<Pv>]| {
        let found = found.to_vec();
        std::panic::catch_unwind(move || judge_commit_outcome(3, &answer, &r, &found)).is_err()
    };

    // The honest cases pass.
    assert!(!refuses(
        Answer::Committed,
        resolved(OperationState::Committed, 9),
        &one
    ));
    assert!(!refuses(
        Answer::None,
        resolved(OperationState::Aborted, 0),
        &[]
    ));
    // A not-found answer states nothing: either outcome may follow.
    let not_found = Answer::Error(SessionError {
        code: tonic::Code::NotFound as u32,
        message: String::new(),
        status: None,
    });
    assert!(!refuses(
        not_found,
        resolved(OperationState::Committed, 9),
        &one
    ));

    assert!(
        refuses(
            refused_as("TRANSACTION_CONFLICT"),
            resolved(OperationState::Committed, 9),
            &one
        ),
        "a false abort passes"
    );
    assert!(
        refuses(Answer::None, resolved(OperationState::Unknown, 0), &[]),
        "an unresolved outcome passes"
    );
    assert!(
        refuses(
            Answer::None,
            resolved(OperationState::Committed, 9),
            &[Some(Pv::IntValue(3)), Some(Pv::IntValue(3))]
        ),
        "a double effect passes"
    );
    assert!(
        refuses(Answer::None, resolved(OperationState::Committed, 9), &[]),
        "a missing committed write passes"
    );
    assert!(
        refuses(Answer::None, resolved(OperationState::Committed, 0), &one),
        "a commit without a commit time passes"
    );
    assert!(
        refuses(Answer::Committed, resolved(OperationState::Aborted, 0), &[]),
        "a false success passes"
    );
    assert!(
        refuses(Answer::None, resolved(OperationState::Aborted, 0), &one),
        "an aborted transaction's write passes"
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
                // Only the leader reports the membership; a follower lists
                // itself alone. Wait until the survivor that leads answers,
                // without node 1. Before, a follower's own entry passed for the
                // membership whenever the other survivor took the leadership.
                let from_leader = s
                    .nodes
                    .iter()
                    .any(|n| n.node_id == s.leader_id && n.role == NodeRole::Leader as i32);
                if from_leader
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
