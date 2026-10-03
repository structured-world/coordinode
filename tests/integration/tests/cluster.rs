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
//! | `constraints_held_before_the_cluster_bind_every_member_after_a_leader_change` | ExecuteCypher, ListConstraints | Constraints that reached the members through the base snapshot are listed active on the new leader and refuse the writes that break them |
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
        })
        .await
        .expect("JoinNode(2) must be accepted");
    wait_for_voters(&mut leader, 2, Duration::from_secs(40)).await;

    leader
        .join_node(JoinNodeRequest {
            node_id: 3,
            address: n3.member_addr(),
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
            address: n2.member_addr(),
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
                transaction_id: 0,
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
    for (id, address) in [(2u64, n2.member_addr()), (3u64, n3.member_addr())] {
        leader
            .join_node(JoinNodeRequest {
                node_id: id,
                address,
                pre_seeded: false,
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
                    after_index: 0,
                    at_timestamp: 0,
                }),
                write_concern: None,
                transaction_id: 0,
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
                transaction_id: 0,
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
                        after_index: 0,
                        at_timestamp: 0,
                    }),
                    write_concern: None,
                    transaction_id: 0,
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
    wait_for_voters(&mut leader, 3, Duration::from_secs(10)).await;
}

/// Run `query` on `node` as a primary read or write.
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
            transaction_id: 0,
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
    let new_leader = loop {
        let mut led = None;
        for member in [&n2, &n3] {
            if cypher_on(member, "CREATE (:Handover {v: 'after'})")
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

    let rows = cypher_on(new_leader, "MATCH (n:Handover) RETURN id(n), n.v")
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
    for (id, address) in [(2u64, n2.member_addr()), (3u64, n3.member_addr())] {
        leader
            .join_node(JoinNodeRequest {
                node_id: id,
                address,
                pre_seeded: false,
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
