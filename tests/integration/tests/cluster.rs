//! Cluster administration integration tests.
//!
//! These tests exercise the ClusterService gRPC endpoints against a real
//! `coordinode` binary.  Because the binary currently bootstraps as a
//! single-node Raft (node_id=1, StubNetwork), multi-node cluster scenarios
//! are covered at the Raft library level in
//! `crates/coordinode-raft/tests/raft_cluster.rs`.
//!
//! What we cover here:
//!
//! | Test | Endpoint | Scenario |
//! |------|----------|---------|
//! | `decommission_last_voter_rejected_with_failed_precondition` | DecommissionNode | Quorum gate blocks removing the only voter |
//! | `get_cluster_status_standalone_reports_single_leader` | GetClusterStatus | Standalone node self-reports as leader |
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

use coordinode_integration::harness::CoordinodeProcess;
use coordinode_integration::proto::admin::{DecommissionNodeRequest, GetClusterStatusRequest};

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
