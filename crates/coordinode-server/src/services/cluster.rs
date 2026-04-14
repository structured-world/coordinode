//! ClusterService gRPC handler — cluster join/leave lifecycle for CE 3-node HA.
//!
//! Implements the admin cluster management protocol:
//! - [`ClusterServiceImpl::join_node`]: initiates join (add_learner → background promotion)
//! - [`ClusterServiceImpl::join_progress`]: streams join phase events until COMPLETE/FAILED
//! - [`ClusterServiceImpl::get_cluster_status`]: current node roles + replication lag
//! - [`ClusterServiceImpl::add_node`]: low-level learner-only add (no auto-promotion)
//! - [`ClusterServiceImpl::remove_node`]: membership remove
//!
//! ## Join Lifecycle
//!
//! ```text
//! Operator:  coordinode admin node join --node leader:7080 --id 3 --addr node3:7080
//!               ↓ gRPC → ClusterService.JoinNode
//! Server:    add_node(3, "node3:7080")  ← add_learner, fast Raft proposal
//!            spawn monitor_and_promote(3, tx)  ← background task
//!               ↓ broadcast JoinProgressEvent every 500ms
//! Operator:  coordinode admin node join ... --follow  ← subscribes JoinProgress stream
//!               LEARNER: lag=50000 (5%) …
//!               LEARNER: lag=1200  (97%) …
//!               READY_CHECK: lag=800, promoting …
//!               PROMOTING …
//!               COMPLETE — node 3 is now a Voter
//! ```

#![allow(clippy::unwrap_used)] // Only used in lock().unwrap() on non-poisoned Mutex

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use futures_util::Stream;
use tokio::sync::broadcast;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt as _;
use tonic::{Request, Response, Status};

use coordinode_raft::cluster::{JoinPhase, JoinProgressEvent, NodeRole, RaftNode};

use crate::proto::admin::cluster::{
    cluster_service_server::ClusterService, AddNodeRequest, ClusterNode, ClusterStatus,
    DecommissionNodeRequest, DecommissionNodeResponse, GetClusterStatusRequest, JoinNodeRequest,
    JoinNodeResponse, JoinProgressRequest, JoinStatus, NodeRole as ProtoNodeRole,
    NodeState as ProtoNodeState, RemoveNodeRequest, RemoveNodeResponse,
};

// ── Join state registry ────────────────────────────────────────────────────────

/// Active join operations keyed by node ID.
/// Entries are removed once the join completes or fails.
type JoinRegistry = Arc<Mutex<HashMap<u64, broadcast::Sender<JoinProgressEvent>>>>;

// ── Service impl ──────────────────────────────────────────────────────────────

pub struct ClusterServiceImpl {
    raft_node: Arc<RaftNode>,
    /// Active join channels. Each JoinNode call inserts an entry; the background
    /// task removes it when the join completes or fails.
    join_registry: JoinRegistry,
}

impl ClusterServiceImpl {
    pub fn new(raft_node: Arc<RaftNode>) -> Self {
        Self {
            raft_node,
            join_registry: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

// ── Proto mapping helpers ──────────────────────────────────────────────────────

fn proto_phase(phase: JoinPhase) -> i32 {
    use crate::proto::admin::cluster::JoinPhase as P;
    match phase {
        JoinPhase::Learner => P::Learner as i32,
        JoinPhase::ReadyCheck => P::ReadyCheck as i32,
        JoinPhase::Promoting => P::Promoting as i32,
        JoinPhase::Complete => P::Complete as i32,
        JoinPhase::Failed => P::Failed as i32,
    }
}

fn event_to_proto(ev: JoinProgressEvent) -> JoinStatus {
    JoinStatus {
        node_id: ev.node_id,
        phase: proto_phase(ev.phase),
        lag_entries: if ev.lag_entries == u64::MAX {
            0
        } else {
            ev.lag_entries
        },
        percent: ev.percent as u32,
        message: ev.message,
    }
}

fn proto_role(role: NodeRole) -> i32 {
    match role {
        NodeRole::Leader => ProtoNodeRole::Leader as i32,
        NodeRole::Follower => ProtoNodeRole::Follower as i32,
        NodeRole::Learner => ProtoNodeRole::Learner as i32,
    }
}

fn proto_state(lag: u64) -> i32 {
    // Healthy: lag ≤ 1 000 entries (readiness threshold).
    // Degraded: lag ≤ 50 000 entries (still catching up but alive).
    // Down: lag > 50 000 (stale or unreachable).
    if lag <= 1_000 {
        ProtoNodeState::Healthy as i32
    } else if lag <= 50_000 {
        ProtoNodeState::Degraded as i32
    } else {
        ProtoNodeState::Down as i32
    }
}

// ── tonic service impl ─────────────────────────────────────────────────────────

#[tonic::async_trait]
impl ClusterService for ClusterServiceImpl {
    // ── Status ──────────────────────────────────────────────────────────────

    async fn get_cluster_status(
        &self,
        _req: Request<GetClusterStatusRequest>,
    ) -> Result<Response<ClusterStatus>, Status> {
        let leader_id = self
            .raft_node
            .current_leader()
            .map(|id| id.to_string())
            .unwrap_or_default();

        let term = self.raft_node.current_term();

        let mut nodes = Vec::new();

        // Build node list from replication status (leader-only).
        // Falls back to empty list if this node is not the leader.
        if let Some(statuses) = self.raft_node.replication_status() {
            for s in statuses {
                let state = proto_state(s.lag_entries);
                nodes.push(ClusterNode {
                    node_id: s.node_id.to_string(),
                    address: String::new(), // address not tracked in NodeReplicationStatus
                    role: proto_role(s.role),
                    state,
                    lag_entries: s.lag_entries,
                });
            }
        } else {
            // Not leader — report self as follower with unknown lag.
            nodes.push(ClusterNode {
                node_id: self.raft_node.node_id().to_string(),
                address: String::new(),
                role: ProtoNodeRole::Follower as i32,
                state: ProtoNodeState::Healthy as i32,
                lag_entries: 0,
            });
        }

        Ok(Response::new(ClusterStatus {
            nodes,
            leader_id,
            raft_term: term,
        }))
    }

    // ── Join lifecycle ──────────────────────────────────────────────────────

    async fn join_node(
        &self,
        req: Request<JoinNodeRequest>,
    ) -> Result<Response<JoinNodeResponse>, Status> {
        let r = req.into_inner();
        let node_id = r.node_id;
        let addr = r.address;
        let pre_seeded = r.pre_seeded;

        tracing::info!(
            node_id,
            addr,
            pre_seeded,
            "JoinNode: initiating cluster join lifecycle"
        );

        // Step 1 (arch §Cluster Join Protocol, Step 2): add node as Learner.
        // This is a Raft proposal and completes quickly. Returns before three-tier
        // recovery starts on the joining node.
        self.raft_node
            .add_node(node_id, addr.clone())
            .await
            .map_err(|e| Status::internal(format!("add_learner failed: {e}")))?;

        // Create progress broadcast channel (capacity 128 events).
        let (tx, _) = broadcast::channel::<JoinProgressEvent>(128);
        self.join_registry
            .lock()
            .unwrap()
            .insert(node_id, tx.clone());

        // Send initial LEARNER event so subscribers get immediate feedback.
        let _ = tx.send(JoinProgressEvent {
            node_id,
            phase: JoinPhase::Learner,
            lag_entries: u64::MAX,
            percent: 0,
            message: format!(
                "Node {node_id} added as Learner at {addr}{}",
                if pre_seeded {
                    " (pre-seeded: Tier 3 skip expected)"
                } else {
                    ""
                }
            ),
        });

        // Step 2 (arch §Cluster Join Protocol, Steps 3-4): background task monitors
        // replication lag and promotes node when lag < READINESS_LAG_THRESHOLD.
        let raft = Arc::clone(&self.raft_node);
        let registry = Arc::clone(&self.join_registry);
        let tx_bg = tx.clone();

        tokio::spawn(async move {
            if let Err(e) = raft.monitor_and_promote(node_id, tx_bg.clone()).await {
                tracing::error!(node_id, error = %e, "JoinNode: background promotion failed");
                let _ = tx_bg.send(JoinProgressEvent {
                    node_id,
                    phase: JoinPhase::Failed,
                    lag_entries: 0,
                    percent: 0,
                    message: format!("Promotion failed: {e}"),
                });
            } else {
                tracing::info!(node_id, "JoinNode: node promoted to Voter");
            }
            // Clean up registry entry after join completes or fails.
            registry.lock().unwrap().remove(&node_id);
        });

        Ok(Response::new(JoinNodeResponse {
            node_id,
            status: "JOIN_INITIATED".to_string(),
        }))
    }

    type JoinProgressStream =
        Pin<Box<dyn Stream<Item = Result<JoinStatus, Status>> + Send + 'static>>;

    async fn join_progress(
        &self,
        req: Request<JoinProgressRequest>,
    ) -> Result<Response<Self::JoinProgressStream>, Status> {
        let node_id = req.into_inner().node_id;

        let rx = {
            let registry = self.join_registry.lock().unwrap();
            registry
                .get(&node_id)
                .ok_or_else(|| {
                    Status::not_found(format!(
                        "No active join for node {node_id}. \
                         Either JoinNode was not called, or the join has already completed."
                    ))
                })?
                .subscribe()
        };

        let stream = BroadcastStream::new(rx)
            // BroadcastStream yields Result<T, BroadcastStreamRecvError> where the error
            // is a Lagged variant (subscriber fell behind). We filter those out — the
            // client will miss intermediate events but will still receive phase transitions.
            .filter_map(|item| match item {
                Ok(event) => Some(Ok(event_to_proto(event))),
                Err(_lagged) => None,
            })
            // Stop streaming once COMPLETE or FAILED is observed.
            .take_while(|item| match item {
                Ok(status) => {
                    use crate::proto::admin::cluster::JoinPhase as P;
                    let phase = status.phase;
                    phase != P::Complete as i32 && phase != P::Failed as i32
                }
                Err(_) => false,
            });

        Ok(Response::new(Box::pin(stream)))
    }

    // ── Low-level node management ───────────────────────────────────────────

    async fn add_node(
        &self,
        req: Request<AddNodeRequest>,
    ) -> Result<Response<ClusterNode>, Status> {
        let _r = req.into_inner();

        // Node ID is auto-assigned from the next available slot for the low-level
        // AddNode (legacy API). Use JoinNode for operator-assigned IDs.
        // For now return an error directing to JoinNode.
        return Err(Status::unimplemented(
            "AddNode is not implemented for auto-assigned IDs. \
             Use JoinNode with an explicit node_id and address.",
        ));
    }

    async fn remove_node(
        &self,
        req: Request<RemoveNodeRequest>,
    ) -> Result<Response<RemoveNodeResponse>, Status> {
        let node_id: u64 = req
            .into_inner()
            .node_id
            .parse()
            .map_err(|e| Status::invalid_argument(format!("invalid node_id: {e}")))?;

        self.raft_node
            .remove_node(node_id)
            .await
            .map_err(|e| Status::internal(format!("remove_node failed: {e}")))?;

        tracing::info!(node_id, "RemoveNode: node removed from cluster membership");

        Ok(Response::new(RemoveNodeResponse {}))
    }

    // ── Decommission lifecycle ──────────────────────────────────────────────

    async fn decommission_node(
        &self,
        req: Request<DecommissionNodeRequest>,
    ) -> Result<Response<DecommissionNodeResponse>, Status> {
        let r = req.into_inner();
        let node_id = r.node_id;
        let pruning = r.pruning;
        let force = r.force;
        let skip_confirmation = r.skip_confirmation;

        // Guard: --force requires explicit confirmation to prevent accidental data loss.
        if force && !skip_confirmation {
            return Err(Status::failed_precondition(
                "Emergency decommission (force=true) requires skip_confirmation=true. \
                 This acknowledges that permanent data loss may occur if the node held \
                 the only copy of any shard. Re-run with --skip-confirmation to proceed.",
            ));
        }

        tracing::info!(
            node_id,
            pruning,
            force,
            "DecommissionNode: initiating decommission lifecycle"
        );

        // Self-decommission path (this node is the leader and is being decommissioned):
        //   1. Transfer leadership to a peer.
        //   2. Forward DecommissionNode to the new leader, which executes Phases 0-2.
        let is_self = node_id == self.raft_node.node_id();
        let is_leader = self.raft_node.is_leader().await;

        if is_self && is_leader && !force {
            return self
                .decommission_self(node_id, pruning, skip_confirmation)
                .await;
        }

        // Normal path: Phase 0 (quorum gate) + Phase 2 (membership remove).
        // Phase 0 is embedded in RaftNode::decommission_node().
        // The current node must be the Raft leader for change_membership to succeed.
        let result = self
            .raft_node
            .decommission_node(node_id, pruning, force)
            .await
            .map_err(|e| {
                // Map quorum/not-voter errors to FailedPrecondition; others to Internal.
                let msg = e.to_string();
                if msg.contains("voter set") || msg.contains("quorum") {
                    Status::failed_precondition(msg)
                } else {
                    Status::internal(msg)
                }
            })?;

        Ok(Response::new(DecommissionNodeResponse {
            node_id: result.node_id,
            message: result.message,
            operator_cleanup_required: result.operator_cleanup_required,
        }))
    }
}

// ── Self-decommission helper ───────────────────────────────────────────────────

impl ClusterServiceImpl {
    /// Handle the self-decommission case: this node is the leader being removed.
    ///
    /// Steps:
    /// 1. Find a peer voter to transfer leadership to.
    /// 2. Transfer leadership via openraft TimeoutNow.
    /// 3. Forward DecommissionNode to the new leader via gRPC (using the peer's address
    ///    from openraft membership config).
    /// 4. Return the forwarded response to the original caller.
    ///
    /// This keeps the service layer responsible for cross-node coordination,
    /// leaving `RaftNode::decommission_node()` to handle only local Raft operations.
    async fn decommission_self(
        &self,
        node_id: u64,
        pruning: bool,
        skip_confirmation: bool,
    ) -> Result<Response<DecommissionNodeResponse>, Status> {
        // Find a peer to transfer leadership to.
        let peer_id = self.raft_node.find_transfer_target().ok_or_else(|| {
            Status::failed_precondition(
                "self-decommission: no peer voter available for leadership transfer — \
                 cannot decommission the only cluster member",
            )
        })?;

        // Get peer's gRPC address from openraft membership config.
        let peer_addr = self.raft_node.node_address(peer_id).ok_or_else(|| {
            Status::internal(format!(
                "self-decommission: peer node {peer_id} has no advertise address in membership \
                 config. Cluster may have been bootstrapped without --addr flags."
            ))
        })?;

        tracing::info!(
            node_id,
            peer_id,
            peer_addr,
            "DecommissionNode self: transferring leadership before membership remove"
        );

        // Phase 1: Transfer leadership. Returns after 500ms or when transfer is confirmed.
        self.raft_node
            .transfer_leadership_to(peer_id)
            .await
            .map_err(|e| Status::internal(format!("leadership transfer failed: {e}")))?;

        tracing::info!(
            node_id,
            peer_id,
            "DecommissionNode self: leadership transferred, forwarding to new leader"
        );

        // Forward DecommissionNode to the new leader (the peer we transferred to).
        // The peer now executes Phases 0-2 and returns the result.
        let endpoint = if peer_addr.starts_with("http://") || peer_addr.starts_with("https://") {
            peer_addr.clone()
        } else {
            format!("http://{peer_addr}")
        };

        let mut client =
            crate::proto::admin::cluster::cluster_service_client::ClusterServiceClient::connect(
                endpoint,
            )
            .await
            .map_err(|e| {
                Status::unavailable(format!(
                    "self-decommission: could not connect to new leader at {peer_addr}: {e}"
                ))
            })?;

        let resp = client
            .decommission_node(DecommissionNodeRequest {
                node_id,
                pruning,
                force: false,
                skip_confirmation,
            })
            .await
            .map_err(|e| {
                Status::internal(format!(
                    "self-decommission: forwarded DecommissionNode to {peer_addr} failed: {e}"
                ))
            })?
            .into_inner();

        tracing::info!(
            node_id,
            message = resp.message,
            "DecommissionNode self: complete"
        );

        Ok(Response::new(resp))
    }
}
