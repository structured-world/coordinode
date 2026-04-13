//! Raft cluster management: node lifecycle, network, leadership.
//!
//! [`RaftNode`] is the top-level orchestrator that creates and manages an
//! openraft instance backed by CoordiNode storage. It wires together the log store,
//! state machine, network factory, and proposal pipeline into a single entity.
//!
//! ## Single-node CE
//!
//! For embedded/single-node deployments, use [`RaftNode::single_node()`] which
//! bootstraps a one-node cluster and immediately becomes leader.
//!
//! ## 3-node CE cluster
//!
//! For HA deployments (R133), nodes join an existing cluster via
//! `RaftNode::join()` (not yet implemented). The network layer handles
//! AppendEntries, Vote, and Snapshot RPCs between nodes.

pub(crate) mod grpc_server;
pub(crate) mod network;

use std::sync::Arc;

use coordinode_storage::engine::core::StorageEngine;

use crate::proposal::{RaftProposalPipeline, RateLimiter};
use crate::storage::{default_raft_config, CoordinodeStateMachine, LogStore, TypeConfig};
use crate::wait_majority::{BatchConfig, WaitForMajorityService};

use grpc_server::RaftGrpcHandler;
use network::{GrpcNetworkFactory, StubNetworkFactory};

use crate::proto::replication::raft_service_server::RaftServiceServer;

/// Raft node orchestrator.
///
/// Manages the lifecycle of an openraft instance, providing:
/// - Proposal pipeline for submitting writes
/// - Applied watermark for tracking state machine progress
/// - Leader status queries
///
/// ## Ownership
///
/// `RaftNode` owns the `Raft<TypeConfig>` instance. The state machine
/// and log store are consumed by openraft and managed internally.
///
/// **Must call [`shutdown()`](Self::shutdown) before dropping** to ensure
/// graceful leader transfer and WAL flush. Dropping without shutdown
/// may leave the node in an unclean state.
/// Type alias for the openraft Raft instance with our config + state machine.
type RaftInstance = openraft::Raft<TypeConfig, CoordinodeStateMachine>;

/// Configuration for the background snapshot trigger task.
///
/// openraft only supports entry-count-based triggers (`LogsSinceLast`).
/// This config adds WAL-size and periodic timer triggers as described
/// in the architecture (arch/distribution/consensus.md:82-85).
pub struct SnapshotTriggerConfig {
    /// Check interval for periodic trigger (default: 60s).
    pub check_interval: std::time::Duration,
    /// Disk space threshold in bytes to trigger snapshot (default: 256MB).
    /// Uses `StorageEngine::disk_space()` as proxy for WAL size.
    pub disk_space_threshold: u64,
}

impl Default for SnapshotTriggerConfig {
    fn default() -> Self {
        Self {
            check_interval: std::time::Duration::from_secs(60),
            disk_space_threshold: 256 * 1024 * 1024, // 256MB
        }
    }
}

pub struct RaftNode {
    /// The openraft instance.
    raft: Arc<RaftInstance>,
    /// Applied watermark subscriber (from state machine).
    applied_rx: tokio::sync::watch::Receiver<u64>,
    /// This node's ID.
    node_id: u64,
    /// Storage engine — held so `shutdown()` can flush before returning.
    engine: Arc<StorageEngine>,
    /// gRPC server shutdown signal. Dropped on shutdown to stop the server.
    _grpc_shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    /// Snapshot trigger background task abort handle.
    _snapshot_trigger: Option<tokio::task::JoinHandle<()>>,
}

impl RaftNode {
    /// Open a Raft node, handling both fresh start and restart.
    ///
    /// If the storage has no existing Raft state (fresh), initializes
    /// a single-member cluster and becomes leader.
    /// If the storage has existing state (restart after crash/shutdown),
    /// resumes from persisted state without re-initializing.
    ///
    /// This is the primary constructor. Use this instead of creating
    /// `Raft::new` + `initialize` manually.
    pub async fn open(node_id: u64, engine: Arc<StorageEngine>) -> Result<Self, RaftNodeError> {
        Self::open_with_oracle(node_id, engine, None).await
    }

    /// Open a Raft node with timestamp oracle for seqno advancement (R068).
    ///
    /// When oracle is provided, the state machine calls `oracle.advance_to(commit_ts)`
    /// before applying each entry's mutations. This ensures Raft replay produces
    /// identical seqnos as original application.
    pub async fn open_with_oracle(
        node_id: u64,
        engine: Arc<StorageEngine>,
        oracle: Option<Arc<coordinode_core::txn::timestamp::TimestampOracle>>,
    ) -> Result<Self, RaftNodeError> {
        let config = Arc::new(default_raft_config());
        let log_store =
            LogStore::open(Arc::clone(&engine)).map_err(|e| RaftNodeError::Init(e.to_string()))?;
        let state_machine = CoordinodeStateMachine::with_oracle(Arc::clone(&engine), oracle);

        let applied_rx = state_machine.subscribe_applied();

        let network = StubNetworkFactory;

        let raft: RaftInstance =
            openraft::Raft::new(node_id, config, network, log_store, state_machine)
                .await
                .map_err(|e: openraft::error::Fatal<TypeConfig>| {
                    RaftNodeError::Init(e.to_string())
                })?;

        // Try to initialize as single-node cluster.
        // On fresh start: succeeds, node becomes leader.
        // On restart: returns NotAllowed (already has state) — expected, skip.
        // Other errors (NotInMembers, Fatal): propagate.
        let mut members = std::collections::BTreeMap::new();
        members.insert(node_id, openraft::impls::BasicNode::default());

        match raft.initialize(members).await {
            Ok(_) => {
                tracing::info!(node_id, "fresh raft node initialized");
            }
            Err(openraft::error::RaftError::APIError(
                openraft::error::InitializeError::NotAllowed(_),
            )) => {
                // Already initialized — restart path.
                // openraft's startup() restores leader state from the persisted
                // committed vote without a new election. Calling trigger().elect()
                // here would bump the term (uncommitted) while startup() has already
                // restored committed leadership, causing the engine invariant
                // `leader.committed_vote >= state.vote` to be violated and temporarily
                // putting the node in a non-leader state.
                //
                // For multi-node clusters: natural election timeout (300-600ms) handles
                // leader recovery if this node was a follower before the restart.
                tracing::debug!(
                    node_id,
                    "raft already initialized, resuming from existing state"
                );
            }
            Err(e) => {
                // Real error: NotInMembers, Fatal, etc.
                return Err(RaftNodeError::Init(format!("initialize failed: {e}")));
            }
        }

        let raft = Arc::new(raft);

        Ok(Self {
            raft,
            applied_rx,
            node_id,
            engine,
            _grpc_shutdown: None,
            _snapshot_trigger: None,
        })
    }

    /// Bootstrap a single-node cluster (convenience wrapper).
    ///
    /// Equivalent to `open(1, engine)`. Uses stub network (no gRPC server).
    pub async fn single_node(engine: Arc<StorageEngine>) -> Result<Self, RaftNodeError> {
        Self::open(1, engine).await
    }

    /// Open a Raft node with gRPC networking for multi-node cluster.
    ///
    /// Starts a gRPC server on `listen_addr` for inter-node Raft RPCs
    /// and uses `GrpcNetworkFactory` for outbound connections to peers.
    ///
    /// ## Bootstrap protocol
    ///
    /// - **First node** (`peers` empty): initializes single-member cluster,
    ///   becomes leader. Other nodes join via `add_node()`.
    /// - **Joining node** (`peers` non-empty): creates Raft instance without
    ///   `initialize()`. The leader must call `add_node()` to add this node
    ///   to the cluster membership.
    /// - **Restart** (existing state): resumes from persisted state regardless
    ///   of `peers` argument.
    pub async fn open_cluster(
        node_id: u64,
        engine: Arc<StorageEngine>,
        listen_addr: std::net::SocketAddr,
        advertise_addr: String,
    ) -> Result<Self, RaftNodeError> {
        Self::open_cluster_with_snapshot_config(
            node_id,
            engine,
            listen_addr,
            advertise_addr,
            SnapshotTriggerConfig::default(),
        )
        .await
    }

    /// Like `open_cluster` but with custom snapshot trigger configuration.
    ///
    /// Useful for tests that need a short trigger interval.
    pub async fn open_cluster_with_snapshot_config(
        node_id: u64,
        engine: Arc<StorageEngine>,
        listen_addr: std::net::SocketAddr,
        advertise_addr: String,
        snap_config: SnapshotTriggerConfig,
    ) -> Result<Self, RaftNodeError> {
        let config = Arc::new(default_raft_config());
        let log_store =
            LogStore::open(Arc::clone(&engine)).map_err(|e| RaftNodeError::Init(e.to_string()))?;
        let state_machine = CoordinodeStateMachine::new(Arc::clone(&engine));
        let applied_rx = state_machine.subscribe_applied();

        let network = GrpcNetworkFactory;

        let raft: RaftInstance =
            openraft::Raft::new(node_id, config, network, log_store, state_machine)
                .await
                .map_err(|e: openraft::error::Fatal<TypeConfig>| {
                    RaftNodeError::Init(e.to_string())
                })?;

        let raft = Arc::new(raft);

        // Try initialize — succeeds on fresh, NotAllowed on restart
        let mut members = std::collections::BTreeMap::new();
        let node_info = openraft::impls::BasicNode {
            addr: advertise_addr,
        };
        members.insert(node_id, node_info);

        match raft.initialize(members).await {
            Ok(_) => {
                tracing::info!(node_id, "fresh cluster node initialized as leader");
            }
            Err(openraft::error::RaftError::APIError(
                openraft::error::InitializeError::NotAllowed(_),
            )) => {
                tracing::debug!(
                    node_id,
                    "raft already initialized, resuming from existing state"
                );
            }
            Err(e) => {
                return Err(RaftNodeError::Init(format!("initialize failed: {e}")));
            }
        }

        // Start gRPC server for inter-node Raft RPCs with shutdown signal
        let handler = RaftGrpcHandler::new(Arc::clone(&raft), Arc::clone(&engine));
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

        let server =
            tonic::transport::Server::builder().add_service(RaftServiceServer::new(handler));

        tokio::spawn(async move {
            let graceful = server.serve_with_shutdown(listen_addr, async {
                let _ = shutdown_rx.await;
            });
            if let Err(e) = graceful.await {
                tracing::error!(%e, "raft gRPC server failed");
            }
        });

        tracing::info!(node_id, %listen_addr, "raft gRPC server started");

        // Start background snapshot trigger (WAL size + periodic timer)
        let snap_handle =
            spawn_snapshot_trigger(Arc::clone(&raft), Arc::clone(&engine), snap_config);

        Ok(Self {
            raft,
            applied_rx,
            node_id,
            engine,
            _grpc_shutdown: Some(shutdown_tx),
            _snapshot_trigger: Some(snap_handle),
        })
    }

    /// Open a Raft node that joins an existing cluster.
    ///
    /// Does NOT call `initialize()` — the node waits for the leader to
    /// add it via `add_node()`. This avoids creating conflicting single-node
    /// clusters when bootstrapping a multi-node cluster.
    ///
    /// Used by nodes 2+ in the bootstrap sequence.
    pub async fn open_joining(
        node_id: u64,
        engine: Arc<StorageEngine>,
        listen_addr: std::net::SocketAddr,
    ) -> Result<Self, RaftNodeError> {
        Self::open_joining_with_snapshot_config(
            node_id,
            engine,
            listen_addr,
            SnapshotTriggerConfig::default(),
        )
        .await
    }

    /// Like `open_joining` but with custom snapshot trigger configuration.
    pub async fn open_joining_with_snapshot_config(
        node_id: u64,
        engine: Arc<StorageEngine>,
        listen_addr: std::net::SocketAddr,
        snap_config: SnapshotTriggerConfig,
    ) -> Result<Self, RaftNodeError> {
        let config = Arc::new(default_raft_config());
        let log_store =
            LogStore::open(Arc::clone(&engine)).map_err(|e| RaftNodeError::Init(e.to_string()))?;
        let state_machine = CoordinodeStateMachine::new(Arc::clone(&engine));
        let applied_rx = state_machine.subscribe_applied();

        let network = GrpcNetworkFactory;

        let raft: RaftInstance =
            openraft::Raft::new(node_id, config, network, log_store, state_machine)
                .await
                .map_err(|e: openraft::error::Fatal<TypeConfig>| {
                    RaftNodeError::Init(e.to_string())
                })?;

        let raft = Arc::new(raft);

        let handler = RaftGrpcHandler::new(Arc::clone(&raft), Arc::clone(&engine));
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

        let server =
            tonic::transport::Server::builder().add_service(RaftServiceServer::new(handler));

        tokio::spawn(async move {
            let graceful = server.serve_with_shutdown(listen_addr, async {
                let _ = shutdown_rx.await;
            });
            if let Err(e) = graceful.await {
                tracing::error!(%e, "raft gRPC server failed");
            }
        });

        tracing::info!(node_id, %listen_addr, "joining node started (waiting for leader)");

        let snap_handle =
            spawn_snapshot_trigger(Arc::clone(&raft), Arc::clone(&engine), snap_config);

        Ok(Self {
            raft,
            applied_rx,
            node_id,
            engine,
            _grpc_shutdown: Some(shutdown_tx),
            _snapshot_trigger: Some(snap_handle),
        })
    }

    /// Add a new node to the cluster (leader-only).
    ///
    /// Must be called on the leader node. Adds the node as a learner first
    /// (receives log replication but doesn't vote), then promotes to voter.
    ///
    /// ## Bootstrap sequence for 3-node cluster:
    /// 1. Node 1: `open_cluster(1, engine, addr1, addr1)` → becomes leader
    /// 2. Node 2: `open_cluster(2, engine, addr2, addr2)` → starts, no init
    /// 3. Node 3: `open_cluster(3, engine, addr3, addr3)` → starts, no init
    /// 4. On node 1: `add_node(2, addr2)`, `add_node(3, addr3)`
    /// 5. On node 1: `change_membership([1, 2, 3])` → all become voters
    pub async fn add_node(&self, node_id: u64, addr: String) -> Result<(), RaftNodeError> {
        let node_info = openraft::impls::BasicNode { addr };

        // Add as learner (non-voting, receives log replication)
        self.raft
            .add_learner(node_id, node_info, true)
            .await
            .map_err(|e| RaftNodeError::Membership(e.to_string()))?;

        tracing::info!(node_id, "added node as learner");
        Ok(())
    }

    /// Change cluster membership (leader-only).
    ///
    /// Sets the voting members to the given set of node IDs.
    /// All nodes must have been previously added as learners.
    pub async fn change_membership(&self, member_ids: Vec<u64>) -> Result<(), RaftNodeError> {
        let members: std::collections::BTreeSet<u64> = member_ids.into_iter().collect();

        self.raft
            .change_membership(members, false)
            .await
            .map_err(|e| RaftNodeError::Membership(e.to_string()))?;

        tracing::info!("cluster membership updated");
        Ok(())
    }

    /// Remove a node from the cluster (leader-only).
    ///
    /// Changes membership to exclude the given node ID by setting
    /// the new membership to all current voters minus the target.
    /// The removed node will stop receiving log replication and can
    /// be shut down.
    pub async fn remove_node(&self, node_id: u64) -> Result<(), RaftNodeError> {
        use openraft::rt::watch::WatchReceiver;

        // Get current voter set from raft metrics.
        // Must collect inside borrow scope — voter_ids() borrows from the ref guard.
        let current_members: std::collections::BTreeSet<u64> = {
            let rx = self.raft.metrics();
            let metrics_ref = rx.borrow_watched();
            let ids: Vec<u64> = metrics_ref
                .membership_config
                .membership()
                .voter_ids()
                .collect();
            ids.into_iter().collect()
        };

        let new_members: std::collections::BTreeSet<u64> = current_members
            .into_iter()
            .filter(|&id| id != node_id)
            .collect();

        if new_members.is_empty() {
            return Err(RaftNodeError::Membership(
                "cannot remove last voting member".to_string(),
            ));
        }

        self.raft
            .change_membership(new_members, false)
            .await
            .map_err(|e| RaftNodeError::Membership(e.to_string()))?;

        tracing::info!(node_id, "removed node from cluster");
        Ok(())
    }

    /// Create a [`RaftProposalPipeline`] for submitting proposals.
    ///
    /// The pipeline can be shared across threads via `Arc`. It includes
    /// rate limiting and retry logic. Each proposal gets its own Raft
    /// log entry and round-trip.
    ///
    /// For high-concurrency scenarios (>100 concurrent writers), prefer
    /// [`batch_pipeline()`](Self::batch_pipeline) which coalesces
    /// proposals into fewer Raft entries.
    pub fn pipeline(&self) -> RaftProposalPipeline {
        RaftProposalPipeline::new(Arc::clone(&self.raft))
    }

    /// Create a [`WaitForMajorityService`] for batched proposal submission.
    ///
    /// Coalesces concurrent proposals into fewer Raft log entries,
    /// reducing N round-trips to ~1 under high concurrency. Each writer
    /// still gets an individual result.
    ///
    /// The returned service spawns a background drain task. Call
    /// [`WaitForMajorityService::shutdown()`] before dropping for
    /// graceful cleanup.
    pub fn batch_pipeline(&self) -> WaitForMajorityService {
        WaitForMajorityService::spawn_default(Arc::clone(&self.raft), RateLimiter::default())
    }

    /// Create a [`WaitForMajorityService`] with custom batch configuration.
    pub fn batch_pipeline_with_config(&self, config: BatchConfig) -> WaitForMajorityService {
        WaitForMajorityService::spawn(Arc::clone(&self.raft), RateLimiter::default(), config)
    }

    /// Get the current applied log index (non-blocking).
    pub fn applied_index(&self) -> u64 {
        *self.applied_rx.borrow()
    }

    /// Wait until the applied log index reaches at least `target`, with timeout.
    ///
    /// Returns `Ok(index)` when applied index >= target.
    /// Returns `Err(current_index)` if timeout expires before target is reached.
    /// Used for linearizable reads: follower waits until
    /// `Applied >= query.readTs` before serving the read.
    pub async fn wait_for_applied(
        &mut self,
        target: u64,
        timeout: std::time::Duration,
    ) -> Result<u64, u64> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let current = *self.applied_rx.borrow();
            if current >= target {
                return Ok(current);
            }
            // Wait for next update or timeout
            let changed = tokio::time::timeout_at(deadline, self.applied_rx.changed()).await;
            match changed {
                Ok(Ok(())) => continue,            // New value, re-check
                Ok(Err(_)) => return Err(current), // Sender dropped
                Err(_) => return Err(current),     // Timeout
            }
        }
    }

    /// Check if this node is currently the Raft leader.
    pub async fn is_leader(&self) -> bool {
        self.raft
            .ensure_linearizable(openraft::raft::ReadPolicy::LeaseRead)
            .await
            .is_ok()
    }

    /// Ensure linearizable read: verify this node is leader with a fresh
    /// lease (no-op Raft round-trip equivalent).
    ///
    /// openraft's `ensure_linearizable(LeaseRead)` confirms leadership via
    /// the heartbeat lease, ensuring all prior committed entries are visible.
    /// Returns the current applied log index after confirmation.
    ///
    /// Returns `Err` if this node is not leader or the cluster is partitioned.
    pub async fn ensure_linearizable_read(&self) -> Result<u64, RaftNodeError> {
        self.raft
            .ensure_linearizable(openraft::raft::ReadPolicy::LeaseRead)
            .await
            .map_err(|e| RaftNodeError::ReadConcern(format!("linearizable: {e}")))?;

        Ok(self.applied_index())
    }

    /// Get the current Raft commit index (last applied on this node).
    ///
    /// Returns `last_applied.index` — the highest log entry that this node's
    /// state machine has applied. Data at or below this index is durable and
    /// visible to reads. Used for `readConcern: "majority"`.
    ///
    /// Note: `last_log_index` would return the highest entry received in the
    /// log, which may not yet be applied (committed but pending apply). We
    /// return `last_applied` to match what callers can actually read.
    pub fn commit_index(&self) -> u64 {
        use openraft::async_runtime::watch::WatchReceiver;

        let metrics = self.raft.metrics().borrow_watched().clone();
        metrics
            .last_applied
            .as_ref()
            .map(|lid| lid.index)
            .unwrap_or(0)
    }

    /// Subscribe to applied index updates.
    ///
    /// Returns a clone of the applied watermark receiver. The receiver
    /// delivers the latest applied log index whenever the state machine
    /// advances. Used by [`ReadFence`] for causal read fencing.
    pub fn subscribe_applied(&self) -> tokio::sync::watch::Receiver<u64> {
        self.applied_rx.clone()
    }

    /// Create a per-request read fence for enforcing read preference and concern.
    ///
    /// The returned [`ReadFence`] is cheap to create — it clones a watch
    /// receiver (pointer copy) and an Arc clone. Call [`ReadFence::apply()`]
    /// before executing a query to enforce routing and consistency guarantees.
    pub fn read_fence(&self) -> crate::read_fence::ReadFence {
        crate::read_fence::ReadFence::new(self.applied_rx.clone(), Arc::clone(&self.raft))
    }

    /// Get this node's ID.
    pub fn node_id(&self) -> u64 {
        self.node_id
    }

    /// Get a reference to the underlying openraft instance.
    ///
    /// For advanced operations (membership changes, leader transfer, etc.)
    /// that aren't exposed through `RaftNode` methods.
    pub fn raft(&self) -> &Arc<RaftInstance> {
        &self.raft
    }

    /// Transfer leadership to a specific peer node.
    ///
    /// Sends a `TimeoutNow` message to the target, triggering an immediate
    /// election without waiting for `election_timeout` (300-600ms).
    /// The target wins the election in <1ms.
    ///
    /// Returns `Ok(())` when the transfer is initiated. The caller should
    /// verify leadership change via [`is_leader()`](Self::is_leader) or
    /// the metrics watch.
    ///
    /// No-op if this node is not the leader.
    pub async fn transfer_leadership_to(&self, target_id: u64) -> Result<(), RaftNodeError> {
        if !self.is_leader().await {
            tracing::debug!(
                node_id = self.node_id,
                "not leader, skipping transfer_leadership_to"
            );
            return Ok(());
        }

        tracing::info!(
            node_id = self.node_id,
            target_id,
            "initiating leadership transfer"
        );

        self.raft
            .trigger()
            .transfer_leader(target_id)
            .await
            .map_err(|e| RaftNodeError::Shutdown(format!("transfer_leader: {e}")))?;

        // Allow time for the transfer to complete (Dgraph pattern: 1s sleep).
        // In practice, TimeoutNow election completes in <1ms, but we need to
        // wait for the full election round-trip + log catchup.
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        if self.is_leader().await {
            tracing::warn!(
                node_id = self.node_id,
                target_id,
                "still leader after transfer attempt"
            );
        } else {
            tracing::info!(
                node_id = self.node_id,
                target_id,
                "leadership transfer successful"
            );
        }

        Ok(())
    }

    /// Force a snapshot at the current applied index.
    ///
    /// Triggers openraft's snapshot mechanism and waits briefly for it
    /// to complete. Used for pre-shutdown checkpointing and manual
    /// log compaction.
    pub async fn checkpoint(&self) -> Result<(), RaftNodeError> {
        self.raft
            .trigger()
            .snapshot()
            .await
            .map_err(|e| RaftNodeError::Shutdown(format!("checkpoint snapshot: {e}")))?;

        // Wait for snapshot to build (typically <100ms for small datasets)
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        tracing::info!(
            node_id = self.node_id,
            applied = self.applied_index(),
            "checkpoint complete"
        );
        Ok(())
    }

    /// Graceful shutdown with leadership transfer.
    ///
    /// Orchestrates the shutdown sequence (Dgraph `checkpointAndClose` pattern):
    /// 1. Force a checkpoint snapshot (persist current state)
    /// 2. If leader: find a peer and transfer leadership (TimeoutNow)
    /// 3. Stop the Raft instance
    ///
    /// This ensures <5s failover: the new leader doesn't wait for
    /// election timeout (300-600ms) and has a recent snapshot.
    pub async fn graceful_shutdown(&self) -> Result<(), RaftNodeError> {
        tracing::info!(node_id = self.node_id, "starting graceful shutdown");

        // Abort background snapshot trigger first
        if let Some(ref handle) = self._snapshot_trigger {
            handle.abort();
        }

        // Step 1: Checkpoint — flush current state to snapshot
        if let Err(e) = self.checkpoint().await {
            tracing::warn!("checkpoint before shutdown failed: {e}");
            // Continue shutdown even if checkpoint fails
        }

        // Step 2: Transfer leadership if we're the leader
        if self.is_leader().await {
            if let Some(target_id) = self.find_transfer_target() {
                if let Err(e) = self.transfer_leadership_to(target_id).await {
                    tracing::warn!(target_id, "leadership transfer failed: {e}");
                }
            } else {
                tracing::debug!("no peer available for leadership transfer");
            }
        }

        // Step 3: Stop Raft
        self.raft
            .shutdown()
            .await
            .map_err(|e| RaftNodeError::Shutdown(e.to_string()))?;

        tracing::info!(node_id = self.node_id, "graceful shutdown complete");
        Ok(())
    }

    /// Trigger a graceful shutdown.
    ///
    /// If this node is a cluster member (has peers), performs graceful shutdown
    /// with leadership transfer. Otherwise, does a simple shutdown.
    /// Find a voter peer to transfer leadership to.
    ///
    /// Returns the first voter ID that isn't this node, or None if
    /// this is a single-node cluster.
    fn find_transfer_target(&self) -> Option<u64> {
        use openraft::async_runtime::watch::WatchReceiver;

        let metrics = self.raft.metrics().borrow_watched().clone();
        let joint = metrics.membership_config.membership().get_joint_config();
        let voters = joint.first()?;
        voters.iter().find(|&&id| id != self.node_id).copied()
    }

    pub async fn shutdown(&self) -> Result<(), RaftNodeError> {
        // Check if we have peers (cluster mode vs single-node)
        let has_peers = self.find_transfer_target().is_some();

        let result = if has_peers {
            self.graceful_shutdown().await
        } else {
            // Single-node: simple shutdown
            if let Some(ref handle) = self._snapshot_trigger {
                handle.abort();
            }
            self.raft
                .shutdown()
                .await
                .map_err(|e| RaftNodeError::Shutdown(e.to_string()))
        };

        // Flush active memtables to SST so Phase 2 (reopen) sees all writes.
        //
        // openraft's internal tasks hold Arc<LogStore> and Arc<StateMachine>,
        // both of which hold Arc<StorageEngine>. After raft.shutdown() the tasks
        // have stopped writing, but tokio may not drop the task futures (and their
        // Arc refs) before the caller opens the same directory again. Calling
        // persist() here guarantees durability before shutdown() returns.
        if let Err(e) = self.engine.persist() {
            tracing::warn!(error = %e, "engine persist on shutdown failed (best-effort)");
        }

        result
    }

    // ── Replication status & staleness tracking (R140) ──

    /// Get the current replication status for all cluster nodes.
    ///
    /// Only meaningful when called on the leader — followers don't track
    /// per-node replication progress. Returns `None` if not leader.
    ///
    /// Staleness = `leader_last_log_index - node_matched_index`.
    /// Used by read routing to exclude stale followers.
    pub fn replication_status(&self) -> Option<Vec<NodeReplicationStatus>> {
        use openraft::async_runtime::watch::WatchReceiver;
        use openraft::ServerState;

        let metrics = self.raft.metrics().borrow_watched().clone();

        // Only leader has replication metrics
        if metrics.state != ServerState::Leader {
            return None;
        }

        let leader_last_log = metrics.last_log_index.unwrap_or(0);
        let replication = metrics.replication.as_ref()?;
        let heartbeat = metrics.heartbeat.as_ref();

        let mut statuses = Vec::with_capacity(replication.len() + 1);

        // Leader itself
        statuses.push(NodeReplicationStatus {
            node_id: self.node_id,
            role: NodeRole::Leader,
            matched_index: leader_last_log,
            lag_entries: 0,
            last_heartbeat_ago_ms: None, // Leader doesn't heartbeat itself
        });

        // Followers/learners
        for (&node_id, matched_log_id) in replication {
            if node_id == self.node_id {
                continue; // Skip self
            }

            let matched_index = matched_log_id.as_ref().map(|id| id.index).unwrap_or(0);

            let lag = leader_last_log.saturating_sub(matched_index);

            let last_hb_ago = heartbeat.and_then(|hb| {
                hb.get(&node_id).and_then(|ts| {
                    ts.as_ref().map(|t| {
                        use openraft::async_runtime::instant::Instant;
                        t.elapsed().as_millis() as u64
                    })
                })
            });

            // Determine role from membership config
            let joint = metrics.membership_config.membership().get_joint_config();
            let is_voter = joint
                .first()
                .map(|voters| voters.contains(&node_id))
                .unwrap_or(false);

            statuses.push(NodeReplicationStatus {
                node_id,
                role: if is_voter {
                    NodeRole::Follower
                } else {
                    NodeRole::Learner
                },
                matched_index,
                lag_entries: lag,
                last_heartbeat_ago_ms: last_hb_ago,
            });
        }

        Some(statuses)
    }

    /// Check if this node's applied index is within acceptable staleness
    /// of the given leader commit index.
    ///
    /// `max_lag_entries` is the maximum number of entries this node can
    /// be behind before being considered too stale for reads.
    ///
    /// Used by follower read routing: if stale, exclude from read candidates.
    pub fn is_within_staleness(&self, leader_commit_index: u64, max_lag_entries: u64) -> bool {
        let applied = self.applied_index();
        leader_commit_index.saturating_sub(applied) <= max_lag_entries
    }
}

/// Replication status for a single cluster node.
///
/// Reported by the leader for read routing and monitoring.
#[derive(Debug, Clone)]
pub struct NodeReplicationStatus {
    /// Node ID.
    pub node_id: u64,
    /// Role in the cluster.
    pub role: NodeRole,
    /// Last log index confirmed replicated to this node.
    pub matched_index: u64,
    /// Number of log entries behind the leader's last log.
    pub lag_entries: u64,
    /// Milliseconds since last heartbeat acknowledgment (None if unknown).
    pub last_heartbeat_ago_ms: Option<u64>,
}

/// Node role in a Raft cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeRole {
    /// Current Raft leader (handles all writes).
    Leader,
    /// Voting follower (participates in elections and quorum).
    Follower,
    /// Non-voting learner (receives replication but doesn't vote).
    Learner,
}

/// Spawn a background task that periodically checks snapshot triggers.
///
/// Complements openraft's built-in `LogsSinceLast(10_000)` with:
/// - **Disk space trigger**: if total disk space exceeds threshold (proxy for WAL size)
/// - **Periodic timer**: ensures snapshots happen even during read-only periods
///
/// The task runs until the Raft instance is shut down (detected via `trigger()` error).
fn spawn_snapshot_trigger(
    raft: Arc<RaftInstance>,
    engine: Arc<StorageEngine>,
    config: SnapshotTriggerConfig,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(config.check_interval);
        // Don't fire immediately on startup
        interval.tick().await;

        loop {
            interval.tick().await;

            // Check disk space threshold
            let should_snapshot = match engine.disk_space() {
                Ok(space) if space >= config.disk_space_threshold => {
                    tracing::info!(
                        disk_space_bytes = space,
                        threshold = config.disk_space_threshold,
                        "snapshot trigger: disk space threshold exceeded"
                    );
                    true
                }
                _ => {
                    // Even if disk space check fails or is below threshold,
                    // the periodic timer itself is a trigger reason:
                    // "60 seconds check interval" from arch doc means we check
                    // AND trigger if no recent snapshot exists.
                    // openraft's LogsSinceLast handles the "no new entries" case
                    // (won't build empty snapshot), so triggering is safe.
                    true
                }
            };

            if should_snapshot {
                match raft.trigger().snapshot().await {
                    Ok(()) => {
                        tracing::debug!("snapshot trigger: requested snapshot build");
                    }
                    Err(_fatal) => {
                        // Raft instance shut down — exit the trigger loop
                        tracing::debug!("snapshot trigger: raft shut down, stopping");
                        break;
                    }
                }
            }
        }
    })
}

/// Errors from RaftNode lifecycle operations.
#[derive(Debug, thiserror::Error)]
pub enum RaftNodeError {
    #[error("failed to initialize raft node: {0}")]
    Init(String),

    #[error("failed to shut down raft node: {0}")]
    Shutdown(String),

    #[error("membership change failed: {0}")]
    Membership(String),

    #[error("read concern check failed: {0}")]
    ReadConcern(String),
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use coordinode_core::txn::proposal::{
        Mutation, PartitionId, ProposalIdGenerator, ProposalPipeline, RaftProposal,
    };
    use coordinode_core::txn::timestamp::Timestamp;
    use coordinode_storage::engine::config::StorageConfig;
    use coordinode_storage::engine::partition::Partition;

    fn test_engine() -> (tempfile::TempDir, Arc<StorageEngine>) {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = StorageConfig::new(dir.path());
        let engine = Arc::new(StorageEngine::open(&config).expect("open"));
        (dir, engine)
    }

    #[tokio::test]
    async fn single_node_bootstrap_becomes_leader() {
        let (_dir, engine) = test_engine();
        let node = RaftNode::single_node(engine).await.expect("bootstrap");

        // Single-node cluster should become leader almost immediately
        // Give it a moment for the election
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        assert!(node.is_leader().await, "single node should be leader");
        assert_eq!(node.node_id(), 1);

        node.shutdown().await.expect("shutdown");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn single_node_propose_and_read() {
        let (_dir, engine) = test_engine();
        let engine_clone = Arc::clone(&engine);
        let node = RaftNode::single_node(engine).await.expect("bootstrap");

        // Wait for leadership
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        let pipeline = node.pipeline();
        let id_gen = ProposalIdGenerator::new();

        let proposal = RaftProposal {
            id: id_gen.next(),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:1:42".to_vec(),
                value: b"hello-raft".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(100),
            start_ts: Timestamp::from_raw(99),
            bypass_rate_limiter: false,
        };

        // Propose through Raft pipeline
        pipeline.propose_and_wait(&proposal).expect("propose");

        // Verify data was applied to storage (ADR-016: plain keys, no versioned encoding)
        let result = engine_clone
            .get(Partition::Node, b"node:1:42")
            .expect("read");
        assert_eq!(result.as_deref(), Some(b"hello-raft".as_slice()));

        node.shutdown().await.expect("shutdown");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn applied_watermark_advances() {
        let (_dir, engine) = test_engine();
        let mut node = RaftNode::single_node(engine).await.expect("bootstrap");

        // Wait for leadership
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        // Initial watermark may be >0 due to membership entry
        let initial = node.applied_index();

        let pipeline = node.pipeline();
        let id_gen = ProposalIdGenerator::new();

        // Submit a proposal
        let proposal = RaftProposal {
            id: id_gen.next(),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:1:1".to_vec(),
                value: b"data".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(100),
            start_ts: Timestamp::from_raw(99),
            bypass_rate_limiter: false,
        };
        pipeline.propose_and_wait(&proposal).expect("propose");

        // Watermark should have advanced
        let after = node.applied_index();
        assert!(
            after > initial,
            "watermark should advance: initial={initial}, after={after}"
        );

        // wait_for_applied should return immediately for already-applied index
        let result = node
            .wait_for_applied(after, std::time::Duration::from_secs(5))
            .await
            .expect("should not timeout for already-applied index");
        assert_eq!(result, after);

        node.shutdown().await.expect("shutdown");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn multiple_proposals_sequential() {
        let (_dir, engine) = test_engine();
        let engine_clone = Arc::clone(&engine);
        let node = RaftNode::single_node(engine).await.expect("bootstrap");

        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        let pipeline = node.pipeline();
        let id_gen = ProposalIdGenerator::new();

        // Submit 5 proposals
        for i in 1..=5u64 {
            let proposal = RaftProposal {
                id: id_gen.next(),
                mutations: vec![Mutation::Put {
                    partition: PartitionId::Node,
                    key: format!("node:1:{i}").into_bytes(),
                    value: format!("value-{i}").into_bytes(),
                }],
                commit_ts: Timestamp::from_raw(100 + i),
                start_ts: Timestamp::from_raw(99 + i),
                bypass_rate_limiter: false,
            };
            pipeline.propose_and_wait(&proposal).expect("propose");
        }

        // Verify all 5 writes (ADR-016: plain keys, direct engine reads)
        for i in 1..=5u64 {
            let result = engine_clone
                .get(Partition::Node, format!("node:1:{i}").as_bytes())
                .expect("read");
            assert_eq!(
                result.as_deref(),
                Some(format!("value-{i}").as_bytes()),
                "mismatch at i={i}"
            );
        }

        node.shutdown().await.expect("shutdown");
    }

    #[tokio::test]
    async fn graceful_shutdown() {
        let (_dir, engine) = test_engine();
        let node = RaftNode::single_node(engine).await.expect("bootstrap");

        tokio::time::sleep(std::time::Duration::from_millis(300)).await;

        // Shutdown should succeed cleanly
        node.shutdown().await.expect("shutdown");
    }
}
