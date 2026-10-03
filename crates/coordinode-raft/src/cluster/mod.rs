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
//! For HA deployments, a node joins an existing cluster via
//! [`RaftNode::open_joining`] as a learner, and
//! [`RaftNode::monitor_and_promote`] promotes it to a voter once it has
//! caught up. The network layer handles AppendEntries, Vote, and Snapshot
//! RPCs between nodes.

pub mod grpc_server;
pub mod nemesis;
pub(crate) mod network;
pub mod version;

use std::sync::Arc;

use coordinode_storage::engine::core::StorageEngine;

use crate::proposal::{RaftProposalPipeline, RateLimiter};
use crate::storage::{
    AppendNotifier, CoordinodeStateMachine, LogStore, TypeConfig, default_raft_config,
};
use crate::wait_majority::{BatchConfig, WaitForMajorityService};

pub use grpc_server::RaftGrpcHandler;
use network::{GrpcNetworkFactory, StubNetworkFactory};
use version::{HandshakeService, VersionGate};

use crate::proto::replication::raft_service_server::RaftServiceServer;

/// Default for how long a membership change waits for the previous one to
/// settle, in ms. A change commits in one replication round to the voters it
/// involves, so running out of this means the group has lost the quorum to
/// commit, which is an error to report rather than a delay to extend.
pub const DEFAULT_MEMBERSHIP_SETTLE_TIMEOUT_MS: u64 = 30_000;

/// Type alias for the openraft Raft instance with our config + state machine.
type RaftInstance = openraft::Raft<TypeConfig, CoordinodeStateMachine>;

/// When a node snapshots its state so the Raft log before it can go.
///
/// A snapshot is taken when any of the three is reached: entries applied
/// since the last one, bytes the log grew by since the last one (a few
/// large entries compact the log as a count of small ones would), or the
/// periodic timer while anything new was applied.
pub struct SnapshotTriggerConfig {
    /// Entries applied since the last snapshot (default 10 000).
    pub logs_since_last: u64,
    /// Bytes the Raft log's segments grew by since the last snapshot
    /// (default 256 MiB).
    pub log_bytes: u64,
    /// Longest time between snapshots while entries are applied (default
    /// 60 s).
    pub check_interval: std::time::Duration,
}

impl Default for SnapshotTriggerConfig {
    fn default() -> Self {
        Self {
            logs_since_last: 10_000,
            log_bytes: 256 * 1024 * 1024,
            check_interval: std::time::Duration::from_secs(60),
        }
    }
}

impl SnapshotTriggerConfig {
    /// The openraft configuration under this snapshot policy: the entry
    /// count is openraft's own trigger; the other two are the trigger task's.
    fn raft_config(&self) -> openraft::Config {
        openraft::Config {
            snapshot_policy: openraft::SnapshotPolicy::LogsSinceLast(self.logs_since_last),
            ..default_raft_config()
        }
    }
}

/// How a node is opened beyond its identity, storage and address.
#[derive(Default)]
pub struct NodeOptions {
    /// When the node snapshots its state.
    pub snapshots: SnapshotTriggerConfig,
    /// The embedding application's format epoch, half of the version pair
    /// members are matched on. A server runs zero.
    pub host_epoch: u64,
}

/// The consensus group a node's handshake speaks for: one group per node
/// today.
const GROUP_ID: u64 = 0;

/// The version gate of member `node_id`, over its state machine's record
/// of the group's pair.
fn version_gate(
    node_id: u64,
    state_machine: &CoordinodeStateMachine,
    host_epoch: u64,
) -> Arc<VersionGate> {
    Arc::new(VersionGate::new(
        node_id,
        GROUP_ID,
        coordinode_core::version::VersionPair::current(host_epoch),
        state_machine.subscribe_group_pair(),
        state_machine.applied_commit_ts_handle(),
    ))
}

/// The frozen version exchange of the member `gate` speaks for, to serve
/// beside its consensus service.
fn handshake_server(
    gate: &Arc<VersionGate>,
) -> crate::proto::internode::version_handshake_server::VersionHandshakeServer<HandshakeService> {
    crate::proto::internode::version_handshake_server::VersionHandshakeServer::new(
        HandshakeService::new(Arc::clone(gate)),
    )
}

/// Keep the gate's view of the leader current, and while this node leads,
/// record its pair as the group's when the group does not run it yet: the
/// first leader at a pair commits it as soon as it leads.
fn spawn_version_watch(
    raft: Arc<RaftInstance>,
    gate: Arc<VersionGate>,
    oracle: Option<Arc<coordinode_core::txn::timestamp::TimestampOracle>>,
) -> tokio::task::JoinHandle<()> {
    // An engine opened without an oracle stamps writes from a counter; the
    // record then takes a wall-clock stamp of its own.
    let oracle =
        oracle.unwrap_or_else(|| Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new()));
    let ids = coordinode_core::txn::proposal::ProposalIdGenerator::with_base(
        coordinode_core::txn::proposal::fresh_proposal_id_base(),
    );
    tokio::spawn(async move {
        use openraft::rt::watch::WatchReceiver;
        let mut metrics = raft.metrics();
        // The term this node last recorded its pair in, so a refused or
        // lost attempt is retried once per change, not in a loop.
        let mut recorded_in: Option<u64> = None;
        loop {
            let (leader, leading_term) = {
                let m = metrics.borrow_watched();
                let leader = m.current_leader.map(|id| {
                    let addr = m
                        .membership_config
                        .membership()
                        .get_node(&id)
                        .map(|n| n.addr.clone())
                        .unwrap_or_default();
                    (id, addr)
                });
                let leading = m.state.is_leader() && m.vote.is_committed();
                (leader, leading.then_some(m.vote.leader_id().term))
            };
            gate.set_leader(leader);
            if let Some(term) = leading_term {
                if recorded_in != Some(term) {
                    if let Some(pair) = gate.pair_to_record() {
                        let request = record_pair_request(pair, ids.next(), oracle.next());
                        match raft.client_write(request).await {
                            Ok(_) => {
                                tracing::info!(%pair, term, "recorded the group's version pair");
                                recorded_in = Some(term);
                            }
                            Err(e) => {
                                tracing::warn!(%e, %pair, "recording the group's version pair");
                            }
                        }
                    } else {
                        recorded_in = Some(term);
                    }
                }
            }
            if metrics.changed().await.is_err() {
                break;
            }
        }
    })
}

/// The entry recording `pair` as the group's.
fn record_pair_request(
    pair: coordinode_core::version::VersionPair,
    id: coordinode_core::txn::proposal::ProposalId,
    at: coordinode_core::txn::timestamp::Timestamp,
) -> crate::storage::Request {
    use coordinode_core::txn::proposal::{MetadataCommand, Mutation, RaftProposal};
    crate::storage::Request::single(RaftProposal {
        id,
        mutations: vec![Mutation::Command(MetadataCommand::RecordGroupPair { pair })],
        commit_ts: at,
        start_ts: at,
        bypass_rate_limiter: true,
    })
}

/// How long opening a node that is its group's only voter waits for it to
/// elect itself: one vote made durable, so seconds only on a stalled disk.
const SINGLE_NODE_ELECTION_WAIT: std::time::Duration = std::time::Duration::from_secs(10);

/// How long a shutdown waits for snapshot work still holding the engine: a
/// full build of a large store takes minutes, and cutting it short would
/// leave the directory locked for the caller that reopens it.
const SNAPSHOT_WORK_DRAIN: std::time::Duration = std::time::Duration::from_secs(300);

/// How long a leadership transfer waits for the target to lead, in ms. An
/// election the target was told to start takes one round to the voters; a
/// transfer still pending after this is failing (target down, partitioned,
/// or beaten), not slow.
pub const LEADERSHIP_TRANSFER_TIMEOUT_MS: u64 = 5_000;

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
pub struct RaftNode {
    /// The openraft instance.
    raft: Arc<RaftInstance>,
    /// Applied watermark subscriber (from state machine).
    applied_rx: tokio::sync::watch::Receiver<u64>,
    /// Snapshot-build counter (from state machine); see
    /// [`RaftNode::snapshot_builds`].
    snapshot_builds: Arc<core::sync::atomic::AtomicU64>,
    /// Snapshot work holding the engine, which `shutdown()` waits out.
    engine_work: crate::storage::EngineWork,
    /// This node's ID.
    node_id: u64,
    /// Address peers dial to reach this node. `None` for a standalone node and
    /// for a joining one, whose address the leader supplies when adding it.
    advertise_addr: Option<String>,
    /// Storage engine — held so `shutdown()` can flush before returning.
    engine: Arc<StorageEngine>,
    /// Shared handle to the Raft oplog, for reading committed entries since a
    /// checkpoint (WAL-replay repair). Shares the `LogStore`'s manager.
    oplog: Arc<std::sync::Mutex<coordinode_storage::oplog::OplogManager>>,
    /// Local-append notifier of the `LogStore`, for `w:1` / `w:N` in the
    /// proposal pipeline.
    append_notifier: Arc<AppendNotifier>,
    /// gRPC server shutdown signal. Taken by `shutdown()` (or dropped with
    /// the node) to stop the accept loop.
    grpc_shutdown: std::sync::Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
    /// The gRPC serve task. `shutdown()` gives its connection drain a short
    /// grace and then aborts it: tonic holds the listener until every open
    /// connection closes, and peers keep idle HTTP/2 channels open
    /// indefinitely — without the abort the port stays bound and a
    /// restarting node is locked out of its own address.
    grpc_task: std::sync::Mutex<Option<tokio::task::JoinHandle<()>>>,
    /// Set by `shutdown()` once the consensus stops, failing every call to a
    /// peer still in flight so the tasks waiting on them let go of the
    /// engine.
    closing: tokio::sync::watch::Sender<bool>,
    /// Snapshot trigger background task abort handle.
    _snapshot_trigger: Option<tokio::task::JoinHandle<()>>,
    /// This member's version view, shared with its network and handler.
    version: Arc<VersionGate>,
    /// Keeps the gate's leader current and records the group's pair.
    version_watch: tokio::task::JoinHandle<()>,
    /// How long a membership change waits for the previous one to settle,
    /// in ms. See [`Self::set_membership_settle_timeout`].
    membership_settle_timeout_ms: core::sync::atomic::AtomicU64,
    /// When a joining learner is promoted, and how long a join may take. See
    /// [`Self::set_join_readiness_lag`] and [`Self::set_join_timeout`].
    join: JoinTuning,
}

/// Default for [`RaftNode::join_readiness_lag`]: close enough that the
/// remaining gap replicates within a few heartbeats once the member votes.
const DEFAULT_JOIN_READINESS_LAG: u64 = 1_000;
/// Default for [`RaftNode::join_timeout`].
const DEFAULT_JOIN_TIMEOUT_MS: u64 = 30 * 60 * 1_000;

/// The join settings an operator may retune while the node runs.
struct JoinTuning {
    readiness_lag: core::sync::atomic::AtomicU64,
    timeout_ms: core::sync::atomic::AtomicU64,
}

impl Default for JoinTuning {
    fn default() -> Self {
        Self {
            readiness_lag: core::sync::atomic::AtomicU64::new(DEFAULT_JOIN_READINESS_LAG),
            timeout_ms: core::sync::atomic::AtomicU64::new(DEFAULT_JOIN_TIMEOUT_MS),
        }
    }
}

impl RaftNode {
    /// How many entries a joining learner may still lack when it is promoted
    /// to a voter.
    pub fn join_readiness_lag(&self) -> u64 {
        self.join
            .readiness_lag
            .load(core::sync::atomic::Ordering::Relaxed)
    }

    /// Change the promotion threshold without a restart; joins already
    /// waiting pick it up on their next poll.
    pub fn set_join_readiness_lag(&self, entries: u64) {
        self.join
            .readiness_lag
            .store(entries, core::sync::atomic::Ordering::Relaxed);
    }

    /// How long a join may take to catch its learner up before it fails.
    pub fn join_timeout(&self) -> std::time::Duration {
        std::time::Duration::from_millis(
            self.join
                .timeout_ms
                .load(core::sync::atomic::Ordering::Relaxed),
        )
    }

    /// Change that bound without a restart, for joins started afterwards.
    pub fn set_join_timeout(&self, timeout: std::time::Duration) {
        let ms = u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX);
        self.join
            .timeout_ms
            .store(ms, core::sync::atomic::Ordering::Relaxed);
    }
    /// How long a membership change waits for the previous one to commit.
    pub fn membership_settle_timeout(&self) -> std::time::Duration {
        std::time::Duration::from_millis(
            self.membership_settle_timeout_ms
                .load(core::sync::atomic::Ordering::Relaxed),
        )
    }

    /// Change that wait without a restart. A change still unsettled when it
    /// runs out is refused with an error naming the previous change, rather
    /// than waited on for as long as the group has no quorum to commit it.
    pub fn set_membership_settle_timeout(&self, timeout: std::time::Duration) {
        let ms = u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX);
        self.membership_settle_timeout_ms
            .store(ms, core::sync::atomic::Ordering::Relaxed);
    }

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

    /// Open a Raft node with timestamp oracle for seqno advancement.
    ///
    /// When oracle is provided, the state machine calls `oracle.advance_to(commit_ts)`
    /// before applying each entry's mutations. This ensures Raft replay produces
    /// identical seqnos as original application.
    pub async fn open_with_oracle(
        node_id: u64,
        engine: Arc<StorageEngine>,
        oracle: Option<Arc<coordinode_core::txn::timestamp::TimestampOracle>>,
    ) -> Result<Self, RaftNodeError> {
        Self::open_with_oracle_and_options(node_id, engine, oracle, NodeOptions::default()).await
    }

    /// Like `open_with_oracle`, with the snapshot policy and host epoch of
    /// `options`.
    pub async fn open_with_oracle_and_options(
        node_id: u64,
        engine: Arc<StorageEngine>,
        oracle: Option<Arc<coordinode_core::txn::timestamp::TimestampOracle>>,
        options: NodeOptions,
    ) -> Result<Self, RaftNodeError> {
        let NodeOptions {
            snapshots: snap_config,
            host_epoch,
        } = options;
        let config = Arc::new(snap_config.raft_config());
        let log_store =
            LogStore::open(Arc::clone(&engine)).map_err(|e| RaftNodeError::Init(e.to_string()))?;
        // Clone the oplog handle before openraft consumes the LogStore, so
        // WAL-replay repair can read committed entries since a checkpoint.
        let oplog = log_store.oplog_handle();
        let append_notifier = log_store.append_notifier();
        // Explicit oracle wins; otherwise fall back to the oracle the
        // engine itself stamps writes with (see the cluster constructors
        // for why the state machine must advance it).
        let oracle = oracle.or_else(|| engine.oracle());
        let state_machine =
            CoordinodeStateMachine::with_oracle(Arc::clone(&engine), oracle.clone())
                .map_err(|e| RaftNodeError::Init(e.to_string()))?
                .with_engine_work(log_store.engine_work());
        let version = version_gate(node_id, &state_machine, host_epoch);

        let applied_rx = state_machine.subscribe_applied();
        let snapshot_builds = state_machine.snapshot_builds_handle();
        let engine_work = state_machine.engine_work_handle();

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
                publish_existing_state_as_group_base(&raft, &engine).await?;
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

        // A node that is its group's only voter elects itself; it takes
        // writes once its vote is durable and the entry opening its term is
        // applied. Returned before that, it would refuse its first writes as
        // not the leader, so the election is waited for here.
        // The metrics catch up with the membership just initialized or
        // restored, so the wait starts once they name the voters: a group of
        // others is left to elect its own leader.
        raft.wait(Some(SINGLE_NODE_ELECTION_WAIT))
            .metrics(
                |m| {
                    let joint = m.membership_config.membership().get_joint_config();
                    let Some(voters) = joint.first().filter(|v| !v.is_empty()) else {
                        return false;
                    };
                    if voters.iter().any(|&id| id != node_id) {
                        return true;
                    }
                    m.state.is_leader()
                        && m.vote.is_committed()
                        && m.last_applied.map(|id| id.index) >= m.last_log_index
                },
                "the only voter elects itself",
            )
            .await
            .map_err(|e| RaftNodeError::Init(format!("the node did not become leader: {e}")))?;

        let raft = Arc::new(raft);
        let snap_handle = spawn_snapshot_trigger(
            Arc::clone(&raft),
            Arc::clone(&engine),
            &engine_work,
            snap_config,
            applied_rx.clone(),
        );
        let version_watch = spawn_version_watch(Arc::clone(&raft), Arc::clone(&version), oracle);

        Ok(Self {
            raft,
            applied_rx,
            snapshot_builds,
            engine_work,
            node_id,
            advertise_addr: None,
            engine,
            oplog,
            append_notifier,
            grpc_shutdown: std::sync::Mutex::new(None),
            grpc_task: std::sync::Mutex::new(None),
            // No peers to call.
            closing: tokio::sync::watch::Sender::new(false),
            _snapshot_trigger: Some(snap_handle),
            version,
            version_watch,
            membership_settle_timeout_ms: core::sync::atomic::AtomicU64::new(
                DEFAULT_MEMBERSHIP_SETTLE_TIMEOUT_MS,
            ),
            join: JoinTuning::default(),
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
        Self::open_cluster_with_options(
            node_id,
            engine,
            listen_addr,
            advertise_addr,
            NodeOptions::default(),
        )
        .await
    }

    /// Like `open_cluster`, with the snapshot policy and host epoch of
    /// `options`.
    pub async fn open_cluster_with_options(
        node_id: u64,
        engine: Arc<StorageEngine>,
        listen_addr: std::net::SocketAddr,
        advertise_addr: String,
        options: NodeOptions,
    ) -> Result<Self, RaftNodeError> {
        let NodeOptions {
            snapshots: snap_config,
            host_epoch,
        } = options;
        let config = Arc::new(snap_config.raft_config());
        let log_store =
            LogStore::open(Arc::clone(&engine)).map_err(|e| RaftNodeError::Init(e.to_string()))?;
        // Clone the oplog handle before openraft consumes the LogStore, so
        // WAL-replay repair can read committed entries since a checkpoint.
        let oplog = log_store.oplog_handle();
        let append_notifier = log_store.append_notifier();
        // Wire the engine's own timestamp oracle into the state machine:
        // applied entries carry the leader's commit timestamps, and the
        // local oracle must advance past them or MVCC readers on this
        // node keep drawing stale snapshots that observe none of the
        // replicated data.
        let state_machine =
            CoordinodeStateMachine::with_oracle(Arc::clone(&engine), engine.oracle())
                .map_err(|e| RaftNodeError::Init(e.to_string()))?
                .with_engine_work(log_store.engine_work());
        let version = version_gate(node_id, &state_machine, host_epoch);
        let applied_rx = state_machine.subscribe_applied();
        let snapshot_builds = state_machine.snapshot_builds_handle();
        let engine_work = state_machine.engine_work_handle();

        let (closing, closing_rx) = tokio::sync::watch::channel(false);
        let network = GrpcNetworkFactory {
            local_node_id: node_id,
            closing: network::Closing::new(closing_rx),
            gate: Arc::clone(&version),
        };

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
            addr: advertise_addr.clone(),
        };
        members.insert(node_id, node_info);

        match raft.initialize(members).await {
            Ok(_) => {
                tracing::info!(node_id, "fresh cluster node initialized as leader");
                publish_existing_state_as_group_base(&raft, &engine).await?;
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

        // Start gRPC server for inter-node Raft RPCs with shutdown signal.
        // The listener is bound EAGERLY: a busy port must fail this open
        // instead of producing a node that reports success while its server
        // task dies unheard — deaf to the cluster, green on every local
        // check. Bound through tokio rather than tonic's TcpIncoming::bind
        // because tokio sets SO_REUSEADDR: a node restarting on its own
        // address must not be locked out by its previous life's TIME_WAIT
        // sockets. Nodelay mirrors tonic's serve(addr) default.
        let listener = tokio::net::TcpListener::bind(listen_addr)
            .await
            .map_err(|e| RaftNodeError::Init(format!("bind {listen_addr}: {e}")))?;
        let incoming =
            tonic::transport::server::TcpIncoming::from(listener).with_nodelay(Some(true));
        let handler = RaftGrpcHandler::new(
            Arc::clone(&raft),
            crate::snapshot::snapshot_dir(&engine),
            Arc::clone(&version),
        );
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

        let server = tonic::transport::Server::builder()
            .add_service(RaftServiceServer::new(handler))
            .add_service(handshake_server(&version));

        let grpc_task = tokio::spawn(async move {
            let graceful = server.serve_with_incoming_shutdown(incoming, async {
                let _ = shutdown_rx.await;
            });
            if let Err(e) = graceful.await {
                tracing::error!(%e, "raft gRPC server failed");
            }
        });

        tracing::info!(node_id, %listen_addr, "raft gRPC server started");

        // Start background snapshot trigger (WAL size + periodic timer)
        let snap_handle = spawn_snapshot_trigger(
            Arc::clone(&raft),
            Arc::clone(&engine),
            &engine_work,
            snap_config,
            applied_rx.clone(),
        );
        let version_watch =
            spawn_version_watch(Arc::clone(&raft), Arc::clone(&version), engine.oracle());

        Ok(Self {
            raft,
            applied_rx,
            snapshot_builds,
            engine_work,
            node_id,
            advertise_addr: Some(advertise_addr),
            engine,
            oplog,
            append_notifier,
            grpc_shutdown: std::sync::Mutex::new(Some(shutdown_tx)),
            grpc_task: std::sync::Mutex::new(Some(grpc_task)),
            closing,
            _snapshot_trigger: Some(snap_handle),
            version,
            version_watch,
            membership_settle_timeout_ms: core::sync::atomic::AtomicU64::new(
                DEFAULT_MEMBERSHIP_SETTLE_TIMEOUT_MS,
            ),
            join: JoinTuning::default(),
        })
    }

    /// Open a cluster-mode bootstrap node for embedding into an existing tonic server.
    ///
    /// Unlike [`Self::open_cluster`], this constructor does **not** start a dedicated
    /// internal gRPC server. Instead it returns the [`RaftGrpcHandler`] for the
    /// caller to register into the main tonic router on `:7080`.
    ///
    /// This is the correct path for `coordinode --mode=full` where `:7080`
    /// serves both client-facing gRPC (CypherService, GraphService, …) and
    /// inter-node Raft RPCs on the same port and the same tonic server instance.
    ///
    /// ## Usage
    ///
    /// ```rust,ignore
    /// let (raft_node, raft_handler) = RaftNode::open_cluster_embedded(
    ///     node_id, engine, advertise_addr,
    /// ).await?;
    ///
    /// // Register Raft RPC handler into the main tonic router:
    /// Server::builder()
    ///     .add_service(CypherServiceServer::new(cypher_svc))
    ///     .add_service(RaftServiceServer::new(raft_handler))
    ///     .serve_with_shutdown(addr, shutdown)
    ///     .await?;
    /// ```
    pub async fn open_cluster_embedded(
        node_id: u64,
        engine: Arc<StorageEngine>,
        advertise_addr: String,
    ) -> Result<(Self, RaftGrpcHandler), RaftNodeError> {
        Self::open_cluster_embedded_with_options(
            node_id,
            engine,
            advertise_addr,
            NodeOptions::default(),
        )
        .await
    }

    /// Like `open_cluster_embedded`, with the snapshot policy and host epoch
    /// of `options`.
    pub async fn open_cluster_embedded_with_options(
        node_id: u64,
        engine: Arc<StorageEngine>,
        advertise_addr: String,
        options: NodeOptions,
    ) -> Result<(Self, RaftGrpcHandler), RaftNodeError> {
        let NodeOptions {
            snapshots: snap_config,
            host_epoch,
        } = options;
        let config = Arc::new(snap_config.raft_config());
        let log_store =
            LogStore::open(Arc::clone(&engine)).map_err(|e| RaftNodeError::Init(e.to_string()))?;
        // Clone the oplog handle before openraft consumes the LogStore, so
        // WAL-replay repair can read committed entries since a checkpoint.
        let oplog = log_store.oplog_handle();
        let append_notifier = log_store.append_notifier();
        // Wire the engine's own timestamp oracle into the state machine:
        // applied entries carry the leader's commit timestamps, and the
        // local oracle must advance past them or MVCC readers on this
        // node keep drawing stale snapshots that observe none of the
        // replicated data.
        let state_machine =
            CoordinodeStateMachine::with_oracle(Arc::clone(&engine), engine.oracle())
                .map_err(|e| RaftNodeError::Init(e.to_string()))?
                .with_engine_work(log_store.engine_work());
        let version = version_gate(node_id, &state_machine, host_epoch);
        let applied_rx = state_machine.subscribe_applied();
        let snapshot_builds = state_machine.snapshot_builds_handle();
        let engine_work = state_machine.engine_work_handle();

        let (closing, closing_rx) = tokio::sync::watch::channel(false);
        let network = GrpcNetworkFactory {
            local_node_id: node_id,
            closing: network::Closing::new(closing_rx),
            gate: Arc::clone(&version),
        };

        let raft: RaftInstance =
            openraft::Raft::new(node_id, config, network, log_store, state_machine)
                .await
                .map_err(|e: openraft::error::Fatal<TypeConfig>| {
                    RaftNodeError::Init(e.to_string())
                })?;

        let raft = Arc::new(raft);

        // Try initialize — succeeds on fresh start, NotAllowed on restart.
        let mut members = std::collections::BTreeMap::new();
        members.insert(
            node_id,
            openraft::impls::BasicNode {
                addr: advertise_addr.clone(),
            },
        );

        match raft.initialize(members).await {
            Ok(_) => {
                tracing::info!(node_id, "fresh cluster node initialized as leader");
                publish_existing_state_as_group_base(&raft, &engine).await?;
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

        // Build the gRPC handler — caller registers it into the main tonic router.
        // No internal gRPC server is started here.
        let handler = RaftGrpcHandler::new(
            Arc::clone(&raft),
            crate::snapshot::snapshot_dir(&engine),
            Arc::clone(&version),
        );

        let snap_handle = spawn_snapshot_trigger(
            Arc::clone(&raft),
            Arc::clone(&engine),
            &engine_work,
            snap_config,
            applied_rx.clone(),
        );
        let version_watch =
            spawn_version_watch(Arc::clone(&raft), Arc::clone(&version), engine.oracle());

        let node = Self {
            raft,
            applied_rx,
            snapshot_builds,
            engine_work,
            node_id,
            advertise_addr: Some(advertise_addr),
            engine,
            oplog,
            append_notifier,
            // no internal server — caller manages the router
            grpc_shutdown: std::sync::Mutex::new(None),
            grpc_task: std::sync::Mutex::new(None),
            closing,
            _snapshot_trigger: Some(snap_handle),
            version,
            version_watch,
            membership_settle_timeout_ms: core::sync::atomic::AtomicU64::new(
                DEFAULT_MEMBERSHIP_SETTLE_TIMEOUT_MS,
            ),
            join: JoinTuning::default(),
        };

        Ok((node, handler))
    }

    /// Open a joining node for embedding into an existing tonic server.
    ///
    /// Like `open_cluster_embedded` but does **not** call `initialize()` — the
    /// node waits for the cluster leader to add it via `add_learner`.
    ///
    /// Use this for nodes 2+ in a multi-node cluster when `--mode=full` is active
    /// (i.e., the Raft handler is embedded into the main tonic router at `:7080`
    /// rather than running in a separate internal gRPC server).
    pub async fn open_joining_embedded(
        node_id: u64,
        engine: Arc<StorageEngine>,
    ) -> Result<(Self, RaftGrpcHandler), RaftNodeError> {
        Self::open_joining_embedded_with_options(node_id, engine, NodeOptions::default()).await
    }

    /// Like `open_joining_embedded`, with the snapshot policy and host epoch
    /// of `options`.
    pub async fn open_joining_embedded_with_options(
        node_id: u64,
        engine: Arc<StorageEngine>,
        options: NodeOptions,
    ) -> Result<(Self, RaftGrpcHandler), RaftNodeError> {
        let NodeOptions {
            snapshots: snap_config,
            host_epoch,
        } = options;
        let config = Arc::new(snap_config.raft_config());
        let log_store =
            LogStore::open(Arc::clone(&engine)).map_err(|e| RaftNodeError::Init(e.to_string()))?;
        // Clone the oplog handle before openraft consumes the LogStore, so
        // WAL-replay repair can read committed entries since a checkpoint.
        let oplog = log_store.oplog_handle();
        let append_notifier = log_store.append_notifier();
        // Wire the engine's own timestamp oracle into the state machine:
        // applied entries carry the leader's commit timestamps, and the
        // local oracle must advance past them or MVCC readers on this
        // node keep drawing stale snapshots that observe none of the
        // replicated data.
        let state_machine =
            CoordinodeStateMachine::with_oracle(Arc::clone(&engine), engine.oracle())
                .map_err(|e| RaftNodeError::Init(e.to_string()))?
                .with_engine_work(log_store.engine_work());
        refuse_join_with_local_data(&engine, &log_store, &state_machine)?;
        let version = version_gate(node_id, &state_machine, host_epoch);
        let applied_rx = state_machine.subscribe_applied();
        let snapshot_builds = state_machine.snapshot_builds_handle();
        let engine_work = state_machine.engine_work_handle();

        let (closing, closing_rx) = tokio::sync::watch::channel(false);
        let network = GrpcNetworkFactory {
            local_node_id: node_id,
            closing: network::Closing::new(closing_rx),
            gate: Arc::clone(&version),
        };

        let raft: RaftInstance =
            openraft::Raft::new(node_id, config, network, log_store, state_machine)
                .await
                .map_err(|e: openraft::error::Fatal<TypeConfig>| {
                    RaftNodeError::Init(e.to_string())
                })?;

        let raft = Arc::new(raft);

        // Build the gRPC handler — caller registers it into the main tonic router.
        let handler = RaftGrpcHandler::new(
            Arc::clone(&raft),
            crate::snapshot::snapshot_dir(&engine),
            Arc::clone(&version),
        );

        let snap_handle = spawn_snapshot_trigger(
            Arc::clone(&raft),
            Arc::clone(&engine),
            &engine_work,
            snap_config,
            applied_rx.clone(),
        );
        let version_watch =
            spawn_version_watch(Arc::clone(&raft), Arc::clone(&version), engine.oracle());

        tracing::info!(
            node_id,
            "joining node started (waiting for leader to add via add_node)"
        );

        let node = Self {
            raft,
            applied_rx,
            snapshot_builds,
            engine_work,
            node_id,
            advertise_addr: None,
            engine,
            oplog,
            append_notifier,
            // no internal server — caller manages the router
            grpc_shutdown: std::sync::Mutex::new(None),
            grpc_task: std::sync::Mutex::new(None),
            closing,
            _snapshot_trigger: Some(snap_handle),
            version,
            version_watch,
            membership_settle_timeout_ms: core::sync::atomic::AtomicU64::new(
                DEFAULT_MEMBERSHIP_SETTLE_TIMEOUT_MS,
            ),
            join: JoinTuning::default(),
        };

        Ok((node, handler))
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
        Self::open_joining_with_options(node_id, engine, listen_addr, NodeOptions::default()).await
    }

    /// Like `open_joining`, with the snapshot policy and host epoch of
    /// `options`.
    pub async fn open_joining_with_options(
        node_id: u64,
        engine: Arc<StorageEngine>,
        listen_addr: std::net::SocketAddr,
        options: NodeOptions,
    ) -> Result<Self, RaftNodeError> {
        let NodeOptions {
            snapshots: snap_config,
            host_epoch,
        } = options;
        let config = Arc::new(snap_config.raft_config());
        let log_store =
            LogStore::open(Arc::clone(&engine)).map_err(|e| RaftNodeError::Init(e.to_string()))?;
        // Clone the oplog handle before openraft consumes the LogStore, so
        // WAL-replay repair can read committed entries since a checkpoint.
        let oplog = log_store.oplog_handle();
        let append_notifier = log_store.append_notifier();
        // Wire the engine's own timestamp oracle into the state machine:
        // applied entries carry the leader's commit timestamps, and the
        // local oracle must advance past them or MVCC readers on this
        // node keep drawing stale snapshots that observe none of the
        // replicated data.
        let state_machine =
            CoordinodeStateMachine::with_oracle(Arc::clone(&engine), engine.oracle())
                .map_err(|e| RaftNodeError::Init(e.to_string()))?
                .with_engine_work(log_store.engine_work());
        refuse_join_with_local_data(&engine, &log_store, &state_machine)?;
        let version = version_gate(node_id, &state_machine, host_epoch);
        let applied_rx = state_machine.subscribe_applied();
        let snapshot_builds = state_machine.snapshot_builds_handle();
        let engine_work = state_machine.engine_work_handle();

        let (closing, closing_rx) = tokio::sync::watch::channel(false);
        let network = GrpcNetworkFactory {
            local_node_id: node_id,
            closing: network::Closing::new(closing_rx),
            gate: Arc::clone(&version),
        };

        let raft: RaftInstance =
            openraft::Raft::new(node_id, config, network, log_store, state_machine)
                .await
                .map_err(|e: openraft::error::Fatal<TypeConfig>| {
                    RaftNodeError::Init(e.to_string())
                })?;

        let raft = Arc::new(raft);

        // Bound eagerly for the same reason as in open_cluster: a busy port
        // is a hard open error, never a deaf node. tokio's bind sets
        // SO_REUSEADDR so a restart on the same address is not locked out
        // by the previous life's TIME_WAIT sockets.
        let listener = tokio::net::TcpListener::bind(listen_addr)
            .await
            .map_err(|e| RaftNodeError::Init(format!("bind {listen_addr}: {e}")))?;
        let incoming =
            tonic::transport::server::TcpIncoming::from(listener).with_nodelay(Some(true));
        let handler = RaftGrpcHandler::new(
            Arc::clone(&raft),
            crate::snapshot::snapshot_dir(&engine),
            Arc::clone(&version),
        );
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

        let server = tonic::transport::Server::builder()
            .add_service(RaftServiceServer::new(handler))
            .add_service(handshake_server(&version));

        let grpc_task = tokio::spawn(async move {
            let graceful = server.serve_with_incoming_shutdown(incoming, async {
                let _ = shutdown_rx.await;
            });
            if let Err(e) = graceful.await {
                tracing::error!(%e, "raft gRPC server failed");
            }
        });

        tracing::info!(node_id, %listen_addr, "joining node started (waiting for leader)");

        let snap_handle = spawn_snapshot_trigger(
            Arc::clone(&raft),
            Arc::clone(&engine),
            &engine_work,
            snap_config,
            applied_rx.clone(),
        );
        let version_watch =
            spawn_version_watch(Arc::clone(&raft), Arc::clone(&version), engine.oracle());

        Ok(Self {
            raft,
            applied_rx,
            snapshot_builds,
            engine_work,
            node_id,
            advertise_addr: None,
            engine,
            oplog,
            append_notifier,
            grpc_shutdown: std::sync::Mutex::new(Some(shutdown_tx)),
            grpc_task: std::sync::Mutex::new(Some(grpc_task)),
            closing,
            _snapshot_trigger: Some(snap_handle),
            version,
            version_watch,
            membership_settle_timeout_ms: core::sync::atomic::AtomicU64::new(
                DEFAULT_MEMBERSHIP_SETTLE_TIMEOUT_MS,
            ),
            join: JoinTuning::default(),
        })
    }

    /// Say why openraft refused a membership change in the terms an operator
    /// acts on. Its own wording for a missing leader is
    /// "has to forward request to: None, None", which names neither the cause
    /// (no quorum to elect one) nor the fact that nothing changed.
    fn describe_membership_refusal(
        e: &openraft::error::RaftError<TypeConfig, openraft::error::ClientWriteError<TypeConfig>>,
    ) -> String {
        match e.forward_to_leader().map(|f| f.leader_id) {
            Some(Some(leader)) => {
                format!("node {leader} is the leader; send the change there. Nothing was changed.")
            }
            Some(None) => "no leader is known: without a quorum the cluster cannot elect one. \
                           Nothing was changed."
                .to_string(),
            None => e.to_string(),
        }
    }

    /// Wait until no membership change is in flight: the membership in effect
    /// is a single configuration, not a joint one, and the log entry that
    /// installed it is committed.
    ///
    /// The membership in effect is visible before it commits, so status can
    /// already show a new voter while the change that added it is still being
    /// replicated, and openraft refuses a second change until the first one
    /// settles. An operator acts on the status; every membership change here
    /// starts by waiting for the previous one rather than failing with it.
    async fn membership_settled(&self) -> Result<(), RaftNodeError> {
        use openraft::LogIdOptionExt;

        self.raft
            .wait(Some(self.membership_settle_timeout()))
            .metrics(
                |m| {
                    m.membership_config.membership().get_joint_config().len() == 1
                        && m.membership_config.log_id().index() <= m.local_committed.index()
                },
                "the previous membership change settles",
            )
            .await
            .map(|_| ())
            .map_err(|e| {
                RaftNodeError::Membership(format!(
                    "a previous membership change is still in progress: {e}"
                ))
            })
    }

    /// Wait until the metrics show the membership written at log `index`.
    ///
    /// openraft answers a membership change before it publishes the new
    /// membership in its metrics. A caller reading the members at once would
    /// see the old ones, and the next change, which starts from the members in
    /// view, would be computed from a set that no longer holds.
    async fn membership_published(&self, index: u64) -> Result<(), RaftNodeError> {
        use openraft::LogIdOptionExt;

        self.raft
            .wait(Some(self.membership_settle_timeout()))
            .metrics(
                |m| m.membership_config.log_id().index() >= Some(index),
                "the membership change shows in the metrics",
            )
            .await
            .map(|_| ())
            .map_err(|e| {
                RaftNodeError::Membership(format!(
                    "the membership change at log {index} did not show: {e}"
                ))
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
        self.publish_own_address().await?;
        self.membership_settled().await?;

        let node_info = openraft::impls::BasicNode { addr };

        // Add as learner (non-voting, receives log replication)
        let written = self
            .raft
            .add_learner(node_id, node_info, true)
            .await
            .map_err(|e| RaftNodeError::Membership(e.to_string()))?;
        self.membership_published(written.log_id.index).await?;

        tracing::info!(node_id, "added node as learner");
        Ok(())
    }

    /// Record this node's advertised address in the membership while it is
    /// still the cluster's only member.
    ///
    /// A directory first opened standalone holds a placeholder address for its
    /// single member, and reopening it in cluster mode resumes that state rather
    /// than initialising again. Peers dial members by this address, so it has to
    /// be right before the first peer joins. Replacing an address is safe only
    /// while no other member exists: with peers present a wrong address can split
    /// the cluster, and the node has to be removed and added back instead.
    async fn publish_own_address(&self) -> Result<(), RaftNodeError> {
        use openraft::rt::watch::WatchReceiver;

        let Some(advertise) = self.advertise_addr.as_ref() else {
            return Ok(());
        };
        let stale = {
            let rx = self.raft.metrics();
            let metrics = rx.borrow_watched();
            let mut members = metrics.membership_config.membership().nodes();
            match (members.next(), members.next()) {
                (Some((id, info)), None) => *id == self.node_id && info.addr != *advertise,
                _ => false,
            }
        };
        if !stale {
            return Ok(());
        }

        let mut nodes = std::collections::BTreeMap::new();
        nodes.insert(
            self.node_id,
            openraft::impls::BasicNode {
                addr: advertise.clone(),
            },
        );
        let written = self
            .raft
            .change_membership(openraft::ChangeMembers::SetNodes(nodes), true)
            .await
            .map_err(|e| RaftNodeError::Membership(e.to_string()))?;
        self.membership_published(written.log_id.index).await?;

        tracing::info!(node_id = self.node_id, addr = %advertise, "published own address");
        Ok(())
    }

    /// Change cluster membership (leader-only).
    ///
    /// Sets the voting members to the given set of node IDs.
    /// All nodes must have been previously added as learners.
    pub async fn change_membership(&self, member_ids: Vec<u64>) -> Result<(), RaftNodeError> {
        self.membership_settled().await?;
        let members: std::collections::BTreeSet<u64> = member_ids.into_iter().collect();

        let written = self
            .raft
            .change_membership(members, false)
            .await
            .map_err(|e| RaftNodeError::Membership(Self::describe_membership_refusal(&e)))?;
        self.membership_published(written.log_id.index).await?;

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

        self.membership_settled().await?;
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

        let written = self
            .raft
            .change_membership(new_members, false)
            .await
            .map_err(|e| RaftNodeError::Membership(Self::describe_membership_refusal(&e)))?;
        self.membership_published(written.log_id.index).await?;

        tracing::info!(node_id, "removed node from cluster");
        Ok(())
    }

    /// Promote an existing learner to a voter at runtime (leader-only).
    ///
    /// The node must already be a learner (added via [`add_node`](Self::add_node)
    /// and caught up on replication). Adds it to the voter set via openraft
    /// `change_membership`; the change commits cluster-wide and takes effect live
    /// with no process restart. Idempotent: a no-op (returns `Ok`) when the node
    /// is already a voter.
    ///
    /// # Errors
    /// - `Membership(…)` if this node is not the leader, the target is not a
    ///   known learner, or openraft rejects the membership change.
    pub async fn promote_to_voter(&self, node_id: u64) -> Result<(), RaftNodeError> {
        use openraft::async_runtime::watch::WatchReceiver;

        self.membership_settled().await?;
        let current_voters: std::collections::BTreeSet<u64> = {
            let metrics = self.raft.metrics().borrow_watched().clone();
            let joint = metrics.membership_config.membership().get_joint_config();
            joint
                .first()
                .map(|voters| voters.iter().copied().collect())
                .unwrap_or_default()
        };

        // Idempotent: already a voter → nothing to do.
        if current_voters.contains(&node_id) {
            return Ok(());
        }

        let mut new_members = current_voters;
        new_members.insert(node_id);

        let written = self
            .raft
            .change_membership(new_members, false)
            .await
            .map_err(|e| {
                RaftNodeError::Membership(format!(
                    "promote_to_voter failed: {}",
                    Self::describe_membership_refusal(&e)
                ))
            })?;
        self.membership_published(written.log_id.index).await?;

        tracing::info!(node_id, "promoted learner to voter");
        Ok(())
    }

    /// Demote a voter back to a learner at runtime (leader-only). The node
    /// keeps receiving log replication but
    /// stops voting — distinct from [`decommission_node`](Self::decommission_node)
    /// / [`remove_node`](Self::remove_node), which drop the node from the cluster
    /// entirely.
    ///
    /// Removes the node from the voter set with openraft `change_membership(…,
    /// retain = true)`, so the demoted node is retained as a learner. The change
    /// commits cluster-wide and takes effect live with no restart. Idempotent: a
    /// no-op when the node is not a voter.
    ///
    /// # Errors
    /// - `Membership("would leave N voter(s) …")` if the demotion would drop the
    ///   voter set below the CE minimum of 2.
    /// - `Membership("node … is the current leader …")` if the target is the
    ///   leader (transfer leadership first).
    /// - `Membership(…)` if this node is not the leader or openraft rejects it.
    pub async fn demote_to_learner(&self, node_id: u64) -> Result<(), RaftNodeError> {
        use openraft::async_runtime::watch::WatchReceiver;

        self.membership_settled().await?;
        let metrics = self.raft.metrics().borrow_watched().clone();
        let current_voters: std::collections::BTreeSet<u64> = {
            let joint = metrics.membership_config.membership().get_joint_config();
            joint
                .first()
                .map(|voters| voters.iter().copied().collect())
                .unwrap_or_default()
        };

        // Idempotent: not a voter → already a learner (or unknown), nothing to do.
        if !current_voters.contains(&node_id) {
            return Ok(());
        }

        // Quorum gate — CE requires ≥ 2 voters after demotion.
        let remaining = current_voters.len().saturating_sub(1);
        if remaining < 2 {
            return Err(RaftNodeError::Membership(format!(
                "cannot demote node {node_id}: would leave {remaining} voter(s), \
                 minimum 2 required for CE cluster quorum"
            )));
        }

        // The leader cannot demote itself out of the voter set in place — a
        // different node must be leader first (mirror of decommission_node).
        if metrics.current_leader == Some(node_id) {
            return Err(RaftNodeError::Membership(format!(
                "node {node_id} is the current Raft leader — transfer leadership before demoting"
            )));
        }

        let new_members: std::collections::BTreeSet<u64> = current_voters
            .into_iter()
            .filter(|&id| id != node_id)
            .collect();

        // retain = true: the removed voter stays as a learner (keeps replicating).
        let written = self
            .raft
            .change_membership(new_members, true)
            .await
            .map_err(|e| {
                RaftNodeError::Membership(format!(
                    "demote_to_learner failed: {}",
                    Self::describe_membership_refusal(&e)
                ))
            })?;
        self.membership_published(written.log_id.index).await?;

        tracing::info!(node_id, "demoted voter to learner");
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
        RaftProposalPipeline::with_append_notifier(
            Arc::clone(&self.raft),
            Arc::clone(&self.append_notifier),
        )
        .with_version_gate(Arc::clone(&self.version))
    }

    /// This member's version view: its pair, its group's, and whether it
    /// serves writes.
    pub fn version(&self) -> &Arc<VersionGate> {
        &self.version
    }

    /// This member's version report: its pair, its group's, whether it
    /// serves writes and why not, each voter's pair as far as it knows, and
    /// how long the group has been unable to write while it is.
    pub fn version_report(&self) -> version::VersionReport {
        use openraft::rt::watch::WatchReceiver;
        let voters: Vec<u64> = self
            .raft
            .metrics()
            .borrow_watched()
            .membership_config
            .membership()
            .voter_ids()
            .collect();
        self.version.report(&voters)
    }

    /// The frozen version exchange, for a caller that serves this node's
    /// [`RaftGrpcHandler`] on its own router: register both.
    pub fn handshake_service(
        &self,
    ) -> crate::proto::internode::version_handshake_server::VersionHandshakeServer<HandshakeService>
    {
        handshake_server(&self.version)
    }

    /// The log store's local-append notifier, so a pipeline built elsewhere
    /// (the server) can wait for `w:1` / `w:N` instead of falling back to a
    /// majority.
    pub fn append_notifier(&self) -> &Arc<AppendNotifier> {
        &self.append_notifier
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

    /// Rebuild `partition` from the local checkpoint at `checkpoint_dir` and
    /// this node's Raft log, with the applies paused: the repair path when
    /// no healthy replica serves the partition. See
    /// [`crate::storage::rebuild_partition_from_checkpoint`].
    ///
    /// Blocks; call it off the async runtime.
    ///
    /// # Errors
    ///
    /// [`RaftNodeError::Init`] with the rebuild's reason.
    pub fn rebuild_partition_from_checkpoint(
        &self,
        checkpoint_dir: &std::path::Path,
        partition: coordinode_storage::engine::partition::Partition,
    ) -> Result<(), RaftNodeError> {
        crate::storage::rebuild_partition_from_checkpoint(
            &self.engine,
            &self.oplog,
            checkpoint_dir,
            partition,
        )
        .map_err(|e| RaftNodeError::Init(e.to_string()))
    }

    /// Number of full snapshot builds this node has performed. Every
    /// build serializes ALL partitions, so an unexpectedly growing
    /// count (without new applied entries) indicates a misfiring
    /// trigger and a leader-stability hazard.
    pub fn snapshot_builds(&self) -> u64 {
        self.snapshot_builds
            .load(core::sync::atomic::Ordering::Relaxed)
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

    /// Check if this node is currently the Raft leader, as its group confirms.
    ///
    /// Leadership the group does not confirm within two election timeouts is
    /// not leadership: by then its followers have called an election. The bound
    /// is ours because openraft's own wait for the read point to apply has none,
    /// and a leader elected just before its peers left waits forever for the
    /// first entry of its term to commit.
    pub async fn is_leader(&self) -> bool {
        // Saturating on purpose: an election timeout too large to double is
        // already a wait as long as the group takes, and the bound stays that.
        let bound = std::time::Duration::from_millis(
            self.raft.config().election_timeout_max.saturating_mul(2),
        );
        matches!(
            tokio::time::timeout(
                bound,
                self.raft
                    .ensure_linearizable(openraft::raft::ReadPolicy::LeaseRead),
            )
            .await,
            Ok(Ok(_))
        )
    }

    /// Whether this node believes it leads, from its own state alone.
    ///
    /// Asks the group nothing, so it answers at once whatever the group's
    /// state: what a node needs when deciding its own shutdown, where waiting
    /// on peers that are gone would keep it from stopping.
    fn leads_by_its_own_account(&self) -> bool {
        use openraft::async_runtime::watch::WatchReceiver;
        let metrics = self.raft.metrics().borrow_watched().clone();
        metrics.state.is_leader() && metrics.current_leader == Some(self.node_id)
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

    /// One past the last log entry this node has applied, `0` when none.
    ///
    /// The exclusive bound for a reader of the Raft log that must see only
    /// applied entries: the log also holds entries that are not committed
    /// yet, which a later leader may truncate and replace.
    pub fn applied_through(&self) -> u64 {
        use openraft::async_runtime::watch::WatchReceiver;

        // The state machine publishes each index as it applies it, before the
        // proposer hears back; openraft's metrics follow later. Only `0` is
        // ambiguous there (nothing applied, or entry 0), and the metrics tell.
        let applied = *self.applied_rx.borrow();
        if applied > 0 {
            return applied + 1;
        }
        self.raft
            .metrics()
            .borrow_watched()
            .last_applied
            .as_ref()
            .map_or(0, |lid| lid.index + 1)
    }

    /// Get the current Raft term.
    pub fn current_term(&self) -> u64 {
        use openraft::async_runtime::watch::WatchReceiver;
        self.raft.metrics().borrow_watched().current_term
    }

    /// The current voter node IDs from the live Raft membership config. A node
    /// that has been demoted to a learner ([`demote_to_learner`](Self::demote_to_learner))
    /// is absent here even though it still receives replication.
    pub fn voter_ids(&self) -> Vec<u64> {
        use openraft::async_runtime::watch::WatchReceiver;
        let rx = self.raft.metrics();
        let m = rx.borrow_watched();
        m.membership_config.membership().voter_ids().collect()
    }

    /// Wait until who leads, or under which term, differs from what the caller
    /// last saw; return the new pair.
    ///
    /// The transition a subscriber usually waits for is `None` becoming a
    /// node: a client attached to a node that had lost touch with the cluster
    /// can be told the moment it can write again, instead of retrying on a
    /// timer whose interval is a guess about how long an election takes.
    /// Returns immediately when the current pair already differs.
    ///
    /// Keeps the consensus library's watch type inside this crate: callers
    /// deal in the pair, not in openraft's metrics.
    pub async fn next_leadership_change(&self, seen: (Option<u64>, u64)) -> (Option<u64>, u64) {
        use openraft::async_runtime::watch::WatchReceiver;
        let mut rx = self.raft.metrics();
        loop {
            let current = {
                let m = rx.borrow_watched();
                (m.current_leader, m.current_term)
            };
            if current != seen {
                return current;
            }
            if rx.changed().await.is_err() {
                // The node is shutting down and will report nothing further.
                return current;
            }
        }
    }

    /// Get the current leader node ID, if known.
    ///
    /// Returns `None` if the cluster has no leader (e.g. election in progress).
    pub fn current_leader(&self) -> Option<u64> {
        use openraft::async_runtime::watch::WatchReceiver;
        self.raft.metrics().borrow_watched().current_leader
    }

    /// The advertised address of a cluster member, as recorded in membership.
    ///
    /// This is where another node reaches it: the address the member published
    /// when it joined, not the address it happens to be listening on locally.
    /// `None` for an id the current membership does not contain, which is the
    /// honest answer for a member that has since been removed.
    pub fn node_addr(&self, node_id: u64) -> Option<String> {
        use openraft::async_runtime::watch::WatchReceiver;
        let metrics = self.raft.metrics().borrow_watched().clone();
        metrics
            .membership_config
            .membership()
            .get_node(&node_id)
            .map(|n| n.addr.clone())
            .filter(|addr| !addr.is_empty())
    }

    /// Subscribe to applied index updates.
    ///
    /// Returns a clone of the applied watermark receiver. The receiver
    /// delivers the latest applied log index whenever the state machine
    /// advances. Used by [`ReadFence`](crate::read_fence::ReadFence) for causal
    /// read fencing.
    pub fn subscribe_applied(&self) -> tokio::sync::watch::Receiver<u64> {
        self.applied_rx.clone()
    }

    /// Create a per-request read fence for enforcing read preference and concern.
    ///
    /// The returned [`ReadFence`](crate::read_fence::ReadFence) is cheap to
    /// create: it clones a watch receiver (pointer copy) and an Arc. Call
    /// [`ReadFence::apply()`](crate::read_fence::ReadFence::apply) before
    /// executing a query to enforce routing and consistency guarantees.
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

    /// Wait until this node's consensus stops on a fatal error (a log or
    /// state-machine storage failure, a panic of the core) and return it:
    /// from then on the node commits nothing. `None` when the core stops
    /// without one, as on shutdown.
    pub async fn consensus_stopped(&self) -> Option<String> {
        use openraft::async_runtime::watch::WatchReceiver;
        let mut metrics = self.raft.metrics();
        loop {
            match &metrics.borrow_watched().running_state {
                Ok(()) => {}
                // A shutdown is reported as `Stopped`: a normal end.
                Err(openraft::error::Fatal::Stopped) => return None,
                Err(fatal) => return Some(fatal.to_string()),
            }
            if metrics.changed().await.is_err() {
                return None;
            }
        }
    }

    /// Transfer leadership to a specific peer node.
    ///
    /// Sends a `TimeoutNow` message to the target, triggering an immediate
    /// election without waiting for `election_timeout` (300-600ms).
    ///
    /// Returns `Ok(())` once `target_id` leads and has committed in its term
    /// (so it serves reads and writes), and [`RaftNodeError::TransferTimeout`]
    /// when that has not happened within [`LEADERSHIP_TRANSFER_TIMEOUT_MS`].
    ///
    /// No-op if this node is not the leader.
    pub async fn transfer_leadership_to(&self, target_id: u64) -> Result<(), RaftNodeError> {
        // The node's own account, not a quorum-confirmed lease: a leader cut
        // off from its peers still holds leadership to hand over, and asking
        // them would wait on peers that may be gone.
        if !self.leads_by_its_own_account() {
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

        // Done when the target leads AND this node has applied an entry of
        // the target's term: that entry is applied only once committed, so
        // the new leader has committed in its term and serves reads and
        // writes, rather than merely having won the vote.
        match self
            .raft
            .wait(Some(std::time::Duration::from_millis(
                LEADERSHIP_TRANSFER_TIMEOUT_MS,
            )))
            .metrics(
                |m| {
                    m.current_leader == Some(target_id)
                        && m.last_applied
                            .as_ref()
                            .map(|l| l.committed_leader_id().term)
                            == Some(m.current_term)
                },
                "leadership transfer",
            )
            .await
        {
            Ok(_) => {
                tracing::info!(
                    node_id = self.node_id,
                    target_id,
                    "leadership transfer successful"
                );
                Ok(())
            }
            Err(openraft::metrics::WaitError::Timeout(..)) => Err(RaftNodeError::TransferTimeout {
                target: target_id,
                timeout_ms: LEADERSHIP_TRANSFER_TIMEOUT_MS,
            }),
            Err(e) => Err(RaftNodeError::Shutdown(format!("transfer_leader: {e}"))),
        }
    }

    /// Force a snapshot at the current applied index and wait until it is
    /// built. Used for pre-shutdown checkpointing and manual log compaction.
    ///
    /// Returns once the snapshot covers every entry applied when the call
    /// began, or an error when it is not built within the snapshot drain
    /// window.
    pub async fn checkpoint(&self) -> Result<(), RaftNodeError> {
        use openraft::async_runtime::watch::WatchReceiver;

        let want = self
            .raft
            .metrics()
            .borrow_watched()
            .last_applied
            .map(|id| id.index);
        self.raft
            .trigger()
            .snapshot()
            .await
            .map_err(|e| RaftNodeError::Shutdown(format!("checkpoint snapshot: {e}")))?;

        // Nothing applied means nothing to snapshot.
        if want.is_some() {
            self.raft
                .wait(Some(SNAPSHOT_WORK_DRAIN))
                .metrics(
                    |m| m.snapshot.map(|id| id.index) >= want,
                    "checkpoint snapshot",
                )
                .await
                .map_err(|e| RaftNodeError::Shutdown(format!("checkpoint snapshot: {e}")))?;
        }

        tracing::info!(
            node_id = self.node_id,
            applied = self.applied_index(),
            "checkpoint complete"
        );
        Ok(())
    }

    /// Graceful shutdown with leadership transfer.
    ///
    /// Orchestrates the shutdown sequence:
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

        // Step 2: Transfer leadership if we're the leader. Each step is logged
        // so a shutdown that does not return shows which one it is stuck in.
        // The node's own account decides: asking the group would wait on
        // peers that may already be gone, and a leader without them could not
        // hand its leadership over anyway.
        let leads = self.leads_by_its_own_account();
        tracing::info!(
            node_id = self.node_id,
            leads,
            "shutdown: leadership checked"
        );
        if leads {
            if let Some(target_id) = self.find_transfer_target() {
                if let Err(e) = self.transfer_leadership_to(target_id).await {
                    tracing::warn!(target_id, "leadership transfer failed: {e}");
                }
            } else {
                tracing::debug!("no peer available for leadership transfer");
            }
        }

        // Step 3: Stop Raft
        tracing::info!(node_id = self.node_id, "shutdown: stopping consensus");
        self.raft
            .shutdown()
            .await
            .map_err(|e| RaftNodeError::Shutdown(e.to_string()))?;

        tracing::info!(node_id = self.node_id, "graceful shutdown complete");
        Ok(())
    }

    /// The voter to hand leadership to: one that acknowledged this leader
    /// within an election timeout, the most caught-up of them (the lowest id
    /// among equals). `None` when this node does not lead or no other voter
    /// is live.
    ///
    /// A voter the leader has not heard from for that long is taken as gone,
    /// as a follower takes its leader: handing leadership to it would wait
    /// out the transfer timeout and leave the group to an election anyway.
    pub fn find_transfer_target(&self) -> Option<u64> {
        use openraft::Instant as _;
        use openraft::async_runtime::watch::WatchReceiver;

        let metrics = self.raft.metrics().borrow_watched().clone();
        let heartbeat = metrics.heartbeat.as_ref()?;
        let replication = metrics.replication.as_ref()?;
        let live_within = std::time::Duration::from_millis(self.raft.config().election_timeout_max);
        let joint = metrics.membership_config.membership().get_joint_config();
        let voters = joint.first()?;
        voters
            .iter()
            .copied()
            .filter(|&id| id != self.node_id)
            .filter(|id| {
                heartbeat
                    .get(id)
                    .copied()
                    .flatten()
                    .is_some_and(|acked| acked.elapsed() <= live_within)
            })
            .max_by_key(|id| {
                let matched = replication
                    .get(id)
                    .and_then(Option::as_ref)
                    .map(|log| log.index);
                (matched, core::cmp::Reverse(*id))
            })
    }

    /// Whether the membership has a voter besides this node.
    fn has_voter_peers(&self) -> bool {
        use openraft::async_runtime::watch::WatchReceiver;

        let metrics = self.raft.metrics().borrow_watched().clone();
        let joint = metrics.membership_config.membership().get_joint_config();
        joint
            .first()
            .is_some_and(|voters| voters.iter().any(|&id| id != self.node_id))
    }

    /// Stop the node. A member of a group with other voters shuts down
    /// gracefully, handing leadership over if it leads; a node alone just
    /// stops.
    pub async fn shutdown(&self) -> Result<(), RaftNodeError> {
        self.version_watch.abort();
        let has_peers = self.has_voter_peers();

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

        // The consensus is down, but a replication task may still sit in a
        // call to a peer that never answers, holding a reader of the log
        // until the call times out. Fail such calls now.
        self.closing.send_replace(true);

        // Free the listen port deterministically. Dropping the sender stops
        // the accept loop, but tonic then holds the LISTENER until every open
        // connection closes — and peers keep idle HTTP/2 channels open
        // indefinitely, so without a bound the port never frees and a node
        // restarting on its own address is locked out. Give the drain a
        // short grace, then abort the serve task; the abort drops the
        // listener.
        let sender = self.grpc_shutdown.lock().ok().and_then(|mut g| g.take());
        drop(sender);
        let task = self.grpc_task.lock().ok().and_then(|mut g| g.take());
        if let Some(mut task) = task {
            if tokio::time::timeout(std::time::Duration::from_millis(500), &mut task)
                .await
                .is_err()
            {
                task.abort();
            }
        }

        // What openraft spawned can outlive its core and hold the engine: a
        // replication task still in a call to a peer keeps its log reader, a
        // snapshot capture or build runs on to its end. The directory stays
        // locked until the last of them lets go. Wait for it: a caller that
        // reopens the directory after this returns must find it free.
        if !self.engine_work.wait_idle(SNAPSHOT_WORK_DRAIN).await {
            tracing::warn!(
                node_id = self.node_id,
                ?SNAPSHOT_WORK_DRAIN,
                "consensus work still holds the engine after shutdown"
            );
        }

        // Flush active memtables to SST so a later reopen sees all writes.
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

    // ── Replication status & staleness tracking ──

    /// Get the current replication status for all cluster nodes.
    ///
    /// Only meaningful when called on the leader — followers don't track
    /// per-node replication progress. Returns `None` if not leader.
    ///
    /// Staleness = `leader_last_log_index - node_matched_index`.
    /// Used by read routing to exclude stale followers.
    pub fn replication_status(&self) -> Option<Vec<NodeReplicationStatus>> {
        use openraft::ServerState;
        use openraft::async_runtime::watch::WatchReceiver;

        let metrics = self.raft.metrics().borrow_watched().clone();

        // Only leader has replication metrics
        if metrics.state != ServerState::Leader {
            return None;
        }

        // Entries held, counted from index 0 (the bootstrap entry): a log whose
        // last index is `i` holds `i + 1`. Log indexes never approach u64::MAX.
        let held = |last: Option<u64>| last.map_or(0, |index| index + 1);
        let leader_last_log = metrics.last_log_index;
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

            let matched_index = matched_log_id.as_ref().map(|id| id.index);
            // A member never holds more than the leader it replicates from:
            // the leader's metrics report both from one state.
            debug_assert!(held(matched_index) <= held(leader_last_log));
            let lag = held(leader_last_log) - held(matched_index);

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
    /// Last log index confirmed replicated to this node; `None` while the node
    /// has not acknowledged any entry (it may be unreachable).
    pub matched_index: Option<u64>,
    /// Number of the leader's log entries this node does not hold: all of them
    /// while it has acknowledged none.
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

// ── Decommission Protocol ─────────────────────────────────────────────────────

/// Result of a successful node decommission.
///
/// Deleting the local data is the operator's responsibility in CE: the
/// server cannot remotely wipe data on the decommissioned node.
#[derive(Debug, Clone)]
pub struct DecommissionResult {
    /// Node ID that was removed from cluster membership.
    pub node_id: u64,
    /// Human-readable summary of phases executed.
    pub message: String,
    /// Set when `pruning=true` was requested.
    /// Operator must manually delete data on the decommissioned node (CE).
    pub operator_cleanup_required: bool,
}

impl RaftNode {
    /// Graceful node decommission.
    ///
    /// ## Step 1: quorum gate (unless `force`)
    /// Verifies that removing `node_id` leaves ≥ 2 voters in the cluster.
    /// CE 3-node minimum: after removal, 2 voters remain (quorum = 2).
    /// Returns `RaftNodeError::Membership` if quorum would be lost or if
    /// `node_id` is not in the current voter set.
    ///
    /// ## Step 2: leadership transfer (unless `force`)
    /// If `node_id` is the current Raft leader, the caller must handle leadership
    /// transfer before calling this method (service layer responsibility). This
    /// method returns `RaftNodeError::Membership` if called as non-leader when
    /// `node_id` is the current leader — the service layer must transfer first.
    ///
    /// ## Step 3: membership remove
    /// Calls `change_membership(remove: node_id)` via openraft. Only succeeds
    /// when this node is the current leader (openraft invariant).
    ///
    /// ## Force path
    /// When `force=true`, skips steps 1 and 2. Useful for permanently
    /// unavailable nodes. May cause data loss if the node held single-copy shards.
    ///
    /// # Errors
    /// - `Membership("not in voter set")` — target not a voter (non-force only)
    /// - `Membership("quorum loss")` — would drop below minimum (non-force only)
    /// - `Membership("target is leader")` — service must transfer leadership first
    /// - `Membership("change_membership failed: …")` — openraft error
    pub async fn decommission_node(
        &self,
        node_id: u64,
        pruning: bool,
        force: bool,
    ) -> Result<DecommissionResult, RaftNodeError> {
        use openraft::async_runtime::watch::WatchReceiver;

        // The gates below read the voter set, which must be the committed one:
        // a change still in flight would be judged by a membership that may
        // not stand, and then refused by openraft anyway.
        self.membership_settled().await?;
        let metrics = self.raft.metrics().borrow_watched().clone();

        // Snapshot current voter set from openraft membership config.
        let current_voters: std::collections::BTreeSet<u64> = {
            let joint = metrics.membership_config.membership().get_joint_config();
            joint
                .first()
                .map(|voters| voters.iter().copied().collect())
                .unwrap_or_default()
        };

        if !force {
            // Step 1a: node must be a current voter.
            if !current_voters.contains(&node_id) {
                return Err(RaftNodeError::Membership(format!(
                    "node {node_id} is not in the current voter set — cannot decommission"
                )));
            }

            // Step 1b: quorum gate — CE requires ≥ 2 voters after removal.
            // Quorum = majority of voters; for 2 voters: majority = 2 (no fault tolerance).
            // For 1 voter: cluster cannot reach consensus.
            // Step 1a proved `node_id` is in the set, so it holds at least one.
            let remaining = current_voters.len() - 1;
            if remaining < 2 {
                return Err(RaftNodeError::Membership(format!(
                    "cannot decommission node {node_id}: would leave {remaining} voter(s), \
                     minimum 2 required for CE cluster quorum. \
                     The cluster must have at least 3 nodes to decommission one."
                )));
            }

            // Step 2: leadership check.
            // If the target node is the current leader and it is NOT us (this node),
            // we cannot call change_membership (only the leader can). Return an error
            // so the service layer can handle forwarding.
            // If the target IS us and we are the leader, the service layer must have
            // already transferred leadership before calling this method.
            let current_leader = metrics.current_leader;
            if current_leader == Some(node_id) && node_id != self.node_id {
                return Err(RaftNodeError::Membership(format!(
                    "node {node_id} is the current Raft leader. \
                     Call DecommissionNode on node {node_id} directly (self-decommission path) \
                     or transfer leadership first via `coordinode admin node transfer-leader`."
                )));
            }
            // If node_id == self.node_id and we are the leader: the service layer is
            // responsible for transferring leadership before calling this method.
            // By the time we reach here, we should no longer be the leader.
            // This is enforced in ClusterServiceImpl::decommission_node().
        }

        // Step 3: membership remove.
        // Compute new voter set (current minus node_id).
        let new_members: std::collections::BTreeSet<u64> = current_voters
            .into_iter()
            .filter(|&id| id != node_id)
            .collect();

        if new_members.is_empty() {
            return Err(RaftNodeError::Membership(
                "cannot remove last voting member — cluster would be empty".to_string(),
            ));
        }

        let written = self
            .raft
            .change_membership(new_members, false)
            .await
            .map_err(|e| {
                RaftNodeError::Membership(format!(
                    "change_membership failed: {}",
                    Self::describe_membership_refusal(&e)
                ))
            })?;
        self.membership_published(written.log_id.index).await?;

        tracing::info!(
            node_id,
            pruning,
            force,
            "decommission: node removed from cluster membership"
        );

        let message = if force {
            format!(
                "Node {node_id} forcibly removed from cluster membership (emergency decommission). \
                 Verify data integrity — some shards may have lost replicas."
            )
        } else {
            format!(
                "Node {node_id} gracefully decommissioned (quorum checked, leadership \
                 moved off, membership removed)."
            )
        };

        Ok(DecommissionResult {
            node_id,
            message,
            operator_cleanup_required: pruning,
        })
    }

    /// Get the gRPC advertise address for a given node ID from openraft membership.
    ///
    /// Returns `None` if the node is not in the current membership config or
    /// if the address is empty. Used for service-layer forwarding in the
    /// self-decommission path.
    pub fn node_address(&self, node_id: u64) -> Option<String> {
        use openraft::async_runtime::watch::WatchReceiver;
        let metrics = self.raft.metrics().borrow_watched().clone();
        // Collect all (id, addr) pairs first — avoids borrow-lifetime issues where
        // `.membership()` returns a reference that borrows from `metrics`.
        let addrs: Vec<(u64, String)> = metrics
            .membership_config
            .membership()
            .nodes()
            .map(|(&id, node)| (id, node.addr.clone()))
            .collect();
        addrs
            .into_iter()
            .find(|(id, _)| *id == node_id)
            .map(|(_, addr)| addr)
            .filter(|addr| !addr.is_empty())
    }
}

// ── Join Protocol ─────────────────────────────────────────────────────────────

/// Phase progression for a node join lifecycle.
///
/// Emitted as part of [`JoinProgressEvent`] during [`RaftNode::monitor_and_promote`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinPhase {
    /// Node added as Learner — receiving log replication, lag closing.
    Learner,
    /// Lag below readiness threshold — about to promote to Voter.
    ReadyCheck,
    /// `change_membership` in progress — node becoming a Voter.
    Promoting,
    /// Node is now a Voter. Join complete.
    Complete,
    /// Join failed (see `message` field for details).
    Failed,
}

/// Progress event emitted during a node join lifecycle.
///
/// Broadcast via `tokio::sync::broadcast` from [`RaftNode::monitor_and_promote`].
/// Consumers (e.g., the `JoinProgress` gRPC stream) subscribe and map these
/// to proto `JoinStatus` messages.
#[derive(Debug, Clone)]
pub struct JoinProgressEvent {
    /// Node ID of the joining node.
    pub node_id: u64,
    /// Current join phase.
    pub phase: JoinPhase,
    /// Number of Raft log entries behind the leader. `u64::MAX` means "not yet known".
    pub lag_entries: u64,
    /// Estimated completion percentage (0–100).
    pub percent: u8,
    /// Human-readable status message.
    pub message: String,
}

impl RaftNode {
    /// Monitor replication lag for a Learner node and promote it to Voter when ready.
    ///
    /// Called after the node has been added as a Learner via [`Self::add_node`].
    /// Polls replication metrics every 500ms until the node has answered and
    /// lacks at most [`Self::join_readiness_lag`] entries, then calls
    /// [`Self::change_membership`] to promote the node to a Voter.
    ///
    /// Broadcasts [`JoinProgressEvent`] at each phase transition and as the
    /// replication metrics move, at most once a second, so callers can stream
    /// progress to operators.
    ///
    /// # Errors
    ///
    /// Returns [`RaftNodeError::Membership`] if `change_membership` fails.
    /// The node remains a Learner on failure (no automatic rollback; the caller
    /// should call [`Self::remove_node`] to clean up).
    ///
    /// # Timeout
    ///
    /// Aborts with an error once [`Self::join_timeout`] passes without the
    /// node catching up, so a permanently stale node cannot hold the join.
    pub async fn monitor_and_promote(
        &self,
        node_id: u64,
        progress_tx: tokio::sync::broadcast::Sender<JoinProgressEvent>,
    ) -> Result<(), RaftNodeError> {
        // Shortest gap between two looks at the replication metrics, so a
        // fast catch-up reports progress at a readable rate.
        const REPORT_GAP: std::time::Duration = std::time::Duration::from_secs(1);
        use openraft::rt::watch::WatchReceiver;

        let timeout = self.join_timeout();
        let deadline = tokio::time::Instant::now() + timeout;
        let mut initial_lag: Option<u64> = None;
        let mut metrics = self.raft.metrics();
        let mut looked_at = tokio::time::Instant::now();

        // Step 1: wait until the node has answered and lags little enough.
        // Replication progress moves the metrics; nothing else can change
        // the answer, so the loop sleeps until they move.
        loop {
            if tokio::time::Instant::now() >= deadline {
                let _ = progress_tx.send(JoinProgressEvent {
                    node_id,
                    phase: JoinPhase::Failed,
                    lag_entries: 0,
                    percent: 0,
                    message: format!(
                        "Join timed out after {} s: node failed to catch up",
                        timeout.as_secs()
                    ),
                });
                return Err(RaftNodeError::Membership(format!(
                    "join timed out: node did not catch up within {} s",
                    timeout.as_secs()
                )));
            }

            tokio::select! {
                changed = metrics.changed() => {
                    if changed.is_err() {
                        return Err(RaftNodeError::Membership(
                            "join aborted: the raft instance shut down".into(),
                        ));
                    }
                }
                () = tokio::time::sleep_until(deadline) => continue,
            }
            tokio::time::sleep_until((looked_at + REPORT_GAP).min(deadline)).await;
            looked_at = tokio::time::Instant::now();

            let lag = match self.lag_for_node(node_id) {
                Some(l) => l,
                None => {
                    // Node not in replication metrics yet — still initializing.
                    let _ = progress_tx.send(JoinProgressEvent {
                        node_id,
                        phase: JoinPhase::Learner,
                        lag_entries: u64::MAX,
                        percent: 0,
                        message: "Waiting for replication to start on joining node".into(),
                    });
                    continue;
                }
            };

            // Record initial lag for percentage calculation (set once, on first measurement).
            let start = *initial_lag.get_or_insert(lag.max(1));

            let threshold = self.join_readiness_lag();
            if lag <= threshold {
                let _ = progress_tx.send(JoinProgressEvent {
                    node_id,
                    phase: JoinPhase::ReadyCheck,
                    lag_entries: lag,
                    percent: 99,
                    message: format!(
                        "Lag {lag} entries — below threshold ({threshold}), promoting to Voter"
                    ),
                });
                break;
            }

            let done = start.saturating_sub(lag);
            let percent = ((done * 98 / start) as u8).min(98);

            let _ = progress_tx.send(JoinProgressEvent {
                node_id,
                phase: JoinPhase::Learner,
                lag_entries: lag,
                percent,
                message: format!("Catching up: {lag} entries behind leader"),
            });
        }

        // Step 2: promote to Voter.
        let _ = progress_tx.send(JoinProgressEvent {
            node_id,
            phase: JoinPhase::Promoting,
            lag_entries: 0,
            percent: 99,
            message: "Promoting node to Voter via change_membership".into(),
        });

        // Collect current voters and include the new node.
        let mut new_members: Vec<u64> = {
            use openraft::async_runtime::watch::WatchReceiver;
            let rx = self.raft.metrics();
            let m = rx.borrow_watched();
            m.membership_config.membership().voter_ids().collect()
        };
        if !new_members.contains(&node_id) {
            new_members.push(node_id);
        }

        if let Err(e) = self.change_membership(new_members).await {
            // Graceful rollback: remove the Learner from membership so it stops
            // consuming replication bandwidth. Best-effort — log if remove fails.
            if let Err(rm_err) = self.remove_node(node_id).await {
                tracing::warn!(
                    node_id,
                    error = %rm_err,
                    "monitor_and_promote: rollback remove_node failed (node remains as Learner)"
                );
            }
            let _ = progress_tx.send(JoinProgressEvent {
                node_id,
                phase: JoinPhase::Failed,
                lag_entries: 0,
                percent: 99,
                message: format!("change_membership failed: {e}"),
            });
            return Err(e);
        }

        let _ = progress_tx.send(JoinProgressEvent {
            node_id,
            phase: JoinPhase::Complete,
            lag_entries: 0,
            percent: 100,
            message: "Node is now a Voter — join complete".into(),
        });

        Ok(())
    }

    /// Replication lag for a specific node ID, if available in leader metrics.
    ///
    /// Returns `None` if this node is not the leader, if the target node is
    /// not yet in the replication metrics, or if it has not acknowledged any
    /// entry: a lag is only known once replication has reached the node.
    fn lag_for_node(&self, node_id: u64) -> Option<u64> {
        let statuses = self.replication_status()?;
        statuses
            .into_iter()
            .find(|s| s.node_id == node_id)
            .and_then(|s| s.matched_index.map(|_| s.lag_entries))
    }
}

/// Spawn a background task that periodically checks snapshot triggers.
///
/// Complements openraft's own entry-count trigger with the other two of
/// [`SnapshotTriggerConfig`]: the bytes the log grew by since the last
/// snapshot, probed after an apply at most once a second, and the periodic
/// timer. Neither fires while nothing was applied since the last snapshot,
/// and the task sleeps until `applied_rx` moves.
///
/// The task runs until the Raft instance is shut down (detected via `trigger()` error).
fn spawn_snapshot_trigger(
    raft: Arc<RaftInstance>,
    engine: Arc<StorageEngine>,
    work: &crate::storage::EngineWork,
    config: SnapshotTriggerConfig,
    mut applied_rx: tokio::sync::watch::Receiver<u64>,
) -> tokio::task::JoinHandle<()> {
    // The task holds the engine until it is dropped, which an abort does
    // later, on the runtime; the shutdown waits for the work guard to go, and
    // the engine goes first.
    let held = TriggerHold {
        engine,
        _work: work.start(),
    };
    tokio::spawn(async move {
        use openraft::rt::watch::WatchReceiver;

        let engine = &held.engine;

        let probe = config.check_interval.min(SNAPSHOT_SIZE_PROBE);
        let metrics_rx = raft.metrics();
        let mut last = tokio::time::Instant::now();
        // Don't probe immediately on startup.
        let mut last_probe = last;
        // The log's size when the last snapshot was asked for; growth is
        // measured from it. A purge that shrinks the log lowers it.
        let mut base = crate::storage::raft_log_bytes(engine).unwrap_or(0);

        loop {
            // The log grows only as entries arrive, so it is looked at after
            // an apply, at most once per probe period: a build in progress is
            // not asked for again before the period ends.
            tokio::time::sleep_until(last_probe + probe).await;
            // Marked seen before the state is read: an entry applied after
            // it wakes the waits below. Read after the pause, so a build that
            // finished during it counts.
            let applied = *applied_rx.borrow_and_update();
            let snapped = metrics_rx
                .borrow_watched()
                .snapshot
                .map(|id| id.index)
                .unwrap_or(0);
            // A snapshot captures every partition, so asking for one when
            // nothing was applied since the last is pure waste: sleep until
            // an entry applies.
            if applied <= snapped {
                if applied_rx.changed().await.is_err() {
                    break;
                }
                continue;
            }
            last_probe = tokio::time::Instant::now();
            let size = match crate::storage::raft_log_bytes(engine) {
                Ok(size) => size,
                Err(e) => {
                    tracing::warn!(%e, "snapshot trigger: cannot size the raft log");
                    base
                }
            };
            base = base.min(size);
            let Some(reason) = snapshot_due(size - base, last.elapsed(), &config) else {
                // Not due yet: the next apply or the interval decides.
                tokio::select! {
                    changed = applied_rx.changed() => {
                        if changed.is_err() {
                            break;
                        }
                    }
                    () = tokio::time::sleep_until(last + config.check_interval) => {}
                }
                continue;
            };

            match raft.trigger().snapshot().await {
                Ok(()) => {
                    tracing::debug!(
                        applied,
                        snapshot = snapped,
                        reason,
                        "snapshot trigger: requested snapshot build"
                    );
                    last = tokio::time::Instant::now();
                    base = size;
                }
                Err(_fatal) => {
                    // Raft instance shut down — exit the trigger loop
                    tracing::debug!("snapshot trigger: raft shut down, stopping");
                    break;
                }
            }
        }
    })
}

/// What the snapshot trigger task holds. Fields drop in order: the engine
/// before the guard that tells the shutdown the task let go of it.
struct TriggerHold {
    engine: Arc<StorageEngine>,
    _work: crate::storage::EngineWorkGuard,
}

/// How often the trigger task sizes the Raft log.
const SNAPSHOT_SIZE_PROBE: std::time::Duration = std::time::Duration::from_secs(1);

/// Why a snapshot is due, given the bytes the log grew by and the time since
/// the last one was asked for; `None` when neither threshold is reached.
fn snapshot_due(
    grown: u64,
    since_last: std::time::Duration,
    config: &SnapshotTriggerConfig,
) -> Option<&'static str> {
    if grown >= config.log_bytes {
        Some("log size")
    } else if since_last >= config.check_interval {
        Some("interval")
    } else {
        None
    }
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

    /// Leadership was handed to `target`, but `target` did not become the
    /// leader in time (it is down, partitioned, or lost the election).
    #[error("leadership did not move to node {target} within {timeout_ms} ms")]
    TransferTimeout { target: u64, timeout_ms: u64 },

    /// A node was asked to join a group while holding data of its own.
    ///
    /// Only the member a group is formed around brings data into it. A
    /// joining node receives the group's state and would have to drop
    /// whatever it held, so the refusal happens before anything is lost:
    /// either start the group from this node, or join with an empty store.
    #[error(
        "this node holds data of its own and cannot join an existing group: a joining node \
         receives the group's state, which would replace what is here. Form the group from \
         this node instead, or point it at an empty data directory"
    )]
    JoinWithLocalData,
}

/// Refuse to join a group while this store holds data of its own.
///
/// What the store holds is what its trees hold plus what its log will replay
/// into them: after a crash an acknowledged write may live only in the log.
/// Both reads are cheap on the empty store a joining node is supposed to
/// have: one key at most per partition, and a log with nothing to replay.
fn refuse_join_with_local_data(
    engine: &StorageEngine,
    log_store: &LogStore,
    state_machine: &CoordinodeStateMachine,
) -> Result<(), RaftNodeError> {
    fn init(e: impl std::fmt::Display) -> RaftNodeError {
        RaftNodeError::Init(e.to_string())
    }
    if engine.holds_user_data().map_err(init)? {
        return Err(RaftNodeError::JoinWithLocalData);
    }
    let from = state_machine.next_to_apply().map_err(init)?;
    if log_store.writes_user_data_from(from).map_err(init)? {
        return Err(RaftNodeError::JoinWithLocalData);
    }
    Ok(())
}

/// Publish what this store already holds as the base state of the group it
/// has just formed, so a member added later receives it.
///
/// A group formed around a store that already holds data starts with a
/// populated state machine and a log that never carried any of it: the data
/// was written before there was a group, by a standalone node or by the
/// non-replicated embedded build. A member added afterwards is caught up
/// from the log by default, and a log that never held the data cannot
/// deliver it, so the new member would come up empty while the leader shows
/// the data, and a later leadership move would make it disappear.
///
/// Taking a snapshot and purging the log up to it leaves the leader with
/// nothing a new member could be caught up from except the snapshot, which
/// is built from the state machine and therefore carries everything. This
/// runs only when the group is formed (the open that initialized it) and
/// only when there is data to publish, so the ordinary empty start does
/// nothing at all.
async fn publish_existing_state_as_group_base(
    raft: &RaftInstance,
    engine: &StorageEngine,
) -> Result<(), RaftNodeError> {
    if !engine
        .holds_user_data()
        .map_err(|e| RaftNodeError::Init(e.to_string()))?
    {
        return Ok(());
    }

    // The snapshot is taken at the applied index, so wait for the membership
    // entry this open just proposed to apply; snapshotting before it would
    // publish a base the group cannot place in its own history.
    let applied = raft
        .wait(Some(std::time::Duration::from_secs(5)))
        .metrics(|m| m.last_applied.is_some(), "the membership entry applies")
        .await
        .ok()
        .and_then(|m| m.last_applied);
    let Some(applied) = applied else {
        return Err(RaftNodeError::Init(
            "a group formed around existing data never applied its own membership entry, so \
             that data could not be published as the group's base state"
                .to_string(),
        ));
    };

    raft.trigger()
        .snapshot()
        .await
        .map_err(|e| RaftNodeError::Init(format!("snapshot of the existing state: {e}")))?;

    let snapshot = raft
        .wait(Some(std::time::Duration::from_secs(10)))
        .metrics(
            |m| m.snapshot.is_some_and(|s| s.index >= applied.index),
            "the snapshot of the existing state completes",
        )
        .await
        .ok()
        .and_then(|m| m.snapshot);
    let Some(snapshot) = snapshot.filter(|s| s.index >= applied.index) else {
        return Err(RaftNodeError::Init(
            "the snapshot carrying this store's existing data never completed, so the data \
             could not be published as the group's base state"
                .to_string(),
        ));
    };

    raft.trigger()
        .purge_log(snapshot.index)
        .await
        .map_err(|e| RaftNodeError::Init(format!("purge of the pre-group log: {e}")))?;

    tracing::info!(
        snapshot_index = snapshot.index,
        "published the store's existing data as the group's base state"
    );
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
