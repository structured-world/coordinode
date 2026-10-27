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
//!
//! ## Node identity
//!
//! A member that talks to peers is opened with its security context, inside
//! [`NodeOptions::connections`]: the node identity it proves and the trust
//! its peers are verified under. There is no unauthenticated channel, so a
//! node opened without one has no peers. Every member is bound to its node
//! identity in the membership; the deployment is identified when its first
//! group forms, and a joining node is provisioned with it.

pub mod admission;
mod catch_up;
pub mod grpc_server;
pub mod nemesis;
pub(crate) mod network;
pub mod timing;
pub mod version;

pub use timing::{ElectionPreference, ElectionTiming, TimingError};

use std::sync::Arc;

pub use coordinode_core::group::{DeploymentId, GroupId};
use coordinode_core::txn::proposal::MetadataCommand;
use coordinode_storage::engine::core::StorageEngine;
pub use coordinode_wire::peer::{PeerConnections, PeerTarget};

use crate::member::{MemberIdentity, MemberNode};
use crate::proposal::{RaftProposalPipeline, RateLimiter};
use crate::storage::{
    AppendNotifier, CoordinodeStateMachine, LogStore, LostCommitted, TypeConfig,
    default_raft_config, has_other_voters,
};
use crate::wait_majority::{BatchConfig, WaitForMajorityService};
use catch_up::CatchUp;

pub use admission::{Admission, DEPLOYMENT_METADATA_KEY, Enrollment};
pub use grpc_server::RaftGrpcHandler;
use network::{GrpcNetworkFactory, StubNetworkFactory};
use version::VersionGate;

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
    /// Shortest time between two snapshots either threshold asks for
    /// (default 60 s). Time alone never asks for one: a snapshot captures
    /// every table of the store, so it is taken when the log has grown, not
    /// on a clock, and never more often than this however fast it grows.
    pub min_interval: std::time::Duration,
}

impl Default for SnapshotTriggerConfig {
    fn default() -> Self {
        Self {
            logs_since_last: 10_000,
            log_bytes: 256 * 1024 * 1024,
            min_interval: std::time::Duration::from_secs(60),
        }
    }
}

impl SnapshotTriggerConfig {
    /// The openraft configuration under this snapshot policy and `timing`.
    /// Both snapshot thresholds are the trigger task's, so the minimum
    /// interval bounds them alike; openraft builds a snapshot only when the
    /// task asks.
    fn raft_config(
        &self,
        timing: ElectionTiming,
        preference: ElectionPreference,
    ) -> openraft::Config {
        // The test harness's generous window: shared CI runners starve
        // heartbeats, and only a longer election timeout tolerates it. It
        // only ever lengthens the group's window, and a member's preference
        // is derived from the result, so a member that stands last still
        // stands after every other.
        let timing = if std::env::var_os("COORDINODE_TEST_RAFT_GENEROUS_TIMEOUTS").is_some() {
            timing.at_least(
                std::time::Duration::from_millis(1500),
                std::time::Duration::from_millis(3000),
            )
        } else {
            timing
        };
        let (heartbeat, election_min, election_max) = timing.for_member(preference).millis();
        openraft::Config {
            snapshot_policy: openraft::SnapshotPolicy::Never,
            heartbeat_interval: heartbeat,
            election_timeout_min: election_min,
            election_timeout_max: election_max,
            ..default_raft_config()
        }
    }
}

/// How a node is opened beyond its identity, storage and address.
#[derive(Default)]
pub struct NodeOptions {
    /// When the node snapshots its state.
    pub snapshots: SnapshotTriggerConfig,
    /// The group's election and heartbeat timing.
    pub timing: ElectionTiming,
    /// When this member stands for election relative to the others.
    pub preference: ElectionPreference,
    /// The embedding application's format epoch, half of the version pair
    /// members are matched on. A server runs zero.
    pub host_epoch: u64,
    /// The consensus group this node is a member of. Every consensus message
    /// it sends names it, and a server hosting several groups dispatches by
    /// it.
    pub group: GroupId,
    /// The node's connections to its peers, made under its security
    /// context: the node identity it proves and the trust its peers are
    /// verified under. Required to open a member that talks to peers. The
    /// nodes of every group one server hosts are opened with the same set,
    /// so the groups two servers share travel over one connection between
    /// them.
    pub connections: Option<Arc<PeerConnections>>,
    /// How a joining node is provisioned: the deployment it joins and the
    /// members it trusts until it has a membership of its own. Required to
    /// join.
    pub enrollment: Option<Enrollment>,
    /// The deployment a group being formed belongs to, when it is not the
    /// deployment's first: another group of the same deployment. `None`
    /// identifies a new deployment.
    pub deployment: Option<DeploymentId>,
}

/// Writes the metadata commands a node proposes on its own behalf: the
/// deployment record, the group's version pair, retired identities.
#[derive(Clone)]
struct Commands {
    raft: Arc<RaftInstance>,
    engine: Arc<crate::engine_hold::EngineHold>,
    oracle: Arc<coordinode_core::txn::timestamp::TimestampOracle>,
    ids: Arc<coordinode_core::txn::proposal::ProposalIdGenerator>,
}

impl Commands {
    fn new(
        raft: Arc<RaftInstance>,
        engine: Arc<StorageEngine>,
        oracle: Option<Arc<coordinode_core::txn::timestamp::TimestampOracle>>,
    ) -> Self {
        // An engine opened without an oracle stamps writes from a counter; a
        // command then takes a wall-clock stamp of its own.
        let oracle = oracle
            .unwrap_or_else(|| Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new()));
        let ids = coordinode_core::txn::proposal::ProposalIdGenerator::with_base(
            coordinode_core::txn::proposal::fresh_proposal_id_base(),
        );
        Self {
            raft,
            // One hold for every clone of the commands: they share it.
            engine: Arc::new(crate::engine_hold::EngineHold::new(
                engine,
                "metadata commands",
            )),
            oracle,
            ids: Arc::new(ids),
        }
    }

    /// Commit `command` through this node, which must lead; returns once it
    /// applied here.
    async fn write(&self, command: MetadataCommand) -> Result<(), String> {
        // Held as not yet logged until it is handed over, so no entry's
        // bound passes over it meanwhile.
        let (at, held) = self
            .engine
            .pending_commits()
            .obligate(|| self.oracle.next().as_raw());
        let mut request = command_request(
            command,
            self.ids.next(),
            coordinode_core::txn::timestamp::Timestamp::from_raw(at),
        );
        let written = match crate::closure::stamp(&self.raft, &self.engine, &mut request).await {
            Ok(()) => self
                .raft
                .client_write(request)
                .await
                .map(|_| ())
                .map_err(|e| e.to_string()),
            Err(e) => Err(e.to_string()),
        };
        drop(held);
        written
    }
}

/// Record the deployment identity `given`, or a fresh one, unless one is
/// recorded already.
async fn found_deployment(
    commands: &Commands,
    recorded: &mut tokio::sync::watch::Receiver<Option<DeploymentId>>,
    given: Option<DeploymentId>,
) -> Result<DeploymentId, RaftNodeError> {
    if let Some(id) = *recorded.borrow() {
        return Ok(id);
    }
    let id = match given {
        Some(id) => id,
        None => {
            let mut id = [0u8; 16];
            getrandom::fill(&mut id)
                .map_err(|e| RaftNodeError::Init(format!("a random deployment identity: {e:?}")))?;
            DeploymentId(id)
        }
    };
    commands
        .write(MetadataCommand::RecordDeployment { id })
        .await
        .map_err(|e| RaftNodeError::Init(format!("record the deployment identity: {e}")))?;
    // The leader answers once the record applied here; the first one stands.
    let recorded = *recorded
        .wait_for(Option::is_some)
        .await
        .map_err(|e| RaftNodeError::Init(format!("the deployment record: {e}")))?;
    recorded.ok_or_else(|| RaftNodeError::Init("the deployment record vanished".to_owned()))
}

/// The version gate of member `node_id` of `group`, over its state
/// machine's record of the group's pair.
fn version_gate(
    node_id: u64,
    group: GroupId,
    state_machine: &CoordinodeStateMachine,
    host_epoch: u64,
) -> Arc<VersionGate> {
    Arc::new(VersionGate::new(
        node_id,
        group,
        coordinode_core::version::VersionPair::current(host_epoch),
        state_machine.subscribe_group_pair(),
        state_machine.applied_commit_ts_handle(),
    ))
}

/// Keep the gate's view of the leader current, and while this node leads,
/// record its pair as the group's when the group does not run it yet: the
/// first leader at a pair commits it as soon as it leads. Keeps `book`, when
/// the node has peers, at the membership in effect.
fn spawn_version_watch(
    raft: Arc<RaftInstance>,
    gate: Arc<VersionGate>,
    engine: Arc<StorageEngine>,
    commands: Commands,
    book: Option<Arc<network::MemberBook>>,
) -> BackgroundTasks {
    let closer =
        crate::closure::spawn_closer(Arc::clone(&raft), Arc::clone(&engine), CLOSURE_SETTLE);
    let space = Arc::clone(engine.space());
    let watch = tokio::spawn(async move {
        use openraft::LogIdOptionExt;
        use openraft::rt::watch::WatchReceiver;
        let mut metrics = raft.metrics();
        // The term this node last recorded its pair in, so a refused or
        // lost attempt is retried once per change, not in a loop.
        let mut recorded_in: Option<u64> = None;
        // The membership the book holds, so it is rewritten once per change.
        let mut booked: Option<Option<u64>> = None;
        loop {
            let (leader, leading_term) = {
                let m = metrics.borrow_watched();
                if let Some(book) = &book {
                    let index = m.membership_config.log_id().index();
                    if booked != Some(index) {
                        book.set(m.membership_config.membership().nodes());
                        booked = Some(index);
                    }
                }
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
            // Whether a record this leader owes waits for disk space: then
            // the task wakes on its own to look again, not only on a change.
            let mut waits_for_space = false;
            if let Some(term) = leading_term {
                // A log append on a full disk fails its fsync and stops the
                // node; the record waits for space like any write. A refused
                // or lost attempt is retried on the next change.
                let room = space.admit().is_ok();
                waits_for_space = recorded_in != Some(term) && !room;
                if recorded_in != Some(term) && room {
                    if let Some(pair) = gate.pair_to_record() {
                        let written = commands
                            .write(MetadataCommand::RecordGroupPair { pair })
                            .await;
                        match written {
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
            if waits_for_space {
                tokio::select! {
                    changed = metrics.changed() => {
                        if changed.is_err() {
                            break;
                        }
                    }
                    () = tokio::time::sleep(SPACE_RECHECK) => {}
                }
            } else if metrics.changed().await.is_err() {
                break;
            }
        }
    });
    BackgroundTasks {
        running: parking_lot::Mutex::new(vec![watch, closer]),
    }
}

/// The node's own background tasks, stopped together when it shuts down:
/// the version watch (follows leadership, records the group's version pair)
/// and the closer (appends a closing entry when commits are left without a
/// bound). Both hold the engine.
struct BackgroundTasks {
    running: parking_lot::Mutex<Vec<tokio::task::JoinHandle<()>>>,
}

impl BackgroundTasks {
    /// Abort the tasks and wait until they are gone: an abort only marks a
    /// task, which lets go of the engine when the runtime next drops it.
    async fn stop(&self) {
        let tasks = core::mem::take(&mut *self.running.lock());
        for task in &tasks {
            task.abort();
        }
        for task in tasks {
            match task.await {
                Err(e) if !e.is_cancelled() => {
                    tracing::warn!(error = %e, "background task ended abnormally");
                }
                _ => {}
            }
        }
    }
}

impl Drop for BackgroundTasks {
    fn drop(&mut self) {
        for task in self.running.get_mut().iter() {
            task.abort();
        }
    }
}

/// How often a leader that owes the group's version record looks again for
/// the disk space to write it.
const SPACE_RECHECK: std::time::Duration = std::time::Duration::from_secs(1);

/// How long a leader waits after a commit lands for a following entry to
/// carry its bound before it appends a closing entry of its own: long enough
/// that a steady stream of writes covers itself, short against how long a
/// member's cut may trail an idle group's last commit.
const CLOSURE_SETTLE: std::time::Duration = std::time::Duration::from_millis(5);

/// The entry carrying `command`.
fn command_request(
    command: MetadataCommand,
    id: coordinode_core::txn::proposal::ProposalId,
    at: coordinode_core::txn::timestamp::Timestamp,
) -> crate::storage::Request {
    use coordinode_core::txn::proposal::{Mutation, RaftProposal};
    crate::storage::Request::single(RaftProposal {
        id,
        mutations: vec![Mutation::Command(command)],
        commit_ts: at,
        start_ts: at,
        bypass_rate_limiter: true,
    })
}

/// How long opening a node that is its group's only voter waits for it to
/// elect itself: one vote made durable, so seconds only on a stalled disk.
const SINGLE_NODE_ELECTION_WAIT: std::time::Duration = std::time::Duration::from_secs(10);

/// Whether a member being opened forms (or resumes) its group, or joins one.
enum Founding {
    /// The group forms around this node, reached at `advertise`.
    Form { advertise: String },
    /// The node joins a group whose leader adds it.
    Join,
}

/// Whether `node_id` is the only voter of the membership in effect.
fn only_voter(raft: &RaftInstance, node_id: u64) -> bool {
    use openraft::rt::watch::WatchReceiver;
    let metrics = raft.metrics();
    let metrics = metrics.borrow_watched();
    let joint = metrics.membership_config.membership().get_joint_config();
    matches!(joint.as_slice(), [voters] if voters.len() == 1 && voters.contains(&node_id))
}

/// Wait until the metrics show the membership just initialized or
/// restored; whether there is one.
async fn await_membership(raft: &RaftInstance) -> Result<bool, RaftNodeError> {
    raft.wait(Some(SINGLE_NODE_ELECTION_WAIT))
        .metrics(
            |m| {
                m.membership_config
                    .membership()
                    .get_joint_config()
                    .first()
                    .is_some_and(|voters| !voters.is_empty())
            },
            "the membership shows in the metrics",
        )
        .await
        .map(|_| true)
        .map_err(|e| RaftNodeError::Init(format!("the membership did not show: {e}")))
}

/// Whether `node_id` leads a group whose only voter it is.
fn leads_alone(raft: &RaftInstance, node_id: u64) -> bool {
    use openraft::rt::watch::WatchReceiver;
    only_voter(raft, node_id) && raft.metrics().borrow_watched().state.is_leader()
}

/// Wait for a group's only voter to elect itself and apply the entry that
/// opens its term, so it takes writes.
async fn await_self_election(raft: &RaftInstance) -> Result<(), RaftNodeError> {
    raft.wait(Some(SINGLE_NODE_ELECTION_WAIT))
        .metrics(
            |m| {
                m.state.is_leader()
                    && m.vote.is_committed()
                    && m.last_applied.map(|id| id.index) >= m.last_log_index
            },
            "the only voter elects itself",
        )
        .await
        .map(|_| ())
        .map_err(|e| RaftNodeError::Init(format!("the node did not become leader: {e}")))
}

/// Schema-partition key recording the deployment a joining store is bound
/// to before the deployment's own record reaches it.
const KEY_ENROLLED_DEPLOYMENT: &[u8] = coordinode_core::group::ENROLLED_DEPLOYMENT_KEY;

/// Schema-partition key recording the members a joining store was
/// provisioned to trust, kept with its enrollment: a store restarted before
/// its group's membership reached it still knows whom to ask who the members
/// are, and so admits the leader that brings it the log.
const KEY_ENROLLED_AUTHORITIES: &[u8] = b"raft:enrolled_authorities";

/// Bind this store to the deployment it is being enrolled in, or refuse an
/// enrollment in another one: a store bootstrapping into a deployment, or
/// already part of one, never takes in another's state. The binding is
/// durable before any peer is heard, so an interrupted bootstrap resumes
/// into the same deployment or is refused.
fn bind_enrollment(
    engine: &StorageEngine,
    state_machine: &CoordinodeStateMachine,
    enrollment: &Enrollment,
) -> Result<(), RaftNodeError> {
    use coordinode_storage::engine::partition::Partition;
    let refuse = |bound: DeploymentId| RaftNodeError::OtherDeployment {
        bound,
        asked: enrollment.deployment,
    };
    if let Some(recorded) = *state_machine.subscribe_deployment().borrow() {
        if recorded != enrollment.deployment {
            return Err(refuse(recorded));
        }
    }
    if let Some(bound) = enrolled_deployment(engine)? {
        if bound != enrollment.deployment {
            return Err(refuse(bound));
        }
    }
    // The authorities first: a recorded deployment then always comes with
    // the members to ask about it.
    let authorities: Vec<MemberNode> = enrollment
        .authorities
        .iter()
        .map(|t| MemberNode::new(t.addr.clone(), Some(MemberIdentity::from(&t.identity))))
        .collect();
    let authorities = rmp_serde::to_vec(&authorities)
        .map_err(|e| RaftNodeError::Init(format!("encode the enrollment's authorities: {e}")))?;
    engine
        .put(Partition::Schema, KEY_ENROLLED_AUTHORITIES, &authorities)
        .and_then(|()| {
            engine.put(
                Partition::Schema,
                KEY_ENROLLED_DEPLOYMENT,
                enrollment.deployment.as_bytes(),
            )
        })
        .and_then(|()| engine.persist())
        .map_err(|e| RaftNodeError::Init(format!("record the enrollment: {e}")))
}

/// The members a store enrolled as a joining member was provisioned to
/// trust; none for a store never enrolled.
fn enrolled_authorities(
    engine: &StorageEngine,
) -> Result<Vec<coordinode_wire::peer::PeerTarget>, RaftNodeError> {
    use coordinode_storage::engine::partition::Partition;
    let Some(bytes) = engine
        .get(Partition::Schema, KEY_ENROLLED_AUTHORITIES)
        .map_err(|e| RaftNodeError::Init(format!("read the enrollment's authorities: {e}")))?
    else {
        return Ok(Vec::new());
    };
    let nodes: Vec<MemberNode> = rmp_serde::from_slice(&bytes).map_err(|e| {
        RaftNodeError::Init(format!("the enrollment's authorities do not decode: {e}"))
    })?;
    Ok(nodes.iter().filter_map(admission::member_target).collect())
}

/// The deployment this store was enrolled in as a joining member, if it
/// was: such a store is a member of a group it did not found, whether or not
/// the group's log has reached it yet.
fn enrolled_deployment(engine: &StorageEngine) -> Result<Option<DeploymentId>, RaftNodeError> {
    use coordinode_storage::engine::partition::Partition;
    let enrolled = engine
        .get(Partition::Schema, KEY_ENROLLED_DEPLOYMENT)
        .map_err(|e| RaftNodeError::Init(format!("read the enrollment: {e}")))?;
    match enrolled.as_deref().map(DeploymentId::from_slice) {
        Some(Some(bound)) => Ok(Some(bound)),
        Some(None) => Err(RaftNodeError::Init(
            "the store's enrollment record does not decode".to_owned(),
        )),
        None => Ok(None),
    }
}

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
    engine: crate::engine_hold::EngineHold,
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
    version_watch: BackgroundTasks,
    /// Committed entries this member's disk lost and has not received again;
    /// until then it neither votes nor campaigns.
    catch_up: CatchUp,
    /// Lifts `catch_up` once a leader has sent the entries again.
    catch_up_watch: BackgroundTasks,
    /// How long a membership change waits for the previous one to settle,
    /// in ms. See [`Self::set_membership_settle_timeout`].
    membership_settle_timeout_ms: core::sync::atomic::AtomicU64,
    /// When a joining learner is promoted, and how long a join may take. See
    /// [`Self::set_join_readiness_lag`] and [`Self::set_join_timeout`].
    join: JoinTuning,
    /// Which peers this member admits; `None` for a node without peers.
    admission: Option<Arc<Admission>>,
    /// The connections to peers, under this node's security context; `None`
    /// for a node without peers.
    connections: Option<Arc<PeerConnections>>,
    /// The deployment, as the applied state records it.
    deployment: tokio::sync::watch::Receiver<Option<DeploymentId>>,
    /// Writes this node's own metadata commands.
    commands: Commands,
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
        // A single-node group talks to no peer, and needs no node identity.
        let NodeOptions {
            snapshots: snap_config,
            timing,
            preference,
            host_epoch,
            group,
            connections: _,
            enrollment: _,
            deployment: forming_in,
        } = options;
        let config = Arc::new(snap_config.raft_config(timing, preference));
        let log_store = LogStore::open(Arc::clone(&engine)).map_err(RaftNodeError::log_open)?;
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
        let version = version_gate(node_id, group, &state_machine, host_epoch);
        let mut deployment = state_machine.subscribe_deployment();

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
        members.insert(node_id, MemberNode::default());

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
        let commands = Commands::new(Arc::clone(&raft), Arc::clone(&engine), oracle);
        // The deployment is identified when its first group forms, so a
        // store that grows later keeps the identity it started with.
        if leads_alone(&raft, node_id) {
            found_deployment(&commands, &mut deployment, forming_in).await?;
        }
        let snap_handle = spawn_snapshot_trigger(
            Arc::clone(&raft),
            Arc::clone(&engine),
            &engine_work,
            snap_config,
            applied_rx.clone(),
        );
        let version_watch = spawn_version_watch(
            Arc::clone(&raft),
            Arc::clone(&version),
            Arc::clone(&engine),
            commands.clone(),
            None,
        );

        Ok(Self {
            raft,
            applied_rx,
            snapshot_builds,
            engine_work,
            node_id,
            advertise_addr: None,
            engine: crate::engine_hold::EngineHold::new(engine, "raft node"),
            oplog,
            append_notifier,
            grpc_shutdown: std::sync::Mutex::new(None),
            grpc_task: std::sync::Mutex::new(None),
            // No peers to call.
            closing: tokio::sync::watch::Sender::new(false),
            _snapshot_trigger: Some(snap_handle),
            version,
            version_watch,
            // A node without peers refuses a disk that lost committed entries.
            catch_up: CatchUp::new(None),
            catch_up_watch: BackgroundTasks {
                running: parking_lot::Mutex::new(Vec::new()),
            },
            membership_settle_timeout_ms: core::sync::atomic::AtomicU64::new(
                DEFAULT_MEMBERSHIP_SETTLE_TIMEOUT_MS,
            ),
            join: JoinTuning::default(),
            admission: None,
            connections: None,
            deployment,
            commands,
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
    ///
    /// The node proves the identity of `connections`' security context and
    /// admits nodes only on its listener.
    pub async fn open_cluster(
        node_id: u64,
        engine: Arc<StorageEngine>,
        listen_addr: std::net::SocketAddr,
        advertise_addr: String,
        connections: Arc<PeerConnections>,
    ) -> Result<Self, RaftNodeError> {
        Self::open_cluster_with_options(
            node_id,
            engine,
            listen_addr,
            advertise_addr,
            NodeOptions {
                connections: Some(connections),
                ..NodeOptions::default()
            },
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
        Self::open_member(
            node_id,
            engine,
            options,
            Founding::Form {
                advertise: advertise_addr,
            },
            Some(listen_addr),
        )
        .await
        .map(|(node, _)| node)
    }

    /// Open a member of a group with peers, under the security context in
    /// `options`: one that forms its group or resumes it, or one that joins
    /// one. With `listen`, the node serves its peers on a listener of its
    /// own that admits nodes only; without it, the caller registers the
    /// returned handler on a server of its own.
    async fn open_member(
        node_id: u64,
        engine: Arc<StorageEngine>,
        options: NodeOptions,
        founding: Founding,
        listen: Option<std::net::SocketAddr>,
    ) -> Result<(Self, RaftGrpcHandler), RaftNodeError> {
        let NodeOptions {
            snapshots: snap_config,
            timing,
            preference,
            host_epoch,
            group,
            connections,
            enrollment,
            deployment: forming_in,
        } = options;
        let connections = connections.ok_or(RaftNodeError::NoNodeIdentity)?;
        let own = connections.context().identity();
        let enrollment = match (&founding, enrollment) {
            (Founding::Join, Some(enrollment)) => Some(enrollment),
            (Founding::Join, None) => return Err(RaftNodeError::NotEnrolled),
            // A node forming its group is enrolled in nothing.
            (Founding::Form { .. }, _) => None,
        };
        let mut config = snap_config.raft_config(timing, preference);
        // Committed entries the disk lost are sent again by a leader when the
        // group has another voter to hold them; alone, nothing could.
        let lost = if has_other_voters(&engine, node_id)
            .map_err(|e| RaftNodeError::Init(e.to_string()))?
        {
            LostCommitted::CatchUp
        } else {
            LostCommitted::Refuse
        };
        let (log_store, lost_through) =
            LogStore::open_with(Arc::clone(&engine), lost).map_err(RaftNodeError::log_open)?;
        let catch_up = CatchUp::new(lost_through.map(|id| id.index));
        let elect = config.enable_elect;
        if catch_up.pending() {
            // Neither campaign nor take leadership back on restart: a leader
            // on this shorter log would overwrite the lost entries on every
            // member.
            config.enable_elect = false;
            config.enable_leader_restore = Some(false);
        }
        let config = Arc::new(config);
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
        if let Founding::Join = founding {
            refuse_join_with_local_data(node_id, &engine, &log_store, &state_machine)?;
        }
        if let Some(enrollment) = &enrollment {
            bind_enrollment(&engine, &state_machine, enrollment)?;
        }
        // The deployment the caller names is a statement of which one this
        // store is in; a store recorded in another is not served under it.
        if let (Some(named), Some(recorded)) =
            (forming_in, *state_machine.subscribe_deployment().borrow())
        {
            if named != recorded {
                return Err(RaftNodeError::OtherDeployment {
                    bound: recorded,
                    asked: named,
                });
            }
        }
        // A store enrolled earlier stays a member of the group it joined,
        // however it is opened now and whether or not that group's log has
        // reached it: it founds nothing, and speaks for the deployment it was
        // enrolled in until the deployment's own record arrives.
        let enrolled_in = enrolled_deployment(&engine)?;
        let version = version_gate(node_id, group, &state_machine, host_epoch);
        let mut deployment = state_machine.subscribe_deployment();
        let deployment_view = admission::DeploymentView::new(
            deployment.clone(),
            enrollment.as_ref().map(|e| e.deployment).or(enrolled_in),
        );
        let applied_rx = state_machine.subscribe_applied();
        let snapshot_builds = state_machine.snapshot_builds_handle();
        let engine_work = state_machine.engine_work_handle();

        let (closing, closing_rx) = tokio::sync::watch::channel(false);
        let book = Arc::new(network::MemberBook::default());
        let network = GrpcNetworkFactory {
            local_node_id: node_id,
            closing: network::Closing::new(closing_rx),
            gate: Arc::clone(&version),
            deployment: deployment_view.clone(),
            connections: Arc::clone(&connections),
            book: Arc::clone(&book),
        };

        let raft: RaftInstance =
            openraft::Raft::new(node_id, config, network, log_store, state_machine)
                .await
                .map_err(|e: openraft::error::Fatal<TypeConfig>| {
                    RaftNodeError::Init(e.to_string())
                })?;

        let raft = Arc::new(raft);
        let commands = Commands::new(Arc::clone(&raft), Arc::clone(&engine), engine.oracle());

        let advertise_addr = match &founding {
            // An enrolled member resumes in its group: nothing to initialize
            // or found, the group's leader brings it the log.
            Founding::Form { advertise } if enrolled_in.is_some() => Some(advertise.clone()),
            Founding::Form { advertise } => {
                // Try initialize — succeeds on fresh, NotAllowed on restart.
                // The member is bound to the identity it proves from the
                // start.
                let mut members = std::collections::BTreeMap::new();
                members.insert(
                    node_id,
                    MemberNode::new(advertise.clone(), Some(MemberIdentity::from(&own))),
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
                // A group of one identifies its deployment as it forms: a
                // node joining it is provisioned with the identity.
                let unrecorded = deployment.borrow().is_none();
                if unrecorded && await_membership(&raft).await? && only_voter(&raft, node_id) {
                    await_self_election(&raft).await?;
                    found_deployment(&commands, &mut deployment, forming_in).await?;
                }
                Some(advertise.clone())
            }
            Founding::Join => None,
        };

        let admission = Arc::new(Admission::new(
            node_id,
            Arc::clone(&raft),
            deployment_view,
            // Reopened without its provisioning, an enrolled member trusts
            // the authorities recorded with its enrollment.
            match enrollment {
                Some(enrollment) => enrollment.authorities,
                None => enrolled_authorities(&engine)?,
            },
            Arc::clone(&connections),
            Arc::clone(&version),
        ));
        let handler = RaftGrpcHandler::new(
            Arc::clone(&raft),
            crate::snapshot::snapshot_dir(&engine),
            Arc::clone(&version),
            Arc::clone(&admission),
            Arc::clone(engine.space()),
            catch_up.clone(),
        );

        // A listener of the node's own, when it has one. It is bound EAGERLY:
        // a busy port must fail this open instead of producing a node that
        // reports success while its server task dies unheard — deaf to the
        // cluster, green on every local check. Bound through tokio, which sets
        // SO_REUSEADDR: a node restarting on its own address must not be
        // locked out by its previous life's TIME_WAIT sockets.
        let (grpc_shutdown, grpc_task) = match listen {
            Some(listen_addr) => {
                let listener = tokio::net::TcpListener::bind(listen_addr)
                    .await
                    .map_err(|e| RaftNodeError::Init(format!("bind {listen_addr}: {e}")))?;
                let incoming = coordinode_wire::peer::incoming(
                    listener,
                    connections.context(),
                    coordinode_wire::peer::Accept {
                        inbound: coordinode_wire::peer::InboundAuth::NodeOnly,
                        plaintext_clients: false,
                        handshake_timeout: connections.timeouts().handshake,
                    },
                )
                .map_err(|e| RaftNodeError::Init(format!("listener TLS: {e}")))?;
                let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
                let server = tonic::transport::Server::builder()
                    .add_service(handler.handshake_service())
                    .add_service(RaftServiceServer::new(handler.clone()));
                let grpc_task = tokio::spawn(async move {
                    let graceful = server.serve_with_incoming_shutdown(incoming, async {
                        let _ = shutdown_rx.await;
                    });
                    if let Err(e) = graceful.await {
                        tracing::error!(%e, "raft gRPC server failed");
                    }
                });
                tracing::info!(node_id, %listen_addr, "raft gRPC server started");
                (Some(shutdown_tx), Some(grpc_task))
            }
            None => (None, None),
        };

        // Start background snapshot trigger (WAL size + periodic timer)
        let snap_handle = spawn_snapshot_trigger(
            Arc::clone(&raft),
            Arc::clone(&engine),
            &engine_work,
            snap_config,
            applied_rx.clone(),
        );
        let version_watch = spawn_version_watch(
            Arc::clone(&raft),
            Arc::clone(&version),
            Arc::clone(&engine),
            commands.clone(),
            Some(book),
        );
        let catch_up_watch = BackgroundTasks {
            running: parking_lot::Mutex::new(
                catch_up::spawn_lift(
                    Arc::clone(&raft),
                    Arc::clone(&engine),
                    engine_work.clone(),
                    catch_up.clone(),
                    elect,
                )
                .into_iter()
                .collect(),
            ),
        };

        if let Founding::Join = founding {
            tracing::info!(node_id, %own, "joining node started (waiting for leader to add it)");
        }

        let node = Self {
            raft,
            applied_rx,
            snapshot_builds,
            engine_work,
            node_id,
            advertise_addr,
            engine: crate::engine_hold::EngineHold::new(engine, "raft node"),
            oplog,
            append_notifier,
            grpc_shutdown: std::sync::Mutex::new(grpc_shutdown),
            grpc_task: std::sync::Mutex::new(grpc_task),
            closing,
            _snapshot_trigger: Some(snap_handle),
            version,
            version_watch,
            catch_up,
            catch_up_watch,
            membership_settle_timeout_ms: core::sync::atomic::AtomicU64::new(
                DEFAULT_MEMBERSHIP_SETTLE_TIMEOUT_MS,
            ),
            join: JoinTuning::default(),
            admission: Some(admission),
            connections: Some(connections),
            deployment,
            commands,
        };
        Ok((node, handler))
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
        connections: Arc<PeerConnections>,
    ) -> Result<(Self, RaftGrpcHandler), RaftNodeError> {
        Self::open_cluster_embedded_with_options(
            node_id,
            engine,
            advertise_addr,
            NodeOptions {
                connections: Some(connections),
                ..NodeOptions::default()
            },
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
        Self::open_member(
            node_id,
            engine,
            options,
            Founding::Form {
                advertise: advertise_addr,
            },
            None,
        )
        .await
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
        connections: Arc<PeerConnections>,
        enrollment: Enrollment,
    ) -> Result<(Self, RaftGrpcHandler), RaftNodeError> {
        Self::open_joining_embedded_with_options(
            node_id,
            engine,
            NodeOptions {
                connections: Some(connections),
                enrollment: Some(enrollment),
                ..NodeOptions::default()
            },
        )
        .await
    }

    /// Like `open_joining_embedded`, with the snapshot policy and host epoch
    /// of `options`.
    pub async fn open_joining_embedded_with_options(
        node_id: u64,
        engine: Arc<StorageEngine>,
        options: NodeOptions,
    ) -> Result<(Self, RaftGrpcHandler), RaftNodeError> {
        Self::open_member(node_id, engine, options, Founding::Join, None).await
    }

    /// Open a Raft node that joins an existing cluster.
    ///
    /// Does NOT call `initialize()` — the node waits for the leader to
    /// add it via `add_node()`. This avoids creating conflicting single-node
    /// clusters when bootstrapping a multi-node cluster.
    ///
    /// Used by nodes 2+ in the bootstrap sequence. The node proves the
    /// identity of `connections`' security context and admits a peer once
    /// a member it trusts (one of `enrollment`'s authorities, until it has
    /// a membership of its own) binds that peer.
    pub async fn open_joining(
        node_id: u64,
        engine: Arc<StorageEngine>,
        listen_addr: std::net::SocketAddr,
        connections: Arc<PeerConnections>,
        enrollment: Enrollment,
    ) -> Result<Self, RaftNodeError> {
        Self::open_joining_with_options(
            node_id,
            engine,
            listen_addr,
            NodeOptions {
                connections: Some(connections),
                enrollment: Some(enrollment),
                ..NodeOptions::default()
            },
        )
        .await
    }

    /// Like `open_joining`, with the snapshot policy and host epoch of
    /// `options`.
    pub async fn open_joining_with_options(
        node_id: u64,
        engine: Arc<StorageEngine>,
        listen_addr: std::net::SocketAddr,
        options: NodeOptions,
    ) -> Result<Self, RaftNodeError> {
        Self::open_member(node_id, engine, options, Founding::Join, Some(listen_addr))
            .await
            .map(|(node, _)| node)
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
    /// 4. On node 1: `add_node(2, addr2, id2)`, `add_node(3, addr3, id3)`
    /// 5. On node 1: `change_membership([1, 2, 3])` → all become voters
    ///
    /// The member is bound to `identity`: the leader dials `addr` expecting
    /// it, and every member admits the node under it alone. This is the
    /// explicit authorization a joining node's bootstrap rests on.
    ///
    /// # Errors
    ///
    /// `identity` already left the deployment (a returning node joins under
    /// a new one), is bound to another member, or `node_id` is a member bound
    /// to another identity.
    ///
    /// For a member already bound to `identity`, only its address changes:
    /// the address is where to find it, the identity who must answer there,
    /// so a member that moved is re-addressed with no other change.
    pub async fn add_node(
        &self,
        node_id: u64,
        addr: String,
        identity: coordinode_wire::peer::PeerIdentity,
    ) -> Result<(), RaftNodeError> {
        use openraft::rt::watch::WatchReceiver;

        self.publish_own_address().await?;
        self.membership_settled().await?;
        self.await_deployment().await?;

        let member = MemberIdentity::from(&identity);
        let retired = coordinode_storage::engine::metadata::identity_retired(
            &self.engine,
            &member.canonical(),
        )
        .map_err(|e| RaftNodeError::Init(format!("read retired identities: {e}")))?;
        if retired {
            return Err(RaftNodeError::Identity(format!(
                "{member} left this deployment and is not admitted again: a returning node joins \
                 under a new identity"
            )));
        }
        let mut readdress = false;
        {
            let metrics = self.raft.metrics();
            let metrics = metrics.borrow_watched();
            for (&id, node) in metrics.membership_config.membership().nodes() {
                let same = node.identity.as_ref() == Some(&member);
                if id != node_id && same {
                    return Err(RaftNodeError::Identity(format!(
                        "{member} is bound to node {id} already"
                    )));
                }
                if id == node_id && !same {
                    return Err(RaftNodeError::Identity(format!(
                        "node {node_id} is a member bound to {} already",
                        node.identity
                            .as_ref()
                            .map_or_else(|| "no identity".to_owned(), ToString::to_string)
                    )));
                }
                readdress |= id == node_id;
            }
        }

        let node_info = MemberNode::new(addr, Some(member));
        if readdress {
            let written = self
                .raft
                .change_membership(
                    openraft::ChangeMembers::SetNodes(std::collections::BTreeMap::from([(
                        node_id, node_info,
                    )])),
                    true,
                )
                .await
                .map_err(|e| RaftNodeError::Membership(Self::describe_membership_refusal(&e)))?;
            self.membership_published(written.log_id.index).await?;
            tracing::info!(node_id, "re-addressed a member");
            return Ok(());
        }

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

    /// Record this node's advertised address and node identity in the
    /// membership while it is still the cluster's only member.
    ///
    /// A directory first opened standalone holds a placeholder address and no
    /// identity for its single member, and reopening it in cluster mode
    /// resumes that state rather than initialising again. Peers dial members
    /// by this address expecting this identity, so both have to be right
    /// before the first peer joins. Once there are peers, a member that moved
    /// is re-addressed through [`Self::add_node`], and its identity stays.
    async fn publish_own_address(&self) -> Result<(), RaftNodeError> {
        use openraft::rt::watch::WatchReceiver;

        let (Some(advertise), Some(connections)) =
            (self.advertise_addr.as_ref(), self.connections.as_ref())
        else {
            return Ok(());
        };
        let own = MemberIdentity::from(&connections.context().identity());
        let stale = {
            let rx = self.raft.metrics();
            let metrics = rx.borrow_watched();
            let mut members = metrics.membership_config.membership().nodes();
            match (members.next(), members.next()) {
                (Some((id, info)), None) => {
                    *id == self.node_id
                        && (info.addr != *advertise || info.identity.as_ref() != Some(&own))
                }
                _ => false,
            }
        };
        if !stale {
            return Ok(());
        }

        let mut nodes = std::collections::BTreeMap::new();
        nodes.insert(self.node_id, MemberNode::new(advertise.clone(), Some(own)));
        let written = self
            .raft
            .change_membership(openraft::ChangeMembers::SetNodes(nodes), true)
            .await
            .map_err(|e| RaftNodeError::Membership(e.to_string()))?;
        self.membership_published(written.log_id.index).await?;

        tracing::info!(node_id = self.node_id, addr = %advertise, "published own address");
        Ok(())
    }

    /// The deployment's identity, recorded when its first group formed.
    async fn await_deployment(&self) -> Result<DeploymentId, RaftNodeError> {
        if let Some(id) = *self.deployment.borrow() {
            return Ok(id);
        }
        if leads_alone(&self.raft, self.node_id) {
            let mut recorded = self.deployment.clone();
            return found_deployment(&self.commands, &mut recorded, None).await;
        }
        Err(RaftNodeError::Membership(
            "this member does not know its deployment yet".to_owned(),
        ))
    }

    /// Change cluster membership (leader-only).
    ///
    /// Sets the voting members to the given set of node IDs.
    /// All nodes must have been previously added as learners. A voter left
    /// out of the set leaves the deployment: its identity is retired as the
    /// change applies.
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
        let (current_members, learner): (std::collections::BTreeSet<u64>, bool) = {
            let rx = self.raft.metrics();
            let metrics_ref = rx.borrow_watched();
            let membership = metrics_ref.membership_config.membership();
            let ids: Vec<u64> = membership.voter_ids().collect();
            let learner = membership.learner_ids().any(|id| id == node_id);
            (ids.into_iter().collect(), learner)
        };

        // A learner is not among the voters; it leaves the nodes instead.
        let change = if learner {
            openraft::ChangeMembers::RemoveNodes(std::collections::BTreeSet::from([node_id]))
        } else {
            let new_members: std::collections::BTreeSet<u64> = current_members
                .into_iter()
                .filter(|&id| id != node_id)
                .collect();
            if new_members.is_empty() {
                return Err(RaftNodeError::Membership(
                    "cannot remove last voting member".to_string(),
                ));
            }
            openraft::ChangeMembers::ReplaceAllVoters(new_members)
        };

        let written = self
            .raft
            .change_membership(change, false)
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
        .with_space_guard(Arc::clone(self.engine.space()))
        .with_closure(Arc::clone(self.engine.arc()))
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
        let metrics = self.raft.metrics().borrow_watched().clone();
        let voters: Vec<u64> = metrics.membership_config.membership().voter_ids().collect();
        // A leader hears every reachable voter; one is reachable as long as a
        // follower would keep its leader: heard from within an election
        // timeout. A follower hears its leader alone and judges no one's
        // reachability.
        let live_within = (metrics.current_leader == Some(self.node_id))
            .then(|| std::time::Duration::from_millis(self.raft.config().election_timeout_max));
        self.version.report(&voters, live_within)
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
        self.batch_pipeline_with_config(BatchConfig::default())
    }

    /// Create a [`WaitForMajorityService`] with custom batch configuration.
    pub fn batch_pipeline_with_config(&self, config: BatchConfig) -> WaitForMajorityService {
        WaitForMajorityService::spawn_stamping(
            Arc::clone(&self.raft),
            RateLimiter::default(),
            config,
            Some(Arc::clone(self.engine.arc())),
        )
        .with_space_guard(Arc::clone(self.engine.space()))
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

    /// Where this node's Raft log starts on disk, kept current by the log as
    /// its segments are purged or rewritten.
    pub fn retained_floor(&self) -> coordinode_storage::oplog::RetainedFloor {
        // A poisoned lock still guards a manager whose floor handle is sound:
        // the handle is only cloned here.
        self.oplog
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .retained_floor()
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

    /// Whether this member's disk lost committed log entries it has not
    /// received again; until it has, it neither votes nor campaigns.
    pub fn catching_up(&self) -> bool {
        self.catch_up.pending()
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

    /// How many members the live membership config names, voters and
    /// learners alike: every one of them receives the group's log.
    pub fn member_count(&self) -> usize {
        use openraft::async_runtime::watch::WatchReceiver;
        let rx = self.raft.metrics();
        let m = rx.borrow_watched();
        m.membership_config.membership().nodes().count()
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
            .with_version_gate(Arc::clone(&self.version))
    }

    /// Get this node's ID.
    pub fn node_id(&self) -> u64 {
        self.node_id
    }

    /// The deployment this member belongs to: the one its applied state
    /// records, or on a joining member the one it was enrolled in. `None`
    /// only before either is known.
    pub fn deployment(&self) -> Option<DeploymentId> {
        match &self.admission {
            Some(admission) => admission.deployment(),
            None => *self.deployment.borrow(),
        }
    }

    /// What a node joining this member's deployment through it enrolls
    /// with: the deployment, and this member, found at `addr`, as the one
    /// it learns the membership from. `None` until the deployment is known
    /// or for a node without peers.
    pub fn enrollment(&self, addr: impl Into<String>) -> Option<Enrollment> {
        Some(Enrollment {
            deployment: self.deployment()?,
            authorities: vec![PeerTarget {
                addr: addr.into(),
                identity: self.node_identity()?,
            }],
        })
    }

    /// The node identity this node proves to its peers; `None` for a node
    /// without peers.
    pub fn node_identity(&self) -> Option<coordinode_wire::peer::PeerIdentity> {
        self.connections.as_ref().map(|c| c.context().identity())
    }

    /// This node's connections to its peers. Their security context is
    /// where a host renews the node's credentials or replaces the trust its
    /// peers are verified under, without reopening the node.
    pub fn connections(&self) -> Option<&Arc<PeerConnections>> {
        self.connections.as_ref()
    }

    /// Which peers this member admits; `None` for a node without peers.
    /// A service a peer calls admits the caller through it.
    pub fn admission(&self) -> Option<&Arc<Admission>> {
        self.admission.as_ref()
    }

    /// The node identity member `node_id` is bound to, if the membership
    /// records it.
    pub fn member_identity(&self, node_id: u64) -> Option<MemberIdentity> {
        use openraft::async_runtime::watch::WatchReceiver;
        self.raft
            .metrics()
            .borrow_watched()
            .membership_config
            .membership()
            .get_node(&node_id)
            .and_then(|node| node.identity.clone())
    }

    /// Member `node_id` as a peer to dial: its address and the identity it
    /// must prove there. `None` when the membership does not contain it or
    /// records no identity for it.
    pub fn member_target(&self, node_id: u64) -> Option<PeerTarget> {
        use openraft::async_runtime::watch::WatchReceiver;
        self.raft
            .metrics()
            .borrow_watched()
            .membership_config
            .membership()
            .get_node(&node_id)
            .and_then(admission::member_target)
    }

    /// Wait until [`Self::member_target`] knows member `node_id`, at most
    /// `timeout`: woken by the membership this node applies, not by polling.
    /// Whether it does.
    pub async fn wait_for_member_target(&self, node_id: u64, timeout: std::time::Duration) -> bool {
        self.raft
            .wait(Some(timeout))
            .metrics(
                |m| {
                    m.membership_config
                        .membership()
                        .get_node(&node_id)
                        .and_then(admission::member_target)
                        .is_some()
                },
                "the member's address is known",
            )
            .await
            .is_ok()
    }

    /// Every other member as a peer to dial.
    pub fn peer_targets(&self) -> Vec<PeerTarget> {
        use openraft::async_runtime::watch::WatchReceiver;
        self.raft
            .metrics()
            .borrow_watched()
            .membership_config
            .membership()
            .nodes()
            .filter(|(&id, _)| id != self.node_id)
            .filter_map(|(_, node)| admission::member_target(node))
            .collect()
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

    /// A live voter, other than this leader, that holds every entry of the
    /// leader's log: one leadership can move to without a write in flight
    /// waiting on it. `None` when this node does not lead or no other voter
    /// has caught up.
    pub fn caught_up_voter(&self) -> Option<u64> {
        use openraft::async_runtime::watch::WatchReceiver;

        let target = self.find_transfer_target()?;
        let metrics = self.raft.metrics().borrow_watched().clone();
        let last = metrics.last_log_index?;
        let matched = metrics
            .replication
            .as_ref()?
            .get(&target)
            .and_then(Option::as_ref)
            .map(|log| log.index)?;
        (matched >= last).then_some(target)
    }

    /// While this member leads, hand leadership to a live voter as soon as one
    /// holds every entry of the log. A member that stands for election last
    /// runs this: it leads only when no other member could win, and gives the
    /// lead back once another can take it without losing a write. Woken by
    /// the engine's metrics, so it costs nothing while nothing changes.
    /// Returns once the node stops.
    pub async fn yield_leadership_when_caught_up(&self) {
        use openraft::async_runtime::watch::WatchReceiver;

        let mut rx = self.raft.metrics();
        loop {
            if let Some(target) = self.caught_up_voter() {
                tracing::info!(
                    node_id = self.node_id,
                    target,
                    "a member that stands last hands leadership back"
                );
                if let Err(e) = self.transfer_leadership_to(target).await {
                    tracing::warn!(target, %e, "handing leadership back failed");
                }
            }
            if rx.changed().await.is_err() {
                return;
            }
        }
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
        self.version_watch.stop().await;
        self.catch_up_watch.stop().await;
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
        self.await_caught_up(node_id, &progress_tx).await?;
        self.promote_caught_up(node_id, &progress_tx).await
    }

    /// Replace member `old` with a new member `new` under `identity` at
    /// `addr` (leader-only): the way back for a member that can no longer
    /// take part, such as one whose vote no copy proves
    /// ([`RaftNodeError::PromiseUnqualified`]). The new member is a node
    /// with its own, separate store; the old one's store is left as it was.
    ///
    /// Each step looks at the membership first, so a call that failed or was
    /// cut short is resumed by calling again with the same arguments, and a
    /// repeated call adds no second member:
    ///
    /// 1. `new` joins as a learner, unless it already has;
    /// 2. once it has caught up, one membership change puts `new` among the
    ///    voters and `old` out of the group, its identity retired, so its
    ///    messages are no longer admitted;
    /// 3. a call after both is done changes nothing.
    ///
    /// # Errors
    ///
    /// `new` equals `old` (a lost member is never let back under the identity
    /// it may have broken a promise under); the errors of [`Self::add_node`];
    /// [`RaftNodeError::Membership`] when this node does not lead, `new` does
    /// not catch up within [`Self::join_timeout`], or the change is refused.
    pub async fn replace_member(
        &self,
        old: u64,
        new: u64,
        addr: String,
        identity: coordinode_wire::peer::PeerIdentity,
        progress_tx: tokio::sync::broadcast::Sender<JoinProgressEvent>,
    ) -> Result<(), RaftNodeError> {
        use openraft::rt::watch::WatchReceiver;

        if old == new {
            return Err(RaftNodeError::Membership(format!(
                "member {old} is replaced under a new node id, never its own"
            )));
        }
        self.membership_settled().await?;
        let membership = |raft: &RaftInstance| {
            let rx = raft.metrics();
            let metrics = rx.borrow_watched();
            let config = metrics.membership_config.membership();
            let nodes: std::collections::BTreeSet<u64> =
                config.nodes().map(|(id, _)| *id).collect();
            let voters: std::collections::BTreeSet<u64> = config.voter_ids().collect();
            (nodes, voters)
        };
        let (nodes, voters) = membership(&self.raft);
        if voters.contains(&new) && !nodes.contains(&old) {
            return Ok(());
        }
        if !nodes.contains(&new) {
            self.add_node(new, addr, identity).await?;
        }
        if !voters.contains(&new) {
            self.await_caught_up(new, &progress_tx).await?;
        }
        let (_, voters) = membership(&self.raft);
        let target: Vec<u64> = voters
            .into_iter()
            .filter(|&id| id != old)
            .chain(std::iter::once(new))
            .collect();
        self.change_membership(target).await?;
        let (nodes, _) = membership(&self.raft);
        if nodes.contains(&old) {
            self.remove_node(old).await?;
        }
        let report = JoinProgressEvent {
            node_id: new,
            phase: JoinPhase::Complete,
            lag_entries: 0,
            percent: 100,
            message: format!("member {new} replaced member {old}"),
        };
        if progress_tx.send(report).is_err() {
            tracing::trace!(old, new, "no one follows the replacement's progress");
        }
        tracing::info!(old, new, "replaced a member");
        Ok(())
    }

    /// Wait until learner `node_id` has answered and lags at most
    /// [`Self::join_readiness_lag`] entries, reporting progress on
    /// `progress_tx`.
    ///
    /// # Errors
    ///
    /// [`RaftNodeError::Membership`] once [`Self::join_timeout`] passes, or
    /// when the consensus engine stops.
    async fn await_caught_up(
        &self,
        node_id: u64,
        progress_tx: &tokio::sync::broadcast::Sender<JoinProgressEvent>,
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
        Ok(())
    }

    /// Add the caught-up learner `node_id` to the voters, removing it from
    /// the membership again when the change fails.
    async fn promote_caught_up(
        &self,
        node_id: u64,
        progress_tx: &tokio::sync::broadcast::Sender<JoinProgressEvent>,
    ) -> Result<(), RaftNodeError> {
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

/// Spawn the background task that asks for snapshots.
///
/// It owns both thresholds of [`SnapshotTriggerConfig`]: the entries
/// applied and the bytes the log grew by since the last snapshot, probed
/// after an apply at most once a second, and asks for a snapshot no sooner
/// than the minimum interval after the last. Nothing fires while nothing
/// was applied since the last snapshot, and the task sleeps until
/// `applied_rx` moves.
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
        engine: crate::engine_hold::EngineHold::new(engine, "snapshot trigger"),
        _work: work.start(),
    };
    tokio::spawn(async move {
        use openraft::rt::watch::WatchReceiver;

        // Taken whole: a block that named only `held.engine` would capture
        // that field alone and drop the work guard at the spawn, so the
        // shutdown would not wait for this task.
        let held = held;
        let engine = &held.engine;

        let probe = config.min_interval.min(SNAPSHOT_SIZE_PROBE);
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
            let growth = LogGrowth {
                entries: applied - snapped,
                bytes: size - base,
            };
            let Some(reason) = snapshot_due(growth, last.elapsed(), &config) else {
                // Not due yet. A log grown enough waits out the minimum
                // interval; otherwise only the next apply can make it due.
                if growth.reaches(&config) {
                    tokio::time::sleep_until(last + config.min_interval).await;
                } else if applied_rx.changed().await.is_err() {
                    break;
                }
                continue;
            };
            // A snapshot is a full copy of the store written to disk; with the
            // disk below its reserve it waits, and the next probe asks again.
            // A hard limit does not stop it: the snapshot is what lets the
            // log be compacted, the room a full member gets back.
            if engine.space().is_paused_by_free_space() {
                tracing::debug!(reason, "snapshot trigger: waiting for disk space");
                continue;
            }

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
    engine: crate::engine_hold::EngineHold,
    _work: crate::storage::EngineWorkGuard,
}

/// How often the trigger task sizes the Raft log.
const SNAPSHOT_SIZE_PROBE: std::time::Duration = std::time::Duration::from_secs(1);

/// How far the log moved since the last snapshot.
#[derive(Debug, Clone, Copy)]
struct LogGrowth {
    /// Entries applied since it.
    entries: u64,
    /// Bytes the log's segments grew by since it was asked for.
    bytes: u64,
}

impl LogGrowth {
    /// Whether either threshold of `config` is reached.
    fn reaches(self, config: &SnapshotTriggerConfig) -> bool {
        self.entries >= config.logs_since_last || self.bytes >= config.log_bytes
    }
}

/// Why a snapshot is due, given how far the log moved and the time since
/// the last one was asked for: a threshold is reached and the minimum
/// interval has passed. `None` otherwise; time alone is no reason.
fn snapshot_due(
    growth: LogGrowth,
    since_last: std::time::Duration,
    config: &SnapshotTriggerConfig,
) -> Option<&'static str> {
    if since_last < config.min_interval {
        None
    } else if growth.bytes >= config.log_bytes {
        Some("log size")
    } else if growth.entries >= config.logs_since_last {
        Some("entries")
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

    /// A node with peers was opened without a node identity. The
    /// inter-node channel is authenticated or absent; a node that has no
    /// identity has no peers.
    #[error(
        "a node that talks to peers needs a node identity: give it its certificate, key and \
         the trust its peers are verified under"
    )]
    NoNodeIdentity,

    /// A joining node was opened without the deployment it joins and the
    /// members it trusts until it has a membership of its own.
    #[error(
        "a joining node needs its enrollment: the deployment it joins and at least one \
         existing member, by address and node identity"
    )]
    NotEnrolled,

    /// The store belongs to another deployment than the one it was asked to
    /// join.
    #[error("this store belongs to deployment {bound}, not to deployment {asked}")]
    OtherDeployment {
        /// The deployment the store is bound to.
        bound: DeploymentId,
        /// The deployment it was asked to join.
        asked: DeploymentId,
    },

    /// A member cannot be bound to this identity.
    #[error("{0}")]
    Identity(String),

    /// An existing member whose vote no copy can prove. It takes no part in
    /// its group: no campaign, no vote, no append acknowledged, since a vote
    /// it forgot could be granted twice in one term. Its store is left as it
    /// was. With a quorum, the group replaces it under a new identity in a
    /// separate directory through its membership; without one it stays
    /// unavailable.
    #[error("{0}")]
    PromiseUnqualified(crate::storage::PromiseUnqualified),
}

impl RaftNodeError {
    /// What opening the log of a member leads to: a promise no copy proves
    /// keeps its own variant, so a host can tell it from a failed open.
    fn log_open(err: std::io::Error) -> Self {
        if crate::storage::PromiseUnqualified::of(&err).is_none() {
            return Self::Init(err.to_string());
        }
        match err
            .into_inner()
            .map(|inner| inner.downcast::<crate::storage::PromiseUnqualified>())
        {
            Some(Ok(unqualified)) => Self::PromiseUnqualified(*unqualified),
            Some(Err(other)) => Self::Init(other.to_string()),
            None => Self::Init("the log could not be opened".to_owned()),
        }
    }
}

/// Refuse to join a group while this store holds data of its own.
///
/// What the store holds is what its trees hold plus what its log will replay
/// into them: after a crash an acknowledged write may live only in the log.
/// Both reads are cheap on the empty store a joining node is supposed to
/// have: one key at most per partition, and a log with nothing to replay.
///
/// A store that already belongs to a member of a group (its membership,
/// applied or still in its log, names `node_id` beside another node) is that
/// member restarting, not a node joining: its data is the group's.
fn refuse_join_with_local_data(
    node_id: u64,
    engine: &StorageEngine,
    log_store: &LogStore,
    state_machine: &CoordinodeStateMachine,
) -> Result<(), RaftNodeError> {
    fn init(e: impl std::fmt::Display) -> RaftNodeError {
        RaftNodeError::Init(e.to_string())
    }
    let from = state_machine.next_to_apply().map_err(init)?;
    if state_machine
        .applied_membership_names(node_id)
        .map_err(init)?
        || log_store.names_member_from(from, node_id).map_err(init)?
    {
        return Ok(());
    }
    if engine.holds_user_data().map_err(init)? {
        return Err(RaftNodeError::JoinWithLocalData);
    }
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
