//! A database that is a member of a replicated group.
//!
//! Compiled only with the `replication` feature. A replicated database is a
//! single-member group from its first open, so growing to several machines
//! is a membership change on the same directory: add a learner, promote it,
//! remove a member. The directory format is the server's; the same directory
//! opens under an embedding host or under the server binary.
//!
//! [`Member`] is the assembly both consumers share: the storage engine, the
//! consensus node, the database over its replicated pipeline, and the work a
//! member keeps doing in the background. [`ReplicatedDatabase`] is what an
//! embedding host opens: a member serving its peers on a listener of its own
//! (the only network surface: consensus, snapshot and segment transfer), with
//! membership, leadership and version on the handle. The server consumes
//! [`Member`] and serves the same inter-node services on its shared port.
//!
//! Every write commits through the leader. A member that is not the leader
//! refuses a write with [`DatabaseError::NotLeader`] naming the leader, and
//! during an election, when there is none, names none; it never applies a
//! caller's write locally, whatever the write concern.

pub mod admission;
mod background;
pub mod executor;
mod follower;
mod index_copy;
pub mod ingress;
pub mod routing;

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_raft::cluster::{GroupId, NodeOptions, RaftGrpcHandler, RaftNodeError};
use coordinode_replicate::SegmentInstaller;
use coordinode_storage::engine::config::StorageConfig;
use coordinode_storage::engine::core::StorageEngine;
use parking_lot::RwLock;

use crate::{Database, DatabaseError};

pub use coordinode_raft::cluster::version::VersionReport;
pub use coordinode_raft::cluster::{
    DeploymentId, ElectionPreference, ElectionTiming, Enrollment, RaftNode, SnapshotTriggerConfig,
    TimingError,
};
pub use coordinode_wire::peer::{PeerIdentity, PeerTarget, PeerTimeouts, SecurityContext};
pub use index_copy::segment_peers;

/// How often a leader looks at its AFTER COMMIT trigger queue at most.
pub const DEFAULT_TRIGGER_DISPATCH_INTERVAL: Duration = Duration::from_secs(1);

/// A failure to open or run a replicated database.
#[derive(Debug, thiserror::Error)]
pub enum ReplicationError {
    /// The store could not be opened.
    #[error("storage: {0}")]
    Storage(#[from] coordinode_storage::error::StorageError),
    /// The consensus node refused or failed: a store of another deployment,
    /// a joiner holding data of its own, no node identity where peers need
    /// one, a membership change without a quorum.
    #[error("consensus: {0}")]
    Consensus(#[from] RaftNodeError),
    /// The database over the store could not be built.
    #[error(transparent)]
    Database(#[from] DatabaseError),
    /// The inter-node listener could not be bound or configured.
    #[error("inter-node listener on {addr}: {reason}")]
    Listener {
        /// The address asked for.
        addr: SocketAddr,
        /// Why it failed.
        reason: String,
    },
}

/// Whether this member forms its group or joins an existing one.
pub enum Founding {
    /// The member the group is formed around, or one resuming its group:
    /// on a fresh store it is the group's only voter, and a store that
    /// already holds data becomes the group's base state. `deployment` names
    /// the deployment an existing deployment's group forms in; `None` founds
    /// a new one.
    Form {
        /// The deployment to form in, when it is not a new one.
        deployment: Option<DeploymentId>,
    },
    /// A member joining a group through its enrollment. A store holding data
    /// of its own is refused before anything is transferred: joining means
    /// receiving the group's history, and two pasts have no defined merge.
    Join(Enrollment),
}

/// How a member reaches its peers.
pub struct Peering {
    /// This database's security context: the node identity it proves and the
    /// trust its peers are verified under. Each database has its own; a host
    /// renews the credentials or replaces the trust on it without reopening.
    pub security: Arc<SecurityContext>,
    /// The address peers reach this member at.
    pub advertise: String,
    /// Whether the member forms its group or joins one.
    pub founding: Founding,
    /// Bounds on the inter-node channel.
    pub timeouts: PeerTimeouts,
}

/// How a member is opened.
pub struct MemberOptions {
    /// This member's id, unique within the deployment.
    pub node_id: u64,
    /// How it reaches its peers; `None` for a member alone, which needs no
    /// node certificate and grows once it is reopened with one.
    pub peering: Option<Peering>,
    /// The group's election and heartbeat timing.
    pub timing: ElectionTiming,
    /// When this member stands for election relative to the others.
    pub preference: ElectionPreference,
    /// The host's format epoch: members are matched on the engine's format
    /// version and on this, and a member that matches neither is read-only.
    /// Raise it when the format of what the host stores changes.
    pub host_epoch: u64,
    /// When the member snapshots its state so the log before it can go.
    pub snapshots: SnapshotTriggerConfig,
}

impl MemberOptions {
    /// Member `node_id` alone, with the rack timing and host epoch zero.
    pub fn new(node_id: u64) -> Self {
        Self {
            node_id,
            peering: None,
            timing: ElectionTiming::default(),
            preference: ElectionPreference::default(),
            host_epoch: 0,
            snapshots: SnapshotTriggerConfig::default(),
        }
    }
}

/// A database over a store that is a member of a replicated group: the
/// engine, the consensus node, the database over its replicated pipeline and
/// the member's background work.
///
/// Opening does not start the background work, so a caller can configure the
/// database first; [`Member::start_background`] starts it.
pub struct Member {
    database: Arc<RwLock<Database>>,
    node: Arc<RaftNode>,
    engine: Arc<StorageEngine>,
    handler: Option<RaftGrpcHandler>,
    preference: ElectionPreference,
    follower: Option<follower::IndexDefinitionFollower>,
    background: Vec<tokio::task::JoinHandle<()>>,
    /// Tells the background tasks to finish: each stops between passes, so
    /// none is cut off holding the database.
    stop: tokio::sync::watch::Sender<bool>,
}

impl Member {
    /// Open the store `storage` describes as a member, as `options` say.
    ///
    /// Must be called within a multi-threaded tokio runtime: consensus runs on
    /// it.
    ///
    /// # Errors
    ///
    /// [`ReplicationError`] naming what refused: the store, the consensus
    /// node (another deployment's store, a joiner holding data, a missing
    /// enrollment) or the database.
    pub async fn open(
        storage: &StorageConfig,
        options: MemberOptions,
    ) -> Result<Self, ReplicationError> {
        // A store the non-replicated build wrote keeps its history in a
        // journal where the consensus log is about to live: its entries are
        // brought into the store first, the cut the group starts from.
        StorageEngine::retire_embedded_journal(storage)?;
        let oracle = Arc::new(TimestampOracle::new());
        let engine = Arc::new(StorageEngine::open_with_oracle(
            storage,
            Arc::clone(&oracle),
        )?);
        let MemberOptions {
            node_id,
            peering,
            timing,
            preference,
            host_epoch,
            snapshots,
        } = options;
        let mut node_options = NodeOptions {
            snapshots,
            timing,
            preference,
            host_epoch,
            group: GroupId::FORMING,
            ..NodeOptions::default()
        };
        let (node, handler) = match peering {
            None => {
                let node = RaftNode::open_with_oracle_and_options(
                    node_id,
                    Arc::clone(&engine),
                    Some(Arc::clone(&oracle)),
                    node_options,
                )
                .await?;
                (node, None)
            }
            Some(peering) => {
                node_options.connections = Some(Arc::new(
                    coordinode_raft::cluster::PeerConnections::new(peering.security)
                        .with_timeouts(peering.timeouts),
                ));
                let (node, handler) = match peering.founding {
                    Founding::Form { deployment } => {
                        node_options.deployment = deployment;
                        RaftNode::open_cluster_embedded_with_options(
                            node_id,
                            Arc::clone(&engine),
                            peering.advertise,
                            node_options,
                        )
                        .await?
                    }
                    Founding::Join(enrollment) => {
                        node_options.enrollment = Some(enrollment);
                        RaftNode::open_joining_embedded_with_options(
                            node_id,
                            Arc::clone(&engine),
                            node_options,
                        )
                        .await?
                    }
                };
                (node, Some(handler))
            }
        };
        let node = Arc::new(node);
        let pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> =
            Arc::new(node.pipeline());
        let database =
            Database::from_engine(storage.data_dir(), Arc::clone(&engine), oracle, pipeline)?;
        // Before the database serves anything: a member that does not lead
        // opens no transaction, and one that leads opens them in its term.
        let (leader, term) = node.next_leadership_change((None, u64::MAX)).await;
        database.set_execution_tenure(node.node_id(), leader, term);
        // The idle reaper sleeps while no transaction is open; a begin wakes
        // it.
        let begun = Arc::new(tokio::sync::Notify::new());
        let mut database = database;
        {
            let begun = Arc::clone(&begun);
            database.set_interactive_begun_hook(Arc::new(move || begun.notify_one()));
        }
        // A copy of an index found damaged on this member alone is replaced
        // from a peer's copy; a member without peers answers from its records
        // instead.
        if node.connections().is_some() {
            database.set_index_copy_source(Arc::new(index_copy::PeerIndexCopy {
                installer: Arc::new(SegmentInstaller::new(Arc::clone(&engine))),
                node: Arc::clone(&node),
                runtime: tokio::runtime::Handle::current(),
            }));
        }
        let database = Arc::new(RwLock::new(database));
        let stop = tokio::sync::watch::Sender::new(false);
        // Not part of the optional background work: a member that is not
        // told when it stops leading would keep transactions of a lost lead.
        let tenure = background::spawn_execution_tenure(
            Arc::clone(&database),
            Arc::clone(&node),
            stop.subscribe(),
        );
        // Not optional either: an idle transaction ends with its timeout
        // outcome whether or not anything else ever begins.
        let reaper = background::spawn_idle_reaper(
            Arc::clone(&database),
            begun,
            background::REAPER_GRANULARITY,
            stop.subscribe(),
        );
        Ok(Self {
            database,
            node,
            engine,
            handler,
            preference,
            follower: None,
            background: vec![tenure, reaper],
            stop,
        })
    }

    /// Start the member's background work: bring index definitions live as
    /// their entries apply, and, while it leads, finish interrupted index
    /// builds and dispatch AFTER COMMIT triggers no more often than
    /// `trigger_dispatch_interval`. Starting it twice does nothing.
    pub fn start_background(&mut self, trigger_dispatch_interval: Duration) {
        if self.follower.is_some() {
            return;
        }
        self.follower = Some(follower::IndexDefinitionFollower::spawn(
            &self.engine,
            Arc::downgrade(&self.database),
        ));
        // With peers a body's writes go through the leader; a member alone
        // runs them as it commits.
        if self.handler.is_some() {
            self.background.push(background::spawn_trigger_dispatch(
                Arc::clone(&self.database),
                Arc::clone(&self.node),
                trigger_dispatch_interval,
                self.stop.subscribe(),
            ));
        }
        self.background.push(background::spawn_index_build_resumer(
            Arc::clone(&self.database),
            Arc::clone(&self.node),
            self.stop.subscribe(),
        ));
        // A member that stands last leads only while no other member can, and
        // gives the lead back once another has caught up.
        if self.preference == ElectionPreference::Last && self.handler.is_some() {
            let node = Arc::clone(&self.node);
            let mut stop = self.stop.subscribe();
            self.background.push(tokio::spawn(async move {
                tokio::select! {
                    () = node.yield_leadership_when_caught_up() => {}
                    () = background::stopped(&mut stop) => {}
                }
            }));
        }
    }

    /// Stop the background work and wait for it: each task finishes the pass
    /// it is in and lets go of the database, and the index definition
    /// follower is joined.
    pub async fn stop_background(&mut self) {
        self.stop.send_replace(true);
        for task in self.background.drain(..) {
            if let Err(e) = task.await {
                tracing::error!(%e, "a member background task failed");
            }
        }
        if let Some(follower) = self.follower.take() {
            // Joining its thread blocks: off the async runtime.
            if let Err(e) = tokio::task::spawn_blocking(move || drop(follower)).await {
                tracing::error!(%e, "stopping the index definition follower failed");
            }
        }
    }

    /// The database.
    pub fn database(&self) -> &Arc<RwLock<Database>> {
        &self.database
    }

    /// The consensus node.
    pub fn node(&self) -> &Arc<RaftNode> {
        &self.node
    }

    /// The storage engine.
    pub fn engine(&self) -> &Arc<StorageEngine> {
        &self.engine
    }

    /// Add the services peers call (the version handshake, consensus and
    /// segment transfer) to `routes`, each refusing a caller that is not an
    /// admitted node before it reads the message. Messages above
    /// `max_decoding_bytes` are refused. Returns `false`, adding nothing, for
    /// a member without peers.
    pub fn add_inter_node_services(
        &self,
        routes: &mut tonic::service::RoutesBuilder,
        max_decoding_bytes: usize,
    ) -> bool {
        use coordinode_raft::cluster::admission::nodes_only;
        use coordinode_raft::proto::replication::raft_service_server::RaftServiceServer;
        use coordinode_replicate::transfer::SegmentTransferHandler;
        use coordinode_replicate::transfer::proto::segment_transfer_service_server::SegmentTransferServiceServer;
        use tonic::service::interceptor::InterceptedService;

        let (Some(handler), Some(admission)) = (&self.handler, self.node.admission()) else {
            return false;
        };
        // The frozen exchange a member of another version is answered with.
        routes.add_service(InterceptedService::new(
            handler.handshake_service(),
            nodes_only,
        ));
        routes.add_service(InterceptedService::new(
            RaftServiceServer::new(handler.clone()),
            nodes_only,
        ));
        let segments = SegmentTransferHandler::new(
            Arc::new(SegmentInstaller::new(Arc::clone(&self.engine))),
            Arc::clone(admission) as _,
        );
        routes.add_service(InterceptedService::new(
            SegmentTransferServiceServer::new(segments)
                .max_decoding_message_size(max_decoding_bytes),
            nodes_only,
        ));
        true
    }

    /// Add the service peers route their clients' commands to, run here by
    /// `executor`, refusing a caller that is not an admitted node before it
    /// reads the message. Returns `false`, adding nothing, for a member
    /// without peers.
    pub fn add_command_routing<E: routing::CommandExecutor>(
        &self,
        routes: &mut tonic::service::RoutesBuilder,
        executor: Arc<E>,
    ) -> bool {
        use coordinode_raft::cluster::admission::nodes_only;
        use tonic::service::interceptor::InterceptedService;

        if self.handler.is_none() || self.node.admission().is_none() {
            return false;
        }
        routes.add_service(InterceptedService::new(
            routing::CommandRoutingHandler::new(executor).into_service(routing::MAX_ROUTED_MESSAGE),
            nodes_only,
        ));
        true
    }
}

impl Drop for Member {
    fn drop(&mut self) {
        // The tasks hold the node, so they would keep it alive and never see
        // it go: they are stopped here.
        for task in self.background.drain(..) {
            task.abort();
        }
    }
}

/// The largest inter-node message a [`ReplicatedDatabase`] accepts: a
/// segment piece and its framing.
const MAX_INTER_NODE_MESSAGE: usize = 64 << 20;

/// How long a stopping listener lets calls in flight finish.
const LISTENER_DRAIN: Duration = Duration::from_millis(500);

/// What changed in who leads: the leader, if one is known, and the term.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Leadership {
    /// The member that leads; `None` while an election is in progress.
    pub leader: Option<u64>,
    /// The term the leader leads in.
    pub term: u64,
}

/// A stream of leadership changes: each [`LeadershipChanges::next`] waits
/// until who leads, or in which term, differs from the last one returned.
pub struct LeadershipChanges {
    node: Arc<RaftNode>,
    seen: (Option<u64>, u64),
}

impl LeadershipChanges {
    /// The next change, or the current leadership once it differs from the
    /// last one returned. A member shutting down returns what it last knew.
    pub async fn next(&mut self) -> Leadership {
        self.seen = self.node.next_leadership_change(self.seen).await;
        Leadership {
            leader: self.seen.0,
            term: self.seen.1,
        }
    }
}

/// A replicated database an embedding host opens: a [`Member`] serving its
/// peers on a listener of its own, with membership, leadership and version on
/// the handle.
pub struct ReplicatedDatabase {
    member: Member,
    ingress: ingress::Ingress,
    timing: ElectionTiming,
    security: Option<Arc<SecurityContext>>,
    local_addr: Option<SocketAddr>,
    listener: Option<(
        tokio::sync::oneshot::Sender<()>,
        tokio::task::JoinHandle<()>,
    )>,
}

impl ReplicatedDatabase {
    /// Open the store `storage` describes as a member and, when it has peers,
    /// serve them on `listen`.
    ///
    /// The listener is the database's only network surface: node identity is
    /// required of every connection, and no client protocol is served. It is
    /// bound before this returns, so a busy port fails the open rather than
    /// leaving a member deaf to its group.
    ///
    /// # Errors
    ///
    /// As [`Member::open`]; a member with peers and no `listen`, or a listener
    /// that cannot be bound, is [`ReplicationError::Listener`].
    pub async fn open(
        storage: &StorageConfig,
        options: MemberOptions,
        listen: Option<SocketAddr>,
    ) -> Result<Self, ReplicationError> {
        let timing = options.timing;
        let security = options.peering.as_ref().map(|p| Arc::clone(&p.security));
        let timeouts = options.peering.as_ref().map(|p| p.timeouts);
        // Refused before the store is opened: nothing would serve the peers.
        let listen = match (&security, listen) {
            (Some(_), None) => {
                return Err(ReplicationError::Listener {
                    addr: SocketAddr::from(([0, 0, 0, 0], 0)),
                    reason: "a member with peers needs an address to serve them on".to_string(),
                });
            }
            (_, listen) => listen,
        };
        let mut member = Member::open(storage, options).await?;
        member.start_background(DEFAULT_TRIGGER_DISPATCH_INTERVAL);
        let admitter = admission::Admitter::new(Arc::clone(member.database()))
            .with_node(Arc::clone(member.node()));
        let ingress = ingress::Ingress::new(admitter.clone());

        let (mut local_addr, mut listener) = (None, None);
        if let (Some(security), Some(timeouts), Some(addr)) = (&security, timeouts, listen) {
            let bound = tokio::net::TcpListener::bind(addr).await.map_err(|e| {
                ReplicationError::Listener {
                    addr,
                    reason: e.to_string(),
                }
            })?;
            local_addr = bound.local_addr().ok();
            let incoming = coordinode_wire::peer::incoming(
                bound,
                security,
                coordinode_wire::peer::Accept {
                    inbound: coordinode_wire::peer::InboundAuth::NodeOnly,
                    plaintext_clients: false,
                    handshake_timeout: timeouts.handshake,
                },
            )
            .map_err(|e| ReplicationError::Listener {
                addr,
                reason: e.to_string(),
            })?;
            let mut routes = tonic::service::Routes::builder();
            member.add_inter_node_services(&mut routes, MAX_INTER_NODE_MESSAGE);
            // Commands its peers' hosts route here run as this member's own
            // host's calls do.
            member.add_command_routing(
                &mut routes,
                Arc::new(executor::MemberCommandExecutor::new(
                    admitter,
                    tokio::runtime::Handle::current(),
                )),
            );
            let router = tonic::transport::Server::builder().add_routes(routes.routes());
            let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
            let task = tokio::spawn(async move {
                let served = router
                    .serve_with_incoming_shutdown(incoming, async {
                        // Either outcome is the signal to stop: a value sent,
                        // or the handle holding the sender dropped.
                        if stopped.await.is_err() {
                            tracing::debug!("inter-node listener stopped with its handle");
                        }
                    })
                    .await;
                if let Err(e) = served {
                    tracing::error!(%e, "inter-node listener failed");
                }
            });
            listener = Some((stop, task));
        }
        Ok(Self {
            member,
            ingress,
            timing,
            security,
            local_addr,
            listener,
        })
    }

    /// Turn an open, non-replicated database into a member of a replicated
    /// group on the same store: the operating-mode transition a host requests
    /// while running, with no export and no reimport.
    ///
    /// Taking the handle is the fence: no statement is admitted on the old
    /// path while the store changes mode, and the handle's background work
    /// (volatile drain, index builds, index workers) is stopped and joined as
    /// it is dropped. The store's journal is then brought into the store, the
    /// cut the group starts from, and the store opens as the member a group is
    /// formed around. `storage` is the configuration `database` was opened
    /// with. A store that holds data cannot join another group, so
    /// `options` names a member that forms one.
    ///
    /// # Errors
    ///
    /// As [`ReplicatedDatabase::open`].
    pub async fn from_database(
        database: Database,
        storage: &StorageConfig,
        options: MemberOptions,
        listen: Option<SocketAddr>,
    ) -> Result<Self, ReplicationError> {
        // Dropping joins the handle's threads: off the async runtime.
        tokio::task::spawn_blocking(move || drop(database))
            .await
            .map_err(|e| {
                ReplicationError::Database(DatabaseError::Other(format!(
                    "closing the database before its transition: {e}"
                )))
            })?;
        Self::open(storage, options, listen).await
    }

    /// The database. Its calls run on this member only: a write on a member
    /// that does not lead is refused with [`DatabaseError::NotLeader`]. Use
    /// [`ReplicatedDatabase::execute`] or [`ReplicatedDatabase::ingress`] to
    /// have commands run wherever they must.
    pub fn database(&self) -> &Arc<RwLock<Database>> {
        self.member.database()
    }

    /// Where a host's commands enter the group through this member: each
    /// runs here when this member may run it and at the leader otherwise,
    /// transactions and operation records included.
    pub fn ingress(&self) -> &ingress::Ingress {
        &self.ingress
    }

    /// Run `statement` here or at the leader, as it needs.
    ///
    /// # Errors
    ///
    /// As [`ingress::Ingress::execute`].
    pub async fn execute(
        &self,
        statement: ingress::Statement,
    ) -> Result<ingress::Routed<routing::RoutedAnswer>, DatabaseError> {
        self.ingress.execute(statement).await
    }

    /// This member's id.
    pub fn node_id(&self) -> u64 {
        self.member.node.node_id()
    }

    /// The address the inter-node listener is bound to; `None` for a member
    /// without peers.
    pub fn local_addr(&self) -> Option<SocketAddr> {
        self.local_addr
    }

    /// The node identity this member proves; `None` for a member without
    /// peers.
    pub fn identity(&self) -> Option<PeerIdentity> {
        self.member.node.node_identity()
    }

    /// The security context this database's peers are authenticated under,
    /// where the host renews its credentials or replaces its trust without
    /// reopening; `None` for a member without peers.
    pub fn security(&self) -> Option<&Arc<SecurityContext>> {
        self.security.as_ref()
    }

    /// The election and heartbeat timing this member was opened with.
    pub fn timing(&self) -> ElectionTiming {
        self.timing
    }

    /// What a node joining this member's group through it enrolls with, this
    /// member being reachable at `addr`. `None` for a member without peers or
    /// before its deployment is recorded.
    pub fn enrollment(&self, addr: impl Into<String>) -> Option<Enrollment> {
        self.member.node.enrollment(addr)
    }

    /// The member that leads, if one is known.
    pub fn leader(&self) -> Option<u64> {
        self.member.node.current_leader()
    }

    /// Whether this member leads, confirmed by a quorum.
    pub async fn is_leader(&self) -> bool {
        self.member.node.is_leader().await
    }

    /// Leadership changes from now on.
    pub fn leadership(&self) -> LeadershipChanges {
        LeadershipChanges {
            node: Arc::clone(&self.member.node),
            // The first `next` returns the current leadership.
            seen: (None, u64::MAX),
        }
    }

    /// Hand leadership to member `target`. Returns once `target` leads and has
    /// committed in its term, so it serves writes; refused when it has not
    /// within the transfer timeout. A member that does not lead does nothing.
    ///
    /// # Errors
    ///
    /// [`ReplicationError::Consensus`] with the transfer's outcome.
    pub async fn transfer_leadership(&self, target: u64) -> Result<(), ReplicationError> {
        Ok(self.member.node.transfer_leadership_to(target).await?)
    }

    /// Add member `node_id`, reachable at `addr` and bound to `identity`, as
    /// a learner: it replicates the group's log without a vote.
    ///
    /// # Errors
    ///
    /// [`ReplicationError::Consensus`]: not the leader, an identity already
    /// bound or retired, no quorum to commit the change.
    pub async fn add_learner(
        &self,
        node_id: u64,
        addr: impl Into<String>,
        identity: PeerIdentity,
    ) -> Result<(), ReplicationError> {
        self.member
            .node
            .add_node(node_id, addr.into(), identity)
            .await?;
        Ok(())
    }

    /// Make learner `node_id` a voter once it has caught up.
    ///
    /// # Errors
    ///
    /// [`ReplicationError::Consensus`]: not the leader, not a learner, no
    /// quorum to commit the change.
    pub async fn promote(&self, node_id: u64) -> Result<(), ReplicationError> {
        self.member.node.promote_to_voter(node_id).await?;
        Ok(())
    }

    /// Make voter `node_id` a learner: it keeps replicating without a vote.
    ///
    /// # Errors
    ///
    /// [`ReplicationError::Consensus`]: not the leader, a change that would
    /// leave the group below its quorum floor.
    pub async fn demote(&self, node_id: u64) -> Result<(), ReplicationError> {
        self.member.node.demote_to_learner(node_id).await?;
        Ok(())
    }

    /// Remove member `node_id` from the group.
    ///
    /// # Errors
    ///
    /// [`ReplicationError::Consensus`]: not the leader, a change that would
    /// leave the group below its quorum floor.
    pub async fn remove(&self, node_id: u64) -> Result<(), ReplicationError> {
        self.member.node.remove_node(node_id).await?;
        Ok(())
    }

    /// This member's version report: its version pair, its group's, whether
    /// it serves writes and why not.
    pub fn version_report(&self) -> VersionReport {
        self.member.node.version_report()
    }

    /// The member assembly underneath, for what the handle does not cover.
    pub fn member(&self) -> &Member {
        &self.member
    }

    /// Stop the member: hand leadership over when it leads and other voters
    /// can take it, stop consensus, then the listener, then the background
    /// work, and close the store. Once this returns the store is released,
    /// so the host can reopen the same directory at once.
    ///
    /// # Errors
    ///
    /// [`ReplicationError::Consensus`] when consensus does not stop cleanly;
    /// the store is released either way.
    pub async fn shutdown(mut self) -> Result<(), ReplicationError> {
        let stopped = self.member.node.shutdown().await;
        if let Some((stop, mut task)) = self.listener.take() {
            // Dropping the sender stops the accept loop, but the server then
            // holds the listener until every open connection closes, and
            // peers keep idle channels open indefinitely. A short grace for
            // calls in flight, then the task is aborted and awaited, which
            // drops the listener (freeing the port) and the services it
            // routes to, which hold the store.
            drop(stop);
            if tokio::time::timeout(LISTENER_DRAIN, &mut task)
                .await
                .is_err()
            {
                task.abort();
                match task.await {
                    Err(e) if !e.is_cancelled() => {
                        tracing::error!(%e, "the inter-node listener failed while stopping");
                    }
                    _ => {}
                }
            }
        }
        self.member.stop_background().await;
        // The last handles on the store: closing the database joins its
        // threads, which blocks, so it runs off the async runtime.
        if let Err(e) = tokio::task::spawn_blocking(move || drop(self)).await {
            tracing::error!(%e, "closing the database failed");
        }
        Ok(stopped?)
    }
}

impl Drop for ReplicatedDatabase {
    fn drop(&mut self) {
        if let Some((stop, task)) = self.listener.take() {
            drop(stop);
            task.abort();
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
