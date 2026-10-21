//! Which peers a member admits, to which of its services.
//!
//! A connection proves a node identity; admission decides what that
//! identity may do here. It must belong to this deployment, name the member
//! the membership binds it to, and hold the role the operation needs: only
//! a voter campaigns, leads or hands leadership over, while any member may
//! ask who the members are or move segments. Every check reads in-memory
//! state: the membership as this member's consensus last published it, and
//! the deployment as its applied state records it.
//!
//! A member that missed a membership change, or has none yet because it is
//! joining, cannot place a peer it was never told about. It then asks a
//! member it already knows (one of its voters, or for a joining member one
//! of the authorities it was provisioned with) who the members are, and
//! admits the peer once a newer membership from that member binds it. No
//! one becomes a member by asking: the asker is answered only when the
//! answering member already counts it as one.

use std::sync::Arc;
use std::time::{Duration, Instant};

use coordinode_core::group::{DeploymentId, GroupId};
use coordinode_wire::peer::{PeerConnectInfo, PeerConnections, PeerTarget};
use openraft::LogIdOptionExt;
use openraft::rt::watch::WatchReceiver;
use serde::{Deserialize, Serialize};
use tonic::metadata::MetadataMap;
use tonic::{Code, Status};
use tonic_types::{ErrorDetails, StatusExt};

use super::version::VersionGate;
use crate::member::{MemberIdentity, MemberNode};
use crate::storage::{CoordinodeStateMachine, TypeConfig};

type RaftInstance = openraft::Raft<TypeConfig, CoordinodeStateMachine>;

/// The request metadata key carrying the caller's deployment on every
/// inter-node call.
pub const DEPLOYMENT_METADATA_KEY: &str = "cn-deployment-bin";

/// `ErrorInfo.reason`: the connection proved no node identity, or one that
/// no longer verifies.
pub const NOT_A_NODE: &str = "NOT_A_NODE";
/// `ErrorInfo.reason`: the caller belongs to another deployment, or named
/// none.
pub const OTHER_DEPLOYMENT: &str = "OTHER_DEPLOYMENT";
/// `ErrorInfo.reason`: the caller's identity is bound to no member here.
pub const NOT_A_MEMBER: &str = "NOT_A_MEMBER";
/// `ErrorInfo.reason`: the call names another member than the one the
/// caller's identity is bound to.
pub const WRONG_MEMBER: &str = "WRONG_MEMBER";
/// `ErrorInfo.reason`: the caller's member lacks the role the operation
/// needs.
pub const ROLE: &str = "ROLE";

/// How a joining member is provisioned: the deployment it joins and the
/// members it trusts to tell it who the others are until it has a
/// membership of its own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Enrollment {
    /// The deployment being joined.
    pub deployment: DeploymentId,
    /// Existing members, each dialed expecting its identity.
    pub authorities: Vec<PeerTarget>,
}

/// What a peer asks of a member.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Operation {
    /// A candidate's vote request.
    Vote,
    /// A leader's entries or heartbeat.
    Append,
    /// A leader's snapshot.
    Snapshot,
    /// A leader handing leadership over.
    TransferLeader,
    /// The version exchange.
    Exchange,
    /// Who the members are.
    Members,
    /// Another peer service of the member: segment transfer, write routing.
    Service,
}

impl Operation {
    /// Whether only a voter may ask it: a learner never campaigns or leads.
    fn needs_voter(self) -> bool {
        matches!(
            self,
            Self::Vote | Self::Append | Self::Snapshot | Self::TransferLeader
        )
    }
}

/// One member of a membership, as a member answering who the members are
/// describes it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ViewMember {
    pub(crate) node_id: u64,
    pub(crate) node: MemberNode,
    pub(crate) voter: bool,
}

/// A membership and the log index it was written at.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub(crate) struct MemberView {
    pub(crate) index: Option<u64>,
    pub(crate) members: Vec<ViewMember>,
}

impl MemberView {
    fn find(&self, identity: &MemberIdentity) -> Option<(u64, bool)> {
        self.members
            .iter()
            .find(|m| m.node.identity.as_ref() == Some(identity))
            .map(|m| (m.node_id, m.voter))
    }
}

/// The answer to who the members are.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct MembersReply {
    pub(crate) deployment: Option<DeploymentId>,
    pub(crate) view: MemberView,
}

/// The deployment a member belongs to: the one its applied state records,
/// or before that the one it was provisioned to join.
#[derive(Clone)]
pub(crate) struct DeploymentView {
    recorded: tokio::sync::watch::Receiver<Option<DeploymentId>>,
    enrolled: Option<DeploymentId>,
}

impl DeploymentView {
    pub(crate) fn new(
        recorded: tokio::sync::watch::Receiver<Option<DeploymentId>>,
        enrolled: Option<DeploymentId>,
    ) -> Self {
        Self { recorded, enrolled }
    }

    pub(crate) fn get(&self) -> Option<DeploymentId> {
        let recorded = *self.recorded.borrow();
        recorded.or(self.enrolled)
    }
}

/// How often a member asks who the members are, at most.
const LEARN_INTERVAL: Duration = Duration::from_secs(1);

/// How long one ask waits for its answer.
const LEARN_TIMEOUT: Duration = Duration::from_secs(5);

/// The newest membership another member told this one about, and whether
/// an ask is under way.
#[derive(Default)]
struct Learning {
    attested: Option<Arc<MemberView>>,
    asking: bool,
    asked_at: Option<Instant>,
}

/// One replica's admission of peers.
pub struct Admission {
    node_id: u64,
    group: GroupId,
    raft: Arc<RaftInstance>,
    deployment: DeploymentView,
    /// The members a joining member was provisioned to trust.
    authorities: Vec<PeerTarget>,
    connections: Arc<PeerConnections>,
    gate: Arc<VersionGate>,
    // no-std: spin::RwLock; read once per inter-node call.
    learning: Arc<parking_lot::RwLock<Learning>>,
}

impl Admission {
    /// The admission of member `node_id` of the group `gate` speaks for.
    pub(crate) fn new(
        node_id: u64,
        raft: Arc<RaftInstance>,
        deployment: DeploymentView,
        authorities: Vec<PeerTarget>,
        connections: Arc<PeerConnections>,
        gate: Arc<VersionGate>,
    ) -> Self {
        Self {
            node_id,
            group: gate.group(),
            raft,
            deployment,
            authorities,
            connections,
            gate,
            learning: Arc::default(),
        }
    }

    /// The deployment this member belongs to: the one its applied state
    /// records, or the one it is joining.
    pub fn deployment(&self) -> Option<DeploymentId> {
        self.deployment.get()
    }

    /// Admit the caller of `op` on the connection `info` describes; the
    /// member it is admitted as.
    pub(crate) fn admit(
        &self,
        info: Option<&PeerConnectInfo>,
        metadata: &MetadataMap,
        op: Operation,
    ) -> Result<u64, Status> {
        let identity = caller_identity(info)?;
        let ours = self.deployment().ok_or_else(|| {
            refuse(
                Code::FailedPrecondition,
                OTHER_DEPLOYMENT,
                "this member does not know its deployment yet".to_owned(),
            )
        })?;
        let theirs = read_deployment(metadata);
        if theirs != Some(ours) {
            return Err(refuse(
                Code::PermissionDenied,
                OTHER_DEPLOYMENT,
                format!(
                    "the caller belongs to deployment {}, this member to {ours}",
                    theirs.map_or_else(|| "none".to_owned(), |d| d.to_string())
                ),
            ));
        }
        self.bind(&identity, op)
    }

    /// Admit again the caller of an established stream, whose deployment
    /// was checked when it opened: its identity must still verify and still
    /// be bound to a member that `op` is open to.
    pub(crate) fn readmit(
        &self,
        info: Option<&PeerConnectInfo>,
        op: Operation,
    ) -> Result<u64, Status> {
        let identity = caller_identity(info)?;
        self.bind(&identity, op)
    }

    /// The member `identity` is bound to, holding the role `op` needs.
    fn bind(&self, identity: &MemberIdentity, op: Operation) -> Result<u64, Status> {
        let Some((node_id, voter)) = self.lookup(identity) else {
            self.learn();
            return Err(refuse(
                Code::PermissionDenied,
                NOT_A_MEMBER,
                format!(
                    "{identity} is no member of {} as far as node {} knows",
                    self.group, self.node_id
                ),
            ));
        };
        if op.needs_voter() && !voter {
            return Err(refuse(
                Code::PermissionDenied,
                ROLE,
                format!("node {node_id} is no voter, and only a voter may ask that"),
            ));
        }
        Ok(node_id)
    }

    /// Refuse a call naming `claimed` as its sender when the caller is
    /// `bound`.
    pub(crate) fn same_member(bound: u64, claimed: u64) -> Result<(), Status> {
        if bound == claimed {
            return Ok(());
        }
        Err(refuse(
            Code::PermissionDenied,
            WRONG_MEMBER,
            format!("the caller is node {bound} and speaks for node {claimed}"),
        ))
    }

    /// The member `identity` is bound to and whether it votes, by the newer
    /// of this member's membership and the last one another member told it.
    fn lookup(&self, identity: &MemberIdentity) -> Option<(u64, bool)> {
        let (index, own) = {
            let metrics = self.raft.metrics();
            let metrics = metrics.borrow_watched();
            let stored = &metrics.membership_config;
            let found = stored
                .membership()
                .nodes()
                .find(|(_, node)| node.identity.as_ref() == Some(identity))
                .map(|(&id, _)| {
                    let voter = stored
                        .membership()
                        .get_joint_config()
                        .iter()
                        .any(|config| config.contains(&id));
                    (id, voter)
                });
            (stored.log_id().index(), found)
        };
        match self.learning.read().attested.as_ref() {
            Some(attested) if attested.index > index => attested.find(identity),
            _ => own,
        }
    }

    /// This member's own membership.
    pub(crate) fn own_view(&self) -> MemberView {
        let metrics = self.raft.metrics();
        let metrics = metrics.borrow_watched();
        let stored = &metrics.membership_config;
        let joint = stored.membership().get_joint_config();
        MemberView {
            index: stored.log_id().index(),
            members: stored
                .membership()
                .nodes()
                .map(|(&node_id, node)| ViewMember {
                    node_id,
                    node: node.clone(),
                    voter: joint.iter().any(|config| config.contains(&node_id)),
                })
                .collect(),
        }
    }

    /// Ask a known member who the members are, unless an ask is under way
    /// or one was made within the last [`LEARN_INTERVAL`].
    fn learn(&self) {
        {
            let mut learning = self.learning.write();
            if learning.asking
                || learning
                    .asked_at
                    .is_some_and(|at| at.elapsed() < LEARN_INTERVAL)
            {
                return;
            }
            learning.asking = true;
            learning.asked_at = Some(Instant::now());
        }
        let own = self.own_view();
        let sources = self.sources(&own);
        let (Some(deployment), Ok(runtime)) =
            (self.deployment(), tokio::runtime::Handle::try_current())
        else {
            self.learning.write().asking = false;
            return;
        };
        let ask = Ask {
            group: self.group,
            deployment,
            known: own.index,
            connections: Arc::clone(&self.connections),
            gate: Arc::clone(&self.gate),
        };
        let learning = Arc::clone(&self.learning);
        runtime.spawn(async move {
            let answer = ask.ask(&sources).await;
            let mut learning = learning.write();
            learning.asking = false;
            if let Some(view) = answer {
                let newer = learning
                    .attested
                    .as_ref()
                    .is_none_or(|known| view.index > known.index);
                if newer {
                    tracing::info!(
                        group = %ask.group,
                        index = ?view.index,
                        members = view.members.len(),
                        "learnt the membership from a known member"
                    );
                    learning.attested = Some(Arc::new(view));
                }
            }
        });
    }

    /// Whom to ask: this member's other voters, or, before it has any, the
    /// authorities it was provisioned with.
    fn sources(&self, own: &MemberView) -> Vec<PeerTarget> {
        let voters: Vec<PeerTarget> = own
            .members
            .iter()
            .filter(|m| m.voter && m.node_id != self.node_id)
            .filter_map(|m| member_target(&m.node))
            .collect();
        if !voters.is_empty() {
            return voters;
        }
        self.authorities.clone()
    }
}

impl coordinode_wire::peer::PeerAdmission for Admission {
    fn admit_peer(
        &self,
        info: Option<&PeerConnectInfo>,
        metadata: &MetadataMap,
    ) -> Result<u64, Status> {
        self.admit(info, metadata, Operation::Service)
    }

    fn stamp(&self, metadata: &mut MetadataMap) {
        if let Some(deployment) = self.deployment() {
            write_deployment(metadata, deployment);
        }
    }
}

/// The peer a member is reached as, `None` for one that records no
/// identity or one that does not parse.
pub(crate) fn member_target(node: &MemberNode) -> Option<PeerTarget> {
    let identity = node.identity.as_ref()?.to_peer().ok()?;
    Some(PeerTarget {
        addr: node.addr.clone(),
        identity,
    })
}

/// One ask for the membership.
struct Ask {
    group: GroupId,
    deployment: DeploymentId,
    known: Option<u64>,
    connections: Arc<PeerConnections>,
    gate: Arc<VersionGate>,
}

impl Ask {
    /// The first newer membership of this deployment a source tells.
    async fn ask(&self, sources: &[PeerTarget]) -> Option<MemberView> {
        for source in sources {
            match tokio::time::timeout(LEARN_TIMEOUT, self.ask_one(source)).await {
                Ok(Ok(view)) if view.index > self.known => return Some(view),
                Ok(Ok(_)) => {}
                Ok(Err(e)) => {
                    tracing::debug!(peer = %source.identity, error = %e, "asking for the membership");
                }
                Err(_) => {
                    tracing::debug!(peer = %source.identity, "asking for the membership timed out")
                }
            }
        }
        None
    }

    async fn ask_one(&self, source: &PeerTarget) -> Result<MemberView, String> {
        use crate::proto::replication::RaftPayload;
        use crate::proto::replication::raft_service_client::RaftServiceClient;

        let channel = self
            .connections
            .channel(source)
            .map_err(|e| e.to_string())?;
        let mut request = tonic::Request::new(RaftPayload {
            data: Vec::new(),
            group: self.group.raw(),
        });
        super::version::write_handshake(request.metadata_mut(), &self.gate.local_handshake());
        write_deployment(request.metadata_mut(), self.deployment);
        let reply = RaftServiceClient::new(channel)
            .members(request)
            .await
            .map_err(|status| status.to_string())?
            .into_inner();
        let reply: MembersReply = rmp_serde::from_slice(&reply.data).map_err(|e| e.to_string())?;
        if reply.deployment != Some(self.deployment) {
            return Err("the member belongs to another deployment".to_owned());
        }
        Ok(reply.view)
    }
}

/// Refuse, before its message is read, a call whose connection proves no
/// node identity: an inter-node service served on a port it shares with
/// clients is wrapped in this (`InterceptedService`), so a client's bytes
/// are never decoded or decompressed by a service that serves only nodes.
/// Which member the node is, and what it may do, is decided afterwards.
///
/// # Errors
///
/// The connection is a client's, or its node identity no longer verifies.
pub fn nodes_only(request: tonic::Request<()>) -> Result<tonic::Request<()>, Status> {
    node_admitted(request.extensions().get::<PeerConnectInfo>())?;
    Ok(request)
}

/// Refuse unless the connection `info` describes still proves a node
/// identity, as of now: the check a long-lived stream repeats before each
/// command it carries, since the one made when it opened does not cover a
/// credential that expired or was withdrawn since.
///
/// # Errors
///
/// The connection is a client's, or its node identity no longer verifies.
pub fn node_admitted(info: Option<&PeerConnectInfo>) -> Result<(), Status> {
    caller_identity(info).map(|_| ())
}

/// The node identity the connection proved, as of now.
fn caller_identity(info: Option<&PeerConnectInfo>) -> Result<MemberIdentity, Status> {
    let peer = info.and_then(|i| i.node.as_ref()).ok_or_else(|| {
        refuse(
            Code::Unauthenticated,
            NOT_A_NODE,
            "the connection proved no node identity".to_owned(),
        )
    })?;
    let identity = peer.identity().map_err(|e| {
        refuse(
            Code::Unauthenticated,
            NOT_A_NODE,
            format!("the node identity no longer verifies: {e}"),
        )
    })?;
    Ok(MemberIdentity::from(&identity))
}

/// The deployment a call names.
pub(crate) fn read_deployment(metadata: &MetadataMap) -> Option<DeploymentId> {
    let value = metadata.get_bin(DEPLOYMENT_METADATA_KEY)?;
    DeploymentId::from_slice(&value.to_bytes().ok()?)
}

/// Name `deployment` in a call's metadata.
pub(crate) fn write_deployment(metadata: &mut MetadataMap, deployment: DeploymentId) {
    metadata.insert_bin(
        DEPLOYMENT_METADATA_KEY,
        tonic::metadata::MetadataValue::from_bytes(deployment.as_bytes()),
    );
}

/// A refusal carrying `reason` in its `ErrorInfo`.
fn refuse(code: Code, reason: &str, message: String) -> Status {
    metrics::counter!("coordinode_peer_calls_refused_total", "reason" => reason.to_owned())
        .increment(1);
    let details = ErrorDetails::with_error_info(
        reason,
        coordinode_core::ERROR_DOMAIN,
        std::collections::HashMap::<String, String>::new(),
    );
    Status::with_error_details(code, message, details)
}

#[cfg(test)]
mod tests;
