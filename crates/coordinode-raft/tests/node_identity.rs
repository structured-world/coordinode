#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! The common peer contract, qualified on real groups under both identity
//! profiles: the built-in SPIFFE profile and an embedding host's adapter.
//!
//! Members prove node identities on every connection, are bound to them in
//! the membership, and are admitted only as the member and role they are
//! bound to, in their own deployment. A peer that is unknown, speaks for
//! another member, belongs to another deployment, lost its trust, let its
//! certificate expire, or left the group is refused; a member that renews
//! its credentials, or moves to another address, keeps working without a
//! restart.

use std::sync::Arc;
use std::time::{Duration, SystemTime};

use coordinode_core::txn::proposal::{
    Mutation, PartitionId, ProposalIdGenerator, ProposalPipeline, RaftProposal,
};
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_raft::cluster::admission::{NOT_A_MEMBER, OTHER_DEPLOYMENT, ROLE, WRONG_MEMBER};
use coordinode_raft::cluster::version::write_handshake;
use coordinode_raft::cluster::{
    DEPLOYMENT_METADATA_KEY, DeploymentId, Enrollment, NodeOptions, PeerConnections, RaftNode,
    RaftNodeError,
};
use coordinode_raft::proto::replication::RaftPayload;
use coordinode_raft::proto::replication::raft_service_client::RaftServiceClient;
use coordinode_raft::storage::TypeConfig;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_test_fixtures::pki::UriAdapter;
use coordinode_test_fixtures::{TestPki, alloc_port};
use coordinode_wire::peer::{PeerIdentity, PeerProfile, PeerTarget, SecurityContext};
use futures_util::StreamExt;
use openraft::rt::watch::WatchReceiver;
use tonic_types::StatusExt;

const TEST_TIMEOUT: Duration = Duration::from_secs(90);

/// How long a write is given to reach a member that should have it.
const ARRIVES: Duration = Duration::from_secs(15);

/// How long a member that must not get a write is watched for it.
const STAYS_AWAY: Duration = Duration::from_secs(3);

// ── Profiles ───────────────────────────────────────────────────────

/// Which source establishes node identities.
enum Profile {
    /// X.509-SVIDs, the SPIFFE ID compared whole.
    Spiffe,
    /// An embedding host's adapter, under its issuer scope.
    Host(Arc<UriAdapter>),
}

/// A test's CA and profile: what every member proves its identity under.
struct Fixture {
    pki: TestPki,
    profile: Profile,
}

impl Fixture {
    fn spiffe() -> Self {
        Self {
            pki: TestPki::new(),
            profile: Profile::Spiffe,
        }
    }

    fn host() -> Self {
        Self::host_in("cluster.test", "host-a")
    }

    /// A SPIFFE fixture whose CA issues identities in `trust_domain`.
    fn spiffe_in(trust_domain: &str) -> Self {
        Self {
            pki: TestPki::with_trust_domain(trust_domain),
            profile: Profile::Spiffe,
        }
    }

    /// A host fixture whose CA issues in `trust_domain` and whose adapter
    /// identifies nodes in issuer scope `scope`.
    fn host_in(trust_domain: &str, scope: &str) -> Self {
        Self {
            pki: TestPki::with_trust_domain(trust_domain),
            profile: Profile::Host(UriAdapter::new(scope)),
        }
    }

    /// The profile peers are verified under, trusting this fixture's CA and
    /// those of `others`.
    fn profile_with(&self, others: &[&TestPki]) -> PeerProfile {
        match &self.profile {
            Profile::Spiffe => self.pki.profile_with(others),
            Profile::Host(adapter) => self.pki.host_profile_with(Arc::clone(adapter), others),
        }
    }

    /// A security context for node `id`, its certificate valid until
    /// `not_after` when given, trusting this fixture's CA and those of
    /// `others`.
    fn context(
        &self,
        id: u64,
        not_after: Option<SystemTime>,
        others: &[&TestPki],
    ) -> Arc<SecurityContext> {
        let (chain, key) = self.pki.credentials(&self.pki.spiffe_id(id), not_after);
        Arc::new(
            SecurityContext::new(
                self.pki.crypto_provider(),
                chain,
                key,
                self.profile_with(others),
            )
            .expect("context"),
        )
    }

    /// The connections of node `id`, under a fresh certificate.
    fn connections(&self, id: u64) -> Arc<PeerConnections> {
        match &self.profile {
            Profile::Spiffe => self.pki.connections(id),
            Profile::Host(adapter) => self.pki.host_connections(id, adapter),
        }
    }

    /// The identity node `id` is bound to.
    fn identity(&self, id: u64) -> PeerIdentity {
        match &self.profile {
            Profile::Spiffe => self.pki.identity(id),
            Profile::Host(adapter) => self.pki.host_identity(id, adapter),
        }
    }

    /// Node `id`, found on `port`.
    fn target(&self, id: u64, port: u16) -> PeerTarget {
        PeerTarget {
            addr: format!("127.0.0.1:{port}"),
            identity: self.identity(id),
        }
    }
}

// ── Members ────────────────────────────────────────────────────────

struct Member {
    id: u64,
    node: RaftNode,
    engine: Arc<StorageEngine>,
    port: u16,
    _dir: tempfile::TempDir,
}

fn open_engine(dir: &tempfile::TempDir) -> Arc<StorageEngine> {
    Arc::new(
        StorageEngine::open(&StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            dir.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]))
        .expect("open engine"),
    )
}

/// Node `id` forming a group, with `connections`.
async fn found(id: u64, connections: Arc<PeerConnections>) -> Member {
    let dir = tempfile::tempdir().expect("dir");
    let engine = open_engine(&dir);
    let port = alloc_port();
    let node = RaftNode::open_cluster(
        id,
        Arc::clone(&engine),
        format!("127.0.0.1:{port}").parse().expect("addr"),
        format!("127.0.0.1:{port}"),
        connections,
    )
    .await
    .expect("open the founder");
    await_leadership(&node).await;
    Member {
        id,
        node,
        engine,
        port,
        _dir: dir,
    }
}

/// Node `id` joining the group `founder` formed, with `connections`,
/// trusting `authority` to tell it who the members are.
async fn join(
    id: u64,
    connections: Arc<PeerConnections>,
    founder: &Member,
    authority: PeerTarget,
) -> Member {
    let dir = tempfile::tempdir().expect("dir");
    let engine = open_engine(&dir);
    let port = alloc_port();
    let node = RaftNode::open_joining(
        id,
        Arc::clone(&engine),
        format!("127.0.0.1:{port}").parse().expect("addr"),
        connections,
        Enrollment {
            deployment: founder.node.deployment().expect("a deployment"),
            authorities: vec![authority],
        },
    )
    .await
    .expect("open a joining node");
    Member {
        id,
        node,
        engine,
        port,
        _dir: dir,
    }
}

/// A group of `size` voters under `fx`, node 1 leading.
async fn group(fx: &Fixture, size: u64) -> Vec<Member> {
    let founder = found(1, fx.connections(1)).await;
    let mut members = vec![];
    for id in 2..=size {
        members.push(join(id, fx.connections(id), &founder, fx.target(1, founder.port)).await);
    }
    for m in &members {
        founder
            .node
            .add_node(m.id, format!("127.0.0.1:{}", m.port), fx.identity(m.id))
            .await
            .expect("add a member");
    }
    founder
        .node
        .change_membership((1..=size).collect())
        .await
        .expect("make the members voters");
    members.insert(0, founder);
    members
}

async fn await_leadership(node: &RaftNode) {
    for _ in 0..150 {
        if node.is_leader().await {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("node {} never led", node.node_id());
}

/// Commit `key` through `leader`.
fn write(leader: &RaftNode, ids: &ProposalIdGenerator, key: &str, ts: u64) {
    leader
        .pipeline()
        .propose_and_wait(&RaftProposal {
            id: ids.next(),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: key.as_bytes().to_vec(),
                value: b"v".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(ts),
            start_ts: Timestamp::from_raw(ts - 1),
            bypass_rate_limiter: false,
        })
        .expect("commit");
}

/// Whether `key` reaches `engine` within `within`.
async fn reaches(engine: &StorageEngine, key: &str, within: Duration) -> bool {
    let deadline = tokio::time::Instant::now() + within;
    while tokio::time::Instant::now() < deadline {
        if engine
            .get(Partition::Node, key.as_bytes())
            .expect("read")
            .is_some()
        {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    false
}

// ── Direct calls ───────────────────────────────────────────────────

/// A client of `member`, dialing as `connections` expecting `member`'s
/// identity under `fx`.
fn client(
    fx: &Fixture,
    connections: &PeerConnections,
    member: &Member,
) -> RaftServiceClient<tonic::transport::Channel> {
    let channel = connections
        .channel(&fx.target(member.id, member.port))
        .expect("channel");
    RaftServiceClient::new(channel)
}

/// `body` as a call to `member` speaking for node `caller`, naming
/// `deployment`.
fn call<T>(
    member: &Member,
    caller: u64,
    deployment: Option<DeploymentId>,
    body: T,
) -> tonic::Request<T> {
    let mut request = tonic::Request::new(body);
    let mut record = member.node.version().local_handshake();
    record.node_id = caller;
    write_handshake(request.metadata_mut(), &record);
    if let Some(deployment) = deployment {
        request.metadata_mut().insert_bin(
            DEPLOYMENT_METADATA_KEY,
            tonic::metadata::MetadataValue::from_bytes(deployment.as_bytes()),
        );
    }
    request
}

fn vote_for(candidate: u64) -> RaftPayload {
    let request = openraft::raft::VoteRequest::<TypeConfig> {
        vote: openraft::type_config::alias::VoteOf::<TypeConfig>::new(1, candidate),
        last_log_id: None,
        leadership_transfer: false,
    };
    RaftPayload {
        data: rmp_serde::to_vec(&request).expect("encode"),
        group: 0,
    }
}

fn members_query() -> RaftPayload {
    RaftPayload {
        data: Vec::new(),
        group: 0,
    }
}

/// The `ErrorInfo.reason` of a refusal.
fn reason(status: &tonic::Status) -> String {
    status
        .get_details_error_info()
        .map(|info| info.reason)
        .unwrap_or_default()
}

fn assert_refused(outcome: Result<impl std::fmt::Debug, tonic::Status>, why: &str) {
    let status = outcome.expect_err("refused");
    assert_eq!(reason(&status), why, "{status:?}");
}

// ── The matrix ─────────────────────────────────────────────────────

/// A group forms and replicates, every member admitted under its bound
/// identity; a joining member is bootstrapped by the authority it was
/// enrolled with.
async fn members_replicate(fx: Fixture) {
    let g = group(&fx, 3).await;
    let ids = ProposalIdGenerator::with_base(1 << 48);
    write(&g[0].node, &ids, "node:1:a", 100);
    for m in &g[1..] {
        assert!(
            reaches(&m.engine, "node:1:a", ARRIVES).await,
            "node {}",
            m.id
        );
        assert_eq!(
            m.node.deployment(),
            g[0].node.deployment(),
            "one deployment"
        );
    }
    for m in g {
        m.node.shutdown().await.expect("shutdown");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn members_replicate_under_spiffe() {
    tokio::time::timeout(TEST_TIMEOUT, members_replicate(Fixture::spiffe()))
        .await
        .expect("in time");
}

#[tokio::test(flavor = "multi_thread")]
async fn members_replicate_under_a_host_adapter() {
    tokio::time::timeout(TEST_TIMEOUT, members_replicate(Fixture::host()))
        .await
        .expect("in time");
}

/// A node of the same CA whose identity no member is bound to is refused
/// on every peer service, however it names itself.
async fn an_unknown_node_is_refused(fx: Fixture) {
    let g = group(&fx, 2).await;
    let outsider = fx.connections(9);
    let deployment = g[0].node.deployment();
    let mut c = client(&fx, &outsider, &g[1]);
    assert_refused(
        c.vote(call(&g[1], 9, deployment, vote_for(9))).await,
        NOT_A_MEMBER,
    );
    assert_refused(
        c.members(call(&g[1], 9, deployment, members_query())).await,
        NOT_A_MEMBER,
    );
    // Speaking for a member changes nothing: the identity is what counts.
    assert_refused(
        c.vote(call(&g[1], 1, deployment, vote_for(1))).await,
        NOT_A_MEMBER,
    );
    for m in g {
        m.node.shutdown().await.expect("shutdown");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn an_unknown_node_is_refused_under_spiffe() {
    tokio::time::timeout(TEST_TIMEOUT, an_unknown_node_is_refused(Fixture::spiffe()))
        .await
        .expect("in time");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_unknown_node_is_refused_under_a_host_adapter() {
    tokio::time::timeout(TEST_TIMEOUT, an_unknown_node_is_refused(Fixture::host()))
        .await
        .expect("in time");
}

/// A member's own identity, presented under another deployment, or with
/// none, is refused: the same CA and the same node ids in two deployments
/// grant nothing across them.
async fn another_deployment_is_refused_under_a_shared_ca(fx: Fixture) {
    let g = group(&fx, 2).await;
    // A second deployment, its node 1 the same identity as ours.
    let other = found(1, fx.connections(1)).await;
    assert_ne!(other.node.deployment(), g[0].node.deployment());
    let as_member = fx.connections(1);
    let mut c = client(&fx, &as_member, &g[1]);
    assert_refused(
        c.members(call(&g[1], 1, other.node.deployment(), members_query()))
            .await,
        OTHER_DEPLOYMENT,
    );
    assert_refused(
        c.members(call(&g[1], 1, None, members_query())).await,
        OTHER_DEPLOYMENT,
    );
    // The second deployment's writes never reach the first's members.
    let ids = ProposalIdGenerator::with_base(2 << 48);
    write(&other.node, &ids, "node:1:elsewhere", 100);
    assert!(!reaches(&g[1].engine, "node:1:elsewhere", STAYS_AWAY).await);
    other.node.shutdown().await.expect("shutdown");
    for m in g {
        m.node.shutdown().await.expect("shutdown");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn another_deployment_is_refused_under_spiffe() {
    tokio::time::timeout(
        TEST_TIMEOUT,
        another_deployment_is_refused_under_a_shared_ca(Fixture::spiffe()),
    )
    .await
    .expect("in time");
}

#[tokio::test(flavor = "multi_thread")]
async fn another_deployment_is_refused_under_a_host_adapter() {
    tokio::time::timeout(
        TEST_TIMEOUT,
        another_deployment_is_refused_under_a_shared_ca(Fixture::host()),
    )
    .await
    .expect("in time");
}

/// A member speaks only for itself, and only a voter campaigns: a call
/// naming another member as its sender, or a vote request from a learner,
/// is refused.
async fn roles_and_senders_are_enforced(fx: Fixture) {
    let founder = found(1, fx.connections(1)).await;
    let learner = join(2, fx.connections(2), &founder, fx.target(1, founder.port)).await;
    founder
        .node
        .add_node(2, format!("127.0.0.1:{}", learner.port), fx.identity(2))
        .await
        .expect("add the learner");
    let deployment = founder.node.deployment();

    let as_learner = fx.connections(2);
    let mut c = client(&fx, &as_learner, &founder);
    // A learner may ask who the members are.
    c.members(call(&founder, 2, deployment, members_query()))
        .await
        .expect("a learner may ask who the members are");
    // It may not campaign.
    assert_refused(
        c.vote(call(&founder, 2, deployment, vote_for(2))).await,
        ROLE,
    );
    // Its version record may not speak for another member.
    assert_refused(
        c.members(call(&founder, 1, deployment, members_query()))
            .await,
        WRONG_MEMBER,
    );

    // A voter campaigning in another member's name is refused.
    founder
        .node
        .change_membership(vec![1, 2])
        .await
        .expect("two voters");
    let mut c = client(&fx, &as_learner, &founder);
    assert_refused(
        c.vote(call(&founder, 2, deployment, vote_for(1))).await,
        WRONG_MEMBER,
    );
    learner.node.shutdown().await.expect("shutdown");
    founder.node.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread")]
async fn roles_and_senders_are_enforced_under_spiffe() {
    tokio::time::timeout(
        TEST_TIMEOUT,
        roles_and_senders_are_enforced(Fixture::spiffe()),
    )
    .await
    .expect("in time");
}

#[tokio::test(flavor = "multi_thread")]
async fn roles_and_senders_are_enforced_under_a_host_adapter() {
    tokio::time::timeout(
        TEST_TIMEOUT,
        roles_and_senders_are_enforced(Fixture::host()),
    )
    .await
    .expect("in time");
}

/// The leader dials a member expecting the identity the membership binds
/// it to: a node proving another identity at that address gets nothing,
/// though it is trusted.
async fn an_unexpected_peer_gets_nothing(fx: Fixture) {
    let founder = found(1, fx.connections(1)).await;
    let node2 = join(2, fx.connections(2), &founder, fx.target(1, founder.port)).await;
    // Node 2 listens there, but the membership expects node 7.
    founder
        .node
        .add_node(7, format!("127.0.0.1:{}", node2.port), fx.identity(7))
        .await
        .expect("add a member at node 2's address");
    let ids = ProposalIdGenerator::with_base(3 << 48);
    write(&founder.node, &ids, "node:1:for-7", 100);
    assert!(
        !reaches(&node2.engine, "node:1:for-7", STAYS_AWAY).await,
        "a peer proving another identity than expected received the log"
    );
    node2.node.shutdown().await.expect("shutdown");
    founder.node.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_unexpected_peer_gets_nothing_under_spiffe() {
    tokio::time::timeout(
        TEST_TIMEOUT,
        an_unexpected_peer_gets_nothing(Fixture::spiffe()),
    )
    .await
    .expect("in time");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_unexpected_peer_gets_nothing_under_a_host_adapter() {
    tokio::time::timeout(
        TEST_TIMEOUT,
        an_unexpected_peer_gets_nothing(Fixture::host()),
    )
    .await
    .expect("in time");
}

/// A member removed from the group is refused on a stream it had open and
/// on a new call, its identity is never admitted again, and it returns
/// under a new identity.
async fn a_removed_member_is_refused(fx: Fixture, renamed: PeerIdentity) {
    let g = group(&fx, 3).await;
    let deployment = g[0].node.deployment();

    // Node 3 leads, and a stream speaking for it is open to node 2.
    g[0].node
        .transfer_leadership_to(3)
        .await
        .expect("node 3 leads");
    let vote = g[2].node.raft().metrics().borrow_watched().vote;
    let heartbeat = || {
        let request = openraft::raft::AppendEntriesRequest::<TypeConfig> {
            vote,
            prev_log_id: None,
            entries: Vec::new(),
            leader_commit: None,
        };
        RaftPayload {
            data: rmp_serde::to_vec(&request).expect("encode"),
            group: 0,
        }
    };
    let as_three = fx.connections(3);
    let (tx, rx) = tokio::sync::mpsc::channel(4);
    tx.send(heartbeat()).await.expect("send");
    let requests = futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|request| (request, rx))
    });
    let mut replies = client(&fx, &as_three, &g[1])
        .stream_append(call(&g[1], 3, deployment, requests))
        .await
        .expect("the stream opens while node 3 is a voter")
        .into_inner();
    replies
        .next()
        .await
        .expect("a reply")
        .expect("the heartbeat is taken");

    // Leadership moves on and node 3 is removed.
    g[2].node
        .transfer_leadership_to(1)
        .await
        .expect("node 1 leads again");
    g[0].node.remove_node(3).await.expect("remove node 3");
    let deadline = tokio::time::Instant::now() + ARRIVES;
    while g[1].node.member_identity(3).is_some() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "node 2 never saw the removal"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // The open stream takes nothing more from it.
    tx.send(heartbeat()).await.expect("send");
    let mut last = None;
    while let Some(reply) = replies.next().await {
        if reply.is_err() {
            last = Some(reply);
            break;
        }
    }
    let status = last.expect("the stream ends").expect_err("with a refusal");
    assert_eq!(reason(&status), NOT_A_MEMBER, "{status:?}");

    // A new call is refused too.
    let mut c = client(&fx, &as_three, &g[1]);
    assert_refused(
        c.members(call(&g[1], 3, deployment, members_query())).await,
        NOT_A_MEMBER,
    );

    // Its identity is never admitted again; under a new one it returns.
    let port = alloc_port();
    assert!(matches!(
        g[0].node
            .add_node(3, format!("127.0.0.1:{port}"), fx.identity(3))
            .await,
        Err(RaftNodeError::Identity(_))
    ));
    g[0].node
        .add_node(3, format!("127.0.0.1:{port}"), renamed)
        .await
        .expect("a returning member under a new identity");

    for m in g {
        m.node.shutdown().await.expect("shutdown");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_removed_member_is_refused_under_spiffe() {
    let fx = Fixture::spiffe();
    let renamed = TestPki::identity_of(&format!("{}-again", fx.pki.spiffe_id(3)));
    tokio::time::timeout(TEST_TIMEOUT, a_removed_member_is_refused(fx, renamed))
        .await
        .expect("in time");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_removed_member_is_refused_under_a_host_adapter() {
    let fx = Fixture::host();
    let renamed = PeerIdentity::Host(coordinode_wire::peer::HostIdentity {
        scope: "host-a".to_owned(),
        subject: format!("{}-again", fx.pki.spiffe_id(3)),
    });
    tokio::time::timeout(TEST_TIMEOUT, a_removed_member_is_refused(fx, renamed))
        .await
        .expect("in time");
}

/// A joining node bound to a deployment stays bound: reopening it for
/// another deployment is refused before it hears anyone, and reopened for
/// its own it completes the join.
async fn an_interrupted_enrollment_resumes_into_its_own_deployment(fx: Fixture) {
    let founder = found(1, fx.connections(1)).await;
    let dir = tempfile::tempdir().expect("dir");
    let port = alloc_port();
    let enrollment = Enrollment {
        deployment: founder.node.deployment().expect("a deployment"),
        authorities: vec![fx.target(1, founder.port)],
    };
    let open = |enrollment: Enrollment| {
        let engine = open_engine(&dir);
        let connections = fx.connections(2);
        async move {
            let node = RaftNode::open_joining(
                2,
                Arc::clone(&engine),
                format!("127.0.0.1:{port}").parse().expect("addr"),
                connections,
                enrollment,
            )
            .await;
            (node, engine)
        }
    };
    // Interrupted before the leader added it.
    let (first, engine) = open(enrollment.clone()).await;
    first.expect("enrolled").shutdown().await.expect("shutdown");
    drop(engine);

    let (elsewhere, engine) = open(Enrollment {
        deployment: DeploymentId([9; 16]),
        ..enrollment.clone()
    })
    .await;
    assert!(
        matches!(elsewhere, Err(RaftNodeError::OtherDeployment { .. })),
        "a node bound to one deployment is not enrolled in another"
    );
    drop(engine);

    let (resumed, engine) = open(enrollment).await;
    let resumed = resumed.expect("resumes into its own deployment");
    founder
        .node
        .add_node(2, format!("127.0.0.1:{port}"), fx.identity(2))
        .await
        .expect("add the node");
    let ids = ProposalIdGenerator::with_base(4 << 48);
    write(&founder.node, &ids, "node:1:joined", 100);
    assert!(reaches(&engine, "node:1:joined", ARRIVES).await);
    resumed.shutdown().await.expect("shutdown");
    founder.node.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_interrupted_enrollment_resumes_under_spiffe() {
    tokio::time::timeout(
        TEST_TIMEOUT,
        an_interrupted_enrollment_resumes_into_its_own_deployment(Fixture::spiffe()),
    )
    .await
    .expect("in time");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_interrupted_enrollment_resumes_under_a_host_adapter() {
    tokio::time::timeout(
        TEST_TIMEOUT,
        an_interrupted_enrollment_resumes_into_its_own_deployment(Fixture::host()),
    )
    .await
    .expect("in time");
}

/// A joining node needs a node identity and an enrollment.
#[tokio::test(flavor = "multi_thread")]
async fn a_member_with_peers_needs_an_identity_and_a_joiner_its_enrollment() {
    let dir = tempfile::tempdir().expect("dir");
    let engine = open_engine(&dir);
    let without_identity = RaftNode::open_cluster_with_options(
        1,
        Arc::clone(&engine),
        format!("127.0.0.1:{}", alloc_port()).parse().expect("addr"),
        "127.0.0.1:1".to_owned(),
        NodeOptions::default(),
    )
    .await;
    assert!(matches!(
        without_identity,
        Err(RaftNodeError::NoNodeIdentity)
    ));
    let without_enrollment = RaftNode::open_joining_with_options(
        2,
        engine,
        format!("127.0.0.1:{}", alloc_port()).parse().expect("addr"),
        NodeOptions {
            connections: Some(Fixture::spiffe().connections(2)),
            ..NodeOptions::default()
        },
    )
    .await;
    assert!(matches!(
        without_enrollment,
        Err(RaftNodeError::NotEnrolled)
    ));
}

/// Credentials renewed while the group runs take effect without a restart:
/// a member whose certificate expired stops being admitted, by peers that
/// had already accepted it too, and once it renews it is back.
async fn an_expired_member_is_cut_off_and_renewal_needs_no_restart(fx: Fixture) {
    let founder = found(1, fx.connections(1)).await;
    // Node 2's certificate lasts a few seconds past the join.
    let expiry = SystemTime::now() + Duration::from_secs(10);
    let context = fx.context(2, Some(expiry), &[]);
    let connections = Arc::new(PeerConnections::new(Arc::clone(&context)));
    let node2 = join(2, connections, &founder, fx.target(1, founder.port)).await;
    founder
        .node
        .add_node(2, format!("127.0.0.1:{}", node2.port), fx.identity(2))
        .await
        .expect("add node 2");
    let ids = ProposalIdGenerator::with_base(5 << 48);
    write(&founder.node, &ids, "node:1:before", 100);
    assert!(reaches(&node2.engine, "node:1:before", ARRIVES).await);

    tokio::time::sleep(
        expiry.duration_since(SystemTime::now()).unwrap_or_default() + Duration::from_secs(1),
    )
    .await;
    write(&founder.node, &ids, "node:1:expired", 200);
    assert!(
        !reaches(&node2.engine, "node:1:expired", STAYS_AWAY).await,
        "a member whose certificate expired is still reached"
    );

    let (chain, key) = fx.pki.node_credentials(2);
    context
        .update_credentials(chain, key)
        .expect("a renewal keeping the identity");
    assert!(
        reaches(&node2.engine, "node:1:expired", ARRIVES).await,
        "the renewed member catches up without a restart"
    );
    node2.node.shutdown().await.expect("shutdown");
    founder.node.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_expired_member_is_cut_off_under_spiffe() {
    tokio::time::timeout(
        TEST_TIMEOUT,
        an_expired_member_is_cut_off_and_renewal_needs_no_restart(Fixture::spiffe()),
    )
    .await
    .expect("in time");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_expired_member_is_cut_off_under_a_host_adapter() {
    tokio::time::timeout(
        TEST_TIMEOUT,
        an_expired_member_is_cut_off_and_renewal_needs_no_restart(Fixture::host()),
    )
    .await
    .expect("in time");
}

/// A credential or trust replacement that is not valid as a whole is
/// refused and the node keeps what it had, its group still replicating.
async fn an_invalid_reload_keeps_the_last_valid_configuration(fx: Fixture) {
    let g = group(&fx, 2).await;
    let context = g[1].node.connections().expect("peers").context();
    let generation = context.generation();

    let (chain, _) = fx.pki.node_credentials(2);
    let (_, other_key) = fx.pki.node_credentials(2);
    assert!(context.update_credentials(chain, other_key).is_err());
    let (moved, key) = fx.pki.node_credentials(5);
    assert!(context.update_credentials(moved, key).is_err());
    // Trust under which the node's own certificate is not valid.
    let stranger = TestPki::with_trust_domain("elsewhere.test");
    let foreign = match &fx.profile {
        Profile::Spiffe => stranger.profile(),
        Profile::Host(adapter) => stranger.host_profile(Arc::clone(adapter)),
    };
    assert!(context.update_trust(foreign).is_err());
    assert_eq!(context.generation(), generation, "nothing was published");
    assert_eq!(context.identity(), fx.identity(2));

    let ids = ProposalIdGenerator::with_base(6 << 48);
    write(&g[0].node, &ids, "node:1:after", 100);
    assert!(reaches(&g[1].engine, "node:1:after", ARRIVES).await);
    for m in g {
        m.node.shutdown().await.expect("shutdown");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn an_invalid_reload_keeps_the_last_valid_configuration_under_spiffe() {
    tokio::time::timeout(
        TEST_TIMEOUT,
        an_invalid_reload_keeps_the_last_valid_configuration(Fixture::spiffe()),
    )
    .await
    .expect("in time");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_invalid_reload_keeps_the_last_valid_configuration_under_a_host_adapter() {
    tokio::time::timeout(
        TEST_TIMEOUT,
        an_invalid_reload_keeps_the_last_valid_configuration(Fixture::host()),
    )
    .await
    .expect("in time");
}

/// Trust withdrawn from a member's trust domain cuts it off on the
/// connections already made, not only on new ones; trust restored brings
/// it back, all without a restart.
///
/// `a` and `b` are two issuers under one profile: two SPIFFE trust domains,
/// or two CAs one embedding host's adapter accepts.
async fn withdrawn_trust_cuts_a_member_off(a: Fixture, b: Fixture) {
    let c1 = a.context(1, None, &[&b.pki]);
    let founder = found(1, Arc::new(PeerConnections::new(Arc::clone(&c1)))).await;
    let node2 = join(
        2,
        Arc::new(PeerConnections::new(b.context(2, None, &[&a.pki]))),
        &founder,
        a.target(1, founder.port),
    )
    .await;
    founder
        .node
        .add_node(2, format!("127.0.0.1:{}", node2.port), b.identity(2))
        .await
        .expect("add the member of the second issuer");
    let ids = ProposalIdGenerator::with_base(7 << 48);
    write(&founder.node, &ids, "node:1:trusted", 100);
    assert!(reaches(&node2.engine, "node:1:trusted", ARRIVES).await);

    c1.update_trust(a.profile_with(&[]))
        .expect("withdraw the second issuer");
    write(&founder.node, &ids, "node:1:withdrawn", 200);
    assert!(
        !reaches(&node2.engine, "node:1:withdrawn", STAYS_AWAY).await,
        "a member of a withdrawn issuer is still reached"
    );

    c1.update_trust(a.profile_with(&[&b.pki]))
        .expect("restore the second issuer");
    assert!(reaches(&node2.engine, "node:1:withdrawn", ARRIVES).await);
    node2.node.shutdown().await.expect("shutdown");
    founder.node.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread")]
async fn withdrawn_trust_cuts_a_member_off_under_spiffe() {
    tokio::time::timeout(
        TEST_TIMEOUT,
        withdrawn_trust_cuts_a_member_off(
            Fixture::spiffe_in("a.test"),
            Fixture::spiffe_in("b.test"),
        ),
    )
    .await
    .expect("in time");
}

#[tokio::test(flavor = "multi_thread")]
async fn withdrawn_trust_cuts_a_member_off_under_a_host_adapter() {
    // One host adapter (one issuer scope) accepting two CAs.
    let a = Fixture::host_in("a.test", "host-a");
    let b = Fixture {
        pki: TestPki::with_trust_domain("b.test"),
        profile: match &a.profile {
            Profile::Host(adapter) => Profile::Host(Arc::clone(adapter)),
            Profile::Spiffe => unreachable!("a host fixture"),
        },
    };
    tokio::time::timeout(TEST_TIMEOUT, withdrawn_trust_cuts_a_member_off(a, b))
        .await
        .expect("in time");
}

/// An embedding host's revocation reaches connections already made once
/// the host asks for its peers to be verified again.
#[tokio::test(flavor = "multi_thread")]
async fn a_host_revocation_cuts_a_member_off() {
    tokio::time::timeout(TEST_TIMEOUT, async {
        let fx = Fixture::host();
        let Profile::Host(adapter) = &fx.profile else {
            unreachable!("a host fixture")
        };
        let adapter = Arc::clone(adapter);
        // Three voters: with one cut off, the other two still commit.
        let g = group(&fx, 3).await;
        let ids = ProposalIdGenerator::with_base(8 << 48);
        write(&g[0].node, &ids, "node:1:before", 100);
        assert!(reaches(&g[2].engine, "node:1:before", ARRIVES).await);

        adapter.revoke(&fx.pki.spiffe_id(3));
        g[0].node
            .connections()
            .expect("peers")
            .context()
            .reverify_peers();
        write(&g[0].node, &ids, "node:1:revoked", 200);
        assert!(reaches(&g[1].engine, "node:1:revoked", ARRIVES).await);
        assert!(
            !reaches(&g[2].engine, "node:1:revoked", STAYS_AWAY).await,
            "a revoked member is still reached"
        );
        for m in g {
            m.node.shutdown().await.expect("shutdown");
        }
    })
    .await
    .expect("in time");
}

/// A member that moves to another address keeps its identity: the leader
/// is told where it is now and reaches it there, no other change made.
async fn a_member_that_moves_keeps_its_identity(fx: Fixture) {
    // Three voters: the two that stay commit while the third moves.
    let mut g = group(&fx, 3).await;
    // Node 3 moves holding its group's log: a member that never received
    // it would wait, at its new address, for a leader that does not know
    // that address yet.
    let ids = ProposalIdGenerator::with_base(9 << 48);
    write(&g[0].node, &ids, "node:1:before", 100);
    assert!(reaches(&g[2].engine, "node:1:before", ARRIVES).await);
    let moved = g.pop().expect("node 3");
    let dir = moved._dir;
    moved.node.shutdown().await.expect("stop node 3");
    drop(moved.node);
    drop(moved.engine);

    let engine = open_engine(&dir);
    let port = alloc_port();
    let node3 = RaftNode::open_cluster(
        3,
        Arc::clone(&engine),
        format!("127.0.0.1:{port}").parse().expect("addr"),
        format!("127.0.0.1:{port}"),
        fx.connections(3),
    )
    .await
    .expect("node 3 back on another port");
    g[0].node
        .add_node(3, format!("127.0.0.1:{port}"), fx.identity(3))
        .await
        .expect("re-address node 3");
    assert_eq!(
        g[0].node.member_identity(3),
        Some(coordinode_raft::member::MemberIdentity::from(
            &fx.identity(3)
        )),
        "the member keeps its identity"
    );
    write(&g[0].node, &ids, "node:1:moved", 200);
    assert!(reaches(&engine, "node:1:moved", ARRIVES).await);
    node3.shutdown().await.expect("shutdown");
    for m in g {
        m.node.shutdown().await.expect("shutdown");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_member_that_moves_keeps_its_identity_under_spiffe() {
    tokio::time::timeout(
        TEST_TIMEOUT,
        a_member_that_moves_keeps_its_identity(Fixture::spiffe()),
    )
    .await
    .expect("in time");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_member_that_moves_keeps_its_identity_under_a_host_adapter() {
    tokio::time::timeout(
        TEST_TIMEOUT,
        a_member_that_moves_keeps_its_identity(Fixture::host()),
    )
    .await
    .expect("in time");
}

/// Two independent databases in one process share no trust: each has its
/// own security context, a node of one is refused by the other, and a
/// credential or trust change in one leaves the other as it was.
async fn two_databases_in_one_process_share_no_trust(fa: Fixture, fb: Fixture) {
    {
        let ga = group(&fa, 2).await;
        let gb = group(&fb, 2).await;

        // A's node 1, dialing B's member, is refused at the handshake.
        let mut cross = client(&fb, &fa.connections(1), &gb[1]);
        assert!(
            cross
                .members(call(&gb[1], 1, gb[0].node.deployment(), members_query()))
                .await
                .is_err(),
            "a node of another database is refused"
        );

        // A renews its leader's credentials and asks for its peers to be
        // verified again; B's contexts see none of it.
        let a1 = ga[0].node.connections().expect("peers").context();
        let b1 = gb[0].node.connections().expect("peers").context();
        let b_generation = b1.generation();
        let (chain, key) = fa.pki.node_credentials(1);
        a1.update_credentials(chain, key).expect("renew in A");
        a1.update_trust(fa.profile_with(&[]))
            .expect("same trust in A");
        a1.reverify_peers();
        assert_eq!(b1.generation(), b_generation, "B's trust never moved");
        assert_eq!(b1.identity(), fb.identity(1));

        let ids = ProposalIdGenerator::with_base(10 << 48);
        write(&ga[0].node, &ids, "node:1:in-a", 100);
        write(&gb[0].node, &ids, "node:1:in-b", 200);
        assert!(reaches(&ga[1].engine, "node:1:in-a", ARRIVES).await);
        assert!(reaches(&gb[1].engine, "node:1:in-b", ARRIVES).await);
        assert!(!reaches(&ga[1].engine, "node:1:in-b", STAYS_AWAY).await);

        for m in ga.into_iter().chain(gb) {
            m.node.shutdown().await.expect("shutdown");
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn two_databases_in_one_process_share_no_trust_under_spiffe() {
    tokio::time::timeout(
        TEST_TIMEOUT,
        two_databases_in_one_process_share_no_trust(
            Fixture::spiffe_in("a.test"),
            Fixture::spiffe_in("b.test"),
        ),
    )
    .await
    .expect("in time");
}

#[tokio::test(flavor = "multi_thread")]
async fn two_databases_in_one_process_share_no_trust_under_host_adapters() {
    // Each database's host adapter identifies nodes in its own issuer scope.
    tokio::time::timeout(
        TEST_TIMEOUT,
        two_databases_in_one_process_share_no_trust(
            Fixture::host_in("a.test", "host-a"),
            Fixture::host_in("b.test", "host-b"),
        ),
    )
    .await
    .expect("in time");
}

/// A node told which deployment it forms its group in, whose store already
/// belongs to another, refuses to open: serving under the recorded one would
/// leave the nodes configured with the named deployment unable to join, with
/// nothing saying why.
#[tokio::test(flavor = "multi_thread")]
async fn a_store_of_another_deployment_refuses_the_one_named() {
    tokio::time::timeout(TEST_TIMEOUT, async {
        let fx = Fixture::spiffe();
        let dir = tempfile::tempdir().expect("dir");
        let engine = open_engine(&dir);
        let port = alloc_port();
        let open = |deployment: Option<DeploymentId>| {
            RaftNode::open_cluster_with_options(
                1,
                Arc::clone(&engine),
                format!("127.0.0.1:{port}").parse().expect("addr"),
                format!("127.0.0.1:{port}"),
                NodeOptions {
                    connections: Some(fx.connections(1)),
                    deployment,
                    ..NodeOptions::default()
                },
            )
        };

        let node = open(None).await.expect("form a group");
        let recorded = node.deployment().expect("the deployment it formed");
        node.shutdown().await.expect("shutdown");
        drop(node);

        let mut other = recorded.0;
        other[0] ^= 0xff;
        let other = DeploymentId(other);
        match open(Some(other)).await {
            Err(RaftNodeError::OtherDeployment { bound, asked }) => {
                assert_eq!((bound, asked), (recorded, other));
            }
            Err(e) => panic!("refused for another reason: {e}"),
            Ok(_) => panic!("opened under another deployment than it names"),
        }

        let node = open(Some(recorded)).await.expect("its own deployment");
        assert_eq!(node.deployment(), Some(recorded));
        node.shutdown().await.expect("shutdown");
    })
    .await
    .expect("in time");
}
