#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! One server hosts replicas of several consensus groups behind one port.
//!
//! Every consensus message names its group and the server dispatches it to
//! its replica of that group, so groups whose members share servers
//! replicate independently, and a message for a group the server does not
//! host, or one whose parts name different groups, is refused by name.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use coordinode_core::txn::proposal::{
    Mutation, PartitionId, ProposalIdGenerator, ProposalPipeline, RaftProposal,
};
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_raft::cluster::grpc_server::{GROUP_NOT_HOSTED, HostError};
use coordinode_raft::cluster::version::write_handshake;
use coordinode_raft::cluster::{GroupId, NodeOptions, PeerConnections, RaftGrpcHandler, RaftNode};
use coordinode_raft::proto::replication::RaftPayload;
use coordinode_raft::proto::replication::raft_service_client::RaftServiceClient;
use coordinode_raft::proto::replication::raft_service_server::RaftServiceServer;
use coordinode_raft::storage::TypeConfig;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_test_fixtures::alloc_port;
use futures_util::StreamExt;
use tonic_types::StatusExt;

const TEST_TIMEOUT: Duration = Duration::from_secs(60);

/// One server's replica of one group.
struct Replica {
    node: RaftNode,
    engine: Arc<StorageEngine>,
    _dir: tempfile::TempDir,
}

fn engine() -> (Arc<StorageEngine>, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    (Arc::new(StorageEngine::open(&config).expect("open")), dir)
}

fn options(group: GroupId, connections: &Arc<PeerConnections>) -> NodeOptions {
    NodeOptions {
        group,
        connections: Arc::clone(connections),
        ..NodeOptions::default()
    }
}

/// Server `node_id`'s replica of `group`: the group's founding member when
/// `forms`, a member waiting to be added otherwise. The server's groups share
/// `connections`.
async fn replica(
    node_id: u64,
    group: GroupId,
    port: u16,
    forms: bool,
    connections: &Arc<PeerConnections>,
) -> (Replica, RaftGrpcHandler) {
    let (engine, dir) = engine();
    let (node, handler) = if forms {
        RaftNode::open_cluster_embedded_with_options(
            node_id,
            Arc::clone(&engine),
            format!("http://127.0.0.1:{port}"),
            options(group, connections),
        )
        .await
        .expect("open the forming member")
    } else {
        RaftNode::open_joining_embedded_with_options(
            node_id,
            Arc::clone(&engine),
            options(group, connections),
        )
        .await
        .expect("open a joining member")
    };
    (
        Replica {
            node,
            engine,
            _dir: dir,
        },
        handler,
    )
}

/// Serve `handler`, and every group it hosts, on `port`.
fn serve(handler: RaftGrpcHandler, port: u16) -> tokio::task::JoinHandle<()> {
    serve_counting(handler, port, Arc::new(AtomicUsize::new(0)))
}

/// [`serve`], counting the connections the server accepts in `accepted`.
fn serve_counting(
    handler: RaftGrpcHandler,
    port: u16,
    accepted: Arc<AtomicUsize>,
) -> tokio::task::JoinHandle<()> {
    let addr: std::net::SocketAddr = format!("127.0.0.1:{port}").parse().expect("addr");
    tokio::spawn(async move {
        let listener = tokio::net::TcpListener::bind(addr).await.expect("bind");
        let incoming = futures_util::stream::unfold(listener, move |listener| {
            let accepted = Arc::clone(&accepted);
            async move {
                let next = listener.accept().await.map(|(stream, _)| {
                    accepted.fetch_add(1, Ordering::SeqCst);
                    stream
                });
                Some((next, listener))
            }
        });
        tonic::transport::Server::builder()
            .add_service(handler.handshake_service())
            .add_service(RaftServiceServer::new(handler))
            .serve_with_incoming(incoming)
            .await
            .expect("serve");
    })
}

async fn await_leadership(node: &RaftNode) {
    for _ in 0..150 {
        if node.is_leader().await {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("node {} never became leader", node.node_id());
}

async fn await_value(engine: &StorageEngine, key: &[u8], expected: &[u8]) -> bool {
    for _ in 0..150 {
        if let Ok(Some(v)) = engine.get(Partition::Node, key) {
            if v.as_ref() == expected {
                return true;
            }
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    false
}

fn put(key: &[u8], value: &[u8]) -> RaftProposal {
    RaftProposal {
        id: ProposalIdGenerator::with_base(1u64 << 48).next(),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: key.to_vec(),
            value: value.to_vec(),
        }],
        commit_ts: Timestamp::from_raw(100),
        start_ts: Timestamp::from_raw(99),
        bypass_rate_limiter: false,
    }
}

/// Make `forming` lead its group with the other two servers as voters.
async fn form_group(forming: &RaftNode, others: [(u64, u16); 2]) {
    await_leadership(forming).await;
    for (id, port) in others {
        forming
            .add_node(id, format!("http://127.0.0.1:{port}"))
            .await
            .expect("add a member");
    }
    let mut voters = vec![forming.node_id()];
    voters.extend(others.iter().map(|(id, _)| *id));
    forming
        .change_membership(voters)
        .await
        .expect("make the members voters");
}

/// Two groups whose members sit on the same three servers, one port each,
/// replicate independently: each group elects its own leader on a
/// different server, every write reaches the three replicas of its own
/// group, and none of the other's. Before messages named their group, a
/// server could serve one group only, and a second group's messages would
/// have gone to the first group's replica.
#[tokio::test(flavor = "multi_thread")]
async fn two_groups_on_the_same_three_servers_replicate_independently() {
    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let a = GroupId::FORMING;
        let b = GroupId(1);
        let ports = [alloc_port(), alloc_port(), alloc_port()];

        // Group a forms on server 1, group b on server 2.
        let mut a_replicas = Vec::new();
        let mut b_replicas = Vec::new();
        let mut servers = Vec::new();
        for (i, port) in ports.iter().copied().enumerate() {
            let id = i as u64 + 1;
            let connections = Arc::new(PeerConnections::default());
            let (ra, ha) = replica(id, a, port, id == 1, &connections).await;
            let (rb, hb) = replica(id, b, port, id == 2, &connections).await;
            ha.host(&hb).expect("host both groups on one server");
            assert_eq!(ha.hosted().groups(), vec![a, b]);
            servers.push(serve(ha, port));
            a_replicas.push(ra);
            b_replicas.push(rb);
        }

        form_group(&a_replicas[0].node, [(2, ports[1]), (3, ports[2])]).await;
        form_group(&b_replicas[1].node, [(1, ports[0]), (3, ports[2])]).await;

        a_replicas[0]
            .node
            .pipeline()
            .propose_and_wait(&put(b"node:1:in-a", b"a"))
            .expect("write to group a");
        b_replicas[1]
            .node
            .pipeline()
            .propose_and_wait(&put(b"node:1:in-b", b"b"))
            .expect("write to group b");

        for r in &a_replicas {
            assert!(
                await_value(&r.engine, b"node:1:in-a", b"a").await,
                "group a's write missing on its member {}",
                r.node.node_id()
            );
        }
        for r in &b_replicas {
            assert!(
                await_value(&r.engine, b"node:1:in-b", b"b").await,
                "group b's write missing on its member {}",
                r.node.node_id()
            );
        }
        for r in &a_replicas {
            assert_eq!(
                r.engine.get(Partition::Node, b"node:1:in-b").expect("read"),
                None,
                "group b's write reached a replica of group a"
            );
        }
        for r in &b_replicas {
            assert_eq!(
                r.engine.get(Partition::Node, b"node:1:in-a").expect("read"),
                None,
                "group a's write reached a replica of group b"
            );
        }

        assert!(a_replicas[0].node.is_leader().await);
        assert!(b_replicas[1].node.is_leader().await);

        for r in a_replicas.iter().chain(b_replicas.iter()) {
            r.node.shutdown().await.expect("shutdown");
        }
        for s in servers {
            s.abort();
        }
    })
    .await;
    assert!(
        result.is_ok(),
        "TIMED OUT: two_groups_on_the_same_three_servers_replicate_independently"
    );
}

/// Two groups led from the same server reach each follower server over one
/// connection: every group, and every client openraft makes for a peer,
/// shares the server's connection to that peer. Before, each group, and each
/// replication restart within one, opened a connection of its own.
#[tokio::test(flavor = "multi_thread")]
async fn groups_led_from_one_server_share_its_connection_to_each_peer() {
    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let a = GroupId::FORMING;
        let b = GroupId(1);
        let ports = [alloc_port(), alloc_port(), alloc_port()];

        let mut replicas = Vec::new();
        let mut servers = Vec::new();
        let mut accepted = Vec::new();
        let mut pools = Vec::new();
        for (i, port) in ports.iter().copied().enumerate() {
            let id = i as u64 + 1;
            let connections = Arc::new(PeerConnections::default());
            // Both groups form on server 1, so it leads both.
            let (ra, ha) = replica(id, a, port, id == 1, &connections).await;
            let (rb, hb) = replica(id, b, port, id == 1, &connections).await;
            ha.host(&hb).expect("host both groups on one server");
            let count = Arc::new(AtomicUsize::new(0));
            servers.push(serve_counting(ha, port, Arc::clone(&count)));
            accepted.push(count);
            pools.push(connections);
            replicas.push((ra, rb));
        }

        let peers = [(2, ports[1]), (3, ports[2])];
        form_group(&replicas[0].0.node, peers).await;
        form_group(&replicas[0].1.node, peers).await;
        for (group, leader) in [(a, &replicas[0].0), (b, &replicas[0].1)] {
            let key = format!("node:1:{}", group.raw());
            leader
                .node
                .pipeline()
                .propose_and_wait(&put(key.as_bytes(), b"v"))
                .expect("write");
        }
        for (ra, rb) in &replicas {
            assert!(await_value(&ra.engine, b"node:1:0", b"v").await);
            assert!(await_value(&rb.engine, b"node:1:1", b"v").await);
        }

        // Server 1 dialled each peer once for both groups.
        assert_eq!(pools[0].len(), 2, "one connection per peer of server 1");
        assert!(replicas[0].0.node.is_leader().await);
        assert!(replicas[0].1.node.is_leader().await);
        for (server, count) in accepted.iter().enumerate().skip(1) {
            assert_eq!(
                count.load(Ordering::SeqCst),
                1,
                "server {} accepted more than the one connection from its leader",
                server + 1
            );
        }

        for (ra, rb) in &replicas {
            ra.node.shutdown().await.expect("shutdown");
            rb.node.shutdown().await.expect("shutdown");
        }
        for s in servers {
            s.abort();
        }
    })
    .await;
    assert!(
        result.is_ok(),
        "TIMED OUT: groups_led_from_one_server_share_its_connection_to_each_peer"
    );
}

fn vote_request() -> Vec<u8> {
    let request = openraft::raft::VoteRequest::<TypeConfig> {
        vote: openraft::type_config::alias::VoteOf::<TypeConfig>::new(1, 9),
        last_log_id: None,
        leadership_transfer: false,
    };
    rmp_serde::to_vec(&request).expect("encode")
}

/// One server hosting the forming group only, and its member.
async fn one_group_server() -> (Replica, u16, tokio::task::JoinHandle<()>) {
    let port = alloc_port();
    let (r, handler) = replica(1, GroupId::FORMING, port, true, &Arc::default()).await;
    let server = serve(handler, port);
    await_leadership(&r.node).await;
    (r, port, server)
}

async fn client(port: u16) -> RaftServiceClient<tonic::transport::Channel> {
    RaftServiceClient::connect(format!("http://127.0.0.1:{port}"))
        .await
        .expect("connect")
}

/// A request for a group the server hosts no replica of is refused with
/// NOT_FOUND carrying `ErrorInfo` (reason, domain, the group), so the
/// sender can tell a missing replica from a failed one.
#[tokio::test(flavor = "multi_thread")]
async fn a_request_for_a_group_not_hosted_is_refused_by_name() {
    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let (r, port, server) = one_group_server().await;

        let status = client(port)
            .await
            .vote(RaftPayload {
                data: vote_request(),
                group: 7,
            })
            .await
            .expect_err("a vote for a group not hosted is refused");
        assert_eq!(status.code(), tonic::Code::NotFound, "{status:?}");
        let info = status
            .get_details_error_info()
            .expect("the refusal carries ErrorInfo");
        assert_eq!(info.reason, GROUP_NOT_HOSTED);
        assert_eq!(info.domain, coordinode_core::ERROR_DOMAIN);
        assert_eq!(info.metadata.get("group").map(String::as_str), Some("7"));

        r.node.shutdown().await.expect("shutdown");
        server.abort();
    })
    .await;
    assert!(
        result.is_ok(),
        "TIMED OUT: a_request_for_a_group_not_hosted_is_refused_by_name"
    );
}

/// A request whose message names the hosted group but whose version record
/// speaks for another is refused before its payload is read: the two
/// parts of one call must name the same group.
#[tokio::test(flavor = "multi_thread")]
async fn a_record_of_another_group_than_its_message_is_refused() {
    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let (r, port, server) = one_group_server().await;

        let mut record = r.node.version().local_handshake();
        record.node_id = 2;
        record.group_id = GroupId(5);
        let mut request = tonic::Request::new(RaftPayload {
            data: vote_request(),
            group: GroupId::FORMING.raw(),
        });
        write_handshake(request.metadata_mut(), &record);

        let status = client(port)
            .await
            .vote(request)
            .await
            .expect_err("a call whose record names another group is refused");
        assert_eq!(status.code(), tonic::Code::FailedPrecondition, "{status:?}");
        assert!(status.message().contains("group 5"), "{status:?}");

        r.node.shutdown().await.expect("shutdown");
        server.abort();
    })
    .await;
    assert!(
        result.is_ok(),
        "TIMED OUT: a_record_of_another_group_than_its_message_is_refused"
    );
}

/// An append stream is routed by its version record; a message in it that
/// names another group ends the stream with INVALID_ARGUMENT instead of
/// reaching the replica the stream was routed to.
#[tokio::test(flavor = "multi_thread")]
async fn a_stream_message_of_another_group_ends_the_stream() {
    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let (r, port, server) = one_group_server().await;

        let mut record = r.node.version().local_handshake();
        record.node_id = 2;
        let append = openraft::raft::AppendEntriesRequest::<TypeConfig> {
            vote: openraft::type_config::alias::VoteOf::<TypeConfig>::new(1, 9),
            prev_log_id: None,
            entries: Vec::new(),
            leader_commit: None,
        };
        let mut request = tonic::Request::new(futures_util::stream::iter([RaftPayload {
            data: rmp_serde::to_vec(&append).expect("encode"),
            group: 3,
        }]));
        write_handshake(request.metadata_mut(), &record);

        let mut replies = client(port)
            .await
            .stream_append(request)
            .await
            .expect("the stream opens: it is routed by its record")
            .into_inner();
        let mut last = None;
        while let Some(reply) = replies.next().await {
            last = Some(reply);
            if matches!(last, Some(Err(_))) {
                break;
            }
        }
        assert!(
            matches!(&last, Some(Err(status)) if status.code() == tonic::Code::InvalidArgument),
            "a message of another group must end the stream, got {last:?}"
        );

        r.node.shutdown().await.expect("shutdown");
        server.abort();
    })
    .await;
    assert!(
        result.is_ok(),
        "TIMED OUT: a_stream_message_of_another_group_ends_the_stream"
    );
}

/// A server cannot host two replicas of one group: adding one for a group
/// already hosted is refused and leaves the hosted set as it was.
#[tokio::test(flavor = "multi_thread")]
async fn a_second_replica_of_a_hosted_group_is_refused() {
    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let port = alloc_port();
        let (first, handler) = replica(1, GroupId::FORMING, port, true, &Arc::default()).await;
        let (second, other) =
            replica(1, GroupId::FORMING, alloc_port(), true, &Arc::default()).await;

        assert_eq!(
            handler.host(&other),
            Err(HostError::AlreadyHosted(GroupId::FORMING))
        );
        assert_eq!(handler.hosted().groups(), vec![GroupId::FORMING]);

        first.node.shutdown().await.expect("shutdown");
        second.node.shutdown().await.expect("shutdown");
    })
    .await;
    assert!(
        result.is_ok(),
        "TIMED OUT: a_second_replica_of_a_hosted_group_is_refused"
    );
}
