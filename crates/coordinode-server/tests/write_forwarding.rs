//! A write sent to a follower reaches the leader anyway.
//!
//! Only the leader can replicate a write, but making every client discover
//! which node that is turns a leadership change into an application problem.
//! A node that knows who leads passes the request along instead, so a client
//! that opened one connection and never thinks about topology keeps working
//! across an election. The response says how many nodes handled it, for a
//! client that does care.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;
use std::time::Duration;

use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_embed::Database;
use coordinode_raft::cluster::RaftNode;
use coordinode_raft::proposal::RaftProposalPipeline;
use coordinode_raft::proto::replication::raft_service_server::RaftServiceServer;
use coordinode_server::proto::{operation, query};
use coordinode_server::services::cypher::CypherServiceImpl;
use coordinode_server::services::operation::OperationServiceImpl;
use coordinode_server::services::statement::StatementExecutor;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use parking_lot::RwLock;

use coordinode_test_fixtures::{TestPki, alloc_port, serve_member_port};

/// The CA the nodes are issued their identities by.
static PKI: std::sync::LazyLock<TestPki> = std::sync::LazyLock::new(TestPki::new);

struct Node {
    db: Arc<RwLock<Database>>,
    raft: Arc<RaftNode>,
    _dir: tempfile::TempDir,
}

/// Open a node and serve both its Raft and its Cypher API on ONE port, the way
/// a deployed node does. That is what makes forwarding meaningful: the address
/// a peer is told to reach in the membership is the same address a client
/// reaches, so a node can pass a client's request to the leader by the address
/// it already knows. A node joins the group of `founder`, which listens on
/// the port given with it; without one it founds a group.
async fn open_node(node_id: u64, port: u16, founder: Option<(&Node, u16)>) -> Node {
    let dir = tempfile::tempdir().unwrap();
    let oracle = Arc::new(TimestampOracle::new());
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = Arc::new(StorageEngine::open_with_oracle(&config, oracle.clone()).unwrap());
    let connections = PKI.connections(node_id);

    let (raft, raft_handler) = match founder {
        None => RaftNode::open_cluster_embedded(
            node_id,
            Arc::clone(&engine),
            format!("127.0.0.1:{port}"),
            connections,
        )
        .await
        .unwrap(),
        // A joining node learns its own advertised address from the leader's
        // add_node call, so it takes none here.
        Some((founder, founder_port)) => {
            let enrollment = founder
                .raft
                .enrollment(format!("127.0.0.1:{founder_port}"))
                .unwrap();
            RaftNode::open_joining_embedded(node_id, Arc::clone(&engine), connections, enrollment)
                .await
                .unwrap()
        }
    };
    let raft = Arc::new(raft);

    let pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> =
        Arc::new(RaftProposalPipeline::new(Arc::clone(raft.raft())));
    let db = Arc::new(RwLock::new(
        Database::from_engine(dir.path(), engine, oracle, pipeline).unwrap(),
    ));

    let executor = StatementExecutor::new(Arc::clone(&db)).with_raft_node(Arc::clone(&raft));
    let cypher = CypherServiceImpl::from_executor(executor.clone());
    let operations = OperationServiceImpl::new(executor);

    let addr: std::net::SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
    let context = Arc::clone(raft.connections().unwrap().context());
    serve_member_port(
        addr,
        &context,
        tonic::service::Routes::new(RaftServiceServer::new(raft_handler))
            .add_service(query::cypher_service_server::CypherServiceServer::new(
                cypher,
            ))
            .add_service(
                operation::operation_service_server::OperationServiceServer::new(operations),
            ),
    )
    .await;

    Node {
        db,
        raft,
        _dir: dir,
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_write_sent_to_a_follower_is_carried_to_the_leader() {
    let (p1, p2) = (alloc_port(), alloc_port());

    let n1 = open_node(1, p1, None).await;
    let _n2 = open_node(2, p2, Some((&n1, p1))).await;

    tokio::time::sleep(Duration::from_millis(800)).await;
    n1.raft
        .add_node(2, format!("127.0.0.1:{p2}"), PKI.identity(2))
        .await
        .unwrap();
    n1.raft.change_membership(vec![1, 2]).await.unwrap();
    tokio::time::sleep(Duration::from_millis(800)).await;

    // The client talks to the FOLLOWER and never learns there is a leader.
    let mut client = query::cypher_service_client::CypherServiceClient::connect(format!(
        "http://127.0.0.1:{p2}"
    ))
    .await
    .unwrap();

    let response = client
        .execute_cypher(query::ExecuteCypherRequest {
            query: "CREATE (:Forwarded {tag: 'via-follower'})".to_string(),
            ..Default::default()
        })
        .await
        .expect("a write to a follower must reach the leader");

    assert_eq!(
        response
            .metadata()
            .get("x-coordinode-hops")
            .map(|v| v.to_str().unwrap_or_default()),
        Some("1"),
        "the response should say the write was carried one hop"
    );

    // It really landed, and on the leader: this is the state Raft replicated,
    // not something the follower accepted locally.
    tokio::time::sleep(Duration::from_millis(500)).await;
    let rows = n1
        .db
        .write()
        .execute_cypher("MATCH (n:Forwarded) RETURN n.tag AS tag")
        .unwrap();
    assert_eq!(
        rows.len(),
        1,
        "the forwarded write is missing on the leader"
    );
    assert_eq!(
        rows[0].get("tag").and_then(|v| v.as_str()),
        Some("via-follower")
    );
}

/// A client that only knows a follower numbers its writes all the same: the
/// enrollment, a bound write and its retry, and an exclusion all reach the
/// leader, and the write takes effect once.
#[tokio::test(flavor = "multi_thread")]
async fn operation_identity_works_through_a_follower() {
    use tonic_types::StatusExt;

    let (p1, p2) = (alloc_port(), alloc_port());
    let n1 = open_node(1, p1, None).await;
    let _n2 = open_node(2, p2, Some((&n1, p1))).await;
    tokio::time::sleep(Duration::from_millis(800)).await;
    n1.raft
        .add_node(2, format!("127.0.0.1:{p2}"), PKI.identity(2))
        .await
        .unwrap();
    n1.raft.change_membership(vec![1, 2]).await.unwrap();
    tokio::time::sleep(Duration::from_millis(800)).await;

    let follower = format!("http://127.0.0.1:{p2}");
    let mut operations =
        operation::operation_service_client::OperationServiceClient::connect(follower.clone())
            .await
            .unwrap();
    let mut cypher = query::cypher_service_client::CypherServiceClient::connect(follower)
        .await
        .unwrap();

    let enrolled = operations
        .enroll_issuer(operation::EnrollIssuerRequest {
            issuer: vec![0x52; 16],
        })
        .await
        .expect("an enrollment through a follower reaches the leader")
        .into_inner();
    let reference = |sequence| operation::OperationRef {
        issuer: vec![0x52; 16],
        incarnation: enrolled.incarnation,
        sequence,
        deployment: enrolled.deployment.clone(),
    };
    let write = |sequence| query::ExecuteCypherRequest {
        query: "CREATE (:Bound {k: 1})".to_string(),
        operation: Some(reference(sequence)),
        ..Default::default()
    };

    cypher
        .execute_cypher(write(enrolled.next_sequence))
        .await
        .expect("a bound write through a follower");
    let refused = cypher
        .execute_cypher(write(enrolled.next_sequence))
        .await
        .expect_err("its retry runs nothing");
    let info = refused.get_details_error_info().expect("ErrorInfo");
    assert_eq!(info.reason, "OPERATION_SETTLED");
    assert_eq!(info.metadata["outcome"], "COMMITTED");

    let excluded = operations
        .resolve_operation(operation::ResolveOperationRequest {
            operation: Some(reference(enrolled.next_sequence + 1)),
            exclude: true,
        })
        .await
        .expect("an exclusion through a follower reaches the leader")
        .into_inner();
    assert_eq!(excluded.state(), operation::OperationState::Aborted);

    tokio::time::sleep(Duration::from_millis(500)).await;
    let rows = n1
        .db
        .write()
        .execute_cypher("MATCH (n:Bound) RETURN n")
        .unwrap();
    assert_eq!(rows.len(), 1, "the write took effect once");
}
