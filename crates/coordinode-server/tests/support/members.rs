//! Members of a group in one process, each serving its Raft, its Cypher and
//! operation APIs and the routing of other members' commands on one port, the
//! way a deployed node does: the address a peer reaches in the membership is
//! the one a client reaches, so a member passes a client's command to the
//! leader by the address it already knows.

use std::sync::Arc;
use std::time::Duration;

use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_embed::Database;
use coordinode_embed::replication::routing::{CommandRoutingHandler, MAX_ROUTED_MESSAGE};
use coordinode_raft::cluster::RaftNode;
use coordinode_raft::proposal::RaftProposalPipeline;
use coordinode_raft::proto::replication::raft_service_server::RaftServiceServer;
use coordinode_server::proto::{operation, query, session};
use coordinode_server::services::cypher::CypherServiceImpl;
use coordinode_server::services::operation::OperationServiceImpl;
use coordinode_server::services::routing::ServerCommandExecutor;
use coordinode_server::services::session::SessionSvc;
use coordinode_server::services::statement::StatementExecutor;
use coordinode_session::SessionRegistry;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use parking_lot::RwLock;

use coordinode_test_fixtures::{TestPki, alloc_port, serve_member_port};

/// The CA the members are issued their identities by.
pub static PKI: std::sync::LazyLock<TestPki> = std::sync::LazyLock::new(TestPki::new);

pub struct Node {
    pub db: Arc<RwLock<Database>>,
    pub raft: Arc<RaftNode>,
    /// What the member's own clients' statements run through.
    pub executor: StatementExecutor,
    _dir: tempfile::TempDir,
}

/// Open a member serving on `port`. It joins the group of `founder`, which
/// listens on the port given with it; without one it founds a group.
pub async fn open_node(node_id: u64, port: u16, founder: Option<(&Node, u16)>) -> Node {
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
        // A joining member learns its own advertised address from the
        // leader's add_node call, so it takes none here.
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
    let operations = OperationServiceImpl::new(executor.clone());
    let sessions = SessionSvc::new(executor.clone(), Arc::new(SessionRegistry::new()))
        .with_cluster(Arc::clone(&raft));
    // What a member passes its clients' commands to the leader through.
    let routing =
        CommandRoutingHandler::new(Arc::new(ServerCommandExecutor::new(executor.clone())))
            .into_service(MAX_ROUTED_MESSAGE);

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
            )
            .add_service(session::session_service_server::SessionServiceServer::new(
                sessions,
            ))
            .add_service(routing),
    )
    .await;

    Node {
        db,
        raft,
        executor,
        _dir: dir,
    }
}

/// A group of two: member 1 leads, member 2 follows. Returns both with the
/// ports they serve on.
pub async fn leader_and_follower() -> ((Node, u16), (Node, u16)) {
    let (p1, p2) = (alloc_port(), alloc_port());
    let n1 = open_node(1, p1, None).await;
    let n2 = open_node(2, p2, Some((&n1, p1))).await;
    tokio::time::sleep(Duration::from_millis(800)).await;
    n1.raft
        .add_node(2, format!("127.0.0.1:{p2}"), PKI.identity(2))
        .await
        .unwrap();
    n1.raft.change_membership(vec![1, 2]).await.unwrap();
    tokio::time::sleep(Duration::from_millis(800)).await;
    ((n1, p1), (n2, p2))
}
