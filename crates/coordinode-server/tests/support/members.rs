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
use coordinode_storage::engine::core::Deployment;
use parking_lot::RwLock;

use coordinode_test_fixtures::{TestPki, alloc_port, serve_member_listener, serve_peers_until};

/// The CA the members are issued their identities by.
pub static PKI: std::sync::LazyLock<TestPki> = std::sync::LazyLock::new(TestPki::new);

pub struct Node {
    pub db: Arc<RwLock<Database>>,
    pub raft: Arc<RaftNode>,
    /// What the member's own clients' statements run through.
    pub executor: StatementExecutor,
    /// The change-stream consumers' registry runs while the member does.
    _registry: coordinode_replicate::RegistryBackground,
    _dir: tempfile::TempDir,
}

/// Open a member serving on `port`. It joins the group of `founder`, which
/// listens on the port given with it and adds it as a learner; without one
/// it founds a group.
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
    let deployment = Deployment::open_with_oracle(&config, oracle.clone()).unwrap();
    // A founder starts the store's namespace; a joiner receives its group's.
    let engine = Arc::new(match founder {
        None => deployment.single_store().unwrap(),
        Some(_) => deployment.member_engine(),
    });
    let connections = PKI.connections(node_id);
    let addr: std::net::SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();

    let (raft, raft_handler, listener) = match founder {
        None => {
            let (raft, handler) = RaftNode::open_cluster_embedded(
                node_id,
                Arc::clone(&engine),
                format!("127.0.0.1:{port}"),
                connections,
            )
            .await
            .unwrap();
            let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
            (Arc::new(raft), handler, listener)
        }
        // A joining member learns its own advertised address from the
        // leader's add_node call, so it takes none here. It serves its
        // peers until the leader has added it and its namespace has
        // arrived, and its clients from then on.
        Some((founder, founder_port)) => {
            let enrollment = founder
                .raft
                .enrollment(format!("127.0.0.1:{founder_port}"))
                .unwrap();
            let (raft, handler) = RaftNode::open_joining_embedded(
                node_id,
                Arc::clone(&engine),
                connections,
                enrollment,
            )
            .await
            .unwrap();
            let raft = Arc::new(raft);
            let context = Arc::clone(raft.connections().unwrap().context());
            let (listener, ()) = serve_peers_until(
                addr,
                &context,
                tonic::service::Routes::new(RaftServiceServer::new(handler.clone())),
                async {
                    // A membership change needs a leader to take it.
                    while !founder.raft.is_leader().await {
                        tokio::time::sleep(Duration::from_millis(50)).await;
                    }
                    founder
                        .raft
                        .add_node(node_id, format!("127.0.0.1:{port}"), PKI.identity(node_id))
                        .await
                        .unwrap();
                    let mut applied = raft.subscribe_applied();
                    while !engine.try_bind().unwrap() {
                        applied.changed().await.unwrap();
                    }
                },
            )
            .await;
            (raft, handler, listener)
        }
    };

    let pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> =
        Arc::new(RaftProposalPipeline::new(Arc::clone(raft.raft())));
    // Change streams over the member's log, with their consumers' registry,
    // as a deployed member serves them.
    let (cdc, registry) = {
        let applied = Arc::clone(&raft);
        let source = Arc::new(coordinode_replicate::LogRetentionSource::new(
            Arc::clone(&engine),
            coordinode_raft::storage::raft_oplog_dirs(&engine, 0)
                .unwrap()
                .all,
            raft.retained_floor(),
            Arc::new(move || applied.applied_through()),
        ));
        let registry = coordinode_replicate::ShardConsumerRegistry::new(
            Arc::clone(&engine),
            Arc::clone(&pipeline),
            Arc::new(
                coordinode_core::txn::proposal::ProposalIdGenerator::with_base(
                    coordinode_core::txn::proposal::fresh_proposal_id_base(),
                ),
            ),
            Arc::new(coordinode_replicate::SystemClock),
            source,
        )
        .unwrap();
        let background = registry
            .start_background(coordinode_replicate::BackgroundConfig::default())
            .unwrap();
        let cdc = coordinode_server::services::cdc::ChangeEventServiceImpl::for_raft_node(
            &engine,
            Arc::clone(&raft),
            registry,
        )
        .unwrap();
        (Arc::new(cdc), background)
    };
    let db = Arc::new(RwLock::new(
        Database::from_engine(dir.path(), engine, oracle, pipeline).unwrap(),
    ));

    let executor = StatementExecutor::new(Arc::clone(&db)).with_raft_node(Arc::clone(&raft));
    let cypher = CypherServiceImpl::from_executor(executor.clone());
    let operations = OperationServiceImpl::new(executor.clone());
    let sessions = SessionSvc::new(executor.clone(), Arc::new(SessionRegistry::new()))
        .with_cluster(Arc::clone(&raft))
        .with_change_streams(cdc);
    // What a member passes its clients' commands to the leader through.
    let routing =
        CommandRoutingHandler::new(Arc::new(ServerCommandExecutor::new(executor.clone())))
            .into_service(MAX_ROUTED_MESSAGE);

    let context = Arc::clone(raft.connections().unwrap().context());
    serve_member_listener(
        listener,
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
    );

    Node {
        db,
        raft,
        executor,
        _registry: registry,
        _dir: dir,
    }
}

/// A flat group of three: member 1 leads, members 2 and 3 follow. Returns
/// them with the ports they serve on.
pub async fn three_members() -> Vec<(Node, u16)> {
    let ports = [alloc_port(), alloc_port(), alloc_port()];
    let n1 = open_node(1, ports[0], None).await;
    let n2 = open_node(2, ports[1], Some((&n1, ports[0]))).await;
    let n3 = open_node(3, ports[2], Some((&n1, ports[0]))).await;
    n1.raft.change_membership(vec![1, 2, 3]).await.unwrap();
    tokio::time::sleep(Duration::from_millis(800)).await;
    vec![(n1, ports[0]), (n2, ports[1]), (n3, ports[2])]
}

/// A group of two: member 1 leads, member 2 follows. Returns both with the
/// ports they serve on.
pub async fn leader_and_follower() -> ((Node, u16), (Node, u16)) {
    let (p1, p2) = (alloc_port(), alloc_port());
    let n1 = open_node(1, p1, None).await;
    let n2 = open_node(2, p2, Some((&n1, p1))).await;
    n1.raft.change_membership(vec![1, 2]).await.unwrap();
    tokio::time::sleep(Duration::from_millis(800)).await;
    ((n1, p1), (n2, p2))
}
