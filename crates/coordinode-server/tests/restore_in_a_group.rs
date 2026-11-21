//! A restore's identifiers stay taken for the whole group.
//!
//! A restore writes the identifiers of its dump and takes their sequences
//! from the node identifier lease before the first record. The lease is
//! granted through the replicated log, so every member knows it: a member
//! that leads later allocates above the restored identifiers too, instead of
//! handing one out again from a ceiling only the old leader had seen.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;
use std::time::Duration;

use coordinode_core::graph::types::Value;
use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_embed::Database;
use coordinode_embed::backup::BackupFormat;
use coordinode_embed::backup::restore::RestoreOptions;
use coordinode_raft::cluster::RaftNode;
use coordinode_raft::proposal::RaftProposalPipeline;
use coordinode_raft::proto::replication::raft_service_server::RaftServiceServer;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::Deployment;
use coordinode_storage::engine::metadata::node_lease_ceiling;
use parking_lot::RwLock;

use coordinode_test_fixtures::{
    TestPki, alloc_port, serve_member_listener, serve_member_port, serve_peers_until,
};

/// The CA the members are issued their identities by.
static PKI: std::sync::LazyLock<TestPki> = std::sync::LazyLock::new(TestPki::new);

struct Node {
    db: Arc<RwLock<Database>>,
    raft: Arc<RaftNode>,
    _dir: tempfile::TempDir,
}

/// Open a member and serve its Raft API on `port`: the founder of a group,
/// or a node joining the founder's, which listens on `founder_port` and adds
/// it as a learner; a joiner's database opens once its namespace has
/// arrived.
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
    let deployment = Deployment::open_with_oracle(&config, oracle.clone()).unwrap();
    // A founder starts the store's namespace; a joiner receives its group's.
    let engine = Arc::new(match founder {
        None => deployment.single_store().unwrap(),
        Some(_) => deployment.member_engine(),
    });

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
    let addr: std::net::SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
    let context = Arc::clone(raft.connections().unwrap().context());
    let routes = tonic::service::Routes::new(RaftServiceServer::new(raft_handler));
    match founder {
        None => {
            serve_member_port(addr, &context, routes).await;
        }
        Some((founder, _)) => {
            let (listener, ()) = serve_peers_until(addr, &context, routes.clone(), async {
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
            })
            .await;
            serve_member_listener(listener, &context, routes);
        }
    }
    let pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> =
        Arc::new(RaftProposalPipeline::new(Arc::clone(raft.raft())));
    let db = Arc::new(RwLock::new(
        Database::from_engine(dir.path(), engine, oracle, pipeline).unwrap(),
    ));

    Node {
        db,
        raft,
        _dir: dir,
    }
}

/// The id of the node a `CREATE ... RETURN id(n) AS id` made on `db`, run off
/// the async runtime since a write waits on the log.
async fn create_on(db: &Arc<RwLock<Database>>) -> u64 {
    let db = Arc::clone(db);
    tokio::task::spawn_blocking(move || {
        let rows = db
            .write()
            .execute_cypher("CREATE (n:Fresh) RETURN id(n) AS id")
            .unwrap();
        match rows[0].get("id") {
            Some(Value::Int(id)) => u64::try_from(*id).unwrap(),
            other => panic!("no id: {other:?}"),
        }
    })
    .await
    .unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_new_leader_allocates_above_the_restored_identifiers() {
    let ports = [alloc_port(), alloc_port(), alloc_port()];
    let n1 = open_node(1, ports[0], None).await;
    let n2 = open_node(2, ports[1], Some((&n1, ports[0]))).await;
    let _n3 = open_node(3, ports[2], Some((&n1, ports[0]))).await;
    n1.raft.change_membership(vec![1, 2, 3]).await.unwrap();
    tokio::time::sleep(Duration::from_millis(800)).await;

    // Well above the first lease, so a fresh allocator starting at zero would
    // hand these very identifiers out.
    const RESTORED: [u64; 2] = [3, 25_000];
    let dump = format!(
        "{}\n{}",
        format_args!(
            r#"{{"type":"node","id":{},"labels":["Kept"],"properties":{{"n":1}}}}"#,
            RESTORED[0]
        ),
        format_args!(
            r#"{{"type":"node","id":{},"labels":["Kept"],"properties":{{"n":2}}}}"#,
            RESTORED[1]
        ),
    );
    {
        let db = Arc::clone(&n1.db);
        tokio::task::spawn_blocking(move || {
            db.read()
                .restore(
                    BackupFormat::Json,
                    &dump.as_bytes(),
                    &RestoreOptions::default(),
                )
                .unwrap()
        })
        .await
        .unwrap();
    }

    n1.raft.transfer_leadership_to(2).await.unwrap();
    assert!(
        n2.raft.is_leader().await,
        "member 2 leads after the transfer"
    );
    let ceiling = node_lease_ceiling(n2.db.read().engine()).unwrap();
    assert!(
        ceiling >= RESTORED[1],
        "the new leader knows the restored lease: ceiling {ceiling}"
    );

    let fresh = create_on(&n2.db).await;
    assert!(
        fresh > RESTORED[1],
        "the new leader handed out {fresh}, at or below a restored identifier"
    );
}
