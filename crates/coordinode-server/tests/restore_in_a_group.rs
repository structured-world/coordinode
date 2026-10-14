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
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::metadata::node_lease_ceiling;
use parking_lot::RwLock;

use coordinode_test_fixtures::{TestPki, alloc_port, serve_member_port};

/// The CA the members are issued their identities by.
static PKI: std::sync::LazyLock<TestPki> = std::sync::LazyLock::new(TestPki::new);

struct Node {
    db: Arc<RwLock<Database>>,
    raft: Arc<RaftNode>,
    _dir: tempfile::TempDir,
}

/// Open a member and serve its Raft API on `port`: the founder of a group,
/// or a node joining the founder's, which listens on `founder_port`.
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

    let addr: std::net::SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
    let context = Arc::clone(raft.connections().unwrap().context());
    serve_member_port(
        addr,
        &context,
        tonic::service::Routes::new(RaftServiceServer::new(raft_handler)),
    )
    .await;

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

    tokio::time::sleep(Duration::from_millis(800)).await;
    for (id, port) in [(2, ports[1]), (3, ports[2])] {
        n1.raft
            .add_node(id, format!("127.0.0.1:{port}"), PKI.identity(id))
            .await
            .unwrap();
    }
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
