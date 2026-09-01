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
use coordinode_server::proto::query;
use coordinode_server::services::cypher::CypherServiceImpl;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use parking_lot::RwLock;

fn alloc_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    port
}

struct Node {
    db: Arc<RwLock<Database>>,
    raft: Arc<RaftNode>,
    _dir: tempfile::TempDir,
}

/// Open a node and serve both its Raft and its Cypher API on ONE port, the way
/// a deployed node does. That is what makes forwarding meaningful: the address
/// a peer is told to reach in the membership is the same address a client
/// reaches, so a node can pass a client's request to the leader by the address
/// it already knows.
async fn open_node(node_id: u64, port: u16, leader: bool) -> Node {
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
    let advertise = format!("http://127.0.0.1:{port}");

    let (raft, raft_handler) = if leader {
        RaftNode::open_cluster_embedded(node_id, Arc::clone(&engine), advertise)
            .await
            .unwrap()
    } else {
        // A joining node learns its own advertised address from the leader's
        // add_node call, so it takes none here.
        let _ = advertise;
        RaftNode::open_joining_embedded(node_id, Arc::clone(&engine))
            .await
            .unwrap()
    };
    let raft = Arc::new(raft);

    let pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> =
        Arc::new(RaftProposalPipeline::new(Arc::clone(raft.raft())));
    let db = Arc::new(RwLock::new(
        Database::from_engine(dir.path(), engine, oracle, pipeline).unwrap(),
    ));

    let cypher = CypherServiceImpl::new(
        Arc::clone(&db),
        Arc::new(coordinode_query::advisor::QueryRegistry::new()),
        Arc::new(coordinode_query::advisor::nplus1::NPlus1Detector::new()),
    )
    .with_raft_node(Arc::clone(&raft));

    let addr: std::net::SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(RaftServiceServer::new(raft_handler))
            .add_service(query::cypher_service_server::CypherServiceServer::new(
                cypher,
            ))
            .serve(addr)
            .await
    });

    Node {
        db,
        raft,
        _dir: dir,
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_write_sent_to_a_follower_is_carried_to_the_leader() {
    let (p1, p2) = (alloc_port(), alloc_port());

    let n1 = open_node(1, p1, true).await;
    let _n2 = open_node(2, p2, false).await;

    tokio::time::sleep(Duration::from_millis(800)).await;
    n1.raft
        .add_node(2, format!("http://127.0.0.1:{p2}"))
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
