//! The label lookup on a replica: the postings a write stages travel in its
//! log entry, so a follower answers a label scan from its own copy with the
//! same members the leader has.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;
use std::time::Duration;

use coordinode_core::graph::types::Value;
use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_embed::Database;
use coordinode_raft::cluster::RaftNode;
use coordinode_raft::proposal::RaftProposalPipeline;
use coordinode_raft::proto::replication::raft_service_server::RaftServiceServer;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;

use coordinode_test_fixtures::alloc_port;

struct Node {
    db: Database,
    raft: Arc<RaftNode>,
    _dir: tempfile::TempDir,
}

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
    let (raft, raft_handler) = if leader {
        RaftNode::open_cluster_embedded(
            node_id,
            Arc::clone(&engine),
            format!("http://127.0.0.1:{port}"),
        )
        .await
        .unwrap()
    } else {
        RaftNode::open_joining_embedded(node_id, Arc::clone(&engine))
            .await
            .unwrap()
    };
    let raft = Arc::new(raft);
    let pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> =
        Arc::new(RaftProposalPipeline::new(Arc::clone(raft.raft())));
    let db = Database::from_engine(dir.path(), engine, oracle, pipeline).unwrap();
    let addr: std::net::SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(RaftServiceServer::new(raft_handler))
            .serve(addr)
            .await
    });
    Node {
        db,
        raft,
        _dir: dir,
    }
}

fn names(db: &mut Database, query: &str) -> Vec<String> {
    let mut out: Vec<String> = db
        .execute_cypher(query)
        .unwrap_or_else(|e| panic!("{query}: {e}"))
        .iter()
        .map(|row| match row.get("name") {
            Some(Value::String(s)) => s.clone(),
            other => panic!("expected a name, got {other:?}"),
        })
        .collect();
    out.sort();
    out
}

/// A follower's label scan, read from its replicated lookup, returns the
/// members the leader's does, after creates, a label added, a label removed
/// and a delete.
#[tokio::test(flavor = "multi_thread")]
async fn a_follower_scans_a_label_from_its_replicated_lookup() {
    let (p1, p2) = (alloc_port(), alloc_port());
    let mut n1 = open_node(1, p1, true).await;
    let mut n2 = open_node(2, p2, false).await;
    tokio::time::sleep(Duration::from_millis(800)).await;
    n1.raft
        .add_node(2, format!("http://127.0.0.1:{p2}"))
        .await
        .unwrap();
    n1.raft.change_membership(vec![1, 2]).await.unwrap();
    // The leader provisions the lookup, as the server's resumer does.
    n1.db.resume_interrupted_index_builds().expect("provision");

    n1.db
        .execute_cypher("CREATE (:A {name: 'a1'}), (:A:B {name: 'a2'}), (:B {name: 'b1'})")
        .expect("create");
    n1.db
        .execute_cypher("MATCH (n {name: 'b1'}) SET n:A")
        .expect("label added");
    n1.db
        .execute_cypher("MATCH (n {name: 'a2'}) REMOVE n:A")
        .expect("label removed");
    n1.db
        .execute_cypher("MATCH (n {name: 'a1'}) DELETE n")
        .expect("delete");
    let on_leader = names(&mut n1.db, "MATCH (n:A) RETURN n.name AS name");
    assert_eq!(on_leader, ["b1"]);
    assert!(
        n1.db
            .explain_cypher("MATCH (n:A) RETURN n")
            .expect("explain")
            .contains("from the built-in label lookup"),
        "the leader reads the lookup"
    );

    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    loop {
        n2.db.refresh_btree_indexes().expect("refresh");
        let ready = n2
            .db
            .explain_cypher("MATCH (n:A) RETURN n")
            .expect("explain")
            .contains("from the built-in label lookup");
        if ready && names(&mut n2.db, "MATCH (n:A) RETURN n.name AS name") == on_leader {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the follower never answered from its lookup"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert_eq!(
        names(&mut n2.db, "MATCH (n:B) RETURN n.name AS name"),
        names(&mut n1.db, "MATCH (n:B) RETURN n.name AS name")
    );
}
