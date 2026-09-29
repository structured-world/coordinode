//! Concurrent writes over gRPC to a replicated node all complete.
//!
//! A write waits for its commit through Raft, and some writes wait on a lock
//! another write holds while it waits for its own commit (the first CREATEs
//! wait for the NodeId lease one of them is being granted). A request handler
//! that did that waiting on a runtime worker held the worker: with as many
//! requests in flight as the runtime has workers, every worker waited on the
//! lock, no worker was left to run Raft, and the commit the lock holder
//! waited for never came.

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
use coordinode_test_fixtures::alloc_port;
use parking_lot::RwLock;

/// More requests in flight than the runtime has workers, on a freshly opened
/// node (no NodeId lease in hand yet), and every one of them commits.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn more_writes_than_workers_all_commit() {
    let port = alloc_port();
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
    let (raft, raft_handler) =
        RaftNode::open_cluster_embedded(1, Arc::clone(&engine), format!("http://127.0.0.1:{port}"))
            .await
            .unwrap();
    let raft = Arc::new(raft);
    for _ in 0..150 {
        if raft.is_leader().await {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
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
    tokio::time::sleep(Duration::from_millis(300)).await;

    let writes: Vec<_> = (0..8)
        .map(|i| {
            tokio::spawn(async move {
                let mut client = query::cypher_service_client::CypherServiceClient::connect(
                    format!("http://127.0.0.1:{port}"),
                )
                .await
                .unwrap();
                client
                    .execute_cypher(query::ExecuteCypherRequest {
                        query: format!("CREATE (:W {{i: {i}}})"),
                        ..Default::default()
                    })
                    .await
            })
        })
        .collect();
    let all = tokio::time::timeout(Duration::from_secs(30), async {
        let mut done = 0;
        for w in writes {
            w.await.unwrap().expect("the write commits");
            done += 1;
        }
        done
    })
    .await;
    assert_eq!(
        all.ok(),
        Some(8),
        "concurrent writes stalled: a handler held a runtime worker while it waited"
    );

    let rows = tokio::task::spawn_blocking(move || {
        db.read()
            .execute_cypher_shared("MATCH (n:W) RETURN count(n) AS n", None, None, None, None)
            .unwrap()
            .rows
    })
    .await
    .unwrap();
    assert_eq!(
        rows[0].get("n"),
        Some(&coordinode_core::graph::types::Value::Int(8))
    );
    raft.shutdown().await.unwrap();
}
