//! Regression test: vector search on a Raft follower must return the
//! same results as on the leader.
//!
//! Reproduces a bug observed end-to-end through the gRPC server: after
//! loading vectors and creating a vector index through the leader, a
//! follower serving reads answered vector top-K queries fast but with
//! ~0.2 recall while the leader answered with full recall. The
//! follower's HNSW index does not reflect the replicated data.

#![allow(clippy::unwrap_used)]

use std::sync::Arc;
use std::time::Duration;

use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_embed::Database;
use coordinode_raft::cluster::RaftNode;
use coordinode_raft::proposal::RaftProposalPipeline;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;

fn alloc_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    port
}

struct ClusterNode {
    db: Database,
    engine: Arc<StorageEngine>,
    oracle: Arc<TimestampOracle>,
    _node: Arc<RaftNode>,
    _dir: tempfile::TempDir,
}

async fn open_node(node_id: u64, port: u16, leader: bool) -> ClusterNode {
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
    let listen_addr: std::net::SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();

    let node = if leader {
        RaftNode::open_cluster(
            node_id,
            Arc::clone(&engine),
            listen_addr,
            format!("http://127.0.0.1:{port}"),
        )
        .await
        .unwrap()
    } else {
        RaftNode::open_joining(node_id, Arc::clone(&engine), listen_addr)
            .await
            .unwrap()
    };
    let node = Arc::new(node);

    let pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> =
        Arc::new(RaftProposalPipeline::new(Arc::clone(node.raft())));
    let db =
        Database::from_engine(dir.path(), Arc::clone(&engine), oracle.clone(), pipeline).unwrap();

    ClusterNode {
        db,
        engine,
        oracle,
        _node: node,
        _dir: dir,
    }
}

/// Deterministic pseudo-random unit-ish vector for row `i`.
fn vec_for(i: usize, dim: usize) -> Vec<f64> {
    (0..dim)
        .map(|d| {
            let x = ((i * 31 + d * 7 + 13) % 1000) as f64;
            x / 1000.0
        })
        .collect()
}

fn top_ids(db: &mut Database, qv: &[f64]) -> Vec<i64> {
    let qv_str = qv
        .iter()
        .map(|x| format!("{x:.6}"))
        .collect::<Vec<_>>()
        .join(", ");
    let cypher = format!(
        "MATCH (n:Item) \
         WITH *, vector_distance(n.embedding, [{qv_str}]) AS d \
         ORDER BY d ASC LIMIT 10 RETURN n.ext_id AS ext_id"
    );
    db.execute_cypher(&cypher)
        .unwrap()
        .iter()
        .map(|row| row.get("ext_id").unwrap().as_int().unwrap())
        .collect()
}

/// After loading vectors and creating a vector index through the
/// leader, a follower must answer vector top-K identically to the
/// leader once replication and index backfill settle.
#[tokio::test(flavor = "multi_thread")]
async fn follower_vector_search_matches_leader() {
    const N: usize = 1500; // above the brute-force threshold so the
                           // planner uses the HNSW access path
    const DIM: usize = 8;

    let p1 = alloc_port();
    let p2 = alloc_port();

    let mut n1 = open_node(1, p1, true).await;
    let mut n2 = open_node(2, p2, false).await;

    tokio::time::sleep(Duration::from_millis(800)).await;
    n1._node
        .add_node(2, format!("http://127.0.0.1:{p2}"))
        .await
        .unwrap();
    n1._node.change_membership(vec![1, 2]).await.unwrap();
    tokio::time::sleep(Duration::from_millis(800)).await;

    // Load through the leader in batches.
    for chunk_start in (0..N).step_by(250) {
        let rows = (chunk_start..(chunk_start + 250).min(N))
            .map(|i| {
                let emb = vec_for(i, DIM)
                    .iter()
                    .map(|x| format!("{x:.6}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{{ext_id: {i}, embedding: [{emb}]}}")
            })
            .collect::<Vec<_>>()
            .join(", ");
        n1.db
            .execute_cypher(&format!(
                "UNWIND [{rows}] AS row \
                 CREATE (n:Item {{ext_id: row.ext_id, embedding: row.embedding}})"
            ))
            .unwrap();
    }

    n1.db
        .execute_cypher(
            "CREATE VECTOR INDEX item_emb ON :Item(embedding) \
             OPTIONS {m: 16, ef_construction: 100, metric: \"euclidean\", dimensions: 8}",
        )
        .unwrap();

    // Let replication + asynchronous index backfill settle on both
    // nodes. Generous bound; the assertion below polls.
    let probes: Vec<Vec<f64>> = (0..20).map(|q| vec_for(q * 71 + 5, DIM)).collect();

    // Diagnostic precondition: the follower must SEE the replicated
    // rows at all. If this count is zero the bug is in follower read
    // visibility (MVCC timestamp not advanced on apply), not in the
    // vector index.
    let leader_count = n1
        .db
        .execute_cypher("MATCH (n:Item) RETURN count(n) AS c")
        .unwrap();
    let follower_count = n2
        .db
        .execute_cypher("MATCH (n:Item) RETURN count(n) AS c")
        .unwrap();
    eprintln!("leader count rows: {leader_count:?}");
    eprintln!("follower count rows: {follower_count:?}");
    eprintln!(
        "oracle ts: leader={:?} follower={:?}",
        n1.oracle.current(),
        n2.oracle.current()
    );
    let engine_rows = |e: &StorageEngine| -> usize {
        use coordinode_storage::engine::partition::Partition;
        e.prefix_scan(Partition::Node, b"").unwrap().count()
    };
    eprintln!(
        "engine Nodes rows: leader={} follower={}",
        engine_rows(&n1.engine),
        engine_rows(&n2.engine)
    );

    let probe_q = {
        let qv_str = probes[0]
            .iter()
            .map(|x| format!("{x:.6}"))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "MATCH (n:Item) \
             WITH *, vector_distance(n.embedding, [{qv_str}]) AS d \
             ORDER BY d ASC LIMIT 10 RETURN n.ext_id AS ext_id"
        )
    };
    eprintln!("leader EXPLAIN: {:?}", n1.db.explain_cypher(&probe_q));
    eprintln!("follower EXPLAIN: {:?}", n2.db.explain_cypher(&probe_q));
    // Property type discriminator: if the replicated embedding
    // deserialises as a different value type on the follower,
    // vector_distance yields null there and brute-force drops all rows.
    let type_q = "MATCH (n:Item) WHERE n.ext_id = 5 RETURN n.embedding AS e LIMIT 1";
    eprintln!("leader embedding: {:?}", n1.db.execute_cypher(type_q));
    eprintln!("follower embedding: {:?}", n2.db.execute_cypher(type_q));
    let dist_q = "MATCH (n:Item) WHERE n.ext_id = 5 \
                  RETURN vector_distance(n.embedding, [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) AS d";
    eprintln!("leader dist: {:?}", n1.db.execute_cypher(dist_q));
    eprintln!("follower dist: {:?}", n2.db.execute_cypher(dist_q));
    let bare_q = "MATCH (n:Item) RETURN n.ext_id AS e LIMIT 3";
    eprintln!("leader bare props: {:?}", n1.db.execute_cypher(bare_q));
    eprintln!("follower bare props: {:?}", n2.db.execute_cypher(bare_q));

    let mut last_mismatch = String::new();
    for _ in 0..30 {
        tokio::time::sleep(Duration::from_millis(500)).await;
        let mut all_match = true;
        last_mismatch.clear();
        for (qi, qv) in probes.iter().enumerate() {
            let leader_ids = top_ids(&mut n1.db, qv);
            let follower_ids = top_ids(&mut n2.db, qv);
            if leader_ids != follower_ids {
                all_match = false;
                last_mismatch =
                    format!("probe {qi}: leader={leader_ids:?} follower={follower_ids:?}");
                break;
            }
        }
        if all_match {
            return; // follower converged to leader results
        }
    }
    panic!("follower vector search never matched leader: {last_mismatch}");
}
