//! Regression test: vector search on a Raft follower must return the
//! same results as on the leader.
//!
//! Reproduces a bug observed end-to-end through the gRPC server: after
//! loading vectors and creating a vector index through the leader, a
//! follower serving reads answered vector top-K queries fast but with
//! ~0.2 recall while the leader answered with full recall. The
//! follower's HNSW index does not reflect the replicated data.

#![allow(clippy::unwrap_used, clippy::panic)]

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

/// Deterministic pseudo-random vector for row `i`. The modulus is a
/// prime far above the row count so no two rows share a vector;
/// duplicate vectors would create distance ties whose ordering
/// legitimately differs between the leader's HNSW path and the
/// follower's scan path.
fn vec_for(i: usize, dim: usize) -> Vec<f64> {
    (0..dim)
        .map(|d| {
            // Multiplicative hash mix so values are spread rather than
            // collinear (a linear ramp degenerates HNSW navigation).
            let h = (i.wrapping_mul(2654435761) ^ d.wrapping_mul(40503)) % 99991;
            h as f64 / 99991.0
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

    // Leader answers through HNSW (approximate), follower through its
    // own local plan; even with a follower-side index the graphs are
    // built independently and legitimately differ in topology. The
    // correctness contract is therefore recall overlap, not identical
    // orderings: every probe's top-k sets must overlap >= 70% and the
    // average across probes >= 90%.
    let mut last_state = String::new();
    for _ in 0..30 {
        tokio::time::sleep(Duration::from_millis(500)).await;
        // The server binary refreshes derived state on every applied
        // entry (subscribe_applied task in main); this poll loop stands
        // in for that wiring at the Database level.
        n2.db.refresh_field_interner().unwrap();
        n2.db.refresh_vector_indexes().unwrap();
        let mut total_overlap = 0.0;
        let mut min_overlap = f64::MAX;
        let mut worst = String::new();
        for (qi, qv) in probes.iter().enumerate() {
            let leader_ids = top_ids(&mut n1.db, qv);
            let follower_ids = top_ids(&mut n2.db, qv);
            let leader_set: std::collections::HashSet<i64> = leader_ids.iter().copied().collect();
            let inter = follower_ids
                .iter()
                .filter(|id| leader_set.contains(id))
                .count();
            let overlap = inter as f64 / leader_ids.len().max(1) as f64;
            total_overlap += overlap;
            if overlap < min_overlap {
                min_overlap = overlap;
                worst = format!("probe {qi}: leader={leader_ids:?} follower={follower_ids:?}");
            }
        }
        let avg_overlap = total_overlap / probes.len() as f64;
        last_state = format!("avg_overlap={avg_overlap:.3} min_overlap={min_overlap:.3} {worst}");
        if avg_overlap >= 0.9 && min_overlap >= 0.7 {
            // Converged. The follower must also be serving through its
            // OWN HNSW access path by now (replicated DDL + local
            // rebuild), not the brute-force scan fallback.
            let follower_plan = n2.db.explain_cypher(&probe_q).unwrap();
            assert!(
                follower_plan.contains("HnswScan"),
                "follower converged but still plans without the index:\n{follower_plan}"
            );
            // Vectors written AFTER the follower's index went live must
            // reach its HNSW through the oplog worker (no rebuild, no
            // re-registration). Search for an exact new vector: only an
            // index that ingested the post-convergence write returns it.
            let emb = vec_for(N + 7, DIM)
                .iter()
                .map(|x| format!("{x:.6}"))
                .collect::<Vec<_>>()
                .join(", ");
            n1.db
                .execute_cypher(&format!(
                    "CREATE (n:Item {{ext_id: {}, embedding: [{emb}]}})",
                    N + 7
                ))
                .unwrap();
            for _ in 0..30 {
                tokio::time::sleep(Duration::from_millis(500)).await;
                let ids = top_ids(&mut n2.db, &vec_for(N + 7, DIM));
                if ids.first() == Some(&((N + 7) as i64)) {
                    return; // live tail reached the follower index
                }
            }
            panic!("post-convergence write never reached the follower HNSW");
        }
    }
    panic!("follower vector search never converged to leader recall: {last_state}");
}
