//! Regression test: vector search on a Raft follower must return the
//! same results as on the leader.
//!
//! Reproduces a bug observed end-to-end through the gRPC server: after
//! loading vectors and creating a vector index through the leader, a
//! follower serving reads answered vector top-K queries fast but with
//! ~0.2 recall while the leader answered with full recall. The
//! follower's HNSW index does not reflect the replicated data.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;
use std::time::Duration;

use coordinode_core::graph::types::Value;
use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_embed::Database;
use coordinode_raft::cluster::RaftNode;
use coordinode_raft::proposal::RaftProposalPipeline;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;

use coordinode_test_fixtures::alloc_port;

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

    let pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> = Arc::new(
        RaftProposalPipeline::new(Arc::clone(node.raft())).with_closure(Arc::clone(&engine)),
    );
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

/// Deterministic vector for row `i` whose coordinates are independent uniform
/// draws (splitmix64). `vec_for` derives every coordinate from one hash of
/// `i`, which puts the rows near a curve; an approximate index then misses
/// some of its own points, so it is no base for asserting that every vector
/// is found.
fn uniform_vec(i: usize, dim: usize) -> Vec<f64> {
    (0..dim as u64)
        .map(|d| {
            let mut z = (i as u64)
                .wrapping_mul(dim as u64)
                .wrapping_add(d)
                .wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            // Rounded as the query text will carry it, so a stored vector
            // and its query are the same point.
            ((z >> 11) as f64 / (1u64 << 53) as f64 * 1e6).round() / 1e6
        })
        .collect()
}

/// `v` as a query value, the list a literal of its numbers would be.
fn vector_value(v: &[f64]) -> Value {
    Value::Array(v.iter().map(|x| Value::Float(*x)).collect())
}

/// Items whose source row or actual index membership has not arrived yet.
/// ANN recall is independent of this complete-delivery invariant.
fn unindexed_items(db: &Database, end: usize) -> Vec<usize> {
    let rows = db
        .execute_cypher_shared(
            "MATCH (n:Item) RETURN id(n) AS node_id, n.ext_id AS ext_id",
            None,
            None,
            None,
            None,
        )
        .unwrap()
        .rows;
    let handle = db.vector_index_registry().get("Item", "embedding").unwrap();
    let graph = handle.read().unwrap();
    let mut indexed = vec![false; end];
    let mut seen = std::collections::HashSet::new();
    for row in rows {
        let id = row.get("node_id").and_then(Value::as_int).unwrap() as u64;
        let ext = row.get("ext_id").and_then(Value::as_int).unwrap() as usize;
        assert!(ext < end, "unexpected source item {ext}");
        assert!(seen.insert(ext), "duplicate source item {ext}");
        indexed[ext] = graph.contains(id);
    }
    indexed
        .into_iter()
        .enumerate()
        .filter_map(|(ext, present)| if present { None } else { Some(ext) })
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
        // The server binary brings replicated vector indexes live on every
        // applied entry (subscribe_applied task in main); this poll loop
        // stands in for that wiring at the Database level.
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

/// A follower builds an index another member defined in the background, while
/// the leader keeps writing. Every vector replicated during that build must be
/// in the follower's index once it finishes: those entries apply on the
/// follower beside its scan, some below the snapshot the scan read.
#[tokio::test(flavor = "multi_thread")]
async fn writes_replicated_during_a_follower_build_reach_its_index() {
    const N: usize = 6000;
    const DIM: usize = 8;

    let p1 = alloc_port();
    let p2 = alloc_port();
    let mut n1 = open_node(1, p1, true).await;
    let n2 = open_node(2, p2, false).await;

    // A membership change needs a leader; it returns once committed.
    for _ in 0..150 {
        if n1._node.is_leader().await {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    n1._node
        .add_node(2, format!("http://127.0.0.1:{p2}"))
        .await
        .unwrap();
    n1._node.change_membership(vec![1, 2]).await.unwrap();

    // Rows go in as one parameter: the statement text stays the same, so it
    // is parsed once and planned from the cache after.
    let create_rows = |db: &mut Database, range: std::ops::Range<usize>| {
        let rows = Value::Array(
            range
                .map(|i| {
                    Value::Map(std::collections::BTreeMap::from([
                        ("ext_id".to_string(), Value::Int(i as i64)),
                        ("embedding".to_string(), vector_value(&uniform_vec(i, DIM))),
                    ]))
                })
                .collect(),
        );
        db.execute_cypher_with_params(
            "UNWIND $rows AS row \
             CREATE (n:Item {ext_id: row.ext_id, embedding: row.embedding})",
            std::collections::HashMap::from([("rows".to_string(), rows)]),
        )
        .unwrap();
    };
    let t = std::time::Instant::now();
    for start in (0..N).step_by(500) {
        create_rows(&mut n1.db, start..start + 500);
    }
    eprintln!("PHASE load {:?}", t.elapsed());
    let t = std::time::Instant::now();
    n1.db
        .execute_cypher(
            "CREATE VECTOR INDEX item_emb ON :Item(embedding) \
             OPTIONS {m: 16, ef_construction: 100, metric: \"euclidean\", dimensions: 8}",
        )
        .unwrap();
    eprintln!("PHASE leader index {:?}", t.elapsed());
    let t = std::time::Instant::now();

    // Bring the index up on the follower once the definition has reached it,
    // as the server does on every applied entry.
    let mut started = false;
    for _ in 0..40 {
        if n2.db.refresh_vector_indexes().unwrap() > 0 {
            started = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(started, "the index definition never reached the follower");
    eprintln!("PHASE follower start {:?}", t.elapsed());
    let t = std::time::Instant::now();

    // Keep writing through the leader for as long as the follower builds.
    let mut late = N;
    let mut during = 0;
    while !n2.db.index_builds().is_empty() {
        create_rows(&mut n1.db, late..late + 20);
        late += 20;
        during += 1;
    }
    assert!(
        during > 0,
        "the follower's build finished before any write landed beside it"
    );

    eprintln!("PHASE follower build {:?} ({during} writes)", t.elapsed());
    let t = std::time::Instant::now();
    // Then keep writing with no build running, which the follower's index
    // takes from its maintenance of applied entries.
    let steady = late..late + 1000;
    for start in steady.clone().step_by(20) {
        create_rows(&mut n1.db, start..start + 20);
    }
    eprintln!("PHASE steady {:?}", t.elapsed());
    let t = std::time::Instant::now();

    // Every committed item must be physically indexed on both members.
    // ANN candidates do not prove complete membership (or lost writes): a
    // bounded HNSW beam may miss even its query's own point. Poll membership,
    // then inspect every live result and payload through the exact index path.
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    let mut missing_follower = unindexed_items(&n2.db, steady.end);
    let mut missing_leader = unindexed_items(&n1.db, steady.end);
    while (!missing_follower.is_empty() || !missing_leader.is_empty())
        && std::time::Instant::now() < deadline
    {
        tokio::time::sleep(Duration::from_millis(200)).await;
        missing_follower = unindexed_items(&n2.db, steady.end);
        missing_leader = unindexed_items(&n1.db, steady.end);
    }
    assert!(
        missing_leader.is_empty() && missing_follower.is_empty(),
        "of {} vectors ({} written during the build), missing from leader {missing_leader:?}, follower {missing_follower:?}",
        steady.end,
        late - N,
    );
    eprintln!("PHASE complete membership {:?}", t.elapsed());

    for (member, db) in [("leader", &n1.db), ("follower", &n2.db)] {
        let rows = db
            .execute_cypher_shared(
                "MATCH (n:Item) RETURN id(n) AS node_id, n.ext_id AS ext_id",
                None,
                None,
                None,
                None,
            )
            .unwrap()
            .rows;
        assert_eq!(rows.len(), steady.end, "{member} source membership");
        let expected: std::collections::HashMap<_, _> = rows
            .iter()
            .map(|row| {
                let id = row.get("node_id").and_then(Value::as_int).unwrap() as u64;
                let ext = row.get("ext_id").and_then(Value::as_int).unwrap() as usize;
                let vector: Vec<f32> = uniform_vec(ext, DIM)
                    .into_iter()
                    .map(|value| value as f32)
                    .collect();
                (id, vector)
            })
            .collect();
        assert_eq!(expected.len(), steady.end, "{member} unique source IDs");
        let handle = db.vector_index_registry().get("Item", "embedding").unwrap();
        let graph = handle.read().unwrap();
        assert_eq!(graph.len(), steady.end, "{member} live index count");
        // Zero and every coordinate basis query check every stored vector's
        // exact scores and complete live ID set. This detects wrong payloads
        // and non-live entries as well as a missing mapping, without assuming
        // perfect ANN recall or weakening the all-writes requirement.
        for axis in 0..=DIM {
            let mut query = vec![0.0; DIM];
            if axis < DIM {
                query[axis] = 1.0;
            }
            let hits =
                graph.search_with_mode(&query, steady.end, coordinode_vector::SearchMode::Exact);
            assert_eq!(hits.len(), expected.len(), "{member} exact live ID count");
            let mut seen = std::collections::HashSet::new();
            for hit in hits {
                assert!(
                    seen.insert(hit.id),
                    "{member} duplicate indexed ID {}",
                    hit.id
                );
                let vector = expected.get(&hit.id).expect("unexpected indexed ID");
                assert_eq!(
                    hit.score,
                    coordinode_vector::metrics::euclidean_distance_squared(&query, vector),
                    "{member} wrong indexed payload for {}, axis {axis}",
                    hit.id,
                );
            }
        }
    }
    eprintln!("PHASE exact payload qualification {:?}", t.elapsed());
    let plan = n2
        .db
        .explain_cypher(
            "MATCH (n:Item) WITH *, vector_distance(n.embedding, [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) AS d \
             ORDER BY d ASC LIMIT 10 RETURN n.ext_id",
        )
        .unwrap();
    assert!(
        plan.contains("HnswScan"),
        "the follower answered without its index:\n{plan}"
    );
}

/// AFTER COMMIT trigger in a real 2-node Raft cluster: the event the leader
/// enqueues is replicated, the leader's dispatch executes the body through the
/// Raft pipeline, and the body's effect (an AuditEntry node) replicates to the
/// follower. The queue and the body's writes both go through consensus, so
/// only the leader runs the body and every node sees its effect.
#[tokio::test(flavor = "multi_thread")]
async fn after_commit_trigger_fires_on_leader_and_replicates_to_follower() {
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

    // Install an AFTER COMMIT trigger through the leader (definition replicates).
    n1.db
        .execute_cypher(
            "CREATE TRIGGER audit_user ON :User CREATE AFTER COMMIT \
             EXECUTE CREATE (e:AuditEntry {action: $event})",
        )
        .unwrap();

    // A user write on the leader enqueues a durable, replicated event. The
    // server's leader-gated worker is not running in this harness, so we invoke
    // the dispatch the worker would call — on the leader, where writes commit.
    n1.db.execute_cypher("CREATE (u:User {id: 1})").unwrap();
    assert_eq!(
        n1.db.after_commit_pending_count(),
        1,
        "leader enqueued the after-commit event"
    );

    let report = n1.db.dispatch_after_commit_triggers();
    assert_eq!(report.fired, 1, "leader dispatch executed the body once");
    assert_eq!(n1.db.after_commit_pending_count(), 0);

    // Leader sees the audit node immediately.
    let leader_audit = n1
        .db
        .execute_cypher("MATCH (e:AuditEntry) RETURN e.action AS act")
        .unwrap();
    assert_eq!(leader_audit.len(), 1);
    assert_eq!(
        leader_audit[0].get("act"),
        Some(&Value::String("CREATE".into()))
    );

    // The body's write replicates: the follower must observe the same audit
    // node once the entry applies. Its statements read the field dictionary
    // as the applies left it, so the new `action` binding needs no refresh
    // task. Poll with a generous bound.
    for _ in 0..40 {
        tokio::time::sleep(Duration::from_millis(200)).await;
        let follower_audit = n2
            .db
            .execute_cypher("MATCH (e:AuditEntry) RETURN e.action AS act")
            .unwrap();
        if follower_audit.len() == 1
            && follower_audit[0].get("act") == Some(&Value::String("CREATE".into()))
        {
            return; // replicated trigger effect reached the follower
        }
    }
    panic!("after-commit trigger effect never replicated to the follower");
}

/// Titles a text search on `db` finds for `words`.
fn text_hits(db: &mut Database, words: &str) -> Vec<String> {
    let mut titles: Vec<String> = db
        .execute_cypher(&format!(
            "MATCH (n:Article) WHERE text_match(n.body, '{words}') RETURN n.title AS t"
        ))
        .unwrap()
        .iter()
        .filter_map(|row| row.get("t").and_then(|v| v.as_str()).map(str::to_string))
        .collect();
    titles.sort();
    titles
}

/// A follower's full-text search sees what the leader committed, including
/// a later change of the text, and nothing a leader transaction rolled
/// back. The follower never runs those statements: its text indexes follow
/// the entries it applies. Before, it updated them only for statements it
/// ran itself, so every leader write after it opened was missing.
#[tokio::test(flavor = "multi_thread")]
async fn follower_text_search_follows_leader_commits() {
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

    n1.db
        .execute_cypher("CREATE TEXT INDEX article_body ON :Article(body)")
        .unwrap();
    n1.db
        .execute_cypher("CREATE (:Article {title: 'a', body: 'replicated words'})")
        .unwrap();
    n1.db
        .execute_cypher("MATCH (n:Article {title: 'a'}) SET n.body = 'revised words'")
        .unwrap();
    let tx = n1.db.begin_transaction();
    n1.db
        .execute_in_transaction(
            tx,
            "CREATE (:Article {title: 'b', body: 'abandoned words'})",
            None,
        )
        .unwrap();
    n1.db.rollback_transaction(tx).unwrap();

    // The server brings replicated text index definitions live on every
    // applied entry; this poll stands in for that task at the Database level.
    let mut last = Vec::new();
    for _ in 0..50 {
        tokio::time::sleep(Duration::from_millis(200)).await;
        n2.db.refresh_text_indexes().unwrap();
        if !n2.db.text_index_registry().has_index("Article", "body") {
            continue;
        }
        last = text_hits(&mut n2.db, "revised");
        if last == ["a"] {
            break;
        }
    }
    assert_eq!(
        last,
        ["a"],
        "the follower misses the leader's committed text"
    );
    assert!(
        text_hits(&mut n2.db, "replicated").is_empty(),
        "the follower still finds the text the leader replaced"
    );
    assert!(
        text_hits(&mut n2.db, "abandoned").is_empty(),
        "the follower finds text of a rolled-back transaction"
    );
}

/// A follower's vector freshness watermark is a fence: a write at or below
/// it is in the index. The leader can hold a commit whose timestamp is
/// allocated and not yet in the log while a later-stamped commit replicates
/// ahead of it; the follower applies the later one first and must not claim
/// the earlier timestamp, since the commit stamped with it may still arrive.
/// Checked on the follower's index registry and on what its vector search
/// RPC reports to a client (body and `coordinode-indexed-hlc` header).
#[tokio::test(flavor = "multi_thread")]
async fn a_follower_watermark_stays_below_a_commit_the_leader_still_holds() {
    use coordinode_core::graph::node::NodeRecord;
    use coordinode_core::txn::proposal::{
        Mutation, PartitionId, ProposalIdGenerator, ProposalPipeline, RaftProposal,
        fresh_proposal_id_base,
    };
    use coordinode_core::txn::timestamp::Timestamp;
    use coordinode_server::proto::query::vector_service_client::VectorServiceClient;
    use coordinode_server::proto::query::vector_service_server::VectorServiceServer;
    use coordinode_server::proto::query::{VectorSearchRequest, VectorSearchResponse};
    use coordinode_server::services::vector::VectorServiceImpl;
    use coordinode_storage::engine::partition::Partition;

    let p1 = alloc_port();
    let p2 = alloc_port();
    let mut n1 = open_node(1, p1, true).await;
    let ClusterNode {
        db: n2_db,
        engine: n2_engine,
        _node: _n2_node,
        _dir: _n2_dir,
        ..
    } = open_node(2, p2, false).await;
    let n2_db = Arc::new(parking_lot::RwLock::new(n2_db));
    // Exercise the follower's real gRPC response, including wire metadata,
    // rather than invoking the service implementation directly.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (stop_rpc, stopped_rpc) = tokio::sync::oneshot::channel();
    let service = VectorServiceImpl::new(Arc::clone(&n2_db));
    let rpc = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(VectorServiceServer::new(service))
            .serve_with_incoming_shutdown(
                tokio_stream::wrappers::TcpListenerStream::new(listener),
                async {
                    let _ = stopped_rpc.await;
                },
            )
            .await
            .unwrap();
    });
    let channel = tonic::transport::Endpoint::from_shared(format!("http://{addr}"))
        .unwrap()
        .connect_timeout(Duration::from_secs(2))
        .timeout(Duration::from_secs(5))
        .connect()
        .await
        .unwrap();
    let mut n2_rpc = VectorServiceClient::new(channel);
    // The watermark a client is served: the response body and its header
    // must agree.
    let served = |resp: tonic::Response<VectorSearchResponse>| {
        let header: u64 = resp
            .metadata()
            .get("coordinode-indexed-hlc")
            .expect("indexed-hlc header present")
            .to_str()
            .unwrap()
            .parse()
            .unwrap();
        let body = resp
            .into_inner()
            .index_health
            .expect("the follower serves a managed index")
            .indexed_hlc;
        assert_eq!(header, body, "header and body disagree");
        body
    };
    let search = || {
        tonic::Request::new(VectorSearchRequest {
            label: "Item".to_string(),
            property: "embedding".to_string(),
            query_vector: Some(coordinode_server::proto::common::Vector {
                values: vec![0.0, 0.0],
            }),
            top_k: 1,
            metric: 0,
        })
    };

    tokio::time::sleep(Duration::from_millis(800)).await;
    n1._node
        .add_node(2, format!("http://127.0.0.1:{p2}"))
        .await
        .unwrap();
    n1._node.change_membership(vec![1, 2]).await.unwrap();

    n1.db
        .execute_cypher(
            "CREATE VECTOR INDEX item_emb ON :Item(embedding) \
             OPTIONS {m: 16, ef_construction: 100, metric: \"euclidean\", dimensions: 2}",
        )
        .unwrap();
    n1.db
        .execute_cypher("CREATE (:Item {ext_id: 0, embedding: [0.0, 0.0]})")
        .unwrap();
    let watermark = |db: &Database| {
        db.vector_index_registry()
            .health_handle("Item", "embedding")
            .map(|h| h.indexed_hlc())
    };
    // The follower brings the replicated index live and folds the first
    // write into it.
    let mut settled = false;
    for _ in 0..50 {
        tokio::time::sleep(Duration::from_millis(200)).await;
        n2_db.read().refresh_vector_indexes().unwrap();
        if watermark(&n2_db.read()).is_some_and(|w| w > 0) {
            settled = true;
            break;
        }
    }
    assert!(settled, "the follower never folded the first write");

    // A commit the leader has stamped and not yet proposed, as one between
    // its admission and its proposal is.
    let held_key = coordinode_core::graph::node::encode_node_key(
        1,
        coordinode_core::graph::node::NodeId::from_raw(u64::MAX - 1),
    );
    let (held_ts, held) = n1
        .engine
        .pending_commits()
        .admit_allocated(
            || n1.oracle.next().as_raw(),
            vec![(Partition::Node, held_key.clone())],
            vec![],
        )
        .expect("admit the held commit");

    // A later-stamped commit replicates and applies on the follower first.
    n1.db
        .execute_cypher("CREATE (:Item {ext_id: 1, embedding: [1.0, 1.0]})")
        .unwrap();
    let mut applied = false;
    for _ in 0..50 {
        tokio::time::sleep(Duration::from_millis(200)).await;
        let found = n2_engine.prefix_scan(Partition::Node, b"").unwrap().count();
        if found >= 2 {
            applied = true;
            break;
        }
    }
    assert!(applied, "the later commit never reached the follower");
    // Long enough for the follower's worker to fold what it applied.
    tokio::time::sleep(Duration::from_secs(1)).await;
    let claimed = watermark(&n2_db.read()).expect("the follower holds the index");
    assert!(
        claimed < held_ts,
        "the follower claims {claimed}, at or past {held_ts}, which the leader's held commit is stamped with"
    );
    let reported = served(n2_rpc.vector_search(search()).await.unwrap());
    assert!(
        reported < held_ts,
        "the follower's RPC reports {reported}, at or past the held commit {held_ts}"
    );

    // The earlier commit lands after the later one. No subsequent user write
    // may be needed to close its timestamp and publish index coverage.
    let fields = n1.db.interner().unwrap();
    let mut record = NodeRecord::new("Item");
    record.set(fields.lookup("ext_id").unwrap(), Value::Int(2));
    record.set(
        fields.lookup("embedding").unwrap(),
        vector_value(&[2.0, 2.0]),
    );
    n1._node
        .pipeline()
        .propose_and_wait(&RaftProposal {
            id: ProposalIdGenerator::with_base(fresh_proposal_id_base()).next(),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: held_key.clone(),
                value: record.to_msgpack().unwrap(),
            }],
            commit_ts: Timestamp::from_raw(held_ts),
            start_ts: Timestamp::from_raw(held_ts),
            bypass_rate_limiter: false,
        })
        .unwrap();
    drop(held);
    let mut passed = false;
    for _ in 0..50 {
        tokio::time::sleep(Duration::from_millis(200)).await;
        if watermark(&n2_db.read()).is_some_and(|w| w >= held_ts) {
            passed = true;
            break;
        }
    }
    assert!(
        passed,
        "the follower watermark never passed the released commit"
    );
    let reported = served(n2_rpc.vector_search(search()).await.unwrap());
    assert!(
        reported >= held_ts,
        "the follower's RPC still reports {reported}, below the released commit {held_ts}"
    );
    assert!(
        n2_engine.get(Partition::Node, &held_key).unwrap().is_some(),
        "coverage passed a commit absent from the follower"
    );
    let mut request = search().into_inner();
    request.query_vector.as_mut().unwrap().values = vec![2.0, 2.0];
    let response = n2_rpc
        .vector_search(tonic::Request::new(request))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(
        response
            .results
            .first()
            .and_then(|r| r.node.as_ref())
            .map(|n| n.node_id),
        Some(u64::MAX - 1),
        "the late commit is missing from the served index"
    );
    drop(n2_rpc);
    stop_rpc.send(()).unwrap();
    rpc.await.unwrap();
}
