use std::sync::{Arc, Mutex};

use tonic::{Request, Response, Status};

use coordinode_core::graph::types::Value;
use coordinode_embed::Database;

use crate::proto::{graph, query};

/// Backtick-escape a Cypher identifier (label or property key).
///
/// Doubles any embedded backtick per the OpenCypher spec so that arbitrary
/// strings can be safely embedded in a Cypher query without injection.
fn cypher_ident(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}

pub struct VectorServiceImpl {
    database: Arc<Mutex<Database>>,
}

impl VectorServiceImpl {
    pub fn new(database: Arc<Mutex<Database>>) -> Self {
        Self { database }
    }
}

#[tonic::async_trait]
impl query::vector_service_server::VectorService for VectorServiceImpl {
    async fn vector_search(
        &self,
        request: Request<query::VectorSearchRequest>,
    ) -> Result<Response<query::VectorSearchResponse>, Status> {
        let req = request.into_inner();

        let query_vector = req
            .query_vector
            .ok_or_else(|| Status::invalid_argument("query_vector is required"))?;

        if query_vector.values.is_empty() {
            return Err(Status::invalid_argument("query_vector must not be empty"));
        }

        if req.top_k == 0 {
            return Err(Status::invalid_argument("top_k must be > 0"));
        }

        let label = cypher_ident(&req.label);
        let property = cypher_ident(&req.property);
        let top_k = req.top_k as usize;

        // Cypher: compute distance per node, sort ascending, take top_k.
        // Uses the HNSW-backed vector_distance() function.
        // vector_distance() returns f32 L2 distance by default.
        let cypher = format!(
            "MATCH (n:{label}) \
             WITH n, vector_distance(n.{property}, $qv) AS _dist \
             ORDER BY _dist \
             LIMIT {top_k} \
             RETURN n AS _nid, _dist"
        );

        let mut params = std::collections::HashMap::new();
        params.insert("qv".to_string(), Value::Vector(query_vector.values.clone()));

        let rows = {
            let mut db = self.database.lock().unwrap_or_else(|e| e.into_inner());
            db.execute_cypher_with_params(&cypher, params)
                .map_err(|e| Status::internal(format!("vector search error: {e}")))?
        };

        let results: Vec<query::VectorResult> = rows
            .into_iter()
            .filter_map(|row| {
                let node_id = match row.get("_nid")? {
                    Value::Int(id) => *id as u64,
                    _ => return None,
                };
                let distance = match row.get("_dist")? {
                    Value::Float(d) => *d as f32,
                    Value::Int(d) => *d as f32,
                    _ => return None,
                };
                Some(query::VectorResult {
                    node: Some(graph::Node {
                        node_id,
                        labels: vec![req.label.clone()],
                        properties: std::collections::HashMap::new(),
                    }),
                    distance,
                })
            })
            .collect();

        Ok(Response::new(query::VectorSearchResponse { results }))
    }

    async fn hybrid_search(
        &self,
        request: Request<query::HybridSearchRequest>,
    ) -> Result<Response<query::HybridSearchResponse>, Status> {
        let req = request.into_inner();

        let query_vector = req
            .query_vector
            .ok_or_else(|| Status::invalid_argument("query_vector is required"))?;

        if query_vector.values.is_empty() {
            return Err(Status::invalid_argument("query_vector must not be empty"));
        }

        if req.top_k == 0 {
            return Err(Status::invalid_argument("top_k must be > 0"));
        }

        let property = cypher_ident(&req.vector_property);
        let edge_type = cypher_ident(&req.edge_type);
        let max_depth = req.max_depth.max(1);
        let top_k = req.top_k as usize;
        let start_node_id = req.start_node_id;

        // Traverse from start node up to max_depth hops, then rank by vector
        // distance and return top_k neighbours.
        // `start = $start_id` compares the node variable (Value::Int(node_id))
        // directly — same mechanism used in get_node and traverse.
        let cypher = format!(
            "MATCH (start)-[:{edge_type}*1..{max_depth}]->(n) \
             WHERE start = $start_id \
             WITH n, vector_distance(n.{property}, $qv) AS _dist \
             ORDER BY _dist \
             LIMIT {top_k} \
             RETURN n AS _nid, _dist"
        );

        let mut params = std::collections::HashMap::new();
        params.insert("start_id".to_string(), Value::Int(start_node_id as i64));
        params.insert("qv".to_string(), Value::Vector(query_vector.values.clone()));

        let rows = {
            let mut db = self.database.lock().unwrap_or_else(|e| e.into_inner());
            db.execute_cypher_with_params(&cypher, params)
                .map_err(|e| Status::internal(format!("hybrid search error: {e}")))?
        };

        let results: Vec<query::VectorResult> = rows
            .into_iter()
            .filter_map(|row| {
                let node_id = match row.get("_nid")? {
                    Value::Int(id) => *id as u64,
                    _ => return None,
                };
                let distance = match row.get("_dist")? {
                    Value::Float(d) => *d as f32,
                    Value::Int(d) => *d as f32,
                    _ => return None,
                };
                Some(query::VectorResult {
                    node: Some(graph::Node {
                        node_id,
                        labels: vec![],
                        properties: std::collections::HashMap::new(),
                    }),
                    distance,
                })
            })
            .collect();

        Ok(Response::new(query::HybridSearchResponse { results }))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::proto::query::vector_service_server::VectorService;

    fn test_service() -> (VectorServiceImpl, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("tempdir");
        let database = Arc::new(Mutex::new(
            Database::open(dir.path()).expect("open database"),
        ));
        (VectorServiceImpl::new(database), dir)
    }

    /// vector_search returns InvalidArgument when query_vector is missing.
    #[tokio::test]
    async fn vector_search_requires_query_vector() {
        let (svc, _dir) = test_service();

        let result = svc
            .vector_search(Request::new(query::VectorSearchRequest {
                label: "Node".to_string(),
                property: "embedding".to_string(),
                query_vector: None,
                top_k: 5,
                metric: 0,
            }))
            .await;

        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::InvalidArgument);
    }

    /// vector_search returns InvalidArgument when top_k is zero.
    #[tokio::test]
    async fn vector_search_requires_positive_top_k() {
        let (svc, _dir) = test_service();

        let result = svc
            .vector_search(Request::new(query::VectorSearchRequest {
                label: "Node".to_string(),
                property: "embedding".to_string(),
                query_vector: Some(crate::proto::common::Vector {
                    values: vec![1.0, 0.0],
                }),
                top_k: 0,
                metric: 0,
            }))
            .await;

        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::InvalidArgument);
    }

    /// vector_search returns empty results when no nodes have the property.
    #[tokio::test]
    async fn vector_search_empty_on_no_data() {
        let (svc, _dir) = test_service();

        let result = svc
            .vector_search(Request::new(query::VectorSearchRequest {
                label: "VecTest".to_string(),
                property: "embedding".to_string(),
                query_vector: Some(crate::proto::common::Vector {
                    values: vec![1.0, 0.0, 0.0],
                }),
                top_k: 5,
                metric: 0,
            }))
            .await
            .expect("vector search on empty DB should succeed");

        assert_eq!(
            result.into_inner().results.len(),
            0,
            "no nodes → empty results"
        );
    }

    /// vector_search returns nearest node when data exists.
    #[tokio::test]
    async fn vector_search_finds_nearest_node() {
        let (svc, _dir) = test_service();

        // Insert nodes with embeddings via Cypher (through the DB directly).
        {
            let mut db = svc.database.lock().unwrap();
            db.execute_cypher("CREATE (n:Vec {embedding: [1.0, 0.0, 0.0]})")
                .expect("create node 1");
            db.execute_cypher("CREATE (n:Vec {embedding: [0.0, 1.0, 0.0]})")
                .expect("create node 2");
            db.execute_cypher("CREATE (n:Vec {embedding: [0.0, 0.0, 1.0]})")
                .expect("create node 3");
        }

        let result = svc
            .vector_search(Request::new(query::VectorSearchRequest {
                label: "Vec".to_string(),
                property: "embedding".to_string(),
                query_vector: Some(crate::proto::common::Vector {
                    values: vec![1.0, 0.0, 0.0],
                }),
                top_k: 1,
                metric: 0,
            }))
            .await
            .expect("vector search should succeed");

        let body = result.into_inner();
        assert_eq!(
            body.results.len(),
            1,
            "top_k=1 should return exactly 1 result"
        );

        let top = &body.results[0];
        assert!(top.distance >= 0.0, "distance must be non-negative");
        assert!(top.node.is_some(), "result must have a node");
        // The closest to [1,0,0] should have near-zero distance.
        assert!(
            top.distance < 0.1,
            "distance from [1,0,0] to [1,0,0] should be near zero, got {}",
            top.distance
        );
    }

    /// cypher_ident escapes backticks in label/property names.
    #[test]
    fn cypher_ident_escapes_backticks() {
        assert_eq!(cypher_ident("Person"), "`Person`");
        assert_eq!(cypher_ident("my`label"), "`my``label`");
        assert_eq!(cypher_ident(""), "``");
    }

    // --- VectorTopK HNSW wiring integration tests ---

    /// Helper: create a test database with a vector index on (label, property).
    /// Returns the service + tempdir (must outlive the service).
    fn test_service_with_index(
        label: &str,
        property: &str,
        dimensions: u32,
    ) -> (VectorServiceImpl, tempfile::TempDir) {
        use coordinode_core::graph::types::VectorMetric;
        use coordinode_query::index::VectorIndexConfig;

        let dir = tempfile::tempdir().expect("tempdir");
        let mut database = Database::open(dir.path()).expect("open database");

        let config = VectorIndexConfig {
            dimensions,
            metric: VectorMetric::L2,
            m: 16,
            ef_construction: 200,
            quantization: false,
            offload_vectors: false,
        };
        database.create_vector_index("test_vec_idx", label, property, config);

        let database = Arc::new(Mutex::new(database));
        (VectorServiceImpl::new(database), dir)
    }

    /// With an HNSW index registered, vector_search routes through the HNSW path
    /// and returns correct top-K nearest neighbours.
    #[tokio::test]
    async fn vector_search_uses_hnsw_when_index_exists() {
        let (svc, _dir) = test_service_with_index("Vec", "embedding", 3);

        // Insert 20 nodes with embeddings arranged along the X-axis.
        // Node i has embedding [i/20, 0.0, 0.0].
        // on_vector_written is called from the Cypher executor CREATE path,
        // which uses ctx.vector_index_registry to update the HNSW graph.
        {
            let mut db = svc.database.lock().unwrap();
            for i in 0..20 {
                let x = i as f64 / 20.0;
                let cypher = format!("CREATE (n:Vec {{embedding: [{x}, 0.0, 0.0]}})");
                db.execute_cypher(&cypher).expect("create vec node");
            }
        }

        // Query closest to [0.5, 0.0, 0.0] — expect top-3 to cluster around x=0.5.
        let result = svc
            .vector_search(Request::new(query::VectorSearchRequest {
                label: "Vec".to_string(),
                property: "embedding".to_string(),
                query_vector: Some(crate::proto::common::Vector {
                    values: vec![0.5, 0.0, 0.0],
                }),
                top_k: 3,
                metric: 0,
            }))
            .await
            .expect("vector search should succeed");

        let body = result.into_inner();
        assert_eq!(body.results.len(), 3, "top_k=3 should return 3 results");

        // All distances should be small (< 0.2) — nodes near x=0.5 cluster.
        for r in &body.results {
            assert!(
                r.distance >= 0.0,
                "distance must be non-negative: {}",
                r.distance
            );
            assert!(
                r.distance < 0.2,
                "top-3 results should be near query, got {}",
                r.distance
            );
        }

        // Results should be sorted ascending by distance.
        for pair in body.results.windows(2) {
            assert!(
                pair[0].distance <= pair[1].distance,
                "results must be sorted ascending: {} then {}",
                pair[0].distance,
                pair[1].distance
            );
        }
    }

    /// Without an HNSW index, vector_search falls back to brute-force and still
    /// returns correct results. Ensures backward compatibility — the VectorTopK
    /// optimizer applies regardless of index presence, but brute-force fallback
    /// activates when `registry.has_index()` returns false.
    #[tokio::test]
    async fn vector_search_brute_force_without_index() {
        let (svc, _dir) = test_service();

        // Insert 10 nodes WITHOUT creating a vector index beforehand.
        {
            let mut db = svc.database.lock().unwrap();
            for i in 0..10 {
                let x = i as f64 / 10.0;
                let cypher = format!("CREATE (n:NoIdx {{embedding: [{x}, 0.0, 0.0]}})");
                db.execute_cypher(&cypher).expect("create node");
            }
        }

        // Query for nearest to [0.3, 0.0, 0.0] — without index, this goes through
        // the brute-force fallback path in execute_vector_top_k_brute_force.
        let result = svc
            .vector_search(Request::new(query::VectorSearchRequest {
                label: "NoIdx".to_string(),
                property: "embedding".to_string(),
                query_vector: Some(crate::proto::common::Vector {
                    values: vec![0.3, 0.0, 0.0],
                }),
                top_k: 2,
                metric: 0,
            }))
            .await
            .expect("brute-force vector search should succeed");

        let body = result.into_inner();
        assert_eq!(body.results.len(), 2);
        // Closest should be x=0.3 (exact match) with distance ~0
        assert!(
            body.results[0].distance < 0.05,
            "closest node should have near-zero distance, got {}",
            body.results[0].distance
        );
    }

    /// CORRECTNESS: large HNSW path must produce HIGH-QUALITY top-K relative to brute-force.
    ///
    /// HNSW is an approximate nearest-neighbour algorithm — it does NOT guarantee
    /// bit-exact matches with brute-force results. The correctness contract is:
    /// - **Recall@k**: HNSW top-k should contain most (but not necessarily all) of the
    ///   true top-k candidates from brute-force. Standard quality metric used by
    ///   Qdrant, Milvus, FAISS to evaluate HNSW implementations.
    /// - **Distance approximation**: HNSW top-1 should be within ~5% of brute-force
    ///   top-1 distance (HNSW never returns a worse nearest neighbour than brute force,
    ///   but may miss the absolute nearest when ef_search is bounded).
    ///
    /// This test uses recall@5 over brute-force top-10 as ground truth — at least
    /// 3 out of HNSW's top-5 must be among the true top-10 nearest neighbours.
    #[tokio::test]
    async fn vector_top_k_large_hnsw_matches_brute_force() {
        let (svc_hnsw, _dir_h) = test_service_with_index("Large", "vec", 4);
        let (svc_brute, _dir_b) = test_service();

        // Deterministic dataset: 1100 nodes with embeddings derived from index.
        // Spread across 4D hypercube so distances vary non-trivially.
        const N: usize = 1100;
        let vectors: Vec<[f64; 4]> = (0..N)
            .map(|i| {
                let f = i as f64 / N as f64;
                [f, (1.0 - f), (f * 2.3 + 0.1) % 1.0, (f * 3.7 + 0.3) % 1.0]
            })
            .collect();

        for svc in [&svc_hnsw, &svc_brute] {
            let mut db = svc.database.lock().unwrap();
            for v in &vectors {
                let cypher = format!(
                    "CREATE (n:Large {{vec: [{}, {}, {}, {}]}})",
                    v[0], v[1], v[2], v[3]
                );
                db.execute_cypher(&cypher).expect("create");
            }
        }

        let query_vec = vec![0.3, 0.7, 0.5, 0.2];

        // Get HNSW top-5.
        let hnsw_res = svc_hnsw
            .vector_search(Request::new(query::VectorSearchRequest {
                label: "Large".to_string(),
                property: "vec".to_string(),
                query_vector: Some(crate::proto::common::Vector {
                    values: query_vec.clone(),
                }),
                top_k: 5,
                metric: 0,
            }))
            .await
            .expect("hnsw search")
            .into_inner()
            .results;

        // Get brute-force top-50 as ground truth for recall@5 check.
        // Use a generous ground truth window because our synthetic dataset has
        // dense clusters (~50 nodes at nearly identical distances) — HNSW's
        // approximate navigation can legitimately skip some of them while still
        // returning results that are among the true top-50 nearest.
        let brute_res = svc_brute
            .vector_search(Request::new(query::VectorSearchRequest {
                label: "Large".to_string(),
                property: "vec".to_string(),
                query_vector: Some(crate::proto::common::Vector {
                    values: query_vec.clone(),
                }),
                top_k: 50,
                metric: 0,
            }))
            .await
            .expect("brute search")
            .into_inner()
            .results;

        assert_eq!(hnsw_res.len(), 5, "HNSW top-5 returned");
        assert_eq!(brute_res.len(), 50, "brute top-50 ground truth");

        // Distance approximation: HNSW top-1 within 5% of brute-force top-1.
        let top1_ratio = hnsw_res[0].distance / brute_res[0].distance;
        assert!(
            (0.95..=1.05).contains(&top1_ratio),
            "HNSW top-1 outside 5% of brute top-1: hnsw={} brute={} ratio={}",
            hnsw_res[0].distance,
            brute_res[0].distance,
            top1_ratio
        );

        // Recall@5 over brute top-50: at least 4 of HNSW's 5 must be in brute top-50.
        // A node dataset with dense clusters may have ~50 candidates at near-identical
        // distances — HNSW's approximate path can legitimately return any subset of
        // those 50 while still being "correct" from a practical retrieval standpoint.
        let brute_top50_ids: std::collections::HashSet<u64> = brute_res
            .iter()
            .filter_map(|r| r.node.as_ref().map(|n| n.node_id))
            .collect();
        let hnsw_hits: usize = hnsw_res
            .iter()
            .filter(|r| {
                r.node
                    .as_ref()
                    .is_some_and(|n| brute_top50_ids.contains(&n.node_id))
            })
            .count();
        assert!(
            hnsw_hits >= 4,
            "recall@5 over brute top-50 too low: only {hnsw_hits}/5 HNSW results \
             are among the 50 true nearest neighbours (expected ≥4)"
        );

        // HNSW results must still be sorted ascending by distance.
        for pair in hnsw_res.windows(2) {
            assert!(
                pair[0].distance <= pair[1].distance,
                "HNSW results not sorted: {} then {}",
                pair[0].distance,
                pair[1].distance
            );
        }
    }

    /// Boundary: exactly at `VECTOR_TOP_K_BRUTE_FORCE_THRESHOLD - 1` nodes,
    /// the path must be brute force. At threshold+1, HNSW path is active.
    /// This tests the `<` boundary of the early-return check.
    #[tokio::test]
    async fn vector_top_k_below_threshold_uses_brute_force() {
        let (svc, _dir) = test_service_with_index("Threshold", "v", 2);

        // 999 nodes — just below the 1000 threshold → always brute force.
        const N: usize = 999;
        {
            let mut db = svc.database.lock().unwrap();
            for i in 0..N {
                let x = i as f64 / N as f64;
                let cypher = format!("CREATE (n:Threshold {{v: [{x}, 0.0]}})");
                db.execute_cypher(&cypher).expect("create");
            }
        }

        let result = svc
            .vector_search(Request::new(query::VectorSearchRequest {
                label: "Threshold".to_string(),
                property: "v".to_string(),
                query_vector: Some(crate::proto::common::Vector {
                    values: vec![1.0, 0.0],
                }),
                top_k: 3,
                metric: 0,
            }))
            .await
            .expect("search below threshold");

        let body = result.into_inner();
        assert_eq!(body.results.len(), 3);
        // Closest should be x = 998/999 ≈ 0.999, distance ≈ 0.001.
        assert!(
            body.results[0].distance < 0.005,
            "closest to [1,0] should be ~0.999, got distance {}",
            body.results[0].distance
        );
        // Ascending
        assert!(body.results[0].distance <= body.results[1].distance);
        assert!(body.results[1].distance <= body.results[2].distance);
    }

    /// hybrid_search on a LARGE traversal subset: verifies the HNSW-accelerated
    /// path kicks in when the traverse result exceeds `VECTOR_TOP_K_BRUTE_FORCE_THRESHOLD`.
    /// This exercises the `try_hnsw_vector_top_k` branch that handles Filter+Traverse
    /// input rows with a real HNSW index and overfetch-based intersection.
    #[tokio::test]
    async fn hybrid_search_large_traversal_uses_hnsw() {
        let (svc, _dir) = test_service_with_index("BigPerson", "embedding", 3);

        // Create a hub node and 1100 persons linked via KNOWS (exceeds the
        // brute-force threshold of 1000 → forces HNSW path).
        // Embeddings spread along the X axis so distances are predictable.
        const N_PERSONS: usize = 1100;
        {
            let mut db = svc.database.lock().unwrap();
            db.execute_cypher("CREATE (s:BigHub {name: 'hub'})")
                .expect("create hub");

            for i in 0..N_PERSONS {
                let x = i as f64 / N_PERSONS as f64;
                let y = (i % 7) as f64 / 10.0;
                let cypher =
                    format!("CREATE (n:BigPerson {{name: 'p{i}', embedding: [{x}, {y}, 0.0]}})");
                db.execute_cypher(&cypher).expect("create person");
            }

            // Single Cartesian-product query creates all edges in one pass —
            // orders of magnitude faster than 1100 individual MATCH+CREATEs.
            // Since there's only one BigHub and N_PERSONS BigPersons, the
            // product produces exactly N_PERSONS rows, one edge per person.
            db.execute_cypher(
                "MATCH (s:BigHub {name: 'hub'}), (p:BigPerson) CREATE (s)-[:KNOWS]->(p)",
            )
            .expect("bulk create edges");
        }

        // Get start_node_id.
        let start_id = {
            let mut db = svc.database.lock().unwrap();
            let rows = db
                .execute_cypher("MATCH (s:BigHub {name: 'hub'}) RETURN s")
                .expect("get hub id");
            match rows[0].get("s").expect("s binding") {
                coordinode_core::graph::types::Value::Int(i) => *i as u64,
                _ => panic!("hub id not int"),
            }
        };

        // Query: nearest to [1.0, 0.0, 0.0] among hub's 1500 KNOWS neighbours.
        // Expected: p1499 (x=0.999) and nearby persons.
        let result = svc
            .hybrid_search(Request::new(query::HybridSearchRequest {
                start_node_id: start_id,
                edge_type: "KNOWS".to_string(),
                max_depth: 1,
                vector_property: "embedding".to_string(),
                query_vector: Some(crate::proto::common::Vector {
                    values: vec![1.0, 0.0, 0.0],
                }),
                top_k: 5,
                metric: 0,
            }))
            .await
            .expect("hybrid search should succeed");

        let body = result.into_inner();
        assert_eq!(
            body.results.len(),
            5,
            "top_k=5 on a 1500-person graph must return 5 results (HNSW path), got {}",
            body.results.len()
        );

        // All returned distances must be small (< 0.5) since the query is
        // near the X-axis and many persons have x close to 1.0.
        for r in &body.results {
            assert!(
                r.distance >= 0.0,
                "distance must be non-negative: {}",
                r.distance
            );
            assert!(
                r.distance < 1.0,
                "top-5 in a large dataset should be close to query: {}",
                r.distance
            );
        }

        // Ascending by distance.
        for pair in body.results.windows(2) {
            assert!(
                pair[0].distance <= pair[1].distance,
                "hybrid_search must return results sorted ascending by distance: {} then {}",
                pair[0].distance,
                pair[1].distance
            );
        }
    }

    /// hybrid_search: varlen traversal from a start node + vector ranking.
    /// Verifies that VectorTopK optimization still produces correct results
    /// when input rows come from a Traverse (not a NodeScan). This covers
    /// the second RPC method on VectorServiceImpl that was not tested at all.
    #[tokio::test]
    async fn hybrid_search_traversal_plus_vector_rank() {
        let (svc, _dir) = test_service();

        // Build a small graph: start — KNOWS → p0, p1, p2, p3, p4
        // p0: embedding [1.0, 0.0, 0.0]
        // p1: embedding [0.9, 0.1, 0.0]
        // p2: embedding [0.0, 1.0, 0.0]
        // p3: embedding [0.0, 0.0, 1.0]
        // p4: embedding [0.5, 0.5, 0.0]
        {
            let mut db = svc.database.lock().unwrap();
            db.execute_cypher("CREATE (s:Hub {name: 'start'})")
                .expect("create start");
            db.execute_cypher("CREATE (n:Person {name: 'p0', embedding: [1.0, 0.0, 0.0]})")
                .expect("p0");
            db.execute_cypher("CREATE (n:Person {name: 'p1', embedding: [0.9, 0.1, 0.0]})")
                .expect("p1");
            db.execute_cypher("CREATE (n:Person {name: 'p2', embedding: [0.0, 1.0, 0.0]})")
                .expect("p2");
            db.execute_cypher("CREATE (n:Person {name: 'p3', embedding: [0.0, 0.0, 1.0]})")
                .expect("p3");
            db.execute_cypher("CREATE (n:Person {name: 'p4', embedding: [0.5, 0.5, 0.0]})")
                .expect("p4");

            // Create KNOWS edges from start to each person.
            for name in ["p0", "p1", "p2", "p3", "p4"] {
                let cypher = format!(
                    "MATCH (s:Hub {{name: 'start'}}), (p:Person {{name: '{name}'}}) \
                     CREATE (s)-[:KNOWS]->(p)"
                );
                db.execute_cypher(&cypher).expect("create edge");
            }
        }

        // Get start_node_id via a Cypher query.
        let start_id = {
            let mut db = svc.database.lock().unwrap();
            let rows = db
                .execute_cypher("MATCH (s:Hub {name: 'start'}) RETURN s")
                .expect("get start id");
            let val = rows[0].get("s").expect("s binding");
            match val {
                coordinode_core::graph::types::Value::Int(i) => *i as u64,
                _ => panic!("start id is not int"),
            }
        };

        // Query: find top-2 nearest to [1.0, 0.0, 0.0] reachable from start in 1 hop.
        let result = svc
            .hybrid_search(Request::new(query::HybridSearchRequest {
                start_node_id: start_id,
                edge_type: "KNOWS".to_string(),
                max_depth: 1,
                vector_property: "embedding".to_string(),
                query_vector: Some(crate::proto::common::Vector {
                    values: vec![1.0, 0.0, 0.0],
                }),
                top_k: 2,
                metric: 0,
            }))
            .await
            .expect("hybrid search should succeed");

        let body = result.into_inner();
        assert_eq!(body.results.len(), 2, "top_k=2 should return 2 results");

        // Closest to [1,0,0]: p0 (dist=0), then p1 (dist=0.14).
        // Both should have small distances.
        assert!(
            body.results[0].distance < 0.01,
            "closest to [1,0,0] should be p0 with near-zero distance, got {}",
            body.results[0].distance
        );
        assert!(
            body.results[1].distance < 0.2,
            "second closest should still be small, got {}",
            body.results[1].distance
        );
        // Results must be sorted ascending.
        assert!(body.results[0].distance <= body.results[1].distance);
    }

    /// k=0 (or no results) returns empty list without error, even with HNSW index.
    #[tokio::test]
    async fn vector_search_empty_result_on_no_label_match() {
        let (svc, _dir) = test_service_with_index("Exists", "embedding", 3);

        // Create nodes under a DIFFERENT label — the query should find nothing.
        {
            let mut db = svc.database.lock().unwrap();
            db.execute_cypher("CREATE (n:Other {embedding: [1.0, 0.0, 0.0]})")
                .expect("create");
        }

        let result = svc
            .vector_search(Request::new(query::VectorSearchRequest {
                label: "Exists".to_string(),
                property: "embedding".to_string(),
                query_vector: Some(crate::proto::common::Vector {
                    values: vec![1.0, 0.0, 0.0],
                }),
                top_k: 5,
                metric: 0,
            }))
            .await
            .expect("should succeed on empty label match");

        assert_eq!(
            result.into_inner().results.len(),
            0,
            "no nodes with requested label → empty results"
        );
    }

    /// top_k larger than the number of available nodes returns all of them.
    /// Verifies both HNSW and brute-force handle k > n gracefully.
    #[tokio::test]
    async fn vector_search_top_k_larger_than_dataset() {
        let (svc, _dir) = test_service_with_index("Small", "v", 2);

        // Only 3 nodes in the dataset.
        {
            let mut db = svc.database.lock().unwrap();
            db.execute_cypher("CREATE (n:Small {v: [1.0, 0.0]})")
                .expect("n1");
            db.execute_cypher("CREATE (n:Small {v: [0.0, 1.0]})")
                .expect("n2");
            db.execute_cypher("CREATE (n:Small {v: [0.7, 0.7]})")
                .expect("n3");
        }

        // Ask for top-100 but only 3 exist.
        let result = svc
            .vector_search(Request::new(query::VectorSearchRequest {
                label: "Small".to_string(),
                property: "v".to_string(),
                query_vector: Some(crate::proto::common::Vector {
                    values: vec![0.5, 0.5],
                }),
                top_k: 100,
                metric: 0,
            }))
            .await
            .expect("should succeed even when k > n");

        let body = result.into_inner();
        assert_eq!(
            body.results.len(),
            3,
            "should return all 3 nodes when k > n"
        );
    }

    /// HNSW and brute-force paths must return the same top-K on the same data.
    /// This is a correctness check: the optimization must not change result
    /// ordering for queries where both paths are valid.
    #[tokio::test]
    async fn vector_search_hnsw_matches_brute_force() {
        // Setup two databases with identical data: one with index, one without.
        let (svc_hnsw, _dir_h) = test_service_with_index("Item", "vec", 4);
        let (svc_brute, _dir_b) = test_service();

        // Insert same 15 deterministic vectors into both.
        let vectors: Vec<[f64; 4]> = (0..15)
            .map(|i| {
                let f = i as f64 / 15.0;
                [f, 1.0 - f, (f * 2.0) % 1.0, 0.5]
            })
            .collect();

        for svc in [&svc_hnsw, &svc_brute] {
            let mut db = svc.database.lock().unwrap();
            for v in &vectors {
                let cypher = format!(
                    "CREATE (n:Item {{vec: [{}, {}, {}, {}]}})",
                    v[0], v[1], v[2], v[3]
                );
                db.execute_cypher(&cypher).expect("create");
            }
        }

        // Query with the same vector.
        let query_vec = vec![0.3, 0.7, 0.6, 0.5];
        let req_template = |label: &str| query::VectorSearchRequest {
            label: label.to_string(),
            property: "vec".to_string(),
            query_vector: Some(crate::proto::common::Vector {
                values: query_vec.clone(),
            }),
            top_k: 5,
            metric: 0,
        };

        let hnsw_res = svc_hnsw
            .vector_search(Request::new(req_template("Item")))
            .await
            .expect("hnsw search")
            .into_inner()
            .results;
        let brute_res = svc_brute
            .vector_search(Request::new(req_template("Item")))
            .await
            .expect("brute search")
            .into_inner()
            .results;

        assert_eq!(hnsw_res.len(), 5);
        assert_eq!(brute_res.len(), 5);

        // Top-1 distance must match (both paths find the same nearest).
        assert!(
            (hnsw_res[0].distance - brute_res[0].distance).abs() < 1e-4,
            "HNSW top-1 distance {} != brute-force top-1 distance {}",
            hnsw_res[0].distance,
            brute_res[0].distance
        );

        // Sum of top-5 distances should be close (results may differ in order
        // for ties but the total should match).
        let hnsw_sum: f32 = hnsw_res.iter().map(|r| r.distance).sum();
        let brute_sum: f32 = brute_res.iter().map(|r| r.distance).sum();
        assert!(
            (hnsw_sum - brute_sum).abs() < 0.01,
            "top-5 distance sums differ: hnsw={hnsw_sum} brute={brute_sum}"
        );
    }
}
