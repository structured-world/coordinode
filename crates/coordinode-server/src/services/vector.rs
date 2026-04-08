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
        params.insert(
            "qv".to_string(),
            Value::Vector(query_vector.values.clone()),
        );

        let rows = {
            let mut db = self
                .database
                .lock()
                .unwrap_or_else(|e| e.into_inner());
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
        let cypher = format!(
            "MATCH (start)-[:{edge_type}*1..{max_depth}]->(n) \
             WHERE id(start) = $start_id \
             WITH n, vector_distance(n.{property}, $qv) AS _dist \
             ORDER BY _dist \
             LIMIT {top_k} \
             RETURN n AS _nid, _dist"
        );

        let mut params = std::collections::HashMap::new();
        params.insert("start_id".to_string(), Value::Int(start_node_id as i64));
        params.insert(
            "qv".to_string(),
            Value::Vector(query_vector.values.clone()),
        );

        let rows = {
            let mut db = self
                .database
                .lock()
                .unwrap_or_else(|e| e.into_inner());
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
        assert_eq!(body.results.len(), 1, "top_k=1 should return exactly 1 result");

        let top = &body.results[0];
        assert!(top.distance >= 0.0, "distance must be non-negative");
        assert!(
            top.node.is_some(),
            "result must have a node"
        );
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
}
