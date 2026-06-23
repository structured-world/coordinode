use std::sync::Arc;

// no-std: spin::RwLock (drop-in).
use parking_lot::RwLock;

use tonic::{Request, Response, Status};

use coordinode_core::graph::node::NodeId;
use coordinode_core::graph::types::Value;
use coordinode_embed::Database;

use crate::proto::{common, graph, query};
use crate::services::cypher::value_to_proto_pub;
use crate::services::db_err_to_status;

/// Backtick-escape a Cypher identifier (label or property key).
///
/// Doubles any embedded backtick per the OpenCypher spec so that arbitrary
/// strings can be safely embedded in a Cypher query without injection.
fn cypher_ident(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}

/// Map a `DistanceMetric` proto integer to the Cypher function and ORDER direction.
///
/// Returns `(fn_name, order)` where `order` is `"ASC"` for distance-based metrics
/// (lower = closer) and `"DESC"` for similarity-based metrics (higher = closer).
///
/// Proto enum values:
///   UNSPECIFIED=0, COSINE=1, L2=2, DOT=3, L1=4
fn metric_cypher(metric: i32) -> (&'static str, &'static str) {
    match metric {
        1 => ("vector_similarity", "DESC"), // COSINE: higher similarity = closer
        3 => ("vector_dot", "DESC"),        // DOT: higher dot product = closer
        4 => ("vector_manhattan", "ASC"),   // L1: lower distance = closer
        _ => ("vector_distance", "ASC"),    // UNSPECIFIED or L2: lower distance = closer
    }
}

/// Convert a raw result row (from a vector/hybrid Cypher query) into a `VectorResult`.
///
/// Expects the row to have been produced by a query that uses `WITH *, ... AS _dist RETURN *`
/// so that all node properties (`n.*`) are present in the row alongside `n.__label__`
/// and `_dist`.
///
/// Returns `None` if the row lacks the required `n` (node_id) or `_dist` columns.
fn row_to_vector_result(
    row: std::collections::BTreeMap<String, Value>,
) -> Option<query::VectorResult> {
    let node_id = match row.get("n")? {
        Value::Int(id) => *id as u64,
        _ => return None,
    };
    let distance = match row.get("_dist")? {
        Value::Float(d) => *d as f32,
        Value::Int(d) => *d as f32,
        _ => return None,
    };
    let label = match row.get("n.__label__") {
        Some(Value::String(s)) => s.clone(),
        _ => String::new(),
    };
    let labels = if label.is_empty() {
        vec![]
    } else {
        vec![label]
    };
    let mut properties: std::collections::HashMap<String, common::PropertyValue> =
        std::collections::HashMap::new();
    for (k, v) in &row {
        if let Some(prop_name) = k.strip_prefix("n.") {
            if !prop_name.starts_with("__") {
                properties.insert(prop_name.to_string(), value_to_proto_pub(v));
            }
        }
    }
    Some(query::VectorResult {
        node: Some(graph::Node {
            node_id,
            labels,
            properties,
            element_id: NodeId::from_raw(node_id).to_element_id(),
        }),
        distance,
    })
}

/// Map the engine-side vector index health snapshot to its wire form. The
/// serving lifecycle, rebuild progress, and the `indexed_hlc` freshness
/// watermark are carried back so a client can fence read-your-writes against
/// the index that answered the query.
fn index_health_to_proto(
    h: coordinode_vector::health::IndexHealthState,
) -> query::VectorIndexHealth {
    use coordinode_vector::health::IndexHealthState as H;
    use query::vector_index_health::ServingState;
    match h {
        H::Ready { indexed_hlc } => query::VectorIndexHealth {
            serving_state: ServingState::Ready as i32,
            rebuild_progress: 0.0,
            eta_ms: 0,
            indexed_hlc,
            offline_reason: String::new(),
        },
        H::Rebuilding {
            progress,
            eta_ms,
            indexed_hlc,
        } => query::VectorIndexHealth {
            serving_state: ServingState::Rebuilding as i32,
            rebuild_progress: progress,
            eta_ms,
            indexed_hlc,
            offline_reason: String::new(),
        },
        H::Offline { reason } => query::VectorIndexHealth {
            serving_state: ServingState::Offline as i32,
            rebuild_progress: 0.0,
            eta_ms: 0,
            indexed_hlc: 0,
            offline_reason: reason,
        },
    }
}

/// Echo the freshness watermark of the serving index as a response-trailer
/// header so a client library can read it without decoding the body. No-op
/// when the query did not run against a managed HNSW index.
fn set_indexed_hlc_header<T>(resp: &mut Response<T>, health: Option<&query::VectorIndexHealth>) {
    if let Some(h) = health {
        if let Ok(val) = h.indexed_hlc.to_string().parse() {
            resp.metadata_mut().insert("coordinode-indexed-hlc", val);
        }
    }
}

pub struct VectorServiceImpl {
    database: Arc<RwLock<Database>>,
}

impl VectorServiceImpl {
    pub fn new(database: Arc<RwLock<Database>>) -> Self {
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

        // Cypher: compute distance/similarity per node, sort, take top_k.
        // `WITH *` preserves all n.* columns through the Project so that labels
        // and properties are available in result rows (VectorTopK optimization
        // operates on the pre-WITH NodeScan rows and passes them through Star).
        let (dist_fn, order_dir) = metric_cypher(req.metric);
        let cypher = format!(
            "MATCH (n:{label}) \
             WITH *, {dist_fn}(n.{property}, $qv) AS _dist \
             ORDER BY _dist {order_dir} \
             LIMIT {top_k} \
             RETURN *"
        );

        let mut params = std::collections::HashMap::new();
        params.insert("qv".to_string(), Value::Vector(query_vector.values.clone()));

        // Shared read access: vector search is read-only and benefits
        // from running in parallel with other queries. Read the serving
        // index's health under the same lock so the watermark reported is
        // consistent with the snapshot the query ran against. The registry
        // keys on the raw (unescaped) label/property.
        let (rows, index_health) = {
            let db = self.database.read();
            let rows = db
                .execute_cypher_shared(&cypher, Some(params), None, None, None)
                .map_err(|e| db_err_to_status("vector search", e))?
                .rows;
            let health = db
                .vector_index_registry()
                .health_snapshot(&req.label, &req.property)
                .map(index_health_to_proto);
            (rows, health)
        };

        let results: Vec<query::VectorResult> =
            rows.into_iter().filter_map(row_to_vector_result).collect();

        let mut response = Response::new(query::VectorSearchResponse {
            results,
            index_health: index_health.clone(),
        });
        set_indexed_hlc_header(&mut response, index_health.as_ref());
        Ok(response)
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
        // distance/similarity and return top_k neighbours.
        // `start = $start_id` compares the node variable (Value::Int(node_id))
        // directly — same mechanism used in get_node and traverse.
        // `WITH *` preserves all n.* columns through the Project so that labels
        // and properties are available in result rows (same as vector_search).
        let (dist_fn, order_dir) = metric_cypher(req.metric);
        let cypher = format!(
            "MATCH (start)-[:{edge_type}*1..{max_depth}]->(n) \
             WHERE start = $start_id \
             WITH *, {dist_fn}(n.{property}, $qv) AS _dist \
             ORDER BY _dist {order_dir} \
             LIMIT {top_k} \
             RETURN *"
        );

        let mut params = std::collections::HashMap::new();
        params.insert("start_id".to_string(), Value::Int(start_node_id as i64));
        params.insert("qv".to_string(), Value::Vector(query_vector.values.clone()));

        // Hybrid search is read-only — take the shared read lock so
        // concurrent search requests run in parallel.
        let rows = {
            let db = self.database.read();
            db.execute_cypher_shared(&cypher, Some(params), None, None, None)
                .map_err(|e| db_err_to_status("hybrid search", e))?
                .rows
        };

        let results: Vec<query::VectorResult> =
            rows.into_iter().filter_map(row_to_vector_result).collect();

        // The hybrid request carries no label, so the vector phase can span
        // several labels' indexes — there is no single index whose health to
        // report. Leave it unset (documented contract on the proto field).
        Ok(Response::new(query::HybridSearchResponse {
            results,
            index_health: None,
        }))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
