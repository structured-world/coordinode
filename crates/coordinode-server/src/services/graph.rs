use std::sync::Arc;

// no-std: spin::RwLock (drop-in). parking_lot::RwLock chosen over
// std::sync per ~/projects/sw/CLAUDE.md hot-path rule.
use parking_lot::RwLock;

use tonic::{Request, Response, Status};

use coordinode_core::graph::node::NodeId;
use coordinode_core::graph::types::Value;
use coordinode_embed::Database;

use crate::proto::{common, graph};
use crate::services::cypher::{proto_to_value_pub, value_to_proto_pub};
use crate::services::db_err_to_status;

/// Backtick-escape a Cypher identifier (label, relationship type, or property key).
fn cypher_ident(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}

pub struct GraphServiceImpl {
    database: Arc<RwLock<Database>>,
}

impl GraphServiceImpl {
    pub fn new(database: Arc<RwLock<Database>>) -> Self {
        Self { database }
    }
}

#[tonic::async_trait]
impl graph::graph_service_server::GraphService for GraphServiceImpl {
    async fn create_node(
        &self,
        request: Request<graph::CreateNodeRequest>,
    ) -> Result<Response<graph::Node>, Status> {
        let req = request.into_inner();

        // Build `:Label1:Label2` part.
        let label_part: String = req
            .labels
            .iter()
            .map(|l| format!(":{}", cypher_ident(l)))
            .collect();

        // Build `{key1: $p0, key2: $p1, ...}` part and collect parameters.
        let mut params: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
        let prop_assignments: Vec<String> = req
            .properties
            .iter()
            .enumerate()
            .map(|(i, (k, v))| {
                let param_name = format!("p{i}");
                params.insert(param_name.clone(), proto_to_value_pub(v));
                format!("{}: ${param_name}", cypher_ident(k))
            })
            .collect();
        let props_part = if prop_assignments.is_empty() {
            String::new()
        } else {
            format!(" {{{}}}", prop_assignments.join(", "))
        };

        let cypher = format!("CREATE (n{label_part}{props_part}) RETURN n");

        let rows = {
            let mut db = self.database.write();
            db.execute_cypher_with_params(&cypher, params)
                .map_err(|e| db_err_to_status("create_node", e))?
        };

        let node_id = match rows.first().and_then(|r| r.get("n")) {
            Some(Value::Int(id)) => *id as u64,
            _ => {
                return Err(Status::internal(
                    "create_node: executor did not return node id",
                ))
            }
        };

        Ok(Response::new(graph::Node {
            node_id,
            labels: req.labels,
            properties: req.properties,
            element_id: NodeId::from_raw(node_id).to_element_id(),
        }))
    }

    async fn create_nodes_batch(
        &self,
        request: Request<graph::CreateNodesBatchRequest>,
    ) -> Result<Response<graph::CreateNodesBatchResponse>, Status> {
        let req = request.into_inner();

        if req.nodes.is_empty() {
            return Ok(Response::new(graph::CreateNodesBatchResponse {
                nodes: vec![],
            }));
        }

        // The Cypher path that powers this handler is a single
        // `UNWIND $rows AS r CREATE (m:Label…) SET m = r RETURN m`
        // — one parse (cached after the first batch), one logical
        // plan, one execute. Cypher can't add labels dynamically per
        // row, so all nodes in a batch must share the same label
        // set. Mixed-label batches are an explicit InvalidArgument.
        let labels = req.nodes[0].labels.clone();
        for (i, n) in req.nodes.iter().enumerate().skip(1) {
            if n.labels != labels {
                return Err(Status::invalid_argument(format!(
                    "create_nodes_batch: nodes[{i}].labels differs from nodes[0].labels; \
                     batches must be label-homogeneous"
                )));
            }
        }
        let label_part: String = labels
            .iter()
            .map(|l| format!(":{}", cypher_ident(l)))
            .collect();

        // Build the $rows list parameter. Each element is a property
        // map keyed by property name — Cypher's `SET m = r` then
        // copies each entry into a node property.
        let rows: Vec<Value> = req
            .nodes
            .iter()
            .map(|n| {
                let map: std::collections::BTreeMap<String, Value> = n
                    .properties
                    .iter()
                    .map(|(k, v)| (k.clone(), proto_to_value_pub(v)))
                    .collect();
                Value::Map(map)
            })
            .collect();

        let cypher = format!("UNWIND $rows AS r CREATE (m{label_part}) SET m = r RETURN m");
        let mut params: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
        params.insert("rows".to_string(), Value::Array(rows));

        let result_rows = {
            let mut db = self.database.write();
            db.execute_cypher_with_params(&cypher, params)
                .map_err(|e| db_err_to_status("create_nodes_batch", e))?
        };

        if result_rows.len() != req.nodes.len() {
            return Err(Status::internal(format!(
                "create_nodes_batch: executor returned {} rows, expected {}",
                result_rows.len(),
                req.nodes.len()
            )));
        }

        let nodes: Result<Vec<graph::Node>, Status> = result_rows
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let node_id = match row.get("m") {
                    Some(Value::Int(id)) => *id as u64,
                    _ => {
                        return Err(Status::internal(format!(
                            "create_nodes_batch: row[{i}] missing node id"
                        )))
                    }
                };
                Ok(graph::Node {
                    node_id,
                    labels: labels.clone(),
                    properties: req.nodes[i].properties.clone(),
                    element_id: NodeId::from_raw(node_id).to_element_id(),
                })
            })
            .collect();

        Ok(Response::new(graph::CreateNodesBatchResponse {
            nodes: nodes?,
        }))
    }

    async fn get_node(
        &self,
        request: Request<graph::GetNodeRequest>,
    ) -> Result<Response<graph::Node>, Status> {
        let req = request.into_inner();

        let mut params: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
        params.insert("id".to_string(), Value::Int(req.node_id as i64));

        // `n = $id` compares the node variable (Value::Int(node_id) from NodeScan)
        // with the target id, which is the correct way to filter by internal node ID.
        // `RETURN *` copies ALL columns from NodeScan: `n`, `n.__label__`, `n.<prop>`.
        // An explicit `RETURN n, n.__label__` would strip `n.*` properties in the Project step.
        // get_node is read-only — shared lock lets parallel reads
        // proceed without blocking on each other.
        let rows = {
            let db = self.database.read();
            db.execute_cypher_shared(
                "MATCH (n) WHERE n = $id RETURN *",
                Some(params),
                None,
                None,
                None,
            )
            .map_err(|e| db_err_to_status("get_node", e))?
            .rows
        };

        let row = rows
            .first()
            .ok_or_else(|| Status::not_found(format!("node {} not found", req.node_id)))?;

        let node_id = match row.get("n") {
            Some(Value::Int(id)) => *id as u64,
            _ => return Err(Status::not_found(format!("node {} not found", req.node_id))),
        };

        // Primary label stored by NodeScan as `n.__label__`.
        let label = match row.get("n.__label__") {
            Some(Value::String(s)) => s.clone(),
            _ => String::new(),
        };

        // Collect all `n.*` properties (excluding `n.__label__` and `n` itself).
        let mut properties: std::collections::HashMap<String, common::PropertyValue> =
            std::collections::HashMap::new();
        for (k, v) in row {
            if let Some(prop_name) = k.strip_prefix("n.") {
                if !prop_name.starts_with("__") {
                    properties.insert(prop_name.to_string(), value_to_proto_pub(v));
                }
            }
        }

        let labels = if label.is_empty() {
            vec![]
        } else {
            vec![label]
        };

        Ok(Response::new(graph::Node {
            node_id,
            labels,
            properties,
            element_id: NodeId::from_raw(node_id).to_element_id(),
        }))
    }

    async fn create_edge(
        &self,
        request: Request<graph::CreateEdgeRequest>,
    ) -> Result<Response<graph::Edge>, Status> {
        let req = request.into_inner();

        let rel_type = cypher_ident(&req.edge_type);

        let mut params: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
        params.insert("src_id".to_string(), Value::Int(req.source_node_id as i64));
        params.insert("dst_id".to_string(), Value::Int(req.target_node_id as i64));

        let prop_assignments: Vec<String> = req
            .properties
            .iter()
            .enumerate()
            .map(|(i, (k, v))| {
                let param_name = format!("ep{i}");
                params.insert(param_name.clone(), proto_to_value_pub(v));
                format!("{}: ${param_name}", cypher_ident(k))
            })
            .collect();
        let props_part = if prop_assignments.is_empty() {
            String::new()
        } else {
            format!(" {{{}}}", prop_assignments.join(", "))
        };

        // Match both endpoints by internal node id (node variable = Value::Int(id)).
        let cypher = format!(
            "MATCH (src), (dst) WHERE src = $src_id AND dst = $dst_id \
             CREATE (src)-[:{rel_type}{props_part}]->(dst) \
             RETURN src, dst"
        );

        let rows = {
            let mut db = self.database.write();
            db.execute_cypher_with_params(&cypher, params)
                .map_err(|e| db_err_to_status("create_edge", e))?
        };

        let row = rows
            .first()
            .ok_or_else(|| Status::not_found("source or target node not found"))?;

        let src_id = match row.get("src") {
            Some(Value::Int(id)) => *id as u64,
            _ => req.source_node_id,
        };
        let dst_id = match row.get("dst") {
            Some(Value::Int(id)) => *id as u64,
            _ => req.target_node_id,
        };

        // Synthetic edge_id: stable for a given (src, dst) pair and always > 0
        // since both src_id and dst_id are > 0 after create_node.
        // Uses a prime-based mixing to avoid collisions between (a,b) and (b,a).
        let edge_id = src_id
            .wrapping_mul(2_654_435_761)
            .wrapping_add(dst_id)
            .max(1);

        Ok(Response::new(graph::Edge {
            edge_id,
            edge_type: req.edge_type,
            source_node_id: src_id,
            target_node_id: dst_id,
            properties: req.properties,
            element_id: format!(
                "{}:{}",
                NodeId::from_raw(src_id).to_element_id(),
                NodeId::from_raw(dst_id).to_element_id()
            ),
        }))
    }

    async fn traverse(
        &self,
        request: Request<graph::TraverseRequest>,
    ) -> Result<Response<graph::TraverseResponse>, Status> {
        let req = request.into_inner();

        if req.start_node_id == 0 {
            return Err(Status::invalid_argument("start_node_id must be > 0"));
        }

        let depth = req.max_depth.max(1);
        let limit = req
            .pagination
            .as_ref()
            .map(|p| p.page_size.max(1) as usize)
            .unwrap_or(100);

        let mut params: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
        params.insert("start_id".to_string(), Value::Int(req.start_node_id as i64));

        // Build direction-aware edge pattern.
        // INBOUND: (start)<-[...]-(n), BOTH: (start)-[...]-(n), OUTBOUND/UNSPECIFIED: (start)-[...]->(n)
        let direction = graph::TraversalDirection::try_from(req.direction)
            .unwrap_or(graph::TraversalDirection::Unspecified);
        let (arrow_left, arrow_right) = match direction {
            graph::TraversalDirection::Inbound => ("<-[", "]-"),
            graph::TraversalDirection::Both => ("-[", "]-"),
            // OUTBOUND or UNSPECIFIED → outbound
            _ => ("-[", "]->"),
        };

        // RETURN * copies all row columns produced by the Traverse operator, which already
        // includes `n.__label__` and all `n.<prop>` columns. An explicit `RETURN n, n.__label__`
        // would strip all `n.*` properties via the Project operator.
        let cypher = if req.edge_type.is_empty() {
            format!(
                "MATCH (start){arrow_left}*1..{depth}{arrow_right}(n) \
                 WHERE start = $start_id \
                 RETURN * \
                 LIMIT {limit}"
            )
        } else {
            let rel_type = cypher_ident(&req.edge_type);
            format!(
                "MATCH (start){arrow_left}:{rel_type}*1..{depth}{arrow_right}(n) \
                 WHERE start = $start_id \
                 RETURN * \
                 LIMIT {limit}"
            )
        };

        // Traversal is read-only — shared lock for parallel reads.
        let rows = {
            let db = self.database.read();
            db.execute_cypher_shared(&cypher, Some(params), None, None, None)
                .map_err(|e| db_err_to_status("traverse", e))?
                .rows
        };

        let nodes: Vec<graph::Node> = rows
            .iter()
            .filter_map(|row| {
                // `RETURN *` copies all columns from the Traverse operator row:
                // `n` = node_id, `n.__label__` = primary label, `n.<prop>` = properties.
                let node_id = match row.get("n")? {
                    Value::Int(id) => *id as u64,
                    _ => return None,
                };

                // Primary label stored by the Traverse operator as `n.__label__`.
                let label = match row.get("n.__label__") {
                    Some(Value::String(s)) => s.clone(),
                    _ => String::new(),
                };
                let labels = if label.is_empty() {
                    vec![]
                } else {
                    vec![label]
                };

                // All node properties are in the row as `n.<prop>` (excluding internal `__*`).
                let mut properties: std::collections::HashMap<String, common::PropertyValue> =
                    std::collections::HashMap::new();
                for (k, v) in row {
                    if let Some(prop_name) = k.strip_prefix("n.") {
                        if !prop_name.starts_with("__") {
                            properties.insert(prop_name.to_string(), value_to_proto_pub(v));
                        }
                    }
                }

                Some(graph::Node {
                    node_id,
                    labels,
                    properties,
                    element_id: NodeId::from_raw(node_id).to_element_id(),
                })
            })
            .collect();

        Ok(Response::new(graph::TraverseResponse {
            nodes,
            edges: vec![],
            pagination: None,
        }))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
