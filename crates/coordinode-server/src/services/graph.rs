use std::sync::Arc;

// no-std: spin::RwLock (drop-in). parking_lot::RwLock over std::sync: its
// uncontended path never enters the kernel.
use parking_lot::RwLock;

use tonic::{Request, Response, Status};

use coordinode_core::graph::node::NodeId;
use coordinode_core::graph::types::Value;
use coordinode_embed::Database;

use crate::proto::{common, graph};
use crate::services::cypher::{proto_to_value_pub, value_to_proto_pub};
use crate::services::db_err_to_status;

/// The version of each node a read returned, as that read saw it: the commit
/// that wrote the state the read returned, read at the read's own timestamp
/// (`read_ts`, inclusive), so a node and its version are of one write
/// whatever landed since. A node the read returned is there at its
/// timestamp, so each has a version.
///
/// # Errors
///
/// The read's timestamp is no longer retained, or a node's original commit
/// is not known; the call fails rather than report a version it cannot name.
pub(crate) fn versions_at(
    db: &Database,
    read_ts: Option<u64>,
    nodes: impl IntoIterator<Item = u64>,
) -> Result<std::collections::HashMap<u64, u64>, Status> {
    let mut versions = std::collections::HashMap::new();
    for node_id in nodes {
        // Only an empty view has no timestamp, and it returns no node.
        let read_ts = read_ts.ok_or_else(|| {
            Status::internal("a read returned a node but named no timestamp it read at")
        })?;
        let version = db
            .node_version_at(NodeId::from_raw(node_id), read_ts)
            .map_err(|e| db_err_to_status("node version", e))?
            .ok_or_else(|| {
                Status::internal(format!(
                    "node {node_id} returned by a read is absent at the read's timestamp"
                ))
            })?;
        versions.insert(node_id, version);
    }
    Ok(versions)
}

/// The node a row of `RETURN *` over a node variable `n` holds.
pub(crate) fn node_of(row: &coordinode_query::executor::Row) -> Option<u64> {
    match row.get("n")? {
        Value::Int(id) => Some(*id as u64),
        _ => None,
    }
}

/// The version [`versions_at`] found for `node_id`.
pub(crate) fn version_of(
    versions: &std::collections::HashMap<u64, u64>,
    node_id: u64,
) -> Result<u64, Status> {
    versions
        .get(&node_id)
        .copied()
        .ok_or_else(|| Status::internal(format!("no version was read for node {node_id}")))
}

/// The timestamp a statement that wrote committed at: the version of every
/// record it wrote.
fn written_at(result: &coordinode_embed::db::CypherResult, call: &str) -> Result<u64, Status> {
    result
        .commit_ts()
        .ok_or_else(|| Status::internal(format!("{call}: the write reported no commit timestamp")))
}

/// Backtick-escape a Cypher identifier (label, relationship type, or property key).
fn cypher_ident(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}

pub struct GraphServiceImpl {
    /// The database of the namespace each call names.
    databases: super::namespace::Databases,
}

impl GraphServiceImpl {
    /// A service answering from `database`'s deployment, each call in the
    /// namespace it names.
    pub fn new(database: Arc<RwLock<Database>>) -> Self {
        Self::serving(super::namespace::Databases::of(database))
    }

    /// A service answering from `databases`, each call in the namespace it
    /// names.
    pub fn serving(databases: super::namespace::Databases) -> Self {
        Self { databases }
    }

    /// Run the write `cypher` a typed call builds in `database`, carrying out
    /// `operation` when the client named one.
    fn write(
        database: &RwLock<Database>,
        cypher: &str,
        params: std::collections::HashMap<String, Value>,
        operation: Option<coordinode_embed::operation::OperationDescriptor>,
    ) -> Result<coordinode_embed::db::CypherResult, coordinode_embed::DatabaseError> {
        let options = coordinode_embed::db::StatementOptions {
            operation,
            ..Default::default()
        };
        super::blocking(|| {
            database.read().execute_cypher_shared_with(
                cypher,
                (!params.is_empty()).then_some(params),
                None,
                &options,
            )
        })
    }
}

#[tonic::async_trait]
impl graph::graph_service_server::GraphService for GraphServiceImpl {
    async fn create_node(
        &self,
        request: Request<graph::CreateNodeRequest>,
    ) -> Result<Response<graph::Node>, Status> {
        let database = self.databases.for_call(request.metadata())?;
        let req = request.into_inner();
        let operation = super::operation::optional_descriptor("operation", req.operation.as_ref())?;

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

        let result = Self::write(&database, &cypher, params, operation)
            .map_err(|e| db_err_to_status("create_node", e))?;
        // The version the node was created at, so a caller that means to
        // write it again conditionally already has what to state.
        let version = written_at(&result, "create_node")?;

        let node_id = match result.rows.first().and_then(|r| r.get("n")) {
            Some(Value::Int(id)) => *id as u64,
            _ => {
                return Err(Status::internal(
                    "create_node: executor did not return node id",
                ));
            }
        };

        Ok(Response::new(graph::Node {
            node_id,
            labels: req.labels,
            properties: req.properties,
            element_id: NodeId::from_raw(node_id).to_element_id(),
            version,
        }))
    }

    async fn create_nodes_batch(
        &self,
        request: Request<graph::CreateNodesBatchRequest>,
    ) -> Result<Response<graph::CreateNodesBatchResponse>, Status> {
        let database = self.databases.for_call(request.metadata())?;
        let req = request.into_inner();
        let operation = super::operation::optional_descriptor("operation", req.operation.as_ref())?;
        // The batch names its operation; one on an entry would bind part of
        // the batch.
        if let Some(i) = req.nodes.iter().position(|n| n.operation.is_some()) {
            return Err(super::error_details::invalid_field(
                format!("nodes[{i}].operation"),
                "an entry of a batch carries no operation; name it on the batch",
            ));
        }

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

        let result = Self::write(&database, &cypher, params, operation)
            .map_err(|e| db_err_to_status("create_nodes_batch", e))?;
        // One statement, one commit: the version of every node it created.
        let version = written_at(&result, "create_nodes_batch")?;
        let result_rows = result.rows;

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
                        )));
                    }
                };
                Ok(graph::Node {
                    node_id,
                    labels: labels.clone(),
                    properties: req.nodes[i].properties.clone(),
                    element_id: NodeId::from_raw(node_id).to_element_id(),
                    version,
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
        let database = self.databases.for_call(request.metadata())?;
        let req = request.into_inner();

        let mut params: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
        params.insert("id".to_string(), Value::Int(req.node_id as i64));

        // `n = $id` compares the node variable (Value::Int(node_id) from NodeScan)
        // with the target id, which is the correct way to filter by internal node ID.
        // `RETURN *` copies ALL columns from NodeScan: `n`, `n.__label__`, `n.<prop>`.
        // An explicit `RETURN n, n.__label__` would strip `n.*` properties in the Project step.
        // get_node is read-only — shared lock lets parallel reads
        // proceed without blocking on each other. The version is the one the
        // read saw, read at its timestamp under the same lock.
        let (rows, versions) = super::blocking(|| {
            let db = database.read();
            let result = db
                .execute_cypher_shared(
                    "MATCH (n) WHERE n = $id RETURN *",
                    Some(params),
                    None,
                    None,
                    None,
                )
                .map_err(|e| db_err_to_status("get_node", e))?;
            let versions = versions_at(
                &db,
                result.write_stats.read_ts,
                result.rows.iter().filter_map(node_of),
            )?;
            Ok::<_, Status>((result.rows, versions))
        })?;

        let row = rows
            .first()
            .ok_or_else(|| Status::not_found(format!("node {} not found", req.node_id)))?;

        let node_id = node_of(row)
            .ok_or_else(|| Status::not_found(format!("node {} not found", req.node_id)))?;

        // Primary label stored by NodeScan as `n.__label__`.
        let label = match row.get("n.__label__") {
            Some(Value::String(s)) => s.clone(),
            _ => String::new(),
        };

        // Collect all `n.*` properties (excluding `n.__label__` and `n` itself).
        let mut properties: std::collections::HashMap<String, common::PropertyValue> =
            std::collections::HashMap::new();
        for (k, v) in row {
            if let Some(prop_name) = k.strip_prefix("n.")
                && !prop_name.starts_with("__")
            {
                properties.insert(prop_name.to_string(), value_to_proto_pub(v));
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
            // What a reader needs to write this node back conditionally: the
            // version of the very state returned.
            version: version_of(&versions, node_id)?,
        }))
    }

    async fn create_edge(
        &self,
        request: Request<graph::CreateEdgeRequest>,
    ) -> Result<Response<graph::Edge>, Status> {
        let database = self.databases.for_call(request.metadata())?;
        let req = request.into_inner();
        let operation = super::operation::optional_descriptor("operation", req.operation.as_ref())?;

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

        let rows = Self::write(&database, &cypher, params, operation)
            .map_err(|e| db_err_to_status("create_edge", e))?
            .rows;

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
        let database = self.databases.for_call(request.metadata())?;
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

        // Traversal is read-only: shared lock for parallel reads. Every node
        // reports the version the traversal saw, read at its timestamp.
        let nodes = super::blocking(|| {
            let db = database.read();
            let result = db
                .execute_cypher_shared(&cypher, Some(params), None, None, None)
                .map_err(|e| db_err_to_status("traverse", e))?;
            let versions = versions_at(
                &db,
                result.write_stats.read_ts,
                result.rows.iter().filter_map(node_of),
            )?;
            traversed_nodes(&result.rows, &versions)
        })?;

        Ok(Response::new(graph::TraverseResponse {
            nodes,
            edges: vec![],
            pagination: None,
        }))
    }
}

/// The nodes a traversal returned, each with the version the traversal saw
/// ([`versions_at`]).
fn traversed_nodes(
    rows: &[coordinode_query::executor::Row],
    versions: &std::collections::HashMap<u64, u64>,
) -> Result<Vec<graph::Node>, Status> {
    rows.iter()
        .filter_map(|row| {
            // `RETURN *` copies all columns from the Traverse operator row:
            // `n` = node_id, `n.__label__` = primary label, `n.<prop>` = properties.
            let node_id = node_of(row)?;

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
                if let Some(prop_name) = k.strip_prefix("n.")
                    && !prop_name.starts_with("__")
                {
                    properties.insert(prop_name.to_string(), value_to_proto_pub(v));
                }
            }

            Some(version_of(versions, node_id).map(|version| graph::Node {
                node_id,
                labels,
                properties,
                element_id: NodeId::from_raw(node_id).to_element_id(),
                version,
            }))
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
