use std::sync::{Arc, Mutex};

use tonic::{Request, Response, Status};

use coordinode_core::graph::types::Value;
use coordinode_embed::Database;

use crate::proto::{common, graph};
use crate::services::cypher::{proto_to_value_pub, value_to_proto_pub};

/// Backtick-escape a Cypher identifier (label, relationship type, or property key).
fn cypher_ident(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}

pub struct GraphServiceImpl {
    database: Arc<Mutex<Database>>,
}

impl GraphServiceImpl {
    pub fn new(database: Arc<Mutex<Database>>) -> Self {
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
            let mut db = self.database.lock().unwrap_or_else(|e| e.into_inner());
            db.execute_cypher_with_params(&cypher, params)
                .map_err(|e| Status::internal(format!("create_node error: {e}")))?
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
        let rows = {
            let mut db = self.database.lock().unwrap_or_else(|e| e.into_inner());
            db.execute_cypher_with_params("MATCH (n) WHERE n = $id RETURN *", params)
                .map_err(|e| Status::internal(format!("get_node error: {e}")))?
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
            let mut db = self.database.lock().unwrap_or_else(|e| e.into_inner());
            db.execute_cypher_with_params(&cypher, params)
                .map_err(|e| Status::internal(format!("create_edge error: {e}")))?
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

        let rows = {
            let mut db = self.database.lock().unwrap_or_else(|e| e.into_inner());
            db.execute_cypher_with_params(&cypher, params)
                .map_err(|e| Status::internal(format!("traverse error: {e}")))?
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
mod tests {
    use super::*;
    use crate::proto::graph::graph_service_server::GraphService;

    fn test_service() -> (GraphServiceImpl, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("tempdir");
        let database = Arc::new(Mutex::new(
            Database::open(dir.path()).expect("open database"),
        ));
        (GraphServiceImpl::new(database), dir)
    }

    /// create_node returns a non-zero node_id.
    #[tokio::test]
    async fn create_node_returns_nonzero_id() {
        let (svc, _dir) = test_service();

        let resp = svc
            .create_node(Request::new(graph::CreateNodeRequest {
                labels: vec!["Person".to_string()],
                properties: std::collections::HashMap::new(),
            }))
            .await
            .expect("create_node should succeed");

        assert!(resp.into_inner().node_id > 0, "node_id must be > 0");
    }

    /// create_node persists the node so it is findable via Cypher.
    #[tokio::test]
    async fn create_node_persists() {
        let (svc, _dir) = test_service();

        svc.create_node(Request::new(graph::CreateNodeRequest {
            labels: vec!["Thing".to_string()],
            properties: {
                let mut m = std::collections::HashMap::new();
                m.insert(
                    "tag".to_string(),
                    common::PropertyValue {
                        value: Some(crate::proto::common::property_value::Value::StringValue(
                            "persist-test".to_string(),
                        )),
                    },
                );
                m
            },
        }))
        .await
        .expect("create should succeed");

        let rows = {
            let mut db = svc.database.lock().unwrap();
            db.execute_cypher("MATCH (n:Thing {tag: 'persist-test'}) RETURN n.tag")
                .expect("cypher should succeed")
        };
        assert_eq!(rows.len(), 1, "node should be findable via Cypher");
    }

    /// get_node returns the created node with matching id.
    #[tokio::test]
    async fn get_node_returns_created_node() {
        let (svc, _dir) = test_service();

        let created = svc
            .create_node(Request::new(graph::CreateNodeRequest {
                labels: vec!["Item".to_string()],
                properties: std::collections::HashMap::new(),
            }))
            .await
            .expect("create should succeed")
            .into_inner();

        let fetched = svc
            .get_node(Request::new(graph::GetNodeRequest {
                node_id: created.node_id,
            }))
            .await
            .expect("get_node should succeed")
            .into_inner();

        assert_eq!(fetched.node_id, created.node_id);
    }

    /// get_node returns labels and properties for the created node.
    ///
    /// Regression: `RETURN n, n.__label__` strips `n.*` in Project; `RETURN *` is required.
    #[tokio::test]
    async fn get_node_returns_labels_and_properties() {
        let (svc, _dir) = test_service();

        let created = svc
            .create_node(Request::new(graph::CreateNodeRequest {
                labels: vec!["Widget".to_string()],
                properties: {
                    let mut m = std::collections::HashMap::new();
                    m.insert(
                        "size".to_string(),
                        common::PropertyValue {
                            value: Some(crate::proto::common::property_value::Value::StringValue(
                                "large".to_string(),
                            )),
                        },
                    );
                    m
                },
            }))
            .await
            .expect("create should succeed")
            .into_inner();

        let fetched = svc
            .get_node(Request::new(graph::GetNodeRequest {
                node_id: created.node_id,
            }))
            .await
            .expect("get_node should succeed")
            .into_inner();

        assert_eq!(fetched.node_id, created.node_id);
        assert!(
            !fetched.labels.is_empty(),
            "get_node must populate labels, got empty vec"
        );
        assert!(
            fetched.labels.contains(&"Widget".to_string()),
            "labels must contain 'Widget', got {:?}",
            fetched.labels
        );
        assert!(
            fetched.properties.contains_key("size"),
            "get_node must populate properties, got {:?}",
            fetched.properties.keys().collect::<Vec<_>>()
        );
    }

    /// get_node returns not_found for a non-existent node.
    #[tokio::test]
    async fn get_node_not_found() {
        let (svc, _dir) = test_service();

        let result = svc
            .get_node(Request::new(graph::GetNodeRequest { node_id: 99999 }))
            .await;

        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::NotFound);
    }

    /// create_edge connects two nodes and returns edge_id > 0.
    #[tokio::test]
    async fn create_edge_returns_nonzero_id() {
        let (svc, _dir) = test_service();

        let a = svc
            .create_node(Request::new(graph::CreateNodeRequest {
                labels: vec!["A".to_string()],
                properties: std::collections::HashMap::new(),
            }))
            .await
            .expect("create A")
            .into_inner();

        let b = svc
            .create_node(Request::new(graph::CreateNodeRequest {
                labels: vec!["B".to_string()],
                properties: std::collections::HashMap::new(),
            }))
            .await
            .expect("create B")
            .into_inner();

        let edge = svc
            .create_edge(Request::new(graph::CreateEdgeRequest {
                edge_type: "KNOWS".to_string(),
                source_node_id: a.node_id,
                target_node_id: b.node_id,
                properties: std::collections::HashMap::new(),
            }))
            .await
            .expect("create_edge should succeed")
            .into_inner();

        assert!(edge.edge_id > 0, "edge_id must be > 0");
        assert_eq!(edge.source_node_id, a.node_id);
        assert_eq!(edge.target_node_id, b.node_id);
    }

    /// Regression: node created via create_node RPC must be visible via CypherService
    /// sharing the same Database instance.
    ///
    /// On the live server, `create_node` was observed to return `node_id = 0` and
    /// the node was not findable via `MATCH`. This test verifies cross-service
    /// persistence: GraphService write → CypherService read must observe the mutation.
    ///
    /// If this test fails, the two services are using different Database instances
    /// (broken wiring in `main.rs`).
    #[tokio::test]
    async fn create_node_visible_via_cypher_service() {
        use crate::proto::query;
        use crate::services::cypher::CypherServiceImpl;
        use coordinode_query::advisor::nplus1::NPlus1Detector;
        use coordinode_query::advisor::QueryRegistry;
        use query::cypher_service_server::CypherService as _;

        let dir = tempfile::tempdir().expect("tempdir");
        let database = Arc::new(Mutex::new(
            Database::open(dir.path()).expect("open database"),
        ));

        // Both services share the same Arc — exactly as main.rs wires them.
        let graph_svc = GraphServiceImpl::new(Arc::clone(&database));
        let cypher_svc = CypherServiceImpl::new(
            Arc::clone(&database),
            Arc::new(QueryRegistry::new()),
            Arc::new(NPlus1Detector::new()),
        );

        // Create node via GraphService RPC.
        let created = graph_svc
            .create_node(Request::new(graph::CreateNodeRequest {
                labels: vec!["RegTest".to_string()],
                properties: std::collections::HashMap::new(),
            }))
            .await
            .expect("create_node should succeed")
            .into_inner();

        assert!(
            created.node_id > 0,
            "node_id must be > 0, got {}",
            created.node_id
        );

        // Verify via CypherService that the node exists.
        let resp = cypher_svc
            .execute_cypher(Request::new(query::ExecuteCypherRequest {
                query: "MATCH (n:RegTest) RETURN n".to_string(),
                parameters: std::collections::HashMap::new(),
            }))
            .await
            .expect("execute_cypher should succeed")
            .into_inner();

        assert_eq!(
            resp.rows.len(),
            1,
            "node created via GraphService must be visible via CypherService, got {} rows",
            resp.rows.len()
        );
    }

    /// cypher_ident escapes backticks.
    #[test]
    fn cypher_ident_escapes() {
        assert_eq!(cypher_ident("Person"), "`Person`");
        assert_eq!(cypher_ident("my`type"), "`my``type`");
    }

    /// Helper: create A -[LINK]-> B and return (svc, a_id, b_id).
    async fn setup_two_nodes_with_edge() -> (GraphServiceImpl, tempfile::TempDir, u64, u64) {
        let (svc, dir) = test_service();

        let a = svc
            .create_node(Request::new(graph::CreateNodeRequest {
                labels: vec!["X".to_string()],
                properties: std::collections::HashMap::new(),
            }))
            .await
            .expect("create A")
            .into_inner();

        let b = svc
            .create_node(Request::new(graph::CreateNodeRequest {
                labels: vec!["Y".to_string()],
                properties: std::collections::HashMap::new(),
            }))
            .await
            .expect("create B")
            .into_inner();

        svc.create_edge(Request::new(graph::CreateEdgeRequest {
            edge_type: "LINK".to_string(),
            source_node_id: a.node_id,
            target_node_id: b.node_id,
            properties: std::collections::HashMap::new(),
        }))
        .await
        .expect("create edge")
        .into_inner();

        (svc, dir, a.node_id, b.node_id)
    }

    /// Regression: traverse OUTBOUND from A must reach B (default behaviour).
    #[tokio::test]
    async fn traverse_outbound_reaches_target() {
        let (svc, _dir, a_id, b_id) = setup_two_nodes_with_edge().await;

        let resp = svc
            .traverse(Request::new(graph::TraverseRequest {
                start_node_id: a_id,
                max_depth: 1,
                edge_type: "LINK".to_string(),
                direction: graph::TraversalDirection::Outbound as i32,
                pagination: None,
            }))
            .await
            .expect("traverse outbound should succeed")
            .into_inner();

        let ids: Vec<u64> = resp.nodes.iter().map(|n| n.node_id).collect();
        assert!(
            ids.contains(&b_id),
            "outbound traverse from A must reach B; got ids={ids:?}"
        );
    }

    /// Regression: traverse INBOUND from B must reach A.
    ///
    /// Previously the service ignored req.direction and always emitted `->`,
    /// so INBOUND from B would return nothing. This test verifies the fix.
    #[tokio::test]
    async fn traverse_inbound_reaches_source() {
        let (svc, _dir, a_id, b_id) = setup_two_nodes_with_edge().await;

        let resp = svc
            .traverse(Request::new(graph::TraverseRequest {
                start_node_id: b_id,
                max_depth: 1,
                edge_type: "LINK".to_string(),
                direction: graph::TraversalDirection::Inbound as i32,
                pagination: None,
            }))
            .await
            .expect("traverse inbound should succeed")
            .into_inner();

        let ids: Vec<u64> = resp.nodes.iter().map(|n| n.node_id).collect();
        assert!(
            ids.contains(&a_id),
            "inbound traverse from B must reach A; got ids={ids:?}"
        );
    }

    /// Regression: traverse BOTH from B must also reach A via the reverse edge.
    #[tokio::test]
    async fn traverse_both_reaches_source_from_target() {
        let (svc, _dir, a_id, b_id) = setup_two_nodes_with_edge().await;

        let resp = svc
            .traverse(Request::new(graph::TraverseRequest {
                start_node_id: b_id,
                max_depth: 1,
                edge_type: "LINK".to_string(),
                direction: graph::TraversalDirection::Both as i32,
                pagination: None,
            }))
            .await
            .expect("traverse both should succeed")
            .into_inner();

        let ids: Vec<u64> = resp.nodes.iter().map(|n| n.node_id).collect();
        assert!(
            ids.contains(&a_id),
            "BOTH traverse from B must reach A; got ids={ids:?}"
        );
    }

    /// Regression: OUTBOUND from B must NOT reach A (edge is A→B, not B→A).
    #[tokio::test]
    async fn traverse_outbound_does_not_traverse_reverse() {
        let (svc, _dir, _a_id, b_id) = setup_two_nodes_with_edge().await;

        let resp = svc
            .traverse(Request::new(graph::TraverseRequest {
                start_node_id: b_id,
                max_depth: 1,
                edge_type: "LINK".to_string(),
                direction: graph::TraversalDirection::Outbound as i32,
                pagination: None,
            }))
            .await
            .expect("traverse outbound from B should succeed")
            .into_inner();

        // B has no outbound LINK edges — result must be empty.
        assert!(
            resp.nodes.is_empty(),
            "outbound traverse from B must return nothing (edge is A→B); got {:?}",
            resp.nodes.len()
        );
    }

    /// Regression G079: traverse must populate labels and properties in returned nodes.
    ///
    /// Before fix: `labels: vec![]`, `properties: HashMap::new()` always.
    /// After fix: labels contain the primary label, properties contain the node's properties.
    #[tokio::test]
    async fn traverse_returns_labels_and_properties() {
        let (svc, _dir) = test_service();

        // Create a source node.
        let src = svc
            .create_node(Request::new(graph::CreateNodeRequest {
                labels: vec!["Source".to_string()],
                properties: std::collections::HashMap::new(),
            }))
            .await
            .expect("create source")
            .into_inner();

        // Create a target node with a label and a property.
        let dst = svc
            .create_node(Request::new(graph::CreateNodeRequest {
                labels: vec!["Target".to_string()],
                properties: {
                    let mut m = std::collections::HashMap::new();
                    m.insert(
                        "color".to_string(),
                        common::PropertyValue {
                            value: Some(crate::proto::common::property_value::Value::StringValue(
                                "blue".to_string(),
                            )),
                        },
                    );
                    m
                },
            }))
            .await
            .expect("create target")
            .into_inner();

        svc.create_edge(Request::new(graph::CreateEdgeRequest {
            edge_type: "POINTS_TO".to_string(),
            source_node_id: src.node_id,
            target_node_id: dst.node_id,
            properties: std::collections::HashMap::new(),
        }))
        .await
        .expect("create edge");

        let resp = svc
            .traverse(Request::new(graph::TraverseRequest {
                start_node_id: src.node_id,
                max_depth: 1,
                edge_type: "POINTS_TO".to_string(),
                direction: graph::TraversalDirection::Outbound as i32,
                pagination: None,
            }))
            .await
            .expect("traverse should succeed")
            .into_inner();

        assert_eq!(resp.nodes.len(), 1, "should return exactly the target node");
        let node = &resp.nodes[0];

        assert_eq!(node.node_id, dst.node_id, "node_id must match");

        // G079 regression: labels must not be empty.
        assert!(
            !node.labels.is_empty(),
            "traverse must populate labels, got empty vec"
        );
        assert!(
            node.labels.contains(&"Target".to_string()),
            "labels must contain 'Target', got {:?}",
            node.labels
        );

        // G079 regression: properties must not be empty.
        assert!(
            node.properties.contains_key("color"),
            "traverse must populate properties, got {:?}",
            node.properties.keys().collect::<Vec<_>>()
        );
    }
}
