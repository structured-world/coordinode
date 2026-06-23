use super::*;
use crate::proto::graph::graph_service_server::GraphService;

fn test_service() -> (GraphServiceImpl, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let database = Arc::new(RwLock::new(
        Database::open(dir.path()).expect("open database"),
    ));
    (GraphServiceImpl::new(database), dir)
}

/// create_nodes_batch ingests N nodes in one request, returns them in
/// input order with well-formed identifiers. Verifies the single-RPC
/// batch path (UNWIND $rows AS r CREATE …) round-trips end-to-end.
#[tokio::test]
async fn create_nodes_batch_returns_all_nodes_in_order() {
    let (svc, _dir) = test_service();

    let mk = |i: u32| {
        let mut props = std::collections::HashMap::new();
        props.insert(
            "i".to_string(),
            common::PropertyValue {
                value: Some(common::property_value::Value::IntValue(i as i64)),
            },
        );
        graph::CreateNodeRequest {
            labels: vec!["BatchItem".to_string()],
            properties: props,
        }
    };
    let batch: Vec<_> = (0..5).map(mk).collect();

    let resp = svc
        .create_nodes_batch(Request::new(graph::CreateNodesBatchRequest {
            nodes: batch,
        }))
        .await
        .expect("create_nodes_batch should succeed");
    let nodes = resp.into_inner().nodes;

    assert_eq!(nodes.len(), 5, "must return one node per input");
    // All ids non-zero, all distinct, labels propagated.
    let mut seen = std::collections::HashSet::new();
    for (i, n) in nodes.iter().enumerate() {
        assert!(n.node_id > 0, "node[{i}] has zero id");
        assert!(seen.insert(n.node_id), "duplicate id on node[{i}]");
        assert_eq!(n.labels, vec!["BatchItem".to_string()]);
        assert_eq!(n.element_id.len(), 13, "node[{i}] missing element_id");
    }
}

/// Mixed-label batches are rejected with InvalidArgument because
/// Cypher can't add labels dynamically per UNWIND row.
#[tokio::test]
async fn create_nodes_batch_mixed_labels_rejected() {
    let (svc, _dir) = test_service();
    let resp = svc
        .create_nodes_batch(Request::new(graph::CreateNodesBatchRequest {
            nodes: vec![
                graph::CreateNodeRequest {
                    labels: vec!["A".to_string()],
                    properties: Default::default(),
                },
                graph::CreateNodeRequest {
                    labels: vec!["B".to_string()],
                    properties: Default::default(),
                },
            ],
        }))
        .await;
    let err = resp.expect_err("mixed labels must error");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);
}

/// Empty batch is a valid no-op — returns empty list, no error.
#[tokio::test]
async fn create_nodes_batch_empty_is_noop() {
    let (svc, _dir) = test_service();
    let resp = svc
        .create_nodes_batch(Request::new(graph::CreateNodesBatchRequest {
            nodes: vec![],
        }))
        .await
        .expect("empty batch must succeed");
    assert!(resp.into_inner().nodes.is_empty());
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

/// create_node populates element_id with a valid Crockford base32 encoding
/// that roundtrips back to the same node_id. Regression: previously the
/// field did not exist on proto Node; clients had no canonical identifier
/// per ADR-022.
#[tokio::test]
async fn create_node_returns_well_formed_element_id() {
    let (svc, _dir) = test_service();

    let resp = svc
        .create_node(Request::new(graph::CreateNodeRequest {
            labels: vec!["Member".to_string()],
            properties: std::collections::HashMap::new(),
        }))
        .await
        .expect("create_node");
    let node = resp.into_inner();

    assert_eq!(
        node.element_id.len(),
        13,
        "element_id must be a 13-char Crockford base32 token, got {:?}",
        node.element_id
    );

    let decoded = coordinode_core::graph::node::NodeId::from_element_id(&node.element_id)
        .expect("element_id must decode");
    assert_eq!(
        decoded.as_raw(),
        node.node_id,
        "element_id must roundtrip to the same node_id"
    );
}

/// create_edge populates element_id with the source:target form (two
/// Crockford tokens joined by ":"). The ordering is canonical (always
/// src:tgt) so callers can compare endpoint identity.
#[tokio::test]
async fn create_edge_returns_well_formed_element_id() {
    let (svc, _dir) = test_service();

    let src = svc
        .create_node(Request::new(graph::CreateNodeRequest {
            labels: vec!["Person".to_string()],
            properties: std::collections::HashMap::new(),
        }))
        .await
        .expect("src")
        .into_inner();
    let dst = svc
        .create_node(Request::new(graph::CreateNodeRequest {
            labels: vec!["Person".to_string()],
            properties: std::collections::HashMap::new(),
        }))
        .await
        .expect("dst")
        .into_inner();

    let edge = svc
        .create_edge(Request::new(graph::CreateEdgeRequest {
            edge_type: "KNOWS".to_string(),
            source_node_id: src.node_id,
            target_node_id: dst.node_id,
            properties: std::collections::HashMap::new(),
        }))
        .await
        .expect("edge")
        .into_inner();

    let parts: Vec<&str> = edge.element_id.split(':').collect();
    assert_eq!(
        parts.len(),
        2,
        "element_id must be src:tgt, got {:?}",
        edge.element_id
    );
    assert_eq!(parts[0], src.element_id);
    assert_eq!(parts[1], dst.element_id);
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
        let mut db = svc.database.write();
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
    let database = Arc::new(RwLock::new(
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
            read_preference: 0, // UNSPECIFIED → Primary
            read_concern: None, // UNSPECIFIED → Local
            write_concern: None,
            transaction_id: 0,
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
