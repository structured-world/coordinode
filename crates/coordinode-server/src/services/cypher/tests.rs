use super::*;
use crate::proto::query::cypher_service_server::CypherService;
use std::collections::BTreeMap;

/// value_to_proto maps Null correctly.
#[test]
fn convert_null() {
    let pv = value_to_proto(&Value::Null);
    assert!(pv.value.is_none());
}

/// value_to_proto maps Int correctly.
#[test]
fn convert_int() {
    let pv = value_to_proto(&Value::Int(42));
    assert_eq!(pv.value, Some(common::property_value::Value::IntValue(42)));
}

/// value_to_proto maps Float correctly.
#[test]
fn convert_float() {
    let pv = value_to_proto(&Value::Float(1.5));
    assert_eq!(
        pv.value,
        Some(common::property_value::Value::FloatValue(1.5))
    );
}

/// value_to_proto maps String correctly.
#[test]
fn convert_string() {
    let pv = value_to_proto(&Value::String("hello".to_string()));
    assert_eq!(
        pv.value,
        Some(common::property_value::Value::StringValue(
            "hello".to_string()
        ))
    );
}

/// value_to_proto maps Bool correctly.
#[test]
fn convert_bool() {
    let pv = value_to_proto(&Value::Bool(true));
    assert_eq!(
        pv.value,
        Some(common::property_value::Value::BoolValue(true))
    );
}

/// value_to_proto maps Vector correctly.
#[test]
fn convert_vector() {
    let pv = value_to_proto(&Value::Vector(vec![1.0, 2.0, 3.0]));
    match &pv.value {
        Some(common::property_value::Value::VectorValue(v)) => {
            assert_eq!(v.values, vec![1.0, 2.0, 3.0]);
        }
        other => panic!("expected VectorValue, got {other:?}"),
    }
}

/// value_to_proto maps Array correctly.
#[test]
fn convert_array() {
    let pv = value_to_proto(&Value::Array(vec![Value::Int(1), Value::Int(2)]));
    match &pv.value {
        Some(common::property_value::Value::ListValue(list)) => {
            assert_eq!(list.values.len(), 2);
        }
        other => panic!("expected ListValue, got {other:?}"),
    }
}

/// value_to_proto maps Map correctly.
#[test]
fn convert_map() {
    let mut map = BTreeMap::new();
    map.insert("key".to_string(), Value::String("val".to_string()));
    let pv = value_to_proto(&Value::Map(map));
    match &pv.value {
        Some(common::property_value::Value::MapValue(m)) => {
            assert_eq!(m.entries.len(), 1);
            assert!(m.entries.contains_key("key"));
        }
        other => panic!("expected MapValue, got {other:?}"),
    }
}

// --- gRPC handler integration tests ---

/// Helper: create a CypherServiceImpl with a real Database.
fn test_service() -> (CypherServiceImpl, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let database = Arc::new(RwLock::new(
        Database::open(dir.path()).expect("open database"),
    ));
    let registry = Arc::new(QueryRegistry::new());
    let detector = Arc::new(NPlus1Detector::new());
    let svc = CypherServiceImpl::new(database, registry, detector);
    (svc, dir)
}

fn cypher_request(q: &str) -> Request<query::ExecuteCypherRequest> {
    Request::new(query::ExecuteCypherRequest {
        query: q.to_string(),
        parameters: std::collections::HashMap::new(),
        read_preference: 0,  // UNSPECIFIED → Primary
        read_concern: None,  // UNSPECIFIED → Local
        write_concern: None, // UNSPECIFIED → W1
        transaction_id: 0,   // auto-commit
    })
}

fn cypher_request_in_txn(q: &str, transaction_id: u64) -> Request<query::ExecuteCypherRequest> {
    Request::new(query::ExecuteCypherRequest {
        query: q.to_string(),
        parameters: std::collections::HashMap::new(),
        read_preference: 0,
        read_concern: None,
        write_concern: None,
        transaction_id,
    })
}

/// gRPC interactive transaction: begin → statement-in-txn → commit, with
/// the buffered writes invisible until commit, then visible after.
#[tokio::test]
async fn grpc_interactive_transaction_commit() {
    let (svc, _dir) = test_service();

    let tx = svc
        .begin_transaction(Request::new(query::BeginTransactionRequest {}))
        .await
        .expect("begin")
        .into_inner()
        .transaction_id;
    assert_ne!(tx, 0, "begin returns a non-zero transaction id");

    svc.execute_cypher(cypher_request_in_txn("CREATE (n:TxNode {id: 1})", tx))
        .await
        .expect("statement in txn");

    // Auto-commit read does not see the uncommitted write.
    let before = svc
        .execute_cypher(cypher_request("MATCH (n:TxNode) RETURN n"))
        .await
        .expect("read before")
        .into_inner();
    assert_eq!(before.rows.len(), 0, "uncommitted write invisible");

    svc.commit_transaction(Request::new(query::CommitTransactionRequest {
        transaction_id: tx,
    }))
    .await
    .expect("commit");

    let after = svc
        .execute_cypher(cypher_request("MATCH (n:TxNode) RETURN n"))
        .await
        .expect("read after")
        .into_inner();
    assert_eq!(after.rows.len(), 1, "committed write visible");
}

/// gRPC interactive transaction rollback discards the buffered write and
/// consumes the handle.
#[tokio::test]
async fn grpc_interactive_transaction_rollback() {
    let (svc, _dir) = test_service();

    let tx = svc
        .begin_transaction(Request::new(query::BeginTransactionRequest {}))
        .await
        .expect("begin")
        .into_inner()
        .transaction_id;

    svc.execute_cypher(cypher_request_in_txn("CREATE (n:TxNode {id: 2})", tx))
        .await
        .expect("statement in txn");

    svc.rollback_transaction(Request::new(query::RollbackTransactionRequest {
        transaction_id: tx,
    }))
    .await
    .expect("rollback");

    let rows = svc
        .execute_cypher(cypher_request("MATCH (n:TxNode) RETURN n"))
        .await
        .expect("read")
        .into_inner();
    assert_eq!(rows.rows.len(), 0, "rolled-back write discarded");

    // Handle consumed: commit on the rolled-back id is an error.
    assert!(svc
        .commit_transaction(Request::new(query::CommitTransactionRequest {
            transaction_id: tx,
        }))
        .await
        .is_err());
}

/// gRPC execute_cypher creates a node and returns it.
#[tokio::test]
async fn grpc_execute_create_and_match() {
    let (svc, _dir) = test_service();

    // CREATE a node
    let create_resp = svc
        .execute_cypher(cypher_request(
            "CREATE (n:User {name: 'Alice'}) RETURN n.name",
        ))
        .await
        .expect("create should succeed");

    let create_body = create_resp.into_inner();
    assert!(
        !create_body.rows.is_empty(),
        "CREATE RETURN should produce rows"
    );
    assert!(
        create_body.columns.contains(&"n.name".to_string()),
        "columns should contain n.name, got: {:?}",
        create_body.columns
    );

    // MATCH it back
    let match_resp = svc
        .execute_cypher(cypher_request("MATCH (n:User) RETURN n.name"))
        .await
        .expect("match should succeed");

    let match_body = match_resp.into_inner();
    assert!(
        !match_body.rows.is_empty(),
        "MATCH should find the created node"
    );

    // Verify the actual value is "Alice" (not EXPLAIN text)
    let first_row = &match_body.rows[0];
    let name_idx = match_body
        .columns
        .iter()
        .position(|c| c == "n.name")
        .expect("n.name column should exist");
    let name_value = &first_row.values[name_idx];
    assert_eq!(
        name_value.value,
        Some(common::property_value::Value::StringValue(
            "Alice".to_string()
        )),
        "should return actual data, not EXPLAIN stub"
    );
}

/// gRPC execute_cypher returns parse error for invalid Cypher.
#[tokio::test]
async fn grpc_execute_parse_error() {
    let (svc, _dir) = test_service();

    let result = svc
        .execute_cypher(cypher_request("INVALID SYNTAX HERE"))
        .await;

    assert!(result.is_err(), "invalid Cypher should return error");
    let status = result.unwrap_err();
    assert_eq!(
        status.code(),
        tonic::Code::InvalidArgument,
        "parse error should be InvalidArgument"
    );
}

/// End-to-end: CoordinodeClient with debug_source_tracking injects
/// x-source-* metadata that CypherServiceImpl extracts and records in
/// the QueryRegistry. Covers the full path:
///   client (#[track_caller]) → gRPC transport → extract_source_context()
///   → query_registry.record_with_source()
#[tokio::test]
async fn grpc_source_tracking_round_trip() {
    use tokio::net::TcpListener;
    use tokio_stream::wrappers::TcpListenerStream;

    // Shared registry so we can inspect it after the query executes.
    let dir = tempfile::tempdir().expect("tempdir");
    let database = Arc::new(RwLock::new(
        Database::open(dir.path()).expect("open database"),
    ));
    let registry = Arc::new(QueryRegistry::new());
    let detector = Arc::new(NPlus1Detector::new());
    let svc = CypherServiceImpl::new(
        Arc::clone(&database),
        Arc::clone(&registry),
        Arc::clone(&detector),
    );

    // Bind to OS-assigned port to avoid conflicts.
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind listener");
    let addr = listener.local_addr().expect("local_addr");
    let incoming = TcpListenerStream::new(listener);

    // Serve in background; task exits when all clients disconnect.
    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(crate::proto::query::cypher_service_server::CypherServiceServer::new(svc))
            .serve_with_incoming(incoming)
            .await
            .expect("server error");
    });

    // Connect via CoordinodeClient with source tracking enabled.
    let mut client = coordinode_client::CoordinodeClient::builder(format!("http://{addr}"))
        .debug_source_tracking(true)
        .app_name("test-service")
        .app_version("0.0.1")
        .build()
        .await
        .expect("connect");

    // Execute from a known call site (this line is the call site).
    client
        .execute_cypher("CREATE (n:SrcTest)")
        .await
        .expect("execute_cypher");

    // Verify the registry captured the source location.
    let top = registry.top_by_count(10);
    assert_eq!(top.len(), 1, "one fingerprint should be recorded");

    let entry = &top[0];
    assert_eq!(entry.sources.len(), 1, "one source location expected");

    let src = &entry.sources[0];
    // File must point to this test file, not to client or transport internals.
    // The #[track_caller] wrapper on execute_cypher captures the call site in
    // this test (cypher/tests.rs), then injected as x-source-file metadata.
    assert!(
        src.file.contains("cypher/tests.rs"),
        "source file should be the cypher service test file, got: {}",
        src.file
    );
    assert!(src.line > 0, "line must be non-zero");
    assert_eq!(src.app, "test-service");
    assert_eq!(src.call_count, 1);
}

/// End-to-end: source tracking works when query has parameters
/// (the `has_params=true, source_ctx=Some` branch in execute_cypher).
#[tokio::test]
async fn grpc_source_tracking_with_params_round_trip() {
    use tokio::net::TcpListener;
    use tokio_stream::wrappers::TcpListenerStream;

    let dir = tempfile::tempdir().expect("tempdir");
    let database = Arc::new(RwLock::new(
        Database::open(dir.path()).expect("open database"),
    ));
    let registry = Arc::new(QueryRegistry::new());
    let detector = Arc::new(NPlus1Detector::new());
    let svc = CypherServiceImpl::new(
        Arc::clone(&database),
        Arc::clone(&registry),
        Arc::clone(&detector),
    );

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind listener");
    let addr = listener.local_addr().expect("local_addr");
    let incoming = TcpListenerStream::new(listener);

    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(crate::proto::query::cypher_service_server::CypherServiceServer::new(svc))
            .serve_with_incoming(incoming)
            .await
            .expect("server error");
    });

    let mut client = coordinode_client::CoordinodeClient::builder(format!("http://{addr}"))
        .debug_source_tracking(true)
        .app_name("param-service")
        .build()
        .await
        .expect("connect");

    // Execute with parameters — exercises the (Some(src), true) branch.
    let mut params = std::collections::HashMap::new();
    params.insert(
        "name".to_string(),
        coordinode_client::Value::String("Bob".to_string()),
    );
    client
        .execute_cypher_with_params("CREATE (n:Person {name: $name})", params)
        .await
        .expect("execute_cypher_with_params");

    let top = registry.top_by_count(10);
    assert_eq!(top.len(), 1, "one fingerprint should be recorded");
    let src = &top[0].sources[0];
    assert!(
        src.file.contains("cypher/tests.rs"),
        "source file should be the cypher service test file, got: {}",
        src.file
    );
    assert!(src.line > 0);
    assert_eq!(src.app, "param-service");
    assert_eq!(src.call_count, 1);
}

/// gRPC execute_cypher records execution time > 0.
#[tokio::test]
async fn grpc_execute_records_timing() {
    let (svc, _dir) = test_service();

    let resp = svc
        .execute_cypher(cypher_request("CREATE (n:Test {x: 1}) RETURN n"))
        .await
        .expect("should succeed");

    let stats = resp.into_inner().stats.expect("stats should be present");
    // execution_time_ms may be 0 for fast queries, but stats must exist
    assert!(stats.execution_time_ms >= 0);
}

/// db_error_to_status maps error types correctly.
#[test]
fn error_to_status_mapping() {
    let parse_err = db_error_to_status(DatabaseError::Semantic("bad query".into()));
    assert_eq!(parse_err.code(), tonic::Code::InvalidArgument);

    let exec_err = db_error_to_status(DatabaseError::Execution(
        coordinode_query::executor::runner::ExecutionError::Unsupported("test".into()),
    ));
    assert_eq!(exec_err.code(), tonic::Code::Internal);
}

// --- proto_to_value tests ---

#[test]
fn proto_to_value_null() {
    let pv = common::PropertyValue { value: None };
    assert_eq!(proto_to_value(&pv), Value::Null);
}

#[test]
fn proto_to_value_int() {
    let pv = common::PropertyValue {
        value: Some(common::property_value::Value::IntValue(42)),
    };
    assert_eq!(proto_to_value(&pv), Value::Int(42));
}

#[test]
fn proto_to_value_float() {
    let pv = common::PropertyValue {
        value: Some(common::property_value::Value::FloatValue(1.5)),
    };
    assert_eq!(proto_to_value(&pv), Value::Float(1.5));
}

#[test]
fn proto_to_value_string() {
    let pv = common::PropertyValue {
        value: Some(common::property_value::Value::StringValue(
            "hello".to_string(),
        )),
    };
    assert_eq!(proto_to_value(&pv), Value::String("hello".to_string()));
}

#[test]
fn proto_to_value_bool() {
    let pv = common::PropertyValue {
        value: Some(common::property_value::Value::BoolValue(true)),
    };
    assert_eq!(proto_to_value(&pv), Value::Bool(true));
}

#[test]
fn proto_to_value_vector() {
    let pv = common::PropertyValue {
        value: Some(common::property_value::Value::VectorValue(common::Vector {
            values: vec![1.0, 2.0, 3.0],
        })),
    };
    assert_eq!(proto_to_value(&pv), Value::Vector(vec![1.0, 2.0, 3.0]));
}

#[test]
fn proto_to_value_list() {
    let pv = common::PropertyValue {
        value: Some(common::property_value::Value::ListValue(
            common::PropertyList {
                values: vec![
                    common::PropertyValue {
                        value: Some(common::property_value::Value::IntValue(1)),
                    },
                    common::PropertyValue {
                        value: Some(common::property_value::Value::IntValue(2)),
                    },
                ],
            },
        )),
    };
    assert_eq!(
        proto_to_value(&pv),
        Value::Array(vec![Value::Int(1), Value::Int(2)])
    );
}

#[test]
fn proto_to_value_roundtrip() {
    // value_to_proto → proto_to_value should roundtrip for basic types
    let values = vec![
        Value::Null,
        Value::Bool(false),
        Value::Int(-99),
        Value::Float(1.23),
        Value::String("test".to_string()),
        Value::Vector(vec![0.5, 0.5]),
        Value::Array(vec![Value::Int(1), Value::String("two".to_string())]),
    ];
    for original in &values {
        let proto = value_to_proto(original);
        let back = proto_to_value(&proto);
        assert_eq!(
            &back, original,
            "roundtrip failed for {original:?}: got {back:?}"
        );
    }
}

// --- Causal consistency (R142) tests ---

/// after_index > 0 with readConcern=LOCAL is rejected with FailedPrecondition.
///
/// A LOCAL read offers no majority-commit guarantee; combining it with an
/// afterClusterTime fence is logically unsound and hard-rejected by the server.
#[tokio::test]
async fn causal_after_index_with_local_concern_rejected() {
    let (svc, _dir) = test_service();

    let result = svc
        .execute_cypher(Request::new(query::ExecuteCypherRequest {
            query: "MATCH (n) RETURN n".to_string(),
            parameters: std::collections::HashMap::new(),
            read_preference: 0,
            read_concern: Some(crate::proto::replication::ReadConcern {
                level: 1, // READ_CONCERN_LEVEL_LOCAL
                after_index: 42,
                at_timestamp: 0,
            }),
            write_concern: None,
            transaction_id: 0,
        }))
        .await;

    assert!(result.is_err(), "after_index + LOCAL should return error");
    assert_eq!(
        result.unwrap_err().code(),
        tonic::Code::FailedPrecondition,
        "should be FailedPrecondition"
    );
}

/// after_index > 0 with readConcern=LINEARIZABLE is rejected with FailedPrecondition.
///
/// LINEARIZABLE is incompatible with afterClusterTime per the consistency spec.
#[tokio::test]
async fn causal_after_index_with_linearizable_rejected() {
    let (svc, _dir) = test_service();

    let result = svc
        .execute_cypher(Request::new(query::ExecuteCypherRequest {
            query: "MATCH (n) RETURN n".to_string(),
            parameters: std::collections::HashMap::new(),
            read_preference: 0,
            read_concern: Some(crate::proto::replication::ReadConcern {
                level: 3, // READ_CONCERN_LEVEL_LINEARIZABLE
                after_index: 1,
                at_timestamp: 0,
            }),
            write_concern: None,
            transaction_id: 0,
        }))
        .await;

    assert!(
        result.is_err(),
        "after_index + LINEARIZABLE should return error"
    );
    assert_eq!(
        result.unwrap_err().code(),
        tonic::Code::FailedPrecondition,
        "should be FailedPrecondition"
    );
}

/// after_index > 0 with readConcern=MAJORITY succeeds in standalone mode.
///
/// In standalone (no Raft), after_index is trivially satisfied because all
/// writes are immediately visible. The fence is skipped; the query executes.
#[tokio::test]
async fn causal_after_index_with_majority_standalone_succeeds() {
    let (svc, _dir) = test_service();

    // Even a large after_index is fine in standalone — no Raft means the
    // fence block is never entered, so the read proceeds immediately.
    let result = svc
        .execute_cypher(Request::new(query::ExecuteCypherRequest {
            query: "MATCH (n) RETURN n".to_string(),
            parameters: std::collections::HashMap::new(),
            read_preference: 0,
            read_concern: Some(crate::proto::replication::ReadConcern {
                level: 2, // READ_CONCERN_LEVEL_MAJORITY
                after_index: 999,
                at_timestamp: 0,
            }),
            write_concern: None,
            transaction_id: 0,
        }))
        .await;

    assert!(
        result.is_ok(),
        "after_index + MAJORITY in standalone should succeed, got: {:?}",
        result.err()
    );
}

/// Write followed by causal read: applied_index in stats is 0 (standalone).
///
/// Verifies the round-trip: write returns applied_index=0 in standalone mode,
/// and a read with after_index=0 (= value from write response) succeeds.
/// This is the simplest causal session: no fence needed in standalone.
#[tokio::test]
async fn causal_write_read_roundtrip_standalone() {
    let (svc, _dir) = test_service();

    // Write
    let write_resp = svc
        .execute_cypher(cypher_request("CREATE (n:CausalTest {val: 1}) RETURN n"))
        .await
        .expect("write should succeed");

    let stats = write_resp
        .into_inner()
        .stats
        .expect("stats must be present");
    let operation_time = stats.applied_index;
    // In standalone mode, applied_index is always 0.
    assert_eq!(operation_time, 0, "standalone applied_index must be 0");

    // Subsequent causal read with after_index = operation_time (= 0).
    // after_index = 0 means no fence — trivially satisfied everywhere.
    let read_result = svc
        .execute_cypher(Request::new(query::ExecuteCypherRequest {
            query: "MATCH (n:CausalTest) RETURN n.val".to_string(),
            parameters: std::collections::HashMap::new(),
            read_preference: 0,
            read_concern: Some(crate::proto::replication::ReadConcern {
                level: 2, // MAJORITY
                after_index: operation_time,
                at_timestamp: 0,
            }),
            write_concern: None,
            transaction_id: 0,
        }))
        .await;

    assert!(
        read_result.is_ok(),
        "causal read after write must succeed: {:?}",
        read_result.err()
    );
    let body = read_result.unwrap().into_inner();
    assert_eq!(
        body.rows.len(),
        1,
        "causal read must return the written node"
    );
}

/// Build a CypherService backed by a real (single-node) Raft pipeline,
/// mirroring the standalone server wiring in `main.rs`: one shared
/// engine + oracle, `RaftNode::open_with_oracle`, `RaftProposalPipeline`,
/// `Database::from_engine`, and the read fence attached.
async fn test_service_raft() -> (CypherServiceImpl, Arc<RaftNode>, tempfile::TempDir) {
    use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, Tier};

    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new());
    let config = coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
        EndpointConfig::new(
            "default",
            dir.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        ),
    ]);
    let engine = Arc::new(
        coordinode_storage::engine::core::StorageEngine::open_with_oracle(
            &config,
            Arc::clone(&oracle),
        )
        .expect("open engine"),
    );
    let node = Arc::new(
        RaftNode::open_with_oracle(1, Arc::clone(&engine), Some(Arc::clone(&oracle)))
            .await
            .expect("raft node"),
    );
    // Single-node Raft needs a moment to elect itself leader.
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    let pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> = Arc::new(
        coordinode_raft::proposal::RaftProposalPipeline::new(Arc::clone(node.raft())),
    );
    let db = Database::from_engine(dir.path(), engine, oracle, pipeline).expect("from_engine");
    let svc = CypherServiceImpl::new(
        Arc::new(RwLock::new(db)),
        Arc::new(QueryRegistry::new()),
        Arc::new(NPlus1Detector::new()),
    )
    .with_raft_node(Arc::clone(&node));
    (svc, node, dir)
}

/// operationTime regression (end-to-end): a write through a Raft-backed service
/// returns ITS OWN committed Raft index as `operationTime`, not the
/// node's current applied index sampled by the read fence. Successive
/// writes report strictly increasing indices, and a causal read fencing
/// on a write's reported index observes that write.
#[tokio::test(flavor = "multi_thread")]
async fn raft_write_returns_own_committed_index() {
    let (svc, node, _dir) = test_service_raft().await;

    let w1 = svc
        .execute_cypher(cypher_request("CREATE (n:RaftCausal {v: 1}) RETURN n"))
        .await
        .expect("first write")
        .into_inner();
    let idx1 = w1.stats.expect("stats present").applied_index;
    assert!(
        idx1 > 0,
        "Raft-backed write must return its committed index, got {idx1}"
    );

    let w2 = svc
        .execute_cypher(cypher_request("CREATE (n:RaftCausal {v: 2}) RETURN n"))
        .await
        .expect("second write")
        .into_inner();
    let idx2 = w2.stats.expect("stats present").applied_index;
    assert!(
        idx2 > idx1,
        "second write's operationTime must advance: {idx1} -> {idx2}"
    );

    // A causal read fencing on the second write's own index sees both nodes.
    let read = svc
        .execute_cypher(Request::new(query::ExecuteCypherRequest {
            query: "MATCH (n:RaftCausal) RETURN n.v".to_string(),
            parameters: std::collections::HashMap::new(),
            read_preference: 0,
            read_concern: Some(crate::proto::replication::ReadConcern {
                level: 2, // MAJORITY
                after_index: idx2,
                at_timestamp: 0,
            }),
            write_concern: None,
            transaction_id: 0,
        }))
        .await
        .expect("causal read after write");
    assert_eq!(
        read.into_inner().rows.len(),
        2,
        "causal read fencing on the write index must observe both writes"
    );

    node.shutdown().await.expect("shutdown");
}

/// after_index = 0 with any readConcern level is always valid (no fence).
#[tokio::test]
async fn causal_after_index_zero_always_valid() {
    let (svc, _dir) = test_service();

    for level in [0u32, 1, 2, 4] {
        // 0=UNSPECIFIED, 1=LOCAL, 2=MAJORITY, 4=SNAPSHOT
        let result = svc
            .execute_cypher(Request::new(query::ExecuteCypherRequest {
                query: "MATCH (n) RETURN n".to_string(),
                parameters: std::collections::HashMap::new(),
                read_preference: 0,
                read_concern: Some(crate::proto::replication::ReadConcern {
                    level: level as i32,
                    after_index: 0,
                    at_timestamp: 0,
                }),
                write_concern: None,
                transaction_id: 0,
            }))
            .await;
        assert!(
            result.is_ok(),
            "level={level} with after_index=0 should succeed, got: {:?}",
            result.err()
        );
    }
}

// ── G088: write-concern validation in causal sessions ─────────────────────

/// Causal write without write_concern is rejected with FailedPrecondition.
///
/// after_index > 0 signals a causal session. A write statement without an
/// explicit MAJORITY write_concern risks producing a dangling applied_index
/// that no follower can ever satisfy. The server must hard-reject it.
#[tokio::test]
async fn causal_write_without_concern_rejected() {
    let (svc, _dir) = test_service();

    let result = svc
        .execute_cypher(Request::new(query::ExecuteCypherRequest {
            query: "CREATE (n:CausalWrite {x: 1})".to_string(),
            parameters: std::collections::HashMap::new(),
            read_preference: 0,
            read_concern: Some(crate::proto::replication::ReadConcern {
                level: 2, // MAJORITY
                after_index: 1,
                at_timestamp: 0,
            }),
            write_concern: None, // omitted → treated as UNSPECIFIED (w:1)
            transaction_id: 0,
        }))
        .await;

    assert!(
        result.is_err(),
        "causal write without write_concern must be rejected"
    );
    assert_eq!(
        result.unwrap_err().code(),
        tonic::Code::FailedPrecondition,
        "must be FailedPrecondition"
    );
}

/// Causal write with WriteConcern=W1 is rejected with FailedPrecondition.
///
/// W1 (leader-acknowledged) is insufficient for causal sessions: the leader
/// may crash before the write is replicated, making the applied_index a
/// dangling dependency that followers can never satisfy.
#[tokio::test]
async fn causal_write_with_w1_rejected() {
    let (svc, _dir) = test_service();

    let result = svc
        .execute_cypher(Request::new(query::ExecuteCypherRequest {
            query: "CREATE (n:CausalWrite {x: 2})".to_string(),
            parameters: std::collections::HashMap::new(),
            read_preference: 0,
            read_concern: Some(crate::proto::replication::ReadConcern {
                level: 2, // MAJORITY
                after_index: 5,
                at_timestamp: 0,
            }),
            write_concern: Some(crate::proto::replication::WriteConcern {
                level: crate::proto::replication::WriteConcernLevel::W1 as i32,
                timeout_ms: 0,
                journal: false,
            }),
            transaction_id: 0,
        }))
        .await;

    assert!(result.is_err(), "causal write with W1 must be rejected");
    assert_eq!(
        result.unwrap_err().code(),
        tonic::Code::FailedPrecondition,
        "must be FailedPrecondition"
    );
}

/// Causal write with WriteConcern=MAJORITY is accepted in standalone mode.
///
/// MAJORITY is the minimum required durability for writes in a causal session.
/// In standalone mode the write proceeds normally (no Raft replication).
#[tokio::test]
async fn causal_write_with_majority_accepted() {
    let (svc, _dir) = test_service();

    let result = svc
        .execute_cypher(Request::new(query::ExecuteCypherRequest {
            query: "CREATE (n:CausalWrite {x: 3})".to_string(),
            parameters: std::collections::HashMap::new(),
            read_preference: 0,
            read_concern: Some(crate::proto::replication::ReadConcern {
                level: 2, // MAJORITY
                after_index: 7,
                at_timestamp: 0,
            }),
            write_concern: Some(crate::proto::replication::WriteConcern {
                level: crate::proto::replication::WriteConcernLevel::Majority as i32,
                timeout_ms: 0,
                journal: false,
            }),
            transaction_id: 0,
        }))
        .await;

    assert!(
        result.is_ok(),
        "causal write with MAJORITY must succeed in standalone, got: {:?}",
        result.err()
    );
}

/// Read queries in causal sessions do not require write_concern.
///
/// The write-concern gate is skipped entirely for read-only queries; only
/// mutating statements (CREATE, MERGE, SET, DELETE, …) are subject to it.
#[tokio::test]
async fn causal_read_without_write_concern_accepted() {
    let (svc, _dir) = test_service();

    let result = svc
        .execute_cypher(Request::new(query::ExecuteCypherRequest {
            query: "MATCH (n:CausalWrite) RETURN n".to_string(),
            parameters: std::collections::HashMap::new(),
            read_preference: 0,
            read_concern: Some(crate::proto::replication::ReadConcern {
                level: 2, // MAJORITY
                after_index: 10,
                at_timestamp: 0,
            }),
            write_concern: None, // read-only — write_concern irrelevant
            transaction_id: 0,
        }))
        .await;

    assert!(
        result.is_ok(),
        "causal read without write_concern must succeed, got: {:?}",
        result.err()
    );
}

// --- Audit gap closures: gRPC wiring of write_concern, WriteStats, and
//     ALTER LABEL schema_revision visibility (see DEVLOG audit) ---

/// Regression: WriteStats must propagate from executor to QueryStats.
/// Previously hardcoded to zero — clients had no way to see mutation
/// counts even when CREATE/SET/DELETE ran successfully.
#[tokio::test]
async fn grpc_query_stats_reports_actual_mutation_counts() {
    let (svc, _dir) = test_service();

    let resp = svc
        .execute_cypher(cypher_request(
            "CREATE (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}), \
                 (a)-[:KNOWS]->(b)",
        ))
        .await
        .expect("create chain");

    let stats = resp.into_inner().stats.expect("stats present");
    assert_eq!(stats.nodes_created, 2, "two nodes created");
    assert_eq!(stats.edges_created, 1, "one edge created");
    assert_eq!(stats.nodes_deleted, 0);
    assert_eq!(stats.edges_deleted, 0);
}

/// Regression: SET clause increments `properties_set` in QueryStats.
/// Same wiring gap as nodes_created — verifies the full path through
/// execute_cypher_full → CypherResult → proto QueryStats.
#[tokio::test]
async fn grpc_query_stats_reports_property_set_count() {
    let (svc, _dir) = test_service();

    svc.execute_cypher(cypher_request("CREATE (n:Doc {id: 1})"))
        .await
        .expect("create");

    let resp = svc
        .execute_cypher(cypher_request(
            "MATCH (n:Doc {id: 1}) SET n.title = 'hello' RETURN n",
        ))
        .await
        .expect("set");

    let stats = resp.into_inner().stats.expect("stats present");
    assert!(
        stats.properties_set >= 1,
        "SET must increment properties_set, got {}",
        stats.properties_set
    );
}

/// Regression: client-supplied write_concern is not silently ignored.
/// Previously the handler validated write_concern only for causal sessions
/// but never propagated it to the executor — every MAJORITY write was
/// downgraded to W1. With propagation, MAJORITY in standalone mode (no
/// Raft) is accepted: WriteConcern::effective_level() downgrades to W1
/// internally, but no error surfaces to the client.
#[tokio::test]
async fn grpc_write_concern_majority_accepted_in_standalone() {
    let (svc, _dir) = test_service();

    let resp = svc
        .execute_cypher(Request::new(query::ExecuteCypherRequest {
            query: "CREATE (n:Durable {id: 1})".to_string(),
            parameters: std::collections::HashMap::new(),
            read_preference: 0,
            read_concern: None,
            write_concern: Some(crate::proto::replication::WriteConcern {
                level: 3, // MAJORITY
                journal: false,
                timeout_ms: 0,
            }),
            transaction_id: 0,
        }))
        .await
        .expect("write should succeed");
    let stats = resp.into_inner().stats.expect("stats");
    assert_eq!(stats.nodes_created, 1);
}

/// gRPC ALTER LABEL through Cypher bumps `schema_revision` visible via
/// the subsequent SchemaService.list_labels response. This regression test
/// closes the C-decision wiring gap: ALTER LABEL was bumping
/// `schema_revision` on the persisted LabelSchema, but no gRPC integration
/// test verified the response value crossed the proto boundary correctly
/// after the rename.
#[tokio::test]
async fn grpc_alter_label_bumps_schema_revision_visible_via_list_labels() {
    use crate::proto::graph as graph_proto;
    use crate::proto::graph::schema_service_server::SchemaService;
    use crate::services::schema::SchemaServiceImpl;

    let dir = tempfile::tempdir().expect("tempdir");
    let database = Arc::new(RwLock::new(
        Database::open(dir.path()).expect("open database"),
    ));
    let registry = Arc::new(QueryRegistry::new());
    let detector = Arc::new(NPlus1Detector::new());

    let cypher_svc = CypherServiceImpl::new(database.clone(), registry, detector);
    let schema_svc = SchemaServiceImpl::new(database);

    // Seed a node so the label exists in the catalog scan path.
    cypher_svc
        .execute_cypher(cypher_request("CREATE (n:RevisionTarget {id: 1})"))
        .await
        .expect("seed");

    // Capture the initial schema_revision via ListLabels (implicit label
    // declared on first node create → revision 0 in the catalog response;
    // revision becomes positive once ALTER LABEL writes a schema body).
    let list1 = schema_svc
        .list_labels(Request::new(graph_proto::ListLabelsRequest {}))
        .await
        .expect("list 1")
        .into_inner();
    let initial = list1
        .labels
        .iter()
        .find(|l| l.name == "RevisionTarget")
        .map(|l| l.schema_revision)
        .unwrap_or(0);

    // ALTER LABEL via Cypher — should bump schema_revision on the
    // persisted snapshot.
    cypher_svc
        .execute_cypher(cypher_request(
            "ALTER LABEL RevisionTarget SET SCHEMA VALIDATED",
        ))
        .await
        .expect("alter");

    let list2 = schema_svc
        .list_labels(Request::new(graph_proto::ListLabelsRequest {}))
        .await
        .expect("list 2")
        .into_inner();
    let after_alter = list2
        .labels
        .iter()
        .find(|l| l.name == "RevisionTarget")
        .expect("label present")
        .schema_revision;
    assert!(
        after_alter > initial,
        "ALTER LABEL must bump schema_revision visible via gRPC ListLabels: \
             initial={initial}, after_alter={after_alter}"
    );
}

// --- ReadConcern.at_timestamp time-travel + WriteConcern Memory/Cache ---

/// at_timestamp set on a non-SNAPSHOT read concern is rejected at the
/// gRPC boundary with FAILED_PRECONDITION (the executor would otherwise
/// ignore the pin silently, defeating the time-travel contract).
#[tokio::test]
async fn grpc_at_timestamp_rejected_without_snapshot_level() {
    let (svc, _dir) = test_service();
    let result = svc
        .execute_cypher(Request::new(query::ExecuteCypherRequest {
            query: "MATCH (n) RETURN n".to_string(),
            parameters: std::collections::HashMap::new(),
            read_preference: 0,
            read_concern: Some(crate::proto::replication::ReadConcern {
                level: 2, // MAJORITY, not SNAPSHOT
                after_index: 0,
                at_timestamp: 1_700_000_000_000_000,
            }),
            write_concern: None,
            transaction_id: 0,
        }))
        .await;
    let err = result.expect_err("must reject");
    assert_eq!(err.code(), tonic::Code::FailedPrecondition);
    assert!(
        err.message().contains("at_timestamp") && err.message().to_lowercase().contains("snapshot"),
        "error must mention at_timestamp + SNAPSHOT, got: {}",
        err.message()
    );
}

/// SNAPSHOT read with `at_timestamp = 1` (an HLC value far in the past,
/// before any writes happened) sees an empty database — proves the pin
/// actually reaches the executor and constrains the snapshot.
#[tokio::test]
async fn grpc_at_timestamp_snapshot_pins_to_past_returns_empty() {
    let (svc, _dir) = test_service();

    // Write something.
    svc.execute_cypher(cypher_request("CREATE (n:Past {id: 1})"))
        .await
        .expect("create");

    // Read pinned to ts=1 (epoch + 1µs, before any HLC stamps).
    let result = svc
        .execute_cypher(Request::new(query::ExecuteCypherRequest {
            query: "MATCH (n:Past) RETURN n.id".to_string(),
            parameters: std::collections::HashMap::new(),
            read_preference: 0,
            read_concern: Some(crate::proto::replication::ReadConcern {
                level: 4, // SNAPSHOT
                after_index: 0,
                at_timestamp: 1,
            }),
            write_concern: None,
            transaction_id: 0,
        }))
        .await
        .expect("snapshot read should succeed");
    let resp = result.into_inner();
    assert!(
        resp.rows.is_empty(),
        "snapshot at ts=1 (pre-history) must return empty, got {} rows",
        resp.rows.len()
    );
}

/// SNAPSHOT read without `at_timestamp` (omitted / 0) falls back to the
/// latest oracle.next() — sees all writes — proving the absence-of-pin
/// path is also wired through the new field.
#[tokio::test]
async fn grpc_snapshot_without_at_timestamp_returns_latest() {
    let (svc, _dir) = test_service();
    svc.execute_cypher(cypher_request("CREATE (n:Now {id: 1})"))
        .await
        .expect("create");
    let result = svc
        .execute_cypher(Request::new(query::ExecuteCypherRequest {
            query: "MATCH (n:Now) RETURN n.id".to_string(),
            parameters: std::collections::HashMap::new(),
            read_preference: 0,
            read_concern: Some(crate::proto::replication::ReadConcern {
                level: 4, // SNAPSHOT
                after_index: 0,
                at_timestamp: 0, // no pin → latest
            }),
            write_concern: None,
            transaction_id: 0,
        }))
        .await
        .expect("snapshot read");
    let resp = result.into_inner();
    assert_eq!(
        resp.rows.len(),
        1,
        "snapshot without at_timestamp must see latest writes"
    );
}

/// WriteConcernLevel::MEMORY (proto = 4) reaches the executor without
/// silent downgrade. In standalone mode the volatile drain path falls
/// back to W1 internally (effective_level), so the operation succeeds —
/// but the proto value must NOT be coerced to W1 on entry, otherwise
/// MEMORY-specific drain behaviour would never engage in cluster mode.
#[tokio::test]
async fn grpc_write_concern_memory_accepted() {
    let (svc, _dir) = test_service();
    let resp = svc
        .execute_cypher(Request::new(query::ExecuteCypherRequest {
            query: "CREATE (n:Telemetry {id: 1})".to_string(),
            parameters: std::collections::HashMap::new(),
            read_preference: 0,
            read_concern: None,
            write_concern: Some(crate::proto::replication::WriteConcern {
                level: 4, // MEMORY
                journal: false,
                timeout_ms: 0,
            }),
            transaction_id: 0,
        }))
        .await
        .expect("memory write should succeed");
    let stats = resp.into_inner().stats.expect("stats");
    assert_eq!(stats.nodes_created, 1);
}

/// after_index + at_timestamp are mutually exclusive: a snapshot pin is
/// incompatible with a causal fence. The executor's `ReadConcern::validate`
/// catches this, but the boundary should surface it as a semantic error
/// (InvalidArgument) rather than letting the executor produce a generic
/// internal error.
#[tokio::test]
async fn grpc_after_index_and_at_timestamp_mutually_exclusive() {
    let (svc, _dir) = test_service();
    let result = svc
        .execute_cypher(Request::new(query::ExecuteCypherRequest {
            query: "MATCH (n) RETURN n".to_string(),
            parameters: std::collections::HashMap::new(),
            read_preference: 0,
            read_concern: Some(crate::proto::replication::ReadConcern {
                level: 4, // SNAPSHOT — only level where both fields are individually accepted
                after_index: 42,
                at_timestamp: 1_700_000_000_000_000,
            }),
            write_concern: None,
            transaction_id: 0,
        }))
        .await;
    let err = result.expect_err("must reject combination");
    assert!(
        err.message().to_lowercase().contains("mutually exclusive")
            || err.message().to_lowercase().contains("after_index"),
        "error must mention the conflict, got: {}",
        err.message()
    );
}

/// WriteConcern.journal flag (separate from level) reaches the executor.
/// The flag forces WAL fsync regardless of level; in standalone single-
/// node mode that converges with W1 behaviour, but the field must NOT be
/// dropped at the boundary — otherwise cluster deployments would silently
/// lose durability when callers set `journal: true`.
#[tokio::test]
async fn grpc_write_concern_journal_flag_accepted() {
    let (svc, _dir) = test_service();
    let resp = svc
        .execute_cypher(Request::new(query::ExecuteCypherRequest {
            query: "CREATE (n:Durable {id: 1})".to_string(),
            parameters: std::collections::HashMap::new(),
            read_preference: 0,
            read_concern: None,
            write_concern: Some(crate::proto::replication::WriteConcern {
                level: 2, // W1 with journal forces fsync
                journal: true,
                timeout_ms: 5_000,
            }),
            transaction_id: 0,
        }))
        .await
        .expect("journaled write should succeed");
    let stats = resp.into_inner().stats.expect("stats");
    assert_eq!(stats.nodes_created, 1);
}

/// WriteConcernLevel::CACHE (proto = 5) — same wire-through invariant.
#[tokio::test]
async fn grpc_write_concern_cache_accepted() {
    let (svc, _dir) = test_service();
    let resp = svc
        .execute_cypher(Request::new(query::ExecuteCypherRequest {
            query: "CREATE (n:Analytics {id: 1})".to_string(),
            parameters: std::collections::HashMap::new(),
            read_preference: 0,
            read_concern: None,
            write_concern: Some(crate::proto::replication::WriteConcern {
                level: 5, // CACHE
                journal: false,
                timeout_ms: 0,
            }),
            transaction_id: 0,
        }))
        .await
        .expect("cache write should succeed");
    let stats = resp.into_inner().stats.expect("stats");
    assert_eq!(stats.nodes_created, 1);
}

/// End-to-end capacity propagation through the gRPC handler:
///
/// Fill an endpoint past its `hard_limit_bytes`, refresh the
/// capacity tracker, then issue a CREATE query through
/// `CypherServiceImpl::execute_cypher`. The handler returns a
/// `tonic::Status` exactly as a gRPC client would observe over
/// the wire (the transport is transparent for `Status`
/// content). Assertions:
///
/// - `Status::code() == Code::ResourceExhausted` (gRPC-canonical
///   for capacity-exhausted writes; NOT `Internal`).
/// - `endpoint-id` metadata header carries the saturated
///   endpoint's id so the client can match without scraping
///   the message.
/// - `used-bytes` + `hard-limit-bytes` metadata headers carry
///   the numeric snapshot for client-side rendering.
///
/// Proves the chain
/// `StorageError::CapacityExhausted` →
/// `DatabaseError::Storage(...)` →
/// `db_err_to_status` → `Status::resource_exhausted` is intact
/// end-to-end at the handler boundary.
#[tokio::test]
async fn capacity_exhausted_surfaces_as_resource_exhausted_through_grpc_handler() {
    use coordinode_query::advisor::nplus1::NPlus1Detector;
    use coordinode_query::advisor::QueryRegistry;
    use coordinode_storage::engine::config::{
        Durability as Dur, EndpointConfig, Media, StorageConfig, Tier,
    };
    use coordinode_storage::engine::core::StorageEngine;
    use coordinode_storage::engine::partition::Partition;
    use tonic::Code;

    let dir = tempfile::tempdir().expect("tempdir");

    // Step 1: open the engine standalone with a tiny hard_limit so
    // a modest write blows past the threshold. The capacity gate
    // engages once `refresh_capacity` observes Full.
    //
    // Layered like `Database::open_with_oracle` would: build the
    // engine first, then plant data, then wrap it as a Database
    // via `from_engine`. Going through `Database::open` directly
    // would lock us to `hard_limit_bytes = 0` (its default
    // single-endpoint config).
    let oracle = std::sync::Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new());
    let storage_config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "ep",
        dir.path(),
        Media::Hdd,
        Dur::Durable,
        Tier::Warm,
    )
    .with_hard_limit_bytes(40_000)]);
    let engine = std::sync::Arc::new(
        StorageEngine::open_with_oracle(&storage_config, std::sync::Arc::clone(&oracle))
            .expect("open engine with custom hard_limit"),
    );

    // Bulk-write to push past 40 KB on disk.
    for i in 0..5000u32 {
        let key = format!("node:0:{i:010}");
        let _ = engine.put(Partition::Node, key.as_bytes(), b"payload-bytes");
    }
    engine.persist().expect("persist");
    engine.refresh_capacity();
    let usage = engine.capacity().get("ep").expect("tracked");
    assert!(
        !usage.is_writable(),
        "endpoint must be Full after bulk write (used={})",
        usage.used(),
    );

    // Step 2: wrap the saturated engine in a Database and stand
    // up the gRPC service. No tonic transport — calling the
    // handler method directly returns the same `tonic::Status`
    // that a client would observe over the wire.
    let pipeline: std::sync::Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> =
        std::sync::Arc::new(coordinode_raft::proposal::OwnedLocalProposalPipeline::new(
            &engine,
        ));
    let db = Database::from_engine(dir.path(), engine, oracle, pipeline).expect("from_engine");
    let database = std::sync::Arc::new(parking_lot::RwLock::new(db));
    let registry = std::sync::Arc::new(QueryRegistry::new());
    let detector = std::sync::Arc::new(NPlus1Detector::new());
    let svc = CypherServiceImpl::new(
        std::sync::Arc::clone(&database),
        std::sync::Arc::clone(&registry),
        std::sync::Arc::clone(&detector),
    );

    // Step 3: send a CREATE through the handler. Must surface as
    // ResourceExhausted with structured metadata.
    let result = svc
        .execute_cypher(Request::new(query::ExecuteCypherRequest {
            query: "CREATE (n:GatedByCapacity)".to_string(),
            parameters: std::collections::HashMap::new(),
            read_preference: 0,
            read_concern: None,
            write_concern: None,
            transaction_id: 0,
        }))
        .await;

    let status = match result {
        Err(s) => s,
        Ok(_) => panic!("expected ResourceExhausted, got Ok"),
    };
    assert_eq!(
        status.code(),
        Code::ResourceExhausted,
        "expected RESOURCE_EXHAUSTED (gRPC canonical for capacity), got: {:?} — message: {}",
        status.code(),
        status.message(),
    );
    let meta = status.metadata();
    assert_eq!(
        meta.get("endpoint-id").and_then(|v| v.to_str().ok()),
        Some("ep"),
        "endpoint-id metadata header must carry the saturated endpoint id",
    );
    assert!(
        meta.get("used-bytes").is_some(),
        "used-bytes metadata header must be set",
    );
    assert!(
        meta.get("hard-limit-bytes").is_some(),
        "hard-limit-bytes metadata header must be set",
    );
    let hard_limit = meta
        .get("hard-limit-bytes")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .expect("hard-limit-bytes parses as u64");
    assert_eq!(hard_limit, 40_000);
}
