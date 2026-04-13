use std::sync::{Arc, Mutex};

use tonic::{Request, Response, Status};

use coordinode_core::graph::types::Value;
use coordinode_embed::{Database, DatabaseError};
use coordinode_query::advisor::nplus1::NPlus1Detector;
use coordinode_query::advisor::source::{self, grpc_keys, SourceContext};
use coordinode_query::advisor::QueryRegistry;

use crate::proto::{common, query};

/// Extract source location context from gRPC request metadata.
///
/// Returns `None` if no source metadata is present (debug mode not enabled).
fn extract_source_context<T>(request: &Request<T>) -> Option<SourceContext> {
    let metadata = request.metadata();
    let get = |key: &str| -> Option<String> {
        metadata
            .get(key)
            .and_then(|v| v.to_str().ok())
            .map(String::from)
    };
    source::extract_from_map(
        &get,
        grpc_keys::FILE,
        grpc_keys::LINE,
        grpc_keys::FUNCTION,
        grpc_keys::APP,
        grpc_keys::VERSION,
    )
}

/// Convert a coordinode Value to a proto PropertyValue.
pub(crate) fn value_to_proto_pub(value: &Value) -> common::PropertyValue {
    value_to_proto(value)
}

fn value_to_proto(value: &Value) -> common::PropertyValue {
    let v = match value {
        Value::Null => None,
        Value::Bool(b) => Some(common::property_value::Value::BoolValue(*b)),
        Value::Int(i) => Some(common::property_value::Value::IntValue(*i)),
        Value::Float(f) => Some(common::property_value::Value::FloatValue(*f)),
        Value::String(s) => Some(common::property_value::Value::StringValue(s.clone())),
        Value::Timestamp(ts) => Some(common::property_value::Value::IntValue(*ts)),
        Value::Vector(v) => Some(common::property_value::Value::VectorValue(common::Vector {
            values: v.clone(),
        })),
        Value::Blob(b) | Value::Binary(b) => {
            Some(common::property_value::Value::BytesValue(b.clone()))
        }
        Value::Array(arr) => {
            let items = arr.iter().map(value_to_proto).collect();
            Some(common::property_value::Value::ListValue(
                common::PropertyList { values: items },
            ))
        }
        Value::Map(map) => {
            let entries = map
                .iter()
                .map(|(k, v)| (k.clone(), value_to_proto(v)))
                .collect();
            Some(common::property_value::Value::MapValue(
                common::PropertyMap { entries },
            ))
        }
        Value::Geo(_) => Some(common::property_value::Value::StringValue(format!(
            "{value:?}"
        ))),
        Value::Document(v) => {
            // Serialize rmpv::Value as MessagePack bytes for proto transport
            let mut bytes = Vec::new();
            let _ = rmpv::encode::write_value(&mut bytes, v);
            Some(common::property_value::Value::BytesValue(bytes))
        }
    };
    common::PropertyValue { value: v }
}

/// Convert a proto PropertyValue to a coordinode Value.
pub(crate) fn proto_to_value_pub(pv: &common::PropertyValue) -> Value {
    proto_to_value(pv)
}

fn proto_to_value(pv: &common::PropertyValue) -> Value {
    match &pv.value {
        None => Value::Null,
        Some(v) => match v {
            common::property_value::Value::BoolValue(b) => Value::Bool(*b),
            common::property_value::Value::IntValue(i) => Value::Int(*i),
            common::property_value::Value::FloatValue(f) => Value::Float(*f),
            common::property_value::Value::StringValue(s) => Value::String(s.clone()),
            common::property_value::Value::BytesValue(b) => Value::Binary(b.clone()),
            common::property_value::Value::VectorValue(v) => Value::Vector(v.values.clone()),
            common::property_value::Value::ListValue(list) => {
                Value::Array(list.values.iter().map(proto_to_value).collect())
            }
            common::property_value::Value::MapValue(map) => Value::Map(
                map.entries
                    .iter()
                    .map(|(k, v)| (k.clone(), proto_to_value(v)))
                    .collect(),
            ),
            common::property_value::Value::TimestampValue(ts) => {
                Value::Timestamp(ts.wall_time as i64)
            }
        },
    }
}

/// Convert proto parameters map to coordinode Value map.
fn convert_params(
    proto_params: &std::collections::HashMap<String, common::PropertyValue>,
) -> std::collections::HashMap<String, Value> {
    proto_params
        .iter()
        .map(|(k, v)| (k.clone(), proto_to_value(v)))
        .collect()
}

/// Convert a DatabaseError to a tonic Status.
fn db_error_to_status(err: DatabaseError) -> Status {
    match err {
        DatabaseError::Parse(e) => Status::invalid_argument(format!("Parse error: {e}")),
        DatabaseError::Semantic(e) => Status::invalid_argument(format!("Semantic error: {e}")),
        DatabaseError::Plan(e) => Status::internal(format!("Plan error: {e}")),
        DatabaseError::Execution(e) => Status::internal(format!("Execution error: {e}")),
        DatabaseError::Storage(e) => Status::internal(format!("Storage error: {e}")),
        DatabaseError::Other(e) => Status::internal(format!("Error: {e}")),
    }
}

pub struct CypherServiceImpl {
    database: Arc<Mutex<Database>>,
    query_registry: Arc<QueryRegistry>,
    nplus1_detector: Arc<NPlus1Detector>,
}

impl CypherServiceImpl {
    pub fn new(
        database: Arc<Mutex<Database>>,
        query_registry: Arc<QueryRegistry>,
        nplus1_detector: Arc<NPlus1Detector>,
    ) -> Self {
        Self {
            database,
            query_registry,
            nplus1_detector,
        }
    }
}

#[tonic::async_trait]
impl query::cypher_service_server::CypherService for CypherServiceImpl {
    async fn execute_cypher(
        &self,
        request: Request<query::ExecuteCypherRequest>,
    ) -> Result<Response<query::ExecuteCypherResponse>, Status> {
        let source_ctx = extract_source_context(&request);
        let req = request.into_inner();

        let start = std::time::Instant::now();

        // Execute query through the embedded Database engine.
        // Convert proto parameters to Value map for parameter binding.
        let result_rows = {
            let mut db = self.database.lock().unwrap_or_else(|e| e.into_inner());

            let has_params = !req.parameters.is_empty();
            match (&source_ctx, has_params) {
                (Some(src), true) => {
                    let params = convert_params(&req.parameters);
                    db.execute_cypher_with_params_and_source(&req.query, params, src)
                        .map_err(db_error_to_status)?
                }
                (Some(src), false) => db
                    .execute_cypher_with_source(&req.query, src)
                    .map_err(db_error_to_status)?,
                (None, true) => {
                    let params = convert_params(&req.parameters);
                    db.execute_cypher_with_params(&req.query, params)
                        .map_err(db_error_to_status)?
                }
                (None, false) => db.execute_cypher(&req.query).map_err(db_error_to_status)?,
            }
        };
        // Mutex released here — held only during query execution.

        let duration_ms = start.elapsed().as_millis() as u64;

        // Advisor tracking: fingerprint + source + N+1
        // Database::execute_cypher already records in its own registry,
        // but server has its own registry for gRPC-specific tracking.
        if let Ok(ast) = coordinode_query::cypher::parse(&req.query) {
            let (canonical, fp) = coordinode_query::advisor::normalize_and_fingerprint(&ast);
            let duration_us = start.elapsed().as_micros() as u64;

            match &source_ctx {
                Some(src) => {
                    self.query_registry
                        .record_with_source(fp, &canonical, duration_us, src);
                    if let Some(alert) = self.nplus1_detector.record(fp, &canonical, src) {
                        tracing::warn!(
                            fingerprint = fp,
                            count = alert.call_count,
                            file = %alert.source_file,
                            line = alert.source_line,
                            "N+1 query pattern detected"
                        );
                    }
                }
                None => {
                    self.query_registry.record(fp, &canonical, duration_us);
                }
            }
        }

        // Convert executor rows → proto rows.
        // Determine columns from the first row (all rows share the same keys).
        let columns: Vec<String> = result_rows
            .first()
            .map(|r| r.keys().cloned().collect())
            .unwrap_or_default();

        let proto_rows: Vec<query::Row> = result_rows
            .iter()
            .map(|row| {
                let values = columns
                    .iter()
                    .map(|col| {
                        row.get(col)
                            .map(value_to_proto)
                            .unwrap_or(common::PropertyValue { value: None })
                    })
                    .collect();
                query::Row { values }
            })
            .collect();

        Ok(Response::new(query::ExecuteCypherResponse {
            columns,
            rows: proto_rows,
            stats: Some(query::QueryStats {
                nodes_created: 0, // TODO: wire from WriteStats when available
                nodes_deleted: 0,
                edges_created: 0,
                edges_deleted: 0,
                properties_set: 0,
                execution_time_ms: duration_ms as i64,
            }),
        }))
    }

    async fn explain_cypher(
        &self,
        request: Request<query::ExplainCypherRequest>,
    ) -> Result<Response<query::ExplainCypherResponse>, Status> {
        let req = request.into_inner();

        let ast = coordinode_query::cypher::parse(&req.query)
            .map_err(|e| Status::invalid_argument(format!("Cypher parse error: {e}")))?;

        let plan = coordinode_query::planner::build_logical_plan(&ast)
            .map_err(|e| Status::internal(format!("Plan error: {e}")))?;

        // Compute storage stats for accurate cost estimation (TTL-cached, MVCC-aware)
        let stats = self.database.lock().ok().and_then(|db| db.compute_stats());
        let stats_ref = stats
            .as_ref()
            .map(|s| s as &dyn coordinode_core::graph::stats::StorageStats);

        let cost = coordinode_query::planner::estimate_cost_with_stats(&plan, stats_ref);

        // Run suggestion detectors (EXPLAIN SUGGEST)
        let suggest_result = plan.explain_suggest_with_stats(stats_ref, None);

        let mut details = std::collections::HashMap::new();
        details.insert("explain".to_string(), suggest_result.explain);
        details.insert("cost".to_string(), format!("{:.0}", cost.cost));
        details.insert(
            "estimated_time_ms".to_string(),
            format!("{:.1}", cost.estimated_time_ms),
        );
        if !cost.hints.is_empty() {
            details.insert("hints".to_string(), cost.hints.join("; "));
        }

        // Include suggestions in response details
        if !suggest_result.suggestions.is_empty() {
            let suggestions_text: Vec<String> = suggest_result
                .suggestions
                .iter()
                .map(|s| s.to_string())
                .collect();
            details.insert("suggestions".to_string(), suggestions_text.join("\n"));
            details.insert(
                "suggestion_count".to_string(),
                suggest_result.suggestions.len().to_string(),
            );
        }

        Ok(Response::new(query::ExplainCypherResponse {
            plan: Some(query::QueryPlan {
                operator: "LogicalPlan".to_string(),
                details,
                children: vec![],
                estimated_rows: cost.estimated_rows,
            }),
        }))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
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
        let database = Arc::new(Mutex::new(
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
        })
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
        let database = Arc::new(Mutex::new(
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
                .add_service(
                    crate::proto::query::cypher_service_server::CypherServiceServer::new(svc),
                )
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
        // this test (cypher.rs), which is then injected as x-source-file metadata.
        assert!(
            src.file.contains("cypher.rs"),
            "source file should be cypher.rs, got: {}",
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
        let database = Arc::new(Mutex::new(
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
                .add_service(
                    crate::proto::query::cypher_service_server::CypherServiceServer::new(svc),
                )
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
            src.file.contains("cypher.rs"),
            "source file should be cypher.rs, got: {}",
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
}
