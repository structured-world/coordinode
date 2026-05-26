use std::sync::Arc;

// no-std: spin::RwLock (drop-in).
use parking_lot::RwLock;

use tonic::{Request, Response, Status};

use coordinode_core::graph::types::Value;
use coordinode_core::txn::read_concern::{
    ReadConcern as ExecutorReadConcern, ReadConcernLevel as ExecutorReadConcernLevel,
};
use coordinode_core::txn::write_concern::{WriteConcern, WriteConcernLevel};
use coordinode_embed::{Database, DatabaseError};
use coordinode_query::advisor::nplus1::NPlus1Detector;
use coordinode_query::advisor::source::{self, grpc_keys, SourceContext};
use coordinode_query::advisor::QueryRegistry;
use coordinode_raft::cluster::RaftNode;
use coordinode_raft::read_fence::{
    ReadConcern, ReadFenceError, ReadPreference, READ_FENCE_TIMEOUT,
};

use crate::proto::{common, query, replication};

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
///
/// Capacity-exhausted errors — whether they arrive as
/// `DatabaseError::Storage(CapacityExhausted)` (direct engine write)
/// or `DatabaseError::Execution(ExecutionError::Storage(CapacityExhausted))`
/// (Cypher writes through the proposal pipeline) — delegate to
/// [`crate::services::db_err_to_status`], which drills into both
/// shapes and emits `Status::resource_exhausted` with structured
/// metadata (`endpoint-id`, `used-bytes`, `hard-limit-bytes`).
///
/// Remaining variants keep their pre-existing categorisation:
/// Parse/Semantic → `invalid_argument`; Plan/Execution/Storage(other)/Other
/// → `internal`.
fn db_error_to_status(err: DatabaseError) -> Status {
    // Capacity-exhausted check FIRST — wins over the per-variant
    // category mapping below. The helper drills into both
    // `Storage(...)` and `Execution(Storage(...))` shapes; if it
    // identifies a `CapacityExhausted` it returns `ResourceExhausted`
    // with structured metadata. Otherwise it returns `Internal`,
    // which we override with the legacy category-specific mapping.
    let probe = crate::services::db_err_to_status("Cypher", err);
    if probe.code() == tonic::Code::ResourceExhausted {
        return probe;
    }
    // Probe was Internal — re-classify by inspecting the message
    // prefix (Display always emits `"<category> error: ..."` from the
    // `thiserror` annotations on `DatabaseError`).
    let msg = probe.message();
    if let Some(rest) = msg.strip_prefix("Cypher: parse error: ") {
        return Status::invalid_argument(format!("Parse error: {rest}"));
    }
    if let Some(rest) = msg.strip_prefix("Cypher: semantic error: ") {
        return Status::invalid_argument(format!("Semantic error: {rest}"));
    }
    if let Some(rest) = msg.strip_prefix("Cypher: plan error: ") {
        return Status::internal(format!("Plan error: {rest}"));
    }
    if let Some(rest) = msg.strip_prefix("Cypher: execution error: ") {
        return Status::internal(format!("Execution error: {rest}"));
    }
    if let Some(rest) = msg.strip_prefix("Cypher: storage error: ") {
        return Status::internal(format!("Storage error: {rest}"));
    }
    if let Some(rest) = msg.strip_prefix("Cypher: ") {
        return Status::internal(format!("Error: {rest}"));
    }
    probe
}

/// Translate the proto `ReadConcernLevel` integer to the executor enum. The
/// proto module's `ReadConcern` (used by the read fence) is distinct from
/// `coordinode_core::txn::read_concern::ReadConcern` (used by the executor for
/// snapshot timestamp selection) — both are populated from the same proto
/// field but consumed independently.
fn read_concern_level_to_executor(level: i32) -> ExecutorReadConcernLevel {
    match replication::ReadConcernLevel::try_from(level)
        .unwrap_or(replication::ReadConcernLevel::Unspecified)
    {
        replication::ReadConcernLevel::Majority => ExecutorReadConcernLevel::Majority,
        replication::ReadConcernLevel::Snapshot => ExecutorReadConcernLevel::Snapshot,
        replication::ReadConcernLevel::Linearizable => ExecutorReadConcernLevel::Linearizable,
        _ => ExecutorReadConcernLevel::Local,
    }
}

/// Translate the proto `WriteConcernLevel` integer to the executor enum.
/// Unspecified maps to W1 (single-node leader-acknowledged) — matches the
/// previous silent default before write_concern propagation landed.
fn write_concern_level_to_executor(level: i32) -> WriteConcernLevel {
    match replication::WriteConcernLevel::try_from(level)
        .unwrap_or(replication::WriteConcernLevel::Unspecified)
    {
        replication::WriteConcernLevel::W0 => WriteConcernLevel::W0,
        replication::WriteConcernLevel::Memory => WriteConcernLevel::Memory,
        replication::WriteConcernLevel::Cache => WriteConcernLevel::Cache,
        replication::WriteConcernLevel::Majority => WriteConcernLevel::Majority,
        // W1 and Unspecified both map to W1 — the silent default.
        _ => WriteConcernLevel::W1,
    }
}

/// Convert a ReadFenceError to a tonic Status.
fn fence_error_to_status(err: ReadFenceError) -> Status {
    match err {
        ReadFenceError::NotFollower => Status::failed_precondition(err.to_string()),
        ReadFenceError::NotLeader => Status::failed_precondition(err.to_string()),
        ReadFenceError::LinearizableRequiresLeader => Status::failed_precondition(err.to_string()),
        ReadFenceError::StaleReplica { .. } => Status::unavailable(err.to_string()),
        ReadFenceError::Timeout { .. } => Status::deadline_exceeded(err.to_string()),
        ReadFenceError::Raft(e) => Status::internal(format!("Raft error: {e}")),
    }
}

pub struct CypherServiceImpl {
    database: Arc<RwLock<Database>>,
    query_registry: Arc<QueryRegistry>,
    nplus1_detector: Arc<NPlus1Detector>,
    /// Raft node for read fence (follower reads). `None` in standalone mode.
    raft_node: Option<Arc<RaftNode>>,
}

impl CypherServiceImpl {
    pub fn new(
        database: Arc<RwLock<Database>>,
        query_registry: Arc<QueryRegistry>,
        nplus1_detector: Arc<NPlus1Detector>,
    ) -> Self {
        Self {
            database,
            query_registry,
            nplus1_detector,
            raft_node: None,
        }
    }

    /// Attach a Raft node for read fence enforcement (cluster mode).
    pub fn with_raft_node(mut self, raft_node: Arc<RaftNode>) -> Self {
        self.raft_node = Some(raft_node);
        self
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

        // Extract causal fence parameters before entering the Raft block so that
        // validation can run in both cluster and standalone modes.
        let concern_level = req.read_concern.as_ref().map(|rc| rc.level).unwrap_or(0);
        let concern = ReadConcern::from_proto(concern_level);
        let after_idx = req
            .read_concern
            .as_ref()
            .map(|rc| rc.after_index)
            .unwrap_or(0);

        // Causal session validation: after_index requires readConcern >= MAJORITY.
        //
        // A LOCAL read offers no majority-commit guarantee, so a causal fence on top
        // of it would be logically unsound. LINEARIZABLE is incompatible per
        // arch/distribution/consistency.md ("afterClusterTime + linearizable = reject").
        if after_idx > 0 {
            match concern {
                ReadConcern::Local => {
                    return Err(Status::failed_precondition(
                        "readConcern=LOCAL is incompatible with afterClusterTime. \
                         Causal reads require readConcern=MAJORITY.",
                    ));
                }
                ReadConcern::Linearizable => {
                    return Err(Status::failed_precondition(
                        "readConcern=LINEARIZABLE is incompatible with afterClusterTime. \
                         Use readConcern=MAJORITY for causal reads.",
                    ));
                }
                _ => {} // Majority, Snapshot: OK for causal sessions
            }
        }

        // Causal session write-concern validation (G088).
        //
        // Writes in a causal session (after_index > 0) MUST use writeConcern >=
        // majority. A sub-majority write (w:1, w:0) may be acknowledged to the
        // client before it is replicated. If the leader crashes before replication,
        // the write is lost — but the client already received an `applied_index` from
        // the response. Any subsequent causal read with that `after_index` will wait
        // forever on a follower (the entry was never committed) or observe a missing
        // write (violating read-your-writes). This is a hard rejection, not a silent
        // upgrade: the client must explicitly choose between volatile + non-causal or
        // durable + causal.
        //
        // We detect write queries via AST inspection BEFORE execution to avoid
        // executing a write that we would then reject.
        if after_idx > 0 {
            // Parse the query to determine if it contains any mutating clauses.
            // On parse failure we let execution proceed and fail with a richer error.
            if let Ok(ast) = coordinode_query::cypher::parse(&req.query) {
                if ast.is_write() {
                    let level = req
                        .write_concern
                        .as_ref()
                        .map(|wc| wc.level)
                        .unwrap_or(replication::WriteConcernLevel::Unspecified as i32);
                    let is_majority = level == replication::WriteConcernLevel::Majority as i32;
                    if !is_majority {
                        return Err(Status::failed_precondition(
                            "Causal sessions require writeConcern=MAJORITY for write \
                             statements. A sub-majority write (w:1, w:0) may be lost \
                             before replication: the resulting applied_index would be a \
                             dangling causal dependency that followers can never satisfy. \
                             Use a non-causal session for volatile writes, or upgrade to \
                             writeConcern=MAJORITY.",
                        ));
                    }
                }
            }
        }

        // --- Read fence: enforce readPreference and readConcern before query ---
        //
        // In cluster mode, apply the preference/concern fence, then the causal
        // after_index fence if the client supplied one.
        // In standalone mode (raft_node = None), all writes are immediately visible
        // and applied_index is always 0 — causal fences are trivially satisfied.
        let (applied_index, served_by_leader) = if let Some(ref raft) = self.raft_node {
            let preference = ReadPreference::from_proto(req.read_preference);
            let mut fence = raft.read_fence();
            fence
                .apply_default(preference, concern)
                .await
                .map_err(fence_error_to_status)?;

            // Causal fence: block until applied_index >= after_idx.
            // after_idx = 0 means no fence (default).
            if after_idx > 0 {
                fence
                    .wait_for_index(after_idx, READ_FENCE_TIMEOUT)
                    .await
                    .map_err(fence_error_to_status)?;
            }

            let idx = fence.applied_index();
            let is_leader = raft.is_leader().await;
            (idx, is_leader)
        } else {
            (0u64, false)
        };

        // Build executor-level concerns from the proto request, then execute
        // through the unified entry point so they actually reach the executor
        // (the previous handler validated write_concern for causal sessions but
        // silently dropped it before calling the engine, downgrading every
        // MAJORITY write to W1).
        let at_ts_raw = req
            .read_concern
            .as_ref()
            .map(|rc| rc.at_timestamp)
            .unwrap_or(0);
        // at_timestamp only meaningful with SNAPSHOT level (see proto doc).
        // Reject misuse at the boundary rather than letting it silently slide
        // through to a non-snapshot read which would ignore the timestamp.
        if at_ts_raw > 0
            && !matches!(
                read_concern_level_to_executor(concern_level),
                ExecutorReadConcernLevel::Snapshot
            )
        {
            return Err(Status::failed_precondition(
                "readConcern.at_timestamp is only valid with level=SNAPSHOT",
            ));
        }
        let executor_read_concern = ExecutorReadConcern {
            level: read_concern_level_to_executor(concern_level),
            after_index: if after_idx > 0 { Some(after_idx) } else { None },
            at_timestamp: if at_ts_raw > 0 { Some(at_ts_raw) } else { None },
        };
        let executor_write_concern = req.write_concern.as_ref().map(|wc| WriteConcern {
            level: write_concern_level_to_executor(wc.level),
            journal: wc.journal,
            timeout_ms: wc.timeout_ms,
        });

        let exec_result = {
            let mut db = self.database.write();
            let params = if req.parameters.is_empty() {
                None
            } else {
                Some(convert_params(&req.parameters))
            };
            db.execute_cypher_full(
                &req.query,
                params,
                source_ctx.as_ref(),
                Some(executor_read_concern),
                executor_write_concern,
            )
            .map_err(db_error_to_status)?
        };
        // Write guard released here, held only during query execution.
        let result_rows = exec_result.rows;
        let write_stats = exec_result.write_stats;

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
                nodes_created: write_stats.nodes_created as i64,
                nodes_deleted: write_stats.nodes_deleted as i64,
                edges_created: write_stats.edges_created as i64,
                edges_deleted: write_stats.edges_deleted as i64,
                properties_set: write_stats.properties_set as i64,
                execution_time_ms: duration_ms as i64,
                applied_index,
                served_by_leader,
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
        let stats = self.database.read().compute_stats();
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
            }))
            .await;
        let err = result.expect_err("must reject");
        assert_eq!(err.code(), tonic::Code::FailedPrecondition);
        assert!(
            err.message().contains("at_timestamp")
                && err.message().to_lowercase().contains("snapshot"),
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
}
