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
use coordinode_replicate::ReplicatedWriter;

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
        Value::MultiVector(rows) => {
            // Wire as list-of-vectors. Lossy on the type tag but preserves
            // all numeric data so the client can reconstruct the matrix.
            let items = rows
                .iter()
                .map(|row| common::PropertyValue {
                    value: Some(common::property_value::Value::VectorValue(common::Vector {
                        values: row.clone(),
                    })),
                })
                .collect();
            Some(common::property_value::Value::ListValue(
                common::PropertyList { values: items },
            ))
        }
        Value::Path(p) => {
            // No dedicated proto Path type yet, so wire the path as a
            // structured map { nodes: [...], rels: [{type, source, target}] }
            // by reusing the Map encoding. A first-class proto Path (and Bolt
            // Path PackStream struct) is the follow-up wire-encoding step.
            let mut m: std::collections::BTreeMap<String, Value> =
                std::collections::BTreeMap::new();
            m.insert(
                "nodes".to_string(),
                Value::Array(p.nodes.iter().map(|n| Value::Int(*n as i64)).collect()),
            );
            m.insert(
                "rels".to_string(),
                Value::Array(
                    p.rels
                        .iter()
                        .map(|r| {
                            let mut rm: std::collections::BTreeMap<String, Value> =
                                std::collections::BTreeMap::new();
                            rm.insert("type".to_string(), Value::String(r.edge_type.clone()));
                            rm.insert("source".to_string(), Value::Int(r.source as i64));
                            rm.insert("target".to_string(), Value::Int(r.target as i64));
                            Value::Map(rm)
                        })
                        .collect(),
                ),
            );
            value_to_proto(&Value::Map(m)).value
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
    /// Write coordination point: Cypher execution routes through this so the
    /// committed Raft index of a write reaches the response as the causal
    /// `operationTime`. Shares the same `Database` handle as `database`
    /// (used directly only for read-only EXPLAIN / stats paths).
    writer: ReplicatedWriter,
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
            writer: ReplicatedWriter::new(Arc::clone(&database)),
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

        // Interactive transaction statement (ADR-042): a non-zero transaction_id
        // runs this statement against the held transaction — reads at its pinned
        // snapshot, writes buffer until CommitTransaction. No per-statement
        // commit and no causal fence (the snapshot was pinned at BEGIN). A zero
        // transaction_id is the auto-commit path below.
        if req.transaction_id != 0 {
            let params = if req.parameters.is_empty() {
                None
            } else {
                Some(convert_params(&req.parameters))
            };
            let rows = self
                .database
                .read()
                .execute_in_transaction(req.transaction_id, &req.query, params)
                .map_err(db_error_to_status)?;
            let columns: Vec<String> = rows
                .first()
                .map(|r| r.keys().cloned().collect())
                .unwrap_or_default();
            let proto_rows: Vec<query::Row> = rows
                .iter()
                .map(|row| query::Row {
                    values: columns
                        .iter()
                        .map(|col| {
                            row.get(col)
                                .map(value_to_proto)
                                .unwrap_or(common::PropertyValue { value: None })
                        })
                        .collect(),
                })
                .collect();
            return Ok(Response::new(query::ExecuteCypherResponse {
                columns,
                rows: proto_rows,
                // Buffered statement: no commit yet, so no mutation stats and no
                // applied_index — those land on the CommitTransaction response.
                stats: Some(query::QueryStats {
                    nodes_created: 0,
                    nodes_deleted: 0,
                    edges_created: 0,
                    edges_deleted: 0,
                    properties_set: 0,
                    execution_time_ms: start.elapsed().as_millis() as i64,
                    applied_index: 0,
                    served_by_leader: false,
                }),
            }));
        }

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

        // Execute under a shared read lock by default. CypherService
        // accepts any Cypher (CREATE / MATCH / SET / …); the
        // shared path runs regular Cypher on `&self` concurrently
        // and explicitly rejects session-SET commands. On rejection
        // (SET requires exclusive access) we re-run under `.write()`
        // through the `&mut self` API. This keeps the hot read path
        // parallel without losing the embedded SET semantics.
        let exec_result = {
            let params = if req.parameters.is_empty() {
                None
            } else {
                Some(convert_params(&req.parameters))
            };
            self.writer
                .execute(
                    &req.query,
                    params,
                    source_ctx.as_ref(),
                    Some(&executor_read_concern),
                    executor_write_concern.as_ref(),
                )
                .map_err(db_error_to_status)?
        };
        // Read / write guard released here, held only during query execution.
        let result_rows = exec_result.rows;
        let write_stats = exec_result.write_stats;

        // Causal operationTime: a replicated write reports its OWN committed
        // Raft index, not the node's current applied index sampled around
        // execution by the read fence (which is not this write's index — the
        // operationTime inaccuracy). Reads keep the fence value. `None` (embedded /
        // non-replicated) falls back to the fence value.
        let applied_index = write_stats.applied_index.unwrap_or(applied_index);

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

    async fn begin_transaction(
        &self,
        _request: Request<query::BeginTransactionRequest>,
    ) -> Result<Response<query::BeginTransactionResponse>, Status> {
        let transaction_id = self.database.read().begin_transaction();
        Ok(Response::new(query::BeginTransactionResponse {
            transaction_id,
        }))
    }

    async fn commit_transaction(
        &self,
        request: Request<query::CommitTransactionRequest>,
    ) -> Result<Response<query::CommitTransactionResponse>, Status> {
        let applied_index = self
            .database
            .read()
            .commit_transaction(request.into_inner().transaction_id)
            .map_err(db_error_to_status)?;
        Ok(Response::new(query::CommitTransactionResponse {
            applied_index,
        }))
    }

    async fn rollback_transaction(
        &self,
        request: Request<query::RollbackTransactionRequest>,
    ) -> Result<Response<query::RollbackTransactionResponse>, Status> {
        self.database
            .read()
            .rollback_transaction(request.into_inner().transaction_id)
            .map_err(db_error_to_status)?;
        Ok(Response::new(query::RollbackTransactionResponse {}))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
