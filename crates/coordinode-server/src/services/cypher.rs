use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

// no-std: spin::RwLock (drop-in).
use parking_lot::RwLock;

use tonic::{Request, Response, Status};

use coordinode_core::graph::types::{PathRel, PathValue, Value, VectorConsistencyMode};
use coordinode_core::txn::read_concern::ReadConcernLevel as ExecutorReadConcernLevel;
use coordinode_core::txn::write_concern::{Journal, WriteAck, WriteConcern};
use coordinode_embed::{Database, DatabaseError};
use coordinode_query::advisor::QueryRegistry;
use coordinode_query::advisor::nplus1::NPlus1Detector;
use coordinode_query::advisor::source::{self, SourceContext, grpc_keys};
use coordinode_raft::cluster::RaftNode;
use coordinode_raft::read_fence::ReadPreference;

use coordinode_core::txn::command::{ActingContext, AdmissionSignal};
use coordinode_embed::db::{CommitOptions, ExpectedVersion, TransactionOptions};
use coordinode_embed::replication::routing::{
    RoutedAnswer, RoutedCommit, RoutedRollback, RoutedSettings, RoutedStatement,
};

use super::routing::routed_settings;
use super::statement::{
    Admission, Admitted, HOPS_HEADER, Requested, StatementExecutor, leader_hint, mismatch_metadata,
};
use crate::config::StatementDefaults;
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
        Value::Timestamp(ts) => Some(common::property_value::Value::TimestampValue(
            common::Timestamp { unix_micros: *ts },
        )),
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
        Value::MultiVector(rows) => Some(common::property_value::Value::MultiVectorValue(
            common::MultiVector {
                rows: rows
                    .iter()
                    .map(|row| common::Vector {
                        values: row.clone(),
                    })
                    .collect(),
            },
        )),
        Value::Path(p) => Some(common::property_value::Value::PathValue(common::Path {
            nodes: p.nodes.clone(),
            rels: p
                .rels
                .iter()
                .map(|r| common::PathRel {
                    edge_type: r.edge_type.clone(),
                    source: r.source,
                    target: r.target,
                })
                .collect(),
        })),
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
            common::property_value::Value::TimestampValue(ts) => Value::Timestamp(ts.unix_micros),
            common::property_value::Value::MultiVectorValue(mv) => {
                Value::MultiVector(mv.rows.iter().map(|row| row.values.clone()).collect())
            }
            common::property_value::Value::PathValue(p) => Value::Path(PathValue {
                nodes: p.nodes.clone(),
                rels: p
                    .rels
                    .iter()
                    .map(|r| PathRel {
                        edge_type: r.edge_type.clone(),
                        source: r.source,
                        target: r.target,
                    })
                    .collect(),
            }),
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
/// Translate a database failure into the status a client sees.
///
/// Classification is by TYPE, never by the text of a message. An earlier
/// version rebuilt the category by matching English prefixes on the rendered
/// error ("Cypher: execution error: /
/// by zero"), which made every message a wire contract: rewording one silently
/// changed the code a client received.
///
/// Whatever a caller may branch on also carries a machine-readable reason in
/// the status details, so branching does not require reading prose either. See
/// [`crate::services::error_details`].
pub(crate) fn db_error_to_status(err: DatabaseError) -> Status {
    use crate::services::error_details::{Reason, status_with_reason};
    use coordinode_query::executor::eval::EvalError;
    use coordinode_query::executor::runner::ExecutionError;
    use coordinode_storage::error::StorageError;
    use tonic::Code;

    let rendered = err.to_string();
    match &err {
        // Faults in the QUERY rather than in the server. Retrying them cannot
        // help and paging someone about them is noise, so they answer
        // INVALID_ARGUMENT, the same class a syntax error gets.
        DatabaseError::Parse(_) => {
            return status_with_reason(
                Code::InvalidArgument,
                format!("Parse error: {rendered}"),
                Reason::QuerySyntax,
                [],
            );
        }
        DatabaseError::Semantic(detail) => {
            return status_with_reason(
                Code::InvalidArgument,
                format!("Semantic error: {detail}"),
                Reason::QuerySemantics,
                [],
            );
        }
        DatabaseError::Execution(ExecutionError::Arithmetic(eval)) => {
            let (reason, metadata) = match eval {
                EvalError::DivideByZero => (Reason::DivideByZero, Vec::new()),
                EvalError::LongOverflow => (Reason::LongOverflow, Vec::new()),
                EvalError::UnknownFunction(name) => (
                    Reason::UnknownFunction,
                    // The name is what a caller acts on, whether to suggest a
                    // spelling or to report which call failed. Reading it back
                    // out of the message would be the same mistake this
                    // classification exists to undo.
                    vec![("function", name.clone())],
                ),
            };
            return status_with_reason(Code::InvalidArgument, eval.to_string(), reason, metadata);
        }
        // A CALL the catalog refused: a fault in the query, like a syntax
        // error, with the procedure and the argument or column it names.
        DatabaseError::Execution(ExecutionError::Procedure(refusal)) => {
            use coordinode_query::procedure::ProcedureError as P;
            let (reason, metadata): (Reason, Vec<(&str, String)>) = match refusal {
                P::Unknown { procedure } => (
                    Reason::UnknownProcedure,
                    vec![("procedure", procedure.clone())],
                ),
                P::MissingArgument {
                    procedure,
                    argument,
                }
                | P::ArgumentType {
                    procedure,
                    argument,
                    ..
                }
                | P::InvalidArgument {
                    procedure,
                    argument,
                    ..
                } => (
                    Reason::ProcedureCall,
                    vec![
                        ("procedure", procedure.clone()),
                        ("argument", argument.clone()),
                    ],
                ),
                P::UnknownOutput { procedure, column } => (
                    Reason::ProcedureCall,
                    vec![("procedure", procedure.clone()), ("column", column.clone())],
                ),
                P::TooManyArguments { procedure, .. } | P::YieldRequired { procedure } => (
                    Reason::ProcedureCall,
                    vec![("procedure", procedure.clone())],
                ),
                // A procedure breaking its own signature, or a registration
                // clash, is the server's fault, not the query's.
                P::OutputShape { .. } | P::DuplicateName { .. } => {
                    return Status::internal(rendered);
                }
            };
            return status_with_reason(Code::InvalidArgument, rendered, reason, metadata);
        }
        DatabaseError::Execution(ExecutionError::SchemaViolation(detail)) => {
            return status_with_reason(
                Code::FailedPrecondition,
                format!("Schema violation: {detail}"),
                Reason::SchemaViolation,
                [],
            );
        }
        DatabaseError::Execution(ExecutionError::DuplicateKey {
            table,
            key,
            element_id,
        }) => {
            return status_with_reason(
                Code::AlreadyExists,
                rendered,
                Reason::DuplicateKey,
                [
                    ("table", table.clone()),
                    ("key", key.clone()),
                    ("element_id", element_id.clone()),
                ],
            );
        }
        DatabaseError::Execution(ExecutionError::UniqueViolation {
            index,
            property,
            value,
            element_id,
        }) => {
            return status_with_reason(
                Code::AlreadyExists,
                rendered,
                Reason::DuplicateKey,
                [
                    ("index", index.clone()),
                    ("property", property.clone()),
                    ("key", value.clone()),
                    ("element_id", element_id.clone()),
                ],
            );
        }
        // Not a duplicate: the value may be free. UNAVAILABLE, since the
        // same write succeeds once the build covers the stored nodes.
        // A spent budget is not a server fault: the statement asked for more
        // than its limit, or for longer than its caller waits.
        DatabaseError::Execution(ExecutionError::Budget(stop)) => {
            use coordinode_core::budget::BudgetStop;
            return match stop {
                BudgetStop::Memory {
                    requested,
                    used,
                    limit,
                } => status_with_reason(
                    Code::ResourceExhausted,
                    rendered,
                    Reason::ExceedsMemoryBudget,
                    [
                        ("requested_bytes", requested.to_string()),
                        ("used_bytes", used.to_string()),
                        ("limit_bytes", limit.to_string()),
                    ],
                ),
                BudgetStop::Deadline => status_with_reason(
                    Code::DeadlineExceeded,
                    rendered,
                    Reason::QueryDeadlineExceeded,
                    [],
                ),
                BudgetStop::Cancelled => {
                    status_with_reason(Code::Cancelled, rendered, Reason::QueryCancelled, [])
                }
            };
        }
        DatabaseError::Execution(ExecutionError::IndexUnresolved {
            label,
            property,
            operator,
        }) => {
            return status_with_reason(
                Code::Unavailable,
                rendered,
                Reason::IndexUnresolved,
                [
                    ("label", label.clone()),
                    ("property", property.clone()),
                    ("operator", operator.to_string()),
                ],
            );
        }
        DatabaseError::SessionSetting(refused) => {
            return super::error_details::invalid_field(refused.setting, refused.reason.clone());
        }
        // A retry of a settled operation ran nothing; the outcome is its
        // answer, carried as metadata a client branches on.
        DatabaseError::OperationSettled {
            outcome,
            intent_differs,
            ..
        }
        | DatabaseError::Execution(ExecutionError::OperationSettled {
            outcome,
            intent_differs,
            ..
        }) => {
            return super::operation::settled_status(rendered, *outcome, *intent_differs);
        }
        DatabaseError::ForeignOperation { descriptor } => {
            return super::operation::foreign_status(rendered, *descriptor);
        }
        DatabaseError::Execution(ExecutionError::UniquenessUnresolved { index, limit }) => {
            return status_with_reason(
                Code::Unavailable,
                rendered,
                Reason::UniquenessUnresolved,
                [("index", index.clone()), ("limit", limit.to_string())],
            );
        }
        DatabaseError::Execution(ExecutionError::ConstraintViolation {
            constraint,
            kind,
            label,
            property,
            element_id,
            ..
        }) => {
            return status_with_reason(
                Code::FailedPrecondition,
                rendered,
                Reason::ConstraintViolation,
                [
                    ("constraint", constraint.clone()),
                    ("kind", kind.to_string()),
                    ("label", label.clone()),
                    ("property", property.clone()),
                    ("element_id", element_id.clone()),
                ],
            );
        }
        DatabaseError::Execution(ExecutionError::KeyImmutable { table, column }) => {
            return status_with_reason(
                Code::FailedPrecondition,
                rendered,
                Reason::KeyImmutable,
                [("table", table.clone()), ("column", column.clone())],
            );
        }
        DatabaseError::Execution(ExecutionError::CatalogObjectExists { object, name }) => {
            return crate::services::error_details::catalog_object_status(
                Code::AlreadyExists,
                rendered,
                Reason::CatalogObjectExists,
                &object.to_string(),
                name,
            );
        }
        DatabaseError::Execution(ExecutionError::CatalogObjectMissing { object, name }) => {
            return crate::services::error_details::catalog_object_status(
                Code::NotFound,
                rendered,
                Reason::CatalogObjectNotFound,
                &object.to_string(),
                name,
            );
        }
        DatabaseError::Execution(ExecutionError::CatalogRefused(_)) => {
            return status_with_reason(
                Code::FailedPrecondition,
                rendered,
                Reason::CatalogChangeRefused,
                [],
            );
        }
        DatabaseError::Execution(ExecutionError::CatalogObjectDiffers {
            object,
            name,
            revision,
        }) => {
            return status_with_reason(
                Code::FailedPrecondition,
                rendered,
                Reason::CatalogObjectDiffers,
                [
                    ("object", object.to_string()),
                    ("name", name.clone()),
                    ("revision", revision.to_string()),
                ],
            );
        }
        DatabaseError::Execution(ExecutionError::CatalogRevisionMismatch {
            object,
            name,
            expected,
            current,
        }) => {
            return status_with_reason(
                Code::FailedPrecondition,
                rendered,
                Reason::CatalogRevisionMismatch,
                [
                    ("object", object.to_string()),
                    ("name", name.clone()),
                    ("expected", expected.to_string()),
                    (
                        "current",
                        current.map(|c| c.to_string()).unwrap_or_default(),
                    ),
                ],
            );
        }
        // Transaction lifecycle. NOT_FOUND rather than INVALID_ARGUMENT: the
        // ID was well-formed, there is simply nothing under it here, which
        // is what a new leader says of a transaction an earlier one held.
        DatabaseError::UnknownTransaction(id) => {
            return status_with_reason(
                Code::NotFound,
                rendered,
                Reason::UnknownTransaction,
                [("transaction", id.to_string())],
            );
        }
        // ALREADY_EXISTS: the start names an ID that is taken, and nothing ran.
        DatabaseError::TransactionIdInUse(id) => {
            return status_with_reason(
                Code::AlreadyExists,
                rendered,
                Reason::TransactionIdInUse,
                [("transaction", id.to_string())],
            );
        }
        // ABORTED like a conflict: nothing of it was applied and the retry is
        // the whole transaction, under a new ID.
        DatabaseError::TransactionAborted { id, end } => {
            use coordinode_embed::TransactionEnd as End;
            let end = match end {
                End::StatementFailed => "STATEMENT_FAILED",
                End::DrainTimedOut => "DRAIN_TIMED_OUT",
                End::IdleExpired => "IDLE_EXPIRED",
                End::RolledBack => "ROLLED_BACK",
            };
            return status_with_reason(
                Code::Aborted,
                rendered,
                Reason::TransactionAborted,
                [("transaction", id.to_string()), ("end", end.to_owned())],
            );
        }
        // ABORTED as gRPC defines it for a concurrency conflict; unlike the
        // others the transaction stays open, so the retry is this command.
        DatabaseError::TransactionBusy(id) => {
            return status_with_reason(
                Code::Aborted,
                rendered,
                Reason::TransactionBusy,
                [("transaction", id.to_string())],
            );
        }
        // The statement exists already, applied once; its rows are gone.
        DatabaseError::StatementAlreadyAdmitted { id, nonce } => {
            return status_with_reason(
                Code::AlreadyExists,
                rendered,
                Reason::StatementAlreadyAdmitted,
                [
                    ("transaction", id.to_string()),
                    ("nonce", nonce.to_string()),
                ],
            );
        }
        DatabaseError::StatementNonceReused { id, nonce } => {
            return status_with_reason(
                Code::InvalidArgument,
                rendered,
                Reason::StatementNonceReused,
                [
                    ("transaction", id.to_string()),
                    ("nonce", nonce.to_string()),
                ],
            );
        }
        DatabaseError::StatementNonceRequired(_) => {
            return super::error_details::invalid_field(
                "nonce",
                "a statement of an ordered transaction is numbered from 1",
            );
        }
        DatabaseError::TransactionConflict { id, .. } => {
            return status_with_reason(
                Code::Aborted,
                rendered,
                Reason::TransactionConflict,
                [("transaction", id.to_string())],
            );
        }
        // The auto-commit spelling of the same refusal: nothing was applied
        // and the retry is the whole statement, so it is ABORTED too, with no
        // transaction id to name.
        DatabaseError::Execution(coordinode_query::executor::runner::ExecutionError::Conflict(
            _,
        )) => {
            return status_with_reason(Code::Aborted, rendered, Reason::TransactionConflict, []);
        }
        // Both spellings of one refusal: an interactive commit fails with the
        // typed variant, an auto-commit statement carries it inside the
        // execution error. ABORTED like a conflict (the retry is the whole
        // transaction), but under its own reason, because the cause is a
        // condition rather than a contended key.
        DatabaseError::InvariantRefused { id, .. } => {
            return status_with_reason(
                Code::Aborted,
                rendered,
                Reason::InvariantRefused,
                [("transaction", id.to_string())],
            );
        }
        DatabaseError::Execution(
            coordinode_query::executor::runner::ExecutionError::InvariantRefused(_),
        ) => {
            return status_with_reason(Code::Aborted, rendered, Reason::InvariantRefused, []);
        }
        // Both spellings of a version mismatch. The versions travel in the
        // metadata rather than only in the message, because the caller's next
        // move is computed from them: retry against what is there, merge, or
        // stop. An absent version is absent from the metadata rather than
        // rendered as a zero, which would read as a real version.
        DatabaseError::RevisionMismatch {
            expected, current, ..
        }
        | DatabaseError::Execution(
            coordinode_query::executor::runner::ExecutionError::RevisionMismatch {
                expected,
                current,
            },
        ) => {
            let mut metadata: Vec<(&str, String)> = Vec::with_capacity(2);
            if let Some(expected) = expected {
                metadata.push(("expected_version", expected.to_string()));
            }
            if let Some(current) = current {
                metadata.push(("current_version", current.to_string()));
            }
            return status_with_reason(Code::Aborted, rendered, Reason::RevisionMismatch, metadata);
        }
        DatabaseError::TransactionTooLarge {
            id,
            buffered,
            limit,
        } => {
            return status_with_reason(
                Code::ResourceExhausted,
                rendered,
                Reason::TransactionTooLarge,
                [
                    ("transaction", id.to_string()),
                    ("buffered_bytes", buffered.to_string()),
                    ("limit_bytes", limit.to_string()),
                ],
            );
        }
        // Both spellings of the same condition: an interactive commit fails
        // with the typed variant, an auto-commit statement carries it inside
        // the execution error. Retryable with a delay (RetryInfo), unlike a
        // conflict: the server is shedding writes until compaction catches up.
        DatabaseError::WriteBackpressure
        | DatabaseError::Execution(
            coordinode_query::executor::runner::ExecutionError::Backpressure,
        ) => {
            return status_with_reason(
                Code::ResourceExhausted,
                rendered,
                Reason::WriteBackpressure,
                [],
            );
        }
        // Both spellings again, for a write that reached a node which is not
        // the leader. FAILED_PRECONDITION rather than UNAVAILABLE: the server
        // is perfectly available, it is the request that came to the wrong
        // node, and the leader id says where the right one is. Nothing was
        // applied, so the retry is the same write, not a repair.
        DatabaseError::NotLeader { leader_id }
        | DatabaseError::Execution(
            coordinode_query::executor::runner::ExecutionError::NotLeader { leader_id },
        ) => {
            let metadata = leader_id
                .map(|id| vec![("leader_id", id.to_string())])
                .unwrap_or_default();
            return status_with_reason(
                Code::FailedPrecondition,
                rendered,
                Reason::NotLeader,
                metadata,
            );
        }
        // A write to a member that does not run its group's version. Like a
        // write to a follower: the request is fine and the server available,
        // it came to a member that takes no writes, and the metadata names
        // where to send it and why this member refuses.
        DatabaseError::Mismatched(m) | DatabaseError::Execution(ExecutionError::Mismatched(m)) => {
            return status_with_reason(
                Code::FailedPrecondition,
                rendered,
                Reason::MemberReadOnly,
                mismatch_metadata(m),
            );
        }
        // A time-travel read older than the retention horizon. OUT_OF_RANGE
        // rather than FAILED_PRECONDITION: the same read is valid at a later
        // timestamp, and the metadata says from which one, so a caller can
        // clamp instead of guessing.
        DatabaseError::OutsideRetention {
            oldest_readable, ..
        }
        | DatabaseError::Execution(ExecutionError::OutsideRetention {
            oldest_readable, ..
        }) => {
            return status_with_reason(
                Code::OutOfRange,
                rendered,
                Reason::OutsideRetention,
                [("oldest_readable_ts", oldest_readable.to_string())],
            );
        }
        // A read at a timestamp whose commits have not landed or are not
        // durable yet. UNAVAILABLE: the same read succeeds shortly, and the
        // server does not wait for it on the caller's behalf.
        DatabaseError::ReadNotReady {
            requested,
            ready_through,
        } => {
            let mut metadata = vec![("requested_ts", requested.to_string())];
            if let Some(ts) = ready_through {
                metadata.push(("ready_through_ts", ts.to_string()));
            }
            return status_with_reason(Code::Unavailable, rendered, Reason::ReadNotReady, metadata);
        }
        // The engine's own guard on a snapshot read below the watermark:
        // the same condition reached through a storage-level read.
        DatabaseError::Storage(StorageError::SnapshotOutsideRetention { watermark, .. })
        | DatabaseError::Execution(ExecutionError::Storage(
            StorageError::SnapshotOutsideRetention { watermark, .. },
        )) => {
            // Refused only for a snapshot below the watermark, so the
            // watermark is at least 1.
            debug_assert!(*watermark > 0, "a retention refusal at watermark 0");
            return status_with_reason(
                Code::OutOfRange,
                rendered,
                Reason::OutsideRetention,
                [("oldest_readable_ts", (*watermark - 1).to_string())],
            );
        }
        _ => {}
    }
    // Capacity exhaustion arrives wrapped in either a Storage or an Execution
    // variant, and the shared helper knows how to find it in both. Everything
    // it does not recognise stays INTERNAL, which is the honest answer for a
    // failure the server cannot attribute to the request.
    crate::services::db_err_to_status("Cypher", err)
}

/// The read concern level a request names in `field`: `None` for
/// UNSPECIFIED, which leaves it to the session and then the server's default.
/// A value this server does not know is refused rather than read as some
/// other level, which could serve a weaker read than the one asked for.
pub(crate) fn read_concern_level(
    field: &str,
    level: i32,
) -> Result<Option<ExecutorReadConcernLevel>, Status> {
    use replication::ReadConcernLevel as Wire;
    match Wire::try_from(level) {
        Ok(Wire::Unspecified) => Ok(None),
        Ok(Wire::Local) => Ok(Some(ExecutorReadConcernLevel::Local)),
        Ok(Wire::Majority) => Ok(Some(ExecutorReadConcernLevel::Majority)),
        Ok(Wire::Linearizable) => Ok(Some(ExecutorReadConcernLevel::Linearizable)),
        Ok(Wire::Snapshot) => Ok(Some(ExecutorReadConcernLevel::Snapshot)),
        Err(_) => Err(super::error_details::invalid_field(
            field,
            format!("read concern level {level} is not one this server knows"),
        )),
    }
}

/// The vector consistency a request names in `field`: `None` for
/// UNSPECIFIED, which leaves it to the session and then to the query's read
/// consistency. A value this server does not know is refused.
pub(crate) fn vector_consistency(
    field: &str,
    mode: i32,
) -> Result<Option<VectorConsistencyMode>, Status> {
    use replication::VectorConsistency as Wire;
    match Wire::try_from(mode) {
        Ok(Wire::Unspecified) => Ok(None),
        Ok(Wire::Current) => Ok(Some(VectorConsistencyMode::Current)),
        Ok(Wire::Snapshot) => Ok(Some(VectorConsistencyMode::Snapshot)),
        Ok(Wire::Exact) => Ok(Some(VectorConsistencyMode::Exact)),
        Err(_) => Err(super::error_details::invalid_field(
            field,
            format!("vector consistency {mode} is not one this server knows"),
        )),
    }
}

/// A vector consistency on the wire.
pub(crate) fn vector_consistency_to_proto(mode: VectorConsistencyMode) -> i32 {
    use replication::VectorConsistency as Wire;
    match mode {
        VectorConsistencyMode::Current => Wire::Current as i32,
        VectorConsistencyMode::Snapshot => Wire::Snapshot as i32,
        VectorConsistencyMode::Exact => Wire::Exact as i32,
    }
}

/// A build-wait bound in milliseconds on the wire, as the engine takes it.
pub(crate) fn build_wait(ms: Option<u32>) -> Option<std::time::Duration> {
    ms.map(|ms| std::time::Duration::from_millis(u64::from(ms)))
}

/// A build-wait bound in whole milliseconds for the wire. A bound beyond
/// what the field holds (about 49 days) is sent as the largest it holds,
/// which waits as long in practice.
pub(crate) fn build_wait_ms(wait: std::time::Duration) -> u32 {
    u32::try_from(wait.as_millis()).unwrap_or(u32::MAX)
}

/// The read preference a request names in `field`: `None` for UNSPECIFIED,
/// which leaves it to the session and then the server's default. A value this
/// server does not know is refused rather than read as PRIMARY.
pub(crate) fn read_preference(
    field: &str,
    preference: i32,
) -> Result<Option<ReadPreference>, Status> {
    use replication::ReadPreference as Wire;
    match Wire::try_from(preference) {
        Ok(Wire::Unspecified) => Ok(None),
        Ok(_) => Ok(Some(ReadPreference::from_proto(preference))),
        Err(_) => Err(super::error_details::invalid_field(
            field,
            format!("read preference {preference} is not one this server knows"),
        )),
    }
}

/// Translate a wire `WriteConcern` to the executor's.
///
/// `w` unset is majority and `journal` unset is journaled, the same defaults
/// the embedded library uses: an acknowledged write survives the loss of a
/// minority unless the caller explicitly asked for less. A value this build
/// does not know, or a combination the engine cannot honour, is refused
/// rather than mapped to something weaker or stronger than what was asked.
pub(crate) fn write_concern_from_proto(
    wc: &replication::WriteConcern,
) -> Result<WriteConcern, Status> {
    use coordinode_core::txn::write_concern::WriteConcernError;
    use replication::write_concern::W;
    use tonic::Code;
    use tonic_types::{ErrorDetails, StatusExt};

    use crate::services::error_details::{ERROR_DOMAIN, Reason};

    // The reason says what kind of failure this is; the BadRequest violation
    // says which field to look at.
    let refuse = |field: &str, message: String| {
        let mut details = ErrorDetails::with_error_info(
            Reason::InvalidWriteConcern.as_str(),
            ERROR_DOMAIN,
            HashMap::new(),
        );
        details.add_bad_request_violation(field, message.clone());
        Status::with_error_details(Code::InvalidArgument, message, details)
    };

    let w = match wc.w {
        None => WriteAck::Majority,
        Some(W::Acks(n)) => WriteAck::Acks(n),
        Some(W::Mode(mode)) => match replication::WriteConcernMode::try_from(mode) {
            Ok(replication::WriteConcernMode::Majority) => WriteAck::Majority,
            Ok(replication::WriteConcernMode::Unspecified) | Err(_) => {
                return Err(refuse(
                    "write_concern.mode",
                    format!(
                        "write_concern.mode {mode} is not a mode this server knows; \
                         leave `w` unset for MAJORITY or name a member count with `acks`"
                    ),
                ));
            }
        },
    };
    let journal = match replication::Journal::try_from(wc.journal) {
        Ok(replication::Journal::Unspecified) | Ok(replication::Journal::Journal) => {
            Journal::Journal
        }
        Ok(replication::Journal::Cache) => Journal::Cache,
        Ok(replication::Journal::Memory) => Journal::Memory,
        Err(_) => {
            return Err(refuse(
                "write_concern.journal",
                format!(
                    "write_concern.journal {} is not a journal level this server knows",
                    wc.journal
                ),
            ));
        }
    };
    let concern = WriteConcern {
        w,
        journal,
        timeout_ms: wc.timeout_ms,
    };
    concern.validate(None).map_err(|e| {
        let field = match e {
            WriteConcernError::TooManyAcks { .. } => "write_concern.acks",
            WriteConcernError::VolatileReplication(_) => "write_concern.journal",
        };
        refuse(field, e.to_string())
    })?;
    Ok(concern)
}

/// Render an executor `WriteConcern` on the wire, so a setting is confirmed
/// by what is in effect rather than by what was asked for.
pub(crate) fn write_concern_to_proto(wc: &WriteConcern) -> replication::WriteConcern {
    use replication::write_concern::W;

    replication::WriteConcern {
        w: Some(match wc.w {
            WriteAck::Acks(n) => W::Acks(n),
            WriteAck::Majority => W::Mode(replication::WriteConcernMode::Majority as i32),
        }),
        journal: match wc.journal {
            Journal::Journal => replication::Journal::Journal,
            Journal::Cache => replication::Journal::Cache,
            Journal::Memory => replication::Journal::Memory,
        } as i32,
        timeout_ms: wc.timeout_ms,
    }
}

pub struct CypherServiceImpl {
    /// The path every statement takes: settings, fence, forwarding,
    /// execution and the advisor, shared with the session.
    executor: StatementExecutor,
}

impl CypherServiceImpl {
    pub fn new(
        database: Arc<RwLock<Database>>,
        query_registry: Arc<QueryRegistry>,
        nplus1_detector: Arc<NPlus1Detector>,
    ) -> Self {
        Self::from_executor(
            StatementExecutor::new(database).with_advisor(query_registry, nplus1_detector),
        )
    }

    /// Serve statements through `executor`, the one the session shares.
    pub fn from_executor(executor: StatementExecutor) -> Self {
        Self { executor }
    }

    /// Fence a request that names no read concern or preference with these
    /// instead of the built-in ones.
    pub fn with_statement_defaults(mut self, defaults: StatementDefaults) -> Self {
        self.executor = self.executor.with_statement_defaults(defaults);
        self
    }

    /// Attach a Raft node for read fence enforcement (cluster mode).
    pub fn with_raft_node(mut self, raft_node: Arc<RaftNode>) -> Self {
        self.executor = self.executor.with_raft_node(raft_node);
        self
    }

    fn database(&self) -> &Arc<RwLock<Database>> {
        self.executor.database()
    }

    /// Run the autonomous statement `req` at `leader` with the settings this
    /// member resolved, and answer with what the leader answered.
    async fn run_at(
        &self,
        leader: u64,
        req: query::ExecuteCypherRequest,
        requested: &Requested,
        source: Option<SourceContext>,
        start: std::time::Instant,
    ) -> Result<Response<query::ExecuteCypherResponse>, Status> {
        let statement = RoutedStatement {
            actor: ActingContext::Trusted,
            params: convert_params(&req.parameters),
            query: req.query,
            settings: routed_settings(requested),
            transaction: None,
            start_transaction: false,
            nonce: None,
            operation: requested.operation,
            deadline: time_left(requested.deadline),
            source,
        };
        let answer = self
            .executor
            .route(leader, statement, &AdmissionSignal::new(|| {}))
            .await?;
        Ok(routed_response(answer, true, start))
    }
}

/// What is left until `deadline`, zero once it passed; `None` without one.
fn time_left(deadline: Option<std::time::Instant>) -> Option<Duration> {
    deadline.map(|d| d.saturating_duration_since(std::time::Instant::now()))
}

/// A routed statement's answer as this member's response. `led` says the
/// member that ran it was leading, which an autonomous statement routed to
/// the leader always was. A write reports its own commit's index, one
/// answered before its commit reports none, and a read the index it was
/// served at, as on the local path.
fn routed_response(
    answer: RoutedAnswer,
    led: bool,
    start: std::time::Instant,
) -> Response<query::ExecuteCypherResponse> {
    let stats = answer.stats;
    let applied_index = match (stats.applied_index, stats.commit_ts) {
        (0, 0) => stats.read_index,
        (0, _) => 0,
        (own, _) => own,
    };
    let mut response = Response::new(query::ExecuteCypherResponse {
        columns: answer.columns,
        rows: answer
            .rows
            .iter()
            .map(|row| query::Row {
                values: row.iter().map(value_to_proto).collect(),
            })
            .collect(),
        stats: Some(query::QueryStats {
            nodes_created: stats.nodes_created,
            nodes_deleted: stats.nodes_deleted,
            edges_created: stats.edges_created,
            edges_deleted: stats.edges_deleted,
            properties_set: stats.properties_set,
            execution_time_ms: start.elapsed().as_millis() as i64,
            applied_index,
            served_by_leader: led,
            commit_ts: stats.commit_ts,
            read_as_of_ts: stats.read_as_of_ts,
        }),
    });
    response.metadata_mut().insert(
        HOPS_HEADER,
        tonic::metadata::MetadataValue::from_static("1"),
    );
    response
}

/// When a call carrying `metadata` must have finished: its `grpc-timeout`
/// from now, or `None` without one. The header is the gRPC over HTTP/2
/// format, at most eight ASCII digits and a unit (`H`, `M`, `S`, `m`, `u`,
/// `n`); a malformed one sets no deadline, as tonic itself treats it.
pub(crate) fn grpc_deadline(metadata: &tonic::metadata::MetadataMap) -> Option<std::time::Instant> {
    let text = metadata.get("grpc-timeout")?.to_str().ok()?;
    let (digits, unit) = text.split_at(text.len().checked_sub(1)?);
    if digits.is_empty() || digits.len() > 8 || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let value: u64 = digits.parse().ok()?;
    let timeout = match unit {
        "H" => Duration::from_secs(value.checked_mul(3600)?),
        "M" => Duration::from_secs(value.checked_mul(60)?),
        "S" => Duration::from_secs(value),
        "m" => Duration::from_millis(value),
        "u" => Duration::from_micros(value),
        "n" => Duration::from_nanos(value),
        _ => return None,
    };
    std::time::Instant::now().checked_add(timeout)
}

/// The memory limit a request names in MiB, as bytes within the ceiling.
pub(crate) fn memory_limit_from_proto(mib: Option<u32>) -> Result<Option<u64>, Status> {
    let Some(mib) = mib else {
        return Ok(None);
    };
    coordinode_embed::db::checked_query_memory_limit(u64::from(mib) << 20)
        .map(Some)
        .map_err(|refused| {
            super::error_details::invalid_field("query_memory_limit_mb", refused.to_string())
        })
}

/// Throws the statement's cancel switch when dropped: when the call's future
/// is dropped because the client cancelled or went away, the statement still
/// running on the blocking pool stops at its next check. Dropped after the
/// statement finished, it changes nothing.
struct CancelOnDrop(coordinode_core::budget::CancelFlag);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

/// The settings a unary request names, as the executor takes them. A level,
/// preference or concern left unspecified is `None`: the server's default.
fn requested_from_proto(
    req: &query::ExecuteCypherRequest,
    deadline: Option<std::time::Instant>,
    cancel: &coordinode_core::budget::CancelFlag,
) -> Result<Requested, Status> {
    let rc = req.read_concern.as_ref();
    Ok(Requested {
        read_concern: match rc {
            Some(rc) => read_concern_level("read_concern.level", rc.level)?,
            None => None,
        },
        after_index: rc.map_or(0, |rc| rc.after_index),
        at_timestamp: rc.map_or(0, |rc| rc.at_timestamp),
        read_preference: read_preference("read_preference", req.read_preference)?,
        write_concern: req
            .write_concern
            .as_ref()
            .map(write_concern_from_proto)
            .transpose()?,
        vector_consistency: vector_consistency("vector_consistency", req.vector_consistency)?,
        vector_build_wait: build_wait(req.vector_build_wait_ms),
        query_memory_limit: memory_limit_from_proto(req.query_memory_limit_mb)?,
        deadline,
        cancel: Some(cancel.clone()),
        operation: super::operation::optional_descriptor("operation", req.operation.as_ref())?,
    })
}

#[tonic::async_trait]
impl query::cypher_service_server::CypherService for CypherServiceImpl {
    async fn execute_cypher(
        &self,
        request: Request<query::ExecuteCypherRequest>,
    ) -> Result<Response<query::ExecuteCypherResponse>, Status> {
        let source_ctx = extract_source_context(&request);
        let deadline = grpc_deadline(request.metadata());
        let cancel = coordinode_core::budget::CancelFlag::new();
        let _cancel_on_drop = CancelOnDrop(cancel.clone());
        let req = request.into_inner();

        let start = std::time::Instant::now();

        if req.start_transaction && req.transaction_id.is_empty() {
            return Err(super::error_details::invalid_field(
                "start_transaction",
                "a statement starts a transaction only under its transaction_id",
            ));
        }
        // Interactive transaction statement: the transaction ID runs this
        // statement against the open transaction: reads at its pinned
        // snapshot, writes buffer until CommitTransaction. No per-statement
        // commit and no causal fence (the snapshot was pinned at the start).
        // The statement flagged as the start opens it first. No ID is the
        // auto-commit path below.
        if !req.transaction_id.is_empty() {
            let transaction =
                super::error_details::transaction_id("transaction_id", &req.transaction_id)?;
            // The start fixes the transaction's concerns from its statement.
            let opening = if req.start_transaction {
                let rc = req.read_concern.as_ref();
                Some(TransactionOptions {
                    actor: None,
                    read_concern: match rc {
                        Some(rc) => read_concern_level("read_concern.level", rc.level)?,
                        None => None,
                    },
                    write_concern: req
                        .write_concern
                        .as_ref()
                        .map(write_concern_from_proto)
                        .transpose()?,
                    ordering: coordinode_core::operations::TxnOrdering::Unordered,
                    drain_timeout: None,
                })
            } else {
                None
            };
            // The transaction's operation is named on its commit: one named on
            // a statement inside it would bind part of the transaction.
            if req.operation.is_some() {
                return Err(super::error_details::invalid_field(
                    "operation",
                    "a statement inside a transaction carries no operation; name it on \
                     CommitTransactionRequest.operation",
                ));
            }
            let params = if req.parameters.is_empty() {
                None
            } else {
                Some(convert_params(&req.parameters))
            };
            let query_memory_limit = memory_limit_from_proto(req.query_memory_limit_mb)?;
            // Another member leads: the statement, and its start, run there.
            if let Some(leader) = self.executor.transaction_leader()? {
                let statement = RoutedStatement {
                    actor: ActingContext::Trusted,
                    query: req.query,
                    params: params.unwrap_or_default(),
                    settings: RoutedSettings {
                        query_memory_limit,
                        read_concern: opening.as_ref().and_then(|o| o.read_concern),
                        write_concern: opening.as_ref().and_then(|o| o.write_concern),
                        ..RoutedSettings::default()
                    },
                    transaction: Some(transaction),
                    start_transaction: opening.is_some(),
                    nonce: None,
                    operation: None,
                    deadline: time_left(deadline),
                    source: source_ctx,
                };
                let answer = self
                    .executor
                    .route(leader, statement, &AdmissionSignal::new(|| {}))
                    .await?;
                return Ok(routed_response(answer, false, start));
            }
            let options = coordinode_embed::db::StatementOptions {
                query_memory_limit,
                deadline,
                cancel: Some(cancel.clone()),
                ..Default::default()
            };
            let rows = super::blocking(|| {
                let db = self.database().read();
                if let Some(opening) = &opening {
                    db.begin_transaction_as(transaction, opening)?;
                }
                db.execute_in_transaction_with(transaction, &req.query, params, &options)
            })
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
                // Buffered statement: no commit yet, so no mutation stats, no
                // applied_index and no commit timestamp — those land on the
                // CommitTransaction response, which is where this statement's
                // writes will actually be committed.
                stats: Some(query::QueryStats {
                    nodes_created: 0,
                    nodes_deleted: 0,
                    edges_created: 0,
                    edges_deleted: 0,
                    properties_set: 0,
                    execution_time_ms: start.elapsed().as_millis() as i64,
                    applied_index: 0,
                    served_by_leader: false,
                    commit_ts: 0,
                    // The transaction's commit decides; on a read-only member
                    // it is refused there.
                    read_as_of_ts: 0,
                }),
            }));
        }

        // Resolve and check the settings, fence the read and wait for the
        // causal position. A statement that needs the leader runs there.
        let requested = requested_from_proto(&req, deadline, &cancel)?;
        let admitted = match self.executor.admit(&req.query, &requested, false).await? {
            Admission::Run(admitted) => admitted,
            Admission::Forward(leader_id) => {
                return self
                    .run_at(leader_id, req, &requested, source_ctx, start)
                    .await;
            }
        };

        let exec_result = {
            let params = if req.parameters.is_empty() {
                None
            } else {
                Some(convert_params(&req.parameters))
            };
            match super::blocking(|| {
                self.executor
                    .execute(&req.query, params, source_ctx.as_ref(), &admitted)
            }) {
                Ok(result) => result,
                Err(e) => {
                    // The write came to a node that cannot replicate it. If we
                    // know who leads, it runs there and its answer is ours:
                    // the client never learns that leadership moved. Nothing
                    // was applied here, so running it there is its first run,
                    // not a repeat.
                    match leader_hint(&e) {
                        Some(leader_id) => {
                            return self
                                .run_at(leader_id, req, &requested, source_ctx, start)
                                .await;
                        }
                        None => return Err(db_error_to_status(e)),
                    }
                }
            }
        };
        let result_rows = exec_result.rows;
        let write_stats = exec_result.write_stats;
        let Admitted {
            applied_index,
            served_by_leader,
            read_as_of_ts,
            ..
        } = admitted;

        // Causal operationTime: a replicated write reports its OWN committed
        // Raft index, not the node's applied index sampled by the read fence,
        // which does not cover this write. A write answered before its commit
        // (a volatile journal, fewer acks than a majority) has no index of its
        // own yet and reports none. Reads keep the fence value.
        let applied_index = match (write_stats.applied_index, write_stats.commit_ts) {
            (Some(own), _) => own,
            (None, Some(_)) => 0,
            (None, None) => applied_index,
        };

        let duration_ms = start.elapsed().as_millis() as u64;
        self.executor.record(&req.query, source_ctx.as_ref(), start);

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
                // The version of what this statement wrote, for a statement
                // that committed on its own. Zero when there was no commit of
                // its own to report: a read, or a statement of an interactive
                // transaction, whose timestamp belongs to the commit that
                // ends it.
                commit_ts: write_stats.commit_ts.unwrap_or(0),
                read_as_of_ts,
            }),
        }))
    }

    async fn explain_cypher(
        &self,
        request: Request<query::ExplainCypherRequest>,
    ) -> Result<Response<query::ExplainCypherResponse>, Status> {
        let req = request.into_inner();

        // The plan ExecuteCypher would run, index selection and push-down
        // included, so what is explained is what executes.
        let (plan, suggest_result, stats) = super::blocking(|| {
            let db = self.database().read();
            let stats = db.compute_stats();
            let plan = db.explain_plan(&req.query, stats.as_ref())?;
            let suggest = db.suggest_for(&plan, stats.as_ref());
            Ok::<_, coordinode_embed::DatabaseError>((plan, suggest, stats))
        })
        .map_err(db_error_to_status)?;
        let stats_ref = stats
            .as_ref()
            .map(|s| s as &dyn coordinode_core::graph::stats::StorageStats);

        let cost = coordinode_query::planner::estimate_cost_with_stats(&plan, stats_ref);

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

    async fn commit_transaction(
        &self,
        request: Request<query::CommitTransactionRequest>,
    ) -> Result<Response<query::CommitTransactionResponse>, Status> {
        let deadline = grpc_deadline(request.metadata());
        let request = request.into_inner();
        let transaction =
            super::error_details::transaction_id("transaction_id", &request.transaction_id)?;
        let operation =
            super::operation::optional_descriptor("operation", request.operation.as_ref())?;
        // The conditions the caller built its statements on, checked by the
        // commit itself. A condition naming a node the caller never read is
        // still a condition: the engine decides it against the state, not
        // against what the caller happens to know.
        let expect: Vec<ExpectedVersion> = request
            .expect
            .iter()
            .map(|e| ExpectedVersion {
                node: coordinode_core::graph::node::NodeId::from_raw(e.node_id),
                version: e.version,
            })
            .collect();
        let receipt = match self.executor.transaction_leader()? {
            Some(leader) => {
                self.executor
                    .router(leader)?
                    .commit(RoutedCommit {
                        actor: ActingContext::Trusted,
                        transaction,
                        last_nonce: None,
                        operation,
                        deadline: time_left(deadline),
                        expect,
                    })
                    .await?
            }
            None => super::blocking(|| {
                self.database().read().commit_transaction_with(
                    transaction,
                    &CommitOptions {
                        operation: operation.as_ref(),
                        expect: &expect,
                        ..CommitOptions::default()
                    },
                )
            })
            .map_err(db_error_to_status)?,
        };
        // `applied_index` 0 = no Raft log (embedded / single node); `commit_ts`
        // is present in every mode.
        Ok(Response::new(query::CommitTransactionResponse {
            applied_index: receipt.applied_index.unwrap_or(0),
            commit_ts: receipt.commit_ts.as_raw(),
        }))
    }

    async fn rollback_transaction(
        &self,
        request: Request<query::RollbackTransactionRequest>,
    ) -> Result<Response<query::RollbackTransactionResponse>, Status> {
        let transaction = super::error_details::transaction_id(
            "transaction_id",
            &request.into_inner().transaction_id,
        )?;
        match self.executor.transaction_leader()? {
            Some(leader) => {
                self.executor
                    .router(leader)?
                    .rollback(RoutedRollback {
                        actor: ActingContext::Trusted,
                        transaction,
                    })
                    .await?
            }
            None => super::blocking(|| self.database().read().rollback_transaction(transaction))
                .map_err(db_error_to_status)?,
        }
        Ok(Response::new(query::RollbackTransactionResponse {}))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
