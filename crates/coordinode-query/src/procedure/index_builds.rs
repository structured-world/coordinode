//! Index build operations through `CALL`: `db.indexBuilds()` lists them,
//! `db.indexBuild(operation, waitMs)` inspects one after an optional bounded
//! wait, `db.cancelIndexBuild(operation)` cancels one.
//!
//! An operation is the generation a build fills, as `CREATE INDEX` and
//! `CREATE CONSTRAINT` return it in their `operation` column. Waiting cancels
//! nothing, and a build never belongs to the statement that admitted it.

use core::time::Duration;
use std::sync::Arc;

use coordinode_core::graph::types::Value;

use super::{
    FieldSignature, Procedure, ProcedureError, ProcedureMode, ProcedureSignature, ValueType,
};
use crate::executor::runner::{ExecutionContext, ExecutionError, operation_value};
use crate::index::{BuildPhase, BuildState, BuildStatus, GenerationId, IndexBuildService};

/// The index build procedures, for the built-in catalog.
pub(super) fn procedures() -> Vec<Arc<dyn Procedure>> {
    vec![
        Arc::new(IndexBuildProcedure::new(Kind::List)),
        Arc::new(IndexBuildProcedure::new(Kind::Inspect)),
        Arc::new(IndexBuildProcedure::new(Kind::Cancel)),
    ]
}

/// Which index build procedure.
#[derive(Clone, Copy)]
enum Kind {
    List,
    Inspect,
    Cancel,
}

struct IndexBuildProcedure {
    kind: Kind,
    signature: ProcedureSignature,
}

/// The columns describing one build, in the order [`status_row`] fills them.
fn with_status_columns(signature: ProcedureSignature) -> ProcedureSignature {
    signature
        .output("operation", ValueType::Integer)
        .output("index", ValueType::String)
        .output("label", ValueType::String)
        .output("state", ValueType::String)
        .output("phase", ValueType::String)
        .output("indexed", ValueType::Integer)
        .output("failure", ValueType::String)
}

impl IndexBuildProcedure {
    fn new(kind: Kind) -> Self {
        let signature = match kind {
            Kind::List => with_status_columns(ProcedureSignature::new(
                "db.indexBuilds",
                ProcedureMode::Read,
                "Every index build with its operation, state and progress.",
            )),
            Kind::Inspect => with_status_columns(
                ProcedureSignature::new(
                    "db.indexBuild",
                    ProcedureMode::Read,
                    "One index build, after waiting up to waitMs for its outcome; waiting \
                     cancels nothing.",
                )
                .input(FieldSignature::new("operation", ValueType::Integer))
                .input(
                    FieldSignature::new("waitMs", ValueType::Integer).with_default(Value::Int(0)),
                ),
            ),
            Kind::Cancel => ProcedureSignature::new(
                "db.cancelIndexBuild",
                ProcedureMode::Schema,
                "Cancel an index build: a new index is withdrawn with the constraint owning \
                 it, a rebuilt one kept failed. A build that already has an outcome keeps it.",
            )
            .input(FieldSignature::new("operation", ValueType::Integer))
            .output("operation", ValueType::Integer)
            .output("cancelled", ValueType::Boolean)
            .output("state", ValueType::String),
        };
        Self { kind, signature }
    }
}

impl Procedure for IndexBuildProcedure {
    fn signature(&self) -> &ProcedureSignature {
        &self.signature
    }

    fn call(
        &self,
        ctx: &mut ExecutionContext<'_>,
        args: Vec<Value>,
    ) -> Result<Vec<Vec<Value>>, ExecutionError> {
        let builds = ctx.index_builds.ok_or_else(|| {
            ExecutionError::Unsupported("index builds are not available here".into())
        })?;
        match self.kind {
            Kind::List => Ok(builds.builds()?.iter().map(status_row).collect()),
            Kind::Inspect => {
                let operation = operation_arg(&self.signature.name, &args)?;
                let wait = wait_arg(&self.signature.name, args.get(1))?;
                let status = builds
                    .status(operation, wait)?
                    .ok_or_else(|| unknown(&self.signature.name, operation))?;
                Ok(vec![status_row(&status)])
            }
            Kind::Cancel => {
                let operation = operation_arg(&self.signature.name, &args)?;
                cancel(ctx, builds, &self.signature.name, operation)
            }
        }
    }
}

/// Cancel build `operation`: a key-shaped build through its record, a
/// vector build this member runs by stopping it.
fn cancel(
    ctx: &ExecutionContext<'_>,
    builds: &IndexBuildService,
    procedure: &str,
    operation: GenerationId,
) -> Result<Vec<Vec<Value>>, ExecutionError> {
    let mut cancelled = builds
        .cancel(operation)
        .map_err(ExecutionError::Unsupported)?;
    if !cancelled {
        cancelled = ctx
            .vector_index_registry()
            .is_some_and(|r| r.cancel_build(operation));
    }
    let state = match builds.status(operation, Duration::ZERO)? {
        Some(status) => state_value(&status),
        // A vector build keeps no record: stopped here, it has no state left.
        None if cancelled => Value::String("CANCELLED".into()),
        None => return Err(unknown(procedure, operation)),
    };
    Ok(vec![vec![
        Value::Int(operation_value(operation)),
        Value::Bool(cancelled),
        state,
    ]])
}

/// One build as the status columns show it.
fn status_row(status: &BuildStatus) -> Vec<Value> {
    let index = status.index.as_ref();
    vec![
        Value::Int(operation_value(status.generation)),
        index
            .and_then(|i| i.name.clone())
            .map_or(Value::Null, Value::String),
        index.map_or(Value::Null, |i| Value::String(i.label.clone())),
        state_value(status),
        status.phase.map_or(Value::Null, |p| {
            Value::String(
                match p {
                    BuildPhase::AwaitingSeat => "AWAITING_SEAT",
                    BuildPhase::AwaitingOlderTransactions => "AWAITING_OLDER_TRANSACTIONS",
                    BuildPhase::Indexing { .. } => "INDEXING",
                }
                .into(),
            )
        }),
        status
            .indexed()
            .map_or(Value::Null, |n| Value::Int(count(n))),
        status
            .failure()
            .map_or(Value::Null, |f| Value::String(f.to_string())),
    ]
}

/// The `state` column: the record's state, or `RUNNING` for a build this
/// member runs for itself, which keeps no record.
fn state_value(status: &BuildStatus) -> Value {
    Value::String(
        match status.record.as_ref().map(|r| &r.state) {
            Some(BuildState::Accepted) => "ACCEPTED",
            Some(BuildState::Running { .. }) | None => "RUNNING",
            Some(BuildState::Published) => "PUBLISHED",
            Some(BuildState::Failed { .. }) => "FAILED",
            Some(BuildState::Cancelled) => "CANCELLED",
        }
        .into(),
    )
}

/// A count as a query value.
fn count(n: u64) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}

/// The `operation` argument as the generation it names.
fn operation_arg(procedure: &str, args: &[Value]) -> Result<GenerationId, ProcedureError> {
    match args.first() {
        Some(Value::Int(n)) => u64::try_from(*n)
            .ok()
            .map(GenerationId::from_raw)
            .ok_or_else(|| ProcedureError::InvalidArgument {
                procedure: procedure.into(),
                argument: "operation".into(),
                reason: format!("{n} is not an index build operation"),
            }),
        _ => Err(ProcedureError::InvalidArgument {
            procedure: procedure.into(),
            argument: "operation".into(),
            reason: "an operation is required, got null".into(),
        }),
    }
}

/// The `waitMs` argument as a duration; null waits not at all.
fn wait_arg(procedure: &str, arg: Option<&Value>) -> Result<Duration, ProcedureError> {
    match arg {
        Some(Value::Int(ms)) => u64::try_from(*ms).map(Duration::from_millis).map_err(|_| {
            ProcedureError::InvalidArgument {
                procedure: procedure.into(),
                argument: "waitMs".into(),
                reason: format!("{ms} is negative"),
            }
        }),
        _ => Ok(Duration::ZERO),
    }
}

/// No build has `operation`.
fn unknown(procedure: &str, operation: GenerationId) -> ExecutionError {
    ProcedureError::InvalidArgument {
        procedure: procedure.into(),
        argument: "operation".into(),
        reason: format!("no index build has operation {}", operation.as_raw()),
    }
    .into()
}
