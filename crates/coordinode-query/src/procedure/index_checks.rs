//! Index checks and rebuilds through `CALL`: `db.checkIndex(index)` starts a
//! check of a B-tree index against its records, `db.indexChecks()` lists
//! every check, `db.indexCheck(operation, waitMs)` inspects one after an
//! optional bounded wait, `db.cancelIndexCheck(operation)` cancels one, and
//! `db.reindex(index, waitMs)` rebuilds an index into a fresh generation.
//!
//! An index is named by its identity (an integer) or its name (a string);
//! a string that looks like a number is a name. A check's operation is the
//! generation it checks; a rebuild's is the generation its build fills, which
//! `db.indexBuild` inspects.

use core::time::Duration;
use std::sync::Arc;

use coordinode_core::graph::types::Value;

use super::{
    FieldSignature, Procedure, ProcedureError, ProcedureMode, ProcedureSignature, ValueType,
};
use crate::executor::runner::{ExecutionContext, ExecutionError, operation_value};
use crate::index::{
    CheckOutcome, CheckPhase, CheckState, CheckStatus, GenerationId, IndexId, IndexSelector,
    Integrity, Mismatch,
};

/// The index check procedures, for the built-in catalog.
pub(super) fn procedures() -> Vec<Arc<dyn Procedure>> {
    vec![
        Arc::new(IndexCheckProcedure::new(Kind::Start)),
        Arc::new(IndexCheckProcedure::new(Kind::List)),
        Arc::new(IndexCheckProcedure::new(Kind::Inspect)),
        Arc::new(IndexCheckProcedure::new(Kind::Cancel)),
        Arc::new(IndexCheckProcedure::new(Kind::Reindex)),
    ]
}

/// Which procedure.
#[derive(Clone, Copy)]
enum Kind {
    Start,
    List,
    Inspect,
    Cancel,
    Reindex,
}

struct IndexCheckProcedure {
    kind: Kind,
    signature: ProcedureSignature,
}

/// The columns describing one check, in the order [`status_row`] fills
/// them.
fn with_status_columns(signature: ProcedureSignature) -> ProcedureSignature {
    signature
        .output("operation", ValueType::Integer)
        .output("indexId", ValueType::Integer)
        .output("integrity", ValueType::String)
        .output("state", ValueType::String)
        .output("phase", ValueType::String)
        .output("passes", ValueType::Integer)
        .output("checked", ValueType::Integer)
        .output("mismatches", ValueType::Integer)
        .output("repaired", ValueType::Integer)
        .output("conflicts", ValueType::Integer)
        .output("rebuiltInto", ValueType::Integer)
        .output("failure", ValueType::String)
}

impl IndexCheckProcedure {
    fn new(kind: Kind) -> Self {
        let index = || FieldSignature::new("index", ValueType::Any);
        let wait = || FieldSignature::new("waitMs", ValueType::Integer).with_default(Value::Int(0));
        let signature = match kind {
            Kind::Start => with_status_columns(
                ProcedureSignature::new(
                    "db.checkIndex",
                    ProcedureMode::Schema,
                    "Check a B-tree index against its records in the background, repairing the \
                     entries that disagree; index is its id or its name.",
                )
                .input(index()),
            ),
            Kind::List => with_status_columns(ProcedureSignature::new(
                "db.indexChecks",
                ProcedureMode::Read,
                "Every index generation's integrity and its latest check.",
            )),
            Kind::Inspect => with_status_columns(
                ProcedureSignature::new(
                    "db.indexCheck",
                    ProcedureMode::Read,
                    "One index check, after waiting up to waitMs for its outcome; waiting \
                     cancels nothing.",
                )
                .input(FieldSignature::new("operation", ValueType::Integer))
                .input(wait()),
            ),
            Kind::Cancel => with_status_columns(
                ProcedureSignature::new(
                    "db.cancelIndexCheck",
                    ProcedureMode::Schema,
                    "Cancel an index check; repairs it committed stay.",
                )
                .input(FieldSignature::new("operation", ValueType::Integer)),
            ),
            Kind::Reindex => ProcedureSignature::new(
                "db.reindex",
                ProcedureMode::Schema,
                "Rebuild a B-tree index from its records into a fresh generation, waiting up to \
                 waitMs for the build; index is its id or its name.",
            )
            .input(index())
            .input(wait())
            .output("operation", ValueType::Integer)
            .output("state", ValueType::String),
        };
        Self { kind, signature }
    }
}

impl Procedure for IndexCheckProcedure {
    fn signature(&self) -> &ProcedureSignature {
        &self.signature
    }

    fn call(
        &self,
        ctx: &mut ExecutionContext<'_>,
        args: Vec<Value>,
    ) -> Result<Vec<Vec<Value>>, ExecutionError> {
        let name = &self.signature.name;
        let builds = ctx.index_builds.ok_or_else(|| {
            ExecutionError::Unsupported("index maintenance is not available here".into())
        })?;
        match self.kind {
            Kind::Start => {
                let def = selector_arg(name, args.first())?
                    .resolve(ctx.engine)
                    .map_err(|e| invalid(name, "index", e.to_string()))?;
                let operation = builds
                    .request_check(def.id)
                    .map_err(|e| ExecutionError::Unsupported(e.to_string()))?;
                Ok(vec![status_row(&status_of(builds, name, operation)?, None)])
            }
            Kind::List => Ok(builds
                .checks()?
                .iter()
                .map(|status| status_row(status, None))
                .collect()),
            Kind::Inspect => {
                let operation = operation_arg(name, args.first())?;
                let wait = wait_arg(name, args.get(1))?;
                let outcome = builds.wait_check(operation, Some(wait))?;
                Ok(vec![status_row(
                    &status_of(builds, name, operation)?,
                    outcome.as_ref(),
                )])
            }
            Kind::Cancel => {
                let operation = operation_arg(name, args.first())?;
                builds
                    .cancel_check(operation)
                    .map_err(ExecutionError::Unsupported)?;
                Ok(vec![status_row(&status_of(builds, name, operation)?, None)])
            }
            Kind::Reindex => {
                let def = selector_arg(name, args.first())?
                    .resolve(ctx.engine)
                    .map_err(|e| invalid(name, "index", e.to_string()))?;
                if def.index_type != crate::index::IndexType::BTree {
                    return Err(invalid(
                        name,
                        "index",
                        format!("index '{def}' is not a B-tree index"),
                    )
                    .into());
                }
                let wait = wait_arg(name, args.get(1))?;
                let def = builds.rebuild(def).map_err(ExecutionError::Unsupported)?;
                // The rebuild waits for transactions older than it; this
                // statement staged nothing, so it leaves that wait rather
                // than wait for the build waiting for it.
                if !wait.is_zero() {
                    ctx.txn.release_from_schema_waits();
                }
                builds.wait(def.generation, Some(wait))?;
                let state = builds
                    .status(def.generation, Duration::ZERO)?
                    .and_then(|s| s.record)
                    .map_or("ACCEPTED", |r| match r.state {
                        crate::index::BuildState::Accepted => "ACCEPTED",
                        crate::index::BuildState::Running { .. } => "RUNNING",
                        crate::index::BuildState::Published => "PUBLISHED",
                        crate::index::BuildState::Failed { .. } => "FAILED",
                        crate::index::BuildState::Cancelled => "CANCELLED",
                    });
                Ok(vec![vec![
                    Value::Int(operation_value(def.generation)),
                    Value::String(state.into()),
                ]])
            }
        }
    }
}

/// The check `operation` as inspection shows it.
fn status_of(
    builds: &crate::index::IndexBuildService,
    procedure: &str,
    operation: GenerationId,
) -> Result<CheckStatus, ExecutionError> {
    builds
        .checks()?
        .into_iter()
        .find(|s| s.generation == operation)
        .ok_or_else(|| {
            invalid(
                procedure,
                "operation",
                format!("no index check has operation {}", operation.as_raw()),
            )
            .into()
        })
}

/// One check as the status columns show it; `outcome` fills the failure of
/// a check that ended without its record saying why.
fn status_row(status: &CheckStatus, outcome: Option<&CheckOutcome>) -> Vec<Value> {
    let record = &status.record;
    let check = record.check.as_ref();
    let int = |n: u64| Value::Int(i64::try_from(n).unwrap_or(i64::MAX));
    let conflicts = record
        .evidence
        .iter()
        .filter(|m| matches!(m, Mismatch::SourceDuplicate { .. }))
        .count();
    vec![
        Value::Int(operation_value(status.generation)),
        Value::Int(index_value(record.index)),
        Value::String(
            match record.integrity {
                Integrity::Unchecked => "UNCHECKED",
                Integrity::Suspect => "SUSPECT",
                Integrity::Verified => "VERIFIED",
            }
            .into(),
        ),
        check.map_or(Value::Null, |c| {
            Value::String(
                match c.state {
                    CheckState::Accepted => "ACCEPTED",
                    CheckState::Running { .. } => "RUNNING",
                    CheckState::Done => "DONE",
                    CheckState::Failed { .. } => "FAILED",
                    CheckState::Cancelled => "CANCELLED",
                }
                .into(),
            )
        }),
        check.map_or(Value::Null, |c| {
            Value::String(
                match c.phase {
                    CheckPhase::Records => "RECORDS",
                    CheckPhase::Entries => "ENTRIES",
                }
                .into(),
            )
        }),
        check.map_or(Value::Null, |c| int(u64::from(c.passes))),
        check.map_or(Value::Null, |c| int(c.checked)),
        check.map_or(Value::Null, |c| int(c.mismatches)),
        check.map_or(Value::Null, |c| int(c.repaired)),
        int(conflicts as u64),
        check
            .and_then(|c| c.rebuilt_into)
            .map_or(Value::Null, |g| Value::Int(operation_value(g))),
        match (check.map(|c| &c.state), outcome) {
            (Some(CheckState::Failed { reason }), _) => Value::String(reason.clone()),
            (_, Some(CheckOutcome::Failed(reason))) => Value::String(reason.clone()),
            _ => Value::Null,
        },
    ]
}

/// An index identity as a query value.
fn index_value(id: IndexId) -> i64 {
    i64::try_from(id.as_raw()).unwrap_or(i64::MAX)
}

/// The `index` argument: an integer names an identity, a string a name.
fn selector_arg(procedure: &str, arg: Option<&Value>) -> Result<IndexSelector, ProcedureError> {
    match arg {
        Some(Value::Int(n)) => u64::try_from(*n)
            .map(|raw| IndexSelector::Id(IndexId::from_raw(raw)))
            .map_err(|_| invalid(procedure, "index", format!("{n} is not an index id"))),
        Some(Value::String(name)) => Ok(IndexSelector::Name(name.clone())),
        other => Err(invalid(
            procedure,
            "index",
            format!("an index id or name is required, got {other:?}"),
        )),
    }
}

/// The `operation` argument as the generation it names.
fn operation_arg(procedure: &str, arg: Option<&Value>) -> Result<GenerationId, ProcedureError> {
    match arg {
        Some(Value::Int(n)) => u64::try_from(*n)
            .map(GenerationId::from_raw)
            .map_err(|_| invalid(procedure, "operation", format!("{n} is not an operation"))),
        _ => Err(invalid(
            procedure,
            "operation",
            "an operation is required, got null".into(),
        )),
    }
}

/// The `waitMs` argument as a duration; null waits not at all.
fn wait_arg(procedure: &str, arg: Option<&Value>) -> Result<Duration, ProcedureError> {
    match arg {
        Some(Value::Int(ms)) => u64::try_from(*ms)
            .map(Duration::from_millis)
            .map_err(|_| invalid(procedure, "waitMs", format!("{ms} is negative"))),
        _ => Ok(Duration::ZERO),
    }
}

fn invalid(procedure: &str, argument: &str, reason: String) -> ProcedureError {
    ProcedureError::InvalidArgument {
        procedure: procedure.into(),
        argument: argument.into(),
        reason,
    }
}
