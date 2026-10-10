//! The CE built-in procedures.

use std::sync::Arc;

use coordinode_core::graph::types::Value;

use super::{Procedure, ProcedureMode, ProcedureSignature, ValueType};
use crate::executor::runner::{ExecutionContext, ExecutionError};

/// Every built-in procedure.
pub(super) fn procedures() -> Vec<Arc<dyn Procedure>> {
    let mut all: Vec<Arc<dyn Procedure>> = vec![
        Arc::new(ListProcedures::new()),
        Arc::new(ListFunctions::new()),
    ];
    all.extend(crate::advisor::procedures::procedures());
    all.extend(super::index_builds::procedures());
    all.extend(super::index_checks::procedures());
    all
}

/// `dbms.procedures()`: the procedures this database answers `CALL` with.
struct ListProcedures {
    signature: ProcedureSignature,
}

impl ListProcedures {
    fn new() -> Self {
        Self {
            signature: ProcedureSignature::new(
                "dbms.procedures",
                ProcedureMode::Dbms,
                "Every procedure CALL can run, with its signature, in name order.",
            )
            .output("name", ValueType::String)
            .output("signature", ValueType::String)
            .output("description", ValueType::String)
            .output("mode", ValueType::String),
        }
    }
}

impl Procedure for ListProcedures {
    fn signature(&self) -> &ProcedureSignature {
        &self.signature
    }

    fn call(
        &self,
        ctx: &mut ExecutionContext<'_>,
        _args: Vec<Value>,
    ) -> Result<Vec<Vec<Value>>, ExecutionError> {
        let registry = ctx.procedures.ok_or_else(|| {
            ExecutionError::Unsupported("no procedure catalog is available here".into())
        })?;
        Ok(registry
            .iter()
            .map(|p| {
                let signature = p.signature();
                vec![
                    Value::String(signature.name.clone()),
                    Value::String(signature.to_string()),
                    Value::String(signature.description.clone()),
                    Value::String(signature.mode.as_str().to_string()),
                ]
            })
            .collect())
    }
}

/// `dbms.functions()`: the functions expressions can call.
struct ListFunctions {
    signature: ProcedureSignature,
}

impl ListFunctions {
    fn new() -> Self {
        Self {
            signature: ProcedureSignature::new(
                "dbms.functions",
                ProcedureMode::Dbms,
                "Every function an expression can call, with its signature, in name order.",
            )
            .output("name", ValueType::String)
            .output("signature", ValueType::String)
            .output("category", ValueType::String)
            .output("description", ValueType::String)
            .output("aggregating", ValueType::Boolean),
        }
    }
}

impl Procedure for ListFunctions {
    fn signature(&self) -> &ProcedureSignature {
        &self.signature
    }

    fn call(
        &self,
        _ctx: &mut ExecutionContext<'_>,
        _args: Vec<Value>,
    ) -> Result<Vec<Vec<Value>>, ExecutionError> {
        Ok(crate::function::catalog()
            .into_iter()
            .map(|f| {
                vec![
                    Value::String(f.name.to_string()),
                    Value::String(f.signature()),
                    Value::String(f.category.to_string()),
                    Value::String(f.description.to_string()),
                    Value::Bool(f.aggregating),
                ]
            })
            .collect())
    }
}
