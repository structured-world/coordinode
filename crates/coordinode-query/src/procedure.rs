//! The procedure catalog `CALL` dispatches to.
//!
//! A procedure declares its signature: typed inputs with optional defaults,
//! typed outputs, and the mode that says what it touches. A call is checked
//! against the signature before the procedure runs, and the signatures are
//! what `dbms.procedures()` lists, so a listed procedure is one `CALL`
//! reaches and a reachable one is listed.
//!
//! The CE catalog is [`ProcedureRegistry::with_builtins`]. An enterprise
//! layer or an embedder adds its own procedures with
//! [`ProcedureRegistry::register`] at startup; there is no loading at run
//! time.

mod builtin;
mod index_builds;

use core::fmt;
use std::collections::BTreeMap;
use std::sync::Arc;

use coordinode_core::graph::types::Value;

use crate::executor::runner::{ExecutionContext, ExecutionError};
use crate::planner::logical::YieldColumn;

/// What a procedure touches, which decides where it may run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcedureMode {
    /// Reads graph data only.
    Read,
    /// Reads and writes graph data.
    Write,
    /// Changes the schema (indexes, constraints).
    Schema,
    /// Reads or changes server state rather than graph data.
    Dbms,
}

impl ProcedureMode {
    /// The mode's name as listings show it.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Read => "READ",
            Self::Write => "WRITE",
            Self::Schema => "SCHEMA",
            Self::Dbms => "DBMS",
        }
    }

    /// Whether a call in this mode changes the database, so a statement
    /// containing it is a write.
    pub fn writes(self) -> bool {
        matches!(self, Self::Write | Self::Schema)
    }
}

impl fmt::Display for ProcedureMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The declared type of a procedure input or output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueType {
    /// Any value.
    Any,
    /// `true` / `false`.
    Boolean,
    /// A 64-bit integer.
    Integer,
    /// A 64-bit float; an integer argument is widened.
    Float,
    /// An integer or a float.
    Number,
    /// A string.
    String,
    /// A map.
    Map,
    /// A list whose elements have the inner type.
    List(Box<ValueType>),
    /// A node, carried as its id.
    Node,
    /// A path.
    Path,
}

impl ValueType {
    /// Convert `value` to this type, or give it back when it does not fit.
    /// NULL fits every type; a procedure decides what a NULL argument means.
    fn coerce(&self, value: Value) -> Result<Value, Value> {
        match (self, value) {
            (_, Value::Null) => Ok(Value::Null),
            (Self::Any, v) => Ok(v),
            (Self::Boolean, v @ Value::Bool(_)) => Ok(v),
            (Self::Integer | Self::Node, v @ Value::Int(_)) => Ok(v),
            (Self::Float, Value::Int(n)) => Ok(Value::Float(n as f64)),
            (Self::Float, v @ Value::Float(_)) => Ok(v),
            (Self::Number, v @ (Value::Int(_) | Value::Float(_))) => Ok(v),
            (Self::String, v @ Value::String(_)) => Ok(v),
            (Self::Map, v @ Value::Map(_)) => Ok(v),
            (Self::Path, v @ Value::Path(_)) => Ok(v),
            (Self::List(inner), Value::Array(items)) => {
                let mut coerced = Vec::with_capacity(items.len());
                let mut items = items.into_iter();
                while let Some(item) = items.next() {
                    match inner.coerce(item) {
                        Ok(v) => coerced.push(v),
                        Err(bad) => {
                            // Hand the list back whole, so the caller can
                            // name what it was given.
                            coerced.push(bad);
                            coerced.extend(items);
                            return Err(Value::Array(coerced));
                        }
                    }
                }
                Ok(Value::Array(coerced))
            }
            (_, v) => Err(v),
        }
    }
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Any => f.write_str("ANY"),
            Self::Boolean => f.write_str("BOOLEAN"),
            Self::Integer => f.write_str("INTEGER"),
            Self::Float => f.write_str("FLOAT"),
            Self::Number => f.write_str("NUMBER"),
            Self::String => f.write_str("STRING"),
            Self::Map => f.write_str("MAP"),
            Self::List(inner) => write!(f, "LIST<{inner}>"),
            Self::Node => f.write_str("NODE"),
            Self::Path => f.write_str("PATH"),
        }
    }
}

/// One input or output of a procedure.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldSignature {
    /// The field's name: the argument name, or the output column.
    pub name: String,
    /// Its declared type.
    pub ty: ValueType,
    /// For an input, the value a call that omits it gets; `None` makes the
    /// input required. Unused on outputs.
    pub default: Option<Value>,
}

impl FieldSignature {
    /// A required input, or an output.
    pub fn new(name: impl Into<String>, ty: ValueType) -> Self {
        Self {
            name: name.into(),
            ty,
            default: None,
        }
    }

    /// An input a call may omit, taking `default`.
    pub fn with_default(mut self, default: Value) -> Self {
        self.default = Some(default);
        self
    }
}

impl fmt::Display for FieldSignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.default {
            Some(default) => write!(f, "{} = {} :: {}", self.name, Literal(default), self.ty),
            None => write!(f, "{} :: {}", self.name, self.ty),
        }
    }
}

/// A default rendered as a Cypher literal.
struct Literal<'a>(&'a Value);

impl fmt::Display for Literal<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Value::Null => f.write_str("null"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Int(n) => write!(f, "{n}"),
            Value::Float(x) => write!(f, "{x}"),
            Value::String(s) => write!(f, "'{s}'"),
            Value::Array(items) => {
                f.write_str("[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{}", Literal(item))?;
                }
                f.write_str("]")
            }
            Value::Map(entries) => {
                f.write_str("{")?;
                for (i, (k, v)) in entries.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{k}: {}", Literal(v))?;
                }
                f.write_str("}")
            }
            other => write!(f, "{other:?}"),
        }
    }
}

/// A procedure's name, mode, description, inputs and outputs.
#[derive(Debug, Clone, PartialEq)]
pub struct ProcedureSignature {
    /// The dotted name `CALL` uses.
    pub name: String,
    /// What the procedure touches.
    pub mode: ProcedureMode,
    /// What it does, as listings show it.
    pub description: String,
    /// Inputs, in argument order.
    pub inputs: Vec<FieldSignature>,
    /// Output columns, in the order each returned row holds them.
    pub outputs: Vec<FieldSignature>,
}

impl ProcedureSignature {
    /// A signature with no inputs and no outputs yet.
    pub fn new(
        name: impl Into<String>,
        mode: ProcedureMode,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            mode,
            description: description.into(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Append an input.
    pub fn input(mut self, field: FieldSignature) -> Self {
        self.inputs.push(field);
        self
    }

    /// Append an output column.
    pub fn output(mut self, name: impl Into<String>, ty: ValueType) -> Self {
        self.outputs.push(FieldSignature::new(name, ty));
        self
    }

    /// The position of output `column`, if the procedure has it.
    pub fn output_index(&self, column: &str) -> Option<usize> {
        self.outputs.iter().position(|o| o.name == column)
    }
}

/// `name(inputs) :: (outputs)`, or `:: VOID` for a procedure without outputs.
impl fmt::Display for ProcedureSignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(", self.name)?;
        for (i, input) in self.inputs.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{input}")?;
        }
        f.write_str(") :: ")?;
        if self.outputs.is_empty() {
            return f.write_str("VOID");
        }
        f.write_str("(")?;
        for (i, output) in self.outputs.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{output}")?;
        }
        f.write_str(")")
    }
}

/// A procedure `CALL` can run.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a procedure",
    label = "does not implement `Procedure`",
    note = "implement `Procedure` with a `ProcedureSignature` and add it with `ProcedureRegistry::register`"
)]
pub trait Procedure: Send + Sync {
    /// The signature calls are checked against and listings show.
    fn signature(&self) -> &ProcedureSignature;

    /// Run the procedure. `args` match the signature's inputs one for one,
    /// defaults filled in and values coerced to the declared types. Each
    /// returned row holds one value per output, in output order.
    fn call(
        &self,
        ctx: &mut ExecutionContext<'_>,
        args: Vec<Value>,
    ) -> Result<Vec<Vec<Value>>, ExecutionError>;
}

/// A refused procedure call or registration. The call ran nothing.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ProcedureError {
    /// No procedure has the name.
    #[error("there is no procedure named `{procedure}`")]
    Unknown {
        /// The name the call used.
        procedure: String,
    },

    /// More arguments than the procedure declares.
    #[error("procedure `{procedure}` takes at most {declared} arguments, {given} given")]
    TooManyArguments {
        /// The procedure.
        procedure: String,
        /// How many inputs it declares.
        declared: usize,
        /// How many the call passed.
        given: usize,
    },

    /// A required argument the call left out.
    #[error("procedure `{procedure}` requires argument `{argument}`")]
    MissingArgument {
        /// The procedure.
        procedure: String,
        /// The missing input.
        argument: String,
    },

    /// An argument whose value does not fit the declared type.
    #[error("argument `{argument}` of procedure `{procedure}` must be {expected}, got {found}")]
    ArgumentType {
        /// The procedure.
        procedure: String,
        /// The input.
        argument: String,
        /// Its declared type.
        expected: String,
        /// The type the call passed.
        found: String,
    },

    /// An argument of the right type whose value the procedure cannot use.
    #[error("argument `{argument}` of procedure `{procedure}`: {reason}")]
    InvalidArgument {
        /// The procedure.
        procedure: String,
        /// The input.
        argument: String,
        /// Why the value is refused.
        reason: String,
    },

    /// A YIELD of a column the procedure does not produce.
    #[error("procedure `{procedure}` has no output `{column}`")]
    UnknownOutput {
        /// The procedure.
        procedure: String,
        /// The column the call yielded.
        column: String,
    },

    /// A CALL inside a larger query that names none of the procedure's
    /// outputs: the clauses around it could not tell what it binds.
    #[error("a CALL of `{procedure}` inside a query must YIELD the outputs it uses")]
    YieldRequired {
        /// The procedure.
        procedure: String,
    },

    /// A procedure returned a row of the wrong width.
    #[error(
        "procedure `{procedure}` returned {found} values in a row, its signature declares {declared}"
    )]
    OutputShape {
        /// The procedure.
        procedure: String,
        /// Its declared output count.
        declared: usize,
        /// The width of the returned row.
        found: usize,
    },

    /// A second procedure registered under a name already taken.
    #[error("a procedure named `{procedure}` is already registered")]
    DuplicateName {
        /// The name.
        procedure: String,
    },
}

/// The procedures a database answers `CALL` with, by name.
#[derive(Clone)]
pub struct ProcedureRegistry {
    procedures: BTreeMap<String, Arc<dyn Procedure>>,
}

impl ProcedureRegistry {
    /// A registry holding no procedures.
    pub fn empty() -> Self {
        Self {
            procedures: BTreeMap::new(),
        }
    }

    /// The CE catalog: the procedure listings and the query advisor.
    pub fn with_builtins() -> Self {
        // Built-in names are distinct; this module's tests hold that.
        let procedures = builtin::procedures()
            .into_iter()
            .map(|p| (p.signature().name.clone(), p))
            .collect();
        Self { procedures }
    }

    /// Add `procedure` under its signature's name. Refused when the name is
    /// taken, so a registration never silently replaces another procedure.
    pub fn register(&mut self, procedure: Arc<dyn Procedure>) -> Result<(), ProcedureError> {
        let name = procedure.signature().name.clone();
        if self.procedures.contains_key(&name) {
            return Err(ProcedureError::DuplicateName { procedure: name });
        }
        self.procedures.insert(name, procedure);
        Ok(())
    }

    /// The procedure named `name`.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Procedure>> {
        self.procedures.get(name)
    }

    /// Every procedure, in name order.
    pub fn iter(&self) -> impl Iterator<Item = &Arc<dyn Procedure>> {
        self.procedures.values()
    }

    /// Whether calling `name` changes the database. An unknown name is not a
    /// write: the call is refused before it could change anything.
    pub fn writes(&self, name: &str) -> bool {
        self.get(name).is_some_and(|p| p.signature().mode.writes())
    }
}

impl Default for ProcedureRegistry {
    fn default() -> Self {
        Self::with_builtins()
    }
}

impl fmt::Debug for ProcedureRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.procedures.keys()).finish()
    }
}

/// Match a call's arguments to the signature: refuse surplus and missing
/// ones, fill defaults, and coerce each to its declared type.
pub(crate) fn bind_arguments(
    signature: &ProcedureSignature,
    args: Vec<Value>,
) -> Result<Vec<Value>, ProcedureError> {
    if args.len() > signature.inputs.len() {
        return Err(ProcedureError::TooManyArguments {
            procedure: signature.name.clone(),
            declared: signature.inputs.len(),
            given: args.len(),
        });
    }
    let mut given = args.into_iter();
    let mut bound = Vec::with_capacity(signature.inputs.len());
    for input in &signature.inputs {
        let value = match (given.next(), &input.default) {
            (Some(value), _) => value,
            (None, Some(default)) => default.clone(),
            (None, None) => {
                return Err(ProcedureError::MissingArgument {
                    procedure: signature.name.clone(),
                    argument: input.name.clone(),
                });
            }
        };
        let value = input
            .ty
            .coerce(value)
            .map_err(|bad| ProcedureError::ArgumentType {
                procedure: signature.name.clone(),
                argument: input.name.clone(),
                expected: input.ty.to_string(),
                found: type_name(&bad).to_string(),
            })?;
        bound.push(value);
    }
    Ok(bound)
}

/// Which output columns a call binds, and under which variables: the
/// positions in each returned row paired with the variable names.
pub(crate) fn bind_yields(
    signature: &ProcedureSignature,
    yields: Option<&[YieldColumn]>,
    standalone: bool,
) -> Result<Vec<(usize, String)>, ProcedureError> {
    match yields {
        Some(columns) => columns
            .iter()
            .map(|y| {
                signature
                    .output_index(&y.column)
                    .map(|i| (i, y.variable.clone()))
                    .ok_or_else(|| ProcedureError::UnknownOutput {
                        procedure: signature.name.clone(),
                        column: y.column.clone(),
                    })
            })
            .collect(),
        None if standalone || signature.outputs.is_empty() => Ok(signature
            .outputs
            .iter()
            .enumerate()
            .map(|(i, o)| (i, o.name.clone()))
            .collect()),
        None => Err(ProcedureError::YieldRequired {
            procedure: signature.name.clone(),
        }),
    }
}

/// The Cypher type name of a value, for argument errors.
fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "NULL",
        Value::Bool(_) => "BOOLEAN",
        Value::Int(_) => "INTEGER",
        Value::Float(_) => "FLOAT",
        Value::String(_) => "STRING",
        Value::Timestamp(_) => "TIMESTAMP",
        Value::Vector(_) | Value::MultiVector(_) => "VECTOR",
        Value::Blob(_) | Value::Binary(_) => "BYTES",
        Value::Array(_) => "LIST",
        Value::Map(_) | Value::Document(_) => "MAP",
        Value::Geo(_) => "POINT",
        Value::Path(_) => "PATH",
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
