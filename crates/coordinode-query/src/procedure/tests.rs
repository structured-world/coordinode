use super::*;

fn slow_queries() -> ProcedureSignature {
    ProcedureSignature::new("db.test.slow", ProcedureMode::Read, "test")
        .input(FieldSignature::new("limit", ValueType::Integer).with_default(Value::Int(20)))
        .input(FieldSignature::new("ratio", ValueType::Float).with_default(Value::Float(0.5)))
        .output("query", ValueType::String)
        .output("count", ValueType::Integer)
}

fn required() -> ProcedureSignature {
    ProcedureSignature::new("db.test.required", ProcedureMode::Write, "test").input(
        FieldSignature::new("ids", ValueType::List(Box::new(ValueType::Integer))),
    )
}

/// The listed signature spells inputs with defaults, types and outputs.
#[test]
fn signature_renders_inputs_defaults_and_outputs() {
    assert_eq!(
        slow_queries().to_string(),
        "db.test.slow(limit = 20 :: INTEGER, ratio = 0.5 :: FLOAT) :: \
         (query :: STRING, count :: INTEGER)"
    );
    assert_eq!(
        required().to_string(),
        "db.test.required(ids :: LIST<INTEGER>) :: VOID"
    );
}

/// Omitted trailing arguments take their defaults.
#[test]
fn missing_arguments_with_defaults_are_filled() {
    let bound = bind_arguments(&slow_queries(), vec![Value::Int(5)]).unwrap();
    assert_eq!(bound, vec![Value::Int(5), Value::Float(0.5)]);
}

/// An integer passed for a FLOAT input is widened.
#[test]
fn integers_widen_to_float_inputs() {
    let bound = bind_arguments(&slow_queries(), vec![Value::Int(5), Value::Int(1)]).unwrap();
    assert_eq!(bound, vec![Value::Int(5), Value::Float(1.0)]);
}

/// A surplus argument is refused, with the counts.
#[test]
fn surplus_arguments_are_refused() {
    let err = bind_arguments(
        &slow_queries(),
        vec![Value::Int(1), Value::Int(2), Value::Int(3)],
    )
    .unwrap_err();
    assert_eq!(
        err,
        ProcedureError::TooManyArguments {
            procedure: "db.test.slow".into(),
            declared: 2,
            given: 3,
        }
    );
}

/// A required argument left out is refused by name.
#[test]
fn missing_required_argument_is_refused() {
    let err = bind_arguments(&required(), Vec::new()).unwrap_err();
    assert!(matches!(
        err,
        ProcedureError::MissingArgument { ref argument, .. } if argument == "ids"
    ));
}

/// A value of the wrong type is refused, naming both types; inside a list
/// every element is checked.
#[test]
fn wrong_types_are_refused() {
    let err = bind_arguments(&slow_queries(), vec![Value::String("ten".into())]).unwrap_err();
    assert_eq!(
        err,
        ProcedureError::ArgumentType {
            procedure: "db.test.slow".into(),
            argument: "limit".into(),
            expected: "INTEGER".into(),
            found: "STRING".into(),
        }
    );
    let err = bind_arguments(
        &required(),
        vec![Value::Array(vec![Value::Int(1), Value::String("x".into())])],
    )
    .unwrap_err();
    assert!(
        matches!(err, ProcedureError::ArgumentType { ref expected, .. } if expected == "LIST<INTEGER>")
    );
}

/// NULL fits any declared type; the procedure decides what it means.
#[test]
fn null_fits_any_type() {
    let bound = bind_arguments(&slow_queries(), vec![Value::Null, Value::Null]).unwrap();
    assert_eq!(bound, vec![Value::Null, Value::Null]);
}

/// YIELD resolves columns to positions and binds aliases.
#[test]
fn yields_resolve_columns_and_aliases() {
    let yields = [
        YieldColumn {
            column: "count".into(),
            variable: "c".into(),
        },
        YieldColumn {
            column: "query".into(),
            variable: "query".into(),
        },
    ];
    let bound = bind_yields(&slow_queries(), Some(&yields), false).unwrap();
    assert_eq!(bound, vec![(1, "c".into()), (0, "query".into())]);
}

/// A yielded column the procedure lacks is refused.
#[test]
fn unknown_yield_is_refused() {
    let yields = [YieldColumn {
        column: "nosuch".into(),
        variable: "nosuch".into(),
    }];
    let err = bind_yields(&slow_queries(), Some(&yields), true).unwrap_err();
    assert!(matches!(err, ProcedureError::UnknownOutput { ref column, .. } if column == "nosuch"));
}

/// Without YIELD a standalone call binds every output; inside a query it
/// must name them, unless the procedure has none.
#[test]
fn absent_yield_depends_on_position_and_outputs() {
    let all = bind_yields(&slow_queries(), None, true).unwrap();
    assert_eq!(all, vec![(0, "query".into()), (1, "count".into())]);
    assert!(matches!(
        bind_yields(&slow_queries(), None, false),
        Err(ProcedureError::YieldRequired { .. })
    ));
    assert!(bind_yields(&required(), None, false).unwrap().is_empty());
}

/// Only WRITE and SCHEMA procedures make a statement a write.
#[test]
fn modes_that_write() {
    assert!(!ProcedureMode::Read.writes());
    assert!(!ProcedureMode::Dbms.writes());
    assert!(ProcedureMode::Write.writes());
    assert!(ProcedureMode::Schema.writes());
}

/// The built-in catalog lists the listings and the advisor, each once.
#[test]
fn builtins_are_registered_once_each() {
    let registry = ProcedureRegistry::with_builtins();
    let names: Vec<&str> = registry
        .iter()
        .map(|p| p.signature().name.as_str())
        .collect();
    assert_eq!(
        names.len(),
        builtin::procedures().len(),
        "a built-in name clashed"
    );
    for name in [
        "dbms.procedures",
        "dbms.functions",
        "db.advisor.suggestions",
        "db.advisor.queryStats",
        "db.advisor.slowQueries",
        "db.advisor.dismiss",
        "db.advisor.reset",
    ] {
        assert!(names.contains(&name), "{name} missing");
    }
    assert!(names.windows(2).all(|w| w[0] < w[1]), "name order");
}

struct Named(ProcedureSignature);

impl Procedure for Named {
    fn signature(&self) -> &ProcedureSignature {
        &self.0
    }

    fn call(
        &self,
        _ctx: &mut ExecutionContext<'_>,
        _args: Vec<Value>,
    ) -> Result<Vec<Vec<Value>>, ExecutionError> {
        Ok(Vec::new())
    }
}

/// A registration under a taken name is refused and keeps the original.
#[test]
fn duplicate_registration_is_refused() {
    let mut registry = ProcedureRegistry::with_builtins();
    let clash = Named(ProcedureSignature::new(
        "dbms.procedures",
        ProcedureMode::Write,
        "impostor",
    ));
    assert_eq!(
        registry.register(Arc::new(clash)),
        Err(ProcedureError::DuplicateName {
            procedure: "dbms.procedures".into()
        })
    );
    assert_eq!(
        registry.get("dbms.procedures").unwrap().signature().mode,
        ProcedureMode::Dbms
    );

    registry.register(Arc::new(Named(required()))).unwrap();
    assert!(registry.writes("db.test.required"));
    assert!(!registry.writes("dbms.procedures"));
    assert!(!registry.writes("db.nosuch"));
}
