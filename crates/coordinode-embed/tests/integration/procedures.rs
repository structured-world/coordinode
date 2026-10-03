//! Integration tests: the procedure catalog, `CALL ... YIELD` and the
//! `dbms.procedures()` / `dbms.functions()` listings.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_core::graph::types::Value;
use coordinode_embed::Database;

fn string_of(row: &std::collections::BTreeMap<String, Value>, column: &str) -> String {
    match row.get(column) {
        Some(Value::String(s)) => s.clone(),
        other => panic!("column {column}: expected a string, got {other:?}"),
    }
}

/// A CALL after other clauses runs once per incoming row and keeps the row's
/// bindings: the clauses before it are part of the query, not discarded.
#[test]
fn a_call_inside_a_query_runs_once_per_incoming_row() {
    let mut db = Database::open_in_memory().expect("open");
    let rows = db
        .execute_cypher(
            "UNWIND [1, 2, 3] AS x CALL db.advisor.reset() YIELD status RETURN x, status",
        )
        .expect("call per row");
    let mut xs: Vec<i64> = rows
        .iter()
        .map(|r| match r.get("x") {
            Some(Value::Int(x)) => *x,
            other => panic!("x lost across CALL: {other:?}"),
        })
        .collect();
    xs.sort_unstable();
    assert_eq!(xs, vec![1, 2, 3]);
    assert!(rows.iter().all(|r| string_of(r, "status") == "OK"));
}

/// Arguments are evaluated against each incoming row.
#[test]
fn call_arguments_see_the_incoming_row() {
    let mut db = Database::open_in_memory().expect("open");
    let rows = db
        .execute_cypher(
            "UNWIND ['0000000000000001', '0000000000000002'] AS id \
             CALL db.advisor.dismiss(id) YIELD id AS dismissed RETURN dismissed",
        )
        .expect("call with row arguments");
    let mut ids: Vec<String> = rows.iter().map(|r| string_of(r, "dismissed")).collect();
    ids.sort();
    assert_eq!(ids, vec!["0000000000000001", "0000000000000002"]);
}

/// `YIELD column AS alias` binds the column under the alias only.
#[test]
fn yield_binds_a_column_under_its_alias() {
    let mut db = Database::open_in_memory().expect("open");
    let rows = db
        .execute_cypher("CALL db.advisor.reset() YIELD status AS s RETURN s")
        .expect("yield alias");
    assert_eq!(rows.len(), 1);
    assert_eq!(string_of(&rows[0], "s"), "OK");
}

/// A standalone CALL without YIELD returns every output column.
#[test]
fn a_standalone_call_without_yield_returns_every_output() {
    let mut db = Database::open_in_memory().expect("open");
    let rows = db
        .execute_cypher("CALL db.advisor.reset()")
        .expect("standalone call");
    assert_eq!(rows.len(), 1);
    assert_eq!(string_of(&rows[0], "status"), "OK");
}

/// Yielding a column the procedure does not produce is an error, not a
/// silently missing column.
#[test]
fn yielding_an_unknown_column_is_refused() {
    let mut db = Database::open_in_memory().expect("open");
    let err = db
        .execute_cypher("CALL db.advisor.reset() YIELD nosuch")
        .expect_err("unknown yield column");
    let msg = err.to_string();
    assert!(msg.contains("nosuch"), "error names the column: {msg}");
}

/// More arguments than the signature declares are refused.
#[test]
fn surplus_arguments_are_refused() {
    let mut db = Database::open_in_memory().expect("open");
    let err = db
        .execute_cypher("CALL db.advisor.reset(1)")
        .expect_err("surplus argument");
    assert!(
        err.to_string().contains("db.advisor.reset"),
        "error names the procedure: {err}"
    );
}

/// A required argument that is missing is refused; one with a default is
/// filled from it.
#[test]
fn missing_arguments_take_defaults_or_are_refused() {
    let mut db = Database::open_in_memory().expect("open");
    db.execute_cypher("CALL db.advisor.slowQueries()")
        .expect("every argument has a default");
    let err = db
        .execute_cypher("CALL db.advisor.dismiss()")
        .expect_err("required argument missing");
    assert!(
        err.to_string().contains("id"),
        "error names the argument: {err}"
    );
}

/// An argument of the wrong type is refused before the procedure runs.
#[test]
fn an_argument_of_the_wrong_type_is_refused() {
    let mut db = Database::open_in_memory().expect("open");
    let err = db
        .execute_cypher("CALL db.advisor.slowQueries('ten')")
        .expect_err("string where an integer is declared");
    assert!(
        err.to_string().contains("limit"),
        "error names the argument: {err}"
    );
}

/// An unknown procedure is refused with its name.
#[test]
fn an_unknown_procedure_is_refused() {
    let mut db = Database::open_in_memory().expect("open");
    let err = db
        .execute_cypher("CALL db.nonexistent()")
        .expect_err("unknown procedure");
    assert!(err.to_string().contains("db.nonexistent"), "{err}");
}

/// `dbms.procedures()` lists every registered procedure with its signature,
/// mode and description, itself included.
#[test]
fn dbms_procedures_lists_the_catalog() {
    let mut db = Database::open_in_memory().expect("open");
    let rows = db
        .execute_cypher("CALL dbms.procedures() YIELD name, signature, description, mode")
        .expect("list procedures");
    let find = |name: &str| {
        rows.iter()
            .find(|r| string_of(r, "name") == name)
            .unwrap_or_else(|| panic!("{name} not listed"))
    };
    let slow = find("db.advisor.slowQueries");
    assert_eq!(
        string_of(slow, "signature"),
        "db.advisor.slowQueries(limit = 20 :: INTEGER, minTime = 100 :: INTEGER) :: \
         (query :: STRING, p99Time :: INTEGER, count :: INTEGER, plan :: STRING, \
         sources :: LIST<STRING>)"
    );
    assert_eq!(string_of(slow, "mode"), "READ");
    assert!(!string_of(slow, "description").is_empty());
    // Resetting the advisor changes this node's state, not graph data.
    assert_eq!(string_of(find("db.advisor.reset"), "mode"), "DBMS");
    find("dbms.procedures");
    find("dbms.functions");

    let mut names: Vec<String> = rows.iter().map(|r| string_of(r, "name")).collect();
    let listed = names.clone();
    names.sort();
    assert_eq!(listed, names, "listed in name order");
}

/// `dbms.functions()` lists the built-in functions; every listed scalar
/// function is one the evaluator runs.
#[test]
fn dbms_functions_lists_functions_the_evaluator_runs() {
    let mut db = Database::open_in_memory().expect("open");
    let rows = db
        .execute_cypher(
            "CALL dbms.functions() YIELD name, signature, category, description, aggregating",
        )
        .expect("list functions");
    let find = |name: &str| {
        rows.iter()
            .find(|r| string_of(r, "name") == name)
            .unwrap_or_else(|| panic!("{name} not listed"))
    };
    let upper = find("toUpper");
    assert_eq!(
        string_of(upper, "signature"),
        "toUpper(input :: STRING) :: STRING"
    );
    assert_eq!(string_of(upper, "category"), "String");
    assert_eq!(upper.get("aggregating"), Some(&Value::Bool(false)));
    assert_eq!(find("count").get("aggregating"), Some(&Value::Bool(true)));
    find("coalesce");
    find("vector_similarity");

    // Every listed non-aggregating function resolves in the evaluator: a
    // call with NULL arguments may return NULL, never "unknown function".
    for row in &rows {
        if row.get("aggregating") == Some(&Value::Bool(true)) {
            continue;
        }
        let name = string_of(row, "name");
        let signature = string_of(row, "signature");
        let arity = signature_arity(&signature);
        let args = vec!["null"; arity].join(", ");
        let query = format!("RETURN {name}({args}) AS v");
        if let Err(e) = db.execute_cypher(&query) {
            let msg = e.to_string();
            assert!(
                !msg.to_lowercase().contains("unknown function"),
                "{name} is listed but not evaluated: {msg}"
            );
        }
    }
}

/// `YIELD ... WHERE` keeps only the output rows the predicate accepts.
#[test]
fn yield_where_filters_output_rows() {
    let mut db = Database::open_in_memory().expect("open");
    let rows = db
        .execute_cypher(
            "CALL dbms.functions() YIELD name, aggregating WHERE aggregating RETURN name",
        )
        .expect("filtered yield");
    let mut names: Vec<String> = rows.iter().map(|r| string_of(r, "name")).collect();
    names.sort();
    assert!(names.contains(&"count".to_string()));
    assert!(names.contains(&"collect".to_string()));
    assert!(!names.contains(&"toUpper".to_string()), "{names:?}");
}

/// `YIELD *` binds every output on a standalone call, and is refused inside
/// a larger query where the following clauses could not see what it binds.
#[test]
fn yield_all_is_standalone_only() {
    let mut db = Database::open_in_memory().expect("open");
    let rows = db
        .execute_cypher("CALL db.advisor.reset() YIELD *")
        .expect("standalone yield all");
    assert_eq!(string_of(&rows[0], "status"), "OK");

    let err = db
        .execute_cypher("UNWIND [1] AS x CALL db.advisor.reset() YIELD * RETURN x")
        .expect_err("yield all inside a query");
    assert!(err.to_string().contains("YIELD *"), "{err}");
}

/// A CALL inside a query that yields nothing from a procedure with outputs
/// is refused rather than binding variables nobody named.
#[test]
fn a_call_inside_a_query_must_yield() {
    let mut db = Database::open_in_memory().expect("open");
    let err = db
        .execute_cypher("UNWIND [1] AS x CALL db.advisor.reset() RETURN x")
        .expect_err("no yield inside a query");
    assert!(err.to_string().contains("YIELD"), "{err}");
}

/// A yielded variable may not shadow one already in scope.
#[test]
fn yielding_over_a_bound_variable_is_refused() {
    let mut db = Database::open_in_memory().expect("open");
    let err = db
        .execute_cypher("UNWIND [1] AS status CALL db.advisor.reset() YIELD status RETURN status")
        .expect_err("redeclared variable");
    assert!(err.to_string().contains("status"), "{err}");
}

/// The clauses before a CALL keep their parameters and index access: the
/// call runs once for each matched node.
#[test]
fn a_call_after_match_sees_the_matched_rows_and_parameters() {
    let mut db = Database::open_in_memory().expect("open");
    db.execute_cypher("CREATE (:Tag {name: 'a'}), (:Tag {name: 'b'}), (:Tag {name: 'c'})")
        .expect("seed");
    let mut params = std::collections::HashMap::new();
    params.insert("skip".to_string(), Value::String("b".into()));
    let rows = db
        .execute_cypher_with_params(
            "MATCH (t:Tag) WHERE t.name <> $skip \
             CALL db.advisor.reset() YIELD status RETURN t.name AS name, status",
            params,
        )
        .expect("match then call");
    let mut names: Vec<String> = rows.iter().map(|r| string_of(r, "name")).collect();
    names.sort();
    assert_eq!(names, vec!["a", "c"]);
}

/// EXPLAIN shows the clauses feeding a CALL under it.
#[test]
fn explain_shows_the_input_of_a_call() {
    let db = Database::open_in_memory().expect("open");
    let plan = db
        .explain_cypher("UNWIND [1, 2] AS x CALL db.advisor.reset() YIELD status RETURN x")
        .expect("explain");
    let call = plan.find("ProcedureCall").expect("call in plan");
    assert!(plan[call..].contains("Unwind"), "{plan}");
}

/// A procedure registered by an embedder is called, listed, and classed by
/// its mode.
#[test]
fn a_registered_procedure_is_callable_and_listed() {
    use coordinode_query::executor::runner::{ExecutionContext, ExecutionError};
    use coordinode_query::procedure::{
        FieldSignature, Procedure, ProcedureError, ProcedureMode, ProcedureSignature, ValueType,
    };
    use std::sync::Arc;

    struct Repeat(ProcedureSignature);
    impl Procedure for Repeat {
        fn signature(&self) -> &ProcedureSignature {
            &self.0
        }
        fn call(
            &self,
            _ctx: &mut ExecutionContext<'_>,
            args: Vec<Value>,
        ) -> Result<Vec<Vec<Value>>, ExecutionError> {
            let Value::Int(times) = args[1] else {
                return Ok(Vec::new());
            };
            Ok((0..times)
                .map(|i| vec![args[0].clone(), Value::Int(i)])
                .collect())
        }
    }

    let mut db = Database::open_in_memory().expect("open");
    let repeat = || {
        Arc::new(Repeat(
            ProcedureSignature::new("app.repeat", ProcedureMode::Write, "Repeat a value.")
                .input(FieldSignature::new("value", ValueType::Any))
                .input(FieldSignature::new("times", ValueType::Integer).with_default(Value::Int(2)))
                .output("value", ValueType::Any)
                .output("index", ValueType::Integer),
        ))
    };
    db.register_procedure(repeat()).expect("register");
    assert!(matches!(
        db.register_procedure(repeat()),
        Err(ProcedureError::DuplicateName { .. })
    ));
    assert!(db.procedures().writes("app.repeat"));

    let rows = db
        .execute_cypher("CALL app.repeat('x') YIELD value, index RETURN value, index")
        .expect("call registered");
    assert_eq!(rows.len(), 2);
    assert!(rows.iter().all(|r| string_of(r, "value") == "x"));

    let listed = db
        .execute_cypher(
            "CALL dbms.procedures() YIELD name, mode WHERE name = 'app.repeat' RETURN mode",
        )
        .expect("listed");
    assert_eq!(listed.len(), 1);
    assert_eq!(string_of(&listed[0], "mode"), "WRITE");
}

/// Function names are case-insensitive, and `size` counts characters.
#[test]
fn function_names_ignore_case_and_size_counts_characters() {
    let mut db = Database::open_in_memory().expect("open");
    let rows = db
        .execute_cypher("RETURN COALESCE(null, 'x') AS c, Size('héllo') AS s, TOUPPER('a') AS u")
        .expect("case-insensitive functions");
    assert_eq!(string_of(&rows[0], "c"), "x");
    assert_eq!(rows[0].get("s"), Some(&Value::Int(5)));
    assert_eq!(string_of(&rows[0], "u"), "A");
}

/// The number of required parameters in a listed signature.
fn signature_arity(signature: &str) -> usize {
    let open = signature.find('(').expect("signature has parameters");
    let close = signature[open..].find(')').expect("closed") + open;
    let params = &signature[open + 1..close];
    if params.trim().is_empty() {
        return 0;
    }
    params
        .split(',')
        .filter(|p| !p.contains('=') && !p.trim_start().starts_with("..."))
        .count()
}
