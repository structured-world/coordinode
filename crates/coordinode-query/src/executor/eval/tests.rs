use super::*;

fn empty_row() -> Row {
    Row::new()
}

fn sample_row() -> Row {
    let mut row = Row::new();
    row.insert("n.name".into(), Value::String("Alice".into()));
    row.insert("n.age".into(), Value::Int(30));
    row.insert("n.score".into(), Value::Float(0.95));
    row.insert("n.active".into(), Value::Bool(true));
    row.insert("n".into(), Value::String("node_ref".into()));
    row
}

#[test]
fn eval_literal() {
    let v = eval_expr(&Expr::Literal(Value::Int(42)), &empty_row());
    assert_eq!(v, Value::Int(42));
}

#[test]
fn eval_variable() {
    let row = sample_row();
    let v = eval_expr(&Expr::Variable("n".into()), &row);
    assert_eq!(v, Value::String("node_ref".into()));
}

#[test]
fn eval_undefined_variable() {
    let v = eval_expr(&Expr::Variable("x".into()), &empty_row());
    assert_eq!(v, Value::Null);
}

#[test]
fn eval_property_access() {
    let row = sample_row();
    let expr = Expr::PropertyAccess {
        expr: Box::new(Expr::Variable("n".into())),
        property: "name".into(),
    };
    let v = eval_expr(&expr, &row);
    assert_eq!(v, Value::String("Alice".into()));
}

#[test]
fn eval_document_dot_notation_single_level() {
    let doc = rmpv::Value::Map(vec![(
        rmpv::Value::String("firmware".into()),
        rmpv::Value::String("2.1.3".into()),
    )]);
    let mut row = Row::new();
    row.insert("n.config".into(), Value::Document(doc));

    // n.config.firmware → "2.1.3"
    let expr = Expr::PropertyAccess {
        expr: Box::new(Expr::PropertyAccess {
            expr: Box::new(Expr::Variable("n".into())),
            property: "config".into(),
        }),
        property: "firmware".into(),
    };
    let v = eval_expr(&expr, &row);
    assert_eq!(v, Value::String("2.1.3".into()));
}

#[test]
fn eval_document_dot_notation_three_levels() {
    let doc = rmpv::Value::Map(vec![(
        rmpv::Value::String("network".into()),
        rmpv::Value::Map(vec![(
            rmpv::Value::String("ssid".into()),
            rmpv::Value::String("home".into()),
        )]),
    )]);
    let mut row = Row::new();
    row.insert("n.config".into(), Value::Document(doc));

    // n.config.network.ssid → "home"
    let expr = Expr::PropertyAccess {
        expr: Box::new(Expr::PropertyAccess {
            expr: Box::new(Expr::PropertyAccess {
                expr: Box::new(Expr::Variable("n".into())),
                property: "config".into(),
            }),
            property: "network".into(),
        }),
        property: "ssid".into(),
    };
    let v = eval_expr(&expr, &row);
    assert_eq!(v, Value::String("home".into()));
}

#[test]
fn eval_document_missing_key_returns_null() {
    let doc = rmpv::Value::Map(vec![(
        rmpv::Value::String("a".into()),
        rmpv::Value::Integer(1.into()),
    )]);
    let mut row = Row::new();
    row.insert("n.data".into(), Value::Document(doc));

    // n.data.nonexistent → Null
    let expr = Expr::PropertyAccess {
        expr: Box::new(Expr::PropertyAccess {
            expr: Box::new(Expr::Variable("n".into())),
            property: "data".into(),
        }),
        property: "nonexistent".into(),
    };
    let v = eval_expr(&expr, &row);
    assert_eq!(v, Value::Null);
}

#[test]
fn eval_document_numeric_returns_value() {
    let doc = rmpv::Value::Map(vec![(
        rmpv::Value::String("count".into()),
        rmpv::Value::Integer(42.into()),
    )]);
    let mut row = Row::new();
    row.insert("n.stats".into(), Value::Document(doc));

    // n.stats.count → Int(42)
    let expr = Expr::PropertyAccess {
        expr: Box::new(Expr::PropertyAccess {
            expr: Box::new(Expr::Variable("n".into())),
            property: "stats".into(),
        }),
        property: "count".into(),
    };
    let v = eval_expr(&expr, &row);
    assert_eq!(v, Value::Int(42));
}

#[test]
fn eval_arithmetic_add() {
    let row = sample_row();
    let expr = Expr::BinaryOp {
        left: Box::new(Expr::PropertyAccess {
            expr: Box::new(Expr::Variable("n".into())),
            property: "age".into(),
        }),
        op: BinaryOperator::Add,
        right: Box::new(Expr::Literal(Value::Int(5))),
    };
    let v = eval_expr(&expr, &row);
    assert_eq!(v, Value::Int(35));
}

#[test]
fn eval_comparison_gt() {
    let row = sample_row();
    let expr = Expr::BinaryOp {
        left: Box::new(Expr::PropertyAccess {
            expr: Box::new(Expr::Variable("n".into())),
            property: "age".into(),
        }),
        op: BinaryOperator::Gt,
        right: Box::new(Expr::Literal(Value::Int(25))),
    };
    let v = eval_expr(&expr, &row);
    assert_eq!(v, Value::Bool(true));
}

#[test]
fn eval_logical_and() {
    let expr = Expr::BinaryOp {
        left: Box::new(Expr::Literal(Value::Bool(true))),
        op: BinaryOperator::And,
        right: Box::new(Expr::Literal(Value::Bool(false))),
    };
    let v = eval_expr(&expr, &empty_row());
    assert_eq!(v, Value::Bool(false));
}

#[test]
fn eval_not() {
    let expr = Expr::UnaryOp {
        op: UnaryOperator::Not,
        expr: Box::new(Expr::Literal(Value::Bool(true))),
    };
    let v = eval_expr(&expr, &empty_row());
    assert_eq!(v, Value::Bool(false));
}

#[test]
fn eval_neg() {
    let expr = Expr::UnaryOp {
        op: UnaryOperator::Neg,
        expr: Box::new(Expr::Literal(Value::Int(42))),
    };
    let v = eval_expr(&expr, &empty_row());
    assert_eq!(v, Value::Int(-42));
}

#[test]
fn eval_string_starts_with() {
    let row = sample_row();
    let expr = Expr::StringMatch {
        expr: Box::new(Expr::PropertyAccess {
            expr: Box::new(Expr::Variable("n".into())),
            property: "name".into(),
        }),
        op: StringOp::StartsWith,
        pattern: Box::new(Expr::Literal(Value::String("Ali".into()))),
    };
    let v = eval_expr(&expr, &row);
    assert_eq!(v, Value::Bool(true));
}

#[test]
fn eval_string_regex_match() {
    let mk = |s: &str, pat: &str| Expr::StringMatch {
        expr: Box::new(Expr::Literal(Value::String(s.into()))),
        op: StringOp::Regex,
        pattern: Box::new(Expr::Literal(Value::String(pat.into()))),
    };
    // `=~` is a whole-string match (anchored).
    assert_eq!(
        eval_expr(&mk("hello", "h.*o"), &empty_row()),
        Value::Bool(true)
    );
    assert_eq!(
        eval_expr(&mk("hello", "hello"), &empty_row()),
        Value::Bool(true)
    );
    assert_eq!(
        eval_expr(&mk("abc123", "[a-z]+[0-9]+"), &empty_row()),
        Value::Bool(true)
    );
    // Substring that is not a whole-string match → false (anchored).
    assert_eq!(
        eval_expr(&mk("hello", "ell"), &empty_row()),
        Value::Bool(false)
    );
    // Invalid regex → false, never a panic.
    assert_eq!(eval_expr(&mk("x", "["), &empty_row()), Value::Bool(false));
}

#[test]
fn eval_list_slice() {
    let arr = || {
        Box::new(Expr::List(
            (1..=5).map(|i| Expr::Literal(Value::Int(i * 10))).collect(),
        ))
    };
    let lit = |i: i64| Some(Box::new(Expr::Literal(Value::Int(i))));
    let int_arr = |xs: &[i64]| Value::Array(xs.iter().map(|i| Value::Int(*i)).collect());
    let mk = |start, end| Expr::Slice {
        expr: arr(),
        start,
        end,
    };
    // [10,20,30,40,50]
    assert_eq!(
        eval_expr(&mk(lit(1), lit(3)), &empty_row()),
        int_arr(&[20, 30])
    );
    assert_eq!(
        eval_expr(&mk(None, lit(2)), &empty_row()),
        int_arr(&[10, 20])
    );
    assert_eq!(
        eval_expr(&mk(lit(3), None), &empty_row()),
        int_arr(&[40, 50])
    );
    // negative start counts from the end
    assert_eq!(
        eval_expr(&mk(lit(-2), None), &empty_row()),
        int_arr(&[40, 50])
    );
    // full slice
    assert_eq!(
        eval_expr(&mk(None, None), &empty_row()),
        int_arr(&[10, 20, 30, 40, 50])
    );
    // empty when start >= end
    assert_eq!(eval_expr(&mk(lit(3), lit(1)), &empty_row()), int_arr(&[]));
}

#[test]
fn eval_list_concat() {
    let l = |xs: &[i64]| {
        Box::new(Expr::List(
            xs.iter().map(|i| Expr::Literal(Value::Int(*i))).collect(),
        ))
    };
    let add = |a, b| Expr::BinaryOp {
        left: a,
        op: BinaryOperator::Add,
        right: b,
    };
    let int_arr = |xs: &[i64]| Value::Array(xs.iter().map(|i| Value::Int(*i)).collect());
    // list + list
    assert_eq!(
        eval_expr(&add(l(&[1, 2]), l(&[3, 4])), &empty_row()),
        int_arr(&[1, 2, 3, 4])
    );
    // list + element (append)
    assert_eq!(
        eval_expr(
            &add(l(&[1, 2]), Box::new(Expr::Literal(Value::Int(3)))),
            &empty_row()
        ),
        int_arr(&[1, 2, 3])
    );
    // element + list (prepend)
    assert_eq!(
        eval_expr(
            &add(Box::new(Expr::Literal(Value::Int(0))), l(&[1, 2])),
            &empty_row()
        ),
        int_arr(&[0, 1, 2])
    );
}

#[test]
fn eval_is_null() {
    let expr = Expr::IsNull {
        expr: Box::new(Expr::Literal(Value::Null)),
        negated: false,
    };
    let v = eval_expr(&expr, &empty_row());
    assert_eq!(v, Value::Bool(true));
}

#[test]
fn eval_is_typed() {
    let mk = |v: Value, t: &str, neg: bool| Expr::IsTyped {
        expr: Box::new(Expr::Literal(v)),
        type_name: t.into(),
        negated: neg,
    };
    assert_eq!(
        eval_expr(&mk(Value::Int(5), "INTEGER", false), &empty_row()),
        Value::Bool(true)
    );
    assert_eq!(
        eval_expr(&mk(Value::Int(5), "STRING", false), &empty_row()),
        Value::Bool(false)
    );
    assert_eq!(
        eval_expr(
            &mk(Value::String("x".into()), "STRING", false),
            &empty_row()
        ),
        Value::Bool(true)
    );
    // IS NOT :: negates.
    assert_eq!(
        eval_expr(&mk(Value::Int(5), "STRING", true), &empty_row()),
        Value::Bool(true)
    );
    assert_eq!(
        eval_expr(&mk(Value::Null, "NULL", false), &empty_row()),
        Value::Bool(true)
    );
    assert_eq!(
        eval_expr(&mk(Value::Float(1.0), "FLOAT", false), &empty_row()),
        Value::Bool(true)
    );
    // case-insensitive type name
    assert_eq!(
        eval_expr(&mk(Value::Bool(true), "boolean", false), &empty_row()),
        Value::Bool(true)
    );
}

#[test]
fn eval_is_not_null() {
    let expr = Expr::IsNull {
        expr: Box::new(Expr::Literal(Value::Int(1))),
        negated: true,
    };
    let v = eval_expr(&expr, &empty_row());
    assert_eq!(v, Value::Bool(true));
}

#[test]
fn eval_in_list() {
    let expr = Expr::In {
        expr: Box::new(Expr::Literal(Value::Int(2))),
        list: Box::new(Expr::List(vec![
            Expr::Literal(Value::Int(1)),
            Expr::Literal(Value::Int(2)),
            Expr::Literal(Value::Int(3)),
        ])),
    };
    let v = eval_expr(&expr, &empty_row());
    assert_eq!(v, Value::Bool(true));
}

#[test]
fn eval_case_generic() {
    let row = sample_row();
    let expr = Expr::Case {
        operand: None,
        when_clauses: vec![
            (
                Expr::BinaryOp {
                    left: Box::new(Expr::PropertyAccess {
                        expr: Box::new(Expr::Variable("n".into())),
                        property: "age".into(),
                    }),
                    op: BinaryOperator::Lt,
                    right: Box::new(Expr::Literal(Value::Int(18))),
                },
                Expr::Literal(Value::String("minor".into())),
            ),
            (
                Expr::Literal(Value::Bool(true)),
                Expr::Literal(Value::String("adult".into())),
            ),
        ],
        else_clause: None,
    };
    let v = eval_expr(&expr, &row);
    assert_eq!(v, Value::String("adult".into()));
}

#[test]
fn eval_case_simple_with_operand() {
    // Simple CASE: `CASE n.age WHEN 30 THEN 'thirty' WHEN 40 THEN 'forty' ELSE 'other' END`.
    let row = sample_row(); // n.age = 30
    let age = || Expr::PropertyAccess {
        expr: Box::new(Expr::Variable("n".into())),
        property: "age".into(),
    };
    let expr = Expr::Case {
        operand: Some(Box::new(age())),
        when_clauses: vec![
            (
                Expr::Literal(Value::Int(30)),
                Expr::Literal(Value::String("thirty".into())),
            ),
            (
                Expr::Literal(Value::Int(40)),
                Expr::Literal(Value::String("forty".into())),
            ),
        ],
        else_clause: Some(Box::new(Expr::Literal(Value::String("other".into())))),
    };
    assert_eq!(eval_expr(&expr, &row), Value::String("thirty".into()));
}

#[test]
fn eval_case_else_and_nested() {
    let row = sample_row(); // n.age = 30
    let age = || Expr::PropertyAccess {
        expr: Box::new(Expr::Variable("n".into())),
        property: "age".into(),
    };
    // No WHEN matches → ELSE branch, which is itself a (nested) simple CASE.
    let inner = Expr::Case {
        operand: Some(Box::new(age())),
        when_clauses: vec![(
            Expr::Literal(Value::Int(30)),
            Expr::Literal(Value::String("nested-thirty".into())),
        )],
        else_clause: None,
    };
    let outer = Expr::Case {
        operand: Some(Box::new(age())),
        when_clauses: vec![(
            Expr::Literal(Value::Int(99)),
            Expr::Literal(Value::String("never".into())),
        )],
        else_clause: Some(Box::new(inner)),
    };
    assert_eq!(
        eval_expr(&outer, &row),
        Value::String("nested-thirty".into())
    );
}

#[test]
fn eval_list_literal() {
    let expr = Expr::List(vec![
        Expr::Literal(Value::Int(1)),
        Expr::Literal(Value::Int(2)),
    ]);
    let v = eval_expr(&expr, &empty_row());
    assert_eq!(v, Value::Array(vec![Value::Int(1), Value::Int(2)]));
}

#[test]
fn eval_coalesce() {
    let expr = Expr::FunctionCall {
        name: "coalesce".into(),
        args: vec![
            Expr::Literal(Value::Null),
            Expr::Literal(Value::String("fallback".into())),
        ],
        distinct: false,
    };
    let v = eval_expr(&expr, &empty_row());
    assert_eq!(v, Value::String("fallback".into()));
}

#[test]
fn eval_size_string() {
    let expr = Expr::FunctionCall {
        name: "size".into(),
        args: vec![Expr::Literal(Value::String("hello".into()))],
        distinct: false,
    };
    let v = eval_expr(&expr, &empty_row());
    assert_eq!(v, Value::Int(5));
}

#[test]
fn eval_path_functions() {
    use coordinode_core::graph::types::{PathRel, PathValue};
    // Path n10 -[:KNOWS]-> n20 -[:KNOWS]-> n30: length 2, 3 nodes.
    let path = Value::Path(PathValue {
        nodes: vec![10, 20, 30],
        rels: vec![
            PathRel {
                edge_type: "KNOWS".into(),
                source: 10,
                target: 20,
            },
            PathRel {
                edge_type: "KNOWS".into(),
                source: 20,
                target: 30,
            },
        ],
    });
    let call = |name: &str| {
        eval_expr(
            &Expr::FunctionCall {
                name: name.into(),
                args: vec![Expr::Literal(path.clone())],
                distinct: false,
            },
            &empty_row(),
        )
    };

    assert_eq!(call("length"), Value::Int(2), "length = relationship count");
    assert_eq!(
        call("nodes"),
        Value::Array(vec![Value::Int(10), Value::Int(20), Value::Int(30)])
    );

    let mut rel0 = std::collections::BTreeMap::new();
    rel0.insert("type".to_string(), Value::String("KNOWS".into()));
    rel0.insert("source".to_string(), Value::Int(10));
    rel0.insert("target".to_string(), Value::Int(20));
    let mut rel1 = std::collections::BTreeMap::new();
    rel1.insert("type".to_string(), Value::String("KNOWS".into()));
    rel1.insert("source".to_string(), Value::Int(20));
    rel1.insert("target".to_string(), Value::Int(30));
    assert_eq!(
        call("relationships"),
        Value::Array(vec![Value::Map(rel0), Value::Map(rel1)])
    );
}

#[test]
fn eval_string_concat() {
    let expr = Expr::BinaryOp {
        left: Box::new(Expr::Literal(Value::String("hello ".into()))),
        op: BinaryOperator::Add,
        right: Box::new(Expr::Literal(Value::String("world".into()))),
    };
    let v = eval_expr(&expr, &empty_row());
    assert_eq!(v, Value::String("hello world".into()));
}

#[test]
fn eval_div_by_zero() {
    let expr = Expr::BinaryOp {
        left: Box::new(Expr::Literal(Value::Int(10))),
        op: BinaryOperator::Div,
        right: Box::new(Expr::Literal(Value::Int(0))),
    };
    let v = eval_expr(&expr, &empty_row());
    assert_eq!(v, Value::Null);
}

#[test]
fn eval_mixed_int_float() {
    let expr = Expr::BinaryOp {
        left: Box::new(Expr::Literal(Value::Int(10))),
        op: BinaryOperator::Mul,
        right: Box::new(Expr::Literal(Value::Float(1.5))),
    };
    let v = eval_expr(&expr, &empty_row());
    assert_eq!(v, Value::Float(15.0));
}

#[test]
fn is_truthy_values() {
    assert!(is_truthy(&Value::Bool(true)));
    assert!(!is_truthy(&Value::Bool(false)));
    assert!(!is_truthy(&Value::Null));
    assert!(is_truthy(&Value::Int(0))); // non-null is truthy
    assert!(is_truthy(&Value::String("".into()))); // non-null is truthy
}

// ====== Vector functions ======

#[test]
fn eval_maxsim_score_matches_kernel() {
    // q = [[1,0],[0,1]], d = [[1,0],[0,1],[0.5,0.5]]
    // Per-q max similarities: 1.0 + 1.0 = 2.0
    let q = Value::MultiVector(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
    let d = Value::MultiVector(vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]]);
    let expr = Expr::FunctionCall {
        name: "maxsim_score".into(),
        args: vec![Expr::Literal(d), Expr::Literal(q)],
        distinct: false,
    };
    let v = eval_expr(&expr, &empty_row());
    if let Value::Float(s) = v {
        assert!((s - 2.0).abs() < 1e-5, "got {s}");
    } else {
        panic!("expected Float, got {v:?}");
    }
}

#[test]
fn eval_maxsim_score_accepts_array_of_arrays() {
    // Same logical input as above but represented as Array<Array<Float>>
    // so the executor can score query matrices arriving as parameter
    // literals (Cypher map / list literals decode to Array, not MultiVector).
    let q = Value::Array(vec![
        Value::Array(vec![Value::Float(1.0), Value::Float(0.0)]),
        Value::Array(vec![Value::Float(0.0), Value::Float(1.0)]),
    ]);
    let d = Value::Array(vec![
        Value::Array(vec![Value::Float(1.0), Value::Float(0.0)]),
        Value::Array(vec![Value::Float(0.0), Value::Float(1.0)]),
    ]);
    let expr = Expr::FunctionCall {
        name: "maxsim_score".into(),
        args: vec![Expr::Literal(d), Expr::Literal(q)],
        distinct: false,
    };
    let v = eval_expr(&expr, &empty_row());
    assert!(matches!(v, Value::Float(_)));
}

#[test]
fn eval_maxsim_score_rejects_mismatched_dim() {
    let q = Value::MultiVector(vec![vec![1.0, 0.0]]);
    let d = Value::Array(vec![Value::Array(vec![
        Value::Float(1.0),
        Value::Float(0.0),
        Value::Float(0.0),
    ])]);
    let expr = Expr::FunctionCall {
        name: "maxsim_score".into(),
        args: vec![Expr::Literal(d), Expr::Literal(q)],
        distinct: false,
    };
    let v = eval_expr(&expr, &empty_row());
    // coerce ok on each, but kernel returns 0 on dim mismatch.
    if let Value::Float(s) = v {
        assert_eq!(s, 0.0);
    } else {
        panic!("expected Float, got {v:?}");
    }
}

#[test]
fn eval_maxsim_score_null_on_missing_arg() {
    let expr = Expr::FunctionCall {
        name: "maxsim_score".into(),
        args: vec![Expr::Literal(Value::Int(42))],
        distinct: false,
    };
    let v = eval_expr(&expr, &empty_row());
    assert!(matches!(v, Value::Null));
}

#[test]
fn eval_vector_distance() {
    let expr = Expr::FunctionCall {
        name: "vector_distance".into(),
        args: vec![
            Expr::Literal(Value::Vector(vec![1.0, 0.0, 0.0])),
            Expr::Literal(Value::Vector(vec![0.0, 1.0, 0.0])),
        ],
        distinct: false,
    };
    let v = eval_expr(&expr, &empty_row());
    if let Value::Float(d) = v {
        // L2 distance between (1,0,0) and (0,1,0) = sqrt(2) ≈ 1.414
        assert!((d - std::f64::consts::SQRT_2).abs() < 0.01);
    } else {
        panic!("expected Float, got {v:?}");
    }
}

#[test]
fn eval_vector_similarity() {
    let expr = Expr::FunctionCall {
        name: "vector_similarity".into(),
        args: vec![
            Expr::Literal(Value::Vector(vec![1.0, 0.0])),
            Expr::Literal(Value::Vector(vec![1.0, 0.0])),
        ],
        distinct: false,
    };
    let v = eval_expr(&expr, &empty_row());
    if let Value::Float(s) = v {
        assert!((s - 1.0).abs() < 0.01); // Identical vectors → cosine 1.0
    } else {
        panic!("expected Float, got {v:?}");
    }
}

#[test]
fn eval_vector_similarity_orthogonal() {
    let expr = Expr::FunctionCall {
        name: "vector_similarity".into(),
        args: vec![
            Expr::Literal(Value::Vector(vec![1.0, 0.0])),
            Expr::Literal(Value::Vector(vec![0.0, 1.0])),
        ],
        distinct: false,
    };
    let v = eval_expr(&expr, &empty_row());
    if let Value::Float(s) = v {
        assert!(s.abs() < 0.01); // Orthogonal → cosine 0.0
    } else {
        panic!("expected Float");
    }
}

#[test]
fn eval_vector_dot() {
    let expr = Expr::FunctionCall {
        name: "vector_dot".into(),
        args: vec![
            Expr::Literal(Value::Vector(vec![1.0, 2.0, 3.0])),
            Expr::Literal(Value::Vector(vec![4.0, 5.0, 6.0])),
        ],
        distinct: false,
    };
    let v = eval_expr(&expr, &empty_row());
    if let Value::Float(d) = v {
        assert!((d - 32.0).abs() < 0.01); // 1*4 + 2*5 + 3*6 = 32
    } else {
        panic!("expected Float");
    }
}

#[test]
fn eval_vector_manhattan() {
    let expr = Expr::FunctionCall {
        name: "vector_manhattan".into(),
        args: vec![
            Expr::Literal(Value::Vector(vec![1.0, 2.0, 3.0])),
            Expr::Literal(Value::Vector(vec![4.0, 6.0, 8.0])),
        ],
        distinct: false,
    };
    let v = eval_expr(&expr, &empty_row());
    if let Value::Float(d) = v {
        assert!((d - 12.0).abs() < 0.01); // |1-4| + |2-6| + |3-8| = 12
    } else {
        panic!("expected Float");
    }
}

#[test]
fn eval_vector_distance_dim_mismatch() {
    let expr = Expr::FunctionCall {
        name: "vector_distance".into(),
        args: vec![
            Expr::Literal(Value::Vector(vec![1.0, 0.0])),
            Expr::Literal(Value::Vector(vec![1.0, 0.0, 0.0])),
        ],
        distinct: false,
    };
    let v = eval_expr(&expr, &empty_row());
    assert_eq!(v, Value::Null); // Dimension mismatch → null
}

#[test]
fn eval_vector_distance_non_vector() {
    let expr = Expr::FunctionCall {
        name: "vector_distance".into(),
        args: vec![
            Expr::Literal(Value::String("not a vector".into())),
            Expr::Literal(Value::Vector(vec![1.0])),
        ],
        distinct: false,
    };
    let v = eval_expr(&expr, &empty_row());
    assert_eq!(v, Value::Null); // Non-vector → null
}

#[test]
fn vector_in_where_clause() {
    // Simulate: WHERE vector_distance(n.embedding, $query) < 0.5
    let mut row = Row::new();
    row.insert("n.embedding".into(), Value::Vector(vec![1.0, 0.0, 0.0]));

    let expr = Expr::BinaryOp {
        left: Box::new(Expr::FunctionCall {
            name: "vector_distance".into(),
            args: vec![
                Expr::PropertyAccess {
                    expr: Box::new(Expr::Variable("n".into())),
                    property: "embedding".into(),
                },
                Expr::Literal(Value::Vector(vec![1.0, 0.0, 0.0])),
            ],
            distinct: false,
        }),
        op: BinaryOperator::Lt,
        right: Box::new(Expr::Literal(Value::Float(0.5))),
    };

    let v = eval_expr(&expr, &row);
    assert_eq!(v, Value::Bool(true)); // Distance 0 < 0.5
}

// --- Spatial function tests ---

/// point() constructs a GeoValue from a map with latitude/longitude.
#[test]
fn point_constructor() {
    let row = Row::new();
    let expr = Expr::FunctionCall {
        name: "point".to_string(),
        args: vec![Expr::MapLiteral(vec![
            ("latitude".to_string(), Expr::Literal(Value::Float(40.7128))),
            (
                "longitude".to_string(),
                Expr::Literal(Value::Float(-74.006)),
            ),
        ])],
        distinct: false,
    };

    let result = eval_expr(&expr, &row);
    assert!(
        matches!(
            result,
            Value::Geo(coordinode_core::graph::types::GeoValue::Point { .. })
        ),
        "point() should return Geo(Point), got: {result:?}"
    );
}

/// point.distance() computes Haversine distance between two points.
#[test]
fn point_distance_haversine() {
    let row = Row::new();
    // New York (40.7128, -74.0060) to London (51.5074, -0.1278) ≈ 5,570 km
    let expr = Expr::FunctionCall {
        name: "point.distance".to_string(),
        args: vec![
            Expr::FunctionCall {
                name: "point".to_string(),
                args: vec![Expr::MapLiteral(vec![
                    ("latitude".to_string(), Expr::Literal(Value::Float(40.7128))),
                    (
                        "longitude".to_string(),
                        Expr::Literal(Value::Float(-74.006)),
                    ),
                ])],
                distinct: false,
            },
            Expr::FunctionCall {
                name: "point".to_string(),
                args: vec![Expr::MapLiteral(vec![
                    ("latitude".to_string(), Expr::Literal(Value::Float(51.5074))),
                    (
                        "longitude".to_string(),
                        Expr::Literal(Value::Float(-0.1278)),
                    ),
                ])],
                distinct: false,
            },
        ],
        distinct: false,
    };

    let result = eval_expr(&expr, &row);
    if let Value::Float(dist) = result {
        // NY to London ≈ 5,570,000 meters ± 1%
        assert!(
            (5_500_000.0..5_650_000.0).contains(&dist),
            "NY to London should be ~5,570km, got {:.0}m",
            dist
        );
    } else {
        panic!("point.distance should return Float, got {result:?}");
    }
}

/// point.distance() with same point = 0.
#[test]
fn point_distance_same_point() {
    let row = Row::new();
    let point_expr = Expr::FunctionCall {
        name: "point".to_string(),
        args: vec![Expr::MapLiteral(vec![
            ("latitude".to_string(), Expr::Literal(Value::Float(0.0))),
            ("longitude".to_string(), Expr::Literal(Value::Float(0.0))),
        ])],
        distinct: false,
    };

    let expr = Expr::FunctionCall {
        name: "point.distance".to_string(),
        args: vec![point_expr.clone(), point_expr],
        distinct: false,
    };

    assert_eq!(eval_expr(&expr, &row), Value::Float(0.0));
}

/// Haversine helper: known distance for antipodal points.
#[test]
fn haversine_antipodal() {
    // North Pole to South Pole ≈ 20,015 km (half circumference)
    let dist = super::haversine_distance(90.0, 0.0, -90.0, 0.0);
    assert!(
        (20_000_000.0..20_040_000.0).contains(&dist),
        "pole to pole should be ~20,015km, got {:.0}m",
        dist
    );
}

// -- Map Projection --

#[test]
fn eval_map_projection_shorthand() {
    let row = sample_row();
    // n { .name, .age } → Map { name: "Alice", age: 30 }
    let expr = Expr::MapProjection {
        expr: Box::new(Expr::Variable("n".into())),
        items: vec![
            MapProjectionItem::Property("name".into()),
            MapProjectionItem::Property("age".into()),
        ],
    };
    let result = eval_expr(&expr, &row);
    if let Value::Map(map) = result {
        assert_eq!(map.get("name"), Some(&Value::String("Alice".into())));
        assert_eq!(map.get("age"), Some(&Value::Int(30)));
        assert_eq!(map.len(), 2);
    } else {
        panic!("expected Map, got: {result:?}");
    }
}

#[test]
fn eval_map_projection_computed() {
    let row = sample_row();
    // n { .name, doubled_age: n.age } with computed
    let expr = Expr::MapProjection {
        expr: Box::new(Expr::Variable("n".into())),
        items: vec![
            MapProjectionItem::Property("name".into()),
            MapProjectionItem::Computed(
                "score_val".into(),
                Expr::PropertyAccess {
                    expr: Box::new(Expr::Variable("n".into())),
                    property: "score".into(),
                },
            ),
        ],
    };
    let result = eval_expr(&expr, &row);
    if let Value::Map(map) = result {
        assert_eq!(map.get("name"), Some(&Value::String("Alice".into())));
        assert_eq!(map.get("score_val"), Some(&Value::Float(0.95)));
    } else {
        panic!("expected Map, got: {result:?}");
    }
}

#[test]
fn eval_map_projection_missing_prop_is_null() {
    let row = sample_row();
    let expr = Expr::MapProjection {
        expr: Box::new(Expr::Variable("n".into())),
        items: vec![MapProjectionItem::Property("nonexistent".into())],
    };
    let result = eval_expr(&expr, &row);
    if let Value::Map(map) = result {
        assert_eq!(map.get("nonexistent"), Some(&Value::Null));
    } else {
        panic!("expected Map");
    }
}

// ---- R520 string functions ----

/// Invoke a scalar function by name with literal arguments against an empty row.
fn call_fn(name: &str, args: Vec<Value>) -> Value {
    let expr = Expr::FunctionCall {
        name: name.into(),
        args: args.into_iter().map(Expr::Literal).collect(),
        distinct: false,
    };
    eval_expr(&expr, &empty_row())
}

fn s(v: &str) -> Value {
    Value::String(v.into())
}

#[test]
fn string_fn_tolower_and_alias() {
    assert_eq!(call_fn("toLower", vec![s("HeLLo")]), s("hello"));
    // `lower` is an alias for `toLower`.
    assert_eq!(call_fn("lower", vec![s("HeLLo")]), s("hello"));
    // Names are case-insensitive per the Cypher spec.
    assert_eq!(call_fn("TOLOWER", vec![s("ABC")]), s("abc"));
}

#[test]
fn string_fn_toupper_and_alias() {
    assert_eq!(call_fn("toUpper", vec![s("HeLLo")]), s("HELLO"));
    assert_eq!(call_fn("upper", vec![s("HeLLo")]), s("HELLO"));
}

#[test]
fn string_fn_trims() {
    assert_eq!(call_fn("trim", vec![s("  hi  ")]), s("hi"));
    assert_eq!(call_fn("ltrim", vec![s("  hi  ")]), s("hi  "));
    assert_eq!(call_fn("rtrim", vec![s("  hi  ")]), s("  hi"));
    assert_eq!(call_fn("btrim", vec![s("  hi  ")]), s("hi"));
}

#[test]
fn string_fn_left_right() {
    assert_eq!(call_fn("left", vec![s("hello"), Value::Int(3)]), s("hel"));
    assert_eq!(call_fn("right", vec![s("hello"), Value::Int(3)]), s("llo"));
    // len beyond the string length clamps to the whole string.
    assert_eq!(call_fn("left", vec![s("hi"), Value::Int(10)]), s("hi"));
    assert_eq!(call_fn("right", vec![s("hi"), Value::Int(10)]), s("hi"));
    // negative length → NULL (no panic).
    assert_eq!(call_fn("left", vec![s("hi"), Value::Int(-1)]), Value::Null);
}

#[test]
fn string_fn_substring() {
    // 0-indexed start, optional length.
    assert_eq!(
        call_fn("substring", vec![s("hello"), Value::Int(1), Value::Int(3)]),
        s("ell")
    );
    // omitted length → to end.
    assert_eq!(
        call_fn("substring", vec![s("hello"), Value::Int(2)]),
        s("llo")
    );
    // start past the end → empty string.
    assert_eq!(call_fn("substring", vec![s("hi"), Value::Int(9)]), s(""));
}

#[test]
fn string_fn_unicode_aware_indexing() {
    // Multi-byte chars: "héllo" — `left`/`substring`/`charLength` count
    // codepoints, not bytes, so we never split inside a codepoint.
    assert_eq!(call_fn("left", vec![s("héllo"), Value::Int(2)]), s("hé"));
    assert_eq!(call_fn("charLength", vec![s("héllo")]), Value::Int(5));
    assert_eq!(
        call_fn("substring", vec![s("héllo"), Value::Int(1), Value::Int(1)]),
        s("é")
    );
}

#[test]
fn string_fn_replace() {
    assert_eq!(
        call_fn("replace", vec![s("a-b-c"), s("-"), s("+")]),
        s("a+b+c")
    );
}

#[test]
fn string_fn_reverse_string_and_list() {
    assert_eq!(call_fn("reverse", vec![s("abc")]), s("cba"));
    // reverse also operates on a list.
    let list = Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
    assert_eq!(
        call_fn("reverse", vec![list]),
        Value::Array(vec![Value::Int(3), Value::Int(2), Value::Int(1)])
    );
}

#[test]
fn string_fn_split() {
    assert_eq!(
        call_fn("split", vec![s("a,b,c"), s(",")]),
        Value::Array(vec![s("a"), s("b"), s("c")])
    );
    // a list of delimiters splits on any of them.
    let delims = Value::Array(vec![s(","), s(";")]);
    assert_eq!(
        call_fn("split", vec![s("a,b;c"), delims]),
        Value::Array(vec![s("a"), s("b"), s("c")])
    );
    // empty delimiter → whole string as a single element.
    assert_eq!(
        call_fn("split", vec![s("abc"), s("")]),
        Value::Array(vec![s("abc")])
    );
}

#[test]
fn string_fn_charlength() {
    assert_eq!(call_fn("charLength", vec![s("hello")]), Value::Int(5));
    assert_eq!(call_fn("charLength", vec![s("")]), Value::Int(0));
}

#[test]
fn string_fn_tostring_variants() {
    assert_eq!(call_fn("toStringOrNull", vec![Value::Int(42)]), s("42"));
    assert_eq!(
        call_fn("toStringOrNull", vec![Value::Bool(true)]),
        s("true")
    );
    // unconvertible → NULL (not an error).
    assert_eq!(
        call_fn("toStringOrNull", vec![Value::Array(vec![])]),
        Value::Null
    );
    // toStringList maps each element.
    let list = Value::Array(vec![Value::Int(1), s("x"), Value::Bool(false)]);
    assert_eq!(
        call_fn("toStringList", vec![list]),
        Value::Array(vec![s("1"), s("x"), s("false")])
    );
}

#[test]
fn string_fn_normalize() {
    // "é" as decomposed (e + combining acute, NFD) normalizes under NFC to the
    // single precomposed codepoint — fewer chars, same rendering.
    let decomposed = s("e\u{0301}");
    let nfc = call_fn("normalize", vec![decomposed.clone()]);
    assert_eq!(nfc, s("\u{00e9}"));
    // explicit form argument.
    assert_eq!(
        call_fn("normalize", vec![decomposed, s("NFC")]),
        s("\u{00e9}")
    );
    // unknown form → NULL.
    assert_eq!(call_fn("normalize", vec![s("a"), s("NFZ")]), Value::Null);
}

#[test]
fn string_fn_null_propagation() {
    // A string function applied to NULL (or a non-string) returns NULL.
    assert_eq!(call_fn("toLower", vec![Value::Null]), Value::Null);
    assert_eq!(call_fn("trim", vec![Value::Int(1)]), Value::Null);
    assert_eq!(
        call_fn("substring", vec![Value::Null, Value::Int(0)]),
        Value::Null
    );
}

#[test]
fn unknown_function_still_returns_null() {
    // The string/math fallthrough must not swallow the unknown-function
    // NULL contract for names it does not own.
    assert_eq!(call_fn("no_such_fn", vec![s("x")]), Value::Null);
}

// ---- R521 math functions ----

#[test]
fn math_fn_abs_preserves_type() {
    assert_eq!(call_fn("abs", vec![Value::Int(-5)]), Value::Int(5));
    assert_eq!(call_fn("abs", vec![Value::Float(-2.5)]), Value::Float(2.5));
    assert_eq!(call_fn("ABS", vec![Value::Int(7)]), Value::Int(7)); // case-insensitive
}

#[test]
fn math_fn_ceil_floor_round_return_float() {
    assert_eq!(call_fn("ceil", vec![Value::Float(1.2)]), Value::Float(2.0));
    assert_eq!(call_fn("floor", vec![Value::Float(1.8)]), Value::Float(1.0));
    assert_eq!(call_fn("round", vec![Value::Float(2.5)]), Value::Float(3.0));
    assert_eq!(
        call_fn("round", vec![Value::Float(-2.5)]),
        Value::Float(-3.0)
    );
    // integer input is accepted (widened to float domain)
    assert_eq!(call_fn("ceil", vec![Value::Int(3)]), Value::Float(3.0));
}

#[test]
fn math_fn_sign() {
    assert_eq!(call_fn("sign", vec![Value::Int(-9)]), Value::Int(-1));
    assert_eq!(call_fn("sign", vec![Value::Int(0)]), Value::Int(0));
    assert_eq!(call_fn("sign", vec![Value::Float(4.2)]), Value::Int(1));
}

#[test]
fn math_fn_sqrt_exp_log() {
    assert_eq!(call_fn("sqrt", vec![Value::Float(9.0)]), Value::Float(3.0));
    assert_eq!(
        call_fn("log10", vec![Value::Float(1000.0)]),
        Value::Float(3.0)
    );
    // ln(e) == 1
    if let Value::Float(v) = call_fn("log", vec![Value::Float(std::f64::consts::E)]) {
        assert!((v - 1.0).abs() < 1e-12);
    } else {
        panic!("log should return a float");
    }
}

#[test]
fn math_fn_constants() {
    if let Value::Float(p) = call_fn("pi", vec![]) {
        assert!((p - std::f64::consts::PI).abs() < 1e-12);
    } else {
        panic!("pi() should return a float");
    }
    if let Value::Float(e) = call_fn("e", vec![]) {
        assert!((e - std::f64::consts::E).abs() < 1e-12);
    } else {
        panic!("e() should return a float");
    }
}

#[test]
fn math_fn_rand_in_unit_interval() {
    // rand() is non-deterministic by spec; assert only the [0, 1) contract.
    for _ in 0..50 {
        match call_fn("rand", vec![]) {
            Value::Float(r) => assert!((0.0..1.0).contains(&r), "rand out of range: {r}"),
            other => panic!("rand should return a float, got {other:?}"),
        }
    }
}

#[test]
fn math_fn_isnan() {
    assert_eq!(
        call_fn("isNaN", vec![Value::Float(f64::NAN)]),
        Value::Bool(true)
    );
    assert_eq!(
        call_fn("isNaN", vec![Value::Float(1.0)]),
        Value::Bool(false)
    );
    assert_eq!(call_fn("isNaN", vec![Value::Int(3)]), Value::Bool(false));
}

#[test]
fn math_fn_to_integer() {
    assert_eq!(call_fn("toInteger", vec![Value::Float(3.7)]), Value::Int(3));
    assert_eq!(call_fn("toInteger", vec![s("42")]), Value::Int(42));
    assert_eq!(call_fn("toInteger", vec![s("3.9")]), Value::Int(3));
    assert_eq!(call_fn("toInteger", vec![s("nope")]), Value::Null);
    assert_eq!(
        call_fn("toIntegerOrNull", vec![Value::Bool(true)]),
        Value::Int(1)
    );
}

#[test]
fn math_fn_to_float_and_boolean() {
    assert_eq!(call_fn("toFloat", vec![Value::Int(5)]), Value::Float(5.0));
    assert_eq!(call_fn("toFloat", vec![s("2.5")]), Value::Float(2.5));
    assert_eq!(call_fn("toFloat", vec![s("x")]), Value::Null);
    assert_eq!(call_fn("toBoolean", vec![s("TRUE")]), Value::Bool(true));
    assert_eq!(call_fn("toBoolean", vec![s("false")]), Value::Bool(false));
    assert_eq!(
        call_fn("toBoolean", vec![Value::Int(0)]),
        Value::Bool(false)
    );
    assert_eq!(call_fn("toBoolean", vec![s("maybe")]), Value::Null);
}

#[test]
fn math_fn_list_conversions() {
    let list = Value::Array(vec![s("1"), s("2"), s("bad")]);
    assert_eq!(
        call_fn("toIntegerList", vec![list]),
        Value::Array(vec![Value::Int(1), Value::Int(2), Value::Null])
    );
    let fl = Value::Array(vec![Value::Int(1), s("2.5")]);
    assert_eq!(
        call_fn("toFloatList", vec![fl]),
        Value::Array(vec![Value::Float(1.0), Value::Float(2.5)])
    );
}

#[test]
fn math_fn_null_propagation() {
    assert_eq!(call_fn("abs", vec![Value::Null]), Value::Null);
    assert_eq!(call_fn("sqrt", vec![s("x")]), Value::Null);
    assert_eq!(call_fn("sign", vec![Value::Null]), Value::Null);
}

// ---- R522 trigonometric functions ----

fn approx(v: Value, want: f64) {
    match v {
        Value::Float(f) => assert!((f - want).abs() < 1e-9, "got {f}, want {want}"),
        other => panic!("expected float, got {other:?}"),
    }
}

#[test]
fn trig_fn_basic() {
    approx(call_fn("sin", vec![Value::Float(0.0)]), 0.0);
    approx(call_fn("cos", vec![Value::Float(0.0)]), 1.0);
    approx(call_fn("tan", vec![Value::Float(0.0)]), 0.0);
    // sin(pi/2) == 1
    approx(
        call_fn("sin", vec![Value::Float(std::f64::consts::FRAC_PI_2)]),
        1.0,
    );
}

#[test]
fn trig_fn_cot_and_inverse() {
    // cot(pi/4) == 1
    approx(
        call_fn("cot", vec![Value::Float(std::f64::consts::FRAC_PI_4)]),
        1.0,
    );
    approx(
        call_fn("asin", vec![Value::Float(1.0)]),
        std::f64::consts::FRAC_PI_2,
    );
    approx(call_fn("acos", vec![Value::Float(1.0)]), 0.0);
    approx(call_fn("atan", vec![Value::Float(0.0)]), 0.0);
}

#[test]
fn trig_fn_atan2() {
    // atan2(1, 1) == pi/4
    approx(
        call_fn("atan2", vec![Value::Float(1.0), Value::Float(1.0)]),
        std::f64::consts::FRAC_PI_4,
    );
    // missing second arg → NULL
    assert_eq!(call_fn("atan2", vec![Value::Float(1.0)]), Value::Null);
}

#[test]
fn trig_fn_haversin_and_conversions() {
    // haversin(0) == 0
    approx(call_fn("haversin", vec![Value::Float(0.0)]), 0.0);
    // degrees(pi) == 180, radians(180) == pi
    approx(
        call_fn("degrees", vec![Value::Float(std::f64::consts::PI)]),
        180.0,
    );
    approx(
        call_fn("radians", vec![Value::Float(180.0)]),
        std::f64::consts::PI,
    );
}

#[test]
fn trig_fn_null_propagation() {
    assert_eq!(call_fn("sin", vec![Value::Null]), Value::Null);
    assert_eq!(call_fn("degrees", vec![s("x")]), Value::Null);
}

// ---- R523 scalar functions ----

#[test]
fn scalar_fn_nullif() {
    assert_eq!(
        call_fn("nullIf", vec![Value::Int(5), Value::Int(5)]),
        Value::Null
    );
    assert_eq!(
        call_fn("nullIf", vec![Value::Int(5), Value::Int(6)]),
        Value::Int(5)
    );
    assert_eq!(call_fn("nullIf", vec![s("a"), s("b")]), s("a"));
}

#[test]
fn scalar_fn_value_type() {
    assert_eq!(
        call_fn("valueType", vec![Value::Int(1)]),
        s("INTEGER NOT NULL")
    );
    assert_eq!(call_fn("valueType", vec![s("x")]), s("STRING NOT NULL"));
    assert_eq!(
        call_fn("valueType", vec![Value::Bool(true)]),
        s("BOOLEAN NOT NULL")
    );
    assert_eq!(
        call_fn("valueType", vec![Value::Float(1.0)]),
        s("FLOAT NOT NULL")
    );
    assert_eq!(call_fn("valueType", vec![Value::Null]), s("NULL"));
    assert_eq!(
        call_fn("valueType", vec![Value::Array(vec![])]),
        s("LIST<ANY> NOT NULL")
    );
}

#[test]
fn scalar_fn_timestamp_is_positive_int() {
    match call_fn("timestamp", vec![]) {
        Value::Int(ms) => assert!(ms > 0, "timestamp must be a positive epoch ms"),
        other => panic!("timestamp should be Int, got {other:?}"),
    }
}

#[test]
fn scalar_fn_random_uuid_format() {
    let uuid_str = |v: Value| match v {
        Value::String(s) => s,
        other => panic!("randomUUID should be a String, got {other:?}"),
    };
    let a = uuid_str(call_fn("randomUUID", vec![]));
    assert_eq!(a.len(), 36, "canonical UUID length");
    assert_eq!(a.as_bytes()[8], b'-');
    assert_eq!(a.as_bytes()[13], b'-');
    assert_eq!(a.as_bytes()[14], b'4', "version-4 nibble");
    // Two calls produce distinct ids.
    let b = uuid_str(call_fn("randomUUID", vec![]));
    assert_ne!(a, b);
}

#[test]
fn scalar_fn_start_end_node_and_properties() {
    let mut row = Row::new();
    row.insert("r.__src__".into(), Value::Int(10));
    row.insert("r.__tgt__".into(), Value::Int(20));
    row.insert("n.name".into(), s("Alice"));
    row.insert("n.age".into(), Value::Int(30));
    row.insert("n.__label__".into(), s("Person"));

    let call = |name: &str, var: &str| {
        eval_expr(
            &Expr::FunctionCall {
                name: name.into(),
                args: vec![Expr::Variable(var.into())],
                distinct: false,
            },
            &row,
        )
    };

    // startNode / endNode resolve the relationship endpoints.
    assert_eq!(call("startNode", "r"), Value::Int(10));
    assert_eq!(call("endNode", "r"), Value::Int(20));

    // properties collects user fields, excluding internal __…__ markers.
    match call("properties", "n") {
        Value::Map(m) => {
            assert_eq!(m.get("name"), Some(&s("Alice")));
            assert_eq!(m.get("age"), Some(&Value::Int(30)));
            assert!(!m.contains_key("__label__"), "internal markers excluded");
        }
        other => panic!("properties should be a Map, got {other:?}"),
    }
}

// ---- R524 list functions ----

#[test]
fn list_fn_head_last_tail() {
    let lst = || Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
    assert_eq!(call_fn("head", vec![lst()]), Value::Int(1));
    assert_eq!(call_fn("last", vec![lst()]), Value::Int(3));
    assert_eq!(
        call_fn("tail", vec![lst()]),
        Value::Array(vec![Value::Int(2), Value::Int(3)])
    );
    // Empty-list edge cases: head/last → NULL, tail → empty list.
    assert_eq!(call_fn("head", vec![Value::Array(vec![])]), Value::Null);
    assert_eq!(
        call_fn("tail", vec![Value::Array(vec![])]),
        Value::Array(vec![])
    );
}

#[test]
fn list_fn_range() {
    assert_eq!(
        call_fn("range", vec![Value::Int(1), Value::Int(4)]),
        Value::Array(vec![
            Value::Int(1),
            Value::Int(2),
            Value::Int(3),
            Value::Int(4)
        ])
    );
    // explicit step
    assert_eq!(
        call_fn("range", vec![Value::Int(0), Value::Int(10), Value::Int(5)]),
        Value::Array(vec![Value::Int(0), Value::Int(5), Value::Int(10)])
    );
    // descending step
    assert_eq!(
        call_fn("range", vec![Value::Int(3), Value::Int(1), Value::Int(-1)]),
        Value::Array(vec![Value::Int(3), Value::Int(2), Value::Int(1)])
    );
    // zero step → NULL (no infinite loop)
    assert_eq!(
        call_fn("range", vec![Value::Int(1), Value::Int(5), Value::Int(0)]),
        Value::Null
    );
}

#[test]
fn list_fn_is_empty() {
    assert_eq!(
        call_fn("isEmpty", vec![Value::Array(vec![])]),
        Value::Bool(true)
    );
    assert_eq!(
        call_fn("isEmpty", vec![Value::Array(vec![Value::Int(1)])]),
        Value::Bool(false)
    );
    assert_eq!(call_fn("isEmpty", vec![s("")]), Value::Bool(true));
    assert_eq!(call_fn("isEmpty", vec![s("x")]), Value::Bool(false));
}

#[test]
fn reduce_folds_list() {
    // reduce(acc = 0, x IN [1,2,3,4] | acc + x) == 10
    let expr = Expr::Reduce {
        acc: "acc".into(),
        init: Box::new(Expr::Literal(Value::Int(0))),
        var: "x".into(),
        list: Box::new(Expr::List(vec![
            Expr::Literal(Value::Int(1)),
            Expr::Literal(Value::Int(2)),
            Expr::Literal(Value::Int(3)),
            Expr::Literal(Value::Int(4)),
        ])),
        expr: Box::new(Expr::BinaryOp {
            left: Box::new(Expr::Variable("acc".into())),
            op: BinaryOperator::Add,
            right: Box::new(Expr::Variable("x".into())),
        }),
    };
    assert_eq!(eval_expr(&expr, &empty_row()), Value::Int(10));
}

#[test]
fn reduce_empty_list_returns_init() {
    // Empty list → accumulator stays at its initial value.
    let expr = Expr::Reduce {
        acc: "acc".into(),
        init: Box::new(Expr::Literal(Value::Int(42))),
        var: "x".into(),
        list: Box::new(Expr::List(vec![])),
        expr: Box::new(Expr::Variable("x".into())),
    };
    assert_eq!(eval_expr(&expr, &empty_row()), Value::Int(42));
}

#[test]
fn reduce_null_list_is_null() {
    let expr = Expr::Reduce {
        acc: "acc".into(),
        init: Box::new(Expr::Literal(Value::Int(0))),
        var: "x".into(),
        list: Box::new(Expr::Literal(Value::Null)),
        expr: Box::new(Expr::Variable("acc".into())),
    };
    assert_eq!(eval_expr(&expr, &empty_row()), Value::Null);
}

#[test]
fn list_comprehension_forms() {
    let src = || {
        Box::new(Expr::List(vec![
            Expr::Literal(Value::Int(1)),
            Expr::Literal(Value::Int(2)),
            Expr::Literal(Value::Int(3)),
            Expr::Literal(Value::Int(4)),
        ]))
    };
    let gt2 = || {
        Box::new(Expr::BinaryOp {
            left: Box::new(Expr::Variable("x".into())),
            op: BinaryOperator::Gt,
            right: Box::new(Expr::Literal(Value::Int(2))),
        })
    };
    let square = || {
        Box::new(Expr::BinaryOp {
            left: Box::new(Expr::Variable("x".into())),
            op: BinaryOperator::Mul,
            right: Box::new(Expr::Variable("x".into())),
        })
    };

    // [x IN list] → the list unchanged.
    assert_eq!(
        eval_expr(
            &Expr::ListComprehension {
                var: "x".into(),
                list: src(),
                pred: None,
                map: None,
            },
            &empty_row(),
        ),
        Value::Array(vec![
            Value::Int(1),
            Value::Int(2),
            Value::Int(3),
            Value::Int(4)
        ])
    );

    // [x IN list WHERE x > 2] → [3, 4].
    assert_eq!(
        eval_expr(
            &Expr::ListComprehension {
                var: "x".into(),
                list: src(),
                pred: Some(gt2()),
                map: None,
            },
            &empty_row(),
        ),
        Value::Array(vec![Value::Int(3), Value::Int(4)])
    );

    // [x IN list | x*x] → [1, 4, 9, 16].
    assert_eq!(
        eval_expr(
            &Expr::ListComprehension {
                var: "x".into(),
                list: src(),
                pred: None,
                map: Some(square()),
            },
            &empty_row(),
        ),
        Value::Array(vec![
            Value::Int(1),
            Value::Int(4),
            Value::Int(9),
            Value::Int(16)
        ])
    );

    // [x IN list WHERE x > 2 | x*x] → [9, 16].
    assert_eq!(
        eval_expr(
            &Expr::ListComprehension {
                var: "x".into(),
                list: src(),
                pred: Some(gt2()),
                map: Some(square()),
            },
            &empty_row(),
        ),
        Value::Array(vec![Value::Int(9), Value::Int(16)])
    );
}

#[test]
fn list_predicate_quantifiers() {
    // list [2,4,6], predicate x > 3 → true for 4 and 6 (2 of 3).
    let mk = |kind| Expr::ListPredicate {
        kind,
        var: "x".into(),
        list: Box::new(Expr::List(vec![
            Expr::Literal(Value::Int(2)),
            Expr::Literal(Value::Int(4)),
            Expr::Literal(Value::Int(6)),
        ])),
        pred: Box::new(Expr::BinaryOp {
            left: Box::new(Expr::Variable("x".into())),
            op: BinaryOperator::Gt,
            right: Box::new(Expr::Literal(Value::Int(3))),
        }),
    };
    assert_eq!(
        eval_expr(&mk(ListPredicateKind::All), &empty_row()),
        Value::Bool(false)
    );
    assert_eq!(
        eval_expr(&mk(ListPredicateKind::Any), &empty_row()),
        Value::Bool(true)
    );
    assert_eq!(
        eval_expr(&mk(ListPredicateKind::None), &empty_row()),
        Value::Bool(false)
    );
    // two elements satisfy the predicate, so `single` is false.
    assert_eq!(
        eval_expr(&mk(ListPredicateKind::Single), &empty_row()),
        Value::Bool(false)
    );
}

#[test]
fn list_predicate_single_and_empty_list() {
    // single: exactly one element (5) is > 3.
    let single = Expr::ListPredicate {
        kind: ListPredicateKind::Single,
        var: "x".into(),
        list: Box::new(Expr::List(vec![
            Expr::Literal(Value::Int(1)),
            Expr::Literal(Value::Int(5)),
        ])),
        pred: Box::new(Expr::BinaryOp {
            left: Box::new(Expr::Variable("x".into())),
            op: BinaryOperator::Gt,
            right: Box::new(Expr::Literal(Value::Int(3))),
        }),
    };
    assert_eq!(eval_expr(&single, &empty_row()), Value::Bool(true));

    // Empty list: all/none → true (vacuous), any/single → false.
    let mk = |kind| Expr::ListPredicate {
        kind,
        var: "x".into(),
        list: Box::new(Expr::List(vec![])),
        pred: Box::new(Expr::Literal(Value::Bool(true))),
    };
    assert_eq!(
        eval_expr(&mk(ListPredicateKind::All), &empty_row()),
        Value::Bool(true)
    );
    assert_eq!(
        eval_expr(&mk(ListPredicateKind::Any), &empty_row()),
        Value::Bool(false)
    );
    assert_eq!(
        eval_expr(&mk(ListPredicateKind::None), &empty_row()),
        Value::Bool(true)
    );
    assert_eq!(
        eval_expr(&mk(ListPredicateKind::Single), &empty_row()),
        Value::Bool(false)
    );
}

#[test]
fn list_fn_keys() {
    // keys(n) on a node variable → its property keys (BTreeMap row → sorted),
    // internal __…__ markers excluded.
    let mut row = Row::new();
    row.insert("n.name".into(), s("Alice"));
    row.insert("n.age".into(), Value::Int(30));
    row.insert("n.__label__".into(), s("Person"));
    let v = eval_expr(
        &Expr::FunctionCall {
            name: "keys".into(),
            args: vec![Expr::Variable("n".into())],
            distinct: false,
        },
        &row,
    );
    assert_eq!(v, Value::Array(vec![s("age"), s("name")]));
}
