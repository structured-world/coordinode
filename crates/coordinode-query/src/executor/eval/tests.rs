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
fn eval_is_null() {
    let expr = Expr::IsNull {
        expr: Box::new(Expr::Literal(Value::Null)),
        negated: false,
    };
    let v = eval_expr(&expr, &empty_row());
    assert_eq!(v, Value::Bool(true));
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
