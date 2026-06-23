use super::*;

#[test]
fn substitute_simple_parameter() {
    let mut expr = Expr::Parameter("name".to_string());
    let mut params = HashMap::new();
    params.insert("name".to_string(), Value::String("Alice".to_string()));

    expr.substitute_params(&params);
    assert_eq!(expr, Expr::Literal(Value::String("Alice".to_string())));
}

#[test]
fn substitute_unknown_parameter_unchanged() {
    let mut expr = Expr::Parameter("missing".to_string());
    expr.substitute_params(&HashMap::new());
    assert_eq!(expr, Expr::Parameter("missing".to_string()));
}

#[test]
fn substitute_in_binary_op() {
    let mut expr = Expr::BinaryOp {
        left: Box::new(Expr::Variable("n.age".to_string())),
        op: BinaryOperator::Gt,
        right: Box::new(Expr::Parameter("min_age".to_string())),
    };
    let mut params = HashMap::new();
    params.insert("min_age".to_string(), Value::Int(18));

    expr.substitute_params(&params);

    if let Expr::BinaryOp { right, .. } = &expr {
        assert_eq!(**right, Expr::Literal(Value::Int(18)));
    } else {
        panic!("expected BinaryOp");
    }
}

#[test]
fn substitute_in_function_args() {
    let mut expr = Expr::FunctionCall {
        name: "vector_distance".to_string(),
        args: vec![
            Expr::Variable("n.embedding".to_string()),
            Expr::Parameter("query_vec".to_string()),
        ],
        distinct: false,
    };
    let mut params = HashMap::new();
    params.insert("query_vec".to_string(), Value::Vector(vec![1.0, 0.0]));

    expr.substitute_params(&params);

    if let Expr::FunctionCall { args, .. } = &expr {
        assert_eq!(args[1], Expr::Literal(Value::Vector(vec![1.0, 0.0])));
    } else {
        panic!("expected FunctionCall");
    }
}

#[test]
fn substitute_in_list() {
    let mut expr = Expr::List(vec![
        Expr::Literal(Value::Int(1)),
        Expr::Parameter("val".to_string()),
    ]);
    let mut params = HashMap::new();
    params.insert("val".to_string(), Value::Int(2));

    expr.substitute_params(&params);

    if let Expr::List(items) = &expr {
        assert_eq!(items[1], Expr::Literal(Value::Int(2)));
    } else {
        panic!("expected List");
    }
}

#[test]
fn substitute_nested_deep() {
    // NOT($param > 10)
    let mut expr = Expr::UnaryOp {
        op: UnaryOperator::Not,
        expr: Box::new(Expr::BinaryOp {
            left: Box::new(Expr::Parameter("x".to_string())),
            op: BinaryOperator::Gt,
            right: Box::new(Expr::Literal(Value::Int(10))),
        }),
    };
    let mut params = HashMap::new();
    params.insert("x".to_string(), Value::Int(5));

    expr.substitute_params(&params);

    if let Expr::UnaryOp {
        expr: inner_box, ..
    } = &expr
    {
        if let Expr::BinaryOp { left, .. } = inner_box.as_ref() {
            assert_eq!(**left, Expr::Literal(Value::Int(5)));
        } else {
            panic!("expected BinaryOp inside UnaryOp");
        }
    } else {
        panic!("expected UnaryOp");
    }
}

#[test]
fn substitute_leaves_literals_and_variables_untouched() {
    let mut expr = Expr::Literal(Value::Int(42));
    expr.substitute_params(&HashMap::from([("x".to_string(), Value::Int(99))]));
    assert_eq!(expr, Expr::Literal(Value::Int(42)));

    let mut expr = Expr::Variable("n".to_string());
    expr.substitute_params(&HashMap::from([("n".to_string(), Value::Int(99))]));
    assert_eq!(expr, Expr::Variable("n".to_string()));
}

// ── is_write ─────────────────────────────────────────────────────────────

/// Pure read queries (MATCH/RETURN) must NOT be classified as writes.
#[test]
fn is_write_read_only_queries() {
    let read_queries = [
        "MATCH (n) RETURN n",
        "MATCH (n:Person) WHERE n.age > 18 RETURN n.name",
        "MATCH (a)-[:KNOWS]->(b) RETURN a, b",
    ];
    for q in &read_queries {
        let ast = crate::cypher::parse(q).expect("parse");
        assert!(!ast.is_write(), "MATCH query must NOT be write: {q}");
    }
}

/// CREATE, MERGE, SET, DELETE, REMOVE must all be classified as writes.
#[test]
fn is_write_mutating_clauses() {
    let write_queries = [
        "CREATE (n:Person {name: 'Alice'})",
        "MERGE (n:Person {id: 1})",
        "MATCH (n) SET n.x = 1",
        "MATCH (n) DELETE n",
        "MATCH (n) REMOVE n.x",
    ];
    for q in &write_queries {
        let ast = crate::cypher::parse(q).expect("parse");
        assert!(ast.is_write(), "mutating query must be write: {q}");
    }
}

/// DDL statements (CREATE/DROP INDEX, ALTER LABEL) must be classified as writes.
#[test]
fn is_write_ddl_clauses() {
    let ddl_queries = [
        "CREATE TEXT INDEX article_body ON :Article(body)",
        "DROP TEXT INDEX article_body",
        "CREATE INDEX email_idx ON :User(email)",
        "DROP INDEX email_idx",
        "CREATE VECTOR INDEX emb_idx ON :Doc(embedding) OPTIONS {metric: \"cosine\"}",
        "DROP VECTOR INDEX emb_idx",
        "ALTER LABEL User SET SCHEMA VALIDATED",
    ];
    for q in &ddl_queries {
        let ast = crate::cypher::parse(q).expect("parse");
        assert!(ast.is_write(), "DDL query must be write: {q}");
    }
}

/// MATCH followed by a write clause must be classified as a write.
#[test]
fn is_write_match_then_write() {
    let queries = [
        "MATCH (n:Person {id: 1}) SET n.updated = true",
        "MATCH (n) WHERE n.active = false DELETE n",
        "MATCH (a), (b) MERGE (a)-[:KNOWS]->(b)",
    ];
    for q in &queries {
        let ast = crate::cypher::parse(q).expect("parse");
        assert!(
            ast.is_write(),
            "MATCH+write must be classified as write: {q}"
        );
    }
}
