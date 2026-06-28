use super::*;
use crate::cypher::ast::Expr as CExpr;
use crate::cypher::parse;
use crate::plan::expr::{BinOp, Expr as PExpr, MapProjItem, Quantifier, StrOp, UnOp};
use coordinode_core::graph::types::Value;

fn lit(i: i64) -> CExpr {
    CExpr::Literal(Value::Int(i))
}

fn lower(e: &CExpr) -> PExpr {
    lower_expr(e).expect("lower")
}

#[test]
fn lowers_atoms() {
    assert_eq!(lower(&lit(7)), PExpr::Literal(Value::Int(7)));
    assert_eq!(
        lower(&CExpr::Parameter("p".into())),
        PExpr::Parameter("p".into())
    );
    assert_eq!(
        lower(&CExpr::Variable("n".into())),
        PExpr::Variable("n".into())
    );
    assert_eq!(lower(&CExpr::Star), PExpr::Star);
}

#[test]
fn lowers_property_access() {
    let e = CExpr::PropertyAccess {
        expr: Box::new(CExpr::Variable("n".into())),
        property: "age".into(),
    };
    match lower(&e) {
        PExpr::Property { base, key } => {
            assert_eq!(*base, PExpr::Variable("n".into()));
            assert_eq!(key, "age");
        }
        other => panic!("expected Property, got {other:?}"),
    }
}

#[test]
fn lowers_every_binary_operator() {
    use crate::cypher::ast::BinaryOperator as B;
    let pairs = [
        (B::Add, BinOp::Add),
        (B::Sub, BinOp::Sub),
        (B::Mul, BinOp::Mul),
        (B::Div, BinOp::Div),
        (B::Modulo, BinOp::Modulo),
        (B::Eq, BinOp::Eq),
        (B::Neq, BinOp::Neq),
        (B::Lt, BinOp::Lt),
        (B::Lte, BinOp::Lte),
        (B::Gt, BinOp::Gt),
        (B::Gte, BinOp::Gte),
        (B::And, BinOp::And),
        (B::Or, BinOp::Or),
        (B::Xor, BinOp::Xor),
    ];
    for (cop, pop) in pairs {
        let e = CExpr::BinaryOp {
            left: Box::new(lit(1)),
            op: cop,
            right: Box::new(lit(2)),
        };
        match lower(&e) {
            PExpr::Binary { op, .. } => assert_eq!(op, pop, "binop {cop:?}"),
            other => panic!("expected Binary, got {other:?}"),
        }
    }
}

#[test]
fn lowers_unary_and_string_and_quantifier_operators() {
    use crate::cypher::ast::{ListPredicateKind, StringOp, UnaryOperator};
    match lower(&CExpr::UnaryOp {
        op: UnaryOperator::Not,
        expr: Box::new(lit(0)),
    }) {
        PExpr::Unary { op: UnOp::Not, .. } => {}
        other => panic!("expected Unary Not, got {other:?}"),
    }
    match lower(&CExpr::StringMatch {
        expr: Box::new(CExpr::Variable("s".into())),
        op: StringOp::Contains,
        pattern: Box::new(CExpr::Literal(Value::String("x".into()))),
    }) {
        PExpr::StringMatch {
            op: StrOp::Contains,
            ..
        } => {}
        other => panic!("expected StringMatch Contains, got {other:?}"),
    }
    match lower(&CExpr::ListPredicate {
        kind: ListPredicateKind::Single,
        var: "x".into(),
        list: Box::new(CExpr::List(vec![lit(1)])),
        pred: Box::new(CExpr::Variable("x".into())),
    }) {
        PExpr::ListQuantifier {
            kind: Quantifier::Single,
            var,
            ..
        } => assert_eq!(var, "x"),
        other => panic!("expected ListQuantifier Single, got {other:?}"),
    }
}

#[test]
fn lowers_collections_and_map_projection() {
    assert_eq!(
        lower(&CExpr::List(vec![lit(1), lit(2)])),
        PExpr::List(vec![
            PExpr::Literal(Value::Int(1)),
            PExpr::Literal(Value::Int(2))
        ])
    );
    match lower(&CExpr::MapLiteral(vec![("k".into(), lit(9))])) {
        PExpr::Map(entries) => {
            assert_eq!(entries.len(), 1);
            assert_eq!(entries[0].0, "k");
        }
        other => panic!("expected Map, got {other:?}"),
    }
    let proj = CExpr::MapProjection {
        expr: Box::new(CExpr::Variable("n".into())),
        items: vec![
            crate::cypher::ast::MapProjectionItem::Property("name".into()),
            crate::cypher::ast::MapProjectionItem::Computed("two".into(), lit(2)),
        ],
    };
    match lower(&proj) {
        PExpr::MapProjection { items, .. } => {
            assert!(matches!(items[0], MapProjItem::Property(ref n) if n == "name"));
            assert!(matches!(items[1], MapProjItem::Computed(ref a, _) if a == "two"));
        }
        other => panic!("expected MapProjection, got {other:?}"),
    }
}

#[test]
fn lowers_case_and_comprehension_and_reduce() {
    let case = CExpr::Case {
        operand: None,
        when_clauses: vec![(CExpr::Literal(Value::Bool(true)), lit(1))],
        else_clause: Some(Box::new(lit(0))),
    };
    match lower(&case) {
        PExpr::Case {
            operand: None,
            branches,
            otherwise: Some(_),
        } => assert_eq!(branches.len(), 1),
        other => panic!("expected Case, got {other:?}"),
    }
    let reduce = CExpr::Reduce {
        acc: "a".into(),
        init: Box::new(lit(0)),
        var: "x".into(),
        list: Box::new(CExpr::List(vec![lit(1)])),
        expr: Box::new(CExpr::Variable("a".into())),
    };
    assert!(matches!(lower(&reduce), PExpr::Reduce { .. }));

    let comp = CExpr::ListComprehension {
        var: "x".into(),
        list: Box::new(CExpr::List(vec![lit(1), lit(2)])),
        pred: Some(Box::new(CExpr::Variable("x".into()))),
        map: Some(Box::new(CExpr::Variable("x".into()))),
    };
    match lower(&comp) {
        PExpr::ListComprehension {
            var,
            filter: Some(_),
            map: Some(_),
            ..
        } => assert_eq!(var, "x"),
        other => panic!("expected ListComprehension, got {other:?}"),
    }
}

#[test]
fn lowers_remaining_scalar_variants() {
    use crate::cypher::ast::StringOp;
    // In
    match lower(&CExpr::In {
        expr: Box::new(lit(1)),
        list: Box::new(CExpr::List(vec![lit(1), lit(2)])),
    }) {
        PExpr::In { .. } => {}
        other => panic!("expected In, got {other:?}"),
    }
    // IsNull (negated) and IsTyped
    match lower(&CExpr::IsNull {
        expr: Box::new(CExpr::Variable("n".into())),
        negated: true,
    }) {
        PExpr::IsNull { negated: true, .. } => {}
        other => panic!("expected IsNull negated, got {other:?}"),
    }
    match lower(&CExpr::IsTyped {
        expr: Box::new(CExpr::Variable("n".into())),
        type_name: "INTEGER".into(),
        negated: false,
    }) {
        PExpr::IsTyped {
            type_name,
            negated: false,
            ..
        } => assert_eq!(type_name, "INTEGER"),
        other => panic!("expected IsTyped, got {other:?}"),
    }
    // Subscript and Slice
    match lower(&CExpr::Subscript {
        expr: Box::new(CExpr::Variable("l".into())),
        index: Box::new(lit(0)),
    }) {
        PExpr::Subscript { .. } => {}
        other => panic!("expected Subscript, got {other:?}"),
    }
    match lower(&CExpr::Slice {
        expr: Box::new(CExpr::Variable("l".into())),
        start: Some(Box::new(lit(1))),
        end: None,
    }) {
        PExpr::Slice {
            start: Some(_),
            end: None,
            ..
        } => {}
        other => panic!("expected Slice, got {other:?}"),
    }
    // FunctionCall (distinct preserved)
    match lower(&CExpr::FunctionCall {
        name: "count".into(),
        args: vec![CExpr::Variable("n".into())],
        distinct: true,
    }) {
        PExpr::Call {
            name,
            distinct: true,
            args,
        } => {
            assert_eq!(name, "count");
            assert_eq!(args.len(), 1);
        }
        other => panic!("expected Call, got {other:?}"),
    }
    // The remaining StringOp variants beyond Contains.
    for (cop, pop) in [
        (StringOp::StartsWith, StrOp::StartsWith),
        (StringOp::EndsWith, StrOp::EndsWith),
        (StringOp::Regex, StrOp::Regex),
    ] {
        match lower(&CExpr::StringMatch {
            expr: Box::new(CExpr::Variable("s".into())),
            op: cop,
            pattern: Box::new(CExpr::Literal(Value::String("x".into()))),
        }) {
            PExpr::StringMatch { op, .. } => assert_eq!(op, pop),
            other => panic!("expected StringMatch, got {other:?}"),
        }
    }
}

/// Subqueries lower to a pre-built neutral subplan (parsed from real Cypher so
/// the embedded `MatchClause` is realistic, then lowered from the WHERE expr).
#[test]
fn lowers_exists_subquery_to_subplan() {
    let query =
        parse("MATCH (a:Person) WHERE EXISTS { MATCH (a)-[:KNOWS]->(b) } RETURN a").expect("parse");
    let where_expr = match &query.clauses[0] {
        crate::cypher::ast::Clause::Match(mc) => mc.where_clause.as_ref().expect("where present"),
        other => panic!("expected Match clause, got {other:?}"),
    };
    match lower(where_expr) {
        PExpr::ExistsSubplan(_) => {}
        other => panic!("expected ExistsSubplan, got {other:?}"),
    }
}

/// A pattern predicate in WHERE also lowers to an existence subplan.
#[test]
fn lowers_pattern_predicate_to_exists_subplan() {
    let query = parse("MATCH (a:Person) WHERE (a)-[:KNOWS]->(:Person) RETURN a").expect("parse");
    let where_expr = match &query.clauses[0] {
        crate::cypher::ast::Clause::Match(mc) => mc.where_clause.as_ref().expect("where"),
        other => panic!("expected Match, got {other:?}"),
    };
    // The WHERE may be the bare pattern predicate or wrapped; lower and assert
    // it contains an ExistsSubplan at the root or as an operand.
    let lowered = lower(where_expr);
    assert!(
        contains_exists_subplan(&lowered),
        "expected an ExistsSubplan somewhere in {lowered:?}"
    );
}

fn contains_exists_subplan(e: &PExpr) -> bool {
    match e {
        PExpr::ExistsSubplan(_) => true,
        PExpr::Unary { operand, .. } => contains_exists_subplan(operand),
        PExpr::Binary { left, right, .. } => {
            contains_exists_subplan(left) || contains_exists_subplan(right)
        }
        _ => false,
    }
}

/// Count / collect subqueries and pattern comprehensions each lower to their
/// neutral subplan form. Built from a real parsed match clause + pattern so the
/// embedded plan is realistic without hand-constructing AST patterns.
#[test]
fn lowers_count_collect_and_pattern_comprehension() {
    let query = parse("MATCH (a)-[:R]->(b) RETURN a").expect("parse");
    let mc = match &query.clauses[0] {
        crate::cypher::ast::Clause::Match(mc) => mc.clone(),
        other => panic!("expected Match, got {other:?}"),
    };

    let count = CExpr::CountSubquery(Box::new(mc.clone()));
    assert!(
        matches!(lower(&count), PExpr::CountSubplan(_)),
        "COUNT subquery lowers to a count subplan"
    );

    let collect = CExpr::CollectSubquery {
        match_clause: Box::new(mc.clone()),
        expr: Box::new(CExpr::Variable("b".into())),
    };
    match lower(&collect) {
        PExpr::CollectSubplan { projection, .. } => {
            assert_eq!(*projection, PExpr::Variable("b".into()));
        }
        other => panic!("expected CollectSubplan, got {other:?}"),
    }

    let pattern = mc.patterns[0].clone();
    let comprehension = CExpr::PatternComprehension {
        pattern: Box::new(pattern),
        where_clause: None,
        map: Box::new(CExpr::Variable("b".into())),
    };
    match lower(&comprehension) {
        PExpr::PatternComprehension { map, .. } => {
            assert_eq!(*map, PExpr::Variable("b".into()));
        }
        other => panic!("expected PatternComprehension, got {other:?}"),
    }
}
