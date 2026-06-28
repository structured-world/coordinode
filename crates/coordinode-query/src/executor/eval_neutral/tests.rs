use super::*;
use crate::cypher::ast::{BinaryOperator, Expr as CExpr, StringOp};
use crate::executor::eval::eval_expr;
use crate::planner::expr_lower::lower_expr;
use coordinode_core::graph::types::Value;

fn row() -> Row {
    let mut r = Row::new();
    r.insert("n".into(), Value::Int(7));
    r.insert("n.name".into(), Value::String("alice".into()));
    r.insert("n.age".into(), Value::Int(30));
    r
}

/// The neutral evaluator must match the cypher evaluator for any expression that
/// lowers cleanly: `eval_expr(c) == eval_neutral(lower(c))`. This is the
/// correctness anchor for the whole neutral path.
fn assert_same(c: &CExpr) {
    let r = row();
    let via_cypher = eval_expr(c, &r);
    let neutral = lower_expr(c).expect("lower");
    let via_neutral = eval_neutral(&neutral, &r);
    assert_eq!(via_cypher, via_neutral, "mismatch for {c:?}");
}

fn lit(i: i64) -> CExpr {
    CExpr::Literal(Value::Int(i))
}

#[test]
fn atoms_and_variable_and_property() {
    assert_same(&lit(42));
    assert_same(&CExpr::Variable("n".into()));
    assert_same(&CExpr::Variable("missing".into()));
    assert_same(&CExpr::PropertyAccess {
        expr: Box::new(CExpr::Variable("n".into())),
        property: "name".into(),
    });
}

#[test]
fn arithmetic_and_comparison_and_logic() {
    for op in [
        BinaryOperator::Add,
        BinaryOperator::Sub,
        BinaryOperator::Mul,
        BinaryOperator::Div,
        BinaryOperator::Modulo,
        BinaryOperator::Eq,
        BinaryOperator::Neq,
        BinaryOperator::Lt,
        BinaryOperator::Gte,
    ] {
        assert_same(&CExpr::BinaryOp {
            left: Box::new(lit(7)),
            op,
            right: Box::new(lit(3)),
        });
    }
    assert_same(&CExpr::BinaryOp {
        left: Box::new(CExpr::Literal(Value::Bool(true))),
        op: BinaryOperator::And,
        right: Box::new(CExpr::Literal(Value::Bool(false))),
    });
    assert_same(&CExpr::UnaryOp {
        op: crate::cypher::ast::UnaryOperator::Not,
        expr: Box::new(CExpr::Literal(Value::Bool(true))),
    });
}

#[test]
fn collections_in_isnull_subscript_slice() {
    let list = CExpr::List(vec![lit(1), lit(2), lit(3)]);
    assert_same(&list);
    assert_same(&CExpr::In {
        expr: Box::new(lit(2)),
        list: Box::new(list.clone()),
    });
    assert_same(&CExpr::In {
        expr: Box::new(lit(9)),
        list: Box::new(list.clone()),
    });
    assert_same(&CExpr::IsNull {
        expr: Box::new(CExpr::Variable("missing".into())),
        negated: false,
    });
    assert_same(&CExpr::Subscript {
        expr: Box::new(list.clone()),
        index: Box::new(lit(0)),
    });
    assert_same(&CExpr::Subscript {
        expr: Box::new(list.clone()),
        index: Box::new(lit(-1)),
    });
    assert_same(&CExpr::Slice {
        expr: Box::new(list),
        start: Some(Box::new(lit(1))),
        end: None,
    });
}

#[test]
fn string_match_and_case_and_istyped() {
    assert_same(&CExpr::StringMatch {
        expr: Box::new(CExpr::Variable("n.name".into())),
        op: StringOp::StartsWith,
        pattern: Box::new(CExpr::Literal(Value::String("al".into()))),
    });
    assert_same(&CExpr::StringMatch {
        expr: Box::new(CExpr::Variable("n.name".into())),
        op: StringOp::Contains,
        pattern: Box::new(CExpr::Literal(Value::String("zz".into()))),
    });
    assert_same(&CExpr::Case {
        operand: None,
        when_clauses: vec![(
            CExpr::BinaryOp {
                left: Box::new(CExpr::Variable("n.age".into())),
                op: BinaryOperator::Gt,
                right: Box::new(lit(18)),
            },
            CExpr::Literal(Value::String("adult".into())),
        )],
        else_clause: Some(Box::new(CExpr::Literal(Value::String("minor".into())))),
    });
    assert_same(&CExpr::IsTyped {
        expr: Box::new(CExpr::Variable("n".into())),
        type_name: "INTEGER".into(),
        negated: false,
    });
}

#[test]
fn comprehension_quantifier_reduce_function() {
    let list = CExpr::List(vec![lit(1), lit(2), lit(3)]);
    assert_same(&CExpr::ListComprehension {
        var: "x".into(),
        list: Box::new(list.clone()),
        pred: None,
        map: Some(Box::new(CExpr::BinaryOp {
            left: Box::new(CExpr::Variable("x".into())),
            op: BinaryOperator::Mul,
            right: Box::new(lit(2)),
        })),
    });
    assert_same(&CExpr::ListPredicate {
        kind: crate::cypher::ast::ListPredicateKind::Any,
        var: "x".into(),
        list: Box::new(list.clone()),
        pred: Box::new(CExpr::BinaryOp {
            left: Box::new(CExpr::Variable("x".into())),
            op: BinaryOperator::Gt,
            right: Box::new(lit(2)),
        }),
    });
    assert_same(&CExpr::Reduce {
        acc: "a".into(),
        init: Box::new(lit(0)),
        var: "x".into(),
        list: Box::new(list),
        expr: Box::new(CExpr::BinaryOp {
            left: Box::new(CExpr::Variable("a".into())),
            op: BinaryOperator::Add,
            right: Box::new(CExpr::Variable("x".into())),
        }),
    });
    // Scalar function via the shared dispatch.
    assert_same(&CExpr::FunctionCall {
        name: "toString".into(),
        args: vec![CExpr::Variable("n".into())],
        distinct: false,
    });
    // Entity-introspection function using the first-arg variable name.
    assert_same(&CExpr::FunctionCall {
        name: "labels".into(),
        args: vec![CExpr::Variable("n".into())],
        distinct: false,
    });
}
