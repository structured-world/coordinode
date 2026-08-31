//! Direct smoke tests for the neutral evaluator.
//!
//! Comprehensive expression semantics (every operator, string / list / map
//! function, comprehension, quantifier, reduce, case) are covered by
//! `executor::eval::tests`, which drives this same evaluator through
//! `lower_expr` + `eval_neutral` (the production evaluation path). These tests
//! exercise `eval_neutral` directly on neutral `Expr` nodes, independent of the
//! lowering layer.

use super::*;
use crate::plan::expr::{BinOp, Expr};
use coordinode_core::graph::types::Value;

fn row() -> Row {
    let mut r = Row::new();
    r.insert("n".into(), Value::Int(7));
    r.insert("n.name".into(), Value::String("alice".into()));
    r
}

#[test]
fn literal_variable_and_property() {
    let r = row();
    // Unwrapped throughout: these expressions carry no arithmetic that can
    // fail, so an error here is a bug rather than an outcome under test. The
    // failing cases live in the eval module's own tests.
    assert_eq!(
        eval_neutral(&Expr::Literal(Value::Int(42)), &r).unwrap(),
        Value::Int(42)
    );
    assert_eq!(
        eval_neutral(&Expr::Variable("n".into()), &r).unwrap(),
        Value::Int(7)
    );
    assert_eq!(
        eval_neutral(&Expr::Variable("missing".into()), &r).unwrap(),
        Value::Null
    );
    let prop = Expr::Property {
        base: Box::new(Expr::Variable("n".into())),
        key: "name".into(),
    };
    assert_eq!(
        eval_neutral(&prop, &r).unwrap(),
        Value::String("alice".into())
    );
}

#[test]
fn binary_arithmetic_and_comparison() {
    let r = row();
    let add = Expr::Binary {
        left: Box::new(Expr::Variable("n".into())),
        op: BinOp::Add,
        right: Box::new(Expr::Literal(Value::Int(3))),
    };
    assert_eq!(eval_neutral(&add, &r).unwrap(), Value::Int(10));

    let gt = Expr::Binary {
        left: Box::new(Expr::Variable("n".into())),
        op: BinOp::Gt,
        right: Box::new(Expr::Literal(Value::Int(3))),
    };
    assert_eq!(eval_neutral(&gt, &r).unwrap(), Value::Bool(true));
}

#[test]
fn list_and_in() {
    let r = row();
    let list = Expr::List(vec![
        Expr::Literal(Value::Int(1)),
        Expr::Literal(Value::Int(2)),
        Expr::Literal(Value::Int(3)),
    ]);
    assert_eq!(
        eval_neutral(&list, &r).unwrap(),
        Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)])
    );
    let in_present = Expr::In {
        item: Box::new(Expr::Literal(Value::Int(2))),
        list: Box::new(list.clone()),
    };
    assert_eq!(eval_neutral(&in_present, &r).unwrap(), Value::Bool(true));
    let in_absent = Expr::In {
        item: Box::new(Expr::Literal(Value::Int(9))),
        list: Box::new(list),
    };
    assert_eq!(eval_neutral(&in_absent, &r).unwrap(), Value::Bool(false));
}
