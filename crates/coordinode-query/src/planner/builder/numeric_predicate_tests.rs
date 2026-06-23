use super::*;
use coordinode_core::graph::types::Value;

#[test]
fn collect_simple_predicates_extracts_numeric_ge() {
    use crate::cypher::ast::BinaryOperator;
    use crate::planner::logical::{NumericCmp, VectorPredicate};

    let pred = Expr::BinaryOp {
        left: Box::new(Expr::PropertyAccess {
            expr: Box::new(Expr::Variable("n".into())),
            property: "score".into(),
        }),
        op: BinaryOperator::Gte,
        right: Box::new(Expr::Literal(Value::Float(0.7))),
    };
    let scan = LogicalOp::NodeScan {
        variable: "n".into(),
        labels: vec!["Item".into()],
        property_filters: vec![],
    };
    let filter = LogicalOp::Filter {
        input: Box::new(scan),
        predicate: pred,
    };
    let mut leaves: Vec<VectorPredicate> = Vec::new();
    collect_simple_property_predicates(&filter, "n", &mut leaves);
    assert_eq!(leaves.len(), 1);
    match &leaves[0] {
        VectorPredicate::PropertyCmp {
            property,
            op,
            value,
        } => {
            assert_eq!(property, "score");
            assert_eq!(*op, NumericCmp::Ge);
            assert_eq!(value, &Value::Float(0.7));
        }
        other => panic!("expected PropertyCmp, got {other:?}"),
    }
}

#[test]
fn collect_simple_predicates_flips_reversed_numeric_cmp() {
    use crate::cypher::ast::BinaryOperator;
    use crate::planner::logical::{NumericCmp, VectorPredicate};

    // 100 <= n.id → PropertyCmp { op: Ge }
    let pred = Expr::BinaryOp {
        left: Box::new(Expr::Literal(Value::Int(100))),
        op: BinaryOperator::Lte,
        right: Box::new(Expr::PropertyAccess {
            expr: Box::new(Expr::Variable("n".into())),
            property: "id".into(),
        }),
    };
    let scan = LogicalOp::NodeScan {
        variable: "n".into(),
        labels: vec!["Item".into()],
        property_filters: vec![],
    };
    let filter = LogicalOp::Filter {
        input: Box::new(scan),
        predicate: pred,
    };
    let mut leaves: Vec<VectorPredicate> = Vec::new();
    collect_simple_property_predicates(&filter, "n", &mut leaves);
    assert_eq!(leaves.len(), 1);
    match &leaves[0] {
        VectorPredicate::PropertyCmp { op, .. } => assert_eq!(*op, NumericCmp::Ge),
        other => panic!("expected PropertyCmp, got {other:?}"),
    }
}

#[test]
fn collect_simple_predicates_accepts_negative_literal() {
    use crate::cypher::ast::{BinaryOperator, UnaryOperator};
    use crate::planner::logical::VectorPredicate;

    let pred = Expr::BinaryOp {
        left: Box::new(Expr::PropertyAccess {
            expr: Box::new(Expr::Variable("n".into())),
            property: "balance".into(),
        }),
        op: BinaryOperator::Gt,
        right: Box::new(Expr::UnaryOp {
            op: UnaryOperator::Neg,
            expr: Box::new(Expr::Literal(Value::Int(50))),
        }),
    };
    let scan = LogicalOp::NodeScan {
        variable: "n".into(),
        labels: vec!["Item".into()],
        property_filters: vec![],
    };
    let filter = LogicalOp::Filter {
        input: Box::new(scan),
        predicate: pred,
    };
    let mut leaves: Vec<VectorPredicate> = Vec::new();
    collect_simple_property_predicates(&filter, "n", &mut leaves);
    assert_eq!(leaves.len(), 1);
    match &leaves[0] {
        VectorPredicate::PropertyCmp { value, .. } => {
            assert_eq!(value, &Value::Int(-50));
        }
        other => panic!("expected PropertyCmp, got {other:?}"),
    }
}
