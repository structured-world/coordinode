use super::*;

// --- EdgeVectorStrategy selection tests ---

/// Fan-out < 200 → always Graph-first regardless of selectivity.
#[test]
fn strategy_low_fanout_graph_first() {
    assert_eq!(
        select_edge_vector_strategy(50.0, 0.5),
        EdgeVectorStrategy::GraphFirst
    );
    assert_eq!(
        select_edge_vector_strategy(199.0, 0.001),
        EdgeVectorStrategy::GraphFirst
    );
    assert_eq!(
        select_edge_vector_strategy(1.0, 0.001),
        EdgeVectorStrategy::GraphFirst
    );
}

/// Fan-out > 10K → always Vector-first regardless of selectivity.
#[test]
fn strategy_high_fanout_vector_first() {
    assert_eq!(
        select_edge_vector_strategy(10_001.0, 0.5),
        EdgeVectorStrategy::VectorFirst
    );
    assert_eq!(
        select_edge_vector_strategy(100_000.0, 0.99),
        EdgeVectorStrategy::VectorFirst
    );
}

/// Fan-out 200-10K + selectivity > 1% → Graph-first.
#[test]
fn strategy_mid_fanout_high_selectivity_graph_first() {
    assert_eq!(
        select_edge_vector_strategy(500.0, 0.05),
        EdgeVectorStrategy::GraphFirst
    );
    assert_eq!(
        select_edge_vector_strategy(5_000.0, 0.5),
        EdgeVectorStrategy::GraphFirst
    );
}

/// Fan-out 200-10K + selectivity < 1% → Vector-first.
#[test]
fn strategy_mid_fanout_low_selectivity_vector_first() {
    assert_eq!(
        select_edge_vector_strategy(500.0, 0.005),
        EdgeVectorStrategy::VectorFirst
    );
    assert_eq!(
        select_edge_vector_strategy(9_999.0, 0.001),
        EdgeVectorStrategy::VectorFirst
    );
}

/// Boundary: fan-out exactly 200 → mid range, depends on selectivity.
#[test]
fn strategy_boundary_200() {
    // At exactly 200, enters the mid range
    assert_eq!(
        select_edge_vector_strategy(200.0, 0.5),
        EdgeVectorStrategy::GraphFirst
    );
    assert_eq!(
        select_edge_vector_strategy(200.0, 0.005),
        EdgeVectorStrategy::VectorFirst
    );
}

/// Boundary: fan-out exactly 10K → mid range.
#[test]
fn strategy_boundary_10k() {
    assert_eq!(
        select_edge_vector_strategy(10_000.0, 0.5),
        EdgeVectorStrategy::GraphFirst
    );
    assert_eq!(
        select_edge_vector_strategy(10_000.0, 0.005),
        EdgeVectorStrategy::VectorFirst
    );
}

/// Display formatting for strategies.
#[test]
fn strategy_display() {
    assert_eq!(
        EdgeVectorStrategy::GraphFirst.to_string(),
        "Graph-First (brute-force)"
    );
    assert_eq!(
        EdgeVectorStrategy::VectorFirst.to_string(),
        "Vector-First (HNSW)"
    );
}

// --- VectorPredicate descriptor tests ---

#[test]
fn vector_predicate_label_eq_constructs() {
    let p = VectorPredicate::LabelEq("Item".into());
    match p {
        VectorPredicate::LabelEq(l) => assert_eq!(l, "Item"),
        _ => panic!("expected LabelEq"),
    }
}

#[test]
fn vector_predicate_property_eq_carries_value() {
    let p = VectorPredicate::PropertyEq {
        property: "category".into(),
        value: coordinode_core::graph::types::Value::String("electronics".into()),
    };
    match p {
        VectorPredicate::PropertyEq { property, value } => {
            assert_eq!(property, "category");
            assert_eq!(
                value,
                coordinode_core::graph::types::Value::String("electronics".into())
            );
        }
        _ => panic!("expected PropertyEq"),
    }
}

#[test]
fn vector_predicate_and_nests() {
    let p = VectorPredicate::And(
        Box::new(VectorPredicate::LabelEq("Item".into())),
        Box::new(VectorPredicate::PropertyEq {
            property: "active".into(),
            value: coordinode_core::graph::types::Value::Bool(true),
        }),
    );
    match p {
        VectorPredicate::And(left, right) => {
            assert!(matches!(*left, VectorPredicate::LabelEq(_)));
            assert!(matches!(*right, VectorPredicate::PropertyEq { .. }));
        }
        _ => panic!("expected And"),
    }
}

#[test]
fn vector_top_k_predicate_defaults_to_none() {
    let op = LogicalOp::VectorTopK {
        input: Box::new(LogicalOp::Empty),
        vector_expr: Expr::Variable("n".into()),
        query_vector: Expr::Variable("q".into()),
        function: "vector_distance".into(),
        k: 5,
        distance_alias: None,
        hnsw_index: None,
        predicate: None,
    };
    if let LogicalOp::VectorTopK { predicate, .. } = op {
        assert!(predicate.is_none());
    } else {
        panic!("expected VectorTopK");
    }
}
