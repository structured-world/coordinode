use super::*;
use crate::cypher::ast::*;
use coordinode_core::graph::types::Value;

fn node_scan(var: &str, label: &str) -> LogicalOp {
    LogicalOp::NodeScan {
        variable: var.to_string(),
        labels: vec![label.to_string()],
        property_filters: vec![],
    }
}

fn filter(input: LogicalOp, predicate: Expr) -> LogicalOp {
    LogicalOp::Filter {
        input: Box::new(input),
        predicate,
    }
}

fn prop_eq(var: &str, prop: &str) -> Expr {
    Expr::BinaryOp {
        left: Box::new(Expr::PropertyAccess {
            expr: Box::new(Expr::Variable(var.to_string())),
            property: prop.to_string(),
        }),
        op: BinaryOperator::Eq,
        right: Box::new(Expr::Literal(Value::String("value".to_string()))),
    }
}

fn traverse_unbounded(input: LogicalOp) -> LogicalOp {
    LogicalOp::Traverse {
        input: Box::new(input),
        source: "a".to_string(),
        edge_types: vec!["KNOWS".to_string()],
        direction: Direction::Outgoing,
        target_variable: "b".to_string(),
        target_labels: vec![],
        length: Some(LengthBound {
            min: Some(1),
            max: None,
        }),
        edge_variable: None,
        target_filters: vec![],
        edge_filters: vec![],
        temporal_filter: None,
        path_variable: None,
    }
}

fn traverse_bounded(input: LogicalOp) -> LogicalOp {
    LogicalOp::Traverse {
        input: Box::new(input),
        source: "a".to_string(),
        edge_types: vec!["KNOWS".to_string()],
        direction: Direction::Outgoing,
        target_variable: "b".to_string(),
        target_labels: vec![],
        length: Some(LengthBound {
            min: Some(1),
            max: Some(5),
        }),
        edge_variable: None,
        target_filters: vec![],
        edge_filters: vec![],
        temporal_filter: None,
        path_variable: None,
    }
}

fn plan(root: LogicalOp) -> LogicalPlan {
    LogicalPlan {
        root,
        snapshot_ts: None,
        vector_consistency: coordinode_core::graph::types::VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
    }
}

// --- MissingIndex tests ---

/// Filter on NodeScan property → suggests CREATE INDEX.
#[test]
fn missing_index_detected() {
    let p = plan(filter(node_scan("n", "User"), prop_eq("n", "email")));
    let suggestions = detect_suggestions(&p, None);

    assert_eq!(suggestions.len(), 1);
    assert_eq!(suggestions[0].kind, SuggestionKind::CreateIndex);
    assert_eq!(suggestions[0].severity, Severity::Critical);
    assert!(suggestions[0].explanation.contains("User"));
    assert!(suggestions[0].explanation.contains("email"));
    assert!(suggestions[0]
        .ddl
        .as_ref()
        .unwrap()
        .contains("ON User(email)"));
}

/// NodeScan without Filter → no suggestions.
#[test]
fn no_missing_index_without_filter() {
    let p = plan(node_scan("n", "User"));
    let suggestions = detect_suggestions(&p, None);
    assert!(suggestions.is_empty());
}

/// Filter on wrong variable → no suggestions (not scanning that label).
#[test]
fn no_missing_index_different_variable() {
    let p = plan(filter(node_scan("n", "User"), prop_eq("m", "email")));
    let suggestions = detect_suggestions(&p, None);
    assert!(suggestions.is_empty());
}

// --- G022: IndexRegistry cross-check tests ---

/// Index exists for (label, property) → no MissingIndex suggestion.
#[test]
fn no_missing_index_when_index_exists() {
    use crate::index::{IndexDefinition, IndexRegistry};

    let reg = IndexRegistry::new();
    reg.register_in_memory(IndexDefinition::btree("user_email", "User", "email"));

    let p = plan(filter(node_scan("n", "User"), prop_eq("n", "email")));
    let suggestions = detect_suggestions(&p, Some(&reg));

    // No MissingIndex suggestion because index exists
    assert!(
        suggestions
            .iter()
            .all(|s| s.kind != SuggestionKind::CreateIndex),
        "should not suggest CreateIndex when index already exists: {suggestions:?}"
    );
}

/// Index exists for different property → still suggests for the unindexed one.
#[test]
fn missing_index_for_unindexed_property() {
    use crate::index::{IndexDefinition, IndexRegistry};

    let reg = IndexRegistry::new();
    reg.register_in_memory(IndexDefinition::btree("user_name", "User", "name"));

    // Filtering on "email" which is NOT indexed (only "name" is)
    let p = plan(filter(node_scan("n", "User"), prop_eq("n", "email")));
    let suggestions = detect_suggestions(&p, Some(&reg));

    assert_eq!(suggestions.len(), 1);
    assert_eq!(suggestions[0].kind, SuggestionKind::CreateIndex);
    assert!(suggestions[0].explanation.contains("email"));
}

/// Index for different label → still suggests (no cross-label suppression).
#[test]
fn missing_index_different_label() {
    use crate::index::{IndexDefinition, IndexRegistry};

    let reg = IndexRegistry::new();
    reg.register_in_memory(IndexDefinition::btree("post_title", "Post", "email"));

    // Filtering User.email but only Post.email is indexed
    let p = plan(filter(node_scan("n", "User"), prop_eq("n", "email")));
    let suggestions = detect_suggestions(&p, Some(&reg));

    assert_eq!(suggestions.len(), 1);
    assert_eq!(suggestions[0].kind, SuggestionKind::CreateIndex);
}

/// Empty registry behaves like None — suggests indexes.
#[test]
fn missing_index_with_empty_registry() {
    use crate::index::IndexRegistry;

    let reg = IndexRegistry::new();
    let p = plan(filter(node_scan("n", "User"), prop_eq("n", "email")));
    let suggestions = detect_suggestions(&p, Some(&reg));

    assert_eq!(suggestions.len(), 1);
    assert_eq!(suggestions[0].kind, SuggestionKind::CreateIndex);
}

// --- UnboundedTraversal tests ---

/// Variable-length path without max → suggests adding bound.
#[test]
fn unbounded_traversal_detected() {
    let p = plan(traverse_unbounded(node_scan("a", "Person")));
    let suggestions = detect_suggestions(&p, None);

    assert_eq!(suggestions.len(), 1);
    assert_eq!(suggestions[0].kind, SuggestionKind::AddDepthBound);
    assert_eq!(suggestions[0].severity, Severity::Warning);
    assert!(suggestions[0].explanation.contains("Unbounded"));
}

/// Bounded traversal → no suggestion.
#[test]
fn bounded_traversal_ok() {
    let p = plan(traverse_bounded(node_scan("a", "Person")));
    let suggestions = detect_suggestions(&p, None);
    assert!(suggestions.is_empty());
}

/// Fixed-length traversal (no length bounds) → no suggestion.
#[test]
fn fixed_length_traversal_ok() {
    let op = LogicalOp::Traverse {
        input: Box::new(node_scan("a", "Person")),
        source: "a".to_string(),
        edge_types: vec!["KNOWS".to_string()],
        direction: Direction::Outgoing,
        target_variable: "b".to_string(),
        target_labels: vec![],
        length: None,
        edge_variable: None,
        target_filters: vec![],
        edge_filters: vec![],
        temporal_filter: None,
        path_variable: None,
    };
    let suggestions = detect_suggestions(&plan(op), None);
    assert!(suggestions.is_empty());
}

// --- CartesianProduct tests ---

/// CartesianProduct → warns about O(n²).
#[test]
fn cartesian_product_detected() {
    let p = plan(LogicalOp::CartesianProduct {
        left: Box::new(node_scan("a", "User")),
        right: Box::new(node_scan("b", "Post")),
    });
    let suggestions = detect_suggestions(&p, None);

    assert_eq!(suggestions.len(), 1);
    assert_eq!(suggestions[0].kind, SuggestionKind::AddJoinPredicate);
    assert_eq!(suggestions[0].severity, Severity::Warning);
    assert!(suggestions[0].explanation.contains("a"));
    assert!(suggestions[0].explanation.contains("b"));
}

// --- KnnWithoutIndex tests ---

/// Sort by distance + Limit → suggests vector index.
#[test]
fn knn_without_index_detected() {
    let sort_op = LogicalOp::Sort {
        input: Box::new(node_scan("n", "Place")),
        items: vec![SortItem {
            expr: Expr::FunctionCall {
                name: "point.distance".to_string(),
                args: vec![
                    Expr::PropertyAccess {
                        expr: Box::new(Expr::Variable("n".to_string())),
                        property: "location".to_string(),
                    },
                    Expr::Literal(Value::Float(0.0)),
                ],
                distinct: false,
            },
            ascending: true,
        }],
    };
    let p = plan(LogicalOp::Limit {
        input: Box::new(sort_op),
        count: Expr::Literal(Value::Int(10)),
    });
    let suggestions = detect_suggestions(&p, None);

    assert_eq!(suggestions.len(), 1);
    assert_eq!(suggestions[0].kind, SuggestionKind::CreateVectorIndex);
    assert_eq!(suggestions[0].severity, Severity::Info);
}

/// Sort without Limit → no KNN suggestion.
#[test]
fn sort_without_limit_no_knn() {
    let sort_op = LogicalOp::Sort {
        input: Box::new(node_scan("n", "Place")),
        items: vec![SortItem {
            expr: Expr::FunctionCall {
                name: "vector_distance".to_string(),
                args: vec![],
                distinct: false,
            },
            ascending: true,
        }],
    };
    let suggestions = detect_suggestions(&plan(sort_op), None);
    assert!(suggestions.is_empty());
}

// --- VectorWithoutPreFilter tests ---

/// VectorFilter directly on NodeScan → suggests pre-filter.
#[test]
fn vector_without_prefilter_detected() {
    let p = plan(LogicalOp::VectorFilter {
        input: Box::new(node_scan("n", "Product")),
        vector_expr: Expr::PropertyAccess {
            expr: Box::new(Expr::Variable("n".to_string())),
            property: "embedding".to_string(),
        },
        query_vector: Expr::Literal(Value::Array(vec![])),
        function: "vector_distance".to_string(),
        less_than: true,
        threshold: 0.5,
        decay_field: None,
        push_down: None,
    });
    let suggestions = detect_suggestions(&p, None);

    assert_eq!(suggestions.len(), 1);
    assert_eq!(suggestions[0].kind, SuggestionKind::AddGraphPreFilter);
    assert_eq!(suggestions[0].severity, Severity::Info);
}

/// VectorFilter after Traverse → no suggestion (has pre-filter).
#[test]
fn vector_with_prefilter_ok() {
    let traverse = LogicalOp::Traverse {
        input: Box::new(node_scan("a", "User")),
        source: "a".to_string(),
        edge_types: vec!["LIKES".to_string()],
        direction: Direction::Outgoing,
        target_variable: "p".to_string(),
        target_labels: vec!["Product".to_string()],
        length: None,
        edge_variable: None,
        target_filters: vec![],
        edge_filters: vec![],
        temporal_filter: None,
        path_variable: None,
    };
    let p = plan(LogicalOp::VectorFilter {
        input: Box::new(traverse),
        vector_expr: Expr::PropertyAccess {
            expr: Box::new(Expr::Variable("p".to_string())),
            property: "embedding".to_string(),
        },
        query_vector: Expr::Literal(Value::Array(vec![])),
        function: "vector_distance".to_string(),
        less_than: true,
        threshold: 0.5,
        decay_field: None,
        push_down: None,
    });
    let suggestions = detect_suggestions(&p, None);
    assert!(suggestions.is_empty());
}

// --- Sorting tests ---

/// Multiple suggestions are sorted by severity (Critical first).
#[test]
fn suggestions_sorted_by_severity() {
    // Build plan with missing index (Critical) on top of unbounded traversal (Warning):
    // Filter(NodeScan(User), n.name = 'x') wrapped in an unbounded traverse
    let filtered_scan = filter(node_scan("n", "User"), prop_eq("n", "name"));
    let p = plan(traverse_unbounded(filtered_scan));
    let suggestions = detect_suggestions(&p, None);

    assert!(
        suggestions.len() >= 2,
        "expected at least 2 suggestions, got {}: {:?}",
        suggestions.len(),
        suggestions
    );
    // First should be Critical (missing index), second Warning (unbounded traversal)
    assert_eq!(suggestions[0].severity, Severity::Critical);
    assert_eq!(suggestions[1].severity, Severity::Warning);
}
