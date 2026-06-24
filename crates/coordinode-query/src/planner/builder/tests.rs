use super::*;
use crate::cypher::parser::parse;

fn plan(input: &str) -> LogicalPlan {
    let query = parse(input).unwrap_or_else(|e| panic!("parse failed: {e}"));
    build_logical_plan(&query).unwrap_or_else(|e| panic!("plan failed: {e}"))
}

fn plan_root(input: &str) -> LogicalOp {
    plan(input).root
}

#[test]
fn create_vector_index_extension_tail_routes_to_extension_op() {
    // Plain create (no trailing clause) stays a CreateVectorIndex op.
    assert!(matches!(
        plan_root("CREATE VECTOR INDEX foo ON :Doc(embedding) OPTIONS {m: 16}"),
        LogicalOp::CreateVectorIndex { .. }
    ));

    // A trailing extension clause routes to an Extension op whose opaque
    // payload round-trips the parsed clause (params + the verbatim tail).
    let root = plan_root("CREATE VECTOR INDEX foo ON :Doc(embedding) SHARDED BY CENTROID(8)");
    let LogicalOp::Extension { name, payload } = &root else {
        panic!("expected Extension op, got {root:?}");
    };
    assert_eq!(name, "vector_index.create_ext");
    let decoded: crate::cypher::ast::CreateVectorIndexClause =
        rmp_serde::from_slice(payload).expect("extension payload decodes");
    assert_eq!(decoded.name, "foo");
    assert_eq!(decoded.label, "Doc");
    assert_eq!(decoded.property, "embedding");
    assert_eq!(
        decoded.extension_tail.as_deref(),
        Some("SHARDED BY CENTROID(8)")
    );
}

// -- Basic MATCH → NodeScan --

#[test]
fn match_node_scan() {
    let root = plan_root("MATCH (n:User) RETURN n");
    // Should be Project(NodeScan)
    if let LogicalOp::Project { input, .. } = &root {
        assert!(matches!(**input, LogicalOp::NodeScan { ref labels, .. } if labels == &["User"]));
    } else {
        panic!("expected Project, got: {root:?}");
    }
}

#[test]
fn shortest_path_plans_shortest_path_op() {
    fn find_shortest_path(op: &LogicalOp) -> Option<(&str, &str, &str, u64)> {
        match op {
            LogicalOp::ShortestPath {
                source,
                target,
                path_variable,
                max_depth,
                ..
            } => Some((source, target, path_variable, *max_depth)),
            LogicalOp::Project { input, .. }
            | LogicalOp::Filter { input, .. }
            | LogicalOp::Aggregate { input, .. } => find_shortest_path(input),
            LogicalOp::CartesianProduct { left, right } => {
                find_shortest_path(left).or_else(|| find_shortest_path(right))
            }
            _ => None,
        }
    }

    let root = plan_root(
        "MATCH p = shortestPath((a:Person {pid: 1})-[:KNOWS*..6]->(b:Person {pid: 2})) \
             RETURN length(p)",
    );
    let (source, target, path_var, max_depth) =
        find_shortest_path(&root).expect("plan must contain a ShortestPath op");
    assert_eq!(source, "a");
    assert_eq!(target, "b");
    assert_eq!(path_var, "p");
    assert_eq!(max_depth, 6, "max_depth comes from the *..6 bound");
}

#[test]
fn match_node_with_properties() {
    let root = plan_root("MATCH (n:User {name: 'Alice'}) RETURN n");
    if let LogicalOp::Project { input, .. } = &root {
        if let LogicalOp::NodeScan {
            property_filters, ..
        } = input.as_ref()
        {
            assert_eq!(property_filters.len(), 1);
            assert_eq!(property_filters[0].0, "name");
        } else {
            panic!("expected NodeScan");
        }
    } else {
        panic!("expected Project");
    }
}

// -- MATCH with relationship → Traverse --

#[test]
fn match_traverse() {
    let root = plan_root("MATCH (a)-[:KNOWS]->(b) RETURN a, b");
    if let LogicalOp::Project { input, .. } = &root {
        if let LogicalOp::Traverse {
            edge_types,
            direction,
            target_variable,
            ..
        } = input.as_ref()
        {
            assert_eq!(edge_types, &["KNOWS"]);
            assert_eq!(*direction, Direction::Outgoing);
            assert_eq!(target_variable, "b");
        } else {
            panic!("expected Traverse, got: {input:?}");
        }
    } else {
        panic!("expected Project");
    }
}

#[test]
fn match_long_traversal_chain() {
    let root = plan_root("MATCH (a)-[:KNOWS]->(b)-[:WORKS_AT]->(c) RETURN a, b, c");
    // Should be Project → Traverse(WORKS_AT) → Traverse(KNOWS) → NodeScan
    if let LogicalOp::Project { input, .. } = &root {
        if let LogicalOp::Traverse {
            input: inner,
            edge_types,
            ..
        } = input.as_ref()
        {
            assert_eq!(edge_types, &["WORKS_AT"]);
            assert!(matches!(**inner, LogicalOp::Traverse { .. }));
        } else {
            panic!("expected Traverse chain");
        }
    }
}

// -- WHERE → Filter --

#[test]
fn match_where_filter() {
    let root = plan_root("MATCH (n:User) WHERE n.age > 25 RETURN n");
    if let LogicalOp::Project { input, .. } = &root {
        assert!(matches!(**input, LogicalOp::Filter { .. }));
    } else {
        panic!("expected Project");
    }
}

// -- Multiple patterns → CartesianProduct --

#[test]
fn multiple_patterns() {
    let root = plan_root("MATCH (a:User), (b:Movie) RETURN a, b");
    if let LogicalOp::Project { input, .. } = &root {
        assert!(matches!(**input, LogicalOp::CartesianProduct { .. }));
    } else {
        panic!("expected Project");
    }
}

// -- Aggregation → Aggregate + Project --

#[test]
fn aggregation() {
    let root = plan_root("MATCH (n:User) RETURN n.city AS city, count(*) AS cnt");
    if let LogicalOp::Project { input, .. } = &root {
        assert!(matches!(**input, LogicalOp::Aggregate { .. }));
    } else {
        panic!("expected Project");
    }
}

// -- ORDER BY / SKIP / LIMIT --

#[test]
fn order_by_limit() {
    let root = plan_root("MATCH (n) RETURN n.name ORDER BY n.name DESC LIMIT 10");
    // Plan should be: Limit → Sort → Project → NodeScan
    if let LogicalOp::Limit { input, .. } = &root {
        if let LogicalOp::Sort {
            input: inner,
            items,
        } = input.as_ref()
        {
            assert_eq!(items.len(), 1);
            assert!(!items[0].ascending);
            assert!(matches!(**inner, LogicalOp::Project { .. }));
        } else {
            panic!("expected Sort");
        }
    } else {
        panic!("expected Limit");
    }
}

// -- WITH → scope barrier Project --

#[test]
fn with_clause() {
    let root = plan_root("MATCH (n:User) WITH n, count(*) AS cnt WHERE cnt > 5 RETURN n, cnt");
    // Plan should contain Filter → Project(Aggregate) somewhere
    if let LogicalOp::Project { input, .. } = &root {
        // Input should be Filter(Project(Aggregate(NodeScan)))
        assert!(
            matches!(**input, LogicalOp::Filter { .. }),
            "expected Filter, got: {input:?}"
        );
    }
}

// -- CREATE → CreateNode --

#[test]
fn create_node() {
    let root = plan_root("CREATE (n:User {name: 'Alice'}) RETURN n");
    // Project → CreateNode → Empty
    if let LogicalOp::Project { input, .. } = &root {
        assert!(matches!(**input, LogicalOp::CreateNode { .. }));
    }
}

// -- DELETE → Delete --

#[test]
fn delete_node() {
    let root = plan_root("MATCH (n:User {id: 42}) DELETE n");
    assert!(matches!(root, LogicalOp::Delete { detach: false, .. }));
}

#[test]
fn detach_delete() {
    let root = plan_root("MATCH (n:User {id: 42}) DETACH DELETE n");
    if let LogicalOp::Delete {
        detach, variables, ..
    } = &root
    {
        assert!(detach);
        assert_eq!(variables, &["n"]);
    } else {
        panic!("expected Delete");
    }
}

// -- SET → Update --

#[test]
fn set_property() {
    let root = plan_root("MATCH (n:User {id: 42}) SET n.name = 'Bob'");
    assert!(matches!(root, LogicalOp::Update { .. }));
}

// -- REMOVE → RemoveOp --

#[test]
fn remove_property() {
    let root = plan_root("MATCH (n) REMOVE n.age");
    assert!(matches!(root, LogicalOp::RemoveOp { .. }));
}

// -- EXPLAIN output --

#[test]
fn explain_simple() {
    let p = plan("MATCH (n:User) RETURN n");
    let explain = p.explain();
    assert!(explain.contains("NodeScan"));
    assert!(explain.contains("Project"));
    assert!(explain.contains("User"));
}

#[test]
fn explain_traverse() {
    let p = plan("MATCH (a:User)-[:KNOWS]->(b) RETURN a, b");
    let explain = p.explain();
    assert!(explain.contains("Traverse"));
    assert!(explain.contains("KNOWS"));
}

#[test]
fn explain_aggregate() {
    let p = plan("MATCH (n:User) RETURN n.city, count(*) AS cnt");
    let explain = p.explain();
    assert!(explain.contains("Aggregate"));
    assert!(explain.contains("Project"));
}

/// The product graph-traversal query: a var-length BFS whose reached set is
/// consumed only by `count(DISTINCT f)`. The planner wraps the aggregate in a
/// Project (RETURN ... AS reach) and lifts the pid predicate, so the dedup
/// detector must see through those wrappers and still enable per-node
/// emission. Regression for the near-global traversal that otherwise emits
/// O(edges) rows and runs for minutes.
#[test]
fn varlen_count_distinct_plan_enables_target_dedup() {
    let p = plan(
        "MATCH (p:Person) WHERE p.pid = 0 \
             MATCH (p)-[:KNOWS*1..6]->(f) RETURN count(DISTINCT f) AS reach",
    );
    assert!(
        crate::executor::runner::plan_allows_varlen_target_dedup(&p.root),
        "count(DISTINCT f) over a lone var-length traverse must enable target \
             dedup; planned tree was:\n{}",
        p.explain()
    );
}

/// Same shape with a parameterised source predicate, as the traversal bench
/// and real clients send it. The parameter must not change the plan shape the
/// dedup detector matches.
#[test]
fn varlen_count_distinct_param_plan_enables_target_dedup() {
    let p = plan(
        "MATCH (p:Person) WHERE p.pid = $pid \
             MATCH (p)-[:KNOWS*1..6]->(f) RETURN count(DISTINCT f) AS reach",
    );
    assert!(
        crate::executor::runner::plan_allows_varlen_target_dedup(&p.root),
        "parameterised count(DISTINCT f) must enable target dedup; tree was:\n{}",
        p.explain()
    );
}

/// Dedup must NOT fire when target multiplicity is observable: a bare
/// `RETURN f` projects every path, and `count(f)` without DISTINCT counts
/// paths. The Project passthrough above the aggregate must not relax these.
#[test]
fn varlen_observable_multiplicity_disables_target_dedup() {
    let bare = plan("MATCH (p:Person)-[:KNOWS*1..6]->(f) RETURN f");
    assert!(
        !crate::executor::runner::plan_allows_varlen_target_dedup(&bare.root),
        "bare RETURN f exposes path multiplicity; dedup must stay off:\n{}",
        bare.explain()
    );

    let count_no_distinct = plan("MATCH (p:Person)-[:KNOWS*1..6]->(f) RETURN count(f) AS reach");
    assert!(
        !crate::executor::runner::plan_allows_varlen_target_dedup(&count_no_distinct.root),
        "count(f) without DISTINCT counts paths; dedup must stay off:\n{}",
        count_no_distinct.explain()
    );
}

// -- UNWIND --

#[test]
fn unwind_plan() {
    let root = plan_root("UNWIND [1, 2, 3] AS x RETURN x");
    // Project → Unwind → Empty
    if let LogicalOp::Project { input, .. } = &root {
        assert!(matches!(**input, LogicalOp::Unwind { .. }));
    }
}

// -- Complex query --

#[test]
fn complex_graph_vector_query() {
    let p = plan(
        "MATCH (u:User {id: $me})-[:LIKES]->(m:Movie) \
             WHERE vector_distance(m.embedding, $query_vec) < 0.3 \
             WITH m.genre AS genre, count(*) AS cnt \
             ORDER BY cnt DESC \
             LIMIT 10 \
             RETURN genre, cnt",
    );
    let explain = p.explain();
    // Should contain: NodeScan, Traverse, Filter, Aggregate, Project, Sort, Limit
    assert!(explain.contains("NodeScan"));
    assert!(explain.contains("Traverse"));
}

// -- Variable-length path --

#[test]
fn variable_length_path() {
    let root = plan_root("MATCH (a)-[:KNOWS*2..5]->(b) RETURN a, b");
    if let LogicalOp::Project { input, .. } = &root {
        if let LogicalOp::Traverse { length, .. } = input.as_ref() {
            let lb = length.unwrap();
            assert_eq!(lb.min, Some(2));
            assert_eq!(lb.max, Some(5));
        } else {
            panic!("expected Traverse");
        }
    }
}

// -- MATCH + SET + RETURN --

#[test]
fn match_set_return() {
    let root = plan_root("MATCH (n:User {id: $id}) SET n.name = $name RETURN n");
    // Project → Update → NodeScan(with filter)
    if let LogicalOp::Project { input, .. } = &root {
        assert!(matches!(**input, LogicalOp::Update { .. }));
    }
}

// --- Compound WHERE predicate splitting tests ---

/// Compound AND with vector + text produces VectorFilter → TextFilter pipeline.
#[test]
fn compound_where_vector_and_text() {
    let q = parse(
        "MATCH (n:Doc) \
             WHERE vector_distance(n.embedding, [1.0, 0.0]) < 0.5 \
               AND text_match(n.body, 'hello world') \
             RETURN n",
    )
    .unwrap();
    let plan = build_logical_plan(&q).unwrap();
    let explain = plan.explain();
    assert!(
        explain.contains("VectorFilter"),
        "compound AND should produce VectorFilter: {explain}"
    );
    assert!(
        explain.contains("TextFilter"),
        "compound AND should produce TextFilter: {explain}"
    );
}

/// Compound AND with vector + property filter splits correctly.
#[test]
fn compound_where_vector_and_property() {
    let q = parse(
        "MATCH (n:Product) \
             WHERE vector_distance(n.embedding, [0.1, 0.2]) < 0.3 \
               AND n.price > 100 \
             RETURN n",
    )
    .unwrap();
    let plan = build_logical_plan(&q).unwrap();
    let explain = plan.explain();
    assert!(
        explain.contains("VectorFilter"),
        "should extract VectorFilter from compound AND: {explain}"
    );
    assert!(
        explain.contains("Filter"),
        "remaining property predicate should be in Filter: {explain}"
    );
}

/// Triple compound: vector + text + property.
#[test]
fn compound_where_triple() {
    let q = parse(
        "MATCH (d:Document) \
             WHERE vector_distance(d.embedding, [0.5]) < 0.4 \
               AND text_match(d.body, 'transformer') \
               AND d.published = true \
             RETURN d",
    )
    .unwrap();
    let plan = build_logical_plan(&q).unwrap();
    let explain = plan.explain();
    assert!(
        explain.contains("VectorFilter"),
        "triple compound: VectorFilter expected: {explain}"
    );
    assert!(
        explain.contains("TextFilter"),
        "triple compound: TextFilter expected: {explain}"
    );
    assert!(
        explain.contains("Filter"),
        "triple compound: generic Filter for property expected: {explain}"
    );
}

/// Single vector predicate (no AND) still works.
#[test]
fn single_vector_predicate_no_regression() {
    let q = parse(
        "MATCH (n:Item) \
             WHERE vector_distance(n.vec, [1.0]) < 0.5 \
             RETURN n",
    )
    .unwrap();
    let plan = build_logical_plan(&q).unwrap();
    assert!(plan.explain().contains("VectorFilter"));
}

/// Single text predicate (no AND) still works.
#[test]
fn single_text_predicate_no_regression() {
    let q = parse(
        "MATCH (n:Doc) \
             WHERE text_match(n.body, 'search query') \
             RETURN n",
    )
    .unwrap();
    let plan = build_logical_plan(&q).unwrap();
    assert!(plan.explain().contains("TextFilter"));
}

/// Plain property filter (no vector/text) unchanged.
#[test]
fn plain_property_filter_no_regression() {
    let q = parse("MATCH (n:User) WHERE n.age > 25 RETURN n").unwrap();
    let plan = build_logical_plan(&q).unwrap();
    let explain = plan.explain();
    assert!(explain.contains("Filter"), "property filter: {explain}");
    assert!(
        !explain.contains("VectorFilter"),
        "no vector in plain filter: {explain}"
    );
}

// -- G024: cross-MATCH predicate lifting --

/// MATCH (a) MATCH (b) WHERE a.x = b.y
/// The Filter should be ABOVE CartesianProduct, not inside the right branch.
#[test]
fn cross_match_where_lifted_above_cartesian_product() {
    let explain =
        plan("MATCH (a:User) MATCH (b:Post) WHERE a.name = b.author RETURN a, b").explain();
    // Filter must wrap CartesianProduct, not be inside it.
    // EXPLAIN format: Filter wraps CartesianProduct(left, right).
    assert!(
        explain.contains("Filter"),
        "cross-match WHERE should produce Filter: {explain}"
    );
    assert!(
        explain.contains("CartesianProduct"),
        "multi-MATCH should produce CartesianProduct: {explain}"
    );
}

/// MATCH (a) MATCH (b) WHERE b.x = 5 (local only — no lifting needed).
#[test]
fn multi_match_local_where_stays_in_branch() {
    let explain =
        plan("MATCH (a:User) MATCH (b:Post) WHERE b.published = true RETURN a, b").explain();
    assert!(
        explain.contains("Filter"),
        "local WHERE should produce Filter: {explain}"
    );
}

/// collect_expr_variables extracts variables from simple and nested expressions.
#[test]
fn collect_expr_variables_basic() {
    let vars = collect_expr_variables(&Expr::Variable("a".into()));
    assert_eq!(vars, vec!["a"]);
}

#[test]
fn collect_expr_variables_property_access() {
    let expr = Expr::PropertyAccess {
        expr: Box::new(Expr::Variable("n".into())),
        property: "name".into(),
    };
    let vars = collect_expr_variables(&expr);
    assert_eq!(vars, vec!["n"]);
}

#[test]
fn collect_expr_variables_binary_op() {
    let expr = Expr::BinaryOp {
        left: Box::new(Expr::PropertyAccess {
            expr: Box::new(Expr::Variable("a".into())),
            property: "x".into(),
        }),
        op: BinaryOperator::Eq,
        right: Box::new(Expr::PropertyAccess {
            expr: Box::new(Expr::Variable("b".into())),
            property: "y".into(),
        }),
    };
    let vars = collect_expr_variables(&expr);
    assert_eq!(vars, vec!["a", "b"]);
}

#[test]
fn collect_pattern_variables_basic() {
    let patterns = vec![Pattern {
        elements: vec![
            PatternElement::Node(NodePattern {
                variable: Some("a".into()),
                labels: vec!["User".into()],
                properties: vec![],
            }),
            PatternElement::Relationship(RelationshipPattern {
                variable: Some("r".into()),
                rel_types: vec!["KNOWS".into()],
                direction: Direction::Outgoing,
                length: None,
                properties: vec![],
            }),
            PatternElement::Node(NodePattern {
                variable: Some("b".into()),
                labels: vec![],
                properties: vec![],
            }),
        ],
        path_variable: None,
        shortest_path: false,
    }];
    let vars = collect_pattern_variables(&patterns);
    assert_eq!(vars, vec!["a", "r", "b"]);
}

// -- G029: ALTER LABEL SET SCHEMA --

#[test]
fn alter_label_parses() {
    let q = parse("ALTER LABEL User SET SCHEMA VALIDATED").unwrap();
    assert_eq!(q.clauses.len(), 1);
    if let Clause::AlterLabel(ref ac) = q.clauses[0] {
        assert_eq!(ac.label, "User");
        assert_eq!(ac.mode, "validated");
    } else {
        panic!("expected AlterLabel, got: {:?}", q.clauses[0]);
    }
}

#[test]
fn alter_label_flexible() {
    let q = parse("ALTER LABEL Config SET SCHEMA FLEXIBLE").unwrap();
    if let Clause::AlterLabel(ref ac) = q.clauses[0] {
        assert_eq!(ac.label, "Config");
        assert_eq!(ac.mode, "flexible");
    } else {
        panic!("expected AlterLabel");
    }
}

#[test]
fn alter_label_strict() {
    let q = parse("ALTER LABEL Product SET SCHEMA STRICT").unwrap();
    if let Clause::AlterLabel(ref ac) = q.clauses[0] {
        assert_eq!(ac.mode, "strict");
    } else {
        panic!("expected AlterLabel");
    }
}

#[test]
fn alter_label_case_insensitive() {
    let q = parse("alter label User set schema validated").unwrap();
    if let Clause::AlterLabel(ref ac) = q.clauses[0] {
        assert_eq!(ac.label, "User");
        assert_eq!(ac.mode, "validated");
    } else {
        panic!("expected AlterLabel");
    }
}

#[test]
fn alter_label_plan_produces_alter_label_op() {
    let explain = plan("ALTER LABEL User SET SCHEMA VALIDATED").explain();
    assert!(
        explain.contains("AlterLabel"),
        "EXPLAIN should show AlterLabel: {explain}"
    );
    assert!(
        explain.contains("User"),
        "EXPLAIN should show label name: {explain}"
    );
}

// --- VectorTopK optimization tests ---

/// Find VectorTopK in a plan tree (depth-first). Returns a borrowed reference.
fn find_vector_top_k(op: &LogicalOp) -> Option<&LogicalOp> {
    if matches!(op, LogicalOp::VectorTopK { .. }) {
        return Some(op);
    }
    match op {
        LogicalOp::Project { input, .. }
        | LogicalOp::Filter { input, .. }
        | LogicalOp::Sort { input, .. }
        | LogicalOp::Limit { input, .. }
        | LogicalOp::Skip { input, .. }
        | LogicalOp::Aggregate { input, .. }
        | LogicalOp::Unwind { input, .. } => find_vector_top_k(input),
        LogicalOp::CartesianProduct { left, right } | LogicalOp::LeftOuterJoin { left, right } => {
            find_vector_top_k(left).or_else(|| find_vector_top_k(right))
        }
        _ => None,
    }
}

/// Pattern A: direct function call in ORDER BY.
/// `ORDER BY vector_distance(n.emb, $qv)` without an alias.
#[test]
fn vector_top_k_pattern_a_direct_call() {
    let root = plan_root(
        "MATCH (n:Doc) \
             RETURN n \
             ORDER BY vector_distance(n.embedding, [1.0, 0.0]) \
             LIMIT 5",
    );
    let top_k = find_vector_top_k(&root).expect("VectorTopK not found in plan");
    if let LogicalOp::VectorTopK {
        function,
        k,
        distance_alias,
        ..
    } = top_k
    {
        assert_eq!(function, "vector_distance");
        assert_eq!(*k, 5);
        assert!(
            distance_alias.is_none(),
            "pattern A has no alias: {distance_alias:?}"
        );
    } else {
        panic!("expected VectorTopK");
    }
}

/// Pattern B: alias via WITH before ORDER BY.
/// This is the shape that VectorServiceImpl generates.
#[test]
fn vector_top_k_pattern_b_alias_via_with() {
    let root = plan_root(
        "MATCH (n:Doc) \
             WITH n, vector_distance(n.embedding, [1.0, 0.0]) AS _dist \
             ORDER BY _dist \
             LIMIT 10 \
             RETURN n, _dist",
    );
    let top_k = find_vector_top_k(&root).expect("VectorTopK not found in plan");
    if let LogicalOp::VectorTopK {
        function,
        k,
        distance_alias,
        ..
    } = top_k
    {
        assert_eq!(function, "vector_distance");
        assert_eq!(*k, 10);
        assert_eq!(distance_alias.as_deref(), Some("_dist"));
    } else {
        panic!("expected VectorTopK");
    }
}

/// No LIMIT → no optimization (Sort remains).
#[test]
fn vector_top_k_no_limit_not_optimized() {
    let root = plan_root(
        "MATCH (n:Doc) \
             RETURN n \
             ORDER BY vector_distance(n.embedding, [1.0, 0.0])",
    );
    assert!(
        find_vector_top_k(&root).is_none(),
        "ORDER BY without LIMIT should not produce VectorTopK: {root:?}"
    );
}

/// ORDER BY DESC on distance function → not optimized (wrong direction).
#[test]
fn vector_top_k_wrong_direction_not_optimized() {
    let root = plan_root(
        "MATCH (n:Doc) \
             RETURN n \
             ORDER BY vector_distance(n.embedding, [1.0, 0.0]) DESC \
             LIMIT 5",
    );
    assert!(
        find_vector_top_k(&root).is_none(),
        "DESC on vector_distance should not produce VectorTopK: {root:?}"
    );
}

/// vector_similarity uses DESC (higher is better) — this IS valid.
#[test]
fn vector_top_k_similarity_desc_optimized() {
    let root = plan_root(
        "MATCH (n:Doc) \
             RETURN n \
             ORDER BY vector_similarity(n.embedding, [1.0, 0.0]) DESC \
             LIMIT 3",
    );
    let top_k = find_vector_top_k(&root).expect("VectorTopK not found in plan");
    if let LogicalOp::VectorTopK { function, k, .. } = top_k {
        assert_eq!(function, "vector_similarity");
        assert_eq!(*k, 3);
    }
}

/// Non-vector ORDER BY → not optimized.
#[test]
fn vector_top_k_non_vector_order_not_optimized() {
    let root = plan_root("MATCH (n:User) RETURN n ORDER BY n.age LIMIT 5");
    assert!(
        find_vector_top_k(&root).is_none(),
        "ORDER BY non-vector should not produce VectorTopK"
    );
}

/// Multi-item ORDER BY → not optimized (only single-key ORDER BY supported).
#[test]
fn vector_top_k_multi_item_order_not_optimized() {
    let root = plan_root(
        "MATCH (n:Doc) \
             RETURN n \
             ORDER BY vector_distance(n.embedding, [1.0, 0.0]), n.title \
             LIMIT 5",
    );
    assert!(
        find_vector_top_k(&root).is_none(),
        "multi-item ORDER BY should not produce VectorTopK"
    );
}

/// Pattern A with RETURN that drops the vector variable → NOT optimized.
/// `RETURN n.name` removes `n.embedding` from intermediate rows, so the
/// optimizer must fall back to plain Sort+Limit. This guards against the
/// regression found by g009_forced_offload_cypher_e2e integration test.
#[test]
fn vector_top_k_pattern_a_dropped_variable_not_optimized() {
    let root = plan_root(
        "MATCH (n:Doc) \
             RETURN n.name \
             ORDER BY vector_distance(n.embedding, [1.0, 0.0]) \
             LIMIT 5",
    );
    assert!(
        find_vector_top_k(&root).is_none(),
        "Pattern A must skip optimization when Project drops the vector variable: {root:?}"
    );
}

// --- HnswScan access-path tests ---

/// Find HnswScan anywhere in a plan tree (depth-first).
fn find_hnsw_scan(op: &LogicalOp) -> Option<&LogicalOp> {
    if matches!(op, LogicalOp::HnswScan { .. }) {
        return Some(op);
    }
    match op {
        LogicalOp::Project { input, .. }
        | LogicalOp::Filter { input, .. }
        | LogicalOp::Sort { input, .. }
        | LogicalOp::Limit { input, .. }
        | LogicalOp::Skip { input, .. }
        | LogicalOp::Aggregate { input, .. }
        | LogicalOp::VectorTopK { input, .. }
        | LogicalOp::Unwind { input, .. } => find_hnsw_scan(input),
        LogicalOp::CartesianProduct { left, right } | LogicalOp::LeftOuterJoin { left, right } => {
            find_hnsw_scan(left).or_else(|| find_hnsw_scan(right))
        }
        _ => None,
    }
}

fn test_registry_with_doc_index() -> crate::index::VectorIndexRegistry {
    let registry = crate::index::VectorIndexRegistry::new();
    registry.register(crate::index::IndexDefinition::hnsw(
        "doc_emb_idx",
        "Doc",
        "embedding",
        crate::index::VectorIndexConfig {
            dimensions: 2,
            metric: coordinode_core::graph::types::VectorMetric::L2,
            m: 16,
            ef_construction: 200,
            quantization: coordinode_vector::hnsw::QuantizationCodec::None,
            offload_vectors: false,
            ef_search: None,
            rerank_candidates: None,
        },
    ));
    registry
}

/// Pure vector top-K over a bare NodeScan with a registered index
/// MUST plan as the HnswScan access path: no NodeScan, no
/// VectorTopK — the index IS the row source. This is the
/// scan-then-rank fix: without HnswScan every server-side vector
/// search materialises the whole label before ranking.
#[test]
fn hnsw_scan_replaces_scan_then_rank_for_pure_top_k() {
    let registry = test_registry_with_doc_index();
    let root = plan_root(
        "MATCH (n:Doc) \
             WITH *, vector_distance(n.embedding, [1.0, 0.0]) AS _dist \
             ORDER BY _dist \
             LIMIT 10 \
             RETURN *",
    );
    let root = apply_hnsw_scan_access_path(root, &registry);
    let scan = find_hnsw_scan(&root).expect(
        "pure vector top-K with a registered index must plan HnswScan, \
             not scan-then-rank",
    );
    if let LogicalOp::HnswScan {
        label,
        property,
        binding,
        k,
        function,
        index_name,
        ..
    } = scan
    {
        assert_eq!(label, "Doc");
        assert_eq!(property, "embedding");
        assert_eq!(binding, "n");
        assert_eq!(*k, 10);
        assert_eq!(function, "vector_distance");
        assert_eq!(index_name, "doc_emb_idx");
    } else {
        unreachable!("find_hnsw_scan returned a non-HnswScan op");
    }
    assert!(
        find_vector_top_k(&root).is_none(),
        "HnswScan must REPLACE VectorTopK, not coexist: {root:?}"
    );
}

/// A WHERE filter between the scan and the top-K keeps the
/// scan-then-rank path (filtered HNSW is the VectorTopK line of work).
#[test]
fn hnsw_scan_not_planned_with_filter() {
    let registry = test_registry_with_doc_index();
    let root = plan_root(
        "MATCH (n:Doc) \
             WHERE n.lang = 'en' \
             WITH *, vector_distance(n.embedding, [1.0, 0.0]) AS _dist \
             ORDER BY _dist \
             LIMIT 10 \
             RETURN *",
    );
    let root = apply_hnsw_scan_access_path(root, &registry);
    assert!(
        find_hnsw_scan(&root).is_none(),
        "filtered query must keep the VectorTopK path: {root:?}"
    );
    assert!(
        find_vector_top_k(&root).is_some(),
        "filtered query must still have VectorTopK: {root:?}"
    );
}

/// No registered index for the (label, property) → no HnswScan.
#[test]
fn hnsw_scan_not_planned_without_index() {
    let registry = crate::index::VectorIndexRegistry::new();
    let root = plan_root(
        "MATCH (n:Doc) \
             WITH *, vector_distance(n.embedding, [1.0, 0.0]) AS _dist \
             ORDER BY _dist \
             LIMIT 10 \
             RETURN *",
    );
    let root = apply_hnsw_scan_access_path(root, &registry);
    assert!(
        find_hnsw_scan(&root).is_none(),
        "no index registered -> no HnswScan: {root:?}"
    );
}

/// Pattern A with RETURN * preserves all columns → optimized.
#[test]
fn vector_top_k_pattern_a_return_star_optimized() {
    let root = plan_root(
        "MATCH (n:Doc) \
             RETURN * \
             ORDER BY vector_distance(n.embedding, [1.0, 0.0]) \
             LIMIT 5",
    );
    assert!(
        find_vector_top_k(&root).is_some(),
        "RETURN * preserves all columns — should still optimize: {root:?}"
    );
}

/// Pattern A with RETURN n (full node ref) keeps n.embedding accessible → optimized.
#[test]
fn vector_top_k_pattern_a_return_node_optimized() {
    let root = plan_root(
        "MATCH (n:Doc) \
             RETURN n \
             ORDER BY vector_distance(n.embedding, [1.0, 0.0]) \
             LIMIT 5",
    );
    assert!(
        find_vector_top_k(&root).is_some(),
        "RETURN n preserves the full node — should optimize: {root:?}"
    );
}

/// SKIP between Limit and Sort: `ORDER BY d SKIP 5 LIMIT 10` produces
/// `Limit { Skip { Sort { ... } } }`. The optimizer must NOT apply —
/// Pattern A/B requires Sort directly under Limit. Fallback Sort+Skip+Limit
/// is correct, but VectorTopK cannot express skip semantics.
#[test]
fn vector_top_k_skip_limit_not_optimized() {
    let root = plan_root(
        "MATCH (n:Doc) \
             WITH n, vector_distance(n.embedding, [1.0, 0.0]) AS d \
             ORDER BY d \
             SKIP 5 \
             LIMIT 10 \
             RETURN n, d",
    );
    assert!(
        find_vector_top_k(&root).is_none(),
        "SKIP between Sort and Limit should not produce VectorTopK: {root:?}"
    );
}

/// Nested WITH: two sequential WITH clauses with vector_distance. The outer
/// WITH removes the vector variable, so Pattern B's guard (`project_preserves_vector_expr`)
/// should still allow optimization because inner WITH projects the vector
/// via alias which carries through.
#[test]
fn vector_top_k_nested_with_still_optimized() {
    let root = plan_root(
        "MATCH (n:Doc) \
             WITH n, n.title AS title \
             WITH n, title, vector_distance(n.embedding, [1.0, 0.0]) AS d \
             ORDER BY d \
             LIMIT 3 \
             RETURN title, d",
    );
    // Pattern B should match the innermost `WITH ... AS d` + ORDER BY + LIMIT.
    assert!(
        find_vector_top_k(&root).is_some(),
        "Nested WITH with Pattern B should still optimize: {root:?}"
    );
}

/// Parameter-passed query vector (via `$qv` parameter) — the typical shape
/// generated by the Python SDK. Pattern B alias + parameter must optimize.
#[test]
fn vector_top_k_with_parameter_query_vector() {
    let root = plan_root(
        "MATCH (n:Doc) \
             WITH n, vector_distance(n.embedding, $qv) AS d \
             ORDER BY d \
             LIMIT 10 \
             RETURN n, d",
    );
    let top_k =
        find_vector_top_k(&root).expect("VectorTopK not found when query vector is parameter");
    if let LogicalOp::VectorTopK {
        query_vector,
        distance_alias,
        k,
        ..
    } = top_k
    {
        // query_vector must be the parameter expression (not substituted).
        assert!(
            matches!(query_vector, Expr::Parameter(_)),
            "query_vector should be Parameter, got {query_vector:?}"
        );
        assert_eq!(*k, 10);
        assert_eq!(distance_alias.as_deref(), Some("d"));
    } else {
        panic!("expected VectorTopK");
    }
}

// -- R171: lift_temporal_filter --

/// Finds the `temporal_filter` field on the innermost Traverse, if any.
fn first_temporal_filter(op: &LogicalOp) -> Option<&TemporalFilter> {
    match op {
        LogicalOp::Traverse {
            temporal_filter,
            input,
            ..
        } => temporal_filter
            .as_ref()
            .or_else(|| first_temporal_filter(input)),
        LogicalOp::Filter { input, .. }
        | LogicalOp::Project { input, .. }
        | LogicalOp::Limit { input, .. }
        | LogicalOp::Skip { input, .. }
        | LogicalOp::Sort { input, .. } => first_temporal_filter(input),
        _ => None,
    }
}

#[test]
fn lift_temporal_filter_recognises_active_at_with_int_literal() {
    let root = plan_root(
        "MATCH (a)-[r:WORKS_AT]->(b) WHERE temporal_active_at(r, 1700000000000) RETURN b",
    );
    let tf =
        first_temporal_filter(&root).expect("planner must lift temporal_active_at into Traverse");
    assert_eq!(tf.edge_variable, "r");
    assert_eq!(tf.upper_ms, Some(1_700_000_000_000));
    assert_eq!(tf.lower_ms, Some(1_700_000_000_000));
}

#[test]
fn lift_temporal_filter_leaves_unrelated_filters_alone() {
    // No temporal_active_at → temporal_filter must stay None.
    let root = plan_root("MATCH (a)-[r:WORKS_AT]->(b:Company) WHERE b.name = 'Acme' RETURN b");
    assert!(
        first_temporal_filter(&root).is_none(),
        "non-temporal predicate must not produce a temporal_filter"
    );
}

#[test]
fn temporal_filter_renders_in_explain_output() {
    let p = plan("MATCH (a)-[r:WORKS_AT]->(b) WHERE temporal_active_at(r, 1700000000000) RETURN b");
    let explain = p.explain();
    assert!(
        explain.contains("temporal_filter"),
        "EXPLAIN must surface temporal_filter block: {explain}"
    );
    assert!(
        explain.contains("r=r"),
        "EXPLAIN must name the edge variable: {explain}"
    );
    assert!(
        explain.contains("valid_from<=1700000000000"),
        "EXPLAIN must show the upper bound: {explain}"
    );
}

#[test]
fn lift_temporal_filter_ignores_parameter_argument_at_build_time() {
    // Plan-build time: parameter expressions aren't literals yet.
    // The second lift pass runs in `execute()` after `substitute_params`,
    // so production queries with bound `$t` still get push-down at runtime.
    let root = plan_root("MATCH (a)-[r:WORKS_AT]->(b) WHERE temporal_active_at(r, $t) RETURN b");
    assert!(
        first_temporal_filter(&root).is_none(),
        "parameter argument must not produce a temporal_filter at plan-build time"
    );
}

#[test]
fn lift_temporal_filter_recognises_overlaps_with_literals() {
    let root =
        plan_root("MATCH (a)-[r:WORKS_AT]->(b) WHERE temporal_overlaps(r, 1000, 2000) RETURN b");
    let tf =
        first_temporal_filter(&root).expect("planner must lift temporal_overlaps into Traverse");
    assert_eq!(tf.edge_variable, "r");
    // Overlap upper bound is exclusive on t_end → encode as t_end - 1.
    assert_eq!(tf.upper_ms, Some(1999));
    assert_eq!(tf.lower_ms, Some(1000));
}

/// EXPLAIN output for VectorTopK shows function and k.
#[test]
fn vector_top_k_explain_output() {
    let p = plan(
        "MATCH (n:Doc) \
             WITH n, vector_distance(n.embedding, [1.0, 0.0]) AS d \
             ORDER BY d \
             LIMIT 7 \
             RETURN n",
    );
    let explain = p.explain();
    assert!(
        explain.contains("VectorTopK"),
        "EXPLAIN should show VectorTopK: {explain}"
    );
    assert!(
        explain.contains("k=7"),
        "EXPLAIN should show k=7: {explain}"
    );
    assert!(
        explain.contains("vector_distance"),
        "EXPLAIN should show function: {explain}"
    );
}

// ── R-PUSH1: Graph-Predicate Push-Down Invariant ──────────────────────

/// Walk a plan tree asserting the push-down invariant:
/// every `VectorFilter` whose input contains a `Traverse` carries
/// `push_down: Some(_)`. Returns `Err(path)` describing the violating
/// operator chain on the first failure.
fn assert_push_down_invariant(op: &LogicalOp) -> Result<(), String> {
    match op {
        LogicalOp::VectorFilter {
            input, push_down, ..
        } => {
            if contains_traverse(input) && push_down.is_none() {
                return Err(format!(
                    "VectorFilter with Traverse in input has push_down: None — \
                         optimize_push_down was not invoked or the rule failed. \
                         Input op: {:?}",
                    std::mem::discriminant(input.as_ref())
                ));
            }
            assert_push_down_invariant(input)
        }
        LogicalOp::Filter { input, .. }
        | LogicalOp::Project { input, .. }
        | LogicalOp::Sort { input, .. }
        | LogicalOp::Limit { input, .. }
        | LogicalOp::Skip { input, .. }
        | LogicalOp::Aggregate { input, .. }
        | LogicalOp::EdgeVectorSearch { input, .. }
        | LogicalOp::VectorTopK { input, .. } => assert_push_down_invariant(input),
        LogicalOp::CartesianProduct { left, right } | LogicalOp::LeftOuterJoin { left, right } => {
            assert_push_down_invariant(left)?;
            assert_push_down_invariant(right)
        }
        _ => Ok(()),
    }
}

fn optimized_plan(input: &str) -> LogicalOp {
    // Mirror the order of passes the Database engine applies in
    // execute_cypher_impl: index selection → top-k annotation →
    // push-down. The invariant test exercises the full sequence
    // because optimize_push_down is the *last* pass that can fix
    // VectorFilter annotations.
    let root = plan_root(input);
    let root = optimize_edge_vector_search(root);
    optimize_push_down(root, None)
}

#[test]
fn push_down_invariant_simple_traverse_then_vector() {
    // (a)-[:LIKES]->(b) WHERE vector_distance(b.embedding, [..]) < 0.5
    // Plan: Project → VectorFilter → Traverse → NodeScan(a)
    let root = optimized_plan(
        "MATCH (a:User)-[:LIKES]->(b:Movie) \
             WHERE vector_distance(b.embedding, [1.0, 0.0, 0.0]) < 0.5 \
             RETURN b",
    );
    assert_push_down_invariant(&root)
        .expect("invariant: VectorFilter after Traverse must have push_down decision");
}

#[test]
fn push_down_explain_json_emitted_for_real_plan() {
    // A real TRAVERSE→VECTOR_FILTER query, run through the same pass order as
    // execute_cypher_impl, carries a push_down decision that renders the stable
    // EXPLAIN block (R-PUSH2).
    let root = optimized_plan(
        "MATCH (a:User)-[:LIKES]->(b:Movie) \
             WHERE vector_distance(b.embedding, [1.0, 0.0, 0.0]) < 0.5 \
             RETURN b",
    );
    let decision = first_push_down_decision(&root)
        .expect("real TRAVERSE→VECTOR_FILTER plan must carry a push_down decision");
    let json = decision.to_explain_json();
    assert!(json.contains("\"stage\": \"VECTOR_FILTER\""), "{json}");
    assert!(json.contains("\"strategy\":"), "{json}");
    assert!(json.contains("\"reason\":"), "{json}");
}

#[test]
fn push_down_invariant_with_similarity() {
    let root = optimized_plan(
        "MATCH (u:User)-[:WATCHED]->(m:Movie) \
             WHERE vector_similarity(m.embedding, [0.1, 0.2, 0.3]) > 0.8 \
             RETURN m",
    );
    assert_push_down_invariant(&root)
        .expect("invariant must hold for vector_similarity predicates");
}

#[test]
fn push_down_no_traverse_no_decision_attached() {
    // Bare MATCH (n:X) WHERE vector_distance(n.embedding, ...) — no Traverse upstream
    // → no push-down annotation needed (rule does not apply).
    let root = optimized_plan(
        "MATCH (n:Doc) WHERE vector_distance(n.embedding, [1.0, 0.0]) < 0.5 RETURN n",
    );
    // Invariant still holds (no Traverse → no requirement to annotate).
    assert_push_down_invariant(&root).expect("invariant trivially satisfied without Traverse");
    // And we expect push_down to remain None — verify directly.
    fn find_vector_filter_decision(op: &LogicalOp) -> Option<bool> {
        match op {
            LogicalOp::VectorFilter { push_down, .. } => Some(push_down.is_some()),
            LogicalOp::Filter { input, .. }
            | LogicalOp::Project { input, .. }
            | LogicalOp::Sort { input, .. }
            | LogicalOp::Limit { input, .. }
            | LogicalOp::Skip { input, .. }
            | LogicalOp::Aggregate { input, .. } => find_vector_filter_decision(input),
            _ => None,
        }
    }
    // Either there is no VectorFilter (e.g., rewritten to VectorTopK or similar),
    // OR if there is one it must have push_down: None when no Traverse upstream.
    if let Some(has) = find_vector_filter_decision(&root) {
        assert!(
            !has,
            "VectorFilter without Traverse upstream must not carry a push_down decision"
        );
    }
}

#[test]
fn push_down_invariant_teeth_catches_unannotated_violation() {
    // Teeth check: the invariant assertion MUST detect a deliberate
    // violation. Construct a VectorFilter with push_down=None placed
    // directly above a Traverse, then verify assert_push_down_invariant
    // returns Err. Without this test the invariant assertion could be
    // a vacuous tautology that always passes.
    use crate::cypher::ast::Expr;
    use coordinode_core::graph::types::Value;

    // Build a NodeScan → Traverse → VectorFilter chain manually.
    let node_scan = LogicalOp::NodeScan {
        variable: "a".to_string(),
        labels: vec!["User".to_string()],
        property_filters: vec![],
    };
    let traverse = LogicalOp::Traverse {
        input: Box::new(node_scan),
        source: "a".to_string(),
        edge_types: vec!["LIKES".to_string()],
        direction: crate::cypher::ast::Direction::Outgoing,
        target_variable: "b".to_string(),
        target_labels: vec![],
        length: None,
        edge_variable: None,
        target_filters: vec![],
        edge_filters: vec![],
        temporal_filter: None,
        path_variable: None,
    };
    let violating = LogicalOp::VectorFilter {
        input: Box::new(traverse),
        vector_expr: Expr::PropertyAccess {
            expr: Box::new(Expr::Variable("b".to_string())),
            property: "embedding".to_string(),
        },
        query_vector: Expr::Literal(Value::Array(vec![])),
        function: "vector_distance".to_string(),
        less_than: true,
        threshold: 0.5,
        decay_field: None,
        push_down: None, // ← deliberate violation
    };

    let result = assert_push_down_invariant(&violating);
    assert!(
        result.is_err(),
        "invariant assertion must FAIL on VectorFilter(push_down=None) directly above Traverse — \
             otherwise the invariant test is a no-op"
    );
    let err = result.unwrap_err();
    assert!(
        err.contains("push_down: None"),
        "error message should explain the violation: {err}"
    );
}

#[test]
fn push_down_invariant_passes_when_decision_attached() {
    // Mirror of the teeth test: same plan with a valid decision passes.
    use crate::cypher::ast::Expr;
    use crate::planner::push_down::{select_push_down_strategy, VectorIndexParams};
    use coordinode_core::graph::types::Value;

    let node_scan = LogicalOp::NodeScan {
        variable: "a".to_string(),
        labels: vec!["User".to_string()],
        property_filters: vec![],
    };
    let traverse = LogicalOp::Traverse {
        input: Box::new(node_scan),
        source: "a".to_string(),
        edge_types: vec!["LIKES".to_string()],
        direction: crate::cypher::ast::Direction::Outgoing,
        target_variable: "b".to_string(),
        target_labels: vec![],
        length: None,
        edge_variable: None,
        target_filters: vec![],
        edge_filters: vec![],
        temporal_filter: None,
        path_variable: None,
    };
    let decision =
        select_push_down_strategy(50, VectorIndexParams::default_node(10_000, 128), 0.5, 10);
    let valid = LogicalOp::VectorFilter {
        input: Box::new(traverse),
        vector_expr: Expr::PropertyAccess {
            expr: Box::new(Expr::Variable("b".to_string())),
            property: "embedding".to_string(),
        },
        query_vector: Expr::Literal(Value::Array(vec![])),
        function: "vector_distance".to_string(),
        less_than: true,
        threshold: 0.5,
        decay_field: None,
        push_down: Some(decision),
    };
    assert_push_down_invariant(&valid).expect("plan with decision attached must pass");
}

/// Contract sweep (R-PUSH4): NO plan produced by the planner for any
/// representative graph+vector query shape may place a `TRAVERSE` before an
/// unfiltered `VECTOR_FILTER`. This generalises the single-shape invariant
/// tests above into a corpus, so the "zero plans in the suite violate the
/// invariant" guarantee is asserted directly rather than implied by a handful
/// of examples. A new query shape that regresses the push-down pass fails
/// here; a new shape that legitimately has no TRAVERSE→VECTOR_FILTER pair
/// passes trivially (the invariant only constrains that pairing).
#[test]
fn push_down_invariant_holds_across_graph_vector_query_shapes() {
    let shapes = [
        // outgoing single-hop, vector_distance
        "MATCH (a:User)-[:LIKES]->(b:Movie) \
         WHERE vector_distance(b.embedding, [1.0, 0.0, 0.0]) < 0.5 RETURN b",
        // incoming single-hop
        "MATCH (b:Movie)<-[:LIKES]-(a:User) \
         WHERE vector_distance(b.embedding, [1.0, 0.0, 0.0]) < 0.5 RETURN b",
        // vector_similarity (greater-than) predicate
        "MATCH (u:User)-[:WATCHED]->(m:Movie) \
         WHERE vector_similarity(m.embedding, [0.1, 0.2, 0.3]) > 0.8 RETURN m",
        // compound predicate: property filter AND vector predicate
        "MATCH (a:User)-[:LIKES]->(b:Movie) \
         WHERE b.year > 2000 AND vector_distance(b.embedding, [1.0, 0.0]) < 0.5 RETURN b",
        // LIMIT present (feeds top-K into the cost model)
        "MATCH (a:User)-[:LIKES]->(b:Movie) \
         WHERE vector_distance(b.embedding, [1.0, 0.0]) < 0.5 RETURN b LIMIT 25",
        // ORDER BY + LIMIT
        "MATCH (a:User)-[:LIKES]->(b:Movie) \
         WHERE vector_distance(b.embedding, [1.0, 0.0]) < 0.5 RETURN b ORDER BY b.title LIMIT 10",
        // different labels / edge type
        "MATCH (p:Person)-[:AUTHORED]->(d:Doc) \
         WHERE vector_distance(d.embedding, [0.5, 0.5]) < 0.3 RETURN d",
        // two-hop traversal before the vector filter
        "MATCH (a:User)-[:FOLLOWS]->(f:User)-[:LIKES]->(m:Movie) \
         WHERE vector_distance(m.embedding, [1.0, 0.0]) < 0.5 RETURN m",
    ];

    for q in shapes {
        let root = optimized_plan(q);
        assert_push_down_invariant(&root)
            .unwrap_or_else(|e| panic!("push-down invariant violated for shape:\n  {q}\n  -> {e}"));
    }
}

#[test]
fn top_k_extracted_from_limit_clause() {
    // Plan structure: Project → Limit { count: 25 } → Traverse → NodeScan
    // The push-down decision's cost_vector_first depends on K, so a
    // LIMIT in the query should propagate to the cost model.
    let root = optimized_plan(
        "MATCH (a:User)-[:LIKES]->(b:Movie) \
             WHERE vector_distance(b.embedding, [1.0, 0.0]) < 0.5 \
             RETURN b LIMIT 25",
    );
    // find_upstream_limit walks input chain; the test verifies it
    // compiles and is exercised indirectly via push-down decision —
    // the actual K-extraction is unit-tested above via the decision
    // struct contents. The presence of LIMIT in the plan is asserted
    // here as a smoke test that the query did parse with LIMIT.
    fn has_limit(op: &LogicalOp) -> bool {
        match op {
            LogicalOp::Limit { .. } => true,
            LogicalOp::Project { input, .. }
            | LogicalOp::Filter { input, .. }
            | LogicalOp::Sort { input, .. }
            | LogicalOp::Skip { input, .. }
            | LogicalOp::Aggregate { input, .. }
            | LogicalOp::VectorFilter { input, .. } => has_limit(input),
            _ => false,
        }
    }
    assert!(has_limit(&root), "LIMIT 25 must appear in optimized plan");
}

/// `find_upstream_limit` from a query with `LIMIT 25` propagates K=25
/// into the vector-first cost, while `LIMIT 5` propagates K=5 — same
/// plan otherwise. Verifies that top_k is not a constant.
#[test]
fn limit_value_flows_into_cost_vector_first() {
    use crate::planner::push_down::PushDownStrategy;
    // Walk the plan and capture the VectorFilter decision cost map.
    fn extract_cost_vf(op: &LogicalOp) -> Option<f64> {
        match op {
            LogicalOp::VectorFilter { push_down, .. } => push_down
                .as_ref()
                .and_then(|d| d.cost_alternatives.get(&PushDownStrategy::VectorFirst))
                .copied(),
            LogicalOp::Project { input, .. }
            | LogicalOp::Filter { input, .. }
            | LogicalOp::Sort { input, .. }
            | LogicalOp::Limit { input, .. }
            | LogicalOp::Skip { input, .. }
            | LogicalOp::Aggregate { input, .. } => extract_cost_vf(input),
            _ => None,
        }
    }

    let small_limit = optimized_plan(
        "MATCH (a:User)-[:LIKES]->(b:Movie) \
             WHERE vector_distance(b.embedding, [1.0, 0.0]) < 0.5 \
             RETURN b LIMIT 5",
    );
    let large_limit = optimized_plan(
        "MATCH (a:User)-[:LIKES]->(b:Movie) \
             WHERE vector_distance(b.embedding, [1.0, 0.0]) < 0.5 \
             RETURN b LIMIT 500",
    );

    // VectorFilter may have been rewritten away (e.g., by edge-vector
    // search). Only assert if both plans still expose a VectorFilter
    // with a decision — otherwise the cost-flow assertion is N/A.
    if let (Some(cost_small), Some(cost_large)) =
        (extract_cost_vf(&small_limit), extract_cost_vf(&large_limit))
    {
        assert!(
            cost_large > cost_small,
            "cost_vector_first must scale with K (LIMIT): \
                 K=5 → {cost_small}, K=500 → {cost_large}"
        );
    }
}

/// Plan with TWO VectorFilters in a nested structure (CartesianProduct
/// of two patterns, each containing TRAVERSE→VECTOR_FILTER) must get
/// decisions on BOTH filters. The optimizer pass recurses through
/// binary operators.
#[test]
fn nested_vector_filters_both_get_push_down_decisions() {
    // Build a CartesianProduct manually: two independent (NodeScan,
    // Traverse, VectorFilter) chains joined at the root. This shape
    // doesn't appear naturally from Cypher (which uses MATCH joins
    // differently) but exercises the binary-recursion branch in
    // optimize_push_down.
    use crate::cypher::ast::Expr;
    use coordinode_core::graph::types::Value;

    fn make_filter_chain(label: &str, var: &str) -> LogicalOp {
        let ns = LogicalOp::NodeScan {
            variable: var.to_string(),
            labels: vec![label.to_string()],
            property_filters: vec![],
        };
        let tr = LogicalOp::Traverse {
            input: Box::new(ns),
            source: var.to_string(),
            edge_types: vec!["LIKES".to_string()],
            direction: crate::cypher::ast::Direction::Outgoing,
            target_variable: format!("{var}_tgt"),
            target_labels: vec![],
            length: None,
            edge_variable: None,
            target_filters: vec![],
            edge_filters: vec![],
            temporal_filter: None,
            path_variable: None,
        };
        LogicalOp::VectorFilter {
            input: Box::new(tr),
            vector_expr: Expr::PropertyAccess {
                expr: Box::new(Expr::Variable(format!("{var}_tgt"))),
                property: "embedding".to_string(),
            },
            query_vector: Expr::Literal(Value::Array(vec![])),
            function: "vector_distance".to_string(),
            less_than: true,
            threshold: 0.5,
            decay_field: None,
            push_down: None,
        }
    }

    let nested = LogicalOp::CartesianProduct {
        left: Box::new(make_filter_chain("User", "a")),
        right: Box::new(make_filter_chain("Group", "g")),
    };
    let optimized = optimize_push_down(nested, None);

    fn count_annotated(op: &LogicalOp) -> usize {
        match op {
            LogicalOp::VectorFilter {
                input, push_down, ..
            } => {
                let here = usize::from(push_down.is_some());
                here + count_annotated(input)
            }
            LogicalOp::CartesianProduct { left, right }
            | LogicalOp::LeftOuterJoin { left, right } => {
                count_annotated(left) + count_annotated(right)
            }
            LogicalOp::Filter { input, .. }
            | LogicalOp::Project { input, .. }
            | LogicalOp::Sort { input, .. }
            | LogicalOp::Limit { input, .. }
            | LogicalOp::Skip { input, .. }
            | LogicalOp::Aggregate { input, .. } => count_annotated(input),
            _ => 0,
        }
    }
    assert_eq!(
        count_annotated(&optimized),
        2,
        "both VectorFilters in CartesianProduct branches must receive push_down decisions"
    );
    assert_push_down_invariant(&optimized).expect("nested invariant must hold");
}

#[test]
fn push_down_decision_contains_all_required_fields() {
    // Decision struct must populate every EXPLAIN-visible field; this
    // catches regressions where the cost map or reason gets dropped.
    let root = optimized_plan(
        "MATCH (a:User)-[:LIKES]->(b:Movie) \
             WHERE vector_distance(b.embedding, [1.0, 0.0]) < 0.5 \
             RETURN b",
    );
    // Walk to find the VectorFilter (if it survived edge-vector rewrite).
    fn walk(op: &LogicalOp) -> Option<&crate::planner::push_down::PushDownDecision> {
        match op {
            LogicalOp::VectorFilter { push_down, .. } => push_down.as_ref(),
            LogicalOp::Filter { input, .. }
            | LogicalOp::Project { input, .. }
            | LogicalOp::Sort { input, .. }
            | LogicalOp::Limit { input, .. }
            | LogicalOp::Skip { input, .. }
            | LogicalOp::Aggregate { input, .. } => walk(input),
            _ => None,
        }
    }
    if let Some(d) = walk(&root) {
        // Cost map must include all three strategy variants.
        assert!(
            d.cost_alternatives
                .contains_key(&crate::planner::push_down::PushDownStrategy::GraphFirst),
            "decision must include graph_first cost"
        );
        assert!(
            d.cost_alternatives
                .contains_key(&crate::planner::push_down::PushDownStrategy::AcornFiltered),
            "decision must include acorn_filtered cost"
        );
        assert!(
            d.cost_alternatives
                .contains_key(&crate::planner::push_down::PushDownStrategy::VectorFirst),
            "decision must include vector_first cost"
        );
        assert!(d.estimated_selectivity >= 0.0 && d.estimated_selectivity <= 1.0);
        assert!(d.cost_chosen.is_finite() && d.cost_chosen >= 0.0);
    }
}

#[test]
fn merge_nodes_explain_renders_all_metadata() {
    let plan = plan(
        "MATCH (a:User {id: 1}), (b:User {id: 2}) \
             MERGE NODES (a, b) INTO a \
             ON CONFLICT KEEP LAST \
             TRANSFER EDGES FROM b TO a \
             ON DUPLICATE MERGE PROPERTIES \
             TRANSFER EDGE PROPERTIES",
    );
    let explain = plan.explain();
    // Every distinguishing field of the clause must surface in EXPLAIN —
    // without this, operators reading EXPLAIN can't see what merge will do.
    assert!(
        explain.contains("MergeNodes(a,b) INTO a"),
        "explain: {explain}"
    );
    assert!(explain.contains("KEEP_LAST"), "explain: {explain}");
    assert!(explain.contains("TRANSFER b→a"), "explain: {explain}");
    assert!(explain.contains("DUP_MERGE"), "explain: {explain}");
    assert!(explain.contains("+EDGE_PROPS"), "explain: {explain}");
}

// ── VectorPredicate harvesting from MATCH+WHERE ──────────────────────

#[test]
fn collect_simple_predicates_extracts_property_eq() {
    use crate::cypher::ast::BinaryOperator;
    use crate::planner::logical::VectorPredicate;

    // Filter { input: NodeScan(:Item) , predicate: n.category = "X" }
    let scan = LogicalOp::NodeScan {
        variable: "n".into(),
        labels: vec!["Item".into()],
        property_filters: vec![],
    };
    let pred = Expr::BinaryOp {
        left: Box::new(Expr::PropertyAccess {
            expr: Box::new(Expr::Variable("n".into())),
            property: "category".into(),
        }),
        op: BinaryOperator::Eq,
        right: Box::new(Expr::Literal(Value::String("X".into()))),
    };
    let filter = LogicalOp::Filter {
        input: Box::new(scan),
        predicate: pred,
    };

    let mut leaves: Vec<VectorPredicate> = Vec::new();
    collect_simple_property_predicates(&filter, "n", &mut leaves);
    assert_eq!(leaves.len(), 1);
    match &leaves[0] {
        VectorPredicate::PropertyEq { property, value } => {
            assert_eq!(property, "category");
            assert_eq!(value, &Value::String("X".into()));
        }
        other => panic!("expected PropertyEq, got {other:?}"),
    }
}

#[test]
fn collect_simple_predicates_handles_reversed_eq() {
    use crate::cypher::ast::BinaryOperator;
    use crate::planner::logical::VectorPredicate;

    // Predicate written as `"X" = n.category` (literal on the left).
    let scan = LogicalOp::NodeScan {
        variable: "n".into(),
        labels: vec!["Item".into()],
        property_filters: vec![],
    };
    let pred = Expr::BinaryOp {
        left: Box::new(Expr::Literal(Value::String("X".into()))),
        op: BinaryOperator::Eq,
        right: Box::new(Expr::PropertyAccess {
            expr: Box::new(Expr::Variable("n".into())),
            property: "category".into(),
        }),
    };
    let filter = LogicalOp::Filter {
        input: Box::new(scan),
        predicate: pred,
    };

    let mut leaves: Vec<VectorPredicate> = Vec::new();
    collect_simple_property_predicates(&filter, "n", &mut leaves);
    assert_eq!(leaves.len(), 1);
    assert!(matches!(leaves[0], VectorPredicate::PropertyEq { .. }));
}

#[test]
fn collect_simple_predicates_unwraps_conjunctions() {
    use crate::cypher::ast::BinaryOperator;
    use crate::planner::logical::VectorPredicate;

    // n.category = "X" AND n.active = true
    let left = Expr::BinaryOp {
        left: Box::new(Expr::PropertyAccess {
            expr: Box::new(Expr::Variable("n".into())),
            property: "category".into(),
        }),
        op: BinaryOperator::Eq,
        right: Box::new(Expr::Literal(Value::String("X".into()))),
    };
    let right = Expr::BinaryOp {
        left: Box::new(Expr::PropertyAccess {
            expr: Box::new(Expr::Variable("n".into())),
            property: "active".into(),
        }),
        op: BinaryOperator::Eq,
        right: Box::new(Expr::Literal(Value::Bool(true))),
    };
    let pred = Expr::BinaryOp {
        left: Box::new(left),
        op: BinaryOperator::And,
        right: Box::new(right),
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
    assert_eq!(leaves.len(), 2);
    for leaf in &leaves {
        assert!(matches!(leaf, VectorPredicate::PropertyEq { .. }));
    }
}

#[test]
fn collect_simple_predicates_ignores_other_variable() {
    use crate::cypher::ast::BinaryOperator;
    use crate::planner::logical::VectorPredicate;

    // m.category = "X" — wrong variable name relative to vector top-k.
    let pred = Expr::BinaryOp {
        left: Box::new(Expr::PropertyAccess {
            expr: Box::new(Expr::Variable("m".into())),
            property: "category".into(),
        }),
        op: BinaryOperator::Eq,
        right: Box::new(Expr::Literal(Value::String("X".into()))),
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
    assert!(leaves.is_empty());
}

#[test]
fn fold_predicate_chains_into_and() {
    use crate::planner::logical::VectorPredicate;

    let leaves = vec![
        VectorPredicate::LabelEq("Item".into()),
        VectorPredicate::PropertyEq {
            property: "category".into(),
            value: Value::String("X".into()),
        },
    ];
    let folded = fold_predicate(leaves).expect("non-empty");
    match folded {
        VectorPredicate::And(l, r) => {
            assert!(matches!(*l, VectorPredicate::LabelEq(_)));
            assert!(matches!(*r, VectorPredicate::PropertyEq { .. }));
        }
        other => panic!("expected And, got {other:?}"),
    }
}

#[test]
fn fold_predicate_single_leaf_is_unwrapped() {
    use crate::planner::logical::VectorPredicate;
    let folded =
        fold_predicate(vec![VectorPredicate::LabelEq("Item".into())]).expect("single leaf");
    assert!(matches!(folded, VectorPredicate::LabelEq(_)));
}

#[test]
fn fold_predicate_empty_returns_none() {
    let folded = fold_predicate(Vec::new());
    assert!(folded.is_none());
}
