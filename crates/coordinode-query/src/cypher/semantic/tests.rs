use super::*;
use crate::cypher::parser::parse;
use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType};

fn parse_and_analyze(input: &str) -> Vec<SemanticError> {
    let query = parse(input).expect("parse should succeed");
    analyze(&query, None)
}

fn parse_and_analyze_with_schema(input: &str, schema: &dyn SchemaProvider) -> Vec<SemanticError> {
    let query = parse(input).expect("parse should succeed");
    analyze(&query, Some(schema))
}

fn make_schema() -> MapSchemaProvider {
    let mut schema = MapSchemaProvider::new();

    let mut user = LabelSchema::new_node_id("User");
    user.add_property(PropertyDef::new("name", PropertyType::String));
    user.add_property(PropertyDef::new("age", PropertyType::Int));
    user.add_property(PropertyDef::new("email", PropertyType::String));
    user.set_mode(coordinode_core::schema::definition::SchemaMode::Strict);
    schema.add_label(user);

    let mut movie = LabelSchema::new_node_id("Movie");
    movie.add_property(PropertyDef::new("title", PropertyType::String));
    schema.add_label(movie);

    let knows = EdgeTypeSchema::new("KNOWS");
    schema.add_edge_type(knows);

    let mut likes = EdgeTypeSchema::new("LIKES");
    likes.add_property(PropertyDef::new("since", PropertyType::Timestamp));
    schema.add_edge_type(likes);

    schema
}

// -- Variable binding --

#[test]
fn valid_match_return() {
    let errors = parse_and_analyze("MATCH (n:User) RETURN n");
    assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
}

#[test]
fn valid_match_where_return() {
    let errors = parse_and_analyze("MATCH (n) WHERE n.age > 25 RETURN n.name");
    assert!(errors.is_empty());
}

#[test]
fn undefined_variable_in_return() {
    let errors = parse_and_analyze("MATCH (n) RETURN m");
    assert_eq!(errors.len(), 1);
    assert!(matches!(
        errors[0],
        SemanticError::UndefinedVariable { ref name } if name == "m"
    ));
}

#[test]
fn undefined_variable_in_where() {
    let errors = parse_and_analyze("MATCH (n) WHERE m.age > 25 RETURN n");
    assert!(errors.iter().any(|e| matches!(
        e,
        SemanticError::UndefinedVariable { ref name } if name == "m"
    )));
}

#[test]
fn relationship_variable_defined() {
    let errors = parse_and_analyze("MATCH (a)-[r:KNOWS]->(b) RETURN a, r, b");
    assert!(errors.is_empty());
}

#[test]
fn multiple_patterns_define_vars() {
    let errors = parse_and_analyze("MATCH (a:User), (b:Movie) RETURN a, b");
    assert!(errors.is_empty());
}

// -- Scope chain (WITH barrier) --

#[test]
fn with_projects_variables() {
    let errors = parse_and_analyze("MATCH (n:User) WITH n RETURN n");
    assert!(errors.is_empty());
}

#[test]
fn with_hides_unprojected_variables() {
    let errors = parse_and_analyze("MATCH (a:User), (b:Movie) WITH a RETURN b");
    // b is not projected through WITH, so RETURN b is an error
    assert!(errors.iter().any(|e| matches!(
        e,
        SemanticError::UndefinedVariable { ref name } if name == "b"
    )));
}

#[test]
fn with_alias_creates_new_variable() {
    let errors = parse_and_analyze("MATCH (n:User) WITH n.name AS username RETURN username");
    assert!(errors.is_empty());
}

#[test]
fn with_alias_old_var_not_visible() {
    let errors = parse_and_analyze("MATCH (n:User) WITH n.name AS username RETURN n");
    // n is not projected, only username
    assert!(errors.iter().any(|e| matches!(
        e,
        SemanticError::UndefinedVariable { ref name } if name == "n"
    )));
}

// -- WITH * (Star projection) --

/// `WITH *` keeps all upstream variables in scope.
#[test]
fn with_star_keeps_variables() {
    let errors = parse_and_analyze("MATCH (n:User) WITH * RETURN n");
    assert!(errors.is_empty(), "WITH * must keep n in scope: {errors:?}");
}

/// `WITH *, expr AS alias` — Star + explicit alias: both the original variable
/// AND the new alias must be in scope after WITH.
///
/// REGRESSION: `analyze_with` used to `break` when it encountered Star, so any
/// aliases listed after Star (e.g. `_dist`) were never added to the new scope.
/// That caused "undefined variable" errors for ORDER BY / RETURN on the alias.
#[test]
fn with_star_plus_alias_both_in_scope() {
    let errors = parse_and_analyze("MATCH (n:User) WITH *, n.age AS age_alias RETURN n, age_alias");
    assert!(
        errors.is_empty(),
        "WITH *, expr AS alias must put both n and age_alias in scope: {errors:?}"
    );
}

/// ORDER BY on an alias introduced alongside Star must work — this is the
/// exact pattern used in vector_search / hybrid_search:
///   `WITH *, vector_distance(n.emb, $qv) AS _dist ORDER BY _dist LIMIT k`
#[test]
fn with_star_alias_usable_in_order_by() {
    let errors = parse_and_analyze(
        "MATCH (n:User) \
             WITH *, n.age AS _score \
             ORDER BY _score DESC \
             LIMIT 10 \
             RETURN n, _score",
    );
    assert!(
        errors.is_empty(),
        "alias from WITH * must be usable in ORDER BY: {errors:?}"
    );
}

/// Original variable from Star projection must still be visible in ORDER BY.
#[test]
fn with_star_original_var_usable_in_order_by() {
    let errors = parse_and_analyze("MATCH (n:User) WITH * ORDER BY n.name RETURN n");
    assert!(
        errors.is_empty(),
        "WITH * must keep n accessible in ORDER BY: {errors:?}"
    );
}

#[test]
fn unwind_introduces_variable() {
    let errors = parse_and_analyze("UNWIND [1, 2, 3] AS x RETURN x");
    assert!(errors.is_empty());
}

// -- Write clause variable checks --

#[test]
fn create_introduces_variables() {
    let errors = parse_and_analyze("CREATE (n:User {name: 'Alice'}) RETURN n");
    assert!(errors.is_empty());
}

#[test]
fn set_requires_defined_variable() {
    let errors = parse_and_analyze("MATCH (n:User) SET m.name = 'Bob' RETURN n");
    assert!(errors.iter().any(|e| matches!(
        e,
        SemanticError::UndefinedVariable { ref name } if name == "m"
    )));
}

#[test]
fn delete_requires_variable() {
    let errors = parse_and_analyze("MATCH (n:User) DELETE n");
    assert!(errors.is_empty());
}

#[test]
fn delete_undefined_variable() {
    let errors = parse_and_analyze("MATCH (n:User) DELETE m");
    assert!(errors.iter().any(|e| matches!(
        e,
        SemanticError::UndefinedVariable { ref name } if name == "m"
    )));
}

#[test]
fn merge_introduces_variable() {
    let errors = parse_and_analyze("MERGE (n:User {email: 'alice@test.com'}) RETURN n");
    assert!(errors.is_empty());
}

#[test]
fn merge_on_match_set_uses_merge_var() {
    let errors = parse_and_analyze(
        "MERGE (n:User {email: 'test@test.com'}) \
             ON MATCH SET n.name = 'Bob' \
             RETURN n",
    );
    assert!(errors.is_empty());
}

// -- RETURN * --

#[test]
fn return_star_with_variables() {
    let errors = parse_and_analyze("MATCH (n) RETURN *");
    assert!(errors.is_empty());
}

#[test]
fn return_star_no_variables() {
    // Edge case: RETURN * without any MATCH
    // This requires creating a query with just RETURN *
    // but our parser requires at least one clause before RETURN
    // Actually, in Cypher you can have standalone RETURN
    // Let's test with an empty scope
    let query = Query {
        clauses: vec![Clause::Return(ReturnClause {
            distinct: false,
            items: vec![ReturnItem {
                expr: Expr::Star,
                alias: None,
            }],
        })],
        hints: Vec::new(),
        unions: Vec::new(),
    };
    let errors = analyze(&query, None);
    assert!(errors
        .iter()
        .any(|e| matches!(e, SemanticError::ReturnStarEmpty)));
}

// -- ORDER BY alias resolution --

#[test]
fn order_by_return_alias() {
    // ORDER BY should see aliases defined in RETURN ... AS
    let errors = parse_and_analyze("MATCH (n) RETURN n.age AS age ORDER BY age");
    assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
}

#[test]
fn order_by_return_alias_expression() {
    // Alias from a function call in RETURN
    let errors = parse_and_analyze("MATCH (n) RETURN count(n) AS cnt ORDER BY cnt");
    assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
}

#[test]
fn order_by_original_variable_still_works() {
    // ORDER BY should still see original variables (not just aliases)
    let errors = parse_and_analyze("MATCH (n) RETURN n.name ORDER BY n.age");
    assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
}

#[test]
fn order_by_undefined_alias_fails() {
    // Alias not defined anywhere should still fail
    let errors = parse_and_analyze("MATCH (n) RETURN n.name ORDER BY nonexistent");
    assert!(errors.iter().any(|e| matches!(
        e,
        SemanticError::UndefinedVariable { ref name } if name == "nonexistent"
    )));
}

// -- Schema validation --

#[test]
fn unknown_label() {
    let schema = make_schema();
    let errors = parse_and_analyze_with_schema("MATCH (n:NonExistent) RETURN n", &schema);
    assert!(errors.iter().any(|e| matches!(
        e,
        SemanticError::UnknownLabel { ref name } if name == "NonExistent"
    )));
}

#[test]
fn known_label() {
    let schema = make_schema();
    let errors = parse_and_analyze_with_schema("MATCH (n:User) RETURN n", &schema);
    assert!(errors.is_empty());
}

#[test]
fn unknown_edge_type() {
    let schema = make_schema();
    let errors =
        parse_and_analyze_with_schema("MATCH (a)-[:NONEXISTENT]->(b) RETURN a, b", &schema);
    assert!(errors.iter().any(|e| matches!(
        e,
        SemanticError::UnknownEdgeType { ref name } if name == "NONEXISTENT"
    )));
}

#[test]
fn known_edge_type() {
    let schema = make_schema();
    let errors = parse_and_analyze_with_schema("MATCH (a)-[:KNOWS]->(b) RETURN a, b", &schema);
    assert!(errors.is_empty());
}

#[test]
fn unknown_property_on_strict_label() {
    let schema = make_schema();
    let errors = parse_and_analyze_with_schema(
        "MATCH (n:User) WHERE n.nonexistent = 'foo' RETURN n",
        &schema,
    );
    assert!(errors.iter().any(|e| matches!(
        e,
        SemanticError::UnknownProperty { ref label, ref property }
            if label == "User" && property == "nonexistent"
    )));
}

#[test]
fn known_property_on_strict_label() {
    let schema = make_schema();
    let errors =
        parse_and_analyze_with_schema("MATCH (n:User) WHERE n.name = 'Alice' RETURN n", &schema);
    assert!(errors.is_empty());
}

#[test]
fn no_schema_skips_validation() {
    // Without schema, unknown labels/properties are not errors
    let errors = parse_and_analyze("MATCH (n:Whatever) WHERE n.anything = 'foo' RETURN n");
    assert!(errors.is_empty());
}

// -- Complex queries --

#[test]
fn complex_with_chain() {
    let errors = parse_and_analyze(
        "MATCH (n:User)-[:KNOWS]->(m:User) \
             WITH n, count(*) AS friend_count \
             WHERE friend_count > 5 \
             RETURN n.name, friend_count",
    );
    assert!(errors.is_empty());
}

#[test]
fn complex_match_create_return() {
    let errors = parse_and_analyze(
        "MATCH (a:User {name: 'Alice'}), (b:User {name: 'Bob'}) \
             CREATE (a)-[:KNOWS]->(b) \
             RETURN a, b",
    );
    assert!(errors.is_empty());
}

#[test]
fn remove_requires_defined_variable() {
    let errors = parse_and_analyze("MATCH (n) REMOVE m.age");
    assert!(errors.iter().any(|e| matches!(
        e,
        SemanticError::UndefinedVariable { ref name } if name == "m"
    )));
}

#[test]
fn set_label_valid() {
    let errors = parse_and_analyze("MATCH (n) SET n:Admin");
    assert!(errors.is_empty());
}

#[test]
fn parameters_always_valid() {
    let errors = parse_and_analyze("MATCH (n:User {id: $userId}) RETURN n");
    assert!(errors.is_empty());
}

#[test]
fn aggregation_in_return() {
    let errors = parse_and_analyze("MATCH (n:User) RETURN n.city, count(*) AS cnt");
    assert!(errors.is_empty());
}

#[test]
fn nested_property_access() {
    let errors = parse_and_analyze("MATCH (n:User) WHERE n.age > 18 AND n.name = 'Alice' RETURN n");
    assert!(errors.is_empty());
}

#[test]
fn union_matching_columns_ok() {
    let errors =
        parse_and_analyze("MATCH (a) RETURN a.name AS name UNION MATCH (b) RETURN b.name AS name");
    assert!(
        !errors
            .iter()
            .any(|e| matches!(e, SemanticError::UnionColumnMismatch { .. })),
        "matching UNION columns should not error: {errors:?}"
    );
}

#[test]
fn union_mismatched_columns_error() {
    let errors =
        parse_and_analyze("MATCH (a) RETURN a.name AS name UNION MATCH (b) RETURN b.age AS age");
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, SemanticError::UnionColumnMismatch { .. })),
        "mismatched UNION columns should error: {errors:?}"
    );
}
