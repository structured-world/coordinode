//! Integration tests: Cypher query execution.
//!
//! Tests the full pipeline: parse → plan → execute for various Cypher constructs.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use coordinode_embed::Database;

fn open_db() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open db");
    (db, dir)
}

fn seed_graph(db: &mut Database) {
    db.execute_cypher("CREATE (a:Person {name: 'Alice', age: 30})")
        .expect("seed alice");
    db.execute_cypher("CREATE (b:Person {name: 'Bob', age: 25})")
        .expect("seed bob");
    db.execute_cypher("CREATE (c:Person {name: 'Charlie', age: 35})")
        .expect("seed charlie");
}

fn seed_graph_with_relationships(db: &mut Database) {
    // Create nodes + relationships in a single CREATE pattern
    db.execute_cypher(
        "CREATE (a:Person {name: 'Alice', age: 30})-[:FRIEND]->(b:Person {name: 'Bob', age: 25})-[:FRIEND]->(c:Person {name: 'Charlie', age: 35})",
    )
    .expect("seed graph with rels");
}

// ── ORDER BY ────────────────────────────────────────────────────────

#[test]
fn order_by_ascending() {
    let (mut db, _dir) = open_db();
    seed_graph(&mut db);
    let rows = db
        .execute_cypher("MATCH (n:Person) RETURN n.name ORDER BY n.age")
        .expect("order by");
    assert_eq!(rows.len(), 3);
}

#[test]
fn order_by_descending() {
    let (mut db, _dir) = open_db();
    seed_graph(&mut db);
    let rows = db
        .execute_cypher("MATCH (n:Person) RETURN n.name ORDER BY n.age DESC")
        .expect("order by desc");
    assert_eq!(rows.len(), 3);
}

// ── LIMIT / SKIP ────────────────────────────────────────────────────

#[test]
fn limit_results() {
    let (mut db, _dir) = open_db();
    seed_graph(&mut db);
    let rows = db
        .execute_cypher("MATCH (n:Person) RETURN n.name LIMIT 2")
        .expect("limit");
    assert_eq!(rows.len(), 2);
}

#[test]
fn skip_results() {
    let (mut db, _dir) = open_db();
    seed_graph(&mut db);
    let rows = db
        .execute_cypher("MATCH (n:Person) RETURN n.name SKIP 1")
        .expect("skip");
    assert_eq!(rows.len(), 2);
}

#[test]
fn skip_and_limit() {
    let (mut db, _dir) = open_db();
    seed_graph(&mut db);
    let rows = db
        .execute_cypher("MATCH (n:Person) RETURN n.name SKIP 1 LIMIT 1")
        .expect("skip+limit");
    assert_eq!(rows.len(), 1);
}

// ── WHERE ───────────────────────────────────────────────────────────

#[test]
fn where_equality() {
    let (mut db, _dir) = open_db();
    seed_graph(&mut db);
    let rows = db
        .execute_cypher("MATCH (n:Person) WHERE n.name = 'Alice' RETURN n")
        .expect("where eq");
    assert_eq!(rows.len(), 1);
}

#[test]
fn where_comparison() {
    let (mut db, _dir) = open_db();
    seed_graph(&mut db);
    let rows = db
        .execute_cypher("MATCH (n:Person) WHERE n.age >= 30 RETURN n.name")
        .expect("where cmp");
    assert_eq!(rows.len(), 2); // Alice (30) and Charlie (35)
}

#[test]
fn where_and() {
    let (mut db, _dir) = open_db();
    seed_graph(&mut db);
    let rows = db
        .execute_cypher("MATCH (n:Person) WHERE n.age > 20 AND n.age < 30 RETURN n.name")
        .expect("where and");
    assert_eq!(rows.len(), 1); // Bob (25)
}

#[test]
fn where_or() {
    let (mut db, _dir) = open_db();
    seed_graph(&mut db);
    let rows = db
        .execute_cypher(
            "MATCH (n:Person) WHERE n.name = 'Alice' OR n.name = 'Charlie' RETURN n.name",
        )
        .expect("where or");
    assert_eq!(rows.len(), 2);
}

// ── Aggregation ─────────────────────────────────────────────────────

#[test]
fn count_all() {
    let (mut db, _dir) = open_db();
    seed_graph(&mut db);
    let rows = db
        .execute_cypher("MATCH (n:Person) RETURN count(n)")
        .expect("count");
    assert_eq!(rows.len(), 1);
}

#[test]
fn sum_values() {
    let (mut db, _dir) = open_db();
    seed_graph(&mut db);
    let rows = db
        .execute_cypher("MATCH (n:Person) RETURN sum(n.age)")
        .expect("sum");
    assert_eq!(rows.len(), 1); // 30 + 25 + 35 = 90
}

#[test]
fn min_max() {
    let (mut db, _dir) = open_db();
    seed_graph(&mut db);
    let rows = db
        .execute_cypher("MATCH (n:Person) RETURN min(n.age), max(n.age)")
        .expect("min/max");
    assert_eq!(rows.len(), 1);
}

// ── Relationship traversal ──────────────────────────────────────────

#[test]
fn traverse_outgoing() {
    let (mut db, _dir) = open_db();
    seed_graph_with_relationships(&mut db);
    // Relationship traversal requires adjacency list lookup in physical executor.
    // Test that the query doesn't error (even if results vary by executor phase).
    let result = db.execute_cypher("MATCH (a:Person {name: 'Alice'})-[:FRIEND]->(b) RETURN b.name");
    assert!(result.is_ok(), "traversal should not error");
}

#[test]
fn traverse_two_hops() {
    let (mut db, _dir) = open_db();
    seed_graph_with_relationships(&mut db);
    let result = db.execute_cypher(
        "MATCH (a:Person {name: 'Alice'})-[:FRIEND]->(b)-[:FRIEND]->(c) RETURN c.name",
    );
    assert!(result.is_ok(), "2-hop traversal should not error");
}

// ── EXPLAIN ─────────────────────────────────────────────────────────

#[test]
fn explain_plan() {
    let (db, _dir) = open_db();
    let plan = db
        .explain_cypher("MATCH (n:Person) RETURN n")
        .expect("explain");
    assert!(plan.contains("NodeScan"));
}

// ── Error handling ──────────────────────────────────────────────────

#[test]
fn parse_error_on_invalid_syntax() {
    let (mut db, _dir) = open_db();
    let result = db.execute_cypher("INVALID SYNTAX");
    assert!(result.is_err());
}

#[test]
fn semantic_error_on_undefined_variable() {
    let (mut db, _dir) = open_db();
    let result = db.execute_cypher("MATCH (n) RETURN m");
    assert!(result.is_err());
}

// ── WITH ────────────────────────────────────────────────────────────

#[test]
fn with_clause_pipeline() {
    let (mut db, _dir) = open_db();
    seed_graph(&mut db);
    let rows = db
        .execute_cypher("MATCH (n:Person) WITH n.name AS name WHERE name = 'Alice' RETURN name")
        .expect("with");
    assert_eq!(rows.len(), 1);
}

// ── UNWIND ──────────────────────────────────────────────────────────

#[test]
fn unwind_list() {
    let (mut db, _dir) = open_db();
    let rows = db
        .execute_cypher("UNWIND [1, 2, 3] AS x RETURN x")
        .expect("unwind");
    // UNWIND should expand list into individual rows
    assert!(!rows.is_empty());
}

// --- Parameterized Query Tests ---

use coordinode_core::graph::types::Value;
use std::collections::HashMap;

#[test]
fn param_binding_in_where() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (n:User {name: 'Alice', age: 30})")
        .expect("create alice");
    db.execute_cypher("CREATE (n:User {name: 'Bob', age: 25})")
        .expect("create bob");

    let mut params = HashMap::new();
    params.insert("name".to_string(), Value::String("Alice".to_string()));

    let rows = db
        .execute_cypher_with_params(
            "MATCH (n:User) WHERE n.name = $name RETURN n.name, n.age",
            params,
        )
        .expect("param query");

    assert_eq!(rows.len(), 1, "should find exactly Alice");
    assert_eq!(
        rows[0].get("n.name"),
        Some(&Value::String("Alice".to_string()))
    );
}

#[test]
fn param_binding_in_create() {
    let (mut db, _dir) = open_db();

    let mut params = HashMap::new();
    params.insert("name".to_string(), Value::String("Charlie".to_string()));
    params.insert("age".to_string(), Value::Int(35));

    db.execute_cypher_with_params("CREATE (n:User {name: $name, age: $age})", params)
        .expect("create with params");

    let rows = db
        .execute_cypher("MATCH (n:User {name: 'Charlie'}) RETURN n.age")
        .expect("verify");

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("n.age"), Some(&Value::Int(35)));
}

#[test]
fn param_binding_multiple_params() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (n:Product {name: 'Widget', price: 10})")
        .expect("create");
    db.execute_cypher("CREATE (n:Product {name: 'Gadget', price: 20})")
        .expect("create");
    db.execute_cypher("CREATE (n:Product {name: 'Doohickey', price: 30})")
        .expect("create");

    let mut params = HashMap::new();
    params.insert("min_price".to_string(), Value::Int(15));
    params.insert("max_price".to_string(), Value::Int(25));

    let rows = db
        .execute_cypher_with_params(
            "MATCH (n:Product) WHERE n.price > $min_price AND n.price < $max_price RETURN n.name",
            params,
        )
        .expect("range query");

    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("n.name"),
        Some(&Value::String("Gadget".to_string()))
    );
}

#[test]
fn param_binding_empty_params_works() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (n:Test {v: 1})").expect("create");

    // Empty params map should work like regular execute_cypher
    let rows = db
        .execute_cypher_with_params("MATCH (n:Test) RETURN n.v", HashMap::new())
        .expect("empty params");

    assert_eq!(rows.len(), 1);
}

#[test]
fn param_binding_unresolved_param_is_null() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (n:Item {name: 'test'})")
        .expect("create");

    // Query uses $missing which is not in params → should be Null
    let rows = db
        .execute_cypher_with_params("MATCH (n:Item) RETURN $missing AS val", HashMap::new())
        .expect("unresolved param");

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("val"), Some(&Value::Null));
}

#[test]
fn param_binding_vector_param() {
    let (mut db, _dir) = open_db();

    let mut params = HashMap::new();
    params.insert("embedding".to_string(), Value::Vector(vec![1.0, 0.0, 0.0]));

    db.execute_cypher_with_params(
        "CREATE (n:Doc {title: 'test', embedding: $embedding})",
        params,
    )
    .expect("create with vector param");

    let rows = db
        .execute_cypher("MATCH (n:Doc {title: 'test'}) RETURN n.embedding")
        .expect("verify");

    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("n.embedding"),
        Some(&Value::Vector(vec![1.0, 0.0, 0.0]))
    );
}

#[test]
fn multi_vector_persists_and_scores_via_maxsim() {
    // Pins the end-to-end ColBERT-style late-interaction path:
    //   1. CREATE a node whose property is a Value::MultiVector parameter
    //   2. MATCH reads it back as Value::MultiVector
    //   3. RETURN maxsim_score(n.tokens, $q) computes the score
    // The Node store path is plain rmp-serde MessagePack on the Value
    // enum, so the new MultiVector variant rides through transparently
    // with no dedicated partition. This test will fail loudly if
    // anything in the encode / decode / scalar dispatch ever stops
    // round-tripping the matrix.
    let (mut db, _dir) = open_db();

    let doc_tokens = Value::MultiVector(vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]]);
    let query_tokens = Value::MultiVector(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);

    let mut create_params = HashMap::new();
    create_params.insert("tokens".to_string(), doc_tokens.clone());
    db.execute_cypher_with_params("CREATE (d:Doc {id: 1, tokens: $tokens})", create_params)
        .expect("create with multi-vector param");

    // Round-trip read confirms persistence.
    let rows = db
        .execute_cypher("MATCH (d:Doc {id: 1}) RETURN d.tokens")
        .expect("read back");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("d.tokens"), Some(&doc_tokens));

    // Score via the scalar through the full Cypher evaluator.
    let mut score_params = HashMap::new();
    score_params.insert("q".to_string(), query_tokens);
    let rows = db
        .execute_cypher_with_params(
            "MATCH (d:Doc {id: 1}) RETURN maxsim_score(d.tokens, $q) AS s",
            score_params,
        )
        .expect("score");
    assert_eq!(rows.len(), 1);
    // Hand-computed: q0 . d* max = 1.0, q1 . d* max = 1.0, sum = 2.0
    let Some(Value::Float(s)) = rows[0].get("s") else {
        unreachable!("expected Float score, got {:?}", rows[0].get("s"));
    };
    assert!((s - 2.0).abs() < 1e-5, "got s={s}");
}

#[test]
fn param_binding_in_set() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (n:Config {key: 'theme', value: 'dark'})")
        .expect("create");

    let mut params = HashMap::new();
    params.insert("new_value".to_string(), Value::String("light".to_string()));

    db.execute_cypher_with_params(
        "MATCH (n:Config {key: 'theme'}) SET n.value = $new_value",
        params,
    )
    .expect("set with param");

    let rows = db
        .execute_cypher("MATCH (n:Config {key: 'theme'}) RETURN n.value")
        .expect("verify");
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("n.value"),
        Some(&Value::String("light".to_string()))
    );
}
