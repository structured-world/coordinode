//! Integration tests: cross-MATCH WHERE predicate lifting.
//!
//! Verifies that WHERE predicates referencing variables from multiple
//! MATCH clauses are correctly placed ABOVE the CartesianProduct,
//! not inside a branch where some variables are not yet in scope.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_core::graph::types::Value;
use coordinode_embed::Database;

fn open_db() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open db");
    (db, dir)
}

// ── Basic cross-MATCH WHERE ─────────────────────────────────────────

/// MATCH (a:User) MATCH (b:Post) WHERE a.name = b.author
/// should return rows where the join condition matches.
/// Previously returned empty because `a` was null in the second MATCH's branch.
#[test]
fn cross_match_where_property_join() {
    let (mut db, _dir) = open_db();

    db.execute_cypher("CREATE (u:User {name: 'Alice'})")
        .expect("create user");
    db.execute_cypher("CREATE (p:Post {title: 'Hello', author: 'Alice'})")
        .expect("create post");
    db.execute_cypher("CREATE (p:Post {title: 'Bye', author: 'Bob'})")
        .expect("create post2");

    let rows = db
        .execute_cypher(
            "MATCH (a:User) MATCH (b:Post) WHERE a.name = b.author RETURN a.name, b.title",
        )
        .expect("cross-match join");

    assert_eq!(rows.len(), 1, "only Alice's post should match");
    assert_eq!(rows[0].get("b.title"), Some(&Value::String("Hello".into())));
}

/// Cross-MATCH with both variables in WHERE — reversed order.
#[test]
fn cross_match_where_reversed_variable_order() {
    let (mut db, _dir) = open_db();

    db.execute_cypher("CREATE (u:User {name: 'Bob'})")
        .expect("create");
    db.execute_cypher("CREATE (c:Comment {text: 'Great', author: 'Bob'})")
        .expect("create");

    let rows = db
        .execute_cypher(
            "MATCH (c:Comment) MATCH (u:User) WHERE c.author = u.name RETURN u.name, c.text",
        )
        .expect("reversed cross-match");

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("c.text"), Some(&Value::String("Great".into())));
}

// ── Mixed local + cross predicates ──────────────────────────────────

/// WHERE with both local predicates (single-branch) and cross predicates.
/// Local predicates should stay in the branch for early filtering.
#[test]
fn cross_match_mixed_local_and_cross_predicates() {
    let (mut db, _dir) = open_db();

    db.execute_cypher("CREATE (u:User {name: 'Alice', active: true})")
        .expect("create");
    db.execute_cypher("CREATE (u:User {name: 'Bob', active: false})")
        .expect("create");
    db.execute_cypher("CREATE (p:Post {title: 'Hello', author: 'Alice'})")
        .expect("create");
    db.execute_cypher("CREATE (p:Post {title: 'World', author: 'Bob'})")
        .expect("create");

    // b.active = true is LOCAL to second MATCH (only references b).
    // a.title = b.author is CROSS (references both a and b).
    let rows = db
        .execute_cypher(
            "MATCH (a:Post) \
             MATCH (b:User) WHERE b.active = true AND a.author = b.name \
             RETURN a.title, b.name",
        )
        .expect("mixed predicates");

    assert_eq!(rows.len(), 1, "only active Alice's post should match");
    assert_eq!(rows[0].get("a.title"), Some(&Value::String("Hello".into())));
}

// ── No cross-scope predicate (all local) ────────────────────────────

/// WHERE only references variables from the current MATCH — should stay in branch.
#[test]
fn multi_match_where_all_local() {
    let (mut db, _dir) = open_db();

    db.execute_cypher("CREATE (u:User {name: 'Alice'})")
        .expect("create");
    db.execute_cypher("CREATE (p:Post {title: 'Hello', published: true})")
        .expect("create");
    db.execute_cypher("CREATE (p:Post {title: 'Draft', published: false})")
        .expect("create");

    // WHERE only references b (second MATCH) — should stay in branch.
    let rows = db
        .execute_cypher(
            "MATCH (a:User) MATCH (b:Post) WHERE b.published = true RETURN a.name, b.title",
        )
        .expect("all-local where");

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("b.title"), Some(&Value::String("Hello".into())));
}

// ── Single MATCH comma-separated (should already work) ──────────────

/// MATCH (a:X), (b:Y) WHERE a.x = b.y — single MATCH, comma-separated.
/// This was always correct because WHERE is applied after CartesianProduct.
#[test]
fn single_match_comma_cross_where() {
    let (mut db, _dir) = open_db();

    db.execute_cypher("CREATE (u:User {name: 'Alice'})")
        .expect("create");
    db.execute_cypher("CREATE (p:Post {title: 'Hi', author: 'Alice'})")
        .expect("create");

    let rows = db
        .execute_cypher("MATCH (a:User), (b:Post) WHERE a.name = b.author RETURN a.name, b.title")
        .expect("comma-separated cross-where");

    assert_eq!(rows.len(), 1);
}

// ── Cross-MATCH with no matching data ───────────────────────────────

/// Cross-MATCH join with no matching rows should return empty, not error.
#[test]
fn cross_match_no_matching_rows() {
    let (mut db, _dir) = open_db();

    db.execute_cypher("CREATE (u:User {name: 'Alice'})")
        .expect("create");
    db.execute_cypher("CREATE (p:Post {author: 'Charlie'})")
        .expect("create");

    let rows = db
        .execute_cypher("MATCH (a:User) MATCH (b:Post) WHERE a.name = b.author RETURN a, b")
        .expect("no match");

    assert!(rows.is_empty());
}

// ── Three MATCH clauses ─────────────────────────────────────────────

/// Three sequential MATCH clauses with cross-scope WHERE on the last one.
#[test]
fn three_match_cross_where() {
    let (mut db, _dir) = open_db();

    db.execute_cypher("CREATE (u:User {name: 'Alice'})")
        .expect("create");
    db.execute_cypher("CREATE (p:Post {title: 'Hi', author: 'Alice'})")
        .expect("create");
    db.execute_cypher("CREATE (c:Comment {text: 'Nice', post_title: 'Hi', commenter: 'Alice'})")
        .expect("create");

    let rows = db
        .execute_cypher(
            "MATCH (u:User) \
             MATCH (p:Post) WHERE p.author = u.name \
             MATCH (c:Comment) WHERE c.post_title = p.title AND c.commenter = u.name \
             RETURN u.name, p.title, c.text",
        )
        .expect("three-match cross-where");

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("c.text"), Some(&Value::String("Nice".into())));
}
