//! Integration tests: TextIndexRegistry + CREATE/DROP TEXT INDEX DDL (G013).
//!
//! Tests the full text index lifecycle through Database:
//! - CREATE TEXT INDEX DDL creates tantivy index and backfills existing nodes
//! - Auto-maintenance: CREATE node → text indexed automatically
//! - text_match() queries use the registry-managed index
//! - DROP TEXT INDEX removes the index
//! - Text index persists across Database reopen

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_core::graph::types::Value;
use coordinode_embed::Database;

fn open_db() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open db");
    (db, dir)
}

// ── CREATE TEXT INDEX DDL ──────────────────────────────────────────

/// CREATE TEXT INDEX creates an index and returns metadata.
#[test]
fn create_text_index_returns_metadata() {
    let (mut db, _dir) = open_db();

    let rows = db
        .execute_cypher("CREATE TEXT INDEX article_body ON :Article(body)")
        .expect("create text index");

    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("index"),
        Some(&Value::String("article_body".into()))
    );
    assert_eq!(rows[0].get("label"), Some(&Value::String("Article".into())));
    assert_eq!(
        rows[0].get("properties"),
        Some(&Value::String("body".into()))
    );
    assert_eq!(
        rows[0].get("default_language"),
        Some(&Value::String("english".into()))
    );
}

/// CREATE TEXT INDEX with explicit LANGUAGE clause.
#[test]
fn create_text_index_with_language() {
    let (mut db, _dir) = open_db();

    let rows = db
        .execute_cypher("CREATE TEXT INDEX article_body ON :Article(body) LANGUAGE 'russian'")
        .expect("create text index");

    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("default_language"),
        Some(&Value::String("russian".into()))
    );
}

/// CREATE TEXT INDEX backfills existing nodes.
#[test]
fn create_text_index_backfills_existing_nodes() {
    let (mut db, _dir) = open_db();

    // Create nodes BEFORE the index.
    db.execute_cypher("CREATE (a:Article {body: 'Rust graph database engine'})")
        .expect("create node 1");
    db.execute_cypher("CREATE (a:Article {body: 'Python machine learning framework'})")
        .expect("create node 2");

    // Now create the text index — should backfill 2 documents.
    let rows = db
        .execute_cypher("CREATE TEXT INDEX article_body ON :Article(body)")
        .expect("create text index");

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("documents_indexed"), Some(&Value::Int(2)));
}

// ── Auto-maintenance on write ──────────────────────────────────────

/// Nodes created AFTER index creation are automatically indexed.
#[test]
fn auto_index_on_create_node() {
    let (mut db, _dir) = open_db();

    // Create index first.
    db.execute_cypher("CREATE TEXT INDEX article_body ON :Article(body)")
        .expect("create index");

    // Create nodes — should be auto-indexed.
    db.execute_cypher("CREATE (a:Article {body: 'Rust graph database'})")
        .expect("create node 1");
    db.execute_cypher("CREATE (a:Article {body: 'TypeScript web framework'})")
        .expect("create node 2");

    // Search via text_match — should find the Rust article.
    let rows = db
        .execute_cypher("MATCH (a:Article) WHERE text_match(a.body, 'rust') RETURN a.body AS body")
        .expect("text search");

    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("body"),
        Some(&Value::String("Rust graph database".into()))
    );
}

/// text_match returns BM25 scores via text_score().
#[test]
fn text_score_returns_bm25() {
    let (mut db, _dir) = open_db();

    db.execute_cypher("CREATE TEXT INDEX article_body ON :Article(body)")
        .expect("create index");
    db.execute_cypher("CREATE (a:Article {body: 'Rust graph database engine for AI'})")
        .expect("create node");

    let rows = db
        .execute_cypher(
            "MATCH (a:Article) WHERE text_match(a.body, 'rust') \
             RETURN text_score(a.body, 'rust') AS score",
        )
        .expect("text score");

    assert_eq!(rows.len(), 1);
    let score = rows[0].get("score").and_then(|v| match v {
        Value::Float(f) => Some(*f),
        _ => None,
    });
    assert!(score.is_some(), "score should be a float");
    assert!(score.unwrap() > 0.0, "BM25 score should be positive");
}

// ── DROP TEXT INDEX ────────────────────────────────────────────────

/// DROP TEXT INDEX removes the index.
#[test]
fn drop_text_index() {
    let (mut db, _dir) = open_db();

    db.execute_cypher("CREATE TEXT INDEX article_body ON :Article(body)")
        .expect("create index");
    db.execute_cypher("CREATE (a:Article {body: 'Rust graph database'})")
        .expect("create node");

    // Verify index works.
    let rows = db
        .execute_cypher("MATCH (a:Article) WHERE text_match(a.body, 'rust') RETURN a.body")
        .expect("search before drop");
    assert_eq!(rows.len(), 1);

    // Drop the index.
    let rows = db
        .execute_cypher("DROP TEXT INDEX article_body")
        .expect("drop index");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("dropped"), Some(&Value::Bool(true)));

    // R-HYB1b: after DROP, text_match() must hard-fail with a clear message
    // rather than silently passing every row through. The old graceful-
    // degradation behaviour was a semantic bug (the opposite of what the
    // filter asked for).
    let err = db
        .execute_cypher("MATCH (a:Article) WHERE text_match(a.body, 'rust') RETURN a.body")
        .expect_err("text_match() after DROP must error, not pass rows through");
    let msg = err.to_string();
    assert!(
        msg.contains("text_match()") && msg.contains("CREATE TEXT INDEX"),
        "error must name text_match() and point at the remedy, got: {msg}"
    );
}

// ── Persistence across reopen ──────────────────────────────────────

/// Text index definition persists across Database reopen.
#[test]
fn text_index_persists_across_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");

    // Session 1: create index + node.
    {
        let mut db = Database::open(dir.path()).expect("open 1");
        db.execute_cypher("CREATE TEXT INDEX article_body ON :Article(body)")
            .expect("create index");
        db.execute_cypher("CREATE (a:Article {body: 'Rust graph database engine'})")
            .expect("create node");

        // Verify search works in session 1.
        let rows = db
            .execute_cypher("MATCH (a:Article) WHERE text_match(a.body, 'rust') RETURN a.body")
            .expect("search session 1");
        assert_eq!(rows.len(), 1);
    }

    // Session 2: reopen and search — index should be rebuilt from stored data.
    {
        let mut db = Database::open(dir.path()).expect("open 2");
        let rows = db
            .execute_cypher(
                "MATCH (a:Article) WHERE text_match(a.body, 'rust') RETURN a.body AS body",
            )
            .expect("search session 2");

        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0].get("body"),
            Some(&Value::String("Rust graph database engine".into()))
        );
    }
}

// ── SET updates index ──────────────────────────────────────────────

/// SET on a text property updates the index — old text no longer matches,
/// new text becomes searchable.
#[test]
fn set_updates_text_index() {
    let (mut db, _dir) = open_db();

    db.execute_cypher("CREATE TEXT INDEX article_body ON :Article(body)")
        .expect("create index");
    db.execute_cypher("CREATE (a:Article {body: 'Rust graph database'})")
        .expect("create node");

    // Verify initial search works.
    let rows = db
        .execute_cypher("MATCH (a:Article) WHERE text_match(a.body, 'rust') RETURN a.body AS body")
        .expect("search before SET");
    assert_eq!(rows.len(), 1);

    // Update the body property.
    db.execute_cypher("MATCH (a:Article) SET a.body = 'Python web framework'")
        .expect("SET body");

    // Old text should no longer match.
    let rows = db
        .execute_cypher("MATCH (a:Article) WHERE text_match(a.body, 'rust') RETURN a.body")
        .expect("search old text after SET");
    assert_eq!(rows.len(), 0, "old text should not match after SET");

    // New text should match.
    let rows = db
        .execute_cypher(
            "MATCH (a:Article) WHERE text_match(a.body, 'python') RETURN a.body AS body",
        )
        .expect("search new text after SET");
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("body"),
        Some(&Value::String("Python web framework".into()))
    );
}

// ── DELETE removes from index ──────────────────────────────────────

/// DETACH DELETE removes the node from the text index.
#[test]
fn delete_node_removes_from_text_index() {
    let (mut db, _dir) = open_db();

    db.execute_cypher("CREATE TEXT INDEX article_body ON :Article(body)")
        .expect("create index");
    db.execute_cypher("CREATE (a:Article {body: 'Rust graph database'})")
        .expect("create node");

    // Verify search finds it.
    let rows = db
        .execute_cypher("MATCH (a:Article) WHERE text_match(a.body, 'rust') RETURN a.body")
        .expect("search before delete");
    assert_eq!(rows.len(), 1);

    // Delete the node.
    db.execute_cypher("MATCH (a:Article) DETACH DELETE a")
        .expect("delete node");

    // Text index should no longer find it.
    let rows = db
        .execute_cypher("MATCH (a:Article) WHERE text_match(a.body, 'rust') RETURN a.body")
        .expect("search after delete");
    assert_eq!(
        rows.len(),
        0,
        "deleted node should not appear in text search"
    );
}

// ── Duplicate CREATE errors ────────────────────────────────────────

/// Creating a duplicate text index on the same (label, property) returns an error.
#[test]
fn duplicate_create_text_index_errors() {
    let (mut db, _dir) = open_db();

    db.execute_cypher("CREATE TEXT INDEX article_body ON :Article(body)")
        .expect("first create should succeed");

    let result = db.execute_cypher("CREATE TEXT INDEX article_body2 ON :Article(body)");
    assert!(
        result.is_err(),
        "duplicate text index on same (label, property) should fail"
    );
}

// ── REMOVE property removes from index ─────────────────────────────

/// REMOVE a.body removes the property from the text index.
#[test]
fn remove_property_removes_from_text_index() {
    let (mut db, _dir) = open_db();

    db.execute_cypher("CREATE TEXT INDEX article_body ON :Article(body)")
        .expect("create index");
    db.execute_cypher("CREATE (a:Article {body: 'Rust graph database', title: 'Intro'})")
        .expect("create node");

    // Verify search finds it.
    let rows = db
        .execute_cypher("MATCH (a:Article) WHERE text_match(a.body, 'rust') RETURN a.body")
        .expect("search before remove");
    assert_eq!(rows.len(), 1);

    // REMOVE the body property (node still exists, just without body).
    db.execute_cypher("MATCH (a:Article) REMOVE a.body")
        .expect("remove body");

    // Text index should no longer find it.
    let rows = db
        .execute_cypher("MATCH (a:Article) WHERE text_match(a.body, 'rust') RETURN a.body")
        .expect("search after remove");
    assert_eq!(
        rows.len(),
        0,
        "removed property should not appear in text search"
    );
}

// ── Non-matching label not indexed ─────────────────────────────────

// ── Multi-field DDL (G016) ─────────────────────────────────────────

/// CREATE TEXT INDEX with multi-field per-analyzer syntax.
#[test]
fn create_text_index_multi_field_syntax() {
    let (mut db, _dir) = open_db();

    let rows = db
        .execute_cypher(
            r#"CREATE TEXT INDEX article_text ON :Article {
                title: { analyzer: "english" },
                body:  { analyzer: "auto_detect" }
            } DEFAULT LANGUAGE "english""#,
        )
        .expect("create multi-field text index");

    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("index"),
        Some(&Value::String("article_text".into()))
    );
    assert_eq!(
        rows[0].get("properties"),
        Some(&Value::String("title, body".into()))
    );
    assert_eq!(
        rows[0].get("default_language"),
        Some(&Value::String("english".into()))
    );
}

/// Diagnostic: verify registry state after multi-field CREATE INDEX.
#[test]
fn multi_field_registry_state() {
    let (mut db, _dir) = open_db();

    db.execute_cypher(
        r#"CREATE TEXT INDEX article_text ON :Article {
            title: { analyzer: "english" },
            body:  { analyzer: "english" }
        }"#,
    )
    .expect("create multi-field index");

    let reg = db.text_index_registry();
    assert!(
        reg.has_index("Article", "title"),
        "registry should have index for title"
    );
    assert!(
        reg.has_index("Article", "body"),
        "registry should have index for body"
    );

    // Write directly to the body index and search.
    reg.on_text_written(
        "Article",
        coordinode_core::graph::node::NodeId::from_raw(42),
        "body",
        "Machine learning framework",
    );
    let results = reg.search("Article", "body", "machine", 10);
    assert!(
        results.is_some(),
        "search should return Some for registered index"
    );
    assert_eq!(
        results.unwrap().len(),
        1,
        "should find the directly-written document"
    );
}

/// Multi-field index: search works on BOTH fields, not just the first.
/// This test verifies that the registry registers the tantivy handle under
/// all indexed properties, so text_match resolves for any field.
#[test]
fn multi_field_search_on_second_field() {
    let (mut db, _dir) = open_db();

    // Create multi-field index first.
    db.execute_cypher(
        r#"CREATE TEXT INDEX article_text ON :Article {
            title: { analyzer: "english" },
            body:  { analyzer: "english" }
        }"#,
    )
    .expect("create multi-field index");

    // Create node with distinct words in title vs body.
    db.execute_cypher(
        "CREATE (a:Article {title: 'Rust Database', body: 'Machine learning framework'})",
    )
    .expect("create node");

    // Search via title field — should find it.
    let rows = db
        .execute_cypher("MATCH (a:Article) WHERE text_match(a.title, 'rust') RETURN a.title AS t")
        .expect("search title");
    assert_eq!(rows.len(), 1, "title search should find the article");

    // Search via body field — MUST also find it (second field in multi-field index).
    let rows = db
        .execute_cypher("MATCH (a:Article) WHERE text_match(a.body, 'machine') RETURN a.body AS b")
        .expect("search body");
    assert_eq!(
        rows.len(),
        1,
        "body search should find the article (second field in multi-field index)"
    );
}

/// Multi-field index backfills existing nodes across all indexed properties.
#[test]
fn multi_field_index_backfills() {
    let (mut db, _dir) = open_db();

    // Create nodes with both title and body.
    db.execute_cypher(
        "CREATE (a:Article {title: 'Rust Database', body: 'A graph engine in Rust'})",
    )
    .expect("create node 1");
    db.execute_cypher(
        "CREATE (a:Article {title: 'Python ML', body: 'Machine learning framework'})",
    )
    .expect("create node 2");

    // Create multi-field index — should backfill both fields.
    let rows = db
        .execute_cypher(
            r#"CREATE TEXT INDEX article_text ON :Article {
                title: { analyzer: "english" },
                body:  { analyzer: "english" }
            }"#,
        )
        .expect("create multi-field index");

    assert_eq!(rows[0].get("documents_indexed"), Some(&Value::Int(2)));

    // Verify backfilled data is searchable on the SECOND field.
    let rows = db
        .execute_cypher("MATCH (a:Article) WHERE text_match(a.body, 'machine') RETURN a.body AS b")
        .expect("search body after backfill");
    assert_eq!(rows.len(), 1, "backfilled body text should be searchable");
}

/// Multi-field index with DEFAULT LANGUAGE and LANGUAGE OVERRIDE.
#[test]
fn multi_field_with_language_override() {
    let (mut db, _dir) = open_db();

    let rows = db
        .execute_cypher(
            r#"CREATE TEXT INDEX idx ON :Post {
                content: { analyzer: "auto_detect" }
            } DEFAULT LANGUAGE "german" LANGUAGE OVERRIDE "lang""#,
        )
        .expect("create index with override");

    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("default_language"),
        Some(&Value::String("german".into()))
    );
}

// ── Non-matching label ────────────────────────────────────────────

/// Nodes with a different label are NOT indexed by a label-specific text index.
#[test]
fn non_matching_label_not_indexed() {
    let (mut db, _dir) = open_db();

    db.execute_cypher("CREATE TEXT INDEX article_body ON :Article(body)")
        .expect("create index");

    // Create a User node (not Article) — should NOT be indexed.
    db.execute_cypher("CREATE (u:User {body: 'Rust expert developer'})")
        .expect("create user");
    // Create an Article node — SHOULD be indexed.
    db.execute_cypher("CREATE (a:Article {body: 'Rust graph database'})")
        .expect("create article");

    // Search should find only the Article, not the User.
    let rows = db
        .execute_cypher("MATCH (a:Article) WHERE text_match(a.body, 'rust') RETURN a.body AS body")
        .expect("search articles");
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("body"),
        Some(&Value::String("Rust graph database".into()))
    );
}
