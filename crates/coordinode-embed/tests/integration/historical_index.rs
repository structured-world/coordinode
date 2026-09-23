//! Integration tests: reads at a named timestamp through vector and full-text
//! indexes.
//!
//! A vector or full-text index on a label that keeps no history holds only the
//! current state, so it cannot say what matched at an earlier timestamp. A
//! read at a named timestamp (`AS OF TIMESTAMP`, `ReadConcern.at_timestamp`)
//! that would be answered by such an index is refused. Vector search has an
//! exact alternative, `vector_consistency('exact')`, which evaluates every
//! vector of the named snapshot and needs no index.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_core::graph::types::Value;
use coordinode_core::txn::read_concern::{ReadConcern, ReadConcernLevel};
use coordinode_embed::{Database, DatabaseError};
use coordinode_query::executor::runner::{ExecutionError, HistoricalIndexKind};

fn open_db() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open db");
    (db, dir)
}

/// Run `statement` in its own transaction and return its commit timestamp.
fn commit(db: &mut Database, statement: &str) -> u64 {
    let tx = db.begin_transaction();
    db.execute_in_transaction(tx, statement, None)
        .expect("statement");
    db.commit_transaction(tx)
        .expect("commit")
        .commit_ts
        .as_raw()
}

/// Names returned by a query, in result order.
fn names(rows: &[coordinode_query::executor::row::Row]) -> Vec<String> {
    rows.iter()
        .map(|row| match row.get("name") {
            Some(Value::String(s)) => s.clone(),
            other => panic!("expected a name column, got {other:?}"),
        })
        .collect()
}

/// Builds the history every vector test reads: at timestamp `before` the two
/// nearest items to [1, 0] are `a` (exactly on it) and then `b`. After
/// `before`, `a` is deleted, `c` is created exactly on [1, 0], and `b` moves
/// far away. The current index therefore ranks `c` first and no longer holds
/// `a`, which is exactly what a read at `before` must not see.
fn vector_history(db: &mut Database) -> u64 {
    db.execute_cypher("CREATE VECTOR INDEX item_emb ON :Item(emb) OPTIONS {metric: \"l2\"}")
        .expect("create vector index");
    commit(db, "CREATE (:Item {name: 'a', emb: [1.0, 0.0]})");
    commit(db, "CREATE (:Item {name: 'b', emb: [0.8, 0.0]})");
    let before = commit(db, "CREATE (:Item {name: 'far', emb: [-5.0, 0.0]})");
    commit(db, "MATCH (n:Item {name: 'a'}) DETACH DELETE n");
    commit(db, "CREATE (:Item {name: 'c', emb: [1.0, 0.0]})");
    commit(db, "MATCH (n:Item {name: 'b'}) SET n.emb = [-9.0, 0.0]");
    before
}

/// Pure vector top-k, the shape the planner answers with `HnswScan`.
fn top2(hint: &str, as_of: Option<u64>) -> String {
    let as_of = as_of.map_or(String::new(), |ts| format!(" AS OF TIMESTAMP {ts}"));
    format!(
        "MATCH (n:Item) WITH n, vector_distance(n.emb, [1.0, 0.0]) AS d \
         ORDER BY d LIMIT 2 RETURN n.name AS name{hint}{as_of}"
    )
}

fn assert_refused(err: DatabaseError, kind: HistoricalIndexKind) {
    match err {
        DatabaseError::Execution(ExecutionError::IndexNotHistorical {
            kind: got, label, ..
        }) => {
            assert_eq!(got, kind, "refusal names the index kind");
            assert_eq!(
                label,
                if kind == HistoricalIndexKind::Vector {
                    "Item"
                } else {
                    "Article"
                }
            );
        }
        other => panic!("expected IndexNotHistorical({kind:?}), got {other:?}"),
    }
}

/// `AS OF TIMESTAMP` over a current-only vector index is refused rather than
/// answered from today's index, which no longer holds `a` and ranks `c`, a
/// node that did not exist yet, first.
#[test]
fn as_of_vector_top_k_on_a_current_only_index_is_refused() {
    let (mut db, _dir) = open_db();
    let before = vector_history(&mut db);
    let err = db
        .execute_cypher(&top2("", Some(before)))
        .expect_err("a current-only index cannot answer a past timestamp");
    assert_refused(err, HistoricalIndexKind::Vector);
}

/// `ReadConcern.at_timestamp` names a timestamp as well, so the same rule
/// holds on that path.
#[test]
fn read_concern_at_timestamp_vector_top_k_is_refused() {
    let (mut db, _dir) = open_db();
    let before = vector_history(&mut db);
    let err = db
        .execute_cypher_full(
            &top2("", None),
            None,
            None,
            Some(ReadConcern {
                level: ReadConcernLevel::Snapshot,
                at_timestamp: Some(before),
                ..ReadConcern::default()
            }),
            None,
        )
        .expect_err("a current-only index cannot answer a named timestamp");
    assert_refused(err, HistoricalIndexKind::Vector);
}

/// `vector_consistency('exact')` answers the same read from the named
/// snapshot: `a` is back, `c` does not exist yet, and `b` ranks by the
/// vector it had then.
#[test]
fn exact_answers_an_as_of_vector_top_k_from_the_named_snapshot() {
    let (mut db, _dir) = open_db();
    let before = vector_history(&mut db);
    let rows = db
        .execute_cypher(&top2(" /*+ vector_consistency('exact') */", Some(before)))
        .expect("exact evaluation at the named timestamp");
    assert_eq!(names(&rows), ["a", "b"]);
}

/// The session setting reaches the same exact path.
#[test]
fn session_exact_answers_an_as_of_vector_top_k() {
    let (mut db, _dir) = open_db();
    let before = vector_history(&mut db);
    db.execute_cypher("SET vector_consistency = 'exact'")
        .expect("set session mode");
    let rows = db
        .execute_cypher(&top2("", Some(before)))
        .expect("exact evaluation at the named timestamp");
    assert_eq!(names(&rows), ["a", "b"]);
}

/// A per-query hint wins over the session setting: `current` asked of one
/// query is honoured while the session says `exact`, and the read at a past
/// timestamp is then refused again.
#[test]
fn a_query_hint_overrides_the_session_setting() {
    let (mut db, _dir) = open_db();
    let before = vector_history(&mut db);
    db.execute_cypher("SET vector_consistency = 'exact'")
        .expect("set session mode");
    let err = db
        .execute_cypher(&top2(" /*+ vector_consistency('current') */", Some(before)))
        .expect_err("the hint selects the index path, which cannot answer the past");
    assert_refused(err, HistoricalIndexKind::Vector);
}

/// An explicit `exact` is not overridden by the index access path: the plan
/// evaluates vectors instead of reading the index, also without a timestamp.
#[test]
fn an_exact_hint_keeps_the_query_off_the_index() {
    let (mut db, _dir) = open_db();
    vector_history(&mut db);
    let query = top2(" /*+ vector_consistency('exact') */", None);
    let plan = db.explain_cypher(&query).expect("explain");
    assert!(
        !plan.contains("HnswScan"),
        "an exact query must not read the index, got:\n{plan}"
    );
    assert!(plan.contains("Vector consistency: exact"), "got:\n{plan}");
    let rows = db.execute_cypher(&query).expect("exact current read");
    assert_eq!(names(&rows), ["c", "far"]);
}

/// Without a timestamp the index keeps serving the current state.
#[test]
fn a_current_read_still_uses_the_index() {
    let (mut db, _dir) = open_db();
    vector_history(&mut db);
    let plan = db.explain_cypher(&top2("", None)).expect("explain");
    assert!(plan.contains("HnswScan"), "got:\n{plan}");
    let rows = db.execute_cypher(&top2("", None)).expect("current read");
    assert_eq!(names(&rows), ["c", "far"]);
}

/// A filtered top-k keeps the scan-then-rank path, which also consults the
/// index; the rule does not depend on which access path the planner took or
/// on how many rows the filter leaves.
#[test]
fn a_filtered_as_of_vector_top_k_is_refused_and_answered_exactly() {
    let (mut db, _dir) = open_db();
    let before = vector_history(&mut db);
    let filtered = |hint: &str| {
        format!(
            "MATCH (n:Item) WHERE n.name <> 'far' \
             WITH n, vector_distance(n.emb, [1.0, 0.0]) AS d \
             ORDER BY d LIMIT 2 RETURN n.name AS name{hint} AS OF TIMESTAMP {before}"
        )
    };
    let err = db
        .execute_cypher(&filtered(""))
        .expect_err("the index path cannot answer the past");
    assert_refused(err, HistoricalIndexKind::Vector);
    let rows = db
        .execute_cypher(&filtered(" /*+ vector_consistency('exact') */"))
        .expect("exact evaluation");
    assert_eq!(names(&rows), ["a", "b"]);
}

/// A label without a vector index is always evaluated exactly, so a past
/// timestamp is answered, not refused.
#[test]
fn as_of_vector_top_k_without_an_index_is_answered() {
    let (mut db, _dir) = open_db();
    commit(&mut db, "CREATE (:Plain {name: 'a', emb: [1.0, 0.0]})");
    let before = commit(&mut db, "CREATE (:Plain {name: 'b', emb: [0.8, 0.0]})");
    commit(&mut db, "MATCH (n:Plain {name: 'a'}) DETACH DELETE n");
    let rows = db
        .execute_cypher(&format!(
            "MATCH (n:Plain) WITH n, vector_distance(n.emb, [1.0, 0.0]) AS d \
             ORDER BY d LIMIT 2 RETURN n.name AS name AS OF TIMESTAMP {before}"
        ))
        .expect("exact evaluation needs no index");
    assert_eq!(names(&rows), ["a", "b"]);
}

/// A full-text index matches today's text, so `text_match` at a past
/// timestamp is refused; there is no exact alternative for full-text yet.
#[test]
fn as_of_text_match_on_a_current_only_index_is_refused() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE TEXT INDEX article_body ON :Article(body)")
        .expect("create text index");
    let before = commit(
        &mut db,
        "CREATE (:Article {name: 'x', body: 'rust storage'})",
    );
    commit(
        &mut db,
        "MATCH (n:Article {name: 'x'}) SET n.body = 'golang services'",
    );
    let current = db
        .execute_cypher(
            "MATCH (n:Article) WHERE text_match(n.body, 'golang') RETURN n.name AS name",
        )
        .expect("current full-text read");
    assert_eq!(names(&current), ["x"]);
    let err = db
        .execute_cypher(&format!(
            "MATCH (n:Article) WHERE text_match(n.body, 'rust') RETURN n.name AS name \
             AS OF TIMESTAMP {before}"
        ))
        .expect_err("a current-only text index cannot answer a past timestamp");
    assert_refused(err, HistoricalIndexKind::FullText);
}
