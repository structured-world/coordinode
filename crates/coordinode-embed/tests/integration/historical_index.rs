//! Integration tests: reads at a named timestamp through vector and full-text
//! indexes.
//!
//! A vector or full-text index holds the current state. A read at a named
//! timestamp (`AS OF TIMESTAMP`, `ReadConcern.at_timestamp`) is answered from
//! it for every node unchanged since that timestamp, and the nodes written
//! after it are read from the store at the timestamp and evaluated exactly, so
//! the answer is the one the named snapshot gives, the same as the exact path
//! (`vector_consistency('exact')`) that evaluates every vector.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_core::graph::types::Value;
use coordinode_core::txn::read_concern::{ReadConcern, ReadConcernLevel};
use coordinode_embed::Database;

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

/// `AS OF TIMESTAMP` through the index answers as the named snapshot: `a` is
/// back, `c` does not exist yet, and `b` ranks by the vector it had then.
#[test]
fn as_of_vector_top_k_through_the_index_answers_the_snapshot() {
    let (mut db, _dir) = open_db();
    let before = vector_history(&mut db);
    let query = top2("", Some(before));
    let plan = db.explain_cypher(&query).expect("explain");
    assert!(
        plan.contains("HnswScan"),
        "the index path is taken:\n{plan}"
    );
    let rows = db
        .execute_cypher(&query)
        .expect("read at the named timestamp");
    assert_eq!(names(&rows), ["a", "b"]);
}

/// `ReadConcern.at_timestamp` names a timestamp as well, and is answered the
/// same way.
#[test]
fn read_concern_at_timestamp_vector_top_k_answers_the_snapshot() {
    let (mut db, _dir) = open_db();
    let before = vector_history(&mut db);
    let rows = db
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
        .expect("read at the named timestamp");
    assert_eq!(names(&rows.rows), ["a", "b"]);
}

/// `vector_consistency('exact')` reads the named snapshot without the index
/// and agrees with the index path.
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
/// query takes the index path while the session says `exact`, and answers
/// the past timestamp all the same.
#[test]
fn a_query_hint_overrides_the_session_setting() {
    let (mut db, _dir) = open_db();
    let before = vector_history(&mut db);
    db.execute_cypher("SET vector_consistency = 'exact'")
        .expect("set session mode");
    let query = top2(" /*+ vector_consistency('current') */", Some(before));
    let plan = db.explain_cypher(&query).expect("explain");
    assert!(
        plan.contains("HnswScan"),
        "the hint selects the index:\n{plan}"
    );
    let rows = db.execute_cypher(&query).expect("index read at the past");
    assert_eq!(names(&rows), ["a", "b"]);
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
/// index; it answers the past timestamp as the exact path does.
#[test]
fn a_filtered_as_of_vector_top_k_answers_the_snapshot() {
    let (mut db, _dir) = open_db();
    let before = vector_history(&mut db);
    let filtered = |hint: &str| {
        format!(
            "MATCH (n:Item) WHERE n.name <> 'far' \
             WITH n, vector_distance(n.emb, [1.0, 0.0]) AS d \
             ORDER BY d LIMIT 2 RETURN n.name AS name{hint} AS OF TIMESTAMP {before}"
        )
    };
    let rows = db
        .execute_cypher(&filtered(""))
        .expect("index-path evaluation");
    assert_eq!(names(&rows), ["a", "b"]);
    let rows = db
        .execute_cypher(&filtered(" /*+ vector_consistency('exact') */"))
        .expect("exact evaluation");
    assert_eq!(names(&rows), ["a", "b"]);
}

/// A label without a vector index is always evaluated exactly, so a past
/// timestamp is answered.
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

/// `text_match` at a past timestamp matches the text each node held then: a
/// node rewritten since is found by its old words and not its new ones, a
/// node deleted since is found, one created since is not; scores rank as the
/// snapshot's corpus does.
#[test]
fn as_of_text_match_answers_the_snapshot() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE TEXT INDEX article_body ON :Article(body)")
        .expect("create text index");
    commit(
        &mut db,
        "CREATE (:Article {name: 'x', body: 'rust storage'})",
    );
    let before = commit(
        &mut db,
        "CREATE (:Article {name: 'y', body: 'rust engines'})",
    );
    commit(
        &mut db,
        "MATCH (n:Article {name: 'x'}) SET n.body = 'golang services'",
    );
    commit(&mut db, "MATCH (n:Article {name: 'y'}) DETACH DELETE n");
    commit(
        &mut db,
        "CREATE (:Article {name: 'z', body: 'rust newcomer'})",
    );

    let at = |db: &mut Database, words: &str, as_of: Option<u64>| -> Vec<String> {
        let as_of = as_of.map_or(String::new(), |ts| format!(" AS OF TIMESTAMP {ts}"));
        let mut found = names(
            &db.execute_cypher(&format!(
                "MATCH (n:Article) WHERE text_match(n.body, '{words}') \
                 RETURN n.name AS name{as_of}"
            ))
            .expect("full-text read"),
        );
        found.sort();
        found
    };
    assert_eq!(at(&mut db, "rust", None), ["z"]);
    assert_eq!(at(&mut db, "golang", None), ["x"]);
    assert_eq!(at(&mut db, "rust", Some(before)), ["x", "y"]);
    assert!(at(&mut db, "golang", Some(before)).is_empty());
    assert!(at(&mut db, "newcomer", Some(before)).is_empty());
}

/// The same snapshot read over a temporal label, whose writes land as new
/// versions of a node rather than in place: a node changed or deleted after
/// the timestamp is answered with its state then.
#[test]
fn as_of_text_match_answers_the_snapshot_of_a_temporal_label() {
    let (mut db, _dir) = open_db();
    db.execute_cypher(
        "CREATE NODE TYPE Emp TEMPORAL WITH (name: STRING, body: STRING, valid_from: INT, valid_to: INT)",
    )
    .expect("temporal label");
    db.execute_cypher("CREATE TEXT INDEX emp_body ON :Emp(body)")
        .expect("create text index");
    let since = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock after epoch")
        .as_micros() as i64
        - 1_000_000;
    commit(
        &mut db,
        &format!("CREATE (:Emp {{name: 'x', body: 'rust storage', valid_from: {since}}})"),
    );
    let before = commit(
        &mut db,
        &format!("CREATE (:Emp {{name: 'y', body: 'rust engines', valid_from: {since}}})"),
    );
    commit(
        &mut db,
        "MATCH (n:Emp {name: 'x'}) SET n.body = 'golang services'",
    );
    commit(&mut db, "MATCH (n:Emp {name: 'y'}) DELETE n");
    commit(
        &mut db,
        &format!("CREATE (:Emp {{name: 'z', body: 'rust newcomer', valid_from: {since}}})"),
    );

    let at = |db: &mut Database, words: &str, as_of: Option<u64>| -> Vec<String> {
        let as_of = as_of.map_or(String::new(), |ts| format!(" AS OF TIMESTAMP {ts}"));
        let mut found = names(
            &db.execute_cypher(&format!(
                "MATCH (n:Emp) WHERE text_match(n.body, '{words}') \
                 RETURN n.name AS name{as_of}"
            ))
            .expect("full-text read"),
        );
        found.sort();
        found
    };
    assert_eq!(at(&mut db, "rust", None), ["z"]);
    assert_eq!(at(&mut db, "golang", None), ["x"]);
    assert_eq!(at(&mut db, "rust", Some(before)), ["x", "y"]);
    assert!(at(&mut db, "golang", Some(before)).is_empty());
    assert!(at(&mut db, "newcomer", Some(before)).is_empty());
}

/// Each document's BM25 score for `word`, by name, as `db` answers
/// `text_match`, at `as_of` when given.
fn text_scores(
    db: &mut Database,
    word: &str,
    as_of: Option<u64>,
) -> std::collections::BTreeMap<String, f64> {
    let as_of = as_of.map_or(String::new(), |ts| format!(" AS OF TIMESTAMP {ts}"));
    db.execute_cypher(&format!(
        "MATCH (n:Doc) WHERE text_match(n.body, '{word}') \
         RETURN n.name AS name, text_score(n.body, '{word}') AS score{as_of}"
    ))
    .expect("full-text read")
    .iter()
    .map(|row| {
        let name = match row.get("name") {
            Some(Value::String(s)) => s.clone(),
            other => panic!("expected a name, got {other:?}"),
        };
        let score = match row.get("score") {
            Some(Value::Float(f)) => *f,
            other => panic!("expected a score, got {other:?}"),
        };
        (name, score)
    })
    .collect()
}

/// A read at a past timestamp answers as a fresh index built from that
/// snapshot alone, scores included. Random churn (create, rewrite, delete)
/// over a small vocabulary; then, at timestamps taken along the way, every
/// word's matches and BM25 scores are compared with a second database that
/// holds only the documents of that snapshot, indexed from nothing and read
/// at its present. The model never reads history, so it checks the history
/// path rather than repeating it: membership, the corpus counts and lengths
/// that BM25 scores against, and the documents rewritten or deleted since.
#[test]
fn as_of_text_ranking_matches_an_index_built_from_the_snapshot() {
    use std::collections::BTreeMap;

    const WORDS: [&str; 6] = ["rust", "graph", "engine", "vector", "index", "storage"];
    // xorshift64: a fixed sequence, so a failure reproduces.
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move |bound: usize| -> usize {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % bound as u64) as usize
    };

    let (mut db, dir) = open_db();
    db.execute_cypher("CREATE TEXT INDEX doc_body ON :Doc(body)")
        .expect("create text index");
    let mut live: BTreeMap<String, String> = BTreeMap::new();
    let mut snapshots: Vec<(u64, BTreeMap<String, String>)> = Vec::new();
    let mut created = 0usize;
    for step in 0..60 {
        let body: Vec<&str> = (0..1 + next(6)).map(|_| WORDS[next(WORDS.len())]).collect();
        let body = body.join(" ");
        let choice = next(3);
        let ts = if live.is_empty() || choice == 0 {
            let name = format!("d{created}");
            created += 1;
            let ts = commit(
                &mut db,
                &format!("CREATE (:Doc {{name: '{name}', body: '{body}'}})"),
            );
            live.insert(name, body);
            ts
        } else {
            let name = live
                .keys()
                .nth(next(live.len()))
                .expect("a live doc")
                .clone();
            if choice == 1 {
                let ts = commit(
                    &mut db,
                    &format!("MATCH (n:Doc {{name: '{name}'}}) SET n.body = '{body}'"),
                );
                live.insert(name, body);
                ts
            } else {
                let ts = commit(
                    &mut db,
                    &format!("MATCH (n:Doc {{name: '{name}'}}) DETACH DELETE n"),
                );
                live.remove(&name);
                ts
            }
        };
        if step % 12 == 11 {
            snapshots.push((ts, live.clone()));
        }
    }

    let compare = |db: &mut Database| {
        for (ts, docs) in &snapshots {
            let (mut model, model_dir) = open_db();
            model
                .execute_cypher("CREATE TEXT INDEX doc_body ON :Doc(body)")
                .expect("create text index");
            for (name, body) in docs {
                commit(
                    &mut model,
                    &format!("CREATE (:Doc {{name: '{name}', body: '{body}'}})"),
                );
            }
            for word in WORDS {
                let past = text_scores(db, word, Some(*ts));
                let expected = text_scores(&mut model, word, None);
                assert_eq!(
                    past.keys().collect::<Vec<_>>(),
                    expected.keys().collect::<Vec<_>>(),
                    "matches of '{word}' at {ts}"
                );
                for (name, score) in &past {
                    let want = expected[name];
                    assert!(
                        (score - want).abs() <= 1e-4 * want.abs().max(1.0),
                        "score of {name} for '{word}' at {ts}: {score} against {want}"
                    );
                }
            }
            // The database first, so its index worker stops before its
            // directory goes away.
            drop(model);
            drop(model_dir);
        }
    };
    compare(&mut db);

    // Reopened, the index is rebuilt from the store: the history still
    // answers each past snapshot exactly.
    drop(db);
    let mut db = Database::open(dir.path()).expect("reopen");
    compare(&mut db);
}
