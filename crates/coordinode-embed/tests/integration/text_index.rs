//! Integration tests: TextIndexRegistry + CREATE/DROP TEXT INDEX DDL.
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

/// Write one more version of temporal node `id` (label `Emp`) straight into
/// storage, as a restore or a correction would.
fn seed_version(
    db: &Database,
    id: u64,
    props: &[(&str, &str)],
    valid_from: i64,
    valid_to: Option<i64>,
) {
    use coordinode_core::graph::node::{NodeId, NodeRecord};
    use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
    use coordinode_core::txn::write_concern::WriteConcern;
    use coordinode_modality::{LocalNodeStore, NodeStore as _};
    use coordinode_storage::engine::transaction::{CommitContext, Transaction};

    let interner = db.interner().expect("interner");
    let field = |name: &str| interner.lookup(name).expect("declared field");
    let mut record = NodeRecord::new("Emp");
    for (name, value) in props {
        record.set(field(name), Value::String((*value).into()));
    }
    record.set(field("valid_from"), Value::Int(valid_from));
    if let Some(to) = valid_to {
        record.set(field("valid_to"), Value::Int(to));
    }
    // Committed below the database's own clock so every later query
    // snapshot sees it; the version keys are new, nothing is shadowed.
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let mut txn = Transaction::begin(db.engine(), Some(&oracle), oracle.next());
    LocalNodeStore
        .put_temporal(&mut txn, 1, NodeId::from_raw(id), valid_from, &record)
        .expect("put version");
    let wc = WriteConcern::majority();
    txn.commit(&CommitContext {
        write_concern: &wc,
        pipeline: None,
        id_gen: None,
        drain_buffer: None,
        nvme_write_buffer: None,
    })
    .expect("commit version");
}

/// Index changes for nodes without a timeline: a text to hold, or `None` to
/// take the node out.
fn planted(texts: &[(u64, Option<&str>)]) -> Vec<coordinode_query::index::text_registry::NodeText> {
    texts
        .iter()
        .map(
            |(id, text)| coordinode_query::index::text_registry::NodeText {
                node_id: coordinode_core::graph::node::NodeId::from_raw(*id),
                text: text.map(str::to_string),
                validity: coordinode_search::tantivy::validity::Validity::ALWAYS,
            },
        )
        .collect()
}

// ── Access path ─────────────────────────────────────────────────────

/// `text_match` over one label reads its matches from the index instead of
/// scanning the label: the plan names `TextIndexScan`, and the rows, columns
/// and scores are those of the scan-and-filter plan, which an inline property
/// filter keeps. The transaction's own uncommitted text is part of the
/// answer, as on the filter path.
#[test]
fn text_match_over_one_label_reads_through_the_index() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE TEXT INDEX article_body ON :Article(body)")
        .expect("create text index");
    for (name, body) in [
        ("a", "rust graph engine"),
        ("b", "golang services"),
        ("c", "rust rust storage"),
    ] {
        db.execute_cypher(&format!(
            "CREATE (:Article {{name: '{name}', kind: 'post', body: '{body}'}})"
        ))
        .expect("create article");
    }

    let indexed = "MATCH (n:Article) WHERE text_match(n.body, 'rust') \
                   RETURN n.name AS name, text_score(n.body, 'rust') AS score \
                   ORDER BY name";
    let scanned = "MATCH (n:Article {kind: 'post'}) WHERE text_match(n.body, 'rust') \
                   RETURN n.name AS name, text_score(n.body, 'rust') AS score \
                   ORDER BY name";
    let plan = db.explain_cypher(indexed).expect("explain");
    assert!(plan.contains("TextIndexScan"), "the index path:\n{plan}");
    assert!(!plan.contains("NodeScan"), "no label scan:\n{plan}");
    let plan = db.explain_cypher(scanned).expect("explain");
    assert!(plan.contains("TextFilter"), "the filter path:\n{plan}");

    let through_index = db.execute_cypher(indexed).expect("index path");
    assert_eq!(
        through_index,
        db.execute_cypher(scanned).expect("filter path")
    );
    let names: Vec<_> = through_index
        .iter()
        .map(|r| r.get("name").cloned())
        .collect();
    assert_eq!(
        names,
        [
            Some(Value::String("a".into())),
            Some(Value::String("c".into()))
        ]
    );

    // An uncommitted article of the reading transaction is found.
    let tx = db.begin_transaction();
    db.execute_in_transaction(
        tx,
        "CREATE (:Article {name: 'd', kind: 'post', body: 'rust draft'})",
        None,
    )
    .expect("own write");
    let own = db
        .execute_in_transaction(tx, indexed, None)
        .expect("index path in the transaction");
    assert_eq!(own.len(), 3, "{own:?}");
    db.rollback_transaction(tx).expect("rollback");
}

// ── Transactional maintenance ──────────────────────────────────────

/// A text change in a transaction that rolls back is never searchable: the
/// node keeps its committed text, and a search for the discarded words
/// finds nothing, while a search for the committed words still finds it.
#[test]
fn a_rolled_back_text_change_is_not_searchable() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE TEXT INDEX article_body ON :Article(body)")
        .expect("create text index");
    db.execute_cypher("CREATE (:Article {title: 'a', body: 'committed words'})")
        .expect("create");

    let tx = db.begin_transaction();
    db.execute_in_transaction(
        tx,
        "MATCH (n:Article {title: 'a'}) SET n.body = 'discarded secret'",
        None,
    )
    .expect("set in transaction");
    db.rollback_transaction(tx).expect("rollback");

    let found = |db: &mut Database, words: &str| {
        db.execute_cypher(&format!(
            "MATCH (n:Article) WHERE text_match(n.body, '{words}') RETURN n.title AS t"
        ))
        .expect("search")
        .len()
    };
    assert_eq!(found(&mut db, "secret"), 0, "the rolled-back text leaked");
    assert_eq!(found(&mut db, "committed"), 1, "the committed text is gone");
}

/// A node whose indexed property stops being text leaves the index: its old
/// words no longer find it. The index follows the committed record, so a
/// value of another type takes the node out as a removal would.
#[test]
fn a_property_set_to_a_non_text_value_leaves_the_index() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE TEXT INDEX article_body ON :Article(body)")
        .expect("create text index");
    db.execute_cypher("CREATE (:Article {title: 'a', body: 'vanishing words'})")
        .expect("create");
    db.execute_cypher("MATCH (n:Article {title: 'a'}) SET n.body = 42")
        .expect("set a number");

    let rows = db
        .execute_cypher(
            "MATCH (n:Article) WHERE text_match(n.body, 'vanishing') RETURN n.title AS t",
        )
        .expect("search");
    assert!(
        rows.is_empty(),
        "the old text still finds the node: {rows:?}"
    );
}

/// A node that loses the indexed label leaves that label's index, and one
/// that gains it is added, as the committed record says.
#[test]
fn a_relabelled_node_follows_its_label_in_the_index() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE TEXT INDEX article_body ON :Article(body)")
        .expect("create text index");
    db.execute_cypher("CREATE (:Article {title: 'a', body: 'wandering words'})")
        .expect("create");
    db.execute_cypher("MATCH (n:Article {title: 'a'}) REMOVE n:Article SET n:Note")
        .expect("relabel");

    let rows = db
        .execute_cypher(
            "MATCH (n:Article) WHERE text_match(n.body, 'wandering') RETURN n.title AS t",
        )
        .expect("search");
    assert!(rows.is_empty(), "a node without the label is still indexed");

    db.execute_cypher("MATCH (n:Note {title: 'a'}) REMOVE n:Note SET n:Article")
        .expect("relabel back");
    let rows = db
        .execute_cypher(
            "MATCH (n:Article) WHERE text_match(n.body, 'wandering') RETURN n.title AS t",
        )
        .expect("search");
    assert_eq!(rows.len(), 1, "a node that gained the label is not indexed");
}

/// A transaction searches its own uncommitted text: its SET is found by the
/// new words and no longer by the old ones, and a node it created is found;
/// a search outside it sees none of that, and after rollback nothing stays.
#[test]
fn a_transaction_searches_its_own_uncommitted_text() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE TEXT INDEX article_body ON :Article(body)")
        .expect("create text index");
    db.execute_cypher("CREATE (:Article {title: 'a', body: 'committed words'})")
        .expect("create");

    let titles_in = |db: &mut Database, tx, words: &str| -> Vec<String> {
        let mut found: Vec<String> = db
            .execute_in_transaction(
                tx,
                &format!(
                    "MATCH (n:Article) WHERE text_match(n.body, '{words}') RETURN n.title AS t"
                ),
                None,
            )
            .expect("search in transaction")
            .into_iter()
            .filter_map(|row| match row.get("t") {
                Some(Value::String(t)) => Some(t.clone()),
                _ => None,
            })
            .collect();
        found.sort();
        found
    };

    let tx = db.begin_transaction();
    db.execute_in_transaction(
        tx,
        "MATCH (n:Article {title: 'a'}) SET n.body = 'private words'",
        None,
    )
    .expect("set");
    db.execute_in_transaction(
        tx,
        "CREATE (:Article {title: 'b', body: 'private draft'})",
        None,
    )
    .expect("create in transaction");

    assert_eq!(titles_in(&mut db, tx, "private"), ["a", "b"]);
    assert!(titles_in(&mut db, tx, "committed").is_empty());
    assert!(
        titles(&mut db, "private").is_empty(),
        "uncommitted text leaked"
    );
    assert_eq!(titles(&mut db, "committed"), ["a"]);

    db.rollback_transaction(tx).expect("rollback");
    assert!(titles(&mut db, "private").is_empty());
    assert_eq!(titles(&mut db, "committed"), ["a"]);
}

// ── Writes the index has not folded ────────────────────────────────

fn node_id(db: &mut Database, title: &str) -> u64 {
    let rows = db
        .execute_cypher(&format!(
            "MATCH (n:Article {{title: '{title}'}}) RETURN id(n) AS id"
        ))
        .expect("id");
    match rows[0].get("id") {
        Some(Value::Int(id)) => *id as u64,
        other => panic!("no id for {title}: {other:?}"),
    }
}

fn titles(db: &mut Database, words: &str) -> Vec<String> {
    let mut found: Vec<String> = db
        .execute_cypher(&format!(
            "MATCH (n:Article) WHERE text_match(n.body, '{words}') RETURN n.title AS t"
        ))
        .expect("search")
        .into_iter()
        .filter_map(|row| match row.get("t") {
            Some(Value::String(t)) => Some(t.clone()),
            _ => None,
        })
        .collect();
    found.sort();
    found
}

/// Hold the text indexes' coverage on a subscription whose events are never
/// released, as a worker that has folded nothing since this call would; the
/// worker's own folds go on, and are waited out so they cannot overwrite what
/// the test plants in the index afterwards.
struct HeldCoverage {
    _held: coordinode_storage::engine::applied::AppliedSubscription,
    worker: std::sync::Arc<coordinode_query::index::IndexCoverage>,
}

impl HeldCoverage {
    fn hold(db: &Database, capacity: usize) -> Self {
        let registry = db.text_index_registry();
        let worker = registry.coverage().expect("the worker's coverage");
        let held = db.engine().subscribe_applied_retained(
            coordinode_storage::engine::partition::Partition::Node,
            capacity,
        );
        registry.set_coverage(std::sync::Arc::new(
            coordinode_query::index::IndexCoverage::new(held.position()),
        ));
        Self {
            _held: held,
            worker,
        }
    }

    fn await_worker(&self) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        while !self.worker.delta(1).is_empty() {
            assert!(
                std::time::Instant::now() < deadline,
                "the worker never folded"
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }
}

/// A search never waits for the text worker: the nodes written by commits it
/// has not folded are read from the store and searched in place of the
/// index's own documents of them. The index here holds stale text for each
/// of them, as a lagging worker leaves it: a rewritten node, a deleted one,
/// and none for a new one.
#[test]
fn a_search_answers_unfolded_writes_from_the_store() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE TEXT INDEX article_body ON :Article(body)")
        .expect("create text index");
    db.execute_cypher("CREATE (:Article {title: 'a', body: 'stale rust words'})")
        .expect("create a");
    db.execute_cypher("CREATE (:Article {title: 'b', body: 'doomed rust words'})")
        .expect("create b");
    db.execute_cypher("CREATE (:Article {title: 'u', body: 'untouched rust words'})")
        .expect("create u");
    let (a, b) = (node_id(&mut db, "a"), node_id(&mut db, "b"));

    let held = HeldCoverage::hold(&db, 1024);
    db.execute_cypher("MATCH (n:Article {title: 'a'}) SET n.body = 'fresh golang words'")
        .expect("rewrite a");
    db.execute_cypher("MATCH (n:Article {title: 'b'}) DETACH DELETE n")
        .expect("delete b");
    db.execute_cypher("CREATE (:Article {title: 'c', body: 'brand new rust'})")
        .expect("create c");
    held.await_worker();
    db.text_index_registry()
        .apply_changes(
            "Article",
            "body",
            &planted(&[
                (a, Some("stale rust words")),
                (b, Some("doomed rust words")),
            ]),
        )
        .expect("plant stale text");

    assert_eq!(titles(&mut db, "rust"), ["c", "u"]);
    assert_eq!(titles(&mut db, "golang"), ["a"]);
    assert!(titles(&mut db, "stale").is_empty());
    assert!(titles(&mut db, "doomed").is_empty());
}

/// A fuzzy search and a search with highlighted snippets answer unfolded
/// writes as the plain one does: the edit-1 neighbourhood and the snippet
/// come from the text the store holds now, never from the stale document
/// the index still has.
#[test]
fn fuzzy_and_snippet_searches_answer_unfolded_writes_from_the_store() {
    use coordinode_search::tantivy::multi_lang::TextRequest;
    use coordinode_search::tantivy::pending::Matches;

    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE TEXT INDEX article_body ON :Article(body)")
        .expect("create text index");
    db.execute_cypher("CREATE (:Article {title: 'a', body: 'stale rust words'})")
        .expect("create a");
    let a = node_id(&mut db, "a");

    let held = HeldCoverage::hold(&db, 1024);
    db.execute_cypher("MATCH (n:Article {title: 'a'}) SET n.body = 'fresh golang words'")
        .expect("rewrite a");
    held.await_worker();
    db.text_index_registry()
        .apply_changes(
            "Article",
            "body",
            &planted(&[(a, Some("stale rust words"))]),
        )
        .expect("plant stale text");

    let search = |request| {
        db.text_search("Article", "body", request, Matches::Top(10))
            .expect("search")
            .expect("the index exists")
    };
    let fuzzy = |query| TextRequest::Fuzzy {
        query,
        snippets: true,
    };
    // One edit away from the new word, and from the stale one.
    let found = search(fuzzy("golanf"));
    assert_eq!(found.iter().map(|r| r.node_id).collect::<Vec<_>>(), [a]);
    assert!(
        found[0].snippet_html.contains("<b>golang</b>"),
        "the snippet is the store's text: {}",
        found[0].snippet_html
    );
    assert!(search(fuzzy("rusq")).is_empty(), "the stale text is gone");

    let found = search(TextRequest::Terms {
        query: "golang",
        language: None,
        snippets: true,
    });
    assert_eq!(found.iter().map(|r| r.node_id).collect::<Vec<_>>(), [a]);
    assert!(
        found[0].snippet_html.contains("fresh"),
        "{}",
        found[0].snippet_html
    );
}

/// When the writes the index lacks are not known (an event was dropped), a
/// search reads every node of the label from the store instead of trusting
/// the index for any of them.
#[test]
fn an_unknown_delta_answers_from_the_store_alone() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE TEXT INDEX article_body ON :Article(body)")
        .expect("create text index");
    db.execute_cypher("CREATE (:Article {title: 'a', body: 'stale rust words'})")
        .expect("create a");
    let a = node_id(&mut db, "a");

    // A queue of one: the second write is dropped, so the delta is unknown.
    let held = HeldCoverage::hold(&db, 1);
    db.execute_cypher("MATCH (n:Article {title: 'a'}) SET n.body = 'fresh golang words'")
        .expect("rewrite a");
    db.execute_cypher("CREATE (:Article {title: 'c', body: 'brand new rust'})")
        .expect("create c");
    held.await_worker();
    db.text_index_registry()
        .apply_changes(
            "Article",
            "body",
            &planted(&[(a, Some("stale rust words"))]),
        )
        .expect("plant stale text");

    assert_eq!(titles(&mut db, "rust"), ["c"]);
    assert_eq!(titles(&mut db, "golang"), ["a"]);
}

// ── Ranking against an index built from the same documents ─────────

const WORDS: [&str; 6] = ["rust", "graph", "engine", "vector", "index", "storage"];
/// Single words, a phrase and a prefix: every query shape a search runs.
const QUERIES: [&str; 8] = [
    "rust",
    "graph",
    "engine",
    "vector",
    "index",
    "storage",
    "\"rust graph\"",
    "stor*",
];

/// Each :Doc's BM25 score for `query`, by name, as `db` answers it (inside
/// transaction `tx` when given).
fn doc_scores(
    db: &mut Database,
    tx: Option<u64>,
    query: &str,
) -> std::collections::BTreeMap<String, f64> {
    let statement = format!(
        "MATCH (n:Doc) WHERE text_match(n.body, '{query}') \
         RETURN n.name AS name, text_score(n.body, '{query}') AS score"
    );
    let rows = match tx {
        Some(tx) => db.execute_in_transaction(tx, &statement, None),
        None => db.execute_cypher(&statement),
    }
    .expect("full-text read");
    rows.iter()
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

/// Compare every query's matches and scores in `db` (inside `tx` when given)
/// with a database that holds only `live`, indexed from nothing.
fn assert_ranks_as_built_from(
    db: &mut Database,
    tx: Option<u64>,
    live: &std::collections::BTreeMap<String, String>,
) {
    let (mut model, model_dir) = open_db();
    model
        .execute_cypher("CREATE TEXT INDEX doc_body ON :Doc(body)")
        .expect("create text index");
    for (name, body) in live {
        model
            .execute_cypher(&format!("CREATE (:Doc {{name: '{name}', body: '{body}'}})"))
            .expect("model doc");
    }
    for query in QUERIES {
        let ours = doc_scores(db, tx, query);
        let expected = doc_scores(&mut model, None, query);
        assert_eq!(
            ours.keys().collect::<Vec<_>>(),
            expected.keys().collect::<Vec<_>>(),
            "matches of {query}"
        );
        for (name, score) in &ours {
            let want = expected[name];
            assert!(
                (score - want).abs() <= 1e-4 * want.abs().max(1.0),
                "score of {name} for {query}: {score} against {want}"
            );
        }
    }
    // The database first, so its index worker stops before its directory
    // goes away.
    drop(model);
    drop(model_dir);
}

/// Random rewrites and deletions of existing :Doc nodes and new ones, over a
/// small vocabulary, applied by `run`; the documents left, by name.
fn churn(
    seed: u64,
    steps: usize,
    live: &mut std::collections::BTreeMap<String, String>,
    created: &mut usize,
    run: &mut dyn FnMut(&str),
) {
    // xorshift64: a fixed sequence, so a failure reproduces.
    let mut state = seed;
    let mut next = move |bound: usize| -> usize {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % bound as u64) as usize
    };
    for _ in 0..steps {
        let body: Vec<&str> = (0..1 + next(6)).map(|_| WORDS[next(WORDS.len())]).collect();
        let body = body.join(" ");
        let choice = next(3);
        if live.is_empty() || choice == 0 {
            let name = format!("d{created}");
            *created += 1;
            run(&format!("CREATE (:Doc {{name: '{name}', body: '{body}'}})"));
            live.insert(name, body);
            continue;
        }
        let name = live
            .keys()
            .nth(next(live.len()))
            .expect("a live doc")
            .clone();
        if choice == 1 {
            run(&format!(
                "MATCH (n:Doc {{name: '{name}'}}) SET n.body = '{body}'"
            ));
            live.insert(name, body);
        } else {
            run(&format!("MATCH (n:Doc {{name: '{name}'}}) DETACH DELETE n"));
            live.remove(&name);
        }
    }
}

/// Commits the index has not folded rank as an index holding them would:
/// the index here keeps the text every document had before the churn, as a
/// lagging worker leaves it, and the search replaces those documents with
/// the store's and scores both sides against the corpus they make together.
/// Words, a phrase and a prefix are compared, matches and BM25 scores, with
/// a database indexed from the surviving documents alone.
#[test]
fn unfolded_writes_rank_as_an_index_holding_them() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE TEXT INDEX doc_body ON :Doc(body)")
        .expect("create text index");
    let mut live = std::collections::BTreeMap::new();
    let mut created = 0;
    churn(
        0x9E37_79B9_7F4A_7C15,
        20,
        &mut live,
        &mut created,
        &mut |s| {
            db.execute_cypher(s).expect("write");
        },
    );
    let before: Vec<(u64, String)> = live
        .iter()
        .map(|(name, body)| {
            let rows = db
                .execute_cypher(&format!(
                    "MATCH (n:Doc {{name: '{name}'}}) RETURN id(n) AS id"
                ))
                .expect("id");
            let id = match rows[0].get("id") {
                Some(Value::Int(id)) => *id as u64,
                other => panic!("no id for {name}: {other:?}"),
            };
            (id, body.clone())
        })
        .collect();
    let before: Vec<(u64, Option<&str>)> = before
        .iter()
        .map(|(id, body)| (*id, Some(body.as_str())))
        .collect();

    let held = HeldCoverage::hold(&db, 4096);
    churn(
        0xD1B5_4A32_D192_ED03,
        40,
        &mut live,
        &mut created,
        &mut |s| {
            db.execute_cypher(s).expect("write");
        },
    );
    held.await_worker();
    db.text_index_registry()
        .apply_changes("Doc", "body", &planted(&before))
        .expect("plant the text before the churn");

    assert_ranks_as_built_from(&mut db, None, &live);
}

/// A transaction's own uncommitted writes rank as an index holding them
/// would: inside it, every query's matches and scores are those of a
/// database indexed from the documents the transaction sees.
#[test]
fn own_writes_rank_as_an_index_holding_them() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE TEXT INDEX doc_body ON :Doc(body)")
        .expect("create text index");
    let mut live = std::collections::BTreeMap::new();
    let mut created = 0;
    churn(
        0x9E37_79B9_7F4A_7C15,
        20,
        &mut live,
        &mut created,
        &mut |s| {
            db.execute_cypher(s).expect("write");
        },
    );

    let tx = db.begin_transaction();
    churn(
        0xD1B5_4A32_D192_ED03,
        40,
        &mut live,
        &mut created,
        &mut |s| {
            db.execute_in_transaction(tx, s, None)
                .expect("write in transaction");
        },
    );
    assert_ranks_as_built_from(&mut db, Some(tx), &live);
    db.rollback_transaction(tx).expect("rollback");
}

/// A commit refused for a write conflict leaves nothing searchable: the
/// winner's text is found, the loser's is not, in the index and through the
/// writes it has not folded alike.
#[test]
fn a_refused_commit_is_not_searchable() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE TEXT INDEX article_body ON :Article(body)")
        .expect("create text index");
    db.execute_cypher("CREATE (:Article {title: 'a', body: 'committed words'})")
        .expect("create");

    let winner = db.begin_transaction();
    let loser = db.begin_transaction();
    db.execute_in_transaction(
        winner,
        "MATCH (n:Article {title: 'a'}) SET n.body = 'fresh golang words'",
        None,
    )
    .expect("winner's write");
    db.execute_in_transaction(
        loser,
        "MATCH (n:Article {title: 'a'}) SET n.body = 'private draft'",
        None,
    )
    .expect("loser's write");
    db.commit_transaction(winner)
        .expect("the first commit lands");
    assert!(
        db.commit_transaction(loser).is_err(),
        "the second writer of the node must be refused"
    );

    assert_eq!(titles(&mut db, "golang"), ["a"]);
    assert!(
        titles(&mut db, "private").is_empty(),
        "the refused text leaked"
    );
    assert!(titles(&mut db, "committed").is_empty());
}

/// An index left behind the store by a crash is not trusted after reopen:
/// the text it held for a node rewritten and one deleted before the crash is
/// replaced by what the store holds, and a node it never took is found.
#[test]
fn an_index_behind_the_store_at_a_crash_is_caught_up_on_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (a, b) = {
        let mut db = Database::open(dir.path()).expect("open db");
        db.execute_cypher("CREATE TEXT INDEX article_body ON :Article(body)")
            .expect("create text index");
        db.execute_cypher("CREATE (:Article {title: 'a', body: 'stale rust words'})")
            .expect("create a");
        db.execute_cypher("CREATE (:Article {title: 'b', body: 'doomed rust words'})")
            .expect("create b");
        let (a, b) = (node_id(&mut db, "a"), node_id(&mut db, "b"));

        let held = HeldCoverage::hold(&db, 1024);
        db.execute_cypher("MATCH (n:Article {title: 'a'}) SET n.body = 'fresh golang words'")
            .expect("rewrite a");
        db.execute_cypher("MATCH (n:Article {title: 'b'}) DETACH DELETE n")
            .expect("delete b");
        db.execute_cypher("CREATE (:Article {title: 'c', body: 'brand new rust'})")
            .expect("create c");
        held.await_worker();
        // What the index holds at the crash: the text before the writes, and
        // nothing of the new node.
        db.text_index_registry()
            .apply_changes(
                "Article",
                "body",
                &planted(&[
                    (a, Some("stale rust words")),
                    (b, Some("doomed rust words")),
                ]),
            )
            .expect("plant stale text");
        let c = node_id(&mut db, "c");
        db.text_index_registry()
            .apply_changes("Article", "body", &planted(&[(c, None)]))
            .expect("take the new node out");
        (a, b)
    };
    let _ = (a, b);

    let mut db = Database::open(dir.path()).expect("reopen");
    assert_eq!(titles(&mut db, "rust"), ["c"]);
    assert_eq!(titles(&mut db, "golang"), ["a"]);
    assert!(titles(&mut db, "stale").is_empty());
    assert!(titles(&mut db, "doomed").is_empty());
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

    // After DROP, text_match() must hard-fail with a clear message
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

// ── Multi-field DDL ────────────────────────────────────────────────

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

// ── Retrieval completeness ────────────────────────────────────────

/// A prefix matches every document holding a word it starts, however many
/// distinct words that is: membership has no expansion cutoff, through the
/// index and through the unfolded writes alike.
#[test]
fn a_prefix_matches_every_word_it_starts() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE TEXT INDEX article_body ON :Article(body)")
        .expect("create text index");
    const WORDS: usize = 80;
    // One statement, so the words land in one segment of the index, where a
    // per-segment expansion limit would bite.
    let rows: Vec<String> = (0..WORDS)
        .map(|i| format!("{{name: 'n{i}', body: 'zeta{i:03}q'}}"))
        .collect();
    db.execute_cypher(&format!(
        "UNWIND [{}] AS r CREATE (:Article {{name: r.name, body: r.body}})",
        rows.join(", ")
    ))
    .expect("create");
    let rows = db
        .execute_cypher("MATCH (n:Article) WHERE text_match(n.body, 'zeta*') RETURN n.name AS name")
        .expect("prefix search");
    assert_eq!(rows.len(), WORDS, "every word the prefix starts");
}

// ── Temporal labels ───────────────────────────────────────────────

/// On a temporal label a search matches the state each node has now, the
/// one a bare MATCH returns: a version that ended, one not yet valid, and
/// one whose validity runs out with no later commit are not matched, on the
/// index path and on the filter path alike.
#[test]
fn a_temporal_label_is_searched_at_its_state_valid_now() {
    const YEAR: i64 = 365 * 24 * 3600 * 1_000_000;
    let now = || {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock after epoch")
            .as_micros() as i64
    };
    let (mut db, _dir) = open_db();
    db.execute_cypher(
        "CREATE NODE TYPE Emp TEMPORAL WITH (name: STRING, kind: STRING, bio: STRING, valid_from: INT, valid_to: INT)",
    )
    .expect("temporal label");
    db.execute_cypher("CREATE TEXT INDEX emp_bio ON :Emp(bio)")
        .expect("create text index");
    let t = now();
    for (name, bio, from, to) in [
        ("current", "golang", t - YEAR, None),
        ("ended", "cobol", t - 2 * YEAR, Some(t - YEAR)),
        ("not yet", "zig", t + YEAR, None),
        ("expiring", "pascal", t - YEAR, Some(t + 2_000_000)),
    ] {
        let to = to.map_or(String::new(), |to| format!(", valid_to: {to}"));
        db.execute_cypher(&format!(
            "CREATE (:Emp {{name: '{name}', kind: 'e', bio: '{bio}', valid_from: {from}{to}}})"
        ))
        .expect("create");
    }

    let search = |db: &mut Database, word: &str| -> [Vec<Value>; 2] {
        ["MATCH (n:Emp)", "MATCH (n:Emp {kind: 'e'})"].map(|head| {
            db.execute_cypher(&format!(
                "{head} WHERE text_match(n.bio, '{word}') RETURN n.name AS name ORDER BY name"
            ))
            .expect("search")
            .iter()
            .map(|r| r["name"].clone())
            .collect()
        })
    };
    let only = |name: &str| {
        [
            vec![Value::String(name.into())],
            vec![Value::String(name.into())],
        ]
    };
    let none: [Vec<Value>; 2] = [Vec::new(), Vec::new()];

    assert_eq!(search(&mut db, "golang"), only("current"));
    assert_eq!(search(&mut db, "cobol"), none, "a version that ended");
    assert_eq!(search(&mut db, "zig"), none, "a version not yet valid");
    assert_eq!(search(&mut db, "pascal"), only("expiring"));

    // A named instant reads each node's state then, through the index too.
    let past = t - 3 * YEAR / 2;
    let rows = db
        .execute_cypher(&format!(
            "MATCH (n:Emp) WHERE temporal_active_at(n, {past}) AND text_match(n.bio, 'cobol') \
             RETURN n.name AS name"
        ))
        .expect("search the past");
    assert_eq!(
        rows.iter().map(|r| r["name"].clone()).collect::<Vec<_>>(),
        [Value::String("ended".into())],
        "the state valid at the named instant"
    );

    // Its validity runs out without another commit.
    while now() < t + 2_500_000 {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    assert_eq!(search(&mut db, "pascal"), none, "a version that ran out");
    assert_eq!(search(&mut db, "golang"), only("current"));

    // The index catches up by itself: with no write, the ended state is
    // folded out and searches stop reading that node from its timeline.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while !db
        .text_index_registry()
        .outside("Emp", "bio", now())
        .is_empty()
    {
        assert!(
            std::time::Instant::now() < deadline,
            "the ended state was never refolded"
        );
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    assert_eq!(search(&mut db, "pascal"), none);
}

/// A temporal label ranks as an index of the states valid now would: every
/// score equals the one an ordinary label holding exactly those texts gets,
/// so ended, future and superseded versions take no part in the corpus.
#[test]
fn a_temporal_label_ranks_as_its_states_valid_now() {
    const YEAR: i64 = 365 * 24 * 3600 * 1_000_000;
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock after epoch")
        .as_micros() as i64;
    let (mut db, _dir) = open_db();
    db.execute_cypher(
        "CREATE NODE TYPE Emp TEMPORAL WITH (name: STRING, bio: STRING, valid_from: INT, valid_to: INT)",
    )
    .expect("temporal label");
    db.execute_cypher("CREATE TEXT INDEX emp_bio ON :Emp(bio)")
        .expect("emp index");
    db.execute_cypher("CREATE TEXT INDEX ref_bio ON :Ref(bio)")
        .expect("reference index");
    /// A version not valid now: its text, start and end in years from now.
    type Other = (&'static str, i64, Option<i64>);
    // (name, bio valid now or None, versions that are not valid now)
    let people: [(&str, Option<&str>, &[Other]); 5] = [
        (
            "a",
            Some("rust graph rust"),
            &[("rust rust rust rust", -3, Some(-1))],
        ),
        ("b", Some("graph engine"), &[]),
        ("c", None, &[("rust storage", -3, Some(-2))]),
        ("d", None, &[("rust rust graph", 1, None)]),
        ("e", Some("rust"), &[("engine engine", 2, None)]),
    ];
    // Every person is one node whose timeline holds all of its versions.
    for (name, now_bio, others) in people {
        let mut versions: Vec<(&str, i64, Option<i64>)> = others
            .iter()
            .map(|(bio, from, to)| (*bio, t + from * YEAR, to.map(|to| t + to * YEAR)))
            .collect();
        if let Some(bio) = now_bio {
            versions.push((bio, t - YEAR / 2, Some(t + YEAR / 2)));
            db.execute_cypher(&format!("CREATE (:Ref {{name: '{name}', bio: '{bio}'}})"))
                .expect("reference node");
        }
        versions.sort_by_key(|(_, from, _)| *from);
        let (bio, from, to) = versions[0];
        let to_clause = to.map_or(String::new(), |to| format!(", valid_to: {to}"));
        let rows = db
            .execute_cypher(&format!(
                "CREATE (n:Emp {{name: '{name}', bio: '{bio}', valid_from: {from}{to_clause}}}) RETURN n"
            ))
            .expect("first version");
        let id = match rows[0].get("n") {
            Some(Value::Int(id)) => *id as u64,
            other => panic!("node id: {other:?}"),
        };
        for (bio, from, to) in &versions[1..] {
            seed_version(&db, id, &[("name", name), ("bio", bio)], *from, *to);
        }
    }
    let scores = |db: &mut Database, label: &str, word: &str| -> Vec<(Value, Value)> {
        db.execute_cypher(&format!(
            "MATCH (n:{label}) WHERE text_match(n.bio, '{word}') \
             RETURN n.name AS name, text_score(n.bio, '{word}') AS score ORDER BY name"
        ))
        .expect("search")
        .iter()
        .map(|r| (r["name"].clone(), r["score"].clone()))
        .collect()
    };
    for word in ["rust", "graph", "engine", "storage"] {
        assert_eq!(
            scores(&mut db, "Emp", word),
            scores(&mut db, "Ref", word),
            "scores for '{word}'"
        );
    }
}
