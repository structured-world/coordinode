//! `ALTER NODE TYPE ... SET IMPLIES (...)` end to end: the new implication is
//! proved over the stored carriers of the type before it is put in force,
//! refused with the node that breaks it, and checked as a declaration is.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_core::graph::types::Value;
use coordinode_embed::Database;

/// `Thing` and `Task`, both STRICT with a title; `Activity` auxiliary.
fn things() -> Database {
    let mut db = Database::open_in_memory().expect("open");
    for ddl in [
        "CREATE NODE TYPE Thing WITH (title: STRING)",
        "CREATE NODE TYPE Task WITH (title: STRING, due: TIMESTAMP)",
        "CREATE LABEL Activity AUXILIARY",
    ] {
        db.execute_cypher(ddl)
            .unwrap_or_else(|e| panic!("{ddl}: {e}"));
    }
    db
}

fn count(db: &mut Database, query: &str) -> usize {
    db.execute_cypher(query)
        .unwrap_or_else(|e| panic!("{query}: {e}"))
        .len()
}

fn refused(db: &mut Database, query: &str) -> String {
    match db.execute_cypher(query) {
        Ok(rows) => panic!("{query} was accepted: {rows:?}"),
        Err(e) => e.to_string(),
    }
}

/// The implication as the statement's `implies` column reports it.
fn implies_of(column: Option<&Value>) -> Vec<String> {
    match column {
        Some(Value::Array(labels)) => labels
            .iter()
            .map(|l| match l {
                Value::String(s) => s.clone(),
                other => panic!("label {other:?}"),
            })
            .collect(),
        other => panic!("implies {other:?}"),
    }
}

/// A type without stored nodes comes to imply a label at once, and its nodes
/// written afterwards are found under it.
#[test]
fn a_type_without_nodes_comes_to_imply_a_label() {
    let mut db = things();
    let rows = db
        .execute_cypher("ALTER NODE TYPE Task SET IMPLIES (Thing)")
        .expect("alter");
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert_eq!(implies_of(rows[0].get("implies")), ["Thing"]);
    assert_eq!(rows[0].get("changed"), Some(&Value::Bool(true)));
    db.execute_cypher("CREATE (:Task {title: 't'})")
        .expect("node");
    assert_eq!(count(&mut db, "MATCH (n:Thing) RETURN n"), 1);
}

/// A scan of a label a type comes to imply finds the type's stored nodes,
/// beside the label's own and those carrying it explicitly, each once; and
/// stops finding them when the implication is dropped, while the others
/// stay. Counted and paged the same way.
#[test]
fn a_label_scan_follows_the_implication_both_ways() {
    let mut db = things();
    db.execute_cypher(
        "CREATE (:Task {title: 'a'}), (:Task:Activity {title: 'b'}), (:Thing {title: 'c'})",
    )
    .expect("nodes");
    db.execute_cypher("ALTER NODE TYPE Task SET IMPLIES (Thing)")
        .expect("gain");
    assert_eq!(count(&mut db, "MATCH (n:Thing) RETURN n"), 3);
    assert_eq!(
        count(
            &mut db,
            "MATCH (n:Thing) RETURN n.title AS t ORDER BY t LIMIT 2"
        ),
        2
    );
    let rows = db
        .execute_cypher("MATCH (n:Thing) RETURN count(n) AS c")
        .expect("count");
    assert_eq!(rows[0].get("c"), Some(&Value::Int(3)));
    db.execute_cypher("CREATE (:Task {title: 'd'})")
        .expect("a task written after the change");
    assert_eq!(count(&mut db, "MATCH (n:Thing) RETURN n"), 4);

    db.execute_cypher("ALTER NODE TYPE Task SET IMPLIES ()")
        .expect("loss");
    assert_eq!(
        count(&mut db, "MATCH (n:Thing) RETURN n"),
        1,
        "the thing alone"
    );
    assert_eq!(
        count(&mut db, "MATCH (n:Task) RETURN n"),
        3,
        "tasks stay tasks"
    );
    assert_eq!(
        count(&mut db, "MATCH (n:Activity) RETURN n"),
        1,
        "an explicit label stays"
    );
    let rows = db
        .execute_cypher("MATCH (n:Thing) RETURN count(n) AS c")
        .expect("count");
    assert_eq!(rows[0].get("c"), Some(&Value::Int(1)));
}

/// The stored nodes of a type are found under a label it comes to imply,
/// by a scan of the label and through an index on it, once the change
/// answers; and leave it when the implication is dropped.
#[test]
fn stored_carriers_follow_the_implication() {
    let mut db = things();
    db.execute_cypher("CREATE INDEX thing_title ON :Thing(title)")
        .expect("index");
    db.execute_cypher("CREATE (:Task {title: 'x'}), (:Task {title: 'y'}), (:Thing {title: 'x'})")
        .expect("nodes");
    db.execute_cypher("ALTER NODE TYPE Task SET IMPLIES (Thing)")
        .expect("alter over stored tasks");
    assert_eq!(count(&mut db, "MATCH (n:Thing) RETURN n"), 3, "scanned");
    assert_eq!(
        count(&mut db, "MATCH (n:Thing {title: 'x'}) RETURN n"),
        2,
        "through the index"
    );
    db.execute_cypher("ALTER NODE TYPE Task SET IMPLIES ()")
        .expect("implication dropped");
    assert_eq!(count(&mut db, "MATCH (n:Thing) RETURN n"), 1, "scanned");
    assert_eq!(
        count(&mut db, "MATCH (n:Thing {title: 'x'}) RETURN n"),
        1,
        "through the index"
    );
}

/// A uniqueness on a label a type comes to imply spans the type's stored
/// nodes: one holding a value a carrier of the label holds refuses the
/// change, which leaves the implication, the label's members and the
/// uniqueness as they were; once the value is free the change goes through
/// and the uniqueness binds the type's nodes.
#[test]
fn a_gained_uniqueness_over_a_held_value_refuses_the_change() {
    let mut db = things();
    db.execute_cypher("CREATE CONSTRAINT thing_title FOR (n:Thing) REQUIRE n.title IS UNIQUE")
        .expect("constraint");
    db.execute_cypher("CREATE (:Thing {title: 'x'}), (:Task {title: 'x'}), (:Task {title: 'y'})")
        .expect("nodes");
    let err = refused(&mut db, "ALTER NODE TYPE Task SET IMPLIES (Thing)");
    assert!(err.contains('x') || err.contains("unique"), "{err}");
    assert_eq!(count(&mut db, "MATCH (n:Thing) RETURN n"), 1, "unchanged");
    db.execute_cypher("CREATE (:Thing {title: 'y'})")
        .expect("the old guarantee covers things alone");
    db.execute_cypher("MATCH (n:Task {title: 'x'}) SET n.title = 'z'")
        .expect("free x");
    db.execute_cypher("MATCH (n:Thing {title: 'y'}) DELETE n")
        .expect("free y");
    db.execute_cypher("ALTER NODE TYPE Task SET IMPLIES (Thing)")
        .expect("no value held twice now");
    assert_eq!(count(&mut db, "MATCH (n:Thing) RETURN n"), 3);
    let err = refused(&mut db, "CREATE (:Task {title: 'x'})");
    assert!(err.contains('x') || err.contains("unique"), "{err}");
    let err = refused(&mut db, "CREATE (:Thing {title: 'z'})");
    assert!(err.contains('z') || err.contains("unique"), "{err}");
}

/// Whether no index keeps parts apart any more: every concluded change is
/// folded into the indexes' own entries.
fn folded(db: &Database) -> bool {
    use coordinode_modality::IndexStore as _;
    coordinode_modality::LocalIndexStore::new(db.engine())
        .list_definitions()
        .expect("definitions")
        .iter()
        .all(|def| def.selection.is_empty())
}

/// Wait up to five seconds for the parts of the changes made to be folded.
fn await_folded(db: &Database) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !folded(db) {
        assert!(std::time::Instant::now() < deadline, "parts left unfolded");
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

/// Once a change is put in force, what it kept apart is folded back into
/// the lookup's and the index's own entries: each scope is one again, the
/// label's counter answers the count and agrees with a scan, and an index
/// lookup finds the same nodes; both ways.
#[test]
fn a_concluded_change_is_folded_into_one_scope() {
    let mut db = things();
    db.execute_cypher("CREATE INDEX thing_title ON :Thing(title)")
        .expect("index");
    db.execute_cypher("CREATE (:Task {title: 'x'}), (:Task {title: 'y'}), (:Thing {title: 'x'})")
        .expect("nodes");
    let counted = |db: &mut Database| {
        let rows = db
            .execute_cypher("MATCH (n:Thing) RETURN count(n) AS c")
            .expect("count");
        rows[0].get("c").cloned()
    };
    for (alter, members, through_index) in [
        ("ALTER NODE TYPE Task SET IMPLIES (Thing)", 3, 2),
        ("ALTER NODE TYPE Task SET IMPLIES ()", 1, 1),
    ] {
        db.execute_cypher(alter).expect(alter);
        await_folded(&db);
        assert_eq!(
            counted(&mut db),
            Some(Value::Int(members)),
            "{alter}: counter"
        );
        assert_eq!(
            count(&mut db, "MATCH (n:Thing) WHERE n.title + '' <> '' RETURN n"),
            members as usize,
            "{alter}: scan"
        );
        assert_eq!(
            count(&mut db, "MATCH (n:Thing {title: 'x'}) RETURN n"),
            through_index,
            "{alter}: index"
        );
    }
}

/// Run `statement` in its own transaction and return its commit timestamp.
fn commit(db: &mut Database, statement: &str) -> u64 {
    let tx = db.begin_transaction().expect("begin");
    db.execute_in_transaction(tx, statement, None)
        .unwrap_or_else(|e| panic!("{statement}: {e}"));
    db.commit_transaction(tx)
        .expect("commit")
        .commit_ts
        .as_raw()
}

/// Names and full-text scores of the things matching `words` through the
/// text index, by name; read at `as_of` when given.
fn text_hits(db: &mut Database, words: &str, as_of: Option<u64>) -> Vec<(String, f64)> {
    let as_of = as_of.map_or(String::new(), |ts| format!(" AS OF TIMESTAMP {ts}"));
    let query = format!(
        "MATCH (n:Thing) WHERE text_match(n.title, '{words}') \
         RETURN n.title AS name, text_score(n.title, '{words}') AS score \
         ORDER BY name{as_of}"
    );
    db.execute_cypher(&query)
        .unwrap_or_else(|e| panic!("{query}: {e}"))
        .iter()
        .map(|row| match (row.get("name"), row.get("score")) {
            (Some(Value::String(name)), Some(Value::Float(score))) => (name.clone(), *score),
            other => panic!("row {other:?}"),
        })
        .collect()
}

/// A database of things only, the oracle a text read is compared with:
/// `titles` as the things a text index on `:Thing(title)` holds.
fn things_holding(dir: &std::path::Path, titles: &[&str]) -> Database {
    let mut db = Database::open(dir).expect("open");
    db.execute_cypher("CREATE NODE TYPE Thing WITH (title: STRING)")
        .expect("type");
    db.execute_cypher("CREATE TEXT INDEX thing_text ON :Thing(title)")
        .expect("text index");
    for title in titles {
        db.execute_cypher(&format!("CREATE (:Thing {{title: '{title}'}})"))
            .expect("thing");
    }
    db
}

fn assert_same_hits(read: &[(String, f64)], oracle: &[(String, f64)], what: &str) {
    assert_eq!(
        read.iter().map(|(n, _)| n).collect::<Vec<_>>(),
        oracle.iter().map(|(n, _)| n).collect::<Vec<_>>(),
        "{what}: {read:?} vs {oracle:?}"
    );
    for ((_, a), (_, b)) in read.iter().zip(oracle) {
        assert!((a - b).abs() < 1e-5, "{what}: {read:?} vs {oracle:?}");
    }
}

/// A text read keeps the interpretation of its view of the catalog. Once
/// the text index has brought a change of what a type implies back to one
/// interpretation, a read at a timestamp before the change still finds the
/// label's carriers then, and ranks them as an index holding only those
/// would; a read now finds the carriers the change put in force. Before the
/// index's own interpretation was compared with the reader's, an index that
/// had settled a later change answered the old view whole.
#[test]
fn a_text_read_keeps_its_views_interpretation_after_a_change() {
    text_reads_across_a_change(
        "CREATE NODE TYPE Task WITH (title: STRING)",
        "ALTER NODE TYPE Task SET IMPLIES (Thing)",
        (&["rust graph"], &["rust graph", "rust rust tools"]),
    );
}

/// The other way round: a type stops implying the label. A read at a
/// timestamp before still finds the type's nodes, which its view of the
/// catalog holds as carriers although the type implies nothing now. Before
/// the carriers were judged by the view's own catalog, the type was passed
/// over for implying nothing in the latest one, and the old read lost them.
#[test]
fn a_text_read_before_a_lost_implication_keeps_its_carriers() {
    text_reads_across_a_change(
        "CREATE NODE TYPE Task IMPLIES (Thing) WITH (title: STRING)",
        "ALTER NODE TYPE Task SET IMPLIES ()",
        (&["rust graph", "rust rust tools"], &["rust graph"]),
    );
}

/// A thing and a task, a text index on `:Thing(title)`, the task's type
/// declared by `task` and changed by `alter`. Once the index settles the
/// change, a read at a timestamp before it ranks as an index of the
/// `then` titles, and a read now as one of the `now` titles.
fn text_reads_across_a_change(task: &str, alter: &str, (then, now): (&[&str], &[&str])) {
    let dirs = (
        tempfile::tempdir().expect("tempdir"),
        tempfile::tempdir().expect("tempdir"),
        tempfile::tempdir().expect("tempdir"),
    );
    let mut db = Database::open(dirs.0.path()).expect("open");
    for ddl in [
        "CREATE NODE TYPE Thing WITH (title: STRING)",
        task,
        "CREATE TEXT INDEX thing_text ON :Thing(title)",
    ] {
        db.execute_cypher(ddl)
            .unwrap_or_else(|e| panic!("{ddl}: {e}"));
    }
    commit(&mut db, "CREATE (:Thing {title: 'rust graph'})");
    let before = commit(&mut db, "CREATE (:Task {title: 'rust rust tools'})");
    db.execute_cypher(alter).expect("alter");
    await_folded(&db);
    // The index itself comes to hold the facts now in force, so a read now
    // is answered from it rather than from the store.
    let in_force = coordinode_query::index::membership::label_carriers(db.engine(), "Thing")
        .expect("carriers");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while db.text_index_registry().held("Thing", "title")
        != Some(coordinode_query::index::selection::Held::built(
            in_force.clone(),
        ))
    {
        assert!(
            std::time::Instant::now() < deadline,
            "the text index never settled the change: {:?}",
            db.text_index_registry().held("Thing", "title")
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    let mut then = things_holding(dirs.1.path(), then);
    let mut now = things_holding(dirs.2.path(), now);
    for words in ["rust", "graph", "tools"] {
        assert_same_hits(
            &text_hits(&mut db, words, Some(before)),
            &text_hits(&mut then, words, None),
            &format!("{words} before the change"),
        );
        assert_same_hits(
            &text_hits(&mut db, words, None),
            &text_hits(&mut now, words, None),
            &format!("{words} now"),
        );
    }
}

/// A change of what a type implies is put in force only once the text index
/// on the label answers readers of the new carriers itself. With the index's
/// preparation held back, the ALTER does not complete; released, it does,
/// and a search right after it is the index's own answer, matching an index
/// that holds the new carriers alone. Before, the change went in force at
/// once and the index, holding only the old carriers, answered new readers
/// by reading every node of the store.
#[test]
fn a_change_waits_for_the_text_index_to_follow_it() {
    use std::sync::mpsc;
    use std::time::Duration;

    let dirs = (
        tempfile::tempdir().expect("tempdir"),
        tempfile::tempdir().expect("tempdir"),
    );
    let mut db = Database::open(dirs.0.path()).expect("open");
    for ddl in [
        "CREATE NODE TYPE Thing WITH (title: STRING)",
        "CREATE NODE TYPE Task WITH (title: STRING)",
        "CREATE TEXT INDEX thing_text ON :Thing(title)",
        "CREATE (:Thing {title: 'rust graph'}), (:Task {title: 'rust rust tools'})",
    ] {
        db.execute_cypher(ddl)
            .unwrap_or_else(|e| panic!("{ddl}: {e}"));
    }
    #[derive(Debug, PartialEq)]
    enum Step {
        /// The worker began preparing the index and is held there.
        Preparing,
        /// The ALTER returned.
        Altered,
    }
    let (steps, step) = mpsc::channel();
    let (release, released) = mpsc::channel::<()>();
    let preparing = steps.clone();
    *coordinode_embed::text_worker::WHILE_PREPARING.lock() = Some(Box::new(move || {
        let _ = preparing.send(Step::Preparing);
        let _ = released.recv();
    }));
    let alter = std::thread::spawn(move || {
        let altered = db.execute_cypher("ALTER NODE TYPE Task SET IMPLIES (Thing)");
        let _ = steps.send(Step::Altered);
        (db, altered)
    });
    // The ALTER returning first means the change went in force before the
    // index was even prepared for it.
    assert_eq!(
        step.recv_timeout(Duration::from_secs(30)),
        Ok(Step::Preparing),
        "the change was put in force before the text index followed it"
    );
    assert!(
        matches!(
            step.recv_timeout(Duration::from_millis(500)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ),
        "the change was put in force while the text index was being prepared"
    );
    release.send(()).expect("release the preparation");
    let (mut db, altered) = alter.join().expect("the ALTER thread");
    altered.expect("alter");

    let carriers = coordinode_query::index::membership::label_carriers(db.engine(), "Thing")
        .expect("carriers");
    let held = db
        .text_index_registry()
        .held("Thing", "title")
        .expect("the index");
    assert!(
        matches!(
            coordinode_query::index::selection::answer(carriers.as_ref(), &held),
            coordinode_query::index::selection::Answer::Whole
                | coordinode_query::index::selection::Answer::Without(_)
        ),
        "right after the change the index answers its readers itself: {held:?}"
    );
    let mut now = things_holding(dirs.1.path(), &["rust graph", "rust rust tools"]);
    for words in ["rust", "graph", "tools"] {
        assert_same_hits(
            &text_hits(&mut db, words, None),
            &text_hits(&mut now, words, None),
            &format!("{words} right after the change"),
        );
    }
}

/// The implication goes A→B→A: Task comes to imply Thing, stops, and comes
/// to imply it again, each change folded before the next. After each one
/// the label scan, its counter, a B-tree lookup and a full-text search all
/// answer the implication in force, and the text index ranks as one holding
/// exactly its carriers: nothing an earlier change kept apart leaks into a
/// later one.
#[test]
fn an_implication_changed_there_and_back_answers_each_state() {
    let dirs = (
        tempfile::tempdir().expect("tempdir"),
        tempfile::tempdir().expect("tempdir"),
        tempfile::tempdir().expect("tempdir"),
    );
    let mut db = Database::open(dirs.0.path()).expect("open");
    for ddl in [
        "CREATE NODE TYPE Thing WITH (title: STRING)",
        "CREATE NODE TYPE Task WITH (title: STRING)",
        "CREATE INDEX thing_title ON :Thing(title)",
        "CREATE TEXT INDEX thing_text ON :Thing(title)",
        "CREATE (:Thing {title: 'rust graph'}), (:Task {title: 'rust rust tools'})",
    ] {
        db.execute_cypher(ddl)
            .unwrap_or_else(|e| panic!("{ddl}: {e}"));
    }
    let mut with_task = things_holding(dirs.1.path(), &["rust graph", "rust rust tools"]);
    let mut without = things_holding(dirs.2.path(), &["rust graph"]);
    for (alter, carriers) in [
        ("ALTER NODE TYPE Task SET IMPLIES (Thing)", 2),
        ("ALTER NODE TYPE Task SET IMPLIES ()", 1),
        ("ALTER NODE TYPE Task SET IMPLIES (Thing)", 2),
    ] {
        db.execute_cypher(alter).expect(alter);
        await_folded(&db);
        let in_force = coordinode_query::index::membership::label_carriers(db.engine(), "Thing")
            .expect("carriers");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while db.text_index_registry().held("Thing", "title")
            != Some(coordinode_query::index::selection::Held::built(
                in_force.clone(),
            ))
        {
            assert!(
                std::time::Instant::now() < deadline,
                "{alter}: the text index never settled"
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        assert_eq!(
            count(&mut db, "MATCH (n:Thing) WHERE n.title + '' <> '' RETURN n"),
            carriers,
            "{alter}: scan"
        );
        let rows = db
            .execute_cypher("MATCH (n:Thing) RETURN count(n) AS c")
            .expect("count");
        assert_eq!(
            rows[0].get("c"),
            Some(&Value::Int(carriers as i64)),
            "{alter}: counter"
        );
        assert_eq!(
            count(
                &mut db,
                "MATCH (n:Thing {title: 'rust rust tools'}) RETURN n"
            ),
            carriers - 1,
            "{alter}: B-tree lookup"
        );
        let oracle = if carriers == 2 {
            &mut with_task
        } else {
            &mut without
        };
        for words in ["rust", "graph", "tools"] {
            assert_same_hits(
                &text_hits(&mut db, words, None),
                &text_hits(oracle, words, None),
                &format!("{alter}: {words}"),
            );
        }
    }
}

/// Titles of the `k` things nearest `[1, 0]` through the vector index on
/// `:Thing(emb)`, nearest first; read at `as_of` when given.
fn nearest_things(db: &mut Database, k: usize, as_of: Option<u64>) -> Vec<String> {
    let as_of = as_of.map_or(String::new(), |ts| format!(" AS OF TIMESTAMP {ts}"));
    let query = format!(
        "MATCH (n:Thing) WITH n, vector_distance(n.emb, [1.0, 0.0]) AS d \
         ORDER BY d LIMIT {k} RETURN n.title AS title{as_of}"
    );
    let plan = db.explain_cypher(&query).expect("explain");
    assert!(plan.contains("HnswScan("), "not the index path:\n{plan}");
    db.execute_cypher(&query)
        .unwrap_or_else(|e| panic!("{query}: {e}"))
        .iter()
        .map(|row| match row.get("title") {
            Some(Value::String(title)) => title.clone(),
            other => panic!("title {other:?}"),
        })
        .collect()
}

/// Wait until the vector index on `:Thing(emb)` holds the carriers of
/// `:Thing` now in force, and nothing of another interpretation.
fn await_vector_settled(db: &Database, what: &str) {
    let in_force = coordinode_query::index::membership::label_carriers(db.engine(), "Thing")
        .expect("carriers");
    let want = Some(coordinode_query::index::selection::Held::built(in_force));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while db.vector_index_registry().held("Thing", "emb") != want {
        assert!(
            std::time::Instant::now() < deadline,
            "{what}: the vector index never settled: {:?}",
            db.vector_index_registry().held("Thing", "emb")
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

/// A vector index on a label whose carriers a change of what a type implies
/// moves follows the change instead of refusing it. The nodes nearest the
/// query, the type's nodes far nearer than the label's own, are searched
/// through the index at each state: before the change and at a timestamp
/// before it, the label's own things alone; after it, the tasks too; and
/// after the implication is dropped again, the things alone. The index
/// itself comes to hold each interpretation in force, so the reads are its
/// own and not the store's.
#[test]
fn a_vector_index_follows_a_change_of_what_a_type_implies() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    for ddl in [
        "CREATE NODE TYPE Thing WITH (title: STRING)",
        "ALTER LABEL Thing SET SCHEMA FLEXIBLE",
        "CREATE NODE TYPE Task WITH (title: STRING)",
        "ALTER LABEL Task SET SCHEMA FLEXIBLE",
        r#"CREATE VECTOR INDEX thing_emb ON :Thing(emb) OPTIONS {metric: "l2"}"#,
    ] {
        db.execute_cypher(ddl)
            .unwrap_or_else(|e| panic!("{ddl}: {e}"));
    }
    // Tasks lie nearer the query than every thing, so a search that kept
    // them out by discarding a beam of nearest nodes would find nothing.
    for i in 0..40 {
        db.execute_cypher(&format!(
            "CREATE (:Task {{title: 'task{i:02}', emb: [1.0, {}]}})",
            f64::from(i) * 0.001
        ))
        .expect("task");
    }
    for i in 0..5 {
        db.execute_cypher(&format!(
            "CREATE (:Thing {{title: 'thing{i}', emb: [{}, 1.0]}})",
            -f64::from(i)
        ))
        .expect("thing");
    }
    let things: Vec<String> = (0..3).map(|i| format!("thing{i}")).collect();
    let tasks: Vec<String> = (0..3).map(|i| format!("task{i:02}")).collect();
    await_vector_settled(&db, "built");
    assert_eq!(nearest_things(&mut db, 3, None), things, "before");

    let before = commit(
        &mut db,
        "CREATE (:Thing {title: 'far', emb: [-100.0, 0.0]})",
    );
    db.execute_cypher("ALTER NODE TYPE Task SET IMPLIES (Thing)")
        .expect("a vector index follows the change");
    await_folded(&db);
    await_vector_settled(&db, "gained");
    assert_eq!(nearest_things(&mut db, 3, None), tasks, "after");
    assert_eq!(
        nearest_things(&mut db, 3, Some(before)),
        things,
        "read at a timestamp before the change"
    );

    db.execute_cypher("ALTER NODE TYPE Task SET IMPLIES ()")
        .expect("dropped");
    await_folded(&db);
    await_vector_settled(&db, "lost");
    assert_eq!(nearest_things(&mut db, 3, None), things, "dropped again");
}

/// A change refused by its proof leaves a vector index answering as before:
/// the members it kept for the withdrawn candidate leave the graph, and a
/// search through the index finds the label's own things alone.
#[test]
fn a_vector_index_drops_what_a_refused_change_kept() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    for ddl in [
        "CREATE NODE TYPE Thing WITH (title: STRING)",
        "ALTER LABEL Thing SET SCHEMA FLEXIBLE",
        "CREATE NODE TYPE Task WITH (title: STRING)",
        "ALTER LABEL Task SET SCHEMA FLEXIBLE",
        r#"CREATE VECTOR INDEX thing_emb ON :Thing(emb) OPTIONS {metric: "l2"}"#,
        "CREATE CONSTRAINT thing_title FOR (n:Thing) REQUIRE n.title IS UNIQUE",
        "CREATE (:Thing {title: 'same', emb: [0.0, 1.0]})",
        "CREATE (:Thing {title: 'thing', emb: [-1.0, 1.0]})",
        "CREATE (:Task {title: 'same', emb: [1.0, 0.0]})",
        "CREATE (:Task {title: 'task', emb: [1.0, 0.001]})",
    ] {
        db.execute_cypher(ddl)
            .unwrap_or_else(|e| panic!("{ddl}: {e}"));
    }
    await_vector_settled(&db, "built");
    let err = refused(&mut db, "ALTER NODE TYPE Task SET IMPLIES (Thing)");
    assert!(err.contains("same") || err.contains("unique"), "{err}");
    await_folded(&db);
    await_vector_settled(&db, "withdrawn");
    assert!(
        db.vector_index_registry().moved("Thing", "emb").is_empty(),
        "nothing kept apart"
    );
    assert_eq!(nearest_things(&mut db, 5, None), ["same", "thing"]);
}

/// The placement and folding of a change keep within the bytes a page may
/// hold. Long titles under a B-tree index, with pages far larger than the
/// bytes allow, are placed in halved pages and the change goes through,
/// every task found through the index. One task whose title alone does not
/// fit stops the work instead of growing past the bound: the change is
/// refused, and nothing it would have placed is left; with room for the
/// title, the same change goes through.
#[test]
fn an_implication_change_keeps_within_its_page_bytes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    let bytes = 64 * 1024;
    db.set_index_build_config(coordinode_query::index::IndexBuildConfig {
        reconcile_page: 512,
        fold_page: 512,
        maintenance_page_bytes: bytes,
        ..db.index_build_config()
    });
    for ddl in [
        "CREATE NODE TYPE Thing WITH (title: STRING)",
        "CREATE NODE TYPE Task WITH (title: STRING)",
        "CREATE INDEX thing_title ON :Thing(title)",
    ] {
        db.execute_cypher(ddl)
            .unwrap_or_else(|e| panic!("{ddl}: {e}"));
    }
    let long = |n: usize| format!("{n:04}{}", "x".repeat(1000));
    for n in 0..60 {
        db.execute_cypher(&format!("CREATE (:Task {{title: '{}'}})", long(n)))
            .expect("task");
    }
    db.execute_cypher("ALTER NODE TYPE Task SET IMPLIES (Thing)")
        .expect("placed in pages that fit");
    await_folded(&db);
    for n in [0, 31, 59] {
        assert_eq!(
            count(
                &mut db,
                &format!("MATCH (n:Thing {{title: '{}'}}) RETURN n", long(n))
            ),
            1,
            "task {n} through the index"
        );
    }
    db.execute_cypher("ALTER NODE TYPE Task SET IMPLIES ()")
        .expect("dropped again");
    await_folded(&db);

    // Within the length a key may take, past what one page may hold once
    // read, decoded and staged.
    let huge = "y".repeat(40_000);
    db.execute_cypher(&format!("CREATE (:Task {{title: '{huge}'}})"))
        .expect("a task whose title alone does not fit a page");
    let err = refused(&mut db, "ALTER NODE TYPE Task SET IMPLIES (Thing)");
    assert!(err.contains("memory") || err.contains("page"), "{err}");
    await_folded(&db);
    assert_eq!(
        count(&mut db, "MATCH (n:Thing) RETURN n"),
        0,
        "nothing of the refused change is left"
    );

    db.set_index_build_config(coordinode_query::index::IndexBuildConfig {
        maintenance_page_bytes: 64 * bytes,
        ..db.index_build_config()
    });
    db.execute_cypher("ALTER NODE TYPE Task SET IMPLIES (Thing)")
        .expect("with room for the title, the same change goes through");
    await_folded(&db);
    assert_eq!(count(&mut db, "MATCH (n:Thing) RETURN n"), 61);
}

/// What a placement learns of the catalog and the label lists it reuses
/// from node to node count against the same bytes as the rows it reads and
/// the postings it stages. A type implying three hundred labels with long
/// names: its one node, that node's postings and the label lists fit 128
/// KiB (they need under 96), but not with the names and closures both
/// interpretations learn (they need over 160). The change is refused,
/// leaving nothing, and goes through once the pages hold 4 MiB.
#[test]
fn an_implication_change_counts_its_catalog_in_the_page_bytes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    db.execute_cypher("CREATE NODE TYPE Task WITH (title: STRING)")
        .expect("task type");
    // Long names: the catalog the placement learns outweighs the postings
    // it stages for the same labels.
    let labels: Vec<String> = (0..300)
        .map(|n| format!("Wide{n:03}{}", "x".repeat(196)))
        .collect();
    for label in &labels {
        db.execute_cypher(&format!("CREATE NODE TYPE {label} WITH (title: STRING)"))
            .unwrap_or_else(|e| panic!("{label}: {e}"));
    }
    db.execute_cypher("CREATE (:Task {title: 't'})")
        .expect("task");
    let change = format!("ALTER NODE TYPE Task SET IMPLIES ({})", labels.join(", "));

    db.set_index_build_config(coordinode_query::index::IndexBuildConfig {
        maintenance_page_bytes: 128 * 1024,
        ..db.index_build_config()
    });
    let err = refused(&mut db, &change);
    assert!(err.contains("memory") || err.contains("page"), "{err}");
    await_folded(&db);
    assert_eq!(
        count(&mut db, &format!("MATCH (n:{}) RETURN n", labels[0])),
        0
    );

    db.set_index_build_config(coordinode_query::index::IndexBuildConfig {
        maintenance_page_bytes: 4 << 20,
        ..db.index_build_config()
    });
    db.execute_cypher(&change)
        .expect("with room for the catalog, the same change goes through");
    await_folded(&db);
    assert_eq!(
        count(&mut db, &format!("MATCH (n:{}) RETURN n", labels[299])),
        1
    );
}

/// One schema package moving a value's holder out of a label and another
/// into it: Chore stops implying Thing while Task starts, a Chore and a Task
/// holding the same title under a uniqueness on Thing. Each interpretation
/// holds the title once, so the change goes through although both holders
/// stand in the index while it is certified; then the uniqueness binds the
/// Task and no longer the Chore.
#[test]
fn a_coupled_change_moves_a_unique_value_between_holders() {
    use coordinode_embed::{Intent, LabelDeclaration, SchemaPackage};
    let mut db = things();
    db.execute_cypher("CREATE NODE TYPE Chore IMPLIES (Thing) WITH (title: STRING)")
        .expect("chore");
    db.execute_cypher("CREATE CONSTRAINT thing_title FOR (n:Thing) REQUIRE n.title IS UNIQUE")
        .expect("constraint");
    db.execute_cypher("CREATE (:Chore {title: 'v'}), (:Task {title: 'v'})")
        .expect("nodes");
    let declared = |name: &str, implies: &[&str]| {
        let current = db
            .label_schemas()
            .expect("schemas")
            .into_iter()
            .find(|s| s.name == name)
            .expect("declared");
        let revision = current.schema_revision;
        let mut schema = current;
        schema.set_implies(implies.iter().map(|l| (*l).to_string()).collect());
        LabelDeclaration {
            schema,
            intent: Intent::Change {
                expected_revision: revision,
            },
        }
    };
    let package = SchemaPackage {
        labels: vec![declared("Chore", &[]), declared("Task", &["Thing"])],
        constraints: Vec::new(),
        edge_types: Vec::new(),
    };
    let published = db.publish_schema(package).expect("publish");
    db.await_schema(&published, std::time::Duration::from_secs(30))
        .expect("await");
    let titles = |db: &mut Database, query: &str| {
        db.execute_cypher(query)
            .expect(query)
            .iter()
            .filter_map(|row| match row.get("l") {
                Some(Value::Array(labels)) => labels.first().cloned(),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        titles(
            &mut db,
            "MATCH (n:Thing {title: 'v'}) RETURN labels(n) AS l"
        ),
        [Value::String("Task".into())],
        "the Task holds the title under Thing now"
    );
    db.execute_cypher("CREATE (:Chore {title: 'v'})")
        .expect("a Chore is no Thing any more");
    let err = refused(&mut db, "CREATE (:Thing {title: 'v'})");
    assert!(err.contains('v') || err.contains("unique"), "{err}");
}

/// Setting the implication a type already has changes nothing.
#[test]
fn the_implication_in_force_is_unchanged() {
    let mut db = things();
    db.execute_cypher("ALTER NODE TYPE Task SET IMPLIES (Thing)")
        .expect("alter");
    let rows = db
        .execute_cypher("ALTER NODE TYPE Task SET IMPLIES (Thing, Thing)")
        .expect("again");
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert_eq!(implies_of(rows[0].get("implies")), ["Thing"]);
    assert_eq!(rows[0].get("changed"), Some(&Value::Bool(false)));
}

/// What the new implication may say is checked as a declaration is: the
/// implied label's properties must be declared by the type with their
/// types, nothing implies an auxiliary label or itself, the implication
/// forms no cycle, and only a declared type label is altered.
#[test]
fn the_new_implication_is_checked_as_a_declaration() {
    let mut db = things();
    db.execute_cypher("CREATE NODE TYPE Note WITH (body: STRING)")
        .expect("type without title");
    let err = refused(&mut db, "ALTER NODE TYPE Note SET IMPLIES (Thing)");
    assert!(err.contains("title"), "{err}");
    let err = refused(&mut db, "ALTER NODE TYPE Task SET IMPLIES (Activity)");
    assert!(err.contains("auxiliary"), "{err}");
    let err = refused(&mut db, "ALTER NODE TYPE Task SET IMPLIES (Task)");
    assert!(err.contains("itself"), "{err}");
    db.execute_cypher("CREATE NODE TYPE Twin WITH (title: STRING)")
        .expect("a type declaring what Thing does");
    db.execute_cypher("ALTER NODE TYPE Twin SET IMPLIES (Thing)")
        .expect("Twin implies Thing");
    let err = refused(&mut db, "ALTER NODE TYPE Thing SET IMPLIES (Twin)");
    assert!(err.contains("cycle") || err.contains("itself"), "{err}");
    let err = refused(&mut db, "ALTER NODE TYPE Activity SET IMPLIES (Thing)");
    assert!(err.contains("auxiliary"), "{err}");
    let err = refused(&mut db, "ALTER NODE TYPE Nowhere SET IMPLIES (Thing)");
    assert!(err.contains("not declared"), "{err}");
}

/// A stored carrier of the type that breaks an explicit constraint of a
/// label it would come to carry refuses the change: the node is named, the
/// implication in force stays, and the constraint keeps holding.
#[test]
fn a_carrier_breaking_a_gained_constraint_refuses_the_change() {
    let mut db = things();
    db.execute_cypher("CREATE CONSTRAINT thing_title FOR (n:Thing) REQUIRE n.title IS NOT NULL")
        .expect("constraint");
    db.execute_cypher("CREATE (:Task)")
        .expect("a task without a title");
    let err = refused(&mut db, "ALTER NODE TYPE Task SET IMPLIES (Thing)");
    assert!(err.contains("Thing") && err.contains("title"), "{err}");
    assert_eq!(count(&mut db, "MATCH (n:Thing) RETURN n"), 0);
    // The implication in force is still none, and the label stays free to
    // change once the carrier is fixed.
    db.execute_cypher("CREATE (:Task)")
        .expect("a task without a title is still a task");
    let rows = db
        .execute_cypher("ALTER NODE TYPE Task SET IMPLIES ()")
        .expect("nothing implied");
    assert_eq!(rows[0].get("changed"), Some(&Value::Bool(false)));
}

/// What publishing a change reads of the catalog and stages for it counts
/// against the statement's own memory: in a statement too small for it the
/// change is refused before anything of it is published, so the same change
/// in a statement with room succeeds afterwards, however many unrelated
/// indexes the catalog holds.
#[test]
fn an_implication_change_publishes_within_the_statement_memory() {
    let mut db = things();
    db.execute_cypher("CREATE INDEX thing_title ON :Thing(title)")
        .expect("index");
    db.execute_cypher("CREATE NODE TYPE Other WITH (title: STRING)")
        .expect("other");
    for n in 0..50 {
        db.execute_cypher(&format!(
            "CREATE INDEX other_title_{n}_{} ON :Other(title)",
            "x".repeat(60)
        ))
        .expect("unrelated index");
    }
    db.set_query_memory_limit(512).expect("limit");
    match db.execute_cypher("ALTER NODE TYPE Task SET IMPLIES (Thing)") {
        Err(coordinode_embed::DatabaseError::Budget(_)) => {}
        other => panic!("not refused by the statement's memory: {other:?}"),
    }
    db.set_query_memory_limit(256 * 1024).expect("limit");
    let rows = db
        .execute_cypher("ALTER NODE TYPE Task SET IMPLIES (Thing)")
        .expect("the change, nothing of the refused one published");
    assert_eq!(implies_of(rows[0].get("implies")), ["Thing"]);
}
