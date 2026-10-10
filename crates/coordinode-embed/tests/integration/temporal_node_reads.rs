//! Integration tests: a bare MATCH on a temporal label returns the state of
//! each node's timeline valid at the query's current instant, whatever order
//! the versions were written in.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_core::graph::node::{NodeId, NodeRecord};
use coordinode_core::graph::types::Value;
use coordinode_embed::Database;

/// One year in microseconds, the valid-time unit.
const YEAR: i64 = 365 * 24 * 3600 * 1_000_000;

/// The shard an in-memory database keeps its nodes in.
const SHARD: u16 = 1;

fn now_us() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock after epoch")
        .as_micros() as i64
}

fn open_db() -> Database {
    let mut db = Database::open_in_memory().expect("open");
    db.execute_cypher(
        "CREATE NODE TYPE Emp TEMPORAL WITH (name: STRING, valid_from: INT, valid_to: INT)",
    )
    .expect("temporal label");
    db
}

/// Create one temporal node through Cypher and return its id.
fn create(db: &mut Database, name: &str, valid_from: i64, valid_to: Option<i64>) -> u64 {
    let to = valid_to.map_or(String::new(), |t| format!(", valid_to: {t}"));
    let rows = db
        .execute_cypher(&format!(
            "CREATE (n:Emp {{name: '{name}', valid_from: {valid_from}{to}}}) RETURN n"
        ))
        .expect("create");
    match rows[0].get("n") {
        Some(Value::Int(id)) => *id as u64,
        other => panic!("node id: {other:?}"),
    }
}

/// Write one more version of node `id` straight into storage, as a restore
/// or a correction would: the write order is the test's to choose.
fn seed_version(db: &Database, id: u64, name: &str, valid_from: i64, valid_to: Option<i64>) {
    use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
    use coordinode_core::txn::write_concern::WriteConcern;
    use coordinode_modality::{LocalNodeStore, NodeStore as _};
    use coordinode_storage::engine::transaction::{CommitContext, Transaction};

    let interner = db.interner().expect("interner");
    let field = |name: &str| interner.lookup(name).expect("declared field");
    let mut record = NodeRecord::new("Emp");
    record.set(field("name"), Value::String(name.into()));
    record.set(field("valid_from"), Value::Int(valid_from));
    if let Some(to) = valid_to {
        record.set(field("valid_to"), Value::Int(to));
    }

    // Committed below the database's own clock so every later query
    // snapshot sees it; the version keys are new, nothing is shadowed.
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let mut txn = Transaction::begin(db.engine(), Some(&oracle), oracle.next());
    LocalNodeStore
        .put_temporal(&mut txn, SHARD, NodeId::from_raw(id), valid_from, &record)
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

fn names(db: &mut Database, query: &str) -> Vec<String> {
    let mut out: Vec<String> = db
        .execute_cypher(query)
        .expect("query")
        .iter()
        .map(|r| match r.get("name") {
            Some(Value::String(s)) => s.clone(),
            other => panic!("name: {other:?}"),
        })
        .collect();
    out.sort();
    out
}

/// A node with a deep history holds its versions together while the one valid
/// now is chosen, and that history is charged to the statement: a limit below
/// it stops the read with the memory refusal, never a row of a version that
/// is not current; within the limit the state valid now is returned.
#[test]
fn a_deep_single_node_history_is_read_within_the_statement_budget() {
    use coordinode_core::budget::BudgetStop;
    use coordinode_embed::db::{DatabaseError, StatementOptions};
    use coordinode_query::executor::runner::ExecutionError;
    let mut db = open_db();
    let now = now_us();
    let id = create(&mut db, "current", now - YEAR, Some(now + 10 * YEAR));
    for k in 0..2_000i64 {
        let from = now - 2 * YEAR - (k + 1) * 1_000_000;
        seed_version(&db, id, &format!("old {k:04}"), from, Some(from + 500_000));
    }
    let query = "MATCH (n:Emp) RETURN n.name AS name";
    let small = StatementOptions {
        query_memory_limit: Some(64 << 10),
        ..StatementOptions::default()
    };
    let refused = db.execute_cypher_shared_with(query, None, None, &small);
    assert!(
        matches!(
            refused,
            Err(DatabaseError::Execution(ExecutionError::Budget(
                BudgetStop::Memory { .. }
            )))
        ),
        "{refused:?}"
    );
    assert_eq!(names(&mut db, query), ["current"]);
}

/// The version valid now is returned; an older backfill written after it
/// and an open-ended future version are not, whatever the write order.
#[test]
fn a_bare_match_returns_the_state_valid_now_not_the_last_written() {
    let mut db = open_db();
    let now = now_us();
    let id = create(&mut db, "current", now - YEAR, Some(now + YEAR));
    seed_version(&db, id, "future", now + YEAR, None);
    seed_version(&db, id, "backfill", now - 3 * YEAR, Some(now - YEAR));

    assert_eq!(
        names(&mut db, "MATCH (n:Emp) RETURN n.name AS name"),
        ["current"]
    );
}

/// A node whose only versions lie in the future, or ended in the past, has
/// no state now and is not matched.
#[test]
fn a_node_without_a_state_now_is_not_matched() {
    let mut db = open_db();
    let now = now_us();
    create(&mut db, "not yet", now + YEAR, None);
    create(&mut db, "ended", now - 2 * YEAR, Some(now - YEAR));
    let gap = create(&mut db, "before the gap", now - 3 * YEAR, Some(now - YEAR));
    seed_version(&db, gap, "after the gap", now + YEAR, None);

    assert!(names(&mut db, "MATCH (n:Emp) RETURN n.name AS name").is_empty());
}

/// A deleted temporal node has a tombstone now and is not matched; the
/// version before the deletion is still history, not a current state.
#[test]
fn a_deleted_temporal_node_is_not_matched() {
    let mut db = open_db();
    let now = now_us();
    create(&mut db, "gone", now - YEAR, None);
    create(&mut db, "kept", now - YEAR, None);
    db.execute_cypher("MATCH (n:Emp {name: 'gone'}) DELETE n")
        .expect("delete");

    assert_eq!(
        names(&mut db, "MATCH (n:Emp) RETURN n.name AS name"),
        ["kept"]
    );
}

/// A SET closes the current version and opens a new one: a bare MATCH sees
/// one row, the new state, and not the closed version beside it.
#[test]
fn a_set_leaves_one_current_row() {
    let mut db = open_db();
    let now = now_us();
    create(&mut db, "before", now - YEAR, None);
    db.execute_cypher("MATCH (n:Emp {name: 'before'}) SET n.name = 'after'")
        .expect("set");

    assert_eq!(
        names(&mut db, "MATCH (n:Emp) RETURN n.name AS name"),
        ["after"]
    );
}

/// The valid-time interval is half open: the start belongs to the version,
/// the end does not. Read through `temporal_active_at` at an explicit
/// instant, which selects that instant instead of now.
#[test]
fn temporal_active_at_reads_the_timeline_at_its_instant() {
    let mut db = open_db();
    let now = now_us();
    let (t1, t2) = (now - 3 * YEAR, now - YEAR);
    let id = create(&mut db, "first", t1, Some(t2));
    seed_version(&db, id, "second", t2, None);

    let at = |db: &mut Database, t: i64| {
        names(
            db,
            &format!("MATCH (n:Emp) WHERE temporal_active_at(n, {t}) RETURN n.name AS name"),
        )
    };
    assert_eq!(at(&mut db, t1), ["first"], "the start is inside");
    assert_eq!(at(&mut db, t2 - 1), ["first"]);
    assert_eq!(at(&mut db, t2), ["second"], "the end is outside");
    assert!(at(&mut db, t1 - 1).is_empty(), "before the first version");

    // A bound parameter names the instant the same way a literal does.
    let rows = db
        .execute_cypher_with_params(
            "MATCH (n:Emp) WHERE temporal_active_at(n, $t) RETURN n.name AS name",
            [("t".to_string(), Value::Int(t2 - 1))].into(),
        )
        .expect("parameterised instant");
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert_eq!(rows[0].get("name"), Some(&Value::String("first".into())));
}

/// Fails the test unless `query` is answered through the B-tree index.
fn assert_index_plan(db: &mut Database, query: &str) {
    let plan = db.explain_cypher(query).expect("explain");
    assert!(plan.contains("IndexScan"), "not an index lookup:\n{plan}");
}

/// The lookups of a node renamed from 'old name' to 'new name': the current
/// value finds it once, the value only a closed version holds finds nothing
/// now and finds the old state at an instant inside that version.
fn assert_renamed_lookups(db: &mut Database, old_at: i64) {
    let by = |name: &str| format!("MATCH (n:Emp {{name: '{name}'}}) RETURN n.name AS name");
    assert_index_plan(db, &by("new name"));
    assert_eq!(names(db, &by("new name")), ["new name"]);
    assert!(names(db, &by("old name")).is_empty());

    let past = format!(
        "MATCH (n:Emp {{name: 'old name'}}) WHERE temporal_active_at(n, {old_at}) \
         RETURN n.name AS name"
    );
    assert_index_plan(db, &past);
    assert_eq!(names(db, &past), ["old name"]);
}

/// An index maintained by the writes answers with the state valid now: a SET
/// adds the new value's entry and keeps the old one for history.
#[test]
fn an_index_lookup_resolves_the_state_valid_now() {
    let mut db = open_db();
    let now = now_us();
    db.execute_cypher("CREATE INDEX emp_name ON :Emp(name)")
        .expect("index");
    create(&mut db, "old name", now - YEAR, None);
    db.execute_cypher("MATCH (n:Emp {name: 'old name'}) SET n.name = 'new name'")
        .expect("rename");

    assert_renamed_lookups(&mut db, now - YEAR / 2);
}

/// The stored entries of index `emp_end` on `Emp(valid_to)` holding `end`.
fn end_entries(db: &Database, end: i64) -> Vec<NodeId> {
    use coordinode_modality::{IndexStore as _, LocalIndexStore};
    use coordinode_storage::engine::transaction::Transaction;
    let index = super::helpers::index_named(db.engine(), "emp_end").expect("index emp_end");
    let mut txn = Transaction::new(
        db.engine(),
        None,
        coordinode_core::txn::timestamp::Timestamp::ZERO,
        None,
    );
    let budget = coordinode_core::budget::QueryBudget::new(
        coordinode_core::budget::DEFAULT_QUERY_MEMORY_LIMIT,
    );
    LocalIndexStore::new(db.engine())
        .scan_exact(&mut txn, &index, &[Value::Int(end)], &budget)
        .expect("scan")
        .expect("indexable")
}

/// Each version has entries of its own: moving the end of one version in
/// place moves exactly its entry, so no stale entry is left behind, and a
/// lookup by the new end finds the node through that version.
#[test]
fn moving_a_versions_end_moves_only_its_entry() {
    let mut db = open_db();
    let now = now_us();
    db.execute_cypher("CREATE INDEX emp_end ON :Emp(valid_to)")
        .expect("index");
    let (first, second) = (now + YEAR, now + 2 * YEAR);
    let id = create(&mut db, "ada", now - YEAR, Some(first));
    assert_eq!(end_entries(&db, first), [NodeId::from_raw(id)]);

    db.execute_cypher(&format!(
        "MATCH (n:Emp {{name: 'ada'}}) SET n.valid_to = {second}"
    ))
    .expect("move the end");
    assert!(end_entries(&db, first).is_empty(), "the old end is gone");
    assert_eq!(end_entries(&db, second), [NodeId::from_raw(id)]);
    assert_eq!(
        names(
            &mut db,
            &format!("MATCH (n:Emp {{valid_to: {second}}}) RETURN n.name AS name")
        ),
        ["ada"]
    );
}

/// A correction that keeps a version's `valid_from`, here its end moved in
/// place to the past, reads alike through the index and through a scan:
/// the current snapshot sees the corrected timeline (no state now, the old
/// state inside the shortened interval), and a snapshot taken before the
/// correction still sees the version as it was, current until its old end.
#[test]
fn a_correction_keeping_valid_from_reads_alike_through_index_and_scan() {
    let mut db = open_db();
    let now = now_us();
    db.execute_cypher("CREATE INDEX emp_end ON :Emp(valid_to)")
        .expect("index");
    let (end, corrected) = (now + YEAR, now - YEAR / 2);
    create(&mut db, "ada", now - YEAR, Some(end));

    let tx = db.begin_transaction();
    db.execute_in_transaction(
        tx,
        &format!("MATCH (n:Emp {{name: 'ada'}}) SET n.valid_to = {corrected}"),
        None,
    )
    .expect("correct the end");
    let corrected_at = db
        .commit_transaction(tx)
        .expect("commit")
        .commit_ts
        .as_raw();
    let before = format!(" AS OF TIMESTAMP {}", corrected_at - 1);

    // The same selection twice: `{valid_to: e}` is answered by the index,
    // `valid_to + 0 = e` by a scan of the label.
    let read = |db: &mut Database, e: i64, instant: Option<i64>, as_of: &str| {
        let at = instant.map_or(String::new(), |t| {
            format!(" WHERE temporal_active_at(n, {t})")
        });
        let indexed = format!("MATCH (n:Emp {{valid_to: {e}}}){at} RETURN n.name AS name{as_of}");
        let scan_at = instant.map_or(String::new(), |t| {
            format!(" AND temporal_active_at(n, {t})")
        });
        let scanned = format!(
            "MATCH (n:Emp) WHERE n.valid_to + 0 = {e}{scan_at} RETURN n.name AS name{as_of}"
        );
        assert_index_plan(db, &indexed);
        let plan = db.explain_cypher(&scanned).expect("explain");
        assert!(!plan.contains("IndexScan"), "not a scan:\n{plan}");
        let (by_index, by_scan) = (names(db, &indexed), names(db, &scanned));
        assert_eq!(by_index, by_scan, "{indexed}");
        by_index
    };

    let inside = now - 3 * YEAR / 4;
    assert!(
        read(&mut db, corrected, None, "").is_empty(),
        "ended before now"
    );
    assert!(
        read(&mut db, end, None, "").is_empty(),
        "the old end is gone"
    );
    assert_eq!(read(&mut db, corrected, Some(inside), ""), ["ada"]);
    assert!(read(&mut db, end, Some(inside), "").is_empty());

    assert_eq!(
        read(&mut db, end, None, &before),
        ["ada"],
        "current before it"
    );
    assert!(read(&mut db, corrected, None, &before).is_empty());
}

/// A UNIQUE index on a temporal label reserves a value for the node once
/// any of its versions held it: after a rename, another node still cannot
/// take the old value, and the node keeps its new one.
#[test]
fn a_unique_value_stays_reserved_after_a_rename() {
    let mut db = open_db();
    let now = now_us();
    db.execute_cypher("CREATE UNIQUE INDEX emp_name_u ON :Emp(name)")
        .expect("unique index");
    create(&mut db, "ada", now - YEAR, None);
    db.execute_cypher("MATCH (n:Emp {name: 'ada'}) SET n.name = 'lovelace'")
        .expect("rename");

    // Refused for the unique constraint, not for any other reason.
    let refused_as_unique =
        |result: Result<_, coordinode_embed::DatabaseError>, what: &str| match result {
            Err(coordinode_embed::DatabaseError::Execution(
                coordinode_query::executor::runner::ExecutionError::UniqueViolation { .. },
            )) => {}
            other => panic!("{what}: expected a unique violation, got {other:?}"),
        };
    refused_as_unique(
        db.execute_cypher(&format!(
            "CREATE (n:Emp {{name: 'ada', valid_from: {now}}})"
        )),
        "the old value stays reserved",
    );
    refused_as_unique(
        db.execute_cypher(&format!(
            "CREATE (n:Emp {{name: 'lovelace', valid_from: {now}}})"
        )),
        "the current value is held",
    );
    // A free value is admitted, so the refusals above came from the index.
    db.execute_cypher(&format!(
        "CREATE (n:Emp {{name: 'babbage', valid_from: {now}}})"
    ))
    .expect("a free value is admitted");
    assert_eq!(
        names(&mut db, "MATCH (n:Emp) RETURN n.name AS name"),
        ["babbage", "lovelace"],
        "the refused creates left nothing behind"
    );
}

/// An index built over existing versions holds every version's values, as
/// one the writes maintained does.
#[test]
fn an_index_built_over_versions_resolves_the_state_valid_now() {
    let mut db = open_db();
    let now = now_us();
    let id = create(&mut db, "old name", now - 3 * YEAR, Some(now - YEAR));
    seed_version(&db, id, "new name", now - YEAR, None);
    db.execute_cypher("CREATE INDEX emp_name ON :Emp(name)")
        .expect("index");

    assert_renamed_lookups(&mut db, now - 2 * YEAR);
}

/// A deleted temporal node keeps its index entries as history, and a lookup
/// of its last value finds nothing now.
#[test]
fn an_index_lookup_skips_a_deleted_temporal_node() {
    let mut db = open_db();
    let now = now_us();
    db.execute_cypher("CREATE INDEX emp_name ON :Emp(name)")
        .expect("index");
    create(&mut db, "gone", now - YEAR, None);
    db.execute_cypher("MATCH (n:Emp {name: 'gone'}) DELETE n")
        .expect("delete");

    let query = "MATCH (n:Emp {name: 'gone'}) RETURN n.name AS name";
    assert_index_plan(&mut db, query);
    assert!(names(&mut db, query).is_empty());
}

/// A pattern predicate tests the neighbour's state valid now: a neighbour
/// deleted since no longer satisfies it.
#[test]
fn a_pattern_predicate_sees_the_neighbour_state_valid_now() {
    let mut db = open_db();
    let now = now_us();
    create(&mut db, "kept", now - YEAR, None);
    create(&mut db, "gone", now - YEAR, None);
    db.execute_cypher("CREATE (:Desk {n: 'a'}), (:Desk {n: 'b'})")
        .expect("desks");
    db.execute_cypher("MATCH (a:Desk {n: 'a'}), (e:Emp {name: 'kept'}) CREATE (a)-[:SEATS]->(e)")
        .expect("edge a");
    db.execute_cypher("MATCH (b:Desk {n: 'b'}), (e:Emp {name: 'gone'}) CREATE (b)-[:SEATS]->(e)")
        .expect("edge b");
    db.execute_cypher("MATCH (e:Emp {name: 'gone'}) DELETE e")
        .expect("delete");

    let desks = |db: &mut Database, query: &str| {
        let mut out: Vec<String> = db
            .execute_cypher(query)
            .expect("query")
            .iter()
            .map(|r| match r.get("n") {
                Some(Value::String(s)) => s.clone(),
                other => panic!("n: {other:?}"),
            })
            .collect();
        out.sort();
        out
    };
    assert_eq!(
        desks(
            &mut db,
            "MATCH (d:Desk) WHERE (d)-[:SEATS]->(:Emp) RETURN d.n AS n"
        ),
        ["a"]
    );
    assert_eq!(
        desks(
            &mut db,
            "MATCH (d:Desk) WHERE NOT (d)-[:SEATS]->(:Emp) RETURN d.n AS n"
        ),
        ["b"]
    );
}

/// A temporal node revised inside an interactive transaction is read in its
/// new state by a later statement of the same transaction, every time: the
/// transaction's own closed and opened versions are read as one timeline in
/// key order, whatever order its write buffer holds them in.
#[test]
fn a_node_revised_in_a_transaction_is_found_later_in_it() {
    let read = |db: &Database, txn: u64, query: &str| -> Vec<i64> {
        db.execute_in_transaction(txn, query, None)
            .expect("statement")
            .iter()
            .map(|r| match r.get("k") {
                Some(Value::Int(k)) => *k,
                other => panic!("k: {other:?}"),
            })
            .collect()
    };
    // The buffer's order varies run to run, so one run proves little.
    for attempt in 0..50 {
        let mut db = Database::open_in_memory().expect("open");
        for ddl in [
            "CREATE NODE TYPE T TEMPORAL",
            "ALTER LABEL T SET SCHEMA FLEXIBLE",
            "CREATE (:T {name: 'p', valid_from: 1, k: 1})",
            "CREATE (:T {name: 'q', valid_from: 1, k: 1})",
            "MATCH (a:T {name: 'p'}), (b:T {name: 'q'}) CREATE (a)-[:E]->(b)",
        ] {
            db.execute_cypher(ddl).expect("setup");
        }

        let txn = db.begin_transaction();
        db.execute_in_transaction(txn, "MATCH (n:T {name: 'p'}) SET n.k = 2", None)
            .expect("revise p");
        db.execute_in_transaction(txn, "MATCH (n:T {name: 'q'}) SET n.k = 3", None)
            .expect("revise q");
        assert_eq!(
            read(&db, txn, "MATCH (n:T {name: 'p'}) RETURN n.k AS k"),
            [2],
            "scan, attempt {attempt}"
        );
        assert_eq!(
            read(&db, txn, "MATCH (n:T) WHERE n.name = 'p' RETURN n.k AS k"),
            [2],
            "filtered scan, attempt {attempt}"
        );
        assert_eq!(
            read(
                &db,
                txn,
                "MATCH (a:T {name: 'p'})-[:E]->(b:T) RETURN b.k AS k"
            ),
            [3],
            "traversal, attempt {attempt}"
        );
        db.commit_transaction(txn).expect("commit");
        assert_eq!(
            names(&mut db, "MATCH (n:T) RETURN n.name AS name"),
            ["p", "q"],
            "one current row per node after commit, attempt {attempt}"
        );
    }
}

/// A traversal lands on the target's state valid now.
#[test]
fn a_traversal_lands_on_the_current_state_of_a_temporal_target() {
    let mut db = open_db();
    let now = now_us();
    let id = create(&mut db, "old", now - 3 * YEAR, Some(now - YEAR));
    seed_version(&db, id, "new", now - YEAR, None);
    db.execute_cypher("CREATE (:Desk {n: 1})").expect("desk");
    db.execute_cypher(&format!(
        "MATCH (d:Desk), (e:Emp) WHERE id(e) = {id} CREATE (d)-[:SEATS]->(e)"
    ))
    .expect("edge");

    assert_eq!(
        names(
            &mut db,
            "MATCH (:Desk)-[:SEATS]->(e:Emp) RETURN e.name AS name"
        ),
        ["new"]
    );
}
