//! Integration tests: a bare MATCH on a temporal label returns the state of
//! each node's timeline valid at the query's current instant, whatever order
//! the versions were written in.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_core::graph::node::{NodeId, NodeRecord};
use coordinode_core::graph::types::Value;
use coordinode_embed::{Database, TransactionId};

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
    let mut record = NodeRecord::new(super::helpers::bind_label(db.engine(), "Emp"));
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
        .put_temporal(
            &mut txn,
            SHARD,
            NodeId::from_raw(id),
            valid_from,
            &record,
            coordinode_core::graph::node::VersionBounds {
                valid_to,
                deleted: false,
            },
        )
        .expect("put version");
    let wc = WriteConcern::majority();
    txn.commit(&CommitContext {
        write_concern: &wc,
        pipeline: None,
        id_gen: None,
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
            Err(DatabaseError::Budget(BudgetStop::Memory { .. }))
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

    let tx = db.begin_transaction().expect("begin");
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

/// Refused for the unique constraint, not for any other reason.
fn refused_as_unique<T: std::fmt::Debug>(
    result: Result<T, coordinode_embed::DatabaseError>,
    what: &str,
) {
    match result {
        Err(coordinode_embed::DatabaseError::UniqueViolation { .. }) => {}
        other => panic!("{what}: expected a unique violation, got {other:?}"),
    }
}

/// The names the nodes hold at valid-time instant `t`, in order.
fn held_at(db: &mut Database, t: i64) -> Vec<String> {
    names(
        db,
        &format!("MATCH (n:Emp) WHERE temporal_active_at(n, {t}) RETURN n.name AS name"),
    )
}

/// A UNIQUE index on a temporal label holds a value to one node at each
/// instant of valid time: a node whose name was another's until a rename
/// may take it from then on, but not over the time the other held it;
/// the current value stays held. Before, any version having held a value
/// reserved it for good.
#[test]
fn a_renamed_value_is_free_after_the_rename_only() {
    let mut db = open_db();
    let now = now_us();
    db.execute_cypher("CREATE UNIQUE INDEX emp_name_u ON :Emp(name)")
        .expect("unique index");
    create(&mut db, "ada", now - YEAR, None);
    db.execute_cypher("MATCH (n:Emp {name: 'ada'}) SET n.name = 'lovelace'")
        .expect("rename");
    let after_rename = now_us() + YEAR;

    refused_as_unique(
        db.execute_cypher(&format!(
            "CREATE (n:Emp {{name: 'ada', valid_from: {}}})",
            now - 2 * YEAR
        )),
        "the old value over the time it was held",
    );
    refused_as_unique(
        db.execute_cypher(&format!(
            "CREATE (n:Emp {{name: 'lovelace', valid_from: {after_rename}}})"
        )),
        "the current value is held for good",
    );
    db.execute_cypher(&format!(
        "CREATE (n:Emp {{name: 'ada', valid_from: {after_rename}}})"
    ))
    .expect("the old value after the rename");
    assert_eq!(
        names(&mut db, "MATCH (n:Emp) RETURN n.name AS name"),
        ["lovelace"],
        "only the renamed node is valid now"
    );
}

/// Half-open intervals: a value held over [a, b) may be held by another
/// node from b, and is refused to it from any earlier start. Past and
/// future intervals count alike: a value one node holds from a future time
/// is refused to another holding it from now on, and admitted to one whose
/// interval ends there.
#[test]
fn unique_values_meet_at_interval_bounds_past_and_future() {
    meet_at_interval_bounds("resolved");
}

/// The same with the index's entries derived at apply instead of carried by
/// the write.
#[test]
fn unique_values_meet_at_interval_bounds_under_derived_maintenance() {
    meet_at_interval_bounds("derived");
}

/// A unique index on `:Emp(name)` kept under `maintenance`.
fn unique_name_index(db: &mut Database, maintenance: &str) {
    db.execute_cypher(&format!(
        "CREATE UNIQUE INDEX emp_name_u ON :Emp(name) OPTIONS {{maintenance: '{maintenance}'}}"
    ))
    .expect("unique index");
}

fn meet_at_interval_bounds(maintenance: &str) {
    let mut db = open_db();
    let now = now_us();
    unique_name_index(&mut db, maintenance);
    let (a, b) = (now - 2 * YEAR, now - YEAR);
    create(&mut db, "ada", a, Some(b));
    refused_as_unique(
        db.execute_cypher(&format!(
            "CREATE (n:Emp {{name: 'ada', valid_from: {}}})",
            b - 1
        )),
        "one instant of overlap",
    );
    create(&mut db, "ada", b, None);

    let future = now + YEAR;
    create(&mut db, "grace", future, None);
    refused_as_unique(
        db.execute_cypher(&format!(
            "CREATE (n:Emp {{name: 'grace', valid_from: {now}}})"
        )),
        "a value held from a future time",
    );
    create(&mut db, "grace", now, Some(future));
    for (t, held) in [
        (a, &["ada"][..]),
        (b, &["ada"][..]),
        (now, &["ada", "grace"][..]),
        (future, &["ada", "grace"][..]),
    ] {
        assert_eq!(
            held_at(&mut db, t),
            held,
            "at {t}: the refused creates left nothing behind"
        );
    }
}

/// One transaction handing a value over, ending one node's holding at b and
/// giving it to another from b, is admitted in either order of its
/// statements: what is decided is the state it leaves, not each step. A
/// handover that would leave the two overlapping is refused whole.
#[test]
fn a_value_is_handed_over_in_one_transaction_in_either_order() {
    hand_over_in_one_transaction("resolved");
}

/// The same under derived maintenance.
#[test]
fn a_value_is_handed_over_under_derived_maintenance() {
    hand_over_in_one_transaction("derived");
}

fn hand_over_in_one_transaction(maintenance: &str) {
    let mut db = open_db();
    let now = now_us();
    unique_name_index(&mut db, maintenance);
    let (a, b) = (now - 2 * YEAR, now - YEAR);
    create(&mut db, "ada", a, None);
    let end = format!("MATCH (n:Emp) WHERE temporal_active_at(n, {a}) SET n.valid_to = {b}");
    let take = |from: i64| format!("CREATE (n:Emp {{name: 'ada', valid_from: {from}}})");

    // Taken first, ended after.
    let tx = db.begin_transaction().expect("begin");
    db.execute_in_transaction(tx, &take(b), None)
        .expect("taken from b");
    let ended = db
        .execute_in_transaction(tx, &format!("{end} RETURN n.name AS name"), None)
        .expect("ended at b");
    assert_eq!(ended.len(), 1, "the first holder is ended: {ended:?}");
    db.commit_transaction(tx)
        .expect("the handover leaves one owner at each instant");
    assert_eq!(held_at(&mut db, a), ["ada"]);
    assert_eq!(held_at(&mut db, b), ["ada"]);

    // Ended first, taken after: a third node from a later time.
    let c = now + YEAR;
    let end_second = format!("MATCH (n:Emp) WHERE temporal_active_at(n, {b}) SET n.valid_to = {c}");
    let tx = db.begin_transaction().expect("begin");
    db.execute_in_transaction(tx, &end_second, None)
        .expect("ended at c");
    db.execute_in_transaction(tx, &take(c), None)
        .expect("taken from c");
    db.commit_transaction(tx).expect("handed over again");

    // Taken from before the end: the two would overlap, and nothing lands.
    let tx = db.begin_transaction().expect("begin");
    db.execute_in_transaction(tx, &take(c - 1), None)
        .expect("staged");
    refused_as_unique(db.commit_transaction(tx), "an overlapping handover");
    for (t, held) in [(a, 1), (b, 1), (c - 1, 1), (c, 1)] {
        assert_eq!(held_at(&mut db, t).len(), held, "at {t}: one owner");
    }
}

/// Two transactions taking one value over overlapping times, each from a
/// view without the other, are not both admitted; over disjoint times they
/// are.
#[test]
fn concurrent_takers_of_one_value_are_held_to_their_times() {
    let mut db = open_db();
    let now = now_us();
    db.execute_cypher("CREATE UNIQUE INDEX emp_name_u ON :Emp(name)")
        .expect("unique index");
    let (a, b, c) = (now - 2 * YEAR, now - YEAR, now);
    let take = |from: i64, to: i64| {
        format!("CREATE (n:Emp {{name: 'ada', valid_from: {from}, valid_to: {to}}})")
    };
    let first = db.begin_transaction().expect("begin");
    let second = db.begin_transaction().expect("begin");
    db.execute_in_transaction(first, &take(a, c), None)
        .expect("first staged");
    db.execute_in_transaction(second, &take(b, c), None)
        .expect("second staged");
    db.commit_transaction(first).expect("the first is admitted");
    match db.commit_transaction(second) {
        Err(
            coordinode_embed::DatabaseError::UniqueViolation { .. }
            | coordinode_embed::DatabaseError::TransactionConflict { .. },
        ) => {}
        other => panic!("the second overlaps the first: {other:?}"),
    }
    assert_eq!(held_at(&mut db, b), ["ada"]);

    // Disjoint times: both land, one owner each.
    let d = now + YEAR;
    db.execute_cypher(&take(c, d)).expect("from c to d");
    db.execute_cypher(&take(d, d + YEAR)).expect("from d on");
    for t in [a, c, d] {
        assert_eq!(held_at(&mut db, t), ["ada"], "at {t}");
    }
}

/// A taker racing the holder's release is decided by the state the commits
/// leave, whichever commits first: taking the value while the holder still
/// has it is refused, and after it ended the holding (a deletion, then the
/// holder taking the value back over a later time) the taker is held to the
/// time the holder left free.
#[test]
fn a_taker_racing_a_release_is_decided_by_the_committed_state() {
    let mut db = open_db();
    let now = now_us();
    db.execute_cypher("CREATE UNIQUE INDEX emp_name_u ON :Emp(name)")
        .expect("unique index");
    let (a, b, c) = (now - 2 * YEAR, now - YEAR, now + YEAR);
    create(&mut db, "ada", a, None);
    let release = format!("MATCH (n:Emp) WHERE temporal_active_at(n, {a}) SET n.valid_to = {b}");
    let take = |from: i64, to: i64| {
        format!("CREATE (n:Emp {{name: 'ada', valid_from: {from}, valid_to: {to}}})")
    };

    // The taker commits first: the holder still holds [b, c).
    let holder = db.begin_transaction().expect("begin");
    let taker = db.begin_transaction().expect("begin");
    db.execute_in_transaction(holder, &release, None)
        .expect("release staged");
    db.execute_in_transaction(taker, &take(b, c), None)
        .expect("take staged");
    refused_as_unique(db.commit_transaction(taker), "taken before the release");
    db.commit_transaction(holder).expect("the release lands");

    // Released: the holder takes the value back from c, and a taker staged
    // against the released state is admitted over [b, c) only.
    let back = db.begin_transaction().expect("begin");
    let taker = db.begin_transaction().expect("begin");
    let late = db.begin_transaction().expect("begin");
    db.execute_in_transaction(
        back,
        &format!("CREATE (n:Emp {{name: 'ada', valid_from: {c}}})"),
        None,
    )
    .expect("back staged");
    db.execute_in_transaction(taker, &take(b, c), None)
        .expect("take staged");
    db.execute_in_transaction(late, &take(b, c + 1), None)
        .expect("late staged");
    db.commit_transaction(back)
        .expect("the value is taken back from c");
    // Writers of one value meet on its revision: a writer that loses that
    // race is retried whole, and the retry is decided by the state it then
    // sees.
    let retried = |db: &mut Database, txn, statement: &str| match db.commit_transaction(txn) {
        Err(coordinode_embed::DatabaseError::TransactionConflict { .. }) => {
            db.execute_cypher(statement).map(|_| ())
        }
        other => other.map(|_| ()),
    };
    retried(&mut db, taker, &take(b, c)).expect("the gap is free");
    refused_as_unique(
        retried(&mut db, late, &take(b, c + 1)),
        "one instant past the gap",
    );
    for t in [a, b, c] {
        assert_eq!(held_at(&mut db, t), ["ada"], "at {t}");
    }
}

/// A label carried by an ordinary type and by a temporal type implying it:
/// the ordinary node holds its value at every instant, so a temporal node
/// cannot take it over any interval, and an ordinary node cannot take a
/// value a temporal node holds over some interval. Values the other kind
/// does not hold are admitted to both.
#[test]
fn an_ordinary_holder_of_a_shared_label_holds_its_value_at_every_instant() {
    let mut db = Database::open_in_memory().expect("open");
    for ddl in [
        "CREATE NODE TYPE Thing WITH (name: STRING)",
        "CREATE NODE TYPE Ev TEMPORAL IMPLIES (Thing) \
         WITH (name: STRING, valid_from: INT, valid_to: INT)",
        "CREATE CONSTRAINT thing_name FOR (n:Thing) REQUIRE n.name IS UNIQUE",
    ] {
        db.execute_cypher(ddl)
            .unwrap_or_else(|e| panic!("{ddl}: {e}"));
    }
    let now = now_us();
    let (a, b) = (now - 2 * YEAR, now - YEAR);
    db.execute_cypher("CREATE (:Thing {name: 'plain'})")
        .expect("an ordinary holder");
    db.execute_cypher(&format!(
        "CREATE (:Ev {{name: 'timed', valid_from: {a}, valid_to: {b}}})"
    ))
    .expect("a temporal holder");
    refused_as_unique(
        db.execute_cypher(&format!(
            "CREATE (:Ev {{name: 'plain', valid_from: {a}, valid_to: {b}}})"
        )),
        "a temporal node over an ordinary holder's value",
    );
    refused_as_unique(
        db.execute_cypher("CREATE (:Thing {name: 'timed'})"),
        "an ordinary node over a temporal holder's value",
    );
    db.execute_cypher(&format!("CREATE (:Ev {{name: 'timed', valid_from: {b}}})"))
        .expect("another temporal node after the first one's time");
    db.execute_cypher("CREATE (:Thing {name: 'free'})")
        .expect("a free value");
}

/// The per-instant rule holds for every shape of key: a long interval
/// enclosing another's is refused; a compound key is held only by the whole
/// tuple, a list by each element, a partial index only by versions its
/// predicate admits. A version without the value holds nothing under a
/// sparse index and NULL under one that is not, one node at each instant.
#[test]
fn every_key_shape_is_held_per_instant() {
    let mut db = Database::open_in_memory().expect("open");
    db.execute_cypher(
        "CREATE NODE TYPE Seat TEMPORAL WITH \
         (room: STRING, desk: INT, open: BOOL, valid_from: INT, valid_to: INT)",
    )
    .expect("temporal label");
    // Lists are not declarable: the label takes them undeclared.
    db.execute_cypher("ALTER LABEL Seat SET SCHEMA FLEXIBLE")
        .expect("flexible");
    db.execute_cypher("CREATE CONSTRAINT seat_key FOR (n:Seat) REQUIRE (n.room, n.desk) IS UNIQUE")
        .expect("compound key");
    db.execute_cypher("CREATE UNIQUE SPARSE INDEX seat_tag ON :Seat(tags)")
        .expect("multikey index");
    db.execute_cypher("CREATE UNIQUE INDEX seat_badge ON :Seat(badge)")
        .expect("non-sparse index");
    db.execute_cypher("CREATE UNIQUE INDEX seat_open ON :Seat(desk) WHERE n.open = true")
        .expect("partial index");
    let now = now_us();
    let (a, b, c) = (now - 3 * YEAR, now - 2 * YEAR, now - YEAR);
    // Each seat gets a badge of its own unless the test leaves it out.
    let mut badges = 0;
    let mut seat = |db: &mut Database, props: &str, from: i64, to: Option<i64>| {
        let to = to.map_or(String::new(), |t| format!(", valid_to: {t}"));
        badges += 1;
        let badge = if props.contains("badge") {
            String::new()
        } else {
            format!(", badge: {badges}")
        };
        db.execute_cypher(&format!(
            "CREATE (n:Seat {{{props}{badge}, valid_from: {from}{to}}})"
        ))
    };

    seat(&mut db, "room: 'r1', desk: 1, tags: ['x', 'y']", b, Some(c)).expect("first holder");
    refused_as_unique(
        seat(&mut db, "room: 'r1', desk: 1", a, None),
        "an interval enclosing the holder's",
    );
    seat(&mut db, "room: 'r1', desk: 2", a, None).expect("another tuple of the compound key");
    refused_as_unique(
        seat(&mut db, "room: 'r2', desk: 3, tags: ['z', 'y']", a, None),
        "one element of a list held over an overlapping time",
    );
    seat(&mut db, "room: 'r2', desk: 3, tags: ['y']", c, None).expect("the element from the end");

    // Only open seats hold their desk under the partial index.
    seat(&mut db, "room: 'r3', desk: 7, open: true", a, Some(c)).expect("an open seat");
    seat(&mut db, "room: 'r4', desk: 7, open: false", a, None)
        .expect("a closed seat holds nothing");
    refused_as_unique(
        seat(&mut db, "room: 'r5', desk: 7, open: true", b, None),
        "an open seat overlapping the first",
    );
    seat(&mut db, "room: 'r5', desk: 7, open: true", c, None).expect("open from the end");

    // Versions without tags hold nothing under the sparse index, whatever
    // their times; versions without a badge hold NULL, one at an instant.
    seat(&mut db, "room: 'r6', desk: 9", a, None).expect("no tags");
    seat(&mut db, "room: 'r7', desk: 9", a, None).expect("no tags again");
    seat(&mut db, "room: 'r8', desk: 9, badge: null", a, Some(b)).expect("NULL over [a, b)");
    refused_as_unique(
        seat(&mut db, "room: 'r9', desk: 9, badge: null", b - 1, None),
        "NULL overlapping its holder",
    );
    seat(&mut db, "room: 'r9', desk: 9, badge: null", b, None).expect("NULL from b");
}

/// Set by the parent test; the child test does nothing without it.
const OWNERSHIP_CRASH_DIR: &str = "COORDINODE_TEMPORAL_UNIQUE_CRASH_DIR";
/// The maintenance the child's index is kept under.
const OWNERSHIP_CRASH_MAINTENANCE: &str = "COORDINODE_TEMPORAL_UNIQUE_CRASH_MAINTENANCE";
/// The instants the child and the parent agree on, passed through the
/// environment: a, b.
const OWNERSHIP_CRASH_TIMES: &str = "COORDINODE_TEMPORAL_UNIQUE_CRASH_TIMES";

/// Child half of `temporal_ownership_is_restored_after_a_kill`: an
/// acknowledged holding over [a, b) and a handover in one transaction, then
/// the process dies without a destructor, as under SIGKILL. Run on its own it
/// has no directory and returns.
#[test]
fn temporal_ownership_crash_child_writes_then_aborts() {
    let Some(dir) = std::env::var_os(OWNERSHIP_CRASH_DIR) else {
        return;
    };
    let maintenance = std::env::var(OWNERSHIP_CRASH_MAINTENANCE).expect("maintenance");
    let times = std::env::var(OWNERSHIP_CRASH_TIMES).expect("times");
    let (a, b) = times.split_once(',').expect("a,b");
    let (a, b): (i64, i64) = (a.parse().expect("a"), b.parse().expect("b"));
    let mut db = Database::open(std::path::Path::new(&dir)).expect("open in child");
    db.execute_cypher(
        "CREATE NODE TYPE Emp TEMPORAL WITH (name: STRING, valid_from: INT, valid_to: INT)",
    )
    .expect("temporal label");
    unique_name_index(&mut db, &maintenance);
    create(&mut db, "ada", a, None);
    // One transaction ends the first holding at b and gives the value to
    // another node from b.
    let txn = db.begin_transaction().expect("begin");
    db.execute_in_transaction(
        txn,
        &format!("MATCH (n:Emp) WHERE temporal_active_at(n, {a}) SET n.valid_to = {b}"),
        None,
    )
    .expect("end staged");
    db.execute_in_transaction(
        txn,
        &format!("CREATE (n:Emp {{name: 'ada', valid_from: {b}}})"),
        None,
    )
    .expect("take staged");
    db.commit_transaction(txn).expect("handover acknowledged");
    super::helpers::die_killed();
}

/// Ownership acknowledged before a kill is enforced after the reopen, under
/// either maintenance: the handed-over value is refused over both holders'
/// times and stays free before the first.
#[test]
fn temporal_ownership_is_restored_after_a_kill() {
    for maintenance in ["resolved", "derived"] {
        let dir = tempfile::tempdir().expect("tempdir");
        let now = now_us();
        let (a, b) = (now - 2 * YEAR, now - YEAR);
        let status = std::process::Command::new(std::env::current_exe().expect("test binary"))
            .args([
                "--exact",
                "integration::temporal_node_reads::temporal_ownership_crash_child_writes_then_aborts",
                "--nocapture",
            ])
            .env(OWNERSHIP_CRASH_DIR, dir.path())
            .env(OWNERSHIP_CRASH_MAINTENANCE, maintenance)
            .env(OWNERSHIP_CRASH_TIMES, format!("{a},{b}"))
            .status()
            .expect("run the child");
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            assert_eq!(
                status.signal(),
                Some(super::helpers::CHILD_KILL_SIGNAL),
                "{maintenance}: the child must be killed after its writes, got {status:?}"
            );
        }
        #[cfg(not(unix))]
        assert!(!status.success(), "the child must die, got {status:?}");

        let mut db = Database::open(dir.path()).expect("reopen after the kill");
        for t in [a, b] {
            assert_eq!(held_at(&mut db, t), ["ada"], "{maintenance} at {t}");
        }
        for (from, what) in [(a, "over the first holder"), (b, "over the second")] {
            refused_as_unique(
                db.execute_cypher(&format!(
                    "CREATE (n:Emp {{name: 'ada', valid_from: {from}}})"
                )),
                &format!("{maintenance}: {what}"),
            );
        }
        db.execute_cypher(&format!(
            "CREATE (n:Emp {{name: 'ada', valid_from: {}, valid_to: {a}}})",
            a - YEAR
        ))
        .expect("free before the first holder");
    }
}

/// Ownership over valid time survives a reopen: after it, an overlap with a
/// holder admitted before is still refused and a value meeting it at a
/// bound is still admitted, under either maintenance.
#[test]
fn temporal_ownership_is_kept_across_a_reopen() {
    for maintenance in ["resolved", "derived"] {
        let dir = tempfile::tempdir().expect("tempdir");
        let now = now_us();
        let (a, b) = (now - 2 * YEAR, now - YEAR);
        {
            let mut db = Database::open(dir.path()).expect("open");
            db.execute_cypher(
                "CREATE NODE TYPE Emp TEMPORAL WITH (name: STRING, valid_from: INT, valid_to: INT)",
            )
            .expect("temporal label");
            unique_name_index(&mut db, maintenance);
            create(&mut db, "ada", a, Some(b));
        }
        let mut db = Database::open(dir.path()).expect("reopen");
        refused_as_unique(
            db.execute_cypher(&format!(
                "CREATE (n:Emp {{name: 'ada', valid_from: {}}})",
                b - 1
            )),
            &format!("{maintenance}: an overlap after the reopen"),
        );
        create(&mut db, "ada", b, None);
        assert_eq!(held_at(&mut db, a), ["ada"], "{maintenance}");
        assert_eq!(held_at(&mut db, b), ["ada"], "{maintenance}");
    }
}

/// A correction stretching a version over another node's interval is
/// refused and leaves both as they were; one ending it earlier is admitted
/// and frees the value from then on. A deleted node's past holding stays,
/// its holding from the deletion on does not.
#[test]
fn a_correction_or_deletion_moves_where_a_value_is_held() {
    let mut db = open_db();
    let now = now_us();
    db.execute_cypher("CREATE UNIQUE INDEX emp_name_u ON :Emp(name)")
        .expect("unique index");
    let (a, b, c) = (now - 3 * YEAR, now - 2 * YEAR, now - YEAR);
    create(&mut db, "ada", a, Some(b));
    create(&mut db, "ada", b, Some(c));
    refused_as_unique(
        db.execute_cypher(&format!(
            "MATCH (n:Emp) WHERE temporal_active_at(n, {a}) SET n.valid_to = {c}"
        )),
        "stretched over the other's interval",
    );
    for t in [a, b] {
        assert_eq!(
            held_at(&mut db, t),
            ["ada"],
            "at {t}: the refusal changed nothing"
        );
    }

    create(&mut db, "babbage", c, None);
    db.execute_cypher("MATCH (n:Emp {name: 'babbage'}) DELETE n")
        .expect("delete");
    refused_as_unique(
        db.execute_cypher(&format!(
            "CREATE (n:Emp {{name: 'babbage', valid_from: {c}}})"
        )),
        "the deleted node held it before its deletion",
    );
    db.execute_cypher(&format!(
        "CREATE (n:Emp {{name: 'babbage', valid_from: {}}})",
        now_us() + YEAR
    ))
    .expect("free after the deletion");
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
    let read = |db: &Database, txn: TransactionId, query: &str| -> Vec<i64> {
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

        let txn = db.begin_transaction().expect("begin");
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
