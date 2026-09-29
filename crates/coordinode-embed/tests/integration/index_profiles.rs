//! Maintenance profiles of B-tree indexes, end to end.
//!
//! A RESOLVED index journals its entries; a DERIVED index journals the work
//! and its exact inputs, and every apply derives the entries. Both profiles
//! must leave the same entries behind and answer the same queries.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_core::graph::types::Value;
use coordinode_core::index::encoding::{index_prefix, unique_index_prefix};
use coordinode_core::txn::proposal::DerivedIndexWork;
use coordinode_embed::Database;
use coordinode_storage::Guard as _;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::oplog::entry::OplogOp;

fn open_db() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open db");
    (db, dir)
}

/// The key prefixes of the index `name`: its unique and non-unique entries.
fn prefixes(name: &str) -> [Vec<u8>; 2] {
    [index_prefix(name), unique_index_prefix(name)]
}

/// Every entry of the index `name`, keys and values, in key order.
fn entries(db: &Database, name: &str) -> Vec<(Vec<u8>, Vec<u8>)> {
    prefixes(name)
        .iter()
        .flat_map(|prefix| {
            db.engine()
                .prefix_scan(Partition::Idx, prefix)
                .expect("scan")
                .map(|item| {
                    let (k, v) = item.into_inner().expect("kv");
                    (k.to_vec(), v.to_vec())
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn count(db: &mut Database, query: &str, params: &[(&str, Value)]) -> i64 {
    let params = params
        .iter()
        .map(|(k, v)| ((*k).to_string(), v.clone()))
        .collect();
    let rows = db
        .execute_cypher_with_params(query, params)
        .expect("count query");
    match rows[0].values().next() {
        Some(Value::Int(n)) => *n,
        other => panic!("expected a count, got {other:?}"),
    }
}

/// The highest journal index written so far.
fn journal_tip(db: &Database) -> u64 {
    db.engine()
        .oplog_read_since(0)
        .expect("read")
        .expect("journal")
        .last()
        .map_or(0, |e| e.index)
}

/// The operations journalled after `from`, each unit frame expanded.
fn journalled_since(db: &Database, from: u64) -> Vec<OplogOp> {
    db.engine()
        .oplog_read_since(from + 1)
        .expect("read")
        .expect("journal")
        .into_iter()
        .flat_map(|e| {
            coordinode_storage::oplog::convert::expand_units(&e.ops)
                .expect("expand")
                .into_owned()
        })
        .collect()
}

/// Journalled inserts of an entry of the index `name`.
fn entry_inserts(ops: &[OplogOp], name: &str) -> usize {
    let prefixes = prefixes(name);
    ops.iter()
        .filter(|op| {
            matches!(op, OplogOp::Insert { key, .. }
                if prefixes.iter().any(|p| key.starts_with(p)))
        })
        .count()
}

/// Journalled DERIVED work for the index `name`.
fn derived_work(ops: &[OplogOp], name: &str) -> usize {
    ops.iter()
        .filter(|op| match op {
            OplogOp::Derive { work } => {
                let work: DerivedIndexWork = rmp_serde::from_slice(work).expect("decode work");
                work.binding.interpretation.name == name
            }
            _ => false,
        })
        .count()
}

fn text(row: &coordinode_query::executor::Row, column: &str) -> String {
    match row.get(column) {
        Some(Value::String(s)) => s.clone(),
        other => panic!("column {column}: expected a string, got {other:?}"),
    }
}

fn int(row: &coordinode_query::executor::Row, column: &str) -> i64 {
    match row.get(column) {
        Some(Value::Int(n)) => *n,
        other => panic!("column {column}: expected an int, got {other:?}"),
    }
}

/// The workload the differential test runs under each profile: unique,
/// multikey and partial indexes, with entries entered, moved, left and
/// removed with their node.
const WORKLOAD: &[&str] = &[
    "CREATE (:U {email: 'a@x', tags: ['red', 'blue'], status: 'active'})",
    "CREATE (:U {email: 'b@x', tags: 'red', status: 'idle'})",
    "CREATE (:U {email: 'c@x', tags: ['green'], status: 'active'})",
    "MATCH (u:U {email: 'a@x'}) SET u.tags = ['blue', 'green']",
    "MATCH (u:U {email: 'b@x'}) SET u.email = 'b2@x', u.status = 'active'",
    "MATCH (u:U {email: 'c@x'}) SET u.status = 'idle'",
    "MATCH (u:U {email: 'b2@x'}) REMOVE u.tags",
    "MATCH (u:U {email: 'c@x'}) DETACH DELETE u",
    "CREATE (:U {email: 'c@x', tags: ['red', 'red'], status: 'active'})",
];

fn run_workload(profile: &str) -> (Database, tempfile::TempDir) {
    let (mut db, dir) = open_db();
    for ddl in [
        "CREATE UNIQUE INDEX u_email ON :U(email)",
        "CREATE INDEX u_tags ON :U(tags)",
        "CREATE INDEX u_active ON :U(email) WHERE n.status = 'active'",
    ] {
        db.execute_cypher(&format!("{ddl} OPTIONS {{maintenance: '{profile}'}}"))
            .expect("index");
    }
    for statement in WORKLOAD {
        db.execute_cypher(statement).expect(statement);
    }
    (db, dir)
}

/// The same writes under DERIVED leave byte-identical entries to RESOLVED
/// and answer lookups the same way: the apply derives what the writer would
/// have written.
#[test]
fn derived_and_resolved_indexes_hold_the_same_entries() {
    let (mut resolved, _r) = run_workload("resolved");
    let (mut derived, _d) = run_workload("derived");

    for name in ["u_email", "u_tags", "u_active"] {
        let expected = entries(&resolved, name);
        assert!(!expected.is_empty(), "{name} holds entries");
        assert_eq!(entries(&derived, name), expected, "entries of {name}");
    }

    let lookups: &[(&str, &str, &str)] = &[
        (
            "MATCH (u:U) WHERE u.email = $v RETURN count(u)",
            "v",
            "b2@x",
        ),
        ("MATCH (u:U) WHERE u.email = $v RETURN count(u)", "v", "b@x"),
        ("MATCH (u:U) WHERE u.tags = $v RETURN count(u)", "v", "red"),
        (
            "MATCH (u:U) WHERE u.tags = $v RETURN count(u)",
            "v",
            "green",
        ),
        (
            "MATCH (u:U) WHERE u.email = $v AND u.status = 'active' RETURN count(u)",
            "v",
            "c@x",
        ),
    ];
    for (query, param, value) in lookups {
        let params = [(*param, Value::String((*value).into()))];
        assert_eq!(
            count(&mut derived, query, &params),
            count(&mut resolved, query, &params),
            "{query} with {value}"
        );
    }
}

/// A DERIVED index journals its work, never its entries; a RESOLVED one
/// journals the entries and no work.
#[test]
fn a_derived_index_journals_its_work_and_not_its_entries() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE INDEX d_name ON :U(name) OPTIONS {maintenance: 'derived'}")
        .expect("derived index");
    db.execute_cypher("CREATE INDEX r_name ON :U(name) OPTIONS {maintenance: 'resolved'}")
        .expect("resolved index");
    let tip = journal_tip(&db);

    db.execute_cypher("CREATE (:U {name: 'ann'})")
        .expect("create");
    db.execute_cypher("MATCH (u:U) SET u.name = 'bea'")
        .expect("set");

    let ops = journalled_since(&db, tip);
    assert_eq!(
        entry_inserts(&ops, "d_name"),
        0,
        "no derived entry is journalled"
    );
    assert_eq!(derived_work(&ops, "d_name"), 2, "one work per write");
    assert_eq!(
        entry_inserts(&ops, "r_name"),
        2,
        "resolved entries are journalled"
    );
    assert_eq!(derived_work(&ops, "r_name"), 0);
    assert_eq!(entries(&db, "d_name").len(), 1);
    assert_eq!(entries(&db, "d_name"), {
        let mut r = entries(&db, "r_name");
        // Same membership under another index name: compare the suffixes.
        let (d, rp) = (index_prefix("d_name"), index_prefix("r_name"));
        for (k, _) in &mut r {
            *k = [d.as_slice(), &k[rp.len()..]].concat();
        }
        r
    });
}

/// Entries derived at apply are what a restart finds.
#[test]
fn derived_entries_survive_a_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");
    let before = {
        let mut db = Database::open(dir.path()).expect("open");
        db.execute_cypher("CREATE INDEX d_name ON :U(name) OPTIONS {maintenance: 'derived'}")
            .expect("index");
        db.execute_cypher("CREATE (:U {name: 'ann'})").expect("ann");
        db.execute_cypher("CREATE (:U {name: 'bea'})").expect("bea");
        entries(&db, "d_name")
    };
    assert_eq!(before.len(), 2);

    let mut db = Database::open(dir.path()).expect("reopen");
    assert_eq!(entries(&db, "d_name"), before);
    assert_eq!(
        count(
            &mut db,
            "MATCH (u:U) WHERE u.name = $n RETURN count(u)",
            &[("n", Value::String("bea".into()))]
        ),
        1
    );
    db.execute_cypher("CREATE (:U {name: 'cid'})")
        .expect("the reopened index is maintained");
    assert_eq!(entries(&db, "d_name").len(), 3);
}

/// Set by the parent test; the child test does nothing without it.
const CRASH_CHILD_DIR: &str = "COORDINODE_DERIVED_CRASH_CHILD_DIR";

/// Child half of `derived_entries_are_derived_again_after_a_kill`: acknowledged
/// writes under a DERIVED index, then the process dies without a destructor,
/// as under SIGKILL. Run on its own it has no directory and returns.
#[test]
fn derived_crash_child_writes_then_aborts() {
    let Some(dir) = std::env::var_os(CRASH_CHILD_DIR) else {
        return;
    };
    let mut db = Database::open(std::path::Path::new(&dir)).expect("open in child");
    db.execute_cypher("CREATE UNIQUE INDEX d_email ON :U(email) OPTIONS {maintenance: 'derived'}")
        .expect("index in child");
    db.execute_cypher("CREATE (:U {email: 'a@x'})")
        .expect("a in child");
    db.execute_cypher("CREATE (:U {email: 'b@x'})")
        .expect("b in child");
    db.execute_cypher("MATCH (u:U {email: 'a@x'}) SET u.email = 'c@x'")
        .expect("move in child");
    std::process::abort();
}

/// Entries a DERIVED index had not yet persisted when the process was killed
/// are derived again from the journalled work on open: the journal holds no
/// entry, so replay must interpret the work the same way the apply did.
#[test]
fn derived_entries_are_derived_again_after_a_kill() {
    let dir = tempfile::tempdir().expect("tempdir");
    let status = std::process::Command::new(std::env::current_exe().expect("test binary"))
        .args([
            "--exact",
            "integration::index_profiles::derived_crash_child_writes_then_aborts",
            "--nocapture",
        ])
        .env(CRASH_CHILD_DIR, dir.path())
        .status()
        .expect("run the child");
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        assert_eq!(
            status.signal(),
            Some(6),
            "the child must die by abort after its writes, got {status:?}"
        );
    }
    #[cfg(not(unix))]
    assert!(!status.success(), "the child must die, got {status:?}");

    let mut db = Database::open(dir.path()).expect("reopen after the kill");
    assert_eq!(entries(&db, "d_email").len(), 2);
    for (email, expected) in [("a@x", 0), ("b@x", 1), ("c@x", 1)] {
        assert_eq!(
            count(
                &mut db,
                "MATCH (u:U) WHERE u.email = $e RETURN count(u)",
                &[("e", Value::String(email.into()))]
            ),
            expected,
            "{email}"
        );
    }
    db.execute_cypher("CREATE (:U {email: 'c@x'})")
        .expect_err("the recovered claim is enforced");
    db.execute_cypher("CREATE (:U {email: 'a@x'})")
        .expect("the released value is free");
}

/// A DERIVED unique index refuses a second holder of a value, and a value
/// its holder leaves is free for the next node.
#[test]
fn a_derived_unique_index_enforces_and_releases_its_values() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE UNIQUE INDEX u_email ON :U(email) OPTIONS {maintenance: 'derived'}")
        .expect("index");
    db.execute_cypher("CREATE (:U {email: 'a@x'})")
        .expect("first holder");
    db.execute_cypher("CREATE (:U {email: 'a@x'})")
        .expect_err("a second holder is refused");

    db.execute_cypher("MATCH (u:U {email: 'a@x'}) SET u.email = 'b@x'")
        .expect("the holder moves");
    db.execute_cypher("CREATE (:U {email: 'a@x'})")
        .expect("the released value is free");
    db.execute_cypher("CREATE (:U {email: 'b@x'})")
        .expect_err("the moved value is held");
    assert_eq!(entries(&db, "u_email").len(), 2);
}

/// Inside an interactive transaction a DERIVED index sees the transaction's
/// own writes, and a rollback leaves no entry behind.
#[test]
fn a_derived_index_sees_own_writes_and_rolls_back() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE INDEX d_name ON :U(name) OPTIONS {maintenance: 'derived'}")
        .expect("index");

    let txn = db.begin_transaction();
    db.execute_in_transaction(txn, "CREATE (:U {name: 'ann'})", None)
        .expect("create");
    let rows = db
        .execute_in_transaction(
            txn,
            "MATCH (u:U) WHERE u.name = 'ann' RETURN count(u) AS n",
            None,
        )
        .expect("read own write");
    assert_eq!(int(&rows[0], "n"), 1);
    db.rollback_transaction(txn).expect("rollback");

    assert!(entries(&db, "d_name").is_empty());
    assert_eq!(
        count(
            &mut db,
            "MATCH (u:U) WHERE u.name = $n RETURN count(u)",
            &[("n", Value::String("ann".into()))]
        ),
        0
    );
}

/// A transition moves an index between profiles under a new epoch without
/// rewriting its entries, and writes after it follow the new profile.
#[test]
fn alter_index_moves_between_profiles_without_a_rebuild() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE INDEX i_name ON :U(name)")
        .expect("index");
    db.execute_cypher("CREATE (:U {name: 'ann'})").expect("ann");
    let before = entries(&db, "i_name");

    let rows = db
        .execute_cypher("ALTER INDEX i_name SET MAINTENANCE DERIVED")
        .expect("transition");
    assert_eq!(text(&rows[0], "maintenance"), "DERIVED");
    assert_eq!(text(&rows[0], "maintenance_source"), "OVERRIDE");
    assert_eq!(int(&rows[0], "previous_epoch"), 1);
    assert_eq!(int(&rows[0], "maintenance_epoch"), 2);
    assert_eq!(entries(&db, "i_name"), before, "no entry is rewritten");

    let tip = journal_tip(&db);
    db.execute_cypher("CREATE (:U {name: 'bea'})").expect("bea");
    let ops = journalled_since(&db, tip);
    assert_eq!(derived_work(&ops, "i_name"), 1);
    assert_eq!(entry_inserts(&ops, "i_name"), 0);

    let rows = db
        .execute_cypher("ALTER INDEX i_name SET MAINTENANCE INHERIT")
        .expect("back to the namespace default");
    assert_eq!(text(&rows[0], "maintenance"), "RESOLVED");
    assert!(text(&rows[0], "maintenance_source").starts_with("NAMESPACE@"));
    assert_eq!(int(&rows[0], "maintenance_epoch"), 3);

    let tip = journal_tip(&db);
    db.execute_cypher("CREATE (:U {name: 'cid'})").expect("cid");
    let ops = journalled_since(&db, tip);
    assert_eq!(derived_work(&ops, "i_name"), 0);
    assert_eq!(entry_inserts(&ops, "i_name"), 1);
    assert_eq!(
        count(
            &mut db,
            "MATCH (u:U) WHERE u.name = $n RETURN count(u)",
            &[("n", Value::String("bea".into()))]
        ),
        1
    );
}

/// Effects staged under one epoch are refused at commit once the index has
/// moved on: the writer retries under the new binding.
#[test]
fn a_transaction_staged_before_a_transition_is_refused_at_commit() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE INDEX d_name ON :U(name) OPTIONS {maintenance: 'derived'}")
        .expect("index");

    let txn = db.begin_transaction();
    db.execute_in_transaction(txn, "CREATE (:U {name: 'ann'})", None)
        .expect("stage under epoch 1");
    db.execute_cypher("ALTER INDEX d_name SET MAINTENANCE RESOLVED")
        .expect("transition");
    let err = db
        .commit_transaction(txn)
        .expect_err("the staged binding is stale");
    assert!(err.to_string().contains("version mismatch"), "{err}");

    assert!(entries(&db, "d_name").is_empty());
    db.execute_cypher("CREATE (:U {name: 'ann'})")
        .expect("the retry commits under the new binding");
    assert_eq!(entries(&db, "d_name").len(), 1);
}

/// The namespace default binds indexes created after it changes; an index
/// created before keeps its own binding.
#[test]
fn the_namespace_default_binds_only_new_indexes() {
    let (mut db, _dir) = open_db();
    let rows = db
        .execute_cypher("CREATE INDEX before_idx ON :U(a)")
        .expect("before");
    assert_eq!(text(&rows[0], "maintenance"), "RESOLVED");
    let old_source = text(&rows[0], "maintenance_source");
    assert!(old_source.starts_with("NAMESPACE@"));

    let rows = db
        .execute_cypher("ALTER NAMESPACE SET INDEX MAINTENANCE DERIVED")
        .expect("namespace default");
    assert_eq!(text(&rows[0], "index_maintenance_default"), "DERIVED");
    let revision = int(&rows[0], "revision");

    let rows = db
        .execute_cypher("CREATE INDEX after_idx ON :U(b)")
        .expect("after");
    assert_eq!(text(&rows[0], "maintenance"), "DERIVED");
    assert_eq!(
        text(&rows[0], "maintenance_source"),
        format!("NAMESPACE@{revision}")
    );
    assert_ne!(text(&rows[0], "maintenance_source"), old_source);

    let tip = journal_tip(&db);
    db.execute_cypher("CREATE (:U {a: 1, b: 2})")
        .expect("write");
    let ops = journalled_since(&db, tip);
    assert_eq!(entry_inserts(&ops, "before_idx"), 1);
    assert_eq!(derived_work(&ops, "before_idx"), 0);
    assert_eq!(entry_inserts(&ops, "after_idx"), 0);
    assert_eq!(derived_work(&ops, "after_idx"), 1);

    let rows = db
        .execute_cypher("CREATE INDEX pinned_idx ON :U(c) OPTIONS {maintenance: 'resolved'}")
        .expect("override");
    assert_eq!(text(&rows[0], "maintenance"), "RESOLVED");
    assert_eq!(text(&rows[0], "maintenance_source"), "OVERRIDE");
}

/// An unknown option, an unknown profile, and a transition of an index that
/// does not exist are refused.
#[test]
fn unknown_maintenance_settings_are_refused() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE INDEX x ON :U(a) OPTIONS {maintenance: 'lazy'}")
        .expect_err("unknown profile");
    db.execute_cypher("CREATE INDEX x ON :U(a) OPTIONS {colour: 'red'}")
        .expect_err("unknown option");
    db.execute_cypher("ALTER INDEX x SET MAINTENANCE LAZY")
        .expect_err("unknown transition target");
    db.execute_cypher("ALTER INDEX missing SET MAINTENANCE DERIVED")
        .expect_err("no such index");
    db.execute_cypher("ALTER NAMESPACE SET INDEX MAINTENANCE INHERIT")
        .expect_err("a namespace has nothing to inherit from");
}
