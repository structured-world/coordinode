//! The built-in label lookup: one posting per node and label it holds,
//! maintained with every write and filled from the stored nodes when a store
//! has none, so a label is enumerated from its own members.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use core::ops::ControlFlow;

use coordinode_core::graph::types::Value;
use coordinode_embed::Database;
use coordinode_modality::index::{IndexStore, LabelScope, LocalIndexStore};
use coordinode_modality::{IndexState, StoreError};
use coordinode_storage::engine::transaction::Transaction;

/// The shard an embedded database keeps its nodes in.
const SHARD: u16 = 1;

fn open_db() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open db");
    (db, dir)
}

/// The members of `label` the lookup holds, in NodeId order: empty when the
/// name has no dictionary id, which no posting can carry.
fn members(db: &Database, label: &str) -> Vec<i64> {
    let lookup = LocalIndexStore::new(db.engine())
        .label_lookup()
        .expect("read")
        .expect("every store has its label lookup");
    let Some(id) = db.interner().expect("dictionary").lookup(label) else {
        return Vec::new();
    };
    let txn = Transaction::new(
        db.engine(),
        None,
        coordinode_core::txn::timestamp::Timestamp::ZERO,
        None,
    );
    let budget = coordinode_core::budget::QueryBudget::new(
        coordinode_core::budget::DEFAULT_QUERY_MEMORY_LIMIT,
    );
    let mut out = Vec::new();
    LocalIndexStore::new(db.engine())
        .label_members(
            &txn,
            LabelScope {
                lookup: &lookup,
                shard: SHARD,
                label: id,
            },
            None,
            &budget,
            |node| {
                out.push(i64::try_from(node.as_raw()).expect("id"));
                Ok::<_, StoreError>(ControlFlow::Continue(()))
            },
        )
        .expect("walk the label");
    out
}

/// The ids of the nodes holding `label`, read from the records alone.
fn holders(db: &mut Database, label: &str) -> Vec<i64> {
    let mut ids: Vec<i64> = db
        .execute_cypher("MATCH (n) RETURN id(n) AS id, labels(n) AS labels")
        .expect("scan")
        .iter()
        .filter(|row| match row.get("labels") {
            Some(Value::Array(labels)) => labels
                .iter()
                .any(|l| matches!(l, Value::String(s) if s == label)),
            other => panic!("expected labels, got {other:?}"),
        })
        .map(|row| match row.get("id") {
            Some(Value::Int(id)) => *id,
            other => panic!("expected an id, got {other:?}"),
        })
        .collect();
    ids.sort_unstable();
    ids
}

fn id_of(db: &mut Database, name: &str) -> i64 {
    let rows = db
        .execute_cypher(&format!(
            "MATCH (n) WHERE n.name + '' = '{name}' RETURN id(n) AS id"
        ))
        .expect("scan");
    assert_eq!(rows.len(), 1, "one node is named {name}");
    match rows[0].get("id") {
        Some(Value::Int(id)) => *id,
        other => panic!("expected an id, got {other:?}"),
    }
}

/// A new store has its lookup ready before its first statement, and every
/// label change of every node moves exactly that node's postings: CREATE
/// with several labels, SET and REMOVE of a label, MERGE creating a node,
/// DELETE. Each label's members always equal the nodes the records give it.
#[test]
fn postings_follow_every_label_change() {
    let (mut db, _dir) = open_db();
    let lookup = LocalIndexStore::new(db.engine())
        .label_lookup()
        .expect("read")
        .expect("provisioned at open");
    assert_eq!(lookup.state, IndexState::Ready, "nothing stored to fill");

    db.execute_cypher("CREATE (:User {name: 'a'}), (:User:Admin {name: 'b'}), (:Post {name: 'p'})")
        .expect("create");
    db.execute_cypher("MERGE (:User {name: 'c'})")
        .expect("merge");
    db.execute_cypher("MATCH (n {name: 'p'}) SET n:Draft")
        .expect("label added");
    db.execute_cypher("MATCH (n {name: 'b'}) REMOVE n:Admin")
        .expect("label removed");
    db.execute_cypher("MATCH (n {name: 'a'}) DELETE n")
        .expect("delete");
    for label in ["User", "Admin", "Post", "Draft", "Nobody"] {
        assert_eq!(members(&db, label), holders(&mut db, label), "{label}");
    }
    let b = id_of(&mut db, "b");
    let c = id_of(&mut db, "c");
    let mut users = vec![b, c];
    users.sort_unstable();
    assert_eq!(members(&db, "User"), users);
    assert!(members(&db, "Admin").is_empty(), "b left Admin");
}

/// A property edit keeps the node's labels and writes no posting; a
/// statement that rolls back leaves no posting behind.
#[test]
fn property_edits_and_rollbacks_leave_postings_alone() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (:User {name: 'a', n: 1})")
        .expect("create");
    let a = id_of(&mut db, "a");
    let engine = db.engine();
    let before = engine.current_seqno();
    db.execute_cypher("MATCH (n:User) SET n.n = 2")
        .expect("edit");
    assert_eq!(members(&db, "User"), vec![a]);
    let lookup = LocalIndexStore::new(db.engine())
        .label_lookup()
        .expect("read")
        .expect("lookup");
    let user = db
        .interner()
        .expect("dictionary")
        .lookup("User")
        .expect("id");
    let posting = coordinode_core::index::encoding::encode_label_posting(
        lookup.generation,
        SHARD,
        user,
        u64::try_from(a).expect("id"),
    );
    assert!(
        !db.engine()
            .written_since_snapshot(
                coordinode_storage::engine::partition::Partition::Idx,
                &posting,
                before,
            )
            .expect("read"),
        "the edit wrote no posting"
    );

    let tx = db.begin_transaction();
    db.execute_in_transaction(tx, "CREATE (:Ghost {name: 'g'})", None)
        .expect("staged");
    db.rollback_transaction(tx).expect("rollback");
    assert!(
        members(&db, "Ghost").is_empty(),
        "a rolled back node has none"
    );
}

/// A temporal node is a member of every label any of its versions holds:
/// its history keeps them, so its postings outlive a later version.
#[test]
fn a_temporal_node_is_a_member_of_the_labels_its_versions_hold() {
    let (mut db, _dir) = open_db();
    db.execute_cypher(
        "CREATE NODE TYPE Emp TEMPORAL WITH (name: STRING, valid_from: INT, valid_to: INT)",
    )
    .expect("temporal label");
    let since = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_micros() as i64
        - 1_000_000;
    db.execute_cypher(&format!("CREATE (:Emp {{name: 'e', valid_from: {since}}})"))
        .expect("create");
    db.execute_cypher("MATCH (n:Emp {name: 'e'}) SET n.name = 'f'")
        .expect("new version");
    assert_eq!(
        members(&db, "Emp").len(),
        1,
        "one node, however many versions"
    );
}

/// A store whose label lookup is missing (written before it existed) has it
/// built from its stored nodes at the next open: the members end up exactly
/// the nodes the records give each label, ordinary and temporal.
#[test]
fn a_store_without_a_lookup_has_it_built_from_its_nodes() {
    let dir = tempfile::tempdir().expect("tempdir");
    {
        let mut db = Database::open(dir.path()).expect("open");
        db.execute_cypher("UNWIND range(1, 300) AS i CREATE (:Item {name: 'i' + i})")
            .expect("items");
        db.execute_cypher("CREATE (:Item:Hot {name: 'h'}), (:Other {name: 'o'})")
            .expect("others");
        forget_label_lookup(&db);
    }
    let mut db = Database::open(dir.path()).expect("reopen");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    loop {
        let state = LocalIndexStore::new(db.engine())
            .label_lookup()
            .expect("read")
            .expect("provisioned again")
            .state
            .clone();
        if state == IndexState::Ready {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "never built: {state:?}"
        );
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    for label in ["Item", "Hot", "Other"] {
        assert_eq!(members(&db, label), holders(&mut db, label), "{label}");
    }
    assert_eq!(members(&db, "Item").len(), 301);
}

// ── Scans ───────────────────────────────────────────────────────────────

/// Names returned by a query, sorted.
fn names(db: &mut Database, query: &str) -> Vec<String> {
    let mut out: Vec<String> = db
        .execute_cypher(query)
        .unwrap_or_else(|e| panic!("{query}: {e}"))
        .iter()
        .map(|row| match row.get("name") {
            Some(Value::String(s)) => s.clone(),
            other => panic!("expected a name, got {other:?}"),
        })
        .collect();
    out.sort();
    out
}

/// A store whose shard is mostly one node far larger than a statement may
/// hold, beside a few small labelled ones.
fn store_with_a_large_unrelated_node() -> (Database, tempfile::TempDir) {
    let (mut db, dir) = open_db();
    db.execute_cypher(
        "CREATE (:Small {name: 's1'}), (:Small:Tagged {name: 's2'}), (:Tagged {name: 't'})",
    )
    .expect("small nodes");
    let blob = "x".repeat(4 << 20);
    db.execute_cypher(&format!("CREATE (:Big {{name: 'big', payload: '{blob}'}})"))
        .expect("large node");
    db.set_query_memory_limit(1 << 20).expect("limit");
    (db, dir)
}

/// A label scan reads the label's members and nothing else: the large node
/// of another label is never read, so a statement whose memory limit it
/// would break answers, an empty label too. The full scan the lookup
/// replaced read every node of the shard and was refused.
#[test]
fn a_label_scan_reads_none_of_the_other_labels_nodes() {
    let (mut db, _dir) = store_with_a_large_unrelated_node();
    assert_eq!(
        names(&mut db, "MATCH (n:Small) RETURN n.name AS name"),
        vec!["s1", "s2"]
    );
    assert!(names(&mut db, "MATCH (n:Missing) RETURN n.name AS name").is_empty());
    assert_eq!(
        names(&mut db, "MATCH (n:Small:Tagged) RETURN n.name AS name"),
        vec!["s2"],
        "a conjunction, each node once"
    );
    assert_eq!(
        names(&mut db, "MATCH (n:Tagged:Small) RETURN n.name AS name"),
        vec!["s2"]
    );
    // The large node is still found by its own label.
    db.set_query_memory_limit(64 << 20).expect("limit");
    assert_eq!(
        names(&mut db, "MATCH (n:Big) RETURN n.name AS name"),
        vec!["big"]
    );
}

/// A LIMIT over a label scan reads only as many members as it keeps: the
/// label's large members after the first few are never read, so a statement
/// whose memory limit all of them would break answers, with the first rows
/// in node order. SKIP is counted in, and a filter above the scan, which may
/// drop rows, reads on until it has enough.
#[test]
fn a_limit_reads_only_the_members_it_keeps() {
    let (mut db, _dir) = open_db();
    let blob = "x".repeat(512 << 10);
    for i in 0..8 {
        db.execute_cypher(&format!("CREATE (:Wide {{i: {i}, payload: '{blob}'}})"))
            .expect("member");
    }
    // Room for the three rows the SKIP and LIMIT below keep, not for eight.
    db.set_query_memory_limit(3 << 20).expect("limit");
    let first = |db: &mut Database, query: &str| -> Vec<i64> {
        db.execute_cypher(query)
            .unwrap_or_else(|e| panic!("{query}: {e}"))
            .iter()
            .map(|row| match row.get("i") {
                Some(Value::Int(i)) => *i,
                other => panic!("expected i, got {other:?}"),
            })
            .collect()
    };
    assert_eq!(
        first(&mut db, "MATCH (n:Wide) RETURN n.i AS i LIMIT 2"),
        [0, 1]
    );
    assert_eq!(
        first(&mut db, "MATCH (n:Wide) RETURN n.i AS i SKIP 1 LIMIT 2"),
        [1, 2]
    );
    assert_eq!(
        first(
            &mut db,
            "MATCH (n:Wide) WHERE n.i >= 1 RETURN n.i AS i LIMIT 1"
        ),
        [1]
    );
    assert!(
        db.execute_cypher("MATCH (n:Wide) RETURN n.i AS i").is_err(),
        "all members do not fit the limit"
    );
}

/// A server-side cursor pages a label's members after the last node it
/// returned: every member once, in order, none of another label, and the
/// last page says it is the last.
#[test]
fn a_cursor_pages_a_labels_members() {
    let (mut db, _dir) = open_db();
    db.execute_cypher(
        "UNWIND range(1, 250) AS i \
         CREATE (:A {name: 'a' + toString(i)}), (:B {name: 'b' + toString(i)})",
    )
    .expect("nodes");
    let mut seen = Vec::new();
    let (mut resume, mut read_ts) = (None, None);
    let mut pages = 0;
    loop {
        let page = db
            .execute_cypher_paged(
                "MATCH (n:A) RETURN n.name AS name",
                None,
                read_ts,
                resume,
                64,
            )
            .expect("page");
        pages += 1;
        assert!(pages < 20, "pages never ended");
        read_ts = Some(page.read_ts);
        seen.extend(page.rows.iter().map(|row| match row.get("name") {
            Some(Value::String(s)) => s.clone(),
            other => panic!("expected a name, got {other:?}"),
        }));
        if page.exhausted {
            break;
        }
        resume = page.last_key;
    }
    assert_eq!(seen.len(), 250, "every member once");
    let mut unique = seen.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(unique.len(), 250, "no member twice");
    assert!(
        seen.iter().all(|n| n.starts_with('a')),
        "none of another label"
    );
    assert!(pages >= 4, "paged, not read whole: {pages}");
}

/// A temporal label is scanned through the lookup too: each member's
/// timeline is resolved, so the scan returns the state valid now, once per
/// node, however many versions it has.
#[test]
fn a_temporal_labels_scan_returns_each_members_current_state() {
    let (mut db, _dir) = open_db();
    db.execute_cypher(
        "CREATE NODE TYPE Emp TEMPORAL WITH (name: STRING, valid_from: INT, valid_to: INT)",
    )
    .expect("temporal label");
    let since = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_micros() as i64
        - 1_000_000;
    for name in ["e", "f"] {
        db.execute_cypher(&format!(
            "CREATE (:Emp {{name: '{name}', valid_from: {since}}})"
        ))
        .expect("create");
    }
    db.execute_cypher("MATCH (n:Emp {name: 'e'}) SET n.name = 'e2'")
        .expect("new version");
    assert_eq!(
        names(&mut db, "MATCH (n:Emp) RETURN n.name AS name"),
        vec!["e2", "f"]
    );
}

/// A node of a temporal label is found by a scan of an ordinary label it
/// also holds: the ordinary label's members include it, read through its
/// versions.
#[test]
fn an_ordinary_label_scan_finds_its_temporal_holders() {
    let (mut db, _dir) = open_db();
    db.execute_cypher(
        "CREATE NODE TYPE Emp TEMPORAL WITH (name: STRING, valid_from: INT, valid_to: INT)",
    )
    .expect("temporal label");
    let since = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_micros() as i64
        - 1_000_000;
    db.execute_cypher(&format!(
        "CREATE (:Emp:Person {{name: 'e', valid_from: {since}}})"
    ))
    .expect("temporal holder");
    db.execute_cypher("CREATE (:Person {name: 'p'})")
        .expect("ordinary holder");
    assert_eq!(
        names(&mut db, "MATCH (n:Person) RETURN n.name AS name"),
        vec!["e", "p"]
    );
}

/// A scan reads the lookup at the statement's snapshot: a snapshot taken
/// before a node left the label still finds it there, and one taken before
/// it joined does not.
#[test]
fn a_scan_at_an_older_snapshot_sees_the_members_of_then() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (:X {name: 'x1'})").expect("x1");
    let before = db.engine().current_seqno();
    db.execute_cypher("CREATE (:X {name: 'x2'})").expect("x2");
    db.execute_cypher("MATCH (n:X {name: 'x1'}) DELETE n")
        .expect("delete");
    let at_before = format!("MATCH (n:X) RETURN n.name AS name AS OF TIMESTAMP {before}");
    assert_eq!(names(&mut db, &at_before), vec!["x1"]);
    assert_eq!(
        names(&mut db, "MATCH (n:X) RETURN n.name AS name"),
        vec!["x2"]
    );
}

/// A transaction sees its own label changes in its label scans: a node it
/// created, and not one it took the label from.
#[test]
fn a_transaction_scans_its_own_label_changes() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (:Y {name: 'old'})").expect("old");
    let tx = db.begin_transaction();
    db.execute_in_transaction(tx, "CREATE (:Y {name: 'new'})", None)
        .expect("create");
    db.execute_in_transaction(tx, "MATCH (n:Y {name: 'old'}) REMOVE n:Y", None)
        .expect("remove");
    let rows = db
        .execute_in_transaction(tx, "MATCH (n:Y) RETURN n.name AS name", None)
        .expect("scan");
    let seen: Vec<String> = rows
        .iter()
        .map(|row| match row.get("name") {
            Some(Value::String(s)) => s.clone(),
            other => panic!("expected a name, got {other:?}"),
        })
        .collect();
    assert_eq!(seen, vec!["new"]);
    db.rollback_transaction(tx).expect("rollback");
    assert_eq!(
        names(&mut db, "MATCH (n:Y) RETURN n.name AS name"),
        vec!["old"]
    );
}

/// A lookup being built answers no scan: until it is ready, and for a
/// snapshot older than the commit that made it ready, a scan reads the
/// records, so no member is missed.
#[test]
fn a_scan_before_the_lookup_is_ready_reads_the_records() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("UNWIND range(1, 4000) AS i CREATE (:Z {name: 'z' + toString(i)})")
        .expect("nodes");
    forget_label_lookup(&db);
    db.refresh_btree_indexes().expect("refresh");
    // Provisioned again and filled in the background.
    db.resume_interrupted_index_builds().expect("provision");
    // Most likely before the lookup is ready; whichever it is, the scan at
    // this snapshot finds every node.
    let before = db.engine().current_seqno();
    let while_building = names(&mut db, "MATCH (n:Z) RETURN n.name AS name").len();
    assert_eq!(while_building, 4000);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while LocalIndexStore::new(db.engine())
        .label_lookup()
        .expect("read")
        .is_none_or(|d| d.state != IndexState::Ready)
    {
        assert!(std::time::Instant::now() < deadline, "never built");
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    assert_eq!(
        names(&mut db, "MATCH (n:Z) RETURN n.name AS name").len(),
        4000
    );
    let at_before = format!("MATCH (n:Z) RETURN n.name AS name AS OF TIMESTAMP {before}");
    assert_eq!(
        names(&mut db, &at_before).len(),
        4000,
        "a snapshot before the lookup was ready reads the records"
    );
}

/// EXPLAIN names the lookup a label scan reads.
#[test]
fn explain_names_the_label_lookup() {
    let (db, _dir) = open_db();
    let plan = db
        .explain_cypher("MATCH (n:Small) RETURN n")
        .expect("explain");
    assert!(
        plan.contains("read the label's members from the built-in label lookup"),
        "{plan}"
    );
}

/// Set by the parent of the crash test; the child does nothing without it.
const CRASH_CHILD_DIR: &str = "COORDINODE_LABEL_LOOKUP_CRASH_DIR";

/// Child half of `label_scans_after_a_crash_match_the_records`: label
/// changes of every kind, then death without destructors.
#[test]
fn crash_child_changes_labels_then_aborts() {
    let Some(dir) = std::env::var_os(CRASH_CHILD_DIR) else {
        return;
    };
    let mut db = Database::open(std::path::Path::new(&dir)).expect("open in child");
    db.execute_cypher("UNWIND range(1, 50) AS i CREATE (:K {name: 'k' + toString(i)})")
        .expect("create");
    db.execute_cypher("MATCH (n:K) WHERE n.name IN ['k1', 'k2', 'k3'] SET n:J")
        .expect("label added");
    db.execute_cypher("MATCH (n:K) WHERE n.name IN ['k4', 'k5'] REMOVE n:K")
        .expect("label removed");
    db.execute_cypher("MATCH (n:K {name: 'k6'}) DELETE n")
        .expect("delete");
    std::process::abort();
}

/// After a kill and the journal replay that follows it, every label's
/// members are exactly the nodes the records give it: the postings are
/// replayed with the writes that staged them.
#[test]
fn label_scans_after_a_crash_match_the_records() {
    let dir = tempfile::tempdir().expect("tempdir");
    let status = std::process::Command::new(std::env::current_exe().expect("test binary"))
        .args([
            "--exact",
            "integration::label_lookup::crash_child_changes_labels_then_aborts",
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
    for label in ["K", "J"] {
        assert_eq!(members(&db, label), holders(&mut db, label), "{label}");
    }
    assert_eq!(
        names(&mut db, "MATCH (n:K) RETURN n.name AS name").len(),
        47
    );
    assert_eq!(
        names(&mut db, "MATCH (n:J) RETURN n.name AS name"),
        vec!["k1", "k2", "k3"]
    );
}

/// Remove every trace of the label lookup from `db`'s storage, as a store
/// written before it existed has none.
fn forget_label_lookup(db: &Database) {
    use coordinode_modality::{IndexBuildRecord, IndexDefinition, IndexIntegrityRecord};
    use coordinode_storage::engine::partition::Partition;
    let engine = db.engine();
    let lookup = LocalIndexStore::new(engine)
        .label_lookup()
        .expect("read")
        .expect("lookup");
    let prefix = coordinode_core::index::encoding::label_postings_prefix(lookup.generation);
    let postings: Vec<Vec<u8>> = engine
        .prefix_scan(Partition::Idx, &prefix)
        .expect("postings")
        .map(|row| row.into_inner().expect("posting").0.to_vec())
        .collect();
    assert!(!postings.is_empty(), "the writes staged postings");
    for key in postings {
        engine.delete(Partition::Idx, &key).expect("drop a posting");
    }
    for key in [
        IndexDefinition::LABEL_LOOKUP_KEY.to_vec(),
        lookup.schema_key(),
        IndexBuildRecord::key_of(lookup.generation),
        IndexIntegrityRecord::key_of(lookup.generation),
    ] {
        engine
            .delete(Partition::Schema, &key)
            .expect("drop the record");
    }
    engine.persist().expect("persist");
}
