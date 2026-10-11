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
