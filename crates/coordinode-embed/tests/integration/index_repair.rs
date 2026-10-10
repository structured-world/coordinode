//! B-tree indexes whose entries disagree with the records they index.
//!
//! An entry is derived from a record; when the two disagree (an entry naming
//! the wrong node, an entry with no record behind it, a record with no
//! entry), the record is the truth. Reads answer from the records rather
//! than return what the entry claims, a value is never reported held by a
//! node that does not hold it, and never accepted as free on the word of an
//! entry that is known wrong.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_core::graph::types::Value;
use coordinode_core::index::encoding::{encode_tuple, encode_unique_entry_key};
use coordinode_embed::Database;
use coordinode_storage::engine::partition::Partition;

use super::helpers::index_named;

fn open_db() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open db");
    (db, dir)
}

/// The id of the one `:U` node whose email is `email`, read without any
/// index: a scan of the label.
fn id_of(db: &mut Database, email: &str) -> i64 {
    let rows = db
        .execute_cypher(&format!(
            "MATCH (u:U) WHERE u.email + '' = '{email}' RETURN id(u) AS id"
        ))
        .expect("scan");
    assert_eq!(rows.len(), 1, "one node holds {email}");
    match rows[0].get("id") {
        Some(Value::Int(id)) => *id,
        other => panic!("expected an id, got {other:?}"),
    }
}

/// The ids of the `:U` nodes the indexed equality on `email` finds.
fn found(db: &mut Database, email: &str) -> Vec<i64> {
    let mut ids: Vec<i64> = db
        .execute_cypher(&format!(
            "MATCH (u:U {{email: '{email}'}}) RETURN id(u) AS id"
        ))
        .expect("lookup")
        .iter()
        .map(|row| match row.get("id") {
            Some(Value::Int(id)) => *id,
            other => panic!("expected an id, got {other:?}"),
        })
        .collect();
    ids.sort_unstable();
    ids
}

/// Overwrite the unique entry of `email` in the index `name` so it names
/// `holder`, as a defect that wrote the wrong node would have left it.
fn misattribute(db: &Database, name: &str, email: &str, holder: i64) {
    let index = index_named(db.engine(), name).expect("index");
    let tuple = encode_tuple(&[Value::String(email.into())]).expect("tuple");
    let key = encode_unique_entry_key(index.generation, &tuple);
    let holder = u64::try_from(holder).expect("id").to_be_bytes();
    db.engine()
        .put(Partition::Idx, &key, &holder)
        .expect("overwrite the entry");
}

/// A unique entry names node B while node A is the one holding the value: a
/// lookup by the value finds A, the node that holds it, and not nothing.
#[test]
fn a_lookup_through_a_misattributed_unique_entry_finds_the_real_holder() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE UNIQUE INDEX u_email ON :U(email)")
        .expect("index");
    db.execute_cypher("CREATE (:U {email: 'a@x'})").expect("a");
    db.execute_cypher("CREATE (:U {email: 'z@x'})").expect("z");
    let owner = id_of(&mut db, "a@x");
    let wrong = id_of(&mut db, "z@x");
    misattribute(&db, "u_email", "a@x", wrong);

    assert_eq!(found(&mut db, "a@x"), vec![owner]);
}

/// A unique value held by A, whose entry names B: inserting it again is a
/// duplicate of A, the real holder, not of B.
#[test]
fn a_duplicate_of_a_misattributed_value_names_the_real_holder() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE UNIQUE INDEX u_email ON :U(email)")
        .expect("index");
    db.execute_cypher("CREATE (:U {email: 'a@x'})").expect("a");
    db.execute_cypher("CREATE (:U {email: 'z@x'})").expect("z");
    let owner = id_of(&mut db, "a@x");
    let wrong = id_of(&mut db, "z@x");
    misattribute(&db, "u_email", "a@x", wrong);

    let err = db
        .execute_cypher("CREATE (:U {email: 'a@x'})")
        .expect_err("a@x is held");
    let message = err.to_string();
    let owner_element =
        coordinode_core::graph::node::NodeId::from_raw(owner as u64).to_element_id();
    assert!(
        message.contains(&owner_element),
        "the duplicate names the holder {owner_element}: {message}"
    );
    assert_eq!(found(&mut db, "a@x"), vec![owner], "nothing was inserted");
}

/// A unique entry left naming B after the value's real holder moved to
/// another value: nobody holds the value, so it can be taken, and the new
/// holder is what a lookup finds.
#[test]
fn a_value_whose_entry_names_a_node_that_never_held_it_can_be_taken() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE UNIQUE INDEX u_email ON :U(email)")
        .expect("index");
    db.execute_cypher("CREATE (:U {email: 'a@x'})").expect("a");
    db.execute_cypher("CREATE (:U {email: 'z@x'})").expect("z");
    let wrong = id_of(&mut db, "z@x");
    misattribute(&db, "u_email", "a@x", wrong);
    db.execute_cypher("MATCH (u:U) WHERE u.email + '' = 'a@x' SET u.email = 'moved@x'")
        .expect("move a@x away");

    db.execute_cypher("CREATE (:U {email: 'a@x'})")
        .expect("a@x is free");
    let taker = id_of(&mut db, "a@x");
    assert_eq!(found(&mut db, "a@x"), vec![taker]);
}

/// A MERGE on a misattributed value matches the node that holds it, as the
/// stuck enrollment needed: no false miss, no false duplicate.
#[test]
fn a_merge_on_a_misattributed_value_matches_the_real_holder() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE UNIQUE INDEX u_email ON :U(email)")
        .expect("index");
    db.execute_cypher("CREATE (:U {email: 'a@x', n: 1})")
        .expect("a");
    db.execute_cypher("CREATE (:U {email: 'z@x'})").expect("z");
    let owner = id_of(&mut db, "a@x");
    let wrong = id_of(&mut db, "z@x");
    misattribute(&db, "u_email", "a@x", wrong);

    db.execute_cypher("MERGE (u:U {email: 'a@x'}) SET u.n = 2")
        .expect("merge");
    let rows = db
        .execute_cypher("MATCH (u:U) WHERE u.email + '' = 'a@x' RETURN id(u) AS id, u.n AS n")
        .expect("scan");
    assert_eq!(rows.len(), 1, "the merge created no second holder");
    assert_eq!(rows[0].get("id"), Some(&Value::Int(owner)));
    assert_eq!(rows[0].get("n"), Some(&Value::Int(2)));
}
