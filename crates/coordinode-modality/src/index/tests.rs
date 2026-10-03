use super::*;

use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_storage::engine::transaction::{CommitContext, CommitError};

use crate::index_def::IndexMaintenance;

/// Logic-test fixture (memory backing, env-flippable).
fn open_engine() -> coordinode_test_fixtures::EngineFixture {
    coordinode_test_fixtures::engine_for_logic()
}

fn commit(t: &mut Transaction) -> Result<(), CommitError> {
    let wc = WriteConcern::majority();
    let ctx = CommitContext {
        write_concern: &wc,
        pipeline: None,
        id_gen: None,
        drain_buffer: None,
        nvme_write_buffer: None,
    };
    t.commit(&ctx).map(|_| ())
}

fn s(v: &str) -> Vec<Value> {
    vec![Value::String(v.into())]
}

fn id(raw: u64) -> NodeId {
    NodeId::from_raw(raw)
}

/// No property is bound to a field id: every value is looked up by name.
fn no_fields(_: &str) -> Option<u32> {
    None
}

/// Stage `node` entering `index` holding `values`; the entries put.
fn enter(
    store: &LocalIndexStore,
    t: &mut Transaction,
    index: &IndexDefinition,
    values: &[Value],
    node: NodeId,
) -> usize {
    store
        .stage_membership(t, index, &no_fields, owner(node), None, Some(values))
        .unwrap()
}

/// `node`, not temporal, as the owner of its entries.
fn owner(node: NodeId) -> EntryOwner {
    EntryOwner::node(node.as_raw())
}

/// Stage `node` leaving `index`, where it held `values`.
fn leave(
    store: &LocalIndexStore,
    t: &mut Transaction,
    index: &IndexDefinition,
    values: &[Value],
    node: NodeId,
) {
    store
        .stage_membership(t, index, &no_fields, owner(node), Some(values), None)
        .unwrap();
}

fn derived(mut index: IndexDefinition) -> IndexDefinition {
    index.maintenance = IndexMaintenance {
        profile: crate::index_def::IndexProfile::Derived,
        ..IndexMaintenance::default()
    };
    index
}

/// Entries of a non-unique index commit with their transaction and are
/// found by value, one per node.
#[test]
fn non_unique_entries_are_found_by_value() {
    let fx = open_engine();
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let store = LocalIndexStore::new(&fx.engine);
    let index = IndexDefinition::btree("user_name", "User", "name");

    let mut t = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
    enter(&store, &mut t, &index, &s("alice"), id(1));
    enter(&store, &mut t, &index, &s("alice"), id(2));
    enter(&store, &mut t, &index, &s("bob"), id(3));
    commit(&mut t).unwrap();

    let mut t = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
    let mut hits = store
        .scan_exact(&mut t, &index, &s("alice"))
        .unwrap()
        .unwrap();
    hits.sort_unstable_by_key(|n| n.as_raw());
    assert_eq!(hits, vec![id(1), id(2)]);
    assert_eq!(
        store.scan_exact(&mut t, &index, &s("carol")).unwrap(),
        Some(Vec::new())
    );
}

/// Nothing is written before the transaction commits: a rolled-back
/// statement leaves no entry behind.
#[test]
fn an_uncommitted_entry_is_invisible_outside_its_transaction() {
    let fx = open_engine();
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let store = LocalIndexStore::new(&fx.engine);
    let index = IndexDefinition::btree("user_name", "User", "name");

    let mut t = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
    enter(&store, &mut t, &index, &s("alice"), id(1));
    assert_eq!(
        store.scan_exact(&mut t, &index, &s("alice")).unwrap(),
        Some(vec![id(1)]),
        "the transaction sees its own entry"
    );
    drop(t);

    let mut t = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
    assert_eq!(
        store.scan_exact(&mut t, &index, &s("alice")).unwrap(),
        Some(Vec::new())
    );
}

/// A removal staged in the transaction hides the entry from the
/// transaction's own lookups.
#[test]
fn a_staged_removal_hides_the_entry_from_its_transaction() {
    let fx = open_engine();
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let store = LocalIndexStore::new(&fx.engine);
    let index = IndexDefinition::btree("user_name", "User", "name");

    let mut t = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
    enter(&store, &mut t, &index, &s("alice"), id(1));
    commit(&mut t).unwrap();

    let mut t = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
    leave(&store, &mut t, &index, &s("alice"), id(1));
    assert_eq!(
        store.scan_exact(&mut t, &index, &s("alice")).unwrap(),
        Some(Vec::new())
    );
}

/// A unique entry names its holder; another node's insert of the value
/// sees the conflict, the holder's own re-insert does not.
#[test]
fn a_unique_value_has_one_holder() {
    let fx = open_engine();
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let store = LocalIndexStore::new(&fx.engine);
    let index = IndexDefinition::btree("user_email", "User", "email").unique();

    let mut t = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
    enter(&store, &mut t, &index, &s("a@x"), id(1));
    commit(&mut t).unwrap();

    let mut t = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
    assert_eq!(
        store
            .unique_conflict(&mut t, &index, &s("a@x"), id(2))
            .unwrap(),
        Some(id(1))
    );
    assert_eq!(
        store
            .unique_conflict(&mut t, &index, &s("a@x"), id(1))
            .unwrap(),
        None
    );
    assert_eq!(
        store.committed_conflict(&index, &s("a@x"), id(2)).unwrap(),
        Some(id(1))
    );
    assert_eq!(
        store.scan_exact(&mut t, &index, &s("a@x")).unwrap(),
        Some(vec![id(1)])
    );
}

/// Two transactions that both found a value free and both claim it write
/// one key, so only the first to commit succeeds, in either profile.
#[test]
fn two_concurrent_claims_of_one_value_conflict() {
    for index in [
        IndexDefinition::btree("user_email", "User", "email").unique(),
        derived(IndexDefinition::btree("user_email", "User", "email").unique()),
    ] {
        let fx = open_engine();
        let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
        let store = LocalIndexStore::new(&fx.engine);

        let mut first = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
        let mut second = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
        for (t, node) in [(&mut first, id(1)), (&mut second, id(2))] {
            assert_eq!(
                store.unique_conflict(t, &index, &s("a@x"), node).unwrap(),
                None
            );
            enter(&store, t, &index, &s("a@x"), node);
        }
        commit(&mut first).unwrap();
        assert!(
            commit(&mut second).is_err(),
            "{:?}: the second claim of one value must not commit",
            index.maintenance.profile
        );
        assert_eq!(
            store.committed_conflict(&index, &s("a@x"), id(2)).unwrap(),
            Some(id(1)),
            "{:?}",
            index.maintenance.profile
        );
    }
}

/// Removing another node's unique entry leaves it: the entry of a stale
/// value is only removed by its holder.
#[test]
fn a_unique_entry_is_removed_only_by_its_holder() {
    let fx = open_engine();
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let store = LocalIndexStore::new(&fx.engine);
    let index = IndexDefinition::btree("user_email", "User", "email").unique();

    let mut t = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
    enter(&store, &mut t, &index, &s("a@x"), id(1));
    commit(&mut t).unwrap();

    let mut t = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
    leave(&store, &mut t, &index, &s("a@x"), id(2));
    commit(&mut t).unwrap();
    assert_eq!(
        store.committed_conflict(&index, &s("a@x"), id(2)).unwrap(),
        Some(id(1))
    );

    let mut t = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
    leave(&store, &mut t, &index, &s("a@x"), id(1));
    commit(&mut t).unwrap();
    assert_eq!(
        store.committed_conflict(&index, &s("a@x"), id(2)).unwrap(),
        None
    );
}

/// A list indexes each element; a value with no key indexes nothing and a
/// lookup of it reports that the index cannot answer.
#[test]
fn lists_index_their_elements_and_unkeyed_values_nothing() {
    let fx = open_engine();
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let store = LocalIndexStore::new(&fx.engine);
    let index = IndexDefinition::btree("user_tag", "User", "tags").unique();

    let mut t = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
    let tags = vec![Value::Array(vec![
        Value::String("x".into()),
        Value::String("y".into()),
        Value::String("x".into()),
    ])];
    assert_eq!(enter(&store, &mut t, &index, &tags, id(1)), 2);
    assert_eq!(
        store.unique_conflict(&mut t, &index, &tags, id(1)).unwrap(),
        None,
        "a list repeating an element does not conflict with itself"
    );
    assert_eq!(
        store
            .unique_conflict(&mut t, &index, &s("y"), id(2))
            .unwrap(),
        Some(id(1))
    );

    let map = vec![Value::Map(Default::default())];
    assert_eq!(enter(&store, &mut t, &index, &map, id(3)), 0);
    assert_eq!(store.scan_exact(&mut t, &index, &map).unwrap(), None);
}

/// Compound entries are told apart by every column.
#[test]
fn compound_entries_are_told_apart_by_every_column() {
    let fx = open_engine();
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let store = LocalIndexStore::new(&fx.engine);
    let index = IndexDefinition::compound("by_city_age", "User", vec!["city".into(), "age".into()]);

    let mut t = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
    let a = vec![Value::String("Oslo".into()), Value::Int(30)];
    let b = vec![Value::String("Oslo".into()), Value::Int(31)];
    enter(&store, &mut t, &index, &a, id(1));
    enter(&store, &mut t, &index, &b, id(2));
    assert_eq!(
        store.scan_exact(&mut t, &index, &a).unwrap(),
        Some(vec![id(1)])
    );
    assert_eq!(
        store.scan_exact(&mut t, &index, &b).unwrap(),
        Some(vec![id(2)])
    );
}

/// Clearing an index in a transaction removes every entry of the index, in
/// both shapes, and no entry of another index, when the transaction
/// commits and not before.
#[test]
fn clearing_an_index_removes_its_entries_only() {
    let fx = open_engine();
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let store = LocalIndexStore::new(&fx.engine);
    let plain = IndexDefinition::btree("i", "User", "name");
    let unique = IndexDefinition::btree("i", "User", "name").unique();
    let other = IndexDefinition::btree("ij", "User", "name");

    let mut t = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
    enter(&store, &mut t, &plain, &s("a"), id(1));
    enter(&store, &mut t, &unique, &s("b"), id(2));
    enter(&store, &mut t, &other, &s("a"), id(3));
    commit(&mut t).unwrap();

    let mut clear = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
    store.clear_txn(&mut clear, "i").unwrap();
    let mut t = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
    assert_eq!(
        store.scan_entry_ids(&mut t, &plain).unwrap(),
        vec![id(1)],
        "nothing is removed before the commit"
    );
    commit(&mut clear).unwrap();

    let mut t = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
    assert!(store.scan_entry_ids(&mut t, &plain).unwrap().is_empty());
    assert!(store.scan_entry_ids(&mut t, &unique).unwrap().is_empty());
    assert_eq!(store.scan_entry_ids(&mut t, &other).unwrap(), vec![id(3)]);
}

/// Versions of a temporal node have entries of their own: two versions
/// holding one value make one candidate node, and a version leaving the
/// value removes only its entry, in both profiles.
#[test]
fn version_entries_answer_once_and_move_alone() {
    for index in [
        IndexDefinition::btree("user_name", "User", "name"),
        derived(IndexDefinition::btree("user_name", "User", "name")),
    ] {
        let fx = open_engine();
        let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
        let store = LocalIndexStore::new(&fx.engine);
        let (first, second) = (EntryOwner::version(1, 100), EntryOwner::version(1, 200));

        let mut t = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
        for version in [first, second] {
            store
                .stage_membership(&mut t, &index, &no_fields, version, None, Some(&s("ada")))
                .unwrap();
        }
        assert_eq!(
            store.scan_exact(&mut t, &index, &s("ada")).unwrap(),
            Some(vec![id(1)]),
            "one candidate for the node's two versions"
        );
        commit(&mut t).unwrap();

        let mut t = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
        store
            .stage_membership(&mut t, &index, &no_fields, first, Some(&s("ada")), None)
            .unwrap();
        commit(&mut t).unwrap();

        let mut t = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
        assert_eq!(
            store.scan_exact(&mut t, &index, &s("ada")).unwrap(),
            Some(vec![id(1)]),
            "the second version still holds the value"
        );
        assert_eq!(store.scan_entry_ids(&mut t, &index).unwrap(), vec![id(1)]);
    }
}

/// A DERIVED index reads its own entries before commit, holds them after,
/// moves them on a change and leaves nothing of a rolled-back statement:
/// the same view a RESOLVED index gives.
#[test]
fn a_derived_index_gives_the_resolved_view() {
    let fx = open_engine();
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let store = LocalIndexStore::new(&fx.engine);
    let index = derived(IndexDefinition::btree("user_name", "User", "name"));

    let mut t = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
    enter(&store, &mut t, &index, &s("alice"), id(1));
    assert_eq!(
        store.scan_exact(&mut t, &index, &s("alice")).unwrap(),
        Some(vec![id(1)]),
        "the transaction sees its own entry"
    );
    commit(&mut t).unwrap();

    let mut t = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
    store
        .stage_membership(
            &mut t,
            &index,
            &no_fields,
            owner(id(1)),
            Some(&s("alice")),
            Some(&s("bob")),
        )
        .unwrap();
    commit(&mut t).unwrap();

    let mut t = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
    assert_eq!(
        store.scan_exact(&mut t, &index, &s("alice")).unwrap(),
        Some(Vec::new())
    );
    assert_eq!(
        store.scan_exact(&mut t, &index, &s("bob")).unwrap(),
        Some(vec![id(1)])
    );
    enter(&store, &mut t, &index, &s("carol"), id(2));
    drop(t);

    let mut t = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
    assert_eq!(
        store.scan_exact(&mut t, &index, &s("carol")).unwrap(),
        Some(Vec::new()),
        "a rolled-back statement leaves no entry"
    );
}

#[test]
fn definition_txn_round_trip() {
    let fx = open_engine();
    let engine = &fx.engine;
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let store = LocalIndexStore::new(engine);
    let def = IndexDefinition::btree("user_email", "User", "email").unique();

    let mut t = Transaction::begin(engine, Some(&oracle), oracle.next());
    store.put_definition_txn(&mut t, &def).expect("put txn");
    commit(&mut t).unwrap();
    let loaded = store
        .load_definition("user_email")
        .expect("load")
        .expect("present after commit");
    assert!(loaded.unique);

    let mut t = Transaction::begin(engine, Some(&oracle), oracle.next());
    store
        .delete_definition_txn(&mut t, "user_email")
        .expect("delete txn");
    commit(&mut t).unwrap();
    assert!(store.load_definition("user_email").expect("load").is_none());
}
