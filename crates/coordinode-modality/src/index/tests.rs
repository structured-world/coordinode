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

/// A query budget no lookup here comes near.
fn budget() -> coordinode_core::budget::QueryBudget {
    coordinode_core::budget::QueryBudget::new(coordinode_core::budget::DEFAULT_QUERY_MEMORY_LIMIT)
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

/// `descriptor` as the index numbered `raw` serving from generation `raw`:
/// the entry tests drive the store without a catalog.
fn bound(descriptor: IndexDescriptor, raw: u64) -> IndexDefinition {
    descriptor.bind(IndexId::from_raw(raw), GenerationId::from_raw(raw))
}

/// Entries of a non-unique index commit with their transaction and are
/// found by value, one per node.
#[test]
fn non_unique_entries_are_found_by_value() {
    let fx = open_engine();
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let store = LocalIndexStore::new(&fx.engine);
    let index = bound(IndexDescriptor::btree("user_name", "User", "name"), 1);

    let mut t = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
    enter(&store, &mut t, &index, &s("alice"), id(1));
    enter(&store, &mut t, &index, &s("alice"), id(2));
    enter(&store, &mut t, &index, &s("bob"), id(3));
    commit(&mut t).unwrap();

    let mut t = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
    let mut hits = store
        .scan_exact(&mut t, &index, &s("alice"), &budget())
        .unwrap()
        .unwrap();
    hits.sort_unstable_by_key(|n| n.as_raw());
    assert_eq!(hits, vec![id(1), id(2)]);
    assert_eq!(
        store
            .scan_exact(&mut t, &index, &s("carol"), &budget())
            .unwrap(),
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
    let index = bound(IndexDescriptor::btree("user_name", "User", "name"), 1);

    let mut t = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
    enter(&store, &mut t, &index, &s("alice"), id(1));
    assert_eq!(
        store
            .scan_exact(&mut t, &index, &s("alice"), &budget())
            .unwrap(),
        Some(vec![id(1)]),
        "the transaction sees its own entry"
    );
    drop(t);

    let mut t = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
    assert_eq!(
        store
            .scan_exact(&mut t, &index, &s("alice"), &budget())
            .unwrap(),
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
    let index = bound(IndexDescriptor::btree("user_name", "User", "name"), 1);

    let mut t = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
    enter(&store, &mut t, &index, &s("alice"), id(1));
    commit(&mut t).unwrap();

    let mut t = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
    leave(&store, &mut t, &index, &s("alice"), id(1));
    assert_eq!(
        store
            .scan_exact(&mut t, &index, &s("alice"), &budget())
            .unwrap(),
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
    let index = bound(
        IndexDescriptor::btree("user_email", "User", "email").unique(),
        1,
    );

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
        store
            .scan_exact(&mut t, &index, &s("a@x"), &budget())
            .unwrap(),
        Some(vec![id(1)])
    );
}

/// Two transactions that both found a value free and both claim it write
/// one key, so only the first to commit succeeds, in either profile.
#[test]
fn two_concurrent_claims_of_one_value_conflict() {
    for index in [
        bound(
            IndexDescriptor::btree("user_email", "User", "email").unique(),
            1,
        ),
        derived(bound(
            IndexDescriptor::btree("user_email", "User", "email").unique(),
            1,
        )),
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
    let index = bound(
        IndexDescriptor::btree("user_email", "User", "email").unique(),
        1,
    );

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
    let index = bound(
        IndexDescriptor::btree("user_tag", "User", "tags").unique(),
        1,
    );

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
    assert_eq!(
        store.scan_exact(&mut t, &index, &map, &budget()).unwrap(),
        None
    );
}

/// Compound entries are told apart by every column.
#[test]
fn compound_entries_are_told_apart_by_every_column() {
    let fx = open_engine();
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let store = LocalIndexStore::new(&fx.engine);
    let index = bound(
        IndexDescriptor::compound("by_city_age", "User", vec!["city".into(), "age".into()]),
        1,
    );

    let mut t = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
    let a = vec![Value::String("Oslo".into()), Value::Int(30)];
    let b = vec![Value::String("Oslo".into()), Value::Int(31)];
    enter(&store, &mut t, &index, &a, id(1));
    enter(&store, &mut t, &index, &b, id(2));
    assert_eq!(
        store.scan_exact(&mut t, &index, &a, &budget()).unwrap(),
        Some(vec![id(1)])
    );
    assert_eq!(
        store.scan_exact(&mut t, &index, &b, &budget()).unwrap(),
        Some(vec![id(2)])
    );
}

/// Clearing a generation in a transaction removes every entry of it, in
/// both shapes, and no entry of another generation, the neighbouring
/// numbers included, when the transaction commits and not before.
#[test]
fn clearing_a_generation_removes_its_entries_only() {
    let fx = open_engine();
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let store = LocalIndexStore::new(&fx.engine);
    let plain = bound(IndexDescriptor::btree("i", "User", "name"), 0x100);
    let unique = bound(IndexDescriptor::btree("i", "User", "name").unique(), 0x100);
    let other = bound(IndexDescriptor::btree("j", "User", "name"), 0x101);
    let before = bound(IndexDescriptor::btree("k", "User", "name").unique(), 0xFF);

    let mut t = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
    enter(&store, &mut t, &plain, &s("a"), id(1));
    enter(&store, &mut t, &unique, &s("b"), id(2));
    enter(&store, &mut t, &other, &s("a"), id(3));
    enter(&store, &mut t, &before, &s("a"), id(4));
    commit(&mut t).unwrap();

    let mut clear = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
    store.clear_txn(&mut clear, plain.generation).unwrap();
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
    assert_eq!(store.scan_entry_ids(&mut t, &before).unwrap(), vec![id(4)]);
}

/// Versions of a temporal node have entries of their own: two versions
/// holding one value make one candidate node, and a version leaving the
/// value removes only its entry, in both profiles.
#[test]
fn version_entries_answer_once_and_move_alone() {
    for index in [
        bound(IndexDescriptor::btree("user_name", "User", "name"), 1),
        derived(bound(
            IndexDescriptor::btree("user_name", "User", "name"),
            1,
        )),
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
            store
                .scan_exact(&mut t, &index, &s("ada"), &budget())
                .unwrap(),
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
            store
                .scan_exact(&mut t, &index, &s("ada"), &budget())
                .unwrap(),
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
    let index = derived(bound(
        IndexDescriptor::btree("user_name", "User", "name"),
        1,
    ));

    let mut t = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
    enter(&store, &mut t, &index, &s("alice"), id(1));
    assert_eq!(
        store
            .scan_exact(&mut t, &index, &s("alice"), &budget())
            .unwrap(),
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
        store
            .scan_exact(&mut t, &index, &s("alice"), &budget())
            .unwrap(),
        Some(Vec::new())
    );
    assert_eq!(
        store
            .scan_exact(&mut t, &index, &s("bob"), &budget())
            .unwrap(),
        Some(vec![id(1)])
    );
    enter(&store, &mut t, &index, &s("carol"), id(2));
    drop(t);

    let mut t = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
    assert_eq!(
        store
            .scan_exact(&mut t, &index, &s("carol"), &budget())
            .unwrap(),
        Some(Vec::new()),
        "a rolled-back statement leaves no entry"
    );
}

/// A published definition is stored under its identity and found by its
/// name; deleting it removes both, and its numbers stay taken.
#[test]
fn publication_round_trip_keeps_numbers_taken() {
    let fx = open_engine();
    let engine = &fx.engine;
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let store = LocalIndexStore::new(engine);

    let mut t = Transaction::begin(engine, Some(&oracle), oracle.next());
    let def = store
        .publish_definition_txn(
            &mut t,
            IndexDescriptor::btree("user_email", "User", "email").unique(),
        )
        .expect("publish");
    commit(&mut t).unwrap();
    assert_eq!(store.resolve_name("user_email").unwrap(), Some(def.id));
    let loaded = store
        .load_definition(def.id)
        .expect("load")
        .expect("present after commit");
    assert_eq!(loaded, def);

    let mut t = Transaction::begin(engine, Some(&oracle), oracle.next());
    store.delete_definition_txn(&mut t, &def).expect("delete");
    commit(&mut t).unwrap();
    assert!(store.load_definition(def.id).unwrap().is_none());
    assert_eq!(store.resolve_name("user_email").unwrap(), None);

    // The same name again is a new object with new numbers: a delayed
    // writer or cleanup bound to the dropped one cannot reach it.
    let mut t = Transaction::begin(engine, Some(&oracle), oracle.next());
    let again = store
        .publish_definition_txn(
            &mut t,
            IndexDescriptor::btree("user_email", "User", "email").unique(),
        )
        .expect("publish");
    commit(&mut t).unwrap();
    assert_ne!(again.id, def.id);
    assert_ne!(again.generation, def.generation);
}

/// A name another live index holds refuses the publication without taking
/// a number; an unnamed index needs no binding and is found by identity.
#[test]
fn a_taken_name_refuses_publication_and_unnamed_indexes_need_none() {
    let fx = open_engine();
    let engine = &fx.engine;
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let store = LocalIndexStore::new(engine);

    let mut t = Transaction::begin(engine, Some(&oracle), oracle.next());
    let first = store
        .publish_definition_txn(&mut t, IndexDescriptor::btree("i", "User", "a"))
        .expect("publish");
    commit(&mut t).unwrap();

    let mut t = Transaction::begin(engine, Some(&oracle), oracle.next());
    assert!(matches!(
        store.publish_definition_txn(&mut t, IndexDescriptor::btree("i", "Post", "b")),
        Err(StoreError::IndexNameTaken(name)) if name == "i"
    ));
    let unnamed = IndexDescriptor {
        name: None,
        ..IndexDescriptor::btree("unused", "Post", "b")
    };
    let second = store
        .publish_definition_txn(&mut t, unnamed)
        .expect("publish unnamed");
    commit(&mut t).unwrap();
    assert_eq!(
        (second.id.as_raw(), second.generation.as_raw()),
        (first.id.as_raw() + 1, first.generation.as_raw() + 1),
        "the refused publication took no number"
    );
    assert_eq!(store.load_definition(second.id).unwrap(), Some(second));
}

/// Two statements publishing at once read the same free numbers; the
/// allocator record lets only one of them commit, so no number is handed
/// out twice. Two of one name conflict the same way.
#[test]
fn concurrent_publications_never_share_a_number_or_a_name() {
    let fx = open_engine();
    let engine = &fx.engine;
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let store = LocalIndexStore::new(engine);

    let mut a = Transaction::begin(engine, Some(&oracle), oracle.next());
    let mut b = Transaction::begin(engine, Some(&oracle), oracle.next());
    let da = store
        .publish_definition_txn(&mut a, IndexDescriptor::btree("a", "User", "x"))
        .expect("publish a");
    let db = store
        .publish_definition_txn(&mut b, IndexDescriptor::btree("b", "User", "y"))
        .expect("publish b");
    assert_eq!(da.id, db.id, "both read the same free number");
    commit(&mut a).unwrap();
    assert!(commit(&mut b).is_err(), "the second must not commit");

    let mut c = Transaction::begin(engine, Some(&oracle), oracle.next());
    let mut d = Transaction::begin(engine, Some(&oracle), oracle.next());
    store
        .publish_definition_txn(&mut c, IndexDescriptor::btree("same", "User", "x"))
        .expect("publish c");
    commit(&mut c).unwrap();
    // `d` began before `c` committed, so its read finds the name free; the
    // binding's condition refuses it at commit.
    let found_free =
        store.publish_definition_txn(&mut d, IndexDescriptor::btree("same", "Post", "y"));
    match found_free {
        Ok(_) => assert!(commit(&mut d).is_err(), "a second binding of one name"),
        Err(e) => assert!(matches!(e, StoreError::IndexNameTaken(_)), "{e}"),
    }
}

/// A description is catalog metadata: editing it rewrites the definition
/// record only, never the generation or an entry.
#[test]
fn a_description_edit_touches_no_entry() {
    let fx = open_engine();
    let engine = &fx.engine;
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let store = LocalIndexStore::new(engine);

    let mut t = Transaction::begin(engine, Some(&oracle), oracle.next());
    let mut def = store
        .publish_definition_txn(&mut t, IndexDescriptor::btree("i", "User", "name"))
        .expect("publish");
    enter(&store, &mut t, &def, &s("a"), id(1));
    commit(&mut t).unwrap();
    let entries_before = engine
        .prefix_scan(Partition::Idx, &entries_prefix(def.generation))
        .unwrap()
        .count();

    let mut t = Transaction::begin(engine, Some(&oracle), oracle.next());
    def.description = Some("emails of active users".into());
    store.put_definition_txn(&mut t, &def).expect("put");
    commit(&mut t).unwrap();

    let loaded = store.load_definition(def.id).unwrap().expect("present");
    assert_eq!(
        loaded.description.as_deref(),
        Some("emails of active users")
    );
    assert_eq!(loaded.generation, def.generation);
    assert_eq!(
        engine
            .prefix_scan(Partition::Idx, &entries_prefix(def.generation))
            .unwrap()
            .count(),
        entries_before
    );
}

/// A rebuild's generation comes from the same allocator: never one an
/// index already used.
#[test]
fn a_new_generation_is_never_one_already_used() {
    let fx = open_engine();
    let engine = &fx.engine;
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let store = LocalIndexStore::new(engine);

    let mut t = Transaction::begin(engine, Some(&oracle), oracle.next());
    let def = store
        .publish_definition_txn(&mut t, IndexDescriptor::btree("i", "User", "name"))
        .expect("publish");
    let rebuilt = store.allocate_generation_txn(&mut t).expect("allocate");
    commit(&mut t).unwrap();
    assert_ne!(rebuilt, def.generation);

    let mut t = Transaction::begin(engine, Some(&oracle), oracle.next());
    let other = store
        .publish_definition_txn(&mut t, IndexDescriptor::btree("j", "User", "name"))
        .expect("publish");
    commit(&mut t).unwrap();
    assert!(other.generation != rebuilt && other.generation != def.generation);
}

/// An index build record this build cannot read refuses the listing, naming
/// the record. Skipping it left the build to nobody: opening the store resumes
/// builds from this list, so an unread record was an index that never
/// finished and never failed.
#[test]
fn an_unreadable_build_record_refuses_the_listing() {
    let fx = open_engine();
    let engine = &fx.engine;
    let key = IndexBuildRecord::key_of(GenerationId::from_raw(7));
    engine
        .put(Partition::Schema, &key, b"not-msgpack-bytes")
        .expect("plant");

    match LocalIndexStore::new(engine).list_builds() {
        Err(StoreError::Storage(StorageError::UnreadableCatalog {
            kind, key: named, ..
        })) => {
            assert_eq!(kind, "index build record");
            assert_eq!(named, coordinode_storage::error::printable_key(&key));
        }
        other => panic!("expected the unreadable record to refuse the listing, got {other:?}"),
    }
}
