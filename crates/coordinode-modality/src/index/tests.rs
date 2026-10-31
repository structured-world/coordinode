use super::*;

use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_storage::engine::transaction::{CommitContext, CommitError};

use crate::index_def::{IndexMaintenance, Interpretation, PartState};

/// A membership both interpretations agree on.
fn settled(labels: &[LabelId]) -> LabelMembership<'_> {
    LabelMembership::settled(labels)
}

/// A membership the interpretations disagree on.
fn both<'a>(in_force: &'a [LabelId], candidate: &'a [LabelId]) -> LabelMembership<'a> {
    LabelMembership {
        in_force,
        candidate,
    }
}

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

/// Two versions of one temporal node hold one unique value. One version
/// leaving it keeps the value held by the other, and keeps the value's
/// ownership revision: a transaction taking the value for another node from
/// a view without that change meets it at commit as a write conflict,
/// whichever of the two commits first, in both profiles. Removing the
/// revision with the leaving owner left the value held with no revision.
///
/// Store level only: the takers stage their entries directly, without the
/// statement's check that refuses a value another owner holds, which is
/// what lets the taker commit first here.
#[test]
fn one_of_several_owners_leaving_keeps_the_values_revision() {
    use coordinode_core::index::encoding::{encode_tuple, encode_unique_claim_key};
    let version = |valid_from| EntryOwner::version(1, valid_from);
    for (index, taker_first) in [
        (
            bound(IndexDescriptor::btree("e", "User", "email").unique(), 1),
            false,
        ),
        (
            bound(IndexDescriptor::btree("e", "User", "email").unique(), 1),
            true,
        ),
        (
            derived(bound(
                IndexDescriptor::btree("e", "User", "email").unique(),
                1,
            )),
            false,
        ),
        (
            derived(bound(
                IndexDescriptor::btree("e", "User", "email").unique(),
                1,
            )),
            true,
        ),
    ] {
        let what = format!(
            "{:?}, taker first: {taker_first}",
            index.maintenance.profile
        );
        let fx = open_engine();
        let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
        let store = LocalIndexStore::new(&fx.engine);
        let mut t = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
        for valid_from in [100, 200] {
            store
                .stage_membership(
                    &mut t,
                    &index,
                    &no_fields,
                    version(valid_from),
                    None,
                    Some(&s("v")),
                )
                .unwrap();
        }
        commit(&mut t).unwrap();

        let mut leaving = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
        let mut taking = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
        store
            .stage_membership(
                &mut leaving,
                &index,
                &no_fields,
                version(100),
                Some(&s("v")),
                Some(&s("w")),
            )
            .unwrap();
        enter(&store, &mut taking, &index, &s("v"), id(2));
        let (first, second) = if taker_first {
            (&mut taking, &mut leaving)
        } else {
            (&mut leaving, &mut taking)
        };
        commit(first).unwrap();
        assert!(
            matches!(commit(second), Err(CommitError::Conflict(_))),
            "{what}: the second commit is refused as a write conflict"
        );

        let claim = encode_unique_claim_key(1, &encode_tuple(&s("v")).unwrap());
        assert!(
            fx.engine.get(Partition::Idx, &claim).unwrap().is_some(),
            "{what}: the value's revision stays while it is held"
        );
        let mut t = Transaction::begin(&fx.engine, Some(&oracle), oracle.next());
        assert_eq!(
            store
                .unique_conflict(&mut t, &index, &s("v"), id(2))
                .unwrap(),
            Some(id(1)),
            "{what}: the other version still holds the value"
        );
    }
}

/// The key buffer a fold step rewrites into is charged exactly the growth
/// of its capacity, before it grows, for keys longer, shorter and longer
/// again than what it held: the bytes charged always equal the capacity.
/// Room counted past the bytes the buffer still held would leave the
/// capacity above what was charged.
#[test]
fn a_rewritten_key_buffer_is_charged_its_whole_growth() {
    let mut buffer: Vec<u8> = Vec::new();
    let mut charged = 0usize;
    for len in [10, 20, 15, 30, 30, 5, 64] {
        super::fit_buffer(&mut buffer, len, |more| {
            charged += more;
            Ok::<(), ()>(())
        })
        .unwrap();
        buffer.extend(core::iter::repeat_n(b'k', len));
        assert_eq!(buffer.len(), len);
        assert_eq!(charged, buffer.capacity(), "after a key of {len} bytes");
    }
    // A refusal leaves the buffer as small as it was.
    let before = buffer.capacity();
    assert!(super::fit_buffer(&mut buffer, before + 1, |_| Err(())).is_err());
    assert_eq!(buffer.capacity(), before);
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

/// Clearing a generation in a transaction removes every entry of it, a
/// unique index's included (one shape for both), and no entry of another
/// generation, the neighbouring numbers included, when the transaction
/// commits and not before.
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
    // The two indexes share the generation, so their entries are one range.
    assert_eq!(
        store.scan_entry_ids(&mut t, &plain).unwrap(),
        vec![id(1), id(2)],
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

/// The namespace has one label lookup: a second publication finds it and
/// stages nothing, and two racing ones commit one, the other refused by
/// the marker's condition rather than publishing a second lookup.
#[test]
fn the_label_lookup_is_published_once() {
    let fx = open_engine();
    let engine = &fx.engine;
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let store = LocalIndexStore::new(engine);
    assert!(store.label_lookup().unwrap().is_none(), "none before");

    let mut a = Transaction::begin(engine, Some(&oracle), oracle.next());
    let mut b = Transaction::begin(engine, Some(&oracle), oracle.next());
    let published = store
        .publish_label_lookup_txn(&mut a, IndexDescriptor::label_lookup())
        .unwrap()
        .expect("the first publishes");
    store
        .publish_label_lookup_txn(&mut b, IndexDescriptor::label_lookup())
        .unwrap()
        .expect("a racer read it absent too");
    commit(&mut a).unwrap();
    assert!(commit(&mut b).is_err(), "the second lookup must not commit");

    let found = store.label_lookup().unwrap().expect("published");
    assert_eq!(found, published);
    assert_eq!(found.index_type, crate::IndexType::Labels);
    assert!(found.name.is_none());
    let mut c = Transaction::begin(engine, Some(&oracle), oracle.next());
    assert!(
        store
            .publish_label_lookup_txn(&mut c, IndexDescriptor::label_lookup())
            .unwrap()
            .is_none(),
        "one is published: nothing staged"
    );
}

/// The nodes holding a label, as a transaction sees them: postings move only
/// for labels that change, the transaction's own staged postings are seen,
/// other labels, shards and generations are not, and a walk resumes after a
/// node without reading what came before it.
#[test]
fn label_members_follow_staged_postings() {
    use core::ops::ControlFlow;
    let fx = open_engine();
    let engine = &fx.engine;
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let store = LocalIndexStore::new(engine);
    let lookup =
        IndexDescriptor::label_lookup().bind(IndexId::from_raw(1), GenerationId::from_raw(4));
    let other_generation =
        IndexDescriptor::label_lookup().bind(IndexId::from_raw(1), GenerationId::from_raw(5));
    let label = |raw| coordinode_core::schema::label_id::LabelId::new(raw).expect("nonzero");
    let (user, admin) = (label(10), label(11));

    let mut t = Transaction::begin(engine, Some(&oracle), oracle.next());
    for n in [3, 1, 2] {
        store
            .stage_label_postings_txn(&mut t, &lookup, 0, id(n), settled(&[]), settled(&[user]))
            .unwrap();
    }
    store
        .stage_label_postings_txn(
            &mut t,
            &lookup,
            0,
            id(2),
            settled(&[user]),
            settled(&[user, admin]),
        )
        .unwrap();
    store
        .stage_label_postings_txn(&mut t, &lookup, 1, id(9), settled(&[]), settled(&[user]))
        .unwrap();
    store
        .stage_label_postings_txn(
            &mut t,
            &other_generation,
            0,
            id(7),
            settled(&[]),
            settled(&[user]),
        )
        .unwrap();
    commit(&mut t).unwrap();

    let members = |txn: &Transaction,
                   label: coordinode_core::schema::label_id::LabelId,
                   after: Option<u64>,
                   stop: usize| {
        let mut out = Vec::new();
        store
            .label_members(
                txn,
                LabelScope {
                    lookup: &lookup,
                    shard: 0,
                    label,
                    interpretation: Interpretation::InForce,
                },
                after.map(id),
                &budget(),
                |node| {
                    out.push(node.as_raw());
                    Ok::<_, StoreError>(if out.len() == stop {
                        ControlFlow::Break(())
                    } else {
                        ControlFlow::Continue(())
                    })
                },
            )
            .unwrap();
        out
    };
    let mut t = Transaction::begin(engine, Some(&oracle), oracle.next());
    assert_eq!(
        members(&t, user, None, usize::MAX),
        vec![1, 2, 3],
        "NodeId order"
    );
    assert_eq!(members(&t, admin, None, usize::MAX), vec![2]);
    assert_eq!(
        members(&t, label(99), None, usize::MAX),
        Vec::<u64>::new(),
        "an empty label"
    );
    assert_eq!(
        members(&t, user, Some(1), usize::MAX),
        vec![2, 3],
        "resumed after 1"
    );
    assert_eq!(
        members(&t, user, Some(u64::MAX), usize::MAX),
        Vec::<u64>::new()
    );
    assert_eq!(
        members(&t, user, None, 2),
        vec![1, 2],
        "a break stops the walk"
    );

    // Node 2 moves from Admin and User to User alone; node 4 joins User;
    // node 1 leaves it. A label held before and after writes nothing.
    store
        .stage_label_postings_txn(
            &mut t,
            &lookup,
            0,
            id(2),
            settled(&[user, admin]),
            settled(&[user]),
        )
        .unwrap();
    store
        .stage_label_postings_txn(&mut t, &lookup, 0, id(4), settled(&[]), settled(&[user]))
        .unwrap();
    store
        .stage_label_postings_txn(&mut t, &lookup, 0, id(1), settled(&[user]), settled(&[]))
        .unwrap();
    assert_eq!(
        members(&t, user, None, usize::MAX),
        vec![2, 3, 4],
        "own writes seen"
    );
    assert!(members(&t, admin, None, usize::MAX).is_empty());
    let other = Transaction::begin(engine, Some(&oracle), oracle.next());
    assert_eq!(
        members(&other, user, None, usize::MAX),
        vec![1, 2, 3],
        "not outside the transaction"
    );
    assert!(
        t.buffered(
            Partition::Idx,
            &coordinode_core::index::encoding::encode_label_posting(lookup.generation, 0, user, 2)
        )
        .is_none(),
        "a label kept writes no posting"
    );
}

/// While an implication change is certified, a label's postings agreed on by
/// both interpretations stay in the lookup's own scope and only the
/// difference is kept apart: the facts in force read their own scope and the
/// retired part, the candidate its own scope and the gained part, each
/// without a node of the other. Put in force, the facts read the gained
/// part; withdrawn, the retired one. Merging scopes keeps NodeId order past
/// a page, a resume and a break; other labels never move.
#[test]
fn a_label_change_keeps_only_its_difference_apart() {
    use core::ops::ControlFlow;
    let fx = open_engine();
    let engine = &fx.engine;
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let store = LocalIndexStore::new(engine);
    let label = |raw| coordinode_core::schema::label_id::LabelId::new(raw).expect("nonzero");
    let (thing, other) = (label(10), label(11));
    let operation = GenerationId::from_raw(20);
    let (gained, retired) = (GenerationId::from_raw(21), GenerationId::from_raw(22));
    let mut lookup =
        IndexDescriptor::label_lookup().bind(IndexId::from_raw(1), GenerationId::from_raw(4));
    let members = |txn: &Transaction,
                   def: &IndexDefinition,
                   label,
                   interpretation,
                   after: Option<u64>,
                   stop: usize| {
        let mut out = Vec::new();
        store
            .label_members(
                txn,
                LabelScope {
                    lookup: def,
                    shard: 0,
                    label,
                    interpretation,
                },
                after.map(id),
                &budget(),
                |node| {
                    out.push(node.as_raw());
                    Ok::<_, StoreError>(if out.len() == stop {
                        ControlFlow::Break(())
                    } else {
                        ControlFlow::Continue(())
                    })
                },
            )
            .unwrap();
        out
    };
    // Nodes 1..=1000 carry Thing; the odd ones are of the type the change
    // removes Thing from. Nodes 2001.. are of the type it adds Thing to.
    let mut t = Transaction::begin(engine, Some(&oracle), oracle.next());
    for n in 1..=1000 {
        store
            .stage_label_postings_txn(
                &mut t,
                &lookup,
                0,
                id(n),
                settled(&[]),
                settled(&[thing, other]),
            )
            .unwrap();
    }
    commit(&mut t).unwrap();

    lookup.propose_parts(thing, operation, gained, retired);
    assert!(lookup.has_proposed());
    assert_eq!(lookup.proposed_operation(thing), Some(operation));
    assert_eq!(lookup.proposed_operation(other), None);
    let mut t = Transaction::begin(engine, Some(&oracle), oracle.next());
    // The losing carriers move to the retired part, as the certification's
    // preparation moves them.
    for n in (1..=1000).step_by(2) {
        store
            .stage_label_postings_txn(
                &mut t,
                &lookup,
                0,
                id(n),
                settled(&[thing, other]),
                both(&[thing, other], &[other]),
            )
            .unwrap();
    }
    // The gaining carriers write the gained part, interleaved by NodeId.
    for n in (1..=1000).step_by(2).map(|n| n + 2000) {
        store
            .stage_label_postings_txn(&mut t, &lookup, 0, id(n), settled(&[]), both(&[], &[thing]))
            .unwrap();
    }
    commit(&mut t).unwrap();
    let mut t = Transaction::begin(engine, Some(&oracle), oracle.next());
    let all: Vec<u64> = (1..=1000).collect();
    let evens: Vec<u64> = (2..=1000).step_by(2).collect();
    let mut kept_and_gained = evens.clone();
    kept_and_gained.extend((1..=1000).step_by(2).map(|n| n + 2000));
    let in_force = Interpretation::InForce;
    let candidate = Interpretation::Candidate(operation);
    assert_eq!(members(&t, &lookup, thing, in_force, None, usize::MAX), all);
    assert_eq!(
        members(&t, &lookup, thing, candidate, None, usize::MAX),
        kept_and_gained,
        "past several merge pages"
    );
    assert_eq!(
        members(&t, &lookup, thing, candidate, Some(999), 3),
        vec![1000, 2001, 2003],
        "a resume and a break across scopes"
    );
    assert_eq!(
        members(&t, &lookup, other, candidate, None, usize::MAX),
        all,
        "a label the change leaves alone"
    );
    assert!(
        t.get(
            Partition::Idx,
            &coordinode_core::index::encoding::encode_label_posting(lookup.generation, 0, thing, 1)
        )
        .unwrap()
        .is_none(),
        "a retired posting left the lookup's own scope"
    );

    let mut withdrawn = lookup.clone();
    assert!(withdrawn.conclude_parts(operation, PartState::Withdrawn));
    assert_eq!(
        members(&t, &withdrawn, thing, in_force, None, usize::MAX),
        all
    );

    assert!(lookup.conclude_parts(operation, PartState::InForce));
    assert!(!lookup.has_proposed());
    assert_eq!(
        members(&t, &lookup, thing, in_force, None, usize::MAX),
        kept_and_gained,
        "in force"
    );
    // A writer after the change judges by one interpretation; a node leaving
    // Thing is cleared from whichever scope holds its posting.
    let mut w = Transaction::begin(engine, Some(&oracle), oracle.next());
    for n in [2, 2001] {
        store
            .stage_label_postings_txn(&mut w, &lookup, 0, id(n), settled(&[thing]), settled(&[]))
            .unwrap();
    }
    assert_eq!(members(&w, &lookup, thing, in_force, None, 2), vec![4, 6]);
    assert_eq!(
        members(&w, &lookup, thing, in_force, Some(1000), 1),
        vec![2003]
    );

    lookup.drop_part(retired);
    let bytes = rmp_serde::to_vec(&lookup).unwrap();
    assert_eq!(
        rmp_serde::from_slice::<IndexDefinition>(&bytes).unwrap(),
        lookup
    );
}

/// A fold moving a gained posting into the lookup's own scope races a writer
/// that takes the node out of the label after the change was put in force.
/// The writer clears the posting from every scope of the label, so both
/// write the gained key: whichever commits second is refused, and the node
/// never comes back under the label from a posting the fold read before.
/// Run in both commit orders.
#[test]
fn a_fold_never_brings_back_a_posting_a_writer_removed() {
    use core::ops::ControlFlow;
    let label = |raw| coordinode_core::schema::label_id::LabelId::new(raw).expect("nonzero");
    let thing = label(10);
    let operation = GenerationId::from_raw(20);
    let (gained, retired) = (GenerationId::from_raw(21), GenerationId::from_raw(22));
    for writer_first in [true, false] {
        let fx = open_engine();
        let engine = &fx.engine;
        let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
        let store = LocalIndexStore::new(engine);
        let mut lookup =
            IndexDescriptor::label_lookup().bind(IndexId::from_raw(1), GenerationId::from_raw(4));
        lookup.propose_parts(thing, operation, gained, retired);
        let mut t = Transaction::begin(engine, Some(&oracle), oracle.next());
        store
            .stage_label_postings_txn(&mut t, &lookup, 0, id(7), settled(&[]), both(&[], &[thing]))
            .unwrap();
        commit(&mut t).unwrap();
        assert!(lookup.conclude_parts(operation, PartState::InForce));

        let mut fold = Transaction::begin(engine, Some(&oracle), oracle.next());
        assert!(
            store
                .move_scope_page_txn(&mut fold, gained, lookup.generation, 16)
                .unwrap(),
            "one posting, moved in one page"
        );
        let mut writer = Transaction::begin(engine, Some(&oracle), oracle.next());
        store
            .stage_label_postings_txn(
                &mut writer,
                &lookup,
                0,
                id(7),
                settled(&[thing]),
                settled(&[]),
            )
            .unwrap();
        let (first, second) = if writer_first {
            (&mut writer, &mut fold)
        } else {
            (&mut fold, &mut writer)
        };
        commit(first).unwrap();
        assert!(
            commit(second).is_err(),
            "writer first: {writer_first}: the second of the two must be refused"
        );

        let mut folded = lookup.clone();
        folded.drop_part(gained);
        folded.drop_part(retired);
        let read = Transaction::begin(engine, Some(&oracle), oracle.next());
        let mut found = Vec::new();
        for def in [&lookup, &folded] {
            store
                .label_members(
                    &read,
                    LabelScope {
                        lookup: def,
                        shard: 0,
                        label: thing,
                        interpretation: Interpretation::InForce,
                    },
                    None,
                    &budget(),
                    |node| {
                        found.push(node.as_raw());
                        Ok::<_, StoreError>(ControlFlow::Continue(()))
                    },
                )
                .unwrap();
        }
        assert_eq!(
            found,
            if writer_first {
                Vec::<u64>::new()
            } else {
                vec![7, 7]
            },
            "writer first: {writer_first}"
        );
    }
}

/// While a change is certified, every posting moved into or out of its parts
/// leaves what the change adds to the label's count pending: each placed
/// once however often a preparation step and a writer place it, each taken
/// out once. Settled, it lands on the counter when put in force and is
/// dropped when withdrawn, so the counter answers the count of the facts in
/// force at every step without reading the label's postings.
#[test]
fn a_change_leaves_its_count_pending_until_it_is_settled() {
    use crate::stats::{LocalStatsStore, StatsStore as _, label_count_key};
    let label = |raw| coordinode_core::schema::label_id::LabelId::new(raw).expect("nonzero");
    let thing = label(10);
    let operation = GenerationId::from_raw(20);
    let (gained, retired) = (GenerationId::from_raw(21), GenerationId::from_raw(22));
    for put_in_force in [true, false] {
        let fx = open_engine();
        let engine = &fx.engine;
        let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
        let store = LocalIndexStore::new(engine);
        let mut lookup =
            IndexDescriptor::label_lookup().bind(IndexId::from_raw(1), GenerationId::from_raw(4));
        // Four carriers in force, counted.
        let mut t = Transaction::begin(engine, Some(&oracle), oracle.next());
        for n in 1..=4 {
            store
                .stage_label_postings_txn(
                    &mut t,
                    &lookup,
                    0,
                    id(n),
                    settled(&[]),
                    settled(&[thing]),
                )
                .unwrap();
            LocalStatsStore.label_added(&mut t, "Thing");
        }
        commit(&mut t).unwrap();
        lookup.propose_parts(thing, operation, gained, retired);
        let pending = || {
            let mut t = Transaction::begin(engine, Some(&oracle), oracle.next());
            LocalStatsStore
                .pending_count(&mut t, operation.as_raw(), thing)
                .unwrap()
        };

        // Nodes 1 and 2 lose Thing; 1 is moved twice, as a retried step would.
        let mut t = Transaction::begin(engine, Some(&oracle), oracle.next());
        for n in [1, 2, 1] {
            store
                .stage_label_postings_txn(
                    &mut t,
                    &lookup,
                    0,
                    id(n),
                    settled(&[thing]),
                    both(&[thing], &[]),
                )
                .unwrap();
        }
        // Nodes 7 and 8 gain it; 7 is placed by a writer and by presence.
        for n in [7, 8] {
            store
                .stage_label_postings_txn(
                    &mut t,
                    &lookup,
                    0,
                    id(n),
                    settled(&[]),
                    both(&[], &[thing]),
                )
                .unwrap();
        }
        store
            .stage_label_presence_txn(&mut t, &lookup, 0, id(7), both(&[], &[thing]))
            .unwrap();
        // Node 9 gains it and is deleted before the change concludes.
        store
            .stage_label_postings_txn(&mut t, &lookup, 0, id(9), settled(&[]), both(&[], &[thing]))
            .unwrap();
        store
            .stage_label_postings_txn(&mut t, &lookup, 0, id(9), both(&[], &[thing]), settled(&[]))
            .unwrap();
        // Node 2, retired, is deleted too: no longer counted by either.
        store
            .stage_label_postings_txn(&mut t, &lookup, 0, id(2), both(&[thing], &[]), settled(&[]))
            .unwrap();
        LocalStatsStore.label_removed(&mut t, "Thing");
        commit(&mut t).unwrap();
        // In force: 1, 3, 4 (2 deleted). Candidate: 3, 4, 7, 8.
        assert_eq!(pending(), 1, "candidate 4 against 3 in force");

        let state = if put_in_force {
            PartState::InForce
        } else {
            PartState::Withdrawn
        };
        assert!(lookup.conclude_parts(operation, state));
        let mut t = Transaction::begin(engine, Some(&oracle), oracle.next());
        LocalStatsStore
            .settle_pending(&mut t, (operation.as_raw(), thing, "Thing"), put_in_force)
            .unwrap();
        commit(&mut t).unwrap();
        assert_eq!(pending(), 0, "settled once");
        let counted = engine
            .get(Partition::Counter, &label_count_key("Thing"))
            .unwrap()
            .map(|b| coordinode_storage::engine::merge::decode_counter(&b).unwrap());
        assert_eq!(
            counted,
            Some(if put_in_force { 4 } else { 3 }),
            "in force: {put_in_force}"
        );
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
