//! Sealing a transaction's DERIVED work, and committing it.

use std::sync::Arc;

use coordinode_core::graph::node::{NodeId, encode_node_key, encode_temporal_node_key};
use coordinode_core::index::derive::{IndexInterpretation, KEY_CODEC, PropertyRef};
use coordinode_core::txn::timestamp::TimestampOracle;

use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use crate::engine::core::StorageEngine;
use crate::engine::partition::Partition;
use crate::engine::transaction::{CommitContext, Transaction};

fn binding() -> IndexBinding {
    IndexBinding {
        epoch: 1,
        interpretation: IndexInterpretation {
            codec: KEY_CODEC,
            name: "user_email".into(),
            unique: false,
            sparse: false,
            properties: vec![PropertyRef {
                field: Some(1),
                name: "email".into(),
            }],
            filter: None,
        },
    }
}

fn s(v: &str) -> Value {
    Value::String(v.into())
}

fn record(email: &str) -> Vec<u8> {
    let mut record = NodeRecord::new("User");
    record.set(1, s(email));
    record.to_msgpack().expect("record")
}

fn node_put(node: u64, email: &str) -> Mutation {
    Mutation::Put {
        partition: PartitionId::Node,
        key: encode_node_key(0, NodeId::from_raw(node)),
        value: record(email),
    }
}

/// Node `id`, not temporal, as the owner of its entries.
fn node(id: u64) -> EntryOwner {
    EntryOwner::node(id)
}

fn derived_of(mutations: &[Mutation]) -> Vec<&DerivedIndexWork> {
    mutations
        .iter()
        .filter_map(|m| match m {
            Mutation::Derive(work) => Some(work),
            _ => None,
        })
        .collect()
}

/// A node whose whole record is in the unit is derived from it, by its
/// position in the final list.
#[test]
fn a_whole_record_in_the_unit_is_the_source() {
    let mut ledger = DerivedLedger::default();
    ledger.stage(&binding(), node(5), None, Some(vec![s("a")]), []);
    let mut unit = vec![node_put(9, "other"), node_put(5, "a")];
    ledger.seal(&mut unit);
    let work = derived_of(&unit);
    assert_eq!(work.len(), 1);
    assert_eq!(work[0].new, DerivedSource::UnitRecord(1));
    assert_eq!(work[0].old, None);
}

/// Each version of a temporal node is its own change, derived from the
/// record put at that version's key: two versions of one node in one unit
/// seal two pieces of work, each naming its own record and valid_from.
#[test]
fn each_temporal_version_is_derived_from_its_own_record() {
    let version_put = |valid_from: i64, email: &str| Mutation::Put {
        partition: PartitionId::Node,
        key: encode_temporal_node_key(0, NodeId::from_raw(5), valid_from),
        value: record(email),
    };
    let mut ledger = DerivedLedger::default();
    let closed = EntryOwner::version(5, 100);
    let opened = EntryOwner::version(5, 200);
    ledger.stage(&binding(), closed, None, Some(vec![s("a")]), []);
    ledger.stage(&binding(), opened, None, Some(vec![s("b")]), []);
    let mut unit = vec![version_put(100, "a"), version_put(200, "b")];
    ledger.seal(&mut unit);
    let work = derived_of(&unit);
    assert_eq!(work.len(), 2, "one change per version: {work:?}");
    assert_eq!((work[0].node_id, work[0].valid_from), (5, Some(100)));
    assert_eq!(work[0].new, DerivedSource::UnitRecord(0));
    assert_eq!((work[1].node_id, work[1].valid_from), (5, Some(200)));
    assert_eq!(work[1].new, DerivedSource::UnitRecord(1));
}

/// A merge operand after the record, or a record that does not give the
/// membership the transaction saw, makes the membership travel as values.
#[test]
fn values_travel_when_the_record_is_not_final_or_disagrees() {
    let mut ledger = DerivedLedger::default();
    ledger.stage(&binding(), node(5), None, Some(vec![s("a")]), []);
    let mut merged = vec![
        node_put(5, "a"),
        Mutation::Merge {
            partition: PartitionId::Node,
            key: encode_node_key(0, NodeId::from_raw(5)),
            operand: vec![1],
        },
    ];
    ledger.clone().seal(&mut merged);
    assert_eq!(
        derived_of(&merged)[0].new,
        DerivedSource::Values(Some(vec![s("a")]))
    );

    let mut disagrees = vec![node_put(5, "not-a")];
    ledger.seal(&mut disagrees);
    assert_eq!(
        derived_of(&disagrees)[0].new,
        DerivedSource::Values(Some(vec![s("a")]))
    );
}

/// Later changes of the same node keep the first old membership, and a
/// change that ends where it began is left out.
#[test]
fn changes_fold_to_first_old_and_last_new() {
    let mut ledger = DerivedLedger::default();
    ledger.stage(
        &binding(),
        node(5),
        Some(vec![s("a")]),
        Some(vec![s("b")]),
        [],
    );
    ledger.stage(
        &binding(),
        node(5),
        Some(vec![s("b")]),
        Some(vec![s("c")]),
        [],
    );
    ledger.stage(
        &binding(),
        node(6),
        Some(vec![s("x")]),
        Some(vec![s("y")]),
        [],
    );
    ledger.stage(
        &binding(),
        node(6),
        Some(vec![s("y")]),
        Some(vec![s("x")]),
        [],
    );
    let mut unit = Vec::new();
    ledger.seal(&mut unit);
    let work = derived_of(&unit);
    assert_eq!(work.len(), 1, "node 6 ends where it began");
    assert_eq!(work[0].node_id, 5);
    assert_eq!(work[0].old, Some(vec![s("a")]));
    assert_eq!(work[0].new, DerivedSource::Values(Some(vec![s("c")])));
}

/// The work a unit seals must derive within the members' fan-out bound, or
/// every member would refuse the unit at application. The staged effects
/// bound the sealed ones from above; when they exceed the bound, the sealed
/// changes are counted exactly: statements that undo each other stage
/// effects but seal none.
#[test]
fn the_sealed_fan_out_is_bounded() {
    let key = |k: &str| k.as_bytes().to_vec();
    let within = {
        let mut ledger = DerivedLedger::default();
        ledger.stage(&binding(), node(5), None, Some(vec![s("a")]), [key("e1")]);
        ledger
    };
    assert_eq!(within.check_fan_out(1), Ok(()));

    let undone = {
        let mut ledger = DerivedLedger::default();
        ledger.stage(
            &binding(),
            node(5),
            Some(vec![s("a")]),
            Some(vec![s("b")]),
            [key("e1"), key("e2")],
        );
        ledger.stage(
            &binding(),
            node(5),
            Some(vec![s("b")]),
            Some(vec![s("a")]),
            [key("e2"), key("e1")],
        );
        ledger
    };
    assert_eq!(undone.check_fan_out(1), Ok(()), "nothing is sealed");

    let moved = {
        let mut ledger = DerivedLedger::default();
        ledger.stage(
            &binding(),
            node(5),
            Some(vec![s("a")]),
            Some(vec![s("b")]),
            [key("e1"), key("e2")],
        );
        ledger.stage(&binding(), node(6), None, Some(vec![s("c")]), [key("e3")]);
        ledger
    };
    assert_eq!(moved.check_fan_out(3), Ok(()));
    assert_eq!(moved.check_fan_out(2), Err(2));
}

/// Committing a transaction with DERIVED work journals the work instead of
/// the index entry, and the index partition holds the entry it derives. The
/// transaction reads its own entry before it commits.
#[test]
fn a_commit_journals_the_work_and_applies_the_entry() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let oracle = Arc::new(TimestampOracle::new());
    let engine = StorageEngine::open_embedded(&config, oracle.clone()).expect("open");
    let index = binding().interpretation;
    let effects = index.membership_effects(node(5), None, Some(&[s("a")]));
    let entry = effects[0].key.clone();

    let mut txn = Transaction::begin(&engine, Some(&oracle), oracle.next());
    let node_key = encode_node_key(0, NodeId::from_raw(5));
    txn.put(Partition::Node, &node_key, &record("a"))
        .expect("put");
    txn.stage_derived(&binding(), node(5), None, Some(vec![s("a")]), &effects)
        .expect("stage");
    assert!(
        txn.get(Partition::Idx, &entry).expect("own read").is_some(),
        "the transaction reads its own entry"
    );
    let wc = coordinode_core::txn::write_concern::WriteConcern::default();
    txn.commit(&CommitContext {
        write_concern: &wc,
        pipeline: None,
        id_gen: None,
        drain_buffer: None,
        nvme_write_buffer: None,
    })
    .expect("commit");

    assert!(engine.get(Partition::Idx, &entry).expect("get").is_some());
    let journal = engine
        .oplog_read_since(0)
        .expect("read journal")
        .expect("journal");
    let recorded = &journal.last().expect("the commit's entry").ops;
    assert!(
        matches!(
            recorded.as_slice(),
            [crate::oplog::entry::OplogOp::Unit { .. }]
        ),
        "the unit is journalled as one frame"
    );
    let ops = crate::oplog::convert::expand_units(recorded).expect("expand");
    assert!(
        ops.iter()
            .any(|op| matches!(op, crate::oplog::entry::OplogOp::Derive { .. })),
        "the work is journalled"
    );
    assert!(
        !ops.iter().any(|op| matches!(
            op,
            crate::oplog::entry::OplogOp::Insert { key, .. } if *key == entry
        )),
        "the entry itself is not"
    );
}
