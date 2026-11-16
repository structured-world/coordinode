//! The change-feed context a commit seals: which units carry it, the labels
//! it keeps for nodes whose rows cannot prove them, and how it travels with
//! a parked transaction.

use std::sync::{Arc, Mutex};

use coordinode_core::graph::node::{NodeId, NodeRecord, encode_node_key};
use coordinode_core::schema::label_id::LabelId;
use coordinode_core::txn::proposal::{
    EFFECTS_INTERPRETATION, NodeBasis, ProposalError, ProposalIdGenerator, ProposalOutcome,
    ProposalPipeline, RaftProposal, UnitEffects,
};
use coordinode_core::txn::write_concern::WriteConcern;
use tempfile::TempDir;

use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use crate::engine::local_pipeline::LocalProposalPipeline;

fn test_engine() -> (Arc<StorageEngine>, Arc<TimestampOracle>, TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path().to_string_lossy().as_ref(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let oracle = Arc::new(TimestampOracle::new());
    let engine = Arc::new(StorageEngine::open_with_oracle(&config, oracle.clone()).expect("open"));
    (engine, oracle, dir)
}

fn txn<'a>(engine: &'a StorageEngine, oracle: &'a TimestampOracle) -> Transaction<'a> {
    let snap = engine.snapshot();
    Transaction::new(engine, Some(oracle), Timestamp::from_raw(snap), Some(snap))
}

/// Applies each proposal locally and keeps its operations.
struct Recording<'a> {
    inner: LocalProposalPipeline<'a>,
    units: Mutex<Vec<Vec<Mutation>>>,
}

impl ProposalPipeline for Recording<'_> {
    fn propose_and_wait(&self, proposal: &RaftProposal) -> Result<ProposalOutcome, ProposalError> {
        self.units
            .lock()
            .expect("units")
            .push(proposal.mutations.clone());
        self.inner.propose_and_wait(proposal)
    }
}

/// Commit `txn` through a recording pipeline and return the unit it sealed.
fn sealed(engine: &StorageEngine, mut txn: Transaction<'_>) -> Vec<Mutation> {
    let recording = Recording {
        inner: LocalProposalPipeline::new(engine),
        units: Mutex::new(Vec::new()),
    };
    let ids = ProposalIdGenerator::new();
    let wc = WriteConcern::default();
    txn.commit(&CommitContext {
        write_concern: &wc,
        pipeline: Some(&recording),
        id_gen: Some(&ids),
    })
    .expect("commit");
    let mut units = recording.units.into_inner().expect("units");
    units.pop().expect("one unit")
}

fn effects_of(unit: &[Mutation]) -> Option<&UnitEffects> {
    unit.iter().find_map(|m| match m {
        Mutation::Effects(effects) => Some(effects),
        _ => None,
    })
}

fn label(raw: u32) -> LabelId {
    LabelId::new(raw).expect("nonzero")
}

fn record(labels: &[u32]) -> Vec<u8> {
    NodeRecord::with_labels(labels.iter().map(|&l| label(l)).collect())
        .to_msgpack()
        .expect("encode")
}

fn key(id: u64) -> Vec<u8> {
    encode_node_key(0, NodeId::from_raw(id))
}

/// Seed node `id` storing `labels`.
fn seed(engine: &StorageEngine, oracle: &TimestampOracle, id: u64, labels: &[u32]) {
    let mut t = txn(engine, oracle);
    t.put(Partition::Node, &key(id), &record(labels))
        .expect("stage");
    sealed(engine, t);
}

/// A unit that changes application data carries the context, after its rows
/// and before nothing else; an ordinary put needs no entry, as its record
/// states its labels. A unit writing only index entries and counters has
/// nothing to project and carries none.
#[test]
fn a_unit_changing_application_data_carries_the_context() {
    let (engine, oracle, _d) = test_engine();
    let mut t = txn(&engine, &oracle);
    t.put(Partition::Node, &key(1), &record(&[3]))
        .expect("stage");
    let unit = sealed(&engine, t);
    assert!(
        matches!(unit.last(), Some(Mutation::Effects(_))),
        "{unit:?}"
    );
    let effects = effects_of(&unit).expect("context");
    assert_eq!(effects.interpretation, EFFECTS_INTERPRETATION);
    assert!(effects.is_empty(), "{effects:?}");

    let mut t = txn(&engine, &oracle);
    t.put(Partition::Idx, b"idx:only", b"").expect("stage");
    t.push_counter_delta(b"counter:only", 1);
    let unit = sealed(&engine, t);
    assert!(effects_of(&unit).is_none(), "{unit:?}");
}

/// A tombstone carries no labels: the unit keeps those the node stored, so
/// a subscriber of any of them gets the removal after the row is gone. A
/// node created and removed in one attempt stored none before it.
#[test]
fn a_removed_node_keeps_the_labels_it_stored() {
    let (engine, oracle, _d) = test_engine();
    seed(&engine, &oracle, 1, &[3, 9]);

    let mut t = txn(&engine, &oracle);
    t.note_node_basis(&key(1), None).expect("basis");
    t.delete(Partition::Node, &key(1)).expect("stage");
    t.put(Partition::Node, &key(2), &record(&[3]))
        .expect("stage");
    t.note_node_basis(&key(2), None).expect("basis");
    t.delete(Partition::Node, &key(2)).expect("stage");
    let unit = sealed(&engine, t);
    assert_eq!(
        effects_of(&unit).expect("context").nodes,
        vec![
            NodeBasis {
                key: key(1),
                labels: vec![label(3), label(9)],
            },
            NodeBasis {
                key: key(2),
                labels: Vec::new(),
            },
        ]
    );
}

/// The labels a writer read stand while the attempt has not changed the
/// node: that read came from the snapshot. Once an earlier write of the
/// attempt changed it, the writer's read saw that write, and the labels
/// before the attempt are read at the snapshot instead. The first record of
/// a node stands, so a node relabelled twice keeps its first labels.
#[test]
fn the_labels_before_are_the_records_before_the_attempt() {
    let (engine, oracle, _d) = test_engine();
    seed(&engine, &oracle, 1, &[3]);
    seed(&engine, &oracle, 2, &[3]);

    let mut t = txn(&engine, &oracle);
    // Read from the snapshot, then relabelled twice.
    t.note_node_basis(&key(1), Some(&[label(3)]))
        .expect("basis");
    t.put(Partition::Node, &key(1), &record(&[3, 5]))
        .expect("stage");
    t.note_node_basis(&key(1), Some(&[label(3), label(5)]))
        .expect("basis");
    t.put(Partition::Node, &key(1), &record(&[3, 6]))
        .expect("stage");
    // Written whole first, then relabelled: the writer's read is its own.
    t.put(Partition::Node, &key(2), &record(&[3, 7]))
        .expect("stage");
    t.note_node_basis(&key(2), Some(&[label(3), label(7)]))
        .expect("basis");
    t.put(Partition::Node, &key(2), &record(&[3, 8]))
        .expect("stage");
    let unit = sealed(&engine, t);
    assert_eq!(
        effects_of(&unit).expect("context").nodes,
        vec![
            NodeBasis {
                key: key(1),
                labels: vec![label(3)],
            },
            NodeBasis {
                key: key(2),
                labels: vec![label(3)],
            },
        ]
    );
}

/// Point deletes of adjacent nodes are sealed as one range removal; the unit
/// names it a run, so its keys are the deleted nodes and nothing else, and
/// each keeps the labels it stored. A removal of a whole range stays
/// unnamed.
#[test]
fn a_coalesced_run_of_deletes_is_named_a_run() {
    let (engine, oracle, _d) = test_engine();
    for id in 10..15 {
        seed(&engine, &oracle, id, &[3]);
    }
    let mut t = txn(&engine, &oracle);
    t.remove_range(Partition::EdgeProp, b"edgeprop:GONE:", b"edgeprop:GONE;")
        .expect("stage");
    for id in 10..15 {
        t.note_node_basis(&key(id), None).expect("basis");
        t.delete(Partition::Node, &key(id)).expect("stage");
    }
    let unit = sealed(&engine, t);
    let ranges: Vec<usize> = unit
        .iter()
        .enumerate()
        .filter(|(_, m)| matches!(m, Mutation::RemoveRange { .. }))
        .map(|(at, _)| at)
        .collect();
    assert_eq!(ranges.len(), 2, "{unit:?}");
    let effects = effects_of(&unit).expect("context");
    assert_eq!(effects.runs, vec![ranges[1] as u32]);
    assert_eq!(effects.nodes.len(), 5);
}

/// A document delta carries no labels: the node's are recorded as it is
/// staged.
#[test]
fn a_document_delta_records_the_labels_it_lands_under() {
    use coordinode_core::graph::doc_delta::{DocDelta, PathTarget};
    use coordinode_core::graph::types::Value;
    let (engine, oracle, _d) = test_engine();
    seed(&engine, &oracle, 1, &[4]);

    let mut t = txn(&engine, &oracle);
    let operand = DocDelta::SetProperty {
        target: PathTarget::PropField(2),
        key: None,
        value: Value::Int(1),
    }
    .encode()
    .expect("encode");
    t.push_node_delta(key(1), operand).expect("stage");
    let unit = sealed(&engine, t);
    assert_eq!(
        effects_of(&unit).expect("context").nodes,
        vec![NodeBasis {
            key: key(1),
            labels: vec![label(4)],
        }]
    );
}

/// The context is part of a parked transaction's state: a node one
/// statement removes is still routed when a later statement commits.
#[test]
fn a_parked_transaction_keeps_its_context() {
    let (engine, oracle, _d) = test_engine();
    seed(&engine, &oracle, 1, &[3]);

    let mut t = txn(&engine, &oracle);
    t.note_node_basis(&key(1), None).expect("basis");
    t.delete(Partition::Node, &key(1)).expect("stage");
    t.note_removed_edge(b"edgeprop:KNOWS:pair");
    let parked = t.take_state();
    drop(t);
    let resumed = Transaction::resume(&engine, Some(&oracle), parked);
    let unit = sealed(&engine, resumed);
    let effects = effects_of(&unit).expect("context");
    assert_eq!(
        effects.nodes,
        vec![NodeBasis {
            key: key(1),
            labels: vec![label(3)],
        }]
    );
    assert_eq!(effects.removed_edges, vec![b"edgeprop:KNOWS:pair".to_vec()]);
}
