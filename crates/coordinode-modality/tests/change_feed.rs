//! The change feed's projection of units sealed by the stores' own writers:
//! what each writer states is enough to route and name every effect, with
//! no read of the rows it changed.

#![allow(clippy::expect_used)]

use std::borrow::Cow;
use std::sync::Mutex;

use coordinode_core::graph::edge::{EdgeProperties, encode_edgeprop_key};
use coordinode_core::graph::node::{NodeId, NodeRecord, encode_node_key};
use coordinode_core::graph::types::Value;
use coordinode_core::schema::label_id::LabelId;
use coordinode_core::txn::proposal::{
    Mutation, ProposalError, ProposalIdGenerator, ProposalOutcome, ProposalPipeline, RaftProposal,
};
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_modality::{EdgeStore as _, LocalEdgeStore, LocalNodeStore, NodeStore as _};
use coordinode_storage::engine::change_feed::{Action, Delivery, Target, project_unit};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::local_pipeline::LocalProposalPipeline;
use coordinode_storage::engine::transaction::{CommitContext, Transaction};

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

struct Db {
    _fx: coordinode_test_fixtures::EngineFixture,
    oracle: TimestampOracle,
}

impl Db {
    fn open() -> Self {
        Self {
            _fx: coordinode_test_fixtures::engine_for_logic(),
            oracle: TimestampOracle::resume_from(Timestamp::from_raw(1)),
        }
    }

    fn engine(&self) -> &StorageEngine {
        &self._fx.engine
    }

    /// Run `stage` in one transaction and return the unit its commit sealed.
    fn unit(&self, stage: impl FnOnce(&mut Transaction<'_>)) -> Vec<Mutation> {
        let engine = self.engine();
        let mut txn = Transaction::begin(engine, Some(&self.oracle), self.oracle.next());
        stage(&mut txn);
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
        recording
            .units
            .into_inner()
            .expect("units")
            .pop()
            .expect("one unit")
    }
}

fn label(raw: u32) -> LabelId {
    LabelId::new(raw).expect("nonzero")
}

fn node(labels: &[u32]) -> NodeRecord {
    NodeRecord::with_labels(labels.iter().map(|&l| label(l)).collect())
}

fn id(raw: u64) -> NodeId {
    NodeId::from_raw(raw)
}

/// A node deleted by key through its store reaches the subscribers of every
/// label it stored, after the deletion committed: the store recorded them as
/// it deleted, so the projection reads no row.
#[test]
fn a_node_deleted_by_key_reaches_its_labels_subscribers() {
    let db = Db::open();
    db.unit(|t| {
        LocalNodeStore
            .put(t, 0, id(7), &node(&[3, 9]))
            .expect("put");
    });
    let unit = db.unit(|t| LocalNodeStore.delete(t, 0, id(7)).expect("delete"));
    let events = project_unit(&unit).expect("project");
    assert_eq!(events.len(), 1, "{events:?}");
    assert_eq!(events[0].action, Action::Remove);
    assert_eq!(
        events[0].target,
        Target::Node(Cow::Borrowed(encode_node_key(0, id(7)).as_slice()))
    );
    let routing = events[0].routing.as_ref().expect("routing");
    assert_eq!(routing.delivery(|l| l == label(9)), Delivery::Leave);
    assert_eq!(routing.delivery(|l| l == label(3)), Delivery::Leave);
    assert_eq!(routing.delivery(|l| l == label(4)), Delivery::None);
}

/// A node created and deleted in one transaction was in no subscriber's
/// scope: its removal reaches none of them.
#[test]
fn a_node_created_and_deleted_at_once_leaves_no_scope() {
    let db = Db::open();
    let unit = db.unit(|t| {
        LocalNodeStore.put(t, 0, id(8), &node(&[3])).expect("put");
        LocalNodeStore.delete(t, 0, id(8)).expect("delete");
    });
    let events = project_unit(&unit).expect("project");
    assert_eq!(events.len(), 1, "{events:?}");
    assert_eq!(events[0].action, Action::Remove);
    assert_eq!(
        events[0]
            .routing
            .as_ref()
            .expect("routing")
            .delivery(|l| l == label(3)),
        Delivery::None
    );
}

/// Writing a relationship is one presence event and one property event,
/// whatever both adjacency directions store; deleting it is one removal,
/// its property record with it; clearing the properties of one that stays
/// is a removal of the properties only.
#[test]
fn relationship_writers_project_presence_and_properties_apart() {
    let db = Db::open();
    for raw in [1, 2] {
        db.unit(|t| {
            LocalNodeStore.put(t, 0, id(raw), &node(&[3])).expect("put");
        });
    }
    let mut props = EdgeProperties::new();
    props.set(1, Value::Int(5));
    let created = db.unit(|t| {
        LocalEdgeStore
            .put_edge(t, "KNOWS", id(1), id(2), Some(&props))
            .expect("edge");
    });
    let pair = encode_edgeprop_key("KNOWS", id(1), id(2));
    let presence = Target::Relationship {
        edge_type: b"KNOWS",
        source: 1,
        target: 2,
    };
    let actions: Vec<(Action, Target<'_>)> = project_unit(&created)
        .expect("project")
        .into_iter()
        .map(|e| (e.action, e.target))
        .collect();
    assert_eq!(actions.len(), 2, "{actions:?}");
    assert!(actions.contains(&(Action::Put, presence.clone())));
    assert!(actions.contains(&(
        Action::Put,
        Target::RelationshipProperties(Cow::Borrowed(pair.as_slice()))
    )));

    let cleared = db.unit(|t| {
        LocalEdgeStore
            .delete_props(t, "KNOWS", id(1), id(2), None)
            .expect("clear");
    });
    let actions: Vec<(Action, Target<'_>)> = project_unit(&cleared)
        .expect("project")
        .into_iter()
        .map(|e| (e.action, e.target))
        .collect();
    assert_eq!(
        actions,
        vec![(
            Action::Remove,
            Target::RelationshipProperties(Cow::Borrowed(pair.as_slice()))
        )]
    );

    let deleted = db.unit(|t| {
        LocalEdgeStore
            .delete_edge(t, "KNOWS", id(1), id(2))
            .expect("delete");
    });
    let actions: Vec<(Action, Target<'_>)> = project_unit(&deleted)
        .expect("project")
        .into_iter()
        .map(|e| (e.action, e.target))
        .collect();
    assert_eq!(actions, vec![(Action::Remove, presence)]);
}

/// A node's forward list removed whole, with its peers unhooked as a detach
/// does, names each relationship it held, so each is removed once: both
/// endpoints deleted in one unit and a self-loop included.
#[test]
fn a_detached_nodes_relationships_are_each_removed_once() {
    let db = Db::open();
    db.unit(|t| {
        for raw in [1, 2] {
            LocalNodeStore.put(t, 0, id(raw), &node(&[3])).expect("put");
        }
        LocalEdgeStore
            .put_edge(t, "KNOWS", id(1), id(2), None)
            .expect("edge");
        LocalEdgeStore
            .put_edge(t, "KNOWS", id(1), id(1), None)
            .expect("loop");
    });
    let unit = db.unit(|t| {
        // Node 1 goes with its lists: each peer of its forward list is
        // unhooked, then the lists are purged.
        for peer in [1, 2] {
            LocalEdgeStore.unhook_from_target(t, "KNOWS", id(1), id(peer));
        }
        LocalEdgeStore.merge_remove_fwd(t, "KNOWS", id(1), 1);
        LocalEdgeStore
            .purge_adj(t, "KNOWS", id(1), true)
            .expect("purge out");
        LocalEdgeStore
            .purge_adj(t, "KNOWS", id(1), false)
            .expect("purge in");
        LocalNodeStore.delete(t, 0, id(1)).expect("delete 1");
        // Node 2 goes too: its in-list, which held 1, is purged with it.
        LocalEdgeStore
            .purge_adj(t, "KNOWS", id(2), false)
            .expect("purge 2");
        LocalNodeStore.delete(t, 0, id(2)).expect("delete 2");
    });
    let events = project_unit(&unit).expect("project");
    let removed: Vec<&Target<'_>> = events
        .iter()
        .filter(|e| matches!(e.target, Target::Relationship { .. }))
        .map(|e| &e.target)
        .collect();
    assert_eq!(
        removed,
        vec![
            &Target::Relationship {
                edge_type: b"KNOWS",
                source: 1,
                target: 1,
            },
            &Target::Relationship {
                edge_type: b"KNOWS",
                source: 1,
                target: 2,
            },
        ],
        "{events:?}"
    );
    assert!(
        events
            .iter()
            .filter(|e| matches!(e.target, Target::Relationship { .. }))
            .all(|e| e.action == Action::Remove)
    );
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(e.target, Target::Node(_)))
            .count(),
        2
    );
}
