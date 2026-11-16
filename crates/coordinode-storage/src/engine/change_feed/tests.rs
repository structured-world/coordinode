//! The projection of units as writers seal them: node routing before and
//! after, relationship presence against its physical forms, runs of deletes,
//! and the units that have no projection.

use std::borrow::Cow;

use coordinode_core::graph::edge::encode_edgeprop_key;
use coordinode_core::graph::node::{NodeId, NodeRecord, encode_node_key};
use coordinode_core::schema::label_id::LabelId;
use coordinode_core::txn::proposal::{
    EFFECTS_INTERPRETATION, Mutation, NodeBasis, PartitionId, UnitEffects,
};

use super::*;
use crate::engine::merge::{encode_add, encode_add_batch, encode_remove};

fn label(raw: u32) -> LabelId {
    LabelId::new(raw).expect("nonzero")
}

fn key(id: u64) -> Vec<u8> {
    encode_node_key(0, NodeId::from_raw(id))
}

fn record(labels: &[u32]) -> Vec<u8> {
    NodeRecord::with_labels(labels.iter().map(|&l| label(l)).collect())
        .to_msgpack()
        .expect("encode")
}

fn basis(id: u64, labels: &[u32]) -> NodeBasis {
    NodeBasis {
        key: key(id),
        labels: labels.iter().map(|&l| label(l)).collect(),
    }
}

fn context(nodes: Vec<NodeBasis>, removed_edges: Vec<Vec<u8>>, runs: Vec<u32>) -> Mutation {
    Mutation::Effects(UnitEffects {
        interpretation: EFFECTS_INTERPRETATION,
        nodes,
        removed_edges,
        runs,
    })
}

fn put(partition: PartitionId, key: Vec<u8>, value: Vec<u8>) -> Mutation {
    Mutation::Put {
        partition,
        key,
        value,
    }
}

fn delete(partition: PartitionId, key: Vec<u8>) -> Mutation {
    Mutation::Delete { partition, key }
}

fn adj(key: &str, operand: Vec<u8>) -> Mutation {
    let (head, id) = key.rsplit_once(':').expect("adj key");
    let mut key = head.as_bytes().to_vec();
    key.push(b':');
    key.extend_from_slice(&id.parse::<u64>().expect("id").to_be_bytes());
    Mutation::Merge {
        partition: PartitionId::Adj,
        key,
        operand,
    }
}

fn pair(edge_type: &str, source: u64, target: u64) -> Vec<u8> {
    encode_edgeprop_key(
        edge_type,
        NodeId::from_raw(source),
        NodeId::from_raw(target),
    )
}

fn relationship(edge_type: &'static str, source: u64, target: u64) -> Target<'static> {
    Target::Relationship {
        edge_type: edge_type.as_bytes(),
        source,
        target,
    }
}

fn labels(raw: &[u32]) -> Vec<LabelId> {
    raw.iter().map(|&l| label(l)).collect()
}

/// What each event does to what, in order, with its ordinal.
fn shape<'u>(events: &[LogicalEvent<'u>]) -> Vec<(u32, Action, Target<'u>)> {
    events
        .iter()
        .map(|e| (e.ordinal, e.action, e.target.clone()))
        .collect()
}

/// A put whose labels did not change routes by its record before and after:
/// it applies to a subscriber of any of its labels, and to nobody else.
#[test]
fn a_put_routes_by_its_record() {
    let unit = vec![
        put(PartitionId::Node, key(1), record(&[3, 9])),
        context(Vec::new(), Vec::new(), Vec::new()),
    ];
    let events = project_unit(&unit).expect("project");
    assert_eq!(events.len(), 1);
    let event = &events[0];
    assert_eq!(
        (event.ordinal, event.action, event.target.clone()),
        (
            0,
            Action::Put,
            Target::Node(Cow::Borrowed(key(1).as_slice()))
        )
    );
    assert_eq!(event.payload, Some(record(&[3, 9]).as_slice()));
    let routing = event.routing.as_ref().expect("routing");
    assert_eq!(routing.before.as_ref(), labels(&[3, 9]).as_slice());
    assert_eq!(routing.delivery(|l| l == label(9)), Delivery::Apply);
    assert_eq!(routing.delivery(|l| l == label(4)), Delivery::None);
}

/// A removed node routes by the labels it stored before the unit, which the
/// context carries: the old row is not read, so the removal reaches its
/// subscribers after the row is collected.
#[test]
fn a_removal_routes_by_the_labels_the_node_had() {
    let unit = vec![
        delete(PartitionId::Node, key(1)),
        context(vec![basis(1, &[3, 9])], Vec::new(), Vec::new()),
    ];
    let events = project_unit(&unit).expect("project");
    assert_eq!(
        shape(&events),
        vec![(
            0,
            Action::Remove,
            Target::Node(Cow::Borrowed(key(1).as_slice()))
        )]
    );
    assert_eq!(events[0].payload, None);
    let routing = events[0].routing.as_ref().expect("routing");
    assert_eq!(routing.delivery(|l| l == label(9)), Delivery::Leave);
    assert_eq!(routing.after.as_ref(), &[] as &[LabelId]);
}

/// A relabelling routes by the labels before and after: a subscriber of the
/// old label only sees the node leave, one of the new label only sees it
/// enter with the written record, and one of both gets the one event once.
#[test]
fn a_relabelling_enters_and_leaves_by_scope() {
    let unit = vec![
        put(PartitionId::Node, key(1), record(&[5])),
        context(vec![basis(1, &[3])], Vec::new(), Vec::new()),
    ];
    let events = project_unit(&unit).expect("project");
    assert_eq!(events.len(), 1);
    let routing = events[0].routing.as_ref().expect("routing");
    assert_eq!(routing.delivery(|l| l == label(3)), Delivery::Leave);
    assert_eq!(routing.delivery(|l| l == label(5)), Delivery::Enter);
    assert_eq!(
        routing.delivery(|l| l == label(3) || l == label(5)),
        Delivery::Apply
    );
    assert_eq!(events[0].payload, Some(record(&[5]).as_slice()));
}

/// A document delta is a patch carrying the delta, routed by the labels the
/// node keeps.
#[test]
fn a_delta_is_a_patch_under_the_labels_it_lands_on() {
    let delta = vec![0x01, 0xAA];
    let unit = vec![
        Mutation::Merge {
            partition: PartitionId::Node,
            key: key(1),
            operand: delta.clone(),
        },
        context(vec![basis(1, &[4])], Vec::new(), Vec::new()),
    ];
    let events = project_unit(&unit).expect("project");
    assert_eq!(events[0].action, Action::Patch);
    assert_eq!(events[0].payload, Some(delta.as_slice()));
    assert_eq!(
        events[0]
            .routing
            .as_ref()
            .expect("routing")
            .delivery(|l| l == label(4)),
        Delivery::Apply
    );
}

/// A removal or a delta without the labels that route it contradicts the
/// context its writer sealed: refused, never routed by a guess.
#[test]
fn a_change_without_its_routing_is_refused() {
    for change in [
        delete(PartitionId::Node, key(1)),
        Mutation::Merge {
            partition: PartitionId::Node,
            key: key(1),
            operand: vec![0x01],
        },
    ] {
        let unit = vec![change, context(Vec::new(), Vec::new(), Vec::new())];
        assert!(
            matches!(project_unit(&unit), Err(ProjectionError::Malformed(_))),
            "{unit:?}"
        );
    }
}

/// A unit changing application data without a context, or under another
/// interpretation, has no projection here: the reader is told, not handed
/// nothing. A unit that changes no application data projects to nothing.
#[test]
fn a_unit_without_a_readable_context_is_unsupported() {
    let bare = vec![put(PartitionId::Node, key(1), record(&[3]))];
    assert!(matches!(
        project_unit(&bare),
        Err(ProjectionError::Unsupported(_))
    ));
    let other = vec![
        put(PartitionId::Node, key(1), record(&[3])),
        Mutation::Effects(UnitEffects {
            interpretation: EFFECTS_INTERPRETATION + 1,
            ..UnitEffects::new()
        }),
    ];
    assert!(matches!(
        project_unit(&other),
        Err(ProjectionError::Unsupported(_))
    ));
    let bookkeeping = vec![
        put(PartitionId::Idx, b"idx:k".to_vec(), Vec::new()),
        put(PartitionId::Counter, b"counter:k".to_vec(), vec![0; 8]),
    ];
    assert_eq!(project_unit(&bookkeeping).expect("project"), Vec::new());
}

/// A relationship's presence follows its forward adjacency change; the
/// reverse twin, history rows and index entries are its physical forms and
/// add nothing. A property record of the pair is its own component.
#[test]
fn reverse_adjacency_and_derived_rows_add_no_events() {
    let props = vec![0x80];
    let unit = vec![
        adj("adj:KNOWS:out:1", encode_add_batch(&[2, 3])),
        adj("adj:KNOWS:in:2", encode_add(1)),
        adj("adj:KNOWS:in:3", encode_add(1)),
        put(PartitionId::EdgeProp, pair("KNOWS", 1, 2), props.clone()),
        put(PartitionId::Idx, b"idx:x".to_vec(), Vec::new()),
        put(PartitionId::NodeHistory, b"nodehist:x".to_vec(), vec![1]),
        context(Vec::new(), Vec::new(), Vec::new()),
    ];
    let events = project_unit(&unit).expect("project");
    let props_key = pair("KNOWS", 1, 2);
    assert_eq!(
        shape(&events),
        vec![
            (0, Action::Put, relationship("KNOWS", 1, 2)),
            (1, Action::Put, relationship("KNOWS", 1, 3)),
            (
                2,
                Action::Put,
                Target::RelationshipProperties(Cow::Borrowed(props_key.as_slice()))
            ),
        ]
    );
    assert_eq!(events[2].payload, Some(props.as_slice()));
}

/// Removing a relationship removes its property record too: one removal.
/// Clearing the properties of a relationship that stays removes only them.
/// An instance's record is the instance, removed on its own.
#[test]
fn presence_and_properties_are_told_apart() {
    let removal = vec![
        delete(PartitionId::EdgeProp, pair("KNOWS", 1, 2)),
        adj("adj:KNOWS:out:1", encode_remove(2)),
        adj("adj:KNOWS:in:2", encode_remove(1)),
        context(Vec::new(), Vec::new(), Vec::new()),
    ];
    assert_eq!(
        shape(&project_unit(&removal).expect("project")),
        vec![(0, Action::Remove, relationship("KNOWS", 1, 2))]
    );

    let props_key = pair("KNOWS", 1, 2);
    let clear = vec![
        delete(PartitionId::EdgeProp, props_key.clone()),
        context(Vec::new(), Vec::new(), Vec::new()),
    ];
    assert_eq!(
        shape(&project_unit(&clear).expect("project")),
        vec![(
            0,
            Action::Remove,
            Target::RelationshipProperties(Cow::Borrowed(props_key.as_slice()))
        )]
    );

    let mut instance = pair("RATED", 1, 2);
    instance.extend_from_slice(b":\x80\0\0\0\0\0\0\x05");
    let instance_removal = vec![
        delete(PartitionId::EdgeProp, instance.clone()),
        context(Vec::new(), Vec::new(), Vec::new()),
    ];
    assert_eq!(
        shape(&project_unit(&instance_removal).expect("project")),
        vec![(
            0,
            Action::Remove,
            Target::RelationshipInstance(Cow::Borrowed(instance.as_slice()))
        )]
    );
}

/// A relationship removed and written again in one unit stands: its
/// property tombstone is a clearing, not part of a removal.
#[test]
fn a_relationship_removed_and_written_again_is_not_removed() {
    let props_key = pair("KNOWS", 1, 2);
    let unit = vec![
        delete(PartitionId::EdgeProp, props_key.clone()),
        adj("adj:KNOWS:out:1", encode_remove(2)),
        adj("adj:KNOWS:out:1", encode_add(2)),
        context(Vec::new(), Vec::new(), Vec::new()),
    ];
    assert_eq!(
        shape(&project_unit(&unit).expect("project")),
        vec![
            (
                0,
                Action::Remove,
                Target::RelationshipProperties(Cow::Borrowed(props_key.as_slice()))
            ),
            (1, Action::Remove, relationship("KNOWS", 1, 2)),
            (2, Action::Put, relationship("KNOWS", 1, 2)),
        ]
    );
}

/// A node's whole forward list goes with it: the relationships it held are
/// named by the context and removed once each, a self-loop also named by a
/// forward removal included, after the unit's other events.
#[test]
fn relationships_removed_with_their_list_are_named_once() {
    let unit = vec![
        delete(PartitionId::Node, key(1)),
        delete(
            PartitionId::Adj,
            b"adj:KNOWS:out:\0\0\0\0\0\0\0\x01".to_vec(),
        ),
        delete(PartitionId::EdgeProp, pair("KNOWS", 1, 2)),
        adj("adj:KNOWS:in:2", encode_remove(1)),
        adj("adj:KNOWS:out:1", encode_remove(1)),
        context(
            vec![basis(1, &[3])],
            vec![pair("KNOWS", 1, 1), pair("KNOWS", 1, 2)],
            Vec::new(),
        ),
    ];
    assert_eq!(
        shape(&project_unit(&unit).expect("project")),
        vec![
            (
                0,
                Action::Remove,
                Target::Node(Cow::Borrowed(key(1).as_slice()))
            ),
            (1, Action::Remove, relationship("KNOWS", 1, 1)),
            (2, Action::Remove, relationship("KNOWS", 1, 2)),
        ]
    );
}

/// A run of point deletes projects to the deletes it stands for, each
/// routed by its own labels; a range removal the unit does not name a run
/// is refused, since its bounds name no targets.
#[test]
fn a_run_projects_to_its_deletes_and_a_range_is_refused() {
    let unit = vec![
        Mutation::RemoveRange {
            partition: PartitionId::Node,
            start: key(10),
            end: key(13),
        },
        context(
            vec![basis(10, &[3]), basis(11, &[4]), basis(12, &[3])],
            Vec::new(),
            vec![0],
        ),
    ];
    let events = project_unit(&unit).expect("project");
    assert_eq!(
        shape(&events),
        vec![
            (0, Action::Remove, Target::Node(Cow::Owned(key(10)))),
            (1, Action::Remove, Target::Node(Cow::Owned(key(11)))),
            (2, Action::Remove, Target::Node(Cow::Owned(key(12)))),
        ]
    );
    assert_eq!(
        events[1].routing.as_ref().expect("routing").before.as_ref(),
        labels(&[4]).as_slice()
    );

    let range = vec![
        Mutation::RemoveRange {
            partition: PartitionId::Node,
            start: key(10),
            end: key(13),
        },
        context(Vec::new(), Vec::new(), Vec::new()),
    ];
    assert!(matches!(
        project_unit(&range),
        Err(ProjectionError::Unsupported(_))
    ));
    // An index range is bookkeeping, whatever it covers.
    let index = vec![
        Mutation::RemoveRange {
            partition: PartitionId::Idx,
            start: b"idx:a".to_vec(),
            end: b"idx:b".to_vec(),
        },
        put(PartitionId::Node, key(1), record(&[3])),
        context(Vec::new(), Vec::new(), Vec::new()),
    ];
    assert_eq!(project_unit(&index).expect("project").len(), 1);
}

/// Ordinals count the unit's logical events in its own order, whatever
/// physical operations lie between them: the same unit always numbers the
/// same way.
#[test]
fn ordinals_follow_the_units_order() {
    let unit = vec![
        put(PartitionId::Idx, b"idx:x".to_vec(), Vec::new()),
        put(PartitionId::Node, key(1), record(&[3])),
        adj("adj:KNOWS:in:2", encode_add(1)),
        adj("adj:KNOWS:out:1", encode_add(2)),
        delete(PartitionId::Node, key(2)),
        context(vec![basis(2, &[3])], Vec::new(), Vec::new()),
    ];
    let first = project_unit(&unit).expect("project");
    assert_eq!(
        first.iter().map(|e| e.ordinal).collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(project_unit(&unit).expect("project"), first);
}
