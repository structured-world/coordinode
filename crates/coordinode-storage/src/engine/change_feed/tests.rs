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

/// Implications recording that none of the types these tests use (labels
/// 1 to 20) implies anything.
fn implying_nothing() -> Vec<coordinode_core::txn::proposal::Implication> {
    (1..=20)
        .map(|raw| coordinode_core::txn::proposal::Implication {
            label: label(raw),
            implies: Vec::new(),
        })
        .collect()
}

fn context(nodes: Vec<NodeBasis>, removed_edges: Vec<Vec<u8>>, runs: Vec<u32>) -> Mutation {
    Mutation::Effects(UnitEffects {
        interpretation: EFFECTS_INTERPRETATION,
        nodes,
        removed_edges,
        runs,
        implications: implying_nothing(),
        ..UnitEffects::new()
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

/// A node of a type that implied labels when its unit committed routes by
/// them too: a put, a delta and a removal reach a subscriber of the implied
/// label, by the unit's own interpretation, and a node of another type does
/// not.
#[test]
fn a_node_routes_by_what_its_type_implied_at_commit() {
    use coordinode_core::txn::proposal::Implication;
    let implications = vec![
        Implication {
            label: label(3),
            implies: labels(&[7, 8]),
        },
        Implication {
            label: label(6),
            implies: Vec::new(),
        },
    ];
    let unit = vec![
        put(PartitionId::Node, key(1), record(&[3])),
        Mutation::Merge {
            partition: PartitionId::Node,
            key: key(2),
            operand: b"delta".to_vec(),
        },
        delete(PartitionId::Node, key(4)),
        put(PartitionId::Node, key(5), record(&[6])),
        Mutation::Effects(UnitEffects {
            interpretation: EFFECTS_INTERPRETATION,
            nodes: vec![
                basis(1, &[3]),
                basis(2, &[3, 9]),
                basis(4, &[3]),
                basis(5, &[6]),
            ],
            implications,
            ..UnitEffects::new()
        }),
    ];
    let events = project_unit(&unit).expect("project");
    let delivered: Vec<Delivery> = events
        .iter()
        .map(|e| {
            e.routing
                .as_ref()
                .expect("routing")
                .delivery(|l| l == label(8))
        })
        .collect();
    assert_eq!(
        delivered,
        [
            Delivery::Apply,
            Delivery::Apply,
            Delivery::Leave,
            Delivery::None
        ]
    );
    // The stored labels stay first, the implied ones follow, each once.
    let routing = events[1].routing.as_ref().expect("routing");
    assert_eq!(routing.after.as_ref(), labels(&[3, 9, 7, 8]).as_slice());
}

/// A change of what a type implies is one scope event per type, at the
/// context's position, routed by the type with what it implied before and
/// after: entering a gained label, leaving a lost one, applying to the type.
#[test]
fn a_scope_change_is_one_event_routed_by_the_type() {
    use coordinode_core::txn::proposal::ScopeChange;
    let unit = vec![Mutation::Effects(UnitEffects {
        interpretation: EFFECTS_INTERPRETATION,
        scope_changes: vec![ScopeChange {
            label: label(3),
            activation: 41,
            before: labels(&[7]),
            after: labels(&[8]),
        }],
        ..UnitEffects::new()
    })];
    let events = project_unit(&unit).expect("project");
    assert_eq!(
        shape(&events),
        vec![(
            0,
            Action::ScopeChange,
            Target::Scope {
                label: label(3),
                activation: 41
            }
        )]
    );
    let routing = events[0].routing.as_ref().expect("routing");
    assert_eq!(routing.delivery(|l| l == label(7)), Delivery::Leave);
    assert_eq!(routing.delivery(|l| l == label(8)), Delivery::Enter);
    assert_eq!(routing.delivery(|l| l == label(3)), Delivery::Apply);
    assert_eq!(routing.delivery(|l| l == label(5)), Delivery::None);
}

/// A put whose labels did not change routes by its record before and after:
/// it applies to a subscriber of any of its labels, and to nobody else.
#[test]
fn a_put_routes_by_its_record() {
    let unit = vec![
        put(PartitionId::Node, key(1), record(&[3, 9])),
        context(vec![basis(1, &[3, 9])], Vec::new(), Vec::new()),
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

/// The deliveries of each event of `unit` to a reader of `label`.
fn deliveries_to(unit: &[Mutation], label_raw: u32) -> Vec<(Action, Delivery)> {
    project_unit(unit)
        .expect("project")
        .iter()
        .map(|e| {
            (
                e.action,
                e.routing
                    .as_ref()
                    .expect("routing")
                    .delivery(|l| l == label(label_raw)),
            )
        })
        .collect()
}

/// A delta after a write that changed the node's labels in the same unit
/// lands on the labels that write left: a reader of the new label receives
/// the record entering and then the delta; a reader of the old label only
/// the record leaving.
#[test]
fn a_delta_after_a_relabelling_put_lands_on_the_new_labels() {
    let unit = vec![
        put(PartitionId::Node, key(1), record(&[5])),
        Mutation::Merge {
            partition: PartitionId::Node,
            key: key(1),
            operand: vec![0x01],
        },
        context(vec![basis(1, &[3])], Vec::new(), Vec::new()),
    ];
    assert_eq!(
        deliveries_to(&unit, 5),
        vec![
            (Action::Put, Delivery::Enter),
            (Action::Patch, Delivery::Apply)
        ]
    );
    assert_eq!(
        deliveries_to(&unit, 3),
        vec![
            (Action::Put, Delivery::Leave),
            (Action::Patch, Delivery::None)
        ]
    );
}

/// A node created and given a delta in the same unit: the delta reaches the
/// reader of its label after the record entered.
#[test]
fn a_delta_on_a_node_created_in_the_unit_reaches_its_label() {
    let unit = vec![
        put(PartitionId::Node, key(1), record(&[5])),
        Mutation::Merge {
            partition: PartitionId::Node,
            key: key(1),
            operand: vec![0x01],
        },
        context(vec![basis(1, &[])], Vec::new(), Vec::new()),
    ];
    assert_eq!(
        deliveries_to(&unit, 5),
        vec![
            (Action::Put, Delivery::Enter),
            (Action::Patch, Delivery::Apply)
        ]
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
/// reverse twin and index entries are its physical forms and add nothing. A
/// property record of the pair is its own component.
#[test]
fn reverse_adjacency_and_derived_rows_add_no_events() {
    let props = vec![0x80];
    let unit = vec![
        adj("adj:KNOWS:out:1", encode_add_batch(&[2, 3])),
        adj("adj:KNOWS:in:2", encode_add(1)),
        adj("adj:KNOWS:in:3", encode_add(1)),
        put(PartitionId::EdgeProp, pair("KNOWS", 1, 2), props.clone()),
        put(PartitionId::Idx, b"idx:x".to_vec(), Vec::new()),
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
        context(vec![basis(1, &[3])], Vec::new(), Vec::new()),
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
        context(vec![basis(1, &[3]), basis(2, &[3])], Vec::new(), Vec::new()),
    ];
    let first = project_unit(&unit).expect("project");
    assert_eq!(
        first.iter().map(|e| e.ordinal).collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(project_unit(&unit).expect("project"), first);
}

/// Records outside nodes and relationships: a blob owner's record is a put
/// and a removal of that owner; the catalog, staging and payload records
/// add nothing; a history assertion goes with the version it states and is
/// refused without it; a blob record of no known kind, a vector index copy
/// and a range over owners are refused, never passed over.
#[test]
fn components_are_projected_or_refused_never_passed_over() {
    use coordinode_core::graph::blob::{CATALOG_PREFIX, DIRECT_OWNER_PREFIX, STAGING_PREFIX};
    use coordinode_core::graph::node::{encode_node_history_prefix, encode_temporal_node_key};
    let owner = [DIRECT_OWNER_PREFIX, b"o1"].concat();
    let unit = vec![
        put(
            PartitionId::BlobRef,
            [STAGING_PREFIX, b"1"].concat(),
            vec![1],
        ),
        put(
            PartitionId::BlobRef,
            [CATALOG_PREFIX, b"c"].concat(),
            vec![2],
        ),
        put(PartitionId::Blob, b"blob:c".to_vec(), vec![3]),
        put(PartitionId::BlobRef, owner.clone(), b"record".to_vec()),
        delete(PartitionId::BlobRef, owner.clone()),
        context(Vec::new(), Vec::new(), Vec::new()),
    ];
    assert_eq!(
        shape(&project_unit(&unit).expect("project")),
        vec![
            (
                0,
                Action::Put,
                Target::BlobOwner(Cow::Borrowed(owner.as_slice()))
            ),
            (
                1,
                Action::Remove,
                Target::BlobOwner(Cow::Borrowed(owner.as_slice()))
            ),
        ]
    );

    let mut history = encode_node_history_prefix(0, NodeId::from_raw(4), 100);
    history.extend_from_slice(&9u64.to_be_bytes());
    let version = encode_temporal_node_key(0, NodeId::from_raw(4), 100);
    let with_version = vec![
        put(PartitionId::NodeHistory, history.clone(), vec![1]),
        put(PartitionId::Node, version.clone(), record(&[3])),
        context(
            vec![NodeBasis {
                key: version,
                labels: labels(&[3]),
            }],
            Vec::new(),
            Vec::new(),
        ),
    ];
    assert_eq!(project_unit(&with_version).expect("project").len(), 1);

    for refused in [
        vec![put(PartitionId::NodeHistory, history, vec![1])],
        vec![put(PartitionId::BlobRef, b"blobnew:x".to_vec(), vec![1])],
        vec![put(PartitionId::VectorF32, b"vec:x".to_vec(), vec![1])],
        vec![Mutation::RemoveRange {
            partition: PartitionId::BlobRef,
            start: DIRECT_OWNER_PREFIX.to_vec(),
            end: b"blobmeta;".to_vec(),
        }],
    ] {
        let mut unit = refused;
        unit.push(context(Vec::new(), Vec::new(), Vec::new()));
        assert!(
            matches!(project_unit(&unit), Err(ProjectionError::Unsupported(_))),
            "{unit:?}"
        );
    }
}

/// A consumer that stops taking is handed no more events, while the unit's
/// whole count is still returned; a run of deletes is handed over one key
/// at a time, never listed first.
#[test]
fn a_projection_hands_over_events_one_at_a_time() {
    let mut deletes: Vec<Mutation> = (1..=50)
        .map(|id| delete(PartitionId::Node, key(id)))
        .collect();
    deletes.push(context(
        (1..=50).map(|id| basis(id, &[3])).collect(),
        Vec::new(),
        Vec::new(),
    ));
    let mut handed = 0;
    let projected = project_unit_with(&deletes, |_| {
        handed += 1;
        if handed == 3 {
            Take::Enough
        } else {
            Take::More
        }
    })
    .expect("project");
    assert_eq!((handed, projected), (3, 50));
}

/// A read takes memory from its budget a block at a time: many small events
/// cost one reservation, not one each.
#[test]
fn a_read_charges_its_events_by_the_block() {
    let units = entry_units(20);
    let budget = coordinode_core::budget::QueryBudget::new(1 << 20);
    let mut charge = budget.empty_charge();
    let (got, _) = unit_changes(&units, &Interest::all(), at(0, 0), &mut charge).expect("read");
    let events: usize = got.iter().map(|u| u.events.len()).sum();
    assert_eq!(events, 21);
    // What the events and units hold, the unspent rest of the block returned.
    let held = budget.memory_used();
    assert!(held > 0 && held <= CHARGE_BLOCK, "{held}");
    assert_eq!(held, charge.bytes());
}

/// The place in entry 7 at unit `sub`, ordinal `ordinal`.
fn at(sub: u32, ordinal: u32) -> ReadPlace {
    ReadPlace {
        entry: 7,
        within: Within { sub, ordinal },
    }
}

/// Two units of one entry, the first of `nodes` node puts, the second of one.
fn entry_units(nodes: u64) -> Vec<ChangeUnit> {
    use coordinode_core::txn::operation::LogPosition;
    let unit = |sub: u32, ids: core::ops::RangeInclusive<u64>| ChangeUnit {
        position: LogPosition { index: 7, sub },
        commit_ts: 100 + u64::from(sub),
        refused: false,
        mutations: ids
            .clone()
            .map(|id| put(PartitionId::Node, key(id), record(&[3])))
            .chain([context(
                ids.map(|id| basis(id, &[])).collect(),
                Vec::new(),
                Vec::new(),
            )])
            .collect(),
    };
    vec![unit(0, 1..=nodes), unit(1, 1000..=1000)]
}

/// The ordinals of `units`' events, by unit, and each unit's covered range.
fn covered(units: &[UnitChanges]) -> Vec<(u32, u32, u32, Vec<u32>)> {
    units
        .iter()
        .map(|u| {
            (
                u.position.sub,
                u.first,
                u.through,
                u.events.iter().map(|e| e.id.ordinal).collect(),
            )
        })
        .collect()
}

/// A read whose memory runs out inside a unit hands over a fragment and
/// stops there; reads continuing from where it stopped deliver the rest in
/// consecutive ranges, each event once, and the units after it whole.
#[test]
fn a_read_stops_inside_a_unit_and_continues_there() {
    let units = entry_units(10);
    let interest = Interest::all();
    let whole = {
        let budget = coordinode_core::budget::QueryBudget::new(1 << 20);
        let mut charge = budget.empty_charge();
        let (whole, stopped) =
            unit_changes(&units, &interest, at(0, 0), &mut charge).expect("read");
        assert_eq!(stopped, None);
        whole
    };
    // Room for the unit and a few of its events a read.
    let event = whole[0].events.len();
    let room = (core::mem::size_of::<UnitChanges>()
        + 4 * core::mem::size_of::<ChangeEvent>()
        + 4 * 64) as u64;
    let mut from = at(0, 0);
    let mut pieces = Vec::new();
    for _ in 0..32 {
        let budget = coordinode_core::budget::QueryBudget::new(room);
        let mut charge = budget.empty_charge();
        let (got, stopped) = unit_changes(&units, &interest, from, &mut charge).expect("read");
        pieces.extend(got);
        match stopped {
            Some(next) => from = next,
            None => break,
        }
    }
    assert_eq!(event, 10);
    let shape = covered(&pieces);
    assert!(shape.len() > 2, "{shape:?}");
    let mut next = 0;
    for (sub, first, through, ordinals) in &shape[..shape.len() - 1] {
        assert_eq!(*sub, 0);
        assert_eq!(*first, next, "consecutive ranges: {shape:?}");
        assert_eq!(*ordinals, (*first..*through).collect::<Vec<_>>());
        next = *through;
    }
    assert_eq!(next, 10, "the first unit reaches its end");
    assert_eq!(shape.last().expect("the second unit"), &(1, 0, 1, vec![0]));
    let joined: Vec<ChangeEvent> = pieces
        .iter()
        .filter(|u| u.position.sub == 0)
        .flat_map(|u| u.events.clone())
        .collect();
    assert_eq!(joined, whole[0].events);
}

/// A read starting inside an entry passes over the units before its place
/// and starts its unit at its ordinal; a read that cannot keep one event
/// refuses rather than hand over nothing.
#[test]
fn a_read_starts_at_its_place_inside_the_entry() {
    let units = entry_units(5);
    let budget = coordinode_core::budget::QueryBudget::new(1 << 20);
    let mut charge = budget.empty_charge();
    let (got, _) = unit_changes(&units, &Interest::all(), at(0, 3), &mut charge).expect("read");
    assert_eq!(
        covered(&got),
        vec![(0, 3, 5, vec![3, 4]), (1, 0, 1, vec![0])]
    );
    let (got, _) = unit_changes(&units, &Interest::all(), at(1, 0), &mut charge).expect("read");
    assert_eq!(covered(&got), vec![(1, 0, 1, vec![0])]);

    let tiny = coordinode_core::budget::QueryBudget::new(8);
    let mut charge = tiny.empty_charge();
    assert!(matches!(
        unit_changes(&units, &Interest::all(), at(0, 0), &mut charge),
        Err(crate::error::StorageError::Budget(_))
    ));
}
