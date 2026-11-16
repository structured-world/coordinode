//! Committed changes read as logical events through the public reader:
//! puts, patches and removals reproduce the source, label scopes are entered
//! and left by the labels a node had and has, relationships are told apart
//! from their physical forms, and an event keeps its identity across readers
//! and reopens.

#![allow(clippy::expect_used, clippy::panic)]

use coordinode_core::budget::QueryBudget;
use coordinode_embed::Database;
use coordinode_embed::changes::{
    Action, ChangeCursor, ChangeEvent, ChangeInterest, Delivery, OwnedTarget, UnitChanges,
};

fn open(dir: &std::path::Path) -> Database {
    let mut db = Database::open(dir).expect("open");
    for ddl in [
        "CREATE NODE TYPE Person WITH (name: STRING)",
        "CREATE NODE TYPE Org WITH (name: STRING)",
        "CREATE LABEL Admin AUXILIARY",
    ] {
        db.execute_cypher(ddl)
            .unwrap_or_else(|e| panic!("{ddl}: {e}"));
    }
    db
}

fn run(db: &mut Database, query: &str) {
    db.execute_cypher(query)
        .unwrap_or_else(|e| panic!("{query}: {e}"));
}

fn interest(labels: Option<&[&str]>, edge_types: Option<&[&str]>) -> ChangeInterest {
    ChangeInterest {
        labels: labels.map(|l| l.iter().map(|s| (*s).to_string()).collect()),
        edge_types: edge_types.map(|t| t.iter().map(|s| (*s).to_string()).collect()),
    }
}

/// Every unit after `cursor` up to the end of what is committed, the cursor
/// moved past them.
fn read_all(db: &Database, cursor: &mut ChangeCursor) -> Vec<UnitChanges> {
    let budget = QueryBudget::new(1 << 24);
    let mut units = Vec::new();
    loop {
        let page = db.read_changes(cursor, 8, &budget).expect("read");
        let next = page.batch.cursor.clone();
        let moved = next.position() != cursor.position();
        units.extend(page.batch.units);
        *cursor = next;
        if !moved {
            return units;
        }
    }
}

/// The events of `units`, in order.
fn events(units: &[UnitChanges]) -> Vec<ChangeEvent> {
    units.iter().flat_map(|u| u.events.clone()).collect()
}

/// Each node event's action and delivery, in order.
fn node_deliveries(events: &[ChangeEvent]) -> Vec<(Action, Delivery)> {
    events
        .iter()
        .filter(|e| matches!(e.target, OwnedTarget::Node(_)))
        .map(|e| (e.action, e.delivery))
        .collect()
}

/// A node written, changed and removed reaches a reader of its label as a
/// put that enters the scope, a put that applies and a removal that leaves:
/// the last write states the record, never whether it was new.
#[test]
fn a_node_is_put_changed_and_removed() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = open(dir.path());
    let mut cursor = db
        .change_cursor(&interest(Some(&["Person"]), None))
        .expect("cursor");
    run(&mut db, "CREATE (:Person {name: 'a'}), (:Org {name: 'o'})");
    run(&mut db, "MATCH (n:Person) SET n.name = 'b'");
    run(&mut db, "MATCH (n:Person) REMOVE n.name");
    run(&mut db, "MATCH (n:Person) DELETE n");
    let events = events(&read_all(&db, &mut cursor));
    assert_eq!(
        node_deliveries(&events),
        vec![
            (Action::Put, Delivery::Apply),
            (Action::Put, Delivery::Apply),
            (Action::Put, Delivery::Apply),
            (Action::Remove, Delivery::Leave),
        ],
        "{events:?}"
    );
    // The Org node is not in the scope; the payloads are the records written.
    assert!(events.iter().all(|e| e.labels.as_ref().is_some()));
    assert!(events[..3].iter().all(|e| e.payload.is_some()));
    assert!(events[3].payload.is_none());
}

/// A removal reaches the readers of every label the node stored, from a
/// cursor saved before and read back after a reopen: what routes it is the
/// unit's own, not a read of a row the deletion removed.
#[test]
fn a_removed_multi_labelled_node_reaches_each_label_after_a_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");
    let saved = {
        let mut db = open(dir.path());
        run(&mut db, "CREATE (:Person:Admin {name: 'a'})");
        let saved = db
            .change_cursor(&interest(Some(&["Admin"]), None))
            .expect("cursor")
            .to_bytes();
        run(&mut db, "MATCH (n:Admin) DETACH DELETE n");
        saved
    };
    let db = Database::open(dir.path()).expect("reopen");
    let mut cursor = ChangeCursor::from_bytes(&saved).expect("cursor bytes");
    let events = events(&read_all(&db, &mut cursor));
    assert_eq!(
        node_deliveries(&events),
        vec![(Action::Remove, Delivery::Leave)],
        "{events:?}"
    );
}

/// RETYPE and an auxiliary label: a reader of the old type sees the node
/// leave, a reader of the new type sees it enter with the written record, a
/// reader of both gets the one event once, and removing the auxiliary label
/// leaves its scope without touching the type's.
#[test]
fn a_retype_and_an_auxiliary_label_move_the_node_between_scopes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = open(dir.path());
    run(&mut db, "CREATE (:Person:Admin {name: 'a'})");
    let mut old = db
        .change_cursor(&interest(Some(&["Person"]), None))
        .expect("cursor");
    let mut new = db
        .change_cursor(&interest(Some(&["Org"]), None))
        .expect("cursor");
    let mut both = db
        .change_cursor(&interest(Some(&["Person", "Org"]), None))
        .expect("cursor");
    let mut admin = db
        .change_cursor(&interest(Some(&["Admin"]), None))
        .expect("cursor");
    run(
        &mut db,
        "MATCH (n:Person) RETYPE NODE n TO Org SET n.name = 'o'",
    );
    run(&mut db, "MATCH (n:Org) REMOVE n:Admin");

    let old = events(&read_all(&db, &mut old));
    let new = events(&read_all(&db, &mut new));
    let both = events(&read_all(&db, &mut both));
    let admin = events(&read_all(&db, &mut admin));
    assert_eq!(node_deliveries(&old), vec![(Action::Put, Delivery::Leave)]);
    assert_eq!(
        node_deliveries(&new),
        vec![
            (Action::Put, Delivery::Enter),
            (Action::Put, Delivery::Apply)
        ]
    );
    assert_eq!(
        node_deliveries(&both),
        vec![
            (Action::Put, Delivery::Apply),
            (Action::Put, Delivery::Apply)
        ]
    );
    assert_eq!(
        node_deliveries(&admin),
        vec![
            (Action::Put, Delivery::Apply),
            (Action::Put, Delivery::Leave)
        ]
    );
    // The retype is one source event, the same for every reader of it.
    assert_eq!(old[0].id, new[0].id);
    assert_eq!(old[0].id, both[0].id);
    assert_eq!(old[0].id, admin[0].id);
    assert_eq!(new[0].payload, both[0].payload);
}

/// The relationship events of each write, by type.
fn relationship_actions(units: &[UnitChanges]) -> Vec<Vec<(Action, &'static str)>> {
    units
        .iter()
        .filter(|u| !u.events.is_empty())
        .map(|u| {
            u.events
                .iter()
                .map(|e| {
                    let kind = match &e.target {
                        OwnedTarget::Relationship { .. } => "presence",
                        OwnedTarget::RelationshipProperties(_) => "properties",
                        OwnedTarget::RelationshipInstance(_) => "instance",
                        OwnedTarget::Node(_) => "node",
                        OwnedTarget::Scope { .. } => "scope",
                    };
                    (e.action, kind)
                })
                .collect()
        })
        .collect()
}

/// A relationship without properties is one presence event; setting a
/// property puts the properties; removing its last property clears them
/// while the relationship stays; deleting it is one removal, its property
/// record with it; reverse adjacency adds nothing to any of them.
#[test]
fn relationship_presence_and_properties_are_distinct_effects() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = open(dir.path());
    run(
        &mut db,
        "CREATE (:Person {name: 'a'}), (:Person {name: 'b'})",
    );
    let mut cursor = db
        .change_cursor(&interest(Some(&[]), Some(&["KNOWS"])))
        .expect("cursor");
    run(
        &mut db,
        "MATCH (a:Person {name: 'a'}), (b:Person {name: 'b'}) CREATE (a)-[:KNOWS]->(b)",
    );
    run(&mut db, "MATCH ()-[r:KNOWS]->() SET r.since = 1");
    run(&mut db, "MATCH ()-[r:KNOWS]->() REMOVE r.since");
    run(&mut db, "MATCH ()-[r:KNOWS]->() DELETE r");
    let units = read_all(&db, &mut cursor);
    assert_eq!(
        relationship_actions(&units),
        vec![
            vec![(Action::Put, "presence")],
            vec![(Action::Put, "properties")],
            vec![(Action::Remove, "properties")],
            vec![(Action::Remove, "presence")],
        ],
        "{units:?}"
    );
}

/// DETACH DELETE of a node with an outgoing relationship and a self-loop
/// removes each relationship once, though the node's own lists go whole and
/// name none of them; the peer that stays is unaffected.
#[test]
fn detach_delete_removes_each_relationship_once() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = open(dir.path());
    run(
        &mut db,
        "CREATE (a:Person {name: 'a'})-[:KNOWS]->(:Person {name: 'b'}), (a)-[:KNOWS]->(a)",
    );
    let mut cursor = db
        .change_cursor(&interest(Some(&["Person"]), Some(&["KNOWS"])))
        .expect("cursor");
    run(&mut db, "MATCH (a:Person {name: 'a'}) DETACH DELETE a");
    let events = events(&read_all(&db, &mut cursor));
    let removed: Vec<(u64, u64)> = events
        .iter()
        .filter_map(|e| match &e.target {
            OwnedTarget::Relationship { source, target, .. } => {
                assert_eq!(e.action, Action::Remove);
                Some((*source, *target))
            }
            _ => None,
        })
        .collect();
    assert_eq!(removed.len(), 2, "{events:?}");
    assert!(
        removed.iter().any(|(s, t)| s == t),
        "the self-loop: {removed:?}"
    );
    assert_eq!(
        node_deliveries(&events),
        vec![(Action::Remove, Delivery::Leave)]
    );
}

/// Readers with overlapping interests get a common event under one
/// identity, and a reader reading again gets the same identities: ordinals
/// number the unit before any reader filters it. Every unit a reader passes
/// is in its batch, an empty one too, with the count of its projection.
#[test]
fn identities_hold_across_interests_and_reads() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = open(dir.path());
    let start_all = db.change_cursor(&ChangeInterest::default()).expect("all");
    // Org nodes only: no relationship type.
    let start_org = db
        .change_cursor(&interest(Some(&["Org"]), Some(&[])))
        .expect("org");
    run(
        &mut db,
        "CREATE (:Person {name: 'p'})-[:KNOWS]->(:Org {name: 'o'})",
    );
    let all = read_all(&db, &mut start_all.clone());
    let again = read_all(&db, &mut start_all.clone());
    let org = read_all(&db, &mut start_org.clone());
    assert_eq!(all, again);
    let org_events = events(&org);
    assert_eq!(org_events.len(), 1, "{org:?}");
    assert!(events(&all).iter().any(|e| e.id == org_events[0].id));
    let unit = org
        .iter()
        .find(|u| !u.events.is_empty())
        .expect("the write's unit");
    assert!(unit.projected > 1, "{unit:?}");
    assert_eq!(
        org.iter().map(|u| u.position).collect::<Vec<_>>(),
        all.iter().map(|u| u.position).collect::<Vec<_>>()
    );
}

/// A temporal node's writes are events of its versions, each named by its
/// version key and routed by its labels; the history rows recording them
/// add no events of their own.
#[test]
fn a_temporal_nodes_versions_are_its_events() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = open(dir.path());
    run(
        &mut db,
        "CREATE NODE TYPE Emp TEMPORAL WITH (name: STRING, valid_from: INT, valid_to: INT)",
    );
    let mut cursor = db
        .change_cursor(&ChangeInterest::default())
        .expect("cursor");
    run(&mut db, "CREATE (:Emp {name: 'Alice', valid_from: 1000})");
    run(&mut db, "MATCH (e:Emp {name: 'Alice'}) CLONE NODE e AS c");
    let events = events(&read_all(&db, &mut cursor));
    assert!(events.len() >= 2, "{events:?}");
    for event in &events {
        let OwnedTarget::Node(key) = &event.target else {
            panic!("only node versions change: {event:?}");
        };
        assert!(
            coordinode_core::graph::node::decode_temporal_node_key(key).is_some(),
            "a version key: {key:?}"
        );
        assert!(event.labels.is_some(), "routed: {event:?}");
    }
}

/// One statement deleting many nodes, whose deletes the commit packs into
/// key ranges, reaches each label's reader as one removal per node with the
/// labels that node had, the same as deletes one at a time.
#[test]
fn a_packed_delete_is_one_removal_per_node() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = open(dir.path());
    run(
        &mut db,
        "UNWIND range(1, 50) AS i CREATE (:Person {name: 'p' + toString(i)})",
    );
    run(&mut db, "CREATE (:Org {name: 'o'})");
    let mut person = db
        .change_cursor(&interest(Some(&["Person"]), None))
        .expect("cursor");
    run(&mut db, "MATCH (n) DETACH DELETE n");
    let events = events(&read_all(&db, &mut person));
    assert_eq!(
        node_deliveries(&events),
        vec![(Action::Remove, Delivery::Leave); 50],
        "{events:?}"
    );
    let mut keys: Vec<&Vec<u8>> = events
        .iter()
        .filter_map(|e| match &e.target {
            OwnedTarget::Node(key) => Some(key),
            _ => None,
        })
        .collect();
    keys.dedup();
    assert_eq!(keys.len(), 50, "each node once");
}

/// A change of what a type implies that its proof refuses puts nothing in
/// force and delivers no scope change; compaction, which rewrites the
/// store's files, delivers nothing either.
#[test]
fn a_refused_change_and_compaction_deliver_nothing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    for ddl in [
        "CREATE NODE TYPE Thing WITH (title: STRING)",
        "CREATE NODE TYPE Task WITH (title: STRING)",
        "CREATE CONSTRAINT thing_title FOR (n:Thing) REQUIRE n.title IS UNIQUE",
    ] {
        run(&mut db, ddl);
    }
    run(
        &mut db,
        "CREATE (:Thing {title: 'x'}), (:Task {title: 'x'})",
    );
    let mut thing = db
        .change_cursor(&interest(Some(&["Thing"]), None))
        .expect("cursor");
    assert!(
        db.execute_cypher("ALTER NODE TYPE Task SET IMPLIES (Thing)")
            .is_err(),
        "the shared title refuses it"
    );
    for partition in [
        coordinode_storage::engine::partition::Partition::Node,
        coordinode_storage::engine::partition::Partition::Schema,
    ] {
        db.engine().force_compaction(partition).expect("compact");
    }
    let units = read_all(&db, &mut thing);
    assert!(events(&units).is_empty(), "{units:?}");
}

/// A read whose memory cannot hold one event of the next unit refuses with
/// the budget, the cursor staying where it was.
#[test]
fn a_read_too_small_for_one_event_refuses() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = open(dir.path());
    let start = db
        .change_cursor(&ChangeInterest::default())
        .expect("cursor");
    run(&mut db, "CREATE (:Person {name: 'a'})");
    let budget = QueryBudget::new(64);
    match db.read_changes(&start, 8, &budget) {
        Err(coordinode_embed::DatabaseError::Budget(_)) => {}
        Err(other) => panic!("refused otherwise: {other}"),
        Ok(page) => panic!("read within 64 bytes: {:?}", page.batch),
    }
}

/// A reader of a label a type implies follows the nodes of that type, which
/// carry the label without storing it, through every change of them.
#[test]
fn a_label_a_type_implies_follows_the_nodes_of_the_type() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = open(dir.path());
    run(
        &mut db,
        "CREATE NODE TYPE Manager IMPLIES (Person) WITH (name: STRING)",
    );
    let mut person = db
        .change_cursor(&interest(Some(&["Person"]), None))
        .expect("cursor");
    let mut org = db
        .change_cursor(&interest(Some(&["Org"]), None))
        .expect("cursor");
    run(&mut db, "CREATE (:Manager {name: 'm'})");
    run(&mut db, "MATCH (n:Manager) SET n.name = 'n'");
    run(&mut db, "MATCH (n:Manager) DELETE n");
    let person = events(&read_all(&db, &mut person));
    assert_eq!(
        node_deliveries(&person),
        vec![
            (Action::Put, Delivery::Apply),
            (Action::Put, Delivery::Apply),
            (Action::Remove, Delivery::Leave),
        ],
        "{person:?}"
    );
    assert!(events(&read_all(&db, &mut org)).is_empty());
}

/// The scope changes one reader receives: each one's type and activation,
/// and how it delivers.
fn scope_changes(events: &[ChangeEvent]) -> Vec<(u32, u64, Delivery)> {
    events
        .iter()
        .filter_map(|e| match e.target {
            OwnedTarget::Scope { label, activation } => {
                assert_eq!(e.action, Action::ScopeChange);
                Some((label.get(), activation, e.delivery))
            }
            _ => None,
        })
        .collect()
}

/// Changing what a type implies is one scope change per type it moves, the
/// types implying it included, delivered to readers of every label it gains
/// or loses with no node written; each activation has its own identity, a
/// change back included; and a node change is routed by the implication it
/// was committed under, however the type has changed since it is read.
#[test]
fn a_change_of_what_a_type_implies_reaches_its_scopes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    for ddl in [
        "CREATE NODE TYPE Thing WITH (title: STRING)",
        "CREATE NODE TYPE Task WITH (title: STRING)",
        "CREATE NODE TYPE Chore IMPLIES (Task) WITH (title: STRING)",
    ] {
        run(&mut db, ddl);
    }
    run(&mut db, "CREATE (:Task {title: 'before'})");
    let mut thing = db
        .change_cursor(&interest(Some(&["Thing"]), None))
        .expect("cursor");
    let mut task = db
        .change_cursor(&interest(Some(&["Task"]), None))
        .expect("cursor");
    run(&mut db, "ALTER NODE TYPE Task SET IMPLIES (Thing)");
    run(&mut db, "CREATE (:Task {title: 'implied'})");
    run(&mut db, "ALTER NODE TYPE Task SET IMPLIES ()");
    run(&mut db, "CREATE (:Task {title: 'plain'})");
    run(&mut db, "ALTER NODE TYPE Task SET IMPLIES (Thing)");

    let thing = events(&read_all(&db, &mut thing));
    let scopes = scope_changes(&thing);
    // Task and Chore move together on each of the three activations.
    assert_eq!(scopes.len(), 6, "{thing:?}");
    let deliveries: Vec<Delivery> = scopes.iter().map(|s| s.2).collect();
    assert_eq!(
        deliveries,
        [
            Delivery::Enter,
            Delivery::Enter,
            Delivery::Leave,
            Delivery::Leave,
            Delivery::Enter,
            Delivery::Enter
        ]
    );
    let mut activations: Vec<u64> = scopes.iter().map(|s| s.1).collect();
    activations.dedup();
    assert_eq!(activations.len(), 3, "one identity per activation");
    assert!(activations[0] != activations[2], "a change back is another");
    // The one Task written while it implied Thing, routed as committed.
    assert_eq!(
        node_deliveries(&thing),
        vec![(Action::Put, Delivery::Apply)],
        "{thing:?}"
    );

    // A reader of the type itself: the type is in its scope before and
    // after, so each change applies.
    let task = events(&read_all(&db, &mut task));
    let task_scopes = scope_changes(&task);
    assert_eq!(task_scopes.len(), 6, "{task:?}");
    assert!(task_scopes.iter().all(|s| s.2 == Delivery::Apply));
    assert_eq!(
        scope_changes(&task)[..2],
        scopes[..2]
            .iter()
            .map(|s| (s.0, s.1, Delivery::Apply))
            .collect::<Vec<_>>()[..]
    );
}
