//! A fixed cut of nodes by logical interest, end to end: the base selects
//! nodes by the labels they carry in the cut's catalog (stored or implied by
//! their type), the continuation is the change feed read under the same
//! interest, and the reader goes on from the cursor the join hands it. A
//! projection built this way equals the one a continuous reader builds.

#![allow(clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;
use std::time::Duration;

use coordinode_core::budget::QueryBudget;
use coordinode_embed::Database;
use coordinode_embed::changes::{Action, ChangeCursor, ChangeInterest, Delivery, OwnedTarget};
use coordinode_embed::consumer::NodeCut;
use coordinode_storage::engine::partition::Partition;

/// A projection of node records by key.
type Nodes = BTreeMap<Vec<u8>, Vec<u8>>;

fn run(db: &mut Database, query: &str) {
    db.execute_cypher(query)
        .unwrap_or_else(|e| panic!("{query}: {e}"));
}

fn names(labels: &[&str]) -> Vec<String> {
    labels.iter().map(|s| (*s).to_string()).collect()
}

fn cut(db: &Database, labels: &[&str]) -> NodeCut {
    db.node_cut(Some(&names(labels)), Duration::from_secs(10))
        .expect("cut")
}

fn budget() -> QueryBudget {
    QueryBudget::new(1 << 24)
}

/// Copy the cut's view, one node read a page: a page selecting none goes
/// on, only a page reading none ends it. Returns the rows and how many
/// pages selected nothing yet were not the end.
fn copy_view(db: &Database, cut: &NodeCut) -> (Nodes, usize) {
    let budget = budget();
    let mut rows = Nodes::new();
    let mut after: Option<Vec<u8>> = None;
    let mut empty_pages = 0;
    loop {
        let page = db
            .node_cut_scan(cut, after.as_deref(), 1, &budget)
            .expect("page");
        let Some(next) = page.next_after() else {
            return (rows, empty_pages);
        };
        if page.rows().is_empty() {
            empty_pages += 1;
        }
        after = Some(next.to_vec());
        rows.extend(page.rows().iter().map(|(k, v)| (k.to_vec(), v.to_vec())));
    }
}

/// Apply a node event to `nodes` as a projection of its reader's scope does.
fn apply(nodes: &mut Nodes, event: &coordinode_embed::changes::ChangeEvent) {
    let OwnedTarget::Node(key) = &event.target else {
        return;
    };
    match (event.action, event.delivery) {
        (Action::Put, Delivery::Enter | Delivery::Apply) => {
            nodes.insert(key.clone(), event.payload.clone().expect("a put's record"));
        }
        (_, Delivery::Leave) | (Action::Remove, _) => {
            nodes.remove(key);
        }
        other => panic!("an event a node projection does not take here: {other:?}"),
    }
}

/// The end of what is committed now: where a cursor opened now starts.
fn log_end(db: &Database) -> u64 {
    db.change_cursor(&ChangeInterest::default())
        .expect("end")
        .position()
}

/// Read the cut's continuation into `nodes` until the join is complete and
/// the log is read to its end; the cursor the reader goes on with, and every
/// event read.
fn read_continuation(
    db: &Database,
    cut: &mut NodeCut,
    nodes: &mut Nodes,
) -> (ChangeCursor, Vec<coordinode_embed::changes::ChangeEvent>) {
    let end = log_end(db);
    let budget = budget();
    let mut events = Vec::new();
    loop {
        let batch = db.node_cut_tail(cut, 4, &budget, None).expect("tail");
        for unit in batch.units() {
            for event in &unit.events {
                apply(nodes, event);
                events.push(event.clone());
            }
        }
        if let Some(joined) = batch.joined()
            && joined.position() >= end
        {
            return (joined.clone(), events);
        }
    }
}

/// A continuous reader of `cursor` up to the end of what is committed now,
/// applied to `nodes`.
fn follow(db: &Database, cursor: &mut ChangeCursor, nodes: &mut Nodes) {
    let end = log_end(db);
    let budget = budget();
    while cursor.position() < end {
        let page = db.read_changes(cursor, 8, &budget, None).expect("read");
        for unit in &page.batch().units {
            for event in &unit.events {
                apply(nodes, event);
            }
        }
        *cursor = page.batch().cursor.clone();
    }
}

/// The node records the source holds now, of nodes `carries` keeps by their
/// stored labels.
fn source_nodes(
    db: &Database,
    carries: impl Fn(&[coordinode_core::schema::label_id::LabelId]) -> bool,
) -> Nodes {
    let engine = db.engine();
    engine
        .snapshot_prefix_scan(&engine.snapshot(), Partition::Node, b"node:")
        .expect("scan")
        .into_iter()
        .filter(|(_, value)| {
            carries(
                &coordinode_core::graph::node::NodeRecord::labels_from_msgpack(value)
                    .expect("labels"),
            )
        })
        .map(|(key, value)| (key, value.to_vec()))
        .collect()
}

fn label_id(db: &Database, name: &str) -> coordinode_core::schema::label_id::LabelId {
    db.change_cursor(&ChangeInterest {
        labels: Some(names(&[name])),
        edge_types: Some(Vec::new()),
        blobs: false,
    })
    .expect("cursor")
    .interest()
    .labels()
    .expect("labels")[0]
}

/// A cut of :Person nodes and its continuation across a RETYPE out of the
/// scope, a RETYPE into it, a delete, an update and a create build the same
/// projection as a reader that followed :Person from the start, and that
/// projection is the :Person nodes of the source. The base skips the
/// :Org node without ending, and the reader goes on with the cursor the
/// join hands it, bound to :Person's identity.
#[test]
fn a_node_cut_builds_what_a_continuous_reader_builds() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    for ddl in [
        "CREATE NODE TYPE Person WITH (name: STRING)",
        "CREATE NODE TYPE Org WITH (name: STRING)",
    ] {
        run(&mut db, ddl);
    }
    let mut continuous = db
        .change_cursor(&ChangeInterest {
            labels: Some(names(&["Person"])),
            edge_types: Some(Vec::new()),
            blobs: false,
        })
        .expect("cursor");
    run(
        &mut db,
        "CREATE (:Person {name: 'a'}), (:Org {name: 'o'}), (:Person {name: 'b'}), \
         (:Person {name: 'c'})",
    );

    let mut cut = cut(&db, &["Person"]);
    let (mut nodes, empty_pages) = copy_view(&db, &cut);
    assert_eq!(nodes.len(), 3, "the three :Person nodes");
    assert!(
        empty_pages >= 1,
        "the :Org node read, selected none, not the end"
    );

    run(&mut db, "MATCH (n:Person {name: 'a'}) RETYPE NODE n TO Org");
    run(&mut db, "MATCH (n:Org {name: 'o'}) RETYPE NODE n TO Person");
    run(&mut db, "MATCH (n:Person {name: 'b'}) DETACH DELETE n");
    run(&mut db, "MATCH (n:Person {name: 'c'}) SET n.name = 'c2'");
    run(&mut db, "CREATE (:Person {name: 'd'})");

    let (joined, events) = read_continuation(&db, &mut cut, &mut nodes);
    let mut followed = Nodes::new();
    follow(&db, &mut continuous, &mut followed);
    let person = label_id(&db, "Person");
    assert_eq!(nodes, followed, "the cut and a continuous reader agree");
    assert_eq!(nodes, source_nodes(&db, |l| l.contains(&person)));
    assert!(
        events
            .iter()
            .any(|e| e.action == Action::Put && e.delivery == Delivery::Leave),
        "the RETYPE out leaves: {events:?}"
    );
    assert!(
        events
            .iter()
            .any(|e| e.action == Action::Put && e.delivery == Delivery::Enter),
        "the RETYPE in enters: {events:?}"
    );
    assert_eq!(
        joined.interest(),
        continuous.interest(),
        "one bound interest"
    );
}

/// A change of what a type implies with no node written reaches the cut's
/// continuation as a scope change entering the scope, and a cut taken after
/// it selects the type's nodes through the implication in its catalog.
#[test]
fn a_change_of_what_a_type_implies_reaches_the_continuation() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    for ddl in [
        "CREATE NODE TYPE Thing WITH (title: STRING)",
        "CREATE NODE TYPE Task WITH (title: STRING)",
    ] {
        run(&mut db, ddl);
    }
    run(&mut db, "CREATE (:Task {title: 't'})");
    let mut before = cut(&db, &["Thing"]);
    let (mut nodes, _) = copy_view(&db, &before);
    assert!(nodes.is_empty(), "Task implies nothing yet");

    run(&mut db, "ALTER NODE TYPE Task SET IMPLIES (Thing)");
    let (_, events) = read_continuation(&db, &mut before, &mut nodes);
    let task = label_id(&db, "Task");
    let scopes: Vec<_> = events
        .iter()
        .filter_map(|e| match e.target {
            OwnedTarget::Scope { label, .. } => Some((e.action, label, e.delivery)),
            _ => None,
        })
        .collect();
    assert_eq!(
        scopes,
        vec![(Action::ScopeChange, task, Delivery::Enter)],
        "the activation is delivered though no node was written: {events:?}"
    );

    let after = cut(&db, &["Thing"]);
    let (implied, _) = copy_view(&db, &after);
    assert_eq!(
        implied,
        source_nodes(&db, |l| l.first() == Some(&task)),
        "the cut's catalog implies Thing for every Task"
    );
}

/// A label dropped and created again under its name between the page and
/// the join: the continuation removes the dropped label's nodes and follows
/// none of the new one's, the joined cursor stays bound to the identity the
/// cut bound, and a reader rebuilding from it keeps that identity. A new cut
/// by name binds the new label.
#[test]
fn a_label_dropped_and_created_again_keeps_the_cut_identity() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    run(
        &mut db,
        "CREATE TABLE Acct (id BIGINT PRIMARY KEY, email STRING)",
    );
    run(
        &mut db,
        "CREATE (:Acct {id: 1, email: 'a'}), (:Acct {id: 2, email: 'b'})",
    );
    let first = label_id(&db, "Acct");
    let mut old = cut(&db, &["Acct"]);
    let (mut nodes, _) = copy_view(&db, &old);
    assert_eq!(nodes.len(), 2);

    run(&mut db, "DROP TABLE Acct");
    run(
        &mut db,
        "CREATE TABLE Acct (id BIGINT PRIMARY KEY, email STRING)",
    );
    run(&mut db, "CREATE (:Acct {id: 3, email: 'c'})");
    let second = label_id(&db, "Acct");
    assert_ne!(first, second);

    let (joined, events) = read_continuation(&db, &mut old, &mut nodes);
    assert!(
        nodes.is_empty(),
        "the dropped label's nodes left: {nodes:?}"
    );
    assert!(
        events
            .iter()
            .any(|e| e.action == Action::Drop && e.target == OwnedTarget::Label(first)),
        "{events:?}"
    );
    assert_eq!(joined.interest().labels(), Some(&[first][..]));

    let rebuilt = db
        .node_cut_for(&joined, Duration::from_secs(10))
        .expect("rebuild");
    assert_eq!(rebuilt.interest(), joined.interest(), "the bound identity");
    assert!(copy_view(&db, &rebuilt).0.is_empty(), "no node carries it");

    let fresh = cut(&db, &["Acct"]);
    assert_eq!(fresh.interest().labels(), Some(&[second][..]));
    assert_eq!(copy_view(&db, &fresh).0.len(), 1, "the new label's node");
}

/// A :Person node written inside the join, above the view's bound: the view
/// lacks it, the continuation hands it over once as entering the scope, under
/// the identity a continuous reader receives it by, and the projection is the
/// source's.
#[test]
fn a_write_inside_the_join_enters_once_under_its_feed_identity() {
    use coordinode_core::graph::node::{NodeId, NodeRecord, encode_node_key};
    use coordinode_core::txn::proposal::{
        Implication, Mutation, NodeBasis, PartitionId, UnitEffects,
    };
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    run(&mut db, "CREATE NODE TYPE Person WITH (name: STRING)");
    run(&mut db, "CREATE (:Person {name: 'a'})");
    let person = label_id(&db, "Person");
    let mut continuous = db
        .change_cursor(&ChangeInterest {
            labels: Some(names(&["Person"])),
            edge_types: Some(Vec::new()),
            blobs: false,
        })
        .expect("cursor");
    let key = encode_node_key(0, NodeId::from_raw(1_000_000));
    let record = NodeRecord::with_labels(vec![person])
        .to_msgpack()
        .expect("record");
    let engine = db.engine_shared();
    let (hook_key, hook_record) = (key.clone(), record.clone());
    let mut once = true;
    db.engine().set_cut_bound_hook(Some(Box::new(move || {
        if std::mem::take(&mut once) {
            let mut effects = UnitEffects::new();
            effects.implications = vec![Implication {
                label: person,
                implies: Vec::new(),
            }];
            // A new node: it stored no labels before the unit.
            effects.nodes = vec![NodeBasis {
                key: hook_key.clone(),
                labels: Vec::new(),
            }];
            let unit = coordinode_storage::engine::scope::store_unit([
                Mutation::Put {
                    partition: PartitionId::Node,
                    key: hook_key.clone(),
                    value: hook_record.clone(),
                },
                Mutation::Effects(effects),
            ]);
            let ts = engine.oracle().expect("embedded").next().as_raw();
            engine
                .commit_logged(
                    &unit,
                    ts,
                    coordinode_core::txn::write_concern::Journal::Journal,
                )
                .expect("commit");
        }
    })));
    let mut cut = cut(&db, &["Person"]);
    db.engine().set_cut_bound_hook(None);
    assert!(
        cut.join_end() > cut.cursor().position(),
        "the join holds it"
    );
    let (mut nodes, _) = copy_view(&db, &cut);
    assert!(!nodes.contains_key(&key), "above the view's bound");
    let (_, events) = read_continuation(&db, &mut cut, &mut nodes);
    let mut followed = Nodes::new();
    let mut fed = Vec::new();
    {
        let end = log_end(&db);
        let budget = budget();
        while continuous.position() < end {
            let page = db
                .read_changes(&continuous, 8, &budget, None)
                .expect("read");
            for unit in &page.batch().units {
                fed.extend(unit.events.iter().cloned());
            }
            continuous = page.batch().cursor.clone();
        }
    }
    for event in &fed {
        apply(&mut followed, event);
    }
    let joined: Vec<_> = events
        .iter()
        .filter(|e| e.target == OwnedTarget::Node(key.clone()))
        .collect();
    assert_eq!(joined.len(), 1, "once: {events:?}");
    assert_eq!(joined[0].delivery, Delivery::Enter);
    let in_feed = fed
        .iter()
        .find(|e| e.target == OwnedTarget::Node(key.clone()))
        .expect("in the feed");
    assert_eq!(joined[0].id, in_feed.id, "the feed's identity");
    assert_eq!(nodes.get(&key), Some(&record));
    assert_eq!(nodes, source_nodes(&db, |l| l.contains(&person)));
}

/// A continuation whose log was reclaimed is refused with RetentionLost and
/// does not move: the reader has not read to the log's end, so it publishes
/// nothing and its previous projection stays; a new cut recovers the state.
#[test]
fn a_continuation_past_retention_is_refused_and_moves_nothing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    run(&mut db, "CREATE NODE TYPE Event WITH (i: INT)");
    let mut cut = cut(&db, &["Event"]);
    for i in 0..4 {
        run(&mut db, &format!("CREATE (:Event {{i: {i}}})"));
    }
    db.persist().expect("persist");
    db.engine().reclaim_footprint();
    let before = cut.cursor().clone();
    let budget = budget();
    let refused = db.node_cut_tail(&mut cut, 4, &budget, None);
    let refusal = format!("{:?}", refused.as_ref().err());
    assert!(refusal.contains("RetentionLost"), "{refusal}");
    assert_eq!(cut.cursor(), &before, "the refusal does not move it");
    let again = db
        .node_cut(Some(&names(&["Event"])), Duration::from_secs(10))
        .expect("a new cut");
    assert_eq!(copy_view(&db, &again).0.len(), 4);
}

/// A cut of nodes refuses to rebuild a reader that also follows
/// relationships: their state is not in a node view.
#[test]
fn a_reader_of_relationships_is_not_rebuilt_from_a_node_cut() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");
    // A cut founds the history a cursor counts entries of.
    drop(db.node_cut(None, Duration::from_secs(10)).expect("cut"));
    let cursor = db
        .change_cursor(&ChangeInterest::default())
        .expect("cursor");
    assert!(
        db.node_cut_for(&cursor, Duration::from_secs(10)).is_err(),
        "relationships and blobs are not in a node view"
    );
}
