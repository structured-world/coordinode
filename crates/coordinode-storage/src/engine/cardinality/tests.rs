use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use crate::engine::transaction::{CommitContext, CommitError};
use coordinode_core::graph::cardinality::{CardinalityBound, IncidentCount};
use coordinode_core::schema::definition::{
    EdgeTypeSchema, PropertyDef, PropertyType, encode_edge_type_current_revision_key,
    encode_edge_type_schema_key,
};
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_core::txn::write_concern::WriteConcern;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use tempfile::TempDir;

const OWNS: &str = "OWNS";

struct Fixture {
    engine: Arc<StorageEngine>,
    oracle: Arc<TimestampOracle>,
    dir: TempDir,
}

fn open_at(dir: TempDir) -> Fixture {
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path().to_string_lossy().as_ref(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let oracle = Arc::new(TimestampOracle::new());
    let engine = Arc::new(StorageEngine::open_with_oracle(&config, oracle.clone()).unwrap());
    Fixture {
        engine,
        oracle,
        dir,
    }
}

fn fixture() -> Fixture {
    open_at(tempfile::tempdir().unwrap())
}

fn txn<'a>(f: &'a Fixture) -> Transaction<'a> {
    let snap = f.engine.snapshot();
    Transaction::new(
        &f.engine,
        Some(&f.oracle),
        Timestamp::from_raw(snap),
        Some(snap),
    )
}

fn commit(txn: &mut Transaction<'_>) -> Result<(), CommitError> {
    let wc = WriteConcern::default();
    let ctx = CommitContext {
        write_concern: &wc,
        pipeline: None,
        id_gen: None,
        drain_buffer: None,
        nvme_write_buffer: None,
    };
    txn.commit(&ctx).map(|_| ())
}

fn node(id: u64) -> NodeId {
    NodeId::from_raw(id)
}

fn descriptor(
    direction: Direction,
    measure: CardinalityMeasure,
    bound: CardinalityBound,
) -> CardinalityDescriptor {
    CardinalityDescriptor {
        edge_type: OWNS.to_string(),
        direction,
        measure,
        bound,
        schema_generation: 1,
    }
}

/// Every direction and measure, so one history checks all four counts.
fn all_descriptors() -> Vec<CardinalityDescriptor> {
    let mut all = Vec::new();
    for direction in [Direction::Outgoing, Direction::Incoming] {
        for measure in [
            CardinalityMeasure::EdgeInstances,
            CardinalityMeasure::DistinctNeighbours,
        ] {
            all.push(descriptor(direction, measure, CardinalityBound::AtLeastOne));
        }
    }
    all
}

/// Publish `OWNS`, discriminated by a string `context` when
/// `discriminated`, declaring `descriptors`, with its profile as the schema
/// store writes it.
fn declare(f: &Fixture, discriminated: bool, descriptors: &[CardinalityDescriptor]) {
    let mut schema = EdgeTypeSchema::new(OWNS);
    schema.add_property(PropertyDef::new("context", PropertyType::String).not_null());
    schema
        .resolve_identity(discriminated.then_some("context"))
        .unwrap();
    for d in descriptors {
        schema.declare_cardinality(d.clone()).unwrap();
    }
    let e = &f.engine;
    e.put(
        Partition::Schema,
        &encode_edge_type_schema_key(OWNS, 1),
        &schema.to_msgpack().unwrap(),
    )
    .unwrap();
    e.put(
        Partition::Schema,
        &encode_edge_type_current_revision_key(OWNS),
        &1u64.to_be_bytes(),
    )
    .unwrap();
    let profile = schema.cardinality_profile().unwrap();
    e.put(
        Partition::Schema,
        &encode_cardinality_profile_key(OWNS),
        &profile.to_msgpack().unwrap(),
    )
    .unwrap();
}

/// Rebuild and cover every descriptor in one commit each.
fn cover(f: &Fixture, descriptors: &[CardinalityDescriptor]) {
    for d in descriptors {
        let mut t = txn(f);
        rebuild(&mut t, d).unwrap();
        commit(&mut t).unwrap();
    }
}

fn instance_key(source: u64, target: u64, disc: &[u8]) -> Vec<u8> {
    let mut key = temporal_edgeprop_pair_prefix(OWNS, node(source), node(target));
    key.extend_from_slice(disc);
    key
}

/// Stage one instance of `(source, target)`: both adjacency sides, and its
/// entry when it has an identity.
fn add(t: &mut Transaction<'_>, source: u64, target: u64, disc: Option<&[u8]>) {
    t.merge_adj_add(&encode_adj_key_forward(OWNS, node(source)), target);
    t.merge_adj_add(&encode_adj_key_reverse(OWNS, node(target)), source);
    if let Some(disc) = disc {
        t.put(
            Partition::EdgeProp,
            &instance_key(source, target, disc),
            b"facets",
        )
        .unwrap();
    }
}

/// Stage removal of one instance; the adjacency goes with the last one.
fn remove(t: &mut Transaction<'_>, source: u64, target: u64, disc: Option<&[u8]>, last: bool) {
    if let Some(disc) = disc {
        t.delete(Partition::EdgeProp, &instance_key(source, target, disc))
            .unwrap();
    }
    if last {
        t.merge_adj_remove(&encode_adj_key_forward(OWNS, node(source)), target);
        t.merge_adj_remove(&encode_adj_key_reverse(OWNS, node(target)), source);
    }
}

/// The counts the engine keeps for `d`, non-zero only.
fn kept(f: &Fixture, d: &CardinalityDescriptor) -> BTreeMap<NodeId, i64> {
    let mut out = BTreeMap::new();
    for guard in f
        .engine
        .prefix_scan(Partition::Counter, &d.counter_prefix())
        .unwrap()
    {
        let (key, value) = guard.into_inner().unwrap();
        let count = crate::engine::merge::decode_counter(&value).unwrap();
        if count != 0 {
            out.insert(d.counter_node(&key).unwrap(), count);
        }
    }
    out
}

/// An independent model: the instance identities of each ordered pair.
#[derive(Default, Clone)]
struct Model {
    pairs: BTreeMap<(u64, u64), BTreeSet<Vec<u8>>>,
}

impl Model {
    /// What `d` counts per node, by its definition over the set E: every
    /// identity for instances, every opposite endpoint for neighbours.
    fn counts(&self, d: &CardinalityDescriptor) -> BTreeMap<NodeId, i64> {
        /// One scope's identities, as (neighbour, key), and its neighbours.
        type Scope = (BTreeSet<(u64, Vec<u8>)>, BTreeSet<u64>);
        let mut per_node: BTreeMap<u64, Scope> = BTreeMap::new();
        for ((s, t), keys) in &self.pairs {
            if keys.is_empty() {
                continue;
            }
            let (scope, other) = match d.direction {
                Direction::Outgoing => (*s, *t),
                Direction::Incoming => (*t, *s),
            };
            let entry = per_node.entry(scope).or_default();
            for key in keys {
                entry.0.insert((other, key.clone()));
            }
            entry.1.insert(other);
        }
        per_node
            .into_iter()
            .map(|(n, (instances, neighbours))| {
                let count = match d.measure {
                    CardinalityMeasure::EdgeInstances => instances.len(),
                    CardinalityMeasure::DistinctNeighbours => neighbours.len(),
                };
                (node(n), count as i64)
            })
            .collect()
    }
}

/// An edge with no properties is an instance and is counted: the count does
/// not depend on a facet row existing, which a property-less edge has none
/// of.
#[test]
fn a_property_less_single_edge_is_counted() {
    let f = fixture();
    let ds = all_descriptors();
    declare(&f, false, &ds);
    cover(&f, &ds);

    let mut t = txn(&f);
    add(&mut t, 1, 2, None);
    add(&mut t, 1, 3, None);
    commit(&mut t).unwrap();

    let out_instances = &ds[0];
    let in_neighbours = &ds[3];
    assert_eq!(kept(&f, out_instances), BTreeMap::from([(node(1), 2)]));
    assert_eq!(
        kept(&f, in_neighbours),
        BTreeMap::from([(node(2), 1), (node(3), 1)])
    );
}

/// Two discriminated instances to one neighbour are two instances and one
/// neighbour, and an instance rewritten in place is the same identity, not
/// a third.
#[test]
fn parallel_instances_count_once_per_neighbour_and_upserts_are_not_new() {
    let f = fixture();
    let ds = all_descriptors();
    declare(&f, true, &ds);
    cover(&f, &ds);

    let mut t = txn(&f);
    add(&mut t, 1, 2, Some(b"work"));
    add(&mut t, 1, 2, Some(b"college"));
    commit(&mut t).unwrap();

    // The same identity written again, as a MERGE or a replay writes it.
    let mut t = txn(&f);
    add(&mut t, 1, 2, Some(b"work"));
    commit(&mut t).unwrap();

    assert_eq!(kept(&f, &ds[0]), BTreeMap::from([(node(1), 2)]));
    assert_eq!(kept(&f, &ds[1]), BTreeMap::from([(node(1), 1)]));
}

/// A count is not read until its rebuild covered it, and the rebuild counts
/// the edges stored before the constraint existed.
#[test]
fn counts_are_evidence_only_after_the_rebuild_covers_them() {
    let f = fixture();
    // Edges written while the type declared nothing: no count kept.
    let mut t = txn(&f);
    add(&mut t, 1, 2, None);
    add(&mut t, 1, 3, None);
    commit(&mut t).unwrap();

    let d = descriptor(
        Direction::Outgoing,
        CardinalityMeasure::EdgeInstances,
        CardinalityBound::AtMostOne,
    );
    declare(&f, false, std::slice::from_ref(&d));
    let none = FxHashMap::default();
    assert_eq!(
        kept_count(&f.engine, node(1), OWNS, d.direction, d.measure, &none).unwrap(),
        None,
        "an uncovered count is never read, however it got there"
    );

    let mut t = txn(&f);
    let rebuilt = rebuild(&mut t, &d).unwrap();
    commit(&mut t).unwrap();
    assert_eq!(
        rebuilt,
        Rebuilt {
            scopes: 1,
            corrected: 1
        }
    );
    assert_eq!(
        kept_count(&f.engine, node(1), OWNS, d.direction, d.measure, &none).unwrap(),
        Some(Ok(2))
    );

    // A second rebuild of the same descriptor would correct it again.
    let mut t = txn(&f);
    rebuild(&mut t, &d).unwrap();
    assert!(
        commit(&mut t).is_err(),
        "the coverage marker is written once, so a second rebuild is refused"
    );
    assert_eq!(kept(&f, &d), BTreeMap::from([(node(1), 2)]));
}

/// Commits landing between the declaration and the rebuild keep their own
/// changes, and the rebuild corrects only what the snapshot shows missing,
/// so the result is exact either way the two land.
#[test]
fn a_rebuild_composes_with_changes_counted_before_it() {
    let f = fixture();
    let mut t = txn(&f);
    add(&mut t, 1, 2, None);
    commit(&mut t).unwrap();

    let d = descriptor(
        Direction::Outgoing,
        CardinalityMeasure::DistinctNeighbours,
        CardinalityBound::AtLeastOne,
    );
    declare(&f, false, std::slice::from_ref(&d));
    // Counted from here on, but the count started from nothing.
    let mut t = txn(&f);
    add(&mut t, 1, 3, None);
    add(&mut t, 4, 5, None);
    commit(&mut t).unwrap();
    assert_eq!(kept(&f, &d), BTreeMap::from([(node(1), 1), (node(4), 1)]));

    cover(&f, std::slice::from_ref(&d));
    assert_eq!(kept(&f, &d), BTreeMap::from([(node(1), 2), (node(4), 1)]));
}

/// Two first instances of one pair cannot both derive their change from the
/// pair being empty: the neighbour would be counted twice. The second is
/// refused while the first is decided, and counted right on retry.
#[test]
fn two_first_instances_of_one_pair_count_one_neighbour() {
    let f = fixture();
    let ds = all_descriptors();
    declare(&f, true, &ds);
    cover(&f, &ds);

    let mut first = txn(&f);
    let mut second = txn(&f);
    add(&mut first, 1, 2, Some(b"a"));
    add(&mut second, 1, 2, Some(b"b"));

    // A commit holds its reservation only while it runs, and these run on
    // one thread, so the first's is held here explicitly: it is the claim
    // its commit states on the pair it counts.
    let mut first_claims = coordinode_core::txn::invariant::ClaimSet::new();
    first_claims.insert(coordinode_core::txn::invariant::Claim::new(
        coordinode_core::txn::invariant::ClaimScope::Pair {
            source: node(1),
            target: node(2),
            edge_type: OWNS.to_string(),
        },
        ClaimPredicate::PairCounted,
        first.schema_generation(),
    ));
    let held = f
        .engine
        .claim_registry()
        .reserve_attempt(&first_claims)
        .unwrap();
    let refused = commit(&mut second).expect_err("the pair is being counted");
    assert!(matches!(refused, CommitError::InvariantRefused { .. }));
    drop(held);

    commit(&mut first).unwrap();
    let mut retry = txn(&f);
    add(&mut retry, 1, 2, Some(b"b"));
    commit(&mut retry).unwrap();

    assert_eq!(kept(&f, &ds[0]), BTreeMap::from([(node(1), 2)]));
    assert_eq!(kept(&f, &ds[1]), BTreeMap::from([(node(1), 1)]));
}

/// Two removals of the last two instances cannot each assume the other
/// keeps the pair adjacent. Whichever lands second sees one instance left
/// and its removal leaving the adjacency with nothing behind it, and is
/// refused rather than counted from a pair that disagrees with itself.
#[test]
fn removing_the_last_two_instances_apart_cannot_leave_a_dangling_pair() {
    let f = fixture();
    let ds = all_descriptors();
    declare(&f, true, &ds);
    cover(&f, &ds);
    let mut t = txn(&f);
    add(&mut t, 1, 2, Some(b"a"));
    add(&mut t, 1, 2, Some(b"b"));
    commit(&mut t).unwrap();

    let mut first = txn(&f);
    let mut second = txn(&f);
    remove(&mut first, 1, 2, Some(b"a"), false);
    remove(&mut second, 1, 2, Some(b"b"), false);
    commit(&mut first).unwrap();
    let refused = commit(&mut second).expect_err("the pair would keep adjacency alone");
    assert!(matches!(refused, CommitError::InvariantRefused { .. }));

    assert_eq!(kept(&f, &ds[0]), BTreeMap::from([(node(1), 1)]));
    assert_eq!(kept(&f, &ds[1]), BTreeMap::from([(node(1), 1)]));
}

/// First instances into different pairs of one node are independent and
/// both counted.
#[test]
fn first_instances_into_different_pairs_both_count() {
    let f = fixture();
    let ds = all_descriptors();
    declare(&f, false, &ds);
    cover(&f, &ds);

    let mut first = txn(&f);
    let mut second = txn(&f);
    add(&mut first, 1, 2, None);
    add(&mut second, 1, 3, None);
    commit(&mut first).unwrap();
    commit(&mut second).unwrap();

    assert_eq!(kept(&f, &ds[1]), BTreeMap::from([(node(1), 2)]));
}

/// A bound over a covered count is decided from the count the commit
/// leaves: an EXACTLY ONE replacement in one commit holds, a second edge
/// breaks AT MOST ONE.
#[test]
fn bounds_are_decided_from_the_kept_count_of_the_post_state() {
    use coordinode_core::txn::invariant::{Claim, ClaimPredicate, ClaimScope};

    let f = fixture();
    let d = descriptor(
        Direction::Outgoing,
        CardinalityMeasure::EdgeInstances,
        CardinalityBound::ExactlyOne,
    );
    declare(&f, false, std::slice::from_ref(&d));
    cover(&f, std::slice::from_ref(&d));
    let bound = |bound| {
        Claim::new(
            ClaimScope::Incident {
                node: node(1),
                edge_type: OWNS.to_string(),
                direction: Direction::Outgoing,
            },
            ClaimPredicate::CardinalityBound {
                measure: CardinalityMeasure::EdgeInstances,
                bound,
            },
            0,
        )
    };

    let mut t = txn(&f);
    add(&mut t, 1, 2, None);
    t.claim(bound(CardinalityBound::ExactlyOne));
    commit(&mut t).unwrap();

    // Replace the one edge with another: the post-state still has one.
    let mut t = txn(&f);
    remove(&mut t, 1, 2, None, true);
    add(&mut t, 1, 3, None);
    t.claim(bound(CardinalityBound::ExactlyOne));
    commit(&mut t).unwrap();

    let mut t = txn(&f);
    add(&mut t, 1, 4, None);
    t.claim(bound(CardinalityBound::AtMostOne));
    let err = commit(&mut t).expect_err("a second edge breaks at most one");
    assert!(matches!(err, CommitError::InvariantRefused { .. }));
    assert_eq!(kept(&f, &d), BTreeMap::from([(node(1), 1)]));

    // Removing the only edge breaks exactly one.
    let mut t = txn(&f);
    remove(&mut t, 1, 3, None, true);
    t.claim(bound(CardinalityBound::ExactlyOne));
    assert!(commit(&mut t).is_err());
}

/// A kept count below zero is corruption, and no bound is decided from it,
/// however the bound reads.
#[test]
fn a_count_below_zero_decides_no_bound() {
    use coordinode_core::txn::invariant::{Claim, ClaimPredicate, ClaimScope};

    let f = fixture();
    let d = descriptor(
        Direction::Outgoing,
        CardinalityMeasure::EdgeInstances,
        CardinalityBound::AtMostOne,
    );
    declare(&f, false, std::slice::from_ref(&d));
    cover(&f, std::slice::from_ref(&d));
    f.engine
        .merge(
            Partition::Counter,
            &d.counter_key(node(1)),
            &crate::engine::merge::encode_counter_delta(-3),
        )
        .unwrap();

    let mut t = txn(&f);
    add(&mut t, 1, 2, None);
    t.claim(Claim::new(
        ClaimScope::Incident {
            node: node(1),
            edge_type: OWNS.to_string(),
            direction: Direction::Outgoing,
        },
        ClaimPredicate::CardinalityBound {
            measure: CardinalityMeasure::EdgeInstances,
            bound: CardinalityBound::AtMostOne,
        },
        0,
    ));
    assert!(matches!(
        commit(&mut t),
        Err(CommitError::InvariantRefused { .. })
    ));
}

/// A direct write path applies its point writes as it goes, leaving no
/// state before them to count from, so it is refused for a counted type
/// rather than leaving the count behind.
#[test]
fn a_direct_write_to_a_counted_type_is_refused() {
    let f = fixture();
    let ds = all_descriptors();
    declare(&f, false, &ds);
    let mut t = Transaction::new(&f.engine, None, Timestamp::from_raw(0), None);
    add(&mut t, 1, 2, None);
    assert!(matches!(
        commit(&mut t),
        Err(CommitError::InvariantRefused { .. })
    ));
}

/// A whole posting dropped with its node takes every pair it held, and the
/// opposite scopes lose their neighbour.
#[test]
fn a_dropped_posting_counts_every_pair_it_held() {
    let f = fixture();
    let ds = all_descriptors();
    declare(&f, false, &ds);
    cover(&f, &ds);
    let mut t = txn(&f);
    add(&mut t, 1, 2, None);
    add(&mut t, 1, 3, None);
    commit(&mut t).unwrap();

    let mut t = txn(&f);
    t.delete(Partition::Adj, &encode_adj_key_forward(OWNS, node(1)))
        .unwrap();
    t.merge_adj_remove(&encode_adj_key_reverse(OWNS, node(2)), 1);
    t.merge_adj_remove(&encode_adj_key_reverse(OWNS, node(3)), 1);
    commit(&mut t).unwrap();

    for d in &ds {
        assert_eq!(kept(&f, d), BTreeMap::new(), "{d:?}");
    }
}

/// Counts and their coverage are durable state: a reopened engine reads the
/// same counts and keeps counting from them.
#[test]
fn counts_survive_a_reopen() {
    let f = fixture();
    let ds = all_descriptors();
    declare(&f, true, &ds);
    cover(&f, &ds);
    let mut t = txn(&f);
    add(&mut t, 1, 2, Some(b"a"));
    add(&mut t, 3, 2, Some(b"a"));
    commit(&mut t).unwrap();
    f.engine.persist().unwrap();

    let Fixture { engine, dir, .. } = f;
    drop(engine);
    let f = open_at(dir);
    let none = FxHashMap::default();
    assert_eq!(
        kept_count(
            &f.engine,
            node(2),
            OWNS,
            Direction::Incoming,
            CardinalityMeasure::DistinctNeighbours,
            &none
        )
        .unwrap(),
        Some(Ok(2))
    );
    let mut t = txn(&f);
    remove(&mut t, 3, 2, Some(b"a"), true);
    commit(&mut t).unwrap();
    assert_eq!(kept(&f, &ds[3]), BTreeMap::from([(node(2), 1)]));
}

/// One step of a random history.
#[derive(Debug, Clone)]
enum Step {
    Add(u64, u64, u8),
    Remove(u64, u64, u8),
    Redirect(u64, u64, u64, u8),
}

fn step() -> impl proptest::strategy::Strategy<Value = Step> {
    use proptest::prelude::*;
    let n = 1u64..5;
    prop_oneof![
        (n.clone(), n.clone(), 0u8..3).prop_map(|(s, t, d)| Step::Add(s, t, d)),
        (n.clone(), n.clone(), 0u8..3).prop_map(|(s, t, d)| Step::Remove(s, t, d)),
        (n.clone(), n.clone(), n, 0u8..3).prop_map(|(s, t, u, d)| Step::Redirect(s, t, u, d)),
    ]
}

/// Stage `step` against the model's current pairs, applying it to the model
/// the way the commit should leave the graph.
fn stage(t: &mut Transaction<'_>, model: &mut Model, step: &Step, discriminated: bool) {
    let disc = |d: u8| {
        if discriminated {
            vec![b'k', d]
        } else {
            Vec::new()
        }
    };
    fn identity(discriminated: bool, key: &[u8]) -> Option<&[u8]> {
        discriminated.then_some(key)
    }
    let id = |key| identity(discriminated, key);
    match step {
        Step::Add(s, tg, d) => {
            let key = disc(*d);
            add(t, *s, *tg, id(&key));
            model.pairs.entry((*s, *tg)).or_default().insert(key);
        }
        Step::Remove(s, tg, d) => {
            let key = disc(*d);
            let Some(keys) = model.pairs.get_mut(&(*s, *tg)) else {
                return;
            };
            if !keys.remove(&key) {
                return;
            }
            remove(t, *s, *tg, id(&key), keys.is_empty());
        }
        Step::Redirect(s, from, to, d) => {
            if from == to {
                return;
            }
            let key = disc(*d);
            let present = model
                .pairs
                .get(&(*s, *from))
                .is_some_and(|keys| keys.contains(&key));
            let taken = model
                .pairs
                .get(&(*s, *to))
                .is_some_and(|keys| keys.contains(&key));
            if !present || taken {
                return;
            }
            let keys = model.pairs.get_mut(&(*s, *from)).unwrap();
            keys.remove(&key);
            remove(t, *s, *from, id(&key), keys.is_empty());
            add(t, *s, *to, id(&key));
            model.pairs.entry((*s, *to)).or_default().insert(key);
        }
    }
}

/// The same step over nodes moved up by `base`, so every history of one
/// property runs on its own nodes of one engine.
fn shift(step: &Step, base: u64) -> Step {
    match *step {
        Step::Add(s, t, d) => Step::Add(s + base, t + base, d),
        Step::Remove(s, t, d) => Step::Remove(s + base, t + base, d),
        Step::Redirect(s, t, u, d) => Step::Redirect(s + base, t + base, u + base, d),
    }
}

/// Run every generated history against one engine, checking after each
/// commit that the kept counts are the set model's, and at the end that an
/// enumeration of the stored edges agrees with both.
fn differential(discriminated: bool) {
    use proptest::test_runner::{Config, TestRunner};

    let f = fixture();
    let ds = all_descriptors();
    declare(&f, discriminated, &ds);
    cover(&f, &ds);
    let histories = proptest::collection::vec(proptest::collection::vec(step(), 1..5), 1..12);
    let next_base = std::sync::atomic::AtomicU64::new(0);
    let mut runner = TestRunner::new(Config::with_cases(48));
    runner
        .run(&histories, |steps| {
            // Steps name nodes 1..5; each history gets the next ten.
            let base = next_base.fetch_add(10, std::sync::atomic::Ordering::Relaxed) + 10;
            let mine = |n: &NodeId| n.as_raw() > base && n.as_raw() <= base + 5;
            let mut model = Model::default();
            for batch in &steps {
                let mut t = txn(&f);
                let mut next = model.clone();
                for step in batch {
                    stage(&mut t, &mut next, &shift(step, base), discriminated);
                }
                commit(&mut t).unwrap();
                model = next;
                for d in &ds {
                    let mut ours = kept(&f, d);
                    ours.retain(|n, _| mine(n));
                    proptest::prop_assert_eq!(&ours, &model.counts(d), "{:?} after {:?}", d, batch);
                }
            }
            let none = std::collections::HashMap::new();
            for d in &ds {
                for (n, count) in model.counts(d) {
                    let counted: IncidentCount =
                        enumerated_count(&f.engine, n, OWNS, d.direction, &[], &none)
                            .unwrap()
                            .unwrap();
                    proptest::prop_assert_eq!(counted.of(d.measure) as i64, count);
                }
            }
            Ok(())
        })
        .unwrap();
}

/// Every count of a discriminated type follows an independent set model
/// through random histories of adds, removes, replays and redirects, several
/// to a commit.
#[test]
fn discriminated_counts_follow_the_set_model() {
    differential(true);
}

/// The same for a single-edge type, whose pair is its one instance.
#[test]
fn single_edge_counts_follow_the_set_model() {
    differential(false);
}
