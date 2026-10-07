use std::collections::{BTreeMap, BTreeSet};

use proptest::prelude::*;

use super::*;

fn n(id: u64) -> NodeId {
    NodeId::from_raw(id)
}

fn at(neighbour: u64, key: &[u8]) -> IncidentInstance {
    IncidentInstance {
        neighbour: n(neighbour),
        key: key.to_vec(),
    }
}

/// Two identities to one neighbour are two instances and one neighbour; a
/// repeated identity counts once; an ordinary property-less edge (empty key)
/// is an instance like any other.
#[test]
fn the_two_measures_count_the_same_set_differently() {
    let count = IncidentCount::of_set([
        at(2, b"work"),
        at(2, b"college"),
        at(2, b"work"),
        at(3, b""),
    ]);
    assert_eq!(
        count,
        IncidentCount {
            instances: 3,
            neighbours: 2
        }
    );
    assert_eq!(count.of(CardinalityMeasure::EdgeInstances), 3);
    assert_eq!(count.of(CardinalityMeasure::DistinctNeighbours), 2);
    assert_eq!(IncidentCount::of_set([]), IncidentCount::default());
}

/// First, non-last and last transitions: only the first identity makes the
/// pair adjacent and only the last one's removal ends it; a repeated write
/// and the removal of an absent identity change nothing.
#[test]
fn a_pair_joins_on_its_first_identity_and_leaves_on_its_last() {
    let mut pair = PairInstances::default();
    let first = pair.upsert(b"a".to_vec());
    assert_eq!(first.membership, Membership::Joined);
    assert_eq!(first.delta(CardinalityMeasure::EdgeInstances), 1);
    assert_eq!(first.delta(CardinalityMeasure::DistinctNeighbours), 1);

    let second = pair.upsert(b"b".to_vec());
    assert_eq!(second.membership, Membership::Unchanged);
    assert_eq!(second.delta(CardinalityMeasure::EdgeInstances), 1);
    assert_eq!(second.delta(CardinalityMeasure::DistinctNeighbours), 0);

    let replay = pair.upsert(b"a".to_vec());
    assert_eq!(
        replay,
        PairTransition::NONE,
        "same identity is one instance"
    );
    assert_eq!(pair.remove(b"zzz"), PairTransition::NONE, "absent identity");

    let non_last = pair.remove(b"a");
    assert_eq!(non_last.membership, Membership::Unchanged);
    assert_eq!(non_last.delta(CardinalityMeasure::EdgeInstances), -1);
    let last = pair.remove(b"b");
    assert_eq!(last.membership, Membership::Left);
    assert_eq!(last.delta(CardinalityMeasure::DistinctNeighbours), -1);
    assert!(!pair.is_adjacent());
}

/// Clearing a pair removes every identity at once and ends adjacency; an
/// empty pair clears to nothing.
#[test]
fn clearing_a_pair_removes_every_identity() {
    let mut pair = PairInstances::from_keys([b"a".to_vec(), b"b".to_vec(), b"c".to_vec()]);
    let cleared = pair.clear();
    assert_eq!(cleared.removed, 3);
    assert_eq!(cleared.delta(CardinalityMeasure::EdgeInstances), -3);
    assert_eq!(cleared.membership, Membership::Left);
    assert_eq!(pair.clear(), PairTransition::NONE);
}

/// A redirect keeps the identity: the instance count of a scope holding
/// both pairs is unchanged while the neighbour count moves. A destination
/// already holding the identity refuses the move and both pairs stay as
/// they were; moving an identity the source lacks changes nothing.
#[test]
fn a_redirect_moves_one_identity_and_refuses_a_collision() {
    let mut from = PairInstances::from_keys([b"a".to_vec(), b"b".to_vec()]);
    let mut to = PairInstances::default();
    let (left, joined) = from.move_to(&mut to, b"a").expect("move");
    assert_eq!(
        left.delta(CardinalityMeasure::EdgeInstances)
            + joined.delta(CardinalityMeasure::EdgeInstances),
        0
    );
    assert_eq!(left.membership, Membership::Unchanged);
    assert_eq!(joined.membership, Membership::Joined);

    let mut held = PairInstances::from_keys([b"b".to_vec()]);
    let collision = from.move_to(&mut held, b"b").expect_err("collision");
    assert_eq!(collision.key, b"b".to_vec());
    assert!(from.contains(b"b") && held.contains(b"b"), "nothing moved");

    assert_eq!(
        from.move_to(&mut to, b"missing").expect("absent"),
        (PairTransition::NONE, PairTransition::NONE)
    );
}

/// A descriptor names the neighbour of a pair only on its own side of the
/// constrained node; a self-loop is counted once on each side.
#[test]
fn a_descriptor_reads_the_neighbour_on_its_side() {
    let descriptor = |direction| CardinalityDescriptor {
        edge_type: "KNOWS".into(),
        direction,
        measure: CardinalityMeasure::DistinctNeighbours,
        bound: CardinalityBound::AtMostOne,
        schema_generation: 1,
    };
    let out = descriptor(Direction::Outgoing);
    let inc = descriptor(Direction::Incoming);
    assert_eq!(out.neighbour_in_scope(n(1), n(1), n(2)), Some(n(2)));
    assert_eq!(out.neighbour_in_scope(n(2), n(1), n(2)), None);
    assert_eq!(inc.neighbour_in_scope(n(2), n(1), n(2)), Some(n(1)));
    assert_eq!(out.neighbour_in_scope(n(5), n(5), n(5)), Some(n(5)));
    assert_eq!(inc.neighbour_in_scope(n(5), n(5), n(5)), Some(n(5)));
}

// ── Differential check against an independent set model ─────────────────

#[derive(Debug, Clone)]
enum Op {
    Upsert(u64, u64, u8),
    Remove(u64, u64, u8),
    Clear(u64, u64),
    Redirect(u64, u64, u8, u64, u64),
}

fn op() -> impl Strategy<Value = Op> {
    // Few nodes and keys, so pairs fill, empty, collide and self-loop often.
    let node = 1u64..4;
    let key = 0u8..3;
    prop_oneof![
        (node.clone(), node.clone(), key.clone()).prop_map(|(s, t, k)| Op::Upsert(s, t, k)),
        (node.clone(), node.clone(), key.clone()).prop_map(|(s, t, k)| Op::Remove(s, t, k)),
        (node.clone(), node.clone()).prop_map(|(s, t)| Op::Clear(s, t)),
        (node.clone(), node.clone(), key, node.clone(), node)
            .prop_map(|(s, t, k, s2, t2)| Op::Redirect(s, t, k, s2, t2)),
    ]
}

/// The system under test: pairs plus counts maintained only from the
/// transitions' deltas, the way a writer maintains them.
#[derive(Default)]
struct Maintained {
    pairs: BTreeMap<(u64, u64), PairInstances>,
    counts: BTreeMap<(u64, Direction, CardinalityMeasure), i64>,
}

impl Maintained {
    fn apply_delta(&mut self, source: u64, target: u64, transition: PairTransition) {
        for (node, direction) in [(source, Direction::Outgoing), (target, Direction::Incoming)] {
            for measure in [
                CardinalityMeasure::EdgeInstances,
                CardinalityMeasure::DistinctNeighbours,
            ] {
                *self.counts.entry((node, direction, measure)).or_default() +=
                    transition.delta(measure);
            }
        }
    }

    fn apply(&mut self, op: &Op) {
        match *op {
            Op::Upsert(s, t, k) => {
                let tr = self.pairs.entry((s, t)).or_default().upsert(vec![k]);
                self.apply_delta(s, t, tr);
            }
            Op::Remove(s, t, k) => {
                let tr = self.pairs.entry((s, t)).or_default().remove(&[k]);
                self.apply_delta(s, t, tr);
            }
            Op::Clear(s, t) => {
                let tr = self.pairs.entry((s, t)).or_default().clear();
                self.apply_delta(s, t, tr);
            }
            Op::Redirect(s, t, k, s2, t2) => {
                if (s, t) == (s2, t2) {
                    return;
                }
                let mut from = self.pairs.remove(&(s, t)).unwrap_or_default();
                let mut to = self.pairs.remove(&(s2, t2)).unwrap_or_default();
                if let Ok((left, joined)) = from.move_to(&mut to, &[k]) {
                    self.apply_delta(s, t, left);
                    self.apply_delta(s2, t2, joined);
                }
                self.pairs.insert((s, t), from);
                self.pairs.insert((s2, t2), to);
            }
        }
    }
}

/// The model: the edges as a plain set, counted by scanning it.
#[derive(Default)]
struct Model {
    edges: BTreeSet<(u64, u64, u8)>,
}

impl Model {
    fn apply(&mut self, op: &Op) {
        match *op {
            Op::Upsert(s, t, k) => {
                self.edges.insert((s, t, k));
            }
            Op::Remove(s, t, k) => {
                self.edges.remove(&(s, t, k));
            }
            Op::Clear(s, t) => self.edges.retain(|&(a, b, _)| (a, b) != (s, t)),
            Op::Redirect(s, t, k, s2, t2) => {
                let collides = self.edges.contains(&(s2, t2, k));
                if (s, t) != (s2, t2) && !collides && self.edges.remove(&(s, t, k)) {
                    self.edges.insert((s2, t2, k));
                }
            }
        }
    }

    fn scope(&self, node: u64, direction: Direction) -> Vec<IncidentInstance> {
        self.edges
            .iter()
            .filter_map(|&(s, t, k)| match direction {
                Direction::Outgoing if s == node => Some(at(t, &[k])),
                Direction::Incoming if t == node => Some(at(s, &[k])),
                _ => None,
            })
            .collect()
    }
}

proptest! {
    /// Counts maintained only from transition deltas equal the counts of the
    /// model's set after every effect, in every scope, for both measures;
    /// and the pairs' adjacency is exactly the model's.
    #[test]
    fn maintained_counts_match_the_set_model(ops in proptest::collection::vec(op(), 0..64)) {
        let mut maintained = Maintained::default();
        let mut model = Model::default();
        for op in &ops {
            maintained.apply(op);
            model.apply(op);
            for node in 1..4 {
                for direction in [Direction::Outgoing, Direction::Incoming] {
                    let expected = IncidentCount::of_set(model.scope(node, direction));
                    for measure in [
                        CardinalityMeasure::EdgeInstances,
                        CardinalityMeasure::DistinctNeighbours,
                    ] {
                        let kept = maintained
                            .counts
                            .get(&(node, direction, measure))
                            .copied()
                            .unwrap_or(0);
                        prop_assert!(kept >= 0, "a count went negative: {op:?}");
                        prop_assert_eq!(kept as u64, expected.of(measure), "{:?} {:?} {:?}", node, direction, measure);
                    }
                }
            }
            for (&(s, t), pair) in &maintained.pairs {
                let adjacent = model.edges.iter().any(|&(a, b, _)| (a, b) == (s, t));
                prop_assert_eq!(pair.is_adjacent(), adjacent, "pair {:?}", (s, t));
            }
        }
    }
}
