use std::sync::Arc;

use coordinode_core::txn::proposal::{Mutation, PartitionId};
use coordinode_core::txn::timestamp::TimestampOracle;
use tempfile::TempDir;

use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};

fn open(dir: &TempDir) -> (StorageEngine, Arc<TimestampOracle>) {
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let oracle = Arc::new(TimestampOracle::new());
    let engine = StorageEngine::open_with_oracle(&config, Arc::clone(&oracle)).expect("open");
    (engine, oracle)
}

/// A unit of the store's namespace writing the node and adjacency trees.
fn node_and_adj() -> Vec<Mutation> {
    crate::engine::scope::store_unit([
        Mutation::Put {
            partition: PartitionId::Node,
            key: b"node:00:0001".to_vec(),
            value: b"n".to_vec(),
        },
        Mutation::Merge {
            partition: PartitionId::Adj,
            key: b"adj:R:out:1".to_vec(),
            operand: crate::engine::merge::encode_add(2),
        },
    ])
}

#[test]
fn a_store_without_raft_applies_has_no_record() {
    let dir = TempDir::new().expect("dir");
    let (engine, _) = open(&dir);
    let coverage = engine.raft_coverage().expect("read");
    assert!(!coverage.has_record());
    assert_eq!(coverage.resume_point(), None);
    assert_eq!(coverage.skip_until(), 0);
}

#[test]
fn an_apply_marks_exactly_the_partitions_it_touched() {
    // The marker rides in each touched partition's batch and nowhere else,
    // so recovery can tell, tree by tree, whether the proposal landed.
    let dir = TempDir::new().expect("dir");
    let (engine, oracle) = open(&dir);
    engine.reset_raft_coverage(0, &[]).expect("establish");
    let applied = engine
        .apply_raft_proposal(&node_and_adj(), oracle.next().as_raw(), 7, 1, |_| false)
        .expect("apply");
    assert_eq!(applied, 2);

    let coverage = engine.raft_coverage().expect("read");
    let tree = |part| engine.tree_id(part);
    assert!(coverage.holds(tree(Partition::Node), 7, 1));
    assert!(coverage.holds(tree(Partition::Adj), 7, 1));
    assert!(
        !coverage.holds(tree(Partition::Counter), 7, 1),
        "untouched tree"
    );
    assert!(
        !coverage.holds(tree(Partition::Node), 7, 0),
        "another proposal of the entry"
    );
    assert_eq!(coverage.skip_until(), 8);
    assert_eq!(coverage.resume_point(), Some((0, &[][..])));
}

#[test]
fn a_skipped_partition_gets_neither_the_effect_nor_the_marker() {
    // Replaying an entry one tree already holds must leave that tree alone:
    // a second merge operand would double the edge.
    let dir = TempDir::new().expect("dir");
    let (engine, oracle) = open(&dir);
    engine.reset_raft_coverage(0, &[]).expect("establish");
    let ts = oracle.next().as_raw();
    engine
        .apply_raft_proposal(&node_and_adj(), ts, 3, 0, |_| false)
        .expect("first apply");
    let before = engine.get(Partition::Adj, b"adj:R:out:1").expect("get");

    let adj = engine.tree_id(Partition::Adj);
    let applied = engine
        .apply_raft_proposal(&node_and_adj(), ts, 3, 0, |tree| tree == adj)
        .expect("replay");
    assert_eq!(applied, 1, "only the Node put is re-applied");
    assert_eq!(
        engine.get(Partition::Adj, b"adj:R:out:1").expect("get"),
        before,
        "the skipped merge must not be applied twice"
    );
}

/// A replay skips a tree, not a partition: an entry writing the same
/// partition of two namespaces, replayed where one namespace's tree already
/// holds it, lands in the other namespace's tree and records it there only.
#[test]
fn a_skip_holds_one_namespaces_tree_only() {
    let dir = TempDir::new().expect("dir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let oracle = Arc::new(TimestampOracle::new());
    let deployment = Deployment::open_with_oracle(&config, Arc::clone(&oracle)).expect("open");
    let a = deployment
        .engine(deployment.create_namespace("a").expect("a"))
        .expect("bind a");
    let b = deployment
        .engine(deployment.create_namespace("b").expect("b"))
        .expect("bind b");
    deployment.reset_raft_coverage(0, &[]).expect("establish");
    let put = |value: &[u8]| Mutation::Put {
        partition: PartitionId::Node,
        key: b"k".to_vec(),
        value: value.to_vec(),
    };
    let mut unit = coordinode_core::txn::proposal::in_namespace(a.namespace(), [put(b"of-a")]);
    unit.extend(coordinode_core::txn::proposal::in_namespace(
        b.namespace(),
        [put(b"of-b")],
    ));
    let a_node = a.tree_id(Partition::Node);
    let applied = deployment
        .apply_raft_proposal(&unit, oracle.next().as_raw(), 5, 0, |tree| tree == a_node)
        .expect("apply");

    assert_eq!(applied, 1, "only b's write lands");
    assert_eq!(a.get(Partition::Node, b"k").expect("read a"), None);
    assert_eq!(
        b.get(Partition::Node, b"k").expect("read b").as_deref(),
        Some(&b"of-b"[..])
    );
    let coverage = deployment.raft_coverage().expect("read");
    assert!(coverage.holds(b.tree_id(Partition::Node), 5, 0));
    assert!(
        !coverage.holds(a_node, 5, 0),
        "the skipped tree is not marked"
    );
}

/// The applies' floor is a tree's: raising one namespace's tree leaves the
/// same partition of another namespace behind it.
#[test]
fn a_floor_is_one_trees() {
    let a = TreeId::namespace(NamespaceId::from_raw(1), Partition::Node);
    let b = TreeId::namespace(NamespaceId::from_raw(2), Partition::Node);
    let mut applies = RaftApplyState::new(3);
    applies.raise_floor(a, 10);
    assert!(applies.skips(a, 9));
    assert!(!applies.skips(a, 10));
    assert!(
        !applies.skips(b, 9),
        "another namespace's tree skips nothing"
    );
    applies.raise_floor(a, 7);
    assert_eq!(applies.floor(a), 10, "a floor never goes back");
    applies.reset(12);
    assert_eq!(applies.floor(a), 0);
}

#[test]
fn a_fold_replaces_the_markers_below_it_with_a_base() {
    let dir = TempDir::new().expect("dir");
    let (engine, oracle) = open(&dir);
    engine.reset_raft_coverage(0, &[]).expect("establish");
    for index in 0..3 {
        engine
            .apply_raft_proposal(&node_and_adj(), oracle.next().as_raw(), index, 0, |_| false)
            .expect("apply");
    }
    engine.fold_raft_coverage(0, 2, b"id-of-1", |_| false);

    let coverage = engine.raft_coverage().expect("read");
    assert_eq!(coverage.resume_point(), Some((2, b"id-of-1".as_slice())));
    let tree = |part| engine.tree_id(part);
    assert!(
        coverage.holds(tree(Partition::Node), 0, 0),
        "below the base"
    );
    assert!(
        coverage.holds(tree(Partition::Node), 2, 0),
        "its marker survives"
    );
    assert!(
        coverage.holds(tree(Partition::Counter), 1, 0),
        "a base covers every tree, touched or not"
    );
    assert_eq!(coverage.skip_until(), 3);
}

#[test]
fn a_reset_does_not_hide_markers_written_after_it_below_its_seqno() {
    // A follower's clock can run ahead of its leader's: the reset after a
    // snapshot install then carries a seqno above the commit_ts of the next
    // entries it applies. A range tombstone suppresses every covered key
    // with a lower seqno, so a reset whose tombstone reached past the
    // snapshot would hide those entries' markers, and a crash would re-apply
    // their merges.
    let dir = TempDir::new().expect("dir");
    let (engine, oracle) = open(&dir);
    // The follower already holds flushed tables from before the snapshot.
    engine.reset_raft_coverage(0, &[]).expect("establish");
    engine
        .apply_raft_proposal(&node_and_adj(), oracle.next().as_raw(), 3, 0, |_| false)
        .expect("pre-snapshot apply");
    engine.persist().expect("persist");
    // The leader's next commit_ts, from before the reset drew its seqno.
    let behind = oracle.next().as_raw();
    engine.reset_raft_coverage(10, b"id-of-9").expect("reset");
    engine
        .apply_raft_proposal(&node_and_adj(), behind, 10, 0, |_| false)
        .expect("apply the next entry");

    let check = |stage: &str| {
        let coverage = engine.raft_coverage().expect("read");
        assert!(
            coverage.holds(engine.tree_id(Partition::Node), 10, 0),
            "{stage}: the entry after the snapshot is recorded"
        );
        assert!(
            coverage.holds(engine.tree_id(Partition::Adj), 10, 0),
            "{stage}"
        );
    };
    check("in the memtable");
    // Compaction brings the marker's table and the tombstone's together,
    // which is where a tombstone reaching past the snapshot would bite.
    engine.persist().expect("persist");
    engine
        .force_compaction(Partition::Node)
        .expect("compact node");
    engine
        .force_compaction(Partition::Adj)
        .expect("compact adj");
    check("after compaction");
}

#[test]
fn a_reset_is_durable_and_leaves_no_marker() {
    // What a snapshot install leaves behind: every tree holds the entries
    // below the snapshot, recorded by the base alone. A snapshot is only
    // installed past what the member applied, so every marker it had is
    // below the new base and goes.
    let dir = TempDir::new().expect("dir");
    {
        let (engine, oracle) = open(&dir);
        engine.reset_raft_coverage(0, &[]).expect("establish");
        engine
            .apply_raft_proposal(&node_and_adj(), oracle.next().as_raw(), 4, 0, |_| false)
            .expect("apply");
        engine.reset_raft_coverage(10, b"id-of-9").expect("reset");
        assert_eq!(engine.raft_durable_floor().expect("floor"), 10);
    }
    let (engine, _) = open(&dir);
    let coverage = engine.raft_coverage().expect("read");
    assert_eq!(coverage.resume_point(), Some((10, b"id-of-9".as_slice())));
    assert!(
        coverage.holds(engine.tree_id(Partition::Counter), 9, 0),
        "the base covers it"
    );
    assert!(
        !coverage.holds(engine.tree_id(Partition::Node), 10, 0),
        "nothing past the snapshot is recorded"
    );
    assert_eq!(coverage.skip_until(), 10, "the old marker is gone");
}
