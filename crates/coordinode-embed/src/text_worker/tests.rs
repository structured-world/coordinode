use coordinode_core::graph::node::{NodeId, encode_node_key};
use coordinode_core::txn::proposal::{Mutation, PartitionId};
use coordinode_query::index::{IndexCoverage, IndexDelta};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

fn engine(dir: &tempfile::TempDir) -> StorageEngine {
    StorageEngine::open(&StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]))
    .expect("open")
}

/// Apply a write of node `id` on `shard` as entry `index`.
fn apply(engine: &StorageEngine, index: u64, shard: u16, id: u64) {
    engine
        .apply_raft_proposal(
            &[Mutation::Put {
                partition: PartitionId::Node,
                key: encode_node_key(shard, NodeId::from_raw(id)),
                value: b"v".to_vec(),
            }],
            10 + index,
            index,
            0,
            |_| false,
        )
        .expect("apply");
}

fn nodes(ids: &[u64]) -> IndexDelta {
    IndexDelta::Nodes(ids.iter().map(|id| NodeId::from_raw(*id)).collect())
}

/// A search learns the nodes of its shard written by entries the worker has
/// not released, whether the worker has taken them or not, and stops seeing
/// them once they are released.
#[test]
fn the_delta_names_the_nodes_of_unreleased_entries() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = engine(&dir);
    let sub = engine.subscribe_applied_retained(Partition::Node, 16);
    let coverage = IndexCoverage::new(sub.position());
    assert_eq!(coverage.delta(1), nodes(&[]));

    apply(&engine, 1, 1, 7);
    apply(&engine, 2, 2, 8);
    apply(&engine, 3, 1, 9);
    assert_eq!(
        coverage.delta(1),
        nodes(&[7, 9]),
        "shard 2's node is not ours"
    );

    // Taken but not folded: still the search's to answer.
    let _ = sub.try_next();
    assert_eq!(coverage.delta(1), nodes(&[7, 9]));
    coverage.release(1);
    assert_eq!(coverage.delta(1), nodes(&[9]));
    let _ = sub.try_next();
    let _ = sub.try_next();
    coverage.release(3);
    assert!(coverage.delta(1).is_empty());
}

/// A write dropped from a full queue leaves the delta unknown, so a search
/// answers every node from the store, until the rebuild that follows is
/// released.
#[test]
fn a_dropped_write_makes_the_delta_unknown_until_the_rebuild_is_released() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = engine(&dir);
    let sub = engine.subscribe_applied_retained(Partition::Node, 1);
    let coverage = IndexCoverage::new(sub.position());

    apply(&engine, 1, 1, 7);
    apply(&engine, 2, 1, 8);
    assert_eq!(coverage.delta(1), IndexDelta::Unknown);
    assert!(coverage.delta(1).contains(NodeId::from_raw(12345)));

    let covered = sub.position().delivered();
    let _ = sub.try_next();
    let _ = sub.try_next();
    coverage.release(covered);
    assert!(coverage.delta(1).is_empty());
}
