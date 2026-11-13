//! Where the material accounting lives: in its own tree, so a data tree's
//! tables span the keys written there and nothing else, and a replayed
//! unit's accounting is derived from all of it.

use coordinode_core::txn::proposal::{Mutation, PartitionId};
use lsm_tree::AbstractTree;
use tempfile::TempDir;

use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, Tier};
use crate::engine::material::{self, unit_bytes};

fn engine(dir: &TempDir) -> StorageEngine {
    StorageEngine::open(&StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]))
    .expect("open")
}

/// The unit records `part`'s scope holds.
fn units(engine: &StorageEngine, part: Partition) -> Vec<material::Unit> {
    let heads = *engine.material_heads();
    let scope = engine.material_scope(&heads, part).expect("scope");
    material::units_from(
        &scope,
        (0, 0),
        lsm_tree::SeqNo::MAX,
        lsm_tree::SeqNo::MAX,
        usize::MAX,
        None,
    )
    .expect("units")
    .0
}

/// Two batches over disjoint key ranges, each flushed into its own table:
/// each table spans its own keys only, so the two never overlap and a
/// compaction can move either down without merging. Accounting records in
/// the data tree would start every table in the reserved namespace, below
/// every user key, and make all of them overlap. The structural control of
/// the regression where compaction merged every flushed table.
#[test]
fn a_data_table_spans_the_keys_written_there_only() {
    let dir = TempDir::new().expect("tempdir");
    let engine = engine(&dir);
    for i in 0..50u32 {
        engine
            .put(Partition::Node, format!("node:a{i:03}").as_bytes(), b"v")
            .expect("put");
    }
    engine.persist().expect("persist");
    for i in 0..50u32 {
        engine
            .put(Partition::Node, format!("node:z{i:03}").as_bytes(), b"v")
            .expect("put");
    }
    engine.persist().expect("persist");

    let tree = engine.tree(Partition::Node).expect("tree");
    let version = tree.current_version();
    let ranges: Vec<_> = version
        .iter_tables()
        .map(|table| table.metadata.key_range.clone())
        .collect();
    assert_eq!(ranges.len(), 2, "one table per flush");
    for range in &ranges {
        assert!(
            !crate::engine::coverage::is_reserved(range.min()),
            "a data table starts in the reserved namespace: {:02x?}",
            &range.min()[..range.min().len().min(8)]
        );
    }
    let refs: Vec<_> = ranges.iter().collect();
    assert!(
        lsm_tree::KeyRange::is_disjoint(&refs),
        "the tables of disjoint writes overlap"
    );
    // The accounting of all of it is in the accounting tree.
    assert_eq!(units(&engine, Partition::Node).len(), 100);
}

/// A replayed Raft proposal whose data one tree already holds still takes
/// its whole accounting: every unit of the unit, the held tree's included,
/// equal operands on one key each their own unit. A proposal the accounting
/// tree already holds takes none again.
#[test]
fn a_raft_proposal_held_in_part_is_accounted_whole() {
    let dir = TempDir::new().expect("tempdir");
    let engine = engine(&dir);
    let operand = crate::engine::merge::encode_add(7);
    let unit = vec![
        Mutation::Put {
            partition: PartitionId::Node,
            key: b"node:held".to_vec(),
            value: b"value".to_vec(),
        },
        Mutation::Merge {
            partition: PartitionId::Adj,
            key: b"adj:V:out:1".to_vec(),
            operand: operand.clone(),
        },
        Mutation::Merge {
            partition: PartitionId::Adj,
            key: b"adj:V:out:1".to_vec(),
            operand: operand.clone(),
        },
    ];
    // The node tree holds the proposal already: only the adjacency data is
    // written, and the accounting of all three units.
    engine
        .apply_raft_proposal(&unit, 0, 5, 0, |p| p == Partition::Node)
        .expect("apply");
    assert!(
        engine
            .get(Partition::Node, b"node:held")
            .expect("get")
            .is_none(),
        "a held tree's data was applied again"
    );
    let node = units(&engine, Partition::Node);
    let adj = units(&engine, Partition::Adj);
    assert_eq!(node.len(), 1, "the held tree's unit was not accounted");
    assert_eq!(adj.len(), 2, "equal operands collapsed into one unit");
    assert_eq!(
        node.iter().chain(&adj).map(|u| u.bytes).sum::<u64>(),
        unit_bytes(b"node:held", b"value") + 2 * unit_bytes(b"adj:V:out:1", &operand)
    );

    // Its accounting already held, a proposal takes none again: no unit,
    // no record, whatever data it applies.
    let before = engine.material_between(0, None).expect("material");
    engine
        .apply_raft_proposal(&unit, 0, 6, 0, |p| p == Partition::Accounting)
        .expect("apply");
    assert_eq!(engine.material_between(0, None).expect("material"), before);
    assert_eq!(units(&engine, Partition::Node).len(), 1);
    assert!(
        engine
            .get(Partition::Node, b"node:held")
            .expect("get")
            .is_some()
    );
}

/// The accounting tree is written by the engine alone, never replicated,
/// transferred or carried in a snapshot.
#[test]
fn the_accounting_is_this_members_own() {
    assert!(Partition::Accounting.proposal_id().is_none());
    assert!(raft_coverage::is_node_local(
        Partition::Accounting,
        b"anything"
    ));
    assert!(Partition::Accounting.user_data_prefix().is_none());
    let dir = TempDir::new().expect("tempdir");
    let engine = engine(&dir);
    engine.put(Partition::Node, b"node:x", b"v").expect("put");
    assert!(
        engine
            .replicated_rows(Partition::Accounting)
            .expect("rows")
            .is_empty(),
        "the accounting would travel to another member"
    );
    assert!(
        engine.copy_partition(Partition::Accounting).is_err(),
        "the accounting was copied for another member"
    );
    assert!(
        engine
            .begin_partition_rebuild(Partition::Accounting, &Vec::new())
            .is_err(),
        "the accounting was rebuilt from another copy"
    );
}
