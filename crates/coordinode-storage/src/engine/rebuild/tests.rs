use std::path::Path;

use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use crate::engine::partition::Partition;

fn open(config: &StorageConfig, dir: &str, next: u64) -> AnyTree {
    config
        .tree_config_at(
            Partition::Schema,
            Path::new(dir),
            lsm_tree::SharedSequenceNumberGenerator::from(lsm_tree::SequenceNumberCounter::new(
                next,
            )),
            None,
        )
        .open()
        .expect("open")
}

/// The answer of `tree` for `key` at `snapshot`: the value, absence, or
/// whether the snapshot is refused.
fn answer(tree: &AnyTree, key: &str, snapshot: u64) -> Result<Option<Vec<u8>>, ()> {
    tree.get(key, snapshot)
        .map(|value| value.map(|v| v.to_vec()))
        .map_err(|_| ())
}

/// A tree written anew from a source whose compaction already collected
/// history below its floor serves exactly what the source served: the same
/// floor, every snapshot below it refused, every one above it answered with
/// the source's value, and after a reopen too.
#[test]
fn a_rewritten_tree_serves_exactly_the_history_its_source_did() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let source = open(&config, "src", 100);
    source.insert("k", "v1", 10);
    source.insert("k", "v2", 20);
    source.insert("j", "w", 30);
    source.remove("j", 40);
    let lock = source.get_flush_lock();
    source.rotate_memtable();
    source.flush(&lock, 0).expect("flush");
    drop(lock);
    source.major_compact(u64::MAX, 25).expect("compact");
    let floor = source.retention_floor();
    assert!(floor > 0, "the source collected history below its floor");
    let snapshots = [0, 5, 11, floor, floor + 1, 21, 31, 41, 100];
    let served: Vec<_> = snapshots
        .iter()
        .flat_map(|&at| ["k", "j"].map(|key| answer(&source, key, at)))
        .collect();
    drop(source);

    rewrite_tree_values(
        &config,
        Partition::Schema,
        (Path::new("src"), Path::new("dst")),
        None,
        &mut |version| Ok(version.value.to_vec()),
    )
    .expect("rewrite");

    let target = open(&config, "dst", 100);
    assert_eq!(target.retention_floor(), floor);
    let rewritten: Vec<_> = snapshots
        .iter()
        .flat_map(|&at| ["k", "j"].map(|key| answer(&target, key, at)))
        .collect();
    assert_eq!(rewritten, served);
    assert!(answer(&target, "k", floor).is_err(), "below the floor");
}
