//! Integration tests: oplog purge gate on cross-partition flush watermark.
//!
//! These tests exercise the second gate added to [`OplogManager::purge_before`]:
//! a segment is eligible for deletion only when **every** LSM partition has
//! flushed an SST that covers the segment's last `commit_ts`. The gate closes
//! the crash-safety hole that lets `applied_index` advance into oplog entries
//! whose mutations still sit in a partition memtable; without it, a kernel
//! crash that drops memtables leaves the state machine with no oplog to
//! replay from.
//!
//! Scenario reproduced: a write batch lands in `Partition::Node` plus an
//! `applied_index` update in `Partition::Schema`. The Schema partition flushes
//! (forced here, naturally common because every applied entry rewrites the
//! same key), while Node stays in memtable. Without R076a, purge would delete
//! the oplog segment and a subsequent crash would lose the Node data
//! permanently. With R076a, purge defers until Node also catches up.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use coordinode_storage::engine::config::StorageConfig;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::oplog::entry::{OplogEntry, OplogOp};
use coordinode_storage::oplog::manager::OplogManager;

// Oplog ops carry partition as an opaque u8 in their serialized form (it is
// only interpreted by the apply path, never by the manager). The purge gate
// looks at footer ts/index only, so the value is arbitrary for these tests.
const NODE_PARTITION_TAG: u8 = 1;

fn open_engine(dir: &std::path::Path) -> Arc<StorageEngine> {
    let config = StorageConfig::new(dir);
    Arc::new(StorageEngine::open(&config).expect("open engine"))
}

fn make_entry(index: u64, ts: u64, key: &str, value: &str) -> OplogEntry {
    OplogEntry {
        ts,
        term: 1,
        index,
        shard: 0,
        ops: vec![OplogOp::Insert {
            partition: NODE_PARTITION_TAG,
            key: key.as_bytes().to_vec(),
            value: value.as_bytes().to_vec(),
        }],
        is_migration: false,
        pre_images: None,
    }
}

#[test]
fn purge_gate_blocks_when_some_partition_lags_in_flush() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open_engine(dir.path());
    let mut oplog =
        OplogManager::open(dir.path(), 0, 64 * 1024 * 1024, 1024, 86400).expect("open oplog");

    // Five entries with monotonic commit_ts. Each carries a Node write that we
    // intentionally do NOT flush — the mutation sits in the Node partition's
    // memtable, which under the pre-R076a behaviour is fine until purge runs.
    for i in 0..5u64 {
        engine
            .put(
                Partition::Node,
                format!("k{i}").as_bytes(),
                format!("v{i}").as_bytes(),
            )
            .expect("put");
        oplog
            .append(&make_entry(i, 1000 + i, &format!("k{i}"), &format!("v{i}")))
            .expect("append oplog");
    }
    oplog.rotate().expect("seal segment");

    // Force-flush a partition that the Node writes did NOT touch — this leaves
    // Node's memtable un-flushed (no SST in Node partition yet) while another
    // partition has SSTs. `min_partition_flushed_seqno` is `min` across all
    // partitions, so it stays at 0 even though one partition has flushed.
    engine
        .put(Partition::Schema, b"raft:sm:applied", b"opaque")
        .expect("put schema");
    engine
        .force_compaction(Partition::Schema)
        .expect("flush schema");

    let safe_ts = engine.min_partition_flushed_seqno();
    assert_eq!(
        safe_ts, 0,
        "min_partition_flushed_seqno must reflect the slowest partition; \
         Node has nothing in SST, so safe_ts == 0"
    );

    // openraft has applied everything (applied_index past the segment), so the
    // first gate is fully satisfied — but the second gate must hold the segment.
    let purged = oplog
        .purge_before(/*applied_index*/ u64::MAX, safe_ts)
        .expect("purge_before");
    assert_eq!(
        purged, 0,
        "purge must be deferred while any partition's memtable trails the segment"
    );
}

#[test]
fn purge_proceeds_once_every_partition_flushes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open_engine(dir.path());
    let mut oplog =
        OplogManager::open(dir.path(), 0, 64 * 1024 * 1024, 1024, 86400).expect("open oplog");

    for i in 0..5u64 {
        engine
            .put(
                Partition::Node,
                format!("k{i}").as_bytes(),
                format!("v{i}").as_bytes(),
            )
            .expect("put");
        oplog
            .append(&make_entry(i, 1000 + i, &format!("k{i}"), &format!("v{i}")))
            .expect("append oplog");
    }
    oplog.rotate().expect("seal segment");

    // Full flush across every partition — this is exactly what would happen
    // either on graceful shutdown (Drop hooks call persist) or after the
    // flush manager has caught up under sustained load. After persist the
    // SST seqno watermark catches up to the last commit_ts.
    engine.persist().expect("persist all partitions");

    let safe_ts = engine.min_partition_flushed_seqno();
    assert!(
        safe_ts >= 1004,
        "after persist all partitions must cover the last commit_ts (1004), got safe_ts={safe_ts}"
    );

    let purged = oplog.purge_before(u64::MAX, safe_ts).expect("purge_before");
    assert_eq!(
        purged, 1,
        "with every partition durable, the segment is safe to drop"
    );
}
