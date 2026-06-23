use super::*;
use std::sync::Arc;

use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::transaction::CommitContext;

/// MVCC test database: writes buffer on the transaction and apply at
/// [`commit`]; reads open a fresh transaction at the latest snapshot.
/// A standalone oracle over the shared logic engine is sufficient —
/// reads pin `engine.snapshot()` (latest committed), not `read_ts`.
struct TestDb {
    _fx: coordinode_test_fixtures::EngineFixture,
    engine: Arc<StorageEngine>,
    oracle: Arc<TimestampOracle>,
}

fn open() -> TestDb {
    let fx = coordinode_test_fixtures::engine_for_logic();
    let engine = Arc::clone(&fx.engine);
    TestDb {
        _fx: fx,
        engine,
        oracle: Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(1))),
    }
}

fn mvcc_txn<'a>(engine: &'a StorageEngine, oracle: &'a TimestampOracle) -> Transaction<'a> {
    let read_ts = oracle.next();
    let snap = engine.snapshot();
    Transaction::new(engine, Some(oracle), read_ts, Some(snap))
}

fn commit(t: &mut Transaction) {
    let wc = WriteConcern::majority();
    let ctx = CommitContext {
        write_concern: &wc,
        pipeline: None,
        id_gen: None,
        drain_buffer: None,
        nvme_write_buffer: None,
    };
    t.commit(&ctx).expect("commit");
}

impl TestDb {
    fn write(&self, f: impl FnOnce(&LocalNodeStore, &mut Transaction)) {
        let mut t = mvcc_txn(&self.engine, &self.oracle);
        let store = LocalNodeStore;
        f(&store, &mut t);
        commit(&mut t);
    }

    fn read(&self) -> Transaction<'_> {
        mvcc_txn(&self.engine, &self.oracle)
    }
}

fn rec(label: &str) -> NodeRecord {
    NodeRecord::new(label)
}

#[test]
fn non_temporal_round_trip() {
    let db = open();
    let id = NodeId::from_raw(7);
    let store = LocalNodeStore;
    assert!(store.get(&db.read(), 0, id).expect("none").is_none());
    db.write(|s, t| s.put(t, 0, id, &rec("User")).expect("put"));
    let got = store.get(&db.read(), 0, id).expect("some").expect("Some");
    assert_eq!(got.primary_label(), "User");
}

#[test]
fn put_overwrites_existing_record() {
    let db = open();
    let id = NodeId::from_raw(1);
    db.write(|s, t| s.put(t, 0, id, &rec("A")).expect("put A"));
    db.write(|s, t| s.put(t, 0, id, &rec("B")).expect("put B"));
    let got = LocalNodeStore
        .get(&db.read(), 0, id)
        .expect("ok")
        .expect("Some");
    assert_eq!(got.primary_label(), "B");
}

#[test]
fn delete_tombstones_record() {
    let db = open();
    let id = NodeId::from_raw(1);
    db.write(|s, t| s.put(t, 0, id, &rec("X")).expect("put"));
    db.write(|s, t| s.delete(t, 0, id).expect("delete"));
    assert!(LocalNodeStore.get(&db.read(), 0, id).expect("ok").is_none());
}

#[test]
fn delete_missing_is_idempotent() {
    let db = open();
    db.write(|s, t| {
        s.delete(t, 0, NodeId::from_raw(999))
            .expect("delete missing")
    });
}

#[test]
fn shards_isolated_by_key_prefix() {
    // Same node_id under different shards must NOT collide.
    let db = open();
    let id = NodeId::from_raw(42);
    db.write(|s, t| {
        s.put(t, 0, id, &rec("ShardZero")).expect("put");
        s.put(t, 1, id, &rec("ShardOne")).expect("put");
    });
    let store = LocalNodeStore;
    let r = db.read();
    assert_eq!(
        store
            .get(&r, 0, id)
            .expect("ok")
            .expect("Some")
            .primary_label(),
        "ShardZero"
    );
    assert_eq!(
        store
            .get(&r, 1, id)
            .expect("ok")
            .expect("Some")
            .primary_label(),
        "ShardOne"
    );
}

#[test]
fn temporal_versions_round_trip() {
    let db = open();
    let id = NodeId::from_raw(11);
    db.write(|s, t| {
        s.put_temporal(t, 0, id, 100, &rec("V1")).expect("put v1");
        s.put_temporal(t, 0, id, 200, &rec("V2")).expect("put v2");
        s.put_temporal(t, 0, id, 300, &rec("V3")).expect("put v3");
    });
    let versions = LocalNodeStore
        .scan_versions(&db.read(), 0, id)
        .expect("scan");
    let labels: Vec<&str> = versions.iter().map(|(_, r)| r.primary_label()).collect();
    let times: Vec<i64> = versions.iter().map(|(t, _)| *t).collect();
    assert_eq!(times, vec![100, 200, 300]);
    assert_eq!(labels, vec!["V1", "V2", "V3"]);
}

#[test]
fn get_at_returns_largest_valid_from_le_query() {
    // Versions at t=100, 200, 300. Queries:
    // - at 99 → None (no version exists yet)
    // - at 100 → V1 (exact match)
    // - at 199 → V1 (largest <= 199)
    // - at 200 → V2 (exact match)
    // - at 1_000_000 → V3 (far future picks newest)
    let db = open();
    let id = NodeId::from_raw(11);
    db.write(|s, t| {
        s.put_temporal(t, 0, id, 100, &rec("V1")).expect("v1");
        s.put_temporal(t, 0, id, 200, &rec("V2")).expect("v2");
        s.put_temporal(t, 0, id, 300, &rec("V3")).expect("v3");
    });
    let store = LocalNodeStore;
    let r = db.read();

    assert!(store.get_at(&r, 0, id, 99).expect("ok").is_none());
    assert_eq!(
        store
            .get_at(&r, 0, id, 100)
            .expect("ok")
            .expect("Some")
            .primary_label(),
        "V1",
    );
    assert_eq!(
        store
            .get_at(&r, 0, id, 199)
            .expect("ok")
            .expect("Some")
            .primary_label(),
        "V1",
    );
    assert_eq!(
        store
            .get_at(&r, 0, id, 200)
            .expect("ok")
            .expect("Some")
            .primary_label(),
        "V2",
    );
    assert_eq!(
        store
            .get_at(&r, 0, id, 1_000_000)
            .expect("ok")
            .expect("Some")
            .primary_label(),
        "V3",
    );
}

#[test]
fn get_at_isolated_per_node_id() {
    // Two distinct nodes in the same shard each have temporal
    // versions — get_at on one must not bleed into the other.
    let db = open();
    let a = NodeId::from_raw(1);
    let b = NodeId::from_raw(2);
    db.write(|s, t| {
        s.put_temporal(t, 0, a, 50, &rec("A50")).expect("a");
        s.put_temporal(t, 0, b, 70, &rec("B70")).expect("b");
    });
    let store = LocalNodeStore;
    let r = db.read();
    assert_eq!(
        store
            .get_at(&r, 0, a, 100)
            .expect("ok")
            .expect("Some")
            .primary_label(),
        "A50",
    );
    assert_eq!(
        store
            .get_at(&r, 0, b, 100)
            .expect("ok")
            .expect("Some")
            .primary_label(),
        "B70",
    );
}

#[test]
fn put_temporal_same_valid_from_overwrites_body() {
    // Two writes at the same (node_id, valid_from) land at the same
    // key — second wins, no version explosion.
    let db = open();
    let id = NodeId::from_raw(40);
    let mut rec_v1 = rec("A");
    rec_v1.set_extra("v", coordinode_core::graph::types::Value::Int(1));
    let mut rec_v2 = rec("A");
    rec_v2.set_extra("v", coordinode_core::graph::types::Value::Int(2));
    db.write(|s, t| {
        s.put_temporal(t, 0, id, 1000, &rec_v1).expect("v1");
        s.put_temporal(t, 0, id, 1000, &rec_v2).expect("v2");
    });
    let versions = LocalNodeStore
        .scan_versions(&db.read(), 0, id)
        .expect("scan");
    assert_eq!(versions.len(), 1, "same valid_from = one row");
    assert_eq!(versions[0].0, 1000);
    assert_eq!(
        versions[0].1.get_extra("v"),
        Some(&coordinode_core::graph::types::Value::Int(2)),
    );
}

#[test]
fn scan_versions_on_empty_node_returns_empty() {
    let db = open();
    let versions = LocalNodeStore
        .scan_versions(&db.read(), 0, NodeId::from_raw(999))
        .expect("ok");
    assert!(versions.is_empty());
}

#[test]
fn get_at_boundary_i64_min_max() {
    // Per ADR-027 valid_from_ms is i64. Test we can write at the
    // extreme boundaries and the sortable encoding still works.
    let db = open();
    let id = NodeId::from_raw(41);
    db.write(|s, t| {
        s.put_temporal(t, 0, id, i64::MIN, &rec("min"))
            .expect("min");
        s.put_temporal(t, 0, id, i64::MAX, &rec("max"))
            .expect("max");
    });
    let store = LocalNodeStore;
    let r = db.read();
    let versions = store.scan_versions(&r, 0, id).expect("scan");
    assert_eq!(versions.len(), 2);
    assert_eq!(versions[0].0, i64::MIN);
    assert_eq!(versions[1].0, i64::MAX);
    // Query at 0 picks the MIN-version (largest valid_from <= 0).
    let active = store.get_at(&r, 0, id, 0).expect("ok").expect("Some");
    assert_eq!(active.primary_label(), "min");
}

#[test]
fn get_at_seqno_returns_version_visible_at_snapshot() {
    let db = open();
    let id = NodeId::from_raw(50);
    db.write(|s, t| s.put(t, 0, id, &rec("v1")).expect("put v1"));
    let snap = db.engine.snapshot();
    db.write(|s, t| s.put(t, 0, id, &rec("v2")).expect("put v2"));
    let store = LocalNodeStore;
    let r = db.read();
    let at_snap = store
        .get_at_seqno(&r, 0, id, snap)
        .expect("ok")
        .expect("Some");
    assert_eq!(at_snap.primary_label(), "v1");
    let latest = store.get(&r, 0, id).expect("ok").expect("Some");
    assert_eq!(latest.primary_label(), "v2");
}

#[test]
fn get_at_seqno_missing_returns_none() {
    let db = open();
    let snap = db.engine.snapshot();
    assert!(LocalNodeStore
        .get_at_seqno(&db.read(), 0, NodeId::from_raw(999), snap)
        .expect("ok")
        .is_none());
}

#[test]
fn scan_shard_yields_every_non_temporal_record() {
    let db = open();
    db.write(|s, t| {
        for i in 0u64..5 {
            s.put(t, 0, NodeId::from_raw(i + 100), &rec(&format!("L{i}")))
                .expect("put");
        }
    });
    let all = LocalNodeStore.scan_shard(&db.read(), 0).expect("scan");
    assert_eq!(all.len(), 5);
    let mut ids: Vec<u64> = all.iter().map(|(id, _)| id.as_raw()).collect();
    ids.sort_unstable();
    assert_eq!(ids, vec![100, 101, 102, 103, 104]);
}

#[test]
fn scan_shard_isolated_per_shard() {
    let db = open();
    db.write(|s, t| {
        s.put(t, 0, NodeId::from_raw(1), &rec("s0")).unwrap();
        s.put(t, 1, NodeId::from_raw(2), &rec("s1")).unwrap();
    });
    let store = LocalNodeStore;
    let r = db.read();
    let shard0 = store.scan_shard(&r, 0).unwrap();
    let shard1 = store.scan_shard(&r, 1).unwrap();
    assert_eq!(shard0.len(), 1);
    assert_eq!(shard1.len(), 1);
    assert_eq!(shard0[0].0, NodeId::from_raw(1));
    assert_eq!(shard1[0].0, NodeId::from_raw(2));
}

#[test]
fn scan_shard_skips_temporal_versions() {
    // 25-byte temporal keys must not leak into the non-temporal
    // scan result.
    let db = open();
    db.write(|s, t| {
        s.put(t, 0, NodeId::from_raw(1), &rec("nt")).unwrap();
        s.put_temporal(t, 0, NodeId::from_raw(2), 1000, &rec("t"))
            .unwrap();
    });
    let all = LocalNodeStore.scan_shard(&db.read(), 0).unwrap();
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].0, NodeId::from_raw(1));
}

#[test]
fn for_each_in_shard_visits_every_record() {
    let db = open();
    db.write(|s, t| {
        for i in 0u64..4 {
            s.put(t, 0, NodeId::from_raw(i + 50), &rec(&format!("N{i}")))
                .expect("put");
        }
    });
    let store = LocalNodeStore;
    let r = db.read();
    let mut seen: Vec<u64> = Vec::new();
    store
        .for_each_in_shard(&r, 0, &mut |id, _rec| {
            seen.push(id.as_raw());
            Ok(())
        })
        .expect("walk");
    seen.sort_unstable();
    assert_eq!(seen, vec![50, 51, 52, 53]);
}

#[test]
fn corrupt_node_bytes_surface_as_decode_error() {
    let db = open();
    // Inject garbage directly at the node key.
    db.engine
        .put(
            Partition::Node,
            &encode_node_key(0, NodeId::from_raw(5)),
            &[0xff, 0xff, 0xff],
        )
        .expect("inject");
    let err = LocalNodeStore
        .get(&db.read(), 0, NodeId::from_raw(5))
        .expect_err("must err");
    assert!(matches!(
        err,
        StoreError::Decode {
            kind: "node record",
            ..
        }
    ));
}

#[test]
fn read_raw_at_snapshot_latest_round_trip() {
    let db = open();
    let id = NodeId::from_raw(42);
    let key = encode_node_key(3, id);
    // Absent → None.
    assert!(LocalNodeStore
        .read_raw_at_snapshot(&db.engine, None, &key)
        .expect("ok")
        .is_none());
    // After a committed write, the raw bytes decode back to the record.
    db.write(|s, t| s.put(t, 3, id, &rec("User")).expect("put"));
    let bytes = LocalNodeStore
        .read_raw_at_snapshot(&db.engine, None, &key)
        .expect("ok")
        .expect("present");
    assert_eq!(
        NodeRecord::from_msgpack(&bytes)
            .expect("decode")
            .primary_label(),
        "User"
    );
}

#[test]
fn for_each_in_shard_at_snapshot_visits_in_key_order_and_breaks_early() {
    let db = open();
    // Two shards; only shard 1's nodes should be visited.
    db.write(|s, t| {
        s.put(t, 1, NodeId::from_raw(3), &rec("User")).expect("put");
        s.put(t, 1, NodeId::from_raw(1), &rec("User")).expect("put");
        s.put(t, 1, NodeId::from_raw(2), &rec("User")).expect("put");
        s.put(t, 2, NodeId::from_raw(9), &rec("Other"))
            .expect("put");
    });

    // Full walk: key order (1, 2, 3), shard-isolated, key matches id.
    let mut seen = Vec::new();
    LocalNodeStore
        .for_each_in_shard_at_snapshot(&db.engine, None, 1, &mut |id, key, record| {
            assert_eq!(key, encode_node_key(1, id));
            assert_eq!(record.primary_label(), "User");
            seen.push(id.as_raw());
            Ok(core::ops::ControlFlow::Continue(()))
        })
        .expect("walk");
    assert_eq!(seen, vec![1, 2, 3]);

    // Early break stops after the first node.
    let mut count = 0usize;
    LocalNodeStore
        .for_each_in_shard_at_snapshot(&db.engine, None, 1, &mut |_, _, _| {
            count += 1;
            Ok(core::ops::ControlFlow::Break(()))
        })
        .expect("walk");
    assert_eq!(count, 1, "break stops after the first visit");
}

#[test]
fn for_each_in_shard_at_snapshot_skips_corrupt_rows() {
    let db = open();
    db.write(|s, t| s.put(t, 0, NodeId::from_raw(1), &rec("User")).expect("put"));
    // Plant a corrupt body in the same shard — must be skipped, not abort.
    db.engine
        .put(
            Partition::Node,
            &encode_node_key(0, NodeId::from_raw(2)),
            &[0xff, 0x00],
        )
        .expect("inject");
    let mut seen = Vec::new();
    LocalNodeStore
        .for_each_in_shard_at_snapshot(&db.engine, None, 0, &mut |id, _, _| {
            seen.push(id.as_raw());
            Ok(core::ops::ControlFlow::Continue(()))
        })
        .expect("walk tolerates corrupt row");
    assert_eq!(seen, vec![1], "corrupt row skipped, valid row visited");
}
