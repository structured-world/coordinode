use super::*;
use std::sync::Arc;

use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::transaction::CommitContext;

/// MVCC test database: writes buffer on the transaction and apply at
/// [`commit`]; reads open a fresh transaction at the latest snapshot.
/// The engine seqno space drives snapshot visibility, so a standalone
/// oracle (for `commit_ts` / `read_ts`) over the shared logic engine
/// is sufficient — no oracle-wired engine needed because every read
/// pins `engine.snapshot()` (latest committed), not `read_ts`.
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

/// Open an MVCC transaction pinned at a fresh read timestamp and the
/// latest committed snapshot.
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
    /// Run writes in a fresh MVCC transaction and commit them.
    fn write(&self, f: impl FnOnce(&LocalEdgeStore, &mut Transaction)) {
        let mut t = mvcc_txn(&self.engine, &self.oracle);
        let store = LocalEdgeStore;
        f(&store, &mut t);
        commit(&mut t);
    }

    /// Fresh read-only MVCC transaction at the latest committed state.
    fn read(&self) -> Transaction<'_> {
        mvcc_txn(&self.engine, &self.oracle)
    }
}

fn props_with(field: u32, value: i64) -> EdgeProperties {
    let mut p = EdgeProperties::new();
    p.set(field, coordinode_core::graph::types::Value::Int(value));
    p
}

#[test]
fn put_edge_creates_forward_and_reverse_adj() {
    let db = open();
    let alice = NodeId::from_raw(1);
    let bob = NodeId::from_raw(2);
    db.write(|s, t| s.put_edge(t, "KNOWS", alice, bob, None).expect("put"));

    let store = LocalEdgeStore;
    let r = db.read();
    assert_eq!(
        store.scan_neighbors_out(&r, "KNOWS", alice).expect("scan"),
        vec![bob],
    );
    assert_eq!(
        store.scan_neighbors_in(&r, "KNOWS", bob).expect("scan"),
        vec![alice],
    );
}

#[test]
fn put_edge_with_props_stores_property_body() {
    let db = open();
    let a = NodeId::from_raw(10);
    let b = NodeId::from_raw(20);
    db.write(|s, t| {
        let mut props = EdgeProperties::new();
        props.set(7, coordinode_core::graph::types::Value::Int(42));
        s.put_edge(t, "OWNS", a, b, Some(&props)).expect("put");
    });

    let store = LocalEdgeStore;
    let r = db.read();
    let loaded = store
        .get_props(&r, "OWNS", a, b)
        .expect("ok")
        .expect("Some");
    assert_eq!(loaded.len(), 1);
}

#[test]
fn edge_without_props_returns_none_from_get_props() {
    // Property-less edge: adj entries exist, edgeprop body doesn't.
    // get_props returns None (NOT "edge does not exist").
    let db = open();
    let a = NodeId::from_raw(1);
    let b = NodeId::from_raw(2);
    db.write(|s, t| s.put_edge(t, "LIKES", a, b, None).expect("put"));
    let store = LocalEdgeStore;
    let r = db.read();
    assert!(store.get_props(&r, "LIKES", a, b).expect("ok").is_none());
}

#[test]
fn get_props_snapshot_reads_body_and_none_via_mvcc_snapshot() {
    // Snapshot-aware read (backup export path, ADR-040): an edge written
    // with a property body is returned through a plain engine snapshot,
    // and a property-less edge returns None (not "missing edge").
    let db = open();
    let a = NodeId::from_raw(11);
    let b = NodeId::from_raw(22);
    db.write(|s, t| {
        s.put_edge(t, "OWNS", a, b, Some(&props_with(7, 42)))
            .expect("put");
    });

    let store = LocalEdgeStore;
    let snap = db.engine.snapshot();
    let loaded = store
        .get_props_snapshot(&db.engine, &snap, "OWNS", a, b)
        .expect("ok")
        .expect("Some");
    assert_eq!(loaded.len(), 1);

    db.write(|s, t| s.put_edge(t, "LIKES", a, b, None).expect("put"));
    let snap2 = db.engine.snapshot();
    assert!(store
        .get_props_snapshot(&db.engine, &snap2, "LIKES", a, b)
        .expect("ok")
        .is_none());
}

#[test]
fn put_props_direct_writes_canonical_body_readable_via_snapshot() {
    // Direct engine write (backup restore path, ADR-016 — no transaction).
    // The body must be readable through a plain snapshot and byte-identical
    // to a transactional put_edge write (single canonical codec).
    let db = open();
    let a = NodeId::from_raw(31);
    let b = NodeId::from_raw(32);
    let store = LocalEdgeStore;
    store
        .put_props_direct(&db.engine, "OWNS", a, b, &props_with(7, 42))
        .expect("direct put");

    let snap = db.engine.snapshot();
    let loaded = store
        .get_props_snapshot(&db.engine, &snap, "OWNS", a, b)
        .expect("ok")
        .expect("Some");
    assert_eq!(loaded.len(), 1);

    // Same body via a transactional put_edge on a different edge reads back
    // to the same encoded bytes through get_props — proves wire identity.
    let c = NodeId::from_raw(33);
    db.write(|s, t| {
        s.put_edge(t, "OWNS", a, c, Some(&props_with(7, 42)))
            .expect("put_edge");
    });
    let r = db.read();
    let via_put_edge = store
        .get_props(&r, "OWNS", a, c)
        .expect("ok")
        .expect("Some");
    assert_eq!(via_put_edge.len(), loaded.len());
}

#[test]
fn multiple_neighbors_listed_in_sorted_order() {
    // The posting list maintains sorted order; scan returns it
    // unchanged. Verified across multi-merge writes.
    let db = open();
    let src = NodeId::from_raw(1);
    db.write(|s, t| {
        // Insert out of order to exercise the merge operator's sort.
        for tgt in [5u64, 3, 8, 1, 2] {
            s.put_edge(t, "F", src, NodeId::from_raw(tgt), None)
                .expect("put");
        }
    });
    let store = LocalEdgeStore;
    let r = db.read();
    let neighbors: Vec<u64> = store
        .scan_neighbors_out(&r, "F", src)
        .expect("scan")
        .iter()
        .map(|n| n.as_raw())
        .collect();
    assert_eq!(neighbors, vec![1, 2, 3, 5, 8]);
}

#[test]
fn delete_edge_removes_from_adjacency_and_props() {
    let db = open();
    let a = NodeId::from_raw(1);
    let b = NodeId::from_raw(2);
    let c = NodeId::from_raw(3);
    // Two outgoing edges from `a`. Delete the (a,b) edge.
    db.write(|s, t| {
        s.put_edge(t, "F", a, b, None).expect("put");
        s.put_edge(t, "F", a, c, None).expect("put");
    });
    db.write(|s, t| s.delete_edge(t, "F", a, b).expect("delete"));

    let store = LocalEdgeStore;
    let r = db.read();
    // `b` is gone from a's forward neighbors; `c` remains.
    let out: Vec<u64> = store
        .scan_neighbors_out(&r, "F", a)
        .expect("scan")
        .iter()
        .map(|n| n.as_raw())
        .collect();
    assert_eq!(out, vec![3]);
    // `a` is gone from b's reverse neighbors.
    assert!(store
        .scan_neighbors_in(&r, "F", b)
        .expect("scan")
        .is_empty());
}

#[test]
fn delete_edge_is_idempotent() {
    let db = open();
    // Never created — delete must still succeed (merge remove on an
    // absent uid is a no-op; edgeprop delete on a missing key is fine).
    db.write(|s, t| {
        s.delete_edge(t, "F", NodeId::from_raw(9), NodeId::from_raw(10))
            .expect("delete missing");
    });
}

#[test]
fn edge_types_are_isolated_in_adj() {
    // Same (src, tgt) pair under different edge types must NOT share
    // adjacency entries. Different keyspaces.
    let db = open();
    let a = NodeId::from_raw(1);
    let b = NodeId::from_raw(2);
    db.write(|s, t| {
        s.put_edge(t, "KNOWS", a, b, None).expect("put");
        s.put_edge(t, "LIKES", a, b, None).expect("put");
    });
    db.write(|s, t| s.delete_edge(t, "KNOWS", a, b).expect("delete KNOWS"));

    let store = LocalEdgeStore;
    let r = db.read();
    // KNOWS gone, LIKES preserved.
    assert!(store
        .scan_neighbors_out(&r, "KNOWS", a)
        .expect("scan")
        .is_empty());
    assert_eq!(
        store.scan_neighbors_out(&r, "LIKES", a).expect("scan"),
        vec![b],
    );
}

#[test]
fn put_temporal_versions_round_trip_via_scan() {
    let db = open();
    let a = NodeId::from_raw(1);
    let b = NodeId::from_raw(2);
    db.write(|s, t| {
        for (vf, salary) in [(1000i64, 50_000), (2000, 60_000), (3000, 70_000)] {
            s.put_edge_temporal(t, "WORKS_AT", a, b, vf, &props_with(1, salary))
                .expect("put temporal");
        }
    });
    let store = LocalEdgeStore;
    let r = db.read();
    let versions = store
        .scan_edge_versions(&r, "WORKS_AT", a, b)
        .expect("scan versions");
    assert_eq!(versions.len(), 3);
    assert_eq!(versions[0].0, 1000);
    assert_eq!(versions[2].0, 3000);
}

#[test]
fn get_props_at_returns_largest_valid_from_le_query() {
    let db = open();
    let a = NodeId::from_raw(1);
    let b = NodeId::from_raw(2);
    db.write(|s, t| {
        s.put_edge_temporal(t, "E", a, b, 1000, &props_with(1, 10))
            .unwrap();
        s.put_edge_temporal(t, "E", a, b, 2000, &props_with(1, 20))
            .unwrap();
        s.put_edge_temporal(t, "E", a, b, 3000, &props_with(1, 30))
            .unwrap();
    });
    let store = LocalEdgeStore;
    let r = db.read();

    // At 1500 — only the 1000-version is visible.
    let p = store
        .get_props_at(&r, "E", a, b, 1500)
        .unwrap()
        .expect("present");
    assert_eq!(
        p.get(1),
        Some(&coordinode_core::graph::types::Value::Int(10))
    );

    // At 2500 — pick the 2000-version (largest <= 2500).
    let p = store
        .get_props_at(&r, "E", a, b, 2500)
        .unwrap()
        .expect("present");
    assert_eq!(
        p.get(1),
        Some(&coordinode_core::graph::types::Value::Int(20))
    );

    // At 3000 — boundary inclusive: pick the 3000-version.
    let p = store
        .get_props_at(&r, "E", a, b, 3000)
        .unwrap()
        .expect("present");
    assert_eq!(
        p.get(1),
        Some(&coordinode_core::graph::types::Value::Int(30))
    );
}

#[test]
fn get_props_at_before_first_version_returns_none() {
    let db = open();
    let a = NodeId::from_raw(1);
    let b = NodeId::from_raw(2);
    db.write(|s, t| {
        s.put_edge_temporal(t, "E", a, b, 5000, &props_with(1, 1))
            .unwrap();
    });
    let store = LocalEdgeStore;
    let r = db.read();
    assert!(store.get_props_at(&r, "E", a, b, 1000).unwrap().is_none());
}

#[test]
fn delete_temporal_version_removes_only_that_version() {
    let db = open();
    let a = NodeId::from_raw(1);
    let b = NodeId::from_raw(2);
    db.write(|s, t| {
        s.put_edge_temporal(t, "E", a, b, 100, &props_with(1, 10))
            .unwrap();
        s.put_edge_temporal(t, "E", a, b, 200, &props_with(1, 20))
            .unwrap();
    });
    db.write(|s, t| s.delete_edge_temporal(t, "E", a, b, 100).unwrap());
    let store = LocalEdgeStore;
    let r = db.read();
    let versions = store.scan_edge_versions(&r, "E", a, b).unwrap();
    assert_eq!(versions.len(), 1);
    assert_eq!(versions[0].0, 200);
}

#[test]
fn delete_temporal_is_idempotent() {
    let db = open();
    db.write(|s, t| {
        s.delete_edge_temporal(t, "E", NodeId::from_raw(1), NodeId::from_raw(2), 9999)
            .expect("idempotent delete");
    });
}

#[test]
fn temporal_writes_isolated_per_pair() {
    let db = open();
    let a = NodeId::from_raw(1);
    let b = NodeId::from_raw(2);
    let c = NodeId::from_raw(3);
    db.write(|s, t| {
        s.put_edge_temporal(t, "E", a, b, 100, &props_with(1, 1))
            .unwrap();
        s.put_edge_temporal(t, "E", a, c, 200, &props_with(1, 2))
            .unwrap();
    });
    let store = LocalEdgeStore;
    let r = db.read();
    let only_ab = store.scan_edge_versions(&r, "E", a, b).unwrap();
    let only_ac = store.scan_edge_versions(&r, "E", a, c).unwrap();
    assert_eq!(only_ab.len(), 1);
    assert_eq!(only_ab[0].0, 100);
    assert_eq!(only_ac.len(), 1);
    assert_eq!(only_ac[0].0, 200);
}

#[test]
fn temporal_put_also_writes_adjacency() {
    // A temporal edge must be visible in the neighbour scan — the
    // adj merge runs as part of the temporal write.
    let db = open();
    let a = NodeId::from_raw(1);
    let b = NodeId::from_raw(2);
    db.write(|s, t| {
        s.put_edge_temporal(t, "E", a, b, 100, &props_with(1, 1))
            .unwrap();
    });
    let store = LocalEdgeStore;
    let r = db.read();
    assert_eq!(store.scan_neighbors_out(&r, "E", a).unwrap(), vec![b]);
    assert_eq!(store.scan_neighbors_in(&r, "E", b).unwrap(), vec![a]);
}

#[test]
fn concurrent_put_edge_on_super_node_preserves_all_edges() {
    // Concurrent merge-operator stress on a single source. Four
    // threads each add 25 distinct out-edges from the same src, each
    // in its own MVCC transaction. After join + commit, all 100 must
    // be visible — add/remove are commutative+idempotent merge
    // operands (adjacency bypasses OCC).
    use std::thread;

    let db = open();
    let src = NodeId::from_raw(1);
    let threads_n = 4u64;
    let per_thread = 25u64;

    let handles: Vec<_> = (0..threads_n)
        .map(|t| {
            let engine = Arc::clone(&db.engine);
            let oracle = Arc::clone(&db.oracle);
            thread::spawn(move || {
                let store = LocalEdgeStore;
                let mut tx = mvcc_txn(&engine, &oracle);
                for i in 0..per_thread {
                    let tgt = NodeId::from_raw(t * per_thread + i + 100);
                    store.put_edge(&mut tx, "F", src, tgt, None).expect("put");
                }
                commit(&mut tx);
            })
        })
        .collect();
    for h in handles {
        h.join().expect("thread join");
    }

    let store = LocalEdgeStore;
    let r = db.read();
    let mut neighbors: Vec<u64> = store
        .scan_neighbors_out(&r, "F", src)
        .expect("scan")
        .iter()
        .map(|n| n.as_raw())
        .collect();
    neighbors.sort_unstable();
    let expected: Vec<u64> = (100..100 + threads_n * per_thread).collect();
    assert_eq!(neighbors, expected, "all concurrent edges must be present");
}

#[test]
fn concurrent_add_then_remove_converges() {
    // Concurrent add(x) and remove(x) from different threads — both
    // ops must compose into a consistent posting list.
    use std::thread;

    let db = open();
    let src = NodeId::from_raw(1);

    // Pre-populate 10 edges so the remover has something to remove.
    db.write(|s, t| {
        for i in 0..10u64 {
            s.put_edge(t, "F", src, NodeId::from_raw(i + 200), None)
                .expect("setup");
        }
    });

    // Adder thread: adds 5 more.
    let engine_a = Arc::clone(&db.engine);
    let oracle_a = Arc::clone(&db.oracle);
    let adder = thread::spawn(move || {
        let store = LocalEdgeStore;
        let mut tx = mvcc_txn(&engine_a, &oracle_a);
        for i in 0..5u64 {
            store
                .put_edge(&mut tx, "F", src, NodeId::from_raw(i + 300), None)
                .expect("add");
        }
        commit(&mut tx);
    });
    // Remover thread: removes 5 of the pre-populated.
    let engine_r = Arc::clone(&db.engine);
    let oracle_r = Arc::clone(&db.oracle);
    let remover = thread::spawn(move || {
        let store = LocalEdgeStore;
        let mut tx = mvcc_txn(&engine_r, &oracle_r);
        for i in 0..5u64 {
            store
                .delete_edge(&mut tx, "F", src, NodeId::from_raw(i + 200))
                .expect("delete");
        }
        commit(&mut tx);
    });
    adder.join().expect("adder");
    remover.join().expect("remover");

    let store = LocalEdgeStore;
    let r = db.read();
    let mut neighbors: Vec<u64> = store
        .scan_neighbors_out(&r, "F", src)
        .expect("scan")
        .iter()
        .map(|n| n.as_raw())
        .collect();
    neighbors.sort_unstable();
    // Expect: (200..205) removed, (205..210) kept, (300..305) added.
    let expected: Vec<u64> = (205u64..210).chain(300..305).collect();
    assert_eq!(neighbors, expected);
}

#[test]
fn concurrent_put_edge_temporal_distinct_versions() {
    // Four threads each write a distinct valid_from version of the
    // same (et, src, tgt). After join, scan_edge_versions returns all
    // four, sorted by valid_from.
    use std::thread;

    let db = open();
    let src = NodeId::from_raw(1);
    let tgt = NodeId::from_raw(2);

    let handles: Vec<_> = (0..4u64)
        .map(|t| {
            let engine = Arc::clone(&db.engine);
            let oracle = Arc::clone(&db.oracle);
            thread::spawn(move || {
                let store = LocalEdgeStore;
                let mut tx = mvcc_txn(&engine, &oracle);
                let vf = (t as i64 + 1) * 1000;
                let mut props = EdgeProperties::new();
                props.set(1, coordinode_core::graph::types::Value::Int(vf));
                store
                    .put_edge_temporal(&mut tx, "E", src, tgt, vf, &props)
                    .expect("put temporal");
                commit(&mut tx);
            })
        })
        .collect();
    for h in handles {
        h.join().expect("join");
    }

    let store = LocalEdgeStore;
    let r = db.read();
    let versions = store.scan_edge_versions(&r, "E", src, tgt).expect("scan");
    let vfs: Vec<i64> = versions.iter().map(|(vf, _)| *vf).collect();
    assert_eq!(vfs, vec![1000, 2000, 3000, 4000]);
}

#[test]
fn corrupt_posting_list_surfaces_as_decode_error() {
    let db = open();
    let a = NodeId::from_raw(1);
    // Inject garbage bytes at the forward-adj key directly.
    db.engine
        .put(
            Partition::Adj,
            &encode_adj_key_forward("F", a),
            &[0xde, 0xad, 0xbe, 0xef],
        )
        .expect("inject");
    let store = LocalEdgeStore;
    let r = db.read();
    let err = store
        .scan_neighbors_out(&r, "F", a)
        .expect_err("must error");
    assert!(matches!(
        err,
        StoreError::Decode {
            kind: "posting list",
            ..
        }
    ));
}

#[test]
fn posting_at_snapshot_reads_committed_adjacency() {
    let db = open();
    let alice = NodeId::from_raw(1);
    let bob = NodeId::from_raw(2);
    let carol = NodeId::from_raw(3);
    db.write(|s, t| {
        s.put_edge(t, "KNOWS", alice, bob, None).expect("put");
        s.put_edge(t, "KNOWS", alice, carol, None).expect("put");
    });

    let store = LocalEdgeStore;
    // Absent key → None.
    assert!(store
        .posting_at_snapshot(&db.engine, None, &encode_adj_key_forward("KNOWS", bob))
        .expect("ok")
        .is_none());
    // Present key → the full peer set (latest committed).
    let fwd = store
        .posting_at_snapshot(&db.engine, None, &encode_adj_key_forward("KNOWS", alice))
        .expect("ok")
        .expect("present");
    let mut peers: Vec<u64> = fwd.iter().collect();
    peers.sort_unstable();
    assert_eq!(peers, vec![2, 3]);
}

#[test]
fn posting_at_snapshot_returns_empty_present_list_not_none() {
    // A present adjacency key whose list is empty must read back as
    // Some(empty), not None — a background scanner relies on this to
    // distinguish "no key" from "key drained" and clean up the latter.
    let db = open();
    let a = NodeId::from_raw(7);
    db.engine
        .put(
            Partition::Adj,
            &encode_adj_key_forward("F", a),
            &PostingList::new().to_bytes().expect("encode empty"),
        )
        .expect("inject empty");
    let got = LocalEdgeStore
        .posting_at_snapshot(&db.engine, None, &encode_adj_key_forward("F", a))
        .expect("ok");
    assert!(got.is_some(), "present empty key reads as Some");
    assert!(got.expect("some").is_empty());
}

// -- Discriminated edges (ADR-029) --

#[test]
fn discriminated_put_get_roundtrip_and_absent_is_none() {
    let db = open();
    let a = NodeId::from_raw(10);
    let b = NodeId::from_raw(20);
    db.write(|s, t| {
        s.put_edge_discriminated(
            t,
            "KNOWS",
            a,
            b,
            &Value::String("work".into()),
            &props_with(1, 100),
        )
        .unwrap();
        s.put_edge_discriminated(
            t,
            "KNOWS",
            a,
            b,
            &Value::String("college".into()),
            &props_with(1, 200),
        )
        .unwrap();
    });
    let store = LocalEdgeStore;
    let r = db.read();
    let work = store
        .get_props_for(&r, "KNOWS", a, b, &Value::String("work".into()))
        .unwrap()
        .expect("work instance present");
    assert_eq!(work.get(1), Some(&Value::Int(100)));
    let college = store
        .get_props_for(&r, "KNOWS", a, b, &Value::String("college".into()))
        .unwrap()
        .expect("college instance present");
    assert_eq!(college.get(1), Some(&Value::Int(200)));
    // A discriminator value with no instance reads as None.
    assert!(store
        .get_props_for(&r, "KNOWS", a, b, &Value::String("gym".into()))
        .unwrap()
        .is_none());
}

#[test]
fn scan_discriminators_sorted_with_set_adjacency() {
    let db = open();
    let a = NodeId::from_raw(10);
    let b = NodeId::from_raw(20);
    db.write(|s, t| {
        s.put_edge_discriminated(
            t,
            "KNOWS",
            a,
            b,
            &Value::String("work".into()),
            &props_with(1, 100),
        )
        .unwrap();
        s.put_edge_discriminated(
            t,
            "KNOWS",
            a,
            b,
            &Value::String("college".into()),
            &props_with(1, 200),
        )
        .unwrap();
    });
    let store = LocalEdgeStore;
    let r = db.read();
    let all = store
        .scan_discriminators(&r, "KNOWS", a, b, &PropertyType::String)
        .unwrap();
    assert_eq!(all.len(), 2);
    // Ascending by order-preserving key: "college" < "work".
    assert_eq!(all[0].0, Value::String("college".into()));
    assert_eq!(all[1].0, Value::String("work".into()));
    assert_eq!(all[0].1.get(1), Some(&Value::Int(200)));
    // Adjacency stays set semantics: the target appears exactly once.
    assert_eq!(store.scan_neighbors_out(&r, "KNOWS", a).unwrap(), vec![b]);
}

#[test]
fn put_edge_discriminated_rejects_unsupported_discriminator() {
    let db = open();
    let a = NodeId::from_raw(1);
    let b = NodeId::from_raw(2);
    db.write(|s, t| {
        let err = s.put_edge_discriminated(t, "E", a, b, &Value::Null, &props_with(1, 1));
        assert!(
            matches!(err, Err(StoreError::Invariant(_))),
            "Null is not a supported discriminator"
        );
    });
}
