use super::*;
use crate::engine::core::StorageEngine;
/// Logic-test fixture — coordinator/OCC tests verify behaviour
/// without depending on disk semantics, so route through MemFs
/// for ~2× speed-up. Returns `(_backing, engine)` keeping the
/// same tuple shape as before (callers do `let (_dir, engine)
/// = open_engine();`); the first element binds to the MemFs
/// `Arc` lifetime guard now instead of a `TempDir`.
fn open_engine() -> (std::sync::Arc<lsm_tree::fs::MemFs>, StorageEngine) {
    let (engine, fs) = crate::internal_test_helpers::memory_engine();
    (fs, engine)
}

#[test]
fn coordinator_visible_through_engine() {
    let (_dir, engine) = open_engine();
    // Public surface unchanged: get / snapshot / contains_key
    // delegate to the inner LocalMultiModalCoordinator.
    let _snap = engine.snapshot();
    let res = engine.get(Partition::Node, b"never").expect("get ok");
    assert!(res.is_none());
    let exists = engine
        .contains_key(Partition::Node, b"never")
        .expect("contains ok");
    assert!(!exists);
}

#[test]
fn coordinator_has_write_after_detects_put() {
    // SequenceNumberCounter::get() returns "next-to-allocate",
    // so snapshot() taken before a put coincides with the put's
    // seqno. Probe with `snapshot - 1` to assert the inequality
    // strictly (same shape as existing
    // has_write_after_detects_newer_write that probes oracle-100
    // with 99).
    let (_dir, engine) = open_engine();
    let before = engine.snapshot();
    engine.put(Partition::Node, b"k", b"v").expect("put");
    assert!(engine
        .has_write_after(Partition::Node, b"k", before.saturating_sub(1))
        .expect("hwa"),);
}

#[test]
fn coordinator_has_write_after_detects_delete() {
    let (_dir, engine) = open_engine();
    engine.put(Partition::Node, b"k", b"v").expect("seed");
    let snap = engine.snapshot();
    engine.delete(Partition::Node, b"k").expect("delete");
    assert!(engine
        .has_write_after(Partition::Node, b"k", snap)
        .expect("hwa"),);
}

#[test]
fn coordinator_tree_returns_every_partition() {
    // Every Partition::all() variant must be openable through
    // the coordinator — guards against future partition additions
    // that forget to register a tree.
    let (_dir, engine) = open_engine();
    for &part in Partition::all() {
        engine
            .coordinator()
            .tree(part)
            .unwrap_or_else(|e| panic!("tree({part:?}) failed: {e}"));
    }
}

#[test]
fn coordinator_has_write_after_no_write_returns_false() {
    // Probe an untouched key at a horizon that no write has
    // crossed → false. Guards against the false-positive case
    // where has_write_after over-reports.
    let (_dir, engine) = open_engine();
    let after = engine.snapshot();
    let observed = engine
        .has_write_after(Partition::Node, b"untouched", after)
        .expect("hwa");
    assert!(!observed);
}

#[test]
fn coordinator_contains_key_round_trip() {
    let (_dir, engine) = open_engine();
    assert!(!engine.contains_key(Partition::Node, b"k").unwrap());
    engine.put(Partition::Node, b"k", b"v").unwrap();
    assert!(engine.contains_key(Partition::Node, b"k").unwrap());
    engine.delete(Partition::Node, b"k").unwrap();
    assert!(!engine.contains_key(Partition::Node, b"k").unwrap());
}

#[test]
fn coordinator_prefix_scan_empty_returns_empty_iter() {
    let (_dir, engine) = open_engine();
    let iter = engine.prefix_scan(Partition::Node, b"never").unwrap();
    let count = iter.count();
    assert_eq!(count, 0);
}

#[test]
fn range_scan_inclusive_bounds_yields_both_endpoints() {
    // lsm-tree's `range` API uses `Bound::Included` for both ends
    // per our wrapper. Seed 3 keys; range over the outer two
    // should yield all three.
    let (_dir, engine) = open_engine();
    engine.put(Partition::Node, b"a", b"1").expect("put a");
    engine.put(Partition::Node, b"m", b"2").expect("put m");
    engine.put(Partition::Node, b"z", b"3").expect("put z");
    let iter = engine
        .range_scan(Partition::Node, b"a", b"z")
        .expect("range");
    let keys: Vec<Vec<u8>> = iter.map(|g| g.into_inner().unwrap().0.to_vec()).collect();
    assert_eq!(
        keys,
        vec![b"a".to_vec(), b"m".to_vec(), b"z".to_vec()],
        "both endpoints inclusive",
    );
}

#[test]
fn range_scan_skips_keys_outside_window() {
    // Pin the "skip dead zones" contract that G101's bbox
    // decomposition relies on — keys outside [start, end] must
    // NOT appear in the iterator at all (not "yielded then
    // filtered").
    let (_dir, engine) = open_engine();
    for k in [b"a" as &[u8], b"b", b"c", b"d", b"e"] {
        engine.put(Partition::Node, k, b"v").expect("put");
    }
    let iter = engine
        .range_scan(Partition::Node, b"b", b"d")
        .expect("range");
    let keys: Vec<Vec<u8>> = iter.map(|g| g.into_inner().unwrap().0.to_vec()).collect();
    assert_eq!(keys, vec![b"b".to_vec(), b"c".to_vec(), b"d".to_vec()]);
}

#[test]
fn range_scan_point_query_single_key() {
    // start == end → single-key probe via range API. Equivalent
    // to a get, but exercised through the range path (used by
    // 1×1-cell bbox decomposition).
    let (_dir, engine) = open_engine();
    engine.put(Partition::Node, b"k", b"v").expect("put");
    let iter = engine
        .range_scan(Partition::Node, b"k", b"k")
        .expect("range");
    let count = iter.count();
    assert_eq!(count, 1);
}

#[test]
fn range_seekable_yields_window_without_seek() {
    // Without any seek, a seekable scan walks the window exactly like
    // range_scan (inclusive bounds, dead zones outside excluded).
    let (_dir, engine) = open_engine();
    for k in [b"a" as &[u8], b"b", b"c", b"d", b"e"] {
        engine.put(Partition::Node, k, b"v").expect("put");
    }
    let seqno = engine.snapshot();
    let it = engine
        .range_seekable(Partition::Node, b"b", b"d", seqno)
        .expect("range_seekable");
    let mut keys = Vec::new();
    for g in it {
        keys.push(g.into_inner().expect("guard").0.to_vec());
    }
    assert_eq!(keys, vec![b"b".to_vec(), b"c".to_vec(), b"d".to_vec()]);
}

#[test]
fn range_seekable_seek_to_skips_dead_zone() {
    // The skip-scan contract G101 relies on: open one iterator over the
    // broad window, jump past a dead zone with seek_to — the in-between
    // keys are skipped at the iterator (never yielded), not post-filtered.
    let (_dir, engine) = open_engine();
    for k in [b"a" as &[u8], b"b", b"c", b"d", b"e", b"f", b"g"] {
        engine.put(Partition::Node, k, b"v").expect("put");
    }
    let seqno = engine.snapshot();
    let mut it = engine
        .range_seekable(Partition::Node, b"a", b"g", seqno)
        .expect("range_seekable");
    let first = it
        .next()
        .expect("first")
        .into_inner()
        .expect("guard")
        .0
        .to_vec();
    assert_eq!(first, b"a".to_vec());
    // Jump past b, c — next yields the first key >= "d".
    it.seek_to(b"d");
    let mut rest = Vec::new();
    for g in it {
        rest.push(g.into_inner().expect("guard").0.to_vec());
    }
    assert_eq!(
        rest,
        vec![b"d".to_vec(), b"e".to_vec(), b"f".to_vec(), b"g".to_vec()]
    );
}

#[test]
fn range_scan_inverted_bounds_yields_empty() {
    // Defensive: when `start > end` lsm-tree's range yields no
    // entries. Pins behaviour for callers that might construct
    // bounds inadvertently (the spatial decomposer guarantees
    // lo ≤ hi internally; this test backstops the API).
    let (_dir, engine) = open_engine();
    engine.put(Partition::Node, b"a", b"v").expect("put");
    engine.put(Partition::Node, b"b", b"v").expect("put");
    let iter = engine
        .range_scan(Partition::Node, b"b", b"a")
        .expect("range");
    assert_eq!(iter.count(), 0, "inverted bounds yield no entries");
}

#[test]
fn range_scan_isolated_per_partition() {
    // A key written to Node must not surface in a Schema range
    // scan over the same byte range, and vice versa. Schema
    // partition has bootstrap entries from engine open, so the
    // assertion is asymmetric: filter to our exact probe key.
    let (_dir, engine) = open_engine();
    let probe_key = b"range_scan_isolation_probe_unique";
    engine
        .put(Partition::Node, probe_key, b"node-value")
        .expect("put node");
    engine
        .put(Partition::Schema, probe_key, b"schema-value")
        .expect("put schema");

    let node_hits: Vec<_> = engine
        .range_scan(Partition::Node, probe_key, probe_key)
        .unwrap()
        .map(|g| {
            let (_k, v) = g.into_inner().unwrap();
            v.to_vec()
        })
        .collect();
    let schema_hits: Vec<_> = engine
        .range_scan(Partition::Schema, probe_key, probe_key)
        .unwrap()
        .map(|g| {
            let (_k, v) = g.into_inner().unwrap();
            v.to_vec()
        })
        .collect();
    assert_eq!(
        node_hits,
        vec![b"node-value".to_vec()],
        "Node range_scan must NOT see Schema partition's value",
    );
    assert_eq!(
        schema_hits,
        vec![b"schema-value".to_vec()],
        "Schema range_scan must NOT see Node partition's value",
    );
}

#[test]
fn coordinator_set_gc_watermark_observable() {
    // set_gc_watermark must be Release-ordered so a subsequent
    // Acquire-load (e.g. inside the compaction filter factory)
    // observes the new value. Smoke-check: write, then construct
    // a fresh factory that reads it.
    let (_dir, engine) = open_engine();
    engine.set_gc_watermark(12345);
    // The retention filter factory reads via Arc<AtomicU64>; we
    // can't get at it directly, but the engine's set_gc_watermark
    // delegates to coordinator.set_gc_watermark which uses
    // store(Release). Smoke: writes don't panic and the engine
    // remains operable.
    engine
        .put(Partition::Node, b"k", b"v")
        .expect("put after gc watermark set");
    let v = engine.get(Partition::Node, b"k").unwrap();
    assert_eq!(v.as_deref(), Some(b"v".as_slice()));
}

#[test]
fn coordinator_concurrent_reads_under_arc_share() {
    // The coordinator's read methods take &self only — Arc-share
    // a StorageEngine across threads and probe in parallel.
    use std::sync::Arc;
    use std::thread;

    let (dir, engine) = open_engine();
    let engine = Arc::new(engine);
    engine.put(Partition::Node, b"k", b"v").unwrap();
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let e = Arc::clone(&engine);
            thread::spawn(move || {
                for _ in 0..50 {
                    let v = e.get(Partition::Node, b"k").expect("get");
                    assert_eq!(v.as_deref(), Some(b"v".as_slice()));
                }
            })
        })
        .collect();
    for h in handles {
        h.join().expect("join");
    }
    drop(dir); // keep TempDir alive for the scope of the test
}

#[test]
fn occ_scope_track_dedupes_and_validates_clean() {
    let (_dir, engine) = open_engine();
    let coord = engine.coordinator();
    let scope = coord.occ_scope_at(engine.snapshot());
    // Track same key twice — set semantics.
    scope.track(Partition::Node, b"k1");
    scope.track(Partition::Node, b"k1");
    scope.track(Partition::Node, b"k2");
    assert_eq!(scope.tracked_count(), 2);
    // No writes since snapshot → no conflict.
    let res = coord.validate_occ(&scope).expect("validate ok");
    assert!(res.is_none());
}

#[test]
fn occ_scope_validate_detects_post_snapshot_write() {
    let (_dir, engine) = open_engine();
    let coord = engine.coordinator();
    // SequenceNumberCounter::get() returns "next-to-allocate", so
    // snapshot()'s value coincides with the next put's seqno. Probe
    // at `snapshot - 1` to assert strict post-snapshot newer-write
    // detection (same shape as coordinator_has_write_after_detects_put).
    let scope = coord.occ_scope_at(engine.snapshot().saturating_sub(1));
    scope.track(Partition::Node, b"hot");
    engine.put(Partition::Node, b"hot", b"v").expect("put");
    let conflict = coord
        .validate_occ(&scope)
        .expect("validate ok")
        .expect("conflict expected");
    assert_eq!(conflict.partition, Partition::Node);
    assert_eq!(conflict.key.as_slice(), b"hot");
}

#[test]
fn occ_scope_skips_commutative_partitions() {
    let (_dir, engine) = open_engine();
    let coord = engine.coordinator();
    let scope = coord.occ_scope_at(engine.snapshot());
    // Adj is commutative — even a post-snapshot write must NOT
    // produce a conflict on it.
    scope.track(Partition::Adj, b"adj_key");
    engine
        .merge(Partition::Adj, b"adj_key", b"any")
        .expect("merge");
    let res = coord.validate_occ(&scope).expect("validate ok");
    assert!(
        res.is_none(),
        "Adj writes are commutative — must not raise OCC conflict"
    );
}

#[test]
fn occ_scope_extend_merges_parallel_collections() {
    let (_dir, engine) = open_engine();
    let coord = engine.coordinator();
    let scope = coord.occ_scope_at(engine.snapshot());
    // Simulate a parallel worker collecting into a local Vec then
    // merging into the shared scope.
    let parallel_keys = vec![
        (Partition::Node, b"p1".to_vec()),
        (Partition::Node, b"p2".to_vec()),
        (Partition::EdgeProp, b"e1".to_vec()),
    ];
    scope.extend(parallel_keys);
    assert_eq!(scope.tracked_count(), 3);
}

#[test]
fn occ_scope_concurrent_track_thread_safe() {
    // OccScope must be Send+Sync so rayon workers can share &OccScope.
    use std::sync::Arc;
    use std::thread;
    let (_dir, engine) = open_engine();
    let scope = Arc::new(engine.coordinator().occ_scope_at(engine.snapshot()));
    let handles: Vec<_> = (0..4)
        .map(|tid| {
            let s = Arc::clone(&scope);
            thread::spawn(move || {
                for i in 0..50 {
                    let key = format!("k{tid}-{i}");
                    s.track(Partition::Node, key.as_bytes());
                }
            })
        })
        .collect();
    for h in handles {
        h.join().expect("join");
    }
    // 4 threads × 50 keys, all distinct.
    assert_eq!(scope.tracked_count(), 200);
}

#[test]
fn occ_scope_validate_empty_scope_is_clean() {
    // Empty scope → never any conflict regardless of writes.
    let (_dir, engine) = open_engine();
    let coord = engine.coordinator();
    let scope = coord.occ_scope_at(engine.snapshot());
    engine
        .put(Partition::Node, b"unrelated", b"v")
        .expect("put");
    let res = coord.validate_occ(&scope).expect("validate ok");
    assert!(
        res.is_none(),
        "empty scope must produce no conflict — nothing was read"
    );
}

#[test]
fn occ_scope_validate_returns_first_conflict_decisive() {
    // Multiple tracked keys, multiple post-snapshot writes — first
    // conflict found short-circuits. The return value identifies
    // *a* conflict; downstream retry handles all of them after
    // restart.
    let (_dir, engine) = open_engine();
    let coord = engine.coordinator();
    let scope = coord.occ_scope_at(engine.snapshot().saturating_sub(1));
    scope.track(Partition::Node, b"a");
    scope.track(Partition::Node, b"b");
    scope.track(Partition::Node, b"c");
    engine.put(Partition::Node, b"a", b"v").expect("put a");
    engine.put(Partition::Node, b"b", b"v").expect("put b");
    engine.put(Partition::Node, b"c", b"v").expect("put c");
    let conflict = coord
        .validate_occ(&scope)
        .expect("validate ok")
        .expect("conflict expected");
    // HashSet iteration order is unspecified, but the partition is
    // pinned and the key is one of the tracked set.
    assert_eq!(conflict.partition, Partition::Node);
    assert!(
        [b"a".as_slice(), b"b".as_slice(), b"c".as_slice()].contains(&conflict.key.as_slice()),
        "conflict key must be one of the tracked keys",
    );
}

#[test]
fn occ_scope_drain_resets_and_extends_after() {
    // drain() must clear the internal set so subsequent track/extend
    // start from an empty state (matches the test path that pulls
    // tracked keys out for assertion then expects an empty scope).
    let (_dir, engine) = open_engine();
    let scope = engine.coordinator().occ_scope_at(engine.snapshot());
    scope.track(Partition::Node, b"k1");
    scope.track(Partition::Node, b"k2");
    let drained = scope.drain();
    assert_eq!(drained.len(), 2);
    assert_eq!(scope.tracked_count(), 0);
    // Post-drain track works.
    scope.track(Partition::Node, b"k3");
    assert_eq!(scope.tracked_count(), 1);
    assert!(scope.contains(Partition::Node, b"k3"));
    assert!(!scope.contains(Partition::Node, b"k1"));
}

#[test]
fn occ_scope_contains_unit() {
    let (_dir, engine) = open_engine();
    let scope = engine.coordinator().occ_scope_at(engine.snapshot());
    scope.track(Partition::EdgeProp, b"ep_key");
    assert!(scope.contains(Partition::EdgeProp, b"ep_key"));
    assert!(!scope.contains(Partition::EdgeProp, b"missing"));
    // Same key, wrong partition → false (partition is part of the
    // identity).
    assert!(!scope.contains(Partition::Node, b"ep_key"));
}

#[test]
fn occ_scope_validate_multi_partition_mixed_outcome() {
    // Conflict on a non-commutative partition + writes on a
    // commutative partition: only the non-commutative key triggers.
    let (_dir, engine) = open_engine();
    let coord = engine.coordinator();
    let scope = coord.occ_scope_at(engine.snapshot().saturating_sub(1));
    scope.track(Partition::Adj, b"adj_key"); // commutative — skipped
    scope.track(Partition::Schema, b"schema_key"); // checked
    engine
        .merge(Partition::Adj, b"adj_key", b"x")
        .expect("merge");
    engine
        .put(Partition::Schema, b"schema_key", b"v")
        .expect("put");
    let conflict = coord
        .validate_occ(&scope)
        .expect("validate ok")
        .expect("schema_key conflict expected");
    assert_eq!(
        conflict.partition,
        Partition::Schema,
        "Adj is commutative, only Schema must surface as conflict",
    );
}

#[test]
fn multimodal_coordinator_dyn_dispatch_works() {
    // Bind to the trait through `&dyn`. This is the contract used
    // by EE Phase 3 (`MultiShardCoordinator`) — Layer 4 / Layer 5
    // hold a trait object, never a concrete type.
    let (_dir, engine) = open_engine();
    let coord: &dyn MultiModalCoordinator = engine.coordinator();
    let snap = coord.snapshot();
    engine.put(Partition::Node, b"dyn_k", b"v").expect("put");
    assert_eq!(
        coord
            .get(Partition::Node, b"dyn_k")
            .expect("dyn get")
            .as_deref(),
        Some(b"v".as_slice()),
    );
    // Snapshot-pinned read sees nothing — the put happened after
    // we captured `snap`.
    assert!(coord
        .snapshot_get(&snap, Partition::Node, b"dyn_k")
        .expect("dyn snap get")
        .is_none());
}

#[test]
fn coordinator_snapshot_isolation_holds() {
    let (_dir, engine) = open_engine();
    engine.put(Partition::Node, b"k", b"v1").expect("put v1");
    let snap = engine.snapshot();
    engine.put(Partition::Node, b"k", b"v2").expect("put v2");
    // Latest read sees v2.
    let latest = engine.get(Partition::Node, b"k").expect("get").unwrap();
    assert_eq!(latest.as_ref(), b"v2");
    // Snapshot read sees v1.
    let pinned = engine
        .snapshot_get(&snap, Partition::Node, b"k")
        .expect("snap get")
        .unwrap();
    assert_eq!(pinned.as_ref(), b"v1");
}
