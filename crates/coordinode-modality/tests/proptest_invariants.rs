//! Property-based tests for store invariants. These check the
//! *contract* of the typed APIs against many randomly generated
//! inputs, catching edge cases per-store unit tests miss.
//!
//! Scope kept tight on purpose — long-running proptest cases hurt
//! workspace test time, so each property uses a small case budget
//! (32-64 cases) and bounded input ranges.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_core::graph::node::{NodeId, NodeRecord};
use coordinode_core::graph::types::Value;
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_modality::{
    Bbox, Crs, IndexStore, LocalIndexStore, LocalNodeStore, LocalSpatialStore, NodeStore, Point,
    SpatialStore,
};
use coordinode_storage::engine::transaction::{CommitContext, Transaction};
use proptest::prelude::*;

/// Logic-test fixture — invariant proptests verify per-modality
/// store behaviour, not persistence. Memory backing → ~2× faster
/// per proptest case → ~4× total wall clock for the 4 spatial /
/// node-versioning / index-ordering invariants (each runs 32+
/// generated cases). Default backend honours
/// `COORDINODE_TEST_BACKEND`; CI matrix flips to disk to catch
/// FS-specific bugs MemFs would miss.
fn open_engine() -> coordinode_test_fixtures::EngineFixture {
    coordinode_test_fixtures::engine_for_logic()
}

proptest! {
    // 32 cases ≈ 4× faster than 128, still covers the property invariants
    // with a reasonable sample of randomly-generated inputs. Override via
    // PROPTEST_CASES env var when running a deep regression check.
    #![proptest_config(ProptestConfig { cases: 32, .. ProptestConfig::default() })]

    /// Index value-sortable encoding: for any two values of the same
    /// type, the lexicographic byte order of their encoded keys must
    /// match the natural type order. The IndexStore contract relies
    /// on this for range scans.
    #[test]
    fn index_int_value_ordering_preserved(
        a in any::<i64>(),
        b in any::<i64>(),
    ) {
        let fx = open_engine();
        let engine = &fx.engine;
        let store = LocalIndexStore::new(engine);
        store
            .put_entry("p", &[Value::Int(a)], NodeId::from_raw(1))
            .unwrap();
        store
            .put_entry("p", &[Value::Int(b)], NodeId::from_raw(2))
            .unwrap();
        let all = store.scan_all("p").unwrap();
        // Two entries regardless of value equality — they share the
        // encoded value but have distinct node_id suffixes.
        prop_assert_eq!(all.len(), 2);
        // Same-name index: keys sorted by (encoded value, node_id).
        // a vs b numeric order => key byte order.
        if a < b {
            // node 1 (a) comes before node 2 (b)
            let first_id = all[0].1.as_raw();
            prop_assert_eq!(first_id, 1);
        } else if a > b {
            let first_id = all[0].1.as_raw();
            prop_assert_eq!(first_id, 2);
        }
    }

    /// NodeStore temporal: for any non-empty set of valid_from
    /// timestamps, scan_versions returns them in ascending order.
    #[test]
    fn node_scan_versions_always_sorted(
        valid_froms in prop::collection::vec(any::<i64>(), 1..16),
    ) {
        let fx = open_engine();
        let engine = &fx.engine;
        let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
        let store = LocalNodeStore;
        let id = NodeId::from_raw(1);
        {
            let read_ts = oracle.next();
            let mut txn = Transaction::new(engine, Some(&oracle), read_ts, Some(engine.snapshot()));
            for &vf in &valid_froms {
                store
                    .put_temporal(&mut txn, 0, id, vf, &NodeRecord::new("L"))
                    .unwrap();
            }
            let wc = WriteConcern::majority();
            let ctx = CommitContext {
                write_concern: &wc,
                pipeline: None,
                id_gen: None,
                drain_buffer: None,
                nvme_write_buffer: None,
            };
            txn.commit(&ctx).unwrap();
        }
        let read_ts = oracle.next();
        let rtxn = Transaction::new(engine, Some(&oracle), read_ts, Some(engine.snapshot()));
        let versions = store.scan_versions(&rtxn, 0, id).unwrap();
        // Deduplicate (same valid_from collapses).
        let mut expected: Vec<i64> = valid_froms;
        expected.sort_unstable();
        expected.dedup();
        let got: Vec<i64> = versions.iter().map(|(vf, _)| *vf).collect();
        prop_assert_eq!(got, expected);
    }

    /// SpatialStore: any point inserted at coordinates that fall
    /// inside a wide bbox is found by scan_within_bbox. Tests the
    /// curve-windowing + post-filter together.
    #[test]
    fn spatial_inserted_point_is_findable(
        x in -1e6_f64 .. 1e6,
        y in -1e6_f64 .. 1e6,
    ) {
        let fx = open_engine();
        let engine = &fx.engine;
        let store = LocalSpatialStore;
        let id = NodeId::from_raw(1);
        let p = Point::new_2d(Crs::Cartesian2d, x, y);
        {
            let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
            let read_ts = oracle.next();
            let mut txn = Transaction::new(engine, Some(&oracle), read_ts, Some(engine.snapshot()));
            store.insert(&mut txn, 1, id, &p).unwrap();
            let wc = WriteConcern::majority();
            let ctx = CommitContext {
                write_concern: &wc,
                pipeline: None,
                id_gen: None,
                drain_buffer: None,
                nvme_write_buffer: None,
            };
            txn.commit(&ctx).unwrap();
        }
        // Bbox covers the whole quantised range we configured.
        let bbox = Bbox {
            lower: Point::new_2d(Crs::Cartesian2d, -1e9, -1e9),
            upper: Point::new_2d(Crs::Cartesian2d, 1e9, 1e9),
        };
        let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
        let read_ts = oracle.next();
        let rtxn = Transaction::new(engine, Some(&oracle), read_ts, Some(engine.snapshot()));
        let hits = store.scan_within_bbox(&rtxn, 1, Crs::Cartesian2d, &bbox).unwrap();
        prop_assert_eq!(hits.len(), 1);
        prop_assert_eq!(hits[0].0, id);
    }

    /// Symmetric Spatial property: a point OUTSIDE the bbox is never
    /// returned by scan_within_bbox, regardless of where it sits on
    /// the Morton curve.
    #[test]
    fn spatial_out_of_bbox_never_returned(
        // Point coords outside the bbox [0, 10] in both axes.
        ox in 11.0_f64 .. 100.0,
        oy in 11.0_f64 .. 100.0,
    ) {
        let fx = open_engine();
        let engine = &fx.engine;
        let store = LocalSpatialStore;
        {
            let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
            let read_ts = oracle.next();
            let mut txn = Transaction::new(engine, Some(&oracle), read_ts, Some(engine.snapshot()));
            store
                .insert(
                    &mut txn,
                    1,
                    NodeId::from_raw(99),
                    &Point::new_2d(Crs::Cartesian2d, ox, oy),
                )
                .unwrap();
            let wc = WriteConcern::majority();
            let ctx = CommitContext {
                write_concern: &wc,
                pipeline: None,
                id_gen: None,
                drain_buffer: None,
                nvme_write_buffer: None,
            };
            txn.commit(&ctx).unwrap();
        }
        let bbox = Bbox {
            lower: Point::new_2d(Crs::Cartesian2d, 0.0, 0.0),
            upper: Point::new_2d(Crs::Cartesian2d, 10.0, 10.0),
        };
        let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
        let read_ts = oracle.next();
        let rtxn = Transaction::new(engine, Some(&oracle), read_ts, Some(engine.snapshot()));
        let hits = store.scan_within_bbox(&rtxn, 1, Crs::Cartesian2d, &bbox).unwrap();
        prop_assert!(hits.is_empty());
    }
}
