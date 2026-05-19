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
use coordinode_modality::{
    Bbox, Crs, IndexStore, LocalIndexStore, LocalNodeStore, LocalSpatialStore, NodeStore, Point,
    SpatialStore,
};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use proptest::prelude::*;
use tempfile::TempDir;

fn open_engine() -> (TempDir, StorageEngine) {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "ep",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("open");
    (dir, engine)
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 128, .. ProptestConfig::default() })]

    /// Index value-sortable encoding: for any two values of the same
    /// type, the lexicographic byte order of their encoded keys must
    /// match the natural type order. The IndexStore contract relies
    /// on this for range scans.
    #[test]
    fn index_int_value_ordering_preserved(
        a in any::<i64>(),
        b in any::<i64>(),
    ) {
        let (_dir, engine) = open_engine();
        let store = LocalIndexStore::new(&engine);
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
        let (_dir, engine) = open_engine();
        let store = LocalNodeStore::new(&engine);
        let id = NodeId::from_raw(1);
        for &vf in &valid_froms {
            store
                .put_temporal(0, id, vf, &NodeRecord::new("L"))
                .unwrap();
        }
        let versions = store.scan_versions(0, id).unwrap();
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
        let (_dir, engine) = open_engine();
        let store = LocalSpatialStore::new(&engine);
        let id = NodeId::from_raw(1);
        let p = Point::new_2d(Crs::Cartesian2d, x, y);
        store.insert(1, id, &p).unwrap();
        // Bbox covers the whole quantised range we configured.
        let bbox = Bbox {
            lower: Point::new_2d(Crs::Cartesian2d, -1e9, -1e9),
            upper: Point::new_2d(Crs::Cartesian2d, 1e9, 1e9),
        };
        let hits = store.scan_within_bbox(1, Crs::Cartesian2d, &bbox).unwrap();
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
        let (_dir, engine) = open_engine();
        let store = LocalSpatialStore::new(&engine);
        store
            .insert(
                1,
                NodeId::from_raw(99),
                &Point::new_2d(Crs::Cartesian2d, ox, oy),
            )
            .unwrap();
        let bbox = Bbox {
            lower: Point::new_2d(Crs::Cartesian2d, 0.0, 0.0),
            upper: Point::new_2d(Crs::Cartesian2d, 10.0, 10.0),
        };
        let hits = store.scan_within_bbox(1, Crs::Cartesian2d, &bbox).unwrap();
        prop_assert!(hits.is_empty());
    }
}
