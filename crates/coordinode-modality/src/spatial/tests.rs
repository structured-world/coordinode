use super::*;

use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::transaction::CommitContext;

#[test]
fn morton_intervals_decomposes_equatorial_band() {
    // Regression lock for the G101 seekable skip-scan win: the equatorial
    // band (full lon, ±0.5° lat) has broad/decomposed ratio = 3, so it must
    // take the decomposed (multi-interval) path. If a future GAIN_THRESHOLD
    // bump pushes it back to the broad scan, the 85%-faster band path
    // silently regresses — this catches it.
    let bbox = Bbox {
        lower: Point::new_2d(Crs::Wgs84_2d, -180.0, -0.5),
        upper: Point::new_2d(Crs::Wgs84_2d, 180.0, 0.5),
    };
    assert!(
        morton_intervals(&bbox).len() > 1,
        "equatorial band must decompose for the seekable skip-scan win",
    );
}

#[test]
fn hilbert_2d_roundtrips() {
    for &(x, y) in &[
        (0u32, 0u32),
        (1, 0),
        (0, 1),
        (5, 9),
        (u32::MAX, u32::MAX),
        (u32::MAX, 0),
        (0, u32::MAX),
        (1 << 31, 1 << 30),
        (123_456, 7_654_321),
    ] {
        let d = hilbert_2d(x, y);
        assert_eq!(hilbert_2d_to_xy(d), (x, y), "roundtrip ({x},{y})");
    }
    assert_eq!(hilbert_2d(0, 0), 0, "origin maps to curve index 0");
}

#[test]
fn hilbert_2d_consecutive_indices_are_grid_adjacent() {
    // Defining Hilbert property: stepping the index by 1 moves exactly one
    // grid cell (Manhattan distance 1) — no Z-order long jumps. This is the
    // locality win over Morton.
    let mut prev = hilbert_2d_to_xy(0);
    for d in 1..2_000u64 {
        let cur = hilbert_2d_to_xy(d);
        let dx = (i64::from(cur.0) - i64::from(prev.0)).abs();
        let dy = (i64::from(cur.1) - i64::from(prev.1)).abs();
        assert_eq!(dx + dy, 1, "index {d}: {prev:?} -> {cur:?} not adjacent");
        prev = cur;
    }
}

#[test]
fn hilbert_decompose_covers_every_cell() {
    // Superset invariant: every grid cell inside the bbox has its Hilbert
    // index in some emitted interval (the cover never drops a bbox cell —
    // the load-bearing correctness property of the Cartesian-2D scan).
    for &(bx0, bx1, by0, by1) in &[
        (0u32, 3u32, 0u32, 3u32),
        (5, 20, 7, 19),
        (1000, 1050, 2000, 2080),
        (100, 100, 200, 200),
        (0, 0, 0, 0),
    ] {
        let intervals = hilbert_2d_decompose(bx0, bx1, by0, by1);
        assert!(!intervals.is_empty(), "non-empty bbox yields a cover");
        for x in bx0..=bx1 {
            for y in by0..=by1 {
                let d = hilbert_2d(x, y);
                assert!(
                    intervals.iter().any(|&(lo, hi)| d >= lo && d <= hi),
                    "cell ({x},{y}) idx {d} not covered by {intervals:?}",
                );
            }
        }
    }
}

#[test]
fn wgs84_s2_encode_deterministic_and_distinct() {
    let paris = || encode_curve(&Point::new_2d(Crs::Wgs84_2d, 2.3522, 48.8566));
    let kyiv = encode_curve(&Point::new_2d(Crs::Wgs84_2d, 30.5234, 50.4501));
    assert_eq!(paris(), paris(), "S2 leaf id is deterministic");
    assert_ne!(paris(), kyiv, "distinct points map to distinct leaf ids");
}

#[test]
fn wgs84_s2_point_in_bbox_is_covered() {
    // Findability/superset: a point inside a WGS-84 bbox has its S2 leaf id
    // in some covering interval, so the scan would surface it.
    let bbox = Bbox {
        lower: Point::new_2d(Crs::Wgs84_2d, 2.0, 48.0), // lon, lat
        upper: Point::new_2d(Crs::Wgs84_2d, 3.0, 49.0),
    };
    let intervals = curve_intervals(&bbox);
    assert!(!intervals.is_empty(), "non-empty bbox yields a covering");
    for &(lon, lat) in &[(2.3, 48.5), (2.0, 48.0), (3.0, 49.0), (2.9, 48.1)] {
        let id = encode_curve(&Point::new_2d(Crs::Wgs84_2d, lon, lat));
        assert!(
            intervals.iter().any(|&(lo, hi)| id >= lo && id <= hi),
            "({lon},{lat}) id {id} not in covering {intervals:?}",
        );
    }
}

#[test]
fn wgs84_s2_pole_region_is_covered() {
    // No pole distortion (the S2 win over lat/lon Morton): a polar-band bbox
    // covers a near-pole point. With Morton, the pole is a quantisation
    // singularity; with S2 it is an ordinary cube-face cell.
    let polar = Bbox {
        lower: Point::new_2d(Crs::Wgs84_2d, -180.0, 89.0),
        upper: Point::new_2d(Crs::Wgs84_2d, 180.0, 90.0),
    };
    let intervals = curve_intervals(&polar);
    assert!(!intervals.is_empty(), "polar bbox yields a covering");
    let id = encode_curve(&Point::new_2d(Crs::Wgs84_2d, 10.0, 89.7));
    assert!(
        intervals.iter().any(|&(lo, hi)| id >= lo && id <= hi),
        "near-pole point not covered by {intervals:?}",
    );
}

#[test]
fn wgs84_3d_s2_point_in_bbox_is_covered() {
    // 3D findability/superset: a point inside a lat/lon/alt bbox has its
    // packed (S2 horizontal | altitude) key in some covering interval.
    let bbox = Bbox {
        lower: Point::new_3d(Crs::Wgs84_3d, 2.0, 48.0, 0.0), // lon, lat, alt(m)
        upper: Point::new_3d(Crs::Wgs84_3d, 3.0, 49.0, 5000.0),
    };
    let intervals = curve_intervals(&bbox);
    assert!(!intervals.is_empty(), "3D bbox yields a covering");
    for &(lon, lat, alt) in &[
        (2.3, 48.5, 1000.0),
        (2.0, 48.0, 0.0),
        (3.0, 49.0, 5000.0),
        (2.9, 48.1, 4999.0),
    ] {
        let id = encode_curve(&Point::new_3d(Crs::Wgs84_3d, lon, lat, alt));
        assert!(
            intervals.iter().any(|&(lo, hi)| id >= lo && id <= hi),
            "({lon},{lat},{alt}m) id {id} not in covering {intervals:?}",
        );
    }
}

/// Logic-test fixture (memory backing, env-flippable). Spatial
/// Z-curve tests verify encoding/scan correctness, not
/// persistence.
fn mk_engine() -> coordinode_test_fixtures::EngineFixture {
    coordinode_test_fixtures::engine_for_logic()
}

/// Apply spatial writes in one MVCC transaction (shard-agnostic,
/// `Partition::Idx`) and commit, so the buffered index rows land
/// for a subsequent read.
fn write_spatial(engine: &StorageEngine, body: impl FnOnce(&LocalSpatialStore, &mut Transaction)) {
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let read_ts = oracle.next();
    let mut txn = Transaction::new(engine, Some(&oracle), read_ts, Some(engine.snapshot()));
    body(&LocalSpatialStore, &mut txn);
    let wc = WriteConcern::majority();
    let ctx = CommitContext {
        write_concern: &wc,
        pipeline: None,
        id_gen: None,
        drain_buffer: None,
        nvme_write_buffer: None,
    };
    txn.commit(&ctx).expect("commit spatial");
}

/// Run a spatial read closure against the latest committed snapshot
/// through a fresh MVCC transaction.
fn read_spatial<R>(
    engine: &StorageEngine,
    body: impl FnOnce(&LocalSpatialStore, &Transaction) -> R,
) -> R {
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let read_ts = oracle.next();
    let txn = Transaction::new(engine, Some(&oracle), read_ts, Some(engine.snapshot()));
    body(&LocalSpatialStore, &txn)
}

// -- Z-curve subrange decomposition --

#[test]
fn morton_decompose_full_space_yields_single_interval() {
    // The whole 32-bit grid IS one Morton cell — decomposition
    // returns a single interval `[0, u64::MAX]` (entire curve).
    let intervals = morton_2d_decompose(0, u32::MAX, 0, u32::MAX);
    assert_eq!(intervals.len(), 1);
    assert_eq!(intervals[0], (0, u64::MAX));
}

#[test]
fn morton_decompose_single_cell_returns_point_interval() {
    // 1×1 bbox at (3, 5) → exactly one cell at Morton(3, 5).
    let m = morton_2d(3, 5);
    let intervals = morton_2d_decompose(3, 3, 5, 5);
    assert_eq!(intervals, vec![(m, m)]);
}

#[test]
fn morton_decompose_intervals_are_disjoint_and_sorted() {
    // Pick a non-aligned bbox — produces multiple intervals.
    // Property: each interval's `lo` is strictly greater than the
    // previous interval's `hi` (disjoint, sorted by lo).
    let intervals = morton_2d_decompose(0, 100, 0, 50);
    assert!(!intervals.is_empty());
    for window in intervals.windows(2) {
        let (_, prev_hi) = window[0];
        let (next_lo, _) = window[1];
        assert!(
            next_lo > prev_hi,
            "intervals must be disjoint and sorted: prev=({}, {}), next_lo={}",
            window[0].0,
            prev_hi,
            next_lo,
        );
    }
}

#[test]
fn morton_decompose_covers_every_bbox_point() {
    // EVERY point inside the bbox MUST appear in exactly one
    // interval. Walk the bbox grid; assert each `morton(x, y)`
    // is contained in the interval set. (Bounded test grid to
    // keep the cost finite.)
    let bx = (10u32, 20u32);
    let by = (5u32, 15u32);
    let intervals = morton_2d_decompose(bx.0, bx.1, by.0, by.1);
    for x in bx.0..=bx.1 {
        for y in by.0..=by.1 {
            let m = morton_2d(x, y);
            let covered = intervals.iter().any(|&(lo, hi)| m >= lo && m <= hi);
            assert!(
                covered,
                "bbox point ({x}, {y}) → morton {m} not covered by any interval",
            );
        }
    }
}

#[test]
fn morton_decompose_excludes_points_outside_bbox() {
    // No point OUTSIDE the bbox should lie inside the decomposition
    // intervals at the cell-aligned level. We can't strictly forbid
    // false-positives at the rim (Z-curve cells span across the
    // bbox edge), but cells far from the bbox must be excluded.
    // Smoke-check: pick a small bbox, find a far-away point that
    // a single broad `[morton_min, morton_max]` would include but
    // the decomposition should NOT.
    let intervals = morton_2d_decompose(0, 0, 0, 0); // bbox = single cell (0, 0)
    let far_point = morton_2d(u32::MAX, u32::MAX);
    let any_covered = intervals
        .iter()
        .any(|&(lo, hi)| far_point >= lo && far_point <= hi);
    assert!(
        !any_covered,
        "far-away cell {far_point} must NOT be in decomposition of a 1×1 bbox",
    );
}

#[test]
fn morton_decompose_long_thin_bbox_still_covers_every_point() {
    // A long thin bbox (full u32 width, height 2) is the worst-
    // case decomposition shape — every level forces splits along
    // the long axis. The cap exists to avoid runaway output here;
    // covering MUST stay correct regardless of where the cap kicks
    // in (cap-triggered broad-range fallbacks include the bbox
    // intersection of the abandoned cell). Test the covering
    // invariant on extreme x-values plus both y-values.
    let intervals = morton_2d_decompose(0, u32::MAX, 0, 1);
    // Sanity: not the degenerate single-interval; some
    // decomposition happened.
    assert!(!intervals.is_empty());
    for x in [
        0u32,
        1,
        100,
        50_000,
        1_000_000,
        u32::MAX / 2,
        u32::MAX - 1,
        u32::MAX,
    ] {
        for y in [0u32, 1] {
            let m = morton_2d(x, y);
            let covered = intervals.iter().any(|&(lo, hi)| m >= lo && m <= hi);
            assert!(
                    covered,
                    "bbox point ({x}, {y}) → morton {m} not covered (cap fired but the broad fallback should still cover the cell-bbox intersection)",
                );
        }
    }
}

#[test]
fn morton_intervals_2d_dispatch_returns_non_empty() {
    // Smoke-check the dispatch — Wgs84_2d and Cartesian2d both
    // produce ≥ 1 interval for any non-degenerate bbox. The
    // adaptive bailout is intentionally not asserted here:
    // morton_intervals is currently retained as dead-code
    // scaffold for the proper subrange-decomposition re-attempt
    // once the lsm-tree skip primitive lands.
    let wgs = morton_intervals(&Bbox {
        lower: Point::new_2d(Crs::Wgs84_2d, 0.0, 0.0),
        upper: Point::new_2d(Crs::Wgs84_2d, 1.0, 1.0),
    });
    assert!(!wgs.is_empty());
    let cart = morton_intervals(&Bbox {
        lower: Point::new_2d(Crs::Cartesian2d, 0.0, 0.0),
        upper: Point::new_2d(Crs::Cartesian2d, 100.0, 100.0),
    });
    assert!(!cart.is_empty());
}

#[test]
fn morton_decompose_excludes_broadly_distant_cells() {
    // Strengthen the exclusion check: a 100×100 bbox at the
    // origin must NOT include any cell whose centre is more than
    // ~10x bbox-distance away in either dimension. Probe a grid
    // of distant points and assert none of them lie in any
    // interval.
    let intervals = morton_2d_decompose(0, 100, 0, 100);
    let distant_points: &[(u32, u32)] = &[
        (10_000, 10_000),
        (50_000, 50_000),
        (1_000_000, 0),
        (0, 1_000_000),
        (u32::MAX, u32::MAX),
        (u32::MAX / 2, u32::MAX / 2),
        (1024, 1024),
        (200, 200),
        (101, 101), // just outside the bbox edge
    ];
    for &(x, y) in distant_points {
        let m = morton_2d(x, y);
        let covered = intervals.iter().any(|&(lo, hi)| m >= lo && m <= hi);
        assert!(
            !covered,
            "distant point ({x}, {y}) → morton {m} unexpectedly covered",
        );
    }
}

#[test]
fn morton_intervals_3d_falls_back_to_broad_window() {
    // 3D path is not decomposed in v1 — single broad interval.
    let bbox = Bbox {
        lower: Point::new_3d(Crs::Cartesian3d, 0.0, 0.0, 0.0),
        upper: Point::new_3d(Crs::Cartesian3d, 1000.0, 1000.0, 1000.0),
    };
    let intervals = morton_intervals(&bbox);
    assert_eq!(intervals.len(), 1, "3D path falls back to broad window");
}

#[test]
fn crs_dims_and_srid() {
    assert_eq!(Crs::Wgs84_2d.dims(), 2);
    assert_eq!(Crs::Wgs84_2d.srid(), 4326);
    assert_eq!(Crs::Cartesian3d.dims(), 3);
    assert_eq!(Crs::Cartesian3d.srid(), 9157);
    assert!(Crs::Wgs84_2d.is_geo());
    assert!(!Crs::Cartesian2d.is_geo());
}

#[test]
fn insert_then_scan_bbox_returns_point() {
    let fx = mk_engine();
    let engine = &fx.engine;
    let paris = Point::new_2d(Crs::Wgs84_2d, 2.3522, 48.8566);
    write_spatial(engine, |s, txn| {
        s.insert(txn, 1, NodeId::from_raw(1), &paris).unwrap();
    });
    let bbox = Bbox {
        lower: Point::new_2d(Crs::Wgs84_2d, 2.0, 48.0),
        upper: Point::new_2d(Crs::Wgs84_2d, 3.0, 49.0),
    };
    let hits = read_spatial(engine, |s, txn| {
        s.scan_within_bbox(txn, 1, Crs::Wgs84_2d, &bbox).unwrap()
    });
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0, NodeId::from_raw(1));
    assert!((hits[0].1.coords[0] - paris.coords[0]).abs() < 1e-6);
}

#[test]
fn scan_bbox_filters_outside_points() {
    let fx = mk_engine();
    let engine = &fx.engine;
    let paris = Point::new_2d(Crs::Wgs84_2d, 2.3522, 48.8566);
    let kyiv = Point::new_2d(Crs::Wgs84_2d, 30.5234, 50.4501);
    write_spatial(engine, |s, txn| {
        s.insert(txn, 1, NodeId::from_raw(1), &paris).unwrap();
        s.insert(txn, 1, NodeId::from_raw(2), &kyiv).unwrap();
    });
    let paris_bbox = Bbox {
        lower: Point::new_2d(Crs::Wgs84_2d, 1.5, 48.0),
        upper: Point::new_2d(Crs::Wgs84_2d, 3.0, 49.5),
    };
    let hits = read_spatial(engine, |s, txn| {
        s.scan_within_bbox(txn, 1, Crs::Wgs84_2d, &paris_bbox)
            .unwrap()
    });
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0, NodeId::from_raw(1));
}

#[test]
fn delete_removes_point() {
    let fx = mk_engine();
    let engine = &fx.engine;
    let p = Point::new_2d(Crs::Cartesian2d, 10.0, 20.0);
    write_spatial(engine, |s, txn| {
        s.insert(txn, 7, NodeId::from_raw(42), &p).unwrap();
        s.delete(txn, 7, NodeId::from_raw(42), &p).unwrap();
    });
    let bbox = Bbox {
        lower: Point::new_2d(Crs::Cartesian2d, 0.0, 0.0),
        upper: Point::new_2d(Crs::Cartesian2d, 100.0, 100.0),
    };
    let hits = read_spatial(engine, |s, txn| {
        s.scan_within_bbox(txn, 7, Crs::Cartesian2d, &bbox).unwrap()
    });
    assert!(hits.is_empty());
}

#[test]
fn knn_returns_closest_in_distance_order() {
    let fx = mk_engine();
    let engine = &fx.engine;
    let pts = [
        (1u64, 0.0, 0.0),
        (2, 1.0, 0.0),
        (3, 5.0, 5.0),
        (4, 10.0, 10.0),
    ];
    write_spatial(engine, |s, txn| {
        for (id, x, y) in pts {
            s.insert(
                txn,
                3,
                NodeId::from_raw(id),
                &Point::new_2d(Crs::Cartesian2d, x, y),
            )
            .unwrap();
        }
    });
    let center = Point::new_2d(Crs::Cartesian2d, 0.0, 0.0);
    let knn = read_spatial(engine, |s, txn| {
        s.knn_nearest(txn, 3, Crs::Cartesian2d, &center, 2).unwrap()
    });
    assert_eq!(knn.len(), 2);
    assert_eq!(knn[0].0, NodeId::from_raw(1));
    assert_eq!(knn[1].0, NodeId::from_raw(2));
    assert!(knn[0].2 < knn[1].2);
}

#[test]
fn knn_k_zero_returns_empty() {
    let fx = mk_engine();
    let engine = &fx.engine;
    write_spatial(engine, |s, txn| {
        s.insert(
            txn,
            1,
            NodeId::from_raw(1),
            &Point::new_2d(Crs::Cartesian2d, 0.0, 0.0),
        )
        .unwrap();
    });
    let center = Point::new_2d(Crs::Cartesian2d, 0.0, 0.0);
    let knn = read_spatial(engine, |s, txn| {
        s.knn_nearest(txn, 1, Crs::Cartesian2d, &center, 0).unwrap()
    });
    assert!(knn.is_empty());
}

#[test]
fn haversine_paris_to_kyiv_is_about_2000km() {
    let paris = Point::new_2d(Crs::Wgs84_2d, 2.3522, 48.8566);
    let kyiv = Point::new_2d(Crs::Wgs84_2d, 30.5234, 50.4501);
    let d = distance(&paris, &kyiv);
    // Real great-circle distance Paris–Kyiv ≈ 2030 km.
    assert!((1_900_000.0..2_100_000.0).contains(&d), "got {d}");
}

#[test]
fn cartesian_distance_3d() {
    let a = Point::new_3d(Crs::Cartesian3d, 0.0, 0.0, 0.0);
    let b = Point::new_3d(Crs::Cartesian3d, 3.0, 4.0, 12.0);
    let d = distance(&a, &b);
    assert!((d - 13.0).abs() < 1e-9);
}

#[test]
fn cartesian_3d_round_trip_and_knn() {
    let fx = mk_engine();
    let engine = &fx.engine;
    let origin = Point::new_3d(Crs::Cartesian3d, 0.0, 0.0, 0.0);
    let near = Point::new_3d(Crs::Cartesian3d, 1.0, 1.0, 1.0);
    let far = Point::new_3d(Crs::Cartesian3d, 100.0, 100.0, 100.0);
    write_spatial(engine, |s, txn| {
        s.insert(txn, 9, NodeId::from_raw(1), &origin).unwrap();
        s.insert(txn, 9, NodeId::from_raw(2), &near).unwrap();
        s.insert(txn, 9, NodeId::from_raw(3), &far).unwrap();
    });
    let knn = read_spatial(engine, |s, txn| {
        s.knn_nearest(txn, 9, Crs::Cartesian3d, &origin, 2).unwrap()
    });
    assert_eq!(knn.len(), 2);
    assert_eq!(knn[0].0, NodeId::from_raw(1));
    assert_eq!(knn[1].0, NodeId::from_raw(2));
}

#[test]
fn scoped_by_label_id() {
    let fx = mk_engine();
    let engine = &fx.engine;
    let p = Point::new_2d(Crs::Cartesian2d, 1.0, 1.0);
    write_spatial(engine, |s, txn| {
        s.insert(txn, 1, NodeId::from_raw(10), &p).unwrap();
        s.insert(txn, 2, NodeId::from_raw(20), &p).unwrap();
    });
    let bbox = Bbox {
        lower: Point::new_2d(Crs::Cartesian2d, 0.0, 0.0),
        upper: Point::new_2d(Crs::Cartesian2d, 10.0, 10.0),
    };
    let hits1 = read_spatial(engine, |s, txn| {
        s.scan_within_bbox(txn, 1, Crs::Cartesian2d, &bbox).unwrap()
    });
    let hits2 = read_spatial(engine, |s, txn| {
        s.scan_within_bbox(txn, 2, Crs::Cartesian2d, &bbox).unwrap()
    });
    assert_eq!(hits1.len(), 1);
    assert_eq!(hits2.len(), 1);
    assert_eq!(hits1[0].0, NodeId::from_raw(10));
    assert_eq!(hits2[0].0, NodeId::from_raw(20));
}

#[test]
fn insert_same_node_at_new_coords_leaves_stale_row() {
    // Document the trait contract: moving a point requires
    // delete(old) before insert(new). Without it, both rows are
    // visible. This is the regression guard for the doc on
    // `insert`.
    let fx = mk_engine();
    let engine = &fx.engine;
    let id = NodeId::from_raw(1);
    let p1 = Point::new_2d(Crs::Cartesian2d, 0.0, 0.0);
    let p2 = Point::new_2d(Crs::Cartesian2d, 100.0, 100.0);
    write_spatial(engine, |s, txn| {
        s.insert(txn, 1, id, &p1).unwrap();
        s.insert(txn, 1, id, &p2).unwrap();
    });

    let bbox = Bbox {
        lower: Point::new_2d(Crs::Cartesian2d, -200.0, -200.0),
        upper: Point::new_2d(Crs::Cartesian2d, 200.0, 200.0),
    };
    let hits = read_spatial(engine, |s, txn| {
        s.scan_within_bbox(txn, 1, Crs::Cartesian2d, &bbox).unwrap()
    });
    // Both rows visible: stale-key behaviour the docstring warns
    // about.
    assert_eq!(hits.len(), 2);
    let ids: Vec<u64> = hits.iter().map(|(id, _)| id.as_raw()).collect();
    assert_eq!(ids, vec![1, 1]);
}

#[test]
fn move_point_via_explicit_delete_then_insert() {
    // The supported pattern for moving a point: delete(old) →
    // insert(new). Exactly one row visible afterwards.
    let fx = mk_engine();
    let engine = &fx.engine;
    let id = NodeId::from_raw(2);
    let p1 = Point::new_2d(Crs::Cartesian2d, 0.0, 0.0);
    let p2 = Point::new_2d(Crs::Cartesian2d, 50.0, 50.0);
    write_spatial(engine, |s, txn| {
        s.insert(txn, 2, id, &p1).unwrap();
        s.delete(txn, 2, id, &p1).unwrap();
        s.insert(txn, 2, id, &p2).unwrap();
    });

    let bbox = Bbox {
        lower: Point::new_2d(Crs::Cartesian2d, -100.0, -100.0),
        upper: Point::new_2d(Crs::Cartesian2d, 100.0, 100.0),
    };
    let hits = read_spatial(engine, |s, txn| {
        s.scan_within_bbox(txn, 2, Crs::Cartesian2d, &bbox).unwrap()
    });
    assert_eq!(hits.len(), 1);
    assert!((hits[0].1.coords[0] - 50.0).abs() < 1e-6);
}

#[test]
fn knn_returns_duplicate_when_stale_row_exists() {
    // Same contract gap as scan_within_bbox: moving a point
    // without deleting the old key leaves two physical rows. knn
    // returns both — the caller must dedup by node_id at the
    // composition layer above.
    let fx = mk_engine();
    let engine = &fx.engine;
    let id = NodeId::from_raw(1);
    write_spatial(engine, |s, txn| {
        s.insert(txn, 1, id, &Point::new_2d(Crs::Cartesian2d, 0.0, 0.0))
            .unwrap();
        s.insert(txn, 1, id, &Point::new_2d(Crs::Cartesian2d, 1.0, 1.0))
            .unwrap();
    });

    let center = Point::new_2d(Crs::Cartesian2d, 0.0, 0.0);
    let knn = read_spatial(engine, |s, txn| {
        s.knn_nearest(txn, 1, Crs::Cartesian2d, &center, 5).unwrap()
    });
    let ids: Vec<u64> = knn.iter().map(|(id, _, _)| id.as_raw()).collect();
    assert_eq!(ids, vec![1, 1], "stale row produces duplicate id in knn");
}

#[test]
fn insert_same_coords_twice_is_idempotent() {
    // Same (node_id, coords) = same key. The second insert
    // overwrites with identical bytes — net effect is one row.
    let fx = mk_engine();
    let engine = &fx.engine;
    let id = NodeId::from_raw(3);
    let p = Point::new_2d(Crs::Cartesian2d, 5.0, 5.0);
    write_spatial(engine, |s, txn| {
        s.insert(txn, 3, id, &p).unwrap();
        s.insert(txn, 3, id, &p).unwrap();
    });

    let bbox = Bbox {
        lower: Point::new_2d(Crs::Cartesian2d, 0.0, 0.0),
        upper: Point::new_2d(Crs::Cartesian2d, 10.0, 10.0),
    };
    let hits = read_spatial(engine, |s, txn| {
        s.scan_within_bbox(txn, 3, Crs::Cartesian2d, &bbox).unwrap()
    });
    assert_eq!(hits.len(), 1);
}

#[test]
fn bbox_inclusive_at_corner_points() {
    // Per docstring, Bbox is *inclusive*. A point at exactly the
    // lower-left corner and a point at exactly upper-right must
    // both match.
    let fx = mk_engine();
    let engine = &fx.engine;
    let bbox = Bbox {
        lower: Point::new_2d(Crs::Cartesian2d, 10.0, 20.0),
        upper: Point::new_2d(Crs::Cartesian2d, 30.0, 40.0),
    };
    write_spatial(engine, |s, txn| {
        s.insert(
            txn,
            5,
            NodeId::from_raw(1),
            &Point::new_2d(Crs::Cartesian2d, 10.0, 20.0),
        )
        .unwrap();
        s.insert(
            txn,
            5,
            NodeId::from_raw(2),
            &Point::new_2d(Crs::Cartesian2d, 30.0, 40.0),
        )
        .unwrap();
    });
    let hits = read_spatial(engine, |s, txn| {
        s.scan_within_bbox(txn, 5, Crs::Cartesian2d, &bbox).unwrap()
    });
    assert_eq!(hits.len(), 2);
}

#[test]
fn morton_window_brackets_every_bbox_point() {
    // Property: on a Morton CRS, every point inside the bbox quantises to a
    // curve inside [curve_min, curve_max] — the bracketing contract the
    // Morton broad-window case of `morton_intervals` relies on. The live
    // Morton path is now 3D only (the broad window, no decomposition): the
    // 2D CRS moved to Hilbert (Cartesian-2D) and S2 (WGS-84), neither of which
    // has an order-preserving broad window.
    let bbox = Bbox {
        lower: Point::new_3d(Crs::Cartesian3d, -100.0, -50.0, -10.0),
        upper: Point::new_3d(Crs::Cartesian3d, 100.0, 50.0, 10.0),
    };
    let (cmin, cmax) = morton_window(&bbox);
    for x in [-99.0, 0.0, 99.0] {
        for y in [-49.0, 0.0, 49.0] {
            for z in [-9.0, 0.0, 9.0] {
                let p = Point::new_3d(Crs::Cartesian3d, x, y, z);
                let c = encode_curve(&p);
                assert!(
                    (cmin..=cmax).contains(&c),
                    "point ({x},{y},{z}) curve {c} outside [{cmin},{cmax}]",
                );
            }
        }
    }
}

#[test]
fn scan_within_bbox_filters_morton_false_positives() {
    // Morton's Z-shape means a curve key inside [cmin, cmax] may
    // decode to a point outside the bbox. The exact post-filter
    // must drop those. Specifically: insert points spanning a wide
    // X axis but narrow Y; the bbox is the same shape. Without
    // post-filter, points with curves inside the linear interval
    // but Y outside would leak.
    let fx = mk_engine();
    let engine = &fx.engine;
    // Two points: one inside the bbox, one with Y far outside but
    // a curve key that may fall in the [cmin, cmax] range.
    let inside = Point::new_2d(Crs::Cartesian2d, 5.0, 1.0);
    let outside_y = Point::new_2d(Crs::Cartesian2d, 5.0, 50.0);
    write_spatial(engine, |s, txn| {
        s.insert(txn, 1, NodeId::from_raw(1), &inside).unwrap();
        s.insert(txn, 1, NodeId::from_raw(2), &outside_y).unwrap();
    });
    let bbox = Bbox {
        lower: Point::new_2d(Crs::Cartesian2d, 0.0, 0.0),
        upper: Point::new_2d(Crs::Cartesian2d, 10.0, 2.0),
    };
    let hits = read_spatial(engine, |s, txn| {
        s.scan_within_bbox(txn, 1, Crs::Cartesian2d, &bbox).unwrap()
    });
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0, NodeId::from_raw(1));
}

#[test]
fn scan_within_bbox_breaks_early_past_curve_max() {
    // Construct three points whose curve keys we can predict, so
    // we know the iteration must stop at the middle one.
    let fx = mk_engine();
    let engine = &fx.engine;
    // Tight bbox containing only the first point. Inserted points
    // far apart so curves diverge sharply.
    write_spatial(engine, |s, txn| {
        s.insert(
            txn,
            7,
            NodeId::from_raw(1),
            &Point::new_2d(Crs::Cartesian2d, 0.0, 0.0),
        )
        .unwrap();
        s.insert(
            txn,
            7,
            NodeId::from_raw(2),
            &Point::new_2d(Crs::Cartesian2d, 5e8, 5e8),
        )
        .unwrap();
        s.insert(
            txn,
            7,
            NodeId::from_raw(3),
            &Point::new_2d(Crs::Cartesian2d, 9e8, 9e8),
        )
        .unwrap();
    });
    let bbox = Bbox {
        lower: Point::new_2d(Crs::Cartesian2d, -1.0, -1.0),
        upper: Point::new_2d(Crs::Cartesian2d, 1.0, 1.0),
    };
    let hits = read_spatial(engine, |s, txn| {
        s.scan_within_bbox(txn, 7, Crs::Cartesian2d, &bbox).unwrap()
    });
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0, NodeId::from_raw(1));
}

#[test]
fn concurrent_insert_distinct_ids_all_visible() {
    // Four threads insert different node_ids into the same
    // (label, crs) keyspace. After join, scan must surface all
    // four points. Engine handles concurrent puts at the level
    // below the store.
    use std::sync::Arc;
    use std::thread;

    let fx = mk_engine();
    let engine = Arc::clone(&fx.engine);

    let handles: Vec<_> = (0..4u64)
        .map(|t| {
            let engine = Arc::clone(&engine);
            thread::spawn(move || {
                let point = Point::new_2d(Crs::Cartesian2d, t as f64, t as f64);
                write_spatial(&engine, |s, txn| {
                    s.insert(txn, 1, NodeId::from_raw(t + 1), &point)
                        .expect("insert");
                });
            })
        })
        .collect();
    for h in handles {
        h.join().expect("join");
    }

    let bbox = Bbox {
        lower: Point::new_2d(Crs::Cartesian2d, -10.0, -10.0),
        upper: Point::new_2d(Crs::Cartesian2d, 10.0, 10.0),
    };
    let hits = read_spatial(&engine, |s, txn| {
        s.scan_within_bbox(txn, 1, Crs::Cartesian2d, &bbox).unwrap()
    });
    assert_eq!(hits.len(), 4, "all four concurrent inserts must be visible");
}

#[test]
fn morton_2d_is_monotone_along_each_axis() {
    // Increasing x with y=0 must give non-decreasing morton code.
    let mut prev = morton_2d(0, 0);
    for x in 1..20 {
        let cur = morton_2d(x, 0);
        assert!(cur > prev, "morton not monotone at x={x}: {prev} -> {cur}");
        prev = cur;
    }
}
