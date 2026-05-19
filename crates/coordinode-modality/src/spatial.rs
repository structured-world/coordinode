//! Spatial store — point indexing for the four supported CRS (WGS-84
//! 2D / 3D, Cartesian 2D / 3D).
//!
//! ## Wire layout
//!
//! Every point lives in [`Partition::Idx`] under a curve-keyed prefix:
//!
//! ```text
//! idx:spatial:<label_id_u32_BE>:<crs_u16_BE>:<curve_u64_BE>:<node_id_u64_BE>
//! → MessagePack { original f64 coordinates }
//! ```
//!
//! The value retains the original `f64` coordinates so the post-filter
//! step (exact Haversine for WGS-84, exact Euclidean for Cartesian)
//! works on the truth, not on the quantised curve position. The curve
//! key is the *index*; the body is the *truth*.
//!
//! ## Curve encoding
//!
//! All four CRS share a single 64-bit curve key. The encoder is
//! pluggable per CRS so the underlying curve is whatever fits the
//! geometry best:
//!
//! | CRS | SRID | Encoding (v1) |
//! |-----|------|---------------|
//! | WGS-84 2D | 4326 | Morton (Z-order), lat/lon quantised to 32 bits each |
//! | WGS-84 3D | 4979 | Morton, lat/lon/alt quantised to 21 bits each |
//! | Cartesian 2D | 7203 | Morton, x/y quantised to 32 bits each |
//! | Cartesian 3D | 9157 | Morton, x/y/z quantised to 21 bits each |
//!
//! Z-order with quantisation is the v1 baseline — it gives correct
//! prefix-equals-region semantics for range scans without depending on
//! an external S2 / Hilbert library. S2-cell encoding for WGS-84 and
//! Hilbert for 2D Cartesian land as a follow-up; the API is unchanged
//! by that swap because callers see only the `u64` curve key plus the
//! post-filter step on the original `f64` body.
//!
//! ## What this store does and doesn't do
//!
//! - **Does:** insert / delete a single point per node, range scan a
//!   curve key window and stream candidate `node_id`s, k-NN nearest by
//!   exact-distance post-filter, bbox containment check.
//! - **Doesn't:** compound spatial+property indexes (that surface lives
//!   in [`IndexStore`](crate::IndexStore)), bitemporal `valid_from`
//!   index keys (ADR-027 follow-up), polygon containment (needs S2
//!   covering), spatial histograms / cardinality estimates (planner
//!   concern). 3D Hilbert and proper S2 cells land as follow-up; the
//!   trait surface is already shaped for them.

use std::cmp::Ordering;

use coordinode_core::graph::node::NodeId;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::Guard;
use serde::{Deserialize, Serialize};

use crate::error::{StoreError, StoreResult};

const SPATIAL_PREFIX: &[u8] = b"idx:spatial:";

/// Coordinate Reference System tag carried in the index key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Crs {
    /// WGS-84 2D — latitude/longitude on the WGS-84 ellipsoid.
    Wgs84_2d = 4326,
    /// WGS-84 3D — adds altitude in metres.
    Wgs84_3d = 4979,
    /// Cartesian 2D — flat plane, Euclidean distance.
    Cartesian2d = 7203,
    /// Cartesian 3D — Euclidean 3-space.
    Cartesian3d = 9157,
}

impl Crs {
    /// SRID code as used in arch docs and the index key encoding.
    pub fn srid(self) -> u16 {
        self as u16
    }

    /// Number of coordinates this CRS carries.
    pub fn dims(self) -> usize {
        match self {
            Self::Wgs84_2d | Self::Cartesian2d => 2,
            Self::Wgs84_3d | Self::Cartesian3d => 3,
        }
    }

    /// True when distances on this CRS are computed via Haversine.
    pub fn is_geo(self) -> bool {
        matches!(self, Self::Wgs84_2d | Self::Wgs84_3d)
    }
}

/// A spatial point: CRS-tagged coordinates kept as `f64` for exact
/// post-filter distance computation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Point {
    /// Reference system the coordinates live in.
    pub crs: Crs,
    /// Coordinate vector: 2 entries (2D CRS) or 3 entries (3D CRS).
    /// For geo CRS: `[longitude, latitude]` or
    /// `[longitude, latitude, altitude]`.
    pub coords: Vec<f64>,
}

impl Point {
    /// Construct a 2D point. Panics in debug on dimension mismatch.
    pub fn new_2d(crs: Crs, c0: f64, c1: f64) -> Self {
        debug_assert_eq!(crs.dims(), 2);
        Self {
            crs,
            coords: vec![c0, c1],
        }
    }

    /// Construct a 3D point. Panics in debug on dimension mismatch.
    pub fn new_3d(crs: Crs, c0: f64, c1: f64, c2: f64) -> Self {
        debug_assert_eq!(crs.dims(), 3);
        Self {
            crs,
            coords: vec![c0, c1, c2],
        }
    }
}

/// Axis-aligned bounding box (inclusive). For 2D CRS, only the first
/// two components are read.
#[derive(Debug, Clone, PartialEq)]
pub struct Bbox {
    /// Lower-left corner.
    pub lower: Point,
    /// Upper-right corner.
    pub upper: Point,
}

/// Layer-4 spatial store contract. Every method is keyed by
/// `(label_id, crs)` — one logical index per `(label, property)` pair
/// in the higher schema layer.
pub trait SpatialStore {
    /// Index a point. Overwrites the existing entry for `node_id`
    /// under the same `(label_id, crs)` if one exists. The old entry's
    /// curve key is computed from the old coordinates by the caller —
    /// at the trait level a write is just a put.
    fn insert(&self, label_id: u32, node_id: NodeId, point: &Point) -> StoreResult<()>;

    /// Remove the entry for `node_id` previously written with `point`.
    /// Idempotent on a missing key.
    fn delete(&self, label_id: u32, node_id: NodeId, point: &Point) -> StoreResult<()>;

    /// Range-scan candidates whose curve key falls inside the
    /// quantised window of `bbox`, then post-filter by exact bbox
    /// containment on the original `f64` coordinates. Returns
    /// `(node_id, point)` pairs in curve-key order.
    fn scan_within_bbox(
        &self,
        label_id: u32,
        crs: Crs,
        bbox: &Bbox,
    ) -> StoreResult<Vec<(NodeId, Point)>>;

    /// k-nearest-neighbour by exact distance from `center`. Scans the
    /// whole `(label_id, crs)` partition and keeps the top-k by exact
    /// distance — fine for trait-level CE; an EE / indexed
    /// implementation can specialise this method.
    fn knn_nearest(
        &self,
        label_id: u32,
        crs: Crs,
        center: &Point,
        k: usize,
    ) -> StoreResult<Vec<(NodeId, Point, f64)>>;
}

/// CE single-shard `SpatialStore` implementation.
pub struct LocalSpatialStore<'a> {
    engine: &'a StorageEngine,
}

impl<'a> LocalSpatialStore<'a> {
    /// Wrap a storage engine for spatial store operations.
    pub fn new(engine: &'a StorageEngine) -> Self {
        Self { engine }
    }
}

fn encode_spatial_prefix(label_id: u32, crs: Crs) -> Vec<u8> {
    let mut out = Vec::with_capacity(SPATIAL_PREFIX.len() + 4 + 2);
    out.extend_from_slice(SPATIAL_PREFIX);
    out.extend_from_slice(&label_id.to_be_bytes());
    out.extend_from_slice(&crs.srid().to_be_bytes());
    out
}

fn encode_spatial_key(label_id: u32, crs: Crs, curve: u64, node_id: NodeId) -> Vec<u8> {
    let mut out = encode_spatial_prefix(label_id, crs);
    out.extend_from_slice(&curve.to_be_bytes());
    out.extend_from_slice(&node_id.as_raw().to_be_bytes());
    out
}

fn encode_body(point: &Point) -> StoreResult<Vec<u8>> {
    rmp_serde::to_vec_named(point).map_err(|e| StoreError::Decode {
        kind: "spatial point",
        message: e.to_string(),
    })
}

fn decode_body(bytes: &[u8]) -> StoreResult<Point> {
    rmp_serde::from_slice(bytes).map_err(|e| StoreError::Decode {
        kind: "spatial point",
        message: e.to_string(),
    })
}

fn decode_node_id_from_key(key: &[u8]) -> Option<NodeId> {
    if key.len() < 8 {
        return None;
    }
    let id_bytes: [u8; 8] = key[key.len() - 8..].try_into().ok()?;
    Some(NodeId::from_raw(u64::from_be_bytes(id_bytes)))
}

/// Quantise an `f64` from `[lo, hi]` to a `u32` linear bucket.
fn quantise_u32(value: f64, lo: f64, hi: f64) -> u32 {
    let clamped = value.clamp(lo, hi);
    let span = (hi - lo).max(f64::MIN_POSITIVE);
    let scaled = ((clamped - lo) / span) * f64::from(u32::MAX);
    scaled.round() as u32
}

/// Quantise an `f64` from `[lo, hi]` to a `u32` bucket truncated to
/// `bits` bits (used for 3D where each component carries only 21
/// bits in the Morton key).
fn quantise_bits(value: f64, lo: f64, hi: f64, bits: u32) -> u32 {
    debug_assert!(bits > 0 && bits <= 32);
    let mask = if bits == 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    };
    let clamped = value.clamp(lo, hi);
    let span = (hi - lo).max(f64::MIN_POSITIVE);
    let scaled = ((clamped - lo) / span) * f64::from(mask);
    (scaled.round() as u32) & mask
}

/// 2D Morton: interleave two u32 streams into a u64 (lowest 64 bits).
fn morton_2d(x: u32, y: u32) -> u64 {
    fn spread(v: u32) -> u64 {
        let mut x = v as u64 & 0x0000_0000_FFFF_FFFF;
        x = (x | (x << 16)) & 0x0000_FFFF_0000_FFFF;
        x = (x | (x << 8)) & 0x00FF_00FF_00FF_00FF;
        x = (x | (x << 4)) & 0x0F0F_0F0F_0F0F_0F0F;
        x = (x | (x << 2)) & 0x3333_3333_3333_3333;
        x = (x | (x << 1)) & 0x5555_5555_5555_5555;
        x
    }
    spread(x) | (spread(y) << 1)
}

/// 3D Morton: interleave three 21-bit streams into a u64.
fn morton_3d(x: u32, y: u32, z: u32) -> u64 {
    fn spread(v: u32) -> u64 {
        let mut x = (v as u64) & 0x1F_FFFF; // 21 bits
        x = (x | (x << 32)) & 0x001F_0000_0000_FFFF;
        x = (x | (x << 16)) & 0x001F_0000_FF00_00FF;
        x = (x | (x << 8)) & 0x100F_00F0_0F00_F00F;
        x = (x | (x << 4)) & 0x10C3_0C30_C30C_30C3;
        x = (x | (x << 2)) & 0x1249_2492_4924_9249;
        x
    }
    spread(x) | (spread(y) << 1) | (spread(z) << 2)
}

/// Encode a 2D or 3D point into a 64-bit curve key per its CRS.
fn encode_curve(point: &Point) -> u64 {
    match point.crs {
        Crs::Wgs84_2d => {
            // lon ∈ [-180, 180], lat ∈ [-90, 90]
            let lon = quantise_u32(point.coords[0], -180.0, 180.0);
            let lat = quantise_u32(point.coords[1], -90.0, 90.0);
            morton_2d(lon, lat)
        }
        Crs::Cartesian2d => {
            let x = quantise_u32(point.coords[0], -1e9, 1e9);
            let y = quantise_u32(point.coords[1], -1e9, 1e9);
            morton_2d(x, y)
        }
        Crs::Wgs84_3d => {
            let lon = quantise_bits(point.coords[0], -180.0, 180.0, 21);
            let lat = quantise_bits(point.coords[1], -90.0, 90.0, 21);
            let alt = quantise_bits(point.coords[2], -11_000.0, 100_000.0, 21);
            morton_3d(lon, lat, alt)
        }
        Crs::Cartesian3d => {
            let x = quantise_bits(point.coords[0], -1e6, 1e6, 21);
            let y = quantise_bits(point.coords[1], -1e6, 1e6, 21);
            let z = quantise_bits(point.coords[2], -1e6, 1e6, 21);
            morton_3d(x, y, z)
        }
    }
}

/// Mean Earth radius in metres (WGS-84 mean).
const EARTH_RADIUS_M: f64 = 6_371_000.0;

/// Haversine distance in metres between two `[lon, lat]` pairs.
fn haversine(lon1: f64, lat1: f64, lon2: f64, lat2: f64) -> f64 {
    let to_rad = std::f64::consts::PI / 180.0;
    let (phi1, phi2) = (lat1 * to_rad, lat2 * to_rad);
    let dphi = (lat2 - lat1) * to_rad;
    let dlam = (lon2 - lon1) * to_rad;
    let a = (dphi / 2.0).sin().powi(2) + phi1.cos() * phi2.cos() * (dlam / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();
    EARTH_RADIUS_M * c
}

/// Exact distance between two points sharing a CRS. Haversine for geo,
/// Euclidean for Cartesian (with the 3D altitude leg added for
/// WGS-84-3D).
pub fn distance(a: &Point, b: &Point) -> f64 {
    debug_assert_eq!(a.crs, b.crs);
    match a.crs {
        Crs::Wgs84_2d => haversine(a.coords[0], a.coords[1], b.coords[0], b.coords[1]),
        Crs::Wgs84_3d => {
            let flat = haversine(a.coords[0], a.coords[1], b.coords[0], b.coords[1]);
            let dalt = a.coords[2] - b.coords[2];
            (flat * flat + dalt * dalt).sqrt()
        }
        Crs::Cartesian2d => {
            let dx = a.coords[0] - b.coords[0];
            let dy = a.coords[1] - b.coords[1];
            (dx * dx + dy * dy).sqrt()
        }
        Crs::Cartesian3d => {
            let dx = a.coords[0] - b.coords[0];
            let dy = a.coords[1] - b.coords[1];
            let dz = a.coords[2] - b.coords[2];
            (dx * dx + dy * dy + dz * dz).sqrt()
        }
    }
}

fn point_in_bbox(point: &Point, bbox: &Bbox) -> bool {
    for i in 0..point.crs.dims() {
        if point.coords[i] < bbox.lower.coords[i] || point.coords[i] > bbox.upper.coords[i] {
            return false;
        }
    }
    true
}

impl SpatialStore for LocalSpatialStore<'_> {
    fn insert(&self, label_id: u32, node_id: NodeId, point: &Point) -> StoreResult<()> {
        let curve = encode_curve(point);
        let key = encode_spatial_key(label_id, point.crs, curve, node_id);
        let body = encode_body(point)?;
        self.engine.put(Partition::Idx, &key, &body)?;
        Ok(())
    }

    fn delete(&self, label_id: u32, node_id: NodeId, point: &Point) -> StoreResult<()> {
        let curve = encode_curve(point);
        let key = encode_spatial_key(label_id, point.crs, curve, node_id);
        self.engine.delete(Partition::Idx, &key)?;
        Ok(())
    }

    fn scan_within_bbox(
        &self,
        label_id: u32,
        crs: Crs,
        bbox: &Bbox,
    ) -> StoreResult<Vec<(NodeId, Point)>> {
        let prefix = encode_spatial_prefix(label_id, crs);
        let mut out = Vec::new();
        let iter = self.engine.prefix_scan(Partition::Idx, &prefix)?;
        for guard in iter {
            let (key, value) = guard.into_inner()?;
            let Some(node_id) = decode_node_id_from_key(&key) else {
                continue;
            };
            let point = decode_body(value.as_ref())?;
            if point_in_bbox(&point, bbox) {
                out.push((node_id, point));
            }
        }
        Ok(out)
    }

    fn knn_nearest(
        &self,
        label_id: u32,
        crs: Crs,
        center: &Point,
        k: usize,
    ) -> StoreResult<Vec<(NodeId, Point, f64)>> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let prefix = encode_spatial_prefix(label_id, crs);
        let mut candidates: Vec<(NodeId, Point, f64)> = Vec::new();
        let iter = self.engine.prefix_scan(Partition::Idx, &prefix)?;
        for guard in iter {
            let (key, value) = guard.into_inner()?;
            let Some(node_id) = decode_node_id_from_key(&key) else {
                continue;
            };
            let point = decode_body(value.as_ref())?;
            let d = distance(center, &point);
            candidates.push((node_id, point, d));
        }
        candidates.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));
        candidates.truncate(k);
        Ok(candidates)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use coordinode_storage::engine::config::{
        Durability, EndpointConfig, Media, StorageConfig, Tier,
    };
    use tempfile::TempDir;

    fn mk_engine() -> (TempDir, StorageEngine) {
        let dir = TempDir::new().unwrap();
        let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "ep",
            dir.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        let engine = StorageEngine::open(&config).unwrap();
        (dir, engine)
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
        let (_dir, engine) = mk_engine();
        let store = LocalSpatialStore::new(&engine);
        let paris = Point::new_2d(Crs::Wgs84_2d, 2.3522, 48.8566);
        store.insert(1, NodeId::from_raw(1), &paris).unwrap();
        let bbox = Bbox {
            lower: Point::new_2d(Crs::Wgs84_2d, 2.0, 48.0),
            upper: Point::new_2d(Crs::Wgs84_2d, 3.0, 49.0),
        };
        let hits = store.scan_within_bbox(1, Crs::Wgs84_2d, &bbox).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, NodeId::from_raw(1));
        assert!((hits[0].1.coords[0] - paris.coords[0]).abs() < 1e-6);
    }

    #[test]
    fn scan_bbox_filters_outside_points() {
        let (_dir, engine) = mk_engine();
        let store = LocalSpatialStore::new(&engine);
        let paris = Point::new_2d(Crs::Wgs84_2d, 2.3522, 48.8566);
        let kyiv = Point::new_2d(Crs::Wgs84_2d, 30.5234, 50.4501);
        store.insert(1, NodeId::from_raw(1), &paris).unwrap();
        store.insert(1, NodeId::from_raw(2), &kyiv).unwrap();
        let paris_bbox = Bbox {
            lower: Point::new_2d(Crs::Wgs84_2d, 1.5, 48.0),
            upper: Point::new_2d(Crs::Wgs84_2d, 3.0, 49.5),
        };
        let hits = store
            .scan_within_bbox(1, Crs::Wgs84_2d, &paris_bbox)
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, NodeId::from_raw(1));
    }

    #[test]
    fn delete_removes_point() {
        let (_dir, engine) = mk_engine();
        let store = LocalSpatialStore::new(&engine);
        let p = Point::new_2d(Crs::Cartesian2d, 10.0, 20.0);
        store.insert(7, NodeId::from_raw(42), &p).unwrap();
        store.delete(7, NodeId::from_raw(42), &p).unwrap();
        let bbox = Bbox {
            lower: Point::new_2d(Crs::Cartesian2d, 0.0, 0.0),
            upper: Point::new_2d(Crs::Cartesian2d, 100.0, 100.0),
        };
        let hits = store.scan_within_bbox(7, Crs::Cartesian2d, &bbox).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn knn_returns_closest_in_distance_order() {
        let (_dir, engine) = mk_engine();
        let store = LocalSpatialStore::new(&engine);
        let pts = [
            (1u64, 0.0, 0.0),
            (2, 1.0, 0.0),
            (3, 5.0, 5.0),
            (4, 10.0, 10.0),
        ];
        for (id, x, y) in pts {
            store
                .insert(
                    3,
                    NodeId::from_raw(id),
                    &Point::new_2d(Crs::Cartesian2d, x, y),
                )
                .unwrap();
        }
        let center = Point::new_2d(Crs::Cartesian2d, 0.0, 0.0);
        let knn = store.knn_nearest(3, Crs::Cartesian2d, &center, 2).unwrap();
        assert_eq!(knn.len(), 2);
        assert_eq!(knn[0].0, NodeId::from_raw(1));
        assert_eq!(knn[1].0, NodeId::from_raw(2));
        assert!(knn[0].2 < knn[1].2);
    }

    #[test]
    fn knn_k_zero_returns_empty() {
        let (_dir, engine) = mk_engine();
        let store = LocalSpatialStore::new(&engine);
        store
            .insert(
                1,
                NodeId::from_raw(1),
                &Point::new_2d(Crs::Cartesian2d, 0.0, 0.0),
            )
            .unwrap();
        let center = Point::new_2d(Crs::Cartesian2d, 0.0, 0.0);
        let knn = store.knn_nearest(1, Crs::Cartesian2d, &center, 0).unwrap();
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
        let (_dir, engine) = mk_engine();
        let store = LocalSpatialStore::new(&engine);
        let origin = Point::new_3d(Crs::Cartesian3d, 0.0, 0.0, 0.0);
        let near = Point::new_3d(Crs::Cartesian3d, 1.0, 1.0, 1.0);
        let far = Point::new_3d(Crs::Cartesian3d, 100.0, 100.0, 100.0);
        store.insert(9, NodeId::from_raw(1), &origin).unwrap();
        store.insert(9, NodeId::from_raw(2), &near).unwrap();
        store.insert(9, NodeId::from_raw(3), &far).unwrap();
        let knn = store.knn_nearest(9, Crs::Cartesian3d, &origin, 2).unwrap();
        assert_eq!(knn.len(), 2);
        assert_eq!(knn[0].0, NodeId::from_raw(1));
        assert_eq!(knn[1].0, NodeId::from_raw(2));
    }

    #[test]
    fn scoped_by_label_id() {
        let (_dir, engine) = mk_engine();
        let store = LocalSpatialStore::new(&engine);
        let p = Point::new_2d(Crs::Cartesian2d, 1.0, 1.0);
        store.insert(1, NodeId::from_raw(10), &p).unwrap();
        store.insert(2, NodeId::from_raw(20), &p).unwrap();
        let bbox = Bbox {
            lower: Point::new_2d(Crs::Cartesian2d, 0.0, 0.0),
            upper: Point::new_2d(Crs::Cartesian2d, 10.0, 10.0),
        };
        let hits1 = store.scan_within_bbox(1, Crs::Cartesian2d, &bbox).unwrap();
        let hits2 = store.scan_within_bbox(2, Crs::Cartesian2d, &bbox).unwrap();
        assert_eq!(hits1.len(), 1);
        assert_eq!(hits2.len(), 1);
        assert_eq!(hits1[0].0, NodeId::from_raw(10));
        assert_eq!(hits2[0].0, NodeId::from_raw(20));
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
}
