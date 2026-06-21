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
//! Z-order with quantisation is the v1 baseline. It is correct (every
//! point inside the bbox has a curve key inside `[morton_min,
//! morton_max]`) and the bbox scan terminates the iterator early once
//! the curve crosses `morton_max`. The Morton curve traces a Z-shape,
//! so the linear interval `[morton_min, morton_max]` is *over*-
//! inclusive — it can contain cells outside the bbox; the exact
//! `point_in_bbox` post-filter on the original `f64` body discards
//! those. Skipping the Z-shape's false-positive sub-intervals
//! (litmax/bigmin) is a follow-up — for v1 the saving is the early
//! break, not subrange skipping. S2-cell encoding for WGS-84 and
//! Hilbert for 2D Cartesian also land as follow-ups; the API surface
//! is unchanged by either swap because callers only see the `u64`
//! curve key + exact post-filter step.
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
//!
//! ## Transaction threading (ADR-041)
//!
//! Spatial point entries are a secondary index in [`Partition::Idx`].
//! Writes (`insert` / `delete`) take `&mut Transaction` and buffer the
//! index-row mutation on it, so the spatial entry commits atomically
//! with the node write that produced the point. Reads
//! (`scan_within_bbox` / `knn_nearest`) take `&Transaction` and walk
//! the committed MVCC snapshot via
//! [`Transaction::base_prefix_scan`] — untracked, so a candidate scan
//! never balloons the OCC read set over an entire label's points. The
//! Layer-3 transaction read path exposes only prefix scans (no
//! engine-level byte-range scan), so the bbox scan walks the
//! `(label_id, crs)` prefix once and applies the Z-curve interval
//! filter in memory.
//!
//! [`Transaction`]: coordinode_storage::engine::transaction::Transaction
//! [`Transaction::base_prefix_scan`]: coordinode_storage::engine::transaction::Transaction::base_prefix_scan

use std::cmp::Ordering;

use coordinode_core::graph::node::NodeId;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::transaction::Transaction;
use coordinode_storage::Guard; // IterGuard trait — `guard.into_inner()` on seekable scans
use s2::cellid::CellID;
use s2::latlng::LatLng;
use s2::rect::Rect;
use s2::region::RegionCoverer;
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
///
/// ## Transaction threading (ADR-041)
///
/// Writes ([`Self::insert`] / [`Self::delete`]) take `&mut Transaction`
/// and buffer the point-index `Partition::Idx` mutation on it, so the
/// spatial entry commits atomically with the node write that produced
/// the point. Reads ([`Self::scan_within_bbox`] / [`Self::knn_nearest`])
/// take `&Transaction` and walk the committed MVCC snapshot via
/// [`Transaction::base_prefix_scan`] — untracked (a candidate scan must
/// not balloon the OCC read set over an entire label's points).
pub trait SpatialStore {
    /// Index a point. Buffers the `(label_id, crs, curve, node_id)`
    /// row on `txn`.
    ///
    /// **Contract gotcha:** writing the *same* `node_id` with
    /// *different* coordinates lands in a *different* physical row
    /// (the curve is part of the key). The old row is NOT
    /// garbage-collected by this call — callers updating a moving
    /// point MUST [`Self::delete`] the OLD coordinates first.
    /// Inserting the same `(node_id, coords)` pair twice IS idempotent.
    fn insert(
        &self,
        txn: &mut Transaction,
        label_id: u32,
        node_id: NodeId,
        point: &Point,
    ) -> StoreResult<()>;

    /// Remove the entry for `node_id` previously written with `point`.
    /// Idempotent on a missing key.
    fn delete(
        &self,
        txn: &mut Transaction,
        label_id: u32,
        node_id: NodeId,
        point: &Point,
    ) -> StoreResult<()>;

    /// Range-scan candidates whose curve key falls inside the
    /// quantised window of `bbox`, then post-filter by exact bbox
    /// containment on the original `f64` coordinates. Returns
    /// `(node_id, point)` pairs in curve-key order.
    fn scan_within_bbox(
        &self,
        txn: &Transaction,
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
        txn: &Transaction,
        label_id: u32,
        crs: Crs,
        center: &Point,
        k: usize,
    ) -> StoreResult<Vec<(NodeId, Point, f64)>>;
}

/// CE single-shard `SpatialStore` implementation. Stateless — all
/// storage access flows through the [`Transaction`] passed to each
/// method (ADR-041).
pub struct LocalSpatialStore;

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

/// Decompose a 2D quantised bbox into a list of disjoint Morton key
/// intervals (sorted by `lo`, contiguous-merged). Each returned
/// `(lo, hi)` represents either:
///   - A cell fully inside the bbox (precise — every key in the
///     interval corresponds to an in-bbox `(x, y)`), or
///   - A cell-bbox intersection at the depth-cap fallback boundary
///     (over-inclusive — post-filter rejects Z-shape false positives
///     within the rim cell).
///
/// Algorithm: recursive power-of-2 quadrant subdivision starting
/// from the full `[0, 2^32)` space (Tropf-Herzog 1981 family).
///
/// **Hard caps** prevent runaway output on adversarial bbox shapes:
/// - `DEPTH_CAP` bounds recursion depth. At max depth the cell is
///   emitted as one broad interval covering the cell-bbox
///   intersection. Worst-case output is bounded by `4^DEPTH_CAP`
///   but typically much smaller (alive cells track bbox perimeter).
/// - `OUTPUT_CAP` is an additional hard limit. Once `out.len()`
///   reaches the cap, recursion short-circuits with a single broad
///   fallback interval at the current cell.
///
/// **Cell merging** post-process: contiguous intervals
/// (`hi[i] + 1 == lo[i+1]`) merge into one. Reduces the per-
/// range_scan setup cost when neighbour cells happen to be Morton-
/// adjacent (common when 4 quadrant children all emit inside).
fn morton_2d_decompose(bx_min: u32, bx_max: u32, by_min: u32, by_max: u32) -> Vec<(u64, u64)> {
    const DEPTH_CAP: u32 = 8; // 2^(32-8) = 16M-per-dim leaf cells
    const OUTPUT_CAP: usize = 256;
    let mut raw: Vec<(u64, u64)> = Vec::new();
    morton_2d_decompose_rec(
        0,
        0,
        1u64 << 32,
        u64::from(bx_min),
        u64::from(bx_max),
        u64::from(by_min),
        u64::from(by_max),
        &mut raw,
        0,
        DEPTH_CAP,
        OUTPUT_CAP,
    );
    // Sort by `lo` and merge contiguous intervals. Recursion emits
    // in Morton-z-order already, so this is mostly a stable confirm
    // + merge pass.
    raw.sort_unstable_by_key(|&(lo, _)| lo);
    let mut merged: Vec<(u64, u64)> = Vec::with_capacity(raw.len());
    for (lo, hi) in raw {
        if let Some(last) = merged.last_mut() {
            if last.1.saturating_add(1) >= lo {
                last.1 = last.1.max(hi);
                continue;
            }
        }
        merged.push((lo, hi));
    }
    merged
}

#[allow(clippy::too_many_arguments)]
fn morton_2d_decompose_rec(
    cx_min: u64,
    cy_min: u64,
    size: u64,
    bx_min: u64,
    bx_max: u64,
    by_min: u64,
    by_max: u64,
    out: &mut Vec<(u64, u64)>,
    depth: u32,
    depth_cap: u32,
    output_cap: usize,
) {
    let cx_max = cx_min + size - 1;
    let cy_max = cy_min + size - 1;

    // Cell entirely outside bbox: skip — this is the win vs the
    // single broad `[morton_min, morton_max]` scan.
    if cx_max < bx_min || cx_min > bx_max || cy_max < by_min || cy_min > by_max {
        return;
    }

    // Cell entirely inside bbox: emit as one interval (precise).
    if cx_min >= bx_min && cx_max <= bx_max && cy_min >= by_min && cy_max <= by_max {
        let lo = morton_2d(cx_min as u32, cy_min as u32);
        let hi = morton_2d(cx_max as u32, cy_max as u32);
        out.push((lo, hi));
        return;
    }

    // Hard caps: at max depth or output cap, emit one broad interval
    // covering the cell-bbox intersection. Post-filter handles
    // intra-cell Z-shape false positives.
    if depth >= depth_cap || size == 1 || out.len() >= output_cap {
        let lo_x = cx_min.max(bx_min) as u32;
        let lo_y = cy_min.max(by_min) as u32;
        let hi_x = cx_max.min(bx_max) as u32;
        let hi_y = cy_max.min(by_max) as u32;
        let lo = morton_2d(lo_x, lo_y);
        let hi = morton_2d(hi_x, hi_y);
        out.push((lo.min(hi), lo.max(hi)));
        return;
    }

    let half = size / 2;
    // Morton z-order: 4 children in (x_high_bit, y_high_bit) order.
    // Recursing in this order keeps `out` approximately sorted by lo.
    let next_depth = depth + 1;
    for (dx, dy) in [(0, 0), (half, 0), (0, half), (half, half)] {
        morton_2d_decompose_rec(
            cx_min + dx,
            cy_min + dy,
            half,
            bx_min,
            bx_max,
            by_min,
            by_max,
            out,
            next_depth,
            depth_cap,
            output_cap,
        );
    }
}

/// Resolve a CRS-and-bbox into the list of (lo, hi) Morton intervals
/// to scan. 2D paths use [`morton_2d_decompose`] with an **adaptive
/// bailout**; 3D paths still return the single broad `[morton_min,
/// morton_max]` interval (8-way octree generalisation is a follow-up
/// since 3D usage is much rarer in practice).
///
/// ## Adaptive bailout
///
/// Decomposition is a NET WIN only when the broad `[morton_min,
/// morton_max]` interval contains substantially more curve range
/// than the decomposed total. The multi-interval path drives one seekable
/// iterator and `seek_to`s each interval (a cheap in-place reposition), so
/// decomposition wins once the broad span is a modest multiple of the
/// decomposed total; near-square bboxes (ratio ≈ 1) stay on the broad scan.
///
/// Rule: use decomposition iff
///   `broad_span / decomposed_total_span >= GAIN_THRESHOLD`
/// AND `decomposed.len() >= 2` (otherwise broad is trivially equal).
/// The equatorial-band scenario (full lon, ±0.5° lat) has ratio = 3 (a clear
/// win — 85% faster at 100k); square / tight scenarios have ratio ≈ 1 (broad
/// wins, no regression).
fn morton_intervals(bbox: &Bbox) -> Vec<(u64, u64)> {
    // **Threshold (seekable skip-scan):** the multi-interval read path drives
    // ONE seekable iterator (`base_range_seekable` → lsm-tree `range_seekable`)
    // and `seek_to`s each interval in place — an SST cursor reposition, not a
    // per-interval iterator reopen. With that cheap jump, decomposition pays off
    // even when the broad curve-span is only a small multiple of the useful
    // (decomposed) span: the dead-zone bytes between intervals are skipped
    // instead of scanned-then-filtered. Bench-tuned to 2: the equatorial band
    // (full lon, ±0.5° lat) has broad/decomposed ratio = 3 — Z-interleaving makes
    // a narrow-lat full-lon strip occupy ~1/3 of the curve, not 0.55% — and
    // decomposes into ~257 intervals; the seekable path runs it 85% faster than
    // the broad scan (44.9ms → 7.0ms at 100k; threshold 4 reverts it to broad,
    // +377%). The band sits exactly at ratio 3, so the threshold is 2 (not 3) for
    // margin: a slightly denser strip that quantises to ratio 2 still decomposes,
    // and reseeks are cheap enough (257 of them at ratio 3 still win 85%) that a
    // ratio-2 shape wins too. Near-square / tight bboxes have ratio ≈ 1, stay
    // below the threshold, and keep the broad prefix scan + early-break (no
    // regression — within noise on every square/tight bench).
    const GAIN_THRESHOLD: u64 = 2;

    let (x_min, x_max, y_min, y_max) = match bbox.lower.crs {
        Crs::Wgs84_2d => (
            quantise_u32(bbox.lower.coords[0], -180.0, 180.0),
            quantise_u32(bbox.upper.coords[0], -180.0, 180.0),
            quantise_u32(bbox.lower.coords[1], -90.0, 90.0),
            quantise_u32(bbox.upper.coords[1], -90.0, 90.0),
        ),
        Crs::Cartesian2d => (
            quantise_u32(bbox.lower.coords[0], -1e9, 1e9),
            quantise_u32(bbox.upper.coords[0], -1e9, 1e9),
            quantise_u32(bbox.lower.coords[1], -1e9, 1e9),
            quantise_u32(bbox.upper.coords[1], -1e9, 1e9),
        ),
        Crs::Wgs84_3d | Crs::Cartesian3d => {
            // 3D follow-up — fall back to the broad window for now.
            let (lo, hi) = morton_window(bbox);
            return vec![(lo, hi)];
        }
    };

    let decomposed = morton_2d_decompose(x_min, x_max, y_min, y_max);
    let broad_lo = morton_2d(x_min, y_min);
    let broad_hi = morton_2d(x_max, y_max);

    if decomposed.len() < 2 {
        return vec![(broad_lo, broad_hi)];
    }

    let broad_span = broad_hi.saturating_sub(broad_lo).saturating_add(1);
    let decomposed_span: u64 = decomposed
        .iter()
        .map(|(lo, hi)| hi.saturating_sub(*lo).saturating_add(1))
        .fold(0u64, u64::saturating_add);

    if decomposed_span == 0 || broad_span / decomposed_span.max(1) < GAIN_THRESHOLD {
        return vec![(broad_lo, broad_hi)];
    }
    decomposed
}

/// Morton bounding range for a 2D bbox. The returned `[min, max]` is
/// the lexicographic interval that contains every cell of the bbox —
/// but also contains cells *outside* the bbox (Morton's Z-shape).
/// Callers MUST post-filter with `point_in_bbox`.
fn morton_window(bbox: &Bbox) -> (u64, u64) {
    match bbox.lower.crs {
        Crs::Wgs84_2d => {
            let lon_min = quantise_u32(bbox.lower.coords[0], -180.0, 180.0);
            let lat_min = quantise_u32(bbox.lower.coords[1], -90.0, 90.0);
            let lon_max = quantise_u32(bbox.upper.coords[0], -180.0, 180.0);
            let lat_max = quantise_u32(bbox.upper.coords[1], -90.0, 90.0);
            (morton_2d(lon_min, lat_min), morton_2d(lon_max, lat_max))
        }
        Crs::Cartesian2d => {
            let x_min = quantise_u32(bbox.lower.coords[0], -1e9, 1e9);
            let y_min = quantise_u32(bbox.lower.coords[1], -1e9, 1e9);
            let x_max = quantise_u32(bbox.upper.coords[0], -1e9, 1e9);
            let y_max = quantise_u32(bbox.upper.coords[1], -1e9, 1e9);
            (morton_2d(x_min, y_min), morton_2d(x_max, y_max))
        }
        Crs::Wgs84_3d => {
            let lon_min = quantise_bits(bbox.lower.coords[0], -180.0, 180.0, 21);
            let lat_min = quantise_bits(bbox.lower.coords[1], -90.0, 90.0, 21);
            let alt_min = quantise_bits(bbox.lower.coords[2], -11_000.0, 100_000.0, 21);
            let lon_max = quantise_bits(bbox.upper.coords[0], -180.0, 180.0, 21);
            let lat_max = quantise_bits(bbox.upper.coords[1], -90.0, 90.0, 21);
            let alt_max = quantise_bits(bbox.upper.coords[2], -11_000.0, 100_000.0, 21);
            (
                morton_3d(lon_min, lat_min, alt_min),
                morton_3d(lon_max, lat_max, alt_max),
            )
        }
        Crs::Cartesian3d => {
            let x_min = quantise_bits(bbox.lower.coords[0], -1e6, 1e6, 21);
            let y_min = quantise_bits(bbox.lower.coords[1], -1e6, 1e6, 21);
            let z_min = quantise_bits(bbox.lower.coords[2], -1e6, 1e6, 21);
            let x_max = quantise_bits(bbox.upper.coords[0], -1e6, 1e6, 21);
            let y_max = quantise_bits(bbox.upper.coords[1], -1e6, 1e6, 21);
            let z_max = quantise_bits(bbox.upper.coords[2], -1e6, 1e6, 21);
            (
                morton_3d(x_min, y_min, z_min),
                morton_3d(x_max, y_max, z_max),
            )
        }
    }
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

/// 2D Hilbert curve index for a 32-bit-per-dimension grid (side `2^32`),
/// yielding a full `u64` key. Standard `xy2d` algorithm (Wikipedia).
///
/// Unlike Morton/Z-order, the Hilbert curve has no long "Z" jumps: consecutive
/// curve positions are always grid-adjacent, so a bbox maps to fewer, tighter
/// runs and proximate points cluster better in the LSM key order. The flip side
/// (vs Morton) is that the index is NOT monotone in either coordinate, so a bbox
/// scan cannot use a single `[corner_lo, corner_hi]` window + early-break — it
/// needs a recursive interval cover (see `hilbert_2d_decompose`).
fn hilbert_2d(x: u32, y: u32) -> u64 {
    hilbert_2d_bits(x, y, 32)
}

/// Hilbert index of `(x, y)` on a `2^bits × 2^bits` grid (`bits` in `1..=32`),
/// generalising [`hilbert_2d`] (the `bits = 32` case). Used to compute the
/// contiguous Hilbert range of an aligned quad at an arbitrary level. `bits == 0`
/// is the degenerate single-cell grid (index 0).
fn hilbert_2d_bits(x: u32, y: u32, bits: u32) -> u64 {
    if bits == 0 {
        return 0;
    }
    // Grid side N = 2^bits; rotation reflects within [0, N). Bit budget:
    // s*s ≤ 2^62, *3 < 2^64; accumulated total is a valid index in [0, 4^bits).
    let n_minus_1: u64 = (1u64 << bits) - 1;
    let mut x = u64::from(x);
    let mut y = u64::from(y);
    let mut d: u64 = 0;
    let mut s: u64 = 1u64 << (bits - 1);
    while s > 0 {
        let rx = u64::from((x & s) > 0);
        let ry = u64::from((y & s) > 0);
        d += s * s * ((3 * rx) ^ ry);
        // rotate/reflect the quadrant so children keep curve continuity
        if ry == 0 {
            if rx == 1 {
                x = n_minus_1 - x;
                y = n_minus_1 - y;
            }
            std::mem::swap(&mut x, &mut y);
        }
        s >>= 1;
    }
    d
}

/// Inverse of [`hilbert_2d`]: map a Hilbert index back to its `(x, y)` grid
/// coordinates. Test-only — verifies the encoder (round-trip + the curve's
/// grid-adjacency property); the production cover needs only the forward map.
#[cfg(test)]
fn hilbert_2d_to_xy(d: u64) -> (u32, u32) {
    let mut x: u64 = 0;
    let mut y: u64 = 0;
    let mut t = d;
    let mut s: u64 = 1;
    while s < (1 << 32) {
        let rx = 1 & (t / 2);
        let ry = 1 & (t ^ rx);
        // inverse rotation (uses the current sub-grid size s; x, y < s here)
        if ry == 0 {
            if rx == 1 {
                x = s - 1 - x;
                y = s - 1 - y;
            }
            std::mem::swap(&mut x, &mut y);
        }
        x += s * rx;
        y += s * ry;
        t /= 4;
        s <<= 1;
    }
    (x as u32, y as u32)
}

/// Contiguous Hilbert index range `[lo, hi]` of an aligned `size × size` quad
/// (`size = 2^k`) anchored at `(cx_min, cy_min)`. The Hilbert curve visits every
/// quadtree node contiguously, so the quad occupies exactly `[prefix·4^k,
/// prefix·4^k + 4^k − 1]`, where `prefix` is the quad's Hilbert index on the
/// coarse `(32−k)`-bit grid. (Rotation only reorders cells *within* the range.)
fn hilbert_quad_range(cx_min: u64, cy_min: u64, size: u64) -> (u64, u64) {
    let k = size.trailing_zeros();
    if k >= 32 {
        return (0, u64::MAX); // whole space
    }
    let coarse_bits = 32 - k;
    let prefix = hilbert_2d_bits((cx_min >> k) as u32, (cy_min >> k) as u32, coarse_bits);
    let span = 1u64 << (2 * k); // 4^k cells
    let lo = prefix << (2 * k);
    // `lo + (span - 1)`, not `lo + span - 1`: the top quad (prefix=3, k=31) has
    // lo + span == 2^64, so the parenthesised form avoids the add overflow.
    (lo, lo + (span - 1))
}

/// Decompose a bbox (in quantised `u32` grid coords) into a minimal set of
/// Hilbert index intervals whose union covers every bbox cell. Mirrors
/// [`morton_2d_decompose`]'s quadtree recursion, but emits each aligned quad's
/// contiguous Hilbert range.
///
/// Unlike Morton, Hilbert has no order-preserving `[corner_lo, corner_hi]`
/// broad window (it is not coordinate-monotone), so the multi-interval cover is
/// the ONLY correct scan shape — there is no broad-window fast path. Boundary
/// quads that can't be resolved within the caps emit their full quad range as a
/// SUPERSET; the exact `point_in_bbox` post-filter removes the false positives.
fn hilbert_2d_decompose(bx_min: u32, bx_max: u32, by_min: u32, by_max: u32) -> Vec<(u64, u64)> {
    // Hilbert boundary leaves emit the WHOLE quad (not a tight intersection, which
    // isn't a contiguous Hilbert range), so resolve boundaries finely (deep cap)
    // and bound the count with the output cap. The G101 seekable skip-scan makes
    // many intervals cheap (in-place reseek), so a generous interval budget pays.
    // Recursion work is O(bbox perimeter), not O(depth) — only boundary quads
    // recurse — so a deep cap buys tight boundary leaves for small bboxes; the
    // output cap bounds the interval count (coarser supersets past it). Both are
    // bench-tunable (no Cartesian-2D bench yet; correctness holds for any value).
    const DEPTH_CAP: u32 = 30;
    const OUTPUT_CAP: usize = 256;
    let mut raw: Vec<(u64, u64)> = Vec::new();
    hilbert_2d_decompose_rec(
        0,
        0,
        1u64 << 32,
        u64::from(bx_min),
        u64::from(bx_max),
        u64::from(by_min),
        u64::from(by_max),
        &mut raw,
        0,
        DEPTH_CAP,
        OUTPUT_CAP,
    );
    raw.sort_unstable_by_key(|&(lo, _)| lo);
    let mut merged: Vec<(u64, u64)> = Vec::with_capacity(raw.len());
    for (lo, hi) in raw {
        if let Some(last) = merged.last_mut() {
            if last.1.saturating_add(1) >= lo {
                last.1 = last.1.max(hi);
                continue;
            }
        }
        merged.push((lo, hi));
    }
    merged
}

#[allow(clippy::too_many_arguments)]
fn hilbert_2d_decompose_rec(
    cx_min: u64,
    cy_min: u64,
    size: u64,
    bx_min: u64,
    bx_max: u64,
    by_min: u64,
    by_max: u64,
    out: &mut Vec<(u64, u64)>,
    depth: u32,
    depth_cap: u32,
    output_cap: usize,
) {
    let cx_max = cx_min + size - 1;
    let cy_max = cy_min + size - 1;

    // Quad entirely outside bbox: skip (the dead-zone win).
    if cx_max < bx_min || cx_min > bx_max || cy_max < by_min || cy_min > by_max {
        return;
    }

    // Quad entirely inside bbox: emit its exact contiguous Hilbert range.
    if cx_min >= bx_min && cx_max <= bx_max && cy_min >= by_min && cy_max <= by_max {
        out.push(hilbert_quad_range(cx_min, cy_min, size));
        return;
    }

    // Boundary quad at a cap: emit the WHOLE quad range as a superset (an
    // intersection sub-rect is not a contiguous Hilbert range). point_in_bbox
    // post-filters the false positives.
    if depth >= depth_cap || size == 1 || out.len() >= output_cap {
        out.push(hilbert_quad_range(cx_min, cy_min, size));
        return;
    }

    let half = size / 2;
    let next_depth = depth + 1;
    for (dx, dy) in [(0, 0), (half, 0), (0, half), (half, half)] {
        hilbert_2d_decompose_rec(
            cx_min + dx,
            cy_min + dy,
            half,
            bx_min,
            bx_max,
            by_min,
            by_max,
            out,
            next_depth,
            depth_cap,
            output_cap,
        );
    }
}

/// Resolve a bbox into the set of curve index intervals to scan, dispatching on
/// the CRS's space-filling curve: Cartesian-2D uses the Hilbert cover (always a
/// multi-interval cover — Hilbert has no order-preserving broad window), every
/// other CRS uses the Morton cover with its adaptive broad-vs-decompose bailout.
fn curve_intervals(bbox: &Bbox) -> Vec<(u64, u64)> {
    match bbox.lower.crs {
        Crs::Wgs84_2d => wgs84_s2_intervals(bbox),
        Crs::Wgs84_3d => wgs84_3d_s2_intervals(bbox),
        Crs::Cartesian2d => {
            let x_min = quantise_u32(bbox.lower.coords[0], -1e9, 1e9);
            let y_min = quantise_u32(bbox.lower.coords[1], -1e9, 1e9);
            let x_max = quantise_u32(bbox.upper.coords[0], -1e9, 1e9);
            let y_max = quantise_u32(bbox.upper.coords[1], -1e9, 1e9);
            hilbert_2d_decompose(x_min, x_max, y_min, y_max)
        }
        _ => morton_intervals(bbox),
    }
}

/// WGS-84-3D packs the horizontal S2 cell into the high bits of the u64 curve
/// key and the quantised altitude into the low [`WGS84_3D_ALT_BITS`] bits, so a
/// single u64 carries both axes (the curve-key invariant). The split trades the
/// bottom S2 levels (~5m horizontal) for ~7m vertical resolution over the alt
/// range — comparable to the prior 21-bit-per-axis Morton, but pole-correct.
const WGS84_3D_ALT_BITS: u32 = 14;
const WGS84_3D_ALT_MASK: u64 = (1 << WGS84_3D_ALT_BITS) - 1;
const WGS84_3D_ALT_LO: f64 = -11_000.0;
const WGS84_3D_ALT_HI: f64 = 100_000.0;

/// S2 covering of a WGS-84-3D bbox → packed (horizontal | altitude) key
/// intervals. Covers the lat/lng rect horizontally with S2 cells (as in
/// [`wgs84_s2_intervals`]); for each cell the interval spans the cell's
/// horizontal key range OR'd with the altitude range `[amin, amax]`. The
/// interval is a superset (intermediate horizontals carry the full alt range);
/// the exact 3D `point_in_bbox` post-filter drops the vertical overhang.
fn wgs84_3d_s2_intervals(bbox: &Bbox) -> Vec<(u64, u64)> {
    // coords: [0] = lon, [1] = lat, [2] = alt (metres).
    let rect = Rect::from_degrees(
        bbox.lower.coords[1],
        bbox.lower.coords[0],
        bbox.upper.coords[1],
        bbox.upper.coords[0],
    );
    let amin = u64::from(quantise_bits(
        bbox.lower.coords[2],
        WGS84_3D_ALT_LO,
        WGS84_3D_ALT_HI,
        WGS84_3D_ALT_BITS,
    ));
    let amax = u64::from(quantise_bits(
        bbox.upper.coords[2],
        WGS84_3D_ALT_LO,
        WGS84_3D_ALT_HI,
        WGS84_3D_ALT_BITS,
    ));
    let coverer = RegionCoverer {
        min_level: 0,
        max_level: 24,
        level_mod: 1,
        max_cells: 16,
    };
    let mut intervals: Vec<(u64, u64)> = coverer
        .covering(&rect)
        .0
        .iter()
        .map(|c| {
            let hmin = c.range_min().0 & !WGS84_3D_ALT_MASK;
            let hmax = c.range_max().0 & !WGS84_3D_ALT_MASK;
            (hmin | amin, hmax | amax)
        })
        .collect();
    intervals.sort_unstable_by_key(|&(lo, _)| lo);
    let mut merged: Vec<(u64, u64)> = Vec::with_capacity(intervals.len());
    for (lo, hi) in intervals {
        if let Some(last) = merged.last_mut() {
            if last.1.saturating_add(1) >= lo {
                last.1 = last.1.max(hi);
                continue;
            }
        }
        merged.push((lo, hi));
    }
    merged
}

/// S2 cell covering of a WGS-84 bbox → leaf-cell-id intervals. The `s2` crate's
/// `RegionCoverer` approximates the lat/lng rectangle with a handful of cells
/// (typically 4-16); each cell's `[range_min, range_max]` is the contiguous
/// leaf-id range of its descendants, so the union is a superset of the bbox's
/// leaf ids. The exact `point_in_bbox` post-filter removes the cells' overhang.
/// No pole / antimeridian distortion — the sphere is covered natively.
fn wgs84_s2_intervals(bbox: &Bbox) -> Vec<(u64, u64)> {
    // coords: [0] = lon, [1] = lat. Rect::from_degrees(lat_lo, lng_lo, lat_hi, lng_hi).
    let rect = Rect::from_degrees(
        bbox.lower.coords[1],
        bbox.lower.coords[0],
        bbox.upper.coords[1],
        bbox.upper.coords[0],
    );
    let coverer = RegionCoverer {
        min_level: 0,
        max_level: 30,
        level_mod: 1,
        max_cells: 16,
    };
    let mut intervals: Vec<(u64, u64)> = coverer
        .covering(&rect)
        .0
        .iter()
        .map(|c| (c.range_min().0, c.range_max().0))
        .collect();
    intervals.sort_unstable_by_key(|&(lo, _)| lo);
    let mut merged: Vec<(u64, u64)> = Vec::with_capacity(intervals.len());
    for (lo, hi) in intervals {
        if let Some(last) = merged.last_mut() {
            if last.1.saturating_add(1) >= lo {
                last.1 = last.1.max(hi);
                continue;
            }
        }
        merged.push((lo, hi));
    }
    merged
}

/// Encode a 2D or 3D point into a 64-bit curve key per its CRS.
fn encode_curve(point: &Point) -> u64 {
    match point.crs {
        Crs::Wgs84_2d => {
            // S2 leaf cell id (level 30): Hilbert order on the sphere, no pole or
            // antimeridian distortion (vs the old lat/lon Morton). coords[0]=lon,
            // coords[1]=lat; the leaf cell id IS the curve key.
            CellID::from(LatLng::from_degrees(point.coords[1], point.coords[0])).0
        }
        Crs::Cartesian2d => {
            // Hilbert (not Morton): better spatial locality for flat Cartesian
            // spaces, per arch/core/spatial.md. WGS-84 keeps Morton until the S2
            // swap; Cartesian-3D keeps Morton (3D Hilbert is a follow-up).
            let x = quantise_u32(point.coords[0], -1e9, 1e9);
            let y = quantise_u32(point.coords[1], -1e9, 1e9);
            hilbert_2d(x, y)
        }
        Crs::Wgs84_3d => {
            // S2 horizontal (top bits of the level-30 leaf) + altitude (low
            // WGS84_3D_ALT_BITS bits), packed in one u64. coords[0]=lon,
            // coords[1]=lat, coords[2]=alt(m). No pole distortion horizontally.
            let leaf = CellID::from(LatLng::from_degrees(point.coords[1], point.coords[0])).0;
            let alt = u64::from(quantise_bits(
                point.coords[2],
                WGS84_3D_ALT_LO,
                WGS84_3D_ALT_HI,
                WGS84_3D_ALT_BITS,
            ));
            (leaf & !WGS84_3D_ALT_MASK) | alt
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
///
/// # Examples
///
/// ```
/// use coordinode_modality::{distance, Crs, Point};
/// let paris = Point::new_2d(Crs::Wgs84_2d, 2.3522, 48.8566);
/// let kyiv = Point::new_2d(Crs::Wgs84_2d, 30.5234, 50.4501);
/// // Great-circle distance is about 2030 km.
/// let d = distance(&paris, &kyiv);
/// assert!((1_900_000.0..2_100_000.0).contains(&d));
/// ```
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

/// Decode the 8-byte curve value from a spatial key
/// (`<prefix><curve:8 BE><node_id:8 BE>`). Returns `None` for a
/// malformed (too-short) key.
fn decode_curve_from_key(key: &[u8]) -> Option<u64> {
    let curve_end = key.len().checked_sub(8)?;
    let curve_start = curve_end.checked_sub(8)?;
    let curve_bytes: [u8; 8] = key.get(curve_start..curve_end)?.try_into().ok()?;
    Some(u64::from_be_bytes(curve_bytes))
}

impl SpatialStore for LocalSpatialStore {
    fn insert(
        &self,
        txn: &mut Transaction,
        label_id: u32,
        node_id: NodeId,
        point: &Point,
    ) -> StoreResult<()> {
        let curve = encode_curve(point);
        let key = encode_spatial_key(label_id, point.crs, curve, node_id);
        let body = encode_body(point)?;
        txn.put(Partition::Idx, &key, &body)?;
        Ok(())
    }

    fn delete(
        &self,
        txn: &mut Transaction,
        label_id: u32,
        node_id: NodeId,
        point: &Point,
    ) -> StoreResult<()> {
        let curve = encode_curve(point);
        let key = encode_spatial_key(label_id, point.crs, curve, node_id);
        txn.delete(Partition::Idx, &key)?;
        Ok(())
    }

    fn scan_within_bbox(
        &self,
        txn: &Transaction,
        label_id: u32,
        crs: Crs,
        bbox: &Bbox,
    ) -> StoreResult<Vec<(NodeId, Point)>> {
        // Per-CRS space-filling-curve decomposition into a set of index
        // intervals covering the bbox, scanned with ONE seekable iterator:
        // `seek_to` each interval's start and drain to its `hi`, so the dead-zone
        // bytes between intervals are skipped at the iterator (no I/O) rather than
        // scanned-then-filtered. Morton CRS (WGS-84 2D, WGS-84-3D, Cartesian-3D)
        // may collapse to a single broad interval; Hilbert (Cartesian-2D) always
        // yields a multi-interval cover (it has no order-preserving broad window).
        //
        // Key layout: `<prefix><curve:8 BE><node_id:8 BE>`. The covering invariant
        // (every bbox point's curve lies in some interval) plus the exact
        // `point_in_bbox` post-filter (boundary-quad supersets admit curve false
        // positives) make the result exact.
        let intervals = curve_intervals(bbox);
        let mut out = Vec::new();
        if intervals.is_empty() {
            return Ok(out);
        }
        let (first_lo, _) = intervals[0];
        let (_, last_hi) = intervals[intervals.len() - 1];
        let lo_key = encode_spatial_key(label_id, crs, first_lo, NodeId::from_raw(0));
        let hi_key = encode_spatial_key(label_id, crs, last_hi, NodeId::from_raw(u64::MAX));
        let mut it = txn.base_range_seekable(Partition::Idx, &lo_key, &hi_key)?;
        for &(lo, hi) in &intervals {
            it.seek_to(&encode_spatial_key(label_id, crs, lo, NodeId::from_raw(0)));
            for guard in it.by_ref() {
                let (key, value) = guard.into_inner()?;
                let Some(curve) = decode_curve_from_key(key.as_ref()) else {
                    continue;
                };
                if curve > hi {
                    break; // past this interval; outer loop seeks to the next lo
                }
                let Some(node_id) = decode_node_id_from_key(key.as_ref()) else {
                    continue;
                };
                let point = decode_body(value.as_ref())?;
                if point_in_bbox(&point, bbox) {
                    out.push((node_id, point));
                }
            }
        }
        Ok(out)
    }

    fn knn_nearest(
        &self,
        txn: &Transaction,
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
        for (key, value) in txn.base_prefix_scan(Partition::Idx, &prefix)? {
            let Some(node_id) = decode_node_id_from_key(&key) else {
                continue;
            };
            let point = decode_body(&value)?;
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
    fn write_spatial(
        engine: &StorageEngine,
        body: impl FnOnce(&LocalSpatialStore, &mut Transaction),
    ) {
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
}
