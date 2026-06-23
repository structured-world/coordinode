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
mod tests;
