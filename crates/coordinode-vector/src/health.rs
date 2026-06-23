//! Index lifecycle state + freshness watermark — surfaced in gRPC responses,
//! EXPLAIN output, and Prometheus metrics so callers can distinguish a
//! fully-built index from one being rebuilt after segment migration, and can
//! gauge how far the index has caught up with committed writes.
//!
//! See `arch/distribution/live-rebalance.md § HNSW under rebalance` for the
//! contract; this module is the canonical type definition that every layer
//! (vector engine, gRPC handlers, EXPLAIN renderer, metrics exporter) imports
//! and dispatches on.
//!
//! The state lives alongside `HnswIndex` as an atomic, separate from the graph
//! itself, so the search path can read it without a lock and the build path
//! can publish progress without blocking readers.
//!
//! # Freshness watermark (read-your-writes)
//!
//! `indexed_hlc` is the HLC (wall-clock-microsecond commit timestamp, ADR-007)
//! of the last oplog entry the index-maintenance worker has applied to the
//! local graph. A client that just wrote at HLC `W` can carry `W` into a
//! follow-up query; comparing it against the served `indexed_hlc` tells the
//! coordinator whether this replica has caught up (`indexed_hlc >= W`) or is
//! lagging — the Pinecone LSN-header pattern. `last_committed_hlc -
//! indexed_hlc` is the per-shard lag.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex;

/// Lifecycle state of a vector index.
///
/// Pure data type — produced by [`HealthSignal::snapshot`], consumed by every
/// layer that needs to dispatch on the current state. Never stored directly;
/// always read via the atomic-backed [`HealthSignal`] type below to avoid
/// locking the search path.
///
/// `Ready` and `Rebuilding` carry `indexed_hlc` — the freshness watermark (see
/// the module docs). `Offline` carries no watermark: an unavailable index has
/// no meaningful catch-up point.
#[derive(Debug, Clone, PartialEq)]
pub enum IndexHealthState {
    /// Index is fully built and serving queries at the documented recall.
    /// `indexed_hlc` is the HLC of the last applied write (0 if none yet).
    Ready {
        /// HLC watermark of the last applied write (read-your-writes fence).
        indexed_hlc: u64,
    },
    /// Index is currently being (re)built. `progress` ∈ [0.0, 1.0]; `eta_ms`
    /// is a best-effort estimate of remaining build time in milliseconds;
    /// `indexed_hlc` is the HLC of the last write folded into the partial
    /// graph so far.
    ///
    /// Callers consult `VECTOR_REBUILD_POLICY` on the label to decide
    /// whether to (a) return `IndexNotReady`, (b) serve from the partial
    /// graph with `accuracy_warning: true`, or (c) block until `Ready`.
    Rebuilding {
        /// Build progress in `[0.0, 1.0]`.
        progress: f32,
        /// Best-effort estimate of remaining build time, in milliseconds.
        eta_ms: u64,
        /// HLC watermark of the last write folded into the partial graph.
        indexed_hlc: u64,
    },
    /// Index is unavailable. `reason` is a short human-readable string
    /// suitable for surfacing to the caller (e.g. "segment_lost", "disk_io",
    /// "manual_disable").
    Offline {
        /// Short human-readable reason for the outage.
        reason: String,
    },
}

impl IndexHealthState {
    /// Whether the index is in a state where search results are trustworthy
    /// at full recall.
    pub fn is_ready(&self) -> bool {
        matches!(self, IndexHealthState::Ready { .. })
    }

    /// Whether the index is mid-rebuild. `progress` available via match.
    pub fn is_rebuilding(&self) -> bool {
        matches!(self, IndexHealthState::Rebuilding { .. })
    }

    /// Whether the index is fully unavailable.
    pub fn is_offline(&self) -> bool {
        matches!(self, IndexHealthState::Offline { .. })
    }

    /// Short label for metrics / logs (no allocation).
    pub fn label(&self) -> &'static str {
        match self {
            IndexHealthState::Ready { .. } => "ready",
            IndexHealthState::Rebuilding { .. } => "rebuilding",
            IndexHealthState::Offline { .. } => "offline",
        }
    }

    /// The freshness watermark for this state, or `None` when offline.
    /// `Ready` / `Rebuilding` expose the HLC of the last applied write;
    /// `Offline` has no meaningful watermark.
    pub fn indexed_hlc(&self) -> Option<u64> {
        match self {
            IndexHealthState::Ready { indexed_hlc }
            | IndexHealthState::Rebuilding { indexed_hlc, .. } => Some(*indexed_hlc),
            IndexHealthState::Offline { .. } => None,
        }
    }
}

/// Encoded discriminant — bits 0..4 of the `state` atomic.
const STATE_READY: u32 = 0;
const STATE_REBUILDING: u32 = 1;
const STATE_OFFLINE: u32 = 2;

/// Atomic lifecycle signal — read on every search call, written by the build
/// path. The `Ready` and `Rebuilding` cases are encoded entirely in atomics
/// (zero allocation, zero locks). The `Offline` case carries a reason
/// `String` which lives behind a `Mutex<Option<String>>` — `Offline` is rare
/// (segment failure, manual disable), so the mutex is uncontended in
/// practice.
///
/// # Atomic layout
///
/// * `state: AtomicU32` — discriminant (low bits) + `progress` percentile
///   `0..=10_000` (high bits, fixed-point hundredths-of-percent). One atomic
///   write publishes both fields consistently.
/// * `eta_ms: AtomicU64` — best-effort ETA in milliseconds. Read after the
///   discriminant; a slight skew vs `state` is acceptable for an estimate.
/// * `indexed_hlc: AtomicU64` — freshness watermark (HLC of the last applied
///   write). Advanced monotonically via [`HealthSignal::advance_indexed_hlc`]
///   (`fetch_max`), so out-of-order or replayed applies never move it
///   backwards. Independent of the lifecycle discriminant: a rebuild keeps
///   advancing it as it folds writes, and a ready index keeps advancing it as
///   the maintenance worker applies the live oplog.
/// * `offline_reason: Mutex<Option<String>>` — only touched on transition
///   to/from `Offline`. Locked from the search path only in the rare
///   `Offline` branch; locked from the build path only on transition.
#[derive(Debug)]
pub struct HealthSignal {
    state: AtomicU32,
    eta_ms: AtomicU64,
    indexed_hlc: AtomicU64,
    offline_reason: Mutex<Option<String>>,
}

impl HealthSignal {
    /// Construct a fresh `Ready` signal with a zero watermark. Use
    /// [`HealthSignal::new_rebuilding`] when a fresh index starts mid-build.
    pub fn new_ready() -> Arc<Self> {
        Arc::new(Self {
            state: AtomicU32::new(encode(STATE_READY, 0)),
            eta_ms: AtomicU64::new(0),
            indexed_hlc: AtomicU64::new(0),
            offline_reason: Mutex::new(None),
        })
    }

    /// Construct a fresh signal that starts in `Rebuilding{progress: 0}`.
    /// Use for replicas receiving a freshly-migrated segment.
    pub fn new_rebuilding() -> Arc<Self> {
        Arc::new(Self {
            state: AtomicU32::new(encode(STATE_REBUILDING, 0)),
            eta_ms: AtomicU64::new(0),
            indexed_hlc: AtomicU64::new(0),
            offline_reason: Mutex::new(None),
        })
    }

    /// Snapshot the current state. Cheap — at most one `Mutex` lock in the
    /// `Offline` branch, otherwise pure atomic reads.
    pub fn snapshot(&self) -> IndexHealthState {
        let raw = self.state.load(Ordering::Acquire);
        let (kind, progress_fp) = decode(raw);
        match kind {
            STATE_READY => IndexHealthState::Ready {
                indexed_hlc: self.indexed_hlc.load(Ordering::Relaxed),
            },
            STATE_REBUILDING => IndexHealthState::Rebuilding {
                progress: progress_fp as f32 / 10_000.0,
                eta_ms: self.eta_ms.load(Ordering::Relaxed),
                indexed_hlc: self.indexed_hlc.load(Ordering::Relaxed),
            },
            STATE_OFFLINE => {
                let reason = self
                    .offline_reason
                    .lock()
                    .ok()
                    .and_then(|g| g.clone())
                    .unwrap_or_else(|| "unknown".to_string());
                IndexHealthState::Offline { reason }
            }
            _ => IndexHealthState::Offline {
                reason: "corrupted_state".to_string(),
            },
        }
    }

    /// The current freshness watermark (HLC of the last applied write), read
    /// without touching the lifecycle discriminant. 0 means nothing applied
    /// yet.
    pub fn indexed_hlc(&self) -> u64 {
        self.indexed_hlc.load(Ordering::Acquire)
    }

    /// Advance the freshness watermark to `hlc` if it is newer. Monotonic
    /// (`fetch_max`): a replayed or out-of-order apply can never move the
    /// watermark backwards, so a `read-your-writes` fence built on it is
    /// sound. Call after the write at `hlc` has been folded into the graph.
    pub fn advance_indexed_hlc(&self, hlc: u64) {
        self.indexed_hlc.fetch_max(hlc, Ordering::Release);
    }

    /// Transition to `Ready`. Clears any stored offline reason. Leaves the
    /// freshness watermark untouched — readiness and freshness are orthogonal.
    pub fn mark_ready(&self) {
        self.state.store(encode(STATE_READY, 0), Ordering::Release);
        self.eta_ms.store(0, Ordering::Relaxed);
        if let Ok(mut g) = self.offline_reason.lock() {
            *g = None;
        }
    }

    /// Update the rebuild progress. `progress` clamped to `[0.0, 1.0]`.
    /// Also transitions to `Rebuilding` if currently in another state.
    pub fn report_rebuild_progress(&self, progress: f32, eta_ms: u64) {
        let clamped = progress.clamp(0.0, 1.0);
        let progress_fp = (clamped * 10_000.0) as u32;
        self.state
            .store(encode(STATE_REBUILDING, progress_fp), Ordering::Release);
        self.eta_ms.store(eta_ms, Ordering::Relaxed);
    }

    /// Transition to `Offline` with a reason. Subsequent searches see the
    /// offline state and surface the reason via the gRPC error message.
    pub fn mark_offline(&self, reason: impl Into<String>) {
        let reason = reason.into();
        if let Ok(mut g) = self.offline_reason.lock() {
            *g = Some(reason);
        }
        self.state
            .store(encode(STATE_OFFLINE, 0), Ordering::Release);
        self.eta_ms.store(0, Ordering::Relaxed);
    }
}

impl Default for HealthSignal {
    /// Same as [`HealthSignal::new_ready`] but without the `Arc`. Use the
    /// constructor when sharing across the search and build paths.
    fn default() -> Self {
        Self {
            state: AtomicU32::new(encode(STATE_READY, 0)),
            eta_ms: AtomicU64::new(0),
            indexed_hlc: AtomicU64::new(0),
            offline_reason: Mutex::new(None),
        }
    }
}

/// Pack discriminant (low 4 bits) + progress percentile (high 28 bits) into
/// one `u32`. 28 bits gives a resolution of 1 / 2²⁸ ≈ 4 × 10⁻⁹, vastly more
/// than the `0..=10_000` (0.01% steps) we actually use — chosen because
/// 28 + 4 = 32 leaves room for future discriminants without re-encoding.
#[inline]
const fn encode(kind: u32, progress_fp: u32) -> u32 {
    (kind & 0xF) | (progress_fp << 4)
}

#[inline]
const fn decode(raw: u32) -> (u32, u32) {
    (raw & 0xF, raw >> 4)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
