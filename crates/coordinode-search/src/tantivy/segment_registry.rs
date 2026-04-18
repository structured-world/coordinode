//! Per-segment commit_ts registry for MVCC snapshot reads.
//!
//! Each tantivy segment in a `TextIndex` carries every document's originating
//! Raft proposal commit_ts as a u64 fast field. The registry caches the
//! `(min_ts, max_ts)` pair per `SegmentId` so that `search_at(T)` can:
//!
//! 1. Skip entirely segments whose `min_ts > T` (nothing visible).
//! 2. Include entirely segments whose `max_ts <= T` (everything visible, no
//!    per-doc filter — the fast path).
//! 3. Fall back to a per-doc `FilterCollector` only for "straddle" segments
//!    where `min_ts <= T < max_ts`.
//!
//! Merge safety: when tantivy merges segments, each component doc keeps its
//! original `commit_ts` fast-field value, so scanning the merged segment's
//! column yields `(min(components.min_ts), max(components.max_ts))`
//! automatically. No manual merge-lineage tracking is required.

use std::collections::HashMap;

use tantivy::index::SegmentId;
use tantivy::Searcher;

/// `(min_commit_ts, max_commit_ts)` observed in a segment's fast-field column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentTsRange {
    pub min_ts: u64,
    pub max_ts: u64,
}

/// Reader-side cache of per-segment `commit_ts` ranges.
///
/// Rebuilt from the searcher on each reload. Segments that disappear (merged
/// into a new `SegmentId`) are dropped; segments that appear (new flush or
/// merge output) are scanned once and cached.
#[derive(Debug, Default)]
pub struct SegmentRegistry {
    entries: HashMap<SegmentId, SegmentTsRange>,
}

impl SegmentRegistry {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    pub fn get(&self, id: &SegmentId) -> Option<SegmentTsRange> {
        self.entries.get(id).copied()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Reconcile the registry against the current searcher.
    ///
    /// For each segment present in the searcher but absent from the registry,
    /// scan its `commit_ts` fast-field column to compute `(min, max)`.
    /// Segments no longer present (consumed by merges) are evicted.
    ///
    /// Runs in O(total_new_docs) the first time a segment is seen and O(0)
    /// thereafter. Called after every `reader.reload()`.
    pub fn reconcile(
        &mut self,
        searcher: &Searcher,
        commit_ts_field_name: &str,
    ) -> Result<(), tantivy::TantivyError> {
        let mut keep: HashMap<SegmentId, SegmentTsRange> =
            HashMap::with_capacity(searcher.segment_readers().len());

        for seg_reader in searcher.segment_readers() {
            let id = seg_reader.segment_id();
            if let Some(existing) = self.entries.get(&id) {
                keep.insert(id, *existing);
                continue;
            }

            let ff_reader = seg_reader.fast_fields();
            let column = ff_reader.u64(commit_ts_field_name)?;
            let max_doc = seg_reader.max_doc();

            let mut min_ts = u64::MAX;
            let mut max_ts = u64::MIN;
            let alive = seg_reader.alive_bitset();
            let mut any_alive = false;
            for doc_id in 0..max_doc {
                if let Some(bitset) = alive {
                    if !bitset.is_alive(doc_id) {
                        continue;
                    }
                }
                any_alive = true;
                let ts = column.first(doc_id).unwrap_or(0);
                if ts < min_ts {
                    min_ts = ts;
                }
                if ts > max_ts {
                    max_ts = ts;
                }
            }

            if !any_alive {
                // Entirely deleted segment — skip. Tantivy will drop it on next merge.
                continue;
            }
            keep.insert(id, SegmentTsRange { min_ts, max_ts });
        }

        self.entries = keep;
        Ok(())
    }

    /// Classify a segment for a snapshot read at `T`.
    ///
    /// Returns:
    /// - `Visibility::All` — every doc in the segment is visible (max_ts ≤ T).
    /// - `Visibility::None` — no doc is visible (min_ts > T) → skip search.
    /// - `Visibility::Straddle` — mixed; caller must apply per-doc filter.
    /// - `Visibility::Unknown` — not in registry (race between commit and
    ///   reconcile) → caller must apply per-doc filter as the safe default.
    pub fn classify(&self, id: &SegmentId, snapshot_ts: u64) -> Visibility {
        match self.entries.get(id) {
            None => Visibility::Unknown,
            Some(range) if range.max_ts <= snapshot_ts => Visibility::All,
            Some(range) if range.min_ts > snapshot_ts => Visibility::None,
            Some(_) => Visibility::Straddle,
        }
    }
}

/// Segment visibility decision for a snapshot read at T.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    All,
    None,
    Straddle,
    Unknown,
}
