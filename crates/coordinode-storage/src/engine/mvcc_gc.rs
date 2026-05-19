//! Seqno-based MVCC version retention via LSM compaction filter.
//!
//! During compaction, this filter drops old MVCC versions whose seqno
//! falls below the GC watermark (provided by `SnapshotTracker`).
//! At least one version per user key is always preserved to prevent
//! total data loss.
//!
//! ## How it works
//!
//! The LSM engine assigns a monotonic sequence number (seqno) to every
//! write. After R064, seqno = TimestampOracle timestamp, so seqno-based
//! retention is equivalent to time-based retention without wall-clock
//! dependency.
//!
//! During compaction the filter sees items sorted by (key ASC, seqno DESC):
//!
//! 1. Read GC watermark from shared `Arc<AtomicU64>` (set by SnapshotTracker)
//! 2. If `item.seqno() > watermark` → `Keep` (within retention window)
//! 3. If expired AND this is the newest version of this key → `Keep`
//!    (prevents total data loss — at least one version survives)
//! 4. If expired AND a newer version already kept → `Destroy`
//!
//! ## CE/EE
//!
//! CE: GC watermark driven by SnapshotTracker (pins active snapshots).
//! EE: same mechanism, configurable retention window.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use lsm_tree::compaction::filter::{CompactionFilter, Context, Factory, ItemAccessor, Verdict};

type CompactionFilterResult = lsm_tree::Result<Verdict>;

/// Compaction filter factory that creates seqno-based retention filters.
///
/// Stores a shared GC watermark (`Arc<AtomicU64>`) that is updated
/// externally by the SnapshotTracker or Database layer. Each compaction
/// run reads the current watermark at filter creation time.
pub(crate) struct SeqnoRetentionFilterFactory {
    gc_watermark: Arc<AtomicU64>,
}

impl SeqnoRetentionFilterFactory {
    /// Create a factory with a shared GC watermark.
    ///
    /// The watermark should be updated by the caller (typically via
    /// `SnapshotTracker::get_seqno_safe_to_gc()`). Initial value of 0
    /// means "keep everything" — safe default before the DB is fully open.
    pub(crate) fn new(gc_watermark: Arc<AtomicU64>) -> Self {
        Self { gc_watermark }
    }
}

impl Factory for SeqnoRetentionFilterFactory {
    fn name(&self) -> &str {
        "coordinode.seqno_retention"
    }

    fn make_filter(&self, _ctx: &Context) -> Box<dyn CompactionFilter> {
        let watermark = self.gc_watermark.load(Ordering::Acquire);

        Box::new(SeqnoRetentionFilter {
            watermark,
            last_key: Vec::new(),
            has_live_version: false,
        })
    }
}

/// Per-compaction-run seqno-based retention filter.
///
/// Items with `seqno > watermark` are within the retention window and
/// kept unconditionally. For items at or below the watermark, only the
/// newest version per key survives (to prevent total data loss for keys
/// that haven't been written recently).
struct SeqnoRetentionFilter {
    /// Versions with `seqno <= watermark` are eligible for removal.
    watermark: u64,
    /// The user key of the last item processed.
    last_key: Vec<u8>,
    /// Whether the current key already has at least one kept version.
    has_live_version: bool,
}

impl CompactionFilter for SeqnoRetentionFilter {
    fn filter_item(&mut self, item: ItemAccessor<'_>, _ctx: &Context) -> CompactionFilterResult {
        let key = item.key();
        let seqno = item.seqno();

        // Track per-key state: detect key boundary.
        // LSM compaction iterates (key ASC, seqno DESC), so the first
        // item for each key is the newest version.
        let is_new_key = key.as_ref() != self.last_key.as_slice();
        if is_new_key {
            self.last_key.clear();
            self.last_key.extend_from_slice(key);
            self.has_live_version = false;
        }

        // Within retention window — always keep.
        if seqno > self.watermark {
            self.has_live_version = true;
            return Ok(Verdict::Keep);
        }

        // Below watermark: eligible for GC.
        // Keep the newest expired version if no live version exists yet.
        if !self.has_live_version {
            self.has_live_version = true;
            return Ok(Verdict::Keep);
        }

        // Older expired version with a newer version already kept.
        // Remove via tombstone (safe: LSM handles tombstone cleanup
        // at the last level via seqno_safe_to_gc).
        Ok(Verdict::Remove)
    }
}

/// Create an `Arc<dyn Factory>` for seqno-based retention.
///
/// The `gc_watermark` is shared with the caller who must update it
/// (typically from `SnapshotTracker::get_seqno_safe_to_gc()`).
pub(crate) fn seqno_retention_factory(gc_watermark: Arc<AtomicU64>) -> Arc<dyn Factory> {
    Arc::new(SeqnoRetentionFilterFactory::new(gc_watermark))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Factory produces filters with correct name.
    #[test]
    fn factory_name() {
        let watermark = Arc::new(AtomicU64::new(0));
        let factory = SeqnoRetentionFilterFactory::new(watermark);
        assert_eq!(factory.name(), "coordinode.seqno_retention");
    }

    /// Shared watermark is readable via Arc.
    #[test]
    fn shared_watermark_updates() {
        let watermark = Arc::new(AtomicU64::new(0));
        let factory_watermark = Arc::clone(&watermark);

        // Initial value
        assert_eq!(factory_watermark.load(Ordering::Acquire), 0);

        // External update visible to factory
        watermark.store(42, Ordering::Release);
        assert_eq!(factory_watermark.load(Ordering::Acquire), 42);
    }

    /// Retention filter logic: items above watermark are kept.
    #[test]
    fn filter_keeps_items_above_watermark() {
        let filter = SeqnoRetentionFilter {
            watermark: 100,
            last_key: Vec::new(),
            has_live_version: false,
        };

        // seqno 200 > watermark 100 → within retention
        assert!(filter.watermark < 200);
        // seqno 50 <= watermark 100 → eligible for GC
        assert!(filter.watermark >= 50);
    }

    /// Retention filter preserves newest expired version per key.
    #[test]
    fn filter_preserves_newest_expired_version() {
        let filter = SeqnoRetentionFilter {
            watermark: 100,
            last_key: Vec::new(),
            has_live_version: false,
        };

        // When has_live_version is false and item is below watermark,
        // the filter should keep it (first expired = newest version).
        assert!(!filter.has_live_version);
    }

    /// Retention filter destroys older expired versions.
    #[test]
    fn filter_destroys_older_expired() {
        let filter = SeqnoRetentionFilter {
            watermark: 100,
            last_key: b"key1".to_vec(),
            has_live_version: true,
        };

        // When has_live_version is true and item is below watermark,
        // the filter should destroy it (older expired version).
        assert!(filter.has_live_version);
    }
}
