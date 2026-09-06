//! Seqno-based MVCC version retention via LSM compaction filter.
//!
//! During compaction, this filter drops MVCC versions at or below the GC
//! watermark that a newer version shadows. At least one version per user
//! key is always preserved, so a key not written since the window opened
//! keeps its current value.
//!
//! ## How it works
//!
//! The LSM engine assigns a monotonic sequence number (seqno) to every
//! write. On an oracle-backed engine seqno = commit timestamp, so seqno-based
//! retention is time-based retention without a wall-clock dependency.
//!
//! During compaction the filter sees items sorted by (key ASC, seqno DESC):
//!
//! 1. Read the GC watermark from the shared `Arc<AtomicU64>` (published by
//!    the `GcWatermarkController`: `min(live pins, consumer floor,
//!    time-travel window)`).
//! 2. `item.seqno() > watermark` → `Keep` (inside the retention window).
//! 3. Expired AND no version of this key kept yet → `Keep` (the key's only
//!    surviving version, no data loss for cold keys).
//! 4. Expired AND a newer version already kept → `Remove`.
//!
//! ## Why this is safe for time travel
//!
//! A compaction output is consulted only by reads at snapshots above its
//! install seqno: the tree keeps every earlier `SuperVersion` (and the tables
//! it references) alive until the watermark passes it, and routes a read at
//! snapshot `S` to the version that was current at `S`. A read at `S` above
//! the install seqno sees the newest version of every key as of the output,
//! which step 3/4 always keep. Reads below the watermark are refused by the
//! engine before they reach the tree.
//!
//! The same rule applies in every deployment mode; what differs is who
//! publishes the watermark (embedded: the engine's own window and pins;
//! cluster: those plus the consumer registry floor).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use lsm_tree::compaction::filter::{CompactionFilter, Context, Factory, ItemAccessor, Verdict};

type CompactionFilterResult = lsm_tree::Result<Verdict>;

/// Compaction filter factory that creates seqno-based retention filters.
///
/// Stores a shared GC watermark (`Arc<AtomicU64>`) that is updated
/// externally by the `GcWatermarkController`. Each compaction run reads the
/// current watermark at filter creation time.
pub(crate) struct SeqnoRetentionFilterFactory {
    gc_watermark: Arc<AtomicU64>,
}

impl SeqnoRetentionFilterFactory {
    /// Create a factory with a shared GC watermark.
    ///
    /// Initial value of 0 means "keep everything" — safe default before the
    /// engine is fully open.
    pub(crate) fn new(gc_watermark: Arc<AtomicU64>) -> Self {
        Self { gc_watermark }
    }
}

impl Factory for SeqnoRetentionFilterFactory {
    fn name(&self) -> &str {
        "coordinode.seqno_retention"
    }

    fn make_filter(&self, _ctx: &Context) -> Box<dyn CompactionFilter> {
        Box::new(SeqnoRetentionFilter::new(
            self.gc_watermark.load(Ordering::Acquire),
        ))
    }
}

/// Per-compaction-run seqno-based retention filter.
///
/// Items with `seqno > watermark` are inside the retention window and kept
/// unconditionally. For items at or below the watermark, only the newest
/// version per key survives (to prevent total data loss for keys that have
/// not been written recently).
struct SeqnoRetentionFilter {
    /// Versions with `seqno <= watermark` are eligible for removal.
    watermark: u64,
    /// The user key of the last item processed.
    last_key: Vec<u8>,
    /// Whether the current key already has at least one kept version.
    has_kept_version: bool,
}

impl SeqnoRetentionFilter {
    fn new(watermark: u64) -> Self {
        Self {
            watermark,
            last_key: Vec::new(),
            has_kept_version: false,
        }
    }

    /// The retention verdict for one item. Items arrive (key ASC, seqno
    /// DESC), so the first item of a key is its newest version.
    fn decide(&mut self, key: &[u8], seqno: u64) -> Verdict {
        if key != self.last_key.as_slice() {
            self.last_key.clear();
            self.last_key.extend_from_slice(key);
            self.has_kept_version = false;
        }

        if seqno > self.watermark {
            self.has_kept_version = true;
            return Verdict::Keep;
        }

        if !self.has_kept_version {
            self.has_kept_version = true;
            return Verdict::Keep;
        }

        // Expired and shadowed by a kept version. Removed via tombstone; the
        // LSM drops the tombstone at the last level.
        Verdict::Remove
    }
}

impl CompactionFilter for SeqnoRetentionFilter {
    fn filter_item(&mut self, item: ItemAccessor<'_>, _ctx: &Context) -> CompactionFilterResult {
        Ok(self.decide(item.key(), item.seqno()))
    }
}

/// Create an `Arc<dyn Factory>` for seqno-based retention.
///
/// The `gc_watermark` is shared with the `GcWatermarkController`, which
/// publishes every change.
pub(crate) fn seqno_retention_factory(gc_watermark: Arc<AtomicU64>) -> Arc<dyn Factory> {
    Arc::new(SeqnoRetentionFilterFactory::new(gc_watermark))
}

#[cfg(test)]
mod tests;
