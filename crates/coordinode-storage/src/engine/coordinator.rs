//! Layer-3 multi-partition coordinator — the partition-keyed dispatch
//! primitive that sits between Layer 4 modality stores and Layer 2
//! per-partition LSM trees.
//!
//! ## Responsibilities (per `arch/core/storage-stack.md` §Layer 3)
//!
//! - **Per-partition tree map** — maintains the `Partition →
//!   AnyTree` mapping, opened once at engine bootstrap.
//! - **MVCC snapshot coordination** — exposes the shared
//!   `SharedSequenceNumberGenerator` so every read or scan can be
//!   pinned to a consistent point-in-time across all partitions.
//! - **OCC primitive** — [`Coordinator::has_write_after`] detects
//!   the post-snapshot write that signals a transaction conflict.
//! - **Merge-operator dispatch** — `merge` on the underlying tree
//!   replays the registered merge operator for the partition; the
//!   coordinator is the call site that surfaces those into typed
//!   APIs above.
//! - **WriteBatch dispatch** — the typed `WriteBatch` builder
//!   atomically commits across partitions; the coordinator hosts
//!   the entry point.
//!
//! ## What this is NOT
//!
//! - **Not Layer 2** — per-LSM-level endpoint routing, cascade
//!   eviction, tier policy live in `engine::routing` /
//!   `engine::compaction`.
//! - **Not Layer 1** — capacity tracking, endpoint config,
//!   page-ECC policy live in `engine::capacity` and
//!   `engine::config`.
//! - **Not replication** — `ReplicatedWriter` (R142a) wraps
//!   Layer-4 store writes from outside, not coordinator calls.
//!   The coordinator only needs to expose a deterministic write
//!   stream (achieved via the shared seqno generator).
//! - **Not consumer registry** — `SeqnoConsumerRegistry` (R137a /
//!   ADR-028) observes [`Coordinator::current_seqno`] from
//!   outside; the coordinator does not pull on retention.
//!
//! ## Wire-in points for R142a / R137a
//!
//! `Coordinator` exposes the seqno generator and the partition-tree
//! handles. R142a wraps the Layer-4 stores (above), not the
//! coordinator. R137a registers a consumer that periodically reads
//! `current_seqno()` and writes back a watermark via
//! [`Coordinator::set_gc_watermark`]. Neither hook requires API
//! changes on `Coordinator`; the seam is the existing accessor
//! surface.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use lsm_tree::{AbstractTree, Guard};

use super::StorageIter;
use crate::engine::partition::Partition;
use crate::error::{StorageError, StorageResult};

/// Multi-partition coordinator. Owns the per-partition LSM tree
/// handles and the shared seqno generator + GC watermark.
///
/// One `Coordinator` is created per [`crate::engine::core::StorageEngine`]
/// at bootstrap and lives for the engine's lifetime. The engine
/// composes the coordinator alongside Layer-2 routing state,
/// Layer-1 capacity state, and the optional standalone WAL.
pub struct Coordinator {
    /// Per-partition LSM tree handles, opened at engine bootstrap.
    /// `Partition::all()` keys every entry; `tree(part)` is the
    /// only sanctioned access path.
    trees: HashMap<Partition, lsm_tree::AnyTree>,
    /// Shared LSM seqno generator. All `Coordinator` writes pull
    /// from this counter; readers pin to `seqno.get()` for
    /// snapshot consistency across partitions.
    seqno: lsm_tree::SharedSequenceNumberGenerator,
    /// Shared block cache across every partition tree. Owned at
    /// the coordinator level so all partition trees share one
    /// physical cache budget.
    cache: Arc<lsm_tree::Cache>,
    /// MVCC GC watermark — the seqno below which version-history
    /// retention can be dropped at compaction time. Observed by
    /// `SeqnoRetentionFilter` (R063) on every compaction.
    gc_watermark: Arc<AtomicU64>,
}

impl Coordinator {
    /// Build a coordinator from the bootstrap state — invoked once
    /// from `StorageEngine::finish_open` after the per-partition
    /// trees have been opened with their routing configuration.
    pub(crate) fn new(
        trees: HashMap<Partition, lsm_tree::AnyTree>,
        seqno: lsm_tree::SharedSequenceNumberGenerator,
        cache: Arc<lsm_tree::Cache>,
        gc_watermark: Arc<AtomicU64>,
    ) -> Self {
        Self {
            trees,
            seqno,
            cache,
            gc_watermark,
        }
    }

    /// Borrow the partition handle for a given logical partition.
    /// Returns `StorageError::PartitionNotFound` if the partition
    /// is not registered (should never happen after a successful
    /// `open` — `Partition::all()` keys every entry).
    pub fn tree(&self, part: Partition) -> StorageResult<&lsm_tree::AnyTree> {
        self.trees
            .get(&part)
            .ok_or_else(|| StorageError::PartitionNotFound {
                name: part.name().to_string(),
            })
    }

    /// Iterator over every (partition, tree) pair. Used by lifecycle
    /// helpers (cascade eviction, persist, disk-space aggregation)
    /// that need to act over all partitions uniformly.
    pub(crate) fn trees(&self) -> &HashMap<Partition, lsm_tree::AnyTree> {
        &self.trees
    }

    /// Shared block cache reference (DRAM-backed). Single physical
    /// allocation across all partition trees so the cache budget is
    /// cluster-wide, not per-partition.
    pub fn cache(&self) -> &Arc<lsm_tree::Cache> {
        &self.cache
    }

    /// Borrow the seqno generator. Exposed so callers that need to
    /// stamp writes outside the standard `put`/`merge`/`delete` path
    /// (Raft apply, recovery replay) can take a fresh seqno.
    pub(crate) fn seqno_generator(&self) -> &lsm_tree::SharedSequenceNumberGenerator {
        &self.seqno
    }

    /// Allocate the next monotonically-increasing seqno. One call =
    /// one consumed value. Used by single-key writes (`put`,
    /// `delete`, `merge`) — multi-key atomic writes pull one seqno
    /// from `WriteBatch` and stamp the whole batch with it.
    pub(crate) fn next_seqno(&self) -> lsm_tree::SeqNo {
        self.seqno.next()
    }

    /// Current "next visible to readers" seqno. Equivalent to
    /// `snapshot()` — the value returned is suitable as a read
    /// visibility bound (readers at this seqno see all writes
    /// committed strictly before).
    pub fn current_seqno(&self) -> lsm_tree::SeqNo {
        self.seqno.get()
    }

    /// Update the MVCC GC watermark. Versions with seqno ≤
    /// `watermark` are eligible for retention drop at the next
    /// compaction by the `SeqnoRetentionFilter` (R063).
    ///
    /// `SeqnoConsumerRegistry` (R137a / ADR-028) drives this from
    /// outside: it computes `shard_floor()` across CDC / snapshot /
    /// backup consumers and writes the result here.
    pub fn set_gc_watermark(&self, watermark: u64) {
        self.gc_watermark.store(watermark, Ordering::Release);
    }

    /// Point-in-time snapshot seqno. Pins reads to the moment of
    /// the call — later writes are invisible. Equivalent to
    /// [`Self::current_seqno`].
    pub fn snapshot(&self) -> lsm_tree::SeqNo {
        self.seqno.get()
    }

    /// Returns a previously-allocated snapshot seqno. The wrapper
    /// is `Option` so future implementations (EE multi-shard) can
    /// signal "snapshot too old, recover from WAL" — currently
    /// always `Some`.
    pub fn snapshot_at(&self, seqno: lsm_tree::SeqNo) -> Option<lsm_tree::SeqNo> {
        Some(seqno)
    }

    /// Read a key at the latest visible seqno. No capacity gate —
    /// reads never consume endpoint budget.
    pub fn get(&self, part: Partition, key: &[u8]) -> StorageResult<Option<bytes::Bytes>> {
        let tree = self.tree(part)?;
        let value = tree.get(key, self.seqno.get())?;
        Ok(value.map(|v| bytes::Bytes::copy_from_slice(&v)))
    }

    /// Read a key through a pinned snapshot. Writes after the
    /// snapshot are invisible.
    pub fn snapshot_get(
        &self,
        snapshot: &lsm_tree::SeqNo,
        part: Partition,
        key: &[u8],
    ) -> StorageResult<Option<bytes::Bytes>> {
        let tree = self.tree(part)?;
        let value = tree.get(key, *snapshot)?;
        Ok(value.map(|v| bytes::Bytes::copy_from_slice(&v)))
    }

    /// Existence check without materialising the value. Same
    /// MVCC visibility rules as [`Self::get`].
    pub fn contains_key(&self, part: Partition, key: &[u8]) -> StorageResult<bool> {
        let tree = self.tree(part)?;
        let value = tree.get(key, self.seqno.get())?;
        Ok(value.is_some())
    }

    /// Prefix scan at the latest visible seqno.
    pub fn prefix_scan(&self, part: Partition, prefix: &[u8]) -> StorageResult<StorageIter> {
        let tree = self.tree(part)?;
        let seqno = self.seqno.get();
        Ok(Box::new(tree.prefix(prefix, seqno, None)))
    }

    /// Prefix scan through a pinned snapshot.
    pub fn prefix_scan_at(
        &self,
        part: Partition,
        prefix: &[u8],
        snapshot: lsm_tree::SeqNo,
    ) -> StorageResult<StorageIter> {
        let tree = self.tree(part)?;
        Ok(Box::new(tree.prefix(prefix, snapshot, None)))
    }

    /// Snapshot-pinned prefix scan, materialised into a `Vec`.
    /// Convenience wrapper for callers that need owned data — the
    /// underlying iterator must be drained before the snapshot is
    /// dropped.
    pub fn snapshot_prefix_scan(
        &self,
        snapshot: &lsm_tree::SeqNo,
        part: Partition,
        prefix: &[u8],
    ) -> StorageResult<Vec<(Vec<u8>, bytes::Bytes)>> {
        let tree = self.tree(part)?;
        let mut results = Vec::new();
        for guard in tree.prefix(prefix, *snapshot, None) {
            let (key, value) = guard.into_inner()?;
            results.push((key.to_vec(), bytes::Bytes::copy_from_slice(&value)));
        }
        Ok(results)
    }

    /// OCC primitive: was `key` in `part` written after
    /// `after_seqno`?
    ///
    /// Returns `true` when:
    /// - the latest live entry has `seqno > after_seqno` (a put or
    ///   merge landed since), OR
    /// - the key has no live entry but DID exist at `after_seqno`
    ///   (a tombstone landed since).
    ///
    /// Otherwise returns `false` (no concurrent write detected).
    pub fn has_write_after(
        &self,
        part: Partition,
        key: &[u8],
        after_seqno: lsm_tree::SeqNo,
    ) -> StorageResult<bool> {
        let tree = self.tree(part)?;
        let entry = tree.get_internal_entry(key, lsm_tree::SeqNo::MAX)?;

        match entry {
            Some(e) if e.key.seqno > after_seqno => Ok(true),
            Some(_) => Ok(false),
            None => {
                let old_val = self.snapshot_get(&after_seqno, part, key)?;
                Ok(old_val.is_some())
            }
        }
    }

    /// Single-key put. Stamps the write with the next seqno and
    /// dispatches to the partition tree. Capacity gating happens at
    /// the [`crate::engine::core::StorageEngine`] level above —
    /// callers must invoke `check_partition_capacity` first.
    pub(crate) fn put_no_capacity_check(
        &self,
        part: Partition,
        key: &[u8],
        value: &[u8],
    ) -> StorageResult<lsm_tree::SeqNo> {
        let tree = self.tree(part)?;
        let seqno = self.seqno.next();
        tree.insert(key, value, seqno);
        Ok(seqno)
    }

    /// Single-key tombstone. Stamps with the next seqno.
    pub(crate) fn delete(&self, part: Partition, key: &[u8]) -> StorageResult<lsm_tree::SeqNo> {
        let tree = self.tree(part)?;
        let seqno = self.seqno.next();
        tree.remove(key, seqno);
        Ok(seqno)
    }

    /// Single-key merge-operand append. Stamps with the next seqno.
    /// Capacity gating handled above.
    pub(crate) fn merge_no_capacity_check(
        &self,
        part: Partition,
        key: &[u8],
        operand: &[u8],
    ) -> StorageResult<lsm_tree::SeqNo> {
        let tree = self.tree(part)?;
        let seqno = self.seqno.next();
        tree.merge(key, operand, seqno);
        Ok(seqno)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
    use crate::engine::core::StorageEngine;
    use tempfile::TempDir;

    fn open_engine() -> (TempDir, StorageEngine) {
        let dir = TempDir::new().expect("tempdir");
        let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "ep",
            dir.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        let engine = StorageEngine::open(&cfg).expect("open");
        (dir, engine)
    }

    #[test]
    fn coordinator_visible_through_engine() {
        let (_dir, engine) = open_engine();
        // Public surface unchanged: get / snapshot / contains_key
        // delegate to the inner Coordinator.
        let _snap = engine.snapshot();
        let res = engine.get(Partition::Node, b"never").expect("get ok");
        assert!(res.is_none());
        let exists = engine
            .contains_key(Partition::Node, b"never")
            .expect("contains ok");
        assert!(!exists);
    }

    #[test]
    fn coordinator_has_write_after_detects_put() {
        // SequenceNumberCounter::get() returns "next-to-allocate",
        // so snapshot() taken before a put coincides with the put's
        // seqno. Probe with `snapshot - 1` to assert the inequality
        // strictly (same shape as existing
        // has_write_after_detects_newer_write that probes oracle-100
        // with 99).
        let (_dir, engine) = open_engine();
        let before = engine.snapshot();
        engine.put(Partition::Node, b"k", b"v").expect("put");
        assert!(engine
            .has_write_after(Partition::Node, b"k", before.saturating_sub(1))
            .expect("hwa"),);
    }

    #[test]
    fn coordinator_has_write_after_detects_delete() {
        let (_dir, engine) = open_engine();
        engine.put(Partition::Node, b"k", b"v").expect("seed");
        let snap = engine.snapshot();
        engine.delete(Partition::Node, b"k").expect("delete");
        assert!(engine
            .has_write_after(Partition::Node, b"k", snap)
            .expect("hwa"),);
    }

    #[test]
    fn coordinator_tree_returns_every_partition() {
        // Every Partition::all() variant must be openable through
        // the coordinator — guards against future partition additions
        // that forget to register a tree.
        let (_dir, engine) = open_engine();
        for &part in Partition::all() {
            engine
                .coordinator()
                .tree(part)
                .unwrap_or_else(|e| panic!("tree({part:?}) failed: {e}"));
        }
    }

    #[test]
    fn coordinator_has_write_after_no_write_returns_false() {
        // Probe an untouched key at a horizon that no write has
        // crossed → false. Guards against the false-positive case
        // where has_write_after over-reports.
        let (_dir, engine) = open_engine();
        let after = engine.snapshot();
        let observed = engine
            .has_write_after(Partition::Node, b"untouched", after)
            .expect("hwa");
        assert!(!observed);
    }

    #[test]
    fn coordinator_contains_key_round_trip() {
        let (_dir, engine) = open_engine();
        assert!(!engine.contains_key(Partition::Node, b"k").unwrap());
        engine.put(Partition::Node, b"k", b"v").unwrap();
        assert!(engine.contains_key(Partition::Node, b"k").unwrap());
        engine.delete(Partition::Node, b"k").unwrap();
        assert!(!engine.contains_key(Partition::Node, b"k").unwrap());
    }

    #[test]
    fn coordinator_prefix_scan_empty_returns_empty_iter() {
        let (_dir, engine) = open_engine();
        let iter = engine.prefix_scan(Partition::Node, b"never").unwrap();
        let count = iter.count();
        assert_eq!(count, 0);
    }

    #[test]
    fn coordinator_set_gc_watermark_observable() {
        // set_gc_watermark must be Release-ordered so a subsequent
        // Acquire-load (e.g. inside the compaction filter factory)
        // observes the new value. Smoke-check: write, then construct
        // a fresh factory that reads it.
        let (_dir, engine) = open_engine();
        engine.set_gc_watermark(12345);
        // The retention filter factory reads via Arc<AtomicU64>; we
        // can't get at it directly, but the engine's set_gc_watermark
        // delegates to coordinator.set_gc_watermark which uses
        // store(Release). Smoke: writes don't panic and the engine
        // remains operable.
        engine
            .put(Partition::Node, b"k", b"v")
            .expect("put after gc watermark set");
        let v = engine.get(Partition::Node, b"k").unwrap();
        assert_eq!(v.as_deref(), Some(b"v".as_slice()));
    }

    #[test]
    fn coordinator_concurrent_reads_under_arc_share() {
        // The coordinator's read methods take &self only — Arc-share
        // a StorageEngine across threads and probe in parallel.
        use std::sync::Arc;
        use std::thread;

        let (dir, engine) = open_engine();
        let engine = Arc::new(engine);
        engine.put(Partition::Node, b"k", b"v").unwrap();
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let e = Arc::clone(&engine);
                thread::spawn(move || {
                    for _ in 0..50 {
                        let v = e.get(Partition::Node, b"k").expect("get");
                        assert_eq!(v.as_deref(), Some(b"v".as_slice()));
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("join");
        }
        drop(dir); // keep TempDir alive for the scope of the test
    }

    #[test]
    fn coordinator_snapshot_isolation_holds() {
        let (_dir, engine) = open_engine();
        engine.put(Partition::Node, b"k", b"v1").expect("put v1");
        let snap = engine.snapshot();
        engine.put(Partition::Node, b"k", b"v2").expect("put v2");
        // Latest read sees v2.
        let latest = engine.get(Partition::Node, b"k").expect("get").unwrap();
        assert_eq!(latest.as_ref(), b"v2");
        // Snapshot read sees v1.
        let pinned = engine
            .snapshot_get(&snap, Partition::Node, b"k")
            .expect("snap get")
            .unwrap();
        assert_eq!(pinned.as_ref(), b"v1");
    }
}
