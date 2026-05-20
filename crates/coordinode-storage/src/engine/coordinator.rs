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
//! - **OCC primitive** — [`MultiModalCoordinator::has_write_after`]
//!   detects the post-snapshot write that signals a transaction
//!   conflict.
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
//! - **Not consumer registry** — `SeqnoConsumerRegistry` observes
//!   [`MultiModalCoordinator::current_seqno`] from outside; the
//!   coordinator does not pull on retention.
//!
//! ## Wire-in points for replication / consumer registry
//!
//! The coordinator exposes the seqno generator and the partition-tree
//! handles. Replicated-writer wraps the Layer-4 stores (above), not
//! the coordinator. The consumer registry periodically reads
//! `current_seqno()` and writes back a watermark via
//! [`MultiModalCoordinator::set_gc_watermark`]. Neither hook requires
//! API changes; the seam is the existing accessor surface.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use lsm_tree::{AbstractTree, Guard};

use super::StorageIter;
use crate::engine::partition::Partition;
use crate::error::{StorageError, StorageResult};

/// Layer-3 contract: multimodal coordinator across the 8 partitions
/// (`arch/core/storage-stack.md` §Layer 3, §Per-layer trait surface).
///
/// CE deployments use [`LocalMultiModalCoordinator`] (single-Raft,
/// single-shard). EE Phase 3 plugs in `MultiShardCoordinator` against
/// the same trait — cross-shard 2PC and distributed snapshot
/// composition happen below this contract; Layer 4 modality stores
/// and Layer 5 query engine bind to the trait, not the concrete impl.
///
/// The trait surface deliberately omits `AnyTree` / `lsm_tree::Cache`
/// handles — per Layer-3 arch responsibility ("Knows: all partitions,
/// merge ops, MVCC snapshots, OCC sets; does NOT know: per-level
/// endpoint placement, tier policy"), exposing LSM internals through
/// the trait would leak Layer 2 concerns. Concrete impls may surface
/// those handles for engine-internal use; trait consumers stay on the
/// typed read/write/scan/seqno/OCC primitives below.
pub trait MultiModalCoordinator: Send + Sync {
    /// Allocate the next monotonically-increasing seqno.
    fn next_seqno(&self) -> lsm_tree::SeqNo;

    /// Current "next visible to readers" seqno — suitable as a read
    /// visibility bound and as the starting point for an OCC scope.
    fn current_seqno(&self) -> lsm_tree::SeqNo;

    /// MVCC snapshot pin — equivalent to [`Self::current_seqno`].
    /// Reads at this seqno see all writes committed strictly before.
    fn snapshot(&self) -> lsm_tree::SeqNo;

    /// Validate / promote a previously-allocated snapshot seqno. The
    /// `Option` lets EE multi-shard signal "snapshot too old, recover
    /// from WAL"; CE always returns `Some`.
    fn snapshot_at(&self, seqno: lsm_tree::SeqNo) -> Option<lsm_tree::SeqNo>;

    /// MVCC GC watermark setter. `SeqnoConsumerRegistry` (R137a /
    /// ADR-028) drives this from outside with the floor across CDC /
    /// snapshot / backup consumers. Versions with seqno ≤ watermark
    /// become eligible for compaction-time drop.
    fn set_gc_watermark(&self, watermark: u64);

    /// Read a key at the latest visible seqno.
    fn get(&self, part: Partition, key: &[u8]) -> StorageResult<Option<bytes::Bytes>>;

    /// Read a key through a pinned snapshot.
    fn snapshot_get(
        &self,
        snapshot: &lsm_tree::SeqNo,
        part: Partition,
        key: &[u8],
    ) -> StorageResult<Option<bytes::Bytes>>;

    /// Existence check at the latest visible seqno.
    fn contains_key(&self, part: Partition, key: &[u8]) -> StorageResult<bool>;

    /// Prefix scan at the latest visible seqno.
    fn prefix_scan(&self, part: Partition, prefix: &[u8]) -> StorageResult<StorageIter>;

    /// Prefix scan through a pinned snapshot.
    fn prefix_scan_at(
        &self,
        part: Partition,
        prefix: &[u8],
        snapshot: lsm_tree::SeqNo,
    ) -> StorageResult<StorageIter>;

    /// Snapshot-pinned prefix scan, materialised owned. Convenience
    /// over `prefix_scan_at` for callers that need eager collection.
    fn snapshot_prefix_scan(
        &self,
        snapshot: &lsm_tree::SeqNo,
        part: Partition,
        prefix: &[u8],
    ) -> StorageResult<Vec<(Vec<u8>, bytes::Bytes)>>;

    /// OCC primitive: was `key` in `part` written after `after_seqno`?
    /// Returns true when a put / merge / tombstone landed since.
    fn has_write_after(
        &self,
        part: Partition,
        key: &[u8],
        after_seqno: lsm_tree::SeqNo,
    ) -> StorageResult<bool>;
}

/// CE single-Raft, single-shard [`MultiModalCoordinator`] implementation.
/// Owns the per-partition LSM tree handles + shared seqno generator
/// + GC watermark + block cache.
///
/// One instance is created per [`crate::engine::core::StorageEngine`]
/// at bootstrap and lives for the engine's lifetime. The engine
/// composes the coordinator alongside Layer-2 routing state, Layer-1
/// capacity state, and the optional standalone WAL.
///
/// **Layer 4 / Layer 5 callers should bind to the
/// [`MultiModalCoordinator`] trait, not this concrete type, so the
/// EE-Phase-3 `MultiShardCoordinator` can swap in transparently.**
/// Engine-internal code that needs the LSM tree handles directly
/// (cascade eviction, persist, capacity refresh) uses the concrete
/// `pub` accessors below — those are NOT part of the trait contract.
pub struct LocalMultiModalCoordinator {
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

impl LocalMultiModalCoordinator {
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
    ///
    /// **Not part of the [`MultiModalCoordinator`] trait** — exposing
    /// `AnyTree` would leak Layer-2 concerns into Layer 3's contract.
    /// Reserved for engine-internal call sites (cascade eviction,
    /// persist, capacity refresh).
    pub fn tree(&self, part: Partition) -> StorageResult<&lsm_tree::AnyTree> {
        self.trees
            .get(&part)
            .ok_or_else(|| StorageError::PartitionNotFound {
                name: part.name().to_string(),
            })
    }

    /// Iterator over every (partition, tree) pair. Used by lifecycle
    /// helpers (cascade eviction, persist, disk-space aggregation)
    /// that need to act over all partitions uniformly. Engine-internal.
    pub(crate) fn trees(&self) -> &HashMap<Partition, lsm_tree::AnyTree> {
        &self.trees
    }

    /// Shared block cache reference (DRAM-backed). Single physical
    /// allocation across all partition trees so the cache budget is
    /// cluster-wide, not per-partition. Engine-internal accessor —
    /// not part of the trait surface.
    pub fn cache(&self) -> &Arc<lsm_tree::Cache> {
        &self.cache
    }

    /// Borrow the seqno generator. Exposed so callers that need to
    /// stamp writes outside the standard `put`/`merge`/`delete` path
    /// (Raft apply, recovery replay) can take a fresh seqno.
    pub(crate) fn seqno_generator(&self) -> &lsm_tree::SharedSequenceNumberGenerator {
        &self.seqno
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

impl MultiModalCoordinator for LocalMultiModalCoordinator {
    fn next_seqno(&self) -> lsm_tree::SeqNo {
        self.seqno.next()
    }

    fn current_seqno(&self) -> lsm_tree::SeqNo {
        self.seqno.get()
    }

    fn snapshot(&self) -> lsm_tree::SeqNo {
        self.seqno.get()
    }

    fn snapshot_at(&self, seqno: lsm_tree::SeqNo) -> Option<lsm_tree::SeqNo> {
        Some(seqno)
    }

    fn set_gc_watermark(&self, watermark: u64) {
        self.gc_watermark.store(watermark, Ordering::Release);
    }

    fn get(&self, part: Partition, key: &[u8]) -> StorageResult<Option<bytes::Bytes>> {
        let tree = self.tree(part)?;
        let value = tree.get(key, self.seqno.get())?;
        Ok(value.map(|v| bytes::Bytes::copy_from_slice(&v)))
    }

    fn snapshot_get(
        &self,
        snapshot: &lsm_tree::SeqNo,
        part: Partition,
        key: &[u8],
    ) -> StorageResult<Option<bytes::Bytes>> {
        let tree = self.tree(part)?;
        let value = tree.get(key, *snapshot)?;
        Ok(value.map(|v| bytes::Bytes::copy_from_slice(&v)))
    }

    fn contains_key(&self, part: Partition, key: &[u8]) -> StorageResult<bool> {
        let tree = self.tree(part)?;
        let value = tree.get(key, self.seqno.get())?;
        Ok(value.is_some())
    }

    fn prefix_scan(&self, part: Partition, prefix: &[u8]) -> StorageResult<StorageIter> {
        let tree = self.tree(part)?;
        let seqno = self.seqno.get();
        Ok(Box::new(tree.prefix(prefix, seqno, None)))
    }

    fn prefix_scan_at(
        &self,
        part: Partition,
        prefix: &[u8],
        snapshot: lsm_tree::SeqNo,
    ) -> StorageResult<StorageIter> {
        let tree = self.tree(part)?;
        Ok(Box::new(tree.prefix(prefix, snapshot, None)))
    }

    fn snapshot_prefix_scan(
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

    fn has_write_after(
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
        // delegate to the inner LocalMultiModalCoordinator.
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
