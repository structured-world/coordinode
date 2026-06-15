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
//! - **Not replication** — the replicated-writer wraps Layer-4 store
//!   writes from outside, not coordinator calls. The coordinator only
//!   needs to expose a deterministic write stream (achieved via the
//!   shared seqno generator).
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

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use lsm_tree::{AbstractTree, Guard};

use super::StorageIter;
use crate::engine::partition::Partition;
use crate::error::{StorageError, StorageResult};

/// OCC read scope — Layer-3 owned read-set tracker for optimistic
/// concurrency control.
///
/// A scope is created at transaction start (pinned to a snapshot
/// seqno) and accumulates the `(Partition, key)` of every read that
/// participates in conflict detection. At commit time the coordinator
/// validates the scope: every tracked key is probed for a write whose
/// seqno is newer than the scope's `read_ts`. If any such write
/// exists the transaction has a read-write conflict and must abort.
///
/// **Thread-safe collection.** The internal set is `Mutex`-guarded so
/// parallel executor paths (rayon worker pools traversing a fan-out
/// edge set) can share a single `&OccScope` and push tracked keys
/// concurrently. Uncontended path is one atomic CAS per insert.
///
/// **Commutative partitions skipped.** Writes to partitions reporting
/// [`Partition::is_commutative`] (currently `Adj`, `Counter`) are
/// conflict-free by construction — merge operators compose concurrent
/// writers at read time. `track` records them but `validate_occ` on
/// the coordinator excludes them from the conflict probe.
pub struct OccScope {
    read_ts: lsm_tree::SeqNo,
    read_set: Mutex<HashSet<(Partition, Vec<u8>)>>,
}

impl OccScope {
    /// Construct an empty scope pinned at the given snapshot seqno.
    /// Prefer [`MultiModalCoordinator::occ_scope_at`] over direct
    /// construction so the scope's `read_ts` is sourced from the
    /// coordinator's snapshot machinery, not an ad-hoc value.
    pub fn new(read_ts: lsm_tree::SeqNo) -> Self {
        Self {
            read_ts,
            read_set: Mutex::new(HashSet::new()),
        }
    }

    /// The pinned snapshot seqno this scope reads at. Conflict
    /// detection at commit time probes "any write seqno > read_ts".
    pub fn read_ts(&self) -> lsm_tree::SeqNo {
        self.read_ts
    }

    /// Record a read against `(part, key)`. Idempotent — a key is
    /// stored at most once regardless of how many times it is read.
    pub fn track(&self, part: Partition, key: &[u8]) {
        if let Ok(mut guard) = self.read_set.lock() {
            guard.insert((part, key.to_vec()));
        }
    }

    /// Bulk-merge tracked keys from another collection (e.g. results
    /// of a rayon parallel block that collected into a local
    /// `Vec<(Partition, Vec<u8>)>` before joining back).
    pub fn extend<I: IntoIterator<Item = (Partition, Vec<u8>)>>(&self, iter: I) {
        if let Ok(mut guard) = self.read_set.lock() {
            guard.extend(iter);
        }
    }

    /// Number of distinct tracked `(part, key)` pairs. Test/EXPLAIN
    /// hook — production code does not inspect this.
    pub fn tracked_count(&self) -> usize {
        self.read_set.lock().map(|g| g.len()).unwrap_or(0)
    }

    /// Drain into a `Vec` for handing back to a parent scope or
    /// emitting in EXPLAIN. Resets the internal set.
    pub fn drain(&self) -> Vec<(Partition, Vec<u8>)> {
        match self.read_set.lock() {
            Ok(mut guard) => guard.drain().collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Is `(part, key)` tracked in this scope? Provided for tests
    /// and EXPLAIN-style introspection; the production OCC path
    /// does not call this — it iterates the full set via
    /// [`MultiModalCoordinator::validate_occ`].
    pub fn contains(&self, part: Partition, key: &[u8]) -> bool {
        match self.read_set.lock() {
            Ok(guard) => guard.contains(&(part, key.to_vec())),
            Err(_) => false,
        }
    }

    /// Typed variant of [`Self::contains`] for the
    /// `(Partition::Node, encode_node_key(shard, id))` pair. Built
    /// for the R165 OCC audit suite so test assertions don't have
    /// to reach into the raw `encode_node_key` encoder (R166
    /// encoder-lockdown follow-up).
    pub fn contains_node(&self, shard: u16, id: coordinode_core::graph::node::NodeId) -> bool {
        let key = coordinode_core::graph::node::encode_node_key(shard, id);
        self.contains(Partition::Node, &key)
    }

    /// Typed variant of [`Self::contains`] for the per-version
    /// temporal node key — matches what `mvcc_get_node_temporal`
    /// records on the OCC scope.
    pub fn contains_node_temporal(
        &self,
        shard: u16,
        id: coordinode_core::graph::node::NodeId,
        valid_from_ms: i64,
    ) -> bool {
        let key = coordinode_core::graph::node::encode_temporal_node_key(shard, id, valid_from_ms);
        self.contains(Partition::Node, &key)
    }

    /// Typed variant of [`Self::contains`] for the non-temporal
    /// edge-property key.
    pub fn contains_edge_props(
        &self,
        edge_type: &str,
        src: coordinode_core::graph::node::NodeId,
        tgt: coordinode_core::graph::node::NodeId,
    ) -> bool {
        let key = coordinode_core::graph::edge::encode_edgeprop_key(edge_type, src, tgt);
        self.contains(Partition::EdgeProp, &key)
    }

    /// Typed variant of [`Self::contains`] for the per-version
    /// temporal edge-property key (25-byte form).
    pub fn contains_edge_props_temporal(
        &self,
        edge_type: &str,
        src: coordinode_core::graph::node::NodeId,
        tgt: coordinode_core::graph::node::NodeId,
        valid_from_ms: i64,
    ) -> bool {
        let key = coordinode_core::graph::edge::encode_temporal_edgeprop_key(
            edge_type,
            src,
            tgt,
            valid_from_ms,
        );
        self.contains(Partition::EdgeProp, &key)
    }

    /// Borrow the tracked set under the lock — internal use only.
    /// Consumed by [`MultiModalCoordinator::validate_occ`].
    pub(crate) fn with_keys<R>(&self, f: impl FnOnce(&HashSet<(Partition, Vec<u8>)>) -> R) -> R {
        match self.read_set.lock() {
            Ok(guard) => f(&guard),
            Err(poisoned) => f(&poisoned.into_inner()),
        }
    }
}

/// First key whose post-snapshot write triggered the conflict.
/// Returned by [`MultiModalCoordinator::validate_occ`] so callers
/// can surface a useful conflict message.
#[derive(Debug, Clone)]
pub struct OccConflict {
    pub partition: Partition,
    pub key: Vec<u8>,
    pub read_ts: lsm_tree::SeqNo,
}

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
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a `MultiModalCoordinator` — Layer 4 stores and Layer 5 executor can't bind to it",
    label = "expected `&dyn MultiModalCoordinator` or `&LocalMultiModalCoordinator`",
    note = "CE deployments obtain a coordinator via `StorageEngine::coordinator()` which returns the \
            single-shard `LocalMultiModalCoordinator`. EE deployments plug `MultiShardCoordinator` \
            against the same trait. If you're implementing a new coordinator, mirror the surface of \
            `LocalMultiModalCoordinator` (seqno + read/write/scan/OCC primitives) and review \
            arch/core/storage-stack.md §Layer 3 before adding methods."
)]
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

    /// MVCC GC watermark setter. The seqno-consumer registry drives
    /// this from outside with the floor across CDC / snapshot /
    /// backup consumers. Versions with seqno ≤ watermark become
    /// eligible for compaction-time drop.
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

    /// Inclusive-bounded range scan at the latest visible seqno.
    /// Yields entries with keys `K` such that `start ≤ K ≤ end`
    /// (both bounds inclusive, in byte-lexicographic order). When
    /// `start > end` the iterator yields no entries — callers must
    /// ensure ordering themselves if they care; this API does not
    /// validate or panic on inversion.
    ///
    /// Used by callers that have already decomposed a query into
    /// disjoint key intervals (e.g. spatial Z-curve subrange
    /// decomposition) — issuing a per-interval `range_scan` avoids
    /// scanning the in-between keys that a single broad
    /// `prefix_scan` would walk.
    fn range_scan(&self, part: Partition, start: &[u8], end: &[u8]) -> StorageResult<StorageIter>;

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

    /// Begin an OCC scope pinned at the given read snapshot. The
    /// returned scope is empty; callers populate it via
    /// [`OccScope::track`] during the transaction body, then submit
    /// it to [`Self::validate_occ`] at commit time.
    ///
    /// The default implementation constructs a plain [`OccScope`].
    /// EE multi-shard impls may override to attach shard-locality
    /// metadata for cross-shard validation.
    fn occ_scope_at(&self, read_ts: lsm_tree::SeqNo) -> OccScope {
        OccScope::new(read_ts)
    }

    /// OCC commit-time validation. For every key tracked in `scope`,
    /// probes the partition for a write whose seqno is strictly
    /// greater than `scope.read_ts`. Commutative partitions
    /// ([`Partition::is_commutative`] = true) are skipped — merge-
    /// operator writes there don't produce read-write conflicts.
    ///
    /// Returns `Ok(None)` when no conflict was detected;
    /// `Ok(Some(OccConflict))` for the first conflicting key found
    /// (subsequent keys are not checked — the conflict is decisive).
    /// `Err` only on storage I/O failure.
    fn validate_occ(&self, scope: &OccScope) -> StorageResult<Option<OccConflict>> {
        let read_ts = scope.read_ts;
        scope.with_keys(|keys| {
            for (part, key) in keys.iter() {
                if part.is_commutative() {
                    continue;
                }
                if self.has_write_after(*part, key, read_ts)? {
                    return Ok(Some(OccConflict {
                        partition: *part,
                        key: key.clone(),
                        read_ts,
                    }));
                }
            }
            Ok(None)
        })
    }
}

/// CE single-Raft, single-shard [`MultiModalCoordinator`] implementation.
/// Owns the per-partition LSM tree handles + shared seqno generator
/// Drives the MVCC GC watermark from the set of currently-pinned read
/// snapshots.
///
/// The watermark is the seqno below which version history and merge operands
/// may be folded or collected at compaction. **Invariant:** the watermark must
/// never exceed the oldest live snapshot, or a reader pinned at that snapshot
/// would observe folded state newer than its pin. With no live snapshots the
/// watermark tracks the current seqno, so compaction folds everything.
///
/// Per-query reads at the current seqno are always `>=` the watermark and need
/// no protection; the registry exists to hold the watermark back for in-flight
/// statements and long-lived consumers (backup, CDC, checkpoint) that pin an
/// older seqno across compactions.
pub struct GcWatermarkController {
    /// Pinned seqno -> reference count. The smallest key is the oldest live
    /// snapshot; an empty map means no reader is pinned.
    pins: Mutex<BTreeMap<u64, usize>>,
    /// The shared atomic the compaction filter reads as its fold/GC threshold.
    gc_watermark: Arc<AtomicU64>,
    /// Current-seqno source; the watermark target when nothing is pinned.
    seqno: lsm_tree::SharedSequenceNumberGenerator,
}

impl GcWatermarkController {
    fn new(gc_watermark: Arc<AtomicU64>, seqno: lsm_tree::SharedSequenceNumberGenerator) -> Self {
        let controller = Self {
            pins: Mutex::new(BTreeMap::new()),
            gc_watermark,
            seqno,
        };
        if let Ok(pins) = controller.pins.lock() {
            controller.recompute(&pins);
        }
        controller
    }

    /// Recompute and publish the watermark: the oldest pinned seqno, or the
    /// current seqno when nothing is pinned. Caller holds the `pins` lock.
    fn recompute(&self, pins: &BTreeMap<u64, usize>) {
        let watermark = pins
            .keys()
            .next()
            .copied()
            .unwrap_or_else(|| self.seqno.get());
        self.gc_watermark.store(watermark, Ordering::Release);
    }

    /// Pin the current snapshot seqno so the watermark cannot advance past it
    /// until the returned guard drops. Returns the pinned seqno.
    pub fn pin(self: &Arc<Self>) -> (u64, SnapshotPin) {
        let seqno = self.seqno.get();
        (seqno, self.pin_at(seqno))
    }

    /// Pin an explicit seqno — for a reader that already holds a snapshot at a
    /// possibly-older point in time (a statement reading at its allocated
    /// `read_ts`, or a long-lived backup / CDC consumer). The watermark will
    /// not advance past `seqno` until the returned guard drops.
    pub fn pin_at(self: &Arc<Self>, seqno: u64) -> SnapshotPin {
        if let Ok(mut pins) = self.pins.lock() {
            *pins.entry(seqno).or_insert(0) += 1;
            self.recompute(&pins);
        }
        SnapshotPin {
            controller: Arc::clone(self),
            seqno,
        }
    }

    /// Advance the watermark toward the current seqno. A no-op while any
    /// snapshot is pinned; call periodically so folding and version GC keep up
    /// during quiescent periods with no read traffic.
    pub fn tick(&self) {
        if let Ok(pins) = self.pins.lock() {
            self.recompute(&pins);
        }
    }

    fn release(&self, seqno: u64) {
        if let Ok(mut pins) = self.pins.lock() {
            if let Some(count) = pins.get_mut(&seqno) {
                *count -= 1;
                if *count == 0 {
                    pins.remove(&seqno);
                }
            }
            self.recompute(&pins);
        }
    }
}

/// RAII guard holding a read-snapshot pin. While alive it keeps the GC
/// watermark at or below its seqno; dropping it releases the pin and lets the
/// watermark advance.
#[must_use = "dropping the pin immediately releases the snapshot"]
pub struct SnapshotPin {
    controller: Arc<GcWatermarkController>,
    seqno: u64,
}

impl SnapshotPin {
    /// The seqno this guard pins reads to.
    pub fn seqno(&self) -> u64 {
        self.seqno
    }
}

impl Drop for SnapshotPin {
    fn drop(&mut self) {
        self.controller.release(self.seqno);
    }
}

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
    /// retention can be dropped at compaction time. Observed by the
    /// seqno-retention compaction filter on every compaction.
    gc_watermark: Arc<AtomicU64>,
    /// Drives `gc_watermark` from live read-snapshot pins so the compaction
    /// filter folds operands / collects versions instead of seeing a frozen
    /// `0` threshold.
    gc_controller: Arc<GcWatermarkController>,
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
        let gc_controller = Arc::new(GcWatermarkController::new(
            Arc::clone(&gc_watermark),
            seqno.clone(),
        ));
        Self {
            trees,
            seqno,
            cache,
            gc_watermark,
            gc_controller,
        }
    }

    /// Pin a read snapshot at the current seqno. While the returned guard is
    /// alive the GC watermark cannot advance past it, so compaction will not
    /// fold or collect state the reader must still observe. Returns the pinned
    /// seqno.
    pub fn pin_snapshot(&self) -> (lsm_tree::SeqNo, SnapshotPin) {
        self.gc_controller.pin()
    }

    /// Pin a read snapshot at an explicit (possibly older) seqno — for a reader
    /// that already allocated its `read_ts`, or a long-lived backup / CDC
    /// consumer.
    pub fn pin_snapshot_at(&self, seqno: lsm_tree::SeqNo) -> SnapshotPin {
        self.gc_controller.pin_at(seqno)
    }

    /// Advance the GC watermark toward the current seqno when no snapshot is
    /// pinned. Call periodically so folding / version GC keep up during quiet
    /// periods.
    pub fn advance_gc_watermark(&self) {
        self.gc_controller.tick();
    }

    /// The current GC watermark value (the seqno below which compaction may
    /// fold operands / collect versions). Observability + test hook.
    pub fn gc_watermark_value(&self) -> u64 {
        self.gc_watermark.load(Ordering::Acquire)
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

    fn range_scan(&self, part: Partition, start: &[u8], end: &[u8]) -> StorageResult<StorageIter> {
        let tree = self.tree(part)?;
        let seqno = self.seqno.get();
        // `lsm_tree::AbstractTree::range` takes a `RangeBounds<K>`;
        // inclusive bounds map directly. Owned `Vec<u8>` so the bounds
        // outlive this call (lsm-tree borrows internally for the scan).
        let range = (
            std::ops::Bound::Included(start.to_vec()),
            std::ops::Bound::Included(end.to_vec()),
        );
        Ok(Box::new(tree.range(range, seqno, None)))
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
    use crate::engine::core::StorageEngine;
    /// Logic-test fixture — coordinator/OCC tests verify behaviour
    /// without depending on disk semantics, so route through MemFs
    /// for ~2× speed-up. Returns `(_backing, engine)` keeping the
    /// same tuple shape as before (callers do `let (_dir, engine)
    /// = open_engine();`); the first element binds to the MemFs
    /// `Arc` lifetime guard now instead of a `TempDir`.
    fn open_engine() -> (std::sync::Arc<lsm_tree::fs::MemFs>, StorageEngine) {
        let (engine, fs) = crate::internal_test_helpers::memory_engine();
        (fs, engine)
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
    fn range_scan_inclusive_bounds_yields_both_endpoints() {
        // lsm-tree's `range` API uses `Bound::Included` for both ends
        // per our wrapper. Seed 3 keys; range over the outer two
        // should yield all three.
        let (_dir, engine) = open_engine();
        engine.put(Partition::Node, b"a", b"1").expect("put a");
        engine.put(Partition::Node, b"m", b"2").expect("put m");
        engine.put(Partition::Node, b"z", b"3").expect("put z");
        let iter = engine
            .range_scan(Partition::Node, b"a", b"z")
            .expect("range");
        let keys: Vec<Vec<u8>> = iter.map(|g| g.into_inner().unwrap().0.to_vec()).collect();
        assert_eq!(
            keys,
            vec![b"a".to_vec(), b"m".to_vec(), b"z".to_vec()],
            "both endpoints inclusive",
        );
    }

    #[test]
    fn range_scan_skips_keys_outside_window() {
        // Pin the "skip dead zones" contract that G101's bbox
        // decomposition relies on — keys outside [start, end] must
        // NOT appear in the iterator at all (not "yielded then
        // filtered").
        let (_dir, engine) = open_engine();
        for k in [b"a" as &[u8], b"b", b"c", b"d", b"e"] {
            engine.put(Partition::Node, k, b"v").expect("put");
        }
        let iter = engine
            .range_scan(Partition::Node, b"b", b"d")
            .expect("range");
        let keys: Vec<Vec<u8>> = iter.map(|g| g.into_inner().unwrap().0.to_vec()).collect();
        assert_eq!(keys, vec![b"b".to_vec(), b"c".to_vec(), b"d".to_vec()]);
    }

    #[test]
    fn range_scan_point_query_single_key() {
        // start == end → single-key probe via range API. Equivalent
        // to a get, but exercised through the range path (used by
        // 1×1-cell bbox decomposition).
        let (_dir, engine) = open_engine();
        engine.put(Partition::Node, b"k", b"v").expect("put");
        let iter = engine
            .range_scan(Partition::Node, b"k", b"k")
            .expect("range");
        let count = iter.count();
        assert_eq!(count, 1);
    }

    #[test]
    fn range_scan_inverted_bounds_yields_empty() {
        // Defensive: when `start > end` lsm-tree's range yields no
        // entries. Pins behaviour for callers that might construct
        // bounds inadvertently (the spatial decomposer guarantees
        // lo ≤ hi internally; this test backstops the API).
        let (_dir, engine) = open_engine();
        engine.put(Partition::Node, b"a", b"v").expect("put");
        engine.put(Partition::Node, b"b", b"v").expect("put");
        let iter = engine
            .range_scan(Partition::Node, b"b", b"a")
            .expect("range");
        assert_eq!(iter.count(), 0, "inverted bounds yield no entries");
    }

    #[test]
    fn range_scan_isolated_per_partition() {
        // A key written to Node must not surface in a Schema range
        // scan over the same byte range, and vice versa. Schema
        // partition has bootstrap entries from engine open, so the
        // assertion is asymmetric: filter to our exact probe key.
        let (_dir, engine) = open_engine();
        let probe_key = b"range_scan_isolation_probe_unique";
        engine
            .put(Partition::Node, probe_key, b"node-value")
            .expect("put node");
        engine
            .put(Partition::Schema, probe_key, b"schema-value")
            .expect("put schema");

        let node_hits: Vec<_> = engine
            .range_scan(Partition::Node, probe_key, probe_key)
            .unwrap()
            .map(|g| {
                let (_k, v) = g.into_inner().unwrap();
                v.to_vec()
            })
            .collect();
        let schema_hits: Vec<_> = engine
            .range_scan(Partition::Schema, probe_key, probe_key)
            .unwrap()
            .map(|g| {
                let (_k, v) = g.into_inner().unwrap();
                v.to_vec()
            })
            .collect();
        assert_eq!(
            node_hits,
            vec![b"node-value".to_vec()],
            "Node range_scan must NOT see Schema partition's value",
        );
        assert_eq!(
            schema_hits,
            vec![b"schema-value".to_vec()],
            "Schema range_scan must NOT see Node partition's value",
        );
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
    fn occ_scope_track_dedupes_and_validates_clean() {
        let (_dir, engine) = open_engine();
        let coord = engine.coordinator();
        let scope = coord.occ_scope_at(engine.snapshot());
        // Track same key twice — set semantics.
        scope.track(Partition::Node, b"k1");
        scope.track(Partition::Node, b"k1");
        scope.track(Partition::Node, b"k2");
        assert_eq!(scope.tracked_count(), 2);
        // No writes since snapshot → no conflict.
        let res = coord.validate_occ(&scope).expect("validate ok");
        assert!(res.is_none());
    }

    #[test]
    fn occ_scope_validate_detects_post_snapshot_write() {
        let (_dir, engine) = open_engine();
        let coord = engine.coordinator();
        // SequenceNumberCounter::get() returns "next-to-allocate", so
        // snapshot()'s value coincides with the next put's seqno. Probe
        // at `snapshot - 1` to assert strict post-snapshot newer-write
        // detection (same shape as coordinator_has_write_after_detects_put).
        let scope = coord.occ_scope_at(engine.snapshot().saturating_sub(1));
        scope.track(Partition::Node, b"hot");
        engine.put(Partition::Node, b"hot", b"v").expect("put");
        let conflict = coord
            .validate_occ(&scope)
            .expect("validate ok")
            .expect("conflict expected");
        assert_eq!(conflict.partition, Partition::Node);
        assert_eq!(conflict.key.as_slice(), b"hot");
    }

    #[test]
    fn occ_scope_skips_commutative_partitions() {
        let (_dir, engine) = open_engine();
        let coord = engine.coordinator();
        let scope = coord.occ_scope_at(engine.snapshot());
        // Adj is commutative — even a post-snapshot write must NOT
        // produce a conflict on it.
        scope.track(Partition::Adj, b"adj_key");
        engine
            .merge(Partition::Adj, b"adj_key", b"any")
            .expect("merge");
        let res = coord.validate_occ(&scope).expect("validate ok");
        assert!(
            res.is_none(),
            "Adj writes are commutative — must not raise OCC conflict"
        );
    }

    #[test]
    fn occ_scope_extend_merges_parallel_collections() {
        let (_dir, engine) = open_engine();
        let coord = engine.coordinator();
        let scope = coord.occ_scope_at(engine.snapshot());
        // Simulate a parallel worker collecting into a local Vec then
        // merging into the shared scope.
        let parallel_keys = vec![
            (Partition::Node, b"p1".to_vec()),
            (Partition::Node, b"p2".to_vec()),
            (Partition::EdgeProp, b"e1".to_vec()),
        ];
        scope.extend(parallel_keys);
        assert_eq!(scope.tracked_count(), 3);
    }

    #[test]
    fn occ_scope_concurrent_track_thread_safe() {
        // OccScope must be Send+Sync so rayon workers can share &OccScope.
        use std::sync::Arc;
        use std::thread;
        let (_dir, engine) = open_engine();
        let scope = Arc::new(engine.coordinator().occ_scope_at(engine.snapshot()));
        let handles: Vec<_> = (0..4)
            .map(|tid| {
                let s = Arc::clone(&scope);
                thread::spawn(move || {
                    for i in 0..50 {
                        let key = format!("k{tid}-{i}");
                        s.track(Partition::Node, key.as_bytes());
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("join");
        }
        // 4 threads × 50 keys, all distinct.
        assert_eq!(scope.tracked_count(), 200);
    }

    #[test]
    fn occ_scope_validate_empty_scope_is_clean() {
        // Empty scope → never any conflict regardless of writes.
        let (_dir, engine) = open_engine();
        let coord = engine.coordinator();
        let scope = coord.occ_scope_at(engine.snapshot());
        engine
            .put(Partition::Node, b"unrelated", b"v")
            .expect("put");
        let res = coord.validate_occ(&scope).expect("validate ok");
        assert!(
            res.is_none(),
            "empty scope must produce no conflict — nothing was read"
        );
    }

    #[test]
    fn occ_scope_validate_returns_first_conflict_decisive() {
        // Multiple tracked keys, multiple post-snapshot writes — first
        // conflict found short-circuits. The return value identifies
        // *a* conflict; downstream retry handles all of them after
        // restart.
        let (_dir, engine) = open_engine();
        let coord = engine.coordinator();
        let scope = coord.occ_scope_at(engine.snapshot().saturating_sub(1));
        scope.track(Partition::Node, b"a");
        scope.track(Partition::Node, b"b");
        scope.track(Partition::Node, b"c");
        engine.put(Partition::Node, b"a", b"v").expect("put a");
        engine.put(Partition::Node, b"b", b"v").expect("put b");
        engine.put(Partition::Node, b"c", b"v").expect("put c");
        let conflict = coord
            .validate_occ(&scope)
            .expect("validate ok")
            .expect("conflict expected");
        // HashSet iteration order is unspecified, but the partition is
        // pinned and the key is one of the tracked set.
        assert_eq!(conflict.partition, Partition::Node);
        assert!(
            [b"a".as_slice(), b"b".as_slice(), b"c".as_slice()].contains(&conflict.key.as_slice()),
            "conflict key must be one of the tracked keys",
        );
    }

    #[test]
    fn occ_scope_drain_resets_and_extends_after() {
        // drain() must clear the internal set so subsequent track/extend
        // start from an empty state (matches the test path that pulls
        // tracked keys out for assertion then expects an empty scope).
        let (_dir, engine) = open_engine();
        let scope = engine.coordinator().occ_scope_at(engine.snapshot());
        scope.track(Partition::Node, b"k1");
        scope.track(Partition::Node, b"k2");
        let drained = scope.drain();
        assert_eq!(drained.len(), 2);
        assert_eq!(scope.tracked_count(), 0);
        // Post-drain track works.
        scope.track(Partition::Node, b"k3");
        assert_eq!(scope.tracked_count(), 1);
        assert!(scope.contains(Partition::Node, b"k3"));
        assert!(!scope.contains(Partition::Node, b"k1"));
    }

    #[test]
    fn occ_scope_contains_unit() {
        let (_dir, engine) = open_engine();
        let scope = engine.coordinator().occ_scope_at(engine.snapshot());
        scope.track(Partition::EdgeProp, b"ep_key");
        assert!(scope.contains(Partition::EdgeProp, b"ep_key"));
        assert!(!scope.contains(Partition::EdgeProp, b"missing"));
        // Same key, wrong partition → false (partition is part of the
        // identity).
        assert!(!scope.contains(Partition::Node, b"ep_key"));
    }

    #[test]
    fn occ_scope_validate_multi_partition_mixed_outcome() {
        // Conflict on a non-commutative partition + writes on a
        // commutative partition: only the non-commutative key triggers.
        let (_dir, engine) = open_engine();
        let coord = engine.coordinator();
        let scope = coord.occ_scope_at(engine.snapshot().saturating_sub(1));
        scope.track(Partition::Adj, b"adj_key"); // commutative — skipped
        scope.track(Partition::Schema, b"schema_key"); // checked
        engine
            .merge(Partition::Adj, b"adj_key", b"x")
            .expect("merge");
        engine
            .put(Partition::Schema, b"schema_key", b"v")
            .expect("put");
        let conflict = coord
            .validate_occ(&scope)
            .expect("validate ok")
            .expect("schema_key conflict expected");
        assert_eq!(
            conflict.partition,
            Partition::Schema,
            "Adj is commutative, only Schema must surface as conflict",
        );
    }

    #[test]
    fn multimodal_coordinator_dyn_dispatch_works() {
        // Bind to the trait through `&dyn`. This is the contract used
        // by EE Phase 3 (`MultiShardCoordinator`) — Layer 4 / Layer 5
        // hold a trait object, never a concrete type.
        let (_dir, engine) = open_engine();
        let coord: &dyn MultiModalCoordinator = engine.coordinator();
        let snap = coord.snapshot();
        engine.put(Partition::Node, b"dyn_k", b"v").expect("put");
        assert_eq!(
            coord
                .get(Partition::Node, b"dyn_k")
                .expect("dyn get")
                .as_deref(),
            Some(b"v".as_slice()),
        );
        // Snapshot-pinned read sees nothing — the put happened after
        // we captured `snap`.
        assert!(coord
            .snapshot_get(&snap, Partition::Node, b"dyn_k")
            .expect("dyn snap get")
            .is_none());
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
