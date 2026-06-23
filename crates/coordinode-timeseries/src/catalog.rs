//! [`BucketCatalog`] — per-shard in-memory open-bucket map.
//!
//! Sits above [`coordinode_modality::TimeSeriesStore`] and turns
//! single-measurement INSERTs into batched whole-bucket writes. See
//! the crate-level docs for the wider Slice A / B / C scope.

use std::collections::{BTreeMap, HashMap};
use std::sync::RwLock;
use std::time::{Duration, SystemTime};

use coordinode_core::graph::node::NodeId;
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_modality::{Bucket, BucketControl, FieldStats, Measurement, TimeSeriesStore};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::transaction::{CommitContext, Transaction};

use crate::clock::IngestionClock;
use crate::config::{CatalogConfig, STRIPE_COUNT};
use crate::error::{CatalogError, CatalogResult};
use crate::key::BucketKey;
use crate::measurement_router::{route, Decision};

/// One open-bucket entry in the catalog. Carries the in-memory
/// buffer of measurements that haven't yet been flushed to the
/// underlying TimeSeriesStore, plus the running control block used
/// for rollover decisions.
#[derive(Debug)]
struct OpenBucket {
    /// The node id this bucket lives at once flushed. Allocated when
    /// the bucket opens.
    node_id: NodeId,
    /// In-memory measurement buffer. Sorted (by timestamp) on flush.
    buffer: Vec<Measurement>,
    /// Per-field stats + count + time bounds. Synced after every
    /// append so [`route`] can read it without scanning the buffer.
    control: BucketControl,
    /// Meta-field value shared by every measurement in this bucket.
    /// Carried so the flush can pass it to [`Bucket::from_measurements`].
    meta: rmpv::Value,
    /// Running byte-size estimate (see [`crate::measurement_router::route`]).
    size_estimate: u32,
    /// Wall-clock time the bucket was opened. Drives future age-
    /// based flushes (Slice C).
    #[allow(dead_code)]
    created_at: SystemTime,
}

/// One stripe of the catalog. Each stripe carries its own
/// `RwLock<HashMap>` so writers on disjoint stripes do not contend.
///
/// **Stripe ownership of `recently_closed`.** Each stripe owns its
/// own LRU of recently-closed buckets keyed by [`BucketKey`]. Keeping
/// the LRU per-stripe (rather than a single global one) preserves
/// the no-cross-stripe-coordination property of the catalog: a Tier
/// 2 re-open touches only its stripe's lock, identical to the
/// fast-path append.
#[derive(Default, Debug)]
struct Stripe {
    open_buckets: HashMap<BucketKey, OpenBucket>,
    recently_closed: RecentlyClosedLru,
}

/// Handle to a bucket that has been flushed and closed but is still
/// eligible for Tier-2 re-open. Holds enough state for the catalog
/// to recover the open-bucket form on re-open without re-reading
/// the bucket body from the underlying store.
#[derive(Debug, Clone)]
struct ClosedBucketHandle {
    /// Node id the bucket lives at in `Partition::Node`.
    node_id: NodeId,
    /// Wall-clock time the bucket flushed. Drives TTL eviction.
    closed_at: SystemTime,
    /// Inclusive event-time bounds the bucket covered when flushed.
    /// Used to decide whether an incoming late measurement falls
    /// within the bucket's window (Tier 2 candidate) or beyond it
    /// (Tier 3 territory, deferred to Slice C).
    time_range: (i64, i64),
    /// Meta-field value the bucket was keyed by. Carried so a
    /// re-open can pass it to subsequent [`Bucket::from_measurements`].
    meta: rmpv::Value,
    /// The bucket's schema as of close. A Tier-2 re-open with a
    /// schema-incompatible measurement falls through to a fresh
    /// rollover instead.
    fields: BTreeMap<String, FieldStats>,
}

/// Bounded, TTL-pruned map of recently-closed buckets — the catalog's
/// Tier-2 fast-path lookup.
///
/// **Capacity.** Hard cap at `MAX_RECENTLY_CLOSED` per stripe (the
/// arch spec sets the catalog-wide LRU at 10_000; with 32 stripes a
/// per-stripe cap of 512 gives the same total while keeping the
/// eviction scan bounded).
///
/// **TTL.** Entries expire at `closed_at + 2 × granularity_span` so
/// late-arrival absorption shrinks gracefully as a series falls
/// behind. The TTL is checked on read (`get_for_reopen`); stale
/// entries also get evicted on insert when the cap is reached.
///
/// **Eviction.** O(n) scan on insert overflow finds the entry with
/// the oldest `closed_at` and removes it. n ≤ 512 → typically <10µs
/// even on commodity hardware; the cap is sized so the scan is
/// negligible vs the per-write `put_bucket` cost.
#[derive(Default, Debug)]
struct RecentlyClosedLru {
    by_key: HashMap<BucketKey, ClosedBucketHandle>,
}

/// Per-stripe LRU capacity. 32 stripes × 512 = 16_384 entries
/// catalog-wide (slightly above the arch 10K target — slightly over
/// is harmless, slightly under means a hot series can lose Tier-2
/// eligibility under bursty churn).
const MAX_RECENTLY_CLOSED: usize = 512;

/// Overflow count above which [`BucketCatalog::compact_if_needed`]
/// triggers a [`TimeSeriesStore::compact_overflow`] call. Matches
/// the arch §Background merge default.
const OVERFLOW_COMPACT_THRESHOLD: usize = 50;

impl RecentlyClosedLru {
    /// Insert / replace the handle for `key`. Evicts the
    /// oldest-`closed_at` entry when at capacity.
    fn insert(&mut self, key: BucketKey, handle: ClosedBucketHandle) {
        if self.by_key.len() >= MAX_RECENTLY_CLOSED && !self.by_key.contains_key(&key) {
            // Find oldest by closed_at; remove it.
            if let Some(oldest_key) = self
                .by_key
                .iter()
                .min_by_key(|(_, h)| h.closed_at)
                .map(|(k, _)| *k)
            {
                self.by_key.remove(&oldest_key);
            }
        }
        self.by_key.insert(key, handle);
    }

    /// Look up the handle for `key`. Returns `None` when the entry
    /// is absent OR has aged past the TTL. The TTL check is done
    /// against `now` so callers can replay deterministic test
    /// clocks; production passes [`SystemTime::now`].
    fn get(&self, key: &BucketKey, now: SystemTime, ttl: Duration) -> Option<&ClosedBucketHandle> {
        let h = self.by_key.get(key)?;
        let age = now.duration_since(h.closed_at).ok()?;
        if age > ttl {
            return None;
        }
        Some(h)
    }

    /// Remove the handle for `key`. Used after a successful Tier-2
    /// reopen so the entry can't be replayed by a concurrent writer
    /// in the same flush window.
    fn remove(&mut self, key: &BucketKey) -> Option<ClosedBucketHandle> {
        self.by_key.remove(key)
    }

    /// Look up the handle WITHOUT the TTL check. Used by the Tier-3
    /// overflow path: a measurement that fails the Tier-2 reopen
    /// gate (because the bucket has aged past `2 × granularity_span`
    /// — re-open would be expensive, compaction already merged) can
    /// still route to that bucket's overflow segment via
    /// [`TimeSeriesStore::put_overflow`]. The LRU is the
    /// `(BucketKey → node_id)` discovery service for that path.
    fn get_any(&self, key: &BucketKey) -> Option<&ClosedBucketHandle> {
        self.by_key.get(key)
    }

    /// Diagnostic — number of currently-eligible handles. Tests only.
    #[cfg(test)]
    fn len(&self) -> usize {
        self.by_key.len()
    }
}

/// Per-shard time-series catalog. One instance per shard; CE
/// deployments use the shard's Raft leader as the single owner.
///
/// **Layer position.** Above [`coordinode_modality::TimeSeriesStore`]
/// (Layer 4), below the query executor that turns OpenCypher
/// `INSERT INTO TimeSeriesLabel` into a `write_measurement` call.
///
/// **Concurrency.** 32 independent stripes spread the lock load.
/// A single [`write_measurement`](BucketCatalog::write_measurement)
/// holds at most one stripe's write lock — no cross-stripe
/// coordination — so 32-way parallel writers see near-linear
/// scaling provided their bucket keys spread across stripes.
///
/// **Flush model (Slice A).** A rollover trigger inside
/// `write_measurement` causes the catalog to:
/// 1. Sort the bucket's buffer by `timestamp_us` (Tier-1 in-buffer
///    late-arrival absorption — out-of-order writes within the
///    window are merged in order),
/// 2. Build a [`Bucket`] body via [`Bucket::from_measurements`],
/// 3. Call [`TimeSeriesStore::put_bucket`] + `mark_closed`
///    (single store round-trip per flush),
/// 4. Drop the open-bucket entry and append the triggering
///    measurement to a fresh open bucket under the same key.
pub struct BucketCatalog<'store, S: TimeSeriesStore> {
    config: CatalogConfig,
    shard_id: u16,
    store: &'store S,
    /// Engine the catalog opens its short-lived bucket transactions
    /// against (ADR-041). The `store` is a stateless typed handle; the
    /// engine is what its buffered writes commit through.
    engine: &'store StorageEngine,
    stripes: [RwLock<Stripe>; STRIPE_COUNT],
    /// Monotonic source of bucket node ids. The catalog issues these
    /// directly today; a follow-up will route them through the
    /// engine's node-id allocator (per-shard) so they coexist with
    /// regular graph nodes.
    next_node_id: std::sync::atomic::AtomicU64,
    /// Monotonic source of overflow arrival seqnos. Catalog-local;
    /// each [`coordinode_modality::OverflowEntry`] gets a unique seqno per catalog
    /// lifetime. After [`Self::compact_if_needed`] runs the seqno
    /// space implicitly resets (the merged base bucket invalidates
    /// the prior overflow seqnos), but we keep the counter
    /// monotonic so a compactor that races a writer can't see
    /// duplicate seqnos.
    next_overflow_seqno: std::sync::atomic::AtomicU64,
    /// Engine-assigned ingestion-time stamp source (bitemporal axis
    /// per ADR-027). The catalog stamps every incoming measurement
    /// via `clock.next()` before buffering — production replicas
    /// share a Raft-leader-stamped clock; CE single-node uses
    /// [`crate::clock::MonotonicHlcClock`]; tests use `ScriptedClock`.
    clock: std::sync::Arc<dyn IngestionClock>,
}

impl<'store, S: TimeSeriesStore> BucketCatalog<'store, S> {
    /// Construct an empty catalog bound to `store` for shard
    /// `shard_id`. `next_node_id_seed` is the starting node id for
    /// freshly-opened buckets — callers pass the shard's allocator
    /// tip so issued ids don't collide with regular graph nodes.
    /// `clock` provides the bitemporal `__ingestion_ts__` stamp
    /// source (see [`crate::IngestionClock`]).
    pub fn new(
        config: CatalogConfig,
        shard_id: u16,
        store: &'store S,
        engine: &'store StorageEngine,
        next_node_id_seed: u64,
        clock: std::sync::Arc<dyn IngestionClock>,
    ) -> CatalogResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            shard_id,
            store,
            engine,
            stripes: std::array::from_fn(|_| RwLock::new(Stripe::default())),
            next_node_id: std::sync::atomic::AtomicU64::new(next_node_id_seed),
            next_overflow_seqno: std::sync::atomic::AtomicU64::new(1),
            clock,
        })
    }

    /// Open a short-lived MVCC transaction against the catalog's
    /// engine, run `body` (the store reads/writes for one logical
    /// bucket operation), then commit it. The catalog owns its bucket
    /// transaction boundaries (ADR-041): bucket bodies are
    /// point-overwrites, so a fresh per-operation transaction
    /// reproduces the prior immediate-write semantics while keeping
    /// multi-step operations atomic (e.g. compaction's base rewrite +
    /// overflow tombstones land in one commit or none). `body` receives
    /// `&Self` so it can reach `store` / `shard_id` / `clock` without
    /// re-borrowing the outer `self`.
    fn run_txn<R>(
        &self,
        body: impl FnOnce(&Self, &mut Transaction) -> CatalogResult<R>,
    ) -> CatalogResult<R> {
        let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
        let read_ts = oracle.next();
        let mut txn = Transaction::new(
            self.engine,
            Some(&oracle),
            read_ts,
            Some(self.engine.snapshot()),
        );
        let out = body(self, &mut txn)?;
        let wc = WriteConcern::majority();
        let ctx = CommitContext {
            write_concern: &wc,
            pipeline: None,
            id_gen: None,
            drain_buffer: None,
            nvme_write_buffer: None,
        };
        txn.commit(&ctx)
            .map_err(|e| CatalogError::Commit(e.to_string()))?;
        Ok(out)
    }

    /// Ingest one measurement. Tier-1 in-buffer late-arrival
    /// absorption is automatic — measurements whose `timestamp_us`
    /// falls inside the current open bucket's window are appended
    /// and sorted on flush. Out-of-window measurements trigger a
    /// time-based rollover.
    ///
    /// **Tier 2 re-open** (Slice B): when no open bucket exists for
    /// `(label_id, meta)` but a recently-closed handle is still in
    /// the stripe's LRU AND the measurement's `timestamp_us` falls
    /// within the handle's time range AND the measurement's fields
    /// are schema-compatible — the catalog reopens the bucket via
    /// [`TimeSeriesStore::reopen_bucket`] and restores it to the
    /// open-bucket map. Late-arrival absorption then proceeds as
    /// usual.
    ///
    /// **Event-time-only ingest** — no `__ingestion_ts__` stamping.
    /// Use this for labels declared as plain
    /// `CREATE LABEL X TIMESERIES` (the 95% IoT / monitoring /
    /// observability workload). Any caller-supplied
    /// `ingestion_ts_us` is overwritten to `None` so the engine-
    /// assigned-only invariant holds.
    ///
    /// Storage payoff: `Bucket::from_measurements` sees an all-None
    /// vec, clears the `ingestion_timestamps` column, and the
    /// per-measurement +8B overhead drops to ~1B msgpack tag per
    /// bucket. Per ADR-027 + ε-policy this is the default.
    ///
    /// Returns `Ok(())` on append (most paths); errors propagate
    /// from the downstream store.
    pub fn write_measurement(
        &self,
        label_id: u16,
        meta: rmpv::Value,
        measurement: Measurement,
    ) -> CatalogResult<()> {
        self.write_measurement_at(label_id, meta, measurement, SystemTime::now())
    }

    /// **Bitemporal ingest** — engine stamps `__ingestion_ts__` via
    /// the catalog's clock. Use this for labels declared as
    /// `CREATE LABEL X TIMESERIES WITH BITEMPORAL` (compliance,
    /// financial replay, ML reproducibility). Per ADR-027 the stamp
    /// is engine-assigned, never user-supplied — any caller value
    /// is overwritten.
    ///
    /// Resolves to the same write path as
    /// [`Self::write_measurement`] after stamping; the only
    /// difference is the populated `ingestion_ts_us`.
    pub fn write_measurement_bitemporal(
        &self,
        label_id: u16,
        meta: rmpv::Value,
        measurement: Measurement,
    ) -> CatalogResult<()> {
        self.write_measurement_bitemporal_at(label_id, meta, measurement, SystemTime::now())
    }

    /// Variant of [`Self::write_measurement`] that accepts an
    /// explicit `now` for the TTL check on the recently-closed LRU.
    /// Used by tests with a stub clock; production calls
    /// `write_measurement` which sources `SystemTime::now()`.
    pub fn write_measurement_at(
        &self,
        label_id: u16,
        meta: rmpv::Value,
        mut measurement: Measurement,
        now: SystemTime,
    ) -> CatalogResult<()> {
        // Event-time-only: ensure NO ingestion stamp leaks through.
        // The engine-assigned-only invariant means even a
        // caller-supplied value is wiped.
        measurement.ingestion_ts_us = None;

        self.write_measurement_inner(label_id, meta, measurement, now)
    }

    /// Bitemporal variant — stamps `ingestion_ts_us` via the clock
    /// before routing through the shared inner path. See
    /// [`Self::write_measurement_bitemporal`] for the semantic
    /// contract.
    pub fn write_measurement_bitemporal_at(
        &self,
        label_id: u16,
        meta: rmpv::Value,
        mut measurement: Measurement,
        now: SystemTime,
    ) -> CatalogResult<()> {
        // Engine-assigned stamp — overwrites caller value per
        // ADR-027 "never user-supplied".
        measurement.ingestion_ts_us = Some(self.clock.next());

        self.write_measurement_inner(label_id, meta, measurement, now)
    }

    /// Shared routing logic — stripe lock acquisition, Tier 1/2/3
    /// decision tree, buffer append. Identical between event-time
    /// and bitemporal paths; the only difference (presence of
    /// `ingestion_ts_us`) is decided by the public entry point
    /// that called us.
    fn write_measurement_inner(
        &self,
        label_id: u16,
        meta: rmpv::Value,
        measurement: Measurement,
        now: SystemTime,
    ) -> CatalogResult<()> {
        let key = BucketKey::from_meta(label_id, &meta);
        let stripe_idx = key.stripe_idx();
        // Poisoned-lock recovery: a poisoned write lock means a
        // prior writer panicked mid-mutation. The state is salvageable
        // (catalog never partially commits a flush) so we recover
        // into_inner() rather than propagating the panic.
        let mut stripe = self.stripes[stripe_idx]
            .write()
            .unwrap_or_else(|p| p.into_inner());

        // Decision against the existing open bucket (if any).
        let needs_rollover = match stripe.open_buckets.get(&key) {
            Some(bucket) => match route(
                &bucket.control,
                bucket.size_estimate,
                &measurement,
                &self.config,
            ) {
                Decision::Append => false,
                Decision::Rollover(reason) => {
                    tracing::debug!(
                        ?reason,
                        label_id,
                        meta_hash = format!("{:#x}", key.meta_hash),
                        "bucket rollover triggered",
                    );
                    true
                }
            },
            None => false, // No bucket yet — fall through to open.
        };

        if needs_rollover {
            let to_flush = stripe
                .open_buckets
                .remove(&key)
                .ok_or(CatalogError::InvalidConfig(
                    "bucket missing during rollover",
                ))?;
            let handle = self.flush_bucket(&to_flush, &meta)?;
            stripe.recently_closed.insert(key, handle);
        }

        // Tier 2 fast-path: no open bucket, but a recently-closed
        // handle matches by key AND time range AND schema.
        if !stripe.open_buckets.contains_key(&key) {
            let ttl = self.config.granularity_span * 2;
            let candidate = stripe.recently_closed.get(&key, now, ttl).and_then(|h| {
                if measurement_fits_recently_closed(h, &measurement, self.config.granularity_span) {
                    Some(h.clone())
                } else {
                    None
                }
            });
            if let Some(handle) = candidate {
                tracing::debug!(
                    label_id,
                    meta_hash = format!("{:#x}", key.meta_hash),
                    node_id = handle.node_id.as_raw(),
                    "Tier-2 reopen",
                );
                let reopened = self.run_txn(|c, txn| {
                    Ok(c.store.reopen_bucket(txn, c.shard_id, handle.node_id)?)
                })?;
                if reopened {
                    // Take the handle out of LRU — the bucket is open again.
                    stripe.recently_closed.remove(&key);
                    let bucket_control = BucketControl {
                        version: 1,
                        count: 0,
                        time_min_us: handle.time_range.0,
                        time_max_us: handle.time_range.1,
                        closed: false,
                        fields_stats: handle.fields.clone(),
                    };
                    stripe.open_buckets.insert(
                        key,
                        OpenBucket {
                            node_id: handle.node_id,
                            buffer: Vec::with_capacity(self.config.max_count as usize),
                            control: bucket_control,
                            meta: handle.meta.clone(),
                            size_estimate: 0,
                            created_at: SystemTime::now(),
                        },
                    );
                }
            }
        }

        // Tier 3 overflow: no open bucket, AND Tier 2 didn't fire
        // (either no in-TTL handle, or the in-TTL handle's range
        // didn't contain the measurement). If a TTL-expired handle
        // for this key DOES contain the measurement's timestamp,
        // route to the bucket's overflow segment instead of opening
        // a fresh bucket.
        //
        // The overflow path covers genuinely late data (sensor
        // offline for hours / retroactive corrections) — too old
        // for a cheap re-open, but identifiable by the LRU's
        // surviving `(BucketKey → node_id)` mapping. Truly orphaned
        // late data (LRU evicted the handle entirely) still surfaces
        // as [`CatalogError::LateBeyondTier1`] for the caller to
        // route or reject.
        if !stripe.open_buckets.contains_key(&key) {
            let overflow_target = stripe.recently_closed.get_any(&key).and_then(|h| {
                if measurement_fits_recently_closed(h, &measurement, self.config.granularity_span) {
                    Some((h.node_id, h.meta.clone()))
                } else {
                    None
                }
            });
            if let Some((node_id, _meta)) = overflow_target {
                let arrival_seqno = self
                    .next_overflow_seqno
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                tracing::debug!(
                    label_id,
                    meta_hash = format!("{:#x}", key.meta_hash),
                    node_id = node_id.as_raw(),
                    arrival_seqno,
                    "Tier-3 overflow",
                );
                drop(stripe); // release lock before store I/O
                let entry = coordinode_modality::OverflowEntry {
                    arrival_seqno,
                    measurement,
                };
                self.run_txn(|c, txn| {
                    c.store
                        .put_overflow(txn, u32::from(label_id), node_id, &entry)?;
                    Ok(())
                })?;
                return Ok(());
            }
        }

        // Append (or open fresh).
        let bucket = stripe.open_buckets.entry(key).or_insert_with(|| {
            let node_id = NodeId::from_raw(
                self.next_node_id
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            );
            OpenBucket {
                node_id,
                buffer: Vec::with_capacity(self.config.max_count as usize),
                control: BucketControl {
                    version: 1,
                    count: 0,
                    time_min_us: i64::MAX,
                    time_max_us: i64::MIN,
                    closed: false,
                    fields_stats: BTreeMap::new(),
                },
                meta: meta.clone(),
                size_estimate: 0,
                created_at: SystemTime::now(),
            }
        });

        bucket.size_estimate = bucket
            .size_estimate
            .saturating_add(estimate_measurement_size(&measurement));
        bucket.control.count = bucket.control.count.saturating_add(1);
        bucket.control.time_min_us = bucket.control.time_min_us.min(measurement.timestamp_us);
        bucket.control.time_max_us = bucket.control.time_max_us.max(measurement.timestamp_us);
        for (name, value) in &measurement.fields {
            let stats = bucket
                .control
                .fields_stats
                .entry(name.clone())
                .or_insert(FieldStats {
                    min: *value,
                    max: *value,
                });
            stats.min = stats.min.min(*value);
            stats.max = stats.max.max(*value);
        }
        bucket.buffer.push(measurement);
        Ok(())
    }

    /// Flush every open bucket. Used at graceful shutdown and from
    /// integration tests that want to assert post-flush state.
    /// Errors short-circuit at the first failed flush — partial
    /// progress is fine because each `put_bucket` is independent.
    pub fn flush_all(&self) -> CatalogResult<()> {
        for stripe_lock in &self.stripes {
            let mut stripe = stripe_lock.write().unwrap_or_else(|p| p.into_inner());
            let drained: Vec<(BucketKey, OpenBucket)> = stripe.open_buckets.drain().collect();
            for (key, bucket) in drained {
                let meta = bucket.meta.clone();
                let handle = self.flush_bucket(&bucket, &meta)?;
                stripe.recently_closed.insert(key, handle);
            }
        }
        Ok(())
    }

    /// Number of currently-open buckets across all stripes. Diagnostic
    /// hook — production code does not read this. Tests use it to
    /// pin "rollover removed the open entry".
    pub fn open_bucket_count(&self) -> usize {
        self.stripes
            .iter()
            .map(|s| s.read().map(|g| g.open_buckets.len()).unwrap_or(0))
            .sum()
    }

    /// Number of currently-tracked recently-closed bucket handles
    /// across all stripes (Tier-2 LRU). Diagnostic only.
    pub fn recently_closed_count(&self) -> usize {
        self.stripes
            .iter()
            .map(|s| {
                s.read()
                    .map(|g| g.recently_closed.by_key.len())
                    .unwrap_or(0)
            })
            .sum()
    }

    /// Inspect the overflow set for `(label_id, bucket_id)` and run
    /// [`TimeSeriesStore::compact_overflow`] when it has grown past
    /// the configured count threshold. Returns `Ok(true)` if a
    /// compaction ran, `Ok(false)` if not.
    ///
    /// Threshold: more than `OVERFLOW_COMPACT_THRESHOLD`
    /// (50) overflow entries triggers compaction. The catalog does
    /// NOT enforce an age threshold here — call sites that care
    /// about "compact stale overflow" run their own age check and
    /// invoke this hook unconditionally.
    ///
    /// **Driver model.** This method is the *trigger primitive*.
    /// Production runs it from a periodic background driver (one
    /// per shard); tests invoke it directly. The catalog itself
    /// does not spawn any threads — driver wiring lives above.
    pub fn compact_if_needed(&self, label_id: u32, bucket_id: NodeId) -> CatalogResult<bool> {
        // One transaction spans the read (scan overflow + base bucket)
        // and the write (rewrite base + tombstone overflow), so the
        // compaction is atomic — a crash cannot leave overflow visible
        // against an already-rewritten base.
        self.run_txn(|c, txn| {
            let overflow = c.store.scan_overflow(txn, label_id, bucket_id)?;
            if overflow.len() <= OVERFLOW_COMPACT_THRESHOLD {
                return Ok(false);
            }
            let Some(base) = c.store.get_bucket(txn, c.shard_id, bucket_id)? else {
                tracing::warn!(
                    label_id,
                    bucket_id = bucket_id.as_raw(),
                    "compact_if_needed: base bucket missing — leaving overflow in place",
                );
                return Ok(false);
            };
            let mut merged: Vec<Measurement> = base.measurements().collect();
            let mut overflow_seqnos: Vec<u64> = Vec::with_capacity(overflow.len());
            for entry in overflow {
                overflow_seqnos.push(entry.arrival_seqno);
                merged.push(entry.measurement);
            }
            merged.sort_by_key(|m| m.timestamp_us);
            // Bitemporal backfill: a base bucket written before the
            // `__ingestion_ts__` axis landed has measurements with
            // `ingestion_ts_us = None`. After compaction the bucket
            // becomes the new authoritative copy; we backfill the missing
            // stamps with `c.clock.next()` so the merged bucket carries
            // a complete ingestion_timestamps column and stays queryable
            // under `AS OF INGESTION_TIME`. Without backfill,
            // `Bucket::from_measurements` would see a half-stamped vec
            // and CLEAR the ingestion_timestamps column to prevent
            // misalignment — losing the bitemporal axis on every legacy
            // bucket touched by compaction.
            for m in &mut merged {
                if m.ingestion_ts_us.is_none() {
                    m.ingestion_ts_us = Some(c.clock.next());
                }
            }
            let merged_bucket = Bucket::from_measurements(base.meta.clone(), merged);
            c.store.compact_overflow(
                txn,
                c.shard_id,
                label_id,
                bucket_id,
                &merged_bucket,
                &overflow_seqnos,
            )?;
            tracing::debug!(
                label_id,
                bucket_id = bucket_id.as_raw(),
                compacted = overflow_seqnos.len(),
                "overflow compacted",
            );
            Ok(true)
        })
    }

    /// Background compactor driver primitive: discover every bucket
    /// with at least one overflow entry and run
    /// [`Self::compact_if_needed`] against each. Returns the count
    /// of buckets that actually compacted (i.e. were above
    /// `OVERFLOW_COMPACT_THRESHOLD`).
    ///
    /// Production wires this into a periodic loop above the catalog
    /// (a `tokio::time::interval` task per shard owns the schedule;
    /// the catalog itself never spawns threads). Tests invoke
    /// directly to verify the discover + compact cycle end to end.
    ///
    /// The discover step uses
    /// [`TimeSeriesStore::list_overflow_buckets`], which is a single
    /// prefix scan of `ts_overflow:` folded to unique
    /// `(label_id, bucket_id)` pairs — cost scales with the number of
    /// stale buckets, not the overflow volume.
    pub fn compact_all_pending(&self) -> CatalogResult<usize> {
        let candidates = self.run_txn(|c, txn| Ok(c.store.list_overflow_buckets(txn)?))?;
        let mut compacted = 0usize;
        for (label_id, bucket_id) in candidates {
            if self.compact_if_needed(label_id, bucket_id)? {
                compacted += 1;
            }
        }
        Ok(compacted)
    }

    /// Persist `bucket` to the underlying TimeSeriesStore and mark
    /// it closed. Sorts the in-memory buffer by timestamp before
    /// columnising — Tier-1 in-buffer late arrivals are merged here.
    /// Returns a [`ClosedBucketHandle`] the caller stores in the
    /// stripe's [`RecentlyClosedLru`].
    fn flush_bucket(
        &self,
        bucket: &OpenBucket,
        meta: &rmpv::Value,
    ) -> CatalogResult<ClosedBucketHandle> {
        let mut sorted = bucket.buffer.clone();
        sorted.sort_by_key(|m| m.timestamp_us);
        let body = Bucket::from_measurements(meta.clone(), sorted);
        // Persist the bucket body and close it in one transaction.
        self.run_txn(|c, txn| {
            c.store.put_bucket(txn, c.shard_id, bucket.node_id, &body)?;
            c.store.mark_closed(txn, c.shard_id, bucket.node_id)?;
            Ok(())
        })?;
        Ok(ClosedBucketHandle {
            node_id: bucket.node_id,
            closed_at: SystemTime::now(),
            time_range: (bucket.control.time_min_us, bucket.control.time_max_us),
            meta: meta.clone(),
            fields: bucket.control.fields_stats.clone(),
        })
    }
}

/// Tier-2 / Tier-3 eligibility check: is the incoming `measurement`
/// part of the closed bucket described by `handle`?
///
/// Two conditions must hold:
/// 1. `timestamp_us` falls STRICTLY within the bucket's stored time
///    range `[time_min, time_max]`. A measurement past the bucket's
///    max belongs to a *later* bucket, never the closed one —
///    routing it to overflow would corrupt the time-series story.
/// 2. The measurement's fields are a SUBSET of the bucket's schema.
///    A reopen with a new column would force a schema rollover
///    immediately after — pointless round-trip — so fall through.
///
/// The `granularity_span` parameter is no longer used for slack
/// (see commit history — the prior slack was so generous that
/// any measurement within one granularity of the bucket's max
/// was deemed "in window", which incorrectly routed
/// post-rollover sequential measurements to the closed bucket's
/// overflow). Kept for signature stability so future tweaks can
/// reintroduce a small-fixed-duration slack if needed.
fn measurement_fits_recently_closed(
    handle: &ClosedBucketHandle,
    measurement: &Measurement,
    _granularity_span: Duration,
) -> bool {
    if measurement.timestamp_us < handle.time_range.0 {
        return false;
    }
    if measurement.timestamp_us > handle.time_range.1 {
        return false;
    }
    if handle.fields.is_empty() {
        return true;
    }
    measurement
        .fields
        .keys()
        .all(|name| handle.fields.contains_key(name))
}

fn estimate_measurement_size(m: &Measurement) -> u32 {
    let mut size: u32 = 8;
    for name in m.fields.keys() {
        size = size.saturating_add(name.len() as u32);
        size = size.saturating_add(16);
    }
    size
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
