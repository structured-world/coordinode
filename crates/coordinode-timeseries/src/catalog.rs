//! [`BucketCatalog`] — per-shard in-memory open-bucket map.
//!
//! Sits above [`coordinode_modality::TimeSeriesStore`] and turns
//! single-measurement INSERTs into batched whole-bucket writes. See
//! the crate-level docs for the wider Slice A / B / C scope.

use std::collections::{BTreeMap, HashMap};
use std::sync::RwLock;
use std::time::{Duration, SystemTime};

use coordinode_core::graph::node::NodeId;
use coordinode_modality::{Bucket, BucketControl, FieldStats, Measurement, TimeSeriesStore};

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
    /// Running byte-size estimate (see [`measurement_router::route`]).
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
    /// Node id the bucket lives at in [`Partition::Node`].
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
    stripes: [RwLock<Stripe>; STRIPE_COUNT],
    /// Monotonic source of bucket node ids. The catalog issues these
    /// directly today; a follow-up will route them through the
    /// engine's node-id allocator (per-shard) so they coexist with
    /// regular graph nodes.
    next_node_id: std::sync::atomic::AtomicU64,
    /// Monotonic source of overflow arrival seqnos. Catalog-local;
    /// each [`OverflowEntry`] gets a unique seqno per catalog
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
    /// [`MonotonicHlcClock`]; tests use `ScriptedClock`.
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
        next_node_id_seed: u64,
        clock: std::sync::Arc<dyn IngestionClock>,
    ) -> CatalogResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            shard_id,
            store,
            stripes: std::array::from_fn(|_| RwLock::new(Stripe::default())),
            next_node_id: std::sync::atomic::AtomicU64::new(next_node_id_seed),
            next_overflow_seqno: std::sync::atomic::AtomicU64::new(1),
            clock,
        })
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
                let reopened = self.store.reopen_bucket(self.shard_id, handle.node_id)?;
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
                self.store.put_overflow(
                    u32::from(label_id),
                    node_id,
                    &coordinode_modality::OverflowEntry {
                        arrival_seqno,
                        measurement,
                    },
                )?;
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
    /// Threshold (arch default — see `arch/core/timeseries.md`
    /// §Background merge): more than `OVERFLOW_COMPACT_THRESHOLD`
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
        let overflow = self.store.scan_overflow(label_id, bucket_id)?;
        if overflow.len() <= OVERFLOW_COMPACT_THRESHOLD {
            return Ok(false);
        }
        // Read the current base bucket; merge with overflow
        // measurements; write back atomically (compact_overflow
        // builds the merge under a WriteBatch covering both the
        // base put and the overflow tombstones).
        let Some(base) = self.store.get_bucket(self.shard_id, bucket_id)? else {
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
        // stamps with `self.clock.next()` so the merged bucket carries
        // a complete ingestion_timestamps column and stays queryable
        // under `AS OF INGESTION_TIME`. Without backfill,
        // `Bucket::from_measurements` would see a half-stamped vec
        // and CLEAR the ingestion_timestamps column to prevent
        // misalignment — losing the bitemporal axis on every legacy
        // bucket touched by compaction.
        for m in &mut merged {
            if m.ingestion_ts_us.is_none() {
                m.ingestion_ts_us = Some(self.clock.next());
            }
        }
        let merged_bucket = Bucket::from_measurements(base.meta.clone(), merged);
        self.store.compact_overflow(
            self.shard_id,
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
        let candidates = self.store.list_overflow_buckets()?;
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
        self.store
            .put_bucket(self.shard_id, bucket.node_id, &body)?;
        self.store.mark_closed(self.shard_id, bucket.node_id)?;
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
mod tests {
    use super::*;
    use coordinode_modality::LocalTimeSeriesStore;
    use coordinode_storage::engine::config::{
        Durability, EndpointConfig, Media, StorageConfig, Tier,
    };
    use coordinode_storage::engine::core::StorageEngine;
    use std::time::Duration;
    use tempfile::TempDir;

    fn mk_engine() -> (TempDir, StorageEngine) {
        let dir = TempDir::new().unwrap();
        let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "ep",
            dir.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        let engine = StorageEngine::open(&cfg).unwrap();
        (dir, engine)
    }

    fn measurement(ts_us: i64, fields: &[(&str, f64)]) -> Measurement {
        let mut m = BTreeMap::new();
        for (k, v) in fields {
            m.insert((*k).to_string(), *v);
        }
        Measurement {
            timestamp_us: ts_us,
            ingestion_ts_us: None,
            fields: m,
        }
    }

    #[test]
    fn invalid_config_rejected_at_construction() {
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);
        let cfg = CatalogConfig {
            max_count: 0,
            ..CatalogConfig::default()
        };
        assert!(BucketCatalog::new(
            cfg,
            0,
            &store,
            1,
            std::sync::Arc::new(crate::clock::MonotonicHlcClock::new())
        )
        .is_err());
    }

    #[test]
    fn single_measurement_appends_without_flush() {
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);
        let cfg = CatalogConfig::arch_defaults();
        let catalog = BucketCatalog::new(
            cfg,
            0,
            &store,
            1,
            std::sync::Arc::new(crate::clock::MonotonicHlcClock::new()),
        )
        .unwrap();

        catalog
            .write_measurement(
                7,
                rmpv::Value::String("s1".into()),
                measurement(100, &[("temp", 22.5)]),
            )
            .unwrap();

        // One open bucket, nothing flushed yet.
        assert_eq!(catalog.open_bucket_count(), 1);
        // Store should NOT have a persisted bucket yet — open bucket
        // lives in catalog memory only.
        let persisted = store
            .get_bucket(0, NodeId::from_raw(1))
            .expect("get_bucket");
        assert!(persisted.is_none(), "no put_bucket should have fired yet");
    }

    #[test]
    fn count_rollover_flushes_and_reopens() {
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);
        let cfg = CatalogConfig {
            max_count: 3,
            ..CatalogConfig::default()
        };
        let catalog = BucketCatalog::new(
            cfg,
            0,
            &store,
            1,
            std::sync::Arc::new(crate::clock::MonotonicHlcClock::new()),
        )
        .unwrap();
        let meta = rmpv::Value::String("s1".into());

        // 3 measurements fill the bucket without rollover.
        for i in 0..3 {
            catalog
                .write_measurement(7, meta.clone(), measurement(i * 1000, &[("temp", 20.0)]))
                .unwrap();
        }
        assert_eq!(catalog.open_bucket_count(), 1);

        // 4th measurement triggers Count rollover: the existing
        // bucket flushes (node_id 1) and a fresh one opens (node_id 2).
        catalog
            .write_measurement(7, meta, measurement(3000, &[("temp", 21.0)]))
            .unwrap();
        assert_eq!(catalog.open_bucket_count(), 1, "post-rollover, one open");

        // Flushed bucket persisted at node_id 1 with 3 measurements.
        let persisted = store
            .get_bucket(0, NodeId::from_raw(1))
            .expect("get_bucket")
            .expect("bucket exists");
        assert_eq!(persisted.control.count, 3);
        assert!(persisted.control.closed, "flushed bucket is closed");
    }

    #[test]
    fn out_of_order_measurements_sorted_on_flush() {
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);
        let cfg = CatalogConfig {
            max_count: 4,
            ..CatalogConfig::default()
        };
        let catalog = BucketCatalog::new(
            cfg,
            0,
            &store,
            1,
            std::sync::Arc::new(crate::clock::MonotonicHlcClock::new()),
        )
        .unwrap();
        let meta = rmpv::Value::String("s1".into());

        // Insert out-of-order: 100, 50, 200, 25 (Tier-1 absorption).
        for ts in &[100i64, 50, 200, 25] {
            catalog
                .write_measurement(7, meta.clone(), measurement(*ts, &[("temp", 20.0)]))
                .unwrap();
        }
        // Force flush — buffer holds 4 measurements; max_count = 4
        // so any next write would roll over, but we want to inspect
        // BEFORE rollover. Use flush_all.
        catalog.flush_all().unwrap();
        let persisted = store.get_bucket(0, NodeId::from_raw(1)).unwrap().unwrap();
        assert_eq!(persisted.timestamps, vec![25, 50, 100, 200]);
    }

    #[test]
    fn time_rollover_when_measurement_outside_granularity() {
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);
        let cfg = CatalogConfig {
            granularity_span: Duration::from_millis(10),
            ..CatalogConfig::default()
        };
        let catalog = BucketCatalog::new(
            cfg,
            0,
            &store,
            1,
            std::sync::Arc::new(crate::clock::MonotonicHlcClock::new()),
        )
        .unwrap();
        let meta = rmpv::Value::String("s1".into());

        // First measurement at t=0, second at t=20ms (>10ms granularity).
        catalog
            .write_measurement(7, meta.clone(), measurement(0, &[("temp", 20.0)]))
            .unwrap();
        catalog
            .write_measurement(7, meta, measurement(20_000, &[("temp", 21.0)]))
            .unwrap();

        // Second write must have triggered Time rollover: bucket 1
        // flushed with 1 measurement, bucket 2 opened with 1.
        let flushed = store.get_bucket(0, NodeId::from_raw(1)).unwrap().unwrap();
        assert_eq!(flushed.control.count, 1);
        assert_eq!(flushed.timestamps, vec![0]);
        assert_eq!(catalog.open_bucket_count(), 1);
    }

    #[test]
    fn schema_rollover_when_new_field_introduced() {
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);
        let cfg = CatalogConfig::arch_defaults();
        let catalog = BucketCatalog::new(
            cfg,
            0,
            &store,
            1,
            std::sync::Arc::new(crate::clock::MonotonicHlcClock::new()),
        )
        .unwrap();
        let meta = rmpv::Value::String("s1".into());

        // Open with {temp}, then write {temp, humidity}.
        catalog
            .write_measurement(7, meta.clone(), measurement(0, &[("temp", 20.0)]))
            .unwrap();
        catalog
            .write_measurement(
                7,
                meta,
                measurement(1000, &[("temp", 21.0), ("humidity", 60.0)]),
            )
            .unwrap();

        let flushed = store.get_bucket(0, NodeId::from_raw(1)).unwrap().unwrap();
        assert_eq!(flushed.control.count, 1);
        assert!(flushed.control.fields_stats.contains_key("temp"));
        assert!(!flushed.control.fields_stats.contains_key("humidity"));
    }

    #[test]
    fn distinct_meta_values_open_distinct_buckets() {
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);
        let cfg = CatalogConfig::arch_defaults();
        let catalog = BucketCatalog::new(
            cfg,
            0,
            &store,
            1,
            std::sync::Arc::new(crate::clock::MonotonicHlcClock::new()),
        )
        .unwrap();

        catalog
            .write_measurement(
                7,
                rmpv::Value::String("s1".into()),
                measurement(100, &[("temp", 20.0)]),
            )
            .unwrap();
        catalog
            .write_measurement(
                7,
                rmpv::Value::String("s2".into()),
                measurement(100, &[("temp", 21.0)]),
            )
            .unwrap();
        assert_eq!(
            catalog.open_bucket_count(),
            2,
            "distinct meta values → distinct buckets",
        );
    }

    #[test]
    fn flush_all_drains_every_stripe() {
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);
        let cfg = CatalogConfig::arch_defaults();
        let catalog = BucketCatalog::new(
            cfg,
            0,
            &store,
            1,
            std::sync::Arc::new(crate::clock::MonotonicHlcClock::new()),
        )
        .unwrap();

        for i in 0..10 {
            catalog
                .write_measurement(
                    7,
                    rmpv::Value::String(format!("s{i}").into()),
                    measurement(100, &[("temp", 20.0)]),
                )
                .unwrap();
        }
        assert!(catalog.open_bucket_count() >= 1);
        catalog.flush_all().unwrap();
        assert_eq!(catalog.open_bucket_count(), 0);
    }

    // -- Tier-2 reopen path --

    #[test]
    fn tier2_reopen_brings_back_recently_closed_bucket() {
        // Flush a bucket via explicit flush_all (no follow-up open
        // bucket), then write a late measurement at a timestamp
        // INSIDE the closed bucket's time range. The Tier-2 path
        // should re-open the same bucket (same node_id) rather
        // than allocate a new one.
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);
        let cfg = CatalogConfig {
            max_count: 100,
            granularity_span: Duration::from_secs(3600),
            ..CatalogConfig::default()
        };
        let catalog = BucketCatalog::new(
            cfg,
            0,
            &store,
            1,
            std::sync::Arc::new(crate::clock::MonotonicHlcClock::new()),
        )
        .unwrap();
        let meta = rmpv::Value::String("s1".into());

        // Buffer 3 measurements at t=0, 1000, 2000, then flush_all.
        for i in 0..3 {
            catalog
                .write_measurement(7, meta.clone(), measurement(i * 1000, &[("temp", 20.0)]))
                .unwrap();
        }
        catalog.flush_all().unwrap();
        // Now bucket 1 closed (node_id 1, time_range [0, 2000]).
        // No open bucket. 1 handle in recently_closed.
        assert_eq!(catalog.open_bucket_count(), 0);
        assert_eq!(catalog.recently_closed_count(), 1);

        // Pre-Tier-2 store state: bucket 1 closed in store.
        let pre = store.get_bucket(0, NodeId::from_raw(1)).unwrap().unwrap();
        assert!(pre.control.closed, "bucket 1 closed in store pre-reopen");

        // Write a late measurement at ts=500 (inside [0, 2000]
        // window). Tier-2 reopen must fire — the store's bucket 1
        // flips to closed=false.
        catalog
            .write_measurement(7, meta.clone(), measurement(500, &[("temp", 22.0)]))
            .unwrap();

        // After reopen:
        // - recently_closed shrinks by 1 (handle taken)
        // - one open bucket (the re-opened one) at node_id 1
        // - no fresh node_id allocated (next_node_id_seed was 1
        //   pre-flush, advanced to 2 when bucket 1 opened, and we
        //   reused node 1 not 2)
        assert_eq!(catalog.recently_closed_count(), 0);
        assert_eq!(catalog.open_bucket_count(), 1);
    }

    #[test]
    fn tier2_skipped_when_measurement_outside_time_range() {
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);
        let cfg = CatalogConfig {
            max_count: 3,
            granularity_span: Duration::from_secs(3600),
            ..CatalogConfig::default()
        };
        let catalog = BucketCatalog::new(
            cfg,
            0,
            &store,
            1,
            std::sync::Arc::new(crate::clock::MonotonicHlcClock::new()),
        )
        .unwrap();
        let meta = rmpv::Value::String("s1".into());

        for i in 0..3 {
            catalog
                .write_measurement(7, meta.clone(), measurement(i * 1000, &[("temp", 20.0)]))
                .unwrap();
        }
        // Rollover.
        catalog
            .write_measurement(
                7,
                meta.clone(),
                measurement(10_000_000_000, &[("temp", 21.0)]),
            )
            .unwrap();

        // Now bucket 1 closed (range 0..2000). Bucket 2 open at
        // 10s. A subsequent measurement at 9.9s should NOT reopen
        // bucket 1 — it lands in bucket 2's window (Tier-1
        // absorption against the open bucket).
        catalog
            .write_measurement(
                7,
                meta.clone(),
                measurement(9_900_000_000, &[("temp", 22.0)]),
            )
            .unwrap();
        // recently_closed unchanged (no reopen happened).
        assert_eq!(catalog.recently_closed_count(), 1);
    }

    #[test]
    fn tier2_skipped_when_schema_drifts() {
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);
        let cfg = CatalogConfig {
            max_count: 3,
            granularity_span: Duration::from_secs(3600),
            ..CatalogConfig::default()
        };
        let catalog = BucketCatalog::new(
            cfg,
            0,
            &store,
            1,
            std::sync::Arc::new(crate::clock::MonotonicHlcClock::new()),
        )
        .unwrap();
        let meta = rmpv::Value::String("s1".into());

        for i in 0..3 {
            catalog
                .write_measurement(7, meta.clone(), measurement(i * 1000, &[("temp", 20.0)]))
                .unwrap();
        }
        // Rollover at t=10s.
        catalog
            .write_measurement(7, meta.clone(), measurement(10_000, &[("temp", 21.0)]))
            .unwrap();

        // Late measurement at t=500 (inside bucket 1 range) BUT
        // with a new field 'humidity' — incompatible schema. Tier-2
        // must NOT fire; the measurement goes to bucket 2 instead.
        catalog
            .write_measurement(
                7,
                meta.clone(),
                measurement(500, &[("temp", 22.0), ("humidity", 60.0)]),
            )
            .unwrap();

        assert_eq!(
            catalog.recently_closed_count(),
            1,
            "recently_closed unchanged when schema drift skips Tier 2",
        );
    }

    #[test]
    fn tier2_ttl_expires_handle() {
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);
        let cfg = CatalogConfig {
            max_count: 2,
            granularity_span: Duration::from_millis(50),
            ..CatalogConfig::default()
        };
        let catalog = BucketCatalog::new(
            cfg,
            0,
            &store,
            1,
            std::sync::Arc::new(crate::clock::MonotonicHlcClock::new()),
        )
        .unwrap();
        let meta = rmpv::Value::String("s1".into());

        // Close bucket 1.
        for i in 0..2 {
            catalog
                .write_measurement(7, meta.clone(), measurement(i * 100, &[("temp", 20.0)]))
                .unwrap();
        }
        catalog
            .write_measurement(7, meta.clone(), measurement(10_000, &[("temp", 21.0)]))
            .unwrap();
        // recently_closed has bucket 1.
        assert_eq!(catalog.recently_closed_count(), 1);

        // Now write at a time well past the TTL (2× 50ms = 100ms).
        // Use write_measurement_at to fake a clock 1s in the future.
        let future = SystemTime::now() + Duration::from_secs(1);
        catalog
            .write_measurement_at(7, meta.clone(), measurement(50, &[("temp", 22.0)]), future)
            .unwrap();

        // The TTL check rejected the handle — Tier 2 did NOT fire.
        // recently_closed still holds the (stale) handle since we
        // don't proactively evict.
        assert_eq!(catalog.recently_closed_count(), 1);
    }

    #[test]
    fn lru_evicts_oldest_on_capacity_overflow() {
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);
        let cfg = CatalogConfig {
            max_count: 1,
            granularity_span: Duration::from_secs(3600),
            ..CatalogConfig::default()
        };
        let catalog = BucketCatalog::new(
            cfg,
            0,
            &store,
            1,
            std::sync::Arc::new(crate::clock::MonotonicHlcClock::new()),
        )
        .unwrap();

        // Force enough flushes on ONE STRIPE to exceed MAX_RECENTLY_CLOSED.
        // Find a label_id whose bucket key lands in stripe 0 for a
        // range of meta values, then fill stripe 0 past capacity.
        let target_stripe = 0usize;
        let mut accepted = 0u64;
        let mut meta_index = 0u64;
        while accepted < (MAX_RECENTLY_CLOSED as u64 + 4) {
            let meta = rmpv::Value::String(format!("meta-{meta_index}").into());
            let key = BucketKey::from_meta(7, &meta);
            if key.stripe_idx() == target_stripe {
                catalog
                    .write_measurement(7, meta.clone(), measurement(0, &[("temp", 20.0)]))
                    .unwrap();
                // Trigger rollover with a second write at a later time.
                catalog
                    .write_measurement(7, meta, measurement(1_000_000, &[("temp", 21.0)]))
                    .unwrap();
                accepted += 1;
            }
            meta_index += 1;
        }

        // Stripe 0's recently_closed must be capped — newer entries
        // pushed older ones out.
        let stripe0 = catalog.stripes[target_stripe].read().unwrap();
        assert_eq!(stripe0.recently_closed.len(), MAX_RECENTLY_CLOSED);
    }

    // -- Tier-3 overflow + compactor --

    #[test]
    fn tier3_routes_to_overflow_when_tier2_ttl_expired() {
        // Flush a bucket, force its handle past TTL via fake-clock
        // `write_measurement_at`. Tier 2 skips on TTL; Tier 3 picks
        // up the same handle (no TTL check) and routes to overflow.
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);
        let cfg = CatalogConfig {
            max_count: 2,
            granularity_span: Duration::from_millis(50),
            ..CatalogConfig::default()
        };
        let catalog = BucketCatalog::new(
            cfg,
            0,
            &store,
            1,
            std::sync::Arc::new(crate::clock::MonotonicHlcClock::new()),
        )
        .unwrap();
        let meta = rmpv::Value::String("s1".into());

        // Close bucket 1 with measurements at t=0, 1ms.
        for i in 0..2 {
            catalog
                .write_measurement(7, meta.clone(), measurement(i * 1000, &[("temp", 20.0)]))
                .unwrap();
        }
        catalog.flush_all().unwrap();
        assert_eq!(catalog.recently_closed_count(), 1);
        assert!(store
            .scan_overflow(7, NodeId::from_raw(1))
            .unwrap()
            .is_empty());

        // Late measurement in-window (ts=500us inside [0, 1000])
        // but `now` is far past TTL → Tier 2 skips → Tier 3 should
        // route to overflow against bucket 1.
        let future = SystemTime::now() + Duration::from_secs(1);
        catalog
            .write_measurement_at(7, meta.clone(), measurement(500, &[("temp", 22.0)]), future)
            .unwrap();

        // Overflow has one entry under bucket 1.
        let overflow = store.scan_overflow(7, NodeId::from_raw(1)).unwrap();
        assert_eq!(overflow.len(), 1);
        assert_eq!(overflow[0].measurement.timestamp_us, 500);
        // No new open bucket allocated for the key.
        assert_eq!(catalog.open_bucket_count(), 0);
    }

    #[test]
    fn tier3_skipped_when_handle_time_range_does_not_contain_measurement() {
        // TTL-expired handle exists for the key but the late
        // measurement timestamp is OUTSIDE the handle's range.
        // Tier 3 must NOT route to that bucket's overflow — instead
        // a fresh bucket opens for the new timestamp.
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);
        let cfg = CatalogConfig {
            max_count: 2,
            granularity_span: Duration::from_millis(50),
            ..CatalogConfig::default()
        };
        let catalog = BucketCatalog::new(
            cfg,
            0,
            &store,
            1,
            std::sync::Arc::new(crate::clock::MonotonicHlcClock::new()),
        )
        .unwrap();
        let meta = rmpv::Value::String("s1".into());

        for i in 0..2 {
            catalog
                .write_measurement(7, meta.clone(), measurement(i * 1000, &[("temp", 20.0)]))
                .unwrap();
        }
        catalog.flush_all().unwrap();
        // Bucket 1 covers [0, 1000]. Now write a measurement far in
        // the future (ts=10s) past TTL → Tier 3 must not target
        // bucket 1's overflow (out of range). A fresh bucket 2 opens.
        let future = SystemTime::now() + Duration::from_secs(1);
        catalog
            .write_measurement_at(7, meta, measurement(10_000_000, &[("temp", 22.0)]), future)
            .unwrap();

        assert!(store
            .scan_overflow(7, NodeId::from_raw(1))
            .unwrap()
            .is_empty());
        assert_eq!(catalog.open_bucket_count(), 1);
    }

    #[test]
    fn compact_if_needed_no_op_below_threshold() {
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);
        let cfg = CatalogConfig::arch_defaults();
        let catalog = BucketCatalog::new(
            cfg,
            0,
            &store,
            1,
            std::sync::Arc::new(crate::clock::MonotonicHlcClock::new()),
        )
        .unwrap();

        // Empty overflow → no compaction, returns Ok(false).
        let did = catalog.compact_if_needed(7, NodeId::from_raw(1)).unwrap();
        assert!(!did);
    }

    #[test]
    fn compact_if_needed_merges_overflow_into_base_when_above_threshold() {
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);
        let cfg = CatalogConfig {
            max_count: 2,
            granularity_span: Duration::from_millis(50),
            ..CatalogConfig::default()
        };
        let catalog = BucketCatalog::new(
            cfg,
            0,
            &store,
            1,
            std::sync::Arc::new(crate::clock::MonotonicHlcClock::new()),
        )
        .unwrap();
        let meta = rmpv::Value::String("s1".into());

        // Set up: bucket 1 closed with 2 measurements at t=0, 1ms.
        for i in 0..2 {
            catalog
                .write_measurement(7, meta.clone(), measurement(i * 1000, &[("temp", 20.0)]))
                .unwrap();
        }
        catalog.flush_all().unwrap();

        // Force 51 overflow entries (> OVERFLOW_COMPACT_THRESHOLD=50).
        let future = SystemTime::now() + Duration::from_secs(1);
        for i in 0..51 {
            catalog
                .write_measurement_at(
                    7,
                    meta.clone(),
                    measurement(500 + i, &[("temp", 30.0 + (i as f64))]),
                    future,
                )
                .unwrap();
        }
        assert_eq!(
            store.scan_overflow(7, NodeId::from_raw(1)).unwrap().len(),
            51,
        );

        // Compact.
        let did = catalog.compact_if_needed(7, NodeId::from_raw(1)).unwrap();
        assert!(did, "compact must run when overflow exceeds threshold");

        // Post-compact: overflow set empty, base bucket has 2 + 51 = 53
        // measurements (merged + sorted).
        assert!(store
            .scan_overflow(7, NodeId::from_raw(1))
            .unwrap()
            .is_empty());
        let base = store.get_bucket(0, NodeId::from_raw(1)).unwrap().unwrap();
        assert_eq!(base.control.count, 53);
        // First two are the originals (0, 1000); next 51 are the
        // overflow entries (500..551). Sorted: 0, 500..551, 1000.
        assert_eq!(base.timestamps[0], 0);
        assert_eq!(base.timestamps[52], 1000);
    }

    #[test]
    fn compact_all_pending_discovers_and_compacts_every_stale_bucket() {
        // Seed two distinct closed buckets each with > 50 overflow
        // entries (above the OVERFLOW_COMPACT_THRESHOLD). The
        // background driver primitive must discover both via
        // `list_overflow_buckets` and compact each one.
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);
        let cfg = CatalogConfig {
            max_count: 2,
            granularity_span: Duration::from_millis(50),
            ..CatalogConfig::default()
        };
        let catalog = BucketCatalog::new(
            cfg,
            0,
            &store,
            1,
            std::sync::Arc::new(crate::clock::MonotonicHlcClock::new()),
        )
        .unwrap();

        // Bucket A under meta "sA": close + 51 overflow.
        let meta_a = rmpv::Value::String("sA".into());
        for i in 0..2 {
            catalog
                .write_measurement(7, meta_a.clone(), measurement(i * 1000, &[("v", 1.0)]))
                .unwrap();
        }
        catalog.flush_all().unwrap();
        let future = SystemTime::now() + Duration::from_secs(1);
        for i in 0..51 {
            catalog
                .write_measurement_at(
                    7,
                    meta_a.clone(),
                    measurement(500 + i, &[("v", 30.0 + (i as f64))]),
                    future,
                )
                .unwrap();
        }

        // Bucket B under meta "sB": close + 51 overflow.
        let meta_b = rmpv::Value::String("sB".into());
        for i in 0..2 {
            catalog
                .write_measurement(7, meta_b.clone(), measurement(i * 1000, &[("v", 2.0)]))
                .unwrap();
        }
        catalog.flush_all().unwrap();
        for i in 0..51 {
            catalog
                .write_measurement_at(
                    7,
                    meta_b.clone(),
                    measurement(500 + i, &[("v", 60.0 + (i as f64))]),
                    future,
                )
                .unwrap();
        }

        // Bucket C: only 10 overflow entries (below threshold —
        // must NOT compact).
        let meta_c = rmpv::Value::String("sC".into());
        for i in 0..2 {
            catalog
                .write_measurement(7, meta_c.clone(), measurement(i * 1000, &[("v", 3.0)]))
                .unwrap();
        }
        catalog.flush_all().unwrap();
        for i in 0..10 {
            catalog
                .write_measurement_at(
                    7,
                    meta_c.clone(),
                    measurement(500 + i, &[("v", 90.0 + (i as f64))]),
                    future,
                )
                .unwrap();
        }

        // Discover-and-compact: exactly 2 buckets compacted (A, B);
        // C's overflow stays in place.
        let compacted = catalog.compact_all_pending().unwrap();
        assert_eq!(compacted, 2, "A and B compacted, C below threshold");

        // Post-compact: A's and B's overflow sets empty; C's intact.
        assert!(store
            .scan_overflow(7, NodeId::from_raw(1))
            .unwrap()
            .is_empty());
        assert!(store
            .scan_overflow(7, NodeId::from_raw(2))
            .unwrap()
            .is_empty());
        assert_eq!(
            store.scan_overflow(7, NodeId::from_raw(3)).unwrap().len(),
            10,
        );
    }

    #[test]
    fn write_measurement_stamps_ingestion_ts_via_clock() {
        // Bitemporal axis (sub-system #3): catalog stamps every
        // incoming measurement with the clock's next() value before
        // buffering. Subsequent flush + read-back surfaces the
        // stamp via Bucket.ingestion_timestamps. Test pins the
        // sequence via ScriptedClock so we know exactly which
        // stamps land on which measurement.
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);
        let cfg = CatalogConfig {
            max_count: 3,
            granularity_span: Duration::from_secs(3600),
            ..CatalogConfig::default()
        };
        let clock = std::sync::Arc::new(crate::clock::ScriptedClock::new(vec![
            1_000_000, 2_000_000, 3_000_000,
        ]));
        let catalog = BucketCatalog::new(cfg, 0, &store, 1, clock).unwrap();
        let meta = rmpv::Value::String("sensor-1".into());

        catalog
            .write_measurement_bitemporal(7, meta.clone(), measurement(100, &[("temp", 20.0)]))
            .unwrap();
        catalog
            .write_measurement_bitemporal(7, meta.clone(), measurement(200, &[("temp", 21.0)]))
            .unwrap();
        catalog
            .write_measurement_bitemporal(7, meta.clone(), measurement(300, &[("temp", 22.0)]))
            .unwrap();
        catalog.flush_all().unwrap();

        // Read back via the store, verify ingestion-ts column has
        // exactly the three stamps in the order the catalog assigned.
        let bucket = store
            .get_bucket(0, NodeId::from_raw(1))
            .unwrap()
            .expect("bucket persisted");
        assert_eq!(bucket.control.count, 3);
        assert_eq!(
            bucket.ingestion_timestamps,
            vec![1_000_000, 2_000_000, 3_000_000],
            "ingestion-ts column must carry the ScriptedClock sequence",
        );
        // Event-time and ingestion-time columns must be the same
        // length (column alignment invariant).
        assert_eq!(
            bucket.timestamps.len(),
            bucket.ingestion_timestamps.len(),
            "ingestion_timestamps must be the same length as timestamps",
        );
    }

    #[test]
    fn write_measurement_ignores_caller_supplied_ingestion_ts() {
        // Per ADR-027: `__ingestion_ts__` is engine-assigned, NEVER
        // user-supplied. Even if the caller pre-sets `ingestion_ts_us`
        // on the Measurement struct, the catalog MUST overwrite it
        // with the clock value — otherwise a malicious client could
        // backdate writes and break `AS OF INGESTION_TIME` semantics.
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);
        let cfg = CatalogConfig::arch_defaults();
        let clock = std::sync::Arc::new(crate::clock::ScriptedClock::new(vec![999]));
        let catalog = BucketCatalog::new(cfg, 0, &store, 1, clock).unwrap();
        let meta = rmpv::Value::String("sensor".into());

        // Caller tries to claim ingestion_ts = 1 (far past) on a
        // bitemporal label — engine must overwrite via clock.
        let mut malicious_m = measurement(100, &[("v", 1.0)]);
        malicious_m.ingestion_ts_us = Some(1);
        catalog
            .write_measurement_bitemporal(7, meta.clone(), malicious_m)
            .unwrap();
        catalog.flush_all().unwrap();

        // Read back: engine MUST have overwritten to the clock's 999.
        let bucket = store
            .get_bucket(0, NodeId::from_raw(1))
            .unwrap()
            .expect("bucket persisted");
        assert_eq!(
            bucket.ingestion_timestamps,
            vec![999],
            "engine must overwrite caller-supplied ingestion_ts with clock value",
        );
    }

    #[test]
    fn non_bitemporal_writes_leave_ingestion_column_empty_for_storage_saving() {
        // ε-policy: `write_measurement` (event-time-only entry point)
        // produces buckets with an empty `ingestion_timestamps` vec,
        // even if the caller pre-set `ingestion_ts_us`. This is the
        // storage payoff — non-bitemporal labels (95% TS workloads)
        // pay zero per-measurement overhead for the bitemporal axis.
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);
        let cfg = CatalogConfig::arch_defaults();
        let clock = std::sync::Arc::new(crate::clock::ScriptedClock::new(vec![1, 2, 3]));
        let catalog = BucketCatalog::new(cfg, 0, &store, 1, clock).unwrap();
        let meta = rmpv::Value::String("iot-sensor".into());

        // Caller even pre-stamps — engine still wipes (engine-assigned-only).
        let mut m1 = measurement(100, &[("temp", 22.5)]);
        m1.ingestion_ts_us = Some(99_999);
        let m2 = measurement(200, &[("temp", 22.7)]);
        let m3 = measurement(300, &[("temp", 22.4)]);

        catalog.write_measurement(7, meta.clone(), m1).unwrap();
        catalog.write_measurement(7, meta.clone(), m2).unwrap();
        catalog.write_measurement(7, meta.clone(), m3).unwrap();
        catalog.flush_all().unwrap();

        let bucket = store
            .get_bucket(0, NodeId::from_raw(1))
            .unwrap()
            .expect("bucket persisted");
        assert_eq!(bucket.timestamps.len(), 3, "event-time column populated");
        assert!(
            bucket.ingestion_timestamps.is_empty(),
            "non-bitemporal label produces empty ingestion column — \
             storage saving by construction (ε-policy)",
        );
        // Iterator yields None for every row — confirms read-path
        // observes the absence correctly.
        for m in bucket.measurements() {
            assert_eq!(
                m.ingestion_ts_us, None,
                "non-bitemporal bucket reads back ingestion_ts_us = None",
            );
        }
    }

    #[test]
    fn non_bitemporal_overflow_path_leaves_stamp_none() {
        // Tier-3 overflow path under event-time-only mode: caller
        // pre-stamp wiped, overflow row stored with None.
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);
        let cfg = CatalogConfig {
            max_count: 2,
            granularity_span: Duration::from_millis(50),
            ..CatalogConfig::default()
        };
        let clock = std::sync::Arc::new(crate::clock::ScriptedClock::new(vec![1, 2, 3]));
        let catalog = BucketCatalog::new(cfg, 0, &store, 1, clock).unwrap();
        let meta = rmpv::Value::String("s".into());

        // Open + flush a bucket via the event-time-only path.
        for i in 0..2 {
            catalog
                .write_measurement(7, meta.clone(), measurement(i * 1000, &[("v", 1.0)]))
                .unwrap();
        }
        catalog.flush_all().unwrap();

        // Late write — routes to Tier-3 overflow when LRU TTL has expired.
        let far_future = SystemTime::now() + Duration::from_secs(3600);
        let mut malicious = measurement(500, &[("v", 99.0)]);
        malicious.ingestion_ts_us = Some(42); // caller tries to stamp
        catalog
            .write_measurement_at(7, meta.clone(), malicious, far_future)
            .unwrap();

        // Tier-3 overflow row must have ingestion_ts_us = None
        // (engine wiped, even on the overflow path).
        let entries = store.scan_overflow(7, NodeId::from_raw(1)).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0].measurement.ingestion_ts_us, None,
            "Tier-3 overflow under event-time-only mode must NOT carry an ingestion stamp",
        );
    }

    #[test]
    fn legacy_bucket_without_ingestion_column_decodes_as_none() {
        // Backward compat: a bucket persisted by a pre-bitemporal
        // engine has NO `ingestion_timestamps` field on the wire.
        // `#[serde(default)]` decodes it as `Vec::new()`. The
        // measurements iterator must then yield `ingestion_ts_us = None`
        // for every row, not panic or yield zeros.
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);

        // Hand-build a v1-shape bucket (no ingestion_timestamps column)
        // via Bucket::from_measurements with all Nones — verifies the
        // half-stamped-clearing branch produces a wire-compatible
        // empty column.
        let m = |ts: i64| measurement(ts, &[("v", 1.0)]);
        let legacy_bucket =
            Bucket::from_measurements(rmpv::Value::String("legacy".into()), vec![m(100), m(200)]);
        assert!(
            legacy_bucket.ingestion_timestamps.is_empty(),
            "all-None input must produce empty ingestion column (legacy shape on wire)",
        );
        store
            .put_bucket(0, NodeId::from_raw(42), &legacy_bucket)
            .unwrap();

        // Round-trip through the store.
        let read_back = store.get_bucket(0, NodeId::from_raw(42)).unwrap().unwrap();
        assert!(
            read_back.ingestion_timestamps.is_empty(),
            "legacy bucket round-trip preserves empty ingestion column",
        );
        // Iterator must yield None for every measurement.
        for ms in read_back.measurements() {
            assert_eq!(
                ms.ingestion_ts_us, None,
                "legacy bucket iterator must yield ingestion_ts_us=None",
            );
        }
    }

    #[test]
    fn compact_preserves_ingestion_axis_for_fully_stamped_data() {
        // Bitemporal happy-path: base bucket fully stamped, overflow
        // entries fully stamped, post-compact merged bucket carries
        // every original stamp (no backfill needed, all stamps from
        // the original writer's clock).
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);
        let cfg = CatalogConfig {
            max_count: 2,
            granularity_span: Duration::from_millis(50),
            ..CatalogConfig::default()
        };
        // ScriptedClock with enough stamps for: 2 base writes + 51
        // overflow writes + backfill safety margin.
        let stamps: Vec<i64> = (1_000_000..1_000_100).collect();
        let clock = std::sync::Arc::new(crate::clock::ScriptedClock::new(stamps.clone()));
        let catalog = BucketCatalog::new(cfg, 0, &store, 1, clock).unwrap();
        let meta = rmpv::Value::String("s".into());

        // 2 base measurements + flush (closes bucket).
        catalog
            .write_measurement(7, meta.clone(), measurement(0, &[("v", 1.0)]))
            .unwrap();
        catalog
            .write_measurement(7, meta.clone(), measurement(1000, &[("v", 2.0)]))
            .unwrap();
        catalog.flush_all().unwrap();

        // 51 overflow entries above threshold — every one stamped by
        // the catalog's clock before routing to overflow.
        let future = SystemTime::now() + Duration::from_secs(1);
        for i in 0..51 {
            catalog
                .write_measurement_at(
                    7,
                    meta.clone(),
                    measurement(500 + i, &[("v", 30.0 + (i as f64))]),
                    future,
                )
                .unwrap();
        }

        // Compact.
        assert!(catalog.compact_if_needed(7, NodeId::from_raw(1)).unwrap());

        // Post-compact: read back, every measurement still has Some
        // ingestion_ts. Column length matches event-time column.
        let merged = store.get_bucket(0, NodeId::from_raw(1)).unwrap().unwrap();
        assert_eq!(merged.timestamps.len(), 53);
        assert_eq!(
            merged.ingestion_timestamps.len(),
            53,
            "every measurement keeps its stamp through compaction",
        );
        // Every stamp must be from the original write (1_000_000..)
        // — backfill code path must NOT fire for fully-stamped data.
        for its in &merged.ingestion_timestamps {
            assert!(
                *its >= 1_000_000 && *its < 1_000_100,
                "stamp {its} is outside the ScriptedClock range — backfill fired \
                 unnecessarily on fully-stamped data",
            );
        }
    }

    #[test]
    fn compact_backfills_legacy_base_bucket_via_clock() {
        // Edge case: a legacy (pre-bitemporal) base bucket on disk
        // with `ingestion_timestamps.is_empty()`. Catalog accepts
        // new overflow writes (stamped). compact_if_needed runs
        // merge — the backfill loop fills every base measurement's
        // missing `ingestion_ts_us` with `clock.next()` so the
        // merged bucket carries a full ingestion_timestamps column.
        // Without backfill, `Bucket::from_measurements` would clear
        // the column (half-stamped → drop) and lose the bitemporal
        // axis on every legacy bucket touched.
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);

        // Plant a legacy bucket directly (no catalog write — bypasses
        // the stamping path). 5 measurements, no ingestion column.
        let legacy_bucket = Bucket::from_measurements(
            rmpv::Value::String("s".into()),
            (0..5)
                .map(|i| measurement(i * 100, &[("v", i as f64)]))
                .collect(),
        );
        assert!(
            legacy_bucket.ingestion_timestamps.is_empty(),
            "plant precondition: legacy bucket has empty ingestion column",
        );
        store
            .put_bucket(0, NodeId::from_raw(1), &legacy_bucket)
            .unwrap();
        // Mark closed manually so compact_if_needed proceeds.
        store.mark_closed(0, NodeId::from_raw(1)).unwrap();

        // Set up catalog with a deterministic clock. Stamps:
        //   5 backfills (for legacy base) + 51 overflow writes
        //   = 56 stamps minimum.
        let stamps: Vec<i64> = (5_000_000..5_000_100).collect();
        let clock = std::sync::Arc::new(crate::clock::ScriptedClock::new(stamps));
        let cfg = CatalogConfig {
            max_count: 2,
            granularity_span: Duration::from_millis(50),
            ..CatalogConfig::default()
        };
        let catalog = BucketCatalog::new(cfg, 0, &store, 100, clock).unwrap();

        // 51 overflow writes — manually since the catalog doesn't
        // know about the planted bucket. Route them directly via
        // store.put_overflow so they target bucket 1.
        for i in 0..51 {
            store
                .put_overflow(
                    7,
                    NodeId::from_raw(1),
                    &coordinode_modality::OverflowEntry {
                        arrival_seqno: i as u64 + 1,
                        measurement: Measurement {
                            timestamp_us: 500 + i as i64,
                            // Overflow measurements pre-stamped (as if
                            // catalog had stamped them in production).
                            ingestion_ts_us: Some(9_000_000 + i as i64),
                            fields: {
                                let mut f = BTreeMap::new();
                                f.insert("v".to_string(), 100.0 + (i as f64));
                                f
                            },
                        },
                    },
                )
                .unwrap();
        }

        // Compact — must backfill the 5 legacy measurements with
        // clock stamps, keep the 51 overflow stamps verbatim.
        assert!(catalog.compact_if_needed(7, NodeId::from_raw(1)).unwrap());

        // Verify: merged bucket has 56 measurements, all stamped.
        let merged = store.get_bucket(0, NodeId::from_raw(1)).unwrap().unwrap();
        assert_eq!(merged.timestamps.len(), 56, "5 base + 51 overflow");
        assert_eq!(
            merged.ingestion_timestamps.len(),
            56,
            "every measurement carries a stamp post-compact (no half-stamped clearing)",
        );
        // Count how many stamps came from the ScriptedClock vs
        // overflow-supplied. Exactly 5 should be from ScriptedClock
        // (the backfill for the legacy base), 51 from overflow.
        let mut backfilled = 0;
        let mut overflow_preserved = 0;
        for its in &merged.ingestion_timestamps {
            if (5_000_000..5_000_100).contains(its) {
                backfilled += 1;
            } else if (9_000_000..9_000_100).contains(its) {
                overflow_preserved += 1;
            } else {
                panic!("unexpected stamp {its} — not in backfill or overflow ranges");
            }
        }
        assert_eq!(backfilled, 5, "5 legacy base measurements backfilled");
        assert_eq!(
            overflow_preserved, 51,
            "51 overflow stamps preserved verbatim through compaction",
        );
    }

    #[test]
    fn overflow_path_preserves_ingestion_stamp() {
        // Tier-3 routing: a measurement that misses Tier-1/Tier-2
        // and routes to overflow MUST still carry the stamp the
        // catalog assigned. Verify by writing far-late, then
        // reading back via scan_overflow.
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);
        let cfg = CatalogConfig {
            max_count: 2,
            granularity_span: Duration::from_millis(50),
            ..CatalogConfig::default()
        };
        let clock = std::sync::Arc::new(crate::clock::ScriptedClock::new(vec![
            777_000, 888_000, 999_000,
        ]));
        let catalog = BucketCatalog::new(cfg, 0, &store, 1, clock).unwrap();
        let meta = rmpv::Value::String("s".into());

        // Open + flush a bucket so the LRU has a handle. Bitemporal
        // path so the foundation Tier-3 stamp test still exercises
        // the stamping route.
        for i in 0..2 {
            catalog
                .write_measurement_bitemporal(7, meta.clone(), measurement(i * 1000, &[("v", 1.0)]))
                .unwrap();
        }
        catalog.flush_all().unwrap();
        // ScriptedClock has burned 2 stamps. The next overflow write
        // gets stamp 999_000.

        // Late write — fits in the closed bucket's window so it routes
        // to Tier-3 overflow once the LRU TTL has expired.
        let far_future = SystemTime::now() + Duration::from_secs(3600);
        catalog
            .write_measurement_bitemporal_at(
                7,
                meta.clone(),
                measurement(500, &[("v", 99.0)]),
                far_future,
            )
            .unwrap();

        // Inspect the overflow row directly.
        let entries = store.scan_overflow(7, NodeId::from_raw(1)).unwrap();
        assert_eq!(entries.len(), 1, "Tier-3 routed the late write to overflow");
        assert_eq!(
            entries[0].measurement.ingestion_ts_us,
            Some(999_000),
            "Tier-3 overflow entry must carry the catalog-assigned stamp",
        );
    }

    #[test]
    fn compact_all_pending_no_op_when_overflow_set_empty() {
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);
        let cfg = CatalogConfig::arch_defaults();
        let catalog = BucketCatalog::new(
            cfg,
            0,
            &store,
            1,
            std::sync::Arc::new(crate::clock::MonotonicHlcClock::new()),
        )
        .unwrap();
        assert_eq!(catalog.compact_all_pending().unwrap(), 0);
    }

    #[test]
    fn concurrent_writers_distinct_keys_scale_across_stripes() {
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);
        let cfg = CatalogConfig {
            max_count: 1000,
            ..CatalogConfig::default()
        };
        let catalog = BucketCatalog::new(
            cfg,
            0,
            &store,
            1,
            std::sync::Arc::new(crate::clock::MonotonicHlcClock::new()),
        )
        .unwrap();

        // Scoped threads so workers can borrow `catalog` (which
        // itself borrows `store`) without the 'static bound that
        // `thread::spawn` requires.
        std::thread::scope(|s| {
            for w in 0..8 {
                let cat = &catalog;
                s.spawn(move || {
                    for i in 0..50 {
                        cat.write_measurement(
                            7,
                            rmpv::Value::String(format!("worker-{w}-key-{i}").into()),
                            Measurement {
                                timestamp_us: i64::from(i),
                                ingestion_ts_us: None,
                                fields: {
                                    let mut f = BTreeMap::new();
                                    f.insert("v".to_string(), 1.0);
                                    f
                                },
                            },
                        )
                        .unwrap();
                    }
                });
            }
        });
        // 8 workers × 50 distinct keys = 400 distinct (label, meta_hash)
        // pairs, all open in catalog memory until flush_all.
        assert_eq!(catalog.open_bucket_count(), 400);
        catalog.flush_all().unwrap();
        assert_eq!(catalog.open_bucket_count(), 0);
    }
}
