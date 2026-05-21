//! [`BucketCatalog`] — per-shard in-memory open-bucket map.
//!
//! Sits above [`coordinode_modality::TimeSeriesStore`] and turns
//! single-measurement INSERTs into batched whole-bucket writes. See
//! the crate-level docs for the wider Slice A / B / C scope.

use std::collections::BTreeMap;
use std::sync::RwLock;
use std::time::SystemTime;

use coordinode_core::graph::node::NodeId;
use coordinode_modality::{Bucket, BucketControl, FieldStats, Measurement, TimeSeriesStore};

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
#[derive(Default, Debug)]
struct Stripe {
    open_buckets: std::collections::HashMap<BucketKey, OpenBucket>,
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
    /// directly today; Slice C will route them through the engine's
    /// node-id allocator (per-shard) so they coexist with regular
    /// graph nodes.
    next_node_id: std::sync::atomic::AtomicU64,
}

impl<'store, S: TimeSeriesStore> BucketCatalog<'store, S> {
    /// Construct an empty catalog bound to `store` for shard
    /// `shard_id`. `next_node_id_seed` is the starting node id for
    /// freshly-opened buckets — callers pass the shard's allocator
    /// tip so issued ids don't collide with regular graph nodes.
    pub fn new(
        config: CatalogConfig,
        shard_id: u16,
        store: &'store S,
        next_node_id_seed: u64,
    ) -> CatalogResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            shard_id,
            store,
            stripes: std::array::from_fn(|_| RwLock::new(Stripe::default())),
            next_node_id: std::sync::atomic::AtomicU64::new(next_node_id_seed),
        })
    }

    /// Ingest one measurement. Tier-1 in-buffer late-arrival
    /// absorption is automatic — measurements whose `timestamp_us`
    /// falls inside the current open bucket's window are appended
    /// and sorted on flush. Out-of-window measurements trigger a
    /// time-based rollover.
    ///
    /// Returns `Ok(())` on append (most paths); returns `Ok(())` on
    /// rollover-flush + re-open + append (also success). Errors
    /// propagate from the downstream store.
    pub fn write_measurement(
        &self,
        label_id: u16,
        meta: rmpv::Value,
        measurement: Measurement,
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

        // Decision against the existing bucket (if any).
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
            self.flush_bucket(&to_flush, &meta)?;
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
            drop(stripe);
            for (_key, bucket) in drained {
                let meta = bucket.meta.clone();
                self.flush_bucket(&bucket, &meta)?;
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

    /// Persist `bucket` to the underlying TimeSeriesStore and mark
    /// it closed. Sorts the in-memory buffer by timestamp before
    /// columnising — Tier-1 in-buffer late arrivals are merged here.
    fn flush_bucket(&self, bucket: &OpenBucket, meta: &rmpv::Value) -> CatalogResult<()> {
        let mut sorted = bucket.buffer.clone();
        sorted.sort_by_key(|m| m.timestamp_us);
        let body = Bucket::from_measurements(meta.clone(), sorted);
        self.store
            .put_bucket(self.shard_id, bucket.node_id, &body)?;
        self.store.mark_closed(self.shard_id, bucket.node_id)?;
        Ok(())
    }
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
        assert!(BucketCatalog::new(cfg, 0, &store, 1).is_err());
    }

    #[test]
    fn single_measurement_appends_without_flush() {
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);
        let cfg = CatalogConfig::arch_defaults();
        let catalog = BucketCatalog::new(cfg, 0, &store, 1).unwrap();

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
        let catalog = BucketCatalog::new(cfg, 0, &store, 1).unwrap();
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
        let catalog = BucketCatalog::new(cfg, 0, &store, 1).unwrap();
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
        let catalog = BucketCatalog::new(cfg, 0, &store, 1).unwrap();
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
        let catalog = BucketCatalog::new(cfg, 0, &store, 1).unwrap();
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
        let catalog = BucketCatalog::new(cfg, 0, &store, 1).unwrap();

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
        let catalog = BucketCatalog::new(cfg, 0, &store, 1).unwrap();

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

    #[test]
    fn concurrent_writers_distinct_keys_scale_across_stripes() {
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);
        let cfg = CatalogConfig {
            max_count: 1000,
            ..CatalogConfig::default()
        };
        let catalog = BucketCatalog::new(cfg, 0, &store, 1).unwrap();

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
