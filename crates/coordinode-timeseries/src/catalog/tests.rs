use super::*;
use coordinode_modality::LocalTimeSeriesStore;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
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

/// Run direct `TimeSeriesStore` writes (test setup / assertions
/// that bypass the catalog) in one committed MVCC transaction.
fn ts_write<R>(
    engine: &StorageEngine,
    body: impl FnOnce(&LocalTimeSeriesStore, &mut Transaction) -> R,
) -> R {
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let read_ts = oracle.next();
    let mut txn = Transaction::new(engine, Some(&oracle), read_ts, Some(engine.snapshot()));
    let out = body(&LocalTimeSeriesStore, &mut txn);
    let wc = WriteConcern::majority();
    let ctx = CommitContext {
        write_concern: &wc,
        pipeline: None,
        id_gen: None,
        drain_buffer: None,
        nvme_write_buffer: None,
    };
    txn.commit(&ctx).expect("commit ts test write");
    out
}

/// Run a direct `TimeSeriesStore` read against the latest committed
/// snapshot.
fn ts_read<R>(
    engine: &StorageEngine,
    body: impl FnOnce(&LocalTimeSeriesStore, &Transaction) -> R,
) -> R {
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let read_ts = oracle.next();
    let txn = Transaction::new(engine, Some(&oracle), read_ts, Some(engine.snapshot()));
    body(&LocalTimeSeriesStore, &txn)
}

#[test]
fn invalid_config_rejected_at_construction() {
    let (_dir, engine) = mk_engine();
    let store = LocalTimeSeriesStore;
    let cfg = CatalogConfig {
        max_count: 0,
        ..CatalogConfig::default()
    };
    assert!(BucketCatalog::new(
        cfg,
        0,
        &store,
        &engine,
        1,
        std::sync::Arc::new(crate::clock::MonotonicHlcClock::new())
    )
    .is_err());
}

#[test]
fn single_measurement_appends_without_flush() {
    let (_dir, engine) = mk_engine();
    let store = LocalTimeSeriesStore;
    let cfg = CatalogConfig::arch_defaults();
    let catalog = BucketCatalog::new(
        cfg,
        0,
        &store,
        &engine,
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
    let persisted = ts_read(&engine, |s, txn| {
        s.get_bucket(txn, 0, NodeId::from_raw(1))
            .expect("get_bucket")
    });
    assert!(persisted.is_none(), "no put_bucket should have fired yet");
}

#[test]
fn count_rollover_flushes_and_reopens() {
    let (_dir, engine) = mk_engine();
    let store = LocalTimeSeriesStore;
    let cfg = CatalogConfig {
        max_count: 3,
        ..CatalogConfig::default()
    };
    let catalog = BucketCatalog::new(
        cfg,
        0,
        &store,
        &engine,
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
    let persisted = ts_read(&engine, |s, txn| {
        s.get_bucket(txn, 0, NodeId::from_raw(1))
            .expect("get_bucket")
    })
    .expect("bucket exists");
    assert_eq!(persisted.control.count, 3);
    assert!(persisted.control.closed, "flushed bucket is closed");
}

#[test]
fn out_of_order_measurements_sorted_on_flush() {
    let (_dir, engine) = mk_engine();
    let store = LocalTimeSeriesStore;
    let cfg = CatalogConfig {
        max_count: 4,
        ..CatalogConfig::default()
    };
    let catalog = BucketCatalog::new(
        cfg,
        0,
        &store,
        &engine,
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
    let persisted = ts_read(&engine, |s, txn| {
        s.get_bucket(txn, 0, NodeId::from_raw(1)).unwrap()
    })
    .unwrap();
    assert_eq!(persisted.timestamps, vec![25, 50, 100, 200]);
}

#[test]
fn time_rollover_when_measurement_outside_granularity() {
    let (_dir, engine) = mk_engine();
    let store = LocalTimeSeriesStore;
    let cfg = CatalogConfig {
        granularity_span: Duration::from_millis(10),
        ..CatalogConfig::default()
    };
    let catalog = BucketCatalog::new(
        cfg,
        0,
        &store,
        &engine,
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
    let flushed = ts_read(&engine, |s, txn| {
        s.get_bucket(txn, 0, NodeId::from_raw(1)).unwrap()
    })
    .unwrap();
    assert_eq!(flushed.control.count, 1);
    assert_eq!(flushed.timestamps, vec![0]);
    assert_eq!(catalog.open_bucket_count(), 1);
}

#[test]
fn schema_rollover_when_new_field_introduced() {
    let (_dir, engine) = mk_engine();
    let store = LocalTimeSeriesStore;
    let cfg = CatalogConfig::arch_defaults();
    let catalog = BucketCatalog::new(
        cfg,
        0,
        &store,
        &engine,
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

    let flushed = ts_read(&engine, |s, txn| {
        s.get_bucket(txn, 0, NodeId::from_raw(1)).unwrap()
    })
    .unwrap();
    assert_eq!(flushed.control.count, 1);
    assert!(flushed.control.fields_stats.contains_key("temp"));
    assert!(!flushed.control.fields_stats.contains_key("humidity"));
}

#[test]
fn distinct_meta_values_open_distinct_buckets() {
    let (_dir, engine) = mk_engine();
    let store = LocalTimeSeriesStore;
    let cfg = CatalogConfig::arch_defaults();
    let catalog = BucketCatalog::new(
        cfg,
        0,
        &store,
        &engine,
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
    let store = LocalTimeSeriesStore;
    let cfg = CatalogConfig::arch_defaults();
    let catalog = BucketCatalog::new(
        cfg,
        0,
        &store,
        &engine,
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
    let store = LocalTimeSeriesStore;
    let cfg = CatalogConfig {
        max_count: 100,
        granularity_span: Duration::from_secs(3600),
        ..CatalogConfig::default()
    };
    let catalog = BucketCatalog::new(
        cfg,
        0,
        &store,
        &engine,
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
    let pre = ts_read(&engine, |s, txn| {
        s.get_bucket(txn, 0, NodeId::from_raw(1)).unwrap()
    })
    .unwrap();
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
    let store = LocalTimeSeriesStore;
    let cfg = CatalogConfig {
        max_count: 3,
        granularity_span: Duration::from_secs(3600),
        ..CatalogConfig::default()
    };
    let catalog = BucketCatalog::new(
        cfg,
        0,
        &store,
        &engine,
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
    let store = LocalTimeSeriesStore;
    let cfg = CatalogConfig {
        max_count: 3,
        granularity_span: Duration::from_secs(3600),
        ..CatalogConfig::default()
    };
    let catalog = BucketCatalog::new(
        cfg,
        0,
        &store,
        &engine,
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
    let store = LocalTimeSeriesStore;
    let cfg = CatalogConfig {
        max_count: 2,
        granularity_span: Duration::from_millis(50),
        ..CatalogConfig::default()
    };
    let catalog = BucketCatalog::new(
        cfg,
        0,
        &store,
        &engine,
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
    let store = LocalTimeSeriesStore;
    let cfg = CatalogConfig {
        max_count: 1,
        granularity_span: Duration::from_secs(3600),
        ..CatalogConfig::default()
    };
    let catalog = BucketCatalog::new(
        cfg,
        0,
        &store,
        &engine,
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
    let store = LocalTimeSeriesStore;
    let cfg = CatalogConfig {
        max_count: 2,
        granularity_span: Duration::from_millis(50),
        ..CatalogConfig::default()
    };
    let catalog = BucketCatalog::new(
        cfg,
        0,
        &store,
        &engine,
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
    assert!(ts_read(&engine, |s, txn| s
        .scan_overflow(txn, 7, NodeId::from_raw(1))
        .unwrap())
    .is_empty());

    // Late measurement in-window (ts=500us inside [0, 1000])
    // but `now` is far past TTL → Tier 2 skips → Tier 3 should
    // route to overflow against bucket 1.
    let future = SystemTime::now() + Duration::from_secs(1);
    catalog
        .write_measurement_at(7, meta.clone(), measurement(500, &[("temp", 22.0)]), future)
        .unwrap();

    // Overflow has one entry under bucket 1.
    let overflow = ts_read(&engine, |s, txn| {
        s.scan_overflow(txn, 7, NodeId::from_raw(1)).unwrap()
    });
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
    let store = LocalTimeSeriesStore;
    let cfg = CatalogConfig {
        max_count: 2,
        granularity_span: Duration::from_millis(50),
        ..CatalogConfig::default()
    };
    let catalog = BucketCatalog::new(
        cfg,
        0,
        &store,
        &engine,
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

    assert!(ts_read(&engine, |s, txn| s
        .scan_overflow(txn, 7, NodeId::from_raw(1))
        .unwrap())
    .is_empty());
    assert_eq!(catalog.open_bucket_count(), 1);
}

#[test]
fn compact_if_needed_no_op_below_threshold() {
    let (_dir, engine) = mk_engine();
    let store = LocalTimeSeriesStore;
    let cfg = CatalogConfig::arch_defaults();
    let catalog = BucketCatalog::new(
        cfg,
        0,
        &store,
        &engine,
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
    let store = LocalTimeSeriesStore;
    let cfg = CatalogConfig {
        max_count: 2,
        granularity_span: Duration::from_millis(50),
        ..CatalogConfig::default()
    };
    let catalog = BucketCatalog::new(
        cfg,
        0,
        &store,
        &engine,
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
        ts_read(&engine, |s, txn| s
            .scan_overflow(txn, 7, NodeId::from_raw(1))
            .unwrap())
        .len(),
        51,
    );

    // Compact.
    let did = catalog.compact_if_needed(7, NodeId::from_raw(1)).unwrap();
    assert!(did, "compact must run when overflow exceeds threshold");

    // Post-compact: overflow set empty, base bucket has 2 + 51 = 53
    // measurements (merged + sorted).
    assert!(ts_read(&engine, |s, txn| s
        .scan_overflow(txn, 7, NodeId::from_raw(1))
        .unwrap())
    .is_empty());
    let base = ts_read(&engine, |s, txn| {
        s.get_bucket(txn, 0, NodeId::from_raw(1)).unwrap()
    })
    .unwrap();
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
    let store = LocalTimeSeriesStore;
    let cfg = CatalogConfig {
        max_count: 2,
        granularity_span: Duration::from_millis(50),
        ..CatalogConfig::default()
    };
    let catalog = BucketCatalog::new(
        cfg,
        0,
        &store,
        &engine,
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
    assert!(ts_read(&engine, |s, txn| s
        .scan_overflow(txn, 7, NodeId::from_raw(1))
        .unwrap())
    .is_empty());
    assert!(ts_read(&engine, |s, txn| s
        .scan_overflow(txn, 7, NodeId::from_raw(2))
        .unwrap())
    .is_empty());
    assert_eq!(
        ts_read(&engine, |s, txn| s
            .scan_overflow(txn, 7, NodeId::from_raw(3))
            .unwrap())
        .len(),
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
    let store = LocalTimeSeriesStore;
    let cfg = CatalogConfig {
        max_count: 3,
        granularity_span: Duration::from_secs(3600),
        ..CatalogConfig::default()
    };
    let clock = std::sync::Arc::new(crate::clock::ScriptedClock::new(vec![
        1_000_000, 2_000_000, 3_000_000,
    ]));
    let catalog = BucketCatalog::new(cfg, 0, &store, &engine, 1, clock).unwrap();
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
    let bucket = ts_read(&engine, |s, txn| {
        s.get_bucket(txn, 0, NodeId::from_raw(1)).unwrap()
    })
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
    let store = LocalTimeSeriesStore;
    let cfg = CatalogConfig::arch_defaults();
    let clock = std::sync::Arc::new(crate::clock::ScriptedClock::new(vec![999]));
    let catalog = BucketCatalog::new(cfg, 0, &store, &engine, 1, clock).unwrap();
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
    let bucket = ts_read(&engine, |s, txn| {
        s.get_bucket(txn, 0, NodeId::from_raw(1)).unwrap()
    })
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
    let store = LocalTimeSeriesStore;
    let cfg = CatalogConfig::arch_defaults();
    let clock = std::sync::Arc::new(crate::clock::ScriptedClock::new(vec![1, 2, 3]));
    let catalog = BucketCatalog::new(cfg, 0, &store, &engine, 1, clock).unwrap();
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

    let bucket = ts_read(&engine, |s, txn| {
        s.get_bucket(txn, 0, NodeId::from_raw(1)).unwrap()
    })
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
    let store = LocalTimeSeriesStore;
    let cfg = CatalogConfig {
        max_count: 2,
        granularity_span: Duration::from_millis(50),
        ..CatalogConfig::default()
    };
    let clock = std::sync::Arc::new(crate::clock::ScriptedClock::new(vec![1, 2, 3]));
    let catalog = BucketCatalog::new(cfg, 0, &store, &engine, 1, clock).unwrap();
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
    let entries = ts_read(&engine, |s, txn| {
        s.scan_overflow(txn, 7, NodeId::from_raw(1)).unwrap()
    });
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
    ts_write(&engine, |s, txn| {
        s.put_bucket(txn, 0, NodeId::from_raw(42), &legacy_bucket)
            .unwrap();
    });

    // Round-trip through the store.
    let read_back = ts_read(&engine, |s, txn| {
        s.get_bucket(txn, 0, NodeId::from_raw(42)).unwrap()
    })
    .unwrap();
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
    let store = LocalTimeSeriesStore;
    let cfg = CatalogConfig {
        max_count: 2,
        granularity_span: Duration::from_millis(50),
        ..CatalogConfig::default()
    };
    // ScriptedClock with enough stamps for: 2 base writes + 51
    // overflow writes + backfill safety margin.
    let stamps: Vec<i64> = (1_000_000..1_000_100).collect();
    let clock = std::sync::Arc::new(crate::clock::ScriptedClock::new(stamps.clone()));
    let catalog = BucketCatalog::new(cfg, 0, &store, &engine, 1, clock).unwrap();
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
    let merged = ts_read(&engine, |s, txn| {
        s.get_bucket(txn, 0, NodeId::from_raw(1)).unwrap()
    })
    .unwrap();
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
    let store = LocalTimeSeriesStore;

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
    ts_write(&engine, |s, txn| {
        s.put_bucket(txn, 0, NodeId::from_raw(1), &legacy_bucket)
            .unwrap();
        // Mark closed manually so compact_if_needed proceeds.
        s.mark_closed(txn, 0, NodeId::from_raw(1)).unwrap();
    });

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
    let catalog = BucketCatalog::new(cfg, 0, &store, &engine, 100, clock).unwrap();

    // 51 overflow writes — manually since the catalog doesn't
    // know about the planted bucket. Route them directly via
    // put_overflow so they target bucket 1.
    ts_write(&engine, |s, txn| {
        for i in 0..51 {
            s.put_overflow(
                txn,
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
    });

    // Compact — must backfill the 5 legacy measurements with
    // clock stamps, keep the 51 overflow stamps verbatim.
    assert!(catalog.compact_if_needed(7, NodeId::from_raw(1)).unwrap());

    // Verify: merged bucket has 56 measurements, all stamped.
    let merged = ts_read(&engine, |s, txn| {
        s.get_bucket(txn, 0, NodeId::from_raw(1)).unwrap()
    })
    .unwrap();
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
    let store = LocalTimeSeriesStore;
    let cfg = CatalogConfig {
        max_count: 2,
        granularity_span: Duration::from_millis(50),
        ..CatalogConfig::default()
    };
    let clock = std::sync::Arc::new(crate::clock::ScriptedClock::new(vec![
        777_000, 888_000, 999_000,
    ]));
    let catalog = BucketCatalog::new(cfg, 0, &store, &engine, 1, clock).unwrap();
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
    let entries = ts_read(&engine, |s, txn| {
        s.scan_overflow(txn, 7, NodeId::from_raw(1)).unwrap()
    });
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
    let store = LocalTimeSeriesStore;
    let cfg = CatalogConfig::arch_defaults();
    let catalog = BucketCatalog::new(
        cfg,
        0,
        &store,
        &engine,
        1,
        std::sync::Arc::new(crate::clock::MonotonicHlcClock::new()),
    )
    .unwrap();
    assert_eq!(catalog.compact_all_pending().unwrap(), 0);
}

#[test]
fn concurrent_writers_distinct_keys_scale_across_stripes() {
    let (_dir, engine) = mk_engine();
    let store = LocalTimeSeriesStore;
    let cfg = CatalogConfig {
        max_count: 1000,
        ..CatalogConfig::default()
    };
    let catalog = BucketCatalog::new(
        cfg,
        0,
        &store,
        &engine,
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
