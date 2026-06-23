use super::*;

/// Logic-test fixture (memory backing, env-flippable). Bucket
/// CRUD + overflow routing tests verify ts-store contracts,
/// not persistence.
fn mk_engine() -> coordinode_test_fixtures::EngineFixture {
    coordinode_test_fixtures::engine_for_logic()
}

use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::transaction::CommitContext;

fn mk_measurement(ts: i64, temp: f64) -> Measurement {
    let mut fields = BTreeMap::new();
    fields.insert("temperature".to_owned(), temp);
    Measurement {
        timestamp_us: ts,
        ingestion_ts_us: None,
        fields,
    }
}

/// Run time-series writes in one MVCC transaction and commit,
/// returning the closure's result (e.g. the `bool` from
/// `mark_closed` / `reopen_bucket`).
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
    txn.commit(&ctx).expect("commit ts");
    out
}

/// Run a time-series read closure against the latest committed
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
fn bucket_round_trip() {
    let fx = mk_engine();
    let engine = &fx.engine;
    let measurements = vec![
        mk_measurement(100, 18.5),
        mk_measurement(200, 22.0),
        mk_measurement(300, 19.5),
    ];
    let bucket = Bucket::from_measurements(
        rmpv::Value::String("sensor-42".into()),
        measurements.clone(),
    );
    let bucket_id = NodeId::from_raw(42);
    ts_write(engine, |s, txn| {
        s.put_bucket(txn, 0, bucket_id, &bucket).unwrap()
    });

    let read_back =
        ts_read(engine, |s, txn| s.get_bucket(txn, 0, bucket_id).unwrap()).expect("present");
    assert_eq!(read_back.control.count, 3);
    assert_eq!(read_back.control.time_min_us, 100);
    assert_eq!(read_back.control.time_max_us, 300);
    let stats = read_back.control.fields_stats.get("temperature").unwrap();
    assert_eq!(stats.min, 18.5);
    assert_eq!(stats.max, 22.0);
    let materialised: Vec<_> = read_back.measurements().collect();
    assert_eq!(materialised, measurements);
}

#[test]
fn heterogeneous_fields_produce_uneven_column_lengths() {
    // Two measurements with different field sets. The bucket
    // builder appends per-field — so "temp" column has one entry
    // (from the first measurement) and "humidity" has one (from
    // the second). This is the documented behaviour: columns are
    // NOT padded with NaN. Catalog above is responsible for
    // detecting "schema change" and rolling over to a new bucket.
    let mut fields_a = BTreeMap::new();
    fields_a.insert("temp".into(), 22.0);
    let mut fields_b = BTreeMap::new();
    fields_b.insert("humidity".into(), 65.0);
    let bucket = Bucket::from_measurements(
        rmpv::Value::Nil,
        vec![
            Measurement {
                timestamp_us: 100,
                ingestion_ts_us: None,
                fields: fields_a,
            },
            Measurement {
                timestamp_us: 200,
                ingestion_ts_us: None,
                fields: fields_b,
            },
        ],
    );
    assert_eq!(bucket.timestamps.len(), 2);
    assert_eq!(bucket.fields.get("temp").map(|c| c.len()), Some(1));
    assert_eq!(bucket.fields.get("humidity").map(|c| c.len()), Some(1));
    // Documented gotcha: column length != bucket count when
    // schemas diverge.
    assert_eq!(bucket.control.count, 2);
}

#[test]
fn overflow_same_seqno_silently_overwrites() {
    // Two writers picking the same arrival_seqno collide at the
    // same key. The second write overwrites — first measurement
    // is lost. Documented hazard: the catalog above must mint
    // strictly-monotonic seqnos per bucket.
    let fx = mk_engine();
    let engine = &fx.engine;
    let bid = NodeId::from_raw(60);

    let first = OverflowEntry {
        arrival_seqno: 7,
        measurement: mk_measurement(100, 1.0),
    };
    let second = OverflowEntry {
        arrival_seqno: 7,
        measurement: mk_measurement(200, 2.0),
    };
    ts_write(engine, |s, txn| {
        s.put_overflow(txn, 1, bid, &first).unwrap();
        s.put_overflow(txn, 1, bid, &second).unwrap();
    });
    let entries = ts_read(engine, |s, txn| s.scan_overflow(txn, 1, bid).unwrap());
    assert_eq!(entries.len(), 1, "second write must overwrite first");
    assert_eq!(entries[0].measurement.timestamp_us, 200);
}

#[test]
fn concurrent_put_overflow_distinct_seqnos_converges() {
    // Four threads write distinct arrival_seqnos into the same
    // bucket's overflow segment, each in its own transaction. All
    // four must be visible after join, sorted by arrival_seqno on
    // scan.
    use std::sync::Arc;
    use std::thread;

    let fx = mk_engine();
    let engine = Arc::clone(&fx.engine);
    let bid = NodeId::from_raw(70);
    let label = 13u32;

    let handles: Vec<_> = (0..4u64)
        .map(|t| {
            let engine = Arc::clone(&engine);
            thread::spawn(move || {
                let entry = OverflowEntry {
                    arrival_seqno: t + 1,
                    measurement: mk_measurement((t as i64 + 1) * 100, t as f64),
                };
                ts_write(&engine, |s, txn| {
                    s.put_overflow(txn, label, bid, &entry).expect("put");
                });
            })
        })
        .collect();
    for h in handles {
        h.join().expect("join");
    }

    let entries = ts_read(&engine, |s, txn| {
        s.scan_overflow(txn, label, bid).expect("scan")
    });
    let seqnos: Vec<u64> = entries.iter().map(|e| e.arrival_seqno).collect();
    assert_eq!(seqnos, vec![1, 2, 3, 4]);
}

#[test]
fn empty_bucket_control_defaults_to_zero_range() {
    let bucket = Bucket::from_measurements(rmpv::Value::String("empty".into()), Vec::new());
    assert_eq!(bucket.control.count, 0);
    assert_eq!(bucket.control.time_min_us, 0);
    assert_eq!(bucket.control.time_max_us, 0);
}

#[test]
fn get_missing_bucket_returns_none() {
    let fx = mk_engine();
    let engine = &fx.engine;
    assert!(ts_read(engine, |s, txn| s
        .get_bucket(txn, 0, NodeId::from_raw(99))
        .unwrap())
    .is_none());
}

#[test]
fn delete_is_idempotent() {
    let fx = mk_engine();
    let engine = &fx.engine;
    ts_write(engine, |s, txn| {
        s.delete_bucket(txn, 0, NodeId::from_raw(7)).unwrap();
        s.delete_bucket(txn, 0, NodeId::from_raw(7)).unwrap();
    });
}

#[test]
fn mark_closed_sets_flag_and_is_idempotent() {
    let fx = mk_engine();
    let engine = &fx.engine;
    let bucket = Bucket::from_measurements(rmpv::Value::Nil, vec![mk_measurement(1, 1.0)]);
    let id = NodeId::from_raw(11);
    ts_write(engine, |s, txn| s.put_bucket(txn, 0, id, &bucket).unwrap());
    assert!(ts_write(engine, |s, txn| s
        .mark_closed(txn, 0, id)
        .unwrap()));
    assert!(ts_write(engine, |s, txn| s
        .mark_closed(txn, 0, id)
        .unwrap()));
    let read = ts_read(engine, |s, txn| s.get_bucket(txn, 0, id).unwrap()).unwrap();
    assert!(read.control.closed);
}

#[test]
fn reopen_bucket_flips_closed_back_to_false() {
    let fx = mk_engine();
    let engine = &fx.engine;
    let bucket = Bucket::from_measurements(rmpv::Value::Nil, vec![mk_measurement(1, 1.0)]);
    let id = NodeId::from_raw(31);
    ts_write(engine, |s, txn| s.put_bucket(txn, 0, id, &bucket).unwrap());
    assert!(ts_write(engine, |s, txn| s
        .mark_closed(txn, 0, id)
        .unwrap()));
    assert!(ts_write(engine, |s, txn| s
        .reopen_bucket(txn, 0, id)
        .unwrap()));
    let read = ts_read(engine, |s, txn| s.get_bucket(txn, 0, id).unwrap()).unwrap();
    assert!(!read.control.closed);
}

#[test]
fn reopen_bucket_on_already_open_is_idempotent() {
    let fx = mk_engine();
    let engine = &fx.engine;
    let bucket = Bucket::from_measurements(rmpv::Value::Nil, vec![mk_measurement(1, 1.0)]);
    let id = NodeId::from_raw(32);
    ts_write(engine, |s, txn| s.put_bucket(txn, 0, id, &bucket).unwrap());
    // Never closed — reopen returns true and leaves closed=false.
    assert!(ts_write(engine, |s, txn| s
        .reopen_bucket(txn, 0, id)
        .unwrap()));
    let read = ts_read(engine, |s, txn| s.get_bucket(txn, 0, id).unwrap()).unwrap();
    assert!(!read.control.closed);
}

#[test]
fn reopen_missing_bucket_returns_false() {
    let fx = mk_engine();
    let engine = &fx.engine;
    assert!(!ts_write(engine, |s, txn| s
        .reopen_bucket(txn, 0, NodeId::from_raw(404))
        .unwrap()));
}

#[test]
fn late_write_flow_close_reopen_append_compact() {
    // End-to-end Tier-2 late-arrival simulation: build a bucket,
    // close it, route one late point through the overflow segment
    // (Tier 3 is the simpler API), then reopen so the catalog can
    // resume in-buffer appends, and finally compact overflow back
    // into the base.
    let fx = mk_engine();
    let engine = &fx.engine;
    let id = NodeId::from_raw(50);
    let label = 11u32;

    let base = Bucket::from_measurements(
        rmpv::Value::String("sensor".into()),
        vec![mk_measurement(100, 1.0), mk_measurement(200, 2.0)],
    );
    ts_write(engine, |s, txn| s.put_bucket(txn, 0, id, &base).unwrap());
    assert!(ts_write(engine, |s, txn| s
        .mark_closed(txn, 0, id)
        .unwrap()));

    // Tier 3: stash a late measurement in overflow.
    let late = OverflowEntry {
        arrival_seqno: 1,
        measurement: mk_measurement(150, 1.5),
    };
    ts_write(engine, |s, txn| {
        s.put_overflow(txn, label, id, &late).unwrap()
    });

    // Catalog decides this bucket is hot again → reopen.
    assert!(ts_write(engine, |s, txn| s
        .reopen_bucket(txn, 0, id)
        .unwrap()));
    let mid = ts_read(engine, |s, txn| s.get_bucket(txn, 0, id).unwrap()).unwrap();
    assert!(!mid.control.closed);

    // Background compactor folds overflow into the base.
    let merged = Bucket::from_measurements(
        rmpv::Value::String("sensor".into()),
        vec![
            mk_measurement(100, 1.0),
            mk_measurement(150, 1.5),
            mk_measurement(200, 2.0),
        ],
    );
    ts_write(engine, |s, txn| {
        s.compact_overflow(txn, 0, label, id, &merged, &[1])
            .unwrap();
    });

    let after = ts_read(engine, |s, txn| s.get_bucket(txn, 0, id).unwrap()).unwrap();
    assert_eq!(after.control.count, 3);
    assert_eq!(after.control.time_min_us, 100);
    assert_eq!(after.control.time_max_us, 200);
    assert!(ts_read(engine, |s, txn| s.scan_overflow(txn, label, id).unwrap()).is_empty());
}

#[test]
fn mark_closed_missing_returns_false() {
    let fx = mk_engine();
    let engine = &fx.engine;
    assert!(!ts_write(engine, |s, txn| s
        .mark_closed(txn, 0, NodeId::from_raw(404))
        .unwrap()));
}

#[test]
fn overflow_round_trip_sorted_by_seqno() {
    let fx = mk_engine();
    let engine = &fx.engine;
    let bid = NodeId::from_raw(5);
    // Insert out of order — scan must return ordered by arrival_seqno.
    ts_write(engine, |s, txn| {
        for seqno in [3u64, 1, 2] {
            let entry = OverflowEntry {
                arrival_seqno: seqno,
                measurement: mk_measurement(seqno as i64 * 1000, seqno as f64),
            };
            s.put_overflow(txn, 7, bid, &entry).unwrap();
        }
    });
    let entries = ts_read(engine, |s, txn| s.scan_overflow(txn, 7, bid).unwrap());
    let seqnos: Vec<_> = entries.iter().map(|e| e.arrival_seqno).collect();
    assert_eq!(seqnos, vec![1, 2, 3]);
}

#[test]
fn overflow_scoped_per_bucket() {
    let fx = mk_engine();
    let engine = &fx.engine;
    let a = NodeId::from_raw(1);
    let b = NodeId::from_raw(2);
    ts_write(engine, |s, txn| {
        s.put_overflow(
            txn,
            9,
            a,
            &OverflowEntry {
                arrival_seqno: 100,
                measurement: mk_measurement(1, 1.0),
            },
        )
        .unwrap();
        s.put_overflow(
            txn,
            9,
            b,
            &OverflowEntry {
                arrival_seqno: 200,
                measurement: mk_measurement(2, 2.0),
            },
        )
        .unwrap();
    });
    let only_a = ts_read(engine, |s, txn| s.scan_overflow(txn, 9, a).unwrap());
    let only_b = ts_read(engine, |s, txn| s.scan_overflow(txn, 9, b).unwrap());
    assert_eq!(only_a.len(), 1);
    assert_eq!(only_a[0].arrival_seqno, 100);
    assert_eq!(only_b.len(), 1);
    assert_eq!(only_b[0].arrival_seqno, 200);
}

#[test]
fn list_overflow_buckets_returns_unique_pairs_across_multiple_entries() {
    let fx = mk_engine();
    let engine = &fx.engine;

    let m = |ts: i64| Measurement {
        timestamp_us: ts,
        ingestion_ts_us: None,
        fields: BTreeMap::new(),
    };

    // Bucket A: label=7, bucket=42, two overflow entries.
    let entry_a1 = OverflowEntry {
        arrival_seqno: 1,
        measurement: m(100),
    };
    let entry_a2 = OverflowEntry {
        arrival_seqno: 2,
        measurement: m(200),
    };
    // Bucket B: label=7, bucket=99, one entry.
    let entry_b = OverflowEntry {
        arrival_seqno: 1,
        measurement: m(300),
    };
    // Bucket C: label=8 (different label), bucket=42 (same bucket_id as A but
    // different (label, bucket) pair), one entry.
    let entry_c = OverflowEntry {
        arrival_seqno: 1,
        measurement: m(400),
    };
    ts_write(engine, |s, txn| {
        s.put_overflow(txn, 7, NodeId::from_raw(42), &entry_a1)
            .unwrap();
        s.put_overflow(txn, 7, NodeId::from_raw(42), &entry_a2)
            .unwrap();
        s.put_overflow(txn, 7, NodeId::from_raw(99), &entry_b)
            .unwrap();
        s.put_overflow(txn, 8, NodeId::from_raw(42), &entry_c)
            .unwrap();
    });

    let mut listed = ts_read(engine, |s, txn| s.list_overflow_buckets(txn).expect("list"));
    listed.sort_by_key(|a| (a.0, a.1.as_raw()));
    assert_eq!(listed.len(), 3, "3 unique (label_id, bucket_id) pairs");
    assert_eq!(listed[0], (7, NodeId::from_raw(42)));
    assert_eq!(listed[1], (7, NodeId::from_raw(99)));
    assert_eq!(listed[2], (8, NodeId::from_raw(42)));
}

#[test]
fn list_overflow_buckets_empty_when_no_overflow() {
    let fx = mk_engine();
    let engine = &fx.engine;
    let listed = ts_read(engine, |s, txn| s.list_overflow_buckets(txn).expect("list"));
    assert!(listed.is_empty());
}

#[test]
fn compact_overflow_writes_base_and_deletes_overflow_atomically() {
    let fx = mk_engine();
    let engine = &fx.engine;
    let label_id = 4u32;
    let bid = NodeId::from_raw(50);

    // Initial bucket + two overflow entries.
    let base = Bucket::from_measurements(
        rmpv::Value::String("s".into()),
        vec![mk_measurement(100, 10.0)],
    );
    ts_write(engine, |s, txn| {
        s.put_bucket(txn, 0, bid, &base).unwrap();
        for seqno in [1u64, 2] {
            s.put_overflow(
                txn,
                label_id,
                bid,
                &OverflowEntry {
                    arrival_seqno: seqno,
                    measurement: mk_measurement(50 + seqno as i64, seqno as f64 * 5.0),
                },
            )
            .unwrap();
        }
    });
    assert_eq!(
        ts_read(engine, |s, txn| s
            .scan_overflow(txn, label_id, bid)
            .unwrap())
        .len(),
        2
    );

    // Compact: merge two overflow points into the base, delete both.
    let merged = Bucket::from_measurements(
        rmpv::Value::String("s".into()),
        vec![
            mk_measurement(51, 5.0),
            mk_measurement(52, 10.0),
            mk_measurement(100, 10.0),
        ],
    );
    ts_write(engine, |s, txn| {
        s.compact_overflow(txn, 0, label_id, bid, &merged, &[1, 2])
            .unwrap();
    });

    let after = ts_read(engine, |s, txn| s.get_bucket(txn, 0, bid).unwrap()).unwrap();
    assert_eq!(after.control.count, 3);
    assert_eq!(after.control.time_min_us, 51);
    assert_eq!(after.control.time_max_us, 100);
    assert!(ts_read(engine, |s, txn| s
        .scan_overflow(txn, label_id, bid)
        .unwrap())
    .is_empty());
}
