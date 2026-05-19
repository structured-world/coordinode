//! Time-series store — bucket-node persistence for TIMESERIES labels
//! plus the late-arrival overflow segment.
//!
//! ## Modeling
//!
//! A TIMESERIES label groups measurements that share a `_meta` value
//! (e.g. one sensor) into bucket nodes. Each bucket is a regular node
//! body whose contents follow a columnar layout:
//!
//! ```text
//! node:<shard>:<bucket_id> → MessagePack {
//!   _control: {
//!     version: u8,                 // 1 = uncompressed buffer, 2 = sorted+compressed
//!     min: { field: f64, ... },    // per-field min (Volcano predicate push-down)
//!     max: { field: f64, ... },    // per-field max
//!     count: u32,
//!     time_min_us: i64,
//!     time_max_us: i64,
//!     closed: bool,
//!   },
//!   _meta: rmpv::Value,            // metaField value (string, number, struct)
//!   timestamps: [i64; count],      // microseconds since epoch, NOT delta-encoded yet
//!   fields: { name -> [f64; count] }
//! }
//! ```
//!
//! Delta-encoded timestamps and SIMD-packed f64 are tracked
//! separately — the bucket envelope on the wire is MessagePack with
//! `Vec<i64>` / `Vec<f64>` columns. The `version` byte is forward-
//! compatible: a v2 bucket with packed columns is decoded the same
//! way once that codec lands.
//!
//! ## What lives in this store vs. above it
//!
//! This store is the persistence layer for buckets — the typed wrapper
//! over `Partition::Node` for bucket reads/writes plus the overflow
//! segment in `Partition::Idx`. Higher-level concerns explicitly out
//! of scope here:
//!
//! - **BucketCatalog** (per-shard in-memory open-bucket map with
//!   striped locking, rollover triggers, recently-closed LRU) — that
//!   state machine sits above the store and uses these methods.
//! - **Late-arrival routing** (Tier 1 in-buffer / Tier 2 bucket re-
//!   open / Tier 3 overflow) — the catalog decides which method to
//!   call; this store implements the actual writes.
//! - **Bitemporal axes** (`__ingestion_ts__` field, ADR-027) — added
//!   per measurement at the catalog layer before [`Bucket::append`].
//!
//! ## Overflow segment
//!
//! Truly late measurements (Tier 3, > 2× granularity span behind) go
//! into per-bucket overflow keys so the closed-and-compacted base
//! bucket need not be re-decoded for one stray point:
//!
//! ```text
//! ts_overflow:<label_id>:<bucket_id>:<arrival_seqno> → Measurement
//! ```
//!
//! Reads merge-sort the base bucket with the overflow scan; the
//! background compactor periodically re-encodes the base and atomic-
//! ally deletes its overflow set via [`Self::compact_overflow`].

use std::collections::BTreeMap;

use coordinode_core::graph::node::{encode_node_key, NodeId};
use coordinode_storage::engine::batch::WriteBatch;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::Guard;
use serde::{Deserialize, Serialize};

use crate::error::{StoreError, StoreResult};

/// Bucket wire-format version. v1 = uncompressed (raw columns),
/// v2 = sorted-and-compressed (delta timestamps + packed floats).
pub const BUCKET_VERSION_RAW: u8 = 1;
/// Bucket wire-format version with columnar compression.
pub const BUCKET_VERSION_COMPRESSED: u8 = 2;

/// Per-field statistics retained in the bucket control block so a
/// filter on the field can prune the bucket without decoding columns.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FieldStats {
    /// Lowest observed value for this field within the bucket.
    pub min: f64,
    /// Highest observed value for this field within the bucket.
    pub max: f64,
}

/// Control block carried in every bucket — header used for predicate
/// push-down and rollover decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketControl {
    /// Wire-format version. See [`BUCKET_VERSION_RAW`] and friends.
    pub version: u8,
    /// Measurement count in the bucket (length of every column).
    pub count: u32,
    /// Lowest observed timestamp (microseconds, event-time axis).
    pub time_min_us: i64,
    /// Highest observed timestamp (microseconds, event-time axis).
    pub time_max_us: i64,
    /// True once the bucket has been closed (no more in-buffer
    /// appends; further writes must go through overflow or re-open).
    pub closed: bool,
    /// Per-field min/max across the bucket.
    pub fields_stats: BTreeMap<String, FieldStats>,
}

impl BucketControl {
    /// Compute the control block from a measurement set.
    pub fn from_measurements(measurements: &[Measurement]) -> Self {
        let mut control = Self {
            version: BUCKET_VERSION_RAW,
            count: measurements.len() as u32,
            time_min_us: i64::MAX,
            time_max_us: i64::MIN,
            closed: false,
            fields_stats: BTreeMap::new(),
        };
        for m in measurements {
            control.time_min_us = control.time_min_us.min(m.timestamp_us);
            control.time_max_us = control.time_max_us.max(m.timestamp_us);
            for (name, value) in &m.fields {
                let stats = control
                    .fields_stats
                    .entry(name.clone())
                    .or_insert(FieldStats {
                        min: *value,
                        max: *value,
                    });
                stats.min = stats.min.min(*value);
                stats.max = stats.max.max(*value);
            }
        }
        if measurements.is_empty() {
            control.time_min_us = 0;
            control.time_max_us = 0;
        }
        control
    }
}

/// Single time-series measurement: one event-time + N named float
/// fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Measurement {
    /// Event-time in microseconds since epoch.
    pub timestamp_us: i64,
    /// Named float fields. Schema-on-write per bucket: a new field
    /// here would normally trigger bucket rollover at the catalog
    /// layer.
    pub fields: BTreeMap<String, f64>,
}

/// A complete bucket: control block + meta value + columnar storage.
///
/// Columns are kept aligned by index — `timestamps[i]` and every
/// `fields[name][i]` describe the same measurement. The columnar
/// layout is what enables predicate push-down to a single column.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bucket {
    /// Header — see [`BucketControl`].
    pub control: BucketControl,
    /// Meta-field value shared by every measurement in the bucket.
    pub meta: rmpv::Value,
    /// Timestamps column (microseconds), aligned with every field
    /// column.
    pub timestamps: Vec<i64>,
    /// Per-field columns, all of length `control.count`.
    pub fields: BTreeMap<String, Vec<f64>>,
}

impl Bucket {
    /// Build a v1 bucket from a measurement vector. Measurements are
    /// stored in arrival order — sorting is the closure-time job and
    /// belongs to the catalog/compactor.
    pub fn from_measurements(meta: rmpv::Value, measurements: Vec<Measurement>) -> Self {
        let control = BucketControl::from_measurements(&measurements);
        let mut timestamps = Vec::with_capacity(measurements.len());
        let mut fields: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for m in measurements {
            timestamps.push(m.timestamp_us);
            for (name, value) in m.fields {
                fields
                    .entry(name)
                    .or_insert_with(|| Vec::with_capacity(control.count as usize))
                    .push(value);
            }
        }
        Self {
            control,
            meta,
            timestamps,
            fields,
        }
    }

    /// Iterate measurements in storage order. Reconstructs each row
    /// from the columnar layout — useful for tests and for the merge
    /// step when overflow is folded into a base bucket.
    pub fn measurements(&self) -> impl Iterator<Item = Measurement> + '_ {
        (0..self.control.count as usize).map(move |i| {
            let mut fields = BTreeMap::new();
            for (name, col) in &self.fields {
                if let Some(v) = col.get(i) {
                    fields.insert(name.clone(), *v);
                }
            }
            Measurement {
                timestamp_us: self.timestamps[i],
                fields,
            }
        })
    }
}

/// Overflow entry for a single late-arriving measurement.
///
/// Stored under `ts_overflow:<label_id>:<bucket_id>:<arrival_seqno>`
/// so the per-bucket scan is a tight key-prefix walk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OverflowEntry {
    /// Seqno assigned at write time — must monotonically increase per
    /// bucket so prefix-scan iteration yields stable order even
    /// across writers.
    pub arrival_seqno: u64,
    /// The late measurement itself.
    pub measurement: Measurement,
}

/// Time-series modality store.
///
/// All methods are keyed by `(label_id, bucket_id)`. `label_id` is the
/// stable numeric id assigned by the schema store to a TIMESERIES
/// label; `bucket_id` is a node id minted at bucket creation time.
pub trait TimeSeriesStore {
    /// Persist a complete bucket. Overwrites any prior body at the
    /// same node id — bucket bodies are written wholesale by the
    /// catalog when it flushes its in-memory buffer.
    fn put_bucket(&self, shard_id: u16, bucket_id: NodeId, bucket: &Bucket) -> StoreResult<()>;

    /// Read the bucket body. Returns `None` when the key is absent.
    fn get_bucket(&self, shard_id: u16, bucket_id: NodeId) -> StoreResult<Option<Bucket>>;

    /// Tombstone the bucket. Idempotent on a missing key. Does *not*
    /// touch the overflow segment — use [`Self::compact_overflow`]
    /// (with an empty merged bucket) or call this *after* draining
    /// overflow.
    fn delete_bucket(&self, shard_id: u16, bucket_id: NodeId) -> StoreResult<()>;

    /// Mark the bucket closed by updating its control block in place.
    /// Reads `bucket`, sets `control.closed = true`, writes back.
    /// Returns `false` if no bucket exists at that key.
    fn mark_closed(&self, shard_id: u16, bucket_id: NodeId) -> StoreResult<bool>;

    /// Append one late measurement to the overflow segment under
    /// `(label_id, bucket_id)`.
    fn put_overflow(
        &self,
        label_id: u32,
        bucket_id: NodeId,
        entry: &OverflowEntry,
    ) -> StoreResult<()>;

    /// All overflow entries for one bucket, in arrival_seqno order.
    fn scan_overflow(&self, label_id: u32, bucket_id: NodeId) -> StoreResult<Vec<OverflowEntry>>;

    /// Atomic compact: write the merged base bucket and tombstone
    /// every overflow key currently visible. The two writes land in
    /// one [`WriteBatch`] so a crash cannot leave overflow visible
    /// against an already-rewritten base.
    fn compact_overflow(
        &self,
        shard_id: u16,
        label_id: u32,
        bucket_id: NodeId,
        merged: &Bucket,
        overflow_seqnos: &[u64],
    ) -> StoreResult<()>;
}

/// CE single-shard implementation of [`TimeSeriesStore`].
///
/// Buckets land in [`Partition::Node`] keyed by the standard node
/// shard+id encoder; overflow entries live in [`Partition::Idx`]
/// under a `ts_overflow:` prefix so prefix-scan stays tight.
pub struct LocalTimeSeriesStore<'a> {
    engine: &'a StorageEngine,
}

impl<'a> LocalTimeSeriesStore<'a> {
    /// Wrap a storage engine for time-series store operations.
    pub fn new(engine: &'a StorageEngine) -> Self {
        Self { engine }
    }
}

const OVERFLOW_PREFIX: &[u8] = b"ts_overflow:";

fn encode_overflow_prefix(label_id: u32, bucket_id: NodeId) -> Vec<u8> {
    let mut out = Vec::with_capacity(OVERFLOW_PREFIX.len() + 4 + 8 + 1);
    out.extend_from_slice(OVERFLOW_PREFIX);
    out.extend_from_slice(&label_id.to_be_bytes());
    out.extend_from_slice(&bucket_id.as_raw().to_be_bytes());
    out.push(b':');
    out
}

fn encode_overflow_key(label_id: u32, bucket_id: NodeId, arrival_seqno: u64) -> Vec<u8> {
    let mut out = encode_overflow_prefix(label_id, bucket_id);
    out.extend_from_slice(&arrival_seqno.to_be_bytes());
    out
}

fn encode_bucket(bucket: &Bucket) -> StoreResult<Vec<u8>> {
    rmp_serde::to_vec_named(bucket).map_err(|e| StoreError::Decode {
        kind: "ts bucket",
        message: e.to_string(),
    })
}

fn decode_bucket(bytes: &[u8]) -> StoreResult<Bucket> {
    rmp_serde::from_slice(bytes).map_err(|e| StoreError::Decode {
        kind: "ts bucket",
        message: e.to_string(),
    })
}

fn encode_overflow_entry(entry: &OverflowEntry) -> StoreResult<Vec<u8>> {
    rmp_serde::to_vec_named(entry).map_err(|e| StoreError::Decode {
        kind: "ts overflow entry",
        message: e.to_string(),
    })
}

fn decode_overflow_entry(bytes: &[u8]) -> StoreResult<OverflowEntry> {
    rmp_serde::from_slice(bytes).map_err(|e| StoreError::Decode {
        kind: "ts overflow entry",
        message: e.to_string(),
    })
}

impl TimeSeriesStore for LocalTimeSeriesStore<'_> {
    fn put_bucket(&self, shard_id: u16, bucket_id: NodeId, bucket: &Bucket) -> StoreResult<()> {
        let key = encode_node_key(shard_id, bucket_id);
        let bytes = encode_bucket(bucket)?;
        self.engine.put(Partition::Node, &key, &bytes)?;
        Ok(())
    }

    fn get_bucket(&self, shard_id: u16, bucket_id: NodeId) -> StoreResult<Option<Bucket>> {
        let key = encode_node_key(shard_id, bucket_id);
        match self.engine.get(Partition::Node, &key)? {
            Some(bytes) => Ok(Some(decode_bucket(bytes.as_ref())?)),
            None => Ok(None),
        }
    }

    fn delete_bucket(&self, shard_id: u16, bucket_id: NodeId) -> StoreResult<()> {
        let key = encode_node_key(shard_id, bucket_id);
        self.engine.delete(Partition::Node, &key)?;
        Ok(())
    }

    fn mark_closed(&self, shard_id: u16, bucket_id: NodeId) -> StoreResult<bool> {
        let key = encode_node_key(shard_id, bucket_id);
        let Some(bytes) = self.engine.get(Partition::Node, &key)? else {
            return Ok(false);
        };
        let mut bucket = decode_bucket(bytes.as_ref())?;
        if bucket.control.closed {
            return Ok(true);
        }
        bucket.control.closed = true;
        let encoded = encode_bucket(&bucket)?;
        self.engine.put(Partition::Node, &key, &encoded)?;
        Ok(true)
    }

    fn put_overflow(
        &self,
        label_id: u32,
        bucket_id: NodeId,
        entry: &OverflowEntry,
    ) -> StoreResult<()> {
        let key = encode_overflow_key(label_id, bucket_id, entry.arrival_seqno);
        let bytes = encode_overflow_entry(entry)?;
        self.engine.put(Partition::Idx, &key, &bytes)?;
        Ok(())
    }

    fn scan_overflow(&self, label_id: u32, bucket_id: NodeId) -> StoreResult<Vec<OverflowEntry>> {
        let prefix = encode_overflow_prefix(label_id, bucket_id);
        let mut out = Vec::new();
        let iter = self.engine.prefix_scan(Partition::Idx, &prefix)?;
        for guard in iter {
            let (_k, v) = guard.into_inner()?;
            out.push(decode_overflow_entry(v.as_ref())?);
        }
        out.sort_by_key(|e| e.arrival_seqno);
        Ok(out)
    }

    fn compact_overflow(
        &self,
        shard_id: u16,
        label_id: u32,
        bucket_id: NodeId,
        merged: &Bucket,
        overflow_seqnos: &[u64],
    ) -> StoreResult<()> {
        let bucket_key = encode_node_key(shard_id, bucket_id);
        let bucket_bytes = encode_bucket(merged)?;
        let mut batch = WriteBatch::new(self.engine);
        batch.put(Partition::Node, bucket_key, bucket_bytes);
        for seqno in overflow_seqnos {
            batch.delete(
                Partition::Idx,
                encode_overflow_key(label_id, bucket_id, *seqno),
            );
        }
        batch.commit()?;
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use coordinode_storage::engine::config::{
        Durability, EndpointConfig, Media, StorageConfig, Tier,
    };
    use coordinode_storage::engine::core::StorageEngine;
    use tempfile::TempDir;

    fn mk_engine() -> (TempDir, StorageEngine) {
        let dir = TempDir::new().expect("tempdir");
        let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "ep",
            dir.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        let engine = StorageEngine::open(&config).expect("open");
        (dir, engine)
    }

    fn mk_measurement(ts: i64, temp: f64) -> Measurement {
        let mut fields = BTreeMap::new();
        fields.insert("temperature".to_owned(), temp);
        Measurement {
            timestamp_us: ts,
            fields,
        }
    }

    #[test]
    fn bucket_round_trip() {
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);
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
        store.put_bucket(0, bucket_id, &bucket).unwrap();

        let read_back = store.get_bucket(0, bucket_id).unwrap().expect("present");
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
    fn empty_bucket_control_defaults_to_zero_range() {
        let bucket = Bucket::from_measurements(rmpv::Value::String("empty".into()), Vec::new());
        assert_eq!(bucket.control.count, 0);
        assert_eq!(bucket.control.time_min_us, 0);
        assert_eq!(bucket.control.time_max_us, 0);
    }

    #[test]
    fn get_missing_bucket_returns_none() {
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);
        assert!(store.get_bucket(0, NodeId::from_raw(99)).unwrap().is_none());
    }

    #[test]
    fn delete_is_idempotent() {
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);
        store.delete_bucket(0, NodeId::from_raw(7)).unwrap();
        store.delete_bucket(0, NodeId::from_raw(7)).unwrap();
    }

    #[test]
    fn mark_closed_sets_flag_and_is_idempotent() {
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);
        let bucket = Bucket::from_measurements(rmpv::Value::Nil, vec![mk_measurement(1, 1.0)]);
        let id = NodeId::from_raw(11);
        store.put_bucket(0, id, &bucket).unwrap();
        assert!(store.mark_closed(0, id).unwrap());
        assert!(store.mark_closed(0, id).unwrap());
        let read = store.get_bucket(0, id).unwrap().unwrap();
        assert!(read.control.closed);
    }

    #[test]
    fn mark_closed_missing_returns_false() {
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);
        assert!(!store.mark_closed(0, NodeId::from_raw(404)).unwrap());
    }

    #[test]
    fn overflow_round_trip_sorted_by_seqno() {
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);
        let bid = NodeId::from_raw(5);
        // Insert out of order — scan must return ordered by arrival_seqno.
        for seqno in [3u64, 1, 2] {
            let entry = OverflowEntry {
                arrival_seqno: seqno,
                measurement: mk_measurement(seqno as i64 * 1000, seqno as f64),
            };
            store.put_overflow(7, bid, &entry).unwrap();
        }
        let entries = store.scan_overflow(7, bid).unwrap();
        let seqnos: Vec<_> = entries.iter().map(|e| e.arrival_seqno).collect();
        assert_eq!(seqnos, vec![1, 2, 3]);
    }

    #[test]
    fn overflow_scoped_per_bucket() {
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);
        let a = NodeId::from_raw(1);
        let b = NodeId::from_raw(2);
        store
            .put_overflow(
                9,
                a,
                &OverflowEntry {
                    arrival_seqno: 100,
                    measurement: mk_measurement(1, 1.0),
                },
            )
            .unwrap();
        store
            .put_overflow(
                9,
                b,
                &OverflowEntry {
                    arrival_seqno: 200,
                    measurement: mk_measurement(2, 2.0),
                },
            )
            .unwrap();
        let only_a = store.scan_overflow(9, a).unwrap();
        let only_b = store.scan_overflow(9, b).unwrap();
        assert_eq!(only_a.len(), 1);
        assert_eq!(only_a[0].arrival_seqno, 100);
        assert_eq!(only_b.len(), 1);
        assert_eq!(only_b[0].arrival_seqno, 200);
    }

    #[test]
    fn compact_overflow_writes_base_and_deletes_overflow_atomically() {
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);
        let label_id = 4u32;
        let bid = NodeId::from_raw(50);

        // Initial bucket + two overflow entries.
        let base = Bucket::from_measurements(
            rmpv::Value::String("s".into()),
            vec![mk_measurement(100, 10.0)],
        );
        store.put_bucket(0, bid, &base).unwrap();
        for seqno in [1u64, 2] {
            store
                .put_overflow(
                    label_id,
                    bid,
                    &OverflowEntry {
                        arrival_seqno: seqno,
                        measurement: mk_measurement(50 + seqno as i64, seqno as f64 * 5.0),
                    },
                )
                .unwrap();
        }
        assert_eq!(store.scan_overflow(label_id, bid).unwrap().len(), 2);

        // Compact: merge two overflow points into the base, delete both.
        let merged = Bucket::from_measurements(
            rmpv::Value::String("s".into()),
            vec![
                mk_measurement(51, 5.0),
                mk_measurement(52, 10.0),
                mk_measurement(100, 10.0),
            ],
        );
        store
            .compact_overflow(0, label_id, bid, &merged, &[1, 2])
            .unwrap();

        let after = store.get_bucket(0, bid).unwrap().unwrap();
        assert_eq!(after.control.count, 3);
        assert_eq!(after.control.time_min_us, 51);
        assert_eq!(after.control.time_max_us, 100);
        assert!(store.scan_overflow(label_id, bid).unwrap().is_empty());
    }
}
