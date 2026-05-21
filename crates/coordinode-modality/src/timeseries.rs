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
//!   per measurement at the catalog layer before the bucket is
//!   written via [`TimeSeriesStore::put_bucket`].
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
//! ally deletes its overflow set via
//! [`TimeSeriesStore::compact_overflow`].

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
/// fields, with an optional **engine-assigned** ingestion-time
/// stamp (the second axis of the bitemporal model from ADR-027).
///
/// ## Bitemporal axes
///
/// CoordiNode time-series carries two distinct timestamps:
///
/// 1. **Event-time** ([`Self::timestamp_us`]) — when the measured
///    event actually happened. User-supplied, may be out-of-order
///    on arrival.
/// 2. **Ingestion-time** ([`Self::ingestion_ts_us`]) —
///    engine-assigned when the catalog accepted the measurement.
///    Strictly monotonic per shard (sourced from the catalog's
///    `IngestionClock`). Never user-supplied, never user-mutable,
///    reserved field name `__ingestion_ts__` in OpenCypher.
///
/// `ingestion_ts_us` is `Option<i64>` purely for backward
/// compatibility: buckets serialized before the bitemporal axis
/// landed decode as `None` and remain readable; freshly-written
/// measurements via the catalog always carry `Some`.
///
/// ## Why both
///
/// Event-time answers "what happened?". Ingestion-time answers
/// "when did the database learn about it?". The bitemporal join —
/// `AS OF INGESTION_TIME $t` — filters by `ingestion_ts_us ≤ $t`,
/// excluding backfills and corrections that arrived after `$t`.
/// Canonical use case: month-end-close replay ("what did we know
/// about events at event-time E, as of ingestion-time
/// T = month-end?").
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Measurement {
    /// Event-time in microseconds since epoch.
    pub timestamp_us: i64,
    /// Engine-assigned ingestion-time in microseconds since epoch.
    /// `None` on legacy measurements (pre-bitemporal buckets);
    /// `Some` on every catalog-stamped write. Strictly monotonic
    /// per shard.
    #[serde(default)]
    pub ingestion_ts_us: Option<i64>,
    /// Named float fields. Schema-on-write per bucket: a new field
    /// here would normally trigger bucket rollover at the catalog
    /// layer.
    pub fields: BTreeMap<String, f64>,
}

/// A complete bucket: control block + meta value + columnar storage.
///
/// Columns are kept aligned by index — `timestamps[i]`,
/// `ingestion_timestamps[i]` (when present), and every
/// `fields[name][i]` describe the same measurement. The columnar
/// layout is what enables predicate push-down to a single column.
///
/// ## Bitemporal `ingestion_timestamps` column
///
/// Carries the engine-assigned ingestion-time stamp per measurement,
/// in parallel with the user-supplied event-time `timestamps`. The
/// column is `Vec<i64>` (not `Vec<Option<i64>>`) for cache density —
/// a sentinel of `i64::MIN` would have to encode "unset" if we ever
/// needed it, but `from_measurements` populates this column for
/// every measurement at write time.
///
/// **Backward compatibility.** Buckets serialized before the
/// bitemporal axis landed (slice B/C of G103) have no
/// `ingestion_timestamps` field on the wire. `#[serde(default)]`
/// gives them an empty `Vec` on decode; reads check `is_empty()` to
/// fall back to the "pre-bitemporal" path that surfaces `None` for
/// ingestion-ts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bucket {
    /// Header — see [`BucketControl`].
    pub control: BucketControl,
    /// Meta-field value shared by every measurement in the bucket.
    pub meta: rmpv::Value,
    /// Event-time column (microseconds), aligned with every field
    /// column.
    pub timestamps: Vec<i64>,
    /// Ingestion-time column (microseconds since epoch), aligned
    /// with `timestamps`. Empty on legacy buckets; populated by
    /// freshly-written buckets via the catalog's `IngestionClock`.
    #[serde(default)]
    pub ingestion_timestamps: Vec<i64>,
    /// Per-field columns, all of length `control.count`.
    pub fields: BTreeMap<String, Vec<f64>>,
}

impl Bucket {
    /// Build a v1 bucket from a measurement vector. Measurements are
    /// stored in arrival order — sorting is the closure-time job and
    /// belongs to the catalog/compactor.
    ///
    /// The `ingestion_timestamps` column is populated when every
    /// measurement carries `Some(ingestion_ts_us)`. If any
    /// measurement has `None` the column is left empty (legacy
    /// bucket shape) — this guards against a half-stamped bucket
    /// where some rows have ingestion-ts and others don't, which
    /// would silently misalign reads.
    ///
    /// # Examples
    ///
    /// ```
    /// use coordinode_modality::{Bucket, Measurement};
    /// use std::collections::BTreeMap;
    ///
    /// let mut fields = BTreeMap::new();
    /// fields.insert("temp".into(), 22.5);
    /// let bucket = Bucket::from_measurements(
    ///     rmpv::Value::String("sensor-1".into()),
    ///     vec![Measurement { timestamp_us: 100, ingestion_ts_us: Some(200), fields }],
    /// );
    /// assert_eq!(bucket.control.count, 1);
    /// assert_eq!(bucket.control.time_min_us, 100);
    /// assert_eq!(bucket.ingestion_timestamps, vec![200]);
    /// ```
    pub fn from_measurements(meta: rmpv::Value, measurements: Vec<Measurement>) -> Self {
        let control = BucketControl::from_measurements(&measurements);
        let mut timestamps = Vec::with_capacity(measurements.len());
        let mut ingestion_timestamps = Vec::with_capacity(measurements.len());
        let mut all_have_ingestion_ts = true;
        let mut fields: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for m in measurements {
            timestamps.push(m.timestamp_us);
            match m.ingestion_ts_us {
                Some(its) => ingestion_timestamps.push(its),
                None => all_have_ingestion_ts = false,
            }
            for (name, value) in m.fields {
                fields
                    .entry(name)
                    .or_insert_with(|| Vec::with_capacity(control.count as usize))
                    .push(value);
            }
        }
        // Half-stamped bucket — clear the column rather than ship a
        // mismatched-length one that would silently misalign reads.
        if !all_have_ingestion_ts {
            ingestion_timestamps.clear();
        }
        Self {
            control,
            meta,
            timestamps,
            ingestion_timestamps,
            fields,
        }
    }

    /// Iterate measurements in storage order. Reconstructs each row
    /// from the columnar layout. Yields the bitemporal
    /// `ingestion_ts_us` when present; `None` on legacy
    /// (pre-bitemporal) buckets where the column is empty.
    pub fn measurements(&self) -> impl Iterator<Item = Measurement> + '_ {
        let have_ingestion = !self.ingestion_timestamps.is_empty();
        (0..self.control.count as usize).map(move |i| {
            let mut fields = BTreeMap::new();
            for (name, col) in &self.fields {
                if let Some(v) = col.get(i) {
                    fields.insert(name.clone(), *v);
                }
            }
            Measurement {
                timestamp_us: self.timestamps[i],
                ingestion_ts_us: if have_ingestion {
                    self.ingestion_timestamps.get(i).copied()
                } else {
                    None
                },
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
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalTimeSeriesStore, TimeSeriesStore, Bucket, Measurement};
    /// # use coordinode_core::graph::node::NodeId;
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # use std::collections::BTreeMap;
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalTimeSeriesStore::new(engine);
    /// let bucket = Bucket::from_measurements(
    ///     rmpv::Value::String("s1".into()),
    ///     vec![Measurement { timestamp_us: 0, ingestion_ts_us: None, fields: BTreeMap::new() }],
    /// );
    /// store.put_bucket(0, NodeId::from_raw(1), &bucket)?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn put_bucket(&self, shard_id: u16, bucket_id: NodeId, bucket: &Bucket) -> StoreResult<()>;

    /// Read the bucket body. Returns `None` when the key is absent.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalTimeSeriesStore, TimeSeriesStore};
    /// # use coordinode_core::graph::node::NodeId;
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalTimeSeriesStore::new(engine);
    /// let _bucket = store.get_bucket(0, NodeId::from_raw(1))?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn get_bucket(&self, shard_id: u16, bucket_id: NodeId) -> StoreResult<Option<Bucket>>;

    /// Tombstone the bucket. Idempotent on a missing key. Does *not*
    /// touch the overflow segment — use [`Self::compact_overflow`]
    /// (with an empty merged bucket) or call this *after* draining
    /// overflow.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalTimeSeriesStore, TimeSeriesStore};
    /// # use coordinode_core::graph::node::NodeId;
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalTimeSeriesStore::new(engine);
    /// store.delete_bucket(0, NodeId::from_raw(1))?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn delete_bucket(&self, shard_id: u16, bucket_id: NodeId) -> StoreResult<()>;

    /// Mark the bucket closed by updating its control block in place.
    /// Reads `bucket`, sets `control.closed = true`, writes back.
    /// Returns `false` if no bucket exists at that key.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalTimeSeriesStore, TimeSeriesStore};
    /// # use coordinode_core::graph::node::NodeId;
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalTimeSeriesStore::new(engine);
    /// let _ok = store.mark_closed(0, NodeId::from_raw(1))?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn mark_closed(&self, shard_id: u16, bucket_id: NodeId) -> StoreResult<bool>;

    /// Re-open a previously closed bucket so the catalog's late-arrival
    /// Tier-2 path can append into it (ADR-027). Returns:
    ///
    /// - `Ok(true)` — bucket existed and `closed` flipped from `true`
    ///   to `false` (or was already `false`).
    /// - `Ok(false)` — bucket does not exist; nothing was written.
    ///
    /// This implementation reads-then-writes; concurrent writers
    /// against the same bucket key must serialise at the catalog
    /// layer above. CAS-equivalent atomicity is enforced because the
    /// engine is single-writer-per-key.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalTimeSeriesStore, TimeSeriesStore};
    /// # use coordinode_core::graph::node::NodeId;
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalTimeSeriesStore::new(engine);
    /// let _ok = store.reopen_bucket(0, NodeId::from_raw(1))?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn reopen_bucket(&self, shard_id: u16, bucket_id: NodeId) -> StoreResult<bool>;

    /// Append one late measurement to the overflow segment under
    /// `(label_id, bucket_id)`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{
    /// #     LocalTimeSeriesStore, TimeSeriesStore, Measurement, OverflowEntry,
    /// # };
    /// # use coordinode_core::graph::node::NodeId;
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # use std::collections::BTreeMap;
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalTimeSeriesStore::new(engine);
    /// let entry = OverflowEntry {
    ///     arrival_seqno: 1,
    ///     measurement: Measurement { timestamp_us: 100, ingestion_ts_us: None, fields: BTreeMap::new() },
    /// };
    /// store.put_overflow(7, NodeId::from_raw(1), &entry)?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn put_overflow(
        &self,
        label_id: u32,
        bucket_id: NodeId,
        entry: &OverflowEntry,
    ) -> StoreResult<()>;

    /// All overflow entries for one bucket, in arrival_seqno order.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalTimeSeriesStore, TimeSeriesStore};
    /// # use coordinode_core::graph::node::NodeId;
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalTimeSeriesStore::new(engine);
    /// let _entries = store.scan_overflow(7, NodeId::from_raw(1))?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn scan_overflow(&self, label_id: u32, bucket_id: NodeId) -> StoreResult<Vec<OverflowEntry>>;

    /// Atomic compact: write the merged base bucket and tombstone
    /// every overflow key currently visible. The two writes land in
    /// one [`WriteBatch`] so a crash cannot leave overflow visible
    /// against an already-rewritten base.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalTimeSeriesStore, TimeSeriesStore, Bucket};
    /// # use coordinode_core::graph::node::NodeId;
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalTimeSeriesStore::new(engine);
    /// let merged = Bucket::from_measurements(rmpv::Value::Nil, vec![]);
    /// store.compact_overflow(0, 7, NodeId::from_raw(1), &merged, &[1, 2])?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn compact_overflow(
        &self,
        shard_id: u16,
        label_id: u32,
        bucket_id: NodeId,
        merged: &Bucket,
        overflow_seqnos: &[u64],
    ) -> StoreResult<()>;

    /// Enumerate every `(label_id, bucket_id)` pair that currently
    /// has at least one overflow entry. Returns pairs in arbitrary
    /// order. Used by the background overflow compactor driver to
    /// discover candidate buckets without an external index — the
    /// engine answers "which buckets are stale?" by scanning the
    /// `ts_overflow:` prefix and folding to unique `(label, bucket)`
    /// pairs.
    ///
    /// Cost: one prefix scan over `ts_overflow:`. The scan walks
    /// every overflow entry but only stores one `(label, bucket)`
    /// pair per group — memory is bounded by the number of stale
    /// buckets, not the overflow volume.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalTimeSeriesStore, TimeSeriesStore};
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalTimeSeriesStore::new(engine);
    /// for (label_id, bucket_id) in store.list_overflow_buckets()? {
    ///     println!("compact candidate: ({label_id}, {})", bucket_id.as_raw());
    /// }
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn list_overflow_buckets(&self) -> StoreResult<Vec<(u32, NodeId)>>;
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
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use coordinode_modality::{
    ///     LocalTimeSeriesStore, TimeSeriesStore, Bucket, Measurement,
    /// };
    /// use coordinode_core::graph::node::NodeId;
    /// use std::collections::BTreeMap;
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/store"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm,
    /// # )]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// let store = LocalTimeSeriesStore::new(engine);
    /// let mut fields = BTreeMap::new();
    /// fields.insert("temp".into(), 22.5);
    /// let bucket = Bucket::from_measurements(
    ///     rmpv::Value::String("sensor-1".into()),
    ///     vec![Measurement { timestamp_us: 100, ingestion_ts_us: None, fields }],
    /// );
    /// store.put_bucket(0, NodeId::from_raw(42), &bucket)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
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

    fn reopen_bucket(&self, shard_id: u16, bucket_id: NodeId) -> StoreResult<bool> {
        let key = encode_node_key(shard_id, bucket_id);
        let Some(bytes) = self.engine.get(Partition::Node, &key)? else {
            return Ok(false);
        };
        let mut bucket = decode_bucket(bytes.as_ref())?;
        if !bucket.control.closed {
            return Ok(true);
        }
        bucket.control.closed = false;
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

    fn list_overflow_buckets(&self) -> StoreResult<Vec<(u32, NodeId)>> {
        // Overflow keys are `ts_overflow:<label_id_u32_BE>:<bucket_id_u64_BE>:<seqno_u64_BE>`.
        // After the OVERFLOW_PREFIX, the next 4 + 8 bytes uniquely
        // identify the `(label_id, bucket_id)` pair. Scan, decode
        // those two fields, fold to a set to drop duplicates from
        // multiple overflow rows under the same bucket.
        let iter = self.engine.prefix_scan(Partition::Idx, OVERFLOW_PREFIX)?;
        let mut seen: std::collections::HashSet<(u32, u64)> = std::collections::HashSet::new();
        for guard in iter {
            let (key, _value) = guard.into_inner()?;
            // Skip malformed entries instead of erroring — a corrupt
            // overflow key shouldn't take down the compactor driver.
            let suffix = match key.get(OVERFLOW_PREFIX.len()..) {
                Some(s) if s.len() >= 12 => s,
                _ => {
                    tracing::warn!(
                        "list_overflow_buckets: skipping malformed overflow key (len={})",
                        key.len(),
                    );
                    continue;
                }
            };
            let label_id = u32::from_be_bytes([suffix[0], suffix[1], suffix[2], suffix[3]]);
            let bucket_raw = u64::from_be_bytes([
                suffix[4], suffix[5], suffix[6], suffix[7], suffix[8], suffix[9], suffix[10],
                suffix[11],
            ]);
            seen.insert((label_id, bucket_raw));
        }
        Ok(seen
            .into_iter()
            .map(|(l, b)| (l, NodeId::from_raw(b)))
            .collect())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    /// Logic-test fixture (memory backing, env-flippable). Bucket
    /// CRUD + overflow routing tests verify ts-store contracts,
    /// not persistence.
    fn mk_engine() -> coordinode_test_fixtures::EngineFixture {
        coordinode_test_fixtures::engine_for_logic()
    }

    fn mk_measurement(ts: i64, temp: f64) -> Measurement {
        let mut fields = BTreeMap::new();
        fields.insert("temperature".to_owned(), temp);
        Measurement {
            timestamp_us: ts,
            ingestion_ts_us: None,
            fields,
        }
    }

    #[test]
    fn bucket_round_trip() {
        let fx = mk_engine();
        let engine = &fx.engine;
        let store = LocalTimeSeriesStore::new(engine);
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
        let store = LocalTimeSeriesStore::new(engine);
        let bid = NodeId::from_raw(60);

        let first = OverflowEntry {
            arrival_seqno: 7,
            measurement: mk_measurement(100, 1.0),
        };
        let second = OverflowEntry {
            arrival_seqno: 7,
            measurement: mk_measurement(200, 2.0),
        };
        store.put_overflow(1, bid, &first).unwrap();
        store.put_overflow(1, bid, &second).unwrap();
        let entries = store.scan_overflow(1, bid).unwrap();
        assert_eq!(entries.len(), 1, "second write must overwrite first");
        assert_eq!(entries[0].measurement.timestamp_us, 200);
    }

    #[test]
    fn concurrent_put_overflow_distinct_seqnos_converges() {
        // Four threads write distinct arrival_seqnos into the same
        // bucket's overflow segment. All four must be visible after
        // join, sorted by arrival_seqno on scan.
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
                    let store = LocalTimeSeriesStore::new(&engine);
                    let entry = OverflowEntry {
                        arrival_seqno: t + 1,
                        measurement: mk_measurement((t as i64 + 1) * 100, t as f64),
                    };
                    store.put_overflow(label, bid, &entry).expect("put");
                })
            })
            .collect();
        for h in handles {
            h.join().expect("join");
        }

        let store = LocalTimeSeriesStore::new(&engine);
        let entries = store.scan_overflow(label, bid).expect("scan");
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
        let store = LocalTimeSeriesStore::new(engine);
        assert!(store.get_bucket(0, NodeId::from_raw(99)).unwrap().is_none());
    }

    #[test]
    fn delete_is_idempotent() {
        let fx = mk_engine();
        let engine = &fx.engine;
        let store = LocalTimeSeriesStore::new(engine);
        store.delete_bucket(0, NodeId::from_raw(7)).unwrap();
        store.delete_bucket(0, NodeId::from_raw(7)).unwrap();
    }

    #[test]
    fn mark_closed_sets_flag_and_is_idempotent() {
        let fx = mk_engine();
        let engine = &fx.engine;
        let store = LocalTimeSeriesStore::new(engine);
        let bucket = Bucket::from_measurements(rmpv::Value::Nil, vec![mk_measurement(1, 1.0)]);
        let id = NodeId::from_raw(11);
        store.put_bucket(0, id, &bucket).unwrap();
        assert!(store.mark_closed(0, id).unwrap());
        assert!(store.mark_closed(0, id).unwrap());
        let read = store.get_bucket(0, id).unwrap().unwrap();
        assert!(read.control.closed);
    }

    #[test]
    fn reopen_bucket_flips_closed_back_to_false() {
        let fx = mk_engine();
        let engine = &fx.engine;
        let store = LocalTimeSeriesStore::new(engine);
        let bucket = Bucket::from_measurements(rmpv::Value::Nil, vec![mk_measurement(1, 1.0)]);
        let id = NodeId::from_raw(31);
        store.put_bucket(0, id, &bucket).unwrap();
        assert!(store.mark_closed(0, id).unwrap());
        assert!(store.reopen_bucket(0, id).unwrap());
        let read = store.get_bucket(0, id).unwrap().unwrap();
        assert!(!read.control.closed);
    }

    #[test]
    fn reopen_bucket_on_already_open_is_idempotent() {
        let fx = mk_engine();
        let engine = &fx.engine;
        let store = LocalTimeSeriesStore::new(engine);
        let bucket = Bucket::from_measurements(rmpv::Value::Nil, vec![mk_measurement(1, 1.0)]);
        let id = NodeId::from_raw(32);
        store.put_bucket(0, id, &bucket).unwrap();
        // Never closed — reopen returns true and leaves closed=false.
        assert!(store.reopen_bucket(0, id).unwrap());
        let read = store.get_bucket(0, id).unwrap().unwrap();
        assert!(!read.control.closed);
    }

    #[test]
    fn reopen_missing_bucket_returns_false() {
        let fx = mk_engine();
        let engine = &fx.engine;
        let store = LocalTimeSeriesStore::new(engine);
        assert!(!store.reopen_bucket(0, NodeId::from_raw(404)).unwrap());
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
        let store = LocalTimeSeriesStore::new(engine);
        let id = NodeId::from_raw(50);
        let label = 11u32;

        let base = Bucket::from_measurements(
            rmpv::Value::String("sensor".into()),
            vec![mk_measurement(100, 1.0), mk_measurement(200, 2.0)],
        );
        store.put_bucket(0, id, &base).unwrap();
        assert!(store.mark_closed(0, id).unwrap());

        // Tier 3: stash a late measurement in overflow.
        let late = OverflowEntry {
            arrival_seqno: 1,
            measurement: mk_measurement(150, 1.5),
        };
        store.put_overflow(label, id, &late).unwrap();

        // Catalog decides this bucket is hot again → reopen.
        assert!(store.reopen_bucket(0, id).unwrap());
        let mid = store.get_bucket(0, id).unwrap().unwrap();
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
        store.compact_overflow(0, label, id, &merged, &[1]).unwrap();

        let after = store.get_bucket(0, id).unwrap().unwrap();
        assert_eq!(after.control.count, 3);
        assert_eq!(after.control.time_min_us, 100);
        assert_eq!(after.control.time_max_us, 200);
        assert!(store.scan_overflow(label, id).unwrap().is_empty());
    }

    #[test]
    fn mark_closed_missing_returns_false() {
        let fx = mk_engine();
        let engine = &fx.engine;
        let store = LocalTimeSeriesStore::new(engine);
        assert!(!store.mark_closed(0, NodeId::from_raw(404)).unwrap());
    }

    #[test]
    fn overflow_round_trip_sorted_by_seqno() {
        let fx = mk_engine();
        let engine = &fx.engine;
        let store = LocalTimeSeriesStore::new(engine);
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
        let fx = mk_engine();
        let engine = &fx.engine;
        let store = LocalTimeSeriesStore::new(engine);
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
    fn list_overflow_buckets_returns_unique_pairs_across_multiple_entries() {
        let fx = mk_engine();
        let engine = &fx.engine;
        let store = LocalTimeSeriesStore::new(engine);

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
        store
            .put_overflow(7, NodeId::from_raw(42), &entry_a1)
            .unwrap();
        store
            .put_overflow(7, NodeId::from_raw(42), &entry_a2)
            .unwrap();
        // Bucket B: label=7, bucket=99, one entry.
        let entry_b = OverflowEntry {
            arrival_seqno: 1,
            measurement: m(300),
        };
        store
            .put_overflow(7, NodeId::from_raw(99), &entry_b)
            .unwrap();
        // Bucket C: label=8 (different label), bucket=42 (same bucket_id as A but
        // different (label, bucket) pair), one entry.
        let entry_c = OverflowEntry {
            arrival_seqno: 1,
            measurement: m(400),
        };
        store
            .put_overflow(8, NodeId::from_raw(42), &entry_c)
            .unwrap();

        let mut listed = store.list_overflow_buckets().expect("list");
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
        let store = LocalTimeSeriesStore::new(engine);
        let listed = store.list_overflow_buckets().expect("list");
        assert!(listed.is_empty());
    }

    #[test]
    fn compact_overflow_writes_base_and_deletes_overflow_atomically() {
        let fx = mk_engine();
        let engine = &fx.engine;
        let store = LocalTimeSeriesStore::new(engine);
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
