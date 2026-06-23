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
//!
//! ## Transaction threading (ADR-041)
//!
//! Every method threads the active [`Transaction`]. Writes
//! (`put_bucket` / `delete_bucket` / `put_overflow` /
//! `compact_overflow`) and the read-modify-write lifecycle transitions
//! (`mark_closed` / `reopen_bucket`) take `&mut Transaction` and buffer
//! their `Partition::Node` / `Partition::Idx` mutations on it, so a
//! compaction (rewrite base bucket + tombstone overflow keys) lands as
//! one commit or none. Reads (`get_bucket` / `scan_overflow` /
//! `list_overflow_buckets`) take `&Transaction` and walk the committed
//! MVCC snapshot via [`Transaction::base_prefix_scan`] /
//! [`Transaction::read_untracked`] — untracked, since the catalog
//! state machine serialises bucket access above this layer.
//!
//! [`Transaction`]: coordinode_storage::engine::transaction::Transaction

use std::collections::BTreeMap;

use coordinode_core::graph::node::{encode_node_key, NodeId};
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::transaction::Transaction;
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
/// bitemporal axis landed have no
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
    /// catalog when it flushes its in-memory buffer. Buffered on `txn`.
    fn put_bucket(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        bucket_id: NodeId,
        bucket: &Bucket,
    ) -> StoreResult<()>;

    /// Read the bucket body. Returns `None` when the key is absent.
    fn get_bucket(
        &self,
        txn: &Transaction,
        shard_id: u16,
        bucket_id: NodeId,
    ) -> StoreResult<Option<Bucket>>;

    /// Tombstone the bucket. Idempotent on a missing key. Does *not*
    /// touch the overflow segment — use [`Self::compact_overflow`]
    /// (with an empty merged bucket) or call this *after* draining
    /// overflow. Buffered on `txn`.
    fn delete_bucket(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        bucket_id: NodeId,
    ) -> StoreResult<()>;

    /// Mark the bucket closed by updating its control block in place.
    /// Reads `bucket`, sets `control.closed = true`, buffers the
    /// write-back on `txn`. Returns `false` if no bucket exists at
    /// that key.
    fn mark_closed(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        bucket_id: NodeId,
    ) -> StoreResult<bool>;

    /// Re-open a previously closed bucket so the catalog's late-arrival
    /// Tier-2 path can append into it (ADR-027). Returns:
    ///
    /// - `Ok(true)` — bucket existed and `closed` flipped from `true`
    ///   to `false` (or was already `false`).
    /// - `Ok(false)` — bucket does not exist; nothing was written.
    ///
    /// Reads-then-writes within `txn`; concurrent writers against the
    /// same bucket key must serialise at the catalog layer above.
    fn reopen_bucket(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        bucket_id: NodeId,
    ) -> StoreResult<bool>;

    /// Append one late measurement to the overflow segment under
    /// `(label_id, bucket_id)`. Buffered on `txn`.
    fn put_overflow(
        &self,
        txn: &mut Transaction,
        label_id: u32,
        bucket_id: NodeId,
        entry: &OverflowEntry,
    ) -> StoreResult<()>;

    /// All overflow entries for one bucket, in arrival_seqno order.
    fn scan_overflow(
        &self,
        txn: &Transaction,
        label_id: u32,
        bucket_id: NodeId,
    ) -> StoreResult<Vec<OverflowEntry>>;

    /// Atomic compact: write the merged base bucket and tombstone
    /// every overflow key currently visible. Both buffer on `txn` so
    /// they commit together — a crash cannot leave overflow visible
    /// against an already-rewritten base.
    fn compact_overflow(
        &self,
        txn: &mut Transaction,
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
    fn list_overflow_buckets(&self, txn: &Transaction) -> StoreResult<Vec<(u32, NodeId)>>;
}

/// CE single-shard implementation of [`TimeSeriesStore`]. Stateless —
/// all storage access flows through the [`Transaction`] passed to each
/// method (ADR-041).
///
/// Buckets land in [`Partition::Node`] keyed by the standard node
/// shard+id encoder; overflow entries live in [`Partition::Idx`]
/// under a `ts_overflow:` prefix so prefix-scan stays tight.
pub struct LocalTimeSeriesStore;

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

impl TimeSeriesStore for LocalTimeSeriesStore {
    fn put_bucket(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        bucket_id: NodeId,
        bucket: &Bucket,
    ) -> StoreResult<()> {
        let key = encode_node_key(shard_id, bucket_id);
        let bytes = encode_bucket(bucket)?;
        txn.put(Partition::Node, &key, &bytes)?;
        Ok(())
    }

    fn get_bucket(
        &self,
        txn: &Transaction,
        shard_id: u16,
        bucket_id: NodeId,
    ) -> StoreResult<Option<Bucket>> {
        let key = encode_node_key(shard_id, bucket_id);
        match txn.read_untracked(Partition::Node, &key)? {
            Some(bytes) => Ok(Some(decode_bucket(&bytes)?)),
            None => Ok(None),
        }
    }

    fn delete_bucket(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        bucket_id: NodeId,
    ) -> StoreResult<()> {
        let key = encode_node_key(shard_id, bucket_id);
        txn.delete(Partition::Node, &key)?;
        Ok(())
    }

    fn mark_closed(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        bucket_id: NodeId,
    ) -> StoreResult<bool> {
        let key = encode_node_key(shard_id, bucket_id);
        let Some(bytes) = txn.read_untracked(Partition::Node, &key)? else {
            return Ok(false);
        };
        let mut bucket = decode_bucket(&bytes)?;
        if bucket.control.closed {
            return Ok(true);
        }
        bucket.control.closed = true;
        let encoded = encode_bucket(&bucket)?;
        txn.put(Partition::Node, &key, &encoded)?;
        Ok(true)
    }

    fn reopen_bucket(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        bucket_id: NodeId,
    ) -> StoreResult<bool> {
        let key = encode_node_key(shard_id, bucket_id);
        let Some(bytes) = txn.read_untracked(Partition::Node, &key)? else {
            return Ok(false);
        };
        let mut bucket = decode_bucket(&bytes)?;
        if !bucket.control.closed {
            return Ok(true);
        }
        bucket.control.closed = false;
        let encoded = encode_bucket(&bucket)?;
        txn.put(Partition::Node, &key, &encoded)?;
        Ok(true)
    }

    fn put_overflow(
        &self,
        txn: &mut Transaction,
        label_id: u32,
        bucket_id: NodeId,
        entry: &OverflowEntry,
    ) -> StoreResult<()> {
        let key = encode_overflow_key(label_id, bucket_id, entry.arrival_seqno);
        let bytes = encode_overflow_entry(entry)?;
        txn.put(Partition::Idx, &key, &bytes)?;
        Ok(())
    }

    fn scan_overflow(
        &self,
        txn: &Transaction,
        label_id: u32,
        bucket_id: NodeId,
    ) -> StoreResult<Vec<OverflowEntry>> {
        let prefix = encode_overflow_prefix(label_id, bucket_id);
        let mut out = Vec::new();
        for (_k, v) in txn.base_prefix_scan(Partition::Idx, &prefix)? {
            out.push(decode_overflow_entry(&v)?);
        }
        out.sort_by_key(|e| e.arrival_seqno);
        Ok(out)
    }

    fn compact_overflow(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        label_id: u32,
        bucket_id: NodeId,
        merged: &Bucket,
        overflow_seqnos: &[u64],
    ) -> StoreResult<()> {
        let bucket_key = encode_node_key(shard_id, bucket_id);
        let bucket_bytes = encode_bucket(merged)?;
        // Both writes buffer on the transaction and commit together —
        // a crash cannot leave overflow visible against an
        // already-rewritten base.
        txn.put(Partition::Node, &bucket_key, &bucket_bytes)?;
        for seqno in overflow_seqnos {
            txn.delete(
                Partition::Idx,
                &encode_overflow_key(label_id, bucket_id, *seqno),
            )?;
        }
        Ok(())
    }

    fn list_overflow_buckets(&self, txn: &Transaction) -> StoreResult<Vec<(u32, NodeId)>> {
        // Overflow keys are `ts_overflow:<label_id_u32_BE>:<bucket_id_u64_BE>:<seqno_u64_BE>`.
        // After the OVERFLOW_PREFIX, the next 4 + 8 bytes uniquely
        // identify the `(label_id, bucket_id)` pair. Scan, decode
        // those two fields, fold to a set to drop duplicates from
        // multiple overflow rows under the same bucket.
        let mut seen: std::collections::HashSet<(u32, u64)> = std::collections::HashSet::new();
        for (key, _value) in txn.base_prefix_scan(Partition::Idx, OVERFLOW_PREFIX)? {
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
mod tests;
