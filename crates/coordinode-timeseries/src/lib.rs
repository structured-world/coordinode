//! Time-series above-store catalog.
//!
//! **no-std tier:** `std-only`. The catalog uses `std::sync::RwLock`
//! (striped concurrency), `std::time::SystemTime` (rollover decisions,
//! LRU TTL), and `std::collections::HashMap` (open-bucket map). The
//! production [`PersistentMonotonicHlcClock`] also embeds an
//! `Arc<StorageEngine>` for restart-monotonicity persistence. None
//! of these are replaceable below `std`, so this crate legitimately
//! occupies the `std-only` tier.
//!
//! This crate implements **BucketCatalog** — the per-shard in-memory
//! state machine that sits **above** [`coordinode_modality::TimeSeriesStore`]
//! and turns individual measurement INSERTs into batched bucket
//! writes. Measurements stream in via [`BucketCatalog::write_measurement`],
//! get accumulated in per-bucket buffers under striped locks, and
//! flush as whole-bucket [`coordinode_modality::Bucket`] writes when
//! a rollover trigger fires (size / count / time / schema change).
//!
//! ## What this crate is
//!
//! - **In-memory open-bucket map** keyed by `(label_id, meta_hash)`,
//!   sharded across 32 stripes for concurrent writes.
//! - **Rollover detection** — produces a flush + close when:
//!     - measurement count in the bucket exceeds `Config.max_count`
//!       (default 10_000), OR
//!     - serialised bucket size exceeds `Config.max_size_bytes`
//!       (default 4 MiB, the BlobStore threshold), OR
//!     - time span (max_ts − min_ts) exceeds the granularity's span
//!       (`Config.granularity_span`), OR
//!     - schema change — incoming measurement has fields that don't
//!       match the bucket's accumulated schema.
//! - **Tier 1 in-buffer late-arrival absorption** — measurements
//!   whose `timestamp_us` falls within the open bucket's time
//!   window are sorted on flush; no Raft re-open round-trip.
//! - **Tier 2 bucket re-open** — a measurement that falls inside a
//!   recently closed bucket's window reopens that bucket through the
//!   per-stripe `recently_closed` LRU.
//! - **Tier 3 overflow segments** — a measurement for a bucket past
//!   its re-open window goes to that bucket's overflow segment via
//!   `TimeSeriesStore::put_overflow`.
//! - **Bitemporal `__ingestion_ts__` axis** — engine-assigned per
//!   measurement for labels declared bitemporal.
//! - **Overflow compaction** — `compact_all_pending` folds overflow
//!   sets past the threshold back into their base buckets; the
//!   catalog's owner drives it on its own schedule.
//! - **`flush_all`** — explicit drain hook (test harnesses, graceful
//!   shutdown, time-tick driver).
//!
//! ## Multi-instance positioning
//!
//! The catalog is **per-shard**, not global, so each shard's
//! `BucketCatalog` instance is the single writer for its shard's
//! open buckets. In CE 3-node HA the catalog runs on the shard's
//! Raft leader; on failover a fresh catalog is built from the
//! recovered open-bucket state. The catalog's reverse-lookup table is
//! not persisted: it rebuilds lazily on the first write per
//! `(label_id, meta_hash)`.

#![deny(clippy::unwrap_used, clippy::expect_used)]
#![warn(missing_docs)]

mod catalog;
mod clock;
mod config;
mod error;
mod key;
mod measurement_router;

pub use catalog::BucketCatalog;
#[cfg(any(test, feature = "test-clock"))]
pub use clock::ScriptedClock;
pub use clock::{
    IngestionClock, MonotonicHlcClock, PersistentMonotonicHlcClock, load_last_stamp,
    persist_last_stamp,
};
pub use config::CatalogConfig;
pub use error::{CatalogError, CatalogResult};
pub use key::BucketKey;
