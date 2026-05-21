//! Time-series above-store catalog (G103, Slice A).
//!
//! This crate implements **BucketCatalog** — the per-shard in-memory
//! state machine that sits **above** [`coordinode_modality::TimeSeriesStore`]
//! and turns individual measurement INSERTs into batched bucket
//! writes. Measurements stream in via [`BucketCatalog::write_measurement`],
//! get accumulated in per-bucket buffers under striped locks, and
//! flush as whole-bucket [`coordinode_modality::Bucket`] writes when
//! a rollover trigger fires (size / count / time / schema change).
//!
//! ## What this crate is (Slice A scope)
//!
//! - **In-memory open-bucket map** keyed by `(label_id, meta_hash)`,
//!   sharded across 32 stripes for concurrent writes.
//! - **Rollover detection** — produces a flush + close when:
//!     - measurement count in the bucket exceeds `Config.max_count`
//!       (arch default 10_000), OR
//!     - serialised bucket size exceeds `Config.max_size_bytes`
//!       (arch default 4 MiB, the BlobStore threshold), OR
//!     - time span (max_ts − min_ts) exceeds the granularity's span
//!       (`Config.granularity_span`), OR
//!     - schema change — incoming measurement has fields that don't
//!       match the bucket's accumulated schema.
//! - **Tier 1 in-buffer late-arrival absorption** — measurements
//!   whose `timestamp_us` falls within the open bucket's time
//!   window are sorted on flush; no Raft re-open round-trip.
//! - **`flush_all`** — explicit drain hook (test harnesses, graceful
//!   shutdown, time-tick driver).
//!
//! ## What this crate is NOT yet (deferred to Slices B and C)
//!
//! - **Tier 2 bucket re-open** — needs the catalog's `recently_closed`
//!   LRU + Raft-CAS-equivalent re-open serialisation.
//! - **Tier 3 overflow segment routing** — needs an "is the targeted
//!   bucket compacted-and-closed" check before falling back to
//!   `TimeSeriesStore::put_overflow`.
//! - **Bitemporal `__ingestion_ts__` axis** — needs the HLC source
//!   plumbed in (engine-assigned per measurement).
//! - **Background overflow compactor** — periodic job that calls
//!   `TimeSeriesStore::compact_overflow` once the overflow set
//!   exceeds the configured count / age threshold.
//!
//! These all build on top of the foundation here — the public
//! catalog surface stays stable across the slices.
//!
//! ## Multi-instance positioning (CLAUDE.md checklist)
//!
//! The catalog is **per-shard**, not global, so each shard's
//! `BucketCatalog` instance is the single writer for its shard's
//! open buckets. In CE 3-node HA the catalog runs on the shard's
//! Raft leader; on failover a fresh catalog is built from the
//! recovered open-bucket state (a future task will persist the
//! catalog's reverse-lookup table; today it rebuilds lazily on
//! first write per `(label_id, meta_hash)`).

#![deny(clippy::unwrap_used, clippy::expect_used)]
#![warn(missing_docs)]

mod catalog;
mod config;
mod error;
mod key;
mod measurement_router;

pub use catalog::BucketCatalog;
pub use config::CatalogConfig;
pub use error::{CatalogError, CatalogResult};
pub use key::BucketKey;
