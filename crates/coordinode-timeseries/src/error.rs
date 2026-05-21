//! Error type for the [`crate::BucketCatalog`] surface. Wraps the
//! downstream [`coordinode_modality::StoreError`] and adds catalog-
//! specific failure modes (flush failure attribution, mis-sharded
//! write).

use thiserror::Error;

use coordinode_modality::StoreError;

/// Result alias for [`BucketCatalog`](crate::BucketCatalog) operations.
pub type CatalogResult<T> = Result<T, CatalogError>;

/// Catalog-layer error. Distinct from [`StoreError`] so callers can
/// tell apart "the underlying TimeSeriesStore wrote and failed" from
/// "the catalog rejected the write before reaching the store".
#[derive(Debug, Error)]
pub enum CatalogError {
    /// The downstream [`coordinode_modality::TimeSeriesStore`] returned an error.
    /// Errors from `put_bucket`, `mark_closed`, `reopen_bucket`,
    /// `put_overflow`, `scan_overflow`, `compact_overflow` propagate
    /// through this variant unchanged.
    #[error("time-series store: {0}")]
    Store(#[from] StoreError),

    /// The incoming measurement's `timestamp_us` is far enough in the
    /// past that even the catalog's recently-closed LRU cannot
    /// absorb it (Tier-3 territory — falls through to overflow once
    /// the routing tier lands).
    ///
    /// Slice A returns this error rather than silently dropping the
    /// measurement; callers can detect it and either retry against
    /// the overflow path (when Slice B lands) or surface a write
    /// rejection to the client.
    #[error(
        "measurement timestamp {timestamp_us} predates bucket window for ({label_id}, meta_hash {meta_hash:#x})"
    )]
    LateBeyondTier1 {
        /// Label of the time-series series.
        label_id: u16,
        /// Hash of the meta-field value identifying the series.
        meta_hash: u64,
        /// The rejected measurement's event-time timestamp (μs).
        timestamp_us: i64,
    },

    /// The catalog config was invalid (e.g. zero granularity span,
    /// zero count or size limit).
    #[error("invalid catalog configuration: {0}")]
    InvalidConfig(&'static str),
}
