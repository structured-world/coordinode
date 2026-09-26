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

    /// The catalog config was invalid (e.g. zero granularity span,
    /// zero count or size limit).
    #[error("invalid catalog configuration: {0}")]
    InvalidConfig(&'static str),

    /// A short-lived bucket transaction failed to commit. The catalog
    /// owns its own transaction boundaries: each logical
    /// bucket operation opens, writes, and commits one transaction;
    /// this variant carries the underlying commit failure.
    #[error("time-series transaction commit: {0}")]
    Commit(String),
}
