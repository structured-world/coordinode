//! Configuration for the [`crate::BucketCatalog`]. Defaults match
//! `arch/core/timeseries.md` §Rollover triggers.

use std::time::Duration;

use crate::error::{CatalogError, CatalogResult};

/// Number of concurrent stripes the catalog spreads its open-bucket
/// map across. Each stripe carries an independent `RwLock`. Power of
/// two so the mod-reduction is a single mask.
pub const STRIPE_COUNT: usize = 32;

/// Tuneable parameters for the catalog. Arch defaults are sized for
/// IoT-style workloads (high-cardinality meta, small measurements).
#[derive(Debug, Clone)]
pub struct CatalogConfig {
    /// Force a rollover once a bucket holds this many measurements.
    /// Arch default: 10_000.
    pub max_count: u32,

    /// Force a rollover once the serialised bucket exceeds this size
    /// in bytes. Arch default: 4 MiB (BlobStore threshold —
    /// individual nodes above this go through BlobStore indirection,
    /// which is a different storage path).
    pub max_size_bytes: u32,

    /// Force a rollover once `time_max_us - time_min_us` exceeds
    /// this many microseconds. Maps to the schema-declared
    /// granularity (SECONDS → 1 h, MINUTES → 1 d, HOURS → 30 d).
    /// The catalog only sees the resolved microsecond value.
    pub granularity_span: Duration,
}

impl CatalogConfig {
    /// Arch-default configuration: 10 000 measurements / 4 MiB /
    /// 1 hour granularity span (matches SECONDS-granularity series).
    pub fn arch_defaults() -> Self {
        Self {
            max_count: 10_000,
            max_size_bytes: 4 * 1024 * 1024,
            granularity_span: Duration::from_secs(3_600),
        }
    }

    /// Validate the config. Returns [`CatalogError::InvalidConfig`]
    /// if any field would cause a degenerate catalog (zero count
    /// limit, zero size limit, zero granularity span).
    pub fn validate(&self) -> CatalogResult<()> {
        if self.max_count == 0 {
            return Err(CatalogError::InvalidConfig("max_count must be > 0"));
        }
        if self.max_size_bytes == 0 {
            return Err(CatalogError::InvalidConfig("max_size_bytes must be > 0"));
        }
        if self.granularity_span.is_zero() {
            return Err(CatalogError::InvalidConfig("granularity_span must be > 0"));
        }
        Ok(())
    }
}

impl Default for CatalogConfig {
    fn default() -> Self {
        Self::arch_defaults()
    }
}
