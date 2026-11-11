pub mod blob;
pub mod cache;
#[cfg(feature = "columnar")]
pub mod columnar;
pub mod compress;
pub mod engine;
pub mod error;
pub mod format;
pub mod oplog;
pub mod placement;
pub mod scrub;

#[cfg(test)]
pub(crate) mod internal_test_helpers;
#[cfg(test)]
#[allow(clippy::expect_used)]
pub(crate) mod test_alloc;

/// Re-export the `Guard` trait so downstream crates can call `into_inner()`
/// on `IterGuardImpl` without directly depending on `lsm_tree`.
pub use lsm_tree::Guard;
