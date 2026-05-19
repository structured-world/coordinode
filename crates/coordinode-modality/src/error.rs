//! Store-layer error type.
//!
//! [`StoreError`] is the unified error returned by every Layer 4 store
//! method. It wraps the underlying [`coordinode_storage::error::StorageError`]
//! verbatim so typed engine errors (capacity-exhausted, checksum
//! mismatch, …) propagate intact through the layers above. The
//! gRPC-layer mapping in `coordinode-server` drills into the wrapped
//! variant to surface the correct status code.
//!
//! Non-storage variants cover decode failures (a corrupted MessagePack
//! value, a stale key encoding from a pre-migration era, …) and
//! invariant violations the store layer enforces above the engine
//! (e.g. a `put_label` whose `Label::name` does not match the key).

use coordinode_storage::error::StorageError;
use thiserror::Error;

/// Errors returned by Layer 4 store operations.
#[derive(Debug, Error)]
pub enum StoreError {
    /// Underlying storage engine error. Preserved verbatim — capacity
    /// exhaustion, page checksum mismatch, and other typed variants
    /// flow through without lossy stringification.
    #[error(transparent)]
    Storage(#[from] StorageError),

    /// Stored bytes do not decode into the expected typed value
    /// (corruption, format-version skew, or a write that bypassed the
    /// typed store API).
    #[error("decode failure for {kind}: {message}")]
    Decode {
        /// Kind of value being decoded (e.g. `"label schema"`,
        /// `"blob chunk"`, `"index entry"`).
        kind: &'static str,
        /// Underlying decode-error description.
        message: String,
    },

    /// Caller violated a store-level invariant (e.g. mismatched name
    /// fields, attempt to overwrite an immutable revision). These are
    /// programmer errors, not storage failures.
    #[error("invariant violation: {0}")]
    Invariant(String),
}

// Convenience: `iter_guard.into_inner()?` inside store scan loops
// produces `lsm_tree::Error`. Route through [`StorageError`] so the
// rest of the chain (capacity-exhausted, checksum, …) still drills
// through correctly.
impl From<lsm_tree::Error> for StoreError {
    fn from(e: lsm_tree::Error) -> Self {
        StoreError::Storage(StorageError::from(e))
    }
}

/// Convenient alias for the common store-method return type.
pub type StoreResult<T> = Result<T, StoreError>;
