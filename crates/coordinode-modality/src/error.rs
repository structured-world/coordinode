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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    /// Module docstring promises CapacityExhausted flows through
    /// StoreError::Storage with the typed variant preserved (so the
    /// gRPC layer can drill into the chain and surface
    /// RESOURCE_EXHAUSTED). Verify by constructing one and matching.
    #[test]
    fn capacity_exhausted_preserved_through_store_error() {
        let storage_err = StorageError::CapacityExhausted {
            endpoint_id: "ep1".to_owned(),
            used_bytes: 100,
            hard_limit_bytes: 100,
        };
        let store_err: StoreError = storage_err.into();
        match store_err {
            StoreError::Storage(StorageError::CapacityExhausted {
                endpoint_id,
                used_bytes,
                hard_limit_bytes,
            }) => {
                assert_eq!(endpoint_id, "ep1");
                assert_eq!(used_bytes, 100);
                assert_eq!(hard_limit_bytes, 100);
            }
            other => panic!("expected wrapped CapacityExhausted, got {other:?}"),
        }
    }

    /// Decode error variant retains kind + message verbatim.
    #[test]
    fn decode_error_carries_kind_and_message() {
        let err = StoreError::Decode {
            kind: "test thing",
            message: "boom".to_owned(),
        };
        let rendered = format!("{err}");
        assert!(rendered.contains("test thing"));
        assert!(rendered.contains("boom"));
    }

    /// Invariant variant exists separately from Storage / Decode so
    /// callers can distinguish "I broke a precondition" from "the
    /// engine broke".
    #[test]
    fn invariant_error_is_distinguishable() {
        let err = StoreError::Invariant("bad state".into());
        assert!(matches!(err, StoreError::Invariant(_)));
    }
}
