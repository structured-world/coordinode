//! Bench harness error type — wraps IO + JSON + git failures.

use thiserror::Error;

/// Errors emitted by the bench harness.
///
/// (No `#[diagnostic::on_unimplemented]` here — that attribute only
/// fires on traits; on enums it's a compiler warning. The canonical
/// error type is documented in the crate `//!` instead.)
#[derive(Debug, Error)]
pub enum BenchError {
    /// Filesystem IO.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialisation.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// Git CLI invocation failed.
    #[error("git command failed: {0}")]
    Git(String),

    /// Generic harness error with an explanation.
    #[error("{0}")]
    Other(String),
}

/// Bench harness result alias — module-internal name kept to
/// avoid colliding with [`crate::BenchReport`] in user-facing
/// re-exports.
pub type BenchResult<T> = std::result::Result<T, BenchError>;
