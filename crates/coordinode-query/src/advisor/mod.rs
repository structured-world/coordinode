//! Query Advisor: built-in query optimization recommendations.
//!
//! The advisor tracks query execution patterns via fingerprinting and provides
//! actionable suggestions for index creation, query rewrites, and schema changes.
//!
//! ## Components
//!
//! - **Fingerprint**: Normalizes queries (strips literals) and computes a 64-bit hash.
//!   Queries that differ only by parameter values share the same fingerprint.
//!
//! - **Stats**: Lock-free latency histogram for streaming percentile estimation.
//!
//! - **Registry**: Thread-safe fingerprint registry with LFU eviction (CE: 1,000 entries).
//!   Zero-allocation hot path for recording stats on known fingerprints.

pub mod detectors;
pub mod fingerprint;
pub mod nplus1;
pub mod procedures;
pub mod registry;
pub mod source;
pub(crate) mod stats;
pub mod suggest;

pub use fingerprint::{fingerprint, normalize, normalize_and_fingerprint};
pub use procedures::{DismissedSet, ProcedureContext, ProcedureRow, execute_procedure};
pub use registry::{QueryRegistry, QueryStats};
pub use source::{
    SourceContext, SourceLocationSnapshot, extract_from_bolt_extra, extract_from_http_headers,
};
pub use suggest::{ExplainSuggestResult, Severity, Suggestion, SuggestionKind};
