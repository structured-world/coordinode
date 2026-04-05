//! Physical query executor: runs logical plans against CoordiNode storage storage.
//!
//! Implements the iterator model for read operations:
//! NodeScan → Traverse → Filter → Project → Aggregate → Sort → Limit/Skip.

pub mod eval;
pub mod row;
pub mod runner;

pub use row::Row;
pub use runner::{
    execute, AdaptiveConfig, ExecutionContext, ExecutionError, FeedbackCache, WriteStats,
};
