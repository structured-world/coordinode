//! Physical query executor: runs logical plans against CoordiNode storage storage.
//!
//! Implements the iterator model for read operations:
//! NodeScan → Traverse → Filter → Project → Aggregate → Sort → Limit/Skip.

pub mod eval;
pub mod eval_neutral;
pub mod row;
pub mod runner;
pub mod vector_predicate;

pub use row::Row;
pub use runner::{
    execute, execute_no_commit, AdaptiveConfig, ExecutionContext, ExecutionError, FeedbackCache,
    WriteStats,
};
