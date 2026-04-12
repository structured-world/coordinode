//! Query planner: AST → logical plan → physical plan.
//!
//! The planner converts a validated Cypher AST into a logical plan tree
//! of relational algebra operators (TRAVERSE, AGGREGATE, PROJECT, etc.).
//! EXPLAIN output is also generated from the logical plan.

pub mod builder;
pub mod logical;

pub use builder::{build_logical_plan, optimize_index_selection, PlanError};
pub use logical::{
    estimate_cost, estimate_cost_with_stats, AggregateItem, CostEstimate, LogicalOp, LogicalPlan,
    ProjectItem,
};
