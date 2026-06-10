//! Query planner: AST → logical plan → physical plan.
//!
//! The planner converts a validated Cypher AST into a logical plan tree
//! of relational algebra operators (TRAVERSE, AGGREGATE, PROJECT, etc.).
//! EXPLAIN output is also generated from the logical plan.

pub mod builder;
pub mod logical;
pub mod push_down;

pub use builder::{
    annotate_vector_top_k, apply_hnsw_scan_access_path, build_logical_plan,
    optimize_index_selection, optimize_push_down, PlanError,
};
pub use logical::{
    estimate_cost, estimate_cost_with_stats, AggregateItem, CostEstimate, LogicalOp, LogicalPlan,
    ProjectItem,
};
pub use push_down::{
    alpha_from_selectivity, cost_acorn_filtered, cost_graph_first, cost_vector_first,
    select_push_down_strategy, PushDownDecision, PushDownReason, PushDownStrategy,
    VectorIndexParams,
};
