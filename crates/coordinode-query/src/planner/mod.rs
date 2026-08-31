//! Query planner: AST → logical plan → physical plan.
//!
//! The planner converts a validated Cypher AST into a logical plan tree
//! of relational algebra operators (TRAVERSE, AGGREGATE, PROJECT, etc.).
//! EXPLAIN output is also generated from the logical plan.

pub mod builder;
pub mod expr_lower;
pub mod logical;
pub mod push_down;

pub use builder::{
    PlanError, annotate_vector_top_k, apply_hnsw_scan_access_path, build_logical_plan,
    optimize_index_selection, optimize_push_down, vector_index_definition_from_clause,
};
pub use expr_lower::lower_expr;
pub use logical::{
    AggregateItem, CostEstimate, LogicalOp, LogicalPlan, ProjectItem, estimate_cost,
    estimate_cost_with_stats,
};
pub use push_down::{
    PushDownDecision, PushDownReason, PushDownStrategy, VectorIndexParams, alpha_from_selectivity,
    cost_acorn_filtered, cost_graph_first, cost_vector_first, select_push_down_strategy,
};
