//! Logical query plan: relational algebra operators for Cypher queries.
//!
//! The logical plan is a tree of operators produced from the Cypher AST.
//! It represents WHAT to compute, not HOW (physical plan handles that).

use std::collections::HashMap;

use crate::cypher::ast::*;
use coordinode_core::graph::stats::StorageStats;
use coordinode_core::graph::types::{Value, VectorConsistencyMode};

/// A logical query plan: a tree of relational algebra operators.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalPlan {
    pub root: LogicalOp,
    /// AS OF TIMESTAMP: if set, read data as of this timestamp expression.
    pub snapshot_ts: Option<Expr>,
    /// Vector MVCC consistency mode for this plan.
    /// Set from session state or per-query hint. Shown in EXPLAIN output
    /// when the plan contains VectorFilter operators.
    pub vector_consistency: VectorConsistencyMode,
}

impl LogicalPlan {
    /// Replace all `Expr::Parameter` nodes in the plan with literal values.
    ///
    /// This is the primary mechanism for parameterized query execution:
    /// bind `$param` references to concrete values before the executor runs.
    pub fn substitute_params(&mut self, params: &HashMap<String, Value>) {
        if params.is_empty() {
            return;
        }
        if let Some(ref mut ts) = self.snapshot_ts {
            ts.substitute_params(params);
        }
        self.root.substitute_params(params);
    }
}

/// A logical operator in the query plan tree.
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalOp {
    /// Scan all nodes, optionally filtered by label.
    /// Produced from MATCH (n:Label).
    NodeScan {
        variable: String,
        labels: Vec<String>,
        /// Inline property filters from the pattern (e.g., {name: "Alice"}).
        property_filters: Vec<(String, Expr)>,
    },

    /// Traverse edges from a source node.
    /// Produced from MATCH (a)-[:TYPE]->(b).
    Traverse {
        input: Box<LogicalOp>,
        /// Source variable (must be in scope from input).
        source: String,
        /// Edge types to traverse.
        edge_types: Vec<String>,
        /// Traversal direction.
        direction: Direction,
        /// Target node variable.
        target_variable: String,
        /// Target node label filter.
        target_labels: Vec<String>,
        /// Variable-length path bounds.
        length: Option<LengthBound>,
        /// Edge variable (optional).
        edge_variable: Option<String>,
        /// Inline property filters on the target node.
        target_filters: Vec<(String, Expr)>,
        /// Inline property filters on the edge.
        edge_filters: Vec<(String, Expr)>,
    },

    /// B-tree index point-lookup: replaces `Filter { input: NodeScan }` when a matching index exists.
    ///
    /// Produced by the index-selection optimizer pass when:
    /// - The input is a `NodeScan` for a single label.
    /// - The WHERE predicate is `variable.property = value_expr` (equality).
    /// - A B-tree index for (label, property) is registered in the registry.
    ///
    /// EXPLAIN display: `IndexScan(variable:Label ON idx_name(property))`
    IndexScan {
        /// Variable name for the result nodes.
        variable: String,
        /// Label this scan covers.
        label: String,
        /// Index name used for the lookup.
        index_name: String,
        /// Property being looked up.
        property: String,
        /// Lookup value expression (literal or parameter).
        value_expr: Expr,
    },

    /// Filter rows by a predicate expression.
    Filter {
        input: Box<LogicalOp>,
        predicate: Expr,
    },

    /// Project columns: select, rename, compute expressions.
    Project {
        input: Box<LogicalOp>,
        items: Vec<ProjectItem>,
        distinct: bool,
    },

    /// Aggregate: GROUP BY + aggregate functions.
    Aggregate {
        input: Box<LogicalOp>,
        /// Non-aggregate expressions form the GROUP BY key.
        group_by: Vec<Expr>,
        /// Aggregate computations.
        aggregates: Vec<AggregateItem>,
    },

    /// Sort by expressions.
    Sort {
        input: Box<LogicalOp>,
        items: Vec<SortItem>,
    },

    /// Limit result count.
    Limit { input: Box<LogicalOp>, count: Expr },

    /// Skip first N results.
    Skip { input: Box<LogicalOp>, count: Expr },

    /// Cartesian product of two inputs (multiple MATCH patterns).
    CartesianProduct {
        left: Box<LogicalOp>,
        right: Box<LogicalOp>,
    },

    /// Create nodes from pattern.
    CreateNode {
        input: Option<Box<LogicalOp>>,
        variable: Option<String>,
        labels: Vec<String>,
        properties: Vec<(String, Expr)>,
    },

    /// Create an edge between existing nodes.
    CreateEdge {
        input: Box<LogicalOp>,
        source: String,
        target: String,
        edge_type: String,
        direction: Direction,
        variable: Option<String>,
        properties: Vec<(String, Expr)>,
    },

    /// Update properties/labels.
    Update {
        input: Box<LogicalOp>,
        items: Vec<SetItem>,
        /// How to handle schema violations: fail the query or skip the node.
        violation_mode: crate::cypher::ast::ViolationMode,
    },

    /// Remove properties/labels.
    RemoveOp {
        input: Box<LogicalOp>,
        items: Vec<RemoveItem>,
    },

    /// Delete nodes/edges.
    Delete {
        input: Box<LogicalOp>,
        variables: Vec<String>,
        detach: bool,
    },

    /// `ATTACH DOCUMENT`: demote a graph node to a nested DOCUMENT property on
    /// another node within a single MVCC transaction. `input` produces rows
    /// with `source_variable` and `target_variable` bound (built by the
    /// planner from the ATTACH pattern as a MATCH + Traverse chain).
    ///
    /// Semantics:
    ///   1. Read all properties from the source node.
    ///   2. Write them as a DOCUMENT into `target_property_path` on the target.
    ///   3. Delete the connecting edge.
    ///   4. Optional `TRANSFER EDGES` re-points matching edges from source to target.
    ///   5. Cascade-delete the source node and its remaining edges, or fail if
    ///      `on_remaining_fail` is true and any untransferred edges remain.
    AttachDocument {
        input: Box<LogicalOp>,
        source_variable: String,
        target_variable: String,
        edge_type: String,
        edge_direction: crate::cypher::ast::EdgeFromSource,
        target_property_path: Vec<String>,
        transfer: Option<crate::cypher::ast::TransferEdgesSpec>,
        on_conflict_replace: bool,
        on_remaining_fail: bool,
    },

    /// `DETACH DOCUMENT`: promote a nested DOCUMENT property to a graph node +
    /// edge within a single MVCC transaction. Optionally re-points existing
    /// edges on the source node to the new target via `TRANSFER EDGES`.
    DetachDocument {
        input: Box<LogicalOp>,
        /// Bound source-node variable from prior MATCH.
        source_variable: String,
        /// Non-empty property path into the source node (e.g. `["address"]`).
        property_path: Vec<String>,
        /// Variable name for the new target node.
        target_variable: String,
        /// Labels to apply to the new target node.
        target_labels: Vec<String>,
        /// Edge type connecting target to source. Already resolved (defaults
        /// derived at build time via `HAS_<UPPER_SNAKE(last_path_segment)>`).
        edge_type: String,
        /// Direction of the edge relative to the source node.
        edge_direction: crate::cypher::ast::EdgeFromSource,
        /// Optional edge variable (not yet exposed downstream).
        edge_variable: Option<String>,
        /// Optional `TRANSFER EDGES` specification.
        transfer: Option<crate::cypher::ast::TransferEdgesSpec>,
    },

    /// MERGE / MERGE ALL: match-or-create with optional ON MATCH SET / ON CREATE SET.
    ///
    /// When `multi = false` (MERGE): unique match — errors if >1 src OR >1 tgt matches.
    /// When `multi = true` (MERGE ALL): Cartesian product — for every (src, tgt) pair
    /// from all matching nodes, find-or-create the relationship.
    Merge {
        /// Pattern scan to find existing nodes/edges.
        pattern: Box<LogicalOp>,
        /// SET items to apply when pattern matches.
        on_match: Vec<SetItem>,
        /// SET items to apply when creating new pattern.
        on_create: Vec<SetItem>,
        /// `true` for MERGE ALL (Cartesian product), `false` for MERGE (unique match).
        multi: bool,
    },

    /// UPSERT MATCH: atomic upsert with ON MATCH SET / ON CREATE CREATE.
    Upsert {
        /// Pattern scan to find existing nodes/edges.
        pattern: Box<LogicalOp>,
        /// SET items to apply when pattern matches.
        on_match: Vec<SetItem>,
        /// Patterns to create when no match found.
        on_create_patterns: Vec<Pattern>,
    },

    /// ALTER LABEL: change schema mode for a label.
    /// DDL operation — modifies schema metadata, not graph data.
    AlterLabel { label: String, mode: String },

    /// CREATE TEXT INDEX: create a full-text search index on label properties.
    CreateTextIndex {
        name: String,
        label: String,
        fields: Vec<crate::cypher::ast::TextIndexFieldSpec>,
        default_language: Option<String>,
        language_override: Option<String>,
    },

    /// DROP TEXT INDEX: remove a full-text search index.
    DropTextIndex { name: String },

    /// CREATE ENCRYPTED INDEX: create a blind-index for encrypted search on a label property.
    CreateEncryptedIndex {
        name: String,
        label: String,
        property: String,
    },

    /// DROP ENCRYPTED INDEX: remove an encrypted search index.
    DropEncryptedIndex { name: String },

    /// CREATE [UNIQUE] [SPARSE] INDEX: create a B-tree index on a label property.
    CreateIndex {
        name: String,
        label: String,
        property: String,
        unique: bool,
        sparse: bool,
        /// Optional partial-index filter predicate.
        filter: Option<crate::index::definition::PartialFilter>,
    },

    /// DROP INDEX: remove a B-tree index by name.
    DropIndex { name: String },

    /// CREATE VECTOR INDEX: build an HNSW index on a label's vector property.
    ///
    /// Backfills existing nodes; future inserts/updates maintain the index incrementally.
    CreateVectorIndex {
        name: String,
        label: String,
        property: String,
        /// HNSW M parameter (default: 16).
        m: usize,
        /// HNSW ef_construction (default: 200).
        ef_construction: usize,
        /// Distance metric: cosine, euclidean, dot (default: cosine).
        metric: coordinode_core::graph::types::VectorMetric,
        /// Vector dimensionality (0 = infer from first vector seen).
        dimensions: u32,
    },

    /// DROP VECTOR INDEX: remove an HNSW vector index by name.
    DropVectorIndex { name: String },

    /// UNWIND: expand a list expression into individual rows.
    /// Produced from UNWIND expr AS variable.
    Unwind {
        input: Box<LogicalOp>,
        /// Expression that evaluates to a list.
        expr: Expr,
        /// Variable name for each element.
        variable: String,
    },

    /// Vector similarity filter: evaluates vector distance per row.
    /// Extracted from WHERE clause when planner detects vector_distance() predicate.
    /// Pipeline stage: placed after TRAVERSE, before AGGREGATE for optimal selectivity.
    VectorFilter {
        input: Box<LogicalOp>,
        /// Expression for the vector property (e.g., n.embedding).
        vector_expr: Expr,
        /// Query vector literal or parameter.
        query_vector: Expr,
        /// Distance function name: "vector_distance", "vector_similarity", etc.
        function: String,
        /// Comparison operator and threshold (e.g., < 0.5).
        /// `true` = keep rows where function(vector, query) < threshold (distance).
        /// `false` = keep rows where function(vector, query) > threshold (similarity).
        less_than: bool,
        /// Threshold value.
        threshold: f64,
        /// Optional decay field expression for COMPUTED VECTOR_DECAY.
        /// When present, the effective score is `vector_score * decay_value`.
        /// Detected from `vector_similarity(...) * decay_field > threshold` pattern.
        decay_field: Option<Expr>,
    },

    /// Edge vector search: vector-first strategy for edge HNSW indexes.
    ///
    /// When estimated fan-out > threshold, the planner selects vector-first
    /// (HNSW search → extract src/tgt → verify pattern) instead of graph-first
    /// (traverse → brute-force vector distance).
    ///
    /// Decision thresholds (from arch/search/vector.md):
    /// - < 200 edges: Graph-first (brute-force cheaper than HNSW overhead)
    /// - 200-10K + selectivity > 1%: Graph-first
    /// - 200-10K + selectivity < 1%: Vector-first
    /// - > 10K edges: Vector-first (full scan too expensive)
    EdgeVectorSearch {
        input: Box<LogicalOp>,
        /// Edge type for the HNSW index lookup.
        edge_type: String,
        /// Vector property name on the edge.
        vector_property: String,
        /// Original vector expression (e.g., r.embedding) for evaluation.
        vector_expr: Expr,
        /// Query vector expression.
        query_vector: Expr,
        /// Distance function name.
        function: String,
        /// Comparison: true = keep where distance < threshold.
        less_than: bool,
        /// Threshold value.
        threshold: f64,
        /// Source node variable (bound from input, used for result verification).
        source_variable: String,
        /// Target node variable (produced by this operator).
        target_variable: String,
        /// Edge variable (optional).
        edge_variable: Option<String>,
        /// Strategy chosen by planner.
        strategy: EdgeVectorStrategy,
    },

    /// Vector top-K search: returns K nearest rows by vector distance.
    ///
    /// Extracted from the pattern `Sort(vector_distance(n.prop, q) ASC) + Limit(K)`
    /// where `n.prop` has an HNSW vector index. Uses `HnswIndex::search(q, K)` for
    /// O(log N) retrieval instead of brute-force O(N) scan.
    ///
    /// Fallback to brute-force computation per row when:
    /// - No HNSW index exists for the (label, property) pair
    /// - Input rows do not correspond to a full NodeScan (e.g. after Filter/Traverse)
    VectorTopK {
        input: Box<LogicalOp>,
        /// Expression for the vector property (e.g., `n.embedding`).
        vector_expr: Expr,
        /// Query vector literal or parameter.
        query_vector: Expr,
        /// Distance function name: `vector_distance`, `vector_similarity`, `vector_dot`, `vector_manhattan`.
        function: String,
        /// Top-K count (from LIMIT clause).
        k: usize,
        /// Optional distance alias (e.g., `AS d` in `WITH n, vector_distance(n.emb, $q) AS d`).
        /// When present, the resulting rows include a column with the computed distance.
        distance_alias: Option<String>,
        /// Planner annotation: `Some("name, metric")` when an HNSW index exists for
        /// (label, property), e.g. `"item_emb, cosine"`. Set by `annotate_vector_top_k`.
        ///
        /// Used in two places:
        /// - EXPLAIN: `HnswScan(item_emb, cosine)` vs `BruteForce`
        /// - Executor: resolves (label, property) by index name for `search_with_loader`
        ///
        /// `None` → "BruteForce" in EXPLAIN; executor falls back to `__label__` detection.
        hnsw_index: Option<String>,
    },

    /// Full-text search filter: evaluates text_match() per row.
    /// Extracted from WHERE clause when planner detects text_match() predicate.
    TextFilter {
        input: Box<LogicalOp>,
        /// Expression for the text field (e.g., a.body).
        text_expr: Expr,
        /// Query string literal.
        query_string: String,
        /// Optional language for query tokenization (3-arg text_match).
        /// When None, uses the index's default language.
        language: Option<String>,
    },

    /// Encrypted search filter: evaluates encrypted_match() per row.
    /// Extracted from WHERE clause when planner detects encrypted_match() predicate.
    /// Uses SSE (Searchable Symmetric Encryption) token lookup via storage-backed index.
    EncryptedFilter {
        input: Box<LogicalOp>,
        /// Expression for the encrypted field (e.g., u.email).
        field_expr: Expr,
        /// Search token expression (parameter or literal bytes).
        token_expr: Expr,
    },

    /// Left outer join for OPTIONAL MATCH: if the right side produces no rows,
    /// emit the left row with NULLs for the right variables.
    LeftOuterJoin {
        left: Box<LogicalOp>,
        right: Box<LogicalOp>,
    },

    /// Shortest path between two bound variables.
    ShortestPath {
        input: Box<LogicalOp>,
        /// Source variable (must be bound).
        source: String,
        /// Target variable (must be bound).
        target: String,
        /// Edge types to traverse.
        edge_types: Vec<String>,
        /// Traversal direction.
        direction: Direction,
        /// Maximum depth (from variable-length bound, capped at 10).
        max_depth: u64,
        /// Variable to bind the resulting path.
        path_variable: String,
    },

    /// Procedure call: CALL db.advisor.suggestions() YIELD ...
    ProcedureCall {
        /// Dotted procedure name.
        procedure: String,
        /// Positional arguments (evaluated to Value).
        args: Vec<Expr>,
        /// YIELD column names (empty = all columns).
        yield_items: Vec<String>,
    },

    /// Rank Fusion (Reciprocal Rank Fusion) operator — materializes the input
    /// row set, scores each row against every method expression, assigns
    /// 1-based competition ranks per method (ties broken by node_id), then
    /// writes `Σ 1/(k + rank_i)` with `k = 60` into the `__rrf_score__`
    /// column on every output row.
    ///
    /// Produced by the planner when `rrf_score([methods…], {vector:…, text:…})`
    /// is detected in a `Project` or `Sort` expression. Rank assignment is
    /// positional over the full input — this is a materializing operator, not
    /// a per-row scalar.
    ///
    /// `shard_overfetch_cap` is `None` for single-node CE. For the distributed
    /// plan (R-HYB5) the shard-local stage sets this to `Some(K * 3)` so
    /// each shard emits its top-K×3 rows with per-method ranks, and the
    /// coordinator stage re-ranks the union without a cap.
    RankFuse {
        input: Box<LogicalOp>,
        /// Non-empty list of method expressions (e.g. `n.embedding`, `r.ctx_emb`,
        /// `n.body`). Each is resolved at execution time to either a vector
        /// (HNSW index or edge vector property, metric from index config) or
        /// text (BM25 via `TextIndexRegistry`).
        methods: Vec<Expr>,
        /// Query vector component (from Map key `vector`). Required when at
        /// least one method resolves to a vector.
        query_vector: Option<Expr>,
        /// Query text component (from Map key `text`). Required when at least
        /// one method resolves to text.
        query_text: Option<Expr>,
        /// Per-shard overfetch cap (distributed RankFuse). `None` in CE /
        /// single-node — always processes the full input. Shape ready for
        /// R-HYB5 without call-site changes.
        shard_overfetch_cap: Option<usize>,
    },

    /// Document-level aggregate score (R-HYB2c).
    ///
    /// For each input row that binds `doc_variable` to a Document node,
    /// traverse outward `HAS_CHUNK` edges, score each chunk against the
    /// query vector (cosine similarity), and compute
    /// `α·max_chunk + β·avg_chunk + γ·coverage` into the `__doc_score__`
    /// column, where `coverage = matching_chunks / total_chunks` and a
    /// chunk is "matching" when its embedding is present and cosine
    /// similarity is non-negative.
    ///
    /// Produced by the planner when `doc_score(doc, query [, α, β, γ])` or
    /// `doc_score(doc, query, {alpha, beta, gamma})` is detected in a
    /// Project / Sort expression. Correlated per input row — not a single
    /// materialising pass like `RankFuse`.
    DocScore {
        input: Box<LogicalOp>,
        /// Variable bound to a Document node whose HAS_CHUNK children get scored.
        doc_variable: String,
        /// Query vector (or expression resolving to Vec<f32>).
        query_vector: Expr,
        /// α weight on max_chunk_score. Default 0.5.
        alpha: Expr,
        /// β weight on avg_chunk_score. Default 0.3.
        beta: Expr,
        /// γ weight on coverage (matching / total). Default 0.2.
        gamma: Expr,
    },

    /// Empty input (no rows, used as leaf for standalone CREATE).
    Empty,
}

impl LogicalOp {
    /// Replace all `Expr::Parameter` nodes in this operator tree with literal values.
    pub fn substitute_params(&mut self, params: &HashMap<String, Value>) {
        match self {
            LogicalOp::NodeScan {
                property_filters, ..
            } => {
                for (_, expr) in property_filters {
                    expr.substitute_params(params);
                }
            }
            LogicalOp::Traverse {
                input,
                target_filters,
                edge_filters,
                ..
            } => {
                input.substitute_params(params);
                for (_, expr) in target_filters {
                    expr.substitute_params(params);
                }
                for (_, expr) in edge_filters {
                    expr.substitute_params(params);
                }
            }
            LogicalOp::IndexScan { value_expr, .. } => {
                value_expr.substitute_params(params);
            }
            LogicalOp::Filter { input, predicate } => {
                input.substitute_params(params);
                predicate.substitute_params(params);
            }
            LogicalOp::Project { input, items, .. } => {
                input.substitute_params(params);
                for item in items {
                    item.expr.substitute_params(params);
                }
            }
            LogicalOp::Aggregate {
                input,
                group_by,
                aggregates,
            } => {
                input.substitute_params(params);
                for expr in group_by {
                    expr.substitute_params(params);
                }
                for agg in aggregates {
                    agg.arg.substitute_params(params);
                }
            }
            LogicalOp::Sort { input, items } => {
                input.substitute_params(params);
                for item in items {
                    item.expr.substitute_params(params);
                }
            }
            LogicalOp::Limit { input, count } | LogicalOp::Skip { input, count } => {
                input.substitute_params(params);
                count.substitute_params(params);
            }
            LogicalOp::CartesianProduct { left, right }
            | LogicalOp::LeftOuterJoin { left, right } => {
                left.substitute_params(params);
                right.substitute_params(params);
            }
            LogicalOp::CreateNode {
                input, properties, ..
            } => {
                if let Some(inp) = input {
                    inp.substitute_params(params);
                }
                for (_, expr) in properties {
                    expr.substitute_params(params);
                }
            }
            LogicalOp::CreateEdge {
                input, properties, ..
            } => {
                input.substitute_params(params);
                for (_, expr) in properties {
                    expr.substitute_params(params);
                }
            }
            LogicalOp::Update {
                input,
                items,
                violation_mode: _,
            } => {
                input.substitute_params(params);
                for item in items {
                    substitute_params_in_set_item(item, params);
                }
            }
            LogicalOp::RemoveOp { input, .. } => {
                input.substitute_params(params);
            }
            LogicalOp::Delete { input, .. } => {
                input.substitute_params(params);
            }
            LogicalOp::DetachDocument {
                input, transfer, ..
            } => {
                input.substitute_params(params);
                if let Some(ref mut t) = transfer {
                    t.predicate.substitute_params(params);
                }
            }
            LogicalOp::AttachDocument {
                input, transfer, ..
            } => {
                input.substitute_params(params);
                if let Some(ref mut t) = transfer {
                    t.predicate.substitute_params(params);
                }
            }
            LogicalOp::Merge {
                pattern,
                on_match,
                on_create,
                ..
            } => {
                pattern.substitute_params(params);
                for item in on_match {
                    substitute_params_in_set_item(item, params);
                }
                for item in on_create {
                    substitute_params_in_set_item(item, params);
                }
            }
            LogicalOp::Upsert {
                pattern,
                on_match,
                on_create_patterns,
            } => {
                pattern.substitute_params(params);
                for item in on_match {
                    substitute_params_in_set_item(item, params);
                }
                for pat in on_create_patterns {
                    substitute_params_in_pattern(pat, params);
                }
            }
            LogicalOp::Unwind { input, expr, .. } => {
                input.substitute_params(params);
                expr.substitute_params(params);
            }
            LogicalOp::VectorFilter {
                input,
                vector_expr,
                query_vector,
                ..
            } => {
                input.substitute_params(params);
                vector_expr.substitute_params(params);
                query_vector.substitute_params(params);
            }
            LogicalOp::VectorTopK {
                input,
                vector_expr,
                query_vector,
                ..
            } => {
                input.substitute_params(params);
                vector_expr.substitute_params(params);
                query_vector.substitute_params(params);
            }
            LogicalOp::EdgeVectorSearch {
                input,
                vector_expr,
                query_vector,
                ..
            } => {
                input.substitute_params(params);
                vector_expr.substitute_params(params);
                query_vector.substitute_params(params);
            }
            LogicalOp::TextFilter { input, .. } => {
                input.substitute_params(params);
            }
            LogicalOp::EncryptedFilter {
                input,
                field_expr,
                token_expr,
            } => {
                input.substitute_params(params);
                field_expr.substitute_params(params);
                token_expr.substitute_params(params);
            }
            LogicalOp::ShortestPath { input, .. } => {
                input.substitute_params(params);
            }
            LogicalOp::ProcedureCall { args, .. } => {
                for arg in args {
                    arg.substitute_params(params);
                }
            }
            LogicalOp::RankFuse {
                input,
                methods,
                query_vector,
                query_text,
                ..
            } => {
                input.substitute_params(params);
                for m in methods {
                    m.substitute_params(params);
                }
                if let Some(qv) = query_vector {
                    qv.substitute_params(params);
                }
                if let Some(qt) = query_text {
                    qt.substitute_params(params);
                }
            }
            LogicalOp::DocScore {
                input,
                query_vector,
                alpha,
                beta,
                gamma,
                ..
            } => {
                input.substitute_params(params);
                query_vector.substitute_params(params);
                alpha.substitute_params(params);
                beta.substitute_params(params);
                gamma.substitute_params(params);
            }
            LogicalOp::AlterLabel { .. }
            | LogicalOp::CreateTextIndex { .. }
            | LogicalOp::DropTextIndex { .. }
            | LogicalOp::CreateEncryptedIndex { .. }
            | LogicalOp::DropEncryptedIndex { .. }
            | LogicalOp::CreateIndex { .. }
            | LogicalOp::DropIndex { .. }
            | LogicalOp::CreateVectorIndex { .. }
            | LogicalOp::DropVectorIndex { .. }
            | LogicalOp::Empty => {}
        }
    }
}

/// Substitute parameters in a SetItem's expressions.
fn substitute_params_in_set_item(item: &mut SetItem, params: &HashMap<String, Value>) {
    match item {
        SetItem::Property { expr, .. }
        | SetItem::PropertyPath { expr, .. }
        | SetItem::ReplaceProperties { expr, .. }
        | SetItem::MergeProperties { expr, .. } => {
            expr.substitute_params(params);
        }
        SetItem::DocFunction { value_expr, .. } => {
            value_expr.substitute_params(params);
        }
        SetItem::AddLabel { .. } => {}
    }
}

/// Substitute parameters in Pattern property expressions.
fn substitute_params_in_pattern(pattern: &mut Pattern, params: &HashMap<String, Value>) {
    for element in &mut pattern.elements {
        match element {
            PatternElement::Node(node) => {
                for (_, expr) in &mut node.properties {
                    expr.substitute_params(params);
                }
            }
            PatternElement::Relationship(rel) => {
                for (_, expr) in &mut rel.properties {
                    expr.substitute_params(params);
                }
            }
        }
    }
}

/// Strategy for edge vector queries: graph-first vs vector-first.
///
/// Selected by the planner based on estimated fan-out and vector selectivity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EdgeVectorStrategy {
    /// Traverse edges first, then brute-force vector distance.
    /// Optimal when fan-out is small (< 200 edges).
    GraphFirst,
    /// Search HNSW index first, then verify graph pattern.
    /// Optimal when fan-out is large (> 10K edges).
    VectorFirst,
}

impl std::fmt::Display for EdgeVectorStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GraphFirst => write!(f, "Graph-First (brute-force)"),
            Self::VectorFirst => write!(f, "Vector-First (HNSW)"),
        }
    }
}

/// Select edge vector strategy based on estimated fan-out and threshold.
///
/// Decision table from arch/search/vector.md:
/// - < 200 edges: Graph-first (brute-force cheaper than HNSW overhead)
/// - 200-10K + threshold > 0.01 (selectivity > 1%): Graph-first
/// - 200-10K + threshold <= 0.01 (selectivity < 1%): Vector-first
/// - > 10K edges: Vector-first (full scan too expensive)
pub fn select_edge_vector_strategy(
    estimated_fan_out: f64,
    vector_selectivity: f64,
) -> EdgeVectorStrategy {
    const BRUTE_FORCE_CEILING: f64 = 200.0;
    const HNSW_FLOOR: f64 = 10_000.0;
    const SELECTIVITY_THRESHOLD: f64 = 0.01;

    if estimated_fan_out < BRUTE_FORCE_CEILING {
        EdgeVectorStrategy::GraphFirst
    } else if estimated_fan_out > HNSW_FLOOR {
        EdgeVectorStrategy::VectorFirst
    } else if vector_selectivity > SELECTIVITY_THRESHOLD {
        EdgeVectorStrategy::GraphFirst
    } else {
        EdgeVectorStrategy::VectorFirst
    }
}

/// A projected column.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectItem {
    pub expr: Expr,
    pub alias: Option<String>,
}

/// An aggregate computation.
#[derive(Debug, Clone, PartialEq)]
pub struct AggregateItem {
    /// Function name (count, sum, avg, min, max, collect, etc.).
    pub function: String,
    /// Argument expression (or Star for count(*)).
    pub arg: Expr,
    /// DISTINCT modifier.
    pub distinct: bool,
    /// Output alias.
    pub alias: Option<String>,
    /// Second argument expression for percentileCont / percentileDisc.
    /// Must evaluate to a Float or Int in [0.0, 1.0].
    /// Accepts literal values (`0.9`) and query parameters (`$p`).
    /// `None` means the argument was absent or not a scalar expression — executor falls back to 0.5.
    pub percentile_expr: Option<Expr>,
}

/// Query cost estimation result.
///
/// Used by EXPLAIN to show estimated resource usage before execution.
/// Cost model: `traversal_depth × avg_fan_out × shards_touched`.
#[derive(Debug, Clone, PartialEq)]
pub struct CostEstimate {
    /// Abstract cost units (higher = more expensive).
    pub cost: f64,
    /// Estimated number of result rows.
    pub estimated_rows: f64,
    /// Estimated execution time in milliseconds.
    pub estimated_time_ms: f64,
    /// Optimization hints (e.g., "reduce traversal depth to *1..3").
    pub hints: Vec<String>,
}

/// Default assumptions for cost estimation when no statistics are available.
struct CostDefaults {
    /// Average number of nodes per label scan.
    node_count: f64,
    /// Average fan-out per edge traversal hop.
    avg_fan_out: f64,
    /// Default WHERE selectivity (fraction of rows passing filter).
    filter_selectivity: f64,
    /// Number of shards (1 for CE single-node/RF=3).
    shards: f64,
    /// Cost per millisecond of processing one row.
    ms_per_row: f64,
    /// Number of distinct labels (for NodeScan selectivity).
    label_count: f64,
}

impl Default for CostDefaults {
    fn default() -> Self {
        Self {
            node_count: 1000.0,
            avg_fan_out: 50.0,
            filter_selectivity: 0.33,
            shards: 1.0,
            ms_per_row: 0.001,
            label_count: 5.0,
        }
    }
}

/// Estimate the cost of executing a logical plan using hardcoded defaults.
///
/// For more accurate estimates with real storage data, use
/// [`estimate_cost_with_stats`].
pub fn estimate_cost(plan: &LogicalPlan) -> CostEstimate {
    estimate_cost_with_stats(plan, None)
}

/// Estimate the cost of executing a logical plan, using real storage
/// statistics when available. Falls back to hardcoded defaults for
/// any statistic not provided.
pub fn estimate_cost_with_stats(
    plan: &LogicalPlan,
    stats: Option<&dyn StorageStats>,
) -> CostEstimate {
    let defaults = match stats {
        Some(s) => {
            let total = s.total_node_count();
            let fan_out = s.avg_fan_out();
            let labels = s.label_count();
            CostDefaults {
                node_count: if total > 0 { total as f64 } else { 1000.0 },
                avg_fan_out: if fan_out > 0.0 { fan_out } else { 50.0 },
                filter_selectivity: 0.33,
                shards: 1.0,
                ms_per_row: 0.001,
                label_count: if labels > 0 { labels as f64 } else { 5.0 },
            }
        }
        None => CostDefaults::default(),
    };
    let mut hints = Vec::new();
    let (cost, rows) = estimate_op_cost(&plan.root, &defaults, stats, &mut hints);
    let time_ms = rows * defaults.ms_per_row;

    CostEstimate {
        cost,
        estimated_rows: rows,
        estimated_time_ms: time_ms,
        hints,
    }
}

/// Check if an expression tree contains a vector function call.
/// Used to apply vector_dims_factor to Filter cost.
fn expr_contains_vector_fn(expr: &Expr) -> bool {
    match expr {
        Expr::FunctionCall { name, .. } => matches!(
            name.as_str(),
            "vector_distance" | "vector_similarity" | "vector_dot" | "vector_manhattan"
        ),
        Expr::BinaryOp { left, right, .. } => {
            expr_contains_vector_fn(left) || expr_contains_vector_fn(right)
        }
        Expr::UnaryOp { expr: inner, .. } => expr_contains_vector_fn(inner),
        _ => false,
    }
}

/// Recursive cost estimation for a single operator.
/// Returns (cost_units, estimated_output_rows).
fn estimate_op_cost(
    op: &LogicalOp,
    defaults: &CostDefaults,
    stats: Option<&dyn StorageStats>,
    hints: &mut Vec<String>,
) -> (f64, f64) {
    match op {
        LogicalOp::NodeScan {
            labels,
            property_filters,
            ..
        } => {
            let base = if labels.is_empty() {
                defaults.node_count
            } else {
                // Use per-label count from stats when available
                let label_count = labels
                    .iter()
                    .filter_map(|l| stats.and_then(|s| s.node_count_for_label(l)))
                    .next();
                match label_count {
                    Some(count) => count as f64,
                    None => defaults.node_count / defaults.label_count,
                }
            };
            let rows = if property_filters.is_empty() {
                base
            } else {
                // Property filter on scan is very selective (usually unique-ish)
                (base * 0.1).max(1.0)
            };
            (rows, rows)
        }

        // IndexScan is a point-lookup: O(log N) cost, ~1 result row on average.
        LogicalOp::IndexScan { .. } => (1.0, 1.0),

        LogicalOp::Traverse {
            input,
            length,
            edge_types,
            ..
        } => {
            let (input_cost, input_rows) = estimate_op_cost(input, defaults, stats, hints);

            // Use per-edge-type fan-out from stats when available
            let fan_out = edge_types
                .iter()
                .filter_map(|et| stats.and_then(|s| s.avg_fan_out_for_type(et)))
                .next()
                .unwrap_or(defaults.avg_fan_out);

            let (depth, fan_out_total) = match length {
                Some(lb) => {
                    let min_h = lb.min.unwrap_or(1) as f64;
                    let max_h = lb.max.unwrap_or(10).min(10) as f64;
                    let avg_depth = (min_h + max_h) / 2.0;
                    // Variable-length: sum of fan_out^d for d in min..max
                    let mut total = 0.0;
                    for d in (min_h as u64)..=(max_h as u64) {
                        total += fan_out.powi(d as i32);
                    }
                    if max_h >= 5.0 {
                        hints.push(format!(
                            "variable-length path *{}..{} may be expensive \
                             (est. {} paths). Consider reducing depth to *{}..3",
                            min_h as u64, max_h as u64, total as u64, min_h as u64,
                        ));
                    }
                    (avg_depth, total)
                }
                None => {
                    // Single hop
                    (1.0, fan_out)
                }
            };

            let rows = input_rows * fan_out_total;
            let cost = input_cost + rows * depth * defaults.shards;
            (cost, rows)
        }

        LogicalOp::Filter { input, predicate } => {
            let (input_cost, input_rows) = estimate_op_cost(input, defaults, stats, hints);
            let rows = input_rows * defaults.filter_selectivity;
            // Vector functions (distance, similarity) are ~100x more expensive
            // than scalar comparisons due to per-dimension computation.
            let per_row_cost = if expr_contains_vector_fn(predicate) {
                1.0 // vector distance: ~1 cost unit per row (dims factor)
            } else {
                0.01 // scalar comparison: cheap
            };
            (input_cost + input_rows * per_row_cost, rows)
        }

        LogicalOp::Aggregate {
            input, group_by, ..
        } => {
            let (input_cost, input_rows) = estimate_op_cost(input, defaults, stats, hints);
            // Aggregate reduces rows to number of groups
            let groups = if group_by.is_empty() {
                1.0
            } else {
                // Assume ~10% unique keys
                (input_rows * 0.1).max(1.0)
            };
            (input_cost + input_rows * 0.02, groups) // aggregation cost per row
        }

        LogicalOp::Project { input, .. } => {
            let (input_cost, input_rows) = estimate_op_cost(input, defaults, stats, hints);
            (input_cost + input_rows * 0.001, input_rows) // projection is very cheap
        }

        LogicalOp::Sort { input, .. } => {
            let (input_cost, input_rows) = estimate_op_cost(input, defaults, stats, hints);
            let sort_cost = if input_rows > 1.0 {
                input_rows * input_rows.log2()
            } else {
                0.0
            };
            (input_cost + sort_cost * 0.01, input_rows)
        }

        LogicalOp::Limit { input, count } => {
            let (input_cost, input_rows) = estimate_op_cost(input, defaults, stats, hints);
            let limit_val = match count {
                Expr::Literal(Value::Int(n)) => *n as f64,
                _ => input_rows, // can't statically determine
            };
            (input_cost, input_rows.min(limit_val))
        }

        LogicalOp::Skip { input, count } => {
            let (input_cost, input_rows) = estimate_op_cost(input, defaults, stats, hints);
            let skip_val = match count {
                Expr::Literal(Value::Int(n)) => *n as f64,
                _ => 0.0,
            };
            (input_cost, (input_rows - skip_val).max(0.0))
        }

        LogicalOp::CartesianProduct { left, right } => {
            let (left_cost, left_rows) = estimate_op_cost(left, defaults, stats, hints);
            let (right_cost, right_rows) = estimate_op_cost(right, defaults, stats, hints);
            let rows = left_rows * right_rows;
            if rows > 10000.0 {
                hints.push(format!(
                    "CartesianProduct produces ~{} rows — consider adding a join predicate",
                    rows as u64,
                ));
            }
            (left_cost + right_cost + rows * 0.01, rows)
        }

        LogicalOp::LeftOuterJoin { left, right } => {
            let (left_cost, left_rows) = estimate_op_cost(left, defaults, stats, hints);
            let (right_cost, right_rows) = estimate_op_cost(right, defaults, stats, hints);
            // OPTIONAL MATCH: at least left_rows (NULLs for unmatched)
            let rows = left_rows.max(left_rows * right_rows * 0.1);
            (left_cost + right_cost + rows * 0.01, rows)
        }

        LogicalOp::VectorFilter {
            input, threshold, ..
        } => {
            let (input_cost, input_rows) = estimate_op_cost(input, defaults, stats, hints);
            let selectivity = (*threshold).clamp(0.01, 1.0);
            let rows = input_rows * selectivity;
            (input_cost + input_rows * 1.0, rows)
        }

        LogicalOp::VectorTopK { input, k, .. } => {
            let (input_cost, input_rows) = estimate_op_cost(input, defaults, stats, hints);
            // With HNSW index: O(log N * ef_search). Without: O(N log K) (partial sort).
            // Assume HNSW is available (optimistic cost estimate) — typical case after
            // the planner has already selected this operator as applicable.
            let k_f = (*k as f64).max(1.0);
            // ef_search ≈ 200 in current config, amortized log N lookup.
            let hnsw_cost = (input_rows.max(1.0).ln() * 200.0).max(1.0);
            let rows = k_f.min(input_rows);
            (input_cost + hnsw_cost, rows)
        }

        LogicalOp::TextFilter { input, .. } => {
            let (input_cost, input_rows) = estimate_op_cost(input, defaults, stats, hints);
            // Text search: tantivy index lookup is fast (inverted index), but
            // selectivity varies. Assume ~10% match rate for typical queries.
            let rows = input_rows * 0.1;
            (input_cost + input_rows * 0.5, rows) // cheaper than vector, more than scalar
        }

        LogicalOp::EncryptedFilter { input, .. } => {
            let (input_cost, input_rows) = estimate_op_cost(input, defaults, stats, hints);
            // SSE encrypted search: hash lookup is very cheap (single key prefix scan).
            // Selectivity is very low — equality match on encrypted field.
            let rows = input_rows * 0.01;
            (input_cost + input_rows * 0.1, rows)
        }

        LogicalOp::Unwind { input, .. } => {
            let (input_cost, input_rows) = estimate_op_cost(input, defaults, stats, hints);
            // Assume average list length of 5
            let rows = input_rows * 5.0;
            (input_cost + rows * 0.001, rows)
        }

        LogicalOp::ShortestPath {
            input, max_depth, ..
        } => {
            let (input_cost, input_rows) = estimate_op_cost(input, defaults, stats, hints);
            // BFS cost: O(V + E) bounded by depth
            let bfs_cost = defaults.avg_fan_out.powi((*max_depth).min(5) as i32);
            (input_cost + input_rows * bfs_cost * 0.01, input_rows)
        }

        // Write operations: cost is mainly I/O
        LogicalOp::CreateNode { input, .. } => {
            let (input_cost, input_rows) = match input {
                Some(inp) => estimate_op_cost(inp, defaults, stats, hints),
                None => (1.0, 1.0),
            };
            (input_cost + input_rows * 1.0, input_rows) // write is expensive
        }
        LogicalOp::CreateEdge { input, .. } => {
            let (input_cost, input_rows) = estimate_op_cost(input, defaults, stats, hints);
            (input_cost + input_rows * 2.0, input_rows) // 2 posting list writes
        }
        LogicalOp::Update { input, items, .. } => {
            let (input_cost, input_rows) = estimate_op_cost(input, defaults, stats, hints);
            (input_cost + input_rows * items.len() as f64, input_rows)
        }
        LogicalOp::Delete { input, .. } => {
            let (input_cost, input_rows) = estimate_op_cost(input, defaults, stats, hints);
            (input_cost + input_rows * 2.0, input_rows)
        }
        LogicalOp::DetachDocument {
            input, transfer, ..
        } => {
            let (input_cost, input_rows) = estimate_op_cost(input, defaults, stats, hints);
            // Per row: read node, create node, create edge, remove property =
            // ~4 storage ops. +5x if TRANSFER EDGES (scan all edge types).
            let base = 4.0;
            let transfer_cost = if transfer.is_some() { 5.0 } else { 0.0 };
            (input_cost + input_rows * (base + transfer_cost), input_rows)
        }
        LogicalOp::AttachDocument {
            input, transfer, ..
        } => {
            let (input_cost, input_rows) = estimate_op_cost(input, defaults, stats, hints);
            // Per row: read node, write delta, delete edge, delete node + cascade = ~5.
            let base = 5.0;
            let transfer_cost = if transfer.is_some() { 5.0 } else { 0.0 };
            (input_cost + input_rows * (base + transfer_cost), input_rows)
        }

        LogicalOp::Merge { pattern, .. } | LogicalOp::Upsert { pattern, .. } => {
            let (pat_cost, pat_rows) = estimate_op_cost(pattern, defaults, stats, hints);
            (pat_cost + pat_rows * 2.0, pat_rows.max(1.0))
        }

        LogicalOp::RemoveOp { input, .. } => {
            let (input_cost, input_rows) = estimate_op_cost(input, defaults, stats, hints);
            (input_cost + input_rows * 1.0, input_rows)
        }

        LogicalOp::EdgeVectorSearch {
            input,
            strategy,
            threshold,
            ..
        } => {
            let (input_cost, input_rows) = estimate_op_cost(input, defaults, stats, hints);
            let selectivity = (*threshold).clamp(0.01, 1.0);
            let cost_multiplier = match strategy {
                EdgeVectorStrategy::GraphFirst => 1.0,
                EdgeVectorStrategy::VectorFirst => 0.3, // HNSW is sub-linear
            };
            (
                input_cost + input_rows * cost_multiplier,
                input_rows * selectivity,
            )
        }

        LogicalOp::RankFuse {
            input,
            methods,
            shard_overfetch_cap,
            ..
        } => {
            let (input_cost, input_rows) = estimate_op_cost(input, defaults, stats, hints);
            // RankFuse materializes input; per-method scoring cost is O(N) for
            // vector similarity and O(log N) + O(matched) for text BM25 lookup.
            // Treat each method as roughly 1 cost unit per input row (similar to
            // VectorFilter). Ranks assignment is O(N log N) sort per method.
            let methods_n = methods.len() as f64;
            let score_cost = input_rows * methods_n;
            let sort_cost = if input_rows > 1.0 {
                input_rows * input_rows.log2() * methods_n
            } else {
                0.0
            };
            let output_rows = match shard_overfetch_cap {
                Some(cap) => input_rows.min(*cap as f64),
                None => input_rows,
            };
            (input_cost + score_cost + sort_cost * 0.01, output_rows)
        }

        LogicalOp::DocScore { input, .. } => {
            let (input_cost, input_rows) = estimate_op_cost(input, defaults, stats, hints);
            // Per input row: one adjacency read + one chunk-node read per HAS_CHUNK
            // child + one cosine computation per chunk. Estimate avg_fan_out chunks
            // per document, each costing roughly one vector-score unit.
            let per_row_cost = defaults.avg_fan_out;
            (input_cost + input_rows * per_row_cost, input_rows)
        }

        LogicalOp::ProcedureCall { .. } => (1.0, 10.0),
        LogicalOp::AlterLabel { .. }
        | LogicalOp::CreateTextIndex { .. }
        | LogicalOp::DropTextIndex { .. }
        | LogicalOp::CreateEncryptedIndex { .. }
        | LogicalOp::DropEncryptedIndex { .. }
        | LogicalOp::CreateIndex { .. }
        | LogicalOp::DropIndex { .. }
        | LogicalOp::CreateVectorIndex { .. }
        | LogicalOp::DropVectorIndex { .. } => (1.0, 1.0),
        LogicalOp::Empty => (0.0, 1.0),
    }
}

impl LogicalPlan {
    /// Format the plan as EXPLAIN text output with cost estimates.
    pub fn explain(&self) -> String {
        self.explain_with_stats(None)
    }

    /// Format the plan as EXPLAIN text output using real storage statistics.
    pub fn explain_with_stats(&self, stats: Option<&dyn StorageStats>) -> String {
        let mut output = String::new();
        let cost = estimate_cost_with_stats(self, stats);
        output.push_str(&format!(
            "Cost: {:.0} | Estimated rows: {:.0} | Est. time: {:.1}ms\n",
            cost.cost, cost.estimated_rows, cost.estimated_time_ms,
        ));
        for hint in &cost.hints {
            output.push_str(&format!("Hint: {hint}\n"));
        }
        // Show vector consistency mode when the plan has vector operations
        if op_contains_vector_filter(&self.root) {
            output.push_str(&format!(
                "Vector consistency: {}\n",
                self.vector_consistency.as_str()
            ));
        }
        output.push('\n');
        explain_op(&self.root, 0, &mut output);
        output
    }

    /// EXPLAIN SUGGEST: format the plan + run all suggestion detectors.
    ///
    /// Returns the standard EXPLAIN output plus ranked optimization suggestions.
    pub fn explain_suggest(&self) -> crate::advisor::suggest::ExplainSuggestResult {
        self.explain_suggest_with_stats(None, None)
    }

    /// EXPLAIN SUGGEST with real storage statistics.
    ///
    /// When `registry` is provided, the MissingIndex detector skips properties
    /// that already have an index — preventing false positive suggestions.
    pub fn explain_suggest_with_stats(
        &self,
        stats: Option<&dyn StorageStats>,
        registry: Option<&crate::index::IndexRegistry>,
    ) -> crate::advisor::suggest::ExplainSuggestResult {
        let explain = self.explain_with_stats(stats);
        let suggestions = crate::advisor::detectors::detect_suggestions(self, registry);

        crate::advisor::suggest::ExplainSuggestResult {
            explain,
            suggestions,
        }
    }
}

/// Check if a logical operator tree contains any VectorFilter, EdgeVectorSearch, or VectorTopK.
fn op_contains_vector_filter(op: &LogicalOp) -> bool {
    match op {
        LogicalOp::VectorFilter { .. }
        | LogicalOp::EdgeVectorSearch { .. }
        | LogicalOp::VectorTopK { .. } => true,
        LogicalOp::Filter { input, .. }
        | LogicalOp::Project { input, .. }
        | LogicalOp::Aggregate { input, .. }
        | LogicalOp::Sort { input, .. }
        | LogicalOp::Limit { input, .. }
        | LogicalOp::Skip { input, .. }
        | LogicalOp::Traverse { input, .. }
        | LogicalOp::Unwind { input, .. }
        | LogicalOp::TextFilter { input, .. }
        | LogicalOp::EncryptedFilter { input, .. }
        | LogicalOp::ShortestPath { input, .. }
        | LogicalOp::RankFuse { input, .. }
        | LogicalOp::DocScore { input, .. } => op_contains_vector_filter(input),
        LogicalOp::CartesianProduct { left, right } | LogicalOp::LeftOuterJoin { left, right } => {
            op_contains_vector_filter(left) || op_contains_vector_filter(right)
        }
        _ => false,
    }
}

fn explain_op(op: &LogicalOp, indent: usize, output: &mut String) {
    let prefix = "  ".repeat(indent);

    match op {
        LogicalOp::NodeScan {
            variable, labels, ..
        } => {
            if labels.is_empty() {
                output.push_str(&format!("{prefix}NodeScan({variable})\n"));
            } else {
                output.push_str(&format!(
                    "{prefix}NodeScan({variable}:{})\n",
                    labels.join(":")
                ));
            }
        }
        LogicalOp::IndexScan {
            variable,
            label,
            index_name,
            property,
            ..
        } => {
            output.push_str(&format!(
                "{prefix}IndexScan({variable}:{label} ON {index_name}({property}))\n"
            ));
        }
        LogicalOp::Traverse {
            input,
            source,
            edge_types,
            direction,
            target_variable,
            length,
            ..
        } => {
            let arrow = match direction {
                Direction::Outgoing => "->",
                Direction::Incoming => "<-",
                Direction::Both => "--",
            };
            let types = if edge_types.is_empty() {
                String::new()
            } else {
                format!(":{}", edge_types.join("|"))
            };
            let len = length.map_or(String::new(), |lb| {
                format!(
                    "*{}..{}",
                    lb.min.map_or(String::new(), |n| n.to_string()),
                    lb.max.map_or(String::new(), |n| n.to_string())
                )
            });
            output.push_str(&format!(
                "{prefix}Traverse({source} {arrow}[{types}{len}]{arrow} {target_variable})\n"
            ));
            explain_op(input, indent + 1, output);
        }
        LogicalOp::Filter { input, predicate } => {
            output.push_str(&format!("{prefix}Filter({predicate:?})\n"));
            explain_op(input, indent + 1, output);
        }
        LogicalOp::Project {
            input,
            items,
            distinct,
        } => {
            let dist = if *distinct { " DISTINCT" } else { "" };
            let cols: Vec<String> = items
                .iter()
                .map(|i| {
                    i.alias
                        .as_ref()
                        .map_or_else(|| format!("{:?}", i.expr), |a| a.clone())
                })
                .collect();
            output.push_str(&format!("{prefix}Project{dist}({})\n", cols.join(", ")));
            explain_op(input, indent + 1, output);
        }
        LogicalOp::Aggregate {
            input,
            group_by,
            aggregates,
        } => {
            let gb: Vec<String> = group_by.iter().map(|e| format!("{e:?}")).collect();
            let aggs: Vec<String> = aggregates
                .iter()
                .map(|a| {
                    a.alias
                        .as_ref()
                        .map_or_else(|| a.function.clone(), |al| al.clone())
                })
                .collect();
            output.push_str(&format!(
                "{prefix}Aggregate(group_by=[{}], aggs=[{}])\n",
                gb.join(", "),
                aggs.join(", ")
            ));
            explain_op(input, indent + 1, output);
        }
        LogicalOp::Sort { input, items } => {
            let cols: Vec<String> = items
                .iter()
                .map(|s| {
                    let dir = if s.ascending { "ASC" } else { "DESC" };
                    format!("{:?} {dir}", s.expr)
                })
                .collect();
            output.push_str(&format!("{prefix}Sort({})\n", cols.join(", ")));
            explain_op(input, indent + 1, output);
        }
        LogicalOp::Limit { input, count } => {
            output.push_str(&format!("{prefix}Limit({count:?})\n"));
            explain_op(input, indent + 1, output);
        }
        LogicalOp::Skip { input, count } => {
            output.push_str(&format!("{prefix}Skip({count:?})\n"));
            explain_op(input, indent + 1, output);
        }
        LogicalOp::CartesianProduct { left, right } => {
            output.push_str(&format!("{prefix}CartesianProduct\n"));
            explain_op(left, indent + 1, output);
            explain_op(right, indent + 1, output);
        }
        LogicalOp::CreateNode {
            input,
            variable,
            labels,
            ..
        } => {
            let var = variable.as_deref().unwrap_or("_");
            output.push_str(&format!("{prefix}CreateNode({var}:{})\n", labels.join(":")));
            if let Some(inp) = input {
                explain_op(inp, indent + 1, output);
            }
        }
        LogicalOp::CreateEdge {
            input,
            source,
            target,
            edge_type,
            ..
        } => {
            output.push_str(&format!(
                "{prefix}CreateEdge({source})-[:{edge_type}]->({target})\n"
            ));
            explain_op(input, indent + 1, output);
        }
        LogicalOp::Update {
            input,
            items,
            violation_mode: _,
        } => {
            output.push_str(&format!("{prefix}Update({} items)\n", items.len()));
            explain_op(input, indent + 1, output);
        }
        LogicalOp::RemoveOp { input, items } => {
            output.push_str(&format!("{prefix}Remove({} items)\n", items.len()));
            explain_op(input, indent + 1, output);
        }
        LogicalOp::Delete {
            input,
            variables,
            detach,
        } => {
            let d = if *detach { "DETACH " } else { "" };
            output.push_str(&format!("{prefix}{d}Delete({})\n", variables.join(", ")));
            explain_op(input, indent + 1, output);
        }
        LogicalOp::AttachDocument {
            input,
            source_variable,
            target_variable,
            edge_type,
            target_property_path,
            transfer,
            on_conflict_replace,
            on_remaining_fail,
            ..
        } => {
            let path = target_property_path.join(".");
            let mut flags = String::new();
            if transfer.is_some() {
                flags.push_str(" TRANSFER");
            }
            if *on_conflict_replace {
                flags.push_str(" REPLACE");
            }
            if *on_remaining_fail {
                flags.push_str(" REMAINING-FAIL");
            }
            output.push_str(&format!(
                "{prefix}AttachDocument(({source_variable})-[:{edge_type}]->({target_variable}) INTO {target_variable}.{path}{flags})\n",
            ));
            explain_op(input, indent + 1, output);
        }
        LogicalOp::DetachDocument {
            input,
            source_variable,
            property_path,
            target_variable,
            target_labels,
            edge_type,
            transfer,
            ..
        } => {
            let labels = if target_labels.is_empty() {
                String::new()
            } else {
                format!(":{}", target_labels.join(":"))
            };
            let path = property_path.join(".");
            let transfer_str = if transfer.is_some() { " TRANSFER" } else { "" };
            output.push_str(&format!(
                "{prefix}DetachDocument({source_variable}.{path} AS ({target_variable}{labels})-[:{edge_type}]->{source_variable}{transfer_str})\n",
            ));
            explain_op(input, indent + 1, output);
        }
        LogicalOp::Merge {
            pattern,
            on_match,
            on_create,
            multi,
        } => {
            let op_name = if *multi { "MergeMany" } else { "Merge" };
            output.push_str(&format!(
                "{prefix}{op_name}(on_match={}, on_create={})\n",
                on_match.len(),
                on_create.len()
            ));
            explain_op(pattern, indent + 1, output);
        }
        LogicalOp::Upsert {
            pattern,
            on_match,
            on_create_patterns,
        } => {
            output.push_str(&format!(
                "{prefix}Upsert(on_match={}, on_create={})\n",
                on_match.len(),
                on_create_patterns.len()
            ));
            explain_op(pattern, indent + 1, output);
        }
        LogicalOp::VectorFilter {
            input,
            function,
            threshold,
            less_than,
            decay_field,
            ..
        } => {
            let op = if *less_than { "<" } else { ">" };
            let decay_info = if decay_field.is_some() {
                " * decay"
            } else {
                ""
            };
            output.push_str(&format!(
                "{prefix}VectorFilter({function}{decay_info} {op} {threshold})\n"
            ));
            explain_op(input, indent + 1, output);
        }
        LogicalOp::EdgeVectorSearch {
            input,
            edge_type,
            function,
            threshold,
            less_than,
            strategy,
            ..
        } => {
            let op = if *less_than { "<" } else { ">" };
            output.push_str(&format!(
                "{prefix}EdgeVectorSearch(:{edge_type}, {function} {op} {threshold}, strategy: {strategy})\n"
            ));
            explain_op(input, indent + 1, output);
        }
        LogicalOp::VectorTopK {
            input,
            function,
            k,
            distance_alias,
            hnsw_index,
            ..
        } => {
            let alias_info = match distance_alias {
                Some(a) => format!(" AS {a}"),
                None => String::new(),
            };
            let strategy = match hnsw_index {
                Some(idx) => format!("HnswScan({idx})"),
                None => "BruteForce".to_string(),
            };
            output.push_str(&format!(
                "{prefix}VectorTopK({function} k={k}{alias_info}, strategy: {strategy})\n"
            ));
            explain_op(input, indent + 1, output);
        }
        LogicalOp::TextFilter {
            input,
            query_string,
            language,
            ..
        } => {
            if let Some(lang) = language {
                output.push_str(&format!(
                    "{prefix}TextFilter(text_match \"{query_string}\", language: \"{lang}\")\n"
                ));
            } else {
                output.push_str(&format!(
                    "{prefix}TextFilter(text_match \"{query_string}\")\n"
                ));
            }
            explain_op(input, indent + 1, output);
        }
        LogicalOp::EncryptedFilter { input, .. } => {
            output.push_str(&format!("{prefix}EncryptedFilter(encrypted_match)\n"));
            explain_op(input, indent + 1, output);
        }
        LogicalOp::Unwind {
            input,
            expr,
            variable,
        } => {
            output.push_str(&format!("{prefix}Unwind({expr:?} AS {variable})\n"));
            explain_op(input, indent + 1, output);
        }
        LogicalOp::LeftOuterJoin { left, right } => {
            output.push_str(&format!("{prefix}LeftOuterJoin\n"));
            explain_op(left, indent + 1, output);
            explain_op(right, indent + 1, output);
        }
        LogicalOp::ShortestPath {
            input,
            source,
            target,
            edge_types,
            max_depth,
            path_variable,
            ..
        } => {
            let types = if edge_types.is_empty() {
                String::new()
            } else {
                format!(":{}", edge_types.join("|"))
            };
            output.push_str(&format!(
                "{prefix}ShortestPath({path_variable} = ({source})-[{types}*..{max_depth}]->({target}))\n"
            ));
            explain_op(input, indent + 1, output);
        }
        LogicalOp::ProcedureCall {
            procedure, args, ..
        } => {
            output.push_str(&format!(
                "{prefix}ProcedureCall ({procedure}, {} args)\n",
                args.len()
            ));
        }
        LogicalOp::RankFuse {
            input,
            methods,
            query_vector,
            query_text,
            shard_overfetch_cap,
        } => {
            let method_strs: Vec<String> = methods.iter().map(|m| format!("{m:?}")).collect();
            let mut query_parts = Vec::new();
            if query_vector.is_some() {
                query_parts.push("vector");
            }
            if query_text.is_some() {
                query_parts.push("text");
            }
            let cap_str = match shard_overfetch_cap {
                Some(cap) => format!(", cap={cap}"),
                None => String::new(),
            };
            output.push_str(&format!(
                "{prefix}RankFuse(methods=[{}], query={{{}}}, k=60{cap_str})\n",
                method_strs.join(", "),
                query_parts.join(", "),
            ));
            explain_op(input, indent + 1, output);
        }
        LogicalOp::DocScore {
            input,
            doc_variable,
            alpha,
            beta,
            gamma,
            ..
        } => {
            output.push_str(&format!(
                "{prefix}DocScore({doc_variable} -[:HAS_CHUNK]-> chunks, α={alpha:?}, β={beta:?}, γ={gamma:?})\n",
            ));
            explain_op(input, indent + 1, output);
        }
        LogicalOp::AlterLabel { label, mode } => {
            output.push_str(&format!("{prefix}AlterLabel({label}, SET SCHEMA {mode})\n"));
        }
        LogicalOp::CreateTextIndex {
            name,
            label,
            fields,
            default_language,
            ..
        } => {
            let lang = default_language.as_deref().unwrap_or("english");
            let props: Vec<&str> = fields.iter().map(|f| f.property.as_str()).collect();
            output.push_str(&format!(
                "{prefix}CreateTextIndex({name} ON :{label}({}), DEFAULT LANGUAGE {lang})\n",
                props.join(", ")
            ));
        }
        LogicalOp::DropTextIndex { name } => {
            output.push_str(&format!("{prefix}DropTextIndex({name})\n"));
        }
        LogicalOp::CreateEncryptedIndex {
            name,
            label,
            property,
        } => {
            output.push_str(&format!(
                "{prefix}CreateEncryptedIndex({name} ON :{label}({property}))\n"
            ));
        }
        LogicalOp::DropEncryptedIndex { name } => {
            output.push_str(&format!("{prefix}DropEncryptedIndex({name})\n"));
        }
        LogicalOp::CreateIndex {
            name,
            label,
            property,
            unique,
            sparse,
            filter,
        } => {
            let mut flags = String::new();
            if *unique {
                flags.push_str(" UNIQUE");
            }
            if *sparse {
                flags.push_str(" SPARSE");
            }
            let filter_str = if let Some(f) = filter {
                format!(", filter={f:?}")
            } else {
                String::new()
            };
            output.push_str(&format!(
                "{prefix}CreateIndex({name}{flags} ON :{label}({property}){filter_str})\n"
            ));
        }
        LogicalOp::DropIndex { name } => {
            output.push_str(&format!("{prefix}DropIndex({name})\n"));
        }
        LogicalOp::CreateVectorIndex {
            name,
            label,
            property,
            m,
            ef_construction,
            metric,
            dimensions,
        } => {
            output.push_str(&format!(
                "{prefix}CreateVectorIndex({name} ON :{label}({property}), m={m}, ef={ef_construction}, metric={metric:?}, dim={dimensions})\n"
            ));
        }
        LogicalOp::DropVectorIndex { name } => {
            output.push_str(&format!("{prefix}DropVectorIndex({name})\n"));
        }
        LogicalOp::Empty => {
            output.push_str(&format!("{prefix}Empty\n"));
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // --- EdgeVectorStrategy selection tests ---

    /// Fan-out < 200 → always Graph-first regardless of selectivity.
    #[test]
    fn strategy_low_fanout_graph_first() {
        assert_eq!(
            select_edge_vector_strategy(50.0, 0.5),
            EdgeVectorStrategy::GraphFirst
        );
        assert_eq!(
            select_edge_vector_strategy(199.0, 0.001),
            EdgeVectorStrategy::GraphFirst
        );
        assert_eq!(
            select_edge_vector_strategy(1.0, 0.001),
            EdgeVectorStrategy::GraphFirst
        );
    }

    /// Fan-out > 10K → always Vector-first regardless of selectivity.
    #[test]
    fn strategy_high_fanout_vector_first() {
        assert_eq!(
            select_edge_vector_strategy(10_001.0, 0.5),
            EdgeVectorStrategy::VectorFirst
        );
        assert_eq!(
            select_edge_vector_strategy(100_000.0, 0.99),
            EdgeVectorStrategy::VectorFirst
        );
    }

    /// Fan-out 200-10K + selectivity > 1% → Graph-first.
    #[test]
    fn strategy_mid_fanout_high_selectivity_graph_first() {
        assert_eq!(
            select_edge_vector_strategy(500.0, 0.05),
            EdgeVectorStrategy::GraphFirst
        );
        assert_eq!(
            select_edge_vector_strategy(5_000.0, 0.5),
            EdgeVectorStrategy::GraphFirst
        );
    }

    /// Fan-out 200-10K + selectivity < 1% → Vector-first.
    #[test]
    fn strategy_mid_fanout_low_selectivity_vector_first() {
        assert_eq!(
            select_edge_vector_strategy(500.0, 0.005),
            EdgeVectorStrategy::VectorFirst
        );
        assert_eq!(
            select_edge_vector_strategy(9_999.0, 0.001),
            EdgeVectorStrategy::VectorFirst
        );
    }

    /// Boundary: fan-out exactly 200 → mid range, depends on selectivity.
    #[test]
    fn strategy_boundary_200() {
        // At exactly 200, enters the mid range
        assert_eq!(
            select_edge_vector_strategy(200.0, 0.5),
            EdgeVectorStrategy::GraphFirst
        );
        assert_eq!(
            select_edge_vector_strategy(200.0, 0.005),
            EdgeVectorStrategy::VectorFirst
        );
    }

    /// Boundary: fan-out exactly 10K → mid range.
    #[test]
    fn strategy_boundary_10k() {
        assert_eq!(
            select_edge_vector_strategy(10_000.0, 0.5),
            EdgeVectorStrategy::GraphFirst
        );
        assert_eq!(
            select_edge_vector_strategy(10_000.0, 0.005),
            EdgeVectorStrategy::VectorFirst
        );
    }

    /// Display formatting for strategies.
    #[test]
    fn strategy_display() {
        assert_eq!(
            EdgeVectorStrategy::GraphFirst.to_string(),
            "Graph-First (brute-force)"
        );
        assert_eq!(
            EdgeVectorStrategy::VectorFirst.to_string(),
            "Vector-First (HNSW)"
        );
    }
}
