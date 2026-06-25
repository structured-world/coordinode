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
    /// R-SNAP1: cross-modality read-consistency mode for this plan.
    ///
    /// Set from an explicit `/*+ read_consistency('mode') */` hint or
    /// auto-promoted by the planner when the query touches >1 modality.
    /// Drives `applied_watermark.wait_for(snapshot_ts, timeout)` at the
    /// executor boundary when it is `Snapshot` or `Exact`.
    pub read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode,
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

/// Pushed-down time-slice filter for a temporal edge traversal.
///
/// Bounds the per-version edgeprop prefix scan: only versions whose
/// `valid_from <= upper_ms` (when set) are read. `lower_ms` is a value-side
/// filter applied after decode — `valid_to` lives in the row, not in the key.
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalFilter {
    /// Edge variable this filter applies to (matches `edge_variable` on Traverse).
    pub edge_variable: String,
    /// Inclusive upper bound on `valid_from` (epoch ms). `None` means open
    /// upward.
    pub upper_ms: Option<i64>,
    /// Strict lower bound on `valid_to` (epoch ms). A version qualifies if
    /// `valid_to > lower_ms` OR `valid_to IS NULL`. `None` means no `valid_to`
    /// constraint.
    pub lower_ms: Option<i64>,
}

/// Score-fusion kernel for hybrid sparse+dense retrieval via the
/// `RankFuse` operator. Selects how per-method scores combine into a
/// single `__hybrid_score__` column.
#[derive(Debug, Clone, PartialEq)]
pub enum FusionStrategy {
    /// Reciprocal Rank Fusion: `Σ 1 / (k + rank_i)`. Rank-based, ignores
    /// raw score scale, robust to outliers. Default `k = 60` matches the
    /// Cormack et al. 2009 baseline.
    Rrf { k: u32 },
    /// Convex Combination: `Σ w_i × min_max_normalised(method_i)`. Score-
    /// aware; weights must be positive but are not normalised by the
    /// operator (caller decides). Methods with min == max contribute
    /// zero (degenerate range, can't normalise).
    ConvexCombination {
        weights: std::collections::BTreeMap<String, f64>,
    },
    /// Distribution-Based Score Fusion: `Σ w_i × z_score(method_i)`.
    /// Score-aware, robust to skewed distributions, but requires a
    /// reasonable sample size (≥ ~20 rows) to estimate μ and σ.
    /// Methods with σ == 0 contribute zero.
    Dbsf {
        weights: std::collections::BTreeMap<String, f64>,
    },
}

impl Default for FusionStrategy {
    fn default() -> Self {
        Self::Rrf { k: 60 }
    }
}

/// Cheap-to-evaluate predicate over a single node, pushed into the HNSW
/// search path so the traversal can skip non-matching branches.
///
/// Restricted on purpose: every variant must be evaluable from a node id
/// alone (after one point-get on `Partition::Node`), without joins or
/// secondary scans. Anything more expensive belongs in a regular
/// post-filter and not in this descriptor.
#[derive(Debug, Clone, PartialEq)]
pub enum VectorPredicate {
    /// Node carries this primary label.
    LabelEq(String),
    /// Node has a property whose value equals the literal.
    PropertyEq {
        property: String,
        value: coordinode_core::graph::types::Value,
    },
    /// Node has a numeric property satisfying a comparison with a literal.
    /// Used for VDBBench-style range filters (`n.id >= 100`,
    /// `n.score < 0.5`). Non-numeric stored values reject the node.
    PropertyCmp {
        property: String,
        op: NumericCmp,
        /// Always a numeric literal at plan time. Stored as a `Value` so
        /// the predicate matches the shape of `PropertyEq` and the
        /// evaluator handles both with one msgpack lookup.
        value: coordinode_core::graph::types::Value,
    },
    /// Conjunction of two predicates; both must hold.
    And(Box<VectorPredicate>, Box<VectorPredicate>),
}

/// Six-way numeric comparison set used by [`VectorPredicate::PropertyCmp`].
/// Stable wire identity: variants are part of the EXPLAIN contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericCmp {
    Gt,
    Ge,
    Lt,
    Le,
}

/// A logical operator in the query plan tree.
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalOp {
    /// An extension operator dispatched to a registered `ExtensionHandler`
    /// (the EE server registers handlers at startup). `name` selects the
    /// handler; `payload` is the opaque, extension-defined config the
    /// parser-extension produced. CE registers no handlers, so this variant
    /// never appears in a plan built by a pure-CE engine.
    Extension { name: String, payload: Vec<u8> },
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
        /// Pushed-down time-slice filter for temporal edges. When set, the
        /// executor bounds the per-version edgeprop prefix scan instead of
        /// loading every stored version and filtering above. Inferred by
        /// `lift_temporal_filter` from a `temporal_active_at` or canonical
        /// `valid_from <= $T AND (valid_to IS NULL OR valid_to > $T)`
        /// predicate elsewhere in the query.
        temporal_filter: Option<TemporalFilter>,
        /// Named-path variable from `p = (a)-[:R*]->(b)`. When set, the
        /// traversal reconstructs and binds the route from source to each
        /// reached target as a path value (the executor forces the
        /// sequential path so the predecessor chain stays exact).
        path_variable: Option<String>,
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

    /// `UNION` / `UNION ALL`: concatenate the result sets of two or more
    /// independent query branches. Every branch must project the same column
    /// names in the same order. When `all` is `false` (plain `UNION`) the
    /// combined result is de-duplicated; `true` (`UNION ALL`) keeps all rows.
    Union { inputs: Vec<LogicalOp>, all: bool },

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

    /// `MERGE NODES (a, b) INTO <target>` — native node-merge operation (R180).
    ///
    /// `input` produces rows binding `source_a` and `source_b` to node columns
    /// (built by the planner from a preceding MATCH). For each input row, the
    /// executor collapses the two nodes into `target` within a single MVCC
    /// transaction: property merge → edge transfer → DETACH DELETE non-target.
    MergeNodes {
        input: Box<LogicalOp>,
        source_a: String,
        source_b: String,
        target: String,
        conflict: crate::cypher::ast::MergeNodesConflictStrategy,
        transfer_edges: Option<crate::cypher::ast::TransferEdgesEndpoints>,
        duplicate: crate::cypher::ast::MergeNodesDuplicateStrategy,
        transfer_edge_properties: bool,
    },

    /// CLONE NODE: deep-copy a bound node into a fresh node, optionally cloning
    /// its incident edges. `input` binds `source`; the clone is bound to
    /// `target`. Single MVCC transaction; goes through the create path (fresh
    /// id, index registration). See arch/compatibility/native-procedures.md.
    CloneNode {
        input: Box<LogicalOp>,
        source: String,
        target: String,
        with_edges: bool,
        with_properties: bool,
        set_items: Vec<SetItem>,
    },

    /// REDIRECT EDGES: re-point a bound node's edges onto another bound node.
    /// `input` binds both `source` and `target`. Single MVCC transaction;
    /// adjacency moves via posting-list merge operators. See
    /// arch/compatibility/native-procedures.md.
    RedirectEdges {
        input: Box<LogicalOp>,
        source: String,
        target: String,
        edge_types: Option<Vec<String>>,
        direction: crate::cypher::ast::RedirectDirection,
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
        /// In-RAM quantization codec, resolved from the Cypher OPTIONS
        /// `quantization` field (or `None` default). Lifted to the
        /// logical layer so the executor can pass it straight through
        /// to `VectorIndexConfig` without re-parsing the string.
        quantization: coordinode_vector::hnsw::QuantizationCodec,
        /// Reader behaviour while the index is in the Building state,
        /// resolved from the Cypher `online_during_build` OPTIONS field.
        online_during_build: crate::index::OnlineDuringBuild,
        /// Default HNSW `ef_search` from the Cypher `ef_search` OPTIONS field
        /// (`None` = engine default 200). Stored on the index definition.
        ef_search: Option<usize>,
        /// Default rerank-candidate count from the Cypher `rerank_candidates`
        /// OPTIONS field (`None` = engine default 100).
        rerank_candidates: Option<usize>,
    },

    /// DROP VECTOR INDEX: remove an HNSW vector index by name.
    DropVectorIndex { name: String },

    /// CREATE EDGE TYPE: declare an edge-type schema entry.
    ///
    /// When `temporal = true`, instances of this edge type carry a
    /// `(valid_from, valid_to)` time interval and the storage layer
    /// keeps one edgeprop entry per `valid_from`, allowing multiple
    /// versions of the edge to coexist between the same source/target.
    CreateEdgeType {
        name: String,
        temporal: bool,
        properties: Vec<crate::cypher::ast::EdgePropertyDecl>,
    },

    /// CREATE NODE TYPE: declare a node-label schema entry (R172a per ADR-027).
    ///
    /// When `temporal = true`, every node of this label carries the bitemporal
    /// `(valid_from, valid_to)` interval and the engine-assigned
    /// `__ingestion_ts__` field; the storage layer keeps one node record per
    /// `(node_id, valid_from)` so multiple versions of the same node coexist.
    /// The TEMPORAL flag is immutable for the lifetime of the label.
    CreateNodeType {
        name: String,
        temporal: bool,
        properties: Vec<crate::cypher::ast::EdgePropertyDecl>,
    },

    /// CREATE TRIGGER — the trigger architecture. Registers a trigger definition in the
    /// schema partition, updates the `(target, event)` index, and (in EE)
    /// notifies trigger workers of the new subscription.
    CreateTrigger {
        clause: crate::cypher::ast::CreateTriggerClause,
    },
    /// DROP TRIGGER — the trigger architecture.
    DropTrigger { name: String },
    /// SHOW TRIGGERS — the trigger architecture. Reads schema partition and returns one row per
    /// registered trigger.
    ShowTriggers,
    /// ALTER TRIGGER — the trigger architecture.
    AlterTrigger {
        clause: crate::cypher::ast::AlterTriggerClause,
    },

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
        /// Graph predicate push-down decision (R-PUSH1). Populated by the
        /// `optimize_push_down` planner pass when the upstream input contains
        /// a `Traverse`. `None` means either the input does not contain a
        /// traversal (no push-down applicable) or the optimizer pass was not
        /// invoked. The invariant contract test asserts that any plan with
        /// a `Traverse` directly preceding a `VectorFilter` carries
        /// `Some(_)` here — see [`optimize_push_down`] and the regression
        /// test `vector_filter_after_traverse_is_always_annotated`.
        push_down: Option<crate::planner::push_down::PushDownDecision>,
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
        /// Pushed-down graph predicate for ACORN-style filtered search.
        ///
        /// Set by the planner when a sibling MATCH / WHERE clause restricts the
        /// candidate set to a label or a small property predicate that can be
        /// evaluated cheaply from a node id. When set, the executor routes
        /// through `HnswIndex::search_with_visibility` with a closure built
        /// from this descriptor, so the HNSW traversal can prune branches that
        /// can't pass the predicate. None falls back to the unfiltered search.
        predicate: Option<VectorPredicate>,
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
        /// Fusion kernel — RRF (default, rank-based) or score-aware variants.
        /// Defaults to `Rrf { k: 60 }` so existing callers see no behavioural
        /// change. New callers wire up `cc_score(...)` or `dbsf_score(...)`.
        fusion: FusionStrategy,
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

    /// Late-interaction (ColBERT-style) top-K via the MaxSim scalar.
    ///
    /// Extracted from the pattern `Sort(maxsim_score(d.tokens, q) DESC) +
    /// Limit(K)` over a Project input. Replaces the generic
    /// Sort + Limit pipeline with a single bounded min-heap pass:
    /// O(N log K) instead of O(N log N) plus avoids materialising the
    /// full sort buffer.
    ///
    /// Brute-force over the input rows in v1; a PLAID-style centroid
    /// pruning pre-step will plug in here when added.
    MaxSimTopK {
        input: Box<LogicalOp>,
        /// Expression for the document's multi-vector property
        /// (e.g. `n.token_embeddings`).
        doc_expr: Expr,
        /// Query matrix expression (literal or parameter).
        query_expr: Expr,
        /// Top-K count (from LIMIT clause).
        k: usize,
        /// Optional score alias (`AS s` in `RETURN ... AS s`). When set,
        /// the resulting rows include a column with the computed score.
        score_alias: Option<String>,
    },

    /// HNSW index access path: a SOURCE operator (peer of NodeScan).
    ///
    /// Replaces the `NodeScan -> [star Project] -> Sort(vector fn) ->
    /// Limit(k)` chain when the query is a pure vector top-K over one
    /// label with a registered HNSW index and no intermediate filters.
    /// The executor asks the index for the top-k candidates and
    /// point-fetches ONLY those k nodes from storage, making the query
    /// O(k) storage reads instead of the O(N) full-label
    /// materialisation that scan-then-rank pays.
    ///
    /// Queries with additional predicates keep the NodeScan +
    /// VectorTopK path (which composes with filtered HNSW search).
    HnswScan {
        /// Label whose index serves the scan.
        label: String,
        /// Vector property the index covers.
        property: String,
        /// Row binding name for the fetched node (the MATCH variable).
        binding: String,
        /// Query vector expression (parameter or literal).
        query_vector: Expr,
        /// Top-K count (from the LIMIT clause).
        k: usize,
        /// Distance function from the original ORDER BY
        /// (`vector_distance` / `vector_similarity` / ...). Determines
        /// result ordering semantics and the score column value.
        function: String,
        /// Optional alias binding the computed distance into result rows.
        distance_alias: Option<String>,
        /// Registered index name (for EXPLAIN and executor lookup).
        index_name: String,
    },

    /// `FOREACH (variable IN list | body)`: for each input row and each element
    /// of `list` (evaluated per row), bind `variable` and run the `body`
    /// sub-plan of updating operators. Pass-through: the input rows continue
    /// downstream unchanged. The body's leaf is `Empty`, which the executor
    /// fills with the per-iteration scope (input row + bound `variable`).
    Foreach {
        input: Box<LogicalOp>,
        variable: String,
        list: Expr,
        body: Box<LogicalOp>,
    },

    /// `CALL { subquery }` / `OPTIONAL CALL { subquery }`: run the `body`
    /// sub-plan per input row, joining its result columns onto each outer row.
    /// A correlated body (leading `WITH` importing outer variables) reads the
    /// per-row scope injected at its `Empty` leaf; an uncorrelated body scans
    /// independently. When `optional` is set and the body yields no rows, the
    /// outer row is preserved (NULLs for the subquery columns).
    CallSubquery {
        input: Box<LogicalOp>,
        body: Box<LogicalOp>,
        optional: bool,
    },

    /// Empty input (no rows, used as leaf for standalone CREATE).
    Empty,
}

impl LogicalOp {
    /// Replace all `Expr::Parameter` nodes in this operator tree with literal values.
    pub fn substitute_params(&mut self, params: &HashMap<String, Value>) {
        match self {
            // Opaque extension payload carries no Expr params to substitute.
            LogicalOp::Extension { .. } => {}
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
            LogicalOp::Union { inputs, .. } => {
                for input in inputs {
                    input.substitute_params(params);
                }
            }
            LogicalOp::Foreach {
                input, list, body, ..
            } => {
                input.substitute_params(params);
                list.substitute_params(params);
                body.substitute_params(params);
            }
            LogicalOp::CallSubquery { input, body, .. } => {
                input.substitute_params(params);
                body.substitute_params(params);
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
            LogicalOp::HnswScan { query_vector, .. } => {
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
            LogicalOp::MaxSimTopK {
                input,
                doc_expr,
                query_expr,
                ..
            } => {
                input.substitute_params(params);
                doc_expr.substitute_params(params);
                query_expr.substitute_params(params);
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
            LogicalOp::MergeNodes {
                input, conflict, ..
            } => {
                input.substitute_params(params);
                if let crate::cypher::ast::MergeNodesConflictStrategy::SetExpressions(items) =
                    conflict
                {
                    for item in items {
                        substitute_params_in_set_item(item, params);
                    }
                }
            }
            LogicalOp::CloneNode {
                input, set_items, ..
            } => {
                input.substitute_params(params);
                for item in set_items {
                    substitute_params_in_set_item(item, params);
                }
            }
            LogicalOp::RedirectEdges { input, .. } => {
                input.substitute_params(params);
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
            | LogicalOp::CreateEdgeType { .. }
            | LogicalOp::CreateNodeType { .. }
            | LogicalOp::CreateTrigger { .. }
            | LogicalOp::DropTrigger { .. }
            | LogicalOp::ShowTriggers
            | LogicalOp::AlterTrigger { .. }
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
        // Extension ops are EE-handled and never appear in a CE plan; give the
        // planner a trivial, neutral cost estimate.
        LogicalOp::Extension { .. } => (1.0, 1.0),
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

        // HnswScan: O(ef * log N) graph walk + k point reads. Cheapest
        // source op when applicable — costed as k so the planner always
        // prefers it over a full NodeScan of the same label.
        LogicalOp::HnswScan { k, .. } => (*k as f64, *k as f64),

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

        LogicalOp::Union { inputs, .. } => {
            // Cost and row count are the sum across branches; the dedup pass for
            // plain UNION adds a near-linear scan over the combined rows.
            let mut total_cost = 0.0;
            let mut total_rows = 0.0;
            for input in inputs {
                let (c, r) = estimate_op_cost(input, defaults, stats, hints);
                total_cost += c;
                total_rows += r;
            }
            (total_cost + total_rows * 0.01, total_rows)
        }

        LogicalOp::Foreach {
            input, list, body, ..
        } => {
            let (input_cost, input_rows) = estimate_op_cost(input, defaults, stats, hints);
            let (body_cost, _) = estimate_op_cost(body, defaults, stats, hints);
            // Body runs once per input row times the list length. The list size
            // is unknown statically; assume a small constant fan-out.
            let list_factor = match list {
                Expr::Literal(Value::Array(items)) => items.len() as f64,
                _ => 4.0,
            };
            // FOREACH is pass-through: row count downstream equals input rows.
            (
                input_cost + input_rows * list_factor * body_cost,
                input_rows,
            )
        }

        LogicalOp::CallSubquery {
            input,
            body,
            optional,
        } => {
            let (input_cost, input_rows) = estimate_op_cost(input, defaults, stats, hints);
            let (body_cost, body_rows) = estimate_op_cost(body, defaults, stats, hints);
            // Body runs once per outer row; output joins outer rows with
            // subquery rows (at least input rows when OPTIONAL).
            let joined = if *optional {
                (input_rows * body_rows).max(input_rows)
            } else {
                input_rows * body_rows
            };
            (input_cost + input_rows * body_cost, joined)
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

        LogicalOp::MaxSimTopK { input, k, .. } => {
            let (input_cost, input_rows) = estimate_op_cost(input, defaults, stats, hints);
            // Brute-force pass: every input row scored, bounded heap of K.
            // Per-row cost: dim_q * dim_d dot products, but the planner only
            // sees row counts. Use a constant factor representative of the
            // MaxSim kernel (~100x a single dot vs vector_distance).
            let k_f = (*k as f64).max(1.0);
            let per_row_cost = 100.0;
            let rows = k_f.min(input_rows);
            (input_cost + input_rows * per_row_cost, rows)
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
        | LogicalOp::DropVectorIndex { .. }
        | LogicalOp::CreateEdgeType { .. }
        | LogicalOp::CreateNodeType { .. }
        | LogicalOp::CreateTrigger { .. }
        | LogicalOp::DropTrigger { .. }
        | LogicalOp::ShowTriggers
        | LogicalOp::AlterTrigger { .. } => (1.0, 1.0),
        LogicalOp::MergeNodes { input, .. } => {
            // Cost ≈ input cost + per-row (property merge + edge transfer over avg_fan_out).
            // Edge transfer dominates: ~2 × avg_fan_out posting-list ops per merge.
            let (in_cost, in_rows) = estimate_op_cost(input, defaults, stats, hints);
            let per_row = 2.0 * defaults.avg_fan_out;
            (in_cost + in_rows * per_row, in_rows)
        }
        LogicalOp::CloneNode {
            input, with_edges, ..
        } => {
            // Cost ≈ input cost + per-row (one node create, plus ~avg_fan_out
            // edge clones when WITH EDGES).
            let (in_cost, in_rows) = estimate_op_cost(input, defaults, stats, hints);
            let per_row = 1.0
                + if *with_edges {
                    defaults.avg_fan_out
                } else {
                    0.0
                };
            (in_cost + in_rows * per_row, in_rows)
        }
        LogicalOp::RedirectEdges { input, .. } => {
            // Cost ≈ input cost + per-row (~2 × avg_fan_out posting-list ops to
            // re-point both endpoints of each moved edge).
            let (in_cost, in_rows) = estimate_op_cost(input, defaults, stats, hints);
            let per_row = 2.0 * defaults.avg_fan_out;
            (in_cost + in_rows * per_row, in_rows)
        }
        LogicalOp::Empty => (0.0, 1.0),
    }
}

impl LogicalPlan {
    /// Format the plan as EXPLAIN text output with cost estimates.
    pub fn explain(&self) -> String {
        self.explain_with_stats(None)
    }

    /// The stable EXPLAIN `push_down` JSON block for this plan's graph→vector
    /// push-down decision, or `None` when no `VectorFilter` carries one (no
    /// `TRAVERSE`→`VECTOR_FILTER` shape, or the pass did not run). Schema is the
    /// public contract in `arch/core/query-engine.md` (R-PUSH2). The block is
    /// the SW Query Advisor's machine-readable view of the strategy choice.
    #[must_use]
    pub fn explain_push_down_json(&self) -> Option<String> {
        crate::planner::builder::first_push_down_decision(&self.root)
            .map(super::push_down::PushDownDecision::to_explain_json)
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
        | LogicalOp::DocScore { input, .. }
        | LogicalOp::MaxSimTopK { input, .. } => op_contains_vector_filter(input),
        LogicalOp::CartesianProduct { left, right } | LogicalOp::LeftOuterJoin { left, right } => {
            op_contains_vector_filter(left) || op_contains_vector_filter(right)
        }
        _ => false,
    }
}

fn explain_op(op: &LogicalOp, indent: usize, output: &mut String) {
    let prefix = "  ".repeat(indent);

    match op {
        LogicalOp::Extension { name, .. } => {
            output.push_str(&format!("{prefix}Extension({name})\n"));
        }
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
        LogicalOp::HnswScan {
            label,
            property,
            binding,
            k,
            function,
            index_name,
            distance_alias,
            ..
        } => {
            let alias_info = distance_alias
                .as_deref()
                .map(|a| format!(" AS {a}"))
                .unwrap_or_default();
            output.push_str(&format!(
                "{prefix}HnswScan({binding}:{label} ON {index_name}({property}), {function} k={k}{alias_info})\n"
            ));
        }
        LogicalOp::Traverse {
            input,
            source,
            edge_types,
            direction,
            target_variable,
            length,
            temporal_filter,
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
            if let Some(tf) = temporal_filter {
                let upper = tf.upper_ms.map_or("∞".to_string(), |t| t.to_string());
                let lower = tf.lower_ms.map_or("-∞".to_string(), |t| t.to_string());
                let nested = "  ".repeat(indent + 1);
                output.push_str(&format!(
                    "{nested}temporal_filter(r={}, valid_from<={}, valid_to>{})\n",
                    tf.edge_variable, upper, lower
                ));
            }
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
        LogicalOp::Union { inputs, all } => {
            let kind = if *all { "UnionAll" } else { "Union" };
            output.push_str(&format!("{prefix}{kind}\n"));
            for input in inputs {
                explain_op(input, indent + 1, output);
            }
        }
        LogicalOp::Foreach {
            input,
            variable,
            body,
            ..
        } => {
            output.push_str(&format!("{prefix}Foreach({variable})\n"));
            explain_op(input, indent + 1, output);
            explain_op(body, indent + 1, output);
        }
        LogicalOp::CallSubquery {
            input,
            body,
            optional,
        } => {
            let kind = if *optional {
                "OptionalCallSubquery"
            } else {
                "CallSubquery"
            };
            output.push_str(&format!("{prefix}{kind}\n"));
            explain_op(input, indent + 1, output);
            explain_op(body, indent + 1, output);
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
            push_down,
            ..
        } => {
            let op = if *less_than { "<" } else { ">" };
            let decay_info = if decay_field.is_some() {
                " * decay"
            } else {
                ""
            };
            // Surface the push-down strategy on the operator line when the
            // planner attached a decision (graph→vector push-down, R-PUSH1/2).
            let strategy_info = push_down
                .as_ref()
                .map(|d| format!(", strategy={}", d.strategy.as_wire_str()))
                .unwrap_or_default();
            output.push_str(&format!(
                "{prefix}VectorFilter({function}{decay_info} {op} {threshold}{strategy_info})\n"
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
        LogicalOp::MaxSimTopK {
            input,
            k,
            score_alias,
            ..
        } => {
            let alias_info = match score_alias {
                Some(a) => format!(" AS {a}"),
                None => String::new(),
            };
            output.push_str(&format!(
                "{prefix}MaxSimTopK(maxsim_score k={k}{alias_info}, strategy: BruteForce)\n"
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
            fusion,
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
            let fusion_str = match fusion {
                FusionStrategy::Rrf { k } => format!("rrf k={k}"),
                FusionStrategy::ConvexCombination { weights } => {
                    format!("cc weights={weights:?}")
                }
                FusionStrategy::Dbsf { weights } => format!("dbsf weights={weights:?}"),
            };
            output.push_str(&format!(
                "{prefix}RankFuse(methods=[{}], query={{{}}}, {fusion_str}{cap_str})\n",
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
            quantization,
            online_during_build,
            ef_search,
            rerank_candidates,
        } => {
            output.push_str(&format!(
                "{prefix}CreateVectorIndex({name} ON :{label}({property}), m={m}, ef={ef_construction}, metric={metric:?}, dim={dimensions}, quant={quantization:?}, online={online_during_build:?}, ef_search={ef_search:?}, rerank={rerank_candidates:?})\n"
            ));
        }
        LogicalOp::DropVectorIndex { name } => {
            output.push_str(&format!("{prefix}DropVectorIndex({name})\n"));
        }
        LogicalOp::CreateEdgeType {
            name,
            temporal,
            properties,
        } => {
            let temporal_marker = if *temporal { " TEMPORAL" } else { "" };
            output.push_str(&format!(
                "{prefix}CreateEdgeType({name}{temporal_marker}, props={})\n",
                properties.len()
            ));
        }
        LogicalOp::CreateNodeType {
            name,
            temporal,
            properties,
        } => {
            let temporal_marker = if *temporal { " TEMPORAL" } else { "" };
            output.push_str(&format!(
                "{prefix}CreateNodeType({name}{temporal_marker}, props={})\n",
                properties.len()
            ));
        }
        LogicalOp::MergeNodes {
            input,
            source_a,
            source_b,
            target,
            conflict,
            transfer_edges,
            duplicate,
            transfer_edge_properties,
        } => {
            let conflict_tag = match conflict {
                crate::cypher::ast::MergeNodesConflictStrategy::KeepFirst => "KEEP_FIRST",
                crate::cypher::ast::MergeNodesConflictStrategy::KeepLast => "KEEP_LAST",
                crate::cypher::ast::MergeNodesConflictStrategy::Coalesce => "COALESCE",
                crate::cypher::ast::MergeNodesConflictStrategy::SetExpressions(_) => "SET",
            };
            let transfer_tag = match transfer_edges {
                Some(t) => format!(" TRANSFER {}→{}", t.src, t.dst),
                None => String::new(),
            };
            let dup_tag = match duplicate {
                crate::cypher::ast::MergeNodesDuplicateStrategy::KeepBoth => "",
                crate::cypher::ast::MergeNodesDuplicateStrategy::MergeProperties => " DUP_MERGE",
                crate::cypher::ast::MergeNodesDuplicateStrategy::KeepTarget => " DUP_KEEP_TGT",
            };
            let props_tag = if *transfer_edge_properties {
                " +EDGE_PROPS"
            } else {
                ""
            };
            output.push_str(&format!(
                "{prefix}MergeNodes({source_a},{source_b}) INTO {target} {conflict_tag}{transfer_tag}{dup_tag}{props_tag}\n"
            ));
            explain_op(input, indent + 1, output);
        }
        LogicalOp::CloneNode {
            input,
            source,
            target,
            with_edges,
            with_properties,
            set_items,
        } => {
            let edges_tag = if *with_edges { " WITH_EDGES" } else { "" };
            let props_tag = if *with_properties {
                " WITH_PROPS"
            } else {
                " LABELS_ONLY"
            };
            let set_tag = if set_items.is_empty() {
                String::new()
            } else {
                format!(" SET×{}", set_items.len())
            };
            output.push_str(&format!(
                "{prefix}CloneNode({source} AS {target}){edges_tag}{props_tag}{set_tag}\n"
            ));
            explain_op(input, indent + 1, output);
        }
        LogicalOp::RedirectEdges {
            input,
            source,
            target,
            edge_types,
            direction,
        } => {
            let dir_tag = match direction {
                crate::cypher::ast::RedirectDirection::Both => "",
                crate::cypher::ast::RedirectDirection::Outgoing => " OUTGOING",
                crate::cypher::ast::RedirectDirection::Incoming => " INCOMING",
            };
            let type_tag = match edge_types {
                Some(ts) => format!(" TYPES[{}]", ts.join(",")),
                None => String::new(),
            };
            output.push_str(&format!(
                "{prefix}RedirectEdges({source} → {target}){dir_tag}{type_tag}\n"
            ));
            explain_op(input, indent + 1, output);
        }
        LogicalOp::CreateTrigger { clause } => {
            output.push_str(&format!(
                "{prefix}CreateTrigger({}, target={:?}, timing={:?})\n",
                clause.name, clause.target, clause.timing
            ));
        }
        LogicalOp::DropTrigger { name } => {
            output.push_str(&format!("{prefix}DropTrigger({name})\n"));
        }
        LogicalOp::ShowTriggers => {
            output.push_str(&format!("{prefix}ShowTriggers\n"));
        }
        LogicalOp::AlterTrigger { clause } => {
            output.push_str(&format!(
                "{prefix}AlterTrigger({}, action={:?})\n",
                clause.name, clause.action
            ));
        }
        LogicalOp::Empty => {
            output.push_str(&format!("{prefix}Empty\n"));
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
