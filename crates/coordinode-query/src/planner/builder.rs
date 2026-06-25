//! Logical plan builder: converts Cypher AST → logical plan tree.
//!
//! Walks the flat clause list and builds a tree of relational algebra operators.
//! Each clause modifies the "current working set" by wrapping the previous
//! operator in a new one.

use super::logical::*;
use crate::cypher::ast::*;
use coordinode_core::graph::types::{Value, VectorConsistencyMode};

/// Error during logical plan construction.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum PlanError {
    #[error("query must contain at least one clause")]
    EmptyQuery,

    #[error("RETURN clause required for read queries")]
    MissingReturn,

    #[error("pattern is empty")]
    EmptyPattern,

    #[error("unsupported pattern structure")]
    UnsupportedPattern,

    #[error("shortestPath() {0}")]
    ShortestPathShape(String),

    #[error(
        "rrf_score() takes exactly 2 arguments: rrf_score([method_exprs...], {{vector: ..., text: ...}}); \
         k=60 is the IR standard (Cormack et al. 2009) and is not tunable"
    )]
    RrfScoreArity,

    #[error(
        "rrf_score(): first argument must be a non-empty list of method expressions \
         (e.g. [n.embedding, c.body]); got {got}"
    )]
    RrfScoreMethodsShape { got: String },

    #[error(
        "rrf_score(): second argument must be a map literal with `vector` and/or `text` keys \
         (e.g. {{vector: $qv, text: $qt}}) or a parameter resolving to such a map; got {got}"
    )]
    RrfScoreQueryShape { got: String },

    #[error(
        "rrf_score(): multiple rrf_score() calls with differing method / query arguments are \
         not supported in a single query (detected at {location}). Use WITH to materialise \
         the first call before invoking the second"
    )]
    RrfScoreMultipleCalls { location: String },

    #[error("rrf_score() cannot appear in {location} — it requires materialised rank assignment")]
    RrfScoreIllegalPosition { location: String },

    #[error(
        "doc_score() takes 2-5 arguments: doc_score(doc, query [, α, β, γ]) or \
         doc_score(doc, query, {{alpha, beta, gamma}}); got {got} argument(s)"
    )]
    DocScoreArity { got: usize },

    #[error(
        "doc_score(): first argument must be a variable bound to a Document node (e.g. `d`); got {got}"
    )]
    DocScoreDocShape { got: String },

    #[error(
        "doc_score(): weights argument must be a map with keys `alpha`, `beta`, `gamma` \
         (e.g. {{alpha: 0.4, beta: 0.4, gamma: 0.2}}) or three positional numeric arguments; \
         got {got}"
    )]
    DocScoreWeightsShape { got: String },

    #[error(
        "doc_score(): multiple doc_score() calls with differing doc / query / weights are \
         not supported in a single query (detected at {location}). Use WITH to materialise \
         the first call before invoking the second"
    )]
    DocScoreMultipleCalls { location: String },

    #[error(
        "doc_score() cannot appear in {location} — it is a correlated aggregate over HAS_CHUNK children"
    )]
    DocScoreIllegalPosition { location: String },

    #[error("failed to encode extension-op payload: {0}")]
    ExtensionPayloadEncode(String),
}

/// Build the optimized logical root for a single query branch (one clause
/// sequence). Shared by the leading query and each `UNION` branch. Returns the
/// optimized root plus any `AS OF TIMESTAMP` snapshot expression found.
fn build_branch_root(clauses: &[Clause]) -> Result<(LogicalOp, Option<Expr>), PlanError> {
    if clauses.is_empty() {
        return Err(PlanError::EmptyQuery);
    }

    let mut current: Option<LogicalOp> = None;
    let mut snapshot_ts: Option<Expr> = None;

    for clause in clauses {
        if let Clause::AsOfTimestamp(expr) = clause {
            snapshot_ts = Some(expr.clone());
            continue;
        }
        current = Some(apply_clause(current, clause)?);
    }

    let root = current.ok_or(PlanError::EmptyQuery)?;

    // Optimization pass: detect edge vector patterns and select strategy
    let root = optimize_edge_vector_search(root);

    // Optimization pass: detect `Sort(vector_distance) + Limit(K)` and rewrite
    // to VectorTopK for HNSW-accelerated top-K search.
    let root = optimize_vector_top_k(root);

    // Planner pass: detect `rrf_score([methods…], query)` in Project / Sort items
    // and lift it into a `LogicalOp::RankFuse` below the innermost Project.
    let root = rewrite_rrf_score(root)?;

    // Planner pass: detect `doc_score(doc, query [,α,β,γ])` in Project / Sort
    // items and lift it into a `LogicalOp::DocScore` below the innermost Project.
    let root = rewrite_doc_score(root)?;

    // Planner pass: lift `temporal_active_at(r, $T)` predicates from Filter into
    // the parent Traverse's `temporal_filter`. Bounds the per-version edgeprop
    // scan in the executor instead of materializing every stored version.
    let root = lift_temporal_filter(root);

    Ok((root, snapshot_ts))
}

/// Build a logical plan from a validated Cypher AST.
pub fn build_logical_plan(query: &Query) -> Result<LogicalPlan, PlanError> {
    if query.clauses.is_empty() {
        return Err(PlanError::EmptyQuery);
    }

    let (mut root, mut snapshot_ts) = build_branch_root(&query.clauses)?;

    // UNION / UNION ALL: build each subsequent branch independently and wrap
    // all branches in a Union node. A plain `UNION` anywhere de-duplicates the
    // combined result (Neo4j forbids mixing UNION and UNION ALL in one query,
    // so in practice every branch shares the same flag).
    if !query.unions.is_empty() {
        let mut inputs = vec![root];
        let mut all = true;
        for branch in &query.unions {
            let (branch_root, branch_snapshot) = build_branch_root(&branch.clauses)?;
            if !branch.all {
                all = false;
            }
            if branch_snapshot.is_some() && snapshot_ts.is_none() {
                snapshot_ts = branch_snapshot;
            }
            inputs.push(branch_root);
        }
        root = LogicalOp::Union { inputs, all };
    }

    // Extract per-query hints. Both hints are optional; an explicit hint
    // always overrides the planner default or auto-promotion.
    let mut vector_consistency_hint: Option<VectorConsistencyMode> = None;
    let mut read_consistency_hint: Option<
        coordinode_core::txn::read_consistency::ReadConsistencyMode,
    > = None;
    for hint in &query.hints {
        match hint {
            crate::cypher::ast::QueryHint::VectorConsistency(mode) => {
                vector_consistency_hint = Some(*mode);
            }
            crate::cypher::ast::QueryHint::ReadConsistency(mode) => {
                read_consistency_hint = Some(*mode);
            }
        }
    }

    // R-SNAP1: auto-promote `read_consistency` to `snapshot` when the query
    // touches >1 modality. An explicit hint always wins — even a hint that
    // sets `current` on a cross-modality query is honoured (user knows
    // something the planner doesn't).
    let read_consistency = read_consistency_hint.unwrap_or_else(|| {
        if modality_count(&root) > 1 {
            coordinode_core::txn::read_consistency::ReadConsistencyMode::Snapshot
        } else {
            coordinode_core::txn::read_consistency::ReadConsistencyMode::Current
        }
    });

    // R-SNAP1 vector-consistency narrower override: if the user explicitly
    // set `vector_consistency`, it wins for the vector modality. Otherwise
    // `vector_consistency` follows `read_consistency` (Current → Current,
    // Snapshot → Snapshot, Exact → Exact) — the three variants map 1:1.
    let vector_consistency = vector_consistency_hint.unwrap_or_else(|| {
        use coordinode_core::txn::read_consistency::ReadConsistencyMode as RC;
        match read_consistency {
            RC::Current => VectorConsistencyMode::Current,
            RC::Snapshot => VectorConsistencyMode::Snapshot,
            RC::Exact => VectorConsistencyMode::Exact,
        }
    });

    Ok(LogicalPlan {
        root,
        snapshot_ts,
        vector_consistency,
        read_consistency,
    })
}

/// R-SNAP1: count the number of distinct modalities a logical plan touches.
///
/// The auto-promotion rule (arch/core/transactions.md § Read Consistency)
/// says:
/// > IF query touches >1 modality (graph + vector, vector + text, etc.):
/// >     read_consistency = 'snapshot'
/// > ELSE IF query is single-modality:
/// >     read_consistency = 'current'
///
/// Modalities recognised here:
/// - **graph** — every read touches the graph (NodeScan / Traverse /
///   IndexScan). A pure graph-only query still counts as one modality.
/// - **vector** — `VectorFilter`, `VectorTopK`, `EdgeVectorSearch`, or a
///   `RankFuse` that carries a vector method.
/// - **text** — `TextFilter`, `EncryptedFilter`, or a `RankFuse` with a
///   text method.
/// - **doc** — `DocScore` (correlated HAS_CHUNK aggregate). Still counts
///   as its own modality for the cross-modality promotion rule, since it
///   mixes graph traversal + vector scoring with a distinct cache column.
fn modality_count(root: &LogicalOp) -> usize {
    // Per `arch/core/transactions.md § Read Consistency`: the auto-promotion
    // rule fires when a query touches MORE THAN ONE modality from the set
    // {graph, vector, text, doc}. Arch doc explicitly names "pure vector KNN"
    // and "pure graph traversal" as single-modality examples.
    //
    // The subtlety: every query begins with a NodeScan / IndexScan — that's
    // the row source, not a distinct "graph modality". A VectorTopK riding on
    // a NodeScan is a single-modality vector query, NOT graph + vector.
    //
    // Graph counts as a distinct modality only when the query actually does
    // graph work — Traverse (multi-hop pattern matching) or ShortestPath. A
    // bare NodeScan/IndexScan is treated as a carrier for whatever modality
    // sits on top; if nothing else is present, the query is graph-only with
    // count = 1.
    //
    // RankFuse/DocScore operate on vector+text / chunk+vector respectively;
    // they do NOT mark graph because the underlying scan is again just the
    // carrier.
    #[derive(Default)]
    struct Mods {
        graph_carrier: bool,  // NodeScan / IndexScan present — not a modality by itself
        graph_explicit: bool, // Traverse / ShortestPath — genuine graph work
        vector: bool,
        text: bool,
        doc: bool,
    }
    fn walk(op: &LogicalOp, m: &mut Mods) {
        match op {
            LogicalOp::NodeScan { .. } | LogicalOp::IndexScan { .. } => {
                m.graph_carrier = true;
            }
            LogicalOp::Traverse { .. } | LogicalOp::ShortestPath { .. } => {
                m.graph_explicit = true;
            }
            LogicalOp::VectorFilter { .. }
            | LogicalOp::VectorTopK { .. }
            | LogicalOp::EdgeVectorSearch { .. } => {
                m.vector = true;
            }
            LogicalOp::TextFilter { .. } | LogicalOp::EncryptedFilter { .. } => {
                m.text = true;
            }
            LogicalOp::RankFuse {
                query_vector,
                query_text,
                ..
            } => {
                if query_vector.is_some() {
                    m.vector = true;
                }
                if query_text.is_some() {
                    m.text = true;
                }
            }
            LogicalOp::DocScore { .. } => {
                m.doc = true;
            }
            _ => {}
        }
        for child in op_children(op) {
            walk(child, m);
        }
    }
    let mut m = Mods::default();
    walk(root, &mut m);

    let explicit = [m.graph_explicit, m.vector, m.text, m.doc]
        .iter()
        .filter(|b| **b)
        .count();

    if explicit == 0 && m.graph_carrier {
        // Pure NodeScan/IndexScan query — single graph modality.
        1
    } else {
        explicit
    }
}

/// Direct children of a logical operator for modality-walk recursion.
/// The first `VectorFilter` push-down decision found in depth-first plan order,
/// if any. Used to surface the EXPLAIN `push_down` block for a plan.
pub(crate) fn first_push_down_decision(
    op: &LogicalOp,
) -> Option<&crate::planner::push_down::PushDownDecision> {
    if let LogicalOp::VectorFilter {
        push_down: Some(decision),
        ..
    } = op
    {
        return Some(decision);
    }
    for child in op_children(op) {
        if let Some(d) = first_push_down_decision(child) {
            return Some(d);
        }
    }
    None
}

fn op_children(op: &LogicalOp) -> Vec<&LogicalOp> {
    match op {
        LogicalOp::Filter { input, .. }
        | LogicalOp::Project { input, .. }
        | LogicalOp::Aggregate { input, .. }
        | LogicalOp::Sort { input, .. }
        | LogicalOp::Limit { input, .. }
        | LogicalOp::Skip { input, .. }
        | LogicalOp::Traverse { input, .. }
        | LogicalOp::Unwind { input, .. }
        | LogicalOp::VectorFilter { input, .. }
        | LogicalOp::VectorTopK { input, .. }
        | LogicalOp::EdgeVectorSearch { input, .. }
        | LogicalOp::TextFilter { input, .. }
        | LogicalOp::EncryptedFilter { input, .. }
        | LogicalOp::ShortestPath { input, .. }
        | LogicalOp::RankFuse { input, .. }
        | LogicalOp::DocScore { input, .. }
        | LogicalOp::Update { input, .. }
        | LogicalOp::RemoveOp { input, .. }
        | LogicalOp::Delete { input, .. }
        | LogicalOp::DetachDocument { input, .. }
        | LogicalOp::AttachDocument { input, .. }
        | LogicalOp::MergeNodes { input, .. }
        | LogicalOp::CloneNode { input, .. }
        | LogicalOp::RedirectEdges { input, .. }
        | LogicalOp::CreateEdge { input, .. } => vec![input],
        LogicalOp::CartesianProduct { left, right } | LogicalOp::LeftOuterJoin { left, right } => {
            vec![left, right]
        }
        LogicalOp::CreateNode { input, .. } => input.as_deref().into_iter().collect(),
        LogicalOp::Merge { pattern, .. } | LogicalOp::Upsert { pattern, .. } => vec![pattern],
        LogicalOp::Union { inputs, .. } => inputs.iter().collect(),
        LogicalOp::Foreach { input, body, .. } => vec![input, body],
        LogicalOp::CallSubquery { input, body, .. } => vec![input, body],
        _ => Vec::new(),
    }
}

/// True when a MATCH clause is a linear traversal whose first node reuses a
/// bare variable already bound by a prior clause: `MATCH (v) ... MATCH
/// (v)-[:R]->(x)`. Such a MATCH continues the traversal out of the bound `v`
/// rather than opening a fresh scan, so the prior clause's filter on `v` is
/// preserved. Restricted to a single relationship-bearing pattern with no
/// re-stated labels / properties on the anchor and no named/shortest path, so
/// every other shape keeps the prior scan-and-join behaviour.
fn pattern_is_bound_continuation(pattern: &Pattern, prior_vars: &[String]) -> bool {
    if pattern.shortest_path || pattern.path_variable.is_some() || pattern.elements.len() < 3 {
        return false;
    }
    match pattern.elements.first() {
        Some(PatternElement::Node(np)) => {
            np.labels.is_empty()
                && np.properties.is_empty()
                && np.variable.as_ref().is_some_and(|v| prior_vars.contains(v))
        }
        _ => false,
    }
}

/// A single-pattern MATCH continues a prior binding when its lone pattern does.
fn match_is_bound_continuation(mc: &MatchClause, prior_vars: &[String]) -> bool {
    matches!(mc.patterns.as_slice(), [p] if pattern_is_bound_continuation(p, prior_vars))
}

/// Apply a single clause to the current plan, producing a new operator.
fn apply_clause(current: Option<LogicalOp>, clause: &Clause) -> Result<LogicalOp, PlanError> {
    match clause {
        Clause::Match(mc) => {
            match current {
                // Continuation: this MATCH traverses out of a node already bound
                // by a prior clause. Build the traversal on top of `existing`
                // (preserving its filters) instead of re-scanning, then apply
                // this MATCH's own WHERE on top.
                Some(existing)
                    if match_is_bound_continuation(mc, &collect_op_variables(&existing)) =>
                {
                    let chain = build_pattern_chain(&mc.patterns[0].elements, Some(existing))?;
                    Ok(match &mc.where_clause {
                        Some(pred) => apply_compound_where(pred, chain),
                        None => chain,
                    })
                }
                Some(existing) if mc.where_clause.is_some() => {
                    // G024: When building on top of a prior clause, the WHERE
                    // may reference variables from both the prior plan (existing)
                    // and this MATCH's patterns. Predicates referencing prior
                    // variables must be lifted ABOVE the CartesianProduct.
                    let prior_vars = collect_op_variables(&existing);
                    let branch_vars = collect_pattern_variables(&mc.patterns);
                    // Safe: guard `mc.where_clause.is_some()` checked above.
                    let Some(predicate) = &mc.where_clause else {
                        unreachable!()
                    };
                    let conjuncts = flatten_and(predicate);

                    let mut local = Vec::new();
                    let mut lifted = Vec::new();

                    for conj in conjuncts {
                        let refs = collect_expr_variables(&conj);
                        // A predicate is "cross-scope" if it references any
                        // variable introduced by the prior plan that is NOT
                        // also introduced by this branch's patterns.
                        let needs_prior = refs
                            .iter()
                            .any(|v| prior_vars.contains(v) && !branch_vars.contains(v));
                        if needs_prior {
                            lifted.push(conj);
                        } else {
                            local.push(conj);
                        }
                    }

                    // Build match op with only local predicates.
                    let local_mc = MatchClause {
                        patterns: mc.patterns.clone(),
                        where_clause: if local.is_empty() {
                            None
                        } else {
                            Some(conjoin_predicates(local))
                        },
                    };
                    let scan = build_match_op(&local_mc)?;

                    let mut result = LogicalOp::CartesianProduct {
                        left: Box::new(existing),
                        right: Box::new(scan),
                    };

                    // Apply lifted predicates above CartesianProduct where
                    // all variables are in scope.
                    if !lifted.is_empty() {
                        result = apply_compound_where(&conjoin_predicates(lifted), result);
                    }

                    Ok(result)
                }
                Some(existing) => {
                    let scan = build_match_op(mc)?;
                    Ok(LogicalOp::CartesianProduct {
                        left: Box::new(existing),
                        right: Box::new(scan),
                    })
                }
                None => Ok(build_match_op(mc)?),
            }
        }
        Clause::OptionalMatch(mc) => {
            let scan = build_match_op(mc)?;
            match current {
                Some(existing) => Ok(LogicalOp::LeftOuterJoin {
                    left: Box::new(existing),
                    right: Box::new(scan),
                }),
                None => Ok(scan),
            }
        }
        Clause::Where(expr) => {
            let input = current.unwrap_or(LogicalOp::Empty);
            Ok(apply_compound_where(expr, input))
        }
        Clause::Return(rc) => {
            let input = current.unwrap_or(LogicalOp::Empty);
            build_return_op(input, rc)
        }
        Clause::With(wc) => {
            let input = current.unwrap_or(LogicalOp::Empty);
            build_with_op(input, wc)
        }
        Clause::Unwind(uc) => {
            let input = current.unwrap_or(LogicalOp::Empty);
            Ok(LogicalOp::Unwind {
                input: Box::new(input),
                expr: uc.expr.clone(),
                variable: uc.variable.clone(),
            })
        }
        Clause::OrderBy(items) => {
            let input = current.unwrap_or(LogicalOp::Empty);
            Ok(LogicalOp::Sort {
                input: Box::new(input),
                items: items.clone(),
            })
        }
        Clause::Skip(expr) => {
            let input = current.unwrap_or(LogicalOp::Empty);
            Ok(LogicalOp::Skip {
                input: Box::new(input),
                count: expr.clone(),
            })
        }
        Clause::Limit(expr) => {
            let input = current.unwrap_or(LogicalOp::Empty);
            Ok(LogicalOp::Limit {
                input: Box::new(input),
                count: expr.clone(),
            })
        }
        Clause::AsOfTimestamp(_) => {
            // Time travel is handled at the storage layer, not in the logical plan.
            // Pass through the current plan unchanged.
            Ok(current.unwrap_or(LogicalOp::Empty))
        }
        Clause::Create(cc) => build_create_op(current, cc),
        Clause::Merge(mc) => {
            let scan = build_pattern_scan(&mc.pattern)?;
            let merge_op = LogicalOp::Merge {
                pattern: Box::new(scan),
                on_match: mc.on_match.clone(),
                on_create: mc.on_create.clone(),
                multi: false,
            };
            match current {
                Some(existing) => Ok(LogicalOp::CartesianProduct {
                    left: Box::new(existing),
                    right: Box::new(merge_op),
                }),
                None => Ok(merge_op),
            }
        }
        Clause::MergeMany(mc) => {
            let scan = build_pattern_scan(&mc.pattern)?;
            let merge_op = LogicalOp::Merge {
                pattern: Box::new(scan),
                on_match: mc.on_match.clone(),
                on_create: mc.on_create.clone(),
                multi: true,
            };
            match current {
                Some(existing) => Ok(LogicalOp::CartesianProduct {
                    left: Box::new(existing),
                    right: Box::new(merge_op),
                }),
                None => Ok(merge_op),
            }
        }
        Clause::Upsert(uc) => {
            let scan = build_pattern_scan(&uc.pattern)?;
            let upsert_op = LogicalOp::Upsert {
                pattern: Box::new(scan),
                on_match: uc.on_match.clone(),
                on_create_patterns: uc.on_create.clone(),
            };
            match current {
                Some(existing) => Ok(LogicalOp::CartesianProduct {
                    left: Box::new(existing),
                    right: Box::new(upsert_op),
                }),
                None => Ok(upsert_op),
            }
        }
        Clause::Delete(dc) => {
            let input = current.unwrap_or(LogicalOp::Empty);
            let variables: Vec<String> = dc
                .exprs
                .iter()
                .filter_map(|e| {
                    if let Expr::Variable(ref v) = e {
                        Some(v.clone())
                    } else {
                        None
                    }
                })
                .collect();
            Ok(LogicalOp::Delete {
                input: Box::new(input),
                variables,
                detach: dc.detach,
            })
        }
        Clause::MergeNodes(mn) => {
            let input = current.unwrap_or(LogicalOp::Empty);
            Ok(LogicalOp::MergeNodes {
                input: Box::new(input),
                source_a: mn.source_a.clone(),
                source_b: mn.source_b.clone(),
                target: mn.target.clone(),
                conflict: mn.conflict.clone(),
                transfer_edges: mn.transfer_edges.clone(),
                duplicate: mn.duplicate.clone(),
                transfer_edge_properties: mn.transfer_edge_properties,
            })
        }
        Clause::CloneNode(cn) => {
            let input = current.unwrap_or(LogicalOp::Empty);
            Ok(LogicalOp::CloneNode {
                input: Box::new(input),
                source: cn.source.clone(),
                target: cn.target.clone(),
                with_edges: cn.with_edges,
                with_properties: cn.with_properties,
                set_items: cn.set_items.clone(),
            })
        }
        Clause::RedirectEdges(re) => {
            let input = current.unwrap_or(LogicalOp::Empty);
            Ok(LogicalOp::RedirectEdges {
                input: Box::new(input),
                source: re.source.clone(),
                target: re.target.clone(),
                edge_types: re.edge_types.clone(),
                direction: re.direction,
            })
        }
        Clause::Set(items, violation_mode) => {
            let input = current.unwrap_or(LogicalOp::Empty);
            Ok(LogicalOp::Update {
                input: Box::new(input),
                items: items.clone(),
                violation_mode: violation_mode.clone(),
            })
        }
        Clause::Remove(items) => {
            let input = current.unwrap_or(LogicalOp::Empty);
            Ok(LogicalOp::RemoveOp {
                input: Box::new(input),
                items: items.clone(),
            })
        }
        Clause::Foreach(fc) => {
            let input = current.unwrap_or(LogicalOp::Empty);
            // Build the body as an independent update chain whose leaf is Empty;
            // the executor injects the per-iteration scope at that leaf.
            let mut body: Option<LogicalOp> = None;
            for body_clause in &fc.body {
                body = Some(apply_clause(body, body_clause)?);
            }
            let body = body.ok_or(PlanError::EmptyQuery)?;
            Ok(LogicalOp::Foreach {
                input: Box::new(input),
                variable: fc.variable.clone(),
                list: fc.list.clone(),
                body: Box::new(body),
            })
        }
        Clause::CallSubquery(cs) => {
            let input = current.unwrap_or(LogicalOp::Empty);
            // Build the subquery body as an independent plan. A leading
            // importing WITH lands on an Empty leaf (the executor injects the
            // outer row there for correlation); an uncorrelated body starts
            // with its own scan.
            let mut body: Option<LogicalOp> = None;
            for body_clause in &cs.body {
                body = Some(apply_clause(body, body_clause)?);
            }
            let body = body.ok_or(PlanError::EmptyQuery)?;
            Ok(LogicalOp::CallSubquery {
                input: Box::new(input),
                body: Box::new(body),
                optional: cs.optional,
            })
        }
        Clause::Call(cc) => Ok(LogicalOp::ProcedureCall {
            procedure: cc.procedure.clone(),
            args: cc.args.clone(),
            yield_items: cc.yield_items.iter().map(|yi| yi.name.clone()).collect(),
        }),
        Clause::AlterLabel(ac) => Ok(LogicalOp::AlterLabel {
            label: ac.label.clone(),
            mode: ac.mode.clone(),
        }),
        Clause::CreateTextIndex(c) => Ok(LogicalOp::CreateTextIndex {
            name: c.name.clone(),
            label: c.label.clone(),
            fields: c.fields.clone(),
            default_language: c.default_language.clone(),
            language_override: c.language_override.clone(),
        }),
        Clause::DropTextIndex(c) => Ok(LogicalOp::DropTextIndex {
            name: c.name.clone(),
        }),
        Clause::CreateEncryptedIndex(c) => Ok(LogicalOp::CreateEncryptedIndex {
            name: c.name.clone(),
            label: c.label.clone(),
            property: c.property.clone(),
        }),
        Clause::DropEncryptedIndex(c) => Ok(LogicalOp::DropEncryptedIndex {
            name: c.name.clone(),
        }),
        Clause::CreateIndex(c) => {
            let filter = c.filter_expr.as_ref().and_then(expr_to_partial_filter);
            Ok(LogicalOp::CreateIndex {
                name: c.name.clone(),
                label: c.label.clone(),
                property: c.property.clone(),
                unique: c.unique,
                sparse: c.sparse,
                filter,
            })
        }
        Clause::DropIndex(c) => Ok(LogicalOp::DropIndex {
            name: c.name.clone(),
        }),
        Clause::CreateVectorIndex(c) => {
            // A trailing extension clause (e.g. an engine-extension's `SHARDED BY
            // ...`) after the known CREATE VECTOR INDEX syntax routes to an
            // extension handler: serialize the parsed clause (params + raw tail)
            // as the opaque payload and emit an Extension op. The base engine
            // registers no handler, so this errors clearly at execution; an
            // extension layer parses the tail in its own handler.
            if c.extension_tail.is_some() {
                let payload = rmp_serde::to_vec(c)
                    .map_err(|e| PlanError::ExtensionPayloadEncode(e.to_string()))?;
                return Ok(LogicalOp::Extension {
                    name: "vector_index.create_ext".to_string(),
                    payload,
                });
            }
            let metric = parse_vector_metric(c.metric.as_deref());
            let quantization = parse_quantization_codec(c.quantization.as_deref());
            let online_during_build = match c
                .online_during_build
                .as_deref()
                .map(str::to_ascii_lowercase)
                .as_deref()
            {
                Some("partial-recall") | Some("partial_recall") => {
                    crate::index::OnlineDuringBuild::PartialRecall
                }
                Some("offline") => crate::index::OnlineDuringBuild::Offline,
                // Default + every other value (including the explicit "block")
                // resolves to Block. Unknown values are silently accepted to match
                // the existing OPTIONS parser tolerance for typos.
                _ => crate::index::OnlineDuringBuild::Block,
            };
            Ok(LogicalOp::CreateVectorIndex {
                name: c.name.clone(),
                label: c.label.clone(),
                property: c.property.clone(),
                m: c.m.unwrap_or(16),
                ef_construction: c.ef_construction.unwrap_or(200),
                metric,
                dimensions: c.dimensions.unwrap_or(0),
                quantization,
                online_during_build,
                ef_search: c.ef_search,
                rerank_candidates: c.rerank_candidates,
            })
        }
        Clause::DropVectorIndex(c) => Ok(LogicalOp::DropVectorIndex {
            name: c.name.clone(),
        }),
        Clause::CreateEdgeType(c) => Ok(LogicalOp::CreateEdgeType {
            name: c.name.clone(),
            temporal: c.temporal,
            properties: c.properties.clone(),
        }),
        Clause::CreateNodeType(c) => Ok(LogicalOp::CreateNodeType {
            name: c.name.clone(),
            temporal: c.temporal,
            properties: c.properties.clone(),
        }),
        Clause::CreateTrigger(c) => Ok(LogicalOp::CreateTrigger { clause: c.clone() }),
        Clause::DropTrigger(c) => Ok(LogicalOp::DropTrigger {
            name: c.name.clone(),
        }),
        Clause::ShowTriggers => Ok(LogicalOp::ShowTriggers),
        Clause::AlterTrigger(c) => Ok(LogicalOp::AlterTrigger { clause: c.clone() }),
        Clause::AttachDocument(ad) => {
            // Synthesize a MATCH for the ATTACH pattern `(a)-[:T]->(u)` so that
            // the executor receives pre-bound `source_variable` / `target_variable`
            // columns in each row (reuses existing NodeScan + Traverse machinery).
            let source_node = NodePattern {
                variable: Some(ad.source_variable.clone()),
                labels: ad.source_labels.clone(),
                properties: Vec::new(),
            };
            let target_node = NodePattern {
                variable: Some(ad.target_variable.clone()),
                labels: ad.target_labels.clone(),
                properties: Vec::new(),
            };
            let rel = RelationshipPattern {
                variable: ad.edge_variable.clone(),
                rel_types: vec![ad.edge_type.clone()],
                direction: match ad.edge_direction {
                    crate::cypher::ast::EdgeFromSource::Outgoing => Direction::Outgoing,
                    crate::cypher::ast::EdgeFromSource::Incoming => Direction::Incoming,
                },
                length: None,
                properties: Vec::new(),
            };
            let pattern = Pattern {
                elements: vec![
                    PatternElement::Node(source_node),
                    PatternElement::Relationship(rel),
                    PatternElement::Node(target_node),
                ],
                path_variable: None,
                shortest_path: false,
            };
            let match_clause = MatchClause {
                patterns: vec![pattern],
                where_clause: None,
            };
            let scan = build_match_op(&match_clause)?;
            // Compose with the prior plan if present (e.g. user wrote
            // `MATCH ... ATTACH ...`), otherwise use the synthesized scan alone.
            let input = match current {
                Some(existing) => LogicalOp::CartesianProduct {
                    left: Box::new(existing),
                    right: Box::new(scan),
                },
                None => scan,
            };
            Ok(LogicalOp::AttachDocument {
                input: Box::new(input),
                source_variable: ad.source_variable.clone(),
                target_variable: ad.target_variable.clone(),
                edge_type: ad.edge_type.clone(),
                edge_direction: ad.edge_direction,
                target_property_path: ad.target_property_path.clone(),
                transfer: ad.transfer.clone(),
                on_conflict_replace: ad.on_conflict_replace,
                on_remaining_fail: ad.on_remaining_fail,
            })
        }
        Clause::DetachDocument(dd) => {
            let input = current.unwrap_or(LogicalOp::Empty);
            Ok(LogicalOp::DetachDocument {
                input: Box::new(input),
                source_variable: dd.source_variable.clone(),
                property_path: dd.property_path.clone(),
                target_variable: dd.target_variable.clone(),
                target_labels: dd.target_labels.clone(),
                edge_type: dd
                    .edge_type
                    .clone()
                    .unwrap_or_else(|| default_edge_type(&dd.property_path)),
                edge_direction: dd.edge_direction,
                edge_variable: dd.edge_variable.clone(),
                transfer: dd.transfer.clone(),
            })
        }
    }
}

/// Derive a default edge type from a property path: `HAS_<UPPER_SNAKE(last)>`.
fn default_edge_type(path: &[String]) -> String {
    let last = path.last().map(String::as_str).unwrap_or("DOCUMENT");
    // Convert camelCase / snake_case to UPPER_SNAKE.
    let mut out = String::from("HAS_");
    let mut prev_lower = false;
    for ch in last.chars() {
        if ch == '_' {
            out.push('_');
            prev_lower = false;
            continue;
        }
        if ch.is_uppercase() {
            if prev_lower {
                out.push('_');
            }
            out.push(ch);
            prev_lower = false;
        } else {
            for upper in ch.to_uppercase() {
                out.push(upper);
            }
            prev_lower = true;
        }
    }
    out
}

/// Parse a metric string into a `VectorMetric`.
///
/// Accepts: "cosine", "euclidean" (alias for L2), "l2", "dot", "dotproduct", "l1".
/// Defaults to `Cosine` for unknown or None.
fn parse_vector_metric(s: Option<&str>) -> coordinode_core::graph::types::VectorMetric {
    use coordinode_core::graph::types::VectorMetric;
    match s.map(|m| m.to_lowercase()).as_deref() {
        Some("cosine") | None => VectorMetric::Cosine,
        Some("euclidean") | Some("l2") => VectorMetric::L2,
        Some("dot") | Some("dotproduct") | Some("dot_product") => VectorMetric::DotProduct,
        Some("l1") | Some("manhattan") => VectorMetric::L1,
        _ => VectorMetric::Cosine,
    }
}

/// Resolve the Cypher OPTIONS `quantization` string to a
/// [`QuantizationCodec`]. Case-insensitive.
///
/// Accepts:
/// - `none` / absent → `None` (f32 in RAM, default)
/// - `sq8` → `Sq8`
/// - `rabitq` / `rabitq-1bit` → `RaBitQ { bits: 1 }`
/// - `rabitq-2bit` / `rabitq-3bit` / `rabitq-4bit` → Extended-RaBitQ
///   at the indicated bit width (R862)
///
/// Unrecognized values fall back to `None` rather than erroring so a
/// typo in a DDL string doesn't fail an entire migration; the planner
/// emits no warning here — the index just behaves as if no codec was
/// requested. (Strict validation lives in the parser layer; this fn
/// is the last-resort fallback.)
fn parse_quantization_codec(s: Option<&str>) -> coordinode_vector::hnsw::QuantizationCodec {
    use coordinode_vector::hnsw::QuantizationCodec;
    match s.map(|q| q.to_lowercase()).as_deref() {
        Some("none") | None => QuantizationCodec::None,
        Some("sq8") => QuantizationCodec::Sq8,
        Some("rabitq") | Some("rabitq-1bit") => QuantizationCodec::RaBitQ { bits: 1 },
        Some("rabitq-2bit") => QuantizationCodec::RaBitQ { bits: 2 },
        Some("rabitq-3bit") => QuantizationCodec::RaBitQ { bits: 3 },
        Some("rabitq-4bit") => QuantizationCodec::RaBitQ { bits: 4 },
        _ => QuantizationCodec::None,
    }
}

/// Reconstruct a vector index definition and its distance metric from a parsed
/// CREATE VECTOR INDEX clause, applying the same option defaults and string
/// parsing the planner uses for the non-extension path. An engine extension
/// (e.g. a sharded-index handler) calls this to rebuild the definition from the
/// clause carried verbatim in an `Extension` op payload, keeping the parsing of
/// metric / quantization / online-build options as a single source of truth.
pub fn vector_index_definition_from_clause(
    clause: &crate::cypher::ast::CreateVectorIndexClause,
) -> (
    crate::index::IndexDefinition,
    coordinode_core::graph::types::VectorMetric,
) {
    let metric = parse_vector_metric(clause.metric.as_deref());
    let quantization = parse_quantization_codec(clause.quantization.as_deref());
    let online_during_build = match clause
        .online_during_build
        .as_deref()
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("partial-recall") | Some("partial_recall") => {
            crate::index::OnlineDuringBuild::PartialRecall
        }
        Some("offline") => crate::index::OnlineDuringBuild::Offline,
        _ => crate::index::OnlineDuringBuild::Block,
    };
    let config = crate::index::VectorIndexConfig {
        dimensions: clause.dimensions.unwrap_or(0),
        metric,
        m: clause.m.unwrap_or(16),
        ef_construction: clause.ef_construction.unwrap_or(200),
        quantization,
        offload_vectors: false,
        ef_search: clause.ef_search,
        rerank_candidates: clause.rerank_candidates,
    };
    let mut def =
        crate::index::IndexDefinition::hnsw(&clause.name, &clause.label, &clause.property, config);
    def.online_during_build = online_during_build;
    (def, metric)
}

/// Attempt to convert a WHERE expression to a `PartialFilter` predicate.
///
/// Supports:
/// - `n.prop = 'value'`  → `PartialFilter::PropertyEquals`
/// - `n.prop = 42`       → `PartialFilter::PropertyEqualsInt`
/// - `n.prop = true`     → `PartialFilter::PropertyEqualsBool`
/// - `EXISTS(n.prop)`    → `PartialFilter::PropertyExists`
/// - `n.prop IS NOT NULL` → `PartialFilter::PropertyExists`
///
/// Returns `None` for unsupported expressions (index is still created; filter
/// is simply not applied during backfill or at write time).
fn expr_to_partial_filter(expr: &Expr) -> Option<crate::index::definition::PartialFilter> {
    use crate::index::definition::PartialFilter;
    use coordinode_core::graph::types::Value;

    match expr {
        // n.prop = <literal>
        Expr::BinaryOp { left, op, right } if *op == BinaryOperator::Eq => {
            let prop = extract_property_name(left)?;
            match right.as_ref() {
                Expr::Literal(Value::String(s)) => Some(PartialFilter::PropertyEquals {
                    property: prop,
                    value: s.clone(),
                }),
                Expr::Literal(Value::Int(n)) => Some(PartialFilter::PropertyEqualsInt {
                    property: prop,
                    value: *n,
                }),
                Expr::Literal(Value::Bool(b)) => Some(PartialFilter::PropertyEqualsBool {
                    property: prop,
                    value: *b,
                }),
                _ => None,
            }
        }
        // n.prop IS NOT NULL
        Expr::IsNull {
            expr: inner,
            negated: true,
        } => {
            let prop = extract_property_name(inner)?;
            Some(PartialFilter::PropertyExists { property: prop })
        }
        // EXISTS(n.prop) — represented as FunctionCall { name: "exists", args: [PropertyAccess] }
        Expr::FunctionCall { name, args, .. } if name.eq_ignore_ascii_case("exists") => {
            args.first().and_then(|a| {
                let prop = extract_property_name(a)?;
                Some(PartialFilter::PropertyExists { property: prop })
            })
        }
        _ => None,
    }
}

/// Extract the property name from a `PropertyAccess` or `Variable` expression.
///
/// Handles `n.status` (PropertyAccess) as well as bare variable strings that
/// encode property access as `"n.status"` (Variable fallback).
fn extract_property_name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::PropertyAccess { property, .. } => Some(property.clone()),
        // Some parser representations encode n.prop as Variable("n.prop")
        Expr::Variable(v) if v.contains('.') => v.split_once('.').map(|(_, prop)| prop.to_string()),
        _ => None,
    }
}

/// Optimization pass: replace `Filter { input: NodeScan { label }, predicate: prop=val }` with
/// `IndexScan` when a matching B-tree index is registered in the registry.
///
/// The rewrite is applied bottom-up. Only equality predicates (`prop = literal/param`) on a
/// single-label `NodeScan` are eligible. Compound predicates (`AND`/`OR`) are not rewritten —
/// only the top-level equality is matched.
///
/// This pass is separate from `build_logical_plan` so callers without an index registry can
/// skip it without changing the planner's public signature.
/// Pull a lifted correlated equality into a correlated `IndexScan` on the join's
/// right side.
///
/// A MATCH building on a prior clause (`UNWIND ... AS e  MATCH (b:L) WHERE
/// b.prop = e.x`) plans as `Filter(CartesianProduct(left, NodeScan(b)),
/// b.prop = e.x)`: the equality references the outer `e`, so it is lifted above
/// the join and cannot be rewritten by the `Filter(NodeScan, ...)` rule. That
/// leaves `b` as a full label scan per outer row. When an index covers
/// `L(prop)` and the key does not reference `b`, rewrite to
/// `CartesianProduct(left, IndexScan(b, prop = e.x))`: the executor runs the
/// right side per outer row (the key resolves against the correlated row), so
/// the endpoint lookup becomes a point read instead of a full scan.
fn try_correlated_index_join(
    input: &LogicalOp,
    predicate: &Expr,
    registry: &crate::index::IndexRegistry,
) -> Option<LogicalOp> {
    let LogicalOp::CartesianProduct { left, right } = input else {
        return None;
    };
    let LogicalOp::NodeScan {
        variable,
        labels,
        property_filters,
    } = right.as_ref()
    else {
        return None;
    };
    if labels.len() != 1 || !property_filters.is_empty() {
        return None;
    }
    let Expr::BinaryOp {
        left: pl,
        op: BinaryOperator::Eq,
        right: pr,
    } = predicate
    else {
        return None;
    };
    let prop = extract_index_property(pl, variable)?;
    if expr_references_var(pr, variable) {
        return None;
    }
    let scan = try_index_rewrite(variable, &labels[0], &prop, pr, registry)?;
    Some(LogicalOp::CartesianProduct {
        left: Box::new(optimize_index_selection((**left).clone(), registry)),
        right: Box::new(scan),
    })
}

pub fn optimize_index_selection(
    op: LogicalOp,
    registry: &crate::index::IndexRegistry,
) -> LogicalOp {
    match op {
        // Filter(NodeScan{single-label, no inline filters}, var.prop = key)
        // -> IndexScan when an index exists and `key` does not reference the
        // scan variable. `key` may be a literal, a parameter, or a correlated
        // outer value (e.g. `WHERE a.pid = e.s` driven per UNWIND row); the
        // executor resolves a correlated key against the outer row.
        LogicalOp::Filter { input, predicate } => {
            if let LogicalOp::NodeScan {
                ref variable,
                ref labels,
                ref property_filters,
            } = *input
            {
                if labels.len() == 1 && property_filters.is_empty() {
                    if let Expr::BinaryOp {
                        ref left,
                        op: BinaryOperator::Eq,
                        ref right,
                    } = predicate
                    {
                        if let Some(prop) = extract_index_property(left, variable) {
                            if !expr_references_var(right, variable) {
                                if let Some(scan) =
                                    try_index_rewrite(variable, &labels[0], &prop, right, registry)
                                {
                                    return scan;
                                }
                            }
                        }
                    }
                }
            }
            // Correlated equality lifted above a join (later MATCH on a prior
            // binding): pull it into a correlated IndexScan on the right side.
            if let Some(rewritten) = try_correlated_index_join(&input, &predicate, registry) {
                return rewritten;
            }
            LogicalOp::Filter {
                input: Box::new(optimize_index_selection(*input, registry)),
                predicate,
            }
        }
        // Bare NodeScan with exactly one inline equality filter on an indexed
        // property -> IndexScan. This is the lowered form of `MATCH (a:L {p: k})`
        // / `WHERE a.p = k`. `k` must not reference the scan variable. Only a
        // single filter is rewritten (a point lookup carries one key); a
        // multi-filter scan stays a NodeScan so residual predicates still apply.
        LogicalOp::NodeScan {
            variable,
            labels,
            property_filters,
        } if labels.len() == 1 && property_filters.len() == 1 => {
            let (prop, key) = &property_filters[0];
            if !expr_references_var(key, &variable) {
                if let Some(scan) = try_index_rewrite(&variable, &labels[0], prop, key, registry) {
                    return scan;
                }
            }
            LogicalOp::NodeScan {
                variable,
                labels,
                property_filters,
            }
        }
        LogicalOp::Project {
            input,
            items,
            distinct,
        } => LogicalOp::Project {
            input: Box::new(optimize_index_selection(*input, registry)),
            items,
            distinct,
        },
        LogicalOp::Sort { input, items } => LogicalOp::Sort {
            input: Box::new(optimize_index_selection(*input, registry)),
            items,
        },
        LogicalOp::Limit { input, count } => LogicalOp::Limit {
            input: Box::new(optimize_index_selection(*input, registry)),
            count,
        },
        LogicalOp::Skip { input, count } => LogicalOp::Skip {
            input: Box::new(optimize_index_selection(*input, registry)),
            count,
        },
        LogicalOp::Traverse {
            input,
            source,
            edge_types,
            direction,
            target_variable,
            target_labels,
            length,
            edge_variable,
            target_filters,
            edge_filters,
            temporal_filter,
            path_variable,
        } => LogicalOp::Traverse {
            input: Box::new(optimize_index_selection(*input, registry)),
            source,
            edge_types,
            direction,
            target_variable,
            target_labels,
            length,
            edge_variable,
            target_filters,
            edge_filters,
            temporal_filter,
            path_variable,
        },
        LogicalOp::Aggregate {
            input,
            group_by,
            aggregates,
        } => LogicalOp::Aggregate {
            input: Box::new(optimize_index_selection(*input, registry)),
            group_by,
            aggregates,
        },
        LogicalOp::CartesianProduct { left, right } => LogicalOp::CartesianProduct {
            left: Box::new(optimize_index_selection(*left, registry)),
            right: Box::new(optimize_index_selection(*right, registry)),
        },
        LogicalOp::LeftOuterJoin { left, right } => LogicalOp::LeftOuterJoin {
            left: Box::new(optimize_index_selection(*left, registry)),
            right: Box::new(optimize_index_selection(*right, registry)),
        },
        LogicalOp::Unwind {
            input,
            expr,
            variable,
        } => LogicalOp::Unwind {
            input: Box::new(optimize_index_selection(*input, registry)),
            expr,
            variable,
        },
        LogicalOp::CreateNode {
            input,
            variable,
            labels,
            properties,
        } => LogicalOp::CreateNode {
            input: input.map(|i| Box::new(optimize_index_selection(*i, registry))),
            variable,
            labels,
            properties,
        },
        LogicalOp::CreateEdge {
            input,
            source,
            target,
            edge_type,
            direction,
            variable,
            properties,
        } => LogicalOp::CreateEdge {
            input: Box::new(optimize_index_selection(*input, registry)),
            source,
            target,
            edge_type,
            direction,
            variable,
            properties,
        },
        // Leaf ops, DDL ops, mutation ops over non-MATCH inputs, and any op
        // without a child relevant to index selection — return as-is.
        other => other,
    }
}

/// Build an `IndexScan` for `(label, property) = value_expr` if a B-tree index
/// is registered for that pair. Returns None when no index matches.
fn try_index_rewrite(
    variable: &str,
    label: &str,
    property: &str,
    value_expr: &Expr,
    registry: &crate::index::IndexRegistry,
) -> Option<LogicalOp> {
    let idx = registry
        .indexes_for_property(label, property)
        .into_iter()
        .next()?;
    Some(LogicalOp::IndexScan {
        variable: variable.to_string(),
        label: label.to_string(),
        index_name: idx.name.clone(),
        property: property.to_string(),
        value_expr: value_expr.clone(),
    })
}

/// True if `expr` references `var` (as `Variable(var)`, `Variable("var.prop")`,
/// or a property access on `var`). Rejects self-referential equality
/// (`a.x = a.y`) from index point-lookup rewriting while allowing literals,
/// parameters, and correlated outer references.
fn expr_references_var(expr: &Expr, var: &str) -> bool {
    match expr {
        Expr::Variable(name) => {
            name == var || name.strip_prefix(var).is_some_and(|s| s.starts_with('.'))
        }
        Expr::PropertyAccess { expr, .. } | Expr::UnaryOp { expr, .. } => {
            expr_references_var(expr, var)
        }
        Expr::BinaryOp { left, right, .. } => {
            expr_references_var(left, var) || expr_references_var(right, var)
        }
        Expr::FunctionCall { args, .. } | Expr::List(args) => {
            args.iter().any(|e| expr_references_var(e, var))
        }
        _ => false,
    }
}

/// Extract a property name from an expression that references `variable.property`.
///
/// Matches `PropertyAccess { expr: Variable(var), property }` and `Variable("var.prop")`.
fn extract_index_property(expr: &Expr, variable: &str) -> Option<String> {
    match expr {
        Expr::PropertyAccess {
            expr: inner,
            property,
        } => {
            if matches!(inner.as_ref(), Expr::Variable(v) if v == variable) {
                Some(property.clone())
            } else {
                None
            }
        }
        // Fallback: some nodes use Variable("n.prop")
        Expr::Variable(v) => {
            let expected_prefix = format!("{variable}.");
            v.strip_prefix(&expected_prefix).map(|s| s.to_string())
        }
        _ => None,
    }
}

/// Annotation pass: walk the plan tree and annotate each `VectorTopK` node with
/// Annotates `VectorTopK` nodes with `hnsw_index = Some("name, metric")` when
/// a matching HNSW index exists in the registry for the (label, property) pair.
///
/// This serves two purposes:
/// 1. **EXPLAIN output** — drives the `strategy: HnswScan(idx, metric) | BruteForce` line.
/// 2. **Executor optimization** — the executor's `try_hnsw_vector_top_k` uses the
///    annotation to resolve (label, property) by index name instead of scanning
///    `rows[0].__label__`. Falls back to row detection when annotation is absent
///    or the index was dropped between planning and execution.
pub fn annotate_vector_top_k(
    op: LogicalOp,
    registry: &crate::index::VectorIndexRegistry,
) -> LogicalOp {
    match op {
        LogicalOp::VectorTopK {
            input,
            vector_expr,
            query_vector,
            function,
            k,
            distance_alias,
            predicate,
            ..
        } => {
            // Determine (variable, property) from the vector expression.
            let (variable, property) = match &vector_expr {
                Expr::PropertyAccess {
                    expr,
                    property: prop,
                } => match expr.as_ref() {
                    Expr::Variable(v) => (Some(v.clone()), Some(prop.clone())),
                    _ => (None, None),
                },
                _ => (None, None),
            };

            // Try to determine label from the input NodeScan (best effort).
            let label = variable.as_deref().and_then(|_| extract_scan_label(&input));

            // Look up the HNSW index definition when (label, property) are known.
            // Store "name, metric" so EXPLAIN can show HnswScan(idx, cosine).
            let hnsw_index = match (label, property) {
                (Some(lbl), Some(prop)) => registry.get_definition(lbl, &prop).map(|def| {
                    let metric = match def
                        .vector_config
                        .as_ref()
                        .map(|c| c.metric)
                        .unwrap_or(coordinode_core::graph::types::VectorMetric::Cosine)
                    {
                        coordinode_core::graph::types::VectorMetric::Cosine => "cosine",
                        coordinode_core::graph::types::VectorMetric::L2 => "l2",
                        coordinode_core::graph::types::VectorMetric::L1 => "l1",
                        coordinode_core::graph::types::VectorMetric::DotProduct => "dot",
                    };
                    format!("{}, {metric}", def.name)
                }),
                _ => None,
            };

            // Synthesise a pushdown predicate from sibling label + simple
            // WHERE leaves matching the same variable. Conservative: only
            // produce a predicate when we have a concrete label OR at least
            // one `var.prop = literal` leaf — anything else stays None and
            // the executor falls back to unfiltered HNSW.
            let predicate = predicate.or_else(|| {
                let var_str = variable.as_deref()?;
                let mut leaves: Vec<crate::planner::logical::VectorPredicate> = Vec::new();
                if let Some(lbl) = label {
                    leaves.push(crate::planner::logical::VectorPredicate::LabelEq(
                        lbl.to_string(),
                    ));
                }
                collect_simple_property_predicates(&input, var_str, &mut leaves);
                fold_predicate(leaves)
            });

            // Recurse into input.
            let input = Box::new(annotate_vector_top_k(*input, registry));

            LogicalOp::VectorTopK {
                input,
                vector_expr,
                query_vector,
                function,
                k,
                distance_alias,
                hnsw_index,
                predicate,
            }
        }

        // Recurse into all other operators with children.
        LogicalOp::Filter { input, predicate } => LogicalOp::Filter {
            input: Box::new(annotate_vector_top_k(*input, registry)),
            predicate,
        },
        LogicalOp::Project {
            input,
            items,
            distinct,
        } => LogicalOp::Project {
            input: Box::new(annotate_vector_top_k(*input, registry)),
            items,
            distinct,
        },
        LogicalOp::Sort { input, items } => LogicalOp::Sort {
            input: Box::new(annotate_vector_top_k(*input, registry)),
            items,
        },
        LogicalOp::Limit { input, count } => LogicalOp::Limit {
            input: Box::new(annotate_vector_top_k(*input, registry)),
            count,
        },
        LogicalOp::Skip { input, count } => LogicalOp::Skip {
            input: Box::new(annotate_vector_top_k(*input, registry)),
            count,
        },
        LogicalOp::Aggregate {
            input,
            group_by,
            aggregates,
        } => LogicalOp::Aggregate {
            input: Box::new(annotate_vector_top_k(*input, registry)),
            group_by,
            aggregates,
        },
        LogicalOp::CartesianProduct { left, right } => LogicalOp::CartesianProduct {
            left: Box::new(annotate_vector_top_k(*left, registry)),
            right: Box::new(annotate_vector_top_k(*right, registry)),
        },
        LogicalOp::LeftOuterJoin { left, right } => LogicalOp::LeftOuterJoin {
            left: Box::new(annotate_vector_top_k(*left, registry)),
            right: Box::new(annotate_vector_top_k(*right, registry)),
        },
        // Leaf nodes, DDL, Traverse, EdgeVectorSearch, and other complex operators.
        // VectorTopK always sits above the NodeScan/Traverse level, so these never
        // contain a VectorTopK child in a valid plan. Return unchanged.
        other => other,
    }
}

/// Access-path pass: replace `VectorTopK { input: bare NodeScan }` with
/// the [`LogicalOp::HnswScan`] SOURCE operator when the query is a pure
/// vector top-K over one label with a registered HNSW index.
///
/// "Bare" means the chain between VectorTopK and NodeScan contains no
/// Filter / Traverse / anything that narrows or widens the row set —
/// only a star-preserving Project is allowed. Those queries keep the
/// scan-then-rank path, which composes with filtered HNSW search.
///
/// Runs AFTER [`annotate_vector_top_k`] so the registry lookup and the
/// label/property extraction logic stay in one place conceptually; this
/// pass re-derives them because the annotation only carries a display
/// string.
pub fn apply_hnsw_scan_access_path(
    op: LogicalOp,
    registry: &crate::index::VectorIndexRegistry,
) -> LogicalOp {
    match op {
        LogicalOp::VectorTopK {
            input,
            vector_expr,
            query_vector,
            function,
            k,
            distance_alias,
            hnsw_index,
            predicate,
        } => {
            let rewrite = hnsw_scan_target(&input, &vector_expr, &function, &predicate, registry);
            match rewrite {
                Some((binding, label, property, index_name)) => LogicalOp::HnswScan {
                    label,
                    property,
                    binding,
                    query_vector,
                    k,
                    function,
                    distance_alias,
                    index_name,
                },
                None => LogicalOp::VectorTopK {
                    input: Box::new(apply_hnsw_scan_access_path(*input, registry)),
                    vector_expr,
                    query_vector,
                    function,
                    k,
                    distance_alias,
                    hnsw_index,
                    predicate,
                },
            }
        }
        LogicalOp::Project {
            input,
            items,
            distinct,
        } => LogicalOp::Project {
            input: Box::new(apply_hnsw_scan_access_path(*input, registry)),
            items,
            distinct,
        },
        LogicalOp::Filter { input, predicate } => LogicalOp::Filter {
            input: Box::new(apply_hnsw_scan_access_path(*input, registry)),
            predicate,
        },
        LogicalOp::Sort { input, items } => LogicalOp::Sort {
            input: Box::new(apply_hnsw_scan_access_path(*input, registry)),
            items,
        },
        LogicalOp::Limit { input, count } => LogicalOp::Limit {
            input: Box::new(apply_hnsw_scan_access_path(*input, registry)),
            count,
        },
        LogicalOp::Skip { input, count } => LogicalOp::Skip {
            input: Box::new(apply_hnsw_scan_access_path(*input, registry)),
            count,
        },
        LogicalOp::Aggregate {
            input,
            group_by,
            aggregates,
        } => LogicalOp::Aggregate {
            input: Box::new(apply_hnsw_scan_access_path(*input, registry)),
            group_by,
            aggregates,
        },
        LogicalOp::Unwind {
            input,
            expr,
            variable,
        } => LogicalOp::Unwind {
            input: Box::new(apply_hnsw_scan_access_path(*input, registry)),
            expr,
            variable,
        },
        LogicalOp::CartesianProduct { left, right } => LogicalOp::CartesianProduct {
            left: Box::new(apply_hnsw_scan_access_path(*left, registry)),
            right: Box::new(apply_hnsw_scan_access_path(*right, registry)),
        },
        LogicalOp::LeftOuterJoin { left, right } => LogicalOp::LeftOuterJoin {
            left: Box::new(apply_hnsw_scan_access_path(*left, registry)),
            right: Box::new(apply_hnsw_scan_access_path(*right, registry)),
        },
        // VectorTopK never hides below other operator kinds in a valid
        // plan; leaves / DDL / writes are returned unchanged.
        other => other,
    }
}

/// Decide whether a `VectorTopK` qualifies for the HnswScan access path.
/// Returns `(binding, label, property, index_name)` when ALL of:
///
/// - the input is a BARE single-label `NodeScan` with no inline
///   property filters (any Filter / Traverse in between disqualifies —
///   those compose with filtered HNSW search instead),
/// - the vector expression is `scan_variable.property`,
/// - the pushdown predicate is absent or just the scan label
///   (annotate_vector_top_k synthesises `LabelEq` even for pure scans),
/// - the registry has an index for `(label, property)` whose metric
///   matches the ORDER BY function, so the index order IS the result
///   order.
fn hnsw_scan_target(
    input: &LogicalOp,
    vector_expr: &Expr,
    function: &str,
    predicate: &Option<crate::planner::logical::VectorPredicate>,
    registry: &crate::index::VectorIndexRegistry,
) -> Option<(String, String, String, String)> {
    let LogicalOp::NodeScan {
        variable,
        labels,
        property_filters,
    } = input
    else {
        return None;
    };
    if labels.len() != 1 || !property_filters.is_empty() {
        return None;
    }
    let label = &labels[0];

    let Expr::PropertyAccess { expr, property } = vector_expr else {
        return None;
    };
    let Expr::Variable(v) = expr.as_ref() else {
        return None;
    };
    if v != variable {
        return None;
    }

    use crate::planner::logical::VectorPredicate;
    match predicate {
        None => {}
        Some(VectorPredicate::LabelEq(l)) if l == label => {}
        Some(_) => return None,
    }

    let def = registry.get_definition(label, property)?;
    let metric = def
        .vector_config
        .as_ref()
        .map(|c| c.metric)
        .unwrap_or(coordinode_core::graph::types::VectorMetric::Cosine);
    use coordinode_core::graph::types::VectorMetric;
    let compatible = matches!(
        (function, metric),
        ("vector_distance", VectorMetric::L2)
            | ("vector_similarity", VectorMetric::Cosine)
            | ("vector_dot", VectorMetric::DotProduct)
            | ("vector_manhattan", VectorMetric::L1)
    );
    if !compatible {
        return None;
    }

    Some((
        variable.clone(),
        label.clone(),
        property.clone(),
        def.name.clone(),
    ))
}

/// Extract a single label from a NodeScan (or Filter over NodeScan) in the plan subtree.
/// Walk an op tree below a VectorTopK and harvest every simple
/// `variable.property = literal` predicate that references the same
/// variable. Pushed as `VectorPredicate::PropertyEq` leaves onto `out`.
///
/// Conservative: only matches `Filter { predicate: BinaryOp Eq }` nodes
/// where both sides reduce to (`PropertyAccess(var, prop)` and
/// `Literal(...)`). Anything more complex (numeric range, IS NULL, OR
/// branches, parameters) is ignored; the executor falls back to the
/// post-filter for those.
fn collect_simple_property_predicates(
    op: &LogicalOp,
    variable: &str,
    out: &mut Vec<crate::planner::logical::VectorPredicate>,
) {
    match op {
        LogicalOp::Filter { input, predicate } => {
            extract_predicate_leaves(predicate, variable, out);
            collect_simple_property_predicates(input, variable, out);
        }
        LogicalOp::Project { input, .. }
        | LogicalOp::Sort { input, .. }
        | LogicalOp::Limit { input, .. }
        | LogicalOp::Skip { input, .. } => {
            collect_simple_property_predicates(input, variable, out);
        }
        _ => {}
    }
}

/// Pull `VectorPredicate::PropertyEq` leaves out of a single Cypher
/// expression. Recurses into top-level `AND` conjunctions so a filter
/// like `n.category = 'X' AND n.active = true` yields two leaves.
fn extract_predicate_leaves(
    expr: &Expr,
    variable: &str,
    out: &mut Vec<crate::planner::logical::VectorPredicate>,
) {
    use crate::cypher::ast::BinaryOperator;

    match expr {
        Expr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
        } => {
            extract_predicate_leaves(left, variable, out);
            extract_predicate_leaves(right, variable, out);
        }
        Expr::BinaryOp {
            left,
            op: BinaryOperator::Eq,
            right,
        } => {
            if let Some(leaf) = property_eq_leaf(left, right, variable)
                .or_else(|| property_eq_leaf(right, left, variable))
            {
                out.push(leaf);
            }
        }
        Expr::BinaryOp {
            left,
            op:
                cmp @ (BinaryOperator::Gt
                | BinaryOperator::Gte
                | BinaryOperator::Lt
                | BinaryOperator::Lte),
            right,
        } => {
            // Numeric comparison: try the prop-on-left form, then prop-on-right
            // (which flips the operator: `5 <= n.score` becomes `n.score >= 5`).
            let op_forward = numeric_cmp_from_op(*cmp);
            if let Some(leaf) = property_cmp_leaf(left, right, variable, op_forward) {
                out.push(leaf);
            } else if let Some(flipped) = flip_numeric_cmp(*cmp) {
                if let Some(leaf) = property_cmp_leaf(right, left, variable, flipped) {
                    out.push(leaf);
                }
            }
        }
        _ => {}
    }
}

fn numeric_cmp_from_op(
    op: crate::cypher::ast::BinaryOperator,
) -> crate::planner::logical::NumericCmp {
    use crate::cypher::ast::BinaryOperator as B;
    use crate::planner::logical::NumericCmp as N;
    match op {
        B::Gt => N::Gt,
        B::Gte => N::Ge,
        B::Lt => N::Lt,
        B::Lte => N::Le,
        // The caller restricts us to the four comparison ops above; this
        // branch is unreachable in practice but keeps the function total.
        _ => N::Eq2Ne(),
    }
}

/// Helper used by `numeric_cmp_from_op` when an unexpected operator slips
/// through. Returning a default-ish variant keeps the planner from panicking;
/// the executor's PropertyCmp arm then rejects the leaf when the variant
/// doesn't match the requested semantic. Marked deliberately inconvenient so
/// nobody mistakes it for a real comparator.
impl crate::planner::logical::NumericCmp {
    #[allow(non_snake_case)]
    fn Eq2Ne() -> Self {
        // Choose Ge as the dead-default; a Gt would equally do — the
        // upstream extract_predicate_leaves guards against this path.
        Self::Ge
    }
}

fn flip_numeric_cmp(
    op: crate::cypher::ast::BinaryOperator,
) -> Option<crate::planner::logical::NumericCmp> {
    use crate::cypher::ast::BinaryOperator as B;
    use crate::planner::logical::NumericCmp as N;
    Some(match op {
        // `lit < prop` ↔ `prop > lit`
        B::Lt => N::Gt,
        B::Lte => N::Ge,
        B::Gt => N::Lt,
        B::Gte => N::Le,
        _ => return None,
    })
}

/// Build a `VectorPredicate::PropertyCmp` from a `(prop_side, literal_side)`
/// pair when `prop_side` is `PropertyAccess(variable, prop)` and
/// `literal_side` is a numeric `Literal(...)`. Non-numeric literals reject.
fn property_cmp_leaf(
    prop_side: &Expr,
    literal_side: &Expr,
    variable: &str,
    op: crate::planner::logical::NumericCmp,
) -> Option<crate::planner::logical::VectorPredicate> {
    let prop_name = match prop_side {
        Expr::PropertyAccess { expr, property } => match expr.as_ref() {
            Expr::Variable(v) if v == variable => property.clone(),
            _ => return None,
        },
        _ => return None,
    };
    // Accept Int / Float literals (and unary-neg of them). Strings / bools /
    // arrays reject — the executor's numeric() helper would reject them too,
    // but rejecting at plan time keeps the predicate descriptor honest.
    let value = match literal_side {
        Expr::Literal(coordinode_core::graph::types::Value::Int(_))
        | Expr::Literal(coordinode_core::graph::types::Value::Float(_)) => match literal_side {
            Expr::Literal(v) => v.clone(),
            _ => return None,
        },
        Expr::UnaryOp {
            op: crate::cypher::ast::UnaryOperator::Neg,
            expr: inner,
        } => match inner.as_ref() {
            Expr::Literal(coordinode_core::graph::types::Value::Int(i)) => {
                coordinode_core::graph::types::Value::Int(-i)
            }
            Expr::Literal(coordinode_core::graph::types::Value::Float(f)) => {
                coordinode_core::graph::types::Value::Float(-f)
            }
            _ => return None,
        },
        _ => return None,
    };
    Some(crate::planner::logical::VectorPredicate::PropertyCmp {
        property: prop_name,
        op,
        value,
    })
}

/// Build a `VectorPredicate::PropertyEq` from a `(prop_side, literal_side)`
/// pair when `prop_side` is `PropertyAccess(variable, prop)` and
/// `literal_side` is a `Literal(...)` of a directly-comparable type.
fn property_eq_leaf(
    prop_side: &Expr,
    literal_side: &Expr,
    variable: &str,
) -> Option<crate::planner::logical::VectorPredicate> {
    let prop_name = match prop_side {
        Expr::PropertyAccess { expr, property } => match expr.as_ref() {
            Expr::Variable(v) if v == variable => property.clone(),
            _ => return None,
        },
        _ => return None,
    };
    let value = match literal_side {
        Expr::Literal(v) => v.clone(),
        _ => return None,
    };
    Some(crate::planner::logical::VectorPredicate::PropertyEq {
        property: prop_name,
        value,
    })
}

/// Combine a vector of leaves into a single `VectorPredicate` via nested
/// `And`. Returns `None` for empty input so the caller falls back to the
/// "no predicate" path.
fn fold_predicate(
    mut leaves: Vec<crate::planner::logical::VectorPredicate>,
) -> Option<crate::planner::logical::VectorPredicate> {
    let first = leaves.pop()?;
    Some(leaves.into_iter().rev().fold(first, |acc, leaf| {
        crate::planner::logical::VectorPredicate::And(Box::new(leaf), Box::new(acc))
    }))
}

fn extract_scan_label(op: &LogicalOp) -> Option<&str> {
    match op {
        LogicalOp::NodeScan { labels, .. } if labels.len() == 1 => Some(&labels[0]),
        LogicalOp::Filter { input, .. } => extract_scan_label(input),
        LogicalOp::IndexScan { label, .. } => Some(label),
        _ => None,
    }
}

/// Walk a plan subtree looking for any `Traverse` node. Used by
/// [`optimize_push_down`] to decide whether the strategy rule applies.
fn contains_traverse(op: &LogicalOp) -> bool {
    match op {
        LogicalOp::Traverse { .. } => true,
        LogicalOp::Project { input, .. }
        | LogicalOp::Filter { input, .. }
        | LogicalOp::VectorFilter { input, .. }
        | LogicalOp::EdgeVectorSearch { input, .. }
        | LogicalOp::VectorTopK { input, .. }
        | LogicalOp::Sort { input, .. }
        | LogicalOp::Limit { input, .. }
        | LogicalOp::Skip { input, .. }
        | LogicalOp::Aggregate { input, .. } => contains_traverse(input),
        LogicalOp::CartesianProduct { left, right } | LogicalOp::LeftOuterJoin { left, right } => {
            contains_traverse(left) || contains_traverse(right)
        }
        _ => false,
    }
}

/// Extract `(label, property)` from a vector expression like `n.embedding`.
/// Returns `None` for non-property expressions or when the property holder
/// is not a single-variable reference.
fn extract_vector_label_prop<'a>(
    vector_expr: &'a Expr,
    plan: &'a LogicalOp,
) -> Option<(&'a str, &'a str)> {
    match vector_expr {
        Expr::PropertyAccess { expr, property } => match expr.as_ref() {
            Expr::Variable(_) => {
                let label = extract_scan_label(plan)?;
                Some((label, property.as_str()))
            }
            _ => None,
        },
        _ => None,
    }
}

/// Compute the push-down decision for a `VectorFilter` whose upstream input
/// contains a `Traverse`. Pulls per-index statistics via [`StorageStats`]
/// when available; falls back to documented defaults from
/// `arch/core/query-engine.md` § Graph Predicate Push-Down otherwise.
///
/// `enclosing_limit` is the LIMIT value of the enclosing ancestor (the
/// optimizer pass walks top-down and remembers Limit values it has seen
/// on the way down). `None` means no LIMIT in scope — falls back to
/// top_k = 100.
///
/// Estimation is approximate by design — the planner compares costs
/// ordinally, and refinement (better selectivity estimation, histograms,
/// per-shard fan-out) is the scope of future tasks (R-PUSH2/R-PUSH4 and
/// later CBO work). What R-PUSH1 guarantees is that the rule fires and
/// produces a deterministic decision; the EXPLAIN-visible cost numbers
/// will tighten over releases without breaking the strategy contract.
fn compute_push_down_decision(
    vector_expr: &Expr,
    input: &LogicalOp,
    threshold: f64,
    less_than: bool,
    enclosing_limit: Option<usize>,
    stats: Option<&dyn coordinode_core::graph::stats::StorageStats>,
) -> crate::planner::push_down::PushDownDecision {
    use crate::planner::push_down::{select_push_down_strategy, VectorIndexParams};

    // ── estimate |C| from the upstream traversal ──────────────────────
    //
    // Conservative default when stats are unavailable: 1000 candidates —
    // above both crossover ceilings (200/500) but below typical HNSW size.
    // This keeps the planner from defaulting to graph-first on huge
    // traversals when we have no information.
    let estimated_candidates = stats
        .and_then(|s| {
            // Use the label of the deepest NodeScan as a fan-out anchor.
            let label = extract_scan_label(input)?;
            s.node_count_for_label(label).map(|n| n as usize)
        })
        .unwrap_or(1_000);

    // ── per-index parameters ───────────────────────────────────────────
    let (label_opt, prop_opt) = extract_vector_label_prop(vector_expr, input)
        .map(|(l, p)| (Some(l), Some(p)))
        .unwrap_or((None, None));

    let index_size = stats
        .and_then(|s| match (label_opt, prop_opt) {
            (Some(l), Some(p)) => s.vector_index_size(l, p).map(|n| n as usize),
            _ => None,
        })
        .unwrap_or(estimated_candidates.saturating_mul(10).max(10_000));

    let index_dim = stats
        .and_then(|s| match (label_opt, prop_opt) {
            (Some(l), Some(p)) => s.vector_index_dim(l, p),
            _ => None,
        })
        .unwrap_or(128);

    let crossover = stats
        .and_then(|s| match (label_opt, prop_opt) {
            (Some(l), Some(p)) => s.vector_index_crossover(l, p),
            _ => None,
        })
        .unwrap_or(500); // arch default for node-typed indexes

    let index = VectorIndexParams {
        size: index_size,
        dim: index_dim,
        m: 16,
        ef_search: 200,
        crossover_threshold: crossover,
    };

    // ── vector selectivity heuristic ───────────────────────────────────
    //
    // For `vector_similarity(...) > 0.9`, low threshold means very few
    // matches; for `vector_distance(...) < 0.1`, similarly selective. The
    // current heuristic is intentionally simple — neutral 0.5 unless the
    // threshold is extreme. R-PUSH2 will refine this from real index
    // distribution stats.
    let vector_selectivity = if less_than {
        // distance < threshold: lower threshold = fewer matches = lower selectivity
        (threshold * 0.5).clamp(0.001, 1.0)
    } else {
        // similarity > threshold: higher threshold = fewer matches = lower selectivity
        ((1.0 - threshold) * 0.5).clamp(0.001, 1.0)
    };

    // ── top-K from LIMIT (passed down from ancestor) ──────────────────
    //
    // The vector-first cost formula includes a `K × graph_verify_cost`
    // term. The optimizer pass walks top-down and remembers the LIMIT
    // value of any ancestor Limit operator it has crossed; that value
    // reaches us via `enclosing_limit`. If `None` (no LIMIT in scope or
    // it was an unevaluated parameter), we use 100 — large enough to
    // make vector_first non-trivially priced, small enough not to
    // dominate the cost comparison.
    let top_k = enclosing_limit.unwrap_or(100);

    select_push_down_strategy(estimated_candidates, index, vector_selectivity, top_k)
}

/// Extract a literal integer value from a `Limit { count }` expression
/// (only `LIMIT 25` style, not `LIMIT $n`). Returns `None` for
/// parameters or computed expressions — those fall through to the
/// default top_k in `compute_push_down_decision`.
fn extract_literal_limit(count: &Expr) -> Option<usize> {
    match count {
        Expr::Literal(coordinode_core::graph::types::Value::Int(n)) if *n > 0 => Some(*n as usize),
        _ => None,
    }
}

/// Planner pass implementing the **graph predicate push-down rule** from
/// `arch/core/query-engine.md` § Graph Predicate Push-Down (R-PUSH1).
///
/// Walks the plan tree bottom-up; for every `VectorFilter` whose upstream
/// input contains a `Traverse`, computes a [`PushDownDecision`] and
/// annotates the operator. The decision picks one of three strategies
/// (graph-first / ACORN-filtered / vector-first) deterministically from
/// the cost model.
///
/// **Invariant** (contract-tested in `push_down_invariant_*` regression
/// tests): no plan emerging from this pass may contain a `VectorFilter`
/// directly preceded by `Traverse` with `push_down == None`.
///
/// Composes with [`optimize_edge_vector_search`] (runs before, may rewrite
/// `VectorFilter` to `EdgeVectorSearch`) and [`annotate_vector_top_k`]
/// (runs in parallel, independent dimension).
pub fn optimize_push_down(
    op: LogicalOp,
    stats: Option<&dyn coordinode_core::graph::stats::StorageStats>,
) -> LogicalOp {
    optimize_push_down_with_limit(op, None, stats)
}

/// Inner recursion that carries the enclosing LIMIT value top-down. When
/// a `Limit { count }` operator is visited, its literal value replaces
/// the enclosing value for all descendants — so a `VectorFilter` beneath
/// it sees the correct top-K for cost estimation.
fn optimize_push_down_with_limit(
    op: LogicalOp,
    enclosing_limit: Option<usize>,
    stats: Option<&dyn coordinode_core::graph::stats::StorageStats>,
) -> LogicalOp {
    match op {
        LogicalOp::VectorFilter {
            input,
            vector_expr,
            query_vector,
            function,
            less_than,
            threshold,
            decay_field,
            push_down,
        } => {
            // Recurse into input first so nested filters get a chance.
            let new_input = optimize_push_down_with_limit(*input, enclosing_limit, stats);

            // If the input contains a Traverse and we don't already have a
            // decision, compute one. Preserve any pre-set decision (e.g.,
            // attached by a future cost-aware build pass).
            let decision = if push_down.is_some() {
                push_down
            } else if contains_traverse(&new_input) {
                Some(compute_push_down_decision(
                    &vector_expr,
                    &new_input,
                    threshold,
                    less_than,
                    enclosing_limit,
                    stats,
                ))
            } else {
                None
            };

            LogicalOp::VectorFilter {
                input: Box::new(new_input),
                vector_expr,
                query_vector,
                function,
                less_than,
                threshold,
                decay_field,
                push_down: decision,
            }
        }

        // Limit refreshes the enclosing top-K for everything below it.
        LogicalOp::Limit { input, count } => {
            let new_limit = extract_literal_limit(&count).or(enclosing_limit);
            LogicalOp::Limit {
                input: Box::new(optimize_push_down_with_limit(*input, new_limit, stats)),
                count,
            }
        }

        // Other unary operators propagate enclosing_limit unchanged.
        LogicalOp::Filter { input, predicate } => LogicalOp::Filter {
            input: Box::new(optimize_push_down_with_limit(
                *input,
                enclosing_limit,
                stats,
            )),
            predicate,
        },
        LogicalOp::Project {
            input,
            items,
            distinct,
        } => LogicalOp::Project {
            input: Box::new(optimize_push_down_with_limit(
                *input,
                enclosing_limit,
                stats,
            )),
            items,
            distinct,
        },
        LogicalOp::Sort { input, items } => LogicalOp::Sort {
            input: Box::new(optimize_push_down_with_limit(
                *input,
                enclosing_limit,
                stats,
            )),
            items,
        },
        LogicalOp::Skip { input, count } => LogicalOp::Skip {
            input: Box::new(optimize_push_down_with_limit(
                *input,
                enclosing_limit,
                stats,
            )),
            count,
        },
        LogicalOp::Aggregate {
            input,
            group_by,
            aggregates,
        } => LogicalOp::Aggregate {
            input: Box::new(optimize_push_down_with_limit(
                *input,
                enclosing_limit,
                stats,
            )),
            group_by,
            aggregates,
        },
        LogicalOp::CartesianProduct { left, right } => LogicalOp::CartesianProduct {
            left: Box::new(optimize_push_down_with_limit(*left, enclosing_limit, stats)),
            right: Box::new(optimize_push_down_with_limit(
                *right,
                enclosing_limit,
                stats,
            )),
        },
        LogicalOp::LeftOuterJoin { left, right } => LogicalOp::LeftOuterJoin {
            left: Box::new(optimize_push_down_with_limit(*left, enclosing_limit, stats)),
            right: Box::new(optimize_push_down_with_limit(
                *right,
                enclosing_limit,
                stats,
            )),
        },
        // Operators with no input or already terminal — pass through unchanged.
        other => other,
    }
}

/// Optimization pass: detect VectorFilter on edge variables after Traverse,
/// and rewrite to EdgeVectorSearch with the appropriate strategy.
///
/// Pattern detected:
///   VectorFilter { input: Traverse { edge_variable: Some(r), ... }, vector_expr: r.prop, ... }
///
/// When the vector expression references the edge variable of the input Traverse,
/// this is an edge vector query. The planner selects graph-first vs vector-first
/// based on estimated fan-out.
fn optimize_edge_vector_search(op: LogicalOp) -> LogicalOp {
    match op {
        LogicalOp::VectorFilter {
            input,
            vector_expr,
            query_vector,
            function,
            less_than,
            threshold,
            decay_field,
            push_down: _,
        } => {
            // First, recursively optimize children
            let input = Box::new(optimize_edge_vector_search(*input));

            // Check if input is a Traverse with an edge variable, and the vector_expr
            // references that edge variable (i.e., this is an edge vector filter).
            // Extract needed values before moving input.
            let edge_info = if let LogicalOp::Traverse {
                ref source,
                ref edge_types,
                ref edge_variable,
                ref target_variable,
                ..
            } = *input
            {
                edge_variable.as_ref().and_then(|ev| {
                    extract_variable_property(&vector_expr, ev).map(|prop| {
                        (
                            source.clone(),
                            edge_types.first().cloned().unwrap_or_default(),
                            ev.clone(),
                            target_variable.clone(),
                            prop,
                        )
                    })
                })
            } else {
                None
            };

            if let Some((source_var, edge_type, edge_var, target_var, prop)) = edge_info {
                // This is an edge vector query. Select strategy.
                // Default fan-out estimate: 200 (average node degree).
                // In future, this will use schema statistics.
                let default_fan_out = 200.0;

                // Vector selectivity approximation from threshold:
                // For distance < T: selectivity ≈ T (most distance thresholds are 0.0-1.0)
                // For similarity > T: selectivity ≈ 1.0 - T
                let selectivity = if less_than {
                    threshold.clamp(0.01, 1.0)
                } else {
                    (1.0 - threshold).clamp(0.01, 1.0)
                };

                let strategy =
                    super::logical::select_edge_vector_strategy(default_fan_out, selectivity);

                return LogicalOp::EdgeVectorSearch {
                    input,
                    edge_type,
                    vector_property: prop,
                    vector_expr,
                    query_vector,
                    function,
                    less_than,
                    threshold,
                    source_variable: source_var,
                    target_variable: target_var,
                    edge_variable: Some(edge_var),
                    strategy,
                };
            }

            // Not an edge vector pattern — keep as VectorFilter
            LogicalOp::VectorFilter {
                input,
                vector_expr,
                query_vector,
                function,
                less_than,
                threshold,
                decay_field,
                push_down: None,
            }
        }

        // Recursively optimize all operator children
        LogicalOp::Filter { input, predicate } => LogicalOp::Filter {
            input: Box::new(optimize_edge_vector_search(*input)),
            predicate,
        },
        LogicalOp::Project {
            input,
            items,
            distinct,
        } => LogicalOp::Project {
            input: Box::new(optimize_edge_vector_search(*input)),
            items,
            distinct,
        },
        LogicalOp::Aggregate {
            input,
            group_by,
            aggregates,
        } => LogicalOp::Aggregate {
            input: Box::new(optimize_edge_vector_search(*input)),
            group_by,
            aggregates,
        },
        LogicalOp::Sort { input, items } => LogicalOp::Sort {
            input: Box::new(optimize_edge_vector_search(*input)),
            items,
        },
        LogicalOp::Limit { input, count } => LogicalOp::Limit {
            input: Box::new(optimize_edge_vector_search(*input)),
            count,
        },
        LogicalOp::Skip { input, count } => LogicalOp::Skip {
            input: Box::new(optimize_edge_vector_search(*input)),
            count,
        },
        LogicalOp::Traverse {
            input,
            source,
            edge_types,
            direction,
            target_variable,
            target_labels,
            length,
            edge_variable,
            target_filters,
            edge_filters,
            temporal_filter,
            path_variable,
        } => LogicalOp::Traverse {
            input: Box::new(optimize_edge_vector_search(*input)),
            source,
            edge_types,
            direction,
            target_variable,
            target_labels,
            length,
            edge_variable,
            target_filters,
            edge_filters,
            temporal_filter,
            path_variable,
        },
        LogicalOp::TextFilter {
            input,
            text_expr,
            query_string,
            language,
        } => LogicalOp::TextFilter {
            input: Box::new(optimize_edge_vector_search(*input)),
            text_expr,
            query_string,
            language,
        },
        LogicalOp::EncryptedFilter {
            input,
            field_expr,
            token_expr,
        } => LogicalOp::EncryptedFilter {
            input: Box::new(optimize_edge_vector_search(*input)),
            field_expr,
            token_expr,
        },
        LogicalOp::CartesianProduct { left, right } => LogicalOp::CartesianProduct {
            left: Box::new(optimize_edge_vector_search(*left)),
            right: Box::new(optimize_edge_vector_search(*right)),
        },
        LogicalOp::LeftOuterJoin { left, right } => LogicalOp::LeftOuterJoin {
            left: Box::new(optimize_edge_vector_search(*left)),
            right: Box::new(optimize_edge_vector_search(*right)),
        },

        // Leaf nodes and ops that don't contain VectorFilter — return as-is
        other => other,
    }
}

/// Optimization pass: detect `Sort(vector_distance) + Limit(K)` patterns and
/// rewrite to `LogicalOp::VectorTopK` for HNSW-accelerated top-K search.
///
/// Detects two shapes:
///
/// **Pattern A (direct function call in ORDER BY):**
/// ```text
/// Limit(k) { Sort([SortItem { expr: vector_distance(n.prop, q), asc: true }]) { X } }
///   → VectorTopK(k, fn, vector_expr=n.prop, query_vector=q, input: X)
/// ```
///
/// **Pattern B (alias via Project, from `WITH` clause):**
/// ```text
/// Limit(k) {
///   Sort([SortItem { expr: Variable("d"), asc: true }]) {
///     Project([..., vector_distance(n.prop, q) AS d]) { X }
///   }
/// }
///   → VectorTopK(k, fn, vector_expr=n.prop, query_vector=q, distance_alias=Some("d"), input: X)
/// ```
///
/// The rewrite unwraps the inner `Project` when its only computation is the
/// vector_distance expression bound to the alias. Other projected columns are
/// preserved by leaving the outer `Project` (if any) untouched.
///
/// `vector_similarity`/`vector_dot` use DESC ordering (higher is better), so
/// we detect `ascending: false` for those.
fn optimize_vector_top_k(op: LogicalOp) -> LogicalOp {
    // Walk the tree bottom-up: recurse into children first, then try to rewrite
    // the current node.
    let op = descend_optimize_top_k(op);
    rewrite_top_k_at_root(op)
}

fn descend_optimize_top_k(op: LogicalOp) -> LogicalOp {
    match op {
        LogicalOp::Filter { input, predicate } => LogicalOp::Filter {
            input: Box::new(optimize_vector_top_k(*input)),
            predicate,
        },
        LogicalOp::Project {
            input,
            items,
            distinct,
        } => LogicalOp::Project {
            input: Box::new(optimize_vector_top_k(*input)),
            items,
            distinct,
        },
        LogicalOp::Aggregate {
            input,
            group_by,
            aggregates,
        } => LogicalOp::Aggregate {
            input: Box::new(optimize_vector_top_k(*input)),
            group_by,
            aggregates,
        },
        LogicalOp::Sort { input, items } => LogicalOp::Sort {
            input: Box::new(optimize_vector_top_k(*input)),
            items,
        },
        LogicalOp::Limit { input, count } => LogicalOp::Limit {
            input: Box::new(optimize_vector_top_k(*input)),
            count,
        },
        LogicalOp::Skip { input, count } => LogicalOp::Skip {
            input: Box::new(optimize_vector_top_k(*input)),
            count,
        },
        LogicalOp::Unwind {
            input,
            expr,
            variable,
        } => LogicalOp::Unwind {
            input: Box::new(optimize_vector_top_k(*input)),
            expr,
            variable,
        },
        LogicalOp::CartesianProduct { left, right } => LogicalOp::CartesianProduct {
            left: Box::new(optimize_vector_top_k(*left)),
            right: Box::new(optimize_vector_top_k(*right)),
        },
        LogicalOp::LeftOuterJoin { left, right } => LogicalOp::LeftOuterJoin {
            left: Box::new(optimize_vector_top_k(*left)),
            right: Box::new(optimize_vector_top_k(*right)),
        },
        // Write/DDL/leaf ops: no children to optimize
        other => other,
    }
}

/// Attempt to rewrite a `Limit { Sort { ... } }` subtree into `VectorTopK`.
///
/// Returns the input unchanged when the pattern does not match.
fn rewrite_top_k_at_root(op: LogicalOp) -> LogicalOp {
    // Match: Limit { count: int literal, input: Sort { single item, ASC, ... } }
    let LogicalOp::Limit {
        input: limit_input,
        count,
    } = op
    else {
        return op_identity_limit(op);
    };

    // k must be a non-negative integer literal.
    let k = match &count {
        Expr::Literal(Value::Int(n)) if *n >= 0 => *n as usize,
        _ => {
            return LogicalOp::Limit {
                input: limit_input,
                count,
            };
        }
    };

    // Inner must be Sort with exactly one item.
    let inner_sort = match *limit_input {
        LogicalOp::Sort { input, items } if items.len() == 1 => (input, items),
        other => {
            return LogicalOp::Limit {
                input: Box::new(other),
                count,
            };
        }
    };
    let (sort_input, mut sort_items) = inner_sort;
    let sort_item = sort_items.remove(0);

    // Try Pattern A: the sort expression is a direct vector_distance call.
    if let Some((vector_expr, query_vector, function)) = match_vector_distance_call(&sort_item.expr)
    {
        // Ascending for distance/manhattan, descending for similarity/dot_product.
        if !is_valid_direction(&function, sort_item.ascending) {
            return reconstruct_limit_sort(k, sort_item, *sort_input);
        }

        // Pattern A correctness check: when Sort.input is a Project that does NOT
        // carry through the vector variable as a full node reference, VectorTopK
        // would see a row missing `n.embedding` and produce an empty result. The
        // unoptimized Sort path, by contrast, computes vector_distance via the
        // original row which is still valid. Fall back to Sort+Limit in this case.
        //
        // This happens for: `MATCH (n) RETURN n.name ORDER BY vector_distance(n.emb,...) LIMIT K`
        // — Project drops `n.embedding`, so VectorTopK input rows don't have it.
        if let LogicalOp::Project { items, .. } = sort_input.as_ref() {
            if !project_preserves_vector_expr(&vector_expr, items) {
                return LogicalOp::Limit {
                    input: Box::new(LogicalOp::Sort {
                        input: sort_input,
                        items: vec![sort_item],
                    }),
                    count,
                };
            }
        }

        return LogicalOp::VectorTopK {
            input: sort_input,
            vector_expr,
            query_vector,
            function,
            k,
            distance_alias: None,
            hnsw_index: None,
            predicate: None,
        };
    }

    // Pattern A2: the sort expression is a direct maxsim_score call.
    // ColBERT-style late-interaction: DESC order always (higher score wins).
    if let Some((doc_expr, query_expr)) = match_maxsim_score_call(&sort_item.expr) {
        if sort_item.ascending {
            // ASC would mean "lowest similarity first" which is never the
            // intent for late-interaction retrieval. Fall back to the
            // generic Sort + Limit so the user's literal request stands.
            return reconstruct_limit_sort(k, sort_item, *sort_input);
        }
        // Correctness check: when Sort.input is a Project that does NOT
        // carry through the variable bound to the doc property, MaxSimTopK
        // would see rows missing the multi-vector and score zero. The
        // generic Sort + Limit path computes against the original row
        // which still binds the variable. Fall back in that case.
        if let LogicalOp::Project { items, .. } = sort_input.as_ref() {
            if !project_preserves_vector_expr(&doc_expr, items) {
                return LogicalOp::Limit {
                    input: Box::new(LogicalOp::Sort {
                        input: sort_input,
                        items: vec![sort_item],
                    }),
                    count,
                };
            }
        }
        return LogicalOp::MaxSimTopK {
            input: sort_input,
            doc_expr,
            query_expr,
            k,
            score_alias: None,
        };
    }

    // Pattern B: sort expr is a variable reference to an alias defined in
    // the inner Project.
    let alias_name = match &sort_item.expr {
        Expr::Variable(v) => Some(v.clone()),
        _ => None,
    };

    if let Some(alias) = alias_name {
        // Inner input must be a Project where one item is `vector_distance(...) AS alias`.
        if let LogicalOp::Project {
            input: proj_input,
            items,
            distinct,
        } = *sort_input
        {
            // Find the project item whose alias matches.
            let matching_idx = items
                .iter()
                .position(|it| it.alias.as_ref() == Some(&alias));
            if let Some(idx) = matching_idx {
                if let Some((vector_expr, query_vector, function)) =
                    match_vector_distance_call(&items[idx].expr)
                {
                    if is_valid_direction(&function, sort_item.ascending) {
                        // Remove the vector_distance item from the Project items;
                        // VectorTopK will write the alias directly into result rows.
                        // Preserve other Project items by wrapping VectorTopK in a
                        // replacement Project.
                        let mut other_items: Vec<ProjectItem> = items
                            .into_iter()
                            .enumerate()
                            .filter_map(|(i, it)| if i == idx { None } else { Some(it) })
                            .collect();
                        // Ensure the alias is still projected as a passthrough so
                        // downstream operators see it.
                        other_items.push(ProjectItem {
                            expr: Expr::Variable(alias.clone()),
                            alias: Some(alias.clone()),
                        });

                        let top_k = LogicalOp::VectorTopK {
                            input: proj_input,
                            vector_expr,
                            query_vector,
                            function,
                            k,
                            distance_alias: Some(alias),
                            hnsw_index: None,
                            predicate: None,
                        };

                        return LogicalOp::Project {
                            input: Box::new(top_k),
                            items: other_items,
                            distinct,
                        };
                    }
                }
            }

            // Pattern B didn't match — rebuild original tree.
            return LogicalOp::Limit {
                input: Box::new(LogicalOp::Sort {
                    input: Box::new(LogicalOp::Project {
                        input: proj_input,
                        items,
                        distinct,
                    }),
                    items: vec![sort_item],
                }),
                count,
            };
        }

        // alias lookup failed — rebuild original tree
        return LogicalOp::Limit {
            input: Box::new(LogicalOp::Sort {
                input: sort_input,
                items: vec![sort_item],
            }),
            count,
        };
    }

    // Neither pattern matched — rebuild original tree
    LogicalOp::Limit {
        input: Box::new(LogicalOp::Sort {
            input: sort_input,
            items: vec![sort_item],
        }),
        count,
    }
}

/// Identity rebuild for non-Limit operators (recursive descent already handled children).
fn op_identity_limit(op: LogicalOp) -> LogicalOp {
    op
}

/// Check whether a Project's items preserve the variable referenced by `vector_expr`.
///
/// `vector_expr` is expected to be `Variable(var).property`. A Project preserves it if:
/// - One of its items is `Expr::Star` (pass-through), OR
/// - One of its items is `Expr::Variable(var)` without alias rename (full node ref), OR
/// - One of its items is the exact same PropertyAccess `var.property`
///
/// When true, VectorTopK can run over the Project's output rows.
/// When false, VectorTopK would see a row missing the vector column — fall back.
fn project_preserves_vector_expr(vector_expr: &Expr, items: &[ProjectItem]) -> bool {
    let Expr::PropertyAccess {
        expr: inner,
        property,
    } = vector_expr
    else {
        return false;
    };
    let Expr::Variable(var_name) = inner.as_ref() else {
        return false;
    };

    for item in items {
        match &item.expr {
            Expr::Star => return true,
            Expr::Variable(v)
                if v == var_name
                    // Only a passthrough if the alias is either None or same as var.
                    // An aliased variable (`n AS m`) would rename the row key.
                    && (item.alias.is_none() || item.alias.as_ref() == Some(v)) =>
            {
                return true;
            }
            Expr::PropertyAccess {
                expr: item_inner,
                property: item_prop,
            } => {
                if let Expr::Variable(item_var) = item_inner.as_ref() {
                    if item_var == var_name && item_prop == property && item.alias.is_none() {
                        return true;
                    }
                }
            }
            _ => {}
        }
    }
    false
}

/// Match `vector_distance(<vector_expr>, <query_vector>)` function call.
/// Returns `(vector_expr, query_vector, function_name)` if matched.
fn match_vector_distance_call(expr: &Expr) -> Option<(Expr, Expr, String)> {
    if let Expr::FunctionCall { name, args, .. } = expr {
        if matches!(
            name.as_str(),
            "vector_distance" | "vector_similarity" | "vector_dot" | "vector_manhattan"
        ) && args.len() == 2
        {
            return Some((args[0].clone(), args[1].clone(), name.clone()));
        }
    }
    None
}

/// Match a `maxsim_score(doc_expr, query_expr)` call shape used by the
/// MaxSimTopK rewrite. Returns `(doc_expr, query_expr)` when the call
/// matches; otherwise `None`.
fn match_maxsim_score_call(expr: &Expr) -> Option<(Expr, Expr)> {
    if let Expr::FunctionCall { name, args, .. } = expr {
        if name == "maxsim_score" && args.len() == 2 {
            return Some((args[0].clone(), args[1].clone()));
        }
    }
    None
}

/// Validate that the ORDER BY direction matches the metric semantics.
///
/// - `vector_distance`, `vector_manhattan`: lower is better → ASC is valid
/// - `vector_similarity`, `vector_dot`: higher is better → DESC is valid
fn is_valid_direction(function: &str, ascending: bool) -> bool {
    match function {
        "vector_distance" | "vector_manhattan" => ascending,
        "vector_similarity" | "vector_dot" => !ascending,
        _ => false,
    }
}

/// Rebuild a `Limit { Sort { ... } }` subtree (used when optimization was rejected).
fn reconstruct_limit_sort(k: usize, sort_item: SortItem, sort_input: LogicalOp) -> LogicalOp {
    LogicalOp::Limit {
        input: Box::new(LogicalOp::Sort {
            input: Box::new(sort_input),
            items: vec![sort_item],
        }),
        count: Expr::Literal(Value::Int(k as i64)),
    }
}

/// Extract the property name if `expr` is `variable.property`.
fn extract_variable_property(expr: &Expr, variable: &str) -> Option<String> {
    if let Expr::PropertyAccess {
        expr: inner,
        property,
    } = expr
    {
        if let Expr::Variable(var) = inner.as_ref() {
            if var == variable {
                return Some(property.clone());
            }
        }
    }
    None
}

/// Build a logical operator tree from a MATCH clause.
fn build_match_op(mc: &MatchClause) -> Result<LogicalOp, PlanError> {
    if mc.patterns.is_empty() {
        return Err(PlanError::EmptyPattern);
    }

    // Build each pattern independently, then join with CartesianProduct
    let mut ops: Vec<LogicalOp> = Vec::new();
    for pattern in &mc.patterns {
        ops.push(build_pattern_scan(pattern)?);
    }

    let mut result = ops.remove(0);
    for op in ops {
        result = LogicalOp::CartesianProduct {
            left: Box::new(result),
            right: Box::new(op),
        };
    }

    // Apply WHERE filter if present.
    // Compound predicate splitting: decomposes AND predicates into
    // VectorFilter → TextFilter → generic Filter pipeline.
    if let Some(ref pred) = mc.where_clause {
        result = apply_compound_where(pred, result);
    }

    Ok(result)
}

/// Build a single-pair shortest-path plan from `p = shortestPath((a)-[*]->(b))`.
///
/// Both endpoints are scanned (carrying their inline label / property filters)
/// and joined, so the BFS runs between each matching `(a, b)` pair. The path is
/// bound to the named-path variable. v1 supports the self-contained form where
/// the endpoint filters live inside the `shortestPath(...)` pattern; endpoints
/// pre-bound by an earlier clause are not yet reused.
fn build_shortest_path(pattern: &Pattern) -> Result<LogicalOp, PlanError> {
    // Capped at the engine's hop ceiling; the executor clamps again to be safe.
    const SHORTEST_PATH_DEFAULT_MAX_HOPS: u64 = 10;

    let (src_np, rel, tgt_np) = match pattern.elements.as_slice() {
        [PatternElement::Node(a), PatternElement::Relationship(r), PatternElement::Node(b)] => {
            (a, r, b)
        }
        _ => {
            return Err(PlanError::ShortestPathShape(
                "expects a single relationship between two nodes, e.g. \
                 shortestPath((a)-[:KNOWS*]->(b))"
                    .to_string(),
            ));
        }
    };

    let source = src_np
        .variable
        .clone()
        .ok_or_else(|| PlanError::ShortestPathShape("source node must be named".to_string()))?;
    let target = tgt_np
        .variable
        .clone()
        .ok_or_else(|| PlanError::ShortestPathShape("target node must be named".to_string()))?;
    let path_variable = pattern.path_variable.clone().ok_or_else(|| {
        PlanError::ShortestPathShape(
            "must be assigned to a path variable, e.g. p = shortestPath(...)".to_string(),
        )
    })?;

    let scan_source = LogicalOp::NodeScan {
        variable: source.clone(),
        labels: src_np.labels.clone(),
        property_filters: src_np.properties.clone(),
    };
    let scan_target = LogicalOp::NodeScan {
        variable: target.clone(),
        labels: tgt_np.labels.clone(),
        property_filters: tgt_np.properties.clone(),
    };
    let input = LogicalOp::CartesianProduct {
        left: Box::new(scan_source),
        right: Box::new(scan_target),
    };

    let max_depth = rel
        .length
        .and_then(|l| l.max)
        .unwrap_or(SHORTEST_PATH_DEFAULT_MAX_HOPS);

    Ok(LogicalOp::ShortestPath {
        input: Box::new(input),
        source,
        target,
        edge_types: rel.rel_types.clone(),
        direction: rel.direction,
        max_depth,
        path_variable,
    })
}

/// Build a scan + traversal chain from a single pattern.
fn build_pattern_scan(pattern: &Pattern) -> Result<LogicalOp, PlanError> {
    if pattern.elements.is_empty() {
        return Err(PlanError::EmptyPattern);
    }

    if pattern.shortest_path {
        return build_shortest_path(pattern);
    }

    let mut current: Option<LogicalOp> = None;

    for element in &pattern.elements {
        match element {
            PatternElement::Node(np) => {
                if current.is_none() {
                    // First node in pattern: NodeScan
                    current = Some(LogicalOp::NodeScan {
                        variable: np.variable.clone().unwrap_or_default(),
                        labels: np.labels.clone(),
                        property_filters: np.properties.clone(),
                    });
                }
                // Subsequent nodes are handled as part of Traverse
            }
            PatternElement::Relationship(rp) => {
                let source = current.ok_or(PlanError::UnsupportedPattern)?;

                // Find the source variable from the previous node
                let source_var = find_last_variable(&source);

                // The next element should be a Node — find it
                // (In our AST, pattern_element alternates Node-Rel-Node)
                // The target node info will be in the NEXT iteration.
                // But we can't look ahead here, so we store the relationship
                // and the next Node iteration will complete the Traverse.
                // Actually, let me restructure: process pairs of (Rel, Node).

                // For now, create a Traverse with empty target (will be filled)
                current = Some(LogicalOp::Traverse {
                    input: Box::new(source),
                    source: source_var,
                    edge_types: rp.rel_types.clone(),
                    direction: rp.direction,
                    target_variable: String::new(), // filled by next Node
                    target_labels: Vec::new(),
                    length: rp.length,
                    edge_variable: rp.variable.clone(),
                    target_filters: Vec::new(),
                    edge_filters: rp.properties.clone(),
                    temporal_filter: None,
                    path_variable: None,
                });
            }
        }
    }

    // Post-process: fill in target info from the pattern structure.
    // Walk the elements in pairs: (Node, Rel, Node, Rel, Node, ...)
    // The first Node is the scan, each (Rel, Node) pair is a Traverse.
    let mut result = build_pattern_chain(&pattern.elements, None)?;

    // Named path on a single-relationship pattern (`p = (a)-[:R*]->(b)` or a
    // one-hop `p = (a)-[:R]->(b)`): bind the route on the lone Traverse so the
    // executor reconstructs it. Multi-relationship linear named paths are not
    // yet projected (the path would have to span multiple Traverse ops).
    if let Some(pv) = &pattern.path_variable {
        if pattern.elements.len() == 3 {
            if let LogicalOp::Traverse { path_variable, .. } = &mut result {
                *path_variable = Some(pv.clone());
            }
        }
    }

    Ok(result)
}

/// Build a chain of NodeScan → Traverse → Traverse from pattern elements.
///
/// When `base` is `Some`, the pattern's first node is already bound by a prior
/// clause: the chain traverses out of that binding instead of opening a fresh
/// `NodeScan`, so `MATCH (p) ... MATCH (p)-[:R]->(f)` continues from the bound
/// `p` rather than re-scanning every node (which would drop the prior filter).
fn build_pattern_chain(
    elements: &[PatternElement],
    base: Option<LogicalOp>,
) -> Result<LogicalOp, PlanError> {
    if elements.is_empty() {
        return Err(PlanError::EmptyPattern);
    }

    // First element must be a Node
    let first_node = match &elements[0] {
        PatternElement::Node(np) => np,
        _ => return Err(PlanError::UnsupportedPattern),
    };

    let mut current = match base {
        Some(input) => input,
        None => LogicalOp::NodeScan {
            variable: first_node.variable.clone().unwrap_or_default(),
            labels: first_node.labels.clone(),
            property_filters: first_node.properties.clone(),
        },
    };

    let source_var = first_node.variable.clone().unwrap_or_default();
    let mut last_var = source_var;

    // Process remaining elements in (Relationship, Node) pairs
    let mut i = 1;
    while i < elements.len() {
        let rel = match &elements[i] {
            PatternElement::Relationship(rp) => rp,
            PatternElement::Node(_) => {
                // Two nodes in a row without a relationship — skip
                i += 1;
                continue;
            }
        };

        let target_node = if i + 1 < elements.len() {
            match &elements[i + 1] {
                PatternElement::Node(np) => np,
                _ => return Err(PlanError::UnsupportedPattern),
            }
        } else {
            // Relationship without target node
            return Err(PlanError::UnsupportedPattern);
        };

        let target_var = target_node.variable.clone().unwrap_or_default();

        current = LogicalOp::Traverse {
            input: Box::new(current),
            source: last_var.clone(),
            edge_types: rel.rel_types.clone(),
            direction: rel.direction,
            target_variable: target_var.clone(),
            target_labels: target_node.labels.clone(),
            length: rel.length,
            edge_variable: rel.variable.clone(),
            target_filters: target_node.properties.clone(),
            edge_filters: rel.properties.clone(),
            temporal_filter: None,
            path_variable: None,
        };

        last_var = target_var;
        i += 2;
    }

    Ok(current)
}

/// Build a RETURN clause into Project (or Aggregate + Project).
fn build_return_op(input: LogicalOp, rc: &ReturnClause) -> Result<LogicalOp, PlanError> {
    // Check if any return items contain aggregation functions
    let has_aggregates = rc.items.iter().any(|item| is_aggregate_expr(&item.expr));

    if has_aggregates {
        // Split items into group-by keys and aggregates
        let mut group_by = Vec::new();
        let mut aggregates = Vec::new();
        let mut project_items = Vec::new();

        for item in &rc.items {
            if is_aggregate_expr(&item.expr) {
                if let Expr::FunctionCall {
                    name,
                    args,
                    distinct,
                } = &item.expr
                {
                    let arg = args.first().cloned().unwrap_or(Expr::Star);
                    // Use alias or function name as the column key
                    let agg_col = item.alias.clone().unwrap_or_else(|| name.clone());
                    // For percentileCont/percentileDisc, store the second argument expression.
                    // Storing the raw Expr (not a pre-evaluated float) lets the executor resolve
                    // query parameters ($p) at runtime against the params map in ExecutionContext.
                    let percentile_expr =
                        if matches!(name.as_str(), "percentileCont" | "percentileDisc") {
                            args.get(1).cloned()
                        } else {
                            None
                        };
                    aggregates.push(AggregateItem {
                        function: name.clone(),
                        arg,
                        distinct: *distinct,
                        alias: Some(agg_col.clone()),
                        percentile_expr,
                    });
                    // Project references the pre-computed aggregate column
                    project_items.push(ProjectItem {
                        expr: Expr::Variable(agg_col.clone()),
                        alias: item.alias.clone(),
                    });
                }
            } else {
                group_by.push(item.expr.clone());
                project_items.push(ProjectItem {
                    expr: item.expr.clone(),
                    alias: item.alias.clone(),
                });
            }
        }

        let agg_op = LogicalOp::Aggregate {
            input: Box::new(input),
            group_by,
            aggregates,
        };

        Ok(LogicalOp::Project {
            input: Box::new(agg_op),
            items: project_items,
            distinct: rc.distinct,
        })
    } else {
        let items: Vec<ProjectItem> = rc
            .items
            .iter()
            .map(|i| ProjectItem {
                expr: i.expr.clone(),
                alias: i.alias.clone(),
            })
            .collect();

        Ok(LogicalOp::Project {
            input: Box::new(input),
            items,
            distinct: rc.distinct,
        })
    }
}

/// Build a WITH clause into Project (scope barrier).
fn build_with_op(input: LogicalOp, wc: &WithClause) -> Result<LogicalOp, PlanError> {
    let has_aggregates = wc.items.iter().any(|item| is_aggregate_expr(&item.expr));

    let mut result = if has_aggregates {
        let mut group_by = Vec::new();
        let mut aggregates = Vec::new();
        let mut project_items = Vec::new();

        for item in &wc.items {
            if is_aggregate_expr(&item.expr) {
                if let Expr::FunctionCall {
                    name,
                    args,
                    distinct,
                } = &item.expr
                {
                    let agg_col = item.alias.clone().unwrap_or_else(|| name.clone());
                    let arg = args.first().cloned().unwrap_or(Expr::Star);
                    let percentile_expr =
                        if matches!(name.as_str(), "percentileCont" | "percentileDisc") {
                            args.get(1).cloned()
                        } else {
                            None
                        };
                    aggregates.push(AggregateItem {
                        function: name.clone(),
                        arg,
                        distinct: *distinct,
                        alias: Some(agg_col.clone()),
                        percentile_expr,
                    });
                    project_items.push(ProjectItem {
                        expr: Expr::Variable(agg_col.clone()),
                        alias: item.alias.clone(),
                    });
                }
            } else {
                group_by.push(item.expr.clone());
                project_items.push(ProjectItem {
                    expr: item.expr.clone(),
                    alias: item.alias.clone(),
                });
            }
        }

        let agg_op = LogicalOp::Aggregate {
            input: Box::new(input),
            group_by,
            aggregates,
        };

        LogicalOp::Project {
            input: Box::new(agg_op),
            items: project_items,
            distinct: wc.distinct,
        }
    } else {
        let items: Vec<ProjectItem> = wc
            .items
            .iter()
            .map(|i| ProjectItem {
                expr: i.expr.clone(),
                alias: i.alias.clone(),
            })
            .collect();

        LogicalOp::Project {
            input: Box::new(input),
            items,
            distinct: wc.distinct,
        }
    };

    // Apply WHERE filter after WITH
    if let Some(ref pred) = wc.where_clause {
        result = LogicalOp::Filter {
            input: Box::new(result),
            predicate: pred.clone(),
        };
    }

    Ok(result)
}

/// Build CREATE operations.
fn build_create_op(current: Option<LogicalOp>, cc: &CreateClause) -> Result<LogicalOp, PlanError> {
    let mut result = current.unwrap_or(LogicalOp::Empty);

    for pattern in &cc.patterns {
        let elements = &pattern.elements;

        // Two-pass: first create all nodes, then all edges.
        // This ensures that both source and target nodes exist in the row
        // before CreateEdge tries to read their IDs.

        // Pass 1: create nodes
        for element in elements {
            if let PatternElement::Node(np) = element {
                let has_content = !np.labels.is_empty() || !np.properties.is_empty();
                if has_content {
                    result = LogicalOp::CreateNode {
                        input: Some(Box::new(result)),
                        variable: np.variable.clone(),
                        labels: np.labels.clone(),
                        properties: np.properties.clone(),
                    };
                }
            }
        }

        // Pass 2: create edges (nodes are now in scope with their IDs)
        for (i, element) in elements.iter().enumerate() {
            if let PatternElement::Relationship(rp) = element {
                let source_var = if i > 0 {
                    match &elements[i - 1] {
                        PatternElement::Node(np) => np.variable.clone().unwrap_or_default(),
                        _ => String::new(),
                    }
                } else {
                    String::new()
                };

                let target_var = if i + 1 < elements.len() {
                    match &elements[i + 1] {
                        PatternElement::Node(np) => np.variable.clone().unwrap_or_default(),
                        _ => String::new(),
                    }
                } else {
                    String::new()
                };

                // Swap source/target for incoming direction
                let (src, tgt) = match rp.direction {
                    Direction::Incoming => (target_var, source_var),
                    _ => (source_var, target_var),
                };

                result = LogicalOp::CreateEdge {
                    input: Box::new(result),
                    source: src,
                    target: tgt,
                    edge_type: rp.rel_types.first().cloned().unwrap_or_default(),
                    direction: rp.direction,
                    variable: rp.variable.clone(),
                    properties: rp.properties.clone(),
                };
            }
        }
    }

    Ok(result)
}

/// Check if an expression is an aggregation function call.
fn is_aggregate_expr(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::FunctionCall { name, .. }
            if matches!(name.as_str(),
                "count" | "sum" | "avg" | "min" | "max" | "collect"
                | "percentileCont" | "percentileDisc" | "stDev" | "stDevP"
            )
    )
}

/// Try to extract a VectorFilter from a WHERE predicate.
///
/// Detects patterns like:
///   `vector_distance(n.embedding, [1.0, 0.0]) < 0.5`
///   `vector_similarity(n.embedding, $query) > 0.8`
///
/// Returns Some(VectorFilter) if the predicate matches, None otherwise.
fn try_extract_vector_filter(expr: &Expr, input: LogicalOp) -> Option<LogicalOp> {
    if let Expr::BinaryOp { left, op, right } = expr {
        // Check: vector_fn(a, b) < threshold  or  vector_fn(a, b) > threshold
        let (fn_expr, threshold_expr, less_than) = match op {
            BinaryOperator::Lt | BinaryOperator::Lte => (left.as_ref(), right.as_ref(), true),
            BinaryOperator::Gt | BinaryOperator::Gte => (left.as_ref(), right.as_ref(), false),
            _ => return None,
        };

        if let Expr::FunctionCall { name, args, .. } = fn_expr {
            let is_vector_fn = matches!(
                name.as_str(),
                "vector_distance" | "vector_similarity" | "vector_dot" | "vector_manhattan"
            );
            if !is_vector_fn || args.len() != 2 {
                return None;
            }

            // Extract threshold as f64 literal
            let threshold = match threshold_expr {
                Expr::Literal(Value::Float(f)) => *f,
                Expr::Literal(Value::Int(i)) => *i as f64,
                _ => return None, // non-literal threshold → keep as Filter
            };

            // VectorFilter is optimized for constant query vectors (HNSW index scan).
            // When both args are row property references (variable-vs-variable comparison),
            // fall through to generic Filter which correctly evaluates both from the row.
            let query_is_constant = matches!(
                &args[1],
                Expr::Literal(_) | Expr::Parameter(_) | Expr::List(_)
            );
            if !query_is_constant {
                return None; // variable-vs-variable → generic Filter
            }

            return Some(LogicalOp::VectorFilter {
                input: Box::new(input),
                vector_expr: args[0].clone(),
                query_vector: args[1].clone(),
                function: name.clone(),
                less_than,
                threshold,
                decay_field: None,
                push_down: None,
            });
        }

        // Detect decay-weighted vector pattern:
        //   vector_fn(a, b) * decay_expr > threshold
        //   decay_expr * vector_fn(a, b) > threshold
        if let Expr::BinaryOp {
            left: mul_left,
            op: BinaryOperator::Mul,
            right: mul_right,
        } = fn_expr
        {
            // Try both orderings: vector_fn * decay or decay * vector_fn
            let (vec_fn, decay_expr) = if matches!(mul_left.as_ref(), Expr::FunctionCall { .. }) {
                (mul_left.as_ref(), mul_right.as_ref())
            } else if matches!(mul_right.as_ref(), Expr::FunctionCall { .. }) {
                (mul_right.as_ref(), mul_left.as_ref())
            } else {
                return None;
            };

            // decay_expr must be a property reference (e.g., n._recency), not a literal.
            // Reject literal multipliers like `vector_similarity(...) * 0.5` — those
            // should stay in generic Filter, not be treated as decay fields.
            let is_property_ref =
                matches!(decay_expr, Expr::PropertyAccess { .. } | Expr::Variable(_));
            if !is_property_ref {
                return None;
            }

            if let Expr::FunctionCall { name, args, .. } = vec_fn {
                let is_vector_fn = matches!(
                    name.as_str(),
                    "vector_distance" | "vector_similarity" | "vector_dot" | "vector_manhattan"
                );
                if !is_vector_fn || args.len() != 2 {
                    return None;
                }

                let threshold = match threshold_expr {
                    Expr::Literal(Value::Float(f)) => *f,
                    Expr::Literal(Value::Int(i)) => *i as f64,
                    _ => return None,
                };

                let query_is_constant = matches!(
                    &args[1],
                    Expr::Literal(_) | Expr::Parameter(_) | Expr::List(_)
                );
                if !query_is_constant {
                    return None;
                }

                return Some(LogicalOp::VectorFilter {
                    input: Box::new(input),
                    vector_expr: args[0].clone(),
                    query_vector: args[1].clone(),
                    function: name.clone(),
                    less_than,
                    threshold,
                    decay_field: Some(decay_expr.clone()),
                    push_down: None,
                });
            }
        }
    }
    None
}

/// Find the last variable name from a logical operator (for traversal chaining).
/// Try to extract a TextFilter from a WHERE predicate.
///
/// Detects: `text_match(field_expr, "query string")`
/// Split a compound WHERE predicate into a pipeline of specialized operators.
///
/// Decomposes AND-connected predicates into:
///   VectorFilter(s) → TextFilter(s) → generic Filter (remaining property predicates)
///
/// Example: `vector_distance(n.e, $q) < 0.3 AND text_match(n.body, "hello") AND n.age > 25`
///   → VectorFilter(input, n.e, $q, <0.3)
///     → TextFilter(_, n.body, "hello")
///       → Filter(_, n.age > 25)
fn apply_compound_where(predicate: &Expr, input: LogicalOp) -> LogicalOp {
    // Step 1: Flatten AND into conjuncts
    let conjuncts = flatten_and(predicate);

    let mut vector_conjuncts = Vec::new();
    let mut text_conjuncts = Vec::new();
    let mut encrypted_conjuncts = Vec::new();
    let mut remaining_conjuncts = Vec::new();

    // Step 2: Classify each conjunct
    for conj in &conjuncts {
        if try_extract_vector_filter(conj, LogicalOp::Empty).is_some() {
            vector_conjuncts.push(conj.clone());
        } else if try_extract_text_filter(conj, LogicalOp::Empty).is_some() {
            text_conjuncts.push(conj.clone());
        } else if try_extract_encrypted_filter(conj, LogicalOp::Empty).is_some() {
            encrypted_conjuncts.push(conj.clone());
        } else {
            remaining_conjuncts.push(conj.clone());
        }
    }

    // If nothing was split (single predicate or no specialized), fast path
    if vector_conjuncts.is_empty() && text_conjuncts.is_empty() && encrypted_conjuncts.is_empty() {
        // No specialized predicates found — try the original extraction on full predicate
        // (handles cases where the top-level expression IS a vector/text/encrypted call, not AND)
        if let Some(vf) = try_extract_vector_filter(predicate, input.clone()) {
            return vf;
        }
        if let Some(tf) = try_extract_text_filter(predicate, input.clone()) {
            return tf;
        }
        if let Some(ef) = try_extract_encrypted_filter(predicate, input.clone()) {
            return ef;
        }
        return LogicalOp::Filter {
            input: Box::new(input),
            predicate: predicate.clone(),
        };
    }

    // Step 3: Build pipeline — vector filters first (most selective typically)
    let mut result = input;
    for vf_expr in &vector_conjuncts {
        if let Some(vf) = try_extract_vector_filter(vf_expr, result.clone()) {
            result = vf;
        }
    }

    // Text filters next
    for tf_expr in &text_conjuncts {
        if let Some(tf) = try_extract_text_filter(tf_expr, result.clone()) {
            result = tf;
        }
    }

    // Encrypted filters next (SSE equality lookups — very cheap hash lookup)
    for ef_expr in &encrypted_conjuncts {
        if let Some(ef) = try_extract_encrypted_filter(ef_expr, result.clone()) {
            result = ef;
        }
    }

    // Remaining predicates as generic Filter (combined with AND)
    if !remaining_conjuncts.is_empty() {
        let combined = conjoin_predicates(remaining_conjuncts);
        result = LogicalOp::Filter {
            input: Box::new(result),
            predicate: combined,
        };
    }

    result
}

/// Collect variables introduced by a list of MATCH patterns.
///
/// Used by G024 predicate lifting to determine which variables
/// a MATCH branch introduces vs which come from prior clauses.
fn collect_pattern_variables(patterns: &[Pattern]) -> Vec<String> {
    let mut vars = Vec::new();
    for pattern in patterns {
        for elem in &pattern.elements {
            match elem {
                PatternElement::Node(np) => {
                    if let Some(v) = &np.variable {
                        vars.push(v.clone());
                    }
                }
                PatternElement::Relationship(rp) => {
                    if let Some(v) = &rp.variable {
                        vars.push(v.clone());
                    }
                }
            }
        }
    }
    vars
}

/// Collect variables introduced by a logical operator subtree.
///
/// Mirrors `collect_introduced_variables` in runner.rs but at the
/// planner level (operates on LogicalOp, not during execution).
fn collect_op_variables(op: &LogicalOp) -> Vec<String> {
    match op {
        LogicalOp::NodeScan { variable, .. } => vec![variable.clone()],
        LogicalOp::Traverse {
            input,
            target_variable,
            edge_variable,
            ..
        } => {
            let mut vars = collect_op_variables(input);
            vars.push(target_variable.clone());
            if let Some(ev) = edge_variable {
                vars.push(ev.clone());
            }
            vars
        }
        // Project (WITH / RETURN) is a scope barrier: only the projected
        // columns survive. Return the OUTPUT variable names (alias, or the bare
        // variable for `WITH a`); `*` passes the input scope through. This is
        // what lets a correlated subquery's leading `WITH a` (whose input is an
        // injected Empty leaf) expose `a` for a following bound-continuation
        // MATCH.
        LogicalOp::Project { input, items, .. } => {
            let mut vars = Vec::new();
            for item in items {
                match &item.alias {
                    Some(a) => vars.push(a.clone()),
                    None => match &item.expr {
                        Expr::Variable(v) => vars.push(v.clone()),
                        Expr::Star => vars.extend(collect_op_variables(input)),
                        _ => {}
                    },
                }
            }
            vars
        }
        LogicalOp::Filter { input, .. }
        | LogicalOp::VectorFilter { input, .. }
        | LogicalOp::TextFilter { input, .. }
        | LogicalOp::EncryptedFilter { input, .. }
        | LogicalOp::Aggregate { input, .. }
        | LogicalOp::Sort { input, .. }
        | LogicalOp::Limit { input, .. }
        | LogicalOp::Skip { input, .. } => collect_op_variables(input),
        LogicalOp::CartesianProduct { left, right } | LogicalOp::LeftOuterJoin { left, right } => {
            let mut vars = collect_op_variables(left);
            vars.extend(collect_op_variables(right));
            vars
        }
        LogicalOp::Unwind {
            input, variable, ..
        } => {
            let mut vars = collect_op_variables(input);
            vars.push(variable.clone());
            vars
        }
        _ => Vec::new(),
    }
}

/// Collect all variable names referenced in an expression.
///
/// Walks the expression tree and returns every `Variable(name)` and
/// the base variable from `PropertyAccess { expr: Variable(name), .. }`.
fn collect_expr_variables(expr: &Expr) -> Vec<String> {
    let mut vars = Vec::new();
    collect_expr_variables_inner(expr, &mut vars);
    vars
}

fn collect_expr_variables_inner(expr: &Expr, vars: &mut Vec<String>) {
    match expr {
        Expr::Variable(name) => vars.push(name.clone()),
        Expr::PropertyAccess { expr, .. } => collect_expr_variables_inner(expr, vars),
        Expr::BinaryOp { left, right, .. } => {
            collect_expr_variables_inner(left, vars);
            collect_expr_variables_inner(right, vars);
        }
        Expr::UnaryOp { expr, .. } => collect_expr_variables_inner(expr, vars),
        Expr::FunctionCall { args, .. } => {
            for arg in args {
                collect_expr_variables_inner(arg, vars);
            }
        }
        Expr::List(items) => {
            for item in items {
                collect_expr_variables_inner(item, vars);
            }
        }
        Expr::MapProjection { expr, items } => {
            collect_expr_variables_inner(expr, vars);
            for item in items {
                if let MapProjectionItem::Computed(_, e) = item {
                    collect_expr_variables_inner(e, vars);
                }
            }
        }
        Expr::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            if let Some(op) = operand {
                collect_expr_variables_inner(op, vars);
            }
            for (cond, result) in when_clauses {
                collect_expr_variables_inner(cond, vars);
                collect_expr_variables_inner(result, vars);
            }
            if let Some(el) = else_clause {
                collect_expr_variables_inner(el, vars);
            }
        }
        Expr::In { expr, list } => {
            collect_expr_variables_inner(expr, vars);
            collect_expr_variables_inner(list, vars);
        }
        Expr::IsNull { expr, .. } => collect_expr_variables_inner(expr, vars),
        Expr::IsTyped { expr, .. } => collect_expr_variables_inner(expr, vars),
        Expr::StringMatch { expr, pattern, .. } => {
            collect_expr_variables_inner(expr, vars);
            collect_expr_variables_inner(pattern, vars);
        }
        Expr::MapLiteral(entries) => {
            for (_, e) in entries {
                collect_expr_variables_inner(e, vars);
            }
        }
        Expr::PatternPredicate(pattern) => {
            for elem in &pattern.elements {
                match elem {
                    PatternElement::Node(node) => {
                        if let Some(ref name) = node.variable {
                            vars.push(name.clone());
                        }
                        for (_, v) in &node.properties {
                            collect_expr_variables_inner(v, vars);
                        }
                    }
                    PatternElement::Relationship(rel) => {
                        if let Some(ref name) = rel.variable {
                            vars.push(name.clone());
                        }
                        for (_, v) in &rel.properties {
                            collect_expr_variables_inner(v, vars);
                        }
                    }
                }
            }
        }
        Expr::Subscript { expr, index } => {
            collect_expr_variables_inner(expr, vars);
            collect_expr_variables_inner(index, vars);
        }
        Expr::Slice { expr, start, end } => {
            collect_expr_variables_inner(expr, vars);
            if let Some(s) = start {
                collect_expr_variables_inner(s, vars);
            }
            if let Some(e) = end {
                collect_expr_variables_inner(e, vars);
            }
        }
        Expr::Reduce {
            acc,
            init,
            var,
            list,
            expr,
        } => {
            collect_expr_variables_inner(init, vars);
            collect_expr_variables_inner(list, vars);
            // acc / var are bound locally inside the fold; drop them from the
            // outer variable dependencies of the step expression.
            let mut inner = Vec::new();
            collect_expr_variables_inner(expr, &mut inner);
            vars.extend(inner.into_iter().filter(|v| v != acc && v != var));
        }
        Expr::ListPredicate {
            var, list, pred, ..
        } => {
            collect_expr_variables_inner(list, vars);
            let mut inner = Vec::new();
            collect_expr_variables_inner(pred, &mut inner);
            vars.extend(inner.into_iter().filter(|v| v != var));
        }
        Expr::ListComprehension {
            var,
            list,
            pred,
            map,
        } => {
            collect_expr_variables_inner(list, vars);
            let mut inner = Vec::new();
            if let Some(p) = pred {
                collect_expr_variables_inner(p, &mut inner);
            }
            if let Some(m) = map {
                collect_expr_variables_inner(m, &mut inner);
            }
            vars.extend(inner.into_iter().filter(|v| v != var));
        }
        // Inner MATCH binds its own scope; outer-correlation vars are
        // provisioned by the outer clauses — no extra deps contributed here.
        Expr::ExistsSubquery(_)
        | Expr::CountSubquery(_)
        | Expr::CollectSubquery { .. }
        | Expr::PatternComprehension { .. } => {}
        Expr::Literal(_) | Expr::Parameter(_) | Expr::Star => {}
    }
}

/// Flatten nested AND expressions into a flat list of conjuncts.
/// `a AND b AND c` → `[a, b, c]`
fn flatten_and(expr: &Expr) -> Vec<Expr> {
    match expr {
        Expr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
        } => {
            let mut result = flatten_and(left);
            result.extend(flatten_and(right));
            result
        }
        other => vec![other.clone()],
    }
}

/// Combine a list of predicates into a single AND expression.
/// Caller guarantees at least one predicate (only called when `!remaining.is_empty()`).
fn conjoin_predicates(mut predicates: Vec<Expr>) -> Expr {
    // Safe: called only from apply_compound_where when predicates is non-empty.
    let first = predicates.remove(0);
    predicates
        .into_iter()
        .fold(first, |acc, pred| Expr::BinaryOp {
            left: Box::new(acc),
            op: BinaryOperator::And,
            right: Box::new(pred),
        })
}

fn try_extract_text_filter(expr: &Expr, input: LogicalOp) -> Option<LogicalOp> {
    if let Expr::FunctionCall { name, args, .. } = expr {
        if name == "text_match" && (args.len() == 2 || args.len() == 3) {
            // Second arg must be a string literal (query string)
            if let Expr::Literal(Value::String(query_str)) = &args[1] {
                // Optional third arg: language string literal
                let language = if args.len() == 3 {
                    if let Expr::Literal(Value::String(lang)) = &args[2] {
                        Some(lang.clone())
                    } else {
                        return None; // 3rd arg must be string literal
                    }
                } else {
                    None
                };
                return Some(LogicalOp::TextFilter {
                    input: Box::new(input),
                    text_expr: args[0].clone(),
                    query_string: query_str.clone(),
                    language,
                });
            }
        }
    }
    None
}

fn try_extract_encrypted_filter(expr: &Expr, input: LogicalOp) -> Option<LogicalOp> {
    if let Expr::FunctionCall { name, args, .. } = expr {
        if name == "encrypted_match" && args.len() == 2 {
            return Some(LogicalOp::EncryptedFilter {
                input: Box::new(input),
                field_expr: args[0].clone(),
                token_expr: args[1].clone(),
            });
        }
    }
    None
}

fn find_last_variable(op: &LogicalOp) -> String {
    match op {
        LogicalOp::NodeScan { variable, .. } => variable.clone(),
        LogicalOp::Traverse {
            target_variable, ..
        } => target_variable.clone(),
        LogicalOp::CartesianProduct { right, .. } => find_last_variable(right),
        LogicalOp::Filter { input, .. }
        | LogicalOp::Project { input, .. }
        | LogicalOp::Sort { input, .. }
        | LogicalOp::Limit { input, .. }
        | LogicalOp::Skip { input, .. } => find_last_variable(input),
        _ => String::new(),
    }
}

// =============================================================================
// R-HYB2b: rrf_score planner post-pass
// =============================================================================

/// Signature captured from the first `rrf_score(...)` / `cc_score(...)` /
/// `dbsf_score(...)` call-site encountered. Multiple call-sites must all
/// produce identical signatures (same method list, same query parts, same
/// fusion variant).
#[derive(Debug, Clone, PartialEq)]
struct RrfCallSig {
    methods: Vec<Expr>,
    query_vector: Option<Expr>,
    query_text: Option<Expr>,
    fusion: crate::planner::logical::FusionStrategy,
}

/// Tags the three fusion scalars by name so the parser can dispatch the
/// correct argument shape (RRF: 2 args, CC/DBSF: 3 args including weights).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FusionScalar {
    Rrf,
    Cc,
    Dbsf,
}

impl FusionScalar {
    fn fn_name(self) -> &'static str {
        match self {
            Self::Rrf => "rrf_score",
            Self::Cc => "cc_score",
            Self::Dbsf => "dbsf_score",
        }
    }

    fn from_name(name: &str) -> Option<Self> {
        match name {
            "rrf_score" => Some(Self::Rrf),
            "cc_score" => Some(Self::Cc),
            "dbsf_score" => Some(Self::Dbsf),
            _ => None,
        }
    }
}

/// Recognise a top-level fusion-scalar FunctionCall. Returns the scalar
/// kind + its argument slice so the parser can validate arity per kind.
fn match_fusion_call(expr: &Expr) -> Option<(FusionScalar, &[Expr])> {
    if let Expr::FunctionCall { name, args, .. } = expr {
        if let Some(kind) = FusionScalar::from_name(name) {
            return Some((kind, args));
        }
    }
    None
}

/// Backwards-compat shim: the rest of the RRF-era code paths still call
/// this name with the implicit `rrf_score` assumption. Delegates to the
/// generalised matcher and forwards only the args for `rrf_score`.
fn match_rrf_score(expr: &Expr) -> Option<&[Expr]> {
    match match_fusion_call(expr) {
        Some((FusionScalar::Rrf, args)) => Some(args),
        _ => None,
    }
}

/// Walk an expression tree and return `true` if any subexpression is a
/// `rrf_score(...)` call. Used to fail-fast on illegal placements (e.g. WHERE).
fn expr_contains_rrf_score(expr: &Expr) -> bool {
    let mut hit = false;
    collect_rrf_presence(expr, &mut hit);
    hit
}

fn collect_rrf_presence(expr: &Expr, hit: &mut bool) {
    if *hit {
        return;
    }
    match expr {
        Expr::FunctionCall { name, args, .. } => {
            if FusionScalar::from_name(name).is_some() {
                *hit = true;
                return;
            }
            for a in args {
                collect_rrf_presence(a, hit);
            }
        }
        Expr::BinaryOp { left, right, .. } => {
            collect_rrf_presence(left, hit);
            collect_rrf_presence(right, hit);
        }
        Expr::UnaryOp { expr, .. } => collect_rrf_presence(expr, hit),
        Expr::PropertyAccess { expr, .. } => collect_rrf_presence(expr, hit),
        Expr::List(items) => {
            for it in items {
                collect_rrf_presence(it, hit);
            }
        }
        Expr::MapLiteral(fields) => {
            for (_, v) in fields {
                collect_rrf_presence(v, hit);
            }
        }
        Expr::In { expr, list } => {
            collect_rrf_presence(expr, hit);
            collect_rrf_presence(list, hit);
        }
        Expr::IsNull { expr, .. } => collect_rrf_presence(expr, hit),
        Expr::StringMatch { expr, pattern, .. } => {
            collect_rrf_presence(expr, hit);
            collect_rrf_presence(pattern, hit);
        }
        Expr::Subscript { expr, index } => {
            collect_rrf_presence(expr, hit);
            collect_rrf_presence(index, hit);
        }
        Expr::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            if let Some(o) = operand.as_deref() {
                collect_rrf_presence(o, hit);
            }
            for (c, v) in when_clauses {
                collect_rrf_presence(c, hit);
                collect_rrf_presence(v, hit);
            }
            if let Some(e) = else_clause.as_deref() {
                collect_rrf_presence(e, hit);
            }
        }
        _ => {}
    }
}

/// Parse and validate a fusion-scalar call. RRF takes two args
/// (methods + query map); CC and DBSF take three (methods + query map +
/// weights map). All three reject empty method lists and unknown query
/// map keys; CC and DBSF additionally require that every weight be a
/// finite non-negative number, and that the weights map name only the
/// supported categories (`vector` / `text`).
fn parse_fusion_signature(kind: FusionScalar, args: &[Expr]) -> Result<RrfCallSig, PlanError> {
    let expected_arity = match kind {
        FusionScalar::Rrf => 2,
        FusionScalar::Cc | FusionScalar::Dbsf => 3,
    };
    if args.len() != expected_arity {
        return Err(PlanError::RrfScoreArity);
    }
    let methods_expr = &args[0];
    let query_expr = &args[1];

    // Methods: Expr::List of at least one element.
    let methods = match methods_expr {
        Expr::List(items) if !items.is_empty() => items.clone(),
        Expr::List(_) => {
            return Err(PlanError::RrfScoreMethodsShape {
                got: "empty list".to_string(),
            });
        }
        other => {
            return Err(PlanError::RrfScoreMethodsShape {
                got: format!("{other:?}"),
            });
        }
    };

    // Query: MapLiteral (post-substitute, parameters resolved to Literal(Map))
    // or Literal(Value::Map). Extract `vector` and `text` keys.
    let (query_vector, query_text) = match query_expr {
        Expr::MapLiteral(fields) => {
            let mut qv = None;
            let mut qt = None;
            for (k, v) in fields {
                match k.as_str() {
                    "vector" => qv = Some(v.clone()),
                    "text" => qt = Some(v.clone()),
                    other => {
                        return Err(PlanError::RrfScoreQueryShape {
                            got: format!(
                                "map with unknown key `{other}` (expected `vector` and/or `text`)"
                            ),
                        });
                    }
                }
            }
            if qv.is_none() && qt.is_none() {
                return Err(PlanError::RrfScoreQueryShape {
                    got: "empty map (needs at least one of `vector`, `text`)".to_string(),
                });
            }
            (qv, qt)
        }
        // Parameter / Variable: defer shape validation to executor.
        // The executor will evaluate and reject non-map values at runtime.
        Expr::Parameter(_) | Expr::Variable(_) => {
            (Some(query_expr.clone()), Some(query_expr.clone()))
        }
        other => {
            return Err(PlanError::RrfScoreQueryShape {
                got: format!("{other:?}"),
            });
        }
    };

    let fusion = match kind {
        FusionScalar::Rrf => crate::planner::logical::FusionStrategy::Rrf { k: 60 },
        FusionScalar::Cc => {
            let weights = parse_fusion_weights(&args[2], kind)?;
            crate::planner::logical::FusionStrategy::ConvexCombination { weights }
        }
        FusionScalar::Dbsf => {
            let weights = parse_fusion_weights(&args[2], kind)?;
            crate::planner::logical::FusionStrategy::Dbsf { weights }
        }
    };

    Ok(RrfCallSig {
        methods,
        query_vector,
        query_text,
        fusion,
    })
}

/// Parse the third (weights) argument of `cc_score` / `dbsf_score`. Must
/// be a `MapLiteral` over the supported categories (`vector`, `text`)
/// with finite non-negative numeric values. Empty maps are rejected.
fn parse_fusion_weights(
    expr: &Expr,
    kind: FusionScalar,
) -> Result<std::collections::BTreeMap<String, f64>, PlanError> {
    let fields = match expr {
        Expr::MapLiteral(fields) => fields,
        other => {
            return Err(PlanError::RrfScoreQueryShape {
                got: format!(
                    "{}: weights must be a map literal, got {other:?}",
                    kind.fn_name()
                ),
            });
        }
    };
    let mut weights = std::collections::BTreeMap::new();
    for (key, value) in fields {
        let key_str = match key.as_str() {
            "vector" | "text" => key.clone(),
            other => {
                return Err(PlanError::RrfScoreQueryShape {
                    got: format!(
                        "{}: weights has unknown key `{other}` (expected `vector` and/or `text`)",
                        kind.fn_name()
                    ),
                });
            }
        };
        let w = match value {
            Expr::Literal(coordinode_core::graph::types::Value::Float(f)) => *f,
            Expr::Literal(coordinode_core::graph::types::Value::Int(i)) => *i as f64,
            // `-0.1` arrives from the parser as UnaryOp::Neg(Literal(...)); fold it
            // so the non-negative check below catches negative literals cleanly.
            Expr::UnaryOp {
                op: crate::cypher::ast::UnaryOperator::Neg,
                expr: inner,
            } => match inner.as_ref() {
                Expr::Literal(coordinode_core::graph::types::Value::Float(f)) => -*f,
                Expr::Literal(coordinode_core::graph::types::Value::Int(i)) => -(*i as f64),
                _ => {
                    return Err(PlanError::RrfScoreQueryShape {
                        got: format!(
                            "{}: weight for `{key_str}` must be a numeric literal, got {value:?}",
                            kind.fn_name()
                        ),
                    });
                }
            },
            other => {
                return Err(PlanError::RrfScoreQueryShape {
                    got: format!(
                        "{}: weight for `{key_str}` must be a numeric literal, got {other:?}",
                        kind.fn_name()
                    ),
                });
            }
        };
        if !w.is_finite() || w < 0.0 {
            return Err(PlanError::RrfScoreQueryShape {
                got: format!(
                    "{}: weight for `{key_str}` must be a finite non-negative number, got {w}",
                    kind.fn_name()
                ),
            });
        }
        weights.insert(key_str, w);
    }
    if weights.is_empty() {
        return Err(PlanError::RrfScoreQueryShape {
            got: format!(
                "{}: weights map is empty (needs at least one of `vector`, `text`)",
                kind.fn_name()
            ),
        });
    }
    Ok(weights)
}

/// Collect the RRF signature (if any) from an expression. Errors if multiple
/// differing signatures are found within a single expression.
fn collect_rrf_sig_in_expr(
    expr: &Expr,
    found: &mut Option<RrfCallSig>,
    location: &str,
) -> Result<(), PlanError> {
    if let Some((kind, args)) = match_fusion_call(expr) {
        let sig = parse_fusion_signature(kind, args)?;
        match found {
            None => *found = Some(sig),
            Some(existing) if *existing == sig => {}
            Some(_) => {
                return Err(PlanError::RrfScoreMultipleCalls {
                    location: location.to_string(),
                });
            }
        }
        // Do not recurse into the args; nested fusion call inside another
        // fusion call's list is pathological and will hit the arity guard.
        return Ok(());
    }
    match expr {
        Expr::BinaryOp { left, right, .. } => {
            collect_rrf_sig_in_expr(left, found, location)?;
            collect_rrf_sig_in_expr(right, found, location)?;
        }
        Expr::UnaryOp { expr, .. } => collect_rrf_sig_in_expr(expr, found, location)?,
        Expr::PropertyAccess { expr, .. } => collect_rrf_sig_in_expr(expr, found, location)?,
        Expr::List(items) => {
            for it in items {
                collect_rrf_sig_in_expr(it, found, location)?;
            }
        }
        Expr::MapLiteral(fields) => {
            for (_, v) in fields {
                collect_rrf_sig_in_expr(v, found, location)?;
            }
        }
        Expr::FunctionCall { args, .. } => {
            for a in args {
                collect_rrf_sig_in_expr(a, found, location)?;
            }
        }
        Expr::In { expr, list } => {
            collect_rrf_sig_in_expr(expr, found, location)?;
            collect_rrf_sig_in_expr(list, found, location)?;
        }
        Expr::IsNull { expr, .. } => collect_rrf_sig_in_expr(expr, found, location)?,
        Expr::StringMatch { expr, pattern, .. } => {
            collect_rrf_sig_in_expr(expr, found, location)?;
            collect_rrf_sig_in_expr(pattern, found, location)?;
        }
        Expr::Subscript { expr, index } => {
            collect_rrf_sig_in_expr(expr, found, location)?;
            collect_rrf_sig_in_expr(index, found, location)?;
        }
        Expr::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            if let Some(o) = operand.as_deref() {
                collect_rrf_sig_in_expr(o, found, location)?;
            }
            for (c, v) in when_clauses {
                collect_rrf_sig_in_expr(c, found, location)?;
                collect_rrf_sig_in_expr(v, found, location)?;
            }
            if let Some(e) = else_clause.as_deref() {
                collect_rrf_sig_in_expr(e, found, location)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// Substitute every fusion-scalar FunctionCall in `expr` with the
/// appropriate fused-score column variable. `rrf_score` rewrites to the
/// historical `__rrf_score__`; `cc_score` and `dbsf_score` rewrite to the
/// universal `__hybrid_score__` column the executor now emits regardless
/// of strategy. RRF also writes `__hybrid_score__` so callers can sort
/// uniformly when reading the fused result.
fn rewrite_rrf_in_expr(expr: &mut Expr) {
    if let Expr::FunctionCall { name, .. } = expr {
        if let Some(kind) = FusionScalar::from_name(name) {
            let column = match kind {
                FusionScalar::Rrf => "__rrf_score__",
                FusionScalar::Cc | FusionScalar::Dbsf => "__hybrid_score__",
            };
            *expr = Expr::Variable(column.to_string());
            return;
        }
    }
    match expr {
        Expr::BinaryOp { left, right, .. } => {
            rewrite_rrf_in_expr(left);
            rewrite_rrf_in_expr(right);
        }
        Expr::UnaryOp { expr, .. } => rewrite_rrf_in_expr(expr),
        Expr::PropertyAccess { expr, .. } => rewrite_rrf_in_expr(expr),
        Expr::List(items) => {
            for it in items {
                rewrite_rrf_in_expr(it);
            }
        }
        Expr::MapLiteral(fields) => {
            for (_, v) in fields {
                rewrite_rrf_in_expr(v);
            }
        }
        Expr::FunctionCall { args, .. } => {
            for a in args {
                rewrite_rrf_in_expr(a);
            }
        }
        Expr::In { expr, list } => {
            rewrite_rrf_in_expr(expr);
            rewrite_rrf_in_expr(list);
        }
        Expr::IsNull { expr, .. } => rewrite_rrf_in_expr(expr),
        Expr::StringMatch { expr, pattern, .. } => {
            rewrite_rrf_in_expr(expr);
            rewrite_rrf_in_expr(pattern);
        }
        Expr::Subscript { expr, index } => {
            rewrite_rrf_in_expr(expr);
            rewrite_rrf_in_expr(index);
        }
        Expr::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            if let Some(o) = operand.as_deref_mut() {
                rewrite_rrf_in_expr(o);
            }
            for (c, v) in when_clauses {
                rewrite_rrf_in_expr(c);
                rewrite_rrf_in_expr(v);
            }
            if let Some(e) = else_clause.as_deref_mut() {
                rewrite_rrf_in_expr(e);
            }
        }
        _ => {}
    }
}

/// Wrap the input of the topmost Project with a RankFuse operator, rewrite
/// rrf_score call-sites, and (if Sort references `__rrf_score__` directly)
/// inject a pass-through projection item so the score survives into Sort.
///
/// Expected plan shapes after this transform:
/// - `Limit → Sort → Project(rewritten, may include pass-through) → RankFuse → rest`
/// - `Sort → Project(rewritten, with pass-through) → RankFuse → rest`
/// - `Project(rewritten) → RankFuse → rest`
fn rewrite_rrf_score(op: LogicalOp) -> Result<LogicalOp, PlanError> {
    // First pass: detect illegal placements (any RRF reference in Filter/WHERE,
    // Aggregate group_by, Merge ON MATCH, etc.) — anywhere that isn't Project
    // items or Sort items.
    validate_rrf_placement(&op)?;

    // Collect the RRF signature from Project / Sort items only.
    let mut sig: Option<RrfCallSig> = None;
    collect_rrf_sig(&op, &mut sig)?;

    let Some(sig) = sig else {
        // No rrf_score usage — return unchanged.
        return Ok(op);
    };

    // Wrap the topmost Project.input with RankFuse and rewrite items.
    let mut sort_touches_rrf = false;
    let op = wrap_rank_fuse(op, &sig, &mut sort_touches_rrf);

    // If Sort (or any op ABOVE Project) referenced __rrf_score__ directly, the
    // Project must pass __rrf_score__ through.
    let op = if sort_touches_rrf {
        ensure_rrf_passthrough(op)
    } else {
        op
    };

    Ok(op)
}

/// Verify no `rrf_score(...)` appears in an illegal position (WHERE,
/// aggregate args, pattern filters, etc.). Only Project items and Sort items
/// are legal sites.
fn validate_rrf_placement(op: &LogicalOp) -> Result<(), PlanError> {
    match op {
        LogicalOp::Filter { input, predicate } => {
            if expr_contains_rrf_score(predicate) {
                return Err(PlanError::RrfScoreIllegalPosition {
                    location: "WHERE clause".to_string(),
                });
            }
            validate_rrf_placement(input)
        }
        LogicalOp::Aggregate {
            input,
            group_by,
            aggregates,
        } => {
            for e in group_by {
                if expr_contains_rrf_score(e) {
                    return Err(PlanError::RrfScoreIllegalPosition {
                        location: "GROUP BY key".to_string(),
                    });
                }
            }
            for a in aggregates {
                if expr_contains_rrf_score(&a.arg) {
                    return Err(PlanError::RrfScoreIllegalPosition {
                        location: "aggregate function argument".to_string(),
                    });
                }
            }
            validate_rrf_placement(input)
        }
        LogicalOp::Project { input, items, .. } => {
            for it in items {
                // rrf_score is legal here; don't recurse into its args —
                // they're structural (method list + query map), not expressions
                // evaluated in the normal sense.
                if match_rrf_score(&it.expr).is_some() {
                    continue;
                }
                // Any OTHER rrf_score inside a nested expression is legal too
                // (we handle blends like `rrf_score(...) * 0.5 + x`).
            }
            validate_rrf_placement(input)
        }
        LogicalOp::Sort { input, items } => {
            for _it in items {
                // rrf_score legal in ORDER BY.
            }
            validate_rrf_placement(input)
        }
        LogicalOp::Limit { input, count } => {
            if expr_contains_rrf_score(count) {
                return Err(PlanError::RrfScoreIllegalPosition {
                    location: "LIMIT expression".to_string(),
                });
            }
            validate_rrf_placement(input)
        }
        LogicalOp::Skip { input, count } => {
            if expr_contains_rrf_score(count) {
                return Err(PlanError::RrfScoreIllegalPosition {
                    location: "SKIP expression".to_string(),
                });
            }
            validate_rrf_placement(input)
        }
        LogicalOp::Traverse { input, .. }
        | LogicalOp::VectorFilter { input, .. }
        | LogicalOp::VectorTopK { input, .. }
        | LogicalOp::EdgeVectorSearch { input, .. }
        | LogicalOp::TextFilter { input, .. }
        | LogicalOp::EncryptedFilter { input, .. }
        | LogicalOp::Unwind { input, .. }
        | LogicalOp::ShortestPath { input, .. }
        | LogicalOp::Update { input, .. }
        | LogicalOp::RemoveOp { input, .. }
        | LogicalOp::Delete { input, .. }
        | LogicalOp::DetachDocument { input, .. }
        | LogicalOp::AttachDocument { input, .. } => validate_rrf_placement(input),
        LogicalOp::CartesianProduct { left, right } | LogicalOp::LeftOuterJoin { left, right } => {
            validate_rrf_placement(left)?;
            validate_rrf_placement(right)
        }
        LogicalOp::CreateNode { input, .. } => {
            if let Some(inp) = input {
                validate_rrf_placement(inp)?;
            }
            Ok(())
        }
        LogicalOp::CreateEdge { input, .. } => validate_rrf_placement(input),
        LogicalOp::Merge { pattern, .. } | LogicalOp::Upsert { pattern, .. } => {
            validate_rrf_placement(pattern)
        }
        LogicalOp::RankFuse { input, .. } => validate_rrf_placement(input),
        _ => Ok(()),
    }
}

/// Traverse Project / Sort operators and collect the (single) RRF signature.
fn collect_rrf_sig(op: &LogicalOp, sig: &mut Option<RrfCallSig>) -> Result<(), PlanError> {
    match op {
        LogicalOp::Project { input, items, .. } => {
            for it in items {
                collect_rrf_sig_in_expr(&it.expr, sig, "RETURN")?;
            }
            collect_rrf_sig(input, sig)
        }
        LogicalOp::Sort { input, items } => {
            for it in items {
                collect_rrf_sig_in_expr(&it.expr, sig, "ORDER BY")?;
            }
            collect_rrf_sig(input, sig)
        }
        LogicalOp::Limit { input, .. } | LogicalOp::Skip { input, .. } => {
            collect_rrf_sig(input, sig)
        }
        _ => Ok(()),
    }
}

/// Walk to the innermost Project and wrap its input with RankFuse. Rewrite all
/// `rrf_score(...)` usages to `Variable("__rrf_score__")` in Project / Sort
/// items along the way. Sets `sort_touches_rrf` if a Sort item directly
/// references rrf_score (which, after rewrite, becomes a Variable ref that
/// Project must preserve).
fn wrap_rank_fuse(op: LogicalOp, sig: &RrfCallSig, sort_touches_rrf: &mut bool) -> LogicalOp {
    match op {
        LogicalOp::Project {
            input,
            mut items,
            distinct,
        } => {
            // Rewrite items first.
            for it in items.iter_mut() {
                rewrite_rrf_in_expr(&mut it.expr);
            }
            // Project.input is where we place RankFuse. Don't recurse further
            // into Project.input — RankFuse wraps whatever match/filter/traverse
            // subtree sits below Project.
            let fused_input = LogicalOp::RankFuse {
                input,
                methods: sig.methods.clone(),
                query_vector: sig.query_vector.clone(),
                query_text: sig.query_text.clone(),
                shard_overfetch_cap: None,
                fusion: sig.fusion.clone(),
            };
            LogicalOp::Project {
                input: Box::new(fused_input),
                items,
                distinct,
            }
        }
        LogicalOp::Sort { input, mut items } => {
            for it in items.iter_mut() {
                if expr_contains_rrf_score(&it.expr) {
                    *sort_touches_rrf = true;
                }
                rewrite_rrf_in_expr(&mut it.expr);
            }
            LogicalOp::Sort {
                input: Box::new(wrap_rank_fuse(*input, sig, sort_touches_rrf)),
                items,
            }
        }
        LogicalOp::Limit { input, count } => LogicalOp::Limit {
            input: Box::new(wrap_rank_fuse(*input, sig, sort_touches_rrf)),
            count,
        },
        LogicalOp::Skip { input, count } => LogicalOp::Skip {
            input: Box::new(wrap_rank_fuse(*input, sig, sort_touches_rrf)),
            count,
        },
        // No Project found along the Sort/Limit/Skip spine — just return as-is.
        // (A plan with Sort referencing rrf_score() but NO Project is ill-formed,
        // but we let the executor guard handle it.)
        other => other,
    }
}

/// If Project's items don't already include `__rrf_score__`, inject a
/// pass-through so Sort (above Project) can read it via `Variable("__rrf_score__")`.
fn ensure_rrf_passthrough(op: LogicalOp) -> LogicalOp {
    match op {
        LogicalOp::Project {
            input,
            mut items,
            distinct,
        } => {
            let already_present = items.iter().any(|it| {
                it.alias.as_deref() == Some("__rrf_score__")
                    || matches!(&it.expr, Expr::Variable(v) if v == "__rrf_score__")
            });
            if !already_present {
                items.push(ProjectItem {
                    expr: Expr::Variable("__rrf_score__".to_string()),
                    alias: Some("__rrf_score__".to_string()),
                });
            }
            LogicalOp::Project {
                input,
                items,
                distinct,
            }
        }
        LogicalOp::Sort { input, items } => LogicalOp::Sort {
            input: Box::new(ensure_rrf_passthrough(*input)),
            items,
        },
        LogicalOp::Limit { input, count } => LogicalOp::Limit {
            input: Box::new(ensure_rrf_passthrough(*input)),
            count,
        },
        LogicalOp::Skip { input, count } => LogicalOp::Skip {
            input: Box::new(ensure_rrf_passthrough(*input)),
            count,
        },
        other => other,
    }
}

// =============================================================================
// R-HYB2c: doc_score planner post-pass
// =============================================================================

/// Canonical signature captured from a `doc_score(...)` call-site. Multiple
/// call-sites in one plan must all produce the same signature.
#[derive(Debug, Clone, PartialEq)]
struct DocScoreSig {
    doc_variable: String,
    query_vector: Expr,
    alpha: Expr,
    beta: Expr,
    gamma: Expr,
}

fn match_doc_score(expr: &Expr) -> Option<&[Expr]> {
    if let Expr::FunctionCall { name, args, .. } = expr {
        if name == "doc_score" {
            return Some(args);
        }
    }
    None
}

fn expr_contains_doc_score(expr: &Expr) -> bool {
    let mut hit = false;
    collect_doc_presence(expr, &mut hit);
    hit
}

fn collect_doc_presence(expr: &Expr, hit: &mut bool) {
    if *hit {
        return;
    }
    match expr {
        Expr::FunctionCall { name, args, .. } => {
            if name == "doc_score" {
                *hit = true;
                return;
            }
            for a in args {
                collect_doc_presence(a, hit);
            }
        }
        Expr::BinaryOp { left, right, .. } => {
            collect_doc_presence(left, hit);
            collect_doc_presence(right, hit);
        }
        Expr::UnaryOp { expr, .. } => collect_doc_presence(expr, hit),
        Expr::PropertyAccess { expr, .. } => collect_doc_presence(expr, hit),
        Expr::List(items) => items.iter().for_each(|e| collect_doc_presence(e, hit)),
        Expr::MapLiteral(fields) => fields
            .iter()
            .for_each(|(_, v)| collect_doc_presence(v, hit)),
        Expr::In { expr, list } => {
            collect_doc_presence(expr, hit);
            collect_doc_presence(list, hit);
        }
        Expr::IsNull { expr, .. } => collect_doc_presence(expr, hit),
        Expr::StringMatch { expr, pattern, .. } => {
            collect_doc_presence(expr, hit);
            collect_doc_presence(pattern, hit);
        }
        Expr::Subscript { expr, index } => {
            collect_doc_presence(expr, hit);
            collect_doc_presence(index, hit);
        }
        Expr::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            if let Some(o) = operand.as_deref() {
                collect_doc_presence(o, hit);
            }
            for (c, v) in when_clauses {
                collect_doc_presence(c, hit);
                collect_doc_presence(v, hit);
            }
            if let Some(e) = else_clause.as_deref() {
                collect_doc_presence(e, hit);
            }
        }
        _ => {}
    }
}

/// Canonical default weight expressions for doc_score.
fn default_doc_weight(value: f64) -> Expr {
    Expr::Literal(Value::Float(value))
}

/// Parse and validate a single `doc_score(args...)` call.
fn parse_doc_signature(args: &[Expr]) -> Result<DocScoreSig, PlanError> {
    if args.len() < 2 || args.len() > 5 || args.len() == 4 {
        return Err(PlanError::DocScoreArity { got: args.len() });
    }

    let doc_variable = match &args[0] {
        Expr::Variable(v) => v.clone(),
        other => {
            return Err(PlanError::DocScoreDocShape {
                got: format!("{other:?}"),
            });
        }
    };

    let query_vector = args[1].clone();

    let (alpha, beta, gamma) = match args.len() {
        2 => (
            default_doc_weight(0.5),
            default_doc_weight(0.3),
            default_doc_weight(0.2),
        ),
        3 => match &args[2] {
            Expr::MapLiteral(fields) => {
                let mut a = default_doc_weight(0.5);
                let mut b = default_doc_weight(0.3);
                let mut g = default_doc_weight(0.2);
                for (k, v) in fields {
                    match k.as_str() {
                        "alpha" => a = v.clone(),
                        "beta" => b = v.clone(),
                        "gamma" => g = v.clone(),
                        other => {
                            return Err(PlanError::DocScoreWeightsShape {
                                got: format!(
                                    "map with unknown key `{other}` (expected `alpha`, `beta`, `gamma`)"
                                ),
                            });
                        }
                    }
                }
                (a, b, g)
            }
            other => {
                return Err(PlanError::DocScoreWeightsShape {
                    got: format!("single-arg weights must be a map literal, got {other:?}"),
                });
            }
        },
        5 => (args[2].clone(), args[3].clone(), args[4].clone()),
        _ => unreachable!("arity guard above excludes other lengths"),
    };

    Ok(DocScoreSig {
        doc_variable,
        query_vector,
        alpha,
        beta,
        gamma,
    })
}

fn collect_doc_sig_in_expr(
    expr: &Expr,
    found: &mut Option<DocScoreSig>,
    location: &str,
) -> Result<(), PlanError> {
    if let Some(args) = match_doc_score(expr) {
        let sig = parse_doc_signature(args)?;
        match found {
            None => *found = Some(sig),
            Some(existing) if *existing == sig => {}
            Some(_) => {
                return Err(PlanError::DocScoreMultipleCalls {
                    location: location.to_string(),
                });
            }
        }
        return Ok(());
    }
    match expr {
        Expr::BinaryOp { left, right, .. } => {
            collect_doc_sig_in_expr(left, found, location)?;
            collect_doc_sig_in_expr(right, found, location)?;
        }
        Expr::UnaryOp { expr, .. } => collect_doc_sig_in_expr(expr, found, location)?,
        Expr::PropertyAccess { expr, .. } => collect_doc_sig_in_expr(expr, found, location)?,
        Expr::List(items) => {
            for it in items {
                collect_doc_sig_in_expr(it, found, location)?;
            }
        }
        Expr::MapLiteral(fields) => {
            for (_, v) in fields {
                collect_doc_sig_in_expr(v, found, location)?;
            }
        }
        Expr::FunctionCall { args, .. } => {
            for a in args {
                collect_doc_sig_in_expr(a, found, location)?;
            }
        }
        Expr::In { expr, list } => {
            collect_doc_sig_in_expr(expr, found, location)?;
            collect_doc_sig_in_expr(list, found, location)?;
        }
        Expr::IsNull { expr, .. } => collect_doc_sig_in_expr(expr, found, location)?,
        Expr::StringMatch { expr, pattern, .. } => {
            collect_doc_sig_in_expr(expr, found, location)?;
            collect_doc_sig_in_expr(pattern, found, location)?;
        }
        Expr::Subscript { expr, index } => {
            collect_doc_sig_in_expr(expr, found, location)?;
            collect_doc_sig_in_expr(index, found, location)?;
        }
        Expr::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            if let Some(o) = operand.as_deref() {
                collect_doc_sig_in_expr(o, found, location)?;
            }
            for (c, v) in when_clauses {
                collect_doc_sig_in_expr(c, found, location)?;
                collect_doc_sig_in_expr(v, found, location)?;
            }
            if let Some(e) = else_clause.as_deref() {
                collect_doc_sig_in_expr(e, found, location)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn rewrite_doc_in_expr(expr: &mut Expr) {
    if let Expr::FunctionCall { name, .. } = expr {
        if name == "doc_score" {
            *expr = Expr::Variable("__doc_score__".to_string());
            return;
        }
    }
    match expr {
        Expr::BinaryOp { left, right, .. } => {
            rewrite_doc_in_expr(left);
            rewrite_doc_in_expr(right);
        }
        Expr::UnaryOp { expr, .. } => rewrite_doc_in_expr(expr),
        Expr::PropertyAccess { expr, .. } => rewrite_doc_in_expr(expr),
        Expr::List(items) => items.iter_mut().for_each(rewrite_doc_in_expr),
        Expr::MapLiteral(fields) => fields.iter_mut().for_each(|(_, v)| rewrite_doc_in_expr(v)),
        Expr::FunctionCall { args, .. } => args.iter_mut().for_each(rewrite_doc_in_expr),
        Expr::In { expr, list } => {
            rewrite_doc_in_expr(expr);
            rewrite_doc_in_expr(list);
        }
        Expr::IsNull { expr, .. } => rewrite_doc_in_expr(expr),
        Expr::StringMatch { expr, pattern, .. } => {
            rewrite_doc_in_expr(expr);
            rewrite_doc_in_expr(pattern);
        }
        Expr::Subscript { expr, index } => {
            rewrite_doc_in_expr(expr);
            rewrite_doc_in_expr(index);
        }
        Expr::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            if let Some(o) = operand.as_deref_mut() {
                rewrite_doc_in_expr(o);
            }
            for (c, v) in when_clauses {
                rewrite_doc_in_expr(c);
                rewrite_doc_in_expr(v);
            }
            if let Some(e) = else_clause.as_deref_mut() {
                rewrite_doc_in_expr(e);
            }
        }
        _ => {}
    }
}

/// Verify no `doc_score(...)` appears in an illegal position (WHERE, GROUP BY,
/// aggregate args, etc.). Only Project items and Sort items are legal sites.
fn validate_doc_placement(op: &LogicalOp) -> Result<(), PlanError> {
    match op {
        LogicalOp::Filter { input, predicate } => {
            if expr_contains_doc_score(predicate) {
                return Err(PlanError::DocScoreIllegalPosition {
                    location: "WHERE clause".to_string(),
                });
            }
            validate_doc_placement(input)
        }
        LogicalOp::Aggregate {
            input,
            group_by,
            aggregates,
        } => {
            for e in group_by {
                if expr_contains_doc_score(e) {
                    return Err(PlanError::DocScoreIllegalPosition {
                        location: "GROUP BY key".to_string(),
                    });
                }
            }
            for a in aggregates {
                if expr_contains_doc_score(&a.arg) {
                    return Err(PlanError::DocScoreIllegalPosition {
                        location: "aggregate function argument".to_string(),
                    });
                }
            }
            validate_doc_placement(input)
        }
        LogicalOp::Project { input, .. } | LogicalOp::Sort { input, .. } => {
            validate_doc_placement(input)
        }
        LogicalOp::Limit { input, count } | LogicalOp::Skip { input, count } => {
            if expr_contains_doc_score(count) {
                return Err(PlanError::DocScoreIllegalPosition {
                    location: "LIMIT/SKIP expression".to_string(),
                });
            }
            validate_doc_placement(input)
        }
        LogicalOp::Traverse { input, .. }
        | LogicalOp::VectorFilter { input, .. }
        | LogicalOp::VectorTopK { input, .. }
        | LogicalOp::EdgeVectorSearch { input, .. }
        | LogicalOp::TextFilter { input, .. }
        | LogicalOp::EncryptedFilter { input, .. }
        | LogicalOp::Unwind { input, .. }
        | LogicalOp::ShortestPath { input, .. }
        | LogicalOp::Update { input, .. }
        | LogicalOp::RemoveOp { input, .. }
        | LogicalOp::Delete { input, .. }
        | LogicalOp::DetachDocument { input, .. }
        | LogicalOp::AttachDocument { input, .. } => validate_doc_placement(input),
        LogicalOp::CartesianProduct { left, right } | LogicalOp::LeftOuterJoin { left, right } => {
            validate_doc_placement(left)?;
            validate_doc_placement(right)
        }
        LogicalOp::CreateNode { input, .. } => {
            if let Some(inp) = input {
                validate_doc_placement(inp)?;
            }
            Ok(())
        }
        LogicalOp::CreateEdge { input, .. } => validate_doc_placement(input),
        LogicalOp::Merge { pattern, .. } | LogicalOp::Upsert { pattern, .. } => {
            validate_doc_placement(pattern)
        }
        LogicalOp::RankFuse { input, .. } | LogicalOp::DocScore { input, .. } => {
            validate_doc_placement(input)
        }
        _ => Ok(()),
    }
}

fn collect_doc_sig(op: &LogicalOp, sig: &mut Option<DocScoreSig>) -> Result<(), PlanError> {
    match op {
        LogicalOp::Project { input, items, .. } => {
            for it in items {
                collect_doc_sig_in_expr(&it.expr, sig, "RETURN")?;
            }
            collect_doc_sig(input, sig)
        }
        LogicalOp::Sort { input, items } => {
            for it in items {
                collect_doc_sig_in_expr(&it.expr, sig, "ORDER BY")?;
            }
            collect_doc_sig(input, sig)
        }
        LogicalOp::Limit { input, .. } | LogicalOp::Skip { input, .. } => {
            collect_doc_sig(input, sig)
        }
        _ => Ok(()),
    }
}

fn wrap_doc_score(op: LogicalOp, sig: &DocScoreSig, sort_touches_doc: &mut bool) -> LogicalOp {
    match op {
        LogicalOp::Project {
            input,
            mut items,
            distinct,
        } => {
            for it in items.iter_mut() {
                rewrite_doc_in_expr(&mut it.expr);
            }
            let fused_input = LogicalOp::DocScore {
                input,
                doc_variable: sig.doc_variable.clone(),
                query_vector: sig.query_vector.clone(),
                alpha: sig.alpha.clone(),
                beta: sig.beta.clone(),
                gamma: sig.gamma.clone(),
            };
            LogicalOp::Project {
                input: Box::new(fused_input),
                items,
                distinct,
            }
        }
        LogicalOp::Sort { input, mut items } => {
            for it in items.iter_mut() {
                if expr_contains_doc_score(&it.expr) {
                    *sort_touches_doc = true;
                }
                rewrite_doc_in_expr(&mut it.expr);
            }
            LogicalOp::Sort {
                input: Box::new(wrap_doc_score(*input, sig, sort_touches_doc)),
                items,
            }
        }
        LogicalOp::Limit { input, count } => LogicalOp::Limit {
            input: Box::new(wrap_doc_score(*input, sig, sort_touches_doc)),
            count,
        },
        LogicalOp::Skip { input, count } => LogicalOp::Skip {
            input: Box::new(wrap_doc_score(*input, sig, sort_touches_doc)),
            count,
        },
        other => other,
    }
}

fn ensure_doc_passthrough(op: LogicalOp) -> LogicalOp {
    match op {
        LogicalOp::Project {
            input,
            mut items,
            distinct,
        } => {
            let already_present = items.iter().any(|it| {
                it.alias.as_deref() == Some("__doc_score__")
                    || matches!(&it.expr, Expr::Variable(v) if v == "__doc_score__")
            });
            if !already_present {
                items.push(ProjectItem {
                    expr: Expr::Variable("__doc_score__".to_string()),
                    alias: Some("__doc_score__".to_string()),
                });
            }
            LogicalOp::Project {
                input,
                items,
                distinct,
            }
        }
        LogicalOp::Sort { input, items } => LogicalOp::Sort {
            input: Box::new(ensure_doc_passthrough(*input)),
            items,
        },
        LogicalOp::Limit { input, count } => LogicalOp::Limit {
            input: Box::new(ensure_doc_passthrough(*input)),
            count,
        },
        LogicalOp::Skip { input, count } => LogicalOp::Skip {
            input: Box::new(ensure_doc_passthrough(*input)),
            count,
        },
        other => other,
    }
}

/// Planner pass: detect a temporal time-slice predicate in a Filter and lift
/// it into the immediately-below Traverse's `temporal_filter` field.
///
/// Recognises `temporal_active_at(r, $T)` (or with literal/Int argument):
/// → sets `upper_ms = T` (for bounded prefix scan) and `lower_ms = T`
///   (server-side `valid_to > T OR NULL` check after key decode).
///
/// Walks the tree recursively; only the immediate `Filter { Traverse }` pair
/// is rewritten. Other shapes fall through unchanged. The original Filter is
/// kept as a safety net for correctness — the predicate is evaluated again
/// against materialized rows. Eliminating it is a future optimization.
pub(crate) fn lift_temporal_filter(op: LogicalOp) -> LogicalOp {
    match op {
        LogicalOp::Filter { input, predicate } => {
            // Try to extract a temporal slice from this predicate.
            let candidate = extract_temporal_slice(&predicate);
            let lifted_input = match (candidate, *input) {
                (
                    Some((edge_var, upper_ms, lower_ms)),
                    LogicalOp::Traverse {
                        input: t_input,
                        source,
                        edge_types,
                        direction,
                        target_variable,
                        target_labels,
                        length,
                        edge_variable,
                        target_filters,
                        edge_filters,
                        temporal_filter,
                        path_variable,
                    },
                ) if edge_variable.as_deref() == Some(edge_var.as_str())
                    && temporal_filter.is_none() =>
                {
                    LogicalOp::Traverse {
                        input: Box::new(lift_temporal_filter(*t_input)),
                        source,
                        edge_types,
                        direction,
                        target_variable,
                        target_labels,
                        length,
                        edge_variable,
                        target_filters,
                        edge_filters,
                        temporal_filter: Some(crate::planner::logical::TemporalFilter {
                            edge_variable: edge_var,
                            upper_ms: Some(upper_ms),
                            lower_ms: Some(lower_ms),
                        }),
                        path_variable,
                    }
                }
                (_, other) => lift_temporal_filter(other),
            };
            LogicalOp::Filter {
                input: Box::new(lifted_input),
                predicate,
            }
        }
        // Recursive descent for everything else (preserves shape).
        LogicalOp::Project {
            input,
            items,
            distinct,
        } => LogicalOp::Project {
            input: Box::new(lift_temporal_filter(*input)),
            items,
            distinct,
        },
        LogicalOp::Sort { input, items } => LogicalOp::Sort {
            input: Box::new(lift_temporal_filter(*input)),
            items,
        },
        LogicalOp::Limit { input, count } => LogicalOp::Limit {
            input: Box::new(lift_temporal_filter(*input)),
            count,
        },
        LogicalOp::Skip { input, count } => LogicalOp::Skip {
            input: Box::new(lift_temporal_filter(*input)),
            count,
        },
        other => other,
    }
}

/// Inspect a Filter predicate and, if it is a literal-argument temporal
/// helper call, extract the edge variable name + the upper / lower millisecond
/// bounds that should be propagated as a `TemporalFilter` on the Traverse.
///
/// Recognises:
/// - `temporal_active_at(r, T)`  → `(r, upper=T,    lower=T)`
/// - `temporal_overlaps(r, T0, T1)` → `(r, upper=T1-1, lower=T0)` (strict end)
///
/// Parameter expressions are NOT lifted here — the planner runs before
/// parameter substitution. A second pass (`lift_temporal_filter`) is invoked
/// AFTER substitution in `execute()` to catch the bound-parameter case.
fn extract_temporal_slice(predicate: &Expr) -> Option<(String, i64, i64)> {
    let Expr::FunctionCall { name, args, .. } = predicate else {
        return None;
    };
    let literal_ts = |e: &Expr| -> Option<i64> {
        match e {
            Expr::Literal(Value::Int(n)) => Some(*n),
            Expr::Literal(Value::Timestamp(n)) => Some(*n),
            _ => None,
        }
    };
    match name.as_str() {
        "temporal_active_at" if args.len() == 2 => {
            let Expr::Variable(var) = &args[0] else {
                return None;
            };
            let t = literal_ts(&args[1])?;
            Some((var.clone(), t, t))
        }
        "temporal_overlaps" if args.len() == 3 => {
            let Expr::Variable(var) = &args[0] else {
                return None;
            };
            let t0 = literal_ts(&args[1])?;
            let t1 = literal_ts(&args[2])?;
            // Overlaps semantics: valid_from < t1. Express as upper bound on
            // valid_from of `t1 - 1` (inclusive) so existing prefix-scan
            // bounded-key logic gives the correct half-open interval.
            Some((var.clone(), t1.saturating_sub(1), t0))
        }
        _ => None,
    }
}

fn rewrite_doc_score(op: LogicalOp) -> Result<LogicalOp, PlanError> {
    validate_doc_placement(&op)?;

    let mut sig: Option<DocScoreSig> = None;
    collect_doc_sig(&op, &mut sig)?;

    let Some(sig) = sig else {
        return Ok(op);
    };

    let mut sort_touches_doc = false;
    let op = wrap_doc_score(op, &sig, &mut sort_touches_doc);
    let op = if sort_touches_doc {
        ensure_doc_passthrough(op)
    } else {
        op
    };

    Ok(op)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod numeric_predicate_tests;
