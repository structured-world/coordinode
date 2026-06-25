//! Suggestion detectors for EXPLAIN SUGGEST.
//!
//! Each detector analyzes a logical plan and produces zero or more suggestions.
//! CE includes 5 detectors:
//! - MissingIndex: full label scan on filtered property
//! - UnboundedTraversal: variable-length path without depth limit
//! - CartesianProduct: unconnected MATCH patterns
//! - KnnWithoutIndex: distance ORDER BY + LIMIT without spatial index
//! - VectorWithoutPreFilter: vector scan without graph narrowing

use crate::index::{IndexRegistry, IndexType};
use crate::planner::logical::{LogicalOp, LogicalPlan};

use super::suggest::{Severity, Suggestion, SuggestionKind};

/// Run all CE detectors on a logical plan and return suggestions.
///
/// When `registry` is provided, the MissingIndex and KnnWithoutIndex detectors
/// will skip suggestions for properties that already have an index — preventing
/// false positives.
///
/// Suggestions are sorted by severity (Critical first, then Warning, then Info).
pub fn detect_suggestions(plan: &LogicalPlan, registry: Option<&IndexRegistry>) -> Vec<Suggestion> {
    let mut suggestions = Vec::new();

    detect_missing_index(&plan.root, &mut suggestions, registry);
    detect_unbounded_traversal(&plan.root, &mut suggestions);
    detect_cartesian_product(&plan.root, &mut suggestions);
    detect_knn_without_index(&plan.root, &mut suggestions, registry);
    detect_vector_without_prefilter(&plan.root, &mut suggestions);

    // Sort by severity descending (Critical > Warning > Info)
    suggestions.sort_by_key(|s| std::cmp::Reverse(s.severity));

    suggestions
}

// --- Detector 1: MissingIndex ---
// Detects: NodeScan (full label scan) followed by Filter on a property
// that could be indexed. Pattern: Filter { input: NodeScan { labels, ... }, predicate }
// where predicate references a property of the scanned label.

fn detect_missing_index(
    op: &LogicalOp,
    suggestions: &mut Vec<Suggestion>,
    registry: Option<&IndexRegistry>,
) {
    match op {
        LogicalOp::Filter { input, predicate } => {
            // Check if input is a NodeScan with labels
            if let LogicalOp::NodeScan {
                labels, variable, ..
            } = input.as_ref()
            {
                if !labels.is_empty() {
                    // Extract property names referenced in the filter predicate
                    let props = extract_property_accesses(predicate, variable);
                    for prop in props {
                        let label = &labels[0];

                        // Skip if an index already covers this label+property
                        if let Some(reg) = registry {
                            if !reg.indexes_for_property(label, &prop).is_empty() {
                                continue;
                            }
                        }

                        suggestions.push(
                            Suggestion::new(
                                SuggestionKind::CreateIndex,
                                Severity::Critical,
                                format!(
                                    "Full label scan on {label}.{prop} — \
                                     filtering {label} nodes by '{prop}' without an index \
                                     requires scanning all {label} nodes"
                                ),
                            )
                            .with_ddl(format!(
                                "CREATE INDEX {}_{} ON {}({})",
                                label.to_lowercase(),
                                prop.to_lowercase(),
                                label,
                                prop,
                            )),
                        );
                    }
                }
            }

            // Also check deeper in the tree
            detect_missing_index(input, suggestions, registry);
        }

        // Recurse into all operator children
        _ => {
            for child in children(op) {
                detect_missing_index(child, suggestions, registry);
            }
        }
    }
}

// --- Detector 2: UnboundedTraversal ---
// Detects: Traverse with variable-length path that has no upper bound.
// Pattern: Traverse { length: Some(LengthBound { max: None, .. }), .. }

fn detect_unbounded_traversal(op: &LogicalOp, suggestions: &mut Vec<Suggestion>) {
    if let LogicalOp::Traverse {
        edge_types,
        length: Some(bounds),
        ..
    } = op
    {
        if bounds.max.is_none() {
            let types_str = if edge_types.is_empty() {
                String::new()
            } else {
                format!(":{}", edge_types.join("|"))
            };
            let min = bounds.min.unwrap_or(1);
            suggestions.push(
                Suggestion::new(
                    SuggestionKind::AddDepthBound,
                    Severity::Warning,
                    format!(
                        "Unbounded variable-length traversal [{types_str}*{min}..] — \
                         may cause exponential fan-out on dense graphs. \
                         Add an upper bound to prevent runaway queries"
                    ),
                )
                .with_rewrite(format!(
                    "Use [{types_str}*{min}..10] or another reasonable upper bound"
                )),
            );
        }
    }

    for child in children(op) {
        detect_unbounded_traversal(child, suggestions);
    }
}

// --- Detector 3: CartesianProduct ---
// Detects: CartesianProduct operator (unconnected MATCH patterns).
// This produces O(left × right) rows, usually a mistake.

fn detect_cartesian_product(op: &LogicalOp, suggestions: &mut Vec<Suggestion>) {
    if let LogicalOp::CartesianProduct { left, right } = op {
        let left_vars = collect_variables(left);
        let right_vars = collect_variables(right);

        suggestions.push(Suggestion::new(
            SuggestionKind::AddJoinPredicate,
            Severity::Warning,
            format!(
                "Cartesian product between disconnected patterns \
                 ({} × {}) — produces O(n²) rows. \
                 Add a relationship or WHERE clause connecting the patterns",
                format_var_list(&left_vars),
                format_var_list(&right_vars),
            ),
        ));

        // Recurse into both sides
        detect_cartesian_product(left, suggestions);
        detect_cartesian_product(right, suggestions);
        return;
    }

    for child in children(op) {
        detect_cartesian_product(child, suggestions);
    }
}

// --- Detector 4: KnnWithoutIndex ---
// Detects: Sort by distance function + Limit, without a dedicated spatial/vector index.
// Pattern: Limit { input: Sort { items containing point.distance or vector_distance } }

fn detect_knn_without_index(
    op: &LogicalOp,
    suggestions: &mut Vec<Suggestion>,
    registry: Option<&IndexRegistry>,
) {
    // Legacy pattern: Limit { Sort { vector_distance } } — present when the
    // planner's VectorTopK optimization did NOT rewrite the subtree (e.g. Sort is
    // on non-vector alias, there's no LIMIT, or the inner Project dropped the
    // vector variable). In these cases we cannot easily extract (label, property)
    // so we emit the suggestion unconditionally — the user benefits from knowing
    // that ORDER BY distance is a KNN query.
    if let LogicalOp::Limit { input, .. } = op {
        if let LogicalOp::Sort { input: _, items } = input.as_ref() {
            for item in items {
                if expr_contains_distance_fn(&item.expr) {
                    suggestions.push(
                        Suggestion::new(
                            SuggestionKind::CreateVectorIndex,
                            Severity::Info,
                            "ORDER BY distance + LIMIT without vector/spatial index — \
                             computing distance for all candidates. \
                             A vector index (HNSW) or spatial index can accelerate KNN queries"
                                .to_string(),
                        )
                        .with_ddl(
                            "CREATE VECTOR INDEX ON <Label>(<property>) \
                             OPTIONS {metric: 'cosine', dimensions: <N>}"
                                .to_string(),
                        ),
                    );
                    break;
                }
            }
        }
    }

    // Optimized pattern: VectorTopK (after planner rewrite). Extract the
    // (label, property) pair from `vector_expr` + upstream NodeScan and check
    // the registry: if an HNSW index already exists, the user is already using
    // acceleration — no suggestion needed (avoids false positives).
    if let LogicalOp::VectorTopK {
        vector_expr, input, ..
    } = op
    {
        if let Some((label, property)) = extract_label_and_property(vector_expr, input) {
            // If we have a registry and an HNSW index exists for this (label, prop),
            // don't emit the suggestion — the query is already optimized.
            let has_hnsw_index = registry
                .map(|r| {
                    r.indexes_for_property(&label, &property)
                        .iter()
                        .any(|idx| idx.index_type == IndexType::Hnsw)
                })
                .unwrap_or(false);

            if !has_hnsw_index {
                suggestions.push(
                    Suggestion::new(
                        SuggestionKind::CreateVectorIndex,
                        Severity::Info,
                        format!(
                            "ORDER BY distance + LIMIT on {label}({property}) without vector \
                             index — computing distance for all candidates. A vector index \
                             (HNSW) can accelerate KNN queries."
                        ),
                    )
                    .with_ddl(format!(
                        "CREATE VECTOR INDEX ON {label}({property}) \
                         OPTIONS {{metric: 'cosine', dimensions: <N>}}"
                    )),
                );
            }
        } else {
            // Could not extract label/property (e.g. missing NodeScan ancestor) —
            // fall back to the generic suggestion.
            suggestions.push(
                Suggestion::new(
                    SuggestionKind::CreateVectorIndex,
                    Severity::Info,
                    "ORDER BY distance + LIMIT without vector/spatial index — \
                     computing distance for all candidates. \
                     A vector index (HNSW) or spatial index can accelerate KNN queries"
                        .to_string(),
                )
                .with_ddl(
                    "CREATE VECTOR INDEX ON <Label>(<property>) \
                     OPTIONS {metric: 'cosine', dimensions: <N>}"
                        .to_string(),
                ),
            );
        }
    }

    for child in children(op) {
        detect_knn_without_index(child, suggestions, registry);
    }
}

/// Extract `(label, property)` for a VectorTopK operator by matching `vector_expr`
/// (must be `Variable(var).prop`) to the nearest upstream `NodeScan` that binds
/// the same variable with a single label.
///
/// Returns `None` if:
/// - `vector_expr` is not `Variable.prop`
/// - no NodeScan with the matching variable is found in `input`
/// - the NodeScan has no labels or multiple labels (ambiguous)
fn extract_label_and_property(
    vector_expr: &crate::cypher::ast::Expr,
    input: &LogicalOp,
) -> Option<(String, String)> {
    use crate::cypher::ast::Expr;

    let (var_name, property) = match vector_expr {
        Expr::PropertyAccess { expr, property } => match expr.as_ref() {
            Expr::Variable(v) => (v.clone(), property.clone()),
            _ => return None,
        },
        _ => return None,
    };

    let label = find_node_scan_label(input, &var_name)?;
    Some((label, property))
}

/// Recursively search for a `NodeScan { variable, labels }` with matching variable.
/// Returns the first label (NodeScan bindings use single-label convention in practice).
fn find_node_scan_label(op: &LogicalOp, var_name: &str) -> Option<String> {
    if let LogicalOp::NodeScan {
        variable, labels, ..
    } = op
    {
        if variable == var_name && !labels.is_empty() {
            return Some(labels[0].clone());
        }
    }
    for child in children(op) {
        if let Some(l) = find_node_scan_label(child, var_name) {
            return Some(l);
        }
    }
    None
}

// --- Detector 5: VectorWithoutPreFilter ---
// Detects: VectorFilter whose input is a NodeScan (no graph narrowing).
// Pattern: VectorFilter { input: NodeScan { .. }, .. }

fn detect_vector_without_prefilter(op: &LogicalOp, suggestions: &mut Vec<Suggestion>) {
    if let LogicalOp::VectorFilter { input, .. } = op {
        if matches!(input.as_ref(), LogicalOp::NodeScan { .. }) {
            suggestions.push(Suggestion::new(
                SuggestionKind::AddGraphPreFilter,
                Severity::Info,
                "Vector search on full label scan without graph pre-filter — \
                 scanning all vectors in the label. \
                 Add a graph traversal or property filter before the vector search \
                 to narrow the candidate set"
                    .to_string(),
            ));
        }
    }

    for child in children(op) {
        detect_vector_without_prefilter(child, suggestions);
    }
}

// --- Helpers ---

/// Get direct children of an operator for recursive traversal.
fn children(op: &LogicalOp) -> Vec<&LogicalOp> {
    match op {
        LogicalOp::Extension { .. }
        | LogicalOp::NodeScan { .. }
        | LogicalOp::HnswScan { .. }
        | LogicalOp::Empty
        | LogicalOp::ProcedureCall { .. }
        | LogicalOp::AlterLabel { .. }
        | LogicalOp::CreateTextIndex { .. }
        | LogicalOp::DropTextIndex { .. }
        | LogicalOp::CreateEncryptedIndex { .. }
        | LogicalOp::DropEncryptedIndex { .. }
        | LogicalOp::CreateVectorIndex { .. }
        | LogicalOp::DropVectorIndex { .. }
        | LogicalOp::CreateEdgeType { .. }
        | LogicalOp::CreateNodeType { .. }
        | LogicalOp::CreateTrigger { .. }
        | LogicalOp::DropTrigger { .. }
        | LogicalOp::ShowTriggers
        | LogicalOp::AlterTrigger { .. } => vec![],

        LogicalOp::Filter { input, .. }
        | LogicalOp::Project { input, .. }
        | LogicalOp::Aggregate { input, .. }
        | LogicalOp::Sort { input, .. }
        | LogicalOp::Limit { input, .. }
        | LogicalOp::Skip { input, .. }
        | LogicalOp::Traverse { input, .. }
        | LogicalOp::CreateEdge { input, .. }
        | LogicalOp::Update { input, .. }
        | LogicalOp::RemoveOp { input, .. }
        | LogicalOp::Delete { input, .. }
        | LogicalOp::DetachDocument { input, .. }
        | LogicalOp::AttachDocument { input, .. }
        | LogicalOp::MergeNodes { input, .. }
        | LogicalOp::CloneNode { input, .. }
        | LogicalOp::RedirectEdges { input, .. }
        | LogicalOp::Unwind { input, .. }
        | LogicalOp::VectorFilter { input, .. }
        | LogicalOp::VectorTopK { input, .. }
        | LogicalOp::EdgeVectorSearch { input, .. }
        | LogicalOp::TextFilter { input, .. }
        | LogicalOp::EncryptedFilter { input, .. }
        | LogicalOp::ShortestPath { input, .. }
        | LogicalOp::RankFuse { input, .. }
        | LogicalOp::DocScore { input, .. }
        | LogicalOp::MaxSimTopK { input, .. } => vec![input],

        LogicalOp::CartesianProduct { left, right } | LogicalOp::LeftOuterJoin { left, right } => {
            vec![left, right]
        }

        LogicalOp::CreateNode { input, .. } => input.as_ref().map_or(vec![], |i| vec![i.as_ref()]),

        LogicalOp::Merge { pattern, .. } | LogicalOp::Upsert { pattern, .. } => vec![pattern],

        LogicalOp::Union { inputs, .. } => inputs.iter().collect(),

        LogicalOp::Foreach { input, body, .. } => vec![input, body],

        LogicalOp::CallSubquery { input, body, .. } => vec![input, body],

        LogicalOp::CreateIndex { .. }
        | LogicalOp::DropIndex { .. }
        | LogicalOp::IndexScan { .. } => vec![],
    }
}

/// Extract property names accessed on a specific variable from an expression.
/// E.g., for variable "n", `n.email = 'foo'` → ["email"].
fn extract_property_accesses(expr: &crate::cypher::ast::Expr, variable: &str) -> Vec<String> {
    let mut props = Vec::new();
    collect_property_accesses(expr, variable, &mut props);
    props
}

fn collect_property_accesses(
    expr: &crate::cypher::ast::Expr,
    variable: &str,
    props: &mut Vec<String>,
) {
    use crate::cypher::ast::Expr;
    match expr {
        Expr::PropertyAccess {
            expr: inner,
            property,
        } => {
            // Check if the inner expression is the variable we're looking for
            if let Expr::Variable(var) = inner.as_ref() {
                if var == variable && !props.contains(property) {
                    props.push(property.clone());
                }
            }
            collect_property_accesses(inner, variable, props);
        }
        Expr::BinaryOp { left, right, .. } => {
            collect_property_accesses(left, variable, props);
            collect_property_accesses(right, variable, props);
        }
        Expr::UnaryOp { expr, .. } => {
            collect_property_accesses(expr, variable, props);
        }
        Expr::FunctionCall { args, .. } => {
            for arg in args {
                collect_property_accesses(arg, variable, props);
            }
        }
        Expr::In { expr, list } => {
            collect_property_accesses(expr, variable, props);
            collect_property_accesses(list, variable, props);
        }
        Expr::IsNull { expr, .. } => {
            collect_property_accesses(expr, variable, props);
        }
        Expr::StringMatch { expr, pattern, .. } => {
            collect_property_accesses(expr, variable, props);
            collect_property_accesses(pattern, variable, props);
        }
        Expr::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            if let Some(op) = operand {
                collect_property_accesses(op, variable, props);
            }
            for (cond, result) in when_clauses {
                collect_property_accesses(cond, variable, props);
                collect_property_accesses(result, variable, props);
            }
            if let Some(el) = else_clause {
                collect_property_accesses(el, variable, props);
            }
        }
        Expr::List(items) => {
            for item in items {
                collect_property_accesses(item, variable, props);
            }
        }
        // Literals, variables, parameters, MapLiteral, Star — no property access to extract
        _ => {}
    }
}

/// Collect all variable names introduced in a subtree.
fn collect_variables(op: &LogicalOp) -> Vec<String> {
    let mut vars = Vec::new();
    collect_vars_recursive(op, &mut vars);
    vars
}

fn collect_vars_recursive(op: &LogicalOp, vars: &mut Vec<String>) {
    match op {
        LogicalOp::NodeScan { variable, .. } if !vars.contains(variable) => {
            vars.push(variable.clone());
        }
        LogicalOp::Traverse {
            target_variable, ..
        } if !vars.contains(target_variable) => {
            vars.push(target_variable.clone());
        }
        _ => {}
    }
    for child in children(op) {
        collect_vars_recursive(child, vars);
    }
}

/// Check if an expression contains a distance/similarity function call.
fn expr_contains_distance_fn(expr: &crate::cypher::ast::Expr) -> bool {
    use crate::cypher::ast::Expr;
    match expr {
        Expr::FunctionCall { name, .. } => {
            let lower = name.to_lowercase();
            lower.contains("distance") || lower.contains("similarity")
        }
        Expr::BinaryOp { left, right, .. } => {
            expr_contains_distance_fn(left) || expr_contains_distance_fn(right)
        }
        Expr::UnaryOp { expr, .. } => expr_contains_distance_fn(expr),
        _ => false,
    }
}

fn format_var_list(vars: &[String]) -> String {
    if vars.is_empty() {
        "?".to_string()
    } else {
        vars.join(", ")
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
