//! Logical plan builder: converts Cypher AST → logical plan tree.
//!
//! Walks the flat clause list and builds a tree of relational algebra operators.
//! Each clause modifies the "current working set" by wrapping the previous
//! operator in a new one.

use super::logical::*;
use crate::cypher::ast::*;
use coordinode_core::graph::types::Value;

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
}

/// Build a logical plan from a validated Cypher AST.
pub fn build_logical_plan(query: &Query) -> Result<LogicalPlan, PlanError> {
    if query.clauses.is_empty() {
        return Err(PlanError::EmptyQuery);
    }

    let mut current: Option<LogicalOp> = None;
    let mut snapshot_ts: Option<Expr> = None;

    for clause in &query.clauses {
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

    // Extract vector_consistency from per-query hints (overrides session default).
    let vector_consistency = query
        .hints
        .iter()
        .map(|h| {
            let crate::cypher::ast::QueryHint::VectorConsistency(mode) = h;
            *mode
        })
        .next()
        .unwrap_or_default();

    Ok(LogicalPlan {
        root,
        snapshot_ts,
        vector_consistency,
    })
}

/// Apply a single clause to the current plan, producing a new operator.
fn apply_clause(current: Option<LogicalOp>, clause: &Clause) -> Result<LogicalOp, PlanError> {
    match clause {
        Clause::Match(mc) => {
            match current {
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
        Clause::Set(items) => {
            let input = current.unwrap_or(LogicalOp::Empty);
            Ok(LogicalOp::Update {
                input: Box::new(input),
                items: items.clone(),
            })
        }
        Clause::Remove(items) => {
            let input = current.unwrap_or(LogicalOp::Empty);
            Ok(LogicalOp::RemoveOp {
                input: Box::new(input),
                items: items.clone(),
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
    }
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
pub fn optimize_index_selection(
    op: LogicalOp,
    registry: &crate::index::IndexRegistry,
) -> LogicalOp {
    // For Filter(NodeScan), check if we can rewrite to IndexScan before recursing.
    if let LogicalOp::Filter { input, predicate } = op {
        // Check if the pattern matches: Filter(NodeScan{single-label}, var.prop = val).
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
                    let label = &labels[0];
                    if let Some(prop) = extract_index_property(left, variable) {
                        if matches!(right.as_ref(), Expr::Literal(_) | Expr::Parameter(_)) {
                            let indexes = registry.indexes_for_property(label, &prop);
                            if let Some(idx) = indexes.into_iter().next() {
                                return LogicalOp::IndexScan {
                                    variable: variable.clone(),
                                    label: label.clone(),
                                    index_name: idx.name.clone(),
                                    property: prop,
                                    value_expr: *right.clone(),
                                };
                            }
                        }
                    }
                }
            }
        }
        // Pattern not matched — recurse into input, keep Filter.
        return LogicalOp::Filter {
            input: Box::new(optimize_index_selection(*input, registry)),
            predicate,
        };
    }

    // For all other operators, recurse into direct children only.
    match op {
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
        // Leaf ops, DDL ops, and any ops without an input field — return as-is.
        other => other,
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
            Expr::Variable(v) if v == var_name => {
                // Only a passthrough if the alias is either None or same as var.
                // An aliased variable (`n AS m`) would rename the row key.
                if item.alias.is_none() || item.alias.as_ref() == Some(v) {
                    return true;
                }
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

/// Build a scan + traversal chain from a single pattern.
fn build_pattern_scan(pattern: &Pattern) -> Result<LogicalOp, PlanError> {
    if pattern.elements.is_empty() {
        return Err(PlanError::EmptyPattern);
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
                });
            }
        }
    }

    // Post-process: fill in target info from the pattern structure.
    // Walk the elements in pairs: (Node, Rel, Node, Rel, Node, ...)
    // The first Node is the scan, each (Rel, Node) pair is a Traverse.
    let result = build_pattern_chain(&pattern.elements)?;

    Ok(result)
}

/// Build a chain of NodeScan → Traverse → Traverse from pattern elements.
fn build_pattern_chain(elements: &[PatternElement]) -> Result<LogicalOp, PlanError> {
    if elements.is_empty() {
        return Err(PlanError::EmptyPattern);
    }

    // First element must be a Node
    let first_node = match &elements[0] {
        PatternElement::Node(np) => np,
        _ => return Err(PlanError::UnsupportedPattern),
    };

    let mut current = LogicalOp::NodeScan {
        variable: first_node.variable.clone().unwrap_or_default(),
        labels: first_node.labels.clone(),
        property_filters: first_node.properties.clone(),
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
                    aggregates.push(AggregateItem {
                        function: name.clone(),
                        arg,
                        distinct: *distinct,
                        alias: Some(agg_col.clone()),
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
                    aggregates.push(AggregateItem {
                        function: name.clone(),
                        arg,
                        distinct: *distinct,
                        alias: Some(agg_col.clone()),
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
        LogicalOp::Filter { input, .. }
        | LogicalOp::VectorFilter { input, .. }
        | LogicalOp::TextFilter { input, .. }
        | LogicalOp::EncryptedFilter { input, .. }
        | LogicalOp::Aggregate { input, .. }
        | LogicalOp::Project { input, .. }
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::cypher::parser::parse;

    fn plan(input: &str) -> LogicalPlan {
        let query = parse(input).unwrap_or_else(|e| panic!("parse failed: {e}"));
        build_logical_plan(&query).unwrap_or_else(|e| panic!("plan failed: {e}"))
    }

    fn plan_root(input: &str) -> LogicalOp {
        plan(input).root
    }

    // -- Basic MATCH → NodeScan --

    #[test]
    fn match_node_scan() {
        let root = plan_root("MATCH (n:User) RETURN n");
        // Should be Project(NodeScan)
        if let LogicalOp::Project { input, .. } = &root {
            assert!(
                matches!(**input, LogicalOp::NodeScan { ref labels, .. } if labels == &["User"])
            );
        } else {
            panic!("expected Project, got: {root:?}");
        }
    }

    #[test]
    fn match_node_with_properties() {
        let root = plan_root("MATCH (n:User {name: 'Alice'}) RETURN n");
        if let LogicalOp::Project { input, .. } = &root {
            if let LogicalOp::NodeScan {
                property_filters, ..
            } = input.as_ref()
            {
                assert_eq!(property_filters.len(), 1);
                assert_eq!(property_filters[0].0, "name");
            } else {
                panic!("expected NodeScan");
            }
        } else {
            panic!("expected Project");
        }
    }

    // -- MATCH with relationship → Traverse --

    #[test]
    fn match_traverse() {
        let root = plan_root("MATCH (a)-[:KNOWS]->(b) RETURN a, b");
        if let LogicalOp::Project { input, .. } = &root {
            if let LogicalOp::Traverse {
                edge_types,
                direction,
                target_variable,
                ..
            } = input.as_ref()
            {
                assert_eq!(edge_types, &["KNOWS"]);
                assert_eq!(*direction, Direction::Outgoing);
                assert_eq!(target_variable, "b");
            } else {
                panic!("expected Traverse, got: {input:?}");
            }
        } else {
            panic!("expected Project");
        }
    }

    #[test]
    fn match_long_traversal_chain() {
        let root = plan_root("MATCH (a)-[:KNOWS]->(b)-[:WORKS_AT]->(c) RETURN a, b, c");
        // Should be Project → Traverse(WORKS_AT) → Traverse(KNOWS) → NodeScan
        if let LogicalOp::Project { input, .. } = &root {
            if let LogicalOp::Traverse {
                input: inner,
                edge_types,
                ..
            } = input.as_ref()
            {
                assert_eq!(edge_types, &["WORKS_AT"]);
                assert!(matches!(**inner, LogicalOp::Traverse { .. }));
            } else {
                panic!("expected Traverse chain");
            }
        }
    }

    // -- WHERE → Filter --

    #[test]
    fn match_where_filter() {
        let root = plan_root("MATCH (n:User) WHERE n.age > 25 RETURN n");
        if let LogicalOp::Project { input, .. } = &root {
            assert!(matches!(**input, LogicalOp::Filter { .. }));
        } else {
            panic!("expected Project");
        }
    }

    // -- Multiple patterns → CartesianProduct --

    #[test]
    fn multiple_patterns() {
        let root = plan_root("MATCH (a:User), (b:Movie) RETURN a, b");
        if let LogicalOp::Project { input, .. } = &root {
            assert!(matches!(**input, LogicalOp::CartesianProduct { .. }));
        } else {
            panic!("expected Project");
        }
    }

    // -- Aggregation → Aggregate + Project --

    #[test]
    fn aggregation() {
        let root = plan_root("MATCH (n:User) RETURN n.city AS city, count(*) AS cnt");
        if let LogicalOp::Project { input, .. } = &root {
            assert!(matches!(**input, LogicalOp::Aggregate { .. }));
        } else {
            panic!("expected Project");
        }
    }

    // -- ORDER BY / SKIP / LIMIT --

    #[test]
    fn order_by_limit() {
        let root = plan_root("MATCH (n) RETURN n.name ORDER BY n.name DESC LIMIT 10");
        // Plan should be: Limit → Sort → Project → NodeScan
        if let LogicalOp::Limit { input, .. } = &root {
            if let LogicalOp::Sort {
                input: inner,
                items,
            } = input.as_ref()
            {
                assert_eq!(items.len(), 1);
                assert!(!items[0].ascending);
                assert!(matches!(**inner, LogicalOp::Project { .. }));
            } else {
                panic!("expected Sort");
            }
        } else {
            panic!("expected Limit");
        }
    }

    // -- WITH → scope barrier Project --

    #[test]
    fn with_clause() {
        let root = plan_root("MATCH (n:User) WITH n, count(*) AS cnt WHERE cnt > 5 RETURN n, cnt");
        // Plan should contain Filter → Project(Aggregate) somewhere
        if let LogicalOp::Project { input, .. } = &root {
            // Input should be Filter(Project(Aggregate(NodeScan)))
            assert!(
                matches!(**input, LogicalOp::Filter { .. }),
                "expected Filter, got: {input:?}"
            );
        }
    }

    // -- CREATE → CreateNode --

    #[test]
    fn create_node() {
        let root = plan_root("CREATE (n:User {name: 'Alice'}) RETURN n");
        // Project → CreateNode → Empty
        if let LogicalOp::Project { input, .. } = &root {
            assert!(matches!(**input, LogicalOp::CreateNode { .. }));
        }
    }

    // -- DELETE → Delete --

    #[test]
    fn delete_node() {
        let root = plan_root("MATCH (n:User {id: 42}) DELETE n");
        assert!(matches!(root, LogicalOp::Delete { detach: false, .. }));
    }

    #[test]
    fn detach_delete() {
        let root = plan_root("MATCH (n:User {id: 42}) DETACH DELETE n");
        if let LogicalOp::Delete {
            detach, variables, ..
        } = &root
        {
            assert!(detach);
            assert_eq!(variables, &["n"]);
        } else {
            panic!("expected Delete");
        }
    }

    // -- SET → Update --

    #[test]
    fn set_property() {
        let root = plan_root("MATCH (n:User {id: 42}) SET n.name = 'Bob'");
        assert!(matches!(root, LogicalOp::Update { .. }));
    }

    // -- REMOVE → RemoveOp --

    #[test]
    fn remove_property() {
        let root = plan_root("MATCH (n) REMOVE n.age");
        assert!(matches!(root, LogicalOp::RemoveOp { .. }));
    }

    // -- EXPLAIN output --

    #[test]
    fn explain_simple() {
        let p = plan("MATCH (n:User) RETURN n");
        let explain = p.explain();
        assert!(explain.contains("NodeScan"));
        assert!(explain.contains("Project"));
        assert!(explain.contains("User"));
    }

    #[test]
    fn explain_traverse() {
        let p = plan("MATCH (a:User)-[:KNOWS]->(b) RETURN a, b");
        let explain = p.explain();
        assert!(explain.contains("Traverse"));
        assert!(explain.contains("KNOWS"));
    }

    #[test]
    fn explain_aggregate() {
        let p = plan("MATCH (n:User) RETURN n.city, count(*) AS cnt");
        let explain = p.explain();
        assert!(explain.contains("Aggregate"));
        assert!(explain.contains("Project"));
    }

    // -- UNWIND --

    #[test]
    fn unwind_plan() {
        let root = plan_root("UNWIND [1, 2, 3] AS x RETURN x");
        // Project → Unwind → Empty
        if let LogicalOp::Project { input, .. } = &root {
            assert!(matches!(**input, LogicalOp::Unwind { .. }));
        }
    }

    // -- Complex query --

    #[test]
    fn complex_graph_vector_query() {
        let p = plan(
            "MATCH (u:User {id: $me})-[:LIKES]->(m:Movie) \
             WHERE vector_distance(m.embedding, $query_vec) < 0.3 \
             WITH m.genre AS genre, count(*) AS cnt \
             ORDER BY cnt DESC \
             LIMIT 10 \
             RETURN genre, cnt",
        );
        let explain = p.explain();
        // Should contain: NodeScan, Traverse, Filter, Aggregate, Project, Sort, Limit
        assert!(explain.contains("NodeScan"));
        assert!(explain.contains("Traverse"));
    }

    // -- Variable-length path --

    #[test]
    fn variable_length_path() {
        let root = plan_root("MATCH (a)-[:KNOWS*2..5]->(b) RETURN a, b");
        if let LogicalOp::Project { input, .. } = &root {
            if let LogicalOp::Traverse { length, .. } = input.as_ref() {
                let lb = length.unwrap();
                assert_eq!(lb.min, Some(2));
                assert_eq!(lb.max, Some(5));
            } else {
                panic!("expected Traverse");
            }
        }
    }

    // -- MATCH + SET + RETURN --

    #[test]
    fn match_set_return() {
        let root = plan_root("MATCH (n:User {id: $id}) SET n.name = $name RETURN n");
        // Project → Update → NodeScan(with filter)
        if let LogicalOp::Project { input, .. } = &root {
            assert!(matches!(**input, LogicalOp::Update { .. }));
        }
    }

    // --- Compound WHERE predicate splitting tests ---

    /// Compound AND with vector + text produces VectorFilter → TextFilter pipeline.
    #[test]
    fn compound_where_vector_and_text() {
        let q = parse(
            "MATCH (n:Doc) \
             WHERE vector_distance(n.embedding, [1.0, 0.0]) < 0.5 \
               AND text_match(n.body, 'hello world') \
             RETURN n",
        )
        .unwrap();
        let plan = build_logical_plan(&q).unwrap();
        let explain = plan.explain();
        assert!(
            explain.contains("VectorFilter"),
            "compound AND should produce VectorFilter: {explain}"
        );
        assert!(
            explain.contains("TextFilter"),
            "compound AND should produce TextFilter: {explain}"
        );
    }

    /// Compound AND with vector + property filter splits correctly.
    #[test]
    fn compound_where_vector_and_property() {
        let q = parse(
            "MATCH (n:Product) \
             WHERE vector_distance(n.embedding, [0.1, 0.2]) < 0.3 \
               AND n.price > 100 \
             RETURN n",
        )
        .unwrap();
        let plan = build_logical_plan(&q).unwrap();
        let explain = plan.explain();
        assert!(
            explain.contains("VectorFilter"),
            "should extract VectorFilter from compound AND: {explain}"
        );
        assert!(
            explain.contains("Filter"),
            "remaining property predicate should be in Filter: {explain}"
        );
    }

    /// Triple compound: vector + text + property.
    #[test]
    fn compound_where_triple() {
        let q = parse(
            "MATCH (d:Document) \
             WHERE vector_distance(d.embedding, [0.5]) < 0.4 \
               AND text_match(d.body, 'transformer') \
               AND d.published = true \
             RETURN d",
        )
        .unwrap();
        let plan = build_logical_plan(&q).unwrap();
        let explain = plan.explain();
        assert!(
            explain.contains("VectorFilter"),
            "triple compound: VectorFilter expected: {explain}"
        );
        assert!(
            explain.contains("TextFilter"),
            "triple compound: TextFilter expected: {explain}"
        );
        assert!(
            explain.contains("Filter"),
            "triple compound: generic Filter for property expected: {explain}"
        );
    }

    /// Single vector predicate (no AND) still works.
    #[test]
    fn single_vector_predicate_no_regression() {
        let q = parse(
            "MATCH (n:Item) \
             WHERE vector_distance(n.vec, [1.0]) < 0.5 \
             RETURN n",
        )
        .unwrap();
        let plan = build_logical_plan(&q).unwrap();
        assert!(plan.explain().contains("VectorFilter"));
    }

    /// Single text predicate (no AND) still works.
    #[test]
    fn single_text_predicate_no_regression() {
        let q = parse(
            "MATCH (n:Doc) \
             WHERE text_match(n.body, 'search query') \
             RETURN n",
        )
        .unwrap();
        let plan = build_logical_plan(&q).unwrap();
        assert!(plan.explain().contains("TextFilter"));
    }

    /// Plain property filter (no vector/text) unchanged.
    #[test]
    fn plain_property_filter_no_regression() {
        let q = parse("MATCH (n:User) WHERE n.age > 25 RETURN n").unwrap();
        let plan = build_logical_plan(&q).unwrap();
        let explain = plan.explain();
        assert!(explain.contains("Filter"), "property filter: {explain}");
        assert!(
            !explain.contains("VectorFilter"),
            "no vector in plain filter: {explain}"
        );
    }

    // -- G024: cross-MATCH predicate lifting --

    /// MATCH (a) MATCH (b) WHERE a.x = b.y
    /// The Filter should be ABOVE CartesianProduct, not inside the right branch.
    #[test]
    fn cross_match_where_lifted_above_cartesian_product() {
        let explain =
            plan("MATCH (a:User) MATCH (b:Post) WHERE a.name = b.author RETURN a, b").explain();
        // Filter must wrap CartesianProduct, not be inside it.
        // EXPLAIN format: Filter wraps CartesianProduct(left, right).
        assert!(
            explain.contains("Filter"),
            "cross-match WHERE should produce Filter: {explain}"
        );
        assert!(
            explain.contains("CartesianProduct"),
            "multi-MATCH should produce CartesianProduct: {explain}"
        );
    }

    /// MATCH (a) MATCH (b) WHERE b.x = 5 (local only — no lifting needed).
    #[test]
    fn multi_match_local_where_stays_in_branch() {
        let explain =
            plan("MATCH (a:User) MATCH (b:Post) WHERE b.published = true RETURN a, b").explain();
        assert!(
            explain.contains("Filter"),
            "local WHERE should produce Filter: {explain}"
        );
    }

    /// collect_expr_variables extracts variables from simple and nested expressions.
    #[test]
    fn collect_expr_variables_basic() {
        let vars = collect_expr_variables(&Expr::Variable("a".into()));
        assert_eq!(vars, vec!["a"]);
    }

    #[test]
    fn collect_expr_variables_property_access() {
        let expr = Expr::PropertyAccess {
            expr: Box::new(Expr::Variable("n".into())),
            property: "name".into(),
        };
        let vars = collect_expr_variables(&expr);
        assert_eq!(vars, vec!["n"]);
    }

    #[test]
    fn collect_expr_variables_binary_op() {
        let expr = Expr::BinaryOp {
            left: Box::new(Expr::PropertyAccess {
                expr: Box::new(Expr::Variable("a".into())),
                property: "x".into(),
            }),
            op: BinaryOperator::Eq,
            right: Box::new(Expr::PropertyAccess {
                expr: Box::new(Expr::Variable("b".into())),
                property: "y".into(),
            }),
        };
        let vars = collect_expr_variables(&expr);
        assert_eq!(vars, vec!["a", "b"]);
    }

    #[test]
    fn collect_pattern_variables_basic() {
        let patterns = vec![Pattern {
            elements: vec![
                PatternElement::Node(NodePattern {
                    variable: Some("a".into()),
                    labels: vec!["User".into()],
                    properties: vec![],
                }),
                PatternElement::Relationship(RelationshipPattern {
                    variable: Some("r".into()),
                    rel_types: vec!["KNOWS".into()],
                    direction: Direction::Outgoing,
                    length: None,
                    properties: vec![],
                }),
                PatternElement::Node(NodePattern {
                    variable: Some("b".into()),
                    labels: vec![],
                    properties: vec![],
                }),
            ],
        }];
        let vars = collect_pattern_variables(&patterns);
        assert_eq!(vars, vec!["a", "r", "b"]);
    }

    // -- G029: ALTER LABEL SET SCHEMA --

    #[test]
    fn alter_label_parses() {
        let q = parse("ALTER LABEL User SET SCHEMA VALIDATED").unwrap();
        assert_eq!(q.clauses.len(), 1);
        if let Clause::AlterLabel(ref ac) = q.clauses[0] {
            assert_eq!(ac.label, "User");
            assert_eq!(ac.mode, "validated");
        } else {
            panic!("expected AlterLabel, got: {:?}", q.clauses[0]);
        }
    }

    #[test]
    fn alter_label_flexible() {
        let q = parse("ALTER LABEL Config SET SCHEMA FLEXIBLE").unwrap();
        if let Clause::AlterLabel(ref ac) = q.clauses[0] {
            assert_eq!(ac.label, "Config");
            assert_eq!(ac.mode, "flexible");
        } else {
            panic!("expected AlterLabel");
        }
    }

    #[test]
    fn alter_label_strict() {
        let q = parse("ALTER LABEL Product SET SCHEMA STRICT").unwrap();
        if let Clause::AlterLabel(ref ac) = q.clauses[0] {
            assert_eq!(ac.mode, "strict");
        } else {
            panic!("expected AlterLabel");
        }
    }

    #[test]
    fn alter_label_case_insensitive() {
        let q = parse("alter label User set schema validated").unwrap();
        if let Clause::AlterLabel(ref ac) = q.clauses[0] {
            assert_eq!(ac.label, "User");
            assert_eq!(ac.mode, "validated");
        } else {
            panic!("expected AlterLabel");
        }
    }

    #[test]
    fn alter_label_plan_produces_alter_label_op() {
        let explain = plan("ALTER LABEL User SET SCHEMA VALIDATED").explain();
        assert!(
            explain.contains("AlterLabel"),
            "EXPLAIN should show AlterLabel: {explain}"
        );
        assert!(
            explain.contains("User"),
            "EXPLAIN should show label name: {explain}"
        );
    }

    // --- VectorTopK optimization tests ---

    /// Find VectorTopK in a plan tree (depth-first). Returns a borrowed reference.
    fn find_vector_top_k(op: &LogicalOp) -> Option<&LogicalOp> {
        if matches!(op, LogicalOp::VectorTopK { .. }) {
            return Some(op);
        }
        match op {
            LogicalOp::Project { input, .. }
            | LogicalOp::Filter { input, .. }
            | LogicalOp::Sort { input, .. }
            | LogicalOp::Limit { input, .. }
            | LogicalOp::Skip { input, .. }
            | LogicalOp::Aggregate { input, .. }
            | LogicalOp::Unwind { input, .. } => find_vector_top_k(input),
            LogicalOp::CartesianProduct { left, right }
            | LogicalOp::LeftOuterJoin { left, right } => {
                find_vector_top_k(left).or_else(|| find_vector_top_k(right))
            }
            _ => None,
        }
    }

    /// Pattern A: direct function call in ORDER BY.
    /// `ORDER BY vector_distance(n.emb, $qv)` without an alias.
    #[test]
    fn vector_top_k_pattern_a_direct_call() {
        let root = plan_root(
            "MATCH (n:Doc) \
             RETURN n \
             ORDER BY vector_distance(n.embedding, [1.0, 0.0]) \
             LIMIT 5",
        );
        let top_k = find_vector_top_k(&root).expect("VectorTopK not found in plan");
        if let LogicalOp::VectorTopK {
            function,
            k,
            distance_alias,
            ..
        } = top_k
        {
            assert_eq!(function, "vector_distance");
            assert_eq!(*k, 5);
            assert!(
                distance_alias.is_none(),
                "pattern A has no alias: {distance_alias:?}"
            );
        } else {
            panic!("expected VectorTopK");
        }
    }

    /// Pattern B: alias via WITH before ORDER BY.
    /// This is the shape that VectorServiceImpl generates.
    #[test]
    fn vector_top_k_pattern_b_alias_via_with() {
        let root = plan_root(
            "MATCH (n:Doc) \
             WITH n, vector_distance(n.embedding, [1.0, 0.0]) AS _dist \
             ORDER BY _dist \
             LIMIT 10 \
             RETURN n, _dist",
        );
        let top_k = find_vector_top_k(&root).expect("VectorTopK not found in plan");
        if let LogicalOp::VectorTopK {
            function,
            k,
            distance_alias,
            ..
        } = top_k
        {
            assert_eq!(function, "vector_distance");
            assert_eq!(*k, 10);
            assert_eq!(distance_alias.as_deref(), Some("_dist"));
        } else {
            panic!("expected VectorTopK");
        }
    }

    /// No LIMIT → no optimization (Sort remains).
    #[test]
    fn vector_top_k_no_limit_not_optimized() {
        let root = plan_root(
            "MATCH (n:Doc) \
             RETURN n \
             ORDER BY vector_distance(n.embedding, [1.0, 0.0])",
        );
        assert!(
            find_vector_top_k(&root).is_none(),
            "ORDER BY without LIMIT should not produce VectorTopK: {root:?}"
        );
    }

    /// ORDER BY DESC on distance function → not optimized (wrong direction).
    #[test]
    fn vector_top_k_wrong_direction_not_optimized() {
        let root = plan_root(
            "MATCH (n:Doc) \
             RETURN n \
             ORDER BY vector_distance(n.embedding, [1.0, 0.0]) DESC \
             LIMIT 5",
        );
        assert!(
            find_vector_top_k(&root).is_none(),
            "DESC on vector_distance should not produce VectorTopK: {root:?}"
        );
    }

    /// vector_similarity uses DESC (higher is better) — this IS valid.
    #[test]
    fn vector_top_k_similarity_desc_optimized() {
        let root = plan_root(
            "MATCH (n:Doc) \
             RETURN n \
             ORDER BY vector_similarity(n.embedding, [1.0, 0.0]) DESC \
             LIMIT 3",
        );
        let top_k = find_vector_top_k(&root).expect("VectorTopK not found in plan");
        if let LogicalOp::VectorTopK { function, k, .. } = top_k {
            assert_eq!(function, "vector_similarity");
            assert_eq!(*k, 3);
        }
    }

    /// Non-vector ORDER BY → not optimized.
    #[test]
    fn vector_top_k_non_vector_order_not_optimized() {
        let root = plan_root("MATCH (n:User) RETURN n ORDER BY n.age LIMIT 5");
        assert!(
            find_vector_top_k(&root).is_none(),
            "ORDER BY non-vector should not produce VectorTopK"
        );
    }

    /// Multi-item ORDER BY → not optimized (only single-key ORDER BY supported).
    #[test]
    fn vector_top_k_multi_item_order_not_optimized() {
        let root = plan_root(
            "MATCH (n:Doc) \
             RETURN n \
             ORDER BY vector_distance(n.embedding, [1.0, 0.0]), n.title \
             LIMIT 5",
        );
        assert!(
            find_vector_top_k(&root).is_none(),
            "multi-item ORDER BY should not produce VectorTopK"
        );
    }

    /// Pattern A with RETURN that drops the vector variable → NOT optimized.
    /// `RETURN n.name` removes `n.embedding` from intermediate rows, so the
    /// optimizer must fall back to plain Sort+Limit. This guards against the
    /// regression found by g009_forced_offload_cypher_e2e integration test.
    #[test]
    fn vector_top_k_pattern_a_dropped_variable_not_optimized() {
        let root = plan_root(
            "MATCH (n:Doc) \
             RETURN n.name \
             ORDER BY vector_distance(n.embedding, [1.0, 0.0]) \
             LIMIT 5",
        );
        assert!(
            find_vector_top_k(&root).is_none(),
            "Pattern A must skip optimization when Project drops the vector variable: {root:?}"
        );
    }

    /// Pattern A with RETURN * preserves all columns → optimized.
    #[test]
    fn vector_top_k_pattern_a_return_star_optimized() {
        let root = plan_root(
            "MATCH (n:Doc) \
             RETURN * \
             ORDER BY vector_distance(n.embedding, [1.0, 0.0]) \
             LIMIT 5",
        );
        assert!(
            find_vector_top_k(&root).is_some(),
            "RETURN * preserves all columns — should still optimize: {root:?}"
        );
    }

    /// Pattern A with RETURN n (full node ref) keeps n.embedding accessible → optimized.
    #[test]
    fn vector_top_k_pattern_a_return_node_optimized() {
        let root = plan_root(
            "MATCH (n:Doc) \
             RETURN n \
             ORDER BY vector_distance(n.embedding, [1.0, 0.0]) \
             LIMIT 5",
        );
        assert!(
            find_vector_top_k(&root).is_some(),
            "RETURN n preserves the full node — should optimize: {root:?}"
        );
    }

    /// SKIP between Limit and Sort: `ORDER BY d SKIP 5 LIMIT 10` produces
    /// `Limit { Skip { Sort { ... } } }`. The optimizer must NOT apply —
    /// Pattern A/B requires Sort directly under Limit. Fallback Sort+Skip+Limit
    /// is correct, but VectorTopK cannot express skip semantics.
    #[test]
    fn vector_top_k_skip_limit_not_optimized() {
        let root = plan_root(
            "MATCH (n:Doc) \
             WITH n, vector_distance(n.embedding, [1.0, 0.0]) AS d \
             ORDER BY d \
             SKIP 5 \
             LIMIT 10 \
             RETURN n, d",
        );
        assert!(
            find_vector_top_k(&root).is_none(),
            "SKIP between Sort and Limit should not produce VectorTopK: {root:?}"
        );
    }

    /// Nested WITH: two sequential WITH clauses with vector_distance. The outer
    /// WITH removes the vector variable, so Pattern B's guard (`project_preserves_vector_expr`)
    /// should still allow optimization because inner WITH projects the vector
    /// via alias which carries through.
    #[test]
    fn vector_top_k_nested_with_still_optimized() {
        let root = plan_root(
            "MATCH (n:Doc) \
             WITH n, n.title AS title \
             WITH n, title, vector_distance(n.embedding, [1.0, 0.0]) AS d \
             ORDER BY d \
             LIMIT 3 \
             RETURN title, d",
        );
        // Pattern B should match the innermost `WITH ... AS d` + ORDER BY + LIMIT.
        assert!(
            find_vector_top_k(&root).is_some(),
            "Nested WITH with Pattern B should still optimize: {root:?}"
        );
    }

    /// Parameter-passed query vector (via `$qv` parameter) — the typical shape
    /// generated by the Python SDK. Pattern B alias + parameter must optimize.
    #[test]
    fn vector_top_k_with_parameter_query_vector() {
        let root = plan_root(
            "MATCH (n:Doc) \
             WITH n, vector_distance(n.embedding, $qv) AS d \
             ORDER BY d \
             LIMIT 10 \
             RETURN n, d",
        );
        let top_k =
            find_vector_top_k(&root).expect("VectorTopK not found when query vector is parameter");
        if let LogicalOp::VectorTopK {
            query_vector,
            distance_alias,
            k,
            ..
        } = top_k
        {
            // query_vector must be the parameter expression (not substituted).
            assert!(
                matches!(query_vector, Expr::Parameter(_)),
                "query_vector should be Parameter, got {query_vector:?}"
            );
            assert_eq!(*k, 10);
            assert_eq!(distance_alias.as_deref(), Some("d"));
        } else {
            panic!("expected VectorTopK");
        }
    }

    /// EXPLAIN output for VectorTopK shows function and k.
    #[test]
    fn vector_top_k_explain_output() {
        let p = plan(
            "MATCH (n:Doc) \
             WITH n, vector_distance(n.embedding, [1.0, 0.0]) AS d \
             ORDER BY d \
             LIMIT 7 \
             RETURN n",
        );
        let explain = p.explain();
        assert!(
            explain.contains("VectorTopK"),
            "EXPLAIN should show VectorTopK: {explain}"
        );
        assert!(
            explain.contains("k=7"),
            "EXPLAIN should show k=7: {explain}"
        );
        assert!(
            explain.contains("vector_distance"),
            "EXPLAIN should show function: {explain}"
        );
    }
}
