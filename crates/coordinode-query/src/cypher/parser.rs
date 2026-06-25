//! Cypher parser: query string → typed AST.
//!
//! Uses pest PEG grammar (`cypher.pest`) to produce a parse tree,
//! then converts it into the typed AST defined in `ast.rs`.

use coordinode_core::graph::types::Value;
use pest::iterators::{Pair, Pairs};
use pest::Parser;

use super::ast::*;
use super::errors::{ErrorSpan, ParseError};

#[derive(pest_derive::Parser)]
#[grammar = "cypher/cypher.pest"]
struct CypherParser;

/// Parse a Cypher query string into a typed AST.
pub fn parse(input: &str) -> Result<Query, ParseError> {
    // Extract /*+ hint */ comments before PEG parsing (PEG strips all /* */ as COMMENT).
    let (cleaned, hints) = extract_query_hints(input);

    let pairs = CypherParser::parse(Rule::query, &cleaned).map_err(|e| {
        let (line, col) = match e.line_col {
            pest::error::LineColLocation::Pos(pos) => pos,
            pest::error::LineColLocation::Span(start, _) => start,
        };
        let (start, end) = match e.location {
            pest::error::InputLocation::Pos(p) => (p, p),
            pest::error::InputLocation::Span(s) => s,
        };
        ParseError::Syntax {
            message: e.to_string(),
            span: ErrorSpan {
                start,
                end,
                line,
                col,
            },
        }
    })?;

    let mut query = build_query(pairs)?;
    query.hints = hints;
    Ok(query)
}

/// Extract `/*+ key('value') */` hints from a query string.
///
/// Returns the cleaned query (hints replaced with whitespace) and a list
/// of parsed hints. Unknown hint keys are silently ignored.
fn extract_query_hints(input: &str) -> (String, Vec<QueryHint>) {
    let mut hints = Vec::new();
    let mut result = input.to_string();

    // Find all /*+ ... */ patterns.
    while let Some(start) = result.find("/*+") {
        if let Some(rel_end) = result[start..].find("*/") {
            let end = start + rel_end + 2;
            let body = result[start + 3..start + rel_end].trim();

            if let Some(hint) = parse_single_hint(body) {
                hints.push(hint);
            }

            // Replace hint with spaces to preserve character positions for error reporting.
            let replacement = " ".repeat(end - start);
            result.replace_range(start..end, &replacement);
        } else {
            break; // Unterminated hint — let PEG report the error
        }
    }

    (result, hints)
}

/// Parse a single hint body like `vector_consistency('snapshot')`.
fn parse_single_hint(body: &str) -> Option<QueryHint> {
    // Split on '(' to get key and value
    let (key, rest) = body.split_once('(')?;
    let key = key.trim();
    let value = rest.strip_suffix(')')?.trim();

    // Strip quotes from value
    let unquoted = if (value.starts_with('\'') && value.ends_with('\''))
        || (value.starts_with('"') && value.ends_with('"'))
    {
        &value[1..value.len() - 1]
    } else {
        value
    };

    match key {
        "vector_consistency" => {
            let mode =
                coordinode_core::graph::types::VectorConsistencyMode::from_str_opt(unquoted)?;
            Some(QueryHint::VectorConsistency(mode))
        }
        "read_consistency" => {
            let mode = coordinode_core::txn::read_consistency::ReadConsistencyMode::from_str_opt(
                unquoted,
            )?;
            Some(QueryHint::ReadConsistency(mode))
        }
        _ => None, // Unknown hint — silently ignored
    }
}

fn build_query(pairs: Pairs<'_, Rule>) -> Result<Query, ParseError> {
    let mut clauses = Vec::new();
    let mut unions = Vec::new();

    for pair in pairs {
        if pair.as_rule() == Rule::query {
            // Walk the top-level `query` node directly so UNION branches are
            // separated from the leading single query.
            for inner in pair.into_inner() {
                match inner.as_rule() {
                    Rule::union_branch => unions.push(build_union_branch(inner)?),
                    Rule::EOI => {}
                    _ => collect_clauses(inner, &mut clauses)?,
                }
            }
        } else {
            collect_clauses(pair, &mut clauses)?;
        }
    }

    if clauses.is_empty() {
        return Err(ParseError::Invalid("empty query".into()));
    }

    Ok(Query {
        clauses,
        hints: Vec::new(),
        unions,
    })
}

/// Build one `UNION [ALL]` branch from a `union_branch` parse node.
fn build_union_branch(pair: Pair<'_, Rule>) -> Result<UnionBranch, ParseError> {
    let mut all = false;
    let mut clauses = Vec::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::kw_union => {}
            Rule::kw_all => all = true,
            _ => collect_clauses(inner, &mut clauses)?,
        }
    }

    if clauses.is_empty() {
        return Err(ParseError::Invalid("empty UNION branch".into()));
    }

    Ok(UnionBranch { all, clauses })
}

/// Recursively collect clause nodes from the parse tree.
fn collect_clauses(pair: Pair<'_, Rule>, clauses: &mut Vec<Clause>) -> Result<(), ParseError> {
    match pair.as_rule() {
        Rule::query => {
            for inner in pair.into_inner() {
                collect_clauses(inner, clauses)?;
            }
        }
        Rule::clause => {
            let inner = first_inner(pair)?;
            build_clause(inner, clauses)?;
        }
        Rule::EOI => {}
        _ => {}
    }
    Ok(())
}

/// Build clause(s) from a parsed clause pair.
/// RETURN clause may produce multiple AST clauses (RETURN + ORDER BY + SKIP + LIMIT).
fn build_clause(pair: Pair<'_, Rule>, clauses: &mut Vec<Clause>) -> Result<(), ParseError> {
    match pair.as_rule() {
        Rule::match_clause => {
            let mc = build_match_clause(pair)?;
            clauses.push(Clause::Match(mc));
        }
        Rule::optional_match_clause => {
            let mc = build_match_clause(pair)?;
            clauses.push(Clause::OptionalMatch(mc));
        }
        Rule::where_clause => {
            let expr = find_expression(pair)?;
            clauses.push(Clause::Where(expr));
        }
        Rule::return_clause => {
            build_return_with_parts(pair, clauses, false)?;
        }
        Rule::with_clause => {
            build_return_with_parts(pair, clauses, true)?;
        }
        Rule::unwind_clause => {
            let uc = build_unwind_clause(pair)?;
            clauses.push(Clause::Unwind(uc));
        }
        Rule::as_of_timestamp_clause => {
            let expr = find_expression(pair)?;
            clauses.push(Clause::AsOfTimestamp(expr));
        }
        Rule::call_clause => {
            let cc = build_call_clause(pair)?;
            clauses.push(Clause::Call(cc));
        }
        Rule::alter_label_clause => {
            let ac = build_alter_label_clause(pair)?;
            clauses.push(Clause::AlterLabel(ac));
        }
        Rule::create_text_index_clause => {
            let c = build_create_text_index_clause(pair)?;
            clauses.push(Clause::CreateTextIndex(c));
        }
        Rule::drop_text_index_clause => {
            let c = build_drop_text_index_clause(pair)?;
            clauses.push(Clause::DropTextIndex(c));
        }
        Rule::create_encrypted_index_clause => {
            let c = build_create_encrypted_index_clause(pair)?;
            clauses.push(Clause::CreateEncryptedIndex(c));
        }
        Rule::drop_encrypted_index_clause => {
            let c = build_drop_encrypted_index_clause(pair)?;
            clauses.push(Clause::DropEncryptedIndex(c));
        }
        Rule::create_index_clause => {
            let c = build_create_index_clause(pair)?;
            clauses.push(Clause::CreateIndex(c));
        }
        Rule::drop_index_clause => {
            let c = build_drop_index_clause(pair)?;
            clauses.push(Clause::DropIndex(c));
        }
        Rule::create_vector_index_clause => {
            let c = build_create_vector_index_clause(pair)?;
            clauses.push(Clause::CreateVectorIndex(c));
        }
        Rule::drop_vector_index_clause => {
            let c = build_drop_vector_index_clause(pair)?;
            clauses.push(Clause::DropVectorIndex(c));
        }
        Rule::create_edge_type_clause => {
            let c = build_create_edge_type_clause(pair)?;
            clauses.push(Clause::CreateEdgeType(c));
        }
        Rule::create_node_type_clause => {
            let c = build_create_node_type_clause(pair)?;
            clauses.push(Clause::CreateNodeType(c));
        }
        Rule::create_trigger_clause => {
            let c = build_create_trigger_clause(pair)?;
            clauses.push(Clause::CreateTrigger(c));
        }
        Rule::drop_trigger_clause => {
            let c = build_drop_trigger_clause(pair)?;
            clauses.push(Clause::DropTrigger(c));
        }
        Rule::show_triggers_clause => {
            clauses.push(Clause::ShowTriggers);
        }
        Rule::alter_trigger_clause => {
            let c = build_alter_trigger_clause(pair)?;
            clauses.push(Clause::AlterTrigger(c));
        }
        Rule::create_clause => {
            let cc = build_create_clause(pair)?;
            clauses.push(Clause::Create(cc));
        }
        Rule::foreach_clause => {
            let fc = build_foreach_clause(pair)?;
            clauses.push(Clause::Foreach(fc));
        }
        Rule::call_subquery_clause => {
            let cs = build_call_subquery_clause(pair)?;
            clauses.push(Clause::CallSubquery(cs));
        }
        Rule::merge_clause => {
            let mc = build_merge_clause(pair)?;
            clauses.push(Clause::Merge(mc));
        }
        Rule::merge_all_clause => {
            let mc = build_merge_clause(pair)?;
            clauses.push(Clause::MergeMany(mc));
        }
        Rule::merge_nodes_clause => {
            let mn = build_merge_nodes_clause(pair)?;
            clauses.push(Clause::MergeNodes(mn));
        }
        Rule::clone_node_clause => {
            let cn = build_clone_node_clause(pair)?;
            clauses.push(Clause::CloneNode(cn));
        }
        Rule::upsert_clause => {
            let uc = build_upsert_clause(pair)?;
            clauses.push(Clause::Upsert(uc));
        }
        Rule::delete_clause => {
            let dc = build_delete_clause(pair, false)?;
            clauses.push(Clause::Delete(dc));
        }
        Rule::detach_delete_clause => {
            let dc = build_delete_clause(pair, true)?;
            clauses.push(Clause::Delete(dc));
        }
        Rule::detach_document_clause => {
            let dd = build_detach_document_clause(pair)?;
            clauses.push(Clause::DetachDocument(dd));
        }
        Rule::attach_document_clause => {
            let ad = build_attach_document_clause(pair)?;
            clauses.push(Clause::AttachDocument(ad));
        }
        Rule::set_clause => {
            let (items, violation_mode) = build_set_clause(pair)?;
            clauses.push(Clause::Set(items, violation_mode));
        }
        Rule::remove_clause => {
            let items = build_remove_clause(pair)?;
            clauses.push(Clause::Remove(items));
        }
        _ => {
            return Err(ParseError::Invalid(format!(
                "unexpected clause: {:?}",
                pair.as_rule()
            )));
        }
    }
    Ok(())
}

fn build_match_clause(pair: Pair<'_, Rule>) -> Result<MatchClause, ParseError> {
    let mut patterns = Vec::new();
    let mut where_clause = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::match_pattern_list => {
                patterns = build_match_pattern_list(inner)?;
            }
            Rule::where_inline => {
                let expr = find_expression(inner)?;
                where_clause = Some(expr);
            }
            _ => {}
        }
    }

    Ok(MatchClause {
        patterns,
        where_clause,
    })
}

/// MATCH-only pattern list: each entry may carry a named-path assignment
/// (`p = ...`) and/or be wrapped in `shortestPath(...)`.
fn build_match_pattern_list(pair: Pair<'_, Rule>) -> Result<Vec<Pattern>, ParseError> {
    let mut patterns = Vec::new();
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::match_pattern {
            patterns.push(build_match_pattern(inner)?);
        }
    }
    Ok(patterns)
}

fn build_match_pattern(pair: Pair<'_, Rule>) -> Result<Pattern, ParseError> {
    let mut path_variable: Option<String> = None;
    let mut shortest_path = false;
    let mut pattern: Option<Pattern> = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::path_assign => {
                // `p =` : the variable is the only token inside.
                for v in inner.into_inner() {
                    if v.as_rule() == Rule::variable {
                        path_variable = Some(v.as_str().to_string());
                    }
                }
            }
            Rule::shortest_path_pattern => {
                shortest_path = true;
                for p in inner.into_inner() {
                    if p.as_rule() == Rule::pattern {
                        pattern = Some(build_pattern(p)?);
                    }
                }
            }
            Rule::pattern => {
                pattern = Some(build_pattern(inner)?);
            }
            _ => {}
        }
    }

    let mut pattern =
        pattern.ok_or_else(|| ParseError::Invalid("empty MATCH pattern".to_string()))?;
    pattern.path_variable = path_variable;
    pattern.shortest_path = shortest_path;
    Ok(pattern)
}

fn build_pattern_list(pair: Pair<'_, Rule>) -> Result<Vec<Pattern>, ParseError> {
    let mut patterns = Vec::new();
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::pattern {
            patterns.push(build_pattern(inner)?);
        }
    }
    Ok(patterns)
}

fn build_pattern(pair: Pair<'_, Rule>) -> Result<Pattern, ParseError> {
    let mut elements = Vec::new();
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::pattern_element {
            build_pattern_element(inner, &mut elements)?;
        }
    }
    Ok(Pattern {
        elements,
        path_variable: None,
        shortest_path: false,
    })
}

fn build_pattern_element(
    pair: Pair<'_, Rule>,
    elements: &mut Vec<PatternElement>,
) -> Result<(), ParseError> {
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::node_pattern => {
                elements.push(PatternElement::Node(build_node_pattern(inner)?));
            }
            Rule::relationship_pattern => {
                elements.push(PatternElement::Relationship(build_relationship_pattern(
                    inner,
                )?));
            }
            _ => {}
        }
    }
    Ok(())
}

fn build_node_pattern(pair: Pair<'_, Rule>) -> Result<NodePattern, ParseError> {
    let mut variable = None;
    let mut labels = Vec::new();
    let mut properties = Vec::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::variable => {
                if let Some(id) = inner.into_inner().next() {
                    let name = extract_identifier(id);
                    if !name.is_empty() {
                        variable = Some(name);
                    }
                }
            }
            Rule::label_list => {
                for label in inner.into_inner() {
                    if label.as_rule() == Rule::identifier {
                        labels.push(extract_identifier(label));
                    }
                }
            }
            Rule::property_map => {
                properties = build_property_map(inner)?;
            }
            _ => {}
        }
    }

    Ok(NodePattern {
        variable,
        labels,
        properties,
    })
}

fn build_relationship_pattern(pair: Pair<'_, Rule>) -> Result<RelationshipPattern, ParseError> {
    let mut variable = None;
    let mut rel_types = Vec::new();
    let mut length = None;
    let mut properties = Vec::new();

    // Determine direction from arrow tokens in the text.
    let text = pair.as_str();
    let has_left = text.starts_with('<');
    let has_right = text.ends_with('>');

    let direction = match (has_left, has_right) {
        (true, false) => Direction::Incoming,
        (false, true) => Direction::Outgoing,
        (true, true) => Direction::Both,
        (false, false) => Direction::Both,
    };

    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::rel_detail {
            for detail in inner.into_inner() {
                match detail.as_rule() {
                    Rule::variable => {
                        if let Some(id) = detail.into_inner().next() {
                            variable = Some(extract_identifier(id));
                        }
                    }
                    Rule::rel_type_list => {
                        for t in detail.into_inner() {
                            if t.as_rule() == Rule::identifier {
                                rel_types.push(extract_identifier(t));
                            }
                        }
                    }
                    Rule::length_spec => {
                        length = Some(build_length_spec(detail)?);
                    }
                    Rule::property_map => {
                        properties = build_property_map(detail)?;
                    }
                    _ => {}
                }
            }
        }
    }

    Ok(RelationshipPattern {
        variable,
        rel_types,
        direction,
        length,
        properties,
    })
}

fn build_length_spec(pair: Pair<'_, Rule>) -> Result<LengthBound, ParseError> {
    let mut min = None;
    let mut max = None;

    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::length_range {
            let text = inner.as_str().trim();

            if let Some((left, right)) = text.split_once("..") {
                min = if left.is_empty() {
                    None
                } else {
                    Some(parse_u64(left)?)
                };
                max = if right.is_empty() {
                    None
                } else {
                    Some(parse_u64(right)?)
                };
            } else {
                let n = parse_u64(text)?;
                min = Some(n);
                max = Some(n);
            }
        }
    }

    Ok(LengthBound { min, max })
}

fn build_property_map(pair: Pair<'_, Rule>) -> Result<Vec<(String, Expr)>, ParseError> {
    let mut props = Vec::new();
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::property_pair {
            let mut children = inner.into_inner();
            let key = extract_identifier(
                children
                    .next()
                    .ok_or_else(|| ParseError::Invalid("missing property key".into()))?,
            );
            let value = build_expression(
                children
                    .next()
                    .ok_or_else(|| ParseError::Invalid("missing property value".into()))?,
            )?;
            props.push((key, value));
        }
    }
    Ok(props)
}

// --- RETURN / WITH ---
// Both RETURN and WITH have the same body structure (items + ORDER BY + SKIP + LIMIT).
// The grammar embeds ORDER BY, SKIP, LIMIT into the return/with clause per OpenCypher spec.
// We decompose them into separate AST clauses for the planner.

fn build_return_with_parts(
    pair: Pair<'_, Rule>,
    clauses: &mut Vec<Clause>,
    is_with: bool,
) -> Result<(), ParseError> {
    let mut distinct = false;
    let mut items = Vec::new();
    let mut order_items = None;
    let mut skip_expr = None;
    let mut limit_expr = None;
    let mut where_clause = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::distinct_kw => distinct = true,
            Rule::return_items => {
                items = build_return_items(inner)?;
            }
            Rule::order_by_part => {
                order_items = Some(build_order_by(inner)?);
            }
            Rule::skip_part => {
                skip_expr = Some(find_expression(inner)?);
            }
            Rule::limit_part => {
                limit_expr = Some(find_expression(inner)?);
            }
            Rule::where_inline => {
                let expr = find_expression(inner)?;
                where_clause = Some(expr);
            }
            _ => {}
        }
    }

    if is_with {
        clauses.push(Clause::With(WithClause {
            distinct,
            items,
            where_clause,
        }));
    } else {
        clauses.push(Clause::Return(ReturnClause { distinct, items }));
    }

    if let Some(items) = order_items {
        clauses.push(Clause::OrderBy(items));
    }
    if let Some(expr) = skip_expr {
        clauses.push(Clause::Skip(expr));
    }
    if let Some(expr) = limit_expr {
        clauses.push(Clause::Limit(expr));
    }

    Ok(())
}

fn build_return_items(pair: Pair<'_, Rule>) -> Result<Vec<ReturnItem>, ParseError> {
    let mut items = Vec::new();
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::return_item {
            items.push(build_return_item(inner)?);
        }
    }
    Ok(items)
}

fn build_return_item(pair: Pair<'_, Rule>) -> Result<ReturnItem, ParseError> {
    let mut expr = None;
    let mut alias = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::star => {
                expr = Some(Expr::Star);
            }
            Rule::expression => {
                expr = Some(build_expression(inner)?);
            }
            Rule::alias => {
                for a in inner.into_inner() {
                    if a.as_rule() == Rule::identifier {
                        alias = Some(extract_identifier(a));
                    }
                }
            }
            _ => {
                if expr.is_none() {
                    expr = Some(build_expression(inner)?);
                }
            }
        }
    }

    Ok(ReturnItem {
        expr: expr.ok_or_else(|| ParseError::Invalid("empty return item".into()))?,
        alias,
    })
}

fn build_unwind_clause(pair: Pair<'_, Rule>) -> Result<UnwindClause, ParseError> {
    let mut expr = None;
    let mut variable = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::expression => {
                expr = Some(build_expression(inner)?);
            }
            Rule::identifier => {
                variable = Some(extract_identifier(inner));
            }
            _ => {}
        }
    }

    Ok(UnwindClause {
        expr: expr.ok_or_else(|| ParseError::Invalid("missing UNWIND expression".into()))?,
        variable: variable
            .ok_or_else(|| ParseError::Invalid("missing UNWIND AS variable".into()))?,
    })
}

fn build_call_clause(pair: Pair<'_, Rule>) -> Result<CallClause, ParseError> {
    let mut procedure = String::new();
    let mut args = Vec::new();
    let mut yield_items = Vec::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::procedure_name => {
                // Dotted name: db.advisor.suggestions
                let parts: Vec<String> = inner
                    .into_inner()
                    .filter(|p| p.as_rule() == Rule::identifier)
                    .map(extract_identifier)
                    .collect();
                procedure = parts.join(".");
            }
            Rule::procedure_args => {
                for arg in inner.into_inner() {
                    if arg.as_rule() == Rule::expression {
                        args.push(build_expression(arg)?);
                    }
                }
            }
            Rule::yield_clause => {
                for yield_inner in inner.into_inner() {
                    if yield_inner.as_rule() == Rule::yield_items {
                        for item in yield_inner.into_inner() {
                            if item.as_rule() == Rule::yield_item {
                                let mut name = String::new();
                                let mut alias = None;
                                for yi in item.into_inner() {
                                    match yi.as_rule() {
                                        Rule::identifier if name.is_empty() => {
                                            name = extract_identifier(yi);
                                        }
                                        Rule::alias => {
                                            for a in yi.into_inner() {
                                                if a.as_rule() == Rule::identifier {
                                                    alias = Some(extract_identifier(a));
                                                }
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                                yield_items.push(YieldItem { name, alias });
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    if procedure.is_empty() {
        return Err(ParseError::Invalid("missing procedure name in CALL".into()));
    }

    Ok(CallClause {
        procedure,
        args,
        yield_items,
    })
}

fn build_order_by(pair: Pair<'_, Rule>) -> Result<Vec<SortItem>, ParseError> {
    let mut items = Vec::new();
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::sort_item {
            items.push(build_sort_item(inner)?);
        }
    }
    Ok(items)
}

fn build_sort_item(pair: Pair<'_, Rule>) -> Result<SortItem, ParseError> {
    let mut expr = None;
    let mut ascending = true;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::sort_direction => {
                let dir = inner.as_str().to_uppercase();
                ascending = dir.starts_with("ASC");
            }
            Rule::expression => {
                expr = Some(build_expression(inner)?);
            }
            _ => {
                if expr.is_none() {
                    expr = Some(build_expression(inner)?);
                }
            }
        }
    }

    Ok(SortItem {
        expr: expr.ok_or_else(|| ParseError::Invalid("empty sort item".into()))?,
        ascending,
    })
}

// --- DDL clauses ---

fn build_alter_label_clause(pair: Pair<'_, Rule>) -> Result<AlterLabelClause, ParseError> {
    let mut label = String::new();
    let mut mode = String::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::identifier => {
                label = inner.as_str().to_string();
            }
            Rule::schema_mode => {
                mode = inner.as_str().to_lowercase();
            }
            _ => {}
        }
    }

    if label.is_empty() {
        return Err(ParseError::Invalid(
            "ALTER LABEL requires label name".into(),
        ));
    }
    if mode.is_empty() {
        return Err(ParseError::Invalid(
            "ALTER LABEL SET SCHEMA requires mode (STRICT, VALIDATED, FLEXIBLE)".into(),
        ));
    }

    Ok(AlterLabelClause { label, mode })
}

fn build_create_text_index_clause(
    pair: Pair<'_, Rule>,
) -> Result<CreateTextIndexClause, ParseError> {
    let mut name = String::new();
    let mut label = String::new();
    let mut fields = Vec::new();
    let mut default_language = None;
    let mut language_override = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::identifier => {
                // First identifier = index name, second = label.
                if name.is_empty() {
                    name = inner.as_str().to_string();
                } else if label.is_empty() {
                    label = inner.as_str().to_string();
                }
            }
            Rule::text_index_single_field => {
                // Single field: (property) [LANGUAGE "..."]
                for sf_inner in inner.into_inner() {
                    match sf_inner.as_rule() {
                        Rule::identifier => {
                            fields.push(TextIndexFieldSpec {
                                property: sf_inner.as_str().to_string(),
                                analyzer: None,
                            });
                        }
                        Rule::text_index_language => {
                            // LANGUAGE "..." in simple syntax → default_language
                            default_language = extract_string_literal(&sf_inner);
                        }
                        _ => {}
                    }
                }
            }
            Rule::text_index_multi_field => {
                // Multi-field: { field: { analyzer: "..." }, ... }
                for field_def in inner.into_inner() {
                    if field_def.as_rule() == Rule::text_index_field_def {
                        let (prop, analyzer) = parse_field_def(field_def);
                        fields.push(TextIndexFieldSpec {
                            property: prop,
                            analyzer,
                        });
                    }
                }
            }
            Rule::text_index_default_language => {
                // DEFAULT LANGUAGE "..."
                default_language = extract_string_literal(&inner);
            }
            Rule::text_index_language_override => {
                // LANGUAGE OVERRIDE "..."
                language_override = extract_string_literal(&inner);
            }
            _ => {}
        }
    }

    if name.is_empty() || label.is_empty() || fields.is_empty() {
        return Err(ParseError::Invalid(
            "CREATE TEXT INDEX requires: name ON :Label(property) or name ON :Label { field: { analyzer: \"...\" } }".into(),
        ));
    }

    Ok(CreateTextIndexClause {
        name,
        label,
        fields,
        default_language,
        language_override,
    })
}

/// Extract a string literal value from a rule that contains a string_literal child.
fn extract_string_literal(pair: &Pair<'_, Rule>) -> Option<String> {
    for inner in pair.clone().into_inner() {
        if inner.as_rule() == Rule::string_literal {
            if let Some(content) = inner.into_inner().next() {
                return Some(content.as_str().to_string());
            }
        }
    }
    None
}

/// Parse a text_index_field_def: `identifier : { analyzer: "..." }`
fn parse_field_def(pair: Pair<'_, Rule>) -> (String, Option<String>) {
    let mut property = String::new();
    let mut analyzer = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::identifier => {
                property = inner.as_str().to_string();
            }
            Rule::text_field_option => {
                for opt in inner.into_inner() {
                    if opt.as_rule() == Rule::text_field_analyzer {
                        analyzer = extract_string_literal(&opt);
                    }
                }
            }
            _ => {}
        }
    }

    (property, analyzer)
}

fn build_drop_text_index_clause(pair: Pair<'_, Rule>) -> Result<DropTextIndexClause, ParseError> {
    let mut name = String::new();

    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::identifier {
            name = inner.as_str().to_string();
        }
    }

    if name.is_empty() {
        return Err(ParseError::Invalid(
            "DROP TEXT INDEX requires index name".into(),
        ));
    }

    Ok(DropTextIndexClause { name })
}

// --- Encrypted Index DDL (SSE, G017) ---

fn build_create_encrypted_index_clause(
    pair: Pair<'_, Rule>,
) -> Result<CreateEncryptedIndexClause, ParseError> {
    let mut identifiers = Vec::new();

    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::identifier {
            identifiers.push(inner.as_str().to_string());
        }
    }

    // Expected: index_name, label, property (3 identifiers after keywords)
    if identifiers.len() < 3 {
        return Err(ParseError::Invalid(
            "CREATE ENCRYPTED INDEX requires: name ON :Label(property)".into(),
        ));
    }

    Ok(CreateEncryptedIndexClause {
        name: identifiers[0].clone(),
        label: identifiers[1].clone(),
        property: identifiers[2].clone(),
    })
}

fn build_drop_encrypted_index_clause(
    pair: Pair<'_, Rule>,
) -> Result<DropEncryptedIndexClause, ParseError> {
    let mut name = String::new();

    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::identifier {
            name = inner.as_str().to_string();
        }
    }

    if name.is_empty() {
        return Err(ParseError::Invalid(
            "DROP ENCRYPTED INDEX requires index name".into(),
        ));
    }

    Ok(DropEncryptedIndexClause { name })
}

fn build_create_index_clause(pair: Pair<'_, Rule>) -> Result<CreateIndexClause, ParseError> {
    let mut unique = false;
    let mut sparse = false;
    let mut identifiers: Vec<String> = Vec::new();
    let mut filter_expr: Option<Expr> = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::kw_unique => unique = true,
            Rule::kw_sparse => sparse = true,
            Rule::identifier => identifiers.push(inner.as_str().to_string()),
            Rule::where_inline => {
                filter_expr = Some(find_expression(inner)?);
            }
            _ => {}
        }
    }

    if identifiers.len() < 3 {
        return Err(ParseError::Invalid(
            "CREATE INDEX requires name, label, and property".into(),
        ));
    }

    Ok(CreateIndexClause {
        name: identifiers[0].clone(),
        label: identifiers[1].clone(),
        property: identifiers[2].clone(),
        unique,
        sparse,
        filter_expr,
    })
}

fn build_create_edge_type_clause(pair: Pair<'_, Rule>) -> Result<CreateEdgeTypeClause, ParseError> {
    let mut name = String::new();
    let mut temporal = false;
    let mut properties: Vec<EdgePropertyDecl> = Vec::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::identifier if name.is_empty() => {
                name = inner.as_str().to_string();
            }
            Rule::kw_temporal => temporal = true,
            Rule::property_decl_list => {
                for decl in inner.into_inner() {
                    if decl.as_rule() == Rule::property_decl {
                        let mut prop_name = String::new();
                        let mut prop_type = String::new();
                        let mut not_null = false;
                        for part in decl.into_inner() {
                            match part.as_rule() {
                                Rule::identifier if prop_name.is_empty() => {
                                    prop_name = part.as_str().to_string();
                                }
                                Rule::property_type_name => {
                                    prop_type = part.as_str().to_uppercase();
                                }
                                Rule::property_decl_modifier => {
                                    not_null = true;
                                }
                                _ => {}
                            }
                        }
                        if !prop_name.is_empty() && !prop_type.is_empty() {
                            properties.push(EdgePropertyDecl {
                                name: prop_name,
                                type_name: prop_type,
                                not_null,
                            });
                        }
                    }
                }
            }
            _ => {}
        }
    }

    if name.is_empty() {
        return Err(ParseError::Invalid(
            "CREATE EDGE TYPE requires a type name".into(),
        ));
    }

    Ok(CreateEdgeTypeClause {
        name,
        temporal,
        properties,
    })
}

/// Build a `CREATE NODE TYPE … [TEMPORAL] [WITH (...)]` clause (R172a per
/// ADR-027). Mirror of `build_create_edge_type_clause` — the grammar shape
/// is identical apart from `kw_edge` → `kw_node`, so the inner walker is
/// structurally the same.
fn build_create_node_type_clause(
    pair: Pair<'_, Rule>,
) -> Result<crate::cypher::ast::CreateNodeTypeClause, ParseError> {
    let mut name = String::new();
    let mut temporal = false;
    let mut properties: Vec<EdgePropertyDecl> = Vec::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::identifier if name.is_empty() => {
                name = inner.as_str().to_string();
            }
            Rule::kw_temporal => temporal = true,
            Rule::property_decl_list => {
                for decl in inner.into_inner() {
                    if decl.as_rule() == Rule::property_decl {
                        let mut prop_name = String::new();
                        let mut prop_type = String::new();
                        let mut not_null = false;
                        for part in decl.into_inner() {
                            match part.as_rule() {
                                Rule::identifier if prop_name.is_empty() => {
                                    prop_name = part.as_str().to_string();
                                }
                                Rule::property_type_name => {
                                    prop_type = part.as_str().to_uppercase();
                                }
                                Rule::property_decl_modifier => {
                                    not_null = true;
                                }
                                _ => {}
                            }
                        }
                        if !prop_name.is_empty() && !prop_type.is_empty() {
                            properties.push(EdgePropertyDecl {
                                name: prop_name,
                                type_name: prop_type,
                                not_null,
                            });
                        }
                    }
                }
            }
            _ => {}
        }
    }

    if name.is_empty() {
        return Err(ParseError::Invalid(
            "CREATE NODE TYPE requires a label name".into(),
        ));
    }

    Ok(crate::cypher::ast::CreateNodeTypeClause {
        name,
        temporal,
        properties,
    })
}

// ----- Trigger DDL builders -----

fn build_create_trigger_clause(pair: Pair<'_, Rule>) -> Result<CreateTriggerClause, ParseError> {
    let mut name: Option<String> = None;
    let mut target: Option<TriggerTarget> = None;
    let mut events = TriggerEvents::default();
    let mut timing: Option<TriggerTiming> = None;
    let mut body_source: Option<String> = None;
    let mut cascade_limit: Option<u32> = None;
    let mut cascade_fanout: Option<u32> = None;
    let mut on_error: Option<OnErrorPolicy> = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::identifier if name.is_none() => {
                name = Some(inner.as_str().to_string());
            }
            Rule::trigger_target => {
                target = Some(parse_trigger_target(inner)?);
            }
            Rule::trigger_option => {
                // Unwrap the option's single inner alternative.
                let opt = inner
                    .into_inner()
                    .next()
                    .ok_or_else(|| ParseError::Invalid("trigger_option: missing inner".into()))?;
                match opt.as_rule() {
                    Rule::trigger_cascade_limit | Rule::trigger_maxdepth => {
                        // MAXDEPTH is a deprecated alias for CASCADE_LIMIT (the trigger architecture).
                        if let Some(v) = parse_trigger_integer_option(opt)? {
                            cascade_limit = Some(v);
                        }
                    }
                    Rule::trigger_cascade_fanout => {
                        if let Some(v) = parse_trigger_integer_option(opt)? {
                            cascade_fanout = Some(v);
                        }
                    }
                    Rule::on_error_clause => {
                        on_error = Some(parse_on_error_clause(opt)?);
                    }
                    other => {
                        return Err(ParseError::Invalid(format!(
                            "trigger_option: unexpected node {other:?}"
                        )));
                    }
                }
            }
            Rule::trigger_event_list => {
                for ev in inner.into_inner() {
                    if ev.as_rule() == Rule::trigger_event {
                        for tok in ev.into_inner() {
                            match tok.as_rule() {
                                Rule::kw_create => events.on_create = true,
                                Rule::kw_update => events.on_update = true,
                                Rule::kw_delete => events.on_delete = true,
                                _ => {}
                            }
                        }
                    }
                }
            }
            Rule::trigger_timing => {
                for tok in inner.into_inner() {
                    match tok.as_rule() {
                        Rule::kw_before => timing = Some(TriggerTiming::BeforeCommit),
                        Rule::kw_after => timing = Some(TriggerTiming::AfterCommit),
                        _ => {}
                    }
                }
            }
            Rule::trigger_body => {
                body_source = Some(inner.as_str().trim().to_string());
            }
            _ => {}
        }
    }

    let name = name.ok_or_else(|| ParseError::Invalid("CREATE TRIGGER requires a name".into()))?;
    let target = target.ok_or_else(|| {
        ParseError::Invalid("CREATE TRIGGER requires ON :Label or [:EdgeType]".into())
    })?;
    if !events.any() {
        return Err(ParseError::Invalid(
            "CREATE TRIGGER requires at least one event (CREATE | UPDATE | DELETE)".into(),
        ));
    }
    let timing = timing.ok_or_else(|| {
        ParseError::Invalid("CREATE TRIGGER requires BEFORE COMMIT or AFTER COMMIT".into())
    })?;
    let body_source = body_source
        .ok_or_else(|| ParseError::Invalid("CREATE TRIGGER requires EXECUTE <clauses>".into()))?;

    Ok(CreateTriggerClause {
        name,
        target,
        events,
        timing,
        body_source,
        cascade_limit,
        cascade_fanout,
        on_error,
    })
}

fn parse_trigger_target(pair: Pair<'_, Rule>) -> Result<TriggerTarget, ParseError> {
    // Two grammar shapes:
    //   - label_list      (single ":Label" — multiple labels rejected here)
    //   - "[" ":" identifier "]"  (edge type)
    let mut iter = pair.into_inner();
    let first = iter.next().ok_or_else(|| {
        ParseError::Invalid("trigger target: expected :Label or [:EdgeType]".into())
    })?;
    match first.as_rule() {
        Rule::label_list => {
            let labels: Vec<String> = first
                .into_inner()
                .filter(|p| p.as_rule() == Rule::identifier)
                .map(|p| p.as_str().to_string())
                .collect();
            if labels.len() != 1 {
                return Err(ParseError::Invalid(format!(
                    "trigger target expects exactly one label, got {} ({:?})",
                    labels.len(),
                    labels
                )));
            }
            Ok(TriggerTarget::Label(
                labels.into_iter().next().unwrap_or_default(),
            ))
        }
        Rule::identifier => Ok(TriggerTarget::EdgeType(first.as_str().to_string())),
        other => Err(ParseError::Invalid(format!(
            "trigger target: unexpected node {other:?}"
        ))),
    }
}

/// Parse the integer argument of `MAXDEPTH n`, `CASCADE_LIMIT n`, or
/// `CASCADE_FANOUT n`. The rule's first numeric child is returned.
fn parse_trigger_integer_option(pair: Pair<'_, Rule>) -> Result<Option<u32>, ParseError> {
    let rule = pair.as_rule();
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::integer_literal {
            let v: u32 = inner.as_str().parse().map_err(|e| {
                ParseError::Invalid(format!(
                    "{rule:?}: invalid integer `{}`: {e}",
                    inner.as_str()
                ))
            })?;
            return Ok(Some(v));
        }
    }
    Err(ParseError::Invalid(format!("{rule:?} requires an integer")))
}

fn parse_on_error_clause(pair: Pair<'_, Rule>) -> Result<OnErrorPolicy, ParseError> {
    let policy = pair
        .into_inner()
        .find(|p| {
            matches!(
                p.as_rule(),
                Rule::on_error_propagate | Rule::on_error_retry | Rule::on_error_dead_letter
            )
        })
        .ok_or_else(|| ParseError::Invalid("ON ERROR: missing policy".into()))?;
    match policy.as_rule() {
        Rule::on_error_propagate => Ok(OnErrorPolicy::Propagate),
        Rule::on_error_dead_letter => Ok(OnErrorPolicy::DeadLetter),
        Rule::on_error_retry => {
            let mut nums: Vec<u32> = Vec::new();
            for tok in policy.into_inner() {
                if tok.as_rule() == Rule::integer_literal {
                    let v: u32 = tok.as_str().parse().map_err(|e| {
                        ParseError::Invalid(format!(
                            "ON ERROR RETRY: invalid integer `{}`: {e}",
                            tok.as_str()
                        ))
                    })?;
                    nums.push(v);
                }
            }
            if nums.is_empty() {
                return Err(ParseError::Invalid(
                    "ON ERROR RETRY requires a retry count".into(),
                ));
            }
            let n = nums[0];
            // Default backoff base = 1000ms when WITH BACKOFF is omitted.
            let backoff_ms = nums.get(1).copied().unwrap_or(1000);
            Ok(OnErrorPolicy::Retry { n, backoff_ms })
        }
        other => Err(ParseError::Invalid(format!(
            "ON ERROR: unexpected policy node {other:?}"
        ))),
    }
}

fn build_drop_trigger_clause(pair: Pair<'_, Rule>) -> Result<DropTriggerClause, ParseError> {
    let name = pair
        .into_inner()
        .find(|p| p.as_rule() == Rule::identifier)
        .map(|p| p.as_str().to_string())
        .ok_or_else(|| ParseError::Invalid("DROP TRIGGER requires a name".into()))?;
    Ok(DropTriggerClause { name })
}

fn build_alter_trigger_clause(pair: Pair<'_, Rule>) -> Result<AlterTriggerClause, ParseError> {
    let mut name: Option<String> = None;
    let mut action: Option<AlterTriggerAction> = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::identifier if name.is_none() => {
                name = Some(inner.as_str().to_string());
            }
            Rule::alter_trigger_disable => action = Some(AlterTriggerAction::Disable),
            Rule::alter_trigger_enable => action = Some(AlterTriggerAction::Enable),
            Rule::alter_trigger_set_body => {
                let body = inner
                    .into_inner()
                    .find(|p| p.as_rule() == Rule::trigger_body)
                    .map(|p| p.as_str().trim().to_string())
                    .ok_or_else(|| {
                        ParseError::Invalid("ALTER TRIGGER SET EXECUTE: missing body".into())
                    })?;
                action = Some(AlterTriggerAction::SetBody(body));
            }
            Rule::alter_trigger_set_on_error => {
                let policy_pair = inner
                    .into_inner()
                    .find(|p| p.as_rule() == Rule::on_error_clause)
                    .ok_or_else(|| {
                        ParseError::Invalid(
                            "ALTER TRIGGER SET ON ERROR: missing on_error clause".into(),
                        )
                    })?;
                action = Some(AlterTriggerAction::SetOnError(parse_on_error_clause(
                    policy_pair,
                )?));
            }
            _ => {}
        }
    }

    let name = name.ok_or_else(|| ParseError::Invalid("ALTER TRIGGER requires a name".into()))?;
    let action = action.ok_or_else(|| {
        ParseError::Invalid("ALTER TRIGGER requires DISABLE | ENABLE | SET ...".into())
    })?;
    Ok(AlterTriggerClause { name, action })
}

fn build_drop_index_clause(pair: Pair<'_, Rule>) -> Result<DropIndexClause, ParseError> {
    let mut name = String::new();

    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::identifier {
            name = inner.as_str().to_string();
        }
    }

    if name.is_empty() {
        return Err(ParseError::Invalid("DROP INDEX requires index name".into()));
    }

    Ok(DropIndexClause { name })
}

fn build_create_vector_index_clause(
    pair: Pair<'_, Rule>,
) -> Result<CreateVectorIndexClause, ParseError> {
    let mut identifiers: Vec<String> = Vec::new();
    let mut m: Option<usize> = None;
    let mut ef_construction: Option<usize> = None;
    let mut metric: Option<String> = None;
    let mut dimensions: Option<u32> = None;
    let mut quantization: Option<String> = None;
    let mut online_during_build: Option<String> = None;
    let mut ef_search: Option<usize> = None;
    let mut rerank_candidates: Option<usize> = None;
    let mut extension_tail: Option<String> = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::identifier => identifiers.push(inner.as_str().to_string()),
            Rule::vector_index_options => {
                for opt in inner.into_inner() {
                    if opt.as_rule() == Rule::vector_index_option {
                        let mut key = String::new();
                        let mut val_str = String::new();
                        for part in opt.into_inner() {
                            match part.as_rule() {
                                Rule::identifier => key = part.as_str().to_string(),
                                Rule::integer_literal => val_str = part.as_str().to_string(),
                                Rule::float_literal => val_str = part.as_str().to_string(),
                                Rule::string_literal => {
                                    // Strip surrounding quotes
                                    let raw = part.as_str();
                                    val_str = raw[1..raw.len() - 1].to_string();
                                }
                                _ => {}
                            }
                        }
                        match key.to_lowercase().as_str() {
                            "m" => m = val_str.parse().ok(),
                            "ef_construction" => ef_construction = val_str.parse().ok(),
                            "metric" => metric = Some(val_str),
                            "dimensions" => dimensions = val_str.parse().ok(),
                            "quantization" => quantization = Some(val_str),
                            "online_during_build" => online_during_build = Some(val_str),
                            "ef_search" => ef_search = val_str.parse().ok(),
                            "rerank_candidates" => rerank_candidates = val_str.parse().ok(),
                            _ => {} // unknown options silently ignored
                        }
                    }
                }
            }
            Rule::extension_tail => {
                extension_tail = Some(inner.as_str().trim().to_string());
            }
            _ => {}
        }
    }

    if identifiers.len() < 3 {
        return Err(ParseError::Invalid(
            "CREATE VECTOR INDEX requires name, label, and property".into(),
        ));
    }

    Ok(CreateVectorIndexClause {
        name: identifiers[0].clone(),
        label: identifiers[1].clone(),
        property: identifiers[2].clone(),
        m,
        ef_construction,
        metric,
        dimensions,
        quantization,
        online_during_build,
        ef_search,
        rerank_candidates,
        extension_tail,
    })
}

fn build_drop_vector_index_clause(
    pair: Pair<'_, Rule>,
) -> Result<DropVectorIndexClause, ParseError> {
    let mut name = String::new();

    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::identifier {
            name = inner.as_str().to_string();
        }
    }

    if name.is_empty() {
        return Err(ParseError::Invalid(
            "DROP VECTOR INDEX requires index name".into(),
        ));
    }

    Ok(DropVectorIndexClause { name })
}

// --- Write clauses ---

fn build_create_clause(pair: Pair<'_, Rule>) -> Result<CreateClause, ParseError> {
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::pattern_list {
            let patterns = build_pattern_list(inner)?;
            return Ok(CreateClause { patterns });
        }
    }
    Err(ParseError::Invalid("CREATE clause missing patterns".into()))
}

fn build_foreach_clause(pair: Pair<'_, Rule>) -> Result<ForeachClause, ParseError> {
    let mut variable = None;
    let mut list = None;
    let mut body = Vec::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::identifier if variable.is_none() => {
                variable = Some(inner.as_str().to_string());
            }
            Rule::expression if list.is_none() => {
                list = Some(build_expression(inner)?);
            }
            Rule::foreach_update_clause => {
                // Unwrap to the actual update clause and reuse build_clause.
                let update = first_inner(inner)?;
                build_clause(update, &mut body)?;
            }
            _ => {}
        }
    }

    let variable =
        variable.ok_or_else(|| ParseError::Invalid("FOREACH missing variable".into()))?;
    let list = list.ok_or_else(|| ParseError::Invalid("FOREACH missing list expression".into()))?;
    if body.is_empty() {
        return Err(ParseError::Invalid(
            "FOREACH body has no update clauses".into(),
        ));
    }

    Ok(ForeachClause {
        variable,
        list,
        body,
    })
}

fn build_call_subquery_clause(pair: Pair<'_, Rule>) -> Result<CallSubqueryClause, ParseError> {
    let mut optional = false;
    let mut body = Vec::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::kw_optional => optional = true,
            Rule::kw_call => {}
            Rule::clause => {
                let actual = first_inner(inner)?;
                build_clause(actual, &mut body)?;
            }
            _ => {}
        }
    }

    if body.is_empty() {
        return Err(ParseError::Invalid(
            "CALL subquery has no body clauses".into(),
        ));
    }

    Ok(CallSubqueryClause { optional, body })
}

fn build_merge_clause(pair: Pair<'_, Rule>) -> Result<MergeClause, ParseError> {
    let mut pattern = None;
    let mut on_match = Vec::new();
    let mut on_create = Vec::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::pattern => {
                pattern = Some(build_pattern(inner)?);
            }
            Rule::merge_action => {
                let action = first_inner(inner)?;
                match action.as_rule() {
                    Rule::on_match_action => {
                        for child in action.into_inner() {
                            if child.as_rule() == Rule::set_items {
                                on_match = build_set_items(child)?;
                            }
                        }
                    }
                    Rule::on_create_action => {
                        for child in action.into_inner() {
                            if child.as_rule() == Rule::set_items {
                                on_create = build_set_items(child)?;
                            }
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    Ok(MergeClause {
        pattern: pattern
            .ok_or_else(|| ParseError::Invalid("MERGE clause missing pattern".into()))?,
        on_match,
        on_create,
    })
}

fn build_merge_nodes_clause(pair: Pair<'_, Rule>) -> Result<MergeNodesClause, ParseError> {
    let mut idents = Vec::with_capacity(3);
    let mut options = Vec::new();
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::identifier => idents.push(inner.as_str().to_string()),
            Rule::merge_nodes_option => options.push(inner),
            _ => {}
        }
    }
    if idents.len() != 3 {
        return Err(ParseError::Invalid(format!(
            "MERGE NODES expected (a, b) INTO target — got {} identifiers",
            idents.len()
        )));
    }
    let source_a = idents.remove(0);
    let source_b = idents.remove(0);
    let target = idents.remove(0);
    if target != source_a && target != source_b {
        return Err(ParseError::Invalid(format!(
            "MERGE NODES INTO `{target}` must be one of `{source_a}`, `{source_b}`"
        )));
    }
    if source_a == source_b {
        return Err(ParseError::Invalid(
            "MERGE NODES requires two distinct source variables".to_string(),
        ));
    }

    let mut conflict = MergeNodesConflictStrategy::default();
    let mut transfer_edges: Option<TransferEdgesEndpoints> = None;
    let mut duplicate: Option<MergeNodesDuplicateStrategy> = None;
    // Default per arch/compatibility/native-procedures.md: edge properties are
    // always transferred. The `TRANSFER EDGE PROPERTIES` clause is a redundant
    // readability ack; its absence does NOT mean "drop properties".
    let mut transfer_edge_properties = true;

    for opt in options {
        let inner = first_inner(opt)?;
        match inner.as_rule() {
            Rule::merge_nodes_on_conflict => {
                // Skip the `ON CONFLICT` literal token pairs (kw_on, kw_conflict)
                // and pick the strategy node.
                let strategy = inner
                    .into_inner()
                    .find(|p| {
                        matches!(
                            p.as_rule(),
                            Rule::mn_strategy_keep_first
                                | Rule::mn_strategy_keep_last
                                | Rule::mn_strategy_coalesce
                                | Rule::mn_strategy_set
                        )
                    })
                    .ok_or_else(|| {
                        ParseError::Invalid("MERGE NODES ON CONFLICT: missing strategy".into())
                    })?;
                conflict = match strategy.as_rule() {
                    Rule::mn_strategy_keep_first => MergeNodesConflictStrategy::KeepFirst,
                    Rule::mn_strategy_keep_last => MergeNodesConflictStrategy::KeepLast,
                    Rule::mn_strategy_coalesce => MergeNodesConflictStrategy::Coalesce,
                    Rule::mn_strategy_set => {
                        let mut items = Vec::new();
                        for c in strategy.into_inner() {
                            if c.as_rule() == Rule::set_items {
                                items = build_set_items(c)?;
                            }
                        }
                        MergeNodesConflictStrategy::SetExpressions(items)
                    }
                    other => {
                        return Err(ParseError::Invalid(format!(
                            "MERGE NODES ON CONFLICT: unexpected strategy node {other:?}"
                        )));
                    }
                };
            }
            Rule::merge_nodes_transfer_edges => {
                let mut ids = Vec::with_capacity(2);
                for c in inner.into_inner() {
                    if c.as_rule() == Rule::identifier {
                        ids.push(c.as_str().to_string());
                    }
                }
                if ids.len() != 2 {
                    return Err(ParseError::Invalid(
                        "MERGE NODES TRANSFER EDGES expected `FROM <src> TO <dst>`".into(),
                    ));
                }
                let dst = ids.remove(1);
                let src = ids.remove(0);
                // Semantic checks: dst must be target; src must be the other source.
                if dst != target {
                    return Err(ParseError::Invalid(format!(
                        "MERGE NODES TRANSFER EDGES TO `{dst}` must match INTO target `{target}`"
                    )));
                }
                let expected_src = if target == source_a {
                    &source_b
                } else {
                    &source_a
                };
                if src != *expected_src {
                    return Err(ParseError::Invalid(format!(
                        "MERGE NODES TRANSFER EDGES FROM `{src}` must be the non-surviving source `{expected_src}`"
                    )));
                }
                transfer_edges = Some(TransferEdgesEndpoints { src, dst });
            }
            Rule::merge_nodes_on_duplicate => {
                let strategy = inner
                    .into_inner()
                    .find(|p| {
                        matches!(
                            p.as_rule(),
                            Rule::mn_dup_keep_both
                                | Rule::mn_dup_merge_properties
                                | Rule::mn_dup_keep_target
                        )
                    })
                    .ok_or_else(|| {
                        ParseError::Invalid("MERGE NODES ON DUPLICATE: missing strategy".into())
                    })?;
                duplicate = Some(match strategy.as_rule() {
                    Rule::mn_dup_keep_both => MergeNodesDuplicateStrategy::KeepBoth,
                    Rule::mn_dup_merge_properties => MergeNodesDuplicateStrategy::MergeProperties,
                    Rule::mn_dup_keep_target => MergeNodesDuplicateStrategy::KeepTarget,
                    other => {
                        return Err(ParseError::Invalid(format!(
                            "MERGE NODES ON DUPLICATE: unexpected strategy node {other:?}"
                        )));
                    }
                });
            }
            Rule::merge_nodes_transfer_edge_props => {
                transfer_edge_properties = true;
            }
            other => {
                return Err(ParseError::Invalid(format!(
                    "MERGE NODES: unexpected option node {other:?}"
                )));
            }
        }
    }

    if duplicate.is_some() && transfer_edges.is_none() {
        return Err(ParseError::Invalid(
            "MERGE NODES ON DUPLICATE requires a TRANSFER EDGES clause".into(),
        ));
    }

    Ok(MergeNodesClause {
        source_a,
        source_b,
        target,
        conflict,
        transfer_edges,
        duplicate: duplicate.unwrap_or_default(),
        transfer_edge_properties,
    })
}

fn build_clone_node_clause(pair: Pair<'_, Rule>) -> Result<CloneNodeClause, ParseError> {
    let mut idents = Vec::with_capacity(2);
    let mut options = Vec::new();
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::identifier => idents.push(inner.as_str().to_string()),
            Rule::clone_node_option => options.push(inner),
            _ => {}
        }
    }
    if idents.len() != 2 {
        return Err(ParseError::Invalid(format!(
            "CLONE NODE expected `a AS b` — got {} identifiers",
            idents.len()
        )));
    }
    let source = idents.remove(0);
    let target = idents.remove(0);
    if source == target {
        return Err(ParseError::Invalid(
            "CLONE NODE source and clone variables must differ".to_string(),
        ));
    }

    let mut with_edges = false;
    // Properties are copied by default; `WITH PROPERTIES` is an explicit,
    // equivalent affirmation (see arch/compatibility/native-procedures.md).
    let mut with_properties = true;
    let mut set_items = Vec::new();
    for opt in options {
        let inner = first_inner(opt)?;
        match inner.as_rule() {
            Rule::clone_node_with_edges => with_edges = true,
            Rule::clone_node_with_properties => with_properties = true,
            Rule::clone_node_set => {
                for c in inner.into_inner() {
                    if c.as_rule() == Rule::set_items {
                        set_items = build_set_items(c)?;
                    }
                }
            }
            other => {
                return Err(ParseError::Invalid(format!(
                    "CLONE NODE: unexpected option node {other:?}"
                )));
            }
        }
    }

    Ok(CloneNodeClause {
        source,
        target,
        with_edges,
        with_properties,
        set_items,
    })
}

fn build_upsert_clause(pair: Pair<'_, Rule>) -> Result<UpsertClause, ParseError> {
    let mut pattern = None;
    let mut on_match = Vec::new();
    let mut on_create = Vec::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::pattern => {
                pattern = Some(build_pattern(inner)?);
            }
            Rule::upsert_action => {
                let action = first_inner(inner)?;
                match action.as_rule() {
                    Rule::upsert_on_match => {
                        for child in action.into_inner() {
                            if child.as_rule() == Rule::set_items {
                                on_match = build_set_items(child)?;
                            }
                        }
                    }
                    Rule::upsert_on_create => {
                        for child in action.into_inner() {
                            if child.as_rule() == Rule::pattern_list {
                                on_create = build_pattern_list(child)?;
                            }
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    Ok(UpsertClause {
        pattern: pattern
            .ok_or_else(|| ParseError::Invalid("UPSERT clause missing pattern".into()))?,
        on_match,
        on_create,
    })
}

fn build_delete_clause(pair: Pair<'_, Rule>, detach: bool) -> Result<DeleteClause, ParseError> {
    let mut exprs = Vec::new();

    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::expression_list {
            for expr_pair in inner.into_inner() {
                if expr_pair.as_rule() == Rule::expression {
                    exprs.push(build_expression(expr_pair)?);
                }
            }
        }
    }

    Ok(DeleteClause { detach, exprs })
}

fn build_detach_document_clause(pair: Pair<'_, Rule>) -> Result<DetachDocumentClause, ParseError> {
    let mut source_variable = String::new();
    let mut property_path: Vec<String> = Vec::new();
    let mut target_pattern: Option<(NodePattern, RelationshipPattern, NodePattern)> = None;
    let mut transfer: Option<TransferEdgesSpec> = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::detach_doc_source => {
                let mut segs: Vec<String> = inner
                    .into_inner()
                    .filter(|p| p.as_rule() == Rule::identifier)
                    .map(extract_identifier)
                    .collect();
                if segs.len() < 2 {
                    return Err(ParseError::Invalid(
                        "DETACH DOCUMENT source must be a property path (e.g. n.address)"
                            .to_string(),
                    ));
                }
                source_variable = segs.remove(0);
                property_path = segs;
            }
            Rule::detach_doc_target => {
                let mut nodes: Vec<NodePattern> = Vec::new();
                let mut rel: Option<RelationshipPattern> = None;
                for part in inner.into_inner() {
                    match part.as_rule() {
                        Rule::node_pattern => nodes.push(build_node_pattern(part)?),
                        Rule::relationship_pattern => rel = Some(build_relationship_pattern(part)?),
                        _ => {}
                    }
                }
                let rel_value = rel.ok_or_else(|| {
                    ParseError::Invalid(
                        "DETACH DOCUMENT target must be (a:Label)-[:TYPE]->(n)".to_string(),
                    )
                })?;
                if nodes.len() != 2 {
                    return Err(ParseError::Invalid(
                        "DETACH DOCUMENT target must be (a:Label)-[:TYPE]->(n)".to_string(),
                    ));
                }
                let tgt = nodes.pop().ok_or_else(|| {
                    ParseError::Invalid("DETACH DOCUMENT: missing target node".to_string())
                })?;
                let src_in_pattern = nodes.pop().ok_or_else(|| {
                    ParseError::Invalid("DETACH DOCUMENT: missing anchor node".to_string())
                })?;
                target_pattern = Some((src_in_pattern, rel_value, tgt));
            }
            Rule::detach_doc_transfer => {
                let mut node_var: Option<String> = None;
                let mut target_var: Option<String> = None;
                let mut predicate: Option<Expr> = None;
                for part in inner.into_inner() {
                    match part.as_rule() {
                        Rule::identifier => {
                            let ident = extract_identifier(part);
                            if node_var.is_none() {
                                node_var = Some(ident);
                            } else if target_var.is_none() {
                                target_var = Some(ident);
                            }
                        }
                        Rule::where_inline => {
                            predicate = Some(find_expression(part)?);
                        }
                        _ => {}
                    }
                }
                let node_variable = node_var.ok_or_else(|| {
                    ParseError::Invalid("TRANSFER EDGES requires ON <node_var>".to_string())
                })?;
                let target_variable = target_var.ok_or_else(|| {
                    ParseError::Invalid("TRANSFER EDGES requires TO <target_var>".to_string())
                })?;
                let predicate = predicate.ok_or_else(|| {
                    ParseError::Invalid("TRANSFER EDGES requires a WHERE predicate".to_string())
                })?;
                transfer = Some(TransferEdgesSpec {
                    node_variable,
                    target_variable,
                    predicate,
                });
            }
            _ => {}
        }
    }

    let (pattern_src, rel, pattern_tgt) = target_pattern
        .ok_or_else(|| ParseError::Invalid("DETACH DOCUMENT missing target pattern".to_string()))?;

    // Identify which node in the pattern is the existing source (`source_variable`)
    // and which is the new target.
    let (target_node, edge_direction) =
        if pattern_tgt.variable.as_deref() == Some(source_variable.as_str()) {
            // (a:Label)-[:TYPE]->(n)   → n is pattern_tgt, `a` is pattern_src.
            //   Direction: edge goes a → n, i.e. relative to `n` (source), edge is Incoming.
            //   BUT: with `dash ~ right_arrow` (->) dir=Outgoing relative to the LEFT node.
            //   Relative to `source_variable` (right side = n), that edge is Incoming.
            let dir = match rel.direction {
                Direction::Outgoing => EdgeFromSource::Incoming, // (a)-[:T]->(n): from n's view, incoming
                Direction::Incoming => EdgeFromSource::Outgoing, // (a)<-[:T]-(n): from n's view, outgoing
                Direction::Both => EdgeFromSource::Incoming,
            };
            (pattern_src, dir)
        } else if pattern_src.variable.as_deref() == Some(source_variable.as_str()) {
            // (n)-[:TYPE]->(a:Label) — source is on the left.
            let dir = match rel.direction {
                Direction::Outgoing => EdgeFromSource::Outgoing,
                Direction::Incoming => EdgeFromSource::Incoming,
                Direction::Both => EdgeFromSource::Outgoing,
            };
            (pattern_tgt, dir)
        } else {
            return Err(ParseError::Invalid(format!(
                "DETACH DOCUMENT AS pattern must reference source variable `{}`",
                source_variable
            )));
        };

    let target_variable = target_node
        .variable
        .ok_or_else(|| ParseError::Invalid("DETACH DOCUMENT target must be named".to_string()))?;
    let target_labels = target_node.labels;
    let edge_type = rel.rel_types.into_iter().next();
    let edge_variable = rel.variable;

    Ok(DetachDocumentClause {
        source_variable,
        property_path,
        target_variable,
        target_labels,
        edge_type,
        edge_direction,
        edge_variable,
        transfer,
    })
}

fn build_attach_document_clause(pair: Pair<'_, Rule>) -> Result<AttachDocumentClause, ParseError> {
    let mut pattern: Option<(NodePattern, RelationshipPattern, NodePattern)> = None;
    let mut target_var: Option<String> = None;
    let mut target_path: Vec<String> = Vec::new();
    let mut transfer: Option<TransferEdgesSpec> = None;
    let mut on_conflict_replace = false;
    let mut on_remaining_fail = false;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::attach_doc_pattern => {
                let mut nodes: Vec<NodePattern> = Vec::new();
                let mut rel: Option<RelationshipPattern> = None;
                for part in inner.into_inner() {
                    match part.as_rule() {
                        Rule::node_pattern => nodes.push(build_node_pattern(part)?),
                        Rule::relationship_pattern => rel = Some(build_relationship_pattern(part)?),
                        _ => {}
                    }
                }
                let rel_value = rel.ok_or_else(|| {
                    ParseError::Invalid("ATTACH pattern missing relationship".to_string())
                })?;
                if nodes.len() != 2 {
                    return Err(ParseError::Invalid(
                        "ATTACH pattern must be `(a)-[:TYPE]->(u)`".to_string(),
                    ));
                }
                let tgt = nodes.pop().ok_or_else(|| {
                    ParseError::Invalid("ATTACH: missing target node".to_string())
                })?;
                let src = nodes.pop().ok_or_else(|| {
                    ParseError::Invalid("ATTACH: missing source node".to_string())
                })?;
                pattern = Some((src, rel_value, tgt));
            }
            Rule::attach_doc_target => {
                let mut segs: Vec<String> = inner
                    .into_inner()
                    .filter(|p| p.as_rule() == Rule::identifier)
                    .map(extract_identifier)
                    .collect();
                if segs.len() < 2 {
                    return Err(ParseError::Invalid(
                        "ATTACH INTO must be a property path (e.g. u.address)".to_string(),
                    ));
                }
                target_var = Some(segs.remove(0));
                target_path = segs;
            }
            Rule::attach_doc_option => {
                for opt in inner.into_inner() {
                    match opt.as_rule() {
                        Rule::attach_doc_transfer => {
                            let mut node_var: Option<String> = None;
                            let mut tgt_var: Option<String> = None;
                            let mut predicate: Option<Expr> = None;
                            for part in opt.into_inner() {
                                match part.as_rule() {
                                    Rule::identifier => {
                                        let ident = extract_identifier(part);
                                        if node_var.is_none() {
                                            node_var = Some(ident);
                                        } else if tgt_var.is_none() {
                                            tgt_var = Some(ident);
                                        }
                                    }
                                    Rule::where_inline => {
                                        predicate = Some(find_expression(part)?);
                                    }
                                    _ => {}
                                }
                            }
                            transfer = Some(TransferEdgesSpec {
                                node_variable: node_var.ok_or_else(|| {
                                    ParseError::Invalid(
                                        "TRANSFER EDGES requires ON <node_var>".to_string(),
                                    )
                                })?,
                                target_variable: tgt_var.ok_or_else(|| {
                                    ParseError::Invalid(
                                        "TRANSFER EDGES requires TO <target_var>".to_string(),
                                    )
                                })?,
                                predicate: predicate.ok_or_else(|| {
                                    ParseError::Invalid(
                                        "TRANSFER EDGES requires a WHERE predicate".to_string(),
                                    )
                                })?,
                            });
                        }
                        Rule::attach_doc_on_conflict => {
                            on_conflict_replace = true;
                        }
                        Rule::attach_doc_on_remaining => {
                            on_remaining_fail = true;
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    let (src_np, rel, tgt_np) =
        pattern.ok_or_else(|| ParseError::Invalid("ATTACH missing pattern".to_string()))?;
    let source_variable = src_np
        .variable
        .clone()
        .ok_or_else(|| ParseError::Invalid("ATTACH: source node must be named".to_string()))?;
    let target_variable = tgt_np
        .variable
        .clone()
        .ok_or_else(|| ParseError::Invalid("ATTACH: target node must be named".to_string()))?;

    let target_property_variable =
        target_var.ok_or_else(|| ParseError::Invalid("ATTACH missing INTO target".to_string()))?;
    if target_property_variable != target_variable {
        return Err(ParseError::Invalid(format!(
            "ATTACH INTO variable `{target_property_variable}` must match pattern target \
             node variable `{target_variable}`"
        )));
    }

    // Edge direction relative to the source. Grammar is `(a)...(u)` with the
    // relationship between them; `rel.direction` already reflects arrow orientation.
    let edge_direction = match rel.direction {
        Direction::Outgoing => EdgeFromSource::Outgoing,
        Direction::Incoming => EdgeFromSource::Incoming,
        Direction::Both => EdgeFromSource::Outgoing,
    };
    let edge_type = rel
        .rel_types
        .into_iter()
        .next()
        .ok_or_else(|| ParseError::Invalid("ATTACH edge type must be specified".to_string()))?;

    Ok(AttachDocumentClause {
        source_variable,
        source_labels: src_np.labels,
        target_variable,
        target_labels: tgt_np.labels,
        edge_type,
        edge_direction,
        edge_variable: rel.variable,
        target_property_variable,
        target_property_path: target_path,
        transfer,
        on_conflict_replace,
        on_remaining_fail,
    })
}

fn build_set_clause(pair: Pair<'_, Rule>) -> Result<(Vec<SetItem>, ViolationMode), ParseError> {
    let mut items = None;
    let mut mode = ViolationMode::Fail;
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::set_items => {
                items = Some(build_set_items(inner)?);
            }
            Rule::on_violation_skip => {
                mode = ViolationMode::Skip;
            }
            _ => {}
        }
    }
    match items {
        Some(i) => Ok((i, mode)),
        None => Err(ParseError::Invalid("SET clause missing items".into())),
    }
}

fn build_set_items(pair: Pair<'_, Rule>) -> Result<Vec<SetItem>, ParseError> {
    let mut items = Vec::new();
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::set_item {
            items.push(build_set_item(inner)?);
        }
    }
    Ok(items)
}

fn build_set_item(pair: Pair<'_, Rule>) -> Result<SetItem, ParseError> {
    let inner = first_inner(pair)?;
    match inner.as_rule() {
        Rule::set_doc_function => {
            // doc_function_name "(" property_or_variable "," expression ")"
            let mut function = String::new();
            let mut ids = Vec::new();
            let mut expr = None;
            for child in inner.into_inner() {
                match child.as_rule() {
                    Rule::doc_function_name => function = child.as_str().to_string(),
                    Rule::property_or_variable => {
                        for id in child.into_inner() {
                            if id.as_rule() == Rule::identifier {
                                ids.push(extract_identifier(id));
                            }
                        }
                    }
                    Rule::expression => expr = Some(build_expression(child)?),
                    _ => {}
                }
            }
            if ids.is_empty() {
                return Err(ParseError::Invalid(
                    "doc function missing property path".into(),
                ));
            }
            Ok(SetItem::DocFunction {
                function,
                variable: ids[0].clone(),
                path: ids[1..].to_vec(),
                value_expr: expr.ok_or_else(|| {
                    ParseError::Invalid("doc function missing value argument".into())
                })?,
            })
        }
        Rule::set_property_item => {
            // identifier ("." identifier)+ = expression
            let mut ids = Vec::new();
            let mut expr = None;
            for child in inner.into_inner() {
                match child.as_rule() {
                    Rule::identifier => ids.push(extract_identifier(child)),
                    Rule::expression => expr = Some(build_expression(child)?),
                    _ => {}
                }
            }
            let expr =
                expr.ok_or_else(|| ParseError::Invalid("SET property missing expression".into()))?;
            if ids.len() == 2 {
                Ok(SetItem::Property {
                    variable: ids[0].clone(),
                    property: ids[1].clone(),
                    expr,
                })
            } else if ids.len() > 2 {
                Ok(SetItem::PropertyPath {
                    variable: ids[0].clone(),
                    path: ids[1..].to_vec(),
                    expr,
                })
            } else {
                Err(ParseError::Invalid(
                    "SET property missing identifiers".into(),
                ))
            }
        }
        Rule::set_merge_props_item => {
            // identifier += expression
            let mut variable = None;
            let mut expr = None;
            for child in inner.into_inner() {
                match child.as_rule() {
                    Rule::identifier => variable = Some(extract_identifier(child)),
                    Rule::expression => expr = Some(build_expression(child)?),
                    _ => {}
                }
            }
            Ok(SetItem::MergeProperties {
                variable: variable
                    .ok_or_else(|| ParseError::Invalid("SET += missing variable".into()))?,
                expr: expr
                    .ok_or_else(|| ParseError::Invalid("SET += missing expression".into()))?,
            })
        }
        Rule::set_replace_props_item => {
            // identifier = expression
            let mut variable = None;
            let mut expr = None;
            for child in inner.into_inner() {
                match child.as_rule() {
                    Rule::identifier => variable = Some(extract_identifier(child)),
                    Rule::expression => expr = Some(build_expression(child)?),
                    _ => {}
                }
            }
            Ok(SetItem::ReplaceProperties {
                variable: variable
                    .ok_or_else(|| ParseError::Invalid("SET = missing variable".into()))?,
                expr: expr.ok_or_else(|| ParseError::Invalid("SET = missing expression".into()))?,
            })
        }
        Rule::set_label_item => {
            // identifier :Label
            let mut variable = None;
            let mut labels = Vec::new();
            for child in inner.into_inner() {
                match child.as_rule() {
                    Rule::identifier if variable.is_none() => {
                        variable = Some(extract_identifier(child));
                    }
                    Rule::label_list => {
                        for label in child.into_inner() {
                            if label.as_rule() == Rule::identifier {
                                labels.push(extract_identifier(label));
                            }
                        }
                    }
                    _ => {}
                }
            }
            let var =
                variable.ok_or_else(|| ParseError::Invalid("SET label missing variable".into()))?;
            // Only one label per SetItem; if multiple, we create multiple SetItems
            // But since we return one SetItem, use the first label
            let label = labels
                .into_iter()
                .next()
                .ok_or_else(|| ParseError::Invalid("SET label missing label name".into()))?;
            Ok(SetItem::AddLabel {
                variable: var,
                label,
            })
        }
        _ => Err(ParseError::Invalid(format!(
            "unexpected set item: {:?}",
            inner.as_rule()
        ))),
    }
}

fn build_remove_clause(pair: Pair<'_, Rule>) -> Result<Vec<RemoveItem>, ParseError> {
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::remove_items {
            return build_remove_items(inner);
        }
    }
    Err(ParseError::Invalid("REMOVE clause missing items".into()))
}

fn build_remove_items(pair: Pair<'_, Rule>) -> Result<Vec<RemoveItem>, ParseError> {
    let mut items = Vec::new();
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::remove_item {
            items.push(build_remove_item(inner)?);
        }
    }
    Ok(items)
}

fn build_remove_item(pair: Pair<'_, Rule>) -> Result<RemoveItem, ParseError> {
    let inner = first_inner(pair)?;
    match inner.as_rule() {
        Rule::remove_property_item => {
            let ids: Vec<String> = inner
                .into_inner()
                .filter(|p| p.as_rule() == Rule::identifier)
                .map(extract_identifier)
                .collect();
            if ids.len() == 2 {
                Ok(RemoveItem::Property {
                    variable: ids[0].clone(),
                    property: ids[1].clone(),
                })
            } else if ids.len() > 2 {
                Ok(RemoveItem::PropertyPath {
                    variable: ids[0].clone(),
                    path: ids[1..].to_vec(),
                })
            } else {
                Err(ParseError::Invalid(
                    "REMOVE property missing identifiers".into(),
                ))
            }
        }
        Rule::remove_label_item => {
            let mut variable = None;
            let mut labels = Vec::new();
            for child in inner.into_inner() {
                match child.as_rule() {
                    Rule::identifier if variable.is_none() => {
                        variable = Some(extract_identifier(child));
                    }
                    Rule::label_list => {
                        for label in child.into_inner() {
                            if label.as_rule() == Rule::identifier {
                                labels.push(extract_identifier(label));
                            }
                        }
                    }
                    _ => {}
                }
            }
            let var = variable
                .ok_or_else(|| ParseError::Invalid("REMOVE label missing variable".into()))?;
            let label = labels
                .into_iter()
                .next()
                .ok_or_else(|| ParseError::Invalid("REMOVE label missing label name".into()))?;
            Ok(RemoveItem::Label {
                variable: var,
                label,
            })
        }
        _ => Err(ParseError::Invalid(format!(
            "unexpected remove item: {:?}",
            inner.as_rule()
        ))),
    }
}

// --- Expressions ---

fn build_expression(pair: Pair<'_, Rule>) -> Result<Expr, ParseError> {
    match pair.as_rule() {
        Rule::expression | Rule::or_expr | Rule::xor_expr | Rule::and_expr => {
            build_binary_chain(pair)
        }
        Rule::not_expr => build_not_expr(pair),
        Rule::comparison => build_comparison(pair),
        Rule::addition | Rule::multiplication => build_binary_chain(pair),
        Rule::unary => build_unary(pair),
        Rule::postfix => build_postfix(pair),
        Rule::atom => build_atom(pair),
        Rule::property_or_variable => build_property_or_variable(pair),
        Rule::function_call => build_function_call(pair),
        Rule::case_expr => build_case_expr(pair),
        Rule::reduce_expr => build_reduce_expr(pair),
        Rule::exists_subquery => build_exists_subquery(pair),
        Rule::count_subquery => build_count_subquery(pair),
        Rule::collect_subquery => build_collect_subquery(pair),
        Rule::list_predicate => build_list_predicate(pair),
        Rule::list_comprehension => build_list_comprehension(pair),
        Rule::pattern_comprehension => build_pattern_comprehension(pair),
        Rule::list_literal => build_list_literal(pair),
        Rule::map_literal => build_map_literal(pair),
        Rule::map_projection => build_map_projection(pair),
        Rule::parameter => build_parameter(pair),
        Rule::string_literal => build_string_literal(pair),
        Rule::float_literal => build_float_literal(pair),
        Rule::integer_literal => build_integer_literal(pair),
        Rule::boolean_literal => build_boolean_literal(pair),
        Rule::null_literal => Ok(Expr::Literal(Value::Null)),
        Rule::pattern_predicate => build_pattern_predicate_expr(pair),
        Rule::star => Ok(Expr::Star),
        _ => Err(ParseError::Invalid(format!(
            "unexpected expression rule: {:?}",
            pair.as_rule()
        ))),
    }
}

fn build_pattern_predicate_expr(pair: Pair<'_, Rule>) -> Result<Expr, ParseError> {
    let mut elements = Vec::new();
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::node_pattern => {
                elements.push(PatternElement::Node(build_node_pattern(inner)?));
            }
            Rule::relationship_pattern => {
                elements.push(PatternElement::Relationship(build_relationship_pattern(
                    inner,
                )?));
            }
            _ => {}
        }
    }
    Ok(Expr::PatternPredicate(Pattern {
        elements,
        path_variable: None,
        shortest_path: false,
    }))
}

fn build_binary_chain(pair: Pair<'_, Rule>) -> Result<Expr, ParseError> {
    let rule = pair.as_rule();
    let mut children: Vec<Pair<'_, Rule>> = pair.into_inner().collect();

    if children.len() == 1 {
        return build_expression(children.remove(0));
    }

    // Logical operators: or/xor/and — all children are sub-expressions
    // (keyword tokens like kw_or are filtered by pest for atomic rules)
    let op = match rule {
        Rule::or_expr => Some(BinaryOperator::Or),
        Rule::xor_expr => Some(BinaryOperator::Xor),
        Rule::and_expr => Some(BinaryOperator::And),
        _ => None,
    };

    if let Some(op) = op {
        // For logical operators, inner pairs alternate: expr, kw_*, expr, kw_*, expr
        // Filter out keyword pairs.
        let exprs: Vec<Pair<'_, Rule>> = children
            .into_iter()
            .filter(|p| !matches!(p.as_rule(), Rule::kw_and | Rule::kw_or | Rule::kw_xor))
            .collect();

        let mut iter = exprs.into_iter();
        let mut result = build_expression(
            iter.next()
                .ok_or_else(|| ParseError::Invalid("empty logical op".into()))?,
        )?;
        for child in iter {
            let right = build_expression(child)?;
            result = Expr::BinaryOp {
                left: Box::new(result),
                op,
                right: Box::new(right),
            };
        }
        return Ok(result);
    }

    // Arithmetic: children alternate expression and operator token
    if rule == Rule::addition || rule == Rule::multiplication {
        let mut iter = children.into_iter();
        let mut result = build_expression(
            iter.next()
                .ok_or_else(|| ParseError::Invalid("empty arithmetic".into()))?,
        )?;

        while let Some(op_pair) = iter.next() {
            let op = match op_pair.as_rule() {
                Rule::add_op => match op_pair.as_str() {
                    "+" => BinaryOperator::Add,
                    "-" => BinaryOperator::Sub,
                    _ => {
                        return Err(ParseError::Invalid(format!(
                            "unknown add op: {}",
                            op_pair.as_str()
                        )));
                    }
                },
                Rule::mul_op => match op_pair.as_str() {
                    "*" => BinaryOperator::Mul,
                    "/" => BinaryOperator::Div,
                    "%" => BinaryOperator::Modulo,
                    _ => {
                        return Err(ParseError::Invalid(format!(
                            "unknown mul op: {}",
                            op_pair.as_str()
                        )));
                    }
                },
                _ => return build_expression(op_pair),
            };

            let right = build_expression(
                iter.next()
                    .ok_or_else(|| ParseError::Invalid("missing right operand".into()))?,
            )?;

            result = Expr::BinaryOp {
                left: Box::new(result),
                op,
                right: Box::new(right),
            };
        }

        return Ok(result);
    }

    // Fallback: single child
    build_expression(children.remove(0))
}

fn build_not_expr(pair: Pair<'_, Rule>) -> Result<Expr, ParseError> {
    let children: Vec<Pair<'_, Rule>> = pair.into_inner().collect();

    let mut has_not = false;
    let mut inner_expr = None;

    for child in children {
        match child.as_rule() {
            Rule::kw_not => has_not = true,
            _ => inner_expr = Some(build_expression(child)?),
        }
    }

    let expr = inner_expr.ok_or_else(|| ParseError::Invalid("empty NOT expression".into()))?;

    if has_not {
        Ok(Expr::UnaryOp {
            op: UnaryOperator::Not,
            expr: Box::new(expr),
        })
    } else {
        Ok(expr)
    }
}

fn build_comparison(pair: Pair<'_, Rule>) -> Result<Expr, ParseError> {
    let mut children: Vec<Pair<'_, Rule>> = pair.into_inner().collect();

    if children.is_empty() {
        return Err(ParseError::Invalid("empty comparison".into()));
    }

    // First child is the left-hand side (addition)
    let left = build_expression(children.remove(0))?;

    // If no comparison_tail, just return the addition
    if children.is_empty() {
        return Ok(left);
    }

    // comparison_tail
    let tail = children.remove(0);
    build_comparison_tail(left, tail)
}

fn build_comparison_tail(left: Expr, tail: Pair<'_, Rule>) -> Result<Expr, ParseError> {
    let children: Vec<Pair<'_, Rule>> = tail.into_inner().collect();

    if children.is_empty() {
        return Err(ParseError::Invalid("empty comparison tail".into()));
    }

    let first = &children[0];

    match first.as_rule() {
        Rule::comp_op => {
            let op = match first.as_str() {
                "=" => BinaryOperator::Eq,
                "<>" => BinaryOperator::Neq,
                "<" => BinaryOperator::Lt,
                "<=" => BinaryOperator::Lte,
                ">" => BinaryOperator::Gt,
                ">=" => BinaryOperator::Gte,
                _ => {
                    return Err(ParseError::Invalid(format!(
                        "unknown comparison op: {}",
                        first.as_str()
                    )));
                }
            };
            let right = build_expression(
                children
                    .into_iter()
                    .nth(1)
                    .ok_or_else(|| ParseError::Invalid("missing comparison RHS".into()))?,
            )?;
            Ok(Expr::BinaryOp {
                left: Box::new(left),
                op,
                right: Box::new(right),
            })
        }
        Rule::is_not_null_op => Ok(Expr::IsNull {
            expr: Box::new(left),
            negated: true,
        }),
        Rule::is_null_op => Ok(Expr::IsNull {
            expr: Box::new(left),
            negated: false,
        }),
        Rule::is_typed_op => {
            let mut negated = false;
            let mut type_name = String::new();
            for p in first.clone().into_inner() {
                match p.as_rule() {
                    Rule::kw_not => negated = true,
                    Rule::type_name => type_name = p.as_str().to_string(),
                    _ => {}
                }
            }
            Ok(Expr::IsTyped {
                expr: Box::new(left),
                type_name,
                negated,
            })
        }
        Rule::in_op => {
            let right = build_expression(
                children
                    .into_iter()
                    .nth(1)
                    .ok_or_else(|| ParseError::Invalid("missing IN RHS".into()))?,
            )?;
            Ok(Expr::In {
                expr: Box::new(left),
                list: Box::new(right),
            })
        }
        Rule::regex_op => {
            let right = build_expression(
                children
                    .into_iter()
                    .nth(1)
                    .ok_or_else(|| ParseError::Invalid("missing =~ RHS".into()))?,
            )?;
            Ok(Expr::StringMatch {
                expr: Box::new(left),
                op: StringOp::Regex,
                pattern: Box::new(right),
            })
        }
        Rule::starts_with_op => {
            let right = build_expression(
                children
                    .into_iter()
                    .nth(1)
                    .ok_or_else(|| ParseError::Invalid("missing STARTS WITH RHS".into()))?,
            )?;
            Ok(Expr::StringMatch {
                expr: Box::new(left),
                op: StringOp::StartsWith,
                pattern: Box::new(right),
            })
        }
        Rule::ends_with_op => {
            let right = build_expression(
                children
                    .into_iter()
                    .nth(1)
                    .ok_or_else(|| ParseError::Invalid("missing ENDS WITH RHS".into()))?,
            )?;
            Ok(Expr::StringMatch {
                expr: Box::new(left),
                op: StringOp::EndsWith,
                pattern: Box::new(right),
            })
        }
        Rule::contains_op => {
            let right = build_expression(
                children
                    .into_iter()
                    .nth(1)
                    .ok_or_else(|| ParseError::Invalid("missing CONTAINS RHS".into()))?,
            )?;
            Ok(Expr::StringMatch {
                expr: Box::new(left),
                op: StringOp::Contains,
                pattern: Box::new(right),
            })
        }
        _ => Err(ParseError::Invalid(format!(
            "unexpected comparison_tail child: {:?}",
            first.as_rule()
        ))),
    }
}

fn build_unary(pair: Pair<'_, Rule>) -> Result<Expr, ParseError> {
    // `unary = { "-" ~ unary | postfix }`
    // In pest, "-" is a string literal — not a named rule, so no pair for it.
    // If negation matched: inner = [unary] (the nested unary after "-")
    // If postfix matched: inner = [postfix]
    // We distinguish by the rule of the single child.
    let child = first_inner(pair)?;

    if child.as_rule() == Rule::unary {
        // This is the `"-" ~ unary` alternative — wrap in Neg
        let inner = build_unary(child)?;
        Ok(Expr::UnaryOp {
            op: UnaryOperator::Neg,
            expr: Box::new(inner),
        })
    } else {
        build_expression(child)
    }
}

/// Build a postfix expression: `atom ("[" expression "]")*`
///
/// Handles zero or more subscript accesses chained after an atom:
/// - `list[0]`         → `Subscript { expr: list, index: 0 }`
/// - `labels(n)[0]`    → `Subscript { expr: labels(n), index: 0 }`
/// - `matrix[0][1]`    → nested Subscript
///
/// In pest, string literals `[` and `]` are consumed without producing pairs,
/// so `postfix`'s inner pairs are: `[atom, expression₁, expression₂, ...]`.
fn build_postfix(pair: Pair<'_, Rule>) -> Result<Expr, ParseError> {
    let mut inner = pair.into_inner();
    let atom_pair = inner
        .next()
        .ok_or_else(|| ParseError::Invalid("postfix: missing atom".into()))?;
    let mut expr = build_expression(atom_pair)?;

    // Each remaining pair is an `index_access`: either a single-element index
    // `[i]` or a slice `[s..e]`.
    for access in inner {
        let body = access
            .into_inner()
            .next()
            .ok_or_else(|| ParseError::Invalid("postfix: empty index access".into()))?;
        expr = match body.as_rule() {
            Rule::list_slice => {
                let mut start = None;
                let mut end = None;
                for b in body.into_inner() {
                    match b.as_rule() {
                        Rule::slice_start => {
                            start = Some(Box::new(build_expression(first_inner(b)?)?))
                        }
                        Rule::slice_end => end = Some(Box::new(build_expression(first_inner(b)?)?)),
                        _ => {}
                    }
                }
                Expr::Slice {
                    expr: Box::new(expr),
                    start,
                    end,
                }
            }
            _ => Expr::Subscript {
                expr: Box::new(expr),
                index: Box::new(build_expression(body)?),
            },
        };
    }
    Ok(expr)
}

fn build_atom(pair: Pair<'_, Rule>) -> Result<Expr, ParseError> {
    let inner = first_inner(pair)?;
    build_expression(inner)
}

fn build_property_or_variable(pair: Pair<'_, Rule>) -> Result<Expr, ParseError> {
    let parts: Vec<String> = pair
        .into_inner()
        .filter(|p| p.as_rule() == Rule::identifier)
        .map(extract_identifier)
        .collect();

    if parts.is_empty() {
        return Err(ParseError::Invalid("empty property/variable".into()));
    }

    let mut expr = Expr::Variable(parts[0].clone());

    for prop in &parts[1..] {
        expr = Expr::PropertyAccess {
            expr: Box::new(expr),
            property: prop.clone(),
        };
    }

    Ok(expr)
}

fn build_function_call(pair: Pair<'_, Rule>) -> Result<Expr, ParseError> {
    let mut name = String::new();
    let mut args = Vec::new();
    let mut distinct = false;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::function_name => {
                // Support dotted function names: point.distance → "point.distance"
                let parts: Vec<String> = inner.into_inner().map(extract_identifier).collect();
                name = parts.join(".");
            }
            Rule::distinct_kw => distinct = true,
            Rule::function_args => {
                for arg in inner.into_inner() {
                    match arg.as_rule() {
                        Rule::star => args.push(Expr::Star),
                        _ => args.push(build_expression(arg)?),
                    }
                }
            }
            _ => {}
        }
    }

    Ok(Expr::FunctionCall {
        name,
        args,
        distinct,
    })
}

fn build_exists_subquery(pair: Pair<'_, Rule>) -> Result<Expr, ParseError> {
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::match_clause {
            return Ok(Expr::ExistsSubquery(Box::new(build_match_clause(inner)?)));
        }
    }
    Err(ParseError::Invalid(
        "EXISTS subquery requires a MATCH clause".into(),
    ))
}

fn build_count_subquery(pair: Pair<'_, Rule>) -> Result<Expr, ParseError> {
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::match_clause {
            return Ok(Expr::CountSubquery(Box::new(build_match_clause(inner)?)));
        }
    }
    Err(ParseError::Invalid(
        "COUNT subquery requires a MATCH clause".into(),
    ))
}

fn build_collect_subquery(pair: Pair<'_, Rule>) -> Result<Expr, ParseError> {
    let mut match_clause: Option<MatchClause> = None;
    let mut expr: Option<Expr> = None;
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::match_clause => match_clause = Some(build_match_clause(inner)?),
            Rule::expression => expr = Some(build_expression(inner)?),
            _ => {}
        }
    }
    let match_clause = match_clause
        .ok_or_else(|| ParseError::Invalid("COLLECT subquery requires a MATCH clause".into()))?;
    let expr = expr.ok_or_else(|| {
        ParseError::Invalid("COLLECT subquery requires a RETURN expression".into())
    })?;
    Ok(Expr::CollectSubquery {
        match_clause: Box::new(match_clause),
        expr: Box::new(expr),
    })
}

fn build_pattern_comprehension(pair: Pair<'_, Rule>) -> Result<Expr, ParseError> {
    let mut pattern: Option<Pattern> = None;
    let mut where_clause: Option<Expr> = None;
    let mut map: Option<Expr> = None;
    let mut saw_where = false;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::pattern => pattern = Some(build_pattern(inner)?),
            Rule::kw_where => saw_where = true,
            _ => {
                let e = build_expression(inner)?;
                if saw_where && where_clause.is_none() {
                    where_clause = Some(e);
                } else {
                    map = Some(e);
                }
            }
        }
    }

    match (pattern, map) {
        (Some(pattern), Some(map)) => Ok(Expr::PatternComprehension {
            pattern: Box::new(pattern),
            where_clause: where_clause.map(Box::new),
            map: Box::new(map),
        }),
        _ => Err(ParseError::Invalid(
            "malformed pattern comprehension".into(),
        )),
    }
}

fn build_list_comprehension(pair: Pair<'_, Rule>) -> Result<Expr, ParseError> {
    let mut var: Option<String> = None;
    let mut list: Option<Expr> = None;
    let mut pred: Option<Expr> = None;
    let mut map: Option<Expr> = None;
    let mut saw_where = false;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::identifier => var = Some(extract_identifier(inner)),
            Rule::kw_where => saw_where = true,
            Rule::kw_in => {}
            _ => {
                let e = build_expression(inner)?;
                if list.is_none() {
                    list = Some(e);
                } else if saw_where && pred.is_none() {
                    pred = Some(e);
                } else {
                    map = Some(e);
                }
            }
        }
    }

    match (var, list) {
        (Some(var), Some(list)) => Ok(Expr::ListComprehension {
            var,
            list: Box::new(list),
            pred: pred.map(Box::new),
            map: map.map(Box::new),
        }),
        _ => Err(ParseError::Invalid("malformed list comprehension".into())),
    }
}

fn build_list_predicate(pair: Pair<'_, Rule>) -> Result<Expr, ParseError> {
    let mut kind: Option<ListPredicateKind> = None;
    let mut var: Option<String> = None;
    let mut exprs: Vec<Expr> = Vec::new();
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::list_quantifier => {
                let q = first_inner(inner)?;
                kind = Some(match q.as_rule() {
                    Rule::kw_all => ListPredicateKind::All,
                    Rule::kw_any => ListPredicateKind::Any,
                    Rule::kw_none => ListPredicateKind::None,
                    Rule::kw_single => ListPredicateKind::Single,
                    _ => return Err(ParseError::Invalid("unknown list quantifier".into())),
                });
            }
            Rule::identifier => var = Some(extract_identifier(inner)),
            Rule::kw_in | Rule::kw_where => {}
            _ => exprs.push(build_expression(inner)?),
        }
    }

    let mut ex = exprs.into_iter();
    match (kind, var, ex.next(), ex.next()) {
        (Some(kind), Some(var), Some(list), Some(pred)) => Ok(Expr::ListPredicate {
            kind,
            var,
            list: Box::new(list),
            pred: Box::new(pred),
        }),
        _ => Err(ParseError::Invalid("malformed list predicate".into())),
    }
}

fn build_reduce_expr(pair: Pair<'_, Rule>) -> Result<Expr, ParseError> {
    // Grammar order: kw_reduce, identifier(acc), expression(init),
    // identifier(var), kw_in, expression(list), expression(step). Identifiers
    // and expressions each keep their relative order, so collect by kind.
    let mut idents: Vec<String> = Vec::new();
    let mut exprs: Vec<Expr> = Vec::new();
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::identifier => idents.push(extract_identifier(inner)),
            Rule::kw_reduce | Rule::kw_in => {}
            _ => exprs.push(build_expression(inner)?),
        }
    }

    let mut ids = idents.into_iter();
    let mut ex = exprs.into_iter();
    match (ids.next(), ids.next(), ex.next(), ex.next(), ex.next()) {
        (Some(acc), Some(var), Some(init), Some(list), Some(expr)) => Ok(Expr::Reduce {
            acc,
            init: Box::new(init),
            var,
            list: Box::new(list),
            expr: Box::new(expr),
        }),
        _ => Err(ParseError::Invalid(
            "malformed reduce(...) expression".into(),
        )),
    }
}

fn build_case_expr(pair: Pair<'_, Rule>) -> Result<Expr, ParseError> {
    let mut operand = None;
    let mut when_clauses = Vec::new();
    let mut else_clause = None;
    let mut first_expr = true;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::expression if first_expr => {
                operand = Some(Box::new(build_expression(inner)?));
                first_expr = false;
            }
            Rule::when_clause => {
                let wc_children = inner.into_inner();
                // Filter keyword pairs
                let exprs: Vec<Pair<'_, Rule>> = wc_children
                    .filter(|p| p.as_rule() == Rule::expression)
                    .collect();
                if exprs.len() >= 2 {
                    let mut iter = exprs.into_iter();
                    let when =
                        build_expression(iter.next().ok_or_else(|| {
                            ParseError::Invalid("missing WHEN condition".into())
                        })?)?;
                    let then =
                        build_expression(iter.next().ok_or_else(|| {
                            ParseError::Invalid("missing THEN expression".into())
                        })?)?;
                    when_clauses.push((when, then));
                }
                first_expr = false;
            }
            Rule::else_clause => {
                for ec in inner.into_inner() {
                    if ec.as_rule() == Rule::expression {
                        else_clause = Some(Box::new(build_expression(ec)?));
                    }
                }
            }
            _ => {
                first_expr = false;
            }
        }
    }

    Ok(Expr::Case {
        operand,
        when_clauses,
        else_clause,
    })
}

fn build_list_literal(pair: Pair<'_, Rule>) -> Result<Expr, ParseError> {
    let mut items = Vec::new();
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::expression {
            items.push(build_expression(inner)?);
        }
    }
    Ok(Expr::List(items))
}

fn build_map_literal(pair: Pair<'_, Rule>) -> Result<Expr, ParseError> {
    let mut entries = Vec::new();
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::map_pair {
            let mut children = inner.into_inner();
            let key = extract_identifier(
                children
                    .next()
                    .ok_or_else(|| ParseError::Invalid("missing map key".into()))?,
            );
            let value = build_expression(
                children
                    .next()
                    .ok_or_else(|| ParseError::Invalid("missing map value".into()))?,
            )?;
            entries.push((key, value));
        }
    }
    Ok(Expr::MapLiteral(entries))
}

fn build_map_projection(pair: Pair<'_, Rule>) -> Result<Expr, ParseError> {
    let mut inner = pair.into_inner();

    // First child: property_or_variable (the base expression)
    let base_pair = inner
        .next()
        .ok_or_else(|| ParseError::Invalid("missing map projection base".into()))?;
    let base_expr = build_property_or_variable(base_pair)?;

    // Remaining children: map_projection_item
    let mut items = Vec::new();
    for item_pair in inner {
        if item_pair.as_rule() != Rule::map_projection_item {
            continue;
        }
        let child = item_pair
            .into_inner()
            .next()
            .ok_or_else(|| ParseError::Invalid("empty map projection item".into()))?;

        match child.as_rule() {
            Rule::map_projection_shorthand => {
                // `.name` → Property("name")
                let prop = extract_identifier(
                    child
                        .into_inner()
                        .next()
                        .ok_or_else(|| ParseError::Invalid("missing shorthand property".into()))?,
                );
                items.push(MapProjectionItem::Property(prop));
            }
            Rule::map_projection_computed => {
                // `alias: expr` → Computed("alias", expr)
                let mut children = child.into_inner();
                let alias = extract_identifier(
                    children
                        .next()
                        .ok_or_else(|| ParseError::Invalid("missing computed alias".into()))?,
                );
                let value_expr = build_expression(
                    children
                        .next()
                        .ok_or_else(|| ParseError::Invalid("missing computed value".into()))?,
                )?;
                items.push(MapProjectionItem::Computed(alias, value_expr));
            }
            _ => {}
        }
    }

    Ok(Expr::MapProjection {
        expr: Box::new(base_expr),
        items,
    })
}

fn build_parameter(pair: Pair<'_, Rule>) -> Result<Expr, ParseError> {
    let name = extract_identifier(
        pair.into_inner()
            .find(|p| p.as_rule() == Rule::identifier)
            .ok_or_else(|| ParseError::Invalid("missing parameter name".into()))?,
    );
    Ok(Expr::Parameter(name))
}

fn build_string_literal(pair: Pair<'_, Rule>) -> Result<Expr, ParseError> {
    let inner = pair
        .into_inner()
        .next()
        .ok_or_else(|| ParseError::Invalid("missing string content".into()))?;

    let raw = inner.as_str();
    let unescaped = raw
        .replace("\\'", "'")
        .replace("\\\"", "\"")
        .replace("\\\\", "\\");

    Ok(Expr::Literal(Value::String(unescaped)))
}

fn build_float_literal(pair: Pair<'_, Rule>) -> Result<Expr, ParseError> {
    let s = pair.as_str();
    let f: f64 = s
        .parse()
        .map_err(|_| ParseError::Invalid(format!("invalid float: {s}")))?;
    Ok(Expr::Literal(Value::Float(f)))
}

fn build_integer_literal(pair: Pair<'_, Rule>) -> Result<Expr, ParseError> {
    let s = pair.as_str();
    let n: i64 = s
        .parse()
        .map_err(|_| ParseError::Invalid(format!("invalid integer: {s}")))?;
    Ok(Expr::Literal(Value::Int(n)))
}

fn build_boolean_literal(pair: Pair<'_, Rule>) -> Result<Expr, ParseError> {
    let inner = first_inner(pair)?;
    let is_true = inner.as_rule() == Rule::kw_true;
    Ok(Expr::Literal(Value::Bool(is_true)))
}

// --- Helpers ---

/// Get the first inner pair, or error.
fn first_inner(pair: Pair<'_, Rule>) -> Result<Pair<'_, Rule>, ParseError> {
    pair.into_inner()
        .next()
        .ok_or_else(|| ParseError::Invalid("expected inner rule".into()))
}

/// Find the first expression rule in a pair's children.
fn find_expression(pair: Pair<'_, Rule>) -> Result<Expr, ParseError> {
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::expression {
            return build_expression(inner);
        }
    }
    Err(ParseError::Invalid("expected expression".into()))
}

/// Extract identifier text, handling backtick-quoted identifiers.
fn extract_identifier(pair: Pair<'_, Rule>) -> String {
    let s = pair.as_str();
    if s.starts_with('`') && s.ends_with('`') && s.len() > 1 {
        s[1..s.len() - 1].to_string()
    } else {
        s.to_string()
    }
}

/// Parse a string as u64.
fn parse_u64(s: &str) -> Result<u64, ParseError> {
    s.trim()
        .parse()
        .map_err(|_| ParseError::Invalid(format!("invalid number: {s}")))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
