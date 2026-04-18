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

    for pair in pairs {
        collect_clauses(pair, &mut clauses)?;
    }

    if clauses.is_empty() {
        return Err(ParseError::Invalid("empty query".into()));
    }

    Ok(Query {
        clauses,
        hints: Vec::new(),
    })
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
        Rule::create_clause => {
            let cc = build_create_clause(pair)?;
            clauses.push(Clause::Create(cc));
        }
        Rule::merge_clause => {
            let mc = build_merge_clause(pair)?;
            clauses.push(Clause::Merge(mc));
        }
        Rule::merge_all_clause => {
            let mc = build_merge_clause(pair)?;
            clauses.push(Clause::MergeMany(mc));
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
            Rule::pattern_list => {
                patterns = build_pattern_list(inner)?;
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
    Ok(Pattern { elements })
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
                            _ => {} // unknown options silently ignored
                        }
                    }
                }
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
    Ok(Expr::PatternPredicate(Pattern { elements }))
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

    // Each remaining pair is an index expression (the content between `[` and `]`).
    for index_pair in inner {
        let index = build_expression(index_pair)?;
        expr = Expr::Subscript {
            expr: Box::new(expr),
            index: Box::new(index),
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
mod tests {
    use super::*;

    // -- Utility --

    fn parse_ok(input: &str) -> Query {
        parse(input).unwrap_or_else(|e| panic!("parse failed for {input:?}: {e}"))
    }

    fn parse_err(input: &str) -> ParseError {
        parse(input).unwrap_err()
    }

    // -- Basic MATCH --

    #[test]
    fn simple_match_return() {
        let q = parse_ok("MATCH (n) RETURN n");
        assert_eq!(q.clauses.len(), 2);
        assert!(matches!(q.clauses[0], Clause::Match(_)));
        assert!(matches!(q.clauses[1], Clause::Return(_)));
    }

    #[test]
    fn match_with_label() {
        let q = parse_ok("MATCH (n:User) RETURN n");
        if let Clause::Match(ref m) = q.clauses[0] {
            if let PatternElement::Node(ref np) = m.patterns[0].elements[0] {
                assert_eq!(np.labels, vec!["User"]);
                assert_eq!(np.variable, Some("n".to_string()));
            } else {
                panic!("expected node pattern");
            }
        } else {
            panic!("expected MATCH clause");
        }
    }

    #[test]
    fn match_with_multiple_labels() {
        let q = parse_ok("MATCH (n:User:Admin) RETURN n");
        if let Clause::Match(ref m) = q.clauses[0] {
            if let PatternElement::Node(ref np) = m.patterns[0].elements[0] {
                assert_eq!(np.labels, vec!["User", "Admin"]);
            } else {
                panic!("expected node pattern");
            }
        } else {
            panic!("expected MATCH clause");
        }
    }

    #[test]
    fn match_with_properties() {
        let q = parse_ok("MATCH (n:User {name: 'Alice', age: 30}) RETURN n");
        if let Clause::Match(ref m) = q.clauses[0] {
            if let PatternElement::Node(ref np) = m.patterns[0].elements[0] {
                assert_eq!(np.properties.len(), 2);
                assert_eq!(np.properties[0].0, "name");
                assert_eq!(np.properties[1].0, "age");
            } else {
                panic!("expected node pattern");
            }
        } else {
            panic!("expected MATCH clause");
        }
    }

    // -- Relationships --

    #[test]
    fn match_outgoing_relationship() {
        let q = parse_ok("MATCH (a)-[r:KNOWS]->(b) RETURN a, b");
        if let Clause::Match(ref m) = q.clauses[0] {
            assert_eq!(m.patterns[0].elements.len(), 3);
            if let PatternElement::Relationship(ref rp) = m.patterns[0].elements[1] {
                assert_eq!(rp.direction, Direction::Outgoing);
                assert_eq!(rp.rel_types, vec!["KNOWS"]);
                assert_eq!(rp.variable, Some("r".to_string()));
            } else {
                panic!("expected relationship pattern");
            }
        } else {
            panic!("expected MATCH clause");
        }
    }

    #[test]
    fn match_incoming_relationship() {
        let q = parse_ok("MATCH (a)<-[:FOLLOWS]-(b) RETURN a");
        if let Clause::Match(ref m) = q.clauses[0] {
            if let PatternElement::Relationship(ref rp) = m.patterns[0].elements[1] {
                assert_eq!(rp.direction, Direction::Incoming);
                assert_eq!(rp.rel_types, vec!["FOLLOWS"]);
            } else {
                panic!("expected relationship pattern");
            }
        } else {
            panic!("expected MATCH clause");
        }
    }

    #[test]
    fn match_undirected_relationship() {
        let q = parse_ok("MATCH (a)-[:KNOWS]-(b) RETURN a");
        if let Clause::Match(ref m) = q.clauses[0] {
            if let PatternElement::Relationship(ref rp) = m.patterns[0].elements[1] {
                assert_eq!(rp.direction, Direction::Both);
            } else {
                panic!("expected relationship pattern");
            }
        } else {
            panic!("expected MATCH clause");
        }
    }

    #[test]
    fn match_variable_length_path() {
        let q = parse_ok("MATCH (a)-[:KNOWS*2..5]->(b) RETURN a, b");
        if let Clause::Match(ref m) = q.clauses[0] {
            if let PatternElement::Relationship(ref rp) = m.patterns[0].elements[1] {
                let lb = rp.length.unwrap();
                assert_eq!(lb.min, Some(2));
                assert_eq!(lb.max, Some(5));
            } else {
                panic!("expected relationship pattern");
            }
        } else {
            panic!("expected MATCH clause");
        }
    }

    // -- WHERE --

    #[test]
    fn match_where() {
        let q = parse_ok("MATCH (n:User) WHERE n.age > 25 RETURN n");
        // WHERE is folded into MATCH
        if let Clause::Match(ref m) = q.clauses[0] {
            assert!(m.where_clause.is_some());
        } else {
            panic!("expected MATCH clause");
        }
    }

    #[test]
    fn where_and_or() {
        let q = parse_ok("MATCH (n) WHERE n.age > 25 AND n.name = 'Alice' RETURN n");
        if let Clause::Match(ref m) = q.clauses[0] {
            let w = m.where_clause.as_ref().unwrap();
            assert!(matches!(
                w,
                Expr::BinaryOp {
                    op: BinaryOperator::And,
                    ..
                }
            ));
        } else {
            panic!("expected MATCH");
        }
    }

    #[test]
    fn where_starts_with() {
        let q = parse_ok("MATCH (n) WHERE n.name STARTS WITH 'A' RETURN n");
        if let Clause::Match(ref m) = q.clauses[0] {
            let w = m.where_clause.as_ref().unwrap();
            assert!(matches!(
                w,
                Expr::StringMatch {
                    op: StringOp::StartsWith,
                    ..
                }
            ));
        } else {
            panic!("expected MATCH");
        }
    }

    #[test]
    fn where_is_null() {
        let q = parse_ok("MATCH (n) WHERE n.email IS NULL RETURN n");
        if let Clause::Match(ref m) = q.clauses[0] {
            let w = m.where_clause.as_ref().unwrap();
            assert!(matches!(w, Expr::IsNull { negated: false, .. }));
        } else {
            panic!("expected MATCH");
        }
    }

    #[test]
    fn where_is_not_null() {
        let q = parse_ok("MATCH (n) WHERE n.email IS NOT NULL RETURN n");
        if let Clause::Match(ref m) = q.clauses[0] {
            let w = m.where_clause.as_ref().unwrap();
            assert!(matches!(w, Expr::IsNull { negated: true, .. }));
        } else {
            panic!("expected MATCH");
        }
    }

    #[test]
    fn where_in_list() {
        let q = parse_ok("MATCH (n) WHERE n.status IN ['active', 'pending'] RETURN n");
        if let Clause::Match(ref m) = q.clauses[0] {
            let w = m.where_clause.as_ref().unwrap();
            assert!(matches!(w, Expr::In { .. }));
        } else {
            panic!("expected MATCH");
        }
    }

    // -- RETURN --

    #[test]
    fn return_with_alias() {
        let q = parse_ok("MATCH (n) RETURN n.name AS username");
        if let Clause::Return(ref rc) = q.clauses[1] {
            assert_eq!(rc.items.len(), 1);
            assert_eq!(rc.items[0].alias, Some("username".to_string()));
        } else {
            panic!("expected RETURN clause");
        }
    }

    #[test]
    fn return_star() {
        let q = parse_ok("MATCH (n) RETURN *");
        if let Clause::Return(ref rc) = q.clauses[1] {
            assert!(matches!(rc.items[0].expr, Expr::Star));
        } else {
            panic!("expected RETURN clause");
        }
    }

    #[test]
    fn return_distinct() {
        let q = parse_ok("MATCH (n) RETURN DISTINCT n.name");
        if let Clause::Return(ref rc) = q.clauses[1] {
            assert!(rc.distinct);
        } else {
            panic!("expected RETURN clause");
        }
    }

    #[test]
    fn return_multiple() {
        let q = parse_ok("MATCH (n) RETURN n.name, n.age, n.email");
        if let Clause::Return(ref rc) = q.clauses[1] {
            assert_eq!(rc.items.len(), 3);
        } else {
            panic!("expected RETURN clause");
        }
    }

    // -- Aggregation --

    #[test]
    fn count_star() {
        let q = parse_ok("MATCH (n:User) RETURN count(*)");
        if let Clause::Return(ref rc) = q.clauses[1] {
            if let Expr::FunctionCall {
                ref name, ref args, ..
            } = rc.items[0].expr
            {
                assert_eq!(name, "count");
                assert!(matches!(args[0], Expr::Star));
            } else {
                panic!("expected function call");
            }
        } else {
            panic!("expected RETURN clause");
        }
    }

    #[test]
    fn aggregation_with_grouping() {
        let q = parse_ok(
            "MATCH (n:User) RETURN n.city AS city, count(*) AS cnt, avg(n.age) AS avg_age",
        );
        if let Clause::Return(ref rc) = q.clauses[1] {
            assert_eq!(rc.items.len(), 3);
            assert_eq!(rc.items[0].alias, Some("city".to_string()));
            assert_eq!(rc.items[1].alias, Some("cnt".to_string()));
            assert_eq!(rc.items[2].alias, Some("avg_age".to_string()));
        } else {
            panic!("expected RETURN clause");
        }
    }

    // -- WITH --

    #[test]
    fn with_clause() {
        let q = parse_ok("MATCH (n:User) WITH n, count(*) AS cnt WHERE cnt > 5 RETURN n.name");
        assert!(q.clauses.iter().any(|c| matches!(c, Clause::With(_))));
    }

    // -- UNWIND --

    #[test]
    fn unwind_clause() {
        let q = parse_ok("UNWIND [1, 2, 3] AS x RETURN x");
        if let Clause::Unwind(ref uc) = q.clauses[0] {
            assert_eq!(uc.variable, "x");
            assert!(matches!(uc.expr, Expr::List(_)));
        } else {
            panic!("expected UNWIND clause");
        }
    }

    // -- Scientific notation in float literals --

    #[test]
    fn float_scientific_with_dot() {
        // 1.5e3 = 1500.0
        let q = parse_ok("RETURN 1.5e3 AS x");
        if let Clause::Return(ref rc) = q.clauses[0] {
            assert!(
                matches!(rc.items[0].expr, Expr::Literal(Value::Float(f)) if (f - 1500.0).abs() < 0.001),
                "expected 1500.0, got {:?}",
                rc.items[0].expr
            );
        }
    }

    #[test]
    fn float_scientific_no_dot() {
        // 7e-05 = 0.00007
        let q = parse_ok("RETURN 7e-05 AS x");
        if let Clause::Return(ref rc) = q.clauses[0] {
            assert!(
                matches!(rc.items[0].expr, Expr::Literal(Value::Float(f)) if (f - 0.00007).abs() < 1e-10),
                "expected 0.00007, got {:?}",
                rc.items[0].expr
            );
        }
    }

    #[test]
    fn float_scientific_positive_exponent() {
        // 3E+2 = 300.0
        let q = parse_ok("RETURN 3E+2 AS x");
        if let Clause::Return(ref rc) = q.clauses[0] {
            assert!(
                matches!(rc.items[0].expr, Expr::Literal(Value::Float(f)) if (f - 300.0).abs() < 0.001),
                "expected 300.0, got {:?}",
                rc.items[0].expr
            );
        }
    }

    #[test]
    fn float_scientific_in_where() {
        // Scientific notation usable in WHERE predicates
        let q = parse_ok("MATCH (n) WHERE n.val > 1.5e-3 RETURN n");
        assert!(!q.clauses.is_empty());
    }

    // -- ORDER BY / SKIP / LIMIT (embedded in RETURN) --

    #[test]
    fn order_by_skip_limit() {
        let q = parse_ok("MATCH (n) RETURN n.name ORDER BY n.name DESC SKIP 10 LIMIT 25");
        assert!(q.clauses.iter().any(|c| matches!(c, Clause::OrderBy(_))));
        assert!(q.clauses.iter().any(|c| matches!(c, Clause::Skip(_))));
        assert!(q.clauses.iter().any(|c| matches!(c, Clause::Limit(_))));

        if let Some(Clause::OrderBy(ref items)) =
            q.clauses.iter().find(|c| matches!(c, Clause::OrderBy(_)))
        {
            assert_eq!(items.len(), 1);
            assert!(!items[0].ascending);
        }
    }

    // -- OPTIONAL MATCH --

    #[test]
    fn optional_match() {
        let q = parse_ok("MATCH (n:User) OPTIONAL MATCH (n)-[:KNOWS]->(m) RETURN n, m");
        assert!(matches!(q.clauses[0], Clause::Match(_)));
        assert!(matches!(q.clauses[1], Clause::OptionalMatch(_)));
    }

    // -- Parameters --

    #[test]
    fn parameter_in_where() {
        let q = parse_ok("MATCH (n) WHERE n.id = $userId RETURN n");
        if let Clause::Match(ref m) = q.clauses[0] {
            let w = m.where_clause.as_ref().unwrap();
            if let Expr::BinaryOp { right, .. } = w {
                assert!(matches!(**right, Expr::Parameter(ref s) if s == "userId"));
            } else {
                panic!("expected binary op");
            }
        }
    }

    #[test]
    fn parameter_in_properties() {
        let q = parse_ok("MATCH (n:User {email: $email}) RETURN n");
        if let Clause::Match(ref m) = q.clauses[0] {
            if let PatternElement::Node(ref np) = m.patterns[0].elements[0] {
                assert!(matches!(np.properties[0].1, Expr::Parameter(ref s) if s == "email"));
            }
        }
    }

    // -- Vector functions --

    #[test]
    fn vector_distance_function() {
        let q = parse_ok(
            "MATCH (m:Movie) WHERE vector_distance(m.embedding, $query_vec) < 0.3 RETURN m",
        );
        if let Clause::Match(ref m) = q.clauses[0] {
            let w = m.where_clause.as_ref().unwrap();
            if let Expr::BinaryOp { left, op, .. } = w {
                assert_eq!(*op, BinaryOperator::Lt);
                if let Expr::FunctionCall { name, args, .. } = left.as_ref() {
                    assert_eq!(name, "vector_distance");
                    assert_eq!(args.len(), 2);
                } else {
                    panic!("expected function call");
                }
            } else {
                panic!("expected binary op");
            }
        }
    }

    // -- AS OF TIMESTAMP --

    #[test]
    fn as_of_timestamp() {
        let q = parse_ok("MATCH (n:User) RETURN n AS OF TIMESTAMP '2025-06-15T10:00:00Z'");
        assert!(q
            .clauses
            .iter()
            .any(|c| matches!(c, Clause::AsOfTimestamp(_))));
    }

    // -- Complex queries --

    #[test]
    fn graph_plus_vector_query() {
        let q = parse_ok(
            "MATCH (u:User {id: $me})-[:LIKES]->(m:Movie) \
             WHERE vector_distance(m.embedding, $query_vec) < 0.3 \
             WITH m.genre AS genre, count(*) AS cnt \
             ORDER BY cnt DESC \
             LIMIT 10 \
             RETURN genre, cnt",
        );
        assert!(q.clauses.len() >= 4);
    }

    #[test]
    fn multiple_patterns() {
        let q = parse_ok("MATCH (a:User), (b:Movie) WHERE a.id = 1 RETURN a, b");
        if let Clause::Match(ref m) = q.clauses[0] {
            assert_eq!(m.patterns.len(), 2);
        }
    }

    // -- Arithmetic --

    #[test]
    fn arithmetic_expression() {
        let q = parse_ok("MATCH (n) RETURN n.price * 1.1 + 5 AS total");
        if let Clause::Return(ref rc) = q.clauses[1] {
            assert_eq!(rc.items[0].alias, Some("total".to_string()));
        }
    }

    // -- Literals --

    #[test]
    fn string_escape() {
        let q = parse_ok("MATCH (n) WHERE n.name = 'it\\'s' RETURN n");
        if let Clause::Match(ref m) = q.clauses[0] {
            let w = m.where_clause.as_ref().unwrap();
            if let Expr::BinaryOp { right, .. } = w {
                assert_eq!(**right, Expr::Literal(Value::String("it's".to_string())));
            }
        }
    }

    #[test]
    fn boolean_literals() {
        let q = parse_ok("MATCH (n) WHERE n.active = true RETURN n");
        if let Clause::Match(ref m) = q.clauses[0] {
            let w = m.where_clause.as_ref().unwrap();
            if let Expr::BinaryOp { right, .. } = w {
                assert_eq!(**right, Expr::Literal(Value::Bool(true)));
            }
        }
    }

    #[test]
    fn null_literal() {
        let q = parse_ok("MATCH (n) WHERE n.deleted = null RETURN n");
        if let Clause::Match(ref m) = q.clauses[0] {
            let w = m.where_clause.as_ref().unwrap();
            if let Expr::BinaryOp { right, .. } = w {
                assert_eq!(**right, Expr::Literal(Value::Null));
            }
        }
    }

    // -- CASE --

    #[test]
    fn case_expression() {
        let q = parse_ok(
            "MATCH (n) RETURN CASE WHEN n.age < 18 THEN 'minor' ELSE 'adult' END AS category",
        );
        if let Clause::Return(ref rc) = q.clauses[1] {
            assert!(matches!(rc.items[0].expr, Expr::Case { .. }));
        }
    }

    // -- Error cases --

    #[test]
    fn empty_query_fails() {
        let err = parse_err("");
        assert!(matches!(err, ParseError::Syntax { .. }));
    }

    #[test]
    fn invalid_syntax_fails() {
        let err = parse_err("MATCH RETURN n");
        assert!(matches!(
            err,
            ParseError::Syntax { .. } | ParseError::Invalid(_)
        ));
    }

    #[test]
    fn backtick_identifier() {
        let q = parse_ok("MATCH (n:`My Label`) RETURN n");
        if let Clause::Match(ref m) = q.clauses[0] {
            if let PatternElement::Node(ref np) = m.patterns[0].elements[0] {
                assert_eq!(np.labels, vec!["My Label"]);
            }
        }
    }

    // -- NOT expression --

    #[test]
    fn not_expression() {
        let q = parse_ok("MATCH (n) WHERE NOT n.active RETURN n");
        if let Clause::Match(ref m) = q.clauses[0] {
            let w = m.where_clause.as_ref().unwrap();
            assert!(matches!(
                w,
                Expr::UnaryOp {
                    op: UnaryOperator::Not,
                    ..
                }
            ));
        }
    }

    // -- Multiple relationship types --

    #[test]
    fn multiple_rel_types() {
        let q = parse_ok("MATCH (a)-[:KNOWS|:FOLLOWS]->(b) RETURN a, b");
        if let Clause::Match(ref m) = q.clauses[0] {
            if let PatternElement::Relationship(ref rp) = m.patterns[0].elements[1] {
                assert_eq!(rp.rel_types, vec!["KNOWS", "FOLLOWS"]);
            }
        }
    }

    // -- count(DISTINCT x) --

    #[test]
    fn count_distinct() {
        let q = parse_ok("MATCH (n) RETURN count(DISTINCT n.city) AS cities");
        if let Clause::Return(ref rc) = q.clauses[1] {
            if let Expr::FunctionCall {
                ref name, distinct, ..
            } = rc.items[0].expr
            {
                assert_eq!(name, "count");
                assert!(distinct);
            }
        }
    }

    // -- Negative number --

    #[test]
    fn negative_number() {
        let q = parse_ok("MATCH (n) WHERE n.temp > -10 RETURN n");
        if let Clause::Match(ref m) = q.clauses[0] {
            let w = m.where_clause.as_ref().unwrap();
            if let Expr::BinaryOp { right, .. } = w {
                assert!(matches!(
                    **right,
                    Expr::UnaryOp {
                        op: UnaryOperator::Neg,
                        ..
                    }
                ));
            }
        }
    }

    // -- Long chain of traversals --

    #[test]
    fn long_traversal_chain() {
        let q = parse_ok("MATCH (a)-[:KNOWS]->(b)-[:WORKS_AT]->(c) RETURN a, b, c");
        if let Clause::Match(ref m) = q.clauses[0] {
            // a, KNOWS, b, WORKS_AT, c
            assert_eq!(m.patterns[0].elements.len(), 5);
        }
    }

    // -- ENDS WITH / CONTAINS --

    #[test]
    fn where_ends_with() {
        let q = parse_ok("MATCH (n) WHERE n.email ENDS WITH '.com' RETURN n");
        if let Clause::Match(ref m) = q.clauses[0] {
            let w = m.where_clause.as_ref().unwrap();
            assert!(matches!(
                w,
                Expr::StringMatch {
                    op: StringOp::EndsWith,
                    ..
                }
            ));
        }
    }

    #[test]
    fn where_contains() {
        let q = parse_ok("MATCH (n) WHERE n.bio CONTAINS 'rust' RETURN n");
        if let Clause::Match(ref m) = q.clauses[0] {
            let w = m.where_clause.as_ref().unwrap();
            assert!(matches!(
                w,
                Expr::StringMatch {
                    op: StringOp::Contains,
                    ..
                }
            ));
        }
    }

    // -- Case insensitive keywords --

    #[test]
    fn case_insensitive_keywords() {
        let q = parse_ok("match (n:User) where n.age > 18 return n.name");
        assert!(matches!(q.clauses[0], Clause::Match(_)));
        assert!(matches!(q.clauses[1], Clause::Return(_)));
    }

    // -- Empty node pattern --

    #[test]
    fn empty_node() {
        let q = parse_ok("MATCH () RETURN *");
        if let Clause::Match(ref m) = q.clauses[0] {
            if let PatternElement::Node(ref np) = m.patterns[0].elements[0] {
                assert!(np.variable.is_none());
                assert!(np.labels.is_empty());
            }
        }
    }

    // -- Map literal in RETURN --

    #[test]
    fn map_literal_return() {
        let q = parse_ok("MATCH (n) RETURN {name: n.name, age: n.age} AS props");
        if let Clause::Return(ref rc) = q.clauses[1] {
            assert!(matches!(rc.items[0].expr, Expr::MapLiteral(_)));
        }
    }

    // ====== Write operations ======

    // -- CREATE --

    #[test]
    fn create_node() {
        let q = parse_ok("CREATE (n:User {name: 'Alice', age: 30})");
        if let Clause::Create(ref cc) = q.clauses[0] {
            assert_eq!(cc.patterns.len(), 1);
            if let PatternElement::Node(ref np) = cc.patterns[0].elements[0] {
                assert_eq!(np.labels, vec!["User"]);
                assert_eq!(np.properties.len(), 2);
            } else {
                panic!("expected node pattern");
            }
        } else {
            panic!("expected CREATE clause");
        }
    }

    #[test]
    fn create_node_and_relationship() {
        let q = parse_ok("CREATE (a:User {name: 'Alice'})-[:KNOWS]->(b:User {name: 'Bob'})");
        if let Clause::Create(ref cc) = q.clauses[0] {
            assert_eq!(cc.patterns[0].elements.len(), 3); // a, KNOWS, b
        } else {
            panic!("expected CREATE clause");
        }
    }

    #[test]
    fn create_multiple_patterns() {
        let q = parse_ok("CREATE (a:User), (b:Movie)");
        if let Clause::Create(ref cc) = q.clauses[0] {
            assert_eq!(cc.patterns.len(), 2);
        } else {
            panic!("expected CREATE clause");
        }
    }

    #[test]
    fn match_create() {
        let q = parse_ok(
            "MATCH (a:User {name: 'Alice'}), (b:User {name: 'Bob'}) CREATE (a)-[:KNOWS]->(b)",
        );
        assert!(matches!(q.clauses[0], Clause::Match(_)));
        assert!(matches!(q.clauses[1], Clause::Create(_)));
    }

    // -- MERGE --

    #[test]
    fn merge_simple() {
        let q = parse_ok("MERGE (n:User {email: 'alice@example.com'})");
        if let Clause::Merge(ref mc) = q.clauses[0] {
            if let PatternElement::Node(ref np) = mc.pattern.elements[0] {
                assert_eq!(np.labels, vec!["User"]);
                assert_eq!(np.properties.len(), 1);
            } else {
                panic!("expected node pattern");
            }
            assert!(mc.on_match.is_empty());
            assert!(mc.on_create.is_empty());
        } else {
            panic!("expected MERGE clause");
        }
    }

    #[test]
    fn merge_on_match_on_create() {
        let q = parse_ok(
            "MERGE (n:User {email: $email}) \
             ON MATCH SET n.login_count = n.login_count + 1 \
             ON CREATE SET n.login_count = 1, n.created = $now",
        );
        if let Clause::Merge(ref mc) = q.clauses[0] {
            assert_eq!(mc.on_match.len(), 1);
            assert_eq!(mc.on_create.len(), 2);
        } else {
            panic!("expected MERGE clause");
        }
    }

    #[test]
    fn merge_on_create_only() {
        let q = parse_ok("MERGE (n:User {email: $email}) ON CREATE SET n.created = $now");
        if let Clause::Merge(ref mc) = q.clauses[0] {
            assert!(mc.on_match.is_empty());
            assert_eq!(mc.on_create.len(), 1);
        } else {
            panic!("expected MERGE clause");
        }
    }

    // -- DELETE --

    #[test]
    fn delete_single() {
        let q = parse_ok("MATCH (n:User {id: 42}) DELETE n");
        if let Clause::Delete(ref dc) = q.clauses[1] {
            assert!(!dc.detach);
            assert_eq!(dc.exprs.len(), 1);
        } else {
            panic!("expected DELETE clause");
        }
    }

    #[test]
    fn detach_delete() {
        let q = parse_ok("MATCH (n:User {id: 42}) DETACH DELETE n");
        if let Clause::Delete(ref dc) = q.clauses[1] {
            assert!(dc.detach);
            assert_eq!(dc.exprs.len(), 1);
        } else {
            panic!("expected DELETE clause");
        }
    }

    #[test]
    fn delete_multiple() {
        let q = parse_ok("MATCH (a)-[r]->(b) DELETE a, r, b");
        if let Clause::Delete(ref dc) = q.clauses[1] {
            assert_eq!(dc.exprs.len(), 3);
        } else {
            panic!("expected DELETE clause");
        }
    }

    // -- SET --

    #[test]
    fn set_property() {
        let q = parse_ok("MATCH (n:User {id: 42}) SET n.name = 'Bob'");
        if let Clause::Set(ref items, _) = q.clauses[1] {
            assert_eq!(items.len(), 1);
            assert!(matches!(
                items[0],
                SetItem::Property {
                    ref variable,
                    ref property,
                    ..
                } if variable == "n" && property == "name"
            ));
        } else {
            panic!("expected SET clause");
        }
    }

    #[test]
    fn set_multiple_properties() {
        let q = parse_ok("MATCH (n) SET n.name = 'Bob', n.age = 30");
        if let Clause::Set(ref items, _) = q.clauses[1] {
            assert_eq!(items.len(), 2);
        } else {
            panic!("expected SET clause");
        }
    }

    #[test]
    fn set_merge_properties() {
        let q = parse_ok("MATCH (n) SET n += {name: 'Bob', age: 30}");
        if let Clause::Set(ref items, _) = q.clauses[1] {
            assert!(matches!(items[0], SetItem::MergeProperties { .. }));
        } else {
            panic!("expected SET clause");
        }
    }

    #[test]
    fn set_replace_properties() {
        let q = parse_ok("MATCH (n) SET n = {name: 'Bob'}");
        if let Clause::Set(ref items, _) = q.clauses[1] {
            assert!(matches!(items[0], SetItem::ReplaceProperties { .. }));
        } else {
            panic!("expected SET clause");
        }
    }

    #[test]
    fn set_label() {
        let q = parse_ok("MATCH (n) SET n:Admin");
        if let Clause::Set(ref items, _) = q.clauses[1] {
            assert!(matches!(
                items[0],
                SetItem::AddLabel {
                    ref variable,
                    ref label,
                    ..
                } if variable == "n" && label == "Admin"
            ));
        } else {
            panic!("expected SET clause");
        }
    }

    #[test]
    fn set_property_path_deep() {
        let q = parse_ok("MATCH (n) SET n.config.network.ssid = 'home'");
        if let Clause::Set(ref items, _) = q.clauses[1] {
            assert_eq!(items.len(), 1);
            assert!(matches!(
                items[0],
                SetItem::PropertyPath {
                    ref variable,
                    ref path,
                    ..
                } if variable == "n" && path == &["config", "network", "ssid"]
            ));
        } else {
            panic!("expected SET clause");
        }
    }

    #[test]
    fn set_property_path_two_levels_is_property() {
        // Two-level path (n.name) should still be SetItem::Property, not PropertyPath.
        let q = parse_ok("MATCH (n) SET n.name = 'Alice'");
        if let Clause::Set(ref items, _) = q.clauses[1] {
            assert!(matches!(items[0], SetItem::Property { .. }));
        } else {
            panic!("expected SET clause");
        }
    }

    #[test]
    fn set_multiple_with_path() {
        let q = parse_ok("MATCH (n) SET n.config.network.ssid = 'home', n.name = 'Alice'");
        if let Clause::Set(ref items, _) = q.clauses[1] {
            assert_eq!(items.len(), 2);
            assert!(matches!(items[0], SetItem::PropertyPath { .. }));
            assert!(matches!(items[1], SetItem::Property { .. }));
        } else {
            panic!("expected SET clause");
        }
    }

    // -- doc_* functions (R165) --

    #[test]
    fn set_doc_push() {
        let q = parse_ok("MATCH (n) SET doc_push(n.tags, 'new')");
        if let Clause::Set(ref items, _) = q.clauses[1] {
            assert!(matches!(
                items[0],
                SetItem::DocFunction {
                    ref function,
                    ref variable,
                    ref path,
                    ..
                } if function == "doc_push" && variable == "n" && path == &["tags"]
            ));
        } else {
            panic!("expected SET clause");
        }
    }

    #[test]
    fn set_doc_inc_nested_path() {
        let q = parse_ok("MATCH (n) SET doc_inc(n.stats.views, 1)");
        if let Clause::Set(ref items, _) = q.clauses[1] {
            assert!(matches!(
                items[0],
                SetItem::DocFunction {
                    ref function,
                    ref variable,
                    ref path,
                    ..
                } if function == "doc_inc" && variable == "n" && path == &["stats", "views"]
            ));
        } else {
            panic!("expected SET clause");
        }
    }

    #[test]
    fn set_multiple_doc_functions() {
        let q = parse_ok("MATCH (n) SET doc_push(n.tags, 'a'), doc_add_to_set(n.labels, 'b')");
        if let Clause::Set(ref items, _) = q.clauses[1] {
            assert_eq!(items.len(), 2);
            assert!(
                matches!(items[0], SetItem::DocFunction { ref function, .. } if function == "doc_push")
            );
            assert!(
                matches!(items[1], SetItem::DocFunction { ref function, .. } if function == "doc_add_to_set")
            );
        } else {
            panic!("expected SET clause");
        }
    }

    // -- REMOVE --

    #[test]
    fn remove_property() {
        let q = parse_ok("MATCH (n) REMOVE n.age");
        if let Clause::Remove(ref items) = q.clauses[1] {
            assert_eq!(items.len(), 1);
            assert!(matches!(
                items[0],
                RemoveItem::Property {
                    ref variable,
                    ref property,
                } if variable == "n" && property == "age"
            ));
        } else {
            panic!("expected REMOVE clause");
        }
    }

    #[test]
    fn remove_label() {
        let q = parse_ok("MATCH (n) REMOVE n:Admin");
        if let Clause::Remove(ref items) = q.clauses[1] {
            assert!(matches!(
                items[0],
                RemoveItem::Label {
                    ref variable,
                    ref label,
                } if variable == "n" && label == "Admin"
            ));
        } else {
            panic!("expected REMOVE clause");
        }
    }

    #[test]
    fn remove_multiple() {
        let q = parse_ok("MATCH (n) REMOVE n.age, n:Admin");
        if let Clause::Remove(ref items) = q.clauses[1] {
            assert_eq!(items.len(), 2);
            assert!(matches!(items[0], RemoveItem::Property { .. }));
            assert!(matches!(items[1], RemoveItem::Label { .. }));
        } else {
            panic!("expected REMOVE clause");
        }
    }

    #[test]
    fn remove_property_path_deep() {
        let q = parse_ok("MATCH (n) REMOVE n.config.network.ssid");
        if let Clause::Remove(ref items) = q.clauses[1] {
            assert_eq!(items.len(), 1);
            assert!(matches!(
                items[0],
                RemoveItem::PropertyPath {
                    ref variable,
                    ref path,
                } if variable == "n" && path == &["config", "network", "ssid"]
            ));
        } else {
            panic!("expected REMOVE clause");
        }
    }

    #[test]
    fn remove_property_two_levels_is_property() {
        let q = parse_ok("MATCH (n) REMOVE n.age");
        if let Clause::Remove(ref items) = q.clauses[1] {
            assert!(matches!(items[0], RemoveItem::Property { .. }));
        } else {
            panic!("expected REMOVE clause");
        }
    }

    // -- UPSERT MATCH --

    #[test]
    fn upsert_match() {
        let q = parse_ok(
            "UPSERT MATCH (u:User {email: 'alice@example.com'}) \
             ON MATCH SET u.login_count = u.login_count + 1 \
             ON CREATE CREATE (u:User {email: 'alice@example.com', login_count: 1}) \
             RETURN u",
        );
        if let Clause::Upsert(ref uc) = q.clauses[0] {
            assert_eq!(uc.on_match.len(), 1);
            assert_eq!(uc.on_create.len(), 1);
        } else {
            panic!("expected UPSERT clause");
        }
        assert!(matches!(q.clauses[1], Clause::Return(_)));
    }

    // -- Complex write queries --

    #[test]
    fn match_set_return() {
        let q = parse_ok("MATCH (n:User {id: $id}) SET n.name = $name, n.updated = $now RETURN n");
        assert!(matches!(q.clauses[0], Clause::Match(_)));
        assert!(matches!(q.clauses[1], Clause::Set(_, _)));
        assert!(matches!(q.clauses[2], Clause::Return(_)));
    }

    #[test]
    fn create_return() {
        let q = parse_ok("CREATE (n:User {name: 'Alice'}) RETURN n");
        assert!(matches!(q.clauses[0], Clause::Create(_)));
        assert!(matches!(q.clauses[1], Clause::Return(_)));
    }

    #[test]
    fn match_delete_return() {
        let q = parse_ok("MATCH (n:User {id: 42})-[r:KNOWS]->(m) DELETE r RETURN n, m");
        assert!(matches!(q.clauses[0], Clause::Match(_)));
        assert!(matches!(q.clauses[1], Clause::Delete(_)));
        assert!(matches!(q.clauses[2], Clause::Return(_)));
    }

    #[test]
    fn merge_with_relationship() {
        let q = parse_ok(
            "MATCH (a:User {name: 'Alice'}), (b:User {name: 'Bob'}) \
             MERGE (a)-[:KNOWS]->(b)",
        );
        assert!(matches!(q.clauses[0], Clause::Match(_)));
        assert!(matches!(q.clauses[1], Clause::Merge(_)));
    }

    // -- Dot-notation (multi-level property access) --

    #[test]
    fn dot_notation_two_levels() {
        // n.config.version → PropertyAccess(PropertyAccess(Variable("n"), "config"), "version")
        let q = parse_ok("MATCH (n) RETURN n.config.version");
        if let Clause::Return(ref ret) = q.clauses[1] {
            let item = &ret.items[0];
            if let Expr::PropertyAccess {
                expr: outer_expr,
                property: outer_prop,
            } = &item.expr
            {
                assert_eq!(outer_prop, "version");
                if let Expr::PropertyAccess {
                    expr: inner_expr,
                    property: inner_prop,
                } = outer_expr.as_ref()
                {
                    assert_eq!(inner_prop, "config");
                    assert!(matches!(inner_expr.as_ref(), Expr::Variable(v) if v == "n"));
                } else {
                    panic!("expected inner PropertyAccess, got: {outer_expr:?}");
                }
            } else {
                panic!("expected PropertyAccess, got: {:?}", item.expr);
            }
        }
    }

    #[test]
    fn dot_notation_three_levels() {
        // n.a.b.c → nested PropertyAccess chain
        let q = parse_ok("MATCH (n) RETURN n.a.b.c");
        if let Clause::Return(ref ret) = q.clauses[1] {
            let item = &ret.items[0];
            // Outermost: PropertyAccess { .., property: "c" }
            if let Expr::PropertyAccess { property, expr } = &item.expr {
                assert_eq!(property, "c");
                // Middle: PropertyAccess { .., property: "b" }
                if let Expr::PropertyAccess { property, expr } = expr.as_ref() {
                    assert_eq!(property, "b");
                    // Inner: PropertyAccess { Variable("n"), property: "a" }
                    if let Expr::PropertyAccess { property, expr } = expr.as_ref() {
                        assert_eq!(property, "a");
                        assert!(matches!(expr.as_ref(), Expr::Variable(v) if v == "n"));
                    } else {
                        panic!("expected PropertyAccess for 'a'");
                    }
                } else {
                    panic!("expected PropertyAccess for 'b'");
                }
            } else {
                panic!("expected PropertyAccess for 'c'");
            }
        }
    }

    #[test]
    fn dot_notation_in_where() {
        // WHERE n.config.enabled = true
        let q = parse_ok("MATCH (n) WHERE n.config.enabled = true RETURN n");
        if let Clause::Match(ref m) = q.clauses[0] {
            let w = m.where_clause.as_ref().expect("WHERE clause");
            // Should be BinaryOp { left: PropertyAccess chain, op: Eq, right: Literal(true) }
            if let Expr::BinaryOp { left, op, .. } = w {
                assert_eq!(*op, BinaryOperator::Eq);
                if let Expr::PropertyAccess { property, .. } = left.as_ref() {
                    assert_eq!(property, "enabled");
                } else {
                    panic!("expected PropertyAccess, got: {left:?}");
                }
            } else {
                panic!("expected BinaryOp");
            }
        }
    }

    // -- Map projection --

    #[test]
    fn map_projection_shorthand() {
        let q = parse_ok("MATCH (n:User) RETURN n { .name, .age }");
        if let Clause::Return(ref rc) = q.clauses[1] {
            if let Expr::MapProjection { expr, items } = &rc.items[0].expr {
                assert!(matches!(expr.as_ref(), Expr::Variable(v) if v == "n"));
                assert_eq!(items.len(), 2);
                assert!(matches!(&items[0], MapProjectionItem::Property(p) if p == "name"));
                assert!(matches!(&items[1], MapProjectionItem::Property(p) if p == "age"));
            } else {
                panic!("expected MapProjection, got: {:?}", rc.items[0].expr);
            }
        }
    }

    #[test]
    fn map_projection_with_computed() {
        let q = parse_ok(
            "MATCH (n:User)-[:WROTE]->(p:Post) RETURN n { .name, posts: collect(p.title) }",
        );
        if let Clause::Return(ref rc) = q.clauses[1] {
            if let Expr::MapProjection { items, .. } = &rc.items[0].expr {
                assert_eq!(items.len(), 2);
                assert!(matches!(&items[0], MapProjectionItem::Property(p) if p == "name"));
                assert!(
                    matches!(&items[1], MapProjectionItem::Computed(alias, _) if alias == "posts")
                );
            } else {
                panic!("expected MapProjection");
            }
        }
    }

    #[test]
    fn map_projection_nested() {
        // Nested map projection: collect(p { .title, .body })
        let q = parse_ok(
            "MATCH (u:User)-[:WROTE]->(p:Post) \
             RETURN u { .name, posts: collect(p { .title, .body }) }",
        );
        if let Clause::Return(ref rc) = q.clauses[1] {
            if let Expr::MapProjection { items, .. } = &rc.items[0].expr {
                assert_eq!(items.len(), 2);
                // Second item: posts: collect(p { .title, .body })
                if let MapProjectionItem::Computed(ref alias, ref cexpr) = items[1] {
                    assert_eq!(alias, "posts");
                    // collect(p { .title, .body })
                    assert!(matches!(cexpr, Expr::FunctionCall { name, .. } if name == "collect"));
                } else {
                    panic!("expected Computed item");
                }
            } else {
                panic!("expected MapProjection");
            }
        }
    }

    // ── Per-query hint extraction (G026) ───────────────────────────────

    #[test]
    fn hint_vector_consistency_snapshot() {
        let q = parse_ok("MATCH (m:Movie) RETURN m /*+ vector_consistency('snapshot') */");
        assert_eq!(q.hints.len(), 1);
        assert_eq!(
            q.hints[0],
            QueryHint::VectorConsistency(
                coordinode_core::graph::types::VectorConsistencyMode::Snapshot
            )
        );
    }

    #[test]
    fn hint_vector_consistency_exact() {
        let q = parse_ok("/*+ vector_consistency('exact') */ MATCH (n:Node) RETURN n");
        assert_eq!(q.hints.len(), 1);
        assert_eq!(
            q.hints[0],
            QueryHint::VectorConsistency(
                coordinode_core::graph::types::VectorConsistencyMode::Exact
            )
        );
    }

    #[test]
    fn hint_unknown_key_ignored() {
        let q = parse_ok("MATCH (n) RETURN n /*+ unknown_hint('value') */");
        assert!(
            q.hints.is_empty(),
            "unknown hints should be silently ignored"
        );
    }

    #[test]
    fn hint_no_hints() {
        let q = parse_ok("MATCH (n) RETURN n");
        assert!(q.hints.is_empty());
    }

    #[test]
    fn hint_with_regular_comment() {
        // Regular comments should still be ignored, only /*+ ... */ are hints
        let q = parse_ok(
            "/* regular comment */ MATCH (n) RETURN n /*+ vector_consistency('snapshot') */",
        );
        assert_eq!(q.hints.len(), 1);
    }

    #[test]
    fn hint_double_quoted_value() {
        let q = parse_ok("MATCH (n) RETURN n /*+ vector_consistency(\"current\") */");
        assert_eq!(q.hints.len(), 1);
        assert_eq!(
            q.hints[0],
            QueryHint::VectorConsistency(
                coordinode_core::graph::types::VectorConsistencyMode::Current
            )
        );
    }

    // --- CREATE TEXT INDEX DDL (G016) ---

    #[test]
    fn create_text_index_simple_syntax() {
        let q = parse_ok("CREATE TEXT INDEX article_body ON :Article(body)");
        assert_eq!(q.clauses.len(), 1);
        match &q.clauses[0] {
            Clause::CreateTextIndex(c) => {
                assert_eq!(c.name, "article_body");
                assert_eq!(c.label, "Article");
                assert_eq!(c.fields.len(), 1);
                assert_eq!(c.fields[0].property, "body");
                assert!(c.fields[0].analyzer.is_none());
                assert!(c.default_language.is_none());
                assert!(c.language_override.is_none());
            }
            other => panic!("expected CreateTextIndex, got {other:?}"),
        }
    }

    #[test]
    fn create_text_index_simple_with_language() {
        let q = parse_ok("CREATE TEXT INDEX idx ON :Article(body) LANGUAGE 'russian'");
        match &q.clauses[0] {
            Clause::CreateTextIndex(c) => {
                assert_eq!(c.fields[0].property, "body");
                assert_eq!(c.default_language.as_deref(), Some("russian"));
            }
            other => panic!("expected CreateTextIndex, got {other:?}"),
        }
    }

    #[test]
    fn create_text_index_multi_field() {
        let q = parse_ok(
            r#"CREATE TEXT INDEX article_text ON :Article {
                title: { analyzer: "english" },
                body:  { analyzer: "auto_detect" }
            } DEFAULT LANGUAGE "english""#,
        );
        match &q.clauses[0] {
            Clause::CreateTextIndex(c) => {
                assert_eq!(c.name, "article_text");
                assert_eq!(c.label, "Article");
                assert_eq!(c.fields.len(), 2);
                assert_eq!(c.fields[0].property, "title");
                assert_eq!(c.fields[0].analyzer.as_deref(), Some("english"));
                assert_eq!(c.fields[1].property, "body");
                assert_eq!(c.fields[1].analyzer.as_deref(), Some("auto_detect"));
                assert_eq!(c.default_language.as_deref(), Some("english"));
            }
            other => panic!("expected CreateTextIndex, got {other:?}"),
        }
    }

    #[test]
    fn create_text_index_multi_field_with_override() {
        let q = parse_ok(
            r#"CREATE TEXT INDEX idx ON :Article {
                title: { analyzer: "russian" }
            } DEFAULT LANGUAGE "english" LANGUAGE OVERRIDE "lang""#,
        );
        match &q.clauses[0] {
            Clause::CreateTextIndex(c) => {
                assert_eq!(c.default_language.as_deref(), Some("english"));
                assert_eq!(c.language_override.as_deref(), Some("lang"));
            }
            other => panic!("expected CreateTextIndex, got {other:?}"),
        }
    }

    #[test]
    fn create_text_index_multi_field_no_modifiers() {
        let q = parse_ok(r#"CREATE TEXT INDEX idx ON :Post { content: { analyzer: "german" } }"#);
        match &q.clauses[0] {
            Clause::CreateTextIndex(c) => {
                assert_eq!(c.fields.len(), 1);
                assert_eq!(c.fields[0].property, "content");
                assert_eq!(c.fields[0].analyzer.as_deref(), Some("german"));
                assert!(c.default_language.is_none());
                assert!(c.language_override.is_none());
            }
            other => panic!("expected CreateTextIndex, got {other:?}"),
        }
    }

    // ── Pattern predicate parsing ────────────────────────────────────────

    /// Helper: extract inline WHERE expression from first MATCH clause.
    fn extract_match_where(q: &Query) -> Option<&Expr> {
        q.clauses.iter().find_map(|c| match c {
            Clause::Match(m) => m.where_clause.as_ref(),
            _ => None,
        })
    }

    #[test]
    fn parse_where_pattern_predicate() {
        let q = parse("MATCH (a), (b) WHERE (a)-[:KNOWS]->(b) RETURN a").unwrap();
        let where_expr = extract_match_where(&q).expect("WHERE clause missing");
        if let Expr::PatternPredicate(p) = where_expr {
            assert_eq!(p.elements.len(), 3); // node, rel, node
        } else {
            panic!("expected PatternPredicate, got {where_expr:?}");
        }
    }

    #[test]
    fn parse_where_not_pattern_predicate() {
        let q = parse("MATCH (a), (b) WHERE NOT (a)-[:KNOWS]->(b) RETURN a").unwrap();
        let where_expr = extract_match_where(&q).expect("WHERE clause missing");
        if let Expr::UnaryOp {
            op: UnaryOperator::Not,
            expr,
        } = where_expr
        {
            assert!(
                matches!(expr.as_ref(), Expr::PatternPredicate(_)),
                "expected PatternPredicate inside NOT, got {expr:?}"
            );
        } else {
            panic!("expected NOT expression, got {where_expr:?}");
        }
    }

    #[test]
    fn parse_pattern_predicate_with_labels() {
        let q = parse("MATCH (a:Person), (b:Person) WHERE (a)-[:KNOWS]->(b) RETURN a").unwrap();
        let where_expr = extract_match_where(&q).expect("WHERE clause missing");
        assert!(
            matches!(where_expr, Expr::PatternPredicate(_)),
            "expected PatternPredicate"
        );
    }

    #[test]
    fn parse_pattern_predicate_undirected() {
        let q = parse("MATCH (a), (b) WHERE (a)-[:KNOWS]-(b) RETURN a").unwrap();
        let where_expr = extract_match_where(&q).expect("WHERE clause missing");
        if let Expr::PatternPredicate(p) = where_expr {
            if let PatternElement::Relationship(rel) = &p.elements[1] {
                assert_eq!(rel.direction, Direction::Both);
            } else {
                panic!("expected relationship element");
            }
        } else {
            panic!("expected PatternPredicate");
        }
    }

    #[test]
    fn parse_pattern_predicate_and_scalar() {
        let q = parse("MATCH (a), (b) WHERE (a)-[:KNOWS]->(b) AND a.age > 30 RETURN a").unwrap();
        let where_expr = extract_match_where(&q).expect("WHERE clause missing");
        if let Expr::BinaryOp { left, op, right } = where_expr {
            assert_eq!(*op, BinaryOperator::And);
            assert!(
                matches!(left.as_ref(), Expr::PatternPredicate(_)),
                "left should be PatternPredicate"
            );
            assert!(matches!(right.as_ref(), Expr::BinaryOp { .. }));
        } else {
            panic!("expected AND expression, got {where_expr:?}");
        }
    }

    #[test]
    fn parse_parenthesized_expr_not_pattern_predicate() {
        let q = parse("MATCH (n) WHERE (n.age) > 5 RETURN n").unwrap();
        let where_expr = extract_match_where(&q).expect("WHERE clause missing");
        assert!(
            !matches!(where_expr, Expr::PatternPredicate(_)),
            "parenthesized expression should not be PatternPredicate"
        );
    }

    // --- CREATE INDEX / DROP INDEX DDL (R-API2) ---

    #[test]
    fn create_index_simple() {
        let q = parse_ok("CREATE INDEX email_idx ON :User(email)");
        assert_eq!(q.clauses.len(), 1);
        match &q.clauses[0] {
            Clause::CreateIndex(c) => {
                assert_eq!(c.name, "email_idx");
                assert_eq!(c.label, "User");
                assert_eq!(c.property, "email");
                assert!(!c.unique);
                assert!(!c.sparse);
                assert!(c.filter_expr.is_none());
            }
            other => panic!("expected CreateIndex, got {other:?}"),
        }
    }

    #[test]
    fn create_unique_index() {
        let q = parse_ok("CREATE UNIQUE INDEX email_idx ON :User(email)");
        match &q.clauses[0] {
            Clause::CreateIndex(c) => {
                assert_eq!(c.name, "email_idx");
                assert!(c.unique, "expected unique=true");
                assert!(!c.sparse);
            }
            other => panic!("expected CreateIndex, got {other:?}"),
        }
    }

    #[test]
    fn create_sparse_index() {
        let q = parse_ok("CREATE SPARSE INDEX opt_idx ON :User(optional_prop)");
        match &q.clauses[0] {
            Clause::CreateIndex(c) => {
                assert_eq!(c.name, "opt_idx");
                assert!(c.sparse, "expected sparse=true");
                assert!(!c.unique);
            }
            other => panic!("expected CreateIndex, got {other:?}"),
        }
    }

    #[test]
    fn create_unique_sparse_index() {
        let q = parse_ok("CREATE UNIQUE SPARSE INDEX us_idx ON :Item(code)");
        match &q.clauses[0] {
            Clause::CreateIndex(c) => {
                assert!(c.unique);
                assert!(c.sparse);
                assert_eq!(c.label, "Item");
                assert_eq!(c.property, "code");
            }
            other => panic!("expected CreateIndex, got {other:?}"),
        }
    }

    #[test]
    fn create_index_with_where_clause() {
        let q = parse_ok("CREATE INDEX active_users ON :User(email) WHERE n.active = true");
        match &q.clauses[0] {
            Clause::CreateIndex(c) => {
                assert_eq!(c.name, "active_users");
                // The filter_expr should be present (partial index).
                assert!(
                    c.filter_expr.is_some(),
                    "expected filter_expr from WHERE clause"
                );
            }
            other => panic!("expected CreateIndex, got {other:?}"),
        }
    }

    #[test]
    fn drop_index_simple() {
        let q = parse_ok("DROP INDEX email_idx");
        assert_eq!(q.clauses.len(), 1);
        match &q.clauses[0] {
            Clause::DropIndex(c) => {
                assert_eq!(c.name, "email_idx");
            }
            other => panic!("expected DropIndex, got {other:?}"),
        }
    }

    #[test]
    fn create_index_does_not_shadow_create_node() {
        // Verify that CREATE INDEX DDL does not interfere with regular CREATE (node) parsing.
        let q = parse_ok("CREATE (n:User {email: 'alice@example.com'})");
        assert_eq!(q.clauses.len(), 1);
        assert!(
            matches!(q.clauses[0], Clause::Create(_)),
            "regular CREATE should still parse as Clause::Create"
        );
    }

    #[test]
    fn subscript_access_on_function_call() {
        // labels(n)[0] → Subscript { expr: FunctionCall("labels", [Variable("n")]), index: Literal(0) }
        let q = parse_ok("MATCH (n) RETURN labels(n)[0] AS lbl");
        if let Clause::Return(ref rc) = q.clauses[1] {
            if let Expr::Subscript {
                ref expr,
                ref index,
            } = rc.items[0].expr
            {
                assert!(
                    matches!(**expr, Expr::FunctionCall { ref name, .. } if name == "labels"),
                    "base must be FunctionCall(labels), got {expr:?}"
                );
                assert_eq!(**index, Expr::Literal(Value::Int(0)), "index must be 0");
            } else {
                panic!("expected Subscript expr, got {:?}", rc.items[0].expr);
            }
        } else {
            panic!("expected RETURN clause at index 1");
        }
    }

    #[test]
    fn subscript_access_on_list_literal() {
        // [1, 2, 3][1] → Subscript { expr: List([1,2,3]), index: Literal(1) }
        let q = parse_ok("RETURN [1, 2, 3][1] AS x");
        if let Clause::Return(ref rc) = q.clauses[0] {
            assert!(
                matches!(rc.items[0].expr, Expr::Subscript { .. }),
                "expected Subscript, got {:?}",
                rc.items[0].expr
            );
        } else {
            panic!("expected RETURN clause");
        }
    }

    #[test]
    fn chained_subscript_access() {
        // matrix[0][1] → Subscript { Subscript { Variable("matrix"), 0 }, 1 }
        let q = parse_ok("RETURN matrix[0][1] AS v");
        if let Clause::Return(ref rc) = q.clauses[0] {
            if let Expr::Subscript {
                ref expr,
                ref index,
            } = rc.items[0].expr
            {
                assert_eq!(**index, Expr::Literal(Value::Int(1)));
                assert!(
                    matches!(**expr, Expr::Subscript { .. }),
                    "inner must also be Subscript, got {expr:?}"
                );
            } else {
                panic!("expected outer Subscript, got {:?}", rc.items[0].expr);
            }
        }
    }

    // ====== DETACH DOCUMENT (R167) ======

    #[test]
    fn detach_document_basic() {
        let q = parse_ok(
            "MATCH (n:User) \
             DETACH DOCUMENT n.address AS (a:Address)-[:HAS_ADDRESS]->(n)",
        );
        let Clause::DetachDocument(ref dd) = q.clauses[1] else {
            panic!("expected DetachDocument, got {:?}", q.clauses[1]);
        };
        assert_eq!(dd.source_variable, "n");
        assert_eq!(dd.property_path, vec!["address"]);
        assert_eq!(dd.target_variable, "a");
        assert_eq!(dd.target_labels, vec!["Address"]);
        assert_eq!(dd.edge_type.as_deref(), Some("HAS_ADDRESS"));
        // (a:Address)-[:HAS_ADDRESS]->(n): edge goes a → n, so from `n`'s
        // perspective it is incoming.
        assert_eq!(dd.edge_direction, EdgeFromSource::Incoming);
        assert!(dd.transfer.is_none());
    }

    #[test]
    fn detach_document_nested_path() {
        let q = parse_ok(
            "MATCH (n:User) \
             DETACH DOCUMENT n.meta.shipping AS (s:ShippingAddress)-[:HAS_SHIPPING]->(n)",
        );
        let Clause::DetachDocument(ref dd) = q.clauses[1] else {
            panic!("expected DetachDocument");
        };
        assert_eq!(dd.property_path, vec!["meta", "shipping"]);
        assert_eq!(dd.target_variable, "s");
    }

    #[test]
    fn detach_document_with_transfer_edges_in_list() {
        let q = parse_ok(
            "MATCH (n:User) \
             DETACH DOCUMENT n.address AS (a:Address)-[:HAS_ADDRESS]->(n) \
             TRANSFER EDGES ON n TO a WHERE type(r) IN ['SHIPS_TO', 'LIVES_AT']",
        );
        let Clause::DetachDocument(ref dd) = q.clauses[1] else {
            panic!("expected DetachDocument");
        };
        let t = dd.transfer.as_ref().expect("transfer spec");
        assert_eq!(t.node_variable, "n");
        assert_eq!(t.target_variable, "a");
        // predicate must be `type(r) IN [...]`.
        assert!(matches!(t.predicate, Expr::In { .. }));
    }

    #[test]
    fn detach_document_requires_property_path() {
        // Just `n` alone — grammar requires at least one `.segment`.
        let err = parse_err(
            "MATCH (n:User) \
             DETACH DOCUMENT n AS (a:Address)-[:HAS_ADDRESS]->(n)",
        );
        // Any parse error is acceptable — the important part is that the query
        // does not parse successfully.
        let _ = err;
    }

    #[test]
    fn detach_document_reverse_relationship() {
        // Mirror form: (n)<-[:HAS_ADDRESS]-(a:Address) — same semantics.
        let q = parse_ok(
            "MATCH (n:User) \
             DETACH DOCUMENT n.address AS (n)<-[:HAS_ADDRESS]-(a:Address)",
        );
        let Clause::DetachDocument(ref dd) = q.clauses[1] else {
            panic!("expected DetachDocument");
        };
        assert_eq!(dd.target_variable, "a");
        assert_eq!(dd.target_labels, vec!["Address"]);
        // (n)<-[:HAS_ADDRESS]-(a): edge goes a → n, still Incoming for n.
        assert_eq!(dd.edge_direction, EdgeFromSource::Incoming);
    }

    // ====== ATTACH DOCUMENT (R168) ======

    #[test]
    fn attach_document_basic() {
        let q = parse_ok("ATTACH (a:Address)-[:HAS_ADDRESS]->(u:User) INTO u.address");
        let Clause::AttachDocument(ref ad) = q.clauses[0] else {
            panic!("expected AttachDocument, got {:?}", q.clauses[0]);
        };
        assert_eq!(ad.source_variable, "a");
        assert_eq!(ad.source_labels, vec!["Address"]);
        assert_eq!(ad.target_variable, "u");
        assert_eq!(ad.target_labels, vec!["User"]);
        assert_eq!(ad.edge_type, "HAS_ADDRESS");
        assert_eq!(ad.edge_direction, EdgeFromSource::Outgoing);
        assert_eq!(ad.target_property_variable, "u");
        assert_eq!(ad.target_property_path, vec!["address"]);
        assert!(ad.transfer.is_none());
        assert!(!ad.on_conflict_replace);
        assert!(!ad.on_remaining_fail);
    }

    #[test]
    fn attach_document_with_transfer_and_options() {
        let q = parse_ok(
            "ATTACH (a:Address)-[:HAS_ADDRESS]->(u:User) INTO u.address \
             TRANSFER EDGES ON a TO u WHERE type(r) = 'SHIPS_TO' \
             ON CONFLICT REPLACE \
             ON REMAINING FAIL",
        );
        let Clause::AttachDocument(ref ad) = q.clauses[0] else {
            panic!("expected AttachDocument");
        };
        assert!(ad.transfer.is_some());
        assert!(ad.on_conflict_replace);
        assert!(ad.on_remaining_fail);
    }

    #[test]
    fn attach_document_nested_target_path() {
        let q = parse_ok("ATTACH (a:Shipping)-[:HAS_SHIPPING]->(u:User) INTO u.meta.shipping");
        let Clause::AttachDocument(ref ad) = q.clauses[0] else {
            panic!("expected AttachDocument");
        };
        assert_eq!(ad.target_property_path, vec!["meta", "shipping"]);
    }

    #[test]
    fn attach_document_target_var_must_match_pattern() {
        // INTO target variable `x` doesn't match pattern target `u` → parse error.
        let err = parse_err("ATTACH (a:Address)-[:HAS_ADDRESS]->(u:User) INTO x.address");
        let msg = format!("{err}");
        assert!(
            msg.contains("match pattern target") || msg.contains("target node variable"),
            "error should mention mismatch: {msg}"
        );
    }

    #[test]
    fn attach_document_requires_edge_type() {
        // Anonymous relationship `-[]->` has no type → build error (ATTACH
        // requires an explicit type for targeted adjacency delete).
        let err = parse_err("ATTACH (a:Address)-[]->(u:User) INTO u.address");
        let _ = err;
    }

    #[test]
    fn on_violation_skip_parsed() {
        // SET ... ON VIOLATION SKIP should set ViolationMode::Skip.
        let q = parse_ok("MATCH (n:X) SET n.y = 1 ON VIOLATION SKIP");
        if let Clause::Set(_, violation_mode) = &q.clauses[1] {
            use crate::cypher::ast::ViolationMode;
            assert_eq!(
                *violation_mode,
                ViolationMode::Skip,
                "ON VIOLATION SKIP should set Skip mode"
            );
        } else {
            panic!("expected Clause::Set at index 1, got {:?}", q.clauses[1]);
        }
    }
}
