//! Query normalization and fingerprinting.
//!
//! Normalizes Cypher queries by replacing literal values with `$` placeholders,
//! then computes a 64-bit fingerprint (truncated SHA-256) of the canonical form.
//! This groups queries that differ only by parameter values under a single fingerprint.

use sha2::{Digest, Sha256};

use crate::cypher::ast::*;

/// Normalize an AST query into a canonical string with literals replaced by `$`.
///
/// The canonical form preserves query structure (labels, relationship types,
/// property names, functions) but replaces all literal values with `$`.
/// This means `MATCH (n:User {id: 42})` and `MATCH (n:User {id: 99})`
/// produce the same canonical string.
pub fn normalize(query: &Query) -> String {
    let mut buf = String::with_capacity(256);
    write_query(&mut buf, query);
    buf
}

/// Compute a 64-bit fingerprint from a normalized query string.
///
/// Uses the first 8 bytes of SHA-256 as a compact fingerprint.
/// Collision probability is ~1/2^64 which is acceptable for a 1K-entry registry.
pub fn fingerprint(normalized: &str) -> u64 {
    let hash = Sha256::digest(normalized.as_bytes());
    u64::from_be_bytes([
        hash[0], hash[1], hash[2], hash[3], hash[4], hash[5], hash[6], hash[7],
    ])
}

/// Normalize an AST query and compute its fingerprint in one pass.
pub fn normalize_and_fingerprint(query: &Query) -> (String, u64) {
    let canonical = normalize(query);
    let fp = fingerprint(&canonical);
    (canonical, fp)
}

// --- AST → canonical string writers ---

fn write_query(buf: &mut String, query: &Query) {
    for (i, clause) in query.clauses.iter().enumerate() {
        if i > 0 {
            buf.push(' ');
        }
        write_clause(buf, clause);
    }
}

fn write_clause(buf: &mut String, clause: &Clause) {
    match clause {
        Clause::Match(m) => {
            buf.push_str("MATCH ");
            write_patterns(buf, &m.patterns);
            if let Some(ref w) = m.where_clause {
                buf.push_str(" WHERE ");
                write_expr(buf, w);
            }
        }
        Clause::OptionalMatch(m) => {
            buf.push_str("OPTIONAL MATCH ");
            write_patterns(buf, &m.patterns);
            if let Some(ref w) = m.where_clause {
                buf.push_str(" WHERE ");
                write_expr(buf, w);
            }
        }
        Clause::Where(expr) => {
            buf.push_str("WHERE ");
            write_expr(buf, expr);
        }
        Clause::Return(ret) => {
            buf.push_str("RETURN ");
            if ret.distinct {
                buf.push_str("DISTINCT ");
            }
            write_return_items(buf, &ret.items);
        }
        Clause::With(with) => {
            buf.push_str("WITH ");
            if with.distinct {
                buf.push_str("DISTINCT ");
            }
            write_return_items(buf, &with.items);
            if let Some(ref w) = with.where_clause {
                buf.push_str(" WHERE ");
                write_expr(buf, w);
            }
        }
        Clause::Unwind(u) => {
            buf.push_str("UNWIND ");
            write_expr(buf, &u.expr);
            buf.push_str(" AS ");
            buf.push_str(&u.variable);
        }
        Clause::OrderBy(items) => {
            buf.push_str("ORDER BY ");
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    buf.push_str(", ");
                }
                write_expr(buf, &item.expr);
                if !item.ascending {
                    buf.push_str(" DESC");
                }
            }
        }
        Clause::Skip(expr) => {
            buf.push_str("SKIP ");
            write_expr(buf, expr);
        }
        Clause::Limit(expr) => {
            buf.push_str("LIMIT ");
            write_expr(buf, expr);
        }
        Clause::AsOfTimestamp(expr) => {
            buf.push_str("AS OF TIMESTAMP ");
            write_expr(buf, expr);
        }
        Clause::Create(c) => {
            buf.push_str("CREATE ");
            write_patterns(buf, &c.patterns);
        }
        Clause::Merge(m) => {
            buf.push_str("MERGE ");
            write_pattern(buf, &m.pattern);
            if !m.on_match.is_empty() {
                buf.push_str(" ON MATCH SET ");
                write_set_items(buf, &m.on_match);
            }
            if !m.on_create.is_empty() {
                buf.push_str(" ON CREATE SET ");
                write_set_items(buf, &m.on_create);
            }
        }
        Clause::MergeMany(m) => {
            buf.push_str("MERGE ALL ");
            write_pattern(buf, &m.pattern);
            if !m.on_match.is_empty() {
                buf.push_str(" ON MATCH SET ");
                write_set_items(buf, &m.on_match);
            }
            if !m.on_create.is_empty() {
                buf.push_str(" ON CREATE SET ");
                write_set_items(buf, &m.on_create);
            }
        }
        Clause::Upsert(u) => {
            buf.push_str("UPSERT ");
            write_pattern(buf, &u.pattern);
            if !u.on_match.is_empty() {
                buf.push_str(" ON MATCH SET ");
                write_set_items(buf, &u.on_match);
            }
            if !u.on_create.is_empty() {
                buf.push_str(" ON CREATE CREATE ");
                write_patterns(buf, &u.on_create);
            }
        }
        Clause::Delete(d) => {
            if d.detach {
                buf.push_str("DETACH ");
            }
            buf.push_str("DELETE ");
            for (i, expr) in d.exprs.iter().enumerate() {
                if i > 0 {
                    buf.push_str(", ");
                }
                write_expr(buf, expr);
            }
        }
        Clause::Set(items) => {
            buf.push_str("SET ");
            write_set_items(buf, items);
        }
        Clause::Remove(items) => {
            buf.push_str("REMOVE ");
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    buf.push_str(", ");
                }
                match item {
                    RemoveItem::Property { variable, property } => {
                        buf.push_str(variable);
                        buf.push('.');
                        buf.push_str(property);
                    }
                    RemoveItem::PropertyPath { variable, path } => {
                        buf.push_str(variable);
                        for seg in path {
                            buf.push('.');
                            buf.push_str(seg);
                        }
                    }
                    RemoveItem::Label { variable, label } => {
                        buf.push_str(variable);
                        buf.push(':');
                        buf.push_str(label);
                    }
                }
            }
        }
        Clause::Call(cc) => {
            buf.push_str("CALL ");
            buf.push_str(&cc.procedure);
            buf.push('(');
            for (i, arg) in cc.args.iter().enumerate() {
                if i > 0 {
                    buf.push_str(", ");
                }
                write_expr(buf, arg);
            }
            buf.push(')');
        }
        Clause::AlterLabel(ac) => {
            buf.push_str("ALTER LABEL ");
            buf.push_str(&ac.label);
            buf.push_str(" SET SCHEMA ");
            buf.push_str(&ac.mode);
        }
        Clause::CreateTextIndex(c) => {
            buf.push_str("CREATE TEXT INDEX ");
            buf.push_str(&c.name);
            buf.push_str(" ON :");
            buf.push_str(&c.label);
            buf.push('(');
            let props: Vec<&str> = c.fields.iter().map(|f| f.property.as_str()).collect();
            buf.push_str(&props.join(", "));
            buf.push(')');
        }
        Clause::DropTextIndex(c) => {
            buf.push_str("DROP TEXT INDEX ");
            buf.push_str(&c.name);
        }
        Clause::CreateEncryptedIndex(c) => {
            buf.push_str("CREATE ENCRYPTED INDEX ");
            buf.push_str(&c.name);
            buf.push_str(" ON :");
            buf.push_str(&c.label);
            buf.push('(');
            buf.push_str(&c.property);
            buf.push(')');
        }
        Clause::DropEncryptedIndex(c) => {
            buf.push_str("DROP ENCRYPTED INDEX ");
            buf.push_str(&c.name);
        }
    }
}

fn write_patterns(buf: &mut String, patterns: &[Pattern]) {
    for (i, pattern) in patterns.iter().enumerate() {
        if i > 0 {
            buf.push_str(", ");
        }
        write_pattern(buf, pattern);
    }
}

fn write_pattern(buf: &mut String, pattern: &Pattern) {
    for elem in &pattern.elements {
        match elem {
            PatternElement::Node(n) => {
                buf.push('(');
                if let Some(ref var) = n.variable {
                    buf.push_str(var);
                }
                for label in &n.labels {
                    buf.push(':');
                    buf.push_str(label);
                }
                if !n.properties.is_empty() {
                    buf.push_str(" {");
                    write_prop_map(buf, &n.properties);
                    buf.push('}');
                }
                buf.push(')');
            }
            PatternElement::Relationship(r) => {
                match r.direction {
                    Direction::Incoming => buf.push_str("<-["),
                    Direction::Outgoing | Direction::Both => buf.push_str("-["),
                }
                if let Some(ref var) = r.variable {
                    buf.push_str(var);
                }
                for (i, rt) in r.rel_types.iter().enumerate() {
                    if i == 0 {
                        buf.push(':');
                    } else {
                        buf.push('|');
                    }
                    buf.push_str(rt);
                }
                if let Some(ref lb) = r.length {
                    buf.push('*');
                    if let Some(min) = lb.min {
                        buf.push_str(&min.to_string());
                    }
                    buf.push_str("..");
                    if let Some(max) = lb.max {
                        buf.push_str(&max.to_string());
                    }
                }
                if !r.properties.is_empty() {
                    buf.push_str(" {");
                    write_prop_map(buf, &r.properties);
                    buf.push('}');
                }
                match r.direction {
                    Direction::Incoming => buf.push_str("]-"),
                    Direction::Outgoing => buf.push_str("]->"),
                    Direction::Both => buf.push_str("]-"),
                }
            }
        }
    }
}

fn write_prop_map(buf: &mut String, props: &[(String, Expr)]) {
    for (i, (key, val)) in props.iter().enumerate() {
        if i > 0 {
            buf.push_str(", ");
        }
        buf.push_str(key);
        buf.push_str(": ");
        write_expr(buf, val);
    }
}

fn write_return_items(buf: &mut String, items: &[ReturnItem]) {
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            buf.push_str(", ");
        }
        write_expr(buf, &item.expr);
        if let Some(ref alias) = item.alias {
            buf.push_str(" AS ");
            buf.push_str(alias);
        }
    }
}

fn write_set_items(buf: &mut String, items: &[SetItem]) {
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            buf.push_str(", ");
        }
        match item {
            SetItem::Property {
                variable,
                property,
                expr,
            } => {
                buf.push_str(variable);
                buf.push('.');
                buf.push_str(property);
                buf.push_str(" = ");
                write_expr(buf, expr);
            }
            SetItem::PropertyPath {
                variable,
                path,
                expr,
            } => {
                buf.push_str(variable);
                for seg in path {
                    buf.push('.');
                    buf.push_str(seg);
                }
                buf.push_str(" = ");
                write_expr(buf, expr);
            }
            SetItem::DocFunction {
                function,
                variable,
                path,
                value_expr,
            } => {
                buf.push_str(function);
                buf.push('(');
                buf.push_str(variable);
                for seg in path {
                    buf.push('.');
                    buf.push_str(seg);
                }
                buf.push_str(", ");
                write_expr(buf, value_expr);
                buf.push(')');
            }
            SetItem::AddLabel { variable, label } => {
                buf.push_str(variable);
                buf.push(':');
                buf.push_str(label);
            }
            SetItem::ReplaceProperties { variable, expr } => {
                buf.push_str(variable);
                buf.push_str(" = ");
                write_expr(buf, expr);
            }
            SetItem::MergeProperties { variable, expr } => {
                buf.push_str(variable);
                buf.push_str(" += ");
                write_expr(buf, expr);
            }
        }
    }
}

/// Write an expression to the canonical string.
///
/// Literals are replaced with `$` — this is the core normalization step.
/// All other expression types preserve their structure.
fn write_expr(buf: &mut String, expr: &Expr) {
    match expr {
        // Core normalization: all literals become `$`
        Expr::Literal(_) => buf.push('$'),

        Expr::Parameter(name) => {
            buf.push('$');
            buf.push_str(name);
        }
        Expr::Variable(name) => buf.push_str(name),
        Expr::Star => buf.push('*'),

        Expr::PropertyAccess { expr, property } => {
            write_expr(buf, expr);
            buf.push('.');
            buf.push_str(property);
        }
        Expr::BinaryOp { left, op, right } => {
            buf.push('(');
            write_expr(buf, left);
            buf.push(' ');
            buf.push_str(binary_op_str(op));
            buf.push(' ');
            write_expr(buf, right);
            buf.push(')');
        }
        Expr::UnaryOp { op, expr } => {
            buf.push_str(unary_op_str(op));
            buf.push(' ');
            write_expr(buf, expr);
        }
        Expr::FunctionCall {
            name,
            args,
            distinct,
        } => {
            buf.push_str(name);
            buf.push('(');
            if *distinct {
                buf.push_str("DISTINCT ");
            }
            for (i, arg) in args.iter().enumerate() {
                if i > 0 {
                    buf.push_str(", ");
                }
                write_expr(buf, arg);
            }
            buf.push(')');
        }
        Expr::List(items) => {
            buf.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    buf.push_str(", ");
                }
                write_expr(buf, item);
            }
            buf.push(']');
        }
        Expr::MapLiteral(entries) => {
            buf.push('{');
            for (i, (key, val)) in entries.iter().enumerate() {
                if i > 0 {
                    buf.push_str(", ");
                }
                buf.push_str(key);
                buf.push_str(": ");
                write_expr(buf, val);
            }
            buf.push('}');
        }
        Expr::In { expr, list } => {
            write_expr(buf, expr);
            buf.push_str(" IN ");
            write_expr(buf, list);
        }
        Expr::IsNull { expr, negated } => {
            write_expr(buf, expr);
            if *negated {
                buf.push_str(" IS NOT NULL");
            } else {
                buf.push_str(" IS NULL");
            }
        }
        Expr::StringMatch { expr, op, pattern } => {
            write_expr(buf, expr);
            buf.push(' ');
            buf.push_str(string_op_str(op));
            buf.push(' ');
            write_expr(buf, pattern);
        }
        Expr::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            buf.push_str("CASE");
            if let Some(ref op) = operand {
                buf.push(' ');
                write_expr(buf, op);
            }
            for (when, then) in when_clauses {
                buf.push_str(" WHEN ");
                write_expr(buf, when);
                buf.push_str(" THEN ");
                write_expr(buf, then);
            }
            if let Some(ref else_expr) = else_clause {
                buf.push_str(" ELSE ");
                write_expr(buf, else_expr);
            }
            buf.push_str(" END");
        }
        Expr::MapProjection { expr, items } => {
            write_expr(buf, expr);
            buf.push('{');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    buf.push(',');
                }
                match item {
                    MapProjectionItem::Property(p) => {
                        buf.push('.');
                        buf.push_str(p);
                    }
                    MapProjectionItem::Computed(alias, value) => {
                        buf.push_str(alias);
                        buf.push(':');
                        write_expr(buf, value);
                    }
                }
            }
            buf.push('}');
        }
        Expr::PatternPredicate(_) => {
            buf.push_str("PATTERN_PRED");
        }
    }
}

fn binary_op_str(op: &BinaryOperator) -> &'static str {
    match op {
        BinaryOperator::Add => "+",
        BinaryOperator::Sub => "-",
        BinaryOperator::Mul => "*",
        BinaryOperator::Div => "/",
        BinaryOperator::Modulo => "%",
        BinaryOperator::Eq => "=",
        BinaryOperator::Neq => "<>",
        BinaryOperator::Lt => "<",
        BinaryOperator::Lte => "<=",
        BinaryOperator::Gt => ">",
        BinaryOperator::Gte => ">=",
        BinaryOperator::And => "AND",
        BinaryOperator::Or => "OR",
        BinaryOperator::Xor => "XOR",
    }
}

fn unary_op_str(op: &UnaryOperator) -> &'static str {
    match op {
        UnaryOperator::Not => "NOT",
        UnaryOperator::Neg => "-",
    }
}

fn string_op_str(op: &StringOp) -> &'static str {
    match op {
        StringOp::StartsWith => "STARTS WITH",
        StringOp::EndsWith => "ENDS WITH",
        StringOp::Contains => "CONTAINS",
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::cypher::parse;

    /// Normalization replaces literal values with `$` placeholder.
    #[test]
    fn normalize_strips_literals() {
        let q1 = parse("MATCH (n:User {id: 42}) RETURN n.name").expect("parse q1");
        let q2 = parse("MATCH (n:User {id: 99}) RETURN n.name").expect("parse q2");

        let n1 = normalize(&q1);
        let n2 = normalize(&q2);

        assert_eq!(
            n1, n2,
            "queries differing only by literal should normalize equally"
        );
        assert!(
            n1.contains('$'),
            "normalized form should contain $ placeholder"
        );
        assert!(
            !n1.contains("42"),
            "normalized form should not contain original literal"
        );
    }

    /// Different query structures produce different fingerprints.
    #[test]
    fn different_structure_different_fingerprint() {
        let q1 = parse("MATCH (n:User) RETURN n").expect("parse q1");
        let q2 = parse("MATCH (n:User) RETURN n.name").expect("parse q2");

        let (_, fp1) = normalize_and_fingerprint(&q1);
        let (_, fp2) = normalize_and_fingerprint(&q2);

        assert_ne!(
            fp1, fp2,
            "different structures should have different fingerprints"
        );
    }

    /// Same structure with different literals produces same fingerprint.
    #[test]
    fn same_structure_same_fingerprint() {
        let q1 = parse("MATCH (n:User {id: 1}) RETURN n").expect("parse q1");
        let q2 = parse("MATCH (n:User {id: 999}) RETURN n").expect("parse q2");

        let (_, fp1) = normalize_and_fingerprint(&q1);
        let (_, fp2) = normalize_and_fingerprint(&q2);

        assert_eq!(
            fp1, fp2,
            "same structure with different literals = same fingerprint"
        );
    }

    /// Fingerprint is deterministic across calls.
    #[test]
    fn fingerprint_is_deterministic() {
        let q = parse("MATCH (n:User) WHERE n.age > 25 RETURN n").expect("parse");
        let (_, fp1) = normalize_and_fingerprint(&q);
        let (_, fp2) = normalize_and_fingerprint(&q);
        assert_eq!(fp1, fp2);
    }

    /// Parameters preserve their names (they're already normalized).
    #[test]
    fn parameters_preserved() {
        let q = parse("MATCH (n:User {id: $uid}) RETURN n").expect("parse");
        let canonical = normalize(&q);
        assert!(
            canonical.contains("$uid"),
            "parameter names should be preserved"
        );
    }

    /// Normalization preserves relationship types and labels.
    #[test]
    fn labels_and_types_preserved() {
        let q = parse("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b").expect("parse");
        let canonical = normalize(&q);
        assert!(canonical.contains("Person"), "labels preserved");
        assert!(canonical.contains("KNOWS"), "relationship types preserved");
    }

    /// Variable-length paths are preserved in canonical form.
    #[test]
    fn variable_length_preserved() {
        let q = parse("MATCH (a)-[:KNOWS*2..5]->(b) RETURN b").expect("parse");
        let canonical = normalize(&q);
        assert!(
            canonical.contains("*2..5"),
            "variable-length bounds preserved"
        );
    }

    /// Write operations normalize correctly.
    #[test]
    fn create_normalizes() {
        let q1 = parse("CREATE (n:User {name: 'Alice'})").expect("parse q1");
        let q2 = parse("CREATE (n:User {name: 'Bob'})").expect("parse q2");

        let (_, fp1) = normalize_and_fingerprint(&q1);
        let (_, fp2) = normalize_and_fingerprint(&q2);
        assert_eq!(
            fp1, fp2,
            "CREATE with different literals = same fingerprint"
        );
    }

    /// OPTIONAL MATCH is distinguished from MATCH.
    #[test]
    fn optional_match_distinct() {
        let q1 = parse("MATCH (n:User) RETURN n").expect("parse q1");
        let q2 =
            parse("MATCH (n:User) OPTIONAL MATCH (n)-[:KNOWS]->(m) RETURN n, m").expect("parse q2");

        let (_, fp1) = normalize_and_fingerprint(&q1);
        let (_, fp2) = normalize_and_fingerprint(&q2);
        assert_ne!(fp1, fp2);
    }
}
