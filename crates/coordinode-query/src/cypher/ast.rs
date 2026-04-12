//! Typed AST nodes for OpenCypher read and write operations.
//!
//! The AST is produced by the parser from a Cypher query string.
//! It represents the syntactic structure of the query before semantic
//! analysis (variable binding, label validation, type checking).

use std::collections::HashMap;

use coordinode_core::graph::types::Value;

/// A complete Cypher query composed of one or more clauses.
#[derive(Debug, Clone, PartialEq)]
pub struct Query {
    pub clauses: Vec<Clause>,
    /// Per-query optimizer hints extracted from `/*+ key('value') */` comments.
    pub hints: Vec<QueryHint>,
}

/// Per-query optimizer hint from `/*+ key('value') */` syntax.
#[derive(Debug, Clone, PartialEq)]
pub enum QueryHint {
    /// Override vector consistency mode for this query only.
    /// Syntax: `/*+ vector_consistency('snapshot') */`
    VectorConsistency(coordinode_core::graph::types::VectorConsistencyMode),
}

/// A single clause in a Cypher query.
#[derive(Debug, Clone, PartialEq)]
pub enum Clause {
    // Read clauses
    Match(MatchClause),
    OptionalMatch(MatchClause),
    Where(Expr),
    Return(ReturnClause),
    With(WithClause),
    Unwind(UnwindClause),
    OrderBy(Vec<SortItem>),
    Skip(Expr),
    Limit(Expr),
    AsOfTimestamp(Expr),

    // Procedure call
    Call(CallClause),

    // DDL clauses
    AlterLabel(AlterLabelClause),
    CreateTextIndex(CreateTextIndexClause),
    DropTextIndex(DropTextIndexClause),
    CreateEncryptedIndex(CreateEncryptedIndexClause),
    DropEncryptedIndex(DropEncryptedIndexClause),
    CreateIndex(CreateIndexClause),
    DropIndex(DropIndexClause),

    // Write clauses
    Create(CreateClause),
    Merge(MergeClause),
    /// MERGE ALL: Cartesian-product relationship merge — all matching src × tgt pairs.
    MergeMany(MergeClause),
    Upsert(UpsertClause),
    Delete(DeleteClause),
    Set(Vec<SetItem>),
    Remove(Vec<RemoveItem>),
}

/// ALTER LABEL clause: change schema mode for a label.
///
/// Syntax: `ALTER LABEL User SET SCHEMA STRICT`
#[derive(Debug, Clone, PartialEq)]
pub struct AlterLabelClause {
    /// Label name to alter.
    pub label: String,
    /// New schema mode.
    pub mode: String,
}

/// Per-field configuration for text index DDL.
#[derive(Debug, Clone, PartialEq)]
pub struct TextIndexFieldSpec {
    /// Property name to index.
    pub property: String,
    /// Analyzer name: language ("english", "russian"), "auto_detect", "none".
    pub analyzer: Option<String>,
}

/// CREATE TEXT INDEX clause.
///
/// Simple syntax:   `CREATE TEXT INDEX idx ON :Label(property) LANGUAGE "english"`
/// Extended syntax: `CREATE TEXT INDEX idx ON :Label { title: { analyzer: "english" }, body: { analyzer: "auto_detect" } } DEFAULT LANGUAGE "english" LANGUAGE OVERRIDE "lang"`
#[derive(Debug, Clone, PartialEq)]
pub struct CreateTextIndexClause {
    /// Index name.
    pub name: String,
    /// Label to index.
    pub label: String,
    /// Fields to index with per-field configuration.
    pub fields: Vec<TextIndexFieldSpec>,
    /// Default language for fields without explicit analyzer (DEFAULT LANGUAGE clause).
    /// In simple syntax, the LANGUAGE value is used here.
    pub default_language: Option<String>,
    /// Node property name for per-node language override (LANGUAGE OVERRIDE clause).
    pub language_override: Option<String>,
}

/// DROP TEXT INDEX clause.
///
/// Syntax: `DROP TEXT INDEX idx_name`
#[derive(Debug, Clone, PartialEq)]
pub struct DropTextIndexClause {
    /// Index name to drop.
    pub name: String,
}

/// CREATE ENCRYPTED INDEX clause (SSE).
///
/// Syntax: `CREATE ENCRYPTED INDEX idx_name ON :Label(property)`
///
/// Creates a searchable symmetric encryption index for equality queries
/// on encrypted fields. The client provides search tokens (HMAC-SHA256);
/// the server matches tokens without seeing plaintext.
#[derive(Debug, Clone, PartialEq)]
pub struct CreateEncryptedIndexClause {
    /// Index name.
    pub name: String,
    /// Label to index.
    pub label: String,
    /// Property name to index.
    pub property: String,
}

/// DROP ENCRYPTED INDEX clause.
///
/// Syntax: `DROP ENCRYPTED INDEX idx_name`
#[derive(Debug, Clone, PartialEq)]
pub struct DropEncryptedIndexClause {
    /// Index name to drop.
    pub name: String,
}

/// CREATE [UNIQUE] [SPARSE] INDEX clause (B-tree single-field index).
///
/// Syntax: `CREATE [UNIQUE] [SPARSE] INDEX idx_name ON :Label(prop) [WHERE pred]`
///
/// The optional WHERE clause restricts index membership to nodes matching
/// a simple predicate (partial index). Supported predicates:
/// - `prop = 'string'`
/// - `prop = 42`
/// - `prop = true/false`
/// - `prop IS NOT NULL`
#[derive(Debug, Clone, PartialEq)]
pub struct CreateIndexClause {
    /// Index name.
    pub name: String,
    /// Label to index.
    pub label: String,
    /// Property to index.
    pub property: String,
    /// Whether this index enforces uniqueness.
    pub unique: bool,
    /// Whether to skip null values (sparse index).
    pub sparse: bool,
    /// Optional WHERE predicate expression for partial index.
    pub filter_expr: Option<Expr>,
}

/// DROP INDEX clause.
///
/// Syntax: `DROP INDEX idx_name`
#[derive(Debug, Clone, PartialEq)]
pub struct DropIndexClause {
    /// Index name to drop.
    pub name: String,
}

/// MATCH clause: one or more patterns.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchClause {
    pub patterns: Vec<Pattern>,
    /// Optional WHERE attached directly to MATCH.
    pub where_clause: Option<Expr>,
}

/// A graph pattern: sequence of node and relationship elements.
///
/// Examples:
///   `(n:User)`
///   `(a)-[:KNOWS]->(b)`
///   `(a)-[:KNOWS*2..5]->(b)`
#[derive(Debug, Clone, PartialEq)]
pub struct Pattern {
    pub elements: Vec<PatternElement>,
}

/// An element in a graph pattern.
#[derive(Debug, Clone, PartialEq)]
pub enum PatternElement {
    Node(NodePattern),
    Relationship(RelationshipPattern),
}

/// A node pattern: `(variable:Label {prop: value})`.
#[derive(Debug, Clone, PartialEq)]
pub struct NodePattern {
    /// Variable name (optional).
    pub variable: Option<String>,
    /// Labels (zero or more).
    pub labels: Vec<String>,
    /// Inline property map.
    pub properties: Vec<(String, Expr)>,
}

/// A relationship pattern: `-[variable:TYPE {props}]->`.
#[derive(Debug, Clone, PartialEq)]
pub struct RelationshipPattern {
    /// Variable name (optional).
    pub variable: Option<String>,
    /// Relationship types (disjunction — any of these types).
    pub rel_types: Vec<String>,
    /// Direction of the relationship.
    pub direction: Direction,
    /// Variable-length traversal bounds (e.g., *2..5).
    pub length: Option<LengthBound>,
    /// Inline property map.
    pub properties: Vec<(String, Expr)>,
}

/// Relationship direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// `-->` or `-[]->`
    Outgoing,
    /// `<--` or `<-[]-`
    Incoming,
    /// `--` or `-[]-`
    Both,
}

/// Variable-length path bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LengthBound {
    /// Minimum hops (default 1).
    pub min: Option<u64>,
    /// Maximum hops (default unbounded, but capped at 10 by engine).
    pub max: Option<u64>,
}

/// RETURN clause.
#[derive(Debug, Clone, PartialEq)]
pub struct ReturnClause {
    pub distinct: bool,
    pub items: Vec<ReturnItem>,
}

/// A single RETURN item: expression with optional alias.
#[derive(Debug, Clone, PartialEq)]
pub struct ReturnItem {
    pub expr: Expr,
    pub alias: Option<String>,
}

/// WITH clause: projection between query parts.
#[derive(Debug, Clone, PartialEq)]
pub struct WithClause {
    pub distinct: bool,
    pub items: Vec<ReturnItem>,
    pub where_clause: Option<Expr>,
}

/// UNWIND clause: `UNWIND expr AS variable`.
#[derive(Debug, Clone, PartialEq)]
pub struct UnwindClause {
    pub expr: Expr,
    pub variable: String,
}

/// ORDER BY item: expression + direction.
#[derive(Debug, Clone, PartialEq)]
pub struct SortItem {
    pub expr: Expr,
    pub ascending: bool,
}

/// Expression tree.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Literal value.
    Literal(Value),

    /// Parameter: `$name`.
    Parameter(String),

    /// Variable reference: `n`.
    Variable(String),

    /// Property access: `n.name`.
    PropertyAccess { expr: Box<Expr>, property: String },

    /// Binary operation: `a + b`, `a AND b`, etc.
    BinaryOp {
        left: Box<Expr>,
        op: BinaryOperator,
        right: Box<Expr>,
    },

    /// Unary operation: `NOT x`, `-x`.
    UnaryOp { op: UnaryOperator, expr: Box<Expr> },

    /// Function call: `count(*)`, `vector_distance(a, b)`.
    FunctionCall {
        name: String,
        args: Vec<Expr>,
        distinct: bool,
    },

    /// List literal: `[1, 2, 3]`.
    List(Vec<Expr>),

    /// Map literal: `{key: value}`.
    MapLiteral(Vec<(String, Expr)>),

    /// Map projection: `n { .name, .age, posts: collect(p { .title }) }`.
    /// Builds a nested JSON object from a variable's properties.
    /// `.prop` shorthand expands to `prop: n.prop`.
    MapProjection {
        expr: Box<Expr>,
        items: Vec<MapProjectionItem>,
    },

    /// `x IN list` or `x IN [1,2,3]`.
    In { expr: Box<Expr>, list: Box<Expr> },

    /// `x IS NULL` / `x IS NOT NULL`.
    IsNull { expr: Box<Expr>, negated: bool },

    /// `x STARTS WITH y`, `x ENDS WITH y`, `x CONTAINS y`.
    StringMatch {
        expr: Box<Expr>,
        op: StringOp,
        pattern: Box<Expr>,
    },

    /// `CASE WHEN ... THEN ... ELSE ... END`.
    Case {
        operand: Option<Box<Expr>>,
        when_clauses: Vec<(Expr, Expr)>,
        else_clause: Option<Box<Expr>>,
    },

    /// Pattern predicate: `(a)-[:R]->(b)` in expression context.
    /// Evaluates to `true` if the pattern matches at least one path, `false` otherwise.
    /// Used in WHERE clauses: `WHERE (a)-[:KNOWS]->(b)` or `WHERE NOT (a)-[:KNOWS]->(b)`.
    PatternPredicate(Pattern),

    /// Star expression for `count(*)` or `RETURN *`.
    Star,
}

impl Expr {
    /// Replace all `Parameter(name)` nodes with `Literal(value)` from the params map.
    /// Unknown parameters (not in the map) remain as `Parameter` (will eval to Null).
    pub fn substitute_params(&mut self, params: &HashMap<String, Value>) {
        match self {
            Expr::Parameter(name) => {
                if let Some(value) = params.get(name.as_str()) {
                    *self = Expr::Literal(value.clone());
                }
            }
            Expr::PropertyAccess { expr, .. } => expr.substitute_params(params),
            Expr::BinaryOp { left, right, .. } => {
                left.substitute_params(params);
                right.substitute_params(params);
            }
            Expr::UnaryOp { expr, .. } => expr.substitute_params(params),
            Expr::FunctionCall { args, .. } => {
                for arg in args {
                    arg.substitute_params(params);
                }
            }
            Expr::List(items) => {
                for item in items {
                    item.substitute_params(params);
                }
            }
            Expr::MapLiteral(entries) => {
                for (_, v) in entries {
                    v.substitute_params(params);
                }
            }
            Expr::MapProjection { expr, items } => {
                expr.substitute_params(params);
                for item in items {
                    if let MapProjectionItem::Computed(_, ref mut value) = item {
                        value.substitute_params(params);
                    }
                }
            }
            Expr::In { expr, list } => {
                expr.substitute_params(params);
                list.substitute_params(params);
            }
            Expr::IsNull { expr, .. } => expr.substitute_params(params),
            Expr::StringMatch { expr, pattern, .. } => {
                expr.substitute_params(params);
                pattern.substitute_params(params);
            }
            Expr::Case {
                operand,
                when_clauses,
                else_clause,
            } => {
                if let Some(op) = operand {
                    op.substitute_params(params);
                }
                for (cond, result) in when_clauses {
                    cond.substitute_params(params);
                    result.substitute_params(params);
                }
                if let Some(el) = else_clause {
                    el.substitute_params(params);
                }
            }
            Expr::PatternPredicate(pattern) => {
                for elem in &mut pattern.elements {
                    if let PatternElement::Node(node) = elem {
                        for (_, v) in &mut node.properties {
                            v.substitute_params(params);
                        }
                    } else if let PatternElement::Relationship(rel) = elem {
                        for (_, v) in &mut rel.properties {
                            v.substitute_params(params);
                        }
                    }
                }
            }
            Expr::Literal(_) | Expr::Variable(_) | Expr::Star => {}
        }
    }
}

/// An item in a map projection expression.
#[derive(Debug, Clone, PartialEq)]
pub enum MapProjectionItem {
    /// Property shorthand: `.name` → includes `name: expr.name`.
    Property(String),
    /// Computed/aliased entry: `alias: expression`.
    Computed(String, Expr),
}

// --- Procedure call type ---

/// CALL clause: invoke a named procedure.
///
/// Example: `CALL db.advisor.suggestions() YIELD id, severity, kind`
#[derive(Debug, Clone, PartialEq)]
pub struct CallClause {
    /// Dotted procedure name (e.g., `db.advisor.suggestions`).
    pub procedure: String,
    /// Positional arguments.
    pub args: Vec<Expr>,
    /// YIELD items: columns to select from the procedure output.
    /// If empty, all columns are returned.
    pub yield_items: Vec<YieldItem>,
}

/// A single item in a YIELD clause.
#[derive(Debug, Clone, PartialEq)]
pub struct YieldItem {
    /// Column name from the procedure output.
    pub name: String,
    /// Optional alias (AS ...).
    pub alias: Option<String>,
}

// --- Write clause types ---

/// CREATE clause: create one or more patterns.
///
/// Example: `CREATE (n:User {name: "Alice"})-[:KNOWS]->(m:User {name: "Bob"})`
#[derive(Debug, Clone, PartialEq)]
pub struct CreateClause {
    pub patterns: Vec<Pattern>,
}

/// MERGE clause: match or create a pattern, with optional actions.
///
/// Example: `MERGE (n:User {email: $email}) ON CREATE SET n.created = now()`
#[derive(Debug, Clone, PartialEq)]
pub struct MergeClause {
    pub pattern: Pattern,
    pub on_match: Vec<SetItem>,
    pub on_create: Vec<SetItem>,
}

/// UPSERT MATCH clause (CoordiNode extension): atomic upsert.
///
/// Example:
/// ```cypher
/// UPSERT MATCH (u:User {email: "alice@example.com"})
/// ON MATCH SET u.login_count = u.login_count + 1
/// ON CREATE CREATE (u:User {email: "alice@example.com", login_count: 1})
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct UpsertClause {
    pub pattern: Pattern,
    pub on_match: Vec<SetItem>,
    pub on_create: Vec<Pattern>,
}

/// DELETE clause: delete nodes or relationships.
///
/// Example: `DELETE n` or `DETACH DELETE n` (also deletes connected edges).
#[derive(Debug, Clone, PartialEq)]
pub struct DeleteClause {
    pub detach: bool,
    pub exprs: Vec<Expr>,
}

/// A single SET operation.
#[derive(Debug, Clone, PartialEq)]
pub enum SetItem {
    /// `n.prop = expr` — set a single property.
    Property {
        variable: String,
        property: String,
        expr: Expr,
    },
    /// `n.a.b.c = expr` — set a nested DOCUMENT path via merge operand.
    PropertyPath {
        variable: String,
        path: Vec<String>,
        expr: Expr,
    },
    /// `doc_push(n.tags, "new")` — document array/numeric mutation via merge operand.
    DocFunction {
        /// Function name: doc_push, doc_pull, doc_add_to_set, doc_inc.
        function: String,
        /// Variable name (e.g., "n").
        variable: String,
        /// Property path (e.g., ["tags"] or ["stats", "views"]).
        path: Vec<String>,
        /// Value expression to push/pull/add/increment.
        value_expr: Expr,
    },
    /// `n = {map}` — replace all properties with map.
    ReplaceProperties { variable: String, expr: Expr },
    /// `n += {map}` — merge properties from map.
    MergeProperties { variable: String, expr: Expr },
    /// `n:Label` — add a label.
    AddLabel { variable: String, label: String },
}

/// A single REMOVE operation.
#[derive(Debug, Clone, PartialEq)]
pub enum RemoveItem {
    /// `n.prop` — remove a property.
    Property { variable: String, property: String },
    /// `n.a.b.c` — remove a nested DOCUMENT path via merge operand.
    PropertyPath { variable: String, path: Vec<String> },
    /// `n:Label` — remove a label.
    Label { variable: String, label: String },
}

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOperator {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Modulo,

    // Comparison
    Eq,
    Neq,
    Lt,
    Lte,
    Gt,
    Gte,

    // Logical
    And,
    Or,
    Xor,
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOperator {
    Not,
    Neg,
}

/// String matching operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringOp {
    StartsWith,
    EndsWith,
    Contains,
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn substitute_simple_parameter() {
        let mut expr = Expr::Parameter("name".to_string());
        let mut params = HashMap::new();
        params.insert("name".to_string(), Value::String("Alice".to_string()));

        expr.substitute_params(&params);
        assert_eq!(expr, Expr::Literal(Value::String("Alice".to_string())));
    }

    #[test]
    fn substitute_unknown_parameter_unchanged() {
        let mut expr = Expr::Parameter("missing".to_string());
        expr.substitute_params(&HashMap::new());
        assert_eq!(expr, Expr::Parameter("missing".to_string()));
    }

    #[test]
    fn substitute_in_binary_op() {
        let mut expr = Expr::BinaryOp {
            left: Box::new(Expr::Variable("n.age".to_string())),
            op: BinaryOperator::Gt,
            right: Box::new(Expr::Parameter("min_age".to_string())),
        };
        let mut params = HashMap::new();
        params.insert("min_age".to_string(), Value::Int(18));

        expr.substitute_params(&params);

        if let Expr::BinaryOp { right, .. } = &expr {
            assert_eq!(**right, Expr::Literal(Value::Int(18)));
        } else {
            panic!("expected BinaryOp");
        }
    }

    #[test]
    fn substitute_in_function_args() {
        let mut expr = Expr::FunctionCall {
            name: "vector_distance".to_string(),
            args: vec![
                Expr::Variable("n.embedding".to_string()),
                Expr::Parameter("query_vec".to_string()),
            ],
            distinct: false,
        };
        let mut params = HashMap::new();
        params.insert("query_vec".to_string(), Value::Vector(vec![1.0, 0.0]));

        expr.substitute_params(&params);

        if let Expr::FunctionCall { args, .. } = &expr {
            assert_eq!(args[1], Expr::Literal(Value::Vector(vec![1.0, 0.0])));
        } else {
            panic!("expected FunctionCall");
        }
    }

    #[test]
    fn substitute_in_list() {
        let mut expr = Expr::List(vec![
            Expr::Literal(Value::Int(1)),
            Expr::Parameter("val".to_string()),
        ]);
        let mut params = HashMap::new();
        params.insert("val".to_string(), Value::Int(2));

        expr.substitute_params(&params);

        if let Expr::List(items) = &expr {
            assert_eq!(items[1], Expr::Literal(Value::Int(2)));
        } else {
            panic!("expected List");
        }
    }

    #[test]
    fn substitute_nested_deep() {
        // NOT($param > 10)
        let mut expr = Expr::UnaryOp {
            op: UnaryOperator::Not,
            expr: Box::new(Expr::BinaryOp {
                left: Box::new(Expr::Parameter("x".to_string())),
                op: BinaryOperator::Gt,
                right: Box::new(Expr::Literal(Value::Int(10))),
            }),
        };
        let mut params = HashMap::new();
        params.insert("x".to_string(), Value::Int(5));

        expr.substitute_params(&params);

        if let Expr::UnaryOp {
            expr: inner_box, ..
        } = &expr
        {
            if let Expr::BinaryOp { left, .. } = inner_box.as_ref() {
                assert_eq!(**left, Expr::Literal(Value::Int(5)));
            } else {
                panic!("expected BinaryOp inside UnaryOp");
            }
        } else {
            panic!("expected UnaryOp");
        }
    }

    #[test]
    fn substitute_leaves_literals_and_variables_untouched() {
        let mut expr = Expr::Literal(Value::Int(42));
        expr.substitute_params(&HashMap::from([("x".to_string(), Value::Int(99))]));
        assert_eq!(expr, Expr::Literal(Value::Int(42)));

        let mut expr = Expr::Variable("n".to_string());
        expr.substitute_params(&HashMap::from([("n".to_string(), Value::Int(99))]));
        assert_eq!(expr, Expr::Variable("n".to_string()));
    }
}
