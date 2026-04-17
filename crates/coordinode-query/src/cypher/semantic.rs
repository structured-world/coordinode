//! Semantic analysis for Cypher AST.
//!
//! Validates a parsed Cypher query for:
//! - Variable binding: ensures all referenced variables are defined
//! - Scope chain: WITH creates scope barriers, upstream vars not visible
//! - Label validation: checks labels exist in schema (when schema provided)
//! - Property validation: checks properties exist on labels (when schema provided)

use std::collections::HashMap;

use coordinode_core::schema::definition::{EdgeTypeSchema, LabelSchema};

use super::ast::*;

/// Semantic analysis error.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SemanticError {
    /// Variable referenced but not defined in current scope.
    #[error("undefined variable `{name}`")]
    UndefinedVariable { name: String },

    /// Variable defined more than once in the same clause.
    #[error("variable `{name}` already defined")]
    DuplicateVariable { name: String },

    /// Label referenced in pattern but not found in schema.
    #[error("unknown label `{name}`")]
    UnknownLabel { name: String },

    /// Edge type referenced in pattern but not found in schema.
    #[error("unknown edge type `{name}`")]
    UnknownEdgeType { name: String },

    /// Property accessed on a label that does not have it.
    #[error("label `{label}` has no property `{property}`")]
    UnknownProperty { label: String, property: String },

    /// Aggregation function used outside RETURN/WITH.
    #[error("aggregation function `{name}` not allowed here")]
    InvalidAggregation { name: String },

    /// RETURN * used but no variables in scope.
    #[error("RETURN * with no variables in scope")]
    ReturnStarEmpty,

    /// DELETE on non-variable expression.
    #[error("DELETE requires a variable, got expression")]
    DeleteNonVariable,
}

/// Provides schema information for semantic validation.
///
/// Schema validation is optional — if no provider is given, label and
/// property checks are skipped (schemaless mode).
pub trait SchemaProvider {
    /// Look up a node label schema by name.
    fn get_label(&self, name: &str) -> Option<&LabelSchema>;

    /// Look up an edge type schema by name.
    fn get_edge_type(&self, name: &str) -> Option<&EdgeTypeSchema>;
}

/// In-memory schema provider for testing and embedded use.
pub struct MapSchemaProvider {
    labels: HashMap<String, LabelSchema>,
    edge_types: HashMap<String, EdgeTypeSchema>,
}

impl Default for MapSchemaProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl MapSchemaProvider {
    pub fn new() -> Self {
        Self {
            labels: HashMap::new(),
            edge_types: HashMap::new(),
        }
    }

    pub fn add_label(&mut self, schema: LabelSchema) {
        self.labels.insert(schema.name.clone(), schema);
    }

    pub fn add_edge_type(&mut self, schema: EdgeTypeSchema) {
        self.edge_types.insert(schema.name.clone(), schema);
    }
}

impl SchemaProvider for MapSchemaProvider {
    fn get_label(&self, name: &str) -> Option<&LabelSchema> {
        self.labels.get(name)
    }

    fn get_edge_type(&self, name: &str) -> Option<&EdgeTypeSchema> {
        self.edge_types.get(name)
    }
}

/// Analyze a parsed Cypher query for semantic correctness.
///
/// Returns a list of semantic errors. An empty list means the query is valid.
/// Schema validation is optional — pass `None` for schemaless mode.
pub fn analyze(query: &Query, schema: Option<&dyn SchemaProvider>) -> Vec<SemanticError> {
    let mut analyzer = Analyzer::new(schema);
    analyzer.analyze_query(query);
    analyzer.errors
}

struct Analyzer<'a> {
    /// Current variable scope: name → labels (if known).
    scope: HashMap<String, Vec<String>>,
    /// Schema provider (optional).
    schema: Option<&'a dyn SchemaProvider>,
    /// Collected errors.
    errors: Vec<SemanticError>,
}

impl<'a> Analyzer<'a> {
    fn new(schema: Option<&'a dyn SchemaProvider>) -> Self {
        Self {
            scope: HashMap::new(),
            schema,
            errors: Vec::new(),
        }
    }

    fn analyze_query(&mut self, query: &Query) {
        for clause in &query.clauses {
            self.analyze_clause(clause);
        }
    }

    fn analyze_clause(&mut self, clause: &Clause) {
        match clause {
            Clause::Match(mc) | Clause::OptionalMatch(mc) => {
                self.analyze_match(mc);
            }
            Clause::Where(expr) => {
                self.check_expr(expr);
            }
            Clause::Return(rc) => {
                self.analyze_return(rc);
            }
            Clause::With(wc) => {
                self.analyze_with(wc);
            }
            Clause::Unwind(uc) => {
                self.analyze_unwind(uc);
            }
            Clause::OrderBy(items) => {
                for item in items {
                    self.check_expr(&item.expr);
                }
            }
            Clause::Skip(expr) | Clause::Limit(expr) | Clause::AsOfTimestamp(expr) => {
                self.check_expr(expr);
            }
            Clause::Create(cc) => {
                self.analyze_create(cc);
            }
            Clause::Merge(mc) => {
                self.analyze_merge(mc);
            }
            Clause::MergeMany(mc) => {
                self.analyze_merge(mc);
            }
            Clause::Upsert(uc) => {
                self.analyze_upsert(uc);
            }
            Clause::Delete(dc) => {
                self.analyze_delete(dc);
            }
            Clause::DetachDocument(dd) => {
                if !self.scope.contains_key(&dd.source_variable) {
                    self.errors.push(SemanticError::UndefinedVariable {
                        name: dd.source_variable.clone(),
                    });
                }
                // Bind the new target variable for subsequent clauses.
                self.scope
                    .insert(dd.target_variable.clone(), dd.target_labels.clone());
                if let Some(ref t) = dd.transfer {
                    self.check_expr(&t.predicate);
                }
            }
            Clause::AttachDocument(ad) => {
                // ATTACH introduces its own pattern — bind both source and
                // target variables into scope for downstream clauses (mirrors
                // MATCH semantics).
                self.scope
                    .insert(ad.source_variable.clone(), ad.source_labels.clone());
                self.scope
                    .insert(ad.target_variable.clone(), ad.target_labels.clone());
                if let Some(ref t) = ad.transfer {
                    self.check_expr(&t.predicate);
                }
            }
            Clause::Set(items, _violation_mode) => {
                for item in items {
                    self.check_set_item(item);
                }
            }
            Clause::Remove(items) => {
                for item in items {
                    self.check_remove_item(item);
                }
            }
            Clause::Call(cc) => {
                // Validate procedure arguments
                for arg in &cc.args {
                    self.check_expr(arg);
                }
            }
            Clause::AlterLabel(_)
            | Clause::CreateTextIndex(_)
            | Clause::DropTextIndex(_)
            | Clause::CreateEncryptedIndex(_)
            | Clause::DropEncryptedIndex(_)
            | Clause::CreateIndex(_)
            | Clause::DropIndex(_)
            | Clause::CreateVectorIndex(_)
            | Clause::DropVectorIndex(_) => {
                // DDL — no variable references to validate.
            }
        }
    }

    fn analyze_match(&mut self, mc: &MatchClause) {
        for pattern in &mc.patterns {
            self.bind_pattern(pattern);
        }
        if let Some(ref expr) = mc.where_clause {
            self.check_expr(expr);
        }
    }

    fn analyze_create(&mut self, cc: &CreateClause) {
        for pattern in &cc.patterns {
            self.bind_pattern(pattern);
        }
    }

    fn analyze_merge(&mut self, mc: &MergeClause) {
        self.bind_pattern(&mc.pattern);
        for item in &mc.on_match {
            self.check_set_item(item);
        }
        for item in &mc.on_create {
            self.check_set_item(item);
        }
    }

    fn analyze_upsert(&mut self, uc: &UpsertClause) {
        self.bind_pattern(&uc.pattern);
        for item in &uc.on_match {
            self.check_set_item(item);
        }
        for pattern in &uc.on_create {
            self.bind_pattern(pattern);
        }
    }

    fn analyze_delete(&mut self, dc: &DeleteClause) {
        for expr in &dc.exprs {
            // DELETE should reference variables
            if let Expr::Variable(ref name) = expr {
                if !self.scope.contains_key(name) {
                    self.errors
                        .push(SemanticError::UndefinedVariable { name: name.clone() });
                }
            } else {
                self.errors.push(SemanticError::DeleteNonVariable);
            }
        }
    }

    fn analyze_return(&mut self, rc: &ReturnClause) {
        // Check for RETURN * with empty scope
        if rc.items.len() == 1 && rc.items[0].expr == Expr::Star && self.scope.is_empty() {
            self.errors.push(SemanticError::ReturnStarEmpty);
        }

        for item in &rc.items {
            if item.expr != Expr::Star {
                self.check_expr(&item.expr);
            }
        }

        // Add RETURN aliases to scope so ORDER BY can reference them.
        // Unlike WITH, RETURN does NOT create a scope barrier — existing
        // variables remain accessible alongside the new aliases.
        for item in &rc.items {
            if let Some(ref alias) = item.alias {
                self.scope.insert(alias.clone(), Vec::new());
            }
        }
    }

    fn analyze_with(&mut self, wc: &WithClause) {
        // First, check all expressions against current scope
        for item in &wc.items {
            if item.expr != Expr::Star {
                self.check_expr(&item.expr);
            }
        }

        // WITH creates a scope barrier: only projected variables survive
        let mut new_scope = HashMap::new();
        for item in &wc.items {
            if item.expr == Expr::Star {
                // Star projects everything — continue to also add explicit aliases
                new_scope.extend(self.scope.clone());
                continue;
            }

            let name = if let Some(ref alias) = item.alias {
                alias.clone()
            } else if let Expr::Variable(ref v) = item.expr {
                v.clone()
            } else {
                // Expression without alias — not accessible by name
                continue;
            };

            // Carry forward labels if projecting a variable
            let labels = if let Expr::Variable(ref v) = item.expr {
                self.scope.get(v).cloned().unwrap_or_default()
            } else {
                Vec::new()
            };

            new_scope.insert(name, labels);
        }

        self.scope = new_scope;

        // Check WHERE after scope change
        if let Some(ref expr) = wc.where_clause {
            self.check_expr(expr);
        }
    }

    fn analyze_unwind(&mut self, uc: &UnwindClause) {
        self.check_expr(&uc.expr);
        // UNWIND introduces a new variable
        self.scope.insert(uc.variable.clone(), Vec::new());
    }

    /// Bind variables from a graph pattern into the current scope.
    fn bind_pattern(&mut self, pattern: &Pattern) {
        for element in &pattern.elements {
            match element {
                PatternElement::Node(np) => {
                    if let Some(ref var) = np.variable {
                        self.scope.insert(var.clone(), np.labels.clone());
                    }
                    // Validate labels against schema
                    self.check_labels(&np.labels);
                    // Check property expressions
                    for (_, expr) in &np.properties {
                        self.check_expr(expr);
                    }
                }
                PatternElement::Relationship(rp) => {
                    if let Some(ref var) = rp.variable {
                        self.scope.insert(var.clone(), rp.rel_types.clone());
                    }
                    // Validate edge types against schema
                    self.check_edge_types(&rp.rel_types);
                    // Check property expressions
                    for (_, expr) in &rp.properties {
                        self.check_expr(expr);
                    }
                }
            }
        }
    }

    /// Check that all variable references in an expression are defined.
    fn check_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Variable(name) => {
                if !self.scope.contains_key(name) {
                    self.errors
                        .push(SemanticError::UndefinedVariable { name: name.clone() });
                }
            }
            Expr::PropertyAccess { expr, property } => {
                self.check_expr(expr);
                // If we can resolve the variable's label, check the property
                if let Expr::Variable(ref var_name) = **expr {
                    self.check_property_on_variable(var_name, property);
                }
            }
            Expr::BinaryOp { left, right, .. } => {
                self.check_expr(left);
                self.check_expr(right);
            }
            Expr::UnaryOp { expr, .. } => {
                self.check_expr(expr);
            }
            Expr::FunctionCall { args, .. } => {
                for arg in args {
                    if *arg != Expr::Star {
                        self.check_expr(arg);
                    }
                }
            }
            Expr::List(items) => {
                for item in items {
                    self.check_expr(item);
                }
            }
            Expr::MapLiteral(entries) => {
                for (_, v) in entries {
                    self.check_expr(v);
                }
            }
            Expr::In { expr, list } => {
                self.check_expr(expr);
                self.check_expr(list);
            }
            Expr::IsNull { expr, .. } => {
                self.check_expr(expr);
            }
            Expr::StringMatch { expr, pattern, .. } => {
                self.check_expr(expr);
                self.check_expr(pattern);
            }
            Expr::Case {
                operand,
                when_clauses,
                else_clause,
            } => {
                if let Some(ref op) = operand {
                    self.check_expr(op);
                }
                for (when, then) in when_clauses {
                    self.check_expr(when);
                    self.check_expr(then);
                }
                if let Some(ref el) = else_clause {
                    self.check_expr(el);
                }
            }
            Expr::MapProjection { expr, items } => {
                self.check_expr(expr);
                for item in items {
                    if let MapProjectionItem::Computed(_, value_expr) = item {
                        self.check_expr(value_expr);
                    }
                }
            }
            Expr::PatternPredicate(pattern) => {
                // Check variables referenced in pattern nodes/relationships
                for elem in &pattern.elements {
                    match elem {
                        PatternElement::Node(node) => {
                            if let Some(ref name) = node.variable {
                                if !self.scope.contains_key(name) {
                                    self.errors.push(SemanticError::UndefinedVariable {
                                        name: name.clone(),
                                    });
                                }
                            }
                            for (_, v) in &node.properties {
                                self.check_expr(v);
                            }
                        }
                        PatternElement::Relationship(rel) => {
                            for (_, v) in &rel.properties {
                                self.check_expr(v);
                            }
                        }
                    }
                }
            }
            Expr::Subscript { expr, index } => {
                self.check_expr(expr);
                self.check_expr(index);
            }
            // Literals, parameters, star — no variable references
            Expr::Literal(_) | Expr::Parameter(_) | Expr::Star => {}
        }
    }

    /// Check that a SET item references valid variables.
    fn check_set_item(&mut self, item: &SetItem) {
        match item {
            SetItem::Property {
                variable,
                property,
                expr,
            } => {
                self.check_var_defined(variable);
                self.check_property_on_variable(variable, property);
                self.check_expr(expr);
            }
            SetItem::PropertyPath {
                variable,
                path,
                expr,
            } => {
                self.check_var_defined(variable);
                // Check the root property (first path segment) on the variable.
                if let Some(first) = path.first() {
                    self.check_property_on_variable(variable, first);
                }
                self.check_expr(expr);
            }
            SetItem::DocFunction {
                variable,
                path,
                value_expr,
                ..
            } => {
                self.check_var_defined(variable);
                if let Some(first) = path.first() {
                    self.check_property_on_variable(variable, first);
                }
                self.check_expr(value_expr);
            }
            SetItem::ReplaceProperties { variable, expr }
            | SetItem::MergeProperties { variable, expr } => {
                self.check_var_defined(variable);
                self.check_expr(expr);
            }
            SetItem::AddLabel { variable, label } => {
                self.check_var_defined(variable);
                self.check_labels(std::slice::from_ref(label));
            }
        }
    }

    /// Check that a REMOVE item references valid variables.
    fn check_remove_item(&mut self, item: &RemoveItem) {
        match item {
            RemoveItem::Property { variable, property } => {
                self.check_var_defined(variable);
                self.check_property_on_variable(variable, property);
            }
            RemoveItem::PropertyPath { variable, path } => {
                self.check_var_defined(variable);
                if let Some(first) = path.first() {
                    self.check_property_on_variable(variable, first);
                }
            }
            RemoveItem::Label { variable, label } => {
                self.check_var_defined(variable);
                self.check_labels(std::slice::from_ref(label));
            }
        }
    }

    fn check_var_defined(&mut self, name: &str) {
        if !self.scope.contains_key(name) {
            self.errors.push(SemanticError::UndefinedVariable {
                name: name.to_string(),
            });
        }
    }

    /// Check that labels exist in the schema (if schema is provided).
    fn check_labels(&mut self, labels: &[String]) {
        if let Some(schema) = self.schema {
            for label in labels {
                if schema.get_label(label).is_none() {
                    self.errors.push(SemanticError::UnknownLabel {
                        name: label.clone(),
                    });
                }
            }
        }
    }

    /// Check that edge types exist in the schema (if schema is provided).
    fn check_edge_types(&mut self, types: &[String]) {
        if let Some(schema) = self.schema {
            for t in types {
                if schema.get_edge_type(t).is_none() {
                    self.errors
                        .push(SemanticError::UnknownEdgeType { name: t.clone() });
                }
            }
        }
    }

    /// Check that a property exists on the label associated with a variable.
    fn check_property_on_variable(&mut self, var_name: &str, property: &str) {
        if let Some(schema) = self.schema {
            if let Some(labels) = self.scope.get(var_name) {
                for label in labels {
                    if let Some(label_schema) = schema.get_label(label) {
                        if label_schema.mode.rejects_unknown()
                            && label_schema.get_property(property).is_none()
                        {
                            self.errors.push(SemanticError::UnknownProperty {
                                label: label.clone(),
                                property: property.to_string(),
                            });
                        }
                    }
                    // Also check edge type schemas
                    if let Some(edge_schema) = schema.get_edge_type(label) {
                        // Edge type schemas don't have strict mode currently,
                        // but we could check if the property exists
                        let _ = edge_schema.get_property(property);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::cypher::parser::parse;
    use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType};

    fn parse_and_analyze(input: &str) -> Vec<SemanticError> {
        let query = parse(input).expect("parse should succeed");
        analyze(&query, None)
    }

    fn parse_and_analyze_with_schema(
        input: &str,
        schema: &dyn SchemaProvider,
    ) -> Vec<SemanticError> {
        let query = parse(input).expect("parse should succeed");
        analyze(&query, Some(schema))
    }

    fn make_schema() -> MapSchemaProvider {
        let mut schema = MapSchemaProvider::new();

        let mut user = LabelSchema::new("User");
        user.add_property(PropertyDef::new("name", PropertyType::String));
        user.add_property(PropertyDef::new("age", PropertyType::Int));
        user.add_property(PropertyDef::new("email", PropertyType::String));
        user.set_strict(true);
        schema.add_label(user);

        let mut movie = LabelSchema::new("Movie");
        movie.add_property(PropertyDef::new("title", PropertyType::String));
        schema.add_label(movie);

        let knows = EdgeTypeSchema::new("KNOWS");
        schema.add_edge_type(knows);

        let mut likes = EdgeTypeSchema::new("LIKES");
        likes.add_property(PropertyDef::new("since", PropertyType::Timestamp));
        schema.add_edge_type(likes);

        schema
    }

    // -- Variable binding --

    #[test]
    fn valid_match_return() {
        let errors = parse_and_analyze("MATCH (n:User) RETURN n");
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
    }

    #[test]
    fn valid_match_where_return() {
        let errors = parse_and_analyze("MATCH (n) WHERE n.age > 25 RETURN n.name");
        assert!(errors.is_empty());
    }

    #[test]
    fn undefined_variable_in_return() {
        let errors = parse_and_analyze("MATCH (n) RETURN m");
        assert_eq!(errors.len(), 1);
        assert!(matches!(
            errors[0],
            SemanticError::UndefinedVariable { ref name } if name == "m"
        ));
    }

    #[test]
    fn undefined_variable_in_where() {
        let errors = parse_and_analyze("MATCH (n) WHERE m.age > 25 RETURN n");
        assert!(errors.iter().any(|e| matches!(
            e,
            SemanticError::UndefinedVariable { ref name } if name == "m"
        )));
    }

    #[test]
    fn relationship_variable_defined() {
        let errors = parse_and_analyze("MATCH (a)-[r:KNOWS]->(b) RETURN a, r, b");
        assert!(errors.is_empty());
    }

    #[test]
    fn multiple_patterns_define_vars() {
        let errors = parse_and_analyze("MATCH (a:User), (b:Movie) RETURN a, b");
        assert!(errors.is_empty());
    }

    // -- Scope chain (WITH barrier) --

    #[test]
    fn with_projects_variables() {
        let errors = parse_and_analyze("MATCH (n:User) WITH n RETURN n");
        assert!(errors.is_empty());
    }

    #[test]
    fn with_hides_unprojected_variables() {
        let errors = parse_and_analyze("MATCH (a:User), (b:Movie) WITH a RETURN b");
        // b is not projected through WITH, so RETURN b is an error
        assert!(errors.iter().any(|e| matches!(
            e,
            SemanticError::UndefinedVariable { ref name } if name == "b"
        )));
    }

    #[test]
    fn with_alias_creates_new_variable() {
        let errors = parse_and_analyze("MATCH (n:User) WITH n.name AS username RETURN username");
        assert!(errors.is_empty());
    }

    #[test]
    fn with_alias_old_var_not_visible() {
        let errors = parse_and_analyze("MATCH (n:User) WITH n.name AS username RETURN n");
        // n is not projected, only username
        assert!(errors.iter().any(|e| matches!(
            e,
            SemanticError::UndefinedVariable { ref name } if name == "n"
        )));
    }

    // -- WITH * (Star projection) --

    /// `WITH *` keeps all upstream variables in scope.
    #[test]
    fn with_star_keeps_variables() {
        let errors = parse_and_analyze("MATCH (n:User) WITH * RETURN n");
        assert!(errors.is_empty(), "WITH * must keep n in scope: {errors:?}");
    }

    /// `WITH *, expr AS alias` — Star + explicit alias: both the original variable
    /// AND the new alias must be in scope after WITH.
    ///
    /// REGRESSION: `analyze_with` used to `break` when it encountered Star, so any
    /// aliases listed after Star (e.g. `_dist`) were never added to the new scope.
    /// That caused "undefined variable" errors for ORDER BY / RETURN on the alias.
    #[test]
    fn with_star_plus_alias_both_in_scope() {
        let errors =
            parse_and_analyze("MATCH (n:User) WITH *, n.age AS age_alias RETURN n, age_alias");
        assert!(
            errors.is_empty(),
            "WITH *, expr AS alias must put both n and age_alias in scope: {errors:?}"
        );
    }

    /// ORDER BY on an alias introduced alongside Star must work — this is the
    /// exact pattern used in vector_search / hybrid_search:
    ///   `WITH *, vector_distance(n.emb, $qv) AS _dist ORDER BY _dist LIMIT k`
    #[test]
    fn with_star_alias_usable_in_order_by() {
        let errors = parse_and_analyze(
            "MATCH (n:User) \
             WITH *, n.age AS _score \
             ORDER BY _score DESC \
             LIMIT 10 \
             RETURN n, _score",
        );
        assert!(
            errors.is_empty(),
            "alias from WITH * must be usable in ORDER BY: {errors:?}"
        );
    }

    /// Original variable from Star projection must still be visible in ORDER BY.
    #[test]
    fn with_star_original_var_usable_in_order_by() {
        let errors = parse_and_analyze("MATCH (n:User) WITH * ORDER BY n.name RETURN n");
        assert!(
            errors.is_empty(),
            "WITH * must keep n accessible in ORDER BY: {errors:?}"
        );
    }

    #[test]
    fn unwind_introduces_variable() {
        let errors = parse_and_analyze("UNWIND [1, 2, 3] AS x RETURN x");
        assert!(errors.is_empty());
    }

    // -- Write clause variable checks --

    #[test]
    fn create_introduces_variables() {
        let errors = parse_and_analyze("CREATE (n:User {name: 'Alice'}) RETURN n");
        assert!(errors.is_empty());
    }

    #[test]
    fn set_requires_defined_variable() {
        let errors = parse_and_analyze("MATCH (n:User) SET m.name = 'Bob' RETURN n");
        assert!(errors.iter().any(|e| matches!(
            e,
            SemanticError::UndefinedVariable { ref name } if name == "m"
        )));
    }

    #[test]
    fn delete_requires_variable() {
        let errors = parse_and_analyze("MATCH (n:User) DELETE n");
        assert!(errors.is_empty());
    }

    #[test]
    fn delete_undefined_variable() {
        let errors = parse_and_analyze("MATCH (n:User) DELETE m");
        assert!(errors.iter().any(|e| matches!(
            e,
            SemanticError::UndefinedVariable { ref name } if name == "m"
        )));
    }

    #[test]
    fn merge_introduces_variable() {
        let errors = parse_and_analyze("MERGE (n:User {email: 'alice@test.com'}) RETURN n");
        assert!(errors.is_empty());
    }

    #[test]
    fn merge_on_match_set_uses_merge_var() {
        let errors = parse_and_analyze(
            "MERGE (n:User {email: 'test@test.com'}) \
             ON MATCH SET n.name = 'Bob' \
             RETURN n",
        );
        assert!(errors.is_empty());
    }

    // -- RETURN * --

    #[test]
    fn return_star_with_variables() {
        let errors = parse_and_analyze("MATCH (n) RETURN *");
        assert!(errors.is_empty());
    }

    #[test]
    fn return_star_no_variables() {
        // Edge case: RETURN * without any MATCH
        // This requires creating a query with just RETURN *
        // but our parser requires at least one clause before RETURN
        // Actually, in Cypher you can have standalone RETURN
        // Let's test with an empty scope
        let query = Query {
            clauses: vec![Clause::Return(ReturnClause {
                distinct: false,
                items: vec![ReturnItem {
                    expr: Expr::Star,
                    alias: None,
                }],
            })],
            hints: Vec::new(),
        };
        let errors = analyze(&query, None);
        assert!(errors
            .iter()
            .any(|e| matches!(e, SemanticError::ReturnStarEmpty)));
    }

    // -- ORDER BY alias resolution --

    #[test]
    fn order_by_return_alias() {
        // ORDER BY should see aliases defined in RETURN ... AS
        let errors = parse_and_analyze("MATCH (n) RETURN n.age AS age ORDER BY age");
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
    }

    #[test]
    fn order_by_return_alias_expression() {
        // Alias from a function call in RETURN
        let errors = parse_and_analyze("MATCH (n) RETURN count(n) AS cnt ORDER BY cnt");
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
    }

    #[test]
    fn order_by_original_variable_still_works() {
        // ORDER BY should still see original variables (not just aliases)
        let errors = parse_and_analyze("MATCH (n) RETURN n.name ORDER BY n.age");
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
    }

    #[test]
    fn order_by_undefined_alias_fails() {
        // Alias not defined anywhere should still fail
        let errors = parse_and_analyze("MATCH (n) RETURN n.name ORDER BY nonexistent");
        assert!(errors.iter().any(|e| matches!(
            e,
            SemanticError::UndefinedVariable { ref name } if name == "nonexistent"
        )));
    }

    // -- Schema validation --

    #[test]
    fn unknown_label() {
        let schema = make_schema();
        let errors = parse_and_analyze_with_schema("MATCH (n:NonExistent) RETURN n", &schema);
        assert!(errors.iter().any(|e| matches!(
            e,
            SemanticError::UnknownLabel { ref name } if name == "NonExistent"
        )));
    }

    #[test]
    fn known_label() {
        let schema = make_schema();
        let errors = parse_and_analyze_with_schema("MATCH (n:User) RETURN n", &schema);
        assert!(errors.is_empty());
    }

    #[test]
    fn unknown_edge_type() {
        let schema = make_schema();
        let errors =
            parse_and_analyze_with_schema("MATCH (a)-[:NONEXISTENT]->(b) RETURN a, b", &schema);
        assert!(errors.iter().any(|e| matches!(
            e,
            SemanticError::UnknownEdgeType { ref name } if name == "NONEXISTENT"
        )));
    }

    #[test]
    fn known_edge_type() {
        let schema = make_schema();
        let errors = parse_and_analyze_with_schema("MATCH (a)-[:KNOWS]->(b) RETURN a, b", &schema);
        assert!(errors.is_empty());
    }

    #[test]
    fn unknown_property_on_strict_label() {
        let schema = make_schema();
        let errors = parse_and_analyze_with_schema(
            "MATCH (n:User) WHERE n.nonexistent = 'foo' RETURN n",
            &schema,
        );
        assert!(errors.iter().any(|e| matches!(
            e,
            SemanticError::UnknownProperty { ref label, ref property }
                if label == "User" && property == "nonexistent"
        )));
    }

    #[test]
    fn known_property_on_strict_label() {
        let schema = make_schema();
        let errors = parse_and_analyze_with_schema(
            "MATCH (n:User) WHERE n.name = 'Alice' RETURN n",
            &schema,
        );
        assert!(errors.is_empty());
    }

    #[test]
    fn no_schema_skips_validation() {
        // Without schema, unknown labels/properties are not errors
        let errors = parse_and_analyze("MATCH (n:Whatever) WHERE n.anything = 'foo' RETURN n");
        assert!(errors.is_empty());
    }

    // -- Complex queries --

    #[test]
    fn complex_with_chain() {
        let errors = parse_and_analyze(
            "MATCH (n:User)-[:KNOWS]->(m:User) \
             WITH n, count(*) AS friend_count \
             WHERE friend_count > 5 \
             RETURN n.name, friend_count",
        );
        assert!(errors.is_empty());
    }

    #[test]
    fn complex_match_create_return() {
        let errors = parse_and_analyze(
            "MATCH (a:User {name: 'Alice'}), (b:User {name: 'Bob'}) \
             CREATE (a)-[:KNOWS]->(b) \
             RETURN a, b",
        );
        assert!(errors.is_empty());
    }

    #[test]
    fn remove_requires_defined_variable() {
        let errors = parse_and_analyze("MATCH (n) REMOVE m.age");
        assert!(errors.iter().any(|e| matches!(
            e,
            SemanticError::UndefinedVariable { ref name } if name == "m"
        )));
    }

    #[test]
    fn set_label_valid() {
        let errors = parse_and_analyze("MATCH (n) SET n:Admin");
        assert!(errors.is_empty());
    }

    #[test]
    fn parameters_always_valid() {
        let errors = parse_and_analyze("MATCH (n:User {id: $userId}) RETURN n");
        assert!(errors.is_empty());
    }

    #[test]
    fn aggregation_in_return() {
        let errors = parse_and_analyze("MATCH (n:User) RETURN n.city, count(*) AS cnt");
        assert!(errors.is_empty());
    }

    #[test]
    fn nested_property_access() {
        let errors =
            parse_and_analyze("MATCH (n:User) WHERE n.age > 18 AND n.name = 'Alice' RETURN n");
        assert!(errors.is_empty());
    }
}
