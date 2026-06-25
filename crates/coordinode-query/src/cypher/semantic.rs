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

    /// UNION branches project different column names. All branches of a
    /// UNION / UNION ALL must return the same columns in the same order.
    #[error("all UNION branches must have the same column names: {left:?} vs {right:?}")]
    UnionColumnMismatch {
        left: Vec<String>,
        right: Vec<String>,
    },
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

/// Output column names of a query branch, taken from its final RETURN clause.
///
/// Returns `None` when the branch has no RETURN or projects `*` (the column
/// set then depends on runtime scope and can't be compared statically — the
/// UNION column check is skipped in that case).
fn return_columns(clauses: &[Clause]) -> Option<Vec<String>> {
    let rc = clauses.iter().rev().find_map(|c| match c {
        Clause::Return(rc) => Some(rc),
        _ => None,
    })?;
    let mut cols = Vec::with_capacity(rc.items.len());
    for item in &rc.items {
        if matches!(item.expr, Expr::Star) {
            return None;
        }
        cols.push(
            item.alias
                .clone()
                .unwrap_or_else(|| column_display_name(&item.expr)),
        );
    }
    Some(cols)
}

/// Derive a stable column name from a projection expression (used only for the
/// UNION column-compatibility check; mirrors the executor's projection naming).
fn column_display_name(expr: &Expr) -> String {
    match expr {
        Expr::Variable(name) => name.clone(),
        Expr::PropertyAccess { expr, property } => {
            format!("{}.{property}", column_display_name(expr))
        }
        Expr::FunctionCall { name, .. } => name.clone(),
        other => format!("{other:?}"),
    }
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

        if query.unions.is_empty() {
            return;
        }

        // Each UNION branch is its own variable scope.
        let first_cols = return_columns(&query.clauses);
        for branch in &query.unions {
            let saved = std::mem::take(&mut self.scope);
            for clause in &branch.clauses {
                self.analyze_clause(clause);
            }
            self.scope = saved;

            // Column-name compatibility: all branches must project the same
            // columns in the same order. `*` projections can't be compared
            // statically, so a branch using `RETURN *` skips the check.
            if let (Some(left), Some(right)) = (&first_cols, return_columns(&branch.clauses)) {
                if *left != right {
                    self.errors.push(SemanticError::UnionColumnMismatch {
                        left: left.clone(),
                        right,
                    });
                }
            }
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
            Clause::MergeNodes(mn) => {
                // Both source variables must be bound by a preceding MATCH.
                for v in [&mn.source_a, &mn.source_b] {
                    if !self.scope.contains_key(v) {
                        self.errors
                            .push(SemanticError::UndefinedVariable { name: v.clone() });
                    }
                }
                if let MergeNodesConflictStrategy::SetExpressions(ref items) = mn.conflict {
                    for item in items {
                        self.check_set_item(item);
                    }
                }
            }
            Clause::CloneNode(cn) => {
                // Source must be bound by a preceding MATCH.
                if !self.scope.contains_key(&cn.source) {
                    self.errors.push(SemanticError::UndefinedVariable {
                        name: cn.source.clone(),
                    });
                }
                // Bind the clone variable for subsequent clauses, inheriting the
                // source's statically-known labels (the clone keeps them).
                let labels = self.scope.get(&cn.source).cloned().unwrap_or_default();
                self.scope.insert(cn.target.clone(), labels);
                for item in &cn.set_items {
                    self.check_set_item(item);
                }
            }
            Clause::RedirectEdges(re) => {
                // Both endpoints must be bound by a preceding MATCH; the clause
                // introduces no new variables.
                for v in [&re.source, &re.target] {
                    if !self.scope.contains_key(v) {
                        self.errors
                            .push(SemanticError::UndefinedVariable { name: v.clone() });
                    }
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
            | Clause::DropVectorIndex(_)
            | Clause::CreateEdgeType(_)
            | Clause::CreateNodeType(_)
            | Clause::CreateTrigger(_)
            | Clause::DropTrigger(_)
            | Clause::ShowTriggers
            | Clause::AlterTrigger(_) => {
                // DDL — no variable references to validate.
                // Edge-type property type names are validated at execution time
                // against `PropertyType` enum in the DDL executor. Trigger body
                // re-parses and validates at execution time too (the body is a
                // raw Cypher string captured by the parser).
            }
            Clause::Foreach(fc) => {
                // The list expression resolves in the outer scope.
                self.check_expr(&fc.list);
                // The loop variable is bound only inside the body; record it,
                // analyze the body clauses, then restore the outer scope.
                let had_var = self.scope.insert(fc.variable.clone(), Vec::new());
                for body_clause in &fc.body {
                    self.analyze_clause(body_clause);
                }
                match had_var {
                    Some(prev) => {
                        self.scope.insert(fc.variable.clone(), prev);
                    }
                    None => {
                        self.scope.remove(&fc.variable);
                    }
                }
            }
            Clause::CallSubquery(cs) => {
                // The subquery runs in its own scope. A leading `WITH` imports
                // outer variables (so we keep the current scope visible while
                // analyzing the body); the body's final RETURN columns become
                // available to subsequent outer clauses.
                let outer = self.scope.clone();
                for body_clause in &cs.body {
                    self.analyze_clause(body_clause);
                }
                // Variables introduced by the subquery's RETURN remain in scope;
                // restore the outer variables that the body may have shadowed.
                for (k, v) in outer {
                    self.scope.entry(k).or_insert(v);
                }
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
        // A named path (`p = ...`, including shortestPath) binds the path
        // variable so RETURN p / length(p) / nodes(p) resolve.
        if let Some(ref pv) = pattern.path_variable {
            self.scope.insert(pv.clone(), Vec::new());
        }
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
            Expr::IsTyped { expr, .. } => {
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
            Expr::Slice { expr, start, end } => {
                self.check_expr(expr);
                if let Some(s) = start {
                    self.check_expr(s);
                }
                if let Some(e) = end {
                    self.check_expr(e);
                }
            }
            Expr::Reduce {
                acc,
                init,
                var,
                list,
                expr,
            } => {
                // init and list resolve in the outer scope; acc and var are
                // bound only while checking the step expression, then restored.
                self.check_expr(init);
                self.check_expr(list);
                let prev_acc = self.scope.insert(acc.clone(), Vec::new());
                let prev_var = self.scope.insert(var.clone(), Vec::new());
                self.check_expr(expr);
                match prev_var {
                    Some(v) => {
                        self.scope.insert(var.clone(), v);
                    }
                    None => {
                        self.scope.remove(var);
                    }
                }
                match prev_acc {
                    Some(v) => {
                        self.scope.insert(acc.clone(), v);
                    }
                    None => {
                        self.scope.remove(acc);
                    }
                }
            }
            Expr::ExistsSubquery(_) | Expr::CountSubquery(_) | Expr::CollectSubquery { .. } => {
                // The inner MATCH introduces its own scope (and may correlate on
                // outer variables); it is fully validated by the logical planner
                // when the subquery is built, so no outer-scope check here.
            }
            Expr::PatternComprehension { .. } => {
                // Same as EXISTS: the inner pattern introduces its own scope and
                // is validated by the planner when the comprehension is built.
            }
            Expr::ListPredicate {
                var, list, pred, ..
            } => {
                // list resolves in the outer scope; var is bound only while
                // checking the predicate, then restored.
                self.check_expr(list);
                let prev = self.scope.insert(var.clone(), Vec::new());
                self.check_expr(pred);
                match prev {
                    Some(v) => {
                        self.scope.insert(var.clone(), v);
                    }
                    None => {
                        self.scope.remove(var);
                    }
                }
            }
            Expr::ListComprehension {
                var,
                list,
                pred,
                map,
            } => {
                self.check_expr(list);
                let prev = self.scope.insert(var.clone(), Vec::new());
                if let Some(p) = pred {
                    self.check_expr(p);
                }
                if let Some(m) = map {
                    self.check_expr(m);
                }
                match prev {
                    Some(v) => {
                        self.scope.insert(var.clone(), v);
                    }
                    None => {
                        self.scope.remove(var);
                    }
                }
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
mod tests;
