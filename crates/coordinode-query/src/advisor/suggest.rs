//! Suggestion types for the query advisor.
//!
//! A `Suggestion` is a concrete, actionable recommendation produced by analyzing
//! a query's logical plan. Each suggestion includes what to change, why, and the
//! expected improvement.

/// Severity of a suggestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Informational: possible improvement, low priority.
    Info,
    /// Warning: likely performance issue, should investigate.
    Warning,
    /// Critical: definite performance problem, high priority.
    Critical,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARNING"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Kind of suggestion — what category of optimization is recommended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SuggestionKind {
    /// Create a missing index for a filtered property.
    CreateIndex,
    /// Add depth bounds to unbounded variable-length traversal.
    AddDepthBound,
    /// Add join predicate to eliminate Cartesian product.
    AddJoinPredicate,
    /// Create a vector index for KNN-style queries.
    CreateVectorIndex,
    /// Add graph pre-filter before vector scan.
    AddGraphPreFilter,
    /// Rewrite N+1 loop as a batched UNWIND query.
    BatchRewrite,
}

impl std::fmt::Display for SuggestionKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CreateIndex => write!(f, "CREATE INDEX"),
            Self::AddDepthBound => write!(f, "ADD DEPTH BOUND"),
            Self::AddJoinPredicate => write!(f, "ADD JOIN"),
            Self::CreateVectorIndex => write!(f, "CREATE VECTOR INDEX"),
            Self::AddGraphPreFilter => write!(f, "ADD PRE-FILTER"),
            Self::BatchRewrite => write!(f, "BATCH REWRITE"),
        }
    }
}

/// A concrete, actionable suggestion from the query advisor.
#[derive(Debug, Clone)]
pub struct Suggestion {
    /// What kind of optimization is recommended.
    pub kind: SuggestionKind,
    /// How important is this suggestion.
    pub severity: Severity,
    /// Human-readable explanation of the problem and fix.
    pub explanation: String,
    /// Executable DDL statement (if applicable).
    pub ddl: Option<String>,
    /// Suggested query rewrite (if applicable).
    pub rewritten_query: Option<String>,
}

impl Suggestion {
    /// Create a new suggestion.
    pub fn new(kind: SuggestionKind, severity: Severity, explanation: impl Into<String>) -> Self {
        Self {
            kind,
            severity,
            explanation: explanation.into(),
            ddl: None,
            rewritten_query: None,
        }
    }

    /// Add a DDL statement to the suggestion.
    pub fn with_ddl(mut self, ddl: impl Into<String>) -> Self {
        self.ddl = Some(ddl.into());
        self
    }

    /// Add a query rewrite to the suggestion.
    pub fn with_rewrite(mut self, rewrite: impl Into<String>) -> Self {
        self.rewritten_query = Some(rewrite.into());
        self
    }
}

impl std::fmt::Display for Suggestion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}: {}", self.severity, self.kind, self.explanation)?;
        if let Some(ddl) = &self.ddl {
            write!(f, "\n  DDL: {ddl}")?;
        }
        if let Some(rewrite) = &self.rewritten_query {
            write!(f, "\n  Rewrite: {rewrite}")?;
        }
        Ok(())
    }
}

/// Result of EXPLAIN SUGGEST: plan explanation + suggestions.
#[derive(Debug, Clone)]
pub struct ExplainSuggestResult {
    /// Standard EXPLAIN output (plan tree + cost).
    pub explain: String,
    /// Ranked suggestions (sorted by severity descending).
    pub suggestions: Vec<Suggestion>,
}

impl std::fmt::Display for ExplainSuggestResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.explain)?;
        if self.suggestions.is_empty() {
            writeln!(f, "No suggestions — query plan looks good.")?;
        } else {
            writeln!(f, "SUGGESTIONS ({}):", self.suggestions.len())?;
            for (i, s) in self.suggestions.iter().enumerate() {
                writeln!(f, "  {}. {s}", i + 1)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
