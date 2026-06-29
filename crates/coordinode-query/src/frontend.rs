//! Query-language frontend: the seam between a dialect's surface syntax and the
//! language-neutral logical IR.
//!
//! Everything below the IR (planner optimizer passes, executor, advisor) is
//! dialect-agnostic. A [`QueryFrontend`] is the one component that knows a
//! specific query language: it parses the surface text, validates it, lowers it
//! into a [`LogicalPlan`], and produces the canonical form + fingerprint the
//! advisor and plan cache key off. [`CypherFrontend`] is the first (and
//! currently only) implementation; a SQL frontend plugs in here without
//! touching anything below the IR.

use crate::cypher::{ParseError, SemanticError};
use crate::planner::builder::PlanError;
use crate::planner::logical::LogicalPlan;

/// A query parsed and lowered into the neutral IR, with the metadata the
/// execution entry point needs: the logical plan, its canonical text form, and
/// the stable fingerprint over that canonical form.
#[derive(Debug, Clone)]
pub struct ParsedQuery {
    /// The unoptimized logical plan. Optimizer passes run on a clone at
    /// execution time so they observe the current index registry / stats.
    pub plan: LogicalPlan,
    /// Canonical (normalized) text form of the query, keyed by the advisor.
    pub canonical: String,
    /// Stable fingerprint over [`ParsedQuery::canonical`].
    pub fingerprint: u64,
}

/// Why a [`QueryFrontend`] could not turn surface text into a [`ParsedQuery`].
#[derive(Debug)]
pub enum FrontendError {
    /// The surface syntax did not parse.
    Parse(ParseError),
    /// The query parsed but failed semantic validation.
    Semantic(Vec<SemanticError>),
    /// The query was valid but could not be lowered into a logical plan.
    Plan(PlanError),
    /// A dialect-agnostic frontend error (parse / lowering failure not tied to
    /// the cypher semantic-analysis types — e.g. the SQL frontend).
    Message(String),
}

impl std::fmt::Display for FrontendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrontendError::Parse(e) => write!(f, "{e}"),
            FrontendError::Semantic(errors) => {
                let joined = errors
                    .iter()
                    .map(|e| e.to_string())
                    .collect::<Vec<_>>()
                    .join("; ");
                write!(f, "{joined}")
            }
            FrontendError::Plan(e) => write!(f, "{e}"),
            FrontendError::Message(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for FrontendError {}

/// A query-language frontend: turns a dialect's surface syntax into the neutral
/// IR plus its canonical form and fingerprint. The layers below the IR consume
/// the [`ParsedQuery`] without ever seeing the originating dialect.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a query frontend",
    label = "this type does not implement `QueryFrontend`",
    note = "the canonical frontend is `CypherFrontend`; build one with `CypherFrontend::new()`"
)]
pub trait QueryFrontend {
    /// Parse, validate, and lower `text` into the neutral IR.
    fn parse(&self, text: &str) -> Result<ParsedQuery, FrontendError>;

    /// Parse `text` and compute its canonical form and fingerprint without
    /// building a plan. Advisor / query-tracking paths key off the fingerprint
    /// but never execute, so they avoid the cost of lowering a full plan.
    /// Canonicalization is dialect-specific, so it belongs to the frontend.
    fn fingerprint(&self, text: &str) -> Result<(String, u64), FrontendError>;
}

/// The Cypher frontend: parses Cypher, runs semantic analysis, lowers to the
/// neutral logical plan, and computes the canonical form + fingerprint.
#[derive(Debug, Clone, Copy, Default)]
pub struct CypherFrontend;

impl CypherFrontend {
    /// Create a Cypher frontend.
    pub fn new() -> Self {
        Self
    }
}

impl QueryFrontend for CypherFrontend {
    fn parse(&self, text: &str) -> Result<ParsedQuery, FrontendError> {
        let ast = crate::cypher::parse(text).map_err(FrontendError::Parse)?;
        let (canonical, fingerprint) = crate::advisor::normalize_and_fingerprint(&ast);
        let errors = crate::cypher::analyze(&ast, None);
        if !errors.is_empty() {
            return Err(FrontendError::Semantic(errors));
        }
        let plan = crate::planner::build_logical_plan(&ast).map_err(FrontendError::Plan)?;
        Ok(ParsedQuery {
            plan,
            canonical,
            fingerprint,
        })
    }

    fn fingerprint(&self, text: &str) -> Result<(String, u64), FrontendError> {
        let ast = crate::cypher::parse(text).map_err(FrontendError::Parse)?;
        Ok(crate::advisor::normalize_and_fingerprint(&ast))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
