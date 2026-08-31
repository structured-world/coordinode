//! Machine-readable failure reasons, carried the canonical way.
//!
//! A gRPC status code is a category, not an identity: `INVALID_ARGUMENT`
//! covers a syntax error, a division by zero and a misspelled function alike,
//! and a client that needs to tell them apart has nothing to key on. Reading
//! the message text instead works until someone rewords it.
//!
//! So every failure a client may reasonably branch on also carries a
//! [`google.rpc.ErrorInfo`] in the `grpc-status-details-bin` trailer, with a
//! stable [`Reason`] string, this server's [`ERROR_DOMAIN`], and whatever
//! values the caller needs to act (the offending function name, the id of a
//! transaction that no longer exists). The code and the message stay exactly
//! as they were, so a client that ignores the details is unaffected: this
//! extends the error surface, it does not change it.
//!
//! The reason strings are part of the API. Renaming one breaks callers as
//! surely as renaming an RPC, so treat this list the way you would treat the
//! proto.

use tonic::{Code, Status};
use tonic_types::{ErrorDetails, StatusExt};

/// Namespace for every reason below, as `ErrorInfo.domain`.
///
/// Reasons are only unique within a domain, so a client matching on one
/// without checking the domain can collide with another service's error when
/// both sit behind the same gateway.
pub const ERROR_DOMAIN: &str = "coordinode.sw.foundation";

/// Why a request failed, in terms a program can act on.
///
/// One variant per situation a caller might handle differently. Where two
/// failures call for the same handling they share a reason, since a
/// distinction nobody can act on is noise the API has to keep forever.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reason {
    /// The query did not parse.
    QuerySyntax,
    /// The query parsed but does not mean anything: an undefined variable, an
    /// unknown label, an aggregate where none may stand.
    QuerySemantics,
    /// A call to a function this server does not implement. The name is in
    /// the metadata under `function`.
    UnknownFunction,
    /// Division or modulo by an integer zero.
    DivideByZero,
    /// An integer operation whose exact result leaves the 64-bit range.
    LongOverflow,
    /// No transaction is held under this id: never opened, or already
    /// finished, aborted or swept. Either way the server holds nothing for it,
    /// so there is nothing left to clean up.
    UnknownTransaction,
    /// A commit lost to a concurrent write. Nothing of the transaction was
    /// applied; retrying the whole transaction is the intended response.
    TransactionConflict,
    /// A transaction buffered more uncommitted data than it may. Splitting the
    /// work into smaller transactions is the fix; retrying as-is will not help.
    TransactionTooLarge,
    /// The write would exceed the endpoint's storage quota. Metadata carries
    /// `endpoint_id`, `used_bytes` and `hard_limit_bytes`.
    CapacityExhausted,
    /// A schema rule refused the write: an undeclared property on a strict
    /// label, or an attempt to set a computed one.
    SchemaViolation,
}

impl Reason {
    /// The wire form. Stable: callers match on these strings.
    pub const fn as_str(self) -> &'static str {
        match self {
            Reason::QuerySyntax => "QUERY_SYNTAX",
            Reason::QuerySemantics => "QUERY_SEMANTICS",
            Reason::UnknownFunction => "UNKNOWN_FUNCTION",
            Reason::DivideByZero => "DIVIDE_BY_ZERO",
            Reason::LongOverflow => "LONG_OVERFLOW",
            Reason::UnknownTransaction => "UNKNOWN_TRANSACTION",
            Reason::TransactionConflict => "TRANSACTION_CONFLICT",
            Reason::TransactionTooLarge => "TRANSACTION_TOO_LARGE",
            Reason::CapacityExhausted => "CAPACITY_EXHAUSTED",
            Reason::SchemaViolation => "SCHEMA_VIOLATION",
        }
    }

    /// Whether repeating the same request unchanged could succeed.
    ///
    /// This is the server's own judgement, published so that a client does not
    /// have to encode a table of ours. A conflict is the one reason here worth
    /// retrying: everything else fails the same way every time until either
    /// the request or the data changes.
    pub const fn is_retryable(self) -> bool {
        matches!(self, Reason::TransactionConflict)
    }
}

/// Build a status that carries `reason` alongside the usual code and message.
///
/// `metadata` holds the values a caller needs to act on the failure rather
/// than merely report it, and is keyed by short snake_case names.
pub fn status_with_reason(
    code: Code,
    message: impl Into<String>,
    reason: Reason,
    metadata: impl IntoIterator<Item = (&'static str, String)>,
) -> Status {
    let metadata: std::collections::HashMap<String, String> = metadata
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    let mut details = ErrorDetails::with_error_info(reason.as_str(), ERROR_DOMAIN, metadata);
    if reason.is_retryable() {
        // Zero delay: a conflict is resolved by re-running the transaction,
        // not by waiting for the server to recover. Saying "retry now"
        // explicitly is what stops a client from inventing a backoff for a
        // condition that backing off does not help.
        details.set_retry_info(Some(std::time::Duration::ZERO));
    }
    Status::with_error_details(code, message, details)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
