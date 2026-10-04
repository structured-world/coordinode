//! Machine-readable failure reasons, carried the canonical way.
//!
//! A gRPC status code is a category, not an identity: `INVALID_ARGUMENT`
//! covers a syntax error, a division by zero and a misspelled function alike,
//! and a client that needs to tell them apart has nothing to key on. Reading
//! the message text instead works until someone rewords it.
//!
//! So every failure a client may reasonably branch on also carries a
//! `google.rpc.ErrorInfo` in the `grpc-status-details-bin` trailer, with a
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
pub const ERROR_DOMAIN: &str = coordinode_core::ERROR_DOMAIN;

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
    /// A `CALL` of a procedure this server does not have. The name is in the
    /// metadata under `procedure`. Terminal.
    UnknownProcedure,
    /// A `CALL` that does not match the procedure's signature: too many or
    /// missing arguments, an argument of the wrong type or value, or a YIELD
    /// of a column it does not produce. Nothing ran. Metadata carries
    /// `procedure`, and `argument` or `column` when one is at fault. Terminal.
    ProcedureCall,
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
    /// A declared condition the write depends on no longer holds, or a
    /// concurrent transaction holds an incompatible claim on it. Nothing was
    /// applied; re-running the transaction re-reads that state.
    ///
    /// Told apart from `TRANSACTION_CONFLICT` so a caller can see which of the
    /// two it hit: a conflict is two transactions over one key, a refusal is
    /// two of them over one condition, which their key sets need not share.
    InvariantRefused,
    /// A write conditioned on a record's version found a different one.
    /// Nothing was applied. The metadata carries `expected_version` and
    /// `current_version`, so a caller decides what to do without reading the
    /// record again; `current_version` is absent when the record is gone.
    ///
    /// Told apart from `TRANSACTION_CONFLICT` because the caller asked for
    /// this: it named a version, and what it needs back is which version is
    /// there, not that something contended.
    RevisionMismatch,
    /// A transaction buffered more uncommitted data than it may. Splitting the
    /// work into smaller transactions is the fix; retrying as-is will not help.
    TransactionTooLarge,
    /// The write would exceed the endpoint's storage quota. Metadata carries
    /// `endpoint_id`, `used_bytes` and `hard_limit_bytes`.
    CapacityExhausted,
    /// The disk under the data directory is below its free-space reserve, so
    /// writes are refused before any reaches the disk; reads go on. Metadata
    /// carries `path`, `available_bytes` and `min_free_bytes`. Retry once
    /// space is freed.
    StorageFull,
    /// A schema rule refused the write: an undeclared property on a strict
    /// label, or an attempt to set a computed one.
    SchemaViolation,
    /// Storage is over its compaction-debt stop threshold and is shedding
    /// new writes until compaction catches up. Nothing was applied; retry
    /// after the advised delay.
    WriteBackpressure,
    /// The write reached a node that is not the leader. Nothing was applied.
    /// Metadata carries `leader_id` when the cluster has named one, so the
    /// caller can retry at the right node instead of guessing.
    NotLeader,
    /// The write reached a member that does not run the version its group
    /// runs, so it is read-only. Nothing was applied. Metadata carries
    /// `member_engine_format`, `member_host_epoch`, `group_engine_format`,
    /// `group_host_epoch`, `behind` (whether the group moved past this
    /// member), `as_of_ts` (what its reads are as of), and `leader_id` and
    /// `leader_addr` when known, so the caller retries at the leader.
    MemberReadOnly,
    /// A time-travel read (`AS OF TIMESTAMP`, `ReadConcern.at_timestamp`)
    /// older than the MVCC retention horizon; that history may be collected.
    /// Metadata carries `oldest_readable_ts`, the earliest timestamp the same
    /// read succeeds at. Terminal for that timestamp: the horizon only moves
    /// forward.
    OutsideRetention,
    /// A read at a named timestamp (`AS OF TIMESTAMP`,
    /// `ReadConcern.at_timestamp`) would be answered by a vector or full-text
    /// index that holds only the current state. Metadata carries
    /// `index_kind` (`vector` or `full-text`), `label`, `property` and
    /// `timestamp`. Terminal for that query: a vector search can be asked
    /// with `vector_consistency('exact')` instead.
    IndexNotHistorical,
    /// A full-text search found the text indexes behind the commits its read
    /// includes, past its wait: answering would miss or misrank committed
    /// writes. Nothing was answered. Metadata carries `folded` and `needed`
    /// (applied-entry positions the indexes hold and the read needed) and
    /// `waited_ms`. Retry after the advised delay.
    TextIndexBehind,
    /// A write concern the server cannot honour: an unknown mode or journal
    /// value, a member count above the group's size, or a volatile journal
    /// level asked of more than one member. Terminal for that request; the
    /// message names the offending combination.
    InvalidWriteConcern,
    /// A write gave a unique key a second holder: a table's key, or a value of
    /// a unique index. Nothing was written. Metadata carries `key` and
    /// `element_id` (the node that holds it), plus `table` for a table key or
    /// `index` and `property` for a unique index. Terminal: the same write
    /// will be refused again.
    DuplicateKey,
    /// A statement tried to change a key column of an existing row. Metadata
    /// carries `table` and `column`. Terminal.
    KeyImmutable,
    /// A write would leave a node breaking a constraint of its label: a
    /// required property missing or null, or a value of another type.
    /// Nothing was written. Metadata carries `constraint`, `kind`, `label`,
    /// `property` and `element_id`. Terminal: the same write is refused again.
    ConstraintViolation,
    /// A change stream's position is no longer covered by the retained log:
    /// the entries after it were purged, so the stream cannot continue from
    /// there without a gap. Metadata carries `requested_index` and
    /// `first_retained_index` when the position is known. Terminal for that
    /// position: resubscribing from the same token is refused again.
    RetentionLost,
    /// A change-stream consumer's registration has ended: cancelled, or a
    /// BOUNDED bound was crossed. Metadata carries `incarnation`, `reason` and
    /// `checkpoint` (the last position acknowledged). Terminal for that
    /// incarnation: registering the id again starts a new one.
    ConsumerTerminated,
    /// A request field has a value the server refuses: missing, malformed
    /// or contradicting another field. Nothing changed. Metadata carries
    /// `field`, the path of the field; `BadRequest` describes the violation.
    /// Terminal: the same request is refused again.
    InvalidField,
    /// A catalog object of the name already exists: a label, edge type,
    /// constraint or index. Nothing changed. Metadata carries `object` and
    /// `name`, also given as `ResourceInfo`. Terminal for that name.
    CatalogObjectExists,
    /// No catalog object of the name exists. Nothing changed. Metadata
    /// carries `object` and `name`, also given as `ResourceInfo`.
    CatalogObjectNotFound,
    /// A catalog change the catalog's current state refuses: a definition
    /// its constraints cannot hold under, a constraint still being
    /// validated, a dependency that forbids it. Nothing changed. Terminal
    /// until that state changes.
    CatalogChangeRefused,
}

impl Reason {
    /// The wire form. Stable: callers match on these strings.
    pub const fn as_str(self) -> &'static str {
        match self {
            Reason::QuerySyntax => "QUERY_SYNTAX",
            Reason::QuerySemantics => "QUERY_SEMANTICS",
            Reason::UnknownFunction => "UNKNOWN_FUNCTION",
            Reason::UnknownProcedure => "UNKNOWN_PROCEDURE",
            Reason::ProcedureCall => "PROCEDURE_CALL",
            Reason::DivideByZero => "DIVIDE_BY_ZERO",
            Reason::LongOverflow => "LONG_OVERFLOW",
            Reason::UnknownTransaction => "UNKNOWN_TRANSACTION",
            Reason::TransactionConflict => "TRANSACTION_CONFLICT",
            Reason::InvariantRefused => "INVARIANT_REFUSED",
            Reason::RevisionMismatch => "REVISION_MISMATCH",
            Reason::TransactionTooLarge => "TRANSACTION_TOO_LARGE",
            Reason::CapacityExhausted => "CAPACITY_EXHAUSTED",
            Reason::StorageFull => "STORAGE_FULL",
            Reason::SchemaViolation => "SCHEMA_VIOLATION",
            Reason::WriteBackpressure => "WRITE_BACKPRESSURE",
            Reason::NotLeader => "NOT_LEADER",
            Reason::MemberReadOnly => "MEMBER_READ_ONLY",
            Reason::OutsideRetention => "OUTSIDE_RETENTION",
            Reason::IndexNotHistorical => "INDEX_NOT_HISTORICAL",
            Reason::TextIndexBehind => "TEXT_INDEX_BEHIND",
            Reason::InvalidWriteConcern => "INVALID_WRITE_CONCERN",
            Reason::DuplicateKey => "DUPLICATE_KEY",
            Reason::KeyImmutable => "KEY_IMMUTABLE",
            Reason::ConstraintViolation => "CONSTRAINT_VIOLATION",
            Reason::RetentionLost => "RETENTION_LOST",
            Reason::ConsumerTerminated => "CONSUMER_TERMINATED",
            Reason::InvalidField => "INVALID_FIELD",
            Reason::CatalogObjectExists => "CATALOG_OBJECT_EXISTS",
            Reason::CatalogObjectNotFound => "CATALOG_OBJECT_NOT_FOUND",
            Reason::CatalogChangeRefused => "CATALOG_CHANGE_REFUSED",
        }
    }

    /// The retry advice for a retryable reason, `None` for a terminal one.
    ///
    /// This is the server's own judgement, published (as `RetryInfo`) so that
    /// a client does not have to encode a table of ours.
    ///
    /// A conflict says "retry now": it is resolved by re-running the
    /// transaction, not by waiting, and the explicit zero delay is what stops
    /// a client from inventing a backoff for a condition backing off does not
    /// help. An invariant refusal answers the same way and for the same
    /// reason: the retry re-reads the state the condition is evaluated
    /// against, and waiting does not change what it will read. Backpressure says the opposite: the server is shedding writes
    /// until compaction catches up, so an immediate retry would bounce off
    /// the same verdict; the delay is a floor for the client's backoff.
    ///
    /// A leader change also says "retry now", and for the same reason a
    /// conflict does: waiting changes nothing, the write has to go somewhere
    /// else. When the metadata names a leader the retry is a redirect; when
    /// an election is still in flight it is a short poll, and the client's own
    /// backoff governs how often, which is why the floor here stays zero.
    /// A read-only member answers the same way: the write belongs at the
    /// leader its metadata names.
    ///
    /// A version mismatch deliberately advises nothing. The caller named a
    /// version, so what happens next is its decision and not a delay: it may
    /// retry against the version in the metadata, merge, or stop. Advising a
    /// retry would tell it to repeat a write whose premise the server has
    /// just disproved.
    pub const fn retry_delay(self) -> Option<std::time::Duration> {
        match self {
            Reason::TransactionConflict
            | Reason::InvariantRefused
            | Reason::NotLeader
            | Reason::MemberReadOnly => Some(std::time::Duration::ZERO),
            Reason::WriteBackpressure => Some(std::time::Duration::from_millis(500)),
            // The worker folds applied entries continuously; an index behind
            // past a whole wait catches up in moments, not instantly.
            Reason::TextIndexBehind => Some(std::time::Duration::from_millis(200)),
            // Space comes back when someone frees it, not soon: a floor that
            // keeps clients from hammering a node that only serves reads.
            Reason::StorageFull => Some(std::time::Duration::from_secs(5)),
            _ => None,
        }
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
    if let Some(delay) = reason.retry_delay() {
        details.set_retry_info(Some(delay));
    }
    Status::with_error_details(code, message, details)
}

/// `UNAVAILABLE` for a full-text search whose indexes did not catch up with
/// the commits it needed within its wait.
pub fn text_index_behind(behind: &coordinode_query::index::TextNotReady) -> Status {
    status_with_reason(
        Code::Unavailable,
        behind.to_string(),
        Reason::TextIndexBehind,
        [
            ("folded", behind.folded.to_string()),
            ("needed", behind.needed.to_string()),
            ("waited_ms", behind.waited_ms.to_string()),
        ],
    )
}

/// `INVALID_ARGUMENT` for request field `field` (its path, such as
/// `properties[2].type`), with the violation in `BadRequest`.
pub fn invalid_field(field: impl Into<String>, description: impl Into<String>) -> Status {
    let field = field.into();
    let description = description.into();
    let metadata = std::collections::HashMap::from([("field".to_string(), field.clone())]);
    let mut details =
        ErrorDetails::with_error_info(Reason::InvalidField.as_str(), ERROR_DOMAIN, metadata);
    details.add_bad_request_violation(field.clone(), description.clone());
    Status::with_error_details(
        Code::InvalidArgument,
        format!("{field}: {description}"),
        details,
    )
}

/// A status naming catalog object `name` of kind `object`, as the subject of
/// `reason`, with the object also given as `ResourceInfo`.
pub fn catalog_object_status(
    code: Code,
    message: impl Into<String>,
    reason: Reason,
    object: &str,
    name: &str,
) -> Status {
    let metadata = std::collections::HashMap::from([
        ("object".to_string(), object.to_string()),
        ("name".to_string(), name.to_string()),
    ]);
    let mut details = ErrorDetails::with_error_info(reason.as_str(), ERROR_DOMAIN, metadata);
    details.set_resource_info(object, name, "", "");
    Status::with_error_details(code, message, details)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
