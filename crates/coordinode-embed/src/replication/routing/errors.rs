//! The peer encoding of the library's error contract: a [`DatabaseError`] the
//! executing member answers with, as the bytes of a `RoutedError` in the
//! answer's body, and back.
//!
//! Every case and field is mapped by hand: no error object is serialized
//! whole, and a case or value this member does not know is an unreadable
//! answer, never a refusal guessed at. The encoding is bounded before it is
//! decoded, and only free text is cut to fit; a field a caller decides on is
//! carried whole or the failure is reported as unreadable.

use coordinode_core::budget::BudgetStop;
use coordinode_core::graph::types::VectorMetric;
use coordinode_core::group::DeploymentId;
use coordinode_core::schema::computed::{ComputedSpec, DecayFormula, TtlScope};
use coordinode_core::schema::definition::{ConstraintKind, MAX_ARRAY_NESTING, PropertyType};
use coordinode_core::txn::command::TransactionId;
use coordinode_core::txn::operation::IssuerId;
use coordinode_core::version::{Mismatch, VersionPair};
use coordinode_query::cypher::ParseError;
use coordinode_query::cypher::errors::ErrorSpan;
use coordinode_query::executor::eval::EvalError;
use coordinode_query::executor::runner::CatalogObject;
use coordinode_query::procedure::ProcedureError;
use coordinode_raft::proto::internode::routing as wire;
use prost::Message;

use crate::db::{DatabaseError, ReadFenceFailure, StorageFailure, TransactionEnd};
use crate::operation::{OperationId, OperationOutcome};
use wire::routed_error::Kind;

/// The version of this encoding. A member refuses a call whose ingress reads
/// another one, before reading the call.
pub const ERROR_CODEC_VERSION: u32 = 2;

/// The largest encoded error a member sends or reads.
pub const MAX_ROUTED_ERROR: usize = 64 << 10;

/// The most of one free-text field an encoded error carries.
const TEXT_LIMIT: usize = 8 << 10;

/// `err` as the bytes of a `RoutedError`, within [`MAX_ROUTED_ERROR`].
///
/// Free text is cut at [`TEXT_LIMIT`]; every other field is carried whole.
/// The size is taken from the moved fields before anything is serialized or
/// rendered; an error whose fields do not fit is sent as undeliverable, which
/// the receiver reads as an answer it could not read, never as another
/// refusal.
pub fn encode(err: DatabaseError) -> Vec<u8> {
    let variant = variant_name(&err);
    let error = to_wire(err);
    let size = error.encoded_len();
    if size <= MAX_ROUTED_ERROR {
        return error.encode_to_vec();
    }
    undeliverable(variant, size).encode_to_vec()
}

fn undeliverable(variant: &str, size: usize) -> wire::RoutedError {
    wire::RoutedError {
        kind: Some(Kind::Undeliverable(wire::Undeliverable {
            variant: variant.to_owned(),
            // usize is at most 64 bits on every supported target.
            size: size as u64,
        })),
    }
}

/// The name of `err`'s variant, for the receiver of an undeliverable error.
fn variant_name(err: &DatabaseError) -> &'static str {
    use DatabaseError as E;
    match err {
        E::Parse(_) => "Parse",
        E::Plan(_) => "Plan",
        E::Semantic(_) => "Semantic",
        E::SessionSetting { .. } => "SessionSetting",
        E::Arithmetic(_) => "Arithmetic",
        E::Procedure(_) => "Procedure",
        E::Unsupported(_) => "Unsupported",
        E::Budget(_) => "Budget",
        E::IndexUnresolved { .. } => "IndexUnresolved",
        E::CascadeOverflow { .. } => "CascadeOverflow",
        E::CascadeFanoutOverflow { .. } => "CascadeFanoutOverflow",
        E::SchemaViolation(_) => "SchemaViolation",
        E::DuplicateKey { .. } => "DuplicateKey",
        E::UniqueViolation { .. } => "UniqueViolation",
        E::UniquenessUnresolved { .. } => "UniquenessUnresolved",
        E::ConstraintViolation { .. } => "ConstraintViolation",
        E::KeyImmutable { .. } => "KeyImmutable",
        E::CatalogObjectExists { .. } => "CatalogObjectExists",
        E::CatalogObjectDiffers { .. } => "CatalogObjectDiffers",
        E::CatalogRevisionMismatch { .. } => "CatalogRevisionMismatch",
        E::CatalogObjectMissing { .. } => "CatalogObjectMissing",
        E::CatalogRefused(_) => "CatalogRefused",
        E::UnknownTransaction(_) => "UnknownTransaction",
        E::TransactionIdInUse(_) => "TransactionIdInUse",
        E::TransactionBusy(_) => "TransactionBusy",
        E::OperationRequired(_) => "OperationRequired",
        E::OperationDiffers(_) => "OperationDiffers",
        E::TransactionCommittedOnTimeout(_) => "TransactionCommittedOnTimeout",
        E::TransactionOutcomeUnknown(_) => "TransactionOutcomeUnknown",
        E::TransactionAborted { .. } => "TransactionAborted",
        E::TransactionConflict { .. } => "TransactionConflict",
        E::InvariantRefused { .. } => "InvariantRefused",
        E::RevisionMismatch { .. } => "RevisionMismatch",
        E::TransactionTooLarge { .. } => "TransactionTooLarge",
        E::WriteBackpressure => "WriteBackpressure",
        E::CapacityExhausted { .. } => "CapacityExhausted",
        E::StorageFull { .. } => "StorageFull",
        E::NotLeader { .. } => "NotLeader",
        E::Mismatched(_) => "Mismatched",
        E::SettingsRefused(_) => "SettingsRefused",
        E::ReadFence { .. } => "ReadFence",
        E::OutsideRetention { .. } => "OutsideRetention",
        E::ReadNotReady { .. } => "ReadNotReady",
        E::OperationSettled { .. } => "OperationSettled",
        E::ForeignOperation { .. } => "ForeignOperation",
        E::Storage(_) => "Storage",
        E::Other(_) => "Other",
        E::PeerUnreachable { .. } => "PeerUnreachable",
        E::PeerOutcomeUnknown { .. } => "PeerOutcomeUnknown",
    }
}

/// The error `bytes` encode.
///
/// # Errors
///
/// What makes the bytes unreadable: too long, not a `RoutedError`, a case or
/// value this member does not know, or a malformed field.
pub fn decode(bytes: &[u8]) -> Result<DatabaseError, String> {
    if bytes.len() > MAX_ROUTED_ERROR {
        return Err(format!(
            "an error of {} bytes, above the {MAX_ROUTED_ERROR}-byte bound",
            bytes.len()
        ));
    }
    let error = wire::RoutedError::decode(bytes)
        .map_err(|e| format!("an error that does not decode: {e}"))?;
    from_wire(error.kind.ok_or("an error of no case")?)
}

/// `text` cut to [`TEXT_LIMIT`] bytes at a character boundary.
fn cut(mut text: String) -> String {
    if text.len() > TEXT_LIMIT {
        let mut end = TEXT_LIMIT;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        text.truncate(end);
    }
    text
}

fn text(message: String) -> wire::Text {
    wire::Text {
        message: cut(message),
    }
}

fn transaction(id: TransactionId) -> wire::TransactionRef {
    wire::TransactionRef {
        transaction_id: id.to_bytes().to_vec(),
    }
}

fn optional_transaction(id: Option<TransactionId>) -> Vec<u8> {
    id.map(|id| id.to_bytes().to_vec()).unwrap_or_default()
}

fn catalog_kind(object: CatalogObject) -> i32 {
    use wire::CatalogObjectKind as K;
    match object {
        CatalogObject::Label => K::Label,
        CatalogObject::EdgeType => K::EdgeType,
        CatalogObject::Constraint => K::Constraint,
        CatalogObject::Index => K::Index,
        CatalogObject::Trigger => K::Trigger,
    }
    .into()
}

fn end_reason(end: TransactionEnd) -> i32 {
    use wire::TransactionEndReason as R;
    match end {
        TransactionEnd::StatementFailed => R::StatementFailed,
        TransactionEnd::IdleExpired => R::IdleExpired,
        TransactionEnd::IdleCommitted => R::IdleCommitted,
        TransactionEnd::IdleCommitRefused => R::IdleCommitRefused,
        TransactionEnd::IdleCommitUnknown => R::IdleCommitUnknown,
        TransactionEnd::RolledBack => R::RolledBack,
        TransactionEnd::StreamLost => R::StreamLost,
    }
    .into()
}

/// An operation's outcome as the routing answer to a resolution states it.
pub(super) fn outcome_to_wire(outcome: OperationOutcome) -> wire::RoutedOutcome {
    let (state, attempt, commit_ts) = match outcome {
        OperationOutcome::Committed { attempt, commit_ts } => {
            (wire::RoutedOperationState::Committed, attempt, commit_ts)
        }
        OperationOutcome::Aborted => (wire::RoutedOperationState::Aborted, 0, 0),
        OperationOutcome::Unknown => (wire::RoutedOperationState::Unknown, 0, 0),
        OperationOutcome::Retired => (wire::RoutedOperationState::Retired, 0, 0),
    };
    wire::RoutedOutcome {
        state: state.into(),
        attempt,
        commit_ts,
    }
}

/// The outcome a routing answer states.
///
/// # Errors
///
/// A state this member does not know.
pub(super) fn outcome_from_wire(outcome: &wire::RoutedOutcome) -> Result<OperationOutcome, String> {
    Ok(match wire::RoutedOperationState::try_from(outcome.state) {
        Ok(wire::RoutedOperationState::Committed) => OperationOutcome::Committed {
            attempt: outcome.attempt,
            commit_ts: outcome.commit_ts,
        },
        Ok(wire::RoutedOperationState::Aborted) => OperationOutcome::Aborted,
        Ok(wire::RoutedOperationState::Unknown) => OperationOutcome::Unknown,
        Ok(wire::RoutedOperationState::Retired) => OperationOutcome::Retired,
        _ => {
            return Err(format!(
                "an operation state {} this member does not know",
                outcome.state
            ));
        }
    })
}

fn procedure_to_wire(refusal: ProcedureError) -> wire::ProcedureRefused {
    use wire::ProcedureRefusal as K;
    // usize is at most 64 bits on every supported target.
    let count = |n: usize| n as u64;
    let mut out = wire::ProcedureRefused::default();
    let kind = match refusal {
        ProcedureError::Unknown { procedure } => {
            out.procedure = procedure;
            K::Unknown
        }
        ProcedureError::TooManyArguments {
            procedure,
            declared,
            given,
        } => {
            out.procedure = procedure;
            out.declared = count(declared);
            out.given = count(given);
            K::TooManyArguments
        }
        ProcedureError::MissingArgument {
            procedure,
            argument,
        } => {
            out.procedure = procedure;
            out.argument = argument;
            K::MissingArgument
        }
        ProcedureError::ArgumentType {
            procedure,
            argument,
            expected,
            found,
        } => {
            out.procedure = procedure;
            out.argument = argument;
            out.expected = expected;
            out.found = found;
            K::ArgumentType
        }
        ProcedureError::InvalidArgument {
            procedure,
            argument,
            reason,
        } => {
            out.procedure = procedure;
            out.argument = argument;
            out.reason = cut(reason);
            K::InvalidArgument
        }
        ProcedureError::UnknownOutput { procedure, column } => {
            out.procedure = procedure;
            out.column = column;
            K::UnknownOutput
        }
        ProcedureError::YieldRequired { procedure } => {
            out.procedure = procedure;
            K::YieldRequired
        }
        ProcedureError::OutputShape {
            procedure,
            declared,
            found,
        } => {
            out.procedure = procedure;
            out.declared = count(declared);
            out.given = count(found);
            K::OutputShape
        }
        ProcedureError::DuplicateName { procedure } => {
            out.procedure = procedure;
            K::DuplicateName
        }
    };
    out.kind = kind.into();
    out
}

fn to_wire(err: DatabaseError) -> wire::RoutedError {
    use DatabaseError as E;
    let kind = match err {
        E::Parse(ParseError::Syntax { message, span }) => Kind::Parse(wire::ParseFailure {
            failure: Some(wire::parse_failure::Failure::Syntax(wire::SyntaxError {
                message: cut(message),
                start: span.start as u64,
                end: span.end as u64,
                line: span.line as u64,
                column: span.col as u64,
            })),
        }),
        E::Parse(ParseError::Invalid(message)) => Kind::Parse(wire::ParseFailure {
            failure: Some(wire::parse_failure::Failure::Invalid(cut(message))),
        }),
        E::Plan(message) => Kind::Plan(text(message)),
        E::Semantic(message) => Kind::Semantic(text(message)),
        E::SessionSetting { setting, reason } => Kind::SessionSetting(wire::SettingRefused {
            setting,
            reason: cut(reason),
        }),
        E::Arithmetic(eval) => Kind::Arithmetic(wire::ArithmeticFailure {
            failure: Some(match eval {
                EvalError::UnknownFunction(name) => {
                    wire::arithmetic_failure::Failure::UnknownFunction(name)
                }
                EvalError::DivideByZero => {
                    wire::arithmetic_failure::Failure::DivideByZero(wire::Empty {})
                }
                EvalError::LongOverflow => {
                    wire::arithmetic_failure::Failure::LongOverflow(wire::Empty {})
                }
            }),
        }),
        E::Procedure(refusal) => Kind::Procedure(procedure_to_wire(refusal)),
        E::Unsupported(message) => Kind::Unsupported(text(message)),
        E::Budget(stop) => Kind::Budget(wire::BudgetSpent {
            stop: Some(match stop {
                BudgetStop::Memory {
                    requested,
                    used,
                    limit,
                } => wire::budget_spent::Stop::Memory(wire::MemoryBudget {
                    requested,
                    used,
                    limit,
                }),
                BudgetStop::Deadline => wire::budget_spent::Stop::Deadline(wire::Empty {}),
                BudgetStop::Cancelled => wire::budget_spent::Stop::Cancelled(wire::Empty {}),
            }),
        }),
        E::IndexUnresolved {
            label,
            property,
            operator,
        } => Kind::IndexUnresolved(wire::IndexUnresolved {
            label,
            property,
            operator,
        }),
        E::CascadeOverflow {
            current,
            limit,
            chain,
        } => Kind::CascadeOverflow(wire::CascadeOverflow {
            current,
            limit,
            chain,
        }),
        E::CascadeFanoutOverflow {
            trigger,
            count,
            limit,
        } => Kind::CascadeFanoutOverflow(wire::CascadeFanoutOverflow {
            trigger,
            count,
            limit,
        }),
        E::SchemaViolation(message) => Kind::SchemaViolation(text(message)),
        E::DuplicateKey {
            table,
            key,
            element_id,
        } => Kind::DuplicateKey(wire::DuplicateKey {
            table,
            key,
            element_id,
        }),
        E::UniqueViolation {
            index,
            property,
            value,
            element_id,
        } => Kind::UniqueViolation(wire::UniqueViolation {
            index,
            property,
            value,
            element_id,
        }),
        E::UniquenessUnresolved { index, limit } => {
            Kind::UniquenessUnresolved(wire::UniquenessUnresolved { index, limit })
        }
        E::ConstraintViolation {
            constraint,
            kind,
            label,
            property,
            element_id,
        } => match constraint_kind_to_wire(&kind) {
            Some(kind) => Kind::ConstraintViolation(wire::ConstraintViolation {
                constraint,
                kind: Some(kind),
                label,
                property,
                element_id,
            }),
            // A type nested past the engine's bound is not carried as some
            // other error; the receiver reads it as unreadable.
            None => return undeliverable("ConstraintViolation", 0),
        },
        E::KeyImmutable { table, column } => {
            Kind::KeyImmutable(wire::KeyImmutable { table, column })
        }
        E::CatalogObjectExists { object, name } => Kind::CatalogObjectExists(wire::CatalogName {
            object: catalog_kind(object),
            name,
        }),
        E::CatalogObjectDiffers {
            object,
            name,
            revision,
        } => Kind::CatalogObjectDiffers(wire::CatalogObjectDiffers {
            object: catalog_kind(object),
            name,
            revision,
        }),
        E::CatalogRevisionMismatch {
            object,
            name,
            expected,
            current,
        } => Kind::CatalogRevisionMismatch(wire::CatalogRevisionMismatch {
            object: catalog_kind(object),
            name,
            expected,
            current,
        }),
        E::CatalogObjectMissing { object, name } => Kind::CatalogObjectMissing(wire::CatalogName {
            object: catalog_kind(object),
            name,
        }),
        E::CatalogRefused(message) => Kind::CatalogRefused(text(message)),
        E::UnknownTransaction(id) => Kind::UnknownTransaction(transaction(id)),
        E::TransactionIdInUse(id) => Kind::TransactionIdInUse(transaction(id)),
        E::TransactionBusy(id) => Kind::TransactionBusy(transaction(id)),
        E::OperationRequired(id) => Kind::OperationRequired(transaction(id)),
        E::OperationDiffers(id) => Kind::OperationDiffers(transaction(id)),
        E::TransactionCommittedOnTimeout(id) => Kind::CommittedOnTimeout(transaction(id)),
        E::TransactionOutcomeUnknown(id) => Kind::OutcomeUnknown(transaction(id)),
        E::TransactionAborted { id, end } => Kind::TransactionAborted(wire::TransactionAborted {
            transaction_id: id.to_bytes().to_vec(),
            end: end_reason(end),
        }),
        E::TransactionConflict {
            transaction,
            message,
        } => Kind::TransactionConflict(wire::TransactionConflict {
            transaction_id: optional_transaction(transaction),
            message: cut(message),
        }),
        E::InvariantRefused {
            transaction,
            reason,
        } => Kind::InvariantRefused(wire::InvariantRefused {
            transaction_id: optional_transaction(transaction),
            reason: cut(reason),
        }),
        E::RevisionMismatch {
            transaction,
            expected,
            current,
        } => Kind::RevisionMismatch(wire::RevisionMismatch {
            transaction_id: optional_transaction(transaction),
            expected,
            current,
        }),
        E::TransactionTooLarge {
            id,
            buffered,
            limit,
        } => Kind::TransactionTooLarge(wire::TransactionTooLarge {
            transaction_id: id.to_bytes().to_vec(),
            buffered,
            limit,
        }),
        E::WriteBackpressure => Kind::WriteBackpressure(wire::Empty {}),
        E::CapacityExhausted {
            endpoint_id,
            used_bytes,
            hard_limit_bytes,
        } => Kind::CapacityExhausted(wire::CapacityExhausted {
            endpoint_id,
            used_bytes,
            hard_limit_bytes,
        }),
        E::StorageFull {
            path,
            available_bytes,
            min_free_bytes,
        } => Kind::StorageFull(wire::StorageFull {
            path,
            available_bytes,
            min_free_bytes,
        }),
        E::NotLeader { leader_id } => Kind::NotLeader(wire::NotLeader { leader_id }),
        E::Mismatched(m) => {
            let (leader_id, leader_address) = match m.leader {
                Some((id, address)) => (Some(id), address),
                None => (None, String::new()),
            };
            Kind::MemberReadOnly(wire::MemberReadOnly {
                own_engine: m.own.engine,
                own_host_epoch: m.own.host_epoch,
                group_engine: m.group.engine,
                group_host_epoch: m.group.host_epoch,
                behind: m.behind,
                leader_id,
                leader_address,
                as_of: m.as_of,
            })
        }
        E::SettingsRefused(message) => Kind::SettingsRefused(text(message)),
        E::ReadFence { kind, message } => Kind::ReadFence(wire::ReadFence {
            kind: match kind {
                ReadFenceFailure::Role => wire::ReadFenceKind::Role,
                ReadFenceFailure::Stale => wire::ReadFenceKind::Stale,
                ReadFenceFailure::Timeout => wire::ReadFenceKind::Timeout,
            }
            .into(),
            message: cut(message),
        }),
        E::OutsideRetention {
            requested,
            oldest_readable,
        } => Kind::OutsideRetention(wire::OutsideRetention {
            requested,
            oldest_readable,
        }),
        E::ReadNotReady {
            requested,
            ready_through,
        } => Kind::ReadNotReady(wire::ReadNotReady {
            requested,
            ready_through,
        }),
        E::OperationSettled {
            op,
            outcome,
            intent_differs,
        } => Kind::OperationSettled(wire::OperationSettled {
            issuer: op.issuer.0.to_vec(),
            incarnation: op.incarnation,
            sequence: op.sequence,
            outcome: Some(outcome_to_wire(outcome)),
            intent_differs,
        }),
        E::ForeignOperation { descriptor } => Kind::ForeignOperation(wire::ForeignOperation {
            deployment: descriptor.as_bytes().to_vec(),
        }),
        E::Storage(failure) => Kind::Storage(text(failure.message().to_owned())),
        E::Other(message) => Kind::Other(text(message)),
        // An ingress's own account of a member it reached; an executing
        // member routes nothing on, so these reach the wire only as text.
        e @ (E::PeerUnreachable { .. } | E::PeerOutcomeUnknown { .. }) => {
            Kind::Other(text(e.to_string()))
        }
    };
    wire::RoutedError { kind: Some(kind) }
}

fn transaction_from(field: &str, bytes: &[u8]) -> Result<TransactionId, String> {
    TransactionId::from_bytes(bytes).map_err(|e| format!("{field}: {e}"))
}

fn optional_transaction_from(field: &str, bytes: &[u8]) -> Result<Option<TransactionId>, String> {
    if bytes.is_empty() {
        return Ok(None);
    }
    transaction_from(field, bytes).map(Some)
}

fn catalog_object(object: i32) -> Result<CatalogObject, String> {
    use wire::CatalogObjectKind as K;
    Ok(match K::try_from(object) {
        Ok(K::Label) => CatalogObject::Label,
        Ok(K::EdgeType) => CatalogObject::EdgeType,
        Ok(K::Constraint) => CatalogObject::Constraint,
        Ok(K::Index) => CatalogObject::Index,
        Ok(K::Trigger) => CatalogObject::Trigger,
        _ => {
            return Err(format!(
                "a catalog object kind {object} this member does not know"
            ));
        }
    })
}

fn transaction_end(end: i32) -> Result<TransactionEnd, String> {
    use wire::TransactionEndReason as R;
    Ok(match R::try_from(end) {
        Ok(R::StatementFailed) => TransactionEnd::StatementFailed,
        Ok(R::IdleExpired) => TransactionEnd::IdleExpired,
        Ok(R::IdleCommitted) => TransactionEnd::IdleCommitted,
        Ok(R::IdleCommitRefused) => TransactionEnd::IdleCommitRefused,
        Ok(R::IdleCommitUnknown) => TransactionEnd::IdleCommitUnknown,
        Ok(R::RolledBack) => TransactionEnd::RolledBack,
        Ok(R::StreamLost) => TransactionEnd::StreamLost,
        _ => return Err(format!("a transaction end {end} this member does not know")),
    })
}

fn usize_from(field: &str, n: u64) -> Result<usize, String> {
    usize::try_from(n).map_err(|_| format!("{field}: {n} does not fit this member"))
}

fn procedure_from_wire(p: wire::ProcedureRefused) -> Result<ProcedureError, String> {
    use wire::ProcedureRefusal as K;
    let procedure = p.procedure;
    Ok(match K::try_from(p.kind) {
        Ok(K::Unknown) => ProcedureError::Unknown { procedure },
        Ok(K::TooManyArguments) => ProcedureError::TooManyArguments {
            procedure,
            declared: usize_from("declared", p.declared)?,
            given: usize_from("given", p.given)?,
        },
        Ok(K::MissingArgument) => ProcedureError::MissingArgument {
            procedure,
            argument: p.argument,
        },
        Ok(K::ArgumentType) => ProcedureError::ArgumentType {
            procedure,
            argument: p.argument,
            expected: p.expected,
            found: p.found,
        },
        Ok(K::InvalidArgument) => ProcedureError::InvalidArgument {
            procedure,
            argument: p.argument,
            reason: p.reason,
        },
        Ok(K::UnknownOutput) => ProcedureError::UnknownOutput {
            procedure,
            column: p.column,
        },
        Ok(K::YieldRequired) => ProcedureError::YieldRequired { procedure },
        Ok(K::OutputShape) => ProcedureError::OutputShape {
            procedure,
            declared: usize_from("declared", p.declared)?,
            found: usize_from("given", p.given)?,
        },
        Ok(K::DuplicateName) => ProcedureError::DuplicateName { procedure },
        _ => {
            return Err(format!(
                "a procedure refusal {} this member does not know",
                p.kind
            ));
        }
    })
}

fn from_wire(kind: Kind) -> Result<DatabaseError, String> {
    use DatabaseError as E;
    Ok(match kind {
        Kind::Parse(parse) => {
            E::Parse(match parse.failure.ok_or("a parse failure of no kind")? {
                wire::parse_failure::Failure::Syntax(s) => ParseError::Syntax {
                    message: s.message,
                    span: ErrorSpan {
                        start: usize_from("start", s.start)?,
                        end: usize_from("end", s.end)?,
                        line: usize_from("line", s.line)?,
                        col: usize_from("column", s.column)?,
                    },
                },
                wire::parse_failure::Failure::Invalid(message) => ParseError::Invalid(message),
            })
        }
        Kind::Plan(t) => E::Plan(t.message),
        Kind::Semantic(t) => E::Semantic(t.message),
        Kind::SessionSetting(s) => E::SessionSetting {
            setting: s.setting,
            reason: s.reason,
        },
        Kind::Arithmetic(a) => {
            E::Arithmetic(match a.failure.ok_or("an arithmetic failure of no kind")? {
                wire::arithmetic_failure::Failure::UnknownFunction(name) => {
                    EvalError::UnknownFunction(name)
                }
                wire::arithmetic_failure::Failure::DivideByZero(_) => EvalError::DivideByZero,
                wire::arithmetic_failure::Failure::LongOverflow(_) => EvalError::LongOverflow,
            })
        }
        Kind::Procedure(p) => E::Procedure(procedure_from_wire(p)?),
        Kind::Unsupported(t) => E::Unsupported(t.message),
        Kind::Budget(b) => E::Budget(match b.stop.ok_or("a spent budget of no kind")? {
            wire::budget_spent::Stop::Memory(m) => BudgetStop::Memory {
                requested: m.requested,
                used: m.used,
                limit: m.limit,
            },
            wire::budget_spent::Stop::Deadline(_) => BudgetStop::Deadline,
            wire::budget_spent::Stop::Cancelled(_) => BudgetStop::Cancelled,
        }),
        Kind::IndexUnresolved(i) => E::IndexUnresolved {
            label: i.label,
            property: i.property,
            operator: i.operator,
        },
        Kind::CascadeOverflow(c) => E::CascadeOverflow {
            current: c.current,
            limit: c.limit,
            chain: c.chain,
        },
        Kind::CascadeFanoutOverflow(c) => E::CascadeFanoutOverflow {
            trigger: c.trigger,
            count: c.count,
            limit: c.limit,
        },
        Kind::SchemaViolation(t) => E::SchemaViolation(t.message),
        Kind::DuplicateKey(d) => E::DuplicateKey {
            table: d.table,
            key: d.key,
            element_id: d.element_id,
        },
        Kind::UniqueViolation(u) => E::UniqueViolation {
            index: u.index,
            property: u.property,
            value: u.value,
            element_id: u.element_id,
        },
        Kind::UniquenessUnresolved(u) => E::UniquenessUnresolved {
            index: u.index,
            limit: u.limit,
        },
        Kind::ConstraintViolation(c) => E::ConstraintViolation {
            kind: Box::new(constraint_kind_from_wire(c.kind)?),
            constraint: c.constraint,
            label: c.label,
            property: c.property,
            element_id: c.element_id,
        },
        Kind::KeyImmutable(k) => E::KeyImmutable {
            table: k.table,
            column: k.column,
        },
        Kind::CatalogObjectExists(c) => E::CatalogObjectExists {
            object: catalog_object(c.object)?,
            name: c.name,
        },
        Kind::CatalogObjectDiffers(c) => E::CatalogObjectDiffers {
            object: catalog_object(c.object)?,
            name: c.name,
            revision: c.revision,
        },
        Kind::CatalogRevisionMismatch(c) => E::CatalogRevisionMismatch {
            object: catalog_object(c.object)?,
            name: c.name,
            expected: c.expected,
            current: c.current,
        },
        Kind::CatalogObjectMissing(c) => E::CatalogObjectMissing {
            object: catalog_object(c.object)?,
            name: c.name,
        },
        Kind::CatalogRefused(t) => E::CatalogRefused(t.message),
        Kind::UnknownTransaction(t) => {
            E::UnknownTransaction(transaction_from("unknown_transaction", &t.transaction_id)?)
        }
        Kind::TransactionIdInUse(t) => E::TransactionIdInUse(transaction_from(
            "transaction_id_in_use",
            &t.transaction_id,
        )?),
        Kind::TransactionBusy(t) => {
            E::TransactionBusy(transaction_from("transaction_busy", &t.transaction_id)?)
        }
        Kind::OperationRequired(t) => {
            E::OperationRequired(transaction_from("operation_required", &t.transaction_id)?)
        }
        Kind::OperationDiffers(t) => {
            E::OperationDiffers(transaction_from("operation_differs", &t.transaction_id)?)
        }
        Kind::CommittedOnTimeout(t) => E::TransactionCommittedOnTimeout(transaction_from(
            "committed_on_timeout",
            &t.transaction_id,
        )?),
        Kind::OutcomeUnknown(t) => {
            E::TransactionOutcomeUnknown(transaction_from("outcome_unknown", &t.transaction_id)?)
        }
        Kind::TransactionAborted(t) => E::TransactionAborted {
            id: transaction_from("transaction_aborted", &t.transaction_id)?,
            end: transaction_end(t.end)?,
        },
        Kind::TransactionConflict(t) => E::TransactionConflict {
            transaction: optional_transaction_from("transaction_conflict", &t.transaction_id)?,
            message: t.message,
        },
        Kind::InvariantRefused(t) => E::InvariantRefused {
            transaction: optional_transaction_from("invariant_refused", &t.transaction_id)?,
            reason: t.reason,
        },
        Kind::RevisionMismatch(t) => E::RevisionMismatch {
            transaction: optional_transaction_from("revision_mismatch", &t.transaction_id)?,
            expected: t.expected,
            current: t.current,
        },
        Kind::TransactionTooLarge(t) => E::TransactionTooLarge {
            id: transaction_from("transaction_too_large", &t.transaction_id)?,
            buffered: t.buffered,
            limit: t.limit,
        },
        Kind::WriteBackpressure(_) => E::WriteBackpressure,
        Kind::CapacityExhausted(c) => E::CapacityExhausted {
            endpoint_id: c.endpoint_id,
            used_bytes: c.used_bytes,
            hard_limit_bytes: c.hard_limit_bytes,
        },
        Kind::StorageFull(s) => E::StorageFull {
            path: s.path,
            available_bytes: s.available_bytes,
            min_free_bytes: s.min_free_bytes,
        },
        Kind::NotLeader(n) => E::NotLeader {
            leader_id: n.leader_id,
        },
        Kind::MemberReadOnly(m) => E::Mismatched(Mismatch {
            own: VersionPair {
                engine: m.own_engine,
                host_epoch: m.own_host_epoch,
            },
            group: VersionPair {
                engine: m.group_engine,
                host_epoch: m.group_host_epoch,
            },
            behind: m.behind,
            leader: m.leader_id.map(|id| (id, m.leader_address)),
            as_of: m.as_of,
        }),
        Kind::SettingsRefused(t) => E::SettingsRefused(t.message),
        Kind::ReadFence(f) => E::ReadFence {
            kind: match wire::ReadFenceKind::try_from(f.kind) {
                Ok(wire::ReadFenceKind::Role) => ReadFenceFailure::Role,
                Ok(wire::ReadFenceKind::Stale) => ReadFenceFailure::Stale,
                Ok(wire::ReadFenceKind::Timeout) => ReadFenceFailure::Timeout,
                _ => {
                    return Err(format!(
                        "a read fence failure {} this member does not know",
                        f.kind
                    ));
                }
            },
            message: f.message,
        },
        Kind::OutsideRetention(o) => E::OutsideRetention {
            requested: o.requested,
            oldest_readable: o.oldest_readable,
        },
        Kind::ReadNotReady(r) => E::ReadNotReady {
            requested: r.requested,
            ready_through: r.ready_through,
        },
        Kind::OperationSettled(o) => E::OperationSettled {
            op: OperationId {
                issuer: IssuerId(
                    <[u8; 16]>::try_from(o.issuer.as_slice())
                        .map_err(|_| "operation_settled.issuer: an issuer is 16 bytes")?,
                ),
                incarnation: o.incarnation,
                sequence: o.sequence,
            },
            outcome: outcome_from_wire(&o.outcome.ok_or("operation_settled: no outcome")?)?,
            intent_differs: o.intent_differs,
        },
        Kind::ForeignOperation(f) => E::ForeignOperation {
            descriptor: DeploymentId::from_slice(&f.deployment)
                .ok_or("foreign_operation.deployment: a deployment is 16 bytes")?,
        },
        Kind::Storage(t) => E::Storage(StorageFailure::reported(t.message)),
        Kind::Other(t) => E::Other(t.message),
        Kind::Undeliverable(u) => {
            return Err(format!(
                "the member's {} error does not fit the {MAX_ROUTED_ERROR}-byte bound \
                 ({} bytes)",
                u.variant, u.size
            ));
        }
    })
}

/// `kind` with its type flattened to a depth and a base; `None` for a type
/// nested past [`MAX_ARRAY_NESTING`], which the engine does not admit.
fn constraint_kind_to_wire(kind: &ConstraintKind) -> Option<wire::RoutedConstraintKind> {
    use wire::routed_constraint_kind::Kind as K;
    let nothing = wire::RoutedNothing {};
    Some(wire::RoutedConstraintKind {
        kind: Some(match kind {
            ConstraintKind::Unique => K::Unique(nothing),
            ConstraintKind::NotNull => K::NotNull(nothing),
            ConstraintKind::NodeKey => K::NodeKey(nothing),
            ConstraintKind::Type(t) => K::Type(property_type_to_wire(t)?),
        }),
    })
}

fn property_type_to_wire(t: &PropertyType) -> Option<wire::RoutedPropertyType> {
    use wire::RoutedScalarType as S;
    use wire::routed_property_type::Base;
    let (depth, base) = t.array_nesting();
    if depth > MAX_ARRAY_NESTING {
        return None;
    }
    let scalar = |s: S| Base::Scalar(s as i32);
    let base = match base {
        PropertyType::String => scalar(S::String),
        PropertyType::Int => scalar(S::Int),
        PropertyType::Float => scalar(S::Float),
        PropertyType::Bool => scalar(S::Bool),
        PropertyType::Timestamp => scalar(S::Timestamp),
        PropertyType::Blob => scalar(S::Blob),
        PropertyType::Map => scalar(S::Map),
        PropertyType::Geo => scalar(S::Geo),
        PropertyType::Binary => scalar(S::Binary),
        PropertyType::Document => scalar(S::Document),
        PropertyType::Vector { dimensions, metric } => Base::Vector(wire::RoutedVectorType {
            dimensions: *dimensions,
            metric: vector_metric_to_wire(*metric) as i32,
        }),
        PropertyType::Computed(spec) => Base::Computed(computed_to_wire(spec)),
        // `array_nesting` returns the first type that is not an array.
        PropertyType::Array(_) => return None,
    };
    Some(wire::RoutedPropertyType {
        // At most MAX_ARRAY_NESTING, checked above.
        array_depth: u32::try_from(depth).ok()?,
        base: Some(base),
    })
}

fn vector_metric_to_wire(metric: VectorMetric) -> wire::RoutedVectorMetric {
    use wire::RoutedVectorMetric as M;
    match metric {
        VectorMetric::Cosine => M::Cosine,
        VectorMetric::L2 => M::L2,
        VectorMetric::DotProduct => M::DotProduct,
        VectorMetric::L1 => M::L1,
    }
}

fn computed_to_wire(spec: &ComputedSpec) -> wire::RoutedComputedSpec {
    use wire::routed_computed_spec::Spec;
    wire::RoutedComputedSpec {
        spec: Some(match spec {
            ComputedSpec::Decay {
                formula,
                initial,
                target,
                duration_secs,
                anchor_field,
            } => Spec::Decay(wire::RoutedDecay {
                formula: Some(decay_formula_to_wire(formula)),
                initial: *initial,
                target: *target,
                duration_secs: *duration_secs,
                anchor_field: anchor_field.clone(),
            }),
            ComputedSpec::Ttl {
                duration_secs,
                anchor_field,
                scope,
                target_field,
            } => Spec::Ttl(wire::RoutedTtl {
                duration_secs: *duration_secs,
                anchor_field: anchor_field.clone(),
                scope: match scope {
                    TtlScope::Field => wire::RoutedTtlScope::Field,
                    TtlScope::Subtree => wire::RoutedTtlScope::Subtree,
                    TtlScope::Node => wire::RoutedTtlScope::Node,
                } as i32,
                target_field: target_field.clone(),
            }),
            ComputedSpec::VectorDecay {
                formula,
                duration_secs,
                anchor_field,
            } => Spec::VectorDecay(wire::RoutedVectorDecay {
                formula: Some(decay_formula_to_wire(formula)),
                duration_secs: *duration_secs,
                anchor_field: anchor_field.clone(),
            }),
        }),
    }
}

fn decay_formula_to_wire(formula: &DecayFormula) -> wire::RoutedDecayFormula {
    use wire::routed_decay_formula::Formula;
    wire::RoutedDecayFormula {
        formula: Some(match formula {
            DecayFormula::Linear => Formula::Linear(wire::RoutedNothing {}),
            DecayFormula::Exponential { lambda } => Formula::ExponentialLambda(*lambda),
            DecayFormula::PowerLaw { tau, alpha } => Formula::PowerLaw(wire::RoutedPowerLaw {
                tau: *tau,
                alpha: *alpha,
            }),
            DecayFormula::Step => Formula::Step(wire::RoutedNothing {}),
        }),
    }
}

/// The constraint kind `kind` carries. The nesting is checked by its count
/// before any type is built.
fn constraint_kind_from_wire(
    kind: Option<wire::RoutedConstraintKind>,
) -> Result<ConstraintKind, String> {
    use wire::routed_constraint_kind::Kind as K;
    match kind.and_then(|k| k.kind) {
        Some(K::Unique(_)) => Ok(ConstraintKind::Unique),
        Some(K::NotNull(_)) => Ok(ConstraintKind::NotNull),
        Some(K::NodeKey(_)) => Ok(ConstraintKind::NodeKey),
        Some(K::Type(t)) => Ok(ConstraintKind::Type(property_type_from_wire(t)?)),
        None => Err("a constraint violation names no kind".to_owned()),
    }
}

fn property_type_from_wire(t: wire::RoutedPropertyType) -> Result<PropertyType, String> {
    use wire::RoutedScalarType as S;
    use wire::routed_property_type::Base;
    let depth = usize::try_from(t.array_depth)
        .ok()
        .filter(|depth| *depth <= MAX_ARRAY_NESTING)
        .ok_or_else(|| {
            format!(
                "a property type nests {} arrays, past the bound of {MAX_ARRAY_NESTING}",
                t.array_depth
            )
        })?;
    let base = match t.base {
        Some(Base::Scalar(s)) => match S::try_from(s) {
            Ok(S::String) => PropertyType::String,
            Ok(S::Int) => PropertyType::Int,
            Ok(S::Float) => PropertyType::Float,
            Ok(S::Bool) => PropertyType::Bool,
            Ok(S::Timestamp) => PropertyType::Timestamp,
            Ok(S::Blob) => PropertyType::Blob,
            Ok(S::Map) => PropertyType::Map,
            Ok(S::Geo) => PropertyType::Geo,
            Ok(S::Binary) => PropertyType::Binary,
            Ok(S::Document) => PropertyType::Document,
            Ok(S::Unspecified) | Err(_) => {
                return Err(format!("scalar type {s} is not one this member knows"));
            }
        },
        Some(Base::Vector(v)) => PropertyType::Vector {
            dimensions: v.dimensions,
            metric: vector_metric_from_wire(v.metric)?,
        },
        Some(Base::Computed(c)) => PropertyType::Computed(computed_from_wire(c)?),
        None => return Err("a property type names no base type".to_owned()),
    };
    Ok(PropertyType::in_arrays(base, depth))
}

fn vector_metric_from_wire(metric: i32) -> Result<VectorMetric, String> {
    use wire::RoutedVectorMetric as M;
    match M::try_from(metric) {
        Ok(M::Cosine) => Ok(VectorMetric::Cosine),
        Ok(M::L2) => Ok(VectorMetric::L2),
        Ok(M::DotProduct) => Ok(VectorMetric::DotProduct),
        Ok(M::L1) => Ok(VectorMetric::L1),
        Ok(M::Unspecified) | Err(_) => Err(format!(
            "vector metric {metric} is not one this member knows"
        )),
    }
}

fn computed_from_wire(spec: wire::RoutedComputedSpec) -> Result<ComputedSpec, String> {
    use wire::routed_computed_spec::Spec;
    Ok(match spec.spec {
        Some(Spec::Decay(d)) => ComputedSpec::Decay {
            formula: decay_formula_from_wire(d.formula)?,
            initial: d.initial,
            target: d.target,
            duration_secs: d.duration_secs,
            anchor_field: d.anchor_field,
        },
        Some(Spec::Ttl(t)) => ComputedSpec::Ttl {
            duration_secs: t.duration_secs,
            anchor_field: t.anchor_field,
            scope: match wire::RoutedTtlScope::try_from(t.scope) {
                Ok(wire::RoutedTtlScope::Field) => TtlScope::Field,
                Ok(wire::RoutedTtlScope::Subtree) => TtlScope::Subtree,
                Ok(wire::RoutedTtlScope::Node) => TtlScope::Node,
                Ok(wire::RoutedTtlScope::Unspecified) | Err(_) => {
                    return Err(format!(
                        "TTL scope {} is not one this member knows",
                        t.scope
                    ));
                }
            },
            target_field: t.target_field,
        },
        Some(Spec::VectorDecay(v)) => ComputedSpec::VectorDecay {
            formula: decay_formula_from_wire(v.formula)?,
            duration_secs: v.duration_secs,
            anchor_field: v.anchor_field,
        },
        None => return Err("a computed type names no spec".to_owned()),
    })
}

fn decay_formula_from_wire(
    formula: Option<wire::RoutedDecayFormula>,
) -> Result<DecayFormula, String> {
    use wire::routed_decay_formula::Formula;
    match formula.and_then(|f| f.formula) {
        Some(Formula::Linear(_)) => Ok(DecayFormula::Linear),
        Some(Formula::ExponentialLambda(lambda)) => Ok(DecayFormula::Exponential { lambda }),
        Some(Formula::PowerLaw(p)) => Ok(DecayFormula::PowerLaw {
            tau: p.tau,
            alpha: p.alpha,
        }),
        Some(Formula::Step(_)) => Ok(DecayFormula::Step),
        None => Err("a decay names no formula".to_owned()),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
