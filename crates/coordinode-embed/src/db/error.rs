//! The library's error contract.
//!
//! A call made on this member and one routed to another fail with the same
//! variants and the same fields: everything a caller needs to decide whether
//! to retry, where, and what happened is a field of the variant, and survives
//! the hop between members. The engine's own error objects stop at this
//! boundary. Those whose cause a caller can act on are classified into a
//! variant here; the rest keep their source only on the member that raised
//! them, as diagnostics ([`StorageFailure::cause`]), never as the one place a
//! field lives.

use coordinode_core::budget::BudgetStop;
use coordinode_core::group::DeploymentId;
use coordinode_core::schema::definition::ConstraintKind;
use coordinode_core::txn::command::TransactionId;
use coordinode_core::version::Mismatch;
use coordinode_modality::StoreError;
use coordinode_query::cypher::ParseError;
use coordinode_query::executor::eval::EvalError;
use coordinode_query::executor::runner::{CatalogObject, ExecutionError};
use coordinode_query::planner::PlanError;
use coordinode_query::procedure::ProcedureError;
use coordinode_storage::error::StorageError;

use super::SessionSettingRefused;
use super::interactive::TransactionEnd;
use crate::operation::{OperationId, OperationOutcome};

/// Error from database operations, whether this member ran them or another
/// member did on its behalf.
#[derive(Debug, thiserror::Error)]
pub enum DatabaseError {
    /// The statement is not valid Cypher.
    #[error("parse error: {0}")]
    Parse(ParseError),

    /// The statement parsed but no plan answers it.
    #[error("plan error: {0}")]
    Plan(String),

    #[error("semantic error: {0}")]
    Semantic(String),

    /// A session SET named a known setting with a value it cannot take.
    #[error("{setting}: {reason}")]
    SessionSetting {
        /// The setting named.
        setting: String,
        /// Why the value was refused.
        reason: String,
    },

    /// Arithmetic with no answer, or a call of a function nothing provides.
    #[error("{0}")]
    Arithmetic(EvalError),

    /// A `CALL` refused before the procedure ran.
    #[error("{0}")]
    Procedure(ProcedureError),

    #[error("unsupported operation: {0}")]
    Unsupported(String),

    /// The statement spent its budget: its memory limit, its deadline, or it
    /// was cancelled. Nothing it produced so far is a complete answer.
    #[error("{0}")]
    Budget(BudgetStop),

    /// An index cannot answer, and reading the records instead would run
    /// `operator` outside the statement's budget. Nothing was read or
    /// written; the same statement succeeds once the index is verified or
    /// rebuilt.
    #[error(
        "the index on :{label}({property}) cannot answer, and reading the records instead \
         would run {operator} outside the statement's resource budget; retry once the index \
         is verified or rebuilt"
    )]
    IndexUnresolved {
        /// The indexed label.
        label: String,
        /// The indexed property.
        property: String,
        /// The operator that would have run unbounded.
        operator: String,
    },

    /// Triggers fired in a chain deeper than its limit. `chain` names them in
    /// firing order.
    #[error("trigger cascade depth exceeded: current={current}, limit={limit}, chain={chain:?}")]
    CascadeOverflow {
        /// The depth reached.
        current: u32,
        /// The limit.
        limit: u32,
        /// The triggers that fired.
        chain: Vec<String>,
    },

    /// One trigger fired more often within one cascade than it allows.
    #[error("trigger cascade fanout exceeded for `{trigger}`: count={count}, limit={limit}")]
    CascadeFanoutOverflow {
        /// The trigger.
        trigger: String,
        /// How often it fired.
        count: u32,
        /// The limit.
        limit: u32,
    },

    /// A STRICT label refused an undeclared property, or a write set a
    /// computed one.
    #[error("schema violation: {0}")]
    SchemaViolation(String),

    /// An insert named a key a row of the table already holds. Nothing was
    /// written.
    #[error(
        "table `{table}` already has a row with key {key} (row {element_id}): key already exists"
    )]
    DuplicateKey {
        /// The table.
        table: String,
        /// The key, rendered.
        key: String,
        /// The elementId of the row that holds the key.
        element_id: String,
    },

    /// A write would give a value of a unique index a second holder. Nothing
    /// was written.
    #[error(
        "unique constraint violated on index `{index}`: property `{property}` already has \
         value {value} (node {element_id})"
    )]
    UniqueViolation {
        /// The unique index.
        index: String,
        /// The indexed properties, comma-separated.
        property: String,
        /// The value, rendered.
        value: String,
        /// The elementId of the node that holds the value.
        element_id: String,
    },

    /// A unique index still being built cannot prove a value free within its
    /// read limit. Nothing was written; the same write succeeds once the
    /// build is done.
    #[error(
        "unique index `{index}` is still being built and proving the value free would read \
         more than {limit} stored nodes; retry once the build is done"
    )]
    UniquenessUnresolved {
        /// The unique index being built.
        index: String,
        /// The most stored node rows one proof reads.
        limit: u64,
    },

    /// A write would leave a node breaking a constraint of its label.
    /// Nothing was written.
    #[error(
        "constraint `{constraint}` ({kind}) violated on :{label}: property `{property}` of node \
         {element_id} {}",
        constraint_problem(kind)
    )]
    ConstraintViolation {
        /// The constraint.
        constraint: String,
        /// What it requires.
        kind: Box<ConstraintKind>,
        /// The constrained label.
        label: String,
        /// The property that breaks it.
        property: String,
        /// The elementId of the node.
        element_id: String,
    },

    /// A statement tried to change the key of an existing row.
    #[error(
        "column `{column}` is part of the key of table `{table}` and cannot be changed; \
         delete the row and insert a new one"
    )]
    KeyImmutable {
        /// The table.
        table: String,
        /// The key column the statement targeted.
        column: String,
    },

    /// A catalog object of the name already exists. Nothing changed.
    #[error("{object} '{name}' already exists")]
    CatalogObjectExists {
        /// What kind of object holds the name.
        object: CatalogObject,
        /// The name.
        name: String,
    },

    /// A catalog object of the name exists with another definition, and the
    /// declaration asked for no change. Nothing changed.
    #[error(
        "{object} '{name}' exists at revision {revision} with another definition; \
         changing it needs an explicit change naming that revision"
    )]
    CatalogObjectDiffers {
        /// What kind of object holds the name.
        object: CatalogObject,
        /// The name.
        name: String,
        /// The revision the object is at.
        revision: u64,
    },

    /// An explicit change named a revision the object is no longer at.
    /// Nothing changed.
    #[error(
        "{object} '{name}' is at revision {current:?}, not the expected {expected}; \
         read it again before changing it"
    )]
    CatalogRevisionMismatch {
        /// What kind of object the change targets.
        object: CatalogObject,
        /// The name.
        name: String,
        /// The revision the change expected.
        expected: u64,
        /// The revision the object is at, `None` when it does not exist.
        current: Option<u64>,
    },

    /// No catalog object of the name exists. Nothing changed.
    #[error("{object} '{name}' not found")]
    CatalogObjectMissing {
        /// What kind of object was looked for.
        object: CatalogObject,
        /// The name.
        name: String,
    },

    /// A catalog change the current catalog refuses. Nothing changed.
    #[error("catalog change refused: {0}")]
    CatalogRefused(String),

    /// No interactive transaction this caller may use is held under this ID.
    ///
    /// Either it was never opened on this leader (a transaction of an earlier
    /// leader among them: a change of leader ends every open transaction), or
    /// it is already finished: committed, rolled back, aborted by a failed
    /// statement, or collected by the idle sweep; or it acts for another
    /// caller, which the ID alone does not reveal. In every case nothing is
    /// held for this caller under it, and a transaction that was never
    /// committed is run again under a new ID. It is not proof that a commit
    /// did not happen: an operation's outcome is resolved by its descriptor.
    #[error("unknown transaction {0}")]
    UnknownTransaction(TransactionId),

    /// A transaction is open under this ID, or ended under it recently: an
    /// ID names one transaction, and a start never joins another. Nothing was
    /// done; a new transaction takes a newly drawn ID.
    #[error("transaction ID {0} is already in use")]
    TransactionIdInUse(TransactionId),

    /// The transaction's queue of commands waiting for their turn is full;
    /// send this one once earlier ones have answered. Nothing was done and
    /// the transaction stays open.
    #[error("transaction {0} has too many commands waiting")]
    TransactionBusy(TransactionId),

    /// A transaction that commits when it stays idle needs the client
    /// operation that commit binds, so that its client can learn the
    /// outcome; its start named none. Nothing was opened.
    #[error(
        "transaction {0} commits at idle expiry and so needs an operation its client can \
         resolve; none was named"
    )]
    OperationRequired(TransactionId),

    /// The commit names another client operation than the one the
    /// transaction was started under. Nothing was done; the transaction
    /// stays open.
    #[error("the commit of transaction {0} names another operation than its start")]
    OperationDiffers(TransactionId),

    /// The transaction stayed idle past the idle timeout and its timeout
    /// outcome committed it, under the operation its start named, which
    /// reports the commit timestamp.
    #[error("transaction {0} was committed when it stayed idle past the idle timeout")]
    TransactionCommittedOnTimeout(TransactionId),

    /// The transaction stayed idle past the idle timeout and the commit its
    /// timeout outcome made failed without proving whether it applied: the
    /// operation its start named resolves how it stands.
    #[error(
        "the commit of transaction {0} at the idle timeout has an unknown outcome; resolve its operation"
    )]
    TransactionOutcomeUnknown(TransactionId),

    /// The transaction ended before anything of it was submitted, so nothing
    /// of it was applied. Running it again is a new transaction.
    #[error("transaction {id} was aborted: {end}")]
    TransactionAborted {
        /// The transaction.
        id: TransactionId,
        /// Why it ended.
        end: TransactionEnd,
    },

    /// A write refused by OCC validation: a key it read was modified by a
    /// concurrent writer after its snapshot was pinned. Nothing was applied;
    /// running the whole transaction (or the statement, outside one) again is
    /// the intended response.
    #[error("{}", conflict_text(transaction.as_ref(), message))]
    TransactionConflict {
        /// The interactive transaction refused; `None` for a statement that
        /// commits on its own.
        transaction: Option<TransactionId>,
        /// What the validation found.
        message: String,
    },

    /// A write refused by a declared condition: the state its result depends
    /// on no longer holds, or a concurrent transaction holds an incompatible
    /// claim on it. Nothing was applied; running it again re-reads that
    /// state.
    ///
    /// Separate from `TransactionConflict` because the two need not touch a
    /// common key: disjoint writes can break one graph condition together,
    /// and reporting that as a write conflict names the wrong cause.
    #[error("{}", invariant_text(transaction.as_ref(), reason))]
    InvariantRefused {
        /// The interactive transaction refused; `None` for a statement that
        /// commits on its own.
        transaction: Option<TransactionId>,
        /// Which condition refused it, in the words of the condition.
        reason: String,
    },

    /// A write conditioned on a record's version found a different one.
    /// Nothing was applied.
    ///
    /// The version that is there now is part of the error: a caller that has
    /// to decide whether to retry, merge or give up needs it, and making it
    /// read the record again would hand it a second race instead of an
    /// answer.
    #[error("{}", revision_text(transaction.as_ref(), *expected, *current))]
    RevisionMismatch {
        /// The interactive transaction refused; `None` for a statement that
        /// commits on its own.
        transaction: Option<TransactionId>,
        /// The version the write was conditioned on; `None` required absence.
        expected: Option<u64>,
        /// The version the record has now; `None` means it is absent.
        current: Option<u64>,
    },

    /// A transaction buffered more uncommitted data than it is allowed to, and
    /// was discarded. Splitting the work into smaller transactions is the fix.
    #[error("transaction {id} exceeded its buffer limit ({buffered} > {limit} bytes)")]
    TransactionTooLarge {
        id: TransactionId,
        buffered: u64,
        limit: u64,
    },

    /// The storage engine is over its compaction-debt stop threshold and
    /// rejected the writes. Nothing was applied; retrying after a short delay
    /// succeeds once compaction catches up.
    #[error(
        "write rejected: storage is over its compaction-debt stop threshold; \
         retry after compaction catches up"
    )]
    WriteBackpressure,

    /// A storage endpoint is at its hard capacity limit and refuses writes.
    /// Nothing was applied.
    #[error(
        "endpoint {endpoint_id:?} capacity exhausted (used={used_bytes}, \
         hard_limit={hard_limit_bytes}); writes rejected until usage falls below the limit"
    )]
    CapacityExhausted {
        /// The endpoint.
        endpoint_id: String,
        /// Bytes it holds.
        used_bytes: u64,
        /// Its hard limit.
        hard_limit_bytes: u64,
    },

    /// The filesystem under a durable endpoint is below its free-space
    /// reserve, so writes are refused before any reaches the disk; reads go
    /// on. Writes are admitted again once space is freed.
    #[error(
        "no space: {available_bytes} bytes free under {path}, below the {min_free_bytes}-byte \
         reserve; writes are refused until space is freed, reads continue"
    )]
    StorageFull {
        /// The endpoint path with the least free space.
        path: String,
        /// Free bytes there at the last reading.
        available_bytes: u64,
        /// The reserve writes need.
        min_free_bytes: u64,
    },

    /// The write reached a node that is not the leader, so it could not be
    /// replicated. Nothing was applied; the same write succeeds at the leader,
    /// whose id is carried here when the cluster has named one.
    #[error("not the leader{}", match leader_id {
        Some(id) => format!("; leader is node {id}"),
        None => String::from("; no leader known yet"),
    })]
    NotLeader {
        /// The node the cluster last named leader, if any. `None` while an
        /// election is in flight.
        leader_id: Option<u64>,
    },

    /// The write reached a member that does not run its group's version: it
    /// is read-only. Nothing was applied. Names both versions and the leader
    /// when known.
    #[error("this member is read-only: {0}")]
    Mismatched(Mismatch),

    /// A read at a timestamp older than the MVCC retention horizon. History
    /// that old may already be collected, so the read is refused; the same
    /// read succeeds at any timestamp from `oldest_readable` on.
    #[error(
        "snapshot timestamp {requested} is older than the MVCC retention horizon: \
         history is readable from timestamp {oldest_readable} on"
    )]
    OutsideRetention {
        /// The timestamp the read asked for.
        requested: u64,
        /// The oldest timestamp still readable when the read ran.
        oldest_readable: u64,
    },

    /// The settings a statement names cannot be served together, or not on
    /// the path it took; refused before anything waited or ran.
    #[error("{0}")]
    SettingsRefused(String),

    /// The read could not be placed or fenced as its settings ask: nothing
    /// was read.
    #[error("{message}")]
    ReadFence {
        /// What stood in the way.
        kind: ReadFenceFailure,
        /// The fence's account of it.
        message: String,
    },

    /// A snapshot read whose timestamp covers a commit not yet applied, or not
    /// yet durable in the store's journal. The read does not wait for it and
    /// is not answered at another timestamp: nothing was read. A read at
    /// `ready_through` or earlier is answerable now.
    #[error(
        "snapshot timestamp {requested} is not ready: a commit it covers has not \
         landed or is not durable yet{}",
        match ready_through {
            Some(ts) => format!("; readable through timestamp {ts} now"),
            None => String::new(),
        }
    )]
    ReadNotReady {
        /// The timestamp the read asked for.
        requested: u64,
        /// The latest timestamp a read is answerable at now, when there is one.
        ready_through: Option<u64>,
    },

    /// The client operation a write carries out already stands as
    /// `outcome`, or cannot be admitted: this attempt ran or applied nothing.
    /// A committed outcome is the operation's answer, reached by an earlier
    /// attempt; `intent_differs` says that attempt meant something else (one
    /// number used for two writes). Anything else says why no attempt takes
    /// effect: aborted (excluded), retired, or not admissible now (an issuer
    /// not enrolled, closed, or past its admission window).
    #[error("operation {op:?} stands as {outcome:?}; this attempt applied nothing")]
    OperationSettled {
        /// The operation.
        op: OperationId,
        /// How it stands.
        outcome: OperationOutcome,
        /// Whether its committed outcome carries another intent.
        intent_differs: bool,
    },

    /// A descriptor names another deployment than this database's: its
    /// operation's record is not here, and nothing here can resolve it.
    #[error("the operation belongs to another deployment ({descriptor}), not this one")]
    ForeignOperation {
        /// The deployment the descriptor names.
        descriptor: DeploymentId,
    },

    /// The storage engine failed in a way no variant above classifies.
    #[error("storage error: {0}")]
    Storage(#[source] StorageFailure),

    /// A failure inside the database a caller cannot act on beyond reporting
    /// it.
    #[error("{0}")]
    Other(String),

    /// The member a command had to run at could not be reached: nothing was
    /// sent to it, so the command did not run. The same command may be sent
    /// again.
    #[error("member {member} could not be reached; the command was not sent: {message}")]
    PeerUnreachable {
        /// The member.
        member: u64,
        /// Why it could not be reached.
        message: String,
    },

    /// A command was sent to the member that runs it, and its answer did not
    /// arrive whole: the command may have run, and may have committed. It is
    /// neither a refusal nor an abort. A write's outcome is resolved through
    /// its client operation; it is never simply sent again.
    #[error(
        "the answer of member {member} was lost ({failure}); the command may have run: {message}"
    )]
    PeerOutcomeUnknown {
        /// The member.
        member: u64,
        /// How the answer was lost.
        failure: PeerFailure,
        /// What the transport or the decoder reported.
        message: String,
    },
}

/// Why a read could not be fenced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadFenceFailure {
    /// This member's role does not serve the read preference or concern: it
    /// needs the leader, or a follower, and this member is not one.
    Role,
    /// This member lags too far behind to serve the read.
    Stale,
    /// The fence did not reach the position or lease it waits for in time.
    Timeout,
}

/// How the answer of a routed command was lost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeerFailure {
    /// The connection failed or closed after the command was sent.
    Transport,
    /// The caller's deadline passed before the answer arrived.
    Deadline,
    /// The answer arrived but is not one this member can read: malformed,
    /// too large, of an unknown kind, or encoded by an incompatible codec.
    Protocol,
}

impl std::fmt::Display for PeerFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Transport => "transport failure",
            Self::Deadline => "deadline passed",
            Self::Protocol => "unreadable answer",
        })
    }
}

/// A storage failure no variant of [`DatabaseError`] classifies: its message,
/// and on the member that raised it, the engine's error as its cause.
#[derive(Debug)]
pub struct StorageFailure {
    message: String,
    cause: Option<Box<StorageError>>,
}

impl StorageFailure {
    /// A failure another member reported: its message, without a cause.
    pub fn reported(message: String) -> Self {
        Self {
            message,
            cause: None,
        }
    }

    /// What failed.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The engine's error, on the member that raised it: a diagnostic, absent
    /// from a failure another member reported.
    pub fn cause(&self) -> Option<&StorageError> {
        self.cause.as_deref()
    }
}

impl std::fmt::Display for StorageFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for StorageFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.cause
            .as_deref()
            .map(|e| e as &(dyn std::error::Error + 'static))
    }
}

impl DatabaseError {
    /// The engine's error behind a [`DatabaseError::Storage`] raised here:
    /// how a caller on this member inspects a failure no variant classifies.
    pub fn storage_cause(&self) -> Option<&StorageError> {
        match self {
            Self::Storage(failure) => failure.cause(),
            _ => None,
        }
    }
}

fn constraint_problem(kind: &ConstraintKind) -> &'static str {
    match kind {
        ConstraintKind::Type(_) => "has a value of another type",
        _ => "is missing or null",
    }
}

fn conflict_text(transaction: Option<&TransactionId>, message: &str) -> String {
    match transaction {
        Some(id) => format!("transaction {id} conflicts with a concurrent write: {message}"),
        None => format!("write conflict: {message}"),
    }
}

fn invariant_text(transaction: Option<&TransactionId>, reason: &str) -> String {
    match transaction {
        Some(id) => format!("transaction {id} was refused by an invariant: {reason}"),
        None => format!("invariant refused the write: {reason}"),
    }
}

fn revision_text(
    transaction: Option<&TransactionId>,
    expected: Option<u64>,
    current: Option<u64>,
) -> String {
    let refusal = format!(
        "record version mismatch, expected {expected:?}, found {current:?}; nothing was applied"
    );
    match transaction {
        Some(id) => format!("transaction {id}: {refusal}"),
        None => refusal,
    }
}

impl From<StorageError> for DatabaseError {
    fn from(e: StorageError) -> Self {
        match e {
            StorageError::CapacityExhausted {
                endpoint_id,
                used_bytes,
                hard_limit_bytes,
            } => Self::CapacityExhausted {
                endpoint_id,
                used_bytes,
                hard_limit_bytes,
            },
            StorageError::OutOfSpace {
                path,
                available_bytes,
                min_free_bytes,
            } => Self::StorageFull {
                path,
                available_bytes,
                min_free_bytes,
            },
            // Refused only for a snapshot below the watermark, so the
            // watermark is at least 1, and the oldest readable timestamp is
            // the one below it: a read at T is served from snapshot T + 1.
            // Snapshot 0 is the bottom of the domain, read by no timestamp:
            // the clamp names the lowest one, not an overflow.
            StorageError::SnapshotOutsideRetention {
                snapshot,
                watermark,
            } if watermark > 0 => Self::OutsideRetention {
                requested: snapshot.saturating_sub(1),
                oldest_readable: watermark - 1,
            },
            other => Self::Storage(StorageFailure {
                message: other.to_string(),
                cause: Some(Box::new(other)),
            }),
        }
    }
}

impl From<StoreError> for DatabaseError {
    fn from(e: StoreError) -> Self {
        match e {
            StoreError::Storage(storage) => storage.into(),
            StoreError::Budget(stop) => Self::Budget(stop),
            other => Self::Other(other.to_string()),
        }
    }
}

impl From<ExecutionError> for DatabaseError {
    fn from(e: ExecutionError) -> Self {
        use ExecutionError as E;
        match e {
            E::Storage(storage) => storage.into(),
            E::Modality(store) => store.into(),
            E::Budget(stop) => Self::Budget(stop),
            E::IndexUnresolved {
                label,
                property,
                operator,
            } => Self::IndexUnresolved {
                label,
                property,
                operator: operator.to_owned(),
            },
            E::Arithmetic(eval) => Self::Arithmetic(eval),
            E::OperationSettled {
                op,
                outcome,
                intent_differs,
            } => Self::OperationSettled {
                op,
                outcome,
                intent_differs,
            },
            E::Unsupported(what) => Self::Unsupported(what),
            E::Procedure(refusal) => Self::Procedure(refusal),
            E::Conflict(message) => Self::TransactionConflict {
                transaction: None,
                message,
            },
            E::InvariantRefused(reason) => Self::InvariantRefused {
                transaction: None,
                reason,
            },
            E::RevisionMismatch { expected, current } => Self::RevisionMismatch {
                transaction: None,
                expected,
                current,
            },
            E::Backpressure => Self::WriteBackpressure,
            E::NotLeader { leader_id } => Self::NotLeader { leader_id },
            E::Mismatched(mismatch) => Self::Mismatched(mismatch),
            E::OutsideRetention {
                requested,
                oldest_readable,
            } => Self::OutsideRetention {
                requested,
                oldest_readable,
            },
            E::SchemaViolation(detail) => Self::SchemaViolation(detail),
            E::DuplicateKey {
                table,
                key,
                element_id,
            } => Self::DuplicateKey {
                table,
                key,
                element_id,
            },
            E::UniqueViolation {
                index,
                property,
                value,
                element_id,
            } => Self::UniqueViolation {
                index,
                property,
                value,
                element_id,
            },
            E::UniquenessUnresolved { index, limit } => Self::UniquenessUnresolved { index, limit },
            E::ConstraintViolation {
                constraint,
                kind,
                label,
                property,
                element_id,
            } => Self::ConstraintViolation {
                constraint,
                kind,
                label,
                property,
                element_id,
            },
            E::KeyImmutable { table, column } => Self::KeyImmutable { table, column },
            E::CatalogObjectExists { object, name } => Self::CatalogObjectExists { object, name },
            E::CatalogObjectDiffers {
                object,
                name,
                revision,
            } => Self::CatalogObjectDiffers {
                object,
                name,
                revision,
            },
            E::CatalogRevisionMismatch {
                object,
                name,
                expected,
                current,
            } => Self::CatalogRevisionMismatch {
                object,
                name,
                expected,
                current,
            },
            E::CatalogObjectMissing { object, name } => Self::CatalogObjectMissing { object, name },
            E::CatalogRefused(reason) => Self::CatalogRefused(reason),
            E::CascadeOverflow {
                current,
                limit,
                chain,
            } => Self::CascadeOverflow {
                current,
                limit,
                chain,
            },
            E::CascadeFanoutOverflow {
                trigger,
                count,
                limit,
            } => Self::CascadeFanoutOverflow {
                trigger,
                count,
                limit,
            },
            // Faults of the engine a caller cannot act on: their text is the
            // whole of what they carry.
            e @ (E::Serialization(_) | E::NodeIdLease(_) | E::FieldDictionary(_)) => {
                Self::Other(e.to_string())
            }
        }
    }
}

impl From<ParseError> for DatabaseError {
    fn from(e: ParseError) -> Self {
        Self::Parse(e)
    }
}

impl From<PlanError> for DatabaseError {
    fn from(e: PlanError) -> Self {
        Self::Plan(e.to_string())
    }
}

impl From<SessionSettingRefused> for DatabaseError {
    fn from(e: SessionSettingRefused) -> Self {
        Self::SessionSetting {
            setting: e.setting.to_owned(),
            reason: e.reason,
        }
    }
}

impl From<BudgetStop> for DatabaseError {
    fn from(stop: BudgetStop) -> Self {
        Self::Budget(stop)
    }
}

impl From<coordinode_query::frontend::FrontendError> for DatabaseError {
    fn from(e: coordinode_query::frontend::FrontendError) -> Self {
        use coordinode_query::frontend::FrontendError as FE;
        match e {
            FE::Parse(p) => Self::Parse(p),
            FE::Plan(p) => p.into(),
            FE::Semantic(errors) => Self::Semantic(
                errors
                    .iter()
                    .map(|e| e.to_string())
                    .collect::<Vec<_>>()
                    .join("; "),
            ),
            FE::Message(m) => Self::Semantic(m),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
