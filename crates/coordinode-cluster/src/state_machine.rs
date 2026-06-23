//! Node-state-machine primitive (R-PROD-STATE-MACHINE, ADR-031 / ADR-038).
//!
//! Every runtime node reconfiguration in CoordiNode is a long-running
//! state-machine transition driven by a metadata-Raft proposal, never a process
//! restart (ADR-031). This module defines the **backend contract** every
//! transition runs on: a small trait plus the operation/progress/event types
//! shared by all R-PROD* transitions (storage role, Raft role, compression,
//! placement scope). Factoring the five cross-cutting mechanisms — idempotency,
//! cancellation, progress, observation, Raft-persisted intermediate state — into
//! one primitive keeps each R-PROD* task a thin state graph plus action
//! callbacks.
//!
//! Two backends implement the contract (ADR-038):
//! - CE `LocalStateMachine` — in-process, zero external deps, binds
//!   [`StateMachineBackend::EventStream`] to a tokio broadcast receiver.
//! - EE `SflowBackend` (opt-in `sflow` feature) — binds `EventStream` to
//!   `sflow`'s own workflow event stream.
//!
//! The trait surface is deliberately minimal: it never exports concepts only one
//! backend needs (workflow DAGs, retry policies, compensation handlers are
//! `sflow` concerns and live on that backend's concrete API). Each backend binds
//! the associated [`EventStream`](StateMachineBackend::EventStream) type to its
//! native stream, so neither boxing nor a forced runtime choice leaks into the
//! trait.

/// Identity of a transition operation, assigned by the backend's registry.
/// Stable for the operation's whole lifecycle, including across crash recovery
/// (the id is part of the Raft-persisted state).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OperationId(pub u64);

/// A node-behaviour state in some R-PROD* state graph (e.g. `"compute"`,
/// `"full-storage"`, `"voter"`, `"learner"`). Opaque to the primitive — each
/// transition's own state graph gives the labels meaning. Every label in a graph
/// must be a valid resting state (the cluster can run there indefinitely), so a
/// cancel can roll back to one.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StateLabel(pub String);

/// The entity a transition targets (a node id, endpoint id, shard id, ...).
/// Part of the idempotency key: a second `start` for the same
/// `(from, to, context)` is a no-op returning the original [`OperationId`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ContextId(pub String);

/// Lifecycle state of a transition operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationState {
    /// Accepted, not yet running (e.g. waiting on a concurrency slot).
    Queued,
    /// Actively driving the state graph.
    InProgress,
    /// Reached the target state; terminal.
    Completed,
    /// A `cancel` was requested; rolling back to the last valid resting state.
    Cancelling,
    /// Rollback finished; the operation did not reach the target state. Terminal.
    Cancelled,
    /// The transition failed; terminal. Cause carried in
    /// [`OperationStatus::error`].
    Failed,
}

impl OperationState {
    /// Whether the operation has reached a terminal state (no further events).
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            OperationState::Completed | OperationState::Cancelled | OperationState::Failed
        )
    }
}

/// Progress of an in-flight transition. Backends update this at least once per
/// second during active work or on every meaningful boundary; progress older
/// than 30s on a non-terminal operation is a backend bug.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Progress {
    /// Human-readable label of the current sub-step.
    pub current_step: String,
    /// Units of work completed (segments transferred, bytes, sub-steps).
    pub completed_units: u64,
    /// Total units of work for the transition (`0` when not yet known).
    pub total_units: u64,
    /// Backend ETA in milliseconds; `None` when unknown.
    pub eta_ms: Option<u64>,
    /// Wall-clock of the last progress update, Unix milliseconds. Caller/back-end
    /// supplied — the primitive never reads the system clock itself.
    pub last_updated_at_ms: i64,
}

impl Progress {
    /// Fraction complete in `[0.0, 1.0]`, or `None` when `total_units` is 0
    /// (work size not yet known).
    pub fn fraction(&self) -> Option<f64> {
        (self.total_units > 0).then(|| {
            (self.completed_units.min(self.total_units) as f64) / (self.total_units as f64)
        })
    }

    /// Whether this progress is stale: non-terminal work whose last update is
    /// older than 30s relative to `now_ms`. Observability surfaces flag this.
    pub fn is_stale(&self, now_ms: i64) -> bool {
        now_ms.saturating_sub(self.last_updated_at_ms) > 30_000
    }
}

/// An event in an operation's [`observe`](StateMachineBackend::observe) stream.
/// A late subscriber receives a single current-state event at attach time, then
/// every subsequent transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HealthEvent {
    /// Operation this event belongs to.
    pub operation: OperationId,
    /// Operation state at the time of the event.
    pub state: OperationState,
    /// Progress snapshot at the time of the event.
    pub progress: Progress,
}

/// Full status of one operation, returned by
/// [`status`](StateMachineBackend::status).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationStatus {
    /// The operation.
    pub id: OperationId,
    /// Current lifecycle state.
    pub state: OperationState,
    /// Latest progress.
    pub progress: Progress,
    /// Failure cause when `state` is [`OperationState::Failed`]; else `None`.
    pub error: Option<String>,
}

/// A transition to start: move `context` from `from` to `to`. The
/// `(from, to, context)` triple is the idempotency key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransitionRequest {
    /// Entity being transitioned.
    pub context: ContextId,
    /// Expected current state (the transition is a no-op if already at `to`).
    pub from: StateLabel,
    /// Target resting state.
    pub to: StateLabel,
}

/// Filter for [`list`](StateMachineBackend::list).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OperationFilter {
    /// Restrict to operations in this state; `None` = any state.
    pub state: Option<OperationState>,
    /// Restrict to operations targeting this context; `None` = any context.
    pub context: Option<ContextId>,
}

/// Lightweight operation descriptor returned by
/// [`list`](StateMachineBackend::list).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationSummary {
    /// The operation.
    pub id: OperationId,
    /// Entity it targets.
    pub context: ContextId,
    /// Current lifecycle state.
    pub state: OperationState,
    /// Current sub-step label.
    pub current_step: String,
}

/// Backend errors surfaced to the API caller. Carry no transition payload.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum BackendError {
    /// No operation with this id (unknown, or evicted after terminal retention).
    #[error("unknown operation {0:?}")]
    UnknownOperation(OperationId),
    /// `to` is not reachable from `from` in the transition's state graph.
    #[error("no transition from {from:?} to {to:?}")]
    NoSuchTransition {
        /// Requested source state.
        from: StateLabel,
        /// Requested target state.
        to: StateLabel,
    },
    /// The configured backend was not compiled in (e.g. `state_machine.backend =
    /// "sflow"` against a binary built without `--features sflow`).
    #[error("backend not compiled in: {0}")]
    BackendNotCompiledIn(String),
    /// Backend-internal failure (persistence, task spawn, ...).
    #[error("state-machine backend error: {0}")]
    Internal(String),
}

/// The contract every state-machine backend honours (ADR-038). Drives one
/// long-running node-reconfiguration transition: idempotent `start`, `status`
/// polling, graceful `cancel`, a `observe` event stream, and `list`.
///
/// Persistence is a backend obligation, not part of this surface: every
/// transition's intermediate state lives in the metadata Raft group (a crash
/// mid-transition recovers by Raft replay), so a backend must not keep state in
/// its own non-Raft store.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a state-machine backend",
    label = "needs `impl StateMachineBackend`",
    note = "implement the CE `LocalStateMachine` (in-process, tokio broadcast events) \
            or the EE `SflowBackend`; see arch/operations/state-machine.md and ADR-038"
)]
pub trait StateMachineBackend: Send + Sync {
    /// Per-operation health/progress event stream. Each backend binds this to its
    /// native stream type — CE `LocalStateMachine` to
    /// `tokio::sync::broadcast::Receiver<HealthEvent>`, the EE sflow backend to
    /// `sflow`'s workflow event stream — so the trait forces neither boxing nor a
    /// runtime choice.
    type EventStream;

    /// Start (or rejoin) a transition. **Idempotent**: a second call with the
    /// same `(from, to, context)` returns the existing [`OperationId`]; a call
    /// where the current state already equals `to` returns a completed handle;
    /// an already-completed transition returns its original terminal handle.
    /// Returns within milliseconds — the work runs in the background.
    ///
    /// # Errors
    /// [`BackendError::NoSuchTransition`] if `to` is unreachable from `from`;
    /// [`BackendError::Internal`] on a persistence / spawn failure.
    fn start(&self, request: TransitionRequest) -> Result<OperationId, BackendError>;

    /// Current status of an operation.
    ///
    /// # Errors
    /// [`BackendError::UnknownOperation`] if `op` is not known.
    fn status(&self, op: OperationId) -> Result<OperationStatus, BackendError>;

    /// Request graceful cancellation: roll back to the most recent valid resting
    /// state. Returns immediately; the operation moves
    /// [`Cancelling`](OperationState::Cancelling) →
    /// [`Cancelled`](OperationState::Cancelled). A second `cancel` is a no-op.
    ///
    /// # Errors
    /// [`BackendError::UnknownOperation`] if `op` is not known.
    fn cancel(&self, op: OperationId) -> Result<(), BackendError>;

    /// Subscribe to an operation's [`HealthEvent`] stream. A subscriber attaching
    /// mid-flight first receives a current-state event, then every subsequent
    /// transition.
    ///
    /// # Errors
    /// [`BackendError::UnknownOperation`] if `op` is not known.
    fn observe(&self, op: OperationId) -> Result<Self::EventStream, BackendError>;

    /// List operations matching `filter` (most recent first).
    fn list(&self, filter: &OperationFilter) -> Vec<OperationSummary>;
}

#[cfg(test)]
mod tests;
