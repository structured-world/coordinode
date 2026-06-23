//! CE default state-machine backend: [`LocalStateMachine`].
//!
//! In-process, zero external services. Each transition is registered as a
//! `(from, to)` edge with an async action; `start` spawns a tokio task that runs
//! the action, reports progress, and writes every state change through an
//! injected [`TransitionLog`] (wired to a metadata-Raft proposal in production,
//! a recorder in tests). Per ADR-038 the event stream is bound to a tokio
//! broadcast receiver.
//!
//! Resting-state model: a registered edge has exactly two resting states,
//! `from` and `to`. A cancel mid-transition rolls back to `from` (the action
//! observes [`ActionContext::is_cancelled`] and unwinds), satisfying the
//! contract's "roll back to the most recent valid resting state". Richer
//! multi-resting-state graphs are an `sflow`-backend concern.
//!
//! Crash recovery (replaying the metadata-Raft log into the registry on restart)
//! is wired where the log lives, above this crate; this module owns the live
//! lifecycle and the write-through persistence calls.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;
use tokio::sync::broadcast;

use super::{
    BackendError, ContextId, HealthEvent, OperationFilter, OperationId, OperationState,
    OperationStatus, OperationSummary, Progress, StateLabel, StateMachineBackend,
    TransitionRequest,
};

/// Capacity of each operation's event broadcast channel. A subscriber that lags
/// past this sees `RecvError::Lagged` and resubscribes — progress is
/// snapshot-able via [`StateMachineBackend::status`], so a dropped frame is not
/// fatal.
const EVENT_CHANNEL_CAPACITY: usize = 256;

/// A persisted checkpoint of a transition's state. The CE production wiring
/// turns each into a metadata-Raft proposal; tests record them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransitionCheckpoint {
    /// Operation being checkpointed.
    pub operation: OperationId,
    /// Entity under transition.
    pub context: ContextId,
    /// Source resting state.
    pub from: StateLabel,
    /// Target resting state.
    pub to: StateLabel,
    /// Lifecycle state at this checkpoint.
    pub state: OperationState,
    /// Progress snapshot at this checkpoint.
    pub progress: Progress,
}

/// Persistence port: every state change is written through here. Production
/// binds this to a metadata-Raft proposal (so a crash recovers by Raft replay);
/// tests bind a recorder. Implementations MUST NOT keep authoritative state in a
/// non-Raft store (ADR-038 recursion constraint).
pub trait TransitionLog: Send + Sync {
    /// Durably record a checkpoint. Called on every state/terminal change and
    /// may be called on progress updates.
    fn record(&self, checkpoint: TransitionCheckpoint);
}

/// Future returned by a transition action.
type ActionFuture = Pin<Box<dyn Future<Output = Result<(), String>> + Send>>;

/// A transition action: drives `context` from `from` to `to`, reporting progress
/// and honouring cancellation via the [`ActionContext`]. Returns `Ok` when the
/// target state is reached (or unwound after a cancel), `Err(reason)` on failure.
type Action = Arc<dyn Fn(ActionContext) -> ActionFuture + Send + Sync>;

/// Per-operation shared state. Held by the registry and the driving task.
struct OpInner {
    id: OperationId,
    request: TransitionRequest,
    state: RwLock<OperationState>,
    progress: RwLock<Progress>,
    error: RwLock<Option<String>>,
    cancel: AtomicBool,
    events: broadcast::Sender<HealthEvent>,
}

impl OpInner {
    fn snapshot_event(&self) -> HealthEvent {
        HealthEvent {
            operation: self.id,
            state: *self.state.read(),
            progress: self.progress.read().clone(),
        }
    }

    /// Move to `state`, persist the checkpoint, and broadcast it.
    fn transition(&self, state: OperationState, log: &dyn TransitionLog) {
        *self.state.write() = state;
        let progress = self.progress.read().clone();
        log.record(TransitionCheckpoint {
            operation: self.id,
            context: self.request.context.clone(),
            from: self.request.from.clone(),
            to: self.request.to.clone(),
            state,
            progress,
        });
        // A send error only means no live subscribers — state is still persisted.
        let _ = self.events.send(self.snapshot_event());
    }
}

/// Handle passed to a transition action for progress + cancellation.
pub struct ActionContext {
    /// The transition being performed.
    pub request: TransitionRequest,
    inner: Arc<OpInner>,
    log: Arc<dyn TransitionLog>,
}

impl ActionContext {
    /// Whether a cancel has been requested. Long-running actions poll this and
    /// unwind to the `from` resting state when it returns `true`.
    pub fn is_cancelled(&self) -> bool {
        self.inner.cancel.load(Ordering::Acquire)
    }

    /// Report progress: update the operation's [`Progress`], persist it, and push
    /// a [`HealthEvent`] to subscribers. `now_ms` is the caller's clock (the
    /// primitive never reads the system clock).
    pub fn report(
        &self,
        current_step: impl Into<String>,
        completed_units: u64,
        total_units: u64,
        eta_ms: Option<u64>,
        now_ms: i64,
    ) {
        {
            let mut p = self.inner.progress.write();
            p.current_step = current_step.into();
            p.completed_units = completed_units;
            p.total_units = total_units;
            p.eta_ms = eta_ms;
            p.last_updated_at_ms = now_ms;
        }
        let state = *self.inner.state.read();
        let progress = self.inner.progress.read().clone();
        self.log.record(TransitionCheckpoint {
            operation: self.inner.id,
            context: self.inner.request.context.clone(),
            from: self.inner.request.from.clone(),
            to: self.inner.request.to.clone(),
            state,
            progress,
        });
        let _ = self.inner.events.send(self.inner.snapshot_event());
    }
}

type IdemKey = (StateLabel, StateLabel, ContextId);

fn idem_key(r: &TransitionRequest) -> IdemKey {
    (r.from.clone(), r.to.clone(), r.context.clone())
}

fn initial_progress() -> Progress {
    Progress {
        current_step: "queued".to_string(),
        completed_units: 0,
        total_units: 0,
        eta_ms: None,
        last_updated_at_ms: 0,
    }
}

/// CE in-process state-machine backend.
pub struct LocalStateMachine {
    next_id: AtomicU64,
    ops: RwLock<HashMap<OperationId, Arc<OpInner>>>,
    idem: RwLock<HashMap<IdemKey, OperationId>>,
    actions: HashMap<(StateLabel, StateLabel), Action>,
    log: Arc<dyn TransitionLog>,
}

/// Builder registering the `(from, to)` transition edges a [`LocalStateMachine`]
/// can drive. Each R-PROD* state graph registers its edges here.
pub struct LocalStateMachineBuilder {
    actions: HashMap<(StateLabel, StateLabel), Action>,
}

impl LocalStateMachineBuilder {
    /// Register an async `action` for the `from -> to` edge.
    pub fn transition<F, Fut>(mut self, from: StateLabel, to: StateLabel, action: F) -> Self
    where
        F: Fn(ActionContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<(), String>> + Send + 'static,
    {
        self.actions
            .insert((from, to), Arc::new(move |ctx| Box::pin(action(ctx))));
        self
    }

    /// Finish, binding the persistence `log` (any concrete [`TransitionLog`],
    /// coerced to the trait object).
    pub fn build<L: TransitionLog + 'static>(self, log: Arc<L>) -> LocalStateMachine {
        LocalStateMachine {
            next_id: AtomicU64::new(1),
            ops: RwLock::new(HashMap::new()),
            idem: RwLock::new(HashMap::new()),
            actions: self.actions,
            log,
        }
    }
}

impl LocalStateMachine {
    /// Start a builder.
    pub fn builder() -> LocalStateMachineBuilder {
        LocalStateMachineBuilder {
            actions: HashMap::new(),
        }
    }

    fn get(&self, op: OperationId) -> Result<Arc<OpInner>, BackendError> {
        self.ops
            .read()
            .get(&op)
            .cloned()
            .ok_or(BackendError::UnknownOperation(op))
    }

    /// Insert a fresh operation in `state` and index it for idempotency. Returns
    /// the shared inner handle.
    fn insert(&self, request: TransitionRequest, state: OperationState) -> Arc<OpInner> {
        let id = OperationId(self.next_id.fetch_add(1, Ordering::Relaxed));
        let (events, _) = broadcast::channel(EVENT_CHANNEL_CAPACITY);
        let inner = Arc::new(OpInner {
            id,
            request: request.clone(),
            state: RwLock::new(state),
            progress: RwLock::new(initial_progress()),
            error: RwLock::new(None),
            cancel: AtomicBool::new(false),
            events,
        });
        self.ops.write().insert(id, Arc::clone(&inner));
        self.idem.write().insert(idem_key(&request), id);
        inner
    }

    /// Spawn the background task that drives `inner` through `action`: mark
    /// InProgress, run the action, then settle on a terminal state (Cancelled if
    /// the cancel flag is set — including a recovered Cancelling op — else
    /// Completed / Failed). Shared by [`start`](StateMachineBackend::start) and
    /// [`recover`](Self::recover).
    fn spawn_driver(&self, inner: Arc<OpInner>, action: Action) {
        let log = Arc::clone(&self.log);
        let request = inner.request.clone();
        tokio::spawn(async move {
            inner.transition(OperationState::InProgress, log.as_ref());
            let ctx = ActionContext {
                request,
                inner: Arc::clone(&inner),
                log: Arc::clone(&log),
            };
            let result = action(ctx).await;
            let terminal = if inner.cancel.load(Ordering::Acquire) {
                OperationState::Cancelled
            } else {
                match result {
                    Ok(()) => OperationState::Completed,
                    Err(reason) => {
                        *inner.error.write() = Some(reason);
                        OperationState::Failed
                    }
                }
            };
            inner.transition(terminal, log.as_ref());
        });
    }

    /// Rebuild the registry from persisted checkpoints after a restart (ADR-038
    /// crash recovery). The caller replays committed metadata-Raft entries
    /// (decoded to [`TransitionCheckpoint`]) in commit order; this folds them to
    /// each operation's last committed state, re-registers every operation
    /// (terminal ones stay queryable and keep deduping repeat starts), and
    /// **resumes** every non-terminal operation by re-driving its action from
    /// the `from` resting state — safe because transition actions are idempotent.
    /// A recovered `Cancelling` op resumes with its cancel flag set so it unwinds
    /// to `Cancelled`. A non-terminal op whose edge is not registered in this
    /// build is settled `Failed` (config mismatch).
    ///
    /// Must run inside a tokio runtime (it may spawn resume tasks). Call once at
    /// startup before serving the topology API.
    pub fn recover<I: IntoIterator<Item = TransitionCheckpoint>>(&self, checkpoints: I) {
        let mut last: HashMap<OperationId, TransitionCheckpoint> = HashMap::new();
        let mut max_id = 0u64;
        for c in checkpoints {
            max_id = max_id.max(c.operation.0);
            last.insert(c.operation, c);
        }
        // New operations must not reuse a recovered id.
        self.next_id.fetch_max(max_id + 1, Ordering::Relaxed);

        for (id, c) in last {
            let request = TransitionRequest {
                context: c.context,
                from: c.from,
                to: c.to,
            };
            let (events, _) = broadcast::channel(EVENT_CHANNEL_CAPACITY);
            let inner = Arc::new(OpInner {
                id,
                request: request.clone(),
                state: RwLock::new(c.state),
                progress: RwLock::new(c.progress),
                error: RwLock::new(None),
                cancel: AtomicBool::new(c.state == OperationState::Cancelling),
                events,
            });
            self.ops.write().insert(id, Arc::clone(&inner));
            self.idem.write().insert(idem_key(&request), id);

            if c.state.is_terminal() {
                continue;
            }
            match self
                .actions
                .get(&(request.from.clone(), request.to.clone()))
            {
                Some(action) => self.spawn_driver(inner, Arc::clone(action)),
                None => {
                    *inner.error.write() =
                        Some("no action registered for recovered transition".to_string());
                    inner.transition(OperationState::Failed, self.log.as_ref());
                }
            }
        }
    }
}

impl StateMachineBackend for LocalStateMachine {
    type EventStream = broadcast::Receiver<HealthEvent>;

    fn start(&self, request: TransitionRequest) -> Result<OperationId, BackendError> {
        // Idempotency: a repeated (from, to, context) returns the original op.
        if let Some(&existing) = self.idem.read().get(&idem_key(&request)) {
            return Ok(existing);
        }

        // Already at the target: a completed no-op handle.
        if request.from == request.to {
            let inner = self.insert(request, OperationState::Completed);
            inner.transition(OperationState::Completed, self.log.as_ref());
            return Ok(inner.id);
        }

        let edge = (request.from.clone(), request.to.clone());
        let action =
            self.actions
                .get(&edge)
                .cloned()
                .ok_or_else(|| BackendError::NoSuchTransition {
                    from: request.from.clone(),
                    to: request.to.clone(),
                })?;

        let inner = self.insert(request, OperationState::Queued);
        inner.transition(OperationState::Queued, self.log.as_ref());
        self.spawn_driver(Arc::clone(&inner), action);

        Ok(inner.id)
    }

    fn status(&self, op: OperationId) -> Result<OperationStatus, BackendError> {
        let inner = self.get(op)?;
        let state = *inner.state.read();
        let progress = inner.progress.read().clone();
        let error = inner.error.read().clone();
        Ok(OperationStatus {
            id: op,
            state,
            progress,
            error,
        })
    }

    fn cancel(&self, op: OperationId) -> Result<(), BackendError> {
        let inner = self.get(op)?;
        // No-op on a terminal operation, or on a repeated cancel.
        if inner.state.read().is_terminal() {
            return Ok(());
        }
        inner.cancel.store(true, Ordering::Release);
        inner.transition(OperationState::Cancelling, self.log.as_ref());
        Ok(())
    }

    fn observe(&self, op: OperationId) -> Result<Self::EventStream, BackendError> {
        let inner = self.get(op)?;
        // Subscribe first, then push a current-state snapshot so a late
        // subscriber receives the current state without polling.
        let rx = inner.events.subscribe();
        let _ = inner.events.send(inner.snapshot_event());
        Ok(rx)
    }

    fn list(&self, filter: &OperationFilter) -> Vec<OperationSummary> {
        let mut out: Vec<(OperationId, OperationSummary)> = self
            .ops
            .read()
            .values()
            .filter_map(|inner| {
                let state = *inner.state.read();
                if filter.state.is_some_and(|s| s != state) {
                    return None;
                }
                if filter
                    .context
                    .as_ref()
                    .is_some_and(|c| *c != inner.request.context)
                {
                    return None;
                }
                Some((
                    inner.id,
                    OperationSummary {
                        id: inner.id,
                        context: inner.request.context.clone(),
                        state,
                        current_step: inner.progress.read().current_step.clone(),
                    },
                ))
            })
            .collect();
        // Most recent first (ids are monotonic).
        out.sort_by_key(|(id, _)| std::cmp::Reverse(*id));
        out.into_iter().map(|(_, s)| s).collect()
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests;
