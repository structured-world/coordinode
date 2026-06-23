use super::*;
use parking_lot::Mutex;
use std::time::Duration;

/// Recording TransitionLog for assertions on persistence.
#[derive(Default)]
struct RecordingLog {
    checkpoints: Mutex<Vec<(OperationId, OperationState)>>,
}
impl TransitionLog for RecordingLog {
    fn record(&self, c: TransitionCheckpoint) {
        self.checkpoints.lock().push((c.operation, c.state));
    }
}

fn label(s: &str) -> StateLabel {
    StateLabel(s.to_string())
}
fn ctx_id(s: &str) -> ContextId {
    ContextId(s.to_string())
}
fn req(from: &str, to: &str, context: &str) -> TransitionRequest {
    TransitionRequest {
        context: ctx_id(context),
        from: label(from),
        to: label(to),
    }
}

/// Build a machine with a single compute->full-storage edge whose action
/// reports one progress step then succeeds.
fn machine(log: Arc<RecordingLog>) -> LocalStateMachine {
    LocalStateMachine::builder()
        .transition(label("compute"), label("full-storage"), |ctx| async move {
            ctx.report("transfer-segments", 1, 1, Some(0), 1);
            Ok(())
        })
        .build(log)
}

/// Await until `op` reaches a terminal state (bounded so a hang fails the test).
async fn await_terminal(m: &LocalStateMachine, op: OperationId) -> OperationState {
    for _ in 0..200 {
        let s = m.status(op).expect("status").state;
        if s.is_terminal() {
            return s;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("operation {op:?} never reached a terminal state");
}

#[tokio::test]
async fn transition_runs_to_completion() {
    let log = Arc::new(RecordingLog::default());
    let m = machine(Arc::clone(&log));
    let op = m
        .start(req("compute", "full-storage", "node-1"))
        .expect("start");

    assert_eq!(await_terminal(&m, op).await, OperationState::Completed);
    let status = m.status(op).expect("status");
    assert_eq!(status.progress.completed_units, 1);
    assert_eq!(status.error, None);
    // Persisted at least Queued -> InProgress -> Completed.
    let states: Vec<_> = log.checkpoints.lock().iter().map(|(_, s)| *s).collect();
    assert!(states.contains(&OperationState::InProgress));
    assert!(states.contains(&OperationState::Completed));
}

#[tokio::test]
async fn start_is_idempotent_on_same_triple() {
    let m = machine(Arc::new(RecordingLog::default()));
    let a = m
        .start(req("compute", "full-storage", "node-1"))
        .expect("a");
    let b = m
        .start(req("compute", "full-storage", "node-1"))
        .expect("b");
    assert_eq!(
        a, b,
        "same (from,to,context) must return the same operation"
    );

    // A different context is a distinct operation.
    let c = m
        .start(req("compute", "full-storage", "node-2"))
        .expect("c");
    assert_ne!(a, c);
}

#[tokio::test]
async fn from_equals_to_completes_immediately() {
    let m = machine(Arc::new(RecordingLog::default()));
    let op = m
        .start(req("full-storage", "full-storage", "node-1"))
        .expect("start");
    assert_eq!(
        m.status(op).expect("status").state,
        OperationState::Completed
    );
}

#[tokio::test]
async fn unknown_transition_is_rejected() {
    let m = machine(Arc::new(RecordingLog::default()));
    let err = m.start(req("compute", "archived", "node-1")).unwrap_err();
    assert!(matches!(err, BackendError::NoSuchTransition { .. }));
}

#[tokio::test]
async fn cancel_mid_transition_rolls_back_to_cancelled() {
    let log = Arc::new(RecordingLog::default());
    // Action loops until cancelled, then unwinds to the from resting state.
    let m = LocalStateMachine::builder()
        .transition(label("compute"), label("full-storage"), |ctx| async move {
            for tick in 0..1000 {
                if ctx.is_cancelled() {
                    return Ok(()); // unwound to `from`
                }
                ctx.report("transfer", tick, 1000, None, tick as i64);
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
            Ok(())
        })
        .build(Arc::clone(&log));

    let op = m
        .start(req("compute", "full-storage", "node-1"))
        .expect("start");
    // Let it enter InProgress, then cancel.
    tokio::time::sleep(Duration::from_millis(10)).await;
    m.cancel(op).expect("cancel");
    assert_eq!(await_terminal(&m, op).await, OperationState::Cancelled);

    // Cancel on a terminal op is a no-op.
    assert!(m.cancel(op).is_ok());
}

#[tokio::test]
async fn failed_action_surfaces_error() {
    let m = LocalStateMachine::builder()
        .transition(label("compute"), label("full-storage"), |_ctx| async move {
            Err("snapshot transfer failed".to_string())
        })
        .build(Arc::new(RecordingLog::default()));
    let op = m
        .start(req("compute", "full-storage", "node-1"))
        .expect("start");
    assert_eq!(await_terminal(&m, op).await, OperationState::Failed);
    assert_eq!(
        m.status(op).expect("status").error.as_deref(),
        Some("snapshot transfer failed")
    );
}

#[tokio::test]
async fn observe_delivers_current_state_then_updates() {
    let m = LocalStateMachine::builder()
        .transition(label("compute"), label("full-storage"), |ctx| async move {
            for tick in 0..5 {
                ctx.report("step", tick, 5, None, tick as i64);
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            Ok(())
        })
        .build(Arc::new(RecordingLog::default()));
    let op = m
        .start(req("compute", "full-storage", "node-1"))
        .expect("start");

    let mut rx = m.observe(op).expect("observe");
    // First event is the current-state snapshot pushed at subscribe time.
    let first = rx.recv().await.expect("snapshot event");
    assert_eq!(first.operation, op);

    // Drain until we see a terminal event.
    let mut saw_terminal = false;
    for _ in 0..100 {
        match rx.recv().await {
            Ok(ev) if ev.state.is_terminal() => {
                saw_terminal = true;
                break;
            }
            Ok(_) => continue,
            Err(broadcast::error::RecvError::Lagged(_)) => continue,
            Err(broadcast::error::RecvError::Closed) => break,
        }
    }
    assert!(saw_terminal, "observer must see a terminal event");
}

#[tokio::test]
async fn list_filters_by_state_and_context() {
    let m = machine(Arc::new(RecordingLog::default()));
    let a = m
        .start(req("compute", "full-storage", "node-1"))
        .expect("a");
    let _b = m
        .start(req("compute", "full-storage", "node-2"))
        .expect("b");
    await_terminal(&m, a).await;

    let all = m.list(&OperationFilter::default());
    assert_eq!(all.len(), 2);
    // Most recent first (monotonic ids).
    assert!(all[0].id > all[1].id);

    let only_node1 = m.list(&OperationFilter {
        state: None,
        context: Some(ctx_id("node-1")),
    });
    assert_eq!(only_node1.len(), 1);
    assert_eq!(only_node1[0].context, ctx_id("node-1"));

    let completed = m.list(&OperationFilter {
        state: Some(OperationState::Completed),
        context: None,
    });
    assert!(completed
        .iter()
        .all(|s| s.state == OperationState::Completed));
}
