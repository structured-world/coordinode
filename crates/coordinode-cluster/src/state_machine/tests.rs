use super::*;

#[test]
fn terminal_states_classified() {
    assert!(OperationState::Completed.is_terminal());
    assert!(OperationState::Cancelled.is_terminal());
    assert!(OperationState::Failed.is_terminal());
    assert!(!OperationState::Queued.is_terminal());
    assert!(!OperationState::InProgress.is_terminal());
    assert!(!OperationState::Cancelling.is_terminal());
}

fn progress(completed: u64, total: u64, updated_ms: i64) -> Progress {
    Progress {
        current_step: "transfer".to_string(),
        completed_units: completed,
        total_units: total,
        eta_ms: None,
        last_updated_at_ms: updated_ms,
    }
}

#[test]
fn fraction_is_none_when_total_unknown() {
    assert_eq!(progress(5, 0, 0).fraction(), None);
}

#[test]
fn fraction_is_clamped_and_correct() {
    assert_eq!(progress(0, 4, 0).fraction(), Some(0.0));
    assert_eq!(progress(1, 4, 0).fraction(), Some(0.25));
    assert_eq!(progress(4, 4, 0).fraction(), Some(1.0));
    // Over-count clamps to 1.0 rather than exceeding it.
    assert_eq!(progress(9, 4, 0).fraction(), Some(1.0));
}

#[test]
fn staleness_uses_30s_window() {
    let p = progress(1, 10, 1_000_000);
    assert!(!p.is_stale(1_000_000)); // same instant
    assert!(!p.is_stale(1_030_000)); // exactly 30s — not yet stale
    assert!(p.is_stale(1_030_001)); // just past 30s
                                    // A clock that went backwards must not report stale (saturating).
    assert!(!p.is_stale(999_999));
}

#[test]
fn idempotency_key_equality() {
    // The (from, to, context) triple is the idempotency key: equal triples are
    // equal requests, so a backend dedupes a repeated start.
    let a = TransitionRequest {
        context: ContextId("node-7".to_string()),
        from: StateLabel("compute".to_string()),
        to: StateLabel("full-storage".to_string()),
    };
    let b = a.clone();
    assert_eq!(a, b);

    let different_context = TransitionRequest {
        context: ContextId("node-8".to_string()),
        ..a.clone()
    };
    assert_ne!(a, different_context);
}

#[test]
fn operation_filter_defaults_to_match_all() {
    let f = OperationFilter::default();
    assert_eq!(f.state, None);
    assert_eq!(f.context, None);
}

#[test]
fn backend_error_messages_carry_no_payload() {
    // Errors are safe to log: they name the operation/states, not transition data.
    let e = BackendError::UnknownOperation(OperationId(42));
    assert_eq!(e.to_string(), "unknown operation OperationId(42)");
    let nt = BackendError::NoSuchTransition {
        from: StateLabel("voter".to_string()),
        to: StateLabel("archived".to_string()),
    };
    assert!(nt.to_string().contains("voter") && nt.to_string().contains("archived"));
}
