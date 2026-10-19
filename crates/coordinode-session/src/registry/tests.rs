use super::*;
use coordinode_core::operations::OperationsView;

fn handle(local: u64) -> TransactionHandle {
    TransactionHandle::new(None, 1, 1, local)
}

#[test]
fn registers_sessions_with_distinct_ids_and_snapshots_them() {
    let reg = SessionRegistry::new();
    let a = reg.register_session("10.0.0.1:1".into());
    let b = reg.register_session("10.0.0.2:2".into());
    assert_ne!(a, b, "session ids are distinct");

    let snap = reg.sessions();
    assert_eq!(snap.len(), 2);
    assert!(
        snap.iter()
            .all(|s| s.in_flight == 0 && s.attached.is_empty())
    );
    assert!(snap.iter().any(|s| s.peer == "10.0.0.1:1"));
}

#[test]
fn close_session_removes_it() {
    let reg = SessionRegistry::new();
    let id = reg.register_session("p".into());
    reg.close_session(id);
    assert!(reg.sessions().is_empty());
}

#[test]
fn in_flight_counts_up_and_down_without_underflow() {
    let reg = SessionRegistry::new();
    let id = reg.register_session("p".into());
    reg.request_started(id);
    reg.request_started(id);
    assert_eq!(reg.sessions()[0].in_flight, 2);
    reg.request_finished(id);
    assert_eq!(reg.sessions()[0].in_flight, 1);
    // Extra finishes must not underflow past zero.
    reg.request_finished(id);
    reg.request_finished(id);
    assert_eq!(reg.sessions()[0].in_flight, 0);
}

/// A session's attachments are reported as the handles a client carries, and
/// detaching one leaves the others.
#[test]
fn attachments_appear_and_detach_one_at_a_time() {
    let reg = SessionRegistry::new();
    let id = reg.register_session("p".into());
    reg.attach(id, handle(1));
    reg.attach(id, handle(2));
    reg.attach(id, handle(2));
    reg.detach(id, &handle(1));

    let attached = reg.sessions()[0].attached.clone();
    assert_eq!(attached, vec![handle(2).to_string()]);
}
