use super::*;

fn history() -> HistoryId {
    HistoryId {
        deployment: DeploymentId([3; 16]),
        group: GroupId(0),
    }
}

/// A replicated write reports its own commit and where its entry landed,
/// qualified by its history on the wire.
#[test]
fn a_replicated_write_reports_its_commit_and_position() {
    let r = receipt(Some(500), Some(42), Some(499), Some(history()));
    assert_eq!(r.operation_time, 500);
    assert_eq!(
        r.position,
        Some(HistoryPosition {
            history: history(),
            index: 42
        })
    );
    let wire = receipt_to_proto(r);
    let qualified = wire
        .position
        .and_then(|p| p.history)
        .expect("qualified by its history");
    assert_eq!(qualified.deployment, vec![3; 16]);
    assert_eq!(qualified.group, 0);
}

/// A replicated write answered before its entry reached the log may still be
/// lost: it gives no operation time, which a later read could never meet.
#[test]
fn a_write_answered_before_its_entry_gives_no_operation_time() {
    let r = receipt(Some(500), None, Some(499), Some(history()));
    assert_eq!(r.operation_time, 0);
    assert!(r.position.is_none());
}

/// A standalone write has no log: its commit timestamp alone is the receipt,
/// never a constant zero.
#[test]
fn a_standalone_write_reports_its_commit() {
    let r = receipt(Some(500), None, Some(499), None);
    assert_eq!(r.operation_time, 500);
    assert!(r.position.is_none());
}

/// A read reports the timestamp it read at and no position.
#[test]
fn a_read_reports_its_view() {
    let r = receipt(None, None, Some(777), Some(history()));
    assert_eq!(r.operation_time, 777);
    assert!(r.position.is_none());
}

/// A position travels with its history and back unchanged.
#[test]
fn a_position_round_trips() {
    let position = HistoryPosition {
        history: history(),
        index: 9,
    };
    let wire = position_to_proto(position);
    assert_eq!(
        position_from_proto("after_position", Some(&wire)).expect("valid"),
        Some(position)
    );
}

/// A position without its history, or with a deployment that is not 16
/// bytes, is refused: it could not be compared with any member's log.
#[test]
fn a_position_without_a_whole_history_is_refused() {
    let headless = replication::HistoryPosition {
        history: None,
        index: 9,
    };
    let err = position_from_proto("after_position", Some(&headless)).expect_err("no history");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);
    let short = replication::HistoryPosition {
        history: Some(replication::HistoryId {
            deployment: vec![1; 8],
            group: 0,
        }),
        index: 9,
    };
    let err = position_from_proto("after_position", Some(&short)).expect_err("short id");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);
}

/// Zero milliseconds allow no wait.
#[test]
fn zero_wait_is_no_wait() {
    assert_eq!(wait_from_proto(0), None);
    assert_eq!(wait_from_proto(250), Some(Duration::from_millis(250)));
}
