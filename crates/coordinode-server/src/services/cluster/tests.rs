use super::*;

fn status(matched_index: Option<u64>, lag_entries: u64) -> NodeReplicationStatus {
    NodeReplicationStatus {
        node_id: 2,
        role: NodeRole::Learner,
        matched_index,
        lag_entries,
        last_heartbeat_ago_ms: None,
    }
}

/// A member that has acknowledged nothing is down, whatever its lag: in a
/// young group the whole log is a few entries, which read as healthy.
#[test]
fn a_member_that_acknowledged_nothing_is_down() {
    assert_eq!(proto_state(&status(None, 3)), ProtoNodeState::Down as i32);
}

/// A member that answers is judged by how far behind it is.
#[test]
fn a_reached_member_is_judged_by_its_lag() {
    assert_eq!(
        proto_state(&status(Some(2), 0)),
        ProtoNodeState::Healthy as i32
    );
    assert_eq!(
        proto_state(&status(Some(2), 1_000)),
        ProtoNodeState::Healthy as i32
    );
    assert_eq!(
        proto_state(&status(Some(2), 1_001)),
        ProtoNodeState::Degraded as i32
    );
    assert_eq!(
        proto_state(&status(Some(2), 50_000)),
        ProtoNodeState::Degraded as i32
    );
    assert_eq!(
        proto_state(&status(Some(2), 50_001)),
        ProtoNodeState::Down as i32
    );
}
