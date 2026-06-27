use super::*;
use crate::proto::session::{Begin, Cancel, Commit, Execute, Rollback};

#[test]
fn to_op_maps_execute_with_handles() {
    let frame = ClientFrame {
        request_id: 1,
        op: Some(client_frame::Op::Execute(Execute {
            query: "RETURN 1".to_string(),
            parameters: Default::default(),
            txid: 7,
            nonce: 3,
        })),
    };
    match to_op(frame) {
        Some(SessionOp::Execute {
            query, txid, nonce, ..
        }) => {
            assert_eq!(query, "RETURN 1");
            assert_eq!(txid, 7);
            assert_eq!(nonce, 3);
        }
        other => panic!("expected Execute op, got {other:?}"),
    }
}

#[test]
fn to_op_maps_begin_ordering_with_unspecified_defaulting_to_ordered() {
    let unordered = ClientFrame {
        request_id: 1,
        op: Some(client_frame::Op::Begin(Begin {
            ordering: ProtoOrdering::Unordered as i32,
            drain_timeout_ms: 50,
        })),
    };
    match to_op(unordered) {
        Some(SessionOp::Begin {
            ordering,
            drain_timeout_ms,
        }) => {
            assert_eq!(ordering, CoreOrdering::Unordered);
            assert_eq!(drain_timeout_ms, 50);
        }
        other => panic!("expected Begin op, got {other:?}"),
    }
    let unspecified = ClientFrame {
        request_id: 1,
        op: Some(client_frame::Op::Begin(Begin {
            ordering: 0,
            drain_timeout_ms: 0,
        })),
    };
    assert!(matches!(
        to_op(unspecified),
        Some(SessionOp::Begin {
            ordering: CoreOrdering::Ordered,
            ..
        })
    ));
}

#[test]
fn to_op_maps_commit_rollback_cancel() {
    let commit = ClientFrame {
        request_id: 1,
        op: Some(client_frame::Op::Commit(Commit {
            txid: 4,
            last_nonce: 9,
        })),
    };
    assert!(matches!(
        to_op(commit),
        Some(SessionOp::Commit {
            txid: 4,
            last_nonce: 9
        })
    ));
    let rollback = ClientFrame {
        request_id: 1,
        op: Some(client_frame::Op::Rollback(Rollback { txid: 4 })),
    };
    assert!(matches!(
        to_op(rollback),
        Some(SessionOp::Rollback { txid: 4 })
    ));
    let cancel = ClientFrame {
        request_id: 1,
        op: Some(client_frame::Op::Cancel(Cancel {
            target_request_id: 8,
        })),
    };
    assert!(matches!(
        to_op(cancel),
        Some(SessionOp::Cancel {
            target_request_id: 8
        })
    ));
}

#[test]
fn to_op_returns_none_for_a_frame_with_no_op() {
    assert!(to_op(ClientFrame {
        request_id: 9,
        op: None
    })
    .is_none());
}

#[test]
fn event_to_frame_tags_the_request_id_and_maps_each_event() {
    let begun = event_to_frame(5, SessionEvent::Begun { txid: 2 });
    assert_eq!(begun.request_id, 5);
    assert!(matches!(begun.event, Some(Event::Begun(Begun { txid: 2 }))));

    let open = event_to_frame(
        6,
        SessionEvent::CursorOpen {
            columns: vec!["c".to_string()],
        },
    );
    match open.event {
        Some(Event::CursorOpen(CursorOpen { columns })) => {
            assert_eq!(columns, vec!["c".to_string()]);
        }
        other => panic!("expected CursorOpen, got {other:?}"),
    }

    let error = event_to_frame(
        7,
        SessionEvent::Error {
            code: ErrorCode::InvalidArgument,
            message: "bad".to_string(),
        },
    );
    match error.event {
        Some(Event::Error(e)) => assert_eq!(e.code, Code::InvalidArgument as u32),
        other => panic!("expected Error, got {other:?}"),
    }
}

#[test]
fn to_op_converts_execute_parameters_to_engine_values() {
    use crate::proto::common::{property_value, PropertyValue};
    use coordinode_core::graph::types::Value;

    let mut parameters = std::collections::HashMap::new();
    parameters.insert(
        "n".to_string(),
        PropertyValue {
            value: Some(property_value::Value::IntValue(5)),
        },
    );
    let frame = ClientFrame {
        request_id: 1,
        op: Some(client_frame::Op::Execute(Execute {
            query: "RETURN $n".to_string(),
            parameters,
            txid: 0,
            nonce: 0,
        })),
    };
    match to_op(frame) {
        Some(SessionOp::Execute { params, .. }) => {
            assert_eq!(params.get("n"), Some(&Value::Int(5)));
        }
        other => panic!("expected Execute op, got {other:?}"),
    }
}

#[test]
fn event_to_frame_maps_rows_through_value_conversion() {
    use coordinode_core::graph::types::Value;

    let frame = event_to_frame(
        3,
        SessionEvent::Rows {
            rows: vec![vec![Value::Int(7), Value::String("x".to_string())]],
        },
    );
    match frame.event {
        Some(Event::Rows(RowBatch { rows })) => {
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].values.len(), 2);
            // The engine values round-trip back through the inverse converter.
            assert_eq!(proto_to_value_pub(&rows[0].values[0]), Value::Int(7));
            assert_eq!(
                proto_to_value_pub(&rows[0].values[1]),
                Value::String("x".to_string())
            );
        }
        other => panic!("expected Rows, got {other:?}"),
    }
}
