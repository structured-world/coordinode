use super::*;
use crate::proto::session::{Begin, Cancel, Commit, Execute, Rollback, SourceLocation};

/// A session whose metadata names no application.
fn anonymous() -> ClientApp {
    ClientApp::default()
}

#[test]
fn to_op_maps_execute_with_handles() {
    let frame = ClientFrame {
        request_id: 1,
        op: Some(client_frame::Op::Execute(Execute {
            query: "RETURN 1".to_string(),
            parameters: Default::default(),
            txid: 7,
            nonce: 3,
            ..Default::default()
        })),
    };
    match to_op(frame.op, &anonymous()) {
        Ok(SessionOp::Execute {
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
    match to_op(unordered.op, &anonymous()) {
        Ok(SessionOp::Begin {
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
        to_op(unspecified.op, &anonymous()),
        Ok(SessionOp::Begin {
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
        to_op(commit.op, &anonymous()),
        Ok(SessionOp::Commit {
            txid: 4,
            last_nonce: 9
        })
    ));
    let rollback = ClientFrame {
        request_id: 1,
        op: Some(client_frame::Op::Rollback(Rollback { txid: 4 })),
    };
    assert!(matches!(
        to_op(rollback.op, &anonymous()),
        Ok(SessionOp::Rollback { txid: 4 })
    ));
    let cancel = ClientFrame {
        request_id: 1,
        op: Some(client_frame::Op::Cancel(Cancel {
            target_request_id: 8,
        })),
    };
    assert!(matches!(
        to_op(cancel.op, &anonymous()),
        Ok(SessionOp::Cancel {
            target_request_id: 8
        })
    ));
}

#[test]
fn to_op_refuses_a_frame_with_no_op() {
    assert!(to_op(None, &anonymous()).is_err());
}

/// A Configure carrying a write concern the server cannot honour is refused
/// at the frame boundary, with the message naming the offending combination;
/// one it can honour lands typed in the settings.
#[test]
fn to_op_configure_maps_or_refuses_the_write_concern() {
    use coordinode_core::txn::write_concern::WriteConcern;

    let configure = |wc: replication::WriteConcern| {
        Some(client_frame::Op::Configure(Configure {
            write_concern: Some(wc),
            ..Default::default()
        }))
    };

    match to_op(
        configure(replication::WriteConcern {
            w: Some(replication::write_concern::W::Acks(1)),
            journal: replication::Journal::Cache as i32,
            timeout_ms: 0,
        }),
        &anonymous(),
    ) {
        Ok(SessionOp::Configure(settings)) => {
            assert_eq!(settings.write_concern, Some(WriteConcern::cache()));
        }
        other => panic!("expected Configure op, got {other:?}"),
    }

    let refused = to_op(
        configure(replication::WriteConcern {
            w: Some(replication::write_concern::W::Acks(2)),
            journal: replication::Journal::Memory as i32,
            timeout_ms: 0,
        }),
        &anonymous(),
    )
    .expect_err("w:2 with j:memory must be refused");
    assert_eq!(refused.code(), Code::InvalidArgument);
    assert!(refused.message().contains("w:2,j:memory"), "got: {refused}");
}

/// A refusal carried through the session core comes out as the status that
/// went in: same code, message and reason. Before, the session answered every
/// engine failure as INTERNAL with the message alone, so a client could not
/// tell a request to send elsewhere from a server fault.
#[test]
fn a_failure_keeps_its_code_and_reason_through_the_session() {
    use tonic_types::StatusExt;

    let original = crate::services::error_details::status_with_reason(
        Code::FailedPrecondition,
        "not the leader".to_string(),
        crate::services::error_details::Reason::NotLeader,
        [("leader_id", "3".to_string())],
    );
    let frame = event_to_frame(9, SessionEvent::Error(failure(&original)));
    let Some(Event::Error(e)) = frame.event else {
        panic!("expected Error, got {:?}", frame.event);
    };
    assert_eq!(e.code, Code::FailedPrecondition as u32);
    assert_eq!(e.message, "not the leader");

    let back = status(failure(&original));
    assert_eq!(back.code(), Code::FailedPrecondition);
    let info = back
        .get_details_error_info()
        .expect("the reason survives the session core");
    assert_eq!(
        info.reason,
        crate::services::error_details::Reason::NotLeader.as_str()
    );
    assert_eq!(
        info.metadata.get("leader_id").map(String::as_str),
        Some("3")
    );
    let canonical = e.status.expect("canonical status");
    assert!(
        canonical
            .details
            .iter()
            .any(|d| d.type_url.ends_with("google.rpc.ErrorInfo")),
        "ErrorInfo rides along on the frame: {:?}",
        canonical.details
    );
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
        SessionEvent::Error(Failure::new(ErrorCode::InvalidArgument, "bad")),
    );
    match error.event {
        Some(Event::Error(e)) => assert_eq!(e.code, Code::InvalidArgument as u32),
        other => panic!("expected Error, got {other:?}"),
    }
}

#[test]
fn to_op_converts_execute_parameters_to_engine_values() {
    use crate::proto::common::{PropertyValue, property_value};
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
            ..Default::default()
        })),
    };
    match to_op(frame.op, &anonymous()) {
        Ok(SessionOp::Execute { params, .. }) => {
            assert_eq!(params.get("n"), Some(&Value::Int(5)));
        }
        other => panic!("expected Execute op, got {other:?}"),
    }
}

/// The settings an Execute names reach the statement, and the parts it leaves
/// unspecified stay unset so the session's apply; its source carries the
/// application the session's metadata named.
#[test]
fn to_op_reads_a_statements_own_settings_and_source() {
    use coordinode_core::txn::write_concern::WriteConcern;

    let client = ClientApp {
        app: "billing".to_string(),
        version: "1.2".to_string(),
    };
    let op = Some(client_frame::Op::Execute(Execute {
        query: "RETURN 1".to_string(),
        read_concern: Some(replication::ReadConcern {
            level: replication::ReadConcernLevel::Majority as i32,
            after_index: 0,
            at_timestamp: 0,
        }),
        write_concern: Some(replication::WriteConcern {
            w: Some(replication::write_concern::W::Acks(1)),
            journal: replication::Journal::Journal as i32,
            timeout_ms: 0,
        }),
        read_preference: Some(replication::ReadPreference::Unspecified as i32),
        source: Some(SourceLocation {
            file: "pay.rs".to_string(),
            line: 12,
            function: "charge".to_string(),
        }),
        ..Default::default()
    }));
    match to_op(op, &client) {
        Ok(SessionOp::Execute {
            settings, source, ..
        }) => {
            assert_eq!(
                settings,
                ConnectionSettings {
                    read_concern: Some(replication::ReadConcernLevel::Majority as u8),
                    write_concern: Some(WriteConcern::w1()),
                    ..ConnectionSettings::default()
                },
                "an unspecified preference and zero positions leave those to the session"
            );
            assert_eq!(
                source,
                Some(StatementSource {
                    file: "pay.rs".to_string(),
                    line: 12,
                    function: "charge".to_string(),
                    app: "billing".to_string(),
                    version: "1.2".to_string(),
                })
            );
        }
        other => panic!("expected Execute op, got {other:?}"),
    }
}

/// A level or preference the server does not know is refused, naming the
/// field, rather than read as LOCAL or PRIMARY: a client asking for a stronger
/// guarantee than the server knows must not be served a weaker one. Applies
/// to a statement and to a Configure alike.
#[test]
fn an_unknown_level_or_preference_is_refused_not_downgraded() {
    let execute = |read_concern, read_preference| {
        Some(client_frame::Op::Execute(Execute {
            query: "RETURN 1".to_string(),
            read_concern,
            read_preference,
            ..Default::default()
        }))
    };
    let unknown_level = Some(replication::ReadConcern {
        level: 99,
        after_index: 0,
        at_timestamp: 0,
    });
    for (op, field) in [
        (execute(unknown_level, None), "read_concern.level"),
        (execute(None, Some(42)), "read_preference"),
        (
            Some(client_frame::Op::Configure(Configure {
                read_concern: unknown_level,
                ..Default::default()
            })),
            "read_concern.level",
        ),
        (
            Some(client_frame::Op::Configure(Configure {
                read_preference: Some(42),
                ..Default::default()
            })),
            "read_preference",
        ),
    ] {
        let refused = to_op(op, &anonymous()).expect_err("an unknown value");
        assert_eq!(refused.code(), Code::InvalidArgument);
        assert!(refused.message().contains("not one this server knows"));
        let details = tonic_types::StatusExt::get_error_details(&refused);
        let violations = details
            .bad_request()
            .expect("BadRequest")
            .field_violations
            .clone();
        assert_eq!(violations[0].field, field);
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
