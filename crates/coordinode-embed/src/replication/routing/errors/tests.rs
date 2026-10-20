use super::*;

use coordinode_core::schema::definition::PropertyType;

fn txid() -> TransactionId {
    TransactionId::from_raw([0x2a; 16])
}

/// One error of every variant an executing member can answer with, each with
/// its fields set to values a mix-up would show.
fn every_variant() -> Vec<DatabaseError> {
    use DatabaseError as E;
    vec![
        E::Parse(ParseError::Syntax {
            message: "expected RETURN".into(),
            span: ErrorSpan {
                start: 3,
                end: 9,
                line: 2,
                col: 4,
            },
        }),
        E::Parse(ParseError::Invalid("no pattern".into())),
        E::Plan("query must contain at least one clause".into()),
        E::Semantic("variable `x` not defined".into()),
        E::SessionSetting {
            setting: "vector_consistency".into(),
            reason: "unknown mode".into(),
        },
        E::Arithmetic(EvalError::DivideByZero),
        E::Arithmetic(EvalError::LongOverflow),
        E::Arithmetic(EvalError::UnknownFunction("lenght".into())),
        E::Procedure(ProcedureError::Unknown {
            procedure: "db.nosuch".into(),
        }),
        E::Procedure(ProcedureError::TooManyArguments {
            procedure: "db.p".into(),
            declared: 2,
            given: 5,
        }),
        E::Procedure(ProcedureError::MissingArgument {
            procedure: "db.p".into(),
            argument: "limit".into(),
        }),
        E::Procedure(ProcedureError::ArgumentType {
            procedure: "db.p".into(),
            argument: "limit".into(),
            expected: "INTEGER".into(),
            found: "STRING".into(),
        }),
        E::Procedure(ProcedureError::InvalidArgument {
            procedure: "db.p".into(),
            argument: "limit".into(),
            reason: "negative".into(),
        }),
        E::Procedure(ProcedureError::UnknownOutput {
            procedure: "db.p".into(),
            column: "nope".into(),
        }),
        E::Procedure(ProcedureError::YieldRequired {
            procedure: "db.p".into(),
        }),
        E::Procedure(ProcedureError::OutputShape {
            procedure: "db.p".into(),
            declared: 3,
            found: 1,
        }),
        E::Procedure(ProcedureError::DuplicateName {
            procedure: "db.p".into(),
        }),
        E::Unsupported("MERGE on a table".into()),
        E::Budget(BudgetStop::Memory {
            requested: 4096,
            used: 1_000_000,
            limit: 1 << 20,
        }),
        E::Budget(BudgetStop::Deadline),
        E::Budget(BudgetStop::Cancelled),
        E::IndexUnresolved {
            label: "U".into(),
            property: "email".into(),
            operator: "Aggregate".into(),
        },
        E::CascadeOverflow {
            current: 9,
            limit: 8,
            chain: vec!["a".into(), "b".into()],
        },
        E::CascadeFanoutOverflow {
            trigger: "t".into(),
            count: 70,
            limit: 64,
        },
        E::SchemaViolation("undeclared property `x`".into()),
        E::DuplicateKey {
            table: "Account".into(),
            key: "7".into(),
            element_id: "0000000000007".into(),
        },
        E::UniqueViolation {
            index: "user_email".into(),
            property: "email".into(),
            value: "\"a@b\"".into(),
            element_id: "0000000000003".into(),
        },
        E::UniquenessUnresolved {
            index: "user_email".into(),
            limit: 100_000,
        },
        E::ConstraintViolation {
            constraint: "item_qty".into(),
            kind: Box::new(ConstraintKind::Type(PropertyType::Int)),
            label: "Item".into(),
            property: "qty".into(),
            element_id: "0000000000007".into(),
        },
        E::KeyImmutable {
            table: "Account".into(),
            column: "id".into(),
        },
        E::CatalogObjectExists {
            object: CatalogObject::Label,
            name: "User".into(),
        },
        E::CatalogObjectDiffers {
            object: CatalogObject::Index,
            name: "i".into(),
            revision: 4,
        },
        E::CatalogRevisionMismatch {
            object: CatalogObject::Constraint,
            name: "c".into(),
            expected: 2,
            current: None,
        },
        E::CatalogObjectMissing {
            object: CatalogObject::Trigger,
            name: "t".into(),
        },
        E::CatalogObjectMissing {
            object: CatalogObject::EdgeType,
            name: "KNOWS".into(),
        },
        E::CatalogRefused("still validating".into()),
        E::UnknownTransaction(txid()),
        E::TransactionIdInUse(txid()),
        E::TransactionBusy(txid()),
        E::OperationRequired(txid()),
        E::OperationDiffers(txid()),
        E::TransactionCommittedOnTimeout(txid()),
        E::TransactionOutcomeUnknown(txid()),
        E::TransactionAborted {
            id: txid(),
            end: TransactionEnd::StatementFailed,
        },
        E::TransactionAborted {
            id: txid(),
            end: TransactionEnd::IdleExpired,
        },
        E::TransactionAborted {
            id: txid(),
            end: TransactionEnd::IdleCommitted,
        },
        E::TransactionAborted {
            id: txid(),
            end: TransactionEnd::IdleCommitRefused,
        },
        E::TransactionAborted {
            id: txid(),
            end: TransactionEnd::IdleCommitUnknown,
        },
        E::TransactionAborted {
            id: txid(),
            end: TransactionEnd::RolledBack,
        },
        E::TransactionAborted {
            id: txid(),
            end: TransactionEnd::StreamLost,
        },
        E::TransactionConflict {
            transaction: Some(txid()),
            message: "key k".into(),
        },
        E::TransactionConflict {
            transaction: None,
            message: "key k".into(),
        },
        E::InvariantRefused {
            transaction: Some(txid()),
            reason: "acyclic".into(),
        },
        E::InvariantRefused {
            transaction: None,
            reason: "acyclic".into(),
        },
        E::RevisionMismatch {
            transaction: Some(txid()),
            expected: Some(4),
            current: None,
        },
        E::RevisionMismatch {
            transaction: None,
            expected: None,
            current: Some(5),
        },
        E::TransactionTooLarge {
            id: txid(),
            buffered: 300,
            limit: 200,
        },
        E::WriteBackpressure,
        E::CapacityExhausted {
            endpoint_id: "hot".into(),
            used_bytes: 10,
            hard_limit_bytes: 8,
        },
        E::StorageFull {
            path: "/data".into(),
            available_bytes: 1,
            min_free_bytes: 2,
        },
        E::NotLeader { leader_id: Some(2) },
        E::NotLeader { leader_id: None },
        E::Mismatched(Mismatch {
            own: VersionPair {
                engine: 3,
                host_epoch: 1,
            },
            group: VersionPair {
                engine: 4,
                host_epoch: 0,
            },
            behind: true,
            leader: Some((2, "10.0.0.2:7080".into())),
            as_of: 88,
        }),
        E::Mismatched(Mismatch {
            own: VersionPair {
                engine: 4,
                host_epoch: 0,
            },
            group: VersionPair {
                engine: 3,
                host_epoch: 0,
            },
            behind: false,
            leader: None,
            as_of: 0,
        }),
        E::SettingsRefused("readConcern.at_timestamp is only valid with level=SNAPSHOT".into()),
        E::ReadFence {
            kind: ReadFenceFailure::Role,
            message: "not the leader".into(),
        },
        E::ReadFence {
            kind: ReadFenceFailure::Stale,
            message: "behind".into(),
        },
        E::ReadFence {
            kind: ReadFenceFailure::Timeout,
            message: "timed out".into(),
        },
        E::OutsideRetention {
            requested: 5,
            oldest_readable: 99,
        },
        E::ReadNotReady {
            requested: 50,
            ready_through: Some(40),
        },
        E::ReadNotReady {
            requested: 50,
            ready_through: None,
        },
        E::OperationSettled {
            op: OperationId {
                issuer: IssuerId([5; 16]),
                incarnation: 2,
                sequence: 9,
            },
            outcome: OperationOutcome::Committed {
                attempt: 3,
                commit_ts: 70,
            },
            intent_differs: true,
        },
        E::OperationSettled {
            op: OperationId {
                issuer: IssuerId([6; 16]),
                incarnation: 1,
                sequence: 1,
            },
            outcome: OperationOutcome::Aborted,
            intent_differs: false,
        },
        E::ForeignOperation {
            descriptor: DeploymentId([9; 16]),
        },
        E::Other("the build of index 'i' was cancelled".into()),
    ]
}

/// Every variant an executing member answers with reaches the ingress as the
/// same variant with the same fields.
#[test]
fn every_variant_round_trips_with_its_fields() {
    for err in every_variant() {
        let sent = format!("{err:?}");
        let rendered = err.to_string();
        let bytes = encode(err);
        assert!(bytes.len() <= MAX_ROUTED_ERROR);
        let back = decode(&bytes).unwrap_or_else(|e| panic!("{sent}: {e}"));
        assert_eq!(format!("{back:?}"), sent);
        assert_eq!(back.to_string(), rendered);
    }
}

/// An unclassified storage failure crosses with its message; its cause stays
/// with the member that raised it.
#[test]
fn a_storage_failure_crosses_without_its_cause() {
    let err = DatabaseError::from(coordinode_storage::error::StorageError::Io("disk".into()));
    let rendered = err.to_string();
    let back = decode(&encode(err)).expect("decoded");
    assert!(back.storage_cause().is_none());
    assert!(matches!(back, DatabaseError::Storage(_)));
    assert_eq!(back.to_string(), rendered);
}

/// Free text past the bound is cut; the fields a caller decides on cross
/// whole, and the error stays within its bound.
#[test]
fn long_text_is_cut_and_fields_are_kept() {
    let err = DatabaseError::TransactionConflict {
        transaction: Some(txid()),
        message: "é".repeat(MAX_ROUTED_ERROR),
    };
    let bytes = encode(err);
    assert!(bytes.len() <= MAX_ROUTED_ERROR);
    match decode(&bytes).expect("decoded") {
        DatabaseError::TransactionConflict {
            transaction,
            message,
        } => {
            assert_eq!(transaction, Some(txid()));
            assert!(message.len() <= TEXT_LIMIT, "{}", message.len());
            assert!(message.chars().all(|c| c == 'é'), "cut at a character");
        }
        other => panic!("{other:?}"),
    }
}

/// An error whose identifying fields alone exceed the bound is sent as an
/// internal failure, which claims nothing about what applied.
#[test]
fn an_error_too_large_to_carry_is_internal() {
    let err = DatabaseError::CascadeOverflow {
        current: 1,
        limit: 1,
        chain: vec!["t".repeat(1 << 10); 128],
    };
    let bytes = encode(err);
    assert!(bytes.len() <= MAX_ROUTED_ERROR);
    assert!(matches!(
        decode(&bytes).expect("decoded"),
        DatabaseError::Other(_)
    ));
}

/// Bytes past the bound are refused before they are decoded.
#[test]
fn bytes_past_the_bound_are_refused_unread() {
    let err = decode(&vec![0; MAX_ROUTED_ERROR + 1]).expect_err("refused");
    assert!(err.contains("bound"), "{err}");
}

/// A case, a value or a field this member does not know is unreadable, never
/// read as some refusal.
#[test]
fn what_this_member_does_not_know_is_unreadable() {
    let unknown_case = wire::RoutedError { kind: None }.encode_to_vec();
    assert!(decode(&unknown_case).is_err());

    let unknown_end = wire::RoutedError {
        kind: Some(Kind::TransactionAborted(wire::TransactionAborted {
            transaction_id: txid().to_bytes().to_vec(),
            end: 99,
        })),
    }
    .encode_to_vec();
    assert!(decode(&unknown_end).is_err());

    let short_id = wire::RoutedError {
        kind: Some(Kind::UnknownTransaction(wire::TransactionRef {
            transaction_id: vec![1; 3],
        })),
    }
    .encode_to_vec();
    assert!(decode(&short_id).is_err());

    let unknown_kind = wire::RoutedError {
        kind: Some(Kind::ConstraintViolation(wire::ConstraintViolation {
            kind: vec![0xc1],
            ..wire::ConstraintViolation::default()
        })),
    }
    .encode_to_vec();
    assert!(decode(&unknown_kind).is_err());

    assert!(decode(&[0xff, 0xff, 0xff]).is_err());
}
