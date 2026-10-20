use super::*;

/// The statement and the interactive commit report one refusal as one
/// variant: a caller branches on the conflict, never on which path found it.
#[test]
fn a_statement_conflict_is_the_transaction_conflict_variant() {
    let err = DatabaseError::from(ExecutionError::Conflict("key k".into()));
    let DatabaseError::TransactionConflict {
        transaction,
        message,
    } = &err
    else {
        panic!("{err:?}");
    };
    assert_eq!(*transaction, None);
    assert_eq!(message, "key k");
    assert_eq!(err.to_string(), "write conflict: key k");
}

/// A version mismatch keeps both versions as fields, absent ones absent.
#[test]
fn a_statement_revision_mismatch_keeps_its_versions() {
    let err = DatabaseError::from(ExecutionError::RevisionMismatch {
        expected: Some(3),
        current: None,
    });
    assert!(matches!(
        err,
        DatabaseError::RevisionMismatch {
            transaction: None,
            expected: Some(3),
            current: None,
        }
    ));
}

/// Capacity exhaustion is the same variant whether the engine raised it
/// directly or the executor carried it from the proposal pipeline.
#[test]
fn capacity_exhaustion_is_one_variant_on_every_path() {
    let storage = || coordinode_storage::error::StorageError::CapacityExhausted {
        endpoint_id: "hot".into(),
        used_bytes: 10,
        hard_limit_bytes: 8,
    };
    for err in [
        DatabaseError::from(storage()),
        DatabaseError::from(ExecutionError::Storage(storage())),
        DatabaseError::from(ExecutionError::Modality(StoreError::Storage(storage()))),
    ] {
        assert!(
            matches!(
                &err,
                DatabaseError::CapacityExhausted {
                    endpoint_id,
                    used_bytes: 10,
                    hard_limit_bytes: 8,
                } if endpoint_id == "hot"
            ),
            "{err:?}"
        );
    }
}

/// A storage read below the retention watermark names the oldest readable
/// timestamp, one below the watermark's snapshot.
#[test]
fn a_storage_retention_refusal_names_the_oldest_readable_timestamp() {
    let err = DatabaseError::from(
        coordinode_storage::error::StorageError::SnapshotOutsideRetention {
            snapshot: 5,
            watermark: 9,
        },
    );
    assert!(matches!(
        err,
        DatabaseError::OutsideRetention {
            requested: 4,
            oldest_readable: 8,
        }
    ));
}

/// A storage failure no variant classifies keeps the engine's error as its
/// local cause, and a reported one has none: the message is the same.
#[test]
fn an_unclassified_storage_failure_keeps_its_cause_only_locally() {
    let err = DatabaseError::from(coordinode_storage::error::StorageError::Io("disk".into()));
    assert!(matches!(
        err.storage_cause(),
        Some(coordinode_storage::error::StorageError::Io(m)) if m == "disk"
    ));
    let DatabaseError::Storage(failure) = &err else {
        panic!("{err:?}");
    };
    let reported = DatabaseError::Storage(StorageFailure::reported(failure.message().to_owned()));
    assert!(reported.storage_cause().is_none());
    assert_eq!(reported.to_string(), err.to_string());
}

/// A spent budget surfaces as the budget variant from the executor and from a
/// store read alike.
#[test]
fn a_spent_budget_is_one_variant_on_every_path() {
    for err in [
        DatabaseError::from(ExecutionError::Budget(BudgetStop::Deadline)),
        DatabaseError::from(ExecutionError::Modality(StoreError::Budget(
            BudgetStop::Deadline,
        ))),
    ] {
        assert!(
            matches!(err, DatabaseError::Budget(BudgetStop::Deadline)),
            "{err:?}"
        );
    }
}
