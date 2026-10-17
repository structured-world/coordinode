use super::*;

const ISSUER: IssuerId = IssuerId([7; ISSUER_ID_LEN]);

fn op(sequence: u64) -> OperationId {
    OperationId {
        issuer: ISSUER,
        incarnation: 2,
        sequence,
    }
}

fn open(retired_below: u64, window: u64) -> IncarnationRecord {
    IncarnationRecord {
        open: true,
        retired_below,
        window,
    }
}

fn binding(sequence: u64, intent: u8, attempt: u64, mode: BindMode) -> OperationBinding {
    OperationBinding {
        op: op(sequence),
        intent: IntentDigest([intent; INTENT_DIGEST_LEN]),
        attempt,
        mode,
    }
}

fn at(index: u64) -> LogPosition {
    LogPosition { index, sub: 0 }
}

/// The first attempt of an operation applies and records its commit; any
/// later unit of it, another attempt or the same attempt retransmitted into
/// a later log entry, applies nothing, whatever it means: one operation has
/// one effect.
#[test]
fn only_the_first_attempt_of_an_operation_applies() {
    let inc = open(1, 10);
    let first = binding(3, 1, 100, BindMode::Commit);
    let BindDecision::Apply(record) = decide_binding(&first, Some(&inc), None, 50, at(7)) else {
        panic!("the first attempt applies");
    };
    assert_eq!(
        record.state,
        OperationState::Committed {
            attempt: 100,
            commit_ts: 50,
            at: at(7),
        }
    );
    for (later, position) in [
        (first, at(8)),
        (binding(3, 1, 101, BindMode::Commit), at(9)),
        (binding(3, 2, 102, BindMode::Commit), at(10)),
        (binding(3, 1, 103, BindMode::Admit), at(11)),
    ] {
        assert_eq!(
            decide_binding(&later, Some(&inc), Some(&record), 50, position),
            BindDecision::Skip,
            "{later:?}"
        );
    }
    // Without a log, a record means the operation is done: nothing applies.
    let local = decide_binding(&first, Some(&inc), None, 50, LogPosition::LOCAL);
    let BindDecision::Apply(local) = local else {
        panic!("the first local attempt applies");
    };
    assert_eq!(
        decide_binding(&first, Some(&inc), Some(&local), 50, LogPosition::LOCAL),
        BindDecision::Skip
    );
}

/// The unit that applied an operation, delivered again (a replay after a
/// crash left some partitions behind it), applies again so those partitions
/// catch up; the record it writes is the one it wrote before.
#[test]
fn the_applying_unit_delivered_again_applies_again() {
    let inc = open(1, 10);
    let first = binding(3, 1, 100, BindMode::Commit);
    let BindDecision::Apply(record) = decide_binding(&first, Some(&inc), None, 50, at(7)) else {
        panic!("the first attempt applies");
    };
    assert_eq!(
        decide_binding(&first, Some(&inc), Some(&record), 50, at(7)),
        BindDecision::Apply(record)
    );
}

/// An admitted attempt completes once, with the same intent; another attempt
/// of the admitted operation is excluded by the admission.
#[test]
fn an_admitted_attempt_and_only_it_completes() {
    let inc = open(1, 10);
    let admit = binding(4, 1, 7, BindMode::Admit);
    let BindDecision::Apply(admitted) = decide_binding(&admit, Some(&inc), None, 10, at(3)) else {
        panic!("admission applies");
    };
    assert_eq!(
        admitted.state,
        OperationState::Admitted {
            attempt: 7,
            at: at(3)
        }
    );
    assert_eq!(
        decide_binding(
            &binding(4, 1, 7, BindMode::Commit),
            Some(&inc),
            Some(&admitted),
            20,
            at(5)
        ),
        BindDecision::Apply(OperationRecord {
            intent: admit.intent,
            state: OperationState::Committed {
                attempt: 7,
                commit_ts: 20,
                at: at(5),
            },
        })
    );
    for other in [
        binding(4, 1, 8, BindMode::Commit),
        binding(4, 2, 7, BindMode::Commit),
    ] {
        assert_eq!(
            decide_binding(&other, Some(&inc), Some(&admitted), 20, at(6)),
            BindDecision::Skip
        );
    }
}

/// An operation binds only inside its incarnation's admission window: not
/// without an enrolled incarnation, not once it is closed, not below the
/// retired prefix and not past the window.
#[test]
fn admission_is_bounded_by_the_incarnation() {
    let attempt = |sequence| binding(sequence, 1, 1, BindMode::Commit);
    let decide = |sequence, inc: Option<&IncarnationRecord>| {
        decide_binding(&attempt(sequence), inc, None, 1, at(1))
    };
    assert_eq!(decide(1, None), BindDecision::Skip, "not enrolled");
    let closed = IncarnationRecord {
        open: false,
        ..open(1, 10)
    };
    assert_eq!(decide(1, Some(&closed)), BindDecision::Skip);
    let inc = open(5, 3);
    assert_eq!(decide(4, Some(&inc)), BindDecision::Skip, "retired");
    assert!(matches!(decide(7, Some(&inc)), BindDecision::Apply(_)));
    assert_eq!(decide(8, Some(&inc)), BindDecision::Skip, "past the window");
}

/// A lookup tells a missing record of a live operation (unknown: an attempt
/// may be on its way) from a retired number and from a closed incarnation,
/// which binds nothing new.
#[test]
fn a_lookup_reports_what_a_missing_record_means() {
    let inc = open(5, 10);
    assert_eq!(
        outcome_of(&op(6), Some(&inc), None),
        OperationOutcome::Unknown
    );
    assert_eq!(
        outcome_of(&op(4), Some(&inc), None),
        OperationOutcome::Retired
    );
    assert_eq!(outcome_of(&op(6), None, None), OperationOutcome::Unknown);
    let closed = IncarnationRecord { open: false, ..inc };
    assert_eq!(
        outcome_of(&op(6), Some(&closed), None),
        OperationOutcome::Aborted
    );
    let excluded = OperationRecord {
        intent: IntentDigest([0; INTENT_DIGEST_LEN]),
        state: OperationState::Excluded,
    };
    assert_eq!(
        outcome_of(&op(6), Some(&inc), Some(&excluded)),
        OperationOutcome::Aborted
    );
    // A released outcome above the retired prefix reads as retired, not as
    // unknown: the number is settled and its outcome is gone.
    let released = OperationRecord {
        intent: IntentDigest([7; INTENT_DIGEST_LEN]),
        state: OperationState::Released,
    };
    assert_eq!(
        outcome_of(&op(6), Some(&inc), Some(&released)),
        OperationOutcome::Retired
    );
}

/// A released operation binds no attempt again, whatever it carries.
#[test]
fn a_released_operation_binds_nothing() {
    let inc = open(1, 10);
    let released = OperationRecord {
        intent: IntentDigest([7; INTENT_DIGEST_LEN]),
        state: OperationState::Released,
    };
    for mode in [BindMode::Commit, BindMode::Admit] {
        assert_eq!(
            decide_binding(
                &binding(3, 7, 4, mode),
                Some(&inc),
                Some(&released),
                20,
                at(9)
            ),
            BindDecision::Skip
        );
    }
}

/// An attempt differs in intent from a recorded one only when the record
/// carries an intent: an operation excluded before any attempt records none.
#[test]
fn intent_differs_only_from_a_recorded_intent() {
    let recorded = OperationRecord {
        intent: IntentDigest([7; INTENT_DIGEST_LEN]),
        state: OperationState::Released,
    };
    assert!(!recorded.intent_differs(&IntentDigest([7; INTENT_DIGEST_LEN])));
    assert!(recorded.intent_differs(&IntentDigest([8; INTENT_DIGEST_LEN])));
    let excluded = OperationRecord {
        intent: IntentDigest([0; INTENT_DIGEST_LEN]),
        state: OperationState::Excluded,
    };
    assert!(!excluded.intent_differs(&IntentDigest([8; INTENT_DIGEST_LEN])));
}

/// Records, keys and descriptors read back what was written, and malformed
/// bytes are refused rather than read as something else.
#[test]
fn encodings_round_trip_and_refuse_malformed_bytes() {
    for state in [
        OperationState::Committed {
            attempt: 9,
            commit_ts: 11,
            at: LogPosition { index: 5, sub: 2 },
        },
        OperationState::Admitted {
            attempt: 3,
            at: at(4),
        },
        OperationState::Excluded,
        OperationState::Released,
    ] {
        let record = OperationRecord {
            intent: IntentDigest([4; INTENT_DIGEST_LEN]),
            state,
        };
        let bytes = record.encode();
        assert_eq!(OperationRecord::decode(&bytes), Some(record));
        assert_eq!(OperationRecord::decode(&bytes[..bytes.len() - 1]), None);
    }
    let inc = open(3, 9);
    assert_eq!(IncarnationRecord::decode(&inc.encode()), Some(inc));
    assert_eq!(IncarnationRecord::decode(&[2; 17]), None);

    let key = operation_key(&op(42));
    assert!(key.starts_with(&incarnation_operations_prefix(&ISSUER, 2)));
    assert_eq!(decode_operation_key(&key), Some(op(42)));
    assert_eq!(
        decode_incarnation_key(&incarnation_key(&ISSUER, 5)),
        Some(5)
    );

    let descriptor = OperationDescriptor {
        op: op(42),
        deployment: DeploymentId([9; 16]),
    };
    let bytes = descriptor.to_bytes();
    assert_eq!(OperationDescriptor::from_bytes(&bytes), Some(descriptor));
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert_eq!(OperationDescriptor::from_bytes(&trailing), None);
    // A descriptor without its domain is not a descriptor: it could be
    // presented to any database.
    assert_eq!(
        OperationDescriptor::from_bytes(&bytes[..bytes.len() - 16]),
        None
    );
    assert_eq!(OperationDescriptor::from_bytes(b"cnop"), None);
}

/// Operation keys sort by number within an incarnation, so a retirement
/// reads them in order.
#[test]
fn operation_keys_sort_by_number() {
    assert!(operation_key(&op(2)) < operation_key(&op(10)));
    assert!(operation_key(&op(255)) < operation_key(&op(256)));
}
