use super::*;

fn record(integrity: Integrity, state: CheckState) -> IndexIntegrityRecord {
    let mut record = IndexIntegrityRecord::new(IndexId::from_raw(1), GenerationId::from_raw(2));
    record.integrity = integrity;
    let mut check = IndexCheck::accepted(0);
    check.state = state;
    check.checked = 10;
    check.repaired = 3;
    record.check = Some(check);
    record
}

/// A check still admitted or running has no outcome for a waiter.
#[test]
fn a_running_check_has_no_outcome() {
    assert_eq!(
        terminal_outcome(&record(Integrity::Suspect, CheckState::Accepted)),
        None
    );
    assert_eq!(
        terminal_outcome(&record(
            Integrity::Suspect,
            CheckState::Running { executor: 9 }
        )),
        None
    );
    let mut none = record(Integrity::Suspect, CheckState::Done);
    none.check = None;
    assert_eq!(terminal_outcome(&none), None, "no check was admitted");
}

/// A finished check reports what the record says: verified with its counts,
/// a rebuild when it escalated, or the records breaking the constraint.
#[test]
fn a_finished_check_reports_its_record() {
    assert_eq!(
        terminal_outcome(&record(Integrity::Verified, CheckState::Done)),
        Some(CheckOutcome::Verified {
            checked: 10,
            repaired: 3
        })
    );

    let mut rebuilt = record(Integrity::Suspect, CheckState::Done);
    rebuilt.check.as_mut().expect("check").rebuilt_into = Some(GenerationId::from_raw(5));
    assert_eq!(
        terminal_outcome(&rebuilt),
        Some(CheckOutcome::Rebuilding {
            generation: GenerationId::from_raw(5)
        })
    );

    let mut conflicted = record(Integrity::Suspect, CheckState::Done);
    let duplicate = Mismatch::SourceDuplicate {
        nodes: [1, 2],
        tuple: vec![7],
    };
    conflicted.report(Mismatch::Extra {
        node: 4,
        valid_from: None,
        tuple: vec![8],
    });
    conflicted.report(duplicate.clone());
    assert_eq!(
        terminal_outcome(&conflicted),
        Some(CheckOutcome::SourceConflicts {
            conflicts: vec![duplicate]
        }),
        "only the conflicts are the outcome; other evidence is history"
    );
}

/// Cancelled and failed checks report as such.
#[test]
fn cancelled_and_failed_checks() {
    assert_eq!(
        terminal_outcome(&record(Integrity::Suspect, CheckState::Cancelled)),
        Some(CheckOutcome::Cancelled)
    );
    assert_eq!(
        terminal_outcome(&record(
            Integrity::Suspect,
            CheckState::Failed {
                reason: "disk".into()
            }
        )),
        Some(CheckOutcome::Failed("disk".into()))
    );
}
