//! Client operation records decided at ordered application.

use std::sync::Arc;

use coordinode_core::graph::stats::counter_delta_operand;
use coordinode_core::txn::operation::{
    BindMode, DEFAULT_ADMISSION_WINDOW, INTENT_DIGEST_LEN, ISSUER_ID_LEN, IntentDigest,
    OperationBinding,
};
use coordinode_core::txn::timestamp::TimestampOracle;
use tempfile::TempDir;

use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};

const ISSUER: IssuerId = IssuerId([0x5a; ISSUER_ID_LEN]);
const COUNTER: &[u8] = b"count:test";

fn config(dir: &TempDir) -> StorageConfig {
    StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )])
}

fn open(dir: &TempDir) -> (StorageEngine, Arc<TimestampOracle>) {
    let oracle = Arc::new(TimestampOracle::new());
    let engine = StorageEngine::open_with_oracle(&config(dir), Arc::clone(&oracle)).unwrap();
    (engine, oracle)
}

fn open_journaled(dir: &TempDir, oracle: &Arc<TimestampOracle>) -> StorageEngine {
    StorageEngine::open_embedded(&config(dir), Arc::clone(oracle)).unwrap()
}

fn op(incarnation: u64, sequence: u64) -> OperationId {
    OperationId {
        issuer: ISSUER,
        incarnation,
        sequence,
    }
}

fn enroll(incarnation: u64) -> Mutation {
    Mutation::Operation(OperationCommand::Enroll {
        issuer: ISSUER,
        incarnation,
        window: DEFAULT_ADMISSION_WINDOW,
    })
}

/// One increment of the test counter, bound to `op` with `intent` as
/// `attempt`.
fn increment(op: OperationId, intent: u8, attempt: u64, mode: BindMode) -> Vec<Mutation> {
    vec![
        Mutation::Merge {
            partition: PartitionId::Counter,
            key: COUNTER.to_vec(),
            operand: counter_delta_operand(1),
        },
        Mutation::Operation(OperationCommand::Bind(OperationBinding {
            op,
            intent: IntentDigest([intent; INTENT_DIGEST_LEN]),
            attempt,
            mode,
        })),
    ]
}

fn counter(engine: &StorageEngine) -> i64 {
    engine
        .get(Partition::Counter, COUNTER)
        .unwrap()
        .map_or(0, |v| crate::engine::merge::decode_counter(&v).unwrap())
}

fn outcome(engine: &StorageEngine, op: &OperationId) -> OperationOutcome {
    operation_outcome(engine, op).unwrap()
}

/// An additive write bound to an operation counts once however many times
/// it is submitted: a second attempt and a different intent apply nothing,
/// and the record names the attempt that committed.
#[test]
fn a_bound_increment_counts_once() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    engine
        .apply_proposal_at(&[enroll(1)], oracle.next().as_raw())
        .unwrap();
    let first = oracle.next().as_raw();
    engine
        .apply_proposal_at(&increment(op(1, 1), 7, 100, BindMode::Commit), first)
        .unwrap();
    for (intent, attempt) in [(7, 101), (8, 102), (7, 100)] {
        engine
            .apply_proposal_at(
                &increment(op(1, 1), intent, attempt, BindMode::Commit),
                oracle.next().as_raw(),
            )
            .unwrap();
    }
    assert_eq!(counter(&engine), 1);
    assert_eq!(
        outcome(&engine, &op(1, 1)),
        OperationOutcome::Committed {
            attempt: 100,
            commit_ts: first
        }
    );
}

/// Without an enrolled incarnation nothing binds, and nothing applies.
#[test]
fn an_unenrolled_operation_applies_nothing() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    engine
        .apply_proposal_at(
            &increment(op(1, 1), 7, 1, BindMode::Commit),
            oracle.next().as_raw(),
        )
        .unwrap();
    assert_eq!(counter(&engine), 0);
    assert_eq!(outcome(&engine, &op(1, 1)), OperationOutcome::Unknown);
}

/// Excluding an operation no attempt bound proves it never takes effect: an
/// attempt arriving after the exclusion applies nothing. Excluding a bound
/// operation changes nothing.
#[test]
fn an_exclusion_fences_a_late_attempt() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    engine
        .apply_proposal_at(&[enroll(1)], oracle.next().as_raw())
        .unwrap();
    let exclude = |op| Mutation::Operation(OperationCommand::Exclude(op));
    engine
        .apply_proposal_at(&[exclude(op(1, 1))], oracle.next().as_raw())
        .unwrap();
    engine
        .apply_proposal_at(
            &increment(op(1, 1), 7, 1, BindMode::Commit),
            oracle.next().as_raw(),
        )
        .unwrap();
    assert_eq!(counter(&engine), 0);
    assert_eq!(outcome(&engine, &op(1, 1)), OperationOutcome::Aborted);

    engine
        .apply_proposal_at(
            &increment(op(1, 2), 7, 2, BindMode::Commit),
            oracle.next().as_raw(),
        )
        .unwrap();
    engine
        .apply_proposal_at(&[exclude(op(1, 2))], oracle.next().as_raw())
        .unwrap();
    assert!(matches!(
        outcome(&engine, &op(1, 2)),
        OperationOutcome::Committed { attempt: 2, .. }
    ));
    assert_eq!(counter(&engine), 1);
}

/// An admitted attempt completes once; another attempt of the admitted
/// operation is refused, and retirement stops in front of it.
#[test]
fn an_admitted_operation_completes_once_and_holds_retirement() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    engine
        .apply_proposal_at(&[enroll(1)], oracle.next().as_raw())
        .unwrap();
    engine
        .apply_proposal_at(
            &[Mutation::Operation(OperationCommand::Bind(
                OperationBinding {
                    op: op(1, 1),
                    intent: IntentDigest([7; INTENT_DIGEST_LEN]),
                    attempt: 9,
                    mode: BindMode::Admit,
                },
            ))],
            oracle.next().as_raw(),
        )
        .unwrap();
    let retire = |below| {
        Mutation::Operation(OperationCommand::Retire {
            issuer: ISSUER,
            incarnation: 1,
            below,
        })
    };
    engine
        .apply_proposal_at(&[retire(5)], oracle.next().as_raw())
        .unwrap();
    assert_eq!(
        outcome(&engine, &op(1, 1)),
        OperationOutcome::Admitted { attempt: 9 },
        "an admitted operation is not retired"
    );
    engine
        .apply_proposal_at(
            &increment(op(1, 1), 7, 10, BindMode::Commit),
            oracle.next().as_raw(),
        )
        .unwrap();
    assert_eq!(counter(&engine), 0, "another attempt is refused");
    engine
        .apply_proposal_at(
            &increment(op(1, 1), 7, 9, BindMode::Commit),
            oracle.next().as_raw(),
        )
        .unwrap();
    assert_eq!(counter(&engine), 1, "the admitted attempt completes");
}

/// An acknowledgement releases one settled outcome above the retired prefix,
/// out of number order, and keeps it settled: the released number reads as
/// retired and binds no attempt again. It cannot release an operation still
/// owed its outcome, nor declare a number nothing is recorded of complete,
/// and retirement later collects the released record with the rest.
#[test]
fn an_acknowledgement_releases_a_settled_outcome_only() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    let apply = |mutations: &[Mutation]| {
        engine
            .apply_proposal_at(mutations, oracle.next().as_raw())
            .unwrap();
    };
    let release = |op| Mutation::Operation(OperationCommand::Release(op));
    apply(&[enroll(1)]);
    // 1 admitted and not complete, 2 a hole, 3 committed, 4 excluded.
    apply(&[Mutation::Operation(OperationCommand::Bind(
        OperationBinding {
            op: op(1, 1),
            intent: IntentDigest([7; INTENT_DIGEST_LEN]),
            attempt: 9,
            mode: BindMode::Admit,
        },
    ))]);
    apply(&increment(op(1, 3), 7, 30, BindMode::Commit));
    apply(&[Mutation::Operation(OperationCommand::Exclude(op(1, 4)))]);

    for sequence in 1..=4 {
        apply(&[release(op(1, sequence))]);
    }
    assert_eq!(
        outcome(&engine, &op(1, 1)),
        OperationOutcome::Admitted { attempt: 9 },
        "an operation still owed its outcome is not released"
    );
    assert_eq!(
        outcome(&engine, &op(1, 2)),
        OperationOutcome::Unknown,
        "a number nothing is recorded of is not declared complete"
    );
    assert_eq!(outcome(&engine, &op(1, 3)), OperationOutcome::Retired);
    assert_eq!(outcome(&engine, &op(1, 4)), OperationOutcome::Retired);

    // The released numbers stay settled: no attempt of them applies.
    apply(&increment(op(1, 3), 7, 31, BindMode::Commit));
    apply(&increment(op(1, 4), 7, 40, BindMode::Commit));
    assert_eq!(counter(&engine), 1);

    // Retirement stops in front of the admitted operation; once it is
    // resolved, the prefix passes the hole and collects the released records.
    let retire = |below| {
        Mutation::Operation(OperationCommand::Retire {
            issuer: ISSUER,
            incarnation: 1,
            below,
        })
    };
    apply(&[retire(5)]);
    assert!(operation_record(&engine, &op(1, 3)).unwrap().is_some());
    apply(&[Mutation::Operation(OperationCommand::Exclude(op(1, 1)))]);
    apply(&[retire(5)]);
    for sequence in 1..=4 {
        assert_eq!(operation_record(&engine, &op(1, sequence)).unwrap(), None);
        assert_eq!(
            outcome(&engine, &op(1, sequence)),
            OperationOutcome::Retired
        );
    }
}

/// A restored image admits no new attempt under a descriptor issued before
/// the restore, and keeps the fences of the history it replaced: a recorded
/// outcome still resolves, an unrecorded number of an image incarnation is
/// aborted, and an incarnation or retired prefix only the replaced history
/// knew is not reopened.
#[test]
fn a_restored_image_is_fenced_against_old_descriptors() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    let apply = |mutations: &[Mutation]| {
        engine
            .apply_proposal_at(mutations, oracle.next().as_raw())
            .unwrap();
    };
    apply(&[enroll(1)]);
    apply(&increment(op(1, 1), 7, 10, BindMode::Commit));
    // What the replaced history knew: incarnation 1 retired below 5, and an
    // incarnation 2 the image never saw.
    let surviving = [
        (
            ISSUER,
            1,
            IncarnationRecord {
                open: true,
                retired_below: 5,
                window: DEFAULT_ADMISSION_WINDOW,
            },
        ),
        (
            ISSUER,
            2,
            IncarnationRecord {
                open: true,
                retired_below: 1,
                window: DEFAULT_ADMISSION_WINDOW,
            },
        ),
    ];
    fence_restored(&engine, &surviving).unwrap();

    let fences = admission_fences(&engine).unwrap();
    assert_eq!(fences.len(), 2);
    assert!(fences.iter().all(|(_, _, record)| !record.open));
    assert_eq!(
        incarnation_record(&engine, &ISSUER, 1)
            .unwrap()
            .unwrap()
            .retired_below,
        5,
        "the replaced history's retirement is kept"
    );
    assert!(matches!(
        outcome(&engine, &op(1, 1)),
        OperationOutcome::Committed { attempt: 10, .. }
    ));
    assert_eq!(outcome(&engine, &op(1, 6)), OperationOutcome::Aborted);
    assert_eq!(outcome(&engine, &op(2, 1)), OperationOutcome::Aborted);
    apply(&increment(op(1, 6), 7, 11, BindMode::Commit));
    apply(&increment(op(2, 1), 7, 12, BindMode::Commit));
    assert_eq!(counter(&engine), 1, "no old descriptor admits an attempt");
}

/// A provisional write whose payload was lost before its drain stays
/// admitted until the host resolves it: excluding it then proves it never
/// takes effect, and the drain of the admitted attempt arriving after the
/// exclusion applies nothing.
#[test]
fn an_exclusion_fences_the_late_drain_of_an_admitted_operation() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    engine
        .apply_proposal_at(&[enroll(1)], oracle.next().as_raw())
        .unwrap();
    engine
        .apply_proposal_at(
            &[Mutation::Operation(OperationCommand::Bind(
                OperationBinding {
                    op: op(1, 1),
                    intent: IntentDigest([7; INTENT_DIGEST_LEN]),
                    attempt: 9,
                    mode: BindMode::Admit,
                },
            ))],
            oracle.next().as_raw(),
        )
        .unwrap();
    engine
        .apply_proposal_at(
            &[Mutation::Operation(OperationCommand::Exclude(op(1, 1)))],
            oracle.next().as_raw(),
        )
        .unwrap();
    assert_eq!(outcome(&engine, &op(1, 1)), OperationOutcome::Aborted);
    engine
        .apply_proposal_at(
            &increment(op(1, 1), 7, 9, BindMode::Commit),
            oracle.next().as_raw(),
        )
        .unwrap();
    assert_eq!(counter(&engine), 0, "the late drain applies nothing");
    assert_eq!(outcome(&engine, &op(1, 1)), OperationOutcome::Aborted);
}

/// Retiring releases the records below the bound, numbers never used
/// included, and refuses every one of them afterwards: a lookup reports them
/// retired, never unknown, and a late attempt applies nothing.
#[test]
fn retirement_releases_records_and_refuses_retired_numbers() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    engine
        .apply_proposal_at(&[enroll(1)], oracle.next().as_raw())
        .unwrap();
    engine
        .apply_proposal_at(
            &increment(op(1, 1), 7, 1, BindMode::Commit),
            oracle.next().as_raw(),
        )
        .unwrap();
    engine
        .apply_proposal_at(
            &[Mutation::Operation(OperationCommand::Retire {
                issuer: ISSUER,
                incarnation: 1,
                below: 4,
            })],
            oracle.next().as_raw(),
        )
        .unwrap();
    assert_eq!(operation_record(&engine, &op(1, 1)).unwrap(), None);
    for sequence in 1..4 {
        assert_eq!(
            outcome(&engine, &op(1, sequence)),
            OperationOutcome::Retired
        );
    }
    engine
        .apply_proposal_at(
            &increment(op(1, 3), 7, 3, BindMode::Commit),
            oracle.next().as_raw(),
        )
        .unwrap();
    assert_eq!(counter(&engine), 1, "a retired number binds nothing");
    assert_eq!(
        incarnation_record(&engine, &ISSUER, 1)
            .unwrap()
            .map(|r| r.retired_below),
        Some(4)
    );
}

/// A new incarnation closes the old one: the old one binds nothing new, and
/// an operation of it with no record is reported aborted. An incarnation at
/// or below one the issuer has is refused.
#[test]
fn a_new_incarnation_fences_the_old_one() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    for incarnation in [1, 2, 1] {
        engine
            .apply_proposal_at(&[enroll(incarnation)], oracle.next().as_raw())
            .unwrap();
    }
    let closed = incarnation_record(&engine, &ISSUER, 1).unwrap().unwrap();
    assert!(!closed.open);
    assert!(
        incarnation_record(&engine, &ISSUER, 2)
            .unwrap()
            .unwrap()
            .open
    );
    engine
        .apply_proposal_at(
            &increment(op(1, 1), 7, 1, BindMode::Commit),
            oracle.next().as_raw(),
        )
        .unwrap();
    assert_eq!(counter(&engine), 0);
    assert_eq!(outcome(&engine, &op(1, 1)), OperationOutcome::Aborted);
}

/// The same attempt in two log entries (a retransmission) applies once; the
/// unit it refused is marked for raw readers of the log, and the unit that
/// applied, delivered again, writes nothing twice.
#[test]
fn a_retransmitted_unit_in_the_log_applies_once() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    let keep = |_| false;
    engine
        .apply_raft_proposal(&[enroll(1)], oracle.next().as_raw(), 1, 0, keep)
        .unwrap();
    let unit = increment(op(1, 1), 7, 5, BindMode::Commit);
    let ts = oracle.next().as_raw();
    engine.apply_raft_proposal(&unit, ts, 2, 0, keep).unwrap();
    engine.apply_raft_proposal(&unit, ts, 3, 0, keep).unwrap();
    assert_eq!(counter(&engine), 1);
    let at = |index| LogPosition { index, sub: 0 };
    assert!(!unit_skipped(&engine, at(2)).unwrap());
    assert!(unit_skipped(&engine, at(3)).unwrap());
    // Delivered again: the applying unit's partitions already hold it.
    engine
        .apply_raft_proposal(&unit, ts, 2, 0, |part| part == Partition::Counter)
        .unwrap();
    assert_eq!(counter(&engine), 1);
    assert!(!unit_skipped(&engine, at(2)).unwrap());
}

/// An operation command anywhere but last in its unit, or beside another
/// command, is refused before anything applies.
#[test]
fn a_misplaced_operation_command_is_refused() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    let mut reversed = increment(op(1, 1), 7, 1, BindMode::Commit);
    reversed.reverse();
    assert!(
        engine
            .apply_proposal_at(&reversed, oracle.next().as_raw())
            .is_err()
    );
    let mut two = increment(op(1, 1), 7, 1, BindMode::Commit);
    two.insert(0, enroll(1));
    assert!(
        engine
            .apply_proposal_at(&two, oracle.next().as_raw())
            .is_err()
    );
    assert_eq!(counter(&engine), 0);
}

/// On the journalled path the journal records what was decided, so a reopen
/// replays the increment once and keeps the record, without deciding again.
#[test]
fn a_journalled_engine_replays_decided_effects_once() {
    let dir = TempDir::new().unwrap();
    let oracle = Arc::new(TimestampOracle::new());
    {
        let engine = open_journaled(&dir, &oracle);
        engine
            .commit_journaled(&[enroll(1)], oracle.next().as_raw())
            .unwrap();
        for attempt in [1, 2] {
            engine
                .commit_journaled(
                    &increment(op(1, 1), 7, attempt, BindMode::Commit),
                    oracle.next().as_raw(),
                )
                .unwrap();
        }
        assert_eq!(counter(&engine), 1);
        // Dropped without a flush: reopening replays the journal.
    }
    let engine = open_journaled(&dir, &oracle);
    assert_eq!(counter(&engine), 1);
    assert!(matches!(
        outcome(&engine, &op(1, 1)),
        OperationOutcome::Committed { attempt: 1, .. }
    ));
}
