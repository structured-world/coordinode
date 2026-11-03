use super::{ClosureLedger, OPEN_CHECKPOINTS};

/// An open position reads the ceiling of its own prefix, not of the log
/// after it: a read there waits on no later commit.
#[test]
fn an_open_position_reads_its_own_prefix() {
    let mut ledger = ClosureLedger::default();
    ledger.commit(1, 10);
    ledger.commit(2, 20);
    ledger.commit(4, 40);
    assert_eq!(ledger.ceiling_through(1), 10);
    assert_eq!(ledger.ceiling_through(2), 20);
    // Entry 3 carries no commit: its prefix stops at entry 2's.
    assert_eq!(ledger.ceiling_through(3), 20);
    assert_eq!(ledger.ceiling_through(4), 40);
    assert_eq!(ledger.applied_ceiling(), 40);
}

/// A position the bound passed keeps a closed ceiling below the bound,
/// whatever applies after it while a lower commit stays open.
#[test]
fn a_closed_position_stays_below_the_bound() {
    let mut ledger = ClosureLedger::default();
    ledger.commit(1, 10);
    ledger.close(11, 2);
    for index in 3..=10_000 {
        ledger.commit(index, 1_000 + index);
    }
    ledger.close(500, 10_000);
    assert_eq!(ledger.ceiling_through(1), 10);
    assert_eq!(ledger.ceiling_through(2), 10);
    assert!(ledger.ceiling_through(1) < ledger.below());
    // The latest open positions read their own ceilings.
    assert_eq!(ledger.ceiling_through(9_900), 10_900);
}

/// A bound that passes every commit closes the whole log, entries without
/// commits included.
#[test]
fn a_bound_past_every_commit_closes_the_log() {
    let mut ledger = ClosureLedger::default();
    ledger.commit(1, 10);
    ledger.commit(5, 50);
    ledger.close(51, 7);
    assert_eq!(ledger.ceiling_through(7), 50);
    assert_eq!(ledger.ceiling_through(3), 50);
    assert_eq!(ledger.applied_ceiling(), 50);
}

/// Merging the oldest open positions never reports a ceiling below the one
/// a position needs.
#[test]
fn merged_positions_never_read_low() {
    let mut ledger = ClosureLedger::default();
    let positions = u64::try_from(OPEN_CHECKPOINTS).expect("small") * 3;
    for index in 1..=positions {
        ledger.commit(index, index * 10);
    }
    for index in 1..=positions {
        assert!(
            ledger.ceiling_through(index) >= index * 10,
            "position {index} reads {} below its own {}",
            ledger.ceiling_through(index),
            index * 10
        );
    }
    // The latest positions stay exact.
    assert_eq!(ledger.ceiling_through(positions), positions * 10);
}

/// An entry the ledger already accounts for changes nothing when a restarted
/// member applies it again.
#[test]
fn an_entry_applied_again_changes_nothing() {
    let mut ledger = ClosureLedger::default();
    ledger.commit(1, 10);
    ledger.commit(2, 20);
    let before = ledger.clone();
    assert!(!ledger.commit(1, 30));
    assert!(!ledger.close(25, 1));
    assert_eq!(ledger, before);
}

/// The stored form reads back to the same ledger, and a record not in it is
/// refused rather than read as a ledger.
#[test]
fn the_record_round_trips_and_refuses_another_form() {
    let mut ledger = ClosureLedger::default();
    ledger.commit(1, 10);
    ledger.close(11, 1);
    ledger.commit(2, 20);
    ledger.commit(3, 30);
    let stored = ledger.encode();
    assert_eq!(ClosureLedger::decode(&stored).expect("decode"), ledger);
    assert!(ClosureLedger::decode(&11u64.to_be_bytes()).is_err());
    assert!(ClosureLedger::decode(&stored[..stored.len() - 1]).is_err());
}
