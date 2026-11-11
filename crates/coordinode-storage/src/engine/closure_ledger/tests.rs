use std::collections::BTreeMap;

use super::{
    BREAKPOINT_PREFIX, Breakpoint, Breakpoints, CACHED_BREAKPOINTS, ClosureLedger, RETIRE_BATCH,
};
use crate::error::StorageResult;

/// The breakpoint records, as a store holds them.
#[derive(Default)]
struct Held(BTreeMap<u64, u64>);

impl Breakpoints for Held {
    fn predecessor(&self, from: u64, through: u64) -> StorageResult<Option<Breakpoint>> {
        Ok(self
            .0
            .range(from..=through)
            .next_back()
            .map(|(&index, &ceiling)| Breakpoint { index, ceiling }))
    }

    fn following(&self, after: u64, limit: usize) -> StorageResult<Vec<Breakpoint>> {
        Ok(self
            .0
            .range(after.saturating_add(1)..)
            .take(limit)
            .map(|(&index, &ceiling)| Breakpoint { index, ceiling })
            .collect())
    }
}

/// A ledger and its records, driven entry by entry as the apply path does:
/// plan, write the records, install.
#[derive(Default)]
struct Log {
    ledger: ClosureLedger,
    held: Held,
}

impl Log {
    /// Entry `index` with `commits` and the bound `below`, while the store
    /// serves reads from `retained` on.
    fn entry(&mut self, index: u64, commits: &[u64], below: u64, retained: u64) {
        let commits: Vec<(u64, u64)> = commits.iter().map(|&ts| (index, ts)).collect();
        let plan = self
            .ledger
            .plan(&commits, below, index, retained, &self.held)
            .expect("plan");
        for b in plan.added() {
            self.held.0.insert(b.index, b.ceiling);
        }
        for b in plan.retired() {
            self.held.0.remove(&b.index);
        }
        self.ledger.install(&plan);
    }

    fn ceiling(&self, index: u64) -> u64 {
        self.ledger
            .ceiling_through(index, &self.held)
            .expect("ceiling")
    }

    /// The ledger as a restart reads it back.
    fn reopened(&self) -> ClosureLedger {
        let newest = self
            .held
            .0
            .iter()
            .rev()
            .map(|(&index, &ceiling)| Ok(Breakpoint { index, ceiling }));
        ClosureLedger::decode(&self.ledger.encode(), newest, &self.held).expect("decode")
    }
}

/// An open position reads the ceiling of its own prefix, not of the log
/// after it: a read there waits on no later commit.
#[test]
fn an_open_position_reads_its_own_prefix() {
    let mut log = Log::default();
    log.entry(1, &[10], 0, 0);
    log.entry(2, &[20], 0, 0);
    log.entry(4, &[40], 0, 0);
    assert_eq!(log.ceiling(1), 10);
    assert_eq!(log.ceiling(2), 20);
    // Entry 3 carries no commit: its prefix stops at entry 2's.
    assert_eq!(log.ceiling(3), 20);
    assert_eq!(log.ceiling(4), 40);
    assert_eq!(log.ledger.applied_ceiling(), 40);
}

/// A prefix closed after more breakpoints piled up than the cache holds
/// keeps its own ceiling, at every position, not only the first: closing it
/// never makes it need a later commit still open. Before, the oldest open
/// positions merged into the earlier position with the later ceiling, and a
/// read after position 1 waited on commit 20 though commit 10 was closed.
#[test]
fn every_position_keeps_its_own_ceiling_past_the_cache() {
    let mut log = Log::default();
    let positions = u64::try_from(CACHED_BREAKPOINTS).expect("small") * 3 + 1;
    for index in 1..=positions {
        log.entry(index, &[index * 10], 0, 0);
    }
    for k in [1, 2, 3, 100, 256, 257, 500, positions - 1] {
        let mut probe = Log {
            ledger: log.ledger.clone(),
            held: Held(log.held.0.clone()),
        };
        probe.entry(positions + 1, &[], k * 10 + 1, 0);
        assert_eq!(probe.ceiling(k), k * 10, "closed position {k}");
        assert_eq!(
            probe.ceiling(k + 1),
            (k + 1) * 10,
            "open position {}",
            k + 1
        );
    }
    for index in 1..=positions {
        assert_eq!(log.ceiling(index), index * 10, "position {index}");
    }
}

/// Closing positions 1 and 2 (commits 10 and 20) does not let position 1
/// need 20 while the store still serves a read at 10: a retained read at 10
/// after position 1 stays ready. Once the store serves nothing below 25, the
/// two are one prefix, and the records go.
#[test]
fn a_retained_view_keeps_the_distinction_closure_alone_would_drop() {
    let mut log = Log::default();
    log.entry(1, &[10], 0, 10);
    log.entry(2, &[20], 0, 10);
    log.entry(3, &[], 21, 10);
    assert_eq!(log.ceiling(1), 10);
    assert_eq!(log.ceiling(2), 20);
    assert_eq!(log.held.0.len(), 2, "both kept while 10 is served");

    log.entry(4, &[], 21, 25);
    assert!(log.held.0.is_empty(), "nothing below 25 is served any more");
    assert!(log.ceiling(1) < 25 && log.ceiling(1) >= 10);
    assert_eq!(log.ceiling(2), 20);
}

/// Retirement runs in bounded batches, one per entry, and never answers a
/// position below its own ceiling or above what the store serves.
#[test]
fn retirement_runs_in_bounded_batches() {
    let mut log = Log::default();
    let positions = u64::try_from(CACHED_BREAKPOINTS * 4).expect("small");
    for index in 1..=positions {
        log.entry(index, &[index * 10], 0, 0);
    }
    let retained = positions * 10 + 1;
    let mut next = positions + 1;
    while !log.held.0.is_empty() {
        let before = log.held.0.len();
        log.entry(next, &[], retained, retained);
        assert!(before - log.held.0.len() <= RETIRE_BATCH);
        assert!(before > log.held.0.len(), "retirement stalled");
        for index in (1..=positions).step_by(37) {
            let c = log.ceiling(index);
            assert!(c >= index * 10 && c < retained, "position {index}: {c}");
        }
        next += 1;
    }
}

/// A restart reads the head and the newest breakpoints only, and answers
/// every position as before, the ones past the cache by a seek.
#[test]
fn a_reopened_ledger_answers_every_position() {
    let mut log = Log::default();
    let positions = u64::try_from(CACHED_BREAKPOINTS * 2).expect("small");
    for index in 1..=positions {
        log.entry(index, &[index * 10], 0, 0);
    }
    log.entry(positions + 1, &[], 51, 30);
    let reopened = Log {
        ledger: log.reopened(),
        held: Held(log.held.0.clone()),
    };
    for index in 1..=positions + 1 {
        assert_eq!(
            reopened.ceiling(index),
            log.ceiling(index),
            "position {index}"
        );
    }
}

/// Entries without commits and commits below the ceiling raise nothing; an
/// entry the ledger already holds changes nothing when applied again.
#[test]
fn empty_entries_late_stamps_and_replays_change_nothing() {
    let mut log = Log::default();
    log.entry(5, &[50], 0, 0);
    log.entry(6, &[30], 0, 0);
    log.entry(7, &[], 0, 0);
    assert_eq!(log.ceiling(6), 50);
    assert_eq!(log.ceiling(7), 50);
    let held = log.held.0.clone();
    let ledger = log.ledger.clone();
    // A restart applies entries 5 and 6 again, with what they carried.
    log.entry(5, &[50], 0, 0);
    log.entry(6, &[30], 0, 0);
    assert_eq!(log.held.0, held);
    assert_eq!(log.ledger, ledger);
    // An entry before the last one accounted for adds nothing, whatever it
    // carries.
    log.entry(5, &[90], 0, 0);
    assert_eq!(log.held.0, held);
    assert_eq!(log.ledger, ledger);
}

/// The head record reads back, and a record of another form is refused
/// rather than read as a ledger.
#[test]
fn the_record_round_trips_and_refuses_another_form() {
    let mut log = Log::default();
    log.entry(1, &[10], 0, 0);
    log.entry(2, &[20], 11, 0);
    assert_eq!(log.reopened(), log.ledger);
    let stored = log.ledger.encode();
    let held = Held::default();
    assert!(ClosureLedger::decode(&11u64.to_be_bytes(), std::iter::empty(), &held).is_err());
    assert!(ClosureLedger::decode(&stored[..stored.len() - 1], std::iter::empty(), &held).is_err());
    let mut old = stored.clone();
    old[0] = 1;
    assert!(ClosureLedger::decode(&old, std::iter::empty(), &held).is_err());
    let b = Breakpoint {
        index: 7,
        ceiling: 70,
    };
    assert!(b.key().starts_with(BREAKPOINT_PREFIX));
    assert_eq!(Breakpoint::decode(&b.key(), &b.value()).expect("decode"), b);
}
