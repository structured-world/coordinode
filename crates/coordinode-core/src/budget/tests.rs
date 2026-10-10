use super::*;

/// A reservation within the limit is held until its charge drops, and the
/// peak remembers the most held at once.
#[test]
fn a_reservation_is_held_until_its_charge_drops() {
    let budget = QueryBudget::new(100);
    {
        let a = budget.reserve(40).expect("within the limit");
        let b = budget.reserve(60).expect("exactly the limit");
        assert_eq!((a.bytes(), b.bytes()), (40, 60));
        assert_eq!(budget.memory_used(), 100);
    }
    assert_eq!(budget.memory_used(), 0, "both returned");
    assert_eq!(budget.memory_peak(), 100);
}

/// Memory handed to the query's later stages stays reserved after its charge
/// is let go, so what they hold still counts against the limit; an empty
/// charge grows like any other.
#[test]
fn a_charge_kept_for_the_query_stays_reserved() {
    let budget = QueryBudget::new(100);
    let mut rows = budget.empty_charge();
    assert_eq!(budget.memory_used(), 0);
    rows.grow(60).expect("within the limit");
    rows.keep_until_query_ends();
    assert_eq!(budget.memory_used(), 60, "still held by the rows handed on");
    assert!(budget.reserve(41).is_err(), "the kept rows count");
    budget.reserve(40).expect("what is left");
}

/// A reservation past the limit is refused before anything is held, names
/// what it asked for, and leaves the budget as it was.
#[test]
fn a_reservation_past_the_limit_is_refused_and_changes_nothing() {
    let budget = QueryBudget::new(100);
    let _held = budget.reserve(70).expect("within the limit");
    let refused = budget.reserve(31);
    assert_eq!(
        refused.err(),
        Some(BudgetStop::Memory {
            requested: 31,
            used: 70,
            limit: 100,
        })
    );
    assert_eq!(budget.memory_used(), 70);
    assert_eq!(budget.memory_peak(), 70, "a refused reservation is no peak");
    assert!(budget.reserve(30).is_ok(), "the rest is still there");
}

/// A request near the top of the integer range is refused, not wrapped into
/// a small number that fits.
#[test]
fn a_huge_request_is_refused_rather_than_wrapping() {
    let budget = QueryBudget::new(QUERY_MEMORY_CEILING);
    let _held = budget.reserve(10).expect("within the limit");
    assert!(matches!(
        budget.reserve(u64::MAX - 5),
        Err(BudgetStop::Memory { .. })
    ));
    assert_eq!(budget.memory_used(), 10);
}

/// A charge grows before the growth it pays for and shrinks as memory is
/// released; a refused growth keeps what it held, and a shrink never returns
/// more than the charge holds.
#[test]
fn a_charge_grows_and_shrinks_with_its_memory() {
    let budget = QueryBudget::new(100);
    let mut charge = budget.reserve(10).expect("within the limit");
    charge.grow(50).expect("within the limit");
    assert_eq!(budget.memory_used(), 60);
    assert!(charge.grow(41).is_err());
    assert_eq!(charge.bytes(), 60, "a refused growth keeps what it held");
    charge.shrink(25);
    assert_eq!((charge.bytes(), budget.memory_used()), (35, 35));
    charge.shrink(1_000);
    assert_eq!((charge.bytes(), budget.memory_used()), (0, 0));
    drop(charge);
    assert_eq!(
        budget.memory_used(),
        0,
        "an emptied charge returns nothing twice"
    );
}

/// Operators sharing one budget from several threads never hold more than
/// the limit together, and a reservation that fits is never refused because
/// another thread's refused one passed through the counter.
#[test]
fn parallel_reservations_share_one_limit() {
    let budget = QueryBudget::new(1_000);
    std::thread::scope(|s| {
        for _ in 0..8 {
            s.spawn(|| {
                for _ in 0..10_000 {
                    // Each thread fits on its own: 8 x 100 is under 1000, so
                    // only an add-then-undo counter could refuse these.
                    let charge = budget.reserve(100).expect("fits beside the others");
                    // A request that never fits must not disturb the others.
                    assert!(budget.reserve(2_000).is_err());
                    drop(charge);
                }
            });
        }
    });
    assert_eq!(budget.memory_used(), 0);
    assert!(budget.memory_peak() <= 1_000);
}

fn fake_now() -> u64 {
    NOW.load(Ordering::Acquire)
}

static NOW: AtomicU64 = AtomicU64::new(0);

/// Work is counted as done; the deadline is checked every CHECK_EVERY units,
/// not per unit, and stops the query once passed.
#[test]
fn the_deadline_is_checked_as_work_accrues() {
    NOW.store(0, Ordering::Release);
    let budget = QueryBudget::new(100).with_deadline(1_000, fake_now);
    for _ in 0..CHECK_EVERY - 1 {
        budget.work(1).expect("before the deadline");
    }
    NOW.store(1_000, Ordering::Release);
    assert_eq!(
        budget.work(1),
        Err(BudgetStop::Deadline),
        "the CHECK_EVERY-th unit checks"
    );
    assert_eq!(budget.work_done(), CHECK_EVERY);
    assert_eq!(budget.check(), Err(BudgetStop::Deadline));
}

/// Work between checks is not interrupted, even past the deadline: the cost
/// of a check is paid once per CHECK_EVERY units.
#[test]
fn work_between_checks_reads_no_clock() {
    NOW.store(5_000, Ordering::Release);
    let budget = QueryBudget::new(100).with_deadline(1_000, fake_now);
    for _ in 0..CHECK_EVERY - 1 {
        budget.work(1).expect("no check yet");
    }
    assert!(budget.work(1).is_err());
}

/// A cancelled query is stopped at its next check, and a budget without a
/// deadline never stops for time.
#[test]
fn cancellation_stops_at_the_next_check() {
    let budget = QueryBudget::new(100);
    budget
        .work(CHECK_EVERY * 3)
        .expect("no deadline, not cancelled");
    budget.cancel();
    assert_eq!(budget.check(), Err(BudgetStop::Cancelled));
    assert_eq!(budget.work(CHECK_EVERY), Err(BudgetStop::Cancelled));
}

/// A part spends from the whole's limit while it lives, and what it kept
/// for the rest of its evaluation returns to the whole when it is dropped;
/// what the whole kept itself stays.
#[test]
fn a_part_returns_what_it_kept_when_it_ends() {
    let whole = Arc::new(QueryBudget::new(100));
    whole.reserve(30).expect("whole").keep_until_query_ends();
    for _ in 0..10 {
        let part = whole.part();
        part.reserve(60).expect("part").keep_until_query_ends();
        assert_eq!(whole.memory_used(), 90);
        assert!(
            matches!(part.reserve(20), Err(BudgetStop::Memory { limit: 100, .. })),
            "the part is held to the whole's limit"
        );
        let mut charge = part.empty_charge();
        charge.grow(5).expect("within");
        charge.shrink(5);
        assert_eq!(whole.memory_used(), 90);
    }
    assert_eq!(whole.memory_used(), 30);
    assert_eq!(whole.memory_peak(), 95);
}

/// A batch hands its work to the budget a full batch at a time, checking
/// then, and the remainder when finished; its scratch is held until it is
/// dropped.
#[test]
fn a_batch_counts_its_work_and_holds_its_scratch() {
    let budget = QueryBudget::new(100);
    let mut batch = BatchedWork::new(&budget);
    batch.reserve(40).expect("scratch");
    assert_eq!(budget.memory_used(), 40);
    for _ in 0..CHECK_EVERY - 1 {
        batch.add(1).expect("below a batch");
    }
    assert_eq!(budget.work_done(), 0, "nothing handed over yet");
    budget.cancel();
    assert_eq!(batch.add(1), Err(BudgetStop::Cancelled));
    assert_eq!(budget.work_done(), CHECK_EVERY);
    batch.add(3).expect("below a batch");
    assert_eq!(batch.finish(), Err(BudgetStop::Cancelled));
    assert_eq!(budget.work_done(), CHECK_EVERY + 3);
    assert_eq!(
        budget.memory_used(),
        0,
        "the scratch returns with the batch"
    );
}

/// A part counts its work toward the whole's checks and stops for the
/// whole's cancellation.
#[test]
fn a_part_works_and_stops_with_the_whole() {
    let whole = Arc::new(QueryBudget::new(100));
    let part = whole.part();
    part.work(CHECK_EVERY - 1).expect("no check yet");
    whole.cancel();
    assert_eq!(part.check(), Err(BudgetStop::Cancelled));
    assert_eq!(part.work(1), Err(BudgetStop::Cancelled));
    assert_eq!(whole.work_done(), CHECK_EVERY);
}
