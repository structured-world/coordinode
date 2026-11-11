use super::*;

/// Wide enough that the bound is not what any of these tests measures; the
/// one test about the bound builds its own table.
const ROOMY: usize = 64;

fn key(part: Partition, k: &[u8]) -> Vec<(Partition, Vec<u8>)> {
    vec![(part, k.to_vec())]
}

fn table() -> PendingCommits {
    PendingCommits::new(ROOMY)
}

/// Admit at a fixed timestamp, which is what the commit path does with the
/// oracle's next value.
fn admit_at(
    pending: &PendingCommits,
    ts: u64,
    scope: Vec<(Partition, Vec<u8>)>,
) -> Result<Admission<'_>, Refusal> {
    pending
        .admit_allocated(|| ts, scope, Vec::new(), Material::Changes)
        .map(|(_, a)| a)
}

/// Admit a commit that writes `scope` and only conditions on `guards`.
fn admit_guarded(
    pending: &PendingCommits,
    ts: u64,
    scope: Vec<(Partition, Vec<u8>)>,
    guards: Vec<(Partition, Vec<u8>)>,
) -> Result<Admission<'_>, Refusal> {
    pending
        .admit_allocated(|| ts, scope, guards, Material::Changes)
        .map(|(_, a)| a)
}

/// Hold `ts` as not yet logged, for an entry no consumer's material is in:
/// what the closed-bound tests hold.
fn unlogged(pending: &Arc<PendingCommits>, ts: u64) -> (u64, Unlogged) {
    pending.obligate(|| ts)
}

fn holder_of(refusal: &Refusal) -> u64 {
    match refusal {
        Refusal::Overlap { holder_ts, .. } => *holder_ts,
        other => panic!("expected an overlap, got {other:?}"),
    }
}

/// Two commits over disjoint keys are both admitted: exclusion is by key, not
/// by the table being busy.
#[test]
fn disjoint_scopes_are_both_admitted() {
    let pending = table();
    let _first = admit_at(&pending, 10, key(Partition::Node, b"a")).expect("first");
    let _second =
        admit_at(&pending, 11, key(Partition::Node, b"b")).expect("another key is another scope");
    assert_eq!(pending.in_flight(), 2);
}

/// The same key from two commits: the later one is refused and told which
/// commit holds it, so the caller can say why rather than that something did.
#[test]
fn the_later_commit_is_refused_and_names_the_holder() {
    let pending = table();
    let _first = admit_at(&pending, 10, key(Partition::Node, b"shared")).expect("first");

    let refusal = admit_at(&pending, 11, key(Partition::Node, b"shared"))
        .expect_err("the key is already spoken for");
    assert_eq!(holder_of(&refusal), 10);
    match refusal {
        Refusal::Overlap { partition, key, .. } => {
            assert_eq!(partition, Partition::Node);
            assert_eq!(key, b"shared".to_vec());
        }
        other => panic!("expected an overlap, got {other:?}"),
    }
}

/// An overlap refuses in either direction, including when the arriving commit
/// will land first.
///
/// Letting it through is the plausible-looking mistake: it lands before the
/// one already in flight, so it looks harmless. It is not, because the commit
/// in flight read its inputs at a snapshot that cannot contain it, and applies
/// over the top of it afterwards. That is the lost update, arriving from the
/// other side.
#[test]
fn an_overlap_refuses_even_when_the_arrival_lands_first() {
    let pending = table();

    let _later =
        admit_at(&pending, 20, key(Partition::Node, b"shared")).expect("nothing else is in flight");

    let refusal = admit_at(&pending, 10, key(Partition::Node, b"shared"))
        .expect_err("the key is spoken for whichever lands first");
    assert_eq!(holder_of(&refusal), 20);
    assert_eq!(pending.in_flight(), 1);
}

/// A key one commit conditions on and another writes refuses whichever
/// arrives second: the writer would move the key after the condition on it
/// was checked.
#[test]
fn a_condition_and_a_write_on_one_key_refuse_each_other() {
    let pending = table();
    {
        let _guard = admit_guarded(&pending, 10, Vec::new(), key(Partition::Node, b"g"))
            .expect("the guard first");
        let refusal = admit_at(&pending, 11, key(Partition::Node, b"g"))
            .expect_err("a writer of a guarded key");
        assert_eq!(holder_of(&refusal), 10);
    }
    let _writer = admit_at(&pending, 12, key(Partition::Node, b"g")).expect("the writer first");
    let refusal = admit_guarded(
        &pending,
        13,
        key(Partition::Node, b"w"),
        key(Partition::Node, b"g"),
    )
    .expect_err("a condition on a key being written");
    assert_eq!(holder_of(&refusal), 12);
}

/// Two commits that only condition on one key leave it as both found it,
/// so neither holds the other up.
#[test]
fn two_conditions_on_one_key_are_both_admitted() {
    let pending = table();
    let _first = admit_guarded(
        &pending,
        10,
        key(Partition::Node, b"a"),
        key(Partition::Node, b"g"),
    )
    .expect("first");
    let _second = admit_guarded(
        &pending,
        11,
        key(Partition::Node, b"b"),
        key(Partition::Node, b"g"),
    )
    .expect("a shared condition is not a conflict");
    assert_eq!(pending.in_flight(), 2);
}

/// A registration lives exactly as long as the commit it belongs to.
#[test]
fn a_finished_commit_holds_nothing() {
    let pending = table();
    {
        let _admitted = admit_at(&pending, 10, key(Partition::Node, b"shared")).expect("first");
        assert_eq!(pending.in_flight(), 1);
    }
    assert_eq!(pending.in_flight(), 0);

    admit_at(&pending, 11, key(Partition::Node, b"shared"))
        .expect("the key was released with the commit that held it");
}

/// The snapshot stops below a commit in flight, so a reader never gets a
/// number that covers a write it cannot see.
#[test]
fn the_snapshot_floor_stops_below_a_commit_in_flight() {
    let pending = table();
    assert_eq!(
        pending.snapshot_floor(|| 100),
        100,
        "with nothing in flight the clock is the answer"
    );

    let _admitted = admit_at(&pending, 50, key(Partition::Node, b"a")).expect("admit");
    assert_eq!(
        pending.snapshot_floor(|| 100),
        50,
        "a reader must not be handed a number that covers an unapplied commit"
    );
}

/// The floor stops at the first hole, not the last: everything below the
/// oldest commit in flight has landed, and nothing above it is proved.
#[test]
fn the_floor_is_the_first_hole_not_the_last() {
    let pending = table();
    let _a = admit_at(&pending, 30, key(Partition::Node, b"a")).expect("a");
    let _b = admit_at(&pending, 20, key(Partition::Node, b"b")).expect("b");
    let _c = admit_at(&pending, 40, key(Partition::Node, b"c")).expect("c");

    assert_eq!(pending.snapshot_floor(|| 100), 20);
}

/// The floor comes back up as the commits holding it finish.
#[test]
fn the_floor_recovers_when_the_commits_below_it_finish() {
    let pending = table();
    {
        let _a = admit_at(&pending, 30, key(Partition::Node, b"a")).expect("a");
        {
            let _b = admit_at(&pending, 20, key(Partition::Node, b"b")).expect("b");
            assert_eq!(pending.snapshot_floor(|| 100), 20);
        }
        assert_eq!(
            pending.snapshot_floor(|| 100),
            30,
            "the older commit finished, so the floor moves up to the next one"
        );
    }
    assert_eq!(pending.snapshot_floor(|| 100), 100);
}

/// A commit whose timestamp is above the clock the reader sees does not hold
/// that reader back: it cannot be inside a snapshot that does not reach it.
#[test]
fn a_commit_above_the_clock_does_not_lower_the_floor() {
    let pending = table();
    let _admitted = admit_at(&pending, 200, key(Partition::Node, b"a")).expect("admit");
    assert_eq!(pending.snapshot_floor(|| 100), 100);
}

/// A commit with no exclusive writes still registers: a reader at its
/// timestamp still has to account for it, and its merge operands still land.
#[test]
fn a_commit_with_no_exclusive_writes_is_still_in_flight() {
    let pending = table();
    let _admitted = admit_at(&pending, 10, Vec::new()).expect("admit");
    assert_eq!(pending.in_flight(), 1);
    assert_eq!(pending.snapshot_floor(|| 50), 10);
}

/// Two commits sharing a timestamp are both held. A table keyed by the
/// timestamp would drop one of them and release the survivor's registration
/// when the other finished, which is the lost update coming back through the
/// mechanism that exists to stop it.
#[test]
fn two_commits_at_one_timestamp_are_both_held() {
    let pending = table();
    let _first = admit_at(&pending, 10, key(Partition::Node, b"a")).expect("first");
    let _second = admit_at(&pending, 10, key(Partition::Node, b"b")).expect("second");
    assert_eq!(pending.in_flight(), 2);

    drop(_first);
    assert_eq!(
        pending.in_flight(),
        1,
        "the commit that finished released its own registration and not the other's"
    );
    assert_eq!(pending.snapshot_floor(|| 100), 10);
}

/// A read at a named timestamp waits for the commits under it and then
/// proceeds, rather than being answered from a state missing one of them.
#[test]
fn a_named_timestamp_waits_for_the_commits_it_covers() {
    use std::sync::Arc;
    use std::time::Duration;

    let pending = Arc::new(PendingCommits::new(ROOMY));

    let holder = Arc::clone(&pending);
    let (started_tx, started_rx) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        let admitted = holder
            .admit_allocated(
                || 50,
                key(Partition::Node, b"a"),
                Vec::new(),
                Material::Changes,
            )
            .map(|(_, a)| a)
            .expect("admit");
        started_tx.send(()).expect("signal");
        std::thread::sleep(Duration::from_millis(60));
        drop(admitted);
    });
    started_rx.recv().expect("the commit is in flight");

    let waited = std::time::Instant::now();
    pending
        .await_complete_at(100, Duration::from_secs(5))
        .expect("the commit landed within the wait");
    assert!(
        waited.elapsed() >= Duration::from_millis(40),
        "the reader returned before the commit it was waiting for finished"
    );
    worker.join().expect("worker");
}

/// When the wait runs out the read is refused and told what it was waiting
/// for. It is not answered from an incomplete state, and it is not quietly
/// moved to a timestamp that happens to be ready.
#[test]
fn an_exhausted_wait_names_what_it_waited_for() {
    use std::time::Duration;

    let pending = table();
    let _admitted = admit_at(&pending, 50, key(Partition::Node, b"a")).expect("admit");

    let blocking = pending
        .await_complete_at(100, Duration::from_millis(20))
        .expect_err("the commit never landed");
    assert_eq!(blocking, 50);
}

/// A commit above the requested timestamp is not in its way.
#[test]
fn a_named_timestamp_does_not_wait_for_commits_above_it() {
    use std::time::Duration;

    let pending = table();
    let _admitted = admit_at(&pending, 200, key(Partition::Node, b"a")).expect("admit");

    pending
        .await_complete_at(100, Duration::from_millis(20))
        .expect("nothing under this timestamp is in flight");
}

/// A task waits for a named timestamp without a thread: its wait is pending
/// while a commit at or below it is in flight, takes no notice of one above,
/// and the commit finishing wakes the task and completes the wait.
#[test]
fn a_task_waiting_for_a_timestamp_is_woken_by_the_commit_it_waits_for() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::Wake;

    struct Count(AtomicUsize);
    impl Wake for Count {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    let pending = Arc::new(table());
    let woken = Arc::new(Count(AtomicUsize::new(0)));
    let waker = Waker::from(Arc::clone(&woken));
    let mut cx = Context::from_waker(&waker);

    let above = admit_at(&pending, 200, key(Partition::Node, b"b")).expect("admit");
    let mut clear = pending.complete_at(100);
    assert!(
        Pin::new(&mut clear).poll(&mut cx).is_ready(),
        "a commit above the timestamp is not in its way"
    );

    let below = admit_at(&pending, 50, key(Partition::Node, b"a")).expect("admit");
    let mut wait = pending.complete_at(100);
    assert!(Pin::new(&mut wait).poll(&mut cx).is_pending());
    assert!(Pin::new(&mut wait).poll(&mut cx).is_pending());
    assert_eq!(pending.in_flight_at(100), Some(50));

    drop(above);
    drop(below);
    assert!(
        woken.0.load(Ordering::SeqCst) >= 1,
        "the finishing commit woke the task"
    );
    assert!(Pin::new(&mut wait).poll(&mut cx).is_ready());
    assert_eq!(pending.in_flight_at(100), None);
}

/// A wait's registration belongs to that wait: waits dropped while their
/// commit is still in flight leave no waker behind, a wait polled again under
/// another waker keeps only the latest one, and dropping one of two waits of
/// the same task leaves the other woken when the commit finishes.
#[test]
fn a_waits_registration_ends_with_the_wait() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::Wake;

    struct Count(AtomicUsize);
    impl Wake for Count {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let count = || Arc::new(Count(AtomicUsize::new(0)));
    let poll = |wait: &mut CompleteAt, by: &Arc<Count>| {
        let waker = Waker::from(Arc::clone(by));
        Pin::new(wait).poll(&mut Context::from_waker(&waker))
    };

    let pending = Arc::new(table());
    let held = admit_at(&pending, 50, key(Partition::Node, b"a")).expect("admit");

    let gone: Vec<_> = (0..8).map(|_| count()).collect();
    for by in &gone {
        let mut wait = pending.complete_at(100);
        assert!(poll(&mut wait, by).is_pending());
        drop(wait);
    }
    for by in &gone {
        assert_eq!(
            Arc::strong_count(by),
            1,
            "a wait dropped before its commit finished left its waker registered"
        );
    }

    let (old, new) = (count(), count());
    let mut moved = pending.complete_at(100);
    assert!(poll(&mut moved, &old).is_pending());
    assert!(poll(&mut moved, &new).is_pending());
    assert_eq!(Arc::strong_count(&old), 1, "the replaced waker is let go");
    assert_eq!(Arc::strong_count(&new), 2, "the latest waker is kept");

    let shared = count();
    let mut left = pending.complete_at(100);
    let mut kept = pending.complete_at(100);
    assert!(poll(&mut left, &shared).is_pending());
    assert!(poll(&mut kept, &shared).is_pending());
    drop(left);

    drop(held);
    assert!(
        shared.0.load(Ordering::SeqCst) >= 1,
        "the kept wait is woken"
    );
    assert!(new.0.load(Ordering::SeqCst) >= 1, "the moved wait is woken");
    assert_eq!(old.0.load(Ordering::SeqCst), 0, "the replaced waker is not");
    assert!(poll(&mut kept, &shared).is_ready());
    assert!(poll(&mut moved, &new).is_ready());
    drop((kept, moved));
    assert_eq!(Arc::strong_count(&shared), 1);
    assert_eq!(Arc::strong_count(&new), 1);
}

/// A wait dropped while its commit finishes on another thread leaves nothing
/// behind, whichever of the two takes the table first.
#[test]
fn a_wait_dropped_as_its_commit_finishes_leaves_nothing_behind() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::Wake;

    struct Count(AtomicUsize);
    impl Wake for Count {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    let pending = Arc::new(table());
    for round in 0..200u64 {
        let by = Arc::new(Count(AtomicUsize::new(0)));
        let waker = Waker::from(Arc::clone(&by));
        let mut wait = pending.complete_at(1_000_000);
        let (ts, admission) = pending
            .admit_allocated(|| round + 1, Vec::new(), Vec::new(), Material::Changes)
            .expect("admit");
        assert_eq!(ts, round + 1);
        assert!(
            Pin::new(&mut wait)
                .poll(&mut Context::from_waker(&waker))
                .is_pending()
        );
        std::thread::scope(|scope| {
            scope.spawn(move || drop(admission));
            drop(wait);
        });
        drop(waker);
        assert_eq!(Arc::strong_count(&by), 1, "round {round} left a waker");
        assert!(by.0.load(Ordering::SeqCst) <= 1, "woken at most once");
    }
}

/// At the ceiling a commit is refused rather than admitted, and refused
/// before its timestamp is taken: a number allocated and then thrown away is
/// a hole in the clock that nobody closes.
#[test]
fn the_table_refuses_rather_than_growing_without_bound() {
    let pending = PendingCommits::new(2);
    let _a = admit_at(&pending, 10, key(Partition::Node, b"a")).expect("a");
    let _b = admit_at(&pending, 11, key(Partition::Node, b"b")).expect("b");

    let mut allocated = false;
    let refusal = pending
        .admit_allocated(
            || {
                allocated = true;
                12
            },
            key(Partition::Node, b"c"),
            Vec::new(),
            Material::Changes,
        )
        .map(|(_, a)| a)
        .expect_err("the table is full");

    assert!(
        matches!(refusal, Refusal::AtCapacity { limit: 2 }),
        "expected the ceiling to answer, got {refusal:?}"
    );
    assert!(
        !allocated,
        "a refused commit must not have taken a timestamp"
    );
    assert_eq!(pending.in_flight(), 2);
}

/// The ceiling is a ceiling on commits in flight, not a one-way latch: once
/// one finishes, the next is admitted.
#[test]
fn the_ceiling_admits_again_once_a_commit_finishes() {
    let pending = PendingCommits::new(1);
    {
        let _a = admit_at(&pending, 10, key(Partition::Node, b"a")).expect("a");
        admit_at(&pending, 11, key(Partition::Node, b"b")).expect_err("full");
    }
    admit_at(&pending, 12, key(Partition::Node, b"b")).expect("the table has room again");
}

/// A waiting commit lands after the commit holding its key instead of being
/// refused by it, and takes its timestamp only then.
#[test]
fn a_waiter_is_admitted_once_the_holder_lands() {
    let pending = table();
    let holder = admit_guarded(&pending, 10, Vec::new(), key(Partition::Schema, b"def"))
        .expect("a writer conditioned on the record");
    std::thread::scope(|s| {
        let waiter = s.spawn(|| {
            pending
                .admit_allocated_waiting(
                    || 20,
                    key(Partition::Schema, b"def"),
                    Vec::new(),
                    Duration::from_secs(10),
                    Material::Changes,
                )
                .map(|(ts, _)| ts)
        });
        std::thread::sleep(Duration::from_millis(100));
        assert!(!waiter.is_finished(), "the waiter waits for the holder");
        drop(holder);
        assert_eq!(waiter.join().expect("waiter thread"), Ok(20));
    });
}

/// While a commit waits, its keys turn away later commits on them, so a
/// stream of new writers cannot keep it waiting; a second waiter on the same
/// key is turned away too, instead of queueing behind the first's retries.
#[test]
fn a_waiter_reserves_its_keys_against_later_commits() {
    let pending = table();
    let holder = admit_guarded(&pending, 10, Vec::new(), key(Partition::Schema, b"def"))
        .expect("the first writer");
    std::thread::scope(|s| {
        let waiter = s.spawn(|| {
            pending
                .admit_allocated_waiting(
                    || 20,
                    key(Partition::Schema, b"def"),
                    Vec::new(),
                    Duration::from_secs(10),
                    Material::Changes,
                )
                .map(|(ts, _)| ts)
        });
        // The reservation is visible once the waiter blocks on the holder.
        let deadline = Instant::now() + Duration::from_secs(5);
        let later = loop {
            match admit_guarded(&pending, 11, Vec::new(), key(Partition::Schema, b"def")) {
                Err(refusal @ Refusal::Reserved { .. }) => break refusal,
                Ok(admitted) => drop(admitted),
                Err(other) => panic!("unexpected refusal {other:?}"),
            }
            assert!(
                Instant::now() < deadline,
                "the waiter never reserved its key"
            );
            std::thread::sleep(Duration::from_millis(5));
        };
        assert_eq!(
            later,
            Refusal::Reserved {
                partition: Partition::Schema,
                key: b"def".to_vec()
            }
        );
        let second = pending.admit_allocated_waiting(
            || 30,
            key(Partition::Schema, b"def"),
            Vec::new(),
            Duration::from_secs(10),
            Material::Changes,
        );
        assert!(
            matches!(second, Err(Refusal::Reserved { .. })),
            "{:?}",
            second.map(|(ts, _)| ts)
        );
        // Another key is untouched by the reservation.
        admit_at(&pending, 12, key(Partition::Node, b"other")).expect("unrelated key");
        drop(holder);
        assert_eq!(waiter.join().expect("waiter thread"), Ok(20));
    });
}

/// A waiter whose holder outlasts the wait is refused with the overlap, and
/// its reservation goes with it.
#[test]
fn a_waiter_gives_up_at_its_deadline_and_releases_its_keys() {
    let pending = table();
    let _holder = admit_guarded(&pending, 10, Vec::new(), key(Partition::Schema, b"def"))
        .expect("a writer that does not land");
    let refused = pending
        .admit_allocated_waiting(
            || 20,
            key(Partition::Schema, b"def"),
            Vec::new(),
            Duration::from_millis(50),
            Material::Changes,
        )
        .map(|(ts, _)| ts)
        .expect_err("the holder outlasts the wait");
    assert_eq!(holder_of(&refused), 10);
    admit_guarded(&pending, 11, Vec::new(), key(Partition::Schema, b"def"))
        .expect("no reservation is left behind");
}

/// The closed bound stops at the lowest timestamp not yet in the log,
/// admitted or held as unlogged alike, and passes the clock once nothing is.
#[test]
fn the_closed_bound_stops_below_every_timestamp_not_yet_logged() {
    let pending = Arc::new(table());
    assert_eq!(pending.stamp_closed_below(|| 50), 51, "nothing held");

    let admitted = admit_at(&pending, 60, key(Partition::Node, b"a")).expect("admit");
    let (held_ts, held) = unlogged(&pending, 55);
    assert_eq!(held_ts, 55);
    assert_eq!(
        pending.stamp_closed_below(|| 70),
        55,
        "the unlogged one is lowest"
    );

    drop(held);
    assert_eq!(pending.stamp_closed_below(|| 70), 60, "the admitted one");
    // Taken while the admission still holds it, the hold outlives it.
    let kept = pending.hold(60);
    drop(admitted);
    assert_eq!(pending.stamp_closed_below(|| 70), 60, "still not logged");
    drop(kept);
    assert_eq!(pending.stamp_closed_below(|| 70), 71);
}

/// The entry carrying a sealed commit closes the bound past it, since a
/// member applies the entry's proposals before its bound; an entry carrying
/// anything else stays below it. A lower timestamp held elsewhere still holds
/// the bound, in whichever order the two are handed over, and a seal names
/// one proposal: another entry's seal does not release this one.
#[test]
fn an_entry_closes_the_commit_sealed_into_it_and_nothing_else() {
    use coordinode_core::txn::proposal::ProposalId;
    let pending = Arc::new(table());
    let (own, other) = (ProposalId::from_raw(7), ProposalId::from_raw(8));

    let admitted = admit_at(&pending, 60, key(Partition::Node, b"a")).expect("admit");
    admitted.seal(own);
    assert_eq!(
        pending.stamp_closed_below_sealing(&mut [own], || 70),
        71,
        "its own entry closes past it"
    );
    assert_eq!(
        pending.stamp_closed_below_sealing(&mut [other], || 70),
        60,
        "another entry does not"
    );
    assert_eq!(pending.stamp_closed_below(|| 70), 60, "nor an unsealed one");

    // A lower timestamp held for another entry still holds the bound.
    let (_, low) = unlogged(&pending, 55);
    assert_eq!(pending.stamp_closed_below_sealing(&mut [own], || 70), 55);
    low.seal(other);
    assert_eq!(
        pending.stamp_closed_below_sealing(&mut [own], || 70),
        55,
        "sealed into another proposal, it holds this entry's bound"
    );
    assert_eq!(
        pending.stamp_closed_below_sealing(&mut [other], || 70),
        60,
        "and the other entry's bound stays below the first commit"
    );
    assert_eq!(
        pending.stamp_closed_below_sealing(&mut [other, own], || 70),
        71,
        "one entry carrying both closes both, in whatever order it lists them"
    );
    drop(low);
    drop(admitted);
    assert_eq!(pending.stamp_closed_below(|| 70), 71);
}

/// A seal does not outlive its registration: a timestamp withdrawn and a new
/// one registered are not sealed by the old seal.
#[test]
fn a_seal_ends_with_its_registration() {
    use coordinode_core::txn::proposal::ProposalId;
    let pending = Arc::new(table());
    let own = ProposalId::from_raw(7);
    let (_, held) = unlogged(&pending, 40);
    held.seal(own);
    drop(held);
    let (_, next) = unlogged(&pending, 50);
    assert_eq!(
        pending.stamp_closed_below_sealing(&mut [own], || 60),
        50,
        "the new registration was never sealed"
    );
    drop(next);
}

/// A closing entry is due only when it would cover a timestamp withdrawn
/// since the last bound; stamping it makes none due again, so closing
/// entries do not breed more of themselves.
#[test]
fn a_closing_entry_is_due_once_per_uncovered_withdrawal() {
    let pending = Arc::new(table());
    pending.stamp_closed_below(|| 10);
    assert_eq!(pending.closure_due(|| 10), None, "nothing withdrawn");

    let (_, held) = unlogged(&pending, 20);
    let bound = pending.stamp_closed_below(|| 20);
    assert_eq!(bound, 20, "the entry carrying it stops below it");
    drop(held);
    assert_eq!(
        pending.closure_due(|| 20),
        Some(21),
        "now in the log, uncovered"
    );

    assert_eq!(pending.stamp_closed_below(|| 20), 21, "the closing entry");
    assert_eq!(
        pending.closure_due(|| 25),
        None,
        "it covered the withdrawal"
    );
}

/// A timestamp taken for a write that never goes to the log (a member's own
/// consensus record) holds the bound while the write runs, but its release
/// makes no closing entry due: nothing reached the log for one to cover.
/// When it did, every committed index this member recorded made a closing
/// entry due, whose commit recorded the index again: an idle group appended
/// about 55 entries a second.
#[test]
fn a_local_timestamp_holds_the_bound_but_makes_no_closing_entry_due() {
    let pending = Arc::new(table());
    pending.stamp_closed_below(|| 10);
    let (ts, held) = pending.obligate_local(|| 20);
    assert_eq!(ts, 20);
    assert_eq!(
        pending.stamp_closed_below(|| 30),
        20,
        "held while it is written"
    );
    drop(held);
    assert_eq!(
        pending.closure_due(|| 40),
        None,
        "nothing in the log to cover"
    );
}

/// A closing entry is not due while a timestamp below the would-be bound is
/// still held: the bound could not pass it, so the entry would cover
/// nothing new.
#[test]
fn a_held_timestamp_keeps_a_closing_entry_from_being_due() {
    let pending = Arc::new(table());
    let (_, low) = unlogged(&pending, 30);
    let (_, high) = unlogged(&pending, 40);
    pending.stamp_closed_below(|| 40);
    drop(high);
    assert_eq!(pending.closure_due(|| 40), None, "30 still holds the bound");
    drop(low);
    assert_eq!(pending.closure_due(|| 40), Some(41));
}

/// A waiter for withdrawals wakes on the next one, and on its deadline
/// without one.
#[test]
fn a_withdrawal_wakes_its_waiter() {
    let pending = Arc::new(table());
    let seen = pending.await_withdrawal(0, std::time::Duration::from_millis(10));
    assert_eq!(seen, 0, "the deadline, with nothing withdrawn");
    let (_, held) = unlogged(&pending, 5);
    let waiter = {
        let pending = Arc::clone(&pending);
        std::thread::spawn(move || pending.await_withdrawal(0, std::time::Duration::from_secs(10)))
    };
    std::thread::sleep(std::time::Duration::from_millis(20));
    drop(held);
    assert_eq!(waiter.join().expect("waiter"), 1);
}

/// A terminal decision takes the material scope only once every write
/// sharing it has landed, and reports the highest timestamp they took. While
/// it waits, a later write that changes the material waits behind it rather
/// than overtaking it; a write that changes none goes on. The waiting write
/// registers once the decision is released.
#[test]
fn a_material_claim_waits_for_sharers_and_is_not_overtaken() {
    let pending = Arc::new(table());
    let first = admit_at(&pending, 10, key(Partition::Node, b"a")).expect("a sharer");
    std::thread::scope(|s| {
        let claimer = s.spawn(|| pending.claim_material(Duration::from_secs(10)));
        // The claim waits for the sharer in flight.
        let deadline = Instant::now() + Duration::from_secs(5);
        while pending.inner.lock().material_claim != Claim::Waiting {
            assert!(Instant::now() < deadline, "the claim never waited");
            std::thread::sleep(Duration::from_millis(2));
        }
        // A later write that changes the material waits behind it.
        let later = s.spawn(|| {
            pending
                .admit_allocated(
                    || 30,
                    key(Partition::Node, b"b"),
                    Vec::new(),
                    Material::Changes,
                )
                .map(|(ts, _)| ts)
        });
        // One that changes none is not held up.
        let (_, registry) = pending
            .admit_allocated(
                || 20,
                key(Partition::Registry, b"r"),
                Vec::new(),
                Material::Untouched,
            )
            .expect("an untouched write goes on");
        drop(registry);
        std::thread::sleep(Duration::from_millis(50));
        assert!(
            !claimer.is_finished(),
            "the claim passed a sharer in flight"
        );
        assert!(!later.is_finished(), "a later sharer overtook the claim");
        drop(first);
        let claim = claimer.join().expect("claimer").expect("the claim");
        assert_eq!(claim.through(), 10, "the sharers' highest timestamp");
        std::thread::sleep(Duration::from_millis(50));
        assert!(!later.is_finished(), "a sharer registered under the claim");
        drop(claim);
        assert_eq!(later.join().expect("later"), Ok(30));
    });
}

/// A decision that keeps the material scope (an ending whose outcome is not
/// known) holds a new write that changes the material only for that write's
/// own bound: past [`MATERIAL_WAIT`] each admission path refuses it with
/// `MaterialClaimed` and holds nothing, and a write that changes no material
/// is admitted at once. Before, the logged and the direct paths waited for
/// the decision with no deadline.
#[test]
fn a_held_decision_refuses_a_changing_write_at_its_deadline() {
    let pending = Arc::new(table());
    let held = pending
        .claim_material(Duration::from_secs(5))
        .expect("nothing shares the material");
    let bounded = |what: &str, refused: Result<u64, Refusal>, started: Instant| {
        let waited = started.elapsed();
        assert_eq!(refused, Err(Refusal::MaterialClaimed), "{what}");
        assert!(
            waited >= MATERIAL_WAIT && waited < MATERIAL_WAIT + Duration::from_secs(3),
            "{what} waited {waited:?}"
        );
    };
    std::thread::scope(|s| {
        let logged = s.spawn(|| {
            let started = Instant::now();
            let refused = pending.obligate_changing(|| 40).map(|(ts, _)| ts);
            (refused, started)
        });
        let direct = s.spawn(|| {
            let started = Instant::now();
            let refused = pending
                .admit_direct(|| 41, Material::Changes)
                .map(|(ts, _)| ts);
            (refused, started)
        });
        let untouched_at = Instant::now();
        let (ts, untouched) = pending.obligate(|| 42);
        assert_eq!(ts, 42);
        assert!(
            untouched_at.elapsed() < Duration::from_secs(1),
            "a write that changes no material waited for the decision"
        );
        drop(untouched);
        let (refused, started) = logged.join().expect("logged");
        bounded("the logged write", refused, started);
        let (refused, started) = direct.join().expect("direct");
        bounded("the direct write", refused, started);
    });
    assert!(
        pending.inner.lock().admitted.is_empty(),
        "a refused write holds nothing"
    );
    drop(held);
    let (ts, after) = pending
        .obligate_changing(|| 43)
        .expect("admitted once the decision is released");
    assert_eq!(ts, 43);
    drop(after);
}

/// A claim whose sharers outlast its wait is refused and holds nothing: the
/// writes that waited behind it go on. A second claim beside a held one is
/// refused at once.
#[test]
fn a_material_claim_that_cannot_be_taken_holds_nothing() {
    let pending = Arc::new(table());
    let sharer = admit_at(&pending, 10, key(Partition::Node, b"a")).expect("a sharer");
    assert_eq!(
        pending
            .claim_material(Duration::from_millis(30))
            .map(|c| c.through()),
        Err(Refusal::MaterialClaimed)
    );
    admit_at(&pending, 11, key(Partition::Node, b"b")).expect("nothing is held");
    drop(sharer);
    let held = pending
        .claim_material(Duration::from_secs(5))
        .expect("taken once the sharers landed");
    assert_eq!(
        pending
            .claim_material(Duration::from_secs(5))
            .map(|c| c.through()),
        Err(Refusal::MaterialClaimed),
        "one decision at a time"
    );
    drop(held);
    let again = pending
        .claim_material(Duration::from_secs(5))
        .expect("free again once released");
    drop(again);
}
