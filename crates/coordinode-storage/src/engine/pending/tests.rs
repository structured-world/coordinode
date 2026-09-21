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
    pending.admit_allocated(|| ts, scope).map(|(_, a)| a)
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
            .admit_allocated(|| 50, key(Partition::Node, b"a"))
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
