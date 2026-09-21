use super::*;

fn key(part: Partition, k: &[u8]) -> Vec<(Partition, Vec<u8>)> {
    vec![(part, k.to_vec())]
}

/// Admit at a fixed timestamp, which is what the commit path does with the
/// oracle's next value.
fn admit_at(
    pending: &PendingCommits,
    ts: u64,
    scope: Vec<(Partition, Vec<u8>)>,
) -> Result<Admission<'_>, Overlap> {
    pending.admit_allocated(|| ts, scope).map(|(_, a)| a)
}

/// Two commits over disjoint keys are both admitted: exclusion is by key, not
/// by the table being busy.
#[test]
fn disjoint_scopes_are_both_admitted() {
    let pending = PendingCommits::new();
    let _first = admit_at(&pending, 10, key(Partition::Node, b"a")).expect("first");
    let _second =
        admit_at(&pending, 11, key(Partition::Node, b"b")).expect("another key is another scope");
    assert_eq!(pending.in_flight(), 2);
}

/// The same key from two commits: the later one is refused and told which
/// commit holds it, so the caller can say why rather than that something did.
#[test]
fn the_later_commit_is_refused_and_names_the_holder() {
    let pending = PendingCommits::new();
    let _first = admit_at(&pending, 10, key(Partition::Node, b"shared")).expect("first");

    let overlap = admit_at(&pending, 11, key(Partition::Node, b"shared"))
        .expect_err("the key is already spoken for");
    assert_eq!(overlap.holder_ts, 10);
    assert_eq!(overlap.key, b"shared".to_vec());
    assert_eq!(overlap.partition, Partition::Node);
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
    let pending = PendingCommits::new();

    let _later =
        admit_at(&pending, 20, key(Partition::Node, b"shared")).expect("nothing else is in flight");

    let overlap = admit_at(&pending, 10, key(Partition::Node, b"shared"))
        .expect_err("the key is spoken for whichever lands first");
    assert_eq!(overlap.holder_ts, 20);
    assert_eq!(pending.in_flight(), 1);
}

/// A registration lives exactly as long as the commit it belongs to.
#[test]
fn a_finished_commit_holds_nothing() {
    let pending = PendingCommits::new();
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
    let pending = PendingCommits::new();
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
    let pending = PendingCommits::new();
    let _a = admit_at(&pending, 30, key(Partition::Node, b"a")).expect("a");
    let _b = admit_at(&pending, 20, key(Partition::Node, b"b")).expect("b");
    let _c = admit_at(&pending, 40, key(Partition::Node, b"c")).expect("c");

    assert_eq!(pending.snapshot_floor(|| 100), 20);
}

/// A commit whose timestamp is above the clock the reader sees does not hold
/// that reader back: it cannot be inside a snapshot that does not reach it.
#[test]
fn a_commit_above_the_clock_does_not_lower_the_floor() {
    let pending = PendingCommits::new();
    let _admitted = admit_at(&pending, 200, key(Partition::Node, b"a")).expect("admit");
    assert_eq!(pending.snapshot_floor(|| 100), 100);
}

/// A commit with no exclusive writes still registers: a reader at its
/// timestamp still has to account for it, and its merge operands still land.
#[test]
fn a_commit_with_no_exclusive_writes_is_still_in_flight() {
    let pending = PendingCommits::new();
    let _admitted = admit_at(&pending, 10, Vec::new()).expect("admit");
    assert_eq!(pending.in_flight(), 1);
    assert_eq!(pending.snapshot_floor(|| 50), 10);
}
