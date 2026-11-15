use super::{Grounds, Round};

const BEFORE: Grounds = Grounds {
    newest_sealed: Some(40),
    floor: u64::MAX,
};

/// A round that succeeded runs again only on a new ground: a segment sealed
/// since it began (also one sealed while it ran, since it records what it
/// read before), or a floor raised. Its own removals, or a journal emptied
/// by it, are no ground.
#[test]
fn a_succeeded_round_runs_again_only_on_a_new_ground() {
    let round = Round {
        checked: BEFORE,
        failed: false,
    };
    assert!(!round.due(BEFORE), "nothing new");
    let emptied = Grounds {
        newest_sealed: None,
        ..BEFORE
    };
    assert!(
        !round.due(emptied),
        "its own removals emptied the sealed list"
    );
    let sealed = Grounds {
        newest_sealed: Some(56),
        ..BEFORE
    };
    assert!(round.due(sealed), "a segment sealed while it ran or since");
    let held = Round {
        checked: Grounds { floor: 0, ..BEFORE },
        failed: false,
    };
    let raised = Grounds { floor: 1, ..BEFORE };
    assert!(held.due(raised), "a checkpoint or consumer floor raised");
}

/// A failed round runs on the next wake: only an event wakes the reclaimer
/// (a table landing, which after a failure follows room coming back or a
/// raised limit), so the failure does not repeat by itself.
#[test]
fn a_failed_round_runs_on_the_next_wake() {
    let round = Round {
        checked: BEFORE,
        failed: true,
    };
    assert!(round.due(BEFORE), "the wake is the new ground");
}
