use super::Round;

/// A round is due again only when it may remove what the last could not: a
/// journal that moved, or room come back after a failure. A round that
/// succeeded, or failed with nothing changed, does not run again by itself.
#[test]
fn a_round_is_due_only_when_the_journal_or_the_room_moved() {
    let done = Round {
        segments: (0, 2),
        charged: 100,
        failed: false,
    };
    assert!(!done.due((0, 2), 100), "nothing new after a success");
    assert!(
        !done.due((0, 2), 50),
        "room alone does not repeat a success"
    );
    assert!(done.due((0, 3), 100), "a segment sealed since");
    assert!(done.due((40, 1), 100), "segments went since");

    let failed = Round {
        failed: true,
        ..done
    };
    assert!(!failed.due((0, 2), 100), "an unchanged failure waits");
    assert!(!failed.due((0, 2), 120), "less room does not help it");
    assert!(failed.due((0, 2), 99), "room came back");
}
