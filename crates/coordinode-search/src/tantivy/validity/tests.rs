use super::*;

fn span(from: Option<i64>, until: Option<i64>) -> Validity {
    Validity { from, until }
}

/// A node is outside at an instant exactly when its interval does not
/// contain it: at or after its end, or before its start.
#[test]
fn outside_is_the_complement_of_contains() {
    let spans = [
        (1, span(Some(10), Some(20))),
        (2, span(None, Some(15))),
        (3, span(Some(12), None)),
        (4, Validity::ALWAYS),
        (5, span(Some(i64::MIN), Some(i64::MAX))),
    ];
    let mut held = Validities::default();
    for (node, validity) in spans {
        held.set(node, validity);
    }
    for at in [i64::MIN, 0, 9, 10, 11, 12, 14, 15, 19, 20, 21, i64::MAX] {
        let expected: Vec<u64> = spans
            .iter()
            .filter(|(_, validity)| !validity.contains(at))
            .map(|(node, _)| *node)
            .collect();
        assert_eq!(held.outside(at), expected, "at {at}");
    }
}

/// Setting a node again replaces its interval; setting it to every instant
/// forgets it.
#[test]
fn a_new_interval_replaces_the_old_one() {
    let mut held = Validities::default();
    held.set(7, span(Some(10), Some(20)));
    held.set(7, span(Some(20), Some(30)));
    assert_eq!(held.outside(15), vec![7]);
    assert!(held.outside(25).is_empty());
    assert_eq!(held.first_end(), Some(30));
    held.set(7, Validity::ALWAYS);
    assert!(held.outside(5).is_empty());
    assert_eq!(held.first_end(), None);
}

/// The first end is the earliest one among the nodes.
#[test]
fn the_first_end_is_the_earliest() {
    let mut held = Validities::default();
    held.set(1, span(None, Some(50)));
    held.set(2, span(Some(0), Some(30)));
    held.set(3, span(Some(5), None));
    assert_eq!(held.first_end(), Some(30));
    held.clear();
    assert_eq!(held.first_end(), None);
}
