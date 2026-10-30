use super::*;

const OP: GenerationId = GenerationId::from_raw(10);
const OLDER: GenerationId = GenerationId::from_raw(4);

fn following(prepared: bool) -> Followed {
    Followed {
        current: Some((OP, prepared)),
        settled: None,
    }
}

fn never(_: GenerationId) -> Result<Option<Change>, StoreError> {
    panic!("the outcome is not asked for while the parts are kept")
}

/// While the change is certified or after it is withdrawn, the facts in
/// force hold: an index following it leaves the gained documents out, and
/// one not following it yet holds the facts in force as they are.
#[test]
fn the_facts_in_force_read_without_the_gained_documents() {
    for change in [Change::Proposed(OP), Change::Withdrawn(OP)] {
        for prepared in [false, true] {
            assert_eq!(
                answer(change, following(prepared), never).unwrap(),
                Answer::Without(Scope::Gained)
            );
        }
        assert_eq!(
            answer(change, Followed::default(), never).unwrap(),
            Answer::Whole
        );
    }
}

/// Once the change is in force, only an index that wrote every stored node
/// in its scope holds it: it leaves the retired documents out. One still
/// preparing, or not following the change at all, cannot answer.
#[test]
fn the_change_in_force_needs_a_prepared_index() {
    let change = Change::InForce(OP);
    assert_eq!(
        answer(change, following(true), never).unwrap(),
        Answer::Without(Scope::Retired)
    );
    assert_eq!(
        answer(change, following(false), never).unwrap(),
        Answer::Unable
    );
    assert_eq!(
        answer(change, Followed::default(), never).unwrap(),
        Answer::Unable
    );
}

/// An index that brought this change, or a later one, back to one
/// interpretation holds the outcome alone, whatever the view still sees.
#[test]
fn a_settled_change_reads_the_whole_index() {
    let settled = Followed {
        current: None,
        settled: Some(OP),
    };
    for change in [
        Change::Proposed(OLDER),
        Change::InForce(OLDER),
        Change::InForce(OP),
        Change::Withdrawn(OP),
    ] {
        assert_eq!(answer(change, settled, never).unwrap(), Answer::Whole);
    }
    assert_eq!(
        answer(Change::InForce(GenerationId::from_raw(11)), settled, never).unwrap(),
        Answer::Unable,
        "a later change in force is not settled"
    );
}

/// Once the parts are folded, the view no longer shows the change: the
/// record of the change the index keeps says which scope belongs to the
/// outcome, and an index keeping a change whose outcome is unknown cannot
/// answer.
#[test]
fn a_folded_change_is_read_by_its_outcome() {
    let in_force = |_: GenerationId| Ok(Some(Change::InForce(OP)));
    let withdrawn = |_: GenerationId| Ok(Some(Change::Withdrawn(OP)));
    let unknown = |_: GenerationId| Ok(None);
    assert_eq!(
        answer(Change::None, following(true), in_force).unwrap(),
        Answer::Without(Scope::Retired)
    );
    assert_eq!(
        answer(Change::None, following(true), withdrawn).unwrap(),
        Answer::Without(Scope::Gained)
    );
    assert_eq!(
        answer(Change::None, following(true), unknown).unwrap(),
        Answer::Unable
    );
    assert_eq!(
        answer(Change::None, Followed::default(), never).unwrap(),
        Answer::Whole
    );
}

/// An index keeping another change than the one the view sees does not
/// hold the view's interpretation.
#[test]
fn another_change_kept_cannot_answer() {
    let other = Followed {
        current: Some((OLDER, true)),
        settled: None,
    };
    for change in [
        Change::Proposed(OP),
        Change::InForce(OP),
        Change::Withdrawn(OP),
    ] {
        assert_eq!(answer(change, other, never).unwrap(), Answer::Unable);
    }
}
