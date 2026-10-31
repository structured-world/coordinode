use super::*;
use coordinode_core::schema::label_id::LabelId;

fn id(raw: u32) -> LabelId {
    LabelId::new(raw).expect("a label id is never zero")
}

fn carriers(implying: &[u32]) -> Option<LabelCarriers> {
    Some(LabelCarriers {
        label: id(1),
        implying: implying.iter().copied().map(id).collect(),
    })
}

/// An index built under one interpretation answers its readers whole and
/// cannot answer a reader of another, whichever of the two came first:
/// the index's interpretation is compared, never how recent it is.
#[test]
fn a_built_index_answers_only_its_interpretation() {
    let old = carriers(&[]);
    let new = carriers(&[2]);
    let held = Held::built(new.clone());
    assert_eq!(answer(new.as_ref(), &held), Answer::Whole);
    assert_eq!(
        answer(old.as_ref(), &held),
        Answer::Unable,
        "a reader before the change"
    );
    let held = Held::built(old.clone());
    assert_eq!(
        answer(new.as_ref(), &held),
        Answer::Unable,
        "a reader after it"
    );
}

/// While a change is followed, a reader of the base leaves the gained
/// documents out, and a reader of the candidate the retired ones, but only
/// once every stored node was written in its scope.
#[test]
fn a_followed_change_answers_both_interpretations() {
    let old = carriers(&[]);
    let new = carriers(&[2]);
    let preparing = Held {
        base: Some(old.clone()),
        candidate: Some((new.clone(), false)),
    };
    assert_eq!(
        answer(old.as_ref(), &preparing),
        Answer::Without(Scope::Gained)
    );
    assert_eq!(answer(new.as_ref(), &preparing), Answer::Preparing);
    let prepared = Held {
        candidate: Some((new.clone(), true)),
        ..preparing
    };
    assert_eq!(
        answer(old.as_ref(), &prepared),
        Answer::Without(Scope::Gained)
    );
    assert_eq!(
        answer(new.as_ref(), &prepared),
        Answer::Without(Scope::Retired)
    );
    assert_eq!(
        answer(carriers(&[3]).as_ref(), &prepared),
        Answer::Unable,
        "a third interpretation"
    );
}

/// While the retired documents of a change put in force are removed, the
/// base is answered by nobody and the candidate still is.
#[test]
fn a_change_being_settled_answers_only_the_candidate() {
    let old = carriers(&[]);
    let new = carriers(&[2]);
    let settling = Held {
        base: None,
        candidate: Some((new.clone(), true)),
    };
    assert_eq!(answer(old.as_ref(), &settling), Answer::Unable);
    assert_eq!(
        answer(new.as_ref(), &settling),
        Answer::Without(Scope::Retired)
    );
}

/// An index not built yet answers nobody, and a label without an identity
/// is an interpretation of its own.
#[test]
fn an_unbuilt_index_answers_nobody() {
    assert_eq!(
        answer(carriers(&[]).as_ref(), &Held::default()),
        Answer::Unable
    );
    assert_eq!(answer(None, &Held::built(None)), Answer::Whole);
    assert_eq!(
        answer(carriers(&[]).as_ref(), &Held::built(None)),
        Answer::Unable
    );
}
