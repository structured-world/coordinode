use super::*;

fn slots(commands: usize, session_commands: usize) -> CommandSlots {
    CommandSlots::new(
        SlotCount::new(commands).expect("commands"),
        SlotCount::new(session_commands).expect("session commands"),
    )
}

/// A member runs no more commands at once than its slots, from all its
/// sessions and single calls together; a freed slot is taken again.
#[test]
fn the_member_bounds_every_command_however_it_arrives() {
    let slots = slots(2, 2);
    let (one, two) = (slots.session(), slots.session());
    let first = one.take().expect("the first command runs");
    let second = two.take().expect("another session's runs beside it");
    assert!(
        slots.take().is_none(),
        "a single call past the member's slots is refused"
    );
    assert!(
        one.take().is_none(),
        "a session's command past them is refused"
    );
    drop(first);
    assert!(slots.take().is_some(), "a freed slot is taken again");
    drop(second);
    assert_eq!(
        slots.available(),
        2,
        "every slot comes back once its command ends"
    );
}

/// A session runs no more commands at once than its own slots, and a session
/// at its bound leaves the member's slots to the others.
#[test]
fn a_session_at_its_bound_takes_nothing_of_the_member() {
    let slots = slots(4, 1);
    let session = slots.session();
    let _running = session.take().expect("the session's one command runs");
    assert!(session.take().is_none(), "its second is refused");
    assert_eq!(slots.available(), 3, "the refusal took no member slot");
    assert!(
        slots.session().take().is_some(),
        "another session still runs"
    );
}

/// A command that only ends work runs on a member out of slots, takes none of
/// them, and a session runs no more such commands at once than its own bound.
#[test]
fn an_ending_runs_on_a_full_member_within_its_sessions_bound() {
    let slots = slots(1, 2);
    let session = slots.session();
    let _busy = slots.take().expect("the member's one slot");
    let first = session.take_ending().expect("an ending on a full member");
    let _second = session.take_ending().expect("a second within the bound");
    assert!(
        session.take_ending().is_none(),
        "a third past the session's bound is refused"
    );
    assert_eq!(slots.available(), 0, "endings took no member slot");
    drop(first);
    assert!(
        session.take_ending().is_some(),
        "a freed one is taken again"
    );
}

/// No slots, or more than can be counted, is no count.
#[test]
fn a_count_of_none_or_too_many_is_refused() {
    assert!(SlotCount::new(0).is_none());
    assert!(SlotCount::new(SlotCount::MAX + 1).is_none());
    assert_eq!(SlotCount::new(1).map(SlotCount::get), Some(1));
    assert_eq!(
        SlotCount::new(SlotCount::MAX).map(SlotCount::get),
        Some(SlotCount::MAX)
    );
}
