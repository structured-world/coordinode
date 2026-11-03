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
