use super::*;

#[test]
fn pack_unpack_roundtrip() {
    for level in [0u8, 1, 7, 16, 100, MAX_LEVEL] {
        for idx in [0u64, 1, 42, 1 << 30, IDX_MASK] {
            let (l, i) = unpack(pack(level, idx));
            assert_eq!(l, level, "level roundtrip");
            assert_eq!(i, idx, "idx roundtrip");
        }
    }
}

#[test]
fn new_is_empty() {
    let ep = EntryPoint::new();
    assert_eq!(ep.load(), None);
    assert_eq!(ep.for_search(), None);
}

#[test]
fn first_promote_installs() {
    let ep = EntryPoint::new();
    let out = ep.try_promote(3, 42);
    assert_eq!(out, PromoteOutcome::Installed);
    assert_eq!(ep.load(), Some((3, 42)));
}

#[test]
fn second_promote_with_lower_level_is_noop() {
    let ep = EntryPoint::new();
    ep.try_promote(5, 100);
    let out = ep.try_promote(3, 99);
    assert_eq!(out, PromoteOutcome::NotNeeded { current_level: 5 });
    assert_eq!(ep.load(), Some((5, 100)));
}

#[test]
fn second_promote_with_equal_level_is_noop() {
    let ep = EntryPoint::new();
    ep.try_promote(5, 100);
    // Two nodes hit the same novel max-level: linearisation rule
    // is "first wins" — the second is a no-op.
    let out = ep.try_promote(5, 200);
    assert_eq!(out, PromoteOutcome::NotNeeded { current_level: 5 });
    assert_eq!(ep.load(), Some((5, 100)));
}

#[test]
fn promote_with_higher_level_replaces() {
    let ep = EntryPoint::new();
    ep.try_promote(3, 42);
    let out = ep.try_promote(7, 99);
    assert_eq!(out, PromoteOutcome::Installed);
    assert_eq!(ep.load(), Some((7, 99)));
}

#[test]
fn for_search_tracks_promote() {
    let ep = EntryPoint::new();
    assert_eq!(ep.for_search(), None);
    ep.try_promote(3, 0);
    assert_eq!(ep.for_search(), Some((0, 3)));
    ep.try_promote(7, 1);
    assert_eq!(ep.for_search(), Some((1, 7)));
    // Lower-level promote attempt does not lower the snapshot.
    ep.try_promote(2, 2);
    assert_eq!(ep.for_search(), Some((1, 7)));
}
