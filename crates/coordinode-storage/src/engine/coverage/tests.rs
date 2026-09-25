use super::*;

#[test]
fn applied_set_advances_over_out_of_order_marks() {
    // Indices applied out of order must not move the prefix past a gap, and
    // filling the gap must absorb every index above it.
    let mut set = AppliedSet::starting_at(5);
    set.mark(7);
    set.mark(8);
    assert_eq!(set.next(), 5, "a gap at 5 holds the prefix");
    assert_eq!(set.above().collect::<Vec<_>>(), vec![7, 8]);
    set.mark(6);
    assert_eq!(set.next(), 5);
    set.mark(5);
    assert_eq!(set.next(), 9, "filling the gap absorbs 6, 7 and 8");
    assert_eq!(set.above().count(), 0);
    set.mark(3);
    assert_eq!(set.next(), 9, "an index below the prefix changes nothing");
}

#[test]
fn marker_keys_sort_by_index_inside_the_reserved_namespace() {
    // Big-endian indices keep the marker range contiguous and ordered, so a
    // range tombstone over [marker(a, 0), marker(b, 0)) removes exactly a..b,
    // every sub of an index included.
    let d = Domain::Journal;
    assert!(d.marker_key(1, 0) < d.marker_key(1, 1));
    assert!(d.marker_key(1, u32::MAX) < d.marker_key(2, 0));
    assert!(d.marker_key(255, 0) < d.marker_key(256, 0));
    let end = d.marker_end();
    assert!(d.marker_key(u64::MAX, u32::MAX).as_slice() < end.as_slice());
    let base = d.base_key();
    assert!(base.as_slice() < d.marker_key(0, 0).as_slice());
    for key in [
        base.as_slice(),
        d.marker_key(0, 0).as_slice(),
        end.as_slice(),
    ] {
        assert!(is_reserved(key));
        assert!(key < USER_KEYSPACE_START);
    }
    assert!(!is_reserved(b"node:00:0001"));
    assert_eq!(decode_marker(&d.marker_key(7, 3)), Some((7, 3)));
    assert_eq!(
        decode_marker(b"\x00cjm"),
        None,
        "a truncated marker is rejected"
    );
}

#[test]
fn clamp_keeps_user_range_deletes_out_of_the_reserved_namespace() {
    assert_eq!(clamp_user_start(b""), USER_KEYSPACE_START);
    assert_eq!(
        clamp_user_start(&Domain::Journal.base_key()),
        USER_KEYSPACE_START
    );
    assert_eq!(clamp_user_start(b"node:"), b"node:");
}

#[test]
fn base_round_trips_with_its_payload_and_rejects_a_short_value() {
    let value = encode_base(42, b"last-log-id");
    assert_eq!(
        decode_base(&value).expect("decode"),
        (42, b"last-log-id".as_slice())
    );
    assert_eq!(
        decode_base(&encode_base(9, &[])).expect("decode"),
        (9, &[][..])
    );
    assert!(decode_base(&[1, 2, 3]).is_err());
}

#[test]
fn fold_is_due_only_after_enough_applied_indices() {
    let coverage = Coverage::new(0);
    for index in 0..FOLD_EVERY - 1 {
        assert!(
            !coverage.mark_applied(index),
            "no fold before {FOLD_EVERY} indices"
        );
    }
    assert!(coverage.mark_applied(FOLD_EVERY - 1));
    let mut folded = coverage.lock_folded();
    folded.set_folded(FOLD_EVERY);
    drop(folded);
    assert!(
        !coverage.mark_applied(FOLD_EVERY),
        "a fold resets the count"
    );
}
