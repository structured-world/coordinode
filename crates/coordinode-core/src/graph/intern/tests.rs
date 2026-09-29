use super::*;

#[test]
fn intern_new_field() {
    let mut interner = FieldInterner::new();
    let id = interner.intern("name");
    assert_eq!(id, 1);
    assert_eq!(interner.len(), 1);
}

#[test]
fn intern_returns_same_id() {
    let mut interner = FieldInterner::new();
    let id1 = interner.intern("name");
    let id2 = interner.intern("name");
    assert_eq!(id1, id2);
    assert_eq!(interner.len(), 1);
}

#[test]
fn intern_multiple_fields() {
    let mut interner = FieldInterner::new();
    let id_name = interner.intern("name");
    let id_age = interner.intern("age");
    let id_email = interner.intern("email");

    assert_eq!(id_name, 1);
    assert_eq!(id_age, 2);
    assert_eq!(id_email, 3);
    assert_eq!(interner.len(), 3);
}

#[test]
fn resolve_existing_id() {
    let mut interner = FieldInterner::new();
    let id = interner.intern("name");
    assert_eq!(interner.resolve(id), Some("name"));
}

#[test]
fn resolve_nonexistent_id() {
    let interner = FieldInterner::new();
    assert_eq!(interner.resolve(999), None);
}

#[test]
fn resolve_reserved_id() {
    let interner = FieldInterner::new();
    assert_eq!(interner.resolve(FieldInterner::RESERVED_ID), None);
}

#[test]
fn lookup_existing() {
    let mut interner = FieldInterner::new();
    interner.intern("name");
    assert_eq!(interner.lookup("name"), Some(1));
}

#[test]
fn lookup_missing() {
    let interner = FieldInterner::new();
    assert_eq!(interner.lookup("name"), None);
}

#[test]
fn empty_interner() {
    let interner = FieldInterner::new();
    assert!(interner.is_empty());
    assert_eq!(interner.len(), 0);
}

#[test]
fn serialize_deserialize_roundtrip() {
    let mut interner = FieldInterner::new();
    interner.intern("name");
    interner.intern("age");
    interner.intern("email");

    let bytes = interner.to_bytes().expect("serialize failed");
    let restored = FieldInterner::from_bytes(&bytes).expect("deserialize failed");

    assert_eq!(restored.len(), 3);
    assert_eq!(restored.lookup("name"), Some(1));
    assert_eq!(restored.lookup("age"), Some(2));
    assert_eq!(restored.lookup("email"), Some(3));
    assert_eq!(restored.resolve(1), Some("name"));
    assert_eq!(restored.resolve(2), Some("age"));
    assert_eq!(restored.resolve(3), Some("email"));
}

#[test]
fn serialize_empty() {
    let interner = FieldInterner::new();
    let bytes = interner.to_bytes().expect("serialize failed");
    let restored = FieldInterner::from_bytes(&bytes).expect("deserialize failed");
    assert!(restored.is_empty());
}

#[test]
fn deserialize_new_ids_after_restore() {
    let mut interner = FieldInterner::new();
    interner.intern("a");
    interner.intern("b");

    let bytes = interner.to_bytes().expect("serialize failed");
    let mut restored = FieldInterner::from_bytes(&bytes).expect("deserialize failed");

    // New field after restore should get next ID
    let id_c = restored.intern("c");
    assert_eq!(id_c, 3);
}

#[test]
fn deserialize_malformed_is_an_error() {
    assert!(FieldInterner::from_bytes(&[]).is_err());
    assert!(FieldInterner::from_bytes(&[0xFF]).is_err());
    // Count says 1 entry but no data follows
    assert!(FieldInterner::from_bytes(&[1, 0, 0, 0]).is_err());
}

/// Serialized bytes of the given (id, name) pairs, in the given order.
fn encoded(pairs: &[(u32, &str)]) -> Vec<u8> {
    let mut buf = (pairs.len() as u32).to_le_bytes().to_vec();
    for (id, name) in pairs {
        buf.extend_from_slice(&id.to_le_bytes());
        buf.extend_from_slice(&(name.len() as u16).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
    }
    buf
}

/// A decoder that accepted a name bound to two ids, or an id bound to two
/// names, would hand stored data two meanings; each is refused instead.
#[test]
fn deserialize_refuses_contradictory_bindings() {
    assert!(matches!(
        FieldInterner::from_bytes(&encoded(&[(1, "a"), (2, "a")])),
        Err(DictionaryError::NameConflict { .. })
    ));
    assert!(matches!(
        FieldInterner::from_bytes(&encoded(&[(1, "a"), (1, "b")])),
        Err(DictionaryError::IdConflict { id: 1, .. })
    ));
    assert!(matches!(
        FieldInterner::from_bytes(&encoded(&[(0, "a")])),
        Err(DictionaryError::ReservedId { .. })
    ));
}

/// Trailing bytes mean the length prefix and the content disagree.
#[test]
fn deserialize_refuses_trailing_bytes() {
    let mut bytes = encoded(&[(1, "a")]);
    bytes.push(0);
    assert!(matches!(
        FieldInterner::from_bytes(&bytes),
        Err(DictionaryError::Malformed(_))
    ));
}

/// A count far larger than the bytes can hold is refused before allocating.
#[test]
fn deserialize_refuses_a_count_the_data_cannot_hold() {
    let bytes = u32::MAX.to_le_bytes();
    assert!(matches!(
        FieldInterner::from_bytes(&bytes),
        Err(DictionaryError::Malformed(_))
    ));
}

/// Clones share the published view; bindings recorded through one handle
/// stay in that handle.
#[test]
fn clones_share_the_published_view_but_not_local_additions() {
    let base = FieldInterner::from_bindings([("a".to_owned(), 1), ("b".to_owned(), 2)])
        .expect("valid bindings");
    let mut query = base.clone();
    query.insert_binding("c", 7).expect("new binding");
    assert_eq!(query.lookup("c"), Some(7));
    assert_eq!(query.resolve(7), Some("c"));
    assert_eq!(base.lookup("c"), None, "the published view is untouched");
    assert_eq!(query.frontier(), 7);
    assert_eq!(base.frontier(), 2);
}

/// A binding that contradicts one already held is refused, and the same
/// binding again is accepted.
#[test]
fn insert_binding_refuses_contradictions_and_accepts_repeats() {
    let mut view = FieldInterner::from_bindings([("a".to_owned(), 1)]).expect("valid");
    view.insert_binding("a", 1).expect("the same binding again");
    assert!(matches!(
        view.insert_binding("a", 2),
        Err(DictionaryError::NameConflict { .. })
    ));
    assert!(matches!(
        view.insert_binding("z", 1),
        Err(DictionaryError::IdConflict { .. })
    ));
    assert!(matches!(
        view.insert_binding("z", 0),
        Err(DictionaryError::ReservedId { .. })
    ));
}

/// Extending a view publishes the union, keeps ids, and refuses a
/// contradiction without changing the view it started from.
#[test]
fn extended_publishes_the_union() {
    let base = FieldInterner::from_bindings([("a".to_owned(), 1)]).expect("valid");
    let next = base
        .extended([("b".to_owned(), 5), ("a".to_owned(), 1)])
        .expect("compatible");
    assert_eq!(next.lookup("b"), Some(5));
    assert_eq!(next.resolve(1), Some("a"));
    assert_eq!(next.frontier(), 5);
    assert!(base.extended([("a".to_owned(), 9)]).is_err());
    assert_eq!(base.lookup("a"), Some(1));
}

/// An adopted id far above the others (a restore can bring one) resolves
/// without an allocation sized to the id.
#[test]
fn a_sparse_id_resolves_without_allocating_its_range() {
    let view = FieldInterner::from_bindings([("low".to_owned(), 1), ("high".to_owned(), u32::MAX)])
        .expect("valid bindings");
    assert_eq!(view.resolve(u32::MAX), Some("high"));
    assert_eq!(view.resolve(1), Some("low"));
    assert_eq!(view.resolve(2), None);
    assert_eq!(view.frontier(), u32::MAX);
}

/// A local intern continues above every binding the handle holds.
#[test]
fn intern_continues_above_adopted_bindings() {
    let mut view = FieldInterner::new();
    view.insert_binding("x", 10).expect("binding");
    assert_eq!(view.intern("y"), 11);
}

#[test]
fn registration_limits() {
    assert!(validate_registration(&["a", "b"]).is_ok());
    let long = "x".repeat(MAX_FIELD_NAME_BYTES + 1);
    assert!(matches!(
        validate_registration(&[long.as_str()]),
        Err(DictionaryError::NameTooLong { .. })
    ));
    let many = vec!["n"; MAX_REGISTRATION_BATCH + 1];
    assert!(matches!(
        validate_registration(&many),
        Err(DictionaryError::BatchTooLarge { .. })
    ));
}

#[test]
fn storage_keys_round_trip() {
    assert_eq!(decode_field_id_key(&field_id_key(42)), Some(42));
    assert_eq!(decode_field_id_key(&field_name_key("x")), None);
    assert_eq!(decode_field_id(&encode_field_id(7)), Some(7));
    assert_eq!(decode_field_id(&[1, 2, 3]), None);
    // Big-endian id keys sort by id, so the last one is the frontier.
    assert!(field_id_key(255) < field_id_key(256));
}

// Varint tests

#[test]
fn varint_single_byte() {
    let mut buf = [0u8; 5];
    let len = encode_varint(0, &mut buf);
    assert_eq!(len, 1);
    assert_eq!(buf[0], 0);

    let len = encode_varint(127, &mut buf);
    assert_eq!(len, 1);
    assert_eq!(buf[0], 127);
}

#[test]
fn varint_two_bytes() {
    let mut buf = [0u8; 5];
    let len = encode_varint(128, &mut buf);
    assert_eq!(len, 2);

    let (val, consumed) = decode_varint(&buf[..len]).expect("decode failed");
    assert_eq!(val, 128);
    assert_eq!(consumed, 2);
}

#[test]
fn varint_roundtrip() {
    let test_values = [0, 1, 127, 128, 255, 256, 16383, 16384, 65535, u32::MAX];
    for &v in &test_values {
        let mut buf = [0u8; 5];
        let len = encode_varint(v, &mut buf);
        let (decoded, consumed) = decode_varint(&buf[..len]).expect("decode failed");
        assert_eq!(decoded, v, "roundtrip failed for {v}");
        assert_eq!(consumed, len);
    }
}

#[test]
fn varint_size_boundaries() {
    let mut buf = [0u8; 5];
    // 1 byte: 0-127
    assert_eq!(encode_varint(127, &mut buf), 1);
    // 2 bytes: 128-16383
    assert_eq!(encode_varint(128, &mut buf), 2);
    assert_eq!(encode_varint(16383, &mut buf), 2);
    // 3 bytes: 16384-2097151
    assert_eq!(encode_varint(16384, &mut buf), 3);
    // 5 bytes: u32::MAX
    assert_eq!(encode_varint(u32::MAX, &mut buf), 5);
}

#[test]
fn varint_decode_empty() {
    assert!(decode_varint(&[]).is_none());
}

#[test]
fn varint_decode_incomplete() {
    // High bit set but no continuation byte
    assert!(decode_varint(&[0x80]).is_none());
}

#[test]
fn iter_fields() {
    let mut interner = FieldInterner::new();
    interner.intern("a");
    interner.intern("b");
    interner.intern("c");

    let mut pairs: Vec<_> = interner.iter().collect();
    pairs.sort_by_key(|&(_, id)| id);
    assert_eq!(pairs, vec![("a", 1), ("b", 2), ("c", 3)]);
}
