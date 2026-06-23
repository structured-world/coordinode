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

    let bytes = interner.to_bytes();
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
    let bytes = interner.to_bytes();
    let restored = FieldInterner::from_bytes(&bytes).expect("deserialize failed");
    assert!(restored.is_empty());
}

#[test]
fn deserialize_new_ids_after_restore() {
    let mut interner = FieldInterner::new();
    interner.intern("a");
    interner.intern("b");

    let bytes = interner.to_bytes();
    let mut restored = FieldInterner::from_bytes(&bytes).expect("deserialize failed");

    // New field after restore should get next ID
    let id_c = restored.intern("c");
    assert_eq!(id_c, 3);
}

#[test]
fn deserialize_malformed_returns_none() {
    assert!(FieldInterner::from_bytes(&[]).is_none());
    assert!(FieldInterner::from_bytes(&[0xFF]).is_none());
    // Count says 1 entry but no data follows
    assert!(FieldInterner::from_bytes(&[1, 0, 0, 0]).is_none());
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
