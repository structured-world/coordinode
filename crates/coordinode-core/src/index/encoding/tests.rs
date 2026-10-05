use super::*;

fn enc(value: Value) -> Vec<u8> {
    encode_tuple(&[value]).expect("indexable")
}

/// Types sort in tag order, values within a type in value order.
#[test]
fn encoding_preserves_order() {
    let ordered = [
        Value::Null,
        Value::Bool(false),
        Value::Bool(true),
        Value::Int(i64::MIN),
        Value::Int(-1),
        Value::Int(0),
        Value::Int(i64::MAX),
        Value::Float(f64::NEG_INFINITY),
        Value::Float(-1.5),
        Value::Float(0.0),
        Value::Float(3.5),
        Value::Float(f64::INFINITY),
        Value::String(String::new()),
        Value::String("a".into()),
        Value::String("a\0".into()),
        Value::String("ab".into()),
        Value::Timestamp(1),
        Value::Timestamp(2),
        Value::Binary(vec![]),
        Value::Binary(vec![0]),
        Value::Binary(vec![1]),
    ];
    for pair in ordered.windows(2) {
        assert!(
            enc(pair[0].clone()) < enc(pair[1].clone()),
            "{:?} must sort before {:?}",
            pair[0],
            pair[1]
        );
    }
}

fn generation(raw: u64) -> GenerationId {
    GenerationId::from_raw(raw)
}

/// A string holding NUL followed by the tuple terminator must not share a
/// prefix with the shorter string: an encoding that wrote NUL unescaped made
/// the entries of "a" a prefix of those of "a\0:x", and a lookup of one
/// returned the other.
#[test]
fn a_string_is_never_a_key_prefix_of_another() {
    let short = entry_value_prefix(generation(1), &enc(Value::String("a".into())));
    let long = encode_entry_key(generation(1), &enc(Value::String("a\0:x".into())), 7);
    assert!(!long.starts_with(&short));
}

/// Composite values have no key; the old encoding mapped them all to NULL,
/// so two maps under a unique index collided with each other and with NULL.
#[test]
fn composite_values_are_unindexable() {
    assert_eq!(
        encode_tuple(&[Value::Map(Default::default())]),
        Err(Unindexable::Kind("a map"))
    );
    assert_eq!(
        encode_tuple(&[Value::Array(vec![])]),
        Err(Unindexable::Kind("a list"))
    );
    assert_eq!(
        encode_tuple(&[Value::Vector(vec![1.0])]),
        Err(Unindexable::Kind("a vector"))
    );
}

/// -0.0 equals 0.0, so it is the same key; NaN equals nothing, so it has none.
#[test]
fn float_keys_follow_float_equality() {
    assert_eq!(enc(Value::Float(-0.0)), enc(Value::Float(0.0)));
    assert_eq!(
        encode_tuple(&[Value::Float(f64::NAN)]),
        Err(Unindexable::NaN)
    );
}

/// A compound tuple decodes one way: shifting bytes between two string
/// elements changes the encoding.
#[test]
fn compound_tuples_are_injective() {
    let a = encode_tuple(&[Value::String("ab".into()), Value::String("c".into())]);
    let b = encode_tuple(&[Value::String("a".into()), Value::String("bc".into())]);
    assert_ne!(a, b);
}

/// Entries sort by value, then by node id, and carry the id at the end.
#[test]
fn non_unique_entries_sort_by_value_then_id() {
    let alice = enc(Value::String("alice".into()));
    let bob = enc(Value::String("bob".into()));
    let k1 = encode_entry_key(generation(3), &alice, 1);
    let k2 = encode_entry_key(generation(3), &alice, 2);
    let k3 = encode_entry_key(generation(3), &bob, 1);
    assert!(k1 < k2 && k2 < k3);
    assert_eq!(decode_entry(generation(3), &k2), Some((2, None)));
    assert!(k1.starts_with(&entry_value_prefix(generation(3), &alice)));
    assert!(k1.starts_with(&entries_prefix(generation(3))));
}

/// An entry of a temporal node's version carries the node and the version
/// start: a node's versions sort together, oldest first (negative starts
/// before positive), and both decode back whatever the tuple holds.
#[test]
fn version_entries_carry_node_and_valid_from() {
    let tricky = encode_tuple(&[
        Value::String("a\0:\u{ff}".into()),
        Value::Int(58),
        Value::Binary(vec![0, b':', 0xFF, 0]),
        Value::Bool(true),
    ])
    .expect("indexable");
    let g = generation(0x3A00_3A00_FF00_003A);
    let older = encode_version_entry_key(g, &tricky, 7, -5);
    let newer = encode_version_entry_key(g, &tricky, 7, 3);
    let other = encode_version_entry_key(g, &tricky, 8, i64::MIN);
    assert!(older < newer && newer < other);
    assert!(older.starts_with(&entry_value_prefix(g, &tricky)));
    assert_eq!(decode_entry(g, &older), Some((7, Some(-5))));
    assert_eq!(decode_entry(g, &newer), Some((7, Some(3))));
    assert_eq!(decode_entry(g, &other), Some((8, Some(i64::MIN))));
    assert_eq!(
        decode_entry(g, &encode_entry_key(g, &tricky, 9)),
        Some((9, None))
    );
}

/// A key of another generation, a truncated owner or a malformed tuple
/// decodes to nothing rather than to a wrong node.
#[test]
fn foreign_or_malformed_keys_decode_to_nothing() {
    let value = enc(Value::Int(1));
    let key = encode_entry_key(generation(1), &value, 1);
    assert_eq!(decode_entry(generation(2), &key), None);
    assert_eq!(decode_entry(generation(1), &key[..key.len() - 1]), None);
    let mut bad = entries_prefix(generation(1)).to_vec();
    bad.extend_from_slice(&[0x7F, b':']);
    bad.extend_from_slice(&1u64.to_be_bytes());
    assert_eq!(decode_entry(generation(1), &bad), None);
}

/// One generation's entries never fall under another's prefix or range,
/// at the boundaries of the fixed-width number too, and the unique and
/// non-unique shapes never meet.
#[test]
fn generations_are_disjoint() {
    let value = enc(Value::Int(1));
    for (a, b) in [
        (1, 2),
        (0xFF, 0x100),
        (u64::MAX - 1, u64::MAX),
        (0, u64::MAX),
    ] {
        let key = encode_entry_key(generation(a), &value, 1);
        assert!(!key.starts_with(&entries_prefix(generation(b))));
        for (start, end) in generation_ranges(generation(b)) {
            assert!(
                !(start.as_slice() <= key.as_slice() && key.as_slice() < end.as_slice()),
                "generation {a} entry inside generation {b}'s range"
            );
        }
    }
    let g = generation(5);
    let unique = encode_unique_entry_key(g, &value);
    assert!(!unique.starts_with(&entries_prefix(g)));
    let [(start, end), (ustart, uend)] = generation_ranges(g);
    let entry = encode_entry_key(g, &value, 1);
    assert!(start <= entry && entry < end);
    assert!(ustart <= unique && unique < uend);
}

/// The last generation's ranges still end: the range end steps the tag.
#[test]
fn the_last_generation_has_a_range_end() {
    let [(start, end), _] = generation_ranges(generation(u64::MAX));
    assert!(start < end);
    let key = encode_entry_key(generation(u64::MAX), &enc(Value::Int(1)), 1);
    assert!(start <= key && key < end);
}

/// The filter prefixes of an entry are found structurally: the generation
/// prefix and the value prefix, however many separator and zero bytes the
/// generation number and the values hold; a key of no entry shape has none.
#[test]
fn scan_prefixes_follow_the_structure() {
    let g = generation(0x3A3A_003A_0000_FF3A);
    let tuple = encode_tuple(&[
        Value::String("a:\0b".into()),
        Value::Binary(vec![b':', 0, 0xFF, b':']),
    ])
    .expect("indexable");
    let key = encode_version_entry_key(g, &tuple, 0x3A00_0000_0000_003A, -1);
    let prefixes: Vec<&[u8]> = entry_scan_prefixes(&key).expect("an entry").collect();
    assert_eq!(
        prefixes,
        vec![
            &entries_prefix(g)[..],
            entry_value_prefix(g, &tuple).as_slice()
        ]
    );

    let unique = encode_unique_entry_key(g, &tuple);
    let prefixes: Vec<&[u8]> = entry_scan_prefixes(&unique).expect("an entry").collect();
    assert_eq!(prefixes, vec![&unique_entries_prefix(g)[..]]);

    assert!(entry_scan_prefixes(b"idx:\0\0\0\x01a:x").is_none());
    assert!(entry_scan_prefixes(&[0x00, 1, 2]).is_none());
    assert!(
        entry_scan_prefixes(&[0x01, 0, 0]).is_none(),
        "shorter than a prefix"
    );
}
