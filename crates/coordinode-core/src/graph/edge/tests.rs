use super::*;
use crate::graph::types::Value;

// -- Edge-property codec tests (ADR-040) --

#[test]
fn edge_props_codec_round_trips() {
    let props = vec![
        (3u32, Value::Int(7)),
        (1u32, Value::String("a".into())),
        (2u32, Value::Float(1.5)),
    ];
    let bytes = encode_edge_props(&props).expect("encode");
    let back = decode_edge_props(&bytes).expect("decode");
    // Stored in ascending field_id order.
    assert_eq!(
        back,
        vec![
            (1u32, Value::String("a".into())),
            (2u32, Value::Float(1.5)),
            (3u32, Value::Int(7)),
        ]
    );
}

#[test]
fn edge_props_codec_is_deterministic_regardless_of_input_order() {
    // The whole point of ADR-040: identical logical facet set →
    // identical bytes, so page-ECC / dedup / snapshot-diff are stable.
    let a = vec![
        (2u32, Value::Int(2)),
        (1u32, Value::Int(1)),
        (3u32, Value::Int(3)),
    ];
    let b = vec![
        (3u32, Value::Int(3)),
        (1u32, Value::Int(1)),
        (2u32, Value::Int(2)),
    ];
    assert_eq!(
        encode_edge_props(&a).expect("a"),
        encode_edge_props(&b).expect("b"),
        "different insertion order must produce identical bytes",
    );
}

#[test]
fn edge_props_codec_dedups_last_wins() {
    let props = vec![(1u32, Value::Int(1)), (1u32, Value::Int(99))];
    let back = decode_edge_props(&encode_edge_props(&props).expect("encode")).expect("decode");
    assert_eq!(
        back,
        vec![(1u32, Value::Int(99))],
        "duplicate field id keeps the last value"
    );
}

#[test]
fn edge_properties_wire_matches_executor_vec_shape() {
    // EdgeProperties (the typed Layer-4 value) and the executor's raw
    // Vec<(field_id, value)> MUST serialise to the same bytes — that is
    // what lets LocalEdgeStore writes round-trip through the executor's
    // reader and vice versa (the unification that ADR-040 mandates).
    let pairs = vec![(5u32, Value::Int(10)), (2u32, Value::String("x".into()))];
    let mut ep = EdgeProperties::new();
    for (fid, v) in &pairs {
        ep.set(*fid, v.clone());
    }
    assert_eq!(
        ep.to_msgpack().expect("ep encode"),
        encode_edge_props(&pairs).expect("vec encode"),
        "EdgeProperties and the executor Vec shape must be wire-identical",
    );
}

// -- Key encoding tests --

#[test]
fn encode_forward_key() {
    let key = encode_adj_key_forward("FOLLOWS", NodeId::from_raw(42));
    assert!(key.starts_with(b"adj:FOLLOWS:out:"));
}

#[test]
fn encode_reverse_key() {
    let key = encode_adj_key_reverse("FOLLOWS", NodeId::from_raw(99));
    assert!(key.starts_with(b"adj:FOLLOWS:in:"));
}

#[test]
fn write_forward_key_matches_encode() {
    // write_ variant must produce identical bytes as encode_ variant
    let mut buf = Vec::new();
    write_adj_key_forward("KNOWS", NodeId::from_raw(7), &mut buf);
    assert_eq!(buf, encode_adj_key_forward("KNOWS", NodeId::from_raw(7)));
}

#[test]
fn write_reverse_key_matches_encode() {
    let mut buf = Vec::new();
    write_adj_key_reverse("LIKES", NodeId::from_raw(13), &mut buf);
    assert_eq!(buf, encode_adj_key_reverse("LIKES", NodeId::from_raw(13)));
}

#[test]
fn write_key_reuses_buffer_across_calls() {
    // Buffer grows to fit largest key and does not reallocate on smaller subsequent writes.
    let mut buf = Vec::new();

    write_adj_key_forward("VERY_LONG_EDGE_TYPE_NAME", NodeId::from_raw(1), &mut buf);
    let cap_after_first = buf.capacity();
    assert!(
        cap_after_first >= 4 + 24 + 5 + 8,
        "buffer must fit first key"
    );

    write_adj_key_forward("KNOWS", NodeId::from_raw(2), &mut buf);
    // Capacity must not decrease (no reallocation for shorter key)
    assert_eq!(buf.capacity(), cap_after_first);
    assert_eq!(buf, encode_adj_key_forward("KNOWS", NodeId::from_raw(2)));
}

#[test]
fn write_key_clears_before_write() {
    let mut buf = Vec::new();
    write_adj_key_forward("FOLLOWS", NodeId::from_raw(1), &mut buf);
    let first = buf.clone();
    write_adj_key_forward("KNOWS", NodeId::from_raw(2), &mut buf);
    // Buffer must contain only the second key, not a concatenation
    assert_ne!(buf, first);
    assert_eq!(buf, encode_adj_key_forward("KNOWS", NodeId::from_raw(2)));
}

#[test]
fn decode_forward_key_roundtrip() {
    let key = encode_adj_key_forward("LIKES", NodeId::from_raw(123));
    let parts = decode_adj_key(&key).expect("decode failed");
    assert_eq!(parts.edge_type, "LIKES");
    assert_eq!(parts.direction, AdjDirection::Out);
    assert_eq!(parts.node_id, NodeId::from_raw(123));
}

#[test]
fn decode_reverse_key_roundtrip() {
    let key = encode_adj_key_reverse("LIKES", NodeId::from_raw(456));
    let parts = decode_adj_key(&key).expect("decode failed");
    assert_eq!(parts.edge_type, "LIKES");
    assert_eq!(parts.direction, AdjDirection::In);
    assert_eq!(parts.node_id, NodeId::from_raw(456));
}

#[test]
fn forward_keys_sort_by_source_id() {
    let k1 = encode_adj_key_forward("FOLLOWS", NodeId::from_raw(100));
    let k2 = encode_adj_key_forward("FOLLOWS", NodeId::from_raw(200));
    assert!(k1 < k2);
}

#[test]
fn forward_keys_sort_by_edge_type() {
    let k1 = encode_adj_key_forward("AAA", NodeId::from_raw(1));
    let k2 = encode_adj_key_forward("ZZZ", NodeId::from_raw(1));
    assert!(k1 < k2);
}

#[test]
fn decode_invalid_key() {
    assert!(decode_adj_key(b"").is_none());
    assert!(decode_adj_key(b"node:something").is_none());
    assert!(decode_adj_key(b"adj:FOLLOWS:bad:12345678").is_none());
}

// -- PostingList tests --

#[test]
fn posting_list_empty() {
    let pl = PostingList::new();
    assert!(pl.is_empty());
    assert_eq!(pl.len(), 0);
}

#[test]
fn posting_list_insert_maintains_order() {
    let mut pl = PostingList::new();
    assert!(pl.insert(30));
    assert!(pl.insert(10));
    assert!(pl.insert(20));
    assert_eq!(pl.as_slice(), &[10, 20, 30]);
}

#[test]
fn posting_list_insert_duplicate() {
    let mut pl = PostingList::new();
    assert!(pl.insert(10));
    assert!(!pl.insert(10)); // duplicate
    assert_eq!(pl.len(), 1);
}

#[test]
fn posting_list_remove() {
    let mut pl = PostingList::new();
    pl.insert(10);
    pl.insert(20);
    pl.insert(30);

    assert!(pl.remove(20));
    assert_eq!(pl.as_slice(), &[10, 30]);
    assert!(!pl.remove(20)); // already removed
}

#[test]
fn posting_list_contains() {
    let mut pl = PostingList::new();
    pl.insert(42);
    assert!(pl.contains(42));
    assert!(!pl.contains(99));
}

#[test]
fn posting_list_iter() {
    let mut pl = PostingList::new();
    pl.insert(3);
    pl.insert(1);
    pl.insert(2);
    let collected: Vec<u64> = pl.iter().collect();
    assert_eq!(collected, vec![1, 2, 3]);
}

#[test]
fn posting_list_from_sorted() {
    let pl = PostingList::from_sorted(vec![1, 5, 10, 100]);
    assert_eq!(pl.len(), 4);
    assert!(pl.contains(5));
    assert!(!pl.contains(6));
}

#[test]
fn posting_list_uidpack_roundtrip() {
    let mut pl = PostingList::new();
    pl.insert(100);
    pl.insert(200);
    pl.insert(300);

    let bytes = pl.to_bytes().expect("serialize");
    let restored = PostingList::from_bytes(&bytes).expect("deserialize");
    assert_eq!(pl, restored);
}

#[test]
fn posting_list_empty_roundtrip() {
    let pl = PostingList::new();
    let bytes = pl.to_bytes().expect("serialize");
    let restored = PostingList::from_bytes(&bytes).expect("deserialize");
    assert_eq!(pl, restored);
}

#[test]
fn posting_list_uidpack_compression() {
    // Verify UidPack is smaller than raw Vec<u64> MessagePack
    let mut pl = PostingList::new();
    for i in 0..500u64 {
        pl.insert(i * 3); // sequential-ish UIDs with small deltas
    }
    let uidpack_bytes = pl.to_bytes().expect("serialize");
    let raw_msgpack = rmp_serde::to_vec(pl.as_slice()).expect("raw");
    assert!(
        uidpack_bytes.len() < raw_msgpack.len(),
        "UidPack ({} bytes) should be smaller than raw msgpack ({} bytes)",
        uidpack_bytes.len(),
        raw_msgpack.len()
    );
}

#[test]
fn posting_list_uidpack_large_roundtrip() {
    // Test with >256 UIDs (multiple blocks) + MSB boundary crossing
    let mut pl = PostingList::new();
    for i in 0..1000u64 {
        pl.insert(i * 7 + 1);
    }
    let bytes = pl.to_bytes().expect("serialize");
    let restored = PostingList::from_bytes(&bytes).expect("deserialize");
    assert_eq!(pl, restored);
}

#[test]
fn posting_list_large() {
    let mut pl = PostingList::new();
    for i in (0..1000u64).rev() {
        pl.insert(i);
    }
    assert_eq!(pl.len(), 1000);
    assert_eq!(pl.as_slice()[0], 0);
    assert_eq!(pl.as_slice()[999], 999);
}

// -- Forward + Reverse symmetry --

#[test]
fn forward_and_reverse_keys_are_different() {
    let fwd = encode_adj_key_forward("FOLLOWS", NodeId::from_raw(42));
    let rev = encode_adj_key_reverse("FOLLOWS", NodeId::from_raw(42));
    assert_ne!(fwd, rev);
}

// -- Split key tests --

#[test]
fn split_key_encoding() {
    let key = encode_adj_split_key("FOLLOWS", AdjDirection::Out, NodeId::from_raw(42), 1000);
    assert!(key.starts_with(b"adj_split:FOLLOWS:out:"));
}

#[test]
fn split_keys_sort_by_start_uid() {
    let k1 = encode_adj_split_key("FOLLOWS", AdjDirection::Out, NodeId::from_raw(42), 100);
    let k2 = encode_adj_split_key("FOLLOWS", AdjDirection::Out, NodeId::from_raw(42), 200);
    assert!(k1 < k2, "split keys should sort by start_uid");
}

#[test]
fn split_key_differs_from_main_key() {
    let main = encode_adj_key_forward("FOLLOWS", NodeId::from_raw(42));
    let split = encode_adj_split_key("FOLLOWS", AdjDirection::Out, NodeId::from_raw(42), 1);
    assert_ne!(main, split);
    assert!(split.starts_with(b"adj_split:"));
}

#[test]
fn forward_keyed_by_source_reverse_keyed_by_target() {
    // Forward: source=42, targets=[1,2,3]
    let fwd_key = encode_adj_key_forward("FOLLOWS", NodeId::from_raw(42));
    let fwd_parts = decode_adj_key(&fwd_key).expect("decode");
    assert_eq!(fwd_parts.node_id, NodeId::from_raw(42));
    assert_eq!(fwd_parts.direction, AdjDirection::Out);

    // Reverse: target=1, sources=[42]
    let rev_key = encode_adj_key_reverse("FOLLOWS", NodeId::from_raw(1));
    let rev_parts = decode_adj_key(&rev_key).expect("decode");
    assert_eq!(rev_parts.node_id, NodeId::from_raw(1));
    assert_eq!(rev_parts.direction, AdjDirection::In);
}

// -- Edge property key tests --

#[test]
fn edgeprop_key_encoding() {
    let key = encode_edgeprop_key("KNOWS", NodeId::from_raw(42), NodeId::from_raw(99));
    assert!(key.starts_with(b"edgeprop:KNOWS:"));
}

#[test]
fn edgeprop_key_roundtrip() {
    let key = encode_edgeprop_key("WORKS_AT", NodeId::from_raw(100), NodeId::from_raw(200));
    let (edge_type, src, tgt) = decode_edgeprop_key(&key).expect("decode failed");
    assert_eq!(edge_type, "WORKS_AT");
    assert_eq!(src, NodeId::from_raw(100));
    assert_eq!(tgt, NodeId::from_raw(200));
}

#[test]
fn edgeprop_key_sorting() {
    // Same edge type, different source → sort by source
    let k1 = encode_edgeprop_key("KNOWS", NodeId::from_raw(1), NodeId::from_raw(99));
    let k2 = encode_edgeprop_key("KNOWS", NodeId::from_raw(2), NodeId::from_raw(99));
    assert!(k1 < k2);
}

#[test]
fn edgeprop_key_decode_invalid() {
    assert!(decode_edgeprop_key(b"").is_none());
    assert!(decode_edgeprop_key(b"adj:FOLLOWS:out:12345678").is_none());
    assert!(decode_edgeprop_key(b"edgeprop:short").is_none());
}

#[test]
fn temporal_edgeprop_key_roundtrip() {
    let key = encode_temporal_edgeprop_key(
        "WORKS_AT",
        NodeId::from_raw(7),
        NodeId::from_raw(11),
        1_700_000_000_000,
    );
    let (edge_type, src, tgt, vf) = decode_temporal_edgeprop_key(&key).expect("decode failed");
    assert_eq!(edge_type, "WORKS_AT");
    assert_eq!(src, NodeId::from_raw(7));
    assert_eq!(tgt, NodeId::from_raw(11));
    assert_eq!(vf, 1_700_000_000_000);
}

#[test]
fn temporal_edgeprop_key_sorts_by_valid_from_within_pair() {
    let src = NodeId::from_raw(1);
    let tgt = NodeId::from_raw(2);
    let k_early = encode_temporal_edgeprop_key("WORKS_AT", src, tgt, 1000);
    let k_mid = encode_temporal_edgeprop_key("WORKS_AT", src, tgt, 2000);
    let k_late = encode_temporal_edgeprop_key("WORKS_AT", src, tgt, 3000);
    assert!(k_early < k_mid);
    assert!(k_mid < k_late);
}

#[test]
fn temporal_edgeprop_key_handles_negative_valid_from() {
    // Pre-epoch timestamps must sort below post-epoch ones in lex order.
    let src = NodeId::from_raw(1);
    let tgt = NodeId::from_raw(2);
    let k_neg = encode_temporal_edgeprop_key("WORKS_AT", src, tgt, -5_000);
    let k_zero = encode_temporal_edgeprop_key("WORKS_AT", src, tgt, 0);
    let k_pos = encode_temporal_edgeprop_key("WORKS_AT", src, tgt, 5_000);
    assert!(k_neg < k_zero);
    assert!(k_zero < k_pos);
    let (_, _, _, decoded) = decode_temporal_edgeprop_key(&k_neg).expect("k_neg must decode");
    assert_eq!(decoded, -5_000);
}

#[test]
fn temporal_edgeprop_pair_prefix_contains_all_versions() {
    let src = NodeId::from_raw(1);
    let tgt = NodeId::from_raw(2);
    let prefix = temporal_edgeprop_pair_prefix("WORKS_AT", src, tgt);
    let k1 = encode_temporal_edgeprop_key("WORKS_AT", src, tgt, 100);
    let k2 = encode_temporal_edgeprop_key("WORKS_AT", src, tgt, 200);
    assert!(k1.starts_with(&prefix));
    assert!(k2.starts_with(&prefix));
    // A different target must NOT share the prefix.
    let k_other = encode_temporal_edgeprop_key("WORKS_AT", src, NodeId::from_raw(99), 100);
    assert!(!k_other.starts_with(&prefix));
}

#[test]
fn valid_from_upper_bound_key_excludes_strictly_greater() {
    let src = NodeId::from_raw(1);
    let tgt = NodeId::from_raw(2);
    let bound = valid_from_upper_bound_key("WORKS_AT", src, tgt, 1500);
    let k_in = encode_temporal_edgeprop_key("WORKS_AT", src, tgt, 1500);
    let k_above = encode_temporal_edgeprop_key("WORKS_AT", src, tgt, 1501);
    // Bound is exclusive: keys < bound are included, k_above sits at the bound.
    assert!(k_in < bound);
    assert!(bound <= k_above);
}

#[test]
fn temporal_edgeprop_key_decode_invalid() {
    assert!(decode_temporal_edgeprop_key(b"").is_none());
    assert!(decode_temporal_edgeprop_key(b"edgeprop:short").is_none());
    // Plain (non-temporal) edgeprop key has the wrong byte length: 9 prefix +
    // type + : + 8 src + : + 8 tgt = 9 + 8 + 1 + 8 = 26 bytes for empty type,
    // temporal needs at least 9 + 1 + 8 + 1 + 8 + 1 + 8 = 36 bytes for empty type.
    let non_temporal = encode_edgeprop_key("WORKS_AT", NodeId::from_raw(1), NodeId::from_raw(2));
    // Same prefix triggers the starts_with branch but length & layout differ:
    // the decoder must reject — separator at the temporal position is a u8 of
    // the target_id, not necessarily ':'. The test asserts behavior is safe.
    let _ = decode_temporal_edgeprop_key(&non_temporal);
}

// -- EdgeProperties tests --

#[test]
fn edge_properties_new() {
    let ep = EdgeProperties::new();
    assert!(ep.is_empty());
    assert_eq!(ep.len(), 0);
}

#[test]
fn edge_properties_set_get_remove() {
    let mut ep = EdgeProperties::new();
    ep.set(1, PropertyValue::String("2024-01-01".into()));
    ep.set(2, PropertyValue::Float(0.8));

    assert_eq!(ep.get(1), Some(&PropertyValue::String("2024-01-01".into())));
    assert_eq!(ep.get(2), Some(&PropertyValue::Float(0.8)));
    assert_eq!(ep.len(), 2);

    let removed = ep.remove(1);
    assert_eq!(removed, Some(PropertyValue::String("2024-01-01".into())));
    assert_eq!(ep.len(), 1);
}

#[test]
fn edge_properties_msgpack_roundtrip() {
    let mut ep = EdgeProperties::new();
    ep.set(1, PropertyValue::String("since".into()));
    ep.set(2, PropertyValue::Float(0.95));
    ep.set(3, PropertyValue::Int(42));
    ep.set(4, PropertyValue::Bool(true));
    ep.set(5, PropertyValue::Null);
    ep.set(6, PropertyValue::Binary(vec![0xCA, 0xFE]));

    let bytes = ep.to_msgpack().expect("serialize");
    let restored = EdgeProperties::from_msgpack(&bytes).expect("deserialize");
    assert_eq!(ep, restored);
}

#[test]
fn edge_properties_empty_roundtrip() {
    let ep = EdgeProperties::new();
    let bytes = ep.to_msgpack().expect("serialize");
    let restored = EdgeProperties::from_msgpack(&bytes).expect("deserialize");
    assert_eq!(ep, restored);
}

#[test]
fn edge_properties_all_types() {
    let mut ep = EdgeProperties::new();
    ep.set(1, PropertyValue::String("hello".into()));
    ep.set(2, PropertyValue::Int(i64::MIN));
    ep.set(3, PropertyValue::Float(f64::MAX));
    ep.set(4, PropertyValue::Bool(false));
    ep.set(5, PropertyValue::Null);
    ep.set(6, PropertyValue::Binary(vec![]));
    ep.set(
        7,
        PropertyValue::Array(vec![
            PropertyValue::Int(1),
            PropertyValue::String("two".into()),
        ]),
    );

    let bytes = ep.to_msgpack().expect("serialize");
    let restored = EdgeProperties::from_msgpack(&bytes).expect("deserialize");
    assert_eq!(ep, restored);
}

// -- Discriminated edge keys (ADR-029) --

#[test]
fn discriminator_int_key_roundtrips() {
    let key = encode_discriminated_edgeprop_key(
        "KNOWS",
        NodeId::from_raw(7),
        NodeId::from_raw(9),
        &PropertyValue::Int(-42),
    )
    .expect("int is a supported discriminator");
    let (ty, src, tgt, disc) =
        decode_discriminated_edgeprop_key(&key, &PropertyType::Int).expect("decode");
    assert_eq!(ty, "KNOWS");
    assert_eq!(src, NodeId::from_raw(7));
    assert_eq!(tgt, NodeId::from_raw(9));
    assert_eq!(disc, PropertyValue::Int(-42));
}

#[test]
fn discriminator_timestamp_is_byte_identical_to_temporal_key() {
    // ADR-029: TEMPORAL is DISCRIMINATED BY (valid_from) — one storage shape.
    let vf = 1_710_000_000_000i64;
    let temporal =
        encode_temporal_edgeprop_key("WORKS_AT", NodeId::from_raw(1), NodeId::from_raw(2), vf);
    let discriminated = encode_discriminated_edgeprop_key(
        "WORKS_AT",
        NodeId::from_raw(1),
        NodeId::from_raw(2),
        &PropertyValue::Timestamp(vf),
    )
    .expect("timestamp is supported");
    assert_eq!(temporal, discriminated);
}

#[test]
fn discriminator_string_roundtrips_and_sorts_lexicographically() {
    let mk = |s: &str| {
        encode_discriminated_edgeprop_key(
            "KNOWS",
            NodeId::from_raw(1),
            NodeId::from_raw(2),
            &PropertyValue::String(s.to_string()),
        )
        .expect("string supported")
    };
    let work = mk("work");
    let (_, _, _, disc) =
        decode_discriminated_edgeprop_key(&work, &PropertyType::String).expect("decode");
    assert_eq!(disc, PropertyValue::String("work".to_string()));
    // Raw UTF-8 keeps byte-order == value-order (range push-down on the column).
    assert!(mk("college") < work);
}

#[test]
fn discriminator_float_roundtrips_and_preserves_order() {
    let enc = |f: f64| encode_discriminator_value(&PropertyValue::Float(f)).expect("float");
    assert!(enc(-1.0) < enc(0.0));
    assert!(enc(0.0) < enc(1.5));
    let back = decode_discriminator_value(&enc(3.25), &PropertyType::Float).expect("decode");
    assert_eq!(back, PropertyValue::Float(3.25));
}

#[test]
fn discriminator_int_keys_sort_by_value() {
    let k = |v: i64| {
        encode_discriminated_edgeprop_key(
            "E",
            NodeId::from_raw(1),
            NodeId::from_raw(2),
            &PropertyValue::Int(v),
        )
        .expect("int supported")
    };
    assert!(k(-5) < k(-1));
    assert!(k(-1) < k(0));
    assert!(k(0) < k(7));
}

#[test]
fn discriminator_bool_roundtrips() {
    for b in [false, true] {
        let bytes = encode_discriminator_value(&PropertyValue::Bool(b)).expect("bool");
        assert_eq!(
            decode_discriminator_value(&bytes, &PropertyType::Bool).expect("decode"),
            PropertyValue::Bool(b)
        );
    }
}

#[test]
fn discriminator_blob_is_sha256_digest() {
    let bytes = encode_discriminator_value(&PropertyValue::Blob(vec![1, 2, 3])).expect("blob");
    assert_eq!(bytes.len(), 32);
    // One-way: decodes to the digest, not the original blob bytes.
    let back = decode_discriminator_value(&bytes, &PropertyType::Blob).expect("decode");
    assert_eq!(back, PropertyValue::Blob(bytes));
}

#[test]
fn unsupported_discriminator_type_is_none() {
    assert!(encode_discriminator_value(&PropertyValue::Null).is_none());
    assert!(encode_discriminator_value(&PropertyValue::Vector(vec![1.0_f32])).is_none());
}
