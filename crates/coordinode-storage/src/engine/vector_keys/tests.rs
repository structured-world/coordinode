use super::*;

#[test]
fn f32_key_round_trip() {
    let key = encode_vec_f32_key(7, 13, 0xDEAD_BEEF_CAFE_F00D);
    assert_eq!(key.len(), VECTOR_KEY_LEN);
    let (label, prop, node) = decode_vector_key(&key).expect("decode");
    assert_eq!(label, 7);
    assert_eq!(prop, 13);
    assert_eq!(node, 0xDEAD_BEEF_CAFE_F00D);
}

#[test]
fn keys_sort_lexically_by_node_id() {
    // Big-endian node-id encoding must keep prefix scans contiguous AND
    // monotonic in node-id order — important for multi_get batches that
    // expect ranged keys to hit a single SST block.
    let k1 = encode_vec_f32_key(1, 1, 1);
    let k2 = encode_vec_f32_key(1, 1, 1_000_000);
    let k3 = encode_vec_f32_key(1, 1, u64::MAX);
    assert!(k1 < k2);
    assert!(k2 < k3);
}

#[test]
fn prefix_is_proper_prefix_of_full_key() {
    let prefix = vec_f32_index_prefix(42, 17);
    let key = encode_vec_f32_key(42, 17, 99);
    assert_eq!(&key[..prefix.len()], &prefix[..]);
}

#[test]
fn decode_rejects_wrong_length() {
    assert!(decode_vector_key(&[0u8; 0]).is_none());
    assert!(decode_vector_key(&[0u8; VECTOR_KEY_LEN - 1]).is_none());
    assert!(decode_vector_key(&[0u8; VECTOR_KEY_LEN + 1]).is_none());
}

#[test]
fn decode_rejects_unknown_prefix() {
    let mut key = encode_vec_f32_key(1, 2, 3);
    key[0..4].copy_from_slice(b"nope");
    assert!(decode_vector_key(&key).is_none());
}

#[test]
fn f32_value_round_trip() {
    let v = vec![0.0_f32, 1.5, -2.25, f32::INFINITY, f32::NEG_INFINITY];
    let bytes = encode_f32_value(&v);
    assert_eq!(bytes.len(), v.len() * 4);
    let back = decode_f32_value(&bytes).expect("decode");
    assert_eq!(back, v);
}

#[test]
fn f32_value_handles_nan_byte_exact() {
    // NaN comparisons aren't equal, but the byte representation must
    // round-trip exactly. Used by recall regression tests that re-encode
    // truth-tier vectors after a codec migration.
    let v = vec![f32::NAN, 0.0];
    let bytes = encode_f32_value(&v);
    let back = decode_f32_value(&bytes).expect("decode");
    assert!(back[0].is_nan());
    assert_eq!(back[1], 0.0);
    // Byte-exact: re-encode of decoded value should match original.
    assert_eq!(encode_f32_value(&back), bytes);
}

#[test]
fn decode_rejects_non_multiple_of_four() {
    assert!(decode_f32_value(&[0u8; 3]).is_none());
    assert!(decode_f32_value(&[0u8; 5]).is_none());
}
