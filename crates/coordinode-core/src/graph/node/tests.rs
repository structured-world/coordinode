use super::*;

// -- NodeId tests --

#[test]
fn node_id_roundtrip() {
    let id = NodeId::from_raw(42);
    assert_eq!(id.as_raw(), 42);
}

#[test]
fn node_id_ordering() {
    let a = NodeId::from_raw(1);
    let b = NodeId::from_raw(2);
    assert!(a < b);
}

#[test]
fn node_id_display() {
    let id = NodeId::from_raw(99);
    assert_eq!(format!("{id}"), "node:99");
}

// -- NodeIdAllocator tests --

#[test]
fn allocator_starts_from_one() {
    let alloc = NodeIdAllocator::new(0);
    assert_eq!(alloc.next().as_raw(), 1);
    assert_eq!(alloc.next().as_raw(), 2);
}

#[test]
fn allocator_resume() {
    let alloc = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    assert_eq!(alloc.next().as_raw(), 101);
}

#[test]
fn allocator_current_does_not_advance() {
    let alloc = NodeIdAllocator::resume_from(NodeId::from_raw(50));
    assert_eq!(alloc.current().as_raw(), 50);
    assert_eq!(alloc.current().as_raw(), 50);
}

#[test]
fn allocator_advance_to() {
    let alloc = NodeIdAllocator::resume_from(NodeId::from_raw(10));
    alloc.advance_to(NodeId::from_raw(100));
    assert_eq!(alloc.next().as_raw(), 101);
}

#[test]
fn allocator_concurrent() {
    use std::collections::BTreeSet;
    use std::sync::Arc;

    let alloc = Arc::new(NodeIdAllocator::new(0));
    let mut handles = Vec::new();

    for _ in 0..8 {
        let alloc = Arc::clone(&alloc);
        handles.push(std::thread::spawn(move || {
            (0..500).map(|_| alloc.next().as_raw()).collect::<Vec<_>>()
        }));
    }

    let mut all: BTreeSet<u64> = BTreeSet::new();
    for h in handles {
        for id in h.join().expect("thread panicked") {
            assert!(all.insert(id), "duplicate ID: {id}");
        }
    }
    assert_eq!(all.len(), 4000);
}

// -- u20/u44 layout tests --

#[test]
fn node_id_compose_extracts_hint_and_sequence() {
    let id = NodeId::compose(42, 1_000_000);
    assert_eq!(id.origin_shard_hint(), 42);
    assert_eq!(id.sequence(), 1_000_000);
}

#[test]
fn node_id_compose_handles_max_hint() {
    let id = NodeId::compose(NODE_ID_MAX_HINT, 1);
    assert_eq!(id.origin_shard_hint(), NODE_ID_MAX_HINT);
    assert_eq!(id.sequence(), 1);
}

#[test]
fn node_id_compose_handles_max_sequence() {
    let id = NodeId::compose(0, NODE_ID_MAX_SEQUENCE);
    assert_eq!(id.origin_shard_hint(), 0);
    assert_eq!(id.sequence(), NODE_ID_MAX_SEQUENCE);
}

#[test]
#[should_panic(expected = "shard_hint")]
fn node_id_compose_panics_on_oversized_hint() {
    NodeId::compose(NODE_ID_MAX_HINT + 1, 0);
}

#[test]
#[should_panic(expected = "sequence")]
fn node_id_compose_panics_on_oversized_sequence() {
    NodeId::compose(0, NODE_ID_MAX_SEQUENCE + 1);
}

#[test]
fn allocator_ce_default_uses_hint_zero() {
    let alloc = NodeIdAllocator::new(0);
    assert_eq!(alloc.shard_hint(), 0);
    let id = alloc.next();
    assert_eq!(id.origin_shard_hint(), 0);
    assert_eq!(id.sequence(), 1);
}

#[test]
fn allocator_ee_hint_carried_in_emitted_ids() {
    let alloc = NodeIdAllocator::new(42);
    let id1 = alloc.next();
    let id2 = alloc.next();
    assert_eq!(id1.origin_shard_hint(), 42);
    assert_eq!(id2.origin_shard_hint(), 42);
    assert_eq!(id1.sequence(), 1);
    assert_eq!(id2.sequence(), 2);
}

#[test]
fn allocator_resume_from_preserves_hint() {
    // EE shard 7 persisted a high-water mark at sequence 1000.
    let persisted = NodeId::compose(7, 1000);
    let alloc = NodeIdAllocator::resume_from(persisted);
    assert_eq!(alloc.shard_hint(), 7);
    let next = alloc.next();
    assert_eq!(next.origin_shard_hint(), 7);
    assert_eq!(next.sequence(), 1001);
}

#[test]
#[should_panic(expected = "advance_to received NodeId from shard_hint")]
fn allocator_advance_to_rejects_cross_hint_id() {
    let alloc = NodeIdAllocator::new(3);
    // Trying to advance against an id from a different shard is a logic
    // bug — sequence space is per-shard.
    alloc.advance_to(NodeId::compose(5, 100));
}

#[test]
#[should_panic(expected = "sequence wrap")]
fn allocator_panics_on_sequence_wrap() {
    // Resume from the last possible sequence value in the 44-bit window.
    // The next `next()` call should detect wrap and panic — without this
    // guard sequence bits would leak into the 20-bit hint window and
    // corrupt routing (phantom shards).
    let alloc = NodeIdAllocator::resume_from(NodeId::compose(0, NODE_ID_MAX_SEQUENCE));
    let _ = alloc.next();
}

#[test]
fn element_id_rejects_too_long_string() {
    // 14 chars — must be rejected (canonical is exactly 13).
    assert!(NodeId::from_element_id("0000000000000X").is_none());
}

#[test]
fn element_id_rejects_one_char_short() {
    // 12 chars — must be rejected.
    assert!(NodeId::from_element_id("000000000000").is_none());
}

#[test]
fn element_id_roundtrips_extreme_combined() {
    // Both fields at their respective maxima — exercises the bit-packing
    // boundary between sequence and hint windows simultaneously.
    let id = NodeId::compose(NODE_ID_MAX_HINT, NODE_ID_MAX_SEQUENCE);
    let s = id.to_element_id();
    assert_eq!(s.len(), 13);
    assert_eq!(NodeId::from_element_id(&s), Some(id));
    assert_eq!(id.origin_shard_hint(), NODE_ID_MAX_HINT);
    assert_eq!(id.sequence(), NODE_ID_MAX_SEQUENCE);
}

// -- elementId base32 encoding tests --

#[test]
fn element_id_roundtrips_zero() {
    let id = NodeId::from_raw(0);
    let s = id.to_element_id();
    assert_eq!(s.len(), 13);
    assert_eq!(NodeId::from_element_id(&s), Some(id));
}

#[test]
fn element_id_roundtrips_max() {
    let id = NodeId::from_raw(u64::MAX);
    let s = id.to_element_id();
    assert_eq!(s.len(), 13);
    assert_eq!(NodeId::from_element_id(&s), Some(id));
}

#[test]
fn element_id_roundtrips_composed_id() {
    let id = NodeId::compose(NODE_ID_MAX_HINT, NODE_ID_MAX_SEQUENCE);
    let s = id.to_element_id();
    let decoded = NodeId::from_element_id(&s).expect("decode");
    assert_eq!(decoded.origin_shard_hint(), NODE_ID_MAX_HINT);
    assert_eq!(decoded.sequence(), NODE_ID_MAX_SEQUENCE);
}

#[test]
fn element_id_within_shard_is_time_sortable() {
    // Within one shard, growing sequence ⇒ growing elementId string.
    let early = NodeId::compose(5, 1).to_element_id();
    let later = NodeId::compose(5, 1_000_000).to_element_id();
    assert!(early < later, "expected {early} < {later}");
}

#[test]
fn element_id_rejects_invalid_input() {
    assert!(NodeId::from_element_id("").is_none());
    assert!(NodeId::from_element_id("TOO_SHORT").is_none());
    assert!(NodeId::from_element_id("INVALID!CHARS").is_none()); // 13 chars but '!' invalid
}

#[test]
fn element_id_normalises_crockford_aliases() {
    // I/L → 1 and O → 0 per Crockford spec.
    let canonical = NodeId::compose(0, 1).to_element_id();
    let with_alias_l = canonical.replace('1', "L");
    let with_alias_i = canonical.replace('1', "I");
    let with_alias_o = canonical.replace('0', "O");
    assert_eq!(
        NodeId::from_element_id(&with_alias_l),
        Some(NodeId::compose(0, 1))
    );
    assert_eq!(
        NodeId::from_element_id(&with_alias_i),
        Some(NodeId::compose(0, 1))
    );
    assert_eq!(
        NodeId::from_element_id(&with_alias_o),
        Some(NodeId::compose(0, 1))
    );
}

// -- Key encoding tests --

#[test]
fn encode_decode_key_roundtrip() {
    let shard = 42u16;
    let id = NodeId::from_raw(12345);
    let key = encode_node_key(shard, id);
    let (dec_shard, dec_id) = decode_node_key(&key).expect("decode failed");
    assert_eq!(dec_shard, shard);
    assert_eq!(dec_id, id);
}

#[test]
fn key_ordering_within_shard() {
    let k1 = encode_node_key(1, NodeId::from_raw(100));
    let k2 = encode_node_key(1, NodeId::from_raw(200));
    assert!(k1 < k2, "keys should sort by node_id within shard");
}

#[test]
fn key_ordering_across_shards() {
    let k1 = encode_node_key(1, NodeId::from_raw(999));
    let k2 = encode_node_key(2, NodeId::from_raw(1));
    assert!(k1 < k2, "shard 1 keys should sort before shard 2");
}

#[test]
fn decode_invalid_key() {
    assert!(decode_node_key(b"").is_none());
    assert!(decode_node_key(b"short").is_none());
    assert!(decode_node_key(b"wrong:prefix12345678").is_none());
}

#[test]
fn key_starts_with_prefix() {
    let key = encode_node_key(0, NodeId::from_raw(1));
    assert!(key.starts_with(b"node:"));
}

// -- NodeRecord tests --

#[test]
fn record_new() {
    let rec = NodeRecord::new("User");
    assert_eq!(rec.primary_label(), "User");
    assert_eq!(rec.labels, vec!["User"]);
    assert!(rec.props.is_empty());
}

#[test]
fn record_with_labels() {
    let rec = NodeRecord::with_labels(vec!["User".into(), "Admin".into()]);
    assert_eq!(rec.primary_label(), "User");
    assert!(rec.has_label("User"));
    assert!(rec.has_label("Admin"));
    assert!(!rec.has_label("Guest"));
}

#[test]
fn record_add_remove_label() {
    let mut rec = NodeRecord::new("User");
    rec.add_label("Admin".into());
    assert!(rec.has_label("Admin"));
    assert_eq!(rec.labels.len(), 2);

    // Duplicate add is no-op
    rec.add_label("Admin".into());
    assert_eq!(rec.labels.len(), 2);

    // Remove label
    assert!(rec.remove_label("Admin"));
    assert!(!rec.has_label("Admin"));
    assert_eq!(rec.labels.len(), 1);

    // Remove non-existent label
    assert!(!rec.remove_label("Guest"));
}

#[test]
fn record_empty_label() {
    let rec = NodeRecord::new("");
    assert!(rec.labels.is_empty());
    assert_eq!(rec.primary_label(), "");
}

#[test]
fn record_set_get_remove() {
    let mut rec = NodeRecord::new("User");
    rec.set(1, PropertyValue::String("Alice".into()));
    rec.set(2, PropertyValue::Int(30));

    assert_eq!(rec.get(1), Some(&PropertyValue::String("Alice".into())));
    assert_eq!(rec.get(2), Some(&PropertyValue::Int(30)));
    assert_eq!(rec.get(99), None);

    let removed = rec.remove(1);
    assert_eq!(removed, Some(PropertyValue::String("Alice".into())));
    assert_eq!(rec.get(1), None);
}

#[test]
fn record_msgpack_roundtrip() {
    let mut rec = NodeRecord::new("Movie");
    rec.set(1, PropertyValue::String("Inception".into()));
    rec.set(2, PropertyValue::Int(2010));
    rec.set(3, PropertyValue::Float(8.8));
    rec.set(4, PropertyValue::Bool(true));
    rec.set(5, PropertyValue::Null);
    rec.set(6, PropertyValue::Binary(vec![0xDE, 0xAD]));
    rec.set(
        7,
        PropertyValue::Array(vec![
            PropertyValue::Int(1),
            PropertyValue::String("two".into()),
        ]),
    );

    let bytes = rec.to_msgpack().expect("serialize failed");
    let restored = NodeRecord::from_msgpack(&bytes).expect("deserialize failed");
    assert_eq!(rec, restored);
}

#[test]
fn record_empty_props_roundtrip() {
    let rec = NodeRecord::new("Empty");
    let bytes = rec.to_msgpack().expect("serialize");
    let restored = NodeRecord::from_msgpack(&bytes).expect("deserialize");
    assert_eq!(rec, restored);
}

#[test]
fn record_msgpack_is_compact() {
    let mut rec = NodeRecord::new("User");
    // With interned IDs (u32 keys), the MessagePack output should be
    // significantly smaller than using string field names
    rec.set(1, PropertyValue::String("Alice".into()));
    rec.set(2, PropertyValue::Int(30));

    let bytes = rec.to_msgpack().expect("serialize");
    // MessagePack with integer keys should be quite compact
    assert!(
        bytes.len() < 50,
        "encoded size {} should be < 50",
        bytes.len()
    );
}

// -- G028: extra overflow map --

#[test]
fn extra_set_and_get() {
    let mut rec = NodeRecord::new("User");
    rec.set_extra("ad_hoc", PropertyValue::String("value".into()));
    assert_eq!(
        rec.get_extra("ad_hoc"),
        Some(&PropertyValue::String("value".into()))
    );
    assert!(rec.get_extra("missing").is_none());
}

#[test]
fn extra_none_by_default() {
    let rec = NodeRecord::new("User");
    assert!(rec.extra.is_none());
    assert!(rec.get_extra("anything").is_none());
}

#[test]
fn extra_roundtrip_msgpack() {
    let mut rec = NodeRecord::new("Config");
    rec.set(1, PropertyValue::String("declared".into()));
    rec.set_extra("dynamic_key", PropertyValue::Int(42));
    rec.set_extra("another", PropertyValue::Bool(true));

    let bytes = rec.to_msgpack().expect("serialize");
    let decoded = NodeRecord::from_msgpack(&bytes).expect("deserialize");

    assert_eq!(
        decoded.props.get(&1),
        Some(&PropertyValue::String("declared".into()))
    );
    assert_eq!(
        decoded.get_extra("dynamic_key"),
        Some(&PropertyValue::Int(42))
    );
    assert_eq!(
        decoded.get_extra("another"),
        Some(&PropertyValue::Bool(true))
    );
}

#[test]
fn extra_none_skipped_in_serialization() {
    // NodeRecord without extra should be backward compatible
    let mut rec = NodeRecord::new("User");
    rec.set(1, PropertyValue::String("Alice".into()));
    let bytes = rec.to_msgpack().expect("serialize");

    // Should deserialize even without extra field (serde default)
    let decoded = NodeRecord::from_msgpack(&bytes).expect("deserialize");
    assert!(decoded.extra.is_none());
}

#[test]
fn backward_compat_roundtrip_without_extra() {
    // NodeRecord without extra should roundtrip cleanly.
    // This verifies backward compat with old format (no extra field).
    let rec = NodeRecord::new("Test");
    let bytes = rec.to_msgpack().expect("serialize");
    let decoded = NodeRecord::from_msgpack(&bytes).expect("deserialize");
    assert_eq!(rec, decoded);
}

// ─── R172b: temporal node key encoding (ADR-027) ──────────────────────

#[test]
fn temporal_node_key_roundtrip() {
    let nid = NodeId(0x0123_4567_89ab_cdef);
    let key = encode_temporal_node_key(42, nid, 1577836800000);
    // 5 + 2 + 1 + 8 + 1 + 8 = 25
    assert_eq!(key.len(), 25);
    let (shard, decoded_nid, vf) =
        decode_temporal_node_key(&key).expect("temporal key must decode");
    assert_eq!(shard, 42);
    assert_eq!(decoded_nid, nid);
    assert_eq!(vf, 1577836800000);
}

#[test]
fn temporal_node_key_versions_sort_chronologically() {
    // Two versions of the same node, different valid_from. Lexicographic
    // byte order over keys must match numeric order over valid_from.
    let nid = NodeId(100);
    let k_early = encode_temporal_node_key(0, nid, 1000);
    let k_late = encode_temporal_node_key(0, nid, 2000);
    assert!(
        k_early < k_late,
        "earlier valid_from must sort before later"
    );

    // Negative valid_from (pre-epoch) must sort before positive.
    let k_neg = encode_temporal_node_key(0, nid, -1);
    let k_zero = encode_temporal_node_key(0, nid, 0);
    assert!(k_neg < k_zero, "negative valid_from must sort first");
    assert!(k_zero < k_early);
}

#[test]
fn temporal_node_id_prefix_is_proper_prefix() {
    // The prefix must be a strict prefix of every per-version key for
    // the same (shard, node_id), and not match any other node_id's keys.
    let prefix = temporal_node_id_prefix(7, NodeId(42));
    // 5 + 2 + 1 + 8 + 1 = 17
    assert_eq!(prefix.len(), 17);

    let k_v1 = encode_temporal_node_key(7, NodeId(42), 100);
    let k_v2 = encode_temporal_node_key(7, NodeId(42), 200);
    let k_other = encode_temporal_node_key(7, NodeId(43), 100);

    assert!(k_v1.starts_with(&prefix));
    assert!(k_v2.starts_with(&prefix));
    assert!(!k_other.starts_with(&prefix));
}

#[test]
fn temporal_node_key_disjoint_from_non_temporal() {
    // The 16-byte non-temporal key and the 25-byte temporal key must
    // never alias — `decode_node_key` rejects the temporal form and
    // `decode_temporal_node_key` rejects the non-temporal form.
    let nid = NodeId(99);
    let non_temp = encode_node_key(5, nid);
    let temp = encode_temporal_node_key(5, nid, 100);

    assert_eq!(non_temp.len(), 16);
    assert_eq!(temp.len(), 25);
    assert!(decode_node_key(&non_temp).is_some());
    assert!(
        decode_node_key(&temp).is_none(),
        "decode_node_key must reject temporal key"
    );
    assert!(decode_temporal_node_key(&temp).is_some());
    assert!(
        decode_temporal_node_key(&non_temp).is_none(),
        "decode_temporal_node_key must reject non-temporal key"
    );
}

#[test]
fn node_write_key_picks_correct_form() {
    let nid = NodeId(42);
    let non_temp = node_write_key(3, nid, None);
    let temp = node_write_key(3, nid, Some(1234));
    assert_eq!(non_temp, encode_node_key(3, nid));
    assert_eq!(temp, encode_temporal_node_key(3, nid, 1234));
}
