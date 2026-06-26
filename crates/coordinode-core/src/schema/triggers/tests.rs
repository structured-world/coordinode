use super::*;
use crate::graph::types::Value;
use std::collections::BTreeMap;

#[test]
fn trigger_key_encoding_is_stable() {
    let k = encode_trigger_key("audit_log");
    assert_eq!(k, b"schema:trigger:audit_log");
}

#[test]
fn trigger_index_key_label_vs_edge_disjoint() {
    // Same identifier "Order" as a label vs. an edge type must produce
    // different index keys so the namespaces don't collide.
    let node = TriggerTargetSchema::label("Order").index_key_segment();
    let edge = TriggerTargetSchema::edge_type("Order").index_key_segment();
    let nk = encode_trigger_index_key(&node, "c");
    let ek = encode_trigger_index_key(&edge, "c");
    assert_ne!(nk, ek);
    assert_eq!(nk, b"schema:trigger_index:n:Order:c");
    assert_eq!(ek, b"schema:trigger_index:e:Order:c");
}

#[test]
fn trigger_scan_prefix_matches_trigger_keys_only() {
    // A trigger-index key starts with `schema:trigger_index:` which DOES
    // share the `schema:trigger` prefix — verify the scan prefix is
    // discriminating via the trailing colon.
    let trig = encode_trigger_key("t");
    let idx = encode_trigger_index_key("n:User", "c");
    assert!(trig.starts_with(trigger_scan_prefix()));
    // Index keys MUST share the literal `schema:trigger_index:` rather
    // than `schema:trigger:` to be distinguishable from definitions.
    assert!(!idx.starts_with(trigger_scan_prefix()));
}

#[test]
fn events_enabled_segments_in_order() {
    let mut e = TriggerEventsSchema::default();
    e.on_update = true;
    e.on_delete = true;
    // Order is fixed: c, u, d.
    assert_eq!(e.enabled_segments(), vec!["u", "d"]);
}

#[test]
fn trigger_schema_serde_roundtrip() {
    let s = TriggerSchema {
        name: "audit".into(),
        target: TriggerTargetSchema::label("User"),
        events: TriggerEventsSchema {
            on_create: true,
            on_update: false,
            on_delete: true,
        },
        timing: TriggerTimingSchema::AfterCommit,
        body_source: "CREATE (e:Log)".into(),
        cascade_limit: Some(7),
        cascade_fanout: None,
        on_error: Some(OnErrorPolicySchema::Retry {
            n: 5,
            backoff_ms: 250,
        }),
        enabled: true,
        created_at_hlc_us: 1_700_000_000_000,
    };
    let bytes = rmp_serde::to_vec(&s).expect("encode");
    let back: TriggerSchema = rmp_serde::from_slice(&bytes).expect("decode");
    assert_eq!(s, back);
}

#[test]
fn trigger_scan_prefix_test_prefix_is_correct_byte_string() {
    assert_eq!(trigger_scan_prefix(), b"schema:trigger:");
}

// ── AFTER COMMIT event journal (R192) ───────────────────────────────────────

#[test]
fn pending_key_encodes_name_and_be_seq() {
    let k = encode_trigger_pending_key("audit", 0x0102030405060708);
    let mut expect = b"trigger_pending:audit:".to_vec();
    expect.extend_from_slice(&0x0102030405060708u64.to_be_bytes());
    assert_eq!(k, expect);
    assert!(k.starts_with(trigger_pending_scan_prefix()));
}

#[test]
fn failure_key_encodes_name_and_be_seq() {
    let k = encode_trigger_failure_key("audit", 42);
    let mut expect = b"trigger_failures:audit:".to_vec();
    expect.extend_from_slice(&42u64.to_be_bytes());
    assert_eq!(k, expect);
    assert!(k.starts_with(trigger_failures_scan_prefix()));
}

#[test]
fn pending_and_failure_prefixes_are_disjoint() {
    let p = encode_trigger_pending_key("t", 1);
    let f = encode_trigger_failure_key("t", 1);
    assert!(!p.starts_with(trigger_failures_scan_prefix()));
    assert!(!f.starts_with(trigger_pending_scan_prefix()));
}

#[test]
fn be_seq_suffix_keeps_queue_in_enqueue_order() {
    // Lexicographic key order over the BE-encoded seq must match numeric
    // order, so a prefix scan drains oldest-first.
    let a = encode_trigger_pending_key("t", 1);
    let b = encode_trigger_pending_key("t", 2);
    let c = encode_trigger_pending_key("t", 256);
    assert!(a < b);
    assert!(b < c);
}

#[test]
fn decode_seq_recovers_be_suffix_and_rejects_short_keys() {
    let k = encode_trigger_pending_key("audit", 123_456);
    assert_eq!(decode_trigger_event_seq(&k), Some(123_456));
    let f = encode_trigger_failure_key("x", u64::MAX);
    assert_eq!(decode_trigger_event_seq(&f), Some(u64::MAX));
    // Fewer than 8 trailing bytes → no seq.
    assert_eq!(decode_trigger_event_seq(b"short"), None);
}

#[test]
fn pending_event_serde_roundtrip() {
    let mut params = BTreeMap::new();
    params.insert("event".to_string(), Value::String("CREATE".into()));
    params.insert("node".to_string(), Value::Int(42));
    let ev = PendingTriggerEvent {
        trigger_name: "audit".into(),
        params,
        attempt: 2,
        generation: 1,
        first_seen_us: 1_700_000_000_000,
        next_attempt_us: 1_700_000_001_000,
    };
    let bytes = rmp_serde::to_vec(&ev).expect("encode");
    let back: PendingTriggerEvent = rmp_serde::from_slice(&bytes).expect("decode");
    assert_eq!(ev, back);
}

#[test]
fn failed_event_serde_roundtrip() {
    let mut params = BTreeMap::new();
    params.insert("event".to_string(), Value::String("DELETE".into()));
    let ev = FailedTriggerEvent {
        trigger_name: "audit".into(),
        params,
        error_chain: vec!["body error".into(), "root cause".into()],
        attempts: 3,
        first_fail_us: 10,
        last_fail_us: 99,
        cascade_overflow: true,
    };
    let bytes = rmp_serde::to_vec(&ev).expect("encode");
    let back: FailedTriggerEvent = rmp_serde::from_slice(&bytes).expect("decode");
    assert_eq!(ev, back);
}
