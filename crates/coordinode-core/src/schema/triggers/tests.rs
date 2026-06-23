use super::*;

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
