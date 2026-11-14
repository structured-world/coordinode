use super::*;

fn entry(retention: ConsumerRetentionPolicy) -> RegistryEntry {
    RegistryEntry {
        consumer_id: "sink-1".into(),
        incarnation: 3,
        kind: ConsumerKind::OplogEvents,
        scope: TopologyScope::Shard(2),
        scope_origin: TopologyScope::Cluster,
        retention,
        state: RegistrationState::Live,
        checkpoint_seqno: 512,
        last_heartbeat_ts_ms: 1_000,
        started_at: 500,
        follows: Some(Box::new(crate::registry::PriorIncarnation {
            incarnation: 2,
            reason: crate::registry::TerminalReason::Cancelled,
            acknowledged: 480,
            retained_from: 400,
        })),
    }
}

/// Only a live STRICT registration holds the source's history: ordinary
/// progress never does, and an ended registration holds nothing whatever
/// its policy was.
#[test]
fn only_a_live_strict_registration_holds_history() {
    assert!(entry(ConsumerRetentionPolicy::Strict).holds_history());
    assert!(!entry(ConsumerRetentionPolicy::Ordinary).holds_history());
    let mut ended = entry(ConsumerRetentionPolicy::Strict);
    ended.state = RegistrationState::Terminated {
        reason: crate::registry::TerminalReason::Cancelled,
        at_ms: 0,
        retained_from: 0,
    };
    assert!(!ended.holds_history());
}

/// A heartbeat record round-trips through its fixed form, a record of any
/// other length is refused, and its key stays apart from the registration's.
#[test]
fn a_heartbeat_record_round_trips_and_has_its_own_key() {
    let hb = Heartbeat {
        incarnation: 3,
        ts_ms: 7_000,
    };
    assert_eq!(Heartbeat::decode(&hb.encode()), Some(hb));
    assert_eq!(Heartbeat::decode(&[0u8; 15]), None);
    assert_ne!(
        encode_heartbeat_key("sink-1"),
        encode_registry_key("sink-1")
    );
    assert!(!encode_heartbeat_key("sink-1").starts_with(REGISTRY_KEY_PREFIX));
}

/// Liveness counts a heartbeat of the record's own incarnation, the later
/// of it and the record's own time, and never one of another incarnation.
#[test]
fn only_a_heartbeat_of_the_same_incarnation_counts() {
    let live = entry(ConsumerRetentionPolicy::Strict);
    let later = Heartbeat {
        incarnation: 3,
        ts_ms: 9_000,
    };
    let earlier = Heartbeat {
        incarnation: 3,
        ts_ms: 500,
    };
    let other = Heartbeat {
        incarnation: 2,
        ts_ms: 9_000,
    };
    assert_eq!(
        live.clone()
            .with_heartbeat(Some(later))
            .last_heartbeat_ts_ms,
        9_000
    );
    assert_eq!(
        live.clone()
            .with_heartbeat(Some(earlier))
            .last_heartbeat_ts_ms,
        1_000
    );
    assert_eq!(
        live.clone()
            .with_heartbeat(Some(other))
            .last_heartbeat_ts_ms,
        1_000
    );
    assert_eq!(live.with_heartbeat(None).last_heartbeat_ts_ms, 1_000);
}

#[test]
fn entry_roundtrips_through_msgpack() {
    for retention in [
        ConsumerRetentionPolicy::Strict,
        ConsumerRetentionPolicy::Ordinary,
    ] {
        let mut e = entry(retention);
        e.state = RegistrationState::Terminated {
            reason: crate::registry::TerminalReason::Cancelled,
            at_ms: 77,
            retained_from: 12,
        };
        let decoded = RegistryEntry::decode(&e.encode().expect("encode")).expect("decode");
        assert_eq!(e, decoded);
    }
}

#[test]
fn key_is_prefixed_and_recoverable() {
    let k = encode_registry_key("kafka-eu");
    assert!(k.starts_with(REGISTRY_KEY_PREFIX));
    assert_eq!(&k[REGISTRY_KEY_PREFIX.len()..], b"kafka-eu");
}
