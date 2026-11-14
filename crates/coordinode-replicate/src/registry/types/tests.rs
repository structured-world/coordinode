use super::*;

#[test]
fn registration_roundtrips_through_msgpack() {
    // The record replicates as msgpack; encode/decode must be lossless across
    // every field, including the scope and the policy.
    for retention in [
        ConsumerRetentionPolicy::Strict,
        ConsumerRetentionPolicy::Ordinary,
    ] {
        let reg = ConsumerRegistration {
            consumer_id: "kafka-sink-eu".to_string(),
            kind: ConsumerKind::OplogEvents,
            scope: TopologyScope::Dc("eu-west".to_string()),
            initial_seqno: InitialSeqno::At(4096),
            retention,
        };
        let bytes = rmp_serde::to_vec(&reg).expect("encode");
        let decoded: ConsumerRegistration = rmp_serde::from_slice(&bytes).expect("decode");
        assert_eq!(reg, decoded);
    }
}

#[test]
fn every_scope_variant_roundtrips() {
    for scope in [
        TopologyScope::Cluster,
        TopologyScope::Dc("dc1".into()),
        TopologyScope::Rack("r7".into()),
        TopologyScope::Node(42),
        TopologyScope::Shard(3),
    ] {
        let bytes = rmp_serde::to_vec(&scope).expect("encode");
        let decoded: TopologyScope = rmp_serde::from_slice(&bytes).expect("decode");
        assert_eq!(scope, decoded);
    }
}

#[test]
fn every_kind_variant_roundtrips() {
    for kind in [
        ConsumerKind::OplogEvents,
        ConsumerKind::LsmStateDelta,
        ConsumerKind::MvccSnapshotPin,
        ConsumerKind::Ephemeral,
    ] {
        let bytes = rmp_serde::to_vec(&kind).expect("encode");
        let decoded: ConsumerKind = rmp_serde::from_slice(&bytes).expect("decode");
        assert_eq!(kind, decoded);
    }
}

/// A handle addresses one incarnation of one consumer: neither another id
/// nor another incarnation of the same id is the same handle.
#[test]
fn a_handle_addresses_one_incarnation() {
    let h = RegisteredHandle::new("c1", 2);
    assert_eq!((h.consumer_id(), h.incarnation()), ("c1", 2));
    assert_ne!(h, RegisteredHandle::new("c2", 2));
    assert_ne!(h, RegisteredHandle::new("c1", 3));
}

/// A stored policy the registry no longer offers (a byte and age quota) does
/// not decode as any policy it does offer: the record is refused, never
/// reused as STRICT or ordinary progress it never was.
#[test]
fn a_withdrawn_quota_policy_does_not_decode() {
    #[derive(Serialize)]
    enum Stored {
        Bounded((u64, u64, Option<u64>)),
    }
    let bytes = rmp_serde::to_vec(&Stored::Bounded((60_000, 1 << 30, None))).expect("encode");
    assert!(rmp_serde::from_slice::<ConsumerRetentionPolicy>(&bytes).is_err());
}

#[test]
fn retention_lost_error_carries_both_positions() {
    let e = RegistryError::RetentionLost {
        checkpoint: 10,
        floor: 25,
    };
    let msg = format!("{e}");
    assert!(msg.contains("10") && msg.contains("25"), "msg: {msg}");
}

#[test]
fn a_terminated_error_names_its_reason_and_checkpoint() {
    let e = RegistryError::Terminated {
        consumer_id: "c".into(),
        incarnation: 4,
        reason: TerminalReason::Cancelled,
        checkpoint: 99,
        retained_from: 57,
    };
    let msg = format!("{e}");
    assert!(
        msg.contains("Cancelled") && msg.contains("99") && msg.contains('4') && msg.contains("57"),
        "msg: {msg}"
    );
}
