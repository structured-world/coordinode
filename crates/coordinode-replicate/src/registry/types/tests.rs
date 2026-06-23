use super::*;

#[test]
fn registration_roundtrips_through_msgpack() {
    // The record replicates through Raft as msgpack; encode/decode must
    // be lossless across every field, including the scope variant.
    let reg = ConsumerRegistration {
        consumer_id: "kafka-sink-eu".to_string(),
        kind: ConsumerKind::OplogEvents,
        scope: TopologyScope::Dc("eu-west".to_string()),
        initial_seqno: InitialSeqno::At(4096),
        ttl_ms: 30_000,
    };
    let bytes = rmp_serde::to_vec(&reg).expect("encode");
    let decoded: ConsumerRegistration = rmp_serde::from_slice(&bytes).expect("decode");
    assert_eq!(reg, decoded);
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

#[test]
fn handle_addresses_its_consumer_id() {
    let h = RegisteredHandle::new("c1");
    assert_eq!(h.consumer_id(), "c1");
    // Distinct ids produce distinct (non-interchangeable) handles.
    assert_ne!(h, RegisteredHandle::new("c2"));
}

#[test]
fn retention_lost_error_carries_both_seqnos() {
    let e = RegistryError::RetentionLost {
        checkpoint: 10,
        floor: 25,
    };
    let msg = format!("{e}");
    assert!(msg.contains("10") && msg.contains("25"), "msg: {msg}");
}
