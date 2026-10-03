use super::*;

#[test]
fn registration_roundtrips_through_msgpack() {
    // The record replicates as msgpack; encode/decode must be lossless across
    // every field, including the scope and the policy.
    for retention in [
        ConsumerRetentionPolicy::Strict,
        ConsumerRetentionPolicy::Bounded(
            ValidatedRetentionBounds::new(60_000, 1 << 30, Some(30_000)).expect("bounds"),
        ),
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

/// A BOUNDED limit of zero is refused: it would be a consumer that ends at
/// once, or one with no bound at all where a bound was promised.
#[test]
fn zero_bounds_are_refused() {
    for (lag, bytes, liveness) in [(0, 1, None), (1, 0, None), (1, 1, Some(0))] {
        assert!(
            matches!(
                ValidatedRetentionBounds::new(lag, bytes, liveness),
                Err(RegistryError::InvalidRetention(_))
            ),
            "({lag}, {bytes}, {liveness:?}) was admitted"
        );
    }
    let b = ValidatedRetentionBounds::new(1, 2, None).expect("finite bounds");
    assert_eq!(
        (
            b.max_progress_lag_ms(),
            b.max_retained_bytes(),
            b.liveness_timeout_ms()
        ),
        (1, 2, None)
    );
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
        reason: TerminalReason::ProgressLagExceeded,
        checkpoint: 99,
    };
    let msg = format!("{e}");
    assert!(
        msg.contains("ProgressLagExceeded") && msg.contains("99") && msg.contains('4'),
        "msg: {msg}"
    );
}
