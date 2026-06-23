use super::*;
use crate::registry::types::InitialSeqno;

fn sample() -> RegistryEntry {
    RegistryEntry {
        consumer_id: "sink-1".into(),
        kind: ConsumerKind::OplogEvents,
        scope: TopologyScope::Shard(2),
        scope_origin: TopologyScope::Cluster,
        checkpoint_seqno: 512,
        last_heartbeat_ts_ms: 1_000,
        ttl_ms: 5_000,
    }
}

#[test]
fn entry_roundtrips_through_msgpack() {
    let e = sample();
    let decoded = RegistryEntry::decode(&e.encode().expect("encode")).expect("decode");
    assert_eq!(e, decoded);
}

#[test]
fn key_is_prefixed_and_recoverable() {
    let k = encode_registry_key("kafka-eu");
    assert!(k.starts_with(REGISTRY_KEY_PREFIX));
    assert_eq!(&k[REGISTRY_KEY_PREFIX.len()..], b"kafka-eu");
}

#[test]
fn expiry_respects_ttl_and_persistent_zero() {
    let e = sample(); // ttl 5000, last hb 1000
    assert!(!e.is_expired(5_000), "within ttl");
    assert!(
        !e.is_expired(6_000),
        "exactly at boundary is not yet expired"
    );
    assert!(e.is_expired(6_001), "past ttl");

    // ttl_ms == 0 → persistent, never expires even far in the future.
    let persistent = RegistryEntry {
        ttl_ms: 0,
        ..sample()
    };
    assert!(!persistent.is_expired(u64::MAX));
}

#[test]
fn initial_seqno_variants_are_distinct() {
    // Guards against an accidental collapse of the initial-seqno modes
    // the registry must honour at register time.
    assert_ne!(InitialSeqno::FromNow, InitialSeqno::FromEarliestRetained);
    assert_ne!(InitialSeqno::At(0), InitialSeqno::FromNow);
    assert_eq!(InitialSeqno::At(7), InitialSeqno::At(7));
}
