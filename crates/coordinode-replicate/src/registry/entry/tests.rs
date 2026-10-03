use super::*;
use crate::registry::types::ValidatedRetentionBounds;

/// A source fixed at construction: head, when the checkpoint's work was
/// produced, and the bytes it requires.
struct Fixed {
    head: u64,
    produced: Option<u64>,
    bytes: Option<u64>,
}

impl RetentionSource for Fixed {
    fn head(&self, _: ConsumerKind) -> u64 {
        self.head
    }
    fn first_retained(&self, _: ConsumerKind) -> u64 {
        0
    }
    fn accounts(&self, _: ConsumerKind) -> bool {
        true
    }
    fn produced_at_ms(&self, _: ConsumerKind, _: u64) -> Option<u64> {
        self.produced
    }
    fn retained_bytes_from(&self, _: ConsumerKind, _: u64) -> Option<u64> {
        self.bytes
    }
    fn admits(&self, _: ConsumerKind) -> bool {
        true
    }
}

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
    }
}

fn bounded(lag: u64, bytes: u64, liveness: Option<u64>) -> ConsumerRetentionPolicy {
    ConsumerRetentionPolicy::Bounded(
        ValidatedRetentionBounds::new(lag, bytes, liveness).expect("bounds"),
    )
}

const CAUGHT_UP: Fixed = Fixed {
    head: 512,
    produced: None,
    bytes: Some(0),
};

#[test]
fn entry_roundtrips_through_msgpack() {
    let mut e = entry(bounded(10, 20, Some(30)));
    e.state = RegistrationState::Terminated {
        reason: TerminalReason::RetainedBytesExceeded,
        at_ms: 77,
    };
    let decoded = RegistryEntry::decode(&e.encode().expect("encode")).expect("decode");
    assert_eq!(e, decoded);
}

#[test]
fn key_is_prefixed_and_recoverable() {
    let k = encode_registry_key("kafka-eu");
    assert!(k.starts_with(REGISTRY_KEY_PREFIX));
    assert_eq!(&k[REGISTRY_KEY_PREFIX.len()..], b"kafka-eu");
    assert!(
        !k.starts_with(LEGACY_KEY_PREFIX),
        "a new record must not be swept as one of the old format"
    );
}

/// STRICT never ends on its own, whatever the clock, lag or size.
#[test]
fn strict_never_terminates() {
    let e = entry(ConsumerRetentionPolicy::Strict);
    let far_behind = Fixed {
        head: u64::MAX,
        produced: Some(0),
        bytes: Some(u64::MAX),
    };
    assert_eq!(e.termination(u64::MAX, &far_behind), None);
    assert_eq!(e.next_deadline_ms(&far_behind), None);
}

/// Liveness ends a BOUNDED consumer strictly past its timeout, and only when
/// one was chosen.
#[test]
fn liveness_is_judged_only_when_chosen_and_strictly_past_it() {
    let e = entry(bounded(u64::MAX, u64::MAX, Some(5_000)));
    assert_eq!(e.termination(6_000, &CAUGHT_UP), None, "at the boundary");
    assert_eq!(
        e.termination(6_001, &CAUGHT_UP),
        Some(TerminalReason::LivenessExpired)
    );
    assert_eq!(e.next_deadline_ms(&CAUGHT_UP), Some(6_001));

    let no_liveness = entry(bounded(u64::MAX, u64::MAX, None));
    assert_eq!(no_liveness.termination(u64::MAX, &CAUGHT_UP), None);
    assert_eq!(no_liveness.next_deadline_ms(&CAUGHT_UP), None);
}

/// A clock behind the stored heartbeat is no evidence of absence.
#[test]
fn a_clock_behind_the_heartbeat_does_not_expire_it() {
    let e = entry(bounded(u64::MAX, u64::MAX, Some(1)));
    assert_eq!(e.termination(0, &CAUGHT_UP), None);
}

/// Progress lag is the age of the oldest unacknowledged work, and only while
/// some is outstanding.
#[test]
fn progress_lag_counts_only_outstanding_work() {
    let e = entry(bounded(1_000, u64::MAX, None));
    let behind = Fixed {
        head: 600,
        produced: Some(10_000),
        bytes: Some(0),
    };
    assert_eq!(e.termination(11_000, &behind), None);
    assert_eq!(
        e.termination(11_001, &behind),
        Some(TerminalReason::ProgressLagExceeded)
    );
    assert_eq!(e.next_deadline_ms(&behind), Some(11_001));

    let idle = Fixed {
        head: 512,
        produced: Some(0),
        bytes: Some(0),
    };
    assert_eq!(e.termination(u64::MAX, &idle), None, "nothing outstanding");
    assert_eq!(e.next_deadline_ms(&idle), None);
}

/// Bytes the checkpoint requires above the limit end the registration.
#[test]
fn retained_bytes_above_the_limit_terminate() {
    let e = entry(bounded(u64::MAX, 4_096, None));
    let at_limit = Fixed {
        head: 512,
        produced: None,
        bytes: Some(4_096),
    };
    let over = Fixed {
        head: 512,
        produced: None,
        bytes: Some(4_097),
    };
    assert_eq!(e.termination(0, &at_limit), None);
    assert_eq!(
        e.termination(0, &over),
        Some(TerminalReason::RetainedBytesExceeded)
    );
}

/// An ended registration is never judged again.
#[test]
fn an_ended_registration_is_not_judged() {
    let mut e = entry(bounded(1, 1, Some(1)));
    e.state = RegistrationState::Terminated {
        reason: TerminalReason::Cancelled,
        at_ms: 0,
    };
    let over_everything = Fixed {
        head: u64::MAX,
        produced: Some(0),
        bytes: Some(u64::MAX),
    };
    assert_eq!(e.termination(u64::MAX, &over_everything), None);
    assert_eq!(e.next_deadline_ms(&over_everything), None);
}
