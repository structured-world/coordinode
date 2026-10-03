use super::*;

use coordinode_core::txn::proposal::{MetadataCommand, Mutation};
use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;

const P: VersionPair = VersionPair {
    engine: 1,
    host_epoch: 0,
};
const P2: VersionPair = VersionPair {
    engine: 2,
    host_epoch: 0,
};

struct Rig {
    engine: Arc<StorageEngine>,
    oracle: Arc<TimestampOracle>,
    applied: tokio::sync::watch::Sender<GroupPairs>,
    _dir: tempfile::TempDir,
}

impl Rig {
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        let oracle = Arc::new(TimestampOracle::new());
        let engine = Arc::new(
            StorageEngine::open_with_oracle(
                &StorageConfig::with_endpoints(vec![EndpointConfig::new(
                    "default",
                    dir.path(),
                    Media::Hdd,
                    Durability::Durable,
                    Tier::Warm,
                )]),
                Arc::clone(&oracle),
            )
            .expect("engine"),
        );
        let (applied, _) = tokio::sync::watch::channel(GroupPairs::from(Vec::new()));
        Self {
            engine,
            oracle,
            applied,
            _dir: dir,
        }
    }

    /// Record `pair` as the group's next pair, as an applied entry would.
    fn record(&self, pair: VersionPair) {
        self.engine
            .apply_proposal_at(
                &[Mutation::Command(MetadataCommand::RecordGroupPair { pair })],
                self.oracle.next().as_raw(),
            )
            .expect("apply");
        let records =
            coordinode_storage::engine::metadata::group_pair_records(&self.engine).expect("read");
        self.applied.send_replace(GroupPairs::from(records));
    }

    fn gate(&self, node_id: u64, pair: VersionPair) -> VersionGate {
        VersionGate::new(
            node_id,
            0,
            pair,
            self.applied.subscribe(),
            Arc::new(AtomicU64::new(42)),
        )
    }
}

fn peer(node_id: u64, pair: VersionPair, group_pair: Option<RecordedPair>) -> Handshake {
    Handshake {
        node_id,
        group_id: 0,
        pair,
        group_pair,
        leader: Some((node_id, format!("10.0.0.{node_id}:7080"))),
    }
}

/// A group that recorded nothing yet, or recorded this member's pair,
/// leaves it matched; peers of its pair are admitted.
#[test]
fn a_member_of_the_recorded_pair_is_matched() {
    let rig = Rig::new();
    let gate = rig.gate(1, P);
    assert_eq!(gate.state(), MemberState::Matched);
    assert_eq!(gate.pair_to_record(), Some(P), "the first leader records");
    rig.record(P);
    assert_eq!(gate.state(), MemberState::Matched);
    assert_eq!(gate.pair_to_record(), None, "already recorded");
    assert_eq!(gate.admit(Ok(peer(2, P, None))), Ok(()));
    assert_eq!(gate.peer_matches(2), Some(true));
}

/// A member updated before its group moved is ahead: read-only, but it
/// still admits members of its own pair and records its pair once it
/// leads.
#[test]
fn a_member_ahead_of_its_group_is_read_only_but_may_lead() {
    let rig = Rig::new();
    rig.record(P);
    let gate = rig.gate(3, P2);
    let MemberState::Mismatched(m) = gate.state() else {
        panic!("ahead of the group");
    };
    assert!(!m.behind);
    assert_eq!((m.own, m.group, m.as_of), (P2, P, 42));
    assert_eq!(gate.admit(Ok(peer(2, P2, None))), Ok(()));
    assert_eq!(gate.pair_to_record(), Some(P2));
    assert!(matches!(
        gate.admit(Ok(peer(1, P, None))),
        Err(Refusal::Pair { theirs, ours }) if theirs == P && ours == P2
    ));
    assert_eq!(gate.peer_matches(1), Some(false));
}

/// Once a peer reports that the group recorded a later pair, a member of
/// the earlier pair is behind: it refuses to follow anyone, names the
/// leader the peer knows, and never records its own pair again.
#[test]
fn a_member_left_behind_learns_it_from_a_peer() {
    let rig = Rig::new();
    rig.record(P);
    let gate = rig.gate(1, P);
    assert_eq!(gate.state(), MemberState::Matched);
    let moved = RecordedPair { pair: P2, seq: 2 };
    gate.observe(&peer(2, P2, Some(moved)));
    assert_eq!(gate.group_pair(), Some(moved));
    let MemberState::Mismatched(m) = gate.state() else {
        panic!("behind the group");
    };
    assert!(m.behind);
    assert_eq!(m.leader, Some((2, "10.0.0.2:7080".into())));
    assert!(m.to_string().contains("update this member"), "{m}");
    assert!(matches!(
        gate.admit(Ok(peer(3, P, None))),
        Err(Refusal::Behind(_))
    ));
    assert_eq!(gate.pair_to_record(), None);
}

/// A member left behind keeps the leader its peer named: its own consensus
/// still believes in the leader of the group as it was.
#[test]
fn a_stale_consensus_does_not_override_the_reported_leader() {
    let rig = Rig::new();
    rig.record(P);
    let gate = rig.gate(1, P);
    gate.set_leader(Some((1, "10.0.0.1:7080".into())));
    gate.observe(&peer(2, P2, Some(RecordedPair { pair: P2, seq: 2 })));
    gate.set_leader(Some((1, "10.0.0.1:7080".into())));
    let MemberState::Mismatched(m) = gate.state() else {
        panic!("behind the group");
    };
    assert_eq!(m.leader.map(|(id, _)| id), Some(2));
}

/// An older report never replaces a later record.
#[test]
fn a_stale_report_changes_nothing() {
    let rig = Rig::new();
    rig.record(P);
    rig.record(P2);
    let gate = rig.gate(1, P2);
    gate.observe(&peer(2, P, Some(RecordedPair { pair: P, seq: 1 })));
    assert_eq!(gate.group_pair().map(|r| r.seq), Some(2));
    assert_eq!(gate.state(), MemberState::Matched);
}

/// A move of the host epoch alone is a move like any other.
#[test]
fn the_host_epoch_alone_separates_members() {
    let rig = Rig::new();
    rig.record(P);
    let gate = rig.gate(1, P);
    let epoch = VersionPair { host_epoch: 5, ..P };
    assert!(matches!(
        gate.admit(Ok(peer(2, epoch, None))),
        Err(Refusal::Pair { .. })
    ));
}

/// A call without a record, or from another group, is refused.
#[test]
fn calls_without_a_record_or_from_another_group_are_refused() {
    let rig = Rig::new();
    let gate = rig.gate(1, P);
    assert!(matches!(
        gate.admit(Err(Refusal::NoHandshake("none".into()))),
        Err(Refusal::NoHandshake(_))
    ));
    let other = Handshake {
        group_id: 9,
        ..peer(2, P, None)
    };
    assert!(matches!(
        gate.admit(Ok(other)),
        Err(Refusal::OtherGroup { theirs: 9, ours: 0 })
    ));
}

/// The record travels in call metadata and comes back unchanged; a call
/// without it is named as such.
#[test]
fn the_record_round_trips_through_metadata() {
    let mut metadata = tonic::metadata::MetadataMap::new();
    assert!(matches!(
        read_handshake(&metadata),
        Err(Refusal::NoHandshake(_))
    ));
    let h = peer(2, P, Some(RecordedPair { pair: P, seq: 1 }));
    write_handshake(&mut metadata, &h);
    assert_eq!(read_handshake(&metadata), Ok(h.clone()));
    let status = refusal_status(
        &Refusal::Pair {
            theirs: P2,
            ours: P,
        },
        &h,
    );
    assert_eq!(status.code(), tonic::Code::FailedPrecondition);
    assert_eq!(read_handshake(status.metadata()), Ok(h));
}
