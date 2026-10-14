#![allow(clippy::expect_used, clippy::unwrap_used)]

use super::*;

/// A recorded identity and the verified identity a connection proves are
/// the same identity in both directions, for both profiles.
#[test]
fn recorded_and_verified_identities_convert_both_ways() {
    let spiffe = PeerIdentity::Spiffe(SpiffeId::parse("spiffe://td/a/b").expect("id"));
    let host = PeerIdentity::Host(HostIdentity {
        scope: "instance-1".to_owned(),
        subject: "part-7".to_owned(),
    });
    for verified in [spiffe, host] {
        let recorded = MemberIdentity::from(&verified);
        assert_eq!(recorded.to_peer().expect("parses"), verified);
    }
}

/// A recorded SPIFFE ID that no longer parses names nobody: it cannot be
/// proven, so it is reported rather than matched loosely.
#[test]
fn an_unparseable_recorded_id_is_reported() {
    let recorded = MemberIdentity::Spiffe("spiffe://td/a/".to_owned());
    assert_eq!(recorded.to_peer(), Err(SpiffeIdError::TrailingSlash));
}

/// The record survives the encoding the membership is stored and
/// replicated in, identity included.
#[test]
fn a_member_round_trips_through_its_stored_encoding() {
    let member = MemberNode::new(
        "10.0.0.2:7080".to_owned(),
        Some(MemberIdentity::Spiffe("spiffe://td/n2".to_owned())),
    );
    let bytes = rmp_serde::to_vec(&member).expect("encode");
    let back: MemberNode = rmp_serde::from_slice(&bytes).expect("decode");
    assert_eq!(back, member);
    let bare = MemberNode::default();
    let back: MemberNode =
        rmp_serde::from_slice(&rmp_serde::to_vec(&bare).expect("encode")).expect("decode");
    assert_eq!(back.identity, None);
}
