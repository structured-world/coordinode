#![allow(clippy::expect_used, clippy::unwrap_used)]

use tonic_types::StatusExt;

use super::*;

fn spiffe(path: &str) -> MemberIdentity {
    MemberIdentity::Spiffe(format!("spiffe://cluster.test/{path}"))
}

fn member(node_id: u64, identity: Option<MemberIdentity>, voter: bool) -> ViewMember {
    ViewMember {
        node_id,
        node: MemberNode::new(format!("127.0.0.1:{}", 7000 + node_id), identity),
        voter,
    }
}

fn reason(status: &Status) -> String {
    status
        .get_details_error_info()
        .map(|info| info.reason)
        .unwrap_or_default()
}

/// A call's deployment travels in its metadata and reads back as written;
/// a call naming none, or bytes that are not one, names no deployment.
#[test]
fn the_deployment_header_round_trips() {
    let id = DeploymentId([9; 16]);
    let mut metadata = MetadataMap::new();
    assert_eq!(read_deployment(&metadata), None);
    write_deployment(&mut metadata, id);
    assert_eq!(read_deployment(&metadata), Some(id));

    let mut short = MetadataMap::new();
    short.insert_bin(
        DEPLOYMENT_METADATA_KEY,
        tonic::metadata::MetadataValue::from_bytes(&[1, 2, 3]),
    );
    assert_eq!(read_deployment(&short), None);
}

/// An identity binds to the member that records it, with that member's
/// role; an identity no member records binds to none, and a member
/// without an identity is never matched.
#[test]
fn a_membership_binds_identities_to_members() {
    let view = MemberView {
        index: Some(4),
        members: vec![
            member(1, Some(spiffe("node/1")), true),
            member(2, Some(spiffe("node/2")), false),
            member(3, None, true),
        ],
    };
    assert_eq!(view.find(&spiffe("node/1")), Some((1, true)));
    assert_eq!(view.find(&spiffe("node/2")), Some((2, false)));
    assert_eq!(view.find(&spiffe("node/3")), None);
    // A prefix of a bound identity is another identity.
    assert_eq!(view.find(&spiffe("node")), None);
}

/// A call speaking for another member than the one its caller is bound
/// to is refused, naming why.
#[test]
fn a_call_speaks_only_for_its_own_member() {
    Admission::same_member(2, 2).expect("its own member");
    let refused = Admission::same_member(2, 3).expect_err("another member");
    assert_eq!(refused.code(), Code::PermissionDenied);
    assert_eq!(reason(&refused), WRONG_MEMBER);
}

/// Only a voter campaigns, leads or hands leadership over; any member
/// may exchange versions, ask who the members are, or use another peer
/// service.
#[test]
fn operations_that_need_a_voter() {
    for op in [
        Operation::Vote,
        Operation::Append,
        Operation::Snapshot,
        Operation::TransferLeader,
    ] {
        assert!(op.needs_voter(), "{op:?}");
    }
    for op in [Operation::Exchange, Operation::Members, Operation::Service] {
        assert!(!op.needs_voter(), "{op:?}");
    }
}
