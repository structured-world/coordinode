use super::*;

/// A grant whose leader stepped down while it waited leaves the statement
/// that needed it refused as on a member that does not lead, so the host
/// knows it ran nothing and may send it again. It never surfaces as an
/// untyped failure saying the outcome is unknown.
#[test]
fn a_grant_lost_with_the_lead_refuses_the_statement_as_not_leader() {
    for lost in [
        ProposalError::LeadershipLost { term: 2 },
        ProposalError::ShuttingDown,
    ] {
        assert_eq!(
            lease_refusal(lost.clone()),
            IdLeaseError::NotLeader { leader_id: None },
            "{lost:?}"
        );
    }
    assert_eq!(
        lease_refusal(ProposalError::NotLeader { leader_id: Some(3) }),
        IdLeaseError::NotLeader { leader_id: Some(3) }
    );
}
