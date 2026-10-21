use super::*;

/// A registration refused because this member does not lead, or stopped
/// leading while it waited, leaves the statement that needed the names
/// refused as on a member that does not lead, so the host knows it ran
/// nothing and may send it again, instead of an untyped failure.
#[test]
fn a_registration_off_the_leader_refuses_the_statement_as_not_leader() {
    assert_eq!(
        registration_refusal(ProposalError::NotLeader { leader_id: Some(2) }),
        DictionaryError::NotLeader { leader_id: Some(2) }
    );
    for lost in [
        ProposalError::NotLeader { leader_id: None },
        ProposalError::LeadershipLost { term: 4 },
        ProposalError::ShuttingDown,
    ] {
        assert_eq!(
            registration_refusal(lost.clone()),
            DictionaryError::NotLeader { leader_id: None },
            "{lost:?}"
        );
    }
    // Any other failure stays a failed registration.
    assert!(matches!(
        registration_refusal(ProposalError::Raft("down".into())),
        DictionaryError::Registration(_)
    ));
}

/// The refusal reaches the statement as the not-leader refusal every write
/// off the leader gets, not as a dictionary failure.
#[test]
fn a_registration_off_the_leader_is_a_not_leader_statement() {
    let refused: coordinode_query::executor::runner::ExecutionError =
        DictionaryError::NotLeader { leader_id: Some(3) }.into();
    assert!(
        matches!(
            refused,
            coordinode_query::executor::runner::ExecutionError::NotLeader { leader_id: Some(3) }
        ),
        "{refused:?}"
    );
}
