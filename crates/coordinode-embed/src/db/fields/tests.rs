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

/// Concurrent first uses of the same names make one proposal between them,
/// not one each: a registration waits for the one in flight and finds the
/// names bound. Each extra proposal is a consensus round the writes of a
/// fresh label waited on in turn (16 writers, 13 of them 108 to 286 ms).
#[test]
fn concurrent_first_uses_of_a_name_make_one_proposal() {
    use coordinode_core::schema::label_id::LabelRegistrar as _;
    let dir = tempfile::tempdir().expect("tempdir");
    let db = crate::Database::open(dir.path()).expect("open");
    // Its own id generator: what it hands out is what this dictionary
    // proposed, whatever the database's own work proposes meanwhile.
    let ids = Arc::new(ProposalIdGenerator::new());
    let dictionary = FieldDictionary::open(
        Arc::clone(&db.engine),
        Arc::clone(&db.pipeline),
        Arc::clone(&ids),
        Arc::clone(&db.oracle),
    )
    .expect("dictionary");
    let used = |ids: &ProposalIdGenerator| ids.next().as_raw();
    let before = used(&ids);
    let writers = 16;
    let start = std::sync::Barrier::new(writers);
    std::thread::scope(|scope| {
        for _ in 0..writers {
            scope.spawn(|| {
                start.wait();
                dictionary
                    .register(&["fresh_a", "fresh_b"])
                    .expect("register");
                dictionary.bind_labels(&["Fresh"]).expect("bind");
            });
        }
    });
    let proposed = used(&ids) - before - 1;
    assert_eq!(
        proposed, 2,
        "one registration and one binding between {writers} writers"
    );
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
