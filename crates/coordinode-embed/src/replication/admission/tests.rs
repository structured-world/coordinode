use super::*;
use coordinode_core::txn::write_concern::WriteConcern;

fn standalone(defaults: AdmissionDefaults) -> (Admitter, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let database = Database::open(dir.path()).expect("open database");
    (
        Admitter::new(Arc::new(RwLock::new(database))).with_defaults(defaults),
        dir,
    )
}

/// What a statement leaves out comes from the defaults, what it names wins,
/// and the write concern it leaves out stays unset for the database to fill.
#[test]
fn a_setting_left_out_is_the_default_and_a_named_one_wins() {
    let (admitter, _dir) = standalone(AdmissionDefaults {
        read_concern: ReadConcernLevel::Majority,
        read_preference: ReadPreference::Nearest,
    });

    let unnamed = admitter
        .check("RETURN 1", &Requested::default())
        .expect("checked");
    assert_eq!(unnamed.read_concern.level, ReadConcernLevel::Majority);
    assert_eq!(unnamed.read_preference, ReadPreference::Nearest);
    assert_eq!(unnamed.write_concern, None);

    let named = admitter
        .check(
            "RETURN 1",
            &Requested {
                read_concern: Some(ReadConcernLevel::Linearizable),
                read_preference: Some(ReadPreference::Primary),
                write_concern: Some(WriteConcern::majority()),
                ..Requested::default()
            },
        )
        .expect("checked");
    assert_eq!(named.read_concern.level, ReadConcernLevel::Linearizable);
    assert_eq!(named.read_preference, ReadPreference::Primary);
    assert_eq!(named.write_concern, Some(WriteConcern::majority()));
}

/// A combination that cannot be served is refused before anything waits or
/// runs, with the refusal that names the settings: a causal read at LOCAL or
/// LINEARIZABLE, a causal write under a concern that can be lost, a pinned
/// timestamp outside SNAPSHOT, a session SET with no session.
#[test]
fn an_unservable_combination_is_refused_before_it_runs() {
    let (admitter, _dir) = standalone(AdmissionDefaults::default());
    let causal = |level| Requested {
        read_concern: Some(level),
        after_index: 1,
        ..Requested::default()
    };
    for level in [ReadConcernLevel::Local, ReadConcernLevel::Linearizable] {
        let refused = admitter
            .check("RETURN 1", &causal(level))
            .expect_err("a causal read needs MAJORITY or SNAPSHOT");
        assert!(
            matches!(refused, DatabaseError::SettingsRefused(_)),
            "{refused:?}"
        );
    }

    let lossy_write = Requested {
        write_concern: Some(WriteConcern::w1()),
        ..causal(ReadConcernLevel::Majority)
    };
    let refused = admitter
        .check("CREATE (:Causal)", &lossy_write)
        .expect_err("a causal write must be majority-journaled");
    assert!(
        matches!(&refused, DatabaseError::SettingsRefused(m) if m.contains("writeConcern")),
        "{refused:?}"
    );
    // The same concern on a read is no concern of the causal check.
    admitter
        .check("RETURN 1", &lossy_write)
        .expect("a read under a weak write concern");

    let pinned = Requested {
        read_concern: Some(ReadConcernLevel::Majority),
        at_timestamp: 7,
        ..Requested::default()
    };
    let refused = admitter
        .check("RETURN 1", &pinned)
        .expect_err("a pinned timestamp needs SNAPSHOT");
    assert!(
        matches!(refused, DatabaseError::SettingsRefused(_)),
        "{refused:?}"
    );

    let refused = admitter
        .check("SET vector_consistency = 'exact'", &Requested::default())
        .expect_err("a session SET has no session here");
    assert!(
        matches!(refused, DatabaseError::SettingsRefused(_)),
        "{refused:?}"
    );
}

/// A statement the fence sends to the leader goes there once; with no leader
/// known it is refused naming none, the explicit no-leader answer, never a
/// role refusal a client would read as a misconfigured read; a statement
/// routed once is answered rather than routed again.
#[test]
fn a_role_refusal_names_the_leader_or_none() {
    let forward = fence_refusal(ReadFenceError::NotLeader, Some(2), false);
    assert!(matches!(forward, Ok(Admission::Forward(2))), "{forward:?}");

    let none = fence_refusal(ReadFenceError::NotLeader, None, false);
    assert!(
        matches!(none, Err(DatabaseError::NotLeader { leader_id: None })),
        "{none:?}"
    );
    let none = fence_refusal(ReadFenceError::LinearizableRequiresLeader, None, true);
    assert!(
        matches!(none, Err(DatabaseError::NotLeader { leader_id: None })),
        "{none:?}"
    );

    // Routed once already: answered with the refusal and the leader this
    // member knows, for the ingress to decide on, never routed on.
    let once = fence_refusal(ReadFenceError::NotLeader, Some(2), true);
    assert!(
        matches!(once, Err(DatabaseError::NotLeader { leader_id: Some(2) })),
        "{once:?}"
    );
    // A follower-only read on the leader is a role refusal whoever leads.
    let not_follower = fence_refusal(ReadFenceError::NotFollower, None, false);
    assert!(
        matches!(
            not_follower,
            Err(DatabaseError::ReadFence {
                kind: ReadFenceFailure::Role,
                ..
            })
        ),
        "{not_follower:?}"
    );
}

/// A standalone member runs every statement here unfenced, and a
/// transaction's commands too: there is no leader elsewhere.
#[tokio::test(flavor = "multi_thread")]
async fn a_standalone_member_runs_everything_here() {
    let (admitter, _dir) = standalone(AdmissionDefaults::default());
    assert!(!admitter.needs_fence());
    match admitter
        .admit("RETURN 1", &Requested::default(), false)
        .await
        .expect("admitted")
    {
        Admission::Run(admitted) => {
            assert_eq!(admitted.applied_index, 0);
            assert!(!admitted.served_by_leader);
        }
        Admission::Forward(leader) => panic!("a standalone member forwarded to {leader}"),
    }
    assert_eq!(admitter.transaction_leader().expect("no election"), None);
    assert!(matches!(
        admitter.router(2),
        Err(DatabaseError::PeerUnreachable { member: 2, .. })
    ));
}
