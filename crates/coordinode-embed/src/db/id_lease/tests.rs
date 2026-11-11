use super::*;

/// A statement's lease grant waiting behind another grant that holds the
/// reserver's lock ends with the statement: its deadline, or its cancel,
/// refuses it with that reason, and no lease is granted for it. Free again,
/// a grant goes through.
#[test]
fn a_grant_behind_a_held_lock_ends_with_its_statement() {
    use crate::db::fields::tests::ending_in;
    use coordinode_core::budget::{BudgetStop, CancelFlag};
    let dir = tempfile::tempdir().expect("tempdir");
    let db = crate::Database::open(dir.path()).expect("open");
    let reserver = LogLeaseReserver::new(
        Arc::clone(&db.engine),
        Arc::clone(&db.pipeline),
        Arc::new(ProposalIdGenerator::new()),
        Arc::clone(&db.oracle),
    );
    let before = reserver.ceiling().expect("ceiling");
    {
        let _held = reserver.granting.lock();
        assert_eq!(
            reserver.reserve(Some(&ending_in(200))),
            Err(IdLeaseError::Stopped(BudgetStop::Deadline))
        );
        let cancel = CancelFlag::new();
        let budget = QueryBudget::new(1 << 20).with_cancel(cancel.clone());
        let refused = std::thread::scope(|s| {
            s.spawn(|| {
                std::thread::sleep(std::time::Duration::from_millis(200));
                cancel.cancel();
            });
            reserver.reserve(Some(&budget))
        });
        assert_eq!(refused, Err(IdLeaseError::Stopped(BudgetStop::Cancelled)));
    }
    assert_eq!(
        reserver.ceiling().expect("ceiling"),
        before,
        "nothing was granted"
    );
    let lease = reserver
        .reserve(Some(&ending_in(5_000)))
        .expect("granted once the lock is free");
    assert_eq!(lease.base, before);
}

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
