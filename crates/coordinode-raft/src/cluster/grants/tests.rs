use std::collections::BTreeSet;

use coordinode_core::txn::proposal::ProposalId;

use super::*;

fn grant(epoch: u64, seq: u64, last_index: u64, available: u64) -> Option<ClientGrant> {
    Some(ClientGrant {
        epoch,
        last_index,
        available,
        seq,
    })
}

fn voters(ids: &[u64]) -> Vec<BTreeSet<u64>> {
    vec![ids.iter().copied().collect()]
}

fn id(n: u64) -> ProposalId {
    ProposalId::from_raw(n)
}

/// A client proposal is admitted only while a majority, the leader
/// included, can pay for it with everything already in flight to each
/// member: admitted hand-overs and appended entries past its reported index.
#[test]
fn a_client_entry_needs_a_majority_that_can_pay_for_it() {
    let grants = LeaderGrants::default();
    let set = voters(&[1, 2, 3]);
    assert_eq!(
        grants.admit(1, 1, true, &set, id(1), 10),
        Err(GrantRefusal::NoRoom),
        "no member reported yet"
    );
    grants.report(1, 2, grant(7, 1, 5, 25));
    grants.admit(1, 1, true, &set, id(1), 10).unwrap();
    grants.admit(1, 1, true, &set, id(2), 10).unwrap();
    assert_eq!(
        grants.admit(1, 1, true, &set, id(3), 10),
        Err(GrantRefusal::NoRoom),
        "20 admitted of 25"
    );
    grants.appended(id(1), 6);
    grants.appended(id(2), 7);
    assert_eq!(
        grants.admit(1, 1, true, &set, id(3), 10),
        Err(GrantRefusal::NoRoom),
        "in the log past index 5, still in flight to member 2"
    );
    grants.report(1, 2, grant(7, 2, 7, 25));
    grants.admit(1, 1, true, &set, id(3), 10).unwrap();
}

/// A reply that arrives after a newer one changes nothing: the room it
/// reported was spent since. Without the order, the late reply gave the
/// spent room back.
#[test]
fn a_late_report_gives_nothing_back() {
    let grants = LeaderGrants::default();
    let set = voters(&[1, 2, 3]);
    grants.report(1, 2, grant(1, 1, 0, 10));
    grants.admit(1, 1, true, &set, id(1), 10).unwrap();
    grants.appended(id(1), 1);
    grants.report(1, 2, grant(1, 2, 1, 0));
    assert_eq!(
        grants.admit(1, 1, true, &set, id(2), 10),
        Err(GrantRefusal::NoRoom)
    );
    // The reply to an earlier append arrives now.
    grants.report(1, 2, grant(1, 1, 0, 10));
    assert_eq!(
        grants.admit(1, 1, true, &set, id(2), 10),
        Err(GrantRefusal::NoRoom),
        "a late reply reopened room already spent"
    );
}

/// A report of an earlier process of the member changes nothing once one
/// of a later process was taken.
#[test]
fn an_earlier_process_does_not_replace_a_later_one() {
    let grants = LeaderGrants::default();
    let set = voters(&[1, 2, 3]);
    grants.report(1, 2, grant(2, 1, 0, 0));
    grants.report(1, 2, grant(1, 9, 0, 10));
    assert_eq!(
        grants.admit(1, 1, true, &set, id(1), 10),
        Err(GrantRefusal::NoRoom),
        "a report of the earlier process replaced the later one"
    );
}

/// A report of an earlier term changes nothing of the current leadership:
/// it neither resets what is in flight nor counts as room.
#[test]
fn a_report_of_an_earlier_term_changes_nothing() {
    let grants = LeaderGrants::default();
    let set = voters(&[1, 2, 3]);
    grants.report(2, 2, grant(1, 1, 0, 10));
    grants.admit(2, 1, true, &set, id(1), 10).unwrap();
    grants.report(1, 2, grant(1, 2, 0, 10));
    grants.report(2, 2, grant(1, 3, 0, 10));
    assert_eq!(
        grants.admit(2, 1, true, &set, id(2), 10),
        Err(GrantRefusal::NoRoom),
        "the earlier term's report reset what is in flight"
    );
    assert_eq!(
        grants.admit(1, 1, true, &set, id(3), 1),
        Err(GrantRefusal::NoRoom),
        "a leadership no longer held admits nothing"
    );
}

/// A report never moves the view to another leadership: one of a newer term
/// leaves the current term's admissions and what is in flight untouched,
/// and is kept for that term, counting once this member admits under it.
/// Before, a report of term 3 started the view over while the leader still
/// led term 2, and term 2 admitted nothing more.
#[test]
fn a_report_cannot_advance_the_leaders_term() {
    let grants = LeaderGrants::default();
    let set = voters(&[1, 2, 3]);
    grants.report(2, 2, grant(1, 1, 0, 10));
    grants.admit(2, 1, true, &set, id(1), 4).unwrap();
    grants.report(3, 3, grant(1, 1, 0, 100));
    assert_eq!(grants.view.lock().term, 2, "the report moved the term");
    grants.admit(2, 1, true, &set, id(2), 4).unwrap();
    assert_eq!(
        grants.admit(2, 1, true, &set, id(3), 4),
        Err(GrantRefusal::NoRoom),
        "term 2 still counts what it admitted"
    );
    // Leading term 3 now, member 3's report for it counts.
    grants.admit(3, 1, true, &set, id(4), 50).unwrap();
}

/// A hand-over proved not made (refused before it reached the core) gives
/// its room back at once: consecutive refusals in one term leave a normal
/// write admitted. Before, each refused attempt stayed counted until the
/// leadership ended.
#[test]
fn a_withdrawn_hand_over_is_not_in_flight() {
    let grants = LeaderGrants::default();
    let set = voters(&[1, 2, 3]);
    grants.report(1, 2, grant(1, 1, 0, 10));
    for n in 0..5 {
        grants.admit(1, 1, true, &set, id(n), 6).unwrap();
        grants.withdraw(1, id(n));
    }
    grants.admit(1, 1, true, &set, id(9), 10).unwrap();
    // A withdrawal from an ended leadership touches nothing.
    grants.withdraw(0, id(9));
    assert_eq!(grants.view.lock().pending_total, 10);
}

/// Every hand-over of one proposal counts until the log takes it: a retry
/// after a timed-out wait may be appended as well as the first.
#[test]
fn every_hand_over_counts_until_the_log_takes_it() {
    let grants = LeaderGrants::default();
    let set = voters(&[1, 2, 3]);
    grants.report(1, 2, grant(1, 1, 0, 10));
    grants.admit(1, 1, true, &set, id(1), 4).unwrap();
    grants.admit(1, 1, true, &set, id(1), 4).unwrap();
    assert_eq!(
        grants.admit(1, 1, true, &set, id(2), 5),
        Err(GrantRefusal::NoRoom),
        "the retry replaced the first hand-over's room"
    );
    grants.appended(id(1), 1);
    grants.appended(id(1), 2);
    assert_eq!(
        grants.admit(1, 1, true, &set, id(2), 5),
        Err(GrantRefusal::NoRoom)
    );
    grants.report(1, 2, grant(1, 2, 2, 10));
    grants.admit(1, 1, true, &set, id(2), 5).unwrap();
}

/// Joint configuration: each voter set needs its own paying majority.
#[test]
fn every_voter_set_of_a_joint_configuration_must_pay() {
    let grants = LeaderGrants::default();
    let joint = vec![
        [1, 2, 3].into_iter().collect::<BTreeSet<u64>>(),
        [1, 4, 5].into_iter().collect(),
    ];
    grants.report(1, 2, grant(1, 1, 0, 100));
    assert_eq!(
        grants.admit(1, 1, true, &joint, id(1), 10),
        Err(GrantRefusal::NoRoom),
        "the new set has no paying majority"
    );
    grants.report(1, 4, None);
    grants.admit(1, 1, true, &joint, id(1), 10).unwrap();
}

/// A leader admits no client entry before its term's first entry is
/// applied, and a newer term starts the view over.
#[test]
fn a_new_leadership_starts_empty_and_waits_for_its_term() {
    let grants = LeaderGrants::default();
    let set = voters(&[1, 2, 3]);
    grants.report(1, 2, grant(1, 1, 0, 100));
    assert_eq!(
        grants.admit(2, 1, false, &set, id(1), 10),
        Err(GrantRefusal::NotReady)
    );
    assert_eq!(
        grants.admit(2, 1, true, &set, id(1), 10),
        Err(GrantRefusal::NoRoom),
        "term 1's report does not speak for term 2"
    );
    grants.report(2, 3, None);
    grants.admit(2, 1, true, &set, id(1), 10).unwrap();
}

/// A leader alone in its group keeps no view.
#[test]
fn a_group_of_one_keeps_nothing() {
    let grants = LeaderGrants::default();
    let set = voters(&[1]);
    for n in 0..100 {
        grants.admit(1, 1, true, &set, id(n), 10).unwrap();
        grants.appended(id(n), n + 1);
    }
    let view = grants.view.lock();
    assert!(view.pending.is_empty() && view.cum.is_empty());
}

/// A voter that lags behind a healthy majority, or never reports, holds
/// back no more than the largest grant: the entries kept stay bounded while
/// the majority commits on, and the lagging voter covers nothing until it
/// catches up. A member that left holds nothing back.
#[test]
fn a_lagging_or_departed_voter_holds_back_a_bounded_view() {
    let grants = LeaderGrants::default();
    let set = voters(&[1, 2, 3]);
    grants.report(1, 3, grant(1, 1, 0, 100));
    let mut seq = 1;
    for n in 1..=10_000u64 {
        grants.admit(1, 1, true, &set, id(n), 10).unwrap();
        grants.appended(id(n), n);
        seq += 1;
        grants.report(1, 2, grant(1, seq, n, 100));
    }
    // The largest grant (100) holds 10 entries of 10: those, the one at the
    // edge, and one appended since the last admission compacted the view.
    assert!(
        grants.view.lock().cum.len() <= 12,
        "kept {} entries for a voter that lags",
        grants.view.lock().cum.len()
    );
    assert!(
        !grants.view.lock().covers(3, 1),
        "the lagging voter covers nothing"
    );
    grants.report(1, 3, grant(1, 2, 10_000, 100));
    assert!(
        grants.view.lock().covers(3, 1),
        "caught up, it covers again"
    );
    // Member 3 leaves.
    grants
        .admit(1, 1, true, &voters(&[1, 2]), id(20_000), 1)
        .unwrap();
    assert!(!grants.view.lock().reports.contains_key(&3));
}
