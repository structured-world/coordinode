use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, Tier};
use coordinode_storage::engine::footprint::{Admit, Footprints};
use openraft::raft::AppendEntriesResponse;

use super::*;

const UNIT: u64 = 4096;

fn footprints(dir: &std::path::Path) -> Footprints {
    Footprints::open(&[
        EndpointConfig::new("ep", dir, Media::Ssd, Durability::Durable, Tier::Warm)
            .with_hard_limit_bytes(100 * UNIT)
            .with_recovery_reserve_bytes(20 * UNIT),
    ])
    .unwrap()
}

fn admitted(f: &Footprints) -> u64 {
    f.get("ep").unwrap().admitted()
}

/// A request the consensus refused outright (a higher vote, a log that does
/// not match) took nothing into the log: its own-work room comes back at
/// once, however often the leader sends it again, without waiting for the
/// log to be applied past indexes it may never reach. Before, every refused
/// request held its room until the log was applied past its last entry.
#[test]
fn a_refused_append_gives_its_own_work_room_back() {
    let dir = tempfile::tempdir().unwrap();
    let f = footprints(dir.path());
    for _ in 0..10 {
        let room = OwnWorkRoom::new(&f, 1_000, f.admit(4 * UNIT, Admit::Recovery).unwrap());
        let resp = AppendEntriesResponse::<TypeConfig>::Conflict;
        let matched = OwnWorkRoom::matched(&resp, room.last);
        room.settle(matched);
        assert_eq!(admitted(&f), 0, "a refused request holds nothing");
    }
}

/// A request whose waiter went away may still be appended by the core: its
/// room is held until the log is applied past it. One the log took is held
/// until applied through what it took.
#[test]
fn an_unanswered_or_taken_append_keeps_its_room_until_applied() {
    let dir = tempfile::tempdir().unwrap();
    let f = footprints(dir.path());
    let unanswered = OwnWorkRoom::new(&f, 5, f.admit(4 * UNIT, Admit::Recovery).unwrap());
    drop(unanswered);
    assert_eq!(admitted(&f), 4 * UNIT, "held: the core may append it");

    let taken = OwnWorkRoom::new(&f, 9, f.admit(2 * UNIT, Admit::Recovery).unwrap());
    let matched = OwnWorkRoom::matched(&AppendEntriesResponse::<TypeConfig>::Success, taken.last);
    taken.settle(matched);
    assert_eq!(admitted(&f), 6 * UNIT);

    f.applied(5, 0, 0);
    assert_eq!(admitted(&f), 2 * UNIT, "the unanswered one, applied");
    f.applied(9, 0, 0);
    assert_eq!(admitted(&f), 0, "the taken one, applied");
}

/// A client entry of `value_bytes` at `index`, by the leader of term 1.
fn client_entry(index: u64, value_bytes: usize) -> crate::storage::Entry {
    use coordinode_core::txn::proposal::{Mutation, PartitionId, ProposalId, RaftProposal};
    use coordinode_core::txn::timestamp::Timestamp;
    use openraft::entry::RaftEntry;
    crate::storage::Entry::new_normal(
        crate::storage::LogId::new(
            crate::storage::CommittedLeaderId {
                term: 1,
                node_id: 1,
            },
            index,
        ),
        crate::storage::Request::single(RaftProposal {
            id: ProposalId::from_raw(index),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: format!("node:1:{index}").into_bytes(),
                value: vec![7u8; value_bytes],
            }],
            commit_ts: Timestamp::from_raw(0),
            start_ts: Timestamp::from_raw(0),
            bypass_rate_limiter: false,
        }),
    )
}

/// An append from the leader of term 1 carrying `entries`, all committed.
fn committed_append(
    entries: Vec<crate::storage::Entry>,
) -> openraft::raft::AppendEntriesRequest<TypeConfig> {
    let last = entries.last().map(|entry| entry.log_id);
    openraft::raft::AppendEntriesRequest {
        vote: crate::storage::Vote::new_committed(1, 1),
        prev_log_id: None,
        entries,
        leader_commit: last,
    }
}

/// A member below its free-space reserve takes no committed client entries,
/// whether its endpoint has no hard limit or its grant would pay for them:
/// they are admitted as the store's own work, out of the upper half of the
/// reserve. Before, the free-space check ran only when the grant could not
/// pay under a hard limit, so a paused follower took every committed entry.
#[test]
fn committed_client_entries_wait_while_the_disk_is_below_its_reserve() {
    use coordinode_storage::engine::config::StorageConfig;
    use coordinode_storage::engine::core::StorageEngine;
    let entry = client_entry(1, 1024);
    let bytes = super::super::grants::entry_room(&entry).1;
    for limited in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let mut endpoint = EndpointConfig::new(
            "default",
            dir.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        );
        if limited {
            endpoint = endpoint.with_hard_limit_bytes(256 << 20);
        }
        let mut config = StorageConfig::with_endpoints(vec![endpoint]);
        if limited {
            config.client_grant_bytes = bytes * 4;
        }
        let engine = StorageEngine::open(&config).unwrap();
        let space = engine.space();
        space.footprints().top_up_grant();
        space.set_reserve(u64::MAX, u64::MAX);
        let append = committed_append(vec![entry.clone()]);
        assert!(
            reserve_entries(space, &append).is_err(),
            "limited {limited}: a paused member takes no committed client entry"
        );
        space.set_reserve(0, 0);
        let _own = reserve_entries(space, &append)
            .unwrap_or_else(|e| panic!("limited {limited}: taken once space is freed: {e}"));
    }
}

/// A committed entry the grant cannot take falls to the recovery reserve
/// for its own room only: an entry of the same request already paid takes
/// nothing again. At a reserve holding the new entry and less than both,
/// the request is taken. Before, the fallback asked the reserve for every
/// committed entry of the request, paid or not, and was refused.
#[test]
fn a_committed_entry_already_paid_takes_no_reserve_again() {
    use coordinode_storage::engine::config::StorageConfig;
    use coordinode_storage::engine::core::StorageEngine;
    let dir = tempfile::tempdir().unwrap();
    let first = client_entry(1, 64 << 10);
    let second = client_entry(2, 64 << 10);
    let bytes = super::super::grants::entry_room(&first).1;
    let mut config = StorageConfig::with_endpoints(vec![
        EndpointConfig::new(
            "default",
            dir.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )
        .with_hard_limit_bytes(256 << 20),
    ]);
    // A batch of 1.2 entries: the first entry spends all but 0.2 of it.
    config.client_grant_bytes = bytes * 6 / 10;
    let engine = StorageEngine::open(&config).unwrap();
    let space = engine.space();
    let footprints = space.footprints();
    footprints.top_up_grant();
    let _own =
        reserve_entries(space, &committed_append(vec![first.clone()])).expect("from the grant");
    footprints.granted_appended(1, 1, bytes);
    assert!(
        footprints.granted_room() < bytes,
        "the grant cannot take the second"
    );

    // The reserve holds the second entry and less than both; no client room.
    let footprint = footprints.get("default").unwrap();
    let limit = footprint.committed() + bytes + bytes / 2;
    footprint.set_limit(limit, limit);
    let both = committed_append(vec![first, second]);
    let _own = reserve_entries(space, &both).expect("the reserve takes the unpaid entry only");
    let held = footprint.admitted();

    // Sent again, and again once the first is applied: nothing more held.
    let _own = reserve_entries(space, &both).expect("a repeat takes nothing again");
    footprints.applied(1, 1, bytes);
    let after_apply = footprint.admitted();
    let _own = reserve_entries(space, &both).expect("an applied entry takes nothing");
    assert!(after_apply <= held);
    assert_eq!(footprint.admitted(), after_apply, "no second payment");
}
