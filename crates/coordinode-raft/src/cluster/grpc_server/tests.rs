use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, Tier};

use super::*;

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

/// An append from the leader of term 1 carrying `entries`, committed through
/// `commit` (none when `None`).
fn append(
    entries: Vec<crate::storage::Entry>,
    commit: Option<u64>,
) -> openraft::raft::AppendEntriesRequest<TypeConfig> {
    let leader_commit = commit.and_then(|index| {
        entries
            .iter()
            .find(|entry| entry.log_id.index == index)
            .map(|entry| entry.log_id)
    });
    openraft::raft::AppendEntriesRequest {
        vote: crate::storage::Vote::new_committed(1, 1),
        prev_log_id: None,
        entries,
        leader_commit,
    }
}

/// A store in `dir`, with a hard limit of `limit` bytes when given.
fn engine(
    dir: &std::path::Path,
    limit: Option<u64>,
) -> Arc<coordinode_storage::engine::core::StorageEngine> {
    use coordinode_storage::engine::config::StorageConfig;
    let mut endpoint =
        EndpointConfig::new("default", dir, Media::Hdd, Durability::Durable, Tier::Warm);
    if let Some(limit) = limit {
        endpoint = endpoint.with_hard_limit_bytes(limit);
    }
    Arc::new(
        coordinode_storage::engine::core::StorageEngine::open(&StorageConfig::with_endpoints(
            vec![endpoint],
        ))
        .unwrap(),
    )
}

/// A member below its free-space reserve takes no committed client entries,
/// with or without a hard limit: they are admitted as the store's own work,
/// out of the upper half of the reserve, and taken once space is freed.
#[test]
fn committed_client_entries_wait_while_the_disk_is_below_its_reserve() {
    let entry = client_entry(1, 1024);
    for limited in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let engine = engine(dir.path(), limited.then_some(256 << 20));
        let space = engine.space();
        space.set_reserve(u64::MAX, u64::MAX);
        let append = append(vec![entry.clone()], Some(1));
        assert!(
            reserve_entries(space, &append).is_err(),
            "limited {limited}: a paused member takes no committed client entry"
        );
        space.set_reserve(0, 0);
        let _rooms = reserve_entries(space, &append)
            .unwrap_or_else(|e| panic!("limited {limited}: taken once space is freed: {e}"));
    }
}

/// A request's rooms are held while it is handled and given back once it is
/// answered, whatever the answer: nothing waits for the log to be applied.
#[test]
fn a_requests_room_comes_back_when_it_is_answered() {
    let dir = tempfile::tempdir().unwrap();
    let engine = engine(dir.path(), Some(256 << 20));
    let space = engine.space();
    let footprint = Arc::clone(space.footprints().get("default").unwrap());
    let before = footprint.admitted();
    let rooms = reserve_entries(
        space,
        &append(vec![client_entry(1, 4096), client_entry(2, 4096)], Some(1)),
    )
    .expect("room for both");
    assert!(footprint.admitted() > before, "held while handled");
    drop(rooms);
    assert_eq!(footprint.admitted(), before, "given back once answered");
}

/// At the client ceiling a member refuses the client entries not committed
/// yet, which the leader sends again once room comes back, and still takes
/// the committed ones out of the recovery reserve: refusing acknowledged
/// history would only leave it behind.
#[test]
fn uncommitted_client_entries_meet_the_client_ceiling_and_committed_ones_do_not() {
    let dir = tempfile::tempdir().unwrap();
    let engine = engine(dir.path(), Some(256 << 20));
    let space = engine.space();
    let footprint = Arc::clone(space.footprints().get("default").unwrap());
    let entry = client_entry(1, 64 << 10);
    let (_, bytes) = entry_room(&entry);
    // No client room left; the reserve holds the entry and more.
    let limit = footprint.committed() + 4 * bytes;
    footprint.set_limit(limit, 4 * bytes);
    assert!(
        reserve_entries(space, &append(vec![entry.clone()], None)).is_err(),
        "an uncommitted client entry finds no client room"
    );
    let _rooms = reserve_entries(space, &append(vec![entry], Some(1)))
        .expect("a committed one is taken from the reserve");
}
