//! Log entries and segments in the current and the old shapes, for tests.

use std::path::Path;

use coordinode_core::txn::proposal::{Mutation, PartitionId, ProposalId, RaftProposal};
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_raft::storage::{CommittedLeaderId, Entry, Request};
use coordinode_storage::engine::config::SyncMethod;
use coordinode_storage::oplog::{OplogEntry, OplogOp, SegmentWriter};
use openraft::entry::RaftEntry as _;
use rmpv::Value;

/// A proposal writing one node at `commit_ts`.
pub(crate) fn proposal(id: u64, commit_ts: u64) -> RaftProposal {
    RaftProposal {
        id: ProposalId::from_raw(id),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: format!("node:1:{id}").into_bytes(),
            value: b"v".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(commit_ts),
        start_ts: Timestamp::from_raw(commit_ts - 1),
        bypass_rate_limiter: false,
    }
}

/// A current Normal entry at `index` carrying `request`.
pub(crate) fn current_entry(index: u64, request: Request) -> Entry {
    Entry::new_normal(
        openraft::LogId::new(
            CommittedLeaderId {
                term: 1,
                node_id: 1,
            },
            index,
        ),
        request,
    )
}

/// `entry` encoded as a build before the closed bound wrote it: the request
/// array without its second field.
pub(crate) fn old_bytes(entry: &Entry) -> Vec<u8> {
    let current = rmp_serde::to_vec(entry).expect("encode");
    let mut value = rmpv::decode::read_value(&mut &current[..]).expect("decode");
    let Value::Array(fields) = &mut value else {
        unreachable!("an entry encodes as an array")
    };
    let Some(Value::Map(payload)) = fields.get_mut(1) else {
        unreachable!("a Normal payload encodes as a map")
    };
    let [(_, Value::Array(request))] = payload.as_mut_slice() else {
        unreachable!("a request encodes as an array")
    };
    request.truncate(1);
    let mut out = Vec::new();
    rmpv::encode::write_value(&mut out, &value).expect("encode");
    out
}

/// A journal record at `index` whose envelope is `data`.
pub(crate) fn record(index: u64, data: Vec<u8>) -> OplogEntry {
    OplogEntry {
        ts: 1000 + index,
        term: 1,
        index,
        shard: 0,
        ops: vec![OplogOp::RaftEntry { data }],
        is_migration: false,
        pre_images: None,
    }
}

/// Write `records` as a segment at `path`, sealed or left open as a crash
/// leaves the active one.
pub(crate) fn write_segment(path: &Path, records: &[OplogEntry], sealed: bool) {
    std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
    let first = records.first().map_or(1, |r| r.index);
    let mut writer = SegmentWriter::create(path, 0, first, SyncMethod::Full).expect("create");
    for r in records {
        writer.append(r).expect("append");
    }
    if sealed {
        writer.seal().expect("seal");
    } else {
        writer.flush_and_sync().expect("sync");
    }
}
