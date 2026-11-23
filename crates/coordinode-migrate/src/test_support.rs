//! Journal records and segments as this build and as 0.10.1 wrote them, for
//! tests.

use std::path::Path;

use coordinode_core::txn::frame::encode_unit;
use coordinode_core::txn::proposal::{Mutation, PartitionId, ProposalId, RaftProposal};
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_raft::storage::{CommittedLeaderId, Entry, Request};
use coordinode_storage::engine::config::SyncMethod;
use coordinode_storage::oplog::{OplogEntry, OplogOp, SegmentWriter};
use openraft::entry::RaftEntry as _;

/// The namespace the store's data is in.
pub(crate) const NAMESPACE: coordinode_core::namespace::NamespaceId =
    coordinode_core::namespace::NamespaceId::from_raw(1);

/// A proposal writing node `id` at `commit_ts`, in the store's namespace.
pub(crate) fn proposal(id: u64, commit_ts: u64) -> RaftProposal {
    RaftProposal {
        id: ProposalId::from_raw(id),
        mutations: coordinode_core::txn::proposal::in_namespace(
            NAMESPACE,
            [Mutation::Put {
                partition: PartitionId::Node,
                key: format!("node:1:{id}").into_bytes(),
                value: b"v".to_vec(),
            }],
        ),
        commit_ts: Timestamp::from_raw(commit_ts),
        start_ts: Timestamp::from_raw(commit_ts - 1),
        bypass_rate_limiter: false,
    }
}

/// The unit frame of [`proposal`] as this build writes it.
pub(crate) fn named_frame(id: u64, commit_ts: u64) -> Vec<u8> {
    encode_unit(
        &proposal(id, commit_ts).mutations,
        Timestamp::from_raw(commit_ts),
    )
    .expect("encode")
}

/// `frame` as 0.10.1 wrote it: the same operations without the namespace
/// marker in front of them.
pub(crate) fn unnamed(frame: &[u8]) -> Vec<u8> {
    let body = &frame[..frame.len() - 4];
    let mut at = 1;
    let varint = |at: &mut usize| {
        let mut value = 0u64;
        let mut shift = 0;
        loop {
            let byte = body[*at];
            *at += 1;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return value;
            }
            shift += 7;
        }
    };
    for _ in 0..3 {
        varint(&mut at);
    }
    at += 1;
    let count_at = at;
    let count = varint(&mut at);
    // The marker: its tag and the namespace's one-byte number.
    assert_eq!(&body[at..at + 2], &[4 | (1 << 3), 1], "a named frame");
    let mut old = body[..count_at].to_vec();
    assert!(
        count - 1 < 0x80,
        "a test unit counts its operations in one byte"
    );
    old.push(u8::try_from(count - 1).expect("one byte"));
    old.extend_from_slice(&body[at + 2..]);
    let crc = crc32c::crc32c(&old);
    old.extend_from_slice(&crc.to_le_bytes());
    old
}

/// A Normal entry at `index` carrying `request`.
pub(crate) fn entry(index: u64, request: Request) -> Entry {
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

/// A record of the store's own journal at `index`: the unit `frame` alone.
pub(crate) fn journal_record(index: u64, frame: Vec<u8>) -> OplogEntry {
    OplogEntry {
        ts: 1000 + index,
        term: 1,
        index,
        shard: 0,
        ops: vec![OplogOp::Unit { frame }],
        is_migration: false,
        pre_images: None,
    }
}

/// A record of a group's Raft log at `index`: the entry's envelope, then the
/// unit `frame`.
pub(crate) fn raft_record(index: u64, frame: Vec<u8>) -> OplogEntry {
    let envelope = rmp_serde::to_vec(&entry(index, Request::closing(0))).expect("encode");
    OplogEntry {
        ts: 1000 + index,
        term: 1,
        index,
        shard: 0,
        ops: vec![
            OplogOp::RaftEntry { data: envelope },
            OplogOp::Unit { frame },
        ],
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
