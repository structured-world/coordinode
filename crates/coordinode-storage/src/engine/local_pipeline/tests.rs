use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use crate::engine::partition::Partition;
use coordinode_core::txn::proposal::{Mutation, PartitionId, ProposalIdGenerator};
use coordinode_core::txn::timestamp::Timestamp;
use std::time::Duration;
use tempfile::TempDir;

fn setup() -> (StorageEngine, TempDir) {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("open");
    (engine, dir)
}

#[test]
fn local_pipeline_put_and_read() {
    let (engine, _dir) = setup();
    let pipeline = LocalProposalPipeline::new(&engine);
    let id_gen = ProposalIdGenerator::new();

    let proposal = RaftProposal {
        id: id_gen.next(),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: b"node:1:1".to_vec(),
            value: b"test-data".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(100),
        start_ts: Timestamp::from_raw(99),
        bypass_rate_limiter: false,
    };

    pipeline
        .propose_and_wait(&proposal)
        .expect("propose should succeed");

    // Verify via direct engine read (plain keys, no MVCC encoding)
    let result = engine.get(Partition::Node, b"node:1:1").expect("read");
    assert_eq!(result.as_deref(), Some(b"test-data".as_slice()));
}

#[test]
fn local_pipeline_delete() {
    let (engine, _dir) = setup();
    let pipeline = LocalProposalPipeline::new(&engine);
    let id_gen = ProposalIdGenerator::new();

    // First write a value
    let put = RaftProposal {
        id: id_gen.next(),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: b"node:1:2".to_vec(),
            value: b"data".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(100),
        start_ts: Timestamp::from_raw(99),
        bypass_rate_limiter: false,
    };
    pipeline.propose_and_wait(&put).expect("put");

    // Snapshot after put — captures the value before deletion
    let snap_after_put = engine.snapshot();

    // Then delete
    let del = RaftProposal {
        id: id_gen.next(),
        mutations: vec![Mutation::Delete {
            partition: PartitionId::Node,
            key: b"node:1:2".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(200),
        start_ts: Timestamp::from_raw(199),
        bypass_rate_limiter: false,
    };
    pipeline.propose_and_wait(&del).expect("delete");

    // Verify: snapshot before delete still sees the value
    let before_del = engine
        .snapshot_get(&snap_after_put, Partition::Node, b"node:1:2")
        .expect("snapshot read");
    assert_eq!(before_del.as_deref(), Some(b"data".as_slice()));

    // Verify: after delete, current value is gone (LSM tombstone)
    let after_del = engine.get(Partition::Node, b"node:1:2").expect("read");
    assert!(after_del.is_none());
}

#[test]
fn local_pipeline_multi_mutation_proposal() {
    let (engine, _dir) = setup();
    let pipeline = LocalProposalPipeline::new(&engine);
    let id_gen = ProposalIdGenerator::new();

    let proposal = RaftProposal {
        id: id_gen.next(),
        mutations: vec![
            Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:1:10".to_vec(),
                value: b"alice".to_vec(),
            },
            Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:1:11".to_vec(),
                value: b"bob".to_vec(),
            },
            Mutation::Put {
                partition: PartitionId::Adj,
                key: b"adj:KNOWS:out:10".to_vec(),
                value: b"posting-list".to_vec(),
            },
            Mutation::Put {
                partition: PartitionId::EdgeProp,
                key: b"edgeprop:KNOWS:10:11".to_vec(),
                value: b"since:2024".to_vec(),
            },
        ],
        commit_ts: Timestamp::from_raw(500),
        start_ts: Timestamp::from_raw(499),
        bypass_rate_limiter: false,
    };

    pipeline.propose_and_wait(&proposal).expect("propose");

    // Verify all four writes via direct engine reads
    assert_eq!(
        engine
            .get(Partition::Node, b"node:1:10")
            .expect("r")
            .as_deref(),
        Some(b"alice".as_slice())
    );
    assert_eq!(
        engine
            .get(Partition::Node, b"node:1:11")
            .expect("r")
            .as_deref(),
        Some(b"bob".as_slice())
    );
    assert_eq!(
        engine
            .get(Partition::Adj, b"adj:KNOWS:out:10")
            .expect("r")
            .as_deref(),
        Some(b"posting-list".as_slice())
    );
    assert_eq!(
        engine
            .get(Partition::EdgeProp, b"edgeprop:KNOWS:10:11")
            .expect("r")
            .as_deref(),
        Some(b"since:2024".as_slice())
    );
}

#[test]
fn local_pipeline_empty_proposal() {
    let (engine, _dir) = setup();
    let pipeline = LocalProposalPipeline::new(&engine);
    let id_gen = ProposalIdGenerator::new();

    let proposal = RaftProposal {
        id: id_gen.next(),
        mutations: vec![],
        commit_ts: Timestamp::from_raw(100),
        start_ts: Timestamp::from_raw(99),
        bypass_rate_limiter: false,
    };

    // Empty proposal should succeed (no-op)
    pipeline.propose_and_wait(&proposal).expect("empty ok");
}

/// LocalProposalPipeline: propose_with_timeout delegates to propose_and_wait
/// (ignores timeout). Proposals complete in µs, timeout irrelevant.
#[test]
fn local_pipeline_propose_with_timeout_succeeds() {
    let (engine, _dir) = setup();
    let pipeline = LocalProposalPipeline::new(&engine);
    let id_gen = ProposalIdGenerator::new();

    let proposal = RaftProposal {
        id: id_gen.next(),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: b"node:1:timeout-local".to_vec(),
            value: b"value".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(100),
        start_ts: Timestamp::from_raw(99),
        bypass_rate_limiter: false,
    };

    // Even with a very short timeout, local pipeline ignores it
    pipeline
        .propose_with_timeout(&proposal, Duration::from_nanos(1))
        .expect("local propose_with_timeout should succeed (timeout ignored)");

    let val = engine
        .get(Partition::Node, b"node:1:timeout-local")
        .expect("read");
    assert_eq!(val.as_deref(), Some(b"value".as_slice()));
}

/// The local (embedded) pipeline has no Raft log, so it reports no
/// committed index — `None`, never a fabricated zero.
#[test]
fn local_pipeline_has_no_committed_index() {
    let (engine, _dir) = setup();
    let pipeline = LocalProposalPipeline::new(&engine);
    let id_gen = ProposalIdGenerator::new();

    let outcome = pipeline
        .propose_and_wait(&RaftProposal {
            id: id_gen.next(),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:1:local-idx".to_vec(),
                value: b"v".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(100),
            start_ts: Timestamp::from_raw(99),
            bypass_rate_limiter: false,
        })
        .expect("propose");
    assert_eq!(outcome.applied_index, None);
}
