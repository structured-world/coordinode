use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use coordinode_core::txn::proposal::{
    Mutation, PartitionId, ProposalIdGenerator, ProposalPipeline as _, RaftProposal,
};
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_raft::proposal::OwnedLocalProposalPipeline;
use coordinode_replicate::{ConsumerKind, SeqnoConsumerRegistry};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::oplog::entry::{OplogEntry, OplogOp};
use coordinode_storage::oplog::manager::OplogManager;
use tokio_stream::StreamExt as _;
use tonic::Request;

use super::ChangeEventServiceImpl;
use crate::proto::replication::cdc::change_stream_service_server::ChangeStreamService;
use crate::proto::replication::cdc::{
    CdcFilters as ProtoCdcFilters, ResumeToken as ProtoResumeToken, SubscribeRequest,
};
use crate::registry::{RegistryTuning, build_consumer_registry};

/// Open a fresh single-endpoint engine in a temp directory, mirroring the
/// registry-construction tests so the CDC service exercises the same wiring
/// production runs.
fn open_engine() -> (Arc<StorageEngine>, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path().to_string_lossy().as_ref(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let oracle = Arc::new(TimestampOracle::new());
    let engine = Arc::new(StorageEngine::open_with_oracle(&config, oracle).expect("open engine"));
    (engine, dir)
}

/// A CDC subscription registers an `oplog_events` consumer for its shard so the
/// oplog retention floor is held at the reader's position (ADR-028), and the
/// registration is released when the stream is dropped (client disconnect).
/// This is the per-consumer retention contract R137d wires; without it the
/// registry would only enforce the static time-travel window.
#[tokio::test]
async fn subscribe_registers_then_unregisters_cdc_consumer() {
    let (engine, _engine_dir) = open_engine();
    let pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> =
        Arc::new(OwnedLocalProposalPipeline::new(&engine));
    // Hold `_bg` for the test's lifetime: dropping it stops the background
    // service that flushes heartbeats and drives eviction.
    let (registry, _bg) = build_consumer_registry(engine, pipeline, RegistryTuning::default());

    let service = ChangeEventServiceImpl::new(
        0,
        Vec::new(),
        registry.clone(),
        super::DEFAULT_CONSUMER_TTL_MS,
        Arc::new(|| 0),
    );

    // No oplog exists, so the stream is empty (caught up immediately)
    // — registration happens synchronously inside `subscribe` regardless.
    let response = service
        .subscribe(Request::new(SubscribeRequest {
            resume_token: None,
            filters: None,
        }))
        .await
        .expect("subscribe");

    let consumers = registry.list_consumers();
    let cdc: Vec<_> = consumers
        .iter()
        .filter(|c| c.kind == ConsumerKind::OplogEvents && c.consumer_id.starts_with("cdc-"))
        .collect();
    assert_eq!(
        cdc.len(),
        1,
        "subscribe must register exactly one cdc oplog_events consumer, got {consumers:?}"
    );

    // Dropping the response drops the receiver stream; the tailing task notices
    // the closed channel on its next poll and unregisters.
    drop(response);

    // Poll interval is 100ms; give the task several cycles to observe the
    // disconnect and release the registration.
    let mut released = false;
    for _ in 0..40 {
        tokio::time::sleep(Duration::from_millis(50)).await;
        let still_present = registry
            .list_consumers()
            .iter()
            .any(|c| c.kind == ConsumerKind::OplogEvents && c.consumer_id.starts_with("cdc-"));
        if !still_present {
            released = true;
            break;
        }
    }
    assert!(
        released,
        "cdc consumer must be unregistered after the stream is dropped"
    );
}

/// A token that names no position of this stream is refused up front, and
/// no consumer is left registered for it: one for another shard, and one
/// whose parts overflow a log index.
#[tokio::test]
async fn subscribe_refuses_a_token_it_cannot_resume() {
    let (engine, _engine_dir) = open_engine();
    let pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> =
        Arc::new(OwnedLocalProposalPipeline::new(&engine));
    let (registry, _bg) = build_consumer_registry(engine, pipeline, RegistryTuning::default());
    let service = ChangeEventServiceImpl::new(
        0,
        Vec::new(),
        registry.clone(),
        super::DEFAULT_CONSUMER_TTL_MS,
        Arc::new(|| 0),
    );

    for (token, what) in [
        (
            ProtoResumeToken {
                shard_id: 7,
                segment_id: 0,
                entry_offset: 0,
            },
            "another shard",
        ),
        (
            ProtoResumeToken {
                shard_id: 0,
                segment_id: u64::MAX,
                entry_offset: 1,
            },
            "past the last log index",
        ),
    ] {
        let status = match service
            .subscribe(Request::new(SubscribeRequest {
                resume_token: Some(token),
                filters: None,
            }))
            .await
        {
            Ok(_) => panic!("a token for {what} was accepted"),
            Err(status) => status,
        };
        assert_eq!(status.code(), tonic::Code::InvalidArgument, "{what}");
    }
    assert!(
        registry
            .list_consumers()
            .iter()
            .all(|c| !c.consumer_id.starts_with("cdc-")),
        "a refused subscription holds no retention"
    );
}

/// An idle stream heartbeats once per poll, so a poll interval that reaches
/// the consumer TTL would let a connected reader expire; the tuning refuses it.
#[test]
fn a_poll_interval_must_be_shorter_than_the_consumer_ttl() {
    let at = |ms| super::CdcStreamTuning {
        poll_interval: Duration::from_millis(ms),
        ..super::CdcStreamTuning::default()
    };
    assert!(at(100).check(30_000).is_ok());
    assert!(at(29_999).check(30_000).is_ok());
    assert!(at(30_000).check(30_000).is_err());
    assert!(at(60_000).check(30_000).is_err());
    assert!(
        super::CdcStreamTuning::default()
            .check(super::DEFAULT_CONSUMER_TTL_MS)
            .is_ok()
    );
}

/// A reader that is connected but slower than the log keeps its registration:
/// while the stream waits for the reader to take more events it still
/// heartbeats, so the TTL reclaims only readers that are gone.
#[tokio::test]
async fn a_slow_reader_is_not_evicted() {
    let (engine, _engine_dir) = open_engine();
    let pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> =
        Arc::new(OwnedLocalProposalPipeline::new(&engine));
    let (registry, _bg) = build_consumer_registry(
        engine,
        pipeline,
        RegistryTuning {
            heartbeat_window_ms: Some(20),
            eviction_interval_ms: Some(50),
        },
    );
    let oplog_dir = tempfile::tempdir().expect("oplog dir");
    write_oplog(oplog_dir.path(), 300);

    let ttl_ms = 400;
    let service = ChangeEventServiceImpl::new(
        0,
        vec![oplog_dir.path().to_path_buf()],
        registry.clone(),
        ttl_ms,
        Arc::new(|| 300),
    )
    .with_tuning(super::CdcStreamTuning {
        poll_interval: Duration::from_millis(50),
        batch_size: std::num::NonZeroUsize::new(256).expect("nonzero"),
    });
    let mut stream = service
        .subscribe(Request::new(SubscribeRequest {
            resume_token: None,
            filters: None,
        }))
        .await
        .expect("subscribe")
        .into_inner();

    // The reader takes nothing for several TTLs; the stream fills its buffer
    // and waits.
    tokio::time::sleep(Duration::from_millis(4 * ttl_ms)).await;
    assert!(
        registry
            .list_consumers()
            .iter()
            .any(|c| c.consumer_id.starts_with("cdc-")),
        "a connected reader was evicted while the stream waited on it"
    );

    for expected in 0..300 {
        assert_eq!(
            next_index(&mut stream, Duration::from_secs(5)).await,
            Some(expected)
        );
    }
}

/// Write `count` node-write entries as oplog segments into `dir`.
fn write_oplog(dir: &std::path::Path, count: u64) {
    let mut mgr =
        OplogManager::open(dir, 0, 64 * 1024 * 1024, 50_000, 7 * 24 * 3600).expect("open oplog");
    for index in 0..count {
        mgr.append(&OplogEntry {
            ts: 1000 + index,
            term: 1,
            index,
            shard: 0,
            ops: vec![OplogOp::Insert {
                partition: 0,
                key: format!("node:{index}").into_bytes(),
                value: b"v".to_vec(),
            }],
            is_migration: false,
            pre_images: None,
        })
        .expect("append");
    }
    mgr.flush().expect("flush");
}

async fn next_index(
    stream: &mut <ChangeEventServiceImpl as ChangeStreamService>::SubscribeStream,
    within: Duration,
) -> Option<u64> {
    tokio::time::timeout(within, stream.next())
        .await
        .ok()
        .flatten()
        .map(|event| event.expect("event").log_index)
}

/// The Raft log holds entries that are not committed yet, which a later
/// leader may truncate and replace. A change stream sends an entry only once
/// this node has applied it.
#[tokio::test]
async fn stream_sends_only_applied_entries() {
    let (engine, _engine_dir) = open_engine();
    let pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> =
        Arc::new(OwnedLocalProposalPipeline::new(&engine));
    let (registry, _bg) = build_consumer_registry(engine, pipeline, RegistryTuning::default());
    let oplog_dir = tempfile::tempdir().expect("oplog dir");
    write_oplog(oplog_dir.path(), 5);

    let applied = Arc::new(AtomicU64::new(3));
    let frontier = Arc::clone(&applied);
    let service = ChangeEventServiceImpl::new(
        0,
        vec![oplog_dir.path().to_path_buf()],
        registry,
        super::DEFAULT_CONSUMER_TTL_MS,
        Arc::new(move || frontier.load(Ordering::Acquire)),
    );
    let mut stream = service
        .subscribe(Request::new(SubscribeRequest {
            resume_token: None,
            filters: None,
        }))
        .await
        .expect("subscribe")
        .into_inner();

    for expected in 0..3 {
        assert_eq!(
            next_index(&mut stream, Duration::from_secs(5)).await,
            Some(expected)
        );
    }
    assert_eq!(
        next_index(&mut stream, Duration::from_millis(400)).await,
        None,
        "entries 3 and 4 are not applied yet"
    );

    applied.store(5, Ordering::Release);
    assert_eq!(
        next_index(&mut stream, Duration::from_secs(5)).await,
        Some(3)
    );
    assert_eq!(
        next_index(&mut stream, Duration::from_secs(5)).await,
        Some(4)
    );
}

/// A stream whose filters match nothing still reads the log, and its
/// retention checkpoint follows: otherwise it would hold the oplog forever.
#[tokio::test]
async fn filtered_stream_advances_its_checkpoint() {
    let (engine, _engine_dir) = open_engine();
    let pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> =
        Arc::new(OwnedLocalProposalPipeline::new(&engine));
    let (registry, _bg) = build_consumer_registry(engine, pipeline, RegistryTuning::default());
    let oplog_dir = tempfile::tempdir().expect("oplog dir");
    write_oplog(oplog_dir.path(), 5);

    let service = ChangeEventServiceImpl::new(
        0,
        vec![oplog_dir.path().to_path_buf()],
        registry.clone(),
        super::DEFAULT_CONSUMER_TTL_MS,
        Arc::new(|| 5),
    );
    let mut stream = service
        .subscribe(Request::new(SubscribeRequest {
            resume_token: None,
            filters: Some(ProtoCdcFilters {
                edge_types: vec!["NEVER_WRITTEN".to_string()],
                is_migration: None,
            }),
        }))
        .await
        .expect("subscribe")
        .into_inner();

    let mut checkpoint = None;
    for _ in 0..40 {
        tokio::time::sleep(Duration::from_millis(50)).await;
        checkpoint = registry
            .list_consumers()
            .iter()
            .find(|c| c.kind == ConsumerKind::OplogEvents && c.consumer_id.starts_with("cdc-"))
            .map(|c| c.checkpoint_seqno);
        if checkpoint == Some(4) {
            break;
        }
    }
    assert_eq!(
        checkpoint,
        Some(4),
        "the checkpoint covers every entry read"
    );
    assert_eq!(
        next_index(&mut stream, Duration::from_millis(200)).await,
        None,
        "no entry matched the filter"
    );
}

/// The service the server builds for a Raft node streams what the node
/// commits: it reads the directories the log is written to, up to the node's
/// applied entries, and each event carries the proposal's commit timestamp
/// and the leader's term.
#[tokio::test(flavor = "multi_thread")]
async fn stream_follows_a_raft_node() {
    let (engine, _engine_dir) = open_engine();
    let node = Arc::new(
        coordinode_raft::cluster::RaftNode::single_node(Arc::clone(&engine))
            .await
            .expect("bootstrap"),
    );
    let (registry_engine, _registry_dir) = open_engine();
    let pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> =
        Arc::new(OwnedLocalProposalPipeline::new(&registry_engine));
    let (registry, _bg) =
        build_consumer_registry(registry_engine, pipeline, RegistryTuning::default());

    let service = ChangeEventServiceImpl::for_raft_node(
        &engine,
        Arc::clone(&node),
        registry,
        super::DEFAULT_CONSUMER_TTL_MS,
    )
    .expect("service");
    let mut stream = service
        .subscribe(Request::new(SubscribeRequest {
            resume_token: None,
            filters: None,
        }))
        .await
        .expect("subscribe")
        .into_inner();

    let raft_pipeline = node.pipeline();
    let ids = ProposalIdGenerator::new();
    for i in 1..=3u64 {
        raft_pipeline
            .propose_and_wait(&RaftProposal {
                id: ids.next(),
                mutations: vec![Mutation::Put {
                    partition: PartitionId::Node,
                    key: format!("cdc-{i}").into_bytes(),
                    value: b"v".to_vec(),
                }],
                commit_ts: Timestamp::from_raw(5000 + i),
                start_ts: Timestamp::from_raw(4999 + i),
                bypass_rate_limiter: false,
            })
            .expect("propose");
    }

    let mut seen = Vec::new();
    while seen.len() < 3 {
        let event = tokio::time::timeout(Duration::from_secs(10), stream.next())
            .await
            .expect("an event for every committed proposal")
            .expect("stream open")
            .expect("event");
        if event.ts > 5000 {
            assert!(event.term >= 1, "the leader's term, got {}", event.term);
            seen.push(event.ts);
        }
    }
    assert_eq!(seen, vec![5001, 5002, 5003]);

    drop(stream);
    node.shutdown().await.expect("shutdown");
}
