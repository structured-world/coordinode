use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use coordinode_core::txn::proposal::{
    Mutation, PartitionId, ProposalIdGenerator, ProposalPipeline as _, RaftProposal,
};
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_raft::proposal::OwnedLocalProposalPipeline;
use coordinode_replicate::{
    ConsumerKind, RegistrationState, RegistryBackground, SeqnoConsumerRegistry,
    ShardConsumerRegistry, TerminalReason,
};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::oplog::entry::{OplogEntry, OplogOp};
use coordinode_storage::oplog::manager::OplogManager;
use tokio_stream::StreamExt as _;
use tonic::{Code, Request};
use tonic_types::StatusExt;

use super::{AppliedFrontier, AppliedSignal, ChangeEventServiceImpl, INCARNATION_METADATA};
use crate::proto::replication::cdc::change_stream_service_server::ChangeStreamService;
use crate::proto::replication::cdc::{
    AcknowledgeSubscriptionRequest, BoundedRetention, CancelSubscriptionRequest,
    CdcFilters as ProtoCdcFilters, ConsumerRetention, ResumeToken as ProtoResumeToken,
    StrictRetention, SubscribeRequest, consumer_retention,
};
use crate::registry::{NodeRetentionSource, RegistryTuning, build_consumer_registry};

/// Open a fresh single-endpoint engine in a temp directory.
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

/// A service over the log in `dirs`, applied up to `applied`, with the
/// registry the server builds over the same log.
struct Fixture {
    service: ChangeEventServiceImpl,
    registry: ShardConsumerRegistry,
    _bg: RegistryBackground,
    _engine_dir: tempfile::TempDir,
}

fn fixture(
    dirs: Vec<PathBuf>,
    applied: AppliedFrontier,
    signal: AppliedSignal,
    tuning: RegistryTuning,
) -> Fixture {
    let (engine, engine_dir) = open_engine();
    let pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> =
        Arc::new(OwnedLocalProposalPipeline::new(&engine));
    let source = Arc::new(NodeRetentionSource::new(
        Arc::clone(&engine),
        dirs.clone(),
        Arc::clone(&applied),
    ));
    let (registry, bg) = build_consumer_registry(engine, pipeline, source, tuning);
    let service = ChangeEventServiceImpl::new(0, dirs, registry.clone(), applied, signal);
    Fixture {
        service,
        registry,
        _bg: bg,
        _engine_dir: engine_dir,
    }
}

fn strict() -> Option<ConsumerRetention> {
    Some(ConsumerRetention {
        policy: Some(consumer_retention::Policy::Strict(StrictRetention {})),
    })
}

fn bounded(liveness_ms: Option<u64>) -> Option<ConsumerRetention> {
    Some(ConsumerRetention {
        policy: Some(consumer_retention::Policy::Bounded(BoundedRetention {
            max_progress_lag_ms: 3_600_000,
            max_retained_bytes: 1 << 40,
            liveness_timeout_ms: liveness_ms,
        })),
    })
}

/// A request registering `id` anew with `retention`.
fn register(id: &str, retention: Option<ConsumerRetention>) -> SubscribeRequest {
    SubscribeRequest {
        resume_token: None,
        filters: None,
        consumer_id: id.to_string(),
        incarnation: 0,
        retention,
    }
}

/// A request resuming incarnation `incarnation` of `id`.
fn resume(id: &str, incarnation: u64) -> SubscribeRequest {
    SubscribeRequest {
        incarnation,
        ..register(id, None)
    }
}

fn state_of(registry: &ShardConsumerRegistry, id: &str) -> Option<RegistrationState> {
    registry
        .list_consumers()
        .into_iter()
        .find(|c| c.consumer_id == id)
        .map(|c| c.state)
}

/// Write `count` node-write entries as oplog segments into `dir`, stamped
/// with the current wall clock so a progress-lag bound sees fresh entries.
fn write_oplog(dir: &std::path::Path, count: u64) {
    let mut mgr =
        OplogManager::open(dir, 0, 64 * 1024 * 1024, 50_000, 7 * 24 * 3600).expect("open oplog");
    let now_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock after the epoch")
        .as_micros() as u64;
    for index in 0..count {
        mgr.append(&OplogEntry {
            ts: now_us + index,
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

/// A stream registers its consumer with the chosen policy, reports the
/// incarnation in the response metadata, and leaving does not end the
/// registration: the consumer is durable, not the connection.
#[tokio::test(flavor = "multi_thread")]
async fn a_stream_registers_a_consumer_that_outlives_the_connection() {
    let oplog_dir = tempfile::tempdir().expect("oplog dir");
    let f = fixture(
        vec![oplog_dir.path().to_path_buf()],
        Arc::new(|| 0),
        None,
        RegistryTuning::default(),
    );
    let response = f
        .service
        .subscribe(Request::new(register("sink", strict())))
        .await
        .expect("subscribe");
    assert_eq!(
        response
            .metadata()
            .get(INCARNATION_METADATA)
            .and_then(|v| v.to_str().ok()),
        Some("1")
    );
    let snapshot = f
        .registry
        .list_consumers()
        .into_iter()
        .find(|c| c.consumer_id == "sink")
        .expect("registered");
    assert_eq!(snapshot.kind, ConsumerKind::OplogEvents);
    drop(response);

    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(
        state_of(&f.registry, "sink"),
        Some(RegistrationState::Live),
        "a disconnect does not end the registration"
    );
}

/// Requests a stream cannot serve are refused up front, and nothing is left
/// registered for them.
#[tokio::test(flavor = "multi_thread")]
async fn subscribe_refuses_requests_it_cannot_serve() {
    let oplog_dir = tempfile::tempdir().expect("oplog dir");
    let f = fixture(
        vec![oplog_dir.path().to_path_buf()],
        Arc::new(|| 0),
        None,
        RegistryTuning::default(),
    );
    let zero_bound = Some(ConsumerRetention {
        policy: Some(consumer_retention::Policy::Bounded(BoundedRetention {
            max_progress_lag_ms: 0,
            max_retained_bytes: 1,
            liveness_timeout_ms: None,
        })),
    });
    let cases = [
        (register("", strict()), "no consumer id"),
        (register("c", None), "no retention policy"),
        (
            register("c", Some(ConsumerRetention { policy: None })),
            "an empty policy",
        ),
        (register("c", zero_bound), "a zero bound"),
        (
            register("c", bounded(Some(10_000))),
            "a liveness timeout no longer than the heartbeat interval",
        ),
        (
            SubscribeRequest {
                resume_token: Some(ProtoResumeToken {
                    shard_id: 7,
                    segment_id: 0,
                    entry_offset: 0,
                }),
                ..register("c", strict())
            },
            "a token for another shard",
        ),
        (
            SubscribeRequest {
                resume_token: Some(ProtoResumeToken {
                    shard_id: 0,
                    segment_id: u64::MAX,
                    entry_offset: 1,
                }),
                ..register("c", strict())
            },
            "a token past the last log index",
        ),
    ];
    for (request, what) in cases {
        let status = match f.service.subscribe(Request::new(request)).await {
            Ok(_) => panic!("{what} was accepted"),
            Err(status) => status,
        };
        assert_eq!(status.code(), Code::InvalidArgument, "{what}: {status:?}");
    }
    assert!(
        f.registry.list_consumers().is_empty(),
        "a refused subscription registered something"
    );
}

/// Registering at a position the log no longer holds is refused with
/// RETENTION_LOST: protection cannot start over history already gone, and
/// reading on from the oldest retained entry would hide the gap.
#[tokio::test(flavor = "multi_thread")]
async fn registering_below_the_retained_log_is_refused_with_retention_lost() {
    // Entries 0-4 and 5-9 in two sealed segments, the first one purged.
    let oplog_dir = tempfile::tempdir().expect("oplog dir");
    let mut mgr = OplogManager::open(oplog_dir.path(), 0, 64 * 1024 * 1024, 50_000, 7 * 24 * 3600)
        .expect("open oplog");
    for index in 0..10u64 {
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
        if index == 4 {
            mgr.rotate().expect("seal");
        }
    }
    mgr.rotate().expect("seal");
    assert_eq!(mgr.purge_before(5).expect("purge"), 1);
    let f = fixture(
        vec![oplog_dir.path().to_path_buf()],
        Arc::new(|| 10),
        None,
        RegistryTuning::default(),
    );

    let status = match f
        .service
        .subscribe(Request::new(SubscribeRequest {
            resume_token: Some(ProtoResumeToken {
                shard_id: 0,
                segment_id: 0,
                entry_offset: 2,
            }),
            ..register("late", strict())
        }))
        .await
    {
        Ok(_) => panic!("a registration below the retained log was accepted"),
        Err(status) => status,
    };
    assert_eq!(status.code(), Code::FailedPrecondition);
    let details = status.get_error_details();
    let info = details.error_info().expect("ErrorInfo");
    assert_eq!(info.reason, "RETENTION_LOST");
    assert_eq!(info.metadata.get("requested_index"), Some(&"2".to_string()));
    assert_eq!(
        info.metadata.get("first_retained_index"),
        Some(&"5".to_string())
    );

    // Without a token the same log streams what it holds.
    let mut fresh = f
        .service
        .subscribe(Request::new(register("fresh", strict())))
        .await
        .expect("subscribe")
        .into_inner();
    for expected in 5..10 {
        assert_eq!(
            next_index(&mut fresh, Duration::from_secs(5)).await,
            Some(expected)
        );
    }
}

/// A connected reader slower than the log keeps a BOUNDED registration: the
/// stream heartbeats while it waits on the reader, so liveness ends only
/// consumers that are gone.
#[tokio::test(flavor = "multi_thread")]
async fn a_slow_bounded_reader_is_not_ended_for_liveness() {
    let oplog_dir = tempfile::tempdir().expect("oplog dir");
    write_oplog(oplog_dir.path(), 300);
    let mut f = fixture(
        vec![oplog_dir.path().to_path_buf()],
        Arc::new(|| 300),
        None,
        RegistryTuning {
            heartbeat_window_ms: Some(20),
            eviction_interval_ms: Some(50),
        },
    );
    f.service = f.service.with_tuning(super::CdcStreamTuning {
        heartbeat_interval: Duration::from_millis(50),
        batch_size: std::num::NonZeroUsize::new(256).expect("nonzero"),
        ..super::CdcStreamTuning::default()
    });
    // Thirty heartbeat intervals: each heartbeat lands through a commit, and
    // a loaded test host can stretch one well past its interval.
    let liveness_ms = 1_500;
    let mut stream = f
        .service
        .subscribe(Request::new(register("slow", bounded(Some(liveness_ms)))))
        .await
        .expect("subscribe")
        .into_inner();

    // The reader takes nothing for two timeouts; the stream fills its buffer
    // and waits.
    tokio::time::sleep(Duration::from_millis(2 * liveness_ms)).await;
    assert_eq!(
        state_of(&f.registry, "slow"),
        Some(RegistrationState::Live),
        "a connected reader was ended while the stream waited on it"
    );
    for expected in 0..300 {
        assert_eq!(
            next_index(&mut stream, Duration::from_secs(5)).await,
            Some(expected)
        );
    }
}

/// A change stream sends an entry only once this node has applied it: the
/// log also holds entries a later leader may replace.
#[tokio::test(flavor = "multi_thread")]
async fn stream_sends_only_applied_entries() {
    let oplog_dir = tempfile::tempdir().expect("oplog dir");
    write_oplog(oplog_dir.path(), 5);
    let applied = Arc::new(AtomicU64::new(3));
    let frontier = Arc::clone(&applied);
    let (applies, changes) = tokio::sync::watch::channel(0u64);
    let f = fixture(
        vec![oplog_dir.path().to_path_buf()],
        Arc::new(move || frontier.load(Ordering::Acquire)),
        Some(changes),
        RegistryTuning::default(),
    );
    let mut stream = f
        .service
        .subscribe(Request::new(register("tail", strict())))
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
    // Only the signal can wake the stream: the heartbeat is far beyond this.
    applied.store(5, Ordering::Release);
    applies.send(5).expect("stream listens");
    assert_eq!(
        next_index(&mut stream, Duration::from_secs(5)).await,
        Some(3)
    );
    assert_eq!(
        next_index(&mut stream, Duration::from_secs(5)).await,
        Some(4)
    );
}

fn checkpoint_of(registry: &ShardConsumerRegistry, id: &str) -> Option<u64> {
    registry
        .list_consumers()
        .into_iter()
        .find(|c| c.consumer_id == id)
        .map(|c| c.checkpoint_seqno)
}

/// The event received next, within `within`.
async fn next_event(
    stream: &mut <ChangeEventServiceImpl as ChangeStreamService>::SubscribeStream,
    within: Duration,
) -> Option<crate::proto::replication::cdc::ChangeEvent> {
    tokio::time::timeout(within, stream.next())
        .await
        .ok()
        .flatten()
        .map(|event| event.expect("event"))
}

fn ack(
    id: &str,
    incarnation: u64,
    position: ProtoResumeToken,
) -> Request<AcknowledgeSubscriptionRequest> {
    Request::new(AcknowledgeSubscriptionRequest {
        consumer_id: id.to_string(),
        incarnation,
        position: Some(position),
    })
}

/// A stream whose filters match nothing still reads the log and reports how
/// far with a progress event: no ops, a position past the dropped entries.
/// Acknowledging it moves the registration, and only that does: a filter
/// that matches nothing does not have to hold the log forever.
#[tokio::test(flavor = "multi_thread")]
async fn filtered_stream_reports_progress_the_client_can_acknowledge() {
    let oplog_dir = tempfile::tempdir().expect("oplog dir");
    write_oplog(oplog_dir.path(), 5);
    let f = fixture(
        vec![oplog_dir.path().to_path_buf()],
        Arc::new(|| 5),
        None,
        RegistryTuning::default(),
    );
    let mut stream = f
        .service
        .subscribe(Request::new(SubscribeRequest {
            filters: Some(ProtoCdcFilters {
                edge_types: vec!["NEVER_WRITTEN".to_string()],
                is_migration: None,
            }),
            ..register("filtered", strict())
        }))
        .await
        .expect("subscribe")
        .into_inner();

    let progress = next_event(&mut stream, Duration::from_secs(5))
        .await
        .expect("a progress event");
    assert!(progress.ops.is_empty(), "no entry matched the filter");
    let position = progress.position.expect("a position");
    assert_eq!((position.segment_id, position.entry_offset), (5, 0));
    assert_eq!(
        checkpoint_of(&f.registry, "filtered"),
        Some(0),
        "sending is not delivery"
    );

    f.service
        .acknowledge_subscription(ack("filtered", 1, position))
        .await
        .expect("acknowledge");
    assert_eq!(checkpoint_of(&f.registry, "filtered"), Some(5));
    assert_eq!(
        next_index(&mut stream, Duration::from_millis(200)).await,
        None,
        "no further entry matched the filter"
    );
}

/// Events a client never got are sent again: a stream that queued a batch
/// and was dropped unread moved nothing, so resuming without a token starts
/// at the same place and re-sends the batch.
#[tokio::test(flavor = "multi_thread")]
async fn a_batch_dropped_unread_is_sent_again_on_resume() {
    let oplog_dir = tempfile::tempdir().expect("oplog dir");
    write_oplog(oplog_dir.path(), 5);
    let f = fixture(
        vec![oplog_dir.path().to_path_buf()],
        Arc::new(|| 5),
        None,
        RegistryTuning::default(),
    );
    let first = f
        .service
        .subscribe(Request::new(register("dropper", strict())))
        .await
        .expect("subscribe")
        .into_inner();
    // The whole log fits the channel: give the stream time to queue it.
    tokio::time::sleep(Duration::from_millis(300)).await;
    drop(first);
    assert_eq!(
        checkpoint_of(&f.registry, "dropper"),
        Some(0),
        "queued events are not delivered"
    );

    let mut resumed = f
        .service
        .subscribe(Request::new(resume("dropper", 1)))
        .await
        .expect("resume")
        .into_inner();
    for expected in 0..5 {
        assert_eq!(
            next_index(&mut resumed, Duration::from_secs(5)).await,
            Some(expected),
            "the unread batch comes again"
        );
    }
}

/// Only a position the client could have received releases history: one
/// past what this node applied is refused, one for another shard too, an
/// earlier one than already acknowledged changes nothing, and an
/// acknowledgement for another incarnation is refused.
#[tokio::test(flavor = "multi_thread")]
async fn acknowledgements_release_only_what_was_received() {
    let oplog_dir = tempfile::tempdir().expect("oplog dir");
    write_oplog(oplog_dir.path(), 5);
    let f = fixture(
        vec![oplog_dir.path().to_path_buf()],
        Arc::new(|| 3),
        None,
        RegistryTuning::default(),
    );
    let _stream = f
        .service
        .subscribe(Request::new(register("acker", strict())))
        .await
        .expect("subscribe");
    let at = |index: u64| ProtoResumeToken {
        shard_id: 0,
        segment_id: index,
        entry_offset: 0,
    };

    let past = f
        .service
        .acknowledge_subscription(ack("acker", 1, at(4)))
        .await
        .expect_err("past the applied entries");
    assert_eq!(past.code(), Code::InvalidArgument);
    let other_shard = f
        .service
        .acknowledge_subscription(ack(
            "acker",
            1,
            ProtoResumeToken {
                shard_id: 9,
                ..at(1)
            },
        ))
        .await
        .expect_err("another shard");
    assert_eq!(other_shard.code(), Code::InvalidArgument);
    assert_eq!(
        checkpoint_of(&f.registry, "acker"),
        Some(0),
        "nothing released"
    );

    f.service
        .acknowledge_subscription(ack("acker", 1, at(3)))
        .await
        .expect("acknowledge what was applied");
    f.service
        .acknowledge_subscription(ack("acker", 1, at(1)))
        .await
        .expect("an earlier position is accepted");
    assert_eq!(
        checkpoint_of(&f.registry, "acker"),
        Some(3),
        "an earlier position does not move it back"
    );

    let stale = f
        .service
        .acknowledge_subscription(ack("acker", 2, at(3)))
        .await
        .expect_err("another incarnation");
    assert_eq!(stale.code(), Code::FailedPrecondition);
}

/// A consumer reconnecting to its incarnation without a token continues
/// from its last acknowledgement, not from the start of the log.
#[tokio::test(flavor = "multi_thread")]
async fn a_resumed_stream_continues_where_its_registration_stands() {
    let oplog_dir = tempfile::tempdir().expect("oplog dir");
    write_oplog(oplog_dir.path(), 8);
    let applied = Arc::new(AtomicU64::new(5));
    let frontier = Arc::clone(&applied);
    let (applies, changes) = tokio::sync::watch::channel(0u64);
    let f = fixture(
        vec![oplog_dir.path().to_path_buf()],
        Arc::new(move || frontier.load(Ordering::Acquire)),
        Some(changes),
        RegistryTuning::default(),
    );
    let mut first = f
        .service
        .subscribe(Request::new(register("resumer", strict())))
        .await
        .expect("subscribe")
        .into_inner();
    let mut last = None;
    for expected in 0..5 {
        let event = next_event(&mut first, Duration::from_secs(5))
            .await
            .expect("an event");
        assert_eq!(event.log_index, expected);
        last = event.position;
    }
    f.service
        .acknowledge_subscription(ack("resumer", 1, last.expect("a position")))
        .await
        .expect("acknowledge what was read");
    assert_eq!(checkpoint_of(&f.registry, "resumer"), Some(5));
    drop(first);

    applied.store(8, Ordering::Release);
    applies.send(8).expect("signal");
    let mut resumed = f
        .service
        .subscribe(Request::new(resume("resumer", 1)))
        .await
        .expect("resume")
        .into_inner();
    for expected in 5..8 {
        assert_eq!(
            next_index(&mut resumed, Duration::from_secs(5)).await,
            Some(expected)
        );
    }
}

/// A live id cannot be registered a second time: ALREADY_EXISTS, and the
/// live registration is untouched.
#[tokio::test(flavor = "multi_thread")]
async fn a_second_registration_of_a_live_id_is_refused() {
    let oplog_dir = tempfile::tempdir().expect("oplog dir");
    let f = fixture(
        vec![oplog_dir.path().to_path_buf()],
        Arc::new(|| 0),
        None,
        RegistryTuning::default(),
    );
    let _first = f
        .service
        .subscribe(Request::new(register("dup", strict())))
        .await
        .expect("subscribe");
    let status = match f
        .service
        .subscribe(Request::new(register("dup", strict())))
        .await
    {
        Ok(_) => panic!("a second registration of a live id was accepted"),
        Err(status) => status,
    };
    assert_eq!(status.code(), Code::AlreadyExists);
    assert_eq!(state_of(&f.registry, "dup"), Some(RegistrationState::Live));
}

/// Cancelling ends the registration: its open stream ends with
/// CONSUMER_TERMINATED, and so does every later resume or cancel of the
/// incarnation; an id never registered is NOT_FOUND.
#[tokio::test(flavor = "multi_thread")]
async fn cancelling_ends_the_registration_and_its_stream() {
    let oplog_dir = tempfile::tempdir().expect("oplog dir");
    let mut f = fixture(
        vec![oplog_dir.path().to_path_buf()],
        Arc::new(|| 0),
        None,
        RegistryTuning::default(),
    );
    f.service = f.service.with_tuning(super::CdcStreamTuning {
        heartbeat_interval: Duration::from_millis(50),
        batch_size: std::num::NonZeroUsize::new(256).expect("nonzero"),
        ..super::CdcStreamTuning::default()
    });
    let mut stream = f
        .service
        .subscribe(Request::new(register("ending", strict())))
        .await
        .expect("subscribe")
        .into_inner();

    f.service
        .cancel_subscription(Request::new(CancelSubscriptionRequest {
            consumer_id: "ending".to_string(),
            incarnation: 1,
        }))
        .await
        .expect("cancel");
    assert!(matches!(
        state_of(&f.registry, "ending"),
        Some(RegistrationState::Terminated {
            reason: TerminalReason::Cancelled,
            ..
        })
    ));

    let terminated = |status: tonic::Status| {
        assert_eq!(status.code(), Code::FailedPrecondition, "{status:?}");
        let details = status.get_error_details();
        let info = details.error_info().expect("ErrorInfo");
        assert_eq!(info.reason, "CONSUMER_TERMINATED");
        assert_eq!(info.metadata.get("reason"), Some(&"Cancelled".to_string()));
    };
    let ended = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("the open stream is told")
        .expect("an item");
    terminated(ended.expect_err("the stream ends with the refusal"));

    match f.service.subscribe(Request::new(resume("ending", 1))).await {
        Ok(_) => panic!("an ended incarnation was resumed"),
        Err(status) => terminated(status),
    }
    terminated(
        f.service
            .cancel_subscription(Request::new(CancelSubscriptionRequest {
                consumer_id: "ending".to_string(),
                incarnation: 1,
            }))
            .await
            .expect_err("cancelled twice"),
    );
    let unknown = f
        .service
        .cancel_subscription(Request::new(CancelSubscriptionRequest {
            consumer_id: "never".to_string(),
            incarnation: 1,
        }))
        .await
        .expect_err("an unknown id");
    assert_eq!(unknown.code(), Code::NotFound);

    // Registering the id again starts a new incarnation.
    let again = f
        .service
        .subscribe(Request::new(register("ending", strict())))
        .await
        .expect("register again");
    assert_eq!(
        again
            .metadata()
            .get(INCARNATION_METADATA)
            .and_then(|v| v.to_str().ok()),
        Some("2")
    );
}

/// A registration refused under write pressure reaches the client as
/// retryable RESOURCE_EXHAUSTED with WRITE_BACKPRESSURE and a retry delay,
/// the same answer a write gets in that state.
#[test]
fn a_registration_refused_under_pressure_is_retryable_backpressure() {
    let status = super::registry_status(coordinode_replicate::RegistryError::Backpressure);
    assert_eq!(status.code(), Code::ResourceExhausted);
    let details = status.get_error_details();
    assert_eq!(
        details.error_info().expect("ErrorInfo").reason,
        "WRITE_BACKPRESSURE"
    );
    assert!(details.retry_info().is_some(), "a retry delay is advised");
}

/// The service the server builds for a Raft node streams what the node
/// commits: up to the node's applied entries, each event carrying the
/// proposal's commit timestamp and the leader's term.
#[tokio::test(flavor = "multi_thread")]
async fn stream_follows_a_raft_node() {
    let (engine, _engine_dir) = open_engine();
    let node = Arc::new(
        coordinode_raft::cluster::RaftNode::single_node(Arc::clone(&engine))
            .await
            .expect("bootstrap"),
    );
    let pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> =
        Arc::new(node.pipeline());
    let applied_node = Arc::clone(&node);
    let source = Arc::new(NodeRetentionSource::new(
        Arc::clone(&engine),
        coordinode_raft::storage::raft_oplog_dirs(&engine, 0)
            .expect("dirs")
            .all,
        Arc::new(move || applied_node.applied_through()),
    ));
    let (registry, _bg) = build_consumer_registry(
        Arc::clone(&engine),
        pipeline,
        source,
        RegistryTuning::default(),
    );
    let service = ChangeEventServiceImpl::for_raft_node(&engine, Arc::clone(&node), registry)
        .expect("service");
    let mut stream = service
        .subscribe(Request::new(register("raft-tail", strict())))
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
        if (5001..=5003).contains(&event.ts) {
            assert!(event.term >= 1, "the leader's term, got {}", event.term);
            seen.push(event.ts);
        }
    }
    assert_eq!(seen, vec![5001, 5002, 5003]);

    drop(stream);
    node.shutdown().await.expect("shutdown");
}
