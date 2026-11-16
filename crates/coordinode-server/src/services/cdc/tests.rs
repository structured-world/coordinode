use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use coordinode_core::graph::node::{NodeId, NodeRecord, encode_node_key};
use coordinode_core::txn::proposal::{
    Mutation, PartitionId, ProposalIdGenerator, ProposalPipeline, RaftProposal, UnitEffects,
};
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_replicate::{
    ConsumerKind, RegistrationState, RegistryBackground, SeqnoConsumerRegistry,
    ShardConsumerRegistry, TerminalReason,
};
use coordinode_storage::engine::change_feed::Interest;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::local_pipeline::OwnedLocalProposalPipeline;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::transaction::{CommitContext, Transaction};
use coordinode_storage::oplog::entry::{OplogEntry, OplogOp};
use coordinode_storage::oplog::manager::{OplogManager, RetainedFloor};
use tokio_stream::StreamExt as _;
use tonic::{Code, Request};
use tonic_types::StatusExt;

use super::{AppliedFrontier, AppliedSignal, ChangeEventServiceImpl, INCARNATION_METADATA};
use crate::proto::replication::cdc::change_stream_service_server::ChangeStreamService;
use crate::proto::replication::cdc::{
    AcknowledgeSubscriptionRequest, CancelSubscriptionRequest, ChangeBatch, ChangeInterest,
    ConsumerRetention, NameList, OrdinaryRetention, StrictRetention, SubscribeRequest,
    consumer_retention, logical_change,
};
use crate::registry::{RegistryTuning, build_consumer_registry};
use coordinode_replicate::LogRetentionSource;

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

/// Found `engine`'s deployment with a commit through `pipeline` (a record
/// keyed by its commit, which binds the history domain first), and name the
/// group whose log the tests write: the history every cursor binds.
fn found_history(engine: &StorageEngine, pipeline: &dyn ProposalPipeline) {
    let oracle = engine.oracle().expect("the store has a clock");
    let mut txn = Transaction::begin(engine, Some(&oracle), oracle.next());
    txn.put_stamped(Partition::Schema, b"test:founded:", b"")
        .expect("stage");
    let ids = ProposalIdGenerator::new();
    let wc = WriteConcern::default();
    txn.commit(&CommitContext {
        write_concern: &wc,
        pipeline: Some(pipeline),
        id_gen: Some(&ids),
    })
    .expect("found the deployment");
    engine.set_log_group(coordinode_core::group::GroupId::FORMING);
}

/// A service over the log in `dirs`, applied up to `applied`, with the
/// registry the server builds over the same log.
struct Fixture {
    service: ChangeEventServiceImpl,
    engine: Arc<StorageEngine>,
    registry: ShardConsumerRegistry,
    _bg: RegistryBackground,
    _engine_dir: tempfile::TempDir,
}

fn fixture(
    dirs: Vec<PathBuf>,
    floor: RetainedFloor,
    applied: AppliedFrontier,
    signal: AppliedSignal,
    tuning: RegistryTuning,
) -> Fixture {
    let (engine, engine_dir) = open_engine();
    let pipeline: Arc<dyn ProposalPipeline> = Arc::new(OwnedLocalProposalPipeline::new(&engine));
    found_history(&engine, &*pipeline);
    let source = Arc::new(LogRetentionSource::new(
        Arc::clone(&engine),
        dirs.clone(),
        floor,
        Arc::clone(&applied),
    ));
    let (registry, bg) =
        build_consumer_registry(Arc::clone(&engine), pipeline, source, tuning).expect("registry");
    let service = ChangeEventServiceImpl::new(
        0,
        Arc::clone(&engine),
        dirs,
        registry.clone(),
        applied,
        signal,
    );
    Fixture {
        service,
        engine,
        registry,
        _bg: bg,
        _engine_dir: engine_dir,
    }
}

impl Fixture {
    /// A cursor at `position` following every change.
    fn cursor_at(&self, position: u64) -> Vec<u8> {
        self.engine
            .change_cursor_at(position, Interest::all())
            .expect("cursor")
            .to_bytes()
    }
}

fn strict() -> Option<ConsumerRetention> {
    Some(ConsumerRetention {
        policy: Some(consumer_retention::Policy::Strict(StrictRetention {})),
    })
}

fn ordinary() -> Option<ConsumerRetention> {
    Some(ConsumerRetention {
        policy: Some(consumer_retention::Policy::Ordinary(OrdinaryRetention {})),
    })
}

/// A request registering `id` anew with `retention`, following every change.
fn register(id: &str, retention: Option<ConsumerRetention>) -> SubscribeRequest {
    SubscribeRequest {
        cursor: Vec::new(),
        interest: None,
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

/// The floor of the log in `dir`, which holds no segment yet.
fn empty_log(dir: &std::path::Path) -> RetainedFloor {
    OplogManager::open(dir, 0, 64 * 1024 * 1024, 50_000, 7 * 24 * 3600)
        .expect("open oplog")
        .retained_floor()
}

/// The entry at `index` carrying one unit, as a commit writes it: a node
/// record put and the unit's effect context.
fn unit_entry(index: u64, ts: u64) -> OplogEntry {
    let record = NodeRecord::with_labels(Vec::new())
        .to_msgpack()
        .expect("record");
    let frame = coordinode_core::txn::frame::encode_unit(
        &[
            Mutation::Put {
                partition: PartitionId::Node,
                key: encode_node_key(0, NodeId::from_raw(index + 1)),
                value: record,
            },
            Mutation::Effects(UnitEffects::new()),
        ],
        Timestamp::from_raw(ts),
    )
    .expect("frame");
    OplogEntry {
        ts,
        term: 1,
        index,
        shard: 0,
        ops: vec![OplogOp::Unit { frame }],
        is_migration: false,
        pre_images: None,
    }
}

/// Write `count` node-write entries as oplog segments into `dir`, stamped
/// with the current wall clock so a progress-lag bound sees fresh entries.
/// Returns where the log starts.
fn write_oplog(dir: &std::path::Path, count: u64) -> RetainedFloor {
    let mut mgr =
        OplogManager::open(dir, 0, 64 * 1024 * 1024, 50_000, 7 * 24 * 3600).expect("open oplog");
    let now_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock after the epoch")
        .as_micros() as u64;
    for index in 0..count {
        mgr.append(&unit_entry(index, now_us + index))
            .expect("append");
    }
    mgr.flush().expect("flush");
    mgr.retained_floor()
}

/// The next batch received, within `within`.
async fn next_batch(
    stream: &mut <ChangeEventServiceImpl as ChangeStreamService>::SubscribeStream,
    within: Duration,
) -> Option<ChangeBatch> {
    tokio::time::timeout(within, stream.next())
        .await
        .ok()
        .flatten()
        .map(|batch| batch.expect("batch"))
}

/// The log indexes of the next `count` units received, each within
/// `within`; fewer when none arrives in time.
async fn next_units(
    stream: &mut <ChangeEventServiceImpl as ChangeStreamService>::SubscribeStream,
    count: usize,
    within: Duration,
) -> Vec<u64> {
    let mut indexes = Vec::new();
    while indexes.len() < count {
        let Some(batch) = next_batch(stream, within).await else {
            break;
        };
        indexes.extend(batch.units.iter().map(|u| u.log_index));
    }
    indexes
}

/// A scrape of the ops port shows how far each live consumer is behind and
/// how old the work it owes is: the gauges are set when the scrape samples
/// the registry, and nothing else sets them.
#[tokio::test(flavor = "multi_thread")]
async fn a_scrape_shows_each_consumers_lag() {
    let oplog_dir = tempfile::tempdir().expect("oplog dir");
    let f = fixture(
        vec![oplog_dir.path().to_path_buf()],
        write_oplog(oplog_dir.path(), 3),
        Arc::new(|| 3),
        None,
        RegistryTuning::default(),
    );
    let _stream = f
        .service
        .subscribe(Request::new(register("behind", ordinary())))
        .await
        .expect("subscribe");

    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        crate::registry::sample_consumer_gauges(&f.registry);
    });
    let scrape = handle.render();
    assert!(
        scrape.contains("registry_consumer_lag_seqno{consumer_id=\"behind\""),
        "{scrape}"
    );
    assert!(
        scrape.contains("registry_consumer_lag_age_seconds{consumer_id=\"behind\"}"),
        "{scrape}"
    );
}

/// A stream registers its consumer with the chosen policy, reports the
/// incarnation in the response metadata, and leaving does not end the
/// registration: the consumer is durable, not the connection.
#[tokio::test(flavor = "multi_thread")]
async fn a_stream_registers_a_consumer_that_outlives_the_connection() {
    let oplog_dir = tempfile::tempdir().expect("oplog dir");
    let f = fixture(
        vec![oplog_dir.path().to_path_buf()],
        empty_log(oplog_dir.path()),
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

/// An open change stream is counted in `coordinode_subscription_active` under
/// its transport while it runs, and no longer once the client has left.
#[test]
fn an_open_stream_is_counted_until_its_client_leaves() {
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let open = |text: &str| -> Option<String> {
        text.lines()
            .find(|l| l.starts_with(r#"coordinode_subscription_active{transport="stream"}"#))
            .map(|l| l.rsplit(' ').next().unwrap_or_default().to_string())
    };
    metrics::with_local_recorder(&recorder, || {
        // One thread, so the stream's spawned task records into this
        // thread's recorder.
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        runtime.block_on(async {
            let oplog_dir = tempfile::tempdir().expect("oplog dir");
            let f = fixture(
                vec![oplog_dir.path().to_path_buf()],
                empty_log(oplog_dir.path()),
                Arc::new(|| 0),
                None,
                RegistryTuning::default(),
            );
            let response = f
                .service
                .subscribe(Request::new(register("sink", strict())))
                .await
                .expect("subscribe");
            let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
            while open(&handle.render()).as_deref() != Some("1") {
                assert!(
                    tokio::time::Instant::now() < deadline,
                    "{}",
                    handle.render()
                );
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            drop(response);
            while open(&handle.render()).as_deref() != Some("0") {
                assert!(
                    tokio::time::Instant::now() < deadline,
                    "{}",
                    handle.render()
                );
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        });
    });
}

/// Requests a stream cannot serve are refused up front, and nothing is left
/// registered for them: a missing id or policy, a cursor no build wrote,
/// a cursor read with another interest than the request's, and a label
/// that is not bound.
#[tokio::test(flavor = "multi_thread")]
async fn subscribe_refuses_requests_it_cannot_serve() {
    let oplog_dir = tempfile::tempdir().expect("oplog dir");
    let f = fixture(
        vec![oplog_dir.path().to_path_buf()],
        empty_log(oplog_dir.path()),
        Arc::new(|| 0),
        None,
        RegistryTuning::default(),
    );
    let types = |names: &[&str]| ChangeInterest {
        labels: None,
        edge_types: Some(NameList {
            names: names.iter().map(|n| (*n).to_string()).collect(),
        }),
    };
    let cases = [
        (register("", strict()), "no consumer id"),
        (register("c", None), "no retention policy"),
        (
            register("c", Some(ConsumerRetention { policy: None })),
            "an empty policy",
        ),
        (
            SubscribeRequest {
                cursor: vec![0xFF, 1, 2],
                ..register("c", strict())
            },
            "bytes no build wrote as a cursor",
        ),
        (
            SubscribeRequest {
                cursor: f.cursor_at(0),
                interest: Some(types(&["KNOWS"])),
                ..register("c", strict())
            },
            "a cursor read with another interest",
        ),
        (
            SubscribeRequest {
                interest: Some(ChangeInterest {
                    labels: Some(NameList {
                        names: vec!["NeverBound".into()],
                    }),
                    edge_types: None,
                }),
                ..register("c", strict())
            },
            "a label that is not bound",
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
        mgr.append(&unit_entry(index, 1000 + index))
            .expect("append");
        if index == 4 {
            mgr.rotate().expect("seal");
        }
    }
    mgr.rotate().expect("seal");
    assert_eq!(mgr.purge_before(5).expect("purge"), 1);
    let f = fixture(
        vec![oplog_dir.path().to_path_buf()],
        mgr.retained_floor(),
        Arc::new(|| 10),
        None,
        RegistryTuning::default(),
    );

    let status = match f
        .service
        .subscribe(Request::new(SubscribeRequest {
            cursor: f.cursor_at(2),
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

    // Without a cursor the same log streams what it holds.
    let mut fresh = f
        .service
        .subscribe(Request::new(register("fresh", strict())))
        .await
        .expect("subscribe")
        .into_inner();
    assert_eq!(
        next_units(&mut fresh, 5, Duration::from_secs(5)).await,
        vec![5, 6, 7, 8, 9]
    );
}

/// An ordinary consumer that stopped keeps nothing of the log: the log is
/// purged past its acknowledged position, and resuming it is refused with
/// RETENTION_LOST naming that position and where the log starts now. It is
/// never resumed past the gap or moved to the newest entry; its registration
/// stays live for a rebuild to acknowledge a held position.
#[tokio::test(flavor = "multi_thread")]
async fn a_stopped_ordinary_consumer_learns_its_history_is_gone() {
    let oplog_dir = tempfile::tempdir().expect("oplog dir");
    let mut mgr = OplogManager::open(oplog_dir.path(), 0, 64 * 1024 * 1024, 50_000, 7 * 24 * 3600)
        .expect("open oplog");
    for index in 0..10u64 {
        mgr.append(&unit_entry(index, 1000 + index))
            .expect("append");
        if index == 4 {
            mgr.rotate().expect("seal");
        }
    }
    mgr.rotate().expect("seal");
    let f = fixture(
        vec![oplog_dir.path().to_path_buf()],
        mgr.retained_floor(),
        Arc::new(|| 10),
        None,
        RegistryTuning::default(),
    );
    let response = f
        .service
        .subscribe(Request::new(register("sink", ordinary())))
        .await
        .expect("subscribe");
    let incarnation: u64 = response
        .metadata()
        .get(INCARNATION_METADATA)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok())
        .expect("incarnation");
    drop(response);
    assert_eq!(
        f.registry.oplog_retention_floor(),
        u64::MAX,
        "an ordinary registration holds no log"
    );

    assert_eq!(mgr.purge_before(5).expect("purge"), 1);
    let status = match f
        .service
        .subscribe(Request::new(resume("sink", incarnation)))
        .await
    {
        Ok(_) => panic!("a stream resumed over a purged position"),
        Err(status) => status,
    };
    assert_eq!(status.code(), Code::FailedPrecondition, "{status:?}");
    let details = status.get_error_details();
    let info = details.error_info().expect("ErrorInfo");
    assert_eq!(info.reason, "RETENTION_LOST");
    assert_eq!(info.metadata.get("requested_index"), Some(&"0".to_string()));
    assert_eq!(
        info.metadata.get("first_retained_index"),
        Some(&"5".to_string())
    );
    assert_eq!(
        state_of(&f.registry, "sink"),
        Some(RegistrationState::Live),
        "the loss is reported, the registration kept"
    );
}

/// A change stream sends a unit only once this node has applied its entry:
/// the log also holds entries a later leader may replace.
#[tokio::test(flavor = "multi_thread")]
async fn stream_sends_only_applied_entries() {
    let oplog_dir = tempfile::tempdir().expect("oplog dir");
    let floor = write_oplog(oplog_dir.path(), 5);
    let applied = Arc::new(AtomicU64::new(3));
    let frontier = Arc::clone(&applied);
    let (applies, changes) = tokio::sync::watch::channel(0u64);
    let f = fixture(
        vec![oplog_dir.path().to_path_buf()],
        floor,
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

    assert_eq!(
        next_units(&mut stream, 3, Duration::from_secs(5)).await,
        vec![0, 1, 2]
    );
    assert!(
        next_batch(&mut stream, Duration::from_millis(400))
            .await
            .is_none(),
        "entries 3 and 4 are not applied yet"
    );
    // Only the signal can wake the stream: the heartbeat is far beyond this.
    applied.store(5, Ordering::Release);
    applies.send(5).expect("stream listens");
    assert_eq!(
        next_units(&mut stream, 2, Duration::from_secs(5)).await,
        vec![3, 4]
    );
}

/// A transaction with more events than one read holds goes out in fragments
/// covering consecutive ordinal ranges, every event once; a subscription
/// started from the cursor of a fragment continues inside the transaction.
#[tokio::test(flavor = "multi_thread")]
async fn a_large_transaction_goes_out_in_fragments_that_resume_inside_it() {
    let oplog_dir = tempfile::tempdir().expect("oplog dir");
    let mut mgr = OplogManager::open(oplog_dir.path(), 0, 64 * 1024 * 1024, 50_000, 7 * 24 * 3600)
        .expect("open oplog");
    let record = NodeRecord::with_labels(Vec::new())
        .to_msgpack()
        .expect("record");
    let mut mutations: Vec<Mutation> = (1..=60u64)
        .map(|id| Mutation::Put {
            partition: PartitionId::Node,
            key: encode_node_key(0, NodeId::from_raw(id)),
            value: record.clone(),
        })
        .collect();
    mutations.push(Mutation::Effects(UnitEffects::new()));
    let frame = coordinode_core::txn::frame::encode_unit(&mutations, Timestamp::from_raw(1_000))
        .expect("frame");
    mgr.append(&OplogEntry {
        ts: 1_000,
        term: 1,
        index: 0,
        shard: 0,
        ops: vec![OplogOp::Unit { frame }],
        is_migration: false,
        pre_images: None,
    })
    .expect("append");
    mgr.flush().expect("flush");
    let mut f = fixture(
        vec![oplog_dir.path().to_path_buf()],
        mgr.retained_floor(),
        Arc::new(|| 1),
        None,
        RegistryTuning::default(),
    );
    // Room for a handful of events a read.
    f.service = f.service.with_tuning(super::CdcStreamTuning {
        buffer_bytes: 2_000,
        ..super::CdcStreamTuning::default()
    });

    let mut stream = f
        .service
        .subscribe(Request::new(register("fragments", strict())))
        .await
        .expect("subscribe")
        .into_inner();
    let mut fragments = Vec::new();
    let mut cursors = Vec::new();
    let mut covered = 0;
    while covered < 60 {
        let batch = next_batch(&mut stream, Duration::from_secs(5))
            .await
            .expect("a fragment");
        for unit in &batch.units {
            assert_eq!(unit.projected, 60);
            assert_eq!(unit.first_ordinal, covered, "consecutive ranges");
            covered = unit.end_ordinal;
            fragments.push(unit.clone());
        }
        cursors.push(batch.cursor);
    }
    assert!(fragments.len() > 2, "{} fragments", fragments.len());
    let ordinals: Vec<u32> = fragments
        .iter()
        .flat_map(|u| u.events.iter().map(|e| e.ordinal))
        .collect();
    assert_eq!(ordinals, (0..60).collect::<Vec<_>>());

    // A new subscription from the cursor after the first fragment starts
    // where that fragment ended.
    let mut resumed = f
        .service
        .subscribe(Request::new(SubscribeRequest {
            cursor: cursors[0].clone(),
            ..register("resumed", strict())
        }))
        .await
        .expect("subscribe")
        .into_inner();
    let batch = next_batch(&mut resumed, Duration::from_secs(5))
        .await
        .expect("a fragment");
    assert_eq!(batch.units[0].first_ordinal, fragments[0].end_ordinal);
    assert_eq!(batch.units[0].events[0].ordinal, fragments[0].end_ordinal);
}

/// A subscription to a label a type implies receives a node of the type by
/// the implication its unit committed under, and a change of what the type
/// implies as one scope event naming the type, its activation and the
/// labels before and after; a subscription to an unrelated label receives
/// neither.
#[tokio::test(flavor = "multi_thread")]
async fn a_label_a_type_implies_receives_its_nodes_and_scope_changes() {
    use crate::proto::replication::cdc::{ChangeAction, ChangeDelivery};
    use coordinode_core::txn::proposal::{Implication, NodeBasis, ScopeChange};
    use coordinode_modality::SchemaStore as _;

    let oplog_dir = tempfile::tempdir().expect("oplog dir");
    let applied = Arc::new(AtomicU64::new(2));
    let frontier = Arc::clone(&applied);
    // The log is written after the fixture binds the labels it names.
    let mut mgr = OplogManager::open(oplog_dir.path(), 0, 64 * 1024 * 1024, 50_000, 7 * 24 * 3600)
        .expect("open oplog");
    let f = fixture(
        vec![oplog_dir.path().to_path_buf()],
        mgr.retained_floor(),
        Arc::new(move || frontier.load(Ordering::Acquire)),
        None,
        RegistryTuning::default(),
    );
    let schemas = coordinode_modality::LocalSchemaStore::new(&f.engine);
    // Other is bound for the unrelated subscription, and stored by no node.
    let [task, thing, _other] =
        ["Task", "Thing", "Other"].map(|name| schemas.bind_label(name).expect("bind"));

    // A Task written while Task implies Thing.
    let key = encode_node_key(0, NodeId::from_raw(1));
    let record = NodeRecord::with_labels(vec![task])
        .to_msgpack()
        .expect("record");
    let mut written = UnitEffects::new();
    written.nodes.push(NodeBasis {
        key: key.clone(),
        labels: vec![task],
    });
    written.implications.push(Implication {
        label: task,
        implies: vec![thing],
    });
    // Task stops implying Thing.
    let mut altered = UnitEffects::new();
    altered.scope_changes.push(ScopeChange {
        label: task,
        activation: 77,
        before: vec![thing],
        after: Vec::new(),
    });
    for (index, mutations) in [
        vec![
            Mutation::Put {
                partition: PartitionId::Node,
                key,
                value: record,
            },
            Mutation::Effects(written),
        ],
        vec![Mutation::Effects(altered)],
    ]
    .into_iter()
    .enumerate()
    {
        let ts = 1_000 + index as u64;
        let frame = coordinode_core::txn::frame::encode_unit(&mutations, Timestamp::from_raw(ts))
            .expect("frame");
        mgr.append(&OplogEntry {
            ts,
            term: 1,
            index: index as u64,
            shard: 0,
            ops: vec![OplogOp::Unit { frame }],
            is_migration: false,
            pre_images: None,
        })
        .expect("append");
    }
    mgr.flush().expect("flush");

    let subscribe = |id: &str, label: &str| SubscribeRequest {
        cursor: f
            .engine
            .change_cursor_at(
                0,
                Interest::new(
                    Some(vec![schemas.label_id(label).expect("id").expect("bound")]),
                    Some(Vec::new()),
                ),
            )
            .expect("cursor")
            .to_bytes(),
        interest: Some(ChangeInterest {
            labels: Some(NameList {
                names: vec![label.to_string()],
            }),
            edge_types: Some(NameList { names: Vec::new() }),
        }),
        ..register(id, strict())
    };
    let mut implied = f
        .service
        .subscribe(Request::new(subscribe("thing", "Thing")))
        .await
        .expect("subscribe")
        .into_inner();
    let mut events = Vec::new();
    while let Some(batch) = next_batch(&mut implied, Duration::from_secs(5)).await {
        events.extend(batch.units.into_iter().flat_map(|u| u.events));
        if events.len() >= 2 {
            break;
        }
    }
    assert_eq!(events.len(), 2, "{events:?}");
    assert!(matches!(
        events[0].target,
        Some(logical_change::Target::Node(_))
    ));
    assert_eq!(events[0].delivery, ChangeDelivery::Apply as i32);
    assert_eq!(events[0].labels_after, ["Task", "Thing"]);
    let Some(logical_change::Target::Scope(scope)) = &events[1].target else {
        panic!("a scope change: {events:?}");
    };
    assert_eq!((scope.label.as_str(), scope.activation), ("Task", 77));
    assert_eq!(events[1].action, ChangeAction::ScopeChange as i32);
    assert_eq!(events[1].delivery, ChangeDelivery::Leave as i32);
    assert_eq!(events[1].labels_before, ["Task", "Thing"]);
    assert_eq!(events[1].labels_after, ["Task"]);

    let mut unrelated = f
        .service
        .subscribe(Request::new(subscribe("other", "Other")))
        .await
        .expect("subscribe")
        .into_inner();
    let mut taken = 0;
    while let Some(batch) = next_batch(&mut unrelated, Duration::from_millis(500)).await {
        taken += batch.units.iter().map(|u| u.events.len()).sum::<usize>();
    }
    assert_eq!(taken, 0);
}

fn checkpoint_of(registry: &ShardConsumerRegistry, id: &str) -> Option<u64> {
    registry
        .list_consumers()
        .into_iter()
        .find(|c| c.consumer_id == id)
        .map(|c| c.checkpoint_seqno)
}

fn ack(id: &str, incarnation: u64, cursor: Vec<u8>) -> Request<AcknowledgeSubscriptionRequest> {
    Request::new(AcknowledgeSubscriptionRequest {
        consumer_id: id.to_string(),
        incarnation,
        cursor,
    })
}

/// A stream whose interest takes nothing still reads the log and reports
/// how far: its batch holds every unit it passed, with no event, and a
/// cursor past them. Acknowledging that cursor moves the registration, and
/// only that does: an interest that takes nothing does not have to hold the
/// log forever.
#[tokio::test(flavor = "multi_thread")]
async fn an_interest_taking_nothing_still_moves_a_cursor_the_client_can_acknowledge() {
    let oplog_dir = tempfile::tempdir().expect("oplog dir");
    let floor = write_oplog(oplog_dir.path(), 5);
    let f = fixture(
        vec![oplog_dir.path().to_path_buf()],
        floor,
        Arc::new(|| 5),
        None,
        RegistryTuning::default(),
    );
    let mut stream = f
        .service
        .subscribe(Request::new(SubscribeRequest {
            interest: Some(ChangeInterest {
                labels: Some(NameList { names: Vec::new() }),
                edge_types: Some(NameList {
                    names: vec!["NEVER_WRITTEN".to_string()],
                }),
            }),
            ..register("filtered", strict())
        }))
        .await
        .expect("subscribe")
        .into_inner();

    let batch = next_batch(&mut stream, Duration::from_secs(5))
        .await
        .expect("a batch");
    assert_eq!(batch.units.len(), 5, "every unit passed is in the batch");
    assert!(
        batch
            .units
            .iter()
            .all(|u| u.events.is_empty() && u.projected == 1),
        "the units have their event, which the interest does not take: {batch:?}"
    );
    assert_eq!(
        checkpoint_of(&f.registry, "filtered"),
        Some(0),
        "sending is not delivery"
    );

    f.service
        .acknowledge_subscription(ack("filtered", 1, batch.cursor))
        .await
        .expect("acknowledge");
    assert_eq!(checkpoint_of(&f.registry, "filtered"), Some(5));
    assert!(
        next_batch(&mut stream, Duration::from_millis(200))
            .await
            .is_none(),
        "nothing further was applied"
    );
}

/// Units a client never got are sent again: a stream that queued a batch and
/// was dropped unread moved nothing, so resuming without a cursor starts at
/// the same place and re-sends it.
#[tokio::test(flavor = "multi_thread")]
async fn a_batch_dropped_unread_is_sent_again_on_resume() {
    let oplog_dir = tempfile::tempdir().expect("oplog dir");
    let floor = write_oplog(oplog_dir.path(), 5);
    let f = fixture(
        vec![oplog_dir.path().to_path_buf()],
        floor,
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
        "queued batches are not delivered"
    );

    let mut resumed = f
        .service
        .subscribe(Request::new(resume("dropper", 1)))
        .await
        .expect("resume")
        .into_inner();
    assert_eq!(
        next_units(&mut resumed, 5, Duration::from_secs(5)).await,
        vec![0, 1, 2, 3, 4],
        "the unread batch comes again"
    );
}

/// Only a cursor the client could have received releases history: one past
/// what this node applied is refused, one of another history or no cursor
/// at all too, an earlier one than already acknowledged changes nothing,
/// and an acknowledgement for another incarnation is refused.
#[tokio::test(flavor = "multi_thread")]
async fn acknowledgements_release_only_what_was_received() {
    let oplog_dir = tempfile::tempdir().expect("oplog dir");
    let floor = write_oplog(oplog_dir.path(), 5);
    let f = fixture(
        vec![oplog_dir.path().to_path_buf()],
        floor,
        Arc::new(|| 3),
        None,
        RegistryTuning::default(),
    );
    let _stream = f
        .service
        .subscribe(Request::new(register("acker", strict())))
        .await
        .expect("subscribe");

    let past = f
        .service
        .acknowledge_subscription(ack("acker", 1, f.cursor_at(4)))
        .await
        .expect_err("past the applied entries");
    assert_eq!(past.code(), Code::InvalidArgument);
    // A cursor of another deployment: its history byte range differs.
    let mut foreign = f.cursor_at(1);
    foreign[2] ^= 0xFF;
    let other_history = f
        .service
        .acknowledge_subscription(ack("acker", 1, foreign))
        .await
        .expect_err("another history");
    assert_eq!(other_history.code(), Code::InvalidArgument);
    let none = f
        .service
        .acknowledge_subscription(ack("acker", 1, Vec::new()))
        .await
        .expect_err("no cursor");
    assert_eq!(none.code(), Code::InvalidArgument);
    assert_eq!(
        checkpoint_of(&f.registry, "acker"),
        Some(0),
        "nothing released"
    );

    f.service
        .acknowledge_subscription(ack("acker", 1, f.cursor_at(3)))
        .await
        .expect("acknowledge what was applied");
    f.service
        .acknowledge_subscription(ack("acker", 1, f.cursor_at(1)))
        .await
        .expect("an earlier cursor is accepted");
    assert_eq!(
        checkpoint_of(&f.registry, "acker"),
        Some(3),
        "an earlier cursor does not move it back"
    );

    let stale = f
        .service
        .acknowledge_subscription(ack("acker", 2, f.cursor_at(3)))
        .await
        .expect_err("another incarnation");
    assert_eq!(stale.code(), Code::FailedPrecondition);
}

/// A consumer reconnecting to its incarnation without a cursor continues
/// from its last acknowledgement, not from the start of the log.
#[tokio::test(flavor = "multi_thread")]
async fn a_resumed_stream_continues_where_its_registration_stands() {
    let oplog_dir = tempfile::tempdir().expect("oplog dir");
    let floor = write_oplog(oplog_dir.path(), 8);
    let applied = Arc::new(AtomicU64::new(5));
    let frontier = Arc::clone(&applied);
    let (applies, changes) = tokio::sync::watch::channel(0u64);
    let f = fixture(
        vec![oplog_dir.path().to_path_buf()],
        floor,
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
    let mut indexes = Vec::new();
    let mut last = None;
    while indexes.len() < 5 {
        let batch = next_batch(&mut first, Duration::from_secs(5))
            .await
            .expect("a batch");
        indexes.extend(batch.units.iter().map(|u| u.log_index));
        last = Some(batch.cursor);
    }
    assert_eq!(indexes, vec![0, 1, 2, 3, 4]);
    f.service
        .acknowledge_subscription(ack("resumer", 1, last.expect("a cursor")))
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
    assert_eq!(
        next_units(&mut resumed, 3, Duration::from_secs(5)).await,
        vec![5, 6, 7]
    );
}

/// A unit recorded without the context that says what it means cannot be
/// projected: the stream ends with CHANGE_FEED_UNSUPPORTED at its entry,
/// having handed over the units before it, and never passes it.
#[tokio::test(flavor = "multi_thread")]
async fn a_unit_without_its_context_ends_the_stream_explicitly() {
    let oplog_dir = tempfile::tempdir().expect("oplog dir");
    let mut mgr = OplogManager::open(oplog_dir.path(), 0, 64 * 1024 * 1024, 50_000, 7 * 24 * 3600)
        .expect("open oplog");
    mgr.append(&unit_entry(0, 1000)).expect("append");
    let bare = coordinode_core::txn::frame::encode_unit(
        &[Mutation::Put {
            partition: PartitionId::Node,
            key: encode_node_key(0, NodeId::from_raw(9)),
            value: NodeRecord::with_labels(Vec::new())
                .to_msgpack()
                .expect("record"),
        }],
        Timestamp::from_raw(1001),
    )
    .expect("frame");
    mgr.append(&OplogEntry {
        ts: 1001,
        term: 1,
        index: 1,
        shard: 0,
        ops: vec![OplogOp::Unit { frame: bare }],
        is_migration: false,
        pre_images: None,
    })
    .expect("append");
    mgr.flush().expect("flush");
    let mut f = fixture(
        vec![oplog_dir.path().to_path_buf()],
        mgr.retained_floor(),
        Arc::new(|| 2),
        None,
        RegistryTuning::default(),
    );
    // One entry a batch, so the unit before is handed over first.
    f.service = f.service.with_tuning(super::CdcStreamTuning {
        batch_size: std::num::NonZeroUsize::new(1).expect("nonzero"),
        ..super::CdcStreamTuning::default()
    });
    let mut stream = f
        .service
        .subscribe(Request::new(register("strict-reader", strict())))
        .await
        .expect("subscribe")
        .into_inner();
    assert_eq!(
        next_units(&mut stream, 1, Duration::from_secs(5)).await,
        vec![0]
    );
    let ended = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("the stream is told")
        .expect("an item");
    let status = ended.expect_err("the stream ends with the refusal");
    assert_eq!(status.code(), Code::FailedPrecondition, "{status:?}");
    let details = status.get_error_details();
    let info = details.error_info().expect("ErrorInfo");
    assert_eq!(info.reason, "CHANGE_FEED_UNSUPPORTED");
    assert_eq!(info.metadata.get("log_index"), Some(&"1".to_string()));
}

/// A live id cannot be registered a second time: ALREADY_EXISTS, and the
/// live registration is untouched.
#[tokio::test(flavor = "multi_thread")]
async fn a_second_registration_of_a_live_id_is_refused() {
    let oplog_dir = tempfile::tempdir().expect("oplog dir");
    let f = fixture(
        vec![oplog_dir.path().to_path_buf()],
        empty_log(oplog_dir.path()),
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
        empty_log(oplog_dir.path()),
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

/// A caught-up stream learns that its registration ended as soon as the end
/// applies, not at its next heartbeat: the stream wakes on the applied entry
/// and the registry relays the write to its record on another thread, so it
/// must also wake on that.
#[tokio::test(flavor = "multi_thread")]
async fn a_parked_stream_learns_of_its_end_without_waiting_for_a_heartbeat() {
    let oplog_dir = tempfile::tempdir().expect("oplog dir");
    let mut f = fixture(
        vec![oplog_dir.path().to_path_buf()],
        empty_log(oplog_dir.path()),
        Arc::new(|| 0),
        None,
        RegistryTuning::default(),
    );
    f.service = f.service.with_tuning(super::CdcStreamTuning {
        heartbeat_interval: Duration::from_secs(600),
        ..super::CdcStreamTuning::default()
    });
    let mut stream = f
        .service
        .subscribe(Request::new(register("parked", strict())))
        .await
        .expect("subscribe")
        .into_inner();
    // Let the stream reach its caught-up wait.
    tokio::time::sleep(Duration::from_millis(300)).await;

    f.service
        .cancel_subscription(Request::new(CancelSubscriptionRequest {
            consumer_id: "parked".to_string(),
            incarnation: 1,
        }))
        .await
        .expect("cancel");
    let ended = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("told well before the heartbeat")
        .expect("an item");
    let status = ended.expect_err("the stream ends with the refusal");
    assert_eq!(status.code(), Code::FailedPrecondition, "{status:?}");
}

/// A registration refused under pressure reaches the client as retryable
/// RESOURCE_EXHAUSTED with WRITE_BACKPRESSURE and a retry delay, the same
/// answer a write gets in that state, naming the pressure (and the endpoint
/// whose limit has no room).
#[test]
fn a_registration_refused_under_pressure_is_retryable_backpressure() {
    use coordinode_replicate::{RegistryError, RetentionPressure};

    let status = super::registry_status(RegistryError::Backpressure(
        RetentionPressure::WritePressure,
    ));
    assert_eq!(status.code(), Code::ResourceExhausted);
    let details = status.get_error_details();
    let info = details.error_info().expect("ErrorInfo");
    assert_eq!(info.reason, "WRITE_BACKPRESSURE");
    assert_eq!(
        info.metadata.get("pressure").map(String::as_str),
        Some("writePressure")
    );
    assert!(details.retry_info().is_some(), "a retry delay is advised");

    let status =
        super::registry_status(RegistryError::Backpressure(RetentionPressure::Footprint {
            endpoint_id: "hot".into(),
        }));
    let info = status
        .get_error_details()
        .error_info()
        .cloned()
        .expect("ErrorInfo");
    assert_eq!(
        info.metadata.get("pressure").map(String::as_str),
        Some("footprint")
    );
    assert_eq!(
        info.metadata.get("endpointId").map(String::as_str),
        Some("hot")
    );
}

/// The service the server builds for a Raft node streams what the node
/// commits: up to the node's applied entries, each unit carrying the
/// proposal's commit timestamp and its node put.
#[tokio::test(flavor = "multi_thread")]
async fn stream_follows_a_raft_node() {
    let (engine, _engine_dir) = open_engine();
    let node = Arc::new(
        coordinode_raft::cluster::RaftNode::single_node(Arc::clone(&engine))
            .await
            .expect("bootstrap"),
    );
    let pipeline: Arc<dyn ProposalPipeline> = Arc::new(node.pipeline());
    found_history(&engine, &*pipeline);
    let applied_node = Arc::clone(&node);
    let source = Arc::new(LogRetentionSource::new(
        Arc::clone(&engine),
        coordinode_raft::storage::raft_oplog_dirs(&engine, 0)
            .expect("dirs")
            .all,
        node.retained_floor(),
        Arc::new(move || applied_node.applied_through()),
    ));
    let (registry, _bg) = build_consumer_registry(
        Arc::clone(&engine),
        Arc::clone(&pipeline),
        source,
        RegistryTuning::default(),
    )
    .expect("registry");
    let service = ChangeEventServiceImpl::for_raft_node(&engine, Arc::clone(&node), registry)
        .expect("service");
    let mut stream = service
        .subscribe(Request::new(register("raft-tail", strict())))
        .await
        .expect("subscribe")
        .into_inner();

    // Each commit takes its timestamp from the store's clock, as a commit
    // does: the leader refuses one below a bound already in its log.
    let oracle = engine.oracle().expect("the store has a clock");
    let raft_pipeline = node.pipeline();
    let ids = ProposalIdGenerator::new();
    let record = NodeRecord::with_labels(Vec::new())
        .to_msgpack()
        .expect("record");
    let mut committed = Vec::new();
    for i in 1..=3u64 {
        let (ts, held) = engine.pending_commits().obligate(|| oracle.next().as_raw());
        raft_pipeline
            .propose_and_wait(&RaftProposal {
                id: ids.next(),
                mutations: vec![
                    Mutation::Put {
                        partition: PartitionId::Node,
                        key: encode_node_key(0, NodeId::from_raw(100 + i)),
                        value: record.clone(),
                    },
                    Mutation::Effects(UnitEffects::new()),
                ],
                commit_ts: Timestamp::from_raw(ts),
                start_ts: Timestamp::from_raw(ts),
                bypass_rate_limiter: false,
            })
            .expect("propose");
        drop(held);
        committed.push(ts);
    }

    let mut seen = Vec::new();
    while seen.len() < 3 {
        let batch = tokio::time::timeout(Duration::from_secs(10), stream.next())
            .await
            .expect("a batch for every committed proposal")
            .expect("stream open")
            .expect("batch");
        for unit in batch.units {
            if committed.contains(&unit.commit_ts) {
                let node = unit.events.iter().find_map(|e| match &e.target {
                    Some(logical_change::Target::Node(node)) => Some(node.node_id),
                    _ => None,
                });
                assert!(node.is_some_and(|id| id > 100), "{unit:?}");
                seen.push(unit.commit_ts);
            }
        }
    }
    assert_eq!(seen, committed);

    drop(stream);
    node.shutdown().await.expect("shutdown");
}
