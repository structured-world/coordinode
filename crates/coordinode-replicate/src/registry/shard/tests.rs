use super::*;
use crate::registry::types::ConsumerKind;
use coordinode_raft::cluster::RaftNode;
use coordinode_raft::proposal::RaftProposalPipeline;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};

/// Test clock the suite advances by hand for deterministic TTL expiry.
struct ManualClock(Mutex<u64>);
impl ManualClock {
    fn new(start: u64) -> Self {
        Self(Mutex::new(start))
    }
    fn set(&self, t: u64) {
        *self.0.lock() = t;
    }
}
impl Clock for ManualClock {
    fn now_ms(&self) -> u64 {
        *self.0.lock()
    }
}

async fn registry_with_clock(
    clock: Arc<dyn Clock>,
) -> (
    ShardConsumerRegistry,
    Arc<StorageEngine>,
    Arc<RaftNode>,
    tempfile::TempDir,
) {
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new());
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = Arc::new(
        StorageEngine::open_with_oracle(&config, Arc::clone(&oracle)).expect("open engine"),
    );
    let node = Arc::new(
        RaftNode::open_with_oracle(1, Arc::clone(&engine), Some(oracle))
            .await
            .expect("raft node"),
    );
    tokio::time::sleep(Duration::from_millis(500)).await;
    let pipeline: Arc<dyn ProposalPipeline> =
        Arc::new(RaftProposalPipeline::new(Arc::clone(node.raft())));
    let id_gen = Arc::new(ProposalIdGenerator::with_base(1u64 << 48));
    let reg = ShardConsumerRegistry::new(Arc::clone(&engine), pipeline, id_gen, clock);
    (reg, engine, node, dir)
}

/// Seqno-space registration (drives the GC watermark / `shard_floor`).
fn registration(id: &str, scope: TopologyScope, ttl_ms: u64) -> ConsumerRegistration {
    ConsumerRegistration {
        consumer_id: id.to_string(),
        kind: ConsumerKind::LsmStateDelta,
        scope,
        initial_seqno: InitialSeqno::At(0),
        ttl_ms,
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn register_checkpoint_unregister_drive_floor() {
    let clock = Arc::new(ManualClock::new(1_000));
    let (reg, _engine, node, _dir) = registry_with_clock(clock.clone()).await;

    assert_eq!(reg.shard_floor(), u64::MAX);

    let h1 = reg
        .register(ConsumerRegistration {
            initial_seqno: InitialSeqno::At(100),
            ..registration("c1", TopologyScope::Cluster, 0)
        })
        .expect("register c1");
    let h2 = reg
        .register(ConsumerRegistration {
            initial_seqno: InitialSeqno::At(250),
            ..registration("c2", TopologyScope::Shard(0), 0)
        })
        .expect("register c2");
    assert_eq!(reg.shard_floor(), 100, "floor is the slowest consumer");

    reg.checkpoint(&h1, 300).expect("checkpoint c1");
    assert_eq!(reg.shard_floor(), 250);

    reg.checkpoint(&h2, 10).expect("stale checkpoint c2");
    assert_eq!(
        reg.shard_floor(),
        250,
        "stale checkpoint must not rewind floor"
    );

    reg.unregister(h2).expect("unregister c2");
    assert_eq!(reg.shard_floor(), 300);

    let listed = reg.list_consumers();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].consumer_id, "c1");
    assert_eq!(listed[0].checkpoint_seqno, 300);

    reg.unregister(h1).expect("unregister c1");
    assert_eq!(reg.shard_floor(), u64::MAX, "no consumers → unconstrained");

    node.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread")]
async fn empty_consumer_id_is_rejected() {
    let clock = Arc::new(ManualClock::new(0));
    let (reg, _engine, node, _dir) = registry_with_clock(clock).await;
    let err = reg
        .register(registration("", TopologyScope::Cluster, 0))
        .unwrap_err();
    assert!(matches!(err, RegistryError::EmptyConsumerId));
    node.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread")]
async fn ce_rejects_dc_and_rack_scopes() {
    let clock = Arc::new(ManualClock::new(0));
    let (reg, _engine, node, _dir) = registry_with_clock(clock).await;
    for scope in [
        TopologyScope::Dc("eu".into()),
        TopologyScope::Rack("r1".into()),
    ] {
        let err = reg.register(registration("c", scope, 0)).unwrap_err();
        assert!(
            matches!(err, RegistryError::UnsupportedScope(_)),
            "CE must reject dc/rack, got {err:?}"
        );
    }
    node.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread")]
async fn checkpoint_unknown_consumer_errors() {
    let clock = Arc::new(ManualClock::new(0));
    let (reg, _engine, node, _dir) = registry_with_clock(clock).await;
    let phantom = RegisteredHandle::new("never-registered");
    assert!(matches!(
        reg.checkpoint(&phantom, 5).unwrap_err(),
        RegistryError::UnknownConsumer(_)
    ));
    assert!(matches!(
        reg.unregister(phantom).unwrap_err(),
        RegistryError::UnknownConsumer(_)
    ));
    node.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread")]
async fn expired_registration_is_excluded_from_floor() {
    let clock = Arc::new(ManualClock::new(1_000));
    let (reg, _engine, node, _dir) = registry_with_clock(clock.clone()).await;

    reg.register(ConsumerRegistration {
        initial_seqno: InitialSeqno::At(50),
        ..registration("ttl-consumer", TopologyScope::Cluster, 5_000)
    })
    .expect("register ttl");
    reg.register(ConsumerRegistration {
        initial_seqno: InitialSeqno::At(900),
        ..registration("persistent", TopologyScope::Cluster, 0)
    })
    .expect("register persistent");
    assert_eq!(
        reg.shard_floor(),
        50,
        "ttl consumer pins the floor while alive"
    );

    clock.set(7_000);
    let floor = reg.core.recompute_floor().expect("recompute");
    assert_eq!(floor, 900, "expired consumer no longer pins retention");
    assert_eq!(
        reg.list_consumers().len(),
        1,
        "expired excluded from listing"
    );

    node.shutdown().await.expect("shutdown");
}

/// With the background service running, heartbeats buffer and flush
/// as a coalesced proposal; the persisted `last_heartbeat_ts` advances
/// without a per-heartbeat Raft round-trip.
#[tokio::test(flavor = "multi_thread")]
async fn batched_heartbeats_flush_and_refresh_liveness() {
    let clock = Arc::new(ManualClock::new(1_000));
    let (reg, _engine, node, _dir) = registry_with_clock(clock.clone()).await;
    let h = reg
        .register(registration("hb", TopologyScope::Cluster, 10_000))
        .expect("register");

    let bg = reg.start_background(BackgroundConfig {
        heartbeat_window_ms: 30,
        eviction_interval_ms: 100_000, // don't evict during this test
    });

    // Buffer several heartbeats at a later clock time; none hit Raft yet.
    clock.set(4_000);
    for _ in 0..5 {
        reg.heartbeat(&h).expect("buffer heartbeat");
    }

    // The first buffered heartbeat opens a 30 ms window; the flush after it
    // is a proposal, whose fsync can take longer than the window on a slow
    // disk, so wait for the write rather than for a fixed time.
    let until = tokio::time::Instant::now() + Duration::from_secs(5);
    let mut listed = reg.list_consumers();
    while listed.first().map(|c| c.last_heartbeat_ts_ms) != Some(4_000)
        && tokio::time::Instant::now() < until
    {
        tokio::time::sleep(Duration::from_millis(20)).await;
        listed = reg.list_consumers();
    }
    assert_eq!(listed.len(), 1);
    assert_eq!(
        listed[0].last_heartbeat_ts_ms, 4_000,
        "coalesced flush advanced last_heartbeat_ts to the buffered time"
    );

    bg.shutdown().await;
    node.shutdown().await.expect("shutdown");
}

/// Feed (a), combine B: a registered consumer holds the engine GC
/// watermark back to its checkpoint (CockroachDB protected-timestamp /
/// TiDB service-safe-point shape); with no consumers the watermark falls
/// to the time-travel window, NOT `u64::MAX` (option A's bug that would
/// GC the whole `AS OF TIMESTAMP` history).
#[tokio::test(flavor = "multi_thread")]
async fn consumer_floor_drives_engine_gc_watermark() {
    let clock = Arc::new(ManualClock::new(1_000));
    let (reg, engine, node, _dir) = registry_with_clock(clock).await;

    // A CDC consumer checkpointed far in the past pins the watermark there,
    // overriding the (huge, ~now) live-pin / current-seqno default.
    let h = reg
        .register(ConsumerRegistration {
            initial_seqno: InitialSeqno::At(100),
            ..registration("cdc", TopologyScope::Cluster, 0)
        })
        .expect("register");
    assert_eq!(
        engine.gc_watermark(),
        100,
        "consumer checkpoint holds GC retention back to its seqno"
    );

    reg.checkpoint(&h, 500).expect("advance checkpoint");
    assert_eq!(
        engine.gc_watermark(),
        500,
        "advancing checkpoint lifts the floor"
    );

    // No consumers → the watermark falls to the retention window, which is
    // a real seqno (≈ now - 7d), never u64::MAX and never below the window.
    reg.unregister(h).expect("unregister");
    let wm = engine.gc_watermark();
    assert_ne!(
        wm,
        u64::MAX,
        "empty registry must NOT collapse to GC-everything"
    );
    assert!(
        wm > 500,
        "with no consumers the time-travel window governs, not the old checkpoint (got {wm})"
    );

    node.shutdown().await.expect("shutdown");
}

/// Lagging-consumer guard: when the engine GC watermark advances past a
/// consumer's checkpoint (operator-forced GC bump), `check_retention`
/// returns `RetentionLost` rather than letting the read silently observe a
/// gap. A protected consumer (checkpoint at/above the watermark) gets its
/// safe checkpoint back.
#[tokio::test(flavor = "multi_thread")]
async fn check_retention_surfaces_retention_lost() {
    let clock = Arc::new(ManualClock::new(1_000));
    let (reg, engine, node, _dir) = registry_with_clock(clock).await;

    let h = reg
        .register(ConsumerRegistration {
            initial_seqno: InitialSeqno::At(100),
            ..registration("cdc", TopologyScope::Cluster, 0)
        })
        .expect("register");
    // Registry pins the watermark at the consumer's checkpoint → protected.
    assert_eq!(reg.check_retention(&h).expect("protected"), 100);

    // Operator force-bumps GC retention above the lagging consumer.
    engine.set_consumer_retention_floor(1_000);
    let lost = reg.check_retention(&h);
    assert!(
        matches!(
            lost,
            Err(RegistryError::RetentionLost { checkpoint: 100, floor }) if floor >= 1_000
        ),
        "expected RetentionLost{{checkpoint:100, floor>=1000}}, got {lost:?}"
    );

    // Unknown consumer → UnknownConsumer, not RetentionLost.
    assert!(matches!(
        reg.check_retention(&RegisteredHandle::new("ghost")),
        Err(RegistryError::UnknownConsumer(_))
    ));

    node.shutdown().await.expect("shutdown");
}

/// Failover recovery: registry state lives in the Raft-replicated
/// `Partition::Registry` keyspace, so a registry constructed fresh over an
/// engine that already holds the replicated entries (the new leader after
/// a failover) recovers both floors + the consumer list with no
/// re-registration. Cross-node replication of the underlying proposals is
/// covered by `coordinode-raft`'s `cluster_multiple_proposals_replicate`;
/// this covers the new-leader-recovers half.
#[tokio::test(flavor = "multi_thread")]
async fn registry_recovers_floors_from_persisted_state() {
    let clock = Arc::new(ManualClock::new(1_000));
    let (reg_a, engine, node, _dir) = registry_with_clock(clock.clone()).await;

    // Seqno consumer (→ gc floor) + oplog consumer (→ oplog floor).
    let h = reg_a
        .register(ConsumerRegistration {
            initial_seqno: InitialSeqno::At(100),
            ..registration("backup", TopologyScope::Cluster, 0)
        })
        .expect("register seqno consumer");
    reg_a.checkpoint(&h, 300).expect("checkpoint");
    reg_a
        .register(ConsumerRegistration {
            consumer_id: "cdc".into(),
            kind: ConsumerKind::OplogEvents,
            scope: TopologyScope::Cluster,
            initial_seqno: InitialSeqno::At(50),
            ttl_ms: 0,
        })
        .expect("register oplog consumer");
    drop(reg_a); // old leader steps down

    // New leader: a fresh registry over the same (replicated) engine.
    let pipeline: Arc<dyn ProposalPipeline> =
        Arc::new(RaftProposalPipeline::new(Arc::clone(node.raft())));
    let id_gen = Arc::new(ProposalIdGenerator::with_base(2u64 << 48));
    let reg_b = ShardConsumerRegistry::new(Arc::clone(&engine), pipeline, id_gen, clock);

    // Floors recovered from the keyspace by `new()`'s recompute, no
    // re-registration needed.
    assert_eq!(
        reg_b.shard_floor(),
        300,
        "seqno floor recovered after failover"
    );
    assert_eq!(
        reg_b.oplog_retention_floor(),
        50,
        "oplog floor recovered after failover"
    );
    let mut ids: Vec<String> = reg_b
        .list_consumers()
        .into_iter()
        .map(|c| c.consumer_id)
        .collect();
    ids.sort();
    assert_eq!(ids, vec!["backup".to_string(), "cdc".to_string()]);

    node.shutdown().await.expect("shutdown");
}

/// Space split: an `OplogEvents` consumer (Raft-index space) feeds the
/// oplog retention floor, NOT the MVCC GC watermark; a `LsmStateDelta`
/// consumer (seqno space) feeds the GC watermark, NOT the oplog floor.
/// Mixing them would compare a microsecond HLC against a Raft index.
#[tokio::test(flavor = "multi_thread")]
async fn floors_are_split_by_consumer_space() {
    let clock = Arc::new(ManualClock::new(1_000));
    let (reg, engine, node, _dir) = registry_with_clock(clock).await;

    // Oplog consumer at Raft index 42 → oplog floor, not the gc watermark.
    let oplog_h = reg
        .register(ConsumerRegistration {
            consumer_id: "cdc-sink".into(),
            kind: ConsumerKind::OplogEvents,
            scope: TopologyScope::Cluster,
            initial_seqno: InitialSeqno::At(42),
            ttl_ms: 0,
        })
        .expect("register oplog consumer");
    assert_eq!(
        reg.oplog_retention_floor(),
        42,
        "oplog consumer drives oplog floor"
    );
    assert_eq!(
        reg.shard_floor(),
        u64::MAX,
        "oplog consumer must NOT enter the seqno floor"
    );
    assert_ne!(
        engine.gc_watermark(),
        42,
        "oplog index 42 must NOT pull the MVCC gc watermark into Raft-index space"
    );

    // Seqno consumer at 1000 → gc watermark + shard_floor, not oplog floor.
    let seqno_h = reg
        .register(ConsumerRegistration {
            initial_seqno: InitialSeqno::At(1_000),
            ..registration("backup", TopologyScope::Cluster, 0)
        })
        .expect("register seqno consumer");
    assert_eq!(reg.shard_floor(), 1_000);
    assert_eq!(
        engine.gc_watermark(),
        1_000,
        "seqno consumer drives gc watermark"
    );
    assert_eq!(
        reg.oplog_retention_floor(),
        42,
        "oplog floor unchanged by seqno consumer"
    );

    reg.unregister(oplog_h).expect("unregister oplog");
    reg.unregister(seqno_h).expect("unregister seqno");
    assert_eq!(reg.oplog_retention_floor(), u64::MAX);
    assert_eq!(reg.shard_floor(), u64::MAX);

    node.shutdown().await.expect("shutdown");
}

/// The eviction sweep removes a registration past its TTL via a Raft
/// proposal and lifts the floor it was pinning. Nothing but the TTL running
/// out schedules that sweep: no write and no heartbeat arrive meanwhile.
#[tokio::test(flavor = "multi_thread")]
async fn eviction_sweep_removes_expired_and_lifts_floor() {
    let (reg, _engine, node, _dir) = registry_with_clock(Arc::new(SystemClock)).await;

    reg.register(ConsumerRegistration {
        initial_seqno: InitialSeqno::At(10),
        ..registration("doomed", TopologyScope::Cluster, 300)
    })
    .expect("register doomed");
    reg.register(ConsumerRegistration {
        initial_seqno: InitialSeqno::At(500),
        ..registration("survivor", TopologyScope::Cluster, 0)
    })
    .expect("register survivor");
    assert_eq!(reg.shard_floor(), 10);

    let bg = reg.start_background(BackgroundConfig {
        heartbeat_window_ms: 100_000,
        eviction_interval_ms: 30,
    });

    // Past the doomed consumer's TTL, with a margin for a loaded machine.
    tokio::time::sleep(Duration::from_millis(900)).await;

    let listed = reg.list_consumers();
    assert_eq!(listed.len(), 1, "expired consumer was evicted");
    assert_eq!(listed[0].consumer_id, "survivor");
    assert_eq!(
        reg.shard_floor(),
        500,
        "floor lifted to survivor after eviction"
    );

    bg.shutdown().await;
    node.shutdown().await.expect("shutdown");
}

/// A registration written through another member's registry reaches this
/// member's floor as it applies: the background service has no timer that
/// would pick it up otherwise.
#[tokio::test(flavor = "multi_thread")]
async fn a_registration_applied_from_elsewhere_moves_the_floor() {
    let clock = Arc::new(ManualClock::new(1_000));
    let (reg, engine, node, _dir) = registry_with_clock(clock.clone()).await;
    let bg = reg.start_background(BackgroundConfig {
        heartbeat_window_ms: 100_000,
        eviction_interval_ms: 30,
    });
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(reg.shard_floor(), u64::MAX);

    // Another member's registry: same replicated keyspace, its own handle.
    let elsewhere = ShardConsumerRegistry::new(
        Arc::clone(&engine),
        Arc::new(RaftProposalPipeline::new(Arc::clone(node.raft()))),
        Arc::new(ProposalIdGenerator::with_base(3u64 << 48)),
        clock,
    );
    elsewhere
        .register(ConsumerRegistration {
            initial_seqno: InitialSeqno::At(77),
            ..registration("remote", TopologyScope::Cluster, 0)
        })
        .expect("register elsewhere");

    let until = tokio::time::Instant::now() + Duration::from_secs(5);
    while reg.shard_floor() != 77 && tokio::time::Instant::now() < until {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(
        reg.shard_floor(),
        77,
        "a registration applied from elsewhere did not reach the floor"
    );

    bg.shutdown().await;
    node.shutdown().await.expect("shutdown");
}

/// A heartbeat that has not been flushed yet still counts: the sweep may run
/// before the flush that would persist it, and a consumer that just proved it
/// is alive must not be evicted for the flush's timing.
#[tokio::test(flavor = "multi_thread")]
async fn a_buffered_heartbeat_keeps_its_consumer_from_eviction() {
    let clock = SystemClock;
    let (reg, _engine, node, _dir) = registry_with_clock(Arc::new(clock)).await;
    let ttl_ms = 400;
    let h = reg
        .register(registration("reader", TopologyScope::Cluster, ttl_ms))
        .expect("register");
    let registered_at = reg.list_consumers()[0].last_heartbeat_ts_ms;

    let bg = reg.start_background(BackgroundConfig {
        heartbeat_window_ms: 100_000, // the heartbeat stays buffered
        eviction_interval_ms: 30,
    });

    // Halfway through the TTL the reader heartbeats; the sweep due when the
    // stored heartbeat runs out finds the buffered one.
    tokio::time::sleep(Duration::from_millis(ttl_ms / 2)).await;
    let beat_at = clock.now_ms();
    reg.heartbeat(&h).expect("buffer heartbeat");
    // Past the stored heartbeat's TTL, inside the buffered one's.
    let until_ms = registered_at + ttl_ms + 100;
    // Already past the point on a slow machine: wait for nothing.
    let left = until_ms.saturating_sub(clock.now_ms());
    tokio::time::sleep(Duration::from_millis(left)).await;

    let listed = reg.list_consumers();
    assert_eq!(
        listed.len(),
        1,
        "a consumer with a fresh buffered heartbeat was evicted"
    );
    assert!(
        listed[0].last_heartbeat_ts_ms >= beat_at,
        "the sweep persisted the buffered heartbeat"
    );

    bg.shutdown().await;
    node.shutdown().await.expect("shutdown");
}

/// A flush whose proposal fails keeps the heartbeats it took from the buffer.
/// They were a consumer's sign of life; dropped with the failed write, the
/// next sweep would evict a consumer that heartbeated in time.
#[tokio::test(flavor = "multi_thread")]
async fn a_failed_flush_keeps_its_heartbeats_buffered() {
    let clock = Arc::new(ManualClock::new(1_000));
    let (reg, _engine, node, _dir) = registry_with_clock(clock.clone()).await;
    let h = reg
        .register(registration("reader", TopologyScope::Cluster, 2_000))
        .expect("register");
    reg.core.batching_on.store(true, Ordering::Release);
    clock.set(2_500);
    reg.heartbeat(&h).expect("buffer heartbeat");

    // The consensus is gone: the flush reads the entry and fails to write it.
    node.shutdown().await.expect("shutdown");
    assert!(
        reg.core.flush_pending_heartbeats().is_err(),
        "a flush with no consensus to write through must fail"
    );

    assert_eq!(
        reg.core.pending_hb.lock().get("reader").copied(),
        Some(2_500),
        "the heartbeat of a failed flush is still buffered"
    );
}

/// A pipeline that, once closed, holds each proposal until the test lets it
/// through, reporting that it holds one.
struct GatedPipeline {
    inner: coordinode_raft::proposal::OwnedLocalProposalPipeline,
    closed: std::sync::atomic::AtomicBool,
    held: std::sync::mpsc::SyncSender<()>,
    release: parking_lot::Mutex<std::sync::mpsc::Receiver<()>>,
}

impl ProposalPipeline for GatedPipeline {
    fn propose_and_wait(
        &self,
        proposal: &coordinode_core::txn::proposal::RaftProposal,
    ) -> Result<
        coordinode_core::txn::proposal::ProposalOutcome,
        coordinode_core::txn::proposal::ProposalError,
    > {
        if self.closed.load(Ordering::Acquire) {
            self.held
                .send(())
                .expect("the test waits for the held proposal");
            self.release
                .lock()
                .recv()
                .expect("the test releases the held proposal");
        }
        self.inner.propose_and_wait(proposal)
    }
}

/// A heartbeat that is being written is still a sign of life: until the
/// write lands, the consumer stays listed and its checkpoint keeps holding the
/// retention floor, though neither the buffer nor the stored entry shows it.
#[test]
fn a_heartbeat_in_flight_keeps_its_consumer_live() {
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new());
    let engine = Arc::new(
        StorageEngine::open_with_oracle(
            &StorageConfig::with_endpoints(vec![EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            )]),
            oracle,
        )
        .expect("open engine"),
    );
    let (held_tx, held) = std::sync::mpsc::sync_channel(1);
    let (release, release_rx) = std::sync::mpsc::channel();
    let pipeline = Arc::new(GatedPipeline {
        inner: coordinode_raft::proposal::OwnedLocalProposalPipeline::new(&engine),
        closed: std::sync::atomic::AtomicBool::new(false),
        held: held_tx,
        release: parking_lot::Mutex::new(release_rx),
    });
    let clock = Arc::new(ManualClock::new(1_000));
    let reg = ShardConsumerRegistry::new(
        Arc::clone(&engine),
        Arc::clone(&pipeline) as Arc<dyn ProposalPipeline>,
        Arc::new(ProposalIdGenerator::with_base(1u64 << 48)),
        clock.clone(),
    );
    let h = reg
        .register(registration("reader", TopologyScope::Cluster, 2_000))
        .expect("register");
    let floor = reg.shard_floor();
    assert_ne!(floor, u64::MAX, "the reader holds the floor");

    reg.core.batching_on.store(true, Ordering::Release);
    clock.set(2_500);
    reg.heartbeat(&h).expect("buffer heartbeat");
    pipeline.closed.store(true, Ordering::Release);
    let flushing = {
        let core = Arc::clone(&reg.core);
        std::thread::spawn(move || core.flush_pending_heartbeats())
    };
    held.recv().expect("the flush proposes");

    // Past the stored heartbeat's TTL (1000 + 2000), inside the one being
    // written (2500 + 2000).
    clock.set(4_000);
    assert!(
        reg.list_consumers()
            .iter()
            .any(|c| c.consumer_id == "reader"),
        "a consumer whose heartbeat is being written was dropped from the list"
    );
    assert_eq!(
        reg.core.recompute_floor().expect("floor"),
        floor,
        "a consumer whose heartbeat is being written stopped holding the floor"
    );

    pipeline.closed.store(false, Ordering::Release);
    release.send(()).expect("release");
    flushing.join().expect("flush thread").expect("flush");
    assert!(
        reg.core.pending_hb.lock().is_empty(),
        "a written heartbeat leaves the buffer"
    );
    assert_eq!(
        reg.list_consumers()
            .iter()
            .find(|c| c.consumer_id == "reader")
            .map(|c| c.last_heartbeat_ts_ms),
        Some(2_500),
        "the heartbeat was written"
    );
}

/// A pipeline whose every proposal takes `delay`, as a slow fsync does.
struct SlowPipeline {
    inner: coordinode_raft::proposal::OwnedLocalProposalPipeline,
    delay: Duration,
}

impl ProposalPipeline for SlowPipeline {
    fn propose_and_wait(
        &self,
        proposal: &coordinode_core::txn::proposal::RaftProposal,
    ) -> Result<
        coordinode_core::txn::proposal::ProposalOutcome,
        coordinode_core::txn::proposal::ProposalError,
    > {
        std::thread::sleep(self.delay);
        self.inner.propose_and_wait(proposal)
    }
}

/// A reader that keeps heartbeating stays registered when every registry
/// write is slow and the background service shares a single-thread runtime
/// with the reader's stream, as a change stream does.
#[tokio::test]
async fn a_slow_flush_does_not_starve_the_heartbeats_it_persists() {
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new());
    let engine = Arc::new(
        StorageEngine::open_with_oracle(
            &StorageConfig::with_endpoints(vec![EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            )]),
            oracle,
        )
        .expect("open engine"),
    );
    let pipeline: Arc<dyn ProposalPipeline> = Arc::new(SlowPipeline {
        inner: coordinode_raft::proposal::OwnedLocalProposalPipeline::new(&engine),
        delay: Duration::from_millis(300),
    });
    let reg = ShardConsumerRegistry::new(
        Arc::clone(&engine),
        pipeline,
        Arc::new(ProposalIdGenerator::with_base(1u64 << 48)),
        Arc::new(SystemClock),
    );
    // Shorter than one write: the reader stays only by the heartbeats that
    // reach the buffer while a write is in flight, which a service holding
    // the runtime thread through the write never lets through. A heartbeat
    // every 50 ms leaves several of them inside each 300 ms write, a margin
    // that holds on a loaded machine.
    let ttl_ms = 200;
    let h = reg
        .register(registration("reader", TopologyScope::Cluster, ttl_ms))
        .expect("register");
    let bg = reg.start_background(BackgroundConfig {
        heartbeat_window_ms: 20,
        eviction_interval_ms: 50,
    });

    // The reader's stream: a heartbeat every 50 ms for several TTLs.
    let beating = {
        let reg = reg.clone();
        tokio::spawn(async move {
            let until = tokio::time::Instant::now() + Duration::from_millis(3 * ttl_ms);
            while tokio::time::Instant::now() < until {
                reg.heartbeat(&h).expect("heartbeat");
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        })
    };
    beating.await.expect("heartbeats");

    assert!(
        reg.list_consumers()
            .iter()
            .any(|c| c.consumer_id == "reader"),
        "a reader that kept heartbeating was evicted"
    );
    bg.shutdown().await;
}

/// A pipeline that runs a one-shot hook inside the next proposal: what
/// happens elsewhere while a registry write is in flight.
struct HookPipeline {
    inner: coordinode_raft::proposal::OwnedLocalProposalPipeline,
    hook: Mutex<Option<Box<dyn FnOnce() + Send>>>,
}

impl ProposalPipeline for HookPipeline {
    fn propose_and_wait(
        &self,
        proposal: &coordinode_core::txn::proposal::RaftProposal,
    ) -> Result<
        coordinode_core::txn::proposal::ProposalOutcome,
        coordinode_core::txn::proposal::ProposalError,
    > {
        let hook = self.hook.lock().take();
        if let Some(hook) = hook {
            hook();
        }
        self.inner.propose_and_wait(proposal)
    }
}

/// The sweep persists buffered heartbeats before judging expiry, and that
/// write can take longer than a TTL. A heartbeat that arrives meanwhile is
/// still only in the buffer: the reader is alive and must not be evicted.
#[test]
fn a_heartbeat_buffered_during_the_sweep_keeps_the_reader() {
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new());
    let engine = Arc::new(
        StorageEngine::open_with_oracle(
            &StorageConfig::with_endpoints(vec![EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            )]),
            oracle,
        )
        .expect("open engine"),
    );
    let pipeline = Arc::new(HookPipeline {
        inner: coordinode_raft::proposal::OwnedLocalProposalPipeline::new(&engine),
        hook: Mutex::new(None),
    });
    let clock = Arc::new(ManualClock::new(1_000));
    let reg = ShardConsumerRegistry::new(
        Arc::clone(&engine),
        Arc::clone(&pipeline) as Arc<dyn ProposalPipeline>,
        Arc::new(ProposalIdGenerator::with_base(1u64 << 48)),
        Arc::clone(&clock) as Arc<dyn Clock>,
    );
    let h = reg
        .register(registration("reader", TopologyScope::Cluster, 400))
        .expect("register");
    let pinned = reg.shard_floor();
    reg.core.batching_on.store(true, Ordering::Release);
    reg.heartbeat(&h).expect("heartbeat");

    // While the sweep's flush is being written, a TTL passes and the reader
    // heartbeats again.
    *pipeline.hook.lock() = Some(Box::new({
        let (reg, h, clock) = (reg.clone(), h.clone(), Arc::clone(&clock));
        move || {
            clock.set(2_000);
            reg.heartbeat(&h).expect("heartbeat during the flush");
        }
    }));
    let swept = reg.core.sweep_evictions().expect("sweep");

    assert_eq!(
        swept.evicted, 0,
        "a reader with a buffered heartbeat was evicted"
    );
    assert_eq!(
        swept.next_expiry_ms,
        Some(2_000 + 400 + 1),
        "the next sweep is due when the buffered heartbeat runs out"
    );
    assert!(
        reg.core.read_entry("reader").expect("read").is_some(),
        "the reader's registration is gone"
    );
    assert_eq!(
        reg.shard_floor(),
        pinned,
        "the live reader stopped pinning retention"
    );
}

/// EE enable path: `with_topology_scopes()` accepts `dc` / `rack` scopes
/// that CE rejects (complements `ce_rejects_dc_and_rack_scopes`).
#[tokio::test(flavor = "multi_thread")]
async fn ee_topology_scopes_accept_dc_and_rack() {
    let clock = Arc::new(ManualClock::new(0));
    let (reg, _engine, node, _dir) = registry_with_clock(clock).await;
    let reg = reg.with_topology_scopes(); // EE
    reg.register(registration("dc-sink", TopologyScope::Dc("eu".into()), 0))
        .expect("EE accepts dc scope");
    reg.register(registration(
        "rack-sink",
        TopologyScope::Rack("r1".into()),
        0,
    ))
    .expect("EE accepts rack scope");
    assert_eq!(reg.list_consumers().len(), 2);
    node.shutdown().await.expect("shutdown");
}

/// The time-travel window belongs to the engine, and the registry never
/// widens the watermark past it: with a tiny engine window and no
/// consumers, the watermark sits exactly `window` below the current seqno
/// after the registry republishes its (empty) consumer floor.
#[tokio::test(flavor = "multi_thread")]
async fn engine_window_governs_when_no_consumer_is_registered() {
    let clock = Arc::new(ManualClock::new(0));
    let (reg, engine, node, _dir) = registry_with_clock(clock).await;
    // 1 ms window. A consumer at 0 pins the watermark there; once it leaves,
    // the registry republishes an empty floor (`u64::MAX`) and the engine's
    // window is what remains: watermark = snapshot - 1_000 µs.
    engine.set_retention_window(Duration::from_millis(1));
    let h = reg
        .register(registration("probe", TopologyScope::Cluster, 0))
        .expect("register");
    assert_eq!(engine.gc_watermark(), 0);
    reg.unregister(h).expect("unregister");
    assert_eq!(engine.gc_watermark(), engine.snapshot() - 1_000);
    node.shutdown().await.expect("shutdown");
}

/// `InitialSeqno` resolution: `FromEarliestRetained` → 0 (replay all);
/// `FromNow` → the current open seqno (only future changes).
#[tokio::test(flavor = "multi_thread")]
async fn initial_seqno_from_now_and_earliest_resolve_correctly() {
    let clock = Arc::new(ManualClock::new(0));
    let (reg, engine, node, _dir) = registry_with_clock(clock).await;

    reg.register(ConsumerRegistration {
        initial_seqno: InitialSeqno::FromEarliestRetained,
        ..registration("replay-all", TopologyScope::Cluster, 0)
    })
    .expect("register earliest");
    assert_eq!(
        reg.shard_floor(),
        0,
        "FromEarliestRetained pins the floor at 0"
    );

    // FromNow on a second registry over the same engine: checkpoint = now.
    let pipeline: Arc<dyn ProposalPipeline> =
        Arc::new(RaftProposalPipeline::new(Arc::clone(node.raft())));
    let reg2 = ShardConsumerRegistry::new(
        Arc::clone(&engine),
        pipeline,
        Arc::new(ProposalIdGenerator::with_base(9u64 << 48)),
        Arc::new(ManualClock::new(0)),
    );
    let before = engine.snapshot();
    let h = reg2
        .register(ConsumerRegistration {
            initial_seqno: InitialSeqno::FromNow,
            ..registration("from-now", TopologyScope::Cluster, 0)
        })
        .expect("register from-now");
    let cp = reg2.check_retention(&h).expect("recorded checkpoint");
    assert!(
        cp >= before,
        "FromNow checkpoint ({cp}) starts at/after the open seqno at registration ({before})"
    );

    node.shutdown().await.expect("shutdown");
}

/// Eager heartbeat path (no background service): `heartbeat` validates the
/// consumer and writes `last_heartbeat_ts` immediately (no buffering).
#[tokio::test(flavor = "multi_thread")]
async fn eager_heartbeat_writes_immediately() {
    let clock = Arc::new(ManualClock::new(1_000));
    let (reg, _engine, node, _dir) = registry_with_clock(clock.clone()).await;
    let h = reg
        .register(registration("hb", TopologyScope::Cluster, 0))
        .expect("register");

    // No start_background → eager path. Advance clock, heartbeat, observe
    // the persisted timestamp move with no flush window.
    clock.set(9_000);
    reg.heartbeat(&h).expect("eager heartbeat");
    let listed = reg.list_consumers();
    assert_eq!(
        listed[0].last_heartbeat_ts_ms, 9_000,
        "eager heartbeat persisted at once"
    );

    // Heartbeat on an unknown consumer errors (eager path validates).
    assert!(matches!(
        reg.heartbeat(&RegisteredHandle::new("ghost")),
        Err(RegistryError::UnknownConsumer(_))
    ));
    node.shutdown().await.expect("shutdown");
}
