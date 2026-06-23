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

/// S4b: with the background service running, heartbeats buffer and flush
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

    // Wait for at least one flush window.
    tokio::time::sleep(Duration::from_millis(120)).await;

    let listed = reg.list_consumers();
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
/// proposal and lifts the floor it was pinning.
#[tokio::test(flavor = "multi_thread")]
async fn eviction_sweep_removes_expired_and_lifts_floor() {
    let clock = Arc::new(ManualClock::new(1_000));
    let (reg, _engine, node, _dir) = registry_with_clock(clock.clone()).await;

    reg.register(ConsumerRegistration {
        initial_seqno: InitialSeqno::At(10),
        ..registration("doomed", TopologyScope::Cluster, 2_000)
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

    // Advance past the doomed consumer's TTL (last hb 1000 + 2000).
    clock.set(4_000);
    tokio::time::sleep(Duration::from_millis(120)).await;

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

/// `with_retention_window_us` actually narrows the GC window: with a tiny
/// window and no consumers, the engine watermark sits just below the
/// current seqno (not 7 days back as the default would put it).
#[tokio::test(flavor = "multi_thread")]
async fn custom_retention_window_narrows_gc_watermark() {
    let clock = Arc::new(ManualClock::new(0));
    let (reg, engine, node, _dir) = registry_with_clock(clock).await;
    // 1 ms window (in µs). No consumers → watermark = snapshot - 1_000.
    let reg = reg.with_retention_window_us(1_000);
    let _ = reg.shard_floor(); // keep `reg` alive past the builder
    let snap = engine.snapshot();
    let wm = engine.gc_watermark();
    let behind = snap.saturating_sub(wm);
    assert!(
        behind < DEFAULT_RETENTION_WINDOW_US / 2,
        "tiny window must keep the watermark near `now` ({behind} µs behind), \
         not the default 7-day span"
    );
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
