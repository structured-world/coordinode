use super::*;
use crate::registry::types::{ConsumerKind, ValidatedRetentionBounds};
use coordinode_core::txn::proposal::RaftProposal;
use coordinode_raft::cluster::RaftNode;
use coordinode_raft::proposal::RaftProposalPipeline;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};

/// Test clock the suite advances by hand.
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

/// A source whose positions, ages and sizes the test sets. One space serves
/// every kind.
#[derive(Default)]
struct FakeSource {
    head: AtomicU64,
    first: AtomicU64,
    /// Clock ms each position was produced at.
    produced: Mutex<HashMap<u64, u64>>,
    /// Bytes required from any position.
    bytes: AtomicU64,
    unaccounted: AtomicBool,
    /// Whether the source is out of room for new retention.
    pressured: AtomicBool,
    /// The last log floor the registry released below.
    released: AtomicU64,
}

impl FakeSource {
    fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }
    fn set_head(&self, head: u64) {
        self.head.store(head, Ordering::Release);
    }
    fn set_first(&self, first: u64) {
        self.first.store(first, Ordering::Release);
    }
    fn produced(&self, position: u64, at_ms: u64) {
        self.produced.lock().insert(position, at_ms);
    }
    fn set_bytes(&self, bytes: u64) {
        self.bytes.store(bytes, Ordering::Release);
    }
    fn set_pressured(&self, pressured: bool) {
        self.pressured.store(pressured, Ordering::Release);
    }
}

impl RetentionSource for FakeSource {
    fn head(&self, _: ConsumerKind) -> u64 {
        self.head.load(Ordering::Acquire)
    }
    fn first_retained(&self, _: ConsumerKind) -> u64 {
        self.first.load(Ordering::Acquire)
    }
    fn accounts(&self, _: ConsumerKind) -> bool {
        !self.unaccounted.load(Ordering::Acquire)
    }
    fn produced_at_ms(&self, _: ConsumerKind, position: u64) -> Option<u64> {
        self.produced.lock().get(&position).copied()
    }
    fn retained_bytes_from(&self, _: ConsumerKind, _: u64) -> Option<u64> {
        Some(self.bytes.load(Ordering::Acquire))
    }
    fn admits(&self, _: ConsumerKind) -> bool {
        !self.pressured.load(Ordering::Acquire)
    }
    fn release_log_below(&self, position: u64) {
        self.released.store(position, Ordering::Release);
    }
}

struct Fixture {
    reg: ShardConsumerRegistry,
    engine: Arc<StorageEngine>,
    node: Arc<RaftNode>,
    source: Arc<FakeSource>,
    _dir: tempfile::TempDir,
}

async fn fixture(clock: Arc<dyn Clock>) -> Fixture {
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
    let source = FakeSource::new();
    let reg = registry_over(&engine, &node, clock, Arc::clone(&source), 1);
    Fixture {
        reg,
        engine,
        node,
        source,
        _dir: dir,
    }
}

/// Another registry over the same replicated keyspace: another member, or
/// this one after a restart.
fn registry_over(
    engine: &Arc<StorageEngine>,
    node: &Arc<RaftNode>,
    clock: Arc<dyn Clock>,
    source: Arc<FakeSource>,
    id_space: u64,
) -> ShardConsumerRegistry {
    let pipeline: Arc<dyn ProposalPipeline> =
        Arc::new(RaftProposalPipeline::new(Arc::clone(node.raft())));
    ShardConsumerRegistry::new(
        Arc::clone(engine),
        pipeline,
        Arc::new(ProposalIdGenerator::with_base(id_space << 48)),
        clock,
        source,
    )
}

fn bounded(lag_ms: u64, bytes: u64, liveness_ms: Option<u64>) -> ConsumerRetentionPolicy {
    ConsumerRetentionPolicy::Bounded(
        ValidatedRetentionBounds::new(lag_ms, bytes, liveness_ms).expect("bounds"),
    )
}

/// A seqno-space registration starting at `at`.
fn registration(id: &str, at: u64, retention: ConsumerRetentionPolicy) -> ConsumerRegistration {
    ConsumerRegistration {
        consumer_id: id.to_string(),
        kind: ConsumerKind::LsmStateDelta,
        scope: TopologyScope::Cluster,
        initial_seqno: InitialSeqno::At(at),
        retention,
    }
}

fn state_of(reg: &ShardConsumerRegistry, id: &str) -> RegistrationState {
    reg.list_consumers()
        .into_iter()
        .find(|c| c.consumer_id == id)
        .expect("listed")
        .state
}

fn ended_for(reg: &ShardConsumerRegistry, id: &str) -> Option<TerminalReason> {
    match state_of(reg, id) {
        RegistrationState::Live => None,
        RegistrationState::Terminated { reason, .. } => Some(reason),
    }
}

/// The floor is the slowest live consumer; checkpoints only advance; a
/// cancelled consumer stops holding the floor but keeps its record.
#[tokio::test(flavor = "multi_thread")]
async fn register_checkpoint_cancel_drive_floor() {
    let f = fixture(Arc::new(ManualClock::new(1_000))).await;
    assert_eq!(f.reg.shard_floor(), u64::MAX);

    let h1 = f
        .reg
        .register(registration("c1", 100, ConsumerRetentionPolicy::Strict))
        .expect("register c1");
    let h2 = f
        .reg
        .register(registration("c2", 250, ConsumerRetentionPolicy::Strict))
        .expect("register c2");
    assert_eq!(f.reg.shard_floor(), 100, "floor is the slowest consumer");
    assert_eq!((h1.incarnation(), h2.incarnation()), (1, 1));

    f.reg.checkpoint(&h1, 300).expect("checkpoint c1");
    assert_eq!(f.reg.shard_floor(), 250);
    f.reg.checkpoint(&h2, 10).expect("stale checkpoint c2");
    assert_eq!(
        f.reg.shard_floor(),
        250,
        "a stale checkpoint must not rewind the floor"
    );

    f.reg.unregister(h2).expect("cancel c2");
    assert_eq!(
        f.reg.shard_floor(),
        300,
        "a cancelled consumer holds nothing"
    );
    assert_eq!(ended_for(&f.reg, "c2"), Some(TerminalReason::Cancelled));
    assert_eq!(ended_for(&f.reg, "c1"), None);

    f.reg.unregister(h1).expect("cancel c1");
    assert_eq!(
        f.reg.shard_floor(),
        u64::MAX,
        "no live consumer, no constraint"
    );
    f.node.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread")]
async fn empty_id_and_ce_topology_scopes_are_refused() {
    let f = fixture(Arc::new(ManualClock::new(0))).await;
    assert!(matches!(
        f.reg
            .register(registration("", 0, ConsumerRetentionPolicy::Strict))
            .unwrap_err(),
        RegistryError::EmptyConsumerId
    ));
    for scope in [
        TopologyScope::Dc("eu".into()),
        TopologyScope::Rack("r1".into()),
    ] {
        let err = f
            .reg
            .register(ConsumerRegistration {
                scope,
                ..registration("c", 0, ConsumerRetentionPolicy::Strict)
            })
            .unwrap_err();
        assert!(matches!(err, RegistryError::UnsupportedScope(_)), "{err:?}");
    }
    f.node.shutdown().await.expect("shutdown");
}

/// EE enable path: `with_topology_scopes()` accepts the scopes CE refuses.
#[tokio::test(flavor = "multi_thread")]
async fn ee_topology_scopes_accept_dc_and_rack() {
    let f = fixture(Arc::new(ManualClock::new(0))).await;
    let reg = registry_over(
        &f.engine,
        &f.node,
        Arc::new(ManualClock::new(0)),
        Arc::clone(&f.source),
        2,
    )
    .with_topology_scopes();
    for (id, scope) in [
        ("dc-sink", TopologyScope::Dc("eu".into())),
        ("rack-sink", TopologyScope::Rack("r1".into())),
    ] {
        reg.register(ConsumerRegistration {
            scope,
            ..registration(id, 0, ConsumerRetentionPolicy::Strict)
        })
        .expect("EE accepts the scope");
    }
    f.node.shutdown().await.expect("shutdown");
}

/// A handle of an id never registered is refused, not treated as a no-op.
#[tokio::test(flavor = "multi_thread")]
async fn a_handle_of_an_unknown_consumer_is_refused() {
    let f = fixture(Arc::new(ManualClock::new(0))).await;
    let phantom = RegisteredHandle::new("never-registered", 1);
    assert!(matches!(
        f.reg.checkpoint(&phantom, 5).unwrap_err(),
        RegistryError::UnknownConsumer(_)
    ));
    assert!(matches!(
        f.reg.heartbeat(&phantom).unwrap_err(),
        RegistryError::UnknownConsumer(_)
    ));
    assert!(matches!(
        f.reg.unregister(phantom).unwrap_err(),
        RegistryError::UnknownConsumer(_)
    ));
    assert!(matches!(
        f.reg.resume("never-registered", 1).unwrap_err(),
        RegistryError::UnknownConsumer(_)
    ));
    f.node.shutdown().await.expect("shutdown");
}

/// STRICT has no automatic expiry: a consumer silent for any length of time,
/// and far behind its source, keeps its protection.
#[tokio::test(flavor = "multi_thread")]
async fn a_strict_consumer_survives_missing_heartbeats_and_any_lag() {
    let clock = Arc::new(ManualClock::new(1_000));
    let f = fixture(clock.clone()).await;
    f.reg
        .register(registration("strict", 5, ConsumerRetentionPolicy::Strict))
        .expect("register");
    f.source.set_head(1_000);
    f.source.produced(5, 1_000);
    f.source.set_bytes(u64::MAX / 2);

    clock.set(1_000 + 365 * 24 * 3_600 * 1_000);
    let swept = f.reg.core.sweep_evictions().expect("sweep");
    assert_eq!(swept.evicted, 0);
    assert_eq!(ended_for(&f.reg, "strict"), None);
    assert_eq!(f.reg.shard_floor(), 5, "it still holds the floor");
    f.node.shutdown().await.expect("shutdown");
}

/// BOUNDED liveness: a consumer silent past its timeout ends, durably, with
/// that reason; the floor it held lifts.
#[tokio::test(flavor = "multi_thread")]
async fn a_silent_bounded_consumer_ends_for_liveness() {
    let clock = Arc::new(ManualClock::new(1_000));
    let f = fixture(clock.clone()).await;
    f.reg
        .register(registration(
            "silent",
            50,
            bounded(60_000, 1 << 30, Some(5_000)),
        ))
        .expect("register");
    f.reg
        .register(registration("strict", 900, ConsumerRetentionPolicy::Strict))
        .expect("register strict");
    assert_eq!(f.reg.shard_floor(), 50);

    clock.set(6_000);
    assert_eq!(
        f.reg.core.sweep_evictions().expect("sweep").evicted,
        0,
        "at the boundary"
    );
    clock.set(6_001);
    assert_eq!(f.reg.core.sweep_evictions().expect("sweep").evicted, 1);
    assert_eq!(
        ended_for(&f.reg, "silent"),
        Some(TerminalReason::LivenessExpired)
    );
    assert_eq!(
        f.reg.shard_floor(),
        900,
        "the ended consumer no longer holds retention"
    );
    f.node.shutdown().await.expect("shutdown");
}

/// BOUNDED progress lag: the oldest unacknowledged work outgrowing the limit
/// ends the registration, and heartbeats do not reset that age.
#[tokio::test(flavor = "multi_thread")]
async fn heartbeats_do_not_reset_progress_lag() {
    let clock = Arc::new(ManualClock::new(10_000));
    let f = fixture(clock.clone()).await;
    let h = f
        .reg
        .register(registration(
            "lagging",
            7,
            bounded(1_000, 1 << 30, Some(60_000)),
        ))
        .expect("register");
    f.source.set_head(20);
    f.source.produced(7, 10_000);

    for t in [10_500, 10_900, 11_000] {
        clock.set(t);
        f.reg.heartbeat(&h).expect("heartbeat");
        assert_eq!(
            f.reg.core.sweep_evictions().expect("sweep").evicted,
            0,
            "at {t}"
        );
    }
    clock.set(11_001);
    f.reg.heartbeat(&h).expect("heartbeat");
    assert_eq!(f.reg.core.sweep_evictions().expect("sweep").evicted, 1);
    assert_eq!(
        ended_for(&f.reg, "lagging"),
        Some(TerminalReason::ProgressLagExceeded)
    );
    f.node.shutdown().await.expect("shutdown");
}

/// An idle source leaves nothing unacknowledged: a consumer caught up with it
/// is not lagging, however long ago its checkpoint last moved.
#[tokio::test(flavor = "multi_thread")]
async fn a_caught_up_consumer_of_an_idle_source_is_not_lagging() {
    let clock = Arc::new(ManualClock::new(0));
    let f = fixture(clock.clone()).await;
    f.reg
        .register(registration("idle", 42, bounded(1_000, 1 << 30, None)))
        .expect("register");
    f.source.set_head(42);
    f.source.produced(42, 0);

    clock.set(1_000_000);
    assert_eq!(f.reg.core.sweep_evictions().expect("sweep").evicted, 0);
    assert_eq!(ended_for(&f.reg, "idle"), None);
    f.node.shutdown().await.expect("shutdown");
}

/// BOUNDED bytes: once the material the checkpoint requires exceeds the
/// limit, the registration ends.
#[tokio::test(flavor = "multi_thread")]
async fn a_consumer_requiring_too_many_bytes_ends() {
    let clock = Arc::new(ManualClock::new(0));
    let f = fixture(clock.clone()).await;
    f.reg
        .register(registration("heavy", 3, bounded(u64::MAX, 4_096, None)))
        .expect("register");
    f.source.set_bytes(4_096);
    assert_eq!(
        f.reg.core.sweep_evictions().expect("sweep").evicted,
        0,
        "at the limit"
    );
    f.source.set_bytes(4_097);
    assert_eq!(f.reg.core.sweep_evictions().expect("sweep").evicted, 1);
    assert_eq!(
        ended_for(&f.reg, "heavy"),
        Some(TerminalReason::RetainedBytesExceeded)
    );
    f.node.shutdown().await.expect("shutdown");
}

/// An ended incarnation refuses every call made through its handle, with the
/// reason and the last checkpoint; its checkpoint does not move.
#[tokio::test(flavor = "multi_thread")]
async fn an_ended_handle_is_refused_everywhere() {
    let clock = Arc::new(ManualClock::new(0));
    let f = fixture(clock).await;
    let h = f
        .reg
        .register(registration("gone", 10, ConsumerRetentionPolicy::Strict))
        .expect("register");
    f.reg.checkpoint(&h, 12).expect("checkpoint");
    f.reg.unregister(h.clone()).expect("cancel");

    let is_ended = |e: RegistryError| {
        matches!(
            e,
            RegistryError::Terminated {
                incarnation: 1,
                reason: TerminalReason::Cancelled,
                checkpoint: 12,
                ..
            }
        )
    };
    assert!(is_ended(f.reg.checkpoint(&h, 99).unwrap_err()));
    assert!(is_ended(f.reg.heartbeat(&h).unwrap_err()));
    assert!(is_ended(f.reg.check_retention(&h).unwrap_err()));
    assert!(is_ended(f.reg.resume("gone", 1).unwrap_err()));
    assert!(is_ended(f.reg.unregister(h).unwrap_err()));
    assert_eq!(
        f.reg
            .list_consumers()
            .iter()
            .find(|c| c.consumer_id == "gone")
            .map(|c| c.checkpoint_seqno),
        Some(12),
        "a refused checkpoint moved nothing"
    );
    f.node.shutdown().await.expect("shutdown");
}

/// A live id cannot be registered over; once it has ended, registering it
/// again starts a new incarnation and the old handle stays refused.
#[tokio::test(flavor = "multi_thread")]
async fn reregistering_an_ended_id_starts_a_new_incarnation() {
    let f = fixture(Arc::new(ManualClock::new(0))).await;
    let first = f
        .reg
        .register(registration("sink", 10, ConsumerRetentionPolicy::Strict))
        .expect("register");
    assert!(matches!(
        f.reg
            .register(registration("sink", 20, ConsumerRetentionPolicy::Strict))
            .unwrap_err(),
        RegistryError::AlreadyRegistered { incarnation: 1, .. }
    ));

    f.reg.unregister(first.clone()).expect("cancel");
    let second = f
        .reg
        .register(registration("sink", 30, ConsumerRetentionPolicy::Strict))
        .expect("register again");
    assert_eq!(second.incarnation(), 2);
    assert!(matches!(
        f.reg.checkpoint(&first, 40).unwrap_err(),
        RegistryError::StaleIncarnation {
            handle: 1,
            current: 2,
            ..
        }
    ));
    assert_eq!(f.reg.resume("sink", 2).expect("resume"), second);
    assert_eq!(
        f.reg.shard_floor(),
        30,
        "the new incarnation holds its own start"
    );
    f.node.shutdown().await.expect("shutdown");
}

/// A heartbeat buffered by an ended incarnation is dropped when flushed: it
/// is no sign of life of the incarnation registered after it.
#[tokio::test(flavor = "multi_thread")]
async fn a_late_heartbeat_of_an_ended_incarnation_keeps_nothing_alive() {
    let clock = Arc::new(ManualClock::new(1_000));
    let f = fixture(clock.clone()).await;
    let old = f
        .reg
        .register(registration("r", 0, ConsumerRetentionPolicy::Strict))
        .expect("register");
    f.reg.unregister(old.clone()).expect("cancel");
    let new = f
        .reg
        .register(registration(
            "r",
            0,
            bounded(u64::MAX, u64::MAX, Some(1_000)),
        ))
        .expect("register again");

    f.reg.core.batching_on.store(true, Ordering::Release);
    clock.set(5_000);
    f.reg.heartbeat(&old).expect("buffered without validation");
    f.reg.core.flush_pending_heartbeats().expect("flush");
    assert!(
        f.reg.core.pending_hb.lock().is_empty(),
        "the stale heartbeat was dropped"
    );
    assert_eq!(
        f.reg.core.sweep_evictions().expect("sweep").evicted,
        1,
        "the new incarnation, never heard from, ended for liveness"
    );
    assert!(matches!(
        f.reg.resume("r", new.incarnation()).unwrap_err(),
        RegistryError::Terminated {
            reason: TerminalReason::LivenessExpired,
            ..
        }
    ));
    f.node.shutdown().await.expect("shutdown");
}

/// A BOUNDED limit needs the source to measure what it bounds; one that
/// cannot is refused, never admitted with an unjudgeable bound.
#[tokio::test(flavor = "multi_thread")]
async fn bounded_is_refused_where_the_source_cannot_measure_it() {
    let f = fixture(Arc::new(ManualClock::new(0))).await;
    f.source.unaccounted.store(true, Ordering::Release);
    assert!(matches!(
        f.reg
            .register(registration("b", 0, bounded(1_000, 1_000, None)))
            .unwrap_err(),
        RegistryError::InvalidRetention(_)
    ));
    f.reg
        .register(registration("s", 0, ConsumerRetentionPolicy::Strict))
        .expect("STRICT needs no measure");
    f.node.shutdown().await.expect("shutdown");
}

/// A registration starting below what the source holds is refused: its
/// protection would start over history already gone.
#[tokio::test(flavor = "multi_thread")]
async fn registering_below_the_retained_source_is_refused() {
    let f = fixture(Arc::new(ManualClock::new(0))).await;
    f.source.set_first(100);
    assert!(matches!(
        f.reg
            .register(registration("late", 99, ConsumerRetentionPolicy::Strict))
            .unwrap_err(),
        RegistryError::RetentionLost {
            checkpoint: 99,
            floor: 100
        }
    ));
    let h = f
        .reg
        .register(ConsumerRegistration {
            initial_seqno: InitialSeqno::FromEarliestRetained,
            ..registration("earliest", 0, ConsumerRetentionPolicy::Strict)
        })
        .expect("from the earliest retained");
    assert_eq!(f.reg.check_retention(&h).expect("held"), 100);
    f.node.shutdown().await.expect("shutdown");
}

/// An acknowledgement and an expiry racing over one record: whichever lands
/// second decides against what the first left. Here the acknowledgement
/// lands after the sweep read the lagging record and before its terminal
/// write; the write finds the record moved, the sweep reads again and finds a
/// consumer no longer lagging.
#[test]
fn an_acknowledgement_racing_an_expiry_keeps_the_consumer() {
    let (reg, clock, source, _pipeline, _dir) = hooked_registry();
    let h = reg
        .register(registration("racer", 7, bounded(1_000, 1 << 30, None)))
        .expect("register");
    source.set_head(20);
    source.produced(7, 0);
    clock.set(5_000);

    *reg.core.before_commit.lock() = Some(Box::new({
        let (reg, h) = (reg.clone(), h.clone());
        move || {
            reg.checkpoint(&h, 20)
                .expect("the acknowledgement lands first")
        }
    }));
    let swept = reg.core.sweep_evictions().expect("sweep");
    assert_eq!(swept.evicted, 0, "the sweep decided on a stale read");
    assert_eq!(ended_for(&reg, "racer"), None);
    assert_eq!(reg.shard_floor(), 20, "the acknowledgement stands");
}

/// The other order: the expiry lands first, and the acknowledgement that
/// follows finds the incarnation ended and changes nothing.
#[test]
fn an_acknowledgement_after_an_expiry_is_refused() {
    let (reg, clock, source, _pipeline, _dir) = hooked_registry();
    let h = reg
        .register(registration("late-ack", 7, bounded(1_000, 1 << 30, None)))
        .expect("register");
    source.set_head(20);
    source.produced(7, 0);
    clock.set(5_000);
    assert_eq!(reg.core.sweep_evictions().expect("sweep").evicted, 1);
    assert!(matches!(
        reg.checkpoint(&h, 20).unwrap_err(),
        RegistryError::Terminated {
            reason: TerminalReason::ProgressLagExceeded,
            checkpoint: 7,
            ..
        }
    ));
    assert_eq!(reg.shard_floor(), u64::MAX);
}

/// A registry reopened over the same store (a restart, a new leader) finds
/// live registrations holding their floors and ended ones still refusing
/// their handles: nothing is resurrected.
#[tokio::test(flavor = "multi_thread")]
async fn a_reopened_registry_keeps_floors_and_terminal_fences() {
    let clock = Arc::new(ManualClock::new(1_000));
    let f = fixture(clock.clone()).await;
    let live = f
        .reg
        .register(registration("backup", 100, ConsumerRetentionPolicy::Strict))
        .expect("register");
    f.reg.checkpoint(&live, 300).expect("checkpoint");
    f.reg
        .register(ConsumerRegistration {
            kind: ConsumerKind::OplogEvents,
            ..registration("cdc", 50, ConsumerRetentionPolicy::Strict)
        })
        .expect("register oplog consumer");
    let ended = f
        .reg
        .register(registration("ended", 1, ConsumerRetentionPolicy::Strict))
        .expect("register");
    f.reg.unregister(ended.clone()).expect("cancel");

    let reopened = registry_over(&f.engine, &f.node, clock, Arc::clone(&f.source), 2);
    assert_eq!(reopened.shard_floor(), 300, "seqno floor recovered");
    assert_eq!(
        reopened.oplog_retention_floor(),
        50,
        "oplog floor recovered"
    );
    assert!(matches!(
        reopened.checkpoint(&ended, 5).unwrap_err(),
        RegistryError::Terminated { .. }
    ));
    assert_eq!(reopened.resume("backup", 1).expect("resume"), live);
    f.node.shutdown().await.expect("shutdown");
}

/// Floors are split by space: an oplog consumer never pulls the MVCC GC
/// watermark into Raft-index space, nor the reverse.
#[tokio::test(flavor = "multi_thread")]
async fn floors_are_split_by_consumer_space() {
    let f = fixture(Arc::new(ManualClock::new(1_000))).await;
    let oplog_h = f
        .reg
        .register(ConsumerRegistration {
            kind: ConsumerKind::OplogEvents,
            ..registration("cdc-sink", 42, ConsumerRetentionPolicy::Strict)
        })
        .expect("register oplog consumer");
    assert_eq!(f.reg.oplog_retention_floor(), 42);
    assert_eq!(f.reg.shard_floor(), u64::MAX);
    assert_ne!(f.engine.gc_watermark(), 42);

    let seqno_h = f
        .reg
        .register(registration(
            "backup",
            1_000,
            ConsumerRetentionPolicy::Strict,
        ))
        .expect("register seqno consumer");
    assert_eq!(f.reg.shard_floor(), 1_000);
    assert_eq!(f.engine.gc_watermark(), 1_000);
    assert_eq!(f.reg.oplog_retention_floor(), 42);

    f.reg.unregister(oplog_h).expect("cancel");
    f.reg.unregister(seqno_h).expect("cancel");
    assert_eq!(f.reg.oplog_retention_floor(), u64::MAX);
    assert_eq!(f.reg.shard_floor(), u64::MAX);
    f.node.shutdown().await.expect("shutdown");
}

/// A seqno consumer holds the engine GC watermark at its checkpoint; with
/// none, the engine's own window governs, never an unconstrained watermark.
#[tokio::test(flavor = "multi_thread")]
async fn consumer_floor_drives_engine_gc_watermark() {
    let f = fixture(Arc::new(ManualClock::new(1_000))).await;
    let h = f
        .reg
        .register(registration("cdc", 100, ConsumerRetentionPolicy::Strict))
        .expect("register");
    assert_eq!(f.engine.gc_watermark(), 100);
    f.reg.checkpoint(&h, 500).expect("advance");
    assert_eq!(f.engine.gc_watermark(), 500);
    f.reg.unregister(h).expect("cancel");
    let wm = f.engine.gc_watermark();
    assert_ne!(
        wm,
        u64::MAX,
        "an empty registry must not collapse to GC-everything"
    );
    assert!(wm > 500, "the time-travel window governs, got {wm}");
    f.node.shutdown().await.expect("shutdown");
}

/// When the engine has collected past a consumer's checkpoint, its read is
/// refused with RetentionLost; a protected consumer gets its checkpoint.
#[tokio::test(flavor = "multi_thread")]
async fn check_retention_surfaces_retention_lost() {
    let f = fixture(Arc::new(ManualClock::new(1_000))).await;
    let h = f
        .reg
        .register(registration("cdc", 100, ConsumerRetentionPolicy::Strict))
        .expect("register");
    assert_eq!(f.reg.check_retention(&h).expect("protected"), 100);
    f.engine.set_consumer_retention_floor(1_000);
    let lost = f.reg.check_retention(&h);
    assert!(
        matches!(
            lost,
            Err(RegistryError::RetentionLost { checkpoint: 100, floor }) if floor >= 1_000
        ),
        "got {lost:?}"
    );
    f.node.shutdown().await.expect("shutdown");
}

/// The time-travel window belongs to the engine: with a tiny window and no
/// consumer, the watermark sits exactly a window below the current seqno.
#[tokio::test(flavor = "multi_thread")]
async fn engine_window_governs_when_no_consumer_is_registered() {
    let f = fixture(Arc::new(ManualClock::new(0))).await;
    f.engine.set_retention_window(Duration::from_millis(1));
    let h = f
        .reg
        .register(registration("probe", 0, ConsumerRetentionPolicy::Strict))
        .expect("register");
    assert_eq!(f.engine.gc_watermark(), 0);
    f.reg.unregister(h).expect("cancel");
    assert_eq!(f.engine.gc_watermark(), f.engine.snapshot() - 1_000);
    f.node.shutdown().await.expect("shutdown");
}

/// `FromNow` starts at the source head, `FromEarliestRetained` at the first
/// retained position.
#[tokio::test(flavor = "multi_thread")]
async fn initial_positions_resolve_against_the_source() {
    let f = fixture(Arc::new(ManualClock::new(0))).await;
    f.source.set_first(11);
    f.source.set_head(77);
    let earliest = f
        .reg
        .register(ConsumerRegistration {
            initial_seqno: InitialSeqno::FromEarliestRetained,
            ..registration("all", 0, ConsumerRetentionPolicy::Strict)
        })
        .expect("register");
    let now = f
        .reg
        .register(ConsumerRegistration {
            initial_seqno: InitialSeqno::FromNow,
            ..registration("new-only", 0, ConsumerRetentionPolicy::Strict)
        })
        .expect("register");
    assert_eq!(f.reg.check_retention(&earliest).expect("held"), 11);
    assert_eq!(f.reg.check_retention(&now).expect("held"), 77);
    f.node.shutdown().await.expect("shutdown");
}

/// With the background service, heartbeats buffer and flush in one
/// coalesced write.
#[tokio::test(flavor = "multi_thread")]
async fn batched_heartbeats_flush_and_refresh_liveness() {
    let clock = Arc::new(ManualClock::new(1_000));
    let f = fixture(clock.clone()).await;
    let h = f
        .reg
        .register(registration(
            "hb",
            0,
            bounded(u64::MAX, u64::MAX, Some(10_000)),
        ))
        .expect("register");
    let bg = f.reg.start_background(BackgroundConfig {
        heartbeat_window_ms: 30,
        eviction_interval_ms: 100_000,
    });
    clock.set(4_000);
    for _ in 0..5 {
        f.reg.heartbeat(&h).expect("buffer heartbeat");
    }
    let until = tokio::time::Instant::now() + Duration::from_secs(5);
    let heard = || {
        f.reg
            .list_consumers()
            .first()
            .map(|c| c.last_heartbeat_ts_ms)
    };
    while heard() != Some(4_000) && tokio::time::Instant::now() < until {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(
        heard(),
        Some(4_000),
        "the coalesced flush wrote the buffered time"
    );
    bg.shutdown().await;
    f.node.shutdown().await.expect("shutdown");
}

/// The background sweep ends a BOUNDED consumer once its liveness runs out,
/// with nothing but the clock to schedule it.
#[tokio::test(flavor = "multi_thread")]
async fn the_background_sweep_ends_a_silent_bounded_consumer() {
    let f = fixture(Arc::new(SystemClock)).await;
    f.reg
        .register(registration(
            "doomed",
            10,
            bounded(u64::MAX, u64::MAX, Some(300)),
        ))
        .expect("register doomed");
    f.reg
        .register(registration(
            "survivor",
            500,
            ConsumerRetentionPolicy::Strict,
        ))
        .expect("register survivor");
    let bg = f.reg.start_background(BackgroundConfig {
        heartbeat_window_ms: 100_000,
        eviction_interval_ms: 30,
    });
    // The terminal record commits first and the floor moves after it (the
    // floor never moves ahead of the record), so a reader can see the ended
    // consumer before the new floor: wait for both.
    let until = tokio::time::Instant::now() + Duration::from_secs(10);
    while (ended_for(&f.reg, "doomed").is_none() || f.reg.shard_floor() != 500)
        && tokio::time::Instant::now() < until
    {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(
        ended_for(&f.reg, "doomed"),
        Some(TerminalReason::LivenessExpired)
    );
    assert_eq!(f.reg.shard_floor(), 500);
    bg.shutdown().await;
    f.node.shutdown().await.expect("shutdown");
}

/// A reader watching its registration hears of writes to its own record as
/// they apply (here, its cancellation) and not of writes to others'.
#[tokio::test(flavor = "multi_thread")]
async fn a_watch_hears_of_its_own_record_only() {
    let f = fixture(Arc::new(ManualClock::new(1_000))).await;
    let bg = f.reg.start_background(BackgroundConfig {
        heartbeat_window_ms: 100_000,
        eviction_interval_ms: 100_000,
    });
    let mine = f
        .reg
        .register(registration("mine", 10, ConsumerRetentionPolicy::Strict))
        .expect("register mine");
    let other = f
        .reg
        .register(registration("other", 10, ConsumerRetentionPolicy::Strict))
        .expect("register other");
    let mut watch = f.reg.watch(&mine);
    assert!(watch.changed(), "the first call");

    f.reg.checkpoint(&other, 20).expect("acknowledge other");
    // The other record's write applies and is relayed before this one.
    f.reg.unregister(mine).expect("cancel mine");
    let until = tokio::time::Instant::now() + Duration::from_secs(10);
    let mut heard = false;
    while !heard && tokio::time::Instant::now() < until {
        heard = watch.changed();
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(heard, "the cancellation was relayed");
    assert!(!watch.changed(), "nothing more applied to it");
    f.reg
        .checkpoint(&other, 30)
        .expect("acknowledge other again");
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(
        !watch.changed(),
        "another record's write does not concern it"
    );
    bg.shutdown().await;
    assert!(watch.changed(), "without the relay every call is a change");
    f.node.shutdown().await.expect("shutdown");
}

/// With the background service running, a registration that can lower the
/// floor still publishes it before returning: the history its position
/// needs must not be collected while a sweep is far away (100 s here).
#[tokio::test(flavor = "multi_thread")]
async fn a_lowering_registration_publishes_the_floor_before_returning() {
    let f = fixture(Arc::new(ManualClock::new(1_000))).await;
    let bg = f.reg.start_background(BackgroundConfig {
        heartbeat_window_ms: 100_000,
        eviction_interval_ms: 100_000,
    });
    f.reg
        .register(registration("late", 250, ConsumerRetentionPolicy::Strict))
        .expect("register late");
    assert_eq!(f.reg.shard_floor(), 250);
    f.reg
        .register(registration("early", 100, ConsumerRetentionPolicy::Strict))
        .expect("register early");
    assert_eq!(f.reg.shard_floor(), 100, "lowered before the call returned");
    bg.shutdown().await;
    f.node.shutdown().await.expect("shutdown");
}

/// With the background service running, an acknowledgement only raises the
/// floor, so the sweep its applied write triggers publishes it.
#[tokio::test(flavor = "multi_thread")]
async fn an_acknowledgement_raises_the_floor_through_the_sweep() {
    let f = fixture(Arc::new(ManualClock::new(1_000))).await;
    let bg = f.reg.start_background(BackgroundConfig {
        heartbeat_window_ms: 100_000,
        eviction_interval_ms: 30,
    });
    let early = f
        .reg
        .register(registration("early", 100, ConsumerRetentionPolicy::Strict))
        .expect("register early");
    f.reg
        .register(registration("late", 250, ConsumerRetentionPolicy::Strict))
        .expect("register late");
    f.reg.checkpoint(&early, 300).expect("acknowledge");
    let until = tokio::time::Instant::now() + Duration::from_secs(10);
    while f.reg.shard_floor() != 250 && tokio::time::Instant::now() < until {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(f.reg.shard_floor(), 250);
    bg.shutdown().await;
    f.node.shutdown().await.expect("shutdown");
}

/// The source keeps what it needs to tell the age of acknowledged log
/// positions; the registry tells it the lowest one any live log consumer can
/// still be asked about, so it can let go of the rest (and of all of it once
/// no log consumer is left).
#[tokio::test(flavor = "multi_thread")]
async fn the_source_learns_the_lowest_acknowledged_log_position() {
    let f = fixture(Arc::new(ManualClock::new(1_000))).await;
    f.source.set_head(1_000);
    let log = |id: &str, at: u64| ConsumerRegistration {
        consumer_id: id.to_string(),
        kind: ConsumerKind::OplogEvents,
        scope: TopologyScope::Cluster,
        initial_seqno: InitialSeqno::At(at),
        retention: ConsumerRetentionPolicy::Strict,
    };
    let early = f.reg.register(log("early", 100)).expect("register early");
    let late = f.reg.register(log("late", 250)).expect("register late");
    assert_eq!(f.source.released.load(Ordering::Acquire), 100);

    f.reg.checkpoint(&early, 300).expect("acknowledge");
    f.reg.core.sweep_evictions().expect("sweep");
    assert_eq!(f.source.released.load(Ordering::Acquire), 250);

    f.reg.unregister(early).expect("cancel early");
    f.reg.unregister(late).expect("cancel late");
    f.reg.core.sweep_evictions().expect("sweep");
    assert_eq!(f.source.released.load(Ordering::Acquire), u64::MAX);
    f.node.shutdown().await.expect("shutdown");
}

/// A registration written through another member's registry reaches this
/// member's floor as it applies.
#[tokio::test(flavor = "multi_thread")]
async fn a_registration_applied_from_elsewhere_moves_the_floor() {
    let clock = Arc::new(ManualClock::new(1_000));
    let f = fixture(clock.clone()).await;
    let bg = f.reg.start_background(BackgroundConfig {
        heartbeat_window_ms: 100_000,
        eviction_interval_ms: 30,
    });
    let elsewhere = registry_over(&f.engine, &f.node, clock, Arc::clone(&f.source), 3);
    elsewhere
        .register(registration("remote", 77, ConsumerRetentionPolicy::Strict))
        .expect("register elsewhere");
    let until = tokio::time::Instant::now() + Duration::from_secs(5);
    while f.reg.shard_floor() != 77 && tokio::time::Instant::now() < until {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(f.reg.shard_floor(), 77);
    bg.shutdown().await;
    f.node.shutdown().await.expect("shutdown");
}

/// A flush whose write fails keeps the heartbeats it took: they were a
/// consumer's sign of life.
#[tokio::test(flavor = "multi_thread")]
async fn a_failed_flush_keeps_its_heartbeats_buffered() {
    let clock = Arc::new(ManualClock::new(1_000));
    let f = fixture(clock.clone()).await;
    let h = f
        .reg
        .register(registration(
            "reader",
            0,
            bounded(u64::MAX, u64::MAX, Some(2_000)),
        ))
        .expect("register");
    f.reg.core.batching_on.store(true, Ordering::Release);
    clock.set(2_500);
    f.reg.heartbeat(&h).expect("buffer heartbeat");
    f.node.shutdown().await.expect("shutdown");
    assert!(f.reg.core.flush_pending_heartbeats().is_err());
    assert_eq!(
        f.reg
            .core
            .pending_hb
            .lock()
            .get(&("reader".to_string(), 1))
            .copied(),
        Some(2_500)
    );
}

/// The sweep persists buffered heartbeats before judging liveness, and a
/// heartbeat that arrives while that write is in flight still counts.
#[test]
fn a_heartbeat_buffered_during_the_sweep_keeps_the_reader() {
    let (reg, clock, _source, pipeline, _dir) = hooked_registry();
    clock.set(1_000);
    let h = reg
        .register(registration(
            "reader",
            0,
            bounded(u64::MAX, u64::MAX, Some(400)),
        ))
        .expect("register");
    reg.core.batching_on.store(true, Ordering::Release);
    reg.heartbeat(&h).expect("heartbeat");

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
        "a reader with a buffered heartbeat was ended"
    );
    assert_eq!(swept.next_deadline_ms, Some(1_000 + 400 + 1));
    assert_eq!(ended_for(&reg, "reader"), None);
}

/// Records of the format before retention policies are deleted by a sweep.
#[tokio::test(flavor = "multi_thread")]
async fn a_sweep_deletes_records_of_the_old_format() {
    let f = fixture(Arc::new(ManualClock::new(0))).await;
    let mut legacy_key = LEGACY_KEY_PREFIX.to_vec();
    legacy_key.extend_from_slice(b"cdc-0-7");
    f.engine
        .put(Partition::Registry, &legacy_key, b"old")
        .expect("seed a legacy record");
    f.reg.core.sweep_evictions().expect("sweep");
    assert!(
        f.engine
            .get(Partition::Registry, &legacy_key)
            .expect("read")
            .is_none(),
        "the legacy record is gone"
    );
    f.node.shutdown().await.expect("shutdown");
}

/// A pipeline that runs a one-shot hook inside the next proposal: what
/// happens elsewhere while a registry write is in flight.
struct HookPipeline {
    inner: coordinode_raft::proposal::OwnedLocalProposalPipeline,
    hook: Mutex<Option<Box<dyn FnOnce() + Send>>>,
    /// Refuse every proposal as a follower would, naming member 2 leader.
    follower: AtomicBool,
}

impl ProposalPipeline for HookPipeline {
    fn propose_and_wait(
        &self,
        proposal: &RaftProposal,
    ) -> Result<
        coordinode_core::txn::proposal::ProposalOutcome,
        coordinode_core::txn::proposal::ProposalError,
    > {
        if self.follower.load(Ordering::Acquire) {
            return Err(coordinode_core::txn::proposal::ProposalError::NotLeader {
                leader_id: Some(2),
            });
        }
        let hook = self.hook.lock().take();
        if let Some(hook) = hook {
            hook();
        }
        self.inner.propose_and_wait(proposal)
    }
}

/// A registry over a local engine whose next proposal can run a hook first.
fn hooked_registry() -> (
    ShardConsumerRegistry,
    Arc<ManualClock>,
    Arc<FakeSource>,
    Arc<HookPipeline>,
    tempfile::TempDir,
) {
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
        follower: AtomicBool::new(false),
    });
    let clock = Arc::new(ManualClock::new(0));
    let source = FakeSource::new();
    let reg = ShardConsumerRegistry::new(
        Arc::clone(&engine),
        Arc::clone(&pipeline) as Arc<dyn ProposalPipeline>,
        Arc::new(ProposalIdGenerator::with_base(1u64 << 48)),
        Arc::clone(&clock) as Arc<dyn Clock>,
        Arc::clone(&source) as Arc<dyn RetentionSource>,
    );
    (reg, clock, source, pipeline, dir)
}

/// On a member that is not the leader the sweep decides nothing: the leader
/// owns every transition. It still refreshes the floor the member publishes
/// and keeps the next deadline, so it acts once it leads; heartbeats buffered
/// here are dropped, since only the leader can record them.
#[test]
fn a_follower_sweep_leaves_transitions_to_the_leader() {
    let (reg, clock, source, pipeline, _dir) = hooked_registry();
    source.set_head(5);
    reg.register(registration("idle", 5, bounded(60_000, 1 << 30, Some(400))))
        .expect("register on the leader");
    let handle = RegisteredHandle::new("idle", 1);

    pipeline.follower.store(true, Ordering::Release);
    reg.core
        .pending_hb
        .lock()
        .insert(("idle".to_string(), 1), 100);
    clock.set(10_000);
    let swept = reg
        .core
        .sweep_evictions()
        .expect("a follower sweep is not an error");
    assert_eq!(swept.evicted, 0);
    assert!(!swept.behind, "a follower schedules no periodic sweep");
    assert_eq!(
        swept.next_deadline_ms,
        Some(401),
        "the deadline stays for when this member leads"
    );
    assert_eq!(ended_for(&reg, "idle"), None, "only the leader ends it");
    assert!(
        reg.core.pending_hb.lock().is_empty(),
        "heartbeats only the leader can record are dropped"
    );
    assert!(
        matches!(
            reg.checkpoint(&handle, 5),
            Err(RegistryError::NotLeader { leader_id: Some(2) })
        ),
        "a transition on a follower names the leader"
    );

    pipeline.follower.store(false, Ordering::Release);
    assert_eq!(
        reg.core.sweep_evictions().expect("sweep as leader").evicted,
        1
    );
    assert_eq!(
        ended_for(&reg, "idle"),
        Some(TerminalReason::LivenessExpired)
    );
}

/// While the source has no room for new retention, a registration is refused
/// as retryable backpressure and records nothing; consumers already admitted
/// still advance and cancel, since that holds no more than they held. Once
/// the pressure eases the same id registers as if it had never been asked.
#[test]
fn a_registration_under_pressure_is_refused_and_admitted_ones_continue() {
    let (reg, _clock, source, _pipeline, _dir) = hooked_registry();
    source.set_head(50);
    let admitted = reg
        .register(registration(
            "admitted",
            10,
            ConsumerRetentionPolicy::Strict,
        ))
        .expect("register before the pressure");

    source.set_pressured(true);
    let refused = reg.register(registration("late", 10, ConsumerRetentionPolicy::Strict));
    assert!(
        matches!(refused, Err(RegistryError::Backpressure)),
        "got {refused:?}"
    );
    assert!(
        reg.list_consumers().iter().all(|c| c.consumer_id != "late"),
        "a refused registration left a record"
    );
    reg.checkpoint(&admitted, 30)
        .expect("an admitted consumer advances under pressure");
    assert_eq!(reg.shard_floor(), 30);
    reg.unregister(admitted)
        .expect("an admitted consumer cancels under pressure");
    assert_eq!(reg.shard_floor(), u64::MAX);

    source.set_pressured(false);
    let late = reg
        .register(registration("late", 10, ConsumerRetentionPolicy::Strict))
        .expect("register once the pressure eases");
    assert_eq!(late.incarnation(), 1, "the refusal used no incarnation");
}
