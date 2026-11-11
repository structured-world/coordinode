use super::*;
use crate::registry::types::{
    ConsumerKind, PriorIncarnation, RetentionPressure, Unmeasured, ValidatedRetentionBounds,
};
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
    /// Clock ms each position's first work was produced at, `None` where the
    /// source has proved none is owed; a position missing here is one whose
    /// evidence the source cannot read.
    produced: Mutex<HashMap<u64, Option<u64>>>,
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
        self.produced.lock().insert(position, Some(at_ms));
    }
    fn no_work(&self, position: u64) {
        self.produced.lock().insert(position, None);
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
    fn produced_at_ms(&self, _: ConsumerKind, position: u64) -> Result<Option<u64>, Unmeasured> {
        self.produced
            .lock()
            .get(&position)
            .copied()
            .ok_or(Unmeasured::Unreadable)
    }
    fn retained_bytes_from(&self, _: ConsumerKind, _: u64) -> Result<u64, Unmeasured> {
        Ok(self.bytes.load(Ordering::Acquire))
    }
    fn admits(&self, _: ConsumerKind) -> Result<(), RetentionPressure> {
        if self.pressured.load(Ordering::Acquire) {
            Err(RetentionPressure::WritePressure)
        } else {
            Ok(())
        }
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
    keep_all_history(&engine);
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
    .expect("open the registry")
}

/// Start the registry's background service with these cadences.
fn start_service(
    reg: &ShardConsumerRegistry,
    heartbeat_window_ms: u64,
    eviction_interval_ms: u64,
) -> RegistryBackground {
    reg.start_background(BackgroundConfig {
        heartbeat_window_ms,
        eviction_interval_ms,
    })
    .expect("start the registry service")
}

/// Stop the service where blocking is allowed: its final flush commits
/// through the log, which this runtime serves.
async fn stop_service(bg: RegistryBackground) {
    tokio::task::spawn_blocking(move || bg.stop())
        .await
        .expect("stop the registry service");
}

/// Keep every MVCC version: with a window longer than the clock has run, the
/// engine has collected nothing, so a registration at any small seqno starts
/// over history still held, as the tests' fixed positions need.
fn keep_all_history(engine: &StorageEngine) {
    engine.set_retention_window(Duration::from_secs(100 * 365 * 24 * 3_600));
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
            ..registration(id, 1, ConsumerRetentionPolicy::Strict)
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
    // Judged on one whole observation: the source can tell its work, none.
    f.source.no_work(3);
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
        .register(registration("r", 1, ConsumerRetentionPolicy::Strict))
        .expect("register");
    f.reg.unregister(old.clone()).expect("cancel");
    let new = f
        .reg
        .register(registration(
            "r",
            1,
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
            .register(registration("b", 1, bounded(1_000, 1_000, None)))
            .unwrap_err(),
        RegistryError::InvalidRetention(_)
    ));
    f.reg
        .register(registration("s", 1, ConsumerRetentionPolicy::Strict))
        .expect("STRICT needs no measure");
    f.node.shutdown().await.expect("shutdown");
}

/// A log registration starting below what the source holds is refused: its
/// protection would start over history already gone.
#[tokio::test(flavor = "multi_thread")]
async fn registering_below_the_retained_source_is_refused() {
    let f = fixture(Arc::new(ManualClock::new(0))).await;
    f.source.set_first(100);
    let log = |id: &str, at| ConsumerRegistration {
        kind: ConsumerKind::OplogEvents,
        ..registration(id, at, ConsumerRetentionPolicy::Strict)
    };
    assert!(matches!(
        f.reg.register(log("late", 99)).unwrap_err(),
        RegistryError::RetentionLost {
            checkpoint: 99,
            floor: 100
        }
    ));
    let h = f
        .reg
        .register(ConsumerRegistration {
            initial_seqno: InitialSeqno::FromEarliestRetained,
            ..log("earliest", 0)
        })
        .expect("from the earliest retained");
    assert_eq!(f.reg.check_retention(&h).expect("held"), 100);
    f.node.shutdown().await.expect("shutdown");
}

/// Before every commit attempt of a registry transition, a heartbeat of the
/// same consumer lands, as the background flush does under a slow log.
fn heartbeat_before_every_commit(reg: &ShardConsumerRegistry, h: &RegisteredHandle, at: u64) {
    let (again, h) = (reg.clone(), h.clone());
    *reg.core.before_commit.lock() = Some(Box::new(move || {
        again.core.clock_heartbeat(&h, at).expect("heartbeat");
        heartbeat_before_every_commit(&again, &h, at);
    }));
}

/// Heartbeats never make an acknowledgement lose: the batched flush writes
/// liveness, not the position, so a consumer whose heartbeats keep landing
/// while its acknowledgement commits still advances, and both stand. Sharing
/// one conditioned record, every attempt of the acknowledgement met a newer
/// heartbeat and the stream was ended after sixteen.
#[test]
fn heartbeats_landing_during_every_acknowledgement_do_not_end_the_stream() {
    let (reg, clock, source, _pipeline, _dir) = hooked_registry();
    let h = reg
        .register(registration("busy", 7, ConsumerRetentionPolicy::Strict))
        .expect("register");
    source.set_head(20);
    clock.set(5_000);
    heartbeat_before_every_commit(&reg, &h, 6_000);
    let acked = reg.checkpoint(&h, 20);
    *reg.core.before_commit.lock() = None;
    acked.expect("an acknowledgement raced only by heartbeats lands");
    assert_eq!(reg.shard_floor(), 20, "the acknowledgement stands");
    let seen = reg
        .list_consumers()
        .into_iter()
        .find(|c| c.consumer_id == "busy")
        .expect("listed");
    assert_eq!(seen.checkpoint_seqno, 20);
    assert_eq!(seen.last_heartbeat_ts_ms, 6_000, "the heartbeats stand");
}

/// A heartbeat landing after the sweep judged a consumer silent and before
/// its terminal write keeps the consumer: heartbeats write their own record,
/// so the expiry is conditioned on that record too and reads again.
#[test]
fn a_heartbeat_racing_a_liveness_expiry_keeps_the_consumer() {
    let (reg, clock, _source, _pipeline, _dir) = hooked_registry();
    clock.set(1_000);
    let h = reg
        .register(registration(
            "quiet",
            7,
            bounded(60_000, 1 << 30, Some(5_000)),
        ))
        .expect("register");
    clock.set(6_001);
    *reg.core.before_commit.lock() = Some(Box::new({
        let (reg, h) = (reg.clone(), h.clone());
        move || reg.core.clock_heartbeat(&h, 6_001).expect("heartbeat")
    }));
    let swept = reg.core.sweep_evictions().expect("sweep");
    assert_eq!(swept.evicted, 0, "the expiry decided on a stale silence");
    assert_eq!(ended_for(&reg, "quiet"), None);
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

/// A change of the material basis published after a consumer was measured
/// and before its ending applies, under which the consumer is within its
/// bounds: the ending, decided where the log applies it, finds the basis
/// moved and writes nothing, and the measure against the new basis ends
/// nobody.
#[test]
fn a_basis_change_racing_an_ending_keeps_a_now_compliant_consumer() {
    let (reg, clock, source, _pipeline, _dir) = hooked_registry();
    reg.register(registration("measured", 7, bounded(1_000, 1 << 30, None)))
        .expect("register");
    source.set_head(20);
    source.produced(7, 0);
    clock.set(5_000);

    *reg.core.before_commit.lock() = Some(Box::new({
        let engine = Arc::clone(&reg.core.engine);
        let source = Arc::clone(&source);
        move || {
            // The new basis no longer requires the old work.
            source.produced(7, 4_900);
            engine
                .put(Partition::Registry, MATERIAL_BASIS_KEY, b"revision 1")
                .expect("the basis moves first");
        }
    }));
    assert_eq!(
        reg.core.sweep_evictions().expect("sweep").evicted,
        0,
        "an ending measured on the old basis landed"
    );
    assert_eq!(ended_for(&reg, "measured"), None);
}

/// An acknowledgement already checked and on its way to the log when an
/// ending applies ahead of it: the ending stands, and the acknowledgement,
/// decided where the log applies it, finds the registration ended and
/// writes nothing. Before, it wrote its live record over the ending.
#[test]
fn an_acknowledgement_in_flight_behind_an_ending_resurrects_nothing() {
    let (reg, clock, source, pipeline, _dir) = hooked_registry();
    let h = reg
        .register(registration("raced", 7, bounded(1_000, 1 << 30, None)))
        .expect("register");
    source.set_head(20);
    source.produced(7, 0);
    clock.set(5_000);

    // The ending takes its timestamp and stops before it applies. Meanwhile
    // an acknowledgement passes its checks (the registration is live),
    // takes a later timestamp and stops before it applies too. The ending
    // applies, then the acknowledgement.
    let (checked_tx, checked) = std::sync::mpsc::channel::<()>();
    let (release_tx, release) = std::sync::mpsc::channel::<()>();
    pipeline.later_hooks.lock().push_back(Box::new(move || {
        checked_tx.send(()).expect("signal");
        release.recv().expect("released");
    }));
    let acknowledging = Arc::new(Mutex::new(None));
    *pipeline.hook.lock() = Some(Box::new({
        let (reg, h, acknowledging) = (reg.clone(), h.clone(), Arc::clone(&acknowledging));
        move || {
            *acknowledging.lock() = Some(std::thread::spawn(move || reg.checkpoint(&h, 20)));
            checked
                .recv_timeout(std::time::Duration::from_secs(10))
                .expect("the acknowledgement passed its checks");
        }
    }));
    reg.core.sweep_evictions().expect("sweep");
    release_tx.send(()).expect("release");
    let acknowledged = acknowledging
        .lock()
        .take()
        .expect("spawned")
        .join()
        .expect("acknowledgement");
    assert_eq!(
        ended_for(&reg, "raced"),
        Some(TerminalReason::ProgressLagExceeded),
        "the acknowledgement wrote a live record over the ending"
    );
    assert!(
        matches!(
            acknowledged,
            Err(RegistryError::Terminated {
                reason: TerminalReason::ProgressLagExceeded,
                ..
            })
        ),
        "the acknowledgement is refused with the ending: {acknowledged:?}"
    );
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

/// Ending a registration releases only its own protection: a read already
/// admitted at that position keeps the history it pinned until it finishes,
/// and only then can collection pass it.
#[test]
fn a_read_in_progress_keeps_its_history_when_the_consumer_ends() {
    let (reg, _clock, _source, _pipeline, _dir) = hooked_registry();
    let engine = Arc::clone(&reg.core.engine);
    let h = reg
        .register(registration("reader", 100, ConsumerRetentionPolicy::Strict))
        .expect("register");
    let reading = engine
        .pin_snapshot_at(100)
        .expect("the read is admitted at the consumer's position");
    reg.unregister(h).expect("cancel");
    engine.set_retention_window(Duration::from_millis(1));
    assert!(
        engine.gc_watermark() <= 100,
        "ending the registration released the history a read still uses"
    );
    drop(reading);
    engine.set_retention_window(Duration::from_millis(1));
    assert!(
        engine.gc_watermark() > 100,
        "with the read finished and the consumer gone nothing holds it"
    );
}

/// A consumer behind the head whose source accounts its space but cannot
/// read when its work was produced is reported as unmeasured, never as a
/// consumer with no work waiting: a zero would read as healthy exactly when
/// the evidence is missing.
#[test]
fn a_lag_the_source_cannot_read_is_not_reported_as_zero() {
    let (reg, clock, source, _pipeline, _dir) = hooked_registry();
    source.set_head(30);
    reg.register(ConsumerRegistration {
        kind: ConsumerKind::OplogEvents,
        ..registration("blind", 10, ConsumerRetentionPolicy::Strict)
    })
    .expect("register");
    clock.set(6_000);

    let lag = || {
        reg.list_consumers()
            .into_iter()
            .find(|c| c.consumer_id == "blind")
            .expect("listed")
            .progress_lag_ms
    };
    assert_eq!(
        lag(),
        Some(Err(Unmeasured::Unreadable)),
        "unreadable work time reported as no lag"
    );
    // Once the source proves nothing is owed from there, the lag is zero.
    source.no_work(10);
    assert_eq!(lag(), Some(Ok(0)));
}

/// The retention summary counts a space's shared history once, names the
/// registrations at its floor (the ones whose progress or cancellation lets
/// it rise), and says when new retention is refused for pressure; each
/// registration's own view says whether it holds the floor and how long its
/// work has waited.
#[test]
fn the_summary_counts_shared_history_once_and_names_who_holds_it() {
    let (reg, clock, source, _pipeline, _dir) = hooked_registry();
    let log = |id: &str, at| ConsumerRegistration {
        kind: ConsumerKind::OplogEvents,
        ..registration(id, at, ConsumerRetentionPolicy::Strict)
    };
    source.set_head(30);
    source.set_bytes(1_000);
    source.produced(10, 1_000);
    source.produced(20, 4_000);
    let slow = reg.register(log("slow", 10)).expect("register slow");
    reg.register(log("fast", 20)).expect("register fast");
    reg.register(registration("state", 5, ConsumerRetentionPolicy::Strict))
        .expect("register state");
    clock.set(6_000);

    let summary = reg.retention_summary().expect("summary");
    let space = |s: RetentionSpace| {
        summary
            .iter()
            .find(|v| v.space == s)
            .cloned()
            .expect("a view per space")
    };
    let log_view = space(RetentionSpace::Log);
    assert_eq!(log_view.floor, Some(10));
    assert_eq!(
        log_view.retained_bytes,
        Some(Ok(1_000)),
        "shared history is counted once, not once per registration"
    );
    assert_eq!(log_view.held_by, vec!["slow".to_string()]);
    assert_eq!(log_view.refused, None);
    assert_eq!(
        space(RetentionSpace::Mvcc).held_by,
        vec!["state".to_string()]
    );

    let listed = reg.list_consumers();
    let view = |id: &str| {
        listed
            .iter()
            .find(|c| c.consumer_id == id)
            .cloned()
            .expect("listed")
    };
    assert!(view("slow").holds_floor && !view("fast").holds_floor);
    assert_eq!(view("slow").progress_lag_ms, Some(Ok(5_000)));
    assert_eq!(view("fast").progress_lag_ms, Some(Ok(2_000)));
    assert_eq!(view("fast").retained_bytes, Some(Ok(1_000)));

    // Cancelling the holder moves the floor and the holder to the next one.
    reg.unregister(slow).expect("cancel");
    let log_view = reg
        .retention_summary()
        .expect("summary")
        .into_iter()
        .find(|v| v.space == RetentionSpace::Log)
        .expect("log view");
    assert_eq!(log_view.floor, Some(20));
    assert_eq!(log_view.held_by, vec!["fast".to_string()]);

    source.set_pressured(true);
    assert!(
        reg.retention_summary()
            .expect("summary")
            .iter()
            .all(|v| v.refused == Some(RetentionPressure::WritePressure)),
        "write pressure is reported as the reason new retention is refused"
    );
}

/// A consumer registering again after its registration ended (a resync)
/// records what it followed: the ended incarnation, why it ended, what it
/// had acknowledged and where the history started then. From where the new
/// one starts it is plain what no incarnation was given; a resync that
/// starts at the last acknowledgement missed nothing.
#[test]
fn a_resync_records_what_it_followed_and_what_was_missed() {
    let (reg, _clock, source, _pipeline, _dir) = hooked_registry();
    let log = |id: &str, initial_seqno| ConsumerRegistration {
        kind: ConsumerKind::OplogEvents,
        initial_seqno,
        ..registration(id, 0, ConsumerRetentionPolicy::Strict)
    };
    source.set_first(10);
    source.set_head(50);
    let first = reg
        .register(log("sink", InitialSeqno::At(20)))
        .expect("register");
    reg.checkpoint(&first, 30).expect("acknowledge");
    source.set_first(40);
    reg.unregister(first).expect("cancel");

    // The history from 30 is gone: the consumer resyncs from what is held.
    reg.register(log("sink", InitialSeqno::FromEarliestRetained))
        .expect("resync");
    let view = reg
        .list_consumers()
        .into_iter()
        .find(|c| c.consumer_id == "sink")
        .expect("listed");
    let follows = view.follows.expect("it follows the ended incarnation");
    assert_eq!(
        follows,
        PriorIncarnation {
            incarnation: 1,
            reason: TerminalReason::Cancelled,
            acknowledged: 30,
            retained_from: 40,
        }
    );
    assert_eq!(view.started_at, 40);
    assert_eq!(follows.missed(view.started_at), Some(30..40));
    assert_eq!(follows.missed(30), None, "a resync from 30 misses nothing");

    let fresh = reg
        .register(log("new", InitialSeqno::FromNow))
        .expect("register new");
    assert!(
        reg.list_consumers()
            .into_iter()
            .find(|c| c.consumer_id == fresh.consumer_id())
            .expect("listed")
            .follows
            .is_none(),
        "a first registration follows nothing"
    );
}

/// An ended registration records where its source's history started when it
/// ended, in its own space: the log's first entry for a log consumer, the
/// engine's collection watermark for an MVCC one. A late handle learns it
/// with the reason and checkpoint, so it can tell a policy ending over
/// history still there from history already gone.
#[test]
fn an_ended_registration_records_where_its_history_started() {
    let (reg, clock, source, _pipeline, _dir) = hooked_registry();
    source.set_first(40);
    source.set_head(90);
    let log = reg
        .register(ConsumerRegistration {
            kind: ConsumerKind::OplogEvents,
            ..registration("log", 60, ConsumerRetentionPolicy::Strict)
        })
        .expect("register log");
    let mvcc = reg
        .register(registration("mvcc", 7, bounded(1_000, 1 << 30, None)))
        .expect("register mvcc");
    source.set_first(55);
    reg.unregister(log.clone()).expect("cancel");
    assert!(matches!(
        reg.checkpoint(&log, 70).unwrap_err(),
        RegistryError::Terminated {
            reason: TerminalReason::Cancelled,
            checkpoint: 60,
            retained_from: 55,
            ..
        }
    ));

    source.produced(7, 0);
    clock.set(5_000);
    assert_eq!(reg.core.sweep_evictions().expect("sweep").evicted, 1);
    let readable = reg.core.engine.oldest_readable_seqno();
    let refused = reg.checkpoint(&mvcc, 20).unwrap_err();
    assert!(
        matches!(
            refused,
            RegistryError::Terminated {
                reason: TerminalReason::ProgressLagExceeded,
                checkpoint: 7,
                retained_from,
                ..
            } if retained_from == readable
        ),
        "the MVCC boundary is the oldest seqno the engine serves ({readable}): {refused:?}"
    );
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
    // A short window lets collection run: only an MVCC consumer holds it.
    f.engine.set_retention_window(Duration::from_millis(1));
    let collected = f.engine.gc_watermark();
    assert!(
        collected > 42,
        "the oplog consumer held MVCC collection back"
    );

    let seqno_h = f
        .reg
        .register(registration(
            "backup",
            collected,
            ConsumerRetentionPolicy::Strict,
        ))
        .expect("register seqno consumer");
    assert_eq!(f.reg.shard_floor(), collected);
    assert_eq!(f.engine.gc_watermark(), collected);
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
    // A short window lets collection run: the consumer alone holds it.
    f.engine.set_retention_window(Duration::from_millis(1));
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
    // An operator-forced floor above the checkpoint, with a window that no
    // longer holds the history either.
    f.engine.set_retention_window(Duration::from_millis(1));
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
    let h = f
        .reg
        .register(registration("probe", 1, ConsumerRetentionPolicy::Strict))
        .expect("register");
    f.engine.set_retention_window(Duration::from_millis(1));
    assert_eq!(f.engine.gc_watermark(), 1, "the consumer holds collection");
    f.reg.unregister(h).expect("cancel");
    assert_eq!(f.engine.gc_watermark(), f.engine.snapshot() - 1_000);
    f.node.shutdown().await.expect("shutdown");
}

/// `FromNow` starts at the source head, `FromEarliestRetained` at the first
/// retained position: the log's for a log consumer, the engine's collection
/// watermark for an MVCC one.
#[tokio::test(flavor = "multi_thread")]
async fn initial_positions_resolve_against_the_source() {
    let f = fixture(Arc::new(ManualClock::new(0))).await;
    f.source.set_first(11);
    f.source.set_head(77);
    let log = |id: &str, initial_seqno| ConsumerRegistration {
        kind: ConsumerKind::OplogEvents,
        initial_seqno,
        ..registration(id, 0, ConsumerRetentionPolicy::Strict)
    };
    let earliest = f
        .reg
        .register(log("all", InitialSeqno::FromEarliestRetained))
        .expect("register");
    let now = f
        .reg
        .register(log("new-only", InitialSeqno::FromNow))
        .expect("register");
    assert_eq!(f.reg.check_retention(&earliest).expect("held"), 11);
    assert_eq!(f.reg.check_retention(&now).expect("held"), 77);

    f.engine.set_retention_window(Duration::from_millis(1));
    let collected = f.engine.gc_watermark();
    let mvcc = f
        .reg
        .register(ConsumerRegistration {
            initial_seqno: InitialSeqno::FromEarliestRetained,
            ..registration("mvcc", 0, ConsumerRetentionPolicy::Strict)
        })
        .expect("register");
    assert!(
        f.reg.check_retention(&mvcc).expect("held") >= collected,
        "an MVCC consumer starts at what the engine still holds"
    );
    f.node.shutdown().await.expect("shutdown");
}

/// Collection catching up while a registration commits never passes the
/// position being registered: the position is held from the moment it is
/// found still collected-free, not from when the registry next publishes its
/// floor. Before, the watermark ran past it in that window and was pulled
/// back afterwards, over versions a compaction could already have dropped.
#[test]
fn collection_racing_a_registration_never_passes_its_position() {
    let (reg, _clock, _source, _pipeline, _dir) = hooked_registry();
    let engine = Arc::clone(&reg.core.engine);
    let passed_to = Arc::new(AtomicU64::new(0));
    *reg.core.before_commit.lock() = Some(Box::new({
        let (engine, passed_to) = (Arc::clone(&engine), Arc::clone(&passed_to));
        move || {
            engine.set_retention_window(Duration::from_millis(1));
            passed_to.store(engine.gc_watermark(), Ordering::Release);
        }
    }));
    let h = reg
        .register(registration("racing", 100, ConsumerRetentionPolicy::Strict))
        .expect("the position was free of collection when registered");
    let reached = passed_to.load(Ordering::Acquire);
    assert!(
        reached <= 100,
        "collection reached {reached} while the registration of 100 was committing"
    );
    assert_eq!(reg.check_retention(&h).expect("covered"), 100);
}

/// A journal purge running while a log registration commits never passes
/// the position being registered, over a store's own journal: the position is
/// held from the check on, under the lock the purge takes. Before, the purge
/// removed it in between, and the consumer was registered over entries the
/// journal no longer had.
#[test]
fn a_journal_purge_racing_a_registration_never_passes_its_position() {
    use coordinode_core::txn::proposal::{Mutation, PartitionId};
    use coordinode_storage::engine::oplog_journal::OplogJournalConfig;

    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new());
    let engine = Arc::new(
        StorageEngine::open_embedded_with_journal(
            &StorageConfig::with_endpoints(vec![EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            )]),
            Arc::clone(&oracle),
            OplogJournalConfig {
                retention_secs: 1,
                max_segment_entries: 2,
                ..OplogJournalConfig::default()
            },
        )
        .expect("open embedded"),
    );
    for i in 0..6u64 {
        engine
            .commit_journaled(
                &[Mutation::Put {
                    partition: PartitionId::Node,
                    key: format!("node:r:{i}").into_bytes(),
                    value: b"v".to_vec(),
                }],
                oracle.next().as_raw(),
            )
            .expect("commit");
    }
    let source: Arc<dyn RetentionSource> = Arc::new(
        crate::registry::LogRetentionSource::over_own_journal(Arc::clone(&engine))
            .expect("inspect")
            .expect("a journal"),
    );
    let reg = ShardConsumerRegistry::new(
        Arc::clone(&engine),
        Arc::new(
            coordinode_storage::engine::local_pipeline::OwnedLocalProposalPipeline::new(&engine),
        ),
        Arc::new(ProposalIdGenerator::with_base(1u64 << 48)),
        Arc::new(ManualClock::new(0)),
        Arc::clone(&source),
    )
    .expect("open the registry");
    let at = source.first_retained(ConsumerKind::OplogEvents);
    let after_purge = Arc::new(AtomicU64::new(0));
    *reg.core.before_commit.lock() = Some(Box::new({
        let (engine, source, after_purge) = (
            Arc::clone(&engine),
            Arc::clone(&source),
            Arc::clone(&after_purge),
        );
        move || {
            engine
                .oplog_purge_expired(4_000_000_000, u64::MAX)
                .expect("purge");
            after_purge.store(
                source.first_retained(ConsumerKind::OplogEvents),
                Ordering::Release,
            );
        }
    }));
    let h = reg
        .register(ConsumerRegistration {
            kind: ConsumerKind::OplogEvents,
            ..registration("tail", at, ConsumerRetentionPolicy::Strict)
        })
        .expect("the position was in the journal when registered");
    let start = after_purge.load(Ordering::Acquire);
    assert!(
        start <= at,
        "a purge removed the journal from {at} up to {start} while it was being registered"
    );
    engine
        .oplog_purge_expired(4_000_000_000, u64::MAX)
        .expect("purge");
    assert!(
        source.first_retained(ConsumerKind::OplogEvents) <= at,
        "the published floor did not keep the journal"
    );
    assert_eq!(reg.check_retention(&h).expect("covered"), at);

    // Acknowledged, the journal below the new position is no longer kept for
    // the consumer once the floor is republished.
    let head = source.head(ConsumerKind::OplogEvents);
    reg.checkpoint(&h, head).expect("acknowledge");
    reg.core.sweep_evictions().expect("sweep");
    engine
        .oplog_purge_expired(4_000_000_000, u64::MAX)
        .expect("purge");
    assert!(
        source.first_retained(ConsumerKind::OplogEvents) > at,
        "the journal the consumer acknowledged was still kept"
    );
}

/// A seqno-space registration below what the engine has already collected is
/// refused, whatever the source says: the engine's watermark decides what MVCC
/// history is held.
#[test]
fn registering_below_collected_history_is_refused() {
    let (reg, _clock, _source, _pipeline, _dir) = hooked_registry();
    reg.core
        .engine
        .set_retention_window(Duration::from_millis(1));
    let collected = reg.core.engine.gc_watermark();
    assert!(collected > 100, "the window let nothing be collected");
    let refused = reg.register(registration("late", 100, ConsumerRetentionPolicy::Strict));
    assert!(
        matches!(
            refused,
            Err(RegistryError::RetentionLost { checkpoint: 100, floor }) if floor >= collected
        ),
        "got {refused:?}"
    );
    assert_eq!(
        reg.shard_floor(),
        u64::MAX,
        "a refused registration holds nothing"
    );
}

/// Floors published out of order would let a scan taken before a registration
/// overwrite the floor that registration published: the floor rises past the
/// registered position and collection may take what it holds. A floor is
/// scanned and published as one step, so the later scan always wins.
#[test]
fn a_stale_floor_scan_never_overwrites_a_newer_registration() {
    let (reg, _clock, _source, _pipeline, _dir) = hooked_registry();
    let (scanned_tx, scanned_rx) = std::sync::mpsc::channel();
    let (resume_tx, resume_rx) = std::sync::mpsc::channel::<()>();
    *reg.core.after_floor_scan.lock() = Some(Box::new(move || {
        scanned_tx.send(()).expect("report the scan");
        resume_rx.recv().expect("resume the publication");
    }));
    let stale = std::thread::spawn({
        let reg = reg.clone();
        move || reg.core.recompute_floor().expect("stale recompute")
    });
    scanned_rx.recv().expect("the stale scan ran");
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let registering = std::thread::spawn({
        let reg = reg.clone();
        move || {
            let h = reg.register(registration("new", 100, ConsumerRetentionPolicy::Strict));
            done_tx.send(()).expect("report the registration");
            h
        }
    });
    // A registration that publishes its floor while the stale scan is held
    // finishes first; one that waits for it does not.
    let _finished_first = done_rx.recv_timeout(Duration::from_secs(2));
    resume_tx.send(()).expect("resume");
    stale.join().expect("stale thread");
    registering
        .join()
        .expect("registering thread")
        .expect("register");
    assert_eq!(
        reg.shard_floor(),
        100,
        "a scan taken before the registration published the floor last"
    );
    reg.core
        .engine
        .set_retention_window(Duration::from_millis(1));
    assert_eq!(
        reg.core.engine.gc_watermark(),
        100,
        "collection is held at the registered position"
    );
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
            1,
            bounded(u64::MAX, u64::MAX, Some(10_000)),
        ))
        .expect("register");
    let bg = start_service(&f.reg, 30, 100_000);
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
    stop_service(bg).await;
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
    let bg = start_service(&f.reg, 100_000, 30);
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
    stop_service(bg).await;
    f.node.shutdown().await.expect("shutdown");
}

/// A reader watching its registration hears of writes to its own record as
/// they apply (here, its cancellation) and not of writes to others', once
/// the notices of the writes made before it watched have been delivered.
#[tokio::test(flavor = "multi_thread")]
async fn a_watch_hears_of_its_own_record_only() {
    let f = fixture(Arc::new(ManualClock::new(1_000))).await;
    let bg = start_service(&f.reg, 100_000, 100_000);
    let mine = f
        .reg
        .register(registration("mine", 10, ConsumerRetentionPolicy::Strict))
        .expect("register mine");
    let other = f
        .reg
        .register(registration("other", 10, ConsumerRetentionPolicy::Strict))
        .expect("register other");
    let mut watch = f.reg.watch(&mine);
    let mut later = f.reg.watch(&other);
    assert!(watch.changed() && later.changed(), "the first calls");

    // The relay delivers notices after their writes applied, in apply order,
    // so the notice of the registration of `mine` can still be on its way
    // and reach the watch after its first call. A write to `other` made after
    // both registrations is heard only once every earlier notice has been
    // delivered: from then on, a change of `mine` is a write made after it.
    f.reg.checkpoint(&other, 20).expect("acknowledge other");
    let until = tokio::time::Instant::now() + Duration::from_secs(10);
    let mut relayed = false;
    while !relayed && tokio::time::Instant::now() < until {
        relayed = later.changed();
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(relayed, "the acknowledgement was relayed");
    // Whatever the registration's notice told it, taken in now.
    watch.changed();

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
    stop_service(bg).await;
    assert!(watch.changed(), "without the relay every call is a change");
    f.node.shutdown().await.expect("shutdown");
}

/// With the background service running, a registration that can lower the
/// floor still publishes it before returning: the history its position
/// needs must not be collected while a sweep is far away (100 s here).
#[tokio::test(flavor = "multi_thread")]
async fn a_lowering_registration_publishes_the_floor_before_returning() {
    let f = fixture(Arc::new(ManualClock::new(1_000))).await;
    let bg = start_service(&f.reg, 100_000, 100_000);
    f.reg
        .register(registration("late", 250, ConsumerRetentionPolicy::Strict))
        .expect("register late");
    assert_eq!(f.reg.shard_floor(), 250);
    f.reg
        .register(registration("early", 100, ConsumerRetentionPolicy::Strict))
        .expect("register early");
    assert_eq!(f.reg.shard_floor(), 100, "lowered before the call returned");
    stop_service(bg).await;
    f.node.shutdown().await.expect("shutdown");
}

/// With the background service running, an acknowledgement only raises the
/// floor, so the sweep its applied write triggers publishes it.
#[tokio::test(flavor = "multi_thread")]
async fn an_acknowledgement_raises_the_floor_through_the_sweep() {
    let f = fixture(Arc::new(ManualClock::new(1_000))).await;
    let bg = start_service(&f.reg, 100_000, 30);
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
    stop_service(bg).await;
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
    let bg = start_service(&f.reg, 100_000, 30);
    let elsewhere = registry_over(&f.engine, &f.node, clock, Arc::clone(&f.source), 3);
    elsewhere
        .register(registration("remote", 77, ConsumerRetentionPolicy::Strict))
        .expect("register elsewhere");
    let until = tokio::time::Instant::now() + Duration::from_secs(5);
    while f.reg.shard_floor() != 77 && tokio::time::Instant::now() < until {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(f.reg.shard_floor(), 77);
    stop_service(bg).await;
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
            1,
            bounded(u64::MAX, u64::MAX, Some(2_000)),
        ))
        .expect("register");
    f.reg.core.batching_on.store(true, Ordering::Release);
    clock.set(2_500);
    f.reg.heartbeat(&h).expect("buffer heartbeat");
    f.node.shutdown().await.expect("shutdown");
    assert!(f.reg.core.flush_pending_heartbeats().is_err());
    assert_eq!(f.reg.core.pending_hb.lock().get("reader", 1), Some(2_500));
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
            1,
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
    // The sweep's flush writes the heartbeat that waited beside its first
    // batch too, so the deadline counts from it.
    assert_eq!(swept.next_deadline_ms, Some(2_000 + 400 + 1));
    assert_eq!(ended_for(&reg, "reader"), None);
}

/// A pipeline that runs a one-shot hook inside the next proposal: what
/// happens elsewhere while a registry write is in flight.
struct HookPipeline {
    inner: coordinode_storage::engine::local_pipeline::OwnedLocalProposalPipeline,
    hook: Mutex<Option<Box<dyn FnOnce() + Send>>>,
    /// Run, in order, by the proposals that come after the one `hook` is
    /// taken by: one each, before it applies.
    later_hooks: Mutex<std::collections::VecDeque<Box<dyn FnOnce() + Send>>>,
    /// Refuse every proposal as a follower would, naming member 2 leader.
    follower: AtomicBool,
    /// Refuse every proposal as a failed write.
    failing: AtomicBool,
    /// Whether each proposal taken was marked the store's own work.
    own_work: Mutex<Vec<bool>>,
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
        } else {
            let later = self.later_hooks.lock().pop_front();
            if let Some(later) = later {
                later();
            }
        }
        if self.failing.load(Ordering::Acquire) {
            return Err(coordinode_core::txn::proposal::ProposalError::Storage(
                "injected write failure".to_string(),
            ));
        }
        self.own_work.lock().push(proposal.bypass_rate_limiter);
        self.inner.propose_and_wait(proposal)
    }
}

/// The leader's heartbeat buffer holds at most its cap of incarnations: a
/// heartbeat of another one flushes the buffered ones first, so none is
/// dropped and the buffer does not grow with the number of consumers.
#[test]
fn the_heartbeat_buffer_stays_within_its_cap() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = reopen(dir.path());
    keep_all_history(&engine);
    let reg = local_registry(&engine)
        .expect("registry")
        .with_pending_heartbeat_cap(2);
    let handles: Vec<RegisteredHandle> = ["a", "b", "c"]
        .into_iter()
        .map(|id| {
            reg.register(registration(id, 1, ConsumerRetentionPolicy::Strict))
                .expect("register")
        })
        .collect();
    reg.core.batching_on.store(true, Ordering::Release);
    for h in &handles {
        reg.heartbeat(h).expect("heartbeat");
        assert!(
            reg.core.pending_hb.lock().len() <= 2,
            "the buffer grew past its cap"
        );
    }
    for id in ["a", "b"] {
        let (stored, _) = reg.core.read_heartbeat(id).expect("read");
        assert!(
            stored.is_some(),
            "{id}'s heartbeat was flushed, not dropped"
        );
    }
}

/// Heartbeats arriving while a flush writes stay within the bound: the
/// flush owns what it took (no copy), the next ones wait within the cap,
/// and one past it waits for the flush instead of growing the buffer. The
/// heartbeats being written stay visible to a sweep, and every one lands.
/// Before, the room was checked and taken under two locks, and concurrent
/// flushes each copied the whole buffer.
#[test]
fn heartbeats_stay_bounded_while_a_flush_writes() {
    let (reg, _clock, _source, pipeline, _dir) = hooked_registry();
    let reg = reg.with_pending_heartbeat_cap(2);
    let ids = ["a", "b", "c", "d", "e", "f"];
    let handles: Vec<RegisteredHandle> = ids
        .iter()
        .map(|id| {
            reg.register(registration(id, 1, ConsumerRetentionPolicy::Strict))
                .expect("register")
        })
        .collect();
    reg.core.batching_on.store(true, Ordering::Release);
    reg.heartbeat(&handles[0]).expect("a");
    reg.heartbeat(&handles[1]).expect("b");

    // The next heartbeat finds the buffer full and flushes it; its write is
    // held inside the pipeline.
    let (held_tx, held_rx) = std::sync::mpsc::channel::<()>();
    let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
    *pipeline.hook.lock() = Some(Box::new(move || {
        held_tx.send(()).expect("report held");
        release_rx.recv().expect("released");
    }));
    std::thread::scope(|s| {
        let flusher = s.spawn(|| reg.heartbeat(&handles[2]));
        held_rx.recv().expect("the flush is writing");
        {
            let pending = reg.core.pending_hb.lock();
            assert_eq!(pending.writing.len(), 2, "the flush took the full buffer");
            assert!(pending.waiting.is_empty());
            assert_eq!(pending.get("a", 1), Some(0), "still visible");
        }
        // Room for two more while the write is held; a third waits for it.
        reg.heartbeat(&handles[3]).expect("d");
        reg.heartbeat(&handles[4]).expect("e");
        let waiter = s.spawn(|| reg.heartbeat(&handles[5]));
        std::thread::sleep(Duration::from_millis(100));
        {
            let pending = reg.core.pending_hb.lock();
            assert!(pending.waiting.len() <= 2, "waiting past the cap");
            assert!(pending.len() <= 4, "the buffer grew past twice its cap");
        }
        release_tx.send(()).expect("release");
        flusher.join().expect("join").expect("c");
        waiter.join().expect("join").expect("f");
    });
    reg.core.flush_pending_heartbeats().expect("flush the rest");
    for id in ids {
        let (stored, _) = reg.core.read_heartbeat(id).expect("read");
        assert!(stored.is_some(), "{id}'s heartbeat landed");
    }
}

/// Writes that keep failing do not grow the buffer: the batch a failed
/// write took stays the batch and is written again first, never merged back
/// into the waiting heartbeats, so waiting and written parts each stay
/// within the cap however many writes fail. A later heartbeat of an
/// incarnation in the failed batch waits beside it and the latest time is
/// what a sweep sees; once writes succeed every heartbeat lands. Before,
/// each failure merged the batch back, and the next one took both.
#[test]
fn repeated_failed_flushes_keep_the_heartbeat_bound() {
    let (reg, clock, _source, pipeline, _dir) = hooked_registry();
    let reg = reg.with_pending_heartbeat_cap(2);
    let handles: Vec<RegisteredHandle> = (0..8)
        .map(|i| {
            reg.register(registration(
                &format!("c{i}"),
                1,
                ConsumerRetentionPolicy::Strict,
            ))
            .expect("register")
        })
        .collect();
    reg.core.batching_on.store(true, Ordering::Release);
    pipeline.failing.store(true, Ordering::Release);
    clock.set(100);
    reg.heartbeat(&handles[0]).expect("c0");
    reg.heartbeat(&handles[1]).expect("c1");
    let within_cap = |round: &str| {
        let pending = reg.core.pending_hb.lock();
        assert!(
            pending.waiting.len() <= 2 && pending.writing.len() <= 2,
            "{round}: writing={} waiting={} past the cap 2",
            pending.writing.len(),
            pending.waiting.len()
        );
    };
    // While the flush writes c0, c1, a newer heartbeat of c0 and one of c2
    // fill the waiting part; then the write fails.
    *pipeline.hook.lock() = Some(Box::new({
        let (reg, clock) = (reg.clone(), Arc::clone(&clock));
        let (h0, h2) = (handles[0].clone(), handles[2].clone());
        move || {
            clock.set(500);
            reg.heartbeat(&h0).expect("c0 during the write");
            reg.heartbeat(&h2).expect("c2 during the write");
        }
    }));
    assert!(reg.core.flush_pending_heartbeats().is_err());
    within_cap("first failure");
    // The latest time of c0 is what a sweep sees.
    assert_eq!(reg.core.pending_hb.lock().get("c0", 1), Some(500));
    // From here every heartbeat finds both parts full; each flush retries
    // the failed batch, fails again, and the heartbeat is refused.
    for (i, handle) in handles.iter().enumerate().skip(3) {
        assert!(reg.heartbeat(handle).is_err(), "c{i} refused");
        within_cap(&format!("c{i}"));
    }

    pipeline.failing.store(false, Ordering::Release);
    reg.core.flush_pending_heartbeats().expect("flush");
    assert!(reg.core.pending_hb.lock().is_empty());
    for (id, ts) in [("c0", Some(500)), ("c1", Some(100)), ("c2", Some(500))] {
        let (stored, _) = reg.core.read_heartbeat(id).expect("read");
        assert_eq!(stored.map(|hb| hb.ts_ms), ts, "{id}");
    }
}

/// The buffer's bound counts the bytes of consumer ids too: ids are of any
/// length, so a cap on their number alone bounds no memory. One incarnation
/// per id is kept: a newer one replaces it, an older one is dropped.
#[test]
fn the_heartbeat_buffer_bounds_the_bytes_of_its_ids() {
    let mut pending = PendingHeartbeats::default();
    let long = "x".repeat(MAX_CONSUMER_ID_BYTES);
    let fill = MAX_PENDING_HEARTBEAT_BYTES / MAX_CONSUMER_ID_BYTES;
    for i in 0..fill {
        let id = format!("{i:05}{}", &long[5..]);
        assert!(pending.admit(&id, 1, 1, usize::MAX), "{i} fits");
    }
    assert!(
        !pending.admit(&format!("zzzzz{}", &long[5..]), 1, 1, usize::MAX),
        "another long id past the byte bound waits for a flush"
    );
    assert!(pending.waiting_bytes <= MAX_PENDING_HEARTBEAT_BYTES);
    let first = format!("00000{}", &long[5..]);
    assert!(pending.admit(&first, 1, 2, usize::MAX), "a later time fits");
    assert!(
        pending.admit(&first, 2, 3, usize::MAX),
        "a newer incarnation"
    );
    assert!(
        pending.admit(&first, 1, 9, usize::MAX),
        "an older one drops"
    );
    assert_eq!(pending.get(&first, 2), Some(3));
    assert_eq!(pending.get(&first, 1), None);
    assert_eq!(pending.waiting.len(), fill);
}

/// A registration or heartbeat of an id longer than a consumer's record
/// keys can hold is refused with a typed error, before anything is copied
/// or stored. Before, the storage refused the key by panicking.
#[test]
fn an_overlong_consumer_id_is_refused() {
    let (reg, _clock, _source, pipeline, _dir) = hooked_registry();
    let id = "x".repeat(MAX_CONSUMER_ID_BYTES + 1);
    assert!(matches!(
        reg.register(registration(&id, 1, ConsumerRetentionPolicy::Strict)),
        Err(RegistryError::ConsumerIdTooLong { len, max })
            if len == MAX_CONSUMER_ID_BYTES + 1 && max == MAX_CONSUMER_ID_BYTES
    ));
    let fits = "y".repeat(MAX_CONSUMER_ID_BYTES);
    reg.register(registration(&fits, 1, ConsumerRetentionPolicy::Strict))
        .expect("the longest accepted id registers");
    reg.core.batching_on.store(true, Ordering::Release);
    let proposals = pipeline.own_work.lock().len();
    assert!(matches!(
        reg.heartbeat(&RegisteredHandle::new(&id, 1)),
        Err(RegistryError::ConsumerIdTooLong { .. })
    ));
    assert!(reg.core.pending_hb.lock().is_empty(), "nothing buffered");
    assert_eq!(pipeline.own_work.lock().len(), proposals, "nothing written");
}

/// The proposals a consumer's progress makes (acknowledgement, heartbeat,
/// cancellation) carry the store's own-work mark, so every member, a
/// follower appending the entry too, admits them from its recovery reserve;
/// a new registration's proposal does not, and is refused with client
/// writes. Before, the mark was set on the leader's thread only and the
/// proposal went out as a client write.
#[test]
fn a_consumers_progress_proposes_the_stores_own_work() {
    let (reg, _clock, _source, pipeline, _dir) = hooked_registry();
    let handle = reg
        .register(registration("sink", 1, ConsumerRetentionPolicy::Strict))
        .expect("register");
    let registered = std::mem::take(&mut *pipeline.own_work.lock());
    reg.checkpoint(&handle, 2).expect("acknowledge");
    reg.heartbeat(&handle).expect("heartbeat");
    reg.unregister(handle).expect("cancel");
    let progress = std::mem::take(&mut *pipeline.own_work.lock());
    assert!(
        !registered.is_empty() && registered.iter().all(|own| !own),
        "a registration is a client write: {registered:?}"
    );
    assert!(
        !progress.is_empty() && progress.iter().all(|own| *own),
        "a consumer's progress is the store's own work: {progress:?}"
    );
}

/// An engine over `dir`, opened afresh as a restart opens it.
fn reopen(dir: &std::path::Path) -> Arc<StorageEngine> {
    Arc::new(
        StorageEngine::open_with_oracle(
            &StorageConfig::with_endpoints(vec![EndpointConfig::new(
                "default",
                dir,
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            )]),
            Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new()),
        )
        .expect("open engine"),
    )
}

/// A registry over `engine` writing through its local pipeline.
fn local_registry(engine: &Arc<StorageEngine>) -> Result<ShardConsumerRegistry, RegistryError> {
    ShardConsumerRegistry::new(
        Arc::clone(engine),
        Arc::new(
            coordinode_storage::engine::local_pipeline::OwnedLocalProposalPipeline::new(engine),
        ),
        Arc::new(ProposalIdGenerator::with_base(1u64 << 48)),
        Arc::new(ManualClock::new(0)),
        FakeSource::new(),
    )
}

/// A store reopened over a STRICT MVCC registration keeps the history it
/// was promised from the open on, not from when its registry recovers:
/// collection that runs in between (flushes and compactions the open starts,
/// here forced, with a window that would collect everything) leaves the
/// promised version in place. Before, the open published no floor, the window
/// alone governed, and the compaction dropped the version for good.
#[test]
fn a_reopened_store_keeps_a_strict_consumers_history_before_its_registry_recovers() {
    let dir = tempfile::tempdir().expect("tempdir");
    let key = b"node:1:promised";
    let promised = {
        let engine = reopen(dir.path());
        keep_all_history(&engine);
        engine.put(Partition::Node, key, b"v1").expect("v1");
        let at = engine.snapshot();
        let reg = local_registry(&engine).expect("registry");
        reg.register(ConsumerRegistration {
            kind: ConsumerKind::MvccSnapshotPin,
            ..registration("backup", at, ConsumerRetentionPolicy::Strict)
        })
        .expect("register");
        engine.put(Partition::Node, key, b"v2").expect("v2");
        engine.put(Partition::Node, key, b"v3").expect("v3");
        engine.persist().expect("persist");
        at
    };

    let engine = reopen(dir.path());
    assert!(engine.consumer_protections_pending());
    engine.set_retention_window(Duration::from_millis(1));
    engine.persist().expect("persist");
    engine.major_compact(Partition::Node).expect("compact");

    let reg = local_registry(&engine).expect("recover");
    assert!(!engine.consumer_protections_pending());
    assert_eq!(
        engine
            .snapshot_get(&promised, Partition::Node, key)
            .expect("the promised snapshot is still served")
            .as_deref(),
        Some(&b"v1"[..])
    );
    let h = reg.resume("backup", 1).expect("resume");
    assert_eq!(reg.check_retention(&h).expect("held"), promised);
}

/// A registration the reopened store cannot read refuses the registry, and
/// the store keeps holding all history: an unrecoverable protection is never
/// taken for none. Before, the registry opened with a warning and no floor,
/// and the window collected below the lost registration.
#[test]
fn an_unreadable_registration_refuses_the_registry_and_keeps_history() {
    let dir = tempfile::tempdir().expect("tempdir");
    {
        let engine = reopen(dir.path());
        engine
            .put(
                Partition::Registry,
                b"consumer:broken",
                b"\xc1 not a record",
            )
            .expect("write");
        engine.persist().expect("persist");
    }
    let engine = reopen(dir.path());
    engine.set_retention_window(Duration::from_millis(1));
    assert_eq!(engine.gc_watermark(), 0, "held from the open");
    assert!(matches!(
        local_registry(&engine),
        Err(RegistryError::Replication(_))
    ));
    assert_eq!(engine.gc_watermark(), 0, "still held after the refusal");
    assert!(engine.consumer_protections_pending());
}

/// A partition cleared above the collection watermark no longer serves reads
/// below its floor, whatever the watermark says: a registration there is
/// refused with that floor, one from the earliest retained starts at it, and
/// a registration it overtook reports its checkpoint lost. Before, all three
/// went by the watermark and promised history the store no longer had.
#[test]
fn a_cleared_partitions_floor_bounds_what_registrations_are_promised() {
    let (reg, _clock, _source, _pipeline, _dir) = hooked_registry();
    let engine = Arc::clone(&reg.core.engine);
    let mvcc = |id: &str, initial_seqno| ConsumerRegistration {
        kind: ConsumerKind::MvccSnapshotPin,
        initial_seqno,
        ..registration(id, 0, ConsumerRetentionPolicy::Strict)
    };
    engine
        .put(Partition::Node, b"node:1:a", b"v")
        .expect("write");
    let before = engine.snapshot();
    let early = reg
        .register(mvcc("early", InitialSeqno::At(before)))
        .expect("register");
    engine.clear_partition(Partition::Node).expect("clear");
    let horizon = engine.oldest_readable_seqno();
    assert!(
        horizon > before && engine.gc_watermark() <= before,
        "the floor is above the watermark: {horizon} vs {}",
        engine.gc_watermark()
    );

    assert!(matches!(
        reg.register(mvcc("late", InitialSeqno::At(before))),
        Err(RegistryError::RetentionLost { checkpoint, floor })
            if checkpoint == before && floor == horizon
    ));
    let earliest = reg
        .register(mvcc("earliest", InitialSeqno::FromEarliestRetained))
        .expect("register");
    assert_eq!(reg.check_retention(&earliest).expect("held"), horizon);
    assert!(matches!(
        reg.check_retention(&early),
        Err(RegistryError::RetentionLost { checkpoint, floor })
            if checkpoint == before && floor == horizon
    ));
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
    keep_all_history(&engine);
    let pipeline = Arc::new(HookPipeline {
        inner: coordinode_storage::engine::local_pipeline::OwnedLocalProposalPipeline::new(&engine),
        hook: Mutex::new(None),
        later_hooks: Mutex::new(std::collections::VecDeque::new()),
        follower: AtomicBool::new(false),
        failing: AtomicBool::new(false),
        own_work: Mutex::new(Vec::new()),
    });
    let clock = Arc::new(ManualClock::new(0));
    let source = FakeSource::new();
    let reg = ShardConsumerRegistry::new(
        Arc::clone(&engine),
        Arc::clone(&pipeline) as Arc<dyn ProposalPipeline>,
        Arc::new(ProposalIdGenerator::with_base(1u64 << 48)),
        Arc::clone(&clock) as Arc<dyn Clock>,
        Arc::clone(&source) as Arc<dyn RetentionSource>,
    )
    .expect("open the registry");
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
    reg.core.pending_hb.lock().admit("idle", 1, 100, usize::MAX);
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
        matches!(
            refused,
            Err(RegistryError::Backpressure(
                RetentionPressure::WritePressure
            ))
        ),
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

/// A registry over a store with no consensus log, measuring the store's own
/// material: what a state delta consumer is charged is the store's fold.
struct MaterialRegistry {
    reg: ShardConsumerRegistry,
    engine: Arc<StorageEngine>,
    pipeline: Arc<dyn ProposalPipeline>,
    ids: Arc<ProposalIdGenerator>,
    _dir: tempfile::TempDir,
}

fn material_registry() -> MaterialRegistry {
    let dir = tempfile::tempdir().expect("tempdir");
    // A clock ahead of the wall clock: the oracle's timestamps follow it.
    let start = 4_000_000_000_000_000u64;
    let oracle = Arc::new(
        coordinode_core::txn::timestamp::TimestampOracle::resume_from(
            coordinode_core::txn::timestamp::Timestamp::from_raw(start),
        ),
    );
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
    keep_all_history(&engine);
    let source: Arc<dyn RetentionSource> = Arc::new(crate::registry::LogRetentionSource::new(
        Arc::clone(&engine),
        Vec::new(),
        coordinode_storage::oplog::manager::OplogManager::open(
            &dir.path().join("log"),
            0,
            1 << 20,
            1_000,
            3_600,
        )
        .expect("open an empty log")
        .retained_floor(),
        Arc::new(|| 0),
    ));
    let pipeline: Arc<dyn ProposalPipeline> = Arc::new(
        coordinode_storage::engine::local_pipeline::OwnedLocalProposalPipeline::new(&engine),
    );
    let ids = Arc::new(ProposalIdGenerator::with_base(1u64 << 48));
    let reg = ShardConsumerRegistry::new(
        Arc::clone(&engine),
        Arc::clone(&pipeline),
        Arc::clone(&ids),
        Arc::new(ManualClock::new(start / 1_000)),
        source,
    )
    .expect("open the registry");
    MaterialRegistry {
        reg,
        engine,
        pipeline,
        ids,
        _dir: dir,
    }
}

/// Commit `len` bytes under `key` as a client transaction does, through the
/// admission every write that changes the material passes; its commit
/// timestamp.
fn commit_value(
    engine: &StorageEngine,
    pipeline: &dyn ProposalPipeline,
    ids: &ProposalIdGenerator,
    key: &[u8],
    len: usize,
) -> Result<u64, CommitError> {
    let oracle = engine.oracle().expect("an oracle");
    let mut txn = Transaction::begin(engine, Some(oracle.as_ref()), oracle.next());
    txn.put(Partition::Node, key, &vec![7; len])
        .expect("stage the write");
    let write_concern = WriteConcern::default();
    let ctx = CommitContext {
        write_concern: &write_concern,
        pipeline: Some(pipeline),
        id_gen: Some(ids),
    };
    txn.commit(&ctx)
        .map(|outcome| outcome.commit_ts.expect("a commit timestamp").as_raw())
}

/// A state delta consumer measured over its byte bound on a large value is
/// ended only if no write that shrank its material was ordered before the
/// ending: a client overwriting the value with a small one between the
/// measurement and the ordered ending lands after it, and the ending was
/// decided on the material as it stood at its own place in the order.
/// Before, the overwrite landed first and the consumer was ended on bytes
/// it no longer required.
#[test]
fn a_smaller_write_racing_an_ending_lands_after_it() {
    let m = material_registry();
    // Before every write the test makes.
    let checkpoint = m.engine.oracle().expect("an oracle").next().as_raw();
    m.reg
        .register(registration(
            "delta",
            checkpoint,
            bounded(1_000_000_000, 64 << 10, None),
        ))
        .expect("register");
    commit_value(
        &m.engine,
        m.pipeline.as_ref(),
        &m.ids,
        b"node:value",
        1 << 20,
    )
    .expect("the large value");
    let small_at = Arc::new(AtomicU64::new(0));
    let writer = Arc::new(Mutex::new(None));
    *m.reg.core.before_commit.lock() = Some(Box::new({
        let (engine, pipeline, ids) = (
            Arc::clone(&m.engine),
            Arc::clone(&m.pipeline),
            Arc::clone(&m.ids),
        );
        let (small_at, writer) = (Arc::clone(&small_at), Arc::clone(&writer));
        move || {
            // Between the measurement and the ordered ending, another client
            // replaces the value with a small one.
            let (done, finished) = std::sync::mpsc::channel();
            *writer.lock() = Some(std::thread::spawn(move || {
                let at = commit_value(&engine, pipeline.as_ref(), &ids, b"node:value", 4)
                    .expect("the small value commits");
                small_at.store(at, Ordering::Release);
                let _ = done.send(());
            }));
            // Given every chance to land first.
            let _ = finished.recv_timeout(Duration::from_millis(300));
        }
    }));
    m.reg.core.sweep_evictions().expect("sweep");
    if let Some(writer) = writer.lock().take() {
        writer.join().expect("the writer");
    }
    let ended_at = m
        .engine
        .record_version(Partition::Registry, &encode_registry_key("delta"))
        .expect("read the record")
        .expect("a record");
    assert_eq!(
        ended_for(&m.reg, "delta"),
        Some(TerminalReason::RetainedBytesExceeded),
        "the consumer requiring the large value is ended"
    );
    let small = small_at.load(Ordering::Acquire);
    assert!(
        small > ended_at,
        "the small value (at {small}) was ordered before the ending (at {ended_at}) decided on the large one"
    );
}

/// A consumer measured within its bound on a first look, then over it on
/// the final one (a large value landed between), is ended only with the
/// material scope taken: the final measure without it is not written, the
/// attempt takes the scope and reads again, and a client overwriting the
/// value with a small one between that measure and the ordered ending
/// lands after the ending. Before, the ending went out on the final measure
/// with no scope held, and the small write landed first.
#[test]
fn an_ending_found_only_on_the_final_measure_takes_the_scope_first() {
    let m = material_registry();
    let checkpoint = m.engine.oracle().expect("an oracle").next().as_raw();
    m.reg
        .register(registration(
            "delta",
            checkpoint,
            bounded(1_000_000_000, 64 << 10, None),
        ))
        .expect("register");
    commit_value(&m.engine, m.pipeline.as_ref(), &m.ids, b"node:value", 4).expect("small");
    // Between the first look (within the bound) and the decision.
    *m.reg.core.before_decide.lock() = Some(Box::new({
        let (engine, pipeline, ids) = (
            Arc::clone(&m.engine),
            Arc::clone(&m.pipeline),
            Arc::clone(&m.ids),
        );
        move || {
            commit_value(&engine, pipeline.as_ref(), &ids, b"node:value", 1 << 20)
                .expect("the large value");
        }
    }));
    let small_at = Arc::new(Mutex::new(None));
    let writer = Arc::new(Mutex::new(None));
    *m.reg.core.before_commit.lock() = Some(Box::new({
        let (engine, pipeline, ids) = (
            Arc::clone(&m.engine),
            Arc::clone(&m.pipeline),
            Arc::clone(&m.ids),
        );
        let (small_at, writer) = (Arc::clone(&small_at), Arc::clone(&writer));
        move || {
            let (done, finished) = std::sync::mpsc::channel();
            *writer.lock() = Some(std::thread::spawn(move || {
                *small_at.lock() = Some(commit_value(
                    &engine,
                    pipeline.as_ref(),
                    &ids,
                    b"node:value",
                    4,
                ));
                let _ = done.send(());
            }));
            let _ = finished.recv_timeout(Duration::from_millis(300));
        }
    }));
    m.reg.core.sweep_evictions().expect("sweep");
    if let Some(writer) = writer.lock().take() {
        writer.join().expect("the writer");
    }
    assert_eq!(
        ended_for(&m.reg, "delta"),
        Some(TerminalReason::RetainedBytesExceeded),
        "the consumer over its bound on the final measure is ended"
    );
    let ended_at = m
        .engine
        .record_version(Partition::Registry, &encode_registry_key("delta"))
        .expect("read the record")
        .expect("a record");
    let written = small_at.lock().take().expect("the writer ran");
    assert!(
        match &written {
            Ok(at) => *at > ended_at,
            Err(e) => matches!(e, CommitError::Backpressure),
        },
        "the small write {written:?} against the ending at {ended_at}"
    );
}

/// A pipeline that, once armed, hands the next ending to the log late, on
/// another thread, and tells its proposer at once that the outcome is not
/// known: the ending can still apply after the proposer gave up waiting.
struct LateEnding {
    inner: Arc<dyn ProposalPipeline>,
    armed: AtomicBool,
    late: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl ProposalPipeline for LateEnding {
    fn propose_and_wait(
        &self,
        proposal: &RaftProposal,
    ) -> Result<
        coordinode_core::txn::proposal::ProposalOutcome,
        coordinode_core::txn::proposal::ProposalError,
    > {
        use coordinode_core::txn::proposal::{MetadataCommand, Mutation, ProposalError};
        let ending = proposal.mutations.iter().any(|m| {
            matches!(
                m,
                Mutation::Command(MetadataCommand::ConditionalRegistryWrite { .. })
            )
        });
        if ending && self.armed.swap(false, Ordering::AcqRel) {
            let (inner, proposal) = (Arc::clone(&self.inner), proposal.clone());
            *self.late.lock() = Some(std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(300));
                let _ = inner.propose_and_wait(&proposal);
            }));
            return Err(ProposalError::Timeout { retries: 3 });
        }
        self.inner.propose_and_wait(proposal)
    }
}

/// An ending whose outcome is not known keeps the material scope until it
/// is: the ending, handed to the log late, applies on the large value it
/// measured; a client overwriting that value meanwhile waits, and lands (or
/// is refused as retryable) after it; the scope is released once a closed
/// bound past the ending applies. A lower timestamp held unlogged keeps
/// every closed bound below the ending meanwhile, so its late entry is
/// still taken.
#[tokio::test(flavor = "multi_thread")]
async fn an_ending_in_doubt_keeps_its_protection_until_its_outcome_is_known() {
    let DoubtRegistry {
        engine,
        oracle,
        node,
        raft,
        late,
        ids,
        reg,
        _dir,
    } = doubt_registry().await;
    let checkpoint = oracle.next().as_raw();
    reg.register(registration(
        "delta",
        checkpoint,
        bounded(1_000_000_000, 64 << 10, None),
    ))
    .expect("register");
    let large = commit_value(&engine, raft.as_ref(), &ids, b"node:value", 1 << 20)
        .expect("the large value");
    assert!(
        engine.await_complete_through(large, Duration::from_secs(10)),
        "the large value is never known complete"
    );
    let (_, low) = engine.pending_commits().obligate(|| oracle.next().as_raw());
    let small = Arc::new(Mutex::new(None));
    let writer = Arc::new(Mutex::new(None));
    *reg.core.before_commit.lock() = Some(Box::new({
        let (engine, raft, ids) = (Arc::clone(&engine), Arc::clone(&raft), Arc::clone(&ids));
        let (small, writer) = (Arc::clone(&small), Arc::clone(&writer));
        move || {
            *writer.lock() = Some(std::thread::spawn(move || {
                *small.lock() = Some(commit_value(&engine, raft.as_ref(), &ids, b"node:value", 4));
            }));
            std::thread::sleep(Duration::from_millis(100));
        }
    }));
    late.armed.store(true, Ordering::Release);
    reg.core.sweep_evictions().expect("sweep");
    // The ending applies late; once the held timestamp goes, a bound past it
    // settles the outcome and the scope is released.
    if let Some(late) = late.late.lock().take() {
        late.join().expect("the late ending");
    }
    drop(low);
    let deadline = Instant::now() + Duration::from_secs(10);
    while !reg.core.settle_unresolved() {
        assert!(Instant::now() < deadline, "the outcome was never settled");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    if let Some(writer) = writer.lock().take() {
        writer.join().expect("the writer");
    }
    assert_eq!(
        ended_for(&reg, "delta"),
        Some(TerminalReason::RetainedBytesExceeded),
        "the late ending applied"
    );
    let ended_at = engine
        .record_version(Partition::Registry, &encode_registry_key("delta"))
        .expect("read the record")
        .expect("a record");
    let written = small.lock().take().expect("the writer ran");
    assert!(
        match &written {
            Ok(at) => *at > ended_at,
            // Refused as retryable while it waited behind the decision.
            Err(e) => matches!(e, CommitError::Backpressure),
        },
        "the small write {written:?} against the late ending at {ended_at}"
    );
    node.shutdown().await.expect("shutdown");
}

/// A registry over a one-member Raft group whose pipeline can hand the next
/// registry write to the log late ([`LateEnding`]).
struct DoubtRegistry {
    engine: Arc<StorageEngine>,
    oracle: Arc<coordinode_core::txn::timestamp::TimestampOracle>,
    node: Arc<RaftNode>,
    raft: Arc<dyn ProposalPipeline>,
    late: Arc<LateEnding>,
    ids: Arc<ProposalIdGenerator>,
    reg: ShardConsumerRegistry,
    _dir: tempfile::TempDir,
}

async fn doubt_registry() -> DoubtRegistry {
    let dir = tempfile::tempdir().expect("tempdir");
    let start = 4_000_000_000_000_000u64;
    let oracle = Arc::new(
        coordinode_core::txn::timestamp::TimestampOracle::resume_from(
            coordinode_core::txn::timestamp::Timestamp::from_raw(start),
        ),
    );
    let engine = Arc::new(
        StorageEngine::open_with_oracle(
            &StorageConfig::with_endpoints(vec![EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            )]),
            Arc::clone(&oracle),
        )
        .expect("open engine"),
    );
    keep_all_history(&engine);
    let node = Arc::new(
        RaftNode::open_with_oracle(1, Arc::clone(&engine), Some(Arc::clone(&oracle)))
            .await
            .expect("raft node"),
    );
    let raft: Arc<dyn ProposalPipeline> =
        Arc::new(RaftProposalPipeline::new(Arc::clone(node.raft())));
    let late = Arc::new(LateEnding {
        inner: Arc::clone(&raft),
        armed: AtomicBool::new(false),
        late: Mutex::new(None),
    });
    let source: Arc<dyn RetentionSource> = Arc::new(crate::registry::LogRetentionSource::new(
        Arc::clone(&engine),
        Vec::new(),
        coordinode_storage::oplog::manager::OplogManager::open(
            &dir.path().join("log"),
            0,
            1 << 20,
            1_000,
            3_600,
        )
        .expect("open an empty log")
        .retained_floor(),
        Arc::new(|| 0),
    ));
    let ids = Arc::new(ProposalIdGenerator::with_base(1u64 << 48));
    let reg = ShardConsumerRegistry::new(
        Arc::clone(&engine),
        Arc::clone(&late) as Arc<dyn ProposalPipeline>,
        Arc::clone(&ids),
        Arc::new(ManualClock::new(start / 1_000)),
        source,
    )
    .expect("open the registry");
    DoubtRegistry {
        engine,
        oracle,
        node,
        raft,
        late,
        ids,
        reg,
        _dir: dir,
    }
}

/// A registry write that is no terminal decision (another consumer's
/// checkpoint) and whose outcome is not known, made while a sweep holds the
/// material scope for an ending, neither takes that scope nor reports its
/// outcome as a decision's: the checkpoint is retried and lands, and the
/// ending, known applied, leaves no decision in doubt behind. Before, the
/// scope lived in shared registry state, and the checkpoint moved it into
/// the doubt record under its own timestamp and returned `OutcomeUnknown`.
#[tokio::test(flavor = "multi_thread")]
async fn a_concurrent_write_in_doubt_does_not_take_a_decisions_scope() {
    let DoubtRegistry {
        engine,
        oracle,
        node,
        raft,
        late,
        ids,
        reg,
        _dir,
    } = doubt_registry().await;
    let reg = Arc::new(reg);
    let at = oracle.next().as_raw();
    reg.register(registration(
        "delta",
        at,
        bounded(1_000_000_000, 64 << 10, None),
    ))
    .expect("register the delta");
    let other = reg
        .register(registration(
            "other",
            at,
            bounded(1_000_000_000, 1 << 40, None),
        ))
        .expect("register the other");
    let large = commit_value(&engine, raft.as_ref(), &ids, b"node:value", 1 << 20)
        .expect("the large value");
    assert!(
        engine.await_complete_through(large, Duration::from_secs(10)),
        "the large value is never known complete"
    );
    // Keeps every closed bound below the checkpoint, so its outcome stays
    // unknown for as long as a decision would wait on it.
    let (_, low) = engine.pending_commits().obligate(|| oracle.next().as_raw());
    let checkpointed = Arc::new(Mutex::new(None));
    *reg.core.before_commit.lock() = Some(Box::new({
        let (reg, late, checkpointed) = (
            Arc::clone(&reg),
            Arc::clone(&late),
            Arc::clone(&checkpointed),
        );
        let seqno = oracle.next().as_raw();
        move || {
            // The sweep holds the scope here; the checkpoint is handed to
            // the log late.
            late.armed.store(true, Ordering::Release);
            *checkpointed.lock() = Some(reg.checkpoint(&other, seqno));
        }
    }));
    reg.core.sweep_evictions().expect("sweep");
    let doubt = reg
        .core
        .unresolved
        .lock()
        .as_ref()
        .map(|d| d.consumer_id.clone());
    if let Some(late) = late.late.lock().take() {
        late.join().expect("the late checkpoint");
    }
    drop(low);
    let checkpointed = checkpointed.lock().take().expect("the checkpoint ran");
    assert!(
        checkpointed.is_ok(),
        "a checkpoint is no decision: retried, not reported in doubt: {checkpointed:?}"
    );
    assert_eq!(doubt, None, "no decision was left in doubt");
    assert_eq!(
        ended_for(&reg, "delta"),
        Some(TerminalReason::RetainedBytesExceeded),
        "the ending applied"
    );
    node.shutdown().await.expect("shutdown");
}
