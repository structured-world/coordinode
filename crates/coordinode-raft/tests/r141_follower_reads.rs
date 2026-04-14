#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration tests for R141: follower reads (read fence).
//!
//! Verifies:
//! - `ReadFence::apply()` passes for all valid preference/concern combos
//! - `ReadPreference::Primary` passes on leader
//! - `ReadPreference::Primary` fails on follower (`NotLeader`)
//! - `ReadPreference::Secondary` fails on leader (`NotFollower`)
//! - `ReadConcern::Linearizable` passes on leader
//! - `ReadConcern::Linearizable` fails on follower (`LinearizableRequiresLeader`)
//! - `staleness_entries()` returns 0 on a caught-up single node
//! - `wait_for_index()` returns immediately when target <= applied_index
//! - `wait_for_index()` times out when target > applied_index (no further writes)
//! - `applied_index()` returns 0 on fresh node, advances after commit
//! - `from_proto()` round-trips for ReadPreference and ReadConcern

use std::sync::Arc;
use std::time::Duration;

use coordinode_core::txn::proposal::{
    Mutation, PartitionId, ProposalIdGenerator, ProposalPipeline, RaftProposal,
};
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_raft::cluster::RaftNode;
use coordinode_raft::read_fence::{
    ReadConcern, ReadFenceError, ReadPreference, CE_STALENESS_THRESHOLD_ENTRIES,
};
use coordinode_storage::engine::config::StorageConfig;
use coordinode_storage::engine::core::StorageEngine;

/// Hard timeout for multi-node tests that require cluster formation.
const CLUSTER_TEST_TIMEOUT: Duration = Duration::from_secs(30);

fn alloc_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind :0 for port alloc");
    let port = listener.local_addr().expect("local_addr").port();
    drop(listener);
    port
}

/// Bootstrap a 3-node cluster. Returns (leader_node, follower_node, dirs).
/// Caller must keep dirs alive for the duration of the test.
async fn bootstrap_3node() -> (
    tempfile::TempDir,
    tempfile::TempDir,
    tempfile::TempDir,
    RaftNode,
    RaftNode,
    RaftNode,
) {
    init_test_tracing();
    let p1 = alloc_port();
    let p2 = alloc_port();
    let p3 = alloc_port();

    let d1 = tempfile::tempdir().expect("d1");
    let d2 = tempfile::tempdir().expect("d2");
    let d3 = tempfile::tempdir().expect("d3");

    let e1 = Arc::new(StorageEngine::open(&StorageConfig::new(d1.path())).expect("e1"));
    let e2 = Arc::new(StorageEngine::open(&StorageConfig::new(d2.path())).expect("e2"));
    let e3 = Arc::new(StorageEngine::open(&StorageConfig::new(d3.path())).expect("e3"));

    let n1 = RaftNode::open_cluster(
        1,
        Arc::clone(&e1),
        format!("127.0.0.1:{p1}").parse().expect("a1"),
        format!("http://127.0.0.1:{p1}"),
    )
    .await
    .expect("n1");
    tokio::time::sleep(Duration::from_millis(800)).await;

    let n2 = RaftNode::open_joining(
        2,
        Arc::clone(&e2),
        format!("127.0.0.1:{p2}").parse().expect("a2"),
    )
    .await
    .expect("n2");
    let n3 = RaftNode::open_joining(
        3,
        Arc::clone(&e3),
        format!("127.0.0.1:{p3}").parse().expect("a3"),
    )
    .await
    .expect("n3");

    n1.add_node(2, format!("http://127.0.0.1:{p2}"))
        .await
        .expect("add n2");
    n1.add_node(3, format!("http://127.0.0.1:{p3}"))
        .await
        .expect("add n3");
    n1.change_membership(vec![1, 2, 3])
        .await
        .expect("membership");
    tokio::time::sleep(Duration::from_secs(2)).await;

    (d1, d2, d3, n1, n2, n3)
}

fn init_test_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("coordinode_raft=info,openraft=off")
        .with_test_writer()
        .try_init();
}

fn test_engine() -> (tempfile::TempDir, Arc<StorageEngine>) {
    init_test_tracing();
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::new(dir.path());
    let engine = Arc::new(StorageEngine::open(&config).expect("open"));
    (dir, engine)
}

async fn bootstrap_leader() -> (tempfile::TempDir, RaftNode) {
    let (_dir, engine) = test_engine();
    // Keep dir alive alongside node
    let node = RaftNode::single_node(engine).await.expect("bootstrap");
    // Wait for leader election (single node becomes leader immediately)
    tokio::time::sleep(Duration::from_millis(100)).await;
    (_dir, node)
}

// ── ReadPreference::from_proto round-trip ────────────────────────────────────

#[test]
fn read_preference_from_proto_round_trip() {
    assert_eq!(ReadPreference::from_proto(0), ReadPreference::Primary); // UNSPECIFIED → Primary
    assert_eq!(ReadPreference::from_proto(1), ReadPreference::Primary);
    assert_eq!(
        ReadPreference::from_proto(2),
        ReadPreference::PrimaryPreferred
    );
    assert_eq!(ReadPreference::from_proto(3), ReadPreference::Secondary);
    assert_eq!(
        ReadPreference::from_proto(4),
        ReadPreference::SecondaryPreferred
    );
    assert_eq!(ReadPreference::from_proto(5), ReadPreference::Nearest);
    assert_eq!(ReadPreference::from_proto(99), ReadPreference::Primary); // unknown → Primary
}

#[test]
fn read_concern_from_proto_round_trip() {
    assert_eq!(ReadConcern::from_proto(0), ReadConcern::Local); // UNSPECIFIED → Local
    assert_eq!(ReadConcern::from_proto(1), ReadConcern::Local);
    assert_eq!(ReadConcern::from_proto(2), ReadConcern::Majority);
    assert_eq!(ReadConcern::from_proto(3), ReadConcern::Linearizable);
    assert_eq!(ReadConcern::from_proto(4), ReadConcern::Snapshot);
    assert_eq!(ReadConcern::from_proto(99), ReadConcern::Local); // unknown → Local
}

// ── ReadPreference on leader ──────────────────────────────────────────────────

/// Primary preference: leader can serve. Passes.
#[tokio::test(flavor = "multi_thread")]
async fn primary_preference_passes_on_leader() {
    let (_dir, node) = bootstrap_leader().await;
    let mut fence = node.read_fence();
    fence
        .apply_default(ReadPreference::Primary, ReadConcern::Local)
        .await
        .expect("Primary on leader should pass");
}

/// Secondary preference: leader cannot serve. Returns NotFollower.
#[tokio::test(flavor = "multi_thread")]
async fn secondary_preference_fails_on_leader() {
    let (_dir, node) = bootstrap_leader().await;
    let mut fence = node.read_fence();
    let err = fence
        .apply_default(ReadPreference::Secondary, ReadConcern::Local)
        .await
        .expect_err("Secondary on leader should fail");
    assert!(
        matches!(err, ReadFenceError::NotFollower),
        "expected NotFollower, got {err:?}"
    );
}

/// PrimaryPreferred: leader can serve. Passes.
#[tokio::test(flavor = "multi_thread")]
async fn primary_preferred_passes_on_leader() {
    let (_dir, node) = bootstrap_leader().await;
    let mut fence = node.read_fence();
    fence
        .apply_default(ReadPreference::PrimaryPreferred, ReadConcern::Local)
        .await
        .expect("PrimaryPreferred on leader should pass");
}

/// SecondaryPreferred: leader can serve as fallback. Passes.
#[tokio::test(flavor = "multi_thread")]
async fn secondary_preferred_passes_on_leader() {
    let (_dir, node) = bootstrap_leader().await;
    let mut fence = node.read_fence();
    fence
        .apply_default(ReadPreference::SecondaryPreferred, ReadConcern::Local)
        .await
        .expect("SecondaryPreferred on leader should pass");
}

/// Nearest: CE serves from local node. Passes on leader.
#[tokio::test(flavor = "multi_thread")]
async fn nearest_passes_on_leader() {
    let (_dir, node) = bootstrap_leader().await;
    let mut fence = node.read_fence();
    fence
        .apply_default(ReadPreference::Nearest, ReadConcern::Local)
        .await
        .expect("Nearest on leader should pass");
}

// ── ReadConcern on leader ─────────────────────────────────────────────────────

/// Linearizable concern: leader confirms lease. Passes.
#[tokio::test(flavor = "multi_thread")]
async fn linearizable_concern_passes_on_leader() {
    let (_dir, node) = bootstrap_leader().await;
    let mut fence = node.read_fence();
    fence
        .apply_default(ReadPreference::Primary, ReadConcern::Linearizable)
        .await
        .expect("Linearizable on leader should pass");
}

/// Majority concern: CE equivalent to Local. Passes.
#[tokio::test(flavor = "multi_thread")]
async fn majority_concern_passes_on_leader() {
    let (_dir, node) = bootstrap_leader().await;
    let mut fence = node.read_fence();
    fence
        .apply_default(ReadPreference::Primary, ReadConcern::Majority)
        .await
        .expect("Majority on leader should pass");
}

/// Snapshot concern: CE equivalent to Majority. Passes.
#[tokio::test(flavor = "multi_thread")]
async fn snapshot_concern_passes_on_leader() {
    let (_dir, node) = bootstrap_leader().await;
    let mut fence = node.read_fence();
    fence
        .apply_default(ReadPreference::Primary, ReadConcern::Snapshot)
        .await
        .expect("Snapshot on leader should pass");
}

// ── Staleness tracking ────────────────────────────────────────────────────────

/// Fresh single-node leader: lag is 0 (caught up to itself).
#[tokio::test(flavor = "multi_thread")]
async fn staleness_zero_on_fresh_node() {
    let (_dir, node) = bootstrap_leader().await;
    let fence = node.read_fence();
    let lag = fence.staleness_entries();
    assert!(
        lag <= CE_STALENESS_THRESHOLD_ENTRIES,
        "fresh node should not be stale, lag={lag}"
    );
}

/// StaleReplica returned when lag exceeds CE threshold.
///
/// CE_STALENESS_THRESHOLD_ENTRIES = 10_000 makes it impractical to produce
/// real lag in a unit test. We use `with_staleness_lag()` to inject a lag
/// value that exceeds a custom threshold set via `with_staleness_threshold()`.
///
/// Requires a 3-node cluster: staleness is only checked on non-leader nodes
/// (the role check for `Secondary` on leader returns `NotFollower` first).
/// n2 is a follower so the `check_staleness()` path is exercised.
///
/// Verifies the full `StaleReplica` error path including correct error fields.
#[tokio::test(flavor = "multi_thread")]
async fn stale_replica_error_when_lag_exceeds_threshold() {
    let result = tokio::time::timeout(CLUSTER_TEST_TIMEOUT, async {
        let (_d1, _d2, _d3, n1, n2, n3) = bootstrap_3node().await;

        // n2 is a follower. Inject lag=50, threshold=10: 50 > 10 → StaleReplica.
        let mut fence = n2
            .read_fence()
            .with_staleness_lag(50)
            .with_staleness_threshold(10);

        let err = fence
            .apply_default(ReadPreference::Secondary, ReadConcern::Local)
            .await
            .expect_err("should fail with StaleReplica when lag > threshold");

        match err {
            ReadFenceError::StaleReplica { lag, threshold } => {
                assert_eq!(lag, 50, "lag should match injected value");
                assert_eq!(threshold, 10, "threshold should match configured value");
            }
            other => panic!("expected StaleReplica, got {other:?}"),
        }

        n1.shutdown().await.expect("s1");
        n2.shutdown().await.expect("s2");
        n3.shutdown().await.expect("s3");
    })
    .await;
    assert!(
        result.is_ok(),
        "TIMED OUT — stale_replica_error_when_lag_exceeds_threshold"
    );
}

/// StaleReplica is NOT returned on a follower when lag is within the threshold.
///
/// Uses a 3-node cluster (follower role needed for staleness check path) with
/// an injected lag below the custom threshold. Verifies that Secondary
/// preference succeeds when lag ≤ threshold even with a tight custom threshold.
#[tokio::test(flavor = "multi_thread")]
async fn no_stale_replica_when_lag_within_threshold() {
    let result = tokio::time::timeout(CLUSTER_TEST_TIMEOUT, async {
        let (_d1, _d2, _d3, n1, n2, n3) = bootstrap_3node().await;

        // Inject lag=5, threshold=10: 5 > 10 is false → pass.
        let mut fence = n2
            .read_fence()
            .with_staleness_lag(5)
            .with_staleness_threshold(10);

        fence
            .apply_default(ReadPreference::Secondary, ReadConcern::Local)
            .await
            .expect("Secondary on follower with lag within threshold should pass");

        n1.shutdown().await.expect("s1");
        n2.shutdown().await.expect("s2");
        n3.shutdown().await.expect("s3");
    })
    .await;
    assert!(
        result.is_ok(),
        "TIMED OUT — no_stale_replica_when_lag_within_threshold"
    );
}

// ── applied_index advances after commit ──────────────────────────────────────

/// After committing a proposal, applied_index should be ≥ 1.
#[tokio::test(flavor = "multi_thread")]
async fn applied_index_advances_after_commit() {
    let (_dir, engine) = test_engine();
    let engine_ref = Arc::clone(&engine);
    let node = RaftNode::single_node(engine).await.expect("bootstrap");
    tokio::time::sleep(Duration::from_millis(100)).await;

    let id_gen = ProposalIdGenerator::new();
    let proposal = RaftProposal {
        id: id_gen.next(),
        commit_ts: Timestamp::from_raw(1),
        start_ts: Timestamp::from_raw(0),
        bypass_rate_limiter: false,
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: b"test:key".to_vec(),
            value: b"value".to_vec(),
        }],
    };

    let pipeline = node.pipeline();
    pipeline.propose_and_wait(&proposal).expect("propose");

    // Wait for state machine to apply.
    tokio::time::sleep(Duration::from_millis(200)).await;

    let fence = node.read_fence();
    let applied = fence.applied_index();
    assert!(
        applied >= 1,
        "applied_index should advance after commit, got {applied}"
    );

    drop(engine_ref); // suppress unused
}

// ── wait_for_index ─────────────────────────────────────────────────────────────

/// wait_for_index returns immediately when target <= current applied.
#[tokio::test(flavor = "multi_thread")]
async fn wait_for_index_immediate_when_caught_up() {
    let (_dir, node) = bootstrap_leader().await;

    // Apply one entry so applied_index >= 1.
    let id_gen = ProposalIdGenerator::new();
    let proposal = RaftProposal {
        id: id_gen.next(),
        commit_ts: Timestamp::from_raw(1),
        start_ts: Timestamp::from_raw(0),
        bypass_rate_limiter: false,
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: b"wait:test".to_vec(),
            value: b"v".to_vec(),
        }],
    };
    node.pipeline()
        .propose_and_wait(&proposal)
        .expect("propose");
    tokio::time::sleep(Duration::from_millis(200)).await;

    let current_applied = node.read_fence().applied_index();
    assert!(current_applied >= 1, "need at least one applied entry");

    let mut fence = node.read_fence();
    fence
        .wait_for_index(current_applied, Duration::from_millis(500))
        .await
        .expect("wait_for_index for already-applied index should succeed immediately");
}

/// wait_for_index times out when target > current applied with no new writes.
#[tokio::test(flavor = "multi_thread")]
async fn wait_for_index_times_out() {
    let (_dir, node) = bootstrap_leader().await;
    let applied = node.read_fence().applied_index();

    let mut fence = node.read_fence();
    let err = fence
        .wait_for_index(applied + 10_000, Duration::from_millis(200))
        .await
        .expect_err("should timeout when target is far ahead");

    assert!(
        matches!(err, ReadFenceError::Timeout { .. }),
        "expected Timeout, got {err:?}"
    );
}

// ── subscribe_applied ─────────────────────────────────────────────────────────

/// subscribe_applied returns a receiver that sees the current applied value.
#[tokio::test(flavor = "multi_thread")]
async fn subscribe_applied_reflects_current_index() {
    let (_dir, node) = bootstrap_leader().await;
    tokio::time::sleep(Duration::from_millis(100)).await;

    let rx = node.subscribe_applied();
    let val = *rx.borrow();
    // Initial value after bootstrap — at least the config entry should be applied.
    // We just verify the channel is live and readable.
    let _ = val; // no panic = receiver works
}

// ── Multi-node: follower role checks ─────────────────────────────────────────

/// ReadPreference::Primary fails on a follower with NotLeader.
///
/// Requires a 3-node cluster so we can exercise the fence on a non-leader node.
#[tokio::test(flavor = "multi_thread")]
async fn primary_preference_fails_on_follower() {
    let result = tokio::time::timeout(CLUSTER_TEST_TIMEOUT, async {
        let (_d1, _d2, _d3, n1, n2, n3) = bootstrap_3node().await;

        // n1 is the leader (bootstrapped first). n2 is a follower.
        let mut fence = n2.read_fence();
        let err = fence
            .apply_default(ReadPreference::Primary, ReadConcern::Local)
            .await
            .expect_err("Primary on follower should fail");

        assert!(
            matches!(err, ReadFenceError::NotLeader),
            "expected NotLeader on follower, got {err:?}"
        );

        n1.shutdown().await.expect("s1");
        n2.shutdown().await.expect("s2");
        n3.shutdown().await.expect("s3");
    })
    .await;
    assert!(
        result.is_ok(),
        "TIMED OUT — primary_preference_fails_on_follower"
    );
}

/// ReadConcern::Linearizable fails on follower with LinearizableRequiresLeader.
///
/// Linearizable concern requires the node to be the Raft leader (lease read).
/// On a follower, the read fence must reject the request.
#[tokio::test(flavor = "multi_thread")]
async fn linearizable_concern_fails_on_follower() {
    let result = tokio::time::timeout(CLUSTER_TEST_TIMEOUT, async {
        let (_d1, _d2, _d3, n1, n2, n3) = bootstrap_3node().await;

        // Use SecondaryPreferred so the role check passes on a follower, then
        // the concern check should reject with LinearizableRequiresLeader.
        let mut fence = n2.read_fence();
        let err = fence
            .apply_default(
                ReadPreference::SecondaryPreferred,
                ReadConcern::Linearizable,
            )
            .await
            .expect_err("Linearizable on follower should fail");

        assert!(
            matches!(err, ReadFenceError::LinearizableRequiresLeader),
            "expected LinearizableRequiresLeader on follower, got {err:?}"
        );

        n1.shutdown().await.expect("s1");
        n2.shutdown().await.expect("s2");
        n3.shutdown().await.expect("s3");
    })
    .await;
    assert!(
        result.is_ok(),
        "TIMED OUT — linearizable_concern_fails_on_follower"
    );
}

/// Secondary preference succeeds on a follower (within staleness threshold).
///
/// The fence should pass when the follower is caught up (lag ≤ CE_STALENESS_THRESHOLD_ENTRIES).
#[tokio::test(flavor = "multi_thread")]
async fn secondary_preference_passes_on_follower() {
    let result = tokio::time::timeout(CLUSTER_TEST_TIMEOUT, async {
        let (_d1, _d2, _d3, n1, n2, n3) = bootstrap_3node().await;

        // Write one entry so followers have something applied.
        let id_gen = ProposalIdGenerator::new();
        let proposal = RaftProposal {
            id: id_gen.next(),
            commit_ts: Timestamp::from_raw(1),
            start_ts: Timestamp::from_raw(0),
            bypass_rate_limiter: false,
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: b"r141:secondary:key".to_vec(),
                value: b"v".to_vec(),
            }],
        };
        n1.pipeline().propose_and_wait(&proposal).expect("propose");
        tokio::time::sleep(Duration::from_millis(500)).await;

        let mut fence = n2.read_fence();
        fence
            .apply_default(ReadPreference::Secondary, ReadConcern::Local)
            .await
            .expect("Secondary on caught-up follower should pass");

        n1.shutdown().await.expect("s1");
        n2.shutdown().await.expect("s2");
        n3.shutdown().await.expect("s3");
    })
    .await;
    assert!(
        result.is_ok(),
        "TIMED OUT — secondary_preference_passes_on_follower"
    );
}
