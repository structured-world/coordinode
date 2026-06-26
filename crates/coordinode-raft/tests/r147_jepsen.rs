#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! R147 — Jepsen-style distributed-correctness suite (Rust-native, in-process).
//!
//! The targeted-invariant approach (vs a general linearizability checker or an
//! external Clojure Jepsen): we drive a real 3-node Raft cluster, inject faults
//! (nemeses), and assert the specific guarantees `consensus.md` promises and the
//! R147 "must pass" list names. The external Clojure+Elle harness is a separate
//! end-of-project deliverable (R926).
//!
//! Increment 1 (this file): **crash-fault invariants** —
//! - `no_data_loss_on_leader_crash`: majority-acked writes survive a leader crash.
//! - `read_your_writes_after_failover`: a client's acked write is readable from
//!   the new leader, and the cluster stays writable after failover.
//!
//! Increment 2 (this file): **network-partition nemesis** —
//! - `partition_minority_cannot_commit_majority_can`: isolate the leader into a
//!   minority; the majority elects a new leader and commits, the minority leader
//!   cannot commit, and on heal the minority converges with no split-brain.
//!
//! Later increments add: clock-skew nemesis and a single-register linearizability
//! checker over a concurrent client workload.

use std::sync::Arc;
use std::time::Duration;

use coordinode_core::txn::proposal::{
    Mutation, PartitionId, ProposalError, ProposalIdGenerator, ProposalPipeline, RaftProposal,
};
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_raft::cluster::{nemesis, RaftNode};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

/// Hard per-test timeout — a hung election/replication fails fast, never spins.
const TEST_TIMEOUT: Duration = Duration::from_secs(60);

// ── Cluster harness (self-contained; mirrors raft_cluster.rs) ───────────────

fn alloc_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind :0");
    let port = listener.local_addr().expect("local_addr").port();
    drop(listener);
    port
}

struct TestNode {
    node: RaftNode,
    engine: Arc<StorageEngine>,
    _dir: tempfile::TempDir,
}

fn engine_at(dir: &std::path::Path) -> Arc<StorageEngine> {
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    Arc::new(StorageEngine::open(&config).expect("open engine"))
}

async fn create_leader(node_id: u64, port: u16) -> TestNode {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = engine_at(dir.path());
    let listen_addr: std::net::SocketAddr = format!("127.0.0.1:{port}").parse().expect("addr");
    let node = RaftNode::open_cluster(
        node_id,
        Arc::clone(&engine),
        listen_addr,
        format!("http://127.0.0.1:{port}"),
    )
    .await
    .expect("open leader");
    TestNode {
        node,
        engine,
        _dir: dir,
    }
}

async fn create_follower(node_id: u64, port: u16) -> TestNode {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = engine_at(dir.path());
    let listen_addr: std::net::SocketAddr = format!("127.0.0.1:{port}").parse().expect("addr");
    let node = RaftNode::open_joining(node_id, Arc::clone(&engine), listen_addr)
        .await
        .expect("open joining");
    TestNode {
        node,
        engine,
        _dir: dir,
    }
}

async fn bootstrap_3_node() -> (TestNode, TestNode, TestNode) {
    let (p1, p2, p3) = (alloc_port(), alloc_port(), alloc_port());
    let n1 = create_leader(1, p1).await;
    let n2 = create_follower(2, p2).await;
    let n3 = create_follower(3, p3).await;
    tokio::time::sleep(Duration::from_millis(800)).await;
    n1.node
        .add_node(2, format!("http://127.0.0.1:{p2}"))
        .await
        .expect("add 2");
    n1.node
        .add_node(3, format!("http://127.0.0.1:{p3}"))
        .await
        .expect("add 3");
    n1.node
        .change_membership(vec![1, 2, 3])
        .await
        .expect("membership");
    tokio::time::sleep(Duration::from_millis(800)).await;
    (n1, n2, n3)
}

// ── Client ops ──────────────────────────────────────────────────────────────

/// Propose a single `Put` through `node`'s pipeline. Returns the committed Raft
/// index on success (the write was majority-committed), or the proposal error
/// (e.g. `NotLeader`).
fn write(
    node: &TestNode,
    id_gen: &ProposalIdGenerator,
    commit_ts: u64,
    key: &str,
    value: &str,
) -> Result<u64, ProposalError> {
    let proposal = RaftProposal {
        id: id_gen.next(),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: key.as_bytes().to_vec(),
            value: value.as_bytes().to_vec(),
        }],
        commit_ts: Timestamp::from_raw(commit_ts),
        start_ts: Timestamp::from_raw(commit_ts.saturating_sub(1)),
        bypass_rate_limiter: false,
    };
    node.node
        .pipeline()
        .propose_and_wait(&proposal)
        .map(|o| o.applied_index.unwrap_or(0))
}

/// Poll until a surviving node reports an elected leader within `survivors`
/// (and not `excluded`). Returns the new leader's node id.
async fn await_new_leader(survivors: &[&TestNode], excluded: u64) -> u64 {
    for _ in 0..150 {
        for n in survivors {
            if let Some(l) = n.node.current_leader() {
                if l != excluded && survivors.iter().any(|s| s.node.node_id() == l) {
                    return l;
                }
            }
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    panic!("no new leader elected (excluded {excluded})");
}

/// Read `key` from a node's applied engine state, retrying to absorb apply lag.
/// `Some(value)` once the key is visible, `None` if it never appears.
async fn await_value(engine: &StorageEngine, key: &str) -> Option<Vec<u8>> {
    for _ in 0..80 {
        if let Ok(Some(v)) = engine.get(Partition::Node, key.as_bytes()) {
            return Some(v.to_vec());
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    None
}

fn node_ref<'a>(nodes: &[&'a TestNode], id: u64) -> &'a TestNode {
    nodes
        .iter()
        .copied()
        .find(|n| n.node.node_id() == id)
        .expect("leader node present among survivors")
}

// ── Invariant: no data loss on leader crash ─────────────────────────────────

/// Every majority-acked write survives a leader crash: after the leader is
/// killed and the surviving majority elects a new leader, all acked values are
/// present on the new leader, and the cluster remains writable.
#[tokio::test(flavor = "multi_thread")]
async fn no_data_loss_on_leader_crash() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("openraft=off")
        .with_test_writer()
        .try_init();

    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let (n1, n2, n3) = bootstrap_3_node().await;
        let id_gen = ProposalIdGenerator::with_base(1u64 << 48);

        // Three majority-acked writes through the leader.
        let keys = ["node:1:dl-1", "node:1:dl-2", "node:1:dl-3"];
        for (i, k) in keys.iter().enumerate() {
            let idx = write(&n1, &id_gen, 100 + i as u64, k, &format!("v{i}"))
                .expect("leader write must commit");
            assert!(idx > 0, "committed write must carry a Raft index");
        }

        // Crash the leader.
        n1.node.shutdown().await.expect("shutdown leader");

        // Surviving majority elects a new leader.
        let survivors = [&n2, &n3];
        let new_leader = await_new_leader(&survivors, 1).await;
        let leader = node_ref(&survivors, new_leader);

        // No data loss: every acked write is present on the new leader.
        for (i, k) in keys.iter().enumerate() {
            let v = await_value(&leader.engine, k).await;
            assert_eq!(
                v.as_deref(),
                Some(format!("v{i}").as_bytes()),
                "acked write {k} lost after leader crash (new leader {new_leader})"
            );
        }

        // Cluster stays writable under the new leader.
        let id_gen2 = ProposalIdGenerator::with_base(new_leader << 48);
        let idx = write(leader, &id_gen2, 200, "node:1:dl-after", "after")
            .expect("new leader must accept writes after failover");
        assert!(idx > 0);

        n2.node.shutdown().await.ok();
        n3.node.shutdown().await.ok();
    })
    .await;
    assert!(result.is_ok(), "TIMED OUT after {TEST_TIMEOUT:?}");
}

// ── Invariant: read-your-writes after failover ──────────────────────────────

/// A client's acked write is still readable after the leader that acked it
/// crashes and a new leader takes over — the write is not silently lost across
/// the failover boundary (read-your-writes survives leader change).
#[tokio::test(flavor = "multi_thread")]
async fn read_your_writes_after_failover() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("openraft=off")
        .with_test_writer()
        .try_init();

    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let (n1, n2, n3) = bootstrap_3_node().await;
        let id_gen = ProposalIdGenerator::with_base(1u64 << 48);

        // Client writes and receives an ack (majority commit).
        let idx =
            write(&n1, &id_gen, 100, "node:1:ryw", "my-write").expect("leader write must commit");
        assert!(idx > 0);

        // The leader that acked the write crashes.
        n1.node.shutdown().await.expect("shutdown leader");

        // Failover.
        let survivors = [&n2, &n3];
        let new_leader = await_new_leader(&survivors, 1).await;
        let leader = node_ref(&survivors, new_leader);

        // The client reads its own write from the new leader — it is there.
        let v = await_value(&leader.engine, "node:1:ryw").await;
        assert_eq!(
            v.as_deref(),
            Some(b"my-write".as_slice()),
            "read-your-writes violated: acked write missing after failover to {new_leader}"
        );

        n2.node.shutdown().await.ok();
        n3.node.shutdown().await.ok();
    })
    .await;
    assert!(result.is_ok(), "TIMED OUT after {TEST_TIMEOUT:?}");
}

// ── Invariant: network partition — minority can't commit, majority can ──────

/// Isolate the leader into a minority. The surviving majority elects a new
/// leader and commits writes under the partition; the isolated old leader
/// cannot commit; on heal the old leader converges to the majority's state with
/// no split-brain (its uncommitted write never appears on the majority).
#[tokio::test(flavor = "multi_thread")]
async fn partition_minority_cannot_commit_majority_can() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("openraft=off")
        .with_test_writer()
        .try_init();
    nemesis::heal(); // defensive: clean matrix at start

    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let (n1, n2, n3) = bootstrap_3_node().await;
        let id_gen = ProposalIdGenerator::with_base(1u64 << 48);

        // Baseline write before the partition — committed on all three.
        write(&n1, &id_gen, 100, "node:1:p-base", "base").expect("baseline commit");

        // Partition: isolate the leader (node 1) from the majority {2,3}.
        nemesis::isolate(1, &[2, 3]);

        // Majority side elects a new leader and commits a write under partition.
        let survivors = [&n2, &n3];
        let new_leader = await_new_leader(&survivors, 1).await;
        let leader = node_ref(&survivors, new_leader);
        let id_gen_maj = ProposalIdGenerator::with_base(new_leader << 48);
        write(leader, &id_gen_maj, 200, "node:1:p-maj", "majority")
            .expect("majority must commit under partition");

        // Minority side: the isolated old leader cannot commit. Bound the
        // (blocking) propose so a never-committing write can't hang the test.
        let mino_pipeline = n1.node.pipeline();
        let mino_proposal = RaftProposal {
            id: ProposalIdGenerator::with_base(9u64 << 48).next(),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:1:p-mino".to_vec(),
                value: b"minority".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(300),
            start_ts: Timestamp::from_raw(299),
            bypass_rate_limiter: false,
        };
        let mino = tokio::time::timeout(
            Duration::from_secs(8),
            tokio::task::spawn_blocking(move || mino_pipeline.propose_and_wait(&mino_proposal)),
        )
        .await;
        // Committed only if the bounded join produced an Ok proposal outcome.
        let committed = matches!(mino, Ok(Ok(Ok(_))));
        assert!(
            !committed,
            "minority leader committed a write under partition (split-brain!)"
        );

        // Heal the partition; the old leader rejoins.
        nemesis::heal();

        // Convergence: the old minority leader catches up to the majority write.
        let v = await_value(&n1.engine, "node:1:p-maj").await;
        assert_eq!(
            v.as_deref(),
            Some(b"majority".as_slice()),
            "minority node did not converge to the majority write after heal"
        );

        // No split-brain: the minority's uncommitted write never reaches the new
        // leader (it was never committed; openraft truncates the uncommitted
        // suffix on rejoin).
        tokio::time::sleep(Duration::from_millis(500)).await;
        let leaked = leader
            .engine
            .get(Partition::Node, b"node:1:p-mino")
            .expect("read");
        assert!(
            leaked.is_none(),
            "minority's uncommitted write leaked to the majority (split-brain!)"
        );

        nemesis::heal();
        n1.node.shutdown().await.ok();
        n2.node.shutdown().await.ok();
        n3.node.shutdown().await.ok();
    })
    .await;
    nemesis::heal();
    assert!(result.is_ok(), "TIMED OUT after {TEST_TIMEOUT:?}");
}
