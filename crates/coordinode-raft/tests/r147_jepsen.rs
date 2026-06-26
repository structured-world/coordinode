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
//! Increment 3a (this file): **single-register linearizability checker** —
//! a Wing-Gong-Lowe checker (validated by its own unit tests) run over a
//! concurrent client workload against the real cluster
//! (`register_history_is_linearizable_under_concurrency`).
//!
//! Increment 3b (this file): **clock-skew nemesis** —
//! - `linearizable_despite_clock_skew`: one follower's HLC is seeded ~2s ahead
//!   of wall clock; the register history stays linearizable and the skewed node
//!   converges, since HLC gossip (advance-to-max on apply) keeps apply-seqnos
//!   monotonic regardless of wall-clock divergence.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use coordinode_core::txn::proposal::{
    Mutation, PartitionId, ProposalError, ProposalIdGenerator, ProposalPipeline, RaftProposal,
};
use coordinode_core::txn::timestamp::{HybridLogicalClock, Timestamp};
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

// ── Single-register linearizability checker (Wing-Gong-Lowe) ─────────────────
//
// A targeted invariant proves a named guarantee; a linearizability checker
// proves something stronger and harder to fake: that an *arbitrary* interleaving
// of concurrent reads and writes admits a single sequential order consistent
// with real-time. We record a history `(op, invoke, complete)` from the live
// cluster, then search for a valid linearization offline.

/// One register operation in a recorded history.
#[derive(Clone, Copy, Debug, PartialEq)]
enum RegOp {
    /// A write of a (unique) value.
    Write(u64),
    /// A read that observed `Some(v)`, or `None` for the empty initial register.
    Read(Option<u64>),
}

/// A history entry: the operation plus its real-time interval, in nanoseconds
/// measured from a common start. `invoke <= complete` always.
#[derive(Clone, Debug)]
struct HistEvent {
    op: RegOp,
    invoke: u64,
    complete: u64,
}

/// Wing-Gong-Lowe linearizability check for a single register (initial value
/// `None`). Returns `true` iff the history admits a total order that both
/// (a) respects real-time precedence — if op A completes before op B is invoked,
/// A precedes B — and (b) obeys register semantics (a read returns the value of
/// the most recent preceding write). Exponential worst-case, pruned by the
/// "minimal op" rule and memoised on dead-end `(remaining-set, value)` states.
/// Bounded to <= 32 events so the remaining-set fits a `u32` bitmask.
fn is_linearizable(history: &[HistEvent]) -> bool {
    assert!(
        history.len() <= 32,
        "history too large for u32 bitmask memo"
    );
    let full: u32 = if history.is_empty() {
        0
    } else {
        ((1u64 << history.len()) - 1) as u32
    };
    let mut dead: HashSet<(u32, Option<u64>)> = HashSet::new();
    linearize(history, full, None, &mut dead)
}

/// Recursive WGL search. `remaining` is the bitmask of not-yet-linearized ops;
/// `value` is the current model register value. `dead` memoises states proven
/// unlinearizable so sibling branches don't re-explore them.
fn linearize(
    h: &[HistEvent],
    remaining: u32,
    value: Option<u64>,
    dead: &mut HashSet<(u32, Option<u64>)>,
) -> bool {
    if remaining == 0 {
        return true;
    }
    if dead.contains(&(remaining, value)) {
        return false;
    }
    // Earliest completion among remaining ops. Any op invoked strictly after
    // this cannot be linearized next: the earliest-completing op finished before
    // it began, so that op must precede it in real time.
    let min_complete = (0..h.len())
        .filter(|&i| remaining & (1 << i) != 0)
        .map(|i| h[i].complete)
        .min()
        .unwrap_or(u64::MAX);
    for i in 0..h.len() {
        if remaining & (1 << i) == 0 {
            continue;
        }
        let e = &h[i];
        if e.invoke > min_complete {
            continue; // real-time: another remaining op must precede this one
        }
        let next = match e.op {
            RegOp::Write(v) => Some(v),
            RegOp::Read(r) => {
                if r != value {
                    continue; // register semantics violated for this order
                }
                value
            }
        };
        if linearize(h, remaining & !(1 << i), next, dead) {
            return true;
        }
    }
    dead.insert((remaining, value));
    false
}

#[test]
fn checker_accepts_valid_histories() {
    // Sequential write then read.
    assert!(is_linearizable(&[
        HistEvent {
            op: RegOp::Write(1),
            invoke: 0,
            complete: 2
        },
        HistEvent {
            op: RegOp::Read(Some(1)),
            invoke: 3,
            complete: 4
        },
    ]));
    // Read overlaps the write; linearizes as write-then-read.
    assert!(is_linearizable(&[
        HistEvent {
            op: RegOp::Write(1),
            invoke: 0,
            complete: 5
        },
        HistEvent {
            op: RegOp::Read(Some(1)),
            invoke: 1,
            complete: 2
        },
    ]));
    // An overlapping read may legally observe the *old* value (read linearizes
    // before the write): None here, since nothing was written yet.
    assert!(is_linearizable(&[
        HistEvent {
            op: RegOp::Write(1),
            invoke: 0,
            complete: 5
        },
        HistEvent {
            op: RegOp::Read(None),
            invoke: 1,
            complete: 2
        },
    ]));
    // Read of the empty initial register, fully before the first write.
    assert!(is_linearizable(&[
        HistEvent {
            op: RegOp::Read(None),
            invoke: 0,
            complete: 1
        },
        HistEvent {
            op: RegOp::Write(1),
            invoke: 2,
            complete: 3
        },
    ]));
}

#[test]
fn checker_rejects_nonlinearizable_histories() {
    // Reads a value that was never written.
    assert!(!is_linearizable(&[
        HistEvent {
            op: RegOp::Write(1),
            invoke: 0,
            complete: 1
        },
        HistEvent {
            op: RegOp::Read(Some(2)),
            invoke: 2,
            complete: 3
        },
    ]));
    // Reads a value whose write is entirely in the future (real-time violation).
    assert!(!is_linearizable(&[
        HistEvent {
            op: RegOp::Read(Some(1)),
            invoke: 0,
            complete: 1
        },
        HistEvent {
            op: RegOp::Write(1),
            invoke: 2,
            complete: 3
        },
    ]));
    // Stale read: W(1), then W(2) fully completes, then a non-overlapping read
    // still returns 1 — no order respects both real-time and register semantics.
    assert!(!is_linearizable(&[
        HistEvent {
            op: RegOp::Write(1),
            invoke: 0,
            complete: 1
        },
        HistEvent {
            op: RegOp::Write(2),
            invoke: 2,
            complete: 3
        },
        HistEvent {
            op: RegOp::Read(Some(1)),
            invoke: 4,
            complete: 5
        },
    ]));
}

// ── Invariant: concurrent register history is linearizable ──────────────────

/// A monotonic writer and three concurrent readers hammer a single register on
/// the live 3-node cluster. The recorded history must be linearizable: every
/// concurrent read interleaves with the write stream in a way that admits one
/// sequential order respecting real-time. Writes go through Raft (majority
/// commit); reads observe the leader's applied state.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn register_history_is_linearizable_under_concurrency() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("openraft=off")
        .with_test_writer()
        .try_init();

    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        let (n1, n2, n3) = bootstrap_3_node().await;

        let recorded = linearizability_workload(&n1, &n1.engine, "node:1:lin-reg").await;
        assert!(
            recorded.iter().any(|e| matches!(e.op, RegOp::Write(_))),
            "no writes committed; history is vacuous"
        );
        assert!(
            is_linearizable(&recorded),
            "register history is not linearizable:\n{recorded:#?}"
        );

        n1.node.shutdown().await.ok();
        n2.node.shutdown().await.ok();
        n3.node.shutdown().await.ok();
    })
    .await;
    assert!(result.is_ok(), "TIMED OUT after {TEST_TIMEOUT:?}");
}

/// A monotonic writer (values 0..5, monotonic commit_ts == log order) and three
/// concurrent readers hammer a single register, returning the recorded history.
/// Writes go through `writer`'s Raft pipeline (majority commit); reads observe
/// `read_engine`'s applied state. Shared by the no-skew and clock-skew tests.
async fn linearizability_workload(
    writer: &TestNode,
    read_engine: &Arc<StorageEngine>,
    key: &'static str,
) -> Vec<HistEvent> {
    let start = Instant::now();
    let history: Arc<Mutex<Vec<HistEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let mut tasks = Vec::new();

    {
        let pipeline = writer.node.pipeline();
        let hist = Arc::clone(&history);
        tasks.push(tokio::spawn(async move {
            let id_gen = ProposalIdGenerator::with_base(1u64 << 48);
            for v in 0..5u64 {
                let proposal = RaftProposal {
                    id: id_gen.next(),
                    mutations: vec![Mutation::Put {
                        partition: PartitionId::Node,
                        key: key.as_bytes().to_vec(),
                        value: v.to_string().into_bytes(),
                    }],
                    commit_ts: Timestamp::from_raw(1000 + v),
                    start_ts: Timestamp::from_raw(999 + v),
                    bypass_rate_limiter: false,
                };
                let invoke = start.elapsed().as_nanos() as u64;
                // block_in_place runs on a worker thread here (tokio::spawn), so
                // the proposal actually commits.
                let ok = pipeline.propose_and_wait(&proposal).is_ok();
                let complete = start.elapsed().as_nanos() as u64;
                if ok {
                    hist.lock().unwrap().push(HistEvent {
                        op: RegOp::Write(v),
                        invoke,
                        complete,
                    });
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
        }));
    }

    for _ in 0..3 {
        let eng = Arc::clone(read_engine);
        let hist = Arc::clone(&history);
        tasks.push(tokio::spawn(async move {
            for _ in 0..5 {
                let invoke = start.elapsed().as_nanos() as u64;
                let read = eng
                    .get(Partition::Node, key.as_bytes())
                    .ok()
                    .flatten()
                    .and_then(|b| String::from_utf8(b.to_vec()).ok())
                    .and_then(|s| s.parse::<u64>().ok());
                let complete = start.elapsed().as_nanos() as u64;
                hist.lock().unwrap().push(HistEvent {
                    op: RegOp::Read(read),
                    invoke,
                    complete,
                });
                tokio::time::sleep(Duration::from_millis(18)).await;
            }
        }));
    }

    for t in tasks {
        t.await.expect("client task");
    }
    let recorded = history.lock().unwrap().clone();
    recorded
}

// ── Clock-skew nemesis: linearizability holds despite per-node clock skew ────
//
// HLC safety does not depend on synchronised wall clocks: every node advances
// its clock to `max(local, observed)` on each Raft apply (the gossip step), so
// apply-seqnos stay monotonic cluster-wide even when one node's wall clock runs
// far ahead. This test seeds one follower ~2s ahead of wall and asserts the same
// register workload still produces a linearizable history.

/// Open an engine whose HLC is seeded `skew_us` microseconds ahead of wall clock
/// (a fast/skewed node). `resume_from` seeds at `max(wall, given)`, so passing
/// `wall + skew_us` puts the node's clock that far ahead.
fn engine_skewed(dir: &std::path::Path, skew_us: u64) -> Arc<StorageEngine> {
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let wall = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before epoch")
        .as_micros() as u64;
    let oracle = Arc::new(HybridLogicalClock::resume_from(Timestamp::from_raw(
        wall + skew_us,
    )));
    Arc::new(StorageEngine::open_with_oracle(&config, oracle).expect("open skewed engine"))
}

async fn create_follower_skewed(node_id: u64, port: u16, skew_us: u64) -> TestNode {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = engine_skewed(dir.path(), skew_us);
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

/// With one follower's clock skewed ~2s ahead of wall, the cluster still
/// produces a linearizable register history, and the skewed node converges to
/// the final write — clock skew does not break safety.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn linearizable_despite_clock_skew() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("openraft=off")
        .with_test_writer()
        .try_init();

    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        // 3-node cluster where node 2 runs ~2s ahead of wall clock.
        let (p1, p2, p3) = (alloc_port(), alloc_port(), alloc_port());
        let n1 = create_leader(1, p1).await;
        let n2 = create_follower_skewed(2, p2, 2_000_000).await; // +2s
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

        let key = "node:1:skew-reg";
        let recorded = linearizability_workload(&n1, &n1.engine, key).await;
        assert!(
            recorded.iter().any(|e| matches!(e.op, RegOp::Write(_))),
            "no writes committed under clock skew; history is vacuous"
        );
        assert!(
            is_linearizable(&recorded),
            "clock skew broke linearizability:\n{recorded:#?}"
        );

        // The skewed follower converged to the final write (gossip kept it in
        // step despite its fast clock).
        let v = await_value(&n2.engine, key).await;
        assert_eq!(
            v.as_deref(),
            Some(b"4".as_slice()),
            "skewed follower did not converge to the final write"
        );

        n1.node.shutdown().await.ok();
        n2.node.shutdown().await.ok();
        n3.node.shutdown().await.ok();
    })
    .await;
    assert!(result.is_ok(), "TIMED OUT after {TEST_TIMEOUT:?}");
}
