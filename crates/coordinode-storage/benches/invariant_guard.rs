//! What the invariant guard costs: throughput on a node everybody references,
//! the latency tail of a commit that has to state and decide conditions, and
//! the memory the guard holds while it does.
//!
//! The interesting case is skew. A popular node is referenced by most writes,
//! and the whole point of the claim model is that referencing it is a
//! compatible right rather than an exclusive one: the writes must not queue
//! behind each other. A mean throughput number would hide exactly that, so
//! this reports the distribution and the guard's own occupancy alongside it.
//!
//! Run: cargo bench -p coordinode-storage --bench invariant_guard

#![allow(clippy::expect_used, clippy::print_stdout)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use coordinode_core::graph::node::NodeId;
use coordinode_core::txn::invariant::{Adjacency, Claim, ClaimPredicate, ClaimScope};
use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::transaction::{CommitContext, Transaction};

/// Commits per measurement. Large enough for the tail to mean something,
/// small enough that the whole bench stays under a minute.
const COMMITS: usize = 4_000;

/// Writer threads for the skew measurement.
const WRITERS: usize = 8;

fn open(dir: &tempfile::TempDir) -> (Arc<StorageEngine>, Arc<TimestampOracle>) {
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let oracle = Arc::new(TimestampOracle::new());
    let engine = Arc::new(StorageEngine::open_with_oracle(&config, oracle.clone()).expect("open"));
    (engine, oracle)
}

fn commit_ctx(wc: &WriteConcern) -> CommitContext<'_> {
    CommitContext {
        write_concern: wc,
        pipeline: None,
        id_gen: None,
        drain_buffer: None,
        nvme_write_buffer: None,
    }
}

/// One edge attached to `hub`, with or without the conditions that protect it.
fn attach(
    engine: &StorageEngine,
    oracle: &TimestampOracle,
    hub: NodeId,
    peer: u64,
    claims: bool,
) -> Duration {
    let snap = engine.snapshot();
    let mut txn = Transaction::new(
        engine,
        Some(oracle),
        coordinode_core::txn::timestamp::Timestamp::from_raw(snap),
        Some(snap),
    );

    let mut key = Vec::with_capacity(24);
    key.extend_from_slice(b"adj:OWNS:out:");
    key.extend_from_slice(&hub.as_raw().to_be_bytes());
    txn.merge_adj_add(&key, peer);

    if claims {
        let generation = txn.schema_generation();
        txn.claim(Claim::new(
            ClaimScope::Node(hub),
            ClaimPredicate::EndpointAlive,
            generation,
        ));
        txn.claim(Claim::new(
            ClaimScope::Node(NodeId::from_raw(peer)),
            ClaimPredicate::EndpointAlive,
            generation,
        ));
    }

    let wc = WriteConcern::default();
    let ctx = commit_ctx(&wc);
    let started = Instant::now();
    txn.commit(&ctx).expect("commit");
    started.elapsed()
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let rank = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[rank]
}

fn report(label: &str, mut samples: Vec<Duration>, wall: Duration) {
    samples.sort_unstable();
    let total: Duration = samples.iter().sum();
    let mean = total / samples.len().max(1) as u32;
    println!(
        "{label:<34} n={:>6}  mean={:>9.3?}  p50={:>9.3?}  p99={:>9.3?}  p999={:>9.3?}  \
         throughput={:>9.0}/s",
        samples.len(),
        mean,
        percentile(&samples, 0.50),
        percentile(&samples, 0.99),
        percentile(&samples, 0.999),
        samples.len() as f64 / wall.as_secs_f64(),
    );
}

/// Every writer references the same node. With claims this is the case the
/// model exists for: the references are compatible, so the writers must not
/// serialise against each other.
fn skewed_node(claims: bool, peers_exist: bool) {
    let dir = tempfile::tempdir().expect("tempdir");
    let (engine, oracle) = open(&dir);
    let hub = NodeId::from_raw(1);

    let hub_key = coordinode_core::graph::node::encode_node_key(engine.node_shard(), hub);
    engine
        .put(Partition::Node, &hub_key, b"hub")
        .expect("put hub");

    // Whether the endpoints being attached to already have rows. They do in
    // any ordinary statement, which resolves them before writing; they do not
    // on the bulk paths that write adjacency ahead of node rows. The two cost
    // different amounts, because an absent row falls through to the version
    // scan and the written-since probe, so both are measured rather than one
    // of them being reported as the price of the guard.
    if peers_exist {
        for w in 0..WRITERS {
            for i in 0..COMMITS / WRITERS {
                let peer = NodeId::from_raw((w * COMMITS + i + 2) as u64);
                let key = coordinode_core::graph::node::encode_node_key(engine.node_shard(), peer);
                engine.put(Partition::Node, &key, b"peer").expect("put");
            }
        }
    }

    let started = Instant::now();
    let samples: Vec<Duration> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..WRITERS)
            .map(|w| {
                let engine = Arc::clone(&engine);
                let oracle = Arc::clone(&oracle);
                scope.spawn(move || {
                    let mut local = Vec::with_capacity(COMMITS / WRITERS);
                    for i in 0..COMMITS / WRITERS {
                        let peer = (w * COMMITS + i + 2) as u64;
                        local.push(attach(&engine, &oracle, hub, peer, claims));
                    }
                    local
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("writer"))
            .collect()
    });
    let wall = started.elapsed();

    let label = match (claims, peers_exist) {
        (false, _) => "skewed node, unguarded",
        (true, true) => "skewed node, guarded",
        (true, false) => "skewed node, guarded (absent peers)",
    };
    report(label, samples, wall);
}

/// What one attempt's conditions occupy while the commit decides them, and
/// what the table holds once it is done.
fn guard_memory() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (engine, oracle) = open(&dir);

    let snap = engine.snapshot();
    let mut txn = Transaction::new(
        engine.as_ref(),
        Some(oracle.as_ref()),
        coordinode_core::txn::timestamp::Timestamp::from_raw(snap),
        Some(snap),
    );

    // A statement that touches a large incident set states many conditions;
    // this is the shape the ceiling is sized against.
    let generation = txn.schema_generation();
    const REFERENCES: usize = 10_000;
    for peer in 0..REFERENCES as u64 {
        txn.claim(Claim::new(
            ClaimScope::Node(NodeId::from_raw(peer + 2)),
            ClaimPredicate::EndpointAlive,
            generation,
        ));
    }
    let bytes_per_claim = std::mem::size_of::<Claim>();
    println!(
        "guard occupancy                    claims={REFERENCES}  \
         struct={bytes_per_claim}B  set≈{}KiB (scope strings excluded)",
        (REFERENCES * bytes_per_claim) / 1024,
    );

    let wc = WriteConcern::default();
    let ctx = commit_ctx(&wc);
    txn.put(Partition::Node, b"node:guarded", b"v")
        .expect("put");
    let started = Instant::now();
    txn.commit(&ctx).expect("commit");
    println!(
        "deciding {REFERENCES} conditions            {:?}  \
         reserved after commit={}",
        started.elapsed(),
        engine.claim_registry().reserved_claims(),
    );
}

/// The refusal path: what a caller waits for before being told no. A refusal
/// has to be cheaper than the work it prevents, or the guard becomes the
/// bottleneck under contention.
fn refusal_latency() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (engine, oracle) = open(&dir);

    let mut key = Vec::new();
    key.extend_from_slice(b"adj:TAGGED:out:");
    key.extend_from_slice(&1u64.to_be_bytes());
    engine
        .merge(
            Partition::Adj,
            &key,
            &coordinode_storage::engine::merge::encode_add(2),
        )
        .expect("merge");

    let wc = WriteConcern::default();
    let mut samples = Vec::with_capacity(COMMITS);
    let started = Instant::now();
    for _ in 0..COMMITS {
        let snap = engine.snapshot();
        let mut txn = Transaction::new(
            engine.as_ref(),
            Some(oracle.as_ref()),
            coordinode_core::txn::timestamp::Timestamp::from_raw(snap),
            Some(snap),
        );
        txn.claim(Claim::new(
            ClaimScope::Pair {
                source: NodeId::from_raw(1),
                target: NodeId::from_raw(2),
                edge_type: "TAGGED".to_string(),
            },
            ClaimPredicate::PairAdjacency {
                observed: Adjacency::Absent,
            },
            txn.schema_generation(),
        ));
        txn.put(Partition::Node, b"node:never", b"v").expect("put");

        let ctx = commit_ctx(&wc);
        let at = Instant::now();
        txn.commit(&ctx).expect_err("the observation does not hold");
        samples.push(at.elapsed());
    }
    report("refusal", samples, started.elapsed());
}

fn main() {
    println!("== invariant guard ==");
    skewed_node(false, true);
    skewed_node(true, true);
    skewed_node(true, false);
    guard_memory();
    refusal_latency();
}
