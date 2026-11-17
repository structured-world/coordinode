//! What an idle consensus group holds in memory, and what a burst of writes
//! larger than its initial queues costs.
//!
//! Sections:
//! - `idle`: heap bytes a single-member group keeps once it leads and sits
//!   idle, per group, at each group count of `GROUP_RESERVE_COUNTS` (default
//!   `1,100`). The engines are opened first, outside the count, so the figure
//!   is the consensus side of a group alone: its core, its tasks, its queues
//!   and their reserves. The groups run on a current-thread runtime, so every
//!   task they spawn allocates on the counted thread; storage threads of the
//!   engines are not counted;
//! - `burst`: one group answering bursts of `GROUP_RESERVE_BURST` (default
//!   256) writers at once, one write each. The first burst is the one that
//!   grows a queue past its initial reserve; the later twenty reuse it.
//!
//! Kept bytes are allocations made while counting and still live at its end,
//! so a reserve shows whether or not its pages were ever touched. Latencies
//! are distributions, a P99 printed from 100 samples up.
//!
//! Run: cargo bench -p coordinode-raft --bench group_reserve

#![allow(clippy::expect_used, clippy::print_stdout)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use coordinode_core::txn::proposal::{
    Mutation, PartitionId, ProposalIdGenerator, ProposalPipeline, RaftProposal,
};
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_raft::cluster::RaftNode;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_test_fixtures::alloc_peak::peak_and_kept_of;

#[global_allocator]
static ALLOCATOR: tracking_allocator::Allocator<std::alloc::System> =
    tracking_allocator::Allocator::system();

/// Whether `section` runs: all unless `GROUP_RESERVE_SECTION` names it.
fn runs(section: &str) -> bool {
    std::env::var("GROUP_RESERVE_SECTION").map_or(true, |only| {
        only.split(',').any(|name| name.trim() == section)
    })
}

fn counts() -> Vec<usize> {
    std::env::var("GROUP_RESERVE_COUNTS")
        .unwrap_or_else(|_| "1,100".into())
        .split(',')
        .map(|n| n.trim().parse().expect("GROUP_RESERVE_COUNTS lists counts"))
        .collect()
}

fn burst_size() -> usize {
    std::env::var("GROUP_RESERVE_BURST")
        .ok()
        .map(|n| n.parse().expect("GROUP_RESERVE_BURST is a count"))
        .unwrap_or(256)
}

fn open_store() -> (tempfile::TempDir, Arc<StorageEngine>) {
    let dir = tempfile::tempdir().expect("dir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = Arc::new(StorageEngine::open(&config).expect("open"));
    (dir, engine)
}

async fn await_leadership(node: &RaftNode) {
    for _ in 0..3_000 {
        if node.is_leader().await {
            return;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    assert!(node.is_leader().await, "no leadership");
}

async fn released(engine: Arc<StorageEngine>) {
    for _ in 0..2_000 {
        if Arc::strong_count(&engine) == 1 {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(
        Arc::strong_count(&engine),
        1,
        "a stopped node holds its engine"
    );
}

/// Heap bytes kept per idle group at each count.
fn idle() {
    for groups in counts() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        let stores: Vec<_> = (0..groups).map(|_| open_store()).collect();
        let (nodes, peak, kept) = peak_and_kept_of(|| {
            runtime.block_on(async {
                let mut nodes = Vec::with_capacity(groups);
                for (_, engine) in &stores {
                    let node = RaftNode::open(1, Arc::clone(engine)).await.expect("open");
                    await_leadership(&node).await;
                    nodes.push(node);
                }
                // Idle: past the first heartbeats and the elections' own work.
                tokio::time::sleep(Duration::from_secs(2)).await;
                nodes
            })
        });
        println!(
            "idle groups={groups:>4}  kept={kept} bytes  per group={:.1} KiB  peak while opening={:.1} KiB per group",
            kept as f64 / groups as f64 / 1024.0,
            peak as f64 / groups as f64 / 1024.0
        );
        runtime.block_on(async {
            for node in nodes {
                node.shutdown().await.expect("stop");
            }
            for (dir, engine) in stores {
                released(engine).await;
                drop(dir);
            }
        });
    }
}

fn put(id_gen: &ProposalIdGenerator, n: u64) -> RaftProposal {
    let ts = 1_000_000 + n;
    RaftProposal {
        id: id_gen.next(),
        mutations: coordinode_storage::engine::scope::store_unit([Mutation::Put {
            partition: PartitionId::Node,
            key: format!("node:1:{n}").into_bytes(),
            value: vec![7; 64],
        }]),
        commit_ts: Timestamp::from_raw(ts),
        start_ts: Timestamp::from_raw(ts - 1),
        bypass_rate_limiter: false,
    }
}

fn report(label: &str, mut samples: Vec<Duration>) {
    samples.sort_unstable();
    let n = samples.len();
    let at = |p: f64| samples[((n - 1) as f64 * p).round() as usize];
    let p99 = if n >= 100 {
        format!("{:>10.3?}", at(0.99))
    } else {
        format!("{:>10}", "n/a")
    };
    println!(
        "{label:<28} n={n:>5}  p50={:>10.3?}  p99={p99}  max={:>10.3?}",
        at(0.50),
        samples[n - 1]
    );
}

/// Bursts of concurrent writes into one group: the first grows its queues,
/// the rest reuse them.
fn bursts() {
    let writers = burst_size();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .max_blocking_threads(writers.max(512))
        .build()
        .expect("runtime");
    runtime.block_on(async {
        let (_dir, engine) = open_store();
        let node = RaftNode::open(1, Arc::clone(&engine)).await.expect("open");
        await_leadership(&node).await;
        let pipeline = Arc::new(node.pipeline());
        let id_gen = Arc::new(ProposalIdGenerator::with_base(1 << 40));
        let mut later = Vec::new();
        for round in 0..21u64 {
            let wall = Instant::now();
            let mut tasks = Vec::with_capacity(writers);
            for w in 0..writers as u64 {
                let (pipeline, id_gen) = (Arc::clone(&pipeline), Arc::clone(&id_gen));
                tasks.push(tokio::task::spawn_blocking(move || {
                    let proposal = put(&id_gen, round * 1_000_000 + w);
                    let t = Instant::now();
                    pipeline.propose_and_wait(&proposal).expect("commit");
                    t.elapsed()
                }));
            }
            let mut samples = Vec::with_capacity(writers);
            for task in tasks {
                samples.push(task.await.expect("a writer"));
            }
            if round == 0 {
                println!("burst of {writers}: first, wall {:.3?}", wall.elapsed());
                report("burst first: write", samples);
            } else {
                later.extend(samples);
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        report("burst later (20): write", later);
        drop(pipeline);
        node.shutdown().await.expect("stop");
        drop(node);
        released(engine).await;
    });
}

fn main() {
    println!("== group reserve ==");
    if runs("idle") {
        idle();
    }
    if runs("burst") {
        bursts();
    }
}
