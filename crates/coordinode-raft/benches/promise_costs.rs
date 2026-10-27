//! What keeping a member's consensus promises costs.
//!
//! Sections, each with its own measurement boundaries:
//! - `vote`: a vote change through the log store;
//! - `vote-same`: saving the vote the store already holds;
//! - `commit`: a commit through a single-member group, the steady write path;
//! - `restart`: a single-member group from open to leadership, and its
//!   shutdown, separately;
//! - `failover`: a three-member group, from the leader's stop to a new
//!   leader as its metrics show it and as a read of its lease confirms it,
//!   and the stopped member's reopen until it has caught up.
//!
//! Latencies are distributions; a P99 is printed only from 100 samples or
//! more, below that the median, the maximum and the samples themselves.
//! Syncs are counted through the storage filesystem; the Raft log and the
//! local journal are written past it, so these counts are partial. The full
//! count comes from a syscall trace of one section at two operation counts:
//! `PROMISE_COSTS_SECTION=vote PROMISE_COSTS_OPS=N` under `strace -f -c`
//! at N and 2N, the difference over N being the syncs per operation without
//! the section's setup.
//!
//! The copy's slot files are counted, with their logical length, between
//! writes at each restart and reopen. Their high water inside a write and
//! the bytes allocated to them are not observable from here; the copy's
//! tests measure them.
//!
//! Run: cargo bench -p coordinode-raft --bench promise_costs

#![allow(clippy::expect_used, clippy::print_stdout)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use coordinode_core::txn::proposal::{
    Mutation, PartitionId, ProposalIdGenerator, ProposalPipeline, RaftProposal,
};
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_raft::cluster::{Enrollment, RaftNode};
use coordinode_raft::storage::{LogStore, Vote};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_test_fixtures::{TestPki, alloc_port};
use lsm_tree::fs::{FaultFs, FaultInjector, StdFs, SyncKind};
use openraft::async_runtime::watch::WatchReceiver as _;
use openraft::storage::RaftLogStorage;

/// The smallest sample a P99 is printed from.
const P99_SAMPLES: usize = 100;
const RESTARTS: usize = 100;
const FAILOVERS: usize = 20;

/// Operations of the per-operation sections, `PROMISE_COSTS_OPS` or these.
fn ops(default: u64) -> u64 {
    std::env::var("PROMISE_COSTS_OPS")
        .ok()
        .map(|n| n.parse().expect("PROMISE_COSTS_OPS is a count"))
        .unwrap_or(default)
}

/// Whether `section` runs: all of them unless `PROMISE_COSTS_SECTION` names one.
fn runs(section: &str) -> bool {
    std::env::var("PROMISE_COSTS_SECTION").map_or(true, |only| only == section)
}

/// A store whose filesystem counts the syncs it is asked for.
struct Store {
    syncs: Arc<FaultInjector>,
    config: StorageConfig,
    dir: tempfile::TempDir,
}

/// Syncs counted through the storage filesystem: files, directories.
type Syncs = (usize, usize);

impl Store {
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("dir");
        let syncs = Arc::new(FaultInjector::new());
        let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            dir.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )])
        .with_fs(Arc::new(FaultFs::with_injector(StdFs, Arc::clone(&syncs))));
        Self { syncs, config, dir }
    }

    fn open(&self) -> Arc<StorageEngine> {
        Arc::new(StorageEngine::open(&self.config).expect("open"))
    }

    /// Syncs since the last call.
    fn take_syncs(&self) -> Syncs {
        let syncs = self.syncs.take_syncs();
        let directories = syncs
            .iter()
            .filter(|(_, kind)| *kind == SyncKind::Directory)
            .count();
        (syncs.len() - directories, directories)
    }

    /// The slot files of the second copy of the consensus records and their
    /// logical length, as they stand between writes. Not the peak: a write
    /// holds one more slot until it removes the spent ones, and the
    /// allocated bytes are the footprint's to count.
    fn copy_slots(&self) -> (usize, u64) {
        let mut slots = (0, 0);
        for entry in std::fs::read_dir(self.dir.path()).expect("list the store") {
            let entry = entry.expect("a store entry");
            if entry
                .file_name()
                .to_string_lossy()
                .starts_with("consensus-vote")
            {
                slots.0 += 1;
                slots.1 += entry.metadata().expect("a slot's metadata").len();
            }
        }
        slots
    }
}

/// The copy's slots as observed between writes: the most seen, and the last.
#[derive(Default)]
struct Slots {
    most: (usize, u64),
    last: (usize, u64),
}

impl Slots {
    fn observe(&mut self, now: (usize, u64)) {
        self.most = (self.most.0.max(now.0), self.most.1.max(now.1));
        self.last = now;
    }

    fn report(&self, label: &str) {
        println!(
            "{label:<30} copy slots between writes: most={} ({} logical bytes)  last={} ({} logical bytes)",
            self.most.0, self.most.1, self.last.0, self.last.1
        );
    }
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    let rank = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[rank]
}

/// A latency distribution and, when counted over the same span, the syncs.
fn report(label: &str, mut samples: Vec<Duration>, syncs: Option<Syncs>) {
    let raw = samples.clone();
    samples.sort_unstable();
    let n = samples.len();
    let p99 = if n >= P99_SAMPLES {
        format!("{:>10.3?}", percentile(&samples, 0.99))
    } else {
        format!("{:>10}", "n/a")
    };
    let syncs = match syncs {
        Some((files, dirs)) => format!(
            "fs file syncs/op={:>7.2}  fs dir syncs/op={:>6.2}",
            files as f64 / n as f64,
            dirs as f64 / n as f64
        ),
        None => "fs syncs not counted over this span".to_string(),
    };
    println!(
        "{label:<30} n={n:>5}  p50={:>10.3?}  p99={p99}  max={:>10.3?}  {syncs}",
        percentile(&samples, 0.50),
        samples[n - 1],
    );
    if n < P99_SAMPLES {
        println!("{:<30} samples: {raw:?}", "");
    }
}

async fn vote_changes() {
    let store = Store::new();
    let mut log = LogStore::open(store.open()).expect("log");
    store.take_syncs();
    let mut samples = Vec::new();
    for term in 1..=ops(300) {
        let t = Instant::now();
        log.save_vote(&Vote::new(term, 1)).await.expect("vote");
        samples.push(t.elapsed());
    }
    report("vote change", samples, Some(store.take_syncs()));
}

async fn same_votes() {
    let store = Store::new();
    let mut log = LogStore::open(store.open()).expect("log");
    let vote = Vote::new(1, 1);
    log.save_vote(&vote).await.expect("vote");
    store.take_syncs();
    let mut samples = Vec::new();
    for _ in 0..ops(300) {
        let t = Instant::now();
        log.save_vote(&vote).await.expect("the same vote");
        samples.push(t.elapsed());
    }
    report("same vote saved again", samples, Some(store.take_syncs()));
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

/// Wait until every handle on `engine` but this one is gone, so its store
/// can open again.
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
        "a stopped node still holds its engine"
    );
}

fn put(id_gen: &ProposalIdGenerator, n: u64) -> RaftProposal {
    let ts = 1_000_000 + n;
    RaftProposal {
        id: id_gen.next(),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: format!("node:1:{n}").into_bytes(),
            value: vec![7; 64],
        }],
        commit_ts: Timestamp::from_raw(ts),
        start_ts: Timestamp::from_raw(ts - 1),
        bypass_rate_limiter: false,
    }
}

async fn commits() {
    let store = Store::new();
    let engine = store.open();
    let node = RaftNode::open(1, Arc::clone(&engine)).await.expect("open");
    await_leadership(&node).await;
    let pipeline = node.pipeline();
    let id_gen = ProposalIdGenerator::with_base(1 << 40);
    store.take_syncs();
    let mut samples = Vec::new();
    for n in 0..ops(1_000) {
        let proposal = put(&id_gen, n);
        let t = Instant::now();
        tokio::task::block_in_place(|| pipeline.propose_and_wait(&proposal)).expect("commit");
        samples.push(t.elapsed());
    }
    report("commit (one member)", samples, Some(store.take_syncs()));
    drop(pipeline);
    node.shutdown().await.expect("shutdown");
    drop(node);
    released(engine).await;
}

async fn restarts() {
    let store = Store::new();
    {
        let engine = store.open();
        let node = RaftNode::open(1, Arc::clone(&engine)).await.expect("open");
        await_leadership(&node).await;
        node.shutdown().await.expect("shutdown");
        drop(node);
        released(engine).await;
    }
    let mut opens = Vec::new();
    let mut shutdowns = Vec::new();
    let (mut open_syncs, mut shutdown_syncs) = ((0, 0), (0, 0));
    let mut slots = Slots::default();
    for _ in 0..RESTARTS {
        store.take_syncs();
        let t = Instant::now();
        let engine = store.open();
        let node = RaftNode::open(1, Arc::clone(&engine))
            .await
            .expect("reopen");
        await_leadership(&node).await;
        opens.push(t.elapsed());
        let (files, dirs) = store.take_syncs();
        open_syncs = (open_syncs.0 + files, open_syncs.1 + dirs);
        slots.observe(store.copy_slots());
        let t = Instant::now();
        node.shutdown().await.expect("shutdown");
        drop(node);
        released(engine).await;
        shutdowns.push(t.elapsed());
        let (files, dirs) = store.take_syncs();
        shutdown_syncs = (shutdown_syncs.0 + files, shutdown_syncs.1 + dirs);
    }
    slots.observe(store.copy_slots());
    report("restart: open to leader", opens, Some(open_syncs));
    report("restart: shutdown", shutdowns, Some(shutdown_syncs));
    slots.report("restart");
}

struct Member {
    id: u64,
    port: u16,
    store: Store,
    /// `None` only while the member is reopened, so its store is free.
    engine: Option<Arc<StorageEngine>>,
    node: Option<RaftNode>,
}

impl Member {
    fn node(&self) -> &RaftNode {
        self.node.as_ref().expect("running")
    }

    fn applied(&self) -> u64 {
        self.node()
            .raft()
            .metrics()
            .borrow_watched()
            .last_applied
            .map_or(0, |id| id.index)
    }
}

/// Open member `id` of a formed group on `engine`, once: its previous life
/// has released the store, and with it its listener; any failure, a refusal
/// to prove its promise included, ends the run with its cause.
async fn reopen(id: u64, port: u16, engine: &Arc<StorageEngine>, pki: &TestPki) -> RaftNode {
    RaftNode::open_cluster(
        id,
        Arc::clone(engine),
        format!("127.0.0.1:{port}").parse().expect("addr"),
        format!("http://127.0.0.1:{port}"),
        pki.connections(id),
    )
    .await
    .expect("the stopped member reopens")
}

/// The member the metrics of the running members name leader of a term
/// they are in, as the leader itself reports it, and the term.
fn leader_in_metrics(members: &[Member]) -> Option<(usize, u64)> {
    members.iter().enumerate().find_map(|(at, member)| {
        let node = member.node.as_ref()?;
        let metrics = node.raft().metrics().borrow_watched().clone();
        (metrics.state.is_leader() && metrics.current_leader == Some(member.id))
            .then_some((at, metrics.current_term))
    })
}

async fn failovers() {
    let pki = TestPki::new();
    let mut members: Vec<Member> = Vec::new();
    for id in 1..=3u64 {
        let store = Store::new();
        let engine = store.open();
        let port = alloc_port();
        let listen = format!("127.0.0.1:{port}").parse().expect("addr");
        let node = match members.first() {
            None => RaftNode::open_cluster(
                id,
                Arc::clone(&engine),
                listen,
                format!("http://127.0.0.1:{port}"),
                pki.connections(id),
            )
            .await
            .expect("founder"),
            Some(founder) => RaftNode::open_joining(
                id,
                Arc::clone(&engine),
                listen,
                pki.connections(id),
                Enrollment {
                    deployment: founder.node().deployment().expect("deployment"),
                    authorities: vec![pki.target(1, format!("127.0.0.1:{}", founder.port))],
                },
            )
            .await
            .expect("joining"),
        };
        members.push(Member {
            id,
            port,
            store,
            engine: Some(engine),
            node: Some(node),
        });
    }
    let founder = members[0].node();
    await_leadership(founder).await;
    for member in &members[1..] {
        founder
            .add_node(
                member.id,
                format!("http://127.0.0.1:{}", member.port),
                pki.identity(member.id),
            )
            .await
            .expect("add");
    }
    founder
        .change_membership(vec![1, 2, 3])
        .await
        .expect("membership");

    let mut in_metrics = Vec::new();
    let mut confirmed = Vec::new();
    let mut terms = Vec::new();
    let mut opens = Vec::new();
    let mut caught_up = Vec::new();
    let (mut election_syncs, mut reopen_syncs) = ((0, 0), (0, 0));
    let mut slots = Slots::default();
    for _ in 0..FAILOVERS {
        let (leader, _) = leader_in_metrics(&members).expect("a leader");
        assert!(
            members[leader].node().is_leader().await,
            "a confirmed leader"
        );
        // Counted from here: the stop and the election.
        for member in &members {
            member.store.take_syncs();
        }
        let stopped = members[leader].node.take().expect("running");
        let t = Instant::now();
        stopped.shutdown().await.expect("stop the leader");
        drop(stopped);
        // The new leader as its own metrics show it, polled without a
        // probe of the group, then confirmed once by a read of its lease.
        let (elected, term) = loop {
            if let Some(found) = leader_in_metrics(&members) {
                break found;
            }
            assert!(t.elapsed() < Duration::from_secs(30), "no new leader");
            tokio::time::sleep(Duration::from_millis(1)).await;
        };
        in_metrics.push(t.elapsed());
        terms.push(term);
        assert!(
            members[elected].node().is_leader().await,
            "the leader its metrics name confirms its lease"
        );
        confirmed.push(t.elapsed());
        for member in &members {
            let (files, dirs) = member.store.take_syncs();
            election_syncs = (election_syncs.0 + files, election_syncs.1 + dirs);
        }

        let engine = members[leader].engine.take().expect("an engine");
        released(engine).await;
        // Counted from here: the reopen until caught up, nothing of the stop.
        members[leader].store.take_syncs();
        let t = Instant::now();
        let engine = members[leader].store.open();
        let node = reopen(members[leader].id, members[leader].port, &engine, &pki).await;
        opens.push(t.elapsed());
        members[leader].engine = Some(engine);
        members[leader].node = Some(node);
        // Caught up: it has applied what the new leader had applied when the
        // member reopened.
        let target = members[elected].applied();
        while members[leader].applied() < target {
            assert!(
                t.elapsed() < Duration::from_secs(30),
                "the reopened member did not catch up"
            );
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        caught_up.push(t.elapsed());
        let (files, dirs) = members[leader].store.take_syncs();
        reopen_syncs = (reopen_syncs.0 + files, reopen_syncs.1 + dirs);
        slots.observe(members[leader].store.copy_slots());
    }
    report("failover: stop to leader in metrics", in_metrics, None);
    report(
        "failover: stop to confirmed leader",
        confirmed,
        Some(election_syncs),
    );
    println!("{:<30} terms: {terms:?}", "");
    report("failover: reopen, open only", opens, None);
    report(
        "failover: reopen to caught up",
        caught_up,
        Some(reopen_syncs),
    );
    slots.report("failover");
    for member in &mut members {
        if let Some(node) = member.node.take() {
            node.shutdown().await.expect("shutdown");
        }
    }
}

fn main() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("runtime");
    runtime.block_on(async {
        println!("== consensus promise costs ==");
        if runs("vote") {
            vote_changes().await;
        }
        if runs("vote-same") {
            same_votes().await;
        }
        if runs("commit") {
            commits().await;
        }
        if runs("restart") {
            restarts().await;
        }
        if runs("failover") {
            failovers().await;
        }
    });
}
