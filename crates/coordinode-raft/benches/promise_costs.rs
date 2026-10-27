//! What keeping a member's consensus promises costs.
//!
//! Each figure is a latency distribution and the syncs the operation issued,
//! counted through the storage filesystem:
//! - a vote change through the log store;
//! - a commit through a single-member group, the steady write path;
//! - a restart of a single-member group, open to leadership;
//! - a failover of a three-member group, from the leader's stop to a new
//!   leader, and the stopped member's reopen.
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
use openraft::storage::RaftLogStorage;

const VOTES: u64 = 300;
const COMMITS: u64 = 1_000;
const RESTARTS: usize = 15;
const FAILOVERS: usize = 4;

/// A store whose filesystem counts the syncs it is asked for.
struct Store {
    syncs: Arc<FaultInjector>,
    config: StorageConfig,
    _dir: tempfile::TempDir,
}

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
        Self {
            syncs,
            config,
            _dir: dir,
        }
    }

    fn open(&self) -> Arc<StorageEngine> {
        Arc::new(StorageEngine::open(&self.config).expect("open"))
    }

    /// Syncs since the last call: files, directories.
    fn take_syncs(&self) -> (usize, usize) {
        let syncs = self.syncs.take_syncs();
        let directories = syncs
            .iter()
            .filter(|(_, kind)| *kind == SyncKind::Directory)
            .count();
        (syncs.len() - directories, directories)
    }
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    let rank = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[rank]
}

fn report(label: &str, mut samples: Vec<Duration>, syncs: (usize, usize)) {
    samples.sort_unstable();
    let n = samples.len();
    println!(
        "{label:<22} n={n:>5}  p50={:>10.3?}  p99={:>10.3?}  max={:>10.3?}  file syncs/op={:>6.2}  dir syncs/op={:>5.2}",
        percentile(&samples, 0.50),
        percentile(&samples, 0.99),
        samples[n - 1],
        syncs.0 as f64 / n as f64,
        syncs.1 as f64 / n as f64,
    );
}

async fn vote_changes() {
    let store = Store::new();
    let mut log = LogStore::open(store.open()).expect("log");
    store.take_syncs();
    let mut samples = Vec::new();
    for term in 1..=VOTES {
        let t = Instant::now();
        log.save_vote(&Vote::new(term, 1)).await.expect("vote");
        samples.push(t.elapsed());
    }
    report("vote change", samples, store.take_syncs());
}

async fn await_leadership(node: &RaftNode) {
    for _ in 0..600 {
        if node.is_leader().await {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert!(node.is_leader().await, "no leadership");
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

async fn commits_and_restarts() {
    let store = Store::new();
    {
        let node = RaftNode::open(1, store.open()).await.expect("open");
        await_leadership(&node).await;
        let pipeline = node.pipeline();
        let id_gen = ProposalIdGenerator::with_base(1 << 40);
        store.take_syncs();
        let mut samples = Vec::new();
        for n in 0..COMMITS {
            let proposal = put(&id_gen, n);
            let t = Instant::now();
            tokio::task::block_in_place(|| pipeline.propose_and_wait(&proposal)).expect("commit");
            samples.push(t.elapsed());
        }
        report("commit (one member)", samples, store.take_syncs());
        node.shutdown().await.expect("shutdown");
    }
    let mut opens = Vec::new();
    let mut leads = Vec::new();
    store.take_syncs();
    for _ in 0..RESTARTS {
        let t = Instant::now();
        let engine = store.open();
        let node = RaftNode::open(1, Arc::clone(&engine))
            .await
            .expect("reopen");
        opens.push(t.elapsed());
        await_leadership(&node).await;
        leads.push(t.elapsed());
        node.shutdown().await.expect("shutdown");
        drop(node);
        for _ in 0..200 {
            if Arc::strong_count(&engine) == 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }
    // Syncs of the whole restart, shutdown included.
    let syncs = store.take_syncs();
    report("restart: open", opens, syncs);
    report("restart: to leader", leads, syncs);
}

struct Member {
    id: u64,
    port: u16,
    store: Store,
    /// `None` only while the member is reopened, so its store is free.
    engine: Option<Arc<StorageEngine>>,
    node: Option<RaftNode>,
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
                    deployment: founder
                        .node
                        .as_ref()
                        .expect("running")
                        .deployment()
                        .expect("deployment"),
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
    let founder = members[0].node.as_ref().expect("running");
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

    let mut elections = Vec::new();
    let mut reopens = Vec::new();
    for _ in 0..FAILOVERS {
        let mut leader = None;
        for (at, member) in members.iter().enumerate() {
            if member.node.as_ref().expect("running").is_leader().await {
                leader = Some(at);
            }
        }
        let leader = leader.expect("a leader");
        let stopped = members[leader].node.take().expect("running");
        let t = Instant::now();
        stopped.shutdown().await.expect("stop the leader");
        drop(stopped);
        'elected: loop {
            for member in &members {
                if let Some(node) = &member.node {
                    if node.is_leader().await {
                        break 'elected;
                    }
                }
            }
            assert!(t.elapsed() < Duration::from_secs(30), "no new leader");
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        elections.push(t.elapsed());
        let member = &mut members[leader];
        // The stopped node's tasks let go of the engine on their own time;
        // the store opens again only once they have.
        for _ in 0..2_000 {
            if member
                .engine
                .as_ref()
                .is_some_and(|e| Arc::strong_count(e) == 1)
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let stopped_engine = member.engine.take().expect("an engine");
        assert_eq!(
            Arc::strong_count(&stopped_engine),
            1,
            "the stopped member still holds its engine"
        );
        drop(stopped_engine);
        let t = Instant::now();
        let engine = member.store.open();
        member.engine = Some(Arc::clone(&engine));
        let mut reopened = None;
        for _ in 0..40 {
            if let Ok(node) = RaftNode::open_cluster(
                member.id,
                Arc::clone(&engine),
                format!("127.0.0.1:{}", member.port).parse().expect("addr"),
                format!("http://127.0.0.1:{}", member.port),
                pki.connections(member.id),
            )
            .await
            {
                reopened = Some(node);
                break;
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
        reopens.push(t.elapsed());
        member.node = Some(reopened.expect("reopen the stopped member"));
        // Back in the group before the next round stops a leader.
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
    let syncs = members.iter().fold((0, 0), |sum, member| {
        let (files, dirs) = member.store.take_syncs();
        (sum.0 + files, sum.1 + dirs)
    });
    report("failover: new leader", elections, (0, 0));
    report("failover: member reopen", reopens, syncs);
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
        vote_changes().await;
        commits_and_restarts().await;
        failovers().await;
    });
}
