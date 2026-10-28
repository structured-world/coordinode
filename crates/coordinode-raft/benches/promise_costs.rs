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
//! With `PROMISE_COSTS_TRACE` set, the commit section also prints each
//! commit's wall-clock start and duration, and runs `iostat -t -x 1` beside
//! itself when the host has it.
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
use coordinode_storage::engine::partition::Partition;
use coordinode_test_fixtures::{TestPki, alloc_port};
use lsm_tree::fs::{Fault, FaultFs, FaultInjector, FaultOp, FaultRule, StdFs, SyncKind};
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

/// Whether `section` runs: all of them unless `PROMISE_COSTS_SECTION` names
/// it, alone or in a comma-separated list.
fn runs(section: &str) -> bool {
    std::env::var("PROMISE_COSTS_SECTION").map_or(true, |only| {
        only.split(',').any(|name| name.trim() == section)
    })
}

/// A repetition count from `var`, or `default`.
fn count(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .map(|n| n.parse().expect("a repetition count"))
        .unwrap_or(default)
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

/// Whether `PROMISE_COSTS_TRACE` asks for every commit's wall-clock start and
/// duration, and for `iostat -t -x 1` beside the section, so a slow commit can
/// be placed against the device's queue and wait at that second.
fn traced() -> bool {
    std::env::var_os("PROMISE_COSTS_TRACE").is_some()
}

/// `iostat` printing to this process's stdout until the guard drops.
struct Iostat(Option<std::process::Child>);

impl Iostat {
    fn start() -> Self {
        match std::process::Command::new("iostat")
            .args(["-t", "-x", "1"])
            .spawn()
        {
            Ok(child) => Self(Some(child)),
            Err(error) => {
                println!("iostat not started: {error}");
                Self(None)
            }
        }
    }
}

impl Drop for Iostat {
    fn drop(&mut self) {
        if let Some(child) = self.0.as_mut() {
            // The sampler only prints; a failed kill or wait leaves nothing to
            // undo, and the trace says where its output ends.
            if let Err(error) = child.kill().and_then(|()| child.wait().map(drop)) {
                println!("iostat not stopped: {error}");
            }
        }
    }
}

fn unix_micros() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("the clock is past the epoch")
        .as_micros()
}

async fn commits() {
    let store = Store::new();
    let engine = store.open();
    let node = RaftNode::open(1, Arc::clone(&engine)).await.expect("open");
    await_leadership(&node).await;
    let pipeline = node.pipeline();
    let id_gen = ProposalIdGenerator::with_base(1 << 40);
    let trace = traced();
    let iostat = trace.then(Iostat::start);
    store.take_syncs();
    let mut samples = Vec::new();
    let mut starts = Vec::new();
    for n in 0..ops(1_000) {
        let proposal = put(&id_gen, n);
        let wall = unix_micros();
        let t = Instant::now();
        tokio::task::block_in_place(|| pipeline.propose_and_wait(&proposal)).expect("commit");
        samples.push(t.elapsed());
        starts.push(wall);
    }
    let syncs = store.take_syncs();
    drop(iostat);
    if trace {
        println!("commit trace: n,start_unix_us,duration_us");
        for (n, (start, took)) in starts.iter().zip(&samples).enumerate() {
            println!("commit trace: {n},{start},{}", took.as_micros());
        }
    }
    report("commit (one member)", samples, Some(syncs));
    drop(pipeline);
    node.shutdown().await.expect("shutdown");
    drop(node);
    released(engine).await;
}

/// Keys of the committed data a recovery image holds and a read reads.
fn data_key(n: u64) -> Vec<u8> {
    format!("node:1:{n}").into_bytes()
}

/// The value [`put`] writes.
const DATA_VALUE: [u8; 64] = [7; 64];

/// Leadership within `limit`, polled.
async fn leads_within(node: &RaftNode, limit: Duration) -> bool {
    let t = Instant::now();
    while t.elapsed() < limit {
        if node.is_leader().await {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    false
}

fn term_of(node: &RaftNode) -> u64 {
    node.raft().metrics().borrow_watched().current_term
}

/// Copy the closed store at `from` to `to`, files and directories alike.
fn copy_tree(from: &std::path::Path, to: &std::path::Path) {
    std::fs::create_dir_all(to).expect("create the image directory");
    for entry in std::fs::read_dir(from).expect("list the image") {
        let entry = entry.expect("an image entry");
        let target = to.join(entry.file_name());
        if entry.file_type().expect("an entry type").is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), &target).expect("copy an image file");
        }
    }
}

/// The second copy's slot files in `dir`, whatever their state.
fn copy_slot_paths(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    std::fs::read_dir(dir)
        .expect("list the store")
        .map(|entry| entry.expect("a store entry"))
        .filter(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .starts_with("consensus-vote")
        })
        .map(|entry| entry.path())
        .collect()
}

/// What a store loses before the measured reopen.
#[derive(Clone, Copy)]
enum Loss {
    /// The Raft tree and the member-local journal; the second copy, where
    /// the build has one, stays.
    RaftTree,
    /// Every slot of the second copy damaged in its middle byte; the Raft
    /// tree stays.
    SecondCopy,
    /// The Raft tree, the member-local journal and every slot of the copy.
    Everything,
}

impl Loss {
    fn name(self) -> &'static str {
        match self {
            Loss::RaftTree => "Raft tree lost",
            Loss::SecondCopy => "second copy damaged",
            Loss::Everything => "Raft tree and every copy lost",
        }
    }

    /// Inflict the loss on the closed store at `dir`. `false`: the loss does
    /// not apply to this build (it keeps no second copy to damage).
    fn inflict(self, dir: &std::path::Path) -> bool {
        let slots = copy_slot_paths(dir);
        let lose_tree = || {
            std::fs::remove_dir_all(dir.join(Partition::Raft.name())).expect("lose the Raft tree");
            let journal = dir.join("local-journal");
            if journal.exists() {
                std::fs::remove_dir_all(journal).expect("lose the member-local journal");
            }
        };
        match self {
            Loss::RaftTree => lose_tree(),
            Loss::SecondCopy => {
                if slots.is_empty() {
                    return false;
                }
                for slot in &slots {
                    let mut bytes = std::fs::read(slot).expect("read a slot");
                    let middle = bytes.len() / 2;
                    bytes[middle] ^= 0xFF;
                    std::fs::write(slot, &bytes).expect("damage a slot");
                }
            }
            Loss::Everything => {
                lose_tree();
                for slot in &slots {
                    std::fs::remove_file(slot).expect("lose a slot");
                }
            }
        }
        true
    }
}

/// Recovery after each [`Loss`], timed from the engine open to leadership or
/// to the refusal, on a copy of one closed image: the member committed
/// `RECOVERY_KEYS` entries and was restarted until its term passed 1, so a
/// group started afresh under its identity shows as a lower term. A reopen
/// that leads without the term it had promised, or without its data, is an
/// incorrect recovery, never a fast one.
async fn recoveries() {
    const RECOVERY_KEYS: u64 = 50;
    let image = Store::new();
    let promised = {
        let id_gen = ProposalIdGenerator::with_base(1 << 41);
        let mut term = 0;
        for round in 0..3 {
            let engine = image.open();
            let node = RaftNode::open(1, Arc::clone(&engine))
                .await
                .expect("open the image");
            await_leadership(&node).await;
            if round == 0 {
                let pipeline = node.pipeline();
                for n in 0..RECOVERY_KEYS {
                    tokio::task::block_in_place(|| pipeline.propose_and_wait(&put(&id_gen, n)))
                        .expect("commit");
                }
            }
            term = term_of(&node);
            node.shutdown().await.expect("shutdown");
            drop(node);
            released(engine).await;
        }
        term
    };
    println!(
        "recovery image: term {promised}, {RECOVERY_KEYS} committed keys, copy slots {}",
        copy_slot_paths(image.dir.path()).len()
    );
    let repeats = count("PROMISE_COSTS_RECOVERIES", 100);
    for loss in [Loss::RaftTree, Loss::SecondCopy, Loss::Everything] {
        let mut outcomes: std::collections::BTreeMap<String, Vec<Duration>> =
            std::collections::BTreeMap::new();
        let mut applies = true;
        for _ in 0..repeats {
            let dir = tempfile::tempdir().expect("dir");
            copy_tree(image.dir.path(), dir.path());
            if !loss.inflict(dir.path()) {
                applies = false;
                break;
            }
            let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            )])
            .with_fs(Arc::new(FaultFs::with_injector(
                StdFs,
                Arc::new(FaultInjector::new()),
            )));
            let t = Instant::now();
            let (outcome, took) = match StorageEngine::open(&config) {
                Err(e) => (format!("refused at engine open: {e}"), t.elapsed()),
                Ok(engine) => {
                    let engine = Arc::new(engine);
                    match RaftNode::open(1, Arc::clone(&engine)).await {
                        Err(e) => {
                            let took = t.elapsed();
                            drop(engine);
                            (format!("refused: {e}"), took)
                        }
                        Ok(node) => {
                            let led = leads_within(&node, Duration::from_secs(10)).await;
                            let took = t.elapsed();
                            let term = term_of(&node);
                            let data = (0..RECOVERY_KEYS).all(|n| {
                                engine
                                    .get(Partition::Node, &data_key(n))
                                    .expect("read the data")
                                    .as_deref()
                                    == Some(DATA_VALUE.as_slice())
                            });
                            let outcome = if !led {
                                "no leadership within 10 s".to_string()
                            } else if term <= promised || !data {
                                format!(
                                    "incorrect recovery (term {term} against promised {promised}, data kept {data})"
                                )
                            } else {
                                "recovered".to_string()
                            };
                            node.shutdown().await.expect("shutdown");
                            drop(node);
                            released(engine).await;
                            (outcome, took)
                        }
                    }
                }
            };
            outcomes.entry(outcome).or_default().push(took);
        }
        if !applies {
            println!(
                "recovery: {}: not applicable, this build keeps no second copy",
                loss.name()
            );
            continue;
        }
        for (outcome, samples) in outcomes {
            report(
                &format!("recovery: {}: {outcome}", loss.name()),
                samples,
                None,
            );
        }
    }
}

/// The second copy's files in `dir`: count, logical bytes and allocated bytes
/// (512-byte blocks the filesystem reports). Allocated bytes leave out what
/// the filesystem spends on metadata and copy-on-write outside the files.
fn copy_footprint(dir: &std::path::Path) -> (usize, u64, u64) {
    let mut footprint = (0, 0, 0);
    for path in copy_slot_paths(dir) {
        let meta = std::fs::metadata(&path).expect("a slot's metadata");
        footprint.0 += 1;
        footprint.1 += meta.len();
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt as _;
            footprint.2 += meta.blocks() * 512;
        }
    }
    footprint
}

/// The most of each footprint field seen, and how often the file count fell
/// between two observations (a removal the observations straddled).
#[derive(Default)]
struct Peak {
    most: (usize, u64, u64),
    last_files: usize,
    falls: usize,
}

impl Peak {
    fn observe(&mut self, dir: &std::path::Path) {
        let now = copy_footprint(dir);
        self.most = (
            self.most.0.max(now.0),
            self.most.1.max(now.1),
            self.most.2.max(now.2),
        );
        if now.0 < self.last_files {
            self.falls += 1;
        }
        self.last_files = now.0;
    }

    fn report(&self, label: &str) {
        println!(
            "{label:<30} copy peak: files={} logical bytes={} allocated bytes={}  file-count falls between observations={}",
            self.most.0, self.most.1, self.most.2, self.falls
        );
    }
}

/// The second copy's peak footprint, observed after every write and before
/// a removal it would make: a normal cycle of vote changes and restarts, then
/// a chain of opens whose copy write fails at its sync (each leaves a slot it
/// does not know of) and a write whose removal of the spent slots fails, so
/// the state just before a removal stays to be observed.
async fn footprints() {
    let store = Store::new();
    let dir = store.dir.path().to_path_buf();
    let mut normal = Peak::default();
    {
        let mut log = LogStore::open(store.open()).expect("log");
        for term in 1..=ops(300) {
            log.save_vote(&Vote::new(term, 1)).await.expect("vote");
            normal.observe(&dir);
        }
    }
    for round in 0..20u64 {
        let mut log = LogStore::open(store.open()).expect("reopen");
        normal.observe(&dir);
        log.save_vote(&Vote::new(1_000 + round, 1))
            .await
            .expect("vote");
        normal.observe(&dir);
    }
    normal.report("copy footprint: normal cycle");

    let mut chain = Peak::default();
    let mut failed = 0;
    chain.observe(&dir);
    for attempt in 0..3u64 {
        let mut log = LogStore::open(store.open()).expect("reopen");
        store.syncs.arm(
            FaultRule::new(
                FaultOp::SyncAll,
                Fault::Error(lsm_tree::io::ErrorKind::Other),
            )
            .on_path("consensus-vote")
            .once(),
        );
        if log.save_vote(&Vote::new(2_000 + attempt, 1)).await.is_err() {
            failed += 1;
        }
        store.syncs.clear();
        chain.observe(&dir);
    }
    {
        let mut log = LogStore::open(store.open()).expect("reopen after the failed writes");
        chain.observe(&dir);
        store.syncs.arm(
            FaultRule::new(
                FaultOp::RemoveFile,
                Fault::Error(lsm_tree::io::ErrorKind::Other),
            )
            .on_path("consensus-vote"),
        );
        log.save_vote(&Vote::new(3_000, 1))
            .await
            .expect("vote, removal refused");
        chain.observe(&dir);
        store.syncs.clear();
        log.save_vote(&Vote::new(3_001, 1))
            .await
            .expect("vote, removal allowed");
        chain.observe(&dir);
    }
    chain.report("copy footprint: failed chain");
    println!(
        "{:<30} copy writes failed at sync: {failed} of 3; files after: {}",
        "",
        copy_footprint(&dir).0
    );
}

/// Point reads through the engine of a single member that committed
/// `READ_KEYS` entries: the latest committed value, read locally (not a
/// linearizable read through the leader's lease), one at a time. Every key is
/// read once before the measured phase and every value is checked.
async fn reads() {
    const READ_KEYS: u64 = 1_000;
    const PASSES: u64 = 5;
    let store = Store::new();
    let engine = store.open();
    let node = RaftNode::open(1, Arc::clone(&engine)).await.expect("open");
    await_leadership(&node).await;
    let pipeline = node.pipeline();
    let id_gen = ProposalIdGenerator::with_base(1 << 42);
    for n in 0..READ_KEYS {
        tokio::task::block_in_place(|| pipeline.propose_and_wait(&put(&id_gen, n)))
            .expect("commit");
    }
    let read = |n: u64| {
        let value = engine.get(Partition::Node, &data_key(n)).expect("read");
        assert_eq!(value.as_deref(), Some(DATA_VALUE.as_slice()), "key {n}");
    };
    for n in 0..READ_KEYS {
        read(n);
    }
    let mut samples = Vec::with_capacity((READ_KEYS * PASSES) as usize);
    let phase = Instant::now();
    for _ in 0..PASSES {
        for n in 0..READ_KEYS {
            let t = Instant::now();
            read(n);
            samples.push(t.elapsed());
        }
    }
    let wall = phase.elapsed();
    println!(
        "read (engine point read, latest, local, concurrency 1): {} reads in {wall:?}, {:.0} reads/s",
        samples.len(),
        samples.len() as f64 / wall.as_secs_f64()
    );
    report("read (engine point read)", samples, None);
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
    for _ in 0..count("PROMISE_COSTS_FAILOVERS", FAILOVERS) {
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
    if traced() {
        println!("failover trace: n,term,stop_to_leader_in_metrics_us,stop_to_confirmed_leader_us");
        for (n, ((metrics, confirmed), term)) in
            in_metrics.iter().zip(&confirmed).zip(&terms).enumerate()
        {
            println!(
                "failover trace: {n},{term},{},{}",
                metrics.as_micros(),
                confirmed.as_micros()
            );
        }
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
        if runs("recovery") {
            recoveries().await;
        }
        if runs("footprint") {
            footprints().await;
        }
        if runs("read") {
            reads().await;
        }
    });
}
