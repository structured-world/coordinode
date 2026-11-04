//! A group moves to a new engine format by majority, on real server
//! processes: members are updated one at a time under a continuous read and
//! write workload, no acknowledged write is lost, reads are served
//! throughout, and the group pauses its writes exactly once.
//!
//! The three members run on one host, so a host that stalls stalls the
//! whole group: its writes pause beyond the move's own pause, and its
//! leader may lose its authority. Such an attempt must still be correct (no
//! acknowledged write lost, every member caught up, one leader at the end).
//! It is retried only when the host's own pressure counters show the stall
//! during every extra pause; an extra pause the host does not account for is
//! a failure at once. The test fails as well when no attempt of three shows
//! the single pause.
//!
//! The server binary must be built with its `test-format-bump` feature (a
//! workspace build with `--all-features`): `COORDINODE_TEST_ENGINE_FORMAT_BUMP`
//! then raises the engine format version a process runs.
//!
//! ## Running
//!
//! ```bash
//! cargo build -p coordinode-server --features test-format-bump
//! cargo nextest run -P cluster -p coordinode-integration --features test-format-bump --test version_moves
//! ```

#![cfg(feature = "test-format-bump")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use coordinode_integration::harness::{
    CoordinodeProcess, free_port, start_cluster_member_expecting_refusal_with_env,
};
use coordinode_integration::proto::admin::{
    GetClusterStatusRequest, JoinNodeRequest, NodeRole,
    cluster_service_client::ClusterServiceClient,
};
use coordinode_integration::proto::common::{PropertyValue, property_value::Value as Pv};
use coordinode_integration::proto::query::{
    ExecuteCypherRequest, cypher_service_client::CypherServiceClient,
};
use coordinode_integration::proto::replication::{
    Journal, ReadConcern, WriteConcern, WriteConcernMode, write_concern::W,
};

const BUMP: &str = "COORDINODE_TEST_ENGINE_FORMAT_BUMP";

fn at(bump: &str) -> [(&str, &str); 1] {
    [(BUMP, bump)]
}

/// The engine format a member's report says it runs.
fn engine_of(member: &CoordinodeProcess) -> u64 {
    member.version_report()["pair"]["engine"]
        .as_u64()
        .expect("the report names the engine format")
}

async fn channel(port: u16, timeout: Duration) -> Option<tonic::transport::Channel> {
    tonic::transport::Endpoint::from_shared(format!("http://[::1]:{port}"))
        .expect("endpoint")
        .connect_timeout(Duration::from_millis(500))
        .timeout(timeout)
        .connect()
        .await
        .ok()
}

async fn cypher_client(port: u16) -> Option<CypherServiceClient<tonic::transport::Channel>> {
    Some(CypherServiceClient::new(
        channel(port, Duration::from_secs(5)).await?,
    ))
}

/// The leader and term the member at `port` reports, if it answers.
async fn leadership_of(port: u16) -> Option<(String, u64)> {
    let mut client = ClusterServiceClient::new(channel(port, Duration::from_millis(300)).await?);
    let status = client
        .get_cluster_status(GetClusterStatusRequest {})
        .await
        .ok()?
        .into_inner();
    Some((status.leader_id, status.raft_term))
}

/// One write attempt: its sequence number and the member it was sent to.
/// The workload sends each number to every member, so a number may land
/// more than once; each attempt is its own node, told apart by `via`.
type Attempt = (i64, u16);

/// Create node `seq` of `:Move` on the member at `port`, marked as sent
/// there, acknowledged at majority with the journal; why not when it is
/// refused.
async fn write_seq(port: u16, seq: i64) -> Result<(), String> {
    let mut client = cypher_client(port)
        .await
        .ok_or_else(|| "no connection".to_string())?;
    let mut parameters = HashMap::new();
    parameters.insert(
        "seq".to_string(),
        PropertyValue {
            value: Some(Pv::IntValue(seq)),
        },
    );
    parameters.insert(
        "via".to_string(),
        PropertyValue {
            value: Some(Pv::IntValue(i64::from(port))),
        },
    );
    client
        .execute_cypher(ExecuteCypherRequest {
            query: "CREATE (:Move {seq: $seq, via: $via})".to_string(),
            parameters,
            read_preference: 0,
            read_concern: None,
            write_concern: Some(WriteConcern {
                w: Some(W::Mode(WriteConcernMode::Majority as i32)),
                journal: Journal::Journal as i32,
                timeout_ms: 0,
            }),
            ..Default::default()
        })
        .await
        .map(drop)
        .map_err(|status| format!("{:?}: {}", status.code(), status.message()))
}

/// The `:Move` sequence numbers the member at `port` holds, read locally
/// from itself; why not when it does not answer.
async fn local_seqs(port: u16) -> Result<BTreeSet<i64>, String> {
    Ok(local_attempts(port)
        .await?
        .into_iter()
        .map(|(seq, _)| seq)
        .collect())
}

/// Every `:Move` attempt the member at `port` holds, read locally from
/// itself; why not when it does not answer.
async fn local_attempts(port: u16) -> Result<BTreeSet<Attempt>, String> {
    let mut client = cypher_client(port)
        .await
        .ok_or_else(|| "no connection".to_string())?;
    let rows = client
        .execute_cypher(ExecuteCypherRequest {
            query: "MATCH (n:Move) RETURN n.seq AS seq, n.via AS via".to_string(),
            parameters: HashMap::new(),
            read_preference: 5, // NEAREST: answered where the request lands
            read_concern: Some(ReadConcern {
                level: 1, // LOCAL
                ..Default::default()
            }),
            write_concern: None,
            ..Default::default()
        })
        .await
        .map_err(|status| format!("{:?}: {}", status.code(), status.message()))?
        .into_inner()
        .rows;
    Ok(rows
        .iter()
        .map(|row| match (&row.values[0].value, &row.values[1].value) {
            (Some(Pv::IntValue(seq)), Some(Pv::IntValue(via))) => {
                (*seq, u16::try_from(*via).expect("a via is a port"))
            }
            other => panic!("unexpected attempt {other:?}"),
        })
        .collect())
}

/// The acknowledged attempts `held` lacks. Every attempt counts on its own:
/// another attempt at the same number, acknowledged or not, does not stand
/// in for it.
fn lost_attempts(acked: &BTreeSet<Attempt>, held: &BTreeSet<Attempt>) -> Vec<Attempt> {
    acked.difference(held).copied().collect()
}

async fn wait_for_voters(client: &mut ClusterServiceClient<tonic::transport::Channel>, n: usize) {
    let deadline = Instant::now() + Duration::from_secs(40);
    loop {
        if let Ok(resp) = client.get_cluster_status(GetClusterStatusRequest {}).await {
            let s = resp.into_inner();
            let voters = s
                .nodes
                .iter()
                .filter(|m| m.role != NodeRole::Learner as i32)
                .count();
            if s.nodes.len() == n && voters == n {
                return;
            }
        }
        assert!(Instant::now() < deadline, "no group of {n} voters");
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
}

/// Three members at format bump 0, member 1 leading.
async fn group_of_three() -> ([CoordinodeProcess; 3], [u16; 3]) {
    let ports = [free_port(), free_port(), free_port()];
    let [p1, p2, p3] = ports;
    let m1 = CoordinodeProcess::start_cluster_member_with_env(1, p1, &[p2, p3], &at("0")).await;
    let m2 = CoordinodeProcess::start_cluster_member_with_env(2, p2, &[p1, p3], &at("0")).await;
    let m3 = CoordinodeProcess::start_cluster_member_with_env(3, p3, &[p1, p2], &at("0")).await;
    let mut leader = m1.cluster_client().await;
    for (id, member) in [(2u64, &m2), (3u64, &m3)] {
        leader
            .join_node(JoinNodeRequest {
                node_id: id,
                address: member.member_addr(),
                pre_seeded: false,
                node_identity: member.member_identity(),
            })
            .await
            .unwrap_or_else(|e| panic!("join {id}: {e}"));
        wait_for_voters(&mut leader, usize::try_from(id).expect("small")).await;
    }
    ([m1, m2, m3], ports)
}

fn peers_of(ports: [u16; 3], i: usize) -> Vec<u16> {
    ports
        .iter()
        .enumerate()
        .filter(|(j, _)| *j != i)
        .map(|(_, p)| *p)
        .collect()
}

/// What the workload saw: every acknowledged write with when it was
/// acknowledged, and every local read a member refused with when.
#[derive(Default)]
struct Observed {
    /// Every acknowledged attempt, with when it was acknowledged.
    acked: Vec<(Attempt, Instant)>,
    /// Every local read a member refused: the member, when the request
    /// began and when it was refused, and why.
    refused_reads: Vec<(u16, Instant, Instant, String)>,
    /// Write attempts that took long: the member, when, how long, and
    /// how it ended. They tell a member that answered slowly from a group
    /// that did not write.
    slow_writes: Vec<(u16, Instant, Duration, Result<(), String>)>,
    /// Each change of the leadership a member reports.
    leadership: Vec<Seen>,
    /// The members' processes' waits, sampled.
    waits: Vec<Waited>,
}

/// A member's report of who leads, first seen in this form in the answer
/// received at `at`; the previous answered request was sent at `before`, and
/// the state it reported was read on the server no earlier, so the change
/// happened between the two.
#[derive(Debug, Clone)]
struct Seen {
    port: u16,
    before: Instant,
    at: Instant,
    /// The leader's node id, empty while the member knows none.
    leader: String,
    term: u64,
}

/// What a member's process had waited, summed over its threads, at `at`:
/// nanoseconds runnable but not running (scheduler run-queue delay), and
/// nanoseconds blocked on block I/O (kept when the kernel's task delay
/// accounting is on, else zero).
#[derive(Debug, Clone, Copy)]
struct Waited {
    port: u16,
    pid: u32,
    at: Instant,
    queued_ns: u64,
    blocked_ns: u64,
}

/// What process `pid` has waited so far, summed over its threads (see
/// [`Waited`]); `None` where the kernel does not report it.
#[cfg(target_os = "linux")]
fn waited_of(pid: u32) -> Option<(u64, u64)> {
    // The kernel reports block-I/O delay in clock ticks of USER_HZ, which
    // Linux fixes at 100 for user space.
    const NS_PER_TICK: u64 = 10_000_000;
    let mut queued = 0u64;
    let mut blocked = 0u64;
    for task in std::fs::read_dir(format!("/proc/{pid}/task")).ok()? {
        let task = task.ok()?.path();
        // schedstat: time on the CPU, time waiting on a run queue, slices.
        let schedstat = std::fs::read_to_string(task.join("schedstat")).ok()?;
        queued += schedstat.split_whitespace().nth(1)?.parse::<u64>().ok()?;
        // stat field 42, delayacct_blkio_ticks; the fields after the
        // parenthesised name start at field 3.
        let stat = std::fs::read_to_string(task.join("stat")).ok()?;
        let after_name = stat.rsplit_once(')')?.1;
        let ticks: u64 = after_name.split_whitespace().nth(42 - 3)?.parse().ok()?;
        blocked += ticks * NS_PER_TICK;
    }
    Some((queued, blocked))
}

#[cfg(not(target_os = "linux"))]
fn waited_of(_pid: u32) -> Option<(u64, u64)> {
    None
}

/// The share of `from..to` (from `started`) that the process on `port`
/// spent waiting for CPU or for block I/O, summed over its threads, read
/// from the samples of one process around it; `None` without such samples
/// (the member was not running throughout, or the kernel reports no waits).
fn waited_share(
    samples: &[Waited],
    port: u16,
    started: Instant,
    (from, to): (Duration, Duration),
) -> Option<f64> {
    let at = |sample: &Waited| sample.at.saturating_duration_since(started);
    let before = samples
        .iter()
        .rev()
        .find(|s| s.port == port && at(s) <= from)?;
    let after = samples.iter().find(|s| s.port == port && at(s) >= to)?;
    if before.pid != after.pid {
        return None;
    }
    let wall = after.at.duration_since(before.at).as_nanos() as f64;
    let waited = (after.queued_ns.checked_sub(before.queued_ns)?
        + after.blocked_ns.checked_sub(before.blocked_ns)?) as f64;
    Some(waited / wall)
}

/// A pause beyond the move's own counts as the host's when the process of
/// the member that led into it spent at least this share of it waiting for
/// CPU or block I/O, summed over its threads. A stopped leader waits on
/// neither, however loaded the host is meanwhile, and a leader at rest
/// waits far less.
///
/// The waits are the whole process's, not its commit path's: a commit held
/// inside the group (a lock) while a loaded host makes the leader's other
/// threads wait is taken for the host's, and that attempt is retried. Such
/// a defect fails the test only when it occurs, masked this way, in every
/// attempt: if the three attempts were independent and each showed it with
/// the same probability `p`, the test would catch it with probability `p`
/// cubed. The attempts are not shown to be independent, so that figure
/// holds only under that assumption; it is not a measured sensitivity.
/// Nothing read from outside
/// the process tells the two apart when the load coincides, and the test
/// accepts this to stay green on a loaded host.
const HOST_STALLED: f64 = 0.25;

/// A leadership as members report it: the leader's node id and the term.
type Identity = (String, u64);

/// Who led as of `at` (from `started`): the last leadership any member
/// reported by then, when it names one.
fn leader_as_of(seen: &[Seen], started: Instant, at: Duration) -> Option<Identity> {
    seen.iter()
        .filter(|s| s.at.saturating_duration_since(started) <= at && !s.leader.is_empty())
        .max_by_key(|s| s.at)
        .map(|s| (s.leader.clone(), s.term))
}

/// When the leadership changed hands after `since` (from `started`): the
/// span in which the member that leads afterwards first reported itself
/// leading under an identity (node and term) other than the one leading as
/// of `since`. Under openraft's advanced leader ids a new identity can keep
/// the term and change the node, so both are compared. `port_of` maps a
/// node id to its member's port.
fn hand_over(
    seen: &[Seen],
    started: Instant,
    since: Duration,
    port_of: impl Fn(&str) -> Option<u16>,
) -> Option<(Duration, Duration)> {
    let before = leader_as_of(seen, started, since);
    seen.iter()
        .filter(|s| s.at.saturating_duration_since(started) > since)
        .filter(|s| !s.leader.is_empty() && Some(s.port) == port_of(&s.leader))
        .filter(|s| before.as_ref() != Some(&(s.leader.clone(), s.term)))
        .min_by_key(|s| s.at)
        .map(|s| {
            (
                s.before.saturating_duration_since(started),
                s.at.saturating_duration_since(started),
            )
        })
}

/// Whether a refused local read is one the member was allowed to refuse:
/// the request was under way while the member was down, beginning before it
/// served again and ending after its update began. A request begun once the
/// member serves again is not.
fn refusal_allowed(
    windows: &[(u16, Instant, Instant)],
    port: u16,
    began: Instant,
    refused: Instant,
) -> bool {
    windows
        .iter()
        .any(|(p, from, to)| *p == port && began <= *to && refused >= *from)
}

/// How an attempt of the move went.
#[derive(Debug)]
enum Outcome {
    /// Correct, and the writes paused once, on the completing update.
    Single,
    /// Correct, with more pauses, each while the host was stalled.
    HostStalled(String),
    /// Correct, with a pause the host does not account for.
    Unexplained(String),
}

/// A stall the test causes during the quiet stretch after the first update,
/// to check how attempts are classified. The stalls are caused only where
/// the kernel reports the waits to classify them by.
#[derive(Clone, Copy)]
enum Stall {
    None,
    /// The leader's process is not scheduled for this long, while busy
    /// threads contend every core of the host at the same time: a stall
    /// inside the group that coincides with a loaded host.
    #[cfg(target_os = "linux")]
    LeaderOnLoadedHost(Duration),
    /// Busy threads contend every core of the host for this long; nothing
    /// of the group is stopped.
    #[cfg(target_os = "linux")]
    LoadedHost(Duration),
}

/// Keep every core of the host busy with `per_core` threads per core while
/// `during` runs.
#[cfg(target_os = "linux")]
async fn with_loaded_host<T>(per_core: usize, during: impl std::future::Future<Output = T>) -> T {
    let busy = Arc::new(AtomicBool::new(true));
    let cores = std::thread::available_parallelism().map_or(4, usize::from);
    let spinners: Vec<_> = (0..per_core * cores)
        .map(|_| {
            let busy = Arc::clone(&busy);
            std::thread::spawn(move || {
                while busy.load(Ordering::Relaxed) {
                    std::hint::spin_loop();
                }
            })
        })
        .collect();
    let out = during.await;
    busy.store(false, Ordering::Relaxed);
    for spinner in spinners {
        spinner.join().expect("spinner");
    }
    out
}

/// Move a group of three from format bump 0 to 1, member 3 first, member 2
/// completing the majority (killed when `kill_completing`), member 1 last,
/// under a continuous workload, with `stall` caused after the first update;
/// check what the move promises. Panics on any incorrect outcome.
async fn move_under_workload(kill_completing: bool, stall: Stall) -> Outcome {
    let ([m1, m2, m3], ports) = group_of_three().await;
    let base = engine_of(&m1);

    let stop = Arc::new(AtomicBool::new(false));
    let observed = Arc::new(Mutex::new(Observed::default()));
    let writer = {
        let (stop, observed) = (Arc::clone(&stop), Arc::clone(&observed));
        tokio::spawn(async move {
            let mut seq = 0i64;
            let mut stragglers = Vec::new();
            while !stop.load(Ordering::Acquire) {
                // Each write goes to every member at once: the group writes
                // as long as one path does. An attempt held by a member that
                // can no longer commit (a leader whose followers moved on
                // waits out its proposal timeout) must not hold the next
                // write back, or the workload would measure that timeout
                // instead of the group. Two members can hold at once (that
                // leader and one just restarted), so the next write starts
                // once one attempt acknowledged, every attempt finished, or
                // `HELD` passed, whichever is first; held attempts keep
                // running and count if they land. Each attempt is its own
                // node, and every acknowledged one is checked on its own.
                const HELD: Duration = Duration::from_millis(300);
                seq += 1;
                let (done_tx, mut done_rx) = tokio::sync::mpsc::unbounded_channel();
                for port in ports {
                    let (observed, done_tx) = (Arc::clone(&observed), done_tx.clone());
                    stragglers.push(tokio::spawn(async move {
                        let at = Instant::now();
                        let written = write_seq(port, seq).await;
                        let took = at.elapsed();
                        let ok = written.is_ok();
                        {
                            let mut observed = observed.lock().unwrap();
                            if took > Duration::from_millis(250) {
                                observed.slow_writes.push((port, at, took, written));
                            }
                            if ok {
                                observed.acked.push(((seq, port), Instant::now()));
                            }
                        }
                        // The round may have moved on; nobody to tell then.
                        done_tx.send(ok).ok();
                    }));
                }
                drop(done_tx);
                let held_until = tokio::time::Instant::now() + HELD;
                while let Ok(Some(false)) =
                    tokio::time::timeout_at(held_until, done_rx.recv()).await
                {}
                stragglers.retain(|attempt| !attempt.is_finished());
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            // Attempts still held when the workload stops finish before the
            // checks read what was acknowledged.
            for attempt in stragglers {
                attempt.await.expect("write attempt");
            }
        })
    };
    let reader = {
        let (stop, observed) = (Arc::clone(&stop), Arc::clone(&observed));
        tokio::spawn(async move {
            while !stop.load(Ordering::Acquire) {
                for port in ports {
                    let began = Instant::now();
                    if let Err(why) = local_seqs(port).await {
                        observed.lock().unwrap().refused_reads.push((
                            port,
                            began,
                            Instant::now(),
                            why,
                        ));
                    }
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        })
    };
    // Each member is watched on its own: one slow to answer does not delay
    // what the others are seen to report.
    let watchers: Vec<_> = ports
        .into_iter()
        .map(|port| {
            let (stop, observed) = (Arc::clone(&stop), Arc::clone(&observed));
            tokio::spawn(async move {
                let mut last: Option<(String, u64)> = None;
                // When the previous answered request was sent: the state it
                // reported may have been read on the server any time after.
                let mut before = Instant::now();
                while !stop.load(Ordering::Acquire) {
                    let asked = Instant::now();
                    if let Some(now) = leadership_of(port).await {
                        let at = Instant::now();
                        if last.as_ref() != Some(&now) {
                            observed.lock().unwrap().leadership.push(Seen {
                                port,
                                before,
                                at,
                                leader: now.0.clone(),
                                term: now.1,
                            });
                            last = Some(now);
                        }
                        before = asked;
                    }
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
            })
        })
        .collect();
    // The processes the members run as now: a restart replaces one.
    let pids = Arc::new(Mutex::new(HashMap::from([
        (ports[0], m1.pid()),
        (ports[1], m2.pid()),
        (ports[2], m3.pid()),
    ])));
    let sampler = {
        let (stop, observed, pids) = (Arc::clone(&stop), Arc::clone(&observed), Arc::clone(&pids));
        tokio::spawn(async move {
            while !stop.load(Ordering::Acquire) {
                let now: Vec<(u16, u32)> = pids
                    .lock()
                    .unwrap()
                    .iter()
                    .map(|(p, pid)| (*p, *pid))
                    .collect();
                for (port, pid) in now {
                    if let Some((queued_ns, blocked_ns)) = waited_of(pid) {
                        observed.lock().unwrap().waits.push(Waited {
                            port,
                            pid,
                            at: Instant::now(),
                            queued_ns,
                            blocked_ns,
                        });
                    }
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        })
    };
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Each member is down from the start of its update until it serves
    // again; a read it refuses is allowed only then.
    let mut down: Vec<(u16, Instant, Instant)> = Vec::new();

    let started = Instant::now();
    let m3 = m3
        .restart_member(3, &peers_of(ports, 2), &at("1"), false)
        .await;
    down.push((ports[2], started, Instant::now()));
    pids.lock().unwrap().insert(ports[2], m3.pid());
    assert_eq!(
        engine_of(&m3),
        base + 1,
        "the server honours the format bump"
    );
    match stall {
        Stall::None => {}
        #[cfg(target_os = "linux")]
        Stall::LeaderOnLoadedHost(pause) => {
            let leader = leadership_of(ports[0])
                .await
                .and_then(|(leader, _)| leader.parse::<u64>().ok())
                .expect("the group has a leader");
            let member = match leader {
                1 => &m1,
                2 => &m2,
                _ => &m3,
            };
            with_loaded_host(2, member.freeze_for(pause)).await;
        }
        #[cfg(target_os = "linux")]
        Stall::LoadedHost(pause) => {
            with_loaded_host(16, tokio::time::sleep(pause)).await;
        }
    }
    tokio::time::sleep(Duration::from_secs(2)).await;

    let completing = Instant::now();
    let m2 = m2
        .restart_member(2, &peers_of(ports, 1), &at("1"), kill_completing)
        .await;
    let update_time = completing.elapsed();
    down.push((ports[1], completing, Instant::now()));
    pids.lock().unwrap().insert(ports[1], m2.pid());
    tokio::time::sleep(Duration::from_secs(5)).await;

    let last = Instant::now();
    let m1 = m1
        .restart_member(1, &peers_of(ports, 0), &at("1"), false)
        .await;
    down.push((ports[0], last, Instant::now()));
    pids.lock().unwrap().insert(ports[0], m1.pid());
    tokio::time::sleep(Duration::from_secs(3)).await;

    stop.store(true, Ordering::Release);
    writer.await.expect("writer");
    reader.await.expect("reader");
    for watcher in watchers {
        watcher.await.expect("watcher");
    }
    sampler.await.expect("sampler");
    let mut observed = std::mem::take(&mut *observed.lock().unwrap());
    // Held attempts land out of order; the pause check reads acks in time.
    observed.acked.sort_by_key(|(_, at)| *at);
    observed.leadership.sort_by_key(|seen| seen.at);

    // No acknowledged attempt is lost, and the member updated last catches
    // up: each acknowledged attempt is looked for on its own.
    let acked: BTreeSet<Attempt> = observed.acked.iter().map(|(attempt, _)| *attempt).collect();
    assert!(
        acked.len() > 50,
        "the workload wrote ({} acks)",
        acked.len()
    );
    let deadline = Instant::now() + Duration::from_secs(40);
    for (member, port) in [(&m1, ports[0]), (&m2, ports[1]), (&m3, ports[2])] {
        loop {
            let held = local_attempts(port).await.unwrap_or_default();
            let lost = lost_attempts(&acked, &held);
            if lost.is_empty() {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "member on {port} lacks {} acknowledged attempts: {:?}",
                lost.len(),
                &lost[..lost.len().min(10)]
            );
            tokio::time::sleep(Duration::from_millis(300)).await;
        }
        let report = member.version_report();
        assert_eq!(engine_of(member), base + 1);
        assert!(report["read_only"].is_null(), "{report}");
    }

    // Reads are served throughout. The reads are local (read concern
    // LOCAL, answered where they land), which a member serves from its own
    // state with no quorum; it may refuse one only while it is down, so a
    // refused request must have been under way then.
    for (port, began, when, why) in &observed.refused_reads {
        assert!(
            refusal_allowed(&down, *port, *began, *when),
            "member on {port} refused a local read while it was up: {why}; request {:?}..{:?}, downtime {:?}",
            began.duration_since(started),
            when.duration_since(started),
            down.iter()
                .map(|(p, from, to)| (*p, from.duration_since(started), to.duration_since(started)))
                .collect::<Vec<_>>()
        );
    }

    // Write pauses: the move's own runs while the completing member
    // restarts. Away from it the workload acknowledges within a couple
    // hundred milliseconds even across the other restarts, and the pause
    // itself runs about a second, so half a second tells them apart.
    const PAUSE: Duration = Duration::from_millis(500);
    // Every gap between consecutive acknowledgements as (from, to), from
    // the first update's start.
    let gaps: Vec<(Duration, Duration)> = observed
        .acked
        .windows(2)
        .map(|w| {
            (
                w[0].1.duration_since(started),
                w[1].1.duration_since(started),
            )
        })
        .collect();
    let pauses: Vec<(Duration, Duration)> = gaps
        .iter()
        .copied()
        .filter(|(from, to)| *to - *from > PAUSE)
        .collect();
    let mut largest: Vec<Duration> = gaps.iter().map(|(from, to)| *to - *from).collect();
    largest.sort_unstable_by(|a, b| b.cmp(a));
    largest.truncate(4);
    let windows: Vec<String> = down
        .iter()
        .map(|(port, from, to)| {
            format!(
                "{port}: {:?}..{:?}",
                from.duration_since(started),
                to.duration_since(started)
            )
        })
        .collect();
    // Attempts at a member that is down time out by the hundred; only
    // those overlapping a pause say anything about it, a few per pause so
    // every pause is accounted for.
    let slow: Vec<String> = pauses
        .iter()
        .flat_map(|(from, to)| {
            observed
                .slow_writes
                .iter()
                .filter(move |(_, at, took, _)| {
                    let begin = at.duration_since(started);
                    begin < *to && begin + *took > *from
                })
                .take(6)
        })
        .map(|(port, at, took, written)| {
            format!(
                "{port} at {:?} took {took:?} {}",
                at.duration_since(started),
                match written {
                    Ok(()) => "acked",
                    Err(why) => why.as_str(),
                }
            )
        })
        .collect();
    let leadership: Vec<String> = observed
        .leadership
        .iter()
        .map(|seen| {
            format!(
                "{} at {:?} (previous answer {:?}) sees leader {:?} term {}",
                seen.port,
                seen.at.duration_since(started),
                seen.before.saturating_duration_since(started),
                seen.leader,
                seen.term
            )
        })
        .collect();
    // A member's node id is its place in `ports`, counted from 1.
    let port_of = |node: &str| {
        node.parse::<usize>()
            .ok()
            .and_then(|n| n.checked_sub(1))
            .and_then(|i| ports.get(i).copied())
    };
    // For each pause, the member that led into it and the share of the
    // pause its process waited for CPU or block I/O.
    let waits: Vec<(Option<Identity>, Option<f64>)> = pauses
        .iter()
        .map(|pause| {
            let leader = leader_as_of(&observed.leadership, started, pause.0);
            let share = leader
                .as_ref()
                .and_then(|(node, _)| port_of(node))
                .and_then(|port| waited_share(&observed.waits, port, started, *pause));
            (leader, share)
        })
        .collect();
    let seen = format!(
        "write pauses {pauses:?}, each with its leader and the share of it that leader's \
         process waited {waits:?}, largest gaps {largest:?}; restarts {windows:?} \
         (ports {ports:?}); slow writes {slow:?}; leadership {leadership:?}"
    );
    eprintln!("{seen}");

    // Whatever else happened, the group ends with one leader every member
    // sees. (Members are sampled, so this does not prove there never were
    // two leaders; under openraft's advanced leader ids two leaders of one
    // term with different node ids are allowed anyway, and safety of the
    // acknowledged writes is what the checks above verify.)
    let mut settled = HashMap::new();
    for seen in &observed.leadership {
        settled.insert(seen.port, (seen.leader.as_str(), seen.term));
    }
    let views: BTreeSet<&(&str, u64)> = settled.values().collect();
    assert!(
        settled.len() == 3
            && views.len() == 1
            && views.iter().all(|(leader, _)| !leader.is_empty()),
        "every member ends seeing the same leader: {settled:?}; {seen}"
    );

    // The move's own pause is the one the leadership changes hands in: the
    // old-format leader gives way once a majority runs the new format, on
    // the completing member's update. The hand-over is when the new leader
    // first reported itself leading under a new identity (node and term),
    // between that answer and its previous one; the pause must overlap that
    // span. A hand-over before the update (a stalled host can cause one) is
    // not the move's.
    let (_, completing_from, completing_to) = down[1];
    let completing_from = completing_from.duration_since(started);
    let (changed_after, changed_by) =
        hand_over(&observed.leadership, started, completing_from, port_of)
            .unwrap_or_else(|| panic!("the leadership never changed hands: {seen}"));
    let own: Vec<&(Duration, Duration)> = pauses
        .iter()
        .filter(|(from, to)| *from <= changed_by && changed_after <= *to)
        .collect();
    assert_eq!(
        own.len(),
        1,
        "one pause holds the hand-over in {changed_after:?}..{changed_by:?}: {seen}"
    );
    let (from, to) = *own[0];
    assert!(
        from <= completing_to.duration_since(started) && to >= completing_from,
        "the hand-over pause lies on the completing member's update: {seen}"
    );
    assert!(
        to - from <= update_time + Duration::from_secs(20),
        "pause {:?} against an update of {update_time:?}",
        to - from
    );

    // And there only. A pause besides it counts as the host's only when the
    // process of the member that led into it waited for CPU or block I/O at
    // least `HOST_STALLED` of the pause: the host held the leader back. A
    // leader stopped or stuck inside the group waits on neither, whatever
    // the rest of the host does, so such a pause is unexplained.
    let extra: Vec<Option<f64>> = pauses
        .iter()
        .zip(&waits)
        .filter(|(pause, _)| !own.contains(pause))
        .map(|(_, (_, share))| *share)
        .collect();
    if extra.is_empty() {
        Outcome::Single
    } else if extra
        .iter()
        .all(|share| share.is_some_and(|share| share >= HOST_STALLED))
    {
        Outcome::HostStalled(seen)
    } else {
        Outcome::Unexplained(seen)
    }
}

/// Run [`move_under_workload`] until an attempt shows the single pause, at
/// most three times, with `first` caused in the first attempt only. Every
/// attempt must be correct; one the host stalled is reported and retried,
/// one with an unexplained pause fails at once, and no attempt with the
/// single pause fails as well.
async fn move_until_single_pause(kill_completing: bool, first: Stall) -> Result<(), String> {
    const ATTEMPTS: usize = 3;
    let mut stalled = Vec::new();
    for attempt in 1..=ATTEMPTS {
        let stall = if attempt == 1 { first } else { Stall::None };
        match move_under_workload(kill_completing, stall).await {
            Outcome::Single => {
                if !stalled.is_empty() {
                    eprintln!(
                        "single pause on attempt {attempt}; host stalls before: {stalled:#?}"
                    );
                }
                return Ok(());
            }
            Outcome::HostStalled(seen) => {
                eprintln!("attempt {attempt}: the host stalled the group: {seen}");
                stalled.push(seen);
            }
            Outcome::Unexplained(seen) => {
                return Err(format!(
                    "attempt {attempt}: a write pause the host does not account for: {seen}"
                ));
            }
        }
    }
    Err(format!(
        "the host stalled every attempt of {ATTEMPTS}: {stalled:#?}"
    ))
}

/// Three members updated one at a time under a continuous workload.
#[tokio::test(flavor = "multi_thread")]
async fn an_engine_format_move_loses_no_acknowledged_write() {
    move_until_single_pause(false, Stall::None)
        .await
        .unwrap_or_else(|why| panic!("{why}"));
}

/// The same with the completing member killed, so its directory carries an
/// unapplied and partly uncommitted tail when it opens at the new format.
#[tokio::test(flavor = "multi_thread")]
async fn a_move_with_the_completing_member_killed_loses_no_acknowledged_write() {
    move_until_single_pause(true, Stall::None)
        .await
        .unwrap_or_else(|why| panic!("{why}"));
}

/// The classification's negative control: in the first attempt only, the
/// leader's process stops for 800 ms while busy threads load every core of
/// the host at the same time; the attempts after it are healthy. The pause
/// inside the group is not taken for the host's although the host is loaded
/// meanwhile, so the test fails at once instead of passing on a later
/// attempt.
#[cfg(target_os = "linux")]
#[tokio::test(flavor = "multi_thread")]
async fn a_stall_inside_the_group_on_a_loaded_host_is_not_retried_away() {
    let result =
        move_until_single_pause(false, Stall::LeaderOnLoadedHost(Duration::from_millis(800))).await;
    assert!(
        result.as_ref().is_err_and(
            |why| why.starts_with("attempt 1: a write pause the host does not account for")
        ),
        "{result:?}"
    );
}

/// Its positive control: busy threads load every core of the host for
/// 3 s with nothing of the group stopped. The leader waits for CPU through
/// the pause that causes, which is taken for the host's: the attempt would
/// be retried.
#[cfg(target_os = "linux")]
#[tokio::test(flavor = "multi_thread")]
async fn a_loaded_host_is_told_from_a_stall_inside_the_group() {
    let outcome = move_under_workload(false, Stall::LoadedHost(Duration::from_secs(3))).await;
    assert!(matches!(outcome, Outcome::HostStalled(_)), "{outcome:?}");
}

/// The loss check's negative control: of the two acknowledged attempts at
/// one number, the one held covers nothing for the other.
#[test]
fn an_acknowledged_attempt_is_not_covered_by_another_at_its_number() {
    let acked = BTreeSet::from([(7, 1001), (7, 1002), (8, 1001)]);
    let held = BTreeSet::from([(7, 1002), (8, 1001), (8, 1003)]);
    assert_eq!(lost_attempts(&acked, &held), [(7, 1001)]);
}

/// The read check's negative control: a request begun once the member
/// serves again may not be refused, though it ends soon after the downtime;
/// one under way while it was down may.
#[test]
fn a_read_begun_after_the_member_serves_again_may_not_be_refused() {
    let t0 = Instant::now();
    let at = |ms: u64| t0 + Duration::from_millis(ms);
    let down = [(1001u16, at(1_000), at(2_000))];
    assert!(refusal_allowed(&down, 1001, at(1_900), at(2_100)));
    assert!(refusal_allowed(&down, 1001, at(500), at(1_100)));
    assert!(!refusal_allowed(&down, 1001, at(2_050), at(2_100)));
    assert!(!refusal_allowed(&down, 1001, at(500), at(900)));
    assert!(!refusal_allowed(&down, 1002, at(1_500), at(1_600)));
}

/// The hand-over check's controls: a new leader node keeping the term is a
/// hand-over; a member that still reports the old identity, or reports the
/// new leader late, does not move the span; the span is the new leader's
/// own, from its previous answer to the one naming itself.
#[test]
fn a_hand_over_is_the_new_leaders_own_change_of_identity() {
    let t0 = Instant::now();
    let at = |ms: u64| t0 + Duration::from_millis(ms);
    let seen = |port: u16, before: u64, now: u64, leader: &str, term: u64| Seen {
        port,
        before: at(before),
        at: at(now),
        leader: leader.to_string(),
        term,
    };
    let port_of = |node: &str| match node {
        "1" => Some(1001),
        "2" => Some(1002),
        "3" => Some(1003),
        _ => None,
    };
    let ms = Duration::from_millis;
    // Node 2 takes over in term 1, the old leader's term.
    let same_term = [
        seen(1001, 0, 0, "1", 1),
        seen(1002, 0, 0, "1", 1),
        seen(1003, 4_000, 6_000, "2", 1),
        seen(1002, 5_000, 5_200, "2", 1),
    ];
    assert_eq!(
        hand_over(&same_term, t0, ms(3_000), port_of),
        Some((ms(5_000), ms(5_200)))
    );
    // Before the hand-over looked for, a change does not count.
    assert_eq!(hand_over(&same_term, t0, ms(5_500), port_of), None);
    // A new leader that answers late: its span is wide, and still its own.
    let late = [seen(1001, 0, 0, "1", 1), seen(1002, 2_000, 9_000, "2", 2)];
    assert_eq!(
        hand_over(&late, t0, ms(1_000), port_of),
        Some((ms(2_000), ms(9_000)))
    );
}

/// Take the reachable members 1 and 2 of a group from format bump `from` to
/// `to`, member 2 first.
async fn move_two(
    m1: CoordinodeProcess,
    m2: CoordinodeProcess,
    ports: [u16; 3],
    to: &str,
) -> (CoordinodeProcess, CoordinodeProcess) {
    let m2 = m2
        .restart_member(2, &peers_of(ports, 1), &at(to), false)
        .await;
    let m1 = m1
        .restart_member(1, &peers_of(ports, 0), &at(to), false)
        .await;
    (m1, m2)
}

/// Write `seq` on whichever of `ports` leads, within 40 s.
async fn write_on_leader(ports: &[u16], seq: i64) {
    let deadline = Instant::now() + Duration::from_secs(40);
    loop {
        for port in ports {
            if write_seq(*port, seq).await.is_ok() {
                return;
            }
        }
        assert!(Instant::now() < deadline, "no member took write {seq}");
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
}

async fn wait_holds(port: u16, seqs: &BTreeSet<i64>) {
    let deadline = Instant::now() + Duration::from_secs(40);
    loop {
        if local_seqs(port)
            .await
            .is_ok_and(|held| seqs.is_subset(&held))
        {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "member on {port} never caught up"
        );
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
}

/// A member away through two moves of its group comes back two formats
/// behind: refused by name, it rejoins through the intermediate format
/// (read-only and behind there, matched and caught up at the group's).
#[tokio::test(flavor = "multi_thread")]
async fn a_member_two_formats_behind_rejoins_through_the_intermediate_format() {
    let ([m1, m2, m3], ports) = group_of_three().await;
    let base = engine_of(&m1);
    write_on_leader(&ports[..2], 1).await;
    let away = m3.stop_keeping_data().await;

    let (m1, m2) = move_two(m1, m2, ports, "1").await;
    write_on_leader(&ports[..2], 2).await;
    let (m1, m2) = move_two(m1, m2, ports, "2").await;
    write_on_leader(&ports[..2], 3).await;

    let (status, printed) = start_cluster_member_expecting_refusal_with_env(
        3,
        ports[2],
        &peers_of(ports, 2),
        away.path(),
        &at("2"),
    )
    .await;
    assert!(!status.success(), "two formats behind refuses: {printed}");
    assert!(
        printed.contains(&format!("engine format {base}")),
        "the refusal names the directory's format: {printed}"
    );

    let m3 = CoordinodeProcess::start_cluster_member_over(
        3,
        ports[2],
        &peers_of(ports, 2),
        away,
        &at("1"),
    )
    .await;
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let report = m3.version_report();
        if report["read_only"]["behind"].as_bool() == Some(true) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "never read-only behind: {report}"
        );
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
    let m3 = m3
        .restart_member(3, &peers_of(ports, 2), &at("2"), false)
        .await;
    wait_holds(ports[2], &BTreeSet::from([1, 2, 3])).await;
    let report = m3.version_report();
    assert_eq!(engine_of(&m3), base + 2);
    assert!(report["read_only"].is_null(), "{report}");
    drop((m1, m2));
}

/// The other way back for a member two formats behind: removed while away,
/// the removal committed first, its directory discarded after, and added
/// again empty at the group's format.
#[tokio::test(flavor = "multi_thread")]
async fn a_member_two_formats_behind_is_removed_and_re_added_empty() {
    use coordinode_integration::proto::admin::DecommissionNodeRequest;

    let ([m1, m2, m3], ports) = group_of_three().await;
    write_on_leader(&ports[..2], 1).await;
    let away = m3.stop_keeping_data().await;
    let (m1, m2) = move_two(m1, m2, ports, "1").await;
    let (m1, m2) = move_two(m1, m2, ports, "2").await;
    write_on_leader(&ports[..2], 2).await;

    // Removed through whichever member leads, then the directory goes.
    let deadline = Instant::now() + Duration::from_secs(40);
    let leader = loop {
        let mut removed = None;
        for member in [&m1, &m2] {
            let decommission = member
                .cluster_client()
                .await
                .decommission_node(DecommissionNodeRequest {
                    node_id: 3,
                    pruning: false,
                    force: false,
                    skip_confirmation: false,
                })
                .await;
            if decommission.is_ok() {
                removed = Some(member);
                break;
            }
        }
        if let Some(member) = removed {
            break member;
        }
        assert!(Instant::now() < deadline, "the removal never committed");
        tokio::time::sleep(Duration::from_millis(500)).await;
    };
    let mut admin = leader.cluster_client().await;
    wait_for_voters(&mut admin, 2).await;
    drop(away);

    // A member that left comes back under a new identity: the one it had
    // is retired. The harness ties a member's identity to its port.
    let returning = free_port();
    let m3 = CoordinodeProcess::start_cluster_member_with_env(
        3,
        returning,
        &peers_of(ports, 2),
        &at("2"),
    )
    .await;
    admin
        .join_node(JoinNodeRequest {
            node_id: 3,
            address: m3.member_addr(),
            pre_seeded: false,
            node_identity: m3.member_identity(),
        })
        .await
        .expect("join 3 again");
    wait_for_voters(&mut admin, 3).await;
    wait_holds(returning, &BTreeSet::from([1, 2])).await;
    assert!(m3.version_report()["read_only"].is_null());
    drop((m1, m2));
}

/// A directory two engine formats behind is refused by name and left as it
/// was; taken through the intermediate format it opens, holding its data.
/// A single-member group is the case where that is the only way.
#[tokio::test(flavor = "multi_thread")]
async fn a_directory_two_formats_behind_goes_through_the_intermediate_one() {
    let (p1, p2) = (free_port(), free_port());
    let m1 = CoordinodeProcess::start_cluster_member_with_env(1, p1, &[p2], &at("0")).await;
    m1.wait_for_leader(Duration::from_secs(20)).await;
    let base = engine_of(&m1);
    write_seq(p1, 1).await.expect("the single member writes");
    let dir = m1.stop_keeping_data().await;

    let (status, printed) =
        start_cluster_member_expecting_refusal_with_env(1, p1, &[p2], dir.path(), &at("2")).await;
    assert!(!status.success(), "two formats ahead refuses: {printed}");
    assert!(
        printed.contains(&format!("engine format {base}"))
            && printed.contains(&format!("opens only engine format {}", base + 2)),
        "the refusal names both formats: {printed}"
    );

    let m1 = CoordinodeProcess::start_cluster_member_over(1, p1, &[p2], dir, &at("1")).await;
    m1.wait_for_leader(Duration::from_secs(20)).await;
    assert_eq!(engine_of(&m1), base + 1);
    let dir = m1.stop_keeping_data().await;
    let m1 = CoordinodeProcess::start_cluster_member_over(1, p1, &[p2], dir, &at("2")).await;
    m1.wait_for_leader(Duration::from_secs(20)).await;
    assert_eq!(engine_of(&m1), base + 2);
    assert_eq!(
        local_seqs(p1).await.expect("it answers"),
        BTreeSet::from([1]),
        "the data came through both migrations"
    );
    write_seq(p1, 2)
        .await
        .expect("and the member writes at the new format");
}
