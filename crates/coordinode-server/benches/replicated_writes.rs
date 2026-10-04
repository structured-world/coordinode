//! Write throughput of one replicated group of three, measured the way the
//! same load is measured against MongoDB: one group of three members behind
//! the node clients talk to, clients over the network, writes acknowledged by
//! a majority with the journal.
//!
//! Each node serves its Raft and Cypher gRPC on one port, as a deployed node
//! does; the clients connect to the leader. Modes:
//!   insert  one node with an indexed `email` per statement
//!   txn     three such nodes in one statement (one transaction)
//!   mix     after each client creates its own nodes, one statement at a
//!           time chosen evenly among insert, update, delete and read, each
//!           finding its node by the indexed `email`
//! Printed: statements per second, latency distribution, and the bytes the
//! leader's Raft log grew by per statement.
//!
//! Run: cargo bench -p coordinode-server --bench replicated_writes -- \
//!        [insert|txn|mix] [threads] [seconds] [resolved|derived] [full|fsync|open_datasync]

#![allow(clippy::expect_used, clippy::print_stdout, clippy::unwrap_used)]

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_embed::Database;
use coordinode_raft::cluster::RaftNode;
use coordinode_raft::proposal::RaftProposalPipeline;
use coordinode_raft::proto::replication::raft_service_server::RaftServiceServer;
use coordinode_server::proto::common::{PropertyValue, property_value};
use coordinode_server::proto::query;
use coordinode_server::services::cypher::CypherServiceImpl;
use coordinode_storage::engine::config::{
    Durability, EndpointConfig, Media, StorageConfig, SyncMethod, Tier,
};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_test_fixtures::alloc_port;
use parking_lot::RwLock;

struct Node {
    db: Arc<RwLock<Database>>,
    raft: Arc<RaftNode>,
    dir: tempfile::TempDir,
}

async fn open_node(node_id: u64, port: u16, leader: bool, sync: SyncMethod) -> Node {
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = Arc::new(TimestampOracle::new());
    let mut config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    config.oplog_sync_method = sync;
    let engine = Arc::new(StorageEngine::open_with_oracle(&config, oracle.clone()).expect("open"));
    let (raft, raft_handler) = if leader {
        RaftNode::open_cluster_embedded(
            node_id,
            Arc::clone(&engine),
            format!("http://127.0.0.1:{port}"),
        )
        .await
        .expect("leader")
    } else {
        RaftNode::open_joining_embedded(node_id, Arc::clone(&engine))
            .await
            .expect("joining")
    };
    let raft = Arc::new(raft);
    let pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> =
        Arc::new(RaftProposalPipeline::new(Arc::clone(raft.raft())));
    let db = Arc::new(RwLock::new(
        Database::from_engine(dir.path(), engine, oracle, pipeline).expect("database"),
    ));
    let cypher = CypherServiceImpl::new(
        Arc::clone(&db),
        Arc::new(coordinode_query::advisor::QueryRegistry::new()),
        Arc::new(coordinode_query::advisor::nplus1::NPlus1Detector::new()),
    )
    .with_raft_node(Arc::clone(&raft));
    let addr: std::net::SocketAddr = format!("127.0.0.1:{port}").parse().expect("addr");
    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(RaftServiceServer::new(raft_handler))
            .add_service(query::cypher_service_server::CypherServiceServer::new(
                cypher,
            ))
            .serve(addr)
            .await
    });
    Node { db, raft, dir }
}

/// Bytes of the Raft log segments under `data_dir`.
fn log_bytes(data_dir: &Path) -> u64 {
    fn walk(dir: &Path) -> u64 {
        std::fs::read_dir(dir).map_or(0, |entries| {
            entries
                .filter_map(Result::ok)
                .map(|e| {
                    let path = e.path();
                    if path.is_dir() {
                        walk(&path)
                    } else {
                        e.metadata().map_or(0, |m| m.len())
                    }
                })
                .sum()
        })
    }
    walk(&data_dir.join("oplog"))
}

fn text(s: String) -> PropertyValue {
    PropertyValue {
        value: Some(property_value::Value::StringValue(s)),
    }
}

fn float(f: f64) -> PropertyValue {
    PropertyValue {
        value: Some(property_value::Value::FloatValue(f)),
    }
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

/// One client: statements against the leader until `deadline`.
async fn client(port: u16, txn: bool, worker: u64, deadline: Instant) -> (Vec<Duration>, usize) {
    let mut client = query::cypher_service_client::CypherServiceClient::connect(format!(
        "http://127.0.0.1:{port}"
    ))
    .await
    .expect("connect");
    let rows = if txn { 3 } else { 1 };
    let query = if txn {
        "CREATE (:User {email: $e0, r: $r0}), (:User {email: $e1, r: $r1}), \
         (:User {email: $e2, r: $r2})"
    } else {
        "CREATE (:User {email: $e0, r: $r0})"
    };
    let (mut samples, mut errors, mut n) = (Vec::new(), 0usize, 0u64);
    while Instant::now() < deadline {
        let mut parameters = HashMap::new();
        for i in 0..rows {
            let key = (worker << 40) | n;
            n += 1;
            parameters.insert(format!("e{i}"), text(format!("user-{key}@example.com")));
            parameters.insert(format!("r{i}"), float((key % 1000) as f64 / 1000.0));
        }
        let t = Instant::now();
        let result = client
            .execute_cypher(query::ExecuteCypherRequest {
                query: query.to_string(),
                parameters,
                ..Default::default()
            })
            .await;
        match result {
            Ok(_) => samples.push(t.elapsed()),
            Err(e) => {
                errors += 1;
                if errors <= 3 {
                    println!("worker {worker}: {e}");
                }
            }
        }
    }
    (samples, errors)
}

/// Nodes each mix client creates before its measured window opens.
const PRELOAD: u64 = 200;

/// The kinds of statement the mix runs, in an even share.
const KINDS: [&str; 4] = ["insert", "update", "delete", "read"];

/// One mix client: its own nodes first, then `seconds` of statements of the
/// kinds in [`KINDS`], each over a node it holds. Latencies by kind.
async fn mix_client(port: u16, worker: u64, seconds: u64) -> ([Vec<Duration>; 4], usize) {
    let mut client = query::cypher_service_client::CypherServiceClient::connect(format!(
        "http://127.0.0.1:{port}"
    ))
    .await
    .expect("connect");
    let mut n = 0u64;
    // xorshift: the share of each kind only needs to be even, not secret.
    let mut state = worker | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut live: Vec<String> = Vec::new();
    let new_email = |n: &mut u64| {
        let key = (worker << 40) | *n;
        *n += 1;
        format!("user-{key}@example.com")
    };
    let call = async |client: &mut query::cypher_service_client::CypherServiceClient<
        tonic::transport::Channel,
    >,
                      query: &str,
                      email: String,
                      r: f64| {
        let mut parameters = HashMap::new();
        parameters.insert("e".to_string(), text(email));
        parameters.insert("r".to_string(), float(r));
        client
            .execute_cypher(query::ExecuteCypherRequest {
                query: query.to_string(),
                parameters,
                ..Default::default()
            })
            .await
    };
    const CREATE: &str = "CREATE (:User {email: $e, r: $r})";
    for _ in 0..PRELOAD {
        let email = new_email(&mut n);
        call(&mut client, CREATE, email.clone(), 0.5)
            .await
            .expect("preload");
        live.push(email);
    }

    let deadline = Instant::now() + Duration::from_secs(seconds);
    let mut samples: [Vec<Duration>; 4] = Default::default();
    let mut errors = 0usize;
    while Instant::now() < deadline {
        let kind = if live.is_empty() {
            0
        } else {
            (next() % 4) as usize
        };
        let r = (next() % 1000) as f64 / 1000.0;
        let (query, email) = match kind {
            0 => (CREATE, new_email(&mut n)),
            1 => (
                "MATCH (u:User {email: $e}) SET u.r = $r",
                live[(next() as usize) % live.len()].clone(),
            ),
            2 => {
                let i = (next() as usize) % live.len();
                (
                    "MATCH (u:User {email: $e}) DETACH DELETE u",
                    live.swap_remove(i),
                )
            }
            _ => (
                "MATCH (u:User {email: $e}) RETURN u.r",
                live[(next() as usize) % live.len()].clone(),
            ),
        };
        let t = Instant::now();
        match call(&mut client, query, email.clone(), r).await {
            Ok(_) => {
                samples[kind].push(t.elapsed());
                if kind == 0 {
                    live.push(email);
                }
            }
            Err(e) => {
                errors += 1;
                if errors <= 3 {
                    println!("worker {worker}: {e}");
                }
            }
        }
    }
    (samples, errors)
}

/// The mix with `threads` clients for `seconds`, one line per kind and a
/// total.
async fn run_mix(port: u16, (round, threads, seconds): (u64, u64, u64), label: &str) {
    let tasks: Vec<_> = (0..threads)
        .map(|w| tokio::spawn(mix_client(port, (round << 8) | w, seconds)))
        .collect();
    let mut by_kind: [Vec<Duration>; 4] = Default::default();
    let mut errors = 0;
    for t in tasks {
        let (s, e) = t.await.expect("client");
        for (all, mine) in by_kind.iter_mut().zip(s) {
            all.extend(mine);
        }
        errors += e;
    }
    let mut total = 0;
    for (kind, samples) in KINDS.iter().zip(by_kind.iter_mut()) {
        samples.sort_unstable();
        total += samples.len();
        println!(
            "{label} {kind:<7} ops={:>6} tps={:>6.0}  p50={:>8.2?} p99={:>8.2?} p999={:>8.2?}",
            samples.len(),
            samples.len() as f64 / seconds as f64,
            percentile(samples, 0.50),
            percentile(samples, 0.99),
            percentile(samples, 0.999),
        );
    }
    println!(
        "{label} total   ops={total:>6} errors={errors} tps={:>6.0}",
        total as f64 / seconds as f64
    );
}

/// Run `round` (distinct per run, so no two runs write the same key).
async fn run(
    leader: &Node,
    port: u16,
    txn: bool,
    (round, threads, seconds): (u64, u64, u64),
    label: &str,
) {
    let before = log_bytes(leader.dir.path());
    let deadline = Instant::now() + Duration::from_secs(seconds);
    let started = Instant::now();
    let tasks: Vec<_> = (0..threads)
        .map(|w| tokio::spawn(client(port, txn, (round << 8) | w, deadline)))
        .collect();
    let mut samples = Vec::new();
    let mut errors = 0;
    for t in tasks {
        let (s, e) = t.await.expect("client");
        samples.extend(s);
        errors += e;
    }
    let wall = started.elapsed();
    let grown = log_bytes(leader.dir.path()) - before;
    samples.sort_unstable();
    let ops = samples.len();
    let total: Duration = samples.iter().sum();
    println!(
        "{label:<22} ops={ops:>6} errors={errors} tps={:>6.0} rows/s={:>6.0}  mean={:>8.2?} p50={:>8.2?} p99={:>8.2?} p999={:>8.2?} max={:>8.2?}  log bytes/op={:>6.1}",
        ops as f64 / wall.as_secs_f64(),
        ops as f64 * if txn { 3.0 } else { 1.0 } / wall.as_secs_f64(),
        total.checked_div(ops.max(1) as u32).unwrap_or_default(),
        percentile(&samples, 0.50),
        percentile(&samples, 0.99),
        percentile(&samples, 0.999),
        samples.last().copied().unwrap_or_default(),
        grown as f64 / ops.max(1) as f64,
    );
}

fn main() {
    let args: Vec<String> = std::env::args()
        .skip(1)
        .filter(|a| a != "--bench")
        .collect();
    let mode = args.first().map_or("insert", String::as_str).to_owned();
    let threads: u64 = args.get(1).map_or(16, |a| a.parse().expect("threads"));
    let seconds: u64 = args.get(2).map_or(30, |a| a.parse().expect("seconds"));
    let profile = args.get(3).map_or("resolved", String::as_str).to_owned();
    let sync = match args.get(4).map_or("full", String::as_str) {
        "full" => SyncMethod::Full,
        "fsync" => SyncMethod::Fsync,
        "open_datasync" => SyncMethod::OpenDatasync,
        other => {
            eprintln!("sync must be full, fsync or open_datasync, got {other}");
            std::process::exit(2);
        }
    };
    let txn = match mode.as_str() {
        "insert" | "mix" => false,
        "txn" => true,
        other => {
            eprintln!("mode must be insert, txn or mix, got {other}");
            std::process::exit(2);
        }
    };

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("runtime");
    runtime.block_on(async {
        let ports = [alloc_port(), alloc_port(), alloc_port()];
        let n1 = open_node(1, ports[0], true, sync).await;
        let n2 = open_node(2, ports[1], false, sync).await;
        let n3 = open_node(3, ports[2], false, sync).await;
        for _ in 0..150 {
            if n1.raft.is_leader().await {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        for (id, port) in [(2u64, ports[1]), (3, ports[2])] {
            n1.raft
                .add_node(id, format!("http://127.0.0.1:{port}"))
                .await
                .expect("add");
        }
        n1.raft
            .change_membership(vec![1, 2, 3])
            .await
            .expect("membership");

        let ddl = format!(
            "CREATE INDEX user_email ON :User(email) OPTIONS {{maintenance: '{profile}'}}"
        );
        tokio::task::block_in_place(|| n1.db.write().execute_cypher(&ddl)).expect("index");

        println!(
            "== one group of three, {threads} clients over gRPC to the leader, w:majority, {profile} index, sync {sync:?} =="
        );
        if mode == "mix" {
            run_mix(ports[0], (1, threads, 3), "warm-up").await;
            run_mix(ports[0], (2, threads, seconds), &format!("mix {profile}")).await;
        } else {
            run(&n1, ports[0], txn, (1, threads, 3), "warm-up").await;
            run(
                &n1,
                ports[0],
                txn,
                (2, threads, seconds),
                &format!("{mode} {profile}"),
            )
            .await;
        }

        for n in [&n3, &n2, &n1] {
            n.raft.shutdown().await.expect("shutdown");
        }
    });
}
