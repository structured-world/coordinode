//! `bench-graph` — server-mode graph-traversal benchmark.
//!
//! Drives a RUNNING coordinode server through the public gRPC client,
//! the same product path the vector server-mode bench uses, but for
//! GRAPH TRAVERSAL instead of vector search. It generates a synthetic
//! social graph (Person nodes + KNOWS edges with a power-law degree
//! distribution via Barabasi-Albert preferential attachment), loads it
//! over Cypher, then runs a k-hop reachability sweep at several
//! concurrency levels and reports QPS, latency percentiles, and the
//! single-node thread-scaling efficiency E_core(t) = QPS(t)/(t·QPS(1)).
//!
//! The traversal query is `MATCH (p:Person {pid:$x})-[:KNOWS*1..H]->(f)
//! RETURN count(DISTINCT f)` — a variable-length BFS that exercises the
//! engine's `execute_varlen_traverse` / `expand_one_hop` hot path and
//! the adaptive rayon parallel switch on high-fan-out frontiers.
//!
//! Competitors (Neo4j, Dgraph) load the SAME generated graph (dumped
//! via `--dump-edges`) and run the equivalent k-hop query; their cells
//! are placed alongside CoordiNode's in `bench-results/` by hand, same
//! convention as the vector competitor cells.

#![forbid(unsafe_code)]
#![warn(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use coordinode_bench::BenchReport;
use coordinode_client::{CoordinodeClient, ReadPreference, Value};
use serde::Serialize;
use tracing::{info, warn};

/// Query-replay rounds per concurrency cell.
const REPLAY_ROUNDS: usize = 5;

#[derive(Parser, Debug)]
#[command(name = "bench-graph", version)]
struct Args {
    /// gRPC endpoint(s) of running coordinode server(s). Accepts a
    /// comma-separated list; query load is round-robined across them so a
    /// replicated multi-node cluster reports read-scaling throughput
    /// (E_node = QPS(N nodes) / QPS(1 node) at fixed concurrency). The load
    /// and sanity phases always use the first endpoint.
    #[arg(long, default_value = "http://127.0.0.1:7080")]
    endpoint: String,

    /// Number of Person nodes in the synthetic graph.
    #[arg(long, default_value_t = 100_000)]
    nodes: u32,

    /// Barabasi-Albert attachment degree (edges added per new node).
    /// Mean undirected degree converges to ~2·m; the KNOWS edges are
    /// materialised in BOTH directions so traversal fan-out per node
    /// averages ~2·m.
    #[arg(long, default_value_t = 8)]
    m: u32,

    /// Comma-separated hop counts to sweep (the `H` in `*1..H`).
    #[arg(long, default_value = "1,2,3")]
    hops: String,

    /// Comma-separated concurrency levels (in-flight queries).
    #[arg(long, default_value = "1,4,8")]
    concurrency: String,

    /// Source persons sampled per round (queries per round = this).
    #[arg(long, default_value_t = 1000)]
    queries: usize,

    /// Persons / edges per UNWIND batch during load.
    #[arg(long, default_value_t = 2000)]
    batch_size: usize,

    /// Skip the load phase (graph already present from a prior run).
    #[arg(long, default_value_t = false)]
    no_load: bool,

    /// Dump the generated edge list to this path (src,dst per line,
    /// global pid space) so competitors load the identical graph.
    #[arg(long)]
    dump_edges: Option<PathBuf>,

    /// Output base directory for the report JSON.
    #[arg(long, default_value = "bench-results")]
    output: PathBuf,

    /// Dataset tag recorded in the report / filename.
    #[arg(long, default_value = "social-ba")]
    dataset_name: String,

    /// Read preference for the traversal queries: primary, primary-preferred,
    /// secondary, secondary-preferred, or nearest. The default `primary` routes
    /// every read to the Raft leader; use `nearest` (or a secondary variant)
    /// when round-robining across a replicated cluster so followers serve their
    /// workers' reads locally instead of rejecting them.
    #[arg(long, default_value = "primary")]
    read_preference: String,
}

impl Args {
    /// Parse `--endpoint` into the list of server endpoints (comma-separated).
    /// Always returns at least one entry.
    fn endpoints(&self) -> Vec<String> {
        let list: Vec<String> = self
            .endpoint
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if list.is_empty() {
            vec!["http://127.0.0.1:7080".to_string()]
        } else {
            list
        }
    }

    /// Parse `--read-preference` into the client enum. Unknown values fall back
    /// to `Primary` with a warning so a typo never silently changes routing.
    fn read_pref(&self) -> ReadPreference {
        match self.read_preference.to_ascii_lowercase().as_str() {
            "primary" => ReadPreference::Primary,
            "primary-preferred" | "primary_preferred" => ReadPreference::PrimaryPreferred,
            "secondary" => ReadPreference::Secondary,
            "secondary-preferred" | "secondary_preferred" => ReadPreference::SecondaryPreferred,
            "nearest" => ReadPreference::Nearest,
            other => {
                warn!(value = other, "unknown read-preference, using primary");
                ReadPreference::Primary
            }
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct GraphCell {
    hops: u32,
    concurrency: usize,
    qps: f64,
    e_core: f64,
    latency_us_p50: f64,
    latency_us_p99: f64,
    mean_reachable: f64,
}

/// Deterministic splitmix64 PRNG — no external rng dep, reproducible
/// graph for every run / every engine.
struct Rng(u64);
impl Rng {
    #[inline]
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    #[inline]
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

/// Barabasi-Albert preferential attachment. Returns directed edges
/// `(src, dst)` with `src < dst` by construction; both directions are
/// materialised at load time. Node ids are the pid space `0..nodes`.
fn gen_ba(nodes: u32, m: u32) -> Vec<(u32, u32)> {
    let m = m.max(1).min(nodes.saturating_sub(1).max(1));
    let mut rng = Rng(0xC007_D11E);
    // `targets` is the attachment multiset: each existing endpoint
    // appears once per incident edge, so sampling uniformly from it is
    // sampling proportional to degree.
    let mut targets: Vec<u32> = Vec::with_capacity((nodes as usize) * (m as usize) * 2);
    let mut edges: Vec<(u32, u32)> = Vec::with_capacity((nodes as usize) * (m as usize));
    // Seed clique of m+1 nodes.
    for a in 0..=m {
        for b in (a + 1)..=m {
            edges.push((a, b));
            targets.push(a);
            targets.push(b);
        }
    }
    for v in (m + 1)..nodes {
        let mut chosen = std::collections::HashSet::with_capacity(m as usize);
        let mut attempts = 0;
        while chosen.len() < m as usize && attempts < (m * 8) {
            attempts += 1;
            let t = targets[rng.below(targets.len() as u64) as usize];
            if t != v {
                chosen.insert(t);
            }
        }
        for &t in &chosen {
            let (s, d) = if t < v { (t, v) } else { (v, t) };
            edges.push((s, d));
            targets.push(t);
            targets.push(v);
        }
    }
    edges
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let args = Args::parse();
    info!(?args, "starting graph-traversal bench");

    let hops: Vec<u32> = args
        .hops
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let concurrencies: Vec<usize> = args
        .concurrency
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if hops.is_empty() || concurrencies.is_empty() {
        return Err("--hops and --concurrency must be non-empty".into());
    }

    info!(
        nodes = args.nodes,
        m = args.m,
        "generating Barabasi-Albert graph"
    );
    let edges = gen_ba(args.nodes, args.m);
    info!(edges = edges.len(), "graph generated");

    if let Some(path) = &args.dump_edges {
        let mut out = String::with_capacity(edges.len() * 12);
        for (s, d) in &edges {
            out.push_str(&s.to_string());
            out.push(',');
            out.push_str(&d.to_string());
            out.push('\n');
        }
        std::fs::write(path, out)?;
        info!(?path, "edge list dumped");
    }

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(run(&args, &edges, &hops, &concurrencies))
}

async fn run(
    args: &Args,
    edges: &[(u32, u32)],
    hops: &[u32],
    concurrencies: &[usize],
) -> Result<(), Box<dyn std::error::Error>> {
    let endpoints = args.endpoints();
    // Load and sanity always target the first endpoint; replication makes the
    // data visible on the others.
    let mut client = CoordinodeClient::connect(endpoints[0].clone()).await?;

    if !args.no_load {
        load_graph(&mut client, args, edges).await?;
        // Sanity: a sampled source must reach a non-empty neighbourhood,
        // otherwise the graph or the traversal query is broken and the
        // timings below would be meaningless.
        let probe = client
            .execute_cypher(
                "MATCH (p:Person) WHERE p.pid = 0 \
                 MATCH (p)-[:KNOWS]->(f) RETURN count(f) AS reach",
            )
            .await?;
        let reach = probe
            .first()
            .and_then(|r| match r.get("reach") {
                Some(Value::Int(n)) => Some(*n),
                _ => None,
            })
            .unwrap_or(0);
        info!(reach_1hop_pid0 = reach, "load sanity");
        if reach == 0 {
            return Err("graph sanity failed: pid 0 has zero 1-hop neighbours".into());
        }
    }

    let queries = Arc::new(sample_sources(args.nodes, args.queries));
    let mut cells: Vec<GraphCell> = Vec::new();

    for &h in hops {
        // Per-hop QPS(1) baseline for the efficiency calc.
        let mut qps1 = 0.0;
        for &c in concurrencies {
            let cell = run_cell(args, &queries, h, c).await?;
            if c == 1 {
                qps1 = cell.qps;
            }
            let e_core = if qps1 > 0.0 {
                cell.qps / (c as f64 * qps1)
            } else {
                0.0
            };
            let cell = GraphCell { e_core, ..cell };
            info!(
                hops = h,
                conc = c,
                qps = cell.qps,
                e_core = e_core,
                p99_us = cell.latency_us_p99,
                mean_reachable = cell.mean_reachable,
                "graph cell"
            );
            cells.push(cell);
        }
    }

    let mut report = BenchReport::new(
        "graph",
        "ldbc-style-traverse",
        &args.dataset_name,
        "coordinode",
        "none",
        env!("CARGO_PKG_VERSION"),
    )?;
    report.record("nodes", args.nodes)?;
    report.record("ba_m", args.m)?;
    report.record("edges", edges.len())?;
    report.record("endpoints", endpoints.len())?;
    report.record("read_preference", args.read_preference.clone())?;
    report.record("cells", serde_json::to_value(&cells)?)?;
    let path = report.write_json(&args.output, Some("GRAPH"))?;
    info!(path = ?path, "report written");
    Ok(())
}

/// Load Person nodes + a pid index + KNOWS edges (both directions).
async fn load_graph(
    client: &mut CoordinodeClient,
    args: &Args,
    edges: &[(u32, u32)],
) -> Result<(), Box<dyn std::error::Error>> {
    let t0 = Instant::now();
    // Index on Person(pid) so the per-edge endpoint lookups during edge
    // creation are point lookups, not full label scans.
    let _ = client
        .execute_cypher("CREATE INDEX person_pid ON :Person(pid)")
        .await;

    // Nodes.
    let insert_nodes = "UNWIND $batch AS pid CREATE (:Person {pid: pid})";
    let mut pid: u32 = 0;
    while pid < args.nodes {
        let end = (pid + args.batch_size as u32).min(args.nodes);
        let batch: Vec<Value> = (pid..end).map(|p| Value::Int(p as i64)).collect();
        let mut params = HashMap::new();
        params.insert("batch".to_string(), Value::List(batch));
        client
            .execute_cypher_with_params(insert_nodes, params)
            .await?;
        pid = end;
    }
    info!(
        nodes = args.nodes,
        secs = t0.elapsed().as_secs_f64(),
        "nodes loaded"
    );

    // Edges, both directions (KNOWS is symmetric in a social graph; the
    // traversal uses outgoing edges so we materialise both halves).
    let t1 = Instant::now();
    // NOTE: an inline property map referencing an unwound field
    // (`MATCH (a:Person {pid: e.s})`) silently resolves to NO match on
    // this engine, while the WHERE-equality form resolves and uses the
    // pid index. Two sequential single-node MATCHes (not a comma
    // multi-pattern, which would cartesian-product) keep each lookup an
    // index point-read.
    let insert_edges = "UNWIND $batch AS e \
         MATCH (a:Person) WHERE a.pid = e.s \
         MATCH (b:Person) WHERE b.pid = e.d \
         CREATE (a)-[:KNOWS]->(b), (b)-[:KNOWS]->(a)";
    let mut i = 0;
    while i < edges.len() {
        let end = (i + args.batch_size).min(edges.len());
        let batch: Vec<Value> = edges[i..end]
            .iter()
            .map(|(s, d)| {
                let mut row = HashMap::new();
                row.insert("s".to_string(), Value::Int(*s as i64));
                row.insert("d".to_string(), Value::Int(*d as i64));
                Value::Map(row)
            })
            .collect();
        let mut params = HashMap::new();
        params.insert("batch".to_string(), Value::List(batch));
        client
            .execute_cypher_with_params(insert_edges, params)
            .await?;
        i = end;
    }
    info!(
        edges = edges.len() * 2,
        secs = t1.elapsed().as_secs_f64(),
        "edges loaded (both directions)"
    );
    Ok(())
}

/// Deterministic source-pid sample for the query workload.
fn sample_sources(nodes: u32, count: usize) -> Vec<u32> {
    let mut rng = Rng(0x5EED_5EED);
    (0..count).map(|_| rng.below(nodes as u64) as u32).collect()
}

async fn run_cell(
    args: &Args,
    sources: &Arc<Vec<u32>>,
    hops: u32,
    concurrency: usize,
) -> Result<GraphCell, Box<dyn std::error::Error>> {
    let cypher = format!(
        "MATCH (p:Person) WHERE p.pid = $pid \
         MATCH (p)-[:KNOWS*1..{hops}]->(f) RETURN count(DISTINCT f) AS reach"
    );
    let total = sources.len() * REPLAY_ROUNDS;
    let next = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let endpoints = args.endpoints();
    let n_sources = sources.len();

    let mut workers = Vec::with_capacity(concurrency.max(1));
    let wall = Instant::now();
    let read_pref = args.read_pref();
    for w in 0..concurrency.max(1) {
        let cypher = cypher.clone();
        // Round-robin workers across endpoints so concurrency spreads evenly
        // over a replicated cluster's nodes.
        let endpoint = endpoints[w % endpoints.len()].clone();
        let sources = Arc::clone(sources);
        let next = Arc::clone(&next);
        workers.push(tokio::spawn(async move {
            let mut client = CoordinodeClient::connect(endpoint)
                .await
                .map_err(|e| e.to_string())?;
            let mut lats: Vec<f64> = Vec::new();
            let mut reach_sum: u64 = 0;
            loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= total {
                    break;
                }
                let pid = sources[i % n_sources];
                let mut params = HashMap::new();
                params.insert("pid".to_string(), Value::Int(pid as i64));
                let t = Instant::now();
                let rows = client
                    .execute_cypher_with_read_preference(&cypher, params, read_pref)
                    .await
                    .map_err(|e| e.to_string())?;
                lats.push(t.elapsed().as_micros() as f64);
                if let Some(row) = rows.first() {
                    if let Some(Value::Int(r)) = row.get("reach") {
                        reach_sum += *r as u64;
                    }
                }
            }
            Ok::<(Vec<f64>, u64), String>((lats, reach_sum))
        }));
    }

    let mut latencies: Vec<f64> = Vec::with_capacity(total);
    let mut reach_sum: u64 = 0;
    for handle in workers {
        let (mut lats, rs) = handle
            .await?
            .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
        latencies.append(&mut lats);
        reach_sum += rs;
    }
    let wall_s = wall.elapsed().as_secs_f64();
    latencies.sort_by(|a, b| a.total_cmp(b));
    let pct = |p: f64| -> f64 {
        if latencies.is_empty() {
            return 0.0;
        }
        let idx = ((latencies.len() as f64) * p).floor() as usize;
        latencies[idx.min(latencies.len() - 1)]
    };
    Ok(GraphCell {
        hops,
        concurrency,
        qps: total as f64 / wall_s.max(f64::EPSILON),
        e_core: 0.0, // filled by caller
        latency_us_p50: pct(0.50),
        latency_us_p99: pct(0.99),
        mean_reachable: reach_sum as f64 / total.max(1) as f64,
    })
}
