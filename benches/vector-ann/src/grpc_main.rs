//! `bench-vector-grpc` — server-mode vector benchmark.
//!
//! Drives a RUNNING coordinode server through the public gRPC client
//! (`coordinode-client`) instead of the in-process `HnswIndex`, so the
//! measured number includes the full product path: client serialisation,
//! HTTP/2 transport, Cypher parse/plan, engine search, result encode.
//! The delta against the in-process bench (`bench-vector-ann`) on the
//! same dataset prices the server overlay.
//!
//! Expects the dataset to be pre-loaded by `bench-vector-load` with an
//! integer `ext_id` property carrying the fvecs row index — server-side
//! node ids are allocator-assigned and do not match groundtruth rows.
//!
//! The query shape mirrors what the `VectorSearch` RPC builds
//! internally (distance + ORDER BY + LIMIT), but projects only
//! `ext_id` and the distance instead of full nodes, so the response
//! does not haul every result's embedding back over the wire.
//!
//! `ef_search` is whatever the server-side index was created with
//! (engine default 200); the request path exposes no per-query
//! override, so this bench emits a single sweep point rather than an
//! ef curve. Overlay pricing only needs matched configs on both sides.

#![forbid(unsafe_code)]
#![warn(clippy::unwrap_used, clippy::expect_used)]

#[allow(dead_code)]
mod fvecs;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use coordinode_bench::BenchReport;
use coordinode_client::{CoordinodeClient, ReadPreference, Value};
use serde::Serialize;
use tracing::info;

use crate::fvecs::{read_fvecs, read_ivecs};

/// Query-replay rounds, matching the in-process bench methodology.
const REPLAY_ROUNDS: usize = 10;

#[derive(Parser, Debug)]
#[command(name = "bench-vector-grpc", version)]
struct Args {
    /// gRPC endpoint(s) of running coordinode server(s), comma
    /// separated, e.g. `http://127.0.0.1:7080` or
    /// `http://n1:7080,http://n2:7080,http://n3:7080`. With multiple
    /// endpoints, workers are assigned round-robin so a cluster run
    /// spreads load across replicas (pick a concurrency that is a
    /// multiple of the endpoint count for an even spread).
    #[arg(long)]
    endpoint: String,

    /// Free-form topology tag recorded in the report and appended to
    /// the output filename (e.g. `L2-leader`, `L3-replicas`), so
    /// cluster cells stay distinguishable from single-node cells on
    /// the same dataset.
    #[arg(long)]
    topology: Option<String>,

    /// Read routing: `primary` (default, leader-only) or `nearest`
    /// (each worker's node serves the read locally — required when
    /// `--endpoint` spreads workers across cluster replicas, since
    /// followers reject leader-only reads).
    #[arg(long, default_value = "primary")]
    read_preference: String,

    /// Path to the query `.fvecs` file.
    #[arg(long)]
    query: PathBuf,

    /// Path to the ground-truth `.ivecs` file (row indices must match
    /// the `ext_id` property values loaded by bench-vector-load).
    #[arg(long)]
    groundtruth: PathBuf,

    /// Dataset identifier for the report (metric inferred from the
    /// `-euclidean` / `-angular` suffix, same as the in-process bench).
    #[arg(long)]
    dataset_name: String,

    /// Label the vectors were loaded under.
    #[arg(long, default_value = "Bench")]
    label: String,

    /// Property name holding the embedding.
    #[arg(long, default_value = "embedding")]
    property: String,

    /// k for recall@k.
    #[arg(long, default_value_t = 10)]
    k: usize,

    /// Number of concurrent in-flight queries. `1` = sequential
    /// latency-per-query convention; `N>1` = aggregate throughput over
    /// N workers (gRPC channel multiplexes one HTTP/2 connection).
    #[arg(long, default_value_t = 1)]
    concurrency: usize,

    /// Cap on queries per replay round (0 = all). Server-mode rounds
    /// are slower than in-process; the cap keeps smoke runs short.
    #[arg(long, default_value_t = 0)]
    max_queries: usize,

    /// Output base directory for the report JSON.
    #[arg(long, default_value = "bench-results")]
    output: PathBuf,
}

#[derive(Debug, Clone, Serialize)]
struct GrpcSweepPoint {
    recall_at_k: f64,
    qps: f64,
    latency_us_mean: f64,
    latency_us_p50: f64,
    latency_us_p95: f64,
    latency_us_p99: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let args = Args::parse();
    info!(?args, "starting server-mode vector bench");

    let (dim, query_flat) = read_fvecs(&args.query)?;
    let (gt_k, gt_flat) = read_ivecs(&args.groundtruth)?;
    let n_test_full = query_flat.len() / dim;
    if gt_flat.len() / gt_k != n_test_full {
        return Err("ground-truth row count != query count".into());
    }
    if gt_k < args.k {
        return Err(format!("ground truth K ({gt_k}) < --k ({})", args.k).into());
    }
    let n_test = if args.max_queries > 0 {
        n_test_full.min(args.max_queries)
    } else {
        n_test_full
    };

    // Same suffix convention as the in-process bench: `-euclidean` -> L2
    // (vector_distance ASC), `-angular` -> cosine (vector_similarity DESC).
    let (score_fn, order_dir) = if args.dataset_name.ends_with("-euclidean") {
        ("vector_distance", "ASC")
    } else if args.dataset_name.ends_with("-angular") {
        ("vector_similarity", "DESC")
    } else {
        return Err("dataset name must end in -euclidean or -angular".into());
    };
    // `WITH *` (not a narrowed projection) is REQUIRED: the planner's
    // VectorTopK rewrite pattern-matches the Sort+Limit over a
    // star-preserving WITH, exactly the shape the VectorSearch RPC
    // emits. A narrowed `WITH n.ext_id AS ...` breaks the match and
    // the query silently degrades to the brute-force O(N) scan
    // (measured: 25 QPS vs HNSW-backed three-digit QPS on the same
    // data). The final RETURN projects only ext_id so the response
    // stays light.
    let cypher = format!(
        "MATCH (n:{label}) \
         WITH *, {score_fn}(n.{prop}, $qv) AS _s \
         ORDER BY _s {order_dir} \
         LIMIT {k} \
         RETURN n.ext_id AS ext_id",
        label = args.label,
        prop = args.property,
        k = args.k,
    );
    info!(%cypher, n_test, concurrency = args.concurrency, "query template ready");

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let point = runtime.block_on(run_rounds(&args, &cypher, &query_flat, dim, n_test))?;

    // Recall against groundtruth is accumulated inside run_rounds via
    // the shared counters; recompute here from the returned point.
    info!(
        recall = point.recall_at_k,
        qps = point.qps,
        p99_us = point.latency_us_p99,
        "server-mode sweep point"
    );

    // gt arrays only feed recall (already folded in); silence the
    // unused warning if compiled without the check below.
    let _ = (&gt_flat, gt_k);

    let version = env!("CARGO_PKG_VERSION").to_string();
    let mut report = BenchReport::new(
        "vector",
        "ann-benchmarks",
        &args.dataset_name,
        "coordinode",
        "none",
        version,
    )?;
    report.record("transport", "grpc")?;
    report.record("concurrency", args.concurrency)?;
    report.record(
        "endpoints",
        args.endpoint.split(',').filter(|s| !s.is_empty()).count(),
    )?;
    if let Some(ref topo) = args.topology {
        report.record("topology", topo.as_str())?;
    }
    report.record("read_preference", args.read_preference.as_str())?;
    report.record("dataset_n_test", n_test)?;
    report.record("dataset_dim", dim)?;
    report.record("k", args.k)?;
    report.record("sweep", vec![point.clone()])?;
    report.record("recall_at_k_peak", point.recall_at_k)?;
    report.record("qps_at_recall_peak", point.qps)?;
    if point.recall_at_k >= 0.95 {
        report.record("qps_at_recall_0_95", point.qps)?;
    }
    // Filename tag keeps cluster cells (L2/L3) from overwriting the
    // single-node GRPC cell for the same dataset.
    let tag = match args.topology {
        Some(ref t) => format!("GRPC-{t}"),
        None => "GRPC".to_string(),
    };
    let path = report.write_json(&args.output, Some(&tag))?;
    info!(path = ?path, "report written");
    Ok(())
}

/// Run REPLAY_ROUNDS over the (possibly capped) query set with the
/// configured concurrency; returns the aggregate point. Recall is
/// computed against the groundtruth rows by `ext_id` match.
async fn run_rounds(
    args: &Args,
    cypher: &str,
    query_flat: &[f32],
    dim: usize,
    n_test: usize,
) -> Result<GrpcSweepPoint, Box<dyn std::error::Error>> {
    let (gt_k, gt_flat) = read_ivecs(&args.groundtruth)?;
    let gt = Arc::new(gt_flat);
    let queries = Arc::new(query_flat.to_vec());

    let total = n_test * REPLAY_ROUNDS;
    let wall = Instant::now();

    // Worker model: each worker owns ONE client (= one HTTP/2 channel,
    // matching a real client process) and pulls the next global query
    // index from a shared atomic counter until the stream is drained.
    // `execute_cypher_with_params` takes `&mut self`, so ownership per
    // worker is also the natural borrow shape.
    type WorkerOut = Result<(Vec<f64>, u64, u64), String>;
    let next = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut workers: Vec<tokio::task::JoinHandle<WorkerOut>> =
        Vec::with_capacity(args.concurrency.max(1));
    // Round-robin worker -> endpoint assignment; single endpoint is
    // the degenerate one-element case.
    let endpoints: Vec<String> = args
        .endpoint
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if endpoints.is_empty() {
        return Err("--endpoint resolved to an empty list".into());
    }
    let read_pref = match args.read_preference.as_str() {
        "primary" => ReadPreference::Primary,
        "nearest" => ReadPreference::Nearest,
        other => return Err(format!("unknown --read-preference '{other}'").into()),
    };
    for w in 0..args.concurrency.max(1) {
        let endpoint = endpoints[w % endpoints.len()].clone();
        let cypher = cypher.to_string();
        let queries = Arc::clone(&queries);
        let gt = Arc::clone(&gt);
        let next = Arc::clone(&next);
        let k = args.k;
        workers.push(tokio::spawn(async move {
            let mut client = CoordinodeClient::connect(endpoint)
                .await
                .map_err(|e| e.to_string())?;
            let mut lats: Vec<f64> = Vec::new();
            let mut hits = 0u64;
            let mut denom = 0u64;
            loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= total {
                    break;
                }
                let q_idx = i % n_test;
                let qv = queries[q_idx * dim..(q_idx + 1) * dim].to_vec();
                let mut params = std::collections::HashMap::new();
                params.insert("qv".to_string(), Value::Vector(qv));
                let t = Instant::now();
                let rows = client
                    .execute_cypher_with_read_preference(&cypher, params, read_pref)
                    .await
                    .map_err(|e| e.to_string())?;
                lats.push(t.elapsed().as_micros() as f64);
                let gt_row: std::collections::HashSet<i64> = gt[q_idx * gt_k..q_idx * gt_k + k]
                    .iter()
                    .map(|x| i64::from(*x))
                    .collect();
                for row in rows.iter().take(k) {
                    if let Some(Value::Int(id)) = row.get("ext_id") {
                        if gt_row.contains(id) {
                            hits += 1;
                        }
                    }
                }
                denom += k as u64;
            }
            Ok((lats, hits, denom))
        }));
    }
    let mut latencies: Vec<f64> = Vec::with_capacity(total);
    let mut hits = 0u64;
    let mut denom = 0u64;
    for handle in workers {
        let (mut lats, h, d) = handle
            .await?
            .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
        latencies.append(&mut lats);
        hits += h;
        denom += d;
    }

    let wall_s = wall.elapsed().as_secs_f64();
    latencies.sort_by(|a, b| a.total_cmp(b));
    let mean = latencies.iter().sum::<f64>() / latencies.len().max(1) as f64;
    let pct = |p: f64| -> f64 {
        if latencies.is_empty() {
            return 0.0;
        }
        let idx = ((latencies.len() as f64) * p).floor() as usize;
        latencies[idx.min(latencies.len() - 1)]
    };
    Ok(GrpcSweepPoint {
        recall_at_k: hits as f64 / denom.max(1) as f64,
        qps: total as f64 / wall_s,
        latency_us_mean: mean,
        latency_us_p50: pct(0.50),
        latency_us_p95: pct(0.95),
        latency_us_p99: pct(0.99),
    })
}
