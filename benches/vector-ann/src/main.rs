//! `bench-vector-ann` — ann-benchmarks adapter for CoordiNode HNSW.
//!
//! Reads an ann-benchmarks HDF5 dataset (e.g.
//! `sift-128-euclidean.hdf5`) from disk, builds the CN HNSW index
//! over `train` vectors, then runs `test` queries at a sweep of
//! `ef_search` values, recording recall@10 and QPS at each point.
//!
//! Output: canonical [`coordinode_bench::BenchReport`] JSON in
//! `bench-results/vector/<dataset>/<sha>-coordinode-<timestamp>.json`.
//!
//! ## Dataset format
//!
//! `ann-benchmarks` HDF5 datasets carry four arrays:
//! - `/train` — `(N, D)` f32, training vectors used to build the index
//! - `/test`  — `(Q, D)` f32, query vectors
//! - `/neighbors` — `(Q, K)` i32, ground-truth nearest neighbour ids
//!   (referencing `/train` row indices)
//! - `/distances` — `(Q, K)` f32, ground-truth distances (unused by us;
//!   we measure recall against the id set, not the distance set)
//!
//! ## Sweep
//!
//! The runner iterates `ef_search ∈ {16, 32, 64, 128, 256, 512}` by
//! default (CLI override available). For each value: search every
//! query, accumulate recall@10 (overlap with ground-truth top-10),
//! measure wall time of the search loop, derive QPS = Q / wall_s.
//!
//! Per ann-benchmarks methodology: 10 query-replay rounds, warm cache,
//! single-thread search (multi-thread is a separate sweep dimension
//! tracked under `qps_per_thread`).

#![forbid(unsafe_code)]
#![warn(clippy::unwrap_used, clippy::expect_used)]

mod fvecs;

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use coordinode_bench::BenchReport;
use coordinode_core::graph::types::VectorMetric;
use coordinode_vector::hnsw::{HnswConfig, HnswIndex, QuantizationCodec};
use serde::Serialize;
use tracing::info;

use crate::fvecs::{read_fvecs, read_ivecs};

/// Read current process resident set size in KiB.
///
/// Linux-only via `/proc/self/status` (VmRSS line). Returns `None` on
/// non-Linux hosts so the report degrades gracefully — the field is
/// dropped from `metrics` rather than reported as a meaningless zero.
/// The bench host (ro / <redacted>) is Linux, so the donor sweep
/// always carries memory numbers; macOS dev runs simply omit them.
fn read_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status
        .lines()
        .find(|l| l.starts_with("VmRSS:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse::<u64>().ok())
}

/// Default ef_search sweep — covers low-recall (ef=10) to
/// high-recall (ef=800) regions. Matches the ann-benchmarks
/// hnswlib default sweep so QPS@recall comparisons stop being
/// penalised by coarser granularity on our side: prior sweep
/// `[16, 32, 64, 128, …]` jumped over the typical recall=0.95
/// inflection point (recall=0.91 at ef=32, recall=0.96 at ef=64
/// on SIFT-128 M=16) — the dashboard then picked the slower
/// ef=64 cell as "QPS@recall≥0.95" while hnswlib reported the
/// faster ef=80 cell sitting just above 0.95. Same code, denser
/// sweep, fairer comparison.
const DEFAULT_EF_SWEEP: &[usize] = &[10, 20, 40, 60, 80, 120, 160, 200, 300, 400, 600, 800];

/// Number of query-replay rounds per sweep point. ann-benchmarks
/// uses 10 by default — accumulates timing across replays and
/// reports QPS as `(rounds * queries) / wall_seconds`.
const REPLAY_ROUNDS: usize = 10;

#[derive(Parser, Debug)]
#[command(name = "bench-vector-ann", version)]
struct Args {
    /// Path to the training `.fvecs` file
    /// (e.g. `<bench-data-root>/datasets/sift/sift_base.fvecs`).
    #[arg(long)]
    train: PathBuf,

    /// Path to the query `.fvecs` file
    /// (e.g. `<bench-data-root>/datasets/sift/sift_query.fvecs`).
    #[arg(long)]
    query: PathBuf,

    /// Path to the ground-truth `.ivecs` file
    /// (e.g. `<bench-data-root>/datasets/sift/sift_groundtruth.ivecs`).
    #[arg(long)]
    groundtruth: PathBuf,

    /// Dataset identifier embedded in the report
    /// (e.g. `sift-128-euclidean`). Free-form; the gh-pages chart
    /// uses this as the chart title key. **Metric is detected
    /// from the `-euclidean` / `-angular` suffix.**
    #[arg(long)]
    dataset_name: String,

    /// HNSW M parameter (max connections per layer).
    #[arg(long, default_value_t = 32)]
    m: usize,

    /// HNSW ef_construction parameter (build-time candidate list).
    #[arg(long, default_value_t = 200)]
    ef_construction: usize,

    /// Codec mode tag — `none` or `zstd`. Recorded in the report
    /// for codec-axis filtering on the dashboard.
    #[arg(long, default_value = "none")]
    codec: String,

    /// Output base directory — bench results land at
    /// `<output>/vector/<dataset_name>/<sha>-coordinode-<stamp>.json`.
    #[arg(long, default_value = "bench-results")]
    output: PathBuf,

    /// k for recall@k. Default 10 per ann-benchmarks.
    #[arg(long, default_value_t = 10)]
    k: usize,

    /// Comma-separated ef_search sweep values (overrides default).
    #[arg(long)]
    ef_sweep: Option<String>,

    /// In-RAM quantization codec: `none` (f32), `sq8`, or `rabitq`.
    /// `rabitq` requires a cosine / angular dataset — on `-euclidean`
    /// data the kernel currently falls back to f32 (the polarisation
    /// identity for L2 is a follow-up). Default: `none`.
    #[arg(long, default_value = "none")]
    quantization: String,

    /// RaBitQ bit-width when `--quantization=rabitq`. `1` is the
    /// classic SIGMOD 2024 sign-bit popcount kernel; `2..=4` selects
    /// Extended-RaBitQ (R862) which trades RAM for recall — 2-bit
    /// typically reaches the same recall as 1-bit + heavy rerank in
    /// 2× the code size and no rerank.
    ///
    /// Ignored when `--quantization` is `none` or `sq8`.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u8).range(1..=4))]
    rabitq_bits: u8,

    /// Thread count for the rayon worker pool used by HNSW build
    /// (`insert_batch` parallel plan + apply phases). `0` = take
    /// the rayon default (one thread per logical core). Set to
    /// a non-zero value for fair vs single-thread competitors
    /// (`--threads 1` for hnswlib default) or fixed-budget scaling
    /// sweeps (`--threads 4`).
    ///
    /// Search remains single-threaded inside this binary: `sweep_one`
    /// loops queries sequentially, so per-query latency / QPS are
    /// always reported on one core regardless of this flag. This
    /// flag only governs the rayon pool used during graph construction.
    #[arg(long, default_value_t = 0)]
    threads: usize,
}

#[derive(Debug, Clone, Serialize)]
struct SweepPoint {
    ef_search: usize,
    recall_at_k: f64,
    /// Mean queries-per-second across REPLAY_ROUNDS.
    qps: f64,
    /// Mean per-query latency (microseconds).
    latency_us_mean: f64,
    /// p50 per-query latency.
    latency_us_p50: f64,
    /// p95 per-query latency.
    latency_us_p95: f64,
    /// p99 per-query latency.
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
    info!(?args, "starting ann-benchmarks adapter");

    // Pin the global rayon pool BEFORE any HNSW operation kicks in.
    // First call to `rayon::current_num_threads()` lazily-inits the
    // pool to one-thread-per-logical-core, so the builder must run
    // first or it silently no-ops on the second attempt.
    //
    // Honoring this knob is what makes the bench thread-budget-fair:
    // setting `--threads 1` matches single-thread competitors
    // (hnswlib default), `--threads 4` matches the fixed-budget
    // sweep against qdrant / hnswlib `set_num_threads(4)`. The
    // value `0` opts back into rayon's heuristic.
    if args.threads > 0 {
        if let Err(e) = rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
        {
            return Err(format!(
                "rayon ThreadPoolBuilder::build_global failed (already initialised?): {e}"
            )
            .into());
        }
        info!(
            rayon_threads = args.threads,
            "rayon pool pinned for build (search remains single-threaded)"
        );
    }

    let sweep: Vec<usize> = match &args.ef_sweep {
        Some(s) => s
            .split(',')
            .map(|x| x.trim().parse::<usize>())
            .collect::<Result<Vec<_>, _>>()?,
        None => DEFAULT_EF_SWEEP.to_vec(),
    };

    // ── Load dataset ────────────────────────────────────────
    info!(train=?args.train, "loading training set");
    let (train_dim, train_flat) = read_fvecs(&args.train)?;
    info!(query=?args.query, "loading query set");
    let (query_dim, query_flat) = read_fvecs(&args.query)?;
    info!(gt=?args.groundtruth, "loading ground-truth");
    let (gt_k, gt_flat) = read_ivecs(&args.groundtruth)?;

    if train_dim != query_dim {
        return Err(format!(
            "train dim ({train_dim}) != query dim ({query_dim}) — wrong dataset pair?"
        )
        .into());
    }
    let d = train_dim;
    let n_train = train_flat.len() / d;
    let n_test = query_flat.len() / d;
    let gt_n = gt_flat.len() / gt_k;
    if gt_n != n_test {
        return Err(format!("ground-truth row count ({gt_n}) != query count ({n_test})").into());
    }
    info!(n_train, dim = d, n_test, gt_k, "dataset loaded");

    if gt_k < args.k {
        return Err(format!("ground truth K ({}) is smaller than --k ({})", gt_k, args.k).into());
    }

    // ── Detect metric from dataset name ─────────────────────
    // ann-benchmarks naming convention: `<set>-<dim>-<metric>`.
    // `euclidean` → L2, `angular` → Cosine (after normalising).
    let metric = if args.dataset_name.ends_with("-euclidean") {
        VectorMetric::L2
    } else if args.dataset_name.ends_with("-angular") {
        VectorMetric::Cosine
    } else {
        return Err(format!(
            "cannot detect metric from dataset name {:?} (expected suffix `-euclidean` or `-angular`)",
            args.dataset_name
        )
        .into());
    };
    info!(?metric, "metric inferred from dataset name");

    // ── Build HNSW ──────────────────────────────────────────
    // RSS BEFORE build: capture the baseline that includes the loaded
    // dataset arrays (train + query + ground-truth) but no index.
    // Difference (after_build - before_build) is the index footprint.
    let rss_kib_before_build = read_rss_kib();
    let build_start = Instant::now();
    let m_max0 = args.m * 2;
    let quantization = match args.quantization.as_str() {
        "none" => QuantizationCodec::None,
        "sq8" => QuantizationCodec::Sq8,
        "rabitq" => QuantizationCodec::RaBitQ {
            bits: args.rabitq_bits,
        },
        other => {
            return Err(format!(
                "--quantization: expected `none`, `sq8`, or `rabitq`, got `{other}`"
            )
            .into())
        }
    };
    let config = HnswConfig {
        m: args.m,
        m_max0,
        ef_construction: args.ef_construction,
        ef_search: sweep[0],
        metric,
        max_dimensions: d as u32,
        quantization,
        ..Default::default()
    };
    let mut index = HnswIndex::new(config);
    info!("building index — this may take many minutes for SIFT1M");
    // Chunked `insert_batch` (R858b). The apply-phase backfill bug that
    // collapsed recall at chunked scale was fixed by folding backfilled
    // candidates into the prune selection; 1k chunks match the API's
    // documented safe-batch upper bound.
    const BUILD_CHUNK: usize = 1_000;
    let mut inserted = 0usize;
    while inserted < n_train {
        let end = (inserted + BUILD_CHUNK).min(n_train);
        let chunk: Vec<(u64, Vec<f32>)> = (inserted..end)
            .map(|row_idx| {
                let start = row_idx * d;
                (row_idx as u64, train_flat[start..start + d].to_vec())
            })
            .collect();
        index.insert_batch(chunk);
        inserted = end;
        info!(
            inserted,
            of = n_train,
            elapsed_s = build_start.elapsed().as_secs(),
            "build progress"
        );
    }
    let build_secs = build_start.elapsed().as_secs_f64();
    let rss_kib_after_build = read_rss_kib();
    info!(build_secs, rss_kib = ?rss_kib_after_build, "build complete");

    // ── ef sweep ────────────────────────────────────────────
    let mut rss_kib_peak = rss_kib_after_build;
    let mut points = Vec::new();
    for ef in &sweep {
        index.set_ef_search(*ef);
        let point = sweep_one(&index, &query_flat, &gt_flat, d, gt_k, n_test, args.k)?;
        info!(
            ef_search = ef,
            recall_at_k = point.recall_at_k,
            qps = point.qps,
            p99_us = point.latency_us_p99,
            "sweep point"
        );
        points.push(point);
        // Search scratch buffers grow with ef_search — track the peak
        // so the report carries both build-time and search-time RSS.
        if let Some(rss) = read_rss_kib() {
            rss_kib_peak = Some(rss_kib_peak.map_or(rss, |p| p.max(rss)));
        }
    }

    // ── Write report ────────────────────────────────────────
    let version = env!("CARGO_PKG_VERSION").to_string();
    let mut report = BenchReport::new(
        "vector",
        "ann-benchmarks",
        &args.dataset_name,
        "coordinode",
        &args.codec,
        version,
    )?;
    report.record("hnsw_m", args.m)?;
    report.record("hnsw_ef_construction", args.ef_construction)?;
    report.record("quantization", args.quantization.clone())?;
    // Effective thread count used by the rayon build pool. Reported so
    // downstream comparisons can group/filter (`1` vs `4` runs are not
    // interchangeable for build-time wall-clock, even though search QPS
    // is single-thread). `0` means rayon-default (logical-core count).
    report.record("rayon_threads", args.threads)?;
    report.record("rayon_threads_effective", rayon::current_num_threads())?;
    // Bit-width only meaningful for `rabitq`; record unconditionally so
    // the dashboard can group/filter by it without per-row None handling.
    if args.quantization == "rabitq" {
        report.record("rabitq_bits", args.rabitq_bits)?;
    }
    report.record("build_secs", build_secs)?;
    if let Some(rss) = rss_kib_before_build {
        report.record("rss_kib_before_build", rss)?;
    }
    if let Some(rss) = rss_kib_after_build {
        report.record("rss_kib_after_build", rss)?;
    }
    if let Some(rss) = rss_kib_peak {
        report.record("rss_kib_peak", rss)?;
    }
    report.record("dataset_n_train", n_train)?;
    report.record("dataset_n_test", n_test)?;
    report.record("dataset_dim", d)?;
    report.record("k", args.k)?;
    report.record("sweep", points.clone())?;
    // Top-line: peak recall achieved and corresponding QPS.
    if let Some(best) = points
        .iter()
        .max_by(|a, b| a.recall_at_k.total_cmp(&b.recall_at_k))
    {
        report.record("recall_at_k_peak", best.recall_at_k)?;
        report.record("qps_at_recall_peak", best.qps)?;
    }
    // Recall ≥ 0.95 — canonical ann-benchmarks dashboard cell.
    if let Some(point) = points.iter().find(|p| p.recall_at_k >= 0.95) {
        report.record("qps_at_recall_0_95", point.qps)?;
    }
    // Filename tag groups runs by (M, codec). Bit-width appended only
    // for rabitq runs so the older `<sha>-coordinode-M<m>-<ts>.json`
    // layout stays valid for `none` / `sq8` (no dashboard breakage).
    let tag = match args.quantization.as_str() {
        "rabitq" => format!("M{}-Q{}", args.m, args.rabitq_bits),
        _ => format!("M{}", args.m),
    };
    let path = report.write_json(&args.output, Some(&tag))?;
    info!(path=?path, "report written");
    Ok(())
}

/// Run one ef_search sweep point — REPLAY_ROUNDS rounds over the
/// full query set, aggregate latency percentiles + QPS. Caller
/// pre-sets `ef_search` on `index` via [`HnswIndex::set_ef_search`].
///
/// `query_flat` and `gt_flat` are row-major flat layouts:
/// `query_flat[q_idx * dim..(q_idx + 1) * dim]` is the q-th query,
/// `gt_flat[q_idx * gt_k..(q_idx + 1) * gt_k]` is the q-th
/// ground-truth row.
fn sweep_one(
    index: &HnswIndex,
    query_flat: &[f32],
    gt_flat: &[i32],
    dim: usize,
    gt_k: usize,
    n_test: usize,
    k: usize,
) -> Result<SweepPoint, Box<dyn std::error::Error>> {
    let mut hits = 0u64;
    let mut total = 0u64;
    let mut latencies_us = Vec::with_capacity(n_test * REPLAY_ROUNDS);
    let timer = Instant::now();
    for _round in 0..REPLAY_ROUNDS {
        for q_idx in 0..n_test {
            let q_start = q_idx * dim;
            let query = &query_flat[q_start..q_start + dim];
            let t = Instant::now();
            let results = index.search(query, k);
            let elapsed_us = t.elapsed().as_micros() as f64;
            latencies_us.push(elapsed_us);
            // Recall@k: count overlap between returned ids and
            // ground-truth top-k ids. Ground-truth rows have
            // `gt_k` entries; we only check the first `k` of them.
            let gt_start = q_idx * gt_k;
            let gt: std::collections::HashSet<i64> = gt_flat[gt_start..gt_start + k]
                .iter()
                .map(|x| i64::from(*x))
                .collect();
            for r in results.iter().take(k) {
                if gt.contains(&(r.id as i64)) {
                    hits += 1;
                }
            }
            total += k as u64;
        }
    }
    let wall = timer.elapsed().as_secs_f64();
    latencies_us.sort_by(|a, b| a.total_cmp(b));
    let qps = (n_test as f64 * REPLAY_ROUNDS as f64) / wall;
    let recall = hits as f64 / total as f64;
    let mean = latencies_us.iter().sum::<f64>() / latencies_us.len() as f64;
    let ef_search = index.config().ef_search;
    Ok(SweepPoint {
        ef_search,
        recall_at_k: recall,
        qps,
        latency_us_mean: mean,
        latency_us_p50: percentile(&latencies_us, 0.50),
        latency_us_p95: percentile(&latencies_us, 0.95),
        latency_us_p99: percentile(&latencies_us, 0.99),
    })
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64) * p).floor() as usize;
    sorted[idx.min(sorted.len() - 1)]
}
