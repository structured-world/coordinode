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
//! Per ann-benchmarks methodology: 10 query-replay rounds, warm cache.
//! Search concurrency is the `--search-threads N` axis (default `1` =
//! sequential per-query latency on one core, the ann-benchmarks
//! convention). `N>1` dispatches the query stream across `N` worker
//! threads and reports aggregate throughput, recorded under
//! `search_threads`; `E_core(N) = QPS(N) / (N * QPS(1))` is derived
//! downstream from the ST and MT runs.

#![forbid(unsafe_code)]
#![warn(clippy::unwrap_used, clippy::expect_used)]

mod fvecs;
mod gt;

use crate::gt::{brute_force_gt, cosine_sim, l2_sq};

use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use coordinode_bench::BenchReport;
use coordinode_core::graph::types::VectorMetric;
use coordinode_vector::hnsw::{HnswConfig, HnswIndex, QuantizationCodec, SearchMode};
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
const DEFAULT_EF_SWEEP: &[usize] = &[
    10, 20, 40, 60, 80, 120, 160, 200, 300, 400, 600, 800, 1000, 1200, 1600, 2000,
];

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
    /// This flag only governs the rayon pool used during HNSW graph
    /// construction. To run search concurrently across multiple
    /// workers and measure aggregate throughput / parallel
    /// efficiency, use `--search-threads N` (independent flag).
    #[arg(long, default_value_t = 0)]
    threads: usize,

    /// RobustPrune α parameter for neighbour selection during HNSW
    /// construction. `0` (default) = AUTO: the engine's metric-tuned
    /// default (Cosine = 1.15, others = off). `1.0` forces the
    /// original "take M closest" strategy off. `> 1.0` (Vamana paper
    /// recommends `1.2`) sets RobustPrune explicitly — sparser, more
    /// diverse graph at the cost of higher build time, for lower
    /// fanout per query and higher QPS at fixed recall.
    #[arg(long, default_value_t = 0.0)]
    alpha_pruning: f32,

    /// Rerank strategy on the RaBitQ search path. `inline` (default)
    /// keeps the traditional CoordiNode per-visit exact f32 rerank that
    /// preserves the highest recall but doubles per-visit work.
    /// `end-of-search` follows qdrant Binary Quantization / chroma /
    /// DiskANN: traverse on cheap distances alone, then rerank the
    /// final ef-sized result heap once. `none` skips rerank entirely
    /// — fastest, lowest recall ceiling.
    #[arg(long, default_value = "inline")]
    rerank_mode: String,

    /// Oversampling factor for `--rerank-mode end-of-search` (qdrant
    /// `oversampling` equivalent). Search traverses with
    /// `frontier_ef = ceil(ef * factor)` candidates, then the exact f32
    /// rerank picks the best `ef`. `1.0` (default) collapses to
    /// "ef in, ef out". `2.0` doubles the frontier; recall climbs back
    /// toward inline-rerank parity at a modest QPS cost.
    ///
    /// Ignored when `--rerank-mode` is not `end-of-search`.
    #[arg(long, default_value_t = 1.0)]
    rerank_oversample: f32,

    /// Number of OS threads to drive concurrent search calls during
    /// the ef sweep. `1` (default) preserves the original sequential
    /// ann-benchmarks methodology (per-query latency on a single
    /// core). `N>1` rebuilds the inner search loop with a local
    /// rayon pool and dispatches queries across `N` workers; the
    /// reported QPS is the aggregate throughput, NOT per-thread.
    /// Used to derive parallel efficiency
    ///   `E_core(N) = QPS(search_threads=N) / (N * QPS(search_threads=1))`
    /// against competitors with the same configuration.
    ///
    /// Independent from `--threads` which governs the HNSW BUILD
    /// rayon pool. Typically set both to the same N for a clean
    /// fixed-budget comparison.
    #[arg(long, default_value_t = 1)]
    search_threads: usize,

    /// If > 0, take only the first N base vectors from the training
    /// set and recompute ground-truth via brute-force kNN against
    /// that subset. The effective dataset name in the report gets
    /// `-Nk` injected before the `-euclidean` / `-angular` suffix
    /// (e.g. `sift-128-euclidean` -> `sift-128-100k-euclidean`) so
    /// the dashboard keeps subset runs separate from full-dataset
    /// runs. Use 100_000 on per-commit CI to keep wall-clock under
    /// ~20 minutes; competitors must run with the same N for
    /// apples-to-apples comparison. `0` (default) = full dataset.
    #[arg(long, default_value_t = 0)]
    subset_size: usize,

    /// Search strategy. `hnsw` (default) walks the index graph using
    /// the configured `ef_search`. `exact` performs a brute-force
    /// linear scan over every indexed vector and returns recall=1.0
    /// top-k. Use `exact` to publish a r=1.0 baseline alongside the
    /// approximate sweep for comparison against qdrant exact mode.
    /// In exact mode the ef sweep collapses to a single placeholder
    /// point because `ef_search` has no effect on brute force.
    #[arg(long, value_enum, default_value_t = SearchModeArg::Hnsw)]
    search_mode: SearchModeArg,

    /// Number of independent HNSW shards to build over the dataset.
    /// `1` (default) builds a single index (today's shape). `N>1`
    /// partitions the train vectors across shards (see
    /// `--shard-routing`); at search time the bench scatters every
    /// query to the routed shard subset and K-way merges the top-K.
    /// The reported QPS is the aggregate across shards, which drives
    /// the per-node parallel-efficiency metric
    ///   `E_node(K) = QPS(n_shards=K) / (K * QPS(n_shards=1))`.
    /// Recorded as `n_shards` in the output JSON.
    #[arg(long, default_value_t = 1)]
    n_shards: usize,

    /// Shard partitioning + query routing strategy when `--n-shards > 1`.
    ///
    /// `modulo` (default): vectors land on shard `id % N`; partitions
    /// are random w.r.t. geometry so every query MUST scatter to all
    /// shards. Measures the worst-case fan-out floor.
    ///
    /// `centroid`: shards are k-means clusters (deterministic seeded
    /// Lloyd iterations on a sample); each vector lands on its nearest
    /// centroid's shard, and a query routes only to the `--route-top-m`
    /// shards whose centroids are closest. Models the IVF-style routed
    /// fan-out where per-query work does not grow with shard count.
    #[arg(long, value_enum, default_value_t = ShardRoutingArg::Modulo)]
    shard_routing: ShardRoutingArg,

    /// Number of closest-centroid shards a query is routed to when
    /// `--shard-routing centroid`. Clamped to `n_shards`. `1` probes
    /// the maximum-savings / lowest-recall point; `n_shards` collapses
    /// to scatter-all (same coverage as `modulo`, different partition
    /// geometry). Ignored for `modulo` routing.
    #[arg(long, default_value_t = 1)]
    route_top_m: usize,

    /// SPANN-style closure replication for `--shard-routing centroid`. A
    /// vector is also written to every centroid whose routing distance
    /// (squared-L2 or cosine) is within `replication_eps` x its nearest
    /// distance, capped at `--max-replicas`. `1.0` (default) replicates only
    /// exact ties (single assignment); `>1.0` duplicates boundary points so a
    /// low `route_top_m` still finds cross-boundary neighbours. Modulo-ignored.
    #[arg(long, default_value_t = 1.0)]
    replication_eps: f32,

    /// Cap on closure replicas per vector (SPANN bounds this at ~8 — beyond
    /// that recall gains vanish while index size keeps growing). `1` disables
    /// replication regardless of `replication_eps`.
    #[arg(long, default_value_t = 1)]
    max_replicas: usize,

    /// Query-adaptive fan-out for `--shard-routing centroid`. A query routes to
    /// every centroid within `route_eps` x its nearest-centroid distance,
    /// capped at `--route-top-m`. `1.0` (default) probes only the nearest (or
    /// exact ties), so interior queries touch one shard and only boundary
    /// queries fan out — cheaper than a fixed `route_top_m` for every query.
    #[arg(long, default_value_t = 1.0)]
    route_eps: f32,

    /// Pyramid meta-index size for `--shard-routing pyramid`: a labeled data
    /// subsample of this many points is the routing index instead of the
    /// `n_shards` centroids. Captures boundary geometry the coarse centroids
    /// miss at high shard counts.
    #[arg(long, default_value_t = 4000)]
    meta_sample: usize,

    /// Pyramid: gather shard labels from this many nearest meta-points per
    /// query (then dedup + cap at `--route-top-m`). Larger = wider coverage.
    #[arg(long, default_value_t = 64)]
    meta_k: usize,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum ShardRoutingArg {
    Modulo,
    Centroid,
    Pyramid,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum SearchModeArg {
    Hnsw,
    Exact,
}

impl From<SearchModeArg> for SearchMode {
    fn from(a: SearchModeArg) -> Self {
        match a {
            SearchModeArg::Hnsw => SearchMode::Hnsw,
            SearchModeArg::Exact => SearchMode::Exact,
        }
    }
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

    let search_mode = SearchMode::from(args.search_mode);
    // In exact mode `ef_search` is inert (brute-force ignores the
    // graph entirely), so collapse the sweep to a single placeholder
    // point. The ef value in the JSON is left at 0 to signal "n/a".
    let sweep: Vec<usize> = if matches!(search_mode, SearchMode::Exact) {
        vec![0]
    } else {
        match &args.ef_sweep {
            Some(s) => s
                .split(',')
                .map(|x| x.trim().parse::<usize>())
                .collect::<Result<Vec<_>, _>>()?,
            None => DEFAULT_EF_SWEEP.to_vec(),
        }
    };

    // ── Load dataset ────────────────────────────────────────
    info!(train=?args.train, "loading training set");
    let (train_dim, mut train_flat) = read_fvecs(&args.train)?;
    info!(query=?args.query, "loading query set");
    let (query_dim, query_flat) = read_fvecs(&args.query)?;
    info!(gt=?args.groundtruth, "loading ground-truth");
    let (mut gt_k, mut gt_flat) = read_ivecs(&args.groundtruth)?;

    if train_dim != query_dim {
        return Err(format!(
            "train dim ({train_dim}) != query dim ({query_dim}) — wrong dataset pair?"
        )
        .into());
    }
    let d = train_dim;
    let mut n_train = train_flat.len() / d;
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
    let metric_suffix = if args.dataset_name.ends_with("-euclidean") {
        "-euclidean"
    } else if args.dataset_name.ends_with("-angular") {
        "-angular"
    } else {
        return Err(format!(
            "cannot detect metric from dataset name {:?} (expected suffix `-euclidean` or `-angular`)",
            args.dataset_name
        )
        .into());
    };
    let metric = match metric_suffix {
        "-euclidean" => VectorMetric::L2,
        _ => VectorMetric::Cosine,
    };
    info!(?metric, "metric inferred from dataset name");

    // ── Optional subsampling ────────────────────────────────
    // For per-commit CI: shrink the base set to the first N vectors
    // and recompute ground-truth against that subset. Wall-clock
    // drops ~10× without changing the relative ranking against
    // competitors (which must run the same N). Brute-force kNN
    // for gt recompute is parallel over queries; on 100k base it
    // takes a few seconds at most.
    let dataset_name = if args.subset_size > 0 && args.subset_size < n_train {
        info!(
            full_n = n_train,
            subset_n = args.subset_size,
            "subsampling training set + recomputing groundtruth"
        );
        train_flat.truncate(args.subset_size * d);
        n_train = args.subset_size;
        let new_gt_k = args.k.max(10);
        let gt_recompute_start = Instant::now();
        gt_flat = brute_force_gt(&train_flat, &query_flat, d, new_gt_k, metric);
        gt_k = new_gt_k;
        info!(
            new_gt_k,
            gt_recompute_secs = gt_recompute_start.elapsed().as_secs_f64(),
            "groundtruth recomputed for subset"
        );
        let stem = args
            .dataset_name
            .strip_suffix(metric_suffix)
            .unwrap_or(args.dataset_name.as_str());
        let k_thousand = args.subset_size / 1_000;
        format!("{stem}-{k_thousand}k{metric_suffix}")
    } else {
        args.dataset_name.clone()
    };

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
        alpha_pruning: args.alpha_pruning,
        rerank_oversample_factor: args.rerank_oversample,
        rerank_mode: match args.rerank_mode.as_str() {
            "inline" => coordinode_vector::hnsw::RerankMode::Inline,
            "end-of-search" => coordinode_vector::hnsw::RerankMode::EndOfSearch,
            "none" => coordinode_vector::hnsw::RerankMode::None,
            other => {
                return Err(format!(
                    "--rerank-mode: expected `inline`, `end-of-search`, or `none`, got `{other}`"
                )
                .into())
            }
        },
        ..Default::default()
    };
    // Build the shard fleet. `n_shards=1` collapses to a single index
    // (today's shape, bit-identical control path). `n_shards>1`
    // partitions the train set by `id % N` so each shard owns roughly
    // `n_train/N` rows. Node ids stay in `[0, n_train)` across the
    // fleet, which means the groundtruth ivecs (id-addressed) still
    // map onto exactly one shard per id and recall is preserved when
    // the bench scatters queries across shards and merges top-K.
    let n_shards = args.n_shards.max(1);
    let mut shards: Vec<HnswIndex> = (0..n_shards)
        .map(|_| HnswIndex::new(config.clone()))
        .collect();
    // Per-vector shard assignment. `modulo` spreads ids round-robin —
    // geometry-blind, every query must scatter to all shards (worst-case
    // fan-out floor). `centroid` clusters the space with deterministic
    // k-means so a query can route to the top-m closest shards only —
    // the IVF-style fan-out the distributed design uses.
    // Centroid partition shared by `centroid` and `pyramid` routing: k-means
    // clusters the space so each shard owns a region. `modulo` skips this.
    let centroids: Option<Vec<Vec<f32>>> = if n_shards > 1
        && matches!(
            args.shard_routing,
            ShardRoutingArg::Centroid | ShardRoutingArg::Pyramid
        ) {
        Some(train_centroids(&train_flat, d, n_train, n_shards, metric))
    } else {
        None
    };
    let assignment: Vec<Vec<usize>> = match &centroids {
        Some(c) => (0..n_train)
            .map(|i| {
                closure_assign(
                    &train_flat[i * d..(i + 1) * d],
                    c,
                    metric,
                    args.replication_eps,
                    args.max_replicas,
                )
            })
            .collect(),
        None => (0..n_train).map(|i| vec![i % n_shards]).collect(),
    };
    let total_assigned: usize = assignment.iter().map(Vec::len).sum();
    info!(
        n_train,
        total_assigned,
        replication_factor = total_assigned as f64 / n_train as f64,
        "closure assignment complete"
    );
    // Query router. `centroid`: route on the k-means centroids (one route
    // point per shard). `pyramid`: route on a labeled subsample (meta-index) —
    // finer than centroids, capturing boundary geometry coarse centroids miss
    // at high shard counts (Pyramid meta-routing). Both share one router type;
    // only the route-point set + labels + probe width differ.
    let router: Option<CentroidRouter> = match (&centroids, args.shard_routing) {
        (Some(c), ShardRoutingArg::Centroid) => Some(CentroidRouter {
            route_points: c.clone(),
            route_labels: (0..n_shards).collect(),
            n_probe: n_shards,
            top_m: args.route_top_m.clamp(1, n_shards),
            route_eps: args.route_eps,
            metric,
        }),
        (Some(c), ShardRoutingArg::Pyramid) => {
            let m = args.meta_sample.clamp(n_shards, n_train);
            let stride = (n_train / m).max(1);
            let mut route_points = Vec::with_capacity(m);
            let mut route_labels = Vec::with_capacity(m);
            for i in (0..n_train).step_by(stride).take(m) {
                let v = &train_flat[i * d..(i + 1) * d];
                route_points.push(v.to_vec());
                route_labels.push(nearest_centroid(v, c, metric));
            }
            info!(meta_points = route_points.len(), "pyramid meta-index built");
            Some(CentroidRouter {
                route_points,
                route_labels,
                n_probe: args.meta_k.clamp(1, m),
                top_m: args.route_top_m.clamp(1, n_shards),
                route_eps: args.route_eps,
                metric,
            })
        }
        _ => None,
    };
    info!(n_shards, routing = ?args.shard_routing, "building shard fleet");
    // Chunked `insert_batch` (R858b). The apply-phase backfill bug that
    // collapsed recall at chunked scale was fixed by folding backfilled
    // candidates into the prune selection; 1k chunks match the API's
    // documented safe-batch upper bound.
    const BUILD_CHUNK: usize = 1_000;
    for (shard_id, shard) in shards.iter_mut().enumerate() {
        let mut buf: Vec<(u64, Vec<f32>)> = Vec::with_capacity(BUILD_CHUNK);
        let mut shard_inserted = 0usize;
        for row_idx in (0..n_train).filter(|i| assignment[*i].contains(&shard_id)) {
            let start = row_idx * d;
            buf.push((row_idx as u64, train_flat[start..start + d].to_vec()));
            if buf.len() == BUILD_CHUNK {
                shard_inserted += buf.len();
                shard.insert_batch(std::mem::take(&mut buf));
                info!(
                    shard_id,
                    inserted = shard_inserted,
                    elapsed_s = build_start.elapsed().as_secs(),
                    "shard build progress"
                );
            }
        }
        if !buf.is_empty() {
            shard_inserted += buf.len();
            shard.insert_batch(buf);
        }
        info!(
            shard_id,
            inserted = shard_inserted,
            elapsed_s = build_start.elapsed().as_secs(),
            "shard build complete"
        );
    }
    let build_secs = build_start.elapsed().as_secs_f64();
    let rss_kib_after_build = read_rss_kib();
    info!(build_secs, rss_kib = ?rss_kib_after_build, "build complete");

    // ── ef sweep ────────────────────────────────────────────
    let mut rss_kib_peak = rss_kib_after_build;
    let mut points = Vec::new();
    for ef in &sweep {
        if !matches!(search_mode, SearchMode::Exact) {
            for shard in &mut shards {
                shard.set_ef_search(*ef);
            }
        }
        let point = sweep_one(
            &shards,
            router.as_ref(),
            &query_flat,
            &gt_flat,
            d,
            gt_k,
            n_test,
            args.k,
            args.search_threads,
            search_mode,
        )?;
        info!(
            ef_search = ef,
            recall_at_k = point.recall_at_k,
            qps = point.qps,
            p99_us = point.latency_us_p99,
            "sweep point"
        );
        // Diagnostic: distance-kernel calls per query for this ef point
        // (single-thread sweeps only: the counter is thread-local and
        // the ST path runs every query on this thread). The counter
        // still includes build-phase calls on the FIRST ef point; read
        // the second and later points for clean per-query numbers.
        #[cfg(feature = "dist-counters")]
        if args.search_threads <= 1 {
            let calls = coordinode_vector::metrics::take_dist_calls();
            let total_q = (n_test * REPLAY_ROUNDS) as u64;
            info!(
                ef_search = ef,
                dist_calls_per_query = calls / total_q.max(1),
                "distance-kernel call volume"
            );
        }
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
        &dataset_name,
        "coordinode",
        &args.codec,
        version,
    )?;
    // Record subset_size so the dashboard can distinguish per-commit
    // smoke runs (100k) from periodic full-dataset runs (0 / 1M+)
    // even within the same dataset family.
    report.record("subset_size", args.subset_size)?;
    report.record("hnsw_m", args.m)?;
    report.record("hnsw_ef_construction", args.ef_construction)?;
    // RobustPrune α; recorded so the dashboard can filter / group runs
    // built with α=1.0 (legacy "take M closest") vs α>1.0 (Vamana-style
    // diverse graph). Records the RESOLVED value (auto -> metric
    // default), not the raw CLI arg, so cells stay comparable.
    report.record("hnsw_alpha_pruning", config.effective_alpha() as f64)?;
    report.record("hnsw_rerank_mode", args.rerank_mode.clone())?;
    report.record("hnsw_rerank_oversample", args.rerank_oversample as f64)?;
    report.record("quantization", args.quantization.clone())?;
    // Effective thread count used by the rayon BUILD pool. `0` means
    // rayon-default (logical-core count).
    report.record("rayon_threads", args.threads)?;
    report.record("rayon_threads_effective", rayon::current_num_threads())?;
    // Threads driving the SEARCH path (independent from build pool).
    // `1` = sequential per-query timing (ann-benchmarks convention).
    // `N>1` = aggregate throughput across N concurrent workers; pair
    // with the same N in competitor runs to derive E_core(N).
    report.record("search_threads", args.search_threads)?;
    // Independent shards built over partitioned data; the bench scatters
    // every query and K-way merges top-K so QPS is aggregate across the
    // fleet. `1` collapses to the single-index control case.
    report.record("n_shards", n_shards)?;
    // Partitioning + routing strategy: `modulo` (geometry-blind,
    // scatter-all) or `centroid` (k-means partitions, top-m routed
    // fan-out). route_top_m only meaningful for centroid.
    report.record(
        "shard_routing",
        match args.shard_routing {
            ShardRoutingArg::Modulo => "modulo",
            ShardRoutingArg::Centroid => "centroid",
            ShardRoutingArg::Pyramid => "pyramid",
        },
    )?;
    if matches!(
        args.shard_routing,
        ShardRoutingArg::Centroid | ShardRoutingArg::Pyramid
    ) {
        report.record("route_top_m", args.route_top_m.clamp(1, n_shards))?;
    }
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
    // Filename tag groups runs by (M, codec, search-mode). Bit-width
    // appended only for rabitq runs so the older
    // `<sha>-coordinode-M<m>-<ts>.json` layout stays valid for `none`
    // / `sq8` (no dashboard breakage). Exact mode gets an `-X` suffix
    // so brute-force baselines never overwrite HNSW runs with the
    // same M and codec at the same timestamp.
    let tag = match args.quantization.as_str() {
        "rabitq" => format!("M{}-Q{}", args.m, args.rabitq_bits),
        _ => format!("M{}", args.m),
    };
    let tag = match args.search_mode {
        SearchModeArg::Exact => format!("{tag}-X"),
        SearchModeArg::Hnsw => tag,
    };
    // Centroid-routed shard runs get a `-Cm` suffix (m = fan-out width)
    // so they never overwrite scatter-all cells with the same M/codec.
    let tag = if n_shards > 1
        && matches!(
            args.shard_routing,
            ShardRoutingArg::Centroid | ShardRoutingArg::Pyramid
        ) {
        format!("{tag}-C{}", args.route_top_m.clamp(1, n_shards))
    } else {
        tag
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
#[allow(clippy::too_many_arguments)]
fn sweep_one(
    shards: &[HnswIndex],
    router: Option<&CentroidRouter>,
    query_flat: &[f32],
    gt_flat: &[i32],
    dim: usize,
    gt_k: usize,
    n_test: usize,
    k: usize,
    search_threads: usize,
    search_mode: SearchMode,
) -> Result<SweepPoint, Box<dyn std::error::Error>> {
    let total_queries = n_test * REPLAY_ROUNDS;
    let timer = Instant::now();

    let (mut latencies_us, hits, total) = if search_threads <= 1 {
        // Sequential single-thread path. Preserves the original
        // ann-benchmarks per-query-latency-on-one-core convention.
        let mut latencies_us = Vec::with_capacity(total_queries);
        let mut hits = 0u64;
        let mut total = 0u64;
        for _round in 0..REPLAY_ROUNDS {
            for q_idx in 0..n_test {
                let q_start = q_idx * dim;
                let query = &query_flat[q_start..q_start + dim];
                let t = Instant::now();
                let results = multi_search(shards, router, query, k, search_mode);
                let elapsed_us = t.elapsed().as_micros() as f64;
                latencies_us.push(elapsed_us);
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
        (latencies_us, hits, total)
    } else {
        // Multi-thread path: N explicit worker threads each pull a
        // contiguous slice of the query stream and walk it in a tight
        // loop. The previous rayon `par_iter` shape spawned a fresh
        // task per query (100k tasks per ef point at the standard
        // 10k queries x 10 replay rounds) and the work-stealing tax
        // showed up as a 24% sift MT4 gap vs hnswlib's straight
        // worker model. With chunked workers the per-query overhead
        // is one branch + one slice index, and the engine path is
        // unchanged. E_core(N) = QPS(N) / (N * QPS(1)) is still the
        // scaling figure, computed off the wall-clock around the
        // join boundary.
        let chunk = total_queries.div_ceil(search_threads);
        let workers: Vec<_> = std::thread::scope(|scope| {
            // Spawn ALL workers first, THEN join. A naive
            // `(0..N).map(spawn).map(join).collect()` chain runs the
            // workers serially because the iterator is lazy: each
            // `spawn` immediately threads through the next `join`
            // before the following item is requested. Materialise the
            // handle vector to force concurrent kick-off.
            let handles: Vec<_> = (0..search_threads)
                .map(|w| {
                    let start = w * chunk;
                    let end = ((w + 1) * chunk).min(total_queries);
                    scope.spawn(move || {
                        let mut lat = Vec::with_capacity(end.saturating_sub(start));
                        let mut hits: u64 = 0;
                        let mut total: u64 = 0;
                        for i in start..end {
                            let q_idx = i % n_test;
                            let q_start = q_idx * dim;
                            let query = &query_flat[q_start..q_start + dim];
                            let t = Instant::now();
                            let results = multi_search(shards, router, query, k, search_mode);
                            lat.push(t.elapsed().as_micros() as f64);
                            let gt_start = q_idx * gt_k;
                            let gt: std::collections::HashSet<i64> = gt_flat
                                [gt_start..gt_start + k]
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
                        (lat, hits, total)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap_or_else(|_| (Vec::new(), 0u64, 0u64)))
                .collect()
        });
        let cap: usize = workers.iter().map(|(l, _, _)| l.len()).sum();
        let mut latencies_us = Vec::with_capacity(cap);
        let mut hits = 0u64;
        let mut total = 0u64;
        for (mut l, h, t) in workers {
            latencies_us.append(&mut l);
            hits += h;
            total += t;
        }
        (latencies_us, hits, total)
    };

    let wall = timer.elapsed().as_secs_f64();
    latencies_us.sort_by(|a, b| a.total_cmp(b));
    let qps = (n_test as f64 * REPLAY_ROUNDS as f64) / wall;
    let recall = hits as f64 / total as f64;
    let mean = latencies_us.iter().sum::<f64>() / latencies_us.len() as f64;
    let ef_search = shards[0].config().ef_search;
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

/// Scatter the query to every shard IN PARALLEL, collect each shard's
/// top-K results, merge into a single global top-K by ascending score.
///
/// `n_shards=1` collapses to a direct call into the single index (no
/// extra allocation, no merge, no rayon overhead), so the existing
/// single-index push cells stay bit-identical with the legacy harness
/// shape. For `n_shards>1` the bench cheats by holding the indices
/// directly (no IPC, no routing) so the measurement isolates the
/// fan-out + merge cost from network or proposal overhead.
///
/// Scatter uses `rayon::par_iter` so each shard's search runs on a
/// separate worker; this is the apples-to-apples model for multi-host
/// sharding where each shard's search hits its own host concurrently.
/// Sequential scatter would just double per-query work and report a
/// meaningless E_node(K) approx 1/K (measured at dfcee7d: E_node ~= 0.27
/// with the legacy sequential loop).
///
/// Lower `score` is "better" across every supported metric (cosine
/// distance, L2-squared, negated dot product), so the merge sorts
/// ascending and takes the first `k`.
fn multi_search(
    shards: &[HnswIndex],
    router: Option<&CentroidRouter>,
    query: &[f32],
    k: usize,
    mode: SearchMode,
) -> Vec<coordinode_vector::hnsw::SearchResult> {
    use rayon::prelude::*;
    if shards.len() == 1 {
        return shards[0].search_with_mode(query, k, mode);
    }
    // Centroid routing prunes the fan-out to the top-m closest shards.
    // Modulo partitions are geometry-blind, so no router = scatter-all.
    let routed: Vec<usize> = match router {
        Some(r) => r.route(query),
        None => (0..shards.len()).collect(),
    };
    if routed.len() == 1 {
        return shards[routed[0]].search_with_mode(query, k, mode);
    }
    let mut merged: Vec<coordinode_vector::hnsw::SearchResult> = routed
        .par_iter()
        .flat_map_iter(|&s| shards[s].search_with_mode(query, k, mode))
        .collect();
    merged.sort_by(|a, b| a.score.total_cmp(&b.score));
    // Closure replication (and overlapping routed shards) can place a vector
    // in several probed shards; keep only its best-scoring occurrence so the
    // top-K is K distinct ids.
    let mut seen = std::collections::HashSet::with_capacity(merged.len());
    merged.retain(|r| seen.insert(r.id));
    merged.truncate(k);
    merged
}

/// Centroid-based shard router: k-means cluster centres plus the
/// per-query fan-out width. Mirrors the IVF centroid index from the
/// distributed design — routing cost is `n_shards` distance computations
/// on the coordinator, after which only `top_m` shards do real work.
struct CentroidRouter {
    /// Route points: the k-means centroids (centroid routing, one per shard)
    /// or a labeled data subsample (pyramid meta-index, many per shard).
    route_points: Vec<Vec<f32>>,
    /// Shard id of each route point. Identity (`j -> j`) for centroid routing;
    /// the point's nearest-centroid shard for pyramid routing.
    route_labels: Vec<usize>,
    /// How many nearest route points to scan for shard labels: `n_shards` for
    /// centroid routing (all of them), `meta_k` for pyramid.
    n_probe: usize,
    top_m: usize,
    route_eps: f32,
    metric: VectorMetric,
}

impl CentroidRouter {
    /// Distinct shard ids the query routes to. Scan the `n_probe` nearest route
    /// points best-first; keep the nearest shard unconditionally, then any
    /// not-yet-seen shard whose route point is within `route_eps` x the nearest
    /// route-point distance, capped at `top_m`. Centroid points + `route_eps=1`
    /// = the nearest shard only; `route_eps > 1` makes the fan-out
    /// query-adaptive; a labeled subsample makes it pyramid meta-routing.
    fn route(&self, query: &[f32]) -> Vec<usize> {
        let mut scored: Vec<(usize, f32)> = self
            .route_points
            .iter()
            .enumerate()
            .map(|(i, p)| (i, centroid_distance(query, p, self.metric)))
            .collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        let thresh = scored[0].1 * self.route_eps.max(1.0);
        let mut out: Vec<usize> = Vec::with_capacity(self.top_m);
        for (i, dist) in scored.into_iter().take(self.n_probe) {
            if out.len() >= self.top_m {
                break;
            }
            let shard = self.route_labels[i];
            if out.contains(&shard) {
                continue;
            }
            if out.is_empty() || dist <= thresh {
                out.push(shard);
            }
        }
        out
    }
}

/// Distance used for centroid training / assignment / routing. Cosine
/// datasets compare by cosine distance, everything else by squared L2
/// (monotonic in L2, cheaper).
fn centroid_distance(v: &[f32], c: &[f32], metric: VectorMetric) -> f32 {
    match metric {
        VectorMetric::Cosine => 1.0 - cosine_sim(v, c),
        _ => l2_sq(v, c),
    }
}

/// SPANN-style closure assignment: the nearest centroid plus every centroid
/// within `eps` x the nearest routing distance, capped at `max_replicas`.
/// Always returns at least the nearest shard. `eps <= 1.0` or
/// `max_replicas <= 1` yields a single shard (no replication). Replicating
/// only boundary points (those with several near-equidistant centroids) lets a
/// low `route_top_m` recover the cross-boundary neighbours a single probe
/// misses (SPANN, NeurIPS 2021).
fn closure_assign(
    v: &[f32],
    centroids: &[Vec<f32>],
    metric: VectorMetric,
    eps: f32,
    max_replicas: usize,
) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = centroids
        .iter()
        .enumerate()
        .map(|(j, c)| (j, centroid_distance(v, c, metric)))
        .collect();
    scored.sort_by(|a, b| a.1.total_cmp(&b.1));
    let thresh = scored[0].1 * eps.max(1.0);
    let cap = max_replicas.max(1);
    let mut out = Vec::with_capacity(cap);
    for (j, dist) in scored {
        if out.is_empty() {
            out.push(j); // always keep the nearest centroid
        } else if out.len() < cap && dist <= thresh {
            out.push(j);
        } else {
            break;
        }
    }
    out
}

/// Index of the centroid closest to `v`.
fn nearest_centroid(v: &[f32], centroids: &[Vec<f32>], metric: VectorMetric) -> usize {
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for (j, c) in centroids.iter().enumerate() {
        let dist = centroid_distance(v, c, metric);
        if dist < best_d {
            best_d = dist;
            best = j;
        }
    }
    best
}

/// Deterministic k-means (Lloyd) over an evenly-strided subsample.
/// Init = evenly spaced sample points (deterministic, no RNG), 10
/// iterations — plenty for k in the single digits. Empty clusters keep
/// their previous centre, which can only happen with duplicate-heavy
/// data and is harmless for routing purposes.
fn train_centroids(
    train_flat: &[f32],
    d: usize,
    n_train: usize,
    k: usize,
    metric: VectorMetric,
) -> Vec<Vec<f32>> {
    const SAMPLE_CAP: usize = 20_000;
    const LLOYD_ITERS: usize = 10;
    let sample_n = n_train.min(SAMPLE_CAP);
    let stride = (n_train / sample_n).max(1);
    let sample: Vec<&[f32]> = (0..n_train)
        .step_by(stride)
        .take(sample_n)
        .map(|i| &train_flat[i * d..(i + 1) * d])
        .collect();
    let mut centroids: Vec<Vec<f32>> = (0..k)
        .map(|j| sample[j * sample.len() / k].to_vec())
        .collect();
    for _ in 0..LLOYD_ITERS {
        let mut sums = vec![vec![0f32; d]; k];
        let mut counts = vec![0usize; k];
        for v in &sample {
            let cl = nearest_centroid(v, &centroids, metric);
            counts[cl] += 1;
            for (s, x) in sums[cl].iter_mut().zip(v.iter()) {
                *s += x;
            }
        }
        for j in 0..k {
            if counts[j] > 0 {
                for (cv, s) in centroids[j].iter_mut().zip(sums[j].iter()) {
                    *cv = s / counts[j] as f32;
                }
            }
        }
    }
    centroids
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64) * p).floor() as usize;
    sorted[idx.min(sorted.len() - 1)]
}
