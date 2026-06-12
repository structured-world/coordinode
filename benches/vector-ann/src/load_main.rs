//! `bench-vector-load` — dataset loader for the server-mode bench.
//!
//! Bulk-loads an fvecs training set into a RUNNING coordinode server
//! through the public gRPC client: batched `UNWIND $batch CREATE`
//! writes, then `CREATE VECTOR INDEX` DDL so subsequent
//! `bench-vector-grpc` searches go through HNSW instead of the
//! brute-force scan.
//!
//! Every node carries an integer `ext_id` property holding the fvecs
//! row index — the search bench maps results back to groundtruth rows
//! through it (server-side node ids are allocator-assigned).
//!
//! The reported wall-clock doubles as a write-path throughput note:
//! on a single node it prices the gRPC+Cypher+engine insert path; on a
//! 3-node cluster the same load prices raft replication.

#![forbid(unsafe_code)]
#![warn(clippy::unwrap_used, clippy::expect_used)]

// The shared fvecs module also exports `read_ivecs`, which only the
// search bins use.
#[allow(dead_code)]
mod fvecs;
#[allow(dead_code)]
mod gt;

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use coordinode_client::{CoordinodeClient, Value};
use tracing::info;

use crate::fvecs::read_fvecs;

#[derive(Parser, Debug)]
#[command(name = "bench-vector-load", version)]
struct Args {
    /// gRPC endpoint of a running coordinode server.
    #[arg(long)]
    endpoint: String,

    /// Path to the training `.fvecs` file.
    #[arg(long)]
    train: PathBuf,

    /// Label to create the nodes under.
    #[arg(long, default_value = "Bench")]
    label: String,

    /// Property name for the embedding.
    #[arg(long, default_value = "embedding")]
    property: String,

    /// Take only the first N train vectors (0 = all). Must match the
    /// `--subset-size` used for the in-process baseline cells.
    #[arg(long, default_value_t = 0)]
    subset_size: usize,

    /// Vectors per UNWIND batch. Large batches amortise the per-RPC
    /// cost; the cap keeps single requests under transport limits
    /// (d=128 f32 -> ~512 B/vector -> 500/batch ~ 256 KB).
    #[arg(long, default_value_t = 500)]
    batch_size: usize,

    /// HNSW M for the index DDL.
    #[arg(long, default_value_t = 16)]
    m: usize,

    /// HNSW ef_construction for the index DDL.
    #[arg(long, default_value_t = 200)]
    ef_construction: usize,

    /// Distance metric for the index DDL: "euclidean" or "cosine".
    #[arg(long, default_value = "euclidean")]
    metric: String,

    /// Skip the CREATE VECTOR INDEX DDL (loads data only).
    #[arg(long, default_value_t = false)]
    no_index: bool,

    /// Optional query `.fvecs` path. Together with
    /// `--write-groundtruth`, recomputes exact k-NN against the
    /// (possibly subset) train set and writes it as `.ivecs`. The
    /// published full-dataset groundtruth is INVALID for a subset (it
    /// references neighbours outside the loaded rows); the search
    /// bench needs this recomputed file to measure recall correctly.
    #[arg(long)]
    query: Option<PathBuf>,

    /// Output path for the recomputed subset groundtruth `.ivecs`.
    #[arg(long)]
    write_groundtruth: Option<PathBuf>,

    /// k for the recomputed groundtruth rows.
    #[arg(long, default_value_t = 10)]
    gt_k: usize,

    /// Shard fan-out: with `--n-shards N --shard-idx I`, load only the
    /// rows where `row % N == I`, keeping the GLOBAL row index as
    /// `ext_id` so a scatter-gather search merged across all shards
    /// scores against the full-dataset groundtruth. The groundtruth
    /// recompute (`--write-groundtruth`) always covers the FULL
    /// (subset) train set regardless of the shard filter.
    #[arg(long, default_value_t = 1)]
    n_shards: usize,

    /// This loader's shard index in `0..n_shards`.
    #[arg(long, default_value_t = 0)]
    shard_idx: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let args = Args::parse();
    info!(?args, "starting vector loader");

    let (dim, mut train_flat) = read_fvecs(&args.train)?;
    let mut n_train = train_flat.len() / dim;
    if args.subset_size > 0 && args.subset_size < n_train {
        train_flat.truncate(args.subset_size * dim);
        n_train = args.subset_size;
    }
    info!(n_train, dim, "training set ready");

    // Recompute groundtruth against the (possibly subset) train set.
    // Runs BEFORE the load so a gt-only invocation works without a
    // server, and a load failure doesn't waste the brute-force pass.
    if let (Some(query_path), Some(gt_path)) = (&args.query, &args.write_groundtruth) {
        let (q_dim, query_flat) = read_fvecs(query_path)?;
        if q_dim != dim {
            return Err(format!("query dim ({q_dim}) != train dim ({dim})").into());
        }
        let metric = match args.metric.as_str() {
            "cosine" => coordinode_core::graph::types::VectorMetric::Cosine,
            _ => coordinode_core::graph::types::VectorMetric::L2,
        };
        let t = Instant::now();
        let gt = crate::gt::brute_force_gt(&train_flat, &query_flat, dim, args.gt_k, metric);
        crate::fvecs::write_ivecs(gt_path, args.gt_k, &gt)?;
        info!(
            path = ?gt_path,
            gt_k = args.gt_k,
            secs = t.elapsed().as_secs_f64(),
            "subset groundtruth written"
        );
    }

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(load(&args, &train_flat, dim, n_train))
}

async fn load(
    args: &Args,
    train_flat: &[f32],
    dim: usize,
    n_train: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut client = CoordinodeClient::connect(args.endpoint.clone()).await?;

    let insert_cypher = format!(
        "UNWIND $batch AS row \
         CREATE (n:{label} {{ext_id: row.ext_id, {prop}: row.emb}})",
        label = args.label,
        prop = args.property,
    );

    if args.shard_idx >= args.n_shards.max(1) {
        return Err(format!(
            "--shard-idx {} out of range for --n-shards {}",
            args.shard_idx, args.n_shards
        )
        .into());
    }
    let wall = Instant::now();
    let mut sent = 0usize;
    let mut loaded = 0usize;
    while sent < n_train {
        let end = (sent + args.batch_size).min(n_train);
        let batch: Vec<Value> = (sent..end)
            // Modulo shard filter; ext_id stays the GLOBAL row index
            // so merged scatter-gather results match the groundtruth.
            .filter(|i| args.n_shards <= 1 || i % args.n_shards == args.shard_idx)
            .map(|i| {
                let mut row = HashMap::new();
                row.insert("ext_id".to_string(), Value::Int(i as i64));
                row.insert(
                    "emb".to_string(),
                    Value::Vector(train_flat[i * dim..(i + 1) * dim].to_vec()),
                );
                Value::Map(row)
            })
            .collect();
        if !batch.is_empty() {
            loaded += batch.len();
            let mut params = HashMap::new();
            params.insert("batch".to_string(), Value::List(batch));
            client
                .execute_cypher_with_params(&insert_cypher, params)
                .await?;
        }
        sent = end;
        if sent % 10_000 < args.batch_size {
            info!(
                sent,
                of = n_train,
                elapsed_s = wall.elapsed().as_secs(),
                "load progress"
            );
        }
    }
    let load_secs = wall.elapsed().as_secs_f64();
    info!(
        load_secs,
        loaded,
        shard_idx = args.shard_idx,
        n_shards = args.n_shards,
        vectors_per_sec = loaded as f64 / load_secs.max(f64::EPSILON),
        "data load complete"
    );

    if !args.no_index {
        let ddl = format!(
            "CREATE VECTOR INDEX bench_emb_idx ON :{label}({prop}) \
             OPTIONS {{m: {m}, ef_construction: {efc}, metric: \"{metric}\", dimensions: {dim}}}",
            label = args.label,
            prop = args.property,
            m = args.m,
            efc = args.ef_construction,
            metric = args.metric,
        );
        info!(%ddl, "creating vector index");
        let t = Instant::now();
        client.execute_cypher(&ddl).await?;
        info!(
            index_build_secs = t.elapsed().as_secs_f64(),
            "vector index ready"
        );
    }
    Ok(())
}
