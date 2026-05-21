//! YCSB gold-bench binary. Runs workload A + C against a tempdir
//! `StorageEngine` and renders the per-modality YCSB report from
//! `arch/benchmarks/methodology.md`. Pass `--standard` to use the
//! published-baseline preset (1 M records / 1 M ops) instead of the
//! default CI preset.

use coordinode_gold_bench::baselines::Baselines;
use coordinode_gold_bench::report;
use coordinode_gold_bench::ycsb::{run_workload, Preset, Workload, WorkloadResult};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use tempfile::TempDir;

fn main() {
    let preset = if std::env::args().any(|a| a == "--standard") {
        Preset::STANDARD
    } else {
        Preset::CI
    };

    eprintln!(
        "gold-ycsb: starting (records={}, operations={})",
        preset.records, preset.operations,
    );

    let dir = TempDir::new().expect("tempdir");
    let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "ep",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&cfg).expect("open engine");

    let result_a = run_workload(&engine, Workload::A, preset);
    let result_c = run_workload(&engine, Workload::C, preset);

    let baselines = Baselines::embedded();
    print_report(&result_a, &result_c, &baselines);
}

fn print_report(a: &WorkloadResult, c: &WorkloadResult, baselines: &Baselines) {
    // Primary competitor: MongoDB 8.x at the chosen comparison
    // codec (zstd) plus the uncompressed CPU baseline. Per
    // methodology §"Codec choice" snappy is explicitly excluded —
    // we standardize on zstd × zstd and none × none.
    let mongo_a_zstd = baselines
        .document
        .ycsb
        .workload_a
        .get("mongodb_8_zstd")
        .expect("mongodb_8_zstd baseline for workload A");
    let _mongo_a_none = baselines
        .document
        .ycsb
        .workload_a
        .get("mongodb_8_none")
        .expect("mongodb_8_none baseline for workload A");
    let mongo_c_zstd = baselines
        .document
        .ycsb
        .workload_c
        .get("mongodb_8_zstd")
        .expect("mongodb_8_zstd baseline for workload C");
    let _mongo_c_none = baselines
        .document
        .ycsb
        .workload_c
        .get("mongodb_8_none")
        .expect("mongodb_8_none baseline for workload C");

    println!(
        "{}",
        report::format_header(a.preset.records, a.preset.operations),
    );
    println!(
        "\n⚠ DISCLAIMER — INTERNAL REGRESSION BENCH only.\n\
         CoordiNode is running embedded in-process here; MongoDB's row is\n\
         a placeholder (the value column reads \"-\" until the harness\n\
         actually runs Mongo via mongoperf / YCSB / networked driver in\n\
         BOTH zstd and uncompressed modes per methodology §Codec choice).\n\
         The ratios below are NOT publishable marketing numbers — they\n\
         measure engine micro-perf, not buyer-comparable workloads.\n\
         For honest comparisons see the v0.4-alpha milestone gate in\n\
         arch/benchmarks/methodology.md.\n",
    );

    println!("\nDocument (YCSB Workload A: 50% read / 50% update)");
    print_codec_block(
        "CN (zstd)",
        a.throughput_ops_s(),
        a.read_p99_us(),
        "MongoDB 8.x (zstd)",
        mongo_a_zstd.throughput_ops_s,
        mongo_a_zstd.read_p99_us,
        &mongo_a_zstd.source,
    );
    print_codec_block(
        "CN (none)",
        a.throughput_ops_s(),
        a.read_p99_us(),
        "MongoDB 8.x (none)",
        None,
        None,
        "harness measurement pending",
    );

    println!("\nDocument (YCSB Workload C: 100% read)");
    print_codec_block(
        "CN (zstd)",
        c.throughput_ops_s(),
        c.read_p99_us(),
        "MongoDB 8.x (zstd)",
        mongo_c_zstd.throughput_ops_s,
        mongo_c_zstd.read_p99_us,
        &mongo_c_zstd.source,
    );
    print_codec_block(
        "CN (none)",
        c.throughput_ops_s(),
        c.read_p99_us(),
        "MongoDB 8.x (none)",
        None,
        None,
        "harness measurement pending",
    );

    println!(
        "\n┌─────────────────────────────────────────────────────────────┐\n\
         │ Document partial — YCSB A + C, internal regression bench    │\n\
         │                                                             │\n\
         │ Per methodology §\"Codec choice\":                            │\n\
         │   • zstd × zstd and none × none are the only valid sweeps   │\n\
         │   • snappy / lz4 are EXCLUDED                               │\n\
         │   • MongoDB zstd / none numbers require harness measurement │\n\
         │                                                             │\n\
         │ Pending for v0.4-alpha gate:                                │\n\
         │   • Networked CoordiNode YCSB (gRPC client→server)          │\n\
         │   • MongoDB harness running both zstd and none modes        │\n\
         │   • Vector vs OpenSearch / SurrealDB (ann-benchmarks SIFT1M)│\n\
         │ Pending for v0.5-alpha:                                     │\n\
         │   • LDBC SNB vs SurrealDB / ArangoDB / Neo4j                │\n\
         │   • UniBench multi-model vs ArangoDB                        │\n\
         └─────────────────────────────────────────────────────────────┘",
    );
}

#[allow(clippy::too_many_arguments)]
fn print_codec_block(
    cn_label: &str,
    cn_throughput: f64,
    cn_p99_us: f64,
    leader_label: &str,
    leader_throughput: Option<f64>,
    leader_p99_us: Option<f64>,
    leader_source: &str,
) {
    println!("  ─────────────────────────────────────────────────────────────");
    println!(
        "  {:<18} → throughput {:>10.0} ops/s   read P99 {:>7.2} µs",
        cn_label, cn_throughput, cn_p99_us,
    );
    match (leader_throughput, leader_p99_us) {
        (Some(t), Some(p)) => {
            let throughput_ratio = report::throughput_ratio(cn_throughput, t);
            let latency_ratio = report::latency_ratio(cn_p99_us, p);
            println!(
                "  {:<18} → throughput {:>10.0} ops/s   read P99 {:>7.2} µs   (CN {:.2}× thr / {:.2}× lat)",
                leader_label, t, p, throughput_ratio, latency_ratio,
            );
        }
        _ => {
            println!(
                "  {:<18} → harness measurement PENDING ({})",
                leader_label, leader_source,
            );
        }
    }
}
