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
    let redis_a = baselines
        .kv
        .ycsb
        .workload_a
        .get("redis")
        .expect("redis baseline for workload A");
    let redis_c = baselines
        .kv
        .ycsb
        .workload_c
        .get("redis")
        .expect("redis baseline for workload C");

    println!(
        "{}",
        report::format_header(a.preset.records, a.preset.operations),
    );

    println!(
        "{}",
        report::format_modality_block(
            "KV (YCSB Workload A: 50% read / 50% update)",
            &[
                report::Row {
                    metric: "Throughput (ops/s)",
                    cn: a.throughput_ops_s(),
                    leader: redis_a.throughput_ops_s,
                    higher_is_better: true,
                },
                report::Row {
                    metric: "Read P99 (µs)",
                    cn: a.read_p99_us(),
                    leader: redis_a.read_p99_us,
                    higher_is_better: false,
                },
            ],
            &format!("Redis ({})", redis_a.source),
        ),
    );

    println!(
        "{}",
        report::format_modality_block(
            "KV (YCSB Workload C: 100% read)",
            &[
                report::Row {
                    metric: "Throughput (ops/s)",
                    cn: c.throughput_ops_s(),
                    leader: redis_c.throughput_ops_s,
                    higher_is_better: true,
                },
                report::Row {
                    metric: "Read P99 (µs)",
                    cn: c.read_p99_us(),
                    leader: redis_c.read_p99_us,
                    higher_is_better: false,
                },
            ],
            &format!("Redis ({})", redis_c.source),
        ),
    );

    let cn_score = report::geometric_mean(&[
        report::throughput_ratio(a.throughput_ops_s(), redis_a.throughput_ops_s),
        report::latency_ratio(a.read_p99_us(), redis_a.read_p99_us),
        report::throughput_ratio(c.throughput_ops_s(), redis_c.throughput_ops_s),
        report::latency_ratio(c.read_p99_us(), redis_c.read_p99_us),
    ]);
    println!(
        "\n┌─────────────────────────────────────────────────────────────┐\n\
         │ Composite (KV partial — YCSB A + C):                        │\n\
         │   Geometric mean of 4 ratios = {cn_score:.3}x vs Redis              │\n\
         │   (LDBC, ann-benchmarks, BEIR, TSBS, SpatialBench pending)  │\n\
         └─────────────────────────────────────────────────────────────┘",
    );
}
