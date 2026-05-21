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
    // Primary competitor: MongoDB 8.x — the multi-model document-
    // store baseline per the rewritten methodology doc. Redis is
    // explicitly NOT a target (in-memory cache, different product).
    let mongo_a = baselines
        .document
        .ycsb
        .workload_a
        .get("mongodb_8")
        .expect("mongodb_8 baseline for workload A");
    let mongo_c = baselines
        .document
        .ycsb
        .workload_c
        .get("mongodb_8")
        .expect("mongodb_8 baseline for workload C");

    let mongo_a_throughput = mongo_a
        .throughput_ops_s
        .expect("mongodb_8 workload_a throughput must be populated");
    let mongo_a_p99 = mongo_a
        .read_p99_us
        .expect("mongodb_8 workload_a read_p99_us must be populated");
    let mongo_c_throughput = mongo_c
        .throughput_ops_s
        .expect("mongodb_8 workload_c throughput must be populated");
    let mongo_c_p99 = mongo_c
        .read_p99_us
        .expect("mongodb_8 workload_c read_p99_us must be populated");

    println!(
        "{}",
        report::format_header(a.preset.records, a.preset.operations),
    );
    println!(
        "\n⚠ DISCLAIMER — this is the INTERNAL REGRESSION BENCH only.\n\
         CoordiNode is running embedded in-process here; MongoDB's number\n\
         is networked client→server. The ratios below are NOT publishable\n\
         marketing numbers — they measure engine micro-perf, not buyer-\n\
         comparable workloads. For honest comparisons see the v0.4-alpha\n\
         milestone gate in arch/benchmarks/methodology.md (requires the\n\
         networked YCSB harness, not yet built).\n",
    );

    println!(
        "{}",
        report::format_modality_block(
            "Document (YCSB Workload A: 50% read / 50% update)",
            &[
                report::Row {
                    metric: "Throughput (ops/s)",
                    cn: a.throughput_ops_s(),
                    leader: mongo_a_throughput,
                    higher_is_better: true,
                },
                report::Row {
                    metric: "Read P99 (µs)",
                    cn: a.read_p99_us(),
                    leader: mongo_a_p99,
                    higher_is_better: false,
                },
            ],
            &format!("MongoDB 8.x ({})", mongo_a.source),
        ),
    );

    println!(
        "{}",
        report::format_modality_block(
            "Document (YCSB Workload C: 100% read)",
            &[
                report::Row {
                    metric: "Throughput (ops/s)",
                    cn: c.throughput_ops_s(),
                    leader: mongo_c_throughput,
                    higher_is_better: true,
                },
                report::Row {
                    metric: "Read P99 (µs)",
                    cn: c.read_p99_us(),
                    leader: mongo_c_p99,
                    higher_is_better: false,
                },
            ],
            &format!("MongoDB 8.x ({})", mongo_c.source),
        ),
    );

    let cn_score = report::geometric_mean(&[
        report::throughput_ratio(a.throughput_ops_s(), mongo_a_throughput),
        report::latency_ratio(a.read_p99_us(), mongo_a_p99),
        report::throughput_ratio(c.throughput_ops_s(), mongo_c_throughput),
        report::latency_ratio(c.read_p99_us(), mongo_c_p99),
    ]);
    println!(
        "\n┌─────────────────────────────────────────────────────────────┐\n\
         │ Document partial — YCSB A + C, internal regression bench    │\n\
         │ vs MongoDB 8.x (networked baseline):                        │\n\
         │   Geometric mean of 4 ratios = {cn_score:.3}x                       │\n\
         │                                                             │\n\
         │ Pending for v0.4-alpha gate per methodology.md:             │\n\
         │   • Networked CoordiNode YCSB (gRPC client→server)          │\n\
         │   • Vector vs OpenSearch / SurrealDB (ann-benchmarks SIFT1M)│\n\
         │ Pending for v0.5-alpha:                                     │\n\
         │   • LDBC SNB vs SurrealDB / ArangoDB / Neo4j                │\n\
         │   • UniBench multi-model vs ArangoDB                        │\n\
         └─────────────────────────────────────────────────────────────┘",
    );
}
