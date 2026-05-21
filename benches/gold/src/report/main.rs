//! `gold-report` binary placeholder. Once LDBC / ann / TSBS / etc.
//! land their per-modality runners, this binary will orchestrate all
//! of them and render the full Composite block from
//! `arch/benchmarks/methodology.md`. For now it prints the
//! "what's currently implemented" status so it has a meaningful exit
//! code in CI.

fn main() {
    println!(
        "gold-report: composite suite is partial — YCSB-only foundation landed.\n\
         Pending modality runners: LDBC SNB, ann-benchmarks SIFT1M,\n\
         Search Benchmark Game, TSBS DevOps, SpatialBench, VectorDBBench.\n\
         Run `cargo run --release -p coordinode-gold-bench --bin gold-ycsb` to\n\
         get the KV partial composite today."
    );
}
