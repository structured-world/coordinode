//! Gold benchmark suite — Level 2 from
//! `arch/benchmarks/methodology.md`. Hosts the YCSB, LDBC, ann-
//! benchmarks etc. harness code; the binaries in
//! `src/{ycsb,report}/main.rs` are thin drivers over the modules
//! here.
//!
//! **no-std tier:** `std-only`. Benchmark harnesses inherently need
//! `std::time::Instant` (latency measurement), `std::fs` (dataset
//! loaders for SIFT1M / LDBC / TSBS), and `tempfile` (engine fixture
//! dirs). Out of scope for any no-std readiness.
//!
//! Status: **YCSB A + C, baseline-comparison report.**
//!
//! Out-of-scope of this initial cut (tracked as follow-up):
//! - LDBC SNB Interactive v2 loader + queries
//! - ann-benchmarks SIFT1M loader + HNSW recall/QPS sweep
//! - Search Benchmark Game wikipedia loader
//! - TSBS DevOps (100 hosts) ingestion + last-point
//! - SpatialBench
//! - VectorDBBench (server-level vs library-level ann-benchmarks)
//! - Hybrid multi-modal benchmark (composite-score showcase)

pub mod baselines;
pub mod report;
pub mod ycsb;
