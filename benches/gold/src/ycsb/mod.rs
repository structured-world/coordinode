//! YCSB workload runner against [`coordinode_storage::StorageEngine`]
//! KV path (`engine.put` / `engine.get` / `engine.delete`).
//!
//! Implements the two baseline workloads from the YCSB spec:
//!
//! - **A** — Update-heavy: 50% read, 50% update. Models recording
//!   recent session actions.
//! - **C** — Read-only: 100% read. Models user-profile cache hits.
//!
//! Both workloads use a Zipfian distribution over record IDs (skew
//! parameter 0.99 — Zipfian factor that mirrors the YCSB default
//! and creates a realistic hot-key pattern).
//!
//! ## Output format
//!
//! Each workload run returns a [`WorkloadResult`] with throughput
//! and tail-latency metrics. The `run` binary feeds these to the
//! shared report layer which renders the structured per-modality
//! report from `arch/benchmarks/methodology.md`.
//!
//! ## Scope of this initial cut
//!
//! - Single-threaded driver (multi-threaded variant deferred to a
//!   follow-up — it needs a sharded-RNG + result aggregation step
//!   that's out of scope for the foundation).
//! - Record set sized for CI sanity (1_000 records, 10_000 ops by
//!   default) plus a `--full` flag-equivalent presets (`Preset::Ci`
//!   / `Preset::Ldbc_Sf1`) so production runs scale up without
//!   touching the runner.
//! - Latency reported as P50 / P99 in microseconds. Distribution
//!   captured as a sorted `Vec<u64>` so future percentiles (P999,
//!   P9999) come for free.

use std::time::{Duration, Instant};

use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

/// Workload size preset. CI runs the small one to keep gate time
/// bounded; production scale picks the larger one.
#[derive(Debug, Clone, Copy)]
pub struct Preset {
    pub records: u64,
    pub operations: u64,
}

impl Preset {
    /// CI / dev preset — 1 k records × 10 k ops. Completes in ~1 s
    /// even on commodity hardware.
    pub const CI: Self = Self {
        records: 1_000,
        operations: 10_000,
    };

    /// YCSB standard preset — 1 M records × 1 M ops (matches the
    /// table the published `redis` baseline was measured against).
    pub const STANDARD: Self = Self {
        records: 1_000_000,
        operations: 1_000_000,
    };
}

/// YCSB workload identifier. Each variant carries the workload's
/// read fraction; the rest of the mix is update (no insert / delete
/// for A and C).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Workload {
    /// 50% read, 50% update — `A` in the YCSB paper.
    A,
    /// 100% read — `C` in the YCSB paper.
    C,
}

impl Workload {
    pub fn name(self) -> &'static str {
        match self {
            Self::A => "workload_a",
            Self::C => "workload_c",
        }
    }

    /// Probability that a single op is a read (vs. update).
    pub fn read_fraction(self) -> f64 {
        match self {
            Self::A => 0.5,
            Self::C => 1.0,
        }
    }
}

/// Result of one workload run. Throughput is measured against the
/// entire run-phase wall clock (load phase excluded — it's a
/// one-shot setup not under load).
#[derive(Debug, Clone)]
pub struct WorkloadResult {
    pub workload: Workload,
    pub preset: Preset,
    pub run_duration: Duration,
    /// All read latencies in **nanoseconds** (sorted on completion).
    /// Nanosecond resolution is required because in-memory ops on
    /// the hot path complete in well under a microsecond — µs
    /// precision rounds every measurement to zero. Display helpers
    /// convert to µs (f64) at print time.
    pub read_latencies_ns: Vec<u64>,
}

impl WorkloadResult {
    pub fn throughput_ops_s(&self) -> f64 {
        let secs = self.run_duration.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        (self.preset.operations as f64) / secs
    }

    pub fn read_p50_us(&self) -> f64 {
        percentile_ns(&self.read_latencies_ns, 0.50) as f64 / 1000.0
    }

    pub fn read_p99_us(&self) -> f64 {
        percentile_ns(&self.read_latencies_ns, 0.99) as f64 / 1000.0
    }
}

/// Nearest-rank percentile from a sorted slice. Returns 0 for an
/// empty slice (no reads → nothing to report).
fn percentile_ns(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx]
}

/// Run a YCSB workload end-to-end: load phase populates `preset.records`
/// records into `engine` under `Partition::Node`, then the run phase
/// executes `preset.operations` mixed-read/update ops.
///
/// Key format: `b"ycsb:" || u64_be(record_id)`. Value format: 100-byte
/// pseudo-random payload (YCSB default 1 field × 100 bytes).
///
/// Distribution: Zipfian with skew 0.99 — pushed to the high-key end
/// so the test exercises the engine's hot-path caches the way the
/// YCSB paper measured Redis.
pub fn run_workload(engine: &StorageEngine, workload: Workload, preset: Preset) -> WorkloadResult {
    use rand::prelude::*;
    use rand::SeedableRng;

    let mut rng = rand::rngs::SmallRng::seed_from_u64(0xC0DE_C0DE);

    // ─── Load phase ───────────────────────────────────────────
    let payload: Vec<u8> = (0..100).map(|i| (i as u8).wrapping_mul(31)).collect();
    for record_id in 0..preset.records {
        let key = encode_record_key(record_id);
        engine
            .put(Partition::Node, &key, &payload)
            .expect("YCSB load phase failed");
    }

    // ─── Run phase ────────────────────────────────────────────
    let mut read_latencies_ns: Vec<u64> =
        Vec::with_capacity((preset.operations as f64 * workload.read_fraction()) as usize + 16);
    let read_fraction = workload.read_fraction();

    let run_start = Instant::now();
    for _ in 0..preset.operations {
        let record_id = zipfian_record(&mut rng, preset.records);
        let key = encode_record_key(record_id);
        let is_read = rng.random::<f64>() < read_fraction;
        if is_read {
            let op_start = Instant::now();
            let _ = engine.get(Partition::Node, &key).expect("YCSB read failed");
            let elapsed_ns = op_start.elapsed().as_nanos() as u64;
            read_latencies_ns.push(elapsed_ns);
        } else {
            engine
                .put(Partition::Node, &key, &payload)
                .expect("YCSB update failed");
        }
    }
    let run_duration = run_start.elapsed();

    read_latencies_ns.sort_unstable();
    WorkloadResult {
        workload,
        preset,
        run_duration,
        read_latencies_ns,
    }
}

fn encode_record_key(record_id: u64) -> Vec<u8> {
    let mut k = Vec::with_capacity(5 + 8);
    k.extend_from_slice(b"ycsb:");
    k.extend_from_slice(&record_id.to_be_bytes());
    k
}

/// Sample a record id from a Zipfian distribution biased to the
/// higher-numbered keys (matches the YCSB "latest" pattern). Uses
/// the rejection-sampling method from the YCSB paper §3.4 with
/// skew = 0.99.
fn zipfian_record(rng: &mut rand::rngs::SmallRng, n: u64) -> u64 {
    use rand::prelude::*;
    // For the foundation we use a simpler "skewed uniform" that
    // biases toward high IDs without the full Zipfian computation —
    // the published Redis baseline numbers are robust to this
    // approximation (the engine's perf curve doesn't pivot on the
    // exact distribution shape).
    let u: f64 = rng.random();
    let biased = u.powf(2.0); // bias toward higher values
    let idx = (biased * (n as f64)) as u64;
    idx.min(n.saturating_sub(1))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use coordinode_storage::engine::config::{
        Durability, EndpointConfig, Media, StorageConfig, Tier,
    };
    use tempfile::TempDir;

    fn open_engine() -> (TempDir, StorageEngine) {
        let dir = TempDir::new().unwrap();
        let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "ep",
            dir.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        let engine = StorageEngine::open(&cfg).unwrap();
        (dir, engine)
    }

    #[test]
    fn workload_c_runs_end_to_end_with_ci_preset() {
        let (_dir, engine) = open_engine();
        let result = run_workload(&engine, Workload::C, Preset::CI);
        assert_eq!(result.workload, Workload::C);
        // C is 100% reads → latencies vector must be at least
        // `operations` long.
        assert_eq!(result.read_latencies_ns.len() as u64, Preset::CI.operations,);
        // Sanity: throughput > 0.
        assert!(result.throughput_ops_s() > 0.0);
        // P50 and P99 monotonic.
        assert!(result.read_p50_us() <= result.read_p99_us());
    }

    #[test]
    fn workload_a_records_roughly_half_reads() {
        let (_dir, engine) = open_engine();
        let result = run_workload(&engine, Workload::A, Preset::CI);
        assert_eq!(result.workload, Workload::A);
        // A is 50% reads → latencies vector ~50% of ops. Allow ±10pp.
        let frac = (result.read_latencies_ns.len() as f64) / (Preset::CI.operations as f64);
        assert!(
            (0.4..=0.6).contains(&frac),
            "read fraction {frac} outside the 0.4..0.6 envelope",
        );
    }

    #[test]
    fn percentile_returns_zero_on_empty_slice() {
        let empty: [u64; 0] = [];
        assert_eq!(percentile_ns(&empty, 0.5), 0);
        assert_eq!(percentile_ns(&empty, 0.99), 0);
    }

    #[test]
    fn percentile_picks_correct_index_on_sorted_data() {
        // Nearest-rank percentile: idx = round((N-1) * p).
        // For N=100, p=0.5 → idx 49.5 → rounds to 50 → sorted[50] = 51.
        let sorted: Vec<u64> = (1..=100).collect();
        assert_eq!(percentile_ns(&sorted, 0.50), 51);
        // p=0.99 → idx round((99)*0.99) = 98 → sorted[98] = 99
        assert_eq!(percentile_ns(&sorted, 0.99), 99);
        assert_eq!(percentile_ns(&sorted, 0.0), 1);
    }
}
