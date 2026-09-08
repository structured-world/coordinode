//! Retained-history sizing: what the MVCC retention window holds on disk
//! against the live version, per write pattern and window length.
//!
//! The engine is opened on the timestamp oracle so seqnos are a clock the
//! harness drives itself: every round of writes lands one simulated second
//! after the previous one and is flushed and compacted by the tree's own
//! Leveled strategy until it finds no work, the way the engine's background
//! worker keeps a live database. The window is fixed for the whole run, so
//! the fold below the watermark and the pruning of retained versions happen
//! during the rounds exactly as they would in steady state; one run per
//! window length, measured once at the end.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use coordinode_core::txn::proposal::{Mutation, PartitionId};
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::retention_stats::RetainedHistory;
use lsm_tree::AbstractTree;
use lsm_tree::compaction::{CompactionAction, Leveled};

/// Simulated wall-clock distance between two rounds of writes.
const ROUND_US: u64 = 1_000_000;

/// Mutations per proposal: one seqno per proposal, so a round of `keys`
/// writes spans `keys / BATCH` seqnos inside its simulated second.
const BATCH: u64 = 1_000;

/// Which keys a round writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pattern {
    /// Every round rewrites the same key set: the window retains one
    /// superseded version per key per round, the shape a hot working set
    /// under updates produces.
    Overwrite,
    /// Every round writes a fresh, ascending key set: nothing is ever
    /// superseded and rounds never overlap, so compaction can move tables
    /// down without rewriting them. Time-sortable ids write like this.
    AppendOnly,
    /// Every round writes fresh keys spread uniformly over the key space:
    /// nothing is superseded, but every round overlaps every other, so each
    /// compaction rewrites what it merges into. Hashed ids and secondary
    /// index entries write like this.
    AppendRandom,
}

impl Pattern {
    pub fn name(self) -> &'static str {
        match self {
            Self::Overwrite => "overwrite",
            Self::AppendOnly => "append-only, ascending",
            Self::AppendRandom => "append-only, random",
        }
    }

    pub const ALL: [Self; 3] = [Self::Overwrite, Self::AppendOnly, Self::AppendRandom];
}

/// Workload size.
#[derive(Debug, Clone, Copy)]
pub struct Shape {
    /// Keys written per round.
    pub keys: u64,
    /// Rounds, one simulated second apart.
    pub rounds: u64,
    /// Value size per write.
    pub value_bytes: usize,
}

impl Shape {
    /// Finishes in well under a minute on a laptop; enough rounds for the
    /// window lengths to differ.
    pub const CI: Self = Self {
        keys: 20_000,
        rounds: 8,
        value_bytes: 256,
    };

    /// Sizing run: a few hundred MiB written per run, minutes of wall clock.
    pub const FULL: Self = Self {
        keys: 200_000,
        rounds: 16,
        value_bytes: 512,
    };

    /// Window lengths to measure, in rounds: everything, half, two, one,
    /// none (only what the most recent install still holds).
    pub fn windows(&self) -> Vec<u64> {
        let mut w = vec![self.rounds, self.rounds / 2, 2, 1, 0];
        w.dedup();
        w
    }
}

/// Result of one run: one pattern, one window length.
#[derive(Debug, Clone)]
pub struct RunResult {
    pub pattern: Pattern,
    pub shape: Shape,
    /// Rounds the window spans; the watermark trails the clock by this
    /// many rounds plus half a round of slack.
    pub window_rounds: u64,
    /// Bytes handed to the engine (keys plus values), for the
    /// retained-per-written figure.
    pub bytes_written: u64,
    pub history: RetainedHistory,
}

impl RunResult {
    pub fn retained_per_written(&self) -> f64 {
        if self.bytes_written == 0 {
            0.0
        } else {
            self.history.retained_bytes as f64 / self.bytes_written as f64
        }
    }
}

/// Runs `pattern` at `shape` with the window fixed at `window_rounds` in a
/// fresh engine under `dir`, and measures the Node partition at the end.
pub fn run(pattern: Pattern, shape: Shape, window_rounds: u64, dir: &Path) -> RunResult {
    let base = future_base();
    let oracle = Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(base)));
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "ep",
        dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine =
        StorageEngine::open_with_oracle(&config, Arc::clone(&oracle)).expect("open engine");
    // Half a round of slack so exactly `window_rounds` rounds of commits sit
    // at or above the watermark at the moment each round settles.
    engine.set_retention_window(Duration::from_micros(
        window_rounds * ROUND_US + ROUND_US / 2,
    ));

    let mut bytes_written = 0u64;
    let mut value = vec![0u8; shape.value_bytes];
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    for round in 0..shape.rounds {
        let round_ts = base + (round + 1) * ROUND_US;
        let mut batch = Vec::with_capacity(BATCH as usize);
        for i in 0..shape.keys {
            let record = match pattern {
                Pattern::Overwrite => i,
                Pattern::AppendOnly => round * shape.keys + i,
                Pattern::AppendRandom => next(&mut rng),
            };
            fill(&mut value, &mut rng);
            let key = encode_key(record);
            bytes_written += (key.len() + value.len()) as u64;
            batch.push(Mutation::Put {
                partition: PartitionId::Node,
                key,
                value: value.clone(),
            });
            if batch.len() as u64 == BATCH {
                let commit_ts = round_ts + i / BATCH;
                engine
                    .apply_proposal_at(&batch, commit_ts)
                    .expect("apply proposal");
                batch.clear();
            }
        }
        if !batch.is_empty() {
            let commit_ts = round_ts + shape.keys / BATCH + 1;
            engine
                .apply_proposal_at(&batch, commit_ts)
                .expect("apply proposal");
        }
        // The commit path moved the oracle, so the watermark now trails this
        // round by the window; settle the round against it.
        engine.advance_gc_watermark();
        settle(&engine);
    }

    RunResult {
        pattern,
        shape,
        window_rounds,
        bytes_written,
        history: engine.retained_history(Partition::Node).expect("stats"),
    }
}

/// Flushes the Node partition and runs its Leveled compaction until the
/// strategy reports nothing left to do, at the engine's current watermark.
fn settle(engine: &StorageEngine) {
    let watermark = engine.gc_watermark();
    let tree = engine.tree(Partition::Node).expect("tree");
    tree.flush_active_memtable(watermark).expect("flush");
    loop {
        let result = tree
            .compact(Arc::new(Leveled::default()), watermark)
            .expect("compaction");
        if result.action == CompactionAction::Nothing {
            break;
        }
    }
}

fn encode_key(record: u64) -> Vec<u8> {
    let mut k = Vec::with_capacity(5 + 8);
    k.extend_from_slice(b"node:");
    k.extend_from_slice(&record.to_be_bytes());
    k
}

/// One xorshift step.
fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Fills `buf` from the xorshift stream so values differ between rounds and
/// compress like data rather than like a constant.
fn fill(buf: &mut [u8], state: &mut u64) {
    for chunk in buf.chunks_mut(8) {
        let bytes = next(state).to_le_bytes();
        chunk.copy_from_slice(&bytes[..chunk.len()]);
    }
}

/// Wall-clock microseconds now plus a wide margin: opening an engine
/// re-anchors the oracle to the wall clock, so a harness that drives the
/// clock itself must start in the future.
fn future_base() -> u64 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_micros();
    u64::try_from(now).expect("fits") + 1_000_000_000_000
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    const SMALL: Shape = Shape {
        keys: 2_000,
        rounds: 4,
        value_bytes: 64,
    };

    /// With every round inside the window an overwrite run holds history;
    /// with the window empty it holds at most the newest install's inputs,
    /// and the live version is the folded one, smaller than the unfolded.
    #[test]
    fn overwrite_history_shrinks_with_the_window() {
        let all = run(
            Pattern::Overwrite,
            SMALL,
            SMALL.rounds,
            TempDir::new().unwrap().path(),
        );
        let none = run(Pattern::Overwrite, SMALL, 0, TempDir::new().unwrap().path());
        assert_eq!(all.bytes_written, 4 * 2_000 * (13 + 64));
        assert!(all.history.retained_bytes > 0);
        assert!(none.history.retained_bytes < all.history.retained_bytes);
        assert!(none.history.live_bytes < all.history.live_bytes);
        assert!(none.history.live_bytes > 0);
    }

    /// Ascending append-only never overlaps between rounds, so a whole-run
    /// window costs no more than the most recent install's inputs.
    #[test]
    fn ascending_append_only_retains_at_most_the_last_install() {
        let all = run(
            Pattern::AppendOnly,
            SMALL,
            SMALL.rounds,
            TempDir::new().unwrap().path(),
        );
        let none = run(
            Pattern::AppendOnly,
            SMALL,
            0,
            TempDir::new().unwrap().path(),
        );
        assert_eq!(all.bytes_written, 4 * 2_000 * (13 + 64));
        assert!(all.history.retained_bytes <= none.history.retained_bytes + all.history.live_bytes);
        assert!(
            (all.history.live_bytes as f64 - none.history.live_bytes as f64).abs()
                < all.history.live_bytes as f64 * 0.05,
            "nothing is superseded, so the window does not change the live size"
        );
    }

    /// Windows are measured in descending order without duplicates.
    #[test]
    fn windows_are_distinct_and_descending() {
        let w = SMALL.windows();
        assert_eq!(w, vec![4, 2, 1, 0]);
        let w = Shape::CI.windows();
        assert_eq!(w, vec![8, 4, 2, 1, 0]);
    }
}
