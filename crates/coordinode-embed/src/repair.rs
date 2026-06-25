//! Embedded checkpoint + WAL-replay-repair (G111).
//!
//! Single-node / embedded deployments have no replica to repair a corrupt
//! partition from, so the durability story is: detect corruption (scrub) and
//! rebuild from the last checkpoint plus the retained oplog journal. This
//! module wires the storage primitives together:
//!
//! - [`create_checkpoint`] snapshots a repair base (and copies the oplog into
//!   it) under `<data_dir>/checkpoints/ckpt-<n>`.
//! - [`verify_and_repair`] scrubs every partition and, for each corrupt one,
//!   rebuilds it from the latest checkpoint plus the oplog entries recorded
//!   after that checkpoint's cursor.
//!
//! Same-disk checkpoints protect against localized corruption only; whole-disk
//! loss requires an off-device backup (PITR).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::error::{StorageError, StorageResult};
use coordinode_storage::scrub::{scrub_all, ScrubConfig};

const CKPT_PREFIX: &str = "ckpt-";

/// The conventional checkpoint root under an engine's data directory.
#[must_use]
pub fn checkpoint_root(data_dir: &Path) -> PathBuf {
    data_dir.join("checkpoints")
}

/// Parse the numeric index from a `ckpt-<n>` directory name.
fn parse_ckpt_index(path: &Path) -> Option<u64> {
    path.file_name()?
        .to_str()?
        .strip_prefix(CKPT_PREFIX)?
        .parse()
        .ok()
}

/// All checkpoint directories under `root`, ascending by index.
fn list_checkpoints(root: &Path) -> StorageResult<Vec<(u64, PathBuf)>> {
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    let entries = std::fs::read_dir(root)
        .map_err(|e| StorageError::Io(format!("read checkpoint root {root:?}: {e}")))?;
    for entry in entries.flatten() {
        let p = entry.path();
        if let Some(idx) = parse_ckpt_index(&p) {
            out.push((idx, p));
        }
    }
    out.sort_by_key(|(idx, _)| *idx);
    Ok(out)
}

/// The newest checkpoint directory under `root`, if any.
#[must_use]
pub fn latest_checkpoint(root: &Path) -> Option<PathBuf> {
    list_checkpoints(root).ok()?.pop().map(|(_, p)| p)
}

/// Create a checkpoint (repair base + oplog copy) under `root`, named
/// `ckpt-<next>` with a monotonic zero-padded index so lexical order = age
/// order. Returns the new checkpoint path.
pub fn create_checkpoint(engine: &StorageEngine, root: &Path) -> StorageResult<PathBuf> {
    std::fs::create_dir_all(root)
        .map_err(|e| StorageError::Io(format!("create checkpoint root {root:?}: {e}")))?;
    let next = list_checkpoints(root)?
        .last()
        .map(|(idx, _)| idx + 1)
        .unwrap_or(0);
    let target = root.join(format!("{CKPT_PREFIX}{next:020}"));
    engine.create_checkpoint(&target)?;
    Ok(target)
}

/// Delete all but the newest `keep` checkpoints under `root`. `keep == 0` is
/// treated as `1` (never prune the only base). Returns the number removed.
pub fn prune_checkpoints(root: &Path, keep: usize) -> StorageResult<usize> {
    let keep = keep.max(1);
    let all = list_checkpoints(root)?;
    if all.len() <= keep {
        return Ok(0);
    }
    let remove_count = all.len() - keep;
    let mut removed = 0;
    for (_, path) in all.into_iter().take(remove_count) {
        std::fs::remove_dir_all(&path)
            .map_err(|e| StorageError::Io(format!("prune checkpoint {path:?}: {e}")))?;
        removed += 1;
    }
    Ok(removed)
}

/// Outcome of a [`verify_and_repair`] pass.
#[derive(Debug, Clone, Default)]
pub struct RepairReport {
    /// Distinct partitions the scrub found corrupt (ascending discriminant).
    pub corrupt_partitions: Vec<Partition>,
    /// Partitions actually rebuilt from a checkpoint this pass.
    pub repaired: Vec<Partition>,
    /// The checkpoint used as the rebuild base, if repair ran.
    pub checkpoint_used: Option<PathBuf>,
    /// `true` if a re-scrub after repair found no remaining corruption.
    /// Trivially `true` when no corruption was found in the first place.
    pub clean_after: bool,
}

impl RepairReport {
    /// `true` if the engine is corruption-free (either it was clean, or repair
    /// rebuilt every corrupt partition and the re-scrub passed).
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.clean_after
    }
}

/// Scrub every partition; rebuild each corrupt one from the latest checkpoint
/// plus oplog replay. A no-op (clean report) when scrub finds nothing.
///
/// If corruption is found but no checkpoint exists, the report lists the
/// corrupt partitions with `clean_after = false` and an empty `repaired` set —
/// the operator must restore from an off-device backup.
pub fn verify_and_repair(
    engine: &StorageEngine,
    checkpoint_root: &Path,
) -> StorageResult<RepairReport> {
    let scrub = scrub_all(engine, &ScrubConfig::default())?;
    if !scrub.has_errors() {
        return Ok(RepairReport {
            clean_after: true,
            ..RepairReport::default()
        });
    }

    // Distinct corrupt partitions, deduplicated and ordered.
    let mut corrupt: Vec<Partition> = scrub.errors.iter().map(|e| e.partition).collect();
    corrupt.sort_by_key(|p| coordinode_storage::placement::partition_wire_tag(*p));
    corrupt.dedup();

    let Some(checkpoint) = latest_checkpoint(checkpoint_root) else {
        tracing::error!(
            ?corrupt,
            "corruption detected but no checkpoint exists; restore from backup"
        );
        return Ok(RepairReport {
            corrupt_partitions: corrupt,
            clean_after: false,
            ..RepairReport::default()
        });
    };

    // One cursor + oplog slice covers every partition's replay-forward.
    let from = StorageEngine::checkpoint_oplog_cursor(&checkpoint)?;
    let oplog_since = engine.oplog_read_since(from)?.unwrap_or_default();

    let mut repaired = Vec::with_capacity(corrupt.len());
    for &part in &corrupt {
        engine.repair_partition_from_checkpoint(&checkpoint, &oplog_since, part)?;
        repaired.push(part);
    }

    let after = scrub_all(engine, &ScrubConfig::default())?;
    Ok(RepairReport {
        corrupt_partitions: corrupt,
        repaired,
        checkpoint_used: Some(checkpoint),
        clean_after: !after.has_errors(),
    })
}

/// Tuning for the optional background checkpoint scheduler.
#[derive(Debug, Clone)]
pub struct CheckpointSchedulerConfig {
    /// How often to take a checkpoint.
    pub interval: Duration,
    /// How many recent checkpoints to retain (older ones are pruned).
    pub keep: usize,
}

impl Default for CheckpointSchedulerConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(3600),
            keep: 3,
        }
    }
}

/// A background thread that periodically checkpoints the engine, prunes old
/// checkpoints, and purges expired oplog segments. Opt-in — an embedded app
/// constructs one and holds it; dropping it stops and joins the thread.
///
/// Off by default: deployments that do not want a repair base (or take
/// checkpoints manually via [`Database::checkpoint`](crate::Database::checkpoint))
/// simply never start it.
pub struct CheckpointScheduler {
    shutdown: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl CheckpointScheduler {
    /// Start the scheduler against `engine` rooted at `data_dir`.
    pub fn start(
        engine: Arc<StorageEngine>,
        data_dir: PathBuf,
        cfg: CheckpointSchedulerConfig,
    ) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let stop = Arc::clone(&shutdown);
        let root = checkpoint_root(&data_dir);
        // Sleep in short ticks so shutdown is responsive regardless of interval.
        let tick = Duration::from_millis(200);
        let handle = std::thread::Builder::new()
            .name("coordinode-checkpoint".into())
            .spawn(move || {
                let mut since_last = Duration::ZERO;
                while !stop.load(Ordering::Relaxed) {
                    if since_last >= cfg.interval {
                        since_last = Duration::ZERO;
                        if let Err(e) = create_checkpoint(&engine, &root) {
                            tracing::error!(error = %e, "scheduled checkpoint failed");
                        }
                        if let Err(e) = prune_checkpoints(&root, cfg.keep) {
                            tracing::warn!(error = %e, "checkpoint prune failed");
                        }
                        let now = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .map(|d| d.as_secs())
                            .unwrap_or(0);
                        if let Err(e) = engine.oplog_purge_expired(now) {
                            tracing::warn!(error = %e, "oplog purge failed");
                        }
                    }
                    std::thread::sleep(tick);
                    since_last += tick;
                }
            });
        let handle = match handle {
            Ok(h) => Some(h),
            Err(e) => {
                tracing::error!(error = %e, "failed to spawn checkpoint scheduler thread");
                None
            }
        };
        Self { shutdown, handle }
    }
}

impl Drop for CheckpointScheduler {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
