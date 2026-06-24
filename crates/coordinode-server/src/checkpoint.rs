//! Periodic local checkpoints — the base for WAL-replay repair.
//!
//! A checkpoint is a hard-link snapshot of every partition tree plus the oplog
//! directory ([`StorageEngine::create_checkpoint`]). WAL-replay repair (the
//! fallback when no healthy replica can serve a corrupt partition) rebuilds the
//! partition by opening the most recent checkpoint and replaying the oplog
//! forward, so a fresh checkpoint must always exist — hence the periodic
//! scheduler. Each node checkpoints its own local storage independently (no
//! leader election); the work is blocking I/O and runs off the async runtime.

use std::path::{Path, PathBuf};

use coordinode_storage::engine::core::StorageEngine;

/// Directory-name prefix for a checkpoint; the suffix is a zero-padded unix-time
/// tag so lexical order is chronological.
const CHECKPOINT_PREFIX: &str = "ckpt-";

/// Take one checkpoint into `dir/ckpt-<now_secs>` and prune to the newest `keep`.
///
/// `now_secs` is supplied by the caller (not read from the clock here) so the
/// cycle is deterministic and testable. `keep` is clamped to at least 1 so the
/// checkpoint just written is never pruned. Returns the new checkpoint's path.
///
/// # Errors
/// If the directory cannot be created, the engine checkpoint fails (e.g. a
/// multi-endpoint engine, which checkpoint does not support yet), or pruning an
/// old checkpoint fails.
pub fn run_checkpoint_cycle(
    engine: &StorageEngine,
    dir: &Path,
    keep: usize,
    now_secs: u64,
) -> Result<PathBuf, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("create checkpoint dir {dir:?}: {e}"))?;
    let target = dir.join(format!("{CHECKPOINT_PREFIX}{now_secs:020}"));
    engine
        .create_checkpoint(&target)
        .map_err(|e| e.to_string())?;
    prune_checkpoints(dir, keep.max(1))?;
    Ok(target)
}

/// Remove all but the newest `keep` `ckpt-*` directories under `dir`. Newest by
/// name, which is chronological because the tag is a zero-padded unix time.
///
/// # Errors
/// If the directory cannot be listed or an old checkpoint cannot be removed.
pub fn prune_checkpoints(dir: &Path, keep: usize) -> Result<(), String> {
    let mut ckpts = list_checkpoints(dir)?;
    ckpts.sort();
    if ckpts.len() > keep {
        for old in &ckpts[..ckpts.len() - keep] {
            std::fs::remove_dir_all(old).map_err(|e| format!("prune checkpoint {old:?}: {e}"))?;
        }
    }
    Ok(())
}

/// The most recent checkpoint directory under `dir` (newest by name, which is
/// chronological), or `None` when there is none. The base for WAL-replay repair.
#[must_use]
pub fn latest_checkpoint(dir: &Path) -> Option<PathBuf> {
    let mut ckpts = list_checkpoints(dir).ok()?;
    ckpts.sort();
    ckpts.pop()
}

/// All `ckpt-*` directories directly under `dir` (unsorted).
fn list_checkpoints(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let entries =
        std::fs::read_dir(dir).map_err(|e| format!("read checkpoint dir {dir:?}: {e}"))?;
    Ok(entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.is_dir()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with(CHECKPOINT_PREFIX))
        })
        .collect())
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests;
