//! [`OplogManager`]: lifecycle management for oplog segment files.
//!
//! Responsibilities:
//! - Append entries to the active segment
//! - Auto-rotate when size/count limits are reached
//! - Scan entries across segments for a given index range
//! - Purge segments whose HLC timestamps fall outside the retention window
//! - Verify checksums of all sealed segments

use std::path::{Path, PathBuf};

use crate::error::{StorageError, StorageResult};
use crate::oplog::entry::{OplogEntry, ShardId};
use crate::oplog::segment::{SegmentReader, SegmentWriter};

// ── Filename helpers ──────────────────────────────────────────────────────────

/// Segment filename: `oplog-{first_index:020}.bin`
///
/// Zero-padded so lexicographic order = index order.
fn segment_path(dir: &Path, first_index: u64) -> PathBuf {
    dir.join(format!("oplog-{first_index:020}.bin"))
}

/// Parse the `first_index` embedded in a segment filename.
fn parse_first_index(path: &Path) -> Option<u64> {
    let stem = path.file_stem()?.to_str()?;
    let idx_str = stem.strip_prefix("oplog-")?;
    idx_str.parse().ok()
}

// ── OplogManager ──────────────────────────────────────────────────────────────

/// Manages the oplog segment lifecycle for one shard.
pub struct OplogManager {
    dir: PathBuf,
    shard_id: ShardId,
    /// Maximum entry bytes per segment before forced rotation.
    max_bytes: u64,
    /// Maximum entries per segment before forced rotation.
    max_entries: u32,
    /// Retention window in seconds. Segments with `last_ts` older than
    /// `now - retention_secs` are eligible for purge.
    retention_secs: u64,
    /// Currently-open writer. `None` between rotations.
    current: Option<SegmentWriter>,
    /// `(first_index, path)` of sealed segments, sorted by first_index.
    pub(crate) sealed: Vec<(u64, PathBuf)>,
}

impl OplogManager {
    /// Open (or create) the oplog directory for `shard_id`.
    ///
    /// Scans existing `oplog-*.bin` files in `dir` and registers them as
    /// sealed segments. The active writer starts as `None`; it is created
    /// lazily on first [`append`](Self::append).
    ///
    /// New segments are written to `dir` only. For multi-endpoint setups
    /// where sealed segments may exist on additional endpoints (left over
    /// from a previous config-driven routing), use [`Self::open_multi`].
    pub fn open(
        dir: &Path,
        shard_id: ShardId,
        max_bytes: u64,
        max_entries: u32,
        retention_secs: u64,
    ) -> StorageResult<Self> {
        Self::open_multi(dir, &[], shard_id, max_bytes, max_entries, retention_secs)
    }

    /// Open the oplog manager with an active write directory plus extra
    /// directories scanned for sealed segments at startup
    /// ([storage-stack.md](../../arch/core/storage-stack.md) Layer 1↔2).
    ///
    /// `active_dir` receives all new segments. `recovery_dirs` are
    /// scanned at startup for `oplog-*.bin` files and merged into the
    /// `sealed` list in chronological order by `first_index`. The active
    /// dir is automatically included in the recovery scan — callers do
    /// NOT need to pass it twice.
    ///
    /// Use case: a config change re-routed the active oplog endpoint;
    /// sealed segments from the previous endpoint remain queryable
    /// through this scan-on-open without manual migration.
    pub fn open_multi(
        active_dir: &Path,
        recovery_dirs: &[PathBuf],
        shard_id: ShardId,
        max_bytes: u64,
        max_entries: u32,
        retention_secs: u64,
    ) -> StorageResult<Self> {
        std::fs::create_dir_all(active_dir)
            .map_err(|e| StorageError::Io(format!("create oplog dir {:?}: {e}", active_dir)))?;

        // Build the scan set: active_dir + recovery_dirs, deduplicated
        // by canonical path so callers can pass the active dir in the
        // recovery list without double-counting.
        let mut scan_dirs: Vec<PathBuf> = Vec::with_capacity(1 + recovery_dirs.len());
        scan_dirs.push(active_dir.to_path_buf());
        for d in recovery_dirs {
            if d != active_dir && !scan_dirs.contains(d) {
                scan_dirs.push(d.clone());
            }
        }

        let mut paths: Vec<(u64, PathBuf)> = Vec::new();
        for dir in &scan_dirs {
            // recovery dirs may not exist (operator pruned them) — skip
            // missing without error.
            if !dir.exists() {
                continue;
            }
            let entries = std::fs::read_dir(dir)
                .map_err(|e| StorageError::Io(format!("read oplog dir {:?}: {e}", dir)))?;
            for entry in entries.flatten() {
                let p = entry.path();
                if let Some(idx) = parse_first_index(&p) {
                    paths.push((idx, p));
                }
            }
        }

        paths.sort_by_key(|&(idx, _)| idx);

        // Reject duplicate first_index across endpoints — the operator
        // must clean up before the engine boots, since two segments at
        // the same first_index represent ambiguous fork history.
        if let Some(window) = paths.windows(2).find(|w| w[0].0 == w[1].0) {
            return Err(StorageError::Io(format!(
                "duplicate oplog segment first_index={} across endpoints: {:?} and {:?} \
                 — operator must reconcile (likely leftover from a previous \
                 config-driven oplog endpoint re-route)",
                window[0].0, window[0].1, window[1].1,
            )));
        }

        Ok(Self {
            dir: active_dir.to_path_buf(),
            shard_id,
            max_bytes,
            max_entries,
            retention_secs,
            current: None,
            sealed: paths,
        })
    }

    /// Append an entry to the active segment.
    ///
    /// Rotates to a new segment automatically when either `max_bytes` or
    /// `max_entries` is reached.
    pub fn append(&mut self, entry: &OplogEntry) -> StorageResult<()> {
        // Auto-rotate if the current segment is over limit
        if let Some(ref w) = self.current {
            if w.total_bytes() >= self.max_bytes || w.entry_count() >= self.max_entries {
                self.rotate()?;
            }
        }

        // Open a new segment on the entry's index
        if self.current.is_none() {
            let path = segment_path(&self.dir, entry.index);
            self.current = Some(SegmentWriter::create(&path, self.shard_id, entry.index)?);
        }

        let writer = self
            .current
            .as_mut()
            .ok_or_else(|| StorageError::Io("no active segment writer after create".to_string()))?;
        writer.append(entry)
    }

    /// Flush user-space buffer and fsync the active segment to storage.
    ///
    /// Must be called after each batch of appends to ensure entries are
    /// durable before `io_completed` is sent to the caller.
    ///
    /// If there is no active segment (nothing appended yet), this is a no-op.
    pub fn flush(&mut self) -> StorageResult<()> {
        if let Some(ref mut writer) = self.current {
            writer.flush_and_sync()
        } else {
            Ok(())
        }
    }

    /// Seal and close the active segment.
    ///
    /// The sealed file is added to the manager's sealed list. The next
    /// [`append`](Self::append) will create a fresh segment.
    pub fn rotate(&mut self) -> StorageResult<()> {
        let Some(writer) = self.current.take() else {
            return Ok(());
        };
        let path = writer.seal()?;
        if let Some(idx) = parse_first_index(&path) {
            self.sealed.push((idx, path));
            self.sealed.sort_by_key(|&(idx, _)| idx);
        }
        Ok(())
    }

    /// Seal the active segment and shut down.
    pub fn close(mut self) -> StorageResult<()> {
        self.rotate()
    }
}

impl Drop for OplogManager {
    /// Seal the active segment on drop so entries are recoverable on restart.
    ///
    /// Errors are silently ignored — `Drop` cannot propagate them. In crash
    /// scenarios the OS will kill the process before `Drop` runs anyway; the
    /// `LogStore::open()` recovery path handles that case.
    fn drop(&mut self) {
        if self.current.is_some() {
            let _ = self.rotate();
        }
    }
}

impl OplogManager {
    /// Return all entries with `index ∈ [from_index, to_index)`.
    ///
    /// **Side effect:** if there is an active (unsealed) writer, it is rotated
    /// first so its entries are visible. The next `append` will start a new
    /// segment.
    pub fn read_range(&mut self, from_index: u64, to_index: u64) -> StorageResult<Vec<OplogEntry>> {
        // Seal the active writer so we can read its entries from disk.
        if self.current.is_some() {
            self.rotate()?;
        }

        let mut result = Vec::new();

        for (first_idx, path) in &self.sealed {
            // Segments whose first_index >= to_index cannot contain entries in range.
            if *first_idx >= to_index {
                break;
            }

            let reader = SegmentReader::open(path)?;

            // Skip segments belonging to a different shard (defensive).
            if reader.header.shard_id != self.shard_id {
                continue;
            }

            for entry in reader.into_entries() {
                if entry.index >= from_index && entry.index < to_index {
                    result.push(entry);
                }
            }
        }

        result.sort_by_key(|e| e.index);
        Ok(result)
    }

    /// Delete segments whose `last_ts` falls outside the retention window.
    ///
    /// Time-only retention: equivalent to [`purge_with_floor`] with an
    /// unconstrained consumer floor (`u64::MAX`).
    ///
    /// [`purge_with_floor`]: Self::purge_with_floor
    pub fn purge_expired(&mut self, now_secs: u64) -> StorageResult<usize> {
        self.purge_with_floor(now_secs, u64::MAX)
    }

    /// Delete segments that are BOTH outside the time window AND fully below
    /// the consumer oplog-retention floor (ADR-028 feed b, logical OR keep):
    ///
    /// ```text
    /// keep segment  iff  last_ts within retention_secs   (time safety net)
    ///                OR  last_index >= oplog_index_floor  (a CDC consumer needs it)
    /// purge         iff  NOT kept
    /// ```
    ///
    /// `oplog_index_floor` is `min(checkpoint)` over `OplogEvents` consumers
    /// from the `SeqnoConsumerRegistry` (Raft-index space), or `u64::MAX` when
    /// no CDC consumer is registered — in which case only the time policy
    /// applies. A segment covering indices `[first_idx, first_idx +
    /// entry_count)` has `last_index = first_idx + entry_count - 1`.
    ///
    /// `now_secs` — current Unix time in seconds. HLC `last_ts` packs wall ms
    /// in the upper bits, so the cutoff is `(now_secs - retention_secs) * 1000
    /// << 18`.
    pub fn purge_with_floor(
        &mut self,
        now_secs: u64,
        oplog_index_floor: u64,
    ) -> StorageResult<usize> {
        let cutoff_ms = now_secs
            .saturating_sub(self.retention_secs)
            .saturating_mul(1_000);
        // HLC: wall-clock ms occupies the upper bits (shift by 18 for the logical counter).
        let cutoff_hlc = cutoff_ms << 18;

        let mut purged = 0usize;
        let mut remaining = Vec::new();

        for (first_idx, path) in self.sealed.drain(..) {
            let reader = SegmentReader::open(&path)?;
            let within_window = reader.footer.last_ts >= cutoff_hlc;
            // last_index = first_idx + entry_count - 1; a segment with at least
            // one entry whose last index reaches the floor is still needed.
            let last_index =
                first_idx.saturating_add(u64::from(reader.footer.entry_count).saturating_sub(1));
            let needed_by_consumer = last_index >= oplog_index_floor;
            if within_window || needed_by_consumer {
                remaining.push((first_idx, path));
            } else {
                std::fs::remove_file(&path).map_err(|e| {
                    StorageError::Io(format!("remove expired segment {:?}: {e}", path))
                })?;
                purged += 1;
            }
        }

        self.sealed = remaining;
        Ok(purged)
    }

    /// `true` if any segments (sealed or partial) are present on disk.
    pub fn has_segments(&self) -> bool {
        !self.sealed.is_empty()
    }

    /// Scan all segments from last to first and return the last valid
    /// [`OplogEntry`] found.
    ///
    /// Used during crash recovery: when the persisted `last_log_id` LSM key is
    /// missing (process died after fsync but before the LSM write), this method
    /// reconstructs the last entry from the segment files — including segments
    /// that were never sealed (no footer) but whose entries were fsynced.
    ///
    /// Returns `Ok(None)` if no valid entries are found in any segment.
    pub fn recover_last_entry(&self) -> StorageResult<Option<OplogEntry>> {
        for (_, path) in self.sealed.iter().rev() {
            // Fast path: normal sealed segment with a valid footer.
            let entries = match SegmentReader::open(path) {
                Ok(r) => r.into_entries(),
                // Slow path: unsealed/partial segment (crashed before seal).
                Err(_) => {
                    SegmentReader::scan_without_footer(path, self.shard_id).unwrap_or_default()
                }
            };
            if let Some(last) = entries.into_iter().last() {
                return Ok(Some(last));
            }
        }
        Ok(None)
    }

    /// Number of sealed segments on disk.
    pub fn sealed_segment_count(&self) -> usize {
        self.sealed.len()
    }

    /// Paths of all sealed segments in ascending index order.
    pub fn sealed_segment_paths(&self) -> Vec<&std::path::Path> {
        self.sealed.iter().map(|(_, p)| p.as_path()).collect()
    }

    /// Verify checksums of all sealed segments.
    ///
    /// Returns the number of segments verified. Returns the first error
    /// encountered if any segment is corrupt.
    pub fn verify_all(&self) -> StorageResult<usize> {
        for (_, path) in &self.sealed {
            // SegmentReader::open validates header magic, all entry crc32s, and footer crc32.
            let _ = SegmentReader::open(path)?;
        }
        Ok(self.sealed.len())
    }

    /// Delete sealed segments whose entries are both delivered to the state
    /// machine and durably stored in SST form.
    ///
    /// Two gates must hold for every entry in a segment before it can be removed:
    ///
    /// 1. **Applied:** the segment's exclusive last index is at most
    ///    `applied_index` — every entry has been passed to `apply()`. The
    ///    check is identical in single-node and clustered deployments: the
    ///    log store always runs under openraft (single-node uses a stub
    ///    network with member set `{1}`, so commit and apply are immediate),
    ///    so `applied_index` is just `state_machine.last_applied().index + 1`.
    /// 2. **State-machine durable:** the segment's `last_ts` (HLC commit_ts
    ///    of the last entry, recorded in the footer at seal time) is at most
    ///    `safe_ts`, the smallest `get_highest_persisted_seqno()` across all
    ///    LSM partitions in the engine.
    ///
    /// Gate (2) closes the crash-safety hole left by gate (1) alone: openraft
    /// drives apply→save_applied→purge ordering correctly, but apply writes
    /// land in per-partition memtables that flush independently. Without
    /// gate (2), a partition whose flush schedule lags behind the smallest
    /// (typically `Schema`, which fills fast because every applied entry
    /// updates the `applied_index` key there) can lose its memtable on kernel
    /// crash while the oplog segment that would replay those mutations has
    /// already been removed. Holding the segment until
    /// `min_partition_flushed_seqno ≥ last_ts` guarantees the openraft
    /// "re-delivers missing entries on restart" contract is satisfiable.
    ///
    /// `safe_ts == 0` (fresh engine, no SST yet) makes the second gate
    /// impossible to satisfy for any non-empty segment, so nothing is purged
    /// — exactly the behavior we want during startup.
    ///
    /// Returns the number of segments removed. Skipping segments is not an
    /// error: openraft retries purge on subsequent snapshots/heartbeats.
    pub fn purge_before(&mut self, applied_index: u64, safe_ts: u64) -> StorageResult<usize> {
        let mut purged = 0usize;
        let mut remaining = Vec::new();

        for (first_idx, path) in self.sealed.drain(..) {
            let reader = SegmentReader::open(&path)?;
            // next_index is the exclusive upper bound for this segment's entries.
            let next_index = first_idx + reader.footer.entry_count as u64;
            let applied = next_index <= applied_index;
            let state_durable = reader.footer.last_ts <= safe_ts;
            if applied && state_durable {
                std::fs::remove_file(&path).map_err(|e| {
                    StorageError::Io(format!("remove purged segment {:?}: {e}", path))
                })?;
                purged += 1;
            } else {
                remaining.push((first_idx, path));
            }
        }

        self.sealed = remaining;
        Ok(purged)
    }

    /// Delete all sealed segments.
    ///
    /// If there is an active writer it is sealed first, then all sealed
    /// segments (including the newly sealed one) are removed. The manager is
    /// left in a clean empty state, ready for new appends.
    ///
    /// Used during Raft log truncation: `truncate_after(index)` reads the
    /// entries to keep, calls `truncate_all`, then re-appends the kept entries.
    pub fn truncate_all(&mut self) -> StorageResult<()> {
        if self.current.is_some() {
            self.rotate()?;
        }
        for (_, path) in self.sealed.drain(..) {
            std::fs::remove_file(&path).map_err(|e| {
                StorageError::Io(format!(
                    "remove segment during truncate_all {:?}: {e}",
                    path
                ))
            })?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic, clippy::cloned_ref_to_slice_refs)]
mod tests;
