//! [`OplogManager`]: lifecycle management for oplog segment files.
//!
//! Responsibilities:
//! - Append entries to the active segment
//! - Auto-rotate when size/count limits are reached
//! - Scan entries across segments for a given index range
//! - Purge segments whose HLC timestamps fall outside the retention window
//! - Verify checksums of all sealed segments

use std::path::{Path, PathBuf};

use crate::engine::config::SyncMethod;
use crate::error::{StorageError, StorageResult};
use crate::oplog::entry::{OplogEntry, ShardId};
use crate::oplog::segment::{SegmentReader, SegmentWriter, TailRecovery};

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
    /// How appends to new segments are made durable.
    sync: SyncMethod,
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
    /// directories scanned for sealed segments at startup (segments written
    /// under an earlier endpoint routing).
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

        // Only the newest segment can have been open for writing when the
        // process died: every earlier one was sealed at rotation. Seal it now,
        // so everything below treats the list as what it claims to be.
        if let Some((_, tail)) = paths.last() {
            match SegmentWriter::recover_tail(tail, shard_id)? {
                TailRecovery::Sealed => {}
                TailRecovery::Resealed {
                    entries,
                    discarded_bytes,
                } => tracing::warn!(
                    segment = %tail.display(),
                    entries,
                    discarded_bytes,
                    "oplog: sealed a segment left open by an unclean shutdown"
                ),
                TailRecovery::Removed => {
                    tracing::warn!(
                        segment = %tail.display(),
                        "oplog: removed a segment that an unclean shutdown left without entries"
                    );
                    paths.pop();
                }
            }
        }

        Ok(Self {
            dir: active_dir.to_path_buf(),
            shard_id,
            max_bytes,
            max_entries,
            retention_secs,
            current: None,
            sealed: paths,
            sync: SyncMethod::default(),
        })
    }

    /// Make appends durable under `sync` from the next segment on; the
    /// default is [`SyncMethod::Full`]. Set it right after opening.
    #[must_use]
    pub fn with_sync_method(mut self, sync: SyncMethod) -> Self {
        self.sync = sync;
        self
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
            // Reads locate a segment by its first index, so a new segment
            // must start above every earlier one. A header-only leftover at
            // this very index is replaced below and does not count.
            if let Some((first, earlier)) = self.sealed.last() {
                if entry.index <= *first && *earlier != path {
                    return Err(StorageError::Io(format!(
                        "oplog entry {} appended below segment {:?}",
                        entry.index, earlier
                    )));
                }
            }
            let writer = SegmentWriter::create_or_replace_empty(
                &path,
                self.shard_id,
                entry.index,
                self.sync,
            )?;
            // `open` registers every file it finds as sealed, including a
            // header-only leftover from a crash. That file has just been
            // replaced by the active writer, so drop the stale registration or
            // reads would visit the same path twice and return each entry
            // twice.
            self.sealed.retain(|(_, sealed_path)| sealed_path != &path);
            self.current = Some(writer);
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

    /// [`flush`](Self::flush) split in two: hand the active segment's
    /// appends to the kernel now and return what makes them durable, to sync
    /// without holding this manager. Entries in segments rotated out since
    /// the last sync are already durable (sealing syncs them). `None` when
    /// no segment is active.
    ///
    /// # Errors
    ///
    /// The flush fails, or the segment cannot be duplicated for the handle.
    pub fn flush_to_os(&mut self) -> StorageResult<Option<crate::oplog::SyncHandle>> {
        self.current
            .as_mut()
            .map(crate::oplog::SegmentWriter::flush_to_os)
            .transpose()
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
    /// Return all entries with `index ∈ [from_index, to_index)`, ascending.
    ///
    /// Only the segments that can hold the range are read, and the active
    /// segment is read in place: a read of the tail costs what it returns,
    /// however long the log is.
    pub fn read_range(&mut self, from_index: u64, to_index: u64) -> StorageResult<Vec<OplogEntry>> {
        let mut result = Vec::new();

        // Segments are ordered by first index and each holds the indexes up
        // to the next one's first, so the range starts in the last segment
        // that begins at or before `from_index`.
        let start = match self
            .sealed
            .partition_point(|&(first_idx, _)| first_idx <= from_index)
        {
            // Every segment begins after `from_index`: read from the first.
            0 => 0,
            n => n - 1,
        };
        for (first_idx, path) in &self.sealed[start..] {
            if *first_idx >= to_index {
                break;
            }

            let reader = SegmentReader::open(path)?;

            // Skip segments belonging to a different shard (defensive).
            if reader.header.shard_id != self.shard_id {
                continue;
            }

            result.extend(
                reader
                    .into_entries()
                    .into_iter()
                    .filter(|entry| entry.index >= from_index && entry.index < to_index),
            );
        }

        if let Some(writer) = &mut self.current {
            result.extend(writer.read_range(from_index, to_index)?);
        }
        Ok(result)
    }

    /// Delete segments that are outside the time window, fully below the
    /// consumer oplog-retention floor, and hold only durable entries
    /// (logical OR keep):
    ///
    /// ```text
    /// keep segment  iff  last_ts within retention_secs   (time safety net)
    ///                OR  last_index >= oplog_index_floor  (a consumer needs it)
    ///                OR  any entry !is_durable(entry)      (crash recovery needs it)
    /// purge         iff  NOT kept
    /// ```
    ///
    /// `oplog_index_floor` is the minimum over registered consumers'
    /// checkpoints (index space): CDC consumers, and the WAL-replay-repair
    /// floor (the latest checkpoint's replay cursor). `u64::MAX` when no
    /// consumer is registered.
    ///
    /// `is_durable` answers whether one entry's writes are all persisted in
    /// their partition trees. A record whose only copy is the journal (its
    /// write still lives in a memtable) must survive any purge, or a crash
    /// after the purge silently loses it. The predicate is consulted only for
    /// segments the other two conditions would drop, so its cost is bounded
    /// by the data actually about to be deleted. Pass `&|_| true` when every
    /// journaled write is known durable (tests, post-persist maintenance).
    ///
    /// `now_secs` — current Unix time in seconds. HLC `last_ts` packs wall ms
    /// in the upper bits, so the cutoff is `(now_secs - retention_secs) * 1000
    /// << 18`.
    pub fn purge_with_floor(
        &mut self,
        now_secs: u64,
        oplog_index_floor: u64,
        is_durable: &dyn Fn(&OplogEntry) -> bool,
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
            // Only a segment already outside both keep conditions pays for
            // the entry scan; SegmentReader::open loaded the entries above.
            let holds_non_durable = !within_window
                && !needed_by_consumer
                && reader.entries().iter().any(|e| !is_durable(e));
            if within_window || needed_by_consumer || holds_non_durable {
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

    /// Delete the sealed segments whose every entry is below `below`.
    ///
    /// The caller decides what `below` may be: for the Raft log, the lowest
    /// index every partition tree durably records as applied (its coverage
    /// base on disk), capped by what openraft asked to purge. Segment
    /// granularity keeps a segment that straddles the bound.
    ///
    /// Returns the number of segments removed.
    pub fn purge_before(&mut self, below: u64) -> StorageResult<usize> {
        // Segments hold ascending indexes, so the purgeable ones are a prefix.
        let mut purgeable = 0usize;
        for (i, (first_idx, path)) in self.sealed.iter().enumerate() {
            // Exclusive upper bound of this segment's entries: the next
            // segment starts above every one of them, so its first index
            // bounds them without reading this one. Only the newest segment,
            // with nothing after it, has its footer read.
            let successor = match self.sealed.get(i + 1) {
                Some(&(next_first, _)) => Some(next_first),
                None => self
                    .current
                    .as_ref()
                    .and_then(SegmentWriter::first_entry_index),
            };
            let next_index = match successor {
                Some(next_first) => next_first,
                None => first_idx + u64::from(SegmentReader::open(path)?.footer.entry_count),
            };
            if next_index > below {
                break;
            }
            purgeable += 1;
        }

        for removed in 0..purgeable {
            let path = &self.sealed[removed].1;
            if let Err(e) = std::fs::remove_file(path) {
                let message = format!("remove purged segment {:?}: {e}", path);
                // Forget only what is gone, so the list still matches the disk.
                self.sealed.drain(..removed);
                return Err(StorageError::Io(message));
            }
        }
        self.sealed.drain(..purgeable);
        Ok(purgeable)
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
