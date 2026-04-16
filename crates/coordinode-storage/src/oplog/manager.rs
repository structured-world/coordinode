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
    /// Scans existing `oplog-*.bin` files and registers them as sealed segments.
    /// The active writer starts as `None`; it is created lazily on first [`append`](Self::append).
    pub fn open(
        dir: &Path,
        shard_id: ShardId,
        max_bytes: u64,
        max_entries: u32,
        retention_secs: u64,
    ) -> StorageResult<Self> {
        std::fs::create_dir_all(dir)
            .map_err(|e| StorageError::Io(format!("create oplog dir {:?}: {e}", dir)))?;

        let mut paths: Vec<(u64, PathBuf)> = std::fs::read_dir(dir)
            .map_err(|e| StorageError::Io(format!("read oplog dir {:?}: {e}", dir)))?
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let p = e.path();
                let idx = parse_first_index(&p)?;
                Some((idx, p))
            })
            .collect();

        paths.sort_by_key(|&(idx, _)| idx);

        Ok(Self {
            dir: dir.to_path_buf(),
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
    /// The sealed file is added to [`sealed`](Self::sealed). The next
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
    /// `now_secs` — current Unix time in seconds.
    ///
    /// HLC timestamps use the upper 46 bits for wall-clock milliseconds, so
    /// the comparison is: `segment.last_ts < cutoff_hlc` where
    /// `cutoff_hlc = (now_secs - retention_secs) * 1000 << 18`.
    ///
    /// Returns the number of segments removed.
    pub fn purge_expired(&mut self, now_secs: u64) -> StorageResult<usize> {
        let cutoff_ms = now_secs
            .saturating_sub(self.retention_secs)
            .saturating_mul(1_000);
        // HLC: wall-clock ms occupies the upper bits (shift by 18 for the logical counter).
        let cutoff_hlc = cutoff_ms << 18;

        let mut purged = 0usize;
        let mut remaining = Vec::new();

        for (first_idx, path) in self.sealed.drain(..) {
            let reader = SegmentReader::open(&path)?;
            if reader.footer.last_ts < cutoff_hlc {
                std::fs::remove_file(&path).map_err(|e| {
                    StorageError::Io(format!("remove expired segment {:?}: {e}", path))
                })?;
                purged += 1;
            } else {
                remaining.push((first_idx, path));
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

    /// Delete sealed segments where every entry has `index < raft_index`.
    ///
    /// A segment is eligible when `first_index + entry_count <= raft_index`
    /// (i.e., its last entry index, inclusive, is strictly less than `raft_index`).
    ///
    /// Used by the Raft log storage layer to purge log entries that have been
    /// applied and snapshotted — equivalent to openraft's `purge_log`.
    ///
    /// Returns the number of segments removed.
    pub fn purge_before(&mut self, raft_index: u64) -> StorageResult<usize> {
        let mut purged = 0usize;
        let mut remaining = Vec::new();

        for (first_idx, path) in self.sealed.drain(..) {
            let reader = SegmentReader::open(&path)?;
            // next_index is the exclusive upper bound for this segment's entries.
            let next_index = first_idx + reader.footer.entry_count as u64;
            if next_index <= raft_index {
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
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::oplog::entry::OplogOp;

    fn make_entry(index: u64, ts: u64) -> OplogEntry {
        OplogEntry {
            ts,
            term: 1,
            index,
            shard: 0,
            ops: vec![OplogOp::Insert {
                partition: 1,
                key: format!("k{index}").into_bytes(),
                value: b"v".to_vec(),
            }],
            is_migration: false,
            pre_images: None,
        }
    }

    fn test_manager(dir: &Path) -> OplogManager {
        OplogManager::open(
            dir,
            0,                // shard_id
            64 * 1024 * 1024, // max_bytes
            50_000,           // max_entries
            7 * 24 * 3600,    // retention_secs
        )
        .expect("open manager")
    }

    #[test]
    fn append_and_read_range() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut mgr = test_manager(dir.path());

        for i in 0..10u64 {
            mgr.append(&make_entry(i, 1000 + i)).expect("append");
        }
        mgr.rotate().expect("rotate");

        let entries = mgr.read_range(0, 10).expect("read_range");
        assert_eq!(entries.len(), 10);
        for (i, e) in entries.iter().enumerate() {
            assert_eq!(e.index, i as u64, "index mismatch at position {i}");
        }
    }

    #[test]
    fn read_range_partial_window() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut mgr = test_manager(dir.path());

        for i in 0..20u64 {
            mgr.append(&make_entry(i, 2000 + i)).expect("append");
        }
        mgr.rotate().expect("rotate");

        let entries = mgr.read_range(5, 15).expect("read_range");
        assert_eq!(entries.len(), 10);
        assert_eq!(entries[0].index, 5);
        assert_eq!(entries[9].index, 14);
    }

    #[test]
    fn read_range_forces_rotation_of_active_writer() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut mgr = test_manager(dir.path());

        for i in 0..5u64 {
            mgr.append(&make_entry(i, 3000 + i)).expect("append");
        }
        // Current writer is NOT explicitly rotated — read_range must seal it.
        let entries = mgr.read_range(0, 5).expect("read_range");
        assert_eq!(entries.len(), 5);
    }

    #[test]
    fn rotation_creates_new_segment_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut mgr = test_manager(dir.path());

        for i in 0..5u64 {
            mgr.append(&make_entry(i, 4000 + i)).expect("append");
        }
        mgr.rotate().expect("rotate first");
        assert_eq!(mgr.sealed.len(), 1);

        for i in 5..10u64 {
            mgr.append(&make_entry(i, 5000 + i)).expect("append");
        }
        mgr.rotate().expect("rotate second");
        assert_eq!(mgr.sealed.len(), 2);
    }

    #[test]
    fn auto_rotation_on_entry_limit() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut mgr = OplogManager::open(
            dir.path(),
            0,
            64 * 1024 * 1024,
            3, // max_entries = 3
            7 * 24 * 3600,
        )
        .expect("open");

        // entries 0-2 → segment 1 (auto-rotate when writing 3)
        // entries 3-5 → segment 2 (auto-rotate when writing 6)
        // entry   6   → current writer
        for i in 0..7u64 {
            mgr.append(&make_entry(i, 5000 + i)).expect("append");
        }
        assert_eq!(mgr.sealed.len(), 2, "should have 2 sealed segments");

        // read_range seals entry 6 → 3 sealed total
        let entries = mgr.read_range(0, 7).expect("read_range");
        assert_eq!(entries.len(), 7);
        assert_eq!(entries[6].index, 6);
    }

    #[test]
    fn auto_rotation_on_byte_limit() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Very small byte limit: 1 byte forces rotation on every append after first
        let mut mgr = OplogManager::open(dir.path(), 0, 1, 50_000, 7 * 24 * 3600).expect("open");

        for i in 0..4u64 {
            mgr.append(&make_entry(i, 6000 + i)).expect("append");
        }
        // Each entry > 1 byte, so after writing entry 0 (total_bytes > 1),
        // entries 1, 2, 3 each trigger a rotation before being written.
        // Sealed: segments for entries 0, 1, 2 = 3 sealed, current has entry 3.
        assert_eq!(mgr.sealed.len(), 3);

        let entries = mgr.read_range(0, 4).expect("read_range");
        assert_eq!(entries.len(), 4);
    }

    #[test]
    fn purge_expired_removes_old_segments() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut mgr =
            OplogManager::open(dir.path(), 0, 64 * 1024 * 1024, 50_000, 3600).expect("open");

        // Entries with ts = 100ms in HLC (100 << 18)
        let old_ts = 100u64 << 18;
        for i in 0..3u64 {
            let mut e = make_entry(i, old_ts);
            e.ts = old_ts;
            mgr.append(&e).expect("append old");
        }
        mgr.rotate().expect("rotate");
        assert_eq!(mgr.sealed.len(), 1);

        // now_secs=10000; cutoff = 10000-3600 = 6400s = 6_400_000ms
        // old segment last_ts ≈ 100ms << 18 ≪ 6_400_000ms << 18 → purged
        let purged = mgr.purge_expired(10_000).expect("purge");
        assert_eq!(purged, 1, "one expired segment should be removed");
        assert_eq!(mgr.sealed.len(), 0);
    }

    #[test]
    fn purge_keeps_recent_segments() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut mgr =
            OplogManager::open(dir.path(), 0, 64 * 1024 * 1024, 50_000, 3600).expect("open");

        // Entries with ts at "now" = 10000s = 10_000_000ms
        let recent_ts = 10_000_000u64 << 18;
        for i in 0..3u64 {
            let mut e = make_entry(i, recent_ts);
            e.ts = recent_ts;
            mgr.append(&e).expect("append recent");
        }
        mgr.rotate().expect("rotate");

        let purged = mgr.purge_expired(10_000).expect("purge");
        assert_eq!(purged, 0, "recent segment must not be purged");
        assert_eq!(mgr.sealed.len(), 1);
    }

    #[test]
    fn verify_all_passes_on_valid_data() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut mgr = test_manager(dir.path());

        for i in 0..5u64 {
            mgr.append(&make_entry(i, 9000 + i)).expect("append");
        }
        mgr.rotate().expect("rotate");

        let count = mgr.verify_all().expect("verify_all");
        assert_eq!(count, 1);
    }

    #[test]
    fn empty_manager_reads_empty_range() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut mgr = test_manager(dir.path());

        let entries = mgr.read_range(0, 100).expect("read_range on empty");
        assert!(entries.is_empty());
    }

    #[test]
    fn purge_before_removes_fully_covered_segments() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut mgr = OplogManager::open(dir.path(), 0, 64 * 1024 * 1024, 5, 86400).expect("open");

        // Segment 1: entries 0-4 (first_idx=0, entry_count=5, next_index=5)
        for i in 0..5u64 {
            mgr.append(&make_entry(i, 1000 + i)).expect("append");
        }
        mgr.rotate().expect("rotate");

        // Segment 2: entries 5-9 (first_idx=5, entry_count=5, next_index=10)
        for i in 5..10u64 {
            mgr.append(&make_entry(i, 2000 + i)).expect("append");
        }
        mgr.rotate().expect("rotate");

        assert_eq!(mgr.sealed.len(), 2);

        // purge_before(5): seg1 next_index=5 <= 5 → purged; seg2 next_index=10 > 5 → kept.
        let purged = mgr.purge_before(5).expect("purge_before");
        assert_eq!(purged, 1, "only the first segment should be purged");
        assert_eq!(mgr.sealed.len(), 1, "second segment must remain");

        // Entries 5-9 must still be readable.
        let entries = mgr.read_range(5, 10).expect("read after purge");
        assert_eq!(entries.len(), 5);
        assert_eq!(entries[0].index, 5);
        assert_eq!(entries[4].index, 9);
    }

    #[test]
    fn truncate_all_clears_all_segments_including_active() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut mgr = test_manager(dir.path());

        // Two sealed segments + an active (unsealed) writer.
        for i in 0..5u64 {
            mgr.append(&make_entry(i, 1000 + i)).expect("append");
        }
        mgr.rotate().expect("rotate first");
        for i in 5..10u64 {
            mgr.append(&make_entry(i, 2000 + i)).expect("append");
        }
        mgr.rotate().expect("rotate second");
        for i in 10..15u64 {
            mgr.append(&make_entry(i, 3000 + i)).expect("append");
        }
        assert_eq!(mgr.sealed.len(), 2);
        assert!(mgr.current.is_some());

        mgr.truncate_all().expect("truncate_all");

        assert_eq!(mgr.sealed.len(), 0, "all sealed segments must be removed");
        assert!(
            mgr.current.is_none(),
            "active writer must be gone after seal+delete"
        );

        // No .bin files should remain on disk.
        let bin_files: Vec<_> = std::fs::read_dir(dir.path())
            .expect("readdir")
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "bin"))
            .collect();
        assert!(
            bin_files.is_empty(),
            "no segment files should remain on disk"
        );
    }

    #[test]
    fn read_range_across_multiple_segments() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut mgr = OplogManager::open(dir.path(), 0, 64 * 1024 * 1024, 5, 86400).expect("open");

        // Writes 15 entries, rotating every 5
        for i in 0..15u64 {
            mgr.append(&make_entry(i, 7000 + i)).expect("append");
        }
        mgr.rotate().expect("rotate");
        assert_eq!(mgr.sealed.len(), 3);

        // Read a window that spans two segments
        let entries = mgr.read_range(3, 12).expect("read_range");
        assert_eq!(entries.len(), 9);
        assert_eq!(entries[0].index, 3);
        assert_eq!(entries[8].index, 11);
    }
}
