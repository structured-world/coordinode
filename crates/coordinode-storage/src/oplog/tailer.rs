//! [`OplogTailer`]: read-only oplog cursor for CDC consumers.
//!
//! The tailer reads sealed segment files directly from the oplog directory
//! without acquiring any lock on the `OplogManager`. Sealed segments are
//! immutable once written, so concurrent reads are safe.
//!
//! ## Usage
//!
//! ```rust,ignore
//! let token = ResumeToken { shard_id: 0, segment_id: 0, entry_offset: 0 };
//! let mut tailer = OplogTailer::new(&data_dir.join("oplog/0"), token);
//!
//! loop {
//!     let batch = tailer.read_next(256, &filters)?;
//!     if batch.is_empty() {
//!         // Caught up — wait for new sealed segments.
//!         std::thread::sleep(Duration::from_millis(100));
//!     } else {
//!         for (entry, token) in batch {
//!             process(entry);
//!             last_token = token;
//!         }
//!     }
//! }
//! ```
//!
//! ## Resume token
//!
//! [`ResumeToken`] identifies a position as `(segment_id, entry_offset)`:
//! - `segment_id` = the `first_index` encoded in the filename
//!   (`oplog-<segment_id:020>.bin`)
//! - `entry_offset` = number of entries already consumed from that segment
//!
//! A zero token (`segment_id=0, entry_offset=0`) means "start from the oldest
//! available segment".
//!
//! ## Filters
//!
//! [`CdcFilters`] supports:
//! - `is_migration`: include/exclude entries flagged as shard-migration traffic
//! - `edge_types`: only deliver entries that touch adjacency keys (`adj:`) for
//!   the given edge types — checked by key prefix (`adj:<TYPE>:`)
//!
//! Note: label-based server-side filtering is not yet supported because node
//! keys do not embed the label. Client-side filtering is recommended.

use std::path::{Path, PathBuf};

use crate::error::{StorageError, StorageResult};
use crate::oplog::entry::{OplogEntry, OplogOp, ShardId};
use crate::oplog::segment::SegmentReader;

// ── Public types ──────────────────────────────────────────────────────────────

/// Position within the oplog used to resume a CDC stream after disconnect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResumeToken {
    /// Shard this stream targets.
    pub shard_id: ShardId,
    /// `first_index` embedded in the segment filename. `0` = start of oldest segment.
    pub segment_id: u64,
    /// Entries already consumed from `segment_id`. Next event is at this offset.
    pub entry_offset: u64,
}

impl ResumeToken {
    /// Token that starts streaming from the oldest available segment.
    pub fn from_start(shard_id: ShardId) -> Self {
        Self {
            shard_id,
            segment_id: 0,
            entry_offset: 0,
        }
    }

    /// True if this is the "start of stream" sentinel.
    pub fn is_start(&self) -> bool {
        self.segment_id == 0 && self.entry_offset == 0
    }
}

/// Server-side filters applied before delivering events to the CDC client.
#[derive(Debug, Default, Clone)]
pub struct CdcFilters {
    /// When non-empty, only deliver entries whose ops touch an adjacency key
    /// (`adj:<TYPE>:out:` or `adj:<TYPE>:in:`) for at least one of these types.
    pub edge_types: Vec<String>,

    /// When `Some(v)`, only deliver entries where `is_migration == v`.
    /// `None` = deliver all entries regardless of the flag.
    pub is_migration: Option<bool>,
}

// ── OplogTailer ───────────────────────────────────────────────────────────────

/// Read-only cursor over sealed oplog segments.
///
/// Reads directly from segment files on disk — no lock on `OplogManager`.
/// Safe to run concurrently with the write path.
pub struct OplogTailer {
    /// Path to the oplog directory for one shard (`<data_dir>/oplog/<shard_id>/`).
    oplog_dir: PathBuf,
    shard_id: ShardId,
    /// Current position within the stream.
    position: ResumeToken,
}

impl OplogTailer {
    /// Create a new tailer starting from `token`.
    ///
    /// Pass `ResumeToken::from_start(shard_id)` to start from the oldest
    /// available segment.
    pub fn new(oplog_dir: &Path, token: ResumeToken) -> Self {
        let shard_id = token.shard_id;
        Self {
            oplog_dir: oplog_dir.to_path_buf(),
            shard_id,
            position: token,
        }
    }

    /// Current position (last acked token + 1 entry consumed).
    pub fn position(&self) -> &ResumeToken {
        &self.position
    }

    /// Read up to `max_entries` entries from the current position, applying
    /// `filters`.
    ///
    /// Returns a list of `(entry, token)` pairs. Each `token` identifies the
    /// position AFTER this entry — store it for reconnect.
    ///
    /// Returns an **empty vec** when caught up (no new sealed segments).
    /// The caller should sleep and retry.
    ///
    /// Side-effect: internal position advances as entries are consumed.
    pub fn read_next(
        &mut self,
        max_entries: usize,
        filters: &CdcFilters,
    ) -> StorageResult<Vec<(OplogEntry, ResumeToken)>> {
        let segments = self.list_segments()?;
        let mut result = Vec::new();

        for (seg_first_index, seg_path) in &segments {
            if result.len() >= max_entries {
                break;
            }

            // Skip segments we've already passed.
            if *seg_first_index < self.position.segment_id {
                continue;
            }

            let is_last = segments
                .last()
                .map(|(idx, _)| idx == seg_first_index)
                .unwrap_or(false);
            let reader = match SegmentReader::open(seg_path) {
                Ok(r) => r,
                // The newest segment is usually still being written (no
                // footer yet). Read its complete-entry prefix so live
                // consumers see entries without waiting up to a full
                // rotation for the seal; the per-entry crc framing makes
                // the prefix read safe.
                Err(_) if is_last => match SegmentReader::open_active(seg_path) {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::debug!(
                            segment = %seg_path.display(),
                            error = %e,
                            "active segment not yet readable"
                        );
                        continue;
                    }
                },
                Err(e) => {
                    // A non-final unreadable segment is real corruption or
                    // a mid-write straggler; skip and let later calls retry.
                    tracing::debug!(
                        segment = %seg_path.display(),
                        error = %e,
                        "skipping unreadable segment"
                    );
                    continue;
                }
            };

            // Skip entries already consumed in this segment.
            let skip = if *seg_first_index == self.position.segment_id {
                self.position.entry_offset as usize
            } else {
                0
            };

            let entries_in_seg = reader.entries();
            let to_read = entries_in_seg.iter().skip(skip);

            for (local_idx, entry) in to_read.enumerate() {
                if result.len() >= max_entries {
                    break;
                }

                if passes_filter(entry, filters) {
                    let next_offset = (skip + local_idx + 1) as u64;
                    let token = ResumeToken {
                        shard_id: self.shard_id,
                        segment_id: *seg_first_index,
                        entry_offset: next_offset,
                    };
                    result.push((entry.clone(), token));
                }
            }

            // Advance position past this segment if we consumed all its entries.
            let seg_entry_count = entries_in_seg.len() as u64;
            let new_offset = if *seg_first_index == self.position.segment_id {
                self.position.entry_offset + (entries_in_seg.len().saturating_sub(skip)) as u64
            } else {
                entries_in_seg.len() as u64
            };

            if new_offset >= seg_entry_count {
                // Move to the next segment on the next call.
                self.position = ResumeToken {
                    shard_id: self.shard_id,
                    segment_id: *seg_first_index,
                    entry_offset: seg_entry_count,
                };
            } else {
                self.position = ResumeToken {
                    shard_id: self.shard_id,
                    segment_id: *seg_first_index,
                    entry_offset: new_offset,
                };
                break; // Partial segment read — stop here.
            }
        }

        Ok(result)
    }

    // ── private ───────────────────────────────────────────────────────────────

    /// List sealed segment files in the oplog directory, sorted by first_index.
    fn list_segments(&self) -> StorageResult<Vec<(u64, PathBuf)>> {
        if !self.oplog_dir.exists() {
            return Ok(vec![]);
        }

        let mut segments: Vec<(u64, PathBuf)> = std::fs::read_dir(&self.oplog_dir)
            .map_err(|e| StorageError::Io(format!("list oplog dir {:?}: {e}", self.oplog_dir)))?
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let p = e.path();
                let stem = p.file_stem()?.to_str()?;
                let idx_str = stem.strip_prefix("oplog-")?;
                let idx: u64 = idx_str.parse().ok()?;
                Some((idx, p))
            })
            .collect();

        segments.sort_by_key(|&(idx, _)| idx);
        Ok(segments)
    }
}

// ── Filter logic ──────────────────────────────────────────────────────────────

/// Returns `true` if `entry` passes all active filters.
fn passes_filter(entry: &OplogEntry, filters: &CdcFilters) -> bool {
    // is_migration filter
    if let Some(expected_migration) = filters.is_migration {
        if entry.is_migration != expected_migration {
            return false;
        }
    }

    // edge_types filter: deliver entry only if at least one op touches an
    // adjacency key for the requested edge type.
    // Adj forward key: `adj:<TYPE>:out:<node_id BE>`
    // Adj reverse key: `adj:<TYPE>:in:<node_id BE>`
    if !filters.edge_types.is_empty() {
        let matches = entry.ops.iter().any(|op| {
            let key = match op {
                OplogOp::Insert { key, .. }
                | OplogOp::Delete { key, .. }
                | OplogOp::Merge { key, .. } => key.as_slice(),
                _ => return false,
            };
            if !key.starts_with(b"adj:") {
                return false;
            }
            // Check if the key starts with `adj:<TYPE>:` for any requested type.
            filters.edge_types.iter().any(|et| {
                let prefix = format!("adj:{et}:");
                key.starts_with(prefix.as_bytes())
            })
        });
        if !matches {
            return false;
        }
    }

    true
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::oplog::entry::{OplogEntry, OplogOp};
    use crate::oplog::manager::OplogManager;

    fn make_entry(index: u64, ts: u64, is_migration: bool) -> OplogEntry {
        OplogEntry {
            ts,
            term: 1,
            index,
            shard: 0,
            ops: vec![OplogOp::Insert {
                partition: 1,
                key: format!("node:k{index}").into_bytes(),
                value: b"v".to_vec(),
            }],
            is_migration,
            pre_images: None,
        }
    }

    fn make_adj_entry(index: u64, edge_type: &str) -> OplogEntry {
        OplogEntry {
            ts: 1000 + index,
            term: 1,
            index,
            shard: 0,
            ops: vec![OplogOp::Insert {
                partition: 2,
                key: format!("adj:{edge_type}:out:00000001").into_bytes(),
                value: b"postinglist".to_vec(),
            }],
            is_migration: false,
            pre_images: None,
        }
    }

    fn open_manager(dir: &std::path::Path) -> OplogManager {
        OplogManager::open(dir, 0, 64 * 1024 * 1024, 50_000, 7 * 24 * 3600).expect("open manager")
    }

    /// Seal the manager so all entries are in sealed segments readable by the tailer.
    fn seal_manager(mgr: &mut OplogManager) {
        mgr.rotate().expect("seal");
    }

    #[test]
    fn tailer_empty_dir_returns_empty() {
        let dir = tempfile::tempdir().expect("tempdir");
        let token = ResumeToken::from_start(0);
        let mut tailer = OplogTailer::new(dir.path(), token);
        let batch = tailer.read_next(100, &CdcFilters::default()).expect("read");
        assert!(batch.is_empty(), "empty dir = empty batch");
    }

    /// Live consumers must see entries in the ACTIVE (unsealed) segment;
    /// waiting for the seal means up to a full rotation of lag.
    #[test]
    fn tailer_reads_active_segment_incrementally() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut mgr = open_manager(dir.path());

        for i in 0..3u64 {
            mgr.append(&make_entry(i, 1000 + i, false)).expect("append");
        }
        mgr.flush().expect("flush");
        // NO seal: the segment is still active.

        let token = ResumeToken::from_start(0);
        let mut tailer = OplogTailer::new(dir.path(), token);
        let batch = tailer.read_next(100, &CdcFilters::default()).expect("read");
        assert_eq!(batch.len(), 3, "active segment entries must be visible");
        assert_eq!(batch[2].0.index, 2);

        // Entries appended AFTER the first read are picked up on the next
        // read from the same tailer position.
        for i in 3..5u64 {
            mgr.append(&make_entry(i, 1000 + i, false)).expect("append");
        }
        mgr.flush().expect("flush");
        let batch = tailer.read_next(100, &CdcFilters::default()).expect("read");
        assert_eq!(batch.len(), 2, "newly appended entries must be visible");
        assert_eq!(batch[0].0.index, 3);
        assert_eq!(batch[1].0.index, 4);
    }

    /// A torn write at the tail of the active segment must not break the
    /// prefix read: complete entries before it are returned.
    #[test]
    fn tailer_active_segment_ignores_torn_tail() {
        use std::io::Write as _;

        let dir = tempfile::tempdir().expect("tempdir");
        let mut mgr = open_manager(dir.path());
        for i in 0..3u64 {
            mgr.append(&make_entry(i, 1000 + i, false)).expect("append");
        }
        mgr.flush().expect("flush");
        drop(mgr); // release the file handle; segment stays unsealed

        // Simulate a torn write: garbage bytes at the end of the file.
        let seg_path = std::fs::read_dir(dir.path())
            .expect("dir")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.extension().is_some_and(|e| e == "oplog"))
            .or_else(|| {
                std::fs::read_dir(dir.path())
                    .ok()?
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .find(|p| p.is_file())
            })
            .expect("segment file");
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&seg_path)
            .expect("open for append");
        f.write_all(&[0x07, 0xde, 0xad, 0xbe]).expect("torn bytes");
        drop(f);

        let token = ResumeToken::from_start(0);
        let mut tailer = OplogTailer::new(dir.path(), token);
        let batch = tailer.read_next(100, &CdcFilters::default()).expect("read");
        assert_eq!(batch.len(), 3, "complete prefix must survive a torn tail");
    }

    #[test]
    fn tailer_reads_sealed_segment() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut mgr = open_manager(dir.path());

        for i in 0..5u64 {
            mgr.append(&make_entry(i, 1000 + i, false)).expect("append");
        }
        seal_manager(&mut mgr);

        let token = ResumeToken::from_start(0);
        let mut tailer = OplogTailer::new(dir.path(), token);
        let batch = tailer.read_next(100, &CdcFilters::default()).expect("read");

        assert_eq!(batch.len(), 5, "all 5 entries must be returned");
        assert_eq!(batch[0].0.index, 0);
        assert_eq!(batch[4].0.index, 4);
        // Token after last entry should point to entry_offset=5
        assert_eq!(batch[4].1.entry_offset, 5);
    }

    #[test]
    fn tailer_resumes_from_token() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut mgr = open_manager(dir.path());

        for i in 0..6u64 {
            mgr.append(&make_entry(i, 1000 + i, false)).expect("append");
        }
        seal_manager(&mut mgr);

        // First batch: consume entries 0-2
        let token = ResumeToken::from_start(0);
        let mut tailer = OplogTailer::new(dir.path(), token);
        let batch1 = tailer.read_next(3, &CdcFilters::default()).expect("read");
        assert_eq!(batch1.len(), 3);
        let resume = batch1[2].1.clone();

        // Second batch from resume token: entries 3-5
        let mut tailer2 = OplogTailer::new(dir.path(), resume);
        let batch2 = tailer2
            .read_next(100, &CdcFilters::default())
            .expect("read");
        assert_eq!(batch2.len(), 3, "must get 3 remaining entries");
        assert_eq!(batch2[0].0.index, 3);
        assert_eq!(batch2[2].0.index, 5);
    }

    #[test]
    fn tailer_filter_is_migration() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut mgr = open_manager(dir.path());

        mgr.append(&make_entry(0, 1000, false)).expect("append");
        mgr.append(&make_entry(1, 1001, true)).expect("append");
        mgr.append(&make_entry(2, 1002, false)).expect("append");
        mgr.append(&make_entry(3, 1003, true)).expect("append");
        seal_manager(&mut mgr);

        let token = ResumeToken::from_start(0);
        let filters = CdcFilters {
            is_migration: Some(false),
            ..Default::default()
        };
        let mut tailer = OplogTailer::new(dir.path(), token);
        let batch = tailer.read_next(100, &filters).expect("read");

        assert_eq!(batch.len(), 2, "only non-migration entries");
        assert!(batch.iter().all(|(e, _)| !e.is_migration));
    }

    #[test]
    fn tailer_filter_edge_type() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut mgr = open_manager(dir.path());

        mgr.append(&make_adj_entry(0, "FOLLOWS")).expect("append");
        mgr.append(&make_adj_entry(1, "LIKES")).expect("append");
        mgr.append(&make_adj_entry(2, "FOLLOWS")).expect("append");
        mgr.append(&make_entry(3, 1003, false)).expect("append"); // non-adj
        seal_manager(&mut mgr);

        let token = ResumeToken::from_start(0);
        let filters = CdcFilters {
            edge_types: vec!["FOLLOWS".to_string()],
            ..Default::default()
        };
        let mut tailer = OplogTailer::new(dir.path(), token);
        let batch = tailer.read_next(100, &filters).expect("read");

        assert_eq!(batch.len(), 2, "only FOLLOWS entries");
        assert_eq!(batch[0].0.index, 0);
        assert_eq!(batch[1].0.index, 2);
    }

    #[test]
    fn tailer_reads_across_multiple_segments() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut mgr = open_manager(dir.path());

        // Segment 1: entries 0-4
        for i in 0..5u64 {
            mgr.append(&make_entry(i, 1000 + i, false)).expect("append");
        }
        seal_manager(&mut mgr);

        // Segment 2: entries 5-9
        for i in 5..10u64 {
            mgr.append(&make_entry(i, 1000 + i, false)).expect("append");
        }
        seal_manager(&mut mgr);

        let token = ResumeToken::from_start(0);
        let mut tailer = OplogTailer::new(dir.path(), token);
        let batch = tailer.read_next(100, &CdcFilters::default()).expect("read");

        assert_eq!(batch.len(), 10, "all 10 entries across 2 segments");
        assert_eq!(batch[0].0.index, 0);
        assert_eq!(batch[9].0.index, 9);
    }

    #[test]
    fn passes_filter_migration() {
        let no_filter = CdcFilters::default();
        let only_normal = CdcFilters {
            is_migration: Some(false),
            ..Default::default()
        };
        let only_migration = CdcFilters {
            is_migration: Some(true),
            ..Default::default()
        };

        let normal = make_entry(0, 1000, false);
        let migration = make_entry(1, 1001, true);

        assert!(passes_filter(&normal, &no_filter));
        assert!(passes_filter(&migration, &no_filter));
        assert!(passes_filter(&normal, &only_normal));
        assert!(!passes_filter(&migration, &only_normal));
        assert!(!passes_filter(&normal, &only_migration));
        assert!(passes_filter(&migration, &only_migration));
    }

    #[test]
    fn passes_filter_edge_type() {
        let follows = make_adj_entry(0, "FOLLOWS");
        let likes = make_adj_entry(1, "LIKES");
        let node = make_entry(2, 1002, false);

        let filter_follows = CdcFilters {
            edge_types: vec!["FOLLOWS".to_string()],
            ..Default::default()
        };
        let filter_both = CdcFilters {
            edge_types: vec!["FOLLOWS".to_string(), "LIKES".to_string()],
            ..Default::default()
        };

        assert!(passes_filter(&follows, &filter_follows));
        assert!(!passes_filter(&likes, &filter_follows));
        assert!(!passes_filter(&node, &filter_follows));

        assert!(passes_filter(&follows, &filter_both));
        assert!(passes_filter(&likes, &filter_both));
        assert!(!passes_filter(&node, &filter_both));
    }
}
