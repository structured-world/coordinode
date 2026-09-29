//! [`OplogTailer`]: read-only oplog cursor for CDC consumers.
//!
//! The tailer reads segment files directly from the oplog directories without
//! acquiring any lock on the `OplogManager`. Sealed segments are immutable
//! once written and the active one is read up to its last complete entry,
//! so concurrent reads are safe.
//!
//! ## Usage
//!
//! ```rust,ignore
//! let token = ResumeToken { shard_id: 0, segment_id: 0, entry_offset: 0 };
//! let mut tailer = OplogTailer::new(&[data_dir.join("oplog/0")], token)?;
//!
//! loop {
//!     let batch = tailer.read_next(256, &filters, applied_index + 1)?;
//!     if batch.is_empty() {
//!         // Caught up — wait for new entries.
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
//! [`ResumeToken`] names a position as `(segment_id, entry_offset)`:
//! - `segment_id` = the `first_index` encoded in the filename
//!   (`oplog-<segment_id:020>.bin`)
//! - `entry_offset` = number of entries already consumed from that segment
//!
//! A segment is named by the index of its first entry and holds consecutive
//! indexes, so the pair stands for one log index, the next one to read:
//! `segment_id + entry_offset`. The tailer resolves a token by that index,
//! never by the file: a truncation of the log rewrites its segments under
//! other names, and a position into a file that is gone would otherwise
//! point nowhere or at another entry.
//!
//! A zero token (`segment_id=0, entry_offset=0`) means "start from the oldest
//! available segment".
//!
//! ## Upper bound
//!
//! [`OplogTailer::read_next`] reads below a caller-given index. A consumer
//! of a Raft log passes one past the last applied entry: the log also holds
//! entries that are not committed yet and that a truncation may replace.
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

use std::path::PathBuf;

use crate::error::{StorageError, StorageResult};
use crate::oplog::convert::expand_units;
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

    /// The log index this token resumes at: the next entry to read.
    ///
    /// # Errors
    ///
    /// A token whose parts overflow a log index cannot have been issued by
    /// a tailer.
    pub fn next_index(&self) -> StorageResult<u64> {
        self.segment_id
            .checked_add(self.entry_offset)
            .ok_or_else(|| {
                StorageError::Io(format!(
                    "resume token {}+{} is past the last log index",
                    self.segment_id, self.entry_offset
                ))
            })
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

/// Read-only cursor over oplog segments.
///
/// Reads directly from segment files on disk — no lock on `OplogManager`.
/// Safe to run concurrently with the write path, including a truncation
/// that rewrites the segments: the cursor is a log index, resolved afresh
/// on every read.
pub struct OplogTailer {
    /// Directories holding one shard's segments (`<endpoint>/oplog/<shard_id>/`
    /// on each endpoint that may hold them). Missing ones are skipped.
    oplog_dirs: Vec<PathBuf>,
    shard_id: ShardId,
    /// The next log index to read.
    next: u64,
}

impl OplogTailer {
    /// Create a new tailer over the segments in `oplog_dirs`, starting from
    /// `token`.
    ///
    /// Pass `ResumeToken::from_start(shard_id)` to start from the oldest
    /// available segment.
    ///
    /// # Errors
    ///
    /// A token that names no log index (see [`ResumeToken::next_index`]).
    pub fn new(oplog_dirs: &[PathBuf], token: ResumeToken) -> StorageResult<Self> {
        Ok(Self {
            oplog_dirs: oplog_dirs.to_vec(),
            shard_id: token.shard_id,
            next: token.next_index()?,
        })
    }

    /// The next log index the tailer reads.
    pub fn next_index(&self) -> u64 {
        self.next
    }

    /// Read up to `max_entries` entries at or past the current position and
    /// below log index `until`, applying `filters`.
    ///
    /// Returns a list of `(entry, token)` pairs. Each `token` identifies the
    /// position AFTER this entry — store it for reconnect. The position
    /// advances past every entry read, whether or not it passed the filters.
    ///
    /// Returns an **empty vec** when caught up. The caller should sleep and
    /// retry. A segment that cannot be read right now (a truncation is
    /// rewriting it, or the newest one is mid-write) ends the read where it
    /// stands; nothing past it is read, so no entry is skipped.
    ///
    /// # Errors
    ///
    /// The oplog directory cannot be listed.
    pub fn read_next(
        &mut self,
        max_entries: usize,
        filters: &CdcFilters,
        until: u64,
    ) -> StorageResult<Vec<(OplogEntry, ResumeToken)>> {
        let segments = self.list_segments()?;
        let mut result = Vec::new();

        for (position, (seg_first_index, seg_path)) in segments.iter().enumerate() {
            if result.len() >= max_entries || self.next >= until {
                break;
            }
            // A segment the cursor has passed ends where the next one begins.
            if segments
                .get(position + 1)
                .is_some_and(|(next_first, _)| *next_first <= self.next)
            {
                continue;
            }

            let is_last = position + 1 == segments.len();
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
                        break;
                    }
                },
                Err(e) => {
                    // Rewritten by a truncation since the listing, or not
                    // readable yet: stop here and resolve the position again
                    // on the next call rather than skip what it holds.
                    tracing::debug!(
                        segment = %seg_path.display(),
                        error = %e,
                        "segment not readable; the read resumes here"
                    );
                    break;
                }
            };

            for entry in reader.entries() {
                if entry.index < self.next {
                    continue;
                }
                if entry.index >= until || result.len() >= max_entries {
                    return Ok(result);
                }
                self.next = entry.index + 1;
                // A reader sees the operations a unit frame encodes.
                let ops = expand_units(&entry.ops)?;
                if passes_filter(entry, &ops, filters) {
                    let token = ResumeToken {
                        shard_id: self.shard_id,
                        segment_id: *seg_first_index,
                        entry_offset: self.next - seg_first_index,
                    };
                    let entry = OplogEntry {
                        ts: entry.ts,
                        term: entry.term,
                        index: entry.index,
                        shard: entry.shard,
                        ops: ops.into_owned(),
                        is_migration: entry.is_migration,
                        pre_images: entry.pre_images.clone(),
                    };
                    result.push((entry, token));
                }
            }
        }

        Ok(result)
    }

    // ── private ───────────────────────────────────────────────────────────────

    /// List the segment files in every oplog directory, sorted by first_index.
    fn list_segments(&self) -> StorageResult<Vec<(u64, PathBuf)>> {
        let mut segments: Vec<(u64, PathBuf)> = Vec::new();
        for dir in &self.oplog_dirs {
            let entries = match std::fs::read_dir(dir) {
                Ok(entries) => entries,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => {
                    return Err(StorageError::Io(format!("list oplog dir {dir:?}: {e}")));
                }
            };
            segments.extend(entries.filter_map(|e| e.ok()).filter_map(|e| {
                let p = e.path();
                let idx = p
                    .file_stem()?
                    .to_str()?
                    .strip_prefix("oplog-")?
                    .parse()
                    .ok()?;
                Some((idx, p))
            }));
        }
        segments.sort_by_key(|&(idx, _)| idx);
        Ok(segments)
    }
}

// ── Filter logic ──────────────────────────────────────────────────────────────

/// Returns `true` if `entry`, whose operations are `ops`, passes all active
/// filters.
fn passes_filter(entry: &OplogEntry, ops: &[OplogOp], filters: &CdcFilters) -> bool {
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
        let matches = ops.iter().any(|op| {
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
mod tests;
