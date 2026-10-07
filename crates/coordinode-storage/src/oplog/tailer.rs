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
use crate::oplog::segment::{HEADER_SIZE, read_frames_from};

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
    /// Whether `next` is a position the reader holds (a resume token, or an
    /// entry already read) rather than "the oldest available". A held
    /// position below the retained log is a gap, never a place to skip from.
    held: bool,
    /// Where the frame of `next` is believed to start, so a read takes only
    /// the bytes appended since the last one. A hint: every frame read
    /// through it must carry the index expected next, and a segment that no
    /// longer matches is located again by index.
    cursor: Option<SegmentCursor>,
}

/// A segment and a byte offset inside it.
struct SegmentCursor {
    /// The segment's first log index (its name).
    first_index: u64,
    path: PathBuf,
    /// Offset of the next frame to read.
    offset: u64,
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
            held: !token.is_start(),
            cursor: None,
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
        let mut result = Vec::new();
        // A cursor that reaches nothing is located afresh, once per call.
        let mut located = false;
        while result.len() < max_entries && self.next < until {
            if self.cursor.is_none() {
                if !self.locate()? {
                    break;
                }
                located = true;
            }
            let Some(cursor) = self.cursor.as_mut() else {
                break;
            };
            // Only the bytes past the cursor: a segment being written is read
            // up to its last complete frame, the crc framing makes that safe.
            let frames = match read_frames_from(&cursor.path, cursor.offset) {
                Ok(frames) => frames,
                Err(e) => {
                    // Rewritten by a truncation, or purged, since the cursor
                    // was set: resolve the position again rather than skip.
                    tracing::debug!(
                        segment = %cursor.path.display(),
                        error = %e,
                        "segment not readable; the read resumes here"
                    );
                    self.cursor = None;
                    if located {
                        break;
                    }
                    continue;
                }
            };
            let mut reached = false;
            let mut stale = false;
            let mut full = false;
            for (entry, end) in frames {
                if entry.index < self.next {
                    cursor.offset = end;
                    continue;
                }
                if entry.index > self.next {
                    // The segment does not continue where the cursor expects:
                    // it was rewritten.
                    stale = true;
                    break;
                }
                if entry.index >= until || result.len() >= max_entries {
                    full = true;
                    break;
                }
                cursor.offset = end;
                self.next = entry.index + 1;
                self.held = true;
                reached = true;
                // A reader sees the operations a unit frame encodes.
                let ops = expand_units(&entry.ops)?.into_owned();
                if passes_filter(&entry, &ops, filters) {
                    let token = ResumeToken {
                        shard_id: self.shard_id,
                        segment_id: cursor.first_index,
                        entry_offset: self.next - cursor.first_index,
                    };
                    result.push((OplogEntry { ops, ..entry }, token));
                }
            }
            if full {
                break;
            }
            if reached {
                // Progress since the last locate: the next one may be needed
                // for the following segment.
                located = false;
            }
            if stale || !reached {
                // `next` is applied, so some segment holds it: the log moved
                // on to a newer segment, or this one was rewritten.
                self.cursor = None;
                if located {
                    break;
                }
            }
        }
        Ok(result)
    }

    // ── private ───────────────────────────────────────────────────────────────

    /// Point the cursor at the start of the segment holding `next`; `false`
    /// when there are no segments.
    ///
    /// # Errors
    ///
    /// The directories cannot be listed, or the reader holds a position the
    /// log no longer retains.
    fn locate(&mut self) -> StorageResult<bool> {
        let segments = self.list_segments()?;
        // Segments are purged as a prefix and named by their first index, so
        // the oldest one says where the retained log begins. A reader holding
        // a position below it would otherwise read on from there and never
        // learn of the entries in between.
        let Some(&(first_retained, _)) = segments.first() else {
            return Ok(false);
        };
        if first_retained > self.next {
            if self.held {
                return Err(StorageError::RetentionLost {
                    requested: self.next,
                    first_retained,
                });
            }
            // "From the oldest available": that is where reading starts.
            self.next = first_retained;
        }
        let Some((first_index, path)) = segments
            .iter()
            .rev()
            .find(|(first, _)| *first <= self.next)
            .cloned()
        else {
            return Ok(false);
        };
        self.cursor = Some(SegmentCursor {
            first_index,
            path,
            offset: HEADER_SIZE,
        });
        Ok(true)
    }

    /// List the segment files in every oplog directory, sorted by first_index.
    fn list_segments(&self) -> StorageResult<Vec<(u64, PathBuf)>> {
        list_segments(&self.oplog_dirs)
    }
}

/// Every segment in `oplog_dirs`, as `(first index, path)` ascending.
pub(crate) fn list_segments(oplog_dirs: &[PathBuf]) -> StorageResult<Vec<(u64, PathBuf)>> {
    let mut segments: Vec<(u64, PathBuf)> = Vec::new();
    for dir in oplog_dirs {
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

/// The first log index the segments in `oplog_dirs` hold, or `None` when
/// there are no segments.
///
/// # Errors
///
/// An oplog directory cannot be listed.
pub fn first_retained_index(oplog_dirs: &[PathBuf]) -> StorageResult<Option<u64>> {
    Ok(list_segments(oplog_dirs)?.first().map(|&(first, _)| first))
}

/// Bytes of the segments a reader positioned at `position` still needs: the
/// one holding `position` and every later one. Segments are named by their
/// first index, so a segment is needed when the next one starts above
/// `position`, and the newest always is.
///
/// # Errors
///
/// An oplog directory cannot be listed, or a segment's size cannot be read.
pub fn bytes_needed_from(oplog_dirs: &[PathBuf], position: u64) -> StorageResult<u64> {
    let segments = list_segments(oplog_dirs)?;
    let mut bytes = 0u64;
    for (i, (_, path)) in segments.iter().enumerate() {
        let needed = segments
            .get(i + 1)
            .is_none_or(|&(next_first, _)| next_first > position);
        if !needed {
            continue;
        }
        let len = match std::fs::metadata(path) {
            Ok(meta) => meta.len(),
            // Purged between the listing and this read: it holds nothing.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => 0,
            Err(e) => return Err(StorageError::Io(format!("size of {path:?}: {e}"))),
        };
        // A sum of file sizes on one machine stays far below u64::MAX.
        bytes += len;
    }
    Ok(bytes)
}

// ── Filter logic ──────────────────────────────────────────────────────────────

/// Whether an entry the tailer returned (its unit frames already expanded)
/// passes `filters`: for a reader that shares entries read once among
/// streams with filters of their own.
pub fn entry_passes(entry: &OplogEntry, filters: &CdcFilters) -> bool {
    passes_filter(entry, &entry.ops, filters)
}

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
