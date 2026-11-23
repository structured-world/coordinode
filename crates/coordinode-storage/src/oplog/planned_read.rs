//! A read of a journal range, named while its [`OplogManager`] is held and
//! carried out after it is let go: the manager's writer keeps appending
//! while a reader reads and decodes the segments, and the reads go through
//! whatever file system the reader passes, in one batch across the files.
//!
//! [`OplogManager`]: super::OplogManager

use std::path::{Path, PathBuf};

use lsm_tree::fs::{BlockBuf, BlockRead, Fs, FsFile, FsOpenOptions};

use crate::error::{StorageError, StorageResult};
use crate::oplog::entry::{OplogEntry, ShardId};
use crate::oplog::manager::RetainedFloor;
use crate::oplog::segment::{FrameSpan, SegmentReader, decode_frames};

/// One file a [`PlannedRead`] reads.
#[derive(Debug)]
pub(crate) enum Part {
    /// A sealed segment, read and checked whole.
    Sealed { path: PathBuf },
    /// Whole frames of the active segment, every one with the kernel.
    Active(FrameSpan),
}

/// The entries of `[from, to)` a journal holds, named by
/// [`OplogManager::plan_read`](super::OplogManager::plan_read) and read with
/// [`Self::read`].
#[derive(Debug)]
#[must_use = "a planned read reads nothing until it is carried out"]
pub struct PlannedRead {
    pub(crate) shard_id: ShardId,
    pub(crate) from: u64,
    pub(crate) to: u64,
    pub(crate) parts: Vec<Part>,
    /// The log's floor, whose counts tell what changed the segments since
    /// the plan was made.
    pub(crate) floor: RetainedFloor,
    /// Cuts of the log as the plan was made: one during the read voids it.
    pub(crate) cuts: u64,
    /// Purges begun as the plan was made: a segment found gone after one
    /// more began was purged.
    pub(crate) purges: u64,
}

impl PlannedRead {
    /// Whether the plan reads no file: the range lies past every segment.
    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    /// [`Self::read`] through the standard file system, for a reader that
    /// holds no engine.
    ///
    /// # Errors
    ///
    /// As [`Self::read`].
    pub fn read_std(self) -> StorageResult<Option<Vec<OplogEntry>>> {
        self.read(&lsm_tree::fs::StdFs)
    }

    /// Read and decode the planned entries through `fs`, every file in one
    /// batched read, ascending by index. The calling thread waits for the
    /// reads; an async caller runs this where blocking is allowed.
    ///
    /// A sealed segment a purge removed after the plan was made holds only
    /// entries below the purge point and is passed over: the caller keeps
    /// to the purge point it holds as it reads.
    ///
    /// Returns `None` when the log was cut back while it was read: what the
    /// files held may not be what the plan named, and the caller plans
    /// again.
    ///
    /// # Errors
    ///
    /// A file cannot be opened or read, or a frame fails its checksum or
    /// does not decode.
    pub fn read(self, fs: &dyn Fs) -> StorageResult<Option<Vec<OplogEntry>>> {
        let options = FsOpenOptions::new().read(true);
        // Each file opened and how much of it is read, then one batch.
        let mut opened: Vec<Opened<'_>> = Vec::with_capacity(self.parts.len());
        for part in &self.parts {
            let path = part.path();
            let file = match fs.open(path, &options) {
                Ok(file) => file,
                // Removed by a purge since the plan: below the purge point.
                // A segment gone otherwise is a hole, not a purge.
                Err(e)
                    if e.kind() == lsm_tree::io::ErrorKind::NotFound
                        && matches!(part, Part::Sealed { .. })
                        && self.floor.cuts() == self.cuts
                        && self.floor.purges() != self.purges =>
                {
                    continue;
                }
                // The log was cut back since the plan: plan again.
                Err(_) if self.floor.cuts() != self.cuts => return Ok(None),
                Err(e) => return Err(StorageError::Io(format!("open segment {path:?}: {e}"))),
            };
            let (offset, len) = match part {
                Part::Sealed { .. } => (
                    0,
                    file.metadata()
                        .map_err(|e| StorageError::Io(format!("measure segment {path:?}: {e}")))?
                        .len,
                ),
                Part::Active(span) => (span.start, span.end - span.start),
            };
            let len = usize::try_from(len)
                .map_err(|_| StorageError::Io(format!("segment {path:?} too large to read")))?;
            opened.push(Opened {
                part,
                file,
                offset,
                data: vec![0u8; len],
            });
        }
        let short = {
            let mut reads: Vec<BlockRead<'_>> = opened
                .iter_mut()
                .map(|o| BlockRead {
                    file: &*o.file,
                    offset: o.offset,
                    buf: BlockBuf::new(&mut o.data),
                })
                .collect();
            let read = fs.read_blocks_batched(&mut reads);
            read.map(|()| reads.iter().position(|read| !read.buf.is_full()))
        };
        // A file cut short or removed under the read: plan again.
        if self.floor.cuts() != self.cuts {
            return Ok(None);
        }
        let short = short.map_err(|e| StorageError::Io(format!("read journal segments: {e}")))?;
        if let Some(short) = short {
            return Err(StorageError::Io(format!(
                "read journal segment {:?}: short read",
                opened[short].part.path()
            )));
        }
        let mut entries = Vec::new();
        for Opened { part, data, .. } in &opened {
            match part {
                Part::Sealed { path } => {
                    let segment = SegmentReader::from_bytes(data, path)?;
                    // Skip segments belonging to a different shard (defensive).
                    if segment.header.shard_id != self.shard_id {
                        continue;
                    }
                    entries.extend(
                        segment
                            .into_entries()
                            .into_iter()
                            .filter(|entry| entry.index >= self.from && entry.index < self.to),
                    );
                }
                Part::Active(span) => {
                    entries.extend(decode_frames(data, span.count, &span.path)?);
                }
            }
        }
        Ok(Some(entries))
    }
}

/// A file of the plan, open, with where its read starts and the bytes read.
struct Opened<'p> {
    part: &'p Part,
    file: Box<dyn FsFile>,
    offset: u64,
    data: Vec<u8>,
}

impl Part {
    fn path(&self) -> &Path {
        match self {
            Self::Sealed { path } => path,
            Self::Active(span) => &span.path,
        }
    }
}
