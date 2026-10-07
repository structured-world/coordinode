//! [`OplogTimeline`]: when each entry of a shard's log was written.
//!
//! A consumer's progress age is the age of the entry at its acknowledged
//! position, and a bound check asks it for every consumer, again and again.
//! Answering by reading the segment that holds the position costs that
//! segment's size on every ask. The timeline keeps instead, per segment, the
//! byte offset of an entry at most every [`MARK_STRIDE`] bytes. A scan finds
//! them reading each byte of a segment once (the open segment only as it
//! grows), and an answer reads at most a stride from the nearest mark. What
//! it keeps below the lowest acknowledged position is released with
//! [`OplogTimeline::release_below`].

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use rustc_hash::FxHashMap;

use crate::error::{StorageError, StorageResult};
use crate::oplog::segment::{HEADER_SIZE, check_header, frame_ends, parse_frames, read_span};
use crate::oplog::tailer::list_segments;

/// Most bytes of log between two marks: an answer reads about this much.
const MARK_STRIDE: u64 = 64 * 1024;

/// Bytes a scan reads at a time.
const SCAN_CHUNK: u64 = 1024 * 1024;

/// Answers kept for positions asked again before they move.
const ANSWERS_KEPT: usize = 4096;

/// Write times of a shard's log entries, read from the segments in its
/// directories.
pub struct OplogTimeline {
    oplog_dirs: Vec<PathBuf>,
    // no-std: spin::Mutex; held for one answer, which reads about a stride or
    // the bytes appended since the last scan.
    inner: parking_lot::Mutex<Inner>,
    /// Bytes read from the segments so far.
    bytes_read: AtomicU64,
}

#[derive(Default)]
struct Inner {
    /// Marks per segment, by the segment's first index.
    segments: BTreeMap<u64, Marks>,
    /// Write time by position, for the positions asked lately.
    answers: FxHashMap<u64, u64>,
}

/// What a scan learned about one segment.
struct Marks {
    path: PathBuf,
    /// `(index, offset)` of an entry at most every [`MARK_STRIDE`] bytes,
    /// ascending; the first is the segment's first entry.
    marks: Vec<(u64, u64)>,
    /// The first index the scan has not passed.
    next: u64,
    /// Where the frame of `next` starts; 0 until the header is checked.
    end: u64,
}

impl Marks {
    fn new(first_index: u64, path: PathBuf) -> Self {
        Self {
            path,
            marks: Vec::new(),
            next: first_index,
            end: 0,
        }
    }
}

impl OplogTimeline {
    /// A timeline over the segments in `oplog_dirs` (the directories of one
    /// shard's log; missing ones are skipped).
    pub fn new(oplog_dirs: Vec<PathBuf>) -> Self {
        Self {
            oplog_dirs,
            inner: parking_lot::Mutex::new(Inner::default()),
            bytes_read: AtomicU64::new(0),
        }
    }

    /// Bytes read from the segments since the timeline was made.
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read.load(Ordering::Relaxed)
    }

    /// The timestamp (HLC wall microseconds) of the entry at `index`, or
    /// `None` when no segment holds it yet or any more.
    ///
    /// # Errors
    ///
    /// A directory cannot be listed, or a segment cannot be read or is not
    /// in this format.
    pub fn written_at(&self, index: u64) -> StorageResult<Option<u64>> {
        let mut inner = self.inner.lock();
        if let Some(&ts) = inner.answers.get(&index) {
            return Ok(Some(ts));
        }
        let segments = list_segments(&self.oplog_dirs)?;
        // A segment no longer on disk under the same name (purged, or
        // rewritten by a truncation) is forgotten.
        inner.segments.retain(|first, marks| {
            segments
                .iter()
                .any(|(f, path)| f == first && *path == marks.path)
        });
        let Some((first, path)) = segments.iter().rev().find(|(f, _)| *f <= index).cloned() else {
            return Ok(None);
        };
        // Marks that no longer match the file (rewritten in place) are
        // rebuilt once.
        for _ in 0..2 {
            let marks = inner
                .segments
                .entry(first)
                .or_insert_with(|| Marks::new(first, path.clone()));
            if marks.next <= index {
                self.scan(marks, index)?;
            }
            if marks.next <= index {
                return Ok(None);
            }
            match self.find(marks, index)? {
                Some(ts) => {
                    if inner.answers.len() >= ANSWERS_KEPT {
                        inner.answers.clear();
                    }
                    inner.answers.insert(index, ts);
                    return Ok(Some(ts));
                }
                None => {
                    inner.segments.remove(&first);
                }
            }
        }
        Ok(None)
    }

    /// Forget what was kept to answer for positions below `position`: no
    /// one asks for them any more (`u64::MAX`: no one asks at all).
    pub fn release_below(&self, position: u64) {
        let mut inner = self.inner.lock();
        inner.answers.retain(|&p, _| p >= position);
        // Segments are named by their first index: the one holding
        // `position` starts at the greatest name at or below it, and every
        // older one holds only entries below it.
        let holding = inner
            .segments
            .range(..=position)
            .next_back()
            .map(|(&first, _)| first);
        if let Some(first) = holding {
            inner.segments = inner.segments.split_off(&first);
        }
    }

    /// Up to `len` bytes of `path` from `offset`, counted.
    fn read(&self, path: &Path, offset: u64, len: u64) -> StorageResult<Vec<u8>> {
        let data = read_span(path, offset, len)?;
        self.bytes_read
            .fetch_add(data.len() as u64, Ordering::Relaxed);
        Ok(data)
    }

    /// Extend the marks of a segment past `index`, or as far as it is
    /// written, reading only what no scan has read before.
    fn scan(&self, marks: &mut Marks, index: u64) -> StorageResult<()> {
        if marks.end == 0 {
            let header = self.read(&marks.path, 0, HEADER_SIZE)?;
            if (header.len() as u64) < HEADER_SIZE {
                // Created, header not written yet.
                return Ok(());
            }
            check_header(&header, &marks.path)?;
            marks.end = HEADER_SIZE;
        }
        let mut chunk = SCAN_CHUNK;
        while marks.next <= index {
            let data = self.read(&marks.path, marks.end, chunk)?;
            let short = (data.len() as u64) < chunk;
            let ends = frame_ends(&data);
            let Some(&last) = ends.last() else {
                if short {
                    // Nothing complete written past here yet.
                    break;
                }
                // One frame longer than the chunk: read it whole.
                chunk = chunk.checked_mul(2).ok_or_else(|| {
                    StorageError::Io(format!("frame in {:?} past any size", marks.path))
                })?;
                continue;
            };
            let mut start = 0u64;
            for end in ends {
                // Within one file, offsets stay far below u64::MAX.
                let at = marks.end + start;
                if marks
                    .marks
                    .last()
                    .is_none_or(|&(_, mark)| at - mark >= MARK_STRIDE)
                {
                    marks.marks.push((marks.next, at));
                }
                // A segment holds consecutive indexes from its name on.
                marks.next += 1;
                start = end;
            }
            marks.end += last;
            chunk = SCAN_CHUNK;
            if short {
                break;
            }
        }
        Ok(())
    }

    /// The timestamp of the entry at `index`, which the marks have passed,
    /// read from the mark before it; `None` when the file no longer matches
    /// the marks.
    fn find(&self, marks: &Marks, index: u64) -> StorageResult<Option<u64>> {
        let after = marks.marks.partition_point(|&(i, _)| i <= index);
        let Some(&(mut next, mut offset)) = after.checked_sub(1).map(|k| &marks.marks[k]) else {
            return Ok(None);
        };
        let mut chunk = 2 * MARK_STRIDE;
        loop {
            let data = self.read(&marks.path, offset, chunk)?;
            let frames = parse_frames(&data, 0);
            let Some(&(_, last)) = frames.last() else {
                if (data.len() as u64) < chunk {
                    // The frames the scan passed are gone.
                    return Ok(None);
                }
                chunk = chunk.checked_mul(2).ok_or_else(|| {
                    StorageError::Io(format!("frame in {:?} past any size", marks.path))
                })?;
                continue;
            };
            for (entry, _) in frames {
                if entry.index != next {
                    return Ok(None);
                }
                if next == index {
                    return Ok(Some(entry.ts));
                }
                next += 1;
            }
            offset += last;
            chunk = 2 * MARK_STRIDE;
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests;
