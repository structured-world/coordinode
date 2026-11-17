//! One reader of a shard's log shared by every change stream.
//!
//! Every stream follows the same log, mostly at its tail. Reading it once
//! per stream multiplies the reads and the decoding of entries by the number
//! of subscribers. The hub reads each entry once, as stored (a unit's frame
//! whole, with the context each stream projects it by), keeps it in memory
//! until every stream reading from memory has passed it (within a byte
//! bound), and hands it to each stream at that stream's own position; each
//! stream projects the units for its own interest. A stream behind the
//! oldest entry the hub holds reads the gap from the log with a tailer of
//! its own, then joins the hub.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;

use rustc_hash::FxHashMap;

use coordinode_storage::error::StorageResult;
use coordinode_storage::oplog::entry::{OplogEntry, OplogOp};
use coordinode_storage::oplog::tailer::{CdcFilters, OplogTailer, ResumeToken};

/// An entry as the hub holds it, shared by every stream it is handed to.
pub(crate) type SharedEntry = Arc<OplogEntry>;

/// Bytes an entry is counted as holding: its keys, values and frames, plus a
/// fixed allowance for the entry and each operation.
fn entry_bytes(entry: &OplogEntry) -> usize {
    entry
        .ops
        .iter()
        .map(|op| {
            64 + match op {
                OplogOp::Insert { key, value, .. } => key.len() + value.len(),
                OplogOp::Delete { key, .. } => key.len(),
                OplogOp::Merge { key, operand, .. } => key.len() + operand.len(),
                OplogOp::RemoveRange { start, end, .. } => start.len() + end.len(),
                OplogOp::RaftEntry { data } => data.len(),
                OplogOp::Unit { frame } => frame.len(),
                OplogOp::Derive { work } => work.len(),
                OplogOp::ColumnarInsert { key, value, .. } => key.len() + value.len(),
                OplogOp::Noop | OplogOp::RaftTruncation { .. } | OplogOp::Namespace { .. } => 0,
            }
        })
        .sum::<usize>()
        + 128
}

/// A tailer handing entries over as stored, from `index` on (the oldest
/// retained entry when 0).
fn frame_tailer(oplog_dirs: &[PathBuf], shard_id: u32, index: u64) -> StorageResult<OplogTailer> {
    let token = if index == 0 {
        ResumeToken::from_start(shard_id)
    } else {
        ResumeToken {
            shard_id,
            segment_id: index,
            entry_offset: 0,
        }
    };
    Ok(OplogTailer::new(oplog_dirs, token)?.keeping_frames())
}

/// What a stream gets from the hub.
pub(crate) enum HubRead {
    /// The entries from the stream's position, and the position after the
    /// last of them.
    Entries {
        entries: Vec<SharedEntry>,
        next: u64,
    },
    /// The stream is behind the oldest entry the hub holds, which is at
    /// this index: it reads up to there from the log itself.
    Behind(u64),
}

/// A stream's registration with the hub; dropping it stops the hub from
/// keeping entries for that stream.
pub(crate) struct HubReader {
    hub: Arc<CdcHub>,
    id: u64,
}

impl Drop for HubReader {
    fn drop(&mut self) {
        let mut inner = self.hub.inner.lock();
        inner.readers.remove(&self.id);
        inner.trim(self.hub.max_bytes);
    }
}

pub(crate) struct CdcHub {
    // no-std: spin::Mutex; held for a read of the newest log bytes and the
    // copy of a few Arc pointers.
    inner: parking_lot::Mutex<Inner>,
    max_bytes: usize,
    oplog_dirs: Vec<PathBuf>,
    shard_id: u32,
}

struct Inner {
    /// Reads the entries after the newest one held.
    tailer: OplogTailer,
    /// Index of the first entry in `ring`.
    base: u64,
    ring: VecDeque<SharedEntry>,
    bytes: usize,
    /// Next index each stream reading from memory reads.
    readers: FxHashMap<u64, u64>,
    next_reader: u64,
}

impl Inner {
    /// One past the newest entry held.
    fn end(&self) -> u64 {
        self.base + self.ring.len() as u64
    }

    /// Release the entries every reader has passed, and the oldest ones past
    /// the byte bound.
    fn trim(&mut self, max_bytes: usize) {
        let floor = self.readers.values().copied().min().unwrap_or(u64::MAX);
        while let Some(front) = self.ring.front() {
            if self.base >= floor && self.bytes <= max_bytes {
                break;
            }
            self.bytes -= entry_bytes(front);
            self.ring.pop_front();
            self.base += 1;
        }
    }
}

impl CdcHub {
    /// A hub over the log in `oplog_dirs` of `shard_id`, holding from index
    /// `from` on (the oldest retained entry when 0) and at most `max_bytes`
    /// of entries.
    pub(crate) fn new(
        oplog_dirs: &[PathBuf],
        shard_id: u32,
        from: u64,
        max_bytes: usize,
    ) -> StorageResult<Self> {
        let tailer = frame_tailer(oplog_dirs, shard_id, from)?;
        let base = tailer.next_index();
        Ok(Self {
            inner: parking_lot::Mutex::new(Inner {
                tailer,
                base,
                ring: VecDeque::new(),
                bytes: 0,
                readers: FxHashMap::default(),
                next_reader: 0,
            }),
            max_bytes,
            oplog_dirs: oplog_dirs.to_vec(),
            shard_id,
        })
    }

    /// A tailer of the same log, handing entries over as the hub does, for a
    /// stream reading a gap behind it.
    pub(crate) fn tailer_at(&self, index: u64) -> StorageResult<OplogTailer> {
        frame_tailer(&self.oplog_dirs, self.shard_id, index)
    }

    /// Register a stream at log index `position`.
    pub(crate) fn reader(self: &Arc<Self>, position: u64) -> HubReader {
        let mut inner = self.inner.lock();
        let id = inner.next_reader;
        inner.next_reader += 1;
        inner.readers.insert(id, position);
        HubReader {
            hub: Arc::clone(self),
            id,
        }
    }

    /// Up to `max` entries from `position` below `until`, for the stream
    /// `reader`.
    ///
    /// # Errors
    ///
    /// The log could not be read, or no longer holds what the hub needs.
    pub(crate) fn read(
        &self,
        reader: &HubReader,
        position: u64,
        max: usize,
        until: u64,
    ) -> StorageResult<HubRead> {
        let mut inner = self.inner.lock();
        // Holding nothing and behind this stream: start where it is rather
        // than read the log since the hub's own start.
        if inner.ring.is_empty() && position > inner.base {
            inner.tailer = frame_tailer(&self.oplog_dirs, self.shard_id, position)?;
            inner.base = position;
        }
        // Read what was applied since, once for every stream, a batch at a
        // time, until this stream has a batch to take.
        let wanted = position
            .checked_add(max as u64)
            .map_or(until, |w| w.min(until));
        while inner.end() < wanted {
            let fresh = inner.tailer.read_next(max, &CdcFilters::default(), until)?;
            if fresh.is_empty() {
                break;
            }
            for (entry, _) in fresh {
                inner.bytes += entry_bytes(&entry);
                inner.ring.push_back(Arc::new(entry));
            }
        }
        // The hub started after this position, or released it before this
        // stream got there.
        if position < inner.base {
            inner.readers.insert(reader.id, position);
            let base = inner.base;
            inner.trim(self.max_bytes);
            return Ok(HubRead::Behind(base));
        }
        let start = usize::try_from(position - inner.base).unwrap_or(usize::MAX);
        let mut entries = Vec::new();
        let mut next = position;
        for shared in inner.ring.iter().skip(start) {
            if entries.len() >= max || next >= until {
                break;
            }
            next = shared.index + 1;
            entries.push(Arc::clone(shared));
        }
        // The stream still needs what it was handed until it says how far
        // it got: a read stopped inside an entry reads that entry again.
        inner.readers.insert(reader.id, position);
        inner.trim(self.max_bytes);
        Ok(HubRead::Entries { entries, next })
    }

    /// Record that the stream `reader` needs nothing before `position` any
    /// more, releasing what no stream needs.
    pub(crate) fn passed(&self, reader: &HubReader, position: u64) {
        let mut inner = self.inner.lock();
        inner.readers.insert(reader.id, position);
        inner.trim(self.max_bytes);
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests;
