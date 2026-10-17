//! Persistent write buffer for `w:cache` write concern crash recovery.
//!
//! `w:cache` writes are acknowledged after landing in RAM (DrainBuffer) and
//! this NVMe-backed file. If the process crashes before the background drain
//! thread hands an entry to the log, the entry can be recovered from the file
//! on restart.
//!
//! # Exactly once
//!
//! Each buffer has an identity, and each entry a sequence number within it.
//! Appending an entry adds a mark to what it writes: [`DRAINED_KEY`] set to
//! the buffer identity and the entry's number, in the same unit as its
//! effects. The log applies the mark exactly when it applies the entry, so
//! on recovery every entry at or below the recorded number is known applied
//! and is skipped, and every entry above it is known not applied. A crash
//! between the drain's apply and the file cleanup cannot apply an entry twice.
//!
//! # Atomic rotation protocol
//!
//! Two kinds of files live in the write buffer directory:
//!
//! - `write_buffer_current.bin` — active writes being appended.
//! - `write_buffer_draining.NNN.bin` — checkpoint being drained (N = generation).
//!
//! On `begin_drain(token)`: current file is atomically renamed to
//! `write_buffer_draining.NNN.bin`. New writes go to a fresh current file.
//!
//! On `complete_drain(token)`: the draining file is deleted. Crash between
//! begin and complete → draining file survives for replay.
//!
//! # File format
//!
//! ```text
//! header: [magic: "CNVB"] [format: u16 = 2] [buffer id: u64]
//! entry:
//!   [entry_size: u32]  total bytes of this entry including the size field
//!   [seq: u64]         the entry's number within the buffer
//!   [mut_count: u32]
//!   [commit_ts: u64]
//!   [start_ts: u64]
//!   for each mutation:
//!     [kind: u8] [partition: u8] [key_len: u32] [key] ([value_len: u32] [value])
//! ```
//!
//! A partial trailing entry (truncated on crash) is skipped. A file without
//! this header was written before entries carried a number, so whether its
//! entries were applied cannot be told: it is kept and refused, never
//! replayed or deleted.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use coordinode_core::txn::drain::{DrainBuffer, DrainBufferFull, DrainEntry, WriteBufferHook};
use coordinode_core::txn::proposal::{Mutation, PartitionId};
use coordinode_core::txn::timestamp::Timestamp;
use tracing::{debug, info, warn};

// ── File names and header ─────────────────────────────────────────

const CURRENT_FILE: &str = "write_buffer_current.bin";
const DRAINING_PREFIX: &str = "write_buffer_draining.";
const DRAINING_SUFFIX: &str = ".bin";

const MAGIC: [u8; 4] = *b"CNVB";
const FORMAT: u16 = 2;
const HEADER_LEN: usize = 4 + 2 + 8;

/// Schema key recording the last entry of a write buffer the log applied:
/// the buffer identity and the entry's number, eight bytes each, big-endian.
/// This node's own record, like the rest of `meta:`.
pub const DRAINED_KEY: &[u8] = b"meta:volatile_drained";

/// The entry number of buffer `id` recorded under [`DRAINED_KEY`] as
/// `value`, when the record is about that buffer.
pub fn drained_through(value: &[u8], id: u64) -> Option<u64> {
    let recorded: [u8; 16] = value.try_into().ok()?;
    let (buffer, seq) = recorded.split_at(8);
    if u64::from_be_bytes(buffer.try_into().ok()?) != id {
        return None;
    }
    Some(u64::from_be_bytes(seq.try_into().ok()?))
}

fn drained_value(id: u64, seq: u64) -> Vec<u8> {
    let mut value = Vec::with_capacity(16);
    value.extend_from_slice(&id.to_be_bytes());
    value.extend_from_slice(&seq.to_be_bytes());
    value
}

// ── Mutation kind bytes ───────────────────────────────────────────

const KIND_PUT: u8 = 0;
const KIND_DELETE: u8 = 1;
const KIND_MERGE: u8 = 2;
const KIND_REMOVE_RANGE: u8 = 3;
/// A metadata command, MessagePack-encoded under the Schema partition tag.
const KIND_COMMAND: u8 = 4;
/// DERIVED index work, MessagePack-encoded under the Idx partition tag.
const KIND_DERIVE: u8 = 5;
/// An operation command, MessagePack-encoded under the Schema partition tag.
const KIND_OPERATION: u8 = 6;

// ── Recovery ──────────────────────────────────────────────────────

/// What a write buffer directory holds after a crash.
#[derive(Debug)]
pub struct RecoveredBuffer {
    /// The buffer the entries belong to.
    pub id: u64,
    /// Its entries with their numbers, in number order.
    pub entries: Vec<(u64, DrainEntry)>,
}

/// Why a write buffer directory cannot be recovered automatically. Its
/// files are left as they are.
#[derive(Debug, thiserror::Error)]
pub enum RecoverError {
    /// A file without the numbered format: whether its entries were applied
    /// cannot be told, so it is neither replayed nor deleted.
    #[error(
        "write buffer file {path} predates numbered entries; it is kept, and the \
         store does not open until it is moved away"
    )]
    Unnumbered {
        /// The file.
        path: PathBuf,
    },
    /// Files of two different buffers: one recovery was interrupted by
    /// another buffer's writes, which this store never does.
    #[error("write buffer files of two buffers ({first} and {second}) in one directory")]
    MixedBuffers {
        /// One buffer identity found.
        first: u64,
        /// Another buffer identity found.
        second: u64,
    },
    /// The files could not be read.
    #[error("write buffer I/O: {0}")]
    Io(#[from] io::Error),
}

/// Why a cache-concern write was not accepted. Nothing of it is in the file
/// or the drain buffer.
#[derive(Debug, thiserror::Error)]
pub enum AppendError {
    /// The drain buffer has no room for it.
    #[error(transparent)]
    Full(DrainBufferFull),
    /// The file could not be written.
    #[error("write buffer I/O: {0}")]
    Io(#[from] io::Error),
}

// ── NvmeWriteBuffer ───────────────────────────────────────────────

/// Persistent write buffer for `w:cache` crash recovery.
///
/// Stores `DrainEntry` mutations in an append-only NVMe file.
/// Survives process crashes; data lost only on power failure (NVMe cache).
///
/// Thread-safe: `Mutex<File>` for append serialization.
pub struct NvmeWriteBuffer {
    dir: PathBuf,
    id: u64,
    current: Mutex<File>,
    generation: AtomicU64,
    /// The number of the last entry appended.
    last_seq: AtomicU64,
}

impl NvmeWriteBuffer {
    /// Start a fresh buffer `id` in `dir`, replacing a current file. Run
    /// [`Self::recover`] and dispose of what it returns first: this does
    /// not look at what the directory held. `id` must not have been used
    /// by this store before; a timestamp of its clock taken at open is one.
    pub fn open(dir: &Path, id: u64) -> io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        let file = create_buffer_file(&dir.join(CURRENT_FILE), id)?;
        info!(dir = %dir.display(), id, "nvme write buffer opened");
        Ok(Self {
            dir: dir.to_owned(),
            id,
            current: Mutex::new(file),
            generation: AtomicU64::new(0),
            last_seq: AtomicU64::new(0),
        })
    }

    /// This buffer's identity.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Number `entry`, add its applied mark (see the module docs), append it
    /// to the current file and put it in `drain`, the drain buffer.
    ///
    /// Called synchronously on the `w:cache` write path before the write is
    /// acknowledged. Written through to the OS, which is what survives a
    /// process crash. All of it under one lock, so entries reach the drain in
    /// number order, which is what makes a recorded number mean that every
    /// entry below it was applied; and the room in `drain` is taken first, so
    /// a write refused for want of room is not in the file to be replayed.
    pub fn append(&self, mut entry: DrainEntry, drain: &DrainBuffer) -> Result<(), AppendError> {
        let mut file = self.current.lock().unwrap_or_else(|e| e.into_inner());
        let seq = self.last_seq.load(Ordering::Relaxed) + 1;
        entry.mark(Mutation::Put {
            partition: PartitionId::Schema,
            key: DRAINED_KEY.to_vec(),
            value: drained_value(self.id, seq),
        });
        let slot = drain
            .reserve(entry.size_bytes() as u64)
            .map_err(AppendError::Full)?;
        let encoded = encode_entry(seq, &entry)?;
        file.seek(SeekFrom::End(0))?;
        file.write_all(&encoded)?;
        file.flush()?;
        self.last_seq.store(seq, Ordering::Relaxed);
        debug!(
            seq,
            mutations = entry.mutations.len(),
            "nvme write buffer appended"
        );
        slot.fill(entry);
        Ok(())
    }

    /// Read every entry the directory holds (current and draining files),
    /// without changing anything. `None` when it holds none.
    ///
    /// Called on startup before the buffer is opened.
    pub fn recover(dir: &Path) -> Result<Option<RecoveredBuffer>, RecoverError> {
        let mut paths = find_draining_files(dir)?;
        let current = dir.join(CURRENT_FILE);
        if current.exists() {
            paths.push(current);
        }
        let mut found: Option<RecoveredBuffer> = None;
        for path in &paths {
            let Some(RecoveredBuffer { id, entries }) = read_buffer_file(path)? else {
                continue;
            };
            match &mut found {
                None => found = Some(RecoveredBuffer { id, entries }),
                Some(buffer) if buffer.id == id => buffer.entries.extend(entries),
                Some(buffer) => {
                    return Err(RecoverError::MixedBuffers {
                        first: buffer.id,
                        second: id,
                    });
                }
            }
        }
        Ok(found
            .map(|mut buffer| {
                buffer.entries.sort_by_key(|(seq, _)| *seq);
                buffer
            })
            .filter(|buffer| !buffer.entries.is_empty()))
    }

    /// Remove every buffer file of `dir`: what [`Self::recover`] returned is
    /// in the log now, or is lost by the concern's contract.
    pub fn discard(dir: &Path) -> io::Result<()> {
        for path in find_draining_files(dir)? {
            std::fs::remove_file(&path)?;
        }
        let current = dir.join(CURRENT_FILE);
        if current.exists() {
            std::fs::remove_file(&current)?;
        }
        Ok(())
    }
}

impl WriteBufferHook for NvmeWriteBuffer {
    /// Checkpoint: atomically rename current file → `draining.<token>.bin`.
    ///
    /// New `w:cache` writes after this point go to a fresh current file of
    /// the same buffer. Returns the generation token for `complete_drain()`.
    fn begin_drain(&self) -> u64 {
        let token = self.generation.fetch_add(1, Ordering::Relaxed);
        // Held across the rename, so no append lands in a file being moved.
        let mut guard = self.current.lock().unwrap_or_else(|e| e.into_inner());
        let current_path = self.dir.join(CURRENT_FILE);
        let draining_path = draining_file_path(&self.dir, token);
        match std::fs::rename(&current_path, &draining_path) {
            Ok(()) => match create_buffer_file(&current_path, self.id) {
                Ok(f) => {
                    *guard = f;
                    debug!(token, "nvme write buffer: checkpoint created");
                }
                Err(e) => {
                    warn!(error = %e, "failed to create fresh current file after rename");
                }
            },
            Err(e) => {
                warn!(token, error = %e, "nvme write buffer: rename failed during begin_drain");
            }
        }
        token
    }

    /// Complete: delete the draining file identified by `token`.
    ///
    /// Called after all Raft proposals for the drained batch succeed.
    /// The data is now in the oplog — no longer needs crash recovery.
    fn complete_drain(&self, token: u64) {
        let draining_path = draining_file_path(&self.dir, token);
        if draining_path.exists() {
            if let Err(e) = std::fs::remove_file(&draining_path) {
                warn!(token, error = %e, "failed to remove draining file after complete");
            } else {
                debug!(token, "nvme write buffer: checkpoint completed");
            }
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────

/// A new, empty buffer file holding only the header of buffer `id`.
fn create_buffer_file(path: &Path, id: u64) -> io::Result<File> {
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    let mut header = Vec::with_capacity(HEADER_LEN);
    header.extend_from_slice(&MAGIC);
    header.extend_from_slice(&FORMAT.to_le_bytes());
    header.extend_from_slice(&id.to_le_bytes());
    file.write_all(&header)?;
    file.flush()?;
    Ok(file)
}

fn draining_file_path(dir: &Path, token: u64) -> PathBuf {
    dir.join(format!("{DRAINING_PREFIX}{token}{DRAINING_SUFFIX}"))
}

fn find_draining_files(dir: &Path) -> io::Result<Vec<PathBuf>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with(DRAINING_PREFIX) && name_str.ends_with(DRAINING_SUFFIX) {
            paths.push(entry.path());
        }
    }
    Ok(paths)
}

/// Encode entry `seq` for NVMe storage.
fn encode_entry(seq: u64, entry: &DrainEntry) -> io::Result<Vec<u8>> {
    let mut body = Vec::new();

    body.extend_from_slice(&seq.to_le_bytes());
    let mut_count = entry.mutations.len() as u32;
    body.extend_from_slice(&mut_count.to_le_bytes());
    body.extend_from_slice(&entry.commit_ts.as_raw().to_le_bytes());
    body.extend_from_slice(&entry.start_ts.as_raw().to_le_bytes());

    for mutation in &entry.mutations {
        encode_mutation(mutation, &mut body)?;
    }

    // Prepend entry_size (includes the 4-byte size field itself).
    let entry_size = (4 + body.len()) as u32;
    let mut out = entry_size.to_le_bytes().to_vec();
    out.append(&mut body);
    Ok(out)
}

fn encode_mutation(mutation: &Mutation, buf: &mut Vec<u8>) -> io::Result<()> {
    match mutation {
        Mutation::Put {
            partition,
            key,
            value,
        } => {
            buf.push(KIND_PUT);
            buf.push(partition_to_u8(*partition));
            write_bytes(key, buf);
            write_bytes(value, buf);
        }
        Mutation::Delete { partition, key } => {
            buf.push(KIND_DELETE);
            buf.push(partition_to_u8(*partition));
            write_bytes(key, buf);
        }
        Mutation::Merge {
            partition,
            key,
            operand,
        } => {
            buf.push(KIND_MERGE);
            buf.push(partition_to_u8(*partition));
            write_bytes(key, buf);
            write_bytes(operand, buf);
        }
        Mutation::RemoveRange {
            partition,
            start,
            end,
        } => {
            buf.push(KIND_REMOVE_RANGE);
            buf.push(partition_to_u8(*partition));
            write_bytes(start, buf);
            write_bytes(end, buf);
        }
        Mutation::Command(command) => {
            buf.push(KIND_COMMAND);
            buf.push(partition_to_u8(PartitionId::Schema));
            let encoded = rmp_serde::to_vec(command).map_err(io::Error::other)?;
            write_bytes(&encoded, buf);
        }
        Mutation::Derive(work) => {
            buf.push(KIND_DERIVE);
            buf.push(partition_to_u8(PartitionId::Idx));
            let encoded = rmp_serde::to_vec(work).map_err(io::Error::other)?;
            write_bytes(&encoded, buf);
        }
        Mutation::Operation(command) => {
            buf.push(KIND_OPERATION);
            buf.push(partition_to_u8(PartitionId::Schema));
            let encoded = rmp_serde::to_vec(command).map_err(io::Error::other)?;
            write_bytes(&encoded, buf);
        }
    }
    Ok(())
}

fn write_bytes(data: &[u8], buf: &mut Vec<u8>) {
    buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
    buf.extend_from_slice(data);
}

fn read_bytes(buf: &[u8], pos: &mut usize) -> Option<Vec<u8>> {
    let len = u32::from_le_bytes(buf.get(*pos..*pos + 4)?.try_into().ok()?) as usize;
    *pos += 4;
    let data = buf.get(*pos..pos.checked_add(len)?)?.to_vec();
    *pos += len;
    Some(data)
}

fn read_u64(buf: &[u8], pos: &mut usize) -> Option<u64> {
    let value = u64::from_le_bytes(buf.get(*pos..*pos + 8)?.try_into().ok()?);
    *pos += 8;
    Some(value)
}

fn partition_to_u8(p: PartitionId) -> u8 {
    match p {
        PartitionId::Node => 0,
        PartitionId::Adj => 1,
        PartitionId::EdgeProp => 2,
        PartitionId::Blob => 3,
        PartitionId::BlobRef => 4,
        PartitionId::Schema => 5,
        PartitionId::Idx => 6,
        PartitionId::Counter => 7,
        PartitionId::VectorF32 => 8,
        PartitionId::Registry => 9,
    }
}

fn u8_to_partition(b: u8) -> Option<PartitionId> {
    match b {
        0 => Some(PartitionId::Node),
        1 => Some(PartitionId::Adj),
        2 => Some(PartitionId::EdgeProp),
        3 => Some(PartitionId::Blob),
        4 => Some(PartitionId::BlobRef),
        5 => Some(PartitionId::Schema),
        6 => Some(PartitionId::Idx),
        7 => Some(PartitionId::Counter),
        8 => Some(PartitionId::VectorF32),
        9 => Some(PartitionId::Registry),
        _ => None,
    }
}

/// Read a buffer file: its buffer identity and numbered entries. `None` for
/// an empty file. A partial trailing entry (truncated on crash) ends the
/// read; a file without the header is refused.
fn read_buffer_file(path: &Path) -> Result<Option<RecoveredBuffer>, RecoverError> {
    let mut data = Vec::new();
    File::open(path)?.read_to_end(&mut data)?;
    if data.is_empty() {
        return Ok(None);
    }
    let unnumbered = || RecoverError::Unnumbered {
        path: path.to_owned(),
    };
    let header = data.get(..HEADER_LEN).ok_or_else(unnumbered)?;
    if header[..4] != MAGIC || u16::from_le_bytes([header[4], header[5]]) != FORMAT {
        return Err(unnumbered());
    }
    let mut id_pos = 6;
    let id = read_u64(header, &mut id_pos).ok_or_else(unnumbered)?;

    let mut entries = Vec::new();
    let mut pos = HEADER_LEN;
    while let Some(size_bytes) = data.get(pos..pos + 4) {
        let entry_size =
            u32::from_le_bytes(size_bytes.try_into().map_err(|_| unnumbered())?) as usize;
        let Some(entry_data) = data
            .get(pos + 4..pos + entry_size)
            .filter(|_| entry_size >= 4)
        else {
            // Partial entry at end of file — truncated on crash, skip.
            break;
        };
        match decode_entry(entry_data) {
            Some(entry) => entries.push(entry),
            None => {
                warn!(path = %path.display(), "corrupt entry in write buffer, skipping the rest");
                break;
            }
        }
        pos += entry_size;
    }
    Ok(Some(RecoveredBuffer { id, entries }))
}

fn decode_entry(data: &[u8]) -> Option<(u64, DrainEntry)> {
    let mut pos = 0;
    let seq = read_u64(data, &mut pos)?;
    let mut_count = u32::from_le_bytes(data.get(pos..pos + 4)?.try_into().ok()?) as usize;
    pos += 4;
    let commit_ts = read_u64(data, &mut pos)?;
    let start_ts = read_u64(data, &mut pos)?;

    let mut mutations = Vec::with_capacity(mut_count.min(data.len()));
    for _ in 0..mut_count {
        let kind = *data.get(pos)?;
        let partition = u8_to_partition(*data.get(pos + 1)?)?;
        pos += 2;

        let mutation = match kind {
            KIND_PUT => {
                let key = read_bytes(data, &mut pos)?;
                let value = read_bytes(data, &mut pos)?;
                Mutation::Put {
                    partition,
                    key,
                    value,
                }
            }
            KIND_DELETE => {
                let key = read_bytes(data, &mut pos)?;
                Mutation::Delete { partition, key }
            }
            KIND_MERGE => {
                let key = read_bytes(data, &mut pos)?;
                let operand = read_bytes(data, &mut pos)?;
                Mutation::Merge {
                    partition,
                    key,
                    operand,
                }
            }
            KIND_REMOVE_RANGE => {
                let start = read_bytes(data, &mut pos)?;
                let end = read_bytes(data, &mut pos)?;
                Mutation::RemoveRange {
                    partition,
                    start,
                    end,
                }
            }
            KIND_COMMAND => {
                let encoded = read_bytes(data, &mut pos)?;
                Mutation::Command(rmp_serde::from_slice(&encoded).ok()?)
            }
            KIND_DERIVE => {
                let encoded = read_bytes(data, &mut pos)?;
                Mutation::Derive(rmp_serde::from_slice(&encoded).ok()?)
            }
            KIND_OPERATION => {
                let encoded = read_bytes(data, &mut pos)?;
                Mutation::Operation(rmp_serde::from_slice(&encoded).ok()?)
            }
            _ => return None,
        };
        mutations.push(mutation);
    }

    Some((
        seq,
        DrainEntry::new(
            mutations,
            Timestamp::from_raw(commit_ts),
            Timestamp::from_raw(start_ts),
        ),
    ))
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::assertions_on_constants
)]
mod tests;
