//! Persistent write buffer for `w:cache` write concern crash recovery.
//!
//! `w:cache` writes are acknowledged after landing in RAM (DrainBuffer) and
//! this NVMe-backed file. If the process crashes before the background drain
//! thread flushes the entry to Raft, the entry can be recovered from the NVMe
//! file on restart.
//!
//! # Atomic rotation protocol
//!
//! Two files live in the write buffer directory:
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
//! # Crash recovery
//!
//! `NvmeWriteBuffer::recover(dir)` reads both files (if present) and returns
//! their entries as `DrainEntry` values to be re-injected into the RAM
//! `DrainBuffer` before the drain thread starts.
//!
//! # File format (per entry)
//!
//! ```text
//! [entry_size: u32]    total bytes of this entry including the size field
//! [mut_count: u32]     number of mutations
//! [commit_ts: u64]
//! [start_ts: u64]
//! for each mutation:
//!   [kind: u8]         0 = Put, 1 = Delete, 2 = Merge
//!   [partition: u8]    PartitionId ordinal
//!   [key_len: u32]
//!   [key: bytes]
//!   if kind != Delete:
//!     [value_len: u32]
//!     [value: bytes]
//! ```
//!
//! A partial trailing entry (truncated on crash) is silently skipped.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use coordinode_core::txn::drain::{DrainEntry, WriteBufferHook};
use coordinode_core::txn::proposal::{Mutation, PartitionId};
use coordinode_core::txn::timestamp::Timestamp;
use tracing::{debug, info, warn};

// ── File names ────────────────────────────────────────────────────

const CURRENT_FILE: &str = "write_buffer_current.bin";
const DRAINING_PREFIX: &str = "write_buffer_draining.";
const DRAINING_SUFFIX: &str = ".bin";

// ── Mutation kind bytes ───────────────────────────────────────────

const KIND_PUT: u8 = 0;
const KIND_DELETE: u8 = 1;
const KIND_MERGE: u8 = 2;

// ── NvmeWriteBuffer ───────────────────────────────────────────────

/// Persistent write buffer for `w:cache` crash recovery.
///
/// Stores `DrainEntry` mutations in an append-only NVMe file.
/// Survives process crashes; data lost only on power failure (NVMe cache).
///
/// Thread-safe: `Mutex<File>` for append serialization.
pub struct NvmeWriteBuffer {
    dir: PathBuf,
    current: Mutex<File>,
    generation: AtomicU64,
}

impl NvmeWriteBuffer {
    /// Open or create the write buffer in the given directory.
    ///
    /// Creates `write_buffer_current.bin` if absent.
    pub fn open(dir: &Path) -> io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join(CURRENT_FILE);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;

        info!(path = %path.display(), "nvme write buffer opened");

        Ok(Self {
            dir: dir.to_owned(),
            current: Mutex::new(file),
            generation: AtomicU64::new(0),
        })
    }

    /// Append a `DrainEntry` to the current NVMe write buffer file.
    ///
    /// Called synchronously during `w:cache` write path — latency should be
    /// ~100µs (NVMe sequential write + fsync). If the NVMe path is absent
    /// (no cache configured), this buffer should not be created.
    pub fn append(&self, entry: &DrainEntry) -> io::Result<()> {
        let encoded = encode_entry(entry);
        let mut file = self.current.lock().unwrap_or_else(|e| e.into_inner());
        file.seek(SeekFrom::End(0))?;
        file.write_all(&encoded)?;
        file.flush()?;
        debug!(
            mutations = entry.mutations.len(),
            "nvme write buffer appended"
        );
        Ok(())
    }

    /// Recover all entries from existing write buffer files (current + draining).
    ///
    /// Called once on startup **before** the drain thread starts. Returns all
    /// un-drained `DrainEntry` values from crash-surviving files.
    ///
    /// After successful recovery, both files are truncated (entries will be
    /// re-injected into the RAM `DrainBuffer` and drained normally).
    pub fn recover(dir: &Path) -> io::Result<Vec<DrainEntry>> {
        let mut all_entries = Vec::new();

        // Recover draining files first (they were being drained when crash happened).
        let draining = find_draining_files(dir)?;
        for path in &draining {
            let entries = read_entries_from_file(path)?;
            let n = entries.len();
            all_entries.extend(entries);
            // Delete after reading — entries will be re-drained from DrainBuffer.
            if let Err(e) = std::fs::remove_file(path) {
                warn!(path = %path.display(), error = %e, "failed to remove draining file");
            } else {
                info!(path = %path.display(), entries = n, "recovered and removed draining file");
            }
        }

        // Recover current file.
        let current_path = dir.join(CURRENT_FILE);
        if current_path.exists() {
            let entries = read_entries_from_file(&current_path)?;
            let n = entries.len();
            if n > 0 {
                all_entries.extend(entries);
                info!(entries = n, "recovered entries from current write buffer");
            }
            // Truncate current file — entries will be re-appended if w:cache writes
            // happen before drain. We truncate here so we don't double-replay on
            // a second restart if the process crashes again before drain.
            truncate_file(&current_path)?;
        }

        Ok(all_entries)
    }
}

impl WriteBufferHook for NvmeWriteBuffer {
    /// Checkpoint: atomically rename current file → draining.<token>.bin.
    ///
    /// New `w:cache` writes after this point go to a fresh current file
    /// (created lazily on next `append()`). Returns the generation token
    /// for `complete_drain()`.
    fn begin_drain(&self) -> u64 {
        let token = self.generation.fetch_add(1, Ordering::Relaxed);

        let current_path = self.dir.join(CURRENT_FILE);
        if !current_path.exists() {
            // Nothing in current file — no-op checkpoint.
            debug!(token, "nvme write buffer: nothing to checkpoint");
            return token;
        }

        let draining_path = draining_file_path(&self.dir, token);
        match std::fs::rename(&current_path, &draining_path) {
            Ok(()) => {
                // Re-open/create fresh current file for new writes.
                let mut guard = self.current.lock().unwrap_or_else(|e| e.into_inner());
                match OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .truncate(false)
                    .open(self.dir.join(CURRENT_FILE))
                {
                    Ok(f) => {
                        *guard = f;
                        debug!(token, "nvme write buffer: checkpoint created");
                    }
                    Err(e) => {
                        warn!(error = %e, "failed to create fresh current file after rename");
                    }
                }
            }
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

fn truncate_file(path: &Path) -> io::Result<()> {
    OpenOptions::new().write(true).truncate(true).open(path)?;
    Ok(())
}

/// Encode a `DrainEntry` to bytes for NVMe storage.
fn encode_entry(entry: &DrainEntry) -> Vec<u8> {
    let mut body = Vec::new();

    let mut_count = entry.mutations.len() as u32;
    body.extend_from_slice(&mut_count.to_le_bytes());
    body.extend_from_slice(&entry.commit_ts.as_raw().to_le_bytes());
    body.extend_from_slice(&entry.start_ts.as_raw().to_le_bytes());

    for mutation in &entry.mutations {
        encode_mutation(mutation, &mut body);
    }

    // Prepend entry_size (includes the 4-byte size field itself).
    let entry_size = (4 + body.len()) as u32;
    let mut out = entry_size.to_le_bytes().to_vec();
    out.append(&mut body);
    out
}

fn encode_mutation(mutation: &Mutation, buf: &mut Vec<u8>) {
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
    }
}

fn write_bytes(data: &[u8], buf: &mut Vec<u8>) {
    buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
    buf.extend_from_slice(data);
}

fn read_bytes(buf: &[u8], pos: &mut usize) -> Option<Vec<u8>> {
    if *pos + 4 > buf.len() {
        return None;
    }
    let len = u32::from_le_bytes([buf[*pos], buf[*pos + 1], buf[*pos + 2], buf[*pos + 3]]) as usize;
    *pos += 4;
    if *pos + len > buf.len() {
        return None;
    }
    let data = buf[*pos..*pos + len].to_vec();
    *pos += len;
    Some(data)
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
        _ => None,
    }
}

/// Read all valid `DrainEntry` values from a file.
/// Partial trailing entries (truncated on crash) are silently ignored.
fn read_entries_from_file(path: &Path) -> io::Result<Vec<DrainEntry>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let mut file = File::open(path)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;

    let mut entries = Vec::new();
    let mut pos = 0;

    while pos + 4 <= data.len() {
        let entry_size =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;

        // Validate: entry_size must cover at least the size field.
        if entry_size < 4 || pos + entry_size > data.len() {
            // Partial entry at end of file — truncated on crash, skip.
            break;
        }

        let entry_data = &data[pos + 4..pos + entry_size];
        match decode_entry(entry_data) {
            Some(entry) => entries.push(entry),
            None => {
                warn!(
                    path = %path.display(),
                    "corrupt entry in write buffer — skipping remaining"
                );
                break;
            }
        }

        pos += entry_size;
    }

    Ok(entries)
}

fn decode_entry(data: &[u8]) -> Option<DrainEntry> {
    let mut pos = 0;

    if pos + 4 > data.len() {
        return None;
    }
    let mut_count =
        u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
    pos += 4;

    if pos + 8 > data.len() {
        return None;
    }
    let commit_ts = u64::from_le_bytes(data[pos..pos + 8].try_into().ok()?);
    pos += 8;

    if pos + 8 > data.len() {
        return None;
    }
    let start_ts = u64::from_le_bytes(data[pos..pos + 8].try_into().ok()?);
    pos += 8;

    let mut mutations = Vec::with_capacity(mut_count);
    for _ in 0..mut_count {
        if pos + 2 > data.len() {
            return None;
        }
        let kind = data[pos];
        let partition = u8_to_partition(data[pos + 1])?;
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
            _ => return None,
        };
        mutations.push(mutation);
    }

    Some(DrainEntry::new(
        mutations,
        Timestamp::from_raw(commit_ts),
        Timestamp::from_raw(start_ts),
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
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_entry(n: usize, ts: u64) -> DrainEntry {
        let mutations = (0..n)
            .map(|i| Mutation::Put {
                partition: PartitionId::Node,
                key: format!("key_{i}").into_bytes(),
                value: vec![i as u8, 42],
            })
            .collect();
        DrainEntry::new(mutations, Timestamp::from_raw(ts), Timestamp::from_raw(1))
    }

    fn delete_entry(ts: u64) -> DrainEntry {
        let mutations = vec![Mutation::Delete {
            partition: PartitionId::Adj,
            key: b"adj:key".to_vec(),
        }];
        DrainEntry::new(mutations, Timestamp::from_raw(ts), Timestamp::from_raw(1))
    }

    fn merge_entry(ts: u64) -> DrainEntry {
        let mutations = vec![Mutation::Merge {
            partition: PartitionId::Node,
            key: b"node:1".to_vec(),
            operand: vec![1, 2, 3, 4],
        }];
        DrainEntry::new(mutations, Timestamp::from_raw(ts), Timestamp::from_raw(1))
    }

    #[test]
    fn roundtrip_put_entry() {
        let entry = test_entry(3, 100);
        let encoded = encode_entry(&entry);
        let decoded = decode_entry(&encoded[4..]).unwrap();

        assert_eq!(decoded.mutations.len(), 3);
        assert_eq!(decoded.commit_ts.as_raw(), 100);
        assert_eq!(decoded.start_ts.as_raw(), 1);

        let Mutation::Put {
            partition,
            key,
            value,
        } = &decoded.mutations[0]
        else {
            assert!(false, "expected Put variant");
            return;
        };
        assert_eq!(*partition, PartitionId::Node);
        assert_eq!(key, b"key_0");
        assert_eq!(value, &[0u8, 42]);
    }

    #[test]
    fn roundtrip_delete_entry() {
        let entry = delete_entry(200);
        let encoded = encode_entry(&entry);
        let decoded = decode_entry(&encoded[4..]).unwrap();
        assert_eq!(decoded.commit_ts.as_raw(), 200);
        let Mutation::Delete { partition, key } = &decoded.mutations[0] else {
            assert!(false, "expected Delete variant");
            return;
        };
        assert_eq!(*partition, PartitionId::Adj);
        assert_eq!(key, b"adj:key");
    }

    #[test]
    fn roundtrip_merge_entry() {
        let entry = merge_entry(300);
        let encoded = encode_entry(&entry);
        let decoded = decode_entry(&encoded[4..]).unwrap();
        let Mutation::Merge { operand, .. } = &decoded.mutations[0] else {
            assert!(false, "expected Merge variant");
            return;
        };
        assert_eq!(operand, &[1u8, 2, 3, 4]);
    }

    #[test]
    fn open_and_append() {
        let dir = TempDir::new().unwrap();
        let wb = NvmeWriteBuffer::open(dir.path()).unwrap();

        wb.append(&test_entry(2, 100)).unwrap();
        wb.append(&test_entry(3, 200)).unwrap();

        let recovered = NvmeWriteBuffer::recover(dir.path()).unwrap();
        assert_eq!(recovered.len(), 2);
        assert_eq!(recovered[0].mutations.len(), 2);
        assert_eq!(recovered[0].commit_ts.as_raw(), 100);
        assert_eq!(recovered[1].mutations.len(), 3);
        assert_eq!(recovered[1].commit_ts.as_raw(), 200);
    }

    #[test]
    fn recover_empty_dir() {
        let dir = TempDir::new().unwrap();
        let entries = NvmeWriteBuffer::recover(dir.path()).unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn recover_clears_file() {
        let dir = TempDir::new().unwrap();
        let wb = NvmeWriteBuffer::open(dir.path()).unwrap();
        wb.append(&test_entry(1, 100)).unwrap();
        drop(wb);

        // First recovery: returns entries.
        let first = NvmeWriteBuffer::recover(dir.path()).unwrap();
        assert_eq!(first.len(), 1);

        // Second recovery: file truncated, no entries.
        let second = NvmeWriteBuffer::recover(dir.path()).unwrap();
        assert!(second.is_empty());
    }

    #[test]
    fn atomic_drain_checkpoint() {
        let dir = TempDir::new().unwrap();
        let wb = NvmeWriteBuffer::open(dir.path()).unwrap();

        wb.append(&test_entry(2, 100)).unwrap();
        wb.append(&test_entry(1, 200)).unwrap();

        // begin_drain checkpoints current → draining
        let token = wb.begin_drain();

        // New entry goes to fresh current
        wb.append(&test_entry(1, 300)).unwrap();

        // complete_drain removes the draining file
        wb.complete_drain(token);

        // Only the post-checkpoint entry remains in current file
        let recovered = NvmeWriteBuffer::recover(dir.path()).unwrap();
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].commit_ts.as_raw(), 300);
    }

    #[test]
    fn recover_draining_file_survives_crash() {
        // Simulate: begin_drain happened, crash before complete_drain.
        let dir = TempDir::new().unwrap();
        let wb = NvmeWriteBuffer::open(dir.path()).unwrap();

        wb.append(&test_entry(2, 100)).unwrap();
        let token = wb.begin_drain();
        // Simulate crash: do NOT call complete_drain.
        drop(wb);

        // On restart: recover finds draining file.
        // 1 DrainEntry was written (with 2 mutations inside it).
        let recovered = NvmeWriteBuffer::recover(dir.path()).unwrap();
        assert_eq!(
            recovered.len(),
            1,
            "draining file entries must be recovered"
        );
        assert_eq!(recovered[0].commit_ts.as_raw(), 100);
        assert_eq!(
            recovered[0].mutations.len(),
            2,
            "entry should contain the 2 original mutations"
        );

        // Draining file was removed after recovery.
        let draining = find_draining_files(dir.path()).unwrap();
        assert!(
            draining.is_empty(),
            "draining file should be removed after recovery"
        );
        let _ = token; // suppress unused warning
    }

    #[test]
    fn write_buffer_hook_begin_complete() {
        // Test WriteBufferHook trait interface.
        let dir = TempDir::new().unwrap();
        let wb = NvmeWriteBuffer::open(dir.path()).unwrap();

        wb.append(&test_entry(1, 100)).unwrap();

        let hook: &dyn WriteBufferHook = &wb;
        let token = hook.begin_drain();
        hook.complete_drain(token);

        // After complete, no files remain.
        let draining = find_draining_files(dir.path()).unwrap();
        assert!(draining.is_empty());
    }

    #[test]
    fn partition_roundtrip() {
        let partitions = [
            PartitionId::Node,
            PartitionId::Adj,
            PartitionId::EdgeProp,
            PartitionId::Blob,
            PartitionId::BlobRef,
            PartitionId::Schema,
            PartitionId::Idx,
            PartitionId::Counter,
        ];
        for p in partitions {
            let b = partition_to_u8(p);
            let back = u8_to_partition(b).unwrap();
            assert_eq!(p, back, "partition {p:?} roundtrip failed");
        }
    }

    #[test]
    fn partial_entry_truncation_handled() {
        // Write a valid entry followed by a partial one (simulates crash mid-write).
        let dir = TempDir::new().unwrap();
        let current_path = dir.path().join(CURRENT_FILE);

        let entry = test_entry(1, 100);
        let encoded = encode_entry(&entry);

        // Write complete entry + partial garbage.
        std::fs::write(&current_path, {
            let mut data = encoded;
            data.extend_from_slice(&[0xFF, 0xFF, 0x00]); // partial/corrupt
            data
        })
        .unwrap();

        let recovered = NvmeWriteBuffer::recover(dir.path()).unwrap();
        // Only the complete entry should be recovered.
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].commit_ts.as_raw(), 100);
    }
}
