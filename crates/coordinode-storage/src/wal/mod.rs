//! Standalone WAL for `OwnedLocalProposalPipeline` (embedded / no-Raft mode).
//!
//! In cluster mode crash recovery = Raft log replay (ADR-017). In standalone
//! mode there is no Raft log. Without a WAL any write that reached the
//! memtable but not an SST flush is lost on crash.
//!
//! The current `persist()`-per-proposal fix (full SST flush after every write)
//! is correct but prohibitively expensive (5-50 ms per write). This module
//! replaces it with an append-only binary journal: write first, apply to
//! memtable second. On crash, replay WAL from start to reconstruct state.
//!
//! ## Record format
//!
//! ```text
//! [8 bytes: lsn         ] — monotonically increasing, 1 per proposal
//! [4 bytes: crc32c      ] — CRC-32/Castagnoli checksum of payload bytes
//! [4 bytes: payload_len ] — byte length of serialised Vec<Mutation>
//! [payload_len bytes    ] — MessagePack-serialised Vec<Mutation>
//! ```
//!
//! ## Lifecycle
//!
//! 1. `open_with_wal()` — creates WAL, replays any existing records
//! 2. `wal_append(mutations)` — write record + fsync (when SyncPerRecord)
//! 3. memtable write in caller (OwnedLocalProposalPipeline)
//! 4. `persist()` flushes all partitions to SST, then calls `checkpoint()`
//! 5. `checkpoint()` rotates: rename → new file → delete old
//!
//! ## Recovery
//!
//! On open with WAL:
//! 1. Delete `standalone.wal.old` if it exists (data already in SST — rotation
//!    invariant: we only rename AFTER confirming SST flush succeeded).
//! 2. If `standalone.wal` exists, replay every valid record into the memtable.
//! 3. Persist (flush memtable to SST) to make replay durable.
//! 4. Delete `standalone.wal`.
//! 5. Open a fresh `standalone.wal` for new writes.

use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};

use coordinode_core::txn::proposal::Mutation;

use crate::error::{StorageError, StorageResult};

/// WAL sync policy: controls when the journal is fsynced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WalSyncPolicy {
    /// fsync after every record append (default).
    ///
    /// Provides full crash safety: a record that returned `Ok` is guaranteed
    /// durable even on sudden power loss.
    #[default]
    SyncPerRecord,

    /// No automatic fsync. The OS may buffer writes for seconds.
    ///
    /// Faster (no fsync stall) but data written since the last OS flush is
    /// at risk on power loss. Suitable for test environments or when the
    /// caller accepts a small data-loss window.
    NoSync,
}

/// Configuration for the standalone WAL.
#[derive(Debug, Clone, Default)]
pub struct WalConfig {
    /// Explicit path to the WAL file.
    ///
    /// When `None` (default) the WAL is placed at `<data_dir>/standalone.wal`.
    pub path: Option<PathBuf>,

    /// Fsync policy. Default: `SyncPerRecord` (full crash safety).
    pub sync: WalSyncPolicy,
}

// ── Constants ────────────────────────────────────────────────────────────────

/// Fixed-size header preceding each payload: 8B lsn + 4B crc32c + 4B len.
const HEADER_LEN: usize = 16;

// ── Internal types ────────────────────────────────────────────────────────────

/// A record decoded during WAL replay.
pub(crate) struct WalReplayRecord {
    pub lsn: u64,
    pub mutations: Vec<Mutation>,
}

// ── StandaloneWal ─────────────────────────────────────────────────────────────

/// Append-only WAL for standalone (no-Raft) mode.
///
/// Created and owned by `StorageEngine` via `open_with_wal`. Wrapped in
/// `Arc<std::sync::Mutex<StandaloneWal>>` so it can be shared with the
/// public `wal_append` method without exposing interior mutability.
pub(crate) struct StandaloneWal {
    file: File,
    /// Active WAL path (e.g. `<data_dir>/standalone.wal`).
    path: PathBuf,
    /// Rotation staging path (`standalone.wal.old`).
    old_path: PathBuf,
    /// LSN to assign to the next `append()` call.
    lsn: u64,
    sync: WalSyncPolicy,
}

impl StandaloneWal {
    /// Derive the `.old` path from the active WAL path.
    fn old_path_for(wal_path: &Path) -> PathBuf {
        let mut p = wal_path.to_path_buf();
        let name = p
            .file_name()
            .map(|n| {
                let mut s = n.to_os_string();
                s.push(".old");
                s
            })
            .unwrap_or_else(|| "standalone.wal.old".into());
        p.set_file_name(name);
        p
    }

    /// Open an existing WAL or create a fresh one.
    ///
    /// Returns `(wal, records)` where `records` are all valid entries that
    /// must be replayed into the memtable before normal operation begins.
    ///
    /// # Rotation invariant
    ///
    /// `standalone.wal.old` is only ever created by `checkpoint()`, which
    /// runs *after* a successful `persist()`.  On recovery the `.old` file's
    /// contents are therefore already in SST — we delete it unconditionally.
    pub fn open(path: PathBuf, sync: WalSyncPolicy) -> StorageResult<(Self, Vec<WalReplayRecord>)> {
        let old_path = Self::old_path_for(&path);

        // Rotation invariant: .old contents are already in SST — just delete.
        if old_path.exists() {
            std::fs::remove_file(&old_path)
                .map_err(|e| StorageError::Io(format!("remove WAL.old on open: {e}")))?;
            tracing::debug!(path = %old_path.display(), "WAL: deleted stale .old file on open");
        }

        // Replay existing WAL (if any).
        let records = if path.exists() {
            tracing::info!(path = %path.display(), "WAL: replaying existing journal");
            let recs = Self::replay_file(&path)?;
            tracing::info!(
                count = recs.len(),
                path = %path.display(),
                "WAL: replay complete"
            );
            recs
        } else {
            vec![]
        };

        let start_lsn = records.iter().map(|r| r.lsn + 1).max().unwrap_or(0);

        // Open in append mode. If we replayed records the caller will persist()
        // then checkpoint() (which creates a fresh file). Until then we append
        // new writes after the old records — harmless since recovery re-reads all.
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| StorageError::Io(format!("open WAL file '{}': {e}", path.display())))?;

        Ok((
            Self {
                file,
                path,
                old_path,
                lsn: start_lsn,
                sync,
            },
            records,
        ))
    }

    /// Scan all valid records from a WAL file.
    ///
    /// Stops at the first CRC mismatch or truncated record (crash mid-write).
    /// All earlier records are valid and must be replayed.
    fn replay_file(path: &Path) -> StorageResult<Vec<WalReplayRecord>> {
        let file = File::open(path).map_err(|e| {
            StorageError::Io(format!("open WAL for replay '{}': {e}", path.display()))
        })?;
        let mut reader = BufReader::new(file);
        let mut records = Vec::new();

        loop {
            let mut header = [0u8; HEADER_LEN];
            match reader.read_exact(&mut header) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => {
                    return Err(StorageError::Io(format!("WAL header read: {e}")));
                }
            }

            let lsn = u64::from_be_bytes([
                header[0], header[1], header[2], header[3], header[4], header[5], header[6],
                header[7],
            ]);
            let stored_crc = u32::from_be_bytes([header[8], header[9], header[10], header[11]]);
            let payload_len =
                u32::from_be_bytes([header[12], header[13], header[14], header[15]]) as usize;

            let mut payload = vec![0u8; payload_len];
            match reader.read_exact(&mut payload) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    // Crash mid-write: truncated payload — discard partial record.
                    tracing::warn!(lsn, "WAL: truncated payload at tail, stopping replay");
                    break;
                }
                Err(e) => {
                    return Err(StorageError::Io(format!("WAL payload read: {e}")));
                }
            }

            // Validate CRC-32/Castagnoli.
            let computed_crc = crc32c::crc32c(&payload);
            if computed_crc != stored_crc {
                tracing::warn!(
                    lsn,
                    stored_crc = format!("{stored_crc:#010x}"),
                    computed_crc = format!("{computed_crc:#010x}"),
                    "WAL: CRC mismatch, stopping replay (corrupt or partial write)"
                );
                break;
            }

            let mutations: Vec<Mutation> = rmp_serde::from_slice(&payload).map_err(|e| {
                StorageError::Serialization(format!("WAL deserialize lsn={lsn}: {e}"))
            })?;

            records.push(WalReplayRecord { lsn, mutations });
        }

        Ok(records)
    }

    /// Append a WAL record for the given mutations.
    ///
    /// Serialises `mutations` as MessagePack, writes the fixed-size header
    /// followed by the payload, and fsyncs if `sync == SyncPerRecord`.
    ///
    /// Returns the LSN assigned to this record.
    ///
    /// # Write order
    ///
    /// The caller **must** apply the mutations to the memtable only AFTER this
    /// method returns `Ok`.  The WAL-first invariant ensures that any record
    /// present in the WAL is eventually applied on recovery.
    pub fn append(&mut self, mutations: &[Mutation]) -> StorageResult<u64> {
        let lsn = self.lsn;
        self.lsn += 1;

        let payload = rmp_serde::to_vec(mutations)
            .map_err(|e| StorageError::Serialization(format!("WAL serialize lsn={lsn}: {e}")))?;

        let crc = crc32c::crc32c(&payload);
        let payload_len = u32::try_from(payload.len()).map_err(|_| {
            StorageError::InvalidConfig("WAL: mutations payload exceeds 4 GiB".into())
        })?;

        // Fixed-size header: [lsn 8B][crc32c 4B][payload_len 4B]
        let mut header = [0u8; HEADER_LEN];
        header[0..8].copy_from_slice(&lsn.to_be_bytes());
        header[8..12].copy_from_slice(&crc.to_be_bytes());
        header[12..16].copy_from_slice(&payload_len.to_be_bytes());

        self.file
            .write_all(&header)
            .map_err(|e| StorageError::Io(format!("WAL write header lsn={lsn}: {e}")))?;
        self.file
            .write_all(&payload)
            .map_err(|e| StorageError::Io(format!("WAL write payload lsn={lsn}: {e}")))?;

        if self.sync == WalSyncPolicy::SyncPerRecord {
            self.file
                .sync_data()
                .map_err(|e| StorageError::Io(format!("WAL fsync lsn={lsn}: {e}")))?;
        }

        Ok(lsn)
    }

    /// Checkpoint the WAL by rotating the file.
    ///
    /// Called after a successful `persist()` (SST flush). All data that was
    /// in the WAL is now safely in SST files, so we can truncate.
    ///
    /// Rotation steps:
    /// 1. Flush OS write buffers.
    /// 2. Rename `standalone.wal` → `standalone.wal.old`.
    /// 3. Create a fresh `standalone.wal`.
    /// 4. Delete `standalone.wal.old` (contents now in SST).
    ///
    /// If the process crashes between steps 2 and 4, recovery sees `.old`
    /// and deletes it (rotation invariant: `.old` always post-dates a
    /// successful `persist()`).
    pub fn checkpoint(&mut self) -> StorageResult<()> {
        // Flush kernel write buffer before rename so no writes are lost if
        // the rename crosses a power cycle boundary.
        self.file
            .flush()
            .map_err(|e| StorageError::Io(format!("WAL flush before checkpoint: {e}")))?;

        // 1. Rename current → .old.
        std::fs::rename(&self.path, &self.old_path)
            .map_err(|e| StorageError::Io(format!("WAL rename to .old: {e}")))?;

        // 2. Open fresh WAL.
        let new_file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.path)
            .map_err(|e| {
                StorageError::Io(format!("WAL create fresh file after checkpoint: {e}"))
            })?;
        self.file = new_file;

        // 3. Delete .old — data is in SST, safe to remove.
        if self.old_path.exists() {
            std::fs::remove_file(&self.old_path)
                .map_err(|e| StorageError::Io(format!("WAL delete .old: {e}")))?;
        }

        tracing::debug!(path = %self.path.display(), "WAL checkpoint (rotation) complete");
        Ok(())
    }

    /// Delete the WAL file from disk.
    ///
    /// Used after a successful replay + persist sequence to leave the engine
    /// in a clean state before resuming normal writes.
    #[allow(dead_code)] // used in tests; may be called by external recovery tooling
    pub fn delete_file(path: &Path) -> StorageResult<()> {
        if path.exists() {
            std::fs::remove_file(path)
                .map_err(|e| StorageError::Io(format!("WAL delete '{}': {e}", path.display())))?;
        }
        Ok(())
    }

    /// Return the active WAL file path.
    #[allow(dead_code)] // used in tests and future diagnostics tooling
    pub fn path(&self) -> &Path {
        &self.path
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use coordinode_core::txn::proposal::{Mutation, PartitionId};
    use tempfile::TempDir;

    use super::*;

    fn make_mutations(n: usize) -> Vec<Mutation> {
        (0..n)
            .map(|i| Mutation::Put {
                partition: PartitionId::Node,
                key: format!("key:{i}").into_bytes(),
                value: format!("val:{i}").into_bytes(),
            })
            .collect()
    }

    // ── open / append / replay ────────────────────────────────────────────────

    /// Fresh open produces empty replay and a usable WAL.
    #[test]
    fn fresh_open_no_replay() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("standalone.wal");
        let (mut wal, records) = StandaloneWal::open(path, WalSyncPolicy::SyncPerRecord).unwrap();
        assert!(
            records.is_empty(),
            "fresh WAL should have no replay records"
        );
        // Append one record to verify WAL is operational.
        let lsn = wal.append(&make_mutations(3)).unwrap();
        assert_eq!(lsn, 0, "first record gets lsn 0");
    }

    /// Records appended to the WAL survive a close+reopen.
    #[test]
    fn append_and_replay() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("standalone.wal");

        let mutations_a = make_mutations(2);
        let mutations_b = vec![Mutation::Delete {
            partition: PartitionId::Schema,
            key: b"schema:user".to_vec(),
        }];

        // Write two records.
        {
            let (mut wal, _) =
                StandaloneWal::open(path.clone(), WalSyncPolicy::SyncPerRecord).unwrap();
            wal.append(&mutations_a).unwrap();
            wal.append(&mutations_b).unwrap();
        }

        // Reopen — both records must replay.
        let (_, records) = StandaloneWal::open(path, WalSyncPolicy::SyncPerRecord).unwrap();
        assert_eq!(records.len(), 2, "two records must replay");
        assert_eq!(records[0].lsn, 0);
        assert_eq!(records[0].mutations, mutations_a);
        assert_eq!(records[1].lsn, 1);
        assert_eq!(records[1].mutations, mutations_b);
    }

    /// LSN is monotonically increasing across appends.
    #[test]
    fn lsn_monotone() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("standalone.wal");
        let (mut wal, _) = StandaloneWal::open(path, WalSyncPolicy::SyncPerRecord).unwrap();
        let lsns: Vec<u64> = (0..5)
            .map(|_| wal.append(&make_mutations(1)).unwrap())
            .collect();
        assert_eq!(lsns, vec![0, 1, 2, 3, 4]);
    }

    /// CRC mismatch in the middle stops replay at the corrupt record.
    #[test]
    fn crc_mismatch_stops_replay() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("standalone.wal");

        // Write two valid records.
        {
            let (mut wal, _) =
                StandaloneWal::open(path.clone(), WalSyncPolicy::SyncPerRecord).unwrap();
            wal.append(&make_mutations(1)).unwrap();
            wal.append(&make_mutations(1)).unwrap();
        }

        // Corrupt the payload of the first record by flipping bytes in the payload
        // region (after the 16-byte header).
        {
            let mut data = std::fs::read(&path).unwrap();
            // First record header is 16 bytes, payload starts at byte 16.
            if data.len() > 17 {
                data[16] ^= 0xFF; // corrupt first byte of first payload
            }
            std::fs::write(&path, &data).unwrap();
        }

        // Replay should stop at the first corrupt record → 0 valid records.
        let (_, records) = StandaloneWal::open(path, WalSyncPolicy::SyncPerRecord).unwrap();
        assert_eq!(records.len(), 0, "replay must stop at CRC mismatch");
    }

    /// Truncated tail (crash mid-write) is handled gracefully.
    #[test]
    fn truncated_tail_stops_replay() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("standalone.wal");

        // Write one complete + one partial record.
        {
            let (mut wal, _) =
                StandaloneWal::open(path.clone(), WalSyncPolicy::SyncPerRecord).unwrap();
            wal.append(&make_mutations(1)).unwrap();
        }

        // Append a partial header (fewer than 16 bytes) simulating crash mid-write.
        {
            let mut file = OpenOptions::new().append(true).open(&path).unwrap();
            file.write_all(&[0xDE, 0xAD, 0xBE, 0xEF]).unwrap(); // partial header
        }

        let (_, records) = StandaloneWal::open(path, WalSyncPolicy::SyncPerRecord).unwrap();
        // First complete record must replay; partial header is silently discarded.
        assert_eq!(
            records.len(),
            1,
            "complete record before corrupt tail must replay"
        );
    }

    // ── checkpoint (rotation) ─────────────────────────────────────────────────

    /// After checkpoint, the WAL file is empty and replay produces no records.
    #[test]
    fn checkpoint_clears_wal() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("standalone.wal");

        let (mut wal, _) = StandaloneWal::open(path.clone(), WalSyncPolicy::SyncPerRecord).unwrap();
        wal.append(&make_mutations(3)).unwrap();

        // Simulate post-persist checkpoint.
        wal.checkpoint().unwrap();

        // After checkpoint: .old is gone, fresh .wal exists, no records to replay.
        let old_path = StandaloneWal::old_path_for(&path);
        assert!(!old_path.exists(), ".old must be deleted after checkpoint");

        let (_, records) = StandaloneWal::open(path, WalSyncPolicy::SyncPerRecord).unwrap();
        assert!(
            records.is_empty(),
            "checkpointed WAL has no records to replay"
        );
    }

    /// Recovery after simulated crash between rotation rename and delete (.old exists).
    #[test]
    fn recovery_with_stale_old_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("standalone.wal");
        let old_path = StandaloneWal::old_path_for(&path);

        // Simulate a .old file left by a crashed rotation (data already in SST).
        std::fs::write(&old_path, b"stale data - already in SST").unwrap();
        // No active WAL.
        assert!(!path.exists());

        // open() must delete .old and start fresh.
        let (_, records) = StandaloneWal::open(path.clone(), WalSyncPolicy::SyncPerRecord).unwrap();
        assert!(records.is_empty(), ".old contents must not be replayed");
        assert!(!old_path.exists(), ".old must be deleted on open");
        assert!(path.exists(), "fresh WAL must be created");
    }

    // ── delete_file ───────────────────────────────────────────────────────────

    /// delete_file removes the WAL file.
    #[test]
    fn delete_file_removes_wal() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("standalone.wal");

        {
            let (mut wal, _) =
                StandaloneWal::open(path.clone(), WalSyncPolicy::SyncPerRecord).unwrap();
            wal.append(&make_mutations(1)).unwrap();
        }
        assert!(path.exists());
        StandaloneWal::delete_file(&path).unwrap();
        assert!(!path.exists());
    }

    /// delete_file is a no-op when the file does not exist.
    #[test]
    fn delete_file_missing_is_noop() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("nonexistent.wal");
        assert!(StandaloneWal::delete_file(&path).is_ok());
    }

    // ── NoSync policy ─────────────────────────────────────────────────────────

    /// NoSync mode appends records without fsync — records still replay correctly.
    #[test]
    fn nosync_append_and_replay() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("standalone.wal");

        {
            let (mut wal, _) = StandaloneWal::open(path.clone(), WalSyncPolicy::NoSync).unwrap();
            for _ in 0..10 {
                wal.append(&make_mutations(2)).unwrap();
            }
        }

        let (_, records) = StandaloneWal::open(path, WalSyncPolicy::SyncPerRecord).unwrap();
        assert_eq!(records.len(), 10);
    }

    // ── Merge mutations ───────────────────────────────────────────────────────

    /// Merge mutations (used for Adj posting lists) survive WAL round-trip.
    #[test]
    fn merge_mutation_survives_wal() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("standalone.wal");

        let merge = Mutation::Merge {
            partition: PartitionId::Adj,
            key: b"adj:KNOWS:out:42".to_vec(),
            operand: vec![1, 2, 3, 4],
        };

        {
            let (mut wal, _) =
                StandaloneWal::open(path.clone(), WalSyncPolicy::SyncPerRecord).unwrap();
            wal.append(std::slice::from_ref(&merge)).unwrap();
        }

        let (_, records) = StandaloneWal::open(path, WalSyncPolicy::SyncPerRecord).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].mutations, vec![merge]);
    }
}
