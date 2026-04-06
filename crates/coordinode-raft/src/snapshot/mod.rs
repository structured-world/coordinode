//! Snapshot serialization and streaming for Raft log compaction.
//!
//! ## Snapshot Format
//!
//! A snapshot captures all KV data across all storage partitions at a
//! consistent point in time. The binary format is:
//!
//! ```text
//! [magic: 4 bytes "CNSN"]
//! [version: 1 byte (currently 1)]
//! [partition_count: 1 byte]
//! [partition_block]*
//! [checksum: 8 bytes (xxh3 of all preceding bytes)]
//! ```
//!
//! Each `partition_block`:
//! ```text
//! [partition_tag: 1 byte (Partition enum ordinal)]
//! [entry_count: 4 bytes (u32 big-endian)]
//! [kv_entry]*
//! ```
//!
//! Each `kv_entry`:
//! ```text
//! [key_len: 4 bytes (u32 big-endian)]
//! [key: key_len bytes]
//! [value_len: 4 bytes (u32 big-endian)]
//! [value: value_len bytes]
//! ```

use std::io::{self, Read as IoRead};

use serde::{Deserialize, Serialize};

use coordinode_storage::engine::batch::WriteBatch;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::Guard;

use coordinode_core::txn::timestamp::Timestamp;

use crate::storage::{SnapshotMeta, Vote};

/// Transfer message for sending a snapshot from leader to follower over gRPC.
///
/// Contains the leader's vote (for validation), snapshot metadata, and the
/// full or incremental binary snapshot data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotTransfer {
    /// Leader's current vote (follower validates leadership).
    pub vote: Vote,
    /// Snapshot metadata: last_log_id, membership, snapshot_id.
    pub meta: SnapshotMeta,
    /// Snapshot data (binary format v1 full or v2 incremental).
    pub data: Vec<u8>,
    /// If set, this is an incremental snapshot containing only changes
    /// after this timestamp. The receiver must already have data up to
    /// this point. `None` = full snapshot.
    #[serde(default)]
    pub since_ts: Option<u64>,
}

/// A list of KV entries for a single partition.
type PartitionEntries = Vec<(Vec<u8>, Vec<u8>)>;

/// Snapshot magic bytes: "CNSN" (CoordiNode SNapshot).
const MAGIC: &[u8; 4] = b"CNSN";

/// Current snapshot format version.
const FORMAT_VERSION: u8 = 1;

/// Partition tag values. Must be stable across versions.
///
/// `Raft` partition is excluded from snapshots (it contains Raft log entries
/// that are serialized separately by the Raft snapshot mechanism itself).
fn partition_tag(p: Partition) -> u8 {
    match p {
        Partition::Node => 0,
        Partition::Adj => 1,
        Partition::EdgeProp => 2,
        Partition::Blob => 3,
        Partition::BlobRef => 4,
        Partition::Schema => 5,
        Partition::Idx => 6,
        Partition::Raft => 7,
        Partition::Counter => 8,
    }
}

fn tag_to_partition(tag: u8) -> Option<Partition> {
    match tag {
        0 => Some(Partition::Node),
        1 => Some(Partition::Adj),
        2 => Some(Partition::EdgeProp),
        3 => Some(Partition::Blob),
        4 => Some(Partition::BlobRef),
        5 => Some(Partition::Schema),
        6 => Some(Partition::Idx),
        // Raft partition (7) is handled below — see snapshot_partitions()
        7 => Some(Partition::Raft),
        8 => Some(Partition::Counter),
        _ => None,
    }
}

/// Partitions included in data snapshots.
///
/// `Partition::Raft` is excluded — its contents (log entries, votes) are
/// managed by openraft's own snapshot mechanism, not as user data.
fn snapshot_partitions() -> impl Iterator<Item = Partition> {
    Partition::all()
        .iter()
        .copied()
        .filter(|&p| p != Partition::Raft)
}

/// Build a full snapshot of all user-data storage partitions.
///
/// Iterates all 7 user-data partitions (excludes `Partition::Raft` which is
/// managed by openraft), serializes every KV pair, and returns the complete
/// snapshot as bytes. Uses xxh3 checksum for integrity.
///
/// This is called by `CoordinodeSnapshotBuilder::build_snapshot()`.
pub fn build_full_snapshot(engine: &StorageEngine) -> io::Result<Vec<u8>> {
    let partitions: Vec<Partition> = snapshot_partitions().collect();
    let mut buf = Vec::with_capacity(64 * 1024); // Start with 64KB

    // Header
    buf.extend_from_slice(MAGIC);
    buf.push(FORMAT_VERSION);
    buf.push(partitions.len() as u8);

    for &part in &partitions {
        let tag = partition_tag(part);

        // Collect all KV entries for this partition.
        // Use an empty prefix to scan ALL keys in the partition.
        let iter = engine
            .prefix_scan(part, &[])
            .map_err(|e| io::Error::other(format!("snapshot scan {}: {e}", part.name())))?;

        let mut entries: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        for guard in iter {
            let (key, value) = guard
                .into_inner()
                .map_err(|e| io::Error::other(format!("snapshot iter {}: {e}", part.name())))?;
            entries.push((key.to_vec(), value.to_vec()));
        }

        // Partition block header
        buf.push(tag);
        let count = entries.len() as u32;
        buf.extend_from_slice(&count.to_be_bytes());

        // KV entries
        for (key, value) in &entries {
            let key_len = key.len() as u32;
            buf.extend_from_slice(&key_len.to_be_bytes());
            buf.extend_from_slice(key);
            let value_len = value.len() as u32;
            buf.extend_from_slice(&value_len.to_be_bytes());
            buf.extend_from_slice(value);
        }

        tracing::debug!(
            partition = part.name(),
            entries = count,
            "snapshot: serialized partition"
        );
    }

    // xxh3 checksum of everything before
    let hash = xxh3_hash(&buf);
    buf.extend_from_slice(&hash.to_le_bytes());

    tracing::info!(
        total_bytes = buf.len(),
        partitions = partitions.len(),
        "snapshot: build complete"
    );

    Ok(buf)
}

fn collect_kv(
    engine: &StorageEngine,
    part: Partition,
    seqno: u64,
) -> io::Result<Vec<(Vec<u8>, Vec<u8>)>> {
    let iter = engine
        .prefix_scan_at(part, &[], seqno)
        .map_err(|e| io::Error::other(format!("incr scan {}: {e}", part.name())))?;
    let mut result = Vec::new();
    for guard in iter {
        let (k, v) = guard
            .into_inner()
            .map_err(|e| io::Error::other(format!("incr iter {}: {e}", part.name())))?;
        result.push((k.to_vec(), v.to_vec()));
    }
    Ok(result)
}

fn collect_kv_current(
    engine: &StorageEngine,
    part: Partition,
) -> io::Result<Vec<(Vec<u8>, Vec<u8>)>> {
    let iter = engine
        .prefix_scan(part, &[])
        .map_err(|e| io::Error::other(format!("incr scan {}: {e}", part.name())))?;
    let mut result = Vec::new();
    for guard in iter {
        let (k, v) = guard
            .into_inner()
            .map_err(|e| io::Error::other(format!("incr iter {}: {e}", part.name())))?;
        result.push((k.to_vec(), v.to_vec()));
    }
    Ok(result)
}

/// Build an incremental snapshot containing only KV pairs modified after `since_ts`.
///
/// Uses native seqno MVCC (ADR-016): compares the current state against a
/// point-in-time snapshot at `since_ts` via sorted merge-diff. Entries that
/// are new, changed, or deleted since `since_ts` are included.
///
/// Schema-partition keys are always included for consistency (Dgraph pattern).
///
/// The binary format is identical to full snapshots (v1) but only contains
/// the delta entries. The receiver applies these via merge-write (overwrite
/// matching keys, no stale key cleanup).
///
/// Returns `Ok(None)` if no changes exist after `since_ts` (nothing to send).
pub fn build_incremental_snapshot(
    engine: &StorageEngine,
    since_ts: Timestamp,
) -> io::Result<Option<Vec<u8>>> {
    let partitions: Vec<Partition> = snapshot_partitions().collect();
    let mut buf = Vec::with_capacity(64 * 1024);
    let mut total_entries = 0u32;

    // Header (same v1 format — receiver distinguishes via SnapshotTransfer.since_ts)
    buf.extend_from_slice(MAGIC);
    buf.push(FORMAT_VERSION);
    buf.push(partitions.len() as u8);

    // The caller passes `since_ts = engine.snapshot()` which already returns
    // `seqno_counter.get()` — a value strictly greater than the last write's
    // seqno. Combined with lsm-tree's strict `<` filter, `prefix_scan_at(ts)`
    // sees exactly the writes up to and including the snapshot boundary.
    let old_seqno = since_ts.as_raw();

    for part in partitions {
        let tag = partition_tag(part);
        let mut changed: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();

        if part == Partition::Schema {
            // Schema partition: always include ALL keys regardless of since_ts.
            // Dgraph pattern: schema/type keys are always sent for consistency.
            let iter = engine
                .prefix_scan(part, &[])
                .map_err(|e| io::Error::other(format!("incr scan {}: {e}", part.name())))?;
            for guard in iter {
                let (raw_key, raw_value) = guard
                    .into_inner()
                    .map_err(|e| io::Error::other(format!("incr iter {}: {e}", part.name())))?;
                changed.push((raw_key.to_vec(), raw_value.to_vec()));
            }
        } else {
            // Native seqno MVCC diff: compare old snapshot vs current to find changes.
            // Both iterators yield entries in sorted key order, enabling merge-join.
            let old_entries = collect_kv(engine, part, old_seqno)?;
            let cur_entries = collect_kv_current(engine, part)?;

            // Sorted merge-diff to find new, changed, and deleted entries.
            let mut oi = 0;
            let mut ci = 0;
            while oi < old_entries.len() || ci < cur_entries.len() {
                match (old_entries.get(oi), cur_entries.get(ci)) {
                    (Some((ok, _)), Some((ck, cv))) => match ok.as_slice().cmp(ck.as_slice()) {
                        std::cmp::Ordering::Less => {
                            // Key in old but not in current → deleted (tombstone)
                            changed.push((ok.clone(), Vec::new()));
                            oi += 1;
                        }
                        std::cmp::Ordering::Greater => {
                            // Key in current but not in old → new entry
                            changed.push((ck.clone(), cv.clone()));
                            ci += 1;
                        }
                        std::cmp::Ordering::Equal => {
                            // Key in both → include only if value changed
                            if old_entries[oi].1 != *cv {
                                changed.push((ck.clone(), cv.clone()));
                            }
                            oi += 1;
                            ci += 1;
                        }
                    },
                    (Some((ok, _)), None) => {
                        // Remaining old keys were deleted
                        changed.push((ok.clone(), Vec::new()));
                        oi += 1;
                    }
                    (None, Some((ck, cv))) => {
                        // Remaining current keys are new
                        changed.push((ck.clone(), cv.clone()));
                        ci += 1;
                    }
                    (None, None) => break,
                }
            }
        }

        // Write partition block
        buf.push(tag);
        let count = changed.len() as u32;
        buf.extend_from_slice(&count.to_be_bytes());
        total_entries += count;

        for (key, value) in &changed {
            let key_len = key.len() as u32;
            buf.extend_from_slice(&key_len.to_be_bytes());
            buf.extend_from_slice(key);
            let value_len = value.len() as u32;
            buf.extend_from_slice(&value_len.to_be_bytes());
            buf.extend_from_slice(value);
        }

        if count > 0 {
            tracing::debug!(
                partition = part.name(),
                entries = count,
                "incremental snapshot: partition delta"
            );
        }
    }

    if total_entries == 0 {
        tracing::debug!("incremental snapshot: no changes since ts={}", since_ts);
        return Ok(None);
    }

    // Checksum
    let hash = xxh3_hash(&buf);
    buf.extend_from_slice(&hash.to_le_bytes());

    tracing::info!(
        total_bytes = buf.len(),
        total_entries,
        since_ts = %since_ts,
        "incremental snapshot: build complete"
    );

    Ok(Some(buf))
}

/// Install an incremental snapshot: merge-write delta KV pairs into CoordiNode storage.
///
/// Unlike full snapshot installation, incremental install:
/// - Writes only the entries present in the snapshot (merge-write)
/// - Does NOT delete stale keys (receiver already has base data)
/// - Tombstones (empty values) cause key deletion
/// - Schema partition `raft:*` keys are skipped (managed by openraft)
///
/// Uses WriteBatch for atomicity: crash before commit = no partial state.
pub fn install_incremental_snapshot(engine: &StorageEngine, data: &[u8]) -> io::Result<()> {
    let mut cursor = io::Cursor::new(data);

    // Validate header (same format as full snapshot)
    let mut magic = [0u8; 4];
    cursor.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::other("invalid snapshot magic"));
    }

    let mut version = [0u8; 1];
    cursor.read_exact(&mut version)?;
    if version[0] != FORMAT_VERSION {
        return Err(io::Error::other(format!(
            "unsupported snapshot version: {}",
            version[0]
        )));
    }

    let mut part_count = [0u8; 1];
    cursor.read_exact(&mut part_count)?;
    let partition_count = part_count[0] as usize;

    // Validate checksum
    let checksum_offset = data
        .len()
        .checked_sub(8)
        .ok_or_else(|| io::Error::other("snapshot too small for checksum"))?;
    let payload = &data[..checksum_offset];
    let expected_hash = u64::from_le_bytes(
        data[checksum_offset..]
            .try_into()
            .map_err(|_| io::Error::other("invalid checksum bytes"))?,
    );
    let actual_hash = xxh3_hash(payload);
    if expected_hash != actual_hash {
        return Err(io::Error::other(format!(
            "incremental snapshot checksum mismatch: expected {expected_hash:#x}, got {actual_hash:#x}"
        )));
    }

    // Parse and apply delta entries
    let mut batch = WriteBatch::new(engine);
    let mut total_written = 0usize;
    let mut total_deleted = 0usize;

    for _ in 0..partition_count {
        let mut tag_buf = [0u8; 1];
        cursor.read_exact(&mut tag_buf)?;
        let partition = tag_to_partition(tag_buf[0])
            .ok_or_else(|| io::Error::other(format!("unknown partition tag: {}", tag_buf[0])))?;

        let mut count_buf = [0u8; 4];
        cursor.read_exact(&mut count_buf)?;
        let entry_count = u32::from_be_bytes(count_buf) as usize;

        for _ in 0..entry_count {
            let mut key_len_buf = [0u8; 4];
            cursor.read_exact(&mut key_len_buf)?;
            let key_len = u32::from_be_bytes(key_len_buf) as usize;
            let mut key = vec![0u8; key_len];
            cursor.read_exact(&mut key)?;

            let mut value_len_buf = [0u8; 4];
            cursor.read_exact(&mut value_len_buf)?;
            let value_len = u32::from_be_bytes(value_len_buf) as usize;
            let mut value = vec![0u8; value_len];
            cursor.read_exact(&mut value)?;

            // Skip Raft partition entirely — managed by openraft, not user data.
            if partition == Partition::Raft {
                continue;
            }

            // Skip raft: keys in Schema partition
            if partition == Partition::Schema && key.starts_with(b"raft:") {
                continue;
            }

            if value.is_empty() {
                // Tombstone: delete this key from the receiver
                batch.delete(partition, key);
                total_deleted += 1;
            } else {
                batch.put(partition, key, value);
                total_written += 1;
            }
        }
    }

    batch
        .commit()
        .map_err(|e| io::Error::other(format!("incremental snapshot commit failed: {e}")))?;

    tracing::info!(
        total_written,
        total_deleted,
        "incremental snapshot install complete"
    );

    Ok(())
}

/// Install a full snapshot: deserialize and write all KV pairs to CoordiNode storage.
///
/// Uses a **two-phase crash-safe** approach:
///
/// **Phase 1 (atomic via WriteBatch):** Write ALL snapshot entries in a
/// single atomic batch. If crash occurs during Phase 1, no writes are
/// visible — old data remains intact. WriteBatch uses the storage
/// write transaction internally (all-or-nothing commit).
///
/// **Phase 2 (idempotent cleanup):** Delete stale keys that exist in
/// the current engine but are absent in the snapshot. If crash occurs
/// during Phase 2, stale keys remain (harmless — cleaned up on next
/// snapshot install).
///
/// **Important:** Raft keys (`raft:*`) in the Schema partition are
/// always preserved — they're managed by openraft, not application data.
pub fn install_full_snapshot(engine: &StorageEngine, data: &[u8]) -> io::Result<()> {
    let mut cursor = io::Cursor::new(data);

    // Validate header
    let mut magic = [0u8; 4];
    cursor.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::other("invalid snapshot magic"));
    }

    let mut version = [0u8; 1];
    cursor.read_exact(&mut version)?;
    if version[0] != FORMAT_VERSION {
        return Err(io::Error::other(format!(
            "unsupported snapshot version: {}",
            version[0]
        )));
    }

    let mut part_count = [0u8; 1];
    cursor.read_exact(&mut part_count)?;
    let partition_count = part_count[0] as usize;

    // Validate checksum before applying anything
    let checksum_offset = data
        .len()
        .checked_sub(8)
        .ok_or_else(|| io::Error::other("snapshot too small for checksum"))?;
    let payload = &data[..checksum_offset];
    let expected_hash = u64::from_le_bytes(
        data[checksum_offset..]
            .try_into()
            .map_err(|_| io::Error::other("invalid checksum bytes"))?,
    );
    let actual_hash = xxh3_hash(payload);
    if expected_hash != actual_hash {
        return Err(io::Error::other(format!(
            "snapshot checksum mismatch: expected {expected_hash:#x}, got {actual_hash:#x}"
        )));
    }

    // ── Parse all entries first (before mutating anything) ───────────
    // Collect entries per partition so we can do atomic Phase 1 + Phase 2.
    let mut parsed_partitions: Vec<(Partition, PartitionEntries)> = Vec::new();

    for _ in 0..partition_count {
        let mut tag_buf = [0u8; 1];
        cursor.read_exact(&mut tag_buf)?;
        let partition = tag_to_partition(tag_buf[0])
            .ok_or_else(|| io::Error::other(format!("unknown partition tag: {}", tag_buf[0])))?;

        let mut count_buf = [0u8; 4];
        cursor.read_exact(&mut count_buf)?;
        let entry_count = u32::from_be_bytes(count_buf) as usize;

        let mut entries = Vec::with_capacity(entry_count);
        for _ in 0..entry_count {
            let mut key_len_buf = [0u8; 4];
            cursor.read_exact(&mut key_len_buf)?;
            let key_len = u32::from_be_bytes(key_len_buf) as usize;
            let mut key = vec![0u8; key_len];
            cursor.read_exact(&mut key)?;

            let mut value_len_buf = [0u8; 4];
            cursor.read_exact(&mut value_len_buf)?;
            let value_len = u32::from_be_bytes(value_len_buf) as usize;
            let mut value = vec![0u8; value_len];
            cursor.read_exact(&mut value)?;

            // Skip Raft partition entirely — managed by openraft, not user data.
            if partition == Partition::Raft {
                continue;
            }

            // Skip raft: keys in Schema partition — managed by openraft
            if partition == Partition::Schema && key.starts_with(b"raft:") {
                continue;
            }

            entries.push((key, value));
        }

        // Don't accumulate Raft partition entries into parsed_partitions
        if partition != Partition::Raft {
            parsed_partitions.push((partition, entries));
        }
    }

    // ── Phase 1: Atomic write of ALL snapshot entries ────────────────
    // WriteBatch uses the storage write_tx internally — all-or-nothing.
    // Crash before commit() = no writes visible, old data intact.
    let mut batch = WriteBatch::new(engine);
    let mut total_written = 0usize;

    for (partition, entries) in &parsed_partitions {
        for (key, value) in entries {
            batch.put(*partition, key.clone(), value.clone());
            total_written += 1;
        }
    }

    batch
        .commit()
        .map_err(|e| io::Error::other(format!("snapshot phase 1 (atomic write) failed: {e}")))?;

    tracing::info!(
        entries = total_written,
        "snapshot phase 1 complete: atomic write committed"
    );

    // ── Phase 2: Idempotent cleanup of stale keys ────────────────────
    // Delete keys present in engine but absent in snapshot.
    // Crash during Phase 2 = stale keys remain (harmless, cleaned on
    // next snapshot install or next full snapshot from leader).
    let mut stale_deleted = 0usize;

    for (partition, snap_entries) in &parsed_partitions {
        // Build a set of snapshot keys for fast lookup
        let snap_keys: std::collections::HashSet<&[u8]> =
            snap_entries.iter().map(|(k, _)| k.as_slice()).collect();

        let iter = engine
            .prefix_scan(*partition, &[])
            .map_err(|e| io::Error::other(format!("cleanup scan {}: {e}", partition.name())))?;

        let mut keys_to_delete = Vec::new();
        for guard in iter {
            let key = guard
                .key()
                .map_err(|e| io::Error::other(format!("cleanup iter {}: {e}", partition.name())))?;

            // Preserve raft: keys in Schema partition
            if *partition == Partition::Schema && key.as_ref().starts_with(b"raft:") {
                continue;
            }

            if !snap_keys.contains(key.as_ref()) {
                keys_to_delete.push(key.to_vec());
            }
        }

        for key in &keys_to_delete {
            engine.delete(*partition, key).map_err(|e| {
                io::Error::other(format!("cleanup delete {}: {e}", partition.name()))
            })?;
        }

        if !keys_to_delete.is_empty() {
            tracing::debug!(
                partition = partition.name(),
                stale_keys = keys_to_delete.len(),
                "snapshot phase 2: cleaned stale keys"
            );
        }
        stale_deleted += keys_to_delete.len();
    }

    tracing::info!(
        total_written,
        stale_deleted,
        "snapshot install complete (two-phase)"
    );
    Ok(())
}

// ── Chunked Snapshot Transfer Protocol ──────────────────────────────
//
// For large snapshots (>1GB), sending the entire CNSN blob in a single
// gRPC message causes OOM on both sender and receiver. The chunked
// protocol splits the transfer into:
//
//   Message 1: SnapshotTransferHeader (vote, meta, since_ts, data_size)
//   Messages 2..N: Raw CNSN data chunks (up to SNAPSHOT_CHUNK_SIZE each)
//
// The receiver writes chunks to a temp file, then installs from the file
// via reader-based installers. Memory usage: O(SNAPSHOT_CHUNK_SIZE).

/// Maximum size of a single snapshot data chunk in bytes (4 MB).
pub const SNAPSHOT_CHUNK_SIZE: usize = 4 * 1024 * 1024;

/// Header message for chunked snapshot transfer.
///
/// Sent as the first gRPC message. Contains all metadata needed to
/// prepare for receiving the snapshot data chunks that follow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotTransferHeader {
    /// Leader's current vote (follower validates leadership).
    pub vote: Vote,
    /// Snapshot metadata: last_log_id, membership, snapshot_id.
    pub meta: SnapshotMeta,
    /// Total size of the CNSN data that follows in subsequent chunks.
    pub data_size: u64,
    /// If set, this is an incremental snapshot containing only changes
    /// after this timestamp. `None` = full snapshot.
    #[serde(default)]
    pub since_ts: Option<u64>,
}

/// A single message in the chunked snapshot transfer stream.
///
/// The first message is always a Header. Subsequent messages are DataChunks
/// containing raw CNSN bytes. The receiver accumulates data chunks until
/// `data_size` bytes have been received.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SnapshotChunkMessage {
    /// First message: metadata about the snapshot transfer.
    Header(SnapshotTransferHeader),
    /// Subsequent messages: raw CNSN data bytes (up to SNAPSHOT_CHUNK_SIZE).
    DataChunk(Vec<u8>),
}

/// Split snapshot CNSN data into chunks for streaming transfer.
///
/// Returns an iterator of byte vectors, each up to `SNAPSHOT_CHUNK_SIZE`.
pub fn chunk_snapshot_data(data: &[u8]) -> impl Iterator<Item = &[u8]> {
    data.chunks(SNAPSHOT_CHUNK_SIZE)
}

/// Install a full snapshot from a reader (file or buffer).
///
/// Identical to `install_full_snapshot` but reads from `impl Read` instead
/// of `&[u8]`, avoiding the need to hold the entire snapshot in memory.
///
/// Uses a **two-phase crash-safe** approach:
/// - Phase 1 (atomic WriteBatch): Write ALL snapshot entries.
/// - Phase 2 (idempotent cleanup): Delete stale keys.
pub fn install_full_snapshot_from_reader(
    engine: &StorageEngine,
    reader: &mut impl IoRead,
) -> io::Result<()> {
    // Read and validate header
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::other("invalid snapshot magic"));
    }

    let mut version = [0u8; 1];
    reader.read_exact(&mut version)?;
    if version[0] != FORMAT_VERSION {
        return Err(io::Error::other(format!(
            "unsupported snapshot version: {}",
            version[0]
        )));
    }

    let mut part_count = [0u8; 1];
    reader.read_exact(&mut part_count)?;
    let partition_count = part_count[0] as usize;

    // Parse all entries (accumulates in memory for WriteBatch atomicity).
    // Note: individual entries are small (KV pairs). The OOM issue was
    // holding the entire serialized CNSN blob; here we parse incrementally.
    let mut parsed_partitions: Vec<(Partition, PartitionEntries)> = Vec::new();

    for _ in 0..partition_count {
        let mut tag_buf = [0u8; 1];
        reader.read_exact(&mut tag_buf)?;
        let partition = tag_to_partition(tag_buf[0])
            .ok_or_else(|| io::Error::other(format!("unknown partition tag: {}", tag_buf[0])))?;

        let mut count_buf = [0u8; 4];
        reader.read_exact(&mut count_buf)?;
        let entry_count = u32::from_be_bytes(count_buf) as usize;

        let mut entries = Vec::with_capacity(entry_count);
        for _ in 0..entry_count {
            let mut key_len_buf = [0u8; 4];
            reader.read_exact(&mut key_len_buf)?;
            let key_len = u32::from_be_bytes(key_len_buf) as usize;
            let mut key = vec![0u8; key_len];
            reader.read_exact(&mut key)?;

            let mut value_len_buf = [0u8; 4];
            reader.read_exact(&mut value_len_buf)?;
            let value_len = u32::from_be_bytes(value_len_buf) as usize;
            let mut value = vec![0u8; value_len];
            reader.read_exact(&mut value)?;

            if partition == Partition::Raft {
                continue;
            }
            if partition == Partition::Schema && key.starts_with(b"raft:") {
                continue;
            }

            entries.push((key, value));
        }

        if partition != Partition::Raft {
            parsed_partitions.push((partition, entries));
        }
    }

    // Read and validate checksum (last 8 bytes after partition data).
    // For reader-based install, we cannot random-access the checksum
    // at the end before reading entries. Instead, we verify it after
    // reading all partition data but BEFORE applying any writes.
    let mut checksum_buf = [0u8; 8];
    reader.read_exact(&mut checksum_buf)?;
    let expected_hash = u64::from_le_bytes(checksum_buf);

    // Recompute hash over the data we parsed (reconstruct the CNSN payload).
    // This is necessary because we read streaming — we couldn't validate
    // the checksum upfront like the &[u8] version does.
    let reconstructed = reconstruct_cnsn_payload(&parsed_partitions, partition_count);
    let actual_hash = xxh3_hash(&reconstructed);
    if expected_hash != actual_hash {
        return Err(io::Error::other(format!(
            "snapshot checksum mismatch: expected {expected_hash:#x}, got {actual_hash:#x}"
        )));
    }

    // Phase 1: Atomic write
    let mut batch = WriteBatch::new(engine);
    let mut total_written = 0usize;

    for (partition, entries) in &parsed_partitions {
        for (key, value) in entries {
            batch.put(*partition, key.clone(), value.clone());
            total_written += 1;
        }
    }

    batch
        .commit()
        .map_err(|e| io::Error::other(format!("snapshot phase 1 (atomic write) failed: {e}")))?;

    tracing::info!(
        entries = total_written,
        "snapshot phase 1 complete: atomic write committed"
    );

    // Phase 2: Stale key cleanup
    let mut stale_deleted = 0usize;
    for (partition, snap_entries) in &parsed_partitions {
        let snap_keys: std::collections::HashSet<&[u8]> =
            snap_entries.iter().map(|(k, _)| k.as_slice()).collect();

        let iter = engine
            .prefix_scan(*partition, &[])
            .map_err(|e| io::Error::other(format!("cleanup scan {}: {e}", partition.name())))?;

        let mut keys_to_delete = Vec::new();
        for guard in iter {
            let key = guard
                .key()
                .map_err(|e| io::Error::other(format!("cleanup iter {}: {e}", partition.name())))?;

            if *partition == Partition::Schema && key.as_ref().starts_with(b"raft:") {
                continue;
            }
            if !snap_keys.contains(key.as_ref()) {
                keys_to_delete.push(key.to_vec());
            }
        }

        for key in &keys_to_delete {
            engine.delete(*partition, key).map_err(|e| {
                io::Error::other(format!("cleanup delete {}: {e}", partition.name()))
            })?;
        }

        if !keys_to_delete.is_empty() {
            tracing::debug!(
                partition = partition.name(),
                stale_keys = keys_to_delete.len(),
                "snapshot phase 2: cleaned stale keys"
            );
        }
        stale_deleted += keys_to_delete.len();
    }

    tracing::info!(
        total_written,
        stale_deleted,
        "snapshot install complete (two-phase, reader)"
    );
    Ok(())
}

/// Install an incremental snapshot from a reader (file or buffer).
///
/// Identical to `install_incremental_snapshot` but reads from `impl Read`.
pub fn install_incremental_snapshot_from_reader(
    engine: &StorageEngine,
    reader: &mut impl IoRead,
) -> io::Result<()> {
    // Read and validate header
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::other("invalid snapshot magic"));
    }

    let mut version = [0u8; 1];
    reader.read_exact(&mut version)?;
    if version[0] != FORMAT_VERSION {
        return Err(io::Error::other(format!(
            "unsupported snapshot version: {}",
            version[0]
        )));
    }

    let mut part_count = [0u8; 1];
    reader.read_exact(&mut part_count)?;
    let partition_count = part_count[0] as usize;

    // Parse delta entries and apply via WriteBatch
    let mut batch = WriteBatch::new(engine);
    let mut total_written = 0usize;
    let mut total_deleted = 0usize;

    // Track parsed entries for checksum verification
    let mut parsed_partitions: Vec<(Partition, PartitionEntries)> = Vec::new();

    for _ in 0..partition_count {
        let mut tag_buf = [0u8; 1];
        reader.read_exact(&mut tag_buf)?;
        let partition = tag_to_partition(tag_buf[0])
            .ok_or_else(|| io::Error::other(format!("unknown partition tag: {}", tag_buf[0])))?;

        let mut count_buf = [0u8; 4];
        reader.read_exact(&mut count_buf)?;
        let entry_count = u32::from_be_bytes(count_buf) as usize;

        let mut entries = Vec::with_capacity(entry_count);
        for _ in 0..entry_count {
            let mut key_len_buf = [0u8; 4];
            reader.read_exact(&mut key_len_buf)?;
            let key_len = u32::from_be_bytes(key_len_buf) as usize;
            let mut key = vec![0u8; key_len];
            reader.read_exact(&mut key)?;

            let mut value_len_buf = [0u8; 4];
            reader.read_exact(&mut value_len_buf)?;
            let value_len = u32::from_be_bytes(value_len_buf) as usize;
            let mut value = vec![0u8; value_len];
            reader.read_exact(&mut value)?;

            entries.push((key, value));
        }

        parsed_partitions.push((partition, entries));
    }

    // Validate checksum
    let mut checksum_buf = [0u8; 8];
    reader.read_exact(&mut checksum_buf)?;
    let expected_hash = u64::from_le_bytes(checksum_buf);
    let reconstructed = reconstruct_cnsn_payload(&parsed_partitions, partition_count);
    let actual_hash = xxh3_hash(&reconstructed);
    if expected_hash != actual_hash {
        return Err(io::Error::other(format!(
            "incremental snapshot checksum mismatch: expected {expected_hash:#x}, got {actual_hash:#x}"
        )));
    }

    // Apply entries
    for (partition, entries) in &parsed_partitions {
        for (key, value) in entries {
            if *partition == Partition::Raft {
                continue;
            }
            if *partition == Partition::Schema && key.starts_with(b"raft:") {
                continue;
            }

            if value.is_empty() {
                batch.delete(*partition, key.clone());
                total_deleted += 1;
            } else {
                batch.put(*partition, key.clone(), value.clone());
                total_written += 1;
            }
        }
    }

    batch
        .commit()
        .map_err(|e| io::Error::other(format!("incremental snapshot commit failed: {e}")))?;

    tracing::info!(
        total_written,
        total_deleted,
        "incremental snapshot install complete (reader)"
    );

    Ok(())
}

/// Reconstruct the CNSN binary payload (without checksum) from parsed partitions.
///
/// Used by reader-based installers to verify the checksum after streaming
/// parse, since they can't random-access the checksum before reading.
fn reconstruct_cnsn_payload(
    parsed_partitions: &[(Partition, PartitionEntries)],
    partition_count: usize,
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(64 * 1024);
    buf.extend_from_slice(MAGIC);
    buf.push(FORMAT_VERSION);
    buf.push(partition_count as u8);

    for (partition, entries) in parsed_partitions {
        buf.push(partition_tag(*partition));
        let count = entries.len() as u32;
        buf.extend_from_slice(&count.to_be_bytes());

        for (key, value) in entries {
            let key_len = key.len() as u32;
            buf.extend_from_slice(&key_len.to_be_bytes());
            buf.extend_from_slice(key);
            let value_len = value.len() as u32;
            buf.extend_from_slice(&value_len.to_be_bytes());
            buf.extend_from_slice(value);
        }
    }

    buf
}

/// xxh3 hash (64-bit) using the standard library-compatible algorithm.
///
/// We use a simple FNV-like hash here for now. In production, replace
/// with xxhash-rust crate for SIMD-accelerated xxh3.
fn xxh3_hash(data: &[u8]) -> u64 {
    // Use FNV-1a as a placeholder. R134 is pre-alpha; we'll switch to
    // xxhash-rust (xxh3) when the crate is added to workspace deps.
    let mut hash: u64 = 0xcbf29ce484222325; // FNV offset basis
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3); // FNV prime
    }
    hash
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use coordinode_storage::engine::config::StorageConfig;
    use coordinode_storage::engine::core::StorageEngine;
    use tempfile::tempdir;

    fn open_engine(dir: &std::path::Path) -> StorageEngine {
        let config = StorageConfig::new(dir);
        StorageEngine::open(&config).expect("open engine")
    }

    #[test]
    fn test_snapshot_roundtrip_empty_db() {
        let dir = tempdir().unwrap();
        let engine = open_engine(dir.path());

        let data = build_full_snapshot(&engine).unwrap();
        assert!(data.len() > 14); // magic(4) + version(1) + count(1) + checksum(8)

        // Install into fresh engine should succeed
        let dir2 = tempdir().unwrap();
        let engine2 = open_engine(dir2.path());
        install_full_snapshot(&engine2, &data).unwrap();
    }

    #[test]
    fn test_snapshot_roundtrip_with_data() {
        let dir = tempdir().unwrap();
        let engine = open_engine(dir.path());

        // Write some data across partitions
        engine.put(Partition::Node, b"node:0:1", b"alice").unwrap();
        engine.put(Partition::Node, b"node:0:2", b"bob").unwrap();
        engine
            .put(Partition::Adj, b"adj:KNOWS:out:1", b"\x02")
            .unwrap();
        engine
            .put(Partition::EdgeProp, b"edgeprop:KNOWS:1:2", b"since=2020")
            .unwrap();
        engine
            .put(Partition::Schema, b"schema:label:User", b"{}")
            .unwrap();
        engine
            .put(Partition::Idx, b"idx:name:alice:1", b"")
            .unwrap();

        let data = build_full_snapshot(&engine).unwrap();

        // Install into fresh engine
        let dir2 = tempdir().unwrap();
        let engine2 = open_engine(dir2.path());

        install_full_snapshot(&engine2, &data).unwrap();

        // Verify data was restored
        assert_eq!(
            engine2
                .get(Partition::Node, b"node:0:1")
                .unwrap()
                .map(|b| b.to_vec()),
            Some(b"alice".to_vec())
        );
        assert_eq!(
            engine2
                .get(Partition::Node, b"node:0:2")
                .unwrap()
                .map(|b| b.to_vec()),
            Some(b"bob".to_vec())
        );
        assert_eq!(
            engine2
                .get(Partition::Adj, b"adj:KNOWS:out:1")
                .unwrap()
                .map(|b| b.to_vec()),
            Some(b"\x02".to_vec())
        );
        assert_eq!(
            engine2
                .get(Partition::EdgeProp, b"edgeprop:KNOWS:1:2")
                .unwrap()
                .map(|b| b.to_vec()),
            Some(b"since=2020".to_vec())
        );
        assert_eq!(
            engine2
                .get(Partition::Schema, b"schema:label:User")
                .unwrap()
                .map(|b| b.to_vec()),
            Some(b"{}".to_vec())
        );
        assert_eq!(
            engine2
                .get(Partition::Idx, b"idx:name:alice:1")
                .unwrap()
                .map(|b| b.to_vec()),
            Some(b"".to_vec())
        );
    }

    #[test]
    fn test_snapshot_preserves_raft_keys_in_schema() {
        let dir = tempdir().unwrap();
        let engine = open_engine(dir.path());

        // Source has raft keys and application keys
        engine
            .put(Partition::Schema, b"raft:vote", b"raft-data")
            .unwrap();
        engine
            .put(Partition::Schema, b"schema:label:User", b"{}")
            .unwrap();

        let data = build_full_snapshot(&engine).unwrap();

        // Target has different raft keys
        let dir2 = tempdir().unwrap();
        let engine2 = open_engine(dir2.path());
        engine2
            .put(Partition::Schema, b"raft:vote", b"target-raft-data")
            .unwrap();
        engine2
            .put(Partition::Schema, b"schema:label:Old", b"old")
            .unwrap();

        install_full_snapshot(&engine2, &data).unwrap();

        // Raft keys preserved (target's own raft data kept)
        assert_eq!(
            engine2
                .get(Partition::Schema, b"raft:vote")
                .unwrap()
                .map(|b| b.to_vec()),
            Some(b"target-raft-data".to_vec())
        );
        // Application data replaced from snapshot
        assert_eq!(
            engine2
                .get(Partition::Schema, b"schema:label:User")
                .unwrap()
                .map(|b| b.to_vec()),
            Some(b"{}".to_vec())
        );
        // Old application data cleared
        assert!(engine2
            .get(Partition::Schema, b"schema:label:Old")
            .unwrap()
            .is_none());
    }

    #[test]
    fn test_snapshot_checksum_validation() {
        let dir = tempdir().unwrap();
        let engine = open_engine(dir.path());
        let mut data = build_full_snapshot(&engine).unwrap();

        // Corrupt a byte in the payload
        if data.len() > 10 {
            data[5] ^= 0xFF;
        }

        let dir2 = tempdir().unwrap();
        let engine2 = open_engine(dir2.path());
        let result = install_full_snapshot(&engine2, &data);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("checksum mismatch"));
    }

    #[test]
    fn test_snapshot_invalid_magic() {
        let data = b"BADMxxxxxxxx";
        let dir = tempdir().unwrap();
        let engine = open_engine(dir.path());
        let result = install_full_snapshot(&engine, data);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("invalid snapshot magic"));
    }

    #[test]
    fn test_snapshot_replaces_existing_data() {
        let dir = tempdir().unwrap();
        let engine = open_engine(dir.path());
        engine
            .put(Partition::Node, b"node:0:1", b"new-value")
            .unwrap();

        let data = build_full_snapshot(&engine).unwrap();

        // Target has different data
        let dir2 = tempdir().unwrap();
        let engine2 = open_engine(dir2.path());
        engine2
            .put(Partition::Node, b"node:0:1", b"old-value")
            .unwrap();
        engine2
            .put(Partition::Node, b"node:0:99", b"stale")
            .unwrap();

        install_full_snapshot(&engine2, &data).unwrap();

        // Snapshot data overwrites
        assert_eq!(
            engine2
                .get(Partition::Node, b"node:0:1")
                .unwrap()
                .map(|b| b.to_vec()),
            Some(b"new-value".to_vec())
        );
        // Stale data removed
        assert!(engine2
            .get(Partition::Node, b"node:0:99")
            .unwrap()
            .is_none());
    }

    // -- Incremental snapshot tests (R135 / G057: native seqno MVCC) --

    #[test]
    fn test_incremental_no_changes_returns_none() {
        let dir = tempdir().unwrap();
        let engine = open_engine(dir.path());

        engine.put(Partition::Node, b"node:0:1", b"alice").unwrap();
        // Snapshot boundary after write — incremental since this point finds nothing.
        let since = Timestamp::from_raw(engine.snapshot());

        let result = build_incremental_snapshot(&engine, since).unwrap();
        assert!(result.is_none(), "no changes after snapshot boundary");
    }

    #[test]
    fn test_incremental_captures_recent_changes() {
        let dir = tempdir().unwrap();
        let engine = open_engine(dir.path());

        // Phase 1: baseline
        engine.put(Partition::Node, b"node:0:1", b"alice").unwrap();
        engine.put(Partition::Node, b"node:0:2", b"bob").unwrap();
        let since = Timestamp::from_raw(engine.snapshot());

        // Phase 2: changes after baseline
        engine
            .put(Partition::Node, b"node:0:1", b"alice-updated")
            .unwrap();
        engine
            .put(Partition::Node, b"node:0:3", b"charlie")
            .unwrap();

        let data = build_incremental_snapshot(&engine, since)
            .unwrap()
            .expect("should have changes");

        // Install into target with baseline data
        let dir2 = tempdir().unwrap();
        let engine2 = open_engine(dir2.path());
        engine2.put(Partition::Node, b"node:0:1", b"alice").unwrap();
        engine2.put(Partition::Node, b"node:0:2", b"bob").unwrap();

        install_incremental_snapshot(&engine2, &data).unwrap();

        // node:0:1 updated
        assert_eq!(
            engine2
                .get(Partition::Node, b"node:0:1")
                .unwrap()
                .map(|b| b.to_vec()),
            Some(b"alice-updated".to_vec())
        );

        // node:0:3 new
        assert_eq!(
            engine2
                .get(Partition::Node, b"node:0:3")
                .unwrap()
                .map(|b| b.to_vec()),
            Some(b"charlie".to_vec())
        );

        // node:0:2 unchanged (not in delta)
        assert_eq!(
            engine2
                .get(Partition::Node, b"node:0:2")
                .unwrap()
                .map(|b| b.to_vec()),
            Some(b"bob".to_vec())
        );
    }

    #[test]
    fn test_incremental_schema_always_included() {
        let dir = tempdir().unwrap();
        let engine = open_engine(dir.path());

        engine
            .put(Partition::Schema, b"schema:label:User", b"{name:string}")
            .unwrap();
        engine
            .put(Partition::Schema, b"raft:vote", b"raft-data")
            .unwrap();

        // Incremental with a future since_ts should still include schema keys
        let data = build_incremental_snapshot(&engine, Timestamp::from_raw(999999))
            .unwrap()
            .expect("schema keys should always be included");

        let dir2 = tempdir().unwrap();
        let engine2 = open_engine(dir2.path());
        engine2
            .put(Partition::Schema, b"raft:vote", b"target-raft")
            .unwrap();

        install_incremental_snapshot(&engine2, &data).unwrap();

        // Schema key installed
        assert_eq!(
            engine2
                .get(Partition::Schema, b"schema:label:User")
                .unwrap()
                .map(|b| b.to_vec()),
            Some(b"{name:string}".to_vec())
        );
        // Raft key preserved (not overwritten by incremental install)
        assert_eq!(
            engine2
                .get(Partition::Schema, b"raft:vote")
                .unwrap()
                .map(|b| b.to_vec()),
            Some(b"target-raft".to_vec())
        );
    }

    #[test]
    fn test_incremental_checksum_validation() {
        let dir = tempdir().unwrap();
        let engine = open_engine(dir.path());

        engine.put(Partition::Node, b"node:0:1", b"data").unwrap();
        // since_ts=0 ensures the write is captured as a change
        let mut data = build_incremental_snapshot(&engine, Timestamp::from_raw(0))
            .unwrap()
            .expect("should have data");

        // Corrupt payload
        if data.len() > 10 {
            data[8] ^= 0xFF;
        }

        let dir2 = tempdir().unwrap();
        let engine2 = open_engine(dir2.path());
        let result = install_incremental_snapshot(&engine2, &data);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("checksum mismatch"));
    }

    #[test]
    fn test_incremental_detects_deleted_key() {
        // With native seqno MVCC, a key present in old snapshot but absent in
        // current state is detected as a deletion (empty value = tombstone).
        let dir = tempdir().unwrap();
        let engine = open_engine(dir.path());

        engine.put(Partition::Node, b"node:0:1", b"alice").unwrap();
        let since = Timestamp::from_raw(engine.snapshot());

        // Delete the key after the snapshot boundary
        engine.delete(Partition::Node, b"node:0:1").unwrap();

        let data = build_incremental_snapshot(&engine, since)
            .unwrap()
            .expect("deletion should be a change");

        // Target has the key
        let dir2 = tempdir().unwrap();
        let engine2 = open_engine(dir2.path());
        engine2
            .put(Partition::Node, b"node:0:1", b"to-be-deleted")
            .unwrap();

        install_incremental_snapshot(&engine2, &data).unwrap();

        // Key should be deleted
        assert!(
            engine2.get(Partition::Node, b"node:0:1").unwrap().is_none(),
            "tombstone should delete key on receiver"
        );
    }

    #[test]
    fn test_incremental_multiple_partitions() {
        // Write data across Node, Adj, EdgeProp at different seqno phases,
        // build incremental, install on fresh engine, verify.
        let dir = tempdir().unwrap();
        let engine = open_engine(dir.path());

        // Phase 1: baseline
        engine.put(Partition::Node, b"node:0:1", b"alice").unwrap();
        engine
            .put(Partition::Adj, b"adj:KNOWS:out:1", b"\x92\x02\x03")
            .unwrap();
        engine
            .put(Partition::EdgeProp, b"edgeprop:KNOWS:1:2", b"since=2020")
            .unwrap();
        let since = Timestamp::from_raw(engine.snapshot());

        // Phase 2: changes after baseline
        engine
            .put(Partition::Node, b"node:0:1", b"alice-v2")
            .unwrap();
        engine
            .put(Partition::Node, b"node:0:5", b"new-node")
            .unwrap();
        engine
            .put(Partition::Adj, b"adj:LIKES:out:5", b"\x92\x01")
            .unwrap();
        // EdgeProp unchanged

        // Schema always included
        engine
            .put(Partition::Schema, b"schema:label:User", b"{}")
            .unwrap();

        let data = build_incremental_snapshot(&engine, since)
            .unwrap()
            .expect("should have changes");

        // Target: has phase 1 data
        let dir2 = tempdir().unwrap();
        let engine2 = open_engine(dir2.path());
        engine2.put(Partition::Node, b"node:0:1", b"alice").unwrap();
        engine2
            .put(Partition::Adj, b"adj:KNOWS:out:1", b"\x92\x02\x03")
            .unwrap();
        engine2
            .put(Partition::EdgeProp, b"edgeprop:KNOWS:1:2", b"since=2020")
            .unwrap();

        install_incremental_snapshot(&engine2, &data).unwrap();

        // Node updated
        assert_eq!(
            engine2
                .get(Partition::Node, b"node:0:1")
                .unwrap()
                .map(|b| b.to_vec()),
            Some(b"alice-v2".to_vec())
        );
        // New node added
        assert_eq!(
            engine2
                .get(Partition::Node, b"node:0:5")
                .unwrap()
                .map(|b| b.to_vec()),
            Some(b"new-node".to_vec())
        );
        // New adj added
        assert!(engine2
            .get(Partition::Adj, b"adj:LIKES:out:5")
            .unwrap()
            .is_some());
        // Unchanged EdgeProp still exists
        assert!(engine2
            .get(Partition::EdgeProp, b"edgeprop:KNOWS:1:2")
            .unwrap()
            .is_some());
        // Schema installed
        assert_eq!(
            engine2
                .get(Partition::Schema, b"schema:label:User")
                .unwrap()
                .map(|b| b.to_vec()),
            Some(b"{}".to_vec())
        );
    }

    #[test]
    fn test_incremental_is_smaller_than_full() {
        // Verify incremental snapshot is smaller than full when most data is unchanged.
        let dir = tempdir().unwrap();
        let engine = open_engine(dir.path());

        // Write 100 nodes
        for i in 0..100u64 {
            engine
                .put(
                    Partition::Node,
                    format!("node:0:{i:03}").as_bytes(),
                    format!("value-{i}-with-some-payload-data").as_bytes(),
                )
                .unwrap();
        }
        let since = Timestamp::from_raw(engine.snapshot());

        // Update only 3
        engine
            .put(Partition::Node, b"node:0:007", b"updated-7")
            .unwrap();
        engine
            .put(Partition::Node, b"node:0:042", b"updated-42")
            .unwrap();
        engine
            .put(Partition::Node, b"node:0:099", b"updated-99")
            .unwrap();

        let full = build_full_snapshot(&engine).unwrap();
        let incr = build_incremental_snapshot(&engine, since)
            .unwrap()
            .expect("should have changes");

        assert!(
            incr.len() < full.len(),
            "incremental ({} bytes) should be smaller than full ({} bytes)",
            incr.len(),
            full.len()
        );
        // With 100 nodes and only 3 changed, incremental should be much smaller
        assert!(
            incr.len() < full.len() / 5,
            "incremental ({} bytes) should be <20% of full ({} bytes)",
            incr.len(),
            full.len()
        );
    }

    #[test]
    fn test_incremental_with_oracle_engine() {
        // Production path: engine opened with TimestampOracle (HLC-like seqnos).
        // Verifies two-snapshot diff works when seqnos are large, non-contiguous
        // values (e.g., microsecond timestamps) instead of small sequential ints.
        use coordinode_core::txn::timestamp::TimestampOracle;
        use std::sync::Arc;

        let dir = tempdir().unwrap();
        let config = StorageConfig::new(dir.path());
        let oracle = Arc::new(TimestampOracle::resume_from(
            coordinode_core::txn::timestamp::Timestamp::from_raw(1_000_000),
        ));
        let engine = StorageEngine::open_with_oracle(&config, oracle).unwrap();

        // Phase 1: baseline writes (oracle seqnos ~1_000_001+)
        engine.put(Partition::Node, b"node:0:1", b"alice").unwrap();
        engine.put(Partition::Node, b"node:0:2", b"bob").unwrap();
        engine
            .put(Partition::Adj, b"adj:KNOWS:out:1", b"\x92\x02")
            .unwrap();
        let since = Timestamp::from_raw(engine.snapshot());

        // Phase 2: changes
        engine
            .put(Partition::Node, b"node:0:1", b"alice-v2")
            .unwrap();
        engine
            .put(Partition::Node, b"node:0:3", b"charlie")
            .unwrap();
        engine.delete(Partition::Node, b"node:0:2").unwrap();

        let data = build_incremental_snapshot(&engine, since)
            .unwrap()
            .expect("should have changes");

        // Install into fresh engine
        let dir2 = tempdir().unwrap();
        let engine2 = open_engine(dir2.path());
        engine2.put(Partition::Node, b"node:0:1", b"alice").unwrap();
        engine2.put(Partition::Node, b"node:0:2", b"bob").unwrap();
        engine2
            .put(Partition::Adj, b"adj:KNOWS:out:1", b"\x92\x02")
            .unwrap();

        install_incremental_snapshot(&engine2, &data).unwrap();

        // node:0:1 updated
        assert_eq!(
            engine2
                .get(Partition::Node, b"node:0:1")
                .unwrap()
                .map(|b| b.to_vec()),
            Some(b"alice-v2".to_vec())
        );
        // node:0:2 deleted
        assert!(
            engine2.get(Partition::Node, b"node:0:2").unwrap().is_none(),
            "node:0:2 should be deleted"
        );
        // node:0:3 new
        assert_eq!(
            engine2
                .get(Partition::Node, b"node:0:3")
                .unwrap()
                .map(|b| b.to_vec()),
            Some(b"charlie".to_vec())
        );
        // adj unchanged (not in delta)
        assert!(engine2
            .get(Partition::Adj, b"adj:KNOWS:out:1")
            .unwrap()
            .is_some());
    }

    #[test]
    fn test_snapshot_transfer_serde_roundtrip() {
        // Verify SnapshotTransfer with since_ts serializes/deserializes correctly
        use crate::storage::Vote;

        let vote = Vote::new(1, 1);

        let transfer = SnapshotTransfer {
            vote,
            meta: openraft::storage::SnapshotMeta {
                last_log_id: None,
                last_membership: openraft::StoredMembership::default(),
                snapshot_id: "test-snap".to_string(),
            },
            data: vec![1, 2, 3],
            since_ts: Some(42000),
        };
        let bytes = rmp_serde::to_vec(&transfer).expect("serialize");
        let decoded: SnapshotTransfer = rmp_serde::from_slice(&bytes).expect("deserialize");
        assert_eq!(decoded.since_ts, Some(42000));
        assert_eq!(decoded.data, vec![1, 2, 3]);

        // Full snapshot (since_ts = None)
        let full_transfer = SnapshotTransfer {
            vote,
            meta: openraft::storage::SnapshotMeta {
                last_log_id: None,
                last_membership: openraft::StoredMembership::default(),
                snapshot_id: "full-snap".to_string(),
            },
            data: vec![4, 5],
            since_ts: None,
        };
        let bytes2 = rmp_serde::to_vec(&full_transfer).expect("serialize");
        let decoded2: SnapshotTransfer = rmp_serde::from_slice(&bytes2).expect("deserialize");
        assert_eq!(decoded2.since_ts, None);
    }

    // ── Chunked Transfer Protocol Tests ────────────────────────────

    #[test]
    fn test_chunk_snapshot_data_single_chunk() {
        // Data smaller than SNAPSHOT_CHUNK_SIZE → single chunk
        let data = vec![0u8; 100];
        let chunks: Vec<&[u8]> = chunk_snapshot_data(&data).collect();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 100);
    }

    #[test]
    fn test_chunk_snapshot_data_multiple_chunks() {
        // Data larger than SNAPSHOT_CHUNK_SIZE → multiple chunks
        let size = SNAPSHOT_CHUNK_SIZE * 2 + 1000;
        let data = vec![0xABu8; size];
        let chunks: Vec<&[u8]> = chunk_snapshot_data(&data).collect();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].len(), SNAPSHOT_CHUNK_SIZE);
        assert_eq!(chunks[1].len(), SNAPSHOT_CHUNK_SIZE);
        assert_eq!(chunks[2].len(), 1000);

        // Reassembled data matches original
        let reassembled: Vec<u8> = chunks.iter().flat_map(|c| c.iter().copied()).collect();
        assert_eq!(reassembled, data);
    }

    #[test]
    fn test_chunk_snapshot_data_exact_boundary() {
        // Data exactly SNAPSHOT_CHUNK_SIZE → single chunk, no remainder
        let data = vec![0u8; SNAPSHOT_CHUNK_SIZE];
        let chunks: Vec<&[u8]> = chunk_snapshot_data(&data).collect();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), SNAPSHOT_CHUNK_SIZE);
    }

    #[test]
    fn test_chunk_snapshot_data_empty() {
        let data: Vec<u8> = Vec::new();
        let chunks: Vec<&[u8]> = chunk_snapshot_data(&data).collect();
        assert_eq!(chunks.len(), 0);
    }

    #[test]
    fn test_snapshot_chunk_message_serde_roundtrip() {
        use crate::storage::Vote;

        // Header message roundtrip
        let header = SnapshotTransferHeader {
            vote: Vote::new(1, 1),
            meta: openraft::storage::SnapshotMeta {
                last_log_id: None,
                last_membership: openraft::StoredMembership::default(),
                snapshot_id: "chunked-test".to_string(),
            },
            data_size: 12345,
            since_ts: Some(42000),
        };
        let msg = SnapshotChunkMessage::Header(header);
        let bytes = rmp_serde::to_vec(&msg).expect("serialize header");
        let decoded: SnapshotChunkMessage =
            rmp_serde::from_slice(&bytes).expect("deserialize header");
        match decoded {
            SnapshotChunkMessage::Header(h) => {
                assert_eq!(h.data_size, 12345);
                assert_eq!(h.since_ts, Some(42000));
                assert_eq!(h.meta.snapshot_id, "chunked-test");
            }
            _ => panic!("expected Header variant"),
        }

        // DataChunk message roundtrip
        let chunk_data = vec![1u8, 2, 3, 4, 5];
        let chunk_msg = SnapshotChunkMessage::DataChunk(chunk_data.clone());
        let bytes2 = rmp_serde::to_vec(&chunk_msg).expect("serialize chunk");
        let decoded2: SnapshotChunkMessage =
            rmp_serde::from_slice(&bytes2).expect("deserialize chunk");
        match decoded2 {
            SnapshotChunkMessage::DataChunk(d) => assert_eq!(d, chunk_data),
            _ => panic!("expected DataChunk variant"),
        }
    }

    #[test]
    fn test_install_full_snapshot_from_reader() {
        // Build snapshot, install via reader, verify data matches
        let dir = tempdir().unwrap();
        let engine = open_engine(dir.path());

        engine.put(Partition::Node, b"node:1", b"alice").unwrap();
        engine
            .put(Partition::EdgeProp, b"ep:1", b"prop_data")
            .unwrap();

        let snapshot_data = build_full_snapshot(&engine).unwrap();

        // Install to fresh engine via reader
        let dir2 = tempdir().unwrap();
        let engine2 = open_engine(dir2.path());

        let mut cursor = std::io::Cursor::new(&snapshot_data);
        install_full_snapshot_from_reader(&engine2, &mut cursor).unwrap();

        assert_eq!(
            engine2
                .get(Partition::Node, b"node:1")
                .unwrap()
                .map(|b| b.to_vec()),
            Some(b"alice".to_vec())
        );
        assert_eq!(
            engine2
                .get(Partition::EdgeProp, b"ep:1")
                .unwrap()
                .map(|b| b.to_vec()),
            Some(b"prop_data".to_vec())
        );
    }

    #[test]
    fn test_install_full_snapshot_from_reader_cleans_stale() {
        // Pre-existing data not in snapshot gets cleaned up
        let dir = tempdir().unwrap();
        let engine = open_engine(dir.path());
        engine.put(Partition::Node, b"node:1", b"alice").unwrap();

        let snapshot_data = build_full_snapshot(&engine).unwrap();

        let dir2 = tempdir().unwrap();
        let engine2 = open_engine(dir2.path());
        engine2
            .put(Partition::Node, b"node:stale", b"old_data")
            .unwrap();

        let mut cursor = std::io::Cursor::new(&snapshot_data);
        install_full_snapshot_from_reader(&engine2, &mut cursor).unwrap();

        // Stale key removed
        assert!(engine2
            .get(Partition::Node, b"node:stale")
            .unwrap()
            .is_none());
        // Snapshot key present
        assert!(engine2.get(Partition::Node, b"node:1").unwrap().is_some());
    }

    #[test]
    fn test_install_incremental_snapshot_from_reader() {
        use coordinode_core::txn::timestamp::Timestamp;

        let dir = tempdir().unwrap();
        let config = StorageConfig::new(dir.path());
        let oracle = std::sync::Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new());
        let engine =
            StorageEngine::open_with_oracle(&config, oracle.clone()).expect("open oracle engine");

        // Write initial data
        engine.put(Partition::Node, b"node:1", b"v1").unwrap();
        engine
            .put(Partition::Adj, b"adj:KNOWS:out:1", b"adj1")
            .unwrap();

        let since = Timestamp::from_raw(oracle.next().as_raw());

        // Write changes after snapshot point
        engine.put(Partition::Node, b"node:1", b"v2").unwrap();
        engine.put(Partition::Node, b"node:2", b"new").unwrap();

        let incr_data = build_incremental_snapshot(&engine, since)
            .unwrap()
            .expect("should have changes");

        // Install to fresh engine via reader
        let dir2 = tempdir().unwrap();
        let engine2 = open_engine(dir2.path());
        engine2.put(Partition::Node, b"node:1", b"v1").unwrap();
        engine2
            .put(Partition::Adj, b"adj:KNOWS:out:1", b"adj1")
            .unwrap();

        let mut cursor = std::io::Cursor::new(&incr_data);
        install_incremental_snapshot_from_reader(&engine2, &mut cursor).unwrap();

        // node:1 updated to v2
        assert_eq!(
            engine2
                .get(Partition::Node, b"node:1")
                .unwrap()
                .map(|b| b.to_vec()),
            Some(b"v2".to_vec())
        );
        // node:2 added
        assert_eq!(
            engine2
                .get(Partition::Node, b"node:2")
                .unwrap()
                .map(|b| b.to_vec()),
            Some(b"new".to_vec())
        );
        // adj unchanged
        assert!(engine2
            .get(Partition::Adj, b"adj:KNOWS:out:1")
            .unwrap()
            .is_some());
    }

    #[test]
    fn test_install_full_from_reader_checksum_validation() {
        // Corrupt data should fail checksum
        let dir = tempdir().unwrap();
        let engine = open_engine(dir.path());
        engine.put(Partition::Node, b"node:1", b"alice").unwrap();

        let mut snapshot_data = build_full_snapshot(&engine).unwrap();

        // Corrupt the checksum bytes (last 8 bytes) to trigger mismatch.
        // Corrupting data bytes could break CNSN parsing before reaching
        // the checksum, so we target the checksum directly.
        let len = snapshot_data.len();
        snapshot_data[len - 1] ^= 0xFF;

        let dir2 = tempdir().unwrap();
        let engine2 = open_engine(dir2.path());

        let mut cursor = std::io::Cursor::new(&snapshot_data);
        let result = install_full_snapshot_from_reader(&engine2, &mut cursor);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("checksum mismatch"));
    }

    #[test]
    fn test_chunked_full_snapshot_roundtrip() {
        // End-to-end: build → chunk → reassemble → install via reader
        let dir = tempdir().unwrap();
        let engine = open_engine(dir.path());
        engine.put(Partition::Node, b"node:1", b"alice").unwrap();
        engine.put(Partition::Node, b"node:2", b"bob").unwrap();
        engine
            .put(Partition::EdgeProp, b"ep:1:2", b"friends")
            .unwrap();

        let snapshot_data = build_full_snapshot(&engine).unwrap();

        // Chunk with small size to test multi-chunk
        let small_chunk_size = 64;
        let chunks: Vec<&[u8]> = snapshot_data.chunks(small_chunk_size).collect();
        assert!(chunks.len() > 1, "should produce multiple chunks");

        // Reassemble
        let reassembled: Vec<u8> = chunks.iter().flat_map(|c| c.iter().copied()).collect();
        assert_eq!(reassembled, snapshot_data);

        // Install via reader from reassembled data
        let dir2 = tempdir().unwrap();
        let engine2 = open_engine(dir2.path());

        let mut cursor = std::io::Cursor::new(&reassembled);
        install_full_snapshot_from_reader(&engine2, &mut cursor).unwrap();

        assert_eq!(
            engine2
                .get(Partition::Node, b"node:1")
                .unwrap()
                .map(|b| b.to_vec()),
            Some(b"alice".to_vec())
        );
        assert_eq!(
            engine2
                .get(Partition::Node, b"node:2")
                .unwrap()
                .map(|b| b.to_vec()),
            Some(b"bob".to_vec())
        );
        assert_eq!(
            engine2
                .get(Partition::EdgeProp, b"ep:1:2")
                .unwrap()
                .map(|b| b.to_vec()),
            Some(b"friends".to_vec())
        );
    }

    #[test]
    fn test_reconstruct_cnsn_payload_matches_original() {
        // Verify that reconstruct_cnsn_payload produces the same bytes as
        // build_full_snapshot (minus the 8-byte checksum at end)
        let dir = tempdir().unwrap();
        let engine = open_engine(dir.path());
        engine.put(Partition::Node, b"node:1", b"test").unwrap();
        engine.put(Partition::Adj, b"adj:X:out:1", b"adj").unwrap();

        let snapshot_data = build_full_snapshot(&engine).unwrap();

        // Parse the snapshot to get partitions
        let payload_without_checksum = &snapshot_data[..snapshot_data.len() - 8];

        // Parse manually for reconstruct
        let partitions: Vec<Partition> = snapshot_partitions().collect();
        let partition_count = partitions.len();

        let mut cursor = std::io::Cursor::new(&snapshot_data);
        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic).unwrap();
        let mut version = [0u8; 1];
        cursor.read_exact(&mut version).unwrap();
        let mut pcount = [0u8; 1];
        cursor.read_exact(&mut pcount).unwrap();

        let mut parsed = Vec::new();
        for _ in 0..pcount[0] {
            let mut tag = [0u8; 1];
            cursor.read_exact(&mut tag).unwrap();
            let part = tag_to_partition(tag[0]).unwrap();
            let mut count_buf = [0u8; 4];
            cursor.read_exact(&mut count_buf).unwrap();
            let count = u32::from_be_bytes(count_buf) as usize;
            let mut entries = Vec::new();
            for _ in 0..count {
                let mut kl = [0u8; 4];
                cursor.read_exact(&mut kl).unwrap();
                let klen = u32::from_be_bytes(kl) as usize;
                let mut key = vec![0u8; klen];
                cursor.read_exact(&mut key).unwrap();
                let mut vl = [0u8; 4];
                cursor.read_exact(&mut vl).unwrap();
                let vlen = u32::from_be_bytes(vl) as usize;
                let mut val = vec![0u8; vlen];
                cursor.read_exact(&mut val).unwrap();
                entries.push((key, val));
            }
            parsed.push((part, entries));
        }

        let reconstructed = reconstruct_cnsn_payload(&parsed, partition_count);
        assert_eq!(reconstructed, payload_without_checksum);
    }
}
