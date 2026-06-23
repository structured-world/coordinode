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
        Partition::VectorF32 => 9,
        Partition::Registry => 10,
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
        9 => Some(Partition::VectorF32),
        10 => Some(Partition::Registry),
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
            // Skip Schema `meta:*` keys — engine-internal per-node
            // configuration (the LSM-level routing computed against this
            // node's endpoint set, never replicated). `raft:*` keys are
            // included here intentionally: existing apply paths filter
            // them on the receiver side, and including them at build time
            // preserves the build↔apply hash-checksum invariant.
            if part == Partition::Schema && key.starts_with(b"meta:") {
                continue;
            }
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

/// Build an incremental snapshot containing only KV pairs modified after `since_ts`.
///
/// Uses native seqno MVCC (ADR-016): the lsm-tree `scan_since_seqno` surfaces
/// only the keys whose version history advanced past `since_ts` (O(delta), not
/// a 2× full-partition scan), and each changed key's merged current value is
/// re-read. Entries that are new, changed, or deleted since `since_ts` are
/// included (a key now absent → tombstone). The GC watermark is pinned at
/// `since_ts` for the build so concurrent compaction cannot drop needed
/// history.
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

    // Pin the GC watermark at `old_seqno` for the duration of the build so a
    // concurrent compaction cannot collect version history the `scan_since`
    // pass still needs. Released when `_pin` drops at function end.
    let _pin = engine.pin_snapshot_at(old_seqno);

    for part in partitions {
        let tag = partition_tag(part);
        let mut changed: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();

        if part == Partition::Schema {
            // Schema partition: always include ALL keys regardless of since_ts.
            // Dgraph pattern: schema/type keys are always sent for consistency.
            // Skip `meta:*` keys (engine-internal per-node routing
            // configuration, never replicated). `raft:*` keys stay in to
            // preserve hash invariant — apply paths filter them on receive.
            let iter = engine
                .prefix_scan(part, &[])
                .map_err(|e| io::Error::other(format!("incr scan {}: {e}", part.name())))?;
            for guard in iter {
                let (raw_key, raw_value) = guard
                    .into_inner()
                    .map_err(|e| io::Error::other(format!("incr iter {}: {e}", part.name())))?;
                if raw_key.starts_with(b"meta:") {
                    continue;
                }
                changed.push((raw_key.to_vec(), raw_value.to_vec()));
            }
        } else {
            // O(delta): the lsm-tree surfaces only the keys whose version
            // history advanced past `old_seqno` (vs. the former 2× full-scan +
            // merge-diff). Re-read each changed key's merged current value —
            // this resolves accumulated merge operands for the adj / counter
            // partitions, so the wire format stays state-based (Put resolved
            // value / tombstone), and the receiver's merge-write apply is
            // unchanged. A key now absent is a tombstone (empty value),
            // matching the deletion case of the old diff.
            let changed_keys = engine
                .changed_keys_since(part, old_seqno)
                .map_err(|e| io::Error::other(format!("incr scan_since {}: {e}", part.name())))?;
            for key in changed_keys {
                match engine
                    .get(part, &key)
                    .map_err(|e| io::Error::other(format!("incr get {}: {e}", part.name())))?
                {
                    Some(value) => changed.push((key, value.to_vec())),
                    None => changed.push((key, Vec::new())),
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
            if partition == Partition::Schema
                && (key.starts_with(b"raft:") || key.starts_with(b"meta:"))
            {
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
            if partition == Partition::Schema
                && (key.starts_with(b"raft:") || key.starts_with(b"meta:"))
            {
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
            if *partition == Partition::Schema
                && (key.as_ref().starts_with(b"raft:") || key.as_ref().starts_with(b"meta:"))
            {
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

/// Maximum size of a single snapshot data chunk in bytes (2 MB).
///
/// Kept below tonic's default 4MB max message size to leave room for
/// the msgpack envelope overhead of `SnapshotChunkMessage::DataChunk`.
pub const SNAPSHOT_CHUNK_SIZE: usize = 2 * 1024 * 1024;

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
            if partition == Partition::Schema
                && (key.starts_with(b"raft:") || key.starts_with(b"meta:"))
            {
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

            if *partition == Partition::Schema
                && (key.as_ref().starts_with(b"raft:") || key.as_ref().starts_with(b"meta:"))
            {
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
            if *partition == Partition::Schema
                && (key.starts_with(b"raft:") || key.starts_with(b"meta:"))
            {
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
mod tests;
