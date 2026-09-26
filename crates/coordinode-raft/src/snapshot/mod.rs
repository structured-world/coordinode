//! Snapshot serialization and streaming for Raft log compaction.
//!
//! ## Snapshot Format
//!
//! A snapshot captures all KV data across all storage partitions at a
//! consistent point in time. The binary format is:
//!
//! ```text
//! [magic: 4 bytes "CNSN"]
//! [version: 1 byte (currently 2; 1 is still read)]
//! [partition_count: 1 byte]
//! [partition_block]*
//! [table_section]          (version 2 and later)
//! [checksum: 8 bytes (FNV-1a 64 of all preceding bytes, little-endian)]
//! ```
//!
//! The `table_section` carries every `STORAGE COLUMNAR` table, whole:
//! ```text
//! [table_count: 4 bytes (u32 big-endian)]
//! ([name_len: u32][name: UTF-8][entry_count: u32][kv_entry]*)*
//! ```
//! A version 1 snapshot has no table section and leaves the receiver's
//! tables as they are.
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

use coordinode_storage::Guard;
use coordinode_storage::engine::batch::WriteBatch;
use coordinode_storage::engine::core::{ColumnarTable, StorageEngine};
use coordinode_storage::engine::partition::Partition;

use crate::storage::{SnapshotMeta, Vote};

/// Transfer message for sending a snapshot from leader to follower over gRPC.
///
/// Contains the leader's vote (for validation), snapshot metadata, and the
/// whole-store binary snapshot data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotTransfer {
    /// Leader's current vote (follower validates leadership).
    pub vote: Vote,
    /// Snapshot metadata: last_log_id and membership.
    pub meta: SnapshotMeta,
    /// Snapshot data in the binary format above.
    pub data: Vec<u8>,
}

/// A list of KV entries for a single partition.
type PartitionEntries = Vec<(Vec<u8>, Vec<u8>)>;

/// Snapshot magic bytes: "CNSN" (CoordiNode SNapshot).
const MAGIC: &[u8; 4] = b"CNSN";

/// Snapshot format version written. Version 2 added the columnar table
/// section; version 1 files, as released builds wrote them, still install,
/// since a dump taken with a released build is how a store moves to this one.
const FORMAT_VERSION: u8 = 2;

/// The version that introduced the columnar table section.
const TABLES_SINCE: u8 = 2;

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
/// managed by openraft) and every `STORAGE COLUMNAR` table, serializes every
/// KV pair, and returns the complete snapshot as bytes, checksummed.
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
        put_entries(&mut buf, &entries)?;

        tracing::debug!(
            partition = part.name(),
            entries = entries.len(),
            "snapshot: serialized partition"
        );
    }

    let tables = engine
        .columnar_tables_at(engine.snapshot())
        .map_err(|e| io::Error::other(format!("snapshot columnar tables: {e}")))?;
    put_tables(&mut buf, &tables)?;

    let hash = fnv1a_64(&buf);
    buf.extend_from_slice(&hash.to_le_bytes());

    tracing::info!(
        total_bytes = buf.len(),
        partitions = partitions.len(),
        columnar_tables = tables.len(),
        "snapshot: build complete"
    );

    Ok(buf)
}

/// Install a full snapshot: deserialize and write all KV pairs to CoordiNode storage.
///
/// Uses a **two-step crash-safe** approach:
///
/// **Step 1 (atomic via WriteBatch):** Write ALL snapshot entries in a
/// single atomic batch. If crash occurs during step 1, no writes are
/// visible — old data remains intact. WriteBatch uses the storage
/// write transaction internally (all-or-nothing commit).
///
/// **Step 2 (idempotent cleanup):** Delete stale keys that exist in
/// the current engine but are absent in the snapshot. If crash occurs
/// during step 2, stale keys remain (harmless — cleaned up on next
/// snapshot install).
///
/// **Important:** Raft keys (`raft:*`) in the Schema partition are
/// always preserved — they're managed by openraft, not application data.
pub fn install_full_snapshot(engine: &StorageEngine, data: &[u8]) -> io::Result<()> {
    apply_full(engine, parse_verified_slice(data)?)
}

/// A parsed snapshot, checksum already verified, with the entries a
/// receiver must never overwrite (the Raft partition, Schema `raft:` and
/// `meta:` keys) left out.
struct ParsedSnapshot {
    partitions: Vec<(Partition, PartitionEntries)>,
    /// `None` for a version 1 snapshot, which says nothing about tables.
    tables: Option<Vec<ColumnarTable>>,
}

/// Whether a receiver installs `key` of `partition` from a snapshot. Raft
/// state is openraft's and routing (`meta:`) is per node.
fn installable(partition: Partition, key: &[u8]) -> bool {
    partition != Partition::Raft
        && !(partition == Partition::Schema
            && (key.starts_with(b"raft:") || key.starts_with(b"meta:")))
}

/// Verify `data`'s trailing checksum, then parse the body. The whole buffer
/// is in hand, so a corrupt header is reported as the checksum failure it is
/// rather than as whatever the parser tripped on first.
fn parse_verified_slice(data: &[u8]) -> io::Result<ParsedSnapshot> {
    // Something that is not a snapshot at all says so before any checksum.
    if !data.starts_with(MAGIC) {
        return Err(io::Error::other("invalid snapshot magic"));
    }
    let (payload, checksum) = data
        .split_last_chunk::<8>()
        .ok_or_else(|| io::Error::other("snapshot too small for checksum"))?;
    let expected = u64::from_le_bytes(*checksum);
    let actual = fnv1a_64(payload);
    if expected != actual {
        return Err(io::Error::other(format!(
            "snapshot checksum mismatch: expected {expected:#x}, got {actual:#x}"
        )));
    }
    let mut cursor = io::Cursor::new(payload);
    let parsed = parse_body(&mut cursor)?;
    if cursor.position() != payload.len() as u64 {
        return Err(io::Error::other("snapshot has bytes past its last section"));
    }
    Ok(parsed)
}

/// Parse a snapshot from a stream, hashing it as it goes, and verify the
/// checksum that follows before returning: nothing is applied from a stream
/// whose checksum fails.
fn parse_verified_stream(reader: &mut impl IoRead) -> io::Result<ParsedSnapshot> {
    let mut hashing = HashingReader {
        inner: &mut *reader,
        hash: FNV_OFFSET,
    };
    let parsed = parse_body(&mut hashing)?;
    let actual = hashing.hash;
    let mut checksum = [0u8; 8];
    reader.read_exact(&mut checksum)?;
    let expected = u64::from_le_bytes(checksum);
    if expected != actual {
        return Err(io::Error::other(format!(
            "snapshot checksum mismatch: expected {expected:#x}, got {actual:#x}"
        )));
    }
    Ok(parsed)
}

fn parse_body(reader: &mut impl IoRead) -> io::Result<ParsedSnapshot> {
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::other("invalid snapshot magic"));
    }
    let mut version = [0u8; 1];
    reader.read_exact(&mut version)?;
    let version = version[0];
    if version == 0 || version > FORMAT_VERSION {
        return Err(io::Error::other(format!(
            "unsupported snapshot version: {version}"
        )));
    }
    let mut part_count = [0u8; 1];
    reader.read_exact(&mut part_count)?;

    let mut partitions = Vec::with_capacity(usize::from(part_count[0]));
    for _ in 0..part_count[0] {
        let mut tag = [0u8; 1];
        reader.read_exact(&mut tag)?;
        let partition = tag_to_partition(tag[0])
            .ok_or_else(|| io::Error::other(format!("unknown partition tag: {}", tag[0])))?;
        let mut entries = read_entries(reader)?;
        entries.retain(|(key, _)| installable(partition, key));
        if partition != Partition::Raft {
            partitions.push((partition, entries));
        }
    }

    let tables = if version >= TABLES_SINCE {
        let count = read_u32(reader)?;
        let mut tables = Vec::new();
        for _ in 0..count {
            let name = String::from_utf8(read_bytes(reader)?)
                .map_err(|_| io::Error::other("snapshot table name is not UTF-8"))?;
            tables.push((name, read_entries(reader)?));
        }
        Some(tables)
    } else {
        None
    };
    Ok(ParsedSnapshot { partitions, tables })
}

fn read_u32(reader: &mut impl IoRead) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

/// A length-prefixed byte string. The length comes from the stream, so the
/// buffer grows with the bytes actually read instead of trusting it up front:
/// a corrupt length ends in an EOF error, not in a huge allocation.
fn read_bytes(reader: &mut impl IoRead) -> io::Result<Vec<u8>> {
    let len = read_u32(reader)?;
    let mut bytes = Vec::new();
    reader.take(u64::from(len)).read_to_end(&mut bytes)?;
    if bytes.len() as u64 != u64::from(len) {
        return Err(io::Error::from(io::ErrorKind::UnexpectedEof));
    }
    Ok(bytes)
}

fn read_entries(reader: &mut impl IoRead) -> io::Result<PartitionEntries> {
    let count = read_u32(reader)?;
    let mut entries = Vec::new();
    for _ in 0..count {
        let key = read_bytes(reader)?;
        let value = read_bytes(reader)?;
        entries.push((key, value));
    }
    Ok(entries)
}

/// Replace the receiver's state with a full snapshot.
///
/// **Step 1 (atomic via WriteBatch):** every snapshot entry in one batch;
/// a crash before it commits leaves the old data intact.
///
/// **Step 2 (idempotent cleanup):** delete keys the engine holds and the
/// snapshot does not; a crash midway leaves stale keys that the next install
/// removes.
///
/// **Columnar tables** are then made exactly the snapshot's, when it lists
/// them.
fn apply_full(engine: &StorageEngine, parsed: ParsedSnapshot) -> io::Result<()> {
    let mut batch = WriteBatch::new(engine);
    let mut total_written = 0usize;
    for (partition, entries) in &parsed.partitions {
        for (key, value) in entries {
            batch.put(*partition, key.clone(), value.clone());
            total_written += 1;
        }
    }
    batch
        .commit()
        .map_err(|e| io::Error::other(format!("snapshot phase 1 (atomic write) failed: {e}")))?;

    let mut stale_deleted = 0usize;
    for (partition, snap_entries) in &parsed.partitions {
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
            if installable(*partition, &key) && !snap_keys.contains(key.as_ref()) {
                keys_to_delete.push(key.to_vec());
            }
        }
        for key in &keys_to_delete {
            engine.delete(*partition, key).map_err(|e| {
                io::Error::other(format!("cleanup delete {}: {e}", partition.name()))
            })?;
        }
        stale_deleted += keys_to_delete.len();
    }

    let tables = install_tables(engine, parsed.tables)?;
    tracing::info!(
        total_written,
        stale_deleted,
        tables,
        "snapshot install complete (two-phase)"
    );
    Ok(())
}

/// Make the receiver's columnar tables the snapshot's; a version 1 snapshot
/// leaves them alone. Returns how many tables were installed.
fn install_tables(engine: &StorageEngine, tables: Option<Vec<ColumnarTable>>) -> io::Result<usize> {
    let Some(tables) = tables else {
        return Ok(0);
    };
    let count = tables.len();
    engine
        .replace_columnar_tables(tables)
        .map_err(|e| io::Error::other(format!("snapshot columnar tables: {e}")))?;
    Ok(count)
}

/// Append `entries` as `[count: u32][kv_entry]*`.
fn put_entries(buf: &mut Vec<u8>, entries: &[(Vec<u8>, Vec<u8>)]) -> io::Result<()> {
    put_u32(buf, entries.len())?;
    for (key, value) in entries {
        put_bytes(buf, key)?;
        put_bytes(buf, value)?;
    }
    Ok(())
}

/// Append the columnar table section.
fn put_tables(buf: &mut Vec<u8>, tables: &[ColumnarTable]) -> io::Result<()> {
    put_u32(buf, tables.len())?;
    for (name, rows) in tables {
        put_bytes(buf, name.as_bytes())?;
        put_entries(buf, rows)?;
    }
    Ok(())
}

fn put_bytes(buf: &mut Vec<u8>, bytes: &[u8]) -> io::Result<()> {
    put_u32(buf, bytes.len())?;
    buf.extend_from_slice(bytes);
    Ok(())
}

/// A length or count as the format's u32; anything larger is refused rather
/// than written truncated.
fn put_u32(buf: &mut Vec<u8>, n: usize) -> io::Result<()> {
    let n = u32::try_from(n)
        .map_err(|_| io::Error::other(format!("snapshot field of {n} exceeds the u32 format")))?;
    buf.extend_from_slice(&n.to_be_bytes());
    Ok(())
}

// ── Chunked Snapshot Transfer Protocol ──────────────────────────────
//
// For large snapshots (>1GB), sending the entire CNSN blob in a single
// gRPC message causes OOM on both sender and receiver. The chunked
// protocol splits the transfer into:
//
//   Message 1: SnapshotTransferHeader (vote, meta, data_size)
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
    /// Snapshot metadata: last_log_id and membership.
    pub meta: SnapshotMeta,
    /// Total size of the CNSN data that follows in subsequent chunks.
    pub data_size: u64,
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
/// of `&[u8]`, avoiding the need to hold the serialized snapshot in memory.
/// The checksum is verified after the parse and before any write.
pub fn install_full_snapshot_from_reader(
    engine: &StorageEngine,
    reader: &mut impl IoRead,
) -> io::Result<()> {
    apply_full(engine, parse_verified_stream(reader)?)
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// The snapshot checksum: FNV-1a 64. Fixed by the format, which released
/// builds already wrote; a stronger hash would be a new format version.
fn fnv1a_64(data: &[u8]) -> u64 {
    fnv1a_fold(FNV_OFFSET, data)
}

fn fnv1a_fold(mut hash: u64, data: &[u8]) -> u64 {
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Hands `inner`'s bytes through, folding each into the snapshot checksum.
struct HashingReader<'a, R> {
    inner: &'a mut R,
    hash: u64,
}

impl<R: IoRead> IoRead for HashingReader<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.hash = fnv1a_fold(self.hash, &buf[..n]);
        Ok(n)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
