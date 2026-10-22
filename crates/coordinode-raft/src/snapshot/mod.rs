//! Whole-store snapshots: the Raft snapshot a leader streams to a follower,
//! and the logical dump an operator takes and restores.
//!
//! A Raft snapshot is the store pinned at the applies' position and streamed
//! as its files (see [`coordinode_storage::engine::store_transfer`]); the
//! follower rebuilds the trees beside its store and installs them from there.
//!
//! ## Dump Format
//!
//! A dump captures all KV data across all storage partitions at a
//! consistent point in time. The binary format is:
//!
//! ```text
//! [magic: 4 bytes "CNSN"]
//! [version: 1 byte (currently 3; 1 and 2 are still read)]
//! [partition_count: 1 byte]
//! [partition_block]*
//! [table_section]          (version 2 and later)
//! [index_inventory]        (version 3 and later)
//! [checksum: 8 bytes (FNV-1a 64 of all preceding bytes, little-endian)]
//! ```
//!
//! The `index_inventory` is `[present: 1 byte][len: u32][bytes]`: what the
//! sender proved about each index copy it carries, opaque to this format
//! (see [`coordinode_storage::engine::index_copies`]). A snapshot without one
//! leaves every index copy unknown to the receiver, which checks them before
//! it uses them.
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

use std::io::{self, BufRead, Read as IoRead, Seek, SeekFrom, Write};

use serde::{Deserialize, Serialize};

use std::path::PathBuf;

use coordinode_storage::engine::batch::WriteBatch;
use coordinode_storage::engine::core::{ColumnarTable, StorageEngine};
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::store_transfer::ReceivedStore;

use crate::storage::{SnapshotMeta, Vote};

mod data;

pub(crate) use data::ChunkReader;
pub use data::SnapshotData;

/// The directory under `engine`'s data directory where snapshots received
/// from a leader are rebuilt before they install. Nothing in it outlives the
/// install; the state machine clears what a crash left when it opens.
pub fn snapshot_dir(engine: &StorageEngine) -> PathBuf {
    engine.data_dir().join("raft-snapshot")
}

/// A directory name under [`snapshot_dir`] no other received snapshot of
/// this process or an earlier one has.
pub(crate) fn receive_target(dir: &std::path::Path) -> PathBuf {
    static NEXT: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let seq = NEXT.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    dir.join(format!("{nanos:032x}-{seq:016x}.recv"))
}

/// Remove everything under `engine`'s [`snapshot_dir`]: a snapshot a crash
/// left half received, or a file an earlier build kept there.
///
/// # Errors
///
/// The directory exists and cannot be removed.
pub(crate) fn clear_snapshot_dir(engine: &StorageEngine) -> io::Result<()> {
    match std::fs::remove_dir_all(snapshot_dir(engine)) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

/// A list of KV entries for a single partition.
type PartitionEntries = Vec<(Vec<u8>, Vec<u8>)>;

/// Snapshot magic bytes: "CNSN" (CoordiNode SNapshot).
const MAGIC: &[u8; 4] = b"CNSN";

/// Snapshot format version written. Version 2 added the columnar table
/// section; version 1 files, as released builds wrote them, still install,
/// since a dump taken with a released build is how a store moves to this one.
const FORMAT_VERSION: u8 = 3;

/// The version that introduced the columnar table section.
const TABLES_SINCE: u8 = 2;

/// The version that introduced the index inventory section.
const INVENTORY_SINCE: u8 = 3;

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

/// Write a full snapshot of every user-data partition (all but
/// `Partition::Raft`, which openraft manages), every `STORAGE COLUMNAR`
/// table and `index_inventory`, the sender's inventory of the index copies
/// the snapshot carries (`None`: the receiver takes them all as unknown), to
/// `out`, from its current position, and return the bytes written.
///
/// Entries stream to `out` as they are scanned, so memory does not grow with
/// the store: each partition's entry count is written as a placeholder and
/// patched once the partition is done, and the checksum is computed by
/// reading the written bytes back.
///
/// # Errors
///
/// A storage scan or an I/O operation on `out` fails, or a field exceeds the
/// format's u32 lengths.
pub fn write_full_snapshot<F>(
    engine: &StorageEngine,
    index_inventory: Option<&[u8]>,
    out: &mut F,
) -> io::Result<u64>
where
    F: IoRead + Write + Seek,
{
    let start = out.stream_position()?;
    let partitions: Vec<Partition> = snapshot_partitions().collect();
    let partition_count = u8::try_from(partitions.len())
        .map_err(|_| io::Error::other("more partitions than the format's u8 count"))?;
    let mut counts: Vec<(u64, usize)> = Vec::with_capacity(partitions.len());

    let mut w = io::BufWriter::new(&mut *out);
    w.write_all(MAGIC)?;
    w.write_all(&[FORMAT_VERSION, partition_count])?;
    for &part in &partitions {
        w.write_all(&[partition_tag(part)])?;
        let count_at = w.stream_position()?;
        w.write_all(&[0; 4])?;
        let iter = engine
            .prefix_scan(part, &[])
            .map_err(|e| io::Error::other(format!("snapshot scan {}: {e}", part.name())))?;
        let mut entries = 0usize;
        for guard in iter {
            let (key, value) = guard
                .into_inner()
                .map_err(|e| io::Error::other(format!("snapshot iter {}: {e}", part.name())))?;
            // Schema `meta:*` keys are this node's own configuration (routing
            // computed against its endpoint set) and never replicate. `raft:*`
            // keys travel and receivers drop them, so the checksum covers
            // exactly what was built.
            if part == Partition::Schema && key.starts_with(b"meta:") {
                continue;
            }
            put_bytes(&mut w, &key)?;
            put_bytes(&mut w, &value)?;
            entries += 1;
        }
        tracing::debug!(
            partition = part.name(),
            entries,
            "snapshot: serialized partition"
        );
        counts.push((count_at, entries));
    }

    let tables = engine
        .columnar_tables_at(engine.snapshot())
        .map_err(|e| io::Error::other(format!("snapshot columnar tables: {e}")))?;
    put_tables(&mut w, &tables)?;
    match index_inventory {
        Some(inventory) => {
            w.write_all(&[1])?;
            put_bytes(&mut w, inventory)?;
        }
        None => w.write_all(&[0])?,
    }
    w.flush()?;
    drop(w);
    let end = out.stream_position()?;

    for (at, entries) in counts {
        out.seek(SeekFrom::Start(at))?;
        put_u32(out, entries)?;
    }

    out.seek(SeekFrom::Start(start))?;
    let body_len = end - start;
    let mut reader = io::BufReader::with_capacity(64 * 1024, (&mut *out).take(body_len));
    let mut hash = FNV_OFFSET;
    let mut hashed = 0u64;
    loop {
        let chunk = reader.fill_buf()?;
        if chunk.is_empty() {
            break;
        }
        hash = fnv1a_fold(hash, chunk);
        let n = chunk.len();
        hashed += n as u64;
        reader.consume(n);
    }
    drop(reader);
    if hashed != body_len {
        return Err(io::Error::from(io::ErrorKind::UnexpectedEof));
    }
    out.seek(SeekFrom::Start(end))?;
    out.write_all(&hash.to_le_bytes())?;
    out.flush()?;

    let total = body_len + 8;
    tracing::info!(
        total_bytes = total,
        partitions = partitions.len(),
        columnar_tables = tables.len(),
        "snapshot: build complete"
    );
    Ok(total)
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
/// the current engine but are absent in the snapshot. A crash during step 2
/// leaves some of them, but the caller records the snapshot's position only
/// after this returns: on restart the node still stands at its old position
/// and catches up either through another install (whose step 2 finishes the
/// cleanup) or through the log, whose deletes remove the same keys.
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
    /// The sender's inventory of the index copies; `None` when it sent
    /// none, which leaves every copy unknown.
    index_inventory: Option<Vec<u8>>,
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
    let index_inventory = if version >= INVENTORY_SINCE {
        let mut present = [0u8; 1];
        reader.read_exact(&mut present)?;
        match present[0] {
            0 => None,
            1 => Some(read_bytes(reader)?),
            other => {
                return Err(io::Error::other(format!(
                    "snapshot index inventory flag {other}"
                )));
            }
        }
    } else {
        None
    };
    Ok(ParsedSnapshot {
        partitions,
        tables,
        index_inventory,
    })
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
/// snapshot does not. A crash midway leaves stale keys, but the position is
/// recorded only after this returns, so the node catches up again from its
/// old position and the reinstall or the log's deletes remove them.
///
/// **Columnar tables** are then made exactly the snapshot's, when it lists
/// them.
/// Check the field dictionary a snapshot carries in its Schema rows before
/// any of it is installed: every id record has the name record that points
/// back at it and the reverse, so the data installed with it can be read.
fn verify_field_dictionary<'a>(
    schema: impl IntoIterator<Item = &'a (Vec<u8>, Vec<u8>)>,
) -> io::Result<()> {
    use coordinode_core::graph::intern::{
        FIELD_ID_KEY_PREFIX, FIELD_NAME_KEY_PREFIX, FieldInterner, decode_field_id,
        decode_field_id_key,
    };
    let invalid = |what: String| io::Error::other(format!("snapshot field dictionary: {what}"));
    let mut by_id = Vec::new();
    let mut by_name = std::collections::HashMap::new();
    for (key, value) in schema {
        if key.starts_with(FIELD_ID_KEY_PREFIX) {
            let id = decode_field_id_key(key)
                .ok_or_else(|| invalid("an id record key names no id".into()))?;
            let name = String::from_utf8(value.clone())
                .map_err(|_| invalid(format!("the name of id {id} is not UTF-8")))?;
            by_id.push((name, id));
        } else if let Some(name) = key.strip_prefix(FIELD_NAME_KEY_PREFIX) {
            let id = decode_field_id(value)
                .ok_or_else(|| invalid("a name record holds no id".into()))?;
            by_name.insert(name.to_vec(), id);
        }
    }
    if by_id.len() != by_name.len() {
        return Err(invalid(format!(
            "{} id records but {} name records",
            by_id.len(),
            by_name.len()
        )));
    }
    for (name, id) in &by_id {
        if by_name.get(name.as_bytes()) != Some(id) {
            return Err(invalid(format!(
                "id {id} and {name:?} do not point at each other"
            )));
        }
    }
    FieldInterner::from_bindings(by_id).map_err(|e| invalid(e.to_string()))?;
    Ok(())
}

fn apply_full(engine: &StorageEngine, parsed: ParsedSnapshot) -> io::Result<()> {
    verify_field_dictionary(
        parsed
            .partitions
            .iter()
            .filter(|(partition, _)| *partition == Partition::Schema)
            .flat_map(|(_, entries)| entries),
    )?;
    // The install replaces dictionary records in two steps; a dictionary read
    // between them would see the old and the new mixed.
    engine.with_metadata_exclusive(|| install_verified(engine, parsed))
}

fn install_verified(engine: &StorageEngine, parsed: ParsedSnapshot) -> io::Result<()> {
    let transfer = |e| io::Error::other(format!("snapshot index copies: {e}"));
    // The index copies the snapshot brings are qualified only by its
    // inventory, taken once they are in place.
    let replaces_index = parsed
        .partitions
        .iter()
        .any(|(partition, _)| *partition == Partition::Idx);
    if replaces_index {
        engine.begin_index_transfer().map_err(transfer)?;
    }
    let mut batch = WriteBatch::new(engine);
    let mut total_written = 0usize;
    // Values move into the batch; only the keys are kept, for the cleanup.
    let mut snapshot_keys = Vec::with_capacity(parsed.partitions.len());
    for (partition, entries) in parsed.partitions {
        let mut keys = std::collections::HashSet::with_capacity(entries.len());
        for (key, value) in entries {
            keys.insert(key.clone());
            batch.put(partition, key, value);
            total_written += 1;
        }
        snapshot_keys.push((partition, keys));
    }
    batch
        .commit()
        .map_err(|e| io::Error::other(format!("snapshot phase 1 (atomic write) failed: {e}")))?;

    let mut stale_deleted = 0usize;
    for (partition, snap_keys) in &snapshot_keys {
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
    if replaces_index {
        engine
            .index_transfer_installed(parsed.index_inventory.as_deref())
            .map_err(transfer)?;
    }
    // Views of the dictionary taken before the install reread it.
    engine.note_field_dictionary_change();
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

/// Write `entries` as `[count: u32][kv_entry]*`.
fn put_entries(w: &mut impl Write, entries: &[(Vec<u8>, Vec<u8>)]) -> io::Result<()> {
    put_u32(w, entries.len())?;
    for (key, value) in entries {
        put_bytes(w, key)?;
        put_bytes(w, value)?;
    }
    Ok(())
}

/// Write the columnar table section.
fn put_tables(w: &mut impl Write, tables: &[ColumnarTable]) -> io::Result<()> {
    put_u32(w, tables.len())?;
    for (name, rows) in tables {
        put_bytes(w, name.as_bytes())?;
        put_entries(w, rows)?;
    }
    Ok(())
}

fn put_bytes(w: &mut impl Write, bytes: &[u8]) -> io::Result<()> {
    put_u32(w, bytes.len())?;
    w.write_all(bytes)
}

/// A length or count as the format's u32; anything larger is refused rather
/// than written truncated.
fn put_u32(w: &mut impl Write, n: usize) -> io::Result<()> {
    let n = u32::try_from(n)
        .map_err(|_| io::Error::other(format!("snapshot field of {n} exceeds the u32 format")))?;
    w.write_all(&n.to_be_bytes())
}

// ── Chunked Snapshot Transfer Protocol ──────────────────────────────
//
// A snapshot travels as a header message (vote, meta, data size) followed
// by chunks of the store transfer stream of up to SNAPSHOT_CHUNK_SIZE. The
// sender reads them from the pinned files as the stream is consumed, and the
// receiver rebuilds the trees from them in its snapshot directory as they
// arrive: neither side holds the snapshot in memory or copies it whole.

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
    /// Total size of the store transfer stream that follows in chunks.
    pub data_size: u64,
}

/// A single message in the chunked snapshot transfer stream.
///
/// The first message is always a Header. Subsequent messages are DataChunks
/// carrying the store transfer stream. The receiver consumes data chunks
/// until `data_size` bytes have been received.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SnapshotChunkMessage {
    /// First message: metadata about the snapshot transfer.
    Header(SnapshotTransferHeader),
    /// Subsequent messages: stream bytes (up to SNAPSHOT_CHUNK_SIZE), one
    /// binary blob on the wire rather than an array of integers.
    DataChunk(#[serde(with = "serde_bytes")] Vec<u8>),
}

/// Install a full snapshot from a reader (file or buffer).
///
/// Like `install_full_snapshot`, but the serialized snapshot is read as it
/// is parsed instead of being held whole. The checksum is verified after the
/// parse and before any write, and the reader must end right after it.
pub fn install_full_snapshot_from_reader(
    engine: &StorageEngine,
    reader: &mut impl IoRead,
) -> io::Result<()> {
    let parsed = parse_verified_stream(reader)?;
    if reader.read(&mut [0u8; 1])? != 0 {
        return Err(io::Error::other("snapshot has bytes past its checksum"));
    }
    apply_full(engine, parsed)
}

/// Install `received`, a Raft snapshot standing at log position `next`
/// (`payload` is the log id of entry `next - 1`), as the store's complete
/// data generation: every replicated partition is replaced, not written
/// over, one at a time, and every tree is bound to `next`, or to its own
/// position for a partition the snapshot holds further along. The entries
/// after it then apply at their own commit timestamps over the installed
/// rows, which a write over the old contents at this node's next seqno would
/// hide. This node's own records (its Raft state, its routing) stay.
///
/// # Errors
///
/// The received store cannot be opened or read, its field dictionary fails
/// its check, or the install fails.
pub fn install_received(
    engine: &StorageEngine,
    received: &ReceivedStore,
    next: u64,
    payload: &[u8],
) -> io::Result<()> {
    let storage = |what: &str, e: coordinode_storage::error::StorageError| {
        io::Error::other(format!("{what}: {e}"))
    };
    let source = StorageEngine::open_checkpoint(received.dir())
        .map_err(|e| storage("open the received snapshot", e))?;
    let schema = source
        .replicated_rows(Partition::Schema)
        .map_err(|e| storage("read the snapshot's schema", e))?;
    verify_field_dictionary(&schema)?;
    drop(schema);
    let partitions: Vec<Partition> = snapshot_partitions().collect();
    engine.with_metadata_exclusive(|| {
        let mut total_written = 0usize;
        engine
            .install_raft_image(
                &partitions,
                |partition| {
                    let rows = source.replicated_rows(partition)?;
                    total_written += rows.len();
                    Ok(rows)
                },
                next,
                payload,
                received.positions(),
                received.index_inventory(),
            )
            .map_err(|e| storage("install the snapshot's partitions", e))?;
        let tables = source
            .columnar_tables_at(source.snapshot())
            .map_err(|e| storage("read the snapshot's columnar tables", e))?;
        let tables = install_tables(engine, Some(tables))?;
        engine.note_field_dictionary_change();
        tracing::info!(total_written, tables, next, "snapshot installed");
        Ok(())
    })
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
