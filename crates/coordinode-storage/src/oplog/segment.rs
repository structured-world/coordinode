//! Segment file: writer and reader for oplog segments.
//!
//! ## On-disk layout
//!
//! ```text
//! ┌──────────────────────────────────────────┐
//! │  Header  18B                             │
//! │    MAGIC        4B  = b"OPLO"            │
//! │    version      2B  LE                   │
//! │    shard_id     4B  LE                   │
//! │    first_index  8B  LE                   │
//! ├──────────────────────────────────────────┤
//! │  Entry  ×N                               │
//! │    varint(payload_len)  1-10B            │
//! │    msgpack_bytes        payload_len B    │
//! │    crc32_le             4B               │
//! ├──────────────────────────────────────────┤
//! │  Footer  32B                             │
//! │    entry_count   4B  LE                  │
//! │    first_ts      8B  LE                  │
//! │    last_ts       8B  LE                  │
//! │    total_bytes   8B  LE                  │
//! │    crc32_le      4B  (covers first 28B)  │
//! └──────────────────────────────────────────┘
//! ```

use std::io::{self, BufWriter, Cursor, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::error::{StorageError, StorageResult};
use crate::oplog::entry::OplogEntry;

/// Oplog segment magic bytes.
pub const MAGIC: [u8; 4] = *b"OPLO";
/// Segment format version.
pub const FORMAT_VERSION: u16 = 1;
/// Header size in bytes.
pub const HEADER_SIZE: u64 = 18;
/// Footer size in bytes.
pub const FOOTER_SIZE: u64 = 32;

#[inline]
fn io_err(e: impl std::fmt::Display) -> StorageError {
    StorageError::Io(e.to_string())
}

// ── Varint (unsigned LEB128) ─────────────────────────────────────────────────

/// Encode `value` as unsigned LEB128 into `out`.
pub(crate) fn encode_varint(mut value: u64, out: &mut Vec<u8>) {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            break;
        }
        out.push(byte | 0x80);
    }
}

/// Decode an unsigned LEB128 varint from `reader`.
pub(crate) fn decode_varint<R: Read>(reader: &mut R) -> io::Result<u64> {
    let mut result: u64 = 0;
    let mut shift = 0u32;
    loop {
        let mut buf = [0u8; 1];
        reader.read_exact(&mut buf)?;
        let byte = buf[0];
        // Guard against overflow before shifting
        if shift < 64 {
            result |= ((byte & 0x7F) as u64) << shift;
        }
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift >= 70 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "varint overflow: more than 10 continuation bytes",
            ));
        }
    }
    Ok(result)
}

// ── Header ───────────────────────────────────────────────────────────────────

/// Decoded segment header.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentHeader {
    pub version: u16,
    pub shard_id: u32,
    pub first_index: u64,
}

fn write_header<W: Write>(w: &mut W, shard_id: u32, first_index: u64) -> io::Result<()> {
    w.write_all(&MAGIC)?;
    w.write_all(&FORMAT_VERSION.to_le_bytes())?;
    w.write_all(&shard_id.to_le_bytes())?;
    w.write_all(&first_index.to_le_bytes())?;
    Ok(())
}

fn read_header<R: Read>(r: &mut R) -> StorageResult<SegmentHeader> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).map_err(io_err)?;
    if magic != MAGIC {
        return Err(StorageError::Io(format!(
            "invalid oplog magic: expected {:?}, got {:?}",
            &MAGIC, &magic
        )));
    }

    let mut ver_buf = [0u8; 2];
    r.read_exact(&mut ver_buf).map_err(io_err)?;
    let version = u16::from_le_bytes(ver_buf);

    let mut shard_buf = [0u8; 4];
    r.read_exact(&mut shard_buf).map_err(io_err)?;
    let shard_id = u32::from_le_bytes(shard_buf);

    let mut idx_buf = [0u8; 8];
    r.read_exact(&mut idx_buf).map_err(io_err)?;
    let first_index = u64::from_le_bytes(idx_buf);

    Ok(SegmentHeader {
        version,
        shard_id,
        first_index,
    })
}

// ── Footer ───────────────────────────────────────────────────────────────────

/// Decoded segment footer.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentFooter {
    pub entry_count: u32,
    pub first_ts: u64,
    pub last_ts: u64,
    /// Total bytes in the entries section (sum of all serialized entry frames).
    pub total_bytes: u64,
}

fn write_footer<W: Write>(w: &mut W, footer: &SegmentFooter) -> io::Result<()> {
    let mut buf = [0u8; 28];
    buf[0..4].copy_from_slice(&footer.entry_count.to_le_bytes());
    buf[4..12].copy_from_slice(&footer.first_ts.to_le_bytes());
    buf[12..20].copy_from_slice(&footer.last_ts.to_le_bytes());
    buf[20..28].copy_from_slice(&footer.total_bytes.to_le_bytes());

    let checksum = crc32fast::hash(&buf);
    w.write_all(&buf)?;
    w.write_all(&checksum.to_le_bytes())?;
    Ok(())
}

fn read_footer<R: Read>(r: &mut R) -> StorageResult<SegmentFooter> {
    let mut buf = [0u8; 32];
    r.read_exact(&mut buf).map_err(io_err)?;

    let expected = crc32fast::hash(&buf[..28]);
    let actual = u32::from_le_bytes([buf[28], buf[29], buf[30], buf[31]]);
    if expected != actual {
        return Err(StorageError::ChecksumMismatch {
            expected,
            actual,
            context: "segment footer".to_string(),
        });
    }

    let entry_count = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let first_ts = u64::from_le_bytes([
        buf[4], buf[5], buf[6], buf[7], buf[8], buf[9], buf[10], buf[11],
    ]);
    let last_ts = u64::from_le_bytes([
        buf[12], buf[13], buf[14], buf[15], buf[16], buf[17], buf[18], buf[19],
    ]);
    let total_bytes = u64::from_le_bytes([
        buf[20], buf[21], buf[22], buf[23], buf[24], buf[25], buf[26], buf[27],
    ]);

    Ok(SegmentFooter {
        entry_count,
        first_ts,
        last_ts,
        total_bytes,
    })
}

// ── SegmentWriter ─────────────────────────────────────────────────────────────

/// Writes entries to a new segment file.
///
/// Call [`SegmentWriter::create`] to open a new segment, [`append`](Self::append)
/// to write entries, and [`seal`](Self::seal) to write the footer and flush.
pub struct SegmentWriter {
    path: PathBuf,
    file: BufWriter<std::fs::File>,
    entry_count: u32,
    first_ts: u64,
    last_ts: u64,
    /// Total bytes written to the entries section.
    total_bytes: u64,
}

impl SegmentWriter {
    /// Create a new segment file at `path` and write the 18-byte header.
    ///
    /// The file must not already exist (`create_new` semantics).
    pub fn create(path: &Path, shard_id: u32, first_index: u64) -> StorageResult<Self> {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|e| StorageError::Io(format!("create segment {:?}: {e}", path)))?;

        let mut w = BufWriter::new(file);
        write_header(&mut w, shard_id, first_index)
            .map_err(|e| StorageError::Io(format!("write header: {e}")))?;

        Ok(Self {
            path: path.to_path_buf(),
            file: w,
            entry_count: 0,
            first_ts: 0,
            last_ts: 0,
            total_bytes: 0,
        })
    }

    /// Append one [`OplogEntry`] to the segment.
    ///
    /// Frame layout: `varint(payload_len) || msgpack_bytes || crc32_le(4B)`.
    pub fn append(&mut self, entry: &OplogEntry) -> StorageResult<()> {
        let payload = entry
            .encode()
            .map_err(|e| StorageError::Serialization(e.to_string()))?;

        let checksum = crc32fast::hash(&payload);

        let mut varint_buf = Vec::with_capacity(10);
        encode_varint(payload.len() as u64, &mut varint_buf);

        self.file.write_all(&varint_buf).map_err(io_err)?;
        self.file.write_all(&payload).map_err(io_err)?;
        self.file
            .write_all(&checksum.to_le_bytes())
            .map_err(io_err)?;

        let frame_bytes = (varint_buf.len() + payload.len() + 4) as u64;
        self.total_bytes += frame_bytes;
        self.entry_count += 1;

        if self.entry_count == 1 {
            self.first_ts = entry.ts;
        }
        self.last_ts = entry.ts;

        Ok(())
    }

    /// Number of entries written so far.
    pub fn entry_count(&self) -> u32 {
        self.entry_count
    }

    /// Total bytes written to the entries section.
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Flush user-space buffer and fsync the active segment to storage.
    ///
    /// This is the "ONE fsync" in the write path:
    ///   `append(entry)` → `flush_and_sync()` → `io_completed(Ok(()))`
    ///
    /// After this call returns without error, the written entries are
    /// durable — a crash will not lose them. The footer is NOT written
    /// here; it is written during [`seal`](Self::seal) at rotation time.
    pub fn flush_and_sync(&mut self) -> StorageResult<()> {
        // Step 1: flush BufWriter's user-space buffer to the kernel buffer.
        self.file
            .flush()
            .map_err(|e| StorageError::Io(format!("flush segment: {e}")))?;
        // Step 2: fsync the kernel buffer to the storage device.
        // sync_data() skips metadata update (atime, etc.) — faster than sync_all()
        // and sufficient for crash safety of written data.
        self.file
            .get_ref()
            .sync_data()
            .map_err(|e| StorageError::Io(format!("sync_data segment: {e}")))?;
        Ok(())
    }

    /// Seal the segment: write the 32-byte footer, flush, and fsync.
    ///
    /// Returns the path of the sealed file.
    pub fn seal(mut self) -> StorageResult<PathBuf> {
        let footer = SegmentFooter {
            entry_count: self.entry_count,
            first_ts: self.first_ts,
            last_ts: self.last_ts,
            total_bytes: self.total_bytes,
        };
        write_footer(&mut self.file, &footer)
            .map_err(|e| StorageError::Io(format!("write footer: {e}")))?;
        self.file
            .flush()
            .map_err(|e| StorageError::Io(format!("flush segment: {e}")))?;
        // Also fsync on seal — the footer must be durable before we remove
        // the active-writer reference and add the path to sealed[].
        self.file
            .get_ref()
            .sync_data()
            .map_err(|e| StorageError::Io(format!("sync_data on seal: {e}")))?;
        Ok(self.path)
    }
}

// ── SegmentReader ─────────────────────────────────────────────────────────────

/// Reads and validates a sealed segment file.
///
/// On [`open`](Self::open), the entire file is read into memory, all entry
/// crc32 checksums are verified, and the footer checksum is verified.
/// Segments are at most `oplog_segment_max_bytes` (default 64 MB), making
/// in-memory loading practical.
#[derive(Debug)]
pub struct SegmentReader {
    /// Decoded segment header.
    pub header: SegmentHeader,
    /// Decoded segment footer.
    pub footer: SegmentFooter,
    entries: Vec<OplogEntry>,
}

impl SegmentReader {
    /// Open, fully validate, and load all entries from a sealed segment.
    pub fn open(path: &Path) -> StorageResult<Self> {
        let data = std::fs::read(path)
            .map_err(|e| StorageError::Io(format!("read segment {:?}: {e}", path)))?;

        let total_len = data.len() as u64;
        let min_len = HEADER_SIZE + FOOTER_SIZE;
        if total_len < min_len {
            return Err(StorageError::Io(format!(
                "segment {:?} too short: {} bytes (minimum {})",
                path, total_len, min_len
            )));
        }

        let mut cursor = Cursor::new(&data);

        // Validate header
        let header = read_header(&mut cursor)?;
        if header.version != FORMAT_VERSION {
            return Err(StorageError::Io(format!(
                "unsupported oplog version {} in {:?}",
                header.version, path
            )));
        }

        // Validate footer (at end of file)
        cursor
            .seek(SeekFrom::End(-(FOOTER_SIZE as i64)))
            .map_err(io_err)?;
        let footer = read_footer(&mut cursor)?;

        // Parse entries section
        cursor.set_position(HEADER_SIZE);
        let entries_end = total_len - FOOTER_SIZE;

        let mut entries = Vec::with_capacity(footer.entry_count as usize);
        for entry_idx in 0..footer.entry_count {
            let pos = cursor.position();
            if pos >= entries_end {
                return Err(StorageError::Io(format!(
                    "segment {:?}: premature end of data at entry {entry_idx} (pos={pos}, entries_end={entries_end})",
                    path
                )));
            }

            let payload_len = decode_varint(&mut cursor)
                .map_err(|e| StorageError::Io(format!("varint at entry {entry_idx}: {e}")))?
                as usize;

            let remaining = (entries_end - cursor.position()) as usize;
            if remaining < payload_len + 4 {
                return Err(StorageError::Io(format!(
                    "segment {:?}: entry {entry_idx} claims {payload_len} bytes but only {remaining} remain",
                    path
                )));
            }

            // Read msgpack payload
            let mut payload = vec![0u8; payload_len];
            cursor
                .read_exact(&mut payload)
                .map_err(|e| StorageError::Io(format!("payload at entry {entry_idx}: {e}")))?;

            // Read and verify crc32
            let mut crc_buf = [0u8; 4];
            cursor
                .read_exact(&mut crc_buf)
                .map_err(|e| StorageError::Io(format!("crc32 at entry {entry_idx}: {e}")))?;

            let expected = crc32fast::hash(&payload);
            let actual = u32::from_le_bytes(crc_buf);
            if expected != actual {
                return Err(StorageError::ChecksumMismatch {
                    expected,
                    actual,
                    context: format!("entry {entry_idx} in {:?}", path),
                });
            }

            let entry = OplogEntry::decode(&payload)
                .map_err(|e| StorageError::Serialization(e.to_string()))?;
            entries.push(entry);
        }

        Ok(Self {
            header,
            footer,
            entries,
        })
    }

    /// All entries in this segment in order.
    pub fn entries(&self) -> &[OplogEntry] {
        &self.entries
    }

    /// Consume the reader and return the entries.
    pub fn into_entries(self) -> Vec<OplogEntry> {
        self.entries
    }

    /// Best-effort forward scan of a segment file that may lack a valid footer.
    ///
    /// Used during crash recovery: after an unclean shutdown the active segment
    /// was fsynced but never sealed (no footer written).  [`open`](Self::open)
    /// would fail because the footer is missing, so this method performs a raw
    /// forward scan and stops at the first parse or checksum error.
    ///
    /// Only entries whose CRC32 matches are returned; any trailing partial
    /// write is silently discarded — it was not durable.
    ///
    /// Returns an error only for I/O failures (file unreadable, wrong magic,
    /// unsupported version, wrong `shard_id`).  An empty `Vec` is returned for
    /// a segment that contains no valid entries (e.g., only a header).
    pub fn scan_without_footer(path: &Path, shard_id: u32) -> StorageResult<Vec<OplogEntry>> {
        let data = std::fs::read(path)
            .map_err(|e| StorageError::Io(format!("read segment {:?}: {e}", path)))?;

        if (data.len() as u64) < HEADER_SIZE {
            // File is too short even for the header — treat as empty.
            return Ok(Vec::new());
        }

        let mut cursor = Cursor::new(&data);
        let header = read_header(&mut cursor)?;

        if header.version != FORMAT_VERSION {
            return Err(StorageError::Io(format!(
                "unsupported oplog version {} in {:?}",
                header.version, path
            )));
        }
        if header.shard_id != shard_id {
            return Ok(Vec::new());
        }

        // Scan forward, collecting entries with valid CRC32.  Stop at the
        // first parse error — that marks the boundary of durable data.
        let mut entries = Vec::new();
        loop {
            let pos_before = cursor.position() as usize;

            // Try to read varint length prefix.
            let payload_len = match decode_varint(&mut cursor) {
                Ok(n) => n as usize,
                Err(_) => break, // EOF or partial varint → done
            };

            let pos_after_varint = cursor.position() as usize;
            let remaining = data.len().saturating_sub(pos_after_varint);

            // Need payload_len bytes + 4 bytes CRC32.
            if remaining < payload_len + 4 {
                // Partial frame — not durable.
                break;
            }

            let payload = &data[pos_after_varint..pos_after_varint + payload_len];
            let crc_bytes =
                &data[pos_after_varint + payload_len..pos_after_varint + payload_len + 4];

            let expected = crc32fast::hash(payload);
            let actual =
                u32::from_le_bytes([crc_bytes[0], crc_bytes[1], crc_bytes[2], crc_bytes[3]]);

            if expected != actual {
                // Checksum mismatch → partial / corrupt tail, stop here.
                break;
            }

            match OplogEntry::decode(payload) {
                Ok(entry) => {
                    entries.push(entry);
                    // Advance cursor past payload + CRC32.
                    cursor.set_position((pos_after_varint + payload_len + 4) as u64);
                }
                Err(_) => {
                    // Deserialization error → corrupt entry, rewind and stop.
                    cursor.set_position(pos_before as u64);
                    break;
                }
            }
        }

        Ok(entries)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::oplog::entry::OplogOp;

    fn make_entry(index: u64, ts: u64) -> OplogEntry {
        OplogEntry {
            ts,
            term: 1,
            index,
            shard: 0,
            ops: vec![OplogOp::Insert {
                partition: 1,
                key: format!("key-{index}").into_bytes(),
                value: b"val".to_vec(),
            }],
            is_migration: false,
            pre_images: None,
        }
    }

    #[test]
    fn varint_roundtrip_values() {
        for &v in &[
            0u64,
            1,
            127,
            128,
            255,
            16_383,
            16_384,
            u32::MAX as u64,
            u64::MAX / 2,
        ] {
            let mut buf = Vec::new();
            encode_varint(v, &mut buf);
            let mut cursor = Cursor::new(&buf);
            let decoded = decode_varint(&mut cursor).expect("decode");
            assert_eq!(v, decoded, "varint roundtrip failed for {v}");
        }
    }

    #[test]
    fn varint_1_byte_for_small_values() {
        let mut buf = Vec::new();
        encode_varint(0, &mut buf);
        assert_eq!(buf.len(), 1);
        encode_varint(127, &mut buf);
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn segment_write_read_roundtrip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("seg-00000.bin");

        let entries: Vec<_> = (0..5u64).map(|i| make_entry(i, 1000 + i)).collect();

        // Write
        let mut writer = SegmentWriter::create(&path, 42, 0).expect("create");
        for e in &entries {
            writer.append(e).expect("append");
        }
        assert_eq!(writer.entry_count(), 5);
        writer.seal().expect("seal");

        // Read back
        let reader = SegmentReader::open(&path).expect("open");
        assert_eq!(reader.header.shard_id, 42);
        assert_eq!(reader.header.first_index, 0);
        assert_eq!(reader.header.version, FORMAT_VERSION);
        assert_eq!(reader.footer.entry_count, 5);
        assert_eq!(reader.footer.first_ts, 1000);
        assert_eq!(reader.footer.last_ts, 1004);
        assert_eq!(reader.entries(), &entries[..]);
    }

    #[test]
    fn empty_segment_roundtrip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("empty.bin");

        let writer = SegmentWriter::create(&path, 1, 100).expect("create");
        writer.seal().expect("seal");

        let reader = SegmentReader::open(&path).expect("open");
        assert_eq!(reader.footer.entry_count, 0);
        assert!(reader.entries().is_empty());
        assert_eq!(reader.header.first_index, 100);
    }

    #[test]
    fn entry_checksum_mismatch_detected() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("corrupt.bin");

        let mut writer = SegmentWriter::create(&path, 0, 0).expect("create");
        writer.append(&make_entry(0, 100)).expect("append");
        writer.seal().expect("seal");

        // Flip a bit in the middle of the entries section
        let mut data = std::fs::read(&path).expect("read");
        let mid = (HEADER_SIZE as usize) + 5; // middle of first entry payload
        data[mid] ^= 0xFF;
        std::fs::write(&path, &data).expect("write corrupt");

        let result = SegmentReader::open(&path);
        assert!(result.is_err(), "should detect entry corruption");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("checksum") || msg.contains("mismatch"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn footer_checksum_mismatch_detected() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bad-footer.bin");

        let mut writer = SegmentWriter::create(&path, 0, 0).expect("create");
        writer.append(&make_entry(0, 100)).expect("append");
        writer.seal().expect("seal");

        // Corrupt the footer checksum bytes (last 4 bytes of file)
        let mut data = std::fs::read(&path).expect("read");
        let len = data.len();
        data[len - 1] ^= 0xFF;
        std::fs::write(&path, &data).expect("write");

        let result = SegmentReader::open(&path);
        assert!(result.is_err(), "should detect footer corruption");
    }

    #[test]
    fn invalid_magic_rejected() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("invalid.bin");

        // Write garbage — valid length (>= 50B) but wrong magic
        let garbage = vec![0xFFu8; 64];
        std::fs::write(&path, &garbage).expect("write");

        let result = SegmentReader::open(&path);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("magic"), "unexpected error: {msg}");
    }

    #[test]
    fn too_short_file_rejected() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("short.bin");
        std::fs::write(&path, b"OPLO").expect("write");

        let result = SegmentReader::open(&path);
        assert!(result.is_err());
    }

    #[test]
    fn large_entry_roundtrip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("large.bin");

        // 16 KB value
        let large_value = vec![0xABu8; 16 * 1024];
        let entry = OplogEntry {
            ts: 9999,
            term: 1,
            index: 0,
            shard: 0,
            ops: vec![crate::oplog::entry::OplogOp::Insert {
                partition: 0,
                key: b"bigkey".to_vec(),
                value: large_value.clone(),
            }],
            is_migration: false,
            pre_images: None,
        };

        let mut writer = SegmentWriter::create(&path, 0, 0).expect("create");
        writer.append(&entry).expect("append");
        writer.seal().expect("seal");

        let reader = SegmentReader::open(&path).expect("open");
        assert_eq!(reader.entries().len(), 1);
        if let crate::oplog::entry::OplogOp::Insert { ref value, .. } = reader.entries()[0].ops[0] {
            assert_eq!(value, &large_value);
        } else {
            panic!("expected Insert op");
        }
    }

    /// `flush_and_sync()` succeeds on a freshly created segment with no entries.
    #[test]
    fn flush_and_sync_empty_writer() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("empty-sync.bin");
        let mut writer = SegmentWriter::create(&path, 0, 0).expect("create");
        // No entries written — flush_and_sync must still succeed.
        writer
            .flush_and_sync()
            .expect("flush_and_sync on empty writer");
    }

    /// `flush_and_sync()` makes appended entries durably readable after re-seal.
    ///
    /// Verifies that calling flush_and_sync() after append() ensures the
    /// BufWriter is flushed. After a subsequent seal() + SegmentReader::open(),
    /// all entries must be present.
    #[test]
    fn flush_and_sync_then_seal_readable() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("sync-then-seal.bin");

        let mut writer = SegmentWriter::create(&path, 7, 0).expect("create");
        for i in 0..4u64 {
            writer.append(&make_entry(i, 2000 + i)).expect("append");
        }
        // Fsync before sealing (the "ONE fsync per write batch" path).
        writer.flush_and_sync().expect("flush_and_sync");

        // After sync, seal and read back.
        writer.seal().expect("seal");
        let reader = SegmentReader::open(&path).expect("open");
        assert_eq!(
            reader.footer.entry_count, 4,
            "all 4 entries must survive flush_and_sync"
        );
        assert_eq!(reader.entries()[0].index, 0);
        assert_eq!(reader.entries()[3].index, 3);
    }
}
