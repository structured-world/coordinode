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

use crate::engine::config::SyncMethod;
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
            MAGIC, magic
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

/// `true` when the last `FOOTER_SIZE` bytes of `data` are a footer whose
/// checksum holds, i.e. the segment was sealed.
fn has_valid_footer(data: &[u8]) -> bool {
    let Some(start) = data.len().checked_sub(FOOTER_SIZE as usize) else {
        return false;
    };
    start >= HEADER_SIZE as usize && read_footer(&mut Cursor::new(&data[start..])).is_ok()
}

/// `true` when the segment at `path` is sealed: its last `FOOTER_SIZE` bytes
/// are a footer whose checksum holds and whose entry bytes span exactly the
/// file between header and footer. Reads only the footer. A sealed segment is
/// never written again, so the answer holds for as long as the file exists.
///
/// # Errors
///
/// The file cannot be opened or read.
pub(crate) fn is_sealed(path: &Path) -> StorageResult<bool> {
    let mut file = std::fs::File::open(path)
        .map_err(|e| StorageError::Io(format!("open segment {path:?}: {e}")))?;
    let len = file
        .metadata()
        .map_err(|e| StorageError::Io(format!("stat segment {path:?}: {e}")))?
        .len();
    let Some(entries) = len.checked_sub(HEADER_SIZE + FOOTER_SIZE) else {
        return Ok(false);
    };
    file.seek(SeekFrom::Start(len - FOOTER_SIZE))
        .map_err(|e| StorageError::Io(format!("seek segment {path:?}: {e}")))?;
    let mut tail = [0u8; FOOTER_SIZE as usize];
    file.read_exact(&mut tail)
        .map_err(|e| StorageError::Io(format!("read footer {path:?}: {e}")))?;
    Ok(read_footer(&mut Cursor::new(&tail[..])).is_ok_and(|f| f.total_bytes == entries))
}

/// The complete entries at the start of an entries section, and where they end.
///
/// Parsing stops at the first frame that is incomplete, fails its checksum or
/// does not decode: after a crash that is the write that never finished, and
/// nothing after it was acknowledged.
struct FramePrefix {
    entries: Vec<OplogEntry>,
    /// Offset in the file just past the last complete frame.
    end: u64,
    first_ts: u64,
    last_ts: u64,
}

fn read_frame_prefix(data: &[u8]) -> FramePrefix {
    let mut prefix = FramePrefix {
        entries: Vec::new(),
        end: HEADER_SIZE,
        first_ts: 0,
        last_ts: 0,
    };
    for (entry, end) in parse_frames(data, HEADER_SIZE) {
        if prefix.entries.is_empty() {
            prefix.first_ts = entry.ts;
        }
        prefix.last_ts = entry.ts;
        prefix.entries.push(entry);
        prefix.end = end;
    }
    prefix
}

/// The complete frames of `data` from offset `start` on, each with the
/// offset just past it; parsing stops at the first frame that is incomplete,
/// fails its checksum or does not decode (a write in progress, or the footer
/// of a sealed segment).
pub(crate) fn parse_frames(data: &[u8], start: u64) -> Vec<(OplogEntry, u64)> {
    let mut cursor = Cursor::new(data);
    cursor.set_position(start);
    let mut frames = Vec::new();
    let total_len = data.len() as u64;
    while let Ok(payload_len) = decode_varint(&mut cursor) {
        let payload_start = cursor.position();
        // `payload_len + 4` cannot overflow a u64 built from a 10-byte varint
        // in practice, but a hostile length must not wrap either.
        let Some(frame_end) = payload_len
            .checked_add(4)
            .and_then(|n| payload_start.checked_add(n))
        else {
            break;
        };
        if frame_end > total_len {
            break;
        }
        let payload = &data[payload_start as usize..(payload_start + payload_len) as usize];
        let crc_at = (payload_start + payload_len) as usize;
        let crc = u32::from_le_bytes([
            data[crc_at],
            data[crc_at + 1],
            data[crc_at + 2],
            data[crc_at + 3],
        ]);
        if crc32fast::hash(payload) != crc {
            break;
        }
        let Ok(entry) = OplogEntry::decode(payload) else {
            break;
        };
        frames.push((entry, frame_end));
        cursor.set_position(frame_end);
    }
    frames
}

/// The complete frames of the segment at `path` from byte `offset` on (at
/// least [`HEADER_SIZE`], the first frame), each with the offset just past
/// it. Only the bytes from `offset` are read, so a reader that follows a
/// growing segment pays for what was appended, not for the whole file.
///
/// # Errors
///
/// The file cannot be opened or read, or (when reading from the first frame)
/// its header is not a segment header of this format.
pub fn read_frames_from(path: &Path, offset: u64) -> StorageResult<Vec<(OplogEntry, u64)>> {
    let mut file = std::fs::File::open(path)
        .map_err(|e| StorageError::Io(format!("open segment {path:?}: {e}")))?;
    let offset = offset.max(HEADER_SIZE);
    if offset == HEADER_SIZE {
        let mut header = [0u8; HEADER_SIZE as usize];
        file.read_exact(&mut header)
            .map_err(|e| StorageError::Io(format!("read segment header {path:?}: {e}")))?;
        let header = read_header(&mut Cursor::new(&header[..]))?;
        if header.version != FORMAT_VERSION {
            return Err(StorageError::Io(format!(
                "unsupported oplog version {} in {path:?}",
                header.version
            )));
        }
    }
    file.seek(SeekFrom::Start(offset))
        .map_err(|e| StorageError::Io(format!("seek segment {path:?}: {e}")))?;
    let mut tail = Vec::new();
    file.read_to_end(&mut tail)
        .map_err(|e| StorageError::Io(format!("read segment {path:?}: {e}")))?;
    Ok(parse_frames(&tail, 0)
        .into_iter()
        .map(|(entry, end)| (entry, offset + end))
        .collect())
}

/// Up to `len` bytes of the file at `path` from byte `offset`, fewer at its
/// end.
///
/// # Errors
///
/// The file cannot be opened or read.
pub(crate) fn read_span(path: &Path, offset: u64, len: u64) -> StorageResult<Vec<u8>> {
    let mut file = std::fs::File::open(path)
        .map_err(|e| StorageError::Io(format!("open segment {path:?}: {e}")))?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|e| StorageError::Io(format!("seek segment {path:?}: {e}")))?;
    let mut data = Vec::new();
    file.take(len)
        .read_to_end(&mut data)
        .map_err(|e| StorageError::Io(format!("read segment {path:?}: {e}")))?;
    Ok(data)
}

/// Refuse `bytes` (at least [`HEADER_SIZE`] of them) unless they open a
/// segment of this format.
///
/// # Errors
///
/// Too short, a wrong magic, or another format version.
pub(crate) fn check_header(bytes: &[u8], path: &Path) -> StorageResult<()> {
    let header = read_header(&mut Cursor::new(bytes))?;
    if header.version != FORMAT_VERSION {
        return Err(StorageError::Io(format!(
            "unsupported oplog version {} in {path:?}",
            header.version
        )));
    }
    Ok(())
}

/// The end of each complete frame at the start of `data` whose checksum
/// holds, relative to `data`, without decoding the payloads. Stops where
/// [`parse_frames`] would for a frame cut short or failing its checksum (a
/// write in progress, or a sealed segment's footer).
pub(crate) fn frame_ends(data: &[u8]) -> Vec<u64> {
    let mut cursor = Cursor::new(data);
    let mut ends = Vec::new();
    let total_len = data.len() as u64;
    while let Ok(payload_len) = decode_varint(&mut cursor) {
        let payload_start = cursor.position();
        let Some(frame_end) = payload_len
            .checked_add(4)
            .and_then(|n| payload_start.checked_add(n))
        else {
            break;
        };
        if frame_end > total_len {
            break;
        }
        // Bounded by `total_len` just above.
        let crc_at = (frame_end - 4) as usize;
        let payload = &data[payload_start as usize..crc_at];
        let crc = u32::from_le_bytes([
            data[crc_at],
            data[crc_at + 1],
            data[crc_at + 2],
            data[crc_at + 3],
        ]);
        if crc32fast::hash(payload) != crc {
            break;
        }
        ends.push(frame_end);
        cursor.set_position(frame_end);
    }
    ends
}

/// Decode `count` consecutive frames filling `data` exactly.
///
/// Unlike [`read_frame_prefix`], any bad frame is an error: the caller asks
/// for frames it knows were written whole.
fn decode_frames(data: &[u8], count: usize, path: &Path) -> StorageResult<Vec<OplogEntry>> {
    let mut cursor = Cursor::new(data);
    let mut entries = Vec::with_capacity(count);
    for entry_idx in 0..count {
        let payload_len = decode_varint(&mut cursor)
            .map_err(|e| StorageError::Io(format!("varint at entry {entry_idx}: {e}")))?;
        let payload_start = cursor.position();
        let frame_end = payload_len
            .checked_add(4)
            .and_then(|n| payload_start.checked_add(n))
            .filter(|&end| end <= data.len() as u64)
            .ok_or_else(|| {
                StorageError::Io(format!(
                    "segment {path:?}: entry {entry_idx} claims {payload_len} bytes past the frames read"
                ))
            })?;
        // Bounded by `data.len()` just above.
        let payload = &data[payload_start as usize..(frame_end - 4) as usize];
        let crc_at = (frame_end - 4) as usize;
        let actual = u32::from_le_bytes([
            data[crc_at],
            data[crc_at + 1],
            data[crc_at + 2],
            data[crc_at + 3],
        ]);
        let expected = crc32fast::hash(payload);
        if expected != actual {
            return Err(StorageError::ChecksumMismatch {
                expected,
                actual,
                context: format!("entry {entry_idx} in {path:?}"),
            });
        }
        entries.push(
            OplogEntry::decode(payload).map_err(|e| StorageError::Serialization(e.to_string()))?,
        );
        cursor.set_position(frame_end);
    }
    if cursor.position() != data.len() as u64 {
        return Err(StorageError::Io(format!(
            "segment {path:?}: {} bytes after the last frame read",
            data.len() as u64 - cursor.position()
        )));
    }
    Ok(entries)
}

/// What [`SegmentWriter::recover_tail`] found and did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TailRecovery {
    /// The segment already carries a valid footer; nothing was changed.
    Sealed,
    /// The segment had no valid footer. Its complete entries were kept, any
    /// torn bytes after them cut off, and a footer written.
    Resealed {
        /// Entries recovered into the sealed segment.
        entries: u32,
        /// Bytes after the last complete entry that were cut off.
        discarded_bytes: u64,
    },
    /// The segment held no complete entry and was removed.
    Removed,
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
    /// `(index, file offset of its frame)` of every entry, ascending by
    /// index: a read of the live tail seeks straight to what it asks for.
    frames: Vec<(u64, u64)>,
    /// Read handle of this file, opened on the first read. A handle of its
    /// own keeps its file position apart from the writer's.
    reader: Option<std::fs::File>,
    /// How an append is made durable.
    sync: SyncMethod,
    /// Handed out by [`Self::flush_to_os`]; the file is duplicated once per
    /// segment, not once per batch.
    sync_handle: Option<SyncHandle>,
}

/// Create `path`, which must not exist, for writing under `sync`.
fn create_file(path: &Path, sync: SyncMethod) -> std::io::Result<std::fs::File> {
    if sync != SyncMethod::OpenDatasync {
        return std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path);
    }
    #[cfg(unix)]
    {
        use rustix::fs::{Mode, OFlags};
        let fd = rustix::fs::open(
            path,
            OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::DSYNC | OFlags::CLOEXEC,
            Mode::from_raw_mode(0o644),
        )?;
        Ok(std::fs::File::from(fd))
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        // FILE_FLAG_WRITE_THROUGH (CreateFileW, Win32 API): writes go
        // through the cache to the medium before the call returns.
        const FILE_FLAG_WRITE_THROUGH: u32 = 0x8000_0000;
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .custom_flags(FILE_FLAG_WRITE_THROUGH)
            .open(path)
    }
}

/// What makes a segment's written bytes durable, apart from the writer: a
/// log that syncs on another thread takes this under the append lock and
/// syncs after releasing it, so appends go on while the sync runs.
#[derive(Clone, Debug)]
pub struct SyncHandle {
    file: std::sync::Arc<std::fs::File>,
    sync: SyncMethod,
}

impl SyncHandle {
    /// A handle that syncs `file` under `sync`, for a log that owns its file
    /// outside a [`SegmentWriter`].
    pub fn new(file: std::fs::File, sync: SyncMethod) -> Self {
        Self {
            file: std::sync::Arc::new(file),
            sync,
        }
    }

    /// Make every byte written to the segment before this handle was taken
    /// durable under the segment's sync method.
    ///
    /// # Errors
    ///
    /// The sync fails.
    pub fn sync(&self) -> StorageResult<()> {
        sync_file(&self.file, self.sync).map_err(|e| StorageError::Io(format!("sync segment: {e}")))
    }
}

/// Make everything written to `file` durable under `sync`.
fn sync_file(file: &std::fs::File, sync: SyncMethod) -> std::io::Result<()> {
    match sync {
        // Each write already went out with O_DSYNC / write-through.
        SyncMethod::OpenDatasync => Ok(()),
        // std reaches the medium: F_FULLFSYNC on Apple, fdatasync on Linux,
        // FlushFileBuffers on Windows.
        SyncMethod::Full => file.sync_data(),
        SyncMethod::Fsync => {
            // Apple's fsync(2) stops at the drive's cache; elsewhere the plain
            // call already reaches the medium, so it is the full one.
            #[cfg(target_vendor = "apple")]
            {
                rustix::fs::fsync(file).map_err(std::io::Error::from)
            }
            #[cfg(not(target_vendor = "apple"))]
            {
                file.sync_data()
            }
        }
    }
}

impl SegmentWriter {
    /// Create a new segment file at `path` and write the 18-byte header;
    /// appends are made durable under `sync`.
    ///
    /// The file must not already exist (`create_new` semantics).
    pub fn create(
        path: &Path,
        shard_id: u32,
        first_index: u64,
        sync: SyncMethod,
    ) -> StorageResult<Self> {
        let file = create_file(path, sync)
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
            frames: Vec::new(),
            reader: None,
            sync,
            sync_handle: None,
        })
    }

    /// Create the segment at `path`, replacing a leftover from an interrupted
    /// write.
    ///
    /// A crash between creating a segment and durably recording its entries
    /// leaves a file that recovery does not count towards `last_log_id`. The
    /// log store then picks that same index again on the next append and
    /// [`create`](Self::create) refuses it, which wedged the database: every
    /// write failed with "File exists" until somebody deleted the file by
    /// hand. A leftover carrying no entries is exactly that, an empty header,
    /// and is replaced. One that carries entries is a real conflict and still
    /// refuses, so recovery never destroys durable data.
    pub fn create_or_replace_empty(
        path: &Path,
        shard_id: u32,
        first_index: u64,
        sync: SyncMethod,
    ) -> StorageResult<Self> {
        match Self::create(path, shard_id, first_index, sync) {
            Ok(writer) => return Ok(writer),
            // Anything other than a pre-existing file is a genuine I/O failure.
            Err(e) if !path.exists() => return Err(e),
            Err(_) => {}
        }

        // The segment was never sealed, so it has no footer to read a count
        // from; scan it directly. An unreadable prefix means nothing survived
        // the crash, which is the same as empty for our purposes.
        let salvaged = SegmentReader::scan_without_footer(path, shard_id).unwrap_or_default();
        if !salvaged.is_empty() {
            return Err(StorageError::Io(format!(
                "segment {:?} already exists and holds {} entries; refusing to \
                 replace it",
                path,
                salvaged.len()
            )));
        }

        std::fs::remove_file(path)
            .map_err(|e| StorageError::Io(format!("remove empty segment {:?}: {e}", path)))?;
        Self::create(path, shard_id, first_index, sync)
    }

    /// Seal the segment at `path` if a crash left it unsealed.
    ///
    /// Entries are made durable one `fsync` at a time, and the footer is only
    /// written when a segment rotates, so a killed process leaves its active
    /// segment with acknowledged entries and no footer, possibly followed by a
    /// frame (or part of a footer) that was being written. This keeps every
    /// complete entry, cuts off what follows the last one, and writes the
    /// footer, so the segment reads like any other. A segment with no complete
    /// entry never held anything acknowledged and is removed.
    ///
    /// Only a missing or invalid footer counts as a crash. A segment with a
    /// valid footer is left untouched even if an entry inside it is damaged:
    /// that is corruption, and cutting there would drop the entries after it.
    /// Safe to repeat: a crash during recovery leaves a state it recovers
    /// from again.
    ///
    /// # Errors
    ///
    /// I/O failures, and a header that is not an oplog segment of `shard_id`
    /// in the supported format (not a crash leftover; refused rather than
    /// rewritten).
    pub fn recover_tail(path: &Path, shard_id: u32) -> StorageResult<TailRecovery> {
        let data = std::fs::read(path)
            .map_err(|e| StorageError::Io(format!("read segment {:?}: {e}", path)))?;
        if has_valid_footer(&data) {
            return Ok(TailRecovery::Sealed);
        }
        let remove = || {
            std::fs::remove_file(path)
                .map_err(|e| StorageError::Io(format!("remove empty segment {:?}: {e}", path)))
                .map(|()| TailRecovery::Removed)
        };
        if (data.len() as u64) < HEADER_SIZE {
            // The header itself never reached the disk.
            return remove();
        }

        let header = read_header(&mut Cursor::new(&data))?;
        if header.version != FORMAT_VERSION {
            return Err(StorageError::Io(format!(
                "unsupported oplog version {} in {:?}",
                header.version, path
            )));
        }
        if header.shard_id != shard_id {
            return Err(StorageError::Io(format!(
                "segment {:?} belongs to shard {}, not {shard_id}",
                path, header.shard_id
            )));
        }

        let prefix = read_frame_prefix(&data);
        if prefix.entries.is_empty() {
            return remove();
        }
        let entries = u32::try_from(prefix.entries.len()).map_err(|_| {
            StorageError::Io(format!(
                "segment {:?} holds more entries than a footer counts",
                path
            ))
        })?;
        let footer = SegmentFooter {
            entry_count: entries,
            first_ts: prefix.first_ts,
            last_ts: prefix.last_ts,
            total_bytes: prefix.end - HEADER_SIZE,
        };

        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .map_err(|e| StorageError::Io(format!("open segment {:?}: {e}", path)))?;
        file.set_len(prefix.end)
            .map_err(|e| StorageError::Io(format!("truncate segment {:?}: {e}", path)))?;
        file.seek(SeekFrom::Start(prefix.end)).map_err(io_err)?;
        write_footer(&mut file, &footer)
            .map_err(|e| StorageError::Io(format!("write footer {:?}: {e}", path)))?;
        file.sync_data()
            .map_err(|e| StorageError::Io(format!("sync_data on recovery {:?}: {e}", path)))?;

        Ok(TailRecovery::Resealed {
            entries,
            discarded_bytes: data.len() as u64 - prefix.end,
        })
    }

    /// Append one [`OplogEntry`] to the segment.
    ///
    /// Frame layout: `varint(payload_len) || msgpack_bytes || crc32_le(4B)`.
    ///
    /// # Errors
    ///
    /// An index not above the last one appended here (reads find entries by
    /// index), a serialization failure or an I/O failure.
    pub fn append(&mut self, entry: &OplogEntry) -> StorageResult<()> {
        if let Some(&(last, _)) = self.frames.last() {
            if entry.index <= last {
                return Err(StorageError::Io(format!(
                    "oplog entry {} appended after entry {last} in {:?}",
                    entry.index, self.path
                )));
            }
        }
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
        self.frames
            .push((entry.index, HEADER_SIZE + self.total_bytes));
        self.total_bytes += frame_bytes;
        self.entry_count += 1;

        if self.entry_count == 1 {
            self.first_ts = entry.ts;
        }
        self.last_ts = entry.ts;

        Ok(())
    }

    /// Entries with `index ∈ [from_index, to_index)`, read back from this
    /// still-open segment.
    ///
    /// Reads only the frames in range, so reading the live tail costs what
    /// it returns, not the size of the segment.
    ///
    /// # Errors
    ///
    /// I/O failures, and a frame in range that fails its checksum or does not
    /// decode (this process wrote it, so that is corruption, not a torn write).
    pub fn read_range(&mut self, from_index: u64, to_index: u64) -> StorageResult<Vec<OplogEntry>> {
        let first = self
            .frames
            .partition_point(|&(index, _)| index < from_index);
        let last = self.frames.partition_point(|&(index, _)| index < to_index);
        if first >= last {
            return Ok(Vec::new());
        }
        let start = self.frames[first].1;
        let end = self
            .frames
            .get(last)
            .map_or(HEADER_SIZE + self.total_bytes, |&(_, offset)| offset);

        // What the reader sees must include every frame appended so far.
        self.file
            .flush()
            .map_err(|e| StorageError::Io(format!("flush segment: {e}")))?;
        let reader = match &mut self.reader {
            Some(reader) => reader,
            slot => slot.insert(std::fs::File::open(&self.path).map_err(|e| {
                StorageError::Io(format!("open segment {:?} for reading: {e}", self.path))
            })?),
        };
        // Offsets are within a file this process wrote; they fit in memory.
        let len = usize::try_from(end - start).map_err(io_err)?;
        let mut data = vec![0u8; len];
        reader.seek(SeekFrom::Start(start)).map_err(io_err)?;
        reader.read_exact(&mut data).map_err(|e| {
            StorageError::Io(format!("read segment {:?} at {start}: {e}", self.path))
        })?;
        decode_frames(&data, last - first, &self.path)
    }

    /// Index of the first entry written here, if any.
    pub fn first_entry_index(&self) -> Option<u64> {
        self.frames.first().map(|&(index, _)| index)
    }

    /// How this writer makes appends durable.
    #[cfg(test)]
    pub(crate) fn sync_method(&self) -> SyncMethod {
        self.sync
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
        // Step 2: make the written bytes durable under the configured method.
        sync_file(self.file.get_ref(), self.sync)
            .map_err(|e| StorageError::Io(format!("sync segment: {e}")))?;
        Ok(())
    }

    /// Hand everything appended so far to the kernel and return the handle
    /// that makes it durable: [`flush_and_sync`](Self::flush_and_sync) split
    /// in two, so the sync can run without holding this writer.
    ///
    /// # Errors
    ///
    /// The flush fails, or the file cannot be duplicated for the handle.
    pub fn flush_to_os(&mut self) -> StorageResult<SyncHandle> {
        self.file
            .flush()
            .map_err(|e| StorageError::Io(format!("flush segment: {e}")))?;
        if let Some(handle) = &self.sync_handle {
            return Ok(handle.clone());
        }
        let file = self
            .file
            .get_ref()
            .try_clone()
            .map_err(|e| StorageError::Io(format!("duplicate segment {:?}: {e}", self.path)))?;
        let handle = SyncHandle {
            file: std::sync::Arc::new(file),
            sync: self.sync,
        };
        Ok(self.sync_handle.insert(handle).clone())
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
        // Also sync on seal — the footer must be durable before we remove
        // the active-writer reference and add the path to sealed[].
        sync_file(self.file.get_ref(), self.sync)
            .map_err(|e| StorageError::Io(format!("sync on seal: {e}")))?;
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

    /// Open a segment that has NOT been sealed yet (no footer) and load
    /// every complete entry from it.
    ///
    /// The per-entry framing (`varint(len) || payload || crc32`) makes a
    /// prefix read safe: parsing stops at the first incomplete or
    /// corrupt frame, which on an actively written segment is simply the
    /// entry the writer has not finished flushing. Live oplog consumers
    /// (index maintenance, CDC) use this to tail the ACTIVE segment
    /// instead of waiting up to a full rotation for it to seal.
    ///
    /// The returned reader carries a synthesized footer describing the
    /// complete prefix; callers must treat the segment as still growing
    /// and re-read it for new entries.
    pub fn open_active(path: &Path) -> StorageResult<Self> {
        let data = std::fs::read(path)
            .map_err(|e| StorageError::Io(format!("read segment {:?}: {e}", path)))?;

        let total_len = data.len() as u64;
        if total_len < HEADER_SIZE {
            return Err(StorageError::Io(format!(
                "segment {:?} too short for a header: {} bytes",
                path, total_len
            )));
        }

        let header = read_header(&mut Cursor::new(&data))?;
        if header.version != FORMAT_VERSION {
            return Err(StorageError::Io(format!(
                "unsupported oplog version {} in {:?}",
                header.version, path
            )));
        }

        // The writer may be mid-flush: the prefix stops at its unfinished frame.
        let prefix = read_frame_prefix(&data);
        let footer = SegmentFooter {
            entry_count: prefix.entries.len() as u32,
            first_ts: prefix.first_ts,
            last_ts: prefix.last_ts,
            total_bytes: prefix.end,
        };
        Ok(Self {
            header,
            footer,
            entries: prefix.entries,
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

        let header = read_header(&mut Cursor::new(&data))?;

        if header.version != FORMAT_VERSION {
            return Err(StorageError::Io(format!(
                "unsupported oplog version {} in {:?}",
                header.version, path
            )));
        }
        if header.shard_id != shard_id {
            return Ok(Vec::new());
        }

        // The first incomplete or failing frame marks the end of durable data.
        Ok(read_frame_prefix(&data).entries)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests;
