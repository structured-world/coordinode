//! Segment piece model: split a segment into per-piece-encoded, checksummed
//! pieces, verify each piece on arrival, and stream-decode them into a sink with
//! an end-to-end checksum.
//!
//! Pieces may be transferred raw or compressed ([`PieceEncoding`]); the manifest
//! records the encoding. Piece checksums are taken over the **wire** (encoded)
//! bytes so a receiver verifies what it received before spending CPU to decode,
//! while the total checksum is over the **raw** segment so the fully decoded
//! result is verified end to end.
//!
//! Assembly is **streaming**: [`SegmentWriter`] decodes each piece into a caller
//! sink ([`std::io::Write`]) as it arrives, holding only one piece at a time —
//! never the whole (16-64 MB) segment — and folds the raw bytes into a running
//! checksum. [`assemble`] is a thin `Vec`-backed convenience over it.

use std::io::{self, Write};

use xxhash_rust::xxh3::{xxh3_64, Xxh3};

/// Index of a piece within a segment.
pub type PieceIndex = u32;

/// Storage media class of an endpoint, which sets the optimal transfer piece
/// size (tuned per storage media type).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaClass {
    /// RAM ↔ RAM: large pieces amortize bulk memcpy.
    Ram,
    /// NVMe: 1 MB is optimal for PCIe transfers.
    Nvme,
    /// SSD: 1 MB.
    Ssd,
    /// HDD: large pieces to minimize seeks.
    Hdd,
}

impl MediaClass {
    /// Optimal piece size in bytes for this media class.
    pub const fn optimal_piece_size(self) -> usize {
        const MB: usize = 1024 * 1024;
        match self {
            MediaClass::Ram => 4 * MB,
            MediaClass::Nvme | MediaClass::Ssd => MB,
            MediaClass::Hdd => 4 * MB,
        }
    }
}

/// Piece size for a cross-tier transfer: the smaller of the two endpoints'
/// optimal sizes, so neither side is fed pieces larger than it prefers.
pub fn cross_tier_piece_size(source: MediaClass, target: MediaClass) -> usize {
    source.optimal_piece_size().min(target.optimal_piece_size())
}

/// zstd compression effort for transfer. Transfer favours cheap compression
/// (the bytes are transient on the wire), so [`ZstdLevel::Fastest`] is the
/// default; cold-tier bulk moves can trade CPU for ratio with the higher levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZstdLevel {
    /// Lowest CPU, lowest ratio — the transfer default.
    #[default]
    Fastest,
    /// Balanced.
    Default,
    /// Higher ratio, more CPU.
    Better,
}

impl ZstdLevel {
    fn to_structured(self) -> structured_zstd::encoding::CompressionLevel {
        use structured_zstd::encoding::CompressionLevel;
        match self {
            ZstdLevel::Fastest => CompressionLevel::Fastest,
            ZstdLevel::Default => CompressionLevel::Default,
            ZstdLevel::Better => CompressionLevel::Better,
        }
    }

    /// Wire discriminant for the transfer header: 0 = fastest, 1 = default,
    /// 2 = better.
    pub fn to_wire(self) -> u32 {
        match self {
            ZstdLevel::Fastest => 0,
            ZstdLevel::Default => 1,
            ZstdLevel::Better => 2,
        }
    }

    /// Reconstruct from the wire discriminant.
    ///
    /// # Errors
    /// [`SwarmError::Source`] for an unknown discriminant.
    pub fn from_wire(value: u32) -> SwarmResult<Self> {
        match value {
            0 => Ok(ZstdLevel::Fastest),
            1 => Ok(ZstdLevel::Default),
            2 => Ok(ZstdLevel::Better),
            other => Err(SwarmError::Source(format!(
                "unknown zstd level discriminant {other}"
            ))),
        }
    }
}

/// How a segment's pieces are encoded on the wire. A segment uses one encoding
/// for all its pieces, chosen from its data kind (hot posting lists compress
/// well with LZ4; cold bulk data with zstd; already-compressed data uses `None`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PieceEncoding {
    /// Pieces transferred raw (no compression).
    None,
    /// Pieces compressed with LZ4 (fast; hot-data segments).
    Lz4,
    /// Pieces compressed with zstd at the given effort (cold-data segments).
    Zstd(ZstdLevel),
}

impl PieceEncoding {
    /// Encode a raw piece into its wire form.
    fn encode(self, raw: &[u8]) -> Vec<u8> {
        match self {
            PieceEncoding::None => raw.to_vec(),
            PieceEncoding::Lz4 => lz4_flex::compress_prepend_size(raw),
            PieceEncoding::Zstd(level) => {
                structured_zstd::encoding::compress_to_vec(raw, level.to_structured())
            }
        }
    }

    /// Stream-decode a wire piece directly into `sink`, holding no more than the
    /// codec's internal window in memory.
    fn decode_into<W: Write>(self, wire: &[u8], sink: &mut W) -> SwarmResult<()> {
        match self {
            PieceEncoding::None => sink
                .write_all(wire)
                .map_err(|e| SwarmError::Source(format!("write piece: {e}"))),
            PieceEncoding::Lz4 => {
                let raw = lz4_flex::decompress_size_prepended(wire)
                    .map_err(|e| SwarmError::Source(format!("lz4 decode: {e}")))?;
                sink.write_all(&raw)
                    .map_err(|e| SwarmError::Source(format!("write piece: {e}")))
            }
            PieceEncoding::Zstd(_) => {
                let mut decoder = structured_zstd::decoding::StreamingDecoder::new(wire)
                    .map_err(|e| SwarmError::Source(format!("zstd decoder: {e}")))?;
                io::copy(&mut decoder, sink)
                    .map(|_| ())
                    .map_err(|e| SwarmError::Source(format!("zstd decode: {e}")))
            }
        }
    }

    /// Wire discriminant pair `(encoding, zstd_level)` for the transfer header:
    /// encoding 0 = none, 1 = lz4, 2 = zstd; the level is 0 for non-zstd.
    pub fn to_wire(self) -> (u32, u32) {
        match self {
            PieceEncoding::None => (0, 0),
            PieceEncoding::Lz4 => (1, 0),
            PieceEncoding::Zstd(level) => (2, level.to_wire()),
        }
    }

    /// Reconstruct from the wire discriminant pair (see [`to_wire`](Self::to_wire)).
    ///
    /// # Errors
    /// [`SwarmError::Source`] for an unknown encoding or zstd-level discriminant.
    pub fn from_wire(encoding: u32, zstd_level: u32) -> SwarmResult<Self> {
        match encoding {
            0 => Ok(PieceEncoding::None),
            1 => Ok(PieceEncoding::Lz4),
            2 => Ok(PieceEncoding::Zstd(ZstdLevel::from_wire(zstd_level)?)),
            other => Err(SwarmError::Source(format!(
                "unknown encoding discriminant {other}"
            ))),
        }
    }
}

/// Errors from segment piece operations.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SwarmError {
    /// Requested piece size was zero — a segment cannot be chunked into pieces.
    #[error("piece size must be non-zero")]
    ZeroPieceSize,
    /// A piece index is outside the segment's piece count.
    #[error("piece index {index} out of range (segment has {count} pieces)")]
    PieceIndexOutOfRange {
        /// The offending index.
        index: PieceIndex,
        /// The segment's piece count.
        count: u32,
    },
    /// A received piece's checksum did not match the manifest.
    #[error("piece {index} checksum mismatch")]
    PieceHashMismatch {
        /// The corrupted piece's index.
        index: PieceIndex,
    },
    /// The assembled segment's checksum did not match the manifest, despite each
    /// piece verifying — indicates wrong piece ordering or a decode that
    /// produced different bytes.
    #[error("assembled segment checksum mismatch")]
    TotalHashMismatch,
    /// Assembly was given the wrong number of pieces.
    #[error("expected {expected} pieces, got {actual}")]
    PieceCountMismatch {
        /// Pieces the manifest describes.
        expected: u32,
        /// Pieces supplied.
        actual: u32,
    },
    /// A piece store or source failed to read or transfer a piece, or a wire
    /// piece could not be decoded (local I/O, transport, or codec error).
    #[error("piece source: {0}")]
    Source(String),
}

/// Convenience result alias for swarm operations.
pub type SwarmResult<T> = Result<T, SwarmError>;

/// Per-piece and whole-segment checksums describing how a segment is chunked for
/// swarm transfer. A receiver verifies each arriving wire piece against
/// `piece_hashes` and the fully assembled (decoded) segment against `total_hash`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentManifest {
    /// Raw (pre-encoding) size of each piece in bytes (the last piece may be
    /// shorter). Wire pieces may be smaller or larger after encoding.
    pub piece_size: usize,
    /// Total raw segment length in bytes.
    pub total_len: usize,
    /// How the wire pieces are encoded.
    pub encoding: PieceEncoding,
    /// xxh3-64 of each wire (encoded) piece, in order.
    pub piece_hashes: Vec<u64>,
    /// xxh3-64 of the whole raw segment.
    pub total_hash: u64,
}

impl SegmentManifest {
    /// Number of pieces the segment is split into.
    pub fn piece_count(&self) -> u32 {
        // A manifest is only ever built by `build_manifest`, which caps piece
        // count at the input length in bytes — far below u32::MAX in practice;
        // the conversion saturates rather than wraps on absurd inputs.
        u32::try_from(self.piece_hashes.len()).unwrap_or(u32::MAX)
    }

    /// Raw byte range `[start, end)` of piece `index` within the segment.
    ///
    /// # Errors
    /// [`SwarmError::PieceIndexOutOfRange`] if `index >= piece_count`.
    pub fn piece_range(&self, index: PieceIndex) -> SwarmResult<(usize, usize)> {
        let count = self.piece_count();
        if index >= count {
            return Err(SwarmError::PieceIndexOutOfRange { index, count });
        }
        let start = (index as usize) * self.piece_size;
        let end = (start + self.piece_size).min(self.total_len);
        Ok((start, end))
    }
}

/// Build a segment's [`SegmentManifest`] in a single streaming pass: encode each
/// raw chunk, checksum the wire bytes, and fold the raw bytes into the total
/// checksum — without materializing the encoded pieces. The source serves pieces
/// on demand via [`PieceEncoding::encode`] over [`SegmentManifest::piece_range`].
///
/// # Errors
/// [`SwarmError::ZeroPieceSize`] if `piece_size == 0`.
pub fn build_manifest(
    data: &[u8],
    piece_size: usize,
    encoding: PieceEncoding,
) -> SwarmResult<SegmentManifest> {
    if piece_size == 0 {
        return Err(SwarmError::ZeroPieceSize);
    }
    let mut piece_hashes = Vec::new();
    if !data.is_empty() {
        for raw in data.chunks(piece_size) {
            piece_hashes.push(xxh3_64(&encoding.encode(raw)));
        }
    }
    Ok(SegmentManifest {
        piece_size,
        total_len: data.len(),
        encoding,
        piece_hashes,
        total_hash: xxh3_64(data),
    })
}

/// Split a segment into encoded wire pieces plus its manifest. Materializes all
/// pieces — a convenience for tests and small segments; the transport streams
/// instead via [`build_manifest`] + on-demand [`PieceEncoding::encode`].
///
/// # Errors
/// [`SwarmError::ZeroPieceSize`] if `piece_size == 0`.
pub fn split_segment(
    data: &[u8],
    piece_size: usize,
    encoding: PieceEncoding,
) -> SwarmResult<(SegmentManifest, Vec<Vec<u8>>)> {
    let manifest = build_manifest(data, piece_size, encoding)?;
    let wire: Vec<Vec<u8>> = if data.is_empty() {
        Vec::new()
    } else {
        data.chunks(piece_size)
            .map(|raw| encoding.encode(raw))
            .collect()
    };
    Ok((manifest, wire))
}

/// Verify a received wire piece's bytes against the manifest's recorded checksum.
///
/// # Errors
/// [`SwarmError::PieceIndexOutOfRange`] if `index` is past the piece count;
/// [`SwarmError::PieceHashMismatch`] if the bytes do not match.
pub fn verify_piece(manifest: &SegmentManifest, index: PieceIndex, wire: &[u8]) -> SwarmResult<()> {
    let expected =
        manifest
            .piece_hashes
            .get(index as usize)
            .ok_or(SwarmError::PieceIndexOutOfRange {
                index,
                count: manifest.piece_count(),
            })?;
    if xxh3_64(wire) != *expected {
        return Err(SwarmError::PieceHashMismatch { index });
    }
    Ok(())
}

/// A [`Write`] that folds everything written into a running xxh3 before passing
/// it to the inner sink — lets the segment's total checksum be computed in the
/// same streaming pass that writes the decoded bytes out.
struct HashingWriter<'a, W: Write> {
    inner: &'a mut W,
    hasher: &'a mut Xxh3,
}

impl<W: Write> Write for HashingWriter<'_, W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.hasher.update(&buf[..n]);
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// Streaming segment assembler: feed wire pieces in index order with [`push`];
/// each is verified, stream-decoded into the sink, and folded into the running
/// total checksum. [`finish`] verifies the piece count and total checksum and
/// returns the sink. Holds only one piece at a time, never the whole segment.
///
/// [`push`]: SegmentWriter::push
/// [`finish`]: SegmentWriter::finish
pub struct SegmentWriter<'a, W: Write> {
    manifest: &'a SegmentManifest,
    sink: W,
    hasher: Xxh3,
    next: PieceIndex,
}

impl<'a, W: Write> SegmentWriter<'a, W> {
    /// Begin assembling `manifest`'s segment into `sink`.
    pub fn new(manifest: &'a SegmentManifest, sink: W) -> Self {
        Self {
            manifest,
            sink,
            hasher: Xxh3::new(),
            next: 0,
        }
    }

    /// Verify the next wire piece, stream-decode it into the sink, and fold its
    /// raw bytes into the running checksum.
    ///
    /// # Errors
    /// [`SwarmError::PieceIndexOutOfRange`] if more pieces are pushed than the
    /// manifest describes; [`SwarmError::PieceHashMismatch`] on corruption;
    /// [`SwarmError::Source`] on a decode or sink-write failure.
    pub fn push(&mut self, wire: &[u8]) -> SwarmResult<()> {
        verify_piece(self.manifest, self.next, wire)?;
        let mut hashing = HashingWriter {
            inner: &mut self.sink,
            hasher: &mut self.hasher,
        };
        self.manifest.encoding.decode_into(wire, &mut hashing)?;
        self.next += 1;
        Ok(())
    }

    /// Finish: verify all pieces were supplied and the total checksum matches,
    /// then return the sink.
    ///
    /// # Errors
    /// [`SwarmError::PieceCountMismatch`] if fewer pieces were pushed than
    /// expected; [`SwarmError::TotalHashMismatch`] if the decoded bytes are wrong.
    pub fn finish(self) -> SwarmResult<W> {
        let expected = self.manifest.piece_count();
        if self.next != expected {
            return Err(SwarmError::PieceCountMismatch {
                expected,
                actual: self.next,
            });
        }
        if self.hasher.digest() != self.manifest.total_hash {
            return Err(SwarmError::TotalHashMismatch);
        }
        Ok(self.sink)
    }
}

/// Assemble wire pieces (in index order) into the raw segment as a `Vec` — a
/// convenience over [`SegmentWriter`] for tests and small segments.
///
/// # Errors
/// Same as [`SegmentWriter::push`] / [`SegmentWriter::finish`].
pub fn assemble(manifest: &SegmentManifest, wire_pieces: &[Vec<u8>]) -> SwarmResult<Vec<u8>> {
    let actual = u32::try_from(wire_pieces.len()).unwrap_or(u32::MAX);
    if actual != manifest.piece_count() {
        return Err(SwarmError::PieceCountMismatch {
            expected: manifest.piece_count(),
            actual,
        });
    }
    let mut writer = SegmentWriter::new(manifest, Vec::with_capacity(manifest.total_len));
    for wire in wire_pieces {
        writer.push(wire)?;
    }
    writer.finish()
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
