//! Segment piece model: split a segment into per-piece-encoded, checksummed
//! pieces, verify each piece on arrival, and assemble the whole with an
//! end-to-end checksum.
//!
//! Pieces may be transferred raw or compressed ([`PieceEncoding`]); the manifest
//! records the encoding. Piece checksums are taken over the **wire** (encoded)
//! bytes so a receiver verifies what it received before spending CPU to decode,
//! while the total checksum is over the **raw** segment so the fully decoded
//! result is verified end to end.

use xxhash_rust::xxh3::xxh3_64;

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
    source
        .optimal_piece_size()
        .min(target.optimal_piece_size())
}

/// How a segment's pieces are encoded on the wire. A segment uses one encoding
/// for all its pieces, chosen from its data kind (hot posting lists compress
/// well with LZ4; already-compressed data uses `None`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PieceEncoding {
    /// Pieces transferred raw (no compression).
    None,
    /// Pieces compressed with LZ4 (fast; hot-data segments).
    Lz4,
}

impl PieceEncoding {
    /// Encode a raw piece into its wire form.
    fn encode(self, raw: &[u8]) -> Vec<u8> {
        match self {
            PieceEncoding::None => raw.to_vec(),
            PieceEncoding::Lz4 => lz4_flex::compress_prepend_size(raw),
        }
    }

    /// Decode a wire piece back to its raw form.
    fn decode(self, wire: &[u8]) -> SwarmResult<Vec<u8>> {
        match self {
            PieceEncoding::None => Ok(wire.to_vec()),
            PieceEncoding::Lz4 => lz4_flex::decompress_size_prepended(wire)
                .map_err(|e| SwarmError::Source(format!("lz4 decode: {e}"))),
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
        // A manifest is only ever built by `split_segment`, which caps piece
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

/// Split a segment into `piece_size`-byte raw pieces, encode each per `encoding`,
/// and build its [`SegmentManifest`]. Returns the manifest plus the owned **wire**
/// (encoded) pieces ready to transfer.
///
/// # Errors
/// [`SwarmError::ZeroPieceSize`] if `piece_size == 0`.
pub fn split_segment(
    data: &[u8],
    piece_size: usize,
    encoding: PieceEncoding,
) -> SwarmResult<(SegmentManifest, Vec<Vec<u8>>)> {
    if piece_size == 0 {
        return Err(SwarmError::ZeroPieceSize);
    }
    let wire: Vec<Vec<u8>> = if data.is_empty() {
        Vec::new()
    } else {
        data.chunks(piece_size).map(|raw| encoding.encode(raw)).collect()
    };
    let piece_hashes: Vec<u64> = wire.iter().map(|p| xxh3_64(p)).collect();
    let manifest = SegmentManifest {
        piece_size,
        total_len: data.len(),
        encoding,
        piece_hashes,
        total_hash: xxh3_64(data),
    };
    Ok((manifest, wire))
}

/// Verify a received wire piece's bytes against the manifest's recorded checksum.
///
/// # Errors
/// [`SwarmError::PieceIndexOutOfRange`] if `index` is past the piece count;
/// [`SwarmError::PieceHashMismatch`] if the bytes do not match.
pub fn verify_piece(manifest: &SegmentManifest, index: PieceIndex, wire: &[u8]) -> SwarmResult<()> {
    let expected = manifest
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

/// Assemble received wire pieces (in index order) into the full raw segment:
/// verify every piece against the manifest, decode it, concatenate, and verify
/// the assembled whole against `total_hash`.
///
/// # Errors
/// [`SwarmError::PieceCountMismatch`] if the wrong number of pieces is supplied;
/// [`SwarmError::PieceHashMismatch`] if any wire piece is corrupt;
/// [`SwarmError::Source`] if a piece fails to decode;
/// [`SwarmError::TotalHashMismatch`] if the reassembled bytes are wrong despite
/// each piece verifying (e.g. misordered pieces).
pub fn assemble(manifest: &SegmentManifest, wire_pieces: &[Vec<u8>]) -> SwarmResult<Vec<u8>> {
    let expected = manifest.piece_count();
    let actual = u32::try_from(wire_pieces.len()).unwrap_or(u32::MAX);
    if actual != expected {
        return Err(SwarmError::PieceCountMismatch { expected, actual });
    }
    let mut out = Vec::with_capacity(manifest.total_len);
    for (i, wire) in wire_pieces.iter().enumerate() {
        // The conversion is bounded by `expected` (checked above) ≤ u32::MAX.
        verify_piece(manifest, i as PieceIndex, wire)?;
        out.extend_from_slice(&manifest.encoding.decode(wire)?);
    }
    if xxh3_64(&out) != manifest.total_hash {
        return Err(SwarmError::TotalHashMismatch);
    }
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn segment(len: usize) -> Vec<u8> {
        // Repetitive enough that LZ4 actually compresses (exercises the codec).
        (0..len).map(|i| ((i / 7) % 17) as u8).collect()
    }

    const ENCODINGS: [PieceEncoding; 2] = [PieceEncoding::None, PieceEncoding::Lz4];

    #[test]
    fn media_class_piece_sizes_match_spec() {
        assert_eq!(MediaClass::Ram.optimal_piece_size(), 4 * 1024 * 1024);
        assert_eq!(MediaClass::Nvme.optimal_piece_size(), 1024 * 1024);
        assert_eq!(MediaClass::Ssd.optimal_piece_size(), 1024 * 1024);
        assert_eq!(MediaClass::Hdd.optimal_piece_size(), 4 * 1024 * 1024);
        assert_eq!(
            cross_tier_piece_size(MediaClass::Hdd, MediaClass::Nvme),
            1024 * 1024
        );
    }

    #[test]
    fn split_then_assemble_round_trips_each_encoding() {
        for enc in ENCODINGS {
            for len in [48 * 1024, 48 * 1024 + 7, 1, 1023] {
                let data = segment(len);
                let (manifest, wire) = split_segment(&data, 1024, enc).expect("split");
                assert_eq!(manifest.encoding, enc);
                let restored = assemble(&manifest, &wire).expect("assemble");
                assert_eq!(restored, data, "round-trip enc={enc:?} len={len}");
            }
        }
    }

    #[test]
    fn lz4_pieces_are_smaller_than_raw_for_compressible_data() {
        let data = segment(64 * 1024);
        let (_, raw) = split_segment(&data, 64 * 1024, PieceEncoding::None).expect("raw");
        let (_, lz4) = split_segment(&data, 64 * 1024, PieceEncoding::Lz4).expect("lz4");
        assert!(
            lz4[0].len() < raw[0].len(),
            "lz4 wire piece ({}) should compress below raw ({})",
            lz4[0].len(),
            raw[0].len()
        );
    }

    #[test]
    fn manifest_piece_count_and_ranges() {
        let data = segment(2500);
        let (manifest, wire) = split_segment(&data, 1000, PieceEncoding::None).expect("split");
        assert_eq!(manifest.piece_count(), 3);
        assert_eq!(wire.len(), 3);
        assert_eq!(manifest.piece_range(0).expect("r0"), (0, 1000));
        assert_eq!(manifest.piece_range(2).expect("r2"), (2000, 2500)); // ragged tail
        assert!(manifest.piece_range(3).is_err(), "out of range");
    }

    #[test]
    fn verify_piece_detects_corruption() {
        let data = segment(4096);
        let (manifest, wire) = split_segment(&data, 1024, PieceEncoding::Lz4).expect("split");
        verify_piece(&manifest, 1, &wire[1]).expect("clean");
        let mut bad = wire[1].clone();
        bad[0] ^= 0xFF;
        assert_eq!(
            verify_piece(&manifest, 1, &bad),
            Err(SwarmError::PieceHashMismatch { index: 1 })
        );
        assert!(matches!(
            verify_piece(&manifest, 99, &wire[0]),
            Err(SwarmError::PieceIndexOutOfRange { index: 99, .. })
        ));
    }

    #[test]
    fn assemble_rejects_corruption_wrong_count_and_misorder() {
        let data = segment(4096);
        let (manifest, wire) = split_segment(&data, 1024, PieceEncoding::None).expect("split");

        assert!(matches!(
            assemble(&manifest, &wire[..3]),
            Err(SwarmError::PieceCountMismatch {
                expected: 4,
                actual: 3
            })
        ));

        let mut corrupt = wire.clone();
        corrupt[2][5] ^= 0xFF;
        assert_eq!(
            assemble(&manifest, &corrupt),
            Err(SwarmError::PieceHashMismatch { index: 2 })
        );

        let mut misordered = wire.clone();
        misordered.swap(0, 1);
        assert!(assemble(&manifest, &misordered).is_err());
    }

    #[test]
    fn empty_segment_has_no_pieces() {
        for enc in ENCODINGS {
            let (manifest, wire) = split_segment(&[], 1024, enc).expect("split empty");
            assert_eq!(manifest.piece_count(), 0);
            assert!(wire.is_empty());
            assert_eq!(
                assemble(&manifest, &[]).expect("assemble empty"),
                Vec::<u8>::new()
            );
        }
    }

    #[test]
    fn zero_piece_size_rejected() {
        assert_eq!(
            split_segment(&segment(10), 0, PieceEncoding::None),
            Err(SwarmError::ZeroPieceSize)
        );
    }
}
