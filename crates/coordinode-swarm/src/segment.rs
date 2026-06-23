//! Segment piece model: split a segment into checksummed pieces, verify each
//! piece on arrival, and assemble the whole with an end-to-end checksum.

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
    source.optimal_piece_size().min(target.optimal_piece_size())
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
    /// piece verifying — indicates wrong piece ordering or count.
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
}

/// Convenience result alias for swarm operations.
pub type SwarmResult<T> = Result<T, SwarmError>;

/// Per-piece and whole-segment checksums describing how a segment is chunked for
/// swarm transfer. A receiver verifies each arriving piece against
/// `piece_hashes` and the fully assembled segment against `total_hash`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentManifest {
    /// Size of each piece in bytes (the last piece may be shorter).
    pub piece_size: usize,
    /// Total segment length in bytes.
    pub total_len: usize,
    /// xxh3-64 of each piece, in order.
    pub piece_hashes: Vec<u64>,
    /// xxh3-64 of the whole segment.
    pub total_hash: u64,
}

impl SegmentManifest {
    /// Number of pieces the segment is split into.
    pub fn piece_count(&self) -> u32 {
        // A manifest is only ever built by `split_segment`, which caps piece
        // count at the input length in bytes — far below u32::MAX in practice;
        // the cast saturates rather than wraps to stay correct on absurd inputs.
        u32::try_from(self.piece_hashes.len()).unwrap_or(u32::MAX)
    }

    /// Byte range `[start, end)` of piece `index` within the segment.
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

/// Split a segment into `piece_size`-byte pieces and build its [`SegmentManifest`].
/// Returns the manifest plus borrowed slices of each piece (the last may be
/// shorter). The caller owns `data`, so no copy is made.
///
/// # Errors
/// [`SwarmError::ZeroPieceSize`] if `piece_size == 0`.
pub fn split_segment(data: &[u8], piece_size: usize) -> SwarmResult<(SegmentManifest, Vec<&[u8]>)> {
    if piece_size == 0 {
        return Err(SwarmError::ZeroPieceSize);
    }
    let pieces: Vec<&[u8]> = if data.is_empty() {
        Vec::new()
    } else {
        data.chunks(piece_size).collect()
    };
    let piece_hashes: Vec<u64> = pieces.iter().map(|p| xxh3_64(p)).collect();
    let manifest = SegmentManifest {
        piece_size,
        total_len: data.len(),
        piece_hashes,
        total_hash: xxh3_64(data),
    };
    Ok((manifest, pieces))
}

/// Verify a received piece's bytes against the manifest's recorded checksum.
///
/// # Errors
/// [`SwarmError::PieceIndexOutOfRange`] if `index` is past the piece count;
/// [`SwarmError::PieceHashMismatch`] if the bytes do not match.
pub fn verify_piece(
    manifest: &SegmentManifest,
    index: PieceIndex,
    bytes: &[u8],
) -> SwarmResult<()> {
    let expected =
        manifest
            .piece_hashes
            .get(index as usize)
            .ok_or(SwarmError::PieceIndexOutOfRange {
                index,
                count: manifest.piece_count(),
            })?;
    if xxh3_64(bytes) != *expected {
        return Err(SwarmError::PieceHashMismatch { index });
    }
    Ok(())
}

/// Assemble received pieces (in index order) into the full segment, verifying
/// every piece against the manifest and the assembled whole against `total_hash`.
///
/// # Errors
/// [`SwarmError::PieceCountMismatch`] if the wrong number of pieces is supplied;
/// [`SwarmError::PieceHashMismatch`] if any piece is corrupt;
/// [`SwarmError::TotalHashMismatch`] if the reassembled bytes are wrong despite
/// each piece verifying (e.g. misordered pieces).
pub fn assemble(manifest: &SegmentManifest, pieces: &[Vec<u8>]) -> SwarmResult<Vec<u8>> {
    let expected = manifest.piece_count();
    let actual = u32::try_from(pieces.len()).unwrap_or(u32::MAX);
    if actual != expected {
        return Err(SwarmError::PieceCountMismatch { expected, actual });
    }
    let mut out = Vec::with_capacity(manifest.total_len);
    for (i, piece) in pieces.iter().enumerate() {
        // The cast is bounded by `expected` (checked above) ≤ u32::MAX.
        verify_piece(manifest, i as PieceIndex, piece)?;
        out.extend_from_slice(piece);
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
        (0..len).map(|i| (i % 251) as u8).collect()
    }

    fn owned(pieces: &[&[u8]]) -> Vec<Vec<u8>> {
        pieces.iter().map(|p| p.to_vec()).collect()
    }

    #[test]
    fn media_class_piece_sizes_match_spec() {
        assert_eq!(MediaClass::Ram.optimal_piece_size(), 4 * 1024 * 1024);
        assert_eq!(MediaClass::Nvme.optimal_piece_size(), 1024 * 1024);
        assert_eq!(MediaClass::Ssd.optimal_piece_size(), 1024 * 1024);
        assert_eq!(MediaClass::Hdd.optimal_piece_size(), 4 * 1024 * 1024);
        // Cross-tier takes the smaller side.
        assert_eq!(
            cross_tier_piece_size(MediaClass::Hdd, MediaClass::Nvme),
            1024 * 1024
        );
    }

    #[test]
    fn split_then_assemble_round_trips() {
        // 48 KB into 1 KB pieces: an exact multiple, then a ragged tail.
        for len in [48 * 1024, 48 * 1024 + 7, 1, 1023] {
            let data = segment(len);
            let (manifest, pieces) = split_segment(&data, 1024).expect("split");
            let restored = assemble(&manifest, &owned(&pieces)).expect("assemble");
            assert_eq!(restored, data, "round-trip for len={len}");
        }
    }

    #[test]
    fn manifest_piece_count_and_ranges() {
        let data = segment(2500);
        let (manifest, pieces) = split_segment(&data, 1000).expect("split");
        assert_eq!(manifest.piece_count(), 3);
        assert_eq!(pieces.len(), 3);
        assert_eq!(manifest.piece_range(0).expect("r0"), (0, 1000));
        assert_eq!(manifest.piece_range(2).expect("r2"), (2000, 2500)); // ragged tail
        assert!(manifest.piece_range(3).is_err(), "out of range");
    }

    #[test]
    fn verify_piece_detects_corruption() {
        let data = segment(4096);
        let (manifest, pieces) = split_segment(&data, 1024).expect("split");
        // Clean piece verifies.
        verify_piece(&manifest, 1, pieces[1]).expect("clean");
        // Flip a byte → mismatch.
        let mut bad = pieces[1].to_vec();
        bad[0] ^= 0xFF;
        assert_eq!(
            verify_piece(&manifest, 1, &bad),
            Err(SwarmError::PieceHashMismatch { index: 1 })
        );
        // Out-of-range index.
        assert!(matches!(
            verify_piece(&manifest, 99, pieces[0]),
            Err(SwarmError::PieceIndexOutOfRange { index: 99, .. })
        ));
    }

    #[test]
    fn assemble_rejects_corruption_wrong_count_and_misorder() {
        let data = segment(4096);
        let (manifest, pieces) = split_segment(&data, 1024).expect("split");
        let mut owned_pieces = owned(&pieces);

        // Wrong count.
        assert!(matches!(
            assemble(&manifest, &owned_pieces[..3]),
            Err(SwarmError::PieceCountMismatch {
                expected: 4,
                actual: 3
            })
        ));

        // Corrupt one piece.
        let mut corrupt = owned_pieces.clone();
        corrupt[2][5] ^= 0xFF;
        assert_eq!(
            assemble(&manifest, &corrupt),
            Err(SwarmError::PieceHashMismatch { index: 2 })
        );

        // Misordered pieces: each piece is individually valid for SOME index,
        // but swapping two equal-length pieces yields the wrong whole. Swap
        // pieces 0 and 1 (both full 1 KB) — their hashes differ, so this trips
        // the per-piece check first; that is correct (corruption caught early).
        owned_pieces.swap(0, 1);
        assert!(assemble(&manifest, &owned_pieces).is_err());
    }

    #[test]
    fn empty_segment_has_no_pieces() {
        let (manifest, pieces) = split_segment(&[], 1024).expect("split empty");
        assert_eq!(manifest.piece_count(), 0);
        assert!(pieces.is_empty());
        assert_eq!(
            assemble(&manifest, &[]).expect("assemble empty"),
            Vec::<u8>::new()
        );
    }

    #[test]
    fn zero_piece_size_rejected() {
        assert_eq!(
            split_segment(&segment(10), 0),
            Err(SwarmError::ZeroPieceSize)
        );
    }
}
