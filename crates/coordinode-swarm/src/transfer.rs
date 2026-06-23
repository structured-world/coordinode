//! Single-source segment transfer: stream a segment's wire pieces from a
//! [`PieceStore`] through a [`SegmentWriter`](crate::SegmentWriter) into a sink,
//! verifying and decoding each piece as it arrives.
//!
//! This is the transport-agnostic core. The gRPC transport (the
//! `TransferPieces` streaming RPC) splits the same flow across the wire: the
//! source streams pieces, the target assembles them with a `SegmentWriter`. It
//! is the bulk-move primitive for replication repair and node resync (and,
//! per-shard, for migration).

use std::collections::HashMap;
use std::io::Write;

use crate::segment::{
    split_segment, PieceEncoding, PieceIndex, SegmentManifest, SegmentWriter, SwarmError,
    SwarmResult,
};

/// Identity of a transferable segment. Matches the placement-layer segment id
/// (`SegmentDescriptor.id`), so a transfer addresses the same segment the
/// segment map and CRUSH placement refer to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SegmentId(pub u64);

/// Source side of a transfer: serves a segment's manifest and its wire (encoded)
/// pieces. A production store reads pieces from local storage and encodes them
/// per the segment's policy; the transport streams them to a target.
pub trait PieceStore {
    /// The segment's transfer manifest (piece checksums + encoding).
    ///
    /// # Errors
    /// Implementation-defined; typically [`SwarmError::Source`] if the segment is
    /// unknown or unreadable.
    fn manifest(&self, segment: SegmentId) -> SwarmResult<SegmentManifest>;

    /// Wire bytes of piece `index` (encoded per the manifest).
    ///
    /// # Errors
    /// [`SwarmError::Source`] if the segment or piece is unavailable.
    fn wire_piece(&self, segment: SegmentId, index: PieceIndex) -> SwarmResult<Vec<u8>>;
}

/// Transfer a whole segment from `store` into `sink`: fetch the manifest, then
/// stream each wire piece through a [`SegmentWriter`] (verify checksum, decode,
/// fold into the running total checksum). Returns the sink on success.
///
/// # Errors
/// Any [`PieceStore`] error, or a verify/decode/total-checksum failure from the
/// writer (see [`SegmentWriter::push`](crate::SegmentWriter::push) /
/// [`finish`](crate::SegmentWriter::finish)).
pub fn transfer<W: Write>(store: &dyn PieceStore, segment: SegmentId, sink: W) -> SwarmResult<W> {
    let manifest = store.manifest(segment)?;
    let mut writer = SegmentWriter::new(&manifest, sink);
    for index in 0..manifest.piece_count() {
        writer.push(&store.wire_piece(segment, index)?)?;
    }
    writer.finish()
}

/// In-memory [`PieceStore`] holding pre-split wire pieces. The base for a
/// storage-backed source and the test/double for the transport; segments are
/// added with [`insert`](LocalPieceStore::insert).
#[derive(Debug, Default)]
pub struct LocalPieceStore {
    segments: HashMap<SegmentId, (SegmentManifest, Vec<Vec<u8>>)>,
}

impl LocalPieceStore {
    /// Empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Split `data` into `piece_size` pieces under `encoding` and hold them under
    /// `segment`, ready to serve.
    ///
    /// # Errors
    /// [`SwarmError::ZeroPieceSize`] if `piece_size == 0`.
    pub fn insert(
        &mut self,
        segment: SegmentId,
        data: &[u8],
        piece_size: usize,
        encoding: PieceEncoding,
    ) -> SwarmResult<()> {
        let (manifest, wire) = split_segment(data, piece_size, encoding)?;
        self.segments.insert(segment, (manifest, wire));
        Ok(())
    }

    fn entry(&self, segment: SegmentId) -> SwarmResult<&(SegmentManifest, Vec<Vec<u8>>)> {
        self.segments
            .get(&segment)
            .ok_or_else(|| SwarmError::Source(format!("unknown segment {}", segment.0)))
    }
}

impl PieceStore for LocalPieceStore {
    fn manifest(&self, segment: SegmentId) -> SwarmResult<SegmentManifest> {
        Ok(self.entry(segment)?.0.clone())
    }

    fn wire_piece(&self, segment: SegmentId, index: PieceIndex) -> SwarmResult<Vec<u8>> {
        let (manifest, wire) = self.entry(segment)?;
        wire.get(index as usize).cloned().ok_or_else(|| {
            SwarmError::Source(format!(
                "segment {} piece {index} out of range ({} pieces)",
                segment.0,
                manifest.piece_count()
            ))
        })
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::segment::ZstdLevel;

    fn segment(len: usize) -> Vec<u8> {
        (0..len).map(|i| ((i / 5) % 13) as u8).collect()
    }

    #[test]
    fn transfer_reconstructs_segment_for_each_encoding() {
        for enc in [
            PieceEncoding::None,
            PieceEncoding::Lz4,
            PieceEncoding::Zstd(ZstdLevel::Fastest),
        ] {
            let data = segment(20 * 1024 + 3);
            let mut store = LocalPieceStore::new();
            store
                .insert(SegmentId(42), &data, 1024, enc)
                .expect("insert");

            let out = transfer(&store, SegmentId(42), Vec::new()).expect("transfer");
            assert_eq!(out, data, "transfer round-trip enc={enc:?}");
        }
    }

    #[test]
    fn transfer_unknown_segment_errors() {
        let store = LocalPieceStore::new();
        assert!(transfer(&store, SegmentId(7), Vec::new()).is_err());
    }

    #[test]
    fn transfer_detects_a_corrupting_store() {
        // A store whose wire_piece flips a byte must be caught by the per-piece
        // checksum — repair never silently writes corrupt data.
        struct Corrupting(LocalPieceStore);
        impl PieceStore for Corrupting {
            fn manifest(&self, s: SegmentId) -> SwarmResult<SegmentManifest> {
                self.0.manifest(s)
            }
            fn wire_piece(&self, s: SegmentId, i: PieceIndex) -> SwarmResult<Vec<u8>> {
                let mut p = self.0.wire_piece(s, i)?;
                if i == 1 {
                    p[0] ^= 0xFF;
                }
                Ok(p)
            }
        }
        let data = segment(4096);
        let mut base = LocalPieceStore::new();
        base.insert(SegmentId(1), &data, 1024, PieceEncoding::None)
            .expect("insert");
        let store = Corrupting(base);
        assert!(matches!(
            transfer(&store, SegmentId(1), Vec::new()),
            Err(SwarmError::PieceHashMismatch { index: 1 })
        ));
    }
}
