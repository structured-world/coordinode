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
mod tests;
