//! Multi-source swarm download: fetch a segment's pieces from several peers at
//! once, rarest-first, choosing the best source per piece.
//!
//! This is the receiver side of the BitTorrent-inspired transfer (ADR-005): a
//! node that needs a segment (replication repair, resync, the target of a
//! migration) pulls each piece from whichever peer scores best for it, verifies
//! the piece on arrival, and — as pieces verify — becomes a holder itself, so in
//! a multi-target swarm its availability rises and it can serve others (O(log N)
//! fanout rather than a single-source bottleneck).
//!
//! It is transport-agnostic: a [`PieceSource`] abstracts one peer's
//! piece-exchange endpoint, so the rarest-first / best-source loop is pure logic
//! and unit-testable without a network. The gRPC piece-exchange wires concrete
//! `PieceSource`s over the wire.

use std::collections::HashMap;
use std::io::Write;

use crate::scheduler::{select_source, SourceCandidate};
use crate::segment::{
    assemble, verify_piece, PieceIndex, SegmentManifest, SwarmError, SwarmResult,
};
use crate::state::{NodeId, PieceBitfield, SwarmState};

/// One peer that can serve verified wire pieces of a segment, plus the scoring
/// metadata used to choose among sources. Abstracts the gRPC piece-exchange so
/// the download loop stays pure logic.
pub trait PieceSource {
    /// This source's peer id within the swarm.
    fn node(&self) -> NodeId;

    /// The pieces this source advertises holding (its bitfield).
    fn bitfield(&self) -> PieceBitfield;

    /// Source-selection metadata (utilization, bandwidth, locality, tit-for-tat,
    /// freshness) from the requesting node's view.
    fn candidate(&self) -> SourceCandidate;

    /// Fetch the wire (encoded) bytes of piece `index` from this source.
    ///
    /// # Errors
    /// [`SwarmError::Source`] (or a transport error mapped onto it) if the piece
    /// is unavailable or the fetch fails.
    fn fetch_piece(&self, index: PieceIndex) -> SwarmResult<Vec<u8>>;
}

/// Download a whole segment from `sources` into `sink`, rarest-first.
///
/// `me` is the local node id (it starts holding no pieces and is marked a holder
/// as each piece verifies). Each iteration picks the rarest piece `me` still
/// needs ([`SwarmState::select_next_piece`]), the best source that holds it
/// ([`select_source`]), fetches and verifies that piece, then records it. When
/// every piece is held the wire pieces are assembled (and whole-segment
/// verified) and written to `sink`.
///
/// # Errors
/// [`SwarmError`] if no source holds a needed piece, a fetched piece fails its
/// checksum, assembly / whole-segment verification fails, or the sink write
/// fails.
pub fn swarm_download<W: Write>(
    me: NodeId,
    manifest: &SegmentManifest,
    sources: &[&dyn PieceSource],
    mut sink: W,
) -> SwarmResult<W> {
    let piece_count = manifest.piece_count();
    let mut state = SwarmState::new(piece_count);
    for src in sources {
        state.set_peer_bitfield(src.node(), src.bitfield());
    }

    let candidates: Vec<SourceCandidate> = sources.iter().map(|s| s.candidate()).collect();
    let by_node: HashMap<NodeId, &dyn PieceSource> =
        sources.iter().map(|s| (s.node(), *s)).collect();

    // Collect wire pieces out of order; assemble in index order at the end.
    let mut wire: Vec<Vec<u8>> = vec![Vec::new(); piece_count as usize];

    // `select_next_piece` returns None once `me` holds every piece. Each loop
    // marks one new piece for `me`, so this terminates in `piece_count` steps.
    while let Some(idx) = state.select_next_piece(me) {
        let source = select_source(&state, idx, &candidates).ok_or_else(|| {
            SwarmError::Source(format!("no source holds piece {idx} (segment incomplete)"))
        })?;
        let src = by_node
            .get(&source)
            .ok_or_else(|| SwarmError::Source(format!("selected unknown source {}", source.0)))?;

        let bytes = src.fetch_piece(idx)?;
        verify_piece(manifest, idx, &bytes)?;
        if let Some(slot) = wire.get_mut(idx as usize) {
            *slot = bytes;
        }
        state.mark_piece(me, idx);
    }

    let segment = assemble(manifest, &wire)?;
    sink.write_all(&segment)
        .map_err(|e| SwarmError::Source(format!("sink write: {e}")))?;
    Ok(sink)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod download_tests;
