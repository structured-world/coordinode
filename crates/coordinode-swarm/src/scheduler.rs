//! Source selection for swarm transfer: score the candidate sources that hold a
//! piece and pick the best, so a target pulls from the idlest, fastest, closest
//! peer — and rewards peers that upload to it (tit-for-tat), which starves
//! leechers that only download.
//!
//! Pairs with [`SwarmState::select_next_piece`](crate::SwarmState::select_next_piece)
//! (rarest-first: *which* piece to fetch) — this answers *from whom*.

use crate::segment::PieceIndex;
use crate::state::{NodeId, SwarmState};

/// Freshness of a piece on a candidate source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Freshness {
    /// The source holds a verified copy.
    Verified,
    /// The source is still receiving the piece (it can forward it shortly, but a
    /// verified holder is preferred).
    InFlight,
}

impl Freshness {
    fn factor(self) -> f64 {
        match self {
            Freshness::Verified => 1.0,
            Freshness::InFlight => 0.8,
        }
    }
}

/// A candidate source for one piece, scored from the requesting target's view.
#[derive(Debug, Clone, Copy)]
pub struct SourceCandidate {
    /// The candidate peer.
    pub node: NodeId,
    /// Source load: 0.0 (idle) .. 1.0 (saturated). Idle nodes score higher.
    pub utilization: f64,
    /// Throughput from this source to the target (normalized; higher is faster).
    pub bandwidth_to_target: f64,
    /// Whether the source shares the target's rack / failure domain.
    pub same_rack: bool,
    /// Tit-for-tat reward (>= 1.0): how much this source has uploaded to the
    /// target. Generous uploaders win ties and edge out leechers.
    pub tit_for_tat: f64,
    /// Whether the source's copy is verified or still in flight.
    pub freshness: Freshness,
}

impl SourceCandidate {
    /// Composite source-selection score (higher is better):
    /// `(1 - utilization) * bandwidth * locality * freshness * tit_for_tat`,
    /// where locality is a 10x bonus for a same-rack source. Utilization is
    /// clamped so a saturated source contributes zero rather than going negative.
    pub fn score(&self) -> f64 {
        let locality = if self.same_rack { 10.0 } else { 1.0 };
        (1.0 - self.utilization).max(0.0)
            * self.bandwidth_to_target
            * locality
            * self.freshness.factor()
            * self.tit_for_tat
    }
}

/// Pick the best source for `piece`: among `candidates` that actually hold it
/// (per `state`), the highest [`score`](SourceCandidate::score), ties broken by
/// lowest node id for determinism. `None` if no candidate holds the piece.
pub fn select_source(
    state: &SwarmState,
    piece: PieceIndex,
    candidates: &[SourceCandidate],
) -> Option<NodeId> {
    let mut best: Option<(f64, NodeId)> = None;
    for c in candidates.iter().filter(|c| state.peer_has(c.node, piece)) {
        let s = c.score();
        let better = match best {
            None => true,
            // Higher score wins; on a tie the lower node id wins (deterministic).
            Some((bs, bn)) => s > bs || (s == bs && c.node < bn),
        };
        if better {
            best = Some((s, c.node));
        }
    }
    best.map(|(_, node)| node)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
