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
mod tests {
    use super::*;
    use crate::state::PieceBitfield;

    fn candidate(id: u64) -> SourceCandidate {
        SourceCandidate {
            node: NodeId(id),
            utilization: 0.0,
            bandwidth_to_target: 1.0,
            same_rack: false,
            tit_for_tat: 1.0,
            freshness: Freshness::Verified,
        }
    }

    #[test]
    fn score_rewards_idle_fast_local_fresh_generous() {
        let base = candidate(0);
        let busy = SourceCandidate {
            utilization: 0.9,
            ..base
        };
        let local = SourceCandidate {
            same_rack: true,
            ..base
        };
        let in_flight = SourceCandidate {
            freshness: Freshness::InFlight,
            ..base
        };
        let generous = SourceCandidate {
            tit_for_tat: 3.0,
            ..base
        };
        assert!(base.score() > busy.score(), "idle beats busy");
        assert!(
            (local.score() - base.score() * 10.0).abs() < 1e-9,
            "same-rack 10x"
        );
        assert!(in_flight.score() < base.score(), "verified beats in-flight");
        assert!(generous.score() > base.score(), "uploader rewarded");
        // A saturated source contributes zero, not a negative score.
        let saturated = SourceCandidate {
            utilization: 1.5,
            ..base
        };
        assert_eq!(saturated.score(), 0.0);
    }

    #[test]
    fn select_source_picks_highest_score_holder() {
        // 1 piece. Three peers hold it; node 2 is same-rack (10x) → wins.
        let mut st = SwarmState::new(1);
        for id in [1u64, 2, 3] {
            st.set_peer_bitfield(NodeId(id), PieceBitfield::full(1));
        }
        let cands = [
            candidate(1),
            SourceCandidate {
                same_rack: true,
                ..candidate(2)
            },
            candidate(3),
        ];
        assert_eq!(select_source(&st, 0, &cands), Some(NodeId(2)));
    }

    #[test]
    fn select_source_skips_non_holders_and_ties_low_id() {
        let mut st = SwarmState::new(1);
        // Only nodes 2 and 3 hold the piece; node 1 (best stats) does NOT.
        st.set_peer_bitfield(NodeId(2), PieceBitfield::full(1));
        st.set_peer_bitfield(NodeId(3), PieceBitfield::full(1));
        let cands = [candidate(1), candidate(2), candidate(3)];
        // 1 is skipped (no copy); 2 and 3 tie → lowest id 2.
        assert_eq!(select_source(&st, 0, &cands), Some(NodeId(2)));
    }

    #[test]
    fn select_source_none_when_no_holder() {
        let st = SwarmState::new(2);
        assert_eq!(select_source(&st, 0, &[candidate(1)]), None);
    }
}
