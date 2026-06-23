//! Swarm scheduling state: which peers hold which pieces, and rarest-first piece
//! selection. Pure logic over piece indices and node ids; no I/O.

use std::collections::HashMap;

use crate::segment::PieceIndex;

/// Cluster node identity within a swarm transfer. Distinct from a graph node id;
/// this is a peer in the transfer mesh. The transport maps the cluster's own
/// node identity onto this when wiring up a swarm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u64);

/// A packed bitfield over a segment's pieces: bit `i` set means "this peer holds
/// piece `i`". Backed by `u64` words.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PieceBitfield {
    words: Vec<u64>,
    len: u32,
}

impl PieceBitfield {
    /// An all-zero bitfield for a segment of `piece_count` pieces.
    pub fn new(piece_count: u32) -> Self {
        let words = (piece_count as usize).div_ceil(64);
        Self {
            words: vec![0; words],
            len: piece_count,
        }
    }

    /// A full bitfield (every piece present) — the source's view of a segment.
    pub fn full(piece_count: u32) -> Self {
        let mut bf = Self::new(piece_count);
        for i in 0..piece_count {
            bf.set(i);
        }
        bf
    }

    /// Number of pieces this bitfield is sized for.
    pub fn len(&self) -> u32 {
        self.len
    }

    /// Whether the bitfield covers zero pieces.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Mark piece `idx` as present. Out-of-range indices are ignored.
    pub fn set(&mut self, idx: PieceIndex) {
        if idx >= self.len {
            return;
        }
        let (w, b) = (idx as usize / 64, idx as usize % 64);
        self.words[w] |= 1u64 << b;
    }

    /// Whether piece `idx` is present.
    pub fn has(&self, idx: PieceIndex) -> bool {
        if idx >= self.len {
            return false;
        }
        let (w, b) = (idx as usize / 64, idx as usize % 64);
        self.words[w] & (1u64 << b) != 0
    }

    /// How many pieces are present.
    pub fn count_set(&self) -> u32 {
        // Each word's popcount; total is bounded by `len` ≤ u32::MAX.
        self.words.iter().map(|w| w.count_ones()).sum()
    }

    /// Whether every piece is present.
    pub fn is_complete(&self) -> bool {
        self.count_set() == self.len
    }
}

/// Per-segment swarm scheduling state: every peer's piece bitfield plus the set
/// of in-flight transfers, enough to drive rarest-first piece selection.
#[derive(Debug, Clone)]
pub struct SwarmState {
    piece_count: u32,
    /// Which pieces each peer currently holds.
    peer_bitfields: HashMap<NodeId, PieceBitfield>,
    /// Pieces currently being transferred: `piece -> (source, target)`.
    in_flight: HashMap<PieceIndex, (NodeId, NodeId)>,
}

impl SwarmState {
    /// Empty swarm for a segment of `piece_count` pieces (no peers yet).
    pub fn new(piece_count: u32) -> Self {
        Self {
            piece_count,
            peer_bitfields: HashMap::new(),
            in_flight: HashMap::new(),
        }
    }

    /// Pieces in the segment.
    pub fn piece_count(&self) -> u32 {
        self.piece_count
    }

    /// Register or replace a peer's bitfield (e.g. on join or a bitfield gossip).
    pub fn set_peer_bitfield(&mut self, node: NodeId, bitfield: PieceBitfield) {
        self.peer_bitfields.insert(node, bitfield);
    }

    /// Record that `node` now holds piece `idx` (after a verified transfer).
    /// Creates the peer's bitfield if unseen.
    pub fn mark_piece(&mut self, node: NodeId, idx: PieceIndex) {
        self.peer_bitfields
            .entry(node)
            .or_insert_with(|| PieceBitfield::new(self.piece_count))
            .set(idx);
    }

    /// Whether `node` currently holds piece `idx`.
    pub fn peer_has(&self, node: NodeId, idx: PieceIndex) -> bool {
        self.peer_bitfields.get(&node).is_some_and(|bf| bf.has(idx))
    }

    /// How many peers currently hold piece `idx`.
    pub fn availability(&self, idx: PieceIndex) -> usize {
        self.peer_bitfields
            .values()
            .filter(|bf| bf.has(idx))
            .count()
    }

    /// Mark a piece transfer in flight (`source -> target`).
    pub fn mark_in_flight(&mut self, idx: PieceIndex, source: NodeId, target: NodeId) {
        self.in_flight.insert(idx, (source, target));
    }

    /// Clear an in-flight transfer (on completion or abort).
    pub fn complete_in_flight(&mut self, idx: PieceIndex) {
        self.in_flight.remove(&idx);
    }

    /// Whether piece `idx` is currently being fetched by `target`.
    fn in_flight_to(&self, idx: PieceIndex, target: NodeId) -> bool {
        matches!(self.in_flight.get(&idx), Some((_, t)) if *t == target)
    }

    /// Rarest-first piece selection: among the pieces `for_node` still needs
    /// (and is not already fetching), return the one held by the fewest peers,
    /// breaking ties by lowest index. Returns `None` when `for_node` needs no
    /// further pieces. Prioritizing rare pieces keeps every piece replicated and
    /// avoids a late-transfer scramble for a single scarce piece.
    pub fn select_next_piece(&self, for_node: NodeId) -> Option<PieceIndex> {
        let have = self.peer_bitfields.get(&for_node);
        (0..self.piece_count)
            .filter(|&idx| have.is_none_or(|bf| !bf.has(idx)))
            .filter(|&idx| !self.in_flight_to(idx, for_node))
            .min_by(|&a, &b| {
                self.availability(a)
                    .cmp(&self.availability(b))
                    .then(a.cmp(&b))
            })
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn bitfield_set_has_count_complete() {
        let mut bf = PieceBitfield::new(130); // spans 3 words
        assert_eq!(bf.len(), 130);
        assert!(!bf.has(0));
        bf.set(0);
        bf.set(65);
        bf.set(129);
        assert!(bf.has(0) && bf.has(65) && bf.has(129));
        assert!(!bf.has(1));
        assert_eq!(bf.count_set(), 3);
        assert!(!bf.is_complete());
        bf.set(200); // out of range → ignored
        assert_eq!(bf.count_set(), 3);

        assert!(PieceBitfield::full(130).is_complete());
    }

    #[test]
    fn select_next_piece_is_rarest_first() {
        // 4 pieces. Source S has all; A has piece 0; B has pieces 0,1.
        // For a fresh node N: availability is p0=3, p1=2, p2=1, p3=1.
        // Rarest = p2 (tie p2/p3 at 1 → lowest index 2).
        let mut st = SwarmState::new(4);
        st.set_peer_bitfield(NodeId(0), PieceBitfield::full(4)); // source
        let mut a = PieceBitfield::new(4);
        a.set(0);
        st.set_peer_bitfield(NodeId(1), a);
        let mut b = PieceBitfield::new(4);
        b.set(0);
        b.set(1);
        st.set_peer_bitfield(NodeId(2), b);

        assert_eq!(st.availability(0), 3);
        assert_eq!(st.availability(2), 1);
        assert_eq!(
            st.select_next_piece(NodeId(99)),
            Some(2),
            "rarest, lowest-index tie"
        );
    }

    #[test]
    fn select_skips_already_held_and_in_flight() {
        let mut st = SwarmState::new(3);
        st.set_peer_bitfield(NodeId(0), PieceBitfield::full(3)); // source has all
                                                                 // N already holds piece 0.
        st.mark_piece(NodeId(1), 0);
        // Piece 1 is in flight to N.
        st.mark_in_flight(1, NodeId(0), NodeId(1));
        // → only piece 2 is selectable for N.
        assert_eq!(st.select_next_piece(NodeId(1)), Some(2));
        // Completing the in-flight does not change that 1 is now (about to be) held;
        // clearing it makes 1 selectable again (rarest tie → lowest index 1).
        st.complete_in_flight(1);
        assert_eq!(st.select_next_piece(NodeId(1)), Some(1));
    }

    #[test]
    fn select_returns_none_when_complete() {
        let mut st = SwarmState::new(2);
        st.set_peer_bitfield(NodeId(0), PieceBitfield::full(2));
        st.set_peer_bitfield(NodeId(1), PieceBitfield::full(2));
        assert_eq!(st.select_next_piece(NodeId(1)), None);
    }

    #[test]
    fn mark_piece_creates_bitfield_and_updates_availability() {
        let mut st = SwarmState::new(4);
        assert_eq!(st.availability(2), 0);
        st.mark_piece(NodeId(5), 2);
        assert_eq!(st.availability(2), 1);
        assert!(st.peer_bitfields.get(&NodeId(5)).expect("created").has(2));
    }
}
