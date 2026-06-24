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

    /// Serialize as little-endian bytes: `ceil(len / 8)` bytes, where piece `i`
    /// is bit `i % 8` of byte `i / 8`. The wire form for a bitfield exchange.
    pub fn to_le_bytes(&self) -> Vec<u8> {
        let n = (self.len as usize).div_ceil(8);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let word = self.words[i / 8];
            out.push((word >> ((i % 8) * 8)) as u8);
        }
        out
    }

    /// Reconstruct a `piece_count`-sized bitfield from its little-endian byte
    /// form ([`Self::to_le_bytes`]). Bytes beyond the bitfield's word backing are
    /// ignored, so a malformed peer message cannot panic or over-allocate.
    pub fn from_le_bytes(bytes: &[u8], piece_count: u32) -> Self {
        let mut bf = Self::new(piece_count);
        for (i, &b) in bytes.iter().enumerate() {
            if let Some(word) = bf.words.get_mut(i / 8) {
                *word |= (b as u64) << ((i % 8) * 8);
            }
        }
        bf
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
mod tests;
