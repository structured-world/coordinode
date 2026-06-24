use std::collections::HashMap;

use super::*;
use crate::scheduler::Freshness;
use crate::segment::{split_segment, PieceEncoding};

/// In-memory [`PieceSource`] holding a subset of a segment's wire pieces.
struct MemSource {
    node: NodeId,
    bitfield: PieceBitfield,
    wire: HashMap<PieceIndex, Vec<u8>>,
}

impl MemSource {
    /// A source holding the pieces at `indices` of the full `wire` set.
    fn holding(node: u64, piece_count: u32, wire: &[Vec<u8>], indices: &[PieceIndex]) -> Self {
        let mut bitfield = PieceBitfield::new(piece_count);
        let mut held = HashMap::new();
        for &idx in indices {
            bitfield.set(idx);
            held.insert(idx, wire[idx as usize].clone());
        }
        Self {
            node: NodeId(node),
            bitfield,
            wire: held,
        }
    }
}

impl PieceSource for MemSource {
    fn node(&self) -> NodeId {
        self.node
    }
    fn bitfield(&self) -> PieceBitfield {
        self.bitfield.clone()
    }
    fn candidate(&self) -> SourceCandidate {
        SourceCandidate {
            node: self.node,
            utilization: 0.0,
            bandwidth_to_target: 1.0,
            same_rack: false,
            tit_for_tat: 1.0,
            freshness: Freshness::Verified,
        }
    }
    fn fetch_piece(&self, index: PieceIndex) -> SwarmResult<Vec<u8>> {
        self.wire
            .get(&index)
            .cloned()
            .ok_or_else(|| SwarmError::Source(format!("piece {index} not held")))
    }
}

fn split(data: &[u8], piece_size: usize) -> (SegmentManifest, Vec<Vec<u8>>) {
    split_segment(data, piece_size, PieceEncoding::None).expect("split")
}

#[test]
fn single_source_holding_all_reconstructs_segment() {
    let data: Vec<u8> = (0..5000u32).map(|i| (i % 251) as u8).collect();
    let (manifest, wire) = split(&data, 512);
    let all: Vec<PieceIndex> = (0..manifest.piece_count()).collect();
    let src = MemSource::holding(2, manifest.piece_count(), &wire, &all);
    let sources: Vec<&dyn PieceSource> = vec![&src];

    let out = swarm_download(NodeId(1), &manifest, &sources, Vec::new()).expect("download");
    assert_eq!(out, data);
}

#[test]
fn two_sources_disjoint_halves_reconstructs_segment() {
    let data: Vec<u8> = (0..8000u32).map(|i| (i % 97) as u8).collect();
    let (manifest, wire) = split(&data, 500);
    let n = manifest.piece_count();
    let evens: Vec<PieceIndex> = (0..n).filter(|i| i % 2 == 0).collect();
    let odds: Vec<PieceIndex> = (0..n).filter(|i| i % 2 == 1).collect();
    let a = MemSource::holding(2, n, &wire, &evens);
    let b = MemSource::holding(3, n, &wire, &odds);
    let sources: Vec<&dyn PieceSource> = vec![&a, &b];

    let out = swarm_download(NodeId(1), &manifest, &sources, Vec::new()).expect("download");
    assert_eq!(
        out, data,
        "multi-source coverage must reconstruct the whole segment"
    );
}

#[test]
fn missing_piece_with_no_source_errors() {
    let data: Vec<u8> = (0..4000u32).map(|i| i as u8).collect();
    let (manifest, wire) = split(&data, 400);
    let n = manifest.piece_count();
    assert!(n >= 3, "need several pieces for this case");
    // Hold every piece except the last → the segment can never complete.
    let all_but_last: Vec<PieceIndex> = (0..n - 1).collect();
    let src = MemSource::holding(2, n, &wire, &all_but_last);
    let sources: Vec<&dyn PieceSource> = vec![&src];

    let err = swarm_download(NodeId(1), &manifest, &sources, Vec::new());
    assert!(err.is_err(), "an unreachable piece must fail the download");
}

#[test]
fn corrupt_piece_fails_verification() {
    let data: Vec<u8> = (0..3000u32).map(|i| (i % 13) as u8).collect();
    let (manifest, wire) = split(&data, 300);
    let n = manifest.piece_count();
    let all: Vec<PieceIndex> = (0..n).collect();
    let mut src = MemSource::holding(2, n, &wire, &all);
    // Corrupt one piece's bytes so its xxh3 no longer matches the manifest.
    if let Some(bytes) = src.wire.get_mut(&0) {
        bytes[0] ^= 0xFF;
    }
    let sources: Vec<&dyn PieceSource> = vec![&src];

    let err = swarm_download(NodeId(1), &manifest, &sources, Vec::new());
    assert!(
        err.is_err(),
        "a piece failing its checksum must abort the download"
    );
}
