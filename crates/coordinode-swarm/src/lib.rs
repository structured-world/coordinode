//! `coordinode-swarm` — BitTorrent-inspired swarm segment transfer.
//!
//! This crate is the **transport mechanism** (how bytes move between nodes) for
//! replication repair, operator-commanded shard migration, and node resync.
//! Every node that receives a piece of a segment immediately becomes a source
//! for other nodes, turning a linear one-to-one transfer into an exponential
//! O(log N) fanout. The orchestration lifecycle (when to block, hand over, catch
//! up) lives in the consensus layer; this crate only moves the bytes.
//!
//! # Current scope
//!
//! The segment **piece model**: a segment is split into fixed-size pieces, each
//! with an xxh3 checksum plus a whole-segment checksum, so a receiver can verify
//! every piece independently as it arrives and the assembled whole at the end.
//! See [`SegmentManifest`], [`split_segment`], [`verify_piece`], [`assemble`],
//! and [`MediaClass`] for media-tuned piece sizing.
//!
//! Swarm scheduling: [`SwarmState`] tracks which peers hold which pieces and
//! drives rarest-first piece selection ([`SwarmState::select_next_piece`]) so
//! scarce pieces replicate first. Source-selection scoring (tit-for-tat
//! fairness, locality) and the gRPC piece-exchange transport build on this in
//! subsequent increments.
//!
//! # Crate tier
//!
//! The piece model is pure logic over `Vec<u8>` (no I/O) and is `alloc`-ready;
//! it is shipped as a plain `std` crate today, matching the sibling cluster
//! crates, and can drop to `no_std + alloc` once the workspace adds the CI job.

mod config;
mod download;
mod scheduler;
mod segment;
mod state;
mod transfer;

pub use config::SwarmConfig;
pub use download::{swarm_download, PieceSource};
pub use scheduler::{select_source, Freshness, SourceCandidate};
pub use segment::{
    assemble, build_manifest, cross_tier_piece_size, split_segment, verify_piece, MediaClass,
    PieceEncoding, PieceIndex, SegmentManifest, SegmentWriter, SwarmError, SwarmResult, ZstdLevel,
};
pub use state::{NodeId, PieceBitfield, SwarmState};
pub use transfer::{transfer, LocalPieceStore, PieceStore, SegmentId};
