//! Inter-node segment transfer: the gRPC transport for replication repair,
//! operator-commanded migration, and node resync.
//!
//! Carries swarm pieces between nodes via the `SegmentTransferService` RPC. The
//! source streams a header (segment id + transfer manifest) followed by one
//! frame per encoded piece; the target verifies each piece, decodes it, and
//! assembles the segment with a [`coordinode_swarm::SegmentWriter`]. The piece
//! model, checksums, encodings, and scheduling live in `coordinode-swarm`; this
//! module is the wire adapter that moves the bytes.

/// Generated gRPC stubs for `coordinode/v1/replication/transfer.proto`:
/// `segment_transfer_service_{server,client}`, `PieceData`,
/// `SegmentTransferHeader`, `PieceFrame`, `TransferAck`.
#[allow(clippy::all, clippy::pedantic, missing_docs)]
pub mod proto {
    tonic::include_proto!("coordinode.v1.replication");
}
