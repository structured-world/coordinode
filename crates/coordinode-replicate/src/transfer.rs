//! Inter-node segment transfer: the gRPC transport for replication repair,
//! operator-commanded migration, and node resync.
//!
//! Carries swarm pieces between nodes via the `SegmentTransferService` RPC. The
//! source streams a header (segment id + transfer manifest) followed by one
//! frame per encoded piece; the target verifies each piece, decodes it, and
//! assembles the segment with a [`coordinode_swarm::SegmentWriter`]. The piece
//! model, checksums, encodings, and scheduling live in `coordinode-swarm`; this
//! module is the wire adapter that moves the bytes.
//!
//! The receive path ([`receive`]) is generic over the frame stream so it is
//! testable without a network; the tonic service ([`SegmentTransferHandler`]) is
//! a thin wrapper, and [`build_frames`] is the matching source side.

use std::sync::Arc;

use coordinode_swarm::{
    PieceEncoding, PieceStore, SegmentId, SegmentManifest, SegmentWriter, SwarmResult,
};
use futures_util::{Stream, StreamExt};
use tonic::{Request, Response, Status, Streaming};

use proto::piece_data::Frame;
use proto::{PieceData, PieceFrame, SegmentTransferHeader, TransferAck};

/// Generated gRPC stubs for `coordinode/v1/replication/transfer.proto`:
/// `segment_transfer_service_{server,client}`, `PieceData`,
/// `SegmentTransferHeader`, `PieceFrame`, `TransferAck`.
#[allow(clippy::all, clippy::pedantic, missing_docs)]
pub mod proto {
    tonic::include_proto!("coordinode.v1.replication");
}

/// Where a fully received, checksum-verified segment is placed on the target.
/// The transport hands off the assembled raw bytes; the implementation decides
/// how they land in local storage (replace a corrupt segment, install a migrated
/// one, fill a resync gap).
pub trait SegmentSink: Send + Sync {
    /// Persist the assembled raw segment bytes under `segment`.
    ///
    /// # Errors
    /// Implementation-defined storage failure, surfaced to the source in the
    /// transfer ack.
    fn store_segment(&self, segment: SegmentId, data: &[u8]) -> Result<(), String>;
}

/// Build the wire frames for one segment transfer: a leading header carrying the
/// manifest, then one frame per wire piece in index order. The source side of a
/// transfer (the matching receive side is [`receive`]).
pub fn build_frames(
    segment: SegmentId,
    manifest: &SegmentManifest,
    wire_pieces: &[Vec<u8>],
) -> Vec<PieceData> {
    let (encoding, zstd_level) = manifest.encoding.to_wire();
    let mut frames = Vec::with_capacity(wire_pieces.len() + 1);
    frames.push(PieceData {
        frame: Some(Frame::Header(SegmentTransferHeader {
            segment_id: segment.0,
            encoding,
            zstd_level,
            piece_hashes: manifest.piece_hashes.clone(),
            total_hash: manifest.total_hash,
            piece_size: manifest.piece_size as u32,
            total_len: manifest.total_len as u64,
        })),
    });
    for (index, wire) in wire_pieces.iter().enumerate() {
        frames.push(PieceData {
            frame: Some(Frame::Piece(PieceFrame {
                index: index as u32,
                wire: wire.clone(),
            })),
        });
    }
    frames
}

/// Gather every wire frame for `segment` from a [`PieceStore`] (the source's
/// manifest plus each wire piece in order), ready to stream to a peer. The
/// network layer wraps these in the client RPC; keeping the gather here (rather
/// than in the connection layer) lets it be tested against [`receive`] without a
/// transport.
///
/// # Errors
/// Any [`PieceStore`] error (unknown segment, unreadable piece).
pub fn frames_for(store: &dyn PieceStore, segment: SegmentId) -> SwarmResult<Vec<PieceData>> {
    let manifest = store.manifest(segment)?;
    let mut wire = Vec::with_capacity(manifest.piece_count() as usize);
    for index in 0..manifest.piece_count() {
        wire.push(store.wire_piece(segment, index)?);
    }
    Ok(build_frames(segment, &manifest, &wire))
}

/// Receive one segment from a frame stream: read the header, assemble the pieces
/// through a [`SegmentWriter`] (verify + decode each), and hand the result to
/// `sink`. A checksum / decode / storage failure yields an ack with `ok = false`
/// (corrupt data is never stored); a malformed stream (no header, stray header)
/// is a protocol error ([`Status`]).
///
/// # Errors
/// [`Status`] for a transport error or a malformed frame sequence.
pub async fn receive<St>(mut stream: St, sink: &dyn SegmentSink) -> Result<TransferAck, Status>
where
    St: Stream<Item = Result<PieceData, Status>> + Unpin,
{
    let header = match stream.next().await.transpose()?.and_then(|pd| pd.frame) {
        Some(Frame::Header(h)) => h,
        Some(Frame::Piece(_)) => {
            return Err(Status::invalid_argument("first frame must be a header"))
        }
        None => return Err(Status::invalid_argument("empty transfer stream")),
    };

    let manifest = manifest_from_header(&header).map_err(Status::invalid_argument)?;
    let segment = SegmentId(header.segment_id);
    let mut writer = SegmentWriter::new(&manifest, Vec::with_capacity(manifest.total_len));
    let mut received = 0u32;

    while let Some(item) = stream.next().await {
        match item?.frame {
            Some(Frame::Piece(p)) => {
                if let Err(e) = writer.push(&p.wire) {
                    return Ok(fail(received, e.to_string()));
                }
                received += 1;
            }
            Some(Frame::Header(_)) => {
                return Err(Status::invalid_argument("unexpected second header"))
            }
            None => return Err(Status::invalid_argument("empty frame")),
        }
    }

    let data = match writer.finish() {
        Ok(d) => d,
        Err(e) => return Ok(fail(received, e.to_string())),
    };
    match sink.store_segment(segment, &data) {
        Ok(()) => Ok(TransferAck {
            ok: true,
            error: String::new(),
            pieces_received: received,
        }),
        Err(e) => Ok(fail(received, e)),
    }
}

fn fail(received: u32, error: String) -> TransferAck {
    TransferAck {
        ok: false,
        error,
        pieces_received: received,
    }
}

fn manifest_from_header(h: &SegmentTransferHeader) -> Result<SegmentManifest, String> {
    let encoding = PieceEncoding::from_wire(h.encoding, h.zstd_level).map_err(|e| e.to_string())?;
    Ok(SegmentManifest {
        piece_size: h.piece_size as usize,
        total_len: h.total_len as usize,
        encoding,
        piece_hashes: h.piece_hashes.clone(),
        total_hash: h.total_hash,
    })
}

/// The target-side tonic service: assembles inbound transfers into the injected
/// [`SegmentSink`].
pub struct SegmentTransferHandler<S: SegmentSink> {
    sink: Arc<S>,
}

impl<S: SegmentSink> SegmentTransferHandler<S> {
    /// Build a handler that stores received segments through `sink`.
    pub fn new(sink: Arc<S>) -> Self {
        Self { sink }
    }
}

#[tonic::async_trait]
impl<S: SegmentSink + 'static> proto::segment_transfer_service_server::SegmentTransferService
    for SegmentTransferHandler<S>
{
    async fn transfer_pieces(
        &self,
        request: Request<Streaming<PieceData>>,
    ) -> Result<Response<TransferAck>, Status> {
        receive(request.into_inner(), self.sink.as_ref())
            .await
            .map(Response::new)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
