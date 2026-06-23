use super::*;
use coordinode_swarm::{split_segment, LocalPieceStore, ZstdLevel};
use parking_lot::Mutex;
use std::collections::HashMap;

#[derive(Default)]
struct CollectingSink {
    stored: Mutex<HashMap<u64, Vec<u8>>>,
}
impl SegmentSink for CollectingSink {
    fn store_segment(&self, segment: SegmentId, data: &[u8]) -> Result<(), String> {
        self.stored.lock().insert(segment.0, data.to_vec());
        Ok(())
    }
}

fn frame_stream(frames: Vec<PieceData>) -> impl Stream<Item = Result<PieceData, Status>> + Unpin {
    futures_util::stream::iter(frames.into_iter().map(Ok))
}

fn segment(len: usize) -> Vec<u8> {
    (0..len).map(|i| ((i / 7) % 11) as u8).collect()
}

#[tokio::test]
async fn round_trips_each_encoding() {
    for enc in [
        PieceEncoding::None,
        PieceEncoding::Lz4,
        PieceEncoding::Zstd(ZstdLevel::Fastest),
    ] {
        let data = segment(9 * 1024 + 17);
        let (manifest, wire) = split_segment(&data, 1024, enc).expect("split");
        let frames = build_frames(SegmentId(5), &manifest, &wire);

        let sink = CollectingSink::default();
        let ack = receive(frame_stream(frames), &sink).await.expect("receive");

        assert!(ack.ok, "ack ok for enc={enc:?}: {}", ack.error);
        assert_eq!(ack.pieces_received as usize, wire.len());
        assert_eq!(sink.stored.lock().get(&5), Some(&data));
    }
}

#[tokio::test]
async fn corrupt_piece_yields_failed_ack_and_stores_nothing() {
    let data = segment(4096);
    let (manifest, wire) = split_segment(&data, 1024, PieceEncoding::None).expect("split");
    let mut frames = build_frames(SegmentId(1), &manifest, &wire);
    // Flip a byte inside the second piece frame's wire bytes.
    if let Some(Frame::Piece(p)) = frames[2].frame.as_mut() {
        p.wire[0] ^= 0xFF;
    }
    let sink = CollectingSink::default();
    let ack = receive(frame_stream(frames), &sink).await.expect("receive");

    assert!(!ack.ok, "corruption must fail the ack");
    assert!(
        sink.stored.lock().is_empty(),
        "corrupt segment must not be stored"
    );
}

#[tokio::test]
async fn source_frames_round_trip_to_target() {
    // frames_for (source gather from a PieceStore) feeds receive (target
    // assemble) — the full transport logic minus the literal wire.
    let data = segment(8 * 1024 + 5);
    let mut store = LocalPieceStore::new();
    store
        .insert(SegmentId(9), &data, 1024, PieceEncoding::Lz4)
        .expect("insert");

    let frames = frames_for(&store, SegmentId(9)).expect("frames_for");
    let sink = CollectingSink::default();
    let ack = receive(frame_stream(frames), &sink).await.expect("receive");

    assert!(ack.ok, "round-trip ack: {}", ack.error);
    assert_eq!(sink.stored.lock().get(&9), Some(&data));
}

#[tokio::test]
async fn missing_header_is_protocol_error() {
    let piece = PieceData {
        frame: Some(Frame::Piece(PieceFrame {
            index: 0,
            wire: vec![1, 2, 3],
        })),
    };
    let sink = CollectingSink::default();
    let result = receive(frame_stream(vec![piece]), &sink).await;
    assert!(
        result.is_err(),
        "a leading piece (no header) must be rejected"
    );
}
