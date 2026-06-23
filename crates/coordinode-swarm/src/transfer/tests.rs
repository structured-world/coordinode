use super::*;
use crate::segment::ZstdLevel;

fn segment(len: usize) -> Vec<u8> {
    (0..len).map(|i| ((i / 5) % 13) as u8).collect()
}

#[test]
fn transfer_reconstructs_segment_for_each_encoding() {
    for enc in [
        PieceEncoding::None,
        PieceEncoding::Lz4,
        PieceEncoding::Zstd(ZstdLevel::Fastest),
    ] {
        let data = segment(20 * 1024 + 3);
        let mut store = LocalPieceStore::new();
        store
            .insert(SegmentId(42), &data, 1024, enc)
            .expect("insert");

        let out = transfer(&store, SegmentId(42), Vec::new()).expect("transfer");
        assert_eq!(out, data, "transfer round-trip enc={enc:?}");
    }
}

#[test]
fn transfer_unknown_segment_errors() {
    let store = LocalPieceStore::new();
    assert!(transfer(&store, SegmentId(7), Vec::new()).is_err());
}

#[test]
fn transfer_detects_a_corrupting_store() {
    // A store whose wire_piece flips a byte must be caught by the per-piece
    // checksum — repair never silently writes corrupt data.
    struct Corrupting(LocalPieceStore);
    impl PieceStore for Corrupting {
        fn manifest(&self, s: SegmentId) -> SwarmResult<SegmentManifest> {
            self.0.manifest(s)
        }
        fn wire_piece(&self, s: SegmentId, i: PieceIndex) -> SwarmResult<Vec<u8>> {
            let mut p = self.0.wire_piece(s, i)?;
            if i == 1 {
                p[0] ^= 0xFF;
            }
            Ok(p)
        }
    }
    let data = segment(4096);
    let mut base = LocalPieceStore::new();
    base.insert(SegmentId(1), &data, 1024, PieceEncoding::None)
        .expect("insert");
    let store = Corrupting(base);
    assert!(matches!(
        transfer(&store, SegmentId(1), Vec::new()),
        Err(SwarmError::PieceHashMismatch { index: 1 })
    ));
}
