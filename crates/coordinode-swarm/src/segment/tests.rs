use super::*;

fn segment(len: usize) -> Vec<u8> {
    // Repetitive enough that compression actually shrinks it.
    (0..len).map(|i| ((i / 7) % 17) as u8).collect()
}

const ENCODINGS: [PieceEncoding; 3] = [
    PieceEncoding::None,
    PieceEncoding::Lz4,
    PieceEncoding::Zstd(ZstdLevel::Fastest),
];

#[test]
fn media_class_piece_sizes_match_spec() {
    assert_eq!(MediaClass::Ram.optimal_piece_size(), 4 * 1024 * 1024);
    assert_eq!(MediaClass::Nvme.optimal_piece_size(), 1024 * 1024);
    assert_eq!(MediaClass::Ssd.optimal_piece_size(), 1024 * 1024);
    assert_eq!(MediaClass::Hdd.optimal_piece_size(), 4 * 1024 * 1024);
    assert_eq!(
        cross_tier_piece_size(MediaClass::Hdd, MediaClass::Nvme),
        1024 * 1024
    );
}

#[test]
fn round_trips_each_encoding_via_split_and_streaming_writer() {
    for enc in ENCODINGS {
        for len in [48 * 1024, 48 * 1024 + 7, 1, 1023] {
            let data = segment(len);
            let (manifest, wire) = split_segment(&data, 1024, enc).expect("split");
            assert_eq!(manifest.encoding, enc);

            // Vec convenience.
            assert_eq!(assemble(&manifest, &wire).expect("assemble"), data);

            // Streaming writer into an arbitrary sink, piece by piece.
            let mut w = SegmentWriter::new(&manifest, Vec::new());
            for piece in &wire {
                w.push(piece).expect("push");
            }
            assert_eq!(
                w.finish().expect("finish"),
                data,
                "stream enc={enc:?} len={len}"
            );
        }
    }
}

#[test]
fn build_manifest_matches_split_and_records_encoding() {
    let data = segment(40 * 1024);
    for enc in ENCODINGS {
        let m1 = build_manifest(&data, 4096, enc).expect("manifest");
        let (m2, _) = split_segment(&data, 4096, enc).expect("split");
        assert_eq!(m1, m2);
        assert_eq!(m1.encoding, enc);
    }
}

#[test]
fn compression_shrinks_compressible_data() {
    let data = segment(64 * 1024);
    let (_, raw) = split_segment(&data, 64 * 1024, PieceEncoding::None).expect("raw");
    for enc in [PieceEncoding::Lz4, PieceEncoding::Zstd(ZstdLevel::Fastest)] {
        let (_, comp) = split_segment(&data, 64 * 1024, enc).expect("comp");
        assert!(
            comp[0].len() < raw[0].len(),
            "{enc:?} wire piece ({}) should compress below raw ({})",
            comp[0].len(),
            raw[0].len()
        );
    }
}

#[test]
fn manifest_piece_count_and_ranges() {
    let data = segment(2500);
    let (manifest, wire) = split_segment(&data, 1000, PieceEncoding::None).expect("split");
    assert_eq!(manifest.piece_count(), 3);
    assert_eq!(wire.len(), 3);
    assert_eq!(manifest.piece_range(0).expect("r0"), (0, 1000));
    assert_eq!(manifest.piece_range(2).expect("r2"), (2000, 2500));
    assert!(manifest.piece_range(3).is_err());
}

#[test]
fn verify_piece_detects_corruption() {
    let data = segment(4096);
    let (manifest, wire) =
        split_segment(&data, 1024, PieceEncoding::Zstd(ZstdLevel::Fastest)).expect("split");
    verify_piece(&manifest, 1, &wire[1]).expect("clean");
    let mut bad = wire[1].clone();
    bad[0] ^= 0xFF;
    assert_eq!(
        verify_piece(&manifest, 1, &bad),
        Err(SwarmError::PieceHashMismatch { index: 1 })
    );
    assert!(matches!(
        verify_piece(&manifest, 99, &wire[0]),
        Err(SwarmError::PieceIndexOutOfRange { index: 99, .. })
    ));
}

#[test]
fn assemble_rejects_corruption_wrong_count_and_misorder() {
    let data = segment(4096);
    let (manifest, wire) = split_segment(&data, 1024, PieceEncoding::None).expect("split");

    assert!(matches!(
        assemble(&manifest, &wire[..3]),
        Err(SwarmError::PieceCountMismatch {
            expected: 4,
            actual: 3
        })
    ));

    let mut corrupt = wire.clone();
    corrupt[2][5] ^= 0xFF;
    assert_eq!(
        assemble(&manifest, &corrupt),
        Err(SwarmError::PieceHashMismatch { index: 2 })
    );

    let mut misordered = wire.clone();
    misordered.swap(0, 1);
    assert!(assemble(&manifest, &misordered).is_err());
}

#[test]
fn writer_finish_detects_short_count() {
    let data = segment(3000);
    let (manifest, wire) = split_segment(&data, 1000, PieceEncoding::None).expect("split");
    let mut w = SegmentWriter::new(&manifest, Vec::new());
    w.push(&wire[0]).expect("push 0");
    assert!(matches!(
        w.finish(),
        Err(SwarmError::PieceCountMismatch {
            expected: 3,
            actual: 1
        })
    ));
}

#[test]
fn empty_segment_has_no_pieces() {
    for enc in ENCODINGS {
        let (manifest, wire) = split_segment(&[], 1024, enc).expect("split empty");
        assert_eq!(manifest.piece_count(), 0);
        assert!(wire.is_empty());
        assert_eq!(
            assemble(&manifest, &[]).expect("assemble empty"),
            Vec::<u8>::new()
        );
    }
}

#[test]
fn zero_piece_size_rejected() {
    assert_eq!(
        build_manifest(&segment(10), 0, PieceEncoding::None),
        Err(SwarmError::ZeroPieceSize)
    );
}

#[test]
fn encoding_wire_discriminants_round_trip() {
    for enc in [
        PieceEncoding::None,
        PieceEncoding::Lz4,
        PieceEncoding::Zstd(ZstdLevel::Fastest),
        PieceEncoding::Zstd(ZstdLevel::Default),
        PieceEncoding::Zstd(ZstdLevel::Better),
    ] {
        let (e, l) = enc.to_wire();
        assert_eq!(PieceEncoding::from_wire(e, l).expect("from_wire"), enc);
    }
    assert!(PieceEncoding::from_wire(7, 0).is_err());
    assert!(PieceEncoding::from_wire(2, 9).is_err());
}
