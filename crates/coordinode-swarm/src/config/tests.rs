use super::*;

#[test]
fn default_is_lz4() {
    assert_eq!(SwarmConfig::default().transfer_encoding, PieceEncoding::Lz4);
}

#[test]
fn parses_codecs_and_levels() {
    assert_eq!(
        SwarmConfig::from_strs("none", "")
            .expect("none")
            .transfer_encoding,
        PieceEncoding::None
    );
    assert_eq!(
        SwarmConfig::from_strs("LZ4", "")
            .expect("lz4")
            .transfer_encoding,
        PieceEncoding::Lz4
    );
    assert_eq!(
        SwarmConfig::from_strs("zstd", "better")
            .expect("zstd")
            .transfer_encoding,
        PieceEncoding::Zstd(ZstdLevel::Better)
    );
    // Empty level → fastest; codec match is case-insensitive.
    assert_eq!(
        SwarmConfig::from_strs("Zstd", "")
            .expect("zstd default")
            .transfer_encoding,
        PieceEncoding::Zstd(ZstdLevel::Fastest)
    );
}

#[test]
fn rejects_unknown_codec_and_level() {
    assert!(SwarmConfig::from_strs("brotli", "").is_err());
    assert!(SwarmConfig::from_strs("zstd", "ultra").is_err());
}
