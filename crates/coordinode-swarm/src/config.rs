//! Operator-facing swarm transfer configuration.
//!
//! The server parses the packaged config (codec + zstd effort) into a
//! [`SwarmConfig`] and hands it to the transfer engine. Kept here so the parse
//! rules live next to the [`PieceEncoding`] they map onto.

use crate::segment::{PieceEncoding, SwarmError, SwarmResult, ZstdLevel};

/// Tunables for segment transfer (replication repair, shard migration, resync).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SwarmConfig {
    /// Encoding applied to pieces on the wire.
    pub transfer_encoding: PieceEncoding,
}

impl Default for SwarmConfig {
    fn default() -> Self {
        // LZ4: fast, low-CPU — the right default for transfer, which favours
        // cheap compression over ratio (the bytes are transient on the wire).
        Self {
            transfer_encoding: PieceEncoding::Lz4,
        }
    }
}

impl SwarmConfig {
    /// Build from operator config strings: a codec (`none` / `lz4` / `zstd`) and,
    /// when the codec is `zstd`, an effort (`fastest` / `default` / `better`).
    /// The effort is ignored for non-zstd codecs.
    ///
    /// # Errors
    /// [`SwarmError::Source`] for an unknown codec or zstd effort.
    pub fn from_strs(codec: &str, zstd_level: &str) -> SwarmResult<Self> {
        Ok(Self {
            transfer_encoding: parse_encoding(codec, zstd_level)?,
        })
    }
}

fn parse_encoding(codec: &str, zstd_level: &str) -> SwarmResult<PieceEncoding> {
    match codec.trim().to_ascii_lowercase().as_str() {
        "none" => Ok(PieceEncoding::None),
        "lz4" => Ok(PieceEncoding::Lz4),
        "zstd" => Ok(PieceEncoding::Zstd(parse_zstd_level(zstd_level)?)),
        other => Err(SwarmError::Source(format!(
            "unknown transfer codec {other:?} (expected none | lz4 | zstd)"
        ))),
    }
}

fn parse_zstd_level(s: &str) -> SwarmResult<ZstdLevel> {
    match s.trim().to_ascii_lowercase().as_str() {
        // Empty = take the codec's default effort.
        "fastest" | "" => Ok(ZstdLevel::Fastest),
        "default" => Ok(ZstdLevel::Default),
        "better" => Ok(ZstdLevel::Better),
        other => Err(SwarmError::Source(format!(
            "unknown zstd level {other:?} (expected fastest | default | better)"
        ))),
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
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
}
