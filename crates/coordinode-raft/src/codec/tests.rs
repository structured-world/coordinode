#![allow(clippy::expect_used, clippy::unwrap_used)]

use super::*;
use crate::proto::replication::RaftPayload;

#[test]
fn round_trips_payload_at_default_level() {
    let msg = RaftPayload {
        data: vec![7u8; 4096],
    };
    // Use the actually-shipped default level so this covers the wire round-trip
    // for whatever the codec runs by default.
    let compressed =
        encode_compressed(&msg, CompressionLevel::Level(wire_zstd_level())).expect("encode");
    let out: RaftPayload = decode_compressed(&compressed).expect("decode");
    assert_eq!(out.data, msg.data);
}

#[test]
fn round_trips_at_max_level() {
    // db4 geo-link profile: level 22.
    let msg = RaftPayload {
        data: (0..10_000u32).map(|i| (i % 251) as u8).collect(),
    };
    let compressed = encode_compressed(&msg, CompressionLevel::Level(22)).expect("encode");
    let out: RaftPayload = decode_compressed(&compressed).expect("decode");
    assert_eq!(out.data, msg.data);
}

#[test]
fn compressible_payload_shrinks_on_the_wire() {
    let msg = RaftPayload {
        data: vec![0u8; 8192],
    };
    let compressed = encode_compressed(&msg, CompressionLevel::Level(3)).expect("encode");
    assert!(
        compressed.len() < 8192,
        "compressible payload must shrink, got {} bytes",
        compressed.len()
    );
}

#[test]
fn empty_payload_round_trips() {
    let msg = RaftPayload { data: Vec::new() };
    let compressed = encode_compressed(&msg, CompressionLevel::Level(3)).expect("encode");
    let out: RaftPayload = decode_compressed(&compressed).expect("decode");
    assert_eq!(out.data, msg.data);
}

#[test]
fn decode_rejects_non_zstd_garbage() {
    // Not a zstd frame (bad magic) — must error, never silently yield a default.
    let result = decode_compressed::<RaftPayload>(&[0xFFu8, 0xFF, 0xFF, 0xFF]);
    assert!(result.is_err(), "garbage must not decode to a message");
}

#[test]
fn measures_wire_bytes_compressed_vs_uncompressed() {
    // Realistic-ish Raft batch payload: many similar key/value records, the kind
    // of mutation batch an AppendEntries carries. `raw` = prost bytes that would
    // travel the wire WITHOUT the codec; the per-level sizes are WITH it.
    let mut data = Vec::new();
    for i in 0..500u32 {
        data.extend_from_slice(format!("node:0:{i:08}").as_bytes());
        data.extend_from_slice(&i.to_be_bytes());
        data.extend_from_slice(format!("value-payload-{}", i % 20).as_bytes());
    }
    let msg = RaftPayload { data };
    let raw = msg.encoded_len();

    // Levels 1..=22 and the safe ultra-fast negatives. (-22 is excluded: it
    // currently panics in structured-zstd 0.0.44's huff0 encoder on real
    // batches — filed upstream; the shipped default is the fast positive 1.)
    for lvl in [-10, -1, 1, 3, 7, 11, 22] {
        let len = encode_compressed(&msg, CompressionLevel::Level(lvl))
            .expect("encode")
            .len();
        eprintln!(
            "wire level {lvl:>3}: uncompressed {raw} -> compressed {len} ({}% of raw)",
            len * 100 / raw
        );
    }

    // Guard: the shipped default level must compress this batch on the wire
    // without panicking and must actually reduce the bytes.
    let default_lvl = wire_zstd_level();
    let c = encode_compressed(&msg, CompressionLevel::Level(default_lvl)).expect("default encode");
    assert!(
        c.len() < raw,
        "default wire level {default_lvl} must reduce a compressible batch: {} vs {raw}",
        c.len()
    );
}

#[test]
fn tiny_message_overhead_is_bounded() {
    // Small RPCs (a vote, a heartbeat) barely compress; the zstd frame adds a
    // small fixed overhead. Document that it stays bounded (not pathological).
    let msg = RaftPayload {
        data: b"vote:term=7:for=2".to_vec(),
    };
    let raw = msg.encoded_len();
    let c = encode_compressed(&msg, CompressionLevel::Level(1)).expect("encode");
    eprintln!("tiny: uncompressed {raw} -> compressed {} bytes", c.len());
    // Frame overhead is a small constant, not a multiple.
    assert!(
        c.len() < raw + 64,
        "tiny-message overhead must stay bounded: {} vs {raw}",
        c.len()
    );
}

#[test]
fn wire_level_set_and_get() {
    set_wire_zstd_level(9);
    assert_eq!(wire_zstd_level(), 9);
    // Restore the process default (1) so other tests see a stable level.
    set_wire_zstd_level(1);
    assert_eq!(wire_zstd_level(), 1);
}
