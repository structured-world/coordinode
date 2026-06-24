#![allow(clippy::expect_used, clippy::unwrap_used)]

use super::*;
use crate::proto::replication::RaftPayload;

#[test]
fn round_trips_payload_at_default_level() {
    let msg = RaftPayload {
        data: vec![7u8; 4096],
    };
    let compressed = encode_compressed(&msg, CompressionLevel::Level(3)).expect("encode");
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
fn wire_level_set_and_get() {
    set_wire_zstd_level(9);
    assert_eq!(wire_zstd_level(), 9);
    // Restore the process default so other tests see a stable level.
    set_wire_zstd_level(3);
    assert_eq!(wire_zstd_level(), 3);
}
