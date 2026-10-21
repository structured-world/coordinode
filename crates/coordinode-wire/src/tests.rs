#![allow(clippy::expect_used, clippy::unwrap_used)]

use super::*;

/// Minimal prost message standing in for a real RPC body — exercises the codec's
/// prost-encode → compress / decompress → prost-decode round-trip without pulling
/// any service's proto into this crate.
#[derive(Clone, PartialEq, ::prost::Message)]
struct TestMsg {
    #[prost(bytes = "vec", tag = "1")]
    data: Vec<u8>,
}

/// `msg` as the codec puts it on the wire at `level`.
fn encode_compressed_vec(msg: &TestMsg, level: i32) -> Result<Vec<u8>, Status> {
    let mut frame = Vec::new();
    encode_compressed(msg, level, &mut frame)?;
    Ok(frame)
}

#[test]
fn round_trips_payload_at_default_level() {
    let msg = TestMsg {
        data: vec![7u8; 4096],
    };
    let compressed = encode_compressed_vec(&msg, wire_zstd_level()).expect("encode");
    let out: TestMsg = decode_compressed(&compressed).expect("decode");
    assert_eq!(out.data, msg.data);
}

#[test]
fn round_trips_at_max_level() {
    let msg = TestMsg {
        data: (0..10_000u32).map(|i| (i % 251) as u8).collect(),
    };
    let compressed = encode_compressed_vec(&msg, 22).expect("encode");
    let out: TestMsg = decode_compressed(&compressed).expect("decode");
    assert_eq!(out.data, msg.data);
}

#[test]
fn compressible_payload_shrinks_on_the_wire() {
    let msg = TestMsg {
        data: vec![0u8; 8192],
    };
    let compressed = encode_compressed_vec(&msg, 3).expect("encode");
    assert!(
        compressed.len() < 8192,
        "compressible payload must shrink, got {} bytes",
        compressed.len()
    );
}

#[test]
fn empty_payload_round_trips() {
    let msg = TestMsg { data: Vec::new() };
    let compressed = encode_compressed_vec(&msg, 3).expect("encode");
    let out: TestMsg = decode_compressed(&compressed).expect("decode");
    assert_eq!(out.data, msg.data);
}

#[test]
fn decode_rejects_non_zstd_garbage() {
    let result = decode_compressed::<TestMsg>(&[0xFFu8, 0xFF, 0xFF, 0xFF]);
    assert!(result.is_err(), "garbage must not decode to a message");
}

#[test]
fn measures_wire_bytes_compressed_vs_uncompressed() {
    // Realistic-ish batch payload: many similar key/value records. `raw` = prost
    // bytes that would travel the wire WITHOUT the codec.
    let mut data = Vec::new();
    for i in 0..500u32 {
        data.extend_from_slice(format!("node:0:{i:08}").as_bytes());
        data.extend_from_slice(&i.to_be_bytes());
        data.extend_from_slice(format!("value-payload-{}", i % 20).as_bytes());
    }
    let msg = TestMsg { data };
    let raw = msg.encoded_len();

    // -22 is excluded: it panics in structured-zstd 0.0.44's huff0 encoder on
    // real batches (filed upstream); the shipped default is the fast positive 1.
    for lvl in [-10, -1, 1, 3, 7, 11, 22] {
        let len = encode_compressed_vec(&msg, lvl).expect("encode").len();
        eprintln!(
            "wire level {lvl:>3}: uncompressed {raw} -> compressed {len} ({}% of raw)",
            len * 100 / raw
        );
    }

    let default_lvl = wire_zstd_level();
    let c = encode_compressed_vec(&msg, default_lvl).expect("default encode");
    assert!(
        c.len() < raw,
        "default wire level {default_lvl} must reduce a compressible batch: {} vs {raw}",
        c.len()
    );
}

/// One thread's reused compressor and decoder carry nothing from one message
/// into the next: messages of very different sizes, interleaved, with the
/// level changed between them, each come back exactly; and a message larger
/// than the scratch a thread keeps leaves the next one unaffected.
#[test]
fn reused_contexts_keep_messages_apart() {
    let sizes = [0usize, 1, 70_000, 3, (5 << 20) + 17, 4096, 129];
    for (i, size) in sizes.into_iter().enumerate() {
        let msg = TestMsg {
            data: (0..size)
                .map(|k| (k.wrapping_mul(2_654_435_761) >> 7 ^ i) as u8)
                .collect(),
        };
        let level = if i % 2 == 0 { 3 } else { 7 };
        let frame = encode_compressed_vec(&msg, level).expect("encode");
        let out: TestMsg = decode_compressed(&frame).expect("decode");
        assert_eq!(out, msg, "message {i} of {size} bytes at level {level}");
    }
    // A corrupt frame after good ones is refused, and the next good one
    // still decodes.
    let good = encode_compressed_vec(
        &TestMsg {
            data: vec![1, 2, 3],
        },
        3,
    )
    .expect("encode");
    let mut bad = good.clone();
    let last = bad.len() - 1;
    bad[last] ^= 0xFF;
    bad.truncate(bad.len() - 2);
    assert!(decode_compressed::<TestMsg>(&bad).is_err());
    let out: TestMsg = decode_compressed(&good).expect("decode after a refusal");
    assert_eq!(out.data, vec![1, 2, 3]);
}

#[test]
fn wire_level_set_and_get() {
    set_wire_zstd_level(9);
    assert_eq!(wire_zstd_level(), 9);
    set_wire_zstd_level(3);
    assert_eq!(wire_zstd_level(), 3);
}
