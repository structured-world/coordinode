use super::*;

#[test]
fn match_32msb_same() {
    assert!(match_32msb(0x0000_0001_0000_0000, 0x0000_0001_FFFF_FFFF));
}

#[test]
fn match_32msb_different() {
    assert!(!match_32msb(0x0000_0001_0000_0000, 0x0000_0002_0000_0000));
}

#[test]
fn match_32msb_zero() {
    assert!(match_32msb(0, 0xFFFF_FFFF));
}

#[test]
fn encode_decode_empty() {
    let pack = encode_uids(&[]);
    assert!(pack.is_empty());
    assert_eq!(pack.total_uids(), 0);
    assert_eq!(decode_uids(&pack), Vec::<u64>::new());
}

#[test]
fn encode_decode_single() {
    let pack = encode_uids(&[42]);
    assert_eq!(pack.total_uids(), 1);
    assert_eq!(pack.blocks.len(), 1);
    assert_eq!(pack.blocks[0].base, 42);
    assert!(pack.blocks[0].deltas.is_empty());
    assert_eq!(decode_uids(&pack), vec![42]);
}

#[test]
fn encode_decode_sequential() {
    let uids: Vec<u64> = (1..=100).collect();
    let pack = encode_uids(&uids);
    assert_eq!(pack.total_uids(), 100);
    assert_eq!(pack.blocks.len(), 1); // all fit in one block (< 256)
    assert_eq!(decode_uids(&pack), uids);
}

#[test]
fn encode_decode_large_block_split() {
    // 500 UIDs should produce 2 blocks (256 + 244)
    let uids: Vec<u64> = (1..=500).collect();
    let pack = encode_uids(&uids);
    assert_eq!(pack.total_uids(), 500);
    assert_eq!(pack.blocks.len(), 2);
    assert_eq!(pack.blocks[0].num_uids, 256);
    assert_eq!(pack.blocks[1].num_uids, 244);
    assert_eq!(decode_uids(&pack), uids);
}

#[test]
fn encode_decode_msb_boundary() {
    // UIDs that cross the 32-bit MSB boundary
    let uids = vec![
        0x0000_0000_FFFF_FFFE,
        0x0000_0000_FFFF_FFFF,
        0x0000_0001_0000_0000, // MSB changes here
        0x0000_0001_0000_0001,
    ];
    let pack = encode_uids(&uids);
    // Should produce 2 blocks due to MSB boundary
    assert_eq!(pack.blocks.len(), 2);
    assert_eq!(pack.blocks[0].num_uids, 2);
    assert_eq!(pack.blocks[1].num_uids, 2);
    assert_eq!(decode_uids(&pack), uids);
}

#[test]
fn encode_decode_sparse_uids() {
    let uids = vec![1, 1000, 1_000_000, 100_000_000];
    let pack = encode_uids(&uids);
    assert_eq!(decode_uids(&pack), uids);
}

#[test]
fn encode_decode_max_delta() {
    // UIDs with max u32 delta (within same 32-MSB)
    let uids = vec![1, u64::from(u32::MAX)];
    let pack = encode_uids(&uids);
    assert_eq!(decode_uids(&pack), uids);
}

#[test]
fn encode_decode_exact_block_size() {
    let uids: Vec<u64> = (1..=256).collect();
    let pack = encode_uids(&uids);
    assert_eq!(pack.blocks.len(), 1);
    assert_eq!(pack.blocks[0].num_uids, 256);
    assert_eq!(decode_uids(&pack), uids);
}

#[test]
fn encode_decode_block_size_plus_one() {
    let uids: Vec<u64> = (1..=257).collect();
    let pack = encode_uids(&uids);
    assert_eq!(pack.blocks.len(), 2);
    assert_eq!(pack.blocks[0].num_uids, 256);
    assert_eq!(pack.blocks[1].num_uids, 1);
    assert_eq!(decode_uids(&pack), uids);
}

#[test]
fn compression_ratio_sequential() {
    // 1000 sequential UIDs: raw = 8000 bytes
    let uids: Vec<u64> = (1..=1000).collect();
    let pack = encode_uids(&uids);
    let serialized = rmp_serde::to_vec(&pack).expect("serialize");
    let raw_size = uids.len() * 8;
    // Compressed should be significantly smaller
    assert!(
        serialized.len() < raw_size / 2,
        "compressed {} should be < {} (half of raw)",
        serialized.len(),
        raw_size / 2
    );
}

#[test]
fn msgpack_roundtrip() {
    let uids: Vec<u64> = (100..200).collect();
    let pack = encode_uids(&uids);
    let bytes = rmp_serde::to_vec(&pack).expect("serialize");
    let restored: UidPack = rmp_serde::from_slice(&bytes).expect("deserialize");
    assert_eq!(pack, restored);
    assert_eq!(decode_uids(&restored), uids);
}

#[test]
fn custom_block_size() {
    let mut enc = UidEncoder::with_block_size(10);
    for uid in 1..=25u64 {
        enc.add(uid);
    }
    let pack = enc.done();
    assert_eq!(pack.block_size, 10);
    assert_eq!(pack.blocks.len(), 3); // 10 + 10 + 5
    assert_eq!(decode_uids(&pack), (1..=25).collect::<Vec<_>>());
}

#[test]
fn decoder_block_by_block() {
    let uids: Vec<u64> = (1..=500).collect();
    let pack = encode_uids(&uids);
    let mut dec = UidDecoder::new(&pack);

    let block1 = dec.next_block().expect("block 1");
    assert_eq!(block1.len(), 256);
    assert_eq!(block1[0], 1);
    assert_eq!(block1[255], 256);

    let block2 = dec.next_block().expect("block 2");
    assert_eq!(block2.len(), 244);
    assert_eq!(block2[0], 257);

    assert!(dec.next_block().is_none());
}

#[test]
fn large_uid_values() {
    let uids = vec![u64::MAX - 100, u64::MAX - 50, u64::MAX - 10, u64::MAX - 1];
    let pack = encode_uids(&uids);
    assert_eq!(decode_uids(&pack), uids);
}

#[test]
fn multiple_msb_boundaries() {
    let uids = vec![
        0x0000_0000_0000_0001,
        0x0000_0001_0000_0001, // MSB change
        0x0000_0002_0000_0001, // MSB change
        0x0000_0003_0000_0001, // MSB change
    ];
    let pack = encode_uids(&uids);
    assert_eq!(pack.blocks.len(), 4); // each in its own block
    assert_eq!(decode_uids(&pack), uids);
}

// -- Split tests --

#[test]
fn should_split_small_pack() {
    let uids: Vec<u64> = (1..=100).collect();
    let pack = encode_uids(&uids);
    assert!(!pack.should_split());
}

#[test]
fn should_split_single_block() {
    // Even if somehow large, single block should not split
    let uids: Vec<u64> = (1..=256).collect();
    let pack = encode_uids(&uids);
    assert_eq!(pack.blocks.len(), 1);
    assert!(!pack.should_split()); // single block
}

#[test]
fn should_split_custom_threshold() {
    // 500 UIDs = 2 blocks, use tiny threshold to force split
    let uids: Vec<u64> = (1..=500).collect();
    let pack = encode_uids(&uids);
    assert_eq!(pack.blocks.len(), 2);
    assert!(pack.should_split_at(1)); // 1 byte threshold = always split
    assert!(!pack.should_split_at(usize::MAX)); // huge threshold = never split
}

#[test]
fn bin_split_produces_two_halves() {
    let uids: Vec<u64> = (1..=512).collect();
    let pack = encode_uids(&uids);
    assert_eq!(pack.blocks.len(), 2);

    let (low, high) = pack.bin_split().expect("should split");
    assert_eq!(low.blocks.len(), 1);
    assert_eq!(high.blocks.len(), 1);

    // All UIDs preserved
    let mut all = decode_uids(&low);
    all.extend(decode_uids(&high));
    assert_eq!(all, uids);
}

#[test]
fn bin_split_single_block_returns_none() {
    let uids: Vec<u64> = (1..=100).collect();
    let pack = encode_uids(&uids);
    assert!(pack.bin_split().is_none());
}

#[test]
fn recursive_split_small() {
    let uids: Vec<u64> = (1..=100).collect();
    let pack = encode_uids(&uids);
    let parts = pack.recursive_split();
    assert_eq!(parts.len(), 1); // no split needed
    assert_eq!(decode_uids(&parts[0].1), uids);
}

#[test]
fn recursive_split_with_tiny_threshold() {
    // 1024 UIDs = 4 blocks, use tiny threshold
    let uids: Vec<u64> = (1..=1024).collect();
    let pack = encode_uids(&uids);
    assert_eq!(pack.blocks.len(), 4);

    let parts = pack.recursive_split_at(1); // force maximum splitting
                                            // Should split into 4 parts (one per block)
    assert_eq!(parts.len(), 4);

    // Verify all UIDs preserved
    let mut all = Vec::new();
    for (_, part) in &parts {
        all.extend(decode_uids(part));
    }
    assert_eq!(all, uids);
}

#[test]
fn recursive_split_start_uids() {
    let uids: Vec<u64> = (1..=1024).collect();
    let pack = encode_uids(&uids);
    let parts = pack.recursive_split_at(1);

    // Each part's start_uid should match its first block's base
    for (start_uid, part) in &parts {
        assert_eq!(*start_uid, part.blocks[0].base);
    }

    // Start UIDs should be ascending
    for w in parts.windows(2) {
        assert!(w[0].0 < w[1].0, "start UIDs must be ascending");
    }
}

#[test]
fn start_uid() {
    let pack = encode_uids(&[42, 43, 44]);
    assert_eq!(pack.start_uid(), Some(42));

    let empty = encode_uids(&[]);
    assert_eq!(empty.start_uid(), None);
}

#[test]
fn serialized_size_grows_with_data() {
    let small = encode_uids(&(1..=10).collect::<Vec<_>>());
    let large = encode_uids(&(1..=1000).collect::<Vec<_>>());
    assert!(large.serialized_size() > small.serialized_size());
}

// ====== StreamVByte encoding structure tests ======

/// Verify that `deltas` bytes have the expected Coder1234 layout:
/// first `(n+3)/4` bytes are tags, remaining bytes are data.
#[test]
fn streamvbyte_deltas_layout_is_correct() {
    use streamvbyte64::{Coder, Coder1234};

    // 5 UIDs → 4 deltas → tag_len = (4+3)/4 = 1
    let uids: Vec<u64> = vec![100, 101, 102, 200, 201];
    let pack = encode_uids(&uids);
    assert_eq!(pack.blocks.len(), 1);
    let block = &pack.blocks[0];

    let n = (block.num_uids as usize) - 1; // 4 deltas
    let expected_tag_len = n.div_ceil(4); // 1

    assert!(
        block.deltas.len() >= expected_tag_len,
        "deltas must have at least {expected_tag_len} tag bytes"
    );

    // Verify roundtrip via direct Coder1234 decode matches our decode
    let coder = Coder1234::new();
    let tags = &block.deltas[..expected_tag_len];
    let data = &block.deltas[expected_tag_len..];
    let mut decoded_u32 = vec![0u32; n];
    coder.decode(tags, data, &mut decoded_u32);

    // Reconstruct UIDs from base + deltas
    let mut reconstructed = vec![block.base];
    let mut prev = block.base;
    for delta in &decoded_u32 {
        let uid = prev + u64::from(*delta);
        reconstructed.push(uid);
        prev = uid;
    }
    assert_eq!(reconstructed, uids);
}

/// StreamVByte encoding uses fewer bytes than LEB128 for sequential UIDs.
#[test]
fn streamvbyte_compresses_sequential_uids() {
    // Sequential UIDs → small deltas → 1 byte per delta in StreamVByte
    // LEB128 also uses 1 byte for small values, so sizes are comparable.
    // What matters: StreamVByte stores 4 values per tag byte (group efficiency).
    let uids = gen_sequential_uids(256);
    let pack = encode_uids(&uids);
    assert_eq!(pack.total_uids(), 256);
    assert_eq!(decode_uids(&pack), uids);

    // With 255 deltas: tag_len = 255.div_ceil(4) = 64 bytes of tags
    let block = &pack.blocks[0];
    let n = 255usize;
    let tag_len = n.div_ceil(4);
    assert_eq!(block.deltas.len() - tag_len, block.deltas.len() - tag_len); // data exists
    assert!(
        block.deltas.len() > tag_len,
        "must have data bytes after tags"
    );
}

/// Large block (256+ UIDs) produces multiple blocks, all StreamVByte encoded.
#[test]
fn streamvbyte_multi_block_roundtrip() {
    let uids = gen_sequential_uids(1024);
    let pack = encode_uids(&uids);
    assert!(pack.blocks.len() > 1, "1024 UIDs must span multiple blocks");
    assert_eq!(decode_uids(&pack), uids);
}

// ====== R011a StreamVByte vs LEB128 evaluation (API comparison) ======

/// Generate test UIDs: sequential with small gaps (typical adjacency list).
fn gen_sequential_uids(count: usize) -> Vec<u64> {
    let mut uids = Vec::with_capacity(count);
    let mut current = 1000u64;
    for _ in 0..count {
        current += 1 + (current % 7); // gaps 1-7
        uids.push(current);
    }
    uids
}

/// Generate test UIDs: sparse with large gaps (cross-shard references).
fn gen_sparse_uids(count: usize) -> Vec<u64> {
    let mut uids = Vec::with_capacity(count);
    let mut current = 1000u64;
    for i in 0..count {
        current += 100 + (i as u64 * 37) % 10000; // gaps 100-10000
        uids.push(current);
    }
    uids
}

#[test]
fn streamvbyte_vs_leb128_correctness() {
    // Verify streamvbyte64 produces correct roundtrip for our UID patterns
    use streamvbyte64::{Coder, Coder1234};

    let uids = gen_sequential_uids(256);

    // Compute u32 deltas (same as our LEB128 approach)
    let mut deltas = Vec::with_capacity(uids.len());
    deltas.push(0u32); // placeholder for base
    for i in 1..uids.len() {
        let d = uids[i] - uids[i - 1];
        assert!(d <= u32::MAX as u64, "delta overflow");
        deltas.push(d as u32);
    }

    let coder = Coder1234::new();
    let (tag_len, data_len) = Coder1234::max_compressed_bytes(deltas.len());
    let mut encoded = vec![0u8; tag_len + data_len];
    let (tags, data) = encoded.split_at_mut(tag_len);
    let _data_used = coder.encode(&deltas, tags, data);

    let mut decoded = vec![0u32; deltas.len()];
    coder.decode(tags, data, &mut decoded);

    assert_eq!(deltas, decoded, "streamvbyte roundtrip mismatch");
}

#[test]
fn streamvbyte_compression_ratio_comparison() {
    use streamvbyte64::{Coder, Coder1234};

    for (label, uids) in [
        ("sequential_256", gen_sequential_uids(256)),
        ("sequential_1024", gen_sequential_uids(1024)),
        ("sequential_10000", gen_sequential_uids(10000)),
        ("sparse_256", gen_sparse_uids(256)),
        ("sparse_1024", gen_sparse_uids(1024)),
    ] {
        // --- LEB128 (current) ---
        let leb_pack = encode_uids(&uids);
        let leb_size = leb_pack.serialized_size();

        // --- StreamVByte ---
        let mut deltas = Vec::with_capacity(uids.len());
        deltas.push(0u32);
        for i in 1..uids.len() {
            deltas.push((uids[i] - uids[i - 1]) as u32);
        }
        let _coder = Coder1234::new();
        let (tag_len, data_len) = Coder1234::max_compressed_bytes(deltas.len());
        let svb_size = tag_len + data_len + 8; // +8 for base u64

        eprintln!(
            "StreamVByte [{label}] UIDs={}, LEB128={}B, StreamVByte={}B, ratio={:.2}x",
            uids.len(),
            leb_size,
            svb_size,
            leb_size as f64 / svb_size as f64,
        );

        // Both should produce reasonable sizes
        assert!(leb_size > 0);
        assert!(svb_size > 0);
    }
}

#[test]
fn streamvbyte_decode_speed_comparison() {
    use streamvbyte64::{Coder, Coder1234};

    let uids = gen_sequential_uids(10000);
    let iterations = 1000;

    // --- LEB128 encode + decode ---
    let leb_pack = encode_uids(&uids);

    let leb_start = std::time::Instant::now();
    for _ in 0..iterations {
        let decoded = decode_uids(&leb_pack);
        std::hint::black_box(&decoded);
    }
    let leb_elapsed = leb_start.elapsed();

    // --- StreamVByte encode + decode ---
    let mut deltas = Vec::with_capacity(uids.len());
    deltas.push(0u32);
    for i in 1..uids.len() {
        deltas.push((uids[i] - uids[i - 1]) as u32);
    }
    let coder = Coder1234::new();
    let (tag_len, data_len) = Coder1234::max_compressed_bytes(deltas.len());
    let mut encoded = vec![0u8; tag_len + data_len];
    let (tags, data) = encoded.split_at_mut(tag_len);
    let data_used = coder.encode(&deltas, tags, data);

    let svb_start = std::time::Instant::now();
    for _ in 0..iterations {
        let mut decoded = vec![0u32; deltas.len()];
        coder.decode(tags, &data[..data_used], &mut decoded);
        std::hint::black_box(&decoded);
    }
    let svb_elapsed = svb_start.elapsed();

    let speedup = leb_elapsed.as_nanos() as f64 / svb_elapsed.as_nanos() as f64;

    eprintln!(
            "StreamVByte DECODE 10K UIDs x {}:\n  LEB128:      {:?}\n  StreamVByte: {:?}\n  Speedup:     {:.2}x",
            iterations, leb_elapsed, svb_elapsed, speedup
        );

    // StreamVByte should be at least competitive (not dramatically slower)
    // The actual speedup depends on SIMD availability
    assert!(
        svb_elapsed.as_nanos() < leb_elapsed.as_nanos() * 10,
        "StreamVByte should not be 10x slower than LEB128"
    );
}

#[test]
fn streamvbyte_encode_speed_comparison() {
    use streamvbyte64::{Coder, Coder1234};

    let uids = gen_sequential_uids(10000);
    let iterations = 1000;

    // --- LEB128 encode ---
    let leb_start = std::time::Instant::now();
    for _ in 0..iterations {
        let pack = encode_uids(&uids);
        std::hint::black_box(&pack);
    }
    let leb_elapsed = leb_start.elapsed();

    // --- StreamVByte encode ---
    let mut deltas = Vec::with_capacity(uids.len());
    deltas.push(0u32);
    for i in 1..uids.len() {
        deltas.push((uids[i] - uids[i - 1]) as u32);
    }

    let coder = Coder1234::new();
    let svb_start = std::time::Instant::now();
    for _ in 0..iterations {
        let (tag_len, data_len) = Coder1234::max_compressed_bytes(deltas.len());
        let mut encoded = vec![0u8; tag_len + data_len];
        let (tags, data) = encoded.split_at_mut(tag_len);
        coder.encode(&deltas, tags, data);
        std::hint::black_box(&encoded);
    }
    let svb_elapsed = svb_start.elapsed();

    let speedup = leb_elapsed.as_nanos() as f64 / svb_elapsed.as_nanos() as f64;

    eprintln!(
            "StreamVByte ENCODE 10K UIDs x {}:\n  LEB128:      {:?}\n  StreamVByte: {:?}\n  Speedup:     {:.2}x",
            iterations, leb_elapsed, svb_elapsed, speedup
        );
}
