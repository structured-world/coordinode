use super::*;

#[test]
fn merge_add_to_empty_base() {
    let op = PostingListMerge;
    let operand = encode_add(42);

    let result = op.merge(b"test", None, &[&operand]).expect("merge failed");
    let plist = PostingList::from_bytes(&result).expect("decode failed");
    assert_eq!(plist.as_slice(), &[42]);
}

#[test]
fn merge_add_to_existing() {
    let op = PostingListMerge;
    let base = PostingList::from_sorted(vec![10, 20, 30]);
    let base_bytes = base.to_bytes().expect("encode failed");
    let operand = encode_add(25);

    let result = op
        .merge(b"test", Some(&base_bytes), &[&operand])
        .expect("merge failed");
    let plist = PostingList::from_bytes(&result).expect("decode failed");
    assert_eq!(plist.as_slice(), &[10, 20, 25, 30]);
}

#[test]
fn merge_remove() {
    let op = PostingListMerge;
    let base = PostingList::from_sorted(vec![10, 20, 30]);
    let base_bytes = base.to_bytes().expect("encode failed");
    let operand = encode_remove(20);

    let result = op
        .merge(b"test", Some(&base_bytes), &[&operand])
        .expect("merge failed");
    let plist = PostingList::from_bytes(&result).expect("decode failed");
    assert_eq!(plist.as_slice(), &[10, 30]);
}

#[test]
fn merge_remove_nonexistent() {
    let op = PostingListMerge;
    let base = PostingList::from_sorted(vec![10, 20]);
    let base_bytes = base.to_bytes().expect("encode failed");
    let operand = encode_remove(99);

    let result = op
        .merge(b"test", Some(&base_bytes), &[&operand])
        .expect("merge failed");
    let plist = PostingList::from_bytes(&result).expect("decode failed");
    assert_eq!(plist.as_slice(), &[10, 20]);
}

#[test]
fn merge_add_batch() {
    let op = PostingListMerge;
    let operand = encode_add_batch(&[30, 10, 20]);

    let result = op.merge(b"test", None, &[&operand]).expect("merge failed");
    let plist = PostingList::from_bytes(&result).expect("decode failed");
    assert_eq!(plist.as_slice(), &[10, 20, 30]);
}

#[test]
fn merge_multiple_operands() {
    let op = PostingListMerge;
    let add1 = encode_add(10);
    let add2 = encode_add(30);
    let add3 = encode_add(20);
    let rm = encode_remove(10);

    let result = op
        .merge(b"test", None, &[&add1, &add2, &add3, &rm])
        .expect("merge failed");
    let plist = PostingList::from_bytes(&result).expect("decode failed");
    assert_eq!(plist.as_slice(), &[20, 30]);
}

#[test]
fn merge_duplicate_add_is_idempotent() {
    let op = PostingListMerge;
    let add = encode_add(42);

    let result = op
        .merge(b"test", None, &[&add, &add, &add])
        .expect("merge failed");
    let plist = PostingList::from_bytes(&result).expect("decode failed");
    assert_eq!(plist.as_slice(), &[42]);
}

#[test]
fn merge_re_merge_stability() {
    // MergeOperator contract: re-merging a merged result with no operands
    // must produce identical bytes.
    let op = PostingListMerge;
    let add1 = encode_add(10);
    let add2 = encode_add(20);

    let first = op
        .merge(b"test", None, &[&add1, &add2])
        .expect("first merge failed");
    let second = op
        .merge(b"test", Some(&first), &[])
        .expect("second merge failed");
    assert_eq!(&*first, &*second, "re-merge must be stable");
}

#[test]
fn merge_pre_merged_postinglist_as_operand() {
    // LSM partial compaction stores merged results as MergeOperand entries.
    // On subsequent compaction, the pre-merged PostingList appears as an
    // operand (not base_value). The merge function must detect and decode it.
    let op = PostingListMerge;

    // Simulate partial compaction: first merge produces a PostingList.
    let add1 = encode_add(10);
    let add2 = encode_add(20);
    let partial = op
        .merge(b"k", None, &[&add1, &add2])
        .expect("partial merge");

    // Now simulate a subsequent compaction where the partial result appears
    // as an operand alongside new merge operands.
    let add3 = encode_add(15);
    let result = op
        .merge(b"k", None, &[&partial, &add3])
        .expect("re-merge with pre-merged operand");
    let plist = PostingList::from_bytes(&result).expect("decode");
    assert_eq!(plist.as_slice(), &[10, 15, 20]);
}

#[test]
fn merge_pre_merged_with_base_and_new_operands() {
    // Pre-merged operand + base value + new operands all together.
    let op = PostingListMerge;

    // Base value from a PUT.
    let base = PostingList::from_sorted(vec![1, 5]);
    let base_bytes = base.to_bytes().expect("encode");

    // Partial merge result from a previous compaction.
    let partial_add = encode_add(10);
    let partial = op.merge(b"k", None, &[&partial_add]).expect("partial");

    // New operand.
    let new_add = encode_add(3);

    let result = op
        .merge(b"k", Some(&base_bytes), &[&partial, &new_add])
        .expect("combined merge");
    let plist = PostingList::from_bytes(&result).expect("decode");
    assert_eq!(plist.as_slice(), &[1, 3, 5, 10]);
}

#[test]
fn merge_output_is_valid_uidpack() {
    // Verify merge output is UidPack format, not raw Vec<u64> msgpack.
    let op = PostingListMerge;
    let batch = encode_add_batch(&[100, 200, 300, 400, 500]);

    let result = op.merge(b"test", None, &[&batch]).expect("merge");

    // Result must be decodable as UidPack.
    let pack: coordinode_core::graph::codec::UidPack =
        rmp_serde::from_slice(&result).expect("should be valid UidPack");
    assert_eq!(pack.total_uids(), 5);
    assert_eq!(pack.block_size, 256);

    // Decode back to UIDs for correctness.
    let uids = coordinode_core::graph::codec::decode_uids(&pack);
    assert_eq!(uids, vec![100, 200, 300, 400, 500]);
}

#[test]
fn merge_uidpack_smaller_than_raw() {
    // 500 sequential UIDs with small deltas — UidPack should compress well.
    let op = PostingListMerge;
    let uids: Vec<u64> = (0..500).map(|i| i * 3 + 1).collect();
    let batch = encode_add_batch(&uids);

    let result = op.merge(b"test", None, &[&batch]).expect("merge");

    // Raw Vec<u64> msgpack would be ~4009 bytes (500 × 8 + overhead).
    let raw_size = rmp_serde::to_vec(&uids).expect("raw").len();
    assert!(
        result.len() < raw_size,
        "UidPack ({} bytes) should be smaller than raw msgpack ({} bytes)",
        result.len(),
        raw_size
    );
}

#[test]
fn merge_large_posting_list_multiple_blocks() {
    // >256 UIDs forces multiple UidBlocks.
    let op = PostingListMerge;
    let uids: Vec<u64> = (0..600).collect();
    let batch = encode_add_batch(&uids);

    let result = op.merge(b"test", None, &[&batch]).expect("merge");
    let pack: coordinode_core::graph::codec::UidPack =
        rmp_serde::from_slice(&result).expect("should be valid UidPack");

    assert!(pack.blocks.len() >= 3, "600 UIDs should produce ≥3 blocks");
    assert_eq!(pack.total_uids(), 600);

    // Full roundtrip: decode and verify all UIDs present.
    let plist = PostingList::from_bytes(&result).expect("decode");
    assert_eq!(plist.len(), 600);
    assert_eq!(plist.as_slice()[0], 0);
    assert_eq!(plist.as_slice()[599], 599);
}

#[test]
fn merge_pre_merged_uidpack_as_operand() {
    // After compaction, a UidPack-encoded result appears as an operand.
    // The merge function must detect it (non-standard tag byte) and decode.
    let op = PostingListMerge;

    // First merge produces a UidPack-encoded result.
    let add1 = encode_add_batch(&[10, 20, 30]);
    let partial = op.merge(b"k", None, &[&add1]).expect("partial merge");

    // Verify partial is valid UidPack (not raw tag byte).
    let first_byte = partial[0];
    assert!(
        first_byte != TAG_ADD && first_byte != TAG_REMOVE && first_byte != TAG_ADD_BATCH,
        "UidPack first byte (0x{first_byte:02x}) must not collide with operand tags"
    );

    // Now the partial UidPack appears as an operand in subsequent compaction.
    let add2 = encode_add(15);
    let result = op
        .merge(b"k", None, &[&partial, &add2])
        .expect("re-merge with UidPack operand");
    let plist = PostingList::from_bytes(&result).expect("decode");
    assert_eq!(plist.as_slice(), &[10, 15, 20, 30]);
}

#[test]
fn invalid_operand_tag_returns_error() {
    let op = PostingListMerge;
    // 0xFF is not a valid tag AND not valid UidPack
    let bad = vec![0xFF, 0, 0, 0, 0, 0, 0, 0, 0];

    assert!(op.merge(b"test", None, &[&bad]).is_err());
}

#[test]
fn empty_operand_returns_error() {
    let op = PostingListMerge;
    let empty: &[u8] = &[];

    assert!(op.merge(b"test", None, &[empty]).is_err());
}

#[test]
fn truncated_add_operand_returns_error() {
    let op = PostingListMerge;
    let truncated = vec![TAG_ADD, 0, 0, 0]; // only 4 bytes, need 8

    assert!(op.merge(b"test", None, &[&truncated]).is_err());
}
