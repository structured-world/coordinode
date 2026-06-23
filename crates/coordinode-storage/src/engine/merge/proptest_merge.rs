use super::*;
use proptest::prelude::*;

/// A single merge operation for proptest.
#[derive(Debug, Clone)]
enum MergeOp {
    Add(u64),
    Remove(u64),
}

/// Strategy: generate a vec of 1..200 Add/Remove operations
/// with UIDs in 0..500 range to ensure overlap and collisions.
fn merge_ops_strategy() -> impl Strategy<Value = Vec<MergeOp>> {
    prop::collection::vec(
        prop_oneof![
            (0..500u64).prop_map(MergeOp::Add),
            (0..500u64).prop_map(MergeOp::Remove),
        ],
        1..200,
    )
}

/// Apply operations to a reference HashSet to compute expected result.
fn expected_uids(ops: &[MergeOp]) -> Vec<u64> {
    let mut set = std::collections::BTreeSet::new();
    for op in ops {
        match op {
            MergeOp::Add(uid) => {
                set.insert(*uid);
            }
            MergeOp::Remove(uid) => {
                set.remove(uid);
            }
        }
    }
    set.into_iter().collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn fuzz_merge_sorted_unique(ops in merge_ops_strategy()) {
        let merger = PostingListMerge;

        // Encode all operations as merge operands.
        let operands: Vec<Vec<u8>> = ops
            .iter()
            .map(|op| match op {
                MergeOp::Add(uid) => encode_add(*uid),
                MergeOp::Remove(uid) => encode_remove(*uid),
            })
            .collect();
        let operand_refs: Vec<&[u8]> = operands.iter().map(|v| v.as_slice()).collect();

        // Apply all operands at once (full merge).
        let result = merger
            .merge(b"test-key", None, &operand_refs)
            .expect("merge must not fail on valid operands");
        let plist = PostingList::from_bytes(&result).expect("decode merged result");

        let expected = expected_uids(&ops);

        // Invariant 1: result matches reference set.
        prop_assert_eq!(plist.as_slice(), expected.as_slice(),
            "merged posting list does not match expected set");

        // Invariant 2: sorted (redundant with BTreeSet, but explicitly checked).
        let slice = plist.as_slice();
        for i in 1..slice.len() {
            prop_assert!(slice[i - 1] < slice[i],
                "not sorted at {}: {} >= {}", i, slice[i-1], slice[i]);
        }
    }

    #[test]
    fn fuzz_incremental_merge_matches_full(ops in merge_ops_strategy()) {
        // Apply operands one at a time (incremental) vs all at once (full).
        // Both must produce identical result.
        let merger = PostingListMerge;

        let operands: Vec<Vec<u8>> = ops
            .iter()
            .map(|op| match op {
                MergeOp::Add(uid) => encode_add(*uid),
                MergeOp::Remove(uid) => encode_remove(*uid),
            })
            .collect();

        // Full merge.
        let operand_refs: Vec<&[u8]> = operands.iter().map(|v| v.as_slice()).collect();
        let full = merger
            .merge(b"k", None, &operand_refs)
            .expect("full merge");

        // Incremental merge: apply one operand at a time, feeding result as base.
        let mut base: Option<Vec<u8>> = None;
        for op in &operands {
            let b_ref = base.as_deref();
            let result = merger
                .merge(b"k", b_ref, &[op.as_slice()])
                .expect("incremental merge");
            base = Some(result.to_vec());
        }

        let incremental = base.unwrap_or_default();
        prop_assert_eq!(&*full, incremental.as_slice(),
            "full merge and incremental merge must produce identical output");
    }

    #[test]
    fn fuzz_batch_add_matches_individual_adds(uids in prop::collection::vec(0..1000u64, 1..100)) {
        let merger = PostingListMerge;

        // Batch add.
        let batch_op = encode_add_batch(&uids);
        let batch_result = merger
            .merge(b"k", None, &[&batch_op])
            .expect("batch merge");

        // Individual adds.
        let individual_ops: Vec<Vec<u8>> = uids.iter().map(|&u| encode_add(u)).collect();
        let individual_refs: Vec<&[u8]> = individual_ops.iter().map(|v| v.as_slice()).collect();
        let individual_result = merger
            .merge(b"k", None, &individual_refs)
            .expect("individual merge");

        prop_assert_eq!(&*batch_result, &*individual_result,
            "batch add must produce same result as individual adds");
    }
}
