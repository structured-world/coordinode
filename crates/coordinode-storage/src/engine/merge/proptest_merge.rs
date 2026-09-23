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

/// Deltas over a small key space, so paths collide and the order-dependent
/// operations actually meet each other.
fn doc_deltas_strategy() -> impl Strategy<Value = Vec<DocDelta>> {
    let key = prop_oneof![Just("a".to_string()), Just("b".to_string())];
    let value = prop_oneof![
        (0..4i64).prop_map(|n| rmpv::Value::Integer(n.into())),
        Just(rmpv::Value::Boolean(true)),
    ];
    let path = key.prop_map(|k| vec![k]);

    prop::collection::vec(
        prop_oneof![
            (path.clone(), value.clone()).prop_map(|(path, value)| DocDelta::SetPath {
                target: PathTarget::Extra,
                path,
                value,
            }),
            path.clone().prop_map(|path| DocDelta::DeletePath {
                target: PathTarget::Extra,
                path,
            }),
            (path.clone(), value.clone()).prop_map(|(path, value)| DocDelta::ArrayPush {
                target: PathTarget::Extra,
                path,
                value,
            }),
            (path.clone(), value.clone()).prop_map(|(path, value)| DocDelta::ArrayPull {
                target: PathTarget::Extra,
                path,
                value,
            }),
            (path.clone(), value).prop_map(|(path, value)| DocDelta::ArrayAddToSet {
                target: PathTarget::Extra,
                path,
                value,
            }),
            (path, 0..4i64).prop_map(|(path, amount)| DocDelta::Increment {
                target: PathTarget::Extra,
                path,
                amount: amount as f64,
            }),
        ],
        1..24,
    )
}

/// Deltas that are usually small and occasionally enormous, so that a chain
/// leaving i64 is generated often enough to be worth asserting about.
fn counter_deltas_strategy() -> impl Strategy<Value = Vec<i64>> {
    prop::collection::vec(
        prop_oneof![
            8 => -1_000..1_000i64,
            1 => Just(i64::MAX),
            1 => Just(i64::MIN),
            1 => Just(i64::MAX / 2),
        ],
        1..32,
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

    /// A counter fold is the sum the same deltas make in an independent model,
    /// and it is that sum however the compaction split the chain.
    ///
    /// The model is an i64 added up with the same checked arithmetic and
    /// nothing else: no base, no encoding, no folding. Deltas are drawn near
    /// the boundaries as well as small, so a chain that would leave the range
    /// is generated and both sides have to refuse it rather than disagree
    /// about what it became.
    #[test]
    fn fuzz_counter_fold_matches_a_plain_sum(deltas in counter_deltas_strategy()) {
        let merger = CounterMerge;

        let model = deltas.iter().try_fold(0i64, |acc, d| acc.checked_add(*d));

        let operands: Vec<Vec<u8>> = deltas.iter().map(|d| encode_counter_delta(*d)).collect();
        let operand_refs: Vec<&[u8]> = operands.iter().map(|v| v.as_slice()).collect();
        let full = merger.merge(b"counter:k", None, &operand_refs);

        match model {
            Some(expected) => {
                let folded = full.expect("a sum inside the range must fold");
                prop_assert_eq!(decode_counter(&folded).expect("decode"), expected);

                // And one delta at a time, the way a compaction leaves it.
                let mut base: Option<Vec<u8>> = None;
                for operand in &operands {
                    let step = merger
                        .merge(b"counter:k", base.as_deref(), &[operand.as_slice()])
                        .expect("each step stays inside the range");
                    base = Some(step.to_vec());
                }
                let incremental = base.expect("at least one delta");
                prop_assert_eq!(decode_counter(&incremental).expect("decode"), expected,
                    "a counter folded one delta at a time is the same sum");
            }
            None => {
                prop_assert!(full.is_err(),
                    "a running total that leaves i64 must be refused, not folded back in");
            }
        }
    }

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

    /// A document fold means the same whatever the compaction split it into.
    ///
    /// The engine may call the operator with any prefix of a chain already
    /// folded into the base, because that is what a compaction leaves behind.
    /// Posting lists are checked for this above; documents carry the harder
    /// cases, an array whose order is observable, a pull that removes the
    /// first match only, a set-add that must not, and a numeric increment
    /// that goes through an encode and a decode of the record at every split.
    ///
    /// The comparison is of the bytes, which is what the engine asks for:
    /// repeated merging must produce identical bytes, or two replicas that
    /// compacted differently could not be compared by checksum.
    #[test]
    fn fuzz_document_partial_fold_matches_full(deltas in doc_deltas_strategy()) {
        let merger = DocumentMerge;

        let operands: Vec<Vec<u8>> = deltas
            .iter()
            .map(|d| d.encode().expect("encode delta"))
            .collect();
        let operand_refs: Vec<&[u8]> = operands.iter().map(|v| v.as_slice()).collect();

        let full = merger
            .merge(b"node:00:00000001", None, &operand_refs)
            .expect("full fold");

        let mut base: Option<Vec<u8>> = None;
        for operand in &operands {
            let folded = merger
                .merge(b"node:00:00000001", base.as_deref(), &[operand.as_slice()])
                .expect("one-at-a-time fold");
            base = Some(folded.to_vec());
        }
        let incremental = base.expect("at least one delta");

        prop_assert_eq!(&full[..], &incremental[..],
            "a compaction that folded a prefix of the chain must leave the same bytes");
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
