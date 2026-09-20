//! What may be folded before the base is known, and what happens if the wrong
//! operator is allowed to.
//!
//! The engine folds a prefix of a chain where it cannot prove the base only
//! for an operator that declares it composes. These tests state both halves of
//! that rule as executable facts: the counter's composition holds however the
//! chain is bracketed and however deep the nesting goes, and the two operators
//! that decline would lose data if they did not, which is why their refusal is
//! a property to test rather than a default nobody looked at.

use super::*;

/// An independent model of a counter chain: a plain sum, with no encoding,
/// no base and no folding.
fn sum(deltas: &[i64]) -> i64 {
    deltas.iter().sum()
}

fn deltas(values: &[i64]) -> Vec<Vec<u8>> {
    values.iter().map(|d| encode_counter_delta(*d)).collect()
}

fn refs(encoded: &[Vec<u8>]) -> Vec<&[u8]> {
    encoded.iter().map(|v| v.as_slice()).collect()
}

/// A composite is an operand like any other, so composing composites has to
/// give what composing the originals gives.
///
/// The engine can fold the same key at several levels over a key's life: what
/// one compaction leaves is an operand the next one may fold again, with more
/// operands beside it. Folding once proves nothing about that; this folds the
/// result of a fold, three levels deep, and then applies it to a base.
#[test]
fn a_composite_of_composites_folds_to_the_same_sum() {
    let merger = CounterMerge;
    let base = 1_000i64;
    let values = [3i64, -1, 8, 2, -5, 13];

    let encoded = deltas(&values);
    let all = refs(&encoded);

    // Level 1: two composites, each over a contiguous run.
    let left = merger.merge(b"counter:k", None, &all[0..3]).expect("fold");
    let right = merger.merge(b"counter:k", None, &all[3..6]).expect("fold");

    // Level 2: one composite over the two composites.
    let both = merger
        .merge(b"counter:k", None, &[&left, &right])
        .expect("fold of folds");

    // Level 3: that composite folded again on its own, which must be identity.
    let again = merger
        .merge(b"counter:k", None, &[&both])
        .expect("fold of a fold of folds");

    let base_bytes = base.to_le_bytes();
    let folded = merger
        .merge(b"counter:k", Some(&base_bytes), &[&again])
        .expect("apply to the base");

    assert_eq!(
        decode_counter(&folded).expect("decode"),
        base + sum(&values),
        "nesting the folds must not change the sum"
    );
}

/// The operator is promised a contiguous prefix, and this is what that promise
/// is worth: dropping one operand from the middle changes the answer.
///
/// The engine states that the operands it hands over are a real prefix of the
/// real chain and never a subset with gaps. Nothing in the operator could
/// detect a gap, so the guarantee is the engine's alone, and a test that shows
/// the difference is what keeps it from being read as a formality.
#[test]
fn a_subset_with_a_gap_is_not_the_same_as_a_prefix() {
    let merger = CounterMerge;
    let values = [5i64, 7, -2];
    let encoded = deltas(&values);
    let all = refs(&encoded);

    let prefix = merger.merge(b"counter:k", None, &all[0..2]).expect("fold");
    assert_eq!(decode_counter(&prefix).expect("decode"), 12);

    // The same two operands with the middle one missing: a subset, not a
    // prefix, and a different value.
    let gapped = merger
        .merge(b"counter:k", None, &[all[0], all[2]])
        .expect("fold");
    assert_ne!(
        decode_counter(&gapped).expect("decode"),
        decode_counter(&prefix).expect("decode"),
        "a gap in the operands changes the result, so only a prefix may be folded"
    );
}

/// Why the posting list declines: a removal folded against an assumed-empty
/// base becomes an empty set, and the removal is gone when the real base
/// arrives.
///
/// This is the data loss the engine's contract exists to prevent, written as a
/// model rather than as prose: fold a chain that removes a member, then apply
/// the result to the base it was really meant for, and the member is back.
#[test]
fn a_removal_folded_without_its_base_would_lose_the_removal() {
    let merger = PostingListMerge;

    // The real base holds three members.
    let base = merger
        .merge(
            b"adj:T:out:1",
            None,
            &[&encode_add(1), &encode_add(2), &encode_add(3)],
        )
        .expect("build the base");

    let remove_two = encode_remove(2);

    // What the chain means against its real base: 2 is gone.
    let honest = merger
        .merge(b"adj:T:out:1", Some(&base), &[&remove_two])
        .expect("apply to the real base");
    let honest = PostingList::from_bytes(&honest).expect("decode");
    assert_eq!(honest.as_slice(), &[1, 3], "the removal takes effect");

    // What folding the same operand without a base would produce, and what
    // happens when that composite later meets the base it was meant for.
    let composed = merger
        .merge(b"adj:T:out:1", None, &[&remove_two])
        .expect("fold with no base");
    let resurrected = merger
        .merge(b"adj:T:out:1", Some(&base), &[&composed])
        .expect("apply the composite to the base");
    let resurrected = PostingList::from_bytes(&resurrected).expect("decode");

    assert_eq!(
        resurrected.as_slice(),
        &[1, 2, 3],
        "the removal is lost, which is why this operator does not compose"
    );
    assert!(
        !merger.composes_operands(),
        "and so it must never declare that it does"
    );
}

/// Why the document declines: a patch folded against an assumed-empty base
/// becomes a whole record, and replaces the real one instead of patching it.
#[test]
fn a_patch_folded_without_its_base_would_replace_the_record() {
    let merger = DocumentMerge;

    let mut original = NodeRecord::new("User");
    original.set_extra(
        "name",
        coordinode_core::graph::types::Value::String("Alice".to_string()),
    );
    let base = encode_node_record(&original).expect("encode the base");

    let patch = DocDelta::SetPath {
        target: PathTarget::Extra,
        path: vec!["age".to_string()],
        value: rmpv::Value::Integer(30.into()),
    };
    let operand = patch.encode().expect("encode the delta");

    // Against its real base the patch adds a field and keeps the rest.
    let honest = merger
        .merge(b"node:00:01", Some(&base), &[&operand])
        .expect("apply to the real base");
    let honest = decode_node_record(&honest).expect("decode");
    assert!(
        honest.get_extra("name").is_some() && honest.get_extra("age").is_some(),
        "patching keeps what was there and adds what is new"
    );

    // Folded without a base it becomes a record of its own, and that record
    // supersedes the real one when they meet.
    let composed = merger
        .merge(b"node:00:01", None, &[&operand])
        .expect("fold with no base");
    let replaced = merger
        .merge(b"node:00:01", Some(&base), &[&composed])
        .expect("apply the composite to the base");
    let replaced = decode_node_record(&replaced).expect("decode");

    assert!(
        replaced.get_extra("name").is_none(),
        "the original field is gone, which is why this operator does not compose"
    );
    assert!(
        !merger.composes_operands(),
        "and so it must never declare that it does"
    );
}

/// A corrupt operand is refused by every operator rather than interpreted.
///
/// Each operator has a fallback arm that reads an operand written by an older
/// engine (a bare msgpack record, a composite a pre-contract compaction left
/// behind). Those arms decode, and decoding is what separates an old format
/// from damage: input that decodes as neither is an error and never a value
/// the caller cannot tell from a real one.
#[test]
fn every_operator_refuses_an_operand_it_cannot_decode() {
    let garbage: &[u8] = &[0xfe, 0x7f, 0x11, 0x22, 0x33];

    assert!(
        PostingListMerge
            .merge(b"adj:T:out:1", None, &[garbage])
            .is_err(),
        "the posting list refuses what is neither a tag it knows nor a list"
    );
    assert!(
        DocumentMerge
            .merge(b"node:00:01", None, &[garbage])
            .is_err(),
        "the document refuses what is neither its prefix nor a record"
    );
    assert!(
        CounterMerge.merge(b"counter:k", None, &[garbage]).is_err(),
        "the counter refuses anything that is not eight bytes"
    );

    // An empty operand is damage in every encoding, not an identity element.
    let empty: &[u8] = &[];
    assert!(
        PostingListMerge
            .merge(b"adj:T:out:1", None, &[empty])
            .is_err()
    );
    assert!(DocumentMerge.merge(b"node:00:01", None, &[empty]).is_err());
    assert!(CounterMerge.merge(b"counter:k", None, &[empty]).is_err());
}
