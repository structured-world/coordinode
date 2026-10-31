//! Membership, entry effects and unit combination: the functions both
//! maintenance profiles share.

use super::*;
use crate::index::encoding::{encode_entry_key, encode_tuple};

/// The generation of the index the tests derive entries of.
const GEN: GenerationId = GenerationId::from_raw(41);

fn prop(name: &str, field: Option<u32>) -> PropertyRef {
    PropertyRef {
        field,
        name: name.into(),
    }
}

fn interp(unique: bool, sparse: bool, filter: Option<MembershipFilter>) -> IndexInterpretation {
    IndexInterpretation {
        codec: KEY_CODEC,
        generation: GEN,
        unique,
        sparse,
        properties: vec![prop("email", Some(1))],
        filter,
    }
}

fn s(v: &str) -> Value {
    Value::String(v.into())
}

fn lookup<'a>(pairs: &'a [(&'a str, Value)]) -> impl Fn(&PropertyRef) -> Option<Value> + 'a {
    move |p| {
        pairs
            .iter()
            .find(|(n, _)| *n == p.name)
            .map(|(_, v)| v.clone())
    }
}

/// A missing property is null, so a non-sparse index holds it and a sparse
/// one skips the node.
#[test]
fn a_missing_property_is_null_and_sparse_skips_it() {
    let dense = interp(false, false, None);
    assert_eq!(dense.membership(&lookup(&[])), Some(vec![Value::Null]));
    let sparse = interp(false, true, None);
    assert_eq!(sparse.membership(&lookup(&[])), None);
    assert_eq!(
        sparse.membership(&lookup(&[("email", s("a"))])),
        Some(vec![s("a")])
    );
}

/// Each filter admits exactly the values of its own type: an integer filter
/// does not admit the string of the same digits, nor a float.
#[test]
fn filters_compare_with_their_declared_type() {
    let status = prop("status", Some(2));
    let cases: [(MembershipFilter, Value, bool); 7] = [
        (
            MembershipFilter::EqualsString(status.clone(), "on".into()),
            s("on"),
            true,
        ),
        (
            MembershipFilter::EqualsString(status.clone(), "on".into()),
            s("off"),
            false,
        ),
        (
            MembershipFilter::EqualsInt(status.clone(), 5),
            Value::Int(5),
            true,
        ),
        (
            MembershipFilter::EqualsInt(status.clone(), 5),
            s("5"),
            false,
        ),
        (
            MembershipFilter::EqualsInt(status.clone(), 5),
            Value::Float(5.0),
            false,
        ),
        (
            MembershipFilter::EqualsBool(status.clone(), true),
            Value::Bool(true),
            true,
        ),
        (MembershipFilter::Exists(status.clone()), Value::Null, false),
    ];
    for (filter, value, admitted) in cases {
        let index = interp(false, false, Some(filter.clone()));
        let got = index.membership(&lookup(&[("email", s("a")), ("status", value.clone())]));
        assert_eq!(got.is_some(), admitted, "{filter:?} on {value:?}");
    }
    let exists = interp(false, false, Some(MembershipFilter::Exists(status)));
    assert_eq!(exists.membership(&lookup(&[("email", s("a"))])), None);
}

/// A record answers by field id first, then by name for a property kept
/// outside the dictionary.
#[test]
fn a_record_answers_by_field_id_then_by_name() {
    let mut record = NodeRecord::new(crate::schema::label_id::FIRST_LABEL_ID);
    record.set(1, s("by-id"));
    record.set_extra("email", s("by-name"));
    let index = interp(false, false, None);
    assert_eq!(index.record_membership(&record), Some(vec![s("by-id")]));

    let unbound = IndexInterpretation {
        properties: vec![prop("email", None)],
        ..interp(false, false, None)
    };
    assert_eq!(unbound.record_membership(&record), Some(vec![s("by-name")]));
}

/// A list indexes each distinct element; a value with no key has no entry.
#[test]
fn lists_index_each_distinct_element() {
    let list = Value::Array(vec![s("b"), s("a"), s("b")]);
    let got = tuples(&[list]);
    let mut want = vec![
        encode_tuple(&[s("a")]).expect("a"),
        encode_tuple(&[s("b")]).expect("b"),
    ];
    want.sort();
    assert_eq!(got, want);
    assert!(tuples(&[Value::Float(f64::NAN)]).is_empty());
}

/// Entering puts every tuple, leaving deletes every tuple, and a change
/// touches only the tuples that differ.
#[test]
fn membership_effects_are_the_difference_of_the_two_states() {
    let index = interp(false, false, None);
    let a = encode_tuple(&[s("a")]).expect("a");
    let c = encode_tuple(&[s("c")]).expect("c");
    let key = |t: &[u8]| encode_entry_key(GEN, t, 7);
    let node = EntryOwner::node(7);

    let enter = index.membership_effects(node, None, Some(&[s("a")]));
    assert_eq!(
        enter,
        vec![EntryEffect {
            key: key(&a),
            value: Some(Vec::new())
        }]
    );

    let leave = index.membership_effects(node, Some(&[s("a")]), None);
    assert_eq!(
        leave,
        vec![EntryEffect {
            key: key(&a),
            value: None
        }]
    );

    let old = [Value::Array(vec![s("a"), s("b")])];
    let new = [Value::Array(vec![s("b"), s("c")])];
    let change = index.membership_effects(node, Some(&old), Some(&new));
    assert_eq!(
        change,
        vec![
            EntryEffect {
                key: key(&a),
                value: None
            },
            EntryEffect {
                key: key(&c),
                value: Some(Vec::new())
            },
        ]
    );
    assert!(
        index
            .membership_effects(node, Some(&[s("b")]), Some(&[s("b")]))
            .is_empty()
    );
}

/// A unique entry is keyed by value and owner, as any entry is: two nodes
/// holding one value are two entries, and admission, not the key, decides
/// whether both may.
#[test]
fn a_unique_entry_is_keyed_by_its_owner() {
    let index = interp(true, false, None);
    let a = encode_tuple(&[s("a")]).expect("a");
    let nine = index.membership_effects(EntryOwner::node(9), None, Some(&[s("a")]));
    assert_eq!(
        nine,
        vec![EntryEffect {
            key: encode_entry_key(GEN, &a, 9),
            value: Some(Vec::new()),
        }]
    );
    let ten = index.membership_effects(EntryOwner::node(10), None, Some(&[s("a")]));
    assert_ne!(nine[0].key, ten[0].key);
}

/// A version of a temporal node has entries of its own, keyed by node and
/// valid_from: two versions holding one value are two entries, and a
/// version leaving a value removes only its own.
#[test]
fn a_version_entry_carries_its_valid_from() {
    use crate::index::encoding::encode_version_entry_key;
    let index = interp(false, false, None);
    let a = encode_tuple(&[s("a")]).expect("a");
    let first = index.membership_effects(EntryOwner::version(7, 100), None, Some(&[s("a")]));
    let second = index.membership_effects(EntryOwner::version(7, 200), None, Some(&[s("a")]));
    assert_eq!(
        first,
        vec![EntryEffect {
            key: encode_version_entry_key(GEN, &a, 7, 100),
            value: Some(Vec::new()),
        }]
    );
    assert_ne!(first[0].key, second[0].key, "one entry per version");
    assert_eq!(
        index.membership_effects(EntryOwner::version(7, 100), Some(&[s("a")]), None),
        vec![EntryEffect {
            key: encode_version_entry_key(GEN, &a, 7, 100),
            value: None,
        }]
    );
}

/// A version of a temporal node holds a unique value through its own entry:
/// a correction of that version leaving the value removes only its entry,
/// and another version holding the value keeps its own.
#[test]
fn a_version_leaving_a_unique_value_removes_only_its_entry() {
    use crate::index::encoding::encode_version_entry_key;
    let index = interp(true, false, None);
    let a = encode_tuple(&[s("a")]).expect("a");
    let b = encode_tuple(&[s("b")]).expect("b");
    assert_eq!(
        index.membership_effects(
            EntryOwner::version(9, 100),
            Some(&[s("a")]),
            Some(&[s("b")])
        ),
        vec![
            EntryEffect {
                key: encode_version_entry_key(GEN, &a, 9, 100),
                value: None,
            },
            EntryEffect {
                key: encode_version_entry_key(GEN, &b, 9, 100),
                value: Some(Vec::new()),
            },
        ]
    );
}

fn unit(key: &[u8], value: Option<Vec<u8>>) -> EntryEffect {
    EntryEffect {
        key: key.to_vec(),
        value,
    }
}

/// For one key the later effect wins; keys keep the order they were first
/// touched in.
#[test]
fn later_effects_replace_earlier_ones() {
    let got = combine_unit_effects([
        unit(b"u", Some(Vec::new())),
        unit(b"n", Some(Vec::new())),
        unit(b"u", None),
        unit(b"n", None),
    ]);
    assert_eq!(
        got,
        vec![
            EntryEffect {
                key: b"u".to_vec(),
                value: None
            },
            EntryEffect {
                key: b"n".to_vec(),
                value: None
            },
        ]
    );
}

mod resolve {
    use super::*;
    use crate::txn::proposal::{
        DerivedIndexWork, DerivedSource, IndexBinding, Mutation, PartitionId,
    };

    fn work(
        index: &IndexInterpretation,
        node_id: u64,
        old: Option<Vec<Value>>,
        new: DerivedSource,
    ) -> Mutation {
        Mutation::Derive(DerivedIndexWork {
            binding: IndexBinding {
                epoch: 1,
                interpretation: index.clone(),
            },
            node_id,
            valid_from: None,
            old,
            new,
        })
    }

    fn record_put(node_id: u64, email: &str) -> Mutation {
        let mut record = NodeRecord::new(crate::schema::label_id::FIRST_LABEL_ID);
        record.set(1, s(email));
        Mutation::Put {
            partition: PartitionId::Node,
            key: format!("node:0:{node_id}").into_bytes(),
            value: record.to_msgpack().expect("record"),
        }
    }

    fn idx_put(key: Vec<u8>, value: Vec<u8>) -> Mutation {
        Mutation::Put {
            partition: PartitionId::Idx,
            key,
            value,
        }
    }

    /// A unit without DERIVED work is returned unchanged, without a copy.
    #[test]
    fn a_unit_without_derived_work_is_borrowed() {
        let unit = vec![record_put(1, "a")];
        assert!(matches!(
            resolve_unit(&unit, 16).expect("resolve"),
            std::borrow::Cow::Borrowed(_)
        ));
    }

    /// The new membership is extracted from the unit's own node record put,
    /// and the old one from the sealed input: an update moves the entry.
    #[test]
    fn membership_comes_from_the_units_record_and_the_sealed_input() {
        let index = interp(false, false, None);
        let unit = vec![
            record_put(7, "new"),
            work(
                &index,
                7,
                Some(vec![s("old")]),
                DerivedSource::UnitRecord(0),
            ),
        ];
        let resolved = resolve_unit(&unit, 16).expect("resolve");
        let old = encode_tuple(&[s("old")]).expect("old");
        let new = encode_tuple(&[s("new")]).expect("new");
        assert_eq!(
            resolved.as_ref(),
            &[
                unit[0].clone(),
                Mutation::Delete {
                    partition: PartitionId::Idx,
                    key: encode_entry_key(GEN, &old, 7),
                },
                idx_put(encode_entry_key(GEN, &new, 7), Vec::new()),
            ]
        );
    }

    /// Work sealed for a version of a temporal node derives that version's
    /// entry, keyed by node and valid_from, at every member.
    #[test]
    fn version_work_derives_the_versions_entry() {
        use crate::index::encoding::encode_version_entry_key;
        let index = interp(false, false, None);
        let Mutation::Derive(mut version_work) =
            work(&index, 7, None, DerivedSource::Values(Some(vec![s("v")])))
        else {
            unreachable!("work builds a Derive mutation")
        };
        version_work.valid_from = Some(-3);
        let tuple = encode_tuple(&[s("v")]).expect("v");
        assert_eq!(
            resolve_unit(&[Mutation::Derive(version_work)], 16)
                .expect("resolve")
                .as_ref(),
            &[idx_put(
                encode_version_entry_key(GEN, &tuple, 7, -3),
                Vec::new()
            )]
        );
    }

    /// A unique value one node releases and another takes in the same unit
    /// ends held by the taker alone, whichever change came first.
    #[test]
    fn a_unique_value_changing_hands_ends_with_the_taker() {
        let index = interp(true, false, None);
        let tuple = encode_tuple(&[s("v")]).expect("v");
        let take = work(&index, 2, None, DerivedSource::Values(Some(vec![s("v")])));
        let release = work(&index, 1, Some(vec![s("v")]), DerivedSource::Values(None));
        for unit in [
            vec![take.clone(), release.clone()],
            vec![release.clone(), take.clone()],
        ] {
            let resolved = resolve_unit(&unit, 16).expect("resolve");
            assert!(resolved.contains(&idx_put(encode_entry_key(GEN, &tuple, 2), Vec::new())));
            assert!(resolved.contains(&Mutation::Delete {
                partition: PartitionId::Idx,
                key: encode_entry_key(GEN, &tuple, 1),
            }));
            assert_eq!(resolved.len(), 2);
        }
    }

    /// Work whose source is not a node record put, or whose record does not
    /// decode, is refused rather than resolved from anything else.
    #[test]
    fn a_bad_source_or_record_is_refused() {
        let index = interp(false, false, None);
        let not_a_record = vec![
            idx_put(b"idx:x".to_vec(), Vec::new()),
            work(&index, 1, None, DerivedSource::UnitRecord(0)),
        ];
        assert_eq!(
            resolve_unit(&not_a_record, 16),
            Err(DeriveError::BadSource(0))
        );
        let garbage = vec![
            Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:0:1".to_vec(),
                value: vec![0xC1],
            },
            work(&index, 1, None, DerivedSource::UnitRecord(0)),
        ];
        assert!(matches!(
            resolve_unit(&garbage, 16),
            Err(DeriveError::Record { ordinal: 0, .. })
        ));
    }

    /// A unit deriving more effects than its budget is refused before the
    /// effects are collected.
    #[test]
    fn the_fan_out_is_bounded() {
        let index = interp(false, false, None);
        let list = Value::Array((0..10).map(|i| s(&format!("v{i}"))).collect());
        let unit = vec![work(
            &index,
            1,
            None,
            DerivedSource::Values(Some(vec![list])),
        )];
        assert_eq!(resolve_unit(&unit, 9), Err(DeriveError::FanOut(9)));
        assert_eq!(resolve_unit(&unit, 10).expect("resolve").len(), 10);
    }
}

/// An interpretation of another key codec is refused, not derived.
#[test]
fn another_codec_is_refused() {
    let other = IndexInterpretation {
        codec: KEY_CODEC + 1,
        ..interp(false, false, None)
    };
    assert_eq!(
        other.check_supported(),
        Err(UnsupportedInterpretation {
            generation: GEN,
            codec: KEY_CODEC + 1
        })
    );
    assert!(interp(false, false, None).check_supported().is_ok());
}
