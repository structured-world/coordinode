//! Frame round trips, the redundancy they remove, and every malformed input
//! the decoder refuses.

use super::*;
use crate::txn::proposal::MetadataCommand;

fn put(partition: PartitionId, key: &[u8], value: &[u8]) -> Mutation {
    Mutation::Put {
        partition,
        key: key.to_vec(),
        value: value.to_vec(),
    }
}

fn proposal(mutations: Vec<Mutation>) -> RaftProposal {
    RaftProposal {
        id: ProposalId::from_raw(0xDEAD_BEEF_0042),
        mutations,
        commit_ts: Timestamp::from_raw(1_790_000_000_000_123),
        start_ts: Timestamp::from_raw(1_790_000_000_000_100),
        bypass_rate_limiter: true,
    }
}

/// One of every operation kind, over several partitions, with repeated
/// prefixes, repeated values and an empty key, value and range.
fn mixed() -> RaftProposal {
    let email = b"alice@example.com-long-enough".to_vec();
    proposal(vec![
        put(PartitionId::Node, b"node:1:42", b"record-bytes-42"),
        put(
            PartitionId::Idx,
            b"idx:\0\0\0\x0auser_email:@alice:\0\0\0\0\0\0\0\x2a",
            b"",
        ),
        Mutation::Merge {
            partition: PartitionId::Adj,
            key: b"adj:KNOWS:out:42".to_vec(),
            operand: email.clone(),
        },
        put(PartitionId::Node, b"node:1:43", &email),
        Mutation::Delete {
            partition: PartitionId::Idx,
            key: b"idx:\0\0\0\x0auser_email:@bob:\0\0\0\0\0\0\0\x2b".to_vec(),
        },
        Mutation::RemoveRange {
            partition: PartitionId::Idx,
            start: b"idx:a".to_vec(),
            end: b"idx:b".to_vec(),
        },
        Mutation::RemoveRange {
            partition: PartitionId::Counter,
            start: Vec::new(),
            end: Vec::new(),
        },
        Mutation::Command(MetadataCommand::RegisterFields {
            names: vec!["email".into(), "age".into()],
        }),
        put(PartitionId::Registry, b"", b""),
        put(PartitionId::VectorF32, b"vec:1:42", &email),
    ])
}

fn decode(bytes: &[u8]) -> Result<RaftProposal, FrameError> {
    decode_proposal(bytes, &DecodeLimits::DEFAULT)
}

/// Decoding a frame yields exactly the proposal encoded: every kind,
/// partition, key, value and the order of the operations.
#[test]
fn a_frame_decodes_to_the_proposal_it_encodes() {
    let original = mixed();
    let frame = encode_proposal(&original).expect("encode");
    assert_eq!(decode(&frame).expect("decode"), original);

    let empty = proposal(Vec::new());
    assert_eq!(
        decode(&encode_proposal(&empty).expect("encode")).expect("decode"),
        empty
    );
}

/// Keys that share a prefix with the previous key of their partition cost
/// only their difference, and an identical value costs a reference.
#[test]
fn repeated_prefixes_and_values_are_written_once() {
    let value = vec![7u8; 64];
    let mutations: Vec<Mutation> = (0..100u64)
        .map(|i| {
            put(
                PartitionId::Idx,
                format!("idx:\0\0\0\x0auser_email:user-{i:04}@example.com:").as_bytes(),
                &value,
            )
        })
        .collect();
    let raw: usize = mutations
        .iter()
        .map(|m| match m {
            Mutation::Put { key, value, .. } => key.len() + value.len(),
            _ => 0,
        })
        .sum();
    let original = proposal(mutations);
    let frame = encode_proposal(&original).expect("encode");
    assert_eq!(decode(&frame).expect("decode"), original);
    // Each later key shares everything up to its differing digits, and each
    // later value is one reference byte.
    assert!(
        frame.len() * 5 < raw,
        "a frame of {} bytes for {raw} raw bytes",
        frame.len()
    );
}

/// A flipped bit anywhere in the frame is caught by the checksum.
#[test]
fn a_changed_byte_fails_the_checksum() {
    let frame = encode_proposal(&mixed()).expect("encode");
    for at in 0..frame.len() {
        let mut bad = frame.clone();
        bad[at] ^= 0x10;
        assert!(decode(&bad).is_err(), "a change at byte {at} was accepted");
    }
}

/// Every proper prefix of a frame is refused, however it is cut.
#[test]
fn a_truncated_frame_is_refused() {
    let frame = encode_proposal(&mixed()).expect("encode");
    for len in 0..frame.len() {
        assert!(
            decode(&frame[..len]).is_err(),
            "a frame cut to {len} bytes was accepted"
        );
    }
}

/// Re-sign `body` with its checksum, so a test reaches the decoder's
/// structural checks rather than stopping at the checksum.
fn signed(mut body: Vec<u8>) -> Vec<u8> {
    let crc = crc32c::crc32c(&body);
    body.extend_from_slice(&crc.to_le_bytes());
    body
}

fn header(op_count: u64) -> Vec<u8> {
    let mut b = vec![FRAME_VERSION];
    put_varint(1, &mut b);
    put_varint(2, &mut b);
    put_varint(3, &mut b);
    b.push(0);
    put_varint(op_count, &mut b);
    b
}

/// A frame of another layout version is refused, not guessed at.
#[test]
fn another_version_is_refused() {
    let mut body = header(0);
    body[0] = FRAME_VERSION + 1;
    assert_eq!(
        decode(&signed(body)),
        Err(FrameError::UnsupportedVersion(FRAME_VERSION + 1))
    );
}

/// Unknown flags and unknown operation tags are refused.
#[test]
fn unknown_flags_and_tags_are_refused() {
    let mut body = header(0);
    let flags_at = body.len() - 2;
    body[flags_at] = 0x80;
    assert_eq!(decode(&signed(body)), Err(FrameError::UnknownFlags(0x80)));

    for tag in [
        6u8,
        7,
        KIND_PUT | (10 << 3),
        KIND_COMMAND | (1 << 3),
        KIND_DERIVE | (1 << 3),
    ] {
        let mut body = header(1);
        body.extend_from_slice(&[tag, 0, 0, 0]);
        assert_eq!(
            decode(&signed(body)),
            Err(FrameError::UnknownTag(tag)),
            "tag {tag:#04x}"
        );
    }
}

/// A key cannot share more bytes than its predecessor has, and a value
/// cannot refer to a slice the frame has not carried yet.
#[test]
fn bad_prefixes_and_references_are_refused() {
    let mut body = header(1);
    body.push(KIND_DELETE | (6 << 3));
    put_varint(3, &mut body); // shares 3 bytes with a partition that has no key yet
    put_varint(0, &mut body);
    body.extend_from_slice(&[0, 0]);
    assert_eq!(
        decode(&signed(body)),
        Err(FrameError::BadPrefix {
            shared: 3,
            available: 0
        })
    );

    let mut body = header(1);
    body.push(KIND_PUT);
    put_varint(0, &mut body);
    put_varint(1, &mut body);
    body.push(b'k');
    put_varint(1, &mut body); // reference to slice 0 before any literal
    assert_eq!(
        decode(&signed(body)),
        Err(FrameError::BadReference { index: 0, count: 0 })
    );
}

/// The expansion budget counts referenced slices too: a small frame that
/// repeats one large value cannot expand past the bound.
#[test]
fn expansion_is_bounded_by_the_budget_not_the_frame_size() {
    let value = vec![1u8; 4096];
    let mutations: Vec<Mutation> = (0..64u32)
        .map(|i| put(PartitionId::Node, &i.to_be_bytes(), &value))
        .collect();
    let frame = encode_proposal(&proposal(mutations)).expect("encode");
    assert!(frame.len() < 8192, "the repeats are references");
    let tight = DecodeLimits {
        max_expanded_bytes: 64 * 1024,
        ..DecodeLimits::DEFAULT
    };
    assert_eq!(
        decode_proposal(&frame, &tight),
        Err(FrameError::OverLimit {
            what: "expanded size",
            limit: 64 * 1024
        })
    );
    assert!(decode(&frame).is_ok());
}

/// Operation counts above the bound, or above what the frame's bytes could
/// hold, are refused before anything is allocated for them.
#[test]
fn impossible_counts_are_refused_before_allocation() {
    let body = header(u64::MAX);
    assert!(matches!(
        decode(&signed(body)),
        Err(FrameError::OverLimit {
            what: "operation count",
            ..
        })
    ));
    let body = header(1000);
    assert!(matches!(
        decode(&signed(body)),
        Err(FrameError::Truncated { .. })
    ));
    let frame = encode_proposal(&mixed()).expect("encode");
    let few = DecodeLimits {
        max_ops: 2,
        ..DecodeLimits::DEFAULT
    };
    assert!(matches!(
        decode_proposal(&frame, &few),
        Err(FrameError::OverLimit {
            what: "operation count",
            ..
        })
    ));
}

/// A varint that does not fit a u64 is refused.
#[test]
fn an_overlong_varint_is_refused() {
    let mut body = vec![FRAME_VERSION];
    body.extend_from_slice(&[0xFF; 10]);
    body.push(0x01);
    assert!(matches!(
        decode(&signed(body)),
        Err(FrameError::VarintOverflow { at: 1 })
    ));
}

fn binding(codec: u32) -> IndexBinding {
    use crate::index::derive::{IndexInterpretation, PropertyRef};
    IndexBinding {
        epoch: 3,
        interpretation: IndexInterpretation {
            codec,
            generation: crate::index::identity::GenerationId::from_raw(4),
            unique: false,
            sparse: false,
            properties: vec![PropertyRef {
                field: Some(1),
                name: "email".into(),
            }],
            filter: None,
        },
    }
}

fn derive(node_id: u64, new: DerivedSource) -> Mutation {
    Mutation::Derive(DerivedIndexWork {
        binding: binding(crate::index::derive::KEY_CODEC),
        node_id,
        valid_from: None,
        old: Some(vec![crate::graph::types::Value::String("old@x".into())]),
        new,
    })
}

/// An owner tag that is neither a node nor a version is refused by the
/// decoder rather than read as some other owner.
#[test]
fn an_unknown_owner_tag_is_refused() {
    const MARK: i64 = 0x7A7A_7A7A_7A7A_7A7A;
    let Mutation::Derive(mut work) = derive(1, DerivedSource::Values(None)) else {
        unreachable!("derive builds a Derive mutation")
    };
    work.valid_from = Some(MARK);
    let mut body = header(1);
    Encoder::default()
        .op(&Mutation::Derive(work), &mut body)
        .expect("op");
    let at = body
        .windows(8)
        .position(|w| w == MARK.to_be_bytes())
        .expect("the version is in the body");
    assert_eq!(body[at - 1], OWNER_VERSION);
    body[at - 1] = 7;
    assert!(matches!(
        decode(&signed(body)),
        Err(FrameError::UnknownTag(7))
    ));
}

/// Work for a version of a temporal node round-trips with its valid_from,
/// negative and extreme starts included, beside work for a plain node.
#[test]
fn version_work_round_trips_its_valid_from() {
    let version = |node_id, valid_from| {
        let Mutation::Derive(mut work) = derive(node_id, DerivedSource::Values(None)) else {
            unreachable!("derive builds a Derive mutation")
        };
        work.valid_from = Some(valid_from);
        Mutation::Derive(work)
    };
    let unit = proposal(vec![
        version(1, -5),
        derive(2, DerivedSource::Values(None)),
        version(3, i64::MIN),
        version(4, i64::MAX),
    ]);
    let frame = encode_proposal(&unit).expect("encode");
    assert_eq!(decode(&frame).expect("decode"), unit);
}

/// DERIVED work round-trips with its binding, inputs and source, and the
/// binding every effect of one index repeats is written once.
#[test]
fn derived_work_round_trips_and_shares_its_binding() {
    let one = proposal(vec![
        put(PartitionId::Node, b"node:1:42", b"record-bytes"),
        derive(42, DerivedSource::UnitRecord(0)),
    ]);
    let frame_one = encode_proposal(&one).expect("encode");
    assert_eq!(decode(&frame_one).expect("decode"), one);

    let mut many = vec![put(PartitionId::Node, b"node:1:42", b"record-bytes")];
    for node in 0..20 {
        many.push(derive(node, DerivedSource::Values(None)));
    }
    let many = proposal(many);
    let frame_many = encode_proposal(&many).expect("encode");
    assert_eq!(decode(&frame_many).expect("decode"), many);
    let binding_bytes = rmp_serde::to_vec(&binding(1)).expect("binding").len();
    assert!(
        frame_many.len() < frame_one.len() + 19 * binding_bytes / 2,
        "{} bytes for 20 effects of one binding of {binding_bytes} bytes",
        frame_many.len()
    );
}

/// The frame of `mutations` as a writer that skipped the encoder's checks
/// would produce it, so the decoder's own refusal is what a test sees.
fn unchecked_frame(mutations: &[Mutation]) -> Vec<u8> {
    let mut body = header(mutations.len() as u64);
    let mut encoder = Encoder::default();
    for mutation in mutations {
        encoder.op(mutation, &mut body).expect("op");
    }
    signed(body)
}

/// A source must name an earlier node record put: a later position, a
/// position past the unit or another kind of operation is refused, by the
/// encoder before the unit reaches a log and by the decoder of a frame that
/// carries one anyway.
#[test]
fn a_source_that_is_not_an_earlier_record_is_refused() {
    let cases = [
        vec![derive(1, DerivedSource::UnitRecord(0))],
        vec![
            put(PartitionId::Idx, b"idx:x", b""),
            derive(1, DerivedSource::UnitRecord(0)),
        ],
        vec![
            derive(1, DerivedSource::UnitRecord(1)),
            put(PartitionId::Node, b"node:1:1", b"r"),
        ],
    ];
    for mutations in cases {
        assert!(
            matches!(
                encode_proposal(&proposal(mutations.clone())),
                Err(FrameError::BadSource { .. })
            ),
            "encode {mutations:?}"
        );
        assert!(
            matches!(
                decode(&unchecked_frame(&mutations)),
                Err(FrameError::BadSource { .. })
            ),
            "decode {mutations:?}"
        );
    }
}

/// A unit the decoder's bounds would refuse is refused by the encoder, so it
/// never reaches a log every member would then fail to apply: too many
/// operations, too many expanded bytes (a reference expands as much as the
/// literal it names), or too large a frame. A unit at the bounds encodes.
#[test]
fn a_unit_beyond_the_decode_bounds_is_not_encoded() {
    let value = vec![7u8; 16];
    let unit = proposal(vec![
        put(PartitionId::Node, b"node:1:1", &value),
        put(PartitionId::Node, b"node:1:2", &value),
    ]);
    let frame = encode_proposal(&unit).expect("encode");
    let decoded = decode_proposal(&frame, &DecodeLimits::DEFAULT).expect("decode");
    assert_eq!(decoded, unit);

    // Two keys of 8 bytes and two values of 16, the second a reference.
    let expanded = 2 * 8 + 2 * 16;
    let exact = DecodeLimits {
        max_frame_bytes: frame.len(),
        max_ops: 2,
        max_expanded_bytes: expanded,
    };
    assert_eq!(encode_within(&unit, &exact).expect("at the bounds"), frame);
    assert_eq!(decode_proposal(&frame, &exact).expect("decode"), unit);

    for (limits, what) in [
        (
            DecodeLimits {
                max_ops: 1,
                ..exact
            },
            "operation count",
        ),
        (
            DecodeLimits {
                max_expanded_bytes: expanded - 1,
                ..exact
            },
            "expanded size",
        ),
        (
            DecodeLimits {
                max_frame_bytes: frame.len() - 1,
                ..exact
            },
            "frame size",
        ),
    ] {
        assert!(
            matches!(
                encode_within(&unit, &limits),
                Err(FrameError::OverLimit { what: w, .. }) if w == what
            ),
            "{what}"
        );
        assert!(decode_proposal(&frame, &limits).is_err(), "decoder {what}");
    }
}

/// A journal's unit round-trips through its frame, and the frame carries the
/// unit's commit timestamp in the proposal header a reader sees.
#[test]
fn a_journal_unit_round_trips() {
    let unit = mixed().mutations;
    let frame = encode_unit(&unit, Timestamp::from_raw(77)).expect("encode");
    assert_eq!(decode_unit(&frame).expect("decode"), unit);
    let read = decode_proposal(&frame, &DecodeLimits::DEFAULT).expect("decode");
    assert_eq!(read.commit_ts, Timestamp::from_raw(77));
}

/// A proposal is admitted exactly when it encodes: the check a pipeline runs
/// before the log agrees with the encoder the log store runs, at every bound
/// and for underivable work, without encoding a unit far from the frame
/// bound.
#[test]
fn the_admission_check_agrees_with_the_encoder() {
    let mut derived = mixed().mutations;
    derived.push(derive(42, DerivedSource::UnitRecord(0)));
    derived.push(derive(43, DerivedSource::Values(None)));
    let bad_source = vec![derive(1, DerivedSource::UnitRecord(0))];
    let units = [mixed(), proposal(derived), proposal(bad_source)];
    for unit in &units {
        let frame_len = encode_proposal(unit).map_or(64, |f| f.len());
        let expanded = (0..=frame_len * 64)
            .find(|&n| {
                encode_within(
                    unit,
                    &DecodeLimits {
                        max_expanded_bytes: n,
                        ..DecodeLimits::DEFAULT
                    },
                )
                .is_ok()
            })
            .unwrap_or(0);
        // A frame is never empty, so `frame_len - 1` is a bound below it.
        for max_frame_bytes in [frame_len - 1, frame_len, usize::MAX >> 1] {
            let below = expanded.checked_sub(1);
            for max_expanded_bytes in [below, Some(expanded), Some(usize::MAX >> 1)]
                .into_iter()
                .flatten()
            {
                for max_ops in [unit.mutations.len() - 1, unit.mutations.len()] {
                    let limits = DecodeLimits {
                        max_frame_bytes,
                        max_ops,
                        max_expanded_bytes,
                    };
                    assert_eq!(
                        check_within(unit, &limits).is_ok(),
                        encode_within(unit, &limits).is_ok(),
                        "{limits:?} for {unit:?}"
                    );
                }
            }
        }
    }
}

/// The encoder counts expansion as the decoder's budget does, for every
/// kind of operation, DERIVED bindings and inputs and their references
/// included: the smallest budget each accepts is the same.
#[test]
fn the_encoder_counts_expansion_as_the_decoder_does() {
    let mut derived = mixed().mutations;
    derived.push(derive(42, DerivedSource::UnitRecord(0)));
    derived.push(derive(43, DerivedSource::Values(None)));
    derived.push(derive(
        44,
        DerivedSource::Values(Some(vec![crate::graph::types::Value::String(
            "new@x".into(),
        )])),
    ));
    for unit in [mixed(), proposal(derived)] {
        let frame = encode_proposal(&unit).expect("encode");
        let within = |max_expanded_bytes| DecodeLimits {
            max_expanded_bytes,
            ..DecodeLimits::DEFAULT
        };
        // The smallest budget the decoder accepts.
        let (mut lo, mut hi) = (0usize, frame.len() * 64);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if decode_proposal(&frame, &within(mid)).is_ok() {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        assert!(lo > 0);
        assert!(encode_within(&unit, &within(lo)).is_ok(), "at {lo}");
        assert!(encode_within(&unit, &within(lo - 1)).is_err(), "below {lo}");
    }
}

/// A binding with a key codec this build does not derive is refused by the
/// encoder, and by the decoder before any of the frame is applied.
#[test]
fn an_unsupported_interpretation_is_refused() {
    let work = Mutation::Derive(DerivedIndexWork {
        binding: binding(crate::index::derive::KEY_CODEC + 1),
        node_id: 1,
        valid_from: None,
        old: None,
        new: DerivedSource::Values(None),
    });
    assert!(matches!(
        encode_proposal(&proposal(vec![work.clone()])),
        Err(FrameError::Unsupported(_))
    ));
    assert!(matches!(
        decode(&unchecked_frame(&[work])),
        Err(FrameError::Unsupported(_))
    ));
}

/// Bytes between the last operation and the checksum are refused.
#[test]
fn trailing_bytes_are_refused() {
    let mut body = header(0);
    body.push(0);
    assert_eq!(decode(&signed(body)), Err(FrameError::Trailing(1)));
}
