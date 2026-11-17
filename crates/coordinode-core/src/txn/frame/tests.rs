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

/// The namespace the units of these tests write.
const NS: crate::namespace::NamespaceId = crate::namespace::NamespaceId::from_raw(1);

/// A proposal of `mutations`, in [`NS`].
fn proposal(mutations: Vec<Mutation>) -> RaftProposal {
    unnamed(crate::txn::proposal::in_namespace(NS, mutations))
}

/// A proposal of `mutations` exactly as given, no namespace named for them.
fn unnamed(mutations: Vec<Mutation>) -> RaftProposal {
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
        // Both range removals among the row changes, the command not counted.
        effects_with(vec![5, 6]),
    ])
}

fn label(raw: u32) -> crate::schema::label_id::LabelId {
    crate::schema::label_id::LabelId::new(raw).expect("a nonzero label id")
}

/// A context naming a node the unit writes, one it removes without a record,
/// and a relationship.
fn effects() -> Mutation {
    effects_with(Vec::new())
}

/// [`effects`] naming the range removals at `runs` as runs of point deletes.
fn effects_with(runs: Vec<u32>) -> Mutation {
    Mutation::Effects(UnitEffects {
        interpretation: crate::txn::proposal::EFFECTS_INTERPRETATION,
        nodes: vec![
            NodeBasis {
                key: b"node:1:42".to_vec(),
                labels: vec![label(1), label(300), label(u32::MAX)],
            },
            NodeBasis {
                key: b"node:1:44".to_vec(),
                labels: Vec::new(),
            },
        ],
        removed_edges: vec![
            b"edgeprop:KNOWS:\0\0\0\0\0\0\0\x2a:\0\0\0\0\0\0\0\x2b".to_vec(),
            b"edgeprop:KNOWS:\0\0\0\0\0\0\0\x2a:\0\0\0\0\0\0\0\x2b:\x80\0\0\0\0\0\0\x01".to_vec(),
        ],
        runs,
        implications: vec![crate::txn::proposal::Implication {
            label: label(1),
            implies: vec![label(2), label(70_000)],
        }],
        scope_changes: vec![crate::txn::proposal::ScopeChange {
            label: label(1),
            activation: u64::MAX,
            before: vec![label(2)],
            after: Vec::new(),
        }],
        dropped: vec![label(70_000)],
    })
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
        // One past the last partition a frame numbers.
        KIND_PUT | ((PARTITIONS.len() as u8) << 3),
        // The one past the namespace marker.
        KIND_COMMAND | (2 << 3),
        KIND_DERIVE | (1 << 3),
        KIND_OPERATION | (1 << 3),
        KIND_EFFECTS | (1 << 3),
    ] {
        let mut body = header(1);
        body.extend_from_slice(&[tag, 0, 0, 0]);
        assert_eq!(
            decode(&signed(body)),
            Err(FrameError::UnknownTag(tag)),
            "tag {tag:#04x}"
        );
    }
    // An operation command this build does not know.
    let mut body = header(1);
    body.extend_from_slice(&[KIND_OPERATION, 9, 0, 0]);
    assert_eq!(
        decode(&signed(body)),
        Err(FrameError::UnknownTag(KIND_OPERATION))
    );
}

/// Every operation command travels in a frame and comes back as it was, next
/// to the data it binds.
#[test]
fn operation_commands_round_trip() {
    use crate::txn::operation::{
        INTENT_DIGEST_LEN, ISSUER_ID_LEN, IntentDigest, IssuerId, OperationBinding,
        OperationCommand, OperationId,
    };
    let issuer = IssuerId([3; ISSUER_ID_LEN]);
    let op = OperationId {
        issuer,
        incarnation: 2,
        sequence: 300,
    };
    for command in [
        OperationCommand::Enroll {
            issuer,
            incarnation: 2,
            window: 1 << 16,
        },
        OperationCommand::Bind(OperationBinding {
            op,
            intent: IntentDigest([5; INTENT_DIGEST_LEN]),
            attempt: u64::MAX,
        }),
        OperationCommand::Bind(OperationBinding {
            op,
            intent: IntentDigest([6; INTENT_DIGEST_LEN]),
            attempt: 1,
        }),
        OperationCommand::Exclude(op),
        OperationCommand::Release(op),
        OperationCommand::Retire {
            issuer,
            incarnation: 2,
            below: 301,
        },
    ] {
        let proposal = RaftProposal {
            id: ProposalId::from_raw(4),
            mutations: crate::txn::proposal::in_namespace(
                NS,
                [
                    Mutation::Put {
                        partition: PartitionId::Node,
                        key: b"node:1:7".to_vec(),
                        value: b"record".to_vec(),
                    },
                    Mutation::Operation(command),
                ],
            ),
            commit_ts: Timestamp::from_raw(10),
            start_ts: Timestamp::from_raw(9),
            bypass_rate_limiter: false,
        };
        let frame = encode_proposal(&proposal).expect("encode");
        assert_eq!(decode(&frame), Ok(proposal), "{command:?}");
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

/// A reader of rows gets the row changes, operation commands and effect
/// context of a frame in their order, its metadata commands and DERIVED work
/// left out and not counted, and literals the left-out work introduced still
/// serve the operations that refer to them. The change feed reads rows this
/// way, so a context lost here would be lost to every subscriber.
#[test]
fn a_reader_of_rows_leaves_commands_and_derived_work_out() {
    let issuer = IssuerId([3; ISSUER_ID_LEN]);
    let op = OperationId {
        issuer,
        incarnation: 1,
        sequence: 7,
    };
    let bind = Mutation::Operation(OperationCommand::Bind(OperationBinding {
        op,
        intent: IntentDigest([6; INTENT_DIGEST_LEN]),
        attempt: 1,
    }));
    // Positions count the namespace marker: the first record put is at 3.
    let full = proposal(vec![
        derive(1, DerivedSource::Values(None)),
        derive(2, DerivedSource::Values(None)),
        put(PartitionId::Node, b"node:1:42", b"record-bytes"),
        Mutation::Command(MetadataCommand::RegisterFields {
            names: vec!["email".into(); 64],
        }),
        derive(42, DerivedSource::UnitRecord(3)),
        put(PartitionId::Node, b"node:1:43", b"record-bytes"),
        effects(),
        bind.clone(),
    ]);
    let frame = encode_proposal(&full).expect("encode");
    let (whole, whole_expanded) =
        decode_proposal_sized(&frame, &DecodeLimits::DEFAULT).expect("decode");
    assert_eq!(whole, full);
    let (rows, rows_expanded) = decode_proposal_rows(&frame, &DecodeLimits::DEFAULT).expect("rows");
    assert_eq!(
        rows.mutations,
        vec![
            Mutation::Namespace(NS),
            put(PartitionId::Node, b"node:1:42", b"record-bytes"),
            put(PartitionId::Node, b"node:1:43", b"record-bytes"),
            effects(),
            bind,
        ]
    );
    assert_eq!((rows.id, rows.commit_ts), (full.id, full.commit_ts));
    assert!(
        rows_expanded < whole_expanded,
        "left-out work is not counted: {rows_expanded} of {whole_expanded}"
    );
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
    let mut body = header(2);
    let mut encoder = Encoder::default();
    encoder
        .op(&Mutation::Namespace(NS), &mut body)
        .expect("marker");
    encoder.op(&Mutation::Derive(work), &mut body).expect("op");
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
    // The record put follows the namespace marker.
    let one = proposal(vec![
        put(PartitionId::Node, b"node:1:42", b"record-bytes"),
        derive(42, DerivedSource::UnitRecord(1)),
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
    // Whole units, the namespace marker at position 0.
    let ns = Mutation::Namespace(NS);
    let cases = [
        // Itself.
        vec![ns.clone(), derive(1, DerivedSource::UnitRecord(1))],
        // The marker.
        vec![
            ns.clone(),
            put(PartitionId::Node, b"node:1:1", b"r"),
            derive(1, DerivedSource::UnitRecord(0)),
        ],
        // Another kind of operation.
        vec![
            ns.clone(),
            put(PartitionId::Idx, b"idx:x", b""),
            derive(1, DerivedSource::UnitRecord(1)),
        ],
        // A later record.
        vec![
            ns.clone(),
            derive(1, DerivedSource::UnitRecord(2)),
            put(PartitionId::Node, b"node:1:1", b"r"),
        ],
        // Past the unit.
        vec![ns, derive(1, DerivedSource::UnitRecord(9))],
    ];
    for mutations in cases {
        assert!(
            matches!(
                encode_proposal(&unnamed(mutations.clone())),
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

    // The namespace marker, two keys of 8 bytes and two values of 16, the
    // second a reference.
    let expanded = NAMESPACE_BYTES + 2 * 8 + 2 * 16;
    let exact = DecodeLimits {
        max_frame_bytes: frame.len(),
        max_ops: 3,
        max_expanded_bytes: expanded,
    };
    assert_eq!(encode_within(&unit, &exact).expect("at the bounds"), frame);
    assert_eq!(decode_proposal(&frame, &exact).expect("decode"), unit);

    for (limits, what) in [
        (
            DecodeLimits {
                max_ops: 2,
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
    // The node put follows the namespace marker.
    let mut derived = mixed().mutations;
    derived.push(derive(42, DerivedSource::UnitRecord(1)));
    derived.push(derive(43, DerivedSource::Values(None)));
    let bad_source = vec![derive(1, DerivedSource::UnitRecord(0))];
    let units = [mixed(), unnamed(derived), proposal(bad_source)];
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
    // The node put follows the namespace marker.
    let mut derived = mixed().mutations;
    derived.push(derive(42, DerivedSource::UnitRecord(1)));
    derived.push(derive(43, DerivedSource::Values(None)));
    derived.push(derive(
        44,
        DerivedSource::Values(Some(vec![crate::graph::types::Value::String(
            "new@x".into(),
        )])),
    ));
    for unit in [mixed(), unnamed(derived)] {
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
        decode(&unchecked_frame(&crate::txn::proposal::in_namespace(
            NS,
            [work]
        ))),
        Err(FrameError::Unsupported(_))
    ));
}

/// The routing basis of removed nodes costs a few bytes each beside the
/// deletes it describes: its keys share the node partition's prefixes, so no
/// identity is written twice.
#[test]
fn an_effect_context_costs_little_beside_the_rows_it_describes() {
    let keys: Vec<Vec<u8>> = (0..200u64)
        .map(|i| {
            crate::graph::node::encode_node_key(1, crate::graph::node::NodeId::from_raw(i * 7))
        })
        .collect();
    let deletes: Vec<Mutation> = keys
        .iter()
        .map(|key| Mutation::Delete {
            partition: PartitionId::Node,
            key: key.clone(),
        })
        .collect();
    let mut described = deletes.clone();
    described.push(Mutation::Effects(UnitEffects {
        interpretation: crate::txn::proposal::EFFECTS_INTERPRETATION,
        nodes: keys
            .iter()
            .map(|key| NodeBasis {
                key: key.clone(),
                labels: vec![label(3), label(9)],
            })
            .collect(),
        ..UnitEffects::new()
    }));
    let bare = encode_proposal(&proposal(deletes)).expect("encode").len();
    let full = encode_proposal(&proposal(described.clone()))
        .expect("encode")
        .len();
    assert_eq!(
        decode(&encode_proposal(&proposal(described.clone())).expect("encode")).expect("decode"),
        proposal(described)
    );
    // Shared and suffix lengths, a short suffix, a count and two ids.
    assert!(
        full - bare <= 8 + keys.len() * 8,
        "{} context bytes for {} nodes",
        full - bare,
        keys.len()
    );
}

/// A run must name a range removal among the row changes before it, in
/// ascending order: a reader enumerates the keys of what it names, so a
/// position naming anything else is refused by the encoder and the decoder
/// alike. Commands between the rows do not count, which keeps the positions
/// the same for a reader of rows.
#[test]
fn a_run_must_name_an_earlier_range_removal() {
    let range = Mutation::RemoveRange {
        partition: PartitionId::Node,
        start: b"node:1:1".to_vec(),
        end: b"node:1:5".to_vec(),
    };
    let command = Mutation::Command(MetadataCommand::RegisterFields {
        names: vec!["x".into()],
    });
    let good = vec![
        put(PartitionId::Node, b"node:1:0", b"r"),
        command.clone(),
        range.clone(),
        effects_with(vec![1]),
    ];
    let frame = encode_proposal(&proposal(good.clone())).expect("encode");
    assert_eq!(decode(&frame).expect("decode"), proposal(good.clone()));
    let (rows, _) = decode_proposal_rows(&frame, &DecodeLimits::DEFAULT).expect("rows");
    // The marker is kept and, like the command, not counted as a row.
    assert_eq!(
        rows.mutations,
        vec![
            Mutation::Namespace(NS),
            good[0].clone(),
            good[2].clone(),
            good[3].clone()
        ]
    );

    for bad in [
        // A put, not a range removal.
        vec![
            put(PartitionId::Node, b"node:1:0", b"r"),
            range.clone(),
            effects_with(vec![0]),
        ],
        // Past every row change before the context.
        vec![range.clone(), effects_with(vec![1])],
        // Not ascending.
        vec![range.clone(), range.clone(), effects_with(vec![1, 0])],
        // Repeated.
        vec![range.clone(), effects_with(vec![0, 0])],
    ] {
        assert!(
            matches!(
                encode_proposal(&proposal(bad.clone())),
                Err(FrameError::Effects(_))
            ),
            "encoded {bad:?}"
        );
        assert!(
            matches!(
                decode(&unchecked_frame(&crate::txn::proposal::in_namespace(
                    NS,
                    bad.clone()
                ))),
                Err(FrameError::Effects(_))
            ),
            "decoded {bad:?}"
        );
    }
}

/// Label id 0 names no label: a context carrying it is refused rather than
/// read as some label.
#[test]
fn a_zero_label_id_in_an_effect_context_is_refused() {
    let mut body = header(1);
    body.push(KIND_EFFECTS);
    put_varint(1, &mut body); // interpretation
    put_varint(1, &mut body); // one node
    put_varint(0, &mut body); // shared
    put_varint(1, &mut body); // suffix length
    body.push(b'n');
    put_varint(1, &mut body); // one label
    put_varint(0, &mut body); // label id 0
    put_varint(0, &mut body); // no edges
    assert_eq!(
        decode(&signed(body)),
        Err(FrameError::Effects("label id out of range"))
    );
}

/// Counts the frame cannot hold are refused before anything is allocated
/// for them.
#[test]
fn impossible_effect_counts_are_refused_before_allocation() {
    let mut nodes = header(1);
    nodes.push(KIND_EFFECTS);
    put_varint(1, &mut nodes);
    put_varint(1 << 40, &mut nodes);
    assert!(matches!(
        decode(&signed(nodes)),
        Err(FrameError::Truncated { .. } | FrameError::OverLimit { .. })
    ));

    let mut labels = header(1);
    labels.push(KIND_EFFECTS);
    put_varint(1, &mut labels);
    put_varint(1, &mut labels);
    put_varint(0, &mut labels);
    put_varint(0, &mut labels);
    put_varint(1 << 30, &mut labels);
    assert!(matches!(
        decode(&signed(labels)),
        Err(FrameError::Truncated { .. } | FrameError::OverLimit { .. })
    ));

    let mut edges = header(1);
    edges.push(KIND_EFFECTS);
    put_varint(1, &mut edges);
    put_varint(0, &mut edges);
    put_varint(1 << 40, &mut edges);
    assert!(matches!(
        decode(&signed(edges)),
        Err(FrameError::Truncated { .. } | FrameError::OverLimit { .. })
    ));
}

/// Bytes between the last operation and the checksum are refused.
#[test]
fn trailing_bytes_are_refused() {
    let mut body = header(0);
    body.push(0);
    assert_eq!(decode(&signed(body)), Err(FrameError::Trailing(1)));
}

/// A unit that writes several namespaces names each before its writes, and
/// decodes with every marker where it was: what belongs to which namespace
/// is in the frame, in order.
#[test]
fn a_unit_spanning_namespaces_keeps_each_marker_in_place() {
    let other = crate::namespace::NamespaceId::from_raw(0x1234_5678_9abc);
    let original = unnamed(vec![
        Mutation::Namespace(NS),
        put(PartitionId::Node, b"node:1:1", b"first"),
        Mutation::Namespace(other),
        put(PartitionId::Node, b"node:1:1", b"second"),
        put(PartitionId::Schema, b"schema:label:A", b"{}"),
    ]);
    let frame = encode_proposal(&original).expect("encode");
    assert_eq!(decode(&frame).expect("decode"), original);
}

/// The deployment's own writes and operation commands name no namespace,
/// and need none.
#[test]
fn the_deployments_writes_need_no_namespace() {
    let original = unnamed(vec![
        put(PartitionId::Registry, b"registry:c", b"v"),
        put(PartitionId::Deployment, b"catalog:x", b"v"),
        Mutation::Command(MetadataCommand::GrantNodeLease {
            base: 0,
            ceiling: 10,
            token: [1; crate::graph::node::NODE_LEASE_TOKEN_LEN],
        }),
    ]);
    let frame = encode_proposal(&original).expect("encode");
    assert_eq!(decode(&frame).expect("decode"), original);
}

/// A namespace's data with no namespace named before it is refused by the
/// encoder, and a frame holding it (as a frame written before units named
/// their namespaces does) is refused by the decoder, before anything reads
/// the unit: a row change, a namespace's metadata command and DERIVED work
/// alike, whether or not a reader of rows decodes the work.
#[test]
fn a_namespaces_data_before_any_namespace_is_refused() {
    let unscoped = [
        put(PartitionId::Node, b"node:1:1", b"v"),
        Mutation::Command(MetadataCommand::RegisterFields {
            names: vec!["email".into()],
        }),
    ];
    for mutation in unscoped {
        let unit = unnamed(vec![
            put(PartitionId::Deployment, b"catalog:x", b"v"),
            mutation.clone(),
        ]);
        assert_eq!(
            encode_proposal(&unit),
            Err(FrameError::Unscoped(crate::txn::proposal::Unscoped {
                at: 1
            })),
            "{mutation:?}"
        );
        assert!(check_proposal(&unit).is_err(), "{mutation:?}");

        // The same unit, written by an encoder that named no namespace.
        let mut body = header(2);
        let mut encoder = Encoder::default();
        for m in &unit.mutations {
            encoder.op(m, &mut body).expect("encode the op");
        }
        let frame = signed(body);
        assert_eq!(
            decode(&frame),
            Err(FrameError::Unscoped(crate::txn::proposal::Unscoped {
                at: 1
            })),
            "{mutation:?}"
        );
    }

    // The tag and a byte of its body: every operation takes two bytes, and
    // the refusal comes before the body is read.
    let mut body = header(1);
    body.extend_from_slice(&[KIND_DERIVE, 0]);
    let frame = signed(body);
    for decoded in [
        decode(&frame).map(drop),
        decode_proposal_rows(&frame, &DecodeLimits::DEFAULT).map(drop),
    ] {
        assert_eq!(
            decoded,
            Err(FrameError::Unscoped(crate::txn::proposal::Unscoped {
                at: 0
            }))
        );
    }
}

/// A proposal in the field-wise form proposals were written in before
/// frames, as a map or as a sequence, is refused: a proposal deserializes
/// from its frame and nothing else.
#[test]
fn a_field_wise_proposal_is_refused() {
    #[derive(serde::Serialize)]
    struct FieldWise {
        id: u64,
        mutations: Vec<Mutation>,
        commit_ts: u64,
        start_ts: u64,
        bypass_rate_limiter: bool,
    }
    let field_wise = FieldWise {
        id: 1,
        mutations: vec![put(PartitionId::Node, b"node:1:1", b"v")],
        commit_ts: 2,
        start_ts: 1,
        bypass_rate_limiter: false,
    };
    for bytes in [
        rmp_serde::to_vec_named(&field_wise).expect("as a map"),
        rmp_serde::to_vec(&field_wise).expect("as a sequence"),
    ] {
        assert!(
            rmp_serde::from_slice::<RaftProposal>(&bytes).is_err(),
            "a field-wise proposal was read"
        );
    }
    // The frame form still reads.
    let framed = rmp_serde::to_vec(&mixed()).expect("framed");
    assert_eq!(
        rmp_serde::from_slice::<RaftProposal>(&framed).expect("a frame"),
        mixed()
    );
}
