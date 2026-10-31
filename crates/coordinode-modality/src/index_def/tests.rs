use super::*;

#[test]
fn btree_descriptor() {
    let idx = IndexDescriptor::btree("user_email", "User", "email").unique();
    assert_eq!(idx.name.as_deref(), Some("user_email"));
    assert_eq!(idx.description, None);
    assert_eq!(idx.label, "User");
    assert_eq!(idx.property(), "email");
    assert_eq!(idx.properties, vec!["email"]);
    assert!(idx.unique);
    assert!(!idx.sparse);
    assert!(!idx.multikey);
    assert!(!idx.is_compound());
    assert_eq!(idx.layout, ENTRY_LAYOUT);
}

#[test]
fn compound_descriptor() {
    let idx = IndexDescriptor::compound(
        "user_label_status",
        "User",
        vec!["label".into(), "status".into()],
    );
    assert!(idx.is_compound());
    assert_eq!(idx.properties.len(), 2);
    assert_eq!(idx.property(), "label");
}

#[test]
fn sparse_descriptor() {
    let idx = IndexDescriptor::btree("user_bio", "User", "bio").sparse();
    assert!(idx.sparse);
}

/// The catalog key is the identity, so a rename moves no record; the name
/// binding is a key of its own, outside the definitions' prefix.
#[test]
fn catalog_keys() {
    let idx = IndexDescriptor::btree("user_email", "User", "email")
        .bind(IndexId::from_raw(0x0102), GenerationId::from_raw(7));
    assert_eq!(
        idx.schema_key(),
        [&b"schema:idx:"[..], &0x0102u64.to_be_bytes()].concat()
    );
    let name_key = IndexDefinition::name_key_of("user_email");
    assert!(!name_key.starts_with(IndexDefinition::SCHEMA_PREFIX));
    assert!(!IndexDefinition::ALLOCATOR_KEY.starts_with(IndexDefinition::SCHEMA_PREFIX));
    assert!(!IndexDefinition::ALLOCATOR_KEY.starts_with(IndexDefinition::NAME_PREFIX));
}

/// The interpretation a DERIVED effect is sealed with carries the
/// generation, not the name: renaming the index changes no entry key.
#[test]
fn the_interpretation_carries_the_generation() {
    let named = IndexDescriptor::btree("a", "User", "email")
        .bind(IndexId::from_raw(1), GenerationId::from_raw(9));
    let mut renamed = named.clone();
    renamed.name = Some("b".into());
    let field = |_: &str| Some(1);
    assert_eq!(named.interpretation(&field), renamed.interpretation(&field));
    assert_eq!(
        named.interpretation(&field).generation,
        GenerationId::from_raw(9)
    );
}

/// Logs name an index by its name, or by its identity when it has none.
#[test]
fn display_names_or_identifies() {
    let named = IndexDescriptor::btree("a", "User", "email")
        .bind(IndexId::from_raw(4), GenerationId::from_raw(4));
    assert_eq!(named.to_string(), "a");
    let mut unnamed = named.clone();
    unnamed.name = None;
    assert_eq!(unnamed.to_string(), "index#4");
}

#[test]
fn new_indexes_default_to_ready_state() {
    let btree = IndexDescriptor::btree("u_email", "User", "email");
    let hnsw = IndexDescriptor::hnsw("u_vec", "User", "vec", VectorIndexConfig::default());
    let compound = IndexDescriptor::compound(
        "u_lbl_status",
        "User",
        vec!["label".into(), "status".into()],
    );
    let text = IndexDescriptor::text(
        "u_text",
        "User",
        vec!["bio".into()],
        TextIndexConfig::default(),
    );
    assert_eq!(btree.state, IndexState::Ready);
    assert_eq!(hnsw.state, IndexState::Ready);
    assert_eq!(compound.state, IndexState::Ready);
    assert_eq!(text.state, IndexState::Ready);
}

#[test]
fn definition_roundtrip_serde() {
    let mut idx = IndexDescriptor::hnsw("v", "L", "p", VectorIndexConfig::default())
        .bind(IndexId::from_raw(3), GenerationId::from_raw(5));
    idx.description = Some("embeddings".into());
    idx.state = IndexState::Building {
        written: 1234,
        estimated_total: 9999,
    };
    let bytes = rmp_serde::to_vec(&idx).expect("encode");
    let back: IndexDefinition = rmp_serde::from_slice(&bytes).expect("decode");
    assert_eq!(back, idx);

    idx.state = IndexState::Failed {
        reason: "build aborted".to_string(),
    };
    let bytes = rmp_serde::to_vec(&idx).expect("encode failed");
    let back: IndexDefinition = rmp_serde::from_slice(&bytes).expect("decode failed");
    assert_eq!(back, idx);
}

fn extra(node: u64) -> Mismatch {
    Mismatch::Extra {
        node,
        valid_from: None,
        tuple: vec![1, 2, 3],
    }
}

/// A fresh integrity record serves as its build state says: nothing is
/// suspect until a disagreement is reported.
#[test]
fn a_fresh_integrity_record_is_unchecked() {
    let record = IndexIntegrityRecord::new(IndexId::from_raw(1), GenerationId::from_raw(2));
    assert_eq!(record.integrity, Integrity::Unchecked);
    assert!(!record.check_pending());
    assert_eq!(record.evidence_revision, 0);
}

/// A report makes the generation suspect and moves the revision; the same
/// disagreement reported again is one report, and the revision a running
/// check started from stays.
#[test]
fn a_report_is_kept_once_and_moves_the_revision() {
    let mut record = IndexIntegrityRecord::new(IndexId::from_raw(1), GenerationId::from_raw(2));
    assert!(record.report(&extra(7)));
    assert_eq!(record.integrity, Integrity::Suspect);
    assert_eq!(record.evidence_revision, 1);
    assert!(!record.report(&extra(7)), "the same disagreement again");
    assert_eq!(record.evidence_revision, 1);
    assert!(record.report(&extra(8)));
    assert_eq!(record.evidence, vec![extra(7), extra(8)]);
    assert_eq!(record.evidence_revision, 2);
}

/// Past the evidence a record keeps, reports still move the revision and
/// keep the generation suspect: a check started before them sees it.
#[test]
fn reports_past_the_cap_still_move_the_revision() {
    let mut record = IndexIntegrityRecord::new(IndexId::from_raw(1), GenerationId::from_raw(2));
    for node in 0..IndexIntegrityRecord::MAX_EVIDENCE as u64 + 5 {
        assert!(record.report(&extra(node)));
    }
    assert_eq!(record.evidence.len(), IndexIntegrityRecord::MAX_EVIDENCE);
    assert_eq!(
        record.evidence_revision,
        IndexIntegrityRecord::MAX_EVIDENCE as u64 + 5
    );
}

/// The state read without the evidence refuses whatever the whole record's
/// decode refuses: a field of the wrong type among those it does not keep
/// (the index, the evidence or one disagreement in it, the revision, the
/// check), or a record cut short. Each malformed record is refused by the
/// whole decode too, and a well-formed one gives the same state both ways;
/// a malformed record never makes a generation trusted.
#[test]
fn an_integrity_state_refuses_what_the_record_refuses() {
    let mut record = IndexIntegrityRecord::new(IndexId::from_raw(1), GenerationId::from_raw(2));
    record.report(&extra(7));
    record.check = Some(IndexCheck {
        state: CheckState::Failed {
            reason: "stopped".into(),
        },
        ..IndexCheck::accepted(1)
    });
    record.integrity = Integrity::Verified;
    record.trusted_from = Some(77);
    let bytes = rmp_serde::to_vec(&record).expect("encode");
    assert_eq!(
        IndexIntegrityRecord::state_from_msgpack(&bytes).expect("state"),
        record.state()
    );

    let verified = Integrity::Verified;
    let none: Option<u8> = None;
    let malformed = [
        (
            "an index that is no number",
            rmp_serde::to_vec(&(
                2u64,
                true,
                verified,
                Vec::<u8>::new(),
                0u64,
                none,
                Some(77u64),
            )),
        ),
        (
            "evidence that is no list",
            rmp_serde::to_vec(&(2u64, 1u64, verified, "x", 0u64, none, Some(77u64))),
        ),
        (
            "a disagreement that is none",
            rmp_serde::to_vec(&(2u64, 1u64, verified, vec!["x"], 0u64, none, Some(77u64))),
        ),
        (
            "a revision that is no number",
            rmp_serde::to_vec(&(
                2u64,
                1u64,
                verified,
                Vec::<u8>::new(),
                "x",
                none,
                Some(77u64),
            )),
        ),
        (
            "a check that is none",
            rmp_serde::to_vec(&(
                2u64,
                1u64,
                verified,
                Vec::<u8>::new(),
                0u64,
                Some("x"),
                Some(77u64),
            )),
        ),
    ];
    for (what, bytes) in malformed {
        let bytes = bytes.expect("encode");
        assert!(
            rmp_serde::from_slice::<IndexIntegrityRecord>(&bytes).is_err(),
            "{what}: the record decodes"
        );
        assert!(
            IndexIntegrityRecord::state_from_msgpack(&bytes).is_err(),
            "{what}: the state was read"
        );
    }
    let cut = &bytes[..bytes.len() - 1];
    assert!(rmp_serde::from_slice::<IndexIntegrityRecord>(cut).is_err());
    assert!(
        IndexIntegrityRecord::state_from_msgpack(cut).is_err(),
        "a cut record was read"
    );
}

/// Bytes encoded as MessagePack `bin`, for records written with a string
/// stored as bytes.
struct Bin(&'static [u8]);

impl serde::Serialize for Bin {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(self.0)
    }
}

/// [`CheckState`] with a failure's reason written as `bin`.
#[derive(serde::Serialize)]
#[allow(dead_code)]
enum BinCheckState {
    Accepted,
    Running { executor: u64 },
    Done,
    Failed { reason: Bin },
    Cancelled,
}

/// [`IndexCheck`] with its state written as [`BinCheckState`].
#[derive(serde::Serialize)]
struct BinCheck {
    state: BinCheckState,
    started_revision: u64,
    passes: u32,
    checked: u64,
    mismatches: u64,
    repaired: u64,
    rebuilt_into: Option<u64>,
    finished_at_ms: Option<u64>,
}

/// The state read without the evidence agrees with the whole record's
/// decode on every disagreement and check of every kind: what one decodes
/// the other gives the same state for. A failure's reason stored as `bin`
/// decodes as a string does when it is UTF-8, both ways, and is refused
/// both ways when it is not.
///
/// The variants are numbered by matches over the types themselves, which
/// stop compiling when a variant is added: that is the point where a new
/// variant must be noticed here. The cases are checked to cover the counts
/// written beside those matches, which are kept in step by hand; a variant
/// added with no count raised and no case written is not caught.
#[test]
fn an_integrity_state_agrees_with_the_record_on_every_shape() {
    const MISMATCHES: usize = 3;
    const CHECK_STATES: usize = 5;
    let mismatch_variant = |m: &Mismatch| match m {
        Mismatch::Missing { .. } => 0,
        Mismatch::Extra { .. } => 1,
        Mismatch::SourceDuplicate { .. } => 2,
    };
    let check_variant = |s: &CheckState| match s {
        CheckState::Accepted => 0,
        CheckState::Running { .. } => 1,
        CheckState::Done => 2,
        CheckState::Failed { .. } => 3,
        CheckState::Cancelled => 4,
    };
    let mismatches = [
        Mismatch::Missing {
            node: 1,
            valid_from: Some(-4),
            tuple: vec![1, 2],
        },
        Mismatch::Extra {
            node: 2,
            valid_from: None,
            tuple: vec![3],
        },
        Mismatch::SourceDuplicate {
            nodes: [5, 6],
            tuple: vec![],
        },
    ];
    let check_states = [
        CheckState::Accepted,
        CheckState::Running { executor: 3 },
        CheckState::Done,
        CheckState::Failed {
            reason: "stopped".into(),
        },
        CheckState::Cancelled,
    ];
    let mut seen = [false; MISMATCHES];
    for found in &mismatches {
        seen[mismatch_variant(found)] = true;
    }
    assert!(
        seen.iter().all(|s| *s),
        "a disagreement of some kind is untested"
    );
    let mut seen = [false; CHECK_STATES];
    for state in &check_states {
        seen[check_variant(state)] = true;
    }
    assert!(seen.iter().all(|s| *s), "a check state is untested");

    let mut record = IndexIntegrityRecord::new(IndexId::from_raw(1), GenerationId::from_raw(2));
    for found in &mismatches {
        record.report(found);
    }
    record.trusted_from = Some(9);
    for state in check_states {
        record.check = Some(IndexCheck {
            state,
            rebuilt_into: Some(GenerationId::from_raw(8)),
            finished_at_ms: Some(10),
            ..IndexCheck::accepted(1)
        });
        let bytes = rmp_serde::to_vec(&record).expect("encode");
        let whole: IndexIntegrityRecord = rmp_serde::from_slice(&bytes).expect("decode");
        assert_eq!(
            IndexIntegrityRecord::state_from_msgpack(&bytes).expect("state"),
            whole.state()
        );
    }

    let with_reason = |reason: &'static [u8]| {
        rmp_serde::to_vec(&(
            2u64,
            1u64,
            Integrity::Verified,
            Vec::<u8>::new(),
            0u64,
            Some(BinCheck {
                state: BinCheckState::Failed {
                    reason: Bin(reason),
                },
                started_revision: 0,
                passes: 0,
                checked: 0,
                mismatches: 0,
                repaired: 0,
                rebuilt_into: None,
                finished_at_ms: None,
            }),
            Some(9u64),
        ))
        .expect("encode")
    };
    let utf8 = with_reason(b"x");
    let whole: IndexIntegrityRecord = rmp_serde::from_slice(&utf8).expect("a UTF-8 bin decodes");
    assert_eq!(
        IndexIntegrityRecord::state_from_msgpack(&utf8).expect("a UTF-8 bin is read"),
        whole.state()
    );
    let not_utf8 = with_reason(&[0xff]);
    assert!(rmp_serde::from_slice::<IndexIntegrityRecord>(&not_utf8).is_err());
    assert!(
        IndexIntegrityRecord::state_from_msgpack(&not_utf8).is_err(),
        "a bin that is no UTF-8 was read"
    );
}

/// A definition read for its kind, label and parts refuses whatever the
/// definition's own decode refuses: a field of the wrong type anywhere, in
/// the descriptor's strings, lists, filter, configurations, state or
/// owner, as well as in the identities. A sound definition with every
/// optional field set gives the same head both ways, and a label stored as
/// bytes is taken when it is UTF-8 and refused when not, both ways.
#[test]
fn a_definition_head_refuses_what_the_definition_refuses() {
    let mut def = IndexDescriptor::compound("thing_ab", "Thing", vec!["a".into(), "b".into()])
        .bind(IndexId::from_raw(3), GenerationId::from_raw(4));
    def.descriptor.description = Some("described".into());
    def.descriptor.filter = Some(PartialFilter::PropertyEquals {
        property: "kind".into(),
        value: "x".into(),
    });
    def.descriptor.vector_config = Some(VectorIndexConfig::default());
    let mut text = TextIndexConfig::default();
    text.fields.insert(
        "a".into(),
        TextFieldConfig {
            analyzer: "english".into(),
        },
    );
    def.descriptor.text_config = Some(text);
    def.descriptor.state = IndexState::Failed {
        reason: "stopped".into(),
    };
    def.descriptor.owner = Some("thing_ab_unique".into());
    def.propose_parts(
        coordinode_core::schema::label_id::LabelId::new(2).unwrap(),
        GenerationId::from_raw(9),
        GenerationId::from_raw(10),
        GenerationId::from_raw(11),
    );
    let bytes = rmp_serde::to_vec(&def).expect("encode");
    let head = |bytes: &[u8]| {
        IndexDefinition::with_head(bytes, |h| {
            (h.index_type, h.label.to_owned(), h.parts.to_vec())
        })
    };
    assert_eq!(
        head(&bytes).expect("head"),
        (IndexType::BTree, "Thing".to_owned(), def.selection.clone())
    );

    // The record as a MessagePack value: [id, generation, descriptor, parts],
    // the descriptor its fields in order.
    let value: rmpv::Value = rmpv::decode::read_value(&mut bytes.as_slice()).expect("value");
    let with = |path: &[usize], replacement: rmpv::Value| {
        let mut value = value.clone();
        let mut at = &mut value;
        for &i in path {
            let items = match at {
                rmpv::Value::Array(items) => Some(items),
                _ => None,
            };
            at = &mut items.expect("an array on the path")[i];
        }
        *at = replacement;
        let mut out = Vec::new();
        rmpv::encode::write_value(&mut out, &value).expect("encode");
        out
    };
    use rmpv::Value as V;
    let malformed = [
        ("an id that is no number", with(&[0], V::Boolean(true))),
        ("a generation that is no number", with(&[1], V::from("x"))),
        ("a name that is no string", with(&[2, 0], V::from(5))),
        (
            "properties that are no strings",
            with(&[2, 3], V::Array(vec![V::from(1)])),
        ),
        ("a flag that is no boolean", with(&[2, 5], V::from("x"))),
        ("a filter of no shape", with(&[2, 8], V::from("x"))),
        (
            "a vector configuration of no shape",
            with(&[2, 10], V::Boolean(true)),
        ),
        (
            "a text configuration of no shape",
            with(&[2, 11], V::from(1)),
        ),
        ("a state of no kind", with(&[2, 12], V::from("Gone"))),
        ("a layout that is no number", with(&[2, 14], V::from("x"))),
        ("an owner that is no string", with(&[2, 16], V::from(1))),
    ];
    for (what, bytes) in malformed {
        assert!(
            rmp_serde::from_slice::<IndexDefinition>(&bytes).is_err(),
            "{what}: the definition decodes"
        );
        assert!(head(&bytes).is_err(), "{what}: the head was read");
    }

    let label_utf8 = with(&[2, 2], V::Binary(b"Thing".to_vec()));
    let whole: IndexDefinition = rmp_serde::from_slice(&label_utf8).expect("a UTF-8 bin label");
    assert_eq!(head(&label_utf8).expect("head").1, whole.label);
    let label_not_utf8 = with(&[2, 2], V::Binary(vec![0xff]));
    assert!(rmp_serde::from_slice::<IndexDefinition>(&label_not_utf8).is_err());
    assert!(
        head(&label_not_utf8).is_err(),
        "a label that is no UTF-8 was read"
    );
}

/// A check is pending until it reaches an outcome.
#[test]
fn a_check_is_pending_until_its_outcome() {
    let mut record = IndexIntegrityRecord::new(IndexId::from_raw(1), GenerationId::from_raw(2));
    record.check = Some(IndexCheck::accepted(0));
    assert!(record.check_pending());
    for state in [CheckState::Running { executor: 3 }, CheckState::Accepted] {
        record.check.as_mut().expect("check").state = state;
        assert!(record.check_pending());
    }
    for state in [
        CheckState::Done,
        CheckState::Cancelled,
        CheckState::Failed { reason: "x".into() },
    ] {
        record.check.as_mut().expect("check").state = state;
        assert!(!record.check_pending());
    }
}

/// The record round-trips through its stored form, and its key sits outside
/// the definition prefix, so a definition listing never meets it.
#[test]
fn integrity_record_roundtrip_and_key() {
    let mut record = IndexIntegrityRecord::new(IndexId::from_raw(4), GenerationId::from_raw(9));
    record.report(&Mismatch::SourceDuplicate {
        nodes: [1, 2],
        tuple: vec![9],
    });
    record.check = Some(IndexCheck::accepted(record.evidence_revision));
    let bytes = rmp_serde::to_vec(&record).expect("encode");
    let back: IndexIntegrityRecord = rmp_serde::from_slice(&bytes).expect("decode");
    assert_eq!(back, record);
    let key = IndexIntegrityRecord::key_of(GenerationId::from_raw(9));
    assert!(key.starts_with(IndexIntegrityRecord::PREFIX));
    assert!(!key.starts_with(IndexDefinition::SCHEMA_PREFIX));
}
