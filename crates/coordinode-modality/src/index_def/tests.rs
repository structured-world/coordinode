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
    assert!(record.report(extra(7)));
    assert_eq!(record.integrity, Integrity::Suspect);
    assert_eq!(record.evidence_revision, 1);
    assert!(!record.report(extra(7)), "the same disagreement again");
    assert_eq!(record.evidence_revision, 1);
    assert!(record.report(extra(8)));
    assert_eq!(record.evidence, vec![extra(7), extra(8)]);
    assert_eq!(record.evidence_revision, 2);
}

/// Past the evidence a record keeps, reports still move the revision and
/// keep the generation suspect: a check started before them sees it.
#[test]
fn reports_past_the_cap_still_move_the_revision() {
    let mut record = IndexIntegrityRecord::new(IndexId::from_raw(1), GenerationId::from_raw(2));
    for node in 0..IndexIntegrityRecord::MAX_EVIDENCE as u64 + 5 {
        assert!(record.report(extra(node)));
    }
    assert_eq!(record.evidence.len(), IndexIntegrityRecord::MAX_EVIDENCE);
    assert_eq!(
        record.evidence_revision,
        IndexIntegrityRecord::MAX_EVIDENCE as u64 + 5
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
    record.report(Mismatch::SourceDuplicate {
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
