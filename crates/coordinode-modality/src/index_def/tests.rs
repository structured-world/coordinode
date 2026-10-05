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
