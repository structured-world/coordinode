use super::*;

#[test]
fn btree_definition() {
    let idx = IndexDefinition::btree("user_email", "User", "email").unique();
    assert_eq!(idx.name, "user_email");
    assert_eq!(idx.label, "User");
    assert_eq!(idx.property(), "email");
    assert_eq!(idx.properties, vec!["email"]);
    assert!(idx.unique);
    assert!(!idx.sparse);
    assert!(!idx.multikey);
    assert!(!idx.is_compound());
}

#[test]
fn compound_definition() {
    let idx = IndexDefinition::compound(
        "user_label_status",
        "User",
        vec!["label".into(), "status".into()],
    );
    assert!(idx.is_compound());
    assert_eq!(idx.properties.len(), 2);
    assert_eq!(idx.property(), "label");
}

#[test]
fn sparse_definition() {
    let idx = IndexDefinition::btree("user_bio", "User", "bio").sparse();
    assert!(idx.sparse);
}

/// A definition stored before the layout field existed decodes as layout 0,
/// the one to rebuild; a new definition carries the current layout.
#[test]
fn a_definition_without_a_layout_is_the_legacy_one() {
    #[derive(serde::Serialize)]
    struct BeforeLayout {
        name: String,
        label: String,
        properties: Vec<String>,
        index_type: IndexType,
        unique: bool,
        sparse: bool,
        multikey: bool,
        filter: Option<PartialFilter>,
        ttl_seconds: Option<u64>,
        vector_config: Option<VectorIndexConfig>,
        text_config: Option<TextIndexConfig>,
        state: IndexState,
        online_during_build: OnlineDuringBuild,
    }
    let bytes = rmp_serde::to_vec(&BeforeLayout {
        name: "u_email".into(),
        label: "User".into(),
        properties: vec!["email".into()],
        index_type: IndexType::BTree,
        unique: true,
        sparse: false,
        multikey: false,
        filter: None,
        ttl_seconds: None,
        vector_config: None,
        text_config: None,
        state: IndexState::Ready,
        online_during_build: OnlineDuringBuild::Block,
    })
    .expect("encode");
    let back: IndexDefinition = rmp_serde::from_slice(&bytes).expect("decode");
    assert_eq!(back.layout, 0);
    assert!(back.unique);

    let current = IndexDefinition::btree("u_email", "User", "email");
    assert_eq!(current.layout, ENTRY_LAYOUT);
    let back: IndexDefinition =
        rmp_serde::from_slice(&rmp_serde::to_vec(&current).expect("encode")).expect("decode");
    assert_eq!(back.layout, ENTRY_LAYOUT);
}

#[test]
fn schema_key() {
    let idx = IndexDefinition::btree("user_email", "User", "email");
    assert_eq!(idx.schema_key(), b"schema:idx:user_email");
}

#[test]
fn new_indexes_default_to_ready_state() {
    let btree = IndexDefinition::btree("u_email", "User", "email");
    let hnsw = IndexDefinition::hnsw("u_vec", "User", "vec", VectorIndexConfig::default());
    let compound = IndexDefinition::compound(
        "u_lbl_status",
        "User",
        vec!["label".into(), "status".into()],
    );
    let text = IndexDefinition::text(
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
fn state_roundtrip_serde() {
    let mut idx = IndexDefinition::hnsw("v", "L", "p", VectorIndexConfig::default());
    idx.state = IndexState::Building {
        written: 1234,
        estimated_total: 9999,
    };
    let bytes = rmp_serde::to_vec(&idx).expect("encode");
    let back: IndexDefinition = rmp_serde::from_slice(&bytes).expect("decode");
    assert_eq!(back.state, idx.state);

    idx.state = IndexState::Failed {
        reason: "build aborted".to_string(),
    };
    let bytes = rmp_serde::to_vec(&idx).expect("encode failed");
    let back: IndexDefinition = rmp_serde::from_slice(&bytes).expect("decode failed");
    assert_eq!(
        back.state,
        IndexState::Failed {
            reason: "build aborted".to_string()
        }
    );
}

#[test]
fn legacy_def_without_state_deserializes_as_ready() {
    // Simulate a pre-state IndexDefinition record by encoding a struct
    // that has the same field layout MINUS the `state` field. rmp-serde
    // accepts the shorter struct because we marked `state` with
    // `#[serde(default)]`.
    #[derive(serde::Serialize)]
    struct LegacyDef {
        name: String,
        label: String,
        properties: Vec<String>,
        index_type: IndexType,
        unique: bool,
        sparse: bool,
        multikey: bool,
        filter: Option<PartialFilter>,
        ttl_seconds: Option<u64>,
        vector_config: Option<VectorIndexConfig>,
    }
    let legacy = LegacyDef {
        name: "u".into(),
        label: "U".into(),
        properties: vec!["v".into()],
        index_type: IndexType::Hnsw,
        unique: false,
        sparse: true,
        multikey: false,
        filter: None,
        ttl_seconds: None,
        vector_config: Some(VectorIndexConfig::default()),
    };
    let bytes = rmp_serde::to_vec(&legacy).expect("encode legacy");
    let back: IndexDefinition = rmp_serde::from_slice(&bytes).expect("decode legacy as current");
    assert_eq!(back.state, IndexState::Ready);
    assert_eq!(back.name, "u");
}
