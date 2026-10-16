use super::*;
use crate::index::TextIndexConfig;
use tempfile::TempDir;

fn test_config() -> TextIndexConfig {
    TextIndexConfig {
        fields: HashMap::new(),
        default_language: "english".to_string(),
        language_override_property: "_language".to_string(),
    }
}

/// The text index `name` as the index numbered `raw`, serving from
/// generation `raw`: the registry is driven without a catalog here.
fn text(
    raw: u64,
    name: &str,
    label: &str,
    properties: Vec<String>,
    config: TextIndexConfig,
) -> IndexDefinition {
    crate::index::IndexDescriptor::text(name, label, properties, config).bind(
        crate::index::IndexId::from_raw(raw),
        crate::index::GenerationId::from_raw(raw),
    )
}

#[test]
fn register_and_lookup() {
    let dir = TempDir::new().unwrap();
    let reg = TextIndexRegistry::new(dir.path());
    let def = text(
        1,
        "article_body",
        "Article",
        vec!["body".into()],
        test_config(),
    );
    reg.register(def).unwrap();

    assert!(reg.has_index("Article", "body"));
    assert!(!reg.has_index("Article", "title"));
    assert!(!reg.has_index("User", "body"));
    assert_eq!(reg.len(), 1);
}

/// Two paths can register one new definition: the statement that created it
/// and the member's follower of applied definitions. The second registration
/// of the same generation leaves the index as it is, and says it did not
/// register it, so only the first rebuilds. Recreating it instead emptied
/// the index the first had just filled and, run beside that rebuild,
/// removed the files it was writing.
#[test]
fn registering_the_same_generation_again_keeps_the_index() {
    let dir = TempDir::new().unwrap();
    let reg = TextIndexRegistry::new(dir.path());
    let def = text(
        1,
        "article_body",
        "Article",
        vec!["body".into()],
        test_config(),
    );
    assert!(reg.register(def.clone()).unwrap(), "the first registers it");
    reg.on_text_written(
        "Article",
        NodeId::from_raw(1),
        "body",
        "Rust graph database engine",
    );

    let second = reg.register(def).unwrap();
    let results = reg.search("Article", "body", "rust", 10).unwrap();
    assert_eq!(results.len(), 1, "the index keeps what it held");
    assert!(!second, "the second finds it registered");
}

#[test]
fn index_and_search() {
    let dir = TempDir::new().unwrap();
    let reg = TextIndexRegistry::new(dir.path());
    let def = text(
        1,
        "article_body",
        "Article",
        vec!["body".into()],
        test_config(),
    );
    reg.register(def).unwrap();

    reg.on_text_written(
        "Article",
        NodeId::from_raw(1),
        "body",
        "Rust graph database engine",
    );
    reg.on_text_written(
        "Article",
        NodeId::from_raw(2),
        "body",
        "Python machine learning framework",
    );
    reg.on_text_written(
        "Article",
        NodeId::from_raw(3),
        "body",
        "Rust async runtime tokio",
    );

    let results = reg.search("Article", "body", "rust", 10).unwrap();
    assert_eq!(results.len(), 2);
    // Both rust docs should be found
    let ids: Vec<u64> = results.iter().map(|r| r.node_id).collect();
    assert!(ids.contains(&1));
    assert!(ids.contains(&3));
}

#[test]
fn delete_document() {
    let dir = TempDir::new().unwrap();
    let reg = TextIndexRegistry::new(dir.path());
    let def = text(
        1,
        "article_body",
        "Article",
        vec!["body".into()],
        test_config(),
    );
    reg.register(def).unwrap();

    reg.on_text_written(
        "Article",
        NodeId::from_raw(1),
        "body",
        "Rust graph database",
    );
    let results = reg.search("Article", "body", "rust", 10).unwrap();
    assert_eq!(results.len(), 1);

    reg.on_text_deleted("Article", NodeId::from_raw(1), "body");
    let results = reg.search("Article", "body", "rust", 10).unwrap();
    assert_eq!(results.len(), 0);
}

#[test]
fn unregister() {
    let dir = TempDir::new().unwrap();
    let reg = TextIndexRegistry::new(dir.path());
    let def = text(
        1,
        "article_body",
        "Article",
        vec!["body".into()],
        test_config(),
    );
    reg.register(def).unwrap();
    assert!(reg.has_index("Article", "body"));

    reg.unregister("Article", "body");
    assert!(!reg.has_index("Article", "body"));
    assert_eq!(reg.len(), 0);
}

#[test]
fn no_index_search_returns_none() {
    let dir = TempDir::new().unwrap();
    let reg = TextIndexRegistry::new(dir.path());
    assert!(reg.search("Article", "body", "rust", 10).is_none());
}

/// Multi-field index: same tantivy handle registered under all properties.
#[test]
fn multi_field_register_all_properties() {
    let dir = TempDir::new().unwrap();
    let reg = TextIndexRegistry::new(dir.path());

    let mut fields = HashMap::new();
    fields.insert(
        "title".to_string(),
        crate::index::TextFieldConfig {
            analyzer: "english".to_string(),
        },
    );
    fields.insert(
        "body".to_string(),
        crate::index::TextFieldConfig {
            analyzer: "english".to_string(),
        },
    );
    let config = TextIndexConfig {
        fields,
        default_language: "english".to_string(),
        language_override_property: "_language".to_string(),
    };
    let def = text(
        1,
        "article_text",
        "Article",
        vec!["title".into(), "body".into()],
        config,
    );
    reg.register(def).unwrap();

    // Both properties should be registered.
    assert!(reg.has_index("Article", "title"));
    assert!(reg.has_index("Article", "body"));
    // Two entries in the HashMap but one underlying index.
    assert_eq!(reg.len(), 2);

    // Write + search on second property should work.
    reg.on_text_written("Article", NodeId::from_raw(1), "body", "Rust graph engine");
    let results = reg.search("Article", "body", "rust", 10).unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn indexes_for_label() {
    let dir = TempDir::new().unwrap();
    let reg = TextIndexRegistry::new(dir.path());
    reg.register(text(
        1,
        "article_body",
        "Article",
        vec!["body".into()],
        test_config(),
    ))
    .unwrap();
    reg.register(text(
        2,
        "article_title",
        "Article",
        vec!["title".into()],
        test_config(),
    ))
    .unwrap();
    reg.register(text(
        3,
        "user_bio",
        "User",
        vec!["bio".into()],
        test_config(),
    ))
    .unwrap();

    assert!(reg.has_index("Article", "body"));
    assert!(reg.has_index("Article", "title"));
    assert!(reg.has_index("User", "bio"));
    assert_eq!(reg.len(), 3);
}
