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

#[test]
fn register_and_lookup() {
    let dir = TempDir::new().unwrap();
    let reg = TextIndexRegistry::new(dir.path());
    let def = IndexDefinition::text(
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

#[test]
fn index_and_search() {
    let dir = TempDir::new().unwrap();
    let reg = TextIndexRegistry::new(dir.path());
    let def = IndexDefinition::text(
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
    let def = IndexDefinition::text(
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
    let def = IndexDefinition::text(
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
    let def = IndexDefinition::text(
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
    reg.register(IndexDefinition::text(
        "article_body",
        "Article",
        vec!["body".into()],
        test_config(),
    ))
    .unwrap();
    reg.register(IndexDefinition::text(
        "article_title",
        "Article",
        vec!["title".into()],
        test_config(),
    ))
    .unwrap();
    reg.register(IndexDefinition::text(
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
