use super::*;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};

fn test_engine(dir: &std::path::Path) -> StorageEngine {
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    StorageEngine::open(&config).expect("open engine")
}

#[test]
fn register_and_lookup() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();

    let idx = IndexDefinition::btree("user_email", "User", "email").unique();
    reg.register(&engine, idx).expect("register");

    assert_eq!(reg.len(), 1);
    assert!(reg.get("user_email").is_some());
    assert_eq!(reg.indexes_for_label("User").len(), 1);
    assert_eq!(reg.indexes_for_label("Movie").len(), 0);
}

#[test]
fn indexes_for_property() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();

    reg.register(
        &engine,
        IndexDefinition::btree("user_email", "User", "email"),
    )
    .expect("register");
    reg.register(&engine, IndexDefinition::btree("user_name", "User", "name"))
        .expect("register");

    let email_idxs = reg.indexes_for_property("User", "email");
    assert_eq!(email_idxs.len(), 1);
    assert_eq!(email_idxs[0].name, "user_email");
}

#[test]
fn on_node_created_unique_ok() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();

    reg.register(
        &engine,
        IndexDefinition::btree("user_email", "User", "email").unique(),
    )
    .expect("register");

    // Create first node — should succeed
    reg.on_node_created(
        &engine,
        NodeId::from_raw(1),
        "User",
        &[("email".to_string(), Value::String("alice@test.com".into()))],
    )
    .expect("first create");

    // Create second node with different email — should succeed
    reg.on_node_created(
        &engine,
        NodeId::from_raw(2),
        "User",
        &[("email".to_string(), Value::String("bob@test.com".into()))],
    )
    .expect("second create");
}

#[test]
fn on_node_created_unique_violation() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();

    reg.register(
        &engine,
        IndexDefinition::btree("user_email", "User", "email").unique(),
    )
    .expect("register");

    reg.on_node_created(
        &engine,
        NodeId::from_raw(1),
        "User",
        &[("email".to_string(), Value::String("alice@test.com".into()))],
    )
    .expect("first");

    // Duplicate email — should fail
    let result = reg.on_node_created(
        &engine,
        NodeId::from_raw(2),
        "User",
        &[("email".to_string(), Value::String("alice@test.com".into()))],
    );

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("unique constraint violated"));
    assert!(err.to_string().contains("user_email"));
}

#[test]
fn on_property_changed_updates_index() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();

    reg.register(
        &engine,
        IndexDefinition::btree("user_email", "User", "email").unique(),
    )
    .expect("register");

    // Create initial entry
    reg.on_node_created(
        &engine,
        NodeId::from_raw(1),
        "User",
        &[("email".to_string(), Value::String("old@test.com".into()))],
    )
    .expect("create");

    // Update property
    reg.on_property_changed(
        &engine,
        NodeId::from_raw(1),
        "User",
        "email",
        Some(&Value::String("old@test.com".into())),
        &Value::String("new@test.com".into()),
    )
    .expect("update");

    // Old value should not be findable
    let old = super::super::ops::index_scan_exact(
        &engine,
        "user_email",
        &Value::String("old@test.com".into()),
    )
    .expect("scan");
    assert!(old.is_empty());

    // New value should be findable
    let new = super::super::ops::index_scan_exact(
        &engine,
        "user_email",
        &Value::String("new@test.com".into()),
    )
    .expect("scan");
    assert_eq!(new, vec![1]);
}

#[test]
fn on_property_changed_unique_violation() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();

    reg.register(
        &engine,
        IndexDefinition::btree("user_email", "User", "email").unique(),
    )
    .expect("register");

    // Two nodes with different emails
    reg.on_node_created(
        &engine,
        NodeId::from_raw(1),
        "User",
        &[("email".to_string(), Value::String("alice@test.com".into()))],
    )
    .expect("create 1");

    reg.on_node_created(
        &engine,
        NodeId::from_raw(2),
        "User",
        &[("email".to_string(), Value::String("bob@test.com".into()))],
    )
    .expect("create 2");

    // Try to change node 2's email to alice's — should fail
    let result = reg.on_property_changed(
        &engine,
        NodeId::from_raw(2),
        "User",
        "email",
        Some(&Value::String("bob@test.com".into())),
        &Value::String("alice@test.com".into()),
    );
    assert!(result.is_err());
}

#[test]
fn on_node_deleted_removes_index() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();

    reg.register(
        &engine,
        IndexDefinition::btree("user_email", "User", "email"),
    )
    .expect("register");

    reg.on_node_created(
        &engine,
        NodeId::from_raw(1),
        "User",
        &[("email".to_string(), Value::String("alice@test.com".into()))],
    )
    .expect("create");

    // Delete
    reg.on_node_deleted(
        &engine,
        NodeId::from_raw(1),
        "User",
        &[("email".to_string(), Value::String("alice@test.com".into()))],
    )
    .expect("delete");

    // Should not be findable
    let results = super::super::ops::index_scan_exact(
        &engine,
        "user_email",
        &Value::String("alice@test.com".into()),
    )
    .expect("scan");
    assert!(results.is_empty());
}

#[test]
fn load_all_from_storage() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());

    // Save definitions
    {
        let reg = IndexRegistry::new();
        reg.register(
            &engine,
            IndexDefinition::btree("idx1", "User", "email").unique(),
        )
        .expect("register");
        reg.register(&engine, IndexDefinition::btree("idx2", "User", "name"))
            .expect("register");
    }

    // Load in new registry
    let reg2 = IndexRegistry::new();
    reg2.load_all(&engine).expect("load");
    assert_eq!(reg2.len(), 2);
    assert!(reg2.get("idx1").is_some());
    assert!(reg2.get("idx2").is_some());
}

#[test]
fn sparse_index_skips_null_on_create() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();

    reg.register(
        &engine,
        IndexDefinition::btree("user_bio", "User", "bio").sparse(),
    )
    .expect("register");

    // Create node with null bio — should be skipped
    reg.on_node_created(
        &engine,
        NodeId::from_raw(1),
        "User",
        &[("bio".to_string(), Value::Null)],
    )
    .expect("create");

    let idx = reg.get("user_bio").expect("get");
    let results = super::super::ops::index_scan(&engine, &idx).expect("scan");
    assert!(results.is_empty());
}

// ====== Partial index ======

#[test]
fn partial_index_filters_on_create() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();

    // Index only active users
    reg.register(
        &engine,
        IndexDefinition::btree("active_email", "User", "email").with_filter(
            super::super::definition::PartialFilter::PropertyEquals {
                property: "status".into(),
                value: "active".into(),
            },
        ),
    )
    .expect("register");

    // Active user — should be indexed
    reg.on_node_created(
        &engine,
        NodeId::from_raw(1),
        "User",
        &[
            ("email".to_string(), Value::String("alice@test.com".into())),
            ("status".to_string(), Value::String("active".into())),
        ],
    )
    .expect("create active");

    // Inactive user — should NOT be indexed
    reg.on_node_created(
        &engine,
        NodeId::from_raw(2),
        "User",
        &[
            ("email".to_string(), Value::String("bob@test.com".into())),
            ("status".to_string(), Value::String("inactive".into())),
        ],
    )
    .expect("create inactive");

    // Only active user should be in index
    let idx = reg.get("active_email").expect("get");
    let results = super::super::ops::index_scan(&engine, &idx).expect("scan");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0], 1); // Only Alice
}

#[test]
fn partial_index_bool_filter() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();

    reg.register(
        &engine,
        IndexDefinition::btree("verified_email", "User", "email").with_filter(
            super::super::definition::PartialFilter::PropertyEqualsBool {
                property: "verified".into(),
                value: true,
            },
        ),
    )
    .expect("register");

    // Verified user
    reg.on_node_created(
        &engine,
        NodeId::from_raw(1),
        "User",
        &[
            ("email".to_string(), Value::String("alice@test.com".into())),
            ("verified".to_string(), Value::Bool(true)),
        ],
    )
    .expect("create");

    // Unverified user
    reg.on_node_created(
        &engine,
        NodeId::from_raw(2),
        "User",
        &[
            ("email".to_string(), Value::String("bob@test.com".into())),
            ("verified".to_string(), Value::Bool(false)),
        ],
    )
    .expect("create");

    let idx = reg.get("verified_email").expect("get");
    let results = super::super::ops::index_scan(&engine, &idx).expect("scan");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0], 1);
}

#[test]
fn partial_index_exists_filter() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();

    reg.register(
        &engine,
        IndexDefinition::btree("user_bio_idx", "User", "bio").with_filter(
            super::super::definition::PartialFilter::PropertyExists {
                property: "bio".into(),
            },
        ),
    )
    .expect("register");

    // User WITH bio
    reg.on_node_created(
        &engine,
        NodeId::from_raw(1),
        "User",
        &[
            ("name".to_string(), Value::String("Alice".into())),
            ("bio".to_string(), Value::String("Developer".into())),
        ],
    )
    .expect("create");

    // User WITHOUT bio (null)
    reg.on_node_created(
        &engine,
        NodeId::from_raw(2),
        "User",
        &[
            ("name".to_string(), Value::String("Bob".into())),
            ("bio".to_string(), Value::Null),
        ],
    )
    .expect("create");

    let idx = reg.get("user_bio_idx").expect("get");
    let results = super::super::ops::index_scan(&engine, &idx).expect("scan");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0], 1);
}

#[test]
fn partial_filter_matches_function() {
    use super::super::definition::PartialFilter;

    let props = vec![
        ("status".to_string(), Value::String("active".into())),
        ("age".to_string(), Value::Int(30)),
    ];

    // String equality
    assert!(PartialFilter::PropertyEquals {
        property: "status".into(),
        value: "active".into(),
    }
    .matches(&props));

    assert!(!PartialFilter::PropertyEquals {
        property: "status".into(),
        value: "inactive".into(),
    }
    .matches(&props));

    // Int equality
    assert!(PartialFilter::PropertyEqualsInt {
        property: "age".into(),
        value: 30,
    }
    .matches(&props));

    // Exists
    assert!(PartialFilter::PropertyExists {
        property: "status".into(),
    }
    .matches(&props));

    // Not exists
    assert!(!PartialFilter::PropertyExists {
        property: "missing".into(),
    }
    .matches(&props));
}
