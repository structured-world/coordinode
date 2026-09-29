use super::*;
use coordinode_core::txn::timestamp::Timestamp;
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

/// A direct-mode transaction: its writes land as they are staged, so each
/// step below sees the ones before it.
fn txn(engine: &StorageEngine) -> Transaction<'_> {
    Transaction::new(engine, None, Timestamp::ZERO, None)
}

fn props(pairs: &[(&str, Value)]) -> Vec<(String, Value)> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), v.clone()))
        .collect()
}

fn s(v: &str) -> Value {
    Value::String(v.into())
}

/// No property is bound to a field id: the tests keep values by name.
fn no_fields(_: &str) -> Option<u32> {
    None
}

fn create(
    reg: &IndexRegistry,
    engine: &StorageEngine,
    id: u64,
    pairs: &[(&str, Value)],
) -> Result<(), IndexWriteError> {
    let props = props(pairs);
    let lookup = props_lookup(&props);
    let mut t = txn(engine);
    reg.on_node_created(
        engine,
        &mut t,
        &NodeState {
            node_id: NodeId::from_raw(id),
            label: "User",
            value_of: &lookup,
        },
        &no_fields,
        &mut Vec::new(),
    )
}

fn change(
    reg: &IndexRegistry,
    engine: &StorageEngine,
    id: u64,
    property: &str,
    before: &[(&str, Value)],
    after: &[(&str, Value)],
) -> Result<(), IndexWriteError> {
    let (before, after) = (props(before), props(after));
    let (before, after) = (props_lookup(&before), props_lookup(&after));
    let mut t = txn(engine);
    reg.on_property_changed(
        engine,
        &mut t,
        &PropertyChange {
            node_id: NodeId::from_raw(id),
            label: "User",
            properties: &[property],
            before: &before,
            after: &after,
        },
        &no_fields,
        &mut Vec::new(),
    )
}

fn lookup(engine: &StorageEngine, index: &IndexDefinition, values: &[Value]) -> Vec<u64> {
    let mut ids: Vec<u64> = LocalIndexStore::new(engine)
        .scan_exact(&mut txn(engine), index, values)
        .expect("scan")
        .expect("indexable")
        .into_iter()
        .map(|n| n.as_raw())
        .collect();
    ids.sort_unstable();
    ids
}

fn all_ids(engine: &StorageEngine, index: &IndexDefinition) -> Vec<u64> {
    LocalIndexStore::new(engine)
        .scan_entry_ids(&mut txn(engine), index)
        .expect("scan")
        .into_iter()
        .map(|n| n.as_raw())
        .collect()
}

#[test]
fn register_and_lookup() {
    let reg = IndexRegistry::new();
    reg.register_in_memory(IndexDefinition::btree("user_email", "User", "email").unique());

    assert_eq!(reg.len(), 1);
    assert!(reg.get("user_email").is_some());
    assert_eq!(reg.indexes_for_label("User").len(), 1);
    assert_eq!(reg.indexes_for_label("Movie").len(), 0);
}

#[test]
fn indexes_for_property() {
    let reg = IndexRegistry::new();
    reg.register_in_memory(IndexDefinition::btree("user_email", "User", "email"));
    reg.register_in_memory(IndexDefinition::btree("user_name", "User", "name"));

    let email_idxs = reg.indexes_for_property("User", "email");
    assert_eq!(email_idxs.len(), 1);
    assert_eq!(email_idxs[0].name, "user_email");
}

#[test]
fn distinct_unique_values_are_accepted() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();
    reg.register_in_memory(IndexDefinition::btree("user_email", "User", "email").unique());

    create(&reg, &engine, 1, &[("email", s("alice@test.com"))]).expect("first create");
    create(&reg, &engine, 2, &[("email", s("bob@test.com"))]).expect("second create");
}

/// A second node taking a unique value is refused, naming the index and the
/// node that holds the value.
#[test]
fn a_taken_unique_value_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();
    reg.register_in_memory(IndexDefinition::btree("user_email", "User", "email").unique());

    create(&reg, &engine, 1, &[("email", s("alice@test.com"))]).expect("first");
    let err =
        create(&reg, &engine, 2, &[("email", s("alice@test.com"))]).expect_err("duplicate email");

    assert!(
        matches!(&err, IndexWriteError::Unique(v) if v.holder == NodeId::from_raw(1)),
        "expected a unique violation held by node 1, got {err:?}"
    );
    assert!(err.to_string().contains("unique constraint violated"));
    assert!(err.to_string().contains("user_email"));
}

#[test]
fn a_changed_value_moves_its_entry() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();
    let index = IndexDefinition::btree("user_email", "User", "email").unique();
    reg.register_in_memory(index.clone());

    create(&reg, &engine, 1, &[("email", s("old@test.com"))]).expect("create");
    change(
        &reg,
        &engine,
        1,
        "email",
        &[("email", s("old@test.com"))],
        &[("email", s("new@test.com"))],
    )
    .expect("update");

    assert!(lookup(&engine, &index, &[s("old@test.com")]).is_empty());
    assert_eq!(lookup(&engine, &index, &[s("new@test.com")]), vec![1]);
}

#[test]
fn changing_to_a_taken_unique_value_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();
    reg.register_in_memory(IndexDefinition::btree("user_email", "User", "email").unique());

    create(&reg, &engine, 1, &[("email", s("alice@test.com"))]).expect("create 1");
    create(&reg, &engine, 2, &[("email", s("bob@test.com"))]).expect("create 2");
    let result = change(
        &reg,
        &engine,
        2,
        "email",
        &[("email", s("bob@test.com"))],
        &[("email", s("alice@test.com"))],
    );
    assert!(matches!(result, Err(IndexWriteError::Unique(_))));
}

/// A compound index reads several properties; changing one of them moves
/// the node's single entry to the new combination.
#[test]
fn a_compound_entry_moves_when_one_of_its_properties_changes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();
    let index =
        IndexDefinition::compound("user_city_age", "User", vec!["city".into(), "age".into()]);
    reg.register_in_memory(index.clone());

    let before = [("city", s("Oslo")), ("age", Value::Int(30))];
    let after = [("city", s("Oslo")), ("age", Value::Int(31))];
    create(&reg, &engine, 1, &before).expect("create");
    change(&reg, &engine, 1, "age", &before, &after).expect("change");

    assert!(lookup(&engine, &index, &[s("Oslo"), Value::Int(30)]).is_empty());
    assert_eq!(
        lookup(&engine, &index, &[s("Oslo"), Value::Int(31)]),
        vec![1]
    );
}

#[test]
fn a_deleted_node_leaves_the_index() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();
    let index = IndexDefinition::btree("user_email", "User", "email");
    reg.register_in_memory(index.clone());

    create(&reg, &engine, 1, &[("email", s("alice@test.com"))]).expect("create");
    let props = props(&[("email", s("alice@test.com"))]);
    reg.on_node_deleted(
        &engine,
        &mut txn(&engine),
        &NodeState {
            node_id: NodeId::from_raw(1),
            label: "User",
            value_of: &props_lookup(&props),
        },
        &no_fields,
    )
    .expect("delete");

    assert!(lookup(&engine, &index, &[s("alice@test.com")]).is_empty());
}

/// Only B-tree indexes have entries: a vector index registered alongside
/// for planning and advice is not maintained here.
#[test]
fn a_vector_index_in_the_registry_gets_no_entries() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();
    let vector = IndexDefinition::hnsw(
        "user_vec",
        "User",
        "email",
        crate::index::definition::VectorIndexConfig::default(),
    );
    reg.register_in_memory(vector.clone());

    assert!(!reg.has_btree_for("User"));
    create(&reg, &engine, 1, &[("email", s("alice@test.com"))]).expect("create");
    assert!(all_ids(&engine, &vector).is_empty());
}

#[test]
fn load_all_from_storage() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    super::super::ops::save_index_definition(
        &engine,
        &IndexDefinition::btree("idx1", "User", "email").unique(),
    )
    .expect("save");
    super::super::ops::save_index_definition(
        &engine,
        &IndexDefinition::btree("idx2", "User", "name"),
    )
    .expect("save");

    let reg = IndexRegistry::new();
    reg.load_all(&engine).expect("load");
    assert_eq!(reg.len(), 2);
    assert!(reg.get("idx1").is_some());
    assert!(reg.get("idx2").is_some());
}

#[test]
fn sparse_index_skips_null_on_create() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();
    let index = IndexDefinition::btree("user_bio", "User", "bio").sparse();
    reg.register_in_memory(index.clone());

    create(&reg, &engine, 1, &[("bio", Value::Null)]).expect("create");
    assert!(all_ids(&engine, &index).is_empty());
}

// ====== Partial index ======

#[test]
fn partial_index_filters_on_create() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();
    let index = IndexDefinition::btree("active_email", "User", "email").with_filter(
        super::super::definition::PartialFilter::PropertyEquals {
            property: "status".into(),
            value: "active".into(),
        },
    );
    reg.register_in_memory(index.clone());

    create(
        &reg,
        &engine,
        1,
        &[("email", s("alice@test.com")), ("status", s("active"))],
    )
    .expect("create active");
    create(
        &reg,
        &engine,
        2,
        &[("email", s("bob@test.com")), ("status", s("inactive"))],
    )
    .expect("create inactive");

    assert_eq!(all_ids(&engine, &index), vec![1]);
}

/// Changing the property a partial index filters on moves the node into or
/// out of the index, though no indexed value changed.
#[test]
fn a_node_leaves_a_partial_index_when_its_filter_stops_matching() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();
    let index = IndexDefinition::btree("active_email", "User", "email").with_filter(
        super::super::definition::PartialFilter::PropertyEquals {
            property: "status".into(),
            value: "active".into(),
        },
    );
    reg.register_in_memory(index.clone());

    let active = [("email", s("alice@test.com")), ("status", s("active"))];
    let inactive = [("email", s("alice@test.com")), ("status", s("inactive"))];
    create(&reg, &engine, 1, &active).expect("create");
    change(&reg, &engine, 1, "status", &active, &inactive).expect("deactivate");

    assert!(all_ids(&engine, &index).is_empty());
}

#[test]
fn partial_index_bool_filter() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();
    let index = IndexDefinition::btree("verified_email", "User", "email").with_filter(
        super::super::definition::PartialFilter::PropertyEqualsBool {
            property: "verified".into(),
            value: true,
        },
    );
    reg.register_in_memory(index.clone());

    create(
        &reg,
        &engine,
        1,
        &[
            ("email", s("alice@test.com")),
            ("verified", Value::Bool(true)),
        ],
    )
    .expect("create");
    create(
        &reg,
        &engine,
        2,
        &[
            ("email", s("bob@test.com")),
            ("verified", Value::Bool(false)),
        ],
    )
    .expect("create");

    assert_eq!(all_ids(&engine, &index), vec![1]);
}

#[test]
fn partial_index_exists_filter() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();
    let index = IndexDefinition::btree("user_bio_idx", "User", "bio").with_filter(
        super::super::definition::PartialFilter::PropertyExists {
            property: "bio".into(),
        },
    );
    reg.register_in_memory(index.clone());

    create(
        &reg,
        &engine,
        1,
        &[("name", s("Alice")), ("bio", s("Developer"))],
    )
    .expect("create");
    create(
        &reg,
        &engine,
        2,
        &[("name", s("Bob")), ("bio", Value::Null)],
    )
    .expect("create");

    assert_eq!(all_ids(&engine, &index), vec![1]);
}

#[test]
fn partial_filter_matches_function() {
    use super::super::definition::PartialFilter;

    let props = vec![
        ("status".to_string(), Value::String("active".into())),
        ("age".to_string(), Value::Int(30)),
    ];

    assert!(
        PartialFilter::PropertyEquals {
            property: "status".into(),
            value: "active".into(),
        }
        .matches(&props)
    );
    assert!(
        !PartialFilter::PropertyEquals {
            property: "status".into(),
            value: "inactive".into(),
        }
        .matches(&props)
    );
    assert!(
        PartialFilter::PropertyEqualsInt {
            property: "age".into(),
            value: 30,
        }
        .matches(&props)
    );
    assert!(
        PartialFilter::PropertyExists {
            property: "status".into(),
        }
        .matches(&props)
    );
    assert!(
        !PartialFilter::PropertyExists {
            property: "missing".into(),
        }
        .matches(&props)
    );
}
