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

/// The shard the tests' nodes live in.
const SHARD: u16 = 1;

/// Store the `User` node `id` with `pairs` as its properties, kept by name,
/// as a write stores the record its entries are derived from: a unique
/// entry's holder is checked against it.
fn put_record(t: &mut Transaction, id: u64, pairs: &[(&str, Value)]) {
    use coordinode_modality::{LocalNodeStore, NodeStore as _};
    let mut record = coordinode_core::graph::node::NodeRecord::new("User");
    for (name, value) in pairs {
        record.set_extra(*name, value.clone());
    }
    LocalNodeStore
        .put(t, SHARD, NodeId::from_raw(id), &record)
        .expect("put node record");
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
    put_record(&mut t, id, pairs);
    reg.on_node_created(
        engine,
        &mut t,
        SHARD,
        &NodeState {
            node_id: NodeId::from_raw(id),
            valid_from: None,
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
    let mut t = txn(engine);
    put_record(&mut t, id, after);
    let (before, after) = (props(before), props(after));
    let (before, after) = (props_lookup(&before), props_lookup(&after));
    reg.on_property_changed(
        engine,
        &mut t,
        SHARD,
        &PropertyChange {
            node_id: NodeId::from_raw(id),
            valid_from: None,
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

/// `descriptor` as the index numbered `raw`, serving from generation `raw`.
fn bound(descriptor: crate::index::IndexDescriptor, raw: u64) -> IndexDefinition {
    descriptor.bind(
        crate::index::IndexId::from_raw(raw),
        crate::index::GenerationId::from_raw(raw),
    )
}

fn btree(name: &str, property: &str, raw: u64) -> IndexDefinition {
    bound(
        crate::index::IndexDescriptor::btree(name, "User", property),
        raw,
    )
}

fn integrity(index: &IndexDefinition, state: Integrity) -> IndexIntegrityRecord {
    let mut record = IndexIntegrityRecord::new(index.id, index.generation);
    record.integrity = state;
    record
}

fn stray(node: u64) -> Mismatch {
    Mismatch::Extra {
        node,
        valid_from: None,
        tuple: vec![1],
    }
}

/// A disagreement found here makes the generation suspect at once and waits
/// to be recorded; nothing else is suspect.
#[test]
fn a_reported_generation_is_suspect_and_its_report_waits() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();
    let index = btree("user_email", "email", 1);
    let other = btree("user_name", "name", 2);
    assert!(!reg.is_suspect(index.generation));
    reg.report_mismatch(&engine, &index, stray(7));
    reg.report_mismatch(&engine, &index, stray(7));
    assert!(reg.is_suspect(index.generation));
    assert!(!reg.is_suspect(other.generation));
    let reports = reg.take_reports(core::time::Duration::ZERO);
    assert_eq!(reports.len(), 1, "one disagreement, reported twice");
    assert_eq!(reports[0].generation, index.generation);
    assert!(reg.take_reports(core::time::Duration::ZERO).is_empty());
    assert!(
        reg.is_suspect(index.generation),
        "taking the report clears nothing"
    );
}

/// A suspicion found here is about this member's copy: a verified record in
/// the catalog proves another member's copy, so neither an older one nor one
/// written after the catalog recorded the suspicion clears it. Only a check
/// verified on this member does.
#[test]
fn a_catalog_verification_does_not_clear_a_suspicion_found_here() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();
    let index = btree("user_email", "email", 1);
    reg.report_mismatch(&engine, &index, stray(7));
    let revision = reg.local_revision(index.generation);
    reg.apply_integrity(&[integrity(&index, Integrity::Verified)]);
    assert!(
        reg.is_suspect(index.generation),
        "the older verified record"
    );
    reg.apply_integrity(&[integrity(&index, Integrity::Suspect)]);
    assert!(reg.is_suspect(index.generation));
    reg.apply_integrity(&[integrity(&index, Integrity::Verified)]);
    assert!(
        reg.is_suspect(index.generation),
        "verified elsewhere after the catalog recorded it"
    );
    assert!(!reg.answers_at(index.generation, u64::MAX));
    assert!(
        reg.verified_here(&engine, index.generation, revision)
            .expect("lift")
    );
    assert!(!reg.is_suspect(index.generation), "verified on this member");
    assert!(
        LocalIndexStore::new(&engine)
            .list_unfit_here()
            .expect("marks")
            .is_empty(),
        "and its stored mark is gone"
    );
}

/// A check proves this member's copy only as it stood when its pass
/// started: a disagreement found here after that keeps the copy unfit,
/// in memory and in storage, whatever the pass then concluded.
#[test]
fn a_finding_after_the_pass_started_survives_its_verification() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();
    let index = btree("user_email", "email", 1);
    reg.report_mismatch(&engine, &index, stray(7));
    let started = reg.local_revision(index.generation);
    reg.report_mismatch(&engine, &index, stray(8));
    assert!(
        !reg.verified_here(&engine, index.generation, started)
            .expect("lift")
    );
    assert!(reg.is_suspect(index.generation));
    assert_eq!(
        LocalIndexStore::new(&engine)
            .list_unfit_here()
            .expect("marks"),
        vec![index.generation]
    );
    // A pass that started with no finding here at all does not lift a
    // finding that came after it either.
    let fresh = btree("user_name", "name", 2);
    let before = reg.local_revision(fresh.generation);
    reg.report_mismatch(&engine, &fresh, stray(9));
    assert!(
        !reg.verified_here(&engine, fresh.generation, before)
            .expect("lift")
    );
    assert!(reg.is_suspect(fresh.generation));
}

/// A finding is stored before the statement that made it goes on: a power
/// cut right after it, before any maintenance ran, still finds the copy
/// unfit when the member opens again.
#[test]
fn a_finding_survives_power_loss_before_any_maintenance() {
    let rig = coordinode_test_fixtures::PowerRig::new();
    let index = btree("user_email", "email", 1);
    {
        let engine = Arc::new(StorageEngine::open(&rig.config()).expect("open"));
        let reg = IndexRegistry::new();
        reg.report_mismatch(&engine, &index, stray(7));
        rig.cut(engine);
    }
    let engine = StorageEngine::open(&rig.config()).expect("reopen");
    let reg = IndexRegistry::new();
    reg.load_all(&engine).expect("load");
    assert!(reg.is_suspect(index.generation));
}

/// A mark whose flush failed is not taken as stored because its key is in
/// memory: the next round writes and flushes it again, and a power cut
/// after that finds it.
#[test]
fn a_mark_whose_flush_failed_is_stored_on_the_next_round() {
    let rig = coordinode_test_fixtures::PowerRig::new();
    let index = btree("user_email", "email", 1);
    {
        let engine = Arc::new(StorageEngine::open(&rig.config()).expect("open"));
        let reg = IndexRegistry::new();
        rig.fail_next(coordinode_test_fixtures::FaultOp::Open, 1);
        reg.report_mismatch(&engine, &index, stray(7));
        assert!(reg.is_suspect(index.generation), "unfit here regardless");
        reg.store_pending_marks(&engine);
        rig.cut(engine);
    }
    let engine = StorageEngine::open(&rig.config()).expect("reopen");
    let reg = IndexRegistry::new();
    reg.load_all(&engine).expect("load");
    assert!(reg.is_suspect(index.generation));
}

/// A generation that left the catalog keeps a mark found here: a reader
/// pinned to an older snapshot, plan or cursor may still reach its
/// entries, and nothing says when the last one is gone.
#[test]
fn a_replaced_generation_keeps_its_mark() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();
    let index = btree("user_email", "email", 1);
    reg.report_mismatch(&engine, &index, stray(7));
    reg.load_all(&engine).expect("load without the generation");
    assert!(reg.is_suspect(index.generation));
    assert!(!reg.answers_at(index.generation, u64::MAX));
    let restarted = IndexRegistry::new();
    restarted.load_all(&engine).expect("load after a restart");
    assert!(restarted.is_suspect(index.generation));
}

/// A generation the catalog records suspect is suspect on a member that
/// never saw the disagreement itself (a follower, a restarted process).
#[test]
fn a_recorded_suspicion_reaches_a_member_that_found_nothing() {
    let reg = IndexRegistry::new();
    let index = btree("user_email", "email", 1);
    reg.apply_integrity(&[integrity(&index, Integrity::Suspect)]);
    assert!(reg.is_suspect(index.generation));
    reg.apply_integrity(&[]);
    assert!(!reg.is_suspect(index.generation));
}

#[test]
fn register_and_lookup() {
    let reg = IndexRegistry::new();
    let index = bound(
        crate::index::IndexDescriptor::btree("user_email", "User", "email").unique(),
        1,
    );
    reg.register_in_memory(index.clone());

    assert_eq!(reg.len(), 1);
    assert_eq!(reg.get("user_email"), Some(index.clone()));
    assert_eq!(reg.get_by_id(index.id), Some(index));
    assert_eq!(reg.indexes_for_label("User").len(), 1);
    assert_eq!(reg.indexes_for_label("Movie").len(), 0);
}

/// A name names the index that holds it now: after a drop and a create of
/// the same name the name finds the new index, and the dropped identity
/// finds nothing, so work bound to it cannot reach its successor.
#[test]
fn a_name_follows_its_index_and_a_dropped_identity_finds_nothing() {
    let reg = IndexRegistry::new();
    let first = btree("user_email", "email", 1);
    reg.register_in_memory(first.clone());
    reg.unregister(first.id);
    assert_eq!(reg.get("user_email"), None);

    let second = btree("user_email", "email", 2);
    reg.register_in_memory(second.clone());
    assert_eq!(reg.get("user_email"), Some(second.clone()));
    assert_eq!(reg.get_by_id(first.id), None);

    // Re-registering an index under another name moves its binding.
    let mut renamed = second.clone();
    renamed.name = Some("by_email".into());
    reg.register_in_memory(renamed.clone());
    assert_eq!(reg.get("user_email"), None);
    assert_eq!(reg.get("by_email"), Some(renamed));
    assert_eq!(reg.len(), 1);
}

#[test]
fn indexes_for_property() {
    let reg = IndexRegistry::new();
    reg.register_in_memory(btree("user_email", "email", 1));
    reg.register_in_memory(btree("user_name", "name", 2));

    let email_idxs = reg.indexes_for_property("User", "email");
    assert_eq!(email_idxs.len(), 1);
    assert_eq!(email_idxs[0].name.as_deref(), Some("user_email"));
}

#[test]
fn distinct_unique_values_are_accepted() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();
    reg.register_in_memory(bound(
        crate::index::IndexDescriptor::btree("user_email", "User", "email").unique(),
        1,
    ));

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
    reg.register_in_memory(bound(
        crate::index::IndexDescriptor::btree("user_email", "User", "email").unique(),
        1,
    ));

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
    let index = bound(
        crate::index::IndexDescriptor::btree("user_email", "User", "email").unique(),
        1,
    );
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
    reg.register_in_memory(bound(
        crate::index::IndexDescriptor::btree("user_email", "User", "email").unique(),
        1,
    ));

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
    let index = bound(
        crate::index::IndexDescriptor::compound(
            "user_city_age",
            "User",
            vec!["city".into(), "age".into()],
        ),
        1,
    );
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
    let index = btree("user_email", "email", 1);
    reg.register_in_memory(index.clone());

    create(&reg, &engine, 1, &[("email", s("alice@test.com"))]).expect("create");
    let props = props(&[("email", s("alice@test.com"))]);
    reg.on_node_deleted(
        &engine,
        &mut txn(&engine),
        &NodeState {
            node_id: NodeId::from_raw(1),
            valid_from: None,
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
    let vector = bound(
        crate::index::IndexDescriptor::hnsw(
            "user_vec",
            "User",
            "email",
            crate::index::definition::VectorIndexConfig::default(),
        ),
        1,
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
    let store = LocalIndexStore::new(&engine);
    let first = store
        .publish_definition_txn(
            &mut txn(&engine),
            crate::index::IndexDescriptor::btree("idx1", "User", "email").unique(),
        )
        .expect("publish");
    let second = store
        .publish_definition_txn(
            &mut txn(&engine),
            crate::index::IndexDescriptor::btree("idx2", "User", "name"),
        )
        .expect("publish");

    let reg = IndexRegistry::new();
    reg.load_all(&engine).expect("load");
    assert_eq!(reg.len(), 2);
    assert_eq!(reg.get("idx1"), Some(first));
    assert_eq!(reg.get("idx2"), Some(second));
}

#[test]
fn sparse_index_skips_null_on_create() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let reg = IndexRegistry::new();
    let index = bound(
        crate::index::IndexDescriptor::btree("user_bio", "User", "bio").sparse(),
        1,
    );
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
    let index = bound(
        crate::index::IndexDescriptor::btree("active_email", "User", "email").with_filter(
            super::super::definition::PartialFilter::PropertyEquals {
                property: "status".into(),
                value: "active".into(),
            },
        ),
        1,
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
    let index = bound(
        crate::index::IndexDescriptor::btree("active_email", "User", "email").with_filter(
            super::super::definition::PartialFilter::PropertyEquals {
                property: "status".into(),
                value: "active".into(),
            },
        ),
        1,
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
    let index = bound(
        crate::index::IndexDescriptor::btree("verified_email", "User", "email").with_filter(
            super::super::definition::PartialFilter::PropertyEqualsBool {
                property: "verified".into(),
                value: true,
            },
        ),
        1,
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
    let index = bound(
        crate::index::IndexDescriptor::btree("user_bio_idx", "User", "bio").with_filter(
            super::super::definition::PartialFilter::PropertyExists {
                property: "bio".into(),
            },
        ),
        1,
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
