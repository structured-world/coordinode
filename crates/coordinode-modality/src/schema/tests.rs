use super::*;
use coordinode_core::schema::definition::{PlacementPolicy, PropertyDef, PropertyType};

/// Logic-test fixture (memory backing, env-flippable via
/// `COORDINODE_TEST_BACKEND`). Schema CRUD has no persistence
/// semantics.
fn open_engine() -> coordinode_test_fixtures::EngineFixture {
    coordinode_test_fixtures::engine_for_logic()
}

fn sample_label() -> LabelSchema {
    let mut schema = LabelSchema::new("User", PlacementPolicy::NodeId);
    schema.add_property(PropertyDef::new("email", PropertyType::String).not_null());
    schema
}

fn sample_edge_type() -> EdgeTypeSchema {
    EdgeTypeSchema::new("KNOWS")
}

#[test]
fn round_trip_label_schema() {
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalSchemaStore::new(engine);
    let schema = sample_label();

    assert!(
        store.load_label("User").expect("load none").is_none(),
        "label must not exist before save"
    );

    store.save_label(&schema).expect("save");
    let loaded = store
        .load_label("User")
        .expect("load some")
        .expect("Some(schema)");
    assert_eq!(loaded.name, schema.name);
    assert_eq!(loaded.schema_revision, schema.schema_revision);
    assert_eq!(loaded.properties.len(), schema.properties.len());
}

#[test]
fn save_label_is_atomic_pointer_and_body() {
    // The save path writes body + pointer in one WriteBatch — a
    // reader can never observe a pointer naming a missing
    // revision. Smoke check: after save, both keys exist; after a
    // second save with a bumped revision, pointer points to the
    // newer revision AND both bodies are still readable (revision
    // history is preserved).
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalSchemaStore::new(engine);

    let mut v1 = sample_label();
    v1.schema_revision = 1;
    store.save_label(&v1).expect("save v1");

    let mut v2 = sample_label();
    v2.schema_revision = 2;
    store.save_label(&v2).expect("save v2");

    // Current load returns v2.
    let cur = store
        .load_label("User")
        .expect("load")
        .expect("Some(schema)");
    assert_eq!(cur.schema_revision, 2);

    // v1 body still readable through its revisioned key.
    let v1_bytes = engine
        .get(Partition::Schema, &encode_label_schema_key("User", 1))
        .expect("get")
        .expect("v1 body");
    let v1_loaded = LabelSchema::from_msgpack(&v1_bytes).expect("decode v1");
    assert_eq!(v1_loaded.schema_revision, 1);
}

#[test]
fn corrupt_pointer_surfaces_as_decode_error() {
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalSchemaStore::new(engine);
    // Inject a corrupt pointer (3 bytes instead of 8).
    engine
        .put(
            Partition::Schema,
            &encode_label_current_revision_key("Corrupt"),
            &[0xff, 0xff, 0xff],
        )
        .expect("inject");
    let err = store.load_label("Corrupt").expect_err("must error");
    assert!(matches!(err, StoreError::Decode { .. }));
}

#[test]
fn pointer_naming_missing_revision_surfaces_as_decode_error() {
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalSchemaStore::new(engine);
    // Pointer says revision 7 exists, but no body at rev 7.
    engine
        .put(
            Partition::Schema,
            &encode_label_current_revision_key("Orphan"),
            &7u64.to_be_bytes(),
        )
        .expect("inject");
    let err = store.load_label("Orphan").expect_err("must error");
    assert!(matches!(
        err,
        StoreError::Decode {
            kind: "label schema",
            ..
        }
    ));
}

#[test]
fn edge_type_round_trip() {
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalSchemaStore::new(engine);
    let schema = EdgeTypeSchema::new("KNOWS");

    assert!(store.load_edge_type("KNOWS").expect("none").is_none());
    store.save_edge_type(&schema).expect("save");
    let loaded = store.load_edge_type("KNOWS").expect("some").expect("Some");
    assert_eq!(loaded.name, "KNOWS");
}

#[test]
fn edge_type_revision_bump_preserves_history() {
    // Symmetric to the label revision-bump test: save v1, save
    // v2, current load returns v2, v1 body still readable via
    // its revisioned key.
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalSchemaStore::new(engine);

    let mut v1 = EdgeTypeSchema::new("KNOWS");
    v1.schema_revision = 1;
    store.save_edge_type(&v1).expect("save v1");

    let mut v2 = EdgeTypeSchema::new("KNOWS");
    v2.schema_revision = 2;
    store.save_edge_type(&v2).expect("save v2");

    let cur = store
        .load_edge_type("KNOWS")
        .expect("ok")
        .expect("Some(schema)");
    assert_eq!(cur.schema_revision, 2);

    let v1_bytes = engine
        .get(Partition::Schema, &encode_edge_type_schema_key("KNOWS", 1))
        .expect("ok")
        .expect("v1 body");
    let v1_loaded = EdgeTypeSchema::from_msgpack(&v1_bytes).expect("decode v1");
    assert_eq!(v1_loaded.schema_revision, 1);
}

#[test]
fn legacy_zero_length_edge_marker_loads_as_none() {
    // Pre-DDL deployments wrote a zero-length value at the
    // revisioned key to mark "edge type exists, no schema body".
    // The store must surface this as `None`, not a decode error.
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalSchemaStore::new(engine);
    engine
        .put(
            Partition::Schema,
            &encode_edge_type_current_revision_key("LEGACY"),
            &1u64.to_be_bytes(),
        )
        .expect("pointer");
    engine
        .put(
            Partition::Schema,
            &encode_edge_type_schema_key("LEGACY", 1),
            b"",
        )
        .expect("empty marker");
    assert!(
        store.load_edge_type("LEGACY").expect("ok").is_none(),
        "legacy zero-length marker must decode as None",
    );
}

// ── list_labels / list_edge_types ─────────────────────────────

#[test]
fn list_labels_returns_every_declared_label_at_current_revision() {
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalSchemaStore::new(engine);
    let mut a = sample_label();
    a.name = "User".into();
    a.schema_revision = 1;
    let mut b = sample_label();
    b.name = "Order".into();
    b.schema_revision = 1;
    let mut c = sample_label();
    c.name = "User".into();
    c.schema_revision = 2;
    store.save_label(&a).expect("save User rev 1");
    store.save_label(&b).expect("save Order rev 1");
    store.save_label(&c).expect("save User rev 2");

    let mut listed = store.list_labels().expect("list");
    listed.sort_by(|l, r| l.name.cmp(&r.name));
    assert_eq!(listed.len(), 2, "one schema per declared label");
    assert_eq!(listed[0].name, "Order");
    assert_eq!(listed[0].schema_revision, 1);
    assert_eq!(listed[1].name, "User");
    assert_eq!(listed[1].schema_revision, 2, "current rev for User");
}

#[test]
fn list_labels_empty_when_no_schemas_declared() {
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalSchemaStore::new(engine);
    let listed = store.list_labels().expect("list");
    assert!(listed.is_empty());
}

#[test]
fn list_edge_types_returns_every_declared_edge_type_at_current_revision() {
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalSchemaStore::new(engine);
    let mut knows_v1 = sample_edge_type();
    knows_v1.name = "KNOWS".into();
    knows_v1.schema_revision = 1;
    let mut owns = sample_edge_type();
    owns.name = "OWNS".into();
    owns.schema_revision = 1;
    let mut knows_v2 = sample_edge_type();
    knows_v2.name = "KNOWS".into();
    knows_v2.schema_revision = 2;
    store.save_edge_type(&knows_v1).expect("save KNOWS rev 1");
    store.save_edge_type(&owns).expect("save OWNS rev 1");
    store.save_edge_type(&knows_v2).expect("save KNOWS rev 2");

    let mut listed = store.list_edge_types().expect("list");
    listed.sort_by(|l, r| l.name.cmp(&r.name));
    assert_eq!(listed.len(), 2, "one schema per declared edge type");
    assert_eq!(listed[0].name, "KNOWS");
    assert_eq!(listed[0].schema_revision, 2, "current rev for KNOWS");
    assert_eq!(listed[1].name, "OWNS");
}

#[test]
fn list_edge_types_skips_legacy_zero_length_markers() {
    // Pre-DDL idempotent existence markers shouldn't surface in
    // the listing — `load_edge_type` returns None for them, so
    // `list_edge_types` filters them out.
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalSchemaStore::new(engine);
    // Real schema for KNOWS.
    let mut real = sample_edge_type();
    real.name = "KNOWS".into();
    real.schema_revision = 1;
    store.save_edge_type(&real).expect("save");
    // Legacy marker for LEGACY edge type — bare pointer + empty body.
    engine
        .put(
            Partition::Schema,
            &encode_edge_type_current_revision_key("LEGACY"),
            &1u64.to_be_bytes(),
        )
        .expect("legacy pointer");
    engine
        .put(
            Partition::Schema,
            &encode_edge_type_schema_key("LEGACY", 1),
            b"",
        )
        .expect("legacy empty body");

    let listed = store.list_edge_types().expect("list");
    assert_eq!(
        listed.len(),
        1,
        "legacy marker must be skipped, only KNOWS surfaces",
    );
    assert_eq!(listed[0].name, "KNOWS");
}

#[test]
fn list_edge_type_names_engine_includes_markers_and_dedups_versions() {
    // The engine name lister parses names straight from
    // `schema:edge_type:<name>:<version>` keys, so unlike
    // `list_edge_types` it INCLUDES zero-length existence markers (a
    // reaper must clean adjacency for every registered type) and dedups
    // multiple versions of the same name to one entry.
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalSchemaStore::new(engine);

    let mut knows_v1 = sample_edge_type();
    knows_v1.name = "KNOWS".into();
    knows_v1.schema_revision = 1;
    store.save_edge_type(&knows_v1).expect("save v1");
    let mut knows_v2 = sample_edge_type();
    knows_v2.name = "KNOWS".into();
    knows_v2.schema_revision = 2;
    store.save_edge_type(&knows_v2).expect("save v2");
    // Legacy zero-length marker — present in keys, absent from decoded list.
    engine
        .put(
            Partition::Schema,
            &encode_edge_type_schema_key("LEGACY", 1),
            b"",
        )
        .expect("legacy marker");

    let mut names = store.list_edge_type_names_engine().expect("names");
    names.sort();
    assert_eq!(
        names,
        vec!["KNOWS".to_string(), "LEGACY".to_string()],
        "marker included, versions deduped to one KNOWS entry",
    );
}
