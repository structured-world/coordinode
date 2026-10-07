use super::*;

#[test]
fn property_def_builder() {
    let prop = PropertyDef::new("email", PropertyType::String)
        .not_null()
        .with_default(Value::String("unknown@example.com".into()));

    assert_eq!(prop.name, "email");
    assert!(matches!(prop.property_type, PropertyType::String));
    assert!(prop.not_null);
    assert!(prop.default.is_some());
}

/// A type constraint that names a type other than the declared one could
/// only ever hold for null, so the definition refuses it.
#[test]
fn constraint_conflict_refuses_a_type_other_than_the_declared_one() {
    let mut schema = LabelSchema::new_node_id("User");
    schema.add_property(PropertyDef::new("age", PropertyType::Int));
    let conflict = schema
        .constraint_conflict(&constraint(
            "t",
            &["age"],
            ConstraintKind::Type(PropertyType::String),
        ))
        .expect("a STRING constraint on an INT property is refused");
    assert!(conflict.contains("declared INT"), "{conflict}");
    assert_eq!(
        schema.constraint_conflict(&constraint(
            "t",
            &["age"],
            ConstraintKind::Type(PropertyType::Int)
        )),
        None
    );
}

/// A presence constraint on a property a STRICT label never stores could
/// never hold for any node; FLEXIBLE stores it, so it may.
#[test]
fn constraint_conflict_refuses_presence_of_an_undeclared_strict_property() {
    let mut strict = LabelSchema::new_node_id("User");
    strict.set_mode(SchemaMode::Strict);
    assert!(
        strict
            .constraint_conflict(&constraint("n", &["email"], ConstraintKind::NotNull))
            .is_some()
    );
    // Uniqueness of a property that is never stored is vacuous, not broken.
    assert_eq!(
        strict.constraint_conflict(&constraint("u", &["email"], ConstraintKind::Unique)),
        None
    );
    let mut flexible = LabelSchema::new_node_id("User");
    flexible.set_mode(SchemaMode::Flexible);
    assert_eq!(
        flexible.constraint_conflict(&constraint("n", &["email"], ConstraintKind::NotNull)),
        None
    );
}

/// A computed property is evaluated, never stored, so nothing can be
/// required of its stored value.
#[test]
fn constraint_conflict_refuses_a_computed_property() {
    let mut schema = LabelSchema::new_node_id("Doc");
    schema.add_property(PropertyDef::computed(
        "_ttl",
        crate::schema::computed::ComputedSpec::Ttl {
            duration_secs: 60,
            anchor_field: "created".into(),
            scope: crate::schema::computed::TtlScope::Node,
            target_field: None,
        },
    ));
    assert!(
        schema
            .constraint_conflict(&constraint("u", &["_ttl"], ConstraintKind::Unique))
            .is_some()
    );
}

#[test]
fn property_type_display() {
    assert_eq!(PropertyType::String.to_string(), "STRING");
    assert_eq!(PropertyType::Int.to_string(), "INT");
    assert_eq!(
        PropertyType::Vector {
            dimensions: 384,
            metric: VectorMetric::Cosine
        }
        .to_string(),
        "VECTOR(384, Cosine)"
    );
    assert_eq!(
        PropertyType::Array(Box::new(PropertyType::String)).to_string(),
        "ARRAY<STRING>"
    );
}

#[test]
fn label_schema_create() {
    let schema = LabelSchema::new_node_id("User");
    assert_eq!(schema.name, "User");
    assert!(schema.properties.is_empty());
    // Default schema mode is Strict — see SchemaMode::default(). Legacy
    // code returned `false` for the now-removed `strict` field because it
    // defaulted independently of `mode`; with the field gone, `is_strict`
    // is derived from `mode` which defaults to Strict.
    assert!(schema.is_strict());
    assert_eq!(schema.schema_revision, 1);
    // CE default: NodeId placement with single PRIMARY shard key.
    assert!(matches!(schema.placement, PlacementPolicy::NodeId));
    assert_eq!(schema.shard_keys.len(), 1);
    assert_eq!(schema.shard_keys[0].state, ShardKeyState::Primary);
    assert_eq!(schema.shard_keys[0].kind, PlacementKind::NodeId);
}

#[test]
fn label_schema_add_properties() {
    let mut schema = LabelSchema::new_node_id("User");
    schema.add_property(PropertyDef::new("name", PropertyType::String).not_null());
    schema.add_property(PropertyDef::new("age", PropertyType::Int));

    assert_eq!(schema.properties.len(), 2);
    // Property additions mutate the current snapshot but
    // do not bump the schema revision: only `ALTER LABEL` operations
    // affecting placement/shard_keys do.
    assert_eq!(schema.schema_revision, 1);
    assert!(schema.get_property("name").is_some());
    assert!(schema.get_property("name").is_some_and(|p| p.not_null));
}

#[test]
fn label_schema_remove_property() {
    let mut schema = LabelSchema::new_node_id("User");
    schema.add_property(PropertyDef::new("name", PropertyType::String));
    schema.add_property(PropertyDef::new("age", PropertyType::Int));

    let removed = schema.remove_property("age");
    assert!(removed.is_some());
    assert_eq!(schema.properties.len(), 1);
    assert!(schema.get_property("age").is_none());

    // Removing non-existent doesn't increment version
    let v_before = schema.schema_revision;
    assert!(schema.remove_property("nonexistent").is_none());
    assert_eq!(schema.schema_revision, v_before);
}

#[test]
fn label_schema_msgpack_roundtrip() {
    let mut schema = LabelSchema::new_node_id("Movie");
    schema.add_property(PropertyDef::new("title", PropertyType::String).not_null());
    schema.add_property(PropertyDef::new(
        "embedding",
        PropertyType::Vector {
            dimensions: 384,
            metric: VectorMetric::Cosine,
        },
    ));
    schema.add_property(PropertyDef::new(
        "tags",
        PropertyType::Array(Box::new(PropertyType::String)),
    ));
    schema.set_mode(SchemaMode::Strict);

    let bytes = schema.to_msgpack().expect("serialize");
    let restored = LabelSchema::from_msgpack(&bytes).expect("deserialize");
    assert_eq!(schema, restored);
}

#[test]
fn edge_type_schema_create() {
    let schema = EdgeTypeSchema::new("FOLLOWS");
    assert_eq!(schema.name, "FOLLOWS");
    assert!(!schema.temporal);
    assert_eq!(schema.schema_revision, 1);
}

#[test]
fn edge_type_schema_temporal() {
    let mut schema = EdgeTypeSchema::new("WORKS_AT");
    schema.set_temporal(true);
    schema.add_property(PropertyDef::new("valid_from", PropertyType::Timestamp).not_null());
    schema.add_property(PropertyDef::new("valid_to", PropertyType::Timestamp));
    schema.add_property(PropertyDef::new("role", PropertyType::String));

    assert!(schema.temporal);
    assert_eq!(schema.properties.len(), 3);
}

/// Identity resolves once from the declared properties: a temporal type
/// naming no discriminator is start-identified, a named one identifies
/// instances whether the type is temporal or not, and a type naming none
/// that is not temporal is single-edge. The resolved discriminator survives
/// serialization.
#[test]
fn edge_identity_resolves_from_the_declaration() {
    let mut works_at = EdgeTypeSchema::new("WORKS_AT");
    works_at.set_temporal(true);
    works_at.resolve_identity(None).expect("shorthand");
    assert!(works_at.is_start_identified());
    assert_eq!(
        works_at.discriminator(),
        Some(&EdgeDiscriminator {
            column: VALID_FROM.into(),
            value_type: PropertyType::Timestamp,
        })
    );

    for temporal in [false, true] {
        let mut asserts = EdgeTypeSchema::new("ASSERTS");
        asserts.set_temporal(temporal);
        asserts.add_property(PropertyDef::new("key", PropertyType::Blob).not_null());
        asserts.resolve_identity(Some("key")).expect("explicit");
        assert_eq!(
            asserts
                .discriminator()
                .map(|d| (d.column.as_str(), &d.value_type)),
            Some(("key", &PropertyType::Blob))
        );
        assert!(!asserts.is_start_identified(), "temporal={temporal}");
        let restored =
            EdgeTypeSchema::from_msgpack(&asserts.to_msgpack().expect("encode")).expect("decode");
        assert_eq!(restored, asserts);
    }

    let mut likes = EdgeTypeSchema::new("LIKES");
    likes.resolve_identity(None).expect("single");
    assert_eq!(likes.discriminator(), None);
    assert!(!likes.is_start_identified());
}

/// A discriminator that cannot identify an instance is refused and leaves
/// the schema unresolved: undeclared, nullable, of a type with no single
/// comparable value, computed; and a temporal type's valid_from declared as
/// anything but a timestamp.
#[test]
fn edge_identity_refuses_what_cannot_identify_an_instance() {
    let refused = |schema: &mut EdgeTypeSchema, declared: Option<&str>, why: &str| {
        let error = schema.resolve_identity(declared).expect_err(why);
        assert!(error.contains(why), "{error}");
        assert_eq!(schema.discriminator(), None);
    };
    let mut schema = EdgeTypeSchema::new("E");
    schema.add_property(PropertyDef::new("nullable", PropertyType::String));
    schema.add_property(PropertyDef::new("map", PropertyType::Map).not_null());
    schema.add_property(
        PropertyDef::new("vector", PropertyType::Array(Box::new(PropertyType::Int))).not_null(),
    );
    refused(&mut schema, Some("missing"), "does not declare");
    refused(&mut schema, Some("nullable"), "NOT NULL");
    refused(&mut schema, Some("map"), "is MAP");
    refused(&mut schema, Some("vector"), "is ARRAY<INT>");
    // A computed value is evaluated when read, not stored: nothing to key.
    schema.add_property(
        PropertyDef::computed(
            "_ttl",
            crate::schema::computed::ComputedSpec::Ttl {
                duration_secs: 60,
                anchor_field: "created".into(),
                scope: crate::schema::computed::TtlScope::Node,
                target_field: None,
            },
        )
        .not_null(),
    );
    refused(&mut schema, Some("_ttl"), "is COMPUTED");

    let mut temporal = EdgeTypeSchema::new("T");
    temporal.set_temporal(true);
    temporal.add_property(PropertyDef::new(VALID_FROM, PropertyType::String));
    refused(&mut temporal, None, "TIMESTAMP");

    // An INT start is Unix microseconds, the shared unit: identified by it.
    let mut micros = EdgeTypeSchema::new("M");
    micros.set_temporal(true);
    micros.add_property(PropertyDef::new(VALID_FROM, PropertyType::Int));
    micros.resolve_identity(None).expect("int start");
    assert!(micros.is_start_identified());
}

#[test]
fn edge_type_schema_msgpack_roundtrip() {
    let mut schema = EdgeTypeSchema::new("KNOWS");
    schema.add_property(
        PropertyDef::new("since", PropertyType::Timestamp)
            .not_null()
            .with_default(Value::Timestamp(0)),
    );
    schema.add_property(
        PropertyDef::new("weight", PropertyType::Float).with_default(Value::Float(1.0)),
    );

    let bytes = schema.to_msgpack().expect("serialize");
    let restored = EdgeTypeSchema::from_msgpack(&bytes).expect("deserialize");
    assert_eq!(schema, restored);
}

#[test]
fn label_schema_key_encoding() {
    let key = encode_label_schema_key("User", 1);
    assert_eq!(&key, b"schema:label:User:1");
}

#[test]
fn label_schema_key_includes_version() {
    let v1 = encode_label_schema_key("User", 1);
    let v2 = encode_label_schema_key("User", 2);
    assert_ne!(v1, v2, "different versions must produce different keys");
    assert!(
        v1 < v2,
        "version ordering preserved by string-encoded suffix"
    );
}

#[test]
fn label_current_revision_pointer_key_encoding() {
    let key = encode_label_current_revision_key("User");
    assert_eq!(&key, b"schema:current_revision:label:User");
}

#[test]
fn edge_type_schema_key_encoding() {
    let key = encode_edge_type_schema_key("FOLLOWS", 1);
    assert_eq!(&key, b"schema:edge_type:FOLLOWS:1");
}

#[test]
fn edge_type_current_revision_pointer_key_encoding() {
    let key = encode_edge_type_current_revision_key("FOLLOWS");
    assert_eq!(&key, b"schema:current_revision:edge_type:FOLLOWS");
}

#[test]
fn migration_state_key_encoding() {
    let key = encode_migration_state_key("Order", 0x42);
    let mut expected = b"schema:migration_state:Order:".to_vec();
    expected.extend_from_slice(&0x42u64.to_be_bytes());
    assert_eq!(key, expected);
}

#[test]
fn migration_state_keys_sort_by_node_id_within_label() {
    let k_low = encode_migration_state_key("Order", 1);
    let k_high = encode_migration_state_key("Order", 1000);
    assert!(
        k_low < k_high,
        "BE node_id encoding preserves numeric ordering"
    );
}

#[test]
fn migration_state_keys_separated_by_label() {
    let k_order = encode_migration_state_key("Order", 1);
    let k_user = encode_migration_state_key("User", 1);
    // Lexicographic separation: "Order" < "User", so prefix scan by label
    // returns docs grouped per label.
    assert!(k_order < k_user);
}

#[test]
fn chunk_assignments_key_encoding() {
    let key = encode_chunk_assignments_key("Order");
    assert_eq!(&key, b"schema:chunks:Order");
}

#[test]
fn edge_type_schema_default_placement_is_colocate_with_source() {
    let schema = EdgeTypeSchema::new("WORKS_AT");
    assert_eq!(schema.placement, EdgePlacement::ColocateWithSource);
    assert!(!schema.temporal);
    assert_eq!(schema.schema_revision, 1);
}

#[test]
fn migration_state_entry_roundtrips() {
    let entry = MigrationStateEntry {
        current_shard: 1,
        target_shard: 5,
        state: MigrationDocState::Migrating,
        enqueued_at: 1_700_000_000_000,
    };
    let bytes = rmp_serde::to_vec(&entry).expect("encode");
    let decoded: MigrationStateEntry = rmp_serde::from_slice(&bytes).expect("decode");
    assert_eq!(decoded, entry);
}

#[test]
fn label_schema_with_hash_placement_msgpack_roundtrip() {
    // Hash placement carries a property name in the variant payload.
    // Ensure the variant + payload survive msgpack encoding without
    // collapsing to the default NodeId placement.
    let mut schema = LabelSchema::new("User", PlacementPolicy::Hash("tenant_id".to_string()));
    schema.add_property(PropertyDef::new("tenant_id", PropertyType::String).not_null());
    let bytes = schema.to_msgpack().expect("encode");
    let decoded = LabelSchema::from_msgpack(&bytes).expect("decode");
    assert!(matches!(
        decoded.placement,
        PlacementPolicy::Hash(ref p) if p == "tenant_id"
    ));
    assert_eq!(decoded.shard_keys.len(), 1);
    assert_eq!(decoded.shard_keys[0].property, "tenant_id");
    assert_eq!(decoded.shard_keys[0].kind, PlacementKind::Hash);
    assert_eq!(decoded.shard_keys[0].state, ShardKeyState::Primary);
}

#[test]
fn label_schema_with_range_placement_msgpack_roundtrip() {
    let mut schema = LabelSchema::new("Event", PlacementPolicy::Range("occurred_at".to_string()));
    schema.add_property(PropertyDef::new("occurred_at", PropertyType::Timestamp).not_null());
    let bytes = schema.to_msgpack().expect("encode");
    let decoded = LabelSchema::from_msgpack(&bytes).expect("decode");
    assert!(matches!(
        decoded.placement,
        PlacementPolicy::Range(ref p) if p == "occurred_at"
    ));
    assert_eq!(decoded.shard_keys[0].kind, PlacementKind::Range);
}

#[test]
fn edge_type_schema_with_target_colocation_msgpack_roundtrip() {
    let mut schema = EdgeTypeSchema::new("OWNED_BY");
    schema.placement = EdgePlacement::ColocateWithTarget;
    let bytes = schema.to_msgpack().expect("encode");
    let decoded = EdgeTypeSchema::from_msgpack(&bytes).expect("decode");
    assert_eq!(decoded.placement, EdgePlacement::ColocateWithTarget);
}

#[test]
fn edge_type_schema_with_replicated_placement_msgpack_roundtrip() {
    let mut schema = EdgeTypeSchema::new("MENTIONS");
    schema.placement = EdgePlacement::Replicated;
    let bytes = schema.to_msgpack().expect("encode");
    let decoded = EdgeTypeSchema::from_msgpack(&bytes).expect("decode");
    assert_eq!(decoded.placement, EdgePlacement::Replicated);
}

#[test]
fn migration_state_entry_legacy_and_migrated_states_roundtrip() {
    // The enum has three states; existing tests cover Migrating —
    // exercise the other two so all variants are wire-stable.
    for state in [MigrationDocState::Legacy, MigrationDocState::Migrated] {
        let entry = MigrationStateEntry {
            current_shard: 2,
            target_shard: 7,
            state,
            enqueued_at: 1_700_000_000_000,
        };
        let bytes = rmp_serde::to_vec(&entry).expect("encode");
        let decoded: MigrationStateEntry = rmp_serde::from_slice(&bytes).expect("decode");
        assert_eq!(decoded, entry);
    }
}

#[test]
fn schema_keys_sort_alphabetically() {
    let k1 = encode_label_schema_key("Actor", 1);
    let k2 = encode_label_schema_key("User", 1);
    assert!(k1 < k2);
}

#[test]
fn property_with_default_value() {
    let prop = PropertyDef::new("status", PropertyType::String)
        .with_default(Value::String("active".into()));
    assert_eq!(prop.default, Some(Value::String("active".into())));
}

#[test]
fn schema_version_stable_across_property_mutations() {
    // Schema revision is bumped only by `ALTER LABEL` operations
    // affecting placement/shard_keys. Property additions and removals
    // mutate the current snapshot in place without bumping it.
    let mut schema = LabelSchema::new_node_id("Test");
    assert_eq!(schema.schema_revision, 1);
    schema.add_property(PropertyDef::new("a", PropertyType::Int));
    assert_eq!(schema.schema_revision, 1);
    schema.add_property(PropertyDef::new("b", PropertyType::String));
    assert_eq!(schema.schema_revision, 1);
    schema.remove_property("a");
    assert_eq!(schema.schema_revision, 1);
}

#[test]
fn schema_mode_default_is_strict() {
    assert_eq!(SchemaMode::default(), SchemaMode::Strict);
    let schema = LabelSchema::new_node_id("Test");
    assert_eq!(schema.mode, SchemaMode::Strict);
}

#[test]
fn schema_mode_properties() {
    assert!(SchemaMode::Strict.rejects_unknown());
    assert!(SchemaMode::Strict.full_interning());
    assert!(SchemaMode::Strict.validates_declared());

    assert!(!SchemaMode::Validated.rejects_unknown());
    assert!(!SchemaMode::Validated.full_interning());
    assert!(SchemaMode::Validated.validates_declared());

    assert!(!SchemaMode::Flexible.rejects_unknown());
    assert!(!SchemaMode::Flexible.full_interning());
    assert!(!SchemaMode::Flexible.validates_declared());
}

#[test]
fn schema_mode_display() {
    assert_eq!(SchemaMode::Strict.to_string(), "STRICT");
    assert_eq!(SchemaMode::Validated.to_string(), "VALIDATED");
    assert_eq!(SchemaMode::Flexible.to_string(), "FLEXIBLE");
}

#[test]
fn set_mode() {
    let mut schema = LabelSchema::new_node_id("Test");
    assert_eq!(schema.mode, SchemaMode::Strict);

    schema.set_mode(SchemaMode::Validated);
    assert_eq!(schema.mode, SchemaMode::Validated);
    assert!(!schema.is_strict());

    schema.set_mode(SchemaMode::Strict);
    assert_eq!(schema.mode, SchemaMode::Strict);
    assert!(schema.is_strict());
}

#[test]
fn schema_mode_msgpack_roundtrip() {
    let mut schema = LabelSchema::new_node_id("Flexible");
    schema.set_mode(SchemaMode::Flexible);
    schema.add_property(PropertyDef::new("name", PropertyType::String));

    let bytes = schema.to_msgpack().expect("serialize");
    let restored = LabelSchema::from_msgpack(&bytes).expect("deserialize");
    assert_eq!(restored.mode, SchemaMode::Flexible);
    assert_eq!(restored.name, "Flexible");
}

// ── COMPUTED properties ───────────────────────────────

#[test]
fn computed_property_def() {
    use crate::schema::computed::{ComputedSpec, DecayFormula};

    let prop = PropertyDef::computed(
        "relevance",
        ComputedSpec::Decay {
            formula: DecayFormula::Linear,
            initial: 1.0,
            target: 0.0,
            duration_secs: 604800,
            anchor_field: "created_at".into(),
        },
    );
    assert!(prop.is_computed());
    assert!(!prop.not_null);
    assert!(prop.default.is_none());
}

#[test]
fn computed_property_display() {
    use crate::schema::computed::{ComputedSpec, DecayFormula};

    let pt = PropertyType::Computed(ComputedSpec::Decay {
        formula: DecayFormula::Exponential { lambda: 0.693 },
        initial: 1.0,
        target: 0.0,
        duration_secs: 86400,
        anchor_field: "created_at".into(),
    });
    let s = format!("{pt}");
    assert!(s.starts_with("COMPUTED("));
}

#[test]
fn schema_with_computed_msgpack_roundtrip() {
    use crate::schema::computed::{ComputedSpec, DecayFormula, TtlScope};

    let mut schema = LabelSchema::new_node_id("Memory");
    schema.add_property(PropertyDef::new("content", PropertyType::String));
    schema.add_property(PropertyDef::new("created_at", PropertyType::Timestamp));
    schema.add_property(PropertyDef::computed(
        "relevance",
        ComputedSpec::Decay {
            formula: DecayFormula::Linear,
            initial: 1.0,
            target: 0.0,
            duration_secs: 604800,
            anchor_field: "created_at".into(),
        },
    ));
    schema.add_property(PropertyDef::computed(
        "_ttl",
        ComputedSpec::Ttl {
            duration_secs: 2592000,
            anchor_field: "created_at".into(),
            scope: TtlScope::Node,
            target_field: None,
        },
    ));

    let bytes = schema.to_msgpack().expect("serialize");
    let restored = LabelSchema::from_msgpack(&bytes).expect("deserialize");

    assert_eq!(restored.name, "Memory");
    assert_eq!(restored.properties.len(), 4);

    let rel = restored.get_property("relevance").expect("relevance prop");
    assert!(rel.is_computed());

    let ttl = restored.get_property("_ttl").expect("_ttl prop");
    assert!(ttl.is_computed());
}

#[test]
fn table_schema_round_trip_with_primary_key_and_layout() {
    let mut schema = LabelSchema::new("Trade", PlacementPolicy::NodeId);
    schema.add_property(PropertyDef::new("trade_id", PropertyType::Int).not_null());
    schema.add_property(PropertyDef::new("symbol", PropertyType::String).not_null());
    schema.make_table(vec!["trade_id".into()]);
    schema.set_storage_layout(StorageLayout::Columnar);

    assert!(schema.is_table());
    assert!(schema.is_columnar());

    let bytes = schema.to_msgpack().expect("serialize");
    let restored = LabelSchema::from_msgpack(&bytes).expect("deserialize");

    let key = ["trade_id".to_string()];
    assert_eq!(restored.table_key(), Some(TableKey::Columns(&key)));
    assert_eq!(restored.storage_layout, StorageLayout::Columnar);
    assert!(restored.is_columnar());
}

/// A table without declared key columns is keyed by row id, and stays a
/// table through a round trip: an empty key list alone would read back as a
/// plain graph label.
#[test]
fn a_table_keyed_by_row_id_round_trips() {
    let mut schema = LabelSchema::new("Event", PlacementPolicy::NodeId);
    schema.make_table(Vec::new());
    assert_eq!(schema.table_key(), Some(TableKey::RowId));

    let bytes = schema.to_msgpack().expect("serialize");
    let restored = LabelSchema::from_msgpack(&bytes).expect("deserialize");
    assert_eq!(restored.table_key(), Some(TableKey::RowId));
    assert!(restored.key_columns().is_empty());
}

#[test]
fn plain_label_is_not_a_table_and_defaults_to_row() {
    let schema = LabelSchema::new("User", PlacementPolicy::NodeId);
    assert!(!schema.is_table());
    assert!(!schema.is_columnar());
    assert_eq!(schema.storage_layout, StorageLayout::Row);
    assert_eq!(schema.table_key(), None);
}

fn constraint(name: &str, properties: &[&str], kind: ConstraintKind) -> NodeConstraint {
    NodeConstraint {
        name: name.to_string(),
        properties: properties.iter().map(|p| p.to_string()).collect(),
        kind,
        state: ConstraintState::Active,
        scope: None,
    }
}

/// Constraints are part of the stored schema: every kind survives a round
/// trip in order, and the accessors find and remove them by name.
#[test]
fn constraints_round_trip_and_are_found_by_name() {
    let mut schema = LabelSchema::new_node_id("User");
    schema.add_constraint(constraint("u_email", &["email"], ConstraintKind::Unique));
    schema.add_constraint(constraint("u_name", &["name"], ConstraintKind::NotNull));
    schema.add_constraint(constraint(
        "u_key",
        &["first", "last"],
        ConstraintKind::NodeKey,
    ));
    schema.add_constraint(constraint(
        "u_age",
        &["age"],
        ConstraintKind::Type(PropertyType::Int),
    ));

    let bytes = schema.to_msgpack().expect("serialize");
    let mut restored = LabelSchema::from_msgpack(&bytes).expect("deserialize");
    assert_eq!(restored, schema);
    let names: Vec<&str> = restored
        .constraints()
        .iter()
        .map(|c| c.name.as_str())
        .collect();
    assert_eq!(names, ["u_email", "u_name", "u_key", "u_age"]);
    assert_eq!(
        restored.constraint("u_key").map(|c| &c.kind),
        Some(&ConstraintKind::NodeKey)
    );

    let removed = restored.remove_constraint("u_name").expect("present");
    assert_eq!(removed.kind, ConstraintKind::NotNull);
    assert!(restored.constraint("u_name").is_none());
    assert!(restored.remove_constraint("u_name").is_none());

    // A state change survives the round trip too: a validating constraint
    // never reads back as active.
    restored.constraint_mut("u_key").expect("present").state = ConstraintState::Validating;
    let bytes = restored.to_msgpack().expect("serialize");
    let again = LabelSchema::from_msgpack(&bytes).expect("deserialize");
    assert_eq!(
        again.constraint("u_key").map(|c| c.state),
        Some(ConstraintState::Validating)
    );
}

/// A schema stored before labels carried constraints decodes with none:
/// the field is the last one of the record and defaults when absent.
#[test]
fn a_schema_record_without_constraints_decodes_with_none() {
    let mut schema = LabelSchema::new_node_id("User");
    schema.add_property(PropertyDef::new("name", PropertyType::String).not_null());
    schema.add_constraint(constraint("u_name", &["name"], ConstraintKind::NotNull));
    let bytes = schema.to_msgpack().expect("serialize");

    let mut fields = rmpv::decode::read_value(&mut bytes.as_slice())
        .expect("decode as a value")
        .as_array()
        .cloned()
        .expect("a label schema is stored as an array of its fields");
    fields.pop();
    let mut older = Vec::new();
    rmpv::encode::write_value(&mut older, &rmpv::Value::Array(fields)).expect("re-encode");

    let restored = LabelSchema::from_msgpack(&older).expect("decode the older record");
    assert!(restored.constraints().is_empty());
    assert_eq!(restored.properties, schema.properties);
}

/// The kinds tell apart what each one checks: presence, per-node checking
/// and index ownership.
#[test]
fn constraint_kinds_say_what_they_check() {
    let unique = constraint("a", &["x"], ConstraintKind::Unique);
    let not_null = constraint("b", &["x"], ConstraintKind::NotNull);
    let key = constraint("c", &["x", "y"], ConstraintKind::NodeKey);
    let typed = constraint("d", &["x"], ConstraintKind::Type(PropertyType::String));

    assert!(unique.owns_index() && !unique.checks_each_node() && !unique.requires_presence());
    assert!(!not_null.owns_index() && not_null.checks_each_node() && not_null.requires_presence());
    assert!(key.owns_index() && key.checks_each_node() && key.requires_presence());
    assert!(!typed.owns_index() && typed.checks_each_node() && !typed.requires_presence());

    assert!(not_null.same_requirement(&constraint("other", &["x"], ConstraintKind::NotNull)));
    assert!(!not_null.same_requirement(&constraint("b", &["y"], ConstraintKind::NotNull)));
    assert!(!key.same_requirement(&constraint("c", &["y", "x"], ConstraintKind::NodeKey)));
    assert_eq!(typed.kind.to_string(), "TYPE STRING");
}

#[test]
fn constraint_name_key_encoding() {
    assert_eq!(
        encode_constraint_name_key("user_email"),
        b"schema:constraint:user_email".to_vec()
    );
}

fn owns(
    direction: crate::graph::cardinality::Direction,
    measure: crate::graph::cardinality::CardinalityMeasure,
) -> crate::graph::cardinality::CardinalityDescriptor {
    crate::graph::cardinality::CardinalityDescriptor {
        edge_type: "OWNS".into(),
        direction,
        measure,
        bound: crate::graph::cardinality::CardinalityBound::AtMostOne,
        schema_generation: 1,
    }
}

/// A type declares at most one constraint per direction and measure, both
/// measures may constrain one side, and the declaration survives the
/// definition's encoding into the profile a commit reads.
#[test]
fn cardinality_is_declared_per_direction_and_measure() {
    use crate::graph::cardinality::{CardinalityMeasure as M, Direction as D};

    let mut schema = EdgeTypeSchema::new("OWNS");
    assert_eq!(schema.cardinality_profile(), None, "nothing declared");
    schema
        .declare_cardinality(owns(D::Outgoing, M::EdgeInstances))
        .expect("instances");
    schema
        .declare_cardinality(owns(D::Outgoing, M::DistinctNeighbours))
        .expect("neighbours, on the same side");
    assert!(
        schema
            .declare_cardinality(owns(D::Outgoing, M::EdgeInstances))
            .is_err(),
        "a second constraint over one side and measure"
    );
    let mut other = owns(D::Incoming, M::EdgeInstances);
    other.edge_type = "KNOWS".into();
    assert!(schema.declare_cardinality(other).is_err(), "another type");

    let restored =
        EdgeTypeSchema::from_msgpack(&schema.to_msgpack().expect("encode")).expect("decode");
    assert_eq!(restored.cardinality(), schema.cardinality());
    let profile = restored.cardinality_profile().expect("declared");
    assert!(!profile.discriminated);
    assert_eq!(profile.descriptors.len(), 2);

    assert_eq!(
        schema.withdraw_cardinality(D::Outgoing, M::EdgeInstances),
        Some(owns(D::Outgoing, M::EdgeInstances))
    );
    assert_eq!(
        schema.withdraw_cardinality(D::Outgoing, M::EdgeInstances),
        None
    );
    assert_eq!(schema.cardinality().len(), 1);
}

/// A temporal type's bound holds at every valid-time instant, which a count
/// of its current edges does not decide, so it is refused here.
#[test]
fn a_temporal_type_declares_no_counted_cardinality() {
    use crate::graph::cardinality::{CardinalityMeasure as M, Direction as D};

    let mut schema = EdgeTypeSchema::new("OWNS");
    schema.set_temporal(true);
    schema.resolve_identity(None).expect("start-identified");
    assert!(
        schema
            .declare_cardinality(owns(D::Outgoing, M::EdgeInstances))
            .is_err()
    );
}

/// The identity shape follows the resolved definition: no definition and an
/// undiscriminated one are single-edge, a discriminator makes instances, a
/// temporal type is temporal whatever identifies it.
#[test]
fn the_identity_shape_follows_the_definition() {
    use crate::graph::cardinality::IdentityShape;

    assert_eq!(IdentityShape::of(None), IdentityShape::Single);
    let mut single = EdgeTypeSchema::new("OWNS");
    single.resolve_identity(None).expect("single");
    assert_eq!(IdentityShape::of(Some(&single)), IdentityShape::Single);

    let mut discriminated = EdgeTypeSchema::new("KNOWS");
    discriminated.add_property(PropertyDef::new("context", PropertyType::String).not_null());
    discriminated
        .resolve_identity(Some("context"))
        .expect("discriminated");
    assert_eq!(
        IdentityShape::of(Some(&discriminated)),
        IdentityShape::Discriminated
    );
    assert!(discriminated.cardinality_profile().is_none());

    let mut temporal = EdgeTypeSchema::new("WORKS_AT");
    temporal.set_temporal(true);
    temporal.resolve_identity(None).expect("temporal");
    assert_eq!(IdentityShape::of(Some(&temporal)), IdentityShape::Temporal);
}
