use super::*;

#[test]
fn property_def_builder() {
    let prop = PropertyDef::new("email", PropertyType::String)
        .not_null()
        .unique()
        .with_default(Value::String("unknown@example.com".into()));

    assert_eq!(prop.name, "email");
    assert!(matches!(prop.property_type, PropertyType::String));
    assert!(prop.not_null);
    assert!(prop.unique);
    assert!(prop.default.is_some());
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
    // Per ADR-023, property additions mutate the current snapshot but
    // do not bump the schema revision — only `ALTER LABEL` operations
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
fn ce_single_shard_chunk_table_roundtrip() {
    let table = ChunkAssignmentTable::ce_single_shard("Order");
    let bytes = table.to_msgpack().expect("encode");
    let decoded = ChunkAssignmentTable::from_msgpack(&bytes).expect("decode");
    assert_eq!(decoded.label, "Order");
    assert_eq!(decoded.ranges, vec![(0, 1)]);
    assert_eq!(decoded.revision, 1);
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
fn chunk_assignment_table_multi_range_roundtrip() {
    // Real EE deployments will carry multiple range/shard pairs —
    // verify multi-entry ranges survive msgpack.
    let table = ChunkAssignmentTable {
        label: "Event".to_string(),
        ranges: vec![(0, 3), (1_000_000, 5), (5_000_000, 8)],
        revision: 12,
    };
    let bytes = table.to_msgpack().expect("encode");
    let decoded = ChunkAssignmentTable::from_msgpack(&bytes).expect("decode");
    assert_eq!(decoded.label, "Event");
    assert_eq!(decoded.ranges, vec![(0, 3), (1_000_000, 5), (5_000_000, 8)]);
    assert_eq!(decoded.revision, 12);
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
    // Per ADR-023, schema revision is bumped only by `ALTER LABEL`
    // operations affecting placement/shard_keys. Property additions and
    // removals mutate the current snapshot in place without bumping
    // version. ALTER LABEL semantics (R210c) bump version explicitly when
    // they ship.
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

// ── COMPUTED properties (R081) ───────────────────────────────

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
    assert!(!prop.unique);
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
