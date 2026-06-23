use super::*;
use crate::graph::types::VectorMetric;

fn base_schema() -> LabelSchema {
    let mut schema = LabelSchema::new_node_id("User");
    schema.add_property(PropertyDef::new("name", PropertyType::String).not_null());
    schema.add_property(PropertyDef::new("age", PropertyType::Int));
    schema
}

#[test]
fn add_property_simple() {
    let mut schema = base_schema();
    let v_before = schema.schema_revision;

    let result = apply_change(
        &mut schema,
        SchemaChange::AddProperty(PropertyDef::new("bio", PropertyType::String)),
    );
    assert!(result.is_ok());
    assert!(schema.get_property("bio").is_some());
    // Per ADR-023, property additions do not bump the schema revision —
    // version tracks placement/shard_keys changes via `ALTER LABEL`.
    assert_eq!(schema.schema_revision, v_before);
}

#[test]
fn add_property_with_default() {
    let mut schema = base_schema();

    let result = apply_change(
        &mut schema,
        SchemaChange::AddProperty(
            PropertyDef::new("status", PropertyType::String)
                .with_default(Value::String("active".into())),
        ),
    );
    assert!(result.is_ok());
    let prop = schema.get_property("status").expect("should exist");
    assert_eq!(prop.default, Some(Value::String("active".into())));
}

#[test]
fn add_property_not_null_with_default_ok() {
    let mut schema = base_schema();

    let result = apply_change(
        &mut schema,
        SchemaChange::AddProperty(
            PropertyDef::new("email", PropertyType::String)
                .not_null()
                .with_default(Value::String("unknown@example.com".into())),
        ),
    );
    assert!(result.is_ok());
}

#[test]
fn add_property_not_null_without_default_rejected() {
    let mut schema = base_schema();

    let result = apply_change(
        &mut schema,
        SchemaChange::AddProperty(PropertyDef::new("email", PropertyType::String).not_null()),
    );
    assert!(matches!(
        result,
        Err(EvolutionError::NotNullWithoutDefault { .. })
    ));
}

#[test]
fn type_change_rejected() {
    let mut schema = base_schema();

    // Try to change "age" from INT to STRING
    let result = apply_change(
        &mut schema,
        SchemaChange::AddProperty(PropertyDef::new("age", PropertyType::String)),
    );
    assert!(matches!(
        result,
        Err(EvolutionError::TypeChangeRejected { .. })
    ));
}

#[test]
fn type_change_same_type_idempotent() {
    let mut schema = base_schema();

    // Adding "age" as INT again should be idempotent
    let result = apply_change(
        &mut schema,
        SchemaChange::AddProperty(PropertyDef::new("age", PropertyType::Int)),
    );
    assert!(result.is_ok());
}

#[test]
fn remove_property() {
    let mut schema = base_schema();
    let v_before = schema.schema_revision;

    let result = apply_change(&mut schema, SchemaChange::RemoveProperty("age".to_string()));
    assert!(result.is_ok());
    assert!(schema.get_property("age").is_none());
    // Per ADR-023, property removals do not bump the schema revision.
    assert_eq!(schema.schema_revision, v_before);
}

#[test]
fn remove_nonexistent_property() {
    let mut schema = base_schema();

    let result = apply_change(
        &mut schema,
        SchemaChange::RemoveProperty("nonexistent".to_string()),
    );
    assert!(matches!(
        result,
        Err(EvolutionError::PropertyNotFound { .. })
    ));
}

#[test]
fn is_type_change_always_rejected() {
    assert!(!is_type_change_allowed(
        &PropertyType::String,
        &PropertyType::Int
    ));
    assert!(!is_type_change_allowed(
        &PropertyType::Int,
        &PropertyType::Int
    ));
    assert!(!is_type_change_allowed(
        &PropertyType::Vector {
            dimensions: 384,
            metric: VectorMetric::Cosine
        },
        &PropertyType::Vector {
            dimensions: 128,
            metric: VectorMetric::L2
        },
    ));
}

#[test]
fn resolve_with_default_present() {
    let prop = PropertyDef::new("status", PropertyType::String)
        .with_default(Value::String("active".into()));

    let val = Value::String("custom".into());
    assert_eq!(resolve_with_default(Some(&val), &prop), Some(&val));
}

#[test]
fn resolve_with_default_missing() {
    let prop = PropertyDef::new("status", PropertyType::String)
        .with_default(Value::String("active".into()));

    let result = resolve_with_default(None, &prop);
    assert_eq!(result, Some(&Value::String("active".into())));
}

#[test]
fn resolve_with_default_missing_no_default() {
    let prop = PropertyDef::new("bio", PropertyType::String);

    let result = resolve_with_default(None, &prop);
    assert_eq!(result, None);
}

#[test]
fn evolution_error_display() {
    let err = EvolutionError::TypeChangeRejected {
        property: "age".into(),
        from: "INT".into(),
        to: "STRING".into(),
    };
    assert_eq!(
        err.to_string(),
        "cannot change type of 'age' from INT to STRING"
    );
}

#[test]
fn multiple_changes_sequential() {
    let mut schema = base_schema();

    apply_change(
        &mut schema,
        SchemaChange::AddProperty(PropertyDef::new("bio", PropertyType::String)),
    )
    .expect("add bio");

    apply_change(
        &mut schema,
        SchemaChange::AddProperty(PropertyDef::new(
            "embedding",
            PropertyType::Vector {
                dimensions: 384,
                metric: VectorMetric::Cosine,
            },
        )),
    )
    .expect("add embedding");

    apply_change(&mut schema, SchemaChange::RemoveProperty("age".to_string())).expect("remove age");

    assert_eq!(schema.properties.len(), 3); // name, bio, embedding
    assert!(schema.get_property("age").is_none());
    assert!(schema.get_property("bio").is_some());
    assert!(schema.get_property("embedding").is_some());
}
