use coordinode_core::schema::definition::{
    ConstraintKind, ConstraintState, LabelSchema, PropertyDef, PropertyType,
};

use super::{ConstraintRequest, plan_constraint};
use crate::executor::runner::ExecutionError;

fn request(kind: ConstraintKind) -> ConstraintRequest {
    ConstraintRequest {
        name: None,
        label: "User".into(),
        properties: vec!["email".into()],
        kind,
        scope: None,
        on_duplicate_rename: None,
        sparse: None,
        maintenance: None,
    }
}

fn user_with(property: PropertyType) -> LabelSchema {
    let mut schema = LabelSchema::new_node_id("User");
    schema
        .properties
        .insert("email".into(), PropertyDef::new("email", property));
    schema
}

/// A uniqueness is published validating, with the sparse unique index it
/// owns named after it: writers enforce it from publication, the build
/// proves it over stored nodes.
#[test]
fn a_uniqueness_is_planned_validating_with_its_owned_index() {
    let planned = plan_constraint(&request(ConstraintKind::Unique), None).unwrap();
    assert_eq!(planned.constraint.name, "User_email_unique");
    assert_eq!(planned.constraint.state, ConstraintState::Validating);
    let owned = planned.owned_index.expect("a uniqueness owns an index");
    assert_eq!(owned.descriptor.name.as_deref(), Some("User_email_unique"));
    assert_eq!(owned.descriptor.owner.as_deref(), Some("User_email_unique"));
    assert!(owned.descriptor.unique);
    assert!(owned.descriptor.sparse);
}

/// A key requires its values of every node, so its index is not sparse.
#[test]
fn a_key_owns_a_dense_index() {
    let planned = plan_constraint(&request(ConstraintKind::NodeKey), None).unwrap();
    assert!(!planned.owned_index.unwrap().descriptor.sparse);
}

/// A presence constraint is checked on each node by the commit publishing
/// it, so it is published active and owns nothing.
#[test]
fn a_presence_constraint_is_planned_active_without_an_index() {
    let planned = plan_constraint(&request(ConstraintKind::NotNull), None).unwrap();
    assert_eq!(planned.constraint.state, ConstraintState::Active);
    assert!(planned.owned_index.is_none());
}

/// The definition the constraint is planned against decides whether it can
/// hold: a type constraint against another declared type is refused before
/// anything is published.
#[test]
fn a_constraint_that_cannot_hold_under_the_definition_is_refused() {
    let schema = user_with(PropertyType::String);
    let refused = plan_constraint(
        &request(ConstraintKind::Type(PropertyType::Int)),
        Some(&schema),
    );
    assert!(
        matches!(refused, Err(ExecutionError::CatalogRefused(ref why)) if why.contains("can never be stored")),
        "{refused:?}"
    );
}

/// A repair rewrites a string; a property the definition declares otherwise
/// cannot be its target.
#[test]
fn a_repair_of_a_non_string_property_is_refused() {
    let schema = user_with(PropertyType::Int);
    let mut asked = request(ConstraintKind::Unique);
    asked.on_duplicate_rename = Some("email".into());
    let refused = plan_constraint(&asked, Some(&schema));
    assert!(
        matches!(refused, Err(ExecutionError::CatalogRefused(ref why)) if why.contains("appends a suffix")),
        "{refused:?}"
    );
    let planned = plan_constraint(&asked, Some(&user_with(PropertyType::String))).unwrap();
    assert_eq!(
        planned.owned_index.unwrap().repair.unwrap().property,
        "email"
    );
}

/// Only a uniqueness can hold among the nodes a predicate admits.
#[test]
fn a_scoped_key_is_refused() {
    let mut asked = request(ConstraintKind::NodeKey);
    asked.scope = Some(
        coordinode_core::index::filter::PartialFilter::PropertyEquals {
            property: "tier".into(),
            value: "gold".into(),
        },
    );
    assert!(matches!(
        plan_constraint(&asked, None),
        Err(ExecutionError::CatalogRefused(_))
    ));
}
