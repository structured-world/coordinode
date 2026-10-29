use std::collections::HashMap;

use super::*;
use crate::schema::definition::{PropertyDef, PropertyType};

/// A catalog of label definitions; a label absent from it is undeclared.
#[derive(Default)]
struct Catalog(HashMap<String, LabelSchema>);

impl Catalog {
    fn with(mut self, schema: LabelSchema) -> Self {
        self.0.insert(schema.name.clone(), schema);
        self
    }

    fn facts(&self, label: &str) -> Result<LabelFacts, String> {
        Ok(self.0.get(label).map(LabelFacts::of).unwrap_or_default())
    }

    fn resolve(&self, labels: &[&str]) -> Result<NodeLabels<'static>, LabelKindError> {
        let labels: Vec<String> = labels.iter().map(|l| (*l).to_string()).collect();
        resolve_node_labels(&labels, |l| self.facts(l))
            .expect("catalog never fails")
            .map(NodeLabels::into_owned)
    }

    fn implied(&self, label: &str) -> Result<ImpliedLabels, LabelKindError> {
        compile_implied(label, |l| self.facts(l)).expect("catalog never fails")
    }

    /// Every label a node written with `labels` carries.
    fn membership(&self, labels: &[&str]) -> Vec<String> {
        let stored = self.resolve(labels).expect("labels type a node");
        let implied = self.implied(stored.primary()).expect("no cycle");
        Membership::new(&stored.labels, &implied)
            .iter()
            .map(str::to_string)
            .collect()
    }

    fn check(&self, schema: &LabelSchema) -> Result<(), LabelDeclarationError> {
        check_label_declaration(schema, |l| Ok::<_, String>(self.0.get(l).cloned()))
            .expect("catalog never fails")
    }
}

fn type_label(name: &str, implies: &[&str], props: &[(&str, PropertyType)]) -> LabelSchema {
    let mut schema = LabelSchema::new_node_id(name);
    schema.set_implies(implies.iter().map(|l| (*l).to_string()).collect());
    for (prop, ty) in props {
        schema.add_property(PropertyDef::new(*prop, ty.clone()));
    }
    schema
}

fn auxiliary(name: &str) -> LabelSchema {
    let mut schema = LabelSchema::new_node_id(name);
    schema.set_auxiliary();
    schema
}

/// Thing <- Goal, with Activity and Policy auxiliary: the shape a consumer
/// with scope labels and ancestor labels declares.
fn goals() -> Catalog {
    Catalog::default()
        .with(type_label("Thing", &[], &[("title", PropertyType::String)]))
        .with(type_label(
            "Goal",
            &["Thing"],
            &[
                ("title", PropertyType::String),
                ("due", PropertyType::Timestamp),
            ],
        ))
        .with(auxiliary("Activity"))
        .with(auxiliary("Policy"))
}

fn names<'a>(labels: &'a NodeLabels<'_>) -> Vec<&'a str> {
    labels.labels.iter().map(String::as_str).collect()
}

// --- resolve_node_labels ---

/// A single undeclared label is a FLEXIBLE type label, stored unchanged: an
/// existing single-label database keeps its records byte for byte.
#[test]
fn a_single_undeclared_label_is_its_own_type() {
    let labels = Catalog::default()
        .resolve(&["User"])
        .expect("one type label");
    assert_eq!(names(&labels), ["User"]);
    assert_eq!(labels.primary(), "User");
}

/// The primary label is found by kind, so the order labels are written in
/// changes nothing that is stored.
#[test]
fn label_order_does_not_choose_the_primary_label() {
    let catalog = goals();
    let a = catalog.resolve(&["Activity", "Goal"]).expect("one type");
    let b = catalog.resolve(&["Goal", "Activity"]).expect("one type");
    assert_eq!(a, b);
    assert_eq!(a.primary(), "Goal");
    assert_eq!(names(&a), ["Goal", "Activity"]);
}

/// A record stores its type and its own auxiliary labels; the labels the
/// type implies are its type's, carried by the node without being stored,
/// so a match on the ancestor finds it.
#[test]
fn implied_labels_are_carried_not_stored() {
    let catalog = goals();
    let labels = catalog.resolve(&["Goal"]).expect("one type");
    assert_eq!(names(&labels), ["Goal"]);
    assert_eq!(catalog.membership(&["Goal"]), ["Goal", "Thing"]);
}

/// An implied label written beside its implying type is not a second type,
/// and writing it does not make it the node's own.
#[test]
fn a_written_implied_label_is_not_a_second_type() {
    let catalog = goals();
    let labels = catalog
        .resolve(&["Thing", "Goal"])
        .expect("Thing is implied");
    assert_eq!(labels.primary(), "Goal");
    assert_eq!(names(&labels), ["Goal"]);
    assert_eq!(catalog.membership(&["Thing", "Goal"]), ["Goal", "Thing"]);
}

/// Implication is transitive.
#[test]
fn implied_labels_are_closed_transitively() {
    let catalog = Catalog::default()
        .with(type_label("Entity", &[], &[]))
        .with(type_label("Thing", &["Entity"], &[]))
        .with(type_label("Goal", &["Thing"], &[]));
    assert_eq!(
        catalog.implied("Goal").expect("no cycle").as_slice(),
        ["Entity", "Thing"]
    );
    assert_eq!(catalog.membership(&["Goal"]), ["Goal", "Entity", "Thing"]);
    let labels = catalog
        .resolve(&["Entity", "Goal"])
        .expect("Entity is implied");
    assert_eq!(labels.primary(), "Goal");
}

/// The node's own auxiliary labels and its type's implied ones merge in byte
/// order, a label in both carried once; membership answers for each.
#[test]
fn membership_merges_implied_and_auxiliary_labels() {
    let catalog = goals().with(auxiliary("Thingamajig"));
    let members = catalog.membership(&["Goal", "Thingamajig", "Activity"]);
    assert_eq!(members, ["Goal", "Activity", "Thing", "Thingamajig"]);
    let stored = catalog.resolve(&["Goal", "Activity"]).expect("one type");
    let implied = catalog.implied("Goal").expect("no cycle");
    let membership = Membership::new(&stored.labels, &implied);
    for label in ["Goal", "Activity", "Thing"] {
        assert!(membership.contains(label), "{label}");
    }
    assert!(!membership.contains("Policy"));
    assert_eq!(membership.primary(), Some("Goal"));
    assert_eq!(membership.len(), 3);
}

/// The first escape: an undeclared label beside a STRICT type is a second
/// type label, so it cannot switch the type's schema off.
#[test]
fn an_undeclared_label_beside_a_type_is_a_second_type() {
    let err = goals().resolve(&["Extra", "Goal"]).expect_err("two types");
    assert_eq!(
        err,
        LabelKindError::TwoTypeLabels {
            first: "Extra".into(),
            second: "Goal".into(),
        }
    );
}

/// The second escape: an auxiliary label alone types nothing.
#[test]
fn an_auxiliary_label_alone_is_refused() {
    let err = goals().resolve(&["Activity"]).expect_err("no type");
    assert_eq!(
        err,
        LabelKindError::NoTypeLabel {
            labels: vec!["Activity".into()],
        }
    );
}

/// Two auxiliary labels without a type are no better than one.
#[test]
fn two_auxiliary_labels_without_a_type_are_refused() {
    let err = goals()
        .resolve(&["Activity", "Policy"])
        .expect_err("no type");
    assert!(matches!(err, LabelKindError::NoTypeLabel { .. }), "{err:?}");
}

/// A node with no label at all has no type label either.
#[test]
fn no_label_is_refused() {
    let err = goals().resolve(&[]).expect_err("no type");
    assert_eq!(err, LabelKindError::NoTypeLabel { labels: vec![] });
}

/// Two declared types that do not imply each other are refused, not merged.
#[test]
fn two_unrelated_declared_types_are_refused() {
    let catalog = goals().with(type_label("Person", &[], &[]));
    let err = catalog.resolve(&["Goal", "Person"]).expect_err("two types");
    assert!(
        matches!(err, LabelKindError::TwoTypeLabels { .. }),
        "{err:?}"
    );
}

/// A label written twice is stored once.
#[test]
fn a_repeated_label_is_stored_once() {
    let labels = goals()
        .resolve(&["Goal", "Activity", "Goal", "Activity"])
        .expect("one type");
    assert_eq!(names(&labels), ["Goal", "Activity"]);
}

/// A cycle the catalog somehow holds leaves no primary label among types
/// written together, and compiling what a type implies says so instead of
/// looping.
#[test]
fn an_implication_cycle_is_refused() {
    let catalog = Catalog::default()
        .with(type_label("A", &["B"], &[]))
        .with(type_label("B", &["A"], &[]));
    let err = catalog.resolve(&["A", "B"]).expect_err("cycle");
    assert!(
        matches!(err, LabelKindError::ImplicationCycle { .. }),
        "{err:?}"
    );
    let err = catalog.implied("A").expect_err("cycle");
    assert!(
        matches!(err, LabelKindError::ImplicationCycle { .. }),
        "{err:?}"
    );
}

/// The catalog's own failure reaches the caller unchanged.
#[test]
fn a_catalog_failure_is_returned() {
    let labels = vec!["Goal".to_string()];
    let err = resolve_node_labels(&labels, |_| Err::<LabelFacts, _>("unreadable"))
        .expect_err("catalog failure");
    assert_eq!(err, "unreadable");
}

// --- check_label_declaration ---

/// The declarations of the documented example are accepted.
#[test]
fn the_documented_declarations_are_accepted() {
    let catalog = goals();
    for label in ["Thing", "Goal", "Activity"] {
        catalog.check(&catalog.0[label]).expect(label);
    }
}

/// An auxiliary label declares nothing a type label carries.
#[test]
fn an_auxiliary_label_declares_nothing() {
    let catalog = Catalog::default();
    let mut with_props = auxiliary("Scope");
    with_props.add_property(PropertyDef::new("x", PropertyType::Int));
    let mut temporal = auxiliary("Scope");
    temporal.set_temporal(true);
    let mut implying = auxiliary("Scope");
    implying.set_implies(vec!["Thing".into()]);
    for (schema, what) in [
        (with_props, "properties"),
        (temporal, "TEMPORAL"),
        (implying, "IMPLIES"),
    ] {
        assert_eq!(
            catalog.check(&schema),
            Err(LabelDeclarationError::AuxiliaryDeclares {
                label: "Scope".into(),
                what,
            })
        );
    }
}

/// A type cannot imply itself, directly or through another label.
#[test]
fn a_type_cannot_imply_itself() {
    let catalog = Catalog::default();
    let direct = type_label("Goal", &["Goal"], &[]);
    assert_eq!(
        catalog.check(&direct),
        Err(LabelDeclarationError::ImpliesItself {
            label: "Goal".into(),
            path: vec![],
        })
    );
    let catalog = catalog.with(type_label("Thing", &["Goal"], &[]));
    let through = type_label("Goal", &["Thing"], &[]);
    assert_eq!(
        catalog.check(&through),
        Err(LabelDeclarationError::ImpliesItself {
            label: "Goal".into(),
            path: vec!["Thing".into()],
        })
    );
}

/// An auxiliary label is never implied: it would type no node and carry no
/// schema, so naming it in IMPLIES is a mistake.
#[test]
fn a_type_cannot_imply_an_auxiliary_label() {
    let catalog = goals();
    let schema = type_label("Task", &["Activity"], &[]);
    assert_eq!(
        catalog.check(&schema),
        Err(LabelDeclarationError::ImpliesAuxiliary {
            label: "Task".into(),
            implied: "Activity".into(),
        })
    );
}

/// The implying schema lists every property of an implied label, so the
/// primary schema is the node's complete property list.
#[test]
fn an_implied_property_must_be_declared_with_its_type() {
    let catalog = goals();
    let missing = type_label("Task", &["Thing"], &[]);
    assert_eq!(
        catalog.check(&missing),
        Err(LabelDeclarationError::ImpliedPropertyMissing {
            label: "Task".into(),
            implied: "Thing".into(),
            property: "title".into(),
            declared: "STRING".into(),
        })
    );
    let other_type = type_label("Task", &["Thing"], &[("title", PropertyType::Int)]);
    assert!(matches!(
        catalog.check(&other_type),
        Err(LabelDeclarationError::ImpliedPropertyMissing { .. })
    ));
}

/// Properties of a transitively implied label count too.
#[test]
fn a_transitively_implied_property_must_be_declared() {
    let catalog = Catalog::default()
        .with(type_label("Entity", &[], &[("id", PropertyType::String)]))
        .with(type_label(
            "Thing",
            &["Entity"],
            &[("id", PropertyType::String)],
        ));
    let schema = type_label("Goal", &["Thing"], &[]);
    assert!(matches!(
        catalog.check(&schema),
        Err(LabelDeclarationError::ImpliedPropertyMissing { implied, .. }) if implied == "Thing"
    ));
}

/// An undeclared implied label is an implicit FLEXIBLE type declaring no
/// property, so nothing is required of the implying schema.
#[test]
fn an_undeclared_label_may_be_implied() {
    let schema = type_label("Goal", &["Thing"], &[]);
    assert_eq!(Catalog::default().check(&schema), Ok(()));
}

/// IMPLIES is stored sorted and without repeats, so equal declarations are
/// equal definitions.
#[test]
fn implies_is_kept_sorted_and_unique() {
    let schema = type_label("Goal", &["Thing", "Entity", "Thing"], &[]);
    assert_eq!(schema.implies(), ["Entity", "Thing"]);
}
