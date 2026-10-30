use super::*;
use coordinode_core::schema::definition::{LabelSchema, PlacementPolicy};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};

fn test_engine(dir: &std::path::Path) -> StorageEngine {
    StorageEngine::open(&StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]))
    .expect("open engine")
}

/// `name` declared implying `implies`, at `revision`.
fn declared(name: &str, implies: &[&str], revision: u64) -> LabelSchema {
    let mut schema = LabelSchema::new(name, PlacementPolicy::NodeId);
    schema.set_implies(implies.iter().map(|l| l.to_string()).collect());
    schema.schema_revision = revision;
    schema
}

/// A view of the catalog is one committed state for its whole pass. The
/// implication of Thing moves from Task to Chore after the view is taken;
/// the carriers it gives are those of the state it was taken at, never a
/// mix in which both types or neither imply Thing. Before the view was
/// pinned, each fact was read at the latest state when first asked for.
#[test]
fn a_view_judges_every_fact_by_the_state_it_was_taken_at() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let schemas = LocalSchemaStore::new(&engine);
    for schema in [
        declared("Thing", &[], 1),
        declared("Task", &["Thing"], 1),
        declared("Chore", &[], 1),
    ] {
        schemas.save_label(&schema).expect("declare");
    }
    let task = schemas.label_id("Task").expect("id").expect("bound");

    let mut taken = CatalogMembership::current(&engine);
    schemas
        .save_label(&declared("Task", &[], 2))
        .expect("Task stops implying Thing");
    schemas
        .save_label(&declared("Chore", &["Thing"], 2))
        .expect("Chore comes to imply Thing");

    let carriers = label_carriers_in(&engine, "Thing", &mut taken)
        .expect("carriers")
        .expect("Thing is bound");
    assert_eq!(carriers.implying, vec![task]);
}
