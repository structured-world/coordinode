use std::sync::Arc;

use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;

use super::*;

fn open_engine() -> (tempfile::TempDir, Arc<StorageEngine>) {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = Arc::new(StorageEngine::open(&config).expect("open"));
    (dir, engine)
}

/// A hold is listed by name while it lives and gone once dropped, shares
/// included, and holds of another engine are not listed.
#[test]
fn live_holds_are_listed_by_name() {
    let (_dir, engine) = open_engine();
    let (_other_dir, other) = open_engine();
    let first = EngineHold::new(Arc::clone(&engine), "first");
    let second = first.share("second");
    let _elsewhere = EngineHold::new(Arc::clone(&other), "elsewhere");

    let mut names = engine_holders(&engine);
    names.sort_unstable();
    assert_eq!(names, ["first", "second"]);

    drop(first);
    assert_eq!(engine_holders(&engine), ["second"]);
    drop(second);
    assert!(engine_holders(&engine).is_empty());
    assert_eq!(Arc::strong_count(&engine), 1);
}
