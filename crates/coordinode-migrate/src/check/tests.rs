use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

use super::check;

/// A record a family's type does not read is reported with its family, key
/// and content; markers sharing a family's prefix are not records and are
/// not reported; every store under the data directory is checked.
#[test]
fn unreadable_records_are_reported_and_markers_are_not() {
    let dir = tempfile::tempdir().expect("tempdir");
    let data = dir.path().join("data");
    {
        let engine =
            StorageEngine::open(&StorageConfig::with_endpoints(vec![EndpointConfig::new(
                "default",
                &data,
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            )]))
            .expect("open");
        // A label schema record of three fields: no current shape.
        let mut broken = Vec::new();
        rmpv::encode::write_value(
            &mut broken,
            &rmpv::Value::Array(vec!["Agent".into(), rmpv::Value::Nil, 1.into()]),
        )
        .expect("encode");
        engine
            .put(Partition::Schema, b"schema:label:Agent:3", &broken)
            .expect("put");
        // A marker under the edge type prefix: not a revisioned record.
        engine
            .put(Partition::Schema, b"schema:edge_type:OWNS:marker", b"\x01")
            .expect("put");
        engine
            .create_checkpoint(&data.join("checkpoints/ckpt-1"))
            .expect("checkpoint");
        engine.persist().expect("persist");
    }

    let checked = check(&data).expect("check");
    assert_eq!(checked.len(), 2, "the store and its checkpoint");
    for store in &checked {
        assert_eq!(store.unreadable.len(), 1, "{}", store.path.display());
        let bad = &store.unreadable[0];
        assert_eq!(bad.what, "label schema");
        assert!(bad.key.contains("schema:label:Agent:3"), "{}", bad.key);
        assert!(bad.content.contains("Agent"), "{}", bad.content);
        assert!(
            store
                .checked
                .iter()
                .any(|(what, n)| *what == "edge type schema" && *n == 0),
            "the marker is not a record: {:?}",
            store.checked
        );
    }
}
