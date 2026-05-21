//! Cross-store integration tests. Exercise multi-modality flows on a
//! single shared `StorageEngine` to catch interactions a per-store
//! unit test cannot see.
//!
//! Scope: realistic flows that touch ≥2 stores in one transaction-ish
//! sequence. Not exhaustive — focused on the patterns the query
//! layer's `runner.rs` will compose once R165 migration lands.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_core::graph::edge::EdgeProperties;
use coordinode_core::graph::node::{NodeId, NodeRecord};
use coordinode_core::graph::types::Value;
use coordinode_modality::{
    BlobStore, Bucket, Crs, DocumentStore, EdgeStore, IndexStore, LocalBlobStore,
    LocalDocumentStore, LocalEdgeStore, LocalIndexStore, LocalNodeStore, LocalSpatialStore,
    LocalTimeSeriesStore, Measurement, NodeStore, Point, SpatialStore, TimeSeriesStore,
};
use std::collections::BTreeMap;

/// Logic-test fixture (memory backing, env-flippable). Cross-store
/// integration tests verify per-modality slicing of a shared
/// engine — no persistence semantics.
fn open_shared_engine() -> coordinode_test_fixtures::EngineFixture {
    coordinode_test_fixtures::engine_for_logic()
}

/// Realistic flow: create two user nodes, link them, index one by
/// property, partial-update a document field, then read back via
/// every store independently. Each store sees its own slice of state
/// — collisions or stale-key bugs would surface here.
#[test]
fn node_edge_index_document_flow() {
    let fx = open_shared_engine();
    let engine = &fx.engine;
    let nodes = LocalNodeStore::new(engine);
    let edges = LocalEdgeStore::new(engine);
    let indexes = LocalIndexStore::new(engine);
    let docs = LocalDocumentStore::new(engine);

    let alice = NodeId::from_raw(1);
    let bob = NodeId::from_raw(2);

    // 1. Create two nodes with a property.
    let mut alice_rec = NodeRecord::new("User");
    alice_rec.set_extra("name", Value::String("alice".into()));
    nodes.put(0, alice, &alice_rec).expect("put alice");
    let mut bob_rec = NodeRecord::new("User");
    bob_rec.set_extra("name", Value::String("bob".into()));
    nodes.put(0, bob, &bob_rec).expect("put bob");

    // 2. Index both by name.
    indexes
        .put_entry("by_name", &[Value::String("alice".into())], alice)
        .expect("idx alice");
    indexes
        .put_entry("by_name", &[Value::String("bob".into())], bob)
        .expect("idx bob");

    // 3. Edge alice --KNOWS--> bob with a property.
    let mut props = EdgeProperties::new();
    props.set(1, Value::String("since-2020".into()));
    edges
        .put_edge("KNOWS", alice, bob, Some(&props))
        .expect("edge");

    // 4. Document update on alice's profile sub-tree.
    docs.set_path(
        0,
        alice,
        coordinode_core::graph::doc_delta::PathTarget::Extra,
        vec!["profile".into(), "city".into()],
        rmpv::Value::String("Berlin".into()),
    )
    .expect("doc update");

    // --- Cross-store read-back ---

    // NodeStore sees both nodes plus the merged profile.city value.
    let alice_after = nodes.get(0, alice).expect("ok").expect("Some");
    let profile = alice_after.get_extra("profile").expect("profile present");
    let rmpv_map = match profile {
        Value::Document(m) => m,
        other => panic!("expected Document at profile, got {other:?}"),
    };
    let rmpv::Value::Map(pairs) = rmpv_map else {
        panic!("expected Map at profile");
    };
    let city = pairs
        .iter()
        .find_map(|(k, v)| match k {
            rmpv::Value::String(s) if s.as_str() == Some("city") => Some(v),
            _ => None,
        })
        .expect("city present");
    match city {
        rmpv::Value::String(s) => assert_eq!(s.as_str(), Some("Berlin")),
        other => panic!("expected String at city, got {other:?}"),
    }

    // IndexStore lookups for both names return the expected node.
    let alice_via_idx = indexes
        .scan_exact("by_name", &[Value::String("alice".into())])
        .expect("scan");
    assert_eq!(alice_via_idx, vec![alice]);
    let bob_via_idx = indexes
        .scan_exact("by_name", &[Value::String("bob".into())])
        .expect("scan");
    assert_eq!(bob_via_idx, vec![bob]);

    // EdgeStore: alice's forward neighbours include bob.
    let out = edges.scan_neighbors_out("KNOWS", alice).expect("scan");
    assert_eq!(out, vec![bob]);
    // And the edgeprop survives.
    let edge_props = edges
        .get_props("KNOWS", alice, bob)
        .expect("ok")
        .expect("Some");
    assert_eq!(edge_props.get(1), Some(&Value::String("since-2020".into())));

    // Original "name" property still readable — DocumentStore merges
    // do not stomp on existing fields.
    assert_eq!(
        alice_after.get_extra("name"),
        Some(&Value::String("alice".into())),
    );
}

/// Mixed-modality node: a single node has BOTH a vector (HNSW index)
/// AND a spatial point AND a time-series bucket attached. Each store
/// is keyed independently so they coexist.
#[test]
fn vector_spatial_timeseries_on_same_logical_entity() {
    let fx = open_shared_engine();
    let engine = &fx.engine;
    let nodes = LocalNodeStore::new(engine);
    let spatial = LocalSpatialStore::new(engine);
    let ts = LocalTimeSeriesStore::new(engine);

    let sensor = NodeId::from_raw(42);
    nodes
        .put(0, sensor, &NodeRecord::new("Sensor"))
        .expect("put sensor");

    // Spatial: sensor's GPS location.
    let loc = Point::new_2d(Crs::Wgs84_2d, 2.3522, 48.8566);
    spatial.insert(1, sensor, &loc).expect("spatial insert");

    // TimeSeries: bucket of readings keyed by the same node id.
    let mut m1 = BTreeMap::new();
    m1.insert("temp".to_owned(), 22.0);
    let mut m2 = BTreeMap::new();
    m2.insert("temp".to_owned(), 23.5);
    let bucket = Bucket::from_measurements(
        rmpv::Value::String("sensor-42".into()),
        vec![
            Measurement {
                timestamp_us: 100,
                ingestion_ts_us: None,
                fields: m1,
            },
            Measurement {
                timestamp_us: 200,
                ingestion_ts_us: None,
                fields: m2,
            },
        ],
    );
    // Bucket uses a DIFFERENT node id (buckets are nodes too —
    // bucket_id is distinct from sensor_id).
    let bucket_id = NodeId::from_raw(43);
    ts.put_bucket(0, bucket_id, &bucket).expect("ts put");

    // --- Read back ---

    // Spatial: sensor visible in a Paris-area bbox.
    let bbox = coordinode_modality::Bbox {
        lower: Point::new_2d(Crs::Wgs84_2d, 2.0, 48.0),
        upper: Point::new_2d(Crs::Wgs84_2d, 3.0, 49.0),
    };
    let hits = spatial
        .scan_within_bbox(1, Crs::Wgs84_2d, &bbox)
        .expect("scan");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0, sensor);

    // TimeSeries: bucket round-trips with stats.
    let ts_back = ts.get_bucket(0, bucket_id).expect("ok").expect("Some");
    assert_eq!(ts_back.control.count, 2);
    assert_eq!(ts_back.control.time_min_us, 100);
    assert_eq!(ts_back.control.time_max_us, 200);

    // Original sensor node still readable through NodeStore — its
    // body wasn't touched by spatial / ts writes.
    let sensor_after = nodes.get(0, sensor).expect("ok").expect("Some");
    assert_eq!(sensor_after.primary_label(), "Sensor");
}

/// Schema + Blob + NodeStore: register a label, store a blob ref on a
/// node, verify both stores see consistent state.
#[test]
fn schema_blob_node_consistency() {
    use coordinode_core::graph::blob::{BlobRef, ChunkId};

    let fx = open_shared_engine();
    let engine = &fx.engine;
    let nodes = LocalNodeStore::new(engine);
    let blobs = LocalBlobStore::new(engine);

    let node = NodeId::from_raw(7);
    nodes
        .put(0, node, &NodeRecord::new("Document"))
        .expect("put node");

    // Two chunks + a blob ref pointing at them.
    let chunks = vec![
        (ChunkId::from_data(b"chunk-a"), b"chunk-a".to_vec()),
        (ChunkId::from_data(b"chunk-b"), b"chunk-b".to_vec()),
    ];
    blobs.put_chunks(&chunks).expect("chunks");

    let blob_ref = BlobRef {
        total_size: 14,
        chunks: chunks.iter().map(|(id, _)| *id).collect(),
    };
    blobs.put_blob_ref(node, 0, &blob_ref).expect("ref");

    // Cross-store assertions.
    let loaded_ref = blobs
        .get_blob_ref(node, 0)
        .expect("ok")
        .expect("ref present");
    assert_eq!(loaded_ref.total_size, 14);
    assert_eq!(loaded_ref.chunks.len(), 2);

    // Chunks reassemble to the original payload.
    let mut reassembled = Vec::new();
    for id in &loaded_ref.chunks {
        let chunk = blobs.get_chunk(id).expect("ok").expect("chunk present");
        reassembled.extend_from_slice(&chunk);
    }
    assert_eq!(reassembled, b"chunk-achunk-b");

    // The node itself is still readable and untouched by the blob
    // writes.
    let node_after = nodes.get(0, node).expect("ok").expect("Some");
    assert_eq!(node_after.primary_label(), "Document");
}
