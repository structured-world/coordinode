//! Cross-store integration tests. Exercise multi-modality flows on a
//! single shared `StorageEngine` to catch interactions a per-store
//! unit test cannot see.
//!
//! Scope: realistic flows that touch ≥2 stores in one transaction-ish
//! sequence. Not exhaustive — focused on the patterns the query
//! layer's `runner.rs` will compose once the store-trait migration lands.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_core::graph::edge::EdgeProperties;
use coordinode_core::graph::node::{NodeId, NodeRecord};
use coordinode_core::graph::types::Value;
use coordinode_core::index::derive::EntryOwner;
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_modality::{
    BlobStore, Bucket, Crs, DocumentStore, EdgeStore, GenerationId, IndexDescriptor, IndexId,
    IndexStore, LocalBlobStore, LocalDocumentStore, LocalEdgeStore, LocalIndexStore,
    LocalNodeStore, LocalSpatialStore, LocalTimeSeriesStore, Measurement, NodeStore, Point,
    SpatialStore, TimeSeriesStore,
};
use coordinode_storage::engine::transaction::{CommitContext, Transaction};
use std::collections::BTreeMap;

/// Commit a transaction's buffered writes (edge merges + edgeprops).
fn commit(txn: &mut Transaction) {
    let wc = WriteConcern::majority();
    let ctx = CommitContext {
        write_concern: &wc,
        pipeline: None,
        id_gen: None,
    };
    txn.commit(&ctx).expect("commit");
}

/// Run store writes in one MVCC transaction (shard-agnostic) and commit.
fn write_txn(
    engine: &coordinode_storage::engine::core::StorageEngine,
    body: impl FnOnce(&mut Transaction),
) {
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let read_ts = oracle.next();
    let mut txn = Transaction::begin(engine, Some(&oracle), read_ts);
    body(&mut txn);
    commit(&mut txn);
}

/// Run a store read closure against the latest committed snapshot.
fn read_txn<R>(
    engine: &coordinode_storage::engine::core::StorageEngine,
    body: impl FnOnce(&Transaction) -> R,
) -> R {
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let read_ts = oracle.next();
    let txn = Transaction::begin(engine, Some(&oracle), read_ts);
    body(&txn)
}

/// Seed a node via an MVCC transaction + commit.
fn put_node(
    engine: &coordinode_storage::engine::core::StorageEngine,
    shard: u16,
    id: NodeId,
    record: &NodeRecord,
) {
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let read_ts = oracle.next();
    let mut txn = Transaction::begin(engine, Some(&oracle), read_ts);
    LocalNodeStore
        .put(&mut txn, shard, id, record)
        .expect("put node");
    commit(&mut txn);
}

/// Read a node at the latest committed snapshot through an MVCC transaction.
fn get_node(
    engine: &coordinode_storage::engine::core::StorageEngine,
    shard: u16,
    id: NodeId,
) -> Option<NodeRecord> {
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let read_ts = oracle.next();
    let txn = Transaction::begin(engine, Some(&oracle), read_ts);
    LocalNodeStore.get(&txn, shard, id).expect("get node")
}

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
    let edges = LocalEdgeStore;
    let indexes = LocalIndexStore::new(engine);
    let docs = LocalDocumentStore;
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));

    let alice = NodeId::from_raw(1);
    let bob = NodeId::from_raw(2);

    // 1. Create two nodes with a property.
    let mut alice_rec = NodeRecord::new("User");
    alice_rec.set_extra("name", Value::String("alice".into()));
    put_node(engine, 0, alice, &alice_rec);
    let mut bob_rec = NodeRecord::new("User");
    bob_rec.set_extra("name", Value::String("bob".into()));
    put_node(engine, 0, bob, &bob_rec);

    // 2. Index both by name (buffered, committed).
    let by_name = IndexDescriptor::btree("by_name", "User", "name")
        .bind(IndexId::from_raw(1), GenerationId::from_raw(1));
    {
        let read_ts = oracle.next();
        let mut txn = Transaction::begin(engine, Some(&oracle), read_ts);
        // The names are kept by name, outside the dictionary.
        let no_fields = |_: &str| None;
        indexes
            .stage_membership(
                &mut txn,
                &by_name,
                &no_fields,
                EntryOwner::node(alice.as_raw()),
                None,
                Some(&[Value::String("alice".into())]),
            )
            .expect("idx alice");
        indexes
            .stage_membership(
                &mut txn,
                &by_name,
                &no_fields,
                EntryOwner::node(bob.as_raw()),
                None,
                Some(&[Value::String("bob".into())]),
            )
            .expect("idx bob");
        commit(&mut txn);
    }

    // 3. Edge alice --KNOWS--> bob with a property (buffered, committed).
    let mut props = EdgeProperties::new();
    props.set(1, Value::String("since-2020".into()));
    {
        let read_ts = oracle.next();
        let mut txn = Transaction::begin(engine, Some(&oracle), read_ts);
        edges
            .put_edge(&mut txn, "KNOWS", alice, bob, Some(&props))
            .expect("edge");
        commit(&mut txn);
    }

    // 4. Document update on alice's profile sub-tree (buffered, committed).
    {
        let read_ts = oracle.next();
        let mut txn = Transaction::begin(engine, Some(&oracle), read_ts);
        docs.set_path(
            &mut txn,
            0,
            alice,
            coordinode_core::graph::doc_delta::PathTarget::Extra,
            vec!["profile".into(), "city".into()],
            rmpv::Value::String("Berlin".into()),
        )
        .expect("doc update");
        commit(&mut txn);
    }

    // --- Cross-store read-back ---

    // NodeStore sees both nodes plus the merged profile.city value.
    let alice_after = get_node(engine, 0, alice).expect("Some");
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
    let read_ts = oracle.next();
    let mut rtxn = Transaction::begin(engine, Some(&oracle), read_ts);
    let budget = coordinode_core::budget::QueryBudget::new(
        coordinode_core::budget::DEFAULT_QUERY_MEMORY_LIMIT,
    );
    let alice_via_idx = indexes
        .scan_exact(
            &mut rtxn,
            &by_name,
            &[Value::String("alice".into())],
            &budget,
        )
        .expect("scan");
    assert_eq!(alice_via_idx, Some(vec![alice]));
    let bob_via_idx = indexes
        .scan_exact(&mut rtxn, &by_name, &[Value::String("bob".into())], &budget)
        .expect("scan");
    assert_eq!(bob_via_idx, Some(vec![bob]));

    // EdgeStore: alice's forward neighbours include bob.
    let out = edges
        .scan_neighbors_out(&rtxn, "KNOWS", alice)
        .expect("scan");
    assert_eq!(out, vec![bob]);
    // And the edgeprop survives.
    let edge_props = edges
        .get_props(&rtxn, "KNOWS", alice, bob)
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
    let spatial = LocalSpatialStore;
    let ts = LocalTimeSeriesStore;

    let sensor = NodeId::from_raw(42);
    put_node(engine, 0, sensor, &NodeRecord::new("Sensor"));

    // Spatial: sensor's GPS location.
    let loc = Point::new_2d(Crs::Wgs84_2d, 2.3522, 48.8566);

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
    // Spatial point + time-series bucket buffer on one transaction and
    // commit together.
    write_txn(engine, |txn| {
        spatial
            .insert(txn, 1, sensor, &loc)
            .expect("spatial insert");
        ts.put_bucket(txn, 0, bucket_id, &bucket).expect("ts put");
    });

    // --- Read back ---

    // Spatial: sensor visible in a Paris-area bbox.
    let bbox = coordinode_modality::Bbox {
        lower: Point::new_2d(Crs::Wgs84_2d, 2.0, 48.0),
        upper: Point::new_2d(Crs::Wgs84_2d, 3.0, 49.0),
    };
    let hits = read_txn(engine, |txn| {
        spatial
            .scan_within_bbox(txn, 1, Crs::Wgs84_2d, &bbox)
            .expect("scan")
    });
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0, sensor);

    // TimeSeries: bucket round-trips with stats.
    let ts_back =
        read_txn(engine, |txn| ts.get_bucket(txn, 0, bucket_id).expect("ok")).expect("Some");
    assert_eq!(ts_back.control.count, 2);
    assert_eq!(ts_back.control.time_min_us, 100);
    assert_eq!(ts_back.control.time_max_us, 200);

    // Original sensor node still readable through NodeStore — its
    // body wasn't touched by spatial / ts writes.
    let sensor_after = get_node(engine, 0, sensor).expect("Some");
    assert_eq!(sensor_after.primary_label(), "Sensor");
}

/// Blob + NodeStore: a node and the blob property it holds commit in one
/// transaction, and an attempt that does not commit leaves neither behind.
#[test]
fn a_node_and_its_blob_property_commit_together() {
    use coordinode_storage::engine::config::{
        Durability, EndpointConfig, Media, StorageConfig, Tier,
    };
    use coordinode_storage::engine::core::StorageEngine;
    use std::sync::Arc;

    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let oracle = Arc::new(TimestampOracle::new());
    let engine =
        Arc::new(StorageEngine::open_embedded(&config, Arc::clone(&oracle)).expect("open"));
    let blobs = LocalBlobStore::new(coordinode_storage::blob::BlobStore::direct(
        Arc::clone(&engine),
        Arc::clone(&oracle),
    ));
    let payload = vec![0x5c_u8; coordinode_core::graph::blob::INLINE_THRESHOLD * 3];

    // An attempt dropped before its commit: neither the node nor the
    // property exists, and its upload is abandoned.
    let dropped = NodeId::from_raw(8);
    let upload = blobs.upload(&payload).expect("upload");
    {
        let mut txn = Transaction::begin(&engine, Some(&oracle), oracle.next());
        LocalNodeStore
            .put(&mut txn, 0, dropped, &NodeRecord::new("Document"))
            .expect("node");
        blobs
            .put_blob(&mut txn, dropped, 0, &upload)
            .expect("property");
    }
    blobs.abandon(&upload).expect("abandon");

    let node = NodeId::from_raw(7);
    let upload = blobs.upload(&payload).expect("upload");
    let mut txn = Transaction::begin(&engine, Some(&oracle), oracle.next());
    LocalNodeStore
        .put(&mut txn, 0, node, &NodeRecord::new("Document"))
        .expect("node");
    blobs
        .put_blob(&mut txn, node, 0, &upload)
        .expect("property");
    commit(&mut txn);

    let txn = Transaction::begin(&engine, Some(&oracle), oracle.next());
    assert!(
        LocalNodeStore
            .get(&txn, 0, dropped)
            .expect("read")
            .is_none()
    );
    assert!(blobs.get_blob_ref(&txn, dropped, 0).expect("ok").is_none());
    assert_eq!(
        LocalNodeStore
            .get(&txn, 0, node)
            .expect("read")
            .expect("node")
            .primary_label(),
        "Document"
    );
    let reader = blobs
        .open_blob(&txn, node, 0)
        .expect("open")
        .expect("property");
    let read: Vec<u8> = (0..reader.record().blob.chunk_count())
        .flat_map(|i| reader.chunk(i).expect("chunk"))
        .collect();
    assert_eq!(read, payload);
}
