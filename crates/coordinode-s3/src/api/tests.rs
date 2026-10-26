use std::sync::Arc;

use bytes::Bytes;
use coordinode_core::graph::blob::{self, ContentId, ScopedContent, SharingScope};
use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_storage::blob::BlobStore;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use http_body_util::BodyExt as _;
use hyper::StatusCode;

use super::{delete_object, get_object, put_object};

fn open_store() -> (BlobStore, tempfile::TempDir) {
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
    (BlobStore::direct(engine, oracle), dir)
}

async fn put(store: &BlobStore, path: &str, data: &[u8]) {
    let answer = put_object(store, path, Bytes::copy_from_slice(data))
        .await
        .expect("put");
    assert_eq!(answer.status(), StatusCode::OK);
}

async fn get(store: &BlobStore, path: &str) -> (StatusCode, Bytes) {
    let answer = get_object(store, path).await.expect("get");
    let status = answer.status();
    (
        status,
        answer.into_body().collect().await.expect("body").to_bytes(),
    )
}

fn refs(store: &BlobStore, data: &[u8]) -> i64 {
    let key = blob::refs_counter_key(&ScopedContent::new(
        SharingScope::STORE,
        ContentId::of(data),
    ));
    store
        .engine()
        .get(Partition::Counter, &key)
        .expect("count")
        .map_or(0, |b| {
            coordinode_storage::engine::merge::decode_counter(&b).expect("decode")
        })
}

/// Two keys holding the same bytes share every chunk, so deleting one
/// object must leave the other readable.
#[tokio::test(flavor = "multi_thread")]
async fn deleting_an_object_leaves_a_copy_readable() {
    let (store, _dir) = open_store();
    let data = vec![0x5A_u8; blob::DEFAULT_CHUNK_SIZE + 17];
    put(&store, "/blobs/original", &data).await;
    put(&store, "/blobs/copy", &data).await;

    let deleted = delete_object(&store, "/blobs/original")
        .await
        .expect("delete");
    assert_eq!(deleted.status(), StatusCode::NO_CONTENT);

    let (status, body) = get(&store, "/blobs/copy").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body.as_ref(), data.as_slice());
    assert_eq!(
        get(&store, "/blobs/original").await.0,
        StatusCode::NOT_FOUND
    );
}

/// Overwriting a key replaces its object and releases the bytes it held in
/// the same commit; putting the same bytes again counts them once.
#[tokio::test(flavor = "multi_thread")]
async fn overwriting_a_key_releases_what_it_held() {
    let (store, _dir) = open_store();
    put(&store, "/blobs/k", b"first version").await;
    put(&store, "/blobs/k", b"first version").await;
    assert_eq!(refs(&store, b"first version"), 1, "a repeat counts once");

    put(&store, "/blobs/k", b"second version").await;
    assert_eq!(refs(&store, b"first version"), 0);
    assert_eq!(refs(&store, b"second version"), 1);
    assert_eq!(get(&store, "/blobs/k").await.1.as_ref(), b"second version");

    let deleted = delete_object(&store, "/blobs/k").await.expect("delete");
    assert_eq!(deleted.status(), StatusCode::NO_CONTENT);
    assert_eq!(refs(&store, b"second version"), 0);
    // Deleting a key that holds nothing succeeds and releases nothing.
    let again = delete_object(&store, "/blobs/k").await.expect("again");
    assert_eq!(again.status(), StatusCode::NO_CONTENT);
    assert_eq!(refs(&store, b"second version"), 0);
}
