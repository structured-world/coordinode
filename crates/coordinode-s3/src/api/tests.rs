use std::sync::Arc;

use coordinode_core::graph::blob;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use http_body_util::BodyExt as _;
use hyper::StatusCode;

use super::{delete_object, get_object, s3_meta_key};

fn open_engine() -> (Arc<StorageEngine>, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    (
        Arc::new(StorageEngine::open(&config).expect("open engine")),
        dir,
    )
}

/// Store `data` under `path` the way `put_object` does: content-addressed
/// chunks, each written once, plus the object's metadata.
fn store_object(engine: &StorageEngine, path: &str, data: &[u8]) {
    let (blob_ref, chunks) = blob::create_blob(data);
    coordinode_storage::blob::put_chunks_durably(engine, &chunks).expect("chunks");
    coordinode_storage::blob::put_meta_durably(
        engine,
        &s3_meta_key(path),
        &blob_ref.to_msgpack().expect("encode"),
    )
    .expect("meta");
}

/// Two keys holding the same bytes share every chunk, so deleting one
/// object must leave the other readable.
#[tokio::test]
async fn deleting_an_object_leaves_a_copy_readable() {
    let (engine, _dir) = open_engine();
    let data = vec![0x5A_u8; blob::DEFAULT_CHUNK_SIZE + 17];
    store_object(&engine, "/blobs/original", &data);
    store_object(&engine, "/blobs/copy", &data);

    let deleted = delete_object(&engine, "/blobs/original")
        .await
        .expect("delete");
    assert_eq!(deleted.status(), StatusCode::NO_CONTENT);

    let copy = get_object(&engine, "/blobs/copy").expect("the copy must still read back");
    assert_eq!(copy.status(), StatusCode::OK);
    let body = copy.into_body().collect().await.expect("body").to_bytes();
    assert_eq!(body.as_ref(), data.as_slice());
}
