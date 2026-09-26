use coordinode_core::graph::blob::{self, encode_blob_key};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use http_body_util::BodyExt as _;
use hyper::StatusCode;

use super::{delete_object, get_object, s3_meta_key};

fn open_engine() -> (StorageEngine, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    (StorageEngine::open(&config).expect("open engine"), dir)
}

/// Store `data` under `path` the way `put_object` does: content-addressed
/// chunks, each written once, plus the object's metadata.
fn store_object(engine: &StorageEngine, path: &str, data: &[u8]) {
    let (blob_ref, chunks) = blob::create_blob(data);
    for (chunk_id, chunk_data) in &chunks {
        let key = encode_blob_key(chunk_id);
        if engine.get(Partition::Blob, &key).expect("get").is_none() {
            engine.put(Partition::Blob, &key, chunk_data).expect("put");
        }
    }
    engine
        .put(
            Partition::BlobRef,
            &s3_meta_key(path),
            &blob_ref.to_msgpack().expect("encode"),
        )
        .expect("put meta");
}

/// Two keys holding the same bytes share every chunk, so deleting one
/// object must leave the other readable.
#[tokio::test]
async fn deleting_an_object_leaves_a_copy_readable() {
    let (engine, _dir) = open_engine();
    let data = vec![0x5A_u8; blob::DEFAULT_CHUNK_SIZE + 17];
    store_object(&engine, "/blobs/original", &data);
    store_object(&engine, "/blobs/copy", &data);

    let deleted = delete_object(&engine, "/blobs/original").expect("delete");
    assert_eq!(deleted.status(), StatusCode::NO_CONTENT);

    let copy = get_object(&engine, "/blobs/copy").expect("the copy must still read back");
    assert_eq!(copy.status(), StatusCode::OK);
    let body = copy.into_body().collect().await.expect("body").to_bytes();
    assert_eq!(body.as_ref(), data.as_slice());
}
