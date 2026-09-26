use std::sync::Arc;

use coordinode_core::graph::blob::{self, encode_blob_key};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use tokio_stream::StreamExt as _;
use tonic::Request;

use super::{BlobServiceImpl, blob_id_from_ref, encode_blobmeta_key};
use crate::proto::graph;
use crate::proto::graph::blob_service_server::BlobService;

fn open_engine() -> (Arc<StorageEngine>, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = Arc::new(StorageEngine::open(&config).expect("open engine"));
    (engine, dir)
}

/// Store `data` the way `upload_blob` does: content-addressed chunks, each
/// written once, plus the blob's metadata. Returns the blob id.
fn store_blob(engine: &StorageEngine, data: &[u8]) -> String {
    let (blob_ref, chunks) = blob::create_blob(data);
    for (chunk_id, chunk_data) in &chunks {
        let key = encode_blob_key(chunk_id);
        if engine.get(Partition::Blob, &key).expect("get").is_none() {
            engine.put(Partition::Blob, &key, chunk_data).expect("put");
        }
    }
    let blob_id = blob_id_from_ref(&blob_ref);
    engine
        .put(
            Partition::BlobRef,
            &encode_blobmeta_key(&blob_id),
            &blob_ref.to_msgpack().expect("encode"),
        )
        .expect("put meta");
    blob_id
}

/// Chunks are content-addressed and shared between blobs, so deleting one
/// blob must not delete a chunk another blob still reads.
#[tokio::test]
async fn deleting_a_blob_leaves_chunks_other_blobs_share() {
    let (engine, _dir) = open_engine();
    let service = BlobServiceImpl::new(Arc::clone(&engine));

    // One full default-size chunk in common, then a distinct tail.
    let shared = vec![0xAB_u8; blob::DEFAULT_CHUNK_SIZE];
    let mut first = shared.clone();
    first.extend_from_slice(b"first tail");
    let mut second = shared;
    second.extend_from_slice(b"second tail");
    let first_id = store_blob(&engine, &first);
    let second_id = store_blob(&engine, &second);

    service
        .delete_blob(Request::new(graph::DeleteBlobRequest { blob_id: first_id }))
        .await
        .expect("delete the first blob");

    let mut stream = service
        .download_blob(Request::new(graph::DownloadBlobRequest {
            blob_id: second_id,
        }))
        .await
        .expect("download the second blob")
        .into_inner();
    let mut downloaded = Vec::new();
    while let Some(chunk) = stream.next().await {
        downloaded.extend_from_slice(&chunk.expect("chunk present").data);
    }
    assert_eq!(
        downloaded, second,
        "the surviving blob must read back whole"
    );
}
