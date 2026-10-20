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

/// Blob payload is held only on the member that writes it, so a member of a
/// group with other members refuses blob writes before anything is written,
/// with a reason a client branches on; a member alone serves them.
#[tokio::test(flavor = "multi_thread")]
async fn a_blob_write_in_a_group_is_refused_before_it_is_written() {
    use coordinode_embed::replication::{
        Founding, MemberOptions, PeerTimeouts, Peering, ReplicatedDatabase,
    };
    use coordinode_test_fixtures::{TestPki, alloc_port};

    let pki = TestPki::new();
    let storage = |dir: &tempfile::TempDir| {
        StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            dir.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )])
    };
    let open = |id: u64, port: u16, founding: Founding, dir: &tempfile::TempDir| {
        let mut options = MemberOptions::new(id);
        options.peering = Some(Peering {
            security: pki.context(id),
            advertise: format!("127.0.0.1:{port}"),
            founding,
            timeouts: PeerTimeouts::default(),
        });
        let config = storage(dir);
        async move {
            ReplicatedDatabase::open(
                &config,
                options,
                Some(format!("127.0.0.1:{port}").parse().expect("addr")),
            )
            .await
            .expect("open member")
        }
    };
    let (dir1, dir2) = (
        tempfile::tempdir().expect("dir"),
        tempfile::tempdir().expect("dir"),
    );
    let (p1, p2) = (alloc_port(), alloc_port());
    let founder = open(1, p1, Founding::Form { deployment: None }, &dir1).await;
    let service = BlobServiceImpl::new(Arc::clone(founder.member().engine()))
        .with_node(Arc::clone(founder.member().node()));

    let alone = service
        .delete_blob(Request::new(graph::DeleteBlobRequest {
            blob_id: "absent".into(),
        }))
        .await
        .expect_err("nothing to delete");
    assert_eq!(
        alone.code(),
        tonic::Code::NotFound,
        "a member alone serves blob writes"
    );

    let enrollment = founder
        .enrollment(format!("127.0.0.1:{p1}"))
        .expect("enrollment");
    let joiner = open(2, p2, Founding::Join(enrollment), &dir2).await;
    founder
        .add_learner(2, format!("127.0.0.1:{p2}"), pki.identity(2))
        .await
        .expect("add learner");

    let refused = service
        .delete_blob(Request::new(graph::DeleteBlobRequest {
            blob_id: "absent".into(),
        }))
        .await
        .expect_err("refused in a group");
    assert_eq!(refused.code(), tonic::Code::FailedPrecondition);
    let info = tonic_types::StatusExt::get_details_error_info(&refused).expect("error info");
    assert_eq!(info.reason, "BLOB_NOT_REPLICATED");
    assert_eq!(info.metadata.get("members").map(String::as_str), Some("2"));

    joiner.shutdown().await.expect("shutdown");
    founder.shutdown().await.expect("shutdown");
}
