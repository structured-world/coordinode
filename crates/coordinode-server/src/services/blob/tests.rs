use std::sync::Arc;

use coordinode_core::graph::blob;
use coordinode_embed::Database;
use coordinode_embed::operation::IssuerId;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use parking_lot::RwLock;
use tokio_stream::StreamExt as _;
use tonic::Request;

use super::BlobServiceImpl;
use crate::proto::graph;
use crate::proto::graph::blob_service_server::BlobService;

fn open_service() -> (BlobServiceImpl, Arc<RwLock<Database>>, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let database = Arc::new(RwLock::new(Database::open(dir.path()).expect("open")));
    (BlobServiceImpl::new(Arc::clone(&database)), database, dir)
}

/// The messages of an upload of `data`, in client-sized parts, the first
/// naming `operation`.
fn messages(
    data: &[u8],
    operation: Option<crate::proto::operation::OperationRef>,
) -> Vec<Result<graph::BlobChunk, tonic::Status>> {
    let mut operation = operation;
    data.chunks(64 * 1024)
        .enumerate()
        .map(|(seq, part)| {
            Ok(graph::BlobChunk {
                data: part.to_vec(),
                sequence: u32::try_from(seq).expect("sequence"),
                operation: operation.take(),
            })
        })
        .collect()
}

/// Upload `data` through the service.
async fn upload(service: &BlobServiceImpl, data: &[u8]) -> graph::UploadBlobResponse {
    service
        .upload(tokio_stream::iter(messages(data, None)))
        .await
        .expect("upload")
}

/// Every chunk of the blob `blob_id` names, concatenated, or the status the
/// download ended with.
async fn download(service: &BlobServiceImpl, blob_id: &str) -> Result<Vec<u8>, tonic::Status> {
    let mut stream = service.download(blob_id).await?;
    let mut data = Vec::new();
    while let Some(chunk) = stream.next().await {
        data.extend_from_slice(&chunk?.data);
    }
    Ok(data)
}

/// Chunks are shared between blobs, so deleting one blob must not take a
/// chunk another blob still reads.
#[tokio::test(flavor = "multi_thread")]
async fn deleting_a_blob_leaves_chunks_other_blobs_share() {
    let (service, _db, _dir) = open_service();

    // One full default-size chunk in common, then a distinct tail.
    let shared = vec![0xAB_u8; blob::DEFAULT_CHUNK_SIZE];
    let mut first = shared.clone();
    first.extend_from_slice(b"first tail");
    let mut second = shared;
    second.extend_from_slice(b"second tail");
    let first_id = upload(&service, &first).await.blob_id;
    let second = (upload(&service, &second).await, second);
    assert_eq!(second.0.dedup_chunks, 1, "the common chunk is shared");

    service
        .delete_blob(Request::new(graph::DeleteBlobRequest { blob_id: first_id }))
        .await
        .expect("delete the first blob");
    assert_eq!(
        download(&service, &second.0.blob_id)
            .await
            .expect("the second blob reads"),
        second.1,
        "the surviving blob must read back whole"
    );
}

/// Two clients uploading the same bytes are two owners: each gets its own
/// blob, and one deleting its blob leaves the other's readable. A blob named
/// by its content would make them one owner, and the first delete would take
/// the second client's blob with it.
#[tokio::test(flavor = "multi_thread")]
async fn independent_uploads_of_the_same_bytes_are_separate_blobs() {
    let (service, _db, _dir) = open_service();
    let data = vec![0x5A_u8; blob::DEFAULT_CHUNK_SIZE + 17];

    let first = upload(&service, &data).await;
    let second = upload(&service, &data).await;
    assert_ne!(
        first.blob_id, second.blob_id,
        "independent uploads are independent owners"
    );

    service
        .delete(&first.blob_id)
        .await
        .expect("delete the first");
    let gone = download(&service, &first.blob_id)
        .await
        .expect_err("the deleted blob is gone");
    assert_eq!(gone.code(), tonic::Code::NotFound);
    assert_eq!(
        download(&service, &second.blob_id)
            .await
            .expect("the second blob survives"),
        data
    );
}

/// An upload retried under its operation after a lost answer stores nothing
/// more: it is refused as settled, naming the blob the first attempt stored,
/// which reads back whole.
#[tokio::test(flavor = "multi_thread")]
async fn an_upload_retried_under_its_operation_names_its_blob() {
    let (service, database, _dir) = open_service();
    let issuer = database
        .read()
        .enroll_issuer(IssuerId([0x61; 16]))
        .expect("enroll");
    let descriptor = issuer.next().expect("descriptor");
    let reference = crate::proto::operation::OperationRef {
        issuer: descriptor.op.issuer.0.to_vec(),
        incarnation: descriptor.op.incarnation,
        sequence: descriptor.op.sequence,
        deployment: descriptor.deployment.as_bytes().to_vec(),
    };
    let data = vec![0x71_u8; 100_000];

    let stored = service
        .upload(tokio_stream::iter(messages(&data, Some(reference.clone()))))
        .await
        .expect("upload");
    let retried = service
        .upload(tokio_stream::iter(messages(&data, Some(reference))))
        .await
        .expect_err("a settled retry stores nothing");
    assert_eq!(retried.code(), tonic::Code::FailedPrecondition);
    let info = tonic_types::StatusExt::get_details_error_info(&retried).expect("error info");
    assert_eq!(info.reason, "OPERATION_SETTLED");
    assert_eq!(
        info.metadata.get("blob_id").map(String::as_str),
        Some(stored.blob_id.as_str())
    );
    assert_eq!(
        download(&service, &stored.blob_id).await.expect("read"),
        data
    );
}

/// An operation named on a later message than the first is refused.
#[tokio::test(flavor = "multi_thread")]
async fn an_operation_on_a_later_message_is_refused() {
    let (service, _db, _dir) = open_service();
    let mut parts = messages(&[1_u8; 200_000], None);
    if let Some(Ok(second)) = parts.get_mut(1) {
        second.operation = Some(crate::proto::operation::OperationRef::default());
    }
    let refused = service
        .upload(tokio_stream::iter(parts))
        .await
        .expect_err("refused");
    assert_eq!(refused.code(), tonic::Code::InvalidArgument);
}

/// A download that has started reads the blob whole even when, before the
/// client consumes the rest of the stream, the blob is deleted, its
/// representations retired and their payload erased: the download reads at
/// the snapshot it began at, and holds it until the stream ends.
#[tokio::test(flavor = "multi_thread")]
async fn a_download_survives_deletion_and_cleanup_of_its_blob() {
    let (service, database, _dir) = open_service();
    // The collector takes a deleted blob at once, and no retained history
    // holds its payload either: only the download does.
    {
        let db = database.read();
        db.set_retention_window(std::time::Duration::ZERO);
        db.set_blob_gc_policy(coordinode_embed::db::GcPolicy {
            grace: std::time::Duration::ZERO,
            compact_after_bytes: 0,
            ..coordinode_embed::db::GcPolicy::default()
        });
    }
    // Distinct chunks, more than the stream buffers ahead of the client.
    let data: Vec<u8> = (0..blob::DEFAULT_CHUNK_SIZE * 12)
        .map(|i| (i / blob::DEFAULT_CHUNK_SIZE) as u8)
        .collect();
    let stored = upload(&service, &data).await;
    let id: coordinode_embed::db::BlobId = stored.blob_id.parse().expect("id");
    let record = database
        .read()
        .blob_record(&id)
        .expect("record")
        .expect("present");

    let mut stream = service.download(&stored.blob_id).await.expect("open");
    let mut read = stream
        .next()
        .await
        .expect("first chunk")
        .expect("chunk")
        .data;
    service.delete(&stored.blob_id).await.expect("delete");
    // Collected under the running download, by this pass or by the worker
    // the deletion woke: every copy retired and its payload deleted, the
    // compaction left waiting for the download.
    database.read().run_blob_gc(true).expect("collect");
    let status = database.read().blob_gc_status().expect("status");
    assert_eq!(status.reclaimed_bytes, data.len() as u64, "{status:?}");
    assert_eq!(status.reclaimed_chunks, record.blob.distinct().len() as u64);
    assert!(status.compaction_pending, "the download holds the payload");
    while let Some(chunk) = stream.next().await {
        read.extend_from_slice(&chunk.expect("chunk").data);
    }
    assert!(read == data, "the download read the blob whole");
    assert_eq!(
        download(&service, &stored.blob_id)
            .await
            .expect_err("gone")
            .code(),
        tonic::Code::NotFound
    );
    // Once the download has let go, the space comes back. Its reader is
    // dropped by the task that streamed it, a moment after the last chunk.
    drop(stream);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        database.read().run_blob_gc(false).expect("collect");
        let status = database.read().blob_gc_status().expect("status");
        if !status.compaction_pending {
            assert!(status.reclaimed_disk_bytes > 0, "{status:?}");
            assert_eq!(status.stale_blob_bytes, 0, "{status:?}");
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the compaction never ran: {status:?}"
        );
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
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
    let service = BlobServiceImpl::new(Arc::clone(founder.member().database()))
        .with_node(Arc::clone(founder.member().node()));

    let alone = upload(&service, b"a member alone serves blob writes").await;
    assert_eq!(
        download(&service, &alone.blob_id).await.expect("read"),
        b"a member alone serves blob writes"
    );

    let enrollment = founder
        .enrollment(format!("127.0.0.1:{p1}"))
        .expect("enrollment");
    let joiner = open(2, p2, Founding::Join(enrollment), &dir2).await;
    founder
        .add_learner(2, format!("127.0.0.1:{p2}"), pki.identity(2))
        .await
        .expect("add learner");

    for refused in [
        service
            .delete_blob(Request::new(graph::DeleteBlobRequest {
                blob_id: alone.blob_id.clone(),
            }))
            .await
            .expect_err("delete refused in a group"),
        service
            .upload(tokio_stream::iter(messages(b"refused", None)))
            .await
            .expect_err("upload refused in a group"),
    ] {
        assert_eq!(refused.code(), tonic::Code::FailedPrecondition);
        let info = tonic_types::StatusExt::get_details_error_info(&refused).expect("error info");
        assert_eq!(info.reason, "BLOB_NOT_REPLICATED");
        assert_eq!(info.metadata.get("members").map(String::as_str), Some("2"));
    }
    // The database refuses on its own as well, for an embedded member.
    assert!(matches!(
        founder
            .member()
            .database()
            .read()
            .upload_blob(b"refused", None),
        Err(coordinode_embed::DatabaseError::Unsupported(_))
    ));

    joiner.shutdown().await.expect("shutdown");
    founder.shutdown().await.expect("shutdown");
}
