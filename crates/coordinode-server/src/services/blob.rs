use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, Streaming};
use tracing::info;

use coordinode_core::graph::blob::OwnerRecord;
use coordinode_embed::db::BlobId;
use coordinode_embed::{Database, DatabaseError};
// no-std: spin::RwLock (drop-in).
use parking_lot::RwLock;

use crate::proto::graph;
use crate::services::cypher::db_error_to_status;
use crate::services::error_details::{Reason, invalid_field, status_with_reason};

/// gRPC BlobService implementation.
///
/// All state is in CoordiNode storage — this service is stateless and safe
/// to run on multiple nodes behind a load balancer.
///
/// Interim blob payload path: an upload is durable on this member before it
/// is answered, and is refused on a member whose group has other members,
/// which the payload would not reach. The storage substrate's extent/stream
/// write, with payload journaled on the members the write concern names,
/// replaces both.
pub struct BlobServiceImpl {
    database: Arc<RwLock<Database>>,
    node: Option<Arc<coordinode_raft::cluster::RaftNode>>,
}

impl BlobServiceImpl {
    /// A service over `database`, standalone.
    pub fn new(database: Arc<RwLock<Database>>) -> Self {
        Self {
            database,
            node: None,
        }
    }

    /// Refuse writes while the group `node` belongs to has other members.
    #[must_use]
    pub fn with_node(mut self, node: Arc<coordinode_raft::cluster::RaftNode>) -> Self {
        self.node = Some(node);
        self
    }

    /// Refuse a blob write that the rest of the group would not hold.
    fn refuse_in_a_group(&self) -> Result<(), Status> {
        let members = self.node.as_ref().map_or(1, |n| n.member_count());
        if members <= 1 {
            return Ok(());
        }
        Err(status_with_reason(
            tonic::Code::FailedPrecondition,
            format!(
                "blob payload is held only on the member that writes it, and this group has \
                 {members} members: a leader change could lose it"
            ),
            Reason::BlobNotReplicated,
            [("members", members.to_string())],
        ))
    }
}

/// The chunks of a download, as the service streams them.
type DownloadStream = ReceiverStream<Result<graph::BlobChunk, Status>>;

#[tonic::async_trait]
impl graph::blob_service_server::BlobService for BlobServiceImpl {
    type DownloadBlobStream = DownloadStream;

    async fn upload_blob(
        &self,
        request: Request<Streaming<graph::BlobChunk>>,
    ) -> Result<Response<graph::UploadBlobResponse>, Status> {
        self.upload(request.into_inner()).await.map(Response::new)
    }

    async fn download_blob(
        &self,
        request: Request<graph::DownloadBlobRequest>,
    ) -> Result<Response<Self::DownloadBlobStream>, Status> {
        self.download(&request.into_inner().blob_id)
            .await
            .map(Response::new)
    }

    async fn delete_blob(
        &self,
        request: Request<graph::DeleteBlobRequest>,
    ) -> Result<Response<graph::DeleteBlobResponse>, Status> {
        self.delete(&request.into_inner().blob_id)
            .await
            .map(Response::new)
    }

    async fn get_blob_meta(
        &self,
        request: Request<graph::GetBlobMetaRequest>,
    ) -> Result<Response<graph::BlobMeta>, Status> {
        self.meta(&request.into_inner().blob_id)
            .await
            .map(Response::new)
    }
}

impl BlobServiceImpl {
    /// Store the object `stream` carries as a new blob, under the operation
    /// its first message names, if any.
    async fn upload<S>(&self, mut stream: S) -> Result<graph::UploadBlobResponse, Status>
    where
        S: tokio_stream::Stream<Item = Result<graph::BlobChunk, Status>> + Unpin,
    {
        use tokio_stream::StreamExt as _;

        let mut data = Vec::new();
        let mut operation = None;
        let mut first = true;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| Status::internal(format!("stream error: {e}")))?;
            match (&chunk.operation, first) {
                (Some(reference), true) => {
                    operation = Some(super::operation::descriptor_from_proto(
                        "operation",
                        reference,
                    )?);
                }
                (Some(_), false) => {
                    return Err(invalid_field(
                        "operation",
                        "the operation is named on the first message of an upload only",
                    ));
                }
                (None, _) => {}
            }
            first = false;
            data.extend_from_slice(&chunk.data);
        }

        if data.is_empty() {
            return Err(Status::invalid_argument("empty blob upload"));
        }
        self.refuse_in_a_group()?;

        // The chunks are durable before the blob naming them is published,
        // and both before the answer: an upload answered is one a crash
        // keeps. Flushing blocks, so it runs off the async runtime.
        let database = Arc::clone(&self.database);
        let descriptor = operation;
        let uploaded = tokio::task::spawn_blocking(move || {
            database.read().upload_blob(&data, descriptor.as_ref())
        })
        .await
        .map_err(|e| Status::internal(format!("blob write did not finish: {e}")))?
        .map_err(|e| match (e, &operation) {
            (
                e @ DatabaseError::OperationSettled {
                    outcome,
                    intent_differs,
                    ..
                },
                Some(descriptor),
            ) => super::operation::settled_status_naming(
                format!("blob upload: {e}"),
                outcome,
                intent_differs,
                vec![("blob_id", BlobId::for_operation(descriptor).to_string())],
            ),
            (e, _) => db_error_to_status(e),
        })?;

        let blob = &uploaded.record.blob;
        info!(
            blob_id = %uploaded.id,
            total_size = blob.total_size,
            chunks = blob.chunk_count(),
            dedup = uploaded.shared,
            "blob uploaded"
        );
        Ok(graph::UploadBlobResponse {
            blob_id: uploaded.id.to_string(),
            chunk_ids: blob.chunks.iter().map(|c| c.digest_hex()).collect(),
            total_size: blob.total_size,
            // At most one per chunk, and a blob's chunk count fits a u32.
            dedup_chunks: uploaded.shared as u32,
        })
    }

    /// Stream the blob `blob_id` names back, chunk by chunk, as it stood
    /// when the download began.
    async fn download(&self, blob_id: &str) -> Result<DownloadStream, Status> {
        let id = parse_id(blob_id)?;
        let database = Arc::clone(&self.database);
        let reader = tokio::task::spawn_blocking(move || database.read().open_blob(&id))
            .await
            .map_err(|e| Status::internal(format!("blob read did not finish: {e}")))?
            .map_err(db_error_to_status)?
            .ok_or_else(|| Status::not_found(format!("blob {blob_id} not found")))?;

        // The reader holds the snapshot every chunk resolves at until the
        // last chunk is sent or the client goes away; a bounded channel lets
        // a slow client slow the reads down rather than buffer the blob.
        let (tx, rx) = mpsc::channel(4);
        tokio::task::spawn_blocking(move || {
            for seq in 0..reader.record().blob.chunk_count() {
                let message = reader
                    .chunk(seq)
                    .map(|data| graph::BlobChunk {
                        data,
                        // A blob's chunk count fits a u32.
                        sequence: seq as u32,
                        operation: None,
                    })
                    .map_err(|e| Status::data_loss(format!("blob chunk {seq}: {e}")));
                let failed = message.is_err();
                if tx.blocking_send(message).is_err() || failed {
                    // The client went away, or the blob cannot be read whole.
                    break;
                }
            }
        });
        Ok(ReceiverStream::new(rx))
    }

    /// Delete the blob `blob_id` names.
    async fn delete(&self, blob_id: &str) -> Result<graph::DeleteBlobResponse, Status> {
        // A delete on one member of a group would leave the others' copy.
        self.refuse_in_a_group()?;
        let id = parse_id(blob_id)?;
        let database = Arc::clone(&self.database);
        let deleted = tokio::task::spawn_blocking(move || database.read().delete_blob(&id))
            .await
            .map_err(|e| Status::internal(format!("blob delete did not finish: {e}")))?
            .map_err(db_error_to_status)?;
        if deleted.is_none() {
            return Err(Status::not_found(format!("blob {blob_id} not found")));
        }
        info!(blob_id = %blob_id, "blob deleted");
        // The chunks' storage is reclaimed in the background, once nothing
        // needs them; this call reclaimed none.
        Ok(graph::DeleteBlobResponse { chunks_deleted: 0 })
    }

    /// The metadata of the blob `blob_id` names.
    async fn meta(&self, blob_id: &str) -> Result<graph::BlobMeta, Status> {
        let id = parse_id(blob_id)?;
        let database = Arc::clone(&self.database);
        let record: OwnerRecord =
            tokio::task::spawn_blocking(move || database.read().blob_record(&id))
                .await
                .map_err(|e| Status::internal(format!("blob read did not finish: {e}")))?
                .map_err(db_error_to_status)?
                .ok_or_else(|| Status::not_found(format!("blob {blob_id} not found")))?;
        let blob = &record.blob;
        Ok(graph::BlobMeta {
            blob_id: blob_id.to_owned(),
            total_size: blob.total_size,
            // A blob's chunk count fits a u32.
            chunk_count: blob.chunk_count() as u32,
            chunk_ids: blob.chunks.iter().map(|c| c.digest_hex()).collect(),
        })
    }
}

/// The blob id `text` names; text that names none is a blob that is not
/// there.
fn parse_id(text: &str) -> Result<BlobId, Status> {
    text.parse()
        .map_err(|_| Status::not_found(format!("blob {text} not found")))
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
