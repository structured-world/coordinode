use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, Streaming};
use tracing::info;

use sha2::Digest;

use coordinode_core::graph::blob::{self, BlobRef, encode_blob_key};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

use crate::proto::graph;
use crate::services::error_details::{Reason, status_with_reason};
use crate::services::storage_err_to_status;

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
    engine: std::sync::Arc<StorageEngine>,
    node: Option<std::sync::Arc<coordinode_raft::cluster::RaftNode>>,
}

impl BlobServiceImpl {
    /// A service over `engine`, standalone.
    pub fn new(engine: std::sync::Arc<StorageEngine>) -> Self {
        Self { engine, node: None }
    }

    /// Refuse writes while the group `node` belongs to has other members.
    #[must_use]
    pub fn with_node(mut self, node: std::sync::Arc<coordinode_raft::cluster::RaftNode>) -> Self {
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

/// Encode a blob_id from a BlobRef (first chunk hash used as identifier).
fn blob_id_from_ref(blob_ref: &BlobRef) -> String {
    if blob_ref.chunks.is_empty() {
        return String::new();
    }
    // Use hash of all chunk IDs concatenated for a unique identifier
    let mut hasher = sha2::Sha256::new();
    for chunk_id in &blob_ref.chunks {
        sha2::Digest::update(&mut hasher, chunk_id.as_bytes());
    }
    let hash = sha2::Digest::finalize(hasher);
    hex::encode(hash)
}

/// Storage key for blob metadata: `blobmeta:<blob_id_hex>`
fn encode_blobmeta_key(blob_id: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(9 + blob_id.len());
    key.extend_from_slice(b"blobmeta:");
    key.extend_from_slice(blob_id.as_bytes());
    key
}

#[tonic::async_trait]
impl graph::blob_service_server::BlobService for BlobServiceImpl {
    type DownloadBlobStream = ReceiverStream<Result<graph::BlobChunk, Status>>;

    async fn upload_blob(
        &self,
        request: Request<Streaming<graph::BlobChunk>>,
    ) -> Result<Response<graph::UploadBlobResponse>, Status> {
        let mut stream = request.into_inner();

        // Collect all chunk data from the stream
        let mut data = Vec::new();
        while let Some(chunk) = stream
            .message()
            .await
            .map_err(|e| Status::internal(format!("stream error: {e}")))?
        {
            data.extend_from_slice(&chunk.data);
        }

        if data.is_empty() {
            return Err(Status::invalid_argument("empty blob upload"));
        }
        self.refuse_in_a_group()?;

        // Create blob chunks
        let (blob_ref, chunks) = blob::create_blob(&data);
        let blob_id = blob_id_from_ref(&blob_ref);
        let meta_key = encode_blobmeta_key(&blob_id);
        let meta_value = blob_ref
            .to_msgpack()
            .map_err(|e| Status::internal(format!("serialize error: {e}")))?;

        // The chunks are durable before the metadata naming them is written,
        // and both before the answer: an upload answered is one a crash keeps.
        // Flushing blocks, so it runs off the async runtime.
        let engine = std::sync::Arc::clone(&self.engine);
        let held = tokio::task::spawn_blocking(move || {
            let held = coordinode_storage::blob::put_chunks_durably(&engine, &chunks)?;
            coordinode_storage::blob::put_meta_durably(&engine, &meta_key, &meta_value)?;
            Ok::<_, coordinode_storage::error::StorageError>(held)
        })
        .await
        .map_err(|e| Status::internal(format!("blob write did not finish: {e}")))?
        .map_err(|e| storage_err_to_status("blob storage", e))?;
        // At most one per chunk, and a blob's chunk count fits a u32.
        let dedup_count = u32::try_from(held).unwrap_or(u32::MAX);

        let chunk_ids: Vec<String> = blob_ref.chunks.iter().map(|id| id.to_hex()).collect();

        info!(
            blob_id = %blob_id,
            total_size = blob_ref.total_size,
            chunks = blob_ref.chunk_count(),
            dedup = dedup_count,
            "blob uploaded"
        );

        Ok(Response::new(graph::UploadBlobResponse {
            blob_id,
            chunk_ids,
            total_size: blob_ref.total_size,
            dedup_chunks: dedup_count,
        }))
    }

    async fn download_blob(
        &self,
        request: Request<graph::DownloadBlobRequest>,
    ) -> Result<Response<Self::DownloadBlobStream>, Status> {
        let blob_id = request.into_inner().blob_id;

        // Load blob metadata
        let meta_key = encode_blobmeta_key(&blob_id);
        let meta_bytes = self
            .engine
            .get(Partition::BlobRef, &meta_key)
            .map_err(|e| storage_err_to_status("blob storage", e))?
            .ok_or_else(|| Status::not_found(format!("blob {blob_id} not found")))?;

        let blob_ref = BlobRef::from_msgpack(&meta_bytes)
            .map_err(|e| Status::internal(format!("deserialize error: {e}")))?;

        // Stream chunks back to client
        let (tx, rx) = mpsc::channel(4);
        let engine = self.engine.clone();

        tokio::spawn(async move {
            for (seq, chunk_id) in blob_ref.chunks.iter().enumerate() {
                let key = encode_blob_key(chunk_id);
                match engine.get(Partition::Blob, &key) {
                    Ok(Some(data)) => {
                        let chunk = graph::BlobChunk {
                            data: data.to_vec(),
                            sequence: seq as u32,
                        };
                        if tx.send(Ok(chunk)).await.is_err() {
                            break; // Client disconnected
                        }
                    }
                    Ok(None) => {
                        let _ = tx
                            .send(Err(Status::internal(format!(
                                "chunk {} missing",
                                chunk_id.to_hex()
                            ))))
                            .await;
                        break;
                    }
                    Err(e) => {
                        let _ = tx.send(Err(storage_err_to_status("blob storage", e))).await;
                        break;
                    }
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn delete_blob(
        &self,
        request: Request<graph::DeleteBlobRequest>,
    ) -> Result<Response<graph::DeleteBlobResponse>, Status> {
        let blob_id = request.into_inner().blob_id;
        // A delete on one member of a group would leave the others' copy.
        self.refuse_in_a_group()?;

        let meta_key = encode_blobmeta_key(&blob_id);
        if self
            .engine
            .get(Partition::BlobRef, &meta_key)
            .map_err(|e| storage_err_to_status("blob storage", e))?
            .is_none()
        {
            return Err(Status::not_found(format!("blob {blob_id} not found")));
        }

        // Only the metadata goes. Chunks are content-addressed and shared
        // with every other blob, object or property holding the same bytes,
        // so one owner cannot tell whether a chunk is still referenced; and
        // an upload racing this delete skips writing a chunk it sees
        // present. Unreferenced chunks are reclaimed by chunk GC instead,
        // so no chunk is deleted here. The removal is durable before the
        // answer, so a deleted blob does not come back after a crash.
        let engine = std::sync::Arc::clone(&self.engine);
        tokio::task::spawn_blocking(move || {
            coordinode_storage::blob::delete_meta_durably(&engine, &meta_key)
        })
        .await
        .map_err(|e| Status::internal(format!("blob delete did not finish: {e}")))?
        .map_err(|e| storage_err_to_status("blob storage", e))?;

        info!(blob_id = %blob_id, "blob deleted");

        Ok(Response::new(graph::DeleteBlobResponse {
            chunks_deleted: 0,
        }))
    }

    async fn get_blob_meta(
        &self,
        request: Request<graph::GetBlobMetaRequest>,
    ) -> Result<Response<graph::BlobMeta>, Status> {
        let blob_id = request.into_inner().blob_id;

        let meta_key = encode_blobmeta_key(&blob_id);
        let meta_bytes = self
            .engine
            .get(Partition::BlobRef, &meta_key)
            .map_err(|e| storage_err_to_status("blob storage", e))?
            .ok_or_else(|| Status::not_found(format!("blob {blob_id} not found")))?;

        let blob_ref = BlobRef::from_msgpack(&meta_bytes)
            .map_err(|e| Status::internal(format!("deserialize error: {e}")))?;

        Ok(Response::new(graph::BlobMeta {
            blob_id,
            total_size: blob_ref.total_size,
            chunk_count: blob_ref.chunk_count() as u32,
            chunk_ids: blob_ref.chunks.iter().map(|id| id.to_hex()).collect(),
        }))
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
