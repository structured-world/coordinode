use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, Streaming};
use tracing::info;

use sha2::Digest;

use coordinode_core::graph::blob::{self, encode_blob_key, BlobRef};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

use crate::proto::graph;
use crate::services::storage_err_to_status;

/// gRPC BlobService implementation.
///
/// All state is in CoordiNode storage — this service is stateless and safe
/// to run on multiple nodes behind a load balancer.
pub struct BlobServiceImpl {
    engine: std::sync::Arc<StorageEngine>,
}

impl BlobServiceImpl {
    pub fn new(engine: std::sync::Arc<StorageEngine>) -> Self {
        Self { engine }
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

        // Create blob chunks
        let (blob_ref, chunks) = blob::create_blob(&data);
        let blob_id = blob_id_from_ref(&blob_ref);

        // Store chunks (deduplicated — only write if not already present)
        let mut dedup_count = 0u32;
        for (chunk_id, chunk_data) in &chunks {
            let key = encode_blob_key(chunk_id);
            match self.engine.get(Partition::Blob, &key) {
                Ok(Some(_)) => {
                    dedup_count += 1; // Already exists
                }
                Ok(None) => {
                    self.engine
                        .put(Partition::Blob, &key, chunk_data)
                        .map_err(|e| storage_err_to_status("blob storage", e))?;
                }
                Err(e) => {
                    return Err(storage_err_to_status("blob storage", e));
                }
            }
        }

        // Store blob metadata (BlobRef as MessagePack under blobmeta: key)
        let meta_key = encode_blobmeta_key(&blob_id);
        let meta_value = blob_ref
            .to_msgpack()
            .map_err(|e| Status::internal(format!("serialize error: {e}")))?;
        self.engine
            .put(Partition::BlobRef, &meta_key, &meta_value)
            .map_err(|e| storage_err_to_status("blob storage", e))?;

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

        // Load blob metadata
        let meta_key = encode_blobmeta_key(&blob_id);
        let meta_bytes = self
            .engine
            .get(Partition::BlobRef, &meta_key)
            .map_err(|e| storage_err_to_status("blob storage", e))?
            .ok_or_else(|| Status::not_found(format!("blob {blob_id} not found")))?;

        let blob_ref = BlobRef::from_msgpack(&meta_bytes)
            .map_err(|e| Status::internal(format!("deserialize error: {e}")))?;

        // Delete each chunk (in Phase 2 with replication, this needs reference counting)
        let mut deleted = 0u32;
        for chunk_id in &blob_ref.chunks {
            let key = encode_blob_key(chunk_id);
            self.engine
                .delete(Partition::Blob, &key)
                .map_err(|e| storage_err_to_status("blob storage", e))?;
            deleted += 1;
        }

        // Delete blob metadata
        self.engine
            .delete(Partition::BlobRef, &meta_key)
            .map_err(|e| storage_err_to_status("blob storage", e))?;

        info!(blob_id = %blob_id, chunks_deleted = deleted, "blob deleted");

        Ok(Response::new(graph::DeleteBlobResponse {
            chunks_deleted: deleted,
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
