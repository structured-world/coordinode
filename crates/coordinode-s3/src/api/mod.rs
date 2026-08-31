//! S3-compatible HTTP API handlers.
//!
//! Implements a subset of the AWS S3 API for CoordiNode BlobStore.
//! CE: GetObject, PutObject, DeleteObject, ListBuckets, ListObjectsV2.

use std::net::SocketAddr;
use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use sha2::Digest;
use tokio::net::TcpListener;
use tracing::{error, info};

use coordinode_core::graph::blob::{self, BlobRef, encode_blob_key};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

use crate::mapping;

/// Start the S3-compatible HTTP server.
pub async fn serve(
    addr: SocketAddr,
    engine: Arc<StorageEngine>,
) -> Result<(), Box<dyn std::error::Error>> {
    let listener = TcpListener::bind(addr).await?;
    info!(addr = %addr, "S3 gateway listening");

    loop {
        let (stream, _) = listener.accept().await?;
        let io = TokioIo::new(stream);
        let engine = engine.clone();

        tokio::spawn(async move {
            if let Err(e) = http1::Builder::new()
                .serve_connection(
                    io,
                    service_fn(move |req| {
                        let engine = engine.clone();
                        async move { handle_request(req, engine).await }
                    }),
                )
                .await
            {
                error!("S3 connection error: {e}");
            }
        });
    }
}

async fn handle_request(
    req: Request<Incoming>,
    engine: Arc<StorageEngine>,
) -> Result<Response<Full<Bytes>>, hyper::Error> {
    let path = req.uri().path().to_string();
    let method = req.method().clone();

    let result = match (&method, path.as_str()) {
        (&Method::GET, "/") => list_buckets(),
        (&Method::PUT, _) if mapping::parse_s3_path(&path).is_some() => {
            put_object(req, &engine, &path).await
        }
        (&Method::GET, _) if mapping::parse_s3_path(&path).is_some() => get_object(&engine, &path),
        (&Method::DELETE, _) if mapping::parse_s3_path(&path).is_some() => {
            delete_object(&engine, &path)
        }
        _ => Ok(xml_response(
            StatusCode::NOT_FOUND,
            "<Error><Code>NoSuchKey</Code><Message>Invalid path</Message></Error>",
        )),
    };

    match result {
        Ok(resp) => Ok(resp),
        Err(msg) => Ok(xml_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("<Error><Code>InternalError</Code><Message>{msg}</Message></Error>"),
        )),
    }
}

/// GET / — list buckets (returns single "blobs" bucket for now).
fn list_buckets() -> Result<Response<Full<Bytes>>, String> {
    let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<ListAllMyBucketsResult>
  <Buckets>
    <Bucket>
      <Name>blobs</Name>
      <CreationDate>2025-01-01T00:00:00Z</CreationDate>
    </Bucket>
  </Buckets>
</ListAllMyBucketsResult>"#;
    Ok(xml_response(StatusCode::OK, xml))
}

/// PUT /<bucket>/<key> — upload object as a blob.
async fn put_object(
    req: Request<Incoming>,
    engine: &StorageEngine,
    path: &str,
) -> Result<Response<Full<Bytes>>, String> {
    let body = req
        .collect()
        .await
        .map_err(|e| format!("body read error: {e}"))?
        .to_bytes();

    if body.is_empty() {
        return Ok(xml_response(
            StatusCode::BAD_REQUEST,
            "<Error><Code>EmptyBody</Code></Error>",
        ));
    }

    let (blob_ref, chunks) = blob::create_blob(&body);

    // Store chunks (with dedup)
    for (chunk_id, chunk_data) in &chunks {
        let key = encode_blob_key(chunk_id);
        if engine
            .get(Partition::Blob, &key)
            .map_err(|e| e.to_string())?
            .is_none()
        {
            engine
                .put(Partition::Blob, &key, chunk_data)
                .map_err(|e| e.to_string())?;
        }
    }

    // Store blob metadata keyed by path
    let meta_key = s3_meta_key(path);
    let meta_value = blob_ref.to_msgpack().map_err(|e| e.to_string())?;
    engine
        .put(Partition::BlobRef, &meta_key, &meta_value)
        .map_err(|e| e.to_string())?;

    // Compute ETag (SHA-256 of full content)
    let etag = format!("\"{}\"", hex::encode(sha2::Sha256::digest(&body)));

    info!(path = %path, size = body.len(), "S3 PutObject");

    let resp = Response::builder()
        .status(StatusCode::OK)
        .header("ETag", etag)
        .body(Full::new(Bytes::new()))
        .map_err(|e| e.to_string())?;
    Ok(resp)
}

/// GET /<bucket>/<key> — download object.
fn get_object(engine: &StorageEngine, path: &str) -> Result<Response<Full<Bytes>>, String> {
    let meta_key = s3_meta_key(path);
    let meta_bytes = engine
        .get(Partition::BlobRef, &meta_key)
        .map_err(|e| e.to_string())?;

    let meta_bytes = match meta_bytes {
        Some(b) => b,
        None => {
            return Ok(xml_response(
                StatusCode::NOT_FOUND,
                "<Error><Code>NoSuchKey</Code></Error>",
            ));
        }
    };

    let blob_ref = BlobRef::from_msgpack(&meta_bytes).map_err(|e| e.to_string())?;

    // Reassemble blob from chunks
    let mut data = Vec::with_capacity(blob_ref.total_size as usize);
    for chunk_id in &blob_ref.chunks {
        let key = encode_blob_key(chunk_id);
        let chunk_data = engine
            .get(Partition::Blob, &key)
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("missing chunk {}", chunk_id.to_hex()))?;
        data.extend_from_slice(&chunk_data);
    }

    let resp = Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/octet-stream")
        .header("Content-Length", data.len())
        .body(Full::new(Bytes::from(data)))
        .map_err(|e| e.to_string())?;
    Ok(resp)
}

/// DELETE /<bucket>/<key> — delete object.
fn delete_object(engine: &StorageEngine, path: &str) -> Result<Response<Full<Bytes>>, String> {
    let meta_key = s3_meta_key(path);
    let meta_bytes = engine
        .get(Partition::BlobRef, &meta_key)
        .map_err(|e| e.to_string())?;

    if let Some(meta_bytes) = meta_bytes {
        let blob_ref = BlobRef::from_msgpack(&meta_bytes).map_err(|e| e.to_string())?;

        // Delete chunks
        for chunk_id in &blob_ref.chunks {
            let key = encode_blob_key(chunk_id);
            engine
                .delete(Partition::Blob, &key)
                .map_err(|e| e.to_string())?;
        }

        // Delete metadata
        engine
            .delete(Partition::BlobRef, &meta_key)
            .map_err(|e| e.to_string())?;

        info!(path = %path, "S3 DeleteObject");
    }

    Response::builder()
        .status(StatusCode::NO_CONTENT)
        .body(Full::new(Bytes::new()))
        .map_err(|e| e.to_string())
}

/// Encode S3 path as BlobRef key: `s3:<path>`.
fn s3_meta_key(path: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(3 + path.len());
    key.extend_from_slice(b"s3:");
    key.extend_from_slice(path.as_bytes());
    key
}

fn xml_response(status: StatusCode, body: &str) -> Response<Full<Bytes>> {
    Response::builder()
        .status(status)
        .header("Content-Type", "application/xml")
        .body(Full::new(Bytes::from(body.to_string())))
        .unwrap_or_else(|_| {
            Response::new(Full::new(Bytes::from(
                "<Error><Code>InternalError</Code></Error>",
            )))
        })
}
