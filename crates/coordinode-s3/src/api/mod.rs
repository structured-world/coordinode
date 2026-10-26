//! S3-compatible HTTP API handlers.
//!
//! Implements a subset of the AWS S3 API for CoordiNode BlobStore.
//! CE: GetObject, PutObject, DeleteObject, ListBuckets, ListObjectsV2.
//!
//! Each object key is an owner of its bytes. Putting an object replaces what
//! its key held and releases it in the same commit; deleting it releases it.
//! Bytes another object shares stay, and storage nothing needs any more is
//! reclaimed in the background.

use std::net::SocketAddr;

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

use coordinode_core::graph::blob::{SharingScope, s3_owner_key};
use coordinode_storage::blob::{BlobStore, OwnerWrite};

use crate::mapping;

/// Start the S3-compatible HTTP server over `store`.
pub async fn serve(addr: SocketAddr, store: BlobStore) -> Result<(), Box<dyn std::error::Error>> {
    let listener = TcpListener::bind(addr).await?;
    info!(addr = %addr, "S3 gateway listening");

    loop {
        let (stream, _) = listener.accept().await?;
        let io = TokioIo::new(stream);
        let store = store.clone();

        tokio::spawn(async move {
            if let Err(e) = http1::Builder::new()
                .serve_connection(
                    io,
                    service_fn(move |req| {
                        let store = store.clone();
                        async move { handle_request(req, store).await }
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
    store: BlobStore,
) -> Result<Response<Full<Bytes>>, hyper::Error> {
    let path = req.uri().path().to_string();
    let method = req.method().clone();

    let result = match (&method, path.as_str()) {
        (&Method::GET, "/") => list_buckets(),
        (&Method::PUT, _) if mapping::parse_s3_path(&path).is_some() => match req.collect().await {
            Ok(body) => put_object(&store, &path, body.to_bytes()).await,
            Err(e) => Err(format!("body read error: {e}")),
        },
        (&Method::GET, _) if mapping::parse_s3_path(&path).is_some() => {
            get_object(&store, &path).await
        }
        (&Method::DELETE, _) if mapping::parse_s3_path(&path).is_some() => {
            delete_object(&store, &path).await
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

/// `PUT /<bucket>/<key>`: store `body` as the object at `path`, replacing
/// what the key held. The object is durable on this gateway's store before
/// the answer.
///
/// Interim blob payload path: the storage substrate's extent/stream write,
/// with the bucket's write concern and protection, replaces this one.
async fn put_object(
    store: &BlobStore,
    path: &str,
    body: Bytes,
) -> Result<Response<Full<Bytes>>, String> {
    if body.is_empty() {
        return Ok(xml_response(
            StatusCode::BAD_REQUEST,
            "<Error><Code>EmptyBody</Code></Error>",
        ));
    }

    // Compute ETag (SHA-256 of full content)
    let etag = format!("\"{}\"", hex::encode(sha2::Sha256::digest(&body)));
    let size = body.len();

    // Flushing blocks, so it runs off the async runtime.
    let writer = store.clone();
    let key = s3_owner_key(path);
    tokio::task::spawn_blocking(move || {
        writer.put(&key, SharingScope::STORE, &body, OwnerWrite::Replace, None)
    })
    .await
    .map_err(|e| format!("object write did not finish: {e}"))?
    .map_err(|e| e.to_string())?;

    info!(path = %path, size, "S3 PutObject");

    let resp = Response::builder()
        .status(StatusCode::OK)
        .header("ETag", etag)
        .body(Full::new(Bytes::new()))
        .map_err(|e| e.to_string())?;
    Ok(resp)
}

/// `GET /<bucket>/<key>`: the object at `path`, read whole at one snapshot.
async fn get_object(store: &BlobStore, path: &str) -> Result<Response<Full<Bytes>>, String> {
    let reader = store.clone();
    let key = s3_owner_key(path);
    let data = tokio::task::spawn_blocking(move || {
        let Some(opened) = reader.open(&key).map_err(|e| e.to_string())? else {
            return Ok(None);
        };
        let blob = &opened.record().blob;
        // A capacity hint only: an object too large for this platform's
        // addresses fails when it is read, not here.
        let mut data = Vec::with_capacity(usize::try_from(blob.total_size).unwrap_or(0));
        for index in 0..blob.chunk_count() {
            data.extend_from_slice(&opened.chunk(index).map_err(|e| e.to_string())?);
        }
        Ok::<_, String>(Some(data))
    })
    .await
    .map_err(|e| format!("object read did not finish: {e}"))??;

    let Some(data) = data else {
        return Ok(xml_response(
            StatusCode::NOT_FOUND,
            "<Error><Code>NoSuchKey</Code></Error>",
        ));
    };
    let resp = Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/octet-stream")
        .header("Content-Length", data.len())
        .body(Full::new(Bytes::from(data)))
        .map_err(|e| e.to_string())?;
    Ok(resp)
}

/// `DELETE /<bucket>/<key>`: delete the object at `path`, durably before the
/// answer. Deleting a key that holds nothing succeeds, as in S3.
async fn delete_object(store: &BlobStore, path: &str) -> Result<Response<Full<Bytes>>, String> {
    let writer = store.clone();
    let key = s3_owner_key(path);
    let deleted = tokio::task::spawn_blocking(move || writer.delete(&key))
        .await
        .map_err(|e| format!("object delete did not finish: {e}"))?
        .map_err(|e| e.to_string())?;
    if deleted.is_some() {
        info!(path = %path, "S3 DeleteObject");
    }

    Response::builder()
        .status(StatusCode::NO_CONTENT)
        .body(Full::new(Bytes::new()))
        .map_err(|e| e.to_string())
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

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
