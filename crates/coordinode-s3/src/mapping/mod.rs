//! S3 to graph concept mapping.
//!
//! Maps S3 concepts to CoordiNode graph storage:
//!
//! | S3 Concept     | Graph Concept          |
//! |----------------|------------------------|
//! | Bucket         | Namespace / Label      |
//! | Object key     | Node ID + property     |
//! | Object data    | BlobStore chunks       |
//! | ETag           | SHA-256 content hash   |
//! | StorageClass   | Tier (hot/warm/cold)   |  (EE only)
//! | Versioning     | MVCC snapshot          |  (EE only)

/// Parse an S3-style path into bucket and key components.
///
/// Path format: `/<bucket>/<key>` where key may contain `/`.
pub(crate) fn parse_s3_path(path: &str) -> Option<(&str, &str)> {
    let trimmed = path.strip_prefix('/')?;
    let slash = trimmed.find('/')?;
    let bucket = &trimmed[..slash];
    let key = &trimmed[slash + 1..];
    if bucket.is_empty() || key.is_empty() {
        return None;
    }
    Some((bucket, key))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
