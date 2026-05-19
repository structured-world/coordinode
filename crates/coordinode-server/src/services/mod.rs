pub mod blob;
pub mod cdc;
pub mod cluster;
pub mod cypher;
pub mod graph;
pub mod health;
pub mod schema;
pub mod text;
pub mod vector;

use coordinode_embed::DatabaseError;
use coordinode_storage::error::StorageError;
use tonic::Status;

/// Convert a [`DatabaseError`] from the embedded database into a
/// [`tonic::Status`] preserving operator-actionable error categories.
///
/// Mapping:
/// - `Storage(CapacityExhausted { .. })` → `Status::resource_exhausted`
///   with the endpoint id, current usage, and limit attached as
///   structured metadata so the client can surface a precise error.
///   This is the gRPC canonical code for "the resource is full"
///   (gRPC standard ResourceExhausted).
/// - Every other variant → `Status::internal` with a stringified
///   description (current legacy behaviour; categorising the rest
///   is a follow-up task).
///
/// Callers that previously did
/// `.map_err(|e| Status::internal(format!("op: {e}")))` should switch
/// to `.map_err(|e| db_err_to_status("op", e))` so that capacity
/// errors propagate as a typed `RESOURCE_EXHAUSTED` to the client.
pub fn db_err_to_status(context: &str, err: DatabaseError) -> Status {
    if let DatabaseError::Storage(StorageError::CapacityExhausted {
        ref endpoint_id,
        used_bytes,
        hard_limit_bytes,
    }) = err
    {
        let msg = format!(
            "{context}: endpoint {endpoint_id:?} capacity exhausted \
             (used={used_bytes}, hard_limit={hard_limit_bytes})"
        );
        let mut status = Status::resource_exhausted(msg);
        // Attach structured metadata so clients can parse the
        // endpoint id / values without screen-scraping the message.
        let meta = status.metadata_mut();
        if let Ok(v) = endpoint_id.parse() {
            meta.insert("endpoint-id", v);
        }
        if let Ok(v) = used_bytes.to_string().parse() {
            meta.insert("used-bytes", v);
        }
        if let Ok(v) = hard_limit_bytes.to_string().parse() {
            meta.insert("hard-limit-bytes", v);
        }
        return status;
    }
    Status::internal(format!("{context}: {err}"))
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod db_err_to_status_tests {
    use super::*;
    use tonic::Code;

    #[test]
    fn capacity_exhausted_maps_to_resource_exhausted_with_metadata() {
        // The whole reason for this helper: capacity errors must
        // propagate as the gRPC-canonical RESOURCE_EXHAUSTED code so
        // clients can pattern-match on it, with structured metadata
        // for the endpoint id and limits.
        let err = DatabaseError::Storage(StorageError::CapacityExhausted {
            endpoint_id: "ep-hot".to_string(),
            used_bytes: 5_000,
            hard_limit_bytes: 4_000,
        });
        let status = db_err_to_status("create_node", err);
        assert_eq!(status.code(), Code::ResourceExhausted);
        let meta = status.metadata();
        assert_eq!(
            meta.get("endpoint-id").map(|v| v.to_str().expect("ascii")),
            Some("ep-hot"),
        );
        assert_eq!(
            meta.get("used-bytes").map(|v| v.to_str().expect("ascii")),
            Some("5000"),
        );
        assert_eq!(
            meta.get("hard-limit-bytes")
                .map(|v| v.to_str().expect("ascii")),
            Some("4000"),
        );
        assert!(status.message().contains("create_node"));
        assert!(status.message().contains("ep-hot"));
    }

    #[test]
    fn other_storage_errors_map_to_internal() {
        let err = DatabaseError::Storage(StorageError::Io("disk gone".into()));
        let status = db_err_to_status("get_node", err);
        assert_eq!(status.code(), Code::Internal);
        assert!(status.message().contains("get_node"));
        assert!(status.message().contains("disk gone"));
    }

    #[test]
    fn semantic_error_maps_to_internal() {
        // Non-storage variants fall through to Internal — the
        // helper's only special case is capacity. Other categories
        // are a future-task to classify (parse → invalid_argument,
        // plan → invalid_argument, etc.).
        let err = DatabaseError::Semantic("bad cypher".into());
        let status = db_err_to_status("execute", err);
        assert_eq!(status.code(), Code::Internal);
    }
}
