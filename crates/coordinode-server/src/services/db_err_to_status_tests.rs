use super::*;
use tonic::Code;

#[test]
fn capacity_exhausted_maps_to_resource_exhausted_with_metadata() {
    // The whole reason for this helper: capacity errors must
    // propagate as the gRPC-canonical RESOURCE_EXHAUSTED code so
    // clients can pattern-match on it, with structured metadata
    // for the endpoint id and limits.
    let err = DatabaseError::from(StorageError::CapacityExhausted {
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

/// A write refused for disk space reaches the client as
/// RESOURCE_EXHAUSTED / STORAGE_FULL in both shapes it surfaces in,
/// with the numbers in the metadata and a retry floor.
#[test]
fn out_of_space_maps_to_resource_exhausted_storage_full() {
    let refusal = || StorageError::OutOfSpace {
        path: "/data".to_string(),
        available_bytes: 100,
        min_free_bytes: 1 << 30,
    };
    for err in [
        DatabaseError::from(refusal()),
        DatabaseError::from(coordinode_query::executor::runner::ExecutionError::Storage(
            refusal(),
        )),
    ] {
        let status = db_err_to_status("execute", err);
        assert_eq!(status.code(), Code::ResourceExhausted);
        assert!(
            status.message().contains("no space"),
            "{}",
            status.message()
        );
        let details = tonic_types::StatusExt::get_error_details(&status);
        let info = details.error_info().expect("error info");
        assert_eq!(info.reason, "STORAGE_FULL");
        assert_eq!(
            info.metadata.get("available_bytes").map(String::as_str),
            Some("100")
        );
        assert_eq!(info.metadata.get("path").map(String::as_str), Some("/data"));
        assert!(details.retry_info().is_some(), "a retry floor is advised");
    }
}

#[test]
fn other_storage_errors_map_to_internal() {
    let err = DatabaseError::from(StorageError::Io("disk gone".into()));
    let status = db_err_to_status("get_node", err);
    assert_eq!(status.code(), Code::Internal);
    assert!(status.message().contains("get_node"));
    assert!(status.message().contains("disk gone"));
}

#[test]
fn semantic_error_maps_to_internal() {
    // Outside the statement path only resource and operation refusals are
    // classified; a statement's own failures go through the Cypher mapping.
    let err = DatabaseError::Semantic("bad cypher".into());
    let status = db_err_to_status("execute", err);
    assert_eq!(status.code(), Code::Internal);
}
