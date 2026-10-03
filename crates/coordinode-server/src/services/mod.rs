pub mod blob;
pub mod cdc;
pub mod cluster;
pub mod cypher;
pub mod error_details;
pub mod graph;
pub mod health;
pub mod schema;
pub mod session;
pub mod text;
pub mod vector;

use coordinode_embed::DatabaseError;
use coordinode_storage::error::StorageError;
use tonic::Status;

/// Run a synchronous database call from a request handler without holding a
/// runtime worker: on a multi-thread runtime the worker hands its queue to
/// another thread while `f` blocks. A call blocks on its commit through Raft,
/// and on locks another call holds while it waits for its own commit; done on
/// a worker, enough such calls in flight take every worker, and the Raft tasks
/// those commits wait for never run. On a current-thread runtime there is no
/// other worker to hand to, and `f` runs in place.
pub(crate) fn blocking<R>(f: impl FnOnce() -> R) -> R {
    match tokio::runtime::Handle::try_current().map(|h| h.runtime_flavor()) {
        Ok(tokio::runtime::RuntimeFlavor::MultiThread) => tokio::task::block_in_place(f),
        _ => f(),
    }
}

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
    // Drill into both `Storage(...)` and `Execution(...)` variants —
    // a capacity-exhausted write can surface through either path.
    // Direct engine writes (e.g. blob upload) raise
    // `DatabaseError::Storage(CapacityExhausted)`. Cypher writes go
    // through the query executor's proposal pipeline, which lifts the
    // `ProposalError::CapacityExhausted` into
    // `ExecutionError::Storage(StorageError::CapacityExhausted)`, then
    // gets wrapped as `DatabaseError::Execution(...)`. Both shapes must
    // map to the same `RESOURCE_EXHAUSTED` Status.
    if let Some(inner) = capacity_exhausted_inner(&err) {
        return capacity_exhausted_status(context, inner);
    }
    if let Some(inner) = out_of_space_inner(&err) {
        return out_of_space_status(context, inner);
    }
    Status::internal(format!("{context}: {err}"))
}

/// Locate a `StorageError::OutOfSpace` in either shape a refused write
/// surfaces in: a direct engine write, or the proposal pipeline's carry
/// through the executor.
fn out_of_space_inner(err: &DatabaseError) -> Option<&StorageError> {
    let storage = match err {
        DatabaseError::Storage(s) => s,
        DatabaseError::Execution(coordinode_query::executor::runner::ExecutionError::Storage(
            s,
        )) => s,
        _ => return None,
    };
    matches!(storage, StorageError::OutOfSpace { .. }).then_some(storage)
}

/// A write refused because the disk is below its free-space reserve:
/// RESOURCE_EXHAUSTED with reason STORAGE_FULL. Reads go on; the write is
/// retried once space is freed. Pre: `err` IS `StorageError::OutOfSpace`.
fn out_of_space_status(context: &str, err: &StorageError) -> Status {
    let StorageError::OutOfSpace {
        path,
        available_bytes,
        min_free_bytes,
    } = err
    else {
        return Status::internal(format!("{context}: {err}"));
    };
    error_details::status_with_reason(
        tonic::Code::ResourceExhausted,
        format!("{context}: {err}"),
        error_details::Reason::StorageFull,
        [
            ("path", path.clone()),
            ("available_bytes", available_bytes.to_string()),
            ("min_free_bytes", min_free_bytes.to_string()),
        ],
    )
}

/// Locate a `StorageError::CapacityExhausted` anywhere in the
/// `DatabaseError` tree. Returns the borrowed inner variant when
/// found, `None` otherwise.
fn capacity_exhausted_inner(err: &DatabaseError) -> Option<&StorageError> {
    match err {
        DatabaseError::Storage(s) if matches!(s, StorageError::CapacityExhausted { .. }) => Some(s),
        DatabaseError::Execution(exec) => {
            // `ExecutionError::Storage(StorageError)` is the carry
            // path from the proposal pipeline mapping in the
            // executor. Pattern-match via the Display chain is
            // fragile; use the typed accessor.
            if let coordinode_query::executor::runner::ExecutionError::Storage(s) = exec {
                if matches!(s, StorageError::CapacityExhausted { .. }) {
                    return Some(s);
                }
            }
            None
        }
        _ => None,
    }
}

/// Same mapping for callers that hold a raw [`StorageError`] (no
/// `DatabaseError` wrapper) — typically subsystems that call
/// `engine.put` / `engine.delete` directly (blob store, admin RPCs,
/// background workers). Behaviour matches [`db_err_to_status`] for
/// the capacity case.
pub fn storage_err_to_status(context: &str, err: StorageError) -> Status {
    if matches!(err, StorageError::CapacityExhausted { .. }) {
        return capacity_exhausted_status(context, &err);
    }
    if matches!(err, StorageError::OutOfSpace { .. }) {
        return out_of_space_status(context, &err);
    }
    Status::internal(format!("{context}: {err}"))
}

/// Shared body for the capacity-exhausted → `RESOURCE_EXHAUSTED`
/// mapping. Pre: `err` IS `StorageError::CapacityExhausted`.
fn capacity_exhausted_status(context: &str, err: &StorageError) -> Status {
    let (endpoint_id, used_bytes, hard_limit_bytes) = match err {
        StorageError::CapacityExhausted {
            endpoint_id,
            used_bytes,
            hard_limit_bytes,
        } => (endpoint_id.as_str(), *used_bytes, *hard_limit_bytes),
        // Unreachable in practice — callers gate on the variant
        // before invoking this helper. Defensively map to Internal.
        _ => return Status::internal(format!("{context}: {err}")),
    };
    let msg = format!(
        "{context}: endpoint {endpoint_id:?} capacity exhausted \
         (used={used_bytes}, hard_limit={hard_limit_bytes})"
    );
    let mut status = error_details::status_with_reason(
        tonic::Code::ResourceExhausted,
        msg,
        error_details::Reason::CapacityExhausted,
        [
            ("endpoint_id", endpoint_id.to_string()),
            ("used_bytes", used_bytes.to_string()),
            ("hard_limit_bytes", hard_limit_bytes.to_string()),
        ],
    );
    // The same three values also stay in flat trailer keys, where they have
    // been since before the canonical details existed. Clients already read
    // them, and a published wire surface is not something to withdraw as a
    // side effect of adding a better one; new callers should prefer the
    // ErrorInfo metadata above.
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
    status
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
            DatabaseError::Storage(refusal()),
            DatabaseError::Execution(coordinode_query::executor::runner::ExecutionError::Storage(
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
