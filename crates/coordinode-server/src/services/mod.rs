pub mod blob;
pub mod cdc;
pub mod cluster;
pub mod cypher;
pub mod error_details;
pub mod graph;
pub mod health;
pub mod operation;
pub mod routing;
pub mod schema;
pub mod session;
pub mod statement;
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

/// Convert a [`DatabaseError`] from a call that is not a statement into a
/// [`tonic::Status`], keeping the categories an operator or client acts on.
///
/// - `CapacityExhausted` and `StorageFull` answer `RESOURCE_EXHAUSTED` with
///   the endpoint, usage and limit attached as structured metadata (gRPC's
///   canonical code for "the resource is full").
/// - A settled or foreign operation answers as its own reason.
/// - Every other variant answers `INTERNAL` with its description.
pub fn db_err_to_status(context: &str, err: DatabaseError) -> Status {
    match &err {
        DatabaseError::CapacityExhausted {
            endpoint_id,
            used_bytes,
            hard_limit_bytes,
        } => {
            return capacity_exhausted_status(context, endpoint_id, *used_bytes, *hard_limit_bytes);
        }
        DatabaseError::StorageFull {
            path,
            available_bytes,
            min_free_bytes,
        } => {
            return error_details::status_with_reason(
                tonic::Code::ResourceExhausted,
                format!("{context}: {err}"),
                error_details::Reason::StorageFull,
                [
                    ("path", path.clone()),
                    ("available_bytes", available_bytes.to_string()),
                    ("min_free_bytes", min_free_bytes.to_string()),
                ],
            );
        }
        DatabaseError::OperationSettled {
            outcome,
            intent_differs,
            ..
        } => {
            return operation::settled_status(
                format!("{context}: {err}"),
                *outcome,
                *intent_differs,
            );
        }
        DatabaseError::ForeignOperation { descriptor } => {
            return operation::foreign_status(format!("{context}: {err}"), *descriptor);
        }
        _ => {}
    }
    Status::internal(format!("{context}: {err}"))
}

/// Same mapping for callers that hold a raw [`StorageError`]: subsystems
/// that call `engine.put` / `engine.delete` directly (blob store, admin
/// RPCs, background workers).
pub fn storage_err_to_status(context: &str, err: StorageError) -> Status {
    db_err_to_status(context, err.into())
}

/// A write refused because an endpoint is at its hard limit.
fn capacity_exhausted_status(
    context: &str,
    endpoint_id: &str,
    used_bytes: u64,
    hard_limit_bytes: u64,
) -> Status {
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
mod db_err_to_status_tests;
