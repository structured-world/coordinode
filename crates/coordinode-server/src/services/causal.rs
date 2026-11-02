//! Causal bounds and receipts on the wire: what a read concern names as its
//! lower bound, and what a finished statement tells its caller for the next.

use std::time::Duration;

use tonic::Status;

use coordinode_core::group::{DeploymentId, GroupId};
use coordinode_core::txn::read_concern::{CausalReceipt, HistoryId, HistoryPosition};

use crate::proto::replication;

/// The causal position a request names, `None` when it names none.
pub(crate) fn position_from_proto(
    field: &str,
    position: Option<&replication::HistoryPosition>,
) -> Result<Option<HistoryPosition>, Status> {
    let Some(position) = position else {
        return Ok(None);
    };
    let history = position.history.as_ref().ok_or_else(|| {
        super::error_details::invalid_field(field, "a causal position names its history")
    })?;
    let deployment = <[u8; 16]>::try_from(history.deployment.as_slice()).map_err(|_| {
        super::error_details::invalid_field(
            format!("{field}.history.deployment"),
            "a deployment identity is 16 bytes",
        )
    })?;
    Ok(Some(HistoryPosition {
        history: HistoryId {
            deployment: DeploymentId(deployment),
            group: GroupId(history.group),
        },
        index: position.index,
    }))
}

/// A position as the wire carries it.
pub(crate) fn position_to_proto(position: HistoryPosition) -> replication::HistoryPosition {
    replication::HistoryPosition {
        history: Some(replication::HistoryId {
            deployment: position.history.deployment.0.to_vec(),
            group: position.history.group.raw(),
        }),
        index: position.index,
    }
}

/// The wait a request allows for a causal bound; zero allows none.
pub(crate) fn wait_from_proto(wait_ms: u32) -> Option<Duration> {
    (wait_ms > 0).then(|| Duration::from_millis(u64::from(wait_ms)))
}

/// The receipt of a statement that committed at `commit_ts` with its entry
/// at `applied_index`, or read at `read_ts`, on a member of `history`.
///
/// A write is answered by its own commit. On a replicated member a write
/// with a timestamp but no entry was answered before it reached the log (a
/// volatile journal, fewer acknowledgements than a majority): it may still be
/// lost, so it gives no operation time a later read could wait for. A read
/// is answered by the timestamp its view was taken at.
pub(crate) fn receipt(
    commit_ts: Option<u64>,
    applied_index: Option<u64>,
    read_ts: Option<u64>,
    history: Option<HistoryId>,
) -> CausalReceipt {
    let position = history
        .zip(applied_index.filter(|index| *index > 0))
        .map(|(history, index)| HistoryPosition { history, index });
    let operation_time = match (commit_ts, position, history) {
        (Some(ts), Some(_), _) | (Some(ts), None, None) => ts,
        (Some(_), None, Some(_)) => 0,
        (None, _, _) => read_ts.unwrap_or(0),
    };
    CausalReceipt {
        operation_time,
        position,
    }
}

/// A receipt as the wire carries it.
pub(crate) fn receipt_to_proto(receipt: CausalReceipt) -> replication::CausalReceipt {
    replication::CausalReceipt {
        operation_time: receipt.operation_time,
        position: receipt.position.map(position_to_proto),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
