//! Replay of `w:cache` writes a crashed process accepted and did not drain.

use std::sync::Arc;

use coordinode_core::txn::proposal::{ProposalIdGenerator, ProposalPipeline, RaftProposal};
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_storage::cache::write_buffer::{DRAINED_KEY, RecoveredBuffer, drained_through};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

use crate::DatabaseError;

/// Hand every entry of `buffer` the log has not applied to `pipeline`, in
/// the order they were accepted, before anything else runs on the store.
///
/// The log records the last applied entry of the buffer in the same unit as
/// its effects, so every entry at or below that record already took effect
/// and is skipped, and every one above it has not. An entry bound to a
/// client operation is also decided by its binding, so one settled since
/// applies nothing.
///
/// # Errors
///
/// An entry the log refused, or the record unreadable: the open fails and
/// the buffer files stay for the next attempt.
pub(super) fn replay_cache_buffer(
    engine: &Arc<StorageEngine>,
    oracle: &Arc<TimestampOracle>,
    pipeline: &dyn ProposalPipeline,
    ids: &ProposalIdGenerator,
    buffer: RecoveredBuffer,
) -> Result<(), DatabaseError> {
    let through = engine
        .get(Partition::Schema, DRAINED_KEY)?
        .and_then(|value| drained_through(&value, buffer.id))
        .unwrap_or(0);
    let mut replayed = 0usize;
    for (seq, entry) in buffer.entries {
        if seq <= through {
            continue;
        }
        // A fresh timestamp, held until the entry is in the log: a bound
        // handed out before the crash may have passed the one it was
        // accepted at.
        let (commit_ts, unlogged) = engine.pending_commits().obligate(|| oracle.next().as_raw());
        pipeline
            .propose_and_wait(&RaftProposal {
                id: ids.next(),
                mutations: entry.mutations,
                commit_ts: Timestamp::from_raw(commit_ts),
                start_ts: entry.start_ts,
                bypass_rate_limiter: true,
            })
            .map_err(|e| {
                DatabaseError::Other(format!(
                    "replaying w:cache entry {seq} of buffer {}: {e}",
                    buffer.id
                ))
            })?;
        drop(unlogged);
        replayed += 1;
    }
    tracing::info!(
        buffer = buffer.id,
        through,
        replayed,
        "w:cache writes not drained before the crash are in the log"
    );
    Ok(())
}
