//! The persisted registry record and its `Partition::Registry` keyspace codec.
//!
//! One [`RegistryEntry`] is stored per consumer id at key
//! `consumer:<consumer_id>` and replicates through the shard's ordering
//! group. A terminated registration keeps its record: the incarnation it
//! carries is what refuses the ended handle and numbers the next one.

use serde::{Deserialize, Serialize};

use super::source::RetentionSource;
use super::types::{
    ConsumerKind, ConsumerRetentionPolicy, RegistrationState, TerminalReason, TopologyScope,
};

/// Key prefix for every registry record within `Partition::Registry`.
pub(crate) const REGISTRY_KEY_PREFIX: &[u8] = b"consumer:";

/// Key prefix of records written before registrations carried a retention
/// policy and an incarnation. Those were all short-lived change-stream
/// registrations; a sweep deletes whatever is left of them.
pub(crate) const LEGACY_KEY_PREFIX: &[u8] = b"registry:";

/// The full replicated state of one registration on this shard.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct RegistryEntry {
    pub consumer_id: String,
    /// Counts the registrations this id has had on this shard, from 1.
    pub incarnation: u64,
    pub kind: ConsumerKind,
    pub scope: TopologyScope,
    pub scope_origin: TopologyScope,
    pub retention: ConsumerRetentionPolicy,
    pub state: RegistrationState,
    pub checkpoint_seqno: u64,
    pub last_heartbeat_ts_ms: u64,
}

impl RegistryEntry {
    /// Whether this registration still holds its protection.
    pub(crate) fn is_live(&self) -> bool {
        matches!(self.state, RegistrationState::Live)
    }

    /// The reason a live BOUNDED registration must end at `now_ms`, or `None`
    /// while every declared bound holds. STRICT, and an already ended
    /// registration, never end here.
    ///
    /// Liveness is judged first: a consumer that stopped heartbeating is gone
    /// whatever its progress was.
    pub(crate) fn termination(
        &self,
        now_ms: u64,
        source: &dyn RetentionSource,
    ) -> Option<TerminalReason> {
        if !self.is_live() {
            return None;
        }
        let ConsumerRetentionPolicy::Bounded(bounds) = self.retention else {
            return None;
        };
        if let Some(timeout) = bounds.liveness_timeout_ms() {
            // A clock read before the stored heartbeat (another member's
            // clock, or a step back) is no evidence of absence: no time has
            // passed for this judgement.
            if now_ms.saturating_sub(self.last_heartbeat_ts_ms) > timeout {
                return Some(TerminalReason::LivenessExpired);
            }
        }
        if self.checkpoint_seqno < source.head(self.kind) {
            if let Some(produced) = source.produced_at_ms(self.kind, self.checkpoint_seqno) {
                // Same clock rule as above: work stamped after `now_ms` has
                // no age yet.
                if now_ms.saturating_sub(produced) > bounds.max_progress_lag_ms() {
                    return Some(TerminalReason::ProgressLagExceeded);
                }
            }
        }
        if let Some(bytes) = source.retained_bytes_from(self.kind, self.checkpoint_seqno) {
            if bytes > bounds.max_retained_bytes() {
                return Some(TerminalReason::RetainedBytesExceeded);
            }
        }
        None
    }

    /// The earliest clock ms at which [`Self::termination`] can change its
    /// answer without anything else happening, or `None` when only a change
    /// to the source or the registry can.
    pub(crate) fn next_deadline_ms(&self, source: &dyn RetentionSource) -> Option<u64> {
        if !self.is_live() {
            return None;
        }
        let ConsumerRetentionPolicy::Bounded(bounds) = self.retention else {
            return None;
        };
        let liveness = bounds
            .liveness_timeout_ms()
            .and_then(|t| self.last_heartbeat_ts_ms.checked_add(t)?.checked_add(1));
        let lag = (self.checkpoint_seqno < source.head(self.kind))
            .then(|| source.produced_at_ms(self.kind, self.checkpoint_seqno))
            .flatten()
            .and_then(|at| at.checked_add(bounds.max_progress_lag_ms())?.checked_add(1));
        match (liveness, lag) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }

    /// Serialize to the replicated msgpack wire form.
    pub(crate) fn encode(&self) -> Result<Vec<u8>, rmp_serde::encode::Error> {
        rmp_serde::to_vec(self)
    }

    /// Deserialize from the replicated msgpack wire form.
    pub(crate) fn decode(bytes: &[u8]) -> Result<Self, rmp_serde::decode::Error> {
        rmp_serde::from_slice(bytes)
    }
}

/// Build the `Partition::Registry` key for a consumer id.
pub(crate) fn encode_registry_key(consumer_id: &str) -> Vec<u8> {
    let mut k = Vec::with_capacity(REGISTRY_KEY_PREFIX.len() + consumer_id.len());
    k.extend_from_slice(REGISTRY_KEY_PREFIX);
    k.extend_from_slice(consumer_id.as_bytes());
    k
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
