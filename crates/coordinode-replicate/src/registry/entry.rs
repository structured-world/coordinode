//! The persisted registry record and its `Partition::Registry` keyspace codec.
//!
//! One [`RegistryEntry`] is stored per consumer at key `registry:<consumer_id>`
//! and replicates through the shard's Raft group. `shard_floor` is
//! `min(checkpoint_seqno)` over the decoded entries in this keyspace.

use serde::{Deserialize, Serialize};

use super::types::{ConsumerKind, TopologyScope};

/// Key prefix for every registry record within `Partition::Registry`.
pub(crate) const REGISTRY_KEY_PREFIX: &[u8] = b"registry:";

/// The full replicated state of one registration on this shard.
///
/// Extends [`ConsumerRegistration`](super::ConsumerRegistration) with the
/// mutable progress (`checkpoint_seqno`, `last_heartbeat_ts_ms`) and
/// `scope_origin` (where the registration was originally placed, for ops
/// clarity when a broader-scope consumer is seen on this shard).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct RegistryEntry {
    pub consumer_id: String,
    pub kind: ConsumerKind,
    pub scope: TopologyScope,
    pub scope_origin: TopologyScope,
    pub checkpoint_seqno: u64,
    pub last_heartbeat_ts_ms: u64,
    pub ttl_ms: u64,
}

impl RegistryEntry {
    /// `true` when `now_ms - last_heartbeat_ts_ms > ttl_ms` and `ttl_ms != 0`.
    /// `ttl_ms == 0` is a persistent registration that never auto-evicts.
    pub(crate) fn is_expired(&self, now_ms: u64) -> bool {
        self.ttl_ms != 0 && now_ms.saturating_sub(self.last_heartbeat_ts_ms) > self.ttl_ms
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
