//! The persisted registry record and its `Partition::Registry` keyspace codec.
//!
//! One [`RegistryEntry`] is stored per consumer id at key
//! `consumer:<consumer_id>` and replicates through the shard's ordering
//! group. A terminated registration keeps its record: the incarnation it
//! carries is what refuses the ended handle and numbers the next one. Its
//! latest heartbeat lives beside it at `heartbeat:<consumer_id>`.

use serde::{Deserialize, Serialize};

use super::source::RetentionSource;
use super::types::{
    ConsumerKind, ConsumerRetentionPolicy, PriorIncarnation, RegistrationState, TerminalReason,
    TopologyScope,
};

/// Key prefix for every registry record within `Partition::Registry`.
pub(crate) const REGISTRY_KEY_PREFIX: &[u8] = b"consumer:";

/// Key prefix of each consumer's heartbeat record, kept apart from its
/// registration record: a heartbeat raises a time by `max` and conditions on
/// nothing, so the batched flush never makes a transition of the same
/// consumer lose its version check, however slow the log is.
pub(crate) const HEARTBEAT_KEY_PREFIX: &[u8] = b"heartbeat:";

/// The longest consumer id a registration accepts: every record key of a
/// consumer (its longest prefix plus the id) must fit the storage's key
/// length, a `u16`.
pub(crate) const MAX_CONSUMER_ID_BYTES: usize = {
    let prefix = if REGISTRY_KEY_PREFIX.len() > HEARTBEAT_KEY_PREFIX.len() {
        REGISTRY_KEY_PREFIX.len()
    } else {
        HEARTBEAT_KEY_PREFIX.len()
    };
    u16::MAX as usize - prefix
};

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
    /// Where this incarnation's checkpoint started.
    pub started_at: u64,
    /// When the registry admitted the read this registration owes (clock
    /// ms): its registration, and for a pinned snapshot each move of its
    /// checkpoint, which is a new view to read, not an acknowledgement of
    /// the old one. A pinned snapshot's work is dated from here, not from
    /// the age of the rows it reads.
    pub obligation_since_ms: u64,
    /// The ended incarnation of the same id this one followed. Boxed: only a
    /// resync carries one, and every transition moves the record by value.
    pub follows: Option<Box<PriorIncarnation>>,
}

impl RegistryEntry {
    /// Whether this registration still holds its protection.
    pub(crate) fn is_live(&self) -> bool {
        matches!(self.state, RegistrationState::Live)
    }

    /// What a registration of the same id following this ended one takes
    /// over from; `None` while this one is live.
    pub(crate) fn as_prior(&self) -> Option<PriorIncarnation> {
        match self.state {
            RegistrationState::Live => None,
            RegistrationState::Terminated {
                reason,
                retained_from,
                ..
            } => Some(PriorIncarnation {
                incarnation: self.incarnation,
                reason,
                acknowledged: self.checkpoint_seqno,
                retained_from,
            }),
        }
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
        // A measure the source could not take ends nothing: the protection
        // stays until the evidence is back. Age and bytes are judged on one
        // observation of the source.
        // Where one observation cannot be taken, a log's measures are judged
        // each on its own (a log whose size is unreadable still has entry
        // times). The MVCC store's two are only ever judged together: asked
        // apart, each would be a separate observation, possibly of states
        // either side of a rebuild.
        let observed = source.observe(self.kind, self.checkpoint_seqno);
        let (produced, bytes) = match &observed {
            Ok(observed) => (Ok(observed.produced_at_ms), Ok(observed.retained_bytes)),
            Err(_) if self.kind.is_seqno_space() => return None,
            Err(_) => (
                source.produced_at_ms(self.kind, self.checkpoint_seqno),
                source.retained_bytes_from(self.kind, self.checkpoint_seqno),
            ),
        };
        // A pinned snapshot's read is outstanding until it is done or moved,
        // new writes or none: its work dates from when it was admitted.
        let (owes, produced) = if self.kind == ConsumerKind::MvccSnapshotPin {
            (true, Ok(Some(self.obligation_since_ms)))
        } else {
            (self.checkpoint_seqno < source.head(self.kind), produced)
        };
        let reason = if owes
            && let Ok(Some(produced)) = produced
            // Same clock rule as above: work stamped after `now_ms` has
            // no age yet.
            && now_ms.saturating_sub(produced) > bounds.max_progress_lag_ms()
        {
            TerminalReason::ProgressLagExceeded
        } else if let Ok(bytes) = bytes
            && bytes > bounds.max_retained_bytes()
        {
            TerminalReason::RetainedBytesExceeded
        } else {
            return None;
        };
        // Checked last, as the ending is about to be written: material the
        // source replaced since it was observed ends nobody.
        if let Ok(observed) = &observed
            && !source.still_current(observed)
        {
            return None;
        }
        Some(reason)
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
        let produced = if self.kind == ConsumerKind::MvccSnapshotPin {
            Some(self.obligation_since_ms)
        } else {
            (self.checkpoint_seqno < source.head(self.kind))
                .then(|| source.produced_at_ms(self.kind, self.checkpoint_seqno).ok())
                .flatten()
                .flatten()
        };
        let lag =
            produced.and_then(|at| at.checked_add(bounds.max_progress_lag_ms())?.checked_add(1));
        match (liveness, lag) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }

    /// This record with the liveness its heartbeat record shows, when that
    /// heartbeat is of this incarnation: an acknowledgement and a heartbeat
    /// are both signs of life, and the later one counts.
    pub(crate) fn with_heartbeat(mut self, heartbeat: Option<Heartbeat>) -> Self {
        if let Some(hb) = heartbeat.filter(|hb| hb.incarnation == self.incarnation) {
            self.last_heartbeat_ts_ms = self.last_heartbeat_ts_ms.max(hb.ts_ms);
        }
        self
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
    prefixed(REGISTRY_KEY_PREFIX, consumer_id)
}

/// Build the `Partition::Registry` key of a consumer's heartbeat record.
pub(crate) fn encode_heartbeat_key(consumer_id: &str) -> Vec<u8> {
    prefixed(HEARTBEAT_KEY_PREFIX, consumer_id)
}

fn prefixed(prefix: &[u8], consumer_id: &str) -> Vec<u8> {
    let mut k = Vec::with_capacity(prefix.len() + consumer_id.len());
    k.extend_from_slice(prefix);
    k.extend_from_slice(consumer_id.as_bytes());
    k
}

/// The latest heartbeat of one consumer: of which registration incarnation,
/// and when. A heartbeat of another incarnation says nothing about the
/// current one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Heartbeat {
    pub incarnation: u64,
    pub ts_ms: u64,
}

impl Heartbeat {
    const LEN: usize = 16;

    /// Fixed 16-byte form: incarnation, then time, both big-endian.
    pub(crate) fn encode(self) -> [u8; Self::LEN] {
        let mut out = [0u8; Self::LEN];
        out[..8].copy_from_slice(&self.incarnation.to_be_bytes());
        out[8..].copy_from_slice(&self.ts_ms.to_be_bytes());
        out
    }

    /// `None` for bytes of any other length.
    pub(crate) fn decode(bytes: &[u8]) -> Option<Self> {
        let bytes: &[u8; Self::LEN] = bytes.try_into().ok()?;
        let (incarnation, ts_ms) = bytes.split_at(8);
        Some(Self {
            incarnation: u64::from_be_bytes(incarnation.try_into().ok()?),
            ts_ms: u64::from_be_bytes(ts_ms.try_into().ok()?),
        })
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
