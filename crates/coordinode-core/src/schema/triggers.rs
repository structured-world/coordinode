//! Trigger schema partition layout (the trigger architecture).
//!
//! Triggers are stored under two key families in the schema partition:
//!
//! - `schema:trigger:<name>` → MessagePack-encoded [`TriggerSchema`].
//!   One key per registered trigger. Read at DDL time and on demand when
//!   the per-mutation probe matches the trigger via the index below.
//!
//! - `schema:trigger_index:<target>:<event>` → MessagePack `Vec<String>` of
//!   trigger names. Built derivatively from definitions: every
//!   `CREATE TRIGGER` adds entries for each `(target, event_kind)` pair the
//!   trigger subscribes to. `DROP TRIGGER` removes them. The probe in the
//!   executor reads this index by `(label_or_edge_type, event)` so the
//!   per-mutation cost is `O(matching_triggers)`, never
//!   `O(total_trigger_count)`. the trigger architecture §3.
//!
//! `target` in the index key is prefixed with `n:` for node labels and
//! `e:` for edge types so the two namespaces never collide on the same
//! string. `event` is one of `c`, `u`, `d` for `CREATE / UPDATE / DELETE`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::graph::types::Value;

/// Persisted form of a trigger registered via `CREATE TRIGGER` (the trigger architecture).
///
/// The body is stored as a raw Cypher source string captured by the parser;
/// the executor re-parses on each firing (cheap relative to body execution,
/// and avoids serialising the AST whose shape may evolve across releases).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TriggerSchema {
    /// Trigger name (handle used by DROP / ALTER / SHOW).
    pub name: String,
    /// Target filter — node label or edge type.
    pub target: TriggerTargetSchema,
    /// Event kinds the trigger subscribes to.
    pub events: TriggerEventsSchema,
    /// Synchronous (`BEFORE COMMIT`) or asynchronous (`AFTER COMMIT`).
    pub timing: TriggerTimingSchema,
    /// Raw Cypher source of the trigger body — re-parsed on firing.
    pub body_source: String,
    /// `MAXDEPTH n` / `CASCADE_LIMIT n` per-trigger override (the trigger architecture L1).
    /// `None` = use cluster default `triggers.max_cascade_depth`.
    pub cascade_limit: Option<u32>,
    /// `CASCADE_FANOUT n` per-trigger override (the trigger architecture L2).
    /// `None` = use cluster default `triggers.max_cascade_fanout`.
    pub cascade_fanout: Option<u32>,
    /// Error-handling policy. `None` = use the default for `timing`
    /// (`BEFORE` → Propagate, `AFTER` → Retry 3 / 1000ms) per the trigger architecture.
    pub on_error: Option<OnErrorPolicySchema>,
    /// Whether the trigger is currently firing. Flipped by
    /// `ALTER TRIGGER … DISABLE / ENABLE`.
    pub enabled: bool,
    /// HLC microseconds at registration time. Used for `SHOW TRIGGERS`
    /// output and tie-breaking when multiple triggers match the same event.
    pub created_at_hlc_us: u64,
}

/// Target filter — node label or edge type. The struct-variant shape (with
/// a single `name` field per variant) is intentional: `rmp_serde` does not
/// support tagged newtype variants, but it does support tagged struct
/// variants, so this encoding is forward-compatible with future fields.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TriggerTargetSchema {
    Label { name: String },
    EdgeType { name: String },
}

impl TriggerTargetSchema {
    /// Construct a label-target value.
    pub fn label(name: impl Into<String>) -> Self {
        Self::Label { name: name.into() }
    }
    /// Construct an edge-type-target value.
    pub fn edge_type(name: impl Into<String>) -> Self {
        Self::EdgeType { name: name.into() }
    }
    /// The inner identifier, regardless of variant.
    pub fn name(&self) -> &str {
        match self {
            Self::Label { name } | Self::EdgeType { name } => name,
        }
    }
}

impl TriggerTargetSchema {
    /// String form used as the `target` segment of trigger-index keys.
    /// `n:Label` for nodes, `e:EdgeType` for edges. The two namespaces are
    /// disjoint so an `:Order` label and an `:Order` edge type never collide.
    pub fn index_key_segment(&self) -> String {
        match self {
            Self::Label { name } => format!("n:{name}"),
            Self::EdgeType { name } => format!("e:{name}"),
        }
    }
}

/// Persisted event-kind bitset (mirrors `cypher::ast::TriggerEvents` but
/// lives in coordinode-core so storage can decode it without depending on
/// coordinode-query).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct TriggerEventsSchema {
    pub on_create: bool,
    pub on_update: bool,
    pub on_delete: bool,
}

impl TriggerEventsSchema {
    /// Returns the event kinds that are enabled, as their single-char index
    /// segments (`c`, `u`, `d`).
    pub fn enabled_segments(self) -> Vec<&'static str> {
        let mut out = Vec::with_capacity(3);
        if self.on_create {
            out.push("c");
        }
        if self.on_update {
            out.push("u");
        }
        if self.on_delete {
            out.push("d");
        }
        out
    }

    /// Returns `true` if at least one event kind is enabled.
    pub fn any(self) -> bool {
        self.on_create || self.on_update || self.on_delete
    }
}

/// Execution timing for a trigger (persisted form).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TriggerTimingSchema {
    BeforeCommit,
    AfterCommit,
}

/// Per-trigger error policy (persisted form, the trigger architecture).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OnErrorPolicySchema {
    Propagate,
    Retry { n: u32, backoff_ms: u32 },
    DeadLetter,
}

/// Encode the primary key for a trigger definition: `schema:trigger:<name>`.
pub fn encode_trigger_key(name: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(15 + name.len());
    key.extend_from_slice(b"schema:trigger:");
    key.extend_from_slice(name.as_bytes());
    key
}

/// Encode the secondary-index key for trigger lookups by `(target, event)`:
/// `schema:trigger_index:<target_segment>:<event_segment>`.
///
/// `target_segment` is produced by `TriggerTargetSchema::index_key_segment`;
/// `event_segment` is one of `c` / `u` / `d`.
pub fn encode_trigger_index_key(target_segment: &str, event_segment: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(21 + target_segment.len() + 1 + event_segment.len());
    key.extend_from_slice(b"schema:trigger_index:");
    key.extend_from_slice(target_segment.as_bytes());
    key.push(b':');
    key.extend_from_slice(event_segment.as_bytes());
    key
}

/// Prefix used to scan every trigger definition (for `SHOW TRIGGERS`).
pub fn trigger_scan_prefix() -> &'static [u8] {
    b"schema:trigger:"
}

// ── AFTER COMMIT event journal (the trigger architecture, ADR-026) ──────────────
//
// AFTER COMMIT triggers do not fire inline. When a mutation matches an
// AFTER COMMIT trigger, the committing transaction enqueues a durable
// [`PendingTriggerEvent`] under `trigger_pending:<name>:<seq>` in the same
// (Raft-replicated) transaction as the user write. The dispatcher consumes the
// queue out-of-band, executes the trigger body, and removes the entry on
// success. On failure the entry is retried in place (per the trigger's
// `ON ERROR RETRY` policy) and, once retries are exhausted (or under
// `DEAD_LETTER`), moved to a [`FailedTriggerEvent`] under
// `trigger_failures:<name>:<seq>`. The two key families live in the same
// schema partition as the definitions, so the queue replicates and survives
// failover with the rest of the metadata.

/// A queued AFTER COMMIT trigger firing awaiting execution by the dispatcher.
///
/// The payload carries the trigger parameters (`$event`, `$before`, `$after`,
/// `$node` / `$src` / `$tgt` / `$edge_type`) computed precisely at commit time,
/// so the dispatcher never reconstructs them from raw storage ops. The trigger
/// body and `ON ERROR` policy are read live from the definition at dispatch
/// time (so `ALTER TRIGGER … SET EXECUTE / SET ON ERROR / DISABLE` take effect
/// on still-queued events).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PendingTriggerEvent {
    /// Trigger this event belongs to (key into `schema:trigger:<name>`).
    pub trigger_name: String,
    /// Bound parameters for the body: `event` / `before` / `after` plus the
    /// node or edge endpoints, exactly as the synchronous param builders
    /// produce them. Keys are the bare `$`-names (no leading `$`).
    pub params: BTreeMap<String, Value>,
    /// Retry attempts already spent. `0` on first enqueue.
    pub attempt: u32,
    /// Async cascade generation: `0` for events enqueued by a user mutation,
    /// `n+1` for events enqueued by a generation-`n` trigger body. Bounds the
    /// async cascade depth (L1, the trigger architecture) — a generation past
    /// the cluster `max_cascade_depth` is dead-lettered instead of executed.
    pub generation: u32,
    /// HLC microseconds when the event was first enqueued.
    pub first_seen_us: u64,
    /// HLC microseconds before which the dispatcher must not re-attempt this
    /// event (exponential backoff). `0` = due immediately.
    pub next_attempt_us: u64,
}

/// A dead-lettered AFTER COMMIT trigger firing — retries exhausted, or a
/// `DEAD_LETTER` / `PROPAGATE` policy on a committed transaction, or a cascade
/// overflow. Inspectable via `SHOW TRIGGER FAILURES` (the trigger architecture).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FailedTriggerEvent {
    /// Trigger this event belongs to.
    pub trigger_name: String,
    /// The body parameters at the time of failure (same shape as the pending form).
    pub params: BTreeMap<String, Value>,
    /// Error chain, newest last (e.g. body parse error → executor error).
    pub error_chain: Vec<String>,
    /// Number of execution attempts made before dead-lettering.
    pub attempts: u32,
    /// HLC microseconds of the first failure.
    pub first_fail_us: u64,
    /// HLC microseconds of the last (dead-lettering) failure.
    pub last_fail_us: u64,
    /// `true` when the event was dead-lettered by a cascade-overflow trip
    /// rather than a body error.
    pub cascade_overflow: bool,
}

/// Encode the key for a queued AFTER COMMIT event:
/// `trigger_pending:<name>:<seq_be>`. `seq` is a monotonic HLC microsecond
/// stamp assigned at enqueue; the big-endian suffix keeps queue order =
/// enqueue order within a trigger when scanned by prefix.
pub fn encode_trigger_pending_key(name: &str, seq: u64) -> Vec<u8> {
    let mut key = Vec::with_capacity(16 + name.len() + 1 + 8);
    key.extend_from_slice(b"trigger_pending:");
    key.extend_from_slice(name.as_bytes());
    key.push(b':');
    key.extend_from_slice(&seq.to_be_bytes());
    key
}

/// Encode the key for a dead-lettered AFTER COMMIT event:
/// `trigger_failures:<name>:<seq_be>`.
pub fn encode_trigger_failure_key(name: &str, seq: u64) -> Vec<u8> {
    let mut key = Vec::with_capacity(17 + name.len() + 1 + 8);
    key.extend_from_slice(b"trigger_failures:");
    key.extend_from_slice(name.as_bytes());
    key.push(b':');
    key.extend_from_slice(&seq.to_be_bytes());
    key
}

/// Prefix scanning every queued AFTER COMMIT event across all triggers.
pub fn trigger_pending_scan_prefix() -> &'static [u8] {
    b"trigger_pending:"
}

/// Prefix scanning every dead-lettered AFTER COMMIT event across all triggers.
pub fn trigger_failures_scan_prefix() -> &'static [u8] {
    b"trigger_failures:"
}

/// Recover the trailing big-endian `seq` from a `trigger_pending:` /
/// `trigger_failures:` key. Returns `None` when the key is too short to carry
/// the 8-byte suffix.
pub fn decode_trigger_event_seq(key: &[u8]) -> Option<u64> {
    if key.len() < 8 {
        return None;
    }
    let suffix: [u8; 8] = key[key.len() - 8..].try_into().ok()?;
    Some(u64::from_be_bytes(suffix))
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default
)]
mod tests;
