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

use serde::{Deserialize, Serialize};

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

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default
)]
mod tests {
    use super::*;

    #[test]
    fn trigger_key_encoding_is_stable() {
        let k = encode_trigger_key("audit_log");
        assert_eq!(k, b"schema:trigger:audit_log");
    }

    #[test]
    fn trigger_index_key_label_vs_edge_disjoint() {
        // Same identifier "Order" as a label vs. an edge type must produce
        // different index keys so the namespaces don't collide.
        let node = TriggerTargetSchema::label("Order").index_key_segment();
        let edge = TriggerTargetSchema::edge_type("Order").index_key_segment();
        let nk = encode_trigger_index_key(&node, "c");
        let ek = encode_trigger_index_key(&edge, "c");
        assert_ne!(nk, ek);
        assert_eq!(nk, b"schema:trigger_index:n:Order:c");
        assert_eq!(ek, b"schema:trigger_index:e:Order:c");
    }

    #[test]
    fn trigger_scan_prefix_matches_trigger_keys_only() {
        // A trigger-index key starts with `schema:trigger_index:` which DOES
        // share the `schema:trigger` prefix — verify the scan prefix is
        // discriminating via the trailing colon.
        let trig = encode_trigger_key("t");
        let idx = encode_trigger_index_key("n:User", "c");
        assert!(trig.starts_with(trigger_scan_prefix()));
        // Index keys MUST share the literal `schema:trigger_index:` rather
        // than `schema:trigger:` to be distinguishable from definitions.
        assert!(!idx.starts_with(trigger_scan_prefix()));
    }

    #[test]
    fn events_enabled_segments_in_order() {
        let mut e = TriggerEventsSchema::default();
        e.on_update = true;
        e.on_delete = true;
        // Order is fixed: c, u, d.
        assert_eq!(e.enabled_segments(), vec!["u", "d"]);
    }

    #[test]
    fn trigger_schema_serde_roundtrip() {
        let s = TriggerSchema {
            name: "audit".into(),
            target: TriggerTargetSchema::label("User"),
            events: TriggerEventsSchema {
                on_create: true,
                on_update: false,
                on_delete: true,
            },
            timing: TriggerTimingSchema::AfterCommit,
            body_source: "CREATE (e:Log)".into(),
            cascade_limit: Some(7),
            cascade_fanout: None,
            on_error: Some(OnErrorPolicySchema::Retry {
                n: 5,
                backoff_ms: 250,
            }),
            enabled: true,
            created_at_hlc_us: 1_700_000_000_000,
        };
        let bytes = rmp_serde::to_vec(&s).expect("encode");
        let back: TriggerSchema = rmp_serde::from_slice(&bytes).expect("decode");
        assert_eq!(s, back);
    }

    #[test]
    fn trigger_scan_prefix_test_prefix_is_correct_byte_string() {
        assert_eq!(trigger_scan_prefix(), b"schema:trigger:");
    }
}
