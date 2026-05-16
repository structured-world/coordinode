//! Node storage: key encoding, ID allocation, and record serialization.
//!
//! Nodes are stored as single KV entries in the `node:` partition:
//!
//! ```text
//! Key:   node:<shard_id u16 BE>:<node_id u64 BE>
//! Value: MessagePack { label: String, props: HashMap<u32, Value> }
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

/// Width of the `origin_shard_hint` field in the NodeId u64.
///
/// Layout: `[20 bits origin_shard_hint][44 bits sequence]`. The hint records
/// the shard where the node was originally created — it is immutable and
/// never rewritten, even across re-shards. CE single-shard deployments use
/// hint = 0 (reserved sentinel meaning "consult the routing layer").
pub const NODE_ID_HINT_BITS: u32 = 20;

/// Width of the `sequence` field in the NodeId u64.
pub const NODE_ID_SEQUENCE_BITS: u32 = 64 - NODE_ID_HINT_BITS;

/// Inclusive maximum value for the `sequence` field — wrap into hint bits is
/// a hard panic in the allocator (would corrupt routing).
pub const NODE_ID_MAX_SEQUENCE: u64 = (1u64 << NODE_ID_SEQUENCE_BITS) - 1;

/// Inclusive maximum value for the `shard_hint` field.
pub const NODE_ID_MAX_HINT: u32 = (1u32 << NODE_ID_HINT_BITS) - 1;

/// A unique 64-bit node identifier with embedded origin-shard hint.
///
/// Layout: `[20 bits origin_shard_hint][44 bits sequence]`. The hint records
/// the shard where the node was originally created (immutable for the
/// lifetime of the node); the sequence is per-shard monotonic. CE uses
/// hint = 0; EE uses coordinator-assigned hints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NodeId(u64);

impl NodeId {
    /// Create a NodeId from a raw u64.
    pub fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Compose a NodeId from `(shard_hint, sequence)`.
    ///
    /// Panics if `shard_hint > NODE_ID_MAX_HINT` or `sequence > NODE_ID_MAX_SEQUENCE`
    /// — both are architectural invariants and a violation indicates a bug in
    /// the caller (coordinator misconfiguration or allocator wrap).
    pub fn compose(shard_hint: u32, sequence: u64) -> Self {
        assert!(
            shard_hint <= NODE_ID_MAX_HINT,
            "shard_hint {shard_hint} exceeds 20-bit ceiling {NODE_ID_MAX_HINT}"
        );
        assert!(
            sequence <= NODE_ID_MAX_SEQUENCE,
            "sequence {sequence} exceeds 44-bit ceiling {NODE_ID_MAX_SEQUENCE}"
        );
        Self((u64::from(shard_hint) << NODE_ID_SEQUENCE_BITS) | sequence)
    }

    /// Get the raw u64 value.
    pub fn as_raw(self) -> u64 {
        self.0
    }

    /// Extract the origin shard hint (top 20 bits).
    ///
    /// In CE this is always 0. In EE it identifies the shard on which the
    /// node was originally created. Re-sharding does not rewrite this value.
    pub fn origin_shard_hint(self) -> u32 {
        (self.0 >> NODE_ID_SEQUENCE_BITS) as u32
    }

    /// Extract the per-shard sequence (bottom 44 bits).
    pub fn sequence(self) -> u64 {
        self.0 & NODE_ID_MAX_SEQUENCE
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "node:{}", self.0)
    }
}

/// Crockford base32 alphabet — 32 characters, no `I`, `L`, `O`, `U`
/// (visually ambiguous or potentially obscene). Chosen for external
/// identifiers because case-insensitive, URL-safe, and copy-paste robust.
const CROCKFORD_BASE32: &[u8; 32] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";

impl NodeId {
    /// Encode the NodeId as a Crockford base32 string for external (`elementId`)
    /// use. 13 ASCII characters, bijective with the underlying u64.
    ///
    /// Encoding splits the 64-bit value into 13 × 5-bit groups (the top group
    /// uses 4 bits; the high bit is always zero by construction since u64 has
    /// 64 bits = 12 × 5 + 4). The result is time-sortable within a shard
    /// (sequence grows monotonically) and stable across re-shards (since NodeId
    /// is immutable). No mapping table is required — the inverse is computed
    /// directly from the characters.
    pub fn to_element_id(self) -> String {
        let raw = self.0;
        // 64 bits → 13 characters, MSB first. Use 4 bits for the top character
        // (highest nibble) and 5 bits for the remaining 12. Every byte we
        // write is a Crockford base32 digit (ASCII subset of UTF-8), so
        // `from_utf8` is guaranteed to succeed and `expect` documents the
        // invariant for readers.
        let mut out = String::with_capacity(13);
        out.push(CROCKFORD_BASE32[((raw >> 60) & 0xF) as usize] as char);
        for i in 0..12 {
            let shift = 55 - (i as u32) * 5;
            let idx = ((raw >> shift) & 0x1F) as usize;
            out.push(CROCKFORD_BASE32[idx] as char);
        }
        out
    }

    /// Decode a Crockford base32 `elementId` back into a NodeId.
    ///
    /// Returns `None` if the input is not exactly 13 valid Crockford base32
    /// characters. Case-insensitive — `I`, `L` are normalised to `1`, `O` to
    /// `0` (Crockford's tolerance rules), other invalid characters fail.
    pub fn from_element_id(s: &str) -> Option<Self> {
        let bytes = s.as_bytes();
        if bytes.len() != 13 {
            return None;
        }
        // First character carries 4 bits (top nibble of the u64).
        let v0 = decode_crockford_char(bytes[0])?;
        if v0 > 0xF {
            return None;
        }
        let mut raw: u64 = u64::from(v0) << 60;
        for i in 0..12 {
            let v = decode_crockford_char(bytes[i + 1])?;
            let shift = 55 - (i as u32) * 5;
            raw |= u64::from(v) << shift;
        }
        Some(Self(raw))
    }
}

/// Decode a single Crockford base32 character to its 5-bit value.
///
/// Accepts case-insensitive `0-9`, `A-H`, `J`, `K`, `M`, `N`, `P-T`, `V-Z`
/// plus Crockford normalisations: `I`/`L` → `1`, `O` → `0`.
fn decode_crockford_char(c: u8) -> Option<u8> {
    match c {
        b'0' | b'O' | b'o' => Some(0),
        b'1' | b'I' | b'i' | b'L' | b'l' => Some(1),
        b'2'..=b'9' => Some(c - b'0'),
        b'A'..=b'H' => Some(c - b'A' + 10),
        b'a'..=b'h' => Some(c - b'a' + 10),
        b'J' | b'j' => Some(18),
        b'K' | b'k' => Some(19),
        b'M' | b'm' => Some(20),
        b'N' | b'n' => Some(21),
        b'P'..=b'T' => Some(c - b'P' + 22),
        b'p'..=b't' => Some(c - b'p' + 22),
        b'V'..=b'Z' => Some(c - b'V' + 27),
        b'v'..=b'z' => Some(c - b'v' + 27),
        _ => None,
    }
}

/// Monotonic per-shard node ID allocator.
///
/// Thread-safe, lock-free. Each call to `next()` increments the local
/// sequence counter and composes it with the configured `shard_hint` into a
/// NodeId. CE constructs with `shard_hint = 0`; EE shard leaders construct
/// with their coordinator-assigned hint.
///
/// Sequence is constrained to `[1, NODE_ID_MAX_SEQUENCE]` — wrap is a hard
/// panic. At 1M writes/sec on a single shard the 44-bit space exhausts in
/// ~540 days; at 10K writes/sec (typical enterprise) in ~55 years — so
/// exhaustion is not a steady-state concern, but the check is mandatory
/// because wrap would corrupt routing (sequence bits leaking into hint bits
/// would map nodes to phantom shards).
pub struct NodeIdAllocator {
    shard_hint: u32,
    counter: AtomicU64,
}

impl NodeIdAllocator {
    /// Create a new allocator for a given shard hint, starting from sequence 0.
    ///
    /// The `shard_hint` must fit in `NODE_ID_HINT_BITS` (≤ 2^20 - 1). CE
    /// callers pass `0`; EE shard leaders pass their coordinator-assigned
    /// hint. The hint is fixed for the allocator's lifetime — changing it
    /// requires creating a new allocator (which never happens in practice —
    /// shards do not change their hint after the coordinator assigns it).
    pub fn new(shard_hint: u32) -> Self {
        assert!(
            shard_hint <= NODE_ID_MAX_HINT,
            "shard_hint {shard_hint} exceeds 20-bit ceiling {NODE_ID_MAX_HINT}"
        );
        Self {
            shard_hint,
            counter: AtomicU64::new(0),
        }
    }

    /// Create an allocator resuming from the last allocated ID.
    ///
    /// The shard hint is inferred from `last_id` so persisted high-water
    /// marks round-trip across restarts without separate bookkeeping.
    pub fn resume_from(last_id: NodeId) -> Self {
        Self {
            shard_hint: last_id.origin_shard_hint(),
            counter: AtomicU64::new(last_id.sequence()),
        }
    }

    /// Allocate the next node ID for this shard.
    ///
    /// Increments the per-shard sequence counter and composes it with the
    /// fixed hint. Panics on wrap into the hint range — see struct docs.
    pub fn next(&self) -> NodeId {
        let sequence = self.counter.fetch_add(1, Ordering::SeqCst) + 1;
        assert!(
            sequence <= NODE_ID_MAX_SEQUENCE,
            "NodeIdAllocator sequence wrap for shard_hint {}: {sequence} > {NODE_ID_MAX_SEQUENCE}",
            self.shard_hint
        );
        NodeId::compose(self.shard_hint, sequence)
    }

    /// Get the current sequence high-water mark composed with the hint.
    pub fn current(&self) -> NodeId {
        NodeId::compose(self.shard_hint, self.counter.load(Ordering::SeqCst))
    }

    /// Advance to at least the given ID's sequence (for recovery).
    ///
    /// The hint must match — advancing across hints is a logic error
    /// (sequence space is per-shard).
    pub fn advance_to(&self, id: NodeId) {
        assert_eq!(
            id.origin_shard_hint(),
            self.shard_hint,
            "advance_to received NodeId from shard_hint {} but allocator is for {}",
            id.origin_shard_hint(),
            self.shard_hint,
        );
        self.counter.fetch_max(id.sequence(), Ordering::SeqCst);
    }

    /// The shard hint this allocator is configured for.
    pub fn shard_hint(&self) -> u32 {
        self.shard_hint
    }
}

// -- Key encoding --

/// Key prefix for node records.
const NODE_KEY_PREFIX: &[u8] = b"node:";

/// Encode a node key: `node:<shard_id u16 BE>:<node_id u64 BE>`.
///
/// Big-endian ensures lexicographic ordering matches numeric ordering,
/// enabling efficient range scans within a shard.
pub fn encode_node_key(shard_id: u16, node_id: NodeId) -> Vec<u8> {
    let mut key = Vec::with_capacity(NODE_KEY_PREFIX.len() + 2 + 1 + 8);
    key.extend_from_slice(NODE_KEY_PREFIX);
    key.extend_from_slice(&shard_id.to_be_bytes());
    key.push(b':');
    key.extend_from_slice(&node_id.as_raw().to_be_bytes());
    key
}

/// Decode a node key back into (shard_id, node_id).
///
/// Returns `None` if the key doesn't match the expected format.
pub fn decode_node_key(key: &[u8]) -> Option<(u16, NodeId)> {
    let prefix_len = NODE_KEY_PREFIX.len();
    // node: (5) + shard (2) + : (1) + id (8) = 16
    if key.len() != prefix_len + 2 + 1 + 8 {
        return None;
    }
    if &key[..prefix_len] != NODE_KEY_PREFIX {
        return None;
    }
    if key[prefix_len + 2] != b':' {
        return None;
    }
    let shard_id = u16::from_be_bytes(key[prefix_len..prefix_len + 2].try_into().ok()?);
    let node_id = u64::from_be_bytes(key[prefix_len + 3..prefix_len + 11].try_into().ok()?);
    Some((shard_id, NodeId(node_id)))
}

// -- Node record --

/// A node record stored in the `node:` partition.
///
/// Properties are stored with interned field IDs (u32 keys) rather than
/// string field names, achieving ~80% reduction in key storage.
///
/// In VALIDATED schema mode, undeclared properties are stored in `extra`
/// with string keys (no interning). Declared properties remain in `props`.
///
/// Nodes support multiple labels per OpenCypher spec: `(n:User:Admin)`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NodeRecord {
    /// The node's labels (e.g., ["User", "Admin"]).
    /// First label is the primary label used for schema lookups.
    pub labels: Vec<String>,

    /// Properties keyed by interned field ID.
    /// Values are MessagePack-compatible via `PropertyValue`.
    pub props: HashMap<u32, PropertyValue>,

    /// Overflow map for undeclared properties in VALIDATED schema mode.
    /// Uses string keys (no interning) to avoid polluting the field interner
    /// with ad-hoc property names. Empty/None in STRICT and FLEXIBLE modes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<HashMap<String, PropertyValue>>,
}

/// A property value — alias for the full type system `Value` enum.
///
/// See `graph::types::Value` for all 12 supported types.
pub type PropertyValue = super::types::Value;

impl NodeRecord {
    /// Create a new node record with a single label and no properties.
    pub fn new(label: impl Into<String>) -> Self {
        let label = label.into();
        Self {
            labels: if label.is_empty() {
                Vec::new()
            } else {
                vec![label]
            },
            props: HashMap::new(),
            extra: None,
        }
    }

    /// Create a node record with multiple labels.
    pub fn with_labels(labels: Vec<String>) -> Self {
        Self {
            labels,
            props: HashMap::new(),
            extra: None,
        }
    }

    /// The primary label (first in the list), or empty string if no labels.
    pub fn primary_label(&self) -> &str {
        self.labels.first().map(|s| s.as_str()).unwrap_or("")
    }

    /// Check if the node has a specific label.
    pub fn has_label(&self, label: &str) -> bool {
        self.labels.iter().any(|l| l == label)
    }

    /// Add a label if not already present.
    pub fn add_label(&mut self, label: String) {
        if !self.has_label(&label) {
            self.labels.push(label);
        }
    }

    /// Remove a label. Returns true if the label was present.
    pub fn remove_label(&mut self, label: &str) -> bool {
        let len_before = self.labels.len();
        self.labels.retain(|l| l != label);
        self.labels.len() < len_before
    }

    /// Set a property by interned field ID.
    pub fn set(&mut self, field_id: u32, value: PropertyValue) {
        self.props.insert(field_id, value);
    }

    /// Get a property by interned field ID.
    pub fn get(&self, field_id: u32) -> Option<&PropertyValue> {
        self.props.get(&field_id)
    }

    /// Remove a property by interned field ID.
    pub fn remove(&mut self, field_id: u32) -> Option<PropertyValue> {
        self.props.remove(&field_id)
    }

    /// Set an undeclared property in the extra overflow map (VALIDATED mode).
    pub fn set_extra(&mut self, name: impl Into<String>, value: PropertyValue) {
        self.extra
            .get_or_insert_with(HashMap::new)
            .insert(name.into(), value);
    }

    /// Get an undeclared property from the extra overflow map.
    pub fn get_extra(&self, name: &str) -> Option<&PropertyValue> {
        self.extra.as_ref()?.get(name)
    }

    /// Serialize to MessagePack bytes.
    pub fn to_msgpack(&self) -> Result<Vec<u8>, rmp_serde::encode::Error> {
        rmp_serde::to_vec(self)
    }

    /// Deserialize from MessagePack bytes.
    ///
    /// Handles both raw msgpack (legacy) and prefix-encoded format from the
    /// DocumentMerge operator (0x00 prefix = full NodeRecord, see ADR-015).
    pub fn from_msgpack(data: &[u8]) -> Result<Self, rmp_serde::decode::Error> {
        if !data.is_empty() && data[0] == crate::graph::doc_delta::PREFIX_NODE_RECORD {
            // Prefix-encoded format (after DocumentMerge): strip 0x00 prefix.
            rmp_serde::from_slice(&data[1..])
        } else if !data.is_empty() && data[0] == crate::graph::doc_delta::PREFIX_DOC_DELTA {
            // This is a raw merge operand, not a full record — cannot decode.
            Err(rmp_serde::decode::Error::Syntax(
                "cannot decode DocDelta merge operand as NodeRecord".to_string(),
            ))
        } else {
            // Legacy: raw msgpack without prefix.
            rmp_serde::from_slice(data)
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    // -- NodeId tests --

    #[test]
    fn node_id_roundtrip() {
        let id = NodeId::from_raw(42);
        assert_eq!(id.as_raw(), 42);
    }

    #[test]
    fn node_id_ordering() {
        let a = NodeId::from_raw(1);
        let b = NodeId::from_raw(2);
        assert!(a < b);
    }

    #[test]
    fn node_id_display() {
        let id = NodeId::from_raw(99);
        assert_eq!(format!("{id}"), "node:99");
    }

    // -- NodeIdAllocator tests --

    #[test]
    fn allocator_starts_from_one() {
        let alloc = NodeIdAllocator::new(0);
        assert_eq!(alloc.next().as_raw(), 1);
        assert_eq!(alloc.next().as_raw(), 2);
    }

    #[test]
    fn allocator_resume() {
        let alloc = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        assert_eq!(alloc.next().as_raw(), 101);
    }

    #[test]
    fn allocator_current_does_not_advance() {
        let alloc = NodeIdAllocator::resume_from(NodeId::from_raw(50));
        assert_eq!(alloc.current().as_raw(), 50);
        assert_eq!(alloc.current().as_raw(), 50);
    }

    #[test]
    fn allocator_advance_to() {
        let alloc = NodeIdAllocator::resume_from(NodeId::from_raw(10));
        alloc.advance_to(NodeId::from_raw(100));
        assert_eq!(alloc.next().as_raw(), 101);
    }

    #[test]
    fn allocator_concurrent() {
        use std::collections::BTreeSet;
        use std::sync::Arc;

        let alloc = Arc::new(NodeIdAllocator::new(0));
        let mut handles = Vec::new();

        for _ in 0..8 {
            let alloc = Arc::clone(&alloc);
            handles.push(std::thread::spawn(move || {
                (0..500).map(|_| alloc.next().as_raw()).collect::<Vec<_>>()
            }));
        }

        let mut all: BTreeSet<u64> = BTreeSet::new();
        for h in handles {
            for id in h.join().expect("thread panicked") {
                assert!(all.insert(id), "duplicate ID: {id}");
            }
        }
        assert_eq!(all.len(), 4000);
    }

    // -- u20/u44 layout tests --

    #[test]
    fn node_id_compose_extracts_hint_and_sequence() {
        let id = NodeId::compose(42, 1_000_000);
        assert_eq!(id.origin_shard_hint(), 42);
        assert_eq!(id.sequence(), 1_000_000);
    }

    #[test]
    fn node_id_compose_handles_max_hint() {
        let id = NodeId::compose(NODE_ID_MAX_HINT, 1);
        assert_eq!(id.origin_shard_hint(), NODE_ID_MAX_HINT);
        assert_eq!(id.sequence(), 1);
    }

    #[test]
    fn node_id_compose_handles_max_sequence() {
        let id = NodeId::compose(0, NODE_ID_MAX_SEQUENCE);
        assert_eq!(id.origin_shard_hint(), 0);
        assert_eq!(id.sequence(), NODE_ID_MAX_SEQUENCE);
    }

    #[test]
    #[should_panic(expected = "shard_hint")]
    fn node_id_compose_panics_on_oversized_hint() {
        NodeId::compose(NODE_ID_MAX_HINT + 1, 0);
    }

    #[test]
    #[should_panic(expected = "sequence")]
    fn node_id_compose_panics_on_oversized_sequence() {
        NodeId::compose(0, NODE_ID_MAX_SEQUENCE + 1);
    }

    #[test]
    fn allocator_ce_default_uses_hint_zero() {
        let alloc = NodeIdAllocator::new(0);
        assert_eq!(alloc.shard_hint(), 0);
        let id = alloc.next();
        assert_eq!(id.origin_shard_hint(), 0);
        assert_eq!(id.sequence(), 1);
    }

    #[test]
    fn allocator_ee_hint_carried_in_emitted_ids() {
        let alloc = NodeIdAllocator::new(42);
        let id1 = alloc.next();
        let id2 = alloc.next();
        assert_eq!(id1.origin_shard_hint(), 42);
        assert_eq!(id2.origin_shard_hint(), 42);
        assert_eq!(id1.sequence(), 1);
        assert_eq!(id2.sequence(), 2);
    }

    #[test]
    fn allocator_resume_from_preserves_hint() {
        // EE shard 7 persisted a high-water mark at sequence 1000.
        let persisted = NodeId::compose(7, 1000);
        let alloc = NodeIdAllocator::resume_from(persisted);
        assert_eq!(alloc.shard_hint(), 7);
        let next = alloc.next();
        assert_eq!(next.origin_shard_hint(), 7);
        assert_eq!(next.sequence(), 1001);
    }

    #[test]
    #[should_panic(expected = "advance_to received NodeId from shard_hint")]
    fn allocator_advance_to_rejects_cross_hint_id() {
        let alloc = NodeIdAllocator::new(3);
        // Trying to advance against an id from a different shard is a logic
        // bug — sequence space is per-shard.
        alloc.advance_to(NodeId::compose(5, 100));
    }

    #[test]
    #[should_panic(expected = "sequence wrap")]
    fn allocator_panics_on_sequence_wrap() {
        // Resume from the last possible sequence value in the 44-bit window.
        // The next `next()` call should detect wrap and panic — without this
        // guard sequence bits would leak into the 20-bit hint window and
        // corrupt routing (phantom shards).
        let alloc = NodeIdAllocator::resume_from(NodeId::compose(0, NODE_ID_MAX_SEQUENCE));
        let _ = alloc.next();
    }

    #[test]
    fn element_id_rejects_too_long_string() {
        // 14 chars — must be rejected (canonical is exactly 13).
        assert!(NodeId::from_element_id("0000000000000X").is_none());
    }

    #[test]
    fn element_id_rejects_one_char_short() {
        // 12 chars — must be rejected.
        assert!(NodeId::from_element_id("000000000000").is_none());
    }

    #[test]
    fn element_id_roundtrips_extreme_combined() {
        // Both fields at their respective maxima — exercises the bit-packing
        // boundary between sequence and hint windows simultaneously.
        let id = NodeId::compose(NODE_ID_MAX_HINT, NODE_ID_MAX_SEQUENCE);
        let s = id.to_element_id();
        assert_eq!(s.len(), 13);
        assert_eq!(NodeId::from_element_id(&s), Some(id));
        assert_eq!(id.origin_shard_hint(), NODE_ID_MAX_HINT);
        assert_eq!(id.sequence(), NODE_ID_MAX_SEQUENCE);
    }

    // -- elementId base32 encoding tests --

    #[test]
    fn element_id_roundtrips_zero() {
        let id = NodeId::from_raw(0);
        let s = id.to_element_id();
        assert_eq!(s.len(), 13);
        assert_eq!(NodeId::from_element_id(&s), Some(id));
    }

    #[test]
    fn element_id_roundtrips_max() {
        let id = NodeId::from_raw(u64::MAX);
        let s = id.to_element_id();
        assert_eq!(s.len(), 13);
        assert_eq!(NodeId::from_element_id(&s), Some(id));
    }

    #[test]
    fn element_id_roundtrips_composed_id() {
        let id = NodeId::compose(NODE_ID_MAX_HINT, NODE_ID_MAX_SEQUENCE);
        let s = id.to_element_id();
        let decoded = NodeId::from_element_id(&s).expect("decode");
        assert_eq!(decoded.origin_shard_hint(), NODE_ID_MAX_HINT);
        assert_eq!(decoded.sequence(), NODE_ID_MAX_SEQUENCE);
    }

    #[test]
    fn element_id_within_shard_is_time_sortable() {
        // Within one shard, growing sequence ⇒ growing elementId string.
        let early = NodeId::compose(5, 1).to_element_id();
        let later = NodeId::compose(5, 1_000_000).to_element_id();
        assert!(early < later, "expected {early} < {later}");
    }

    #[test]
    fn element_id_rejects_invalid_input() {
        assert!(NodeId::from_element_id("").is_none());
        assert!(NodeId::from_element_id("TOO_SHORT").is_none());
        assert!(NodeId::from_element_id("INVALID!CHARS").is_none()); // 13 chars but '!' invalid
    }

    #[test]
    fn element_id_normalises_crockford_aliases() {
        // I/L → 1 and O → 0 per Crockford spec.
        let canonical = NodeId::compose(0, 1).to_element_id();
        let with_alias_l = canonical.replace('1', "L");
        let with_alias_i = canonical.replace('1', "I");
        let with_alias_o = canonical.replace('0', "O");
        assert_eq!(
            NodeId::from_element_id(&with_alias_l),
            Some(NodeId::compose(0, 1))
        );
        assert_eq!(
            NodeId::from_element_id(&with_alias_i),
            Some(NodeId::compose(0, 1))
        );
        assert_eq!(
            NodeId::from_element_id(&with_alias_o),
            Some(NodeId::compose(0, 1))
        );
    }

    // -- Key encoding tests --

    #[test]
    fn encode_decode_key_roundtrip() {
        let shard = 42u16;
        let id = NodeId::from_raw(12345);
        let key = encode_node_key(shard, id);
        let (dec_shard, dec_id) = decode_node_key(&key).expect("decode failed");
        assert_eq!(dec_shard, shard);
        assert_eq!(dec_id, id);
    }

    #[test]
    fn key_ordering_within_shard() {
        let k1 = encode_node_key(1, NodeId::from_raw(100));
        let k2 = encode_node_key(1, NodeId::from_raw(200));
        assert!(k1 < k2, "keys should sort by node_id within shard");
    }

    #[test]
    fn key_ordering_across_shards() {
        let k1 = encode_node_key(1, NodeId::from_raw(999));
        let k2 = encode_node_key(2, NodeId::from_raw(1));
        assert!(k1 < k2, "shard 1 keys should sort before shard 2");
    }

    #[test]
    fn decode_invalid_key() {
        assert!(decode_node_key(b"").is_none());
        assert!(decode_node_key(b"short").is_none());
        assert!(decode_node_key(b"wrong:prefix12345678").is_none());
    }

    #[test]
    fn key_starts_with_prefix() {
        let key = encode_node_key(0, NodeId::from_raw(1));
        assert!(key.starts_with(b"node:"));
    }

    // -- NodeRecord tests --

    #[test]
    fn record_new() {
        let rec = NodeRecord::new("User");
        assert_eq!(rec.primary_label(), "User");
        assert_eq!(rec.labels, vec!["User"]);
        assert!(rec.props.is_empty());
    }

    #[test]
    fn record_with_labels() {
        let rec = NodeRecord::with_labels(vec!["User".into(), "Admin".into()]);
        assert_eq!(rec.primary_label(), "User");
        assert!(rec.has_label("User"));
        assert!(rec.has_label("Admin"));
        assert!(!rec.has_label("Guest"));
    }

    #[test]
    fn record_add_remove_label() {
        let mut rec = NodeRecord::new("User");
        rec.add_label("Admin".into());
        assert!(rec.has_label("Admin"));
        assert_eq!(rec.labels.len(), 2);

        // Duplicate add is no-op
        rec.add_label("Admin".into());
        assert_eq!(rec.labels.len(), 2);

        // Remove label
        assert!(rec.remove_label("Admin"));
        assert!(!rec.has_label("Admin"));
        assert_eq!(rec.labels.len(), 1);

        // Remove non-existent label
        assert!(!rec.remove_label("Guest"));
    }

    #[test]
    fn record_empty_label() {
        let rec = NodeRecord::new("");
        assert!(rec.labels.is_empty());
        assert_eq!(rec.primary_label(), "");
    }

    #[test]
    fn record_set_get_remove() {
        let mut rec = NodeRecord::new("User");
        rec.set(1, PropertyValue::String("Alice".into()));
        rec.set(2, PropertyValue::Int(30));

        assert_eq!(rec.get(1), Some(&PropertyValue::String("Alice".into())));
        assert_eq!(rec.get(2), Some(&PropertyValue::Int(30)));
        assert_eq!(rec.get(99), None);

        let removed = rec.remove(1);
        assert_eq!(removed, Some(PropertyValue::String("Alice".into())));
        assert_eq!(rec.get(1), None);
    }

    #[test]
    fn record_msgpack_roundtrip() {
        let mut rec = NodeRecord::new("Movie");
        rec.set(1, PropertyValue::String("Inception".into()));
        rec.set(2, PropertyValue::Int(2010));
        rec.set(3, PropertyValue::Float(8.8));
        rec.set(4, PropertyValue::Bool(true));
        rec.set(5, PropertyValue::Null);
        rec.set(6, PropertyValue::Binary(vec![0xDE, 0xAD]));
        rec.set(
            7,
            PropertyValue::Array(vec![
                PropertyValue::Int(1),
                PropertyValue::String("two".into()),
            ]),
        );

        let bytes = rec.to_msgpack().expect("serialize failed");
        let restored = NodeRecord::from_msgpack(&bytes).expect("deserialize failed");
        assert_eq!(rec, restored);
    }

    #[test]
    fn record_empty_props_roundtrip() {
        let rec = NodeRecord::new("Empty");
        let bytes = rec.to_msgpack().expect("serialize");
        let restored = NodeRecord::from_msgpack(&bytes).expect("deserialize");
        assert_eq!(rec, restored);
    }

    #[test]
    fn record_msgpack_is_compact() {
        let mut rec = NodeRecord::new("User");
        // With interned IDs (u32 keys), the MessagePack output should be
        // significantly smaller than using string field names
        rec.set(1, PropertyValue::String("Alice".into()));
        rec.set(2, PropertyValue::Int(30));

        let bytes = rec.to_msgpack().expect("serialize");
        // MessagePack with integer keys should be quite compact
        assert!(
            bytes.len() < 50,
            "encoded size {} should be < 50",
            bytes.len()
        );
    }

    // -- G028: extra overflow map --

    #[test]
    fn extra_set_and_get() {
        let mut rec = NodeRecord::new("User");
        rec.set_extra("ad_hoc", PropertyValue::String("value".into()));
        assert_eq!(
            rec.get_extra("ad_hoc"),
            Some(&PropertyValue::String("value".into()))
        );
        assert!(rec.get_extra("missing").is_none());
    }

    #[test]
    fn extra_none_by_default() {
        let rec = NodeRecord::new("User");
        assert!(rec.extra.is_none());
        assert!(rec.get_extra("anything").is_none());
    }

    #[test]
    fn extra_roundtrip_msgpack() {
        let mut rec = NodeRecord::new("Config");
        rec.set(1, PropertyValue::String("declared".into()));
        rec.set_extra("dynamic_key", PropertyValue::Int(42));
        rec.set_extra("another", PropertyValue::Bool(true));

        let bytes = rec.to_msgpack().expect("serialize");
        let decoded = NodeRecord::from_msgpack(&bytes).expect("deserialize");

        assert_eq!(
            decoded.props.get(&1),
            Some(&PropertyValue::String("declared".into()))
        );
        assert_eq!(
            decoded.get_extra("dynamic_key"),
            Some(&PropertyValue::Int(42))
        );
        assert_eq!(
            decoded.get_extra("another"),
            Some(&PropertyValue::Bool(true))
        );
    }

    #[test]
    fn extra_none_skipped_in_serialization() {
        // NodeRecord without extra should be backward compatible
        let mut rec = NodeRecord::new("User");
        rec.set(1, PropertyValue::String("Alice".into()));
        let bytes = rec.to_msgpack().expect("serialize");

        // Should deserialize even without extra field (serde default)
        let decoded = NodeRecord::from_msgpack(&bytes).expect("deserialize");
        assert!(decoded.extra.is_none());
    }

    #[test]
    fn backward_compat_roundtrip_without_extra() {
        // NodeRecord without extra should roundtrip cleanly.
        // This verifies backward compat with old format (no extra field).
        let rec = NodeRecord::new("Test");
        let bytes = rec.to_msgpack().expect("serialize");
        let decoded = NodeRecord::from_msgpack(&bytes).expect("deserialize");
        assert_eq!(rec, decoded);
    }
}
