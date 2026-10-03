//! Node storage: key encoding, ID allocation, and record serialization.
//!
//! Nodes are stored as single KV entries in the `node:` partition:
//!
//! ```text
//! Key:   node:<shard_id u16 BE>:<node_id u64 BE>
//! Value: MessagePack [labels, props: map<u32, Value>, extra?: map<String, Value>]
//! ```
//!
//! Both maps are written with keys ascending, so equal records have equal bytes.

use alloc::sync::Arc;
use core::sync::atomic::{AtomicU64, Ordering};
use std::collections::HashMap;

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

/// Length of the token that identifies one NodeId lease grant.
pub const NODE_LEASE_TOKEN_LEN: usize = 16;

/// Schema key prefix of the granted NodeId leases. One write-once record per
/// grant, keyed by its ceiling (big-endian, so the last record holds the
/// ceiling every new grant starts from) and holding the grant's token.
pub const NODE_LEASE_KEY_PREFIX: &[u8] = b"ids:node_lease:";

/// Schema key of the lease record whose ceiling is `ceiling`.
pub fn node_lease_key(ceiling: u64) -> Vec<u8> {
    let mut key = Vec::with_capacity(NODE_LEASE_KEY_PREFIX.len() + 8);
    key.extend_from_slice(NODE_LEASE_KEY_PREFIX);
    key.extend_from_slice(&ceiling.to_be_bytes());
    key
}

/// The ceiling a lease record key names, or `None` for any other key.
pub fn decode_node_lease_key(key: &[u8]) -> Option<u64> {
    let raw: [u8; 8] = key.strip_prefix(NODE_LEASE_KEY_PREFIX)?.try_into().ok()?;
    Some(u64::from_be_bytes(raw))
}

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

/// A range of sequences `(base, ceiling]` granted to one allocator by the
/// replicated log: no other allocator is ever granted any of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdLease {
    /// The last sequence below the range.
    pub base: u64,
    /// The last sequence in the range.
    pub ceiling: u64,
}

/// Why an allocator could not hand out an identifier.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum IdLeaseError {
    /// The shard's sequence space is used up.
    #[error("the NodeId sequence space of shard {shard_hint} is exhausted")]
    Exhausted {
        /// The shard whose space ran out.
        shard_hint: u32,
    },
    /// The log did not grant a lease, e.g. because this member does not lead.
    #[error("no NodeId lease was granted: {0}")]
    NotGranted(String),
    /// The reserver answered with a range the allocator cannot use.
    #[error("the NodeId lease ({base}, {ceiling}] is not a valid range")]
    InvalidLease {
        /// The lease's base.
        base: u64,
        /// The lease's ceiling.
        ceiling: u64,
    },
}

/// Takes NodeId leases from the replicated log for a [`NodeIdAllocator`].
///
/// Each successful call returns a range no other call, on any member, ever
/// returns, and above every range granted before it; it is durable before
/// it returns, so no crash lets it be granted again.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot grant NodeId leases",
    label = "not an `IdLeaseReserver`",
    note = "the embedded database grants leases through its proposal pipeline"
)]
pub trait IdLeaseReserver: Send + Sync {
    /// Take a new lease.
    ///
    /// # Errors
    ///
    /// The log refused or could not complete the grant.
    fn reserve(&self) -> Result<IdLease, IdLeaseError>;
}

/// Bits of the counter and ceiling words that carry the lease epoch, above
/// the sequence.
const LEASE_EPOCH_SHIFT: u32 = NODE_ID_SEQUENCE_BITS;

/// Lease epochs wrap within 20 bits. Leases take even epochs only (each new
/// one is two past the last), so a draw that carries past the top of the
/// sequence space lands on an odd epoch no lease is ever read against.
const LEASE_EPOCH_MASK: u64 = (1 << (64 - LEASE_EPOCH_SHIFT)) - 1;

/// Per-shard NodeId allocator that hands out sequences only from a lease.
///
/// Lock-free. `counter` and `ceiling` each carry `[lease epoch: 20][44 bits]`:
/// the last sequence drawn and the lease's last sequence. A draw is one
/// `fetch_add` and one load, and counts only when its epoch is the ceiling's
/// and it does not pass the ceiling. A new lease publishes its ceiling, with
/// the next epoch, before its counter, so a sequence drawn under an old lease
/// is never read against a new one: a range granted to someone else is never
/// taken for ours. Sequences skipped on a lease change are gaps, never
/// duplicates.
///
/// Sequence is constrained to `[1, NODE_ID_MAX_SEQUENCE]`: a wrap would leak
/// sequence bits into the hint and map nodes to phantom shards.
pub struct NodeIdAllocator {
    shard_hint: u32,
    counter: AtomicU64,
    ceiling: AtomicU64,
    reserver: Option<Arc<dyn IdLeaseReserver>>,
}

impl NodeIdAllocator {
    /// An allocator that owns the whole sequence space of `shard_hint`,
    /// starting at 1: for a store no other member writes to (tests, tools).
    ///
    /// # Panics
    ///
    /// `shard_hint` exceeds `NODE_ID_MAX_HINT`.
    pub fn new(shard_hint: u32) -> Self {
        Self::unleased(NodeId::compose(Self::checked_hint(shard_hint), 0))
    }

    /// An allocator that owns the sequence space above `last_id`, with its
    /// shard hint, and hands out `last_id + 1` first.
    pub fn resume_from(last_id: NodeId) -> Self {
        Self::unleased(last_id)
    }

    /// An allocator that takes its sequences from leases `reserver` grants.
    /// It holds none until the first draw asks for one.
    ///
    /// # Panics
    ///
    /// `shard_hint` exceeds `NODE_ID_MAX_HINT`.
    pub fn leased(shard_hint: u32, reserver: Arc<dyn IdLeaseReserver>) -> Self {
        Self {
            shard_hint: Self::checked_hint(shard_hint),
            counter: AtomicU64::new(0),
            ceiling: AtomicU64::new(0),
            reserver: Some(reserver),
        }
    }

    fn unleased(last_id: NodeId) -> Self {
        Self {
            shard_hint: last_id.origin_shard_hint(),
            counter: AtomicU64::new(last_id.sequence()),
            ceiling: AtomicU64::new(NODE_ID_MAX_SEQUENCE),
            reserver: None,
        }
    }

    fn checked_hint(shard_hint: u32) -> u32 {
        assert!(
            shard_hint <= NODE_ID_MAX_HINT,
            "shard_hint {shard_hint} exceeds 20-bit ceiling {NODE_ID_MAX_HINT}"
        );
        shard_hint
    }

    /// Allocate the next node ID for this shard, taking a new lease when the
    /// current one is used up.
    ///
    /// # Errors
    ///
    /// The sequence space is exhausted, or a lease was needed and not
    /// granted.
    pub fn next(&self) -> Result<NodeId, IdLeaseError> {
        loop {
            let drawn = self.counter.fetch_add(1, Ordering::AcqRel) + 1;
            let ceiling = self.ceiling.load(Ordering::Acquire);
            let (drawn_epoch, lease_epoch) =
                (drawn >> LEASE_EPOCH_SHIFT, ceiling >> LEASE_EPOCH_SHIFT);
            if drawn_epoch == lease_epoch {
                if drawn & NODE_ID_MAX_SEQUENCE <= ceiling & NODE_ID_MAX_SEQUENCE {
                    return Ok(NodeId::compose(
                        self.shard_hint,
                        drawn & NODE_ID_MAX_SEQUENCE,
                    ));
                }
                self.renew(ceiling)?;
            } else if drawn_epoch == (lease_epoch + 1) & LEASE_EPOCH_MASK {
                // The draw carried past the top of the sequence space into
                // the odd epoch no lease ever uses.
                self.renew(ceiling)?;
            }
            // Otherwise the lease changed after the draw: draw again.
        }
    }

    /// Replace the lease whose ceiling word is `exhausted`. Several threads
    /// may exhaust it at once; the first to install a new lease wins and the
    /// leases the others took are left unused.
    fn renew(&self, exhausted: u64) -> Result<(), IdLeaseError> {
        let Some(reserver) = &self.reserver else {
            return Err(IdLeaseError::Exhausted {
                shard_hint: self.shard_hint,
            });
        };
        let lease = reserver.reserve()?;
        if lease.base >= lease.ceiling || lease.ceiling > NODE_ID_MAX_SEQUENCE {
            return Err(IdLeaseError::InvalidLease {
                base: lease.base,
                ceiling: lease.ceiling,
            });
        }
        let epoch = ((exhausted >> LEASE_EPOCH_SHIFT) + 2) & LEASE_EPOCH_MASK;
        let tagged = |sequence: u64| (epoch << LEASE_EPOCH_SHIFT) | sequence;
        if self
            .ceiling
            .compare_exchange(
                exhausted,
                tagged(lease.ceiling),
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            self.counter.store(tagged(lease.base), Ordering::Release);
        }
        Ok(())
    }

    /// The shard hint this allocator is configured for.
    pub fn shard_hint(&self) -> u32 {
        self.shard_hint
    }
}

// -- Key encoding --

/// Key prefix for node records. Public so that a scan of every node row
/// names the prefix from here rather than repeating the literal.
pub const NODE_KEY_PREFIX: &[u8] = b"node:";

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
/// Returns `None` if the key doesn't match the expected non-temporal format.
/// For temporal node keys (with the 9-byte valid_from suffix), use
/// [`decode_temporal_node_key`].
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

/// Encode a temporal node key:
/// `node:<shard_id u16 BE>:<node_id u64 BE>:<valid_from sortable u64 BE>`.
///
/// One entry per version. Multiple versions of the same `node_id` coexist
/// on a temporal label; an "active at valid-time T" query prefix-scans
/// [`temporal_node_id_prefix`] and stops at the upper bound. Symmetric with
/// the temporal-edge layout.
///
/// 17-byte suffix total: `node:` (5) + shard (2) + `:` (1) + id (8) + `:` (1) + valid_from (8).
pub fn encode_temporal_node_key(shard_id: u16, node_id: NodeId, valid_from_ms: i64) -> Vec<u8> {
    let mut key = Vec::with_capacity(NODE_KEY_PREFIX.len() + 2 + 1 + 8 + 1 + 8);
    key.extend_from_slice(NODE_KEY_PREFIX);
    key.extend_from_slice(&shard_id.to_be_bytes());
    key.push(b':');
    key.extend_from_slice(&node_id.as_raw().to_be_bytes());
    key.push(b':');
    key.extend_from_slice(&crate::graph::edge::encode_valid_from_sortable(
        valid_from_ms,
    ));
    key
}

/// Decode a temporal node key into `(shard_id, node_id, valid_from)`.
///
/// Returns `None` if the key isn't well-formed. The temporal form is
/// 25 bytes total (8 more than the non-temporal 16-byte form).
pub fn decode_temporal_node_key(key: &[u8]) -> Option<(u16, NodeId, i64)> {
    let prefix_len = NODE_KEY_PREFIX.len();
    // node: (5) + shard (2) + : (1) + id (8) + : (1) + valid_from (8) = 25
    if key.len() != prefix_len + 2 + 1 + 8 + 1 + 8 {
        return None;
    }
    if &key[..prefix_len] != NODE_KEY_PREFIX {
        return None;
    }
    if key[prefix_len + 2] != b':' || key[prefix_len + 11] != b':' {
        return None;
    }
    let shard_id = u16::from_be_bytes(key[prefix_len..prefix_len + 2].try_into().ok()?);
    let node_id = u64::from_be_bytes(key[prefix_len + 3..prefix_len + 11].try_into().ok()?);
    let vf_bytes: [u8; 8] = key[prefix_len + 12..prefix_len + 20].try_into().ok()?;
    let valid_from = crate::graph::edge::decode_valid_from_sortable(vf_bytes);
    Some((shard_id, NodeId(node_id), valid_from))
}

/// Prefix matching every version of a given temporal `node_id` within a
/// shard: `node:<shard>:<node_id>:`. Use for full-version enumeration on a
/// temporal label (e.g. AS-OF reads, version history, post-delete invariant
/// checks).
pub fn temporal_node_id_prefix(shard_id: u16, node_id: NodeId) -> Vec<u8> {
    let mut key = Vec::with_capacity(NODE_KEY_PREFIX.len() + 2 + 1 + 8 + 1);
    key.extend_from_slice(NODE_KEY_PREFIX);
    key.extend_from_slice(&shard_id.to_be_bytes());
    key.push(b':');
    key.extend_from_slice(&node_id.as_raw().to_be_bytes());
    key.push(b':');
    key
}

/// Pick the right node-key format for a write: temporal labels supply
/// `Some(valid_from_ms)` and get the per-version key; non-temporal labels
/// pass `None` and get the standard 16-byte key.
///
/// Mirror of `edgeprop_write_key` in the executor — the key-format choice
/// is centralised here so call sites stay declarative ("does this write
/// belong to a temporal label?" → pass the option) and never see the
/// 8-vs-17-byte branch directly.
pub fn node_write_key(shard_id: u16, node_id: NodeId, valid_from_ms: Option<i64>) -> Vec<u8> {
    match valid_from_ms {
        Some(vf) => encode_temporal_node_key(shard_id, node_id, vf),
        None => encode_node_key(shard_id, node_id),
    }
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
    #[serde(serialize_with = "serialize_sorted")]
    pub props: HashMap<u32, PropertyValue>,

    /// Overflow map for undeclared properties in VALIDATED schema mode.
    /// Uses string keys (no interning) to avoid polluting the field interner
    /// with ad-hoc property names. Empty/None in STRICT and FLEXIBLE modes.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_sorted_opt"
    )]
    pub extra: Option<HashMap<String, PropertyValue>>,
}

/// A property map serialised with its keys ascending, so equal records encode
/// to equal bytes whatever the map's iteration order: the rule the edge
/// property codec follows. It is still a MessagePack map and decodes as one.
struct SortedProps<'a, K>(&'a HashMap<K, PropertyValue>);

impl<K: Ord + Serialize> Serialize for SortedProps<'_, K> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Records usually carry a handful of properties; sorting them inline
        // keeps the canonical order off the heap on the write path.
        let mut entries: smallvec::SmallVec<[(&K, &PropertyValue); 16]> = self.0.iter().collect();
        entries.sort_unstable_by(|a, b| a.0.cmp(b.0));
        serializer.collect_map(entries)
    }
}

fn serialize_sorted<K: Ord + Serialize, S: serde::Serializer>(
    map: &HashMap<K, PropertyValue>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    SortedProps(map).serialize(serializer)
}

fn serialize_sorted_opt<K: Ord + Serialize, S: serde::Serializer>(
    map: &Option<HashMap<K, PropertyValue>>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    map.as_ref().map(SortedProps).serialize(serializer)
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

    /// Remove an undeclared property from the extra overflow map. An emptied
    /// map goes back to `None`, so the record encodes as one that never had it.
    pub fn remove_extra(&mut self, name: &str) -> Option<PropertyValue> {
        let extra = self.extra.as_mut()?;
        let removed = extra.remove(name);
        if extra.is_empty() {
            self.extra = None;
        }
        removed
    }

    /// Serialize to MessagePack bytes.
    pub fn to_msgpack(&self) -> Result<Vec<u8>, rmp_serde::encode::Error> {
        rmp_serde::to_vec(self)
    }

    /// Deserialize from MessagePack bytes.
    ///
    /// Handles both raw msgpack (legacy) and prefix-encoded format from the
    /// DocumentMerge operator (0x00 prefix = full NodeRecord).
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
mod tests;
