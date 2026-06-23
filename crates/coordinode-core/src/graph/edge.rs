//! Edge storage: posting list key encoding and adjacency list management.
//!
//! Edges are stored as posting lists per edge type, following the Dgraph pattern:
//!
//! ```text
//! Forward:  adj:<EDGE_TYPE>:out:<source_id u64 BE>  → sorted [target_id, ...]
//! Reverse:  adj:<EDGE_TYPE>:in:<target_id u64 BE>   → sorted [source_id, ...]
//! ```
//!
//! Both forward and reverse posting lists are maintained on every write/delete
//! to enable efficient traversal in both directions.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::node::{NodeId, PropertyValue};
use crate::schema::definition::PropertyType;

// -- Key encoding --

/// Write a forward adjacency key into `buf`, clearing it first.
///
/// Use this in tight loops to avoid repeated heap allocations:
/// ```no_run
/// # use coordinode_core::graph::edge::write_adj_key_forward;
/// # use coordinode_core::graph::node::NodeId;
/// let mut buf = Vec::with_capacity(64);
/// write_adj_key_forward("KNOWS", NodeId::from_raw(42), &mut buf);
/// // use buf as &[u8]
/// ```
pub fn write_adj_key_forward(edge_type: &str, source_id: NodeId, buf: &mut Vec<u8>) {
    buf.clear();
    buf.reserve(4 + edge_type.len() + 5 + 8);
    buf.extend_from_slice(b"adj:");
    buf.extend_from_slice(edge_type.as_bytes());
    buf.extend_from_slice(b":out:");
    buf.extend_from_slice(&source_id.as_raw().to_be_bytes());
}

/// Write a reverse adjacency key into `buf`, clearing it first.
///
/// Use this in tight loops to avoid repeated heap allocations:
/// ```no_run
/// # use coordinode_core::graph::edge::write_adj_key_reverse;
/// # use coordinode_core::graph::node::NodeId;
/// let mut buf = Vec::with_capacity(64);
/// write_adj_key_reverse("KNOWS", NodeId::from_raw(99), &mut buf);
/// // use buf as &[u8]
/// ```
pub fn write_adj_key_reverse(edge_type: &str, target_id: NodeId, buf: &mut Vec<u8>) {
    buf.clear();
    buf.reserve(4 + edge_type.len() + 4 + 8);
    buf.extend_from_slice(b"adj:");
    buf.extend_from_slice(edge_type.as_bytes());
    buf.extend_from_slice(b":in:");
    buf.extend_from_slice(&target_id.as_raw().to_be_bytes());
}

/// Encode a forward adjacency key: `adj:<edge_type>:out:<source_id BE>`.
///
/// Allocates a new `Vec<u8>`. In tight traversal loops, prefer [`write_adj_key_forward`]
/// with a reused buffer to avoid repeated allocations.
pub fn encode_adj_key_forward(edge_type: &str, source_id: NodeId) -> Vec<u8> {
    let mut key = Vec::new();
    write_adj_key_forward(edge_type, source_id, &mut key);
    key
}

/// Encode a reverse adjacency key: `adj:<edge_type>:in:<target_id BE>`.
///
/// Allocates a new `Vec<u8>`. In tight traversal loops, prefer [`write_adj_key_reverse`]
/// with a reused buffer to avoid repeated allocations.
pub fn encode_adj_key_reverse(edge_type: &str, target_id: NodeId) -> Vec<u8> {
    let mut key = Vec::new();
    write_adj_key_reverse(edge_type, target_id, &mut key);
    key
}

/// Encode a split adjacency key: `adj_split:<edge_type>:<dir>:<node_id BE>:<start_uid BE>`.
///
/// Split keys reference a specific part of a super-node's posting list.
/// The `start_uid` identifies which split fragment this key points to.
pub fn encode_adj_split_key(
    edge_type: &str,
    direction: AdjDirection,
    node_id: NodeId,
    start_uid: u64,
) -> Vec<u8> {
    let dir_str = match direction {
        AdjDirection::Out => ":out:",
        AdjDirection::In => ":in:",
    };
    let mut key = Vec::with_capacity(10 + edge_type.len() + dir_str.len() + 8 + 8);
    key.extend_from_slice(b"adj_split:");
    key.extend_from_slice(edge_type.as_bytes());
    key.extend_from_slice(dir_str.as_bytes());
    key.extend_from_slice(&node_id.as_raw().to_be_bytes());
    key.push(b':');
    key.extend_from_slice(&start_uid.to_be_bytes());
    key
}

// -- Edge property key encoding --

/// Encode an edge property key: `edgeprop:<edge_type>:<source_id BE>:<target_id BE>`.
///
/// Edge properties (facets) are stored separately from the posting list,
/// keyed by the specific edge `(type, source, target)`. This allows updating
/// edge properties without rewriting the entire posting list.
pub fn encode_edgeprop_key(edge_type: &str, source_id: NodeId, target_id: NodeId) -> Vec<u8> {
    let mut key = Vec::with_capacity(9 + edge_type.len() + 1 + 8 + 1 + 8);
    key.extend_from_slice(b"edgeprop:");
    key.extend_from_slice(edge_type.as_bytes());
    key.push(b':');
    key.extend_from_slice(&source_id.as_raw().to_be_bytes());
    key.push(b':');
    key.extend_from_slice(&target_id.as_raw().to_be_bytes());
    key
}

/// Decode an edge property key into its components.
///
/// Returns `None` if the key doesn't match the expected format.
pub fn decode_edgeprop_key(key: &[u8]) -> Option<(String, NodeId, NodeId)> {
    if !key.starts_with(b"edgeprop:") {
        return None;
    }
    // Last 17 bytes: src(8) + ':' + tgt(8)
    if key.len() < 9 + 1 + 8 + 1 + 8 {
        return None;
    }
    let src_start = key.len() - 8 - 1 - 8;
    let tgt_start = key.len() - 8;

    // Check separator between src and tgt
    if key[src_start + 8] != b':' {
        return None;
    }

    let edge_type_end = src_start - 1; // position of ':' before src
    if key[edge_type_end] != b':' {
        return None;
    }

    let edge_type = std::str::from_utf8(&key[9..edge_type_end]).ok()?;
    let source_id = u64::from_be_bytes(key[src_start..src_start + 8].try_into().ok()?);
    let target_id = u64::from_be_bytes(key[tgt_start..tgt_start + 8].try_into().ok()?);

    Some((
        edge_type.to_string(),
        NodeId::from_raw(source_id),
        NodeId::from_raw(target_id),
    ))
}

// -- Temporal edge property keys --
//
// Temporal edges keep one edgeprop entry per valid_from. The key is the
// regular `(type, src, tgt)` triple followed by a sortable i64-BE encoding
// of the validity start. Sorting by `valid_from` lets a single prefix scan
// of `edgeprop:<TYPE>:<src>:<tgt>:` enumerate every version of the edge in
// chronological order, and a bounded range scan with `valid_from_upper_bound_key`
// answers "active at time T" queries in O(log versions).

/// Encode `valid_from` (Unix epoch milliseconds, signed) as 8 bytes sorted
/// lexicographically ascending by numeric value. Negative values use sign-flip
/// so timestamps before the epoch sort before positive ones; the result is a
/// total order matching `i64` comparison.
pub(crate) fn encode_valid_from_sortable(valid_from_ms: i64) -> [u8; 8] {
    ((valid_from_ms as u64) ^ (1u64 << 63)).to_be_bytes()
}

pub(crate) fn decode_valid_from_sortable(bytes: [u8; 8]) -> i64 {
    (u64::from_be_bytes(bytes) ^ (1u64 << 63)) as i64
}

/// Encode a temporal edge property key:
/// `edgeprop:<edge_type>:<source_id BE>:<target_id BE>:<valid_from sortable>`.
///
/// One entry per version. Multiple versions of the same `(src, tgt)` pair
/// coexist; an "active at T" query prefix-scans `temporal_edgeprop_pair_prefix`
/// and stops at `valid_from_upper_bound_key`.
pub fn encode_temporal_edgeprop_key(
    edge_type: &str,
    source_id: NodeId,
    target_id: NodeId,
    valid_from_ms: i64,
) -> Vec<u8> {
    let mut key = Vec::with_capacity(9 + edge_type.len() + 1 + 8 + 1 + 8 + 1 + 8);
    key.extend_from_slice(b"edgeprop:");
    key.extend_from_slice(edge_type.as_bytes());
    key.push(b':');
    key.extend_from_slice(&source_id.as_raw().to_be_bytes());
    key.push(b':');
    key.extend_from_slice(&target_id.as_raw().to_be_bytes());
    key.push(b':');
    key.extend_from_slice(&encode_valid_from_sortable(valid_from_ms));
    key
}

/// Decode a temporal edge property key into `(edge_type, src, tgt, valid_from)`.
/// Returns `None` if the key isn't well-formed.
pub fn decode_temporal_edgeprop_key(key: &[u8]) -> Option<(String, NodeId, NodeId, i64)> {
    if !key.starts_with(b"edgeprop:") {
        return None;
    }
    // Tail: src(8) + ':' + tgt(8) + ':' + valid_from(8) = 27 bytes.
    if key.len() < 9 + 1 + 8 + 1 + 8 + 1 + 8 {
        return None;
    }
    let vf_start = key.len() - 8;
    let tgt_start = vf_start - 1 - 8;
    let src_start = tgt_start - 1 - 8;
    if key[src_start + 8] != b':' || key[tgt_start + 8] != b':' {
        return None;
    }
    let edge_type_end = src_start - 1;
    if key[edge_type_end] != b':' {
        return None;
    }

    let edge_type = std::str::from_utf8(&key[9..edge_type_end]).ok()?;
    let source_id = u64::from_be_bytes(key[src_start..src_start + 8].try_into().ok()?);
    let target_id = u64::from_be_bytes(key[tgt_start..tgt_start + 8].try_into().ok()?);
    let valid_from_bytes: [u8; 8] = key[vf_start..vf_start + 8].try_into().ok()?;
    let valid_from = decode_valid_from_sortable(valid_from_bytes);

    Some((
        edge_type.to_string(),
        NodeId::from_raw(source_id),
        NodeId::from_raw(target_id),
        valid_from,
    ))
}

/// Prefix matching every temporal edgeprop entry for a given `(type, src, tgt)`
/// pair: `edgeprop:<TYPE>:<src>:<tgt>:`. Use for full-version enumeration.
pub fn temporal_edgeprop_pair_prefix(
    edge_type: &str,
    source_id: NodeId,
    target_id: NodeId,
) -> Vec<u8> {
    let mut key = Vec::with_capacity(9 + edge_type.len() + 1 + 8 + 1 + 8 + 1);
    key.extend_from_slice(b"edgeprop:");
    key.extend_from_slice(edge_type.as_bytes());
    key.push(b':');
    key.extend_from_slice(&source_id.as_raw().to_be_bytes());
    key.push(b':');
    key.extend_from_slice(&target_id.as_raw().to_be_bytes());
    key.push(b':');
    key
}

/// Exclusive upper bound for a prefix scan that collects every version with
/// `valid_from <= upper_ms`. Suffix encodes `upper_ms + 1` in the sortable form
/// so the bound itself doesn't include `upper_ms + 1` but does include all
/// values `<= upper_ms`.
///
/// Saturates at `i64::MAX`: if asked for an unbounded scan, the caller should
/// instead use `temporal_edgeprop_pair_prefix` plus the LSM range's natural end.
pub fn valid_from_upper_bound_key(
    edge_type: &str,
    source_id: NodeId,
    target_id: NodeId,
    upper_ms: i64,
) -> Vec<u8> {
    let bound = upper_ms.saturating_add(1);
    encode_temporal_edgeprop_key(edge_type, source_id, target_id, bound)
}

// -- Discriminated edge property keys (ADR-029) --
//
// A `DISCRIMINATED BY (col)` edge type stores one edgeprop entry per discriminator
// value, keyed `edgeprop:<TYPE>:<src>:<tgt>:<discriminator>`. The discriminator is
// the LAST key component, so its encoding is order-preserving: a literal-equality
// predicate is a point lookup, a range predicate is a bounded prefix scan, and a
// bare pair prefix (`temporal_edgeprop_pair_prefix`, which is discriminator-agnostic)
// enumerates every instance. Adj posting set semantics are unchanged — a target
// appears iff at least one instance exists.
//
// TEMPORAL is the special case `DISCRIMINATED BY (valid_from)`: a `Timestamp`
// discriminator encodes to the same i64 sortable bytes as `valid_from`, so the two
// share one storage shape (no parallel path). `temporal_edgeprop_key_matches_disc`
// in the tests proves the byte-identity.

/// Order-preserving 8-byte encoding of an `f64` (total order matching `f64`
/// comparison, NaN sorting last). Positive: set the sign bit; negative: flip all
/// bits. The inverse of [`decode_f64_sortable`].
pub(crate) fn encode_f64_sortable(v: f64) -> [u8; 8] {
    let bits = v.to_bits();
    let sortable = if bits & (1u64 << 63) == 0 {
        bits | (1u64 << 63)
    } else {
        !bits
    };
    sortable.to_be_bytes()
}

pub(crate) fn decode_f64_sortable(bytes: [u8; 8]) -> f64 {
    let sortable = u64::from_be_bytes(bytes);
    let bits = if sortable & (1u64 << 63) != 0 {
        sortable & !(1u64 << 63)
    } else {
        !sortable
    };
    f64::from_bits(bits)
}

/// Encode a discriminator value as an order-preserving key suffix (ADR-029).
///
/// Supported discriminator types: `Int` / `Timestamp` (8-byte sign-flipped BE,
/// byte-identical to temporal `valid_from`), `Float` (8-byte sortable), `Bool`
/// (1 byte), `String` (raw UTF-8) and `Blob` (32-byte SHA-256 of the blob bytes).
/// `String` uses raw UTF-8, not a length prefix: the discriminator is the last key
/// component (so the encoding is unambiguous without one) and raw UTF-8 keeps
/// byte-order == value-order, which the planner relies on for range push-down on
/// the discriminator column. Returns `None` for any other (unsupported) `Value`.
pub fn encode_discriminator_value(value: &PropertyValue) -> Option<Vec<u8>> {
    match value {
        PropertyValue::Int(v) | PropertyValue::Timestamp(v) => {
            Some(encode_valid_from_sortable(*v).to_vec())
        }
        PropertyValue::Float(f) => Some(encode_f64_sortable(*f).to_vec()),
        PropertyValue::Bool(b) => Some(vec![u8::from(*b)]),
        PropertyValue::String(s) => Some(s.as_bytes().to_vec()),
        PropertyValue::Blob(bytes) => {
            use sha2::{Digest, Sha256};
            Some(Sha256::digest(bytes).to_vec())
        }
        _ => None,
    }
}

/// Decode a discriminator key suffix back into a `Value`, given the declared
/// discriminator [`PropertyType`] (read from the edge type schema by the caller —
/// the store stays schema-agnostic). `Blob` is one-way (the suffix is a SHA-256,
/// not the original bytes); it decodes to `Value::Blob(<32-byte digest>)`,
/// sufficient for equality but not for recovering the original blob.
pub fn decode_discriminator_value(bytes: &[u8], kind: &PropertyType) -> Option<PropertyValue> {
    match kind {
        PropertyType::Int => Some(PropertyValue::Int(decode_valid_from_sortable(
            bytes.try_into().ok()?,
        ))),
        PropertyType::Timestamp => Some(PropertyValue::Timestamp(decode_valid_from_sortable(
            bytes.try_into().ok()?,
        ))),
        PropertyType::Float => Some(PropertyValue::Float(decode_f64_sortable(
            bytes.try_into().ok()?,
        ))),
        PropertyType::Bool => match bytes {
            [0] => Some(PropertyValue::Bool(false)),
            [1] => Some(PropertyValue::Bool(true)),
            _ => None,
        },
        PropertyType::String => Some(PropertyValue::String(
            std::str::from_utf8(bytes).ok()?.to_string(),
        )),
        PropertyType::Blob => Some(PropertyValue::Blob(bytes.to_vec())),
        _ => None,
    }
}

/// Encode a discriminated edge property key:
/// `edgeprop:<edge_type>:<src BE>:<tgt BE>:<discriminator>`.
///
/// Returns `None` if `discriminator` is not an ADR-029-supported type.
pub fn encode_discriminated_edgeprop_key(
    edge_type: &str,
    source_id: NodeId,
    target_id: NodeId,
    discriminator: &PropertyValue,
) -> Option<Vec<u8>> {
    let suffix = encode_discriminator_value(discriminator)?;
    let mut key = temporal_edgeprop_pair_prefix(edge_type, source_id, target_id);
    key.extend_from_slice(&suffix);
    Some(key)
}

/// Decode a discriminated edge property key into
/// `(edge_type, src, tgt, discriminator)`, given the discriminator's declared type.
///
/// Forward-parses (edge type names are colon-free identifiers) so a variable-length
/// `String` discriminator decodes unambiguously.
pub fn decode_discriminated_edgeprop_key(
    key: &[u8],
    kind: &PropertyType,
) -> Option<(String, NodeId, NodeId, PropertyValue)> {
    let rest = key.strip_prefix(b"edgeprop:")?;
    let type_end = rest.iter().position(|&b| b == b':')?;
    let edge_type = std::str::from_utf8(&rest[..type_end]).ok()?.to_string();
    let tail = &rest[type_end + 1..];
    // src(8) ':' tgt(8) ':' <discriminator>
    if tail.len() < 8 + 1 + 8 + 1 {
        return None;
    }
    if tail[8] != b':' || tail[17] != b':' {
        return None;
    }
    let source_id = u64::from_be_bytes(tail[0..8].try_into().ok()?);
    let target_id = u64::from_be_bytes(tail[9..17].try_into().ok()?);
    let discriminator = decode_discriminator_value(&tail[18..], kind)?;
    Some((
        edge_type,
        NodeId::from_raw(source_id),
        NodeId::from_raw(target_id),
        discriminator,
    ))
}

// -- Edge properties (facets) --

/// Properties on a specific edge (facets).
///
/// Stored separately from the posting list at key `edgeprop:<TYPE>:<src>:<tgt>`.
/// Uses interned field IDs (u32 keys) and `PropertyValue` values,
/// same as `NodeRecord` for consistency.
///
/// Supported types: string, int64, float64, bool, datetime (as string),
/// vector, binary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EdgeProperties {
    /// Properties keyed by interned field ID.
    pub props: HashMap<u32, PropertyValue>,
}

impl EdgeProperties {
    /// Create empty edge properties.
    pub fn new() -> Self {
        Self {
            props: HashMap::new(),
        }
    }

    /// Set a property by interned field ID.
    pub fn set(&mut self, field_id: u32, value: PropertyValue) {
        self.props.insert(field_id, value);
    }

    /// Get a property by interned field ID.
    pub fn get(&self, field_id: u32) -> Option<&PropertyValue> {
        self.props.get(&field_id)
    }

    /// Remove a property.
    pub fn remove(&mut self, field_id: u32) -> Option<PropertyValue> {
        self.props.remove(&field_id)
    }

    /// Number of properties.
    pub fn len(&self) -> usize {
        self.props.len()
    }

    /// Whether there are no properties.
    pub fn is_empty(&self) -> bool {
        self.props.is_empty()
    }

    /// Serialize to the canonical edge-property wire format (ADR-040).
    pub fn to_msgpack(&self) -> Result<Vec<u8>, rmp_serde::encode::Error> {
        let pairs: Vec<(u32, PropertyValue)> =
            self.props.iter().map(|(k, v)| (*k, v.clone())).collect();
        encode_edge_props(&pairs)
    }

    /// Deserialize from the canonical edge-property wire format (ADR-040).
    pub fn from_msgpack(data: &[u8]) -> Result<Self, rmp_serde::decode::Error> {
        let props = decode_edge_props(data)?.into_iter().collect();
        Ok(Self { props })
    }
}

impl Default for EdgeProperties {
    fn default() -> Self {
        Self::new()
    }
}

/// Encode an edge-property facet set to the single canonical on-disk wire
/// format (ADR-040): a MessagePack array of `(field_id, value)` pairs
/// sorted ascending by `field_id`, duplicate field ids collapsed last-wins.
///
/// This is the ONLY edge-property value encoder in the tree. Every writer
/// — the query executor, [`EdgeProperties::to_msgpack`], backup restore —
/// routes through it so the `edgeprop:` partition holds exactly one byte
/// layout (no dual-format readers, per ADR-021). Sorting makes the bytes
/// deterministic: the same logical facet set always yields the same bytes
/// regardless of insertion order or `HashMap` iteration order, which is a
/// hard requirement for page-ECC, block dedup, and snapshot diffing.
pub fn encode_edge_props(
    props: &[(u32, PropertyValue)],
) -> Result<Vec<u8>, rmp_serde::encode::Error> {
    // BTreeMap gives ascending-by-field_id order and last-wins dedup in one
    // pass; the typical edge carries 1–5 facets, so the cost is negligible.
    let ordered: std::collections::BTreeMap<u32, &PropertyValue> =
        props.iter().map(|(k, v)| (*k, v)).collect();
    let pairs: Vec<(u32, &PropertyValue)> = ordered.into_iter().collect();
    rmp_serde::to_vec(&pairs)
}

/// Decode an edge-property facet set from the canonical wire format
/// (ADR-040). Counterpart to [`encode_edge_props`]. Returns the pairs in
/// stored (sorted-by-`field_id`) order.
pub fn decode_edge_props(
    bytes: &[u8],
) -> Result<Vec<(u32, PropertyValue)>, rmp_serde::decode::Error> {
    rmp_serde::from_slice(bytes)
}

/// Direction of an adjacency key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdjDirection {
    Out,
    In,
}

/// Decoded adjacency key components.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdjKeyParts {
    pub edge_type: String,
    pub direction: AdjDirection,
    pub node_id: NodeId,
}

/// Decode an adjacency key into its components.
///
/// Returns `None` if the key doesn't match the expected format.
pub fn decode_adj_key(key: &[u8]) -> Option<AdjKeyParts> {
    if !key.starts_with(b"adj:") {
        return None;
    }
    // Key must end with 8 bytes of node ID
    if key.len() < 4 + 1 + 4 + 8 {
        return None;
    }

    // The last 8 bytes are the node ID (binary, big-endian)
    let id_bytes = &key[key.len() - 8..];
    let node_id = u64::from_be_bytes(id_bytes.try_into().ok()?);

    // The middle part (between "adj:" and the 8-byte ID) contains edge_type + direction
    let middle = &key[4..key.len() - 8];
    let middle_str = std::str::from_utf8(middle).ok()?;

    if let Some(edge_type) = middle_str.strip_suffix(":out:") {
        return Some(AdjKeyParts {
            edge_type: edge_type.to_string(),
            direction: AdjDirection::Out,
            node_id: NodeId::from_raw(node_id),
        });
    }

    if let Some(edge_type) = middle_str.strip_suffix(":in:") {
        return Some(AdjKeyParts {
            edge_type: edge_type.to_string(),
            direction: AdjDirection::In,
            node_id: NodeId::from_raw(node_id),
        });
    }

    None
}

// -- Posting list --

/// A sorted list of node IDs representing adjacency (edges).
///
/// In-memory representation is `Vec<u64>` for O(log n) insert/remove.
/// On-disk format uses group-varint UidPack encoding (2-4x compression).
/// Split posting lists for super-nodes (>512KB) are tracked separately.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct PostingList {
    /// Sorted UIDs in ascending order.
    uids: Vec<u64>,
}

impl PostingList {
    /// Create an empty posting list.
    pub fn new() -> Self {
        Self { uids: Vec::new() }
    }

    /// Create a posting list from an already-sorted slice.
    ///
    /// # Panics
    ///
    /// Debug-asserts that the input is sorted and deduplicated.
    pub fn from_sorted(uids: Vec<u64>) -> Self {
        debug_assert!(
            uids.windows(2).all(|w| w[0] < w[1]),
            "uids must be sorted and unique"
        );
        Self { uids }
    }

    /// Insert a UID into the posting list (maintains sorted order).
    ///
    /// Returns `true` if the UID was newly inserted, `false` if already present.
    pub fn insert(&mut self, uid: u64) -> bool {
        match self.uids.binary_search(&uid) {
            Ok(_) => false, // already present
            Err(pos) => {
                self.uids.insert(pos, uid);
                true
            }
        }
    }

    /// Remove a UID from the posting list.
    ///
    /// Returns `true` if the UID was found and removed.
    pub fn remove(&mut self, uid: u64) -> bool {
        match self.uids.binary_search(&uid) {
            Ok(pos) => {
                self.uids.remove(pos);
                true
            }
            Err(_) => false,
        }
    }

    /// Check if a UID is in the posting list.
    pub fn contains(&self, uid: u64) -> bool {
        self.uids.binary_search(&uid).is_ok()
    }

    /// Number of UIDs in the posting list.
    pub fn len(&self) -> usize {
        self.uids.len()
    }

    /// Whether the posting list is empty.
    pub fn is_empty(&self) -> bool {
        self.uids.is_empty()
    }

    /// Iterator over UIDs in sorted order.
    pub fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        self.uids.iter().copied()
    }

    /// Get the underlying sorted UIDs slice.
    pub fn as_slice(&self) -> &[u64] {
        &self.uids
    }

    /// Serialize to compact binary format (StreamVByte UidPack encoding, V5).
    ///
    /// UIDs are delta-encoded in 256-UID blocks with StreamVByte Coder1234,
    /// achieving 2-4x compression compared to raw u64 encoding with
    /// SIMD-accelerated batch decode.
    pub fn to_bytes(&self) -> Result<Vec<u8>, rmp_serde::encode::Error> {
        let pack = super::codec::encode_uids(&self.uids);
        rmp_serde::to_vec(&pack)
    }

    /// Deserialize from compact binary format (group-varint UidPack encoding).
    pub fn from_bytes(data: &[u8]) -> Result<Self, rmp_serde::decode::Error> {
        let pack: super::codec::UidPack = rmp_serde::from_slice(data)?;
        let uids = super::codec::decode_uids(&pack);
        Ok(Self { uids })
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
