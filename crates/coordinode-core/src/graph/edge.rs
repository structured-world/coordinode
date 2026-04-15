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

    /// Serialize to MessagePack bytes.
    pub fn to_msgpack(&self) -> Result<Vec<u8>, rmp_serde::encode::Error> {
        rmp_serde::to_vec(&self.props)
    }

    /// Deserialize from MessagePack bytes.
    pub fn from_msgpack(data: &[u8]) -> Result<Self, rmp_serde::decode::Error> {
        let props: HashMap<u32, PropertyValue> = rmp_serde::from_slice(data)?;
        Ok(Self { props })
    }
}

impl Default for EdgeProperties {
    fn default() -> Self {
        Self::new()
    }
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
mod tests {
    use super::*;

    // -- Key encoding tests --

    #[test]
    fn encode_forward_key() {
        let key = encode_adj_key_forward("FOLLOWS", NodeId::from_raw(42));
        assert!(key.starts_with(b"adj:FOLLOWS:out:"));
    }

    #[test]
    fn encode_reverse_key() {
        let key = encode_adj_key_reverse("FOLLOWS", NodeId::from_raw(99));
        assert!(key.starts_with(b"adj:FOLLOWS:in:"));
    }

    #[test]
    fn write_forward_key_matches_encode() {
        // write_ variant must produce identical bytes as encode_ variant
        let mut buf = Vec::new();
        write_adj_key_forward("KNOWS", NodeId::from_raw(7), &mut buf);
        assert_eq!(buf, encode_adj_key_forward("KNOWS", NodeId::from_raw(7)));
    }

    #[test]
    fn write_reverse_key_matches_encode() {
        let mut buf = Vec::new();
        write_adj_key_reverse("LIKES", NodeId::from_raw(13), &mut buf);
        assert_eq!(buf, encode_adj_key_reverse("LIKES", NodeId::from_raw(13)));
    }

    #[test]
    fn write_key_reuses_buffer_across_calls() {
        // Buffer grows to fit largest key and does not reallocate on smaller subsequent writes.
        let mut buf = Vec::new();

        write_adj_key_forward("VERY_LONG_EDGE_TYPE_NAME", NodeId::from_raw(1), &mut buf);
        let cap_after_first = buf.capacity();
        assert!(
            cap_after_first >= 4 + 24 + 5 + 8,
            "buffer must fit first key"
        );

        write_adj_key_forward("KNOWS", NodeId::from_raw(2), &mut buf);
        // Capacity must not decrease (no reallocation for shorter key)
        assert_eq!(buf.capacity(), cap_after_first);
        assert_eq!(buf, encode_adj_key_forward("KNOWS", NodeId::from_raw(2)));
    }

    #[test]
    fn write_key_clears_before_write() {
        let mut buf = Vec::new();
        write_adj_key_forward("FOLLOWS", NodeId::from_raw(1), &mut buf);
        let first = buf.clone();
        write_adj_key_forward("KNOWS", NodeId::from_raw(2), &mut buf);
        // Buffer must contain only the second key, not a concatenation
        assert_ne!(buf, first);
        assert_eq!(buf, encode_adj_key_forward("KNOWS", NodeId::from_raw(2)));
    }

    #[test]
    fn decode_forward_key_roundtrip() {
        let key = encode_adj_key_forward("LIKES", NodeId::from_raw(123));
        let parts = decode_adj_key(&key).expect("decode failed");
        assert_eq!(parts.edge_type, "LIKES");
        assert_eq!(parts.direction, AdjDirection::Out);
        assert_eq!(parts.node_id, NodeId::from_raw(123));
    }

    #[test]
    fn decode_reverse_key_roundtrip() {
        let key = encode_adj_key_reverse("LIKES", NodeId::from_raw(456));
        let parts = decode_adj_key(&key).expect("decode failed");
        assert_eq!(parts.edge_type, "LIKES");
        assert_eq!(parts.direction, AdjDirection::In);
        assert_eq!(parts.node_id, NodeId::from_raw(456));
    }

    #[test]
    fn forward_keys_sort_by_source_id() {
        let k1 = encode_adj_key_forward("FOLLOWS", NodeId::from_raw(100));
        let k2 = encode_adj_key_forward("FOLLOWS", NodeId::from_raw(200));
        assert!(k1 < k2);
    }

    #[test]
    fn forward_keys_sort_by_edge_type() {
        let k1 = encode_adj_key_forward("AAA", NodeId::from_raw(1));
        let k2 = encode_adj_key_forward("ZZZ", NodeId::from_raw(1));
        assert!(k1 < k2);
    }

    #[test]
    fn decode_invalid_key() {
        assert!(decode_adj_key(b"").is_none());
        assert!(decode_adj_key(b"node:something").is_none());
        assert!(decode_adj_key(b"adj:FOLLOWS:bad:12345678").is_none());
    }

    // -- PostingList tests --

    #[test]
    fn posting_list_empty() {
        let pl = PostingList::new();
        assert!(pl.is_empty());
        assert_eq!(pl.len(), 0);
    }

    #[test]
    fn posting_list_insert_maintains_order() {
        let mut pl = PostingList::new();
        assert!(pl.insert(30));
        assert!(pl.insert(10));
        assert!(pl.insert(20));
        assert_eq!(pl.as_slice(), &[10, 20, 30]);
    }

    #[test]
    fn posting_list_insert_duplicate() {
        let mut pl = PostingList::new();
        assert!(pl.insert(10));
        assert!(!pl.insert(10)); // duplicate
        assert_eq!(pl.len(), 1);
    }

    #[test]
    fn posting_list_remove() {
        let mut pl = PostingList::new();
        pl.insert(10);
        pl.insert(20);
        pl.insert(30);

        assert!(pl.remove(20));
        assert_eq!(pl.as_slice(), &[10, 30]);
        assert!(!pl.remove(20)); // already removed
    }

    #[test]
    fn posting_list_contains() {
        let mut pl = PostingList::new();
        pl.insert(42);
        assert!(pl.contains(42));
        assert!(!pl.contains(99));
    }

    #[test]
    fn posting_list_iter() {
        let mut pl = PostingList::new();
        pl.insert(3);
        pl.insert(1);
        pl.insert(2);
        let collected: Vec<u64> = pl.iter().collect();
        assert_eq!(collected, vec![1, 2, 3]);
    }

    #[test]
    fn posting_list_from_sorted() {
        let pl = PostingList::from_sorted(vec![1, 5, 10, 100]);
        assert_eq!(pl.len(), 4);
        assert!(pl.contains(5));
        assert!(!pl.contains(6));
    }

    #[test]
    fn posting_list_uidpack_roundtrip() {
        let mut pl = PostingList::new();
        pl.insert(100);
        pl.insert(200);
        pl.insert(300);

        let bytes = pl.to_bytes().expect("serialize");
        let restored = PostingList::from_bytes(&bytes).expect("deserialize");
        assert_eq!(pl, restored);
    }

    #[test]
    fn posting_list_empty_roundtrip() {
        let pl = PostingList::new();
        let bytes = pl.to_bytes().expect("serialize");
        let restored = PostingList::from_bytes(&bytes).expect("deserialize");
        assert_eq!(pl, restored);
    }

    #[test]
    fn posting_list_uidpack_compression() {
        // Verify UidPack is smaller than raw Vec<u64> MessagePack
        let mut pl = PostingList::new();
        for i in 0..500u64 {
            pl.insert(i * 3); // sequential-ish UIDs with small deltas
        }
        let uidpack_bytes = pl.to_bytes().expect("serialize");
        let raw_msgpack = rmp_serde::to_vec(pl.as_slice()).expect("raw");
        assert!(
            uidpack_bytes.len() < raw_msgpack.len(),
            "UidPack ({} bytes) should be smaller than raw msgpack ({} bytes)",
            uidpack_bytes.len(),
            raw_msgpack.len()
        );
    }

    #[test]
    fn posting_list_uidpack_large_roundtrip() {
        // Test with >256 UIDs (multiple blocks) + MSB boundary crossing
        let mut pl = PostingList::new();
        for i in 0..1000u64 {
            pl.insert(i * 7 + 1);
        }
        let bytes = pl.to_bytes().expect("serialize");
        let restored = PostingList::from_bytes(&bytes).expect("deserialize");
        assert_eq!(pl, restored);
    }

    #[test]
    fn posting_list_large() {
        let mut pl = PostingList::new();
        for i in (0..1000u64).rev() {
            pl.insert(i);
        }
        assert_eq!(pl.len(), 1000);
        assert_eq!(pl.as_slice()[0], 0);
        assert_eq!(pl.as_slice()[999], 999);
    }

    // -- Forward + Reverse symmetry --

    #[test]
    fn forward_and_reverse_keys_are_different() {
        let fwd = encode_adj_key_forward("FOLLOWS", NodeId::from_raw(42));
        let rev = encode_adj_key_reverse("FOLLOWS", NodeId::from_raw(42));
        assert_ne!(fwd, rev);
    }

    // -- Split key tests --

    #[test]
    fn split_key_encoding() {
        let key = encode_adj_split_key("FOLLOWS", AdjDirection::Out, NodeId::from_raw(42), 1000);
        assert!(key.starts_with(b"adj_split:FOLLOWS:out:"));
    }

    #[test]
    fn split_keys_sort_by_start_uid() {
        let k1 = encode_adj_split_key("FOLLOWS", AdjDirection::Out, NodeId::from_raw(42), 100);
        let k2 = encode_adj_split_key("FOLLOWS", AdjDirection::Out, NodeId::from_raw(42), 200);
        assert!(k1 < k2, "split keys should sort by start_uid");
    }

    #[test]
    fn split_key_differs_from_main_key() {
        let main = encode_adj_key_forward("FOLLOWS", NodeId::from_raw(42));
        let split = encode_adj_split_key("FOLLOWS", AdjDirection::Out, NodeId::from_raw(42), 1);
        assert_ne!(main, split);
        assert!(split.starts_with(b"adj_split:"));
    }

    #[test]
    fn forward_keyed_by_source_reverse_keyed_by_target() {
        // Forward: source=42, targets=[1,2,3]
        let fwd_key = encode_adj_key_forward("FOLLOWS", NodeId::from_raw(42));
        let fwd_parts = decode_adj_key(&fwd_key).expect("decode");
        assert_eq!(fwd_parts.node_id, NodeId::from_raw(42));
        assert_eq!(fwd_parts.direction, AdjDirection::Out);

        // Reverse: target=1, sources=[42]
        let rev_key = encode_adj_key_reverse("FOLLOWS", NodeId::from_raw(1));
        let rev_parts = decode_adj_key(&rev_key).expect("decode");
        assert_eq!(rev_parts.node_id, NodeId::from_raw(1));
        assert_eq!(rev_parts.direction, AdjDirection::In);
    }

    // -- Edge property key tests --

    #[test]
    fn edgeprop_key_encoding() {
        let key = encode_edgeprop_key("KNOWS", NodeId::from_raw(42), NodeId::from_raw(99));
        assert!(key.starts_with(b"edgeprop:KNOWS:"));
    }

    #[test]
    fn edgeprop_key_roundtrip() {
        let key = encode_edgeprop_key("WORKS_AT", NodeId::from_raw(100), NodeId::from_raw(200));
        let (edge_type, src, tgt) = decode_edgeprop_key(&key).expect("decode failed");
        assert_eq!(edge_type, "WORKS_AT");
        assert_eq!(src, NodeId::from_raw(100));
        assert_eq!(tgt, NodeId::from_raw(200));
    }

    #[test]
    fn edgeprop_key_sorting() {
        // Same edge type, different source → sort by source
        let k1 = encode_edgeprop_key("KNOWS", NodeId::from_raw(1), NodeId::from_raw(99));
        let k2 = encode_edgeprop_key("KNOWS", NodeId::from_raw(2), NodeId::from_raw(99));
        assert!(k1 < k2);
    }

    #[test]
    fn edgeprop_key_decode_invalid() {
        assert!(decode_edgeprop_key(b"").is_none());
        assert!(decode_edgeprop_key(b"adj:FOLLOWS:out:12345678").is_none());
        assert!(decode_edgeprop_key(b"edgeprop:short").is_none());
    }

    // -- EdgeProperties tests --

    #[test]
    fn edge_properties_new() {
        let ep = EdgeProperties::new();
        assert!(ep.is_empty());
        assert_eq!(ep.len(), 0);
    }

    #[test]
    fn edge_properties_set_get_remove() {
        let mut ep = EdgeProperties::new();
        ep.set(1, PropertyValue::String("2024-01-01".into()));
        ep.set(2, PropertyValue::Float(0.8));

        assert_eq!(ep.get(1), Some(&PropertyValue::String("2024-01-01".into())));
        assert_eq!(ep.get(2), Some(&PropertyValue::Float(0.8)));
        assert_eq!(ep.len(), 2);

        let removed = ep.remove(1);
        assert_eq!(removed, Some(PropertyValue::String("2024-01-01".into())));
        assert_eq!(ep.len(), 1);
    }

    #[test]
    fn edge_properties_msgpack_roundtrip() {
        let mut ep = EdgeProperties::new();
        ep.set(1, PropertyValue::String("since".into()));
        ep.set(2, PropertyValue::Float(0.95));
        ep.set(3, PropertyValue::Int(42));
        ep.set(4, PropertyValue::Bool(true));
        ep.set(5, PropertyValue::Null);
        ep.set(6, PropertyValue::Binary(vec![0xCA, 0xFE]));

        let bytes = ep.to_msgpack().expect("serialize");
        let restored = EdgeProperties::from_msgpack(&bytes).expect("deserialize");
        assert_eq!(ep, restored);
    }

    #[test]
    fn edge_properties_empty_roundtrip() {
        let ep = EdgeProperties::new();
        let bytes = ep.to_msgpack().expect("serialize");
        let restored = EdgeProperties::from_msgpack(&bytes).expect("deserialize");
        assert_eq!(ep, restored);
    }

    #[test]
    fn edge_properties_all_types() {
        let mut ep = EdgeProperties::new();
        ep.set(1, PropertyValue::String("hello".into()));
        ep.set(2, PropertyValue::Int(i64::MIN));
        ep.set(3, PropertyValue::Float(f64::MAX));
        ep.set(4, PropertyValue::Bool(false));
        ep.set(5, PropertyValue::Null);
        ep.set(6, PropertyValue::Binary(vec![]));
        ep.set(
            7,
            PropertyValue::Array(vec![
                PropertyValue::Int(1),
                PropertyValue::String("two".into()),
            ]),
        );

        let bytes = ep.to_msgpack().expect("serialize");
        let restored = EdgeProperties::from_msgpack(&bytes).expect("deserialize");
        assert_eq!(ep, restored);
    }
}
