//! Type-preserving binary key encoding for index values.
//!
//! Encodes Values into a binary format that preserves ordering under
//! lexicographic byte comparison. This is critical for B-tree indexes
//! stored in the LSM sorted keyspace.

use crate::graph::types::Value;

/// Type tag bytes for ordering: Null < Bool < Int < Float < String.
const TAG_NULL: u8 = 0x00;
const TAG_BOOL: u8 = 0x10;
const TAG_INT: u8 = 0x20;
const TAG_FLOAT: u8 = 0x30;
const TAG_STRING: u8 = 0x40;
const TAG_TIMESTAMP: u8 = 0x50;

/// Encode a Value into a binary-comparable key.
///
/// The encoding preserves ordering: NULL < FALSE < TRUE < integers < floats < strings.
/// Integers and floats use sign-flipped big-endian encoding for correct ordering.
pub fn encode_value(value: &Value) -> Vec<u8> {
    match value {
        Value::Null => vec![TAG_NULL],
        Value::Bool(b) => vec![TAG_BOOL, if *b { 1 } else { 0 }],
        Value::Int(n) => {
            let mut buf = Vec::with_capacity(9);
            buf.push(TAG_INT);
            // Flip sign bit so negative values sort before positive
            let encoded = (*n as u64) ^ (1u64 << 63);
            buf.extend_from_slice(&encoded.to_be_bytes());
            buf
        }
        Value::Float(f) => {
            let mut buf = Vec::with_capacity(9);
            buf.push(TAG_FLOAT);
            // IEEE 754 float encoding with sign-bit handling for correct ordering
            let bits = f.to_bits();
            let encoded = if *f >= 0.0 {
                bits ^ (1u64 << 63) // positive: flip sign bit
            } else {
                !bits // negative: flip all bits
            };
            buf.extend_from_slice(&encoded.to_be_bytes());
            buf
        }
        Value::String(s) => {
            let mut buf = Vec::with_capacity(1 + s.len() + 1);
            buf.push(TAG_STRING);
            // Use raw UTF-8 bytes — lexicographic ordering matches string ordering
            buf.extend_from_slice(s.as_bytes());
            buf.push(0x00); // Null terminator for proper prefix ordering
            buf
        }
        Value::Timestamp(t) => {
            let mut buf = Vec::with_capacity(9);
            buf.push(TAG_TIMESTAMP);
            let encoded = (*t as u64) ^ (1u64 << 63);
            buf.extend_from_slice(&encoded.to_be_bytes());
            buf
        }
        // Other types are not indexable via B-tree
        _ => vec![TAG_NULL],
    }
}

/// Encode a compound key from multiple values.
///
/// Each value is encoded sequentially with a separator byte (0xFF) between them.
/// This preserves the multi-column ordering: first column is primary sort,
/// second column is secondary sort within identical first-column values, etc.
pub fn encode_compound_value(values: &[Value]) -> Vec<u8> {
    if values.len() == 1 {
        return encode_value(&values[0]);
    }

    let mut buf = Vec::new();
    for (i, value) in values.iter().enumerate() {
        if i > 0 {
            buf.push(0xFF); // Separator between compound fields
        }
        buf.extend_from_slice(&encode_value(value));
    }
    buf
}

/// Build a full index entry key: `idx:<name>:<encoded_value>:<node_id>`.
pub fn encode_index_key(index_name: &str, value: &Value, node_id: u64) -> Vec<u8> {
    let encoded_value = encode_value(value);
    let mut key = Vec::with_capacity(4 + index_name.len() + 1 + encoded_value.len() + 1 + 8);
    key.extend_from_slice(b"idx:");
    key.extend_from_slice(index_name.as_bytes());
    key.push(b':');
    key.extend_from_slice(&encoded_value);
    key.push(b':');
    key.extend_from_slice(&node_id.to_be_bytes());
    key
}

/// Build a compound index entry key: `idx:<name>:<encoded_values>:<node_id>`.
pub fn encode_compound_index_key(index_name: &str, values: &[Value], node_id: u64) -> Vec<u8> {
    let encoded = encode_compound_value(values);
    let mut key = Vec::with_capacity(4 + index_name.len() + 1 + encoded.len() + 1 + 8);
    key.extend_from_slice(b"idx:");
    key.extend_from_slice(index_name.as_bytes());
    key.push(b':');
    key.extend_from_slice(&encoded);
    key.push(b':');
    key.extend_from_slice(&node_id.to_be_bytes());
    key
}

/// Decode a node_id from the last 8 bytes of an index key.
pub fn decode_node_id_from_index_key(key: &[u8]) -> Option<u64> {
    if key.len() < 8 {
        return None;
    }
    let id_bytes = &key[key.len() - 8..];
    Some(u64::from_be_bytes([
        id_bytes[0],
        id_bytes[1],
        id_bytes[2],
        id_bytes[3],
        id_bytes[4],
        id_bytes[5],
        id_bytes[6],
        id_bytes[7],
    ]))
}

#[cfg(test)]
mod tests;
