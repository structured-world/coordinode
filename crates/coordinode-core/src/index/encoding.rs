//! Order-preserving, injective key encoding for index values, and the key
//! layouts of the index partition built from it.
//!
//! A value encodes as a type tag followed by its payload. Every element is
//! self-delimiting, so a tuple of values is the plain concatenation of its
//! elements and decodes one way: two distinct tuples never share an
//! encoding, and no complete tuple is a proper prefix of another. Byte order
//! is value order within a type and tag order across types:
//! NULL < false < true < integers < floats < strings < timestamps < binary.
//!
//! Strings and byte strings end in `0x00`; a `0x00` inside is written
//! `0x00 0xFF`, which no terminator can be followed by (every tag is below
//! `0xFF`). This is the escaping of the FoundationDB tuple layer, used for the
//! same reason: the encoding stays injective and keeps byte order.

use crate::graph::types::Value;

/// Type tags, in the order the types sort.
const TAG_NULL: u8 = 0x05;
const TAG_FALSE: u8 = 0x10;
const TAG_TRUE: u8 = 0x11;
const TAG_INT: u8 = 0x20;
const TAG_FLOAT: u8 = 0x30;
const TAG_STRING: u8 = 0x40;
const TAG_TIMESTAMP: u8 = 0x50;
const TAG_BINARY: u8 = 0x60;

/// Prefix of the entries of an index that allows several nodes per value.
const INDEX_PREFIX: &[u8] = b"idx:";
/// Prefix of the entries of a unique index: one entry per value.
const UNIQUE_PREFIX: &[u8] = b"uidx:";

/// A value that has no index key.
///
/// Such a value is not indexed: an equality on it is answered by a scan,
/// which applies the query's own comparison. NaN is here because it equals
/// nothing, itself included, so no index entry could ever be the right answer
/// to an equality on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unindexable {
    /// A float NaN.
    NaN,
    /// A value of a composite or opaque type, named.
    Kind(&'static str),
}

impl core::fmt::Display for Unindexable {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NaN => f.write_str("NaN"),
            Self::Kind(kind) => f.write_str(kind),
        }
    }
}

/// Append the encoding of `value` to `out`.
///
/// # Errors
///
/// [`Unindexable`] for NaN and for values of a type that has no key.
pub fn encode_element(value: &Value, out: &mut Vec<u8>) -> Result<(), Unindexable> {
    match value {
        Value::Null => out.push(TAG_NULL),
        Value::Bool(false) => out.push(TAG_FALSE),
        Value::Bool(true) => out.push(TAG_TRUE),
        Value::Int(n) => {
            out.push(TAG_INT);
            out.extend_from_slice(&((*n as u64) ^ (1 << 63)).to_be_bytes());
        }
        Value::Float(f) => {
            if f.is_nan() {
                return Err(Unindexable::NaN);
            }
            // -0.0 and 0.0 are one key: they compare equal.
            let bits = if *f == 0.0 { 0 } else { f.to_bits() };
            let ordered = if *f >= 0.0 { bits ^ (1 << 63) } else { !bits };
            out.push(TAG_FLOAT);
            out.extend_from_slice(&ordered.to_be_bytes());
        }
        Value::String(s) => {
            out.push(TAG_STRING);
            escape(s.as_bytes(), out);
        }
        Value::Timestamp(t) => {
            out.push(TAG_TIMESTAMP);
            out.extend_from_slice(&((*t as u64) ^ (1 << 63)).to_be_bytes());
        }
        Value::Binary(b) => {
            out.push(TAG_BINARY);
            escape(b, out);
        }
        Value::Vector(_) | Value::MultiVector(_) => return Err(Unindexable::Kind("a vector")),
        Value::Blob(_) => return Err(Unindexable::Kind("a blob")),
        Value::Array(_) => return Err(Unindexable::Kind("a list")),
        Value::Map(_) | Value::Document(_) => return Err(Unindexable::Kind("a map")),
        Value::Geo(_) => return Err(Unindexable::Kind("a geographic value")),
        Value::Path(_) => return Err(Unindexable::Kind("a path")),
    }
    Ok(())
}

/// Encode a tuple of values: the concatenation of their elements.
///
/// # Errors
///
/// [`Unindexable`] when any value has no key.
pub fn encode_tuple(values: &[Value]) -> Result<Vec<u8>, Unindexable> {
    let mut out = Vec::with_capacity(values.len() * 10);
    for value in values {
        encode_element(value, &mut out)?;
    }
    Ok(out)
}

fn escape(bytes: &[u8], out: &mut Vec<u8>) {
    for &b in bytes {
        out.push(b);
        if b == 0 {
            out.push(0xFF);
        }
    }
    out.push(0);
}

/// `<prefix><name length u32 BE><name>:`. The length makes the name
/// self-delimiting whatever it contains; the trailing `:` makes the prefix a
/// boundary the partition's prefix bloom filter indexes.
fn named_prefix(prefix: &[u8], name: &str) -> Vec<u8> {
    // Index names are identifiers bounded far below 4 GiB by the parser.
    let len = name.len() as u32;
    let mut out = Vec::with_capacity(prefix.len() + 4 + name.len() + 1);
    out.extend_from_slice(prefix);
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(name.as_bytes());
    out.push(b':');
    out
}

/// Prefix of every entry of the non-unique index `name`.
pub fn index_prefix(name: &str) -> Vec<u8> {
    named_prefix(INDEX_PREFIX, name)
}

/// Prefix of the entries of the non-unique index `name` holding `tuple`
/// (an [`encode_tuple`] result): `idx:<name>:<tuple>:`.
///
/// Only entries of exactly this tuple carry it: the tuple is self-delimiting
/// and every entry of one index has the same arity.
pub fn index_value_prefix(name: &str, tuple: &[u8]) -> Vec<u8> {
    let mut out = index_prefix(name);
    out.extend_from_slice(tuple);
    out.push(b':');
    out
}

/// Entry of `node_id` under `tuple` in the non-unique index `name`:
/// `idx:<name>:<tuple>:<node_id u64 BE>`, empty value.
pub fn encode_index_key(name: &str, tuple: &[u8], node_id: u64) -> Vec<u8> {
    let mut out = index_value_prefix(name, tuple);
    out.extend_from_slice(&node_id.to_be_bytes());
    out
}

/// Prefix of every entry of the unique index `name`.
pub fn unique_index_prefix(name: &str) -> Vec<u8> {
    named_prefix(UNIQUE_PREFIX, name)
}

/// Entry of `tuple` in the unique index `name`: `uidx:<name>:<tuple>`, whose
/// value is the holder's node id (u64 BE).
///
/// Keyed by the value alone, the entry is its own uniqueness claim: two
/// transactions inserting one value write one key, and write-write conflict
/// detection lets one of them commit.
pub fn encode_unique_index_key(name: &str, tuple: &[u8]) -> Vec<u8> {
    let mut out = unique_index_prefix(name);
    out.extend_from_slice(tuple);
    out
}

/// Node id in the trailing 8 bytes of a non-unique entry key, or in the value
/// of a unique entry.
pub fn decode_node_id(bytes: &[u8]) -> Option<u64> {
    let tail: [u8; 8] = bytes.get(bytes.len().checked_sub(8)?..)?.try_into().ok()?;
    Some(u64::from_be_bytes(tail))
}

/// Prefix of every entry the index `name` wrote under the layout that
/// preceded [`encode_index_key`]: `idx:<name>:`. Disjoint from the current
/// layouts, whose name is length-prefixed and so starts with a zero byte for
/// any name a parser accepts.
pub fn legacy_index_prefix(name: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(INDEX_PREFIX.len() + name.len() + 1);
    out.extend_from_slice(INDEX_PREFIX);
    out.extend_from_slice(name.as_bytes());
    out.push(b':');
    out
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
