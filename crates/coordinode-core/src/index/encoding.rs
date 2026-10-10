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

use super::identity::GenerationId;
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

/// First byte of an entry of an index that allows several nodes per value.
const TAG_ENTRIES: u8 = 0x01;
/// First byte of an entry of a unique index: one entry per value.
const TAG_UNIQUE_ENTRIES: u8 = 0x02;
/// Ends the tuple of a non-unique entry. No element starts with it, so it
/// marks where the tuple ends and the owner begins.
const TUPLE_END: u8 = b':';
/// Length of a generation prefix: the shape tag and the generation, u64 BE.
pub const GENERATION_PREFIX_LEN: usize = 9;
/// First bytes of the index partition keys that belong to a generation: a key
/// starting with one of them is `tag / GenerationId:u64_BE / …`. No other
/// family of the index partition starts with these bytes.
pub const GENERATION_TAGS: [u8; 2] = [TAG_ENTRIES, TAG_UNIQUE_ENTRIES];

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

fn generation_prefix(tag: u8, generation: GenerationId) -> [u8; GENERATION_PREFIX_LEN] {
    let mut out = [0u8; GENERATION_PREFIX_LEN];
    out[0] = tag;
    out[1..].copy_from_slice(&generation.as_raw().to_be_bytes());
    out
}

/// Prefix of every entry of the non-unique index generation `generation`:
/// the shape tag and the generation. Fixed width, so it ends where the tuple
/// begins whatever the generation is.
pub fn entries_prefix(generation: GenerationId) -> [u8; GENERATION_PREFIX_LEN] {
    generation_prefix(TAG_ENTRIES, generation)
}

/// Prefix of the entries of the non-unique `generation` holding `tuple` (an
/// [`encode_tuple`] result): `<prefix><tuple>:`.
///
/// Only entries of exactly this tuple carry it: the tuple is self-delimiting
/// and every entry of one index has the same arity.
pub fn entry_value_prefix(generation: GenerationId, tuple: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(GENERATION_PREFIX_LEN + tuple.len() + 17);
    out.extend_from_slice(&entries_prefix(generation));
    out.extend_from_slice(tuple);
    out.push(TUPLE_END);
    out
}

/// Entry of `node_id` under `tuple` in the non-unique `generation`:
/// `<prefix><tuple>:<node_id u64 BE>`, empty value.
pub fn encode_entry_key(generation: GenerationId, tuple: &[u8], node_id: u64) -> Vec<u8> {
    let mut out = entry_value_prefix(generation, tuple);
    out.extend_from_slice(&node_id.to_be_bytes());
    out
}

/// Entry of the version of temporal node `node_id` that starts at
/// `valid_from` under `tuple` in the non-unique `generation`:
/// `<prefix><tuple>:<node_id u64 BE><valid_from>`, empty value, with
/// `valid_from` in the order-preserving form of an integer. A node's
/// versions sort together, oldest first.
pub fn encode_version_entry_key(
    generation: GenerationId,
    tuple: &[u8],
    node_id: u64,
    valid_from: i64,
) -> Vec<u8> {
    let mut out = encode_entry_key(generation, tuple, node_id);
    out.extend_from_slice(&((valid_from as u64) ^ (1 << 63)).to_be_bytes());
    out
}

/// Prefix of every entry of the unique index generation `generation`.
pub fn unique_entries_prefix(generation: GenerationId) -> [u8; GENERATION_PREFIX_LEN] {
    generation_prefix(TAG_UNIQUE_ENTRIES, generation)
}

/// Entry of `tuple` in the unique `generation`: `<prefix><tuple>`, whose
/// value is the holder's node id (u64 BE).
///
/// Keyed by the value alone, the entry is its own uniqueness claim: two
/// transactions inserting one value write one key, and write-write conflict
/// detection lets one of them commit.
pub fn encode_unique_entry_key(generation: GenerationId, tuple: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(GENERATION_PREFIX_LEN + tuple.len());
    out.extend_from_slice(&unique_entries_prefix(generation));
    out.extend_from_slice(tuple);
    out
}

/// The key ranges, `[start, end)`, holding every entry of `generation` in
/// either shape: what removing a generation's entries removes.
pub fn generation_ranges(generation: GenerationId) -> [(Vec<u8>, Vec<u8>); 2] {
    [
        entries_prefix(generation),
        unique_entries_prefix(generation),
    ]
    .map(|prefix| (prefix.to_vec(), prefix_end(&prefix)))
}

/// Smallest key strictly greater than every key starting with `prefix`. The
/// prefixes here start with a tag below `0xFF`, so an end always exists.
fn prefix_end(prefix: &[u8]) -> Vec<u8> {
    let mut end = prefix.to_vec();
    while let Some(last) = end.pop() {
        if last < 0xFF {
            end.push(last + 1);
            break;
        }
    }
    end
}

/// The scan prefixes of an index partition key a prefix filter records:
/// its generation prefix and, for a non-unique entry, the prefix through its
/// tuple. Both are structural boundaries, found by decoding the shape and
/// the tuple, never by searching for a separator byte, which a value or a
/// generation number may contain. `None` for a key of no entry shape.
pub fn entry_scan_prefixes(key: &[u8]) -> Option<impl Iterator<Item = &[u8]>> {
    let tag = *key.first()?;
    if (tag != TAG_ENTRIES && tag != TAG_UNIQUE_ENTRIES) || key.len() < GENERATION_PREFIX_LEN {
        return None;
    }
    let value = (tag == TAG_ENTRIES)
        .then(|| tuple_len(&key[GENERATION_PREFIX_LEN..]))
        .flatten()
        .map(|len| &key[..GENERATION_PREFIX_LEN + len + 1]);
    Some(core::iter::once(&key[..GENERATION_PREFIX_LEN]).chain(value))
}

/// The owner a non-unique entry key of `generation` names: the node id, and
/// the version's `valid_from` for an entry of one version of a temporal node
/// ([`encode_version_entry_key`]). `None` for a key that is no entry of that
/// generation.
pub fn decode_entry(generation: GenerationId, key: &[u8]) -> Option<(u64, Option<i64>)> {
    decode_entry_parts(generation, key).map(|(_, node_id, valid_from)| (node_id, valid_from))
}

/// The tuple (as [`encode_tuple`] returns it) and the owner a non-unique
/// entry key of `generation` names. `None` for a key that is no entry of
/// that generation.
pub fn decode_entry_parts(
    generation: GenerationId,
    key: &[u8],
) -> Option<(&[u8], u64, Option<i64>)> {
    let rest = key.strip_prefix(entries_prefix(generation).as_slice())?;
    let len = tuple_len(rest)?;
    let owner = rest.get(len..)?.strip_prefix(&[TUPLE_END])?;
    let node_id = u64::from_be_bytes(owner.get(..8)?.try_into().ok()?);
    let valid_from = match owner.len() {
        8 => None,
        16 => {
            let raw = u64::from_be_bytes(owner.get(8..)?.try_into().ok()?);
            Some((raw ^ (1 << 63)) as i64)
        }
        _ => return None,
    };
    Some((&rest[..len], node_id, valid_from))
}

/// The tuple a unique entry key of `generation` holds. `None` for a key that
/// is no unique entry of that generation.
pub fn decode_unique_entry_tuple(generation: GenerationId, key: &[u8]) -> Option<&[u8]> {
    key.strip_prefix(unique_entries_prefix(generation).as_slice())
}

/// Length of the encoded tuple `bytes` starts with: the elements up to the
/// [`TUPLE_END`] that ends it, which is no element tag. `None` when an
/// element is malformed or the tuple does not end.
fn tuple_len(bytes: &[u8]) -> Option<usize> {
    let mut at = 0;
    loop {
        match *bytes.get(at)? {
            TUPLE_END => return Some(at),
            TAG_NULL | TAG_FALSE | TAG_TRUE => at += 1,
            TAG_INT | TAG_FLOAT | TAG_TIMESTAMP => at += 9,
            TAG_STRING | TAG_BINARY => {
                at += 1;
                // Escaped payload: a `0x00` not followed by `0xFF` ends it.
                loop {
                    let byte = *bytes.get(at)?;
                    at += 1;
                    if byte == 0 {
                        if bytes.get(at) == Some(&0xFF) {
                            at += 1;
                        } else {
                            break;
                        }
                    }
                }
            }
            _ => return None,
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
