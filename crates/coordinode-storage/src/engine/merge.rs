//! Merge operators for LSM storage keyspaces.
//!
//! Three merge operators for different partitions:
//!
//! - **`PostingListMerge`** (`adj:` partition): conflict-free concurrent edge writes
//!   via Add/Remove delta operands on sorted UID posting lists.
//!
//! - **`DocumentMerge`** (`node:` partition): path-targeted partial document updates
//!   via `DocDelta` operands (SetPath, DeletePath, ArrayPush, etc.). Eliminates
//!   read-modify-write for nested DOCUMENT properties. See ADR-015.
//!
//! - **`CounterMerge`** (`counter:` partition): atomic i64 increment/decrement.
//!   Base value is i64 LE, operands are i64 LE deltas. Result = base + sum(deltas).
//!   Used for node degree cache, analytics counters. Conflict-free by construction.

use coordinode_core::graph::doc_delta::{
    DocDelta, PathTarget, PREFIX_DOC_DELTA, PREFIX_NODE_RECORD,
};
use coordinode_core::graph::edge::PostingList;
use coordinode_core::graph::node::NodeRecord;
use lsm_tree::{Error as LsmError, MergeOperator, UserValue};

/// Tag bytes for merge operand encoding.
const TAG_ADD: u8 = 0x01;
const TAG_REMOVE: u8 = 0x02;
const TAG_ADD_BATCH: u8 = 0x03;

/// Encode a single Add operand: `[TAG_ADD, uid_be_bytes(8)]`.
pub fn encode_add(uid: u64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(9);
    buf.push(TAG_ADD);
    buf.extend_from_slice(&uid.to_be_bytes());
    buf
}

/// Encode a single Remove operand: `[TAG_REMOVE, uid_be_bytes(8)]`.
pub fn encode_remove(uid: u64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(9);
    buf.push(TAG_REMOVE);
    buf.extend_from_slice(&uid.to_be_bytes());
    buf
}

/// Encode a batch Add operand: `[TAG_ADD_BATCH, count_be(4), uid_be(8), ...]`.
pub fn encode_add_batch(uids: &[u64]) -> Vec<u8> {
    let count = uids.len() as u32;
    let mut buf = Vec::with_capacity(1 + 4 + uids.len() * 8);
    buf.push(TAG_ADD_BATCH);
    buf.extend_from_slice(&count.to_be_bytes());
    for &uid in uids {
        buf.extend_from_slice(&uid.to_be_bytes());
    }
    buf
}

/// Merge operator for adjacency posting lists (`adj:` keyspace).
///
/// Operand format:
///   - `[0x01, uid(8B)]`        — Add single UID
///   - `[0x02, uid(8B)]`        — Remove single UID
///   - `[0x03, count(4B), uid(8B)×N]` — Add batch
///
/// Base value: UidPack-encoded posting list (group-varint, from `PostingList::to_bytes`).
/// Result: UidPack-encoded posting list with operands applied in order.
pub struct PostingListMerge;

impl MergeOperator for PostingListMerge {
    fn merge(
        &self,
        _key: &[u8],
        base_value: Option<&[u8]>,
        operands: &[&[u8]],
    ) -> Result<UserValue, LsmError> {
        // Decode the base posting list (or start empty).
        let mut plist = match base_value {
            Some(data) if !data.is_empty() => {
                PostingList::from_bytes(data).map_err(|_| LsmError::MergeOperator)?
            }
            _ => PostingList::new(),
        };

        // Apply each operand in chronological order.
        for operand in operands {
            if operand.is_empty() {
                return Err(LsmError::MergeOperator);
            }

            match operand[0] {
                TAG_ADD => {
                    if operand.len() != 9 {
                        return Err(LsmError::MergeOperator);
                    }
                    let uid = u64::from_be_bytes(
                        operand[1..9]
                            .try_into()
                            .map_err(|_| LsmError::MergeOperator)?,
                    );
                    plist.insert(uid);
                }
                TAG_REMOVE => {
                    if operand.len() != 9 {
                        return Err(LsmError::MergeOperator);
                    }
                    let uid = u64::from_be_bytes(
                        operand[1..9]
                            .try_into()
                            .map_err(|_| LsmError::MergeOperator)?,
                    );
                    plist.remove(uid);
                }
                TAG_ADD_BATCH => {
                    if operand.len() < 5 {
                        return Err(LsmError::MergeOperator);
                    }
                    let count = u32::from_be_bytes(
                        operand[1..5]
                            .try_into()
                            .map_err(|_| LsmError::MergeOperator)?,
                    ) as usize;
                    let expected_len = 5 + count * 8;
                    if operand.len() != expected_len {
                        return Err(LsmError::MergeOperator);
                    }
                    for i in 0..count {
                        let offset = 5 + i * 8;
                        let uid = u64::from_be_bytes(
                            operand[offset..offset + 8]
                                .try_into()
                                .map_err(|_| LsmError::MergeOperator)?,
                        );
                        plist.insert(uid);
                    }
                }
                _ => {
                    // Pre-merged PostingList from a previous partial compaction.
                    // LSM compaction stores partial merge results as MergeOperand
                    // entries. On subsequent compaction, these pre-merged PostingLists
                    // appear as operands (not base_value). Decode and merge contents.
                    let prev =
                        PostingList::from_bytes(operand).map_err(|_| LsmError::MergeOperator)?;
                    for &uid in prev.as_slice() {
                        plist.insert(uid);
                    }
                }
            }
        }

        // Re-encode as UidPack (group-varint compressed).
        let bytes = plist.to_bytes().map_err(|_| LsmError::MergeOperator)?;
        Ok(bytes.into())
    }
}

// ---------------------------------------------------------------------------
// Document merge operator (node: partition)
// ---------------------------------------------------------------------------

/// Merge operator for node records with DOCUMENT properties (`node:` keyspace).
///
/// Values in the `node:` partition use a prefix-byte discriminator:
/// - `0x00` + msgpack — full `NodeRecord` (from PUT / CREATE / SET whole node)
/// - `0x01` + msgpack — `DocDelta` merge operand (from partial path update)
///
/// During compaction, the merge function decodes the base NodeRecord and
/// applies all DocDelta operands in seqno order, producing a single
/// merged NodeRecord with prefix `0x00`.
///
/// Pre-merged NodeRecords (from previous compaction) may appear as operands —
/// the merge function detects them by the `0x00` prefix and uses the latest
/// one as the new base, discarding older bases.
pub struct DocumentMerge;

impl MergeOperator for DocumentMerge {
    fn merge(
        &self,
        _key: &[u8],
        base_value: Option<&[u8]>,
        operands: &[&[u8]],
    ) -> Result<UserValue, LsmError> {
        // Decode the base NodeRecord (from a PUT), or start with None.
        let mut record = match base_value {
            Some(data) if !data.is_empty() => Some(decode_node_record(data)?),
            _ => None,
        };

        // Lazily-initialized rmpv representation of rec.extra for batching.
        //
        // Extra-targeting path deltas (SetPath, DeletePath, ArrayPush, Increment)
        // all require HashMap → rmpv → HashMap conversion. Processing them one by one
        // costs one full round-trip per delta. By accumulating changes in an in-progress
        // rmpv::Value, consecutive Extra deltas share a single extra_to_rmpv + rmpv_to_extra
        // pair regardless of count. The doc is lazily created on first Extra delta and
        // flushed back to rec.extra only when the record base resets or at the end.
        let mut extra_doc: Option<rmpv::Value> = None;

        // Apply each operand in chronological (seqno) order.
        for operand in operands {
            if operand.is_empty() {
                return Err(LsmError::MergeOperator);
            }

            match operand[0] {
                PREFIX_NODE_RECORD => {
                    // Flush pending Extra doc into the current record before reset.
                    if let (Some(doc), Some(rec)) = (extra_doc.take(), &mut record) {
                        rmpv_to_extra(rec, &doc);
                    }
                    // Pre-merged NodeRecord from a previous compaction or a new PUT.
                    // This supersedes the current base — use it as the new base.
                    record = Some(decode_node_record(operand)?);
                }
                PREFIX_DOC_DELTA => {
                    let delta =
                        DocDelta::decode(&operand[1..]).map_err(|_| LsmError::MergeOperator)?;

                    // Ensure we have a base record. If the original PUT was compacted
                    // away but merge operands remain, create a minimal empty record.
                    let rec = record.get_or_insert_with(|| NodeRecord::new(""));

                    apply_delta_to_record_batched(rec, &delta, &mut extra_doc);
                }
                _ => {
                    // Unknown prefix — flush and treat as legacy NodeRecord.
                    if let (Some(doc), Some(rec)) = (extra_doc.take(), &mut record) {
                        rmpv_to_extra(rec, &doc);
                    }
                    let rec =
                        NodeRecord::from_msgpack(operand).map_err(|_| LsmError::MergeOperator)?;
                    record = Some(rec);
                }
            }
        }

        // Final flush: write accumulated Extra doc back to rec.extra.
        if let (Some(doc), Some(rec)) = (extra_doc, &mut record) {
            rmpv_to_extra(rec, &doc);
        }

        // Encode the merged result with the 0x00 prefix.
        let rec = record.unwrap_or_else(|| NodeRecord::new(""));
        encode_node_record(&rec)
    }
}

/// Decode a NodeRecord from storage bytes.
/// Handles both prefixed (0x00 + msgpack) and legacy bare msgpack formats.
fn decode_node_record(data: &[u8]) -> Result<NodeRecord, LsmError> {
    if data.is_empty() {
        return Err(LsmError::MergeOperator);
    }

    if data[0] == PREFIX_NODE_RECORD {
        // Prefixed format: skip the 0x00 byte.
        NodeRecord::from_msgpack(&data[1..]).map_err(|_| LsmError::MergeOperator)
    } else {
        // Legacy format: bare msgpack without prefix.
        NodeRecord::from_msgpack(data).map_err(|_| LsmError::MergeOperator)
    }
}

/// Encode a NodeRecord with the 0x00 prefix byte.
fn encode_node_record(rec: &NodeRecord) -> Result<UserValue, LsmError> {
    let msgpack = rec.to_msgpack().map_err(|_| LsmError::MergeOperator)?;
    let mut buf = Vec::with_capacity(1 + msgpack.len());
    buf.push(PREFIX_NODE_RECORD);
    buf.extend_from_slice(&msgpack);
    Ok(buf.into())
}

/// Apply a DocDelta to a NodeRecord, accumulating Extra-targeting deltas into
/// an in-progress rmpv document to avoid redundant round-trips.
///
/// `extra_doc` is the caller-owned lazy representation of `rec.extra` as an
/// `rmpv::Value::Map`. When `Some`, Extra-targeting deltas are applied directly
/// to it without re-serializing `rec.extra`. When `None`, it is lazily created
/// on the first Extra delta via `extra_to_rmpv`.
///
/// **The caller is responsible for flushing `extra_doc` back to `rec.extra`**
/// (via `rmpv_to_extra`) after all deltas are processed and before any base
/// record reset. A typical call site looks like:
///
/// ```ignore
/// let mut extra_doc = None;
/// for delta in deltas {
///     apply_delta_to_record_batched(rec, delta, &mut extra_doc);
/// }
/// if let Some(doc) = extra_doc.take() {
///     rmpv_to_extra(rec, &doc);
/// }
/// ```
///
/// ## Batch effect
///
/// With N Extra-targeting deltas, the cost is:
/// - **Before**: N × (extra_to_rmpv + delta.apply + rmpv_to_extra)
/// - **After**: 1 × extra_to_rmpv + N × delta.apply + 1 × rmpv_to_extra
///
/// PropField and RemoveProperty deltas are always applied immediately and
/// do not interact with `extra_doc`.
fn apply_delta_to_record_batched(
    rec: &mut NodeRecord,
    delta: &DocDelta,
    extra_doc: &mut Option<rmpv::Value>,
) {
    // RemoveProperty operates on the NodeRecord or in-progress doc directly.
    if let DocDelta::RemoveProperty { target, key } = delta {
        match target {
            PathTarget::PropField(field_id) => {
                rec.props.remove(field_id);
            }
            PathTarget::Extra => {
                match extra_doc {
                    Some(doc) => {
                        // Apply removal to the in-progress rmpv doc to preserve order.
                        if let (rmpv::Value::Map(entries), Some(k)) = (doc, key) {
                            let rmpv_key = rmpv::Value::String(rmpv::Utf8String::from(k.as_str()));
                            entries.retain(|(ek, _)| ek != &rmpv_key);
                        }
                    }
                    None => {
                        // No pending doc — apply directly to rec.extra.
                        if let (Some(extra), Some(k)) = (&mut rec.extra, key) {
                            extra.remove(k);
                            if extra.is_empty() {
                                rec.extra = None;
                            }
                        }
                    }
                }
            }
        }
        return;
    }

    match delta.target() {
        PathTarget::Extra => {
            // Lazily initialize the rmpv doc from rec.extra on first Extra delta.
            // Subsequent Extra deltas reuse the same doc — no redundant round-trip.
            let doc = extra_doc.get_or_insert_with(|| extra_to_rmpv(rec));
            delta.apply(doc);
        }
        PathTarget::PropField(field_id) => {
            // PropField deltas operate on rec.props and don't interact with
            // rec.extra or extra_doc — no flush needed.
            let mut doc = match rec.props.get(field_id) {
                Some(v) => value_to_rmpv(v),
                None => rmpv::Value::Map(Vec::new()),
            };
            delta.apply(&mut doc);
            rec.set(*field_id, rmpv_to_value(&doc));
        }
    }
}

/// Convert NodeRecord's extra map to an rmpv::Value::Map for delta application.
fn extra_to_rmpv(rec: &NodeRecord) -> rmpv::Value {
    match &rec.extra {
        Some(extra) => {
            let entries: Vec<(rmpv::Value, rmpv::Value)> = extra
                .iter()
                .map(|(k, v)| {
                    let key = rmpv::Value::String(k.as_str().into());
                    let val = value_to_rmpv(v);
                    (key, val)
                })
                .collect();
            rmpv::Value::Map(entries)
        }
        None => rmpv::Value::Map(Vec::new()),
    }
}

/// Write back an rmpv::Value::Map to NodeRecord's extra map.
fn rmpv_to_extra(rec: &mut NodeRecord, doc: &rmpv::Value) {
    if let rmpv::Value::Map(entries) = doc {
        if entries.is_empty() {
            rec.extra = None;
        } else {
            let mut extra = std::collections::HashMap::new();
            for (k, v) in entries {
                if let rmpv::Value::String(key) = k {
                    if let Some(s) = key.as_str() {
                        extra.insert(s.to_string(), rmpv_to_value(v));
                    }
                }
            }
            rec.extra = Some(extra);
        }
    }
}

/// Convert coordinode Value to rmpv::Value for merge processing.
fn value_to_rmpv(v: &coordinode_core::graph::types::Value) -> rmpv::Value {
    match v {
        coordinode_core::graph::types::Value::Null => rmpv::Value::Nil,
        coordinode_core::graph::types::Value::Bool(b) => rmpv::Value::Boolean(*b),
        coordinode_core::graph::types::Value::Int(i) => rmpv::Value::Integer((*i).into()),
        coordinode_core::graph::types::Value::Float(f) => rmpv::Value::F64(*f),
        coordinode_core::graph::types::Value::String(s) => rmpv::Value::String(s.as_str().into()),
        coordinode_core::graph::types::Value::Document(d) => d.clone(),
        coordinode_core::graph::types::Value::Array(arr) => {
            rmpv::Value::Array(arr.iter().map(value_to_rmpv).collect())
        }
        _ => rmpv::Value::Nil, // Other types not relevant for document merge.
    }
}

/// Convert rmpv::Value back to coordinode Value.
fn rmpv_to_value(v: &rmpv::Value) -> coordinode_core::graph::types::Value {
    match v {
        rmpv::Value::Nil => coordinode_core::graph::types::Value::Null,
        rmpv::Value::Boolean(b) => coordinode_core::graph::types::Value::Bool(*b),
        rmpv::Value::Integer(i) => {
            coordinode_core::graph::types::Value::Int(i.as_i64().unwrap_or(0))
        }
        rmpv::Value::F32(f) => coordinode_core::graph::types::Value::Float(f64::from(*f)),
        rmpv::Value::F64(f) => coordinode_core::graph::types::Value::Float(*f),
        rmpv::Value::String(s) => {
            coordinode_core::graph::types::Value::String(s.as_str().unwrap_or("").to_string())
        }
        rmpv::Value::Array(_) | rmpv::Value::Map(_) => {
            coordinode_core::graph::types::Value::Document(v.clone())
        }
        rmpv::Value::Binary(b) => coordinode_core::graph::types::Value::Binary(b.clone()),
        rmpv::Value::Ext(_, _) => coordinode_core::graph::types::Value::Null,
    }
}

// ---------------------------------------------------------------------------
// Posting list merge tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn merge_add_to_empty_base() {
        let op = PostingListMerge;
        let operand = encode_add(42);

        let result = op.merge(b"test", None, &[&operand]).expect("merge failed");
        let plist = PostingList::from_bytes(&result).expect("decode failed");
        assert_eq!(plist.as_slice(), &[42]);
    }

    #[test]
    fn merge_add_to_existing() {
        let op = PostingListMerge;
        let base = PostingList::from_sorted(vec![10, 20, 30]);
        let base_bytes = base.to_bytes().expect("encode failed");
        let operand = encode_add(25);

        let result = op
            .merge(b"test", Some(&base_bytes), &[&operand])
            .expect("merge failed");
        let plist = PostingList::from_bytes(&result).expect("decode failed");
        assert_eq!(plist.as_slice(), &[10, 20, 25, 30]);
    }

    #[test]
    fn merge_remove() {
        let op = PostingListMerge;
        let base = PostingList::from_sorted(vec![10, 20, 30]);
        let base_bytes = base.to_bytes().expect("encode failed");
        let operand = encode_remove(20);

        let result = op
            .merge(b"test", Some(&base_bytes), &[&operand])
            .expect("merge failed");
        let plist = PostingList::from_bytes(&result).expect("decode failed");
        assert_eq!(plist.as_slice(), &[10, 30]);
    }

    #[test]
    fn merge_remove_nonexistent() {
        let op = PostingListMerge;
        let base = PostingList::from_sorted(vec![10, 20]);
        let base_bytes = base.to_bytes().expect("encode failed");
        let operand = encode_remove(99);

        let result = op
            .merge(b"test", Some(&base_bytes), &[&operand])
            .expect("merge failed");
        let plist = PostingList::from_bytes(&result).expect("decode failed");
        assert_eq!(plist.as_slice(), &[10, 20]);
    }

    #[test]
    fn merge_add_batch() {
        let op = PostingListMerge;
        let operand = encode_add_batch(&[30, 10, 20]);

        let result = op.merge(b"test", None, &[&operand]).expect("merge failed");
        let plist = PostingList::from_bytes(&result).expect("decode failed");
        assert_eq!(plist.as_slice(), &[10, 20, 30]);
    }

    #[test]
    fn merge_multiple_operands() {
        let op = PostingListMerge;
        let add1 = encode_add(10);
        let add2 = encode_add(30);
        let add3 = encode_add(20);
        let rm = encode_remove(10);

        let result = op
            .merge(b"test", None, &[&add1, &add2, &add3, &rm])
            .expect("merge failed");
        let plist = PostingList::from_bytes(&result).expect("decode failed");
        assert_eq!(plist.as_slice(), &[20, 30]);
    }

    #[test]
    fn merge_duplicate_add_is_idempotent() {
        let op = PostingListMerge;
        let add = encode_add(42);

        let result = op
            .merge(b"test", None, &[&add, &add, &add])
            .expect("merge failed");
        let plist = PostingList::from_bytes(&result).expect("decode failed");
        assert_eq!(plist.as_slice(), &[42]);
    }

    #[test]
    fn merge_re_merge_stability() {
        // MergeOperator contract: re-merging a merged result with no operands
        // must produce identical bytes.
        let op = PostingListMerge;
        let add1 = encode_add(10);
        let add2 = encode_add(20);

        let first = op
            .merge(b"test", None, &[&add1, &add2])
            .expect("first merge failed");
        let second = op
            .merge(b"test", Some(&first), &[])
            .expect("second merge failed");
        assert_eq!(&*first, &*second, "re-merge must be stable");
    }

    #[test]
    fn merge_pre_merged_postinglist_as_operand() {
        // LSM partial compaction stores merged results as MergeOperand entries.
        // On subsequent compaction, the pre-merged PostingList appears as an
        // operand (not base_value). The merge function must detect and decode it.
        let op = PostingListMerge;

        // Simulate partial compaction: first merge produces a PostingList.
        let add1 = encode_add(10);
        let add2 = encode_add(20);
        let partial = op
            .merge(b"k", None, &[&add1, &add2])
            .expect("partial merge");

        // Now simulate a subsequent compaction where the partial result appears
        // as an operand alongside new merge operands.
        let add3 = encode_add(15);
        let result = op
            .merge(b"k", None, &[&partial, &add3])
            .expect("re-merge with pre-merged operand");
        let plist = PostingList::from_bytes(&result).expect("decode");
        assert_eq!(plist.as_slice(), &[10, 15, 20]);
    }

    #[test]
    fn merge_pre_merged_with_base_and_new_operands() {
        // Pre-merged operand + base value + new operands all together.
        let op = PostingListMerge;

        // Base value from a PUT.
        let base = PostingList::from_sorted(vec![1, 5]);
        let base_bytes = base.to_bytes().expect("encode");

        // Partial merge result from a previous compaction.
        let partial_add = encode_add(10);
        let partial = op.merge(b"k", None, &[&partial_add]).expect("partial");

        // New operand.
        let new_add = encode_add(3);

        let result = op
            .merge(b"k", Some(&base_bytes), &[&partial, &new_add])
            .expect("combined merge");
        let plist = PostingList::from_bytes(&result).expect("decode");
        assert_eq!(plist.as_slice(), &[1, 3, 5, 10]);
    }

    #[test]
    fn merge_output_is_valid_uidpack() {
        // Verify merge output is UidPack format, not raw Vec<u64> msgpack.
        let op = PostingListMerge;
        let batch = encode_add_batch(&[100, 200, 300, 400, 500]);

        let result = op.merge(b"test", None, &[&batch]).expect("merge");

        // Result must be decodable as UidPack.
        let pack: coordinode_core::graph::codec::UidPack =
            rmp_serde::from_slice(&result).expect("should be valid UidPack");
        assert_eq!(pack.total_uids(), 5);
        assert_eq!(pack.block_size, 256);

        // Decode back to UIDs for correctness.
        let uids = coordinode_core::graph::codec::decode_uids(&pack);
        assert_eq!(uids, vec![100, 200, 300, 400, 500]);
    }

    #[test]
    fn merge_uidpack_smaller_than_raw() {
        // 500 sequential UIDs with small deltas — UidPack should compress well.
        let op = PostingListMerge;
        let uids: Vec<u64> = (0..500).map(|i| i * 3 + 1).collect();
        let batch = encode_add_batch(&uids);

        let result = op.merge(b"test", None, &[&batch]).expect("merge");

        // Raw Vec<u64> msgpack would be ~4009 bytes (500 × 8 + overhead).
        let raw_size = rmp_serde::to_vec(&uids).expect("raw").len();
        assert!(
            result.len() < raw_size,
            "UidPack ({} bytes) should be smaller than raw msgpack ({} bytes)",
            result.len(),
            raw_size
        );
    }

    #[test]
    fn merge_large_posting_list_multiple_blocks() {
        // >256 UIDs forces multiple UidBlocks.
        let op = PostingListMerge;
        let uids: Vec<u64> = (0..600).collect();
        let batch = encode_add_batch(&uids);

        let result = op.merge(b"test", None, &[&batch]).expect("merge");
        let pack: coordinode_core::graph::codec::UidPack =
            rmp_serde::from_slice(&result).expect("should be valid UidPack");

        assert!(pack.blocks.len() >= 3, "600 UIDs should produce ≥3 blocks");
        assert_eq!(pack.total_uids(), 600);

        // Full roundtrip: decode and verify all UIDs present.
        let plist = PostingList::from_bytes(&result).expect("decode");
        assert_eq!(plist.len(), 600);
        assert_eq!(plist.as_slice()[0], 0);
        assert_eq!(plist.as_slice()[599], 599);
    }

    #[test]
    fn merge_pre_merged_uidpack_as_operand() {
        // After compaction, a UidPack-encoded result appears as an operand.
        // The merge function must detect it (non-standard tag byte) and decode.
        let op = PostingListMerge;

        // First merge produces a UidPack-encoded result.
        let add1 = encode_add_batch(&[10, 20, 30]);
        let partial = op.merge(b"k", None, &[&add1]).expect("partial merge");

        // Verify partial is valid UidPack (not raw tag byte).
        let first_byte = partial[0];
        assert!(
            first_byte != TAG_ADD && first_byte != TAG_REMOVE && first_byte != TAG_ADD_BATCH,
            "UidPack first byte (0x{first_byte:02x}) must not collide with operand tags"
        );

        // Now the partial UidPack appears as an operand in subsequent compaction.
        let add2 = encode_add(15);
        let result = op
            .merge(b"k", None, &[&partial, &add2])
            .expect("re-merge with UidPack operand");
        let plist = PostingList::from_bytes(&result).expect("decode");
        assert_eq!(plist.as_slice(), &[10, 15, 20, 30]);
    }

    #[test]
    fn invalid_operand_tag_returns_error() {
        let op = PostingListMerge;
        // 0xFF is not a valid tag AND not valid UidPack
        let bad = vec![0xFF, 0, 0, 0, 0, 0, 0, 0, 0];

        assert!(op.merge(b"test", None, &[&bad]).is_err());
    }

    #[test]
    fn empty_operand_returns_error() {
        let op = PostingListMerge;
        let empty: &[u8] = &[];

        assert!(op.merge(b"test", None, &[empty]).is_err());
    }

    #[test]
    fn truncated_add_operand_returns_error() {
        let op = PostingListMerge;
        let truncated = vec![TAG_ADD, 0, 0, 0]; // only 4 bytes, need 8

        assert!(op.merge(b"test", None, &[&truncated]).is_err());
    }
}

/// Proptest fuzzing for posting list merge operator.
///
/// Generates random sequences of Add/Remove operations and verifies
/// that the result is always sorted and contains exactly the expected UIDs.
#[cfg(test)]
#[allow(clippy::expect_used)]
mod proptest_merge {
    use super::*;
    use proptest::prelude::*;

    /// A single merge operation for proptest.
    #[derive(Debug, Clone)]
    enum MergeOp {
        Add(u64),
        Remove(u64),
    }

    /// Strategy: generate a vec of 1..200 Add/Remove operations
    /// with UIDs in 0..500 range to ensure overlap and collisions.
    fn merge_ops_strategy() -> impl Strategy<Value = Vec<MergeOp>> {
        prop::collection::vec(
            prop_oneof![
                (0..500u64).prop_map(MergeOp::Add),
                (0..500u64).prop_map(MergeOp::Remove),
            ],
            1..200,
        )
    }

    /// Apply operations to a reference HashSet to compute expected result.
    fn expected_uids(ops: &[MergeOp]) -> Vec<u64> {
        let mut set = std::collections::BTreeSet::new();
        for op in ops {
            match op {
                MergeOp::Add(uid) => {
                    set.insert(*uid);
                }
                MergeOp::Remove(uid) => {
                    set.remove(uid);
                }
            }
        }
        set.into_iter().collect()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(500))]

        #[test]
        fn fuzz_merge_sorted_unique(ops in merge_ops_strategy()) {
            let merger = PostingListMerge;

            // Encode all operations as merge operands.
            let operands: Vec<Vec<u8>> = ops
                .iter()
                .map(|op| match op {
                    MergeOp::Add(uid) => encode_add(*uid),
                    MergeOp::Remove(uid) => encode_remove(*uid),
                })
                .collect();
            let operand_refs: Vec<&[u8]> = operands.iter().map(|v| v.as_slice()).collect();

            // Apply all operands at once (full merge).
            let result = merger
                .merge(b"test-key", None, &operand_refs)
                .expect("merge must not fail on valid operands");
            let plist = PostingList::from_bytes(&result).expect("decode merged result");

            let expected = expected_uids(&ops);

            // Invariant 1: result matches reference set.
            prop_assert_eq!(plist.as_slice(), expected.as_slice(),
                "merged posting list does not match expected set");

            // Invariant 2: sorted (redundant with BTreeSet, but explicitly checked).
            let slice = plist.as_slice();
            for i in 1..slice.len() {
                prop_assert!(slice[i - 1] < slice[i],
                    "not sorted at {}: {} >= {}", i, slice[i-1], slice[i]);
            }
        }

        #[test]
        fn fuzz_incremental_merge_matches_full(ops in merge_ops_strategy()) {
            // Apply operands one at a time (incremental) vs all at once (full).
            // Both must produce identical result.
            let merger = PostingListMerge;

            let operands: Vec<Vec<u8>> = ops
                .iter()
                .map(|op| match op {
                    MergeOp::Add(uid) => encode_add(*uid),
                    MergeOp::Remove(uid) => encode_remove(*uid),
                })
                .collect();

            // Full merge.
            let operand_refs: Vec<&[u8]> = operands.iter().map(|v| v.as_slice()).collect();
            let full = merger
                .merge(b"k", None, &operand_refs)
                .expect("full merge");

            // Incremental merge: apply one operand at a time, feeding result as base.
            let mut base: Option<Vec<u8>> = None;
            for op in &operands {
                let b_ref = base.as_deref();
                let result = merger
                    .merge(b"k", b_ref, &[op.as_slice()])
                    .expect("incremental merge");
                base = Some(result.to_vec());
            }

            let incremental = base.unwrap_or_default();
            prop_assert_eq!(&*full, incremental.as_slice(),
                "full merge and incremental merge must produce identical output");
        }

        #[test]
        fn fuzz_batch_add_matches_individual_adds(uids in prop::collection::vec(0..1000u64, 1..100)) {
            let merger = PostingListMerge;

            // Batch add.
            let batch_op = encode_add_batch(&uids);
            let batch_result = merger
                .merge(b"k", None, &[&batch_op])
                .expect("batch merge");

            // Individual adds.
            let individual_ops: Vec<Vec<u8>> = uids.iter().map(|&u| encode_add(u)).collect();
            let individual_refs: Vec<&[u8]> = individual_ops.iter().map(|v| v.as_slice()).collect();
            let individual_result = merger
                .merge(b"k", None, &individual_refs)
                .expect("individual merge");

            prop_assert_eq!(&*batch_result, &*individual_result,
                "batch add must produce same result as individual adds");
        }
    }
}

// ---------------------------------------------------------------------------
// Document merge tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod doc_merge_tests {
    use super::*;
    use coordinode_core::graph::doc_delta::PathTarget;
    use coordinode_core::graph::types::Value;

    /// Helper: create a NodeRecord with a Document property in extra.
    fn make_record_with_doc(key: &str, doc: rmpv::Value) -> NodeRecord {
        let mut rec = NodeRecord::new("TestLabel");
        rec.set_extra(key, Value::Document(doc));
        rec
    }

    /// Helper: encode a NodeRecord as a storage value (with 0x00 prefix).
    fn encode_rec(rec: &NodeRecord) -> Vec<u8> {
        let msgpack = rec.to_msgpack().expect("encode");
        let mut buf = Vec::with_capacity(1 + msgpack.len());
        buf.push(PREFIX_NODE_RECORD);
        buf.extend_from_slice(&msgpack);
        buf
    }

    fn make_rmpv_map(entries: Vec<(&str, rmpv::Value)>) -> rmpv::Value {
        rmpv::Value::Map(
            entries
                .into_iter()
                .map(|(k, v)| (rmpv::Value::String(k.into()), v))
                .collect(),
        )
    }

    #[test]
    fn doc_merge_set_path_on_empty_base() {
        let op = DocumentMerge;
        let delta = DocDelta::SetPath {
            target: PathTarget::Extra,
            path: vec!["config".into(), "ssid".into()],
            value: rmpv::Value::String("home".into()),
        };
        let operand = delta.encode().expect("encode delta");

        let result = op.merge(b"node:0:1", None, &[&operand]).expect("merge");
        let rec = decode_node_record(&result).expect("decode result");

        // The extra map should contain config.ssid = "home".
        let doc_val = rec.get_extra("config").expect("config key");
        if let Value::Document(doc) = doc_val {
            let ssid = coordinode_core::graph::document::extract_at_path(doc, &["ssid"]);
            assert_eq!(ssid, rmpv::Value::String("home".into()));
        } else {
            panic!("expected Document, got {doc_val:?}");
        }
    }

    #[test]
    fn doc_merge_set_path_on_existing_record() {
        let op = DocumentMerge;

        let initial_doc = make_rmpv_map(vec![("ssid", rmpv::Value::String("old".into()))]);
        let rec = make_record_with_doc("config", initial_doc);
        let base = encode_rec(&rec);

        let delta = DocDelta::SetPath {
            target: PathTarget::Extra,
            path: vec!["config".into(), "ssid".into()],
            value: rmpv::Value::String("new".into()),
        };
        let operand = delta.encode().expect("encode");

        let result = op
            .merge(b"node:0:1", Some(&base), &[&operand])
            .expect("merge");
        let merged = decode_node_record(&result).expect("decode");

        let doc_val = merged.get_extra("config").expect("config key");
        if let Value::Document(doc) = doc_val {
            let ssid = coordinode_core::graph::document::extract_at_path(doc, &["ssid"]);
            assert_eq!(ssid, rmpv::Value::String("new".into()));
        } else {
            panic!("expected Document, got {doc_val:?}");
        }
    }

    #[test]
    fn doc_merge_multiple_deltas_different_paths() {
        let op = DocumentMerge;

        let delta1 = DocDelta::SetPath {
            target: PathTarget::Extra,
            path: vec!["a".into()],
            value: rmpv::Value::Integer(1.into()),
        };
        let delta2 = DocDelta::SetPath {
            target: PathTarget::Extra,
            path: vec!["b".into()],
            value: rmpv::Value::Integer(2.into()),
        };
        let op1 = delta1.encode().expect("encode");
        let op2 = delta2.encode().expect("encode");

        let result = op.merge(b"node:0:1", None, &[&op1, &op2]).expect("merge");
        let rec = decode_node_record(&result).expect("decode");

        let a = rec.get_extra("a").expect("key a");
        assert_eq!(a, &Value::Int(1));
        let b = rec.get_extra("b").expect("key b");
        assert_eq!(b, &Value::Int(2));
    }

    #[test]
    fn doc_merge_increment() {
        let op = DocumentMerge;

        let initial_doc = rmpv::Value::Integer(10.into());
        let rec = make_record_with_doc("counter", initial_doc);
        let base = encode_rec(&rec);

        let delta = DocDelta::Increment {
            target: PathTarget::Extra,
            path: vec!["counter".into()],
            amount: 5.0,
        };
        let operand = delta.encode().expect("encode");

        let result = op
            .merge(b"node:0:1", Some(&base), &[&operand])
            .expect("merge");
        let merged = decode_node_record(&result).expect("decode");

        let val = merged.get_extra("counter").expect("counter key");
        assert_eq!(val, &Value::Int(15));
    }

    #[test]
    fn doc_merge_array_push() {
        let op = DocumentMerge;

        let initial_doc = rmpv::Value::Array(vec![rmpv::Value::String("a".into())]);
        let rec = make_record_with_doc("tags", initial_doc);
        let base = encode_rec(&rec);

        let delta = DocDelta::ArrayPush {
            target: PathTarget::Extra,
            path: vec!["tags".into()],
            value: rmpv::Value::String("b".into()),
        };
        let operand = delta.encode().expect("encode");

        let result = op
            .merge(b"node:0:1", Some(&base), &[&operand])
            .expect("merge");
        let merged = decode_node_record(&result).expect("decode");

        let val = merged.get_extra("tags").expect("tags key");
        if let Value::Document(rmpv::Value::Array(arr)) = val {
            assert_eq!(arr.len(), 2);
            assert_eq!(arr[0], rmpv::Value::String("a".into()));
            assert_eq!(arr[1], rmpv::Value::String("b".into()));
        } else {
            panic!("expected Document(Array), got {val:?}");
        }
    }

    #[test]
    fn doc_merge_add_to_set_dedup() {
        let op = DocumentMerge;

        let initial_doc = rmpv::Value::Array(vec![rmpv::Value::String("a".into())]);
        let rec = make_record_with_doc("tags", initial_doc);
        let base = encode_rec(&rec);

        // Add "a" again (duplicate) and "b" (new).
        let delta1 = DocDelta::ArrayAddToSet {
            target: PathTarget::Extra,
            path: vec!["tags".into()],
            value: rmpv::Value::String("a".into()),
        };
        let delta2 = DocDelta::ArrayAddToSet {
            target: PathTarget::Extra,
            path: vec!["tags".into()],
            value: rmpv::Value::String("b".into()),
        };
        let op1 = delta1.encode().expect("encode");
        let op2 = delta2.encode().expect("encode");

        let result = op
            .merge(b"node:0:1", Some(&base), &[&op1, &op2])
            .expect("merge");
        let merged = decode_node_record(&result).expect("decode");

        let val = merged.get_extra("tags").expect("tags key");
        if let Value::Document(rmpv::Value::Array(arr)) = val {
            assert_eq!(arr.len(), 2, "dedup should prevent duplicate 'a'");
            assert_eq!(arr[0], rmpv::Value::String("a".into()));
            assert_eq!(arr[1], rmpv::Value::String("b".into()));
        } else {
            panic!("expected Document(Array), got {val:?}");
        }
    }

    #[test]
    fn doc_merge_delete_path() {
        let op = DocumentMerge;

        let rec = {
            let mut r = NodeRecord::new("Test");
            r.set_extra("keep", Value::Int(1));
            r.set_extra("remove", Value::Int(2));
            r
        };
        let base = encode_rec(&rec);

        let delta = DocDelta::DeletePath {
            target: PathTarget::Extra,
            path: vec!["remove".into()],
        };
        let operand = delta.encode().expect("encode");

        let result = op
            .merge(b"node:0:1", Some(&base), &[&operand])
            .expect("merge");
        let merged = decode_node_record(&result).expect("decode");

        assert!(merged.get_extra("keep").is_some());
        assert!(merged.get_extra("remove").is_none());
    }

    #[test]
    fn doc_merge_re_merge_stability() {
        // Re-merging a merged result with no operands must be stable.
        let op = DocumentMerge;

        let delta = DocDelta::SetPath {
            target: PathTarget::Extra,
            path: vec!["x".into()],
            value: rmpv::Value::Integer(42.into()),
        };
        let operand = delta.encode().expect("encode");

        let first = op
            .merge(b"node:0:1", None, &[&operand])
            .expect("first merge");
        let second = op
            .merge(b"node:0:1", Some(&first), &[])
            .expect("second merge");

        // Both should decode to equivalent NodeRecords.
        let rec1 = decode_node_record(&first).expect("decode 1");
        let rec2 = decode_node_record(&second).expect("decode 2");
        assert_eq!(rec1, rec2, "re-merge must be stable");
    }

    #[test]
    fn doc_merge_pre_merged_record_as_operand() {
        // A previously merged NodeRecord appears as an operand in subsequent
        // compaction. The merge function should detect the 0x00 prefix and
        // use it as the new base.
        let op = DocumentMerge;

        let delta1 = DocDelta::SetPath {
            target: PathTarget::Extra,
            path: vec!["a".into()],
            value: rmpv::Value::Integer(1.into()),
        };
        let op1 = delta1.encode().expect("encode");

        // First merge produces a NodeRecord.
        let partial = op.merge(b"node:0:1", None, &[&op1]).expect("partial merge");

        // Subsequent compaction: partial result as operand + new delta.
        let delta2 = DocDelta::SetPath {
            target: PathTarget::Extra,
            path: vec!["b".into()],
            value: rmpv::Value::Integer(2.into()),
        };
        let op2 = delta2.encode().expect("encode");

        let result = op
            .merge(b"node:0:1", None, &[&partial, &op2])
            .expect("re-merge");
        let rec = decode_node_record(&result).expect("decode");

        assert_eq!(rec.get_extra("a"), Some(&Value::Int(1)));
        assert_eq!(rec.get_extra("b"), Some(&Value::Int(2)));
    }

    #[test]
    fn doc_merge_preserves_labels_and_props() {
        // Merge must preserve existing labels and interned props.
        let op = DocumentMerge;

        let mut rec = NodeRecord::new("User");
        rec.add_label("Admin".into());
        rec.set(1, Value::String("Alice".into())); // interned prop
        rec.set_extra("doc_field", Value::Document(make_rmpv_map(vec![])));
        let base = encode_rec(&rec);

        let delta = DocDelta::SetPath {
            target: PathTarget::Extra,
            path: vec!["doc_field".into(), "nested".into()],
            value: rmpv::Value::Boolean(true),
        };
        let operand = delta.encode().expect("encode");

        let result = op
            .merge(b"node:0:1", Some(&base), &[&operand])
            .expect("merge");
        let merged = decode_node_record(&result).expect("decode");

        // Labels preserved.
        assert!(merged.has_label("User"));
        assert!(merged.has_label("Admin"));
        // Interned prop preserved.
        assert_eq!(merged.get(1), Some(&Value::String("Alice".into())));
        // Doc field updated.
        let doc_val = merged.get_extra("doc_field").expect("doc_field");
        if let Value::Document(doc) = doc_val {
            let nested = coordinode_core::graph::document::extract_at_path(doc, &["nested"]);
            assert_eq!(nested, rmpv::Value::Boolean(true));
        } else {
            panic!("expected Document, got {doc_val:?}");
        }
    }

    #[test]
    fn doc_merge_empty_operand_returns_error() {
        let op = DocumentMerge;
        let empty: &[u8] = &[];
        assert!(op.merge(b"node:0:1", None, &[empty]).is_err());
    }

    // --- PropField (G064) tests ---

    #[test]
    fn doc_merge_prop_field_set_path() {
        // SetPath targeting props[42] creates nested document in interned props.
        let op = DocumentMerge;
        let mut base = NodeRecord::new("Device");
        base.set(42, Value::Document(rmpv::Value::Map(vec![])));
        let base_bytes = encode_rec(&base);

        let delta = DocDelta::SetPath {
            target: PathTarget::PropField(42),
            path: vec!["network".into(), "ssid".into()],
            value: rmpv::Value::String("home".into()),
        };
        let operand = delta.encode().expect("encode");

        let result = op
            .merge(b"node:0:1", Some(&base_bytes), &[&operand])
            .expect("merge");
        let merged = decode_node_record(&result).expect("decode");

        if let Some(Value::Document(doc)) = merged.props.get(&42) {
            let val = coordinode_core::graph::document::extract_at_path(doc, &["network", "ssid"]);
            assert_eq!(val, rmpv::Value::String("home".into()));
        } else {
            panic!("expected Document at props[42]");
        }
    }

    #[test]
    fn doc_merge_prop_field_creates_doc_from_nothing() {
        // PropField delta on non-existent base creates empty record with doc in props.
        let op = DocumentMerge;

        let delta = DocDelta::SetPath {
            target: PathTarget::PropField(10),
            path: vec!["key".into()],
            value: rmpv::Value::Integer(99.into()),
        };
        let operand = delta.encode().expect("encode");

        let result = op.merge(b"node:0:1", None, &[&operand]).expect("merge");
        let merged = decode_node_record(&result).expect("decode");

        if let Some(Value::Document(doc)) = merged.props.get(&10) {
            let val = coordinode_core::graph::document::extract_at_path(doc, &["key"]);
            assert_eq!(val, rmpv::Value::Integer(99.into()));
        } else {
            panic!("expected Document at props[10]");
        }
    }

    #[test]
    fn doc_merge_prop_field_delete_path() {
        // DeletePath removes a nested key from a Document in props.
        let op = DocumentMerge;
        let doc = make_rmpv_map(vec![
            ("a", rmpv::Value::Integer(1.into())),
            ("b", rmpv::Value::Integer(2.into())),
        ]);
        let mut base = NodeRecord::new("Test");
        base.set(7, Value::Document(doc));
        let base_bytes = encode_rec(&base);

        let delta = DocDelta::DeletePath {
            target: PathTarget::PropField(7),
            path: vec!["a".into()],
        };
        let operand = delta.encode().expect("encode");

        let result = op
            .merge(b"node:0:1", Some(&base_bytes), &[&operand])
            .expect("merge");
        let merged = decode_node_record(&result).expect("decode");

        if let Some(Value::Document(doc)) = merged.props.get(&7) {
            assert_eq!(
                coordinode_core::graph::document::extract_at_path(doc, &["a"]),
                rmpv::Value::Nil
            );
            assert_eq!(
                coordinode_core::graph::document::extract_at_path(doc, &["b"]),
                rmpv::Value::Integer(2.into())
            );
        } else {
            panic!("expected Document at props[7]");
        }
    }

    #[test]
    fn doc_merge_prop_field_increment() {
        // Increment on a numeric field inside a Document in props.
        let op = DocumentMerge;
        let doc = make_rmpv_map(vec![("views", rmpv::Value::Integer(10.into()))]);
        let mut base = NodeRecord::new("Stats");
        base.set(5, Value::Document(doc));
        let base_bytes = encode_rec(&base);

        let delta = DocDelta::Increment {
            target: PathTarget::PropField(5),
            path: vec!["views".into()],
            amount: 3.0,
        };
        let operand = delta.encode().expect("encode");

        let result = op
            .merge(b"node:0:1", Some(&base_bytes), &[&operand])
            .expect("merge");
        let merged = decode_node_record(&result).expect("decode");

        if let Some(Value::Document(doc)) = merged.props.get(&5) {
            assert_eq!(
                coordinode_core::graph::document::extract_at_path(doc, &["views"]),
                rmpv::Value::Integer(13.into())
            );
        } else {
            panic!("expected Document at props[5]");
        }
    }

    #[test]
    fn doc_merge_prop_field_preserves_extra_and_other_props() {
        // PropField delta only touches its target field, not extra or other props.
        let op = DocumentMerge;
        let mut base = NodeRecord::new("Mixed");
        base.set(1, Value::String("name_val".into()));
        base.set(2, Value::Document(rmpv::Value::Map(vec![])));
        base.set_extra("overflow_key", Value::Int(42));
        let base_bytes = encode_rec(&base);

        let delta = DocDelta::SetPath {
            target: PathTarget::PropField(2),
            path: vec!["nested".into()],
            value: rmpv::Value::String("val".into()),
        };
        let operand = delta.encode().expect("encode");

        let result = op
            .merge(b"node:0:1", Some(&base_bytes), &[&operand])
            .expect("merge");
        let merged = decode_node_record(&result).expect("decode");

        // Other prop untouched.
        assert_eq!(merged.get(1), Some(&Value::String("name_val".into())));
        // Extra untouched.
        assert_eq!(merged.get_extra("overflow_key"), Some(&Value::Int(42)));
        // Target prop updated.
        if let Some(Value::Document(doc)) = merged.props.get(&2) {
            assert_eq!(
                coordinode_core::graph::document::extract_at_path(doc, &["nested"]),
                rmpv::Value::String("val".into())
            );
        } else {
            panic!("expected Document at props[2]");
        }
    }

    #[test]
    fn doc_merge_mixed_extra_and_prop_field() {
        // Extra delta and PropField delta in same merge call.
        let op = DocumentMerge;
        let mut base = NodeRecord::new("Mixed");
        base.set(3, Value::Document(rmpv::Value::Map(vec![])));
        let base_bytes = encode_rec(&base);

        let d1 = DocDelta::SetPath {
            target: PathTarget::Extra,
            path: vec!["overflow".into()],
            value: rmpv::Value::String("extra_val".into()),
        };
        let d2 = DocDelta::SetPath {
            target: PathTarget::PropField(3),
            path: vec!["nested".into()],
            value: rmpv::Value::String("prop_val".into()),
        };
        let op1 = d1.encode().expect("encode");
        let op2 = d2.encode().expect("encode");

        let result = op
            .merge(b"node:0:1", Some(&base_bytes), &[&op1, &op2])
            .expect("merge");
        let merged = decode_node_record(&result).expect("decode");

        assert_eq!(
            merged.get_extra("overflow"),
            Some(&Value::String("extra_val".into()))
        );
        if let Some(Value::Document(doc)) = merged.props.get(&3) {
            assert_eq!(
                coordinode_core::graph::document::extract_at_path(doc, &["nested"]),
                rmpv::Value::String("prop_val".into())
            );
        } else {
            panic!("expected Document at props[3]");
        }
    }

    #[test]
    fn doc_merge_legacy_bare_record_as_base() {
        // Legacy NodeRecord without 0x00 prefix (pre-R163 data).
        let op = DocumentMerge;

        let rec = NodeRecord::new("Legacy");
        let bare_msgpack = rec.to_msgpack().expect("encode");

        let delta = DocDelta::SetPath {
            target: PathTarget::Extra,
            path: vec!["x".into()],
            value: rmpv::Value::Integer(1.into()),
        };
        let operand = delta.encode().expect("encode");

        let result = op
            .merge(b"node:0:1", Some(&bare_msgpack), &[&operand])
            .expect("merge with legacy base");
        let merged = decode_node_record(&result).expect("decode");

        assert!(merged.has_label("Legacy"));
        assert_eq!(merged.get_extra("x"), Some(&Value::Int(1)));
    }

    // ── RemoveProperty (R083 TTL reaper) ─────────────────────────────

    #[test]
    fn doc_merge_remove_property_prop_field() {
        // RemoveProperty with PropField removes entire props[field_id].
        let op = DocumentMerge;
        let mut base = NodeRecord::new("Session");
        base.set(3, Value::Timestamp(1_000_000));
        base.set(5, Value::String("keep me".into()));
        let base_bytes = encode_rec(&base);

        let delta = DocDelta::RemoveProperty {
            target: PathTarget::PropField(3),
            key: None,
        };
        let operand = delta.encode().expect("encode");

        let result = op
            .merge(b"node:0:1", Some(&base_bytes), &[&operand])
            .expect("merge");
        let merged = decode_node_record(&result).expect("decode");

        assert!(!merged.props.contains_key(&3), "field 3 should be removed");
        assert_eq!(
            merged.props.get(&5),
            Some(&Value::String("keep me".into())),
            "field 5 should be preserved"
        );
    }

    #[test]
    fn doc_merge_remove_property_extra_key() {
        // RemoveProperty with Extra target removes a key from the extra map.
        let op = DocumentMerge;
        let mut base = NodeRecord::new("Validated");
        base.set_extra("temp_field", Value::Int(42));
        base.set_extra("keep_field", Value::Int(99));
        let base_bytes = encode_rec(&base);

        let delta = DocDelta::RemoveProperty {
            target: PathTarget::Extra,
            key: Some("temp_field".into()),
        };
        let operand = delta.encode().expect("encode");

        let result = op
            .merge(b"node:0:1", Some(&base_bytes), &[&operand])
            .expect("merge");
        let merged = decode_node_record(&result).expect("decode");

        assert!(
            merged.get_extra("temp_field").is_none(),
            "temp_field should be removed"
        );
        assert_eq!(
            merged.get_extra("keep_field"),
            Some(&Value::Int(99)),
            "keep_field should be preserved"
        );
    }

    #[test]
    fn doc_merge_remove_property_idempotent() {
        // Removing a non-existent property is a no-op.
        let op = DocumentMerge;
        let mut base = NodeRecord::new("Test");
        base.set(1, Value::Int(100));
        let base_bytes = encode_rec(&base);

        let delta = DocDelta::RemoveProperty {
            target: PathTarget::PropField(999), // doesn't exist
            key: None,
        };
        let operand = delta.encode().expect("encode");

        let result = op
            .merge(b"node:0:1", Some(&base_bytes), &[&operand])
            .expect("merge");
        let merged = decode_node_record(&result).expect("decode");

        assert_eq!(merged.props.get(&1), Some(&Value::Int(100)));
    }

    // ── R099: Extra-delta batching (batch rmpv round-trips) ──────────────────

    #[test]
    fn doc_merge_multiple_extra_deltas_batched_same_result() {
        // Multiple consecutive Extra-targeting deltas must produce the same
        // result whether applied one at a time (old code) or via the batching
        // path (new code). Tests that extra_doc accumulation is correct.
        let op = DocumentMerge;

        let initial = make_rmpv_map(vec![("x", rmpv::Value::Integer(0.into()))]);
        let rec = make_record_with_doc("counters", initial);
        let base = encode_rec(&rec);

        // Three consecutive SetPath deltas, all targeting PathTarget::Extra.
        let d1 = DocDelta::SetPath {
            target: PathTarget::Extra,
            path: vec!["counters".into(), "x".into()],
            value: rmpv::Value::Integer(1.into()),
        };
        let d2 = DocDelta::SetPath {
            target: PathTarget::Extra,
            path: vec!["counters".into(), "y".into()],
            value: rmpv::Value::Integer(2.into()),
        };
        let d3 = DocDelta::SetPath {
            target: PathTarget::Extra,
            path: vec!["counters".into(), "z".into()],
            value: rmpv::Value::Integer(3.into()),
        };

        let op1 = d1.encode().expect("encode d1");
        let op2 = d2.encode().expect("encode d2");
        let op3 = d3.encode().expect("encode d3");

        let result = op
            .merge(b"node:0:1", Some(&base), &[&op1, &op2, &op3])
            .expect("merge");
        let merged = decode_node_record(&result).expect("decode");

        let doc_val = merged.get_extra("counters").expect("counters key");
        let Value::Document(doc) = doc_val else {
            panic!("expected Document, got {doc_val:?}");
        };
        use coordinode_core::graph::document::extract_at_path;
        assert_eq!(
            extract_at_path(doc, &["x"]),
            rmpv::Value::Integer(1.into()),
            "x should be 1"
        );
        assert_eq!(
            extract_at_path(doc, &["y"]),
            rmpv::Value::Integer(2.into()),
            "y should be 2"
        );
        assert_eq!(
            extract_at_path(doc, &["z"]),
            rmpv::Value::Integer(3.into()),
            "z should be 3"
        );
    }

    #[test]
    fn doc_merge_mixed_extra_and_propfield_deltas() {
        // Mixed Extra and PropField deltas. PropField must apply correctly
        // regardless of pending extra_doc state (they don't interact — PropField
        // operates on rec.props, not rec.extra / extra_doc accumulator).
        let op = DocumentMerge;

        // Field 7 stores a Document-typed property (PropField target requires Document).
        let field7_doc = make_rmpv_map(vec![("inner", rmpv::Value::Integer(10.into()))]);
        let mut base_rec = NodeRecord::new("Mixed");
        base_rec.set(7, Value::Document(field7_doc));
        let initial_meta = make_rmpv_map(vec![("a", rmpv::Value::Integer(0.into()))]);
        base_rec.set_extra("meta", Value::Document(initial_meta));
        let base = encode_rec(&base_rec);

        // Extra delta — sets meta.a = 42.
        let d_extra = DocDelta::SetPath {
            target: PathTarget::Extra,
            path: vec!["meta".into(), "a".into()],
            value: rmpv::Value::Integer(42.into()),
        };
        // PropField delta — sets field 7's "inner" key to 99.
        let d_prop = DocDelta::SetPath {
            target: PathTarget::PropField(7),
            path: vec!["inner".into()],
            value: rmpv::Value::Integer(99.into()),
        };
        // Another Extra delta — sets meta.b = 7.
        let d_extra2 = DocDelta::SetPath {
            target: PathTarget::Extra,
            path: vec!["meta".into(), "b".into()],
            value: rmpv::Value::Integer(7.into()),
        };

        let op1 = d_extra.encode().expect("encode d_extra");
        let op2 = d_prop.encode().expect("encode d_prop");
        let op3 = d_extra2.encode().expect("encode d_extra2");

        let result = op
            .merge(b"node:0:1", Some(&base), &[&op1, &op2, &op3])
            .expect("merge");
        let merged = decode_node_record(&result).expect("decode");

        // PropField 7's "inner" key should be updated to 99.
        let field7 = merged.props.get(&7).expect("field 7 must exist");
        let Value::Document(f7_doc) = field7 else {
            panic!("expected Document for field 7, got {field7:?}");
        };
        use coordinode_core::graph::document::extract_at_path;
        assert_eq!(
            extract_at_path(f7_doc, &["inner"]),
            rmpv::Value::Integer(99.into()),
            "field7.inner = 99"
        );

        // Extra "meta" should have both changes from d_extra and d_extra2.
        let doc_val = merged.get_extra("meta").expect("meta key");
        let Value::Document(doc) = doc_val else {
            panic!("expected Document, got {doc_val:?}");
        };
        assert_eq!(
            extract_at_path(doc, &["a"]),
            rmpv::Value::Integer(42.into()),
            "meta.a = 42"
        );
        assert_eq!(
            extract_at_path(doc, &["b"]),
            rmpv::Value::Integer(7.into()),
            "meta.b = 7"
        );
    }

    #[test]
    fn doc_merge_setpath_then_remove_property_extra() {
        // SetPath (Extra) followed by RemoveProperty (Extra) — the removal must
        // be applied AFTER the set. Both deltas are applied to the same extra_doc
        // accumulator so ordering is preserved.
        let op = DocumentMerge;

        let initial = make_rmpv_map(vec![
            ("keep", rmpv::Value::Integer(1.into())),
            ("drop", rmpv::Value::Integer(2.into())),
        ]);
        let rec = make_record_with_doc("data", initial);
        let base = encode_rec(&rec);

        // Set "data.new_field" first.
        let d_set = DocDelta::SetPath {
            target: PathTarget::Extra,
            path: vec!["data".into(), "new_field".into()],
            value: rmpv::Value::Boolean(true),
        };
        // Then remove the entire "drop" top-level extra key.
        let d_remove = DocDelta::RemoveProperty {
            target: PathTarget::Extra,
            key: Some("drop".into()),
        };

        let op1 = d_set.encode().expect("encode d_set");
        let op2 = d_remove.encode().expect("encode d_remove");

        let result = op
            .merge(b"node:0:1", Some(&base), &[&op1, &op2])
            .expect("merge");
        let merged = decode_node_record(&result).expect("decode");

        // "drop" was a top-level extra key and should be gone.
        assert!(
            merged.get_extra("drop").is_none(),
            "'drop' key should have been removed"
        );

        // "data" with its nested values (keep + new_field) should remain.
        let doc_val = merged.get_extra("data").expect("data key");
        let Value::Document(doc) = doc_val else {
            panic!("expected Document, got {doc_val:?}");
        };
        use coordinode_core::graph::document::extract_at_path;
        assert_eq!(
            extract_at_path(doc, &["new_field"]),
            rmpv::Value::Boolean(true),
            "data.new_field should be true"
        );
        assert_eq!(
            extract_at_path(doc, &["keep"]),
            rmpv::Value::Integer(1.into()),
            "data.keep should be 1"
        );
    }

    #[test]
    fn doc_merge_base_reset_flushes_extra_doc() {
        // When a PREFIX_NODE_RECORD operand appears mid-stream (base reset),
        // the pending extra_doc must be flushed into the previous record before
        // the reset, then the new base starts fresh. This tests the flush-before-reset
        // invariant in DocumentMerge::merge.
        let op = DocumentMerge;

        // First base: node with "score" = 0 in extra.
        let initial = make_rmpv_map(vec![("value", rmpv::Value::Integer(0.into()))]);
        let base1_rec = make_record_with_doc("score", initial);
        let base1 = encode_rec(&base1_rec);

        // Delta applied to base1 (sets score.value = 10).
        let d1 = DocDelta::SetPath {
            target: PathTarget::Extra,
            path: vec!["score".into(), "value".into()],
            value: rmpv::Value::Integer(10.into()),
        };

        // Second base reset (new full NodeRecord with different label).
        let mut base2_rec = NodeRecord::new("Replacement");
        base2_rec.set_extra("flag", Value::Bool(true));
        let base2 = encode_rec(&base2_rec);

        // Delta applied to base2 (sets a new extra key).
        let d2 = DocDelta::SetPath {
            target: PathTarget::Extra,
            path: vec!["note".into()],
            value: rmpv::Value::String("after reset".into()),
        };

        let op1 = d1.encode().expect("encode d1");
        let op2 = d2.encode().expect("encode d2");

        // Operands: delta1, then a full NodeRecord (base reset), then delta2.
        let result = op
            .merge(b"node:0:1", Some(&base1), &[&op1, &base2, &op2])
            .expect("merge");
        let merged = decode_node_record(&result).expect("decode");

        // Final state should be base2 + d2 applied ("Replacement" label, flag=true, note="after reset").
        // d1's effect should be gone (applied to base1 which was replaced).
        assert!(
            merged.has_label("Replacement"),
            "label should be from base2 (Replacement)"
        );
        assert_eq!(
            merged.get_extra("flag"),
            Some(&Value::Bool(true)),
            "flag from base2 should be present"
        );
        assert_eq!(
            merged.get_extra("note"),
            Some(&Value::String("after reset".into())),
            "note from d2 should be set"
        );
        // score from base1 is gone — the reset wiped it.
        assert!(
            merged.get_extra("score").is_none(),
            "score from base1 should be gone after reset"
        );
    }
}

// ─── CounterMerge (R163b) ────────────────────────────────────────────

/// Encode a counter delta operand: i64 little-endian (8 bytes).
pub fn encode_counter_delta(delta: i64) -> Vec<u8> {
    delta.to_le_bytes().to_vec()
}

/// Decode a counter value from raw bytes (i64 little-endian).
pub fn decode_counter(data: &[u8]) -> i64 {
    if data.len() >= 8 {
        i64::from_le_bytes(data[..8].try_into().unwrap_or([0; 8]))
    } else {
        0
    }
}

/// Merge operator for atomic i64 counters on the `counter:` partition.
///
/// Base value: i64 LE (8 bytes). Zero if absent.
/// Operands: i64 LE deltas (8 bytes each).
/// Result: base + sum(operands), encoded as i64 LE.
///
/// Conflict-free: addition is commutative and associative, so concurrent
/// increments from multiple writers produce the correct total regardless
/// of operand order or partial compaction.
pub struct CounterMerge;

impl MergeOperator for CounterMerge {
    fn merge(
        &self,
        _key: &[u8],
        base_value: Option<&[u8]>,
        operands: &[&[u8]],
    ) -> Result<UserValue, LsmError> {
        let mut total = match base_value {
            Some(data) => decode_counter(data),
            None => 0,
        };

        for operand in operands {
            total = total.wrapping_add(decode_counter(operand));
        }

        Ok(total.to_le_bytes().to_vec().into())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod counter_merge_tests {
    use super::*;

    #[test]
    fn counter_merge_no_base_single_delta() {
        let merger = CounterMerge;
        let delta = encode_counter_delta(5);
        let result = merger.merge(b"counter:test", None, &[&delta]).unwrap();
        assert_eq!(decode_counter(&result), 5);
    }

    #[test]
    fn counter_merge_with_base_and_deltas() {
        let merger = CounterMerge;
        let base = 100i64.to_le_bytes();
        let d1 = encode_counter_delta(10);
        let d2 = encode_counter_delta(-3);
        let d3 = encode_counter_delta(7);
        let result = merger
            .merge(b"counter:x", Some(&base), &[&d1, &d2, &d3])
            .unwrap();
        assert_eq!(decode_counter(&result), 114); // 100 + 10 - 3 + 7
    }

    #[test]
    fn counter_merge_no_base_no_operands() {
        let merger = CounterMerge;
        let result = merger.merge(b"counter:empty", None, &[]).unwrap();
        assert_eq!(decode_counter(&result), 0);
    }

    #[test]
    fn counter_merge_negative_result() {
        let merger = CounterMerge;
        let base = 5i64.to_le_bytes();
        let d1 = encode_counter_delta(-10);
        let result = merger.merge(b"counter:neg", Some(&base), &[&d1]).unwrap();
        assert_eq!(decode_counter(&result), -5);
    }

    #[test]
    fn counter_merge_wrapping_overflow() {
        let merger = CounterMerge;
        let base = i64::MAX.to_le_bytes();
        let d1 = encode_counter_delta(1);
        let result = merger.merge(b"counter:wrap", Some(&base), &[&d1]).unwrap();
        // wrapping_add: MAX + 1 = MIN
        assert_eq!(decode_counter(&result), i64::MIN);
    }

    #[test]
    fn counter_merge_multiple_compaction_rounds() {
        // Simulate: first compaction merges base+d1+d2, second merges result+d3.
        let merger = CounterMerge;
        let d1 = encode_counter_delta(10);
        let d2 = encode_counter_delta(20);
        let round1 = merger.merge(b"counter:multi", None, &[&d1, &d2]).unwrap();
        assert_eq!(decode_counter(&round1), 30);

        let d3 = encode_counter_delta(5);
        let round2 = merger
            .merge(b"counter:multi", Some(&round1), &[&d3])
            .unwrap();
        assert_eq!(decode_counter(&round2), 35);
    }

    #[test]
    fn counter_encode_decode_roundtrip() {
        for val in [0i64, 1, -1, 42, -999, i64::MAX, i64::MIN] {
            let encoded = encode_counter_delta(val);
            assert_eq!(decode_counter(&encoded), val);
        }
    }

    #[test]
    fn counter_decode_short_data() {
        // Less than 8 bytes → 0
        assert_eq!(decode_counter(&[]), 0);
        assert_eq!(decode_counter(&[1, 2, 3]), 0);
    }

    #[test]
    fn counter_merge_through_storage_engine() {
        // Integration: merge through real StorageEngine + Counter partition.
        use crate::engine::config::StorageConfig;
        use crate::engine::core::StorageEngine;
        use crate::engine::partition::Partition;

        let dir = tempfile::tempdir().unwrap();
        let config = StorageConfig::new(dir.path());
        let engine = StorageEngine::open(&config).unwrap();

        let key = b"counter:degree:42";

        // First merge: no base, delta +10
        engine
            .merge(Partition::Counter, key, &encode_counter_delta(10))
            .unwrap();
        // Second merge: delta +5
        engine
            .merge(Partition::Counter, key, &encode_counter_delta(5))
            .unwrap();
        // Third merge: delta -3
        engine
            .merge(Partition::Counter, key, &encode_counter_delta(-3))
            .unwrap();

        // Read back: should be 10 + 5 - 3 = 12
        let value = engine.get(Partition::Counter, key).unwrap().unwrap();
        assert_eq!(
            decode_counter(&value),
            12,
            "counter should be 12 after 3 merges"
        );
    }

    #[test]
    fn counter_merge_concurrent_increments() {
        // Integration: concurrent merges from multiple threads.
        use crate::engine::config::StorageConfig;
        use crate::engine::core::StorageEngine;
        use crate::engine::partition::Partition;
        use std::sync::Arc;

        let dir = tempfile::tempdir().unwrap();
        let config = StorageConfig::new(dir.path());
        let engine = Arc::new(StorageEngine::open(&config).unwrap());

        let key = b"counter:concurrent";
        let n_threads = 4;
        let increments_per_thread = 100;

        let mut handles = Vec::new();
        for _ in 0..n_threads {
            let engine = Arc::clone(&engine);
            handles.push(std::thread::spawn(move || {
                for _ in 0..increments_per_thread {
                    engine
                        .merge(Partition::Counter, key, &encode_counter_delta(1))
                        .unwrap();
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        let value = engine.get(Partition::Counter, key).unwrap().unwrap();
        let total = decode_counter(&value);
        assert_eq!(
            total,
            (n_threads * increments_per_thread) as i64,
            "concurrent increments should sum correctly"
        );
    }
}
