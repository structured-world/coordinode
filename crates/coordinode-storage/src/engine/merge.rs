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

/// Apply a sequence of `DocDelta` operands to a `NodeRecord` in memory.
///
/// Mirrors the `DocumentMerge` LSM merge path but operates on a pre-loaded
/// record. Callers use this when they need to materialise the post-delta
/// state synchronously rather than queue merge operands — the canonical
/// case is the R172c Phase 3b temporal close+open path, where a nested
/// `SET n.doc.a.b = …` on a temporal node must produce the full new-version
/// NodeRecord (not a merge operand on the non-temporal key).
///
/// Batches Extra-targeting deltas through a shared `rmpv` representation
/// for the same throughput characteristic as the merge operator.
/// PropField and RemoveProperty deltas are applied immediately.
pub fn apply_doc_deltas_to_record(rec: &mut NodeRecord, deltas: &[DocDelta]) {
    let mut extra_doc: Option<rmpv::Value> = None;
    for delta in deltas {
        apply_delta_to_record_batched(rec, delta, &mut extra_doc);
    }
    if let Some(doc) = extra_doc {
        rmpv_to_extra(rec, &doc);
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
mod tests;

/// Proptest fuzzing for posting list merge operator.
///
/// Generates random sequences of Add/Remove operations and verifies
/// that the result is always sorted and contains exactly the expected UIDs.
#[cfg(test)]
#[allow(clippy::expect_used)]
mod proptest_merge;

// ---------------------------------------------------------------------------
// Document merge tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod doc_merge_tests;

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
mod counter_merge_tests;
