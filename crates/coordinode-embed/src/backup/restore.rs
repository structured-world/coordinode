//! Restore graph data from backup files into CoordiNode storage.
//!
//! Supports JSON Lines and binary formats for import.
//! Cypher import executes statements through the query engine.

use std::io::{BufRead, Read};

use coordinode_core::graph::intern::FieldInterner;
use coordinode_core::graph::node::NodeId;
use coordinode_core::graph::types::Value;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

use super::export::BackupEntry;

/// Errors during restore.
#[derive(Debug, thiserror::Error)]
pub enum RestoreError {
    #[error("storage error: {0}")]
    Storage(String),

    #[error("deserialization error: {0}")]
    Deserialization(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid backup format: {0}")]
    InvalidFormat(String),
}

/// Statistics from a restore operation.
#[derive(Debug, Default, Clone)]
pub struct RestoreStats {
    pub nodes: u64,
    pub edges: u64,
    pub schema_entries: u64,
}

/// Restore from a binary (MessagePack) backup dump.
///
/// Reads length-prefixed MessagePack entries and writes them
/// directly to storage partitions. Fastest restore method.
/// ADR-016: writes use plain engine.put() — oracle auto-stamps seqno.
pub fn restore_binary<R: Read>(
    engine: &StorageEngine,
    reader: &mut R,
) -> Result<(RestoreStats, Option<FieldInterner>), RestoreError> {
    let mut stats = RestoreStats::default();
    let mut interner = None;
    let mut len_buf = [0u8; 4];

    loop {
        match reader.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(RestoreError::Io(e)),
        }

        let entry_len = u32::from_le_bytes(len_buf) as usize;
        let mut entry_buf = vec![0u8; entry_len];
        reader.read_exact(&mut entry_buf)?;

        let entry: BackupEntry = rmp_serde::from_slice(&entry_buf)
            .map_err(|e| RestoreError::Deserialization(e.to_string()))?;

        match entry {
            BackupEntry::Interner(data) => {
                interner = FieldInterner::from_bytes(&data);
            }
            BackupEntry::Node { key, value } => {
                engine
                    .put(Partition::Node, &key, &value)
                    .map_err(|e| RestoreError::Storage(e.to_string()))?;
                stats.nodes += 1;
            }
            BackupEntry::Adj { key, value } => {
                // Adj keys are raw (no MVCC timestamps) — write directly to engine.
                engine
                    .put(Partition::Adj, &key, &value)
                    .map_err(|e| RestoreError::Storage(e.to_string()))?;
                let key_str = std::str::from_utf8(&key).unwrap_or("");
                if key_str.contains(":out:") {
                    stats.edges += 1;
                }
            }
            BackupEntry::EdgeProp { key, value } => {
                engine
                    .put(Partition::EdgeProp, &key, &value)
                    .map_err(|e| RestoreError::Storage(e.to_string()))?;
            }
            BackupEntry::Schema { key, value } => {
                // Schema is not MVCC-versioned — write directly.
                engine
                    .put(Partition::Schema, &key, &value)
                    .map_err(|e| RestoreError::Storage(e.to_string()))?;
                stats.schema_entries += 1;
            }
        }
    }

    Ok((stats, interner))
}

/// Restore from JSON Lines backup.
///
/// Each line is a JSON object with `"type": "node"` or `"type": "edge"`.
/// Nodes are created via direct storage writes; edges are created by
/// encoding the adjacency keys.
///
/// Requires an existing FieldInterner (or creates a new one).
/// ADR-016: writes use plain engine.put() — oracle auto-stamps seqno.
pub fn restore_json<R: BufRead>(
    engine: &StorageEngine,
    interner: &mut FieldInterner,
    shard_id: u16,
    reader: &mut R,
) -> Result<RestoreStats, RestoreError> {
    use coordinode_core::graph::edge::{encode_adj_key_forward, encode_adj_key_reverse};
    use coordinode_core::graph::node::NodeRecord;
    use coordinode_modality::{LocalNodeStore, NodeStore as _};
    let node_store = LocalNodeStore::new(engine);

    let mut stats = RestoreStats::default();

    for line_result in reader.lines() {
        let line = line_result?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let obj: serde_json::Value = serde_json::from_str(line)
            .map_err(|e| RestoreError::Deserialization(format!("invalid JSON: {e}")))?;

        let entity_type = obj
            .get("type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| RestoreError::InvalidFormat("missing 'type' field".into()))?;

        match entity_type {
            "node" => {
                let id = obj
                    .get("id")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| RestoreError::InvalidFormat("node missing 'id'".into()))?;

                let labels: Vec<String> = obj
                    .get("labels")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default();

                let mut record = NodeRecord::with_labels(labels);

                if let Some(props) = obj.get("properties").and_then(|v| v.as_object()) {
                    for (name, json_val) in props {
                        let field_id = interner.intern(name);
                        let value = json_to_value(json_val);
                        record.set(field_id, value);
                    }
                }

                node_store
                    .put(shard_id, NodeId::from_raw(id), &record)
                    .map_err(|e| RestoreError::Storage(e.to_string()))?;

                stats.nodes += 1;
            }
            "edge" => {
                let source = obj
                    .get("source")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| RestoreError::InvalidFormat("edge missing 'source'".into()))?;
                let target = obj
                    .get("target")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| RestoreError::InvalidFormat("edge missing 'target'".into()))?;
                let edge_type = obj
                    .get("edge_type")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        RestoreError::InvalidFormat("edge missing 'edge_type'".into())
                    })?;

                // Write forward adjacency via merge operator (raw key, no MVCC).
                let fwd_key = encode_adj_key_forward(edge_type, NodeId::from_raw(source));
                engine
                    .merge(
                        Partition::Adj,
                        &fwd_key,
                        &coordinode_storage::engine::merge::encode_add(target),
                    )
                    .map_err(|e| RestoreError::Storage(e.to_string()))?;

                // Write reverse adjacency via merge operator.
                let rev_key = encode_adj_key_reverse(edge_type, NodeId::from_raw(target));
                engine
                    .merge(
                        Partition::Adj,
                        &rev_key,
                        &coordinode_storage::engine::merge::encode_add(source),
                    )
                    .map_err(|e| RestoreError::Storage(e.to_string()))?;

                // Write edge properties if present
                if let Some(props) = obj.get("properties").and_then(|v| v.as_object()) {
                    if !props.is_empty() {
                        let mut prop_map = std::collections::HashMap::new();
                        for (name, json_val) in props {
                            let field_id = interner.intern(name);
                            prop_map.insert(field_id, json_to_value(json_val));
                        }
                        let ep_key = coordinode_core::graph::edge::encode_edgeprop_key(
                            edge_type,
                            NodeId::from_raw(source),
                            NodeId::from_raw(target),
                        );
                        let ep_value = rmp_serde::to_vec(&prop_map)
                            .map_err(|e| RestoreError::Deserialization(e.to_string()))?;
                        engine
                            .put(Partition::EdgeProp, &ep_key, &ep_value)
                            .map_err(|e| RestoreError::Storage(e.to_string()))?;
                    }
                }

                stats.edges += 1;
            }
            other => {
                return Err(RestoreError::InvalidFormat(format!(
                    "unknown entity type: {other}"
                )));
            }
        }
    }

    Ok(stats)
}

/// Convert JSON value to CoordiNode Value.
fn json_to_value(v: &serde_json::Value) -> Value {
    match v {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else if let Some(f) = n.as_f64() {
                Value::Float(f)
            } else {
                Value::Null
            }
        }
        serde_json::Value::String(s) => Value::String(s.clone()),
        serde_json::Value::Array(arr) => {
            let items: Vec<Value> = arr.iter().map(json_to_value).collect();
            Value::Array(items)
        }
        serde_json::Value::Object(obj) => {
            // Check for special types
            if let Some(ts) = obj.get("_timestamp").and_then(|v| v.as_i64()) {
                return Value::Timestamp(ts);
            }
            if let Some(doc_val) = obj.get("_document") {
                return Value::Document(json_to_rmpv(doc_val));
            }
            let map: std::collections::BTreeMap<String, Value> = obj
                .iter()
                .map(|(k, v)| (k.clone(), json_to_value(v)))
                .collect();
            Value::Map(map)
        }
    }
}

/// Convert a serde_json::Value to rmpv::Value for Document restore.
fn json_to_rmpv(v: &serde_json::Value) -> rmpv::Value {
    match v {
        serde_json::Value::Null => rmpv::Value::Nil,
        serde_json::Value::Bool(b) => rmpv::Value::Boolean(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                rmpv::Value::Integer(i.into())
            } else if let Some(f) = n.as_f64() {
                rmpv::Value::F64(f)
            } else {
                rmpv::Value::Nil
            }
        }
        serde_json::Value::String(s) => rmpv::Value::String(s.clone().into()),
        serde_json::Value::Array(arr) => rmpv::Value::Array(arr.iter().map(json_to_rmpv).collect()),
        serde_json::Value::Object(obj) => {
            // Sort keys alphabetically to produce deterministic rmpv::Map order.
            // serde_json::Map is BTreeMap (sorted), so iterating already yields
            // sorted keys — but we sort explicitly to be safe.
            let mut entries: Vec<(rmpv::Value, rmpv::Value)> = obj
                .iter()
                .map(|(k, v)| (rmpv::Value::String(k.clone().into()), json_to_rmpv(v)))
                .collect();
            entries.sort_by(|(a, _), (b, _)| {
                let ak = a.as_str().unwrap_or("");
                let bk = b.as_str().unwrap_or("");
                ak.cmp(bk)
            });
            rmpv::Value::Map(entries)
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::backup::export::value_to_json;

    #[test]
    fn document_json_roundtrip_via_marker() {
        // Use alphabetically sorted keys so JSON roundtrip preserves order
        let doc = rmpv::Value::Map(vec![
            (
                rmpv::Value::String("count".into()),
                rmpv::Value::Integer(42.into()),
            ),
            (
                rmpv::Value::String("key".into()),
                rmpv::Value::String("value".into()),
            ),
        ]);
        let original = Value::Document(doc);

        // export → JSON
        let json = value_to_json(&original);
        // JSON should have _document wrapper
        assert!(json.is_object());
        assert!(json.get("_document").is_some());

        // import → Value
        let restored = json_to_value(&json);
        assert_eq!(original, restored);
    }

    #[test]
    fn document_nested_json_roundtrip() {
        let doc = rmpv::Value::Map(vec![(
            rmpv::Value::String("nested".into()),
            rmpv::Value::Map(vec![(
                rmpv::Value::String("deep".into()),
                rmpv::Value::Array(vec![
                    rmpv::Value::Integer(1.into()),
                    rmpv::Value::Boolean(true),
                ]),
            )]),
        )]);
        let original = Value::Document(doc);
        let json = value_to_json(&original);
        let restored = json_to_value(&json);
        assert_eq!(original, restored);
    }

    #[test]
    fn map_not_confused_with_document() {
        // Regular Map should NOT become Document on restore
        let map = Value::Map(std::collections::BTreeMap::from([(
            "name".to_string(),
            Value::String("test".into()),
        )]));
        let json = value_to_json(&map);
        let restored = json_to_value(&json);
        assert!(matches!(restored, Value::Map(_)));
    }
}
