//! Restore graph data from backup files into CoordiNode storage.
//!
//! Supports JSON Lines and binary formats for import.
//! Cypher restore parses CoordiNode's own cypher dump format directly.

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
                        // Executor-native wire shape: Vec<(field_id, Value)>
                        // (runner.rs decodes exactly this) — a HashMap would
                        // make the restored props unreadable by queries.
                        let prop_vec: Vec<(u32, Value)> = props
                            .iter()
                            .map(|(name, json_val)| {
                                (interner.intern(name), json_to_value(json_val))
                            })
                            .collect();
                        let ep_key = coordinode_core::graph::edge::encode_edgeprop_key(
                            edge_type,
                            NodeId::from_raw(source),
                            NodeId::from_raw(target),
                        );
                        let ep_value = rmp_serde::to_vec(&prop_vec)
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

/// Restore from a Cypher (OpenCypher `CREATE` statements) backup dump.
///
/// Parses the statement form emitted by [`super::export::export_cypher`]:
///
/// ```text
/// CREATE (n<id>:<L1>:<L2> {<props>});
/// CREATE (n<src>)-[:<TYPE> {<props>}]->(n<tgt>);
/// ```
///
/// and writes directly to storage, preserving the original node ids so
/// edges link to their endpoints. Property values are JSON literals (see
/// `format_cypher_props`), parsed via the shared [`json_to_value`].
///
/// This is the round-trip path for CoordiNode's own cypher dumps. It is
/// deliberately NOT a general OpenCypher importer: arbitrary external
/// cypher (foreign schemas, computed expressions, multi-statement scope)
/// must go through the query engine.
pub fn restore_cypher<R: BufRead>(
    engine: &StorageEngine,
    interner: &mut FieldInterner,
    shard_id: u16,
    reader: &mut R,
) -> Result<RestoreStats, RestoreError> {
    use coordinode_core::graph::edge::{
        encode_adj_key_forward, encode_adj_key_reverse, encode_edgeprop_key,
    };
    use coordinode_core::graph::node::NodeRecord;
    use coordinode_modality::{LocalNodeStore, NodeStore as _};
    let node_store = LocalNodeStore::new(engine);
    let mut stats = RestoreStats::default();

    // One statement per line, each terminated by `;`. Export emits JSON
    // values, which escape newlines, so line-based splitting is safe.
    for line_result in reader.lines() {
        let line = line_result?;
        let stmt = line.trim().trim_end_matches(';').trim();
        if stmt.is_empty() || stmt.starts_with("//") {
            continue;
        }
        let body = stmt
            .strip_prefix("CREATE ")
            .ok_or_else(|| {
                RestoreError::InvalidFormat(format!("expected CREATE statement: {stmt}"))
            })?
            .trim();

        if body.contains(")-[") {
            let (source, edge_type, target, props) = parse_cypher_edge(body)?;
            let fwd = encode_adj_key_forward(&edge_type, NodeId::from_raw(source));
            engine
                .merge(
                    Partition::Adj,
                    &fwd,
                    &coordinode_storage::engine::merge::encode_add(target),
                )
                .map_err(|e| RestoreError::Storage(e.to_string()))?;
            let rev = encode_adj_key_reverse(&edge_type, NodeId::from_raw(target));
            engine
                .merge(
                    Partition::Adj,
                    &rev,
                    &coordinode_storage::engine::merge::encode_add(source),
                )
                .map_err(|e| RestoreError::Storage(e.to_string()))?;
            if !props.is_empty() {
                // Executor-native wire shape: Vec<(field_id, Value)>
                // (runner.rs decodes exactly this) — a HashMap would make
                // the restored props unreadable by queries.
                let prop_vec: Vec<(u32, Value)> = props
                    .into_iter()
                    .map(|(name, val)| (interner.intern(&name), val))
                    .collect();
                let ep_key = encode_edgeprop_key(
                    &edge_type,
                    NodeId::from_raw(source),
                    NodeId::from_raw(target),
                );
                let ep_value = rmp_serde::to_vec(&prop_vec)
                    .map_err(|e| RestoreError::Deserialization(e.to_string()))?;
                engine
                    .put(Partition::EdgeProp, &ep_key, &ep_value)
                    .map_err(|e| RestoreError::Storage(e.to_string()))?;
            }
            stats.edges += 1;
        } else {
            let (id, labels, props) = parse_cypher_node(body)?;
            let mut record = NodeRecord::with_labels(labels);
            for (name, val) in props {
                record.set(interner.intern(&name), val);
            }
            node_store
                .put(shard_id, NodeId::from_raw(id), &record)
                .map_err(|e| RestoreError::Storage(e.to_string()))?;
            stats.nodes += 1;
        }
    }
    Ok(stats)
}

/// Decoded `(node_id, labels, properties)` from a cypher node statement.
type CypherNode = (u64, Vec<String>, Vec<(String, Value)>);
/// Decoded `(source_id, edge_type, target_id, properties)` from a cypher edge.
type CypherEdge = (u64, String, u64, Vec<(String, Value)>);

/// Parse `(n<id>:<labels> {<props>})` or `(n<id>:<labels>)`.
fn parse_cypher_node(body: &str) -> Result<CypherNode, RestoreError> {
    let inner = body
        .strip_prefix('(')
        .and_then(|s| s.strip_suffix(')'))
        .ok_or_else(|| RestoreError::InvalidFormat(format!("malformed node: {body}")))?;
    // Split the variable+labels head from the optional ` {props}` tail.
    let (head, props) = match inner.find(" {") {
        Some(brace) => {
            let head = inner[..brace].trim();
            let props_block = inner[brace + 2..]
                .trim_end()
                .strip_suffix('}')
                .ok_or_else(|| {
                    RestoreError::InvalidFormat(format!("unterminated props: {inner}"))
                })?;
            (head, parse_cypher_props(props_block)?)
        }
        None => (inner.trim(), Vec::new()),
    };
    // head = `n<id>:<L1>:<L2>` (labels may be empty).
    let head = head.strip_prefix('n').ok_or_else(|| {
        RestoreError::InvalidFormat(format!("node var must start with n: {head}"))
    })?;
    let (id_str, label_str) = match head.find(':') {
        Some(c) => (&head[..c], &head[c + 1..]),
        None => (head, ""),
    };
    let id: u64 = id_str
        .trim()
        .parse()
        .map_err(|_| RestoreError::InvalidFormat(format!("bad node id: {id_str}")))?;
    let labels: Vec<String> = label_str
        .split(':')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();
    Ok((id, labels, props))
}

/// Parse `(n<src>)-[:<TYPE> {<props>}]->(n<tgt>)`.
fn parse_cypher_edge(body: &str) -> Result<CypherEdge, RestoreError> {
    let rel_open = body
        .find(")-[")
        .ok_or_else(|| RestoreError::InvalidFormat(format!("malformed edge: {body}")))?;
    let rel_close = body
        .find("]->(")
        .ok_or_else(|| RestoreError::InvalidFormat(format!("malformed edge: {body}")))?;
    let src = parse_node_ref(&body[..rel_open + 1])?;
    let tgt_str = body[rel_close + 4..]
        .trim_end()
        .strip_suffix(')')
        .ok_or_else(|| RestoreError::InvalidFormat(format!("malformed edge target: {body}")))?;
    let tgt = parse_pid(tgt_str)?;
    // rel = `:<TYPE>` or `:<TYPE> {props}`.
    let rel = body[rel_open + 3..rel_close]
        .trim()
        .strip_prefix(':')
        .ok_or_else(|| RestoreError::InvalidFormat(format!("edge missing type: {body}")))?
        .trim();
    let (edge_type, props) = match rel.find(" {") {
        Some(brace) => {
            let props_block = rel[brace + 2..]
                .trim_end()
                .strip_suffix('}')
                .ok_or_else(|| RestoreError::InvalidFormat(format!("unterminated props: {rel}")))?;
            (
                rel[..brace].trim().to_string(),
                parse_cypher_props(props_block)?,
            )
        }
        None => (rel.to_string(), Vec::new()),
    };
    Ok((src, edge_type, tgt, props))
}

/// Parse `(n<id>)` -> id.
fn parse_node_ref(s: &str) -> Result<u64, RestoreError> {
    let inner = s
        .trim()
        .strip_prefix('(')
        .and_then(|s| s.strip_suffix(')'))
        .ok_or_else(|| RestoreError::InvalidFormat(format!("malformed node ref: {s}")))?;
    parse_pid(inner.trim())
}

/// Parse `n<id>` -> id.
fn parse_pid(s: &str) -> Result<u64, RestoreError> {
    s.strip_prefix('n')
        .and_then(|d| d.trim().parse().ok())
        .ok_or_else(|| RestoreError::InvalidFormat(format!("bad node ref: {s}")))
}

/// Parse a cypher property block `key: <json>, key2: <json>` (no braces)
/// as emitted by `format_cypher_props`: bare identifier keys, JSON values.
fn parse_cypher_props(s: &str) -> Result<Vec<(String, Value)>, RestoreError> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for seg in split_top_level(s, ',') {
        let seg = seg.trim();
        if seg.is_empty() {
            continue;
        }
        let colon = find_top_level(seg, ':')
            .ok_or_else(|| RestoreError::InvalidFormat(format!("bad property: {seg}")))?;
        let key = seg[..colon].trim().to_string();
        let val_str = seg[colon + 1..].trim();
        let json: serde_json::Value = serde_json::from_str(val_str).map_err(|e| {
            RestoreError::Deserialization(format!("property value '{val_str}': {e}"))
        })?;
        out.push((key, json_to_value(&json)));
    }
    Ok(out)
}

/// Split `s` on `delim` at the top nesting level only (ignores delimiters
/// inside `"..."` strings and `[]` / `{}` brackets).
fn split_top_level(s: &str, delim: char) -> Vec<String> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut depth = 0i32;
    let mut in_str = false;
    let mut escaped = false;
    for (i, c) in s.char_indices() {
        if in_str {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '[' | '{' => depth += 1,
            ']' | '}' => depth -= 1,
            _ if c == delim && depth == 0 => {
                parts.push(s[start..i].to_string());
                start = i + c.len_utf8();
            }
            _ => {}
        }
    }
    parts.push(s[start..].to_string());
    parts
}

/// Byte index of the first top-level `delim` in `s`, or None.
fn find_top_level(s: &str, delim: char) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_str = false;
    let mut escaped = false;
    for (i, c) in s.char_indices() {
        if in_str {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '[' | '{' => depth += 1,
            ']' | '}' => depth -= 1,
            _ if c == delim && depth == 0 => return Some(i),
            _ => {}
        }
    }
    None
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
            if let Some(mv_val) = obj.get("_multi_vector") {
                if let Some(arr) = mv_val.as_array() {
                    let rows: Vec<Vec<f32>> = arr
                        .iter()
                        .filter_map(|row| row.as_array())
                        .map(|row| {
                            row.iter()
                                .filter_map(|x| x.as_f64().map(|f| f as f32))
                                .collect()
                        })
                        .collect();
                    if let Some(v) = Value::try_multi_vector(rows) {
                        return v;
                    }
                    return Value::Null;
                }
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
