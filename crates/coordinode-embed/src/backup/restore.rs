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
    only_labels: Option<&std::collections::HashSet<String>>,
) -> Result<RestoreStats, RestoreError> {
    let node_store = coordinode_modality::LocalNodeStore::new(engine);
    let mut stats = RestoreStats::default();
    // Selective restore: with `only_labels`, keep only nodes carrying a matching
    // label and drop edges whose endpoints were filtered out. Exports list nodes
    // before edges, so `kept` is complete by the time edges are read.
    let mut kept: std::collections::HashSet<u64> = std::collections::HashSet::new();

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
                let labels = json_labels(obj.get("labels"));
                if let Some(filter) = only_labels {
                    if !labels.iter().any(|l| filter.contains(l)) {
                        continue;
                    }
                    kept.insert(id);
                }
                write_node_record(
                    &node_store,
                    interner,
                    shard_id,
                    id,
                    labels,
                    obj.get("properties").and_then(|v| v.as_object()),
                )?;
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
                if only_labels.is_some() && (!kept.contains(&source) || !kept.contains(&target)) {
                    continue;
                }
                write_edge_record(
                    engine,
                    interner,
                    source,
                    target,
                    edge_type,
                    obj.get("properties").and_then(|v| v.as_object()),
                )?;
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

/// Restore from a Neo4j APOC json-export dump (`apoc.export.json.all`).
///
/// APOC emits JSON Lines of `{"type":"node"|"relationship", ...}` where ids
/// are stringified Neo4j internal ids. This is the same on-disk path our own
/// json restore uses, modulo three mechanical differences handled here:
/// string ids, a `relationship` record (vs our `edge`) carrying nested
/// `start`/`end` objects, and `label` (vs `edge_type`). We never execute
/// APOC; this reads its portable output and writes straight to storage, the
/// same way [`restore_json`] does. Records of other types (graph metadata)
/// are skipped.
pub fn restore_apoc_json<R: BufRead>(
    engine: &StorageEngine,
    interner: &mut FieldInterner,
    shard_id: u16,
    reader: &mut R,
    only_labels: Option<&std::collections::HashSet<String>>,
) -> Result<RestoreStats, RestoreError> {
    let node_store = coordinode_modality::LocalNodeStore::new(engine);
    let mut stats = RestoreStats::default();
    // Selective restore: keep only label-matching nodes; drop edges to dropped
    // nodes (APOC lists nodes before relationships).
    let mut kept: std::collections::HashSet<u64> = std::collections::HashSet::new();

    for line_result in reader.lines() {
        let line = line_result?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let obj: serde_json::Value = serde_json::from_str(line)
            .map_err(|e| RestoreError::Deserialization(format!("invalid JSON: {e}")))?;

        match obj.get("type").and_then(|v| v.as_str()) {
            Some("node") => {
                let id = apoc_id(obj.get("id"), "node")?;
                let labels = json_labels(obj.get("labels"));
                if let Some(filter) = only_labels {
                    if !labels.iter().any(|l| filter.contains(l)) {
                        continue;
                    }
                    kept.insert(id);
                }
                write_node_record(
                    &node_store,
                    interner,
                    shard_id,
                    id,
                    labels,
                    obj.get("properties").and_then(|v| v.as_object()),
                )?;
                stats.nodes += 1;
            }
            Some("relationship") => {
                let source = apoc_id(
                    obj.get("start").and_then(|s| s.get("id")),
                    "relationship start",
                )?;
                let target = apoc_id(obj.get("end").and_then(|e| e.get("id")), "relationship end")?;
                let edge_type = obj.get("label").and_then(|v| v.as_str()).ok_or_else(|| {
                    RestoreError::InvalidFormat("relationship missing 'label'".into())
                })?;
                if only_labels.is_some() && (!kept.contains(&source) || !kept.contains(&target)) {
                    continue;
                }
                write_edge_record(
                    engine,
                    interner,
                    source,
                    target,
                    edge_type,
                    obj.get("properties").and_then(|v| v.as_object()),
                )?;
                stats.edges += 1;
            }
            _ => {}
        }
    }

    Ok(stats)
}

/// Collect a JSON labels array into owned strings.
fn json_labels(v: Option<&serde_json::Value>) -> Vec<String> {
    v.and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default()
}

/// Parse an APOC stringified internal id into a `u64`. Tolerates a raw
/// numeric id too, so non-standard exporters that skip the quoting still load.
fn apoc_id(v: Option<&serde_json::Value>, what: &str) -> Result<u64, RestoreError> {
    let v = v.ok_or_else(|| RestoreError::InvalidFormat(format!("{what} missing id")))?;
    if let Some(n) = v.as_u64() {
        return Ok(n);
    }
    v.as_str()
        .and_then(|s| s.parse::<u64>().ok())
        .ok_or_else(|| RestoreError::InvalidFormat(format!("{what} id is not a numeric string")))
}

/// Write one node (labels + properties) to storage, interning property names.
fn write_node_record(
    node_store: &coordinode_modality::LocalNodeStore,
    interner: &mut FieldInterner,
    shard_id: u16,
    id: u64,
    labels: Vec<String>,
    props: Option<&serde_json::Map<String, serde_json::Value>>,
) -> Result<(), RestoreError> {
    use coordinode_core::graph::node::NodeRecord;
    use coordinode_modality::NodeStore as _;

    let mut record = NodeRecord::with_labels(labels);
    if let Some(props) = props {
        for (name, json_val) in props {
            let field_id = interner.intern(name);
            record.set(field_id, json_to_value(json_val));
        }
    }
    node_store
        .put(shard_id, NodeId::from_raw(id), &record)
        .map_err(|e| RestoreError::Storage(e.to_string()))
}

/// Write one edge: both adjacency directions (merge operator, raw keys) plus
/// optional edge properties in executor-native shape.
fn write_edge_record(
    engine: &StorageEngine,
    interner: &mut FieldInterner,
    source: u64,
    target: u64,
    edge_type: &str,
    props: Option<&serde_json::Map<String, serde_json::Value>>,
) -> Result<(), RestoreError> {
    use coordinode_core::graph::edge::{encode_adj_key_forward, encode_adj_key_reverse};

    let fwd_key = encode_adj_key_forward(edge_type, NodeId::from_raw(source));
    engine
        .merge(
            Partition::Adj,
            &fwd_key,
            &coordinode_storage::engine::merge::encode_add(target),
        )
        .map_err(|e| RestoreError::Storage(e.to_string()))?;

    let rev_key = encode_adj_key_reverse(edge_type, NodeId::from_raw(target));
    engine
        .merge(
            Partition::Adj,
            &rev_key,
            &coordinode_storage::engine::merge::encode_add(source),
        )
        .map_err(|e| RestoreError::Storage(e.to_string()))?;

    if let Some(props) = props {
        if !props.is_empty() {
            // Executor-native wire shape: Vec<(field_id, Value)> (runner.rs
            // decodes exactly this). A HashMap would make the restored props
            // unreadable by queries.
            let prop_vec: Vec<(u32, Value)> = props
                .iter()
                .map(|(name, json_val)| (interner.intern(name), json_to_value(json_val)))
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
    Ok(())
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

/// Restore from a Neo4j APOC cypher-export dump (`apoc.export.cypher.all`).
///
/// This is a structural parser, NOT a Cypher engine: it recognises the
/// statement shapes APOC emits and writes node/edge records straight to
/// storage, the same way [`restore_cypher`] does for our own dumps. APOC is
/// never executed (it is a Neo4j plugin that does not run correctly on a
/// sharded cluster).
///
/// Both APOC export modes are handled:
/// - **plain** (`useOptimizations: {type: "NONE"}`): one `CREATE (...)`
///   statement per node and a `MATCH (...), (...) CREATE (a)-[:T]->(b)`
///   per relationship. Node ids come from the `UNIQUE IMPORT ID` property.
/// - **unwind-batch** (APOC's default): `UNWIND [ {...}, ... ] AS row CREATE
///   (n{...: row._id}) SET n += row.properties` and the relationship variant.
///
/// Schema statements (`CREATE CONSTRAINT` / `CREATE INDEX` / `DROP ...`),
/// transaction markers (`:begin` / `:commit` / `BEGIN` / `COMMIT`), and the
/// `UNIQUE IMPORT LABEL` cleanup pass are skipped: CoordiNode rebuilds those
/// natively. Cypher functions / temporal literals in values are a hard error
/// (we import data, not evaluate Cypher).
pub fn restore_apoc_cypher<R: BufRead>(
    engine: &StorageEngine,
    interner: &mut FieldInterner,
    shard_id: u16,
    reader: &mut R,
) -> Result<RestoreStats, RestoreError> {
    let node_store = coordinode_modality::LocalNodeStore::new(engine);
    let mut stats = RestoreStats::default();

    let mut text = String::new();
    reader.read_to_string(&mut text)?;

    // Transaction markers sit on their own line with no `;`, so they would
    // otherwise glue onto the following statement and hide it. Drop them
    // before splitting; they carry no graph data.
    let filtered = text
        .lines()
        .filter(|l| {
            let u = l.trim().to_uppercase();
            !(matches!(u.as_str(), "BEGIN" | "COMMIT" | ":BEGIN" | ":COMMIT")
                || u.starts_with("SCHEMA AWAIT"))
        })
        .collect::<Vec<_>>()
        .join("\n");

    for stmt in split_statements(&filtered) {
        let stmt = stmt.trim();
        if stmt.is_empty() {
            continue;
        }
        apply_apoc_cypher_stmt(stmt, engine, &node_store, interner, shard_id, &mut stats)?;
    }
    Ok(stats)
}

/// Classify and apply one APOC cypher statement.
fn apply_apoc_cypher_stmt(
    stmt: &str,
    engine: &StorageEngine,
    node_store: &coordinode_modality::LocalNodeStore,
    interner: &mut FieldInterner,
    shard_id: u16,
    stats: &mut RestoreStats,
) -> Result<(), RestoreError> {
    let upper = stmt.to_uppercase();

    // Transaction markers, schema DDL, and the import-label cleanup pass:
    // CoordiNode rebuilds schema natively, so these carry no graph data.
    let is_skippable = stmt.starts_with("//")
        || matches!(upper.as_str(), "BEGIN" | "COMMIT" | ":BEGIN" | ":COMMIT")
        || upper.starts_with("SCHEMA AWAIT")
        || upper.starts_with("CREATE CONSTRAINT")
        || upper.starts_with("DROP CONSTRAINT")
        || upper.starts_with("CREATE INDEX")
        || upper.starts_with("DROP INDEX")
        || upper.starts_with("CREATE RANGE INDEX")
        || upper.starts_with("CREATE TEXT INDEX")
        || upper.starts_with("CREATE POINT INDEX")
        || upper.starts_with("CREATE LOOKUP INDEX")
        || upper.starts_with("CREATE FULLTEXT INDEX")
        || upper.starts_with("CREATE VECTOR INDEX")
        || (upper.starts_with("MATCH (N:") && upper.contains("REMOVE"));
    if is_skippable {
        return Ok(());
    }

    if upper.starts_with("UNWIND") {
        apply_apoc_unwind(stmt, engine, node_store, interner, shard_id, stats)
    } else if upper.starts_with("CREATE (") || upper.starts_with("CREATE(") {
        apply_apoc_plain_node(stmt, node_store, interner, shard_id, stats)
    } else if upper.starts_with("MATCH") && stmt.contains("]->") {
        apply_apoc_plain_rel(stmt, engine, interner, stats)
    } else {
        // Unknown maintenance statement (e.g. a vendor-specific clause):
        // skip rather than fail; only CREATE/UNWIND carry graph data.
        Ok(())
    }
}

/// Apply an `UNWIND [...] AS row CREATE/MATCH ...` batch (node or relationship).
fn apply_apoc_unwind(
    stmt: &str,
    engine: &StorageEngine,
    node_store: &coordinode_modality::LocalNodeStore,
    interner: &mut FieldInterner,
    shard_id: u16,
    stats: &mut RestoreStats,
) -> Result<(), RestoreError> {
    let lb = stmt
        .find('[')
        .ok_or_else(|| RestoreError::InvalidFormat("UNWIND without a list".into()))?;
    let mut lit = CypherLit::new(&stmt[lb..]);
    let rows = lit.list()?;
    let rest = lit.rest();
    let rows = rows
        .as_array()
        .ok_or_else(|| RestoreError::InvalidFormat("UNWIND list is not an array".into()))?;

    if rest.contains("]->") {
        let edge_type = extract_reltype(&rest)?;
        for row in rows {
            let source = nested_id(row, "start")?;
            let target = nested_id(row, "end")?;
            let props = row.get("properties").and_then(|v| v.as_object());
            write_edge_record(engine, interner, source, target, &edge_type, props)?;
            stats.edges += 1;
        }
    } else {
        let labels = extract_create_labels(&rest)?;
        for row in rows {
            let id = row
                .get("_id")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| RestoreError::InvalidFormat("UNWIND node row missing _id".into()))?;
            let props = row.get("properties").and_then(|v| v.as_object());
            write_node_record(node_store, interner, shard_id, id, labels.clone(), props)?;
            stats.nodes += 1;
        }
    }
    Ok(())
}

/// `row["<side>"]["_id"]` as a `u64` (relationship endpoint).
fn nested_id(row: &serde_json::Value, side: &str) -> Result<u64, RestoreError> {
    row.get(side)
        .and_then(|s| s.get("_id"))
        .and_then(|v| v.as_u64())
        .ok_or_else(|| RestoreError::InvalidFormat(format!("UNWIND rel row missing {side}._id")))
}

/// Apply a plain `CREATE (:Labels {props, `UNIQUE IMPORT ID`: N});` node.
fn apply_apoc_plain_node(
    stmt: &str,
    node_store: &coordinode_modality::LocalNodeStore,
    interner: &mut FieldInterner,
    shard_id: u16,
    stats: &mut RestoreStats,
) -> Result<(), RestoreError> {
    let body = stmt["CREATE".len()..].trim();
    let inner = body
        .strip_prefix('(')
        .and_then(|s| s.strip_suffix(')'))
        .ok_or_else(|| RestoreError::InvalidFormat(format!("malformed node: {stmt}")))?;
    let brace = first_brace(inner).ok_or_else(|| {
        RestoreError::InvalidFormat(format!("node without UNIQUE IMPORT ID props: {stmt}"))
    })?;
    let labels = parse_label_list(&inner[..brace]);
    let map = CypherLit::new(&inner[brace..]).map()?;
    let mut obj = map
        .as_object()
        .cloned()
        .ok_or_else(|| RestoreError::InvalidFormat("node props not a map".into()))?;
    let id = obj
        .remove("UNIQUE IMPORT ID")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| {
            RestoreError::InvalidFormat(
                "plain node missing `UNIQUE IMPORT ID` (constraint-keyed nodes are not supported)"
                    .into(),
            )
        })?;
    write_node_record(node_store, interner, shard_id, id, labels, Some(&obj))?;
    stats.nodes += 1;
    Ok(())
}

/// Apply a plain `MATCH (a{id:X}), (b{id:Y}) CREATE (a)-[:T {props}]->(b);`.
fn apply_apoc_plain_rel(
    stmt: &str,
    engine: &StorageEngine,
    interner: &mut FieldInterner,
    stats: &mut RestoreStats,
) -> Result<(), RestoreError> {
    let cpos = find_top_keyword(stmt, "CREATE")
        .ok_or_else(|| RestoreError::InvalidFormat(format!("rel without CREATE: {stmt}")))?;
    let (match_part, create_part) = stmt.split_at(cpos);
    let ids = extract_import_ids(match_part);
    if ids.len() != 2 {
        return Err(RestoreError::InvalidFormat(format!(
            "expected 2 UNIQUE IMPORT IDs in MATCH, got {}: {stmt}",
            ids.len()
        )));
    }
    let edge_type = extract_reltype(create_part)?;
    let props = extract_rel_props(create_part)?;
    write_edge_record(engine, interner, ids[0], ids[1], &edge_type, props.as_ref())?;
    stats.edges += 1;
    Ok(())
}

/// Pull the relationship type out of a `-[var:`TYPE` {props}]->` pattern.
fn extract_reltype(s: &str) -> Result<String, RestoreError> {
    let open = s
        .find("-[")
        .ok_or_else(|| RestoreError::InvalidFormat("missing relationship pattern".into()))?;
    let close = s[open..]
        .find("]->")
        .map(|i| open + i)
        .ok_or_else(|| RestoreError::InvalidFormat("missing `]->`".into()))?;
    let inner = &s[open + 2..close];
    let colon = inner
        .find(':')
        .ok_or_else(|| RestoreError::InvalidFormat("relationship missing type".into()))?;
    Ok(strip_label_token(&inner[colon + 1..]))
}

/// Properties of a `-[var:`TYPE` {props}]->` pattern, if any.
fn extract_rel_props(
    s: &str,
) -> Result<Option<serde_json::Map<String, serde_json::Value>>, RestoreError> {
    let open = s
        .find("-[")
        .ok_or_else(|| RestoreError::InvalidFormat("missing relationship pattern".into()))?;
    let close = s[open..]
        .find("]->")
        .map(|i| open + i)
        .ok_or_else(|| RestoreError::InvalidFormat("missing `]->`".into()))?;
    let inner = &s[open + 2..close];
    match first_brace(inner) {
        Some(b) => Ok(CypherLit::new(&inner[b..]).map()?.as_object().cloned()),
        None => Ok(None),
    }
}

/// Labels of a `CREATE (var:`L1`:`L2` ...)` clause inside a larger statement.
fn extract_create_labels(s: &str) -> Result<Vec<String>, RestoreError> {
    let open = s
        .find("CREATE (")
        .map(|i| i + "CREATE (".len())
        .or_else(|| s.find("CREATE(").map(|i| i + "CREATE(".len()))
        .ok_or_else(|| RestoreError::InvalidFormat("UNWIND batch without CREATE".into()))?;
    let head_end = s[open..]
        .find(['{', ')'])
        .map(|i| open + i)
        .unwrap_or(s.len());
    Ok(parse_label_list(&s[open..head_end]))
}

/// Every integer following a `UNIQUE IMPORT ID` occurrence, in order. Used to
/// read both relationship endpoints out of a plain `MATCH ..., ...` head.
fn extract_import_ids(s: &str) -> Vec<u64> {
    const NEEDLE: &str = "UNIQUE IMPORT ID";
    let mut ids = Vec::new();
    let mut from = 0;
    while let Some(p) = s[from..].find(NEEDLE) {
        let after = from + p + NEEDLE.len();
        if let Some(colon) = s[after..].find(':') {
            let num: String = s[after + colon + 1..]
                .trim_start()
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            if let Ok(n) = num.parse::<u64>() {
                ids.push(n);
            }
        }
        from = after;
    }
    ids
}

/// Split `head` (`var:`L1`:`L2``) into labels, dropping the leading variable
/// and the synthetic `UNIQUE IMPORT LABEL`.
fn parse_label_list(head: &str) -> Vec<String> {
    let mut segs = Vec::new();
    let mut buf = String::new();
    let mut in_bt = false;
    for c in head.chars() {
        match c {
            '`' => {
                in_bt = !in_bt;
                buf.push(c);
            }
            ':' if !in_bt => segs.push(std::mem::take(&mut buf)),
            _ => buf.push(c),
        }
    }
    segs.push(buf);
    segs.into_iter()
        .skip(1) // first segment is the node variable, not a label
        .map(|s| s.trim().trim_matches('`').to_string())
        .filter(|s| !s.is_empty() && s != "UNIQUE IMPORT LABEL")
        .collect()
}

/// A single label/type token: backtick-quoted `` `Name` `` or bare, stopping
/// at whitespace or `{`.
fn strip_label_token(s: &str) -> String {
    let s = s.trim();
    if let Some(rest) = s.strip_prefix('`') {
        if let Some(end) = rest.find('`') {
            return rest[..end].to_string();
        }
    }
    s.split(|c: char| c.is_whitespace() || c == '{')
        .next()
        .unwrap_or("")
        .to_string()
}

/// Byte index of the first `{` not inside a quoted run, or None.
fn first_brace(s: &str) -> Option<usize> {
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for (i, c) in s.char_indices() {
        if let Some(q) = quote {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == q {
                quote = None;
            }
            continue;
        }
        match c {
            '"' | '\'' | '`' => quote = Some(c),
            '{' => return Some(i),
            _ => {}
        }
    }
    None
}

/// Byte index of `kw` at bracket-depth 0 outside any quoted run, or None.
fn find_top_keyword(s: &str, kw: &str) -> Option<usize> {
    let mut depth = 0i32;
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for (i, c) in s.char_indices() {
        if let Some(q) = quote {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == q {
                quote = None;
            }
            continue;
        }
        match c {
            '"' | '\'' | '`' => quote = Some(c),
            '[' | '{' | '(' => depth += 1,
            ']' | '}' | ')' => depth -= 1,
            _ if depth == 0 && s[i..].starts_with(kw) => return Some(i),
            _ => {}
        }
    }
    None
}

/// Split a cypher script into statements on top-level `;`, honoring quoted
/// runs (`"`, `'`, `` ` ``) and `()` / `[]` / `{}` nesting.
fn split_statements(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut depth = 0i32;
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for c in text.chars() {
        if let Some(q) = quote {
            buf.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == q {
                quote = None;
            }
            continue;
        }
        match c {
            '"' | '\'' | '`' => {
                quote = Some(c);
                buf.push(c);
            }
            '[' | '{' | '(' => {
                depth += 1;
                buf.push(c);
            }
            ']' | '}' | ')' => {
                depth -= 1;
                buf.push(c);
            }
            ';' if depth == 0 => out.push(std::mem::take(&mut buf)),
            _ => buf.push(c),
        }
    }
    if !buf.trim().is_empty() {
        out.push(buf);
    }
    out
}

/// A minimal recursive parser for the Cypher literal subset APOC emits: maps
/// with backtick / bare / quoted keys, lists, double/single-quoted strings,
/// numbers, `true`/`false`/`null`, and arbitrary nesting. It does NOT evaluate
/// Cypher functions (`datetime(...)` etc.) — such a value is a hard error,
/// because this path imports data, it does not execute Cypher.
struct CypherLit {
    chars: Vec<char>,
    pos: usize,
}

impl CypherLit {
    fn new(s: &str) -> Self {
        Self {
            chars: s.chars().collect(),
            pos: 0,
        }
    }

    /// Remaining unconsumed input (after a `list`/`map` parse).
    fn rest(&self) -> String {
        self.chars[self.pos..].iter().collect()
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.chars.get(self.pos).copied();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(c) if c.is_whitespace()) {
            self.pos += 1;
        }
    }

    fn err(msg: impl Into<String>) -> RestoreError {
        RestoreError::InvalidFormat(format!("cypher literal: {}", msg.into()))
    }

    fn expect(&mut self, c: char) -> Result<(), RestoreError> {
        self.skip_ws();
        if self.peek() == Some(c) {
            self.pos += 1;
            Ok(())
        } else {
            Err(Self::err(format!("expected '{c}'")))
        }
    }

    fn value(&mut self) -> Result<serde_json::Value, RestoreError> {
        self.skip_ws();
        match self.peek() {
            Some('{') => self.map(),
            Some('[') => self.list(),
            Some('"') | Some('\'') => Ok(serde_json::Value::String(self.string()?)),
            Some(c) if c == '-' || c == '+' || c.is_ascii_digit() => self.number(),
            Some(_) => self.bareword(),
            None => Err(Self::err("unexpected end")),
        }
    }

    fn map(&mut self) -> Result<serde_json::Value, RestoreError> {
        self.expect('{')?;
        let mut obj = serde_json::Map::new();
        loop {
            self.skip_ws();
            match self.peek() {
                Some('}') => {
                    self.pos += 1;
                    break;
                }
                None => return Err(Self::err("unterminated map")),
                _ => {}
            }
            let key = self.key()?;
            self.expect(':')?;
            let val = self.value()?;
            obj.insert(key, val);
            self.skip_ws();
            match self.advance() {
                Some(',') => {}
                Some('}') => break,
                _ => return Err(Self::err("expected ',' or '}'")),
            }
        }
        Ok(serde_json::Value::Object(obj))
    }

    fn list(&mut self) -> Result<serde_json::Value, RestoreError> {
        self.expect('[')?;
        let mut arr = Vec::new();
        loop {
            self.skip_ws();
            match self.peek() {
                Some(']') => {
                    self.pos += 1;
                    break;
                }
                None => return Err(Self::err("unterminated list")),
                _ => {}
            }
            arr.push(self.value()?);
            self.skip_ws();
            match self.advance() {
                Some(',') => {}
                Some(']') => break,
                _ => return Err(Self::err("expected ',' or ']'")),
            }
        }
        Ok(serde_json::Value::Array(arr))
    }

    /// A map key: backtick-quoted, string-quoted, or a bare identifier.
    fn key(&mut self) -> Result<String, RestoreError> {
        self.skip_ws();
        match self.peek() {
            Some('`') => self.backtick(),
            Some('"') | Some('\'') => self.string(),
            Some(_) => {
                let start = self.pos;
                while let Some(c) = self.peek() {
                    if c.is_whitespace() || c == ':' {
                        break;
                    }
                    self.pos += 1;
                }
                if self.pos == start {
                    return Err(Self::err("empty key"));
                }
                Ok(self.chars[start..self.pos].iter().collect())
            }
            None => Err(Self::err("expected key")),
        }
    }

    fn backtick(&mut self) -> Result<String, RestoreError> {
        self.expect('`')?;
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c == '`' {
                let s: String = self.chars[start..self.pos].iter().collect();
                self.pos += 1;
                return Ok(s);
            }
            self.pos += 1;
        }
        Err(Self::err("unterminated backtick"))
    }

    fn string(&mut self) -> Result<String, RestoreError> {
        let quote = self.advance().ok_or_else(|| Self::err("expected string"))?;
        let mut out = String::new();
        while let Some(c) = self.advance() {
            if c == '\\' {
                match self.advance() {
                    Some('n') => out.push('\n'),
                    Some('t') => out.push('\t'),
                    Some('r') => out.push('\r'),
                    Some('b') => out.push('\u{8}'),
                    Some('f') => out.push('\u{c}'),
                    Some('u') => {
                        let mut code = 0u32;
                        for _ in 0..4 {
                            let h = self.advance().ok_or_else(|| Self::err("bad \\u escape"))?;
                            code = code * 16
                                + h.to_digit(16).ok_or_else(|| Self::err("bad \\u hex"))?;
                        }
                        out.push(char::from_u32(code).ok_or_else(|| Self::err("bad codepoint"))?);
                    }
                    Some(other) => out.push(other),
                    None => return Err(Self::err("dangling escape")),
                }
            } else if c == quote {
                return Ok(out);
            } else {
                out.push(c);
            }
        }
        Err(Self::err("unterminated string"))
    }

    fn number(&mut self) -> Result<serde_json::Value, RestoreError> {
        let start = self.pos;
        if matches!(self.peek(), Some('+') | Some('-')) {
            self.pos += 1;
        }
        let mut is_float = false;
        while let Some(c) = self.peek() {
            match c {
                '0'..='9' => self.pos += 1,
                '.' | 'e' | 'E' => {
                    is_float = true;
                    self.pos += 1;
                }
                '+' | '-' => self.pos += 1, // exponent sign
                _ => break,
            }
        }
        let s: String = self.chars[start..self.pos].iter().collect();
        if is_float {
            let f: f64 = s
                .parse()
                .map_err(|_| Self::err(format!("bad number '{s}'")))?;
            serde_json::Number::from_f64(f)
                .map(serde_json::Value::Number)
                .ok_or_else(|| Self::err("non-finite number"))
        } else {
            let i: i64 = s
                .parse()
                .map_err(|_| Self::err(format!("bad integer '{s}'")))?;
            Ok(serde_json::Value::Number(i.into()))
        }
    }

    fn bareword(&mut self) -> Result<serde_json::Value, RestoreError> {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' {
                self.pos += 1;
            } else {
                break;
            }
        }
        let word: String = self.chars[start..self.pos].iter().collect();
        match word.to_lowercase().as_str() {
            "true" => Ok(serde_json::Value::Bool(true)),
            "false" => Ok(serde_json::Value::Bool(false)),
            "null" => Ok(serde_json::Value::Null),
            _ => Err(Self::err(format!(
                "unsupported value '{word}' (cypher functions and temporals are not imported)"
            ))),
        }
    }
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

/// Restore from a Hetionet "hetnet" JSON document (the dhimmel/hetio source
/// format that the `hetnetpy` library turns into a Neo4j graph). The document
/// is a single object: `{nodes: [{kind, identifier, name, data}], edges:
/// [{source_id: [kind, id], target_id: [kind, id], kind, data}]}`.
///
/// Node identity is a `(kind, identifier)` pair (identifier may be a string or
/// an integer), so this mints a sequential node id per node and resolves edge
/// endpoints through that map. `kind` becomes the node label / relationship
/// type; `name` and `data` become properties. This mirrors what hetnetpy does
/// when it writes the hetnet into Neo4j, so the dataset loads straight from the
/// JSON with no Neo4j round trip.
pub fn restore_hetio_json<R: BufRead>(
    engine: &StorageEngine,
    interner: &mut FieldInterner,
    shard_id: u16,
    reader: &mut R,
    only_labels: Option<&std::collections::HashSet<String>>,
) -> Result<RestoreStats, RestoreError> {
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct HetNode {
        kind: String,
        identifier: serde_json::Value,
        #[serde(default)]
        name: Option<serde_json::Value>,
        #[serde(default)]
        data: serde_json::Map<String, serde_json::Value>,
    }
    #[derive(Deserialize)]
    struct HetEdge {
        source_id: (String, serde_json::Value),
        target_id: (String, serde_json::Value),
        kind: String,
        #[serde(default)]
        data: serde_json::Map<String, serde_json::Value>,
    }
    #[derive(Deserialize)]
    struct HetnetDoc {
        nodes: Vec<HetNode>,
        edges: Vec<HetEdge>,
    }

    let doc: HetnetDoc = serde_json::from_reader(reader)
        .map_err(|e| RestoreError::Deserialization(format!("hetnet json: {e}")))?;

    let node_store = coordinode_modality::LocalNodeStore::new(engine);
    let mut stats = RestoreStats::default();
    let mut id_map: std::collections::HashMap<(String, String), u64> =
        std::collections::HashMap::with_capacity(doc.nodes.len());

    for (idx, n) in doc.nodes.iter().enumerate() {
        let id = idx as u64;
        // Selective restore: a node whose kind is filtered out is never added to
        // the id map, so edges referencing it resolve to None and drop below.
        if let Some(filter) = only_labels {
            if !filter.contains(&n.kind) {
                continue;
            }
        }
        id_map.insert((n.kind.clone(), ident_key(&n.identifier)), id);
        let mut props = n.data.clone();
        if let Some(name) = &n.name {
            props.insert("name".to_string(), name.clone());
        }
        props.insert("identifier".to_string(), n.identifier.clone());
        write_node_record(
            &node_store,
            interner,
            shard_id,
            id,
            vec![n.kind.clone()],
            Some(&props),
        )?;
        stats.nodes += 1;
    }

    for e in &doc.edges {
        let src = id_map.get(&(e.source_id.0.clone(), ident_key(&e.source_id.1)));
        let tgt = id_map.get(&(e.target_id.0.clone(), ident_key(&e.target_id.1)));
        let (Some(&src), Some(&tgt)) = (src, tgt) else {
            // Endpoint not present in the node set: skip the dangling edge.
            continue;
        };
        let props = if e.data.is_empty() {
            None
        } else {
            Some(&e.data)
        };
        write_edge_record(engine, interner, src, tgt, &e.kind, props)?;
        stats.edges += 1;
    }
    Ok(stats)
}

/// Canonical string key for a hetnet identifier (string or integer) so a node
/// and the edge endpoints that reference it resolve to the same map entry.
fn ident_key(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::backup::export::value_to_json;

    #[test]
    fn cypher_lit_parses_map_with_backtick_keys_and_mixed_values() {
        let src = r#"{`name`:"admins", `count`:3, `active`:true, `score`:1.5, `tags`:["a","b"], `UNIQUE IMPORT ID`:7}"#;
        let v = CypherLit::new(src).map().unwrap();
        let obj = v.as_object().unwrap();
        assert_eq!(obj["name"], serde_json::json!("admins"));
        assert_eq!(obj["count"], serde_json::json!(3));
        assert_eq!(obj["active"], serde_json::json!(true));
        assert_eq!(obj["score"], serde_json::json!(1.5));
        assert_eq!(obj["tags"], serde_json::json!(["a", "b"]));
        assert_eq!(obj["UNIQUE IMPORT ID"], serde_json::json!(7));
    }

    #[test]
    fn cypher_lit_parses_float_embedding_array() {
        // The vector-embedding case: a list of floats must round-trip.
        let src = "[0.1, -0.25, 3.0, 1.0e-3]";
        let v = CypherLit::new(src).list().unwrap();
        let arr = v.as_array().unwrap();
        assert_eq!(arr.len(), 4);
        assert_eq!(arr[0].as_f64().unwrap(), 0.1);
        assert_eq!(arr[1].as_f64().unwrap(), -0.25);
        assert_eq!(arr[3].as_f64().unwrap(), 0.001);
    }

    #[test]
    fn cypher_lit_rest_exposes_trailing_input() {
        let mut lit = CypherLit::new("[1,2] AS row CREATE (n)");
        let _ = lit.list().unwrap();
        assert_eq!(lit.rest().trim(), "AS row CREATE (n)");
    }

    #[test]
    fn cypher_lit_rejects_function_value() {
        // datetime(...) is a function, not data — must fail loudly.
        let err = CypherLit::new(r#"{`when`: datetime("2020")}"#).map();
        assert!(err.is_err());
    }

    #[test]
    fn split_statements_handles_multiline_and_semicolon_in_string() {
        let text = "CREATE (n {`s`:\"a;b\"});\nUNWIND [1,2] AS row\nCREATE (m);\n";
        let stmts: Vec<String> = split_statements(text)
            .into_iter()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        assert_eq!(stmts.len(), 2, "semicolon inside string must not split");
        assert!(stmts[1].starts_with("UNWIND"));
    }

    #[test]
    fn parse_label_list_drops_var_and_import_label() {
        assert_eq!(
            parse_label_list("n:`User`:`HighValue`"),
            vec!["User", "HighValue"]
        );
        // Anonymous node (no var) plus the synthetic import label.
        assert_eq!(
            parse_label_list(":`Group`:`UNIQUE IMPORT LABEL`"),
            vec!["Group"]
        );
    }

    #[test]
    fn extract_helpers_read_reltype_and_import_ids() {
        assert_eq!(
            extract_reltype("CREATE (a)-[r:`MEMBER_OF` {`since`:2020}]->(b)").unwrap(),
            "MEMBER_OF"
        );
        let ids = extract_import_ids(
            "MATCH (n1:`UNIQUE IMPORT LABEL`{`UNIQUE IMPORT ID`:5}), (n2:`L`{`UNIQUE IMPORT ID`:8})",
        );
        assert_eq!(ids, vec![5, 8]);
        assert_eq!(
            extract_create_labels("CREATE (n:`Person`{`UNIQUE IMPORT ID`: row._id}) SET n += x")
                .unwrap(),
            vec!["Person"]
        );
    }

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
