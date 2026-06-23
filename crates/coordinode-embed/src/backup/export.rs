//! Export graph data from CoordiNode storage to portable formats.
//!
//! Scans all partitions (Node, Adj, EdgeProp, Schema) and writes
//! each entity to the output writer in the selected format.

use std::collections::HashMap;
use std::io::Write;

use coordinode_core::graph::edge::PostingList;
use coordinode_core::graph::intern::FieldInterner;
use coordinode_core::graph::node::{self, NodeId, NodeRecord};
use coordinode_core::graph::types::Value;
use coordinode_modality::edge::{EdgeStore, LocalEdgeStore};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::StorageSnapshot;
use coordinode_storage::Guard;

/// Build node key prefix for a shard: `node:<shard_id_be_bytes>`
fn node_shard_prefix(shard_id: u16) -> Vec<u8> {
    let mut prefix = Vec::with_capacity(7);
    prefix.extend_from_slice(b"node:");
    prefix.extend_from_slice(&shard_id.to_be_bytes());
    prefix
}

/// Wire-format version of the binary backup dump. Bump on any
/// incompatible change to [`BackupEntry`] layout or to the value
/// encodings stored inside it. Restore refuses dumps newer than the
/// version it understands (see `restore_binary`).
pub const BINARY_FORMAT_VERSION: u32 = 1;

/// Human-readable producer string embedded in the binary dump manifest:
/// `coordinode-embed/<crate-version>`. Used only for diagnostics in
/// restore errors; never gates compatibility on its own.
pub fn producer_tag() -> String {
    concat!("coordinode-embed/", env!("CARGO_PKG_VERSION")).to_string()
}

/// Deterministic FNV-1a 64-bit fingerprint of the schema partition.
///
/// Hashes the schema `(key, value)` pairs in key-sorted order so the
/// result is stable across runs and independent of scan order. A dump's
/// fingerprint identifies the label / property-type contract it was
/// produced under; restoring into a non-empty database whose schema
/// fingerprint differs risks merging incompatible type definitions, so
/// `restore_binary` rejects that mismatch unless forced.
pub fn schema_fingerprint(engine: &StorageEngine) -> Result<u64, ExportError> {
    let iter = engine
        .prefix_scan(Partition::Schema, b"schema:")
        .map_err(|e| ExportError::Storage(e.to_string()))?;

    let mut pairs: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    for guard in iter {
        let (k, v) = guard
            .into_inner()
            .map_err(|e| ExportError::Storage(e.to_string()))?;
        pairs.push((k.to_vec(), v.to_vec()));
    }
    pairs.sort_by(|a, b| a.0.cmp(&b.0));

    // FNV-1a 64-bit: deterministic, no random seed, dependency-free.
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    let mut mix = |bytes: &[u8]| {
        for &b in bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        // Length-delimit each field so (k=ab, v=c) and (k=a, v=bc) differ.
        hash ^= 0xff;
        hash = hash.wrapping_mul(FNV_PRIME);
    };
    for (k, v) in &pairs {
        mix(k);
        mix(v);
    }
    Ok(hash)
}

/// Errors during backup export.
#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    #[error("storage error: {0}")]
    Storage(String),

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Export all graph data as JSON Lines to a writer.
///
/// Output format (one JSON object per line):
/// ```json
/// {"type":"node","id":1,"labels":["User"],"properties":{"name":"Alice","age":30}}
/// {"type":"edge","source":1,"target":2,"type":"FOLLOWS","properties":{}}
/// ```
///
/// Properties use resolved field names (not interned IDs).
pub fn export_json<W: Write>(
    engine: &StorageEngine,
    interner: &FieldInterner,
    shard_id: u16,
    snapshot: &StorageSnapshot,
    writer: &mut W,
) -> Result<ExportStats, ExportError> {
    let mut stats = ExportStats::default();

    // Export nodes (snapshot prefix_scan returns point-in-time consistent keys)
    let node_prefix = node_shard_prefix(shard_id);
    let entries: Vec<(Vec<u8>, Vec<u8>)> = engine
        .snapshot_prefix_scan(snapshot, Partition::Node, &node_prefix)
        .map_err(|e| ExportError::Storage(e.to_string()))?
        .into_iter()
        .map(|(k, v)| (k, v.to_vec()))
        .collect();

    for (key_bytes, value_bytes) in &entries {
        let Some((_shard, node_id)) = node::decode_node_key(key_bytes) else {
            continue;
        };

        let record = NodeRecord::from_msgpack(value_bytes)
            .map_err(|e| ExportError::Serialization(e.to_string()))?;

        let props = resolve_properties(&record.props, interner);

        let json = serde_json::json!({
            "type": "node",
            "id": node_id.as_raw(),
            "labels": record.labels,
            "properties": props,
        });

        writeln!(writer, "{json}")?;
        stats.nodes += 1;
    }

    // Export edges (forward adjacency only — reverse is derived)
    let adj_prefix = b"adj:";
    let entries: Vec<(Vec<u8>, Vec<u8>)> = {
        let iter = engine
            .prefix_scan(Partition::Adj, adj_prefix)
            .map_err(|e| ExportError::Storage(e.to_string()))?;
        let mut result = Vec::new();
        for guard in iter {
            let (k, v) = guard
                .into_inner()
                .map_err(|e| ExportError::Storage(e.to_string()))?;
            result.push((k.to_vec(), v.to_vec()));
        }
        result
    };

    for (key_bytes, value_bytes) in &entries {
        let key_str = match std::str::from_utf8(key_bytes) {
            Ok(s) => s,
            Err(_) => continue,
        };

        if !key_str.contains(":out:") {
            continue;
        }

        let Some((edge_type, source_id)) = parse_adj_forward_key(key_str) else {
            continue;
        };

        let posting_list = PostingList::from_bytes(value_bytes)
            .map_err(|e| ExportError::Serialization(e.to_string()))?;

        for target_id in posting_list.iter() {
            let edge_props =
                load_edge_properties(engine, snapshot, &edge_type, source_id, target_id, interner)?;

            let json = serde_json::json!({
                "type": "edge",
                "source": source_id,
                "target": target_id,
                "edge_type": edge_type,
                "properties": edge_props,
            });

            writeln!(writer, "{json}")?;
            stats.edges += 1;
        }
    }

    Ok(stats)
}

/// Export all graph data as OpenCypher CREATE statements.
///
/// Output:
/// ```cypher
/// CREATE (n1:User {name: "Alice", age: 30});
/// CREATE (n1)-[:FOLLOWS]->(n2);
/// ```
pub fn export_cypher<W: Write>(
    engine: &StorageEngine,
    interner: &FieldInterner,
    shard_id: u16,
    snapshot: &StorageSnapshot,
    writer: &mut W,
) -> Result<ExportStats, ExportError> {
    let mut stats = ExportStats::default();

    // Export nodes
    let node_prefix = node_shard_prefix(shard_id);
    let entries: Vec<(Vec<u8>, Vec<u8>)> = engine
        .snapshot_prefix_scan(snapshot, Partition::Node, &node_prefix)
        .map_err(|e| ExportError::Storage(e.to_string()))?
        .into_iter()
        .map(|(k, v)| (k, v.to_vec()))
        .collect();

    for (key_bytes, value_bytes) in &entries {
        let Some((_shard, node_id)) = node::decode_node_key(key_bytes) else {
            continue;
        };

        let record = NodeRecord::from_msgpack(value_bytes)
            .map_err(|e| ExportError::Serialization(e.to_string()))?;

        let props = resolve_properties(&record.props, interner);
        let labels = record.labels.join(":");
        let props_str = format_cypher_props(&props);

        let id = node_id.as_raw();
        if props_str.is_empty() {
            writeln!(writer, "CREATE (n{id}:{labels});")?;
        } else {
            writeln!(writer, "CREATE (n{id}:{labels} {{{props_str}}});")?;
        }
        stats.nodes += 1;
    }

    // Export edges
    let adj_prefix = b"adj:";
    let entries: Vec<(Vec<u8>, Vec<u8>)> = {
        let iter = engine
            .prefix_scan(Partition::Adj, adj_prefix)
            .map_err(|e| ExportError::Storage(e.to_string()))?;
        let mut result = Vec::new();
        for guard in iter {
            let (k, v) = guard
                .into_inner()
                .map_err(|e| ExportError::Storage(e.to_string()))?;
            result.push((k.to_vec(), v.to_vec()));
        }
        result
    };

    for (key_bytes, value_bytes) in &entries {
        let key_str = match std::str::from_utf8(key_bytes) {
            Ok(s) => s,
            Err(_) => continue,
        };

        if !key_str.contains(":out:") {
            continue;
        }

        let Some((edge_type, source_id)) = parse_adj_forward_key(key_str) else {
            continue;
        };

        let posting_list = PostingList::from_bytes(value_bytes)
            .map_err(|e| ExportError::Serialization(e.to_string()))?;

        for target_id in posting_list.iter() {
            let edge_props =
                load_edge_properties(engine, snapshot, &edge_type, source_id, target_id, interner)?;

            if edge_props.is_empty() {
                writeln!(
                    writer,
                    "CREATE (n{source_id})-[:{edge_type}]->(n{target_id});"
                )?;
            } else {
                let props_str = format_cypher_props(&edge_props);
                writeln!(
                    writer,
                    "CREATE (n{source_id})-[:{edge_type} {{{props_str}}}]->(n{target_id});"
                )?;
            }
            stats.edges += 1;
        }
    }

    Ok(stats)
}

/// Export all graph data as binary (MessagePack) dump.
///
/// Format: sequence of MessagePack-encoded `BackupEntry` values.
/// Not human-readable but fastest for backup/restore.
pub fn export_binary<W: Write>(
    engine: &StorageEngine,
    interner: &FieldInterner,
    shard_id: u16,
    snapshot: &StorageSnapshot,
    writer: &mut W,
) -> Result<ExportStats, ExportError> {
    let mut stats = ExportStats::default();

    // Manifest first so restore can validate compatibility before any write.
    let manifest = BackupEntry::Manifest {
        format_version: BINARY_FORMAT_VERSION,
        producer: producer_tag(),
        schema_fingerprint: schema_fingerprint(engine)?,
    };
    let encoded =
        rmp_serde::to_vec(&manifest).map_err(|e| ExportError::Serialization(e.to_string()))?;
    writer.write_all(&(encoded.len() as u32).to_le_bytes())?;
    writer.write_all(&encoded)?;

    // Interner second (needed for restore)
    let interner_bytes = interner.to_bytes();
    let header = BackupEntry::Interner(interner_bytes);
    let encoded =
        rmp_serde::to_vec(&header).map_err(|e| ExportError::Serialization(e.to_string()))?;
    writer.write_all(&(encoded.len() as u32).to_le_bytes())?;
    writer.write_all(&encoded)?;

    // Export nodes (snapshot-consistent user keys)
    let node_prefix = node_shard_prefix(shard_id);
    let entries: Vec<(Vec<u8>, Vec<u8>)> = engine
        .snapshot_prefix_scan(snapshot, Partition::Node, &node_prefix)
        .map_err(|e| ExportError::Storage(e.to_string()))?
        .into_iter()
        .map(|(k, v)| (k, v.to_vec()))
        .collect();

    for (key_bytes, value_bytes) in &entries {
        let entry = BackupEntry::Node {
            key: key_bytes.clone(),
            value: value_bytes.clone(),
        };
        let encoded =
            rmp_serde::to_vec(&entry).map_err(|e| ExportError::Serialization(e.to_string()))?;
        writer.write_all(&(encoded.len() as u32).to_le_bytes())?;
        writer.write_all(&encoded)?;
        stats.nodes += 1;
    }

    // Export adjacency
    let adj_prefix = b"adj:";
    let entries: Vec<(Vec<u8>, Vec<u8>)> = {
        let iter = engine
            .prefix_scan(Partition::Adj, adj_prefix)
            .map_err(|e| ExportError::Storage(e.to_string()))?;
        let mut result = Vec::new();
        for guard in iter {
            let (k, v) = guard
                .into_inner()
                .map_err(|e| ExportError::Storage(e.to_string()))?;
            result.push((k.to_vec(), v.to_vec()));
        }
        result
    };

    for (key_bytes, value_bytes) in &entries {
        let entry = BackupEntry::Adj {
            key: key_bytes.clone(),
            value: value_bytes.clone(),
        };
        let encoded =
            rmp_serde::to_vec(&entry).map_err(|e| ExportError::Serialization(e.to_string()))?;
        writer.write_all(&(encoded.len() as u32).to_le_bytes())?;
        writer.write_all(&encoded)?;

        let key_str = std::str::from_utf8(key_bytes).unwrap_or("");
        if key_str.contains(":out:") {
            if let Ok(pl) = PostingList::from_bytes(value_bytes) {
                stats.edges += pl.len() as u64;
            }
        }
    }

    // Export edge properties
    let ep_prefix = b"edgeprop:";
    let entries: Vec<(Vec<u8>, Vec<u8>)> = engine
        .snapshot_prefix_scan(snapshot, Partition::EdgeProp, ep_prefix)
        .map_err(|e| ExportError::Storage(e.to_string()))?
        .into_iter()
        .map(|(k, v)| (k, v.to_vec()))
        .collect();

    for (key_bytes, value_bytes) in &entries {
        let entry = BackupEntry::EdgeProp {
            key: key_bytes.clone(),
            value: value_bytes.clone(),
        };
        let encoded =
            rmp_serde::to_vec(&entry).map_err(|e| ExportError::Serialization(e.to_string()))?;
        writer.write_all(&(encoded.len() as u32).to_le_bytes())?;
        writer.write_all(&encoded)?;
    }

    // Export schema (schema is not MVCC-versioned — read directly)
    let schema_prefix = b"schema:";
    let iter = engine
        .prefix_scan(Partition::Schema, schema_prefix)
        .map_err(|e| ExportError::Storage(e.to_string()))?;

    for guard in iter {
        let (key_bytes, value_bytes) = guard
            .into_inner()
            .map_err(|e| ExportError::Storage(e.to_string()))?;

        let entry = BackupEntry::Schema {
            key: key_bytes.to_vec(),
            value: value_bytes.to_vec(),
        };
        let encoded =
            rmp_serde::to_vec(&entry).map_err(|e| ExportError::Serialization(e.to_string()))?;
        writer.write_all(&(encoded.len() as u32).to_le_bytes())?;
        writer.write_all(&encoded)?;
    }

    Ok(stats)
}

/// A single entry in a binary backup dump.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub enum BackupEntry {
    /// Dump manifest. Written as the very first entry so restore can
    /// validate compatibility before touching storage.
    Manifest {
        /// Wire-format version; see [`BINARY_FORMAT_VERSION`].
        format_version: u32,
        /// Producing build, e.g. `coordinode-embed/0.4.3` (diagnostics only).
        producer: String,
        /// FNV-1a fingerprint of the source schema partition; see
        /// [`schema_fingerprint`].
        schema_fingerprint: u64,
    },
    /// Field interner state (follows the manifest).
    Interner(Vec<u8>),
    /// Node: raw key + MessagePack value.
    Node { key: Vec<u8>, value: Vec<u8> },
    /// Adjacency list: raw key + MessagePack posting list.
    Adj { key: Vec<u8>, value: Vec<u8> },
    /// Edge properties: raw key + MessagePack value.
    EdgeProp { key: Vec<u8>, value: Vec<u8> },
    /// Schema metadata: raw key + MessagePack value.
    Schema { key: Vec<u8>, value: Vec<u8> },
}

/// Statistics from a backup export operation.
#[derive(Debug, Default, Clone)]
pub struct ExportStats {
    pub nodes: u64,
    pub edges: u64,
}

// -- Internal helpers --

/// Resolve interned property IDs to human-readable names.
fn resolve_properties(
    props: &HashMap<u32, Value>,
    interner: &FieldInterner,
) -> serde_json::Map<String, serde_json::Value> {
    let mut map = serde_json::Map::new();
    for (&field_id, value) in props {
        let name = interner
            .resolve(field_id)
            .unwrap_or("_unknown_")
            .to_string();
        map.insert(name, value_to_json(value));
    }
    map
}

/// Convert a CoordiNode Value to serde_json::Value.
pub(crate) fn value_to_json(value: &Value) -> serde_json::Value {
    match value {
        Value::Null => serde_json::Value::Null,
        Value::Bool(b) => serde_json::Value::Bool(*b),
        Value::Int(i) => serde_json::json!(i),
        Value::Float(f) => serde_json::json!(f),
        Value::String(s) => serde_json::Value::String(s.clone()),
        Value::Timestamp(ts) => serde_json::json!({"_timestamp": ts}),
        Value::Vector(v) => serde_json::json!(v),
        Value::Blob(b) => serde_json::json!({"_blob_ref": hex::encode(b)}),
        Value::Array(arr) => {
            let items: Vec<serde_json::Value> = arr.iter().map(value_to_json).collect();
            serde_json::Value::Array(items)
        }
        Value::Map(m) => {
            let obj: serde_json::Map<String, serde_json::Value> = m
                .iter()
                .map(|(k, v)| (k.clone(), value_to_json(v)))
                .collect();
            serde_json::Value::Object(obj)
        }
        Value::Geo(g) => serde_json::json!({"_geo": g}),
        Value::Binary(b) => serde_json::json!({"_binary": hex::encode(b)}),
        Value::Document(v) => serde_json::json!({"_document": rmpv_to_json(v)}),
        Value::MultiVector(rows) => serde_json::json!({"_multi_vector": rows}),
        Value::Path(p) => serde_json::json!({"_path": {
            "nodes": p.nodes,
            "rels": p.rels.iter().map(|r| serde_json::json!({
                "type": r.edge_type,
                "source": r.source,
                "target": r.target,
            })).collect::<Vec<_>>(),
        }}),
    }
}

/// Convert an rmpv::Value to serde_json::Value for export.
fn rmpv_to_json(v: &rmpv::Value) -> serde_json::Value {
    match v {
        rmpv::Value::Nil => serde_json::Value::Null,
        rmpv::Value::Boolean(b) => serde_json::Value::Bool(*b),
        rmpv::Value::Integer(i) => {
            if let Some(n) = i.as_i64() {
                serde_json::json!(n)
            } else if let Some(n) = i.as_u64() {
                serde_json::json!(n)
            } else {
                serde_json::Value::Null
            }
        }
        rmpv::Value::F32(f) => serde_json::json!(f),
        rmpv::Value::F64(f) => serde_json::json!(f),
        rmpv::Value::String(s) => {
            serde_json::Value::String(s.as_str().unwrap_or_default().to_string())
        }
        rmpv::Value::Binary(b) => serde_json::json!({"_binary": hex::encode(b)}),
        rmpv::Value::Array(arr) => serde_json::Value::Array(arr.iter().map(rmpv_to_json).collect()),
        rmpv::Value::Map(entries) => {
            let obj: serde_json::Map<String, serde_json::Value> = entries
                .iter()
                .filter_map(|(k, v)| {
                    let key = match k {
                        rmpv::Value::String(s) => s.as_str().map(|s| s.to_string()),
                        _ => Some(format!("{k}")),
                    };
                    key.map(|k| (k, rmpv_to_json(v)))
                })
                .collect();
            serde_json::Value::Object(obj)
        }
        rmpv::Value::Ext(type_id, data) => {
            serde_json::json!({"_ext": {"type": type_id, "data": hex::encode(data)}})
        }
    }
}

/// Format properties as Cypher property string: `name: "Alice", age: 30`
fn format_cypher_props(props: &serde_json::Map<String, serde_json::Value>) -> String {
    props
        .iter()
        .map(|(k, v)| format!("{k}: {v}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Parse a forward adjacency key: `adj:<TYPE>:out:<src_id_be_bytes>`
fn parse_adj_forward_key(key_str: &str) -> Option<(String, u64)> {
    // Format: adj:<edge_type>:out:<8 bytes big-endian u64>
    let stripped = key_str.strip_prefix("adj:")?;
    let out_pos = stripped.find(":out:")?;
    let edge_type = &stripped[..out_pos];
    let id_bytes = &stripped.as_bytes()[out_pos + 5..];
    if id_bytes.len() != 8 {
        return None;
    }
    let source_id = u64::from_be_bytes(id_bytes.try_into().ok()?);
    Some((edge_type.to_string(), source_id))
}

/// Load edge properties for a specific (type, src, tgt) triple.
fn load_edge_properties(
    engine: &StorageEngine,
    snapshot: &StorageSnapshot,
    edge_type: &str,
    source_id: u64,
    target_id: u64,
    interner: &FieldInterner,
) -> Result<serde_json::Map<String, serde_json::Value>, ExportError> {
    // Typed snapshot-aware read instead of hand-rolling the edge-prop key plus
    // a raw snapshot_get plus a manual decode: the EdgeStore decodes through the
    // single canonical edge-property codec and returns EdgeProperties, whose
    // `props` map (`HashMap<u32, Value>`) is exactly the shape
    // `resolve_properties` consumes.
    let store = LocalEdgeStore;
    match store
        .get_props_snapshot(
            engine,
            snapshot,
            edge_type,
            NodeId::from_raw(source_id),
            NodeId::from_raw(target_id),
        )
        .map_err(|e| ExportError::Storage(e.to_string()))?
    {
        Some(props) => Ok(resolve_properties(&props.props, interner)),
        None => Ok(serde_json::Map::new()),
    }
}

// hex module for blob/binary encoding
mod hex {
    pub fn encode(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }
}
