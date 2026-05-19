//! Document store — path-targeted partial updates on DOCUMENT
//! properties of [`NodeRecord`] via the [`DocumentMerge`] operator
//! (ADR-015).
//!
//! Document-typed properties (Mongo-like nested maps + arrays) are
//! mutated through commutative [`DocDelta`] operands rather than
//! read-modify-write. Each [`DocDelta`] is encoded with the
//! `PREFIX_DOC_DELTA` byte and written via the engine's
//! `merge(Partition::Node, …)` path; the merge function replays
//! operands in seqno order against the base `NodeRecord` during
//! reads and compaction.
//!
//! ## What this store exposes
//!
//! Every method is a typed wrapper around `engine.merge(...)` that
//! hides:
//!
//! - Operand framing (the `PREFIX_DOC_DELTA` prefix byte).
//! - MessagePack encoding of the [`DocDelta`] enum.
//! - The node-key shape (`encode_node_key(shard, id)`).
//!
//! The caller chooses [`PathTarget`] (Extra map vs. interned
//! PropField) and supplies the path + value. The store does not
//! materialise the post-merge `NodeRecord` for inspection — for that,
//! callers use [`NodeStore::get`](crate::NodeStore::get), which
//! triggers the merge transparently.
//!
//! ## Read side
//!
//! Reading a node that has accumulated [`DocDelta`] operands goes
//! through the existing [`NodeStore`](crate::NodeStore) — the merge
//! operator collapses the delta history during `get`. This store is
//! write-only on purpose: no read API exists at the doc level, only
//! at the node level. That matches how the merge operator works (one
//! materialisation point per read).
//!
//! [`DocumentMerge`]: coordinode_storage::engine::merge::DocumentMerge
//! [`NodeRecord`]: coordinode_core::graph::node::NodeRecord

use coordinode_core::graph::doc_delta::{DocDelta, PathTarget};
use coordinode_core::graph::node::{encode_node_key, NodeId};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

use crate::error::{StoreError, StoreResult};

/// Layer 4 document store: typed write of [`DocDelta`] operands
/// against DOCUMENT-typed properties of a node.
pub trait DocumentStore {
    /// Set a value at a dotted path on the node's document property.
    /// Intermediate maps are created as needed.
    fn set_path(
        &self,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
        value: rmpv::Value,
    ) -> StoreResult<()>;

    /// Delete a value at a dotted path. Idempotent on missing path.
    fn delete_path(
        &self,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
    ) -> StoreResult<()>;

    /// Append a value to an array at path. Creates the array if
    /// missing.
    fn array_push(
        &self,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
        value: rmpv::Value,
    ) -> StoreResult<()>;

    /// Remove the first occurrence of a value from an array at path.
    /// Idempotent.
    fn array_pull(
        &self,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
        value: rmpv::Value,
    ) -> StoreResult<()>;

    /// Add value to array only if not already present.
    fn array_add_to_set(
        &self,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
        value: rmpv::Value,
    ) -> StoreResult<()>;

    /// Numeric increment at path (commutative).
    fn increment(
        &self,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
        amount: f64,
    ) -> StoreResult<()>;

    /// Remove a top-level property from the node's record. For
    /// `PathTarget::Extra` the caller supplies the key; for
    /// `PathTarget::PropField(_)` the field id in the target is used.
    fn remove_property(
        &self,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        key: Option<String>,
    ) -> StoreResult<()>;
}

/// CE single-shard implementation of [`DocumentStore`].
pub struct LocalDocumentStore<'a> {
    engine: &'a StorageEngine,
}

impl<'a> LocalDocumentStore<'a> {
    /// Wrap a storage engine for document-store operations.
    pub fn new(engine: &'a StorageEngine) -> Self {
        Self { engine }
    }

    fn write_delta(&self, shard_id: u16, node_id: NodeId, delta: DocDelta) -> StoreResult<()> {
        let operand = delta.encode().map_err(|e| StoreError::Decode {
            kind: "doc delta",
            message: format!("encode: {e}"),
        })?;
        self.engine.merge(
            Partition::Node,
            &encode_node_key(shard_id, node_id),
            &operand,
        )?;
        Ok(())
    }
}

impl DocumentStore for LocalDocumentStore<'_> {
    fn set_path(
        &self,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
        value: rmpv::Value,
    ) -> StoreResult<()> {
        self.write_delta(
            shard_id,
            node_id,
            DocDelta::SetPath {
                target,
                path,
                value,
            },
        )
    }

    fn delete_path(
        &self,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
    ) -> StoreResult<()> {
        self.write_delta(shard_id, node_id, DocDelta::DeletePath { target, path })
    }

    fn array_push(
        &self,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
        value: rmpv::Value,
    ) -> StoreResult<()> {
        self.write_delta(
            shard_id,
            node_id,
            DocDelta::ArrayPush {
                target,
                path,
                value,
            },
        )
    }

    fn array_pull(
        &self,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
        value: rmpv::Value,
    ) -> StoreResult<()> {
        self.write_delta(
            shard_id,
            node_id,
            DocDelta::ArrayPull {
                target,
                path,
                value,
            },
        )
    }

    fn array_add_to_set(
        &self,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
        value: rmpv::Value,
    ) -> StoreResult<()> {
        self.write_delta(
            shard_id,
            node_id,
            DocDelta::ArrayAddToSet {
                target,
                path,
                value,
            },
        )
    }

    fn increment(
        &self,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
        amount: f64,
    ) -> StoreResult<()> {
        self.write_delta(
            shard_id,
            node_id,
            DocDelta::Increment {
                target,
                path,
                amount,
            },
        )
    }

    fn remove_property(
        &self,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        key: Option<String>,
    ) -> StoreResult<()> {
        self.write_delta(shard_id, node_id, DocDelta::RemoveProperty { target, key })
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::node::{LocalNodeStore, NodeStore};
    use coordinode_core::graph::node::NodeRecord;
    use coordinode_core::graph::types::Value;
    use coordinode_storage::engine::config::{
        Durability, EndpointConfig, Media, StorageConfig, Tier,
    };
    use tempfile::TempDir;

    fn open_engine() -> (TempDir, StorageEngine) {
        let dir = TempDir::new().expect("tempdir");
        let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "ep",
            dir.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        let engine = StorageEngine::open(&config).expect("open");
        (dir, engine)
    }

    /// Round-trip a SetPath against the Extra map and verify the
    /// merge operator surfaces it on the next read through NodeStore.
    #[test]
    fn set_path_on_extra_visible_after_node_get() {
        let (_dir, engine) = open_engine();
        // First seed a base node so the merge has something to merge
        // into.
        let nodes = LocalNodeStore::new(&engine);
        let id = NodeId::from_raw(1);
        nodes.put(0, id, &NodeRecord::new("User")).expect("put");

        let docs = LocalDocumentStore::new(&engine);
        docs.set_path(
            0,
            id,
            PathTarget::Extra,
            vec!["profile".to_string(), "city".to_string()],
            rmpv::Value::String("Berlin".into()),
        )
        .expect("set_path");

        // Read back via NodeStore — the merge operator collapses the
        // delta history transparently. The nested Document value at
        // extra.profile is an rmpv::Value::Map; we walk it to find
        // the city key without depending on a specific Value variant
        // ordering inside the rmpv map.
        let rec = nodes.get(0, id).expect("ok").expect("Some");
        let extra = rec
            .get_extra("profile")
            .expect("profile key present in extra after merge");
        let Value::Document(rmpv_map) = extra else {
            panic!("expected Value::Document at extra.profile, got {extra:?}");
        };
        let rmpv::Value::Map(pairs) = rmpv_map else {
            panic!("expected rmpv::Value::Map, got {rmpv_map:?}");
        };
        let city_value = pairs
            .iter()
            .find_map(|(k, v)| match k {
                rmpv::Value::String(s) if s.as_str() == Some("city") => Some(v),
                _ => None,
            })
            .expect("city key present");
        match city_value {
            rmpv::Value::String(s) => assert_eq!(s.as_str(), Some("Berlin")),
            other => panic!("expected String, got {other:?}"),
        }
    }

    /// Increment commutativity — apply +1 three times, expect +3.
    #[test]
    fn increment_accumulates_over_multiple_deltas() {
        let (_dir, engine) = open_engine();
        let nodes = LocalNodeStore::new(&engine);
        let id = NodeId::from_raw(7);
        nodes.put(0, id, &NodeRecord::new("Counter")).expect("put");

        let docs = LocalDocumentStore::new(&engine);
        for _ in 0..3 {
            docs.increment(0, id, PathTarget::Extra, vec!["hits".to_string()], 1.0)
                .expect("inc");
        }

        let rec = nodes.get(0, id).expect("ok").expect("Some");
        let hits = rec.get_extra("hits").expect("hits present");
        // Extra entries flow through rmpv at the merge boundary, so
        // numeric increments land as Value::Document(F64) rather
        // than Value::Float — both shapes mean the same number.
        // After 3× +1 accumulations the value lands as an integer
        // through the path's interim representation. Accept both
        // numeric variants — the contract is "+1 applied three
        // times produces a value representing 3".
        let v = match hits {
            Value::Document(rmpv::Value::F64(f)) => *f,
            Value::Document(rmpv::Value::Integer(i)) => {
                i.as_f64().expect("integer convertible to f64")
            }
            Value::Float(f) => *f,
            Value::Int(i) => *i as f64,
            other => panic!("expected numeric, got {other:?}"),
        };
        assert!(
            (v - 3.0).abs() < f64::EPSILON,
            "increment must accumulate, got {v}",
        );
    }

    /// DeletePath on a missing path is idempotent — no error, no
    /// observable state change.
    #[test]
    fn delete_path_missing_is_idempotent() {
        let (_dir, engine) = open_engine();
        let nodes = LocalNodeStore::new(&engine);
        let id = NodeId::from_raw(99);
        nodes.put(0, id, &NodeRecord::new("X")).expect("put");

        let docs = LocalDocumentStore::new(&engine);
        // Path doesn't exist — delete must succeed silently.
        docs.delete_path(
            0,
            id,
            PathTarget::Extra,
            vec!["never".to_string(), "existed".to_string()],
        )
        .expect("delete missing");

        // Node still readable, no change.
        nodes.get(0, id).expect("ok").expect("Some");
    }

    fn read_array_len(rec: &coordinode_core::graph::node::NodeRecord, key: &str) -> usize {
        let v = rec.get_extra(key).expect("key present");
        match v {
            Value::Document(rmpv::Value::Array(arr)) => arr.len(),
            Value::Array(arr) => arr.len(),
            other => panic!("expected array, got {other:?}"),
        }
    }

    fn read_numeric(rec: &coordinode_core::graph::node::NodeRecord, key: &str) -> f64 {
        let v = rec.get_extra(key).expect("key present");
        match v {
            Value::Document(rmpv::Value::F64(f)) => *f,
            Value::Document(rmpv::Value::Integer(i)) => {
                i.as_f64().expect("integer convertible to f64")
            }
            Value::Float(f) => *f,
            Value::Int(i) => *i as f64,
            other => panic!("expected numeric, got {other:?}"),
        }
    }

    #[test]
    fn array_push_creates_array_if_missing() {
        let (_dir, engine) = open_engine();
        let nodes = LocalNodeStore::new(&engine);
        let id = NodeId::from_raw(20);
        nodes.put(0, id, &NodeRecord::new("L")).expect("put");

        let docs = LocalDocumentStore::new(&engine);
        docs.array_push(
            0,
            id,
            PathTarget::Extra,
            vec!["scores".to_string()],
            rmpv::Value::F64(42.0),
        )
        .expect("push");

        let rec = nodes.get(0, id).expect("ok").expect("Some");
        assert_eq!(read_array_len(&rec, "scores"), 1);
    }

    #[test]
    fn array_push_appends_in_order() {
        let (_dir, engine) = open_engine();
        let nodes = LocalNodeStore::new(&engine);
        let id = NodeId::from_raw(21);
        nodes.put(0, id, &NodeRecord::new("L")).expect("put");

        let docs = LocalDocumentStore::new(&engine);
        for v in [1.0, 2.0, 3.0] {
            docs.array_push(
                0,
                id,
                PathTarget::Extra,
                vec!["xs".to_string()],
                rmpv::Value::F64(v),
            )
            .expect("push");
        }
        let rec = nodes.get(0, id).expect("ok").expect("Some");
        assert_eq!(read_array_len(&rec, "xs"), 3);
    }

    #[test]
    fn array_pull_missing_value_is_noop() {
        // Seed an array, pull a value not in it — array unchanged.
        let (_dir, engine) = open_engine();
        let nodes = LocalNodeStore::new(&engine);
        let id = NodeId::from_raw(22);
        nodes.put(0, id, &NodeRecord::new("L")).expect("put");

        let docs = LocalDocumentStore::new(&engine);
        docs.array_push(
            0,
            id,
            PathTarget::Extra,
            vec!["tags".to_string()],
            rmpv::Value::String("a".into()),
        )
        .expect("seed");
        docs.array_pull(
            0,
            id,
            PathTarget::Extra,
            vec!["tags".to_string()],
            rmpv::Value::String("not-there".into()),
        )
        .expect("pull");
        let rec = nodes.get(0, id).expect("ok").expect("Some");
        assert_eq!(read_array_len(&rec, "tags"), 1);
    }

    #[test]
    fn array_pull_missing_path_is_noop() {
        let (_dir, engine) = open_engine();
        let nodes = LocalNodeStore::new(&engine);
        let id = NodeId::from_raw(23);
        nodes.put(0, id, &NodeRecord::new("L")).expect("put");

        let docs = LocalDocumentStore::new(&engine);
        docs.array_pull(
            0,
            id,
            PathTarget::Extra,
            vec!["never".to_string()],
            rmpv::Value::String("x".into()),
        )
        .expect("pull missing");
        // Node still readable.
        nodes.get(0, id).expect("ok").expect("Some");
    }

    #[test]
    fn increment_accepts_negative_delta() {
        let (_dir, engine) = open_engine();
        let nodes = LocalNodeStore::new(&engine);
        let id = NodeId::from_raw(24);
        nodes.put(0, id, &NodeRecord::new("L")).expect("put");

        let docs = LocalDocumentStore::new(&engine);
        docs.increment(0, id, PathTarget::Extra, vec!["v".to_string()], 10.0)
            .expect("inc up");
        docs.increment(0, id, PathTarget::Extra, vec!["v".to_string()], -3.5)
            .expect("inc down");
        let rec = nodes.get(0, id).expect("ok").expect("Some");
        assert!((read_numeric(&rec, "v") - 6.5).abs() < 1e-9);
    }

    #[test]
    fn increment_creates_value_if_missing() {
        let (_dir, engine) = open_engine();
        let nodes = LocalNodeStore::new(&engine);
        let id = NodeId::from_raw(25);
        nodes.put(0, id, &NodeRecord::new("L")).expect("put");

        let docs = LocalDocumentStore::new(&engine);
        // No prior value — increment from scratch lands at delta.
        docs.increment(0, id, PathTarget::Extra, vec!["fresh".to_string()], 7.0)
            .expect("inc");
        let rec = nodes.get(0, id).expect("ok").expect("Some");
        assert!((read_numeric(&rec, "fresh") - 7.0).abs() < 1e-9);
    }

    #[test]
    fn remove_property_missing_is_idempotent() {
        let (_dir, engine) = open_engine();
        let nodes = LocalNodeStore::new(&engine);
        let id = NodeId::from_raw(26);
        nodes.put(0, id, &NodeRecord::new("L")).expect("put");

        let docs = LocalDocumentStore::new(&engine);
        docs.remove_property(0, id, PathTarget::Extra, Some("never".to_string()))
            .expect("remove missing");
        nodes.get(0, id).expect("ok").expect("Some");
    }

    #[test]
    fn set_path_deep_nesting_creates_intermediates() {
        // Four-level nested path on an empty Extra — every
        // intermediate map must be created so the leaf set lands.
        let (_dir, engine) = open_engine();
        let nodes = LocalNodeStore::new(&engine);
        let id = NodeId::from_raw(27);
        nodes.put(0, id, &NodeRecord::new("L")).expect("put");

        let docs = LocalDocumentStore::new(&engine);
        docs.set_path(
            0,
            id,
            PathTarget::Extra,
            vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
            ],
            rmpv::Value::String("deep".into()),
        )
        .expect("set deep");

        fn descend<'a>(map: &'a rmpv::Value, key: &str) -> &'a rmpv::Value {
            let rmpv::Value::Map(pairs) = map else {
                panic!("expected Map descending at {key}, got {map:?}");
            };
            pairs
                .iter()
                .find_map(|(k, v)| match k {
                    rmpv::Value::String(s) if s.as_str() == Some(key) => Some(v),
                    _ => None,
                })
                .unwrap_or_else(|| panic!("key {key} missing"))
        }

        let rec = nodes.get(0, id).expect("ok").expect("Some");
        let a = rec.get_extra("a").expect("a present");
        let a_inner = match a {
            Value::Document(m) => m,
            other => panic!("expected Document at a, got {other:?}"),
        };
        let b = descend(a_inner, "b");
        let c = descend(b, "c");
        let d = descend(c, "d");
        match d {
            rmpv::Value::String(s) => assert_eq!(s.as_str(), Some("deep")),
            other => panic!("expected String at d, got {other:?}"),
        }
    }

    /// ArrayAddToSet dedups — adding the same value twice yields a
    /// single-element array.
    #[test]
    fn array_add_to_set_deduplicates() {
        let (_dir, engine) = open_engine();
        let nodes = LocalNodeStore::new(&engine);
        let id = NodeId::from_raw(5);
        nodes.put(0, id, &NodeRecord::new("Y")).expect("put");

        let docs = LocalDocumentStore::new(&engine);
        for _ in 0..3 {
            docs.array_add_to_set(
                0,
                id,
                PathTarget::Extra,
                vec!["tags".to_string()],
                rmpv::Value::String("rust".into()),
            )
            .expect("add");
        }

        let rec = nodes.get(0, id).expect("ok").expect("Some");
        let tags = rec.get_extra("tags").expect("tags present");
        // Same rmpv shape: arrays in `extra` come back wrapped in
        // Value::Document(rmpv::Value::Array(...)).
        let len = match tags {
            Value::Document(rmpv::Value::Array(arr)) => arr.len(),
            Value::Array(arr) => arr.len(),
            other => panic!("expected array, got {other:?}"),
        };
        assert_eq!(len, 1, "AddToSet must dedup, got len={len}");
    }
}
