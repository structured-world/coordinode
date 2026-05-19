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
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalDocumentStore, DocumentStore};
    /// # use coordinode_core::graph::{doc_delta::PathTarget, node::NodeId};
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalDocumentStore::new(&engine);
    /// store.set_path(
    ///     0, NodeId::from_raw(1), PathTarget::Extra,
    ///     vec!["a".into(), "b".into()],
    ///     rmpv::Value::String("hello".into()),
    /// )?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn set_path(
        &self,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
        value: rmpv::Value,
    ) -> StoreResult<()>;

    /// Delete a value at a dotted path. Idempotent on missing path.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalDocumentStore, DocumentStore};
    /// # use coordinode_core::graph::{doc_delta::PathTarget, node::NodeId};
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalDocumentStore::new(&engine);
    /// store.delete_path(0, NodeId::from_raw(1), PathTarget::Extra,
    ///                   vec!["a".into(), "b".into()])?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn delete_path(
        &self,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
    ) -> StoreResult<()>;

    /// Append a value to an array at path. Creates the array if
    /// missing.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalDocumentStore, DocumentStore};
    /// # use coordinode_core::graph::{doc_delta::PathTarget, node::NodeId};
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalDocumentStore::new(&engine);
    /// store.array_push(
    ///     0, NodeId::from_raw(1), PathTarget::Extra,
    ///     vec!["tags".into()],
    ///     rmpv::Value::String("rust".into()),
    /// )?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalDocumentStore, DocumentStore};
    /// # use coordinode_core::graph::{doc_delta::PathTarget, node::NodeId};
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalDocumentStore::new(&engine);
    /// store.array_pull(0, NodeId::from_raw(1), PathTarget::Extra,
    ///                  vec!["tags".into()],
    ///                  rmpv::Value::String("rust".into()))?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn array_pull(
        &self,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
        value: rmpv::Value,
    ) -> StoreResult<()>;

    /// Add value to array only if not already present.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalDocumentStore, DocumentStore};
    /// # use coordinode_core::graph::{doc_delta::PathTarget, node::NodeId};
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalDocumentStore::new(&engine);
    /// store.array_add_to_set(0, NodeId::from_raw(1), PathTarget::Extra,
    ///                        vec!["tags".into()],
    ///                        rmpv::Value::String("rust".into()))?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn array_add_to_set(
        &self,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
        value: rmpv::Value,
    ) -> StoreResult<()>;

    /// Numeric increment at path (commutative).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalDocumentStore, DocumentStore};
    /// # use coordinode_core::graph::{doc_delta::PathTarget, node::NodeId};
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalDocumentStore::new(&engine);
    /// store.increment(0, NodeId::from_raw(1), PathTarget::Extra, vec!["v".into()], 1.0)?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalDocumentStore, DocumentStore};
    /// # use coordinode_core::graph::{doc_delta::PathTarget, node::NodeId};
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalDocumentStore::new(&engine);
    /// store.remove_property(0, NodeId::from_raw(1), PathTarget::Extra, Some("name".into()))?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use coordinode_modality::{LocalDocumentStore, DocumentStore};
    /// use coordinode_core::graph::node::NodeId;
    /// use coordinode_core::graph::doc_delta::PathTarget;
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/store"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm,
    /// # )]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// let store = LocalDocumentStore::new(&engine);
    /// store.set_path(
    ///     0,
    ///     NodeId::from_raw(1),
    ///     PathTarget::Extra,
    ///     vec!["profile".into(), "city".into()],
    ///     rmpv::Value::String("Berlin".into()),
    /// )?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
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

    #[test]
    fn delete_path_existing_path_removes_value() {
        let (_dir, engine) = open_engine();
        let nodes = LocalNodeStore::new(&engine);
        let id = NodeId::from_raw(28);
        nodes.put(0, id, &NodeRecord::new("L")).expect("put");

        let docs = LocalDocumentStore::new(&engine);
        docs.set_path(
            0,
            id,
            PathTarget::Extra,
            vec!["a".into(), "b".into()],
            rmpv::Value::String("hello".into()),
        )
        .expect("set");
        docs.delete_path(0, id, PathTarget::Extra, vec!["a".into(), "b".into()])
            .expect("delete");

        let rec = nodes.get(0, id).expect("ok").expect("Some");
        // After delete, "a" map exists but lacks "b" — or the whole
        // "a" entry may be gone if it became empty. Either way, "b"
        // must not be reachable.
        if let Some(Value::Document(rmpv::Value::Map(pairs))) = rec.get_extra("a") {
            let has_b = pairs
                .iter()
                .any(|(k, _)| matches!(k, rmpv::Value::String(s) if s.as_str() == Some("b")));
            assert!(!has_b, "b must be gone after delete_path");
        }
    }

    #[test]
    fn array_pull_existing_value_removes_one() {
        // Push 3 entries, pull one — array has 2 left.
        let (_dir, engine) = open_engine();
        let nodes = LocalNodeStore::new(&engine);
        let id = NodeId::from_raw(29);
        nodes.put(0, id, &NodeRecord::new("L")).expect("put");

        let docs = LocalDocumentStore::new(&engine);
        for s in ["a", "b", "c"] {
            docs.array_push(
                0,
                id,
                PathTarget::Extra,
                vec!["tags".into()],
                rmpv::Value::String(s.into()),
            )
            .expect("push");
        }
        docs.array_pull(
            0,
            id,
            PathTarget::Extra,
            vec!["tags".into()],
            rmpv::Value::String("b".into()),
        )
        .expect("pull");

        let rec = nodes.get(0, id).expect("ok").expect("Some");
        assert_eq!(read_array_len(&rec, "tags"), 2);
    }

    #[test]
    fn remove_property_existing_property_clears_it() {
        let (_dir, engine) = open_engine();
        let nodes = LocalNodeStore::new(&engine);
        let id = NodeId::from_raw(30);
        nodes.put(0, id, &NodeRecord::new("L")).expect("put");

        let docs = LocalDocumentStore::new(&engine);
        docs.set_path(
            0,
            id,
            PathTarget::Extra,
            vec!["temp".into()],
            rmpv::Value::String("v".into()),
        )
        .expect("seed");
        docs.remove_property(0, id, PathTarget::Extra, Some("temp".into()))
            .expect("remove");

        let rec = nodes.get(0, id).expect("ok").expect("Some");
        assert!(
            rec.get_extra("temp").is_none(),
            "temp must be gone after remove_property",
        );
    }

    #[test]
    fn set_path_overwrites_leaf_with_different_type() {
        // Set leaf as String, overwrite with Integer at same path —
        // result reflects last write (DocDelta::SetPath replaces, not
        // merges).
        let (_dir, engine) = open_engine();
        let nodes = LocalNodeStore::new(&engine);
        let id = NodeId::from_raw(31);
        nodes.put(0, id, &NodeRecord::new("L")).expect("put");

        let docs = LocalDocumentStore::new(&engine);
        docs.set_path(
            0,
            id,
            PathTarget::Extra,
            vec!["x".into()],
            rmpv::Value::String("hello".into()),
        )
        .expect("set string");
        docs.set_path(
            0,
            id,
            PathTarget::Extra,
            vec!["x".into()],
            rmpv::Value::Integer(42.into()),
        )
        .expect("set int");

        let rec = nodes.get(0, id).expect("ok").expect("Some");
        let v = rec.get_extra("x").expect("x present");
        match v {
            Value::Document(rmpv::Value::Integer(i)) => {
                assert_eq!(i.as_i64(), Some(42));
            }
            Value::Int(i) => assert_eq!(*i, 42),
            other => panic!("expected integer after overwrite, got {other:?}"),
        }
    }

    #[test]
    fn set_path_on_propfield_target_uses_interned_id() {
        // PathTarget::PropField(field_id) writes into the interned
        // property map instead of `extra`. Visible through the
        // typed accessor on NodeRecord.
        let (_dir, engine) = open_engine();
        let nodes = LocalNodeStore::new(&engine);
        let id = NodeId::from_raw(32);
        let mut base = NodeRecord::new("L");
        base.set_extra("placeholder", Value::Int(0));
        nodes.put(0, id, &base).expect("put");

        let docs = LocalDocumentStore::new(&engine);
        docs.set_path(
            0,
            id,
            PathTarget::PropField(7),
            Vec::new(), // top-level write at field 7
            rmpv::Value::String("via_field".into()),
        )
        .expect("set propfield");

        let rec = nodes.get(0, id).expect("ok").expect("Some");
        let v = rec
            .props
            .get(&7)
            .unwrap_or_else(|| panic!("field 7 must be set; props={:?}", rec.props));
        // Same shape question: rmpv vs raw — accept either.
        match v {
            Value::Document(rmpv::Value::String(s)) => {
                assert_eq!(s.as_str(), Some("via_field"));
            }
            Value::String(s) => assert_eq!(s.as_str(), "via_field"),
            other => panic!("expected string at field 7, got {other:?}"),
        }
    }

    #[test]
    fn concurrent_set_path_distinct_keys_converges() {
        // Four threads each set a distinct key under extra; merge
        // operator collapses all four deltas into a single node body
        // on read.
        use std::sync::Arc;
        use std::thread;

        let (_dir, engine) = open_engine();
        let engine = Arc::new(engine);
        let id = NodeId::from_raw(100);
        let nodes = LocalNodeStore::new(&engine);
        nodes.put(0, id, &NodeRecord::new("Multi")).expect("seed");

        let handles: Vec<_> = (0..4u64)
            .map(|t| {
                let engine = Arc::clone(&engine);
                thread::spawn(move || {
                    let docs = LocalDocumentStore::new(&engine);
                    docs.set_path(
                        0,
                        id,
                        PathTarget::Extra,
                        vec![format!("k{t}")],
                        rmpv::Value::Integer((t as i64).into()),
                    )
                    .expect("set_path");
                })
            })
            .collect();
        for h in handles {
            h.join().expect("join");
        }

        let rec = nodes.get(0, id).expect("ok").expect("Some");
        for t in 0u64..4 {
            assert!(
                rec.get_extra(&format!("k{t}")).is_some(),
                "key k{t} missing after concurrent set_path",
            );
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
