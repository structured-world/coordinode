//! Document store — path-targeted partial updates on DOCUMENT
//! properties of [`NodeRecord`] via the [`DocumentMerge`] operator
//! (ADR-015).
//!
//! Document-typed properties (Mongo-like nested maps + arrays) are
//! mutated through commutative [`DocDelta`] operands rather than
//! read-modify-write. Each [`DocDelta`] is encoded with the
//! `PREFIX_DOC_DELTA` byte and buffered on the transaction; the
//! merge function replays operands in seqno order against the base
//! `NodeRecord` during reads and compaction.
//!
//! ## Transaction threading (ADR-041)
//!
//! Every method takes an explicit `&mut Transaction`. The encoded
//! [`DocDelta`] operand is buffered via
//! [`Transaction::push_node_delta`] and applied at
//! [`Transaction::commit`] through the node-partition merge path, so a
//! set of document mutations lands atomically (or not at all). Reads in
//! the same transaction surface the pending operands via
//! [`NodeStore::get_raw_tracked`](crate::NodeStore::get_raw_tracked),
//! which materialises buffered deltas before the OCC-tracked read.
//!
//! ## What this store exposes
//!
//! Every method is a typed wrapper around
//! [`Transaction::push_node_delta`] that hides:
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
use coordinode_storage::engine::transaction::Transaction;

use crate::error::{StoreError, StoreResult};

/// Layer 4 document store: typed buffering of [`DocDelta`] operands
/// against DOCUMENT-typed properties of a node. Every method buffers a
/// merge operand on the supplied [`Transaction`]; the operands apply
/// atomically at [`Transaction::commit`].
pub trait DocumentStore {
    /// Set a value at a dotted path on the node's document property.
    /// Intermediate maps are created as needed.
    fn set_path(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
        value: rmpv::Value,
    ) -> StoreResult<()>;

    /// Delete a value at a dotted path. Idempotent on missing path.
    fn delete_path(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
    ) -> StoreResult<()>;

    /// Append a value to an array at path. Creates the array if
    /// missing.
    fn array_push(
        &self,
        txn: &mut Transaction,
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
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
        value: rmpv::Value,
    ) -> StoreResult<()>;

    /// Add value to array only if not already present.
    fn array_add_to_set(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
        value: rmpv::Value,
    ) -> StoreResult<()>;

    /// Numeric increment at path (commutative).
    fn increment(
        &self,
        txn: &mut Transaction,
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
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        key: Option<String>,
    ) -> StoreResult<()>;
}

/// CE single-shard implementation of [`DocumentStore`]. Stateless — all
/// storage access flows through the [`Transaction`] passed to each
/// method (ADR-041).
pub struct LocalDocumentStore;

impl LocalDocumentStore {
    /// Encode `delta` and buffer it as a node merge operand on `txn`.
    /// The operand carries the `PREFIX_DOC_DELTA` framing byte and is
    /// applied at commit via the node-partition merge path.
    fn write_delta(
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        delta: DocDelta,
    ) -> StoreResult<()> {
        let operand = delta.encode().map_err(|e| StoreError::Decode {
            kind: "doc delta",
            message: format!("encode: {e}"),
        })?;
        txn.push_node_delta(encode_node_key(shard_id, node_id), operand);
        Ok(())
    }
}

impl DocumentStore for LocalDocumentStore {
    fn set_path(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
        value: rmpv::Value,
    ) -> StoreResult<()> {
        Self::write_delta(
            txn,
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
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
    ) -> StoreResult<()> {
        Self::write_delta(
            txn,
            shard_id,
            node_id,
            DocDelta::DeletePath { target, path },
        )
    }

    fn array_push(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
        value: rmpv::Value,
    ) -> StoreResult<()> {
        Self::write_delta(
            txn,
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
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
        value: rmpv::Value,
    ) -> StoreResult<()> {
        Self::write_delta(
            txn,
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
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
        value: rmpv::Value,
    ) -> StoreResult<()> {
        Self::write_delta(
            txn,
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
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
        amount: f64,
    ) -> StoreResult<()> {
        Self::write_delta(
            txn,
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
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        key: Option<String>,
    ) -> StoreResult<()> {
        Self::write_delta(
            txn,
            shard_id,
            node_id,
            DocDelta::RemoveProperty { target, key },
        )
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::node::{LocalNodeStore, NodeStore};
    use coordinode_core::graph::node::NodeRecord;
    use coordinode_core::graph::types::Value;
    use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
    use coordinode_core::txn::write_concern::WriteConcern;
    use coordinode_storage::engine::core::StorageEngine;
    use coordinode_storage::engine::transaction::{CommitContext, Transaction};

    /// Logic-test fixture (memory backing, env-flippable). Document
    /// property tests verify dot-notation + merge-op contracts.
    fn open_engine() -> coordinode_test_fixtures::EngineFixture {
        coordinode_test_fixtures::engine_for_logic()
    }

    /// Open an MVCC transaction (shard 0), commit it, returning the
    /// committed outcome. Shared spine for `put_node` and `commit_docs`.
    fn run_committed(engine: &StorageEngine, body: impl FnOnce(&mut Transaction)) {
        let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
        let read_ts = oracle.next();
        let mut txn = Transaction::new(engine, Some(&oracle), read_ts, Some(engine.snapshot()));
        body(&mut txn);
        let wc = WriteConcern::majority();
        let ctx = CommitContext {
            write_concern: &wc,
            pipeline: None,
            id_gen: None,
            drain_buffer: None,
            nvme_write_buffer: None,
        };
        txn.commit(&ctx).expect("commit txn");
    }

    /// Seed a base node via an MVCC transaction (shard 0) + commit, so a
    /// subsequent document merge has a record to merge into.
    fn put_node(engine: &StorageEngine, id: NodeId, record: &NodeRecord) {
        run_committed(engine, |txn| {
            LocalNodeStore.put(txn, 0, id, record).expect("put node");
        });
    }

    /// Run document-store ops inside one MVCC transaction (shard 0) and
    /// commit, so the buffered merge operands land for a subsequent
    /// read. The closure receives a stateless [`LocalDocumentStore`]
    /// and the open transaction.
    fn commit_docs(
        engine: &StorageEngine,
        body: impl FnOnce(&LocalDocumentStore, &mut Transaction),
    ) {
        run_committed(engine, |txn| body(&LocalDocumentStore, txn));
    }

    /// Read a node (shard 0) at the latest committed snapshot through an
    /// MVCC transaction.
    fn get_node(engine: &StorageEngine, id: NodeId) -> Option<NodeRecord> {
        let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
        let read_ts = oracle.next();
        let txn = Transaction::new(engine, Some(&oracle), read_ts, Some(engine.snapshot()));
        LocalNodeStore.get(&txn, 0, id).expect("get node")
    }

    /// Round-trip a SetPath against the Extra map and verify the
    /// merge operator surfaces it on the next read through NodeStore.
    #[test]
    fn set_path_on_extra_visible_after_node_get() {
        let fx = open_engine();
        let engine = &fx.engine;
        // First seed a base node so the merge has something to merge
        // into.
        let id = NodeId::from_raw(1);
        put_node(engine, id, &NodeRecord::new("User"));

        commit_docs(engine, |docs, txn| {
            docs.set_path(
                txn,
                0,
                id,
                PathTarget::Extra,
                vec!["profile".to_string(), "city".to_string()],
                rmpv::Value::String("Berlin".into()),
            )
            .expect("set_path");
        });

        // Read back via NodeStore — the merge operator collapses the
        // delta history transparently. The nested Document value at
        // extra.profile is an rmpv::Value::Map; we walk it to find
        // the city key without depending on a specific Value variant
        // ordering inside the rmpv map.
        let rec = get_node(engine, id).expect("Some");
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
        let fx = open_engine();
        let engine = &fx.engine;
        let id = NodeId::from_raw(7);
        put_node(engine, id, &NodeRecord::new("Counter"));

        commit_docs(engine, |docs, txn| {
            for _ in 0..3 {
                docs.increment(txn, 0, id, PathTarget::Extra, vec!["hits".to_string()], 1.0)
                    .expect("inc");
            }
        });

        let rec = get_node(engine, id).expect("Some");
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
        let fx = open_engine();
        let engine = &fx.engine;
        let id = NodeId::from_raw(99);
        put_node(engine, id, &NodeRecord::new("X"));

        // Path doesn't exist — delete must succeed silently.
        commit_docs(engine, |docs, txn| {
            docs.delete_path(
                txn,
                0,
                id,
                PathTarget::Extra,
                vec!["never".to_string(), "existed".to_string()],
            )
            .expect("delete missing");
        });

        // Node still readable, no change.
        get_node(engine, id).expect("Some");
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
        let fx = open_engine();
        let engine = &fx.engine;
        let id = NodeId::from_raw(20);
        put_node(engine, id, &NodeRecord::new("L"));

        commit_docs(engine, |docs, txn| {
            docs.array_push(
                txn,
                0,
                id,
                PathTarget::Extra,
                vec!["scores".to_string()],
                rmpv::Value::F64(42.0),
            )
            .expect("push");
        });

        let rec = get_node(engine, id).expect("Some");
        assert_eq!(read_array_len(&rec, "scores"), 1);
    }

    #[test]
    fn array_push_appends_in_order() {
        let fx = open_engine();
        let engine = &fx.engine;
        let id = NodeId::from_raw(21);
        put_node(engine, id, &NodeRecord::new("L"));

        commit_docs(engine, |docs, txn| {
            for v in [1.0, 2.0, 3.0] {
                docs.array_push(
                    txn,
                    0,
                    id,
                    PathTarget::Extra,
                    vec!["xs".to_string()],
                    rmpv::Value::F64(v),
                )
                .expect("push");
            }
        });
        let rec = get_node(engine, id).expect("Some");
        assert_eq!(read_array_len(&rec, "xs"), 3);
    }

    #[test]
    fn array_pull_missing_value_is_noop() {
        // Seed an array, pull a value not in it — array unchanged.
        let fx = open_engine();
        let engine = &fx.engine;
        let id = NodeId::from_raw(22);
        put_node(engine, id, &NodeRecord::new("L"));

        commit_docs(engine, |docs, txn| {
            docs.array_push(
                txn,
                0,
                id,
                PathTarget::Extra,
                vec!["tags".to_string()],
                rmpv::Value::String("a".into()),
            )
            .expect("seed");
            docs.array_pull(
                txn,
                0,
                id,
                PathTarget::Extra,
                vec!["tags".to_string()],
                rmpv::Value::String("not-there".into()),
            )
            .expect("pull");
        });
        let rec = get_node(engine, id).expect("Some");
        assert_eq!(read_array_len(&rec, "tags"), 1);
    }

    #[test]
    fn array_pull_missing_path_is_noop() {
        let fx = open_engine();
        let engine = &fx.engine;
        let id = NodeId::from_raw(23);
        put_node(engine, id, &NodeRecord::new("L"));

        commit_docs(engine, |docs, txn| {
            docs.array_pull(
                txn,
                0,
                id,
                PathTarget::Extra,
                vec!["never".to_string()],
                rmpv::Value::String("x".into()),
            )
            .expect("pull missing");
        });
        // Node still readable.
        get_node(engine, id).expect("Some");
    }

    #[test]
    fn increment_accepts_negative_delta() {
        let fx = open_engine();
        let engine = &fx.engine;
        let id = NodeId::from_raw(24);
        put_node(engine, id, &NodeRecord::new("L"));

        commit_docs(engine, |docs, txn| {
            docs.increment(txn, 0, id, PathTarget::Extra, vec!["v".to_string()], 10.0)
                .expect("inc up");
            docs.increment(txn, 0, id, PathTarget::Extra, vec!["v".to_string()], -3.5)
                .expect("inc down");
        });
        let rec = get_node(engine, id).expect("Some");
        assert!((read_numeric(&rec, "v") - 6.5).abs() < 1e-9);
    }

    #[test]
    fn increment_creates_value_if_missing() {
        let fx = open_engine();
        let engine = &fx.engine;
        let id = NodeId::from_raw(25);
        put_node(engine, id, &NodeRecord::new("L"));

        // No prior value — increment from scratch lands at delta.
        commit_docs(engine, |docs, txn| {
            docs.increment(
                txn,
                0,
                id,
                PathTarget::Extra,
                vec!["fresh".to_string()],
                7.0,
            )
            .expect("inc");
        });
        let rec = get_node(engine, id).expect("Some");
        assert!((read_numeric(&rec, "fresh") - 7.0).abs() < 1e-9);
    }

    #[test]
    fn remove_property_missing_is_idempotent() {
        let fx = open_engine();
        let engine = &fx.engine;
        let id = NodeId::from_raw(26);
        put_node(engine, id, &NodeRecord::new("L"));

        commit_docs(engine, |docs, txn| {
            docs.remove_property(txn, 0, id, PathTarget::Extra, Some("never".to_string()))
                .expect("remove missing");
        });
        get_node(engine, id).expect("Some");
    }

    #[test]
    fn set_path_deep_nesting_creates_intermediates() {
        // Four-level nested path on an empty Extra — every
        // intermediate map must be created so the leaf set lands.
        let fx = open_engine();
        let engine = &fx.engine;
        let id = NodeId::from_raw(27);
        put_node(engine, id, &NodeRecord::new("L"));

        commit_docs(engine, |docs, txn| {
            docs.set_path(
                txn,
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
        });

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

        let rec = get_node(engine, id).expect("Some");
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
        let fx = open_engine();
        let engine = &fx.engine;
        let id = NodeId::from_raw(28);
        put_node(engine, id, &NodeRecord::new("L"));

        commit_docs(engine, |docs, txn| {
            docs.set_path(
                txn,
                0,
                id,
                PathTarget::Extra,
                vec!["a".into(), "b".into()],
                rmpv::Value::String("hello".into()),
            )
            .expect("set");
            docs.delete_path(txn, 0, id, PathTarget::Extra, vec!["a".into(), "b".into()])
                .expect("delete");
        });

        let rec = get_node(engine, id).expect("Some");
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
        let fx = open_engine();
        let engine = &fx.engine;
        let id = NodeId::from_raw(29);
        put_node(engine, id, &NodeRecord::new("L"));

        commit_docs(engine, |docs, txn| {
            for s in ["a", "b", "c"] {
                docs.array_push(
                    txn,
                    0,
                    id,
                    PathTarget::Extra,
                    vec!["tags".into()],
                    rmpv::Value::String(s.into()),
                )
                .expect("push");
            }
            docs.array_pull(
                txn,
                0,
                id,
                PathTarget::Extra,
                vec!["tags".into()],
                rmpv::Value::String("b".into()),
            )
            .expect("pull");
        });

        let rec = get_node(engine, id).expect("Some");
        assert_eq!(read_array_len(&rec, "tags"), 2);
    }

    #[test]
    fn remove_property_existing_property_clears_it() {
        let fx = open_engine();
        let engine = &fx.engine;
        let id = NodeId::from_raw(30);
        put_node(engine, id, &NodeRecord::new("L"));

        commit_docs(engine, |docs, txn| {
            docs.set_path(
                txn,
                0,
                id,
                PathTarget::Extra,
                vec!["temp".into()],
                rmpv::Value::String("v".into()),
            )
            .expect("seed");
            docs.remove_property(txn, 0, id, PathTarget::Extra, Some("temp".into()))
                .expect("remove");
        });

        let rec = get_node(engine, id).expect("Some");
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
        let fx = open_engine();
        let engine = &fx.engine;
        let id = NodeId::from_raw(31);
        put_node(engine, id, &NodeRecord::new("L"));

        commit_docs(engine, |docs, txn| {
            docs.set_path(
                txn,
                0,
                id,
                PathTarget::Extra,
                vec!["x".into()],
                rmpv::Value::String("hello".into()),
            )
            .expect("set string");
            docs.set_path(
                txn,
                0,
                id,
                PathTarget::Extra,
                vec!["x".into()],
                rmpv::Value::Integer(42.into()),
            )
            .expect("set int");
        });

        let rec = get_node(engine, id).expect("Some");
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
        let fx = open_engine();
        let engine = &fx.engine;
        let id = NodeId::from_raw(32);
        let mut base = NodeRecord::new("L");
        base.set_extra("placeholder", Value::Int(0));
        put_node(engine, id, &base);

        commit_docs(engine, |docs, txn| {
            docs.set_path(
                txn,
                0,
                id,
                PathTarget::PropField(7),
                Vec::new(), // top-level write at field 7
                rmpv::Value::String("via_field".into()),
            )
            .expect("set propfield");
        });

        let rec = get_node(engine, id).expect("Some");
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
        // Four threads each set a distinct key under extra in its own
        // transaction; the merge operator collapses all four deltas
        // into a single node body on read.
        use std::sync::Arc;
        use std::thread;

        let fx = open_engine();
        let engine = Arc::clone(&fx.engine);
        let id = NodeId::from_raw(100);
        put_node(&engine, id, &NodeRecord::new("Multi"));

        let handles: Vec<_> = (0..4u64)
            .map(|t| {
                let engine = Arc::clone(&engine);
                thread::spawn(move || {
                    commit_docs(&engine, |docs, txn| {
                        docs.set_path(
                            txn,
                            0,
                            id,
                            PathTarget::Extra,
                            vec![format!("k{t}")],
                            rmpv::Value::Integer((t as i64).into()),
                        )
                        .expect("set_path");
                    });
                })
            })
            .collect();
        for h in handles {
            h.join().expect("join");
        }

        let rec = get_node(&engine, id).expect("Some");
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
        let fx = open_engine();
        let engine = &fx.engine;
        let id = NodeId::from_raw(5);
        put_node(engine, id, &NodeRecord::new("Y"));

        commit_docs(engine, |docs, txn| {
            for _ in 0..3 {
                docs.array_add_to_set(
                    txn,
                    0,
                    id,
                    PathTarget::Extra,
                    vec!["tags".to_string()],
                    rmpv::Value::String("rust".into()),
                )
                .expect("add");
            }
        });

        let rec = get_node(engine, id).expect("Some");
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
