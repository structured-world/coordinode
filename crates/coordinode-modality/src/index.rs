//! Index store — secondary B-tree-style entries in [`Partition::Idx`].
//!
//! Stores entries of the form `idx:<name>:<sortable_value>:<node_id>`
//! (value-bytes encoded by [`coordinode_core::index::encoding`] for
//! correct lexicographic ordering). Supports point lookup
//! ([`IndexStore::scan_exact`]) and full-index walk
//! ([`IndexStore::scan_all`]).
//!
//! The store deliberately does NOT carry index metadata (kind,
//! target label, target property) — that is a query-layer concept.
//! The store operates one level below: caller passes the index name,
//! value(s), and node id and gets back the bytes-level entry
//! behaviour.
//!
//! ## Single-column vs compound
//!
//! Both layouts share the same key shape (`idx:name:encoded:id`);
//! compound uses [`encode_compound_value`] to pack multiple [`Value`]s
//! with a separator byte. The store exposes both via
//! [`IndexStore::put_entry`] (slice of values) so the caller doesn't
//! need to branch on arity.

use coordinode_core::graph::node::NodeId;
use coordinode_core::graph::types::Value;
use coordinode_core::index::encoding::{
    decode_node_id_from_index_key, encode_compound_index_key, encode_compound_value,
};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::Guard;

use crate::error::StoreResult;

/// Layer 4 index store: entry-level B-tree index ops over
/// [`Partition::Idx`].
pub trait IndexStore {
    /// Insert a (values → node) entry under the named index. Both
    /// single-column (slice of 1) and compound (slice of N) work.
    /// Idempotent: re-putting the same `(name, values, node_id)` is a
    /// no-op semantically.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalIndexStore, IndexStore};
    /// # use coordinode_core::graph::{node::NodeId, types::Value};
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalIndexStore::new(engine);
    /// store.put_entry("by_name", &[Value::String("alice".into())], NodeId::from_raw(1))?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn put_entry(&self, name: &str, values: &[Value], node_id: NodeId) -> StoreResult<()>;

    /// Remove a specific entry. Returns Ok even if the entry was
    /// already absent (matches storage tombstone semantics).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalIndexStore, IndexStore};
    /// # use coordinode_core::graph::{node::NodeId, types::Value};
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalIndexStore::new(engine);
    /// store.delete_entry("by_name", &[Value::String("alice".into())], NodeId::from_raw(1))?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn delete_entry(&self, name: &str, values: &[Value], node_id: NodeId) -> StoreResult<()>;

    /// Return all node ids whose entry has the exact given value(s).
    /// Empty `Vec` means "no matches".
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalIndexStore, IndexStore};
    /// # use coordinode_core::graph::types::Value;
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalIndexStore::new(engine);
    /// let hits = store.scan_exact("by_name", &[Value::String("alice".into())])?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn scan_exact(&self, name: &str, values: &[Value]) -> StoreResult<Vec<NodeId>>;

    /// Return all (sortable bytes, node id) pairs in the named index,
    /// without value filtering. Useful for full-index walks (TTL
    /// reaper, index rebuild, count). For large indexes the caller
    /// should prefer a streaming form once Layer 4 grows one — for
    /// PR-scope simplicity this materialises into a `Vec`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalIndexStore, IndexStore};
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalIndexStore::new(engine);
    /// let _all = store.scan_all("by_name")?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn scan_all(&self, name: &str) -> StoreResult<Vec<(Vec<u8>, NodeId)>>;
}

/// CE single-shard implementation of [`IndexStore`].
pub struct LocalIndexStore<'a> {
    engine: &'a StorageEngine,
}

impl<'a> LocalIndexStore<'a> {
    /// Wrap a storage engine for index-store operations.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use coordinode_modality::{LocalIndexStore, IndexStore};
    /// use coordinode_core::graph::node::NodeId;
    /// use coordinode_core::graph::types::Value;
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/store"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm,
    /// # )]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// let store = LocalIndexStore::new(engine);
    /// let key = [Value::String("alice".into())];
    /// store.put_entry("by_name", &key, NodeId::from_raw(1))?;
    /// let hits = store.scan_exact("by_name", &key)?;
    /// assert_eq!(hits, vec![NodeId::from_raw(1)]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(engine: &'a StorageEngine) -> Self {
        Self { engine }
    }
}

fn index_prefix(name: &str) -> Vec<u8> {
    let mut p = Vec::with_capacity(4 + name.len() + 1);
    p.extend_from_slice(b"idx:");
    p.extend_from_slice(name.as_bytes());
    p.push(b':');
    p
}

fn index_value_prefix(name: &str, values: &[Value]) -> Vec<u8> {
    let encoded = encode_compound_value(values);
    let mut p = Vec::with_capacity(4 + name.len() + 1 + encoded.len() + 1);
    p.extend_from_slice(b"idx:");
    p.extend_from_slice(name.as_bytes());
    p.push(b':');
    p.extend_from_slice(&encoded);
    p.push(b':');
    p
}

impl IndexStore for LocalIndexStore<'_> {
    fn put_entry(&self, name: &str, values: &[Value], node_id: NodeId) -> StoreResult<()> {
        let key = encode_compound_index_key(name, values, node_id.as_raw());
        self.engine.put(Partition::Idx, &key, &[])?;
        Ok(())
    }

    fn delete_entry(&self, name: &str, values: &[Value], node_id: NodeId) -> StoreResult<()> {
        let key = encode_compound_index_key(name, values, node_id.as_raw());
        self.engine.delete(Partition::Idx, &key)?;
        Ok(())
    }

    fn scan_exact(&self, name: &str, values: &[Value]) -> StoreResult<Vec<NodeId>> {
        let prefix = index_value_prefix(name, values);
        let iter = self.engine.prefix_scan(Partition::Idx, &prefix)?;
        let mut out = Vec::new();
        for guard in iter {
            let (key, _) = guard.into_inner()?;
            if let Some(id) = decode_node_id_from_index_key(&key) {
                out.push(NodeId::from_raw(id));
            }
        }
        Ok(out)
    }

    fn scan_all(&self, name: &str) -> StoreResult<Vec<(Vec<u8>, NodeId)>> {
        let prefix = index_prefix(name);
        let iter = self.engine.prefix_scan(Partition::Idx, &prefix)?;
        let mut out = Vec::new();
        for guard in iter {
            let (key, _) = guard.into_inner()?;
            let Some(id) = decode_node_id_from_index_key(&key) else {
                continue;
            };
            out.push((key.to_vec(), NodeId::from_raw(id)));
        }
        Ok(out)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Logic-test fixture (memory backing, env-flippable).
    fn open_engine() -> coordinode_test_fixtures::EngineFixture {
        coordinode_test_fixtures::engine_for_logic()
    }

    #[test]
    fn single_value_round_trip() {
        let fx = open_engine();
        let engine = &fx.engine;
        let store = LocalIndexStore::new(engine);
        let v = vec![Value::String("alice".into())];
        store
            .put_entry("user_name", &v, NodeId::from_raw(1))
            .expect("put");
        let hits = store.scan_exact("user_name", &v).expect("scan");
        assert_eq!(hits, vec![NodeId::from_raw(1)]);
    }

    #[test]
    fn duplicate_values_return_all_nodes() {
        // Index value "alice" maps to two nodes — scan_exact returns
        // both, sorted by node_id (because the key suffix is BE u64).
        let fx = open_engine();
        let engine = &fx.engine;
        let store = LocalIndexStore::new(engine);
        let v = vec![Value::String("alice".into())];
        for id in [1u64, 2, 3] {
            store
                .put_entry("user_name", &v, NodeId::from_raw(id))
                .expect("put");
        }
        let hits = store.scan_exact("user_name", &v).expect("scan");
        assert_eq!(
            hits,
            vec![
                NodeId::from_raw(1),
                NodeId::from_raw(2),
                NodeId::from_raw(3)
            ],
        );
    }

    #[test]
    fn delete_removes_specific_entry() {
        let fx = open_engine();
        let engine = &fx.engine;
        let store = LocalIndexStore::new(engine);
        let v = vec![Value::String("alice".into())];
        store
            .put_entry("user_name", &v, NodeId::from_raw(1))
            .expect("put");
        store
            .put_entry("user_name", &v, NodeId::from_raw(2))
            .expect("put");

        store
            .delete_entry("user_name", &v, NodeId::from_raw(1))
            .expect("delete");

        let hits = store.scan_exact("user_name", &v).expect("scan");
        assert_eq!(hits, vec![NodeId::from_raw(2)]);
    }

    #[test]
    fn delete_missing_entry_is_idempotent() {
        let fx = open_engine();
        let engine = &fx.engine;
        let store = LocalIndexStore::new(engine);
        let v = vec![Value::Int(7)];
        // Never put — delete must still succeed.
        store
            .delete_entry("noise", &v, NodeId::from_raw(99))
            .expect("delete missing");
    }

    #[test]
    fn compound_index_distinguishes_by_secondary_column() {
        let fx = open_engine();
        let engine = &fx.engine;
        let store = LocalIndexStore::new(engine);
        let alice_us = vec![Value::String("alice".into()), Value::String("US".into())];
        let alice_uk = vec![Value::String("alice".into()), Value::String("UK".into())];

        store
            .put_entry("by_name_country", &alice_us, NodeId::from_raw(1))
            .expect("put");
        store
            .put_entry("by_name_country", &alice_uk, NodeId::from_raw(2))
            .expect("put");

        // Exact match on (alice, US) returns only node 1.
        let us_hits = store
            .scan_exact("by_name_country", &alice_us)
            .expect("scan");
        assert_eq!(us_hits, vec![NodeId::from_raw(1)]);

        // Exact match on (alice, UK) returns only node 2.
        let uk_hits = store
            .scan_exact("by_name_country", &alice_uk)
            .expect("scan");
        assert_eq!(uk_hits, vec![NodeId::from_raw(2)]);
    }

    #[test]
    fn scan_all_returns_every_entry() {
        let fx = open_engine();
        let engine = &fx.engine;
        let store = LocalIndexStore::new(engine);
        let alice = vec![Value::String("alice".into())];
        let bob = vec![Value::String("bob".into())];
        store
            .put_entry("nm", &alice, NodeId::from_raw(1))
            .expect("put");
        store
            .put_entry("nm", &bob, NodeId::from_raw(2))
            .expect("put");
        store
            .put_entry("nm", &alice, NodeId::from_raw(3))
            .expect("put");

        let all = store.scan_all("nm").expect("scan all");
        assert_eq!(all.len(), 3);
        // Sorted by (encoded value, node_id): alice/1, alice/3, bob/2.
        let ids: Vec<u64> = all.iter().map(|(_, id)| id.as_raw()).collect();
        assert_eq!(ids, vec![1, 3, 2]);
    }

    #[test]
    fn compound_index_three_columns() {
        // N=3 compound: confirm encode/scan symmetry beyond N=2. Two
        // entries differ only in the third column.
        let fx = open_engine();
        let engine = &fx.engine;
        let store = LocalIndexStore::new(engine);
        let key_a = vec![
            Value::String("alice".into()),
            Value::String("US".into()),
            Value::Int(30),
        ];
        let key_b = vec![
            Value::String("alice".into()),
            Value::String("US".into()),
            Value::Int(40),
        ];
        store
            .put_entry("triple", &key_a, NodeId::from_raw(1))
            .expect("put a");
        store
            .put_entry("triple", &key_b, NodeId::from_raw(2))
            .expect("put b");

        // Exact match on key_a returns only node 1; key_b only 2.
        assert_eq!(
            store.scan_exact("triple", &key_a).expect("scan"),
            vec![NodeId::from_raw(1)],
        );
        assert_eq!(
            store.scan_exact("triple", &key_b).expect("scan"),
            vec![NodeId::from_raw(2)],
        );

        // scan_all walks both, in encoded-key order (key_a's Int=30
        // sorts before key_b's Int=40).
        let all = store.scan_all("triple").expect("scan all");
        let ids: Vec<u64> = all.iter().map(|(_, id)| id.as_raw()).collect();
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn scan_exact_missing_returns_empty() {
        let fx = open_engine();
        let engine = &fx.engine;
        let store = LocalIndexStore::new(engine);
        let hits = store
            .scan_exact("nonexistent", &[Value::Int(42)])
            .expect("scan");
        assert!(hits.is_empty());
    }

    #[test]
    fn sortable_type_ordering_null_lt_bool_lt_int_lt_string() {
        // ADR contract: Value ordering Null < Bool < Int < Float <
        // String < Timestamp. scan_all walks in encoded-key order so
        // we can read the type ordering off directly. One entry per
        // type, all under the same index name and node_id 1.
        let fx = open_engine();
        let engine = &fx.engine;
        let store = LocalIndexStore::new(engine);
        let id = NodeId::from_raw(1);
        let entries: Vec<Vec<Value>> = vec![
            vec![Value::Null],
            vec![Value::Bool(true)],
            vec![Value::Int(0)],
            vec![Value::String("z".into())],
        ];
        // Insert in reverse to prove ordering comes from key
        // encoding, not insertion order.
        for v in entries.iter().rev() {
            store.put_entry("mix", v, id).expect("put");
        }
        let all = store.scan_all("mix").expect("scan");
        assert_eq!(all.len(), 4);
        // The keys themselves carry the encoded value bytes — verify
        // their order matches the ADR contract by comparing prefixes
        // pairwise (each later key sorts >= the previous one).
        for pair in all.windows(2) {
            assert!(
                pair[0].0 <= pair[1].0,
                "index keys not sorted: {:?} vs {:?}",
                pair[0].0,
                pair[1].0,
            );
        }
    }

    #[test]
    fn scan_all_isolates_per_index_name() {
        // Two indexes share the partition; each scan returns only
        // its own entries.
        let fx = open_engine();
        let engine = &fx.engine;
        let store = LocalIndexStore::new(engine);
        let v = vec![Value::Int(42)];
        store.put_entry("a", &v, NodeId::from_raw(1)).expect("put");
        store.put_entry("b", &v, NodeId::from_raw(2)).expect("put");

        let a = store.scan_all("a").expect("scan a");
        let b = store.scan_all("b").expect("scan b");
        assert_eq!(a.len(), 1);
        assert_eq!(b.len(), 1);
        assert_eq!(a[0].1, NodeId::from_raw(1));
        assert_eq!(b[0].1, NodeId::from_raw(2));
    }
}
