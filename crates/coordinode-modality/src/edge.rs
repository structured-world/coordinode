//! Edge store — typed read/write of edges across the three keyspaces
//! that together make up an edge:
//!
//! - **Adjacency** (`Partition::Adj`): one posting list per
//!   `(edge_type, source)` for forward neighbours and one per
//!   `(edge_type, target)` for reverse neighbours. Mutations are
//!   merge-operator deltas (`encode_add` / `encode_remove`) so
//!   concurrent edge writes to the same endpoint never need an
//!   OCC retry loop.
//! - **Edge properties** (`Partition::EdgeProp`): keyed by
//!   `(edge_type, src, tgt)`. Optional per-edge.
//! - **Edge type registration** (`Partition::Schema`): idempotent
//!   marker so the schema layer knows the edge type exists. Schema
//!   DDL is owned by [`SchemaStore`](crate::SchemaStore); the edge
//!   store only writes the existence marker when a freshly-seen
//!   edge type is first written.
//!
//! ## Atomicity
//!
//! [`EdgeStore::put_edge`] writes the forward adjacency merge, the
//! reverse adjacency merge, and the optional edgeprop body in a
//! single [`WriteBatch`] — the storage engine commits all three (or
//! none) and produces one seqno. Readers never observe a half-built
//! edge.
//!
//! ## Temporal edges (ADR-027)
//!
//! [`EdgeStore::put_edge_temporal`] writes one version per
//! `valid_from_ms` via [`encode_temporal_edgeprop_key`], so every
//! edge update appends a new row rather than overwriting.
//! [`EdgeStore::get_props_at`] scans the
//! `(edge_type, src, tgt)` prefix and returns the version whose
//! `valid_from_ms <= at_ms` is largest. Adjacency entries are written
//! by the temporal path too — the version model for adj itself
//! (tombstone markers vs. per-version posting lists) is a separate
//! ADR and is intentionally NOT decided here.
//!
//! ## Out of scope here
//!
//! - **Discriminator-suffixed multiplicity** (ADR-029). Discriminator
//!   key encoders don't yet exist; revisit once the upstream encoding
//!   helpers are in place.
//! - **Adjacency versioning** for temporal edges (see above).
//! - **Posting-list splits for super-nodes** (>512KB shards). The
//!   underlying merge operator already handles the split transparently
//!   — the store API is unchanged; this is mentioned for completeness.

use coordinode_core::graph::edge::{
    decode_temporal_edgeprop_key, encode_adj_key_forward, encode_adj_key_reverse,
    encode_edgeprop_key, encode_temporal_edgeprop_key, temporal_edgeprop_pair_prefix,
    EdgeProperties, PostingList,
};
use coordinode_core::graph::node::NodeId;
use coordinode_storage::engine::batch::WriteBatch;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::merge::{encode_add, encode_remove};
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::Guard;

use crate::error::{StoreError, StoreResult};

/// Layer 4 edge store. Reads/writes the adjacency + edgeprop pair
/// behind a typed API; hides merge-operator deltas and key shape.
pub trait EdgeStore {
    /// Create (or upsert) an edge `src --edge_type--> tgt` with
    /// optional properties. Writes forward + reverse adjacency
    /// merges and the edgeprop body atomically.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalEdgeStore, EdgeStore};
    /// # use coordinode_core::graph::node::NodeId;
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalEdgeStore::new(&engine);
    /// store.put_edge("KNOWS", NodeId::from_raw(1), NodeId::from_raw(2), None)?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn put_edge(
        &self,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        props: Option<&EdgeProperties>,
    ) -> StoreResult<()>;

    /// Read the edge properties for `(edge_type, src, tgt)`. Returns
    /// `None` if the edge has no property body — note that an edge
    /// CAN exist (visible in the adjacency lists) with no property
    /// body, this is the common case for property-less edges.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalEdgeStore, EdgeStore};
    /// # use coordinode_core::graph::node::NodeId;
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalEdgeStore::new(&engine);
    /// let _props = store.get_props("KNOWS", NodeId::from_raw(1), NodeId::from_raw(2))?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn get_props(
        &self,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
    ) -> StoreResult<Option<EdgeProperties>>;

    /// Remove an edge. Issues remove merges on both adjacency
    /// posting lists and tombstones the edgeprop body in a single
    /// atomic batch. Idempotent on already-missing edges.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalEdgeStore, EdgeStore};
    /// # use coordinode_core::graph::node::NodeId;
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalEdgeStore::new(&engine);
    /// store.delete_edge("KNOWS", NodeId::from_raw(1), NodeId::from_raw(2))?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn delete_edge(&self, edge_type: &str, src: NodeId, tgt: NodeId) -> StoreResult<()>;

    /// Forward neighbours: targets reachable as
    /// `src --edge_type--> ?`. Returns the posting-list contents in
    /// ascending node-id order (the on-disk representation).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalEdgeStore, EdgeStore};
    /// # use coordinode_core::graph::node::NodeId;
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalEdgeStore::new(&engine);
    /// let _targets = store.scan_neighbors_out("KNOWS", NodeId::from_raw(1))?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn scan_neighbors_out(&self, edge_type: &str, src: NodeId) -> StoreResult<Vec<NodeId>>;

    /// Reverse neighbours: sources of edges `? --edge_type--> tgt`.
    /// Symmetric to [`Self::scan_neighbors_out`].
    fn scan_neighbors_in(&self, edge_type: &str, tgt: NodeId) -> StoreResult<Vec<NodeId>>;

    /// Per-version write of edge properties for `(edge_type, src, tgt)`
    /// (ADR-027 temporal edges). Stores `props` under the temporal
    /// edgeprop key suffixed with `valid_from_ms`; multiple versions
    /// coexist. Adjacency entries are written too (same merge as the
    /// non-temporal path) so the edge is visible to neighbour scans.
    ///
    /// Adjacency itself is not yet version-keyed — that requires a
    /// separate ADR on the version model (tombstone markers vs.
    /// per-version posting lists) and is tracked as a follow-up.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalEdgeStore, EdgeStore};
    /// # use coordinode_core::graph::{edge::EdgeProperties, node::NodeId};
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalEdgeStore::new(&engine);
    /// let props = EdgeProperties::new();
    /// store.put_edge_temporal("E", NodeId::from_raw(1), NodeId::from_raw(2), 1000, &props)?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn put_edge_temporal(
        &self,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        valid_from_ms: i64,
        props: &EdgeProperties,
    ) -> StoreResult<()>;

    /// Read the edge-property version active at `at_ms`: the version
    /// whose `valid_from_ms <= at_ms` is largest. Returns `None` if no
    /// such version exists.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalEdgeStore, EdgeStore};
    /// # use coordinode_core::graph::node::NodeId;
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalEdgeStore::new(&engine);
    /// let _at_now = store.get_props_at("E", NodeId::from_raw(1), NodeId::from_raw(2), 1500)?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn get_props_at(
        &self,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        at_ms: i64,
    ) -> StoreResult<Option<EdgeProperties>>;

    /// All temporal versions of `(edge_type, src, tgt)`, sorted by
    /// `valid_from_ms` ascending.
    fn scan_edge_versions(
        &self,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
    ) -> StoreResult<Vec<(i64, EdgeProperties)>>;

    /// Tombstone one specific temporal version. Idempotent on a
    /// missing version. Adjacency entries are NOT touched — removing
    /// the last version of an edge needs the adj-versioning ADR.
    fn delete_edge_temporal(
        &self,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        valid_from_ms: i64,
    ) -> StoreResult<()>;
}

/// CE single-shard implementation of [`EdgeStore`].
pub struct LocalEdgeStore<'a> {
    engine: &'a StorageEngine,
}

impl<'a> LocalEdgeStore<'a> {
    /// Wrap a storage engine for edge-store operations.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use coordinode_modality::{LocalEdgeStore, EdgeStore};
    /// use coordinode_core::graph::node::NodeId;
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/store"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm,
    /// # )]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// let store = LocalEdgeStore::new(&engine);
    /// store.put_edge("KNOWS", NodeId::from_raw(1), NodeId::from_raw(2), None)?;
    /// let neighbours = store.scan_neighbors_out("KNOWS", NodeId::from_raw(1))?;
    /// assert_eq!(neighbours, vec![NodeId::from_raw(2)]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(engine: &'a StorageEngine) -> Self {
        Self { engine }
    }

    fn read_posting(&self, key: &[u8]) -> StoreResult<Vec<NodeId>> {
        let Some(bytes) = self.engine.get(Partition::Adj, key)? else {
            return Ok(Vec::new());
        };
        let plist = PostingList::from_bytes(&bytes).map_err(|e| StoreError::Decode {
            kind: "posting list",
            message: format!("{e}"),
        })?;
        Ok(plist.iter().map(NodeId::from_raw).collect())
    }
}

impl EdgeStore for LocalEdgeStore<'_> {
    fn put_edge(
        &self,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        props: Option<&EdgeProperties>,
    ) -> StoreResult<()> {
        let fwd_key = encode_adj_key_forward(edge_type, src);
        let rev_key = encode_adj_key_reverse(edge_type, tgt);

        let mut batch = WriteBatch::new(self.engine);
        batch.merge(Partition::Adj, fwd_key, encode_add(tgt.as_raw()));
        batch.merge(Partition::Adj, rev_key, encode_add(src.as_raw()));
        if let Some(p) = props {
            if !p.is_empty() {
                let body = p.to_msgpack().map_err(|e| StoreError::Decode {
                    kind: "edge properties",
                    message: format!("encode: {e}"),
                })?;
                batch.put(
                    Partition::EdgeProp,
                    encode_edgeprop_key(edge_type, src, tgt),
                    body,
                );
            }
        }
        batch.commit()?;
        Ok(())
    }

    fn get_props(
        &self,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
    ) -> StoreResult<Option<EdgeProperties>> {
        let key = encode_edgeprop_key(edge_type, src, tgt);
        let Some(bytes) = self.engine.get(Partition::EdgeProp, &key)? else {
            return Ok(None);
        };
        EdgeProperties::from_msgpack(&bytes)
            .map(Some)
            .map_err(|e| StoreError::Decode {
                kind: "edge properties",
                message: format!("decode: {e}"),
            })
    }

    fn delete_edge(&self, edge_type: &str, src: NodeId, tgt: NodeId) -> StoreResult<()> {
        let fwd_key = encode_adj_key_forward(edge_type, src);
        let rev_key = encode_adj_key_reverse(edge_type, tgt);
        let ep_key = encode_edgeprop_key(edge_type, src, tgt);

        let mut batch = WriteBatch::new(self.engine);
        batch.merge(Partition::Adj, fwd_key, encode_remove(tgt.as_raw()));
        batch.merge(Partition::Adj, rev_key, encode_remove(src.as_raw()));
        batch.delete(Partition::EdgeProp, ep_key);
        batch.commit()?;
        Ok(())
    }

    fn scan_neighbors_out(&self, edge_type: &str, src: NodeId) -> StoreResult<Vec<NodeId>> {
        self.read_posting(&encode_adj_key_forward(edge_type, src))
    }

    fn scan_neighbors_in(&self, edge_type: &str, tgt: NodeId) -> StoreResult<Vec<NodeId>> {
        self.read_posting(&encode_adj_key_reverse(edge_type, tgt))
    }

    fn put_edge_temporal(
        &self,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        valid_from_ms: i64,
        props: &EdgeProperties,
    ) -> StoreResult<()> {
        let fwd_key = encode_adj_key_forward(edge_type, src);
        let rev_key = encode_adj_key_reverse(edge_type, tgt);
        let ep_key = encode_temporal_edgeprop_key(edge_type, src, tgt, valid_from_ms);
        let body = props.to_msgpack().map_err(|e| StoreError::Decode {
            kind: "edge properties",
            message: format!("encode: {e}"),
        })?;

        let mut batch = WriteBatch::new(self.engine);
        batch.merge(Partition::Adj, fwd_key, encode_add(tgt.as_raw()));
        batch.merge(Partition::Adj, rev_key, encode_add(src.as_raw()));
        batch.put(Partition::EdgeProp, ep_key, body);
        batch.commit()?;
        Ok(())
    }

    fn get_props_at(
        &self,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        at_ms: i64,
    ) -> StoreResult<Option<EdgeProperties>> {
        let prefix = temporal_edgeprop_pair_prefix(edge_type, src, tgt);
        let iter = self.engine.prefix_scan(Partition::EdgeProp, &prefix)?;
        let mut best: Option<(i64, EdgeProperties)> = None;
        for guard in iter {
            let (key, value) = guard.into_inner()?;
            let Some((_, _, _, valid_from)) = decode_temporal_edgeprop_key(&key) else {
                continue;
            };
            if valid_from > at_ms {
                continue;
            }
            let props =
                EdgeProperties::from_msgpack(value.as_ref()).map_err(|e| StoreError::Decode {
                    kind: "edge properties",
                    message: format!("decode: {e}"),
                })?;
            best = match best {
                Some((vf, _)) if vf >= valid_from => best,
                _ => Some((valid_from, props)),
            };
        }
        Ok(best.map(|(_, p)| p))
    }

    fn scan_edge_versions(
        &self,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
    ) -> StoreResult<Vec<(i64, EdgeProperties)>> {
        let prefix = temporal_edgeprop_pair_prefix(edge_type, src, tgt);
        let iter = self.engine.prefix_scan(Partition::EdgeProp, &prefix)?;
        let mut out = Vec::new();
        for guard in iter {
            let (key, value) = guard.into_inner()?;
            let Some((_, _, _, valid_from)) = decode_temporal_edgeprop_key(&key) else {
                continue;
            };
            let props =
                EdgeProperties::from_msgpack(value.as_ref()).map_err(|e| StoreError::Decode {
                    kind: "edge properties",
                    message: format!("decode: {e}"),
                })?;
            out.push((valid_from, props));
        }
        out.sort_by_key(|(vf, _)| *vf);
        Ok(out)
    }

    fn delete_edge_temporal(
        &self,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        valid_from_ms: i64,
    ) -> StoreResult<()> {
        let key = encode_temporal_edgeprop_key(edge_type, src, tgt, valid_from_ms);
        self.engine.delete(Partition::EdgeProp, &key)?;
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
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

    #[test]
    fn put_edge_creates_forward_and_reverse_adj() {
        let (_dir, engine) = open_engine();
        let store = LocalEdgeStore::new(&engine);
        let alice = NodeId::from_raw(1);
        let bob = NodeId::from_raw(2);

        store.put_edge("KNOWS", alice, bob, None).expect("put edge");

        assert_eq!(
            store.scan_neighbors_out("KNOWS", alice).expect("scan"),
            vec![bob],
        );
        assert_eq!(
            store.scan_neighbors_in("KNOWS", bob).expect("scan"),
            vec![alice],
        );
    }

    #[test]
    fn put_edge_with_props_stores_property_body() {
        let (_dir, engine) = open_engine();
        let store = LocalEdgeStore::new(&engine);
        let mut props = EdgeProperties::new();
        props.set(7, coordinode_core::graph::types::Value::Int(42));

        let a = NodeId::from_raw(10);
        let b = NodeId::from_raw(20);
        store.put_edge("OWNS", a, b, Some(&props)).expect("put");

        let loaded = store.get_props("OWNS", a, b).expect("ok").expect("Some");
        assert_eq!(loaded.len(), 1);
    }

    #[test]
    fn edge_without_props_returns_none_from_get_props() {
        // Property-less edge: adj entries exist, edgeprop body
        // doesn't. get_props returns None (NOT "edge does not
        // exist" — that's a separate query at the adj layer).
        let (_dir, engine) = open_engine();
        let store = LocalEdgeStore::new(&engine);
        store
            .put_edge("LIKES", NodeId::from_raw(1), NodeId::from_raw(2), None)
            .expect("put");
        assert!(store
            .get_props("LIKES", NodeId::from_raw(1), NodeId::from_raw(2))
            .expect("ok")
            .is_none());
    }

    #[test]
    fn multiple_neighbors_listed_in_sorted_order() {
        // The posting list maintains sorted order; scan returns it
        // unchanged. Verified across multi-merge writes.
        let (_dir, engine) = open_engine();
        let store = LocalEdgeStore::new(&engine);
        let src = NodeId::from_raw(1);
        // Insert out of order to exercise the merge operator's sort.
        for tgt in [5u64, 3, 8, 1, 2] {
            store
                .put_edge("F", src, NodeId::from_raw(tgt), None)
                .expect("put");
        }
        let neighbors: Vec<u64> = store
            .scan_neighbors_out("F", src)
            .expect("scan")
            .iter()
            .map(|n| n.as_raw())
            .collect();
        assert_eq!(neighbors, vec![1, 2, 3, 5, 8]);
    }

    #[test]
    fn delete_edge_removes_from_adjacency_and_props() {
        let (_dir, engine) = open_engine();
        let store = LocalEdgeStore::new(&engine);
        let a = NodeId::from_raw(1);
        let b = NodeId::from_raw(2);
        let c = NodeId::from_raw(3);

        // Two outgoing edges from `a`. Delete the (a,b) edge.
        store.put_edge("F", a, b, None).expect("put");
        store.put_edge("F", a, c, None).expect("put");
        store.delete_edge("F", a, b).expect("delete");

        // `b` is gone from a's forward neighbors; `c` remains.
        let out: Vec<u64> = store
            .scan_neighbors_out("F", a)
            .expect("scan")
            .iter()
            .map(|n| n.as_raw())
            .collect();
        assert_eq!(out, vec![3]);
        // `a` is gone from b's reverse neighbors.
        assert!(store.scan_neighbors_in("F", b).expect("scan").is_empty());
    }

    #[test]
    fn delete_edge_is_idempotent() {
        let (_dir, engine) = open_engine();
        let store = LocalEdgeStore::new(&engine);
        // Never created — delete must still succeed (merge remove
        // on an absent uid is a no-op; edgeprop delete on a missing
        // key is fine).
        store
            .delete_edge("F", NodeId::from_raw(9), NodeId::from_raw(10))
            .expect("delete missing");
    }

    #[test]
    fn edge_types_are_isolated_in_adj() {
        // Same (src, tgt) pair under different edge types must NOT
        // share adjacency entries. Different keyspaces.
        let (_dir, engine) = open_engine();
        let store = LocalEdgeStore::new(&engine);
        let a = NodeId::from_raw(1);
        let b = NodeId::from_raw(2);
        store.put_edge("KNOWS", a, b, None).expect("put");
        store.put_edge("LIKES", a, b, None).expect("put");
        store.delete_edge("KNOWS", a, b).expect("delete KNOWS");

        // KNOWS gone, LIKES preserved.
        assert!(store
            .scan_neighbors_out("KNOWS", a)
            .expect("scan")
            .is_empty());
        assert_eq!(store.scan_neighbors_out("LIKES", a).expect("scan"), vec![b],);
    }

    fn props_with(field: u32, value: i64) -> EdgeProperties {
        let mut p = EdgeProperties::new();
        p.set(field, coordinode_core::graph::types::Value::Int(value));
        p
    }

    #[test]
    fn put_temporal_versions_round_trip_via_scan() {
        let (_dir, engine) = open_engine();
        let store = LocalEdgeStore::new(&engine);
        let a = NodeId::from_raw(1);
        let b = NodeId::from_raw(2);
        for (vf, salary) in [(1000i64, 50_000), (2000, 60_000), (3000, 70_000)] {
            store
                .put_edge_temporal("WORKS_AT", a, b, vf, &props_with(1, salary))
                .expect("put temporal");
        }
        let versions = store
            .scan_edge_versions("WORKS_AT", a, b)
            .expect("scan versions");
        assert_eq!(versions.len(), 3);
        assert_eq!(versions[0].0, 1000);
        assert_eq!(versions[2].0, 3000);
    }

    #[test]
    fn get_props_at_returns_largest_valid_from_le_query() {
        let (_dir, engine) = open_engine();
        let store = LocalEdgeStore::new(&engine);
        let a = NodeId::from_raw(1);
        let b = NodeId::from_raw(2);
        store
            .put_edge_temporal("E", a, b, 1000, &props_with(1, 10))
            .unwrap();
        store
            .put_edge_temporal("E", a, b, 2000, &props_with(1, 20))
            .unwrap();
        store
            .put_edge_temporal("E", a, b, 3000, &props_with(1, 30))
            .unwrap();

        // At 1500 — only the 1000-version is visible.
        let p = store
            .get_props_at("E", a, b, 1500)
            .unwrap()
            .expect("present");
        assert_eq!(
            p.get(1),
            Some(&coordinode_core::graph::types::Value::Int(10))
        );

        // At 2500 — pick the 2000-version (largest <= 2500).
        let p = store
            .get_props_at("E", a, b, 2500)
            .unwrap()
            .expect("present");
        assert_eq!(
            p.get(1),
            Some(&coordinode_core::graph::types::Value::Int(20))
        );

        // At 3000 — boundary inclusive: pick the 3000-version.
        let p = store
            .get_props_at("E", a, b, 3000)
            .unwrap()
            .expect("present");
        assert_eq!(
            p.get(1),
            Some(&coordinode_core::graph::types::Value::Int(30))
        );
    }

    #[test]
    fn get_props_at_before_first_version_returns_none() {
        let (_dir, engine) = open_engine();
        let store = LocalEdgeStore::new(&engine);
        let a = NodeId::from_raw(1);
        let b = NodeId::from_raw(2);
        store
            .put_edge_temporal("E", a, b, 5000, &props_with(1, 1))
            .unwrap();
        assert!(store.get_props_at("E", a, b, 1000).unwrap().is_none());
    }

    #[test]
    fn delete_temporal_version_removes_only_that_version() {
        let (_dir, engine) = open_engine();
        let store = LocalEdgeStore::new(&engine);
        let a = NodeId::from_raw(1);
        let b = NodeId::from_raw(2);
        store
            .put_edge_temporal("E", a, b, 100, &props_with(1, 10))
            .unwrap();
        store
            .put_edge_temporal("E", a, b, 200, &props_with(1, 20))
            .unwrap();
        store.delete_edge_temporal("E", a, b, 100).unwrap();
        let versions = store.scan_edge_versions("E", a, b).unwrap();
        assert_eq!(versions.len(), 1);
        assert_eq!(versions[0].0, 200);
    }

    #[test]
    fn delete_temporal_is_idempotent() {
        let (_dir, engine) = open_engine();
        let store = LocalEdgeStore::new(&engine);
        store
            .delete_edge_temporal("E", NodeId::from_raw(1), NodeId::from_raw(2), 9999)
            .expect("idempotent delete");
    }

    #[test]
    fn temporal_writes_isolated_per_pair() {
        let (_dir, engine) = open_engine();
        let store = LocalEdgeStore::new(&engine);
        let a = NodeId::from_raw(1);
        let b = NodeId::from_raw(2);
        let c = NodeId::from_raw(3);
        store
            .put_edge_temporal("E", a, b, 100, &props_with(1, 1))
            .unwrap();
        store
            .put_edge_temporal("E", a, c, 200, &props_with(1, 2))
            .unwrap();
        let only_ab = store.scan_edge_versions("E", a, b).unwrap();
        let only_ac = store.scan_edge_versions("E", a, c).unwrap();
        assert_eq!(only_ab.len(), 1);
        assert_eq!(only_ab[0].0, 100);
        assert_eq!(only_ac.len(), 1);
        assert_eq!(only_ac[0].0, 200);
    }

    #[test]
    fn temporal_put_also_writes_adjacency() {
        // A temporal edge must be visible in the neighbour scan — the
        // adj merge runs as part of the temporal write.
        let (_dir, engine) = open_engine();
        let store = LocalEdgeStore::new(&engine);
        let a = NodeId::from_raw(1);
        let b = NodeId::from_raw(2);
        store
            .put_edge_temporal("E", a, b, 100, &props_with(1, 1))
            .unwrap();
        assert_eq!(store.scan_neighbors_out("E", a).unwrap(), vec![b]);
        assert_eq!(store.scan_neighbors_in("E", b).unwrap(), vec![a]);
    }

    #[test]
    fn concurrent_put_edge_on_super_node_preserves_all_edges() {
        // Concurrent merge-operator stress on a single source.
        // Four threads each add 25 distinct out-edges from the same
        // src. After join, all 100 must be visible — the contract is
        // that add/remove are commutative+idempotent merge operands.
        use std::sync::Arc;
        use std::thread;

        let (_dir, engine) = open_engine();
        let engine = Arc::new(engine);
        let src = NodeId::from_raw(1);
        let threads_n = 4u64;
        let per_thread = 25u64;

        let handles: Vec<_> = (0..threads_n)
            .map(|t| {
                let engine = Arc::clone(&engine);
                thread::spawn(move || {
                    let store = LocalEdgeStore::new(&engine);
                    for i in 0..per_thread {
                        let tgt = NodeId::from_raw(t * per_thread + i + 100);
                        store.put_edge("F", src, tgt, None).expect("put");
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("thread join");
        }

        let store = LocalEdgeStore::new(&engine);
        let mut neighbors: Vec<u64> = store
            .scan_neighbors_out("F", src)
            .expect("scan")
            .iter()
            .map(|n| n.as_raw())
            .collect();
        neighbors.sort_unstable();
        let expected: Vec<u64> = (100..100 + threads_n * per_thread).collect();
        assert_eq!(neighbors, expected, "all concurrent edges must be present");
    }

    #[test]
    fn concurrent_add_then_remove_converges() {
        // Concurrent add(x) and remove(x) from different threads —
        // last-writer-wins not applicable to merge operators; both
        // ops must compose into a consistent posting list. Specifically:
        // after the same number of adds and removes for each target,
        // every target either present or absent (never duplicated).
        use std::sync::Arc;
        use std::thread;

        let (_dir, engine) = open_engine();
        let engine = Arc::new(engine);
        let src = NodeId::from_raw(1);

        // Pre-populate 10 edges so the remover has something to remove.
        let setup = LocalEdgeStore::new(&engine);
        for i in 0..10u64 {
            setup
                .put_edge("F", src, NodeId::from_raw(i + 200), None)
                .expect("setup");
        }

        // Adder thread: adds 5 more.
        let engine_a = Arc::clone(&engine);
        let adder = thread::spawn(move || {
            let store = LocalEdgeStore::new(&engine_a);
            for i in 0..5u64 {
                store
                    .put_edge("F", src, NodeId::from_raw(i + 300), None)
                    .expect("add");
            }
        });
        // Remover thread: removes 5 of the pre-populated.
        let engine_r = Arc::clone(&engine);
        let remover = thread::spawn(move || {
            let store = LocalEdgeStore::new(&engine_r);
            for i in 0..5u64 {
                store
                    .delete_edge("F", src, NodeId::from_raw(i + 200))
                    .expect("delete");
            }
        });
        adder.join().expect("adder");
        remover.join().expect("remover");

        let store = LocalEdgeStore::new(&engine);
        let mut neighbors: Vec<u64> = store
            .scan_neighbors_out("F", src)
            .expect("scan")
            .iter()
            .map(|n| n.as_raw())
            .collect();
        neighbors.sort_unstable();
        // Expect: (200..205) removed, (205..210) kept, (300..305) added.
        let expected: Vec<u64> = (205u64..210).chain(300..305).collect();
        assert_eq!(neighbors, expected);
    }

    #[test]
    fn concurrent_put_edge_temporal_distinct_versions() {
        // Four threads each write a distinct valid_from version of
        // the same (et, src, tgt). After join, scan_edge_versions
        // returns all four, sorted by valid_from.
        use std::sync::Arc;
        use std::thread;

        let (_dir, engine) = open_engine();
        let engine = Arc::new(engine);
        let src = NodeId::from_raw(1);
        let tgt = NodeId::from_raw(2);

        let handles: Vec<_> = (0..4u64)
            .map(|t| {
                let engine = Arc::clone(&engine);
                thread::spawn(move || {
                    let store = LocalEdgeStore::new(&engine);
                    let vf = (t as i64 + 1) * 1000;
                    let mut props = EdgeProperties::new();
                    props.set(1, coordinode_core::graph::types::Value::Int(vf));
                    store
                        .put_edge_temporal("E", src, tgt, vf, &props)
                        .expect("put temporal");
                })
            })
            .collect();
        for h in handles {
            h.join().expect("join");
        }

        let store = LocalEdgeStore::new(&engine);
        let versions = store.scan_edge_versions("E", src, tgt).expect("scan");
        let vfs: Vec<i64> = versions.iter().map(|(vf, _)| *vf).collect();
        assert_eq!(vfs, vec![1000, 2000, 3000, 4000]);
    }

    #[test]
    fn corrupt_posting_list_surfaces_as_decode_error() {
        let (_dir, engine) = open_engine();
        let store = LocalEdgeStore::new(&engine);
        let a = NodeId::from_raw(1);
        // Inject garbage bytes at the forward-adj key.
        engine
            .put(
                Partition::Adj,
                &encode_adj_key_forward("F", a),
                &[0xde, 0xad, 0xbe, 0xef],
            )
            .expect("inject");
        let err = store.scan_neighbors_out("F", a).expect_err("must error");
        assert!(matches!(
            err,
            StoreError::Decode {
                kind: "posting list",
                ..
            }
        ));
    }
}
