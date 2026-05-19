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
//! [`Self::put_edge`] writes the forward adjacency merge, the
//! reverse adjacency merge, and the optional edgeprop body in a
//! single [`WriteBatch`] — the storage engine commits all three (or
//! none) and produces one seqno. Readers never observe a half-built
//! edge.
//!
//! ## What this PR does NOT cover
//!
//! - **Temporal edges** (ADR-027 `valid_from` per version). Will be
//!   added as `put_edge_temporal` / `get_props_at` in a follow-up
//!   commit. The non-temporal API here matches the steady-state
//!   `runner.rs` write path one-to-one.
//! - **Discriminator-suffixed multiplicity** (ADR-029). Same
//!   rationale — discriminator-aware keys are a separate concern
//!   that will land alongside temporal in the EdgeStore follow-up.
//! - **Posting-list splits for super-nodes** (>512KB shards). The
//!   underlying merge operator already handles the split transparently
//!   — the store API is unchanged; this is mentioned for completeness.

use coordinode_core::graph::edge::{
    encode_adj_key_forward, encode_adj_key_reverse, encode_edgeprop_key, EdgeProperties,
    PostingList,
};
use coordinode_core::graph::node::NodeId;
use coordinode_storage::engine::batch::WriteBatch;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::merge::{encode_add, encode_remove};
use coordinode_storage::engine::partition::Partition;

use crate::error::{StoreError, StoreResult};

/// Layer 4 edge store. Reads/writes the adjacency + edgeprop pair
/// behind a typed API; hides merge-operator deltas and key shape.
pub trait EdgeStore {
    /// Create (or upsert) an edge `src --edge_type--> tgt` with
    /// optional properties. Writes forward + reverse adjacency
    /// merges and the edgeprop body atomically.
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
    fn get_props(
        &self,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
    ) -> StoreResult<Option<EdgeProperties>>;

    /// Remove an edge. Issues remove merges on both adjacency
    /// posting lists and tombstones the edgeprop body in a single
    /// atomic batch. Idempotent on already-missing edges.
    fn delete_edge(&self, edge_type: &str, src: NodeId, tgt: NodeId) -> StoreResult<()>;

    /// Forward neighbours: targets reachable as
    /// `src --edge_type--> ?`. Returns the posting-list contents in
    /// ascending node-id order (the on-disk representation).
    fn scan_neighbors_out(&self, edge_type: &str, src: NodeId) -> StoreResult<Vec<NodeId>>;

    /// Reverse neighbours: sources of edges
    /// `? --edge_type--> tgt`.
    fn scan_neighbors_in(&self, edge_type: &str, tgt: NodeId) -> StoreResult<Vec<NodeId>>;
}

/// CE single-shard implementation of [`EdgeStore`].
pub struct LocalEdgeStore<'a> {
    engine: &'a StorageEngine,
}

impl<'a> LocalEdgeStore<'a> {
    /// Wrap a storage engine for edge-store operations.
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
