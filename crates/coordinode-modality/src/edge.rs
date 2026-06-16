//! Edge store — typed read/write of edges across the three keyspaces
//! that together make up an edge:
//!
//! - **Adjacency** (`Partition::Adj`): one posting list per
//!   `(edge_type, source)` for forward neighbours and one per
//!   `(edge_type, target)` for reverse neighbours. Mutations are
//!   commutative merge-operator deltas buffered on the transaction
//!   (`Transaction::merge_adj_add` / `merge_adj_remove`) so concurrent
//!   edge writes to the same endpoint never need an OCC retry loop.
//! - **Edge properties** (`Partition::EdgeProp`): keyed by
//!   `(edge_type, src, tgt)`. Optional per-edge.
//!
//! ## Transaction threading (ADR-041)
//!
//! Every method takes an explicit `&mut Transaction` (writes) or
//! `&Transaction` (reads). Writes buffer on the transaction and are
//! applied atomically by [`Transaction::commit`]; the forward
//! adjacency merge, the reverse adjacency merge, and the optional
//! edgeprop body all land in one commit (or none), so readers never
//! observe a half-built edge. Adjacency reads overlay the
//! transaction's pending merge operands (read-your-own-writes) on the
//! committed posting list.
//!
//! ## Temporal edges (ADR-027)
//!
//! [`EdgeStore::put_edge_temporal`] writes one version per
//! `valid_from_ms` via [`encode_temporal_edgeprop_key`], so every
//! edge update appends a new row rather than overwriting.
//! [`EdgeStore::get_props_at`] scans the `(edge_type, src, tgt)`
//! prefix and returns the version whose `valid_from_ms <= at_ms` is
//! largest. Adjacency entries are written by the temporal path too.
//! Adjacency versioning itself (tombstone markers vs. per-version
//! posting lists) is a separate ADR and intentionally NOT decided here.

use coordinode_core::graph::edge::{
    decode_temporal_edgeprop_key, encode_adj_key_forward, encode_adj_key_reverse,
    encode_edge_props, encode_edgeprop_key, encode_temporal_edgeprop_key,
    temporal_edgeprop_pair_prefix, valid_from_upper_bound_key, write_adj_key_forward,
    write_adj_key_reverse, EdgeProperties, PostingList,
};
use coordinode_core::graph::node::NodeId;
use coordinode_core::graph::types::Value;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::transaction::Transaction;

use crate::error::{StoreError, StoreResult};

/// Layer 4 edge store. Reads/writes the adjacency + edgeprop pair
/// behind a typed API over a [`Transaction`]; hides merge-operator
/// deltas and key shape from callers.
pub trait EdgeStore {
    /// Create (or upsert) an edge `src --edge_type--> tgt` with
    /// optional properties. Buffers forward + reverse adjacency merges
    /// and the edgeprop body on the transaction; they commit atomically.
    fn put_edge(
        &self,
        txn: &mut Transaction,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        props: Option<&EdgeProperties>,
    ) -> StoreResult<()>;

    /// Read the edge properties for `(edge_type, src, tgt)`. Returns
    /// `None` if the edge has no property body — an edge CAN exist
    /// (visible in the adjacency lists) with no property body, the
    /// common case for property-less edges.
    fn get_props(
        &self,
        txn: &Transaction,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
    ) -> StoreResult<Option<EdgeProperties>>;

    /// Remove an edge. Buffers remove merges on both adjacency posting
    /// lists and a tombstone on the edgeprop body; commits atomically.
    /// Idempotent on already-missing edges.
    fn delete_edge(
        &self,
        txn: &mut Transaction,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
    ) -> StoreResult<()>;

    /// Forward neighbours: targets reachable as `src --edge_type--> ?`.
    /// Returns the posting-list contents (read-your-own-writes applied)
    /// in ascending node-id order.
    fn scan_neighbors_out(
        &self,
        txn: &Transaction,
        edge_type: &str,
        src: NodeId,
    ) -> StoreResult<Vec<NodeId>>;

    /// Reverse neighbours: sources of edges `? --edge_type--> tgt`.
    /// Symmetric to [`Self::scan_neighbors_out`].
    fn scan_neighbors_in(
        &self,
        txn: &Transaction,
        edge_type: &str,
        tgt: NodeId,
    ) -> StoreResult<Vec<NodeId>>;

    /// Per-version write of edge properties for `(edge_type, src, tgt)`
    /// (ADR-027 temporal edges). Stores `props` under the temporal
    /// edgeprop key suffixed with `valid_from_ms`; multiple versions
    /// coexist. Adjacency entries are written too (same merge as the
    /// non-temporal path) so the edge is visible to neighbour scans.
    fn put_edge_temporal(
        &self,
        txn: &mut Transaction,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        valid_from_ms: i64,
        props: &EdgeProperties,
    ) -> StoreResult<()>;

    /// Read the edge-property version active at `at_ms`: the version
    /// whose `valid_from_ms <= at_ms` is largest. `None` if none exists.
    fn get_props_at(
        &self,
        txn: &Transaction,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        at_ms: i64,
    ) -> StoreResult<Option<EdgeProperties>>;

    /// All temporal versions of `(edge_type, src, tgt)`, sorted by
    /// `valid_from_ms` ascending.
    fn scan_edge_versions(
        &self,
        txn: &Transaction,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
    ) -> StoreResult<Vec<(i64, EdgeProperties)>>;

    /// Tombstone one specific temporal version. Idempotent on a missing
    /// version. Adjacency entries are NOT touched — removing the last
    /// version of an edge needs the adj-versioning ADR.
    fn delete_edge_temporal(
        &self,
        txn: &mut Transaction,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        valid_from_ms: i64,
    ) -> StoreResult<()>;

    // ── Query-execution edge-property API (executor `Vec<(field_id, Value)>`
    // shape) ──────────────────────────────────────────────────────────────
    // The query layer reads/writes edge properties as a flat
    // `Vec<(interned_field_id, Value)>` (no HashMap), matching the on-disk
    // codec (ADR-040). `valid_from_ms: Some(vf)` selects the per-version
    // temporal key; `None` the non-temporal key. These own both key encoding
    // and the value codec (Layer-4 responsibility).

    /// OCC-tracked raw edge-property read: encode the key (temporal or not),
    /// read the raw bytes joining the read-set in MVCC mode (a property a
    /// transaction reads must be conflict-checked). Returns the raw bytes —
    /// the query layer decodes them (it owns the diagnostic error contract),
    /// while the store owns key encoding.
    fn get_props_raw_tracked(
        &self,
        txn: &mut Transaction,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        valid_from_ms: Option<i64>,
    ) -> StoreResult<Option<Vec<u8>>>;

    /// Buffer an edge-property write in executor shape (encodes value + key).
    fn put_props(
        &self,
        txn: &mut Transaction,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        valid_from_ms: Option<i64>,
        props: &[(u32, Value)],
    ) -> StoreResult<()>;

    /// Tombstone an edge-property body (non-temporal or per-version).
    fn delete_props(
        &self,
        txn: &mut Transaction,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        valid_from_ms: Option<i64>,
    ) -> StoreResult<()>;

    /// Physically move a non-temporal edge-property body from one endpoint pair
    /// to another (read old bytes → write at new key → tombstone old). No-op
    /// when the old key has no body. Used by edge-rewiring (MERGE NODES /
    /// TRANSFER EDGES) — the body moves verbatim, no decode/re-encode.
    fn move_props(
        &self,
        txn: &mut Transaction,
        edge_type: &str,
        old_src: NodeId,
        old_tgt: NodeId,
        new_src: NodeId,
        new_tgt: NodeId,
    ) -> StoreResult<()>;

    /// Stateless parallel-read of an edge-property body: encode the key and
    /// read raw bytes from the engine at `snapshot` (or latest when `None`),
    /// no [`Transaction`]. Returns the key alongside the bytes so a parallel
    /// worker records it in its own OCC accumulator (mirror of
    /// [`crate::NodeStore::read_at_snapshot`] — Variant A snapshot read).
    fn edgeprop_at_snapshot(
        &self,
        engine: &StorageEngine,
        snapshot: Option<lsm_tree::SeqNo>,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
    ) -> StoreResult<(Vec<u8>, Option<Vec<u8>>)>;

    /// Count the live temporal edge-property versions of `(edge_type, src, tgt)`:
    /// snapshot versions minus those tombstoned in this transaction's write
    /// buffer. Used post-delete to decide whether the adjacency posting (which
    /// tracks pair existence, not version count) can be cleared.
    fn count_live_versions(
        &self,
        txn: &mut Transaction,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
    ) -> StoreResult<usize>;

    /// Tombstone every temporal edge-property version of `(edge_type, src, tgt)`
    /// (DETACH / edge-delete on a temporal edge type, which keeps one row per
    /// `valid_from`).
    fn delete_all_versions(
        &self,
        txn: &mut Transaction,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
    ) -> StoreResult<()>;

    /// Move every temporal edge-property version from one endpoint pair to
    /// another, preserving each version's `valid_from` (edge-rewiring on a
    /// temporal edge type). A version already present at the new key is kept
    /// (same-`valid_from` duplicates collapse); the old version is tombstoned.
    fn transfer_all_versions(
        &self,
        txn: &mut Transaction,
        edge_type: &str,
        old_src: NodeId,
        old_tgt: NodeId,
        new_src: NodeId,
        new_tgt: NodeId,
    ) -> StoreResult<()>;

    /// OCC-tracked scan of all temporal edge-property versions of
    /// `(edge_type, src, tgt)`, optionally bounded to `valid_from <= upper_ms`.
    /// Returns `(valid_from, raw_props_bytes)` pairs in key order — the store
    /// decodes the per-version key (no error contract on key decode) and hands
    /// back the property bytes for the query layer to decode (it owns the
    /// diagnostic contract). Joins the read-set in MVCC mode.
    fn scan_versions_raw_tracked(
        &self,
        txn: &mut Transaction,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        upper_ms: Option<i64>,
    ) -> StoreResult<Vec<(i64, Vec<u8>)>>;

    // ── Query-execution adjacency API (typed, posting-list returning) ──────
    // The query traversal reads posting lists (efficient set ops) and buffers
    // commutative merge operands by `(edge_type, node)`; the store owns
    // adj-key encoding. Adjacency stays off the OCC path (commutative merges).

    /// Write the forward-adjacency key for `(edge_type, src)` into a reusable
    /// buffer (clears it first). The traversal hot path reuses one buffer
    /// across edge types to avoid a per-step allocation on super-nodes; this
    /// keeps the key encoding in Layer 4 without giving up that reuse.
    fn write_fwd_key(&self, buf: &mut Vec<u8>, edge_type: &str, src: NodeId);
    /// Write the reverse-adjacency key for `(edge_type, tgt)` into a buffer.
    fn write_rev_key(&self, buf: &mut Vec<u8>, edge_type: &str, tgt: NodeId);

    /// Posting read with read-your-own-writes for a pre-built adjacency key
    /// (paired with [`Self::write_fwd_key`] / [`Self::write_rev_key`]).
    /// `None` when the list is empty.
    fn posting_for_key(
        &self,
        txn: &Transaction,
        adj_key: &[u8],
    ) -> StoreResult<Option<PostingList>>;

    /// Forward-adjacency posting read with read-your-own-writes (`src`'s
    /// out-neighbours for `edge_type`). `None` when the list is empty.
    fn posting_fwd(
        &self,
        txn: &Transaction,
        edge_type: &str,
        src: NodeId,
    ) -> StoreResult<Option<PostingList>>;

    /// Reverse-adjacency posting read (`tgt`'s in-neighbours).
    fn posting_rev(
        &self,
        txn: &Transaction,
        edge_type: &str,
        tgt: NodeId,
    ) -> StoreResult<Option<PostingList>>;

    /// Stateless posting read for a pre-built adjacency key, straight from the
    /// engine at `snapshot` (latest when `None`), with no [`Transaction`].
    /// `None` only when the adjacency key is absent; an existing key with an
    /// empty list returns `Some(empty)` (unlike the RYOW [`Self::posting_fwd`]
    /// which collapses empty to `None`) so a background scanner can tell
    /// "no key" from "key present but drained" and clean up the latter. For
    /// background walks (TTL reaper) with no MVCC transaction; mirrors the
    /// [`crate::NodeStore::read_at_snapshot`] Variant A read model.
    fn posting_at_snapshot(
        &self,
        engine: &StorageEngine,
        snapshot: Option<lsm_tree::SeqNo>,
        adj_key: &[u8],
    ) -> StoreResult<Option<PostingList>>;

    /// Buffer a forward-adjacency add (`src` gains out-neighbour `uid`).
    fn merge_add_fwd(&self, txn: &mut Transaction, edge_type: &str, src: NodeId, uid: u64);
    /// Buffer a reverse-adjacency add (`tgt` gains in-neighbour `uid`).
    fn merge_add_rev(&self, txn: &mut Transaction, edge_type: &str, tgt: NodeId, uid: u64);
    /// Buffer a forward-adjacency remove (`src` loses out-neighbour `uid`).
    fn merge_remove_fwd(&self, txn: &mut Transaction, edge_type: &str, src: NodeId, uid: u64);
    /// Buffer a reverse-adjacency remove (`tgt` loses in-neighbour `uid`).
    fn merge_remove_rev(&self, txn: &mut Transaction, edge_type: &str, tgt: NodeId, uid: u64);

    /// Buffered wholesale delete (tombstone) of a node's forward posting list.
    fn delete_adj_fwd(
        &self,
        txn: &mut Transaction,
        edge_type: &str,
        src: NodeId,
    ) -> StoreResult<()>;
    /// Buffered wholesale delete of a node's reverse posting list.
    fn delete_adj_rev(
        &self,
        txn: &mut Transaction,
        edge_type: &str,
        tgt: NodeId,
    ) -> StoreResult<()>;

    /// Buffered purge of a node's posting list on a direction (`forward`),
    /// also dropping any pending merge operands for that key so a buffered
    /// edge cannot resurrect the posting list a node-delete cascade tombstoned.
    fn purge_adj(
        &self,
        txn: &mut Transaction,
        edge_type: &str,
        node: NodeId,
        forward: bool,
    ) -> StoreResult<()>;
}

/// CE single-shard implementation of [`EdgeStore`]. Stateless — all
/// storage access flows through the [`Transaction`] passed to each
/// method (ADR-041).
pub struct LocalEdgeStore;

impl LocalEdgeStore {
    fn decode_err(e: impl core::fmt::Display) -> StoreError {
        StoreError::Decode {
            kind: "posting list",
            message: format!("{e}"),
        }
    }

    /// Read a posting list at `adj_key` with read-your-own-writes:
    /// buffered point overlay (tombstone/put), else the committed base
    /// posting list (snapshot-aware), then the transaction's pending
    /// adjacency merge operands.
    fn read_posting_list(txn: &Transaction, adj_key: &[u8]) -> StoreResult<PostingList> {
        // Buffered point overlay wins over on-disk state; probe only
        // when the write buffer holds something (skips a key allocation
        // on the read-only hot path).
        let buffered = if txn.write_buffer_is_empty() {
            None
        } else {
            txn.buffered(Partition::Adj, adj_key)
        };
        let mut plist = match buffered {
            Some(None) => PostingList::new(),
            Some(Some(bytes)) => PostingList::from_bytes(bytes).map_err(Self::decode_err)?,
            None => match txn.adj_base_get(adj_key)? {
                Some(b) => PostingList::from_bytes(&b).map_err(Self::decode_err)?,
                None => PostingList::new(),
            },
        };
        if let Some(adds) = txn.merge_adj_adds().get(adj_key) {
            for &uid in adds {
                plist.insert(uid);
            }
        }
        if let Some(removes) = txn.merge_adj_removes().get(adj_key) {
            for &uid in removes {
                plist.remove(uid);
            }
        }
        Ok(plist)
    }

    fn read_posting(txn: &Transaction, adj_key: &[u8]) -> StoreResult<Vec<NodeId>> {
        Ok(Self::read_posting_list(txn, adj_key)?
            .iter()
            .map(NodeId::from_raw)
            .collect())
    }

    /// Encode the edge-property key for `(edge_type, src, tgt)`, selecting the
    /// per-version temporal key when `valid_from_ms` is set.
    fn edgeprop_key(
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        valid_from_ms: Option<i64>,
    ) -> Vec<u8> {
        match valid_from_ms {
            Some(vf) => encode_temporal_edgeprop_key(edge_type, src, tgt, vf),
            None => encode_edgeprop_key(edge_type, src, tgt),
        }
    }

    fn encode_props(props: &EdgeProperties) -> StoreResult<Vec<u8>> {
        props.to_msgpack().map_err(|e| StoreError::Decode {
            kind: "edge properties",
            message: format!("encode: {e}"),
        })
    }

    fn decode_props(bytes: &[u8]) -> StoreResult<EdgeProperties> {
        EdgeProperties::from_msgpack(bytes).map_err(|e| StoreError::Decode {
            kind: "edge properties",
            message: format!("decode: {e}"),
        })
    }
}

impl EdgeStore for LocalEdgeStore {
    fn put_edge(
        &self,
        txn: &mut Transaction,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        props: Option<&EdgeProperties>,
    ) -> StoreResult<()> {
        let fwd_key = encode_adj_key_forward(edge_type, src);
        let rev_key = encode_adj_key_reverse(edge_type, tgt);
        txn.merge_adj_add(&fwd_key, tgt.as_raw());
        txn.merge_adj_add(&rev_key, src.as_raw());
        if let Some(p) = props {
            if !p.is_empty() {
                let body = Self::encode_props(p)?;
                txn.put(
                    Partition::EdgeProp,
                    &encode_edgeprop_key(edge_type, src, tgt),
                    &body,
                )?;
            }
        }
        Ok(())
    }

    fn get_props(
        &self,
        txn: &Transaction,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
    ) -> StoreResult<Option<EdgeProperties>> {
        let key = encode_edgeprop_key(edge_type, src, tgt);
        match txn.read_untracked(Partition::EdgeProp, &key)? {
            Some(bytes) => Self::decode_props(&bytes).map(Some),
            None => Ok(None),
        }
    }

    fn delete_edge(
        &self,
        txn: &mut Transaction,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
    ) -> StoreResult<()> {
        let fwd_key = encode_adj_key_forward(edge_type, src);
        let rev_key = encode_adj_key_reverse(edge_type, tgt);
        txn.merge_adj_remove(&fwd_key, tgt.as_raw());
        txn.merge_adj_remove(&rev_key, src.as_raw());
        txn.delete(
            Partition::EdgeProp,
            &encode_edgeprop_key(edge_type, src, tgt),
        )?;
        Ok(())
    }

    fn scan_neighbors_out(
        &self,
        txn: &Transaction,
        edge_type: &str,
        src: NodeId,
    ) -> StoreResult<Vec<NodeId>> {
        Self::read_posting(txn, &encode_adj_key_forward(edge_type, src))
    }

    fn scan_neighbors_in(
        &self,
        txn: &Transaction,
        edge_type: &str,
        tgt: NodeId,
    ) -> StoreResult<Vec<NodeId>> {
        Self::read_posting(txn, &encode_adj_key_reverse(edge_type, tgt))
    }

    fn put_edge_temporal(
        &self,
        txn: &mut Transaction,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        valid_from_ms: i64,
        props: &EdgeProperties,
    ) -> StoreResult<()> {
        let fwd_key = encode_adj_key_forward(edge_type, src);
        let rev_key = encode_adj_key_reverse(edge_type, tgt);
        let ep_key = encode_temporal_edgeprop_key(edge_type, src, tgt, valid_from_ms);
        let body = Self::encode_props(props)?;
        txn.merge_adj_add(&fwd_key, tgt.as_raw());
        txn.merge_adj_add(&rev_key, src.as_raw());
        txn.put(Partition::EdgeProp, &ep_key, &body)?;
        Ok(())
    }

    fn get_props_at(
        &self,
        txn: &Transaction,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        at_ms: i64,
    ) -> StoreResult<Option<EdgeProperties>> {
        let prefix = temporal_edgeprop_pair_prefix(edge_type, src, tgt);
        let mut best: Option<(i64, EdgeProperties)> = None;
        for (key, value) in txn.base_prefix_scan(Partition::EdgeProp, &prefix)? {
            let Some((_, _, _, valid_from)) = decode_temporal_edgeprop_key(&key) else {
                continue;
            };
            if valid_from > at_ms {
                continue;
            }
            let props = Self::decode_props(&value)?;
            best = match best {
                Some((vf, _)) if vf >= valid_from => best,
                _ => Some((valid_from, props)),
            };
        }
        Ok(best.map(|(_, p)| p))
    }

    fn scan_edge_versions(
        &self,
        txn: &Transaction,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
    ) -> StoreResult<Vec<(i64, EdgeProperties)>> {
        let prefix = temporal_edgeprop_pair_prefix(edge_type, src, tgt);
        let mut out = Vec::new();
        for (key, value) in txn.base_prefix_scan(Partition::EdgeProp, &prefix)? {
            let Some((_, _, _, valid_from)) = decode_temporal_edgeprop_key(&key) else {
                continue;
            };
            out.push((valid_from, Self::decode_props(&value)?));
        }
        out.sort_by_key(|(vf, _)| *vf);
        Ok(out)
    }

    fn delete_edge_temporal(
        &self,
        txn: &mut Transaction,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        valid_from_ms: i64,
    ) -> StoreResult<()> {
        let key = encode_temporal_edgeprop_key(edge_type, src, tgt, valid_from_ms);
        txn.delete(Partition::EdgeProp, &key)?;
        Ok(())
    }

    fn get_props_raw_tracked(
        &self,
        txn: &mut Transaction,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        valid_from_ms: Option<i64>,
    ) -> StoreResult<Option<Vec<u8>>> {
        let key = Self::edgeprop_key(edge_type, src, tgt, valid_from_ms);
        Ok(txn.get(Partition::EdgeProp, &key)?)
    }

    fn put_props(
        &self,
        txn: &mut Transaction,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        valid_from_ms: Option<i64>,
        props: &[(u32, Value)],
    ) -> StoreResult<()> {
        let key = Self::edgeprop_key(edge_type, src, tgt, valid_from_ms);
        let bytes = encode_edge_props(props).map_err(|e| StoreError::Decode {
            kind: "edge properties",
            message: format!("encode: {e}"),
        })?;
        txn.put(Partition::EdgeProp, &key, &bytes)?;
        Ok(())
    }

    fn delete_props(
        &self,
        txn: &mut Transaction,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        valid_from_ms: Option<i64>,
    ) -> StoreResult<()> {
        let key = Self::edgeprop_key(edge_type, src, tgt, valid_from_ms);
        txn.delete(Partition::EdgeProp, &key)?;
        Ok(())
    }

    fn move_props(
        &self,
        txn: &mut Transaction,
        edge_type: &str,
        old_src: NodeId,
        old_tgt: NodeId,
        new_src: NodeId,
        new_tgt: NodeId,
    ) -> StoreResult<()> {
        let old_key = encode_edgeprop_key(edge_type, old_src, old_tgt);
        if let Some(bytes) = txn.get(Partition::EdgeProp, &old_key)? {
            let new_key = encode_edgeprop_key(edge_type, new_src, new_tgt);
            txn.put(Partition::EdgeProp, &new_key, &bytes)?;
            txn.delete(Partition::EdgeProp, &old_key)?;
        }
        Ok(())
    }

    fn edgeprop_at_snapshot(
        &self,
        engine: &StorageEngine,
        snapshot: Option<lsm_tree::SeqNo>,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
    ) -> StoreResult<(Vec<u8>, Option<Vec<u8>>)> {
        let key = encode_edgeprop_key(edge_type, src, tgt);
        let bytes = match snapshot {
            Some(snap) => engine
                .snapshot_get(&snap, Partition::EdgeProp, &key)?
                .map(|b| b.to_vec()),
            None => engine.get(Partition::EdgeProp, &key)?.map(|b| b.to_vec()),
        };
        Ok((key, bytes))
    }

    fn count_live_versions(
        &self,
        txn: &mut Transaction,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
    ) -> StoreResult<usize> {
        let prefix = temporal_edgeprop_pair_prefix(edge_type, src, tgt);
        let scan = txn.prefix_scan(Partition::EdgeProp, &prefix)?;
        let mut remaining = 0_usize;
        for (key, _) in scan {
            // A version tombstoned in this transaction's write buffer is no
            // longer live even though the snapshot still lists it.
            if !matches!(txn.buffered(Partition::EdgeProp, &key), Some(None)) {
                remaining += 1;
            }
        }
        Ok(remaining)
    }

    fn delete_all_versions(
        &self,
        txn: &mut Transaction,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
    ) -> StoreResult<()> {
        let versions = self.scan_versions_raw_tracked(txn, edge_type, src, tgt, None)?;
        for (vf, _) in versions {
            let key = encode_temporal_edgeprop_key(edge_type, src, tgt, vf);
            txn.delete(Partition::EdgeProp, &key)?;
        }
        Ok(())
    }

    fn transfer_all_versions(
        &self,
        txn: &mut Transaction,
        edge_type: &str,
        old_src: NodeId,
        old_tgt: NodeId,
        new_src: NodeId,
        new_tgt: NodeId,
    ) -> StoreResult<()> {
        let versions = self.scan_versions_raw_tracked(txn, edge_type, old_src, old_tgt, None)?;
        for (vf, bytes) in versions {
            let new_key = encode_temporal_edgeprop_key(edge_type, new_src, new_tgt, vf);
            // Same-version duplicate at the new key collapses to the existing one.
            if txn.get(Partition::EdgeProp, &new_key)?.is_none() {
                txn.put(Partition::EdgeProp, &new_key, &bytes)?;
            }
            let old_key = encode_temporal_edgeprop_key(edge_type, old_src, old_tgt, vf);
            txn.delete(Partition::EdgeProp, &old_key)?;
        }
        Ok(())
    }

    fn scan_versions_raw_tracked(
        &self,
        txn: &mut Transaction,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        upper_ms: Option<i64>,
    ) -> StoreResult<Vec<(i64, Vec<u8>)>> {
        let prefix = temporal_edgeprop_pair_prefix(edge_type, src, tgt);
        let scanned = txn.prefix_scan(Partition::EdgeProp, &prefix)?;
        let bound = upper_ms.map(|ms| valid_from_upper_bound_key(edge_type, src, tgt, ms));
        let mut out = Vec::new();
        for (key, bytes) in scanned {
            if let Some(ref b) = bound {
                if &key >= b {
                    continue;
                }
            }
            if let Some((_, _, _, vf)) = decode_temporal_edgeprop_key(&key) {
                out.push((vf, bytes));
            }
        }
        Ok(out)
    }

    fn write_fwd_key(&self, buf: &mut Vec<u8>, edge_type: &str, src: NodeId) {
        write_adj_key_forward(edge_type, src, buf);
    }

    fn write_rev_key(&self, buf: &mut Vec<u8>, edge_type: &str, tgt: NodeId) {
        write_adj_key_reverse(edge_type, tgt, buf);
    }

    fn posting_for_key(
        &self,
        txn: &Transaction,
        adj_key: &[u8],
    ) -> StoreResult<Option<PostingList>> {
        let plist = Self::read_posting_list(txn, adj_key)?;
        Ok((!plist.is_empty()).then_some(plist))
    }

    fn posting_fwd(
        &self,
        txn: &Transaction,
        edge_type: &str,
        src: NodeId,
    ) -> StoreResult<Option<PostingList>> {
        let plist = Self::read_posting_list(txn, &encode_adj_key_forward(edge_type, src))?;
        Ok((!plist.is_empty()).then_some(plist))
    }

    fn posting_rev(
        &self,
        txn: &Transaction,
        edge_type: &str,
        tgt: NodeId,
    ) -> StoreResult<Option<PostingList>> {
        let plist = Self::read_posting_list(txn, &encode_adj_key_reverse(edge_type, tgt))?;
        Ok((!plist.is_empty()).then_some(plist))
    }

    fn posting_at_snapshot(
        &self,
        engine: &StorageEngine,
        snapshot: Option<lsm_tree::SeqNo>,
        adj_key: &[u8],
    ) -> StoreResult<Option<PostingList>> {
        let bytes = match snapshot {
            Some(snap) => engine.snapshot_get(&snap, Partition::Adj, adj_key)?,
            None => engine.get(Partition::Adj, adj_key)?,
        };
        match bytes {
            Some(b) => Ok(Some(PostingList::from_bytes(&b).map_err(Self::decode_err)?)),
            None => Ok(None),
        }
    }

    fn merge_add_fwd(&self, txn: &mut Transaction, edge_type: &str, src: NodeId, uid: u64) {
        txn.merge_adj_add(&encode_adj_key_forward(edge_type, src), uid);
    }

    fn merge_add_rev(&self, txn: &mut Transaction, edge_type: &str, tgt: NodeId, uid: u64) {
        txn.merge_adj_add(&encode_adj_key_reverse(edge_type, tgt), uid);
    }

    fn merge_remove_fwd(&self, txn: &mut Transaction, edge_type: &str, src: NodeId, uid: u64) {
        txn.merge_adj_remove(&encode_adj_key_forward(edge_type, src), uid);
    }

    fn merge_remove_rev(&self, txn: &mut Transaction, edge_type: &str, tgt: NodeId, uid: u64) {
        txn.merge_adj_remove(&encode_adj_key_reverse(edge_type, tgt), uid);
    }

    fn delete_adj_fwd(
        &self,
        txn: &mut Transaction,
        edge_type: &str,
        src: NodeId,
    ) -> StoreResult<()> {
        txn.delete(Partition::Adj, &encode_adj_key_forward(edge_type, src))?;
        Ok(())
    }

    fn delete_adj_rev(
        &self,
        txn: &mut Transaction,
        edge_type: &str,
        tgt: NodeId,
    ) -> StoreResult<()> {
        txn.delete(Partition::Adj, &encode_adj_key_reverse(edge_type, tgt))?;
        Ok(())
    }

    fn purge_adj(
        &self,
        txn: &mut Transaction,
        edge_type: &str,
        node: NodeId,
        forward: bool,
    ) -> StoreResult<()> {
        let key = if forward {
            encode_adj_key_forward(edge_type, node)
        } else {
            encode_adj_key_reverse(edge_type, node)
        };
        txn.delete(Partition::Adj, &key)?;
        txn.drop_adj_merges(&key);
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
    use coordinode_core::txn::write_concern::WriteConcern;
    use coordinode_storage::engine::core::StorageEngine;
    use coordinode_storage::engine::transaction::CommitContext;

    /// MVCC test database: writes buffer on the transaction and apply at
    /// [`commit`]; reads open a fresh transaction at the latest snapshot.
    /// The engine seqno space drives snapshot visibility, so a standalone
    /// oracle (for `commit_ts` / `read_ts`) over the shared logic engine
    /// is sufficient — no oracle-wired engine needed because every read
    /// pins `engine.snapshot()` (latest committed), not `read_ts`.
    struct TestDb {
        _fx: coordinode_test_fixtures::EngineFixture,
        engine: Arc<StorageEngine>,
        oracle: Arc<TimestampOracle>,
    }

    fn open() -> TestDb {
        let fx = coordinode_test_fixtures::engine_for_logic();
        let engine = Arc::clone(&fx.engine);
        TestDb {
            _fx: fx,
            engine,
            oracle: Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(1))),
        }
    }

    /// Open an MVCC transaction pinned at a fresh read timestamp and the
    /// latest committed snapshot.
    fn mvcc_txn<'a>(engine: &'a StorageEngine, oracle: &'a TimestampOracle) -> Transaction<'a> {
        let read_ts = oracle.next();
        let snap = engine.snapshot();
        Transaction::new(engine, Some(oracle), read_ts, Some(snap))
    }

    fn commit(t: &mut Transaction) {
        let wc = WriteConcern::majority();
        let ctx = CommitContext {
            write_concern: &wc,
            pipeline: None,
            id_gen: None,
            drain_buffer: None,
            nvme_write_buffer: None,
        };
        t.commit(&ctx).expect("commit");
    }

    impl TestDb {
        /// Run writes in a fresh MVCC transaction and commit them.
        fn write(&self, f: impl FnOnce(&LocalEdgeStore, &mut Transaction)) {
            let mut t = mvcc_txn(&self.engine, &self.oracle);
            let store = LocalEdgeStore;
            f(&store, &mut t);
            commit(&mut t);
        }

        /// Fresh read-only MVCC transaction at the latest committed state.
        fn read(&self) -> Transaction<'_> {
            mvcc_txn(&self.engine, &self.oracle)
        }
    }

    fn props_with(field: u32, value: i64) -> EdgeProperties {
        let mut p = EdgeProperties::new();
        p.set(field, coordinode_core::graph::types::Value::Int(value));
        p
    }

    #[test]
    fn put_edge_creates_forward_and_reverse_adj() {
        let db = open();
        let alice = NodeId::from_raw(1);
        let bob = NodeId::from_raw(2);
        db.write(|s, t| s.put_edge(t, "KNOWS", alice, bob, None).expect("put"));

        let store = LocalEdgeStore;
        let r = db.read();
        assert_eq!(
            store.scan_neighbors_out(&r, "KNOWS", alice).expect("scan"),
            vec![bob],
        );
        assert_eq!(
            store.scan_neighbors_in(&r, "KNOWS", bob).expect("scan"),
            vec![alice],
        );
    }

    #[test]
    fn put_edge_with_props_stores_property_body() {
        let db = open();
        let a = NodeId::from_raw(10);
        let b = NodeId::from_raw(20);
        db.write(|s, t| {
            let mut props = EdgeProperties::new();
            props.set(7, coordinode_core::graph::types::Value::Int(42));
            s.put_edge(t, "OWNS", a, b, Some(&props)).expect("put");
        });

        let store = LocalEdgeStore;
        let r = db.read();
        let loaded = store
            .get_props(&r, "OWNS", a, b)
            .expect("ok")
            .expect("Some");
        assert_eq!(loaded.len(), 1);
    }

    #[test]
    fn edge_without_props_returns_none_from_get_props() {
        // Property-less edge: adj entries exist, edgeprop body doesn't.
        // get_props returns None (NOT "edge does not exist").
        let db = open();
        let a = NodeId::from_raw(1);
        let b = NodeId::from_raw(2);
        db.write(|s, t| s.put_edge(t, "LIKES", a, b, None).expect("put"));
        let store = LocalEdgeStore;
        let r = db.read();
        assert!(store.get_props(&r, "LIKES", a, b).expect("ok").is_none());
    }

    #[test]
    fn multiple_neighbors_listed_in_sorted_order() {
        // The posting list maintains sorted order; scan returns it
        // unchanged. Verified across multi-merge writes.
        let db = open();
        let src = NodeId::from_raw(1);
        db.write(|s, t| {
            // Insert out of order to exercise the merge operator's sort.
            for tgt in [5u64, 3, 8, 1, 2] {
                s.put_edge(t, "F", src, NodeId::from_raw(tgt), None)
                    .expect("put");
            }
        });
        let store = LocalEdgeStore;
        let r = db.read();
        let neighbors: Vec<u64> = store
            .scan_neighbors_out(&r, "F", src)
            .expect("scan")
            .iter()
            .map(|n| n.as_raw())
            .collect();
        assert_eq!(neighbors, vec![1, 2, 3, 5, 8]);
    }

    #[test]
    fn delete_edge_removes_from_adjacency_and_props() {
        let db = open();
        let a = NodeId::from_raw(1);
        let b = NodeId::from_raw(2);
        let c = NodeId::from_raw(3);
        // Two outgoing edges from `a`. Delete the (a,b) edge.
        db.write(|s, t| {
            s.put_edge(t, "F", a, b, None).expect("put");
            s.put_edge(t, "F", a, c, None).expect("put");
        });
        db.write(|s, t| s.delete_edge(t, "F", a, b).expect("delete"));

        let store = LocalEdgeStore;
        let r = db.read();
        // `b` is gone from a's forward neighbors; `c` remains.
        let out: Vec<u64> = store
            .scan_neighbors_out(&r, "F", a)
            .expect("scan")
            .iter()
            .map(|n| n.as_raw())
            .collect();
        assert_eq!(out, vec![3]);
        // `a` is gone from b's reverse neighbors.
        assert!(store
            .scan_neighbors_in(&r, "F", b)
            .expect("scan")
            .is_empty());
    }

    #[test]
    fn delete_edge_is_idempotent() {
        let db = open();
        // Never created — delete must still succeed (merge remove on an
        // absent uid is a no-op; edgeprop delete on a missing key is fine).
        db.write(|s, t| {
            s.delete_edge(t, "F", NodeId::from_raw(9), NodeId::from_raw(10))
                .expect("delete missing");
        });
    }

    #[test]
    fn edge_types_are_isolated_in_adj() {
        // Same (src, tgt) pair under different edge types must NOT share
        // adjacency entries. Different keyspaces.
        let db = open();
        let a = NodeId::from_raw(1);
        let b = NodeId::from_raw(2);
        db.write(|s, t| {
            s.put_edge(t, "KNOWS", a, b, None).expect("put");
            s.put_edge(t, "LIKES", a, b, None).expect("put");
        });
        db.write(|s, t| s.delete_edge(t, "KNOWS", a, b).expect("delete KNOWS"));

        let store = LocalEdgeStore;
        let r = db.read();
        // KNOWS gone, LIKES preserved.
        assert!(store
            .scan_neighbors_out(&r, "KNOWS", a)
            .expect("scan")
            .is_empty());
        assert_eq!(
            store.scan_neighbors_out(&r, "LIKES", a).expect("scan"),
            vec![b],
        );
    }

    #[test]
    fn put_temporal_versions_round_trip_via_scan() {
        let db = open();
        let a = NodeId::from_raw(1);
        let b = NodeId::from_raw(2);
        db.write(|s, t| {
            for (vf, salary) in [(1000i64, 50_000), (2000, 60_000), (3000, 70_000)] {
                s.put_edge_temporal(t, "WORKS_AT", a, b, vf, &props_with(1, salary))
                    .expect("put temporal");
            }
        });
        let store = LocalEdgeStore;
        let r = db.read();
        let versions = store
            .scan_edge_versions(&r, "WORKS_AT", a, b)
            .expect("scan versions");
        assert_eq!(versions.len(), 3);
        assert_eq!(versions[0].0, 1000);
        assert_eq!(versions[2].0, 3000);
    }

    #[test]
    fn get_props_at_returns_largest_valid_from_le_query() {
        let db = open();
        let a = NodeId::from_raw(1);
        let b = NodeId::from_raw(2);
        db.write(|s, t| {
            s.put_edge_temporal(t, "E", a, b, 1000, &props_with(1, 10))
                .unwrap();
            s.put_edge_temporal(t, "E", a, b, 2000, &props_with(1, 20))
                .unwrap();
            s.put_edge_temporal(t, "E", a, b, 3000, &props_with(1, 30))
                .unwrap();
        });
        let store = LocalEdgeStore;
        let r = db.read();

        // At 1500 — only the 1000-version is visible.
        let p = store
            .get_props_at(&r, "E", a, b, 1500)
            .unwrap()
            .expect("present");
        assert_eq!(
            p.get(1),
            Some(&coordinode_core::graph::types::Value::Int(10))
        );

        // At 2500 — pick the 2000-version (largest <= 2500).
        let p = store
            .get_props_at(&r, "E", a, b, 2500)
            .unwrap()
            .expect("present");
        assert_eq!(
            p.get(1),
            Some(&coordinode_core::graph::types::Value::Int(20))
        );

        // At 3000 — boundary inclusive: pick the 3000-version.
        let p = store
            .get_props_at(&r, "E", a, b, 3000)
            .unwrap()
            .expect("present");
        assert_eq!(
            p.get(1),
            Some(&coordinode_core::graph::types::Value::Int(30))
        );
    }

    #[test]
    fn get_props_at_before_first_version_returns_none() {
        let db = open();
        let a = NodeId::from_raw(1);
        let b = NodeId::from_raw(2);
        db.write(|s, t| {
            s.put_edge_temporal(t, "E", a, b, 5000, &props_with(1, 1))
                .unwrap();
        });
        let store = LocalEdgeStore;
        let r = db.read();
        assert!(store.get_props_at(&r, "E", a, b, 1000).unwrap().is_none());
    }

    #[test]
    fn delete_temporal_version_removes_only_that_version() {
        let db = open();
        let a = NodeId::from_raw(1);
        let b = NodeId::from_raw(2);
        db.write(|s, t| {
            s.put_edge_temporal(t, "E", a, b, 100, &props_with(1, 10))
                .unwrap();
            s.put_edge_temporal(t, "E", a, b, 200, &props_with(1, 20))
                .unwrap();
        });
        db.write(|s, t| s.delete_edge_temporal(t, "E", a, b, 100).unwrap());
        let store = LocalEdgeStore;
        let r = db.read();
        let versions = store.scan_edge_versions(&r, "E", a, b).unwrap();
        assert_eq!(versions.len(), 1);
        assert_eq!(versions[0].0, 200);
    }

    #[test]
    fn delete_temporal_is_idempotent() {
        let db = open();
        db.write(|s, t| {
            s.delete_edge_temporal(t, "E", NodeId::from_raw(1), NodeId::from_raw(2), 9999)
                .expect("idempotent delete");
        });
    }

    #[test]
    fn temporal_writes_isolated_per_pair() {
        let db = open();
        let a = NodeId::from_raw(1);
        let b = NodeId::from_raw(2);
        let c = NodeId::from_raw(3);
        db.write(|s, t| {
            s.put_edge_temporal(t, "E", a, b, 100, &props_with(1, 1))
                .unwrap();
            s.put_edge_temporal(t, "E", a, c, 200, &props_with(1, 2))
                .unwrap();
        });
        let store = LocalEdgeStore;
        let r = db.read();
        let only_ab = store.scan_edge_versions(&r, "E", a, b).unwrap();
        let only_ac = store.scan_edge_versions(&r, "E", a, c).unwrap();
        assert_eq!(only_ab.len(), 1);
        assert_eq!(only_ab[0].0, 100);
        assert_eq!(only_ac.len(), 1);
        assert_eq!(only_ac[0].0, 200);
    }

    #[test]
    fn temporal_put_also_writes_adjacency() {
        // A temporal edge must be visible in the neighbour scan — the
        // adj merge runs as part of the temporal write.
        let db = open();
        let a = NodeId::from_raw(1);
        let b = NodeId::from_raw(2);
        db.write(|s, t| {
            s.put_edge_temporal(t, "E", a, b, 100, &props_with(1, 1))
                .unwrap();
        });
        let store = LocalEdgeStore;
        let r = db.read();
        assert_eq!(store.scan_neighbors_out(&r, "E", a).unwrap(), vec![b]);
        assert_eq!(store.scan_neighbors_in(&r, "E", b).unwrap(), vec![a]);
    }

    #[test]
    fn concurrent_put_edge_on_super_node_preserves_all_edges() {
        // Concurrent merge-operator stress on a single source. Four
        // threads each add 25 distinct out-edges from the same src, each
        // in its own MVCC transaction. After join + commit, all 100 must
        // be visible — add/remove are commutative+idempotent merge
        // operands (adjacency bypasses OCC).
        use std::thread;

        let db = open();
        let src = NodeId::from_raw(1);
        let threads_n = 4u64;
        let per_thread = 25u64;

        let handles: Vec<_> = (0..threads_n)
            .map(|t| {
                let engine = Arc::clone(&db.engine);
                let oracle = Arc::clone(&db.oracle);
                thread::spawn(move || {
                    let store = LocalEdgeStore;
                    let mut tx = mvcc_txn(&engine, &oracle);
                    for i in 0..per_thread {
                        let tgt = NodeId::from_raw(t * per_thread + i + 100);
                        store.put_edge(&mut tx, "F", src, tgt, None).expect("put");
                    }
                    commit(&mut tx);
                })
            })
            .collect();
        for h in handles {
            h.join().expect("thread join");
        }

        let store = LocalEdgeStore;
        let r = db.read();
        let mut neighbors: Vec<u64> = store
            .scan_neighbors_out(&r, "F", src)
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
        // Concurrent add(x) and remove(x) from different threads — both
        // ops must compose into a consistent posting list.
        use std::thread;

        let db = open();
        let src = NodeId::from_raw(1);

        // Pre-populate 10 edges so the remover has something to remove.
        db.write(|s, t| {
            for i in 0..10u64 {
                s.put_edge(t, "F", src, NodeId::from_raw(i + 200), None)
                    .expect("setup");
            }
        });

        // Adder thread: adds 5 more.
        let engine_a = Arc::clone(&db.engine);
        let oracle_a = Arc::clone(&db.oracle);
        let adder = thread::spawn(move || {
            let store = LocalEdgeStore;
            let mut tx = mvcc_txn(&engine_a, &oracle_a);
            for i in 0..5u64 {
                store
                    .put_edge(&mut tx, "F", src, NodeId::from_raw(i + 300), None)
                    .expect("add");
            }
            commit(&mut tx);
        });
        // Remover thread: removes 5 of the pre-populated.
        let engine_r = Arc::clone(&db.engine);
        let oracle_r = Arc::clone(&db.oracle);
        let remover = thread::spawn(move || {
            let store = LocalEdgeStore;
            let mut tx = mvcc_txn(&engine_r, &oracle_r);
            for i in 0..5u64 {
                store
                    .delete_edge(&mut tx, "F", src, NodeId::from_raw(i + 200))
                    .expect("delete");
            }
            commit(&mut tx);
        });
        adder.join().expect("adder");
        remover.join().expect("remover");

        let store = LocalEdgeStore;
        let r = db.read();
        let mut neighbors: Vec<u64> = store
            .scan_neighbors_out(&r, "F", src)
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
        // Four threads each write a distinct valid_from version of the
        // same (et, src, tgt). After join, scan_edge_versions returns all
        // four, sorted by valid_from.
        use std::thread;

        let db = open();
        let src = NodeId::from_raw(1);
        let tgt = NodeId::from_raw(2);

        let handles: Vec<_> = (0..4u64)
            .map(|t| {
                let engine = Arc::clone(&db.engine);
                let oracle = Arc::clone(&db.oracle);
                thread::spawn(move || {
                    let store = LocalEdgeStore;
                    let mut tx = mvcc_txn(&engine, &oracle);
                    let vf = (t as i64 + 1) * 1000;
                    let mut props = EdgeProperties::new();
                    props.set(1, coordinode_core::graph::types::Value::Int(vf));
                    store
                        .put_edge_temporal(&mut tx, "E", src, tgt, vf, &props)
                        .expect("put temporal");
                    commit(&mut tx);
                })
            })
            .collect();
        for h in handles {
            h.join().expect("join");
        }

        let store = LocalEdgeStore;
        let r = db.read();
        let versions = store.scan_edge_versions(&r, "E", src, tgt).expect("scan");
        let vfs: Vec<i64> = versions.iter().map(|(vf, _)| *vf).collect();
        assert_eq!(vfs, vec![1000, 2000, 3000, 4000]);
    }

    #[test]
    fn corrupt_posting_list_surfaces_as_decode_error() {
        let db = open();
        let a = NodeId::from_raw(1);
        // Inject garbage bytes at the forward-adj key directly.
        db.engine
            .put(
                Partition::Adj,
                &encode_adj_key_forward("F", a),
                &[0xde, 0xad, 0xbe, 0xef],
            )
            .expect("inject");
        let store = LocalEdgeStore;
        let r = db.read();
        let err = store
            .scan_neighbors_out(&r, "F", a)
            .expect_err("must error");
        assert!(matches!(
            err,
            StoreError::Decode {
                kind: "posting list",
                ..
            }
        ));
    }

    #[test]
    fn posting_at_snapshot_reads_committed_adjacency() {
        let db = open();
        let alice = NodeId::from_raw(1);
        let bob = NodeId::from_raw(2);
        let carol = NodeId::from_raw(3);
        db.write(|s, t| {
            s.put_edge(t, "KNOWS", alice, bob, None).expect("put");
            s.put_edge(t, "KNOWS", alice, carol, None).expect("put");
        });

        let store = LocalEdgeStore;
        // Absent key → None.
        assert!(store
            .posting_at_snapshot(&db.engine, None, &encode_adj_key_forward("KNOWS", bob))
            .expect("ok")
            .is_none());
        // Present key → the full peer set (latest committed).
        let fwd = store
            .posting_at_snapshot(&db.engine, None, &encode_adj_key_forward("KNOWS", alice))
            .expect("ok")
            .expect("present");
        let mut peers: Vec<u64> = fwd.iter().collect();
        peers.sort_unstable();
        assert_eq!(peers, vec![2, 3]);
    }

    #[test]
    fn posting_at_snapshot_returns_empty_present_list_not_none() {
        // A present adjacency key whose list is empty must read back as
        // Some(empty), not None — a background scanner relies on this to
        // distinguish "no key" from "key drained" and clean up the latter.
        let db = open();
        let a = NodeId::from_raw(7);
        db.engine
            .put(
                Partition::Adj,
                &encode_adj_key_forward("F", a),
                &PostingList::new().to_bytes().expect("encode empty"),
            )
            .expect("inject empty");
        let got = LocalEdgeStore
            .posting_at_snapshot(&db.engine, None, &encode_adj_key_forward("F", a))
            .expect("ok");
        assert!(got.is_some(), "present empty key reads as Some");
        assert!(got.expect("some").is_empty());
    }
}
