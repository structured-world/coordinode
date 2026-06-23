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
    decode_discriminated_edgeprop_key, decode_temporal_edgeprop_key, encode_adj_key_forward,
    encode_adj_key_reverse, encode_discriminated_edgeprop_key, encode_edge_props,
    encode_edgeprop_key, encode_temporal_edgeprop_key, temporal_edgeprop_pair_prefix,
    valid_from_upper_bound_key, write_adj_key_forward, write_adj_key_reverse, EdgeProperties,
    PostingList,
};
use coordinode_core::graph::node::NodeId;
use coordinode_core::graph::types::Value;
use coordinode_core::schema::definition::PropertyType;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::transaction::Transaction;
use coordinode_storage::engine::StorageSnapshot;

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

    /// Snapshot-aware read of edge properties for `(edge_type, src, tgt)`,
    /// for callers that hold an MVCC [`StorageSnapshot`] rather than a
    /// [`Transaction`] — backup export takes one consistent snapshot up front
    /// and reads every edge through it (ADR-040). Same semantics as
    /// [`Self::get_props`]: `None` when the edge carries no property body.
    fn get_props_snapshot(
        &self,
        engine: &StorageEngine,
        snapshot: &StorageSnapshot,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
    ) -> StoreResult<Option<EdgeProperties>>;

    /// Direct, non-transactional write of the edge property body for
    /// `(edge_type, src, tgt)`, for the backup restore path that applies writes
    /// straight to the engine (ADR-016: the oracle auto-stamps the seqno) rather
    /// than through a [`Transaction`]. Writes ONLY the edgeprop body — adjacency
    /// is restored separately — and encodes through the single canonical
    /// edge-property codec, so restored bytes are identical to a put_edge write.
    /// A typed helper so restore never hand-rolls the edge-prop key.
    fn put_props_direct(
        &self,
        engine: &StorageEngine,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        props: &EdgeProperties,
    ) -> StoreResult<()>;

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

    /// Write one instance of a `DISCRIMINATED BY (col)` edge (ADR-029): the
    /// edgeprop body is keyed by the discriminator value, and the adjacency
    /// posting gets a set-semantics add (a target appears iff at least one
    /// instance exists for the pair). `discriminator` must be an ADR-029
    /// supported type (Int / Timestamp / Float / Bool / String / Blob); a
    /// non-supported value is a [`StoreError::Invariant`] — the schema validates
    /// the column type at DDL time, so this cannot happen for correct callers.
    fn put_edge_discriminated(
        &self,
        txn: &mut Transaction,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        discriminator: &Value,
        props: &EdgeProperties,
    ) -> StoreResult<()>;

    /// Point-read the properties of one discriminated edge instance, keyed by
    /// the exact `discriminator` value. `None` if no such instance exists.
    fn get_props_for(
        &self,
        txn: &Transaction,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        discriminator: &Value,
    ) -> StoreResult<Option<EdgeProperties>>;

    /// Enumerate every discriminated instance of `(edge_type, src, tgt)`,
    /// returning `(discriminator_value, props)` pairs sorted ascending by the
    /// order-preserving key encoding. `discriminator_kind` is the column's
    /// declared [`PropertyType`] (read from the edge type schema by the caller),
    /// needed to decode the discriminator suffix.
    fn scan_discriminators(
        &self,
        txn: &Transaction,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        discriminator_kind: &PropertyType,
    ) -> StoreResult<Vec<(Value, EdgeProperties)>>;

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

    fn get_props_snapshot(
        &self,
        engine: &StorageEngine,
        snapshot: &StorageSnapshot,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
    ) -> StoreResult<Option<EdgeProperties>> {
        let key = encode_edgeprop_key(edge_type, src, tgt);
        match engine.snapshot_get(snapshot, Partition::EdgeProp, &key)? {
            Some(bytes) => Self::decode_props(&bytes).map(Some),
            None => Ok(None),
        }
    }

    fn put_props_direct(
        &self,
        engine: &StorageEngine,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        props: &EdgeProperties,
    ) -> StoreResult<()> {
        let key = encode_edgeprop_key(edge_type, src, tgt);
        let value = Self::encode_props(props)?;
        engine.put(Partition::EdgeProp, &key, &value)?;
        Ok(())
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

    fn put_edge_discriminated(
        &self,
        txn: &mut Transaction,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        discriminator: &Value,
        props: &EdgeProperties,
    ) -> StoreResult<()> {
        let ep_key = encode_discriminated_edgeprop_key(edge_type, src, tgt, discriminator)
            .ok_or_else(|| {
                StoreError::Invariant(format!(
                    "unsupported discriminator value for edge type {edge_type}: {discriminator:?}"
                ))
            })?;
        let body = Self::encode_props(props)?;
        txn.merge_adj_add(&encode_adj_key_forward(edge_type, src), tgt.as_raw());
        txn.merge_adj_add(&encode_adj_key_reverse(edge_type, tgt), src.as_raw());
        txn.put(Partition::EdgeProp, &ep_key, &body)?;
        Ok(())
    }

    fn get_props_for(
        &self,
        txn: &Transaction,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        discriminator: &Value,
    ) -> StoreResult<Option<EdgeProperties>> {
        let key = encode_discriminated_edgeprop_key(edge_type, src, tgt, discriminator)
            .ok_or_else(|| {
                StoreError::Invariant(format!(
                    "unsupported discriminator value for edge type {edge_type}: {discriminator:?}"
                ))
            })?;
        match txn.read_untracked(Partition::EdgeProp, &key)? {
            Some(bytes) => Self::decode_props(&bytes).map(Some),
            None => Ok(None),
        }
    }

    fn scan_discriminators(
        &self,
        txn: &Transaction,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        discriminator_kind: &PropertyType,
    ) -> StoreResult<Vec<(Value, EdgeProperties)>> {
        let prefix = temporal_edgeprop_pair_prefix(edge_type, src, tgt);
        let mut rows: Vec<(Vec<u8>, Value, EdgeProperties)> = Vec::new();
        for (key, value) in txn.base_prefix_scan(Partition::EdgeProp, &prefix)? {
            if let Some((_, _, _, disc)) =
                decode_discriminated_edgeprop_key(&key, discriminator_kind)
            {
                rows.push((key, disc, Self::decode_props(&value)?));
            }
        }
        // Sort by the order-preserving key so callers see discriminators in
        // ascending value order regardless of the scan iterator's ordering.
        rows.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(rows.into_iter().map(|(_, d, p)| (d, p)).collect())
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
mod tests;
