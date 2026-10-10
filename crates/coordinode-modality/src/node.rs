//! Node store — typed read/write of [`NodeRecord`] in
//! [`Partition::Node`], with first-class support for both
//! non-temporal labels (one row per node id) and
//! temporal labels (one row per `valid_from` version).
//!
//! The store hides:
//!
//! - Key format selection between the 16-byte non-temporal key
//!   (`node:<shard><id>`) and the 25-byte temporal key
//!   (`node:<shard><id>:<valid_from>`).
//! - MessagePack encode / decode at the (de)serialization boundary.
//! - Shard-scoped scans (prefix walks over a known shard).
//! - Per-id temporal version walks (prefix walks over a known id).
//!
//! ## Transaction threading
//!
//! Every method takes an explicit `&mut Transaction` (writes) or
//! `&Transaction` (reads). Writes buffer on the transaction and apply
//! at [`Transaction::commit`]; reads go through the transaction's
//! read path (`read_untracked` for points, `base_prefix_scan` /
//! `base_prefix_iter` for prefix walks, `snapshot_get_at` for pinned
//! visibility probes).
//!
//! ## Temporal semantics
//!
//! Temporal labels store one row per version. The "current as-of T"
//! read scans the per-id prefix and picks the row whose
//! `valid_from <= T` is largest. The store exposes this directly via
//! [`NodeStore::get_at`] so callers don't reimplement the seek-and-pick.

use coordinode_core::graph::node::{
    NodeId, NodeRecord, decode_node_key, decode_temporal_node_key, encode_node_key,
    encode_temporal_node_key, temporal_node_id_prefix,
};
use coordinode_core::index::derive::EntryOwner;
use coordinode_core::txn::invariant::{Claim, ClaimPredicate, ClaimScope};
use coordinode_storage::Guard;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::transaction::{PagedScan, Transaction};

use crate::error::{StoreError, StoreResult};

/// Visitor for [`NodeStore::for_each_in_shard_at_snapshot`]: receives each
/// node's id, raw storage key, and decoded record, returning
/// [`ControlFlow::Break`](core::ops::ControlFlow::Break) to stop the walk early.
pub type ShardVisitor<'a> =
    dyn FnMut(NodeId, &[u8], &NodeRecord) -> StoreResult<core::ops::ControlFlow<()>> + 'a;

/// Visitor for [`NodeStore::for_each_row_in_shard`]: receives each raw node
/// row as `(key, value)`, returning
/// [`ControlFlow::Break`](core::ops::ControlFlow::Break) to stop the walk early.
pub type RowVisitor<'a> = dyn FnMut(&[u8], &[u8]) -> StoreResult<core::ops::ControlFlow<()>> + 'a;

/// Layer 4 node store. Reads/writes [`NodeRecord`] via shard-aware
/// keys over a [`Transaction`]; supports temporal and non-temporal
/// flavours.
pub trait NodeStore {
    /// Read a non-temporal node record by (shard, id). Returns `None`
    /// if the key is absent or has been tombstoned. Untracked — does not
    /// join the OCC read-set (cheap read path for bulk / background work).
    fn get(
        &self,
        txn: &Transaction,
        shard_id: u16,
        node_id: NodeId,
    ) -> StoreResult<Option<NodeRecord>>;

    /// Batch counterpart of [`Self::get`]: read many node records by id in one
    /// call, returning a record (or `None`) per input id in the same order.
    /// Resolves all keys through a single batched engine `multi_get` (one
    /// version-snapshot acquisition + batched bloom/SST traversal) instead of a
    /// per-id lookup loop. Untracked, same as [`Self::get`]. Use it to
    /// materialize the node records behind an index or vector-search result set.
    fn get_many(
        &self,
        txn: &Transaction,
        shard_id: u16,
        node_ids: &[NodeId],
    ) -> StoreResult<Vec<Option<NodeRecord>>>;

    /// Tracked raw read for the query-execution path: encode the key, apply
    /// node-delta read-your-own-writes, then read the raw record bytes joining
    /// the transaction's OCC read-set (in MVCC mode; direct mode tracks
    /// nothing). Returns the raw MessagePack bytes — the query layer decodes
    /// them (it owns the decode-error contract), while the store owns key
    /// encoding + delta materialisation.
    fn get_raw_tracked(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
    ) -> StoreResult<Option<Vec<u8>>>;

    /// OCC-tracked raw read of an exact temporal node version (25-byte key at
    /// `valid_from_ms`). Returns raw bytes — the query layer decodes (keeps its
    /// diagnostic contract). `None` when no row exists at that exact version.
    fn get_temporal_raw_tracked(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        valid_from_ms: i64,
    ) -> StoreResult<Option<Vec<u8>>>;

    /// Tombstone an exact temporal node version (25-byte key at `valid_from_ms`).
    fn delete_temporal(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        valid_from_ms: i64,
    ) -> StoreResult<()>;

    /// Untracked raw read (RYOW write-buffer → snapshot → engine): encode the
    /// key and return the raw bytes WITHOUT joining the OCC read-set and
    /// WITHOUT materialising pending deltas. For schema-introspection peeks
    /// that must not affect conflict detection (the transaction must be pinned
    /// to the statement snapshot — see the executor prelude's `sync_txn_state`).
    /// The caller decodes (keeps its diagnostic contract).
    fn peek_raw(
        &self,
        txn: &Transaction,
        shard_id: u16,
        node_id: NodeId,
    ) -> StoreResult<Option<Vec<u8>>>;

    /// Write a non-temporal node record. Overwrites any prior body at
    /// the same key. Buffers on the transaction; applied at commit.
    fn put(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        record: &NodeRecord,
    ) -> StoreResult<()>;

    /// Tombstone a non-temporal node record. Idempotent on a missing
    /// key.
    fn delete(&self, txn: &mut Transaction, shard_id: u16, node_id: NodeId) -> StoreResult<()>;

    /// Buffer a document-delta operand for `(shard, id)` (encodes the node
    /// key; `operand` is a pre-built `DocDelta::encode()` blob). Applied via
    /// [`LocalNodeStore::materialize_pending_deltas`] on the next read and
    /// drained at commit — the `SET n.path = x` / `REMOVE n.path` write path.
    fn buffer_node_delta(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        operand: Vec<u8>,
    );

    /// The key prefix that matches every node row in a shard. The query layer
    /// uses it for a full-shard node scan without naming the key encoder.
    fn shard_scan_prefix(&self, shard_id: u16) -> Vec<u8>;

    /// The key prefix that matches every temporal version of one node id.
    fn version_prefix(&self, shard_id: u16, node_id: NodeId) -> Vec<u8>;

    /// OCC-tracked prefix scan over `Partition::Node` for a store-built prefix
    /// (from [`Self::shard_scan_prefix`] / [`Self::version_prefix`]). Returns
    /// `(key, raw bytes)` with the transaction's write-buffer overlaid.
    fn prefix_scan_tracked(
        &self,
        txn: &mut Transaction,
        prefix: &[u8],
    ) -> StoreResult<Vec<(Vec<u8>, Vec<u8>)>>;

    /// OCC-tracked, keyset-resumed page of a prefix scan over `Partition::Node`:
    /// the memory-bounded source for a server-side cursor. Reads the committed
    /// snapshot only (no write-buffer overlay), so the cursor path uses it for
    /// read queries and falls back to [`Self::prefix_scan_tracked`] otherwise.
    fn prefix_scan_paged_tracked(
        &self,
        txn: &mut Transaction,
        prefix: &[u8],
        start_after: Option<&[u8]>,
        limit: usize,
    ) -> StoreResult<PagedScan>;

    /// Make `txn`'s commit conditional on the node rows under `keys` (store
    /// keys from a scan) being unchanged since its snapshot. Returns `false`
    /// when one already changed, so the caller reads again without waiting
    /// for the commit to refuse.
    ///
    /// A transaction that derived data from rows it only read (an index
    /// backfill) needs this: at the default level a commit validates its own
    /// writes, and a concurrent write to a row it read would go unnoticed.
    ///
    /// # Errors
    ///
    /// A storage failure.
    fn condition_unchanged(&self, txn: &mut Transaction, keys: &[Vec<u8>]) -> StoreResult<bool>;

    /// Read the temporal version of a node valid at `at_ms`: returns
    /// the version whose `valid_from <= at_ms` is largest. Returns
    /// `None` if the node has no version at-or-before that instant.
    fn get_at(
        &self,
        txn: &Transaction,
        shard_id: u16,
        node_id: NodeId,
        at_ms: i64,
    ) -> StoreResult<Option<NodeRecord>>;

    /// Write a per-version temporal node record. Each `valid_from`
    /// gets its own key — versions accumulate, prior versions remain
    /// readable through [`Self::get_at`] and [`Self::scan_versions`].
    fn put_temporal(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        valid_from_ms: i64,
        record: &NodeRecord,
    ) -> StoreResult<()>;

    /// All temporal versions of one node, in valid_from order.
    fn scan_versions(
        &self,
        txn: &Transaction,
        shard_id: u16,
        node_id: NodeId,
    ) -> StoreResult<Vec<(i64, NodeRecord)>>;

    /// Read a non-temporal node at a specific MVCC snapshot seqno
    /// (writes after the snapshot are invisible). Required by the
    /// query layer's MVCC read path.
    fn get_at_seqno(
        &self,
        txn: &Transaction,
        shard_id: u16,
        node_id: NodeId,
        snapshot: lsm_tree::SeqNo,
    ) -> StoreResult<Option<NodeRecord>>;

    /// Existence check at a pinned snapshot — equivalent to
    /// `get_at_seqno(...).is_some()` but skips decoding the
    /// `NodeRecord` body. Use on hot paths (MVCC visibility filter,
    /// reachability) where the caller only needs yes/no.
    fn contains_at_seqno(
        &self,
        txn: &Transaction,
        shard_id: u16,
        node_id: NodeId,
        snapshot: lsm_tree::SeqNo,
    ) -> StoreResult<bool>;

    /// Stateless typed parallel read. Decode while borrowing the selected
    /// storage value, without copying its serialized payload. Return the key
    /// so the caller can retain its OCC dependency independently of the value.
    fn read_record_at_snapshot(
        &self,
        engine: &StorageEngine,
        snapshot: Option<lsm_tree::SeqNo>,
        shard_id: u16,
        node_id: NodeId,
    ) -> StoreResult<(Vec<u8>, Option<NodeRecord>)>;

    /// Stateless raw read of a node record by its already-encoded key, at
    /// `snapshot` (latest when `None`), with no [`Transaction`]. Returns the
    /// raw MessagePack bytes (caller decodes). For background scanners that
    /// hold a node key from a prior scan and want to re-read it without a
    /// transaction.
    fn read_raw_at_snapshot(
        &self,
        engine: &StorageEngine,
        snapshot: Option<lsm_tree::SeqNo>,
        node_key: &[u8],
    ) -> StoreResult<Option<Vec<u8>>>;

    /// Stateless shard walk at `snapshot` (latest when `None`), with no
    /// [`Transaction`]. Invokes `visit(node_id, node_key, record)` for every
    /// non-temporal node record in the shard, in key order; the visitor
    /// returns [`ControlFlow::Break`] to stop early (e.g. a reaper hitting its
    /// per-pass deletion budget). The raw node key is passed through so the
    /// caller can build keyed mutations without re-encoding. Mirrors
    /// [`Self::for_each_in_shard`] but engine-bound (background path) rather
    /// than transaction-bound. Rows whose body fails to decode are skipped
    /// (not surfaced as an error) so one corrupt record never aborts the walk.
    ///
    /// [`ControlFlow::Break`]: core::ops::ControlFlow::Break
    fn for_each_in_shard_at_snapshot(
        &self,
        engine: &StorageEngine,
        snapshot: Option<lsm_tree::SeqNo>,
        shard_id: u16,
        visit: &mut ShardVisitor<'_>,
    ) -> StoreResult<()>;

    /// Rough number of node records held, for progress and sizing.
    ///
    /// Read off LSM metadata, so it costs no scan and counts versions and
    /// tombstones the levels still hold: an upper bound, never a row count.
    fn approximate_count(&self, engine: &StorageEngine) -> StoreResult<usize>;

    /// Watch the node rows written from now on, returned with a snapshot:
    /// every node write is either visible at the snapshot or delivered by
    /// the tap, whatever timestamp it lands at. The tap yields node keys;
    /// decode them with `decode_node_key`. See
    /// [`StorageEngine::tap_writes`] for the guarantee and its cost.
    fn tap_writes(
        &self,
        engine: &StorageEngine,
    ) -> StoreResult<(coordinode_storage::engine::tap::WriteTap, lsm_tree::SeqNo)>;

    /// The nodes of `shard_id` written (created, changed, merged or deleted)
    /// at or after storage seqno `since`, found from the stored version
    /// history in time proportional to those writes. A read at timestamp T
    /// passes T + 1: the result is every node whose state differs from the
    /// snapshot at T, complete for any T still readable.
    fn written_since(
        &self,
        engine: &StorageEngine,
        shard_id: u16,
        since: u64,
    ) -> StoreResult<Vec<NodeId>>;

    /// Every version of temporal node `node_id`, as `(valid_from, record)` in
    /// ascending `valid_from` order, as `txn` sees them: its snapshot with
    /// its own buffered writes over it. Untracked. Empty for a node without
    /// a timeline.
    fn versions(
        &self,
        txn: &Transaction,
        shard_id: u16,
        node_id: NodeId,
    ) -> StoreResult<Vec<(i64, NodeRecord)>>;

    /// Every node row of `shard_id` in key order, raw, streamed from
    /// `engine` at the latest state: each node's own row and each version of
    /// a temporal node, which sort together under the node's id. `visit`
    /// returns [`ControlFlow::Break`] to stop early.
    ///
    /// [`ControlFlow::Break`]: core::ops::ControlFlow::Break
    fn for_each_row_in_shard(
        &self,
        engine: &StorageEngine,
        shard_id: u16,
        visit: &mut RowVisitor<'_>,
    ) -> StoreResult<()>;

    /// Every node row of `shard_id` in key order, raw, as `txn` sees them:
    /// its snapshot with its own buffered writes over it. Untracked.
    fn shard_rows(&self, txn: &Transaction, shard_id: u16) -> StoreResult<Vec<(Vec<u8>, Vec<u8>)>>;

    /// Iterate every non-temporal node record in a shard, latest
    /// visible seqno. Yields `(NodeId, NodeRecord)` pairs in key
    /// order. Materialised into a `Vec` — callers walking very large
    /// shards (>1M nodes) should prefer [`Self::for_each_in_shard`]
    /// which streams in constant memory.
    fn scan_shard(
        &self,
        txn: &Transaction,
        shard_id: u16,
    ) -> StoreResult<Vec<(NodeId, NodeRecord)>>;

    /// Streaming shard walk. Invokes `visit(node_id, record)` for
    /// every non-temporal entry; bails on the first visitor error.
    /// Constant memory regardless of shard size — unlike
    /// [`Self::scan_shard`] which collects into a `Vec`.
    fn for_each_in_shard(
        &self,
        txn: &Transaction,
        shard_id: u16,
        visit: &mut dyn FnMut(NodeId, NodeRecord) -> StoreResult<()>,
    ) -> StoreResult<()>;

    /// One page of the rows of `shard_id` after the row key `after`, at
    /// most `limit`, in key order, decoded, as `txn` sees them: each node's
    /// own row and each version of a temporal node, named by its owner.
    /// Untracked.
    ///
    /// # Errors
    ///
    /// A storage failure or an undecodable record.
    fn rows_page(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        after: Option<&[u8]>,
        limit: usize,
    ) -> StoreResult<RowPage>;

    /// The row of `owner` (a node, or one version of a temporal node), as
    /// `txn` sees it. Untracked.
    ///
    /// # Errors
    ///
    /// A storage failure or an undecodable record.
    fn row(
        &self,
        txn: &Transaction,
        shard_id: u16,
        owner: EntryOwner,
    ) -> StoreResult<Option<NodeRecord>>;

    /// The latest committed row of `owner`, with the version of its write.
    ///
    /// # Errors
    ///
    /// A storage failure or an undecodable record.
    fn latest_row(
        &self,
        engine: &StorageEngine,
        shard_id: u16,
        owner: EntryOwner,
    ) -> StoreResult<Option<(NodeRecord, u64)>>;

    /// Write the transaction only while the row of `owner` is at `version`
    /// when it commits (`None`: while there is none).
    ///
    /// # Errors
    ///
    /// A storage failure.
    fn expect_row_txn(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        owner: EntryOwner,
        version: Option<u64>,
    ) -> StoreResult<()>;
}

/// One page of a shard's rows ([`NodeStore::rows_page`]).
#[derive(Debug, Clone)]
pub struct RowPage {
    /// Each row's owner and record, in key order.
    pub rows: Vec<(EntryOwner, NodeRecord)>,
    /// The key of the last row read, where the next page resumes; `None`
    /// for an empty page.
    pub resume: Option<Vec<u8>>,
    /// No row of the shard follows this page.
    pub exhausted: bool,
}

/// The key of the row of `owner`.
fn row_key(shard_id: u16, owner: EntryOwner) -> Vec<u8> {
    let node = NodeId::from_raw(owner.node_id);
    match owner.valid_from {
        Some(valid_from) => encode_temporal_node_key(shard_id, node, valid_from),
        None => encode_node_key(shard_id, node),
    }
}

/// CE single-shard implementation of [`NodeStore`]. Stateless — all
/// storage access flows through the [`Transaction`] passed to each
/// method.
pub struct LocalNodeStore;

impl LocalNodeStore {
    fn decode_record(bytes: &[u8]) -> StoreResult<NodeRecord> {
        NodeRecord::from_msgpack(bytes).map_err(|e| StoreError::Decode {
            kind: "node record",
            message: format!("{e}"),
        })
    }

    fn encode_record(record: &NodeRecord) -> StoreResult<Vec<u8>> {
        record.to_msgpack().map_err(|e| StoreError::Decode {
            kind: "node record",
            message: format!("encode: {e}"),
        })
    }

    /// The storage key of `node_id`'s record on `shard_id`: what a writer
    /// that stages changes by key (a conditioned page, a document delta)
    /// names the record by.
    pub fn record_key(shard_id: u16, node_id: NodeId) -> Vec<u8> {
        encode_node_key(shard_id, node_id)
    }

    /// Read-your-own-writes for buffered document deltas (`SET n.path = x`):
    /// apply every pending [`DocDelta`](coordinode_core::graph::doc_delta::DocDelta)
    /// for `node_key` against the current record and buffer the materialised
    /// result, so a subsequent read in the same transaction sees the update.
    /// Drains the applied deltas (idempotent on a second call). The deltas
    /// are applied as the commit's merge applies them, so the read sees what
    /// will be stored. A node-modality concern that the modality-agnostic
    /// [`Transaction`] does not own.
    pub fn materialize_pending_deltas(txn: &mut Transaction, node_key: &[u8]) -> StoreResult<()> {
        if !txn.node_deltas().iter().any(|(k, _)| k == node_key) {
            return Ok(());
        }
        // The current value is read untracked: this is an internal
        // read-modify-write, not a user read joining the OCC set.
        let Some(record) = Self::post_state(txn, node_key)? else {
            return Ok(());
        };
        txn.node_deltas_mut().retain(|(k, _)| k != node_key);
        let new_bytes = record.to_msgpack().map_err(|e| StoreError::Decode {
            kind: "node record",
            message: format!("RYOW encode: {e}"),
        })?;
        txn.put(Partition::Node, node_key, &new_bytes)?;
        Ok(())
    }

    /// The record at `node_key` as the transaction leaves it: its buffered
    /// or stored record with the transaction's pending document deltas
    /// applied, staging nothing. `None` for a record the transaction deletes
    /// or that does not exist and has no pending deltas.
    ///
    /// # Errors
    ///
    /// A storage failure, or a record or delta that does not decode.
    pub fn post_state(txn: &Transaction, node_key: &[u8]) -> StoreResult<Option<NodeRecord>> {
        let base = txn.read_untracked(Partition::Node, node_key)?;
        Self::fold_pending(txn, node_key, base)
    }

    /// [`Self::post_state`] through the transaction's tracked read: the read
    /// joins the transaction's read set as any user read does, and the
    /// pending deltas stay pending, so a writer that changes the record
    /// with deltas can read it between them without folding them into a
    /// whole write.
    ///
    /// # Errors
    ///
    /// A storage failure, or a record or delta that does not decode.
    ///
    /// Also returns the size of the stored or buffered record the deltas
    /// apply to, which tells a writer whether rewriting the record whole is
    /// cheap.
    pub fn post_state_tracked(
        txn: &mut Transaction,
        node_key: &[u8],
    ) -> StoreResult<Option<(NodeRecord, usize)>> {
        let base = txn.get(Partition::Node, node_key)?;
        let size = base.as_ref().map_or(0, Vec::len);
        Ok(Self::fold_pending(txn, node_key, base)?.map(|record| (record, size)))
    }

    /// `base` (the record as stored or buffered) with the transaction's
    /// pending document deltas for `node_key` applied.
    fn fold_pending(
        txn: &Transaction,
        node_key: &[u8],
        base: Option<Vec<u8>>,
    ) -> StoreResult<Option<NodeRecord>> {
        use coordinode_core::graph::doc_delta::{DocDelta, PREFIX_DOC_DELTA};

        let mut deltas = Vec::new();
        for (_, operand) in txn.node_deltas().iter().filter(|(k, _)| k == node_key) {
            let decoded = match operand.split_first() {
                Some((&PREFIX_DOC_DELTA, body)) => {
                    DocDelta::decode(body).map_err(|e| StoreError::Decode {
                        kind: "document delta",
                        message: e.to_string(),
                    })?
                }
                _ => {
                    return Err(StoreError::Decode {
                        kind: "document delta",
                        message: "operand without the document-delta prefix".to_string(),
                    });
                }
            };
            deltas.push(decoded);
        }
        let mut record = match base {
            Some(bytes) => NodeRecord::from_msgpack(&bytes).map_err(|e| StoreError::Decode {
                kind: "node record",
                message: e.to_string(),
            })?,
            None if deltas.is_empty() => return Ok(None),
            None => NodeRecord::new(""),
        };
        coordinode_storage::engine::merge::apply_doc_deltas_to_record(&mut record, &deltas);
        Ok(Some(record))
    }
}

impl NodeStore for LocalNodeStore {
    fn get(
        &self,
        txn: &Transaction,
        shard_id: u16,
        node_id: NodeId,
    ) -> StoreResult<Option<NodeRecord>> {
        let key = encode_node_key(shard_id, node_id);
        match txn.read_untracked(Partition::Node, &key)? {
            Some(bytes) => Self::decode_record(&bytes).map(Some),
            None => Ok(None),
        }
    }

    fn get_many(
        &self,
        txn: &Transaction,
        shard_id: u16,
        node_ids: &[NodeId],
    ) -> StoreResult<Vec<Option<NodeRecord>>> {
        let keys: Vec<Vec<u8>> = node_ids
            .iter()
            .map(|id| encode_node_key(shard_id, *id))
            .collect();
        let key_refs: Vec<&[u8]> = keys.iter().map(Vec::as_slice).collect();
        let raws = txn.multi_read_untracked(Partition::Node, &key_refs)?;
        raws.into_iter()
            .map(|opt| opt.map(|bytes| Self::decode_record(&bytes)).transpose())
            .collect()
    }

    fn get_raw_tracked(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
    ) -> StoreResult<Option<Vec<u8>>> {
        let key = encode_node_key(shard_id, node_id);
        // RYOW: surface pending document deltas for this node before the
        // OCC-tracked read.
        Self::materialize_pending_deltas(txn, &key)?;
        Ok(txn.get(Partition::Node, &key)?)
    }

    fn peek_raw(
        &self,
        txn: &Transaction,
        shard_id: u16,
        node_id: NodeId,
    ) -> StoreResult<Option<Vec<u8>>> {
        let key = encode_node_key(shard_id, node_id);
        Ok(txn.read_untracked(Partition::Node, &key)?)
    }

    fn get_temporal_raw_tracked(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        valid_from_ms: i64,
    ) -> StoreResult<Option<Vec<u8>>> {
        let key = encode_temporal_node_key(shard_id, node_id, valid_from_ms);
        Ok(txn.get(Partition::Node, &key)?)
    }

    fn delete_temporal(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        valid_from_ms: i64,
    ) -> StoreResult<()> {
        let key = encode_temporal_node_key(shard_id, node_id, valid_from_ms);
        txn.delete(Partition::Node, &key)?;
        Ok(())
    }

    fn put(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        record: &NodeRecord,
    ) -> StoreResult<()> {
        let key = encode_node_key(shard_id, node_id);
        let body = Self::encode_record(record)?;
        txn.put(Partition::Node, &key, &body)?;
        Ok(())
    }

    fn delete(&self, txn: &mut Transaction, shard_id: u16, node_id: NodeId) -> StoreResult<()> {
        // Destroying the identity invalidates every reference right held on
        // it, so the deletion says so: an attempt attaching an edge to this
        // node has stated that the node stays, and the two cannot both be
        // right. Without this the write sets are disjoint (one deletes a node
        // row, the other merges into an adjacency list) and validation admits
        // both, leaving an edge pointing at a node that no longer exists.
        let generation = txn.schema_generation();
        txn.claim(Claim::new(
            ClaimScope::Node(node_id),
            ClaimPredicate::EndpointDestroyed,
            generation,
        ));
        let key = encode_node_key(shard_id, node_id);
        txn.delete(Partition::Node, &key)?;
        Ok(())
    }

    fn buffer_node_delta(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        operand: Vec<u8>,
    ) {
        let key = encode_node_key(shard_id, node_id);
        txn.push_node_delta(key, operand);
    }

    fn shard_scan_prefix(&self, shard_id: u16) -> Vec<u8> {
        // The shard component of the node key: everything before the node id.
        // A prefix scan with this bytes matches every node row in the shard
        // (both 16-byte non-temporal and 25-byte temporal keys).
        encode_node_key(shard_id, NodeId::from_raw(0))[..8].to_vec()
    }

    fn version_prefix(&self, shard_id: u16, node_id: NodeId) -> Vec<u8> {
        temporal_node_id_prefix(shard_id, node_id)
    }

    fn prefix_scan_tracked(
        &self,
        txn: &mut Transaction,
        prefix: &[u8],
    ) -> StoreResult<Vec<(Vec<u8>, Vec<u8>)>> {
        Ok(txn.prefix_scan(Partition::Node, prefix)?)
    }

    fn prefix_scan_paged_tracked(
        &self,
        txn: &mut Transaction,
        prefix: &[u8],
        start_after: Option<&[u8]>,
        limit: usize,
    ) -> StoreResult<PagedScan> {
        Ok(txn.prefix_scan_paged(Partition::Node, prefix, start_after, limit)?)
    }

    fn condition_unchanged(&self, txn: &mut Transaction, keys: &[Vec<u8>]) -> StoreResult<bool> {
        // A snapshot at S sees the versions below S; a row whose latest
        // version is at or above it changed after the rows were read.
        let seen_below = txn.snapshot();
        for key in keys {
            let version = txn.record_version(Partition::Node, key)?;
            if seen_below.is_some_and(|s| version.is_some_and(|v| v >= s)) {
                return Ok(false);
            }
            txn.expect_version(Partition::Node, key, version)?;
        }
        Ok(true)
    }

    fn get_at(
        &self,
        txn: &Transaction,
        shard_id: u16,
        node_id: NodeId,
        at_ms: i64,
    ) -> StoreResult<Option<NodeRecord>> {
        // Walk all versions for the id; pick the largest valid_from
        // <= at_ms. The per-id prefix is small (versions per node is
        // O(few hundred) typical), so an in-memory pick is fine and
        // avoids a reverse-seek API on the engine.
        let prefix = temporal_node_id_prefix(shard_id, node_id);
        let mut best: Option<(i64, Vec<u8>)> = None;
        for (key, value) in txn.base_prefix_scan(Partition::Node, &prefix)? {
            let Some((_, _, vf)) = decode_temporal_node_key(&key) else {
                continue;
            };
            if vf > at_ms {
                continue;
            }
            match &best {
                Some((cur, _)) if *cur >= vf => {}
                _ => best = Some((vf, value)),
            }
        }
        match best {
            Some((_, bytes)) => Self::decode_record(&bytes).map(Some),
            None => Ok(None),
        }
    }

    fn put_temporal(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        valid_from_ms: i64,
        record: &NodeRecord,
    ) -> StoreResult<()> {
        let key = encode_temporal_node_key(shard_id, node_id, valid_from_ms);
        let body = Self::encode_record(record)?;
        txn.put(Partition::Node, &key, &body)?;
        Ok(())
    }

    fn scan_versions(
        &self,
        txn: &Transaction,
        shard_id: u16,
        node_id: NodeId,
    ) -> StoreResult<Vec<(i64, NodeRecord)>> {
        let prefix = temporal_node_id_prefix(shard_id, node_id);
        let mut out = Vec::new();
        for (key, value) in txn.base_prefix_scan(Partition::Node, &prefix)? {
            let Some((_, _, vf)) = decode_temporal_node_key(&key) else {
                continue;
            };
            out.push((vf, Self::decode_record(&value)?));
        }
        // Engine returns keys in sorted byte order; the valid_from
        // suffix uses sortable encoding so the natural iteration order
        // is already chronological. Documented invariant.
        Ok(out)
    }

    fn get_at_seqno(
        &self,
        txn: &Transaction,
        shard_id: u16,
        node_id: NodeId,
        snapshot: lsm_tree::SeqNo,
    ) -> StoreResult<Option<NodeRecord>> {
        let key = encode_node_key(shard_id, node_id);
        match txn.snapshot_get_at(snapshot, Partition::Node, &key)? {
            Some(bytes) => Self::decode_record(&bytes).map(Some),
            None => Ok(None),
        }
    }

    fn contains_at_seqno(
        &self,
        txn: &Transaction,
        shard_id: u16,
        node_id: NodeId,
        snapshot: lsm_tree::SeqNo,
    ) -> StoreResult<bool> {
        let key = encode_node_key(shard_id, node_id);
        Ok(txn
            .snapshot_get_at(snapshot, Partition::Node, &key)?
            .is_some())
    }

    fn read_record_at_snapshot(
        &self,
        engine: &StorageEngine,
        snapshot: Option<lsm_tree::SeqNo>,
        shard_id: u16,
        node_id: NodeId,
    ) -> StoreResult<(Vec<u8>, Option<NodeRecord>)> {
        let key = encode_node_key(shard_id, node_id);
        let record = match snapshot {
            Some(snap) => engine.with_snapshot_value(&snap, Partition::Node, &key, |bytes| {
                bytes.map(Self::decode_record).transpose()
            })??,
            None => {
                let bytes = engine.get(Partition::Node, &key)?;
                bytes.as_deref().map(Self::decode_record).transpose()?
            }
        };
        Ok((key, record))
    }

    fn read_raw_at_snapshot(
        &self,
        engine: &StorageEngine,
        snapshot: Option<lsm_tree::SeqNo>,
        node_key: &[u8],
    ) -> StoreResult<Option<Vec<u8>>> {
        let bytes = match snapshot {
            Some(snap) => engine
                .snapshot_get(&snap, Partition::Node, node_key)?
                .map(|b| b.to_vec()),
            None => engine.get(Partition::Node, node_key)?.map(|b| b.to_vec()),
        };
        Ok(bytes)
    }

    fn for_each_in_shard_at_snapshot(
        &self,
        engine: &StorageEngine,
        snapshot: Option<lsm_tree::SeqNo>,
        shard_id: u16,
        visit: &mut ShardVisitor<'_>,
    ) -> StoreResult<()> {
        let mut prefix = Vec::with_capacity(8);
        prefix.extend_from_slice(b"node:");
        prefix.extend_from_slice(&shard_id.to_be_bytes());
        prefix.push(b':');

        // Decode + dispatch one scanned (key, value) pair; returns whether to
        // continue. Non-temporal node keys are exactly 16 bytes
        // (node:<shard:2><id:8>); temporal versions add a suffix — skip those.
        let mut handle = |key: &[u8], value: &[u8]| -> StoreResult<core::ops::ControlFlow<()>> {
            if key.len() != 16 {
                return Ok(core::ops::ControlFlow::Continue(()));
            }
            let id_bytes: [u8; 8] = match key[8..16].try_into() {
                Ok(b) => b,
                Err(_) => return Ok(core::ops::ControlFlow::Continue(())),
            };
            let node_id = NodeId::from_raw(u64::from_be_bytes(id_bytes));
            // Skip rows whose body fails to decode rather than aborting the
            // whole walk — a single corrupt record must not take down a
            // background shard scan (TTL reaper, maintenance).
            let Ok(record) = Self::decode_record(value) else {
                return Ok(core::ops::ControlFlow::Continue(()));
            };
            visit(node_id, key, &record)
        };

        match snapshot {
            Some(snap) => {
                for (key, value) in engine.snapshot_prefix_scan(&snap, Partition::Node, &prefix)? {
                    if handle(&key, &value)?.is_break() {
                        break;
                    }
                }
            }
            None => {
                for guard in engine.prefix_scan(Partition::Node, &prefix)? {
                    let (key, value) = guard.into_inner()?;
                    if handle(&key, &value)?.is_break() {
                        break;
                    }
                }
            }
        }
        Ok(())
    }

    fn approximate_count(&self, engine: &StorageEngine) -> StoreResult<usize> {
        Ok(engine.approximate_len(Partition::Node)?)
    }

    fn tap_writes(
        &self,
        engine: &StorageEngine,
    ) -> StoreResult<(coordinode_storage::engine::tap::WriteTap, lsm_tree::SeqNo)> {
        Ok(engine.tap_writes(Partition::Node)?)
    }

    fn written_since(
        &self,
        engine: &StorageEngine,
        shard_id: u16,
        since: u64,
    ) -> StoreResult<Vec<NodeId>> {
        // A temporal node's versions each have a key of their own: a write to
        // any of them changes the node.
        let mut nodes: Vec<NodeId> = engine
            .changed_keys_since(Partition::Node, since)?
            .iter()
            .filter_map(|key| coordinode_core::graph::node::decode_written_node(key))
            .filter(|(shard, _)| *shard == shard_id)
            .map(|(_, id)| id)
            .collect();
        nodes.sort_unstable();
        nodes.dedup();
        Ok(nodes)
    }

    fn versions(
        &self,
        txn: &Transaction,
        shard_id: u16,
        node_id: NodeId,
    ) -> StoreResult<Vec<(i64, NodeRecord)>> {
        let rows = txn.prefix_scan(Partition::Node, &temporal_node_id_prefix(shard_id, node_id))?;
        let mut versions = Vec::with_capacity(rows.len());
        for (key, value) in rows {
            if let Some((_, _, valid_from)) =
                coordinode_core::graph::node::decode_temporal_node_key(&key)
            {
                versions.push((valid_from, Self::decode_record(&value)?));
            }
        }
        Ok(versions)
    }

    fn for_each_row_in_shard(
        &self,
        engine: &StorageEngine,
        shard_id: u16,
        visit: &mut RowVisitor<'_>,
    ) -> StoreResult<()> {
        for guard in engine.prefix_scan(Partition::Node, &self.shard_scan_prefix(shard_id))? {
            let (key, value) = guard.into_inner()?;
            if visit(&key, &value)?.is_break() {
                break;
            }
        }
        Ok(())
    }

    fn shard_rows(&self, txn: &Transaction, shard_id: u16) -> StoreResult<Vec<(Vec<u8>, Vec<u8>)>> {
        Ok(txn.prefix_scan(Partition::Node, &self.shard_scan_prefix(shard_id))?)
    }

    fn for_each_in_shard(
        &self,
        txn: &Transaction,
        shard_id: u16,
        visit: &mut dyn FnMut(NodeId, NodeRecord) -> StoreResult<()>,
    ) -> StoreResult<()> {
        let mut prefix = Vec::with_capacity(8);
        prefix.extend_from_slice(b"node:");
        prefix.extend_from_slice(&shard_id.to_be_bytes());
        prefix.push(b':');
        let iter = txn.base_prefix_iter(Partition::Node, &prefix)?;
        for guard in iter {
            let (key, value) = guard.into_inner()?;
            if key.len() != 16 {
                continue;
            }
            let id_bytes: [u8; 8] = match key[8..16].try_into() {
                Ok(b) => b,
                Err(_) => continue,
            };
            let node_id = NodeId::from_raw(u64::from_be_bytes(id_bytes));
            visit(node_id, Self::decode_record(&value)?)?;
        }
        Ok(())
    }

    fn scan_shard(
        &self,
        txn: &Transaction,
        shard_id: u16,
    ) -> StoreResult<Vec<(NodeId, NodeRecord)>> {
        // Non-temporal node keys are exactly 16 bytes (node:<shard:2><id:8>);
        // temporal keys add a :<vf:8> suffix making them 25 bytes. Filter
        // to the 16-byte form so callers don't have to disambiguate.
        let mut prefix = Vec::with_capacity(8);
        prefix.extend_from_slice(b"node:");
        prefix.extend_from_slice(&shard_id.to_be_bytes());
        prefix.push(b':');
        let mut out = Vec::new();
        for (key, value) in txn.base_prefix_scan(Partition::Node, &prefix)? {
            // 16-byte non-temporal keys only — skip temporal versions
            // (those have a trailing valid_from suffix).
            if key.len() != 16 {
                continue;
            }
            let id_bytes: [u8; 8] = match key[8..16].try_into() {
                Ok(b) => b,
                Err(_) => continue,
            };
            let node_id = NodeId::from_raw(u64::from_be_bytes(id_bytes));
            out.push((node_id, Self::decode_record(&value)?));
        }
        Ok(out)
    }

    fn rows_page(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        after: Option<&[u8]>,
        limit: usize,
    ) -> StoreResult<RowPage> {
        let start = self.shard_scan_prefix(shard_id);
        // Exclusive end past every row key of the shard: a row key is at
        // most 25 bytes, the prefix 8.
        let mut end = start.clone();
        end.extend_from_slice(&[0xFF; 32]);
        let page = txn.range_scan_paged(Partition::Node, &start, &end, after, limit)?;
        let mut rows = Vec::with_capacity(page.rows.len());
        for (key, value) in &page.rows {
            let owner = match decode_node_key(key) {
                Some((shard, id)) if shard == shard_id => EntryOwner::node(id.as_raw()),
                Some(_) => continue,
                None => match decode_temporal_node_key(key) {
                    Some((shard, id, valid_from)) if shard == shard_id => {
                        EntryOwner::version(id.as_raw(), valid_from)
                    }
                    _ => continue,
                },
            };
            rows.push((owner, Self::decode_record(value)?));
        }
        Ok(RowPage {
            rows,
            resume: page.last_key,
            exhausted: page.exhausted,
        })
    }

    fn row(
        &self,
        txn: &Transaction,
        shard_id: u16,
        owner: EntryOwner,
    ) -> StoreResult<Option<NodeRecord>> {
        txn.read_untracked(Partition::Node, &row_key(shard_id, owner))?
            .map(|bytes| Self::decode_record(&bytes))
            .transpose()
    }

    fn latest_row(
        &self,
        engine: &StorageEngine,
        shard_id: u16,
        owner: EntryOwner,
    ) -> StoreResult<Option<(NodeRecord, u64)>> {
        engine
            .get_versioned(Partition::Node, &row_key(shard_id, owner))?
            .map(|(bytes, version)| Ok((Self::decode_record(&bytes)?, version)))
            .transpose()
    }

    fn expect_row_txn(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        owner: EntryOwner,
        version: Option<u64>,
    ) -> StoreResult<()> {
        Ok(txn.expect_version(Partition::Node, &row_key(shard_id, owner), version)?)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests;
