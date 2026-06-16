//! Node store — typed read/write of [`NodeRecord`] in
//! [`Partition::Node`], with first-class support for both
//! non-temporal labels (one row per node id, ADR pre-027) and
//! temporal labels (one row per `valid_from` version, ADR-027).
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
//! ## Transaction threading (ADR-041)
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
    decode_temporal_node_key, encode_node_key, encode_temporal_node_key, temporal_node_id_prefix,
    NodeId, NodeRecord,
};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::transaction::Transaction;
use coordinode_storage::Guard;

use crate::error::{StoreError, StoreResult};

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
    /// [`Self::materialize_pending_deltas`] on the next read and drained at
    /// commit — the `SET n.path = x` / `REMOVE n.path` write path.
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

    /// Stateless parallel-read: encode the node key and read its raw bytes
    /// directly from the engine at `snapshot` (or latest when `None`), with
    /// no [`Transaction`]. Returns the encoded key alongside the bytes so a
    /// parallel worker can record it in its own OCC accumulator and merge into
    /// the statement's read-set after the parallel section. The shared unit is
    /// the snapshot seqno (a `Copy` value), so rayon workers read concurrently
    /// without a `&mut Transaction` — the read path the sharded scatter-gather
    /// model also uses (snapshot-parameterised, not transaction-bound).
    fn read_at_snapshot(
        &self,
        engine: &StorageEngine,
        snapshot: Option<lsm_tree::SeqNo>,
        shard_id: u16,
        node_id: NodeId,
    ) -> StoreResult<(Vec<u8>, Option<Vec<u8>>)>;

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
}

/// CE single-shard implementation of [`NodeStore`]. Stateless — all
/// storage access flows through the [`Transaction`] passed to each
/// method (ADR-041).
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

    /// Read-your-own-writes for buffered document deltas (`SET n.path = x`):
    /// apply every pending [`DocDelta`](coordinode_core::graph::doc_delta::DocDelta)
    /// for `node_key` against the current record and buffer the materialised
    /// result, so a subsequent read in the same transaction sees the update.
    /// Drains the applied deltas (idempotent on a second call). A node-modality
    /// concern that the modality-agnostic [`Transaction`] does not own.
    pub fn materialize_pending_deltas(txn: &mut Transaction, node_key: &[u8]) -> StoreResult<()> {
        use coordinode_core::graph::doc_delta::{DocDelta, PathTarget};
        use coordinode_core::graph::types::Value;

        let matching: Vec<Vec<u8>> = txn
            .node_deltas()
            .iter()
            .filter(|(k, _)| k == node_key)
            .map(|(_, op)| op.clone())
            .collect();
        if matching.is_empty() {
            return Ok(());
        }
        // Remove materialised deltas from the buffer.
        txn.node_deltas_mut().retain(|(k, _)| k != node_key);

        // Current value via read-your-own-writes (untracked: this is an
        // internal read-modify-write, not a user read joining the OCC set).
        let current = txn.read_untracked(Partition::Node, node_key)?;
        let mut record = match current {
            Some(ref bytes) => NodeRecord::from_msgpack(bytes).map_err(|e| StoreError::Decode {
                kind: "node record",
                message: format!("RYOW decode: {e}"),
            })?,
            None => NodeRecord::new(""),
        };
        for operand in &matching {
            if let Ok(delta) = DocDelta::decode(&operand[1..]) {
                match delta.target() {
                    PathTarget::PropField(field_id) => {
                        let mut doc = match record.props.get(field_id) {
                            Some(v) => v.to_rmpv(),
                            None => rmpv::Value::Map(Vec::new()),
                        };
                        delta.apply(&mut doc);
                        record.set(*field_id, Value::Document(doc));
                    }
                    PathTarget::Extra => {
                        // Extra-targeted deltas handled by the merge function.
                    }
                }
            }
        }
        let new_bytes = record.to_msgpack().map_err(|e| StoreError::Decode {
            kind: "node record",
            message: format!("RYOW encode: {e}"),
        })?;
        txn.put(Partition::Node, node_key, &new_bytes)?;
        Ok(())
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

    fn read_at_snapshot(
        &self,
        engine: &StorageEngine,
        snapshot: Option<lsm_tree::SeqNo>,
        shard_id: u16,
        node_id: NodeId,
    ) -> StoreResult<(Vec<u8>, Option<Vec<u8>>)> {
        let key = encode_node_key(shard_id, node_id);
        let bytes = match snapshot {
            Some(snap) => engine
                .snapshot_get(&snap, Partition::Node, &key)?
                .map(|b| b.to_vec()),
            None => engine.get(Partition::Node, &key)?.map(|b| b.to_vec()),
        };
        Ok((key, bytes))
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
    /// A standalone oracle over the shared logic engine is sufficient —
    /// reads pin `engine.snapshot()` (latest committed), not `read_ts`.
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
        fn write(&self, f: impl FnOnce(&LocalNodeStore, &mut Transaction)) {
            let mut t = mvcc_txn(&self.engine, &self.oracle);
            let store = LocalNodeStore;
            f(&store, &mut t);
            commit(&mut t);
        }

        fn read(&self) -> Transaction<'_> {
            mvcc_txn(&self.engine, &self.oracle)
        }
    }

    fn rec(label: &str) -> NodeRecord {
        NodeRecord::new(label)
    }

    #[test]
    fn non_temporal_round_trip() {
        let db = open();
        let id = NodeId::from_raw(7);
        let store = LocalNodeStore;
        assert!(store.get(&db.read(), 0, id).expect("none").is_none());
        db.write(|s, t| s.put(t, 0, id, &rec("User")).expect("put"));
        let got = store.get(&db.read(), 0, id).expect("some").expect("Some");
        assert_eq!(got.primary_label(), "User");
    }

    #[test]
    fn put_overwrites_existing_record() {
        let db = open();
        let id = NodeId::from_raw(1);
        db.write(|s, t| s.put(t, 0, id, &rec("A")).expect("put A"));
        db.write(|s, t| s.put(t, 0, id, &rec("B")).expect("put B"));
        let got = LocalNodeStore
            .get(&db.read(), 0, id)
            .expect("ok")
            .expect("Some");
        assert_eq!(got.primary_label(), "B");
    }

    #[test]
    fn delete_tombstones_record() {
        let db = open();
        let id = NodeId::from_raw(1);
        db.write(|s, t| s.put(t, 0, id, &rec("X")).expect("put"));
        db.write(|s, t| s.delete(t, 0, id).expect("delete"));
        assert!(LocalNodeStore.get(&db.read(), 0, id).expect("ok").is_none());
    }

    #[test]
    fn delete_missing_is_idempotent() {
        let db = open();
        db.write(|s, t| {
            s.delete(t, 0, NodeId::from_raw(999))
                .expect("delete missing")
        });
    }

    #[test]
    fn shards_isolated_by_key_prefix() {
        // Same node_id under different shards must NOT collide.
        let db = open();
        let id = NodeId::from_raw(42);
        db.write(|s, t| {
            s.put(t, 0, id, &rec("ShardZero")).expect("put");
            s.put(t, 1, id, &rec("ShardOne")).expect("put");
        });
        let store = LocalNodeStore;
        let r = db.read();
        assert_eq!(
            store
                .get(&r, 0, id)
                .expect("ok")
                .expect("Some")
                .primary_label(),
            "ShardZero"
        );
        assert_eq!(
            store
                .get(&r, 1, id)
                .expect("ok")
                .expect("Some")
                .primary_label(),
            "ShardOne"
        );
    }

    #[test]
    fn temporal_versions_round_trip() {
        let db = open();
        let id = NodeId::from_raw(11);
        db.write(|s, t| {
            s.put_temporal(t, 0, id, 100, &rec("V1")).expect("put v1");
            s.put_temporal(t, 0, id, 200, &rec("V2")).expect("put v2");
            s.put_temporal(t, 0, id, 300, &rec("V3")).expect("put v3");
        });
        let versions = LocalNodeStore
            .scan_versions(&db.read(), 0, id)
            .expect("scan");
        let labels: Vec<&str> = versions.iter().map(|(_, r)| r.primary_label()).collect();
        let times: Vec<i64> = versions.iter().map(|(t, _)| *t).collect();
        assert_eq!(times, vec![100, 200, 300]);
        assert_eq!(labels, vec!["V1", "V2", "V3"]);
    }

    #[test]
    fn get_at_returns_largest_valid_from_le_query() {
        // Versions at t=100, 200, 300. Queries:
        // - at 99 → None (no version exists yet)
        // - at 100 → V1 (exact match)
        // - at 199 → V1 (largest <= 199)
        // - at 200 → V2 (exact match)
        // - at 1_000_000 → V3 (far future picks newest)
        let db = open();
        let id = NodeId::from_raw(11);
        db.write(|s, t| {
            s.put_temporal(t, 0, id, 100, &rec("V1")).expect("v1");
            s.put_temporal(t, 0, id, 200, &rec("V2")).expect("v2");
            s.put_temporal(t, 0, id, 300, &rec("V3")).expect("v3");
        });
        let store = LocalNodeStore;
        let r = db.read();

        assert!(store.get_at(&r, 0, id, 99).expect("ok").is_none());
        assert_eq!(
            store
                .get_at(&r, 0, id, 100)
                .expect("ok")
                .expect("Some")
                .primary_label(),
            "V1",
        );
        assert_eq!(
            store
                .get_at(&r, 0, id, 199)
                .expect("ok")
                .expect("Some")
                .primary_label(),
            "V1",
        );
        assert_eq!(
            store
                .get_at(&r, 0, id, 200)
                .expect("ok")
                .expect("Some")
                .primary_label(),
            "V2",
        );
        assert_eq!(
            store
                .get_at(&r, 0, id, 1_000_000)
                .expect("ok")
                .expect("Some")
                .primary_label(),
            "V3",
        );
    }

    #[test]
    fn get_at_isolated_per_node_id() {
        // Two distinct nodes in the same shard each have temporal
        // versions — get_at on one must not bleed into the other.
        let db = open();
        let a = NodeId::from_raw(1);
        let b = NodeId::from_raw(2);
        db.write(|s, t| {
            s.put_temporal(t, 0, a, 50, &rec("A50")).expect("a");
            s.put_temporal(t, 0, b, 70, &rec("B70")).expect("b");
        });
        let store = LocalNodeStore;
        let r = db.read();
        assert_eq!(
            store
                .get_at(&r, 0, a, 100)
                .expect("ok")
                .expect("Some")
                .primary_label(),
            "A50",
        );
        assert_eq!(
            store
                .get_at(&r, 0, b, 100)
                .expect("ok")
                .expect("Some")
                .primary_label(),
            "B70",
        );
    }

    #[test]
    fn put_temporal_same_valid_from_overwrites_body() {
        // Two writes at the same (node_id, valid_from) land at the same
        // key — second wins, no version explosion.
        let db = open();
        let id = NodeId::from_raw(40);
        let mut rec_v1 = rec("A");
        rec_v1.set_extra("v", coordinode_core::graph::types::Value::Int(1));
        let mut rec_v2 = rec("A");
        rec_v2.set_extra("v", coordinode_core::graph::types::Value::Int(2));
        db.write(|s, t| {
            s.put_temporal(t, 0, id, 1000, &rec_v1).expect("v1");
            s.put_temporal(t, 0, id, 1000, &rec_v2).expect("v2");
        });
        let versions = LocalNodeStore
            .scan_versions(&db.read(), 0, id)
            .expect("scan");
        assert_eq!(versions.len(), 1, "same valid_from = one row");
        assert_eq!(versions[0].0, 1000);
        assert_eq!(
            versions[0].1.get_extra("v"),
            Some(&coordinode_core::graph::types::Value::Int(2)),
        );
    }

    #[test]
    fn scan_versions_on_empty_node_returns_empty() {
        let db = open();
        let versions = LocalNodeStore
            .scan_versions(&db.read(), 0, NodeId::from_raw(999))
            .expect("ok");
        assert!(versions.is_empty());
    }

    #[test]
    fn get_at_boundary_i64_min_max() {
        // Per ADR-027 valid_from_ms is i64. Test we can write at the
        // extreme boundaries and the sortable encoding still works.
        let db = open();
        let id = NodeId::from_raw(41);
        db.write(|s, t| {
            s.put_temporal(t, 0, id, i64::MIN, &rec("min"))
                .expect("min");
            s.put_temporal(t, 0, id, i64::MAX, &rec("max"))
                .expect("max");
        });
        let store = LocalNodeStore;
        let r = db.read();
        let versions = store.scan_versions(&r, 0, id).expect("scan");
        assert_eq!(versions.len(), 2);
        assert_eq!(versions[0].0, i64::MIN);
        assert_eq!(versions[1].0, i64::MAX);
        // Query at 0 picks the MIN-version (largest valid_from <= 0).
        let active = store.get_at(&r, 0, id, 0).expect("ok").expect("Some");
        assert_eq!(active.primary_label(), "min");
    }

    #[test]
    fn get_at_seqno_returns_version_visible_at_snapshot() {
        let db = open();
        let id = NodeId::from_raw(50);
        db.write(|s, t| s.put(t, 0, id, &rec("v1")).expect("put v1"));
        let snap = db.engine.snapshot();
        db.write(|s, t| s.put(t, 0, id, &rec("v2")).expect("put v2"));
        let store = LocalNodeStore;
        let r = db.read();
        let at_snap = store
            .get_at_seqno(&r, 0, id, snap)
            .expect("ok")
            .expect("Some");
        assert_eq!(at_snap.primary_label(), "v1");
        let latest = store.get(&r, 0, id).expect("ok").expect("Some");
        assert_eq!(latest.primary_label(), "v2");
    }

    #[test]
    fn get_at_seqno_missing_returns_none() {
        let db = open();
        let snap = db.engine.snapshot();
        assert!(LocalNodeStore
            .get_at_seqno(&db.read(), 0, NodeId::from_raw(999), snap)
            .expect("ok")
            .is_none());
    }

    #[test]
    fn scan_shard_yields_every_non_temporal_record() {
        let db = open();
        db.write(|s, t| {
            for i in 0u64..5 {
                s.put(t, 0, NodeId::from_raw(i + 100), &rec(&format!("L{i}")))
                    .expect("put");
            }
        });
        let all = LocalNodeStore.scan_shard(&db.read(), 0).expect("scan");
        assert_eq!(all.len(), 5);
        let mut ids: Vec<u64> = all.iter().map(|(id, _)| id.as_raw()).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![100, 101, 102, 103, 104]);
    }

    #[test]
    fn scan_shard_isolated_per_shard() {
        let db = open();
        db.write(|s, t| {
            s.put(t, 0, NodeId::from_raw(1), &rec("s0")).unwrap();
            s.put(t, 1, NodeId::from_raw(2), &rec("s1")).unwrap();
        });
        let store = LocalNodeStore;
        let r = db.read();
        let shard0 = store.scan_shard(&r, 0).unwrap();
        let shard1 = store.scan_shard(&r, 1).unwrap();
        assert_eq!(shard0.len(), 1);
        assert_eq!(shard1.len(), 1);
        assert_eq!(shard0[0].0, NodeId::from_raw(1));
        assert_eq!(shard1[0].0, NodeId::from_raw(2));
    }

    #[test]
    fn scan_shard_skips_temporal_versions() {
        // 25-byte temporal keys must not leak into the non-temporal
        // scan result.
        let db = open();
        db.write(|s, t| {
            s.put(t, 0, NodeId::from_raw(1), &rec("nt")).unwrap();
            s.put_temporal(t, 0, NodeId::from_raw(2), 1000, &rec("t"))
                .unwrap();
        });
        let all = LocalNodeStore.scan_shard(&db.read(), 0).unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].0, NodeId::from_raw(1));
    }

    #[test]
    fn for_each_in_shard_visits_every_record() {
        let db = open();
        db.write(|s, t| {
            for i in 0u64..4 {
                s.put(t, 0, NodeId::from_raw(i + 50), &rec(&format!("N{i}")))
                    .expect("put");
            }
        });
        let store = LocalNodeStore;
        let r = db.read();
        let mut seen: Vec<u64> = Vec::new();
        store
            .for_each_in_shard(&r, 0, &mut |id, _rec| {
                seen.push(id.as_raw());
                Ok(())
            })
            .expect("walk");
        seen.sort_unstable();
        assert_eq!(seen, vec![50, 51, 52, 53]);
    }

    #[test]
    fn corrupt_node_bytes_surface_as_decode_error() {
        let db = open();
        // Inject garbage directly at the node key.
        db.engine
            .put(
                Partition::Node,
                &encode_node_key(0, NodeId::from_raw(5)),
                &[0xff, 0xff, 0xff],
            )
            .expect("inject");
        let err = LocalNodeStore
            .get(&db.read(), 0, NodeId::from_raw(5))
            .expect_err("must err");
        assert!(matches!(
            err,
            StoreError::Decode {
                kind: "node record",
                ..
            }
        ));
    }
}
