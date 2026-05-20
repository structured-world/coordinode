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
use coordinode_storage::engine::coordinator::MultiModalCoordinator;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::Guard;

use crate::error::{StoreError, StoreResult};

/// Layer 4 node store. Reads/writes [`NodeRecord`] via shard-aware
/// keys; supports temporal and non-temporal flavours.
pub trait NodeStore {
    /// Read a non-temporal node record by (shard, id). Returns `None`
    /// if the key is absent or has been tombstoned.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalNodeStore, NodeStore};
    /// # use coordinode_core::graph::node::{NodeId, NodeRecord};
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalNodeStore::new(&engine);
    /// let id = NodeId::from_raw(1);
    /// assert!(store.get(0, id)?.is_none());
    /// store.put(0, id, &NodeRecord::new("User"))?;
    /// assert!(store.get(0, id)?.is_some());
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn get(&self, shard_id: u16, node_id: NodeId) -> StoreResult<Option<NodeRecord>>;

    /// Write a non-temporal node record. Overwrites any prior body
    /// at the same key. Atomicity matches the underlying engine
    /// (single put = single LSM mutation).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalNodeStore, NodeStore};
    /// # use coordinode_core::graph::node::{NodeId, NodeRecord};
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalNodeStore::new(&engine);
    /// store.put(0, NodeId::from_raw(1), &NodeRecord::new("User"))?;
    /// // Second write overwrites the first.
    /// store.put(0, NodeId::from_raw(1), &NodeRecord::new("Admin"))?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn put(&self, shard_id: u16, node_id: NodeId, record: &NodeRecord) -> StoreResult<()>;

    /// Tombstone a non-temporal node record. Idempotent on a missing
    /// key.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalNodeStore, NodeStore};
    /// # use coordinode_core::graph::node::NodeId;
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalNodeStore::new(&engine);
    /// // Deleting a never-existed key is a no-op (succeeds).
    /// store.delete(0, NodeId::from_raw(404))?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn delete(&self, shard_id: u16, node_id: NodeId) -> StoreResult<()>;

    /// Read the temporal version of a node valid at `at_ms`:
    /// returns the version whose `valid_from <= at_ms` is largest.
    /// Returns `None` if the node has no version at-or-before that
    /// instant.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalNodeStore, NodeStore};
    /// # use coordinode_core::graph::node::{NodeId, NodeRecord};
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalNodeStore::new(&engine);
    /// let id = NodeId::from_raw(1);
    /// store.put_temporal(0, id, 1000, &NodeRecord::new("v1"))?;
    /// store.put_temporal(0, id, 2000, &NodeRecord::new("v2"))?;
    /// // At 1500, the v1 version is active.
    /// assert!(store.get_at(0, id, 1500)?.is_some());
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn get_at(&self, shard_id: u16, node_id: NodeId, at_ms: i64)
        -> StoreResult<Option<NodeRecord>>;

    /// Write a per-version temporal node record. Each `valid_from`
    /// gets its own key — versions accumulate, prior versions remain
    /// readable through [`Self::get_at`] and [`Self::scan_versions`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalNodeStore, NodeStore};
    /// # use coordinode_core::graph::node::{NodeId, NodeRecord};
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalNodeStore::new(&engine);
    /// store.put_temporal(0, NodeId::from_raw(1), 1000, &NodeRecord::new("v1"))?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn put_temporal(
        &self,
        shard_id: u16,
        node_id: NodeId,
        valid_from_ms: i64,
        record: &NodeRecord,
    ) -> StoreResult<()>;

    /// All temporal versions of one node, in valid_from order.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalNodeStore, NodeStore};
    /// # use coordinode_core::graph::node::{NodeId, NodeRecord};
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalNodeStore::new(&engine);
    /// let versions = store.scan_versions(0, NodeId::from_raw(1))?;
    /// for (valid_from, _record) in &versions { let _ = valid_from; }
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn scan_versions(&self, shard_id: u16, node_id: NodeId) -> StoreResult<Vec<(i64, NodeRecord)>>;

    /// Read a non-temporal node at a specific MVCC snapshot seqno
    /// (writes after the snapshot are invisible). Required by the
    /// query layer's MVCC read path.
    fn get_at_seqno(
        &self,
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
        shard_id: u16,
        node_id: NodeId,
        snapshot: lsm_tree::SeqNo,
    ) -> StoreResult<bool>;

    /// Iterate every non-temporal node record in a shard, latest
    /// visible seqno. Yields `(NodeId, NodeRecord)` pairs in key
    /// order. Materialised into a `Vec` — callers walking very
    /// large shards (>1M nodes) should prefer
    /// [`Self::for_each_in_shard`] which streams in constant
    /// memory.
    fn scan_shard(&self, shard_id: u16) -> StoreResult<Vec<(NodeId, NodeRecord)>>;

    /// Streaming shard walk. Invokes `visit(node_id, record)` for
    /// every non-temporal entry; bails on the first visitor error.
    /// Constant memory regardless of shard size — unlike
    /// [`Self::scan_shard`] which collects into a `Vec`.
    fn for_each_in_shard(
        &self,
        shard_id: u16,
        visit: &mut dyn FnMut(NodeId, NodeRecord) -> StoreResult<()>,
    ) -> StoreResult<()>;
}

/// CE single-shard implementation of [`NodeStore`].
pub struct LocalNodeStore<'a> {
    engine: &'a StorageEngine,
}

impl<'a> LocalNodeStore<'a> {
    /// Wrap a storage engine for node-store operations. The store is
    /// stateless beyond the borrow — cheap to construct, clone the
    /// engine handle to share across threads.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use coordinode_modality::{LocalNodeStore, NodeStore};
    /// use coordinode_core::graph::node::{NodeId, NodeRecord};
    /// use coordinode_storage::engine::config::{
    ///     Durability, EndpointConfig, Media, StorageConfig, Tier,
    /// };
    /// use coordinode_storage::engine::core::StorageEngine;
    ///
    /// let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    ///     "ep", std::path::Path::new("/tmp/store"),
    ///     Media::Hdd, Durability::Durable, Tier::Warm,
    /// )]);
    /// let engine = StorageEngine::open(&cfg)?;
    /// let store = LocalNodeStore::new(&engine);
    /// store.put(0, NodeId::from_raw(1), &NodeRecord::new("User"))?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(engine: &'a StorageEngine) -> Self {
        Self { engine }
    }

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
}

impl NodeStore for LocalNodeStore<'_> {
    fn get(&self, shard_id: u16, node_id: NodeId) -> StoreResult<Option<NodeRecord>> {
        let key = encode_node_key(shard_id, node_id);
        let Some(bytes) = self.engine.get(Partition::Node, &key)? else {
            return Ok(None);
        };
        Self::decode_record(&bytes).map(Some)
    }

    fn put(&self, shard_id: u16, node_id: NodeId, record: &NodeRecord) -> StoreResult<()> {
        let key = encode_node_key(shard_id, node_id);
        let body = Self::encode_record(record)?;
        self.engine.put(Partition::Node, &key, &body)?;
        Ok(())
    }

    fn delete(&self, shard_id: u16, node_id: NodeId) -> StoreResult<()> {
        let key = encode_node_key(shard_id, node_id);
        self.engine.delete(Partition::Node, &key)?;
        Ok(())
    }

    fn get_at(
        &self,
        shard_id: u16,
        node_id: NodeId,
        at_ms: i64,
    ) -> StoreResult<Option<NodeRecord>> {
        // Walk all versions for the id; pick the largest valid_from
        // <= at_ms. The per-id prefix is small (versions per node is
        // O(few hundred) typical), so an in-memory pick is fine and
        // avoids a reverse-seek API on the engine.
        let prefix = temporal_node_id_prefix(shard_id, node_id);
        let iter = self.engine.prefix_scan(Partition::Node, &prefix)?;
        let mut best: Option<(i64, Vec<u8>)> = None;
        for guard in iter {
            let (key, value) = guard.into_inner()?;
            let Some((_, _, vf)) = decode_temporal_node_key(&key) else {
                continue;
            };
            if vf > at_ms {
                continue;
            }
            match &best {
                Some((cur, _)) if *cur >= vf => {}
                _ => best = Some((vf, value.to_vec())),
            }
        }
        match best {
            Some((_, bytes)) => Self::decode_record(&bytes).map(Some),
            None => Ok(None),
        }
    }

    fn put_temporal(
        &self,
        shard_id: u16,
        node_id: NodeId,
        valid_from_ms: i64,
        record: &NodeRecord,
    ) -> StoreResult<()> {
        let key = encode_temporal_node_key(shard_id, node_id, valid_from_ms);
        let body = Self::encode_record(record)?;
        self.engine.put(Partition::Node, &key, &body)?;
        Ok(())
    }

    fn scan_versions(&self, shard_id: u16, node_id: NodeId) -> StoreResult<Vec<(i64, NodeRecord)>> {
        let prefix = temporal_node_id_prefix(shard_id, node_id);
        let iter = self.engine.prefix_scan(Partition::Node, &prefix)?;
        let mut out = Vec::new();
        for guard in iter {
            let (key, value) = guard.into_inner()?;
            let Some((_, _, vf)) = decode_temporal_node_key(&key) else {
                continue;
            };
            out.push((vf, Self::decode_record(&value)?));
        }
        // Engine returns keys in sorted byte order; the valid_from
        // suffix uses sortable encoding so the natural iteration
        // order is already chronological. Documented invariant.
        Ok(out)
    }

    fn get_at_seqno(
        &self,
        shard_id: u16,
        node_id: NodeId,
        snapshot: lsm_tree::SeqNo,
    ) -> StoreResult<Option<NodeRecord>> {
        let key = encode_node_key(shard_id, node_id);
        let Some(bytes) =
            self.engine
                .coordinator()
                .snapshot_get(&snapshot, Partition::Node, &key)?
        else {
            return Ok(None);
        };
        Self::decode_record(&bytes).map(Some)
    }

    fn contains_at_seqno(
        &self,
        shard_id: u16,
        node_id: NodeId,
        snapshot: lsm_tree::SeqNo,
    ) -> StoreResult<bool> {
        let key = encode_node_key(shard_id, node_id);
        Ok(self
            .engine
            .coordinator()
            .snapshot_get(&snapshot, Partition::Node, &key)?
            .is_some())
    }

    fn for_each_in_shard(
        &self,
        shard_id: u16,
        visit: &mut dyn FnMut(NodeId, NodeRecord) -> StoreResult<()>,
    ) -> StoreResult<()> {
        let mut prefix = Vec::with_capacity(8);
        prefix.extend_from_slice(b"node:");
        prefix.extend_from_slice(&shard_id.to_be_bytes());
        prefix.push(b':');
        let iter = self.engine.prefix_scan(Partition::Node, &prefix)?;
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

    fn scan_shard(&self, shard_id: u16) -> StoreResult<Vec<(NodeId, NodeRecord)>> {
        // Non-temporal node keys are exactly 16 bytes (node:<shard:2><id:8>);
        // temporal keys add a :<vf:8> suffix making them 25 bytes. Filter
        // to the 16-byte form so callers don't have to disambiguate.
        let mut prefix = Vec::with_capacity(8);
        prefix.extend_from_slice(b"node:");
        prefix.extend_from_slice(&shard_id.to_be_bytes());
        prefix.push(b':');
        let iter = self.engine.prefix_scan(Partition::Node, &prefix)?;
        let mut out = Vec::new();
        for guard in iter {
            let (key, value) = guard.into_inner()?;
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

    fn rec(label: &str) -> NodeRecord {
        NodeRecord::new(label)
    }

    #[test]
    fn non_temporal_round_trip() {
        let (_dir, engine) = open_engine();
        let store = LocalNodeStore::new(&engine);
        let id = NodeId::from_raw(7);
        assert!(store.get(0, id).expect("none").is_none());
        store.put(0, id, &rec("User")).expect("put");
        let got = store.get(0, id).expect("some").expect("Some");
        assert_eq!(got.primary_label(), "User");
    }

    #[test]
    fn put_overwrites_existing_record() {
        let (_dir, engine) = open_engine();
        let store = LocalNodeStore::new(&engine);
        let id = NodeId::from_raw(1);
        store.put(0, id, &rec("A")).expect("put A");
        store.put(0, id, &rec("B")).expect("put B");
        let got = store.get(0, id).expect("ok").expect("Some");
        assert_eq!(got.primary_label(), "B");
    }

    #[test]
    fn delete_tombstones_record() {
        let (_dir, engine) = open_engine();
        let store = LocalNodeStore::new(&engine);
        let id = NodeId::from_raw(1);
        store.put(0, id, &rec("X")).expect("put");
        store.delete(0, id).expect("delete");
        assert!(store.get(0, id).expect("ok").is_none());
    }

    #[test]
    fn delete_missing_is_idempotent() {
        let (_dir, engine) = open_engine();
        let store = LocalNodeStore::new(&engine);
        store
            .delete(0, NodeId::from_raw(999))
            .expect("delete missing");
    }

    #[test]
    fn shards_isolated_by_key_prefix() {
        // Same node_id under different shards must NOT collide.
        let (_dir, engine) = open_engine();
        let store = LocalNodeStore::new(&engine);
        let id = NodeId::from_raw(42);
        store.put(0, id, &rec("ShardZero")).expect("put");
        store.put(1, id, &rec("ShardOne")).expect("put");
        assert_eq!(
            store.get(0, id).expect("ok").expect("Some").primary_label(),
            "ShardZero"
        );
        assert_eq!(
            store.get(1, id).expect("ok").expect("Some").primary_label(),
            "ShardOne"
        );
    }

    #[test]
    fn temporal_versions_round_trip() {
        let (_dir, engine) = open_engine();
        let store = LocalNodeStore::new(&engine);
        let id = NodeId::from_raw(11);
        store.put_temporal(0, id, 100, &rec("V1")).expect("put v1");
        store.put_temporal(0, id, 200, &rec("V2")).expect("put v2");
        store.put_temporal(0, id, 300, &rec("V3")).expect("put v3");

        let versions = store.scan_versions(0, id).expect("scan");
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
        let (_dir, engine) = open_engine();
        let store = LocalNodeStore::new(&engine);
        let id = NodeId::from_raw(11);
        store.put_temporal(0, id, 100, &rec("V1")).expect("v1");
        store.put_temporal(0, id, 200, &rec("V2")).expect("v2");
        store.put_temporal(0, id, 300, &rec("V3")).expect("v3");

        assert!(store.get_at(0, id, 99).expect("ok").is_none());
        assert_eq!(
            store
                .get_at(0, id, 100)
                .expect("ok")
                .expect("Some")
                .primary_label(),
            "V1",
        );
        assert_eq!(
            store
                .get_at(0, id, 199)
                .expect("ok")
                .expect("Some")
                .primary_label(),
            "V1",
        );
        assert_eq!(
            store
                .get_at(0, id, 200)
                .expect("ok")
                .expect("Some")
                .primary_label(),
            "V2",
        );
        assert_eq!(
            store
                .get_at(0, id, 1_000_000)
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
        let (_dir, engine) = open_engine();
        let store = LocalNodeStore::new(&engine);
        let a = NodeId::from_raw(1);
        let b = NodeId::from_raw(2);
        store.put_temporal(0, a, 50, &rec("A50")).expect("a");
        store.put_temporal(0, b, 70, &rec("B70")).expect("b");

        assert_eq!(
            store
                .get_at(0, a, 100)
                .expect("ok")
                .expect("Some")
                .primary_label(),
            "A50",
        );
        assert_eq!(
            store
                .get_at(0, b, 100)
                .expect("ok")
                .expect("Some")
                .primary_label(),
            "B70",
        );
    }

    #[test]
    fn put_temporal_same_valid_from_overwrites_body() {
        // Two writes at the same (node_id, valid_from) land at the
        // same key — second wins, no version explosion.
        let (_dir, engine) = open_engine();
        let store = LocalNodeStore::new(&engine);
        let id = NodeId::from_raw(40);

        let mut rec_v1 = rec("A");
        rec_v1.set_extra("v", coordinode_core::graph::types::Value::Int(1));
        let mut rec_v2 = rec("A");
        rec_v2.set_extra("v", coordinode_core::graph::types::Value::Int(2));

        store.put_temporal(0, id, 1000, &rec_v1).expect("v1");
        store.put_temporal(0, id, 1000, &rec_v2).expect("v2");

        let versions = store.scan_versions(0, id).expect("scan");
        assert_eq!(versions.len(), 1, "same valid_from = one row");
        assert_eq!(versions[0].0, 1000);
        assert_eq!(
            versions[0].1.get_extra("v"),
            Some(&coordinode_core::graph::types::Value::Int(2)),
        );
    }

    #[test]
    fn scan_versions_on_empty_node_returns_empty() {
        let (_dir, engine) = open_engine();
        let store = LocalNodeStore::new(&engine);
        let versions = store.scan_versions(0, NodeId::from_raw(999)).expect("ok");
        assert!(versions.is_empty());
    }

    #[test]
    fn get_at_boundary_i64_min_max() {
        // Per ADR-027 valid_from_ms is i64. Test we can write at the
        // extreme boundaries and the sortable encoding still works.
        let (_dir, engine) = open_engine();
        let store = LocalNodeStore::new(&engine);
        let id = NodeId::from_raw(41);

        store
            .put_temporal(0, id, i64::MIN, &rec("min"))
            .expect("min");
        store
            .put_temporal(0, id, i64::MAX, &rec("max"))
            .expect("max");
        let versions = store.scan_versions(0, id).expect("scan");
        assert_eq!(versions.len(), 2);
        assert_eq!(versions[0].0, i64::MIN);
        assert_eq!(versions[1].0, i64::MAX);
        // Query at 0 picks the MIN-version (largest valid_from <= 0).
        let active = store.get_at(0, id, 0).expect("ok").expect("Some");
        assert_eq!(active.primary_label(), "min");
    }

    #[test]
    fn get_at_seqno_returns_version_visible_at_snapshot() {
        let (_dir, engine) = open_engine();
        let store = LocalNodeStore::new(&engine);
        let id = NodeId::from_raw(50);
        store.put(0, id, &rec("v1")).expect("put v1");
        let snap = engine.snapshot();
        store.put(0, id, &rec("v2")).expect("put v2");
        let at_snap = store.get_at_seqno(0, id, snap).expect("ok").expect("Some");
        assert_eq!(at_snap.primary_label(), "v1");
        let latest = store.get(0, id).expect("ok").expect("Some");
        assert_eq!(latest.primary_label(), "v2");
    }

    #[test]
    fn get_at_seqno_missing_returns_none() {
        let (_dir, engine) = open_engine();
        let store = LocalNodeStore::new(&engine);
        let snap = engine.snapshot();
        assert!(store
            .get_at_seqno(0, NodeId::from_raw(999), snap)
            .expect("ok")
            .is_none());
    }

    #[test]
    fn scan_shard_yields_every_non_temporal_record() {
        let (_dir, engine) = open_engine();
        let store = LocalNodeStore::new(&engine);
        for i in 0u64..5 {
            store
                .put(0, NodeId::from_raw(i + 100), &rec(&format!("L{i}")))
                .expect("put");
        }
        let all = store.scan_shard(0).expect("scan");
        assert_eq!(all.len(), 5);
        let mut ids: Vec<u64> = all.iter().map(|(id, _)| id.as_raw()).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![100, 101, 102, 103, 104]);
    }

    #[test]
    fn scan_shard_isolated_per_shard() {
        let (_dir, engine) = open_engine();
        let store = LocalNodeStore::new(&engine);
        store.put(0, NodeId::from_raw(1), &rec("s0")).unwrap();
        store.put(1, NodeId::from_raw(2), &rec("s1")).unwrap();
        let shard0 = store.scan_shard(0).unwrap();
        let shard1 = store.scan_shard(1).unwrap();
        assert_eq!(shard0.len(), 1);
        assert_eq!(shard1.len(), 1);
        assert_eq!(shard0[0].0, NodeId::from_raw(1));
        assert_eq!(shard1[0].0, NodeId::from_raw(2));
    }

    #[test]
    fn scan_shard_skips_temporal_versions() {
        // 25-byte temporal keys must not leak into the
        // non-temporal scan result.
        let (_dir, engine) = open_engine();
        let store = LocalNodeStore::new(&engine);
        store.put(0, NodeId::from_raw(1), &rec("nt")).unwrap();
        store
            .put_temporal(0, NodeId::from_raw(2), 1000, &rec("t"))
            .unwrap();
        let all = store.scan_shard(0).unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].0, NodeId::from_raw(1));
    }

    #[test]
    fn corrupt_node_bytes_surface_as_decode_error() {
        let (_dir, engine) = open_engine();
        let store = LocalNodeStore::new(&engine);
        engine
            .put(
                Partition::Node,
                &encode_node_key(0, NodeId::from_raw(5)),
                &[0xff, 0xff, 0xff],
            )
            .expect("inject");
        let err = store.get(0, NodeId::from_raw(5)).expect_err("must err");
        assert!(matches!(
            err,
            StoreError::Decode {
                kind: "node record",
                ..
            }
        ));
    }
}
