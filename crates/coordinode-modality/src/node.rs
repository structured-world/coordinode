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
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::Guard;

use crate::error::{StoreError, StoreResult};

/// Layer 4 node store. Reads/writes [`NodeRecord`] via shard-aware
/// keys; supports temporal and non-temporal flavours.
pub trait NodeStore {
    /// Read a non-temporal node record by (shard, id). Returns `None`
    /// if the key is absent or has been tombstoned.
    fn get(&self, shard_id: u16, node_id: NodeId) -> StoreResult<Option<NodeRecord>>;

    /// Write a non-temporal node record. Overwrites any prior body
    /// at the same key. Atomicity matches the underlying engine
    /// (single put = single LSM mutation).
    fn put(&self, shard_id: u16, node_id: NodeId, record: &NodeRecord) -> StoreResult<()>;

    /// Tombstone a non-temporal node record. Idempotent on a missing
    /// key.
    fn delete(&self, shard_id: u16, node_id: NodeId) -> StoreResult<()>;

    /// Read the temporal version of a node valid at `at_ms`:
    /// returns the version whose `valid_from <= at_ms` is largest.
    /// Returns `None` if the node has no version at-or-before that
    /// instant.
    fn get_at(&self, shard_id: u16, node_id: NodeId, at_ms: i64)
        -> StoreResult<Option<NodeRecord>>;

    /// Write a per-version temporal node record. Each `valid_from`
    /// gets its own key — versions accumulate, prior versions remain
    /// readable through [`Self::get_at`] and [`Self::scan_versions`].
    fn put_temporal(
        &self,
        shard_id: u16,
        node_id: NodeId,
        valid_from_ms: i64,
        record: &NodeRecord,
    ) -> StoreResult<()>;

    /// All temporal versions of one node, in valid_from order.
    fn scan_versions(&self, shard_id: u16, node_id: NodeId) -> StoreResult<Vec<(i64, NodeRecord)>>;
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
