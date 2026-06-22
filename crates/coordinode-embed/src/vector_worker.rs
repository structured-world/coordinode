//! Incremental HNSW maintenance from the unified oplog.
//!
//! [`VectorIndexWorker`] tails the oplog and feeds every replicated
//! node write whose `(label, property)` carries a registered vector
//! index into the local HNSW graph. This is how a Raft follower's
//! index stays current AFTER its bootstrap rebuild: the one-shot
//! backfill covers history, the worker covers the live tail. The
//! graph itself is never replicated; each replica derives its own.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use coordinode_core::graph::intern::FieldInterner;
use coordinode_core::graph::node::NodeRecord;
use coordinode_query::index::VectorIndexRegistry;
use coordinode_storage::oplog::entry::OplogOp;
use coordinode_storage::oplog::tailer::{CdcFilters, OplogTailer, ResumeToken};
use parking_lot::RwLock;

/// Wire discriminant of the Node partition in oplog ops (matches the
/// raft layer's partition encoding).
const NODE_PARTITION_U8: u8 = 0;

/// How long the worker sleeps when the oplog has no new entries.
const IDLE_POLL: Duration = Duration::from_millis(50);

/// Max entries pulled from the oplog per batch.
const BATCH: usize = 256;

/// Background thread that tails the oplog and applies vector inserts
/// to the in-memory HNSW indexes of this process.
pub struct VectorIndexWorker {
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl VectorIndexWorker {
    /// Spawn the worker tailing `oplog_dir` from `start` onwards.
    ///
    /// `start` should be the position recorded AFTER the bootstrap
    /// scan/backfill finished, so history is covered by the rebuild
    /// and the live tail by this worker. HNSW insert is an upsert per
    /// node id, so overlap between the two is harmless.
    pub fn spawn(
        oplog_dir: PathBuf,
        start: ResumeToken,
        registry: Arc<VectorIndexRegistry>,
        interner: Arc<RwLock<FieldInterner>>,
    ) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_flag = Arc::clone(&stop);
        let handle = std::thread::Builder::new()
            .name("vec-oplog-worker".to_string())
            .spawn(move || run(oplog_dir, start, registry, interner, stop_flag))
            .ok();
        Self { stop, handle }
    }

    /// Signal the worker to stop and wait for it to exit.
    pub fn shutdown(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for VectorIndexWorker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn run(
    oplog_dir: PathBuf,
    start: ResumeToken,
    registry: Arc<VectorIndexRegistry>,
    interner: Arc<RwLock<FieldInterner>>,
    stop: Arc<AtomicBool>,
) {
    let mut tailer = OplogTailer::new(&oplog_dir, start);
    let filters = CdcFilters::default();
    tracing::info!(dir = %oplog_dir.display(), "vector oplog worker started");

    while !stop.load(Ordering::Relaxed) {
        let batch = match tailer.read_next(BATCH, &filters) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(%e, "vector oplog worker read error; retrying");
                std::thread::sleep(IDLE_POLL);
                continue;
            }
        };
        if batch.is_empty() {
            std::thread::sleep(IDLE_POLL);
            continue;
        }
        // Track the highest commit HLC consumed in this batch. After the
        // worker has applied every entry up to `max_ts`, every index it
        // maintains for this shard has seen all writes up to `max_ts` — the
        // ones with no vector-write at `max_ts` are still current as of it.
        // This is the per-shard read-your-writes freshness watermark.
        let mut max_ts = 0u64;
        for (entry, _token) in batch {
            for op in &entry.ops {
                let OplogOp::Insert {
                    partition,
                    key,
                    value,
                } = op
                else {
                    continue;
                };
                if *partition != NODE_PARTITION_U8 {
                    continue;
                }
                apply_node_write(key, value, &registry, &interner);
            }
            max_ts = max_ts.max(entry.ts);
        }
        if max_ts > 0 {
            registry.advance_indexed_hlc_all(max_ts);
        }
    }
    tracing::info!("vector oplog worker stopped");
}

/// Feed one replicated node record into every registered vector index
/// that covers one of its properties.
fn apply_node_write(
    key: &[u8],
    value: &[u8],
    registry: &VectorIndexRegistry,
    interner: &RwLock<FieldInterner>,
) {
    let Some((_shard, node_id)) = coordinode_core::graph::node::decode_node_key(key) else {
        return;
    };
    let Ok(record) = NodeRecord::from_msgpack(value) else {
        return;
    };
    let label = record.primary_label();
    let props = registry.indexed_properties(label);
    if props.is_empty() {
        return;
    }
    let guard = interner.read();
    for prop in props {
        let Some(field_id) = guard.lookup(&prop) else {
            continue;
        };
        let Some(val) = record.props.get(&field_id) else {
            continue;
        };
        if let Some(vec) = crate::db::try_extract_vector(val) {
            registry.on_vector_written(label, node_id, &prop, &vec);
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use coordinode_core::graph::node::{encode_node_key, NodeId};
    use coordinode_query::index::{IndexDefinition, VectorIndexConfig};
    use coordinode_storage::oplog::entry::OplogEntry;
    use coordinode_storage::oplog::manager::OplogManager;

    fn node_insert_op(node_id: u64, vec: Vec<f32>, interner: &RwLock<FieldInterner>) -> OplogOp {
        use coordinode_core::graph::types::Value;
        let field_id = interner.write().intern("embedding");
        let mut record = NodeRecord::new("Item");
        record.props.insert(field_id, Value::Vector(vec));
        OplogOp::Insert {
            partition: NODE_PARTITION_U8,
            key: encode_node_key(1, NodeId::from_raw(node_id)),
            value: record.to_msgpack().unwrap(),
        }
    }

    fn entry(index: u64, ops: Vec<OplogOp>) -> OplogEntry {
        OplogEntry {
            ts: index,
            term: 1,
            index,
            shard: 0,
            ops,
            is_migration: false,
            pre_images: None,
        }
    }

    /// Node writes appended to the oplog AFTER the worker started must
    /// land in the registered HNSW index without any rebuild.
    #[test]
    fn worker_applies_live_oplog_inserts() {
        let dir = tempfile::tempdir().unwrap();
        let oplog_dir = dir.path().join("oplog").join("0");
        std::fs::create_dir_all(&oplog_dir).unwrap();

        let registry = Arc::new(VectorIndexRegistry::new());
        let config = VectorIndexConfig {
            dimensions: 4,
            metric: coordinode_core::graph::types::VectorMetric::L2,
            m: 8,
            ef_construction: 32,
            quantization: coordinode_vector::hnsw::QuantizationCodec::None,
            offload_vectors: false,
            ef_search: None,
            rerank_candidates: None,
            shard_strategy: Default::default(),
        };
        registry.register(IndexDefinition::hnsw(
            "item_emb",
            "Item",
            "embedding",
            config,
        ));

        let interner = Arc::new(RwLock::new(FieldInterner::new()));
        // Pre-intern so the worker resolves the property id the records use.
        interner.write().intern("embedding");

        let worker = VectorIndexWorker::spawn(
            oplog_dir.clone(),
            ResumeToken::from_start(0),
            Arc::clone(&registry),
            Arc::clone(&interner),
        );

        let mut mgr =
            OplogManager::open(&oplog_dir, 0, 64 * 1024 * 1024, 50_000, 7 * 24 * 3600).unwrap();
        for i in 0..20u64 {
            let op = node_insert_op(i + 1, vec![i as f32, 0.0, 0.0, 0.0], &interner);
            mgr.append(&entry(i, vec![op])).unwrap();
        }
        mgr.flush().unwrap();

        // Poll until the worker has drained the tail.
        let handle = registry.get("Item", "embedding").unwrap();
        let mut indexed = 0;
        for _ in 0..100 {
            std::thread::sleep(Duration::from_millis(50));
            indexed = handle.read().map(|h| h.len()).unwrap_or(0);
            if indexed == 20 {
                break;
            }
        }
        worker.shutdown();
        assert_eq!(indexed, 20, "all live oplog inserts must reach the index");

        // Freshness watermark must advance to the last applied entry's HLC
        // (entries carry ts = index, 0..=19). This is the read-your-writes
        // fence: a reader that wrote at HLC <= 19 sees a current index.
        assert_eq!(
            registry
                .health_snapshot("Item", "embedding")
                .and_then(|h| h.indexed_hlc()),
            Some(19),
            "worker must advance the index freshness watermark to the last applied entry ts"
        );

        // And the index must actually answer with them.
        let results = handle.read().unwrap().search(&[5.0, 0.0, 0.0, 0.0], 1);
        assert_eq!(results.len(), 1);
    }
}
