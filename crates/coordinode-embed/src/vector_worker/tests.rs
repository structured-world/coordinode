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
