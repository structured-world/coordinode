use std::time::Duration;

use super::*;
use coordinode_core::graph::node::{NodeRecord, encode_node_key};
use coordinode_core::graph::types::Value;
use coordinode_core::txn::proposal::{Mutation, PartitionId};
use coordinode_query::index::{GenerationId, IndexDescriptor, IndexId, VectorIndexConfig};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::partition::Partition;

const SHARD: u16 = 1;

/// A dictionary fixed for the test: the worker only reads bindings.
struct FixedFields(coordinode_core::graph::intern::FieldInterner);

impl FieldRegistrar for FixedFields {
    fn register(
        &self,
        names: &[&str],
    ) -> Result<Vec<u32>, coordinode_core::graph::intern::DictionaryError> {
        names
            .iter()
            .map(|n| {
                self.0.lookup(n).ok_or_else(|| {
                    coordinode_core::graph::intern::DictionaryError::Registration(
                        "fixed dictionary".into(),
                    )
                })
            })
            .collect()
    }

    fn adopt(
        &self,
        _: &coordinode_core::graph::intern::FieldInterner,
    ) -> Result<(), coordinode_core::graph::intern::DictionaryError> {
        Err(
            coordinode_core::graph::intern::DictionaryError::Registration(
                "fixed dictionary".into(),
            ),
        )
    }

    fn view(
        &self,
    ) -> Result<
        coordinode_core::graph::intern::FieldInterner,
        coordinode_core::graph::intern::DictionaryError,
    > {
        Ok(self.0.clone())
    }
}

struct Fixture {
    engine: Arc<StorageEngine>,
    registry: Arc<VectorIndexRegistry>,
    fields: Arc<FixedFields>,
    field: u32,
    _dir: tempfile::TempDir,
}

fn fixture() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    // Oracle-backed, as every deployment's engine: an apply lands at its
    // commit timestamp, which is what the freshness watermark is in.
    let engine = Arc::new(
        StorageEngine::open_with_oracle(
            &StorageConfig::with_endpoints(vec![EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            )]),
            Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new()),
        )
        .unwrap(),
    );
    let registry = Arc::new(VectorIndexRegistry::new());
    registry.register(
        IndexDescriptor::hnsw(
            "item_emb",
            "Item",
            "embedding",
            VectorIndexConfig {
                dimensions: 4,
                metric: coordinode_core::graph::types::VectorMetric::L2,
                m: 8,
                ef_construction: 32,
                quantization: coordinode_vector::hnsw::QuantizationCodec::None,
                offload_vectors: false,
                ef_search: None,
                rerank_candidates: None,
            },
        )
        .bind(IndexId::from_raw(1), GenerationId::from_raw(1)),
    );
    let mut interner = coordinode_core::graph::intern::FieldInterner::new();
    let field = interner.intern("embedding");
    Fixture {
        engine,
        registry,
        fields: Arc::new(FixedFields(interner)),
        field,
        _dir: dir,
    }
}

impl Fixture {
    fn put_item(&self, id: u64, x: f32) -> Mutation {
        let mut record = NodeRecord::new("Item");
        record
            .props
            .insert(self.field, Value::Vector(vec![x, 0.0, 0.0, 0.0]));
        Mutation::Put {
            partition: PartitionId::Node,
            key: encode_node_key(SHARD, NodeId::from_raw(id)),
            value: record.to_msgpack().unwrap(),
        }
    }

    /// Apply `mutations` as Raft entry `index` at commit timestamp `index`.
    fn apply(&self, index: u64, mutations: &[Mutation]) {
        self.engine
            .apply_raft_proposal(mutations, index, index, 0, |_| false)
            .unwrap();
    }

    fn spawn(&self, capacity: usize) -> VectorIndexWorker {
        let applied = self
            .engine
            .subscribe_applied_retained(Partition::Node, capacity);
        let coverage = Arc::new(coordinode_query::index::IndexCoverage::new(
            applied.position(),
        ));
        self.registry.set_coverage(Arc::clone(&coverage));
        VectorIndexWorker::spawn(
            Arc::clone(&self.engine),
            applied,
            Arc::clone(&self.registry),
            Arc::clone(&self.fields) as Arc<dyn FieldRegistrar>,
            coverage,
            SHARD,
        )
    }

    fn indexed(&self) -> usize {
        let handle = self.registry.get("Item", "embedding").unwrap();
        handle.read().map(|h| h.len()).unwrap_or(0)
    }

    /// Wait until the index holds `n` nodes, up to five seconds.
    fn await_indexed(&self, n: usize) -> usize {
        self.await_indexed_for(n, 100)
    }

    /// Wait until the index holds `n` nodes, up to `polls` × 50 ms.
    fn await_indexed_for(&self, n: usize, polls: usize) -> usize {
        for _ in 0..polls {
            if self.indexed() == n {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        self.indexed()
    }
}

/// Entries applied after the worker started reach the index, and the
/// freshness watermark covers the last one's commit timestamp.
#[test]
fn applied_entries_reach_the_index() {
    let fx = fixture();
    let worker = fx.spawn(1024);

    for id in 1..=20u64 {
        fx.apply(id, &[fx.put_item(id, id as f32)]);
    }

    let indexed = fx.await_indexed(20);
    let watermark = || {
        fx.registry
            .health_snapshot("Item", "embedding")
            .and_then(|h| h.indexed_hlc())
            .unwrap_or(0)
    };
    for _ in 0..100 {
        if watermark() >= 20 {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    worker.shutdown();
    assert_eq!(indexed, 20, "every applied node reaches the index");
    assert!(
        watermark() >= 20,
        "the watermark covers the last applied entry: {}",
        watermark()
    );
    let handle = fx.registry.get("Item", "embedding").unwrap();
    let nearest = handle.read().unwrap().search(&[5.0, 0.0, 0.0, 0.0], 1);
    assert_eq!(nearest.first().map(|r| r.id), Some(5));
}

/// An applied deletion takes the node out of the index, and an applied write
/// that strips the vector does too; the other nodes stay.
#[test]
fn applied_deletions_leave_the_index() {
    let fx = fixture();
    let worker = fx.spawn(1024);
    for id in 1..=5u64 {
        fx.apply(id, &[fx.put_item(id, id as f32)]);
    }
    assert_eq!(fx.await_indexed(5), 5);

    fx.apply(
        6,
        &[Mutation::Delete {
            partition: PartitionId::Node,
            key: encode_node_key(SHARD, NodeId::from_raw(3)),
        }],
    );
    fx.apply(
        7,
        &[Mutation::Put {
            partition: PartitionId::Node,
            key: encode_node_key(SHARD, NodeId::from_raw(4)),
            value: NodeRecord::new("Item").to_msgpack().unwrap(),
        }],
    );

    let indexed = fx.await_indexed(3);
    worker.shutdown();
    assert_eq!(indexed, 3, "the deleted and the stripped node left");
    let handle = fx.registry.get("Item", "embedding").unwrap();
    let graph = handle.read().unwrap();
    for id in [1u64, 2, 5] {
        assert!(graph.contains(id), "node {id} left the index");
    }
}

/// A store replaced by a snapshot is read afresh: the nodes it holds reach
/// the index though no entry carried them.
#[test]
fn a_snapshot_installed_is_rebuilt_from_the_store() {
    let fx = fixture();
    let worker = fx.spawn(1024);

    // Written into the store outside the applies and the commits, as an
    // installed snapshot writes it, with no event per key; then the install
    // is recorded.
    for id in 1..=5u64 {
        fx.engine
            .apply_mutation(&fx.put_item(id, id as f32))
            .unwrap();
    }
    fx.engine.reset_raft_coverage(6, &[]).unwrap();

    let indexed = fx.await_indexed(5);
    worker.shutdown();
    assert_eq!(indexed, 5, "the snapshot's nodes are indexed");
    assert!(
        fx.registry
            .health_snapshot("Item", "embedding")
            .is_some_and(|h| h.is_ready()),
        "the rebuilt index is ready again"
    );
}

/// Entries keep applying for the whole length of an index build and after
/// it: each one is in the index at the end, whether the build folded it
/// (applied while the build watched) or the worker inserted it (applied after
/// the build handed the index over). An entry applied as the build stops
/// watching used to fall between the two.
#[test]
fn every_entry_applied_across_a_build_reaches_the_index() {
    let fx = fixture();
    // Enough nodes that the build is still scanning when the applies start.
    for id in 1..=3000u64 {
        fx.engine
            .apply_proposal_at(&[fx.put_item(id, id as f32)], id)
            .unwrap();
    }
    let worker = fx.spawn(1024);
    let hnsw = fx.registry.get("Item", "embedding").unwrap();
    let health = fx.registry.health_handle("Item", "embedding").unwrap();
    health.report_rebuild_progress(0.0, 0);
    let token = fx.registry.new_build_token();

    let mut next = 100_000u64;
    std::thread::scope(|scope| {
        let build = scope.spawn(|| {
            VectorBuild {
                engine: &fx.engine,
                token: &token,
                shard_id: SHARD,
                targets: &[BuildTarget {
                    hnsw: hnsw.as_ref(),
                    health: health.as_ref(),
                    label: "Item",
                    field_id: fx.field,
                }],
            }
            .run()
        });
        // Bounded so the worker can catch up in the test's time.
        while !build.is_finished() && next < 120_000 {
            fx.apply(next, &[fx.put_item(next, next as f32)]);
            next += 1;
        }
        build.join().unwrap().unwrap();
    });
    // And a stretch after it, maintained by the worker alone.
    for _ in 0..200 {
        fx.apply(next, &[fx.put_item(next, next as f32)]);
        next += 1;
    }

    let expected = 3000 + usize::try_from(next - 100_000).unwrap();
    let indexed = fx.await_indexed_for(expected, 1200);
    worker.shutdown();
    let graph = hnsw.read().unwrap();
    let missing: Vec<u64> = (100_000..next).filter(|&id| !graph.contains(id)).collect();
    assert!(missing.is_empty(), "applied but not indexed: {missing:?}");
    assert_eq!(indexed, expected);
}

/// A field dictionary that can be made unreadable, as a store that cannot be
/// read for a while leaves it.
struct FlakyFields {
    inner: FixedFields,
    failing: std::sync::atomic::AtomicBool,
    /// How many times a view was asked for while failing.
    refused: std::sync::atomic::AtomicUsize,
}

impl FieldRegistrar for FlakyFields {
    fn register(
        &self,
        names: &[&str],
    ) -> Result<Vec<u32>, coordinode_core::graph::intern::DictionaryError> {
        self.inner.register(names)
    }

    fn adopt(
        &self,
        bindings: &coordinode_core::graph::intern::FieldInterner,
    ) -> Result<(), coordinode_core::graph::intern::DictionaryError> {
        self.inner.adopt(bindings)
    }

    fn view(
        &self,
    ) -> Result<
        coordinode_core::graph::intern::FieldInterner,
        coordinode_core::graph::intern::DictionaryError,
    > {
        use std::sync::atomic::Ordering::SeqCst;
        if self.failing.load(SeqCst) {
            self.refused.fetch_add(1, SeqCst);
            return Err(
                coordinode_core::graph::intern::DictionaryError::Registration("unavailable".into()),
            );
        }
        self.inner.view()
    }
}

/// A round the worker could neither fold nor rebuild advances nothing: not
/// the freshness watermark, and not the released position when a later
/// round folds, which would release the failed events with its own and
/// leave their node to an index that never took it.
#[test]
fn a_failed_fold_advances_neither_the_watermark_nor_the_release() {
    use std::sync::atomic::Ordering::SeqCst;

    let fx = fixture();
    let fields = Arc::new(FlakyFields {
        inner: FixedFields(fx.fields.0.clone()),
        failing: std::sync::atomic::AtomicBool::new(true),
        refused: std::sync::atomic::AtomicUsize::new(0),
    });
    let applied = fx.engine.subscribe_applied_retained(Partition::Node, 1024);
    let coverage = Arc::new(coordinode_query::index::IndexCoverage::new(
        applied.position(),
    ));
    fx.registry.set_coverage(Arc::clone(&coverage));
    let worker = VectorIndexWorker::spawn(
        Arc::clone(&fx.engine),
        applied,
        Arc::clone(&fx.registry),
        Arc::clone(&fields) as Arc<dyn FieldRegistrar>,
        Arc::clone(&coverage),
        SHARD,
    );
    let wait = |what: &str, done: &dyn Fn() -> bool| {
        for _ in 0..12_000 {
            if done() {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(done(), "never: {what}");
    };
    let watermark = || {
        fx.registry
            .health_snapshot("Item", "embedding")
            .and_then(|h| h.indexed_hlc())
    };

    // The fold and the rebuild after it both fail.
    fx.apply(1, &[fx.put_item(1, 1.0)]);
    wait("the fold and its rebuild were refused", &|| {
        fields.refused.load(SeqCst) >= 2
    });
    assert_eq!(
        watermark(),
        Some(0),
        "the watermark claims a write the index does not hold"
    );

    fields.failing.store(false, SeqCst);
    fx.apply(2, &[fx.put_item(2, 2.0)]);
    wait("the second write is folded", &|| {
        !coverage.delta(SHARD).contains(NodeId::from_raw(2))
    });
    let indexed = fx
        .registry
        .get("Item", "embedding")
        .unwrap()
        .read()
        .unwrap()
        .contains(1);
    worker.shutdown();
    assert!(
        indexed || coverage.delta(SHARD).contains(NodeId::from_raw(1)),
        "node 1 is neither in the index nor answered from the store"
    );
}

/// The freshness watermark is a read-your-writes fence: a writer whose
/// commit is at or below it is served an index holding that write. A commit
/// that took its timestamp first and lands after a later one must not be
/// covered by the later one's: the watermark stays below it until it has
/// landed and the worker has folded it.
#[test]
fn the_watermark_stays_below_a_commit_still_in_flight() {
    let dir = tempfile::tempdir().unwrap();
    let oracle = Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new());
    let engine = Arc::new(
        StorageEngine::open_with_oracle(
            &StorageConfig::with_endpoints(vec![EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            )]),
            Arc::clone(&oracle),
        )
        .unwrap(),
    );
    // A snapshot steps behind a commit in flight at once.
    engine.set_snapshot_wait_ms(0);
    let fx = Fixture {
        engine,
        ..fixture()
    };
    let worker = fx.spawn(1024);
    let watermark = || {
        fx.registry
            .health_snapshot("Item", "embedding")
            .and_then(|h| h.indexed_hlc())
            .unwrap_or(0)
    };

    // The early commit has its timestamp and holds its key, not yet landed.
    let early_key = encode_node_key(SHARD, NodeId::from_raw(5));
    let (early, admission) = fx
        .engine
        .pending_commits()
        .admit_allocated(
            || oracle.next().as_raw(),
            vec![(Partition::Node, early_key)],
            Vec::new(),
        )
        .expect("admit the early commit");
    // A later commit lands and is folded.
    let late = oracle.next().as_raw();
    fx.engine
        .apply_proposal_at(&[fx.put_item(1, 1.0)], late)
        .unwrap();
    assert_eq!(fx.await_indexed(1), 1);
    // Give a wrongly advanced watermark the time to show.
    std::thread::sleep(Duration::from_millis(100));
    assert!(
        watermark() < early,
        "the watermark {} covers the commit at {early}, which has not landed",
        watermark()
    );

    fx.engine
        .apply_proposal_at(&[fx.put_item(5, 5.0)], early)
        .unwrap();
    drop(admission);
    // Another commit wakes the worker after the early one is visible.
    let after = oracle.next().as_raw();
    fx.engine
        .apply_proposal_at(&[fx.put_item(6, 6.0)], after)
        .unwrap();
    assert_eq!(fx.await_indexed(3), 3);
    for _ in 0..100 {
        if watermark() >= late {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    worker.shutdown();
    assert!(
        watermark() >= late,
        "the watermark never covered the commits that landed: {}",
        watermark()
    );
}

/// A worker that falls behind its queue loses no node: the applies drop the
/// events it could not take and it rebuilds from the store, which holds them.
#[test]
fn a_worker_behind_its_queue_rebuilds_and_loses_nothing() {
    let fx = fixture();
    // Hold the index's write lock so the worker stalls on its first fold
    // while the applies overflow its one-slot queue.
    let handle = fx.registry.get("Item", "embedding").unwrap();
    let held = handle.write().unwrap();
    let worker = fx.spawn(1);
    for id in 1..=50u64 {
        fx.apply(id, &[fx.put_item(id, id as f32)]);
    }
    drop(held);

    let indexed = fx.await_indexed(50);
    worker.shutdown();
    assert_eq!(
        indexed, 50,
        "every node is indexed despite the dropped events"
    );
}
