use std::sync::Arc;
use std::time::{Duration, Instant};

use coordinode_core::graph::intern::FieldInterner;
use coordinode_core::graph::node::{NodeId, NodeRecord};
use coordinode_core::graph::types::Value;
use coordinode_core::txn::proposal::{Mutation, PartitionId};
use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::transaction::Transaction;
use coordinode_vector::health::HealthSignal;

use super::{BuildOutcome, BuildTarget, VectorBuild};
use crate::index::{IndexDefinition, VectorIndexConfig, VectorIndexRegistry};

struct Fixture {
    engine: Arc<StorageEngine>,
    oracle: Arc<TimestampOracle>,
    registry: VectorIndexRegistry,
    field: u32,
    _dir: tempfile::TempDir,
}

/// An oracle-backed engine, where commit timestamps are seqnos as in every
/// deployment, with a 3-dimensional `Doc.embedding` index registered and
/// marked rebuilding, as a build starts it.
fn fixture() -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = Arc::new(TimestampOracle::new());
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
        .expect("open engine"),
    );
    let registry = VectorIndexRegistry::new();
    registry.register(IndexDefinition::hnsw(
        "emb",
        "Doc",
        "embedding",
        VectorIndexConfig {
            dimensions: 3,
            metric: coordinode_core::graph::types::VectorMetric::Cosine,
            m: 16,
            ef_construction: 200,
            quantization: coordinode_vector::hnsw::QuantizationCodec::None,
            offload_vectors: false,
            ef_search: None,
            rerank_candidates: None,
        },
    ));
    registry
        .health_handle("Doc", "embedding")
        .expect("health")
        .report_rebuild_progress(0.0, 0);
    let field = FieldInterner::new().intern("embedding");
    Fixture {
        engine,
        oracle,
        registry,
        field,
        _dir: dir,
    }
}

impl Fixture {
    /// Apply a `Doc` node with `vector` at `commit_ts`, as a Raft apply or a
    /// commit that reserved its timestamp early would.
    fn apply_doc_at(&self, id: u64, vector: [f32; 3], commit_ts: u64) {
        let mut record = NodeRecord::new("Doc");
        record.set(self.field, Value::Vector(vector.to_vec()));
        self.engine
            .apply_proposal_at(
                &[Mutation::Put {
                    partition: PartitionId::Node,
                    key: coordinode_core::graph::node::encode_node_key(0, NodeId::from_raw(id)),
                    value: record.to_msgpack().expect("encode"),
                }],
                commit_ts,
            )
            .expect("apply");
    }

    fn health(&self) -> Arc<HealthSignal> {
        self.registry
            .health_handle("Doc", "embedding")
            .expect("health")
    }

    /// A transaction left open, the way a statement still running is.
    fn open_transaction(&self) -> Transaction<'_> {
        Transaction::begin(&self.engine, Some(&self.oracle), self.oracle.next())
    }

    /// Run a build on its own thread, `during` it on this one, and return
    /// how the build ended.
    fn build_while(&self, during: impl FnOnce(&HealthSignal)) -> Result<BuildOutcome, String> {
        let hnsw = self.registry.get("Doc", "embedding").expect("hnsw");
        let health = self.health();
        let token = self.registry.new_build_token();
        std::thread::scope(|scope| {
            let build = scope.spawn(|| self.build(&hnsw, &health, &token));
            during(&health);
            build.join().expect("build thread")
        })
    }

    fn build(
        &self,
        hnsw: &std::sync::RwLock<coordinode_vector::hnsw::HnswIndex>,
        health: &HealthSignal,
        token: &crate::index::BuildToken,
    ) -> Result<BuildOutcome, String> {
        VectorBuild {
            engine: &self.engine,
            token,
            shard_id: 0,
            targets: &[BuildTarget {
                hnsw,
                health,
                label: "Doc",
                field_id: self.field,
            }],
        }
        .run()
    }

    fn indexed(&self, query: [f32; 3], id: u64) -> bool {
        let hnsw = self.registry.get("Doc", "embedding").expect("hnsw");
        let graph = hnsw.read().expect("hnsw");
        graph.search(&query, 1).iter().any(|r| r.id == id)
    }

    fn graph_len(&self) -> usize {
        let hnsw = self.registry.get("Doc", "embedding").expect("hnsw");
        let graph = hnsw.read().expect("hnsw");
        graph.len()
    }
}

/// Wait until the build has handed maintenance to the writers and is
/// waiting for the transactions older than the handover.
fn await_handover(health: &HealthSignal) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while health.snapshot().is_rebuilding() {
        assert!(Instant::now() < deadline, "the build never handed over");
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// A write that lands while the index is being built reaches it, whatever
/// its commit timestamp. The one here was timestamped before the build
/// started and applies once the build has scanned and handed over: below
/// every snapshot the build took, so no timestamp selects it, and its writer
/// left the vector to the build. A transaction open before the handover
/// keeps the build folding until it ends.
#[test]
fn a_write_landing_below_every_build_snapshot_reaches_the_index() {
    let fx = fixture();
    fx.apply_doc_at(1, [0.0, 1.0, 0.0], fx.oracle.next().as_raw());
    let early = fx.oracle.next().as_raw();
    let older = fx.open_transaction();

    let outcome = fx.build_while(|health| {
        await_handover(health);
        fx.apply_doc_at(7, [1.0, 0.0, 0.0], early);
        drop(older);
    });

    assert_eq!(outcome, Ok(BuildOutcome::Complete { scanned: 1 }));
    assert_eq!(fx.graph_len(), 2, "the scanned node and the late one");
    assert!(fx.indexed([1.0, 0.0, 0.0], 7), "the late write is indexed");
}

/// A partition cleared during the build cannot be listed key by key: the
/// build starts over from a fresh snapshot and indexes what the partition
/// holds afterwards.
// Clearing a partition wholesale is a storage operation no store exposes.
#[allow(clippy::disallowed_types)]
#[test]
fn a_build_whose_partition_is_replaced_scans_again() {
    let fx = fixture();
    let older = fx.open_transaction();

    let outcome = fx.build_while(|health| {
        await_handover(health);
        fx.engine
            .clear_partition(coordinode_storage::engine::partition::Partition::Node)
            .expect("clear");
        fx.apply_doc_at(9, [0.0, 0.0, 1.0], fx.oracle.next().as_raw());
        drop(older);
    });

    assert!(matches!(outcome, Ok(BuildOutcome::Complete { .. })));
    assert!(
        fx.indexed([0.0, 0.0, 1.0], 9),
        "the node written after the clear is indexed"
    );
}

/// A cancelled build stops while it waits for older transactions, rather
/// than holding the tap open until they end.
#[test]
fn a_cancelled_build_stops_while_it_waits() {
    let fx = fixture();
    let _older = fx.open_transaction();
    let hnsw = fx.registry.get("Doc", "embedding").expect("hnsw");
    let health = fx.health();
    let token = fx.registry.new_build_token();

    let outcome = std::thread::scope(|scope| {
        let build = scope.spawn(|| fx.build(&hnsw, &health, &token));
        await_handover(&health);
        token.cancel();
        build.join().expect("build thread")
    });
    assert_eq!(outcome, Ok(BuildOutcome::Cancelled));
}

/// With no transaction older than the handover the build ends on its own,
/// marks the index ready, and publishes a freshness watermark covering the
/// store as it stood.
#[test]
fn a_build_with_nothing_to_wait_for_ends_ready_and_fresh() {
    let fx = fixture();
    fx.apply_doc_at(1, [0.0, 1.0, 0.0], fx.oracle.next().as_raw());
    let before = fx.engine.snapshot();

    let outcome = fx.build_while(|_| {});

    assert_eq!(outcome, Ok(BuildOutcome::Complete { scanned: 1 }));
    let health = fx.health().snapshot();
    assert!(health.is_ready());
    assert!(
        fx.health().indexed_hlc() >= before,
        "the watermark covers every write made before the build"
    );
}
