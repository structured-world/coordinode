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
use crate::index::{VectorIndexConfig, VectorIndexRegistry};

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
    registry.register(
        crate::index::IndexDescriptor::hnsw(
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
        )
        .bind(
            crate::index::IndexId::from_raw(1),
            crate::index::GenerationId::from_raw(1),
        ),
    );
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
        self.apply_at(put(0, id, &record("Doc", self.field, vector)), commit_ts);
    }

    /// Apply `record` as node `id` of `shard`, at a fresh commit timestamp.
    fn apply_record(&self, shard: u16, id: u64, record: &NodeRecord) {
        self.apply_at(put(shard, id, record), self.oracle.next().as_raw());
    }

    /// Delete node `id` of shard 0.
    fn delete_node(&self, id: u64) {
        self.apply_at(
            Mutation::Delete {
                partition: PartitionId::Node,
                key: coordinode_core::graph::node::encode_node_key(0, NodeId::from_raw(id)),
            },
            self.oracle.next().as_raw(),
        );
    }

    fn apply_at(&self, mutation: Mutation, commit_ts: u64) {
        self.engine
            .apply_proposal_at(&[mutation], commit_ts)
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

/// A node of `label` carrying `vector` in `field`.
fn record(label: &str, field: u32, vector: [f32; 3]) -> NodeRecord {
    let mut record = NodeRecord::new(label);
    record.set(field, Value::Vector(vector.to_vec()));
    record
}

fn put(shard: u16, id: u64, record: &NodeRecord) -> Mutation {
    Mutation::Put {
        partition: PartitionId::Node,
        key: coordinode_core::graph::node::encode_node_key(shard, NodeId::from_raw(id)),
        value: record.to_msgpack().expect("encode"),
    }
}

/// Wait until the build has handed maintenance to the writers and is
/// waiting for the transactions older than the handover.
fn await_handover(health: &HealthSignal) {
    // The handover follows the scan, which inserts every vector into the graph
    // of an unoptimised build: seconds alone, several times that on a CI host
    // shared with other builds. The bound only turns a hang into a failure.
    let deadline = Instant::now() + Duration::from_secs(60);
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

/// A build finishes while writes never stop. Waiting for the write tap to
/// run dry before handing over, or before closing, would wait forever when
/// the partition is written without pause, and the index would stay
/// rebuilding for as long as the load lasted. The writer here maintains the
/// index the way the write path does: it inserts its own vector, which the
/// registry leaves to the build while the index is rebuilding.
#[test]
fn a_build_finishes_under_writes_that_never_pause() {
    let fx = fixture();
    // Directions spread over the sphere: under cosine, vectors along one
    // growing coordinate are near-duplicates, and every insert degenerates.
    let spread = |id: u64| {
        [
            (id % 97) as f32 + 1.0,
            (id % 89) as f32 - 44.0,
            (id % 83) as f32 - 41.0,
        ]
    };
    for id in 1..=2000u64 {
        fx.apply_doc_at(id, spread(id), fx.oracle.next().as_raw());
    }
    let done = std::sync::atomic::AtomicBool::new(false);
    // Only a guard against a test that never ends: the property is that the
    // build finishes while the writer is still writing, which a build waiting
    // for a quiet tap never does, however long the writer is given.
    let deadline = Instant::now() + Duration::from_secs(120);

    let (outcome, written) = std::thread::scope(|scope| {
        let writer = scope.spawn(|| {
            let mut id = 1_000_000u64;
            while !done.load(std::sync::atomic::Ordering::Relaxed) && Instant::now() < deadline {
                let vector = spread(id);
                fx.apply_doc_at(id, vector, fx.oracle.next().as_raw());
                fx.registry
                    .on_vector_written("Doc", NodeId::from_raw(id), "embedding", &vector);
                id += 1;
            }
            id - 1_000_000
        });
        let outcome = fx.build_while(|_| {});
        done.store(true, std::sync::atomic::Ordering::Relaxed);
        (outcome, writer.join().expect("writer"))
    });

    assert!(
        Instant::now() < deadline,
        "the build ran until the writer gave up: it never finished under load"
    );
    assert!(matches!(outcome, Ok(BuildOutcome::Complete { .. })));
    assert!(written > 0, "the writer never ran beside the build");
    assert!(fx.health().snapshot().is_ready());
}

/// Thousands of writes landing after the handover are folded in more than one
/// chunk: every one of them is a member when the build ends, beside every
/// scanned node.
#[test]
fn every_node_written_after_the_handover_is_a_member() {
    let fx = fixture();
    let v = |id: u64| [1.0, id as f32, (id % 7) as f32];
    for id in 1..=3000u64 {
        fx.apply_doc_at(id, v(id), fx.oracle.next().as_raw());
    }
    let older = fx.open_transaction();

    let outcome = fx.build_while(|health| {
        await_handover(health);
        for id in 3001..=5000u64 {
            fx.apply_doc_at(id, v(id), fx.oracle.next().as_raw());
        }
        drop(older);
    });

    assert_eq!(outcome, Ok(BuildOutcome::Complete { scanned: 3000 }));
    let hnsw = fx.registry.get("Doc", "embedding").expect("hnsw");
    let graph = hnsw.read().expect("graph");
    let absent: Vec<u64> = (1..=5000).filter(|&id| !graph.contains(id)).collect();
    assert!(absent.is_empty(), "not in the graph: {absent:?}");
    assert_eq!(graph.len(), 5000);
}

/// A node that stops being a member while the build runs (deleted,
/// relabelled, or stripped of its vector) is taken out of the graph, never
/// re-inserted from what the tap delivered; a node that stays a member stays.
#[test]
fn a_node_leaving_the_index_during_the_build_is_removed() {
    let fx = fixture();
    for id in 1..=4 {
        fx.apply_doc_at(id, [0.0, 1.0, id as f32], fx.oracle.next().as_raw());
    }
    let older = fx.open_transaction();

    let outcome = fx.build_while(|health| {
        await_handover(health);
        fx.delete_node(1);
        fx.apply_record(0, 2, &record("Other", fx.field, [0.0, 1.0, 2.0]));
        fx.apply_record(0, 3, &NodeRecord::new("Doc"));
        drop(older);
    });

    assert_eq!(outcome, Ok(BuildOutcome::Complete { scanned: 4 }));
    let hnsw = fx.registry.get("Doc", "embedding").expect("hnsw");
    let graph = hnsw.read().expect("graph");
    for (id, why) in [(1, "deleted"), (2, "relabelled"), (3, "stripped")] {
        assert!(!graph.contains(id), "the {why} node {id} is still indexed");
    }
    assert!(graph.contains(4), "the untouched node left the index");
    assert_eq!(graph.len(), 1);
}

/// A vector rewritten while the build runs ends up in the index at its new
/// value: the fold reads the record as it stands, not as it was scanned.
#[test]
fn a_vector_rewritten_during_the_build_is_indexed_at_its_new_value() {
    let fx = fixture();
    fx.apply_doc_at(1, [0.0, 1.0, 0.0], fx.oracle.next().as_raw());
    fx.apply_doc_at(2, [0.0, 0.0, 1.0], fx.oracle.next().as_raw());
    let older = fx.open_transaction();

    let outcome = fx.build_while(|health| {
        await_handover(health);
        fx.apply_record(0, 1, &record("Doc", fx.field, [1.0, 0.0, 0.0]));
        drop(older);
    });

    assert!(matches!(outcome, Ok(BuildOutcome::Complete { .. })));
    assert!(fx.indexed([1.0, 0.0, 0.0], 1), "found at its new value");
    let hnsw = fx.registry.get("Doc", "embedding").expect("hnsw");
    assert_eq!(hnsw.read().expect("graph").len(), 2, "still a member, once");
}

/// The build indexes one shard: a node another shard owns, landing while it
/// runs, is delivered by the tap and left out.
#[test]
fn a_write_to_another_shard_is_not_folded() {
    let fx = fixture();
    fx.apply_doc_at(1, [0.0, 1.0, 0.0], fx.oracle.next().as_raw());
    let older = fx.open_transaction();

    let outcome = fx.build_while(|health| {
        await_handover(health);
        fx.apply_record(1, 5, &record("Doc", fx.field, [1.0, 0.0, 0.0]));
        drop(older);
    });

    assert_eq!(outcome, Ok(BuildOutcome::Complete { scanned: 1 }));
    assert_eq!(fx.graph_len(), 1, "only this shard's node");
}

/// A build cancelled before it reaches the tap stops inside the scan, at the
/// first progress check, instead of reading the whole shard.
#[test]
fn a_cancelled_build_stops_during_the_scan() {
    let fx = fixture();
    for id in 1..=3000u64 {
        fx.apply_doc_at(id, [1.0, id as f32, 0.0], fx.oracle.next().as_raw());
    }
    let hnsw = fx.registry.get("Doc", "embedding").expect("hnsw");
    let health = fx.health();
    let token = fx.registry.new_build_token();
    token.cancel();

    assert_eq!(
        fx.build(&hnsw, &health, &token),
        Ok(BuildOutcome::Cancelled)
    );
    assert!(
        fx.graph_len() < 3000,
        "the scan went on after the cancellation: {} inserted",
        fx.graph_len()
    );
}

/// Several indexes over one shard share a build: one scan fills each with its
/// own label's vectors, and a write landing during the build reaches the
/// index it belongs to and no other.
#[test]
fn one_build_fills_every_index_of_the_shard() {
    let fx = fixture();
    let mut interner = FieldInterner::new();
    assert_eq!(interner.intern("embedding"), fx.field);
    let pixels = interner.intern("pixels");
    fx.registry.register_for_build(
        crate::index::IndexDescriptor::hnsw(
            "img",
            "Img",
            "pixels",
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
        )
        .bind(
            crate::index::IndexId::from_raw(2),
            crate::index::GenerationId::from_raw(2),
        ),
        None,
    );
    fx.apply_doc_at(1, [0.0, 1.0, 0.0], fx.oracle.next().as_raw());
    fx.apply_record(0, 2, &record("Img", pixels, [0.0, 0.0, 1.0]));

    let docs = fx.registry.get("Doc", "embedding").expect("docs");
    let imgs = fx.registry.get("Img", "pixels").expect("imgs");
    let doc_health = fx.health();
    let img_health = fx.registry.health_handle("Img", "pixels").expect("health");
    let token = fx.registry.new_build_token();
    let older = fx.open_transaction();

    let outcome = std::thread::scope(|scope| {
        let build = scope.spawn(|| {
            VectorBuild {
                engine: &fx.engine,
                token: &token,
                shard_id: 0,
                targets: &[
                    BuildTarget {
                        hnsw: &docs,
                        health: &doc_health,
                        label: "Doc",
                        field_id: fx.field,
                    },
                    BuildTarget {
                        hnsw: &imgs,
                        health: &img_health,
                        label: "Img",
                        field_id: pixels,
                    },
                ],
            }
            .run()
        });
        await_handover(&doc_health);
        fx.apply_record(0, 3, &record("Img", pixels, [1.0, 0.0, 0.0]));
        drop(older);
        build.join().expect("build thread")
    });

    assert_eq!(outcome, Ok(BuildOutcome::Complete { scanned: 2 }));
    let docs = docs.read().expect("docs");
    let imgs = imgs.read().expect("imgs");
    assert_eq!(docs.len(), 1, "only the Doc node");
    assert_eq!(imgs.len(), 2, "the scanned Img node and the late one");
    assert!(imgs.search(&[1.0, 0.0, 0.0], 1).iter().any(|r| r.id == 3));
    assert!(img_health.snapshot().is_ready());
}
