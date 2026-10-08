use coordinode_core::graph::node::{NodeId, encode_node_key};
use coordinode_core::txn::proposal::{Mutation, PartitionId};
use coordinode_query::index::{IndexCoverage, IndexDelta};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

fn engine(dir: &tempfile::TempDir) -> StorageEngine {
    StorageEngine::open(&StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]))
    .expect("open")
}

/// Apply a write of node `id` on `shard` as entry `index`.
fn apply(engine: &StorageEngine, index: u64, shard: u16, id: u64) {
    engine
        .apply_raft_proposal(
            &[Mutation::Put {
                partition: PartitionId::Node,
                key: encode_node_key(shard, NodeId::from_raw(id)),
                value: b"v".to_vec(),
            }],
            10 + index,
            index,
            0,
            |_| false,
        )
        .expect("apply");
}

fn nodes(ids: &[u64]) -> IndexDelta {
    IndexDelta::Nodes(ids.iter().map(|id| NodeId::from_raw(*id)).collect())
}

/// A search learns the nodes of its shard written by entries the worker has
/// not released, whether the worker has taken them or not, and stops seeing
/// them once they are released.
#[test]
fn the_delta_names_the_nodes_of_unreleased_entries() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = engine(&dir);
    let sub = engine.subscribe_applied_retained(Partition::Node, 16);
    let coverage = IndexCoverage::new(sub.position());
    assert_eq!(coverage.delta(1), nodes(&[]));

    apply(&engine, 1, 1, 7);
    apply(&engine, 2, 2, 8);
    apply(&engine, 3, 1, 9);
    assert_eq!(
        coverage.delta(1),
        nodes(&[7, 9]),
        "shard 2's node is not ours"
    );

    // Taken but not folded: still the search's to answer.
    let _ = sub.try_next();
    assert_eq!(coverage.delta(1), nodes(&[7, 9]));
    coverage.release(1);
    assert_eq!(coverage.delta(1), nodes(&[9]));
    let _ = sub.try_next();
    let _ = sub.try_next();
    coverage.release(3);
    assert!(coverage.delta(1).is_empty());
}

/// A field dictionary that can be made unreadable, as a store that cannot be
/// read for a while leaves it.
struct FlakyFields {
    fields: coordinode_core::graph::intern::FieldInterner,
    failing: std::sync::atomic::AtomicBool,
    /// How many times a view was asked for while failing.
    refused: std::sync::atomic::AtomicUsize,
}

impl coordinode_core::graph::intern::FieldRegistrar for FlakyFields {
    fn register(
        &self,
        _: &[&str],
    ) -> Result<Vec<u32>, coordinode_core::graph::intern::DictionaryError> {
        Err(coordinode_core::graph::intern::DictionaryError::Malformed(
            "read-only in this test".into(),
        ))
    }

    fn adopt(
        &self,
        _: &coordinode_core::graph::intern::FieldInterner,
    ) -> Result<(), coordinode_core::graph::intern::DictionaryError> {
        Err(coordinode_core::graph::intern::DictionaryError::Malformed(
            "read-only in this test".into(),
        ))
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
            return Err(coordinode_core::graph::intern::DictionaryError::Malformed(
                "unavailable".into(),
            ));
        }
        Ok(self.fields.clone())
    }
}

/// Apply node `id` of :Article on shard 1 with `body` as entry `index`.
fn apply_article(engine: &StorageEngine, index: u64, id: u64, body_field: u32, body: &str) {
    let mut record = coordinode_core::graph::node::NodeRecord::new("Article");
    record.set(
        body_field,
        coordinode_core::graph::types::Value::String(body.into()),
    );
    engine
        .apply_raft_proposal(
            &[Mutation::Put {
                partition: PartitionId::Node,
                key: encode_node_key(1, NodeId::from_raw(id)),
                value: record.to_msgpack().expect("encode"),
            }],
            10 + index,
            index,
            0,
            |_| false,
        )
        .expect("apply");
}

/// Wait until `done` holds, failing the test after a minute.
fn eventually(what: &str, done: impl Fn() -> bool) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while !done() {
        assert!(std::time::Instant::now() < deadline, "never: {what}");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

/// A batch the worker could neither fold nor rebuild over stays unreleased
/// when a later batch folds: releasing that one releases every event below
/// it, and the failed batch's node would then be answered by an index that
/// never took its text.
#[test]
fn a_failed_fold_is_not_released_by_the_next_one() {
    use std::sync::atomic::Ordering::SeqCst;

    let dir = tempfile::tempdir().expect("tempdir");
    let engine = std::sync::Arc::new(engine(&dir));
    let registry = std::sync::Arc::new(coordinode_query::index::TextIndexRegistry::new(
        dir.path().join("text"),
    ));
    registry
        .register(
            coordinode_query::index::IndexDescriptor::text(
                "article_body",
                "Article",
                vec!["body".into()],
                coordinode_query::index::TextIndexConfig {
                    fields: Default::default(),
                    default_language: "english".into(),
                    language_override_property: "_language".into(),
                },
            )
            .bind(
                coordinode_query::index::IndexId::from_raw(1),
                coordinode_query::index::GenerationId::from_raw(1),
            ),
        )
        .expect("register");
    let mut interner = coordinode_core::graph::intern::FieldInterner::new();
    let body = interner.intern("body");
    let fields = std::sync::Arc::new(FlakyFields {
        fields: interner,
        failing: std::sync::atomic::AtomicBool::new(true),
        refused: std::sync::atomic::AtomicUsize::new(0),
    });
    let sub = engine.subscribe_applied_retained(Partition::Node, 64);
    let coverage = std::sync::Arc::new(IndexCoverage::new(sub.position()));
    let worker = super::TextIndexWorker::spawn(
        std::sync::Arc::clone(&engine),
        sub,
        std::sync::Arc::clone(&registry),
        fields.clone(),
        std::sync::Arc::clone(&coverage),
        1,
    );

    // The fold and the rebuild after it both fail.
    apply_article(&engine, 1, 7, body, "first words");
    eventually("the fold and its rebuild were refused", || {
        fields.refused.load(SeqCst) >= 2
    });
    fields.failing.store(false, SeqCst);

    apply_article(&engine, 2, 8, body, "second words");
    eventually("the second write is folded", || {
        !coverage.delta(1).contains(NodeId::from_raw(8))
    });

    let indexed = registry
        .search("Article", "body", "first", 10)
        .expect("index")
        .iter()
        .any(|hit| hit.node_id == 7);
    assert!(
        indexed || coverage.delta(1).contains(NodeId::from_raw(7)),
        "node 7 is neither in the index nor answered from the store"
    );
    worker.shutdown();
}

/// A write dropped from a full queue leaves the delta unknown, so a search
/// answers every node from the store, until the rebuild that follows is
/// released.
#[test]
fn a_dropped_write_makes_the_delta_unknown_until_the_rebuild_is_released() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = engine(&dir);
    let sub = engine.subscribe_applied_retained(Partition::Node, 1);
    let coverage = IndexCoverage::new(sub.position());

    apply(&engine, 1, 1, 7);
    apply(&engine, 2, 1, 8);
    assert_eq!(coverage.delta(1), IndexDelta::Unknown);
    assert!(coverage.delta(1).contains(NodeId::from_raw(12345)));

    let covered = sub.position().delivered();
    let _ = sub.try_next();
    let _ = sub.try_next();
    coverage.release(covered);
    assert!(coverage.delta(1).is_empty());
}
