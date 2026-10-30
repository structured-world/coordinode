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
    let article = coordinode_modality::LocalSchemaStore::new(engine)
        .bind_label("Article")
        .expect("bind label");
    let mut record = coordinode_core::graph::node::NodeRecord::new(article);
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

/// Apply node `id` of type `label` on shard 1 with `title` as entry `index`.
fn apply_titled(
    engine: &StorageEngine,
    index: u64,
    id: u64,
    (label, title_field, title): (&str, u32, &str),
) {
    let label = coordinode_modality::LocalSchemaStore::new(engine)
        .bind_label(label)
        .expect("bind label");
    let mut record = coordinode_core::graph::node::NodeRecord::new(label);
    record.set(
        title_field,
        coordinode_core::graph::types::Value::String(title.into()),
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

/// A node written before a type came to imply the label is folded by the
/// worker only after the catalog changed, before the worker followed the
/// change. The document is judged by the interpretation the index holds, so
/// a reader of the earlier view, whose snapshot does not count the node as
/// written since, is not answered with a document of the later one. Before
/// the fold judged by the latest catalog, the index held the node's document
/// under the earlier interpretation and that reader found the task as a
/// thing.
#[test]
fn a_late_fold_writes_under_the_interpretation_the_index_holds() {
    use coordinode_core::schema::definition::{LabelSchema, PlacementPolicy};
    use coordinode_modality::{LocalSchemaStore, SchemaStore as _};
    use coordinode_query::index::membership::label_carriers;

    let dir = tempfile::tempdir().expect("tempdir");
    let engine = std::sync::Arc::new(engine(&dir));
    let schemas = LocalSchemaStore::new(&engine);
    let mut task = LabelSchema::new("Task", PlacementPolicy::NodeId);
    for schema in [
        LabelSchema::new("Thing", PlacementPolicy::NodeId),
        task.clone(),
    ] {
        schemas.save_label(&schema).expect("declare");
    }
    let registry = std::sync::Arc::new(coordinode_query::index::TextIndexRegistry::new(
        dir.path().join("text"),
    ));
    registry
        .register(
            coordinode_query::index::IndexDescriptor::text(
                "thing_title",
                "Thing",
                vec!["title".into()],
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
    let title = interner.intern("title");
    let sub = engine.subscribe_applied_retained(Partition::Node, 64);
    apply_titled(&engine, 1, 1, ("Thing", title, "rust graph"));
    apply_titled(&engine, 2, 7, ("Task", title, "rust tools"));
    let before = label_carriers(&engine, "Thing").expect("carriers");
    registry
        .rebuild_index("Thing", "title", || {
            coordinode_query::index::text_registry::stored_texts(
                &engine, 1, &interner, "Thing", "title", 0,
            )
        })
        .expect("build");

    // The type comes to imply the label in the catalog, and the worker
    // folds the task's earlier write before following that.
    task.set_implies(vec!["Thing".into()]);
    task.schema_revision = 2;
    schemas.save_label(&task).expect("implies");
    let worker = super::Worker {
        engine: std::sync::Arc::clone(&engine),
        position: sub.position(),
        coverage: std::sync::Arc::new(IndexCoverage::new(sub.position())),
        applied: sub,
        registry: std::sync::Arc::clone(&registry),
        fields: std::sync::Arc::new(FlakyFields {
            fields: interner,
            failing: std::sync::atomic::AtomicBool::new(false),
            refused: std::sync::atomic::AtomicUsize::new(0),
        }),
        shard_id: 1,
        stop: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
    };
    worker
        .fold_nodes(&[NodeId::from_raw(7)])
        .expect("fold the late write");

    let held = registry
        .with_index("Thing", "title", |index, held| {
            Ok((index.contains(7)?, held.clone()))
        })
        .expect("index")
        .expect("registered");
    assert_eq!(
        held.1,
        coordinode_query::index::selection::Held::built(before),
        "the index still holds the earlier interpretation"
    );
    assert!(
        !held.0,
        "the task's document is not in an index holding the interpretation without it"
    );
}

/// The text index definition on `:Thing(title)`.
fn thing_title() -> coordinode_query::index::IndexDefinition {
    coordinode_query::index::IndexDescriptor::text(
        "thing_title",
        "Thing",
        vec!["title".into()],
        coordinode_query::index::TextIndexConfig {
            fields: Default::default(),
            default_language: "english".into(),
            language_override_property: "_language".into(),
        },
    )
    .bind(
        coordinode_query::index::IndexId::from_raw(1),
        coordinode_query::index::GenerationId::from_raw(1),
    )
}

/// A store holding a thing and a task, `Task` not implying `Thing`, a text
/// index on `:Thing(title)` built over it, and a worker over both, not
/// running: the test drives each step itself.
struct Rig {
    _dir: tempfile::TempDir,
    engine: std::sync::Arc<StorageEngine>,
    registry: std::sync::Arc<coordinode_query::index::TextIndexRegistry>,
    worker: super::Worker,
    /// `Thing`'s carriers before and after `Task` comes to imply it.
    carriers: (
        Option<coordinode_query::index::membership::LabelCarriers>,
        Option<coordinode_query::index::membership::LabelCarriers>,
    ),
}

impl Rig {
    fn new() -> Self {
        use coordinode_core::schema::definition::{LabelSchema, PlacementPolicy};
        use coordinode_modality::{LocalSchemaStore, SchemaStore as _};
        use coordinode_query::index::membership::{LabelCarriers, label_carriers};

        let dir = tempfile::tempdir().expect("tempdir");
        let engine = std::sync::Arc::new(engine(&dir));
        let schemas = LocalSchemaStore::new(&engine);
        for name in ["Thing", "Task"] {
            schemas
                .save_label(&LabelSchema::new(name, PlacementPolicy::NodeId))
                .expect("declare");
        }
        let registry = std::sync::Arc::new(coordinode_query::index::TextIndexRegistry::new(
            dir.path().join("text"),
        ));
        registry.register(thing_title()).expect("register");
        let mut interner = coordinode_core::graph::intern::FieldInterner::new();
        let title = interner.intern("title");
        let sub = engine.subscribe_applied_retained(Partition::Node, 64);
        apply_titled(&engine, 1, 1, ("Thing", title, "rust graph"));
        apply_titled(&engine, 2, 7, ("Task", title, "rust tools"));
        let before = label_carriers(&engine, "Thing").expect("carriers");
        let task = schemas.label_id("Task").expect("id").expect("bound");
        let after = before.clone().map(|c| LabelCarriers {
            implying: vec![task],
            ..c
        });
        registry
            .rebuild_index("Thing", "title", || {
                coordinode_query::index::text_registry::stored_texts(
                    &engine, 1, &interner, "Thing", "title", 0,
                )
            })
            .expect("build");
        let worker = super::Worker {
            engine: std::sync::Arc::clone(&engine),
            position: sub.position(),
            coverage: std::sync::Arc::new(IndexCoverage::new(sub.position())),
            applied: sub,
            registry: std::sync::Arc::clone(&registry),
            fields: std::sync::Arc::new(FlakyFields {
                fields: interner,
                failing: std::sync::atomic::AtomicBool::new(false),
                refused: std::sync::atomic::AtomicUsize::new(0),
            }),
            shard_id: 1,
            stop: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        };
        Self {
            _dir: dir,
            engine,
            registry,
            worker,
            carriers: (before, after),
        }
    }

    /// Rebuild the index from the store, as an index build on another
    /// thread does.
    fn rebuild_elsewhere(&self) {
        let interner = self.worker.fields.view().expect("fields");
        self.registry
            .rebuild_index("Thing", "title", || {
                coordinode_query::index::text_registry::stored_texts(
                    &self.engine,
                    1,
                    &interner,
                    "Thing",
                    "title",
                    0,
                )
            })
            .expect("rebuild");
    }

    fn held(&self) -> (coordinode_query::index::selection::Held, u64) {
        self.registry.held_at("Thing", "title").expect("registered")
    }
}

/// The worker decides to settle a change put in force at one revision of
/// the index's interpretations; before it settles, another thread rebuilds
/// the index. The settling is refused: the rebuilt documents and what they
/// hold stay, rather than a change decided against the documents before
/// being applied to them.
#[test]
fn a_settle_decided_before_a_rebuild_changes_nothing() {
    use coordinode_query::index::selection::Held;
    let rig = Rig::new();
    let (before, after) = rig.carriers.clone();
    let interner = rig.worker.fields.view().expect("fields");
    // The index follows the change: the task's document is kept as gained.
    let ((), _) = rig
        .registry
        .with_index_at("Thing", "title", None, |_, held| {
            *held = Held {
                base: Some(before.clone()),
                candidate: Some((after.clone(), true)),
            };
            Ok(())
        })
        .expect("follow")
        .expect("registered");
    let read = coordinode_storage::engine::transaction::Transaction::new(
        &rig.engine,
        None,
        coordinode_core::txn::timestamp::Timestamp::ZERO,
        None,
    );
    let nodes =
        coordinode_query::index::text_registry::read_nodes(&read, 1, &[NodeId::from_raw(7)])
            .expect("read");
    assert!(
        rig.registry
            .apply_rows(("Thing", "title"), &interner, (&nodes, 0), None, |_| Some(
                (before.clone(), Some(after.clone()))
            ))
            .expect("gain")
    );
    let (_, decided) = rig.held();

    rig.rebuild_elsewhere();
    let rebuilt = rig.held();
    rig.worker
        .settle(&thing_title(), "title", (true, &after), decided)
        .expect("settle");

    assert_eq!(rig.held(), rebuilt, "the rebuild's interpretation stands");
    assert_eq!(rebuilt.0, Held::built(before));
    let thing = rig
        .registry
        .with_index("Thing", "title", |index, _| index.contains(1))
        .expect("read")
        .expect("registered");
    assert!(thing, "the rebuilt document of the thing stays");
}

/// The worker follows a change that makes `Task` imply `Thing`, which is
/// then withdrawn; once the worker has decided to settle the withdrawal and
/// before it drops anything, the implication is put in force by another
/// path and another thread rebuilds the index, the task's document now
/// shared. Whatever the worker does after, the index holds every document
/// of the interpretation it says it holds. Before settling was one guarded
/// step, it removed the documents it had listed before the rebuild, the
/// task's among them, from the rebuilt index.
#[test]
fn a_settle_raced_by_a_rebuild_keeps_the_rebuilt_documents() {
    use coordinode_core::schema::definition::{LabelCandidate, LabelSchema, PlacementPolicy};
    use coordinode_modality::{LocalSchemaStore, SchemaStore as _};
    let rig = Rig::new();
    let (_, after) = rig.carriers.clone();
    let schemas = LocalSchemaStore::new(&rig.engine);
    let task = |implies: Option<Vec<String>>, candidate: Option<Vec<String>>, revision| {
        let mut schema = LabelSchema::new("Task", PlacementPolicy::NodeId);
        if let Some(implies) = implies {
            schema.set_implies(implies);
        }
        if candidate.is_some() {
            schema.set_candidate(LabelCandidate {
                properties: Default::default(),
                mode: schema.mode,
                implies: candidate,
                operation: coordinode_query::index::GenerationId::from_raw(40),
            });
        }
        schema.schema_revision = revision;
        schema
    };
    // The change is certified, and the worker keeps the task apart for it.
    schemas
        .save_label(&task(None, Some(vec!["Thing".into()]), 2))
        .expect("propose");
    rig.worker.sync_selections().expect("prepare");
    assert_eq!(
        rig.registry
            .held("Thing", "title")
            .expect("registered")
            .candidate,
        Some((after.clone(), true)),
        "prepared"
    );

    // Withdrawn; while the worker settles that, the implication is put in
    // force elsewhere and the index rebuilt under it.
    schemas.save_label(&task(None, None, 3)).expect("withdraw");
    let (engine, registry) = (
        std::sync::Arc::clone(&rig.engine),
        std::sync::Arc::clone(&rig.registry),
    );
    let interner = rig.worker.fields.view().expect("fields");
    *super::SETTLE_DECIDED.lock() = Some(Box::new(move || {
        LocalSchemaStore::new(&engine)
            .save_label(&task(Some(vec!["Thing".into()]), None, 4))
            .expect("put in force elsewhere");
        registry
            .rebuild_index("Thing", "title", || {
                coordinode_query::index::text_registry::stored_texts(
                    &engine, 1, &interner, "Thing", "title", 0,
                )
            })
            .expect("rebuild");
    }));
    rig.worker.sync_selections().expect("settle");
    assert!(
        super::SETTLE_DECIDED.lock().is_none(),
        "the worker settled the withdrawal"
    );

    let (held, task_indexed) = rig
        .registry
        .with_index("Thing", "title", |index, held| {
            Ok((held.clone(), index.contains(7)?))
        })
        .expect("read")
        .expect("registered");
    assert_eq!(held.base, Some(after), "the rebuild's interpretation");
    assert!(
        task_indexed,
        "the task is a thing under it, and its document is there"
    );
}

/// A rebuild leaving the index holding the very interpretations it held
/// before still moves their revision: a preparation decided before it is
/// refused, so equal interpretations never pass for unchanged documents.
#[test]
fn a_rebuild_to_equal_interpretations_refuses_a_step_decided_before() {
    let rig = Rig::new();
    let (before, after) = rig.carriers.clone();
    let interner = rig.worker.fields.view().expect("fields");
    let (held, decided) = rig.held();

    rig.rebuild_elsewhere();
    let (rebuilt, revision) = rig.held();
    assert_eq!(rebuilt, held, "the rebuild holds the same interpretation");
    assert!(revision > decided);

    rig.worker
        .prepare(
            &thing_title(),
            "title",
            (&before, &after),
            decided,
            &interner,
        )
        .expect("prepare");
    assert_eq!(rig.held(), (rebuilt, revision), "nothing was begun");
}

/// The index is dropped and created again under the same label and
/// property; a step decided against the dropped one is refused on the new
/// one, whose interpretations start their own revision above it.
#[test]
fn a_recreated_index_refuses_a_step_decided_on_the_dropped_one() {
    let rig = Rig::new();
    let (before, after) = rig.carriers.clone();
    let interner = rig.worker.fields.view().expect("fields");
    let (_, decided) = rig.held();

    rig.registry.unregister("Thing", "title");
    rig.registry
        .register(thing_title())
        .expect("register again");
    let (fresh, revision) = rig.held();
    assert!(revision > decided);

    rig.worker
        .prepare(
            &thing_title(),
            "title",
            (&before, &after),
            decided,
            &interner,
        )
        .expect("prepare");
    rig.worker
        .settle(&thing_title(), "title", (true, &after), decided)
        .expect("settle");
    assert_eq!(rig.held(), (fresh, revision), "the new index is untouched");
}
