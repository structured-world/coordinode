use super::*;
use coordinode_core::schema::definition::{PlacementPolicy, PropertyDef, PropertyType};

/// Logic-test fixture (memory backing, env-flippable via
/// `COORDINODE_TEST_BACKEND`). Schema CRUD has no persistence
/// semantics.
fn open_engine() -> coordinode_test_fixtures::EngineFixture {
    coordinode_test_fixtures::engine_for_logic()
}

fn sample_label() -> LabelSchema {
    let mut schema = LabelSchema::new("User", PlacementPolicy::NodeId);
    schema.add_property(PropertyDef::new("email", PropertyType::String).not_null());
    schema
}

fn sample_edge_type() -> EdgeTypeSchema {
    EdgeTypeSchema::new("KNOWS")
}

#[test]
fn round_trip_label_schema() {
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalSchemaStore::new(engine);
    let schema = sample_label();

    assert!(
        store.load_label("User").expect("load none").is_none(),
        "label must not exist before save"
    );

    store.save_label(&schema).expect("save");
    let loaded = store
        .load_label("User")
        .expect("load some")
        .expect("Some(schema)");
    assert_eq!(loaded.name, schema.name);
    assert_eq!(loaded.schema_revision, schema.schema_revision);
    assert_eq!(loaded.properties.len(), schema.properties.len());
}

#[test]
fn save_label_is_atomic_pointer_and_body() {
    // The save path writes body + pointer in one WriteBatch — a
    // reader can never observe a pointer naming a missing
    // revision. Smoke check: after save, both keys exist; after a
    // second save with a bumped revision, pointer points to the
    // newer revision AND both bodies are still readable (revision
    // history is preserved).
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalSchemaStore::new(engine);

    let mut v1 = sample_label();
    v1.schema_revision = 1;
    store.save_label(&v1).expect("save v1");

    let mut v2 = sample_label();
    v2.schema_revision = 2;
    store.save_label(&v2).expect("save v2");

    // Current load returns v2.
    let cur = store
        .load_label("User")
        .expect("load")
        .expect("Some(schema)");
    assert_eq!(cur.schema_revision, 2);

    // v1 body still readable through its revisioned key, under the label's
    // identity.
    let id = store.label_id("User").expect("binding").expect("bound");
    let v1_bytes = engine
        .get(Partition::Schema, &encode_label_schema_key(id, 1))
        .expect("get")
        .expect("v1 body");
    let v1_loaded = LabelSchema::from_msgpack(&v1_bytes).expect("decode v1");
    assert_eq!(v1_loaded.schema_revision, 1);
}

/// A label schema read while revisions of it commit sees one whole
/// revision: the pointer and the body it names are read from one snapshot,
/// so a commit applying between the two reads cannot leave the reader with a
/// pointer to a body it does not see.
#[test]
fn a_label_read_during_revision_commits_sees_a_whole_revision() {
    use coordinode_core::txn::timestamp::TimestampOracle;
    use coordinode_core::txn::write_concern::WriteConcern;
    use coordinode_storage::engine::config::{
        Durability, EndpointConfig, Media, StorageConfig, Tier,
    };
    use coordinode_storage::engine::transaction::CommitContext;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

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
    let done = AtomicBool::new(false);
    std::thread::scope(|s| {
        s.spawn(|| {
            let store = LocalSchemaStore::new(&engine);
            let wc = WriteConcern::majority();
            for revision in 1..=2_000 {
                let mut schema = sample_label();
                schema.schema_revision = revision;
                let mut txn = Transaction::begin(&engine, Some(&oracle), oracle.next());
                store.save_label_txn(&mut txn, &schema).expect("stage");
                txn.commit(&CommitContext {
                    write_concern: &wc,
                    pipeline: None,
                    id_gen: None,
                })
                .expect("commit revision");
            }
            done.store(true, Ordering::Release);
        });
        let store = LocalSchemaStore::new(&engine);
        while !done.load(Ordering::Acquire) {
            store
                .load_label("User")
                .expect("a whole revision or none, never a torn one");
        }
    });
}

#[test]
fn corrupt_pointer_surfaces_as_decode_error() {
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalSchemaStore::new(engine);
    // Inject a corrupt pointer (3 bytes instead of 8).
    engine
        .put(
            Partition::Schema,
            &encode_label_current_revision_key("Corrupt"),
            &[0xff, 0xff, 0xff],
        )
        .expect("inject");
    let err = store.load_label("Corrupt").expect_err("must error");
    assert!(matches!(err, StoreError::Decode { .. }));
}

#[test]
fn pointer_naming_missing_revision_surfaces_as_decode_error() {
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalSchemaStore::new(engine);
    // The binding says revision 7 is declared, but no body is at rev 7.
    let binding = coordinode_core::schema::label_id::LabelBinding {
        id: coordinode_core::schema::label_id::FIRST_LABEL_ID,
        revision: 7,
        state: coordinode_core::schema::label_id::BindingState::Declared,
    };
    engine
        .put(
            Partition::Schema,
            &encode_label_current_revision_key("Orphan"),
            &binding.encode(),
        )
        .expect("inject");
    let err = store.load_label("Orphan").expect_err("must error");
    assert!(matches!(
        err,
        StoreError::Decode {
            kind: "label schema",
            ..
        }
    ));
}

#[test]
fn edge_type_round_trip() {
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalSchemaStore::new(engine);
    let schema = EdgeTypeSchema::new("KNOWS");

    assert!(store.load_edge_type("KNOWS").expect("none").is_none());
    store.save_edge_type(&schema).expect("save");
    let loaded = store.load_edge_type("KNOWS").expect("some").expect("Some");
    assert_eq!(loaded.name, "KNOWS");
}

/// A temporal edge type whose identity was never resolved is refused at the
/// store, so no writer finds a published temporal type without a key shape;
/// once resolved it is stored with its discriminator.
#[test]
fn a_temporal_edge_type_with_unresolved_identity_is_not_stored() {
    let fx = open_engine();
    let store = LocalSchemaStore::new(&fx.engine);
    let mut schema = EdgeTypeSchema::new("WORKS_AT");
    schema.set_temporal(true);

    let refused = store.save_edge_type(&schema).expect_err("unresolved");
    assert!(
        matches!(&refused, StoreError::Invariant(why) if why.contains("never resolved")),
        "{refused:?}"
    );
    assert!(store.load_edge_type("WORKS_AT").expect("load").is_none());

    schema.resolve_identity(None).expect("resolve");
    store.save_edge_type(&schema).expect("save");
    let loaded = store
        .load_edge_type("WORKS_AT")
        .expect("load")
        .expect("Some");
    assert!(loaded.is_start_identified());
}

#[test]
fn edge_type_revision_bump_preserves_history() {
    // Symmetric to the label revision-bump test: save v1, save
    // v2, current load returns v2, v1 body still readable via
    // its revisioned key.
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalSchemaStore::new(engine);

    let mut v1 = EdgeTypeSchema::new("KNOWS");
    v1.schema_revision = 1;
    store.save_edge_type(&v1).expect("save v1");

    let mut v2 = EdgeTypeSchema::new("KNOWS");
    v2.schema_revision = 2;
    store.save_edge_type(&v2).expect("save v2");

    let cur = store
        .load_edge_type("KNOWS")
        .expect("ok")
        .expect("Some(schema)");
    assert_eq!(cur.schema_revision, 2);

    let v1_bytes = engine
        .get(Partition::Schema, &encode_edge_type_schema_key("KNOWS", 1))
        .expect("ok")
        .expect("v1 body");
    let v1_loaded = EdgeTypeSchema::from_msgpack(&v1_bytes).expect("decode v1");
    assert_eq!(v1_loaded.schema_revision, 1);
}

#[test]
fn legacy_zero_length_edge_marker_loads_as_none() {
    // Pre-DDL deployments wrote a zero-length value at the
    // revisioned key to mark "edge type exists, no schema body".
    // The store must surface this as `None`, not a decode error.
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalSchemaStore::new(engine);
    engine
        .put(
            Partition::Schema,
            &encode_edge_type_current_revision_key("LEGACY"),
            &1u64.to_be_bytes(),
        )
        .expect("pointer");
    engine
        .put(
            Partition::Schema,
            &encode_edge_type_schema_key("LEGACY", 1),
            b"",
        )
        .expect("empty marker");
    assert!(
        store.load_edge_type("LEGACY").expect("ok").is_none(),
        "legacy zero-length marker must decode as None",
    );
}

// ── list_labels / list_edge_types ─────────────────────────────

#[test]
fn list_labels_returns_every_declared_label_at_current_revision() {
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalSchemaStore::new(engine);
    let mut a = sample_label();
    a.name = "User".into();
    a.schema_revision = 1;
    let mut b = sample_label();
    b.name = "Order".into();
    b.schema_revision = 1;
    let mut c = sample_label();
    c.name = "User".into();
    c.schema_revision = 2;
    store.save_label(&a).expect("save User rev 1");
    store.save_label(&b).expect("save Order rev 1");
    store.save_label(&c).expect("save User rev 2");

    let mut listed = store.list_labels().expect("list");
    listed.sort_by(|l, r| l.name.cmp(&r.name));
    assert_eq!(listed.len(), 2, "one schema per declared label");
    assert_eq!(listed[0].name, "Order");
    assert_eq!(listed[0].schema_revision, 1);
    assert_eq!(listed[1].name, "User");
    assert_eq!(listed[1].schema_revision, 2, "current rev for User");
}

#[test]
fn list_labels_empty_when_no_schemas_declared() {
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalSchemaStore::new(engine);
    let listed = store.list_labels().expect("list");
    assert!(listed.is_empty());
}

#[test]
fn list_edge_types_returns_every_declared_edge_type_at_current_revision() {
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalSchemaStore::new(engine);
    let mut knows_v1 = sample_edge_type();
    knows_v1.name = "KNOWS".into();
    knows_v1.schema_revision = 1;
    let mut owns = sample_edge_type();
    owns.name = "OWNS".into();
    owns.schema_revision = 1;
    let mut knows_v2 = sample_edge_type();
    knows_v2.name = "KNOWS".into();
    knows_v2.schema_revision = 2;
    store.save_edge_type(&knows_v1).expect("save KNOWS rev 1");
    store.save_edge_type(&owns).expect("save OWNS rev 1");
    store.save_edge_type(&knows_v2).expect("save KNOWS rev 2");

    let mut listed = store.list_edge_types().expect("list");
    listed.sort_by(|l, r| l.name.cmp(&r.name));
    assert_eq!(listed.len(), 2, "one schema per declared edge type");
    assert_eq!(listed[0].name, "KNOWS");
    assert_eq!(listed[0].schema_revision, 2, "current rev for KNOWS");
    assert_eq!(listed[1].name, "OWNS");
}

#[test]
fn list_edge_types_skips_legacy_zero_length_markers() {
    // Pre-DDL idempotent existence markers shouldn't surface in
    // the listing — `load_edge_type` returns None for them, so
    // `list_edge_types` filters them out.
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalSchemaStore::new(engine);
    // Real schema for KNOWS.
    let mut real = sample_edge_type();
    real.name = "KNOWS".into();
    real.schema_revision = 1;
    store.save_edge_type(&real).expect("save");
    // Legacy marker for LEGACY edge type — bare pointer + empty body.
    engine
        .put(
            Partition::Schema,
            &encode_edge_type_current_revision_key("LEGACY"),
            &1u64.to_be_bytes(),
        )
        .expect("legacy pointer");
    engine
        .put(
            Partition::Schema,
            &encode_edge_type_schema_key("LEGACY", 1),
            b"",
        )
        .expect("legacy empty body");

    let listed = store.list_edge_types().expect("list");
    assert_eq!(
        listed.len(),
        1,
        "legacy marker must be skipped, only KNOWS surfaces",
    );
    assert_eq!(listed[0].name, "KNOWS");
}

#[test]
fn list_edge_type_names_engine_includes_markers_and_dedups_versions() {
    // The engine name lister parses names straight from
    // `schema:edge_type:<name>:<version>` keys, so unlike
    // `list_edge_types` it INCLUDES zero-length existence markers (a
    // reaper must clean adjacency for every registered type) and dedups
    // multiple versions of the same name to one entry.
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalSchemaStore::new(engine);

    let mut knows_v1 = sample_edge_type();
    knows_v1.name = "KNOWS".into();
    knows_v1.schema_revision = 1;
    store.save_edge_type(&knows_v1).expect("save v1");
    let mut knows_v2 = sample_edge_type();
    knows_v2.name = "KNOWS".into();
    knows_v2.schema_revision = 2;
    store.save_edge_type(&knows_v2).expect("save v2");
    // Legacy zero-length marker — present in keys, absent from decoded list.
    engine
        .put(
            Partition::Schema,
            &encode_edge_type_schema_key("LEGACY", 1),
            b"",
        )
        .expect("legacy marker");

    let mut names = store.list_edge_type_names_engine().expect("names");
    names.sort();
    assert_eq!(
        names,
        vec!["KNOWS".to_string(), "LEGACY".to_string()],
        "marker included, versions deduped to one KNOWS entry",
    );
}

/// The cardinality profile is kept beside the definition by both save
/// paths: written while a constraint is declared, gone with the last one,
/// and a withdrawn constraint takes its counts and coverage with it, so one
/// declared again later is rebuilt rather than resumed from counts that
/// stopped changing. Changing only the bound keeps the counts, which do not
/// depend on it.
#[test]
fn edge_type_saves_keep_the_cardinality_profile_and_drop_withdrawn_counts() {
    use coordinode_core::graph::cardinality::{
        CardinalityBound, CardinalityDescriptor, CardinalityMeasure, CardinalityProfile, Direction,
        encode_cardinality_profile_key,
    };
    use coordinode_core::graph::node::NodeId;
    use coordinode_core::txn::timestamp::Timestamp;
    use coordinode_storage::engine::merge::encode_counter_delta;

    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalSchemaStore::new(engine);
    let descriptor = |bound| CardinalityDescriptor {
        edge_type: "KNOWS".into(),
        direction: Direction::Outgoing,
        measure: CardinalityMeasure::DistinctNeighbours,
        bound,
        schema_generation: 1,
    };
    let profile = || -> Option<CardinalityProfile> {
        engine
            .get(Partition::Schema, &encode_cardinality_profile_key("KNOWS"))
            .expect("get")
            .map(|b| CardinalityProfile::from_msgpack(&b).expect("decode"))
    };
    let count_key = descriptor(CardinalityBound::AtMostOne).counter_key(NodeId::from_raw(7));
    let coverage = descriptor(CardinalityBound::AtMostOne).coverage_key();

    // Declared through the batch path.
    let mut schema = sample_edge_type();
    schema
        .declare_cardinality(descriptor(CardinalityBound::AtMostOne))
        .expect("declare");
    store.save_edge_type(&schema).expect("save");
    assert_eq!(
        profile().expect("profile").descriptors,
        vec![descriptor(CardinalityBound::AtMostOne)]
    );
    engine
        .merge(Partition::Counter, &count_key, &encode_counter_delta(1))
        .expect("count");
    engine
        .put(Partition::Schema, &coverage, b"")
        .expect("coverage");

    // Only the bound changes, through the transactional path.
    let mut schema = sample_edge_type();
    schema.schema_revision = 2;
    schema
        .declare_cardinality(descriptor(CardinalityBound::ExactlyOne))
        .expect("declare");
    // A direct transaction applies as it stages, which is all a schema
    // save needs to be observed here.
    let mut txn = Transaction::new(engine, None, Timestamp::from_raw(0), None);
    store.save_edge_type_txn(&mut txn, &schema).expect("save");
    assert!(
        engine
            .get(Partition::Counter, &count_key)
            .expect("get")
            .is_some(),
        "a bound change keeps the counts"
    );
    assert!(
        engine
            .get(Partition::Schema, &coverage)
            .expect("get")
            .is_some()
    );

    // Withdrawn: the profile, the counts and the coverage go.
    let mut schema = sample_edge_type();
    schema.schema_revision = 3;
    let mut txn = Transaction::new(engine, None, Timestamp::from_raw(0), None);
    store.save_edge_type_txn(&mut txn, &schema).expect("save");
    assert_eq!(profile(), None);
    assert!(
        engine
            .get(Partition::Counter, &count_key)
            .expect("get")
            .is_none()
    );
    assert!(
        engine
            .get(Partition::Schema, &coverage)
            .expect("get")
            .is_none()
    );
}

// -- label identity --

/// An MVCC transaction on a fresh oracle over `engine`, and the commit of
/// one.
fn identity_txn<'a>(
    engine: &'a StorageEngine,
    oracle: &'a coordinode_core::txn::timestamp::TimestampOracle,
) -> Transaction<'a> {
    Transaction::begin(engine, Some(oracle), oracle.next())
}

fn commit_identity(
    txn: &mut Transaction<'_>,
) -> Result<(), coordinode_storage::engine::transaction::CommitError> {
    let wc = coordinode_core::txn::write_concern::WriteConcern::majority();
    txn.commit(&coordinode_storage::engine::transaction::CommitContext {
        write_concern: &wc,
        pipeline: None,
        id_gen: None,
    })
    .map(drop)
}

/// A label's first use allocates its identity, and every later use, read or
/// write, finds the same one; a name never used has none, and a read never
/// allocates one.
#[test]
fn a_label_is_bound_once_and_found_after() {
    let fx = open_engine();
    let engine = &fx.engine;
    let oracle = coordinode_core::txn::timestamp::TimestampOracle::new();
    let store = LocalSchemaStore::new(engine);

    let mut read = identity_txn(engine, &oracle);
    assert_eq!(store.label_id_txn(&mut read, "Goal").expect("read"), None);
    assert!(
        read.buffered(
            Partition::Schema,
            &encode_label_current_revision_key("Goal")
        )
        .is_none()
    );

    let mut txn = identity_txn(engine, &oracle);
    let goal = store.bind_label_txn(&mut txn, "Goal").expect("bind");
    assert_eq!(store.bind_label_txn(&mut txn, "Goal").expect("again"), goal);
    let thing = store.bind_label_txn(&mut txn, "Thing").expect("bind");
    assert_ne!(goal, thing, "two names, two identities");
    commit_identity(&mut txn).expect("commit");

    assert_eq!(store.label_id("Goal").expect("read"), Some(goal));
    assert_eq!(
        store.label_name(goal).expect("read").as_deref(),
        Some("Goal")
    );
    assert_eq!(
        store.load_label("Goal").expect("read"),
        None,
        "implicit: no definition"
    );
    let mut later = identity_txn(engine, &oracle);
    assert_eq!(
        store.bind_label_txn(&mut later, "Goal").expect("bind"),
        goal
    );
}

/// Two attempts using a new label at once both write its binding, so one
/// commits and the other is refused; run again, it finds the identity the
/// first allocated. Neither name ends bound to two identities.
#[test]
fn concurrent_first_uses_of_a_label_converge_on_one_identity() {
    let fx = open_engine();
    let engine = &fx.engine;
    let oracle = coordinode_core::txn::timestamp::TimestampOracle::new();
    let store = LocalSchemaStore::new(engine);

    let mut first = identity_txn(engine, &oracle);
    let mut second = identity_txn(engine, &oracle);
    let won = store.bind_label_txn(&mut first, "Goal").expect("bind");
    store.bind_label_txn(&mut second, "Goal").expect("bind");
    commit_identity(&mut first).expect("the first commits");
    assert!(
        commit_identity(&mut second).is_err(),
        "the second conflicts"
    );

    let mut retry = identity_txn(engine, &oracle);
    assert_eq!(store.bind_label_txn(&mut retry, "Goal").expect("bind"), won);
    commit_identity(&mut retry).expect("nothing new to write");
    assert_eq!(store.label_id("Goal").expect("read"), Some(won));
}

/// Declaring a label used so far without a definition keeps its identity,
/// and so does every later revision; dropping it retires the identity, and
/// a label created again under the name gets a new one, whose definitions
/// never meet the old one's. The retired identity keeps its name, so a
/// record written before the drop still reads with it.
#[test]
fn a_label_keeps_its_identity_until_dropped_and_a_recreation_gets_a_new_one() {
    let fx = open_engine();
    let engine = &fx.engine;
    let oracle = coordinode_core::txn::timestamp::TimestampOracle::new();
    let store = LocalSchemaStore::new(engine);

    let mut txn = identity_txn(engine, &oracle);
    let implicit = store.bind_label_txn(&mut txn, "User").expect("bind");
    commit_identity(&mut txn).expect("commit");

    let mut txn = identity_txn(engine, &oracle);
    store
        .save_label_txn(&mut txn, &sample_label())
        .expect("declare");
    commit_identity(&mut txn).expect("commit");
    assert_eq!(store.label_id("User").expect("read"), Some(implicit));
    let mut v2 = sample_label();
    v2.schema_revision = 2;
    let mut txn = identity_txn(engine, &oracle);
    store.save_label_txn(&mut txn, &v2).expect("revise");
    commit_identity(&mut txn).expect("commit");
    assert_eq!(store.label_id("User").expect("read"), Some(implicit));

    let mut txn = identity_txn(engine, &oracle);
    store.drop_label_txn(&mut txn, "User").expect("drop");
    commit_identity(&mut txn).expect("commit");
    assert_eq!(store.label_id("User").expect("read"), None);
    assert_eq!(store.load_label("User").expect("read"), None);

    let mut txn = identity_txn(engine, &oracle);
    store
        .save_label_txn(&mut txn, &sample_label())
        .expect("recreate");
    commit_identity(&mut txn).expect("commit");
    let recreated = store.label_id("User").expect("read").expect("bound");
    assert_ne!(recreated, implicit, "a recreation is a new identity");
    let current = store.load_label("User").expect("read").expect("declared");
    assert_eq!(current.schema_revision, 1);
    assert!(
        engine
            .get(Partition::Schema, &encode_label_schema_key(implicit, 2))
            .expect("get")
            .is_some(),
        "the old identity's definitions stay under it"
    );
    assert_eq!(
        store.label_name(implicit).expect("read").as_deref(),
        Some("User")
    );
    let names = store.label_names().expect("list");
    assert_eq!(
        names,
        vec![
            (implicit, "User".to_string()),
            (recreated, "User".to_string())
        ]
    );
}

/// An identity is never handed out twice: the next one follows the newest
/// identity record, even when the label holding it was dropped since, and a
/// catalog allocation racing another of the same identity is a conflict.
#[test]
fn an_allocation_never_reuses_an_identity() {
    use coordinode_core::schema::label_id::label_id_key;
    let fx = open_engine();
    let engine = &fx.engine;
    let oracle = coordinode_core::txn::timestamp::TimestampOracle::new();
    let store = LocalSchemaStore::new(engine);

    let first = store.bind_label("First").expect("bind");
    let second = store.bind_label("Second").expect("bind");
    let mut txn = identity_txn(engine, &oracle);
    store.drop_label_txn(&mut txn, "Second").expect("drop");
    commit_identity(&mut txn).expect("commit");
    let third = store.bind_label("Third").expect("bind");
    assert!(first < second && second < third, "{first} {second} {third}");

    // Two catalog transactions take the same next identity: the one that
    // commits second meets the other's record and is refused.
    let mut racing = identity_txn(engine, &oracle);
    let mut winner = identity_txn(engine, &oracle);
    let a = store.bind_label_txn(&mut racing, "Fourth").expect("bind");
    let b = store.bind_label_txn(&mut winner, "Fifth").expect("bind");
    assert_eq!(a, b, "both read the same newest record");
    commit_identity(&mut winner).expect("the first commit");
    assert!(
        commit_identity(&mut racing).is_err(),
        "the same identity record was written concurrently"
    );
    assert_eq!(
        engine
            .get(Partition::Schema, &label_id_key(b))
            .expect("get")
            .as_deref(),
        Some(&b"Fifth"[..])
    );
    assert_eq!(store.label_id("Fourth").expect("read"), None);
}
