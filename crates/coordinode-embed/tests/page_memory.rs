//! What a page of index maintenance allocates against what it charges,
//! counted by the process allocator: a row refused by a page's memory is
//! never copied, a fitting page allocates no more than it charged, and the
//! bound charged before decoding a record covers every byte the decode
//! allocates. The counts are of every allocation made, freed or not, which
//! is never less than what is held at once; the engine's own threads
//! allocate beside the reading thread, so the reads are sized well above
//! what they may allocate.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::alloc::System;
use std::collections::{BTreeMap, HashMap};

use coordinode_core::budget::QueryBudget;
use coordinode_core::graph::node::NodeRecord;
use coordinode_core::graph::types::Value;
use coordinode_core::schema::label_id::LabelId;
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::transaction::Transaction;
use coordinode_storage::error::StorageError;
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

/// What `f` returns, and the bytes allocated while it ran.
fn allocated_by<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let region = Region::new(GLOBAL);
    let value = f();
    let stats = region.change();
    // Growth in place counts as allocation; shrinking frees nothing new.
    let grown = usize::try_from(stats.bytes_reallocated.max(0)).unwrap_or(0);
    (value, stats.bytes_allocated + grown)
}

fn engine() -> (
    StorageEngine,
    std::sync::Arc<TimestampOracle>,
    tempfile::TempDir,
) {
    let dir = tempfile::tempdir().unwrap();
    let oracle = std::sync::Arc::new(TimestampOracle::new());
    let engine = StorageEngine::open_with_oracle(
        &StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            dir.path().to_string_lossy().as_ref(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]),
        std::sync::Arc::clone(&oracle),
    )
    .unwrap();
    (engine, oracle, dir)
}

/// Allocation besides the rows a page keeps: the iterator over the
/// memtable, the bounds of the range, and what the engine's own threads
/// allocate meanwhile. Far below the rows these reads are sized at.
const BESIDE_THE_ROWS: usize = 128 * 1024;

/// A page whose budget a row does not fit is refused before that row is
/// copied: a 4 MiB row read from the memtable against 16 KiB of budget
/// allocates nothing near the row, and the budget never holds more than
/// its limit. Copying the row first and checking the page afterwards would
/// allocate the whole row before the refusal.
#[test]
fn a_page_refuses_a_row_before_copying_it() {
    let (engine, oracle, _dir) = engine();
    let row = 4 << 20;
    engine.put(Partition::Node, b"p:a", b"small").unwrap();
    engine
        .put(Partition::Node, b"p:b", &vec![7u8; row])
        .unwrap();
    let mut txn = Transaction::begin(
        &engine,
        Some(&oracle),
        Timestamp::from_raw(engine.snapshot()),
    );
    let budget = QueryBudget::new(16 * 1024);

    let (refused, allocated) =
        allocated_by(|| txn.prefix_scan_paged_charged(Partition::Node, b"p:", None, 8, &budget));

    assert!(
        matches!(refused, Err(StorageError::Budget(_))),
        "{:?}",
        refused.map(|p| p.rows.len())
    );
    assert!(
        allocated < BESIDE_THE_ROWS,
        "the refused row was copied: {allocated} bytes allocated"
    );
    assert!(budget.memory_peak() <= budget.memory_limit());
}

/// Every byte a fitting page keeps is charged before it is taken: the read
/// allocates no more than it charged and the reading itself, the rows,
/// their list and the cursor included.
#[test]
fn a_fitting_page_allocates_no_more_than_it_charged() {
    let (engine, oracle, _dir) = engine();
    for n in 0..40u32 {
        let key = format!("p:{n:04}");
        engine
            .put(
                Partition::Node,
                key.as_bytes(),
                &vec![1u8; 40_000 + 371 * n as usize],
            )
            .unwrap();
    }
    let mut txn = Transaction::begin(
        &engine,
        Some(&oracle),
        Timestamp::from_raw(engine.snapshot()),
    );
    let budget = QueryBudget::new(64 << 20);

    let (page, allocated) = allocated_by(|| {
        txn.prefix_scan_paged_charged(Partition::Node, b"p:", None, 25, &budget)
            .unwrap()
    });

    assert_eq!(page.rows.len(), 25);
    let kept: usize = page
        .rows
        .iter()
        .map(|(k, v)| k.capacity() + v.capacity())
        .sum::<usize>()
        + page.rows.capacity() * core::mem::size_of::<(Vec<u8>, Vec<u8>)>()
        + page.last_key.as_ref().map_or(0, Vec::capacity);
    let charged = budget.memory_used() as usize;
    assert!(charged >= kept, "charged {charged} for {kept} bytes kept");
    assert!(
        allocated <= charged + BESIDE_THE_ROWS,
        "{allocated} allocated, {charged} charged"
    );
}

fn encode(record: &NodeRecord) -> Vec<u8> {
    record.to_msgpack().unwrap()
}

/// The bound charged before decoding the properties an index reads covers
/// every byte the decode allocates, for strings, nested arrays, maps (whose
/// B-tree leaf holds room for eleven entries), vectors, blobs and
/// undeclared properties, and leaves out the properties it does not read.
/// No engine runs here, so every allocation counted is the decode's.
#[test]
fn the_projection_bound_covers_its_decode() {
    let mut map = BTreeMap::new();
    map.insert("k".to_string(), Value::Int(1));
    let mut wide = BTreeMap::new();
    for n in 0..40 {
        wide.insert(format!("key{n}"), Value::String("v".repeat(n)));
    }
    let shapes = [
        Value::Int(7),
        Value::String("t".repeat(1000)),
        Value::Array((0..200i64).map(Value::Int).collect()),
        Value::Array(vec![Value::Array(vec![Value::Map(map.clone()); 3]); 5]),
        Value::Map(map),
        Value::Map(wide),
        Value::Vector(vec![0.5; 384]),
        Value::Blob(vec![9; 4096]),
    ];
    for (n, shape) in shapes.into_iter().enumerate() {
        let mut props = HashMap::new();
        props.insert(1u32, shape.clone());
        props.insert(2u32, Value::String("x".repeat(100_000)));
        let mut extra = HashMap::new();
        extra.insert("loose".to_string(), shape);
        extra.insert("other".to_string(), Value::String("y".repeat(50_000)));
        let mut record = NodeRecord {
            labels: vec![LabelId::new(3).unwrap()],
            props,
            extra: Some(extra),
        };
        let bytes = encode(&record);
        // The same record without the properties no projection reads.
        record.props.remove(&2);
        if let Some(extra) = record.extra.as_mut() {
            extra.remove("other");
        }
        let read_only = encode(&record);
        for (fields, names) in [
            (&[1u32][..], &[][..]),
            (&[][..], &["loose"][..]),
            (&[1u32][..], &["loose"][..]),
            (&[9u32][..], &["absent"][..]),
        ] {
            let bound = NodeRecord::projection_bound(&bytes, fields, names).unwrap();
            let (decoded, allocated) =
                allocated_by(|| NodeRecord::project_from_msgpack(&bytes, fields, names).unwrap());
            assert!(
                allocated <= bound,
                "shape {n}, fields {fields:?}, names {names:?}: {allocated} allocated, {bound} bound"
            );
            assert_eq!(
                bound,
                NodeRecord::projection_bound(&read_only, fields, names).unwrap(),
                "shape {n}: the bound counts a property it does not read"
            );
            drop(decoded);
        }
    }
}

/// The combinations of a compound key over two lists are refused before
/// any is built: two lists of three hundred strings make ninety thousand
/// tuples of megabytes, and a budget refusing the combinations leaves only
/// the encoding of the six hundred elements allocated. Building them and
/// charging afterwards would allocate them all first.
#[test]
fn compound_combinations_are_refused_before_they_are_built() {
    use coordinode_core::index::derive::tuples_within;
    let list = |p: &str| {
        Value::Array(
            (0..300)
                .map(|n| Value::String(format!("{p}{n:05}{}", "x".repeat(20))))
                .collect(),
        )
    };
    let values = [list("a"), list("b")];
    let limit = 64 * 1024;
    let budget = QueryBudget::new(limit);
    let (refused, allocated) = allocated_by(|| {
        tuples_within(&values, |bytes| {
            budget
                .reserve(bytes as u64)
                .map_err(coordinode_modality::StoreError::from)?
                .keep_until_query_ends();
            Ok::<(), coordinode_modality::StoreError>(())
        })
    });
    assert!(refused.is_err(), "the combinations fit {limit} bytes");
    assert!(
        allocated < 2 * limit as usize,
        "{allocated} bytes allocated before the refusal"
    );
    assert!(budget.memory_peak() <= budget.memory_limit());
}

/// Under a compound uniqueness, a change placing a node whose two lists make
/// three hundred sixty thousand combinations is refused by a small page
/// budget before the combinations are built: the uniqueness checks that run
/// ahead of the staging count them against the same budget. Building them
/// first would allocate tens of megabytes before the refusal.
///
/// A change allocates far more than the combinations in its own machinery
/// (catalog, certification, engine threads, each halved page tried again),
/// all of it the same whatever the lists hold. So the measure is a
/// difference: lists of six hundred elements against lists of three
/// hundred (a quarter as many combinations), both refused at the count.
/// Built, the extra combinations alone would take over twenty megabytes.
#[test]
fn a_unique_change_refuses_compound_combinations_before_building_them() {
    let refused_as_budget = |outcome: &Result<_, coordinode_embed::DatabaseError>| {
        assert!(
            matches!(outcome, Err(coordinode_embed::DatabaseError::Budget(_))),
            "not the page's memory refusal: {:?}",
            outcome.as_ref().err()
        );
    };
    let (quarter, quarter_allocated) = compound_change(300, 256 * 1024);
    refused_as_budget(&quarter);
    let (refused, allocated) = compound_change(600, 256 * 1024);
    refused_as_budget(&refused);
    let grown = allocated.saturating_sub(quarter_allocated);
    assert!(
        grown < ENGINE_NOISE,
        "{grown} more bytes for four times the combinations \
         ({allocated} against {quarter_allocated})"
    );
}

/// A unit whose DERIVED work names lists making four million combinations,
/// past what the unit may derive, is refused before any is built, on apply
/// and on replay alike: the count is taken from the lists' lengths. Building
/// them first would allocate hundreds of megabytes before the refusal.
#[test]
fn a_derived_unit_past_its_budget_is_refused_before_expanding() {
    use coordinode_core::index::derive::{
        DeriveError, IndexInterpretation, KEY_CODEC, PropertyRef, resolve_unit,
    };
    use coordinode_core::index::identity::GenerationId;
    use coordinode_core::txn::proposal::{DerivedIndexWork, DerivedSource, IndexBinding, Mutation};
    let list = |p: &str| {
        Value::Array(
            (0..2048)
                .map(|n| Value::String(format!("{p}{n:05}")))
                .collect(),
        )
    };
    let unit = vec![Mutation::Derive(DerivedIndexWork {
        binding: IndexBinding {
            epoch: 1,
            interpretation: IndexInterpretation {
                codec: KEY_CODEC,
                generation: GenerationId::from_raw(3),
                unique: false,
                sparse: false,
                properties: ["a", "b"]
                    .into_iter()
                    .map(|name| PropertyRef {
                        field: None,
                        name: name.into(),
                    })
                    .collect(),
                filter: None,
            },
        },
        node_id: 1,
        valid_from: None,
        old: None,
        new: DerivedSource::Values(Some(vec![list("a"), list("b")])),
    })];
    let (refused, allocated) = allocated_by(|| resolve_unit(&unit, 1 << 10));
    assert_eq!(refused.err(), Some(DeriveError::FanOut(1 << 10)));
    assert!(
        allocated < 64 * 1024,
        "{allocated} bytes allocated before the refusal"
    );
}

/// What the engine's own threads allocate during one statement varies by
/// some megabytes from run to run under load, and the allocator counts every
/// thread. The two sizes compared are chosen so that building the extra
/// combinations, tuples alone, would take more than this beside the noise.
const ENGINE_NOISE: usize = 16 << 20;

/// A write under a compound uniqueness whose two lists make three hundred
/// sixty thousand combinations is refused by the statement's memory before
/// they are built, its uniqueness checks counting them against the same
/// budget as its writes. Measured, as the change above, against lists a
/// quarter as many combinations: the extra ones built would be over twenty
/// megabytes of tuples.
#[test]
fn a_unique_write_refuses_compound_combinations_before_building_them() {
    let refused_as_budget = |outcome: &Result<_, coordinode_embed::DatabaseError>| {
        assert!(
            matches!(outcome, Err(coordinode_embed::DatabaseError::Budget(_))),
            "not the statement's memory refusal: {:?}",
            outcome.as_ref().err()
        );
    };
    let (quarter, quarter_allocated) = compound_write(300);
    refused_as_budget(&quarter);
    let (refused, allocated) = compound_write(600);
    refused_as_budget(&refused);
    let grown = allocated.saturating_sub(quarter_allocated);
    assert!(
        grown < ENGINE_NOISE,
        "{grown} more bytes for four times the combinations \
         ({allocated} against {quarter_allocated})"
    );
}

/// A write of a Thing whose lists hold `n` elements each under a compound
/// uniqueness, with a statement memory of 256 KiB, and the bytes it
/// allocated.
fn compound_write(
    n: usize,
) -> (
    Result<Vec<std::collections::BTreeMap<String, Value>>, coordinode_embed::DatabaseError>,
    usize,
) {
    use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType, SchemaMode};
    let mut db = coordinode_embed::Database::open_in_memory().expect("open");
    let mut schema = LabelSchema::new_node_id("Thing");
    schema.set_mode(SchemaMode::Flexible);
    for property in ["a", "b"] {
        schema.add_property(PropertyDef::new(
            property,
            PropertyType::Array(Box::new(PropertyType::String)),
        ));
    }
    db.create_label_schema(schema).expect("thing");
    db.execute_cypher("CREATE CONSTRAINT thing_ab FOR (n:Thing) REQUIRE (n.a, n.b) IS UNIQUE")
        .expect("uniqueness");
    db.set_query_memory_limit(256 * 1024).expect("limit");
    let list = |p: &str| {
        (0..n)
            .map(|i| format!("'{p}{i:05}{}'", "x".repeat(20)))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let write = format!("CREATE (:Thing {{a: [{}], b: [{}]}})", list("a"), list("b"));
    allocated_by(|| db.execute_cypher(&write))
}

/// The change of what Task implies over one Task whose lists hold `n`
/// elements each, under a compound uniqueness on what it comes to imply,
/// with pages of `page_bytes`, and the bytes it allocated.
fn compound_change(
    n: usize,
    page_bytes: u64,
) -> (
    Result<Vec<std::collections::BTreeMap<String, Value>>, coordinode_embed::DatabaseError>,
    usize,
) {
    use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType, SchemaMode};
    let mut db = coordinode_embed::Database::open_in_memory().expect("open");
    for name in ["Thing", "Task"] {
        let mut schema = LabelSchema::new_node_id(name);
        schema.set_mode(SchemaMode::Flexible);
        for property in ["a", "b"] {
            schema.add_property(PropertyDef::new(
                property,
                PropertyType::Array(Box::new(PropertyType::String)),
            ));
        }
        db.create_label_schema(schema).expect(name);
    }
    db.execute_cypher("CREATE CONSTRAINT thing_ab FOR (n:Thing) REQUIRE (n.a, n.b) IS UNIQUE")
        .expect("uniqueness");
    let list = |p: &str| {
        (0..n)
            .map(|n| format!("'{p}{n:05}{}'", "x".repeat(20)))
            .collect::<Vec<_>>()
            .join(", ")
    };
    db.execute_cypher(&format!(
        "CREATE (:Task {{a: [{}], b: [{}]}})",
        list("a"),
        list("b")
    ))
    .expect("task");
    db.set_index_build_config(coordinode_query::index::IndexBuildConfig {
        maintenance_page_bytes: page_bytes,
        ..db.index_build_config()
    });

    allocated_by(|| db.execute_cypher("ALTER NODE TYPE Task SET IMPLIES (Thing)"))
}

/// The bounds charged before an index definition is read cover what reading
/// it allocates, measured from the fields read and not from the record's
/// length: a definition with six hundred parts and a long text
/// configuration's worth of bytes skipped. Reading only its parts, and
/// reading it for placement, each allocate within their bound, and a large
/// skipped configuration adds nothing to either.
#[test]
fn the_definition_bounds_cover_their_decodes() {
    use coordinode_core::schema::label_id::LabelId;
    use coordinode_modality::index_def::{GenerationId, IndexDefinition, IndexDescriptor, IndexId};
    let mut def = IndexDescriptor::compound(
        "wide",
        "Thing",
        (0..40).map(|n| format!("property_{n:03}")).collect(),
    )
    .bind(IndexId::from_raw(7), GenerationId::from_raw(9));
    for n in 1..=300u32 {
        def.propose_parts(
            LabelId::new(n).unwrap(),
            GenerationId::from_raw(11),
            GenerationId::from_raw(u64::from(n) * 2 + 100),
            GenerationId::from_raw(u64::from(n) * 2 + 101),
        );
    }
    let bytes = rmp_serde::to_vec(&def).unwrap();

    let parts_bound = IndexDefinition::selection_bound(&bytes).unwrap();
    let (parts, allocated) =
        allocated_by(|| IndexDefinition::selection_from_msgpack(&bytes).unwrap());
    assert_eq!(parts.len(), 600);
    assert!(
        allocated <= parts_bound,
        "{allocated} allocated, {parts_bound} bound"
    );

    let (kept_bound, measuring) =
        allocated_by(|| IndexDefinition::for_placement_bound(&bytes).unwrap());
    assert_eq!(measuring, 0, "measuring allocated");
    let (kept, allocated) =
        allocated_by(|| IndexDefinition::for_placement_from_msgpack(&bytes).unwrap());
    assert_eq!(kept, def);
    assert!(
        allocated <= kept_bound,
        "{allocated} allocated, {kept_bound} bound"
    );
    assert!(
        kept_bound < bytes.len() * 4,
        "the bound is not measured from the fields: {kept_bound} for {} bytes",
        bytes.len()
    );
}

/// Reading what a label implies from its stored definition allocates the
/// implied names alone, within the bound charged for it, whatever else the
/// definition declares: a thousand properties are skipped unread, both by
/// the decode and by the bound, which measuring allocates nothing.
#[test]
fn the_implication_facts_skip_the_rest_of_the_definition() {
    use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType};
    let mut schema = LabelSchema::new_node_id("Wide");
    for n in 0..1000 {
        schema.add_property(PropertyDef::new(
            format!("property_{n:04}_{}", "x".repeat(40)),
            PropertyType::String,
        ));
    }
    schema.set_implies(vec!["Thing".into(), "Work".into()]);
    let bytes = schema.to_msgpack().expect("encode");
    let (bound, measuring) =
        allocated_by(|| LabelSchema::implication_facts_bound(&bytes).expect("bound"));
    assert_eq!(measuring, 0, "measuring allocated");
    assert!(bound < 1024, "the bound counts the properties: {bound}");
    let (facts, allocated) =
        allocated_by(|| LabelSchema::implication_facts_from_msgpack(&bytes, false).expect("facts"));
    assert_eq!(facts.implies, ["Thing", "Work"]);
    assert!(allocated <= bound, "{allocated} allocated, {bound} bound");
    assert!(
        allocated < 1024,
        "{allocated} bytes allocated for two names: the properties were decoded"
    );
}

/// The labels a record's header counts are what decoding them allocates.
#[test]
fn the_labels_length_covers_their_decode() {
    for count in [0u32, 1, 2, 17, 300] {
        let record = NodeRecord {
            labels: (1..=count).filter_map(LabelId::new).collect(),
            props: HashMap::new(),
            extra: None,
        };
        let bytes = encode(&record);
        let len = NodeRecord::labels_len_from_msgpack(&bytes).unwrap();
        assert_eq!(len, count as usize);
        let (labels, allocated) = allocated_by(|| NodeRecord::labels_from_msgpack(&bytes).unwrap());
        assert_eq!(labels.len(), count as usize);
        assert!(
            allocated <= len * core::mem::size_of::<LabelId>(),
            "{count} labels: {allocated} allocated"
        );
    }
}
