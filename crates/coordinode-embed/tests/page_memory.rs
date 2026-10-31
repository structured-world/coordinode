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
            budget.reserve(bytes as u64)?.keep_until_query_ends();
            Ok::<(), coordinode_core::budget::BudgetStop>(())
        })
    });
    assert!(refused.is_err(), "the combinations fit {limit} bytes");
    assert!(
        allocated < 2 * limit as usize,
        "{allocated} bytes allocated before the refusal"
    );
    assert!(budget.memory_peak() <= budget.memory_limit());
}

/// Reading what a label implies from its stored definition allocates the
/// implied names alone, within the bound charged for it, whatever else the
/// definition declares: a thousand properties are skipped unread.
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
    let bound = LabelSchema::implication_facts_bound(bytes.len());
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
