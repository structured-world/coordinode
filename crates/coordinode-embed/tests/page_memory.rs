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

/// A row's column is found by one keyed lookup: a short name allocates
/// nothing, a long one no more than [`column_lookup_bytes`] says, which a
/// charged reader reserves first, however wide the row and whether or not
/// the column is there.
#[test]
fn a_column_lookup_allocates_what_it_is_charged() {
    use coordinode_query::executor::row::{Row, column_lookup_bytes, property};
    let long = "a".repeat(64 << 10);
    let mut row = Row::new();
    row.insert("n".to_string(), Value::Int(1));
    for i in 0..20_000 {
        row.insert(format!("n.c{i}"), Value::Int(i));
    }
    row.insert(format!("n.{long}"), Value::Int(3));
    row.insert("n.b".to_string(), Value::Int(4));
    for (name, expected) in [
        (long.as_str(), Some(&Value::Int(3))),
        (&long[1..], None),
        ("b", Some(&Value::Int(4))),
        ("x", None),
    ] {
        let (found, allocated) = least_allocated_by(|| property(&row, "n", name));
        assert_eq!(found, expected, "{}", name.len());
        assert!(
            allocated as u64 <= column_lookup_bytes("n", name),
            "a name of {} bytes allocated {allocated}",
            name.len()
        );
    }
    assert_eq!(column_lookup_bytes("n", "b"), 0);
}

/// What `f` returns, and the fewest bytes allocated over a few calls of it:
/// the count is the whole process's, so a thread of the harness or a one-off
/// initialization can land in one call; what `f` itself allocates lands in
/// every one.
fn least_allocated_by<T>(mut f: impl FnMut() -> T) -> (T, usize) {
    let (mut value, mut least) = allocated_by(&mut f);
    for _ in 0..4 {
        let (again, allocated) = allocated_by(&mut f);
        value = again;
        least = least.min(allocated);
    }
    (value, least)
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

/// A nearest-neighbour read through a vector index whose candidate records
/// do not fit the statement's memory is refused before it reads them all:
/// two thousand 16 KiB records against 2 MiB of budget allocate a batch of
/// them, well under the whole 32 MiB, before the refusal. Reading every
/// candidate first and charging them afterwards would allocate all of them.
#[test]
fn a_vector_index_read_refuses_candidates_before_reading_them_all() {
    let mut db = coordinode_embed::Database::open_in_memory().unwrap();
    for ddl in [
        "CREATE NODE TYPE Doc WITH (title: STRING)",
        "ALTER LABEL Doc SET SCHEMA FLEXIBLE",
        r#"CREATE VECTOR INDEX doc_emb ON :Doc(emb) OPTIONS {metric: "l2"}"#,
    ] {
        db.execute_cypher(ddl).unwrap();
    }
    let body = "b".repeat(16 << 10);
    for i in 0..2000 {
        db.execute_cypher(&format!(
            "CREATE (:Doc {{title: '{body}', emb: [{}, 0.0]}})",
            f64::from(i)
        ))
        .unwrap();
    }
    let query = "MATCH (n:Doc) WITH n, vector_distance(n.emb, [0.0, 0.0]) AS d \
                 ORDER BY d LIMIT 2000 RETURN n.title AS title";
    // The index has taken every node once a read through it finds them all.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    while db.execute_cypher(query).unwrap().len() < 2000 {
        assert!(
            std::time::Instant::now() < deadline,
            "the index never caught up"
        );
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    assert!(db.explain_cypher(query).unwrap().contains("HnswScan("));
    db.set_query_memory_limit(2 << 20).unwrap();

    let (refused, allocated) = allocated_by(|| db.execute_cypher(query));

    assert!(
        matches!(refused, Err(coordinode_embed::DatabaseError::Budget(_))),
        "{:?}",
        refused.map(|rows| rows.len())
    );
    assert!(
        allocated < 20 << 20,
        "the candidates were read past the budget: {allocated} bytes allocated"
    );
}

/// A temporal node an index finds resolves to its state from its versions,
/// each charged before it is decoded. A 4 MiB version against 6 MiB of
/// statement memory: the row the scan hands over fits, the row with the
/// version's decode does not, so the read is refused with nothing past the
/// tree's own read allocated. Decoding the version and charging it
/// afterwards would allocate it whole first.
#[test]
fn a_temporal_index_read_refuses_a_version_before_decoding_it() {
    let mut db = coordinode_embed::Database::open_in_memory().unwrap();
    for ddl in [
        "CREATE NODE TYPE Emp TEMPORAL WITH \
         (name: STRING, tag: STRING, valid_from: INT, valid_to: INT)",
        "CREATE INDEX emp_tag ON :Emp(tag)",
    ] {
        db.execute_cypher(ddl).unwrap();
    }
    let body = "b".repeat(4 << 20);
    db.execute_cypher(&format!(
        "CREATE (:Emp {{name: '{body}', tag: 'find', valid_from: 0}})"
    ))
    .unwrap();
    let query = "MATCH (n:Emp {tag: 'find'}) RETURN n.tag AS tag";
    assert!(db.explain_cypher(query).unwrap().contains("IndexScan"));
    db.set_query_memory_limit(6 << 20).unwrap();

    // What the tree's own read of the node rows allocates, decoding nothing:
    // the lower read, which hands each value over as it holds it.
    let snapshot = db.engine().snapshot();
    let ((), lower) = least_allocated_by(|| {
        for guard in db
            .engine()
            .snapshot_prefix_iter(
                &snapshot,
                Partition::Node,
                coordinode_core::graph::node::NODE_KEY_PREFIX,
            )
            .unwrap()
        {
            drop(guard.into_inner().unwrap());
        }
    });
    // The statement's first run compiles and keeps its plan; the read is
    // measured on the next.
    let _ = db.execute_cypher(query);

    let (refused, allocated) = allocated_by(|| db.execute_cypher(query));

    assert!(
        matches!(refused, Err(coordinode_embed::DatabaseError::Budget(_))),
        "{:?}",
        refused.map(|rows| rows.len())
    );
    assert!(
        allocated < lower + BESIDE_THE_ROWS,
        "the version was decoded past the budget: {allocated} bytes allocated, \
         {lower} by the tree's read"
    );
}

/// A database holding one temporal `Emp` node, tagged `src`, whose name is
/// 4 MiB long; its id. Statements run against 10 MiB.
fn one_large_temporal_node() -> (coordinode_embed::Database, i64) {
    let mut db = coordinode_embed::Database::open_in_memory().unwrap();
    db.execute_cypher(
        "CREATE NODE TYPE Emp TEMPORAL WITH \
         (name: STRING, tag: STRING, valid_from: INT, valid_to: INT)",
    )
    .unwrap();
    let body = "b".repeat(4 << 20);
    let rows = db
        .execute_cypher(&format!(
            "CREATE (e:Emp {{name: '{body}', tag: 'src', valid_from: 0}}) RETURN id(e) AS id"
        ))
        .unwrap();
    let id = rows[0]
        .get("id")
        .and_then(|value| match value {
            Value::Int(id) => Some(*id),
            _ => None,
        })
        .expect("the node's id");
    // Reading the state peaks near 8.4 MiB (the scanned row beside the
    // decode's bound): against 10 MiB a statement reading only the tag fits.
    db.set_query_memory_limit(10 << 20).unwrap();
    (db, id)
}

/// A node read by its id holds the row it hands on for the statement: the
/// row a temporal node's state builds carries its 4 MiB name, and a
/// projection keeping two more copies of the name beside that row does not
/// fit 10 MiB, so the read is refused, and the next statement runs with the
/// whole budget again. A row left unpaid would let the projection fit.
#[test]
fn a_node_read_by_id_holds_its_row_for_the_statement() {
    let (mut db, id) = one_large_temporal_node();
    let read = db.execute_cypher(&format!(
        "MATCH (e:Emp) WHERE id(e) = {id} RETURN e.name AS n, e.name AS m"
    ));
    assert!(
        matches!(read, Err(coordinode_embed::DatabaseError::Budget(_))),
        "{:?}",
        read.map(|rows| rows.len())
    );
    let tag = db
        .execute_cypher(&format!(
            "MATCH (e:Emp) WHERE id(e) = {id} RETURN e.tag AS t"
        ))
        .expect("the budget is whole again");
    assert_eq!(tag[0].get("t"), Some(&Value::String("src".into())));
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

/// The bound charged before decoding a whole record covers every byte the
/// decode allocates, its labels and every property included, for the same
/// shapes as a projection, with several labels and an overflow map. No
/// engine runs here, so every allocation counted is the decode's.
#[test]
fn the_record_bound_covers_its_decode() {
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
        for extra_too in [false, true] {
            let mut props = HashMap::new();
            for field in 0..17u32 {
                props.insert(field, shape.clone());
            }
            let extra = extra_too.then(|| {
                (0..9)
                    .map(|k| (format!("loose{k}"), shape.clone()))
                    .collect::<HashMap<_, _>>()
            });
            let record = NodeRecord {
                labels: (1..=5).map(|l| LabelId::new(l).unwrap()).collect(),
                props,
                extra,
            };
            assert_record_bound(&record, &format!("shape {n}, extra {extra_too}"));
        }
    }
    // Small and edge records: no labels or one, no properties or one, empty
    // and one-element containers, a one-entry overflow map.
    let mut one = BTreeMap::new();
    one.insert("k".to_string(), Value::Null);
    let small = [
        Value::Null,
        Value::Bool(true),
        Value::String(String::new()),
        Value::Array(Vec::new()),
        Value::Array(vec![Value::Int(1)]),
        Value::Map(BTreeMap::new()),
        Value::Map(one),
        Value::Vector(Vec::new()),
        Value::Blob(Vec::new()),
    ];
    for labels in [0, 1] {
        let bare = NodeRecord {
            labels: (1..=labels).map(|l| LabelId::new(l).unwrap()).collect(),
            props: HashMap::new(),
            extra: None,
        };
        assert_record_bound(&bare, &format!("{labels} labels, no properties"));
        for (n, value) in small.iter().enumerate() {
            let mut record = bare.clone();
            record.set(1, value.clone());
            assert_record_bound(&record, &format!("{labels} labels, small {n}"));
            let mut loose = bare.clone();
            loose.set_extra("x", value.clone());
            assert_record_bound(&loose, &format!("{labels} labels, loose small {n}"));
        }
    }
}

/// The decode bound of `record`'s encoding covers what decoding it
/// allocates and what the decoded record holds past its own struct.
fn assert_record_bound(record: &NodeRecord, what: &str) {
    let bytes = encode(record);
    let bound = NodeRecord::decode_bound(&bytes).unwrap();
    let (decoded, allocated) = allocated_by(|| NodeRecord::from_msgpack(&bytes).unwrap());
    assert!(
        allocated <= bound,
        "{what}: {allocated} allocated, {bound} bound"
    );
    let held = decoded.held_bytes() as usize - core::mem::size_of::<NodeRecord>();
    assert!(held <= bound, "{what}: holds {held}, {bound} bound");
    assert_eq!(&decoded, record, "{what}");
}

/// A record of label 1 whose one string property is `bytes` long, with a
/// nested list of maps beside it, so a decode expands it.
fn sized_record(bytes: usize) -> NodeRecord {
    let mut map = BTreeMap::new();
    map.insert("k".to_string(), Value::Int(1));
    let mut record = NodeRecord::new(LabelId::new(1).unwrap());
    record.set(1, Value::String("x".repeat(bytes)));
    record.set(2, Value::Array(vec![Value::Map(map); 8]));
    record
}

/// Store node `id` of shard 0 as `record`, committed in its own transaction.
fn store_node(engine: &StorageEngine, oracle: &TimestampOracle, id: u64, record: &NodeRecord) {
    use coordinode_core::graph::node::NodeId;
    use coordinode_modality::{LocalNodeStore, NodeStore as _};
    let mut txn = Transaction::begin(engine, Some(oracle), oracle.next());
    LocalNodeStore
        .put(&mut txn, 0, NodeId::from_raw(id), record)
        .unwrap();
    let write_concern = coordinode_core::txn::write_concern::WriteConcern::majority();
    txn.commit(&coordinode_storage::engine::transaction::CommitContext {
        write_concern: &write_concern,
        pipeline: None,
        id_gen: None,
    })
    .unwrap();
}

/// What a batch read under a budget holds: the room its list allocated,
/// `capacity` slots, and each record's heap.
fn held_by(records: &[Option<NodeRecord>], capacity: usize) -> u64 {
    (capacity * core::mem::size_of::<Option<NodeRecord>>()) as u64
        + records
            .iter()
            .flatten()
            .map(|r| r.held_bytes() - core::mem::size_of::<NodeRecord>() as u64)
            .sum::<u64>()
}

/// A batch read under a budget answers as the unmetered one does, through a
/// snapshot and without one: a record per id in the ids' order, `None` for
/// an id with no record, a repeated id's record twice. Its charge pays for
/// what it returns, list and records, and goes back when dropped.
#[test]
fn a_charged_batch_read_returns_what_the_unmetered_one_does() {
    use coordinode_core::graph::node::NodeId;
    use coordinode_modality::{LocalNodeStore, NodeStore as _};
    let (engine, oracle, _dir) = engine();
    for id in 1..=6 {
        store_node(&engine, &oracle, id, &sized_record(1000 * id as usize));
    }
    // A missing id, a repeated one, out of order.
    let ids: Vec<NodeId> = [3, 9, 1, 3, 6, 2]
        .into_iter()
        .map(NodeId::from_raw)
        .collect();
    let snapshot = Transaction::begin(
        &engine,
        Some(&oracle),
        Timestamp::from_raw(engine.snapshot()),
    );
    let latest = Transaction::new(&engine, None, Timestamp::from_raw(engine.snapshot()), None);
    for txn in [&snapshot, &latest] {
        let unmetered = LocalNodeStore.get_many(txn, 0, &ids).unwrap();
        let budget = QueryBudget::new(64 << 20);
        let (records, charge) = LocalNodeStore
            .get_many_charged(txn, 0, &ids, &budget)
            .unwrap();
        assert_eq!(records, unmetered);
        assert_eq!(records[1], None, "no record for the missing id");
        assert_eq!(records[0], records[3], "the repeated id read twice");
        assert_eq!(
            charge.bytes(),
            held_by(&records, records.capacity()),
            "it pays for what it returns"
        );
        assert_eq!(budget.memory_used(), charge.bytes());
        drop(charge);
        assert_eq!(
            budget.memory_used(),
            0,
            "the charge goes back with its records"
        );
    }
}

/// The list a batch read returns is paid for as allocated, an empty input,
/// small ones and ones whose ids have no record included: a list of none,
/// one, two or three slots holds room for exactly those, so its charge is
/// what it holds.
#[test]
fn a_small_or_empty_batch_pays_for_the_room_its_list_holds() {
    use coordinode_core::graph::node::NodeId;
    use coordinode_modality::{LocalNodeStore, NodeStore as _};
    let (engine, oracle, _dir) = engine();
    store_node(&engine, &oracle, 1, &sized_record(100));
    let txn = Transaction::begin(
        &engine,
        Some(&oracle),
        Timestamp::from_raw(engine.snapshot()),
    );
    for ids in [Vec::new(), vec![1u64], vec![1, 7], vec![7, 8, 9], vec![7]] {
        let ids: Vec<NodeId> = ids.into_iter().map(NodeId::from_raw).collect();
        let budget = QueryBudget::new(1 << 20);
        let (records, charge) = LocalNodeStore
            .get_many_charged(&txn, 0, &ids, &budget)
            .unwrap();
        assert_eq!(
            records.capacity(),
            ids.len(),
            "{ids:?}: room for exactly them"
        );
        assert_eq!(
            charge.bytes(),
            held_by(&records, records.capacity()),
            "{ids:?}"
        );
    }
}

/// What a batch read keeps aside for its keys is charged before it exists:
/// twenty thousand ids with no record allocate no more than the budget's
/// peak beside what the tree's own batched read allocates for them. The
/// keys, their views and the read's lists of misses taken unpaid would pass
/// the peak by more than a megabyte.
#[test]
fn a_batch_read_charges_what_it_keeps_aside_for_its_keys() {
    use coordinode_core::graph::node::NodeId;
    use coordinode_modality::{LocalNodeStore, NodeStore as _};
    let (engine, oracle, _dir) = engine();
    store_node(&engine, &oracle, 1, &sized_record(100));
    let ids: Vec<NodeId> = (1_000..21_000).map(NodeId::from_raw).collect();
    let snapshot = Transaction::begin(
        &engine,
        Some(&oracle),
        Timestamp::from_raw(engine.snapshot()),
    );
    let latest = Transaction::new(&engine, None, Timestamp::from_raw(engine.snapshot()), None);
    // What the tree's own batched read allocates for these keys, measured
    // through the engine's snapshot read, which keeps nothing aside of its
    // own outside the index partition: the tree's per-key state and its
    // list of results, the lower read's to account for.
    let keys: Vec<Vec<u8>> = ids
        .iter()
        .map(|id| LocalNodeStore::record_key(0, *id))
        .collect();
    let refs: Vec<&[u8]> = keys.iter().map(Vec::as_slice).collect();
    let snap = engine.snapshot();
    let ((), tree_read) = least_allocated_by(|| {
        engine
            .snapshot_multi_get_each(&snap, Partition::Node, &refs, None, |_, _| {
                Ok::<(), StorageError>(())
            })
            .unwrap();
    });
    for txn in [&snapshot, &latest] {
        let budget = QueryBudget::new(64 << 20);
        let (read, allocated) = least_allocated_by(|| {
            LocalNodeStore
                .get_many_charged(txn, 0, &ids, &budget)
                .map(|(records, _)| records.len())
        });
        assert_eq!(read.unwrap(), ids.len());
        let peak = budget.memory_peak() as usize;
        assert!(
            allocated <= peak + tree_read + BESIDE_THE_ROWS,
            "{allocated} allocated, {peak} charged at most, {tree_read} by the tree's read"
        );
    }
}

/// A batch read sees what its own transaction wrote and deleted, as the
/// unmetered read does.
#[test]
fn a_charged_batch_read_sees_its_own_writes() {
    use coordinode_core::graph::node::NodeId;
    use coordinode_modality::{LocalNodeStore, NodeStore as _};
    let (engine, oracle, _dir) = engine();
    for id in 1..=3 {
        store_node(&engine, &oracle, id, &sized_record(500));
    }
    let mut txn = Transaction::begin(
        &engine,
        Some(&oracle),
        Timestamp::from_raw(engine.snapshot()),
    );
    let written = sized_record(2000);
    LocalNodeStore
        .put(&mut txn, 0, NodeId::from_raw(2), &written)
        .unwrap();
    LocalNodeStore
        .delete(&mut txn, 0, NodeId::from_raw(3))
        .unwrap();
    let ids: Vec<NodeId> = (1..=3).map(NodeId::from_raw).collect();
    let budget = QueryBudget::new(64 << 20);
    let (records, _charge) = LocalNodeStore
        .get_many_charged(&txn, 0, &ids, &budget)
        .unwrap();
    assert_eq!(records[0], Some(sized_record(500)));
    assert_eq!(records[1], Some(written), "its own write");
    assert_eq!(records[2], None, "its own delete");
    assert_eq!(records, LocalNodeStore.get_many(&txn, 0, &ids).unwrap());
}

/// A record whose decode does not fit the budget is refused before it is
/// copied or decoded: a 4 MiB record read from the memtable against 16 KiB
/// of budget allocates nothing near the record, and the budget is left as it
/// was. Copying it out and charging after the decode would allocate it
/// twice over first.
#[test]
fn a_record_past_the_budget_is_refused_before_it_is_decoded() {
    use coordinode_core::graph::node::NodeId;
    use coordinode_modality::{LocalNodeStore, NodeStore as _};
    let (engine, oracle, _dir) = engine();
    store_node(&engine, &oracle, 1, &sized_record(4 << 20));
    let txn = Transaction::begin(
        &engine,
        Some(&oracle),
        Timestamp::from_raw(engine.snapshot()),
    );
    let budget = QueryBudget::new(16 * 1024);
    let ids = [NodeId::from_raw(1)];
    let (refused, allocated) = allocated_by(|| {
        LocalNodeStore
            .get_many_charged(&txn, 0, &ids, &budget)
            .map(|_| ())
    });
    assert!(
        matches!(refused, Err(coordinode_modality::StoreError::Budget(_))),
        "{refused:?}"
    );
    assert!(
        allocated < BESIDE_THE_ROWS,
        "the refused record was copied or decoded: {allocated} bytes allocated"
    );
    assert_eq!(budget.memory_used(), 0, "nothing stays charged");
    assert!(budget.memory_peak() <= budget.memory_limit());
}

/// A refusal in the middle of a batch gives back every charge the batch
/// took: forty 16 KiB records against room for a few of them stop the read
/// with the budget's refusal, the budget never holds past its limit, and
/// nothing stays charged once the read returns.
#[test]
fn a_refusal_inside_a_batch_gives_back_every_charge() {
    use coordinode_core::graph::node::NodeId;
    use coordinode_modality::{LocalNodeStore, NodeStore as _};
    let (engine, oracle, _dir) = engine();
    for id in 1..=40 {
        store_node(&engine, &oracle, id, &sized_record(16 << 10));
    }
    let txn = Transaction::begin(
        &engine,
        Some(&oracle),
        Timestamp::from_raw(engine.snapshot()),
    );
    let budget = QueryBudget::new(128 * 1024);
    let ids: Vec<NodeId> = (1..=40).map(NodeId::from_raw).collect();
    let refused = LocalNodeStore.get_many_charged(&txn, 0, &ids, &budget);
    assert!(
        matches!(refused, Err(coordinode_modality::StoreError::Budget(_))),
        "{:?}",
        refused.map(|(records, _)| records.len())
    );
    assert!(budget.memory_peak() <= budget.memory_limit());
    assert!(
        budget.memory_peak() > 64 * 1024,
        "some records were read first"
    );
    assert_eq!(budget.memory_used(), 0, "every charge went back");
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

/// Publishing an implication change reads whole only the definitions on the
/// labels it moves, and the lookup: three hundred large definitions on
/// another label are passed over by their kind and label, their properties
/// never decoded. Decoding the catalog whole first would allocate every
/// one of them.
#[test]
fn publishing_a_change_passes_over_unrelated_definitions() {
    use coordinode_modality::index_def::{GenerationId, IndexDescriptor, IndexId};
    use coordinode_modality::{IndexStore as _, LocalIndexStore};
    let (engine, oracle, _dir) = engine();
    let store = LocalIndexStore::new(&engine);
    let mut txn = Transaction::begin(&engine, Some(&oracle), oracle.next());
    for n in 0..300u64 {
        let def = IndexDescriptor::compound(
            format!("other_{n}"),
            "Other",
            (0..40)
                .map(|p| format!("property_{p:03}_{}", "x".repeat(100)))
                .collect(),
        )
        .bind(IndexId::from_raw(n + 1), GenerationId::from_raw(n + 1));
        store.put_definition_txn(&mut txn, &def).unwrap();
    }
    txn.commit(&coordinode_storage::engine::transaction::CommitContext {
        write_concern: &coordinode_core::txn::write_concern::WriteConcern::majority(),
        pipeline: None,
        id_gen: None,
    })
    .unwrap();

    let label = LabelId::new(5).unwrap();
    let mut txn = Transaction::begin(&engine, Some(&oracle), oracle.next());
    let (proposed, allocated) = allocated_by(|| {
        coordinode_query::index::reconcile::propose_parts_txn(
            &engine,
            &mut txn,
            GenerationId::from_raw(10_000),
            &[(label, "Task".to_string())],
        )
    });
    proposed.expect("proposed");
    assert!(
        allocated < BESIDE_THE_ROWS * 2,
        "{allocated} bytes allocated passing over the catalog"
    );
}

/// Reporting a disagreement to an integrity record copies it only when the
/// record keeps it: one already held, or one past the evidence the record
/// keeps, allocates nothing, though the latter still moves the revision.
#[test]
fn a_report_the_record_does_not_keep_allocates_nothing() {
    use coordinode_modality::index_def::{GenerationId, IndexId, IndexIntegrityRecord, Mismatch};
    let extra = |node: u64| Mismatch::Extra {
        node,
        valid_from: None,
        tuple: vec![b'x'; 4096],
    };
    let mut record = IndexIntegrityRecord::new(IndexId::from_raw(1), GenerationId::from_raw(2));
    for node in 0..IndexIntegrityRecord::MAX_EVIDENCE as u64 {
        assert!(record.report(&extra(node)));
    }
    let held = extra(0);
    let (again, allocated) = allocated_by(|| record.report(&held));
    assert!(!again);
    assert_eq!(allocated, 0, "a disagreement held already was copied");
    let past = extra(u64::MAX);
    let revision = record.evidence_revision;
    let (new, allocated) = allocated_by(|| record.report(&past));
    assert!(new);
    assert_eq!(allocated, 0, "a disagreement past the evidence was copied");
    assert_eq!(record.evidence_revision, revision + 1);
}

/// What the registry takes from an integrity record is read without the
/// evidence it keeps: a record holding its full evidence of 4 KiB tuples
/// gives the same state as decoding it whole, and reading the state
/// allocates nothing.
#[test]
fn an_integrity_state_is_read_without_its_evidence() {
    use coordinode_modality::index_def::{
        GenerationId, IndexId, IndexIntegrityRecord, Integrity, Mismatch,
    };
    let mut record = IndexIntegrityRecord::new(IndexId::from_raw(1), GenerationId::from_raw(2));
    for node in 0..IndexIntegrityRecord::MAX_EVIDENCE as u64 {
        record.report(&Mismatch::Extra {
            node,
            valid_from: None,
            tuple: vec![b'x'; 4096],
        });
    }
    record.integrity = Integrity::Verified;
    record.trusted_from = Some(77);
    record.check = Some(coordinode_modality::index_def::IndexCheck {
        state: coordinode_modality::index_def::CheckState::Failed {
            reason: "x".repeat(4096),
        },
        ..coordinode_modality::index_def::IndexCheck::accepted(1)
    });
    let bytes = rmp_serde::to_vec(&record).unwrap();
    let (state, allocated) =
        allocated_by(|| IndexIntegrityRecord::state_from_msgpack(&bytes).unwrap());
    assert_eq!(state, record.state());
    assert_eq!(allocated, 0, "the evidence was decoded");
}

/// Restating an attempt's claims at commit allocates nothing, whatever it
/// drops: six scopes each holding two bounds whose trends become one (more
/// dropped than any inline list of positions would hold), and a scope held
/// under three revisions, whose positions stay several, keep one of each,
/// and the set still finds every claim it kept.
#[test]
fn restating_claims_allocates_nothing() {
    use coordinode_core::graph::node::NodeId;
    use coordinode_core::txn::invariant::{
        CardinalityBound, CardinalityMeasure, Claim, ClaimPredicate, ClaimScope, ClaimSet,
        CountTrend, Direction,
    };
    let scope = |node: u64| ClaimScope::Incident {
        node: NodeId::from_raw(node),
        edge_type: "OWNS".into(),
        direction: Direction::Outgoing,
    };
    let bound = |trend| ClaimPredicate::CardinalityBound {
        measure: CardinalityMeasure::EdgeInstances,
        bound: CardinalityBound::AtMostOne,
        trend,
    };
    let mut claims = ClaimSet::new();
    for node in 0..6 {
        claims.insert(Claim::new(scope(node), bound(CountTrend::Grows), 1));
        claims.insert(Claim::new(scope(node), bound(CountTrend::Shrinks), 1));
    }
    for revision in 1..=3 {
        claims.insert(Claim::new(scope(9), bound(CountTrend::Grows), revision));
    }
    assert_eq!(claims.len(), 15);

    let ((), allocated) = allocated_by(|| {
        claims.restate(|_, predicate| {
            if let ClaimPredicate::CardinalityBound { trend, .. } = predicate {
                *trend = CountTrend::Changes;
            }
        });
    });
    assert_eq!(allocated, 0, "restating allocated");
    assert_eq!(
        claims.len(),
        9,
        "one of each of six scopes, three revisions"
    );
    for node in 0..6 {
        assert!(!claims.insert(Claim::new(scope(node), bound(CountTrend::Changes), 1)));
    }
    for revision in 1..=3 {
        assert!(!claims.insert(Claim::new(scope(9), bound(CountTrend::Changes), revision)));
    }
    assert_eq!(claims.len(), 9);
}

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
        least_allocated_by(|| IndexDefinition::for_placement_bound(&bytes).unwrap());
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
        least_allocated_by(|| LabelSchema::implication_facts_bound(&bytes).expect("bound"));
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
