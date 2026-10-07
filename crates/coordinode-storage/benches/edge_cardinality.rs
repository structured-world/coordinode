//! What keeping exact relationship cardinality costs: the commit of an edge
//! whose type has a declared constraint against one whose type has none, on
//! a node every writer attaches to; the tail when every writer also decides
//! a bound over that node; a pair every writer adds instances to; the bytes
//! the counts hold; and rebuilding a constraint over stored edges.
//!
//! Skew is the case that matters. Counting a pair is exclusive only for that
//! pair, so writers attaching distinct neighbours to one hub must not queue
//! behind each other; a bound over the hub is one predicate every writer
//! decides, and those do coordinate. Both are reported as distributions, with
//! the refusals a writer retried.
//!
//! Run: cargo bench -p coordinode-storage --bench edge_cardinality

#![allow(clippy::expect_used, clippy::print_stdout)]

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use coordinode_core::graph::cardinality::{
    CardinalityBound, CardinalityDescriptor, CardinalityMeasure, Direction,
    encode_cardinality_profile_key,
};
use coordinode_core::graph::edge::{
    encode_adj_key_forward, encode_adj_key_reverse, temporal_edgeprop_pair_prefix,
};
use coordinode_core::graph::node::NodeId;
use coordinode_core::schema::definition::{
    EdgeTypeSchema, PropertyDef, PropertyType, encode_edge_type_current_revision_key,
    encode_edge_type_schema_key,
};
use coordinode_core::txn::invariant::{Claim, ClaimPredicate, ClaimScope};
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::transaction::{CommitContext, CommitError, Transaction};

/// Commits per measurement.
const COMMITS: usize = 4_000;

/// Writer threads.
const WRITERS: usize = 8;

fn open(dir: &tempfile::TempDir) -> (Arc<StorageEngine>, Arc<TimestampOracle>) {
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let oracle = Arc::new(TimestampOracle::new());
    let engine = Arc::new(StorageEngine::open_with_oracle(&config, oracle.clone()).expect("open"));
    (engine, oracle)
}

fn descriptor(direction: Direction, measure: CardinalityMeasure) -> CardinalityDescriptor {
    CardinalityDescriptor {
        edge_type: "OWNS".into(),
        direction,
        measure,
        bound: CardinalityBound::AtLeastOne,
        schema_generation: 1,
    }
}

/// Define `OWNS`, declaring both measures on both sides when `counted`, and
/// cover the counts.
fn define(engine: &StorageEngine, oracle: &TimestampOracle, discriminated: bool, counted: bool) {
    let mut schema = EdgeTypeSchema::new("OWNS");
    schema.add_property(PropertyDef::new("context", PropertyType::String).not_null());
    schema
        .resolve_identity(discriminated.then_some("context"))
        .expect("identity");
    let mut declared = Vec::new();
    if counted {
        for direction in [Direction::Outgoing, Direction::Incoming] {
            for measure in [
                CardinalityMeasure::EdgeInstances,
                CardinalityMeasure::DistinctNeighbours,
            ] {
                let d = descriptor(direction, measure);
                schema.declare_cardinality(d.clone()).expect("declare");
                declared.push(d);
            }
        }
    }
    engine
        .put(
            Partition::Schema,
            &encode_edge_type_schema_key("OWNS", 1),
            &schema.to_msgpack().expect("encode"),
        )
        .expect("body");
    engine
        .put(
            Partition::Schema,
            &encode_edge_type_current_revision_key("OWNS"),
            &1u64.to_be_bytes(),
        )
        .expect("pointer");
    if let Some(profile) = schema.cardinality_profile() {
        engine
            .put(
                Partition::Schema,
                &encode_cardinality_profile_key("OWNS"),
                &profile.to_msgpack().expect("encode"),
            )
            .expect("profile");
    }
    for d in &declared {
        let mut txn = begin(engine, oracle);
        coordinode_storage::engine::cardinality::rebuild(&mut txn, d).expect("rebuild");
        commit(&mut txn).expect("cover");
    }
}

fn begin<'a>(engine: &'a StorageEngine, oracle: &'a TimestampOracle) -> Transaction<'a> {
    let snap = engine.snapshot();
    Transaction::new(engine, Some(oracle), Timestamp::from_raw(snap), Some(snap))
}

fn commit(txn: &mut Transaction<'_>) -> Result<(), CommitError> {
    let wc = WriteConcern::default();
    let ctx = CommitContext {
        write_concern: &wc,
        pipeline: None,
        id_gen: None,
        drain_buffer: None,
        nvme_write_buffer: None,
    };
    txn.commit(&ctx).map(|_| ())
}

/// Stage one instance of `(source, target)`.
fn stage(txn: &mut Transaction<'_>, source: u64, target: u64, disc: Option<u64>) {
    txn.merge_adj_add(
        &encode_adj_key_forward("OWNS", NodeId::from_raw(source)),
        target,
    );
    txn.merge_adj_add(
        &encode_adj_key_reverse("OWNS", NodeId::from_raw(target)),
        source,
    );
    if let Some(disc) = disc {
        let mut key = temporal_edgeprop_pair_prefix(
            "OWNS",
            NodeId::from_raw(source),
            NodeId::from_raw(target),
        );
        key.extend_from_slice(&disc.to_be_bytes());
        txn.put(Partition::EdgeProp, &key, b"facets").expect("put");
    }
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn report(label: &str, mut samples: Vec<Duration>, wall: Duration, retries: u64) {
    samples.sort_unstable();
    println!(
        "{label:<40} n={:>5}  p50={:>9.3?}  p99={:>9.3?}  p999={:>9.3?}  \
         throughput={:>8.0}/s  retries={retries}",
        samples.len(),
        percentile(&samples, 0.50),
        percentile(&samples, 0.99),
        percentile(&samples, 0.999),
        samples.len() as f64 / wall.as_secs_f64(),
    );
}

/// Every writer attaches a new neighbour to node 1, retrying a refused
/// attempt until it lands; each sample is one edge from first attempt to
/// commit. `bound` also decides AT LEAST ONE over the hub's outgoing edges.
fn hub(label: &str, counted: bool, bound: bool) {
    let dir = tempfile::tempdir().expect("tempdir");
    let (engine, oracle) = open(&dir);
    define(&engine, &oracle, false, counted);
    let retries = AtomicU64::new(0);

    let started = Instant::now();
    let samples: Vec<Duration> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..WRITERS)
            .map(|w| {
                let (engine, oracle, retries) = (&engine, &oracle, &retries);
                scope.spawn(move || {
                    let mut local = Vec::with_capacity(COMMITS / WRITERS);
                    for i in 0..COMMITS / WRITERS {
                        let peer = (2 + w * COMMITS + i) as u64;
                        let at = Instant::now();
                        loop {
                            let mut txn = begin(engine, oracle);
                            stage(&mut txn, 1, peer, None);
                            if bound {
                                let generation = txn.schema_generation();
                                txn.claim(Claim::new(
                                    ClaimScope::Incident {
                                        node: NodeId::from_raw(1),
                                        edge_type: "OWNS".into(),
                                        direction: Direction::Outgoing,
                                    },
                                    ClaimPredicate::CardinalityBound {
                                        measure: CardinalityMeasure::DistinctNeighbours,
                                        bound: CardinalityBound::AtLeastOne,
                                    },
                                    generation,
                                ));
                            }
                            match commit(&mut txn) {
                                Ok(()) => break,
                                Err(_) => {
                                    retries.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        }
                        local.push(at.elapsed());
                    }
                    local
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("writer"))
            .collect()
    });
    report(
        label,
        samples,
        started.elapsed(),
        retries.load(Ordering::Relaxed),
    );
}

/// Every writer adds a new instance to the same discriminated pair: the one
/// shape that counting serialises, since each instance changes that pair's
/// counts.
fn hot_pair() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (engine, oracle) = open(&dir);
    define(&engine, &oracle, true, true);
    let retries = AtomicU64::new(0);
    let started = Instant::now();
    let samples: Vec<Duration> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..WRITERS)
            .map(|w| {
                let (engine, oracle, retries) = (&engine, &oracle, &retries);
                scope.spawn(move || {
                    let mut local = Vec::with_capacity(COMMITS / WRITERS / 4);
                    for i in 0..COMMITS / WRITERS / 4 {
                        let disc = (w * COMMITS + i) as u64;
                        let at = Instant::now();
                        loop {
                            let mut txn = begin(engine, oracle);
                            stage(&mut txn, 1, 2, Some(disc));
                            if commit(&mut txn).is_ok() {
                                break;
                            }
                            retries.fetch_add(1, Ordering::Relaxed);
                        }
                        local.push(at.elapsed());
                    }
                    local
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("writer"))
            .collect()
    });
    report(
        "hot discriminated pair, counted",
        samples,
        started.elapsed(),
        retries.load(Ordering::Relaxed),
    );
}

/// The bytes one kept count holds, and rebuilding a constraint over `edges`
/// stored edges spread over `edges / 8` source nodes.
fn state_and_rebuild(edges: u64) {
    let dir = tempfile::tempdir().expect("tempdir");
    let (engine, oracle) = open(&dir);
    define(&engine, &oracle, false, false);
    let mut txn = begin(&engine, &oracle);
    for e in 0..edges {
        stage(&mut txn, 1 + e / 8, 1_000_000 + e, None);
        if e % 10_000 == 9_999 {
            commit(&mut txn).expect("load");
            txn = begin(&engine, &oracle);
        }
    }
    commit(&mut txn).expect("load");
    engine.persist().expect("flush");

    let d = descriptor(Direction::Outgoing, CardinalityMeasure::EdgeInstances);
    let mut schema = EdgeTypeSchema::new("OWNS");
    schema.add_property(PropertyDef::new("context", PropertyType::String).not_null());
    schema.resolve_identity(None).expect("identity");
    schema.declare_cardinality(d.clone()).expect("declare");
    engine
        .put(
            Partition::Schema,
            &encode_cardinality_profile_key("OWNS"),
            &schema
                .cardinality_profile()
                .expect("profile")
                .to_msgpack()
                .expect("encode"),
        )
        .expect("profile");

    let started = Instant::now();
    let mut txn = begin(&engine, &oracle);
    let rebuilt = coordinode_storage::engine::cardinality::rebuild(&mut txn, &d).expect("rebuild");
    commit(&mut txn).expect("cover");
    let took = started.elapsed();
    let key = d.counter_key(NodeId::from_raw(1)).len();
    println!(
        "rebuild over {edges} edges                 scopes={}  took={took:.3?}  \
         per scope: key={key}B value=8B",
        rebuilt.scopes,
    );
}

fn main() {
    println!("== edge cardinality ==");
    hub("hub, type without a constraint", false, false);
    hub("hub, counted", true, false);
    hub("hub, counted, bound decided by each", true, true);
    hot_pair();
    state_and_rebuild(100_000);
    state_and_rebuild(1_000_000);
}
