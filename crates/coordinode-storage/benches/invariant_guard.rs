//! What the invariant guard costs: throughput on a node everybody references,
//! the latency tail of a commit that has to state and decide conditions, and
//! the memory the guard holds while it does.
//!
//! The interesting case is skew. A popular node is referenced by most writes,
//! and the whole point of the claim model is that referencing it is a
//! compatible right rather than an exclusive one: the writes must not queue
//! behind each other. A mean throughput number would hide exactly that, so
//! this reports the distribution and the guard's own occupancy alongside it.
//!
//! Run: cargo bench -p coordinode-storage --bench invariant_guard

#![allow(clippy::expect_used, clippy::print_stdout)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use coordinode_core::graph::node::NodeId;
use coordinode_core::txn::invariant::{Adjacency, Claim, ClaimPredicate, ClaimScope};
use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::transaction::{CommitContext, Transaction};

/// Commits per measurement. Large enough for the tail to mean something,
/// small enough that the whole bench stays under a minute.
const COMMITS: usize = 4_000;

/// Writer threads for the skew measurement.
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

fn commit_ctx(wc: &WriteConcern) -> CommitContext<'_> {
    CommitContext {
        write_concern: wc,
        pipeline: None,
        id_gen: None,
        drain_buffer: None,
        nvme_write_buffer: None,
    }
}

/// One edge attached to `hub`, with or without the conditions that protect it.
fn attach(
    engine: &StorageEngine,
    oracle: &TimestampOracle,
    hub: NodeId,
    peer: u64,
    claims: bool,
) -> Duration {
    let snap = engine.snapshot();
    let mut txn = Transaction::new(
        engine,
        Some(oracle),
        coordinode_core::txn::timestamp::Timestamp::from_raw(snap),
        Some(snap),
    );

    let mut key = Vec::with_capacity(24);
    key.extend_from_slice(b"adj:OWNS:out:");
    key.extend_from_slice(&hub.as_raw().to_be_bytes());
    txn.merge_adj_add(&key, peer);

    if claims {
        let generation = txn.schema_generation();
        txn.claim(Claim::new(
            ClaimScope::Node(hub),
            ClaimPredicate::EndpointAlive,
            generation,
        ));
        txn.claim(Claim::new(
            ClaimScope::Node(NodeId::from_raw(peer)),
            ClaimPredicate::EndpointAlive,
            generation,
        ));
    }

    let wc = WriteConcern::default();
    let ctx = commit_ctx(&wc);
    let started = Instant::now();
    txn.commit(&ctx).expect("commit");
    started.elapsed()
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let rank = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[rank]
}

fn report(label: &str, mut samples: Vec<Duration>, wall: Duration) {
    samples.sort_unstable();
    let total: Duration = samples.iter().sum();
    let mean = total / samples.len().max(1) as u32;
    println!(
        "{label:<34} n={:>6}  mean={:>9.3?}  p50={:>9.3?}  p99={:>9.3?}  p999={:>9.3?}  \
         throughput={:>9.0}/s",
        samples.len(),
        mean,
        percentile(&samples, 0.50),
        percentile(&samples, 0.99),
        percentile(&samples, 0.999),
        samples.len() as f64 / wall.as_secs_f64(),
    );
}

/// Every writer references the same node. With claims this is the case the
/// model exists for: the references are compatible, so the writers must not
/// serialise against each other.
fn skewed_node(claims: bool, peers_exist: bool) {
    let dir = tempfile::tempdir().expect("tempdir");
    let (engine, oracle) = open(&dir);
    let hub = NodeId::from_raw(1);

    let hub_key = coordinode_core::graph::node::encode_node_key(engine.node_shard(), hub);
    engine
        .put(Partition::Node, &hub_key, b"hub")
        .expect("put hub");

    // Whether the endpoints being attached to already have rows. They do in
    // any ordinary statement, which resolves them before writing; they do not
    // on the bulk paths that write adjacency ahead of node rows. The two cost
    // different amounts, because an absent row falls through to the version
    // scan and the written-since probe, so both are measured rather than one
    // of them being reported as the price of the guard.
    if peers_exist {
        for w in 0..WRITERS {
            for i in 0..COMMITS / WRITERS {
                let peer = NodeId::from_raw((w * COMMITS + i + 2) as u64);
                let key = coordinode_core::graph::node::encode_node_key(engine.node_shard(), peer);
                engine.put(Partition::Node, &key, b"peer").expect("put");
            }
        }
    }

    let started = Instant::now();
    let samples: Vec<Duration> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..WRITERS)
            .map(|w| {
                let engine = Arc::clone(&engine);
                let oracle = Arc::clone(&oracle);
                scope.spawn(move || {
                    let mut local = Vec::with_capacity(COMMITS / WRITERS);
                    for i in 0..COMMITS / WRITERS {
                        let peer = (w * COMMITS + i + 2) as u64;
                        local.push(attach(&engine, &oracle, hub, peer, claims));
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
    let wall = started.elapsed();

    let label = match (claims, peers_exist) {
        (false, _) => "skewed node, unguarded",
        (true, true) => "skewed node, guarded",
        (true, false) => "skewed node, guarded (absent peers)",
    };
    report(label, samples, wall);
}

/// Every writer writes nodes of one label. With the schema read stated, each
/// commit carries the label's schema as a condition and decides it at commit;
/// the claims of the writers are compatible, so they must not serialise.
fn one_label_writes(schema_read: bool) {
    use coordinode_core::schema::definition::{
        LabelSchema, encode_label_current_revision_key, encode_label_schema_key,
    };

    let dir = tempfile::tempdir().expect("tempdir");
    let (engine, oracle) = open(&dir);
    let schema = LabelSchema::new_node_id("Doc");
    engine
        .put(
            Partition::Schema,
            &encode_label_schema_key("Doc", schema.schema_revision),
            &schema.to_msgpack().expect("encode"),
        )
        .expect("schema body");
    engine
        .put(
            Partition::Schema,
            &encode_label_current_revision_key("Doc"),
            &schema.schema_revision.to_be_bytes(),
        )
        .expect("schema pointer");

    let started = Instant::now();
    let samples: Vec<Duration> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..WRITERS)
            .map(|w| {
                let engine = Arc::clone(&engine);
                let oracle = Arc::clone(&oracle);
                let revision = schema.schema_revision;
                scope.spawn(move || {
                    let wc = WriteConcern::default();
                    let mut local = Vec::with_capacity(COMMITS / WRITERS);
                    for i in 0..COMMITS / WRITERS {
                        let snap = engine.snapshot();
                        let mut txn = Transaction::new(
                            engine.as_ref(),
                            Some(oracle.as_ref()),
                            coordinode_core::txn::timestamp::Timestamp::from_raw(snap),
                            Some(snap),
                        );
                        if schema_read {
                            txn.note_label_schema_read("Doc", revision);
                        }
                        let id = NodeId::from_raw((w * COMMITS + i + 2) as u64);
                        let key =
                            coordinode_core::graph::node::encode_node_key(engine.node_shard(), id);
                        txn.put(Partition::Node, &key, b"doc").expect("put");
                        let ctx = commit_ctx(&wc);
                        let at = Instant::now();
                        txn.commit(&ctx).expect("commit");
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
    let label = if schema_read {
        "one label, schema read stated"
    } else {
        "one label, no schema condition"
    };
    report(label, samples, started.elapsed());
}

/// What one attempt's conditions occupy while the commit decides them, and
/// what the table holds once it is done.
fn guard_memory() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (engine, oracle) = open(&dir);

    let snap = engine.snapshot();
    let mut txn = Transaction::new(
        engine.as_ref(),
        Some(oracle.as_ref()),
        coordinode_core::txn::timestamp::Timestamp::from_raw(snap),
        Some(snap),
    );

    // A statement that touches a large incident set states many conditions;
    // this is the shape the ceiling is sized against.
    let generation = txn.schema_generation();
    const REFERENCES: usize = 10_000;
    for peer in 0..REFERENCES as u64 {
        txn.claim(Claim::new(
            ClaimScope::Node(NodeId::from_raw(peer + 2)),
            ClaimPredicate::EndpointAlive,
            generation,
        ));
    }
    let bytes_per_claim = std::mem::size_of::<Claim>();
    println!(
        "guard occupancy                    claims={REFERENCES}  \
         struct={bytes_per_claim}B  set≈{}KiB (scope strings excluded)",
        (REFERENCES * bytes_per_claim) / 1024,
    );

    let wc = WriteConcern::default();
    let ctx = commit_ctx(&wc);
    txn.put(Partition::Node, b"node:guarded", b"v")
        .expect("put");
    let started = Instant::now();
    txn.commit(&ctx).expect("commit");
    println!(
        "deciding {REFERENCES} conditions            {:?}  \
         reserved after commit={}",
        started.elapsed(),
        engine.claim_registry().reserved_claims(),
    );
}

/// The refusal path: what a caller waits for before being told no. A refusal
/// has to be cheaper than the work it prevents, or the guard becomes the
/// bottleneck under contention.
fn refusal_latency() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (engine, oracle) = open(&dir);

    let mut key = Vec::new();
    key.extend_from_slice(b"adj:TAGGED:out:");
    key.extend_from_slice(&1u64.to_be_bytes());
    engine
        .merge(
            Partition::Adj,
            &key,
            &coordinode_storage::engine::merge::encode_add(2),
        )
        .expect("merge");

    let wc = WriteConcern::default();
    let mut samples = Vec::with_capacity(COMMITS);
    let started = Instant::now();
    for _ in 0..COMMITS {
        let snap = engine.snapshot();
        let mut txn = Transaction::new(
            engine.as_ref(),
            Some(oracle.as_ref()),
            coordinode_core::txn::timestamp::Timestamp::from_raw(snap),
            Some(snap),
        );
        txn.claim(Claim::new(
            ClaimScope::Pair {
                source: NodeId::from_raw(1),
                target: NodeId::from_raw(2),
                edge_type: "TAGGED".to_string(),
            },
            ClaimPredicate::PairAdjacency {
                observed: Adjacency::Absent,
            },
            txn.schema_generation(),
        ));
        txn.put(Partition::Node, b"node:never", b"v").expect("put");

        let ctx = commit_ctx(&wc);
        let at = Instant::now();
        txn.commit(&ctx).expect_err("the observation does not hold");
        samples.push(at.elapsed());
    }
    report("refusal", samples, started.elapsed());
}

/// The field id the unique-value measurements index.
const EMAIL: u32 = 1;

/// A unique index over `:User(email)`, as a claim states it.
fn email_interpretation() -> coordinode_core::index::derive::IndexInterpretation {
    use coordinode_core::index::derive::{IndexInterpretation, KEY_CODEC, PropertyRef};
    IndexInterpretation {
        codec: KEY_CODEC,
        generation: coordinode_core::index::identity::GenerationId::from_raw(3),
        unique: true,
        sparse: false,
        properties: vec![PropertyRef {
            field: Some(EMAIL),
            name: "email".into(),
        }],
        filter: None,
    }
}

/// `stored` users with distinct emails, as a unique index's build finds them.
fn store_users(engine: &StorageEngine, stored: u64) {
    use coordinode_core::graph::node::NodeRecord;
    use coordinode_core::graph::types::Value;
    for id in 1..=stored {
        let mut record = NodeRecord::new("User");
        record.set(EMAIL, Value::String(format!("{id}@stored")));
        let key = coordinode_core::graph::node::encode_node_key(
            engine.node_shard(),
            NodeId::from_raw(id),
        );
        engine
            .put(Partition::Node, &key, &record.to_msgpack().expect("encode"))
            .expect("put user");
    }
}

/// One new user taking a free email while the index is being built, the
/// build having covered all but the last `uncovered` of `stored` users.
fn take_free_email(
    engine: &StorageEngine,
    oracle: &TimestampOracle,
    id: u64,
    stored: u64,
    uncovered: u64,
) -> Duration {
    use coordinode_core::graph::node::{NodeRecord, encode_node_key};
    use coordinode_core::graph::types::Value;
    use coordinode_core::index::derive::tuples;
    use coordinode_core::txn::invariant::UncoveredSource;

    let snap = engine.snapshot();
    let mut txn = Transaction::new(
        engine,
        Some(oracle),
        coordinode_core::txn::timestamp::Timestamp::from_raw(snap),
        Some(snap),
    );
    let email = Value::String(format!("{id}@new"));
    let mut record = NodeRecord::new("User");
    record.set(EMAIL, email.clone());
    let shard = engine.node_shard();
    txn.put(
        Partition::Node,
        &encode_node_key(shard, NodeId::from_raw(id)),
        &record.to_msgpack().expect("encode"),
    )
    .expect("put");
    let interpretation = email_interpretation();
    let generation = txn.schema_generation();
    for tuple in tuples(&[email]) {
        txn.claim(Claim::new(
            ClaimScope::UniqueValue {
                generation: interpretation.generation,
                tuple,
            },
            ClaimPredicate::UniqueHolder {
                node: NodeId::from_raw(id),
                uncovered: Some(Box::new(UncoveredSource {
                    shard_id: shard,
                    label: "User".into(),
                    interpretation: interpretation.clone(),
                    covered_through: (uncovered < stored)
                        .then(|| encode_node_key(shard, NodeId::from_raw(stored - uncovered))),
                    read_limit: u64::MAX,
                })),
            },
            generation,
        ));
    }
    let wc = WriteConcern::default();
    let ctx = commit_ctx(&wc);
    let started = Instant::now();
    txn.commit(&ctx).expect("commit");
    started.elapsed()
}

/// What proving a unique value free costs a writer while the index is being
/// built, by how many stored nodes the build has not reached: the commit
/// reads exactly those. Users written by the measurement land after the
/// stored ones, so they add to the read as a real build's concurrent writes
/// would.
fn unique_while_building(stored: u64, uncovered: u64, commits: usize) {
    let dir = tempfile::tempdir().expect("tempdir");
    let (engine, oracle) = open(&dir);
    store_users(&engine, stored);
    engine.persist().expect("flush");

    let started = Instant::now();
    let samples: Vec<Duration> = (0..commits as u64)
        .map(|i| take_free_email(&engine, &oracle, stored + 1 + i, stored, uncovered))
        .collect();
    report(
        &format!("unique building, {uncovered}+{commits}/2 unread"),
        samples,
        started.elapsed(),
    );
}

/// Writers that state no unique claim, committing beside one that reads
/// `uncovered` stored nodes per commit: whether the read holds them up.
fn bystanders_beside_unique_reads(stored: u64, uncovered: u64) {
    let dir = tempfile::tempdir().expect("tempdir");
    let (engine, oracle) = open(&dir);
    store_users(&engine, stored);
    engine.persist().expect("flush");
    let done = std::sync::atomic::AtomicBool::new(false);

    let started = Instant::now();
    let samples: Vec<Duration> = std::thread::scope(|scope| {
        let claimant = {
            let (engine, oracle, done) = (&engine, &oracle, &done);
            scope.spawn(move || {
                let mut id = 10 * stored;
                while !done.load(std::sync::atomic::Ordering::Relaxed) {
                    id += 1;
                    take_free_email(engine, oracle, id, stored, uncovered);
                }
            })
        };
        let handles: Vec<_> = (0..WRITERS)
            .map(|w| {
                let engine = Arc::clone(&engine);
                let oracle = Arc::clone(&oracle);
                scope.spawn(move || {
                    let wc = WriteConcern::default();
                    // A node of another label: the claimant reads it and
                    // passes over it.
                    let doc = coordinode_core::graph::node::NodeRecord::new("Doc")
                        .to_msgpack()
                        .expect("encode");
                    let mut local = Vec::with_capacity(COMMITS / WRITERS);
                    for i in 0..COMMITS / WRITERS {
                        // From the start of the transaction: a writer held up
                        // waiting for its snapshot waits as surely as one held
                        // up in its commit.
                        let at = Instant::now();
                        let snap = engine.snapshot();
                        let mut txn = Transaction::new(
                            engine.as_ref(),
                            Some(oracle.as_ref()),
                            coordinode_core::txn::timestamp::Timestamp::from_raw(snap),
                            Some(snap),
                        );
                        let id = NodeId::from_raw((20 * stored) + (w * COMMITS + i) as u64);
                        let key =
                            coordinode_core::graph::node::encode_node_key(engine.node_shard(), id);
                        txn.put(Partition::Node, &key, &doc).expect("put");
                        let ctx = commit_ctx(&wc);
                        txn.commit(&ctx).expect("commit");
                        local.push(at.elapsed());
                    }
                    local
                })
            })
            .collect();
        let samples = handles
            .into_iter()
            .flat_map(|h| h.join().expect("writer"))
            .collect();
        done.store(true, std::sync::atomic::Ordering::Relaxed);
        claimant.join().expect("claimant");
        samples
    });
    report(
        &format!("bystanders, claimant reads {uncovered}"),
        samples,
        started.elapsed(),
    );
}

fn main() {
    println!("== invariant guard ==");
    skewed_node(false, true);
    skewed_node(true, true);
    skewed_node(true, false);
    one_label_writes(false);
    one_label_writes(true);
    guard_memory();
    refusal_latency();
    const STORED: u64 = 100_000;
    unique_while_building(STORED, 0, 2_000);
    unique_while_building(STORED, 1_000, 1_000);
    unique_while_building(STORED, 10_000, 200);
    unique_while_building(STORED, 100_000, 30);
    bystanders_beside_unique_reads(STORED, 0);
    bystanders_beside_unique_reads(STORED, 10_000);
    bystanders_beside_unique_reads(STORED, 100_000);
}
