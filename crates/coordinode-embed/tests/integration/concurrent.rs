//! Integration tests: concurrent UPSERT with CAS conflict detection.
//!
//! Verifies that when multiple threads execute UPSERTs on the same node,
//! the CAS mechanism prevents lost updates and data corruption.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

use coordinode_core::graph::intern::FieldInterner;
use coordinode_core::graph::node::{NodeId, NodeIdAllocator};
use coordinode_core::graph::types::Value;
use coordinode_query::cypher::ast::Expr;
use coordinode_query::executor::runner::{ExecutionError, execute};
use coordinode_query::plan::SetItem;
use coordinode_query::planner::logical::{LogicalOp, LogicalPlan};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;

/// Lower a cypher test expression into the neutral IR carried by operator fields.
fn nx(e: Expr) -> coordinode_query::plan::expr::Expr {
    coordinode_query::planner::lower_expr(&e).expect("lower test expression")
}

/// Helper: create an UPSERT plan that matches User by name and sets age.
fn upsert_plan(name: &str, age: i64) -> LogicalPlan {
    LogicalPlan {
        snapshot_ts: None,
        vector_consistency: coordinode_core::graph::types::VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Upsert {
            pattern: Box::new(LogicalOp::NodeScan {
                variable: "n".into(),
                labels: vec!["User".into()],
                property_filters: vec![(
                    "name".into(),
                    nx(Expr::Literal(Value::String(name.to_string()))),
                )],
            }),
            on_match: vec![SetItem::Property {
                variable: "n".into(),
                property: "age".into(),
                expr: nx(Expr::Literal(Value::Int(age))),
            }],
            on_create_patterns: vec![],
        },
    }
}

/// Helper: create a CREATE plan for a User node.
fn create_user_plan(name: &str, age: i64) -> LogicalPlan {
    LogicalPlan {
        snapshot_ts: None,
        vector_consistency: coordinode_core::graph::types::VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::CreateNode {
            input: None,
            variable: Some("n".into()),
            labels: vec!["User".into()],
            properties: vec![
                ("name".into(), nx(Expr::Literal(Value::String(name.into())))),
                ("age".into(), nx(Expr::Literal(Value::Int(age)))),
            ],
        },
    }
}

/// Helper: read a User node's age by scanning the node: partition directly.
///
/// Uses raw storage scan with a shared interner to resolve field names correctly.
/// The interner must be the same one used during writes (or contain the same mappings).
fn read_user_age(
    engine: &StorageEngine,
    interner: &mut FieldInterner,
    allocator: &NodeIdAllocator,
    target_name: &str,
) -> Option<i64> {
    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: coordinode_core::graph::types::VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Project {
            input: Box::new(LogicalOp::NodeScan {
                variable: "n".into(),
                labels: vec!["User".into()],
                property_filters: vec![(
                    "name".into(),
                    nx(Expr::Literal(Value::String(target_name.into()))),
                )],
            }),
            items: vec![coordinode_query::planner::logical::ProjectItem {
                alias: Some("n.age".into()),
                expr: nx(Expr::PropertyAccess {
                    expr: Box::new(Expr::Variable("n".into())),
                    property: "age".into(),
                }),
            }],
            distinct: false,
        },
    };
    let mut ctx = super::helpers::make_ctx_legacy(engine, interner, allocator);
    let rows = execute(&plan, &mut ctx).expect("read");
    rows.first().and_then(|r| match r.get("n.age") {
        Some(Value::Int(age)) => Some(*age),
        _ => None,
    })
}

/// Concurrent UPSERTs on the same node: no data corruption, CAS detects conflicts.
///
/// 4 threads × 50 iterations = 200 UPSERTs on the same node.
/// Each thread tries to set n.age = thread_id.
/// Results: success + conflict = 200 total, final age is one of 0-3.
#[test]
fn concurrent_upsert_data_consistency() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = Arc::new(StorageEngine::open(&config).expect("open"));
    let allocator = Arc::new(NodeIdAllocator::resume_from(NodeId::from_raw(1000)));

    // Create initial node: Alice, age=0.
    // Keep the interner alive for the read_user_age call at the end.
    let mut setup_interner = FieldInterner::new();
    {
        let plan = create_user_plan("Alice", 0);
        let mut ctx = super::helpers::make_ctx_legacy(&engine, &mut setup_interner, &allocator);
        execute(&plan, &mut ctx).expect("create Alice");
    }

    let num_threads = 4u64;
    let iterations_per_thread = 50u64;
    let conflict_count = Arc::new(AtomicU64::new(0));
    let success_count = Arc::new(AtomicU64::new(0));

    // Serialize the interner so each thread can reconstruct the same field mappings.
    let interner_bytes = Arc::new(setup_interner.to_bytes().expect("serialize interner"));

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let engine = Arc::clone(&engine);
            let allocator = Arc::clone(&allocator);
            let conflicts = Arc::clone(&conflict_count);
            let successes = Arc::clone(&success_count);
            let interner_bytes = Arc::clone(&interner_bytes);

            thread::spawn(move || {
                let mut interner =
                    FieldInterner::from_bytes(&interner_bytes).expect("deserialize interner");

                for _ in 0..iterations_per_thread {
                    let plan = upsert_plan("Alice", thread_id as i64);
                    let mut ctx =
                        super::helpers::make_ctx_legacy(&engine, &mut interner, &allocator);
                    match execute(&plan, &mut ctx) {
                        Ok(_) => {
                            successes.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(ExecutionError::Conflict(_)) => {
                            conflicts.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(e) => {
                            panic!("unexpected error in thread {thread_id}: {e}");
                        }
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("thread should not panic");
    }

    let total_successes = success_count.load(Ordering::Relaxed);
    let total_conflicts = conflict_count.load(Ordering::Relaxed);
    let total_expected = num_threads * iterations_per_thread;

    // Every execution must result in either success or conflict — no other outcome.
    assert_eq!(
        total_successes + total_conflicts,
        total_expected,
        "all executions must be accounted for: {total_successes} success + {total_conflicts} conflict != {total_expected}"
    );

    // At least some operations must succeed (the first iteration always succeeds).
    assert!(total_successes > 0, "at least one UPSERT should succeed");

    // Verify data consistency: final age must be one of the thread IDs.
    let final_age = read_user_age(&engine, &mut setup_interner, &allocator, "Alice")
        .expect("Alice should exist");
    assert!(
        (0..num_threads as i64).contains(&final_age),
        "final age ({final_age}) must be one of the thread IDs (0-{num_threads})"
    );

    // Log results for visibility.
    eprintln!(
        "Concurrent UPSERT results: {total_successes} success, {total_conflicts} conflicts out of {total_expected} total"
    );
}

/// Auto-commit read-modify-write from several threads on one node, through the
/// public API: every increment that committed a change is in the final value.
///
/// A statement reads at the timestamp it is given, and that timestamp can
/// already cover a commit that has taken its number but not applied. Read
/// there, the statement misses that commit; if the commit lands before this
/// one registers its own write, nothing is in flight to collide with, and a
/// validation that looks only at writes after the snapshot finds nothing
/// either. Both report success and one increment is gone.
///
/// Only a response that set the property counts as an increment: a
/// statement that matched no node succeeds with nothing written, and counting
/// it would report a lost update that never happened. Such statements are
/// counted apart and retried, like refusals, so the counted increments are
/// exactly what the value must reach, and a statement that failed to find a
/// node that exists is reported as what it is.
#[test]
fn concurrent_auto_commit_increments_lose_nothing() {
    use coordinode_embed::{Database, DatabaseError};
    const THREADS: u64 = 8;
    const PER_THREAD: u64 = 100;

    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    db.execute_cypher("CREATE (:Tally {id: 0, v: 0})")
        .expect("seed");

    let committed = AtomicU64::new(0);
    let refused = AtomicU64::new(0);
    let matched_nothing = AtomicU64::new(0);
    thread::scope(|s| {
        for _ in 0..THREADS {
            let (db, committed, refused, matched_nothing) =
                (&db, &committed, &refused, &matched_nothing);
            s.spawn(move || {
                let mut done = 0;
                while done < PER_THREAD {
                    match db.execute_cypher_shared(
                        "MATCH (n:Tally {id: 0}) SET n.v = n.v + 1",
                        None,
                        None,
                        None,
                        None,
                    ) {
                        Ok(result) if result.write_stats.properties_set == 1 => {
                            done += 1;
                            committed.fetch_add(1, Ordering::Relaxed);
                        }
                        Ok(result) => {
                            assert_eq!(
                                result.write_stats.properties_set, 0,
                                "one node, so at most one property is set"
                            );
                            matched_nothing.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(DatabaseError::Execution(ExecutionError::Conflict(_))) => {
                            refused.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(e) => panic!("unexpected failure: {e}"),
                    }
                }
            });
        }
    });

    let rows = db
        .execute_cypher("MATCH (n:Tally {id: 0}) RETURN n.v AS v")
        .expect("read tally");
    let committed = committed.into_inner();
    let matched_nothing = matched_nothing.into_inner();
    assert_eq!(committed, THREADS * PER_THREAD);
    assert_eq!(
        rows[0].get("v"),
        Some(&Value::Int(i64::try_from(committed).expect("fits"))),
        "{committed} increments committed ({} refused and retried, {matched_nothing} \
         matched no node), the value must hold all of them",
        refused.into_inner()
    );
    assert_eq!(
        matched_nothing, 0,
        "the node exists throughout, yet {matched_nothing} statements did not find it"
    );
}

/// Two writers, one known key, driven directly through the transaction
/// layer: both read the same version, both stage an increment, and at most
/// one of them may commit. The second must be refused, whatever order the
/// commits run in, because its write was computed from a base the first one
/// replaced.
#[test]
fn two_writers_on_one_key_cannot_both_commit_from_the_same_base() {
    use coordinode_core::txn::proposal::ProposalIdGenerator;
    use coordinode_core::txn::timestamp::TimestampOracle;
    use coordinode_core::txn::write_concern::WriteConcern;
    use coordinode_raft::proposal::OwnedLocalProposalPipeline;
    use coordinode_storage::engine::partition::Partition;
    use coordinode_storage::engine::transaction::{CommitContext, CommitError, Transaction};

    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = Arc::new(TimestampOracle::new());
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine =
        Arc::new(StorageEngine::open_embedded(&config, Arc::clone(&oracle)).expect("open"));
    let pipeline = OwnedLocalProposalPipeline::new(&engine);
    let ids = ProposalIdGenerator::new();
    let concern = WriteConcern::majority();
    let ctx = CommitContext {
        pipeline: Some(&pipeline),
        id_gen: Some(&ids),
        write_concern: &concern,
        drain_buffer: None,
        nvme_write_buffer: None,
    };
    let key = b"node:\0\x01:counter".to_vec();

    // Seed the base both writers read.
    let mut seed = Transaction::begin(&engine, Some(&oracle), oracle.next());
    seed.put(Partition::Node, &key, b"0").expect("stage seed");
    seed.commit(&ctx).expect("seed commits");

    for first_commits_first in [true, false] {
        let mut a = Transaction::begin(&engine, Some(&oracle), oracle.next());
        let mut b = Transaction::begin(&engine, Some(&oracle), oracle.next());
        let base_a = a.get(Partition::Node, &key).expect("a reads");
        let base_b = b.get(Partition::Node, &key).expect("b reads");
        assert_eq!(base_a, base_b, "both writers start from the same version");
        a.put(Partition::Node, &key, b"a").expect("a stages");
        b.put(Partition::Node, &key, b"b").expect("b stages");

        let (first, second) = if first_commits_first {
            (&mut a, &mut b)
        } else {
            (&mut b, &mut a)
        };
        first.commit(&ctx).expect("the first writer commits");
        match second.commit(&ctx) {
            Err(CommitError::Conflict(_)) => {}
            other => {
                panic!("the second writer committed over the first from the same base: {other:?}")
            }
        }
    }
}

/// A reader never sees part of a commit: batches of a thousand writes each are
/// observed whole or not at all, however the reads interleave with them.
#[test]
fn a_reader_sees_a_thousand_write_commit_whole_or_not_at_all() {
    use coordinode_core::txn::proposal::ProposalIdGenerator;
    use coordinode_core::txn::timestamp::TimestampOracle;
    use coordinode_core::txn::write_concern::WriteConcern;
    use coordinode_raft::proposal::OwnedLocalProposalPipeline;
    use coordinode_storage::engine::partition::Partition;
    use coordinode_storage::engine::transaction::{CommitContext, Transaction};
    use std::sync::atomic::AtomicBool;

    const WRITES: usize = 1000;
    const BATCHES: usize = 20;
    const PREFIX: &[u8] = b"node:\0\x01:batch:";

    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = Arc::new(TimestampOracle::new());
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine =
        Arc::new(StorageEngine::open_embedded(&config, Arc::clone(&oracle)).expect("open"));
    let pipeline = OwnedLocalProposalPipeline::new(&engine);
    let ids = ProposalIdGenerator::new();
    let concern = WriteConcern::majority();
    let done = AtomicBool::new(false);

    thread::scope(|scope| {
        let reader = scope.spawn(|| {
            let mut reads = 0u64;
            while !done.load(Ordering::Acquire) {
                let txn = Transaction::begin(&engine, Some(&oracle), oracle.next());
                let seen = txn
                    .prefix_scan(Partition::Node, PREFIX)
                    .expect("scan")
                    .len();
                assert_eq!(
                    seen % WRITES,
                    0,
                    "a reader saw {seen} keys, part of a {WRITES}-write commit"
                );
                reads += 1;
            }
            reads
        });

        let ctx = CommitContext {
            pipeline: Some(&pipeline),
            id_gen: Some(&ids),
            write_concern: &concern,
            drain_buffer: None,
            nvme_write_buffer: None,
        };
        for batch in 0..BATCHES {
            let mut txn = Transaction::begin(&engine, Some(&oracle), oracle.next());
            for i in 0..WRITES {
                let mut key = PREFIX.to_vec();
                key.extend_from_slice(&((batch * WRITES + i) as u64).to_be_bytes());
                txn.put(Partition::Node, &key, b"x").expect("stage");
            }
            txn.commit(&ctx).expect("the batch commits");
        }
        done.store(true, Ordering::Release);
        let reads = reader.join().expect("the reader saw only whole commits");
        assert!(
            reads > 0,
            "the reader never read while the batches committed"
        );
    });

    let txn = Transaction::begin(&engine, Some(&oracle), oracle.next());
    assert_eq!(
        txn.prefix_scan(Partition::Node, PREFIX)
            .expect("scan")
            .len(),
        WRITES * BATCHES
    );
}

/// An auto-commit statement refused at commit because another commit in
/// flight holds its node (here one conditioned on the node, as a background
/// index repair is) applied nothing, and runs again once that commit lands:
/// the caller sees it succeed, not a conflict it never caused.
#[test]
fn an_auto_commit_statement_refused_by_a_commit_in_flight_runs_again() {
    use coordinode_core::graph::node::encode_node_key;
    use coordinode_embed::Database;
    use coordinode_storage::engine::partition::Partition;

    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    db.execute_cypher("CREATE (:U {name: 'a', n: 1})")
        .expect("create");
    let id = match db
        .execute_cypher("MATCH (u:U) RETURN id(u) AS id")
        .expect("id")[0]
        .get("id")
    {
        Some(Value::Int(id)) => u64::try_from(*id).expect("id"),
        other => panic!("expected an id, got {other:?}"),
    };
    let node = encode_node_key(1, NodeId::from_raw(id));

    let (held, release) = std::sync::mpsc::channel::<()>();
    let pending = Arc::clone(db.engine().pending_commits());
    thread::scope(|scope| {
        let holder = scope.spawn(move || {
            let (_ts, admission) = pending
                .admit_allocated(|| u64::MAX - 1, Vec::new(), vec![(Partition::Node, node)])
                .expect("admit the holder");
            held.send(()).expect("signal");
            std::thread::sleep(std::time::Duration::from_millis(20));
            drop(admission);
        });
        release.recv().expect("held");
        db.execute_cypher("MATCH (u:U {name: 'a'}) SET u.n = 2")
            .expect("the statement lands once the holder has");
        holder.join().expect("holder");
    });
    let rows = db
        .execute_cypher("MATCH (u:U) RETURN u.n AS n")
        .expect("read");
    assert_eq!(rows[0].get("n"), Some(&Value::Int(2)));
}

/// Two threads doing UPSERTs with different ON MATCH SET values.
/// Verifies the final value is set by the last successful writer (serializable).
#[test]
fn concurrent_upsert_last_writer_wins() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = Arc::new(StorageEngine::open(&config).expect("open"));
    let allocator = Arc::new(NodeIdAllocator::resume_from(NodeId::from_raw(2000)));

    // Create initial node: Bob, age=0
    let mut setup_interner = FieldInterner::new();
    {
        let plan = create_user_plan("Bob", 0);
        let mut ctx = super::helpers::make_ctx_legacy(&engine, &mut setup_interner, &allocator);
        execute(&plan, &mut ctx).expect("create Bob");
    }

    // Thread A: sets age=100, Thread B: sets age=200
    // Both run 100 iterations. After all complete, age must be 100 or 200.
    let interner_bytes = Arc::new(setup_interner.to_bytes().expect("serialize interner"));

    let handles: Vec<_> = [100i64, 200i64]
        .iter()
        .map(|&target_age| {
            let engine = Arc::clone(&engine);
            let allocator = Arc::clone(&allocator);
            let interner_bytes = Arc::clone(&interner_bytes);

            thread::spawn(move || {
                let mut interner =
                    FieldInterner::from_bytes(&interner_bytes).expect("deserialize interner");
                let mut successes = 0u64;
                let mut conflicts = 0u64;

                for _ in 0..100 {
                    let plan = upsert_plan("Bob", target_age);
                    let mut ctx =
                        super::helpers::make_ctx_legacy(&engine, &mut interner, &allocator);
                    match execute(&plan, &mut ctx) {
                        Ok(_) => successes += 1,
                        Err(ExecutionError::Conflict(_)) => conflicts += 1,
                        Err(e) => panic!("unexpected error: {e}"),
                    }
                }
                (successes, conflicts)
            })
        })
        .collect();

    let mut total_successes = 0u64;
    let mut total_conflicts = 0u64;
    for handle in handles {
        let (s, c) = handle.join().expect("thread ok");
        total_successes += s;
        total_conflicts += c;
    }

    assert_eq!(total_successes + total_conflicts, 200);
    assert!(total_successes > 0);

    // Final value must be exactly 100 or 200 (no partial/corrupted state).
    let final_age =
        read_user_age(&engine, &mut setup_interner, &allocator, "Bob").expect("Bob should exist");
    assert!(
        final_age == 100 || final_age == 200,
        "final age ({final_age}) must be 100 or 200 — no corruption"
    );

    eprintln!(
        "Last-writer-wins: {total_successes} success, {total_conflicts} conflicts. Final age: {final_age}"
    );
}
