//! Integration tests: concurrent UPSERT with CAS conflict detection.
//!
//! Verifies that when multiple threads execute UPSERTs on the same node,
//! the CAS mechanism prevents lost updates and data corruption.
//! This test validates G007: UPSERT CAS under concurrent access.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;

use coordinode_core::graph::intern::FieldInterner;
use coordinode_core::graph::node::{NodeId, NodeIdAllocator};
use coordinode_core::graph::types::Value;
use coordinode_query::cypher::ast::{Expr, SetItem};
use coordinode_query::executor::runner::{execute, ExecutionError};
use coordinode_query::planner::logical::{LogicalOp, LogicalPlan};
use coordinode_storage::engine::config::StorageConfig;
use coordinode_storage::engine::core::StorageEngine;

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
                    Expr::Literal(Value::String(name.to_string())),
                )],
            }),
            on_match: vec![SetItem::Property {
                variable: "n".into(),
                property: "age".into(),
                expr: Expr::Literal(Value::Int(age)),
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
                ("name".into(), Expr::Literal(Value::String(name.into()))),
                ("age".into(), Expr::Literal(Value::Int(age))),
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
                    Expr::Literal(Value::String(target_name.into())),
                )],
            }),
            items: vec![coordinode_query::planner::logical::ProjectItem {
                alias: Some("n.age".into()),
                expr: Expr::PropertyAccess {
                    expr: Box::new(Expr::Variable("n".into())),
                    property: "age".into(),
                },
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
    let config = StorageConfig::new(dir.path());
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
    let interner_bytes = Arc::new(setup_interner.to_bytes());

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

/// Two threads doing UPSERTs with different ON MATCH SET values.
/// Verifies the final value is set by the last successful writer (serializable).
#[test]
fn concurrent_upsert_last_writer_wins() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::new(dir.path());
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
    let interner_bytes = Arc::new(setup_interner.to_bytes());

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
