//! What a lookup costs when its index answers, when the records answer for
//! an index found wrong, and what memory each holds at its peak: the
//! statement budget's healthy path, its fallback, and the overhead of
//! counting.
//!
//! Each path reports p50 and p99 over `READS` lookups, and the smallest
//! statement memory limit it completes within (its peak charged memory,
//! found by bisection).
//!
//! Run: cargo bench -p coordinode-embed --bench repair_budget

#![allow(clippy::expect_used, clippy::print_stdout, clippy::unwrap_used)]

use std::time::{Duration, Instant};

use coordinode_core::graph::types::Value;
use coordinode_core::index::encoding::{encode_tuple, encode_unique_entry_key};
use coordinode_embed::Database;
use coordinode_embed::db::StatementOptions;
use coordinode_modality::{IndexStore as _, LocalIndexStore};
use coordinode_storage::engine::partition::Partition;

/// Nodes of the indexed label.
const USERS: usize = 20_000;

/// Nodes of another label sharing the shard.
const OTHERS: usize = 20_000;

/// Lookups per measured path.
const READS: usize = 300;

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    let rank = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[rank]
}

/// p50 and p99 of `READS` runs of `query`, each expected to return one row.
fn latency(db: &Database, query: &str) -> (Duration, Duration) {
    let mut samples: Vec<Duration> = (0..READS)
        .map(|_| {
            let started = Instant::now();
            let rows = db
                .execute_cypher_shared(query, None, None, None, None)
                .expect("read")
                .rows;
            assert_eq!(rows.len(), 1, "{query}");
            started.elapsed()
        })
        .collect();
    samples.sort_unstable();
    (percentile(&samples, 0.5), percentile(&samples, 0.99))
}

/// The smallest memory limit, in bytes, `query` completes within.
fn peak(db: &Database, query: &str) -> u64 {
    let runs = |limit: u64| {
        let options = StatementOptions {
            query_memory_limit: Some(limit),
            ..StatementOptions::default()
        };
        db.execute_cypher_shared_with(query, None, None, &options)
            .is_ok()
    };
    let (mut low, mut high) = (1u64, 256u64 << 20);
    assert!(runs(high), "{query} does not fit the default limit");
    while low < high {
        let mid = low + (high - low) / 2;
        if runs(mid) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    low
}

fn report(name: &str, db: &Database, query: &str) {
    let (p50, p99) = latency(db, query);
    let bytes = peak(db, query);
    println!(
        "{name:<28} p50={p50:>10.3?} p99={p99:>10.3?} peak={:>9.1} KiB",
        bytes as f64 / 1024.0
    );
}

fn main() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    db.execute_cypher("CREATE UNIQUE INDEX u_email ON :U(email)")
        .expect("index");
    db.execute_cypher(&format!(
        "UNWIND range(1, {USERS}) AS i CREATE (:U {{email: 'u' + toString(i) + '@x', n: i}})"
    ))
    .expect("users");
    db.execute_cypher(&format!(
        "UNWIND range(1, {OTHERS}) AS i CREATE (:O {{k: i, pad: 'abcdefghijklmnopqrstuvwxyz'}})"
    ))
    .expect("others");

    let lookup = "MATCH (u:U {email: 'u7@x'}) RETURN u.n AS n";
    report("index answers", &db, lookup);
    report(
        "label scan, no index",
        &db,
        "MATCH (u:U) WHERE u.email + '' = 'u7@x' RETURN u.n AS n",
    );

    // The entry for 'u7@x' now names another node: the lookup finds the
    // index wrong and answers from the records.
    let store = LocalIndexStore::new(db.engine());
    let id = store.resolve_name("u_email").unwrap().unwrap();
    let index = store.load_definition(id).unwrap().unwrap();
    let wrong = match db
        .execute_cypher("MATCH (u:U {email: 'u8@x'}) RETURN id(u) AS id")
        .expect("id")[0]
        .get("id")
    {
        Some(Value::Int(id)) => *id as u64,
        other => panic!("id: {other:?}"),
    };
    let tuple = encode_tuple(&[Value::String("u7@x".into())]).expect("tuple");
    db.engine()
        .put(
            Partition::Idx,
            &encode_unique_entry_key(index.generation, &tuple),
            &wrong.to_be_bytes(),
        )
        .expect("misattribute");
    report("records answer (fallback)", &db, lookup);
    report(
        "fallback under aggregate",
        &db,
        "MATCH (u:U {email: 'u7@x'}) RETURN count(u) AS n",
    );
}
