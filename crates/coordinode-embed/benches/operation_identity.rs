//! The cost of client operation identity on the durable write path: a bound
//! autocommit write against the same write unbound, an outcome lookup, an
//! acknowledgement and a retirement, and the bytes the records retain before
//! and after retirement. The bound and unbound arms alternate so neither is
//! always measured after the other.
#![allow(clippy::expect_used, clippy::print_stdout)]

use std::time::{Duration, Instant};

use coordinode_embed::Database;
use coordinode_embed::db::StatementOptions;
use coordinode_embed::operation::{IssuerId, OperationDescriptor, Resolve};
use coordinode_storage::engine::partition::Partition;

/// Writes per arm.
const ROUNDS: usize = 2000;

fn report(label: &str, mut times: Vec<Duration>) {
    times.sort_unstable();
    let at = |q: usize| times[(times.len() - 1) * q / 1000];
    println!(
        "{label}: n={} p50={:?} p99={:?} p999={:?}",
        times.len(),
        at(500),
        at(990),
        at(999)
    );
}

fn write(db: &Database, operation: Option<OperationDescriptor>) -> Duration {
    let options = StatementOptions {
        operation,
        ..StatementOptions::default()
    };
    let start = Instant::now();
    db.execute_cypher_shared_with("CREATE (:Bench {k: 1})", None, None, &options)
        .expect("write");
    start.elapsed()
}

/// The bytes of every operation and incarnation record the store holds.
fn retained(db: &Database) -> (usize, usize) {
    let engine = db.engine_shared();
    let mut records = 0;
    let mut bytes = 0;
    for prefix in [&b"op:b:"[..], &b"op:i:"[..]] {
        for guard in engine.prefix_scan(Partition::Schema, prefix).expect("scan") {
            let (key, value) = guard.into_inner().expect("record");
            records += 1;
            bytes += key.len() + value.len();
        }
    }
    (records, bytes)
}

fn main() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");
    let issuer = db.enroll_issuer(IssuerId([0x42; 16])).expect("enroll");

    let mut unbound = Vec::with_capacity(ROUNDS);
    let mut bound = Vec::with_capacity(ROUNDS);
    let mut done = Vec::with_capacity(ROUNDS);
    for round in 0..ROUNDS {
        let order = if round % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        };
        for is_bound in order {
            if is_bound {
                let op = issuer.next().expect("number");
                bound.push(write(&db, Some(op)));
                done.push(op);
            } else {
                unbound.push(write(&db, None));
            }
        }
    }
    report("write unbound", unbound);
    report("write bound", bound);

    let mut lookups = Vec::with_capacity(done.len());
    for op in &done {
        let start = Instant::now();
        std::hint::black_box(db.resolve_operation(op, Resolve::Lookup).expect("lookup"));
        lookups.push(start.elapsed());
    }
    report("lookup", lookups);

    let (records, bytes) = retained(&db);
    println!(
        "retained before retirement: records={records} bytes={bytes} per_operation={}",
        bytes / done.len()
    );

    // Acknowledge every other outcome out of order, then retire the prefix.
    let mut acks = Vec::with_capacity(done.len() / 2);
    for op in done.iter().rev().step_by(2) {
        let start = Instant::now();
        db.acknowledge_operation(op).expect("acknowledge");
        acks.push(start.elapsed());
    }
    report("acknowledge", acks);
    let bound_at = issuer.next().expect("number");
    let start = Instant::now();
    db.retire_operations(&bound_at).expect("retire");
    println!("retire {} numbers: {:?}", done.len(), start.elapsed());
    let (records, bytes) = retained(&db);
    println!("retained after retirement: records={records} bytes={bytes}");
}
