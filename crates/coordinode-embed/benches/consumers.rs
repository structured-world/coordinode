//! Change-stream consumer retention in the embedded database: what the
//! registrations cost the write path (none, one or sixteen registered, arms
//! alternating), how long an acknowledgement, a registration and a
//! cancellation take, what listing and summing up a hundred registrations
//! costs, the journal a stalled STRICT consumer keeps through reclamation,
//! and how long an open with a hundred registrations takes.
#![allow(clippy::expect_used, clippy::print_stdout)]

use std::time::{Duration, Instant};

use coordinode_embed::Database;
use coordinode_embed::consumer::{
    ConsumerKind, ConsumerRegistration, ConsumerRetentionPolicy, InitialSeqno, RetentionSpace,
    TopologyScope,
};

/// Writes per arm of the write-path comparison and per acknowledgement run.
const ROUNDS: usize = 2000;
/// Registrations for the listing, summary and open measurements.
const MANY: usize = 100;

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

fn consumer(id: &str) -> ConsumerRegistration {
    ConsumerRegistration {
        consumer_id: id.to_string(),
        kind: ConsumerKind::OplogEvents,
        scope: TopologyScope::Shard(0),
        initial_seqno: InitialSeqno::FromNow,
        retention: ConsumerRetentionPolicy::Strict,
    }
}

fn write(db: &Database, label: &str, k: usize) -> Duration {
    let start = Instant::now();
    db.execute_cypher_shared(
        &format!("CREATE (:{label} {{k: {k}}})"),
        None,
        None,
        None,
        None,
    )
    .expect("write");
    start.elapsed()
}

/// Write latency on stores with no, one and sixteen registrations, the arms
/// alternating so none is always measured after the others.
fn write_path() {
    let dirs: Vec<_> = (0..3)
        .map(|_| tempfile::tempdir().expect("tempdir"))
        .collect();
    let dbs: Vec<Database> = dirs
        .iter()
        .map(|d| Database::open(d.path()).expect("open"))
        .collect();
    for (db, n) in dbs.iter().zip([0usize, 1, 16]) {
        for i in 0..n {
            db.register_consumer(consumer(&format!("c{i}")))
                .expect("register");
        }
    }
    let mut times = [
        Vec::with_capacity(ROUNDS),
        Vec::with_capacity(ROUNDS),
        Vec::with_capacity(ROUNDS),
    ];
    for round in 0..ROUNDS {
        let mut arms = [0usize, 1, 2];
        arms.rotate_left(round % 3);
        for arm in arms {
            times[arm].push(write(&dbs[arm], "W", round));
        }
    }
    let [none, one, sixteen] = times;
    report("write, no consumer", none);
    report("write, 1 consumer", one);
    report("write, 16 consumers", sixteen);
}

/// An acknowledgement after each write, and registration and cancellation.
fn transitions() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");
    let h = db.register_consumer(consumer("acker")).expect("register");
    let mut acks = Vec::with_capacity(ROUNDS);
    for k in 0..ROUNDS {
        write(&db, "A", k);
        let head = db
            .engine()
            .journal_next_index()
            .expect("head")
            .expect("journal");
        let start = Instant::now();
        db.acknowledge_consumer(&h, head).expect("acknowledge");
        acks.push(start.elapsed());
    }
    report("acknowledge", acks);

    let mut registers = Vec::with_capacity(MANY);
    let mut cancels = Vec::with_capacity(MANY);
    for i in 0..MANY {
        let start = Instant::now();
        let h = db
            .register_consumer(consumer(&format!("r{i}")))
            .expect("register");
        registers.push(start.elapsed());
        let start = Instant::now();
        db.cancel_consumer(h).expect("cancel");
        cancels.push(start.elapsed());
    }
    report("register", registers);
    report("cancel", cancels);
}

/// Listing and summing up a hundred live registrations, and the open of a
/// store holding them.
fn many_registrations() {
    let dir = tempfile::tempdir().expect("tempdir");
    {
        let db = Database::open(dir.path()).expect("open");
        for i in 0..MANY {
            db.register_consumer(consumer(&format!("m{i}")))
                .expect("register");
        }
        let mut lists = Vec::with_capacity(200);
        let mut summaries = Vec::with_capacity(200);
        for _ in 0..200 {
            let start = Instant::now();
            db.consumers_list().expect("list");
            lists.push(start.elapsed());
            let start = Instant::now();
            db.consumer_retention().expect("summary");
            summaries.push(start.elapsed());
        }
        report(&format!("list {MANY} consumers"), lists);
        report(&format!("summary over {MANY} consumers"), summaries);
    }
    let mut opens = Vec::with_capacity(10);
    for _ in 0..10 {
        let start = Instant::now();
        let db = Database::open(dir.path()).expect("reopen");
        opens.push(start.elapsed());
        drop(db);
    }
    report(&format!("open with {MANY} registrations"), opens);
}

/// The journal a STRICT consumer that stopped acknowledging keeps through
/// reclamation, against the same writes with none registered.
fn retained_bytes() {
    let held_dir = tempfile::tempdir().expect("tempdir");
    let free_dir = tempfile::tempdir().expect("tempdir");
    let held = Database::open(held_dir.path()).expect("open");
    let free = Database::open(free_dir.path()).expect("open");
    held.register_consumer(consumer("stalled"))
        .expect("register");
    for k in 0..ROUNDS {
        write(&held, "R", k);
        write(&free, "R", k);
    }
    for db in [&held, &free] {
        db.persist().expect("persist");
        db.engine().reclaim_footprint();
    }
    let kept = |db: &Database| {
        db.engine()
            .oplog_read_since(0)
            .expect("read")
            .expect("journal")
            .len()
    };
    let bytes = held
        .consumer_retention()
        .expect("summary")
        .into_iter()
        .find(|v| v.space == RetentionSpace::Log)
        .and_then(|v| v.retained_bytes)
        .unwrap_or(0);
    println!(
        "retained after {ROUNDS} writes: stalled consumer keeps {} entries ({bytes} bytes), \
         none registered keeps {}",
        kept(&held),
        kept(&free)
    );
}

fn main() {
    // `CONSUMER_PARTS` picks parts by name (`write`, `transitions`, `many`,
    // `retained`); all run when it is unset.
    let parts = std::env::var("CONSUMER_PARTS").unwrap_or_default();
    let runs = |part: &str| parts.is_empty() || parts.split(',').any(|p| p == part);
    if runs("write") {
        write_path();
    }
    if runs("transitions") {
        transitions();
    }
    if runs("many") {
        many_registrations();
    }
    if runs("retained") {
        retained_bytes();
    }
}
