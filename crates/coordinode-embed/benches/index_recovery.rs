//! What reopening costs after a process died while its index copies were in
//! use: every such copy answers from the records until a check verifies it.
//! A child process builds labels with indexes beside a label of large
//! unrelated records, uses every index and dies without destructors; this
//! process opens the store again and reports how long the open took, when
//! each index served again, when all did, and the latency of an indexed
//! lookup meanwhile against the same lookup once every index serves.
//!
//! Sizes: `INDEX_RECOVERY_NODES` nodes per indexed label (default 20000),
//! `INDEX_RECOVERY_LARGE` large records (default 20000) of
//! `INDEX_RECOVERY_PAYLOAD` bytes each (default 4096).
//!
//! Run: cargo bench -p coordinode-embed --bench index_recovery

#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    clippy::unwrap_used
)]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use coordinode_core::graph::types::Value;
use coordinode_embed::Database;
use coordinode_modality::{IndexStore as _, LocalIndexStore};

/// Set for the child; it builds the store at this directory, then dies.
const CHILD_DIR: &str = "INDEX_RECOVERY_CHILD_DIR";

/// Labels that carry indexes.
const LABELS: [&str; 3] = ["A", "B", "C"];

fn size(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn nodes() -> usize {
    size("INDEX_RECOVERY_NODES", 20_000)
}

/// The indexed lookup the foreground issues: its index's copy answers from
/// the records until it is checked.
fn lookup(db: &Database, i: usize) -> Duration {
    let n = i % nodes() + 1;
    let query = format!("MATCH (a:A {{code: 'a{n}'}}) RETURN a.n AS n");
    let started = Instant::now();
    let rows = db
        .execute_cypher_shared(&query, None, None, None, None)
        .expect("lookup")
        .rows;
    let took = started.elapsed();
    assert_eq!(rows.len(), 1, "{query}");
    took
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    let rank = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[rank]
}

fn report(name: &str, mut samples: Vec<Duration>) {
    if samples.is_empty() {
        println!("{name:<34} lookups=0");
        return;
    }
    samples.sort_unstable();
    println!(
        "{name:<34} lookups={:>6} p50={:>10.3?} p99={:>10.3?} p999={:>10.3?}",
        samples.len(),
        percentile(&samples, 0.5),
        percentile(&samples, 0.99),
        percentile(&samples, 0.999)
    );
}

/// The child: indexed labels, a label of large records, every index used,
/// then death without destructors, as under SIGKILL.
fn child(dir: &std::path::Path) {
    let mut db = Database::open(dir).expect("open in child");
    for label in LABELS {
        let l = label.to_lowercase();
        db.execute_cypher(&format!("CREATE UNIQUE INDEX {l}_code ON :{label}(code)"))
            .expect("unique index");
        db.execute_cypher(&format!("CREATE INDEX {l}_group ON :{label}(grp)"))
            .expect("index");
        db.execute_cypher(&format!("CREATE INDEX {l}_n ON :{label}(n)"))
            .expect("index");
    }
    let n = nodes();
    const CHUNK: usize = 5_000;
    for label in LABELS {
        let l = label.to_lowercase();
        for base in (0..n).step_by(CHUNK) {
            let last = (base + CHUNK).min(n);
            db.execute_cypher(&format!(
                "UNWIND range({}, {last}) AS i CREATE (:{label} {{code: '{l}' + toString(i), \
                 grp: i % 97, n: i}})",
                base + 1
            ))
            .expect("indexed nodes");
        }
    }
    let large = size("INDEX_RECOVERY_LARGE", 20_000);
    let width = size("INDEX_RECOVERY_PAYLOAD", 4096);
    // Text that does not compress, so a large record stays its size on disk.
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut payload = || {
        (0..width)
            .map(|_| {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                let n = (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 58) as usize;
                char::from(b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_"[n])
            })
            .collect::<String>()
    };
    const LARGE_CHUNK: usize = 500;
    for base in (0..large).step_by(LARGE_CHUNK) {
        let count = LARGE_CHUNK.min(large - base);
        let bodies = (0..count).map(|_| Value::String(payload())).collect();
        let params = std::collections::HashMap::from([
            ("base".to_owned(), Value::Int(base as i64)),
            ("bodies".to_owned(), Value::Array(bodies)),
        ]);
        db.execute_cypher_with_params(
            "UNWIND range(0, size($bodies) - 1) AS i CREATE (:Large {k: $base + i, body: $bodies[i]})",
            params,
        )
        .expect("large records");
    }
    // Every copy on disk and in use, as copies that have served a while are.
    db.engine().persist().expect("the copies on disk");
    for label in LABELS {
        let l = label.to_lowercase();
        for query in [
            format!("MATCH (x:{label} {{code: '{l}1'}}) RETURN x.n"),
            format!("MATCH (x:{label} {{grp: 1}}) RETURN count(x)"),
            format!("MATCH (x:{label} {{n: 1}}) RETURN count(x)"),
        ] {
            db.execute_cypher(&query).expect("use the index");
        }
    }
    std::process::abort();
}

fn main() {
    if let Some(dir) = std::env::var_os(CHILD_DIR) {
        child(std::path::Path::new(&dir));
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let built = Instant::now();
    let status = std::process::Command::new(std::env::current_exe().expect("bench binary"))
        .env(CHILD_DIR, dir.path())
        .status()
        .expect("run the child");
    assert!(!status.success(), "the child must die by abort: {status:?}");
    println!(
        "store built by the child in {:?} (nodes per label {}, large records {})",
        built.elapsed(),
        nodes(),
        size("INDEX_RECOVERY_LARGE", 20_000)
    );

    let opening = Instant::now();
    let db = Database::open(dir.path()).expect("reopen");
    let opened = opening.elapsed();
    let names: BTreeMap<u64, String> = LocalIndexStore::new(db.engine())
        .list_definitions()
        .expect("definitions")
        .into_iter()
        .map(|def| (def.generation.as_raw(), def.to_string()))
        .collect();
    let unfit = || -> Vec<u64> {
        LocalIndexStore::new(db.engine())
            .list_unfit_here()
            .expect("marks")
            .into_iter()
            .map(|copy| copy.generation.as_raw())
            .collect()
    };
    let first = unfit();
    println!(
        "open took {opened:?}; copies to check: {} of {} indexes",
        first.len(),
        names.len()
    );
    assert!(
        !first.is_empty(),
        "the child's copies must be unfit after its death"
    );

    // Lookups while the checks run, and when each copy serves again.
    let checking = AtomicBool::new(true);
    let (during, served) = std::thread::scope(|s| {
        let foreground = s.spawn(|| {
            let mut samples = Vec::new();
            let mut i = 0;
            while checking.load(Ordering::Relaxed) {
                samples.push(lookup(&db, i));
                i += 1;
            }
            samples
        });
        let mut served = BTreeMap::new();
        let deadline = Instant::now() + Duration::from_secs(1800);
        loop {
            let left = unfit();
            for generation in &first {
                if !left.contains(generation) && !served.contains_key(generation) {
                    served.insert(*generation, opening.elapsed());
                }
            }
            if left.is_empty() {
                break;
            }
            assert!(Instant::now() < deadline, "copies still unfit: {left:?}");
            std::thread::sleep(Duration::from_millis(10));
        }
        checking.store(false, Ordering::Relaxed);
        (foreground.join().expect("foreground"), served)
    });
    for (generation, at) in &served {
        println!(
            "  {:<12} serves again {:?} after the open began",
            names.get(generation).map_or("?", String::as_str),
            at
        );
    }
    let all = served.values().max().copied().unwrap_or_default();
    let first_served = served.values().min().copied().unwrap_or_default();
    println!("first copy served after {first_served:?}; every copy after {all:?}");
    report("lookup while copies are checked", during);
    let after: Vec<Duration> = (0..2_000).map(|i| lookup(&db, i)).collect();
    report("lookup once every copy serves", after);
}
