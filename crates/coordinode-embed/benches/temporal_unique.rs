//! What admitting a temporal unique value costs as its history grows.
//!
//! One node holds a unique code over valid time; its end is corrected again
//! and again, so the facts known about it grow while what holds now stays
//! one owner over one interval. A correction claims the code again over its
//! new interval, and a competing writer taking the code over an overlapping
//! time is refused at its commit. Both are measured in windows as the
//! history grows: admission reads the owners of the code as they stand, so
//! neither may grow with the superseded facts behind them. Each case runs
//! under RESOLVED and DERIVED maintenance.
//!
//! Run: cargo bench -p coordinode-embed --bench temporal_unique

#![allow(clippy::expect_used, clippy::print_stdout)]

use std::time::{Duration, Instant};

use coordinode_embed::{Database, DatabaseError};

/// Corrections between two measured windows.
const STEPS: [usize; 4] = [0, 1_000, 10_000, 30_000];

/// Statements per measured window.
const WINDOW: usize = 500;

/// A valid time far past any clock this runs at.
const FUTURE: i64 = 4_000_000_000_000_000;

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let rank = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[rank]
}

fn line(label: &str, mut samples: Vec<Duration>) {
    samples.sort_unstable();
    let total: Duration = samples.iter().sum();
    println!(
        "{label:<44} n={:>5}  mean={:>9.3?}  p50={:>9.3?}  p99={:>9.3?}  p999={:>9.3?}",
        samples.len(),
        total / samples.len().max(1) as u32,
        percentile(&samples, 0.50),
        percentile(&samples, 0.99),
        percentile(&samples, 0.999),
    );
}

/// Move the holder's end to the `i`-th of two alternating instants.
fn correct(db: &mut Database, i: usize) -> Duration {
    let end = FUTURE + (i % 2) as i64;
    let t = Instant::now();
    db.execute_cypher(&format!(
        "MATCH (n:Emp) WHERE n.name = 'a' SET n.valid_to = {end}"
    ))
    .expect("correction");
    t.elapsed()
}

/// A competing writer taking the code over the holder's time.
fn refused(db: &mut Database) -> Duration {
    let t = Instant::now();
    let outcome = db.execute_cypher("CREATE (:Emp {name: 'b', code: 'k', valid_from: 1})");
    assert!(
        matches!(outcome, Err(DatabaseError::UniqueViolation { .. })),
        "the overlap must be refused: {outcome:?}"
    );
    t.elapsed()
}

fn run(maintenance: &str) {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    for statement in [
        "CREATE NODE TYPE Emp TEMPORAL WITH \
         (name: STRING, code: STRING, valid_from: INT, valid_to: INT)"
            .to_string(),
        format!(
            "CREATE UNIQUE INDEX emp_code ON :Emp(code) OPTIONS {{maintenance: '{maintenance}'}}"
        ),
        format!("CREATE (:Emp {{name: 'a', code: 'k', valid_from: 0, valid_to: {FUTURE}}})"),
    ] {
        db.execute_cypher(&statement).expect("setup");
    }
    let mut done = 0usize;
    for target in STEPS {
        while done < target {
            correct(&mut db, done);
            done += 1;
        }
        let corrections: Vec<Duration> = (0..WINDOW)
            .map(|_| {
                let d = correct(&mut db, done);
                done += 1;
                d
            })
            .collect();
        line(
            &format!("{maintenance}, history {target:>6}: correction"),
            corrections,
        );
        let refusals: Vec<Duration> = (0..WINDOW).map(|_| refused(&mut db)).collect();
        line(
            &format!("{maintenance}, history {target:>6}: refused overlap"),
            refusals,
        );
    }
}

fn main() {
    println!("== temporal unique admission against the length of history ==");
    for maintenance in ["resolved", "derived"] {
        run(maintenance);
    }
}
