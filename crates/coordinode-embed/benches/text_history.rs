//! What a full-text read costs at the present and at a past timestamp.
//!
//! A corpus of documents is loaded and the index is let catch up; the time
//! then is T0. The documents are rewritten in steps, and after each step the
//! same full-text query runs at the present and `AS OF TIMESTAMP T0`. A read
//! at the present answers from the index; a read at T0 answers from it for
//! the documents unchanged since and evaluates the rewritten ones from the
//! snapshot, so its cost grows with the writes after T0. Each line is a
//! distribution of single-query latencies, with the store's size on disk.
//!
//! Run: cargo bench -p coordinode-embed --bench text_history

#![allow(clippy::expect_used, clippy::print_stdout)]

use std::path::Path;
use std::time::{Duration, Instant};

use coordinode_embed::Database;

/// Documents in the corpus.
const DOCS: usize = 20_000;

/// Words per document.
const WORDS_PER_DOC: usize = 8;

/// Distinct words; the queried one is in about WORDS_PER_DOC / VOCABULARY of
/// the documents.
const VOCABULARY: usize = 200;

/// Queries per measured case.
const QUERIES: usize = 400;

/// Documents rewritten after T0, cumulative, one line pair per step.
const CHURN: [usize; 4] = [0, 200, 2_000, 10_000];

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let rank = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[rank]
}

fn line(label: &str, mut samples: Vec<Duration>, rows: usize) {
    samples.sort_unstable();
    let total: Duration = samples.iter().sum();
    println!(
        "{label:<34} rows={rows:>5}  mean={:>9.3?}  p50={:>9.3?}  p99={:>9.3?}  p999={:>9.3?}",
        total / samples.len() as u32,
        percentile(&samples, 0.50),
        percentile(&samples, 0.99),
        percentile(&samples, 0.999),
    );
}

/// A deterministic word sequence, so every run reads the same corpus.
struct Words(u64);

impl Words {
    fn next(&mut self) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % VOCABULARY as u64) as usize
    }

    fn body(&mut self) -> String {
        (0..WORDS_PER_DOC)
            .map(|_| format!("w{}", self.next()))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Wait until the text index worker has folded every applied commit, so a
/// present read measures the index rather than the unfolded writes.
fn await_folded(db: &Database) {
    let deadline = Instant::now() + Duration::from_secs(300);
    // The embedded database keeps its nodes on shard 1.
    while !db
        .text_index_registry()
        .coverage()
        .is_none_or(|coverage| coverage.delta(1).is_empty())
    {
        assert!(
            Instant::now() < deadline,
            "the text worker did not catch up"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// Bytes the store holds under `dir`.
fn size_on_disk(dir: &Path) -> u64 {
    let mut total = 0;
    let mut pending = vec![dir.to_path_buf()];
    while let Some(path) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&path) else {
            continue;
        };
        for entry in entries.flatten() {
            match entry.metadata() {
                Ok(meta) if meta.is_dir() => pending.push(entry.path()),
                Ok(meta) => total += meta.len(),
                Err(_) => {}
            }
        }
    }
    total
}

/// Run `query` `QUERIES` times; the latencies and the rows of the last run.
fn measure(db: &Database, query: &str) -> (Vec<Duration>, usize) {
    let mut rows = 0;
    let samples = (0..QUERIES)
        .map(|_| {
            let started = Instant::now();
            rows = db
                .execute_cypher_shared(query, None, None, None, None)
                .expect("full-text read")
                .rows
                .len();
            started.elapsed()
        })
        .collect();
    (samples, rows)
}

fn main() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    db.execute_cypher("CREATE TEXT INDEX doc_body ON :Doc(body)")
        .expect("create text index");
    let mut words = Words(0x9E37_79B9_7F4A_7C15);
    for chunk in 0..DOCS / 100 {
        let tx = db.begin_transaction();
        for i in 0..100 {
            let id = chunk * 100 + i;
            db.execute_in_transaction(
                tx,
                &format!("CREATE (:Doc {{id: {id}, body: '{}'}})", words.body()),
                None,
            )
            .expect("load");
        }
        db.commit_transaction(tx).expect("commit load");
    }
    await_folded(&db);
    let t0 = db.read_ts().as_raw();
    println!("== full-text read, {DOCS} docs x {WORDS_PER_DOC} words, vocabulary {VOCABULARY} ==");

    let query = "MATCH (n:Doc) WHERE text_match(n.body, 'w7') RETURN n.id";
    let mut rewritten = 0;
    for churn in CHURN {
        while rewritten < churn {
            let tx = db.begin_transaction();
            for _ in 0..100.min(churn - rewritten) {
                db.execute_in_transaction(
                    tx,
                    &format!(
                        "MATCH (n:Doc {{id: {}}}) SET n.body = '{}'",
                        rewritten % DOCS,
                        words.body()
                    ),
                    None,
                )
                .expect("rewrite");
                rewritten += 1;
            }
            db.commit_transaction(tx).expect("commit rewrite");
        }
        await_folded(&db);
        println!(
            "-- {churn} rewritten after T0, store {:.1} MiB",
            size_on_disk(dir.path()) as f64 / (1024.0 * 1024.0)
        );
        let (samples, rows) = measure(&db, query);
        line("  present", samples, rows);
        let (samples, rows) = measure(&db, &format!("{query} AS OF TIMESTAMP {t0}"));
        line("  as of T0", samples, rows);
    }
}
