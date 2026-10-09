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
//!
//! `TEXT_HISTORY_CHURN` (comma-separated counts, e.g. `10000`) measures only
//! those steps, the rewrites still applied in between, to profile one case.
//!
//! Each step also reports the process's resident memory and open file
//! descriptors, the bytes it passed to write calls against the text bytes the
//! statements wrote (write amplification), and the text indexes' size; the
//! run ends with a reopen, timed until the index answers a read. Memory, file
//! descriptors and written bytes come from `/proc` and read `n/a` elsewhere.

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

/// The value of `key` (kB, in bytes) in `/proc/self/status`.
fn proc_status(key: &str) -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with(key))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    kb.checked_mul(1024)
}

/// Open file descriptors of this process.
fn open_fds() -> Option<usize> {
    Some(std::fs::read_dir("/proc/self/fd").ok()?.count())
}

/// Bytes this process has passed to write system calls so far. Counted at
/// the call rather than at the block device, so a store on a memory file
/// system (a temporary directory often is) is measured too.
fn written_bytes() -> Option<u64> {
    let io = std::fs::read_to_string("/proc/self/io").ok()?;
    let line = io.lines().find(|l| l.starts_with("wchar:"))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

fn mib(bytes: u64) -> String {
    format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
}

fn or_na<T>(value: Option<T>, show: impl Fn(T) -> String) -> String {
    value.map_or_else(|| "n/a".to_string(), show)
}

/// Resources and write amplification at this point: `logical` is the text
/// bytes the statements have written since `written_from`.
fn resources(db: &Database, dir: &Path, logical: u64, written_from: Option<u64>) {
    // The counter only grows, so a later reading is never below an earlier.
    let physical = written_bytes()
        .zip(written_from)
        .and_then(|(now, from)| now.checked_sub(from));
    println!(
        "   rss {}  fds {}  store {}  text indexes {}  written {} for {} of text (x{})",
        or_na(proc_status("VmRSS:"), mib),
        or_na(open_fds(), |n| n.to_string()),
        mib(size_on_disk(dir)),
        mib(size_on_disk(db.text_index_registry().base_dir())),
        or_na(physical, mib),
        mib(logical),
        or_na(physical.filter(|_| logical > 0), |p| format!(
            "{:.1}",
            p as f64 / logical as f64
        )),
    );
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
    // T0 has to stay readable for the whole run, longer than the default
    // window keeps history.
    db.set_retention_window(Duration::from_secs(3600));
    db.execute_cypher("CREATE TEXT INDEX doc_body ON :Doc(body)")
        .expect("create text index");
    let written_from = written_bytes();
    let mut logical = 0u64;
    let mut words = Words(0x9E37_79B9_7F4A_7C15);
    for chunk in 0..DOCS / 100 {
        let tx = db.begin_transaction();
        for i in 0..100 {
            let id = chunk * 100 + i;
            let body = words.body();
            logical += body.len() as u64;
            db.execute_in_transaction(
                tx,
                &format!("CREATE (:Doc {{id: {id}, body: '{body}'}})"),
                None,
            )
            .expect("load");
        }
        db.commit_transaction(tx).expect("commit load");
    }
    await_folded(&db);
    let t0 = db.read_ts().as_raw();
    println!("== full-text read, {DOCS} docs x {WORDS_PER_DOC} words, vocabulary {VOCABULARY} ==");
    println!("-- loaded");
    resources(&db, dir.path(), logical, written_from);

    let measured: Vec<usize> = std::env::var("TEXT_HISTORY_CHURN").map_or_else(
        |_| CHURN.to_vec(),
        |list| {
            list.split(',')
                .map(|n| n.trim().parse().expect("TEXT_HISTORY_CHURN: a count"))
                .collect()
        },
    );
    let query = "MATCH (n:Doc) WHERE text_match(n.body, 'w7') RETURN n.id";
    let mut rewritten = 0;
    for churn in measured {
        while rewritten < churn {
            let tx = db.begin_transaction();
            for _ in 0..100.min(churn - rewritten) {
                let body = words.body();
                logical += body.len() as u64;
                db.execute_in_transaction(
                    tx,
                    &format!(
                        "MATCH (n:Doc {{id: {}}}) SET n.body = '{body}'",
                        rewritten % DOCS,
                    ),
                    None,
                )
                .expect("rewrite");
                rewritten += 1;
            }
            db.commit_transaction(tx).expect("commit rewrite");
        }
        await_folded(&db);
        println!("-- {churn} rewritten after T0");
        resources(&db, dir.path(), logical, written_from);
        let (samples, rows) = measure(&db, query);
        line("  present", samples, rows);
        let (samples, rows) = measure(&db, &format!("{query} AS OF TIMESTAMP {t0}"));
        line("  as of T0", samples, rows);
    }

    // Recovery: the index is rebuilt from the store on open; ready is the
    // first read answered from it.
    drop(db);
    let started = Instant::now();
    let db = Database::open(dir.path()).expect("reopen");
    let opened = started.elapsed();
    await_folded(&db);
    let rows = db
        .execute_cypher_shared(query, None, None, None, None)
        .expect("first read after reopen")
        .rows
        .len();
    println!(
        "-- reopen: open {opened:.3?}, ready to read {:.3?} (rows={rows})",
        started.elapsed()
    );
    resources(&db, dir.path(), logical, written_from);
}
