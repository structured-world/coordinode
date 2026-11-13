//! Kill fuzzing: a writer process killed from outside at random moments.
//!
//! The parent starts a child that writes acknowledged nodes on one thread
//! while another runs maintenance at random: flush, checkpoint, a full
//! rebuild of the node partition from the latest checkpoint, a major
//! compaction. The parent kills it with SIGKILL after a random delay (inside
//! the open, inside a rebuild, inside a compaction, wherever it lands), opens
//! the directory itself and checks it: the open succeeds, no rebuild is left
//! pending, every acknowledged node is there and found through its index, and
//! a scrub finds nothing to repair. The next child continues on the same
//! directory, so the cuts accumulate.
//!
//! It runs only when `COORDINODE_KILL_FUZZ_SECS` names a time budget, outside
//! the ordinary suite. `COORDINODE_KILL_FUZZ_SEED` replays a reported run.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeSet;
use std::io::BufRead as _;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use coordinode_core::graph::types::Value;
use coordinode_embed::Database;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

/// The parent's time budget in seconds; without it the parent returns.
const BUDGET_SECS: &str = "COORDINODE_KILL_FUZZ_SECS";
/// A seed to replay; drawn from the clock otherwise.
const SEED: &str = "COORDINODE_KILL_FUZZ_SEED";
/// Set by the parent for the child: the directory it writes.
const CHILD_DIR: &str = "COORDINODE_KILL_FUZZ_CHILD_DIR";
/// Set by the parent for the child: its first node id and its own seed.
const CHILD_FIRST_ID: &str = "COORDINODE_KILL_FUZZ_CHILD_FIRST_ID";
const CHILD_SEED: &str = "COORDINODE_KILL_FUZZ_CHILD_SEED";
/// Ids each child may use; children never share one.
const IDS_PER_CHILD: i64 = 1_000_000;
/// The longest a child runs before its kill.
const MAX_KILL_DELAY_MS: u64 = 1_500;

/// xorshift64: a seeded sequence for the kill delays and maintenance steps.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok().map(|v| v.parse().expect(name))
}

/// Child half: write and maintain until killed. Run on its own it has no
/// directory and returns.
#[test]
fn kill_fuzz_child() {
    let Some(dir) = std::env::var_os(CHILD_DIR) else {
        return;
    };
    let first_id = env_u64(CHILD_FIRST_ID).expect("first id") as i64;
    let mut rng = Rng::new(env_u64(CHILD_SEED).expect("child seed"));
    let db = Arc::new(Database::open(Path::new(&dir)).expect("open in the child"));
    // A rebuild replays the journal it read; writes landing meanwhile would
    // be uncovered entries it refuses for, so writes pause around it.
    let gate = Arc::new(parking_lot::RwLock::new(()));

    let writer = {
        let db = Arc::clone(&db);
        let gate = Arc::clone(&gate);
        std::thread::spawn(move || {
            for id in first_id..first_id + IDS_PER_CHILD {
                let _held = gate.read();
                let x = (id % 97) as f64 / 97.0;
                db.execute_cypher_shared(
                    &format!(
                        "CREATE (:K {{id: {id}, vec: [{x}, 1.0, 0.5], body: 'word{}'}})",
                        id % 13
                    ),
                    None,
                    None,
                    None,
                    None,
                )
                .expect("write in the child");
                println!("ack {id}");
            }
        })
    };

    let mut checkpoint = None;
    loop {
        std::thread::sleep(Duration::from_millis(rng.below(50)));
        match rng.below(4) {
            0 => db.persist().expect("persist"),
            1 => checkpoint = Some(db.checkpoint(2).expect("checkpoint")),
            2 => {
                if let Some(ckpt) = &checkpoint {
                    let _paused = gate.write();
                    rebuild_nodes(db.engine(), ckpt);
                }
            }
            _ => db.engine().major_compact(Partition::Node).expect("compact"),
        }
        assert!(!writer.is_finished(), "the writer stopped");
    }
}

/// Rebuild the node partition from `ckpt` and the journal after it, as a
/// repair of that partition does.
fn rebuild_nodes(engine: &StorageEngine, ckpt: &Path) {
    let floor = engine.checkpoint_replay_floor(ckpt).expect("replay floor");
    let since = engine
        .oplog_read_since(floor)
        .expect("read the journal")
        .expect("a journal");
    engine
        .repair_partition_from_checkpoint(ckpt, &since, Partition::Node)
        .expect("rebuild");
}

/// Every file under `dir` with its length, for a failure report.
fn listing(dir: &Path) -> String {
    let mut out = String::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            match entry.metadata() {
                Ok(m) if m.is_dir() => stack.push(path),
                Ok(m) => out.push_str(&format!(
                    "{:>12} {}\n",
                    m.len(),
                    path.strip_prefix(dir).unwrap_or(&path).display()
                )),
                Err(e) => out.push_str(&format!("{:>12} {} ({e})\n", "?", path.display())),
            }
        }
    }
    out
}

/// Open `dir` after a kill and check it holds every acknowledged node.
fn check(dir: &Path, acked: &BTreeSet<i64>) -> Result<(), String> {
    let mut db = Database::open(dir).map_err(|e| format!("open: {e}"))?;
    let pending = db.engine().pending_rebuilds().map_err(|e| format!("{e}"))?;
    if !pending.is_empty() {
        return Err(format!("rebuilds left pending: {pending:?}"));
    }
    let rows = db
        .execute_cypher("MATCH (k:K) RETURN k.id AS id")
        .map_err(|e| format!("scan: {e}"))?;
    let found: BTreeSet<i64> = rows
        .iter()
        .filter_map(|r| match r.get("id") {
            Some(Value::Int(i)) => Some(*i),
            _ => None,
        })
        .collect();
    let lost: Vec<i64> = acked.difference(&found).copied().take(20).collect();
    if !lost.is_empty() {
        return Err(format!("acknowledged nodes lost (first 20): {lost:?}"));
    }
    // The newest and the oldest acknowledged node through the index.
    for id in [acked.first(), acked.last()].into_iter().flatten() {
        let rows = db
            .execute_cypher(&format!("MATCH (k:K {{id: {id}}}) RETURN k.id AS id"))
            .map_err(|e| format!("indexed lookup: {e}"))?;
        if rows.len() != 1 {
            return Err(format!("indexed lookup of {id} found {} nodes", rows.len()));
        }
    }
    let report = db.verify_and_repair().map_err(|e| format!("scrub: {e}"))?;
    if !report.is_clean() {
        return Err(format!("scrub not clean: {report:?}"));
    }
    Ok(())
}

/// A writer killed at random moments loses no acknowledged write, and its
/// directory opens clean every time.
#[test]
fn a_writer_killed_at_random_moments_loses_nothing() {
    let Some(budget) = env_u64(BUDGET_SECS) else {
        return;
    };
    let seed = env_u64(SEED).unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos() as u64
    });
    eprintln!("kill fuzz: seed {seed}, budget {budget}s");
    let mut rng = Rng::new(seed);
    let dir = tempfile::tempdir().expect("tempdir");
    {
        let mut db = Database::open(dir.path()).expect("first open");
        db.execute_cypher("CREATE INDEX k_id ON :K(id)")
            .expect("b-tree index");
        db.execute_cypher(r#"CREATE VECTOR INDEX k_vec ON :K(vec) OPTIONS {metric: "l2"}"#)
            .expect("vector index");
        db.execute_cypher("CREATE TEXT INDEX k_body ON :K(body)")
            .expect("text index");
    }

    let started = Instant::now();
    let mut acked = BTreeSet::new();
    let mut round = 0i64;
    while started.elapsed() < Duration::from_secs(budget) {
        round += 1;
        let delay = Duration::from_millis(rng.below(MAX_KILL_DELAY_MS));
        let mut child = std::process::Command::new(std::env::current_exe().expect("test binary"))
            .args([
                "--exact",
                "integration::kill_fuzz::kill_fuzz_child",
                "--nocapture",
            ])
            .env(CHILD_DIR, dir.path())
            .env(CHILD_FIRST_ID, (round * IDS_PER_CHILD).to_string())
            .env(CHILD_SEED, rng.next().to_string())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("start the child");
        let stdout = child.stdout.take().expect("child stdout");
        let reader = std::thread::spawn(move || {
            std::io::BufReader::new(stdout)
                .lines()
                .map_while(Result::ok)
                .filter_map(|l| {
                    l.strip_prefix("ack ")
                        .map(|id| id.parse::<i64>().expect("ack id"))
                })
                .collect::<Vec<_>>()
        });
        std::thread::sleep(delay);
        let early = child.try_wait().expect("poll the child");
        child.kill().expect("kill the child");
        let status = child.wait().expect("reap the child");
        acked.extend(reader.join().expect("read the acks"));
        let mut stderr = String::new();
        if let Some(mut e) = child.stderr.take() {
            std::io::Read::read_to_string(&mut e, &mut stderr).expect("child stderr");
        }
        let report = |what: String| -> ! {
            panic!(
                "kill fuzz seed {seed}, round {round}, kill after {delay:?}: {what}\n\
                 child: {status:?}\n--- child stderr ---\n{stderr}\n--- files ---\n{}",
                listing(dir.path())
            )
        };
        if let Some(exited) = early {
            report(format!("the child ended before its kill: {exited:?}"));
        }
        if let Err(e) = check(dir.path(), &acked) {
            report(e);
        }
    }
    eprintln!(
        "kill fuzz: {round} kills, {} acknowledged nodes, all kept",
        acked.len()
    );
}
