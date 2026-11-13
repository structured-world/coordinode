//! What opening a database that keeps a checkpoint costs: the open verifies
//! the trees the checkpoint backs, the schema tree among them.
//!
//! A database with `OPEN_CKPT_ROWS` schema rows of 512 bytes (default 20000,
//! about 10 MiB, so a second pass over the schema tree shows in the bytes
//! read) and one checkpoint is opened `OPEN_CKPT_REPS` times (default 10).
//! One JSON line reports the schema tree's size, each open's wall time, and
//! the bytes each open read through the filesystem (Linux
//! `/proc/self/io` `rchar`, page cache hits included, so a second pass
//! over a tree counts however warm the cache is).
//!
//! Run: `cargo bench -p coordinode-embed --bench open_checkpointed`.

#![allow(clippy::expect_used, clippy::print_stdout)]

use std::time::Instant;

use coordinode_embed::Database;
use coordinode_storage::engine::partition::Partition;
use lsm_tree::AbstractTree as _;

fn env(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// `rchar` of `/proc/self/io`: bytes this process read through the
/// filesystem; 0 where the file does not exist.
fn rchar() -> u64 {
    std::fs::read_to_string("/proc/self/io")
        .ok()
        .and_then(|text| {
            text.lines()
                .find_map(|line| line.strip_prefix("rchar:")?.trim().parse().ok())
        })
        .unwrap_or(0)
}

fn main() {
    let rows = env("OPEN_CKPT_ROWS", 20_000);
    let reps = env("OPEN_CKPT_REPS", 10);
    let dir = tempfile::tempdir().expect("tempdir");
    let schema_bytes = {
        let db = Database::open(dir.path()).expect("create");
        // Values that do not compress, so the tree is as large on disk as
        // the rows are.
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut value = vec![0_u8; 512];
        for i in 0..rows {
            for byte in &mut value {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *byte = state.to_le_bytes()[0];
            }
            db.engine()
                .put(
                    Partition::Schema,
                    format!("schema:bench:{i:08}").as_bytes(),
                    &value,
                )
                .expect("schema row");
        }
        db.persist().expect("persist");
        db.checkpoint(1).expect("checkpoint");
        db.engine()
            .tree(Partition::Schema)
            .expect("schema tree")
            .disk_space()
    };

    let mut open_ms = Vec::new();
    let mut read = Vec::new();
    for _ in 0..reps {
        let before = rchar();
        let at = Instant::now();
        let db = Database::open(dir.path()).expect("open");
        open_ms.push(at.elapsed().as_secs_f64() * 1000.0);
        read.push(rchar() - before);
        drop(db);
    }
    let mut sorted = open_ms.clone();
    sorted.sort_by(f64::total_cmp);
    let report = serde_json::json!({
        "schema_rows": rows,
        "schema_bytes": schema_bytes,
        "open_ms": open_ms,
        "open_ms_median": sorted[sorted.len() / 2],
        "read_bytes": read,
    });
    println!("{report}");
}
