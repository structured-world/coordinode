//! A total owned-footprint ceiling: the bytes every file of the database
//! takes on disk (journal, tables, indexes, metadata) never pass the
//! endpoint's `hard_limit_bytes`, measured as the file system allocates them,
//! after every acknowledged write, after close and after reopen.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::Path;

use coordinode_embed::{Database, DatabaseError};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};

const LIMIT: u64 = 8 * 1024 * 1024;

/// Bytes the file system allocated to every file under `root`, and their
/// lengths: `(allocated, logical)`, each file once however many names it
/// has (a checkpoint links the tables it holds).
fn owned_bytes(root: &Path) -> (u64, u64) {
    let mut allocated = 0u64;
    let mut logical = 0u64;
    let mut seen = std::collections::HashSet::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let Ok(meta) = entry.metadata() else { continue };
            if meta.is_dir() {
                pending.push(entry.path());
            } else if meta.is_file() && seen.insert(file_id(&entry.path(), &meta)) {
                logical += meta.len();
                allocated += allocated_of(&meta);
            }
        }
    }
    (allocated, logical)
}

#[cfg(unix)]
fn file_id(_: &Path, meta: &std::fs::Metadata) -> (u64, u64) {
    use std::os::unix::fs::MetadataExt;
    (meta.dev(), meta.ino())
}

/// The file behind `path` (its volume and file index), however many names
/// it has. A file that cannot be opened counts under its path.
#[cfg(not(unix))]
fn file_id(path: &Path, _: &std::fs::Metadata) -> Result<same_file::Handle, std::path::PathBuf> {
    same_file::Handle::from_path(path).map_err(|_| path.to_path_buf())
}

#[cfg(unix)]
fn allocated_of(meta: &std::fs::Metadata) -> u64 {
    use std::os::unix::fs::MetadataExt;
    // st_blocks counts 512-byte units on every unix (POSIX stat).
    meta.blocks() * 512
}

#[cfg(not(unix))]
fn allocated_of(meta: &std::fs::Metadata) -> u64 {
    meta.len().div_ceil(4096) * 4096
}

fn create(key: u64) -> String {
    format!(
        "CREATE (:Footprint {{key:{key}, payload:'{}'}})",
        payload(key)
    )
}

/// Write rows from `from` until the cap refuses one; returns how many were
/// acknowledged. Every acknowledged write leaves the footprint inside the
/// cap.
fn fill(db: &mut Database, root: &Path, from: u64) -> u64 {
    for key in from..from + 4096 {
        let result = db.execute_cypher(&create(key));
        assert_within(root, &format!("write {key}"));
        match result {
            Ok(_) => {}
            Err(DatabaseError::CapacityExhausted { .. }) => return key - from,
            Err(other) => panic!("unexpected refusal: {other:?}"),
        }
    }
    panic!("the cap never refused a write");
}

/// What the database's footprint charges covers what its files take: no
/// file under the endpoint grew past the account.
fn assert_accounted(db: &Database, root: &Path, at: &str) {
    let footprints = db.engine().footprints();
    // A snapshot the background work (flushes, compactions) did not move: the
    // files read the same before and after the account. Growth is charged
    // before it is written, so the account then covers what the files take.
    let (mut charged, mut allocated) = (0, 0);
    for _ in 0..50 {
        let before = owned_bytes(root).0;
        charged = footprints.iter().map(|f| f.charged()).sum::<u64>();
        allocated = owned_bytes(root).0;
        if before == allocated {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    if charged >= allocated {
        return;
    }
    // Name the files that take more than their charge: the writes that
    // bypassed the account.
    let mut short = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).unwrap().flatten() {
            let meta = entry.metadata().unwrap();
            if meta.is_dir() {
                pending.push(entry.path());
            } else if meta.is_file() {
                let held = footprints.file(&entry.path()).map_or(0, |c| c.bytes());
                if allocated_of(&meta) > held {
                    short.push((entry.path(), meta.len(), allocated_of(&meta), held));
                }
            }
        }
    }
    panic!(
        "{at}: the footprint charges {charged} for files taking {allocated}; \
         (path, length, allocated, charged) short: {short:?}"
    );
}

fn count(db: &mut Database) -> i64 {
    let rows = db
        .execute_cypher("MATCH (n:Footprint) RETURN count(n) AS n")
        .unwrap();
    match rows[0].get("n") {
        Some(coordinode_core::graph::types::Value::Int(n)) => *n,
        other => panic!("count returned {other:?}"),
    }
}

fn capped(root: &Path) -> StorageConfig {
    StorageConfig::with_endpoints(vec![
        EndpointConfig::new("owned", root, Media::Hdd, Durability::Durable, Tier::Warm)
            .with_hard_limit_bytes(LIMIT),
    ])
}

/// 64 KiB of text no codec shrinks, distinct per `seed`.
fn payload(seed: u64) -> String {
    use std::fmt::Write;
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut out = String::with_capacity(64 * 1024);
    for _ in 0..8192 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        write!(&mut out, "{:08x}", state as u32).unwrap();
    }
    out
}

fn assert_within(root: &Path, at: &str) {
    let (allocated, logical) = owned_bytes(root);
    if allocated > LIMIT {
        let first = overallocated(root);
        std::thread::sleep(std::time::Duration::from_secs(2));
        let later = overallocated(root);
        panic!(
            "{at}: allocated={allocated} logical={logical} exceeds limit={LIMIT}; \
             files allocated past their length: {first:?}; two seconds later: {later:?}; \
             total later: {:?}",
            owned_bytes(root)
        );
    }
}

/// The files the file system allocated more to than their length rounded to
/// whole blocks, with `(length, allocated)`: what an account kept by length
/// cannot see.
fn overallocated(root: &Path) -> Vec<(std::path::PathBuf, u64, u64)> {
    let mut out = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let Ok(meta) = entry.metadata() else { continue };
            if meta.is_dir() {
                pending.push(entry.path());
            } else if meta.is_file() && allocated_of(&meta) > meta.len().div_ceil(4096) * 4096 {
                out.push((entry.path(), meta.len(), allocated_of(&meta)));
            }
        }
    }
    out
}

/// Writes one at a time until the cap refuses one: every acknowledged write
/// already fits with what it grows (journal, memtable flush, metadata), the
/// refusal is the typed capacity error, and closing and reopening the full
/// database stays inside the cap with every acknowledged write readable.
#[test]
fn sequential_writes_stop_before_the_footprint_passes_the_cap() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open_with_config(capped(dir.path())).unwrap();
    assert_within(dir.path(), "open");
    let mut accepted = 0u64;
    let mut refused = false;
    for key in 0..512u64 {
        let query = format!(
            "CREATE (:Footprint {{key:{key}, payload:'{}'}})",
            payload(key)
        );
        let mut result = db.execute_cypher(&query);
        assert_within(dir.path(), &format!("write {key}"));
        if matches!(result, Err(DatabaseError::CapacityExhausted { .. })) {
            // A compaction running in the background holds the room its
            // output may take until it ends: full is a refusal that stands
            // once the background work settled.
            std::thread::sleep(std::time::Duration::from_millis(500));
            result = db.execute_cypher(&query);
            assert_within(dir.path(), &format!("write {key} again"));
        }
        match result {
            Ok(_) => accepted += 1,
            Err(DatabaseError::CapacityExhausted { .. }) => {
                refused = true;
                break;
            }
            Err(other) => panic!("unexpected refusal: {other:?}"),
        }
    }
    assert!(refused, "the cap never refused a write");
    // Reclamation retires the journal its tables cover: the cap holds data,
    // not two copies of it, up to the recovery reserve, table overhead and
    // the room held for flushes that have not run yet.
    assert!(
        accepted * 64 * 1024 >= LIMIT / 2,
        "only {accepted} writes of 64 KiB fit under {LIMIT}"
    );
    // A full database still answers reads.
    let rows = db
        .execute_cypher("MATCH (n:Footprint) RETURN count(n) AS n")
        .unwrap();
    assert_eq!(
        rows[0].get("n"),
        Some(&coordinode_core::graph::types::Value::Int(accepted as i64)),
        "reads go on while writes are refused"
    );
    assert_accounted(&db, dir.path(), "full");
    drop(db);
    assert_within(dir.path(), "close");
    let mut db = Database::open_with_config(capped(dir.path())).unwrap();
    assert_within(dir.path(), "reopen");
    assert_accounted(&db, dir.path(), "reopen");
    let rows = db
        .execute_cypher("MATCH (n:Footprint) RETURN count(n) AS n")
        .unwrap();
    assert_eq!(
        rows[0].get("n"),
        Some(&coordinode_core::graph::types::Value::Int(accepted as i64)),
        "every acknowledged write survives"
    );
    drop(db);
    assert_within(dir.path(), "second close");
}

/// Eight writers race on one capped database: admission is a reservation,
/// so the writes accepted together still fit, whatever their interleaving.
#[test]
fn concurrent_writers_cannot_overcommit_the_cap() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open_with_config(capped(dir.path())).unwrap();
    let accepted = std::sync::atomic::AtomicU64::new(0);
    std::thread::scope(|scope| {
        for writer in 0..8u64 {
            let db = &db;
            let accepted = &accepted;
            scope.spawn(move || {
                for n in 0..64u64 {
                    let key = writer * 1000 + n;
                    let query = format!(
                        "CREATE (:Footprint {{key:{key}, payload:'{}'}})",
                        payload(key)
                    );
                    match db.execute_cypher_shared(&query, None, None, None, None) {
                        Ok(_) => {
                            accepted.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        Err(DatabaseError::CapacityExhausted { .. }) => return,
                        Err(other) => panic!("unexpected refusal: {other:?}"),
                    }
                }
            });
        }
    });
    assert_within(dir.path(), "after the race");
    let accepted = accepted.into_inner();
    assert!(accepted > 0, "no writer got in");
    drop(db);
    assert_within(dir.path(), "close");
    let mut db = Database::open_with_config(capped(dir.path())).unwrap();
    let rows = db
        .execute_cypher("MATCH (n:Footprint) RETURN count(n) AS n")
        .unwrap();
    assert_eq!(
        rows[0].get("n"),
        Some(&coordinode_core::graph::types::Value::Int(accepted as i64))
    );
}

/// Set by the parent of the kill test; the child does nothing without it.
const KILL_CHILD_DIR: &str = "COORDINODE_FOOTPRINT_KILL_CHILD_DIR";

/// Child half of `a_full_store_killed_reopens_inside_the_cap_with_its_writes`:
/// fill a capped database until refused, record how many writes were
/// acknowledged, then die without a destructor, as under SIGKILL.
#[test]
fn footprint_child_fills_then_aborts() {
    let Some(dir) = std::env::var_os(KILL_CHILD_DIR) else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let root = dir.join("db");
    let mut db = Database::open_with_config(capped(&root)).expect("open in child");
    let accepted = fill(&mut db, &root, 0);
    std::fs::write(dir.join("accepted"), accepted.to_string()).expect("record");
    std::process::abort();
}

/// A full capped database killed mid-flight (journal tail unsealed, memtables
/// unflushed) replays on reopen inside the cap and keeps every acknowledged
/// write; the replay is the store's own work and never passes the cap.
#[test]
fn a_full_store_killed_reopens_inside_the_cap_with_its_writes() {
    let dir = tempfile::tempdir().unwrap();
    let status = std::process::Command::new(std::env::current_exe().expect("test binary"))
        .args([
            "--exact",
            "integration::owned_footprint::footprint_child_fills_then_aborts",
            "--nocapture",
        ])
        .env(KILL_CHILD_DIR, dir.path())
        .status()
        .expect("run the child");
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        assert_eq!(
            status.signal(),
            Some(6),
            "the child must die by abort once full, got {status:?}"
        );
    }
    #[cfg(not(unix))]
    assert!(!status.success(), "the child must die, got {status:?}");
    let accepted: i64 = std::fs::read_to_string(dir.path().join("accepted"))
        .expect("the child recorded its writes")
        .parse()
        .unwrap();
    assert!(accepted > 0);
    let root = dir.path().join("db");
    assert_within(&root, "after the kill");

    let mut db = Database::open_with_config(capped(&root)).expect("reopen after the kill");
    assert_within(&root, "after replay");
    assert_accounted(&db, &root, "after replay");
    assert_eq!(
        count(&mut db),
        accepted,
        "every acknowledged write survives"
    );
    drop(db);
    assert_within(&root, "close after replay");
}

/// A checkpoint under the cap is charged before it is made (sealed journal
/// segments linked, charged once; tables cloned or linked as the file system
/// allows, a clone charged whole) and keeps the journal a rebuild from it
/// replays: filling the database after it leaves that journal in place, the
/// checkpoint whole, and the footprint inside the cap.
#[test]
fn a_checkpoint_under_the_cap_is_charged_and_keeps_its_history() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open_with_config(capped(dir.path())).unwrap();
    for key in 0..24 {
        db.execute_cypher(&create(key)).unwrap();
    }
    let checkpoint = db.checkpoint(1).expect("checkpoint under the cap");
    assert_within(dir.path(), "after the checkpoint");
    assert_accounted(&db, dir.path(), "after the checkpoint");

    fill(&mut db, dir.path(), 24);
    assert_accounted(&db, dir.path(), "full with a checkpoint");
    let floor =
        coordinode_storage::engine::core::StorageEngine::checkpoint_replay_floor(&checkpoint)
            .expect("the checkpoint's replay floor");
    let segments = dir.path().join("oplog").join("0");
    let first_kept = std::fs::read_dir(&segments)
        .unwrap()
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            name.strip_prefix("oplog-")?
                .strip_suffix(".bin")?
                .parse::<u64>()
                .ok()
        })
        .min()
        .expect("journal segments");
    assert!(
        first_kept <= floor,
        "reclamation kept the journal from the checkpoint's floor on \
         (floor {floor}, first kept segment starts at {first_kept})"
    );
    assert!(checkpoint.exists(), "nothing deletes the checkpoint to fit");
    assert_within(dir.path(), "full with a checkpoint");
}

/// Space-separated words for a text index, about 64 KiB, distinct per `seed`.
fn words(seed: u64) -> String {
    payload(seed)
        .as_bytes()
        .chunks(8)
        .map(|word| std::str::from_utf8(word).unwrap())
        .collect::<Vec<_>>()
        .join(" ")
}

/// A text index grows its segments inside the cap: indexed writes stop with
/// the typed refusal before the footprint passes the cap, and the index
/// still answers.
#[test]
fn a_text_index_grows_inside_the_cap() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open_with_config(capped(dir.path())).unwrap();
    db.execute_cypher("CREATE TEXT INDEX article_body ON :Article(body)")
        .unwrap();
    let mut accepted = 0u64;
    for key in 0..1024u64 {
        let body = if key == 0 {
            format!("needle {}", words(key))
        } else {
            words(key)
        };
        let result = db.execute_cypher(&format!("CREATE (:Article {{key:{key}, body:'{body}'}})"));
        assert_within(dir.path(), &format!("indexed write {key}"));
        match result {
            Ok(_) => accepted += 1,
            Err(DatabaseError::CapacityExhausted { .. }) => break,
            Err(other) => panic!("unexpected refusal: {other:?}"),
        }
    }
    assert!(
        accepted > 0 && accepted < 1024,
        "the cap refused an indexed write"
    );
    let rows = db
        .execute_cypher("MATCH (n:Article) WHERE text_match(n.body, 'needle') RETURN n.key AS k")
        .expect("the index answers on a full database");
    assert_accounted(&db, dir.path(), "full with a text index");
    assert_eq!(rows.len(), 1);
}

/// A dump larger than the cap restored into a capped database is refused
/// before the footprint passes the cap, the database still reading; one
/// that fits restores whole.
#[test]
fn a_restore_past_the_cap_is_refused_without_passing_it() {
    use coordinode_embed::backup::restore::RestoreOptions;
    use coordinode_embed::backup::{BackupFormat, export};

    let source_dir = tempfile::tempdir().unwrap();
    let mut source = Database::open(source_dir.path()).unwrap();
    for key in 0..200 {
        source.execute_cypher(&create(key)).unwrap();
    }
    let dump = |db: &Database| {
        let mut buf = Vec::new();
        let snapshot = db.engine().snapshot();
        export::export_binary(
            db.engine(),
            &db.interner().expect("dictionary"),
            1,
            &snapshot,
            &mut buf,
        )
        .expect("export");
        buf
    };
    let large = dump(&source);
    assert!(
        large.len() as u64 > LIMIT,
        "the dump is larger than the cap"
    );

    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open_with_config(capped(dir.path())).unwrap();
    let refused = db.restore(BackupFormat::Binary, &large, &RestoreOptions::default());
    assert!(refused.is_err(), "a dump past the cap is refused");
    assert!(
        refused.unwrap_err().to_string().contains("capacity"),
        "the refusal names the exhausted capacity"
    );
    assert_within(dir.path(), "after the refused restore");
    count(&mut db);

    let small_dir = tempfile::tempdir().unwrap();
    let mut small_source = Database::open(small_dir.path()).unwrap();
    for key in 0..20 {
        small_source.execute_cypher(&create(key)).unwrap();
    }
    let small = dump(&small_source);
    let fits = tempfile::tempdir().unwrap();
    let mut db = Database::open_with_config(capped(fits.path())).unwrap();
    db.restore(BackupFormat::Binary, &small, &RestoreOptions::default())
        .expect("a dump that fits restores");
    assert_within(fits.path(), "after the restore");
    assert_accounted(&db, fits.path(), "after the restore");
    assert_eq!(count(&mut db), 20);
}

/// Run `statement`, which a raised limit admits: a compaction the full
/// store held back may take the new room for a moment first, its output
/// charged with what the file system may allocate ahead of it until it is
/// closed, and gives it back when it ends.
fn admitted_once_room_settles(db: &mut Database, statement: &str) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        match db.execute_cypher(statement) {
            Ok(_) => return,
            Err(DatabaseError::CapacityExhausted { .. })
                if std::time::Instant::now() < deadline =>
            {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            Err(other) => panic!("a raised limit admits writes: {other:?}"),
        }
    }
}

/// The cap changes on a live database: a raised limit admits writes; a
/// limit lowered below what the database holds deletes nothing, refuses
/// writes and keeps reads; raising it again admits writes.
#[test]
fn a_limit_changed_at_runtime_takes_effect_without_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open_with_config(capped(dir.path())).unwrap();
    let accepted = fill(&mut db, dir.path(), 0);

    db.set_hard_limit("owned", 2 * LIMIT, None).unwrap();
    admitted_once_room_settles(&mut db, &create(accepted));

    db.set_hard_limit("owned", LIMIT / 4, None).unwrap();
    assert!(matches!(
        db.execute_cypher(&create(accepted + 1)),
        Err(DatabaseError::CapacityExhausted { .. })
    ));
    assert_eq!(
        count(&mut db),
        accepted as i64 + 1,
        "a lowered limit deletes nothing and reads go on"
    );

    db.set_hard_limit("owned", 2 * LIMIT, None).unwrap();
    admitted_once_room_settles(&mut db, &create(accepted + 1));
    assert!(
        db.set_hard_limit("missing", LIMIT, None).is_err(),
        "an endpoint without a limit is named in an error"
    );
}
