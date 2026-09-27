//! Round-trip test for the raft-snapshot backup format.
//!
//! A full Raft snapshot deliberately excludes the `meta:` Schema keys, which
//! include the field interner. A standalone backup therefore frames the
//! interner alongside the snapshot and restores it separately. These tests run
//! the `coordinode` binary, and check that property names resolve in a
//! freshly restored database, i.e. the interner survived the round trip.
#![allow(clippy::expect_used)]

use coordinode_core::graph::types::Value;
use coordinode_embed::Database;

/// Run `coordinode <args>` and require it to succeed.
fn coordinode(args: &[&std::ffi::OsStr]) {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_coordinode"))
        .args(args)
        .output()
        .expect("run coordinode");
    assert!(
        output.status.success(),
        "coordinode {args:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn raft_snapshot_round_trips_data_and_interner() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let (src_dir, dst_dir) = (dir.path().join("src"), dir.path().join("dst"));
    let backup = dir.path().join("db.snap");
    {
        let mut src = Database::open(&src_dir).expect("open src");
        src.execute_cypher("CREATE (a:User {name: 'Alice', age: 30})")
            .expect("create alice");
        src.execute_cypher("CREATE (b:User {name: 'Bob', age: 25})")
            .expect("create bob");
    }

    let format = std::ffi::OsStr::new("raft-snapshot");
    coordinode(&[
        "backup".as_ref(),
        "--data".as_ref(),
        src_dir.as_os_str(),
        "--output".as_ref(),
        backup.as_os_str(),
        "--format".as_ref(),
        format,
    ]);
    coordinode(&[
        "restore".as_ref(),
        "--data".as_ref(),
        dst_dir.as_os_str(),
        "--input".as_ref(),
        backup.as_os_str(),
        "--format".as_ref(),
        format,
    ]);
    let mut dst = Database::open(&dst_dir).expect("open dst");

    // Property names must resolve: an inline `{name: 'Alice'}` filter only
    // matches if the interner mapped "name" -> field id (proves the interner
    // was carried, not lost with the snapshot's meta exclusion).
    let alice = dst
        .execute_cypher("MATCH (n:User {name: 'Alice'}) RETURN n.age")
        .expect("query alice");
    assert_eq!(alice.len(), 1, "Alice resolves by name after restore");
    assert_eq!(
        alice[0].get("n.age"),
        Some(&Value::Int(30)),
        "Alice's age property survived the round trip"
    );

    let all = dst
        .execute_cypher("MATCH (n:User) RETURN n.name")
        .expect("query all users");
    assert_eq!(all.len(), 2, "both users restored");
}

/// 0.6 could also write a delta of the changes after a seqno. A delta cannot
/// stand alone, and a seqno does not bound what changed after it (an entry
/// can apply later at a lower commit timestamp), so restore refuses one and
/// says why rather than installing part of a database.
#[test]
fn an_incremental_backup_is_refused_by_restore() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let input = dir.path().join("delta.snap");
    // [mode 1][interner length 0][no interner][no snapshot]
    std::fs::write(&input, [1u8, 0, 0, 0, 0]).expect("write delta");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_coordinode"))
        .arg("restore")
        .arg("--data")
        .arg(dir.path().join("store"))
        .arg("--input")
        .arg(&input)
        .arg("--format")
        .arg("raft-snapshot")
        .output()
        .expect("run coordinode restore");
    assert!(!output.status.success(), "a delta must not restore");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("incremental raft-snapshot backup"),
        "the refusal names the reason, got: {stderr}"
    );
}
