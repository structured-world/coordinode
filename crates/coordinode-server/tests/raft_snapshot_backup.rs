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

/// A restored image carries its operation outcomes and admits nothing new
/// under a descriptor issued before the restore: a committed write resolves
/// as committed and its retry runs nothing, an unused number is refused, the
/// old incarnation cannot be resumed, and enrollment opens a fresh one above
/// every incarnation the image or the replaced store held.
#[test]
fn a_restored_snapshot_keeps_outcomes_and_fences_old_descriptors() {
    use coordinode_embed::db::StatementOptions;
    use coordinode_embed::operation::{IssuerId, OperationOutcome, Resolve};

    const ISSUER: IssuerId = IssuerId([0x31; 16]);
    let options = |op| StatementOptions {
        operation: Some(op),
        ..StatementOptions::default()
    };
    let dir = tempfile::tempdir().expect("tmpdir");
    let (src_dir, dst_dir) = (dir.path().join("src"), dir.path().join("dst"));
    let backup = dir.path().join("db.snap");
    let (done, unused) = {
        let src = Database::open(&src_dir).expect("open src");
        let issuer = src.enroll_issuer(ISSUER).expect("enroll");
        let done = issuer.next().expect("number");
        let unused = issuer.next().expect("number");
        src.execute_cypher_shared_with("CREATE (:Paid {k: 1})", None, None, &options(done))
            .expect("bound write");
        (done, unused)
    };
    // The store the image replaces enrolled the issuer three times.
    {
        let dst = Database::open(&dst_dir).expect("open dst");
        for _ in 0..3 {
            dst.enroll_issuer(ISSUER)
                .expect("enroll in the replaced store");
        }
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

    assert!(matches!(
        dst.resolve_operation(&done, Resolve::Lookup)
            .expect("lookup"),
        OperationOutcome::Committed { .. }
    ));
    assert!(
        dst.execute_cypher_shared_with("CREATE (:Paid {k: 1})", None, None, &options(done))
            .is_err(),
        "a retry of the committed write runs nothing"
    );
    assert!(
        dst.execute_cypher_shared_with("CREATE (:Paid {k: 2})", None, None, &options(unused))
            .is_err(),
        "an unused number of the old domain is refused"
    );
    assert_eq!(
        dst.resolve_operation(&unused, Resolve::Lookup)
            .expect("lookup"),
        OperationOutcome::Aborted
    );
    assert_eq!(
        dst.execute_cypher("MATCH (n:Paid) RETURN n")
            .expect("count")
            .len(),
        1
    );
    assert!(
        dst.resume_issuer(ISSUER, done.op.incarnation, unused.op.sequence)
            .is_err(),
        "the old incarnation is closed"
    );
    let fresh = dst.enroll_issuer(ISSUER).expect("enroll after restore");
    assert_eq!(
        fresh.incarnation(),
        4,
        "above every incarnation the image or the replaced store held"
    );
    let next = fresh.next().expect("number");
    dst.execute_cypher_shared_with("CREATE (:Paid {k: 3})", None, None, &options(next))
        .expect("a fresh incarnation writes");
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
