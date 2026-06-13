//! Round-trip test for the raft-snapshot backup format.
//!
//! A full Raft snapshot deliberately excludes the `meta:` Schema keys, which
//! include the field interner. A standalone backup therefore frames the
//! interner alongside the snapshot blob and restores it separately. This test
//! exercises that exact sequence at the API level (the CLI arms in `main.rs`
//! do the same framing) and verifies that property names resolve in a freshly
//! restored database, i.e. the interner survived the round trip.
#![allow(clippy::expect_used)]

use coordinode_core::graph::types::Value;
use coordinode_embed::Database;

#[test]
fn raft_snapshot_round_trips_data_and_interner() {
    let src_dir = tempfile::tempdir().expect("src tmpdir");
    let mut src = Database::open(src_dir.path()).expect("open src");
    src.execute_cypher("CREATE (a:User {name: 'Alice', age: 30})")
        .expect("create alice");
    src.execute_cypher("CREATE (b:User {name: 'Bob', age: 25})")
        .expect("create bob");

    // What the backup writes: framed interner + full snapshot blob.
    let interner_bytes = src.interner().to_bytes();
    let snapshot =
        coordinode_raft::snapshot::build_full_snapshot(src.engine()).expect("build snapshot");

    // What the restore does into a fresh database: interner first (the snapshot
    // omits it), then install the snapshot data.
    let dst_dir = tempfile::tempdir().expect("dst tmpdir");
    let mut dst = Database::open(dst_dir.path()).expect("open dst");
    dst.persist_field_interner_bytes(&interner_bytes)
        .expect("restore interner");
    coordinode_raft::snapshot::install_full_snapshot(dst.engine(), &snapshot)
        .expect("install snapshot");

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
