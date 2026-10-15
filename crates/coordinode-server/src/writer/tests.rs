use super::*;

fn writer() -> (ReplicatedWriter, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open database");
    (ReplicatedWriter::new(Arc::new(RwLock::new(db))), dir)
}

fn defaults() -> StatementOptions {
    StatementOptions::default()
}

/// A write through the embedded (non-replicated) path returns no
/// committed index — `applied_index` is `None`, not a fabricated 0.
#[test]
fn embedded_write_has_no_committed_index() {
    let (w, _dir) = writer();
    let result = w
        .execute("CREATE (n:Probe {v: 1}) RETURN n", None, None, &defaults())
        .expect("write should succeed");
    assert_eq!(result.write_stats.nodes_created, 1);
    assert_eq!(
        result.write_stats.applied_index, None,
        "embedded pipeline has no Raft log → no committed index"
    );
}

/// A read-only statement records no write and no committed index.
#[test]
fn read_only_has_no_committed_index() {
    let (w, _dir) = writer();
    w.execute("CREATE (n:Probe {v: 1})", None, None, &defaults())
        .expect("seed write");
    let result = w
        .execute("MATCH (n:Probe) RETURN n.v", None, None, &defaults())
        .expect("read should succeed");
    assert!(!result.write_stats.has_mutations());
    assert_eq!(result.write_stats.applied_index, None);
}

/// A session `SET` sent through the writer is refused and leaves the
/// database's setting as it was: the writer runs many clients' statements
/// against one database, and a setting written there would reach every one
/// of them. Before, it fell back to the exclusive path and changed the
/// database for all clients.
#[test]
fn a_session_set_does_not_reach_the_database() {
    use coordinode_core::graph::types::VectorConsistencyMode;

    let (w, _dir) = writer();
    let result = w.execute(
        "SET vector_consistency = 'snapshot'",
        None,
        None,
        &defaults(),
    );
    assert!(
        matches!(result, Err(DatabaseError::Semantic(_))),
        "refused, got {:?}",
        result.map(|r| r.rows)
    );
    assert_eq!(
        w.database().read().vector_consistency(),
        VectorConsistencyMode::default(),
        "the database keeps its own setting"
    );
}
