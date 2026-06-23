use super::*;

fn writer() -> (ReplicatedWriter, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open database");
    (ReplicatedWriter::new(Arc::new(RwLock::new(db))), dir)
}

/// A write through the embedded (non-replicated) path returns no
/// committed index — `applied_index` is `None`, not a fabricated 0.
#[test]
fn embedded_write_has_no_committed_index() {
    let (w, _dir) = writer();
    let result = w
        .execute("CREATE (n:Probe {v: 1}) RETURN n", None, None, None, None)
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
    w.execute("CREATE (n:Probe {v: 1})", None, None, None, None)
        .expect("seed write");
    let result = w
        .execute("MATCH (n:Probe) RETURN n.v", None, None, None, None)
        .expect("read should succeed");
    assert!(!result.write_stats.has_mutations());
    assert_eq!(result.write_stats.applied_index, None);
}

/// The shared→exclusive fallback routes a session `SET` command to the
/// `&mut` path instead of surfacing the rejection.
#[test]
fn session_set_falls_back_to_exclusive() {
    let (w, _dir) = writer();
    let result = w.execute(
        "SET vector_consistency = 'snapshot'",
        None,
        None,
        None,
        None,
    );
    assert!(
        result.is_ok(),
        "session SET must route through exclusive access, got {:?}",
        result.err()
    );
}
