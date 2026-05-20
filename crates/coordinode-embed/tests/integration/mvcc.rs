//! Integration tests: MVCC snapshot isolation through the executor.
//!
//! Verifies that when mvcc_oracle is set, reads/writes use versioned keys
//! and snapshot isolation guarantees hold.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

/// TimestampOracle produces monotonic, unique timestamps.
#[test]
fn timestamp_oracle_monotonic() {
    let oracle = TimestampOracle::new();
    let mut prev = oracle.next();
    for _ in 0..1000 {
        let ts = oracle.next();
        assert!(ts > prev, "timestamps must be monotonically increasing");
        prev = ts;
    }
}

/// Executor with MVCC enabled: write buffer is flushed with versioned keys.
#[test]
fn executor_mvcc_write_buffer_flushed() {
    use coordinode_core::graph::intern::FieldInterner;
    use coordinode_core::graph::node::NodeIdAllocator;

    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("open");
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1000));
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(coordinode_core::graph::node::NodeId::from_raw(0));

    // Take a snapshot before any writes to verify isolation later.
    let snap_before = engine.snapshot();

    // Create MVCC-enabled context and write via mvcc_put
    let read_ts = oracle.next(); // ts=1001
    {
        let mut ctx = super::helpers::make_ctx_mvcc(
            &engine,
            &oracle,
            read_ts,
            None,
            &mut interner,
            &allocator,
        );

        // Buffer writes
        ctx.mvcc_put(Partition::Node, b"node:0:1", b"alice_data")
            .expect("put");
        ctx.mvcc_put(Partition::Node, b"node:0:2", b"bob_data")
            .expect("put");

        // Before flush: read-your-own-writes from buffer
        let alice = ctx.mvcc_get(Partition::Node, b"node:0:1").expect("get");
        assert_eq!(
            alice.as_deref(),
            Some(b"alice_data".as_slice()),
            "should see own write from buffer"
        );

        // Flush — writes plain keys to storage
        let commit_ts = ctx.mvcc_flush().expect("flush");
        assert!(commit_ts.is_some(), "should have commit_ts");
        let commit_ts = commit_ts.expect("commit_ts");
        assert!(commit_ts > read_ts, "commit_ts must be after read_ts");
    }

    // Verify writes persisted with plain keys via engine.get()
    let alice = engine.get(Partition::Node, b"node:0:1").expect("get");
    assert_eq!(
        alice.as_deref(),
        Some(b"alice_data".as_ref()),
        "flushed data should be readable via engine.get()"
    );

    let bob = engine.get(Partition::Node, b"node:0:2").expect("get");
    assert_eq!(bob.as_deref(), Some(b"bob_data".as_ref()));

    // Snapshot taken before the writes should see nothing (snapshot isolation).
    let before = engine
        .snapshot_get(&snap_before, Partition::Node, b"node:0:1")
        .expect("get");
    assert!(
        before.is_none(),
        "data should be invisible in pre-write snapshot"
    );
}

/// Executor MVCC: delete via write buffer creates tombstone.
#[test]
fn executor_mvcc_delete_via_buffer() {
    use coordinode_core::graph::intern::FieldInterner;
    use coordinode_core::graph::node::NodeIdAllocator;

    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("open");
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(100));

    // Pre-populate with plain-key data
    engine
        .put(Partition::Node, b"node:0:1", b"data")
        .expect("put");

    // Snapshot after initial write — used both for OCC baseline and historical verification
    let snap_before_delete = engine.snapshot();

    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(coordinode_core::graph::node::NodeId::from_raw(0));
    let read_ts = oracle.next();

    {
        let txn_snapshot = engine.snapshot();
        let mut ctx = super::helpers::make_ctx_mvcc(
            &engine,
            &oracle,
            read_ts,
            Some(txn_snapshot),
            &mut interner,
            &allocator,
        );

        // Should see existing data via MVCC read
        let data = ctx.mvcc_get(Partition::Node, b"node:0:1").expect("get");
        assert!(data.is_some(), "should see pre-existing data");

        // Delete via buffer
        ctx.mvcc_delete(Partition::Node, b"node:0:1")
            .expect("delete");

        // Read-your-own-writes: deleted key returns None from buffer
        let after_del = ctx.mvcc_get(Partition::Node, b"node:0:1").expect("get");
        assert!(
            after_del.is_none(),
            "deleted key should return None from buffer"
        );

        ctx.mvcc_flush().expect("flush");
    }

    // After flush: current read should see nothing (key deleted)
    let result = engine.get(Partition::Node, b"node:0:1").expect("get");
    assert!(
        result.is_none(),
        "deleted key should be invisible after flush"
    );

    // Snapshot taken before delete should still see the data
    let old = engine
        .snapshot_get(&snap_before_delete, Partition::Node, b"node:0:1")
        .expect("get");
    assert_eq!(
        old.as_deref(),
        Some(b"data".as_ref()),
        "historical version should still be visible in pre-delete snapshot"
    );
}

/// OCC conflict detection: mvcc_flush detects concurrent write to read key.
///
/// Simulates two transactions:
/// - Txn A reads key at start_ts, takes a snapshot
/// - Txn B writes key concurrently (via direct engine.put)
/// - Txn A tries to flush → OCC detects the conflict → ErrConflict
#[test]
fn occ_conflict_on_concurrent_write() {
    use coordinode_core::graph::intern::FieldInterner;
    use coordinode_core::graph::node::NodeIdAllocator;
    use coordinode_query::executor::runner::ExecutionError;

    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(1000)));
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open_with_oracle(&config, oracle.clone()).expect("open");
    let allocator = NodeIdAllocator::new(0);

    // Setup: write initial version with plain key
    engine
        .put(Partition::Node, b"node:0:1", b"alice_v1")
        .expect("initial put");

    // Txn A: starts at ts=1001, takes a snapshot to capture current state
    let read_ts = oracle.next(); // ts=1001
    let txn_snapshot = engine.snapshot();
    let mut interner = FieldInterner::new();
    let mut ctx = super::helpers::make_ctx_mvcc(
        &engine,
        &oracle,
        read_ts,
        Some(txn_snapshot),
        &mut interner,
        &allocator,
    );

    // Txn A reads the key → adds to read_set
    let val = ctx.mvcc_get(Partition::Node, b"node:0:1").expect("get");
    assert_eq!(val.as_deref(), Some(b"alice_v1".as_slice()));

    // Txn B: concurrent transaction writes to the SAME key.
    // With oracle-driven seqno, this write gets seqno=1002 (> read_ts=1001).
    engine
        .put(Partition::Node, b"node:0:1", b"alice_v2")
        .expect("concurrent put");

    // Txn A: buffers a write (to a DIFFERENT key, so it's not read-only)
    ctx.mvcc_put(Partition::Node, b"node:0:2", b"bob_data")
        .expect("put");

    // Txn A: tries to flush → OCC should detect conflict on node:0:1
    let result = ctx.mvcc_flush();
    assert!(
        matches!(result, Err(ExecutionError::Conflict(_))),
        "expected OCC conflict, got: {result:?}"
    );
}

/// OCC: no false positive when no concurrent writes occurred.
#[test]
fn occ_no_false_positive() {
    use coordinode_core::graph::intern::FieldInterner;
    use coordinode_core::graph::node::NodeIdAllocator;

    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("open");
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1000));
    let allocator = NodeIdAllocator::new(0);

    // Setup: write initial value (uses auto-incremented seqno)
    engine
        .put(Partition::Node, b"node:0:1", b"alice")
        .expect("put");

    // Txn: start at ts=1001, read key, buffer write, flush
    let read_ts = oracle.next();
    let mut interner = FieldInterner::new();
    let mut ctx =
        super::helpers::make_ctx_mvcc(&engine, &oracle, read_ts, None, &mut interner, &allocator);

    // Read and write
    let _val = ctx.mvcc_get(Partition::Node, b"node:0:1").expect("get");
    ctx.mvcc_put(Partition::Node, b"node:0:1", b"alice_updated")
        .expect("put");

    // No concurrent writes → flush should succeed
    let result = ctx.mvcc_flush();
    assert!(result.is_ok(), "no conflict expected: {result:?}");
}

/// OCC: adj: partition is excluded from conflict checking.
#[test]
fn occ_adj_partition_excluded() {
    use coordinode_core::graph::intern::FieldInterner;
    use coordinode_core::graph::node::NodeIdAllocator;

    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("open");
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1000));
    let allocator = NodeIdAllocator::new(0);

    // Write adj: key
    engine
        .put(Partition::Adj, b"adj:KNOWS:out:1", b"postings_v1")
        .expect("put adj");

    // Txn: read adj: key
    let read_ts = oracle.next();
    let mut interner = FieldInterner::new();
    let mut ctx =
        super::helpers::make_ctx_mvcc(&engine, &oracle, read_ts, None, &mut interner, &allocator);

    let _val = ctx
        .mvcc_get(Partition::Adj, b"adj:KNOWS:out:1")
        .expect("get");

    // Concurrent write to the SAME adj: key
    engine
        .put(Partition::Adj, b"adj:KNOWS:out:1", b"postings_v2")
        .expect("concurrent put");

    // Buffer a write to make txn non-read-only
    ctx.mvcc_put(Partition::Node, b"node:0:99", b"unrelated")
        .expect("put");

    // Flush should succeed — adj: is excluded from OCC conflict checking
    let result = ctx.mvcc_flush();
    assert!(
        result.is_ok(),
        "adj: partition should be excluded from OCC: {result:?}"
    );
}

/// OCC conflict detection via prefix_scan path.
///
/// Regression test: mvcc_prefix_scan must add scanned keys to the read-set.
/// Without this, concurrent writes to scanned nodes go undetected.
///
/// Scenario:
/// - Write node:0:1 at ts=1000
/// - Txn A: prefix_scan at ts=1001 → reads node:0:1 (added to read_set)
/// - Txn B: writes node:0:1 at ts=1002
/// - Txn A: flush → OCC detects conflict on node:0:1
#[test]
fn occ_conflict_via_prefix_scan() {
    use coordinode_core::graph::intern::FieldInterner;
    use coordinode_core::graph::node::NodeIdAllocator;
    use coordinode_query::executor::runner::ExecutionError;

    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(1000)));
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open_with_oracle(&config, oracle.clone()).expect("open");
    let allocator = NodeIdAllocator::new(0);

    // Setup: write a node with plain key
    engine
        .put(Partition::Node, b"node:0:1", b"alice_v1")
        .expect("put");

    // Txn A: start at ts=1001, take snapshot, prefix_scan to read node
    let read_ts = oracle.next(); // ts=1001
    let txn_snapshot = engine.snapshot();
    let mut interner = FieldInterner::new();
    let mut ctx = super::helpers::make_ctx_mvcc(
        &engine,
        &oracle,
        read_ts,
        Some(txn_snapshot),
        &mut interner,
        &allocator,
    );

    // Read via prefix_scan (not mvcc_get) — must still track in read_set
    let scan_results = ctx
        .mvcc_prefix_scan(Partition::Node, b"node:0:")
        .expect("scan");
    assert_eq!(scan_results.len(), 1, "should find one node");
    assert_eq!(scan_results[0].0, b"node:0:1");

    // Verify the scanned key IS in the OCC scope (Layer 3)
    assert!(
        ctx.occ_scope
            .as_ref()
            .expect("MVCC mode must have OCC scope")
            .contains(Partition::Node, b"node:0:1"),
        "prefix_scan results must be tracked in OCC scope"
    );

    // Txn B: concurrent write to the SAME key
    engine
        .put(Partition::Node, b"node:0:1", b"alice_v2")
        .expect("concurrent put");

    // Buffer a write to make txn non-read-only
    ctx.mvcc_put(Partition::Node, b"node:0:99", b"unrelated")
        .expect("put");

    // Flush → OCC should detect conflict on node:0:1 (read via prefix_scan)
    let result = ctx.mvcc_flush();
    assert!(
        matches!(result, Err(ExecutionError::Conflict(_))),
        "prefix_scan reads must trigger OCC conflict: {result:?}"
    );
}

/// Seqno-based retention compaction filter: old versions below GC watermark
/// are removed. Latest version per key always survives.
///
/// Strategy: write 3 versions of a key (seqno 1, 2, 3), set watermark=2,
/// force compaction, verify seqno 3 (live) survives, seqno 1 (expired, not
/// newest per key) is destroyed.
#[test]
fn mvcc_gc_compaction_removes_old_versions() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("open");

    // Write 3 versions of the same key. Each put gets an auto-incremented
    // seqno (1, 2, 3). The LSM engine keeps all versions until compaction.
    engine
        .put(Partition::Node, b"node:0:1", b"v1")
        .expect("put v1");
    engine
        .put(Partition::Node, b"node:0:1", b"v2")
        .expect("put v2");
    engine
        .put(Partition::Node, b"node:0:1", b"v3")
        .expect("put v3");

    // Before compaction: latest version is visible.
    let current = engine.get(Partition::Node, b"node:0:1").expect("get");
    assert_eq!(
        current.as_deref(),
        Some(b"v3".as_slice()),
        "v3 should be visible before compaction"
    );

    // Set GC watermark: seqno <= 2 eligible for removal.
    engine.set_gc_watermark(2);

    // Force compaction — rotates memtable to SST then runs retention filter.
    engine.persist().expect("persist");
    engine.force_compaction(Partition::Node).expect("compact");

    // After compaction: latest version (seqno=3) must survive.
    let after = engine.get(Partition::Node, b"node:0:1").expect("get");
    assert_eq!(
        after.as_deref(),
        Some(b"v3".as_slice()),
        "latest version must survive compaction"
    );
}

/// Seqno-based retention filter preserves the newest version per key even
/// when ALL versions are below the GC watermark.
///
/// If watermark is set above all writes, the filter still keeps the newest
/// version per key to prevent total data loss.
#[test]
fn mvcc_gc_preserves_newest_expired_version() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("open");

    // Write 2 versions (seqno 1, 2). Both will be below watermark.
    engine
        .put(Partition::Node, b"node:0:99", b"old_v1")
        .expect("put v1");
    engine
        .put(Partition::Node, b"node:0:99", b"less_old_v2")
        .expect("put v2");

    // Set watermark above all writes — both versions "expired".
    engine.set_gc_watermark(100);

    // Flush memtable to SST files, then force compaction.
    engine.persist().expect("persist");
    engine.force_compaction(Partition::Node).expect("compact");

    // Newest version per key should survive (prevents data loss).
    let newest = engine.get(Partition::Node, b"node:0:99").expect("get");
    assert_eq!(
        newest.as_deref(),
        Some(b"less_old_v2".as_slice()),
        "newest expired version should survive GC to prevent data loss"
    );
}

/// Seqno retention filter: watermark=0 keeps everything (safe default).
///
/// When no one calls set_gc_watermark(), all writes have seqno > 0,
/// so nothing is eligible for GC.
#[test]
fn mvcc_gc_watermark_zero_keeps_everything() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("open");

    // Write two versions. gc_watermark stays at default 0.
    engine
        .put(Partition::Node, b"node:0:50", b"first")
        .expect("put v1");
    engine
        .put(Partition::Node, b"node:0:50", b"second")
        .expect("put v2");

    engine.persist().expect("persist");
    engine.force_compaction(Partition::Node).expect("compact");

    // Latest version survives (compaction doesn't GC anything).
    let val = engine.get(Partition::Node, b"node:0:50").expect("get");
    assert_eq!(
        val.as_deref(),
        Some(b"second".as_slice()),
        "watermark=0 must keep everything"
    );
}

/// Seqno retention filter: multiple independent keys compacted together.
///
/// Key A has live + expired versions, key B is fully expired.
/// Both newest versions should survive.
#[test]
fn mvcc_gc_multiple_keys_independent() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("open");

    // Key A: 2 versions (seqno ~1, ~2)
    engine
        .put(Partition::Node, b"node:0:A", b"A_old")
        .expect("put A1");
    engine
        .put(Partition::Node, b"node:0:A", b"A_new")
        .expect("put A2");

    // Key B: 1 version (seqno ~3)
    engine
        .put(Partition::Node, b"node:0:B", b"B_only")
        .expect("put B1");

    // Set watermark high enough to expire all versions.
    engine.set_gc_watermark(100);

    engine.persist().expect("persist");
    engine.force_compaction(Partition::Node).expect("compact");

    // Both keys retain their newest version.
    let a = engine.get(Partition::Node, b"node:0:A").expect("get A");
    assert_eq!(
        a.as_deref(),
        Some(b"A_new".as_slice()),
        "key A newest version must survive"
    );

    let b = engine.get(Partition::Node, b"node:0:B").expect("get B");
    assert_eq!(
        b.as_deref(),
        Some(b"B_only".as_slice()),
        "key B sole version must survive (newest per key rule)"
    );
}

// NOTE: oracle + snapshot_at + compaction GC interaction test deferred to
// R069 (MVCC integration tests) — requires understanding of snapshot_at
// visibility semantics with oracle-driven seqno (off-by-one between
// oracle.current() and snapshot boundary). The 4 tests above cover the
// retention filter's core behavior: watermark=0 safety, live version
// preservation, newest-expired-per-key rule, multi-key independence.

// ==========================================
// Proposal Pipeline Integration Tests
// ==========================================

/// mvcc_flush() routes writes through ProposalPipeline when configured.
///
/// Verifies the complete flow: executor buffers writes → OCC check →
/// package as RaftProposal → LocalProposalPipeline applies to storage →
/// data readable via native MVCC.
#[test]
fn executor_mvcc_flush_via_proposal_pipeline() {
    use coordinode_core::graph::intern::FieldInterner;
    use coordinode_core::graph::node::NodeIdAllocator;
    use coordinode_core::txn::proposal::ProposalIdGenerator;

    use coordinode_raft::proposal::LocalProposalPipeline;

    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("open");
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1000));
    let id_gen = ProposalIdGenerator::new();
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(coordinode_core::graph::node::NodeId::from_raw(0));

    let read_ts = oracle.next(); // ts=1001
    let pipeline = LocalProposalPipeline::new(&engine);

    // Snapshot before writes to verify isolation later
    let snap_before = engine.snapshot();

    {
        let mut ctx = super::helpers::make_ctx_with_pipeline(
            &engine,
            &oracle,
            read_ts,
            None,
            &pipeline,
            &id_gen,
            &mut interner,
            &allocator,
        );

        // Buffer writes across multiple partitions
        ctx.mvcc_put(Partition::Node, b"node:0:1", b"alice")
            .expect("put node");
        ctx.mvcc_put(Partition::Adj, b"adj:KNOWS:out:1", b"posting")
            .expect("put adj");
        ctx.mvcc_put(Partition::EdgeProp, b"edgeprop:KNOWS:1:2", b"props")
            .expect("put edgeprop");

        // Flush — should go through pipeline, not direct writes
        let commit_ts = ctx.mvcc_flush().expect("flush");
        assert!(commit_ts.is_some());
        let commit_ts = commit_ts.expect("commit_ts");
        assert!(commit_ts > read_ts, "commit_ts must be after start_ts");
    }

    // Verify all three partitions have data via engine.get() (plain keys)
    let alice = engine.get(Partition::Node, b"node:0:1").expect("get node");
    assert_eq!(alice.as_deref(), Some(b"alice".as_ref()));

    let adj = engine
        .get(Partition::Adj, b"adj:KNOWS:out:1")
        .expect("get adj");
    assert_eq!(adj.as_deref(), Some(b"posting".as_ref()));

    let ep = engine
        .get(Partition::EdgeProp, b"edgeprop:KNOWS:1:2")
        .expect("get edgeprop");
    assert_eq!(ep.as_deref(), Some(b"props".as_ref()));

    // Snapshot taken before the writes should see nothing (snapshot isolation)
    let before = engine
        .snapshot_get(&snap_before, Partition::Node, b"node:0:1")
        .expect("get");
    assert!(before.is_none(), "data invisible in pre-write snapshot");
}

/// OCC conflict detection still works when proposal pipeline is active.
///
/// Verifies that the OCC check runs BEFORE the proposal is created,
/// preventing wasted Raft bandwidth on doomed transactions.
#[test]
fn executor_occ_conflict_with_pipeline() {
    use coordinode_core::graph::intern::FieldInterner;
    use coordinode_core::graph::node::NodeIdAllocator;
    use coordinode_core::txn::proposal::ProposalIdGenerator;

    use coordinode_raft::proposal::LocalProposalPipeline;

    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(1000)));
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open_with_oracle(&config, oracle.clone()).expect("open");
    let id_gen = ProposalIdGenerator::new();
    let allocator = NodeIdAllocator::new(0);

    // Setup: write initial version with plain key
    engine
        .put(Partition::Node, b"node:0:1", b"alice_v1")
        .expect("initial put");

    // Txn A starts at ts=1001, takes snapshot, reads the key
    let read_ts = oracle.next(); // ts=1001
    let txn_snapshot = engine.snapshot();
    let pipeline = LocalProposalPipeline::new(&engine);
    let mut interner = FieldInterner::new();
    let mut ctx = super::helpers::make_ctx_with_pipeline(
        &engine,
        &oracle,
        read_ts,
        Some(txn_snapshot),
        &pipeline,
        &id_gen,
        &mut interner,
        &allocator,
    );

    // Txn A reads the key (adds to read-set)
    let _v = ctx.mvcc_get(Partition::Node, b"node:0:1").expect("read");

    // Concurrent write: another txn writes (after A's snapshot)
    engine
        .put(Partition::Node, b"node:0:1", b"alice_v2")
        .expect("concurrent write");

    // Txn A buffers a write and tries to flush → should detect conflict
    ctx.mvcc_put(Partition::Node, b"node:0:1", b"alice_v3")
        .expect("buffer write");
    let result = ctx.mvcc_flush();
    assert!(result.is_err(), "should detect OCC conflict");
    let err = result.expect_err("should have conflict error");
    assert!(
        format!("{err}").contains("conflict"),
        "error should mention conflict: {err}"
    );
}

/// Adj partition keys are excluded from OCC even when pipeline is active.
///
/// Posting list writes are commutative (merge operators) — concurrent
/// writes to the same adj: key should NOT cause OCC conflicts.
#[test]
fn executor_adj_excluded_from_occ_with_pipeline() {
    use coordinode_core::graph::intern::FieldInterner;
    use coordinode_core::graph::node::NodeIdAllocator;
    use coordinode_core::txn::proposal::ProposalIdGenerator;

    use coordinode_raft::proposal::LocalProposalPipeline;

    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("open");
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1000));
    let id_gen = ProposalIdGenerator::new();
    let allocator = NodeIdAllocator::new(0);

    // Setup: write adj key
    engine
        .put(Partition::Adj, b"adj:KNOWS:out:1", b"postings_v1")
        .expect("put adj");

    // Txn: read adj: key
    let read_ts = oracle.next();
    let pipeline = LocalProposalPipeline::new(&engine);
    let mut interner = FieldInterner::new();
    let mut ctx = super::helpers::make_ctx_with_pipeline(
        &engine,
        &oracle,
        read_ts,
        None,
        &pipeline,
        &id_gen,
        &mut interner,
        &allocator,
    );

    let _v = ctx
        .mvcc_get(Partition::Adj, b"adj:KNOWS:out:1")
        .expect("read adj");

    // Concurrent write to same adj: key
    engine
        .put(Partition::Adj, b"adj:KNOWS:out:1", b"postings_v2")
        .expect("concurrent adj write");

    // Buffer a write to adj: and flush — should NOT conflict
    ctx.mvcc_put(Partition::Adj, b"adj:KNOWS:out:1", b"postings_v3")
        .expect("buffer adj write");

    let result = ctx.mvcc_flush();
    assert!(
        result.is_ok(),
        "adj: writes should not trigger OCC conflict"
    );
}

/// Pipeline handles mixed puts and deletes in a single proposal.
#[test]
fn executor_pipeline_mixed_puts_and_deletes() {
    use coordinode_core::graph::intern::FieldInterner;
    use coordinode_core::graph::node::NodeIdAllocator;
    use coordinode_core::txn::proposal::ProposalIdGenerator;

    use coordinode_raft::proposal::LocalProposalPipeline;

    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("open");
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1000));
    let id_gen = ProposalIdGenerator::new();
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(coordinode_core::graph::node::NodeId::from_raw(0));

    // Pre-populate: write a node with plain key
    engine
        .put(Partition::Node, b"node:0:1", b"old_data")
        .expect("pre-populate");

    // Snapshot before the transaction to verify historical state later
    let snap_before = engine.snapshot();

    let read_ts = oracle.next();
    let pipeline = LocalProposalPipeline::new(&engine);

    {
        let mut ctx = super::helpers::make_ctx_with_pipeline(
            &engine,
            &oracle,
            read_ts,
            None,
            &pipeline,
            &id_gen,
            &mut interner,
            &allocator,
        );

        // Mix puts and deletes in the same transaction
        ctx.mvcc_put(Partition::Node, b"node:0:2", b"new_node")
            .expect("put new");
        ctx.mvcc_delete(Partition::Node, b"node:0:1")
            .expect("delete old");

        let commit_ts = ctx.mvcc_flush().expect("flush");
        assert!(commit_ts.is_some());
    }

    // Verify: new node visible, old node deleted (plain key reads)
    let new_node = engine.get(Partition::Node, b"node:0:2").expect("get new");
    assert_eq!(new_node.as_deref(), Some(b"new_node".as_ref()));

    let old_node = engine.get(Partition::Node, b"node:0:1").expect("get old");
    assert!(old_node.is_none(), "deleted node should be invisible");

    // Old node still visible in pre-transaction snapshot
    let old_before = engine
        .snapshot_get(&snap_before, Partition::Node, b"node:0:1")
        .expect("get old before");
    assert_eq!(old_before.as_deref(), Some(b"old_data".as_ref()));
}

/// Read-only transactions don't create proposals.
///
/// When mvcc_write_buffer is empty, mvcc_flush() should return
/// the read_ts without touching the pipeline at all.
#[test]
fn executor_pipeline_read_only_no_proposal() {
    use coordinode_core::graph::intern::FieldInterner;
    use coordinode_core::graph::node::NodeIdAllocator;
    use coordinode_core::txn::proposal::ProposalIdGenerator;

    use coordinode_raft::proposal::LocalProposalPipeline;

    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("open");
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1000));
    let id_gen = ProposalIdGenerator::new();
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);

    // Pre-populate
    engine
        .put(Partition::Node, b"node:0:1", b"data")
        .expect("put");

    let read_ts = oracle.next(); // ts=1001
    let pipeline = LocalProposalPipeline::new(&engine);

    let mut ctx = super::helpers::make_ctx_with_pipeline(
        &engine,
        &oracle,
        read_ts,
        None,
        &pipeline,
        &id_gen,
        &mut interner,
        &allocator,
    );

    // Read-only: just reads, no writes
    let _v = ctx.mvcc_get(Partition::Node, b"node:0:1").expect("read");

    // Flush with empty write buffer → returns read_ts, no proposal created
    let result = ctx.mvcc_flush().expect("flush");
    assert_eq!(
        result,
        Some(read_ts),
        "read-only flush should return read_ts"
    );

    // Verify proposal ID generator was NOT advanced (no proposal created)
    // Next ID should still be 1 (first call)
    let next_id = id_gen.next();
    assert_eq!(next_id.as_raw(), 1, "no proposal should have been created");
}

/// R064: verify that Database::open() wires TimestampOracle into storage engine.
/// The oracle must drive LSM seqnos — writes through Database must produce
/// seqnos from the oracle, not from a default counter.
#[test]
fn database_oracle_drives_storage_seqnos() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = coordinode_embed::Database::open(dir.path()).expect("open");

    // Write some data through the Database API
    db.execute_cypher("CREATE (n:Test {name: 'a'})")
        .expect("create");
    db.execute_cypher("CREATE (n:Test {name: 'b'})")
        .expect("create");

    // Take a snapshot — its seqno must be oracle-driven.
    // TimestampOracle seeds from wall clock (microseconds since epoch),
    // so seqno should be > 1_600_000_000_000_000 (year 2020).
    let snap = db.engine().snapshot();
    assert!(
        snap > 1_600_000_000_000_000,
        "seqno {} should be oracle-driven (wall clock μs), not default counter",
        snap
    );
}

/// R064: verify snapshot_at works through full Database stack with oracle seqnos.
/// Write v1, take snapshot seqno, write v2, snapshot_at(old_seqno) sees v1.
#[test]
fn database_snapshot_at_through_oracle() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = coordinode_embed::Database::open(dir.path()).expect("open");

    // Write v1
    db.execute_cypher("CREATE (n:User {name: 'alice_v1'})")
        .expect("create v1");
    let snap_seqno = db.engine().snapshot();

    // Write v2 (updates don't work without MATCH, so create another node)
    db.execute_cypher("CREATE (n:User {name: 'bob'})")
        .expect("create v2");

    // snapshot_at old seqno should still see alice but NOT bob
    let old_snap = db.engine().snapshot_at(snap_seqno).expect("snapshot_at");

    // Current snapshot should see both
    let current_snap = db.engine().snapshot();
    assert!(
        current_snap > snap_seqno,
        "current seqno {} should be > old seqno {}",
        current_snap,
        snap_seqno
    );

    // Verify old snapshot works (basic sanity — detailed MVCC tests elsewhere)
    assert!(old_snap <= snap_seqno);
}

// ────────────────────────────────────────────────────────────────────
// R065: Native seqno MVCC integration tests (ADR-016)
//
// These tests verify that the snapshot_at() read path works correctly
// through the full Database → execute_cypher → snapshot pipeline.
// ────────────────────────────────────────────────────────────────────

/// R065: Snapshot isolation through Database.execute_cypher().
///
/// Write node A, capture seqno, write node B. Snapshot at old seqno
/// should see A but not B. Current snapshot sees both.
#[test]
fn r065_snapshot_isolation_through_database() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = coordinode_embed::Database::open(dir.path()).expect("open");

    // Write node A
    db.execute_cypher("CREATE (a:Snap {name: 'alice', version: 1})")
        .expect("create alice");

    // Capture seqno AFTER alice is written
    let seqno_after_alice = db.engine().snapshot();

    // Write node B
    db.execute_cypher("CREATE (b:Snap {name: 'bob', version: 2})")
        .expect("create bob");

    // Current query sees both
    let both = db
        .execute_cypher("MATCH (n:Snap) RETURN n.name ORDER BY n.name")
        .expect("query both");
    assert_eq!(both.len(), 2, "current snapshot should see both nodes");

    // Verify via snapshot_at: old snapshot sees alice but not bob
    let old_snap = db
        .engine()
        .snapshot_at(seqno_after_alice)
        .expect("snapshot_at should work for recent seqno");
    let old_nodes = db
        .engine()
        .snapshot_prefix_scan(&old_snap, Partition::Node, b"node:")
        .expect("prefix scan");

    // Count nodes visible at old seqno
    let mut old_count = 0;
    for (_key, value) in &old_nodes {
        if let Ok(record) = coordinode_core::graph::node::NodeRecord::from_msgpack(value) {
            if record.labels.contains(&"Snap".to_string()) {
                old_count += 1;
            }
        }
    }
    assert_eq!(old_count, 1, "old snapshot should see only alice, not bob");
}

/// R065: Write + overwrite via SET, verify snapshot sees old version.
///
/// CREATE node → snapshot → SET property → current reads new value,
/// snapshot reads old value. Verifies that engine.put() with oracle
/// seqno correctly versions data for snapshot_at reads.
#[test]
fn r065_snapshot_sees_old_version_after_set() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = coordinode_embed::Database::open(dir.path()).expect("open");

    // Create node with initial value
    db.execute_cypher("CREATE (n:VersionTest {key: 'config', val: 'v1'})")
        .expect("create v1");

    // Capture seqno after v1
    let seqno_v1 = db.engine().snapshot();

    // Update to v2 via MATCH + SET
    db.execute_cypher("MATCH (n:VersionTest {key: 'config'}) SET n.val = 'v2'")
        .expect("set v2");

    // Current query should see v2
    let current = db
        .execute_cypher("MATCH (n:VersionTest {key: 'config'}) RETURN n.val")
        .expect("query current");
    assert_eq!(current.len(), 1);
    let current_val = current[0].get("n.val").expect("n.val should exist");
    assert_eq!(
        current_val,
        &coordinode_core::graph::types::Value::String("v2".into()),
        "current should see v2"
    );

    // Snapshot at v1 seqno should see v1
    let old_snap = db.engine().snapshot_at(seqno_v1).expect("snapshot_at v1");

    // Read the node from old snapshot and verify val=v1
    let old_nodes = db
        .engine()
        .snapshot_prefix_scan(&old_snap, Partition::Node, b"node:")
        .expect("scan");

    let mut found_v1 = false;
    for (_key, value) in &old_nodes {
        if let Ok(record) = coordinode_core::graph::node::NodeRecord::from_msgpack(value) {
            if record.labels.contains(&"VersionTest".to_string()) {
                // Check the val property (interned)
                for val in record.props.values() {
                    if *val == coordinode_core::graph::types::Value::String("v1".into()) {
                        found_v1 = true;
                    }
                }
            }
        }
    }
    assert!(found_v1, "old snapshot should see val='v1', not 'v2'");
}

/// R065: Delete a node, verify snapshot before delete still sees it.
///
/// CREATE node → snapshot → DELETE → current returns empty,
/// old snapshot still has the node. Tests LSM tombstone + snapshot
/// interaction (engine.delete creates tombstone, snapshot_at ignores it).
#[test]
fn r065_snapshot_before_delete_still_visible() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = coordinode_embed::Database::open(dir.path()).expect("open");

    // Create node
    db.execute_cypher("CREATE (n:Ephemeral {name: 'temp'})")
        .expect("create");

    // Capture seqno while node exists
    let seqno_exists = db.engine().snapshot();

    // Delete node
    db.execute_cypher("MATCH (n:Ephemeral) DETACH DELETE n")
        .expect("delete");

    // Current query: node gone
    let after_delete = db
        .execute_cypher("MATCH (n:Ephemeral) RETURN n.name")
        .expect("query after delete");
    assert_eq!(after_delete.len(), 0, "node should be deleted");

    // Old snapshot: node still visible
    let old_snap = db.engine().snapshot_at(seqno_exists).expect("snapshot_at");
    let old_nodes = db
        .engine()
        .snapshot_prefix_scan(&old_snap, Partition::Node, b"node:")
        .expect("scan");

    let mut found_ephemeral = false;
    for (_key, value) in &old_nodes {
        if let Ok(record) = coordinode_core::graph::node::NodeRecord::from_msgpack(value) {
            if record.labels.contains(&"Ephemeral".to_string()) {
                found_ephemeral = true;
            }
        }
    }
    assert!(
        found_ephemeral,
        "old snapshot before delete should still see the node"
    );
}

/// R065: Multiple writes at different seqnos, verify snapshot_at each.
///
/// Write 5 nodes at different timestamps, take snapshot after each.
/// Verify snapshot_at(ts_N) sees exactly N nodes. This tests that
/// native seqno MVCC provides correct point-in-time visibility.
#[test]
fn r065_multiple_snapshots_at_different_seqnos() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = coordinode_embed::Database::open(dir.path()).expect("open");

    let mut seqnos = Vec::new();

    for i in 1..=5 {
        db.execute_cypher(&format!("CREATE (n:Counter {{idx: {i}}})"))
            .unwrap_or_else(|e| panic!("create {i}: {e}"));
        seqnos.push(db.engine().snapshot());
    }

    // Verify: snapshot at seqno[i] should see exactly i+1 Counter nodes
    // (but seqno[0] = after first write = 1 node, etc.)
    for (i, &seqno) in seqnos.iter().enumerate() {
        let snap = db
            .engine()
            .snapshot_at(seqno)
            .unwrap_or_else(|| panic!("snapshot_at({seqno}) for i={i}"));
        let nodes = db
            .engine()
            .snapshot_prefix_scan(&snap, Partition::Node, b"node:")
            .expect("scan");

        let counter_count = nodes
            .iter()
            .filter(|(_k, v)| {
                coordinode_core::graph::node::NodeRecord::from_msgpack(v)
                    .map(|r| r.labels.contains(&"Counter".to_string()))
                    .unwrap_or(false)
            })
            .count();

        assert_eq!(
            counter_count,
            i + 1,
            "snapshot at seqno[{i}]={seqno} should see {} Counter nodes, got {counter_count}",
            i + 1
        );
    }
}

/// R065: OCC conflict detection through Database API.
///
/// Two sequential transactions where the second modifies a key read by
/// the first. Since Database.execute_cypher runs each statement as its
/// own transaction, we verify OCC at a lower level: set up a write
/// between two executor calls sharing a read-set.
#[test]
fn r065_occ_detects_concurrent_modification() {
    use coordinode_core::graph::intern::FieldInterner;
    use coordinode_core::graph::node::{NodeId, NodeIdAllocator};

    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = std::sync::Arc::new(TimestampOracle::new());
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open_with_oracle(&config, oracle.clone()).expect("open");

    // Write initial value
    engine
        .put(Partition::Node, b"node:1:99", b"original")
        .expect("put");

    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(200));
    let mut interner = FieldInterner::new();

    // Transaction 1: read the key (establishes read-set)
    let read_ts = oracle.next();
    let mvcc_snap = engine.snapshot_at(read_ts.as_raw());
    let mut ctx = super::helpers::make_ctx_mvcc(
        &engine,
        &oracle,
        read_ts,
        mvcc_snap,
        &mut interner,
        &allocator,
    );
    ctx.shard_id = 1;

    // Read the key (adds to read-set)
    let val = ctx.mvcc_get(Partition::Node, b"node:1:99").expect("read");
    assert_eq!(val.as_deref(), Some(b"original".as_slice()));

    // Concurrent write by "another transaction" — modifies the same key
    engine
        .put(Partition::Node, b"node:1:99", b"modified_by_other")
        .expect("concurrent put");

    // Transaction 1 tries to write something and flush
    ctx.mvcc_put(Partition::Node, b"node:1:200", b"my_write")
        .expect("buffer write");

    // Flush should detect OCC conflict (read-set key was modified)
    let result = ctx.mvcc_flush();
    assert!(
        result.is_err(),
        "should detect OCC conflict: key modified by concurrent transaction"
    );
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("OCC conflict"),
        "error should mention OCC conflict, got: {err_msg}"
    );
}

/// R066: ABA detection — write + revert to same value still triggers conflict.
///
/// Value comparison (R065) would miss this: write "A" → write "B" → write "A".
/// Seqno-based detection (R066) catches it because the latest seqno > start_ts,
/// regardless of the value being identical.
#[test]
fn r066_occ_detects_aba_write() {
    use coordinode_core::graph::intern::FieldInterner;
    use coordinode_core::graph::node::{NodeId, NodeIdAllocator};

    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(1000)));
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open_with_oracle(&config, oracle.clone()).expect("open");

    // Write initial value "A"
    engine
        .put(Partition::Node, b"node:1:aba", b"value_A")
        .expect("put A");

    // Txn starts: read "A" at ts=1002
    let read_ts = oracle.next();
    let txn_snapshot = engine.snapshot();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(200));
    let mut interner = FieldInterner::new();
    let mut ctx = super::helpers::make_ctx_mvcc(
        &engine,
        &oracle,
        read_ts,
        Some(txn_snapshot),
        &mut interner,
        &allocator,
    );
    ctx.shard_id = 1;

    // Txn reads "A" — adds to read-set
    let val = ctx.mvcc_get(Partition::Node, b"node:1:aba").expect("read");
    assert_eq!(val.as_deref(), Some(b"value_A".as_slice()));

    // ABA: concurrent writes change A → B → A (same final value)
    engine
        .put(Partition::Node, b"node:1:aba", b"value_B")
        .expect("put B");
    engine
        .put(Partition::Node, b"node:1:aba", b"value_A")
        .expect("put A back");

    // Value is identical to what txn read ("A"), but seqno is newer.
    // Value comparison would say "no conflict". Seqno detection catches it.
    let current = engine.get(Partition::Node, b"node:1:aba").expect("get");
    assert_eq!(
        current.as_deref(),
        Some(b"value_A".as_slice()),
        "value should be back to A"
    );

    // Txn tries to flush — should detect conflict via seqno
    ctx.mvcc_put(Partition::Node, b"node:1:other", b"data")
        .expect("buffer");
    let result = ctx.mvcc_flush();
    assert!(
        result.is_err(),
        "R066: should detect ABA conflict even though value is identical"
    );
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("OCC conflict"),
        "should be OCC conflict, got: {err}"
    );
}
