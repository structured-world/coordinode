use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_core::graph::edge::PostingList;
use std::sync::Arc;
use tempfile::TempDir;

fn test_engine() -> (Arc<StorageEngine>, Arc<TimestampOracle>, TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path().to_string_lossy().as_ref(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let oracle = Arc::new(TimestampOracle::new());
    let engine = Arc::new(StorageEngine::open_with_oracle(&config, oracle.clone()).unwrap());
    (engine, oracle, dir)
}

fn mvcc_txn<'a>(engine: &'a StorageEngine, oracle: &'a TimestampOracle) -> Transaction<'a> {
    let snap = engine.snapshot();
    Transaction::new(engine, Some(oracle), Timestamp::from_raw(snap), Some(snap))
}

/// Two attempts that each satisfy an at-most-one bound alone cannot both
/// commit, and the second is refused before anything of it is applied.
///
/// This is the schedule snapshot isolation admits: the two write different
/// adjacency keys, so first-committer-wins sees no conflict, and each
/// validated the bound against a state where it held. Only the claim they
/// both state about the bound tells them apart.
#[test]
fn two_attempts_deciding_one_bound_cannot_both_commit() {
    use coordinode_core::graph::node::NodeId;
    use coordinode_core::txn::invariant::{
        CardinalityMeasure, Claim, ClaimPredicate, ClaimScope, Direction,
    };

    let (engine, oracle, _d) = test_engine();
    let wc = WriteConcern::default();
    let ctx = CommitContext {
        write_concern: &wc,
        pipeline: None,
        id_gen: None,
        drain_buffer: None,
        nvme_write_buffer: None,
    };

    let bound = || {
        Claim::new(
            ClaimScope::Incident {
                node: NodeId::from_raw(1),
                edge_type: "OWNS".to_string(),
                direction: Direction::Outgoing,
            },
            ClaimPredicate::CardinalityBound {
                measure: CardinalityMeasure::DistinctNeighbours,
                at_most: Some(1),
                at_least: None,
            },
            0,
        )
    };

    // Both attempts are open at once, which is what makes this a race rather
    // than a sequence.
    let mut first = mvcc_txn(&engine, &oracle);
    let mut second = mvcc_txn(&engine, &oracle);

    first.merge_adj_add(b"adj:OWNS:out:\x00\x00\x00\x00\x00\x00\x00\x01", 2);
    first.claim(bound());
    second.merge_adj_add(b"adj:OWNS:out:\x00\x00\x00\x00\x00\x00\x00\x01", 3);
    second.claim(bound());

    first
        .commit(&ctx)
        .expect("the first attempt decides the bound");

    let err = second
        .commit(&ctx)
        .expect_err("the second cannot decide the same bound");
    assert!(
        matches!(err, CommitError::InvariantRefused { .. }),
        "expected the invariant to refuse, got {err:?}"
    );
}

/// An attempt whose condition no longer holds is refused even with nobody
/// else in flight: the registry sees attempts, the evaluation sees the state
/// they have all committed, and neither is sufficient alone.
#[test]
fn a_condition_broken_by_committed_state_refuses_the_commit() {
    use coordinode_core::graph::node::NodeId;
    use coordinode_core::txn::invariant::{Adjacency, Claim, ClaimPredicate, ClaimScope};

    let (engine, oracle, _d) = test_engine();
    let wc = WriteConcern::default();
    let ctx = CommitContext {
        write_concern: &wc,
        pipeline: None,
        id_gen: None,
        drain_buffer: None,
        nvme_write_buffer: None,
    };

    // Somebody already made the pair adjacent.
    engine
        .merge(
            Partition::Adj,
            b"adj:TAGGED:out:\x00\x00\x00\x00\x00\x00\x00\x01",
            &crate::engine::merge::encode_add(2),
        )
        .expect("merge");

    // This attempt was built on having seen the pair absent.
    let mut txn = mvcc_txn(&engine, &oracle);
    txn.claim(Claim::new(
        ClaimScope::Pair {
            source: NodeId::from_raw(1),
            target: NodeId::from_raw(2),
            edge_type: "TAGGED".to_string(),
        },
        ClaimPredicate::PairAdjacency {
            observed: Adjacency::Absent,
        },
        0,
    ));
    txn.put(Partition::Node, b"node:claimed", b"v").unwrap();

    let err = txn.commit(&ctx).expect_err("the observation is stale");
    assert!(matches!(err, CommitError::InvariantRefused { .. }));
    assert_eq!(
        engine.get(Partition::Node, b"node:claimed").unwrap(),
        None,
        "a refused commit applied nothing"
    );
}

/// An attempt that states no condition is untouched by any of this: the
/// ordinary write path does not pay for a guard it does not need.
#[test]
fn an_attempt_with_no_claims_commits_as_before() {
    let (engine, oracle, _d) = test_engine();
    let wc = WriteConcern::default();
    let ctx = CommitContext {
        write_concern: &wc,
        pipeline: None,
        id_gen: None,
        drain_buffer: None,
        nvme_write_buffer: None,
    };

    let mut txn = mvcc_txn(&engine, &oracle);
    txn.put(Partition::Node, b"node:plain", b"v").unwrap();
    txn.commit(&ctx).expect("commit");

    assert_eq!(
        engine
            .get(Partition::Node, b"node:plain")
            .unwrap()
            .as_deref(),
        Some(&b"v"[..])
    );
    assert_eq!(
        engine.claim_registry().reserved_claims(),
        0,
        "no claims were reserved for an attempt that stated none"
    );
}

/// Concurrent read-modify-write on one key loses nothing.
///
/// Each writer reads the value at its snapshot, writes the successor and
/// commits, retrying whenever it is told the key was taken. First-committer
/// -wins says every successful commit must have seen the value its predecessor
/// wrote, so the final value equals the number of successful commits. It does
/// not follow from validation alone: validation reads committed state, and
/// between one writer deciding the key is untouched and its write appearing,
/// another writer decides the same thing about the same key. Both used to be
/// applied, and the later one replaced the earlier without either being told.
#[test]
fn concurrent_writers_on_one_key_lose_no_update() {
    use std::sync::atomic::{AtomicU64, Ordering};

    let (engine, oracle, _d) = test_engine();
    const WRITERS: usize = 8;
    const PER_WRITER: usize = 40;
    const KEY: &[u8] = b"node:contended";

    engine
        .put(Partition::Node, KEY, &0u64.to_le_bytes())
        .expect("seed");

    let commits = AtomicU64::new(0);
    std::thread::scope(|scope| {
        for _ in 0..WRITERS {
            let engine = Arc::clone(&engine);
            let oracle = Arc::clone(&oracle);
            let commits = &commits;
            scope.spawn(move || {
                let wc = WriteConcern::default();
                for _ in 0..PER_WRITER {
                    // Retry until this writer's own increment lands. A refusal
                    // is the contract working, not a failure.
                    loop {
                        let snap = engine.snapshot();
                        let mut txn = Transaction::new(
                            &engine,
                            Some(&oracle),
                            Timestamp::from_raw(snap),
                            Some(snap),
                        );
                        let current = txn
                            .get(Partition::Node, KEY)
                            .expect("read")
                            .map(|v| {
                                u64::from_le_bytes(v.as_slice().try_into().expect("eight bytes"))
                            })
                            .unwrap_or(0);
                        txn.put(Partition::Node, KEY, &(current + 1).to_le_bytes())
                            .expect("stage");

                        let ctx = CommitContext {
                            write_concern: &wc,
                            pipeline: None,
                            id_gen: None,
                            drain_buffer: None,
                            nvme_write_buffer: None,
                        };
                        match txn.commit(&ctx) {
                            Ok(_) => {
                                commits.fetch_add(1, Ordering::Relaxed);
                                break;
                            }
                            Err(CommitError::Conflict(_)) => continue,
                            // A refusal of any other kind is a real failure,
                            // and naming it here says which one it was.
                            Err(e) => panic!("unexpected commit failure: {e:?}"),
                        }
                    }
                }
            });
        }
    });

    let stored = engine
        .get(Partition::Node, KEY)
        .expect("read")
        .expect("the key was seeded");
    let final_value = u64::from_le_bytes((&stored[..]).try_into().expect("eight bytes"));
    assert_eq!(
        final_value,
        commits.load(Ordering::Relaxed),
        "every successful commit must be represented in the value; a lower \
         value is an update that was applied and then silently replaced"
    );
    assert_eq!(
        final_value,
        (WRITERS * PER_WRITER) as u64,
        "and every writer's work must eventually land"
    );
    assert_eq!(
        engine.pending_commits().in_flight(),
        0,
        "no commit holds an admission after it is done"
    );
}

/// Folding the mutations does not fold what they protect.
///
/// The commit coalesces a run of adjacency adds on one key into a single
/// batch operand, which is the point of the merge path. The conditions those
/// adds stated are held apart from the operands and survive that reduction:
/// the attempt still carries one claim per endpoint it referenced, and the
/// commit still decides each of them.
#[test]
fn composing_the_deltas_preserves_the_claims_they_carried() {
    use coordinode_core::graph::node::NodeId;
    use coordinode_core::txn::invariant::{Claim, ClaimPredicate, ClaimScope};

    let (engine, oracle, _d) = test_engine();
    let wc = WriteConcern::default();
    let ctx = CommitContext {
        write_concern: &wc,
        pipeline: None,
        id_gen: None,
        drain_buffer: None,
        nvme_write_buffer: None,
    };

    let mut txn = mvcc_txn(&engine, &oracle);
    let hub = b"adj:OWNS:out:\x00\x00\x00\x00\x00\x00\x00\x01";
    for peer in 2..8_u64 {
        txn.merge_adj_add(hub, peer);
        txn.claim(Claim::new(
            ClaimScope::Node(NodeId::from_raw(peer)),
            ClaimPredicate::EndpointAlive,
            txn.schema_generation(),
        ));
    }

    assert_eq!(
        txn.claims().len(),
        6,
        "one claim per endpoint referenced, whatever the operands fold into"
    );

    txn.commit(&ctx).expect("commit");

    // The six adds became one batch operand; the six conditions did not
    // become one condition, and each was decided.
    let stored = engine
        .get(Partition::Adj, hub)
        .expect("get")
        .expect("posting");
    let plist = PostingList::from_bytes(&stored).expect("decode");
    assert_eq!(plist.as_slice(), &[2, 3, 4, 5, 6, 7]);
}

/// A refusal is clean, which is what makes the retry it advises safe: nothing
/// of the attempt is applied and the guard budget it held is returned. A
/// refusal that left either behind would turn a retry into a second attempt
/// racing the remains of the first.
#[test]
fn a_refused_attempt_leaves_neither_writes_nor_reservations() {
    use coordinode_core::graph::node::NodeId;
    use coordinode_core::txn::invariant::{Adjacency, Claim, ClaimPredicate, ClaimScope};

    let (engine, oracle, _d) = test_engine();
    let wc = WriteConcern::default();
    let ctx = CommitContext {
        write_concern: &wc,
        pipeline: None,
        id_gen: None,
        drain_buffer: None,
        nvme_write_buffer: None,
    };

    let mut txn = mvcc_txn(&engine, &oracle);
    txn.put(Partition::Node, b"node:refused", b"v").unwrap();
    txn.merge_adj_add(b"adj:TAGGED:out:\x00\x00\x00\x00\x00\x00\x00\x01", 2);
    txn.claim(Claim::new(
        ClaimScope::Pair {
            source: NodeId::from_raw(1),
            target: NodeId::from_raw(2),
            edge_type: "TAGGED".to_string(),
        },
        // An observation that was already untrue when it was made: the
        // shortest way to a refusal that is nobody else's fault.
        ClaimPredicate::PairAdjacency {
            observed: Adjacency::Present,
        },
        txn.schema_generation(),
    ));

    let err = txn.commit(&ctx).expect_err("the observation does not hold");
    assert!(matches!(err, CommitError::InvariantRefused { .. }));

    assert_eq!(
        engine.get(Partition::Node, b"node:refused").unwrap(),
        None,
        "a refused attempt applied none of its writes"
    );
    assert_eq!(
        engine
            .get(
                Partition::Adj,
                b"adj:TAGGED:out:\x00\x00\x00\x00\x00\x00\x00\x01"
            )
            .unwrap(),
        None,
        "nor any of its operands"
    );
    assert_eq!(
        engine.claim_registry().reserved_claims(),
        0,
        "and holds nothing that would refuse its own retry"
    );
}

/// A counter whose staged deltas leave i64 is refused at commit, not written
/// and met again in a compaction.
///
/// A counter operand carries no base, so nothing between here and the fold
/// can tell that the sum cannot exist. If the operand were written, the
/// caller would be told the write succeeded and the failure would surface in
/// a compaction, which has nobody to answer and would stop making progress on
/// that partition. The decision belongs where the deltas are assembled.
#[test]
fn a_counter_that_would_leave_the_range_is_refused_before_it_is_written() {
    let (engine, oracle, _d) = test_engine();
    let wc = WriteConcern::default();
    let ctx = CommitContext {
        write_concern: &wc,
        pipeline: None,
        id_gen: None,
        drain_buffer: None,
        nvme_write_buffer: None,
    };

    let mut txn = mvcc_txn(&engine, &oracle);
    txn.put(Partition::Node, b"node:ovf", b"v").unwrap();
    txn.push_counter_delta(b"counter:hot", i64::MAX);
    txn.push_counter_delta(b"counter:hot", 1);

    let err = txn
        .commit(&ctx)
        .expect_err("a sum outside i64 cannot commit");
    assert!(
        matches!(err, CommitError::CounterOverflow { .. }),
        "expected CounterOverflow, got {err:?}"
    );

    // Nothing from the transaction reached the engine: the refusal is before
    // the durable promise, so the ordinary write in it is gone too.
    assert_eq!(
        engine.get(Partition::Node, b"node:ovf").unwrap(),
        None,
        "a refused commit must leave no write behind"
    );
    assert_eq!(
        engine.get(Partition::Counter, b"counter:hot").unwrap(),
        None,
        "and no counter operand"
    );
}

/// Adjacency operands staged in one transaction apply in the order they were
/// staged, not in an order the commit path chose.
///
/// Both operands of a remove-then-add pair carry the same commit timestamp, so
/// nothing downstream can reorder them back: whatever order the commit emits is
/// the order the merge operator sees and the only order the key will ever have.
/// A transaction that detaches an edge and reattaches it, which a MERGE or a
/// delete-then-create in one statement does, means the edge to be present at
/// the end.
#[test]
fn adjacency_operands_keep_the_order_they_were_staged_in() {
    let (engine, oracle, _d) = test_engine();
    let wc = WriteConcern::default();
    let ctx = CommitContext {
        write_concern: &wc,
        pipeline: None,
        id_gen: None,
        drain_buffer: None,
        nvme_write_buffer: None,
    };

    // Add, then remove: the member is gone at the end.
    let mut txn = mvcc_txn(&engine, &oracle);
    txn.merge_adj_add(b"adj:T:out:gone", 7);
    txn.merge_adj_remove(b"adj:T:out:gone", 7);
    txn.commit(&ctx).expect("commit");

    let gone = engine.get(Partition::Adj, b"adj:T:out:gone").unwrap();
    let gone = gone.map(|v| PostingList::from_bytes(&v).unwrap());
    assert_eq!(
        gone.as_ref().map(|p| p.as_slice()).unwrap_or(&[]),
        &[] as &[u64],
        "add then remove leaves nothing"
    );

    // Remove, then add: the member is present at the end. The removal is of
    // something that was not there, which is exactly the shape a reattach
    // takes when the edge is written again in the same transaction.
    let mut txn = mvcc_txn(&engine, &oracle);
    txn.merge_adj_remove(b"adj:T:out:kept", 7);
    txn.merge_adj_add(b"adj:T:out:kept", 7);
    txn.commit(&ctx).expect("commit");

    let kept = engine.get(Partition::Adj, b"adj:T:out:kept").unwrap();
    let kept = kept.map(|v| PostingList::from_bytes(&v).unwrap());
    assert_eq!(
        kept.as_ref().map(|p| p.as_slice()).unwrap_or(&[]),
        &[7],
        "remove then add leaves the member, because that is the order it was staged in"
    );
}

#[test]
fn put_then_get_reads_own_write() {
    let (engine, oracle, _d) = test_engine();
    let mut txn = mvcc_txn(&engine, &oracle);
    txn.put(Partition::Node, b"k1", b"v1").unwrap();
    // Read-your-own-writes from the buffer, before any flush.
    assert_eq!(
        txn.get(Partition::Node, b"k1").unwrap().as_deref(),
        Some(&b"v1"[..])
    );
}

#[test]
fn delete_buffers_tombstone_visible_to_own_read() {
    let (engine, oracle, _d) = test_engine();
    let mut txn = mvcc_txn(&engine, &oracle);
    txn.put(Partition::Node, b"k", b"v").unwrap();
    txn.delete(Partition::Node, b"k").unwrap();
    assert_eq!(txn.get(Partition::Node, b"k").unwrap(), None);
}

#[test]
fn get_absent_key_returns_none() {
    let (engine, oracle, _d) = test_engine();
    let mut txn = mvcc_txn(&engine, &oracle);
    assert_eq!(txn.get(Partition::Node, b"missing").unwrap(), None);
}

#[test]
fn reads_leave_no_occ_scope() {
    // Reads are not conflict-tracked at the default level: commit validates
    // the WRITE set, so tracking every read would be per-read work feeding a
    // check that never runs. The scope stays unmaterialised until FOR UPDATE
    // or the serializable level asks for it.
    let (engine, oracle, _d) = test_engine();
    let mut txn = mvcc_txn(&engine, &oracle);
    txn.get(Partition::Node, b"r1").unwrap();
    txn.get(Partition::EdgeProp, b"r2").unwrap();
    assert!(
        txn.occ_scope.is_none(),
        "plain reads must not materialise a scope"
    );
    // Writing and reading back one's own write does not change that.
    txn.put(Partition::Node, b"w1", b"v").unwrap();
    txn.get(Partition::Node, b"w1").unwrap();
    assert!(txn.occ_scope.is_none());
}

#[test]
fn into_state_resume_preserves_buffer_occ_and_read_ts() {
    // The interactive-transaction park/resume cycle (ADR-042): a
    // transaction's progress survives being parked as TransactionState
    // and rebuilt with fresh engine/oracle borrows.
    let (engine, oracle, _d) = test_engine();
    let mut txn = mvcc_txn(&engine, &oracle);
    let read_ts = txn.read_ts();
    txn.put(Partition::Node, b"k1", b"v1").unwrap();
    txn.delete(Partition::Node, b"k2").unwrap();
    txn.get(Partition::Node, b"r1").unwrap();
    let occ_before = txn.occ_scope.as_ref().map(|s| s.tracked_count());

    // Park and rebuild.
    let state = txn.into_state();
    let mut resumed = Transaction::resume(&engine, Some(&oracle), state);

    // read_ts pinned (repeatable read across statements).
    assert_eq!(resumed.read_ts(), read_ts);
    // Buffered write + tombstone survive (read-your-own-writes still works).
    assert_eq!(
        resumed.get(Partition::Node, b"k1").unwrap().as_deref(),
        Some(&b"v1"[..])
    );
    assert_eq!(resumed.get(Partition::Node, b"k2").unwrap(), None);
    // The (unmaterialised) OCC scope state survives the park/resume cycle:
    // reads do not create one on either side of the boundary.
    resumed.get(Partition::Node, b"r1").unwrap();
    assert_eq!(
        resumed.occ_scope.as_ref().map(|s| s.tracked_count()),
        occ_before,
    );
    assert!(occ_before.is_none());
}

#[test]
fn prefix_scan_overlays_buffer_over_snapshot() {
    let (engine, oracle, _d) = test_engine();
    // Seed a committed row directly.
    engine.put(Partition::Node, b"p:a", b"old").unwrap();
    let mut txn = Transaction::new(
        &engine,
        Some(&oracle),
        Timestamp::from_raw(engine.snapshot()),
        Some(engine.snapshot()),
    );
    // Buffer overrides the committed value and adds a new key.
    txn.put(Partition::Node, b"p:a", b"new").unwrap();
    txn.put(Partition::Node, b"p:b", b"b").unwrap();
    let mut got = txn.prefix_scan(Partition::Node, b"p:").unwrap();
    got.sort();
    assert_eq!(
        got,
        vec![
            (b"p:a".to_vec(), b"new".to_vec()),
            (b"p:b".to_vec(), b"b".to_vec()),
        ]
    );
}

#[test]
fn prefix_scan_buffered_tombstone_does_not_hide_storage_row() {
    // Behavioural parity with the executor: a buffered in-transaction
    // delete does NOT remove a storage row from a prefix scan (only a
    // buffered *value* overlays). Point reads still see the tombstone via
    // `get`; scans surface the snapshot row.
    let (engine, oracle, _d) = test_engine();
    engine.put(Partition::Node, b"p:x", b"v").unwrap();
    let mut txn = Transaction::new(
        &engine,
        Some(&oracle),
        Timestamp::from_raw(engine.snapshot()),
        Some(engine.snapshot()),
    );
    txn.delete(Partition::Node, b"p:x").unwrap();
    // Point read sees the tombstone (RYOW).
    assert_eq!(txn.get(Partition::Node, b"p:x").unwrap(), None);
    // Scan still surfaces the storage row (documented parity behaviour).
    assert_eq!(
        txn.prefix_scan(Partition::Node, b"p:").unwrap(),
        vec![(b"p:x".to_vec(), b"v".to_vec())]
    );
}

#[test]
fn legacy_mode_writes_directly_no_buffer() {
    let (engine, _oracle, _d) = test_engine();
    // No oracle → legacy: put hits the engine immediately.
    let mut txn = Transaction::new(&engine, None, Timestamp::from_raw(0), None);
    txn.put(Partition::Node, b"lk", b"lv").unwrap();
    assert!(txn.is_mvcc().eq(&false));
    // Visible through a fresh engine read (not via buffer).
    assert_eq!(
        engine.get(Partition::Node, b"lk").unwrap().as_deref(),
        Some(&b"lv"[..])
    );
}

#[test]
fn prefix_scan_paged_walks_the_prefix_in_keyset_pages() {
    let (engine, oracle, _d) = test_engine();
    // Five committed rows under "p:", plus one outside it.
    for i in 0..5u8 {
        engine.put(Partition::Node, &[b'p', b':', i], &[i]).unwrap();
    }
    engine.put(Partition::Node, b"q:z", b"x").unwrap();
    let mut txn = Transaction::new(
        &engine,
        Some(&oracle),
        Timestamp::from_raw(engine.snapshot()),
        Some(engine.snapshot()),
    );

    // Page in batches of two, resuming by keyset off the last key.
    let mut all = Vec::new();
    let mut resume: Option<Vec<u8>> = None;
    loop {
        let page = txn
            .prefix_scan_paged(Partition::Node, b"p:", resume.as_deref(), 2)
            .unwrap();
        all.extend(page.rows.clone());
        if page.exhausted {
            assert!(page.rows.len() <= 2);
            break;
        }
        assert_eq!(page.rows.len(), 2, "a non-final page is full");
        resume = page.last_key;
        assert!(resume.is_some());
    }

    // All five prefix rows, in key order, none from outside the prefix.
    assert_eq!(all.len(), 5);
    for (idx, (key, _)) in all.iter().enumerate() {
        assert_eq!(key, &vec![b'p', b':', idx as u8]);
    }
}

#[test]
fn prefix_scan_paged_empty_prefix_is_exhausted_with_no_last_key() {
    let (engine, oracle, _d) = test_engine();
    engine.put(Partition::Node, b"q:a", b"x").unwrap();
    let mut txn = Transaction::new(
        &engine,
        Some(&oracle),
        Timestamp::from_raw(engine.snapshot()),
        Some(engine.snapshot()),
    );
    let page = txn
        .prefix_scan_paged(Partition::Node, b"p:", None, 10)
        .unwrap();
    assert!(page.rows.is_empty());
    assert!(page.exhausted);
    assert!(page.last_key.is_none());
}

#[test]
fn prefix_scan_paged_exact_limit_reports_exhausted() {
    let (engine, oracle, _d) = test_engine();
    engine.put(Partition::Node, b"p:a", b"1").unwrap();
    engine.put(Partition::Node, b"p:b", b"2").unwrap();
    let mut txn = Transaction::new(
        &engine,
        Some(&oracle),
        Timestamp::from_raw(engine.snapshot()),
        Some(engine.snapshot()),
    );
    // Exactly `limit` matching rows: exhausted, no phantom extra page.
    let page = txn
        .prefix_scan_paged(Partition::Node, b"p:", None, 2)
        .unwrap();
    assert_eq!(page.rows.len(), 2);
    assert!(page.exhausted);
    assert_eq!(page.last_key, Some(b"p:b".to_vec()));
}

/// The write-admission gate at the single commit locus: when the engine's
/// cached write pressure is `Stop`, a commit CARRYING WRITES is rejected
/// with a retryable backpressure error and nothing is applied; a read-only
/// commit passes untouched (there is nothing to admit).
#[test]
fn commit_rejects_writes_under_stop_pressure() {
    // The compaction monitor overwrites the cached tier every poll cycle
    // with the real (healthy) verdict, so the forced tier below must not
    // race with it: give the monitor an hour-long poll interval and let its
    // single startup tick land before forcing.
    let dir = tempfile::tempdir().unwrap();
    let config = {
        let mut c = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            dir.path().to_string_lossy().as_ref(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        c.compaction_poll_interval_ms = 3_600_000;
        c
    };
    let oracle = Arc::new(TimestampOracle::new());
    let engine = StorageEngine::open_with_oracle(&config, oracle.clone()).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(100));

    let wc = WriteConcern::default();
    let ctx = CommitContext {
        write_concern: &wc,
        pipeline: None,
        id_gen: None,
        drain_buffer: None,
        nvme_write_buffer: None,
    };
    engine.force_write_pressure(2); // Stop

    // Read-only transaction: commits fine under Stop.
    let mut ro = mvcc_txn(&engine, &oracle);
    ro.get(Partition::Node, b"r").unwrap();
    ro.commit(&ctx).expect("read-only commit passes");

    // Write transaction: rejected, and the write must not be visible.
    let mut txn = mvcc_txn(&engine, &oracle);
    txn.put(Partition::Node, b"bp:k", b"v").unwrap();
    let err = txn
        .commit(&ctx)
        .expect_err("write commit under Stop must be rejected");
    assert!(
        matches!(err, CommitError::Backpressure),
        "expected Backpressure, got {err:?}"
    );
    assert_eq!(engine.get(Partition::Node, b"bp:k").unwrap(), None);

    // Pressure clears: the same write commits.
    engine.force_write_pressure(0);
    let mut txn = mvcc_txn(&engine, &oracle);
    txn.put(Partition::Node, b"bp:k", b"v").unwrap();
    txn.commit(&ctx).expect("commit after drain");
    assert_eq!(
        engine.get(Partition::Node, b"bp:k").unwrap().as_deref(),
        Some(&b"v"[..])
    );
}

/// A pipeline on a member that is not the leader: every proposal is refused.
struct NotLeaderPipeline;

impl coordinode_core::txn::proposal::ProposalPipeline for NotLeaderPipeline {
    fn propose_and_wait(
        &self,
        _proposal: &coordinode_core::txn::proposal::RaftProposal,
    ) -> Result<
        coordinode_core::txn::proposal::ProposalOutcome,
        coordinode_core::txn::proposal::ProposalError,
    > {
        Err(coordinode_core::txn::proposal::ProposalError::NotLeader { leader_id: Some(2) })
    }
}

/// A write concern decides when the caller is answered, never whether the
/// write is replicated. With w:0 the commit used to apply straight to the
/// local engine, bypassing the pipeline: on a member that is not the leader
/// that left a record no other member would ever see. The commit must go to
/// the pipeline like any other, so here it fails and nothing is written.
#[test]
fn a_fire_and_forget_commit_never_writes_outside_the_pipeline() {
    let (engine, oracle, _d) = test_engine();
    let wc = WriteConcern::w0();
    let pipeline = NotLeaderPipeline;
    let ids = coordinode_core::txn::proposal::ProposalIdGenerator::new();
    let ctx = CommitContext {
        write_concern: &wc,
        pipeline: Some(&pipeline),
        id_gen: Some(&ids),
        drain_buffer: None,
        nvme_write_buffer: None,
    };

    let mut txn = mvcc_txn(&engine, &oracle);
    txn.put(Partition::Node, b"w0:node", b"v").unwrap();
    txn.merge_adj_add(b"w0:adj", 7);
    let err = txn
        .commit(&ctx)
        .expect_err("a member that is not the leader cannot commit, whatever the write concern");
    assert!(
        matches!(err, CommitError::NotLeader { .. }),
        "expected NotLeader, got {err:?}"
    );

    assert_eq!(
        engine.get(Partition::Node, b"w0:node").unwrap(),
        None,
        "a refused commit left a local record"
    );
    assert_eq!(
        engine.get(Partition::Adj, b"w0:adj").unwrap(),
        None,
        "a refused commit left a local adjacency operand"
    );
}
