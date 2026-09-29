//! Integration tests for embedded oplog-journal crash recovery.
//!
//! These validate the full round-trip an embedded engine performs:
//!   open_embedded → oplog_append + apply → drop (simulated crash) →
//!   open_embedded → verify the un-flushed tail is replayed from the journal.
//!
//! They mimic `OwnedLocalProposalPipeline`: a commit_ts is drawn from the
//! oracle and the proposal goes through `commit_journaled`, the same call the
//! real pipeline makes (journal, then apply at that ts with the coverage
//! marker). The recovery rule under test is per-partition coverage: an entry
//! is replayed into a partition iff that partition's record lacks its index.

use std::sync::Arc;

use coordinode_core::txn::proposal::{Mutation, PartitionId};
use coordinode_core::txn::timestamp::TimestampOracle;
use tempfile::TempDir;

use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, Tier};

/// A persistent (oplog-eligible) single-endpoint config rooted at `dir`.
fn durable_cfg(dir: &TempDir) -> StorageConfig {
    StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )])
}

/// Write a batch the way the embedded pipeline does: draw a commit_ts from the
/// oracle, journal the mutations at that ts, then apply them all at that ts.
/// Returns the commit_ts.
fn write_batch(
    engine: &StorageEngine,
    oracle: &Arc<TimestampOracle>,
    mutations: &[Mutation],
) -> u64 {
    let commit_ts = oracle.next().as_raw();
    engine
        .commit_journaled(mutations, commit_ts)
        .expect("commit_journaled");
    commit_ts
}

#[test]
fn embedded_engine_has_journal_on_durable_endpoint() {
    let dir = TempDir::new().expect("temp dir");
    let oracle = Arc::new(TimestampOracle::new());
    let engine = StorageEngine::open_embedded(&durable_cfg(&dir), oracle).expect("open");
    assert!(
        engine.has_journal(),
        "a Durable endpoint must get a retained oplog journal"
    );
}

/// Segment files under `root`, at any depth.
fn journal_segments(root: &std::path::Path) -> usize {
    std::fs::read_dir(root).map_or(0, |entries| {
        entries
            .filter_map(Result::ok)
            .map(|e| {
                let path = e.path();
                if path.is_dir() {
                    journal_segments(&path)
                } else {
                    usize::from(
                        path.file_name()
                            .and_then(|n| n.to_str())
                            .is_some_and(|n| n.starts_with("oplog-") && n.ends_with(".bin")),
                    )
                }
            })
            .sum()
    })
}

/// The embedded journal takes its segment rotation from the storage
/// configuration: it was opened with fixed defaults, so the operator's oplog
/// settings never reached it.
#[test]
fn the_embedded_journal_rotates_by_the_storage_configuration() {
    let dir = TempDir::new().expect("temp dir");
    let mut config = durable_cfg(&dir);
    config.oplog_segment_max_entries = 2;
    let oracle = Arc::new(TimestampOracle::new());
    let engine = StorageEngine::open_embedded(&config, oracle.clone()).expect("open");
    for i in 0..5u8 {
        write_batch(
            &engine,
            &oracle,
            &[Mutation::Put {
                partition: PartitionId::Node,
                key: vec![b'k', i],
                value: vec![i],
            }],
        );
    }
    assert!(
        journal_segments(dir.path()) >= 3,
        "5 entries at 2 per segment need at least three segments, found {}",
        journal_segments(dir.path())
    );
}

#[test]
fn put_survives_crash_via_journal_replay() {
    let dir = TempDir::new().expect("temp dir");

    // Step 1: journal + apply, then "crash" (drop without persist).
    {
        let oracle = Arc::new(TimestampOracle::new());
        let engine =
            StorageEngine::open_embedded(&durable_cfg(&dir), oracle.clone()).expect("open");
        write_batch(
            &engine,
            &oracle,
            &[Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:00:0001".to_vec(),
                value: b"survives".to_vec(),
            }],
        );
        // Drop without persist() — memtable lost, only the oplog has the write.
    }

    // Step 2: reopen — recovery must replay the journalled Put.
    {
        let oracle = Arc::new(TimestampOracle::new());
        let engine = StorageEngine::open_embedded(&durable_cfg(&dir), oracle).expect("reopen");
        let val = engine.get(Partition::Node, b"node:00:0001").expect("get");
        assert_eq!(
            val.as_deref(),
            Some(b"survives".as_slice()),
            "journal replay must restore an un-flushed Put"
        );
    }
}

#[test]
fn multi_partition_entry_replays_as_one_batch_at_its_ts() {
    // One journal entry spanning three partitions and a merge operand must
    // come back from replay exactly as it was committed: every op visible at a
    // snapshot one past the entry's ts, none of them at the ts itself, on
    // every partition. This pins the single-seqno MVCC contract across a
    // crash: a replayed entry is as atomic to a reader as a live one.
    let dir = TempDir::new().expect("temp dir");
    let mutations = [
        Mutation::Put {
            partition: PartitionId::Node,
            key: b"node:00:0007".to_vec(),
            value: b"n".to_vec(),
        },
        Mutation::Put {
            partition: PartitionId::Schema,
            key: b"schema:label:Anchor".to_vec(),
            value: b"{}".to_vec(),
        },
        Mutation::Merge {
            partition: PartitionId::Adj,
            key: b"adj:R:out:7".to_vec(),
            operand: crate::engine::merge::encode_add(9),
        },
        Mutation::Put {
            partition: PartitionId::Node,
            key: b"node:00:0008".to_vec(),
            value: b"m".to_vec(),
        },
    ];

    let commit_ts = {
        let oracle = Arc::new(TimestampOracle::new());
        let engine =
            StorageEngine::open_embedded(&durable_cfg(&dir), oracle.clone()).expect("open");
        write_batch(&engine, &oracle, &mutations)
        // Drop without persist(): only the oplog holds the entry.
    };

    let oracle = Arc::new(TimestampOracle::new());
    let engine = StorageEngine::open_embedded(&durable_cfg(&dir), oracle).expect("reopen");
    let probes: [(Partition, &[u8]); 4] = [
        (Partition::Node, b"node:00:0007"),
        (Partition::Schema, b"schema:label:Anchor"),
        (Partition::Adj, b"adj:R:out:7"),
        (Partition::Node, b"node:00:0008"),
    ];
    for (part, key) in probes {
        assert_eq!(
            engine.snapshot_get(&commit_ts, part, key).expect("get"),
            None,
            "{}:{}: nothing of the entry is visible before its ts",
            part.name(),
            String::from_utf8_lossy(key)
        );
        assert!(
            engine
                .snapshot_get(&(commit_ts + 1), part, key)
                .expect("get")
                .is_some(),
            "{}:{}: every op of the entry is visible at its ts after replay",
            part.name(),
            String::from_utf8_lossy(key)
        );
    }
}

#[test]
fn delete_survives_crash_via_journal_replay() {
    let dir = TempDir::new().expect("temp dir");

    // Step 1: write a value and make it durable (flushed to SST).
    {
        let oracle = Arc::new(TimestampOracle::new());
        let engine =
            StorageEngine::open_embedded(&durable_cfg(&dir), oracle.clone()).expect("open");
        write_batch(
            &engine,
            &oracle,
            &[Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:00:0002".to_vec(),
                value: b"to-delete".to_vec(),
            }],
        );
        engine.persist().expect("persist");

        // Now journal a Delete but do NOT persist — it lives only in the oplog.
        write_batch(
            &engine,
            &oracle,
            &[Mutation::Delete {
                partition: PartitionId::Node,
                key: b"node:00:0002".to_vec(),
            }],
        );
        // Crash.
    }

    // Step 2: reopen — the Delete must be replayed over the durable Put.
    {
        let oracle = Arc::new(TimestampOracle::new());
        let engine = StorageEngine::open_embedded(&durable_cfg(&dir), oracle).expect("reopen");
        let val = engine.get(Partition::Node, b"node:00:0002").expect("get");
        assert_eq!(
            val, None,
            "journal replay must restore an un-flushed Delete over flushed data"
        );
    }
}

#[test]
fn already_flushed_entries_are_not_replayed_over_newer_state() {
    // The critical no-double-apply guarantee: an entry whose data is already
    // durable in SST (entry.ts <= partition persisted seqno) must be SKIPPED on
    // recovery, while a later un-flushed entry to the same key IS replayed.
    let dir = TempDir::new().expect("temp dir");

    {
        let oracle = Arc::new(TimestampOracle::new());
        let engine =
            StorageEngine::open_embedded(&durable_cfg(&dir), oracle.clone()).expect("open");

        // A: durable (flushed). Its journal entry stays retained.
        write_batch(
            &engine,
            &oracle,
            &[Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:00:0003".to_vec(),
                value: b"v1-durable".to_vec(),
            }],
        );
        engine.persist().expect("persist A");

        // B: overwrite the same key, NOT flushed — only in the oplog.
        write_batch(
            &engine,
            &oracle,
            &[Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:00:0003".to_vec(),
                value: b"v2-journalled".to_vec(),
            }],
        );
        // Crash.
    }

    {
        let oracle = Arc::new(TimestampOracle::new());
        let engine = StorageEngine::open_embedded(&durable_cfg(&dir), oracle).expect("reopen");
        let val = engine.get(Partition::Node, b"node:00:0003").expect("get");
        // Recovery skips the already-durable A (ts <= persisted) and replays
        // only B, so the final value is B — never a stale resurrection of A.
        assert_eq!(
            val.as_deref(),
            Some(b"v2-journalled".as_slice()),
            "recovery must skip flushed entries and replay only the un-flushed tail"
        );
    }
}

#[test]
fn merge_replayed_once_not_doubled() {
    // A merge that is durable must be skipped on recovery (not re-applied),
    // while an un-flushed merge must be replayed exactly once. Posting-list
    // merges on Adj are additive, so a double-apply would be observable as a
    // duplicated edge — here we assert the recovered value byte-matches a
    // single clean application.
    let dir = TempDir::new().expect("temp dir");
    let operand = crate::engine::merge::encode_add(9);

    // Reference: a clean single application (no crash) for byte comparison.
    let reference = {
        let ref_dir = TempDir::new().expect("temp dir");
        let oracle = Arc::new(TimestampOracle::new());
        let engine =
            StorageEngine::open_embedded(&durable_cfg(&ref_dir), oracle.clone()).expect("open");
        write_batch(
            &engine,
            &oracle,
            &[Mutation::Merge {
                partition: PartitionId::Adj,
                key: b"adj:00:0008".to_vec(),
                operand: operand.clone(),
            }],
        );
        engine.persist().expect("persist");
        engine
            .get(Partition::Adj, b"adj:00:0008")
            .expect("get")
            .map(|v| v.to_vec())
    };

    {
        let oracle = Arc::new(TimestampOracle::new());
        let engine =
            StorageEngine::open_embedded(&durable_cfg(&dir), oracle.clone()).expect("open");
        write_batch(
            &engine,
            &oracle,
            &[Mutation::Merge {
                partition: PartitionId::Adj,
                key: b"adj:00:0008".to_vec(),
                operand: operand.clone(),
            }],
        );
        // Crash without persist — the merge lives only in the oplog.
    }

    {
        let oracle = Arc::new(TimestampOracle::new());
        let engine = StorageEngine::open_embedded(&durable_cfg(&dir), oracle).expect("reopen");
        let recovered = engine
            .get(Partition::Adj, b"adj:00:0008")
            .expect("get")
            .map(|v| v.to_vec());
        assert_eq!(
            recovered, reference,
            "a replayed merge must apply exactly once (byte-identical to a single clean apply)"
        );
    }
}

/// A store whose "power" can be cut: writes go through a fault injector over a
/// crash simulator. Dropping an engine flushes its memtables, so a plain drop
/// is a clean shutdown, not a crash; [`PowerRig::cut`] first makes every write
/// and sync fail, so that last flush never reaches the disk, then rolls every
/// file back to its last fsync.
struct PowerRig {
    dir: TempDir,
    crash: Arc<lsm_tree::fs::CrashFs>,
    faults: Arc<lsm_tree::fs::FaultInjector>,
    fs: Arc<dyn lsm_tree::fs::Fs>,
}

impl PowerRig {
    fn new() -> Self {
        let dir = TempDir::new().expect("temp dir");
        let crash = Arc::new(lsm_tree::fs::CrashFs::new(lsm_tree::fs::StdFs));
        let faults = Arc::new(lsm_tree::fs::FaultInjector::new());
        let fs: Arc<dyn lsm_tree::fs::Fs> = Arc::new(lsm_tree::fs::FaultFs::with_injector(
            lsm_tree::fs::CrashFs::clone(&crash),
            Arc::clone(&faults),
        ));
        Self {
            dir,
            crash,
            faults,
            fs,
        }
    }

    fn config(&self) -> StorageConfig {
        durable_cfg(&self.dir).with_fs(Arc::clone(&self.fs))
    }

    fn open(&self) -> (StorageEngine, Arc<TimestampOracle>) {
        let oracle = Arc::new(TimestampOracle::new());
        let engine = StorageEngine::open_embedded(&self.config(), oracle.clone()).expect("open");
        (engine, oracle)
    }

    /// Lose power: nothing written from now on is durable, the engine goes
    /// away, and every file falls back to what was fsynced before the cut.
    fn cut(&self, engine: StorageEngine) {
        use lsm_tree::fs::{Fault, FaultOp, FaultRule};
        let refuse = Fault::Error(lsm_tree::io::ErrorKind::Other);
        for op in [
            FaultOp::Open,
            FaultOp::Write,
            FaultOp::SyncAll,
            FaultOp::SyncData,
            FaultOp::Rename,
        ] {
            self.faults.arm(FaultRule::new(op, refuse));
        }
        drop(engine);
        self.faults.clear();
        self.crash.crash();
    }
}

/// Journal and apply `mutations` at an explicit `commit_ts`, the way a late
/// finalize does: the timestamp was reserved earlier than a commit that
/// already landed, so the physical apply follows a larger timestamp.
fn write_batch_at(engine: &StorageEngine, mutations: &[Mutation], commit_ts: u64) {
    engine
        .commit_journaled(mutations, commit_ts)
        .expect("commit_journaled");
}

#[test]
fn late_finalize_behind_a_flushed_newer_commit_survives_a_crash() {
    // T100 reserves its timestamp first; T200 commits and reaches an SST;
    // then T100 finalizes into the memtable only. A partition's highest
    // persisted seqno is now 200, so a recovery that trusts it as coverage
    // skips T100 and loses an acknowledged write.
    let rig = PowerRig::new();
    {
        let (engine, oracle) = rig.open();
        let t100 = oracle.next().as_raw();
        let t200 = oracle.next().as_raw();
        write_batch_at(
            &engine,
            &[Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:00:0200".to_vec(),
                value: b"t200".to_vec(),
            }],
            t200,
        );
        engine.persist().expect("persist t200");
        write_batch_at(
            &engine,
            &[Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:00:0100".to_vec(),
                value: b"t100".to_vec(),
            }],
            t100,
        );
        // Power loss: T100 is only in the journal.
        rig.cut(engine);
    }

    let (engine, _) = rig.open();
    assert_eq!(
        engine
            .get(Partition::Node, b"node:00:0100")
            .expect("get")
            .as_deref(),
        Some(b"t100".as_slice()),
        "the late-finalized commit must be replayed"
    );
    assert_eq!(
        engine
            .get(Partition::Node, b"node:00:0200")
            .expect("get")
            .as_deref(),
        Some(b"t200".as_slice()),
        "the flushed newer commit must survive"
    );
}

#[test]
fn late_finalized_merge_is_replayed_exactly_once() {
    // The same late-finalize order with a non-idempotent merge on the key the
    // flushed commit also merged into: skipping it loses an edge, replaying the
    // flushed one as well doubles it. Only exact coverage gets both right.
    let key = b"adj:R:out:42".to_vec();
    let first = crate::engine::merge::encode_add(1);
    let second = crate::engine::merge::encode_add(2);

    // Reference: both operands applied once each, with no crash.
    let reference = {
        let ref_dir = TempDir::new().expect("temp dir");
        let oracle = Arc::new(TimestampOracle::new());
        let engine =
            StorageEngine::open_embedded(&durable_cfg(&ref_dir), oracle.clone()).expect("open");
        let t100 = oracle.next().as_raw();
        let t200 = oracle.next().as_raw();
        for (operand, ts) in [(&second, t200), (&first, t100)] {
            write_batch_at(
                &engine,
                &[Mutation::Merge {
                    partition: PartitionId::Adj,
                    key: key.clone(),
                    operand: operand.clone(),
                }],
                ts,
            );
        }
        engine.persist().expect("persist");
        engine
            .get(Partition::Adj, &key)
            .expect("get")
            .map(|v| v.to_vec())
    };

    let rig = PowerRig::new();
    {
        let (engine, oracle) = rig.open();
        let t100 = oracle.next().as_raw();
        let t200 = oracle.next().as_raw();
        write_batch_at(
            &engine,
            &[Mutation::Merge {
                partition: PartitionId::Adj,
                key: key.clone(),
                operand: second.clone(),
            }],
            t200,
        );
        engine.persist().expect("persist t200");
        write_batch_at(
            &engine,
            &[Mutation::Merge {
                partition: PartitionId::Adj,
                key: key.clone(),
                operand: first.clone(),
            }],
            t100,
        );
        rig.cut(engine);
    }

    let (engine, _) = rig.open();
    assert_eq!(
        engine
            .get(Partition::Adj, &key)
            .expect("get")
            .map(|v| v.to_vec()),
        reference,
        "each merge operand must be applied exactly once after recovery"
    );
}

#[test]
fn recovery_twice_in_a_row_applies_each_merge_once() {
    // A recovery that replays the tail and then crashes again before any
    // flush must not apply the tail twice on the second open.
    let key = b"adj:R:out:77".to_vec();
    let operand = crate::engine::merge::encode_add(5);
    let reference = {
        let ref_dir = TempDir::new().expect("temp dir");
        let oracle = Arc::new(TimestampOracle::new());
        let engine =
            StorageEngine::open_embedded(&durable_cfg(&ref_dir), oracle.clone()).expect("open");
        write_batch(
            &engine,
            &oracle,
            &[Mutation::Merge {
                partition: PartitionId::Adj,
                key: key.clone(),
                operand: operand.clone(),
            }],
        );
        engine.persist().expect("persist");
        engine
            .get(Partition::Adj, &key)
            .expect("get")
            .map(|v| v.to_vec())
    };
    let rig = PowerRig::new();
    {
        let (engine, oracle) = rig.open();
        write_batch(
            &engine,
            &oracle,
            &[Mutation::Merge {
                partition: PartitionId::Adj,
                key: key.clone(),
                operand: operand.clone(),
            }],
        );
        rig.cut(engine);
    }
    for _ in 0..2 {
        let (engine, _) = rig.open();
        assert_eq!(
            engine
                .get(Partition::Adj, &key)
                .expect("get")
                .map(|v| v.to_vec()),
            reference,
            "a repeated recovery must not re-apply a covered merge"
        );
        rig.cut(engine);
    }
}

#[test]
fn late_finalized_point_and_range_batch_survives_a_crash() {
    // One entry carrying a range delete and a put, finalized behind a flushed
    // newer commit: after the crash the range delete must still hide the rows
    // it covered, and the put must be back.
    let rig = PowerRig::new();
    {
        let (engine, oracle) = rig.open();
        let t_seed = oracle.next().as_raw();
        write_batch_at(
            &engine,
            &[
                Mutation::Put {
                    partition: PartitionId::Node,
                    key: b"node:00:r1".to_vec(),
                    value: b"doomed".to_vec(),
                },
                Mutation::Put {
                    partition: PartitionId::Node,
                    key: b"node:00:r2".to_vec(),
                    value: b"doomed".to_vec(),
                },
            ],
            t_seed,
        );
        engine.persist().expect("persist seed");
        let t100 = oracle.next().as_raw();
        let t200 = oracle.next().as_raw();
        write_batch_at(
            &engine,
            &[Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:00:0200".to_vec(),
                value: b"t200".to_vec(),
            }],
            t200,
        );
        engine.persist().expect("persist t200");
        write_batch_at(
            &engine,
            &[
                Mutation::RemoveRange {
                    partition: PartitionId::Node,
                    start: b"node:00:r".to_vec(),
                    end: b"node:00:s".to_vec(),
                },
                Mutation::Put {
                    partition: PartitionId::Node,
                    key: b"node:00:0100".to_vec(),
                    value: b"t100".to_vec(),
                },
            ],
            t100,
        );
        rig.cut(engine);
    }

    let (engine, _) = rig.open();
    for key in [b"node:00:r1".as_slice(), b"node:00:r2"] {
        assert_eq!(
            engine.get(Partition::Node, key).expect("get"),
            None,
            "{}: the late-finalized range delete must be replayed",
            String::from_utf8_lossy(key)
        );
    }
    assert_eq!(
        engine
            .get(Partition::Node, b"node:00:0100")
            .expect("get")
            .as_deref(),
        Some(b"t100".as_slice()),
        "the put in the same entry must be replayed"
    );
}

#[test]
fn an_entry_flushed_in_one_partition_is_replayed_only_into_the_other() {
    // One entry merges into Adj and Counter; Adj reaches an SST on its own
    // while Counter's memtable is lost. Recovery must re-apply the entry to
    // Counter only: skipping it loses the delta, replaying it into Adj too
    // doubles the edge.
    let adj_key = b"adj:R:out:9".to_vec();
    let counter_key = b"counter:degree:9".to_vec();
    let entry = [
        Mutation::Merge {
            partition: PartitionId::Adj,
            key: adj_key.clone(),
            operand: crate::engine::merge::encode_add(3),
        },
        Mutation::Merge {
            partition: PartitionId::Counter,
            key: counter_key.clone(),
            operand: crate::engine::merge::encode_counter_delta(1),
        },
    ];
    let reference = {
        let ref_dir = TempDir::new().expect("temp dir");
        let oracle = Arc::new(TimestampOracle::new());
        let engine =
            StorageEngine::open_embedded(&durable_cfg(&ref_dir), oracle.clone()).expect("open");
        write_batch(&engine, &oracle, &entry);
        (
            engine.get(Partition::Adj, &adj_key).expect("get adj"),
            engine
                .get(Partition::Counter, &counter_key)
                .expect("get counter"),
        )
    };

    let rig = PowerRig::new();
    {
        let (engine, oracle) = rig.open();
        write_batch(&engine, &oracle, &entry);
        engine
            .tree(Partition::Adj)
            .expect("adj tree")
            .flush_active_memtable(0)
            .expect("flush adj alone");
        rig.cut(engine);
    }

    let (engine, _) = rig.open();
    assert_eq!(
        (
            engine.get(Partition::Adj, &adj_key).expect("get adj"),
            engine
                .get(Partition::Counter, &counter_key)
                .expect("get counter"),
        ),
        reference,
        "each partition must hold the entry exactly once after recovery"
    );
}

#[test]
fn folded_coverage_survives_a_crash_and_later_entries_replay_once() {
    // A purge folds the applied prefix into every tree's base and removes the
    // markers under it. Entries below the fold are covered by the base alone;
    // an entry after it is covered by nothing on disk and is replayed, once.
    let key = b"counter:degree:77".to_vec();
    let delta = |d| Mutation::Merge {
        partition: PartitionId::Counter,
        key: key.clone(),
        operand: crate::engine::merge::encode_counter_delta(d),
    };
    let rig = PowerRig::new();
    {
        let (engine, oracle) = rig.open();
        for _ in 0..3 {
            write_batch(&engine, &oracle, &[delta(1)]);
        }
        // Keeps every segment (inside the window) but folds and persists.
        engine.oplog_purge_expired(0, u64::MAX).expect("purge");
        write_batch(&engine, &oracle, &[delta(10)]);
        rig.cut(engine);
    }
    for _ in 0..2 {
        let (engine, _) = rig.open();
        let value = engine
            .get(Partition::Counter, &key)
            .expect("get")
            .expect("counter present");
        assert_eq!(
            crate::engine::merge::decode_counter(&value).expect("decode"),
            13,
            "three folded deltas and one replayed delta, each exactly once"
        );
        rig.cut(engine);
    }
}

#[test]
fn a_journal_without_a_coverage_record_is_refused_untouched() {
    // A store whose journal holds entries but whose trees carry no coverage
    // record (what a release before apply coverage leaves behind) cannot say
    // which entries are on disk. Opening it must fail without writing.
    let dir = TempDir::new().expect("temp dir");
    {
        let oracle = Arc::new(TimestampOracle::new());
        let engine =
            StorageEngine::open_embedded(&durable_cfg(&dir), oracle.clone()).expect("open");
        write_batch(
            &engine,
            &oracle,
            &[Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:00:0001".to_vec(),
                value: b"v".to_vec(),
            }],
        );
        // Strip the record the way a legacy store never had it.
        let at = engine.next_seqno();
        for &part in Partition::all() {
            let tree = engine.tree(part).expect("tree");
            let domain = coverage::Domain::Journal;
            tree.remove(domain.base_key(), at);
            tree.remove_range(
                domain.marker_key(0, 0).to_vec(),
                domain.marker_key(u64::MAX, u32::MAX).to_vec(),
                at,
            );
        }
        engine.persist().expect("persist");
    }
    let oracle = Arc::new(TimestampOracle::new());
    let refused = StorageEngine::open_embedded(&durable_cfg(&dir), oracle);
    assert!(
        matches!(
            refused,
            Err(StorageError::CoverageUnprovable { entries: 1, .. })
        ),
        "a journal with no coverage record must be refused, got {:?}",
        refused.err()
    );
}

#[test]
fn coverage_keys_stay_out_of_user_scans() {
    // The reserved namespace is engine state: a fresh store holds no user
    // data, a whole-partition scan returns only user rows, and the changed-key
    // feed neither reports the markers nor trips over the fold's tombstone.
    let dir = TempDir::new().expect("temp dir");
    let oracle = Arc::new(TimestampOracle::new());
    let engine = StorageEngine::open_embedded(&durable_cfg(&dir), oracle.clone()).expect("open");
    assert!(
        !engine.holds_user_data().expect("holds_user_data"),
        "a fresh store's coverage record is not user data"
    );
    let since = engine.current_seqno();
    write_batch(
        &engine,
        &oracle,
        &[Mutation::Put {
            partition: PartitionId::Blob,
            key: b"blob:aa".to_vec(),
            value: b"x".to_vec(),
        }],
    );
    engine.oplog_purge_expired(0, u64::MAX).expect("fold");
    let rows: Vec<Vec<u8>> = engine
        .prefix_scan(Partition::Blob, b"")
        .expect("scan")
        .map(|g| g.into_inner().expect("row").0.to_vec())
        .collect();
    assert_eq!(rows, vec![b"blob:aa".to_vec()]);
    assert_eq!(
        engine
            .changed_keys_since(Partition::Blob, since)
            .expect("changed keys"),
        vec![b"blob:aa".to_vec()]
    );
    assert!(engine.holds_user_data().expect("holds_user_data"));
}

#[cfg(feature = "columnar")]
#[test]
fn columnar_row_survives_crash_via_journal_replay() {
    // A columnar table write is journalled at its commit_ts before it touches
    // the table tree. With NO explicit flush, a crash loses the memtable — the
    // oplog must replay the row on reopen, exactly like the partition path.
    let dir = TempDir::new().expect("temp dir");

    {
        let oracle = Arc::new(TimestampOracle::new());
        let engine =
            StorageEngine::open_embedded(&durable_cfg(&dir), oracle.clone()).expect("open");
        let ts = oracle.next().as_raw();
        engine
            .columnar_insert("Trade", b"row:0001".to_vec(), b"AAPL".to_vec(), ts)
            .expect("columnar_insert");
        // Crash: drop without flush_columnar_table / persist.
    }

    {
        let oracle = Arc::new(TimestampOracle::new());
        let engine = StorageEngine::open_embedded(&durable_cfg(&dir), oracle).expect("reopen");
        let rows = engine.columnar_scan("Trade", u64::MAX).expect("scan");
        assert_eq!(
            rows,
            vec![(b"row:0001".to_vec(), b"AAPL".to_vec())],
            "journal replay must restore an un-flushed columnar row"
        );
    }
}

#[cfg(feature = "columnar")]
#[test]
fn flushed_columnar_row_not_replayed_over_newer_state() {
    // A columnar row made durable (flushed to SST) before the crash must be
    // SKIPPED on recovery (entry.ts <= tree persisted seqno), while a later
    // un-flushed overwrite IS replayed — no stale resurrection of the old value.
    let dir = TempDir::new().expect("temp dir");

    {
        let oracle = Arc::new(TimestampOracle::new());
        let engine =
            StorageEngine::open_embedded(&durable_cfg(&dir), oracle.clone()).expect("open");
        let ts1 = oracle.next().as_raw();
        engine
            .columnar_insert("Trade", b"row:0007".to_vec(), b"v1-durable".to_vec(), ts1)
            .expect("insert v1");
        engine.flush_columnar_table("Trade").expect("flush");

        let ts2 = oracle.next().as_raw();
        engine
            .columnar_insert(
                "Trade",
                b"row:0007".to_vec(),
                b"v2-journalled".to_vec(),
                ts2,
            )
            .expect("insert v2");
        // Crash: v2 lives only in the oplog.
    }

    {
        let oracle = Arc::new(TimestampOracle::new());
        let engine = StorageEngine::open_embedded(&durable_cfg(&dir), oracle).expect("reopen");
        let rows = engine.columnar_scan("Trade", u64::MAX).expect("scan");
        assert_eq!(
            rows,
            vec![(b"row:0007".to_vec(), b"v2-journalled".to_vec())],
            "recovery must skip the flushed row and replay only the un-flushed overwrite"
        );
    }
}

#[cfg(feature = "columnar")]
#[test]
fn late_finalized_columnar_row_survives_a_crash() {
    // The late-finalize order on a columnar table: the newer row reaches an
    // SST, the older-timestamped one is only in the memtable when power goes.
    // A table whose highest persisted seqno stands in for coverage skips it.
    let rig = PowerRig::new();
    {
        let (engine, oracle) = rig.open();
        let t100 = oracle.next().as_raw();
        let t200 = oracle.next().as_raw();
        engine
            .columnar_insert("Trade", b"row:0200".to_vec(), b"t200".to_vec(), t200)
            .expect("insert t200");
        engine.flush_columnar_table("Trade").expect("flush t200");
        engine
            .columnar_insert("Trade", b"row:0100".to_vec(), b"t100".to_vec(), t100)
            .expect("insert t100");
        rig.cut(engine);
    }
    let (engine, _) = rig.open();
    assert_eq!(
        engine.columnar_scan("Trade", u64::MAX).expect("scan"),
        vec![
            (b"row:0100".to_vec(), b"t100".to_vec()),
            (b"row:0200".to_vec(), b"t200".to_vec()),
        ],
        "both rows, and nothing of the table's coverage record"
    );
}

#[cfg(feature = "columnar")]
#[test]
fn a_dropped_columnar_table_is_not_resurrected_by_replay() {
    // Rows journalled for a table that was then dropped belong to that table
    // alone: replay must not recreate it, and a new table under the same name
    // starts empty rather than inheriting them.
    let rig = PowerRig::new();
    {
        let (engine, oracle) = rig.open();
        engine
            .columnar_insert(
                "Scratch",
                b"row:old".to_vec(),
                b"old".to_vec(),
                oracle.next().as_raw(),
            )
            .expect("insert");
        engine.drop_columnar_table("Scratch").expect("drop");
        engine.persist().expect("persist");
        rig.cut(engine);
    }
    {
        let (engine, _) = rig.open();
        assert!(
            engine.columnar_table_tree("Scratch").is_none(),
            "replay must not recreate a dropped table"
        );
        engine.create_columnar_table("Scratch").expect("recreate");
        rig.cut(engine);
    }
    let (engine, _) = rig.open();
    assert_eq!(
        engine.columnar_scan("Scratch", u64::MAX).expect("scan"),
        Vec::<(Vec<u8>, Vec<u8>)>::new(),
        "a table recreated under the old name starts empty"
    );
}

#[cfg(feature = "columnar")]
#[test]
fn folded_columnar_coverage_hides_its_record_and_replays_nothing_twice() {
    // After a fold the table's markers are gone and its base covers them; a
    // row committed after the fold and lost in a crash is replayed once, and
    // the scan never shows the record.
    let rig = PowerRig::new();
    {
        let (engine, oracle) = rig.open();
        for i in 0..3u8 {
            engine
                .columnar_insert(
                    "Trade",
                    vec![b'r', b'0' + i],
                    vec![i],
                    oracle.next().as_raw(),
                )
                .expect("insert");
        }
        engine.oplog_purge_expired(0, u64::MAX).expect("fold");
        engine
            .columnar_insert("Trade", b"r9".to_vec(), vec![9], oracle.next().as_raw())
            .expect("insert after fold");
        rig.cut(engine);
    }
    for _ in 0..2 {
        let (engine, _) = rig.open();
        assert_eq!(
            engine.columnar_scan("Trade", u64::MAX).expect("scan"),
            vec![
                (b"r0".to_vec(), vec![0]),
                (b"r1".to_vec(), vec![1]),
                (b"r2".to_vec(), vec![2]),
                (b"r9".to_vec(), vec![9]),
            ]
        );
        rig.cut(engine);
    }
}

/// One commit of the crash campaign: a put naming it, an edge carrying its
/// number and a +1 on a shared counter, across three trees.
fn campaign_commit(i: u64) -> Vec<Mutation> {
    vec![
        Mutation::Put {
            partition: PartitionId::Node,
            key: format!("node:00:c{i:04}").into_bytes(),
            value: b"x".to_vec(),
        },
        Mutation::Merge {
            partition: PartitionId::Adj,
            key: b"adj:C:out:1".to_vec(),
            operand: crate::engine::merge::encode_add(i),
        },
        Mutation::Merge {
            partition: PartitionId::Counter,
            key: b"counter:campaign".to_vec(),
            operand: crate::engine::merge::encode_counter_delta(1),
        },
    ]
}

const CAMPAIGN_COMMITS: u64 = 24;

/// Run the campaign workload until the first failure: late-finalized
/// timestamps, a full flush, a flush of one tree alone and a fold along the
/// way. Returns the commits acknowledged before the failure.
fn run_campaign(engine: &StorageEngine, oracle: &TimestampOracle) -> Vec<u64> {
    // Timestamps reserved up front and used in pairs out of order, so every
    // other commit finalizes behind a newer one.
    let ts: Vec<u64> = (0..CAMPAIGN_COMMITS)
        .map(|_| oracle.next().as_raw())
        .collect();
    let mut acked = Vec::new();
    for i in 0..CAMPAIGN_COMMITS {
        let slot = if i % 2 == 0 { i + 1 } else { i - 1 };
        if engine
            .commit_journaled(&campaign_commit(i), ts[slot as usize])
            .is_err()
        {
            return acked;
        }
        acked.push(i);
        let step = match i {
            7 => engine.persist(),
            11 => engine
                .tree(Partition::Adj)
                .and_then(|t| Ok(t.flush_active_memtable(0)?)),
            15 => engine.oplog_purge_expired(0, u64::MAX).map(|_| ()),
            _ => Ok(()),
        };
        if step.is_err() {
            return acked;
        }
    }
    acked
}

/// What a recovered store must hold: every acknowledged commit, and every
/// commit that is there at all exactly once in every tree it touched.
fn check_campaign(engine: &StorageEngine, acked: &[u64], label: &str) {
    let present: Vec<u64> = (0..CAMPAIGN_COMMITS)
        .filter(|i| {
            engine
                .get(Partition::Node, format!("node:00:c{i:04}").as_bytes())
                .expect("get")
                .is_some()
        })
        .collect();
    for i in acked {
        assert!(present.contains(i), "{label}: acknowledged commit {i} lost");
    }
    let counter = engine
        .get(Partition::Counter, b"counter:campaign")
        .expect("get counter")
        .map(|v| crate::engine::merge::decode_counter(&v).expect("decode"))
        .unwrap_or(0);
    assert_eq!(
        counter,
        present.len() as i64,
        "{label}: the counter must count each present commit once (present {present:?})"
    );
    let edges: Vec<u64> = engine
        .get(Partition::Adj, b"adj:C:out:1")
        .expect("get adj")
        .map(|v| {
            coordinode_core::graph::edge::PostingList::from_bytes(&v)
                .expect("posting list")
                .iter()
                .collect()
        })
        .unwrap_or_default();
    assert_eq!(
        edges, present,
        "{label}: the edges must be the present commits"
    );
}

/// One cut point: the k-th operation of `op` fails, the power goes, and the
/// store must reopen consistent, twice. Returns whether the workload reached
/// the cut (it completes untouched once `k` exceeds its operations).
fn cut_at(op: lsm_tree::fs::FaultOp, k: u64) -> bool {
    use lsm_tree::fs::{Fault, FaultRule};
    let rig = PowerRig::new();
    let (engine, oracle) = rig.open();
    rig.faults
        .arm(FaultRule::new(op, Fault::Error(lsm_tree::io::ErrorKind::Other)).skip(k));
    let acked = run_campaign(&engine, &oracle);
    let reached = (acked.len() as u64) < CAMPAIGN_COMMITS;
    rig.cut(engine);
    for round in 0..2 {
        let (engine, _) = rig.open();
        check_campaign(&engine, &acked, &format!("{op:?} cut at {k}, open {round}"));
        rig.cut(engine);
    }
    reached
}

/// A power cut at every operation of one kind the workload performs. The
/// cut points are independent stores, so they run on parallel threads in
/// windows until a whole window completes the workload untouched. Returns
/// how many cut points the workload reached.
fn cut_at_every(op: lsm_tree::fs::FaultOp) -> u64 {
    let window = std::thread::available_parallelism().map_or(4, |n| n.get() as u64);
    let mut reached = 0;
    let mut start = 0;
    loop {
        let hits: Vec<bool> = std::thread::scope(|s| {
            let handles: Vec<_> = (start..start + window)
                .map(|k| s.spawn(move || cut_at(op, k)))
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("cut point thread"))
                .collect()
        });
        let window_hits = hits.iter().filter(|&&hit| hit).count() as u64;
        reached += window_hits;
        if window_hits < window {
            return reached;
        }
        start += window;
    }
}

#[test]
fn a_power_cut_at_any_sync_leaves_every_commit_once() {
    // Every journal append and every table/manifest publish ends in a full
    // sync, so this walks the whole durability order of the workload.
    assert!(cut_at_every(lsm_tree::fs::FaultOp::SyncAll) > 0);
}

#[test]
fn a_power_cut_at_any_write_leaves_every_commit_once() {
    // A failed write leaves a torn record or table behind; recovery must
    // read past it exactly as past a cut at a sync.
    assert!(cut_at_every(lsm_tree::fs::FaultOp::Write) > 0);
}

#[test]
fn in_memory_engine_has_no_journal() {
    // A fully volatile (in-memory) config has no oplog-eligible endpoint, so
    // open_with_oracle / open_embedded must not create a journal.
    let config = StorageConfig::with_endpoints_no_persistence(vec![EndpointConfig::new(
        "mem",
        std::path::Path::new("/coordinode-test-in-memory"),
        Media::Ram,
        Durability::Volatile,
        Tier::Memory,
    )])
    .with_fs(Arc::new(lsm_tree::fs::MemFs::new()) as Arc<dyn lsm_tree::fs::Fs>);
    let oracle = Arc::new(TimestampOracle::new());
    let engine = StorageEngine::open_embedded(&config, oracle).expect("open in-memory");
    assert!(
        !engine.has_journal(),
        "a fully volatile config must not create a disk journal"
    );
}

mod derived {
    use coordinode_core::graph::node::NodeRecord;
    use coordinode_core::graph::types::Value;
    use coordinode_core::index::derive::{IndexInterpretation, KEY_CODEC, PropertyRef};
    use coordinode_core::index::encoding::{encode_index_key, encode_tuple};
    use coordinode_core::txn::proposal::{DerivedIndexWork, DerivedSource, IndexBinding};

    use super::*;

    fn email_index() -> IndexInterpretation {
        IndexInterpretation {
            codec: KEY_CODEC,
            name: "user_email".into(),
            unique: false,
            sparse: false,
            properties: vec![PropertyRef {
                field: Some(1),
                name: "email".into(),
            }],
            filter: None,
        }
    }

    fn entry_key(email: &str, node_id: u64) -> Vec<u8> {
        let tuple = encode_tuple(&[Value::String(email.into())]).expect("tuple");
        encode_index_key("user_email", &tuple, node_id)
    }

    /// A node record put for `email`, then the DERIVED work moving the node's
    /// entry from `old` to what that record holds.
    fn unit(node_id: u64, email: &str, old: Option<&str>) -> Vec<Mutation> {
        let mut record = NodeRecord::new("User");
        record.set(1, Value::String(email.into()));
        vec![
            Mutation::Put {
                partition: PartitionId::Node,
                key: format!("node:00:{node_id:04}").into_bytes(),
                value: record.to_msgpack().expect("record"),
            },
            Mutation::Derive(DerivedIndexWork {
                binding: IndexBinding {
                    epoch: 1,
                    interpretation: email_index(),
                },
                node_id,
                old: old.map(|o| vec![Value::String(o.into())]),
                new: DerivedSource::UnitRecord(0),
            }),
        ]
    }

    fn has(engine: &StorageEngine, key: &[u8]) -> bool {
        engine.get(Partition::Idx, key).expect("get").is_some()
    }

    /// The entries a unit's DERIVED work derives land with the unit: the
    /// new value's entry is there, the old value's is gone.
    #[test]
    fn derived_entries_land_with_their_unit() {
        let dir = TempDir::new().expect("temp dir");
        let oracle = Arc::new(TimestampOracle::new());
        let engine =
            StorageEngine::open_embedded(&durable_cfg(&dir), oracle.clone()).expect("open");
        write_batch(&engine, &oracle, &unit(1, "a@x", None));
        assert!(has(&engine, &entry_key("a@x", 1)));
        write_batch(&engine, &oracle, &unit(1, "b@x", Some("a@x")));
        assert!(has(&engine, &entry_key("b@x", 1)));
        assert!(!has(&engine, &entry_key("a@x", 1)));
    }

    /// The node partition reaches disk, the index partition does not, and
    /// power is lost. Recovery skips the entry for the node partition, which
    /// holds it, yet still derives the index entry from the journalled node
    /// record: data already on disk must not cost the index its entry.
    #[test]
    fn an_index_behind_its_data_is_derived_again_from_the_journal() {
        let rig = PowerRig::new();
        {
            let (engine, oracle) = rig.open();
            write_batch(&engine, &oracle, &unit(7, "late@x", None));
            engine
                .tree(Partition::Node)
                .expect("node tree")
                .flush_active_memtable(0)
                .expect("flush the node partition alone");
            rig.cut(engine);
        }
        for attempt in 0..2 {
            let (engine, _) = rig.open();
            assert!(
                engine
                    .get(Partition::Node, b"node:00:0007")
                    .expect("get")
                    .is_some(),
                "the node record was on disk"
            );
            assert!(
                has(&engine, &entry_key("late@x", 7)),
                "recovery {attempt} must derive the entry of a node already on disk"
            );
            rig.cut(engine);
        }
    }
}
