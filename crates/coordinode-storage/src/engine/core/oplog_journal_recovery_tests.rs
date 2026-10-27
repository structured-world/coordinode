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
        let mut config = durable_cfg(&self.dir).with_fs(Arc::clone(&self.fs));
        // Volatile commits stay unsynced until a test syncs them, so a cut
        // finds them where the test left them.
        config.volatile_flush_interval_ms = 60_000;
        config
    }

    fn open(&self) -> (StorageEngine, Arc<TimestampOracle>) {
        self.open_with(&self.config())
    }

    fn open_with(&self, config: &StorageConfig) -> (StorageEngine, Arc<TimestampOracle>) {
        let oracle = Arc::new(TimestampOracle::new());
        let engine = StorageEngine::open_embedded(config, oracle.clone()).expect("open");
        (engine, oracle)
    }

    /// Open the store as a group member opens it: its log is a consensus
    /// log, so its own records go to the member-local journal.
    fn open_group(&self, config: &StorageConfig) -> StorageResult<StorageEngine> {
        StorageEngine::open_with_oracle(config, Arc::new(TimestampOracle::new()))
    }

    /// Lose power: nothing written from now on is durable, the engine goes
    /// away, and every file falls back to what was fsynced before the cut,
    /// the journal's unsynced tail included.
    fn cut(&self, engine: StorageEngine) {
        let journal = engine.journal_crash_point(CrashKind::PowerCut);
        self.stop(engine);
        cut_journal(journal);
        self.crash.crash();
    }

    /// Kill the process: what it handed to the operating system stays, its
    /// own buffers (the journal's among them) are lost.
    fn kill(&self, engine: StorageEngine) {
        let journal = engine.journal_crash_point(CrashKind::Kill);
        self.stop(engine);
        cut_journal(journal);
    }

    /// Take `engine` away with every table write refused, so the flush its
    /// drop attempts never lands.
    fn stop(&self, engine: StorageEngine) {
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
    }
}

/// Cut the journal segment a crash leaves short back to `point`, after the
/// engine that wrote it is gone (its drop syncs and seals the segment, which
/// the crash never let happen).
fn cut_journal(point: Option<(std::path::PathBuf, u64)>) {
    if let Some((path, len)) = point {
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .and_then(|file| file.set_len(len))
            .expect("cut the journal segment");
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

/// An index entry lost in a power cut together with its generation's first
/// binding is replayed from the journal into a fresh installation of that
/// generation, beside an earlier generation whose installation was already on
/// disk; both read back by their logical keys, also after another reopen,
/// and nothing is stored under a generation's own number.
#[test]
fn an_index_entry_and_its_new_binding_lost_to_a_crash_are_replayed() {
    use coordinode_core::index::encoding::encode_unique_entry_key;
    use coordinode_core::index::identity::GenerationId;
    let rig = PowerRig::new();
    let early = encode_unique_entry_key(GenerationId::from_raw(40), b"\x40a\x00");
    let late = encode_unique_entry_key(GenerationId::from_raw(41), b"\x40b\x00");
    let put = |key: &[u8], value: &[u8]| Mutation::Put {
        partition: PartitionId::Idx,
        key: key.to_vec(),
        value: value.to_vec(),
    };
    {
        let (engine, oracle) = rig.open();
        write_batch_at(&engine, &[put(&early, b"n1")], oracle.next().as_raw());
        engine.persist().expect("persist the first generation");
        write_batch_at(&engine, &[put(&late, b"n2")], oracle.next().as_raw());
        // Power loss: the second generation's binding and entry were only in
        // the memtable; its commit is in the journal.
        rig.cut(engine);
    }
    for round in ["replayed", "reopened"] {
        let (engine, _) = rig.open();
        for (key, value) in [(&early, b"n1"), (&late, b"n2")] {
            assert_eq!(
                engine.get(Partition::Idx, key).expect("get").as_deref(),
                Some(value.as_slice()),
                "{round}"
            );
        }
        let installations = engine.installations.current();
        let (Some(a), Some(b)) = (
            installations.installation(40),
            installations.installation(41),
        ) else {
            panic!("{round}: both generations bound");
        };
        assert_ne!(a, b, "{round}");
        engine.persist().expect("persist");
    }
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

/// Applies a commit's proposal the way the embedded pipeline does: journaled,
/// then applied at its timestamp.
struct JournaledPipeline<'e>(&'e StorageEngine);

impl coordinode_core::txn::proposal::ProposalPipeline for JournaledPipeline<'_> {
    fn propose_and_wait(
        &self,
        proposal: &coordinode_core::txn::proposal::RaftProposal,
    ) -> Result<
        coordinode_core::txn::proposal::ProposalOutcome,
        coordinode_core::txn::proposal::ProposalError,
    > {
        self.0
            .commit_journaled(&proposal.mutations, proposal.commit_ts.as_raw())
            .map(|()| coordinode_core::txn::proposal::ProposalOutcome::local())
            .map_err(|e| coordinode_core::txn::proposal::ProposalError::Storage(e.to_string()))
    }
}

#[test]
fn kept_cardinality_counts_are_replayed_once_after_a_power_cut() {
    // Committed edges whose count changes reached only the journal come back
    // with their counts, once: a count replayed twice, or not at all, would
    // decide every later bound over the node from a wrong number.
    use crate::engine::transaction::{CommitContext, CommitError, Transaction};
    use coordinode_core::graph::cardinality::{
        CardinalityBound, CardinalityDescriptor, CardinalityMeasure, Direction,
        encode_cardinality_profile_key,
    };
    use coordinode_core::graph::edge::{encode_adj_key_forward, encode_adj_key_reverse};
    use coordinode_core::graph::node::NodeId;
    use coordinode_core::schema::definition::{
        EdgeTypeSchema, encode_edge_type_current_revision_key, encode_edge_type_schema_key,
    };
    use coordinode_core::txn::invariant::{Claim, ClaimPredicate, ClaimScope, CountTrend};
    use coordinode_core::txn::timestamp::Timestamp;
    use coordinode_core::txn::write_concern::WriteConcern;

    let d = CardinalityDescriptor {
        edge_type: "OWNS".into(),
        direction: Direction::Outgoing,
        measure: CardinalityMeasure::DistinctNeighbours,
        bound: CardinalityBound::AtMostOne,
        schema_generation: 1,
    };
    let wc = WriteConcern::default();
    let commit = |engine: &StorageEngine, txn: &mut Transaction<'_>| -> Result<(), CommitError> {
        let pipeline = JournaledPipeline(engine);
        let ctx = CommitContext {
            write_concern: &wc,
            pipeline: Some(&pipeline),
            id_gen: None,
        };
        txn.commit(&ctx).map(|_| ())
    };
    /// A transaction deciding node 1's AT MOST ONE.
    fn begin<'a>(engine: &'a StorageEngine, oracle: &'a TimestampOracle) -> Transaction<'a> {
        let snap = engine.snapshot();
        let mut txn = Transaction::new(engine, Some(oracle), Timestamp::from_raw(snap), Some(snap));
        txn.claim(Claim::new(
            ClaimScope::Incident {
                node: NodeId::from_raw(1),
                edge_type: "OWNS".to_string(),
                direction: Direction::Outgoing,
            },
            ClaimPredicate::CardinalityBound {
                measure: CardinalityMeasure::DistinctNeighbours,
                bound: CardinalityBound::AtMostOne,
                trend: CountTrend::Grows,
            },
            0,
        ));
        txn
    }
    let add = |txn: &mut Transaction<'_>, target: u64| {
        txn.merge_adj_add(&encode_adj_key_forward("OWNS", NodeId::from_raw(1)), target);
        txn.merge_adj_add(&encode_adj_key_reverse("OWNS", NodeId::from_raw(target)), 1);
    };
    let kept = |engine: &StorageEngine| {
        engine
            .get(Partition::Counter, &d.counter_key(NodeId::from_raw(1)))
            .expect("read count")
            .map(|bytes| crate::engine::merge::decode_counter(&bytes).expect("decode"))
    };

    let rig = PowerRig::new();
    {
        let (engine, oracle) = rig.open();
        let mut schema = EdgeTypeSchema::new("OWNS");
        schema.declare_cardinality(d.clone()).expect("declare");
        let mut definition = Transaction::new(
            &engine,
            Some(&oracle),
            Timestamp::from_raw(engine.snapshot()),
            Some(engine.snapshot()),
        );
        for (key, value) in [
            (
                encode_edge_type_schema_key("OWNS", 1),
                schema.to_msgpack().expect("encode"),
            ),
            (
                encode_edge_type_current_revision_key("OWNS"),
                1u64.to_be_bytes().to_vec(),
            ),
            (
                encode_cardinality_profile_key("OWNS"),
                schema
                    .cardinality_profile()
                    .expect("profile")
                    .to_msgpack()
                    .expect("encode"),
            ),
        ] {
            definition
                .put(Partition::Schema, &key, &value)
                .expect("stage");
        }
        commit(&engine, &mut definition).expect("define");
        let snap = engine.snapshot();
        let mut cover = Transaction::new(
            &engine,
            Some(&oracle),
            Timestamp::from_raw(snap),
            Some(snap),
        );
        crate::engine::cardinality::rebuild(&mut cover, &d).expect("rebuild");
        commit(&engine, &mut cover).expect("cover");
        engine.persist().expect("persist the definition");

        // The edge and its count change reach only the journal.
        let mut first = begin(&engine, &oracle);
        add(&mut first, 2);
        commit(&engine, &mut first).expect("first edge");
        assert_eq!(kept(&engine), Some(1));
        rig.cut(engine);
    }

    for _ in 0..2 {
        let (engine, oracle) = rig.open();
        assert_eq!(kept(&engine), Some(1), "the count is back, once");
        let mut second = begin(&engine, &oracle);
        add(&mut second, 3);
        assert!(
            matches!(
                commit(&engine, &mut second),
                Err(CommitError::InvariantRefused { .. })
            ),
            "the recovered count still holds the bound"
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

/// Every partition tree of `engine` holds no table: nothing it applied has
/// been written to disk outside the journal.
fn holds_no_table(engine: &StorageEngine) -> bool {
    use lsm_tree::AbstractTree;
    Partition::all()
        .iter()
        .all(|&part| engine.tree(part).expect("tree").table_count() == 0)
}

/// Opening a fresh store writes no table: its coverage base waits in the
/// trees' memtables for their first flush, and a store whose trees hold no
/// table has nothing on disk the journal could disagree with.
#[test]
fn a_fresh_store_opens_without_writing_a_table() {
    let rig = PowerRig::new();
    let (engine, _) = rig.open();
    assert!(
        holds_no_table(&engine),
        "opening a fresh store wrote tables"
    );
    rig.cut(engine);
}

/// A power cut before a fresh store's first table flush keeps every
/// journaled commit: with no table and no coverage record anywhere, the
/// whole journal replays exactly.
#[test]
fn a_power_cut_before_a_fresh_stores_first_flush_keeps_its_commits() {
    let rig = PowerRig::new();
    let (engine, oracle) = rig.open();
    commit_at(&engine, &oracle, &two_tree_commit("a", 1), Journal::Journal);
    rig.cut(engine);
    let (engine, oracle) = rig.open();
    assert_eq!(two_tree_state(&engine, "a"), (true, vec![1]));
    commit_at(&engine, &oracle, &two_tree_commit("b", 2), Journal::Journal);
    rig.cut(engine);
    let (engine, _) = rig.open();
    // Both commits add to the one adjacency list, each exactly once.
    assert_eq!(two_tree_state(&engine, "a"), (true, vec![1, 2]));
    assert_eq!(two_tree_state(&engine, "b"), (true, vec![1, 2]));
    rig.cut(engine);
}

/// Entries replayed at open are not flushed at open: the journal still
/// holds them, so a second power cut replays them again, each once.
#[test]
fn entries_replayed_at_open_stay_in_the_journal() {
    let rig = PowerRig::new();
    let (engine, oracle) = rig.open();
    commit_at(&engine, &oracle, &two_tree_commit("a", 1), Journal::Journal);
    rig.cut(engine);
    let (engine, _) = rig.open();
    assert!(holds_no_table(&engine), "the replay was flushed at open");
    rig.cut(engine);
    let (engine, _) = rig.open();
    assert_eq!(two_tree_state(&engine, "a"), (true, vec![1]));
    rig.cut(engine);
}

/// Writes that land in two partitions make a persist write tables for those
/// two trees only: coverage bookkeeping for the journal entries is no reason
/// to write a table for every other partition.
#[test]
fn a_persist_writes_tables_only_for_trees_that_took_writes() {
    use lsm_tree::AbstractTree;
    let rig = PowerRig::new();
    let (engine, oracle) = rig.open();
    let tables = |engine: &StorageEngine| -> Vec<usize> {
        Partition::all()
            .iter()
            .map(|&part| engine.tree(part).expect("tree").table_count())
            .collect()
    };
    engine.persist().expect("persist");
    let before = tables(&engine);
    for i in 0..300u64 {
        commit_at(
            &engine,
            &oracle,
            &two_tree_commit(&format!("k{i}"), i),
            Journal::Journal,
        );
    }
    engine.persist().expect("persist");
    let after = tables(&engine);
    let written = before.iter().zip(&after).filter(|(b, a)| a > b).count();
    assert_eq!(
        written,
        2,
        "trees that got a table: {written} of {}",
        after.len()
    );
    rig.cut(engine);
}

/// A routine coverage fold writes only the trees its entries reached, so the
/// persist after it writes tables for those trees alone; the fold before a
/// purge brings every other tree's base up, and the store reopens over the
/// purged journal with every commit once.
#[test]
fn a_routine_fold_writes_only_the_trees_its_entries_reached() {
    use lsm_tree::AbstractTree;
    let rig = PowerRig::new();
    let mut config = rig.config();
    config.oplog_segment_max_entries = 512;
    config.oplog_retention_secs = 1;
    let (engine, oracle) = rig.open_with(&config);
    let tables = |engine: &StorageEngine| -> Vec<usize> {
        Partition::all()
            .iter()
            .map(|&part| engine.tree(part).expect("tree").table_count())
            .collect()
    };
    engine.persist().expect("persist");
    let before = tables(&engine);
    let commits = crate::engine::coverage::FOLD_EVERY + 100;
    for i in 0..commits {
        commit_at(
            &engine,
            &oracle,
            &two_tree_commit(&format!("k{i}"), i),
            Journal::Cache,
        );
    }
    engine.persist().expect("persist");
    let after = tables(&engine);
    let written = before.iter().zip(&after).filter(|(b, a)| a > b).count();
    assert_eq!(written, 2, "a fold wrote tables for untouched trees");

    let purged = engine
        .oplog_purge_expired(4_000_000_000, u64::MAX)
        .expect("purge");
    assert!(purged > 0, "the journal kept every segment");
    rig.cut(engine);
    let (engine, _) = rig.open_with(&config);
    assert!(two_tree_state(&engine, "k0").0, "the first commit lost");
    assert!(
        two_tree_state(&engine, &format!("k{}", commits - 1)).0,
        "the last commit lost"
    );
    rig.cut(engine);
}

/// A tree lost whole (its folder gone, recreated empty on open) while the
/// journal still holds every entry from the first one is rebuilt by
/// replaying all of them into it; the tree that kept its tables and record
/// takes none of them a second time.
#[test]
fn a_lost_tree_is_rebuilt_from_a_complete_journal() {
    let rig = PowerRig::new();
    let (engine, oracle) = rig.open();
    commit_at(&engine, &oracle, &two_tree_commit("a", 1), Journal::Journal);
    engine.persist().expect("persist");
    commit_at(&engine, &oracle, &two_tree_commit("b", 2), Journal::Journal);
    let adj = rig.config().endpoints[0].path.join(Partition::Adj.name());
    rig.cut(engine);
    std::fs::remove_dir_all(&adj).expect("remove the adjacency tree");
    let (engine, _) = rig.open();
    assert_eq!(two_tree_state(&engine, "a"), (true, vec![1, 2]));
    assert_eq!(two_tree_state(&engine, "b"), (true, vec![1, 2]));
    rig.cut(engine);
}

/// A journal segment lost from the middle takes entries no tree covers: the
/// open refuses rather than replay the entries around the hole as if it were
/// not there.
#[test]
fn a_segment_lost_from_the_middle_of_the_journal_is_refused() {
    let rig = PowerRig::new();
    let mut config = rig.config();
    config.oplog_segment_max_entries = 2;
    let (engine, oracle) = rig.open_with(&config);
    for i in 0..6u64 {
        commit_at(
            &engine,
            &oracle,
            &two_tree_commit(&format!("k{i}"), i),
            Journal::Journal,
        );
    }
    rig.cut(engine);
    let journal = config.endpoints[0].path.join("oplog").join("0");
    let mut segments: Vec<std::path::PathBuf> = std::fs::read_dir(&journal)
        .expect("journal dir")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("oplog-") && n.ends_with(".bin"))
        })
        .collect();
    segments.sort();
    assert!(segments.len() >= 3, "segments: {segments:?}");
    std::fs::remove_file(&segments[1]).expect("remove a middle segment");
    let oracle = Arc::new(TimestampOracle::new());
    let refused = StorageEngine::open_embedded(&config, oracle);
    assert!(
        matches!(refused, Err(StorageError::CoverageUnprovable { .. })),
        "a hole in the journal must be refused, got {:?}",
        refused.err()
    );
}

/// A tree lost whole after the journal was purged cannot be rebuilt from
/// what the journal still holds: the open refuses rather than replay part of
/// its history into an empty tree.
#[test]
fn a_lost_tree_behind_a_purged_journal_is_refused() {
    let rig = PowerRig::new();
    let mut config = rig.config();
    config.oplog_segment_max_entries = 2;
    config.oplog_retention_secs = 1;
    let (engine, oracle) = rig.open_with(&config);
    for i in 0..6u64 {
        commit_at(
            &engine,
            &oracle,
            &two_tree_commit(&format!("k{i}"), i),
            Journal::Journal,
        );
    }
    let purged = engine
        .oplog_purge_expired(4_000_000_000, u64::MAX)
        .expect("purge");
    assert!(purged > 0, "the journal kept every segment");
    let adj = config.endpoints[0].path.join(Partition::Adj.name());
    rig.cut(engine);
    std::fs::remove_dir_all(&adj).expect("remove the adjacency tree");
    let oracle = Arc::new(TimestampOracle::new());
    let refused = StorageEngine::open_embedded(&config, oracle);
    assert!(
        matches!(refused, Err(StorageError::CoverageUnprovable { .. })),
        "a lost tree behind a purged journal must be refused, got {:?}",
        refused.err()
    );
}

/// A group member's own record, put into the Raft tree.
fn member_put(key: &[u8], value: &[u8]) -> LocalWrite {
    LocalWrite::Put {
        partition: Partition::Raft,
        key: key.to_vec(),
        value: value.to_vec(),
    }
}

/// The member-local journal's segment directory under `config`.
fn local_journal_dir(config: &StorageConfig) -> std::path::PathBuf {
    crate::engine::local_journal::LocalJournal::dir(&config.endpoints[0].path)
}

/// A group member's record needed before use (a vote) costs one sync of the
/// member-local journal and no table, and a power cut right after keeps it.
#[test]
fn a_member_record_before_use_costs_one_journal_sync_and_survives_a_power_cut() {
    let rig = PowerRig::new();
    let config = rig.config();
    let engine = rig.open_group(&config).expect("open");
    let syncs = engine.journal_syncs().expect("a member-local journal");
    let flushes = engine.memtable_flushes();
    engine
        .commit_local(vec![member_put(b"vote", b"v1")], LocalSync::BeforeUse)
        .expect("commit");
    assert_eq!(engine.journal_syncs(), Some(syncs + 1), "journal syncs");
    assert_eq!(engine.memtable_flushes(), flushes, "tables written");
    rig.cut(engine);
    let engine = rig.open_group(&config).expect("reopen");
    assert_eq!(
        engine
            .get(Partition::Raft, b"vote")
            .expect("get")
            .as_deref(),
        Some(b"v1".as_slice())
    );
    rig.cut(engine);
}

/// A deferred record a power cut takes is gone and the store reopens; the
/// records written after it reuse no index and survive the next cut.
#[test]
fn a_deferred_member_record_lost_to_a_power_cut_leaves_the_store_openable() {
    let rig = PowerRig::new();
    let config = rig.config();
    let engine = rig.open_group(&config).expect("open");
    engine
        .commit_local(vec![member_put(b"mark", b"m")], LocalSync::Deferred)
        .expect("commit");
    rig.cut(engine);
    let engine = rig.open_group(&config).expect("reopen");
    assert_eq!(engine.get(Partition::Raft, b"mark").expect("get"), None);
    engine
        .commit_local(vec![member_put(b"vote", b"v2")], LocalSync::BeforeUse)
        .expect("commit");
    rig.cut(engine);
    let engine = rig.open_group(&config).expect("reopen");
    assert_eq!(
        engine
            .get(Partition::Raft, b"vote")
            .expect("get")
            .as_deref(),
        Some(b"v2".as_slice())
    );
    rig.cut(engine);
}

/// A flush of a tree holding a deferred record syncs the member-local journal
/// first: a table never holds an entry the journal lost, which would leave
/// the tree covering an index no journal holds and refuse every later open.
#[test]
fn a_tree_flush_syncs_the_member_local_journal_first() {
    let rig = PowerRig::new();
    let config = rig.config();
    let engine = rig.open_group(&config).expect("open");
    // A durable record first, so the deferred one sits in the middle of the
    // journal, where an index lost from it is a hole every tree must cover.
    engine
        .commit_local(vec![member_put(b"z", b"0")], LocalSync::BeforeUse)
        .expect("commit");
    engine
        .commit_local(vec![member_put(b"a", b"1")], LocalSync::Deferred)
        .expect("commit");
    engine.persist_partition(Partition::Raft).expect("flush");
    rig.cut(engine);
    let engine = rig.open_group(&config).expect("reopen");
    assert_eq!(
        engine.get(Partition::Raft, b"a").expect("get").as_deref(),
        Some(b"1".as_slice())
    );
    engine
        .commit_local(vec![member_put(b"b", b"2")], LocalSync::BeforeUse)
        .expect("commit");
    rig.cut(engine);
    let engine = rig
        .open_group(&config)
        .expect("reopen after a later record");
    assert_eq!(
        engine.get(Partition::Raft, b"b").expect("get").as_deref(),
        Some(b"2".as_slice())
    );
    rig.cut(engine);
}

/// Compaction folds and flushes only the trees holding member records, drops
/// the journal segments they cover, and the store reopens over the shortened
/// journal with every record.
#[test]
fn a_compaction_shortens_the_member_local_journal_and_keeps_every_record() {
    use lsm_tree::AbstractTree;
    let rig = PowerRig::new();
    let mut config = rig.config();
    config.oplog_segment_max_entries = 256;
    let engine = rig.open_group(&config).expect("open");
    let tables = |engine: &StorageEngine| -> Vec<usize> {
        Partition::all()
            .iter()
            .map(|&part| engine.tree(part).expect("tree").table_count())
            .collect()
    };
    let before = tables(&engine);
    let records = 4096 + 10;
    for i in 0..records {
        let key = format!("r{i}");
        engine
            .commit_local(vec![member_put(key.as_bytes(), b"x")], LocalSync::Deferred)
            .expect("commit");
    }
    let after = tables(&engine);
    let written = before.iter().zip(&after).filter(|(b, a)| a > b).count();
    assert_eq!(
        written, 1,
        "compaction wrote tables for trees without records"
    );
    let segments = std::fs::read_dir(local_journal_dir(&config))
        .expect("journal dir")
        .filter_map(Result::ok)
        .count();
    assert!(segments <= 2, "the compaction left {segments} segments");
    engine
        .commit_local(vec![member_put(b"last", b"y")], LocalSync::BeforeUse)
        .expect("commit");
    rig.cut(engine);
    let engine = rig.open_group(&config).expect("reopen");
    for key in [
        "r0".to_owned(),
        format!("r{}", records - 1),
        "last".to_owned(),
    ] {
        assert!(
            engine
                .get(Partition::Raft, key.as_bytes())
                .expect("get")
                .is_some(),
            "{key} lost"
        );
    }
    rig.cut(engine);
}

/// Files under `root`, at any depth.
fn files_under(root: &std::path::Path) -> usize {
    std::fs::read_dir(root).map_or(0, |entries| {
        entries
            .filter_map(Result::ok)
            .map(|e| {
                let path = e.path();
                if path.is_dir() { files_under(&path) } else { 1 }
            })
            .sum()
    })
}

/// With the journal on one device and the tables on another, a table flush
/// still syncs the journal first and a power cut loses only what neither
/// device made durable: each device's own sync is what counts, never the
/// other's.
#[test]
fn a_journal_and_tables_on_different_devices_recover_together() {
    let rig = PowerRig::new();
    let journal_dir = rig.dir.path().join("journal");
    let tables_dir = rig.dir.path().join("fast");
    let mut config = StorageConfig::with_endpoints(vec![
        EndpointConfig::new(
            "journal",
            &journal_dir,
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        ),
        EndpointConfig::new(
            "fast",
            &tables_dir,
            Media::Nvme,
            Durability::Durable,
            Tier::Hot,
        ),
    ])
    .with_fs(Arc::clone(&rig.fs));
    config.volatile_flush_interval_ms = 60_000;
    let (engine, oracle) = rig.open_with(&config);
    commit_at(&engine, &oracle, &two_tree_commit("a", 1), Journal::Journal);
    commit_at(&engine, &oracle, &two_tree_commit("b", 2), Journal::Cache);
    // Level-0 tables go to the hot endpoint, the journal to the first one.
    let before = files_under(&tables_dir);
    engine.persist_partition(Partition::Node).expect("flush");
    engine.persist_partition(Partition::Adj).expect("flush");
    assert!(
        files_under(&tables_dir) > before,
        "the flush wrote no table on the second device"
    );
    commit_at(&engine, &oracle, &two_tree_commit("c", 3), Journal::Cache);
    assert!(
        journal_segments(&journal_dir) > 0 && journal_segments(&tables_dir) == 0,
        "the journal is not on its own device"
    );
    // Any sync of the journal makes `c` durable too (a background table
    // flush syncs it first), so whether `c` may survive is read from the
    // durable view the cut comes down on, not assumed.
    let now = engine.current_seqno();
    let c_synced = engine.durable_view(now) == now;
    rig.cut(engine);
    let (engine, _) = rig.open_with(&config);
    let edges = if c_synced { vec![1, 2, 3] } else { vec![1, 2] };
    assert_eq!(two_tree_state(&engine, "a"), (true, edges.clone()));
    assert!(
        two_tree_state(&engine, "b").0,
        "the flushed volatile commit was lost"
    );
    // Whole or not at all, and only if a device synced it.
    assert_eq!(
        two_tree_state(&engine, "c"),
        (c_synced, edges),
        "c synced before the cut: {c_synced}"
    );
    rig.cut(engine);
}

/// A node's own record in the journal of a store that has one is replayed
/// after a crash like any entry, but subscribers that follow the store's
/// commits never see it: it is this node's recovery state, not the data.
#[test]
fn a_node_record_in_the_store_journal_is_recovered_but_never_published() {
    let rig = PowerRig::new();
    let (engine, oracle) = rig.open();
    let sub = engine.subscribe_applied(Partition::Node, 16);
    engine
        .commit_local(
            vec![LocalWrite::Put {
                partition: Partition::Node,
                key: b"node:00:own".to_vec(),
                value: b"o".to_vec(),
            }],
            LocalSync::BeforeUse,
        )
        .expect("commit the node's record");
    commit_at(
        &engine,
        &oracle,
        &two_tree_commit("shared", 1),
        Journal::Journal,
    );
    let mut published = Vec::new();
    while let Some(event) = sub.try_next() {
        if let crate::engine::applied::AppliedEvent::Keys { keys, .. } = event {
            published.extend(keys.iter().map(|k| k.to_vec()));
        }
    }
    assert!(
        published.contains(&b"node:00:shared".to_vec()),
        "the shared commit was not published: {published:?}"
    );
    assert!(
        !published.contains(&b"node:00:own".to_vec()),
        "the node's own record reached a subscriber"
    );
    rig.cut(engine);
    let (engine, _) = rig.open();
    assert_eq!(
        engine
            .get(Partition::Node, b"node:00:own")
            .expect("get")
            .as_deref(),
        Some(b"o".as_slice()),
        "the node's record was lost to the crash"
    );
    rig.cut(engine);
}

/// A crash inside a compaction, after the trees holding member records were
/// flushed but before (or while) the covered segments were removed, leaves
/// the store openable with every record: the segments still on disk replay
/// over trees that already hold them, and the removed ones were covered.
#[test]
fn a_crash_inside_a_member_local_compaction_keeps_every_record() {
    // 0: crash before the first removal; 1: after the oldest segment went.
    for removed_before_crash in [0usize, 1] {
        let rig = PowerRig::new();
        let mut config = rig.config();
        config.oplog_segment_max_entries = 256;
        let dir = local_journal_dir(&config);
        let engine = rig.open_group(&config).expect("open");
        // Below the count that compacts on its own; the last syncs them all.
        let records = 2000;
        for i in 0..records {
            let sync = if i + 1 == records {
                LocalSync::BeforeUse
            } else {
                LocalSync::Deferred
            };
            engine
                .commit_local(vec![member_put(format!("r{i}").as_bytes(), b"x")], sync)
                .expect("commit");
        }
        // The journal as the compaction finds it: every record synced.
        let mut saved: Vec<(std::path::PathBuf, Vec<u8>)> = std::fs::read_dir(&dir)
            .expect("journal dir")
            .filter_map(Result::ok)
            .map(|e| {
                let path = e.path();
                let bytes = std::fs::read(&path).expect("read segment");
                (path, bytes)
            })
            .collect();
        saved.sort();
        assert!(
            saved.len() > 2,
            "segments before compaction: {}",
            saved.len()
        );
        engine
            .local_journal
            .as_ref()
            .expect("a member-local journal")
            .compact(engine.coordinator.trees(), engine.next_seqno(), |tree| {
                engine.flush_memtables(tree)
            })
            .expect("compact");
        assert!(
            saved.iter().all(|(path, _)| !path.exists()),
            "the compaction left a covered segment"
        );
        rig.cut(engine);
        // Put back the segments the compaction removed, except the ones a
        // crash part-way through the removal would already have taken.
        for (path, bytes) in saved.iter().skip(removed_before_crash) {
            if !path.exists() {
                std::fs::write(path, bytes).expect("restore segment");
            }
        }
        let engine = rig
            .open_group(&config)
            .unwrap_or_else(|e| panic!("reopen after {removed_before_crash} removals: {e:?}"));
        for key in ["r0".to_owned(), format!("r{}", records - 1)] {
            assert!(
                engine
                    .get(Partition::Raft, key.as_bytes())
                    .expect("get")
                    .is_some(),
                "{key} lost after {removed_before_crash} removals"
            );
        }
        engine
            .commit_local(vec![member_put(b"after", b"a")], LocalSync::BeforeUse)
            .expect("commit after the reopen");
        rig.cut(engine);
        let engine = rig.open_group(&config).expect("reopen again");
        assert!(
            engine
                .get(Partition::Raft, b"after")
                .expect("get")
                .is_some()
        );
        rig.cut(engine);
    }
}

/// A segment lost from the middle of the member-local journal takes records
/// no tree covers: the open refuses rather than replay around the hole.
#[test]
fn a_segment_lost_from_the_middle_of_the_member_local_journal_is_refused() {
    let rig = PowerRig::new();
    let mut config = rig.config();
    config.oplog_segment_max_entries = 2;
    let engine = rig.open_group(&config).expect("open");
    for i in 0..6u8 {
        engine
            .commit_local(vec![member_put(&[b'k', i], b"x")], LocalSync::BeforeUse)
            .expect("commit");
    }
    rig.cut(engine);
    let mut segments: Vec<std::path::PathBuf> = std::fs::read_dir(local_journal_dir(&config))
        .expect("journal dir")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .collect();
    segments.sort();
    assert!(segments.len() >= 3, "segments: {segments:?}");
    std::fs::remove_file(&segments[1]).expect("remove a middle segment");
    let refused = rig.open_group(&config);
    assert!(
        matches!(refused, Err(StorageError::CoverageUnprovable { .. })),
        "a hole in the member-local journal must be refused, got {:?}",
        refused.err()
    );
}

/// A write of the second copy of the consensus records whose sync failed is
/// not taken as done: the bytes may read back and still never reach the
/// disk. The next refresh with the same records writes them again, and they
/// survive a power cut; a refresh after a write that did complete writes
/// nothing.
#[test]
fn a_promise_copy_whose_sync_failed_is_written_again() {
    use crate::engine::promise::PromiseCopy;
    use lsm_tree::fs::{Fault, FaultOp, FaultRule};
    let rig = PowerRig::new();
    let config = rig.config();
    let engine = rig.open_group(&config).expect("open");
    let incarnation = [5u8; 16];
    engine
        .write_promise_copy(&incarnation, b"given")
        .expect("first write");
    rig.faults.arm(
        FaultRule::new(
            FaultOp::SyncAll,
            Fault::Error(lsm_tree::io::ErrorKind::Other),
        )
        .times(1),
    );
    assert!(
        engine.write_promise_copy(&incarnation, b"next").is_err(),
        "the failed sync must fail the write"
    );
    rig.faults.clear();
    engine
        .refresh_promise_copy(&incarnation, || Ok(b"next".to_vec()))
        .expect("refresh");
    rig.cut(engine);
    let engine = rig.open_group(&config).expect("reopen");
    assert_eq!(
        engine.read_promise_copy().expect("read"),
        PromiseCopy::Held {
            incarnation,
            payload: b"next".to_vec()
        },
        "the records survive the cut"
    );
    rig.cut(engine);
}

/// Records this process already wrote durably are not written again: an
/// unchanged vote costs no file access at all.
#[test]
fn a_promise_copy_already_durable_is_not_written_again() {
    let rig = PowerRig::new();
    let config = rig.config();
    let engine = rig.open_group(&config).expect("open");
    let incarnation = [5u8; 16];
    engine
        .refresh_promise_copy(&incarnation, || Ok(b"given".to_vec()))
        .expect("write");
    let opened = rig.faults.open_count();
    engine
        .refresh_promise_copy(&incarnation, || Ok(b"given".to_vec()))
        .expect("refresh what is already durable");
    assert_eq!(
        rig.faults.open_count(),
        opened,
        "records already written durably are not touched again"
    );
    rig.cut(engine);
}

/// A store that ran in a group is refused when opened with a journal of its
/// own, as a store outside any group is: it would write outside the group's
/// log. The refusal changes nothing, and the store opens as a member again
/// with its records.
#[test]
fn a_member_store_is_refused_an_own_journal_and_left_as_it_was() {
    let rig = PowerRig::new();
    let config = rig.config();
    let engine = rig.open_group(&config).expect("open");
    engine
        .bind_consensus_log(&[3u8; 16])
        .expect("bind the consensus log");
    engine
        .commit_local(vec![member_put(b"vote", b"v3")], LocalSync::BeforeUse)
        .expect("commit");
    rig.cut(engine);
    let refused = StorageEngine::open_embedded(&config, Arc::new(TimestampOracle::new()));
    assert!(
        matches!(refused, Err(StorageError::GroupMemberStore { .. })),
        "a member's store opened with its own journal, got {:?}",
        refused.err()
    );
    assert!(
        !config.endpoints[0].path.join("oplog").join("0").exists(),
        "the refused open started a journal of its own"
    );
    let engine = rig.open_group(&config).expect("reopen as a member");
    assert_eq!(
        engine
            .get(Partition::Raft, b"vote")
            .expect("get")
            .as_deref(),
        Some(b"v3".as_slice())
    );
    rig.cut(engine);
}

/// A store opened without a journal by something that is not a group (an
/// offline check, a test harness) is still a store outside any group: no
/// consensus log was bound to it, so it opens with its own journal again.
#[test]
fn a_store_opened_without_a_journal_but_no_group_is_still_its_own() {
    let rig = PowerRig::new();
    let config = rig.config();
    let engine = rig.open_group(&config).expect("open without a journal");
    engine
        .commit_local(vec![member_put(b"mark", b"m")], LocalSync::BeforeUse)
        .expect("commit");
    rig.stop(engine);
    let (engine, _) = rig.open_with(&config);
    assert_eq!(
        engine
            .get(Partition::Raft, b"mark")
            .expect("get")
            .as_deref(),
        Some(b"m".as_slice())
    );
    rig.cut(engine);
}

/// The retirement of a journal into its store, cut short after it covered
/// every tree and removed the oldest segments (the order it removes them
/// in), is finished by the next one, and the store then opens with every
/// entry and a journal that starts above them.
#[test]
fn a_journal_retirement_cut_short_is_finished_by_the_next_one() {
    let dir = TempDir::new().expect("tmp");
    let mut config = durable_cfg(&dir);
    config.oplog_segment_max_entries = 2;
    let names = ["a", "b", "c", "d", "e"];
    {
        let oracle = Arc::new(TimestampOracle::new());
        let engine = StorageEngine::open_embedded(&config, Arc::clone(&oracle)).expect("open");
        for (n, name) in names.iter().enumerate() {
            commit_at(
                &engine,
                &oracle,
                &two_tree_commit(name, n as u64),
                Journal::Journal,
            );
        }
    }
    // The retirement's first step, then part of its removal.
    {
        let engine = StorageEngine::open_embedded(&config, Arc::new(TimestampOracle::new()))
            .expect("reopen");
        engine.persist_coverage_base().expect("cover every tree");
    }
    let mut segments: Vec<_> = std::fs::read_dir(dir.path().join("oplog").join("0"))
        .expect("journal")
        .map(|e| e.expect("entry").path())
        .collect();
    segments.sort();
    assert!(segments.len() >= 3, "segments: {segments:?}");
    for segment in &segments[..segments.len() / 2] {
        std::fs::remove_file(segment).expect("remove");
    }

    assert!(StorageEngine::retire_embedded_journal(&config).expect("retire"));
    let oracle = Arc::new(TimestampOracle::new());
    let engine = StorageEngine::open_embedded(&config, Arc::clone(&oracle)).expect("open");
    for name in names {
        assert_eq!(
            engine
                .get(Partition::Node, format!("node:00:{name}").as_bytes())
                .expect("get")
                .as_deref(),
            Some(name.as_bytes()),
            "{name}"
        );
    }
    commit_at(&engine, &oracle, &two_tree_commit("f", 5), Journal::Journal);
    drop(engine);
    let engine = StorageEngine::open_embedded(&config, Arc::new(TimestampOracle::new()))
        .expect("open after a write on the retired store");
    assert!(
        engine
            .get(Partition::Node, b"node:00:f")
            .expect("get")
            .is_some()
    );
}

/// A power cut after the retirement removed the journal's segments but
/// before the directory sync can bring any subset of them back: the removals
/// were not durable. Whichever come back, a gap among them included, the
/// next retirement finishes the job, and the store holds every entry and
/// takes writes again.
#[test]
fn retired_segments_that_come_back_after_a_cut_are_retired_again() {
    let names = ["a", "b", "c", "d", "e", "f", "g"];
    // Which of the removed segments the cut brings back, by position.
    let subsets: [fn(usize) -> bool; 4] = [|_| true, |i| i % 2 == 0, |i| i % 2 == 1, |i| i > 0];
    for (round, comes_back) in subsets.into_iter().enumerate() {
        let dir = TempDir::new().expect("tmp");
        let mut config = durable_cfg(&dir);
        config.oplog_segment_max_entries = 2;
        {
            let oracle = Arc::new(TimestampOracle::new());
            let engine = StorageEngine::open_embedded(&config, Arc::clone(&oracle)).expect("open");
            for (n, name) in names.iter().enumerate() {
                commit_at(
                    &engine,
                    &oracle,
                    &two_tree_commit(name, n as u64),
                    Journal::Journal,
                );
            }
        }
        let journal = dir.path().join("oplog").join("0");
        let mut segments: Vec<_> = std::fs::read_dir(&journal)
            .expect("journal")
            .map(|e| e.expect("entry").path())
            .collect();
        segments.sort();
        assert!(segments.len() >= 3, "segments: {segments:?}");
        let saved: Vec<(std::path::PathBuf, Vec<u8>)> = segments
            .iter()
            .map(|p| (p.clone(), std::fs::read(p).expect("read segment")))
            .collect();

        assert!(StorageEngine::retire_embedded_journal(&config).expect("retire"));
        // The cut: the removals were not durable, some segments are back.
        for (i, (path, bytes)) in saved.iter().enumerate() {
            if comes_back(i) {
                std::fs::write(path, bytes).expect("bring back");
            }
        }

        StorageEngine::retire_embedded_journal(&config)
            .unwrap_or_else(|e| panic!("round {round}: retire again: {e}"));
        let oracle = Arc::new(TimestampOracle::new());
        let engine = StorageEngine::open_embedded(&config, Arc::clone(&oracle))
            .unwrap_or_else(|e| panic!("round {round}: open: {e}"));
        for name in names {
            assert_eq!(
                engine
                    .get(Partition::Node, format!("node:00:{name}").as_bytes())
                    .expect("get")
                    .as_deref(),
                Some(name.as_bytes()),
                "round {round}: {name}"
            );
        }
        commit_at(
            &engine,
            &oracle,
            &two_tree_commit("z", 99),
            Journal::Journal,
        );
        drop(engine);
        let engine = StorageEngine::open_embedded(&config, Arc::new(TimestampOracle::new()))
            .unwrap_or_else(|e| panic!("round {round}: reopen after a write: {e}"));
        assert!(
            engine
                .get(Partition::Node, b"node:00:z")
                .expect("get")
                .is_some(),
            "round {round}"
        );
    }
}

/// A member's own record is written while client writes are refused for
/// space, out of the upper half of the reserve; once free space falls into
/// the lower half, which the flushes retiring those records need, it is
/// refused too.
#[test]
fn member_records_use_the_upper_half_of_the_space_reserve() {
    let rig = PowerRig::new();
    let config = rig.config();
    let engine = rig.open_group(&config).expect("open");
    let space = engine.space();
    let free = space.available_bytes();
    // Free space just under the reserve: client writes stop, records go on.
    space.set_reserve(free + (64 << 20), u64::MAX);
    assert!(space.admit().is_err(), "client writes are admitted");
    engine
        .commit_local(vec![member_put(b"vote", b"v")], LocalSync::BeforeUse)
        .expect("a record within the recovery half");
    // Free space under half the reserve: the rest is the flushes' to use.
    space.set_reserve(free.checked_mul(3).expect("a measured disk"), u64::MAX);
    assert!(
        matches!(
            engine.commit_local(vec![member_put(b"vote", b"w")], LocalSync::BeforeUse),
            Err(StorageError::OutOfSpace { .. })
        ),
        "a record took the space its retirement needs"
    );
    space.set_reserve(0, 0);
    rig.cut(engine);
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
    // Every journal append and every table/manifest publish ends in a sync:
    // a full one, or a data sync where the trees order their writes with a
    // barrier. Cutting at both walks the whole durability order.
    let full = cut_at_every(lsm_tree::fs::FaultOp::SyncAll);
    let data = cut_at_every(lsm_tree::fs::FaultOp::SyncData);
    assert!(full + data > 0, "the workload reached no sync");
}

#[test]
fn a_power_cut_at_any_write_leaves_every_commit_once() {
    // A failed write leaves a torn record or table behind; recovery must
    // read past it exactly as past a cut at a sync.
    assert!(cut_at_every(lsm_tree::fs::FaultOp::Write) > 0);
}

/// A commit writing `name` into the node tree and an edge numbered `n` into
/// the adjacency tree: one entry, two trees.
fn two_tree_commit(name: &str, n: u64) -> Vec<Mutation> {
    vec![
        Mutation::Put {
            partition: PartitionId::Node,
            key: format!("node:00:{name}").into_bytes(),
            value: name.as_bytes().to_vec(),
        },
        Mutation::Merge {
            partition: PartitionId::Adj,
            key: b"adj:V:out:1".to_vec(),
            operand: crate::engine::merge::encode_add(n),
        },
    ]
}

/// Whether `name`'s node row is there, and the edges the adjacency tree holds.
fn two_tree_state(engine: &StorageEngine, name: &str) -> (bool, Vec<u64>) {
    let node = engine
        .get(Partition::Node, format!("node:00:{name}").as_bytes())
        .expect("get node")
        .is_some();
    let edges = engine
        .get(Partition::Adj, b"adj:V:out:1")
        .expect("get adj")
        .map(|v| {
            coordinode_core::graph::edge::PostingList::from_bytes(&v)
                .expect("posting list")
                .iter()
                .collect()
        })
        .unwrap_or_default();
    (node, edges)
}

fn commit_at(engine: &StorageEngine, oracle: &TimestampOracle, m: &[Mutation], level: Journal) {
    engine
        .commit_logged(m, oracle.next().as_raw(), level)
        .expect("commit");
}

/// A volatile commit is visible as soon as it returns, and a power cut
/// before its sync takes it, never the journaled commit before it.
#[test]
fn a_volatile_commit_is_visible_at_once_and_a_power_cut_takes_only_it() {
    let rig = PowerRig::new();
    {
        let (engine, oracle) = rig.open();
        commit_at(&engine, &oracle, &two_tree_commit("a", 1), Journal::Journal);
        commit_at(&engine, &oracle, &two_tree_commit("b", 2), Journal::Memory);
        assert_eq!(
            two_tree_state(&engine, "b"),
            (true, vec![1, 2]),
            "a volatile commit reads back the moment it returns"
        );
        rig.cut(engine);
    }
    let (engine, _) = rig.open();
    assert_eq!(two_tree_state(&engine, "a"), (true, vec![1]));
    assert!(
        !two_tree_state(&engine, "b").0,
        "the unsynced commit is lost"
    );
}

/// A journaled commit syncs the journal up to itself, so the volatile
/// commits before it survive a power cut with it.
#[test]
fn a_journaled_commit_makes_the_volatile_commits_before_it_durable() {
    let rig = PowerRig::new();
    {
        let (engine, oracle) = rig.open();
        commit_at(&engine, &oracle, &two_tree_commit("a", 1), Journal::Memory);
        commit_at(&engine, &oracle, &two_tree_commit("b", 2), Journal::Cache);
        commit_at(&engine, &oracle, &two_tree_commit("c", 3), Journal::Journal);
        rig.cut(engine);
    }
    let (engine, _) = rig.open();
    for name in ["a", "b", "c"] {
        assert!(two_tree_state(&engine, name).0, "{name} must survive");
    }
    assert_eq!(two_tree_state(&engine, "c").1, vec![1, 2, 3]);
}

/// Log before data: flushing one tree syncs the journal first. Without that,
/// a volatile commit's node row reached a table while its journal entry, and
/// so its edge in the other tree, was lost to the cut: half a commit.
#[test]
fn a_table_flush_syncs_the_journal_before_it_writes() {
    let rig = PowerRig::new();
    {
        let (engine, oracle) = rig.open();
        commit_at(&engine, &oracle, &two_tree_commit("a", 1), Journal::Memory);
        engine
            .persist_partition(Partition::Node)
            .expect("flush the node tree");
        commit_at(&engine, &oracle, &two_tree_commit("b", 2), Journal::Memory);
        rig.cut(engine);
    }
    let (engine, _) = rig.open();
    let (a, edges) = two_tree_state(&engine, "a");
    assert!(
        a && edges.first() == Some(&1),
        "the commit whose row reached a table is whole: {edges:?}"
    );
    // A background flush may have synced the later commit too; either way it
    // is whole or absent.
    let (b, _) = two_tree_state(&engine, "b");
    assert_eq!(
        edges,
        if b { vec![1, 2] } else { vec![1] },
        "a commit is never half there"
    );
}

/// The background flusher syncs a volatile commit within its interval with
/// no other write to do it.
#[test]
fn the_flusher_syncs_a_volatile_commit_within_its_interval() {
    let rig = PowerRig::new();
    let mut config = rig.config();
    config.volatile_flush_interval_ms = 10;
    {
        let (engine, oracle) = rig.open_with(&config);
        commit_at(&engine, &oracle, &two_tree_commit("a", 1), Journal::Memory);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while engine.journal_crash_point(CrashKind::PowerCut).is_some() {
            assert!(
                std::time::Instant::now() < deadline,
                "the flusher never synced the commit"
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        rig.cut(engine);
    }
    let (engine, _) = rig.open();
    assert_eq!(two_tree_state(&engine, "a"), (true, vec![1]));
}

/// A killed process keeps what `cache` handed to the operating system and
/// loses what `memory` left in its own buffer, in log order.
#[test]
fn a_killed_process_keeps_cache_commits_and_loses_memory_ones() {
    let rig = PowerRig::new();
    {
        let (engine, oracle) = rig.open();
        commit_at(&engine, &oracle, &two_tree_commit("a", 1), Journal::Cache);
        commit_at(&engine, &oracle, &two_tree_commit("b", 2), Journal::Memory);
        rig.kill(engine);
    }
    let (engine, _) = rig.open();
    assert_eq!(two_tree_state(&engine, "a"), (true, vec![1]));
    assert!(
        !two_tree_state(&engine, "b").0,
        "the buffered commit is lost"
    );
}

/// A durable view stops below a volatile commit until the journal syncs it,
/// and choosing the view syncs nothing: the commit is still unsynced after
/// any number of choices, and a journaled commit, which syncs everything
/// before it, brings it into the view.
#[test]
fn a_durable_view_excludes_an_unsynced_commit_without_syncing() {
    let rig = PowerRig::new();
    let (engine, oracle) = rig.open();
    let ts = oracle.next().as_raw();
    engine
        .commit_logged(&two_tree_commit("a", 1), ts, Journal::Memory)
        .expect("commit");
    let fresh = oracle.next().as_raw();
    for _ in 0..3 {
        assert!(
            engine.durable_view(fresh) <= ts,
            "a durable view reaches past an unsynced commit"
        );
    }
    assert!(
        engine.journal_crash_point(CrashKind::PowerCut).is_some(),
        "choosing a view synced the journal"
    );
    commit_at(&engine, &oracle, &two_tree_commit("b", 2), Journal::Journal);
    let fresh = oracle.next().as_raw();
    assert_eq!(
        engine.durable_view(fresh),
        fresh,
        "everything is durable now"
    );
    rig.cut(engine);
}

/// Commit timestamps need not follow journal order: an unsynced commit
/// stamped lower than an earlier unsynced one bounds the view, and a sync
/// lifts the bound past both.
#[test]
fn a_durable_view_follows_the_lowest_unsynced_timestamp() {
    let rig = PowerRig::new();
    let (engine, oracle) = rig.open();
    let low = oracle.next().as_raw();
    let high = oracle.next().as_raw();
    engine
        .commit_logged(&two_tree_commit("high", 2), high, Journal::Memory)
        .expect("high");
    assert!(engine.durable_view(oracle.next().as_raw()) <= high);
    engine
        .commit_logged(&two_tree_commit("low", 1), low, Journal::Cache)
        .expect("low");
    assert!(engine.durable_view(oracle.next().as_raw()) <= low);
    crate::engine::oplog_journal::sync_journal(engine.oplog.as_ref().expect("journal"))
        .expect("sync");
    let fresh = oracle.next().as_raw();
    assert_eq!(engine.durable_view(fresh), fresh);
    rig.cut(engine);
}

/// The unsynced tail is bounded: past sixteen times the flush size the
/// writer syncs before it is answered, with the flusher held off.
#[test]
fn the_unsynced_tail_is_bounded() {
    let rig = PowerRig::new();
    let mut config = rig.config();
    config.volatile_flush_bytes = 1;
    let (engine, oracle) = rig.open_with(&config);
    commit_at(&engine, &oracle, &two_tree_commit("a", 1), Journal::Memory);
    assert!(
        engine.journal_crash_point(CrashKind::PowerCut).is_none(),
        "a write past the cap left the tail unsynced"
    );
    rig.cut(engine);
    let (engine, _) = rig.open();
    assert_eq!(two_tree_state(&engine, "a"), (true, vec![1]));
}

/// A journaled retry of an operation a volatile write already bound applies
/// nothing, and answers with the recorded commit: that answer promises the
/// binding is durable, so the retry syncs the journal before it returns.
/// Without that, a power cut lost the operation the retry reported
/// committed.
#[test]
fn a_journaled_retry_of_a_volatile_operation_makes_it_durable() {
    use coordinode_core::txn::operation::{
        DEFAULT_ADMISSION_WINDOW, INTENT_DIGEST_LEN, ISSUER_ID_LEN, IntentDigest, IssuerId,
        OperationBinding, OperationCommand, OperationId, OperationOutcome,
    };
    let issuer = IssuerId([0x6b; ISSUER_ID_LEN]);
    let op = OperationId {
        issuer,
        incarnation: 1,
        sequence: 1,
    };
    let bound = |attempt| {
        vec![
            Mutation::Merge {
                partition: PartitionId::Counter,
                key: b"counter:retry".to_vec(),
                operand: crate::engine::merge::encode_counter_delta(1),
            },
            Mutation::Operation(OperationCommand::Bind(OperationBinding {
                op,
                intent: IntentDigest([7; INTENT_DIGEST_LEN]),
                attempt,
            })),
        ]
    };
    let rig = PowerRig::new();
    {
        let (engine, oracle) = rig.open();
        let enroll = Mutation::Operation(OperationCommand::Enroll {
            issuer,
            incarnation: 1,
            window: DEFAULT_ADMISSION_WINDOW,
        });
        commit_at(&engine, &oracle, &[enroll], Journal::Journal);
        commit_at(&engine, &oracle, &bound(1), Journal::Memory);
        commit_at(&engine, &oracle, &bound(2), Journal::Journal);
        assert!(matches!(
            crate::engine::operations::operation_outcome(&engine, &op).expect("outcome"),
            OperationOutcome::Committed { attempt: 1, .. }
        ));
        rig.cut(engine);
    }
    let (engine, _) = rig.open();
    assert!(
        matches!(
            crate::engine::operations::operation_outcome(&engine, &op).expect("outcome"),
            OperationOutcome::Committed { attempt: 1, .. }
        ),
        "the outcome the journaled retry reported survives"
    );
    let counter = engine
        .get(Partition::Counter, b"counter:retry")
        .expect("get")
        .map(|v| crate::engine::merge::decode_counter(&v).expect("decode"))
        .unwrap_or(0);
    assert_eq!(counter, 1, "applied once");
}

/// After a failed sync the journal takes no write and no table flush runs:
/// a later sync reporting success would not prove the lost pages durable.
#[test]
fn a_failed_journal_sync_refuses_every_later_write_and_flush() {
    let rig = PowerRig::new();
    let (engine, oracle) = rig.open();
    commit_at(&engine, &oracle, &two_tree_commit("a", 1), Journal::Memory);
    {
        let oplog = engine.oplog.as_ref().expect("journal");
        let mut journal = oplog.lock().expect("lock");
        let through = journal.next_index();
        assert!(
            journal
                .note_sync(through, Err(StorageError::Io("injected".into())))
                .is_err()
        );
    }
    for level in [Journal::Journal, Journal::Cache, Journal::Memory] {
        assert!(
            engine
                .commit_logged(&two_tree_commit("b", 2), oracle.next().as_raw(), level)
                .is_err(),
            "{level} write accepted after a failed sync"
        );
    }
    assert!(
        engine.persist().is_err(),
        "a table flush ran with the journal not durable"
    );
    rig.cut(engine);
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
    use coordinode_core::index::encoding::{encode_entry_key, encode_tuple};
    use coordinode_core::index::identity::GenerationId;
    use coordinode_core::txn::proposal::{DerivedIndexWork, DerivedSource, IndexBinding};

    use super::*;

    /// The generation of the email index.
    const EMAIL_GEN: GenerationId = GenerationId::from_raw(1);

    fn email_index() -> IndexInterpretation {
        IndexInterpretation {
            codec: KEY_CODEC,
            generation: EMAIL_GEN,
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
        encode_entry_key(EMAIL_GEN, &tuple, node_id)
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
                valid_from: None,
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

    /// DERIVED work applied while a replacement of its generation is
    /// prepared reaches the replacement: once published, the entry it
    /// derived is there and the one it removed is gone, as in the copy it
    /// replaced.
    #[test]
    fn derived_entries_applied_during_a_replacement_reach_it() {
        let dir = TempDir::new().expect("temp dir");
        let oracle = Arc::new(TimestampOracle::new());
        let engine =
            StorageEngine::open_embedded(&durable_cfg(&dir), oracle.clone()).expect("open");
        write_batch(&engine, &oracle, &unit(1, "a@x", None));
        engine.stage_generation(EMAIL_GEN).expect("stage");
        write_batch(&engine, &oracle, &unit(1, "b@x", Some("a@x")));
        let history = engine.export_generation_history(EMAIL_GEN).expect("export");
        engine
            .import_generation_history(EMAIL_GEN, &history.entries)
            .expect("import");
        engine
            .finish_generation_import(EMAIL_GEN, history.covers_through, history.history_from)
            .expect("finish");
        engine.publish_generation(EMAIL_GEN).expect("publish");
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
