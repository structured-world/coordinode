//! Settling the Schema tree before the open reads it.
//!
//! The Schema tree holds what the rest of the open depends on: the routes of
//! every other tree, the closure ledger, this member's own records, the
//! apply coverage the journal replay trusts. A tree the open salvaged with
//! loss, whose rebuild a crash cut short, or (where the open verifies the
//! trees a checkpoint backs) that fails its checksums is settled in two
//! steps. Before anything reads it, its own records are proved (from the
//! intent of the rebuild cut short, or from the tree when nothing it lost can
//! be one of them) and the open is refused otherwise: an absent key in such a
//! tree is not an answer. Then, once the journal is read and the other trees
//! say which of its entries they hold, and before the journal is replayed,
//! the tree is rebuilt from the latest checkpoint and the journal where they
//! prove it, and the open is refused where they do not.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use lsm_tree::{AbstractTree as _, AnyTree, Guard as _, ScanSinceEvent};

use super::{
    OpenRepair, RebuildIntents, RebuildTarget, Rows, encode_kept_rows, read_kept_rows,
    read_rebuild_intents, remove_rebuild_intent, write_rebuild_intent, write_rebuild_intent_with,
};
use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use crate::engine::coverage::{self, Domain, TreeCoverage};
use crate::engine::footprint::Footprints;
use crate::engine::oplog_journal::{apply_oplog_ops_at, op_partition};
use crate::engine::partition::Partition;
use crate::engine::rebuild::{
    RebuildRefusal, checkpoint_root, latest_checkpoint, tree_history, uncovered_entry,
};
use crate::error::{StorageError, StorageResult};
use crate::oplog::entry::{OplogEntry, OplogOp};

/// Where, under the data directory, the open stages its copy of a
/// checkpoint's Schema tree: the checkpoint itself is copied, never opened,
/// so it stays as it was.
const SOURCE_DIR: &str = "schema-source";

/// The key ranges, `[start, end)`, of this member's own records in the
/// Schema tree: its routes and records (`meta:`) and its consensus state
/// machine's (`raft:`).
const MEMBER_RANGES: [(&[u8], &[u8]); 2] = [(b"meta:", b"meta;"), (b"raft:", b"raft;")];

/// The closure ledger's records, the head and the breakpoints, as a range:
/// read at the open as well, on a store whose log is a group's.
pub(super) const CLOSURE_RANGE: (&[u8], &[u8]) = (b"\x00cl", b"\x00cm");

/// A rebuild of the Schema tree at the open stops right after its clear, as
/// a crash there would.
#[cfg(any(test, feature = "test-faults"))]
pub static CUT_AFTER_CLEAR: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// What settling the Schema tree left for the rest of the open.
#[derive(Default)]
pub(super) struct Settled {
    /// The version of the tree whose blocks this open verified, changing
    /// nothing: a verification later in the open of the same version reuses
    /// the answer.
    pub(super) verified: Option<u64>,
    /// The rebuild the tree is owed, run once the journal is read.
    pub(super) rebuild: Option<SchemaRebuild>,
}

/// The open's view of the Schema tree, before anything reads it.
pub(super) struct Bootstrap<'a> {
    pub(super) config: &'a StorageConfig,
    pub(super) schema: &'a AnyTree,
    /// The repair the open ran on the tree, if any.
    pub(super) repair: Option<&'a OpenRepair>,
    pub(super) seqno: &'a lsm_tree::SharedSequenceNumberGenerator,
    /// Whether the store keeps a journal of its own, the source a rebuild
    /// replays.
    pub(super) journaled: bool,
    pub(super) maintenance: &'a Arc<lsm_tree::rate_limiter::RateLimiter>,
}

impl Bootstrap<'_> {
    /// Prove the tree's own records before anything reads the tree, and say
    /// which rebuild it is owed.
    ///
    /// # Errors
    ///
    /// [`StorageError::SchemaRecoveryRefused`] when the tree's own records
    /// cannot be proved; nothing is written then. Otherwise a read or write
    /// failure.
    pub(super) fn settle(self) -> StorageResult<Settled> {
        let data_dir = self.config.data_dir();
        let on_disk = data_dir.is_dir();
        let intents = if on_disk {
            read_rebuild_intents(data_dir)?
        } else {
            RebuildIntents::default()
        };
        let pending = intents.partitions.contains(&Partition::Schema);
        let lossy = self.repair.filter(|repair| repair.is_lossy());
        let checkpoint = if self.journaled && on_disk {
            latest_checkpoint(&checkpoint_root(data_dir))?
        } else {
            None
        };

        // Where the open goes on to verify the trees a checkpoint backs, the
        // Schema tree is verified first, here, and only once.
        let verifies = self.journaled
            && (checkpoint.is_some()
                || !intents.partitions.is_empty()
                || !intents.tables.is_empty());
        let mut verified = None;
        let mut corrupt = false;
        if verifies && !pending && lossy.is_none() {
            let report = self.schema.verify_checksum_with(
                &lsm_tree::verify::VerifyOptions::default()
                    .rate_limiter(Arc::clone(self.maintenance)),
            );
            if report.errors.is_empty() {
                verified = Some(self.schema.current_version().id());
            } else {
                for error in &report.errors {
                    tracing::error!(%error, "the Schema tree holds a corrupt block");
                }
                corrupt = true;
            }
        }
        // The journal replay trusts the tree's coverage record.
        let coverage_damaged = self.journaled
            && match TreeCoverage::read(self.schema, Domain::Journal) {
                Ok(_) => false,
                Err(e) if e.is_data_damage() => true,
                Err(e) => return Err(e),
            };
        if !(pending || lossy.is_some() || corrupt || coverage_damaged) {
            return Ok(Settled {
                verified,
                rebuild: None,
            });
        }

        let kept = self.kept_records(data_dir, pending, lossy)?;
        if pending {
            // The clear took them; read as the tree's from here on. The scope's
            // accounting is counted again once the open can.
            write_rebuild_intent(data_dir, RebuildTarget::Recount(Partition::Schema))?;
            let at = self.seqno.next();
            for (key, value) in &kept {
                self.schema.insert(key.as_slice(), value.as_slice(), at);
            }
        }
        // Owed from here until a rebuild has it on disk, with the records it
        // keeps: the journal replay leaves the tree to the rebuild. A tree
        // whose coverage record is damaged is owed one even with nothing here
        // to run it, as the open owes any such tree one.
        let owed = !pending && (checkpoint.is_some() || coverage_damaged);
        if owed {
            write_rebuild_intent_with(
                data_dir,
                RebuildTarget::Partition(Partition::Schema),
                &encode_kept_rows(&kept)?,
            )?;
        }
        // With no checkpoint, nothing here rebuilds the tree: what it lost
        // stays owed to the repair that follows the open, and the records the
        // open reads from it are the ones proved above.
        Ok(Settled {
            verified: None,
            rebuild: checkpoint.map(|checkpoint| SchemaRebuild { checkpoint, kept }),
        })
    }

    /// This member's own records in the tree, from what proves them: the
    /// intent of a rebuild cut short, which holds them as they were before
    /// its clear, or the tree itself when nothing it lost can be one of
    /// them.
    fn kept_records(
        &self,
        data_dir: &Path,
        pending: bool,
        lossy: Option<&OpenRepair>,
    ) -> StorageResult<Rows> {
        let lost = || StorageError::SchemaRecoveryRefused(RebuildRefusal::LocalStateLost);
        if pending {
            return read_kept_rows(data_dir)?.ok_or_else(lost);
        }
        if let Some(repair) = lossy
            && !member_records_intact(repair, self.journaled)
        {
            tracing::error!(
                lost = ?repair.report.lost_coverage,
                unknowable = ?repair.report.unknowable_losses,
                "the Schema tree's salvage may have lost records the open reads first"
            );
            return Err(lost());
        }
        match member_rows(self.schema) {
            Err(e) if e.is_data_damage() => Err(lost()),
            rows => rows,
        }
    }
}

/// A rebuild of the Schema tree from the latest checkpoint and the journal,
/// owed by the open: run once the journal is read, before it is replayed.
pub(super) struct SchemaRebuild {
    checkpoint: PathBuf,
    kept: Rows,
}

/// What a [`SchemaRebuild`] runs against.
pub(super) struct RebuildInputs<'a> {
    pub(super) config: &'a StorageConfig,
    pub(super) schema: &'a AnyTree,
    /// The repair the open ran on the tree, if any: met by the rebuild.
    pub(super) repair: Option<&'a OpenRepair>,
    pub(super) seqno: &'a lsm_tree::SharedSequenceNumberGenerator,
    pub(super) footprints: &'a Footprints,
    pub(super) repair_policy: lsm_tree::RepairPolicy,
    /// Every entry the journal holds.
    pub(super) entries: &'a [OplogEntry],
    /// One past the last entry applied: the journal's next index, raised to
    /// what any other tree holds.
    pub(super) next: u64,
}

impl SchemaRebuild {
    /// Rebuild the tree: the checkpoint copy's history at its own seqnos,
    /// the kept records, every journal entry the copy lacks at its own
    /// timestamp, and the record that the tree holds every entry applied.
    ///
    /// # Errors
    ///
    /// [`StorageError::SchemaRecoveryRefused`] when the checkpoint's copy is
    /// damaged or an applied entry is in neither it nor the journal; the tree
    /// is not cleared then. Otherwise a read or write failure, which leaves
    /// the rebuild owed.
    pub(super) fn run(self, inputs: RebuildInputs<'_>) -> StorageResult<()> {
        let RebuildInputs {
            config,
            schema,
            repair,
            seqno,
            footprints,
            repair_policy,
            entries,
            next,
        } = inputs;
        let data_dir = config.data_dir();

        // 1. Prove the tree's state before anything changes.
        let staging = data_dir.join(SOURCE_DIR);
        let source = open_source(
            config,
            footprints,
            repair_policy,
            &self.checkpoint,
            &staging,
        )?;
        let held = TreeCoverage::read(&source, Domain::Journal)?;
        if let Some(refusal) = uncovered_entry(&held, entries, (next, &[])) {
            drop(source);
            remove_staging(&staging, footprints)?;
            return Err(StorageError::SchemaRecoveryRefused(refusal));
        }
        let source_floor = source.retention_floor();

        // 2. Clear, the intent and the scope's recount owed already.
        write_rebuild_intent(data_dir, RebuildTarget::Recount(Partition::Schema))?;
        for tree in [schema, &source] {
            if let Some(max) = tree.get_highest_seqno() {
                seqno.fetch_max(max + 1);
            }
            if let Some(first_servable) = tree.retention_floor().checked_add(1) {
                seqno.fetch_max(first_servable);
            }
        }
        schema.clear()?;
        #[cfg(any(test, feature = "test-faults"))]
        if CUT_AFTER_CLEAR.swap(false, std::sync::atomic::Ordering::AcqRel) {
            return Err(StorageError::Io("cut short after the clear".into()));
        }

        // 3. The copy's history at its own seqnos, but for its coverage
        //    record and the records of the member it was taken on, then this
        //    member's own.
        let mut versions = 0usize;
        for event in tree_history(&source)? {
            versions += usize::from(reinstall(schema, event));
        }
        drop(source);
        let at = seqno.next();
        for (key, value) in &self.kept {
            schema.insert(key.as_slice(), value.as_slice(), at);
        }

        // 4. The entries the copy lacks, each at its own timestamp with its
        //    marker, then the record of every entry applied.
        let mut replayed = 0usize;
        for entry in entries {
            if held.contains(entry.index, 0) {
                continue;
            }
            let ops = crate::oplog::convert::resolve_entry_ops(&entry.ops)?;
            let ops: Vec<&OplogOp> = ops
                .iter()
                .filter(|op| op_partition(op) == Some(Partition::Schema))
                .collect();
            if ops.is_empty() {
                continue;
            }
            apply_oplog_ops_at(schema, None, &ops, entry.ts, entry.index, Domain::Journal)?;
            replayed += 1;
        }
        schema.insert(
            Domain::Journal.base_key(),
            coverage::encode_base(next, &[]),
            seqno.next(),
        );

        // 5. On disk, then served from the copy's floor, then no longer owed.
        flush(config, schema, footprints)?;
        if source_floor < schema.retention_floor() {
            schema.lower_retention_floor(source_floor)?;
        }
        remove_rebuild_intent(data_dir, RebuildTarget::Partition(Partition::Schema))?;
        if let Some(repair) = repair {
            schema.acknowledge_repair_report(repair.report.id)?;
        }
        remove_staging(&staging, footprints)?;
        tracing::warn!(
            versions,
            replayed,
            checkpoint = %self.checkpoint.display(),
            "the Schema tree was rebuilt from the latest checkpoint and the journal"
        );
        Ok(())
    }
}

/// The checkpoint's Schema tree, from a copy staged at `staging`.
fn open_source(
    config: &StorageConfig,
    footprints: &Footprints,
    repair_policy: lsm_tree::RepairPolicy,
    checkpoint: &Path,
    staging: &Path,
) -> StorageResult<AnyTree> {
    // A copy an earlier open left behind is of no use.
    remove_staging(staging, footprints)?;
    let tree_dir = Partition::Schema.name();
    // Charged file by file before it is written; a copy cut short by the
    // limit or a failure is removed with what it charged.
    if let Err(e) = copy_dir(
        &checkpoint.join(tree_dir),
        &staging.join(tree_dir),
        footprints,
    ) {
        remove_staging(staging, footprints)?;
        return Err(e);
    }
    let mut config = config.clone();
    config.endpoints = vec![EndpointConfig::new(
        "source",
        staging,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )];
    let seqno =
        lsm_tree::SharedSequenceNumberGenerator::from(lsm_tree::SequenceNumberCounter::default());
    let (source, repair) = config
        .to_tree_config(Partition::Schema, seqno)
        .open_or_repair(repair_policy)?;
    if repair.is_some_and(|report| {
        !matches!(
            report.wal_replay_scope(),
            lsm_tree::WalReplayScope::TailOnly
        )
    }) {
        drop(source);
        remove_staging(staging, footprints)?;
        return Err(StorageError::SchemaRecoveryRefused(
            RebuildRefusal::CheckpointDamaged,
        ));
    }
    Ok(source)
}

/// Write `tree`'s memtables to tables, durable when this returns. No journal
/// entry is appended while the open runs, so none is owed a sync first.
fn flush(config: &StorageConfig, tree: &AnyTree, footprints: &Footprints) -> StorageResult<()> {
    let lock = tree.get_flush_lock();
    tree.rotate_memtable();
    footprints.flushing(tree.id());
    tree.flush(&lock, 0)?;
    tree.persist_flushes()?;
    footprints.persisted(tree.id());
    if config.oplog_sync_method.tree_sync_mode() == lsm_tree::fs::SyncMode::Barrier {
        tree.sync_devices()?;
    }
    Ok(())
}

/// Whether nothing a salvage lost from the Schema tree can be a record the
/// open reads before the store is anything: every lost table's key range is
/// known and misses them. A store with a journal of its own has no closure
/// ledger to lose.
fn member_records_intact(repair: &OpenRepair, journaled: bool) -> bool {
    if !repair.report.unknowable_losses.is_empty() {
        return false;
    }
    let closure = (!journaled).then_some(CLOSURE_RANGE);
    repair.report.lost_coverage.iter().all(|(_, min, max, _)| {
        MEMBER_RANGES
            .iter()
            .copied()
            .chain(closure)
            .all(|(start, end)| min.as_ref() >= end || max.as_ref() < start)
    })
}

/// This member's own records in the Schema tree, read at its newest state.
fn member_rows(schema: &AnyTree) -> StorageResult<Rows> {
    let mut rows = Rows::new();
    for (prefix, _) in MEMBER_RANGES {
        for guard in schema.prefix(prefix, lsm_tree::SeqNo::MAX, None) {
            let (key, value) = guard.into_inner()?;
            rows.push((key.to_vec(), value.to_vec()));
        }
    }
    Ok(rows)
}

/// Write one event of the checkpoint copy's history into `schema` at its own
/// seqno. Returns whether it belongs to the rebuilt tree: the copy's
/// coverage record does not (the rebuilt tree's is written once proved),
/// nor do the records of the member the copy was taken on.
fn reinstall(schema: &AnyTree, event: ScanSinceEvent) -> bool {
    let own = |key: &[u8]| {
        coverage::is_reserved(key) || super::raft_coverage::is_node_local(Partition::Schema, key)
    };
    match event {
        ScanSinceEvent::RangeTombstone {
            start_key,
            end_key,
            seqno,
        } => {
            let start = coverage::clamp_user_start(&start_key);
            if start >= end_key.as_ref() {
                return false;
            }
            schema.remove_range(start, end_key.as_ref(), seqno);
        }
        ScanSinceEvent::Insert { key, value, seqno } => {
            if own(&key) {
                return false;
            }
            schema.insert(key, value, seqno);
        }
        ScanSinceEvent::MergeOperand {
            key,
            operand,
            seqno,
        } => {
            if own(&key) {
                return false;
            }
            schema.merge(key, operand, seqno);
        }
        ScanSinceEvent::PointTombstone { key, seqno } => {
            if own(&key) {
                return false;
            }
            schema.remove(key, seqno);
        }
        ScanSinceEvent::WeakTombstone { key, seqno } => {
            if own(&key) {
                return false;
            }
            schema.remove_weak(key, seqno);
        }
    }
    true
}

/// Copy the directory `from` to `to`, file by file, each charged to
/// `footprints` before it is written.
fn copy_dir(from: &Path, to: &Path, footprints: &Footprints) -> StorageResult<()> {
    let io = |what: &str, path: &Path, e: std::io::Error| {
        StorageError::Io(format!("{what} {path:?}: {e}"))
    };
    // no-std: StoragePort trait, caller-injected
    std::fs::create_dir_all(to).map_err(|e| io("create", to, e))?;
    let mut stack: Vec<(PathBuf, PathBuf)> = vec![(from.to_path_buf(), to.to_path_buf())];
    while let Some((from, to)) = stack.pop() {
        for entry in std::fs::read_dir(&from).map_err(|e| io("read", &from, e))? {
            let entry = entry.map_err(|e| io("read", &from, e))?;
            let (source, target) = (entry.path(), to.join(entry.file_name()));
            if entry
                .file_type()
                .map_err(|e| io("inspect", &source, e))?
                .is_dir()
            {
                std::fs::create_dir_all(&target).map_err(|e| io("create", &target, e))?;
                stack.push((source, target));
            } else {
                super::copy_new(&source, &target, footprints)?;
            }
        }
    }
    Ok(())
}

/// Remove the staged copy at `staging`, if there is one, and once it is
/// gone what the footprint was charged for it. A removal that fails keeps
/// the charge of whatever is left.
fn remove_staging(staging: &Path, footprints: &Footprints) -> StorageResult<()> {
    // no-std: StoragePort trait, caller-injected
    match std::fs::remove_dir_all(staging) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(StorageError::Io(format!("remove {staging:?}: {e}"))),
    }
    footprints.removed_tree(staging);
    Ok(())
}
