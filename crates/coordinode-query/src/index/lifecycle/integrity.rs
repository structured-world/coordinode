//! Logical checks of B-tree index generations against the records their
//! entries are derived from.
//!
//! A generation is checked both ways: every stored record of the label for
//! the entries its own values give it, and every entry for a record that
//! holds its value. A disagreement is repaired entry by entry, each repair
//! committed only while the record and the entry are still as the check
//! read them and the index still serves from the checked generation; damage
//! too wide to repair that way rebuilds the index into a fresh generation.
//! Two records holding one unique value are reported, never resolved by
//! changing a record.
//!
//! The durable state is one [`IndexIntegrityRecord`] per generation: what is
//! known about it, the disagreements reported, and the latest check. Reads
//! and writes that meet a disagreement report it to the registry; the
//! maintenance thread of [`IndexBuildService`] records the report and admits
//! a check. A check moves through its states as a build does: every move is
//! a catalog commit conditioned on the record the mover read, and the
//! executor's token fences one that lost the check. Only a complete check
//! with no disagreement left, and no report arriving while it ran, marks
//! the generation verified; repairing one entry proves nothing about the
//! others.

use std::sync::Arc;
use std::time::{Duration, Instant};

use coordinode_core::graph::node::{NodeId, NodeRecord};
use coordinode_core::index::derive::{EntryOwner, IndexInterpretation, tuples};
use coordinode_modality::{
    CheckPhase, CheckState, IndexCheck, IndexIntegrityRecord, IndexStore as _, Integrity,
    LatestEntry, LocalIndexStore, LocalNodeStore, Mismatch, NodeStore as _, StoreError,
};
use coordinode_storage::engine::transaction::Transaction;
use rustc_hash::FxHashMap;

use super::{
    IndexBuildService, MOVE_ATTEMPTS, Shared, begin, cannot_move_here, is_contention, next_token,
    panic_message, text,
};
use crate::index::definition::{
    BuildFailure, GenerationId, IndexBuildRecord, IndexDefinition, IndexId, IndexState, IndexType,
};
use crate::index::registry::IntegrityReport;

/// Complete passes a check makes before it gives up repairing entry by
/// entry and rebuilds: a pass that repaired something is followed by one
/// that confirms the repairs, so two passes settle any damage the first one
/// found, and a third finding more means the index is being damaged as it is
/// checked.
const MAX_PASSES: u32 = 3;

/// How long the maintenance thread waits for a report before it looks at
/// the periodic checks again.
const MAINTENANCE_TICK: Duration = Duration::from_secs(1);

/// How often a waiter whose executor lost the check looks at the record.
const LOST_POLL: Duration = Duration::from_millis(20);

/// Where an executor of this process stands with one check.
pub(super) enum CheckSlot {
    /// An executor thread has it.
    Running,
    /// The executor reached this outcome.
    Done(CheckOutcome),
    /// The executor lost the check to another one, whose outcome the record
    /// will carry.
    Lost,
}

/// The outcome of a check, as a waiter receives it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckOutcome {
    /// Every entry agrees with its record; the generation is verified.
    Verified {
        /// Records and entries checked.
        checked: u64,
        /// Disagreements repaired on the way.
        repaired: u64,
    },
    /// Records themselves break the unique constraint (two hold one value):
    /// no entry can represent them, the generation stays suspect, and the
    /// records are left as they are.
    SourceConflicts {
        /// The conflicts found.
        conflicts: Vec<Mismatch>,
    },
    /// The damage was too wide to repair entry by entry: the index is being
    /// rebuilt into `generation`.
    Rebuilding {
        /// The generation the rebuild fills.
        generation: GenerationId,
    },
    /// The check could not finish.
    Failed(String),
    /// The check was cancelled, or its generation stopped serving.
    Cancelled,
}

/// One check as inspection shows it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckStatus {
    /// The checked generation, the check's identity.
    pub generation: GenerationId,
    /// Its integrity record.
    pub record: IndexIntegrityRecord,
    /// Whether an executor of this process runs it now.
    pub running_here: bool,
}

/// What an executor took: the definition it checks, the version of its
/// record every repair is bound to, and its token.
struct Taken {
    def: IndexDefinition,
    def_version: Option<u64>,
    token: u64,
}

/// How a check's executor ended.
enum Executed {
    Outcome(CheckOutcome),
    Lost,
}

/// What one page of a check found.
#[derive(Default)]
struct Found {
    checked: u64,
    mismatches: Vec<Mismatch>,
}

/// What one pass of a check did.
#[derive(Default)]
struct Pass {
    checked: u64,
    found: u64,
    repaired: u64,
    conflicts: Vec<Mismatch>,
}

impl IndexBuildService {
    /// Start the maintenance thread: it records the disagreements reads and
    /// writes report, admits a check of each generation they name, and asks
    /// for the periodic check of every B-tree index. Stopped by
    /// [`Self::shutdown`].
    ///
    /// # Errors
    ///
    /// The thread could not be started.
    pub fn start_maintenance(&self) -> Result<(), String> {
        let shared = Arc::clone(&self.shared);
        let thread = std::thread::Builder::new()
            .name("index-integrity".into())
            .spawn(move || {
                let service = IndexBuildService { shared };
                while !service
                    .shared
                    .stopping
                    .load(core::sync::atomic::Ordering::Acquire)
                {
                    let reports = service.shared.env.registry().take_reports(MAINTENANCE_TICK);
                    if !reports.is_empty() {
                        service.record_reports(reports);
                    }
                    service.sweep();
                }
            })
            .map_err(|e| format!("start the index maintenance thread: {e}"))?;
        self.shared.executors.lock().push(thread);
        Ok(())
    }

    /// Record `reports` in the integrity records of their generations, and
    /// admit and run a check of each. A member that takes no writes records
    /// nothing; its own reads keep answering from the records.
    fn record_reports(&self, reports: Vec<IntegrityReport>) {
        let mut by_generation: FxHashMap<GenerationId, (IndexId, Vec<Mismatch>)> =
            FxHashMap::default();
        for report in reports {
            by_generation
                .entry(report.generation)
                .or_insert_with(|| (report.index, Vec::new()))
                .1
                .push(report.found);
        }
        for (generation, (index, found)) in by_generation {
            match self.record(index, generation, found) {
                Ok(true) => {
                    if let Err(e) = self.submit_check(generation) {
                        tracing::warn!(generation = generation.as_raw(), error = %e,
                            "could not start the check of a reported index generation");
                    }
                }
                Ok(false) => {}
                Err(e) => tracing::warn!(generation = generation.as_raw(), error = %e,
                    "could not record an index disagreement; this member answers from the \
                     records until a check runs"),
            }
        }
    }

    /// Record `found` against `generation` of `index` and admit a check
    /// unless one is pending. `false` when nothing was recorded: the index
    /// no longer serves from that generation, or this member takes no
    /// writes.
    fn record(
        &self,
        index: IndexId,
        generation: GenerationId,
        found: Vec<Mismatch>,
    ) -> Result<bool, String> {
        let env = self.shared.env.as_ref();
        let store = LocalIndexStore::new(env.engine());
        let serving = store
            .load_definition(index)
            .map_err(text)?
            .is_some_and(|d| d.generation == generation);
        if !serving {
            return Ok(false);
        }
        for _ in 0..MOVE_ATTEMPTS {
            let (mut record, version) = match store.load_integrity(generation).map_err(text)? {
                Some((record, version)) => (record, Some(version)),
                None => (IndexIntegrityRecord::new(index, generation), None),
            };
            let mut new = false;
            for mismatch in &found {
                new |= record.report(mismatch.clone());
            }
            if !new && record.integrity == Integrity::Suspect && record.check_pending() {
                return Ok(true);
            }
            if !record.check_pending() {
                record.check = Some(IndexCheck::accepted(record.evidence_revision));
            }
            let committed = {
                let mut txn = begin(env);
                store
                    .put_integrity_txn(&mut txn, &record, version)
                    .map_err(text)?;
                env.commit_catalog(&mut txn)
            };
            match committed {
                Ok(()) => {
                    self.refresh_integrity()?;
                    return Ok(true);
                }
                Err(e) if is_contention(&e) => continue,
                Err(e) if cannot_move_here(&e) => return Ok(false),
                Err(e) => return Err(e.to_string()),
            }
        }
        Err(format!(
            "the integrity record of generation {} kept moving under a report",
            generation.as_raw()
        ))
    }

    /// Bring this process's registry in line with the integrity records.
    fn refresh_integrity(&self) -> Result<(), String> {
        let store = LocalIndexStore::new(self.shared.env.engine());
        let records = store.list_integrity().map_err(text)?;
        self.shared.env.registry().apply_integrity(&records);
        Ok(())
    }

    /// Ask for the check of every B-tree index whose generation has not been
    /// checked within the configured interval. The first sweep of a process
    /// waits one interval, so a restart does not check every index at once.
    fn sweep(&self) {
        let Some(interval) = self.shared.config.read().check_interval else {
            return;
        };
        {
            let mut last = self.shared.last_sweep.lock();
            match *last {
                Some(at) if at.elapsed() < interval => return,
                Some(_) => *last = Some(Instant::now()),
                None => {
                    *last = Some(Instant::now());
                    return;
                }
            }
        }
        let store = LocalIndexStore::new(self.shared.env.engine());
        let now = wall_clock_ms();
        let interval_ms = u64::try_from(interval.as_millis()).unwrap_or(u64::MAX);
        for def in self.shared.env.registry().all() {
            if def.index_type != IndexType::BTree || def.state != IndexState::Ready {
                continue;
            }
            let recent = match store.load_integrity(def.generation) {
                Ok(Some((record, _))) => record.check.as_ref().is_some_and(|c| {
                    !c.state.is_terminal()
                        || c.finished_at_ms
                            .is_some_and(|at| now.saturating_sub(at) < interval_ms)
                }),
                Ok(None) => false,
                Err(e) => {
                    tracing::warn!(index = %def, error = %e, "could not read an index's integrity record");
                    continue;
                }
            };
            if recent {
                continue;
            }
            match self.request_check(def.id) {
                Ok(_) => {}
                // Not this member's to check: the one that takes writes does.
                Err(CheckRequestError::NotHere) => return,
                Err(e) => {
                    tracing::warn!(index = %def, error = %e, "could not start a periodic index check")
                }
            }
        }
    }

    /// Admit a check of the generation the index `index` serves from and run
    /// it on an executor of this process; a check already admitted is run as
    /// it is. Returns the generation, which identifies the check.
    ///
    /// # Errors
    ///
    /// The index is not a ready B-tree index, this member takes no writes, or
    /// the catalog could not be read or written.
    pub fn request_check(&self, index: IndexId) -> Result<GenerationId, CheckRequestError> {
        let env = self.shared.env.as_ref();
        let store = LocalIndexStore::new(env.engine());
        let def = store
            .load_definition(index)
            .map_err(|e| CheckRequestError::Other(e.to_string()))?
            .ok_or(CheckRequestError::NoSuchIndex)?;
        if def.index_type != IndexType::BTree {
            return Err(CheckRequestError::NotChecked(def.to_string()));
        }
        if def.state != IndexState::Ready {
            return Err(CheckRequestError::NotReady(def.to_string()));
        }
        let generation = def.generation;
        for _ in 0..MOVE_ATTEMPTS {
            let (mut record, version) = match store
                .load_integrity(generation)
                .map_err(|e| CheckRequestError::Other(e.to_string()))?
            {
                Some((record, version)) => (record, Some(version)),
                None => (IndexIntegrityRecord::new(index, generation), None),
            };
            if record.check_pending() {
                self.submit_check(generation)
                    .map_err(CheckRequestError::Other)?;
                return Ok(generation);
            }
            record.check = Some(IndexCheck::accepted(record.evidence_revision));
            let committed = {
                let mut txn = begin(env);
                store
                    .put_integrity_txn(&mut txn, &record, version)
                    .map_err(|e| CheckRequestError::Other(e.to_string()))?;
                env.commit_catalog(&mut txn)
            };
            match committed {
                Ok(()) => {
                    self.submit_check(generation)
                        .map_err(CheckRequestError::Other)?;
                    return Ok(generation);
                }
                Err(e) if is_contention(&e) => continue,
                Err(e) if cannot_move_here(&e) => return Err(CheckRequestError::NotHere),
                Err(e) => return Err(CheckRequestError::Other(e.to_string())),
            }
        }
        Err(CheckRequestError::Other(format!(
            "the integrity record of generation {} kept moving under the request",
            generation.as_raw()
        )))
    }

    /// Run the check of `generation` on an executor of this process, unless
    /// one already has it.
    ///
    /// # Errors
    ///
    /// The executor thread could not be started.
    pub fn submit_check(&self, generation: GenerationId) -> Result<(), String> {
        if self
            .shared
            .stopping
            .load(core::sync::atomic::Ordering::Acquire)
        {
            return Err("the index build service is shutting down".into());
        }
        {
            let mut checks = self.shared.checks.lock();
            if matches!(checks.get(&generation), Some(CheckSlot::Running)) {
                return Ok(());
            }
            checks.insert(generation, CheckSlot::Running);
        }
        let shared = Arc::clone(&self.shared);
        let spawned = std::thread::Builder::new()
            .name(format!("index-check-{}", generation.as_raw()))
            .spawn(move || {
                let executed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    shared.execute_check(generation)
                }))
                .unwrap_or_else(|panic| {
                    Executed::Outcome(CheckOutcome::Failed(panic_message(&*panic)))
                });
                let slot = match executed {
                    Executed::Outcome(outcome) => CheckSlot::Done(outcome),
                    Executed::Lost => CheckSlot::Lost,
                };
                let mut checks = shared.checks.lock();
                checks.insert(generation, slot);
                shared.check_done.notify_all();
            });
        match spawned {
            Ok(executor) => {
                let mut executors = self.shared.executors.lock();
                executors.retain(|e| !e.is_finished());
                executors.push(executor);
                Ok(())
            }
            Err(e) => {
                self.shared.checks.lock().remove(&generation);
                Err(format!("start the check executor: {e}"))
            }
        }
    }

    /// The outcome of the check of `generation`, waiting up to `timeout`
    /// (`None`: until it has one). `None` when it has none yet. Waiting
    /// cancels nothing.
    ///
    /// # Errors
    ///
    /// The integrity record could not be read.
    pub fn wait_check(
        &self,
        generation: GenerationId,
        timeout: Option<Duration>,
    ) -> Result<Option<CheckOutcome>, StoreError> {
        let deadline = timeout.map(|t| Instant::now() + t);
        {
            let mut checks = self.shared.checks.lock();
            loop {
                match checks.get(&generation) {
                    Some(CheckSlot::Done(outcome)) => return Ok(Some(outcome.clone())),
                    Some(CheckSlot::Running) => {}
                    Some(CheckSlot::Lost) | None => break,
                }
                match deadline {
                    Some(at) => {
                        if self
                            .shared
                            .check_done
                            .wait_until(&mut checks, at)
                            .timed_out()
                        {
                            return Ok(None);
                        }
                    }
                    None => self.shared.check_done.wait(&mut checks),
                }
            }
        }
        let store = LocalIndexStore::new(self.shared.env.engine());
        loop {
            let outcome = match store.load_integrity(generation)? {
                None => Some(CheckOutcome::Cancelled),
                Some((record, _)) => terminal_outcome(&record),
            };
            if outcome.is_some() {
                return Ok(outcome);
            }
            let Some(at) = deadline else {
                std::thread::sleep(LOST_POLL);
                continue;
            };
            let now = Instant::now();
            if now >= at {
                return Ok(None);
            }
            std::thread::sleep(LOST_POLL.min(at - now));
        }
    }

    /// Cancel the check of `generation`. `false` when it already had an
    /// outcome or none was admitted. An executor running it stops at its
    /// next page.
    ///
    /// # Errors
    ///
    /// A storage failure, a commit failure other than a conflict, or moves
    /// kept winning [`MOVE_ATTEMPTS`] times in a row.
    pub fn cancel_check(&self, generation: GenerationId) -> Result<bool, String> {
        let env = self.shared.env.as_ref();
        let store = LocalIndexStore::new(env.engine());
        for _ in 0..MOVE_ATTEMPTS {
            let Some((mut record, version)) = store.load_integrity(generation).map_err(text)?
            else {
                return Ok(false);
            };
            let Some(check) = record.check.as_mut() else {
                return Ok(false);
            };
            if check.state.is_terminal() {
                return Ok(false);
            }
            check.state = CheckState::Cancelled;
            check.finished_at_ms = Some(wall_clock_ms());
            let committed = {
                let mut txn = begin(env);
                store
                    .put_integrity_txn(&mut txn, &record, Some(version))
                    .map_err(text)?;
                env.commit_catalog(&mut txn)
            };
            match committed {
                Ok(()) => return Ok(true),
                Err(e) if is_contention(&e) => continue,
                Err(e) => return Err(e.to_string()),
            }
        }
        Err(format!(
            "the check of generation {} kept moving under the cancellation",
            generation.as_raw()
        ))
    }

    /// Every integrity record, each with whether an executor of this
    /// process runs its check now.
    ///
    /// # Errors
    ///
    /// The records could not be read.
    pub fn checks(&self) -> Result<Vec<CheckStatus>, StoreError> {
        let store = LocalIndexStore::new(self.shared.env.engine());
        let running = self.shared.checks.lock();
        Ok(store
            .list_integrity()?
            .into_iter()
            .map(|record| CheckStatus {
                generation: record.generation,
                running_here: matches!(running.get(&record.generation), Some(CheckSlot::Running)),
                record,
            })
            .collect())
    }

    /// Take up every check the catalog records without an outcome, as after
    /// a restart. Returns their generations.
    ///
    /// # Errors
    ///
    /// The records could not be read, or an executor could not start.
    pub fn resume_checks(&self) -> Result<Vec<GenerationId>, String> {
        let store = LocalIndexStore::new(self.shared.env.engine());
        let mut resumed = Vec::new();
        for record in store.list_integrity().map_err(text)? {
            if record.check_pending() {
                self.submit_check(record.generation)?;
                resumed.push(record.generation);
            }
        }
        self.refresh_integrity()?;
        Ok(resumed)
    }

    /// Rebuild the B-tree index `def` from the records into a fresh
    /// generation of the same index, on an executor of this process: the
    /// definition is published building in the new generation, with the
    /// entries of the generation it served from removed and the build
    /// admitted, in one catalog commit conditioned on the definition as
    /// read. Writers maintain the new generation from then on, and unique
    /// values are proved free from the records until it is built. A build the
    /// records refuse keeps the index, marked failed, so its constraint
    /// still holds for new writes while lookups stop using it. Returns the
    /// definition as published.
    ///
    /// # Errors
    ///
    /// The catalog could not be read or written, or the executor could not
    /// start.
    pub fn rebuild(&self, mut def: IndexDefinition) -> Result<IndexDefinition, String> {
        use coordinode_modality::ENTRY_LAYOUT;
        let env = self.shared.env.as_ref();
        let store = LocalIndexStore::new(env.engine());
        def.layout = ENTRY_LAYOUT;
        def.state = IndexState::Building {
            written: 0,
            estimated_total: 0,
        };
        let replaced = store.definition_version(def.id).map_err(text)?;
        let retired = def.generation;
        let committed = {
            let mut txn = begin(env);
            store
                .expect_definition_txn(&mut txn, def.id, replaced)
                .map_err(text)?;
            def.generation = store.allocate_generation_txn(&mut txn).map_err(text)?;
            store.clear_txn(&mut txn, retired).map_err(text)?;
            store.put_definition_txn(&mut txn, &def).map_err(text)?;
            store
                .put_build_txn(
                    &mut txn,
                    &IndexBuildRecord::accepted(def.id, def.generation, BuildFailure::Keep),
                    None,
                )
                .map_err(text)?;
            env.commit_catalog(&mut txn)
        };
        committed.map_err(|e| e.to_string())?;
        env.registry()
            .register_published(env.engine(), def.clone())
            .map_err(|e| e.to_string())?;
        self.submit(def.generation)?;
        Ok(def)
    }
}

/// Why a check could not be admitted.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CheckRequestError {
    /// No index has that identity.
    #[error("no such index")]
    NoSuchIndex,
    /// The index is of a kind a check does not cover.
    #[error("index '{0}' is not a B-tree index; only B-tree indexes are checked")]
    NotChecked(String),
    /// The index is still being built or its build failed.
    #[error("index '{0}' is not ready; it is checked once it is built")]
    NotReady(String),
    /// This member takes no writes; the member that does runs checks.
    #[error("this member takes no writes; the leader runs index checks")]
    NotHere,
    /// Any other failure.
    #[error("{0}")]
    Other(String),
}

/// The waiter's view of a record's latest check, `None` while it runs.
fn terminal_outcome(record: &IndexIntegrityRecord) -> Option<CheckOutcome> {
    let check = record.check.as_ref()?;
    match &check.state {
        CheckState::Accepted | CheckState::Running { .. } => None,
        CheckState::Cancelled => Some(CheckOutcome::Cancelled),
        CheckState::Failed { reason } => Some(CheckOutcome::Failed(reason.clone())),
        CheckState::Done => Some(match (check.rebuilt_into, record.integrity) {
            (Some(generation), _) => CheckOutcome::Rebuilding { generation },
            (None, Integrity::Verified) => CheckOutcome::Verified {
                checked: check.checked,
                repaired: check.repaired,
            },
            (None, _) => CheckOutcome::SourceConflicts {
                conflicts: record
                    .evidence
                    .iter()
                    .filter(|m| matches!(m, Mismatch::SourceDuplicate { .. }))
                    .cloned()
                    .collect(),
            },
        }),
    }
}

/// Milliseconds since the epoch by the wall clock; 0 when the clock reads
/// before it.
fn wall_clock_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

/// Whether `record`, a stored state of node `node` (or one version of it),
/// gives `def` an entry under `tuple`, by the index's own interpretation.
fn record_holds(
    def: &IndexDefinition,
    interpretation: &IndexInterpretation,
    record: &NodeRecord,
    tuple: &[u8],
) -> bool {
    record.primary_label() == def.label
        && interpretation
            .record_membership(record)
            .is_some_and(|held| tuples(&held).iter().any(|t| t.as_slice() == tuple))
}

/// Whether node `node` holds `tuple` in `def`, as `txn` sees it: its own
/// record, or any version of a temporal node.
fn node_holds(
    txn: &Transaction<'_>,
    shard_id: u16,
    def: &IndexDefinition,
    interpretation: &IndexInterpretation,
    node: NodeId,
    tuple: &[u8],
) -> Result<bool, StoreError> {
    if LocalNodeStore
        .get(txn, shard_id, node)?
        .is_some_and(|r| record_holds(def, interpretation, &r, tuple))
    {
        return Ok(true);
    }
    Ok(LocalNodeStore
        .versions(txn, shard_id, node)?
        .iter()
        .any(|(_, r)| record_holds(def, interpretation, r, tuple)))
}

/// The owner a mismatch names.
fn owner_of(node: u64, valid_from: Option<i64>) -> EntryOwner {
    EntryOwner {
        node_id: node,
        valid_from,
    }
}

impl Shared {
    /// Take the check of `generation`, run it to its outcome and record it.
    fn execute_check(self: &Arc<Self>, generation: GenerationId) -> Executed {
        let _seat = self.quiet_seat();
        if self.stopping.load(core::sync::atomic::Ordering::Acquire) {
            return Executed::Lost;
        }
        match self.take_check(generation) {
            Ok(Ok(taken)) => self.run_check(generation, &taken).unwrap_or_else(|e| {
                match self.conclude_failed(generation, &taken, &e) {
                    Ok(executed) => executed,
                    Err(e) => Executed::Outcome(CheckOutcome::Failed(e)),
                }
            }),
            Ok(Err(executed)) => executed,
            Err(e) => Executed::Outcome(CheckOutcome::Failed(e)),
        }
    }

    /// Take the check of `generation`: its state moves to running under a
    /// fresh token, conditioned on the record as read. A check whose index
    /// no longer serves from the generation is cancelled instead.
    fn take_check(&self, generation: GenerationId) -> Result<Result<Taken, Executed>, String> {
        let env = self.env.as_ref();
        let store = LocalIndexStore::new(env.engine());
        for _ in 0..MOVE_ATTEMPTS {
            let Some((mut record, version)) = store.load_integrity(generation).map_err(text)?
            else {
                return Ok(Err(Executed::Outcome(CheckOutcome::Cancelled)));
            };
            if let Some(outcome) = terminal_outcome(&record) {
                return Ok(Err(Executed::Outcome(outcome)));
            }
            let def = store
                .load_definition(record.index)
                .map_err(text)?
                .filter(|d| {
                    d.generation == generation
                        && d.index_type == IndexType::BTree
                        && d.state == IndexState::Ready
                });
            let def_version = store.definition_version(record.index).map_err(text)?;
            let token = next_token();
            let Some(check) = record.check.as_mut() else {
                return Ok(Err(Executed::Outcome(CheckOutcome::Cancelled)));
            };
            if check.state == CheckState::Accepted {
                check.started_revision = record.evidence_revision;
            }
            check.state = match &def {
                Some(_) => CheckState::Running { executor: token },
                None => CheckState::Cancelled,
            };
            let committed = {
                let mut txn = begin(env);
                store
                    .put_integrity_txn(&mut txn, &record, Some(version))
                    .map_err(text)?;
                env.commit_catalog(&mut txn)
            };
            match (committed, def) {
                (Ok(()), Some(def)) => {
                    return Ok(Ok(Taken {
                        def,
                        def_version,
                        token,
                    }));
                }
                (Ok(()), None) => return Ok(Err(Executed::Outcome(CheckOutcome::Cancelled))),
                (Err(e), _) if is_contention(&e) => continue,
                (Err(e), _) if cannot_move_here(&e) => return Ok(Err(Executed::Lost)),
                (Err(e), _) => return Err(e.to_string()),
            }
        }
        Err(format!(
            "the check of generation {} kept moving while it was being taken",
            generation.as_raw()
        ))
    }

    /// Run the passes of the check `taken` holds and record its outcome.
    fn run_check(
        self: &Arc<Self>,
        generation: GenerationId,
        taken: &Taken,
    ) -> Result<Executed, String> {
        let config = *self.config.read();
        let mut repaired_total = 0u64;
        loop {
            let pass = match self.pass(
                generation,
                taken,
                config.check_page,
                config.check_max_repairs,
                &mut repaired_total,
            )? {
                Some(pass) => pass,
                None => return self.lost_check(generation, taken),
            };
            let Some(record) = self.held_record(generation, taken)? else {
                return self.lost_check(generation, taken);
            };
            let passes = record.0.check.as_ref().map_or(0, |c| c.passes);
            // Every disagreement repaired, none left: the generation agrees
            // with its records, unless a report arrived meanwhile, which the
            // next pass looks for.
            if pass.found == 0 && pass.conflicts.is_empty() {
                let started = record.0.check.as_ref().map_or(0, |c| c.started_revision);
                if record.0.evidence_revision == started {
                    return self.conclude_check(generation, taken, |rec, check| {
                        rec.integrity = Integrity::Verified;
                        rec.evidence.clear();
                        check.state = CheckState::Done;
                    });
                }
                self.checkpoint(generation, taken, |rec, check| {
                    check.started_revision = rec.evidence_revision;
                })?;
            } else if pass.found == pass.conflicts.len() as u64 {
                // Only records breaking the constraint remain: no entry can
                // fix them.
                let conflicts = pass.conflicts.clone();
                return self.conclude_check(generation, taken, move |rec, check| {
                    for conflict in &conflicts {
                        rec.report(conflict.clone());
                    }
                    rec.integrity = Integrity::Suspect;
                    check.state = CheckState::Done;
                });
            }
            if repaired_total > config.check_max_repairs || passes >= MAX_PASSES {
                return self.escalate(generation, taken);
            }
        }
    }

    /// One complete pass over both halves, resuming from the record's cursor:
    /// what it found and repaired. `None` when the check stopped being this
    /// executor's.
    fn pass(
        &self,
        generation: GenerationId,
        taken: &Taken,
        page: usize,
        max_repairs: u64,
        repaired_total: &mut u64,
    ) -> Result<Option<Pass>, String> {
        let env = self.env.as_ref();
        let field_of_fields = env.fields()?;
        let field_of = |name: &str| field_of_fields.lookup(name);
        let interpretation = taken.def.interpretation(&field_of);
        let mut pass = Pass::default();
        loop {
            if self.stopping.load(core::sync::atomic::Ordering::Acquire) {
                return Ok(None);
            }
            let Some((record, _)) = self.held_record(generation, taken)? else {
                return Ok(None);
            };
            let Some(check) = record.check.clone() else {
                return Ok(None);
            };
            let (found, next, exhausted) = match check.phase {
                CheckPhase::Records => {
                    self.records_page(taken, &interpretation, check.cursor.as_deref(), page)?
                }
                CheckPhase::Entries => {
                    self.entries_page(taken, &interpretation, check.cursor.as_deref(), page)?
                }
            };
            let mut repaired = 0u64;
            let mut conflicts = Vec::new();
            let mut repairable = Vec::new();
            for mismatch in found.mismatches {
                match mismatch {
                    Mismatch::SourceDuplicate { .. } => conflicts.push(mismatch),
                    other => repairable.push(other),
                }
            }
            // Counts of rows and entries a check reads: far below the type's
            // range.
            pass.checked += found.checked;
            pass.found += (repairable.len() + conflicts.len()) as u64;
            if !repairable.is_empty() && *repaired_total <= max_repairs {
                repaired = self.repair(taken, &interpretation, &repairable, &mut conflicts)?;
                *repaired_total += repaired;
                pass.repaired += repaired;
            }
            pass.conflicts.extend(conflicts.iter().cloned());
            let phase_done = exhausted;
            let moved = self.checkpoint(generation, taken, |rec, check| {
                check.checked += found.checked;
                check.mismatches += (repairable.len() + conflicts.len()) as u64;
                check.repaired += repaired;
                for conflict in &conflicts {
                    rec.report(conflict.clone());
                }
                if phase_done {
                    check.cursor = None;
                    match check.phase {
                        CheckPhase::Records => check.phase = CheckPhase::Entries,
                        CheckPhase::Entries => {
                            check.phase = CheckPhase::Records;
                            check.passes += 1;
                        }
                    }
                } else {
                    check.cursor = next.clone();
                }
            })?;
            if !moved {
                return Ok(None);
            }
            if phase_done && check.phase == CheckPhase::Entries {
                return Ok(Some(pass));
            }
        }
    }

    /// One page of records of the checked label after `after`, each checked
    /// for the entries its values give it: what was wrong, the key the next
    /// page resumes after, and whether the records are exhausted.
    fn records_page(
        &self,
        taken: &Taken,
        interpretation: &IndexInterpretation,
        after: Option<&[u8]>,
        page: usize,
    ) -> Result<(Found, Option<Vec<u8>>, bool), String> {
        let env = self.env.as_ref();
        let shard_id = env.shard_id();
        let def = &taken.def;
        let store = LocalIndexStore::new(env.engine());
        let mut txn = begin(env);
        let rows = LocalNodeStore
            .rows_page(&mut txn, shard_id, after, page)
            .map_err(text)?;
        let mut found = Found::default();
        for (owner, record) in &rows.rows {
            if record.primary_label() != def.label {
                continue;
            }
            // A count of the rows a check reads, far below the type's range.
            found.checked += 1;
            let Some(values) = interpretation.record_membership(record) else {
                continue;
            };
            let node = NodeId::from_raw(owner.node_id);
            for tuple in tuples(&values) {
                let missing = if def.unique {
                    match store.unique_holder(&txn, def, &tuple).map_err(text)? {
                        Some(Some(h)) if h == node => false,
                        Some(Some(h))
                            if node_holds(&txn, shard_id, def, interpretation, h, &tuple)
                                .map_err(text)? =>
                        {
                            found.mismatches.push(Mismatch::SourceDuplicate {
                                nodes: [
                                    h.as_raw().min(owner.node_id),
                                    h.as_raw().max(owner.node_id),
                                ],
                                tuple,
                            });
                            continue;
                        }
                        _ => true,
                    }
                } else {
                    !store.has_entry(&txn, def, &tuple, *owner).map_err(text)?
                };
                if missing {
                    found.mismatches.push(Mismatch::Missing {
                        node: owner.node_id,
                        valid_from: owner.valid_from,
                        tuple,
                    });
                }
            }
        }
        Ok((found, rows.resume, rows.exhausted))
    }

    /// One page of the generation's entries after `after`, each checked for
    /// a record holding its value.
    fn entries_page(
        &self,
        taken: &Taken,
        interpretation: &IndexInterpretation,
        after: Option<&[u8]>,
        page: usize,
    ) -> Result<(Found, Option<Vec<u8>>, bool), String> {
        let env = self.env.as_ref();
        let shard_id = env.shard_id();
        let def = &taken.def;
        let store = LocalIndexStore::new(env.engine());
        let mut txn = begin(env);
        let entries = store
            .entries_page(&mut txn, def, after, page)
            .map_err(text)?;
        let mut found = Found::default();
        for entry in &entries.entries {
            // A count of the entries a check reads, far below the type's
            // range.
            found.checked += 1;
            // An entry whose holder does not decode names no node: it is as
            // wrong as one naming a node that does not hold the value.
            let Some(owner) = entry.owner else {
                found.mismatches.push(Mismatch::Extra {
                    node: u64::MAX,
                    valid_from: None,
                    tuple: entry.tuple.clone(),
                });
                continue;
            };
            let holds = if def.unique {
                node_holds(
                    &txn,
                    shard_id,
                    def,
                    interpretation,
                    NodeId::from_raw(owner.node_id),
                    &entry.tuple,
                )
                .map_err(text)?
            } else {
                LocalNodeStore
                    .row(&txn, shard_id, owner)
                    .map_err(text)?
                    .is_some_and(|r| record_holds(def, interpretation, &r, &entry.tuple))
            };
            if !holds {
                found.mismatches.push(Mismatch::Extra {
                    node: owner.node_id,
                    valid_from: owner.valid_from,
                    tuple: entry.tuple.clone(),
                });
            }
        }
        Ok((found, entries.resume, entries.exhausted))
    }

    /// Repair `mismatches` in one commit, each only while its record and its
    /// entry are still as found and the index still serves from the checked
    /// generation. A missing unique entry whose value another record holds
    /// is a conflict, appended to `conflicts`. Returns how many were
    /// repaired; a repair that met a concurrent change is left for the next
    /// pass.
    fn repair(
        &self,
        taken: &Taken,
        interpretation: &IndexInterpretation,
        mismatches: &[Mismatch],
        conflicts: &mut Vec<Mismatch>,
    ) -> Result<u64, String> {
        let env = self.env.as_ref();
        let engine = env.engine();
        let shard_id = env.shard_id();
        let def = &taken.def;
        let store = LocalIndexStore::new(engine);
        let latest = Transaction::new(
            engine,
            None,
            coordinode_core::txn::timestamp::Timestamp::ZERO,
            None,
        );
        let mut txn = begin(env);
        // Conditions before the writes they guard: a direct-mode engine
        // decides a condition when it is stated.
        store
            .expect_definition_txn(&mut txn, def.id, taken.def_version)
            .map_err(text)?;
        let mut repaired = 0u64;
        for mismatch in mismatches {
            match mismatch {
                Mismatch::Missing {
                    node,
                    valid_from,
                    tuple,
                } => {
                    let owner = owner_of(*node, *valid_from);
                    // The record as it stands now, not as the page read it:
                    // a record that moved off the value needs no entry.
                    let Some((record, row_version)) = LocalNodeStore
                        .latest_row(engine, shard_id, owner)
                        .map_err(text)?
                    else {
                        continue;
                    };
                    if !record_holds(def, interpretation, &record, tuple) {
                        continue;
                    }
                    let current = store.latest_entry(def, tuple, owner).map_err(text)?;
                    match current {
                        // The entry is there by now: nothing to repair.
                        Some(_) if !def.unique => continue,
                        Some(LatestEntry {
                            holder: Some(h), ..
                        }) if h.as_raw() == *node => continue,
                        Some(LatestEntry {
                            holder: Some(h), ..
                        }) if node_holds(&latest, shard_id, def, interpretation, h, tuple)
                            .map_err(text)? =>
                        {
                            conflicts.push(Mismatch::SourceDuplicate {
                                nodes: [h.as_raw().min(*node), h.as_raw().max(*node)],
                                tuple: tuple.clone(),
                            });
                            continue;
                        }
                        _ => {}
                    }
                    LocalNodeStore
                        .expect_row_txn(&mut txn, shard_id, owner, Some(row_version))
                        .map_err(text)?;
                    store
                        .repair_entry_txn(
                            &mut txn,
                            def,
                            tuple,
                            owner,
                            true,
                            current.map(|e| e.version),
                        )
                        .map_err(text)?;
                    repaired += 1;
                }
                Mismatch::Extra {
                    node,
                    valid_from,
                    tuple,
                } => {
                    let owner = owner_of(*node, *valid_from);
                    let Some(current) = store.latest_entry(def, tuple, owner).map_err(text)? else {
                        continue;
                    };
                    // Still the entry the check found: naming the same node,
                    // or still naming none it can decode.
                    if def.unique && current.holder.map_or(u64::MAX, NodeId::as_raw) != *node {
                        continue;
                    }
                    let holds = if def.unique {
                        node_holds(
                            &latest,
                            shard_id,
                            def,
                            interpretation,
                            NodeId::from_raw(*node),
                            tuple,
                        )
                        .map_err(text)?
                    } else {
                        LocalNodeStore
                            .latest_row(engine, shard_id, owner)
                            .map_err(text)?
                            .is_some_and(|(r, _)| record_holds(def, interpretation, &r, tuple))
                    };
                    if holds {
                        continue;
                    }
                    // A writer giving the node the value back writes this
                    // same entry, which moves its version and refuses the
                    // removal.
                    store
                        .repair_entry_txn(&mut txn, def, tuple, owner, false, Some(current.version))
                        .map_err(text)?;
                    repaired += 1;
                }
                Mismatch::SourceDuplicate { .. } => {}
            }
        }
        if repaired == 0 {
            return Ok(0);
        }
        match env.commit_page(&mut txn) {
            Ok(()) => Ok(repaired),
            Err(e) if is_contention(&e) => Ok(0),
            Err(e) => Err(e.to_string()),
        }
    }

    /// The integrity record of `generation` while its check is still the
    /// one `taken` holds.
    fn held_record(
        &self,
        generation: GenerationId,
        taken: &Taken,
    ) -> Result<Option<(IndexIntegrityRecord, u64)>, String> {
        let store = LocalIndexStore::new(self.env.engine());
        Ok(store
            .load_integrity(generation)
            .map_err(text)?
            .filter(|(record, _)| {
                record.check.as_ref().is_some_and(|c| {
                    c.state
                        == CheckState::Running {
                            executor: taken.token,
                        }
                })
            }))
    }

    /// Move the held check's record as `update` says, conditioned on the
    /// record as read and retried while the check is still this
    /// executor's. `false` when it no longer is.
    fn checkpoint(
        &self,
        generation: GenerationId,
        taken: &Taken,
        mut update: impl FnMut(&mut IndexIntegrityRecord, &mut IndexCheck),
    ) -> Result<bool, String> {
        let env = self.env.as_ref();
        let store = LocalIndexStore::new(env.engine());
        for _ in 0..MOVE_ATTEMPTS {
            let Some((mut record, version)) = self.held_record(generation, taken)? else {
                return Ok(false);
            };
            let Some(mut check) = record.check.take() else {
                return Ok(false);
            };
            update(&mut record, &mut check);
            record.check = Some(check);
            let committed = {
                let mut txn = begin(env);
                store
                    .put_integrity_txn(&mut txn, &record, Some(version))
                    .map_err(text)?;
                env.commit_catalog(&mut txn)
            };
            match committed {
                Ok(()) => return Ok(true),
                Err(e) if is_contention(&e) => continue,
                Err(e) if cannot_move_here(&e) => return Ok(false),
                Err(e) => return Err(e.to_string()),
            }
        }
        Err(format!(
            "the check of generation {} kept moving under its executor",
            generation.as_raw()
        ))
    }

    /// Record the held check's outcome as `update` sets it.
    fn conclude_check(
        &self,
        generation: GenerationId,
        taken: &Taken,
        mut update: impl FnMut(&mut IndexIntegrityRecord, &mut IndexCheck),
    ) -> Result<Executed, String> {
        let now = wall_clock_ms();
        let landed = self.checkpoint(generation, taken, |record, check| {
            update(record, check);
            check.finished_at_ms = Some(now);
        })?;
        let env = self.env.as_ref();
        let store = LocalIndexStore::new(env.engine());
        env.registry()
            .apply_integrity(&store.list_integrity().map_err(text)?);
        if !landed {
            return self.lost_check(generation, taken);
        }
        let record = store.load_integrity(generation).map_err(text)?;
        Ok(Executed::Outcome(
            record
                .and_then(|(record, _)| terminal_outcome(&record))
                .unwrap_or(CheckOutcome::Cancelled),
        ))
    }

    /// Record that the held check failed with `reason`.
    fn conclude_failed(
        &self,
        generation: GenerationId,
        taken: &Taken,
        reason: &str,
    ) -> Result<Executed, String> {
        self.conclude_check(generation, taken, |_, check| {
            check.state = CheckState::Failed {
                reason: reason.to_string(),
            };
        })
    }

    /// Rebuild the checked index into a fresh generation and record that as
    /// the check's outcome.
    fn escalate(
        self: &Arc<Self>,
        generation: GenerationId,
        taken: &Taken,
    ) -> Result<Executed, String> {
        let service = IndexBuildService {
            shared: Arc::clone(self),
        };
        let rebuilt = service.rebuild(taken.def.clone())?;
        tracing::warn!(
            index = %taken.def,
            from = generation.as_raw(),
            into = rebuilt.generation.as_raw(),
            "an index disagreed with its records too widely to repair entry by entry; \
             rebuilding it"
        );
        self.conclude_check(generation, taken, |rec, check| {
            rec.integrity = Integrity::Suspect;
            check.rebuilt_into = Some(rebuilt.generation);
            check.state = CheckState::Done;
        })
    }

    /// The outcome after the executor holding `taken` found its check moved:
    /// the outcome the record carries, or [`Executed::Lost`] while another
    /// executor runs it.
    fn lost_check(&self, generation: GenerationId, _taken: &Taken) -> Result<Executed, String> {
        let store = LocalIndexStore::new(self.env.engine());
        Ok(match store.load_integrity(generation).map_err(text)? {
            None => Executed::Outcome(CheckOutcome::Cancelled),
            Some((record, _)) => match terminal_outcome(&record) {
                Some(outcome) => Executed::Outcome(outcome),
                None => Executed::Lost,
            },
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
