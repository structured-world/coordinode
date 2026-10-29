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
    CheckState, IndexCheck, IndexIntegrityRecord, IndexStore as _, Integrity, LatestEntry,
    LocalIndexStore, LocalNodeStore, Mismatch, NodeStore as _, StoreError,
};
use coordinode_storage::engine::transaction::Transaction;
use rustc_hash::FxHashMap;

use super::{
    IndexBuildService, Shared, begin, cannot_move_here, is_contention, move_attempts, next_token,
    panic_message, text,
};
use crate::index::definition::{
    BuildFailure, GenerationId, IndexBuildRecord, IndexDefinition, IndexId, IndexState, IndexType,
};
use crate::index::membership::CatalogMembership;
use crate::index::registry::IntegrityReport;

mod verify;

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
    /// The index the record names, while its definition exists. `None`
    /// once the index is dropped.
    pub index: Option<super::BuildIndex>,
}

/// What an executor took: the definition it checks, which every repair is
/// bound to as stored, and its token.
struct Taken {
    def: IndexDefinition,
    token: u64,
}

/// How a check's executor ended.
enum Executed {
    Outcome(CheckOutcome),
    Lost,
}

/// Log how the check of `generation` ended and how long it took: lookups
/// read the records instead of the generation until it is verified.
fn log_check_end(generation: GenerationId, executed: &Executed, took: core::time::Duration) {
    let generation = generation.as_raw();
    let took_ms = took.as_millis();
    match executed {
        Executed::Outcome(CheckOutcome::Verified { checked, repaired }) => tracing::info!(
            generation,
            checked,
            repaired,
            took_ms,
            "index generation checked: verified"
        ),
        Executed::Outcome(CheckOutcome::SourceConflicts { conflicts }) => tracing::warn!(
            generation,
            conflicts = conflicts.len(),
            took_ms,
            "index generation checked: records break its constraint, it stays suspect"
        ),
        Executed::Outcome(CheckOutcome::Rebuilding { generation: into }) => tracing::warn!(
            generation,
            into = into.as_raw(),
            took_ms,
            "index generation checked: damage too wide to repair, rebuilding"
        ),
        Executed::Outcome(CheckOutcome::Failed(reason)) => tracing::warn!(
            generation,
            %reason,
            took_ms,
            "index generation check failed"
        ),
        Executed::Outcome(CheckOutcome::Cancelled) => {
            tracing::info!(generation, took_ms, "index generation check cancelled")
        }
        Executed::Lost => tracing::info!(
            generation,
            took_ms,
            "index generation check stopped: another executor holds it"
        ),
    }
}

/// Log how the check of this member's own copy of `generation` ended.
fn log_local_end(generation: GenerationId, outcome: &Result<LocalOutcome, String>) {
    let generation = generation.as_raw();
    match outcome {
        Ok(LocalOutcome::Verified) => tracing::info!(
            generation,
            "this member's index copy agrees with its records; it serves again"
        ),
        Ok(LocalOutcome::Replaced) => tracing::info!(
            generation,
            "this member's index copy was replaced from a peer and agrees with its records; it \
             serves again"
        ),
        Ok(LocalOutcome::Disagrees { found }) => tracing::warn!(
            generation,
            found,
            "this member's index copy disagrees with its records; it answers from the records \
             until it is replaced"
        ),
        Ok(LocalOutcome::Stopped) => {}
        Err(e) => tracing::warn!(
            generation,
            error = %e,
            "the check of this member's index copy failed"
        ),
    }
}

/// Log what one pass over the records cost, for the generations it read.
fn log_pass(targets: &[verify::Target<'_>], stats: &verify::VerifyStats) {
    tracing::info!(
        generations = targets.len(),
        source_rows = stats.source_rows,
        source_bytes = stats.source_bytes,
        decoded_rows = stats.decoded_rows,
        decoded_bytes = stats.decoded_bytes,
        expected_entries = stats.expected_entries,
        stored_entries = stats.stored_entries,
        spilled_runs = stats.spilled_runs,
        spilled_bytes = stats.spilled_bytes,
        scan_ms = stats.scan.as_millis(),
        sort_ms = stats.sort.as_millis(),
        compare_ms = stats.compare.as_millis(),
        "index check pass read the records"
    );
    for (name, value) in [
        (
            "coordinode_index_check_source_rows_total",
            stats.source_rows,
        ),
        (
            "coordinode_index_check_source_bytes_total",
            stats.source_bytes,
        ),
        (
            "coordinode_index_check_decoded_rows_total",
            stats.decoded_rows,
        ),
        (
            "coordinode_index_check_decoded_bytes_total",
            stats.decoded_bytes,
        ),
        (
            "coordinode_index_check_expected_entries_total",
            stats.expected_entries,
        ),
        (
            "coordinode_index_check_stored_entries_total",
            stats.stored_entries,
        ),
        (
            "coordinode_index_check_spilled_runs_total",
            stats.spilled_runs,
        ),
        (
            "coordinode_index_check_spilled_bytes_total",
            stats.spilled_bytes,
        ),
    ] {
        metrics::counter!(name).increment(value);
    }
    for (phase, took) in [
        ("scan", stats.scan),
        ("sort", stats.sort),
        ("compare", stats.compare),
    ] {
        metrics::histogram!("coordinode_index_check_phase_seconds", "phase" => phase)
            .record(took.as_secs_f64());
    }
}

/// The directory under the data directory a check pass spills to. Its files
/// are unnamed and vanish with the process, so nothing there outlives one.
const CHECK_SCRATCH_DIR: &str = "index-check";

/// How a check of this member's own copy ended.
enum LocalOutcome {
    /// Every entry agrees with this member's records: the mark is lifted.
    Verified,
    /// The copy disagreed, a replacement filled from a peer was published,
    /// and it agrees: the mark is lifted.
    Replaced,
    /// Entries missing or wrong: the copy stays unfit here.
    Disagrees { found: u64 },
    /// The process is stopping, the generation stopped serving, or a finding
    /// arrived during the pass; nothing is concluded.
    Stopped,
}

/// Generations waiting for the next shared pass, and whether a thread runs
/// the passes.
#[derive(Default)]
pub(super) struct CheckQueue {
    pending: Vec<GenerationId>,
    running: bool,
}

/// A check the pass thread holds: one generation, carried from pass to pass
/// until it concludes.
struct Active {
    generation: GenerationId,
    taken: Taken,
    /// Disagreements repaired by this executor across its passes.
    repaired: u64,
    started: Instant,
    /// This member's evidence revision when the latest pass began reading.
    local_revision: u64,
}

/// What became of disagreements a read reported.
enum ReportFate {
    /// Recorded against the generation, with a check admitted.
    Recorded,
    /// Not recorded: the index no longer serves from the generation, or this
    /// member takes no writes and checks its own copy instead.
    NotHere,
    /// Nothing to record: every one was resolved before it was looked at.
    Resolved,
}

/// Run `f`, a panic in it reported as its error.
fn guarded<T>(f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
        .unwrap_or_else(|panic| Err(panic_message(&*panic)))
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
                    let env = service.shared.env.as_ref();
                    let reports = env.registry().take_reports(MAINTENANCE_TICK);
                    if !reports.is_empty() {
                        service.record_reports(reports);
                    }
                    if env.registry().take_overflow() {
                        service.admit_unfit_here();
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
    pub(super) fn record_reports(&self, reports: Vec<IntegrityReport>) {
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
                Ok(ReportFate::Recorded) => {
                    if let Err(e) = self.submit_check(generation) {
                        tracing::warn!(generation = generation.as_raw(), error = %e,
                            "could not start the check of a reported index generation");
                    }
                }
                // Not recorded here: this member's own copy is checked here.
                Ok(ReportFate::NotHere) => self.submit_local_check(generation),
                // Every reported disagreement was resolved before it was
                // recorded: by a check that ran meanwhile, or by a writer.
                Ok(ReportFate::Resolved) => {}
                Err(e) => tracing::warn!(generation = generation.as_raw(), error = %e,
                    "could not record an index disagreement; this member answers from the \
                     records until a check runs"),
            }
        }
    }

    /// Admit the check of every generation this member found its copy of
    /// unfit, as its marks name them: for findings whose reports were
    /// dropped at the queue's bound, and marks read back at open.
    fn admit_unfit_here(&self) {
        let registry = self.shared.env.registry();
        for generation in registry.unfit_here() {
            let Some(index) = registry.index_of_generation(generation) else {
                // A generation that left the catalog: nothing serves from it
                // to check.
                continue;
            };
            match self.record(index, generation, Vec::new()) {
                Ok(ReportFate::Recorded) => {
                    if let Err(e) = self.submit_check(generation) {
                        tracing::warn!(generation = generation.as_raw(), error = %e,
                            "could not start the check of an index generation found unfit");
                    }
                }
                // A member that takes no writes checks its own copy itself:
                // the shared check another member runs proves only that
                // member's copy.
                Ok(ReportFate::NotHere) => self.submit_local_check(generation),
                // A mark is never resolved before it is recorded.
                Ok(ReportFate::Resolved) => {}
                Err(e) => tracing::warn!(generation = generation.as_raw(), error = %e,
                    "could not record an index generation found unfit"),
            }
        }
    }

    /// Record `found` against `generation` of `index` and admit a check
    /// unless one is pending. With nothing in `found`, the generation is
    /// recorded suspect on this member's mark alone.
    ///
    /// A reported disagreement is what a read saw earlier; one a check or a
    /// writer has resolved since no longer counts, or a check that verified
    /// the generation meanwhile would be undone by evidence it already
    /// covered. Each is looked at again as the entries and records stand.
    fn record(
        &self,
        index: IndexId,
        generation: GenerationId,
        found: Vec<Mismatch>,
    ) -> Result<ReportFate, String> {
        let env = self.shared.env.as_ref();
        let store = LocalIndexStore::new(env.engine());
        let Some(def) = store
            .load_definition(index)
            .map_err(text)?
            .filter(|d| d.generation == generation)
        else {
            return Ok(ReportFate::NotHere);
        };
        let found = if found.is_empty() {
            found
        } else {
            let fields = env.fields()?;
            let field_of = |name: &str| fields.lookup(name);
            let interpretation = def.interpretation(&field_of);
            let mut live = Vec::with_capacity(found.len());
            for mismatch in found {
                if self
                    .shared
                    .still_disagrees(&def, &interpretation, &mismatch)?
                {
                    live.push(mismatch);
                }
            }
            if live.is_empty() {
                return Ok(ReportFate::Resolved);
            }
            live
        };
        for () in move_attempts() {
            let (mut record, version) = match store.load_integrity(generation).map_err(text)? {
                Some((record, version)) => (record, Some(version)),
                None => (IndexIntegrityRecord::new(index, generation), None),
            };
            let mut new = false;
            for mismatch in &found {
                new |= record.report(mismatch.clone());
            }
            if found.is_empty() && record.integrity != Integrity::Suspect {
                // This member's mark alone: the revision moves, so a check
                // already running passes again before it verifies.
                record.integrity = Integrity::Suspect;
                record.evidence_revision += 1;
                new = true;
            }
            if !new && record.integrity == Integrity::Suspect && record.check_pending() {
                return Ok(ReportFate::Recorded);
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
                    return Ok(ReportFate::Recorded);
                }
                Err(e) if is_contention(&e) => continue,
                Err(e) if cannot_move_here(&e) => return Ok(ReportFate::NotHere),
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
        for () in move_attempts() {
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
        // Lookups that cannot use the generation meanwhile read the records,
        // so an operator needs to see the check run and end.
        tracing::info!(
            generation = generation.as_raw(),
            "checking an index generation"
        );
        let start = {
            let mut queue = self.shared.check_queue.lock();
            queue.pending.push(generation);
            !core::mem::replace(&mut queue.running, true)
        };
        if !start {
            return Ok(());
        }
        let shared = Arc::clone(&self.shared);
        let spawned = std::thread::Builder::new()
            .name("index-check".into())
            .spawn(move || shared.check_rounds());
        match spawned {
            Ok(executor) => {
                let mut executors = self.shared.executors.lock();
                executors.retain(|e| !e.is_finished());
                executors.push(executor);
                Ok(())
            }
            Err(e) => {
                let pending = {
                    let mut queue = self.shared.check_queue.lock();
                    queue.running = false;
                    core::mem::take(&mut queue.pending)
                };
                let mut checks = self.shared.checks.lock();
                for generation in pending {
                    checks.remove(&generation);
                }
                Err(format!("start the check executor: {e}"))
            }
        }
    }

    /// Check this member's own copy of `generation` against this member's
    /// records, on an executor of this process, unless one already does: a
    /// read-only pass both ways that writes nothing shared. A copy that
    /// agrees has its local mark lifted; one that disagrees stays unfit here
    /// and answers from the records.
    pub fn submit_local_check(&self, generation: GenerationId) {
        if self
            .shared
            .stopping
            .load(core::sync::atomic::Ordering::Acquire)
            || !self.shared.local_checks.lock().insert(generation)
        {
            return;
        }
        let start = {
            let mut queue = self.shared.local_queue.lock();
            queue.pending.push(generation);
            !core::mem::replace(&mut queue.running, true)
        };
        if !start {
            return;
        }
        let shared = Arc::clone(&self.shared);
        let spawned = std::thread::Builder::new()
            .name("index-local-check".into())
            .spawn(move || shared.local_rounds());
        match spawned {
            Ok(executor) => {
                let mut executors = self.shared.executors.lock();
                executors.retain(|e| !e.is_finished());
                executors.push(executor);
            }
            Err(e) => {
                let pending = {
                    let mut queue = self.shared.local_queue.lock();
                    queue.running = false;
                    core::mem::take(&mut queue.pending)
                };
                let mut local = self.shared.local_checks.lock();
                for generation in pending {
                    local.remove(&generation);
                    tracing::warn!(generation = generation.as_raw(), error = %e,
                        "could not start the check of this member's index copy");
                }
            }
        }
    }

    /// Whether a local check of `generation` runs on this member now.
    pub fn local_check_running(&self, generation: GenerationId) -> bool {
        self.shared.local_checks.lock().contains(&generation)
    }

    /// Fill replacements of this member's copies from `source`: a local
    /// check that finds a copy disagreeing with this member's records then
    /// replaces it, guarded before it serves and checked again before its
    /// mark is lifted. Without a source such a copy stays unfit here.
    pub fn set_copy_source(&self, source: Arc<dyn super::IndexCopySource>) {
        *self.shared.copy_source.write() = Some(source);
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
    /// outcome or none was admitted. An executor running it repairs nothing
    /// more once its current pass over the records ends.
    ///
    /// # Errors
    ///
    /// A storage failure, a commit failure other than a conflict, or moves
    /// kept winning for [`super::MOVE_PATIENCE`].
    pub fn cancel_check(&self, generation: GenerationId) -> Result<bool, String> {
        let env = self.shared.env.as_ref();
        let store = LocalIndexStore::new(env.engine());
        for () in move_attempts() {
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
        let records = store.list_integrity()?;
        let mut out = Vec::with_capacity(records.len());
        for record in records {
            let index = store
                .load_definition(record.index)?
                .map(|d| super::BuildIndex::of(&d));
            let running_here = matches!(
                self.shared.checks.lock().get(&record.generation),
                Some(CheckSlot::Running)
            );
            out.push(CheckStatus {
                generation: record.generation,
                running_here,
                index,
                record,
            });
        }
        Ok(out)
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
        // Marks read back at open name generations no report will.
        self.admit_unfit_here();
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
        // Read after the version the commit is conditioned on: a rebuild
        // meeting another one replaces only the generation it was asked to.
        if store
            .load_definition(def.id)
            .map_err(text)?
            .is_none_or(|d| d.generation != retired)
        {
            return Err(format!(
                "index '{def}' no longer serves from generation {}",
                retired.as_raw()
            ));
        }
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
    check_outcome(record, record.check.as_ref()?)
}

/// The waiter's view of `check` on `record`, `None` while it runs.
fn check_outcome(record: &IndexIntegrityRecord, check: &IndexCheck) -> Option<CheckOutcome> {
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
    catalog: &mut CatalogMembership<'_>,
) -> Result<bool, StoreError> {
    Ok(catalog.carries(record, &def.label)?
        && interpretation
            .record_membership(record)
            .is_some_and(|held| tuples(&held).iter().any(|t| t.as_slice() == tuple)))
}

/// [`record_holds`] for a record that may be absent: an absent one holds
/// nothing.
fn maybe_holds(
    def: &IndexDefinition,
    interpretation: &IndexInterpretation,
    record: Option<&NodeRecord>,
    tuple: &[u8],
    catalog: &mut CatalogMembership<'_>,
) -> Result<bool, StoreError> {
    match record {
        Some(record) => record_holds(def, interpretation, record, tuple, catalog),
        None => Ok(false),
    }
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
    catalog: &mut CatalogMembership<'_>,
) -> Result<bool, StoreError> {
    let own = LocalNodeStore.get(txn, shard_id, node)?;
    if maybe_holds(def, interpretation, own.as_ref(), tuple, catalog)? {
        return Ok(true);
    }
    for (_, record) in LocalNodeStore.versions(txn, shard_id, node)? {
        if record_holds(def, interpretation, &record, tuple, catalog)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// The owner a mismatch names.
fn owner_of(node: u64, valid_from: Option<i64>) -> EntryOwner {
    EntryOwner {
        node_id: node,
        valid_from,
    }
}

impl Shared {
    /// Whether `mismatch` still holds between `def`'s entries and the
    /// records as they stand now. Records that hold one unique value twice
    /// stay a disagreement: no entry resolves them.
    fn still_disagrees(
        &self,
        def: &IndexDefinition,
        interpretation: &IndexInterpretation,
        mismatch: &Mismatch,
    ) -> Result<bool, String> {
        let engine = self.env.engine();
        let shard_id = self.env.shard_id();
        let store = LocalIndexStore::new(engine);
        let mut catalog = CatalogMembership::current(engine);
        Ok(match mismatch {
            Mismatch::Missing {
                node,
                valid_from,
                tuple,
            } => {
                let owner = owner_of(*node, *valid_from);
                let row = LocalNodeStore
                    .latest_row(engine, shard_id, owner)
                    .map_err(text)?;
                let holds = maybe_holds(
                    def,
                    interpretation,
                    row.as_ref().map(|(r, _)| r),
                    tuple,
                    &mut catalog,
                )
                .map_err(text)?;
                holds
                    && match store.latest_entry(def, tuple, owner).map_err(text)? {
                        None => true,
                        Some(_) if !def.unique => false,
                        Some(entry) => entry.holder.map(NodeId::as_raw) != Some(*node),
                    }
            }
            Mismatch::Extra {
                node,
                valid_from,
                tuple,
            } => {
                let owner = owner_of(*node, *valid_from);
                match store.latest_entry(def, tuple, owner).map_err(text)? {
                    None => false,
                    Some(entry)
                        if def.unique && entry.holder.map_or(u64::MAX, NodeId::as_raw) != *node =>
                    {
                        false
                    }
                    Some(_) => {
                        let holds = if def.unique {
                            let latest = Transaction::new(
                                engine,
                                None,
                                coordinode_core::txn::timestamp::Timestamp::ZERO,
                                None,
                            );
                            node_holds(
                                &latest,
                                shard_id,
                                def,
                                interpretation,
                                NodeId::from_raw(*node),
                                tuple,
                                &mut catalog,
                            )
                            .map_err(text)?
                        } else {
                            let row = LocalNodeStore
                                .latest_row(engine, shard_id, owner)
                                .map_err(text)?;
                            maybe_holds(
                                def,
                                interpretation,
                                row.as_ref().map(|(r, _)| r),
                                tuple,
                                &mut catalog,
                            )
                            .map_err(text)?
                        };
                        !holds
                    }
                }
            }
            Mismatch::SourceDuplicate { .. } => true,
        })
    }

    /// Run local checks until none waits: each round reads this member's
    /// records once for every copy waiting, then concludes each.
    fn local_rounds(&self) {
        loop {
            let pending = {
                let mut queue = self.local_queue.lock();
                if queue.pending.is_empty() {
                    queue.running = false;
                    return;
                }
                core::mem::take(&mut queue.pending)
            };
            let ended = {
                let _seat = self.quiet_seat();
                guarded(|| Ok(self.local_round(&pending)))
            };
            let ended = ended.unwrap_or_else(|e| {
                pending
                    .iter()
                    .map(|&generation| (generation, Err(e.clone())))
                    .collect()
            });
            let mut local = self.local_checks.lock();
            for (generation, outcome) in ended {
                local.remove(&generation);
                log_local_end(generation, &outcome);
            }
        }
    }

    /// Read this member's copies of `pending` both ways against this
    /// member's records in one pass, writing nothing shared, and lift the
    /// local mark of each that agrees. Records breaking a unique constraint
    /// are the records' disagreement, not the copy's: the catalog keeps such
    /// a generation suspect on every member.
    fn local_round(
        &self,
        pending: &[GenerationId],
    ) -> Vec<(GenerationId, Result<LocalOutcome, String>)> {
        let env = self.env.as_ref();
        let registry = env.registry();
        let mut ended = Vec::with_capacity(pending.len());
        let mut defs = Vec::with_capacity(pending.len());
        for &generation in pending {
            match registry
                .index_of_generation(generation)
                .and_then(|id| registry.get_by_id(id))
                .filter(|d| {
                    d.generation == generation
                        && d.index_type == IndexType::BTree
                        && d.state == IndexState::Ready
                }) {
                Some(def) => defs.push(def),
                None => ended.push((generation, Ok(LocalOutcome::Stopped))),
            }
        }
        if defs.is_empty() {
            return ended;
        }
        let fields = match env.fields() {
            Ok(fields) => fields,
            Err(e) => {
                ended.extend(defs.iter().map(|d| (d.generation, Err(e.clone()))));
                return ended;
            }
        };
        let field_of = |name: &str| fields.lookup(name);
        let interpretations: Vec<IndexInterpretation> =
            defs.iter().map(|d| d.interpretation(&field_of)).collect();
        // Taken before anything is read: a finding after it keeps the copy
        // unfit whatever the pass concludes.
        let revisions: Vec<u64> = defs
            .iter()
            .map(|d| registry.local_revision(d.generation))
            .collect();
        let found = match self.local_pass(&defs, &interpretations) {
            Ok(Some(found)) => found,
            Ok(None) => {
                ended.extend(
                    defs.iter()
                        .map(|d| (d.generation, Ok(LocalOutcome::Stopped))),
                );
                return ended;
            }
            Err(e) => {
                ended.extend(defs.iter().map(|d| (d.generation, Err(e.clone()))));
                return ended;
            }
        };
        for (((def, interpretation), revision), found) in
            defs.iter().zip(&interpretations).zip(revisions).zip(found)
        {
            let outcome = guarded(|| self.local_conclude(def, interpretation, revision, found));
            ended.push((def.generation, outcome));
        }
        ended
    }

    /// Conclude the local check of `def`'s copy, which a pass taken at
    /// `revision` found `found` entries of missing or wrong: lift the mark,
    /// or replace the copy from a peer and lift it once the replacement is
    /// proved.
    fn local_conclude(
        &self,
        def: &IndexDefinition,
        interpretation: &IndexInterpretation,
        revision: u64,
        found: u64,
    ) -> Result<LocalOutcome, String> {
        let env = self.env.as_ref();
        let registry = env.registry();
        let generation = def.generation;
        if found == 0 {
            let lifted = registry
                .verified_here(env.engine(), def, revision, self.read_time())
                .map_err(|e| e.to_string())?;
            return Ok(if lifted {
                LocalOutcome::Verified
            } else {
                LocalOutcome::Stopped
            });
        }
        let source = self.copy_source.read().clone();
        let Some(source) = source else {
            return Ok(LocalOutcome::Disagrees { found });
        };
        let proof = self.replace_copy(generation, source.as_ref())?;
        // The replacement serves once proved: by the peer's proof of the copy
        // its history came from, under this generation's interpretation, or
        // else by a pass reading it agreeing. A finding after this revision,
        // or a disagreement in the pass, keeps the mark.
        let revision = registry.local_revision(generation);
        let proved_by_peer = proof
            .as_deref()
            .is_some_and(|proof| crate::index::transfer::import_proves(proof, def));
        if !proved_by_peer {
            let Some(found) = self.local_pass(
                core::slice::from_ref(def),
                core::slice::from_ref(interpretation),
            )?
            else {
                return Ok(LocalOutcome::Stopped);
            };
            let found = found.first().copied().unwrap_or(0);
            if found > 0 {
                return Ok(LocalOutcome::Disagrees { found });
            }
        }
        let lifted = registry
            .verified_here(env.engine(), def, revision, self.read_time())
            .map_err(|e| e.to_string())?;
        Ok(if lifted {
            LocalOutcome::Replaced
        } else {
            LocalOutcome::Stopped
        })
    }

    /// What a check pass may hold in memory, and where it spills the rest:
    /// a directory under the data directory, so scratch files never outgrow
    /// the volume the store already lives on.
    fn check_budget(&self) -> verify::Budget {
        verify::Budget {
            memory_bytes: self.config.read().check_memory,
            scratch: Some(self.env.engine().data_dir().join(CHECK_SCRATCH_DIR)),
        }
    }

    /// The snapshot a pass ending now has read up to: the oracle's time, or
    /// 0 on an engine without one.
    pub(super) fn read_time(&self) -> u64 {
        self.env
            .oracle()
            .map_or(0, |oracle| oracle.current().as_raw())
    }

    /// Replace this member's copy of `generation` with one `source` fills
    /// from a peer: guarded before it is published, so it never serves
    /// unguarded. The generation stays unfit here until its copy is proved.
    /// Returns the peer's proof of its copy, when its history carried one.
    fn replace_copy(
        &self,
        generation: GenerationId,
        source: &dyn super::IndexCopySource,
    ) -> Result<Option<Vec<u8>>, String> {
        let env = self.env.as_ref();
        let engine = env.engine();
        let proof = source.prepare(generation)?;
        let copy = match env.registry().arm_replacement(engine, generation) {
            Ok(copy) => copy,
            Err(e) => {
                if let Err(dropped) = engine.abandon_generation(generation) {
                    tracing::warn!(generation = generation.as_raw(), error = %dropped,
                        "an unguarded replacement was not dropped; the next open drops it");
                }
                return Err(format!("guard the replacement: {e}"));
            }
        };
        engine
            .publish_generation(generation)
            .map_err(|e| format!("publish the replacement: {e}"))?;
        env.registry().replacement_published(copy);
        Ok(proof)
    }

    /// One pass both ways over this member's copies of `defs` and this
    /// member's records, writing nothing: per generation, the entries
    /// missing or wrong. Records breaking a unique constraint are not
    /// counted. `None` when the process is stopping.
    fn local_pass(
        &self,
        defs: &[IndexDefinition],
        interpretations: &[IndexInterpretation],
    ) -> Result<Option<Vec<u64>>, String> {
        let targets: Vec<verify::Target<'_>> = defs
            .iter()
            .zip(interpretations)
            .map(|(def, interpretation)| verify::Target {
                def,
                interpretation,
            })
            .collect();
        let Some((findings, stats)) = self.read_pass(&targets)? else {
            return Ok(None);
        };
        log_pass(&targets, &stats);
        Ok(Some(
            findings
                .iter()
                .map(|found| {
                    // Disagreements of one pass, far below the type's range.
                    found
                        .mismatches
                        .iter()
                        .filter(|m| !matches!(m, Mismatch::SourceDuplicate { .. }))
                        .count() as u64
                })
                .collect(),
        ))
    }

    /// Read `targets` against this member's records in one pass at one
    /// snapshot. `None` when the process is stopping.
    fn read_pass(
        &self,
        targets: &[verify::Target<'_>],
    ) -> Result<Option<(Vec<verify::Findings>, verify::VerifyStats)>, String> {
        verify::verify(
            self.env.engine(),
            self.env.shard_id(),
            targets,
            &self.check_budget(),
            &self.stopping,
        )
        .map_err(|e| e.to_string())
    }

    /// Run checks until none waits or is held: each round reads the records
    /// once for every check this executor holds, then repairs and concludes
    /// each by what that pass found. One seat for the whole round: the
    /// checks share it as they share the read.
    fn check_rounds(self: &Arc<Self>) {
        let mut active: Vec<Active> = Vec::new();
        loop {
            let pending = {
                let mut queue = self.check_queue.lock();
                if queue.pending.is_empty() && active.is_empty() {
                    queue.running = false;
                    return;
                }
                core::mem::take(&mut queue.pending)
            };
            let _seat = self.quiet_seat();
            if self.stopping.load(core::sync::atomic::Ordering::Acquire) {
                let started = Instant::now();
                for generation in pending {
                    self.finish_check(generation, Executed::Lost, started);
                }
                for held in active.drain(..) {
                    let executed = self
                        .lost_check(held.generation, &held.taken)
                        .unwrap_or_else(|e| Executed::Outcome(CheckOutcome::Failed(e)));
                    self.finish_check(held.generation, executed, held.started);
                }
                continue;
            }
            for generation in pending {
                let started = Instant::now();
                match guarded(|| self.take_check(generation)) {
                    Ok(Ok(taken)) => active.push(Active {
                        generation,
                        taken,
                        repaired: 0,
                        started,
                        local_revision: 0,
                    }),
                    Ok(Err(executed)) => self.finish_check(generation, executed, started),
                    Err(e) => self.finish_check(
                        generation,
                        Executed::Outcome(CheckOutcome::Failed(e)),
                        started,
                    ),
                }
            }
            if !active.is_empty() {
                active = self.shared_pass(active);
            }
        }
    }

    /// One pass over the records for every check in `active`, then each
    /// repaired and concluded by what it found. Returns the checks that need
    /// another pass.
    fn shared_pass(self: &Arc<Self>, active: Vec<Active>) -> Vec<Active> {
        let env = self.env.as_ref();
        let mut held = Vec::with_capacity(active.len());
        for mut check in active {
            // Only checks still this executor's, of generations the index
            // still serves from, are read: repairs to a generation the index
            // left are refused at commit and would count as damage.
            let ready = guarded(|| {
                if self.held_record(check.generation, &check.taken)?.is_none() {
                    return self.lost_check(check.generation, &check.taken).map(Some);
                }
                if !self.still_serves(check.generation, &check.taken)? {
                    return self
                        .conclude_retired(check.generation, &check.taken)
                        .map(Some);
                }
                Ok(None)
            });
            match ready {
                Ok(None) => {
                    // Taken before the pass reads anything: a report after it
                    // keeps this member's copy unfit whatever the pass finds.
                    check.local_revision = env.registry().local_revision(check.generation);
                    held.push(check);
                }
                Ok(Some(executed)) => self.finish_check(check.generation, executed, check.started),
                Err(e) => self.fail_check(&check, &e),
            }
        }
        if held.is_empty() {
            return held;
        }
        let fields = match env.fields() {
            Ok(fields) => fields,
            Err(e) => {
                for check in &held {
                    self.fail_check(check, &e);
                }
                return Vec::new();
            }
        };
        let field_of = |name: &str| fields.lookup(name);
        let interpretations: Vec<IndexInterpretation> = held
            .iter()
            .map(|check| check.taken.def.interpretation(&field_of))
            .collect();
        let read = {
            let targets: Vec<verify::Target<'_>> = held
                .iter()
                .zip(&interpretations)
                .map(|(check, interpretation)| verify::Target {
                    def: &check.taken.def,
                    interpretation,
                })
                .collect();
            let read = guarded(|| self.read_pass(&targets));
            if let Ok(Some((_, stats))) = &read {
                log_pass(&targets, stats);
            }
            read
        };
        let findings = match read {
            Ok(Some((findings, _))) => findings,
            Ok(None) => {
                for check in held {
                    let executed = self
                        .lost_check(check.generation, &check.taken)
                        .unwrap_or_else(|e| Executed::Outcome(CheckOutcome::Failed(e)));
                    self.finish_check(check.generation, executed, check.started);
                }
                return Vec::new();
            }
            Err(e) => {
                for check in &held {
                    self.fail_check(check, &e);
                }
                return Vec::new();
            }
        };
        let generations = held.len();
        let recording = Instant::now();
        let syncs_before = env.engine().journal_syncs();
        let mut again = Vec::with_capacity(held.len());
        for ((mut check, interpretation), found) in
            held.into_iter().zip(&interpretations).zip(findings)
        {
            match guarded(|| self.after_pass(&mut check, interpretation, found)) {
                Ok(None) => again.push(check),
                Ok(Some(executed)) => self.finish_check(check.generation, executed, check.started),
                Err(e) => self.fail_check(&check, &e),
            }
        }
        // The counter only grows, so the later reading is never below the
        // earlier one.
        let journal_syncs = env
            .engine()
            .journal_syncs()
            .zip(syncs_before)
            .map(|(after, before)| after - before);
        let recorded = recording.elapsed();
        // Syncs of the whole member while the pass recorded: a concurrent
        // writer's count with them, so they bound the pass's own from above.
        tracing::info!(
            generations,
            record_ms = recorded.as_millis(),
            journal_syncs_meanwhile = journal_syncs,
            "index check pass recorded its repairs and outcomes"
        );
        metrics::histogram!("coordinode_index_check_phase_seconds", "phase" => "record")
            .record(recorded.as_secs_f64());
        again
    }

    /// Record that the check `check` holds failed with `reason`, and end it.
    fn fail_check(&self, check: &Active, reason: &str) {
        let executed = self
            .conclude_failed(check.generation, &check.taken, reason)
            .unwrap_or_else(|e| Executed::Outcome(CheckOutcome::Failed(e)));
        self.finish_check(check.generation, executed, check.started);
    }

    /// Hand `executed` to the waiters of `generation`'s check.
    fn finish_check(&self, generation: GenerationId, executed: Executed, started: Instant) {
        log_check_end(generation, &executed, started.elapsed());
        let slot = match executed {
            Executed::Outcome(outcome) => CheckSlot::Done(outcome),
            Executed::Lost => CheckSlot::Lost,
        };
        let mut checks = self.checks.lock();
        checks.insert(generation, slot);
        self.check_done.notify_all();
    }

    /// Take the check of `generation`: its state moves to running under a
    /// fresh token, conditioned on the record as read. A check whose index
    /// no longer serves from the generation is cancelled instead.
    fn take_check(&self, generation: GenerationId) -> Result<Result<Taken, Executed>, String> {
        let env = self.env.as_ref();
        let store = LocalIndexStore::new(env.engine());
        for () in move_attempts() {
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
                    return Ok(Ok(Taken { def, token }));
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

    /// Repair and record what one pass found for the check `check` holds,
    /// and conclude it when the pass settles it. `None` when it needs
    /// another pass.
    fn after_pass(
        self: &Arc<Self>,
        check: &mut Active,
        interpretation: &IndexInterpretation,
        found: verify::Findings,
    ) -> Result<Option<Executed>, String> {
        let config = *self.config.read();
        let generation = check.generation;
        let taken = &check.taken;
        let Some((_, held_version)) = self.held_record(generation, taken)? else {
            return self.lost_check(generation, taken).map(Some);
        };
        let mut conflicts = Vec::new();
        let mut missing = Vec::new();
        let mut extra = Vec::new();
        for mismatch in found.mismatches {
            match mismatch {
                Mismatch::SourceDuplicate { .. } => conflicts.push(mismatch),
                Mismatch::Missing { .. } => missing.push(mismatch),
                Mismatch::Extra { .. } => extra.push(mismatch),
            }
        }
        // Disagreements one pass finds, far below the type's range.
        let found_count = (conflicts.len() + missing.len() + extra.len()) as u64;
        let mut repaired = 0u64;
        // Missing entries first and extra ones after, never in one commit: a
        // unique value whose entry names the wrong holder is both, and the
        // removal must not meet the replacement in the same write.
        for batch in missing
            .chunks(config.check_repair_batch)
            .chain(extra.chunks(config.check_repair_batch))
        {
            if check.repaired + repaired > config.check_max_repairs {
                break;
            }
            repaired += self.repair(taken, held_version, interpretation, batch, &mut conflicts)?;
        }
        check.repaired += repaired;
        let conflict_count = conflicts.len() as u64;
        // A generation found damaged, or repaired by this check, is proved
        // from the conclusion on; an older snapshot still reads the entries
        // as they were then. A direct-mode engine keeps no older snapshots to
        // protect.
        let now = self.env.oracle().map(|oracle| oracle.current().as_raw());
        let finished_at = wall_clock_ms();
        let mut passes = 0;
        let mut recorded = None;
        // The pass's progress and, when it settles the check, its conclusion
        // in one commit: a clean check costs the journal one write, not one
        // for its progress and another for its outcome.
        let moved = self.checkpoint(generation, taken, |rec, held| {
            held.checked += found.checked;
            held.mismatches += found_count;
            held.repaired += repaired;
            for conflict in &conflicts {
                rec.report(conflict.clone());
            }
            held.passes += 1;
            passes = held.passes;
            recorded = None;
            if found_count == 0 {
                // Every disagreement repaired, none left: the generation
                // agrees with its records, unless a report arrived since the
                // pass began, which the next pass looks for.
                if rec.evidence_revision != held.started_revision {
                    held.started_revision = rec.evidence_revision;
                    return;
                }
                if rec.evidence_revision > 0 || held.repaired > 0 {
                    rec.trusted_from = now.or(rec.trusted_from);
                }
                rec.integrity = Integrity::Verified;
                rec.evidence.clear();
            } else if found_count == conflict_count {
                // Only records breaking the constraint remain: no entry can
                // fix them.
                rec.integrity = Integrity::Suspect;
            } else {
                return;
            }
            held.state = CheckState::Done;
            held.finished_at_ms = Some(finished_at);
            recorded = check_outcome(rec, held);
        })?;
        if !moved {
            return self.lost_check(generation, taken).map(Some);
        }
        if let Some(outcome) = recorded {
            let env = self.env.as_ref();
            env.registry().apply_integrity(
                &LocalIndexStore::new(env.engine())
                    .list_integrity()
                    .map_err(text)?,
            );
            // The pass read this member's copy whole, after the revision it
            // took: that copy is proved, unless a report found it disagreeing
            // since.
            if let CheckOutcome::Verified { .. } = &outcome {
                env.registry()
                    .verified_here(
                        env.engine(),
                        &taken.def,
                        check.local_revision,
                        now.unwrap_or(0),
                    )
                    .map_err(|e| e.to_string())?;
            }
            return Ok(Some(Executed::Outcome(outcome)));
        }
        if check.repaired > config.check_max_repairs || passes >= MAX_PASSES {
            // A check of a generation the index left must not rebuild the
            // index it now serves from.
            if !self.still_serves(generation, taken)? {
                return self.conclude_retired(generation, taken).map(Some);
            }
            return self.escalate(generation, taken).map(Some);
        }
        Ok(None)
    }

    /// Repair `mismatches` in one commit, each only while its record and its
    /// entry are still as found, the index still serves from the checked
    /// generation and the check is still this executor's (its record at
    /// `held_version`, as the page read it). A missing unique entry whose
    /// value another record holds is a conflict, appended to `conflicts`.
    /// Returns how many were repaired; a repair that met a concurrent change
    /// is left for the next pass, and one whose check another executor took
    /// over or a cancellation ended commits nothing.
    fn repair(
        &self,
        taken: &Taken,
        held_version: u64,
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
        // The definition as it stands, still the one checked: an index
        // rebuilt, dropped or changed meanwhile takes no repair.
        let Some(def_version) = store.definition_version_as(def).map_err(text)? else {
            return Ok(0);
        };
        let mut txn = begin(env);
        // Conditions before the writes they guard: a direct-mode engine
        // decides a condition when it is stated.
        store
            .expect_definition_txn(&mut txn, def.id, Some(def_version))
            .map_err(text)?;
        store
            .expect_integrity_txn(&mut txn, def.generation, held_version)
            .map_err(text)?;
        let mut repaired = 0u64;
        let mut catalog = CatalogMembership::current(engine);
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
                    if !record_holds(def, interpretation, &record, tuple, &mut catalog)
                        .map_err(text)?
                    {
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
                        }) if node_holds(
                            &latest,
                            shard_id,
                            def,
                            interpretation,
                            h,
                            tuple,
                            &mut catalog,
                        )
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
                            &mut catalog,
                        )
                        .map_err(text)?
                    } else {
                        let row = LocalNodeStore
                            .latest_row(engine, shard_id, owner)
                            .map_err(text)?;
                        maybe_holds(
                            def,
                            interpretation,
                            row.as_ref().map(|(r, _)| r),
                            tuple,
                            &mut catalog,
                        )
                        .map_err(text)?
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

    /// Whether the index `taken` checks still serves from `generation`, built.
    fn still_serves(&self, generation: GenerationId, taken: &Taken) -> Result<bool, String> {
        Ok(LocalIndexStore::new(self.env.engine())
            .load_definition(taken.def.id)
            .map_err(text)?
            .is_some_and(|d| d.generation == generation && d.state == IndexState::Ready))
    }

    /// Record that the held check ended because its index stopped serving
    /// from the checked generation: dropped, rebuilt or replaced.
    fn conclude_retired(
        &self,
        generation: GenerationId,
        taken: &Taken,
    ) -> Result<Executed, String> {
        self.conclude_check(generation, taken, |_, check| {
            check.state = CheckState::Cancelled;
        })
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
        for () in move_attempts() {
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
        // The outcome as this commit records it. Read back afterwards, the
        // record may already carry the next check, admitted by a report that
        // arrived meanwhile, and say nothing of this one.
        let mut recorded = None;
        let landed = self.checkpoint(generation, taken, |record, check| {
            update(record, check);
            check.finished_at_ms = Some(now);
            recorded = check_outcome(record, check);
        })?;
        let env = self.env.as_ref();
        let store = LocalIndexStore::new(env.engine());
        env.registry()
            .apply_integrity(&store.list_integrity().map_err(text)?);
        if !landed {
            return self.lost_check(generation, taken);
        }
        recorded.map(Executed::Outcome).ok_or_else(|| {
            format!(
                "the check of generation {} concluded without an outcome",
                generation.as_raw()
            )
        })
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
