//! Durable background index builds.
//!
//! A CREATE or a rebuild admits its build with the publication of the index:
//! an [`IndexBuildRecord`] committed in the same catalog commit as the
//! definition. From then on the build belongs to the engine, not to the
//! request that asked for it: an executor thread of [`IndexBuildService`]
//! takes it, fills the generation and publishes the outcome, and a caller
//! only waits for that outcome. A waiter that gives up cancels nothing.
//!
//! Every move of a build (taken, published, failed, cancelled) is a catalog
//! commit conditioned on the record the mover read. A mover whose commit
//! meets another reads the record again and retries while the move is still
//! its own to make, so publication, failure and cancellation cannot all
//! land, and none is lost to a transient conflict. An executor that lost its
//! build (a cancellation, a drop, another executor after a leader change)
//! cannot finish it, and its waiters follow the record to the outcome the
//! winner reaches. The pages of a backfill are bound to the definition
//! record, which a cancellation or a drop moves in the same commit, so a
//! fenced executor writes no further entry either.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use coordinode_core::graph::intern::FieldInterner;
use coordinode_core::txn::proposal::{ProposalIdGenerator, ProposalPipeline};
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_modality::{
    BuildFailure, BuildState, IndexBuildRecord, IndexStore as _, LocalIndexStore, StoreError,
};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::transaction::{CommitError, Transaction};
use rustc_hash::FxHashMap;

use coordinode_core::graph::node::NodeId;
use coordinode_core::graph::types::Value;
use coordinode_modality::DuplicateRepair;

use super::build::{Backfill, BackfillError, BackfillProgress, DuplicateRepairer};
use super::definition::{GenerationId, IndexDefinition, IndexState};
use super::registry::{IndexRegistry, UniqueViolation};
use super::text_registry::TextIndexRegistry;

/// A build a member runs for itself: an index whose structure every member
/// derives from the data it holds (a full-text index), so the build has no
/// catalog moves and its readiness is the member's own. Returns how many
/// records it indexed.
pub type LocalBuild = Box<dyn FnOnce(&dyn BuildEnvironment) -> Result<u64, BuildError> + Send>;

/// What a build executor needs from the deployment that runs it: the store,
/// the field dictionary, the registry writers consult, and the commit path
/// every write of the build goes through, so pages and catalog moves
/// replicate like any other write.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot run index builds",
    label = "a build environment is required here",
    note = "an embedded database provides one over its engine, log and field dictionary"
)]
pub trait BuildEnvironment: Send + Sync {
    /// The storage engine the index lives in.
    fn engine(&self) -> &StorageEngine;

    /// The timestamp oracle; `None` for a direct-mode engine.
    fn oracle(&self) -> Option<&TimestampOracle>;

    /// The field dictionary as it stands now.
    ///
    /// # Errors
    ///
    /// The dictionary could not be read.
    fn fields(&self) -> Result<FieldInterner, String>;

    /// The shard whose nodes are indexed.
    fn shard_id(&self) -> u16;

    /// The registry writers maintain indexes through.
    fn registry(&self) -> &IndexRegistry;

    /// The full-text indexes this member keeps, `None` where it keeps none.
    fn text_registry(&self) -> Option<&TextIndexRegistry>;

    /// Commit one backfill page through the deployment's write path.
    ///
    /// # Errors
    ///
    /// The commit failed; a conflict is read again by the backfill.
    fn commit_page(&self, txn: &mut Transaction<'_>) -> Result<(), CommitError>;

    /// Commit one catalog move of a build through the deployment's write
    /// path.
    ///
    /// # Errors
    ///
    /// The commit failed; a conflict means another move met this one.
    fn commit_catalog(&self, txn: &mut Transaction<'_>) -> Result<(), CommitError>;

    /// The log a statement the build runs commits through (a duplicate
    /// repair), with the proposal ids it takes; `None` for a member whose
    /// commits apply to its engine directly.
    fn statement_log(&self) -> Option<(&dyn ProposalPipeline, &ProposalIdGenerator)>;
}

/// Why a build did not publish its index.
#[derive(Debug, Clone, thiserror::Error)]
pub enum BuildError {
    /// The stored data holds a value of the unique index twice.
    #[error(transparent)]
    Duplicate(UniqueViolation),
    /// Any other failure, as the build record keeps it.
    #[error("{0}")]
    Other(String),
}

/// The outcome of a build, as a waiter receives it.
#[derive(Debug, Clone)]
pub enum IndexBuildOutcome {
    /// The index was published ready.
    Published {
        /// Nodes indexed, `None` when the outcome was read back from the
        /// catalog rather than from the executor that published it.
        indexed: Option<u64>,
    },
    /// The build failed; per its record the index was withdrawn or kept
    /// failed.
    Failed(BuildError),
    /// The build was cancelled, or its index dropped, before it finished.
    Cancelled,
}

/// Where an executor of this process stands with one build.
enum Slot {
    /// An executor thread has it.
    Running,
    /// The executor reached this outcome.
    Done(IndexBuildOutcome),
    /// The executor lost the build to another one (after a leader change),
    /// whose outcome the record will carry.
    Lost,
}

/// How an executor's attempt at a build ended.
enum Executed {
    /// The build has this outcome.
    Outcome(IndexBuildOutcome),
    /// Another executor holds the build; the record will say how it ends.
    Lost,
}

/// How the final move of an executor's build went.
enum Concluded {
    /// The move landed.
    Landed,
    /// Another mover ended the build first, with this outcome.
    Ended(IndexBuildOutcome),
    /// Another executor took the build over.
    Lost,
}

struct Shared {
    env: Arc<dyn BuildEnvironment>,
    /// Retunable while builds run: a seat and a backfill read it when they
    /// start.
    config: parking_lot::RwLock<IndexBuildConfig>,
    slots: parking_lot::Mutex<FxHashMap<GenerationId, Slot>>,
    finished: parking_lot::Condvar,
    /// Builds executing now, bounded by `config.max_running`.
    running: parking_lot::Mutex<usize>,
    vacancy: parking_lot::Condvar,
    /// Where each build an executor of this process holds stands now.
    phases: parking_lot::Mutex<FxHashMap<GenerationId, BuildPhase>>,
    /// The last node key each backfill of this process committed entries
    /// through. Kept in memory: a build no executor here holds is taken as
    /// covering nothing, which only makes a write read more.
    covered: parking_lot::Mutex<FxHashMap<GenerationId, Vec<u8>>>,
    /// Set by [`IndexBuildService::shutdown`]: executors stop between pages
    /// and no new one starts.
    stopping: core::sync::atomic::AtomicBool,
    /// The executor threads started here, joined at shutdown so none
    /// outlives the storage it holds.
    executors: parking_lot::Mutex<Vec<std::thread::JoinHandle<()>>>,
}

/// How the engine runs its index builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexBuildConfig {
    /// Builds filling indexes at once (at least one); the others wait for a
    /// seat with their builds accepted.
    pub max_running: usize,
    /// How long a key-shaped backfill waits for the transactions opened
    /// before its index existed to end; a build that outwaits it fails.
    pub older_transactions_wait: Duration,
    /// The most stored node rows a write reads to prove a value it takes in
    /// a unique index still being built free; past it the write is refused
    /// as unresolved, to be retried once the build is done.
    pub unique_admission_read_limit: u64,
    /// How long a statement that creates an index waits for its build when
    /// it names no bound of its own. A statement that outwaits it returns
    /// with the build still running and the operation that identifies it;
    /// the build goes on without the statement.
    pub statement_wait: Duration,
}

/// The most stored node rows a write reads by default to prove a value
/// free in a unique index still being built.
pub const DEFAULT_UNIQUE_ADMISSION_READ_LIMIT: u64 = 100_000;

/// How long a statement creating an index waits for its build by default.
pub const DEFAULT_STATEMENT_WAIT: Duration = Duration::from_secs(60);

impl Default for IndexBuildConfig {
    fn default() -> Self {
        Self {
            max_running: 2,
            older_transactions_wait: super::build::DEFAULT_OLDER_TRANSACTIONS_WAIT,
            unique_admission_read_limit: DEFAULT_UNIQUE_ADMISSION_READ_LIMIT,
            statement_wait: DEFAULT_STATEMENT_WAIT,
        }
    }
}

impl IndexBuildConfig {
    /// At least one build runs, or none would ever start.
    fn normalized(self) -> Self {
        Self {
            max_running: self.max_running.max(1),
            ..self
        }
    }
}

/// Where a build an executor of this process holds stands now. Kept in
/// memory: it changes every page, and only the outcome is durable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildPhase {
    /// Waiting for a seat: as many builds as the engine runs at once are
    /// filling indexes.
    AwaitingSeat,
    /// Waiting for the transactions opened before the index to end.
    AwaitingOlderTransactions,
    /// Filling the index; the records indexed so far, `None` for a build
    /// that does not report them.
    Indexing {
        /// Records indexed so far.
        indexed: Option<u64>,
    },
}

/// One build as inspection shows it: its durable record when it has one (a
/// member's own build has none) and, while an executor of this process
/// holds it, where it stands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildStatus {
    /// The generation the build fills, and the operation that identifies
    /// it.
    pub generation: GenerationId,
    /// The build's record in the catalog.
    pub record: Option<IndexBuildRecord>,
    /// Where it stands on this process's executor.
    pub phase: Option<BuildPhase>,
    /// The index the record names, while its definition exists. `None`
    /// once the index is withdrawn or dropped.
    pub index: Option<BuildIndex>,
}

impl BuildStatus {
    /// Nodes indexed so far: the executor's live count while it fills,
    /// else the record's. Progress, not proof of coverage.
    pub fn indexed(&self) -> Option<u64> {
        match (&self.phase, &self.record) {
            (Some(BuildPhase::Indexing { indexed: Some(n) }), _) => Some(*n),
            (_, Some(record)) => Some(record.indexed),
            (_, None) => None,
        }
    }

    /// Why the build failed, once it has.
    pub fn failure(&self) -> Option<&str> {
        match self.record.as_ref().map(|r| &r.state) {
            Some(BuildState::Failed { reason }) => Some(reason),
            _ => None,
        }
    }
}

/// The index a build fills, as inspection names it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildIndex {
    /// The index's name; `None` for an unnamed index.
    pub name: Option<String>,
    /// The label whose nodes it covers.
    pub label: String,
}

/// The build of a full-text index over `properties` of `label`: each
/// property's index is filled from the texts the member stores, holding the
/// index while it scans, so a commit after the scan starts reaches it
/// through the text worker and lands after the backfill rather than under
/// it. Returns how many distinct nodes it indexed.
pub fn text_build(label: String, properties: Vec<String>) -> LocalBuild {
    Box::new(move |env: &dyn BuildEnvironment| {
        let registry = env
            .text_registry()
            .ok_or_else(|| BuildError::Other("this member keeps no full-text indexes".into()))?;
        let fields = env.fields().map_err(BuildError::Other)?;
        let mut nodes = rustc_hash::FxHashSet::default();
        for property in &properties {
            registry
                .rebuild_index(&label, property, || {
                    let texts = super::text_registry::stored_texts(
                        env.engine(),
                        env.shard_id(),
                        &fields,
                        &label,
                        property,
                        crate::executor::runner::wall_clock_us(),
                    )?;
                    nodes.extend(texts.iter().filter(|t| t.text.is_some()).map(|t| t.node_id));
                    Ok(texts)
                })
                .map_err(|e| BuildError::Other(format!("backfill text index: {e}")))?;
        }
        Ok(nodes.len() as u64)
    })
}

/// Attempts one move of a build makes against other moves of the same
/// build. A build moves a few times in its life (taken, published or
/// failed, cancelled), so a mover that loses this many races in a row faces
/// a fault, not contention.
pub const MOVE_ATTEMPTS: usize = 16;

/// The message a panic carried, as a build failure states it.
fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    panic
        .downcast_ref::<&'static str>()
        .map(|s| (*s).to_string())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "panic in the build executor".to_string())
}

/// How often a waiter whose executor lost the build looks at the record.
const LOST_POLL: Duration = Duration::from_millis(20);

/// A token for one take of a build, never handed out twice: unique within
/// this process by the counter, and across processes (a member that died, or
/// another member) by the seed, drawn from the clock and process id once.
fn next_token() -> u64 {
    static SEED: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let seed = *SEED.get_or_init(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64)
            ^ u64::from(std::process::id()).rotate_left(32)
    });
    // An identifier, not a quantity: wrapping keeps it distinct.
    seed.wrapping_add(NEXT.fetch_add(1, Ordering::Relaxed))
}

/// The engine-owned executor of index builds: one thread per build taken,
/// at most a bounded number filling indexes at once, the rest waiting with
/// their builds accepted.
#[derive(Clone)]
pub struct IndexBuildService {
    shared: Arc<Shared>,
}

impl IndexBuildService {
    /// A service running builds over `env` as `config` says.
    pub fn new(env: Arc<dyn BuildEnvironment>, config: IndexBuildConfig) -> Self {
        Self {
            shared: Arc::new(Shared {
                env,
                config: parking_lot::RwLock::new(config.normalized()),
                slots: parking_lot::Mutex::new(FxHashMap::default()),
                finished: parking_lot::Condvar::new(),
                running: parking_lot::Mutex::new(0),
                vacancy: parking_lot::Condvar::new(),
                phases: parking_lot::Mutex::new(FxHashMap::default()),
                covered: parking_lot::Mutex::new(FxHashMap::default()),
                stopping: core::sync::atomic::AtomicBool::new(false),
                executors: parking_lot::Mutex::new(Vec::new()),
            }),
        }
    }

    /// Stop every executor of this process and wait for each to end: a
    /// backfill stops before its next page, a build waiting for a seat does
    /// not start, and no new build starts. A build stopped this way keeps
    /// its record running, and the next process to open the storage
    /// finishes it. Call before the storage closes.
    pub fn shutdown(&self) {
        self.shared
            .stopping
            .store(true, core::sync::atomic::Ordering::Release);
        {
            // Under the lock the seat waiters check the flag with, so none
            // misses this wake.
            let _running = self.shared.running.lock();
            self.shared.vacancy.notify_all();
        }
        let executors = std::mem::take(&mut *self.shared.executors.lock());
        for executor in executors {
            // The build itself runs under `catch_unwind`; a join error is a
            // panic in settling it, which the next opening resumes from.
            if let Err(panic) = executor.join() {
                tracing::error!(
                    panic = panic_message(&*panic),
                    "an index build executor panicked while settling its build"
                );
            }
        }
    }

    /// The last node key the backfill of `generation` running on this
    /// process has committed entries through; `None` when no executor here
    /// runs it or it has committed no page yet.
    pub fn covered_through(&self, generation: GenerationId) -> Option<Vec<u8>> {
        self.shared.covered.lock().get(&generation).cloned()
    }

    /// How the service runs builds now.
    pub fn config(&self) -> IndexBuildConfig {
        *self.shared.config.read()
    }

    /// Retune the service while it runs. A higher `max_running` seats
    /// waiting builds at once; a lower one lets the running builds finish
    /// and seats no more until fewer run. A new `older_transactions_wait`
    /// applies to backfills that start after the call.
    pub fn set_config(&self, config: IndexBuildConfig) {
        *self.shared.config.write() = config.normalized();
        // Under the seat lock, so a build checking the limit either sees the
        // new one or is already waiting when the wakeup comes.
        let _running = self.shared.running.lock();
        self.shared.vacancy.notify_all();
    }

    /// Run the build of `generation` on an executor of this process, unless
    /// one already has it. A submitter that waits for the build from inside a
    /// transaction of its own releases it from the backfill's wait first
    /// ([`Transaction::release_from_schema_waits`](coordinode_storage::engine::transaction::Transaction::release_from_schema_waits)).
    ///
    /// # Errors
    ///
    /// The executor thread could not be started.
    pub fn submit(&self, generation: GenerationId) -> Result<(), String> {
        self.spawn(generation, move |shared| shared.execute(generation))
    }

    /// Run `build`, a build this member runs for itself, as the build of
    /// `generation` on an executor of this process, unless one already has
    /// it. It takes a seat like any other build, and [`Self::wait`] reports
    /// its outcome.
    ///
    /// # Errors
    ///
    /// The executor thread could not be started.
    pub fn run_local(&self, generation: GenerationId, build: LocalBuild) -> Result<(), String> {
        self.spawn(generation, move |shared| {
            let _seat = shared.seat(generation);
            shared.set_phase(generation, BuildPhase::Indexing { indexed: None });
            Executed::Outcome(match build(shared.env.as_ref()) {
                Ok(indexed) => IndexBuildOutcome::Published {
                    indexed: Some(indexed),
                },
                Err(e) => IndexBuildOutcome::Failed(e),
            })
        })
    }

    /// Run `work` for the build of `generation` on a thread of its own,
    /// unless an executor of this process already has the build, and keep
    /// its outcome for the waiters.
    fn spawn(
        &self,
        generation: GenerationId,
        work: impl FnOnce(&Arc<Shared>) -> Executed + Send + 'static,
    ) -> Result<(), String> {
        if self
            .shared
            .stopping
            .load(core::sync::atomic::Ordering::Acquire)
        {
            return Err("the index build service is shutting down".into());
        }
        {
            let mut slots = self.shared.slots.lock();
            if matches!(slots.get(&generation), Some(Slot::Running)) {
                return Ok(());
            }
            slots.insert(generation, Slot::Running);
        }
        let shared = Arc::clone(&self.shared);
        let spawned = std::thread::Builder::new()
            .name(format!("index-build-{}", generation.as_raw()))
            .spawn(move || {
                let executed =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| work(&shared)))
                        .unwrap_or_else(|panic| {
                            Executed::Outcome(IndexBuildOutcome::Failed(BuildError::Other(
                                panic_message(&*panic),
                            )))
                        });
                let slot = match executed {
                    Executed::Outcome(outcome) => Slot::Done(outcome),
                    Executed::Lost => Slot::Lost,
                };
                shared.phases.lock().remove(&generation);
                shared.covered.lock().remove(&generation);
                shared.slots.lock().insert(generation, slot);
                shared.finished.notify_all();
            });
        match spawned {
            Ok(executor) => {
                let mut executors = self.shared.executors.lock();
                // Ended executors leave the list here, so it holds only the
                // ones a shutdown may have to wait for.
                executors.retain(|e| !e.is_finished());
                executors.push(executor);
                Ok(())
            }
            Err(e) => {
                self.shared.slots.lock().remove(&generation);
                Err(format!("start the build executor: {e}"))
            }
        }
    }

    /// The outcome of the build of `generation`, waiting up to `timeout`
    /// (`None`: until it has one). `None` when it has none yet. Waiting
    /// cancels nothing.
    ///
    /// # Errors
    ///
    /// The build record could not be read.
    pub fn wait(
        &self,
        generation: GenerationId,
        timeout: Option<Duration>,
    ) -> Result<Option<IndexBuildOutcome>, StoreError> {
        let deadline = timeout.map(|t| Instant::now() + t);
        let mut slots = self.shared.slots.lock();
        loop {
            match slots.get(&generation) {
                Some(Slot::Done(outcome)) => return Ok(Some(outcome.clone())),
                Some(Slot::Running) => match deadline {
                    Some(at) => {
                        if self.shared.finished.wait_until(&mut slots, at).timed_out() {
                            return Ok(None);
                        }
                    }
                    None => self.shared.finished.wait(&mut slots),
                },
                // Another executor finishes it, or no executor of this
                // process has it: the record answers.
                Some(Slot::Lost) | None => {
                    drop(slots);
                    return self.follow_record(generation, deadline);
                }
            }
        }
    }

    /// The outcome an executor of this process reached for the build of
    /// `generation`, once it ends; `None` when none had the build, or the one
    /// that had it ended without an outcome: it could not take the build, or
    /// another executor took it over. Unlike [`Self::wait`], it does not
    /// follow the record.
    pub fn wait_here(&self, generation: GenerationId) -> Option<IndexBuildOutcome> {
        let mut slots = self.shared.slots.lock();
        loop {
            match slots.get(&generation) {
                Some(Slot::Done(outcome)) => return Some(outcome.clone()),
                Some(Slot::Running) => self.shared.finished.wait(&mut slots),
                Some(Slot::Lost) | None => return None,
            }
        }
    }

    /// Wait on the record of `generation` until it carries an outcome, up to
    /// `deadline`.
    fn follow_record(
        &self,
        generation: GenerationId,
        deadline: Option<Instant>,
    ) -> Result<Option<IndexBuildOutcome>, StoreError> {
        let store = LocalIndexStore::new(self.shared.env.engine());
        loop {
            let outcome = match store.load_build(generation)? {
                None => Some(IndexBuildOutcome::Cancelled),
                Some((record, _)) => terminal_outcome(&record.state),
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

    /// Cancel the build of `generation`: its record is cancelled and, in the
    /// same commit, an index being created is withdrawn with the constraint
    /// that owns it, and an index being rebuilt is kept, failed. `false`
    /// when the build already had an outcome. An executor running it stops
    /// at its next page. A cancellation that meets another move of the
    /// build reads the record again and retries, until the build is
    /// cancelled or has an outcome of its own.
    ///
    /// # Errors
    ///
    /// A storage failure, a commit failure other than a conflict, or moves
    /// kept winning [`MOVE_ATTEMPTS`] times in a row.
    pub fn cancel(&self, generation: GenerationId) -> Result<bool, String> {
        let env = self.shared.env.as_ref();
        let store = LocalIndexStore::new(env.engine());
        for _ in 0..MOVE_ATTEMPTS {
            let Some((mut record, version)) = store.load_build(generation).map_err(text)? else {
                return Ok(false);
            };
            if record.state.is_terminal() {
                return Ok(false);
            }
            let def = store
                .load_definition(record.index)
                .map_err(text)?
                .filter(|d| d.generation == generation);
            let def_version = store.definition_version(record.index).map_err(text)?;
            record.state = BuildState::Cancelled;
            let committed = {
                let mut txn = begin(env);
                store
                    .put_build_txn(&mut txn, &record, Some(version))
                    .map_err(text)?;
                if let Some(def) = &def {
                    store
                        .expect_definition_txn(&mut txn, def.id, def_version)
                        .map_err(text)?;
                    stage_failure(env, &mut txn, def, record.on_failure, "cancelled")
                        .map_err(text)?;
                }
                env.commit_catalog(&mut txn)
            };
            match committed {
                Ok(()) => {
                    if let Some(def) = def {
                        settle_registry(env, def, record.on_failure, "cancelled").map_err(text)?;
                    }
                    return Ok(true);
                }
                Err(e) if is_contention(&e) => continue,
                Err(e) => return Err(e.to_string()),
            }
        }
        Err(format!(
            "the build of generation {} kept moving under the cancellation",
            generation.as_raw()
        ))
    }

    /// Take up every build the catalog records without an outcome, as after
    /// a restart: each is submitted to an executor of this process. Returns
    /// their generations.
    ///
    /// # Errors
    ///
    /// The records could not be read, or an executor could not be started.
    pub fn resume(&self) -> Result<Vec<GenerationId>, String> {
        let store = LocalIndexStore::new(self.shared.env.engine());
        let mut resumed = Vec::new();
        for record in store.list_builds().map_err(text)? {
            if !record.state.is_terminal() {
                self.submit(record.generation)?;
                resumed.push(record.generation);
            }
        }
        Ok(resumed)
    }

    /// Every build the catalog records, and every build an executor of this
    /// process holds, each with its record and where it stands here.
    ///
    /// # Errors
    ///
    /// The records could not be read.
    pub fn builds(&self) -> Result<Vec<BuildStatus>, StoreError> {
        let store = LocalIndexStore::new(self.shared.env.engine());
        let records = store.list_builds()?;
        let mut phases = self.shared.phases.lock().clone();
        let mut out = Vec::with_capacity(records.len() + phases.len());
        for record in records {
            out.push(BuildStatus {
                generation: record.generation,
                phase: phases.remove(&record.generation),
                index: index_of(&store, &record)?,
                record: Some(record),
            });
        }
        out.extend(phases.into_iter().map(|(generation, phase)| BuildStatus {
            generation,
            record: None,
            phase: Some(phase),
            index: None,
        }));
        out.sort_by_key(|s| s.generation);
        Ok(out)
    }

    /// The build of `generation` as [`Self::builds`] shows it, after waiting
    /// up to `wait` for it to reach an outcome; `None` when neither the
    /// catalog records it nor an executor of this process holds it. Waiting
    /// cancels nothing.
    ///
    /// # Errors
    ///
    /// The record could not be read.
    pub fn status(
        &self,
        generation: GenerationId,
        wait: Duration,
    ) -> Result<Option<BuildStatus>, StoreError> {
        if !wait.is_zero() {
            self.wait(generation, Some(wait))?;
        }
        let store = LocalIndexStore::new(self.shared.env.engine());
        let record = store.load_build(generation)?.map(|(record, _)| record);
        let phase = self.shared.phases.lock().get(&generation).copied();
        if record.is_none() && phase.is_none() {
            return Ok(None);
        }
        let index = match &record {
            Some(record) => index_of(&store, record)?,
            None => None,
        };
        Ok(Some(BuildStatus {
            generation,
            record,
            phase,
            index,
        }))
    }
}

/// The index `record` builds, while its definition exists.
fn index_of(
    store: &LocalIndexStore<'_>,
    record: &IndexBuildRecord,
) -> Result<Option<BuildIndex>, StoreError> {
    Ok(store.load_definition(record.index)?.map(|d| BuildIndex {
        name: d.name.clone(),
        label: d.label.clone(),
    }))
}

/// What an executor took: the definition it fills, the version of its
/// record every page is bound to, the token the build record holds, and
/// whether the build repairs the duplicates it meets.
struct Taken {
    def: IndexDefinition,
    def_version: Option<u64>,
    token: u64,
    repair: Option<DuplicateRepair>,
}

impl Shared {
    /// Take the build of `generation`, fill its generation and publish the
    /// outcome.
    fn execute(self: &Arc<Self>, generation: GenerationId) -> Executed {
        let _seat = self.seat(generation);
        // A seat granted because the service is stopping starts nothing:
        // the build stays as recorded for the next process.
        if self.stopping.load(core::sync::atomic::Ordering::Acquire) {
            return Executed::Lost;
        }
        self.try_execute(generation)
            .unwrap_or_else(|e| Executed::Outcome(IndexBuildOutcome::Failed(BuildError::Other(e))))
    }

    fn try_execute(self: &Arc<Self>, generation: GenerationId) -> Result<Executed, String> {
        let taken = match self.take(generation)? {
            Ok(taken) => taken,
            Err(executed) => return Ok(executed),
        };
        // From here the record names this executor: whatever stops the fill,
        // an error or a panic, settles it, or no waiter could trust the
        // outcome it is told and a restart would resume a failed build.
        let filled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.fill(generation, &taken)
        }));
        let def = &taken.def;
        match filled {
            Ok(Ok(Ok(indexed))) => self.publish(generation, &taken, indexed),
            Ok(Ok(Err(BackfillError::Superseded))) => self.lost_to_mover(generation, taken.token),
            // The process is closing: the record stays running under this
            // token, and the next process to open the storage takes it over.
            Ok(Ok(Err(BackfillError::Stopped))) => Ok(Executed::Lost),
            Ok(Ok(Err(BackfillError::Duplicate(v)))) => {
                self.fail(generation, &taken, BuildError::Duplicate(v))
            }
            Ok(Ok(Err(other))) => self.fail(
                generation,
                &taken,
                BuildError::Other(format!("build index '{def}': {other}")),
            ),
            Ok(Err(e)) => self.fail(
                generation,
                &taken,
                BuildError::Other(format!("build index '{def}': {e}")),
            ),
            Err(panic) => self.fail(
                generation,
                &taken,
                BuildError::Other(format!("build index '{def}': {}", panic_message(&*panic))),
            ),
        }
    }

    /// Fill the generation `taken` holds. The outer error is the
    /// environment's, the inner the backfill's.
    fn fill(
        self: &Arc<Self>,
        generation: GenerationId,
        taken: &Taken,
    ) -> Result<Result<u64, BackfillError>, String> {
        let env = self.env.as_ref();
        let fields = env.fields()?;
        // A repair is a statement of its own, whose unique admission reads
        // how far this build has covered the label through the service.
        let service = IndexBuildService {
            shared: Arc::clone(self),
        };
        let repair_property = taken.repair.as_ref().map(|r| r.property.as_str());
        let run_repair = |node: NodeId, old: Option<&Value>| {
            super::repair::rename_duplicate(
                &service,
                env,
                &super::repair::RepairBuild {
                    generation,
                    token: taken.token,
                    index: &taken.def,
                    property: repair_property.unwrap_or_default(),
                },
                node,
                old,
                &super::repair::suffix,
            )
            .map(|_| ())
        };
        let repair = repair_property.map(|property| DuplicateRepairer {
            property,
            run: &run_repair,
        });
        let progress = |p: BackfillProgress| {
            self.set_phase(
                generation,
                match p {
                    BackfillProgress::AwaitingOlderTransactions => {
                        BuildPhase::AwaitingOlderTransactions
                    }
                    BackfillProgress::Indexed(indexed) => BuildPhase::Indexing {
                        indexed: Some(indexed),
                    },
                },
            );
        };
        let covered = |through: &[u8]| {
            self.covered.lock().insert(generation, through.to_vec());
        };
        // Read before the backfill starts: a guard taken inside the expression
        // below would live as long as the whole build, and a retune waiting
        // for it would hold up every reader of the configuration.
        let older_transactions_wait = self.config.read().older_transactions_wait;
        Ok(Backfill {
            engine: env.engine(),
            oracle: env.oracle(),
            interner: &fields,
            shard_id: env.shard_id(),
            definition_version: taken.def_version,
            older_transactions_wait,
            progress: Some(&progress),
            covered: Some(&covered),
            repair,
            stop: Some(&self.stopping),
        }
        .run(&taken.def, &mut |txn| env.commit_page(txn)))
    }

    /// Take the build of `generation`: its record moves to running under a
    /// fresh token, conditioned on the record as read. `Err` with how the
    /// attempt ended when there is nothing to take: the build has an outcome,
    /// its index is gone (the record is then removed and the build reported
    /// cancelled), or it kept moving.
    fn take(&self, generation: GenerationId) -> Result<Result<Taken, Executed>, String> {
        let env = self.env.as_ref();
        let store = LocalIndexStore::new(env.engine());
        for _ in 0..MOVE_ATTEMPTS {
            let Some((mut record, version)) = store.load_build(generation).map_err(text)? else {
                return Ok(Err(Executed::Outcome(IndexBuildOutcome::Cancelled)));
            };
            if let Some(outcome) = terminal_outcome(&record.state) {
                return Ok(Err(Executed::Outcome(outcome)));
            }
            let def = store
                .load_definition(record.index)
                .map_err(text)?
                .filter(|d| d.generation == generation);
            let def_version = store.definition_version(record.index).map_err(text)?;
            let token = next_token();
            // Ended before the backfill starts, which waits for every
            // transaction older than its boundary to end.
            let committed = {
                let mut txn = begin(env);
                match &def {
                    Some(_) => {
                        record.state = BuildState::Running { executor: token };
                        store
                            .put_build_txn(&mut txn, &record, Some(version))
                            .map_err(text)?;
                    }
                    // The index was dropped or moved to another generation:
                    // nothing is left to build, and the record goes with it.
                    None => store
                        .delete_build_txn(&mut txn, generation, version)
                        .map_err(text)?,
                }
                env.commit_catalog(&mut txn)
            };
            match (committed, def) {
                (Ok(()), Some(def)) => {
                    return Ok(Ok(Taken {
                        def,
                        def_version,
                        token,
                        repair: record.on_duplicate,
                    }));
                }
                (Ok(()), None) => return Ok(Err(Executed::Outcome(IndexBuildOutcome::Cancelled))),
                (Err(e), _) if is_contention(&e) => continue,
                // Nothing moved: the build is the leader's to take, and the
                // record tells this member's waiters how it ends.
                (Err(e), _) if cannot_move_here(&e) => return Ok(Err(Executed::Lost)),
                (Err(e), _) => return Err(e.to_string()),
            }
        }
        Err(format!(
            "the build of generation {} kept moving while it was being taken",
            generation.as_raw()
        ))
    }

    /// Publish the taken index ready with its record moved to published, in
    /// one commit conditioned on both records as taken.
    fn publish(
        &self,
        generation: GenerationId,
        taken: &Taken,
        indexed: u64,
    ) -> Result<Executed, String> {
        let env = self.env.as_ref();
        let store = LocalIndexStore::new(env.engine());
        let mut def = taken.def.clone();
        def.state = IndexState::Ready;
        let concluded = self.conclude(generation, taken, |txn, record| {
            record.state = BuildState::Published;
            record.indexed = indexed;
            store.put_definition_txn(txn, &def)?;
            if let Some(owner) = &def.owner {
                crate::executor::runner::stage_constraint_activation(
                    env.engine(),
                    txn,
                    &def.label,
                    owner,
                )?;
            }
            Ok(())
        })?;
        match concluded {
            Concluded::Landed => {
                env.registry()
                    .register_published(env.engine(), def)
                    .map_err(|e| e.to_string())?;
                Ok(Executed::Outcome(IndexBuildOutcome::Published {
                    indexed: Some(indexed),
                }))
            }
            Concluded::Ended(outcome) => Ok(Executed::Outcome(outcome)),
            Concluded::Lost => Ok(Executed::Lost),
        }
    }

    /// Settle a build whose fill stopped with `error`: per its record, the
    /// index is withdrawn with its constraint or kept failed, with the
    /// record moved to failed, in one commit conditioned on both records as
    /// taken.
    fn fail(
        &self,
        generation: GenerationId,
        taken: &Taken,
        error: BuildError,
    ) -> Result<Executed, String> {
        let env = self.env.as_ref();
        let def = &taken.def;
        let reason = error.to_string();
        let mut on_failure = BuildFailure::Withdraw;
        let concluded = self.conclude(generation, taken, |txn, record| {
            record.state = BuildState::Failed {
                reason: reason.clone(),
            };
            on_failure = record.on_failure;
            stage_failure(env, txn, def, record.on_failure, &reason)
        });
        match concluded {
            Ok(Concluded::Landed) => {
                settle_registry(env, def.clone(), on_failure, &reason).map_err(text)?;
                Ok(Executed::Outcome(IndexBuildOutcome::Failed(error)))
            }
            Ok(Concluded::Ended(outcome)) => Ok(Executed::Outcome(outcome)),
            Ok(Concluded::Lost) => Ok(Executed::Lost),
            // The failure stays the answer, and what was left behind is
            // named: the index is still published, still maintained.
            Err(e) => Ok(Executed::Outcome(IndexBuildOutcome::Failed(
                BuildError::Other(format!(
                    "{reason}; the index '{def}' was not withdrawn: {e}"
                )),
            ))),
        }
    }

    /// Commit the final move of the build the executor holding `taken`
    /// took: `stage` sets the record's terminal state and stages the
    /// catalog effects; the commit is conditioned on the build record still
    /// holding the executor's token and on the definition as taken, and is
    /// retried while a conflicting move leaves the build the executor's.
    fn conclude(
        &self,
        generation: GenerationId,
        taken: &Taken,
        mut stage: impl FnMut(&mut Transaction<'_>, &mut IndexBuildRecord) -> Result<(), StoreError>,
    ) -> Result<Concluded, String> {
        let env = self.env.as_ref();
        let store = LocalIndexStore::new(env.engine());
        for _ in 0..MOVE_ATTEMPTS {
            let Some((mut record, version)) = store.load_build(generation).map_err(text)? else {
                return Ok(Concluded::Ended(IndexBuildOutcome::Cancelled));
            };
            if record.state
                != (BuildState::Running {
                    executor: taken.token,
                })
            {
                return Ok(match terminal_outcome(&record.state) {
                    Some(outcome) => Concluded::Ended(outcome),
                    None => Concluded::Lost,
                });
            }
            let committed = {
                let mut txn = begin(env);
                // Conditions before the writes they guard: a direct-mode
                // engine decides a condition when it is stated and applies
                // each write as it is staged.
                store
                    .expect_definition_txn(&mut txn, taken.def.id, taken.def_version)
                    .map_err(text)?;
                stage(&mut txn, &mut record).map_err(text)?;
                store
                    .put_build_txn(&mut txn, &record, Some(version))
                    .map_err(text)?;
                env.commit_catalog(&mut txn)
            };
            match committed {
                Ok(()) => return Ok(Concluded::Landed),
                Err(e) if is_contention(&e) => continue,
                // The record still names this executor; the next leader
                // takes the build over and concludes it.
                Err(e) if cannot_move_here(&e) => return Ok(Concluded::Lost),
                Err(e) => return Err(e.to_string()),
            }
        }
        Err(format!(
            "the build of generation {} kept moving while it was being concluded",
            generation.as_raw()
        ))
    }

    /// The outcome after the executor holding `token` found its build moved:
    /// the outcome the record carries, or [`Executed::Lost`] while another
    /// executor still runs it.
    fn lost_to_mover(&self, generation: GenerationId, token: u64) -> Result<Executed, String> {
        let env = self.env.as_ref();
        let store = LocalIndexStore::new(env.engine());
        for _ in 0..MOVE_ATTEMPTS {
            let Some((record, version)) = store.load_build(generation).map_err(text)? else {
                return Ok(Executed::Outcome(IndexBuildOutcome::Cancelled));
            };
            if let Some(outcome) = terminal_outcome(&record.state) {
                return Ok(Executed::Outcome(outcome));
            }
            if record.state != (BuildState::Running { executor: token }) {
                return Ok(Executed::Lost);
            }
            let present = store
                .load_definition(record.index)
                .map_err(text)?
                .is_some_and(|d| d.generation == generation);
            if present {
                return Ok(Executed::Outcome(IndexBuildOutcome::Failed(
                    BuildError::Other(
                        "the index definition changed while it was being built".to_string(),
                    ),
                )));
            }
            // The index was dropped: the build ends with it, and its record
            // is this executor's to remove.
            let committed = {
                let mut txn = begin(env);
                store
                    .delete_build_txn(&mut txn, generation, version)
                    .map_err(text)?;
                env.commit_catalog(&mut txn)
            };
            match committed {
                Ok(()) => return Ok(Executed::Outcome(IndexBuildOutcome::Cancelled)),
                Err(e) if is_contention(&e) => continue,
                Err(e) => return Err(e.to_string()),
            }
        }
        Err(format!(
            "the build of generation {} kept moving after its index was dropped",
            generation.as_raw()
        ))
    }

    /// A seat for the build of `generation` among the builds filling
    /// indexes now, waited for while all are taken; the build is shown
    /// awaiting a seat meanwhile. Released when dropped.
    fn seat(&self, generation: GenerationId) -> Seat<'_> {
        let mut running = self.running.lock();
        if *running >= self.config.read().max_running {
            self.set_phase(generation, BuildPhase::AwaitingSeat);
            while *running >= self.config.read().max_running
                && !self.stopping.load(core::sync::atomic::Ordering::Acquire)
            {
                self.vacancy.wait(&mut running);
            }
        }
        *running += 1;
        self.set_phase(generation, BuildPhase::Indexing { indexed: None });
        Seat { shared: self }
    }

    /// Record where the build of `generation` stands on this process.
    fn set_phase(&self, generation: GenerationId, phase: BuildPhase) {
        self.phases.lock().insert(generation, phase);
    }
}

struct Seat<'a> {
    shared: &'a Shared,
}

impl Drop for Seat<'_> {
    fn drop(&mut self) {
        *self.shared.running.lock() -= 1;
        self.shared.vacancy.notify_one();
    }
}

/// Whether a failed commit met another move, so reading the record again
/// may let the mover retry.
fn is_contention(e: &CommitError) -> bool {
    matches!(
        e,
        CommitError::Conflict(_) | CommitError::RevisionMismatch { .. }
    )
}

/// Whether a failed commit was refused because this member takes no writes
/// now (it does not lead, or runs another version than its group): nothing
/// moved, and the member that does takes the build.
fn cannot_move_here(e: &CommitError) -> bool {
    matches!(
        e,
        CommitError::NotLeader { .. } | CommitError::Mismatched(_)
    )
}

/// The waiter's view of a terminal state, `None` for a build in progress.
fn terminal_outcome(state: &BuildState) -> Option<IndexBuildOutcome> {
    match state {
        BuildState::Published => Some(IndexBuildOutcome::Published { indexed: None }),
        BuildState::Failed { reason } => {
            Some(IndexBuildOutcome::Failed(BuildError::Other(reason.clone())))
        }
        BuildState::Cancelled => Some(IndexBuildOutcome::Cancelled),
        BuildState::Accepted | BuildState::Running { .. } => None,
    }
}

/// Stage what a build that ends without its index leaves: a created index
/// withdrawn, its entries cleared and its constraint withdrawn; a rebuilt
/// one kept with the definition failed, so its constraint still holds for
/// new writes.
fn stage_failure(
    env: &dyn BuildEnvironment,
    txn: &mut Transaction<'_>,
    def: &IndexDefinition,
    on_failure: BuildFailure,
    reason: &str,
) -> Result<(), StoreError> {
    let store = LocalIndexStore::new(env.engine());
    match on_failure {
        BuildFailure::Withdraw => {
            store.delete_definition_txn(txn, def)?;
            store.clear_txn(txn, def.generation)?;
            if let Some(owner) = &def.owner {
                crate::executor::runner::stage_constraint_withdrawal(
                    env.engine(),
                    txn,
                    &def.label,
                    owner,
                )?;
            }
        }
        BuildFailure::Keep => {
            let mut failed = def.clone();
            failed.state = IndexState::Failed {
                reason: reason.to_string(),
            };
            store.put_definition_txn(txn, &failed)?;
        }
    }
    Ok(())
}

/// Bring this process's registry in line with a committed failure or
/// cancellation of `def`'s build.
fn settle_registry(
    env: &dyn BuildEnvironment,
    def: IndexDefinition,
    on_failure: BuildFailure,
    reason: &str,
) -> Result<(), StoreError> {
    match on_failure {
        BuildFailure::Withdraw => env.registry().unregister(def.id),
        BuildFailure::Keep => {
            let mut failed = def;
            failed.state = IndexState::Failed {
                reason: reason.to_string(),
            };
            env.registry().register_published(env.engine(), failed)?;
        }
    }
    Ok(())
}

/// A transaction for one catalog move of a build.
fn begin(env: &dyn BuildEnvironment) -> Transaction<'_> {
    match env.oracle() {
        Some(oracle) => Transaction::begin(env.engine(), Some(oracle), oracle.next()),
        None => Transaction::new(env.engine(), None, Timestamp::ZERO, None),
    }
}

fn text(e: StoreError) -> String {
    e.to_string()
}

#[cfg(test)]
pub(crate) mod test_env;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
