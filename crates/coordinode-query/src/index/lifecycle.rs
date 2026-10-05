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
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_modality::{
    BuildFailure, BuildState, IndexBuildRecord, IndexStore as _, LocalIndexStore, StoreError,
};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::transaction::{CommitError, Transaction};
use rustc_hash::FxHashMap;

use super::build::{Backfill, BackfillError};
use super::definition::{GenerationId, IndexDefinition, IndexState};
use super::registry::{IndexRegistry, UniqueViolation};

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
    slots: parking_lot::Mutex<FxHashMap<GenerationId, Slot>>,
    finished: parking_lot::Condvar,
    /// Builds executing now, bounded by `max_running`.
    running: parking_lot::Mutex<usize>,
    vacancy: parking_lot::Condvar,
    max_running: usize,
}

/// Attempts one move of a build makes against other moves of the same
/// build. A build moves a few times in its life (taken, published or
/// failed, cancelled), so a mover that loses this many races in a row faces
/// a fault, not contention.
pub const MOVE_ATTEMPTS: usize = 16;

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
    /// A service running builds over `env`, at most `max_running` filling at
    /// once (at least one).
    pub fn new(env: Arc<dyn BuildEnvironment>, max_running: usize) -> Self {
        Self {
            shared: Arc::new(Shared {
                env,
                slots: parking_lot::Mutex::new(FxHashMap::default()),
                finished: parking_lot::Condvar::new(),
                running: parking_lot::Mutex::new(0),
                vacancy: parking_lot::Condvar::new(),
                max_running: max_running.max(1),
            }),
        }
    }

    /// Run the build of `generation` on an executor of this process, unless
    /// one already has it. `own_open` is how many transactions the submitter
    /// itself holds open while it waits (the statement that created the
    /// index); the backfill does not wait for them to end.
    ///
    /// # Errors
    ///
    /// The executor thread could not be started.
    pub fn submit(&self, generation: GenerationId, own_open: usize) -> Result<(), String> {
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
                let executed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    shared.execute(generation, own_open)
                }))
                .unwrap_or_else(|panic| {
                    let reason = panic
                        .downcast_ref::<&'static str>()
                        .map(|s| (*s).to_string())
                        .or_else(|| panic.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "panic in the build executor".to_string());
                    Executed::Outcome(IndexBuildOutcome::Failed(BuildError::Other(reason)))
                });
                let slot = match executed {
                    Executed::Outcome(outcome) => Slot::Done(outcome),
                    Executed::Lost => Slot::Lost,
                };
                shared.slots.lock().insert(generation, slot);
                shared.finished.notify_all();
            });
        if let Err(e) = spawned {
            self.shared.slots.lock().remove(&generation);
            return Err(format!("start the build executor: {e}"));
        }
        Ok(())
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
                self.submit(record.generation, 0)?;
                resumed.push(record.generation);
            }
        }
        Ok(resumed)
    }

    /// The builds the catalog records.
    ///
    /// # Errors
    ///
    /// The records could not be read.
    pub fn builds(&self) -> Result<Vec<IndexBuildRecord>, StoreError> {
        LocalIndexStore::new(self.shared.env.engine()).list_builds()
    }
}

/// What an executor took: the definition it fills, the version of its
/// record every page is bound to, and the token the build record holds.
struct Taken {
    def: IndexDefinition,
    def_version: Option<u64>,
    token: u64,
}

impl Shared {
    /// Take the build of `generation`, fill its generation and publish the
    /// outcome.
    fn execute(&self, generation: GenerationId, own_open: usize) -> Executed {
        let _seat = self.seat();
        self.try_execute(generation, own_open)
            .unwrap_or_else(|e| Executed::Outcome(IndexBuildOutcome::Failed(BuildError::Other(e))))
    }

    fn try_execute(&self, generation: GenerationId, own_open: usize) -> Result<Executed, String> {
        let taken = match self.take(generation)? {
            Ok(taken) => taken,
            Err(executed) => return Ok(executed),
        };
        let env = self.env.as_ref();
        let fields = env.fields()?;
        let built = Backfill {
            engine: env.engine(),
            oracle: env.oracle(),
            interner: &fields,
            shard_id: env.shard_id(),
            own_open,
            definition_version: taken.def_version,
        }
        .run(&taken.def, &mut |txn| env.commit_page(txn));

        match built {
            Ok(indexed) => self.publish(generation, &taken, indexed),
            Err(failure) => self.fail(generation, &taken, failure),
        }
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
                    }));
                }
                (Ok(()), None) => return Ok(Err(Executed::Outcome(IndexBuildOutcome::Cancelled))),
                (Err(e), _) if is_contention(&e) => continue,
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

    /// Settle a build whose backfill stopped with `failure`: per its record,
    /// the index is withdrawn with its constraint or kept failed, with the
    /// record moved to failed, in one commit conditioned on both records as
    /// taken. A backfill that stopped because its definition moved (a
    /// cancellation, a drop) defers to whoever moved it.
    fn fail(
        &self,
        generation: GenerationId,
        taken: &Taken,
        failure: BackfillError,
    ) -> Result<Executed, String> {
        let env = self.env.as_ref();
        let def = &taken.def;
        if matches!(failure, BackfillError::Superseded) {
            return self.lost_to_mover(generation, taken.token);
        }
        let error = match failure {
            BackfillError::Duplicate(v) => BuildError::Duplicate(v),
            other => BuildError::Other(format!("build index '{def}': {other}")),
        };
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

    /// A seat among the builds filling indexes now, waited for while all are
    /// taken. Released when dropped.
    fn seat(&self) -> Seat<'_> {
        let mut running = self.running.lock();
        while *running >= self.max_running {
            self.vacancy.wait(&mut running);
        }
        *running += 1;
        Seat { shared: self }
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
