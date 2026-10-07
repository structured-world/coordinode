//! A build environment over a real engine, for tests of this crate: one
//! member whose commits go straight to the engine, as an embedded database
//! without a log commits them.

use std::sync::Arc;

use coordinode_core::graph::intern::FieldInterner;
use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::transaction::{CommitContext, CommitError, Transaction};

use super::{BuildEnvironment, IndexBuildConfig, IndexBuildService};
use crate::index::IndexRegistry;

/// The engine, oracle, field dictionary and registry of one test member.
pub(crate) struct TestEnv {
    pub(crate) engine: Arc<StorageEngine>,
    pub(crate) oracle: Option<Arc<TimestampOracle>>,
    fields: parking_lot::Mutex<FieldInterner>,
    pub(crate) registry: IndexRegistry,
    fault: parking_lot::Mutex<Option<Fault>>,
    hold: PageHold,
    _dir: Option<tempfile::TempDir>,
}

/// Backfill pages held before they commit, once a number of them has.
#[derive(Default)]
struct PageHold {
    /// Pages committed so far, the count past which the next waits, and
    /// whether one is waiting now.
    state: parking_lot::Mutex<(usize, Option<usize>, bool)>,
    moved: parking_lot::Condvar,
}

/// A fault a test member injects into the builds it runs.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Fault {
    /// The field dictionary cannot be read.
    FieldsUnavailable,
    /// Committing a backfill page panics.
    PanicOnPage,
}

impl TestEnv {
    /// A member over a new engine with an oracle, in a directory of its own.
    #[allow(clippy::expect_used)]
    pub(crate) fn open() -> Arc<Self> {
        use coordinode_storage::engine::config::{
            Durability, EndpointConfig, Media, StorageConfig, Tier,
        };
        let dir = tempfile::tempdir().expect("tempdir");
        let oracle = Arc::new(TimestampOracle::resume_from(
            coordinode_core::txn::timestamp::Timestamp::from_raw(1),
        ));
        let engine = Arc::new(
            StorageEngine::open_with_oracle(
                &StorageConfig::with_endpoints(vec![EndpointConfig::new(
                    "default",
                    dir.path(),
                    Media::Hdd,
                    Durability::Durable,
                    Tier::Warm,
                )]),
                Arc::clone(&oracle),
            )
            .expect("open engine"),
        );
        Arc::new(Self {
            engine,
            oracle: Some(oracle),
            fields: parking_lot::Mutex::new(FieldInterner::new()),
            registry: IndexRegistry::new(),
            fault: parking_lot::Mutex::new(None),
            hold: PageHold::default(),
            _dir: Some(dir),
        })
    }

    /// A member over `engine`, with `oracle` when it runs one.
    pub(crate) fn over(
        engine: Arc<StorageEngine>,
        oracle: Option<Arc<TimestampOracle>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            engine,
            oracle,
            fields: parking_lot::Mutex::new(FieldInterner::new()),
            registry: IndexRegistry::new(),
            fault: parking_lot::Mutex::new(None),
            hold: PageHold::default(),
            _dir: None,
        })
    }

    /// Hold every backfill page after the first `pages` before it commits,
    /// until [`Self::release_pages`].
    pub(crate) fn hold_pages_after(&self, pages: usize) {
        self.hold.state.lock().1 = Some(pages);
    }

    /// Wait until a backfill page is held: every page before it is
    /// committed and reported.
    pub(crate) fn await_held_page(&self) {
        let mut state = self.hold.state.lock();
        while !state.2 {
            self.hold.moved.wait(&mut state);
        }
    }

    /// Let held backfill pages commit.
    pub(crate) fn release_pages(&self) {
        self.hold.state.lock().1 = None;
        self.hold.moved.notify_all();
    }

    /// Make `fields` the dictionary builds resolve properties through.
    pub(crate) fn set_fields(&self, fields: &FieldInterner) {
        *self.fields.lock() = fields.clone();
    }

    /// Make the builds this member runs from now on meet `fault`.
    pub(crate) fn inject(&self, fault: Option<Fault>) {
        *self.fault.lock() = fault;
    }

    /// The field dictionary builds resolve properties through.
    pub(crate) fn fields_now(&self) -> FieldInterner {
        self.fields.lock().clone()
    }

    /// A build service over this member.
    pub(crate) fn service(self: &Arc<Self>, max_running: usize) -> IndexBuildService {
        self.service_with(IndexBuildConfig {
            max_running,
            ..IndexBuildConfig::default()
        })
    }

    /// A build service over this member configured as `config` says.
    pub(crate) fn service_with(self: &Arc<Self>, config: IndexBuildConfig) -> IndexBuildService {
        IndexBuildService::new(Arc::clone(self) as Arc<dyn BuildEnvironment>, config)
    }

    /// A transaction as the member opens one.
    pub(crate) fn begin(&self) -> Transaction<'_> {
        match &self.oracle {
            Some(oracle) => Transaction::begin(&self.engine, Some(oracle), oracle.next()),
            None => Transaction::new(
                &self.engine,
                None,
                coordinode_core::txn::timestamp::Timestamp::ZERO,
                None,
            ),
        }
    }
}

/// Commit `txn` straight to the engine.
pub(crate) fn commit(txn: &mut Transaction<'_>) -> Result<(), CommitError> {
    let wc = WriteConcern::majority();
    txn.commit(&CommitContext {
        write_concern: &wc,
        pipeline: None,
        id_gen: None,
        drain_buffer: None,
        nvme_write_buffer: None,
    })
    .map(|_| ())
}

impl BuildEnvironment for TestEnv {
    fn engine(&self) -> &StorageEngine {
        &self.engine
    }

    fn oracle(&self) -> Option<&TimestampOracle> {
        self.oracle.as_deref()
    }

    fn fields(&self) -> Result<FieldInterner, String> {
        if matches!(*self.fault.lock(), Some(Fault::FieldsUnavailable)) {
            return Err("field dictionary unavailable".into());
        }
        Ok(self.fields_now())
    }

    fn shard_id(&self) -> u16 {
        1
    }

    fn registry(&self) -> &IndexRegistry {
        &self.registry
    }

    fn text_registry(&self) -> Option<&crate::index::TextIndexRegistry> {
        None
    }

    // The injected fault is a panic: what the executor must survive.
    #[allow(clippy::panic)]
    fn commit_page(&self, txn: &mut Transaction<'_>) -> Result<(), CommitError> {
        if matches!(*self.fault.lock(), Some(Fault::PanicOnPage)) {
            panic!("page commit broke");
        }
        {
            let mut state = self.hold.state.lock();
            while state.1.is_some_and(|after| state.0 >= after) {
                // Tell a waiter the page is held, then wait for the release.
                state.2 = true;
                self.hold.moved.notify_all();
                self.hold.moved.wait(&mut state);
            }
            state.2 = false;
        }
        commit(txn)?;
        self.hold.state.lock().0 += 1;
        Ok(())
    }

    fn commit_catalog(&self, txn: &mut Transaction<'_>) -> Result<(), CommitError> {
        commit(txn)
    }

    fn statement_log(
        &self,
    ) -> Option<(
        &dyn coordinode_core::txn::proposal::ProposalPipeline,
        &coordinode_core::txn::proposal::ProposalIdGenerator,
    )> {
        None
    }
}
