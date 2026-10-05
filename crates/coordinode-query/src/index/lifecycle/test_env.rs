//! A build environment over a real engine, for tests of this crate: one
//! member whose commits go straight to the engine, as an embedded database
//! without a log commits them.

use std::sync::Arc;

use coordinode_core::graph::intern::FieldInterner;
use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::transaction::{CommitContext, CommitError, Transaction};

use super::{BuildEnvironment, IndexBuildService};
use crate::index::IndexRegistry;

/// The engine, oracle, field dictionary and registry of one test member.
pub(crate) struct TestEnv {
    pub(crate) engine: Arc<StorageEngine>,
    pub(crate) oracle: Option<Arc<TimestampOracle>>,
    fields: parking_lot::Mutex<FieldInterner>,
    pub(crate) registry: IndexRegistry,
    _dir: Option<tempfile::TempDir>,
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
            _dir: None,
        })
    }

    /// Make `fields` the dictionary builds resolve properties through.
    pub(crate) fn set_fields(&self, fields: &FieldInterner) {
        *self.fields.lock() = fields.clone();
    }

    /// The field dictionary builds resolve properties through.
    pub(crate) fn fields_now(&self) -> FieldInterner {
        self.fields.lock().clone()
    }

    /// A build service over this member.
    pub(crate) fn service(self: &Arc<Self>, max_running: usize) -> IndexBuildService {
        IndexBuildService::new(Arc::clone(self) as Arc<dyn BuildEnvironment>, max_running)
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
        Ok(self.fields_now())
    }

    fn shard_id(&self) -> u16 {
        1
    }

    fn registry(&self) -> &IndexRegistry {
        &self.registry
    }

    fn commit_page(&self, txn: &mut Transaction<'_>) -> Result<(), CommitError> {
        commit(txn)
    }

    fn commit_catalog(&self, txn: &mut Transaction<'_>) -> Result<(), CommitError> {
        commit(txn)
    }
}
