//! The build environment of an embedded database: index builds commit
//! through the database's write pipeline, so their pages and catalog moves
//! replicate like any other write, and read the field dictionary as it
//! stands when they run.

use std::sync::Arc;

use coordinode_core::graph::intern::FieldInterner;
use coordinode_core::txn::proposal::{ProposalIdGenerator, ProposalPipeline};
use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_query::index::{BuildEnvironment, IndexRegistry, TextIndexRegistry};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::transaction::{CommitContext, CommitError, Transaction};

use super::fields::FieldDictionary;

/// Builds filling indexes at once; the others wait with their builds
/// accepted.
pub(super) const MAX_RUNNING: usize = 2;

/// What a build of this database runs over.
pub(super) struct DatabaseBuilds {
    pub(super) engine: Arc<StorageEngine>,
    pub(super) oracle: Arc<TimestampOracle>,
    pub(super) pipeline: Arc<dyn ProposalPipeline>,
    pub(super) proposal_id_gen: Arc<ProposalIdGenerator>,
    pub(super) fields: Arc<FieldDictionary>,
    pub(super) registry: Arc<IndexRegistry>,
    pub(super) text_registry: Arc<TextIndexRegistry>,
    pub(super) shard_id: u16,
}

impl DatabaseBuilds {
    /// Commit `txn` through the pipeline at majority: a build's writes are
    /// durable whatever concern a session writes its own data at.
    fn commit(&self, txn: &mut Transaction<'_>) -> Result<(), CommitError> {
        let wc = WriteConcern::majority();
        txn.commit(&CommitContext {
            write_concern: &wc,
            pipeline: Some(self.pipeline.as_ref()),
            id_gen: Some(&self.proposal_id_gen),
            drain_buffer: None,
            nvme_write_buffer: None,
        })
        .map(|_| ())
    }
}

impl BuildEnvironment for DatabaseBuilds {
    fn engine(&self) -> &StorageEngine {
        &self.engine
    }

    fn oracle(&self) -> Option<&TimestampOracle> {
        Some(&self.oracle)
    }

    fn fields(&self) -> Result<FieldInterner, String> {
        self.fields.current().map_err(|e| e.to_string())
    }

    fn shard_id(&self) -> u16 {
        self.shard_id
    }

    fn registry(&self) -> &IndexRegistry {
        &self.registry
    }

    fn text_registry(&self) -> Option<&TextIndexRegistry> {
        Some(&self.text_registry)
    }

    fn commit_page(&self, txn: &mut Transaction<'_>) -> Result<(), CommitError> {
        self.commit(txn)
    }

    fn commit_catalog(&self, txn: &mut Transaction<'_>) -> Result<(), CommitError> {
        self.commit(txn)
    }
}
