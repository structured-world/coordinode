//! Change-stream consumers of a database that is not a member of a
//! replicated group: their registrations, and the history the store keeps
//! for them.
//!
//! Each consumer registers with an explicit retention policy. A STRICT one
//! keeps what it needs until it is cancelled; a BOUNDED one until a declared
//! bound is crossed, when its registration ends with the reason. The store's
//! journal (for log consumers) and its MVCC history (for the others) are
//! kept from the lowest position a live registration needs. The registry is
//! the one the server runs; here it runs over the store's own journal, with
//! its background service on a thread of its own, so a database without an
//! async runtime has it.

use std::sync::Arc;

use coordinode_core::txn::proposal::{ProposalIdGenerator, ProposalPipeline};
use coordinode_replicate::{
    BackgroundConfig, ConsumerRegistration, ConsumerSnapshot, LogRetentionSource, RegisteredHandle,
    RegistryBackground, SeqnoConsumerRegistry, ShardConsumerRegistry, SystemClock,
};
use coordinode_storage::engine::core::StorageEngine;

use super::{Database, DatabaseError};

/// The consumer registry of a database and its background service.
pub(crate) struct Consumers {
    registry: ShardConsumerRegistry,
    /// Stopped and waited for when the database closes: its threads hold the
    /// store, which a reopen of the same directory needs released.
    service: Option<RegistryBackground>,
}

impl Drop for Consumers {
    fn drop(&mut self) {
        if let Some(service) = self.service.take() {
            service.stop();
        }
    }
}

impl Consumers {
    /// The registry over `engine`'s own journal and MVCC history, writing
    /// its records through `pipeline`. `None` for a store that keeps no
    /// journal of its own (a member of a replicated group, an in-memory
    /// store), which holds no history for consumers this way.
    ///
    /// # Errors
    ///
    /// The journal could not be inspected, or the service's thread could not
    /// be started.
    pub(crate) fn open(
        engine: &Arc<StorageEngine>,
        pipeline: &Arc<dyn ProposalPipeline>,
    ) -> Result<Option<Self>, DatabaseError> {
        let Some(source) = LogRetentionSource::over_own_journal(Arc::clone(engine))? else {
            return Ok(None);
        };
        let registry = ShardConsumerRegistry::new(
            Arc::clone(engine),
            Arc::clone(pipeline),
            Arc::new(ProposalIdGenerator::with_base(
                coordinode_core::txn::proposal::fresh_proposal_id_base(),
            )),
            Arc::new(SystemClock),
            Arc::new(source),
        );
        let service = registry.start_background(BackgroundConfig::default())?;
        Ok(Some(Self {
            registry,
            service: Some(service),
        }))
    }
}

impl Database {
    /// The registry, or why this database has none.
    fn consumers(&self) -> Result<&ShardConsumerRegistry, DatabaseError> {
        self.consumers.as_ref().map(|c| &c.registry).ok_or_else(|| {
            DatabaseError::Unsupported(
                "this database keeps no journal of its own to hold history for consumers".into(),
            )
        })
    }

    /// Register a change-stream consumer with the retention policy it chose,
    /// returning the handle of its new registration.
    ///
    /// Its starting position is held from here on: a log consumer's in the
    /// store's journal, any other's in the MVCC history.
    ///
    /// # Errors
    ///
    /// [`DatabaseError::Consumer`] when the id is empty or already live, the
    /// policy cannot be admitted, the starting position is no longer held
    /// (`RetentionLost`) or the store is under pressure (`Backpressure`,
    /// retryable); [`DatabaseError::Unsupported`] on a database without a
    /// journal of its own.
    pub fn register_consumer(
        &self,
        registration: ConsumerRegistration,
    ) -> Result<RegisteredHandle, DatabaseError> {
        Ok(self.consumers()?.register(registration)?)
    }

    /// Take up a live registration again, as a consumer restarting does.
    ///
    /// # Errors
    ///
    /// [`DatabaseError::Consumer`] when the id was never registered, another
    /// incarnation holds it, or this incarnation has ended.
    pub fn resume_consumer(
        &self,
        consumer_id: &str,
        incarnation: u64,
    ) -> Result<RegisteredHandle, DatabaseError> {
        Ok(self.consumers()?.resume(consumer_id, incarnation)?)
    }

    /// Acknowledge that the consumer holds everything before `position`;
    /// what lies below it is no longer kept for it. A lower position than
    /// already acknowledged changes nothing.
    ///
    /// # Errors
    ///
    /// [`DatabaseError::Consumer`] when the handle no longer addresses a live
    /// registration.
    pub fn acknowledge_consumer(
        &self,
        handle: &RegisteredHandle,
        position: u64,
    ) -> Result<(), DatabaseError> {
        Ok(self.consumers()?.checkpoint(handle, position)?)
    }

    /// Record that the consumer is alive. It proves liveness only: it never
    /// acknowledges anything or resets how long the consumer's work has
    /// waited.
    ///
    /// # Errors
    ///
    /// [`DatabaseError::Consumer`] when the handle no longer addresses a live
    /// registration.
    pub fn consumer_heartbeat(&self, handle: &RegisteredHandle) -> Result<(), DatabaseError> {
        Ok(self.consumers()?.heartbeat(handle)?)
    }

    /// Cancel the registration: it ends, and the history it held is no
    /// longer kept for it (other consumers may still keep it).
    ///
    /// # Errors
    ///
    /// [`DatabaseError::Consumer`] when the handle no longer addresses a live
    /// registration.
    pub fn cancel_consumer(&self, handle: RegisteredHandle) -> Result<(), DatabaseError> {
        Ok(self.consumers()?.unregister(handle)?)
    }

    /// The registration's acknowledged position, once the store is checked
    /// to still hold the history from it. A consumer calls this before
    /// reading from its position, so missing history is refused, never
    /// skipped.
    ///
    /// # Errors
    ///
    /// [`DatabaseError::Consumer`] with `RetentionLost` when the history is
    /// gone, or when the handle no longer addresses a live registration.
    pub fn consumer_position(&self, handle: &RegisteredHandle) -> Result<u64, DatabaseError> {
        Ok(self.consumers()?.check_retention(handle)?)
    }

    /// Every registration, live or ended, with its policy, state and
    /// acknowledged position.
    ///
    /// # Errors
    ///
    /// [`DatabaseError::Unsupported`] on a database without a journal of its
    /// own.
    pub fn consumers_list(&self) -> Result<Vec<ConsumerSnapshot>, DatabaseError> {
        Ok(self.consumers()?.list_consumers())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests;
