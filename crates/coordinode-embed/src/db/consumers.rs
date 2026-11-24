//! Change-stream consumers of a database: their registrations, and the
//! history the store keeps for them, over its own journal or, for a member
//! of a replicated group, over its group's log.
//!
//! Each consumer registers with an explicit retention policy. A STRICT one
//! keeps what it needs until it is cancelled; an ordinary one records its
//! position only, and learns of history the store no longer holds as
//! `RetentionLost`. The store's journal (for log consumers) and its MVCC
//! history (for the others) are kept from the lowest position a live STRICT
//! registration needs. The registry is the one the server runs, with its
//! background service on a thread of its own, so a database without an async
//! runtime has it.

use std::sync::Arc;

use coordinode_core::txn::proposal::{ProposalIdGenerator, ProposalPipeline};
use coordinode_replicate::{
    BackgroundConfig, ConsumerRegistration, ConsumerSnapshot, LogRetentionSource, RegisteredHandle,
    RegistryBackground, RetentionSource, RetentionSummary, SeqnoConsumerRegistry,
    ShardConsumerRegistry, SystemClock,
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

/// The history a member of a replicated group serves its consumers from:
/// its group's log, and the background service's tuning.
pub(crate) struct GroupConsumers {
    /// The group's log and the MVCC history this member applied from it.
    pub(crate) source: Arc<dyn RetentionSource>,
    /// The background service's tuning.
    pub(crate) background: BackgroundConfig,
}

impl Consumers {
    /// The registry of `engine`'s consumers, writing its records through
    /// `pipeline`: over the store's own journal and MVCC history, or, for a
    /// member of a replicated group (one voter alone included), over its
    /// group's log as `group` gives it. The registrations recorded in an
    /// earlier run are taken up again by this one registry, and their
    /// history stays held until it publishes their floors. `None` for a
    /// store with neither (an in-memory store), which holds no history for
    /// consumers.
    ///
    /// # Errors
    ///
    /// The journal could not be inspected, a stored registration could not
    /// be recovered, the store holds registrations and has neither a journal
    /// of its own nor a group's log to serve them, or the service's thread
    /// could not be started.
    pub(crate) fn open(
        engine: &Arc<StorageEngine>,
        pipeline: &Arc<dyn ProposalPipeline>,
        group: Option<GroupConsumers>,
    ) -> Result<Option<Self>, DatabaseError> {
        let (source, background): (Arc<dyn RetentionSource>, BackgroundConfig) = match (
            LogRetentionSource::over_own_journal(Arc::clone(engine))?,
            group,
        ) {
            (Some(own), _) => (Arc::new(own), BackgroundConfig::default()),
            (None, Some(group)) => (group.source, group.background),
            (None, None) => {
                // The engine holds every version for these registrations
                // until their floors are published, which nothing here
                // would do.
                if engine.consumer_protections_pending() {
                    return Err(DatabaseError::Unsupported(
                        "the store holds consumer registrations, and this database keeps \
                             neither a journal of its own nor a group's log to serve them; \
                             cancel them on a standalone open first"
                            .into(),
                    ));
                }
                return Ok(None);
            }
        };
        let registry = ShardConsumerRegistry::new(
            Arc::clone(engine),
            Arc::clone(pipeline),
            Arc::new(ProposalIdGenerator::with_base(
                coordinode_core::txn::proposal::fresh_proposal_id_base(),
            )),
            Arc::new(SystemClock),
            source,
        )?;
        let service = registry.start_background(background)?;
        Ok(Some(Self {
            registry,
            service: Some(service),
        }))
    }
}

impl Database {
    /// The consumer registry of this database's deployment, shared with
    /// every handle and session of it: the one a server's change streams
    /// register through. `None` for a database with no history to serve
    /// consumers from (an in-memory store).
    #[must_use]
    pub fn consumer_registry(&self) -> Option<ShardConsumerRegistry> {
        self.consumers.as_ref().map(|c| c.registry.clone())
    }

    /// The registry, or why this database has none.
    fn consumers(&self) -> Result<&ShardConsumerRegistry, DatabaseError> {
        self.consumers.as_ref().map(|c| &c.registry).ok_or_else(|| {
            DatabaseError::Unsupported(
                "this database keeps no history to hold for consumers".into(),
            )
        })
    }

    /// Register a change-stream consumer with the retention policy it chose,
    /// returning the handle of its new registration.
    ///
    /// An ordinary registration records progress without retaining history.
    /// A strict registration holds its starting position in the journal for
    /// a log consumer, or in MVCC history for other consumers.
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

    /// Acknowledge that the consumer holds everything before `position`.
    /// For a strict registration this also releases its retention below that
    /// position. A lower position than already acknowledged changes nothing.
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

    /// What the registrations hold of the store's history in each space
    /// (journal and MVCC): the lowest position, the bytes from it counted
    /// once however many registrations share them, the registrations at that
    /// position, and whether a new registration is admitted now.
    ///
    /// # Errors
    ///
    /// [`DatabaseError::Unsupported`] on a database without a journal of its
    /// own; [`DatabaseError::Consumer`] when the records cannot be read.
    pub fn consumer_retention(&self) -> Result<Vec<RetentionSummary>, DatabaseError> {
        Ok(self.consumers()?.retention_summary()?)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests;
