//! A projection this database keeps from a change feed: the effects a host
//! derives from committed changes and the progress through them, published
//! in one transaction.
//!
//! A projection has two positions. *Received* is how far its changes have
//! been taken in durably: applied, or staged here while a transaction larger
//! than one read arrives in fragments. *Applied* is how far every change has
//! taken effect: everything before it is in the projection's state, nothing
//! after it is. Readers of the projection see whole transactions only: a
//! staged transaction takes effect, with the progress past it, in the
//! transaction that receives its last fragment.

use std::collections::HashMap;

use coordinode_core::graph::types::Value;
use coordinode_core::txn::command::TransactionId;
use coordinode_query::executor::row::Row;
use coordinode_storage::engine::change_feed::{ChangeBatch, ChangeCursor, UnitChanges};
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::transaction::Transaction;

use super::{Database, DatabaseError};

/// Prefix of a projection's progress record, followed by its name.
const PROGRESS: &[u8] = b"projection:progress:";
/// Prefix of a projection's staged fragments, followed by its name, a zero
/// byte and the fragment's number, big-endian.
const STAGE: &[u8] = b"projection:stage:";

/// How far a projection has received and applied its changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectionProgress {
    /// Where the projection's next read starts: past everything it applied
    /// and every fragment it staged.
    pub received: ChangeCursor,
    /// Past every change that has taken effect in the projection.
    pub applied: ChangeCursor,
}

impl ProjectionProgress {
    /// Whether fragments of a transaction are staged and not yet applied.
    #[must_use]
    pub fn has_staged(&self) -> bool {
        self.received != self.applied
    }
}

/// The record of a projection's progress, as stored.
#[derive(serde::Serialize, serde::Deserialize)]
struct ProgressRecord {
    received: Vec<u8>,
    applied: Vec<u8>,
    /// Fragments staged of the transaction being received.
    staged: u32,
}

impl ProgressRecord {
    fn progress(&self) -> Result<ProjectionProgress, DatabaseError> {
        let cursor = |bytes: &[u8]| {
            ChangeCursor::from_bytes(bytes).map_err(|e| {
                DatabaseError::Other(format!("a stored projection cursor cannot be read: {e}"))
            })
        };
        Ok(ProjectionProgress {
            received: cursor(&self.received)?,
            applied: cursor(&self.applied)?,
        })
    }
}

/// The transaction a batch of changes is applied in: the host's writes run
/// here, and commit with the projection's progress or not at all.
pub struct ProjectionTransaction<'d> {
    db: &'d Database,
    id: TransactionId,
}

impl ProjectionTransaction<'_> {
    /// Run `query` in the transaction, as
    /// [`Database::execute_in_transaction`] does.
    ///
    /// # Errors
    ///
    /// As [`Database::execute_in_transaction`]; a failed statement ends the
    /// transaction, and nothing of the batch is applied.
    pub fn execute(
        &self,
        query: &str,
        params: Option<HashMap<String, Value>>,
    ) -> Result<Vec<Row>, DatabaseError> {
        self.db.execute_in_transaction(self.id, query, params)
    }

    /// The interactive transaction's ID.
    #[must_use]
    pub fn id(&self) -> TransactionId {
        self.id
    }
}

fn progress_key(name: &str) -> Vec<u8> {
    let mut key = PROGRESS.to_vec();
    key.extend_from_slice(name.as_bytes());
    key
}

fn stage_key(name: &str, fragment: u32) -> Vec<u8> {
    let mut key = STAGE.to_vec();
    key.extend_from_slice(name.as_bytes());
    key.push(0);
    key.extend_from_slice(&fragment.to_be_bytes());
    key
}

fn decode<T: serde::de::DeserializeOwned>(bytes: &[u8], what: &str) -> Result<T, DatabaseError> {
    rmp_serde::from_slice(bytes)
        .map_err(|e| DatabaseError::Other(format!("a stored {what} cannot be read: {e}")))
}

fn encode<T: serde::Serialize>(value: &T, what: &str) -> Result<Vec<u8>, DatabaseError> {
    rmp_serde::to_vec(value).map_err(|e| DatabaseError::Other(format!("encoding a {what}: {e}")))
}

fn read_progress(
    txn: &mut Transaction<'_>,
    name: &str,
) -> Result<Option<ProgressRecord>, DatabaseError> {
    txn.get(Partition::Schema, &progress_key(name))?
        .map(|bytes| decode(&bytes, "projection progress"))
        .transpose()
}

impl Database {
    /// Start keeping projection `name` from `from`, the cursor its first
    /// read starts at. A projection already kept under the name is left as
    /// it stands and its progress returned: starting again after a crash
    /// resumes it.
    ///
    /// # Errors
    ///
    /// [`DatabaseError::ProjectionOutOfStep`] for a cursor inside a
    /// transaction, whose beginning the projection would never receive; the
    /// commit's errors.
    pub fn start_projection(
        &self,
        name: &str,
        from: &ChangeCursor,
    ) -> Result<ProjectionProgress, DatabaseError> {
        if !from.at_unit_start() {
            return Err(DatabaseError::ProjectionOutOfStep {
                name: name.to_string(),
                reason: "a projection starts at a transaction's beginning, not inside one".into(),
            });
        }
        let id = self.begin_transaction()?;
        let started = self.with_open_transaction(id, |txn| {
            if let Some(record) = read_progress(txn, name)? {
                return Ok(record);
            }
            let record = ProgressRecord {
                received: from.to_bytes(),
                applied: from.to_bytes(),
                staged: 0,
            };
            txn.put(
                Partition::Schema,
                &progress_key(name),
                &encode(&record, "projection progress")?,
            )?;
            Ok(record)
        });
        self.finish(id, started)?.progress()
    }

    /// How far projection `name` has received and applied its changes, as
    /// last committed; `None` when no projection is kept under the name.
    ///
    /// # Errors
    ///
    /// A record that cannot be read.
    pub fn projection_progress(
        &self,
        name: &str,
    ) -> Result<Option<ProjectionProgress>, DatabaseError> {
        self.engine
            .get(Partition::Schema, &progress_key(name))?
            .map(|bytes| decode::<ProgressRecord>(&bytes, "projection progress")?.progress())
            .transpose()
    }

    /// Apply `batch`, read from `from`, to projection `name` in one
    /// transaction: `apply` runs for each whole transaction of the batch
    /// with the host's writes for it, and the projection's progress moves
    /// past the batch in the same commit. A transaction the batch delivers
    /// only the start or a middle of is staged here, with the progress
    /// received past it but not applied; the batch delivering its end runs
    /// `apply` for each staged fragment in order and then for the last one,
    /// all in that batch's transaction. A failed `apply` or commit applies
    /// nothing of the batch, and the batch is applied again from the same
    /// cursor.
    ///
    /// `apply` returning `Ok` claims the projection holds the effects it was
    /// handed. For a [`ScopeChange`](coordinode_storage::engine::change_feed::Action::ScopeChange)
    /// event, whose nodes enter or leave label scopes with no event of their
    /// own, that means the projection reconciled the type's nodes; one that
    /// cannot returns an error, the projection's progress staying before the
    /// change until it is rebuilt.
    ///
    /// # Errors
    ///
    /// [`DatabaseError::UnknownProjection`] for a name not started;
    /// [`DatabaseError::ProjectionOutOfStep`] when `from` is not the
    /// projection's received cursor; what `apply` returns; the commit's
    /// errors, among them a conflict with another applier of the same
    /// projection.
    pub fn apply_changes(
        &self,
        name: &str,
        from: &ChangeCursor,
        batch: &ChangeBatch,
        mut apply: impl FnMut(&ProjectionTransaction<'_>, &UnitChanges) -> Result<(), DatabaseError>,
    ) -> Result<ProjectionProgress, DatabaseError> {
        let id = self.begin_transaction()?;
        let applied = self.apply_in(id, name, from, batch, &mut apply);
        self.finish(id, applied)?.progress()
    }

    fn apply_in(
        &self,
        id: TransactionId,
        name: &str,
        from: &ChangeCursor,
        batch: &ChangeBatch,
        apply: &mut impl FnMut(&ProjectionTransaction<'_>, &UnitChanges) -> Result<(), DatabaseError>,
    ) -> Result<ProgressRecord, DatabaseError> {
        let out_of_step = |reason: &str| DatabaseError::ProjectionOutOfStep {
            name: name.to_string(),
            reason: reason.to_string(),
        };
        let mut record = self
            .with_open_transaction(id, |txn| read_progress(txn, name))?
            .ok_or_else(|| DatabaseError::UnknownProjection(name.to_string()))?;
        if record.received != from.to_bytes() {
            return Err(out_of_step(
                "the changes were read from another cursor than the one it received through",
            ));
        }
        let tx = ProjectionTransaction { db: self, id };
        let mut applied = ChangeCursor::from_bytes(&record.applied)
            .map_err(|e| DatabaseError::Other(format!("a stored projection cursor: {e}")))?;
        for unit in &batch.units {
            if record.staged == 0 && unit.first != 0 {
                return Err(out_of_step(
                    "a transaction continues whose beginning it did not receive",
                ));
            }
            if !unit.completes_unit() {
                // Received, not applied: kept until its last fragment.
                let fragment = record.staged;
                let value = encode(unit, "staged fragment")?;
                self.with_open_transaction(id, |txn| {
                    Ok(txn.put(Partition::Schema, &stage_key(name, fragment), &value)?)
                })?;
                record.staged = fragment
                    .checked_add(1)
                    .ok_or_else(|| out_of_step("a transaction staged in over 2^32 fragments"))?;
                applied = from.before_unit(unit.position);
                continue;
            }
            // The fragments staged before this last one, in order, one at a
            // time.
            for fragment in 0..record.staged {
                let key = stage_key(name, fragment);
                let staged: UnitChanges = self.with_open_transaction(id, |txn| {
                    let bytes = txn.get(Partition::Schema, &key)?.ok_or_else(|| {
                        DatabaseError::Other(format!(
                            "projection {name}: staged fragment {fragment} is missing"
                        ))
                    })?;
                    txn.delete(Partition::Schema, &key)?;
                    decode(&bytes, "staged fragment")
                })?;
                if staged.position != unit.position {
                    return Err(DatabaseError::Other(format!(
                        "projection {name}: staged fragment {fragment} is of another transaction"
                    )));
                }
                apply(&tx, &staged)?;
            }
            record.staged = 0;
            apply(&tx, unit)?;
        }
        record.received = batch.cursor.to_bytes();
        if record.staged == 0 {
            applied = batch.cursor.clone();
        }
        record.applied = applied.to_bytes();
        let value = encode(&record, "projection progress")?;
        self.with_open_transaction(id, |txn| {
            Ok(txn.put(Partition::Schema, &progress_key(name), &value)?)
        })?;
        Ok(record)
    }

    /// Commit `id` when `staged` succeeded, roll it back when it did not.
    fn finish<T>(
        &self,
        id: TransactionId,
        staged: Result<T, DatabaseError>,
    ) -> Result<T, DatabaseError> {
        match staged {
            Ok(value) => {
                self.commit_transaction(id)?;
                Ok(value)
            }
            Err(e) => {
                match self.rollback_transaction(id) {
                    // A failed statement ended it already: nothing is left
                    // to undo.
                    Ok(())
                    | Err(
                        DatabaseError::UnknownTransaction(_)
                        | DatabaseError::TransactionAborted { .. },
                    ) => {}
                    Err(rollback) => {
                        tracing::warn!(error = %rollback, "a projection transaction did not roll back");
                    }
                }
                Err(e)
            }
        }
    }
}
