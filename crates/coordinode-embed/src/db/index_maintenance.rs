//! Checking B-tree indexes against their records, and rebuilding them.
//!
//! An index is derived from the records it covers. A check reads both and
//! repairs the entries that disagree; damage too wide to repair entry by
//! entry rebuilds the index into a fresh generation. Reads and writes that
//! meet a disagreement start a check by themselves, and every index is
//! checked periodically; these calls start one on request and inspect them.

use core::time::Duration;

use coordinode_query::executor::runner::CatalogObject;
use coordinode_query::index::{
    CheckOutcome, CheckRequestError, CheckStatus, GenerationId, IndexCopySource, IndexSelector,
    IndexSelectorError, IndexType,
};

use super::{Database, DatabaseError};

impl Database {
    /// Start a check of the index `index` against its records, in the
    /// background. Returns the operation identifying it (the generation
    /// checked), which [`Self::index_check`] inspects and
    /// [`Self::cancel_index_check`] cancels. A check already running for that
    /// generation is returned as it is.
    ///
    /// # Errors
    ///
    /// No index matches; it is not a ready B-tree index; this member takes
    /// no writes (the leader runs checks); or the catalog could not be
    /// written.
    pub fn check_index(&self, index: &IndexSelector) -> Result<GenerationId, DatabaseError> {
        let def = index
            .resolve(&self.engine)
            .map_err(|e| selector_error(index, e))?;
        self.index_builds
            .request_check(def.id)
            .map_err(check_request_error)
    }

    /// The check `operation`, after waiting up to `wait` for its outcome;
    /// `None` when no check has that operation. Waiting cancels nothing.
    ///
    /// # Errors
    ///
    /// The record could not be read.
    pub fn index_check(
        &self,
        operation: GenerationId,
        wait: Duration,
    ) -> Result<Option<(CheckStatus, Option<CheckOutcome>)>, DatabaseError> {
        let outcome = self.index_builds.wait_check(operation, Some(wait))?;
        let status = self
            .index_builds
            .checks()?
            .into_iter()
            .find(|status| status.generation == operation);
        Ok(status.map(|status| (status, outcome)))
    }

    /// Every check the catalog records: per generation, what is known about
    /// its entries, the disagreements reported, and its latest check with
    /// how far it got and what it found and repaired.
    ///
    /// # Errors
    ///
    /// The records could not be read.
    pub fn index_checks(&self) -> Result<Vec<CheckStatus>, DatabaseError> {
        Ok(self.index_builds.checks()?)
    }

    /// Fill replacements of this member's index copies from `source`, its
    /// peers' copies: a copy found disagreeing with this member's records is
    /// replaced, guarded, checked and then serves again. A member without a
    /// source keeps such a copy out of use and answers from its records.
    pub fn set_index_copy_source(&self, source: std::sync::Arc<dyn IndexCopySource>) {
        self.index_builds.set_copy_source(source);
    }

    /// Flushes of this member's index recovery state made since it opened:
    /// use guards stored before a copy's first use, and marks of copies
    /// found disagreeing with their records. A lookup through a copy already
    /// in use makes none.
    pub fn index_recovery_flushes(&self) -> u64 {
        self.index_registry.recovery_flushes()
    }

    /// Cancel the check `operation`. Returns whether a check was stopped;
    /// `false` when it already had an outcome or none has the operation.
    /// Repairs it committed stay.
    ///
    /// # Errors
    ///
    /// The cancellation could not be committed.
    pub fn cancel_index_check(&self, operation: GenerationId) -> Result<bool, DatabaseError> {
        self.index_builds
            .cancel_check(operation)
            .map_err(DatabaseError::Other)
    }

    /// Rebuild the B-tree index `index` from its records into a fresh
    /// generation, in the background, and wait up to `wait` for the build.
    /// Writers maintain the new generation at once; lookups answer from the
    /// records until it is built, and unique values are proved free from
    /// them. A rebuild the records refuse (two hold one unique value) keeps
    /// the index, marked failed, with its constraint still enforced. Returns
    /// the build's operation, which [`Self::index_build`] inspects.
    ///
    /// # Errors
    ///
    /// No index matches; it is not a B-tree index; or the catalog could not
    /// be written.
    pub fn reindex(
        &self,
        index: &IndexSelector,
        wait: Duration,
    ) -> Result<GenerationId, DatabaseError> {
        let def = index
            .resolve(&self.engine)
            .map_err(|e| selector_error(index, e))?;
        if def.index_type != IndexType::BTree {
            return Err(DatabaseError::CatalogRefused(format!(
                "index '{def}' is not a B-tree index; only B-tree indexes are rebuilt this way"
            )));
        }
        let def = self
            .index_builds
            .rebuild(def)
            .map_err(DatabaseError::Other)?;
        self.index_builds.wait(def.generation, Some(wait))?;
        Ok(def.generation)
    }
}

/// A refused check as the error a caller can tell apart: a missing index,
/// an index or a member that cannot run one, or a storage failure.
fn check_request_error(e: CheckRequestError) -> DatabaseError {
    match e {
        CheckRequestError::NoSuchIndex => DatabaseError::CatalogObjectMissing {
            object: CatalogObject::Index,
            name: String::new(),
        },
        CheckRequestError::NotChecked(_) | CheckRequestError::NotReady(_) => {
            DatabaseError::CatalogRefused(e.to_string())
        }
        CheckRequestError::NotHere => DatabaseError::NotLeader { leader_id: None },
        CheckRequestError::Other(reason) => DatabaseError::Unsupported(reason),
    }
}

/// An index a selector names that does not resolve.
fn selector_error(index: &IndexSelector, e: IndexSelectorError) -> DatabaseError {
    match e {
        IndexSelectorError::NotFound(_) => DatabaseError::CatalogObjectMissing {
            object: CatalogObject::Index,
            name: match index {
                IndexSelector::Name(name) => name.clone(),
                IndexSelector::Id(id) => id.as_raw().to_string(),
            },
        },
        IndexSelectorError::Storage(reason) => DatabaseError::Other(reason),
    }
}
