//! Committed changes as logical events: what a reader of the database's log
//! receives, unit by unit, from a cursor it keeps.
//!
//! Each event is a put, patch or removal of a node, a relationship, its
//! properties or one of its instances, with its identity in the log and how
//! it reaches the reader's interest: a node a reader follows by label enters,
//! stays in or leaves that scope, a removal or relabelling included, routed
//! by the labels the node had without reading it again. A cursor is opaque
//! bytes the reader saves; it holds nothing of the log, and one that falls
//! behind what the log keeps is told so rather than skipped forward.

use coordinode_core::budget::{MemoryCharge, QueryBudget};
use coordinode_modality::change_interest::{InterestRefusal, resolve_change_interest};
use coordinode_storage::engine::change_feed::{ChangeBatch, ChangeCursor, Interest};

use super::{Database, DatabaseError};

/// What a reader of committed changes asks for, by name: nodes followed by
/// label, relationships by type. `None` takes every event of that kind; an
/// empty list takes none.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChangeInterest {
    /// Labels whose nodes the reader follows: a node event reaches it when
    /// the node carries one of them before or after the event.
    pub labels: Option<Vec<String>>,
    /// Relationship types whose relationships the reader follows.
    pub edge_types: Option<Vec<String>>,
}

/// A read of committed changes: whole units with the events the reader's
/// interest takes, and the cursor to continue from. The units' memory stays
/// charged to the read's budget while the batch is held.
pub struct ChangePage<'b> {
    /// The units read and the cursor to continue from.
    pub batch: ChangeBatch,
    _charge: MemoryCharge<'b>,
}

impl Database {
    /// The interest `interest` names, bound to the labels its names are
    /// bound to now: a label dropped and created again later is another
    /// label, whose nodes this interest does not follow.
    fn change_interest(&self, interest: &ChangeInterest) -> Result<Interest, DatabaseError> {
        let resolved = resolve_change_interest(
            &self.engine,
            interest.labels.as_deref(),
            interest.edge_types.as_deref(),
        )
        .map_err(coordinode_query::executor::ExecutionError::from)?;
        resolved.map_err(|refusal| match refusal {
            InterestRefusal::Unbound(_) => DatabaseError::Semantic(refusal.to_string()),
            InterestRefusal::Implied(_) => DatabaseError::Unsupported(refusal.to_string()),
        })
    }

    /// A cursor after every change committed and durable now: a reader
    /// starting here receives what commits from now on.
    ///
    /// # Errors
    ///
    /// [`DatabaseError::Unsupported`] for a label a type implies;
    /// [`DatabaseError::Semantic`] for a label not bound; a catalog or log
    /// that cannot be read.
    pub fn change_cursor(&self, interest: &ChangeInterest) -> Result<ChangeCursor, DatabaseError> {
        let interest = self.change_interest(interest)?;
        if self.engine.has_journal() {
            return Ok(self.engine.change_cursor_at_end(interest)?);
        }
        let end = self.group_log_end()?;
        Ok(self.engine.change_cursor_at(end, interest)?)
    }

    /// A cursor at log `position`: where a reader that holds everything
    /// before it continues, as a registered consumer's acknowledged position
    /// says.
    ///
    /// # Errors
    ///
    /// As [`Self::change_cursor`].
    pub fn change_cursor_at(
        &self,
        position: u64,
        interest: &ChangeInterest,
    ) -> Result<ChangeCursor, DatabaseError> {
        let interest = self.change_interest(interest)?;
        Ok(self.engine.change_cursor_at(position, interest)?)
    }

    /// Read the changes after `cursor`: the units of up to `max_entries` log
    /// entries, within `budget`, only those this database applied and made
    /// durable, each whole with the events of it the cursor's interest takes.
    /// A unit with none of them is there too, with none: the reader sees
    /// every unit it passed.
    ///
    /// # Errors
    ///
    /// [`DatabaseError::Storage`] carrying `RetentionLost` when the log no
    /// longer holds the cursor's position, `ChangeCursor` for a cursor of
    /// another history or interpretation, `ChangeFeedUnsupported` for a unit
    /// the feed cannot read, `Budget` for a first entry that does not fit
    /// the budget on its own. On each of them the cursor stays valid.
    pub fn read_changes<'b>(
        &self,
        cursor: &ChangeCursor,
        max_entries: usize,
        budget: &'b QueryBudget,
    ) -> Result<ChangePage<'b>, DatabaseError> {
        let (batch, charge) = if self.engine.has_journal() {
            self.engine.read_changes(cursor, max_entries, budget)?
        } else {
            self.group_read_changes(cursor, max_entries, budget)?
        };
        Ok(ChangePage {
            batch,
            _charge: charge,
        })
    }

    /// The end of what this database applied of its group's log.
    #[cfg(feature = "replication")]
    fn group_log_end(&self) -> Result<u64, DatabaseError> {
        self.engine
            .group_applied_through()
            .checked_add(1)
            .ok_or_else(|| DatabaseError::Other("the applied log index overflowed".into()))
    }

    /// Without replication a database applies no group's log.
    #[cfg(not(feature = "replication"))]
    fn group_log_end(&self) -> Result<u64, DatabaseError> {
        Err(DatabaseError::Unsupported(
            "reading a group's log needs the replication feature".into(),
        ))
    }

    /// [`Self::read_changes`] over the log of the group this database
    /// applies, up to the last entry it applied: an entry past it may never
    /// commit.
    #[cfg(feature = "replication")]
    fn group_read_changes<'b>(
        &self,
        cursor: &ChangeCursor,
        max_entries: usize,
        budget: &'b QueryBudget,
    ) -> Result<(ChangeBatch, MemoryCharge<'b>), DatabaseError> {
        // The log of the group this database is a member of: its shard.
        const GROUP_LOG_SHARD: u32 = 0;
        let dirs = coordinode_raft::storage::raft_oplog_dirs(&self.engine, GROUP_LOG_SHARD)
            .map_err(|e| DatabaseError::Other(format!("the group's log: {e}")))?
            .all;
        let until = self.group_log_end()?;
        Ok(self
            .engine
            .read_changes_log(cursor, &dirs, until, max_entries, budget)?)
    }

    /// Without replication a database applies no group's log.
    #[cfg(not(feature = "replication"))]
    fn group_read_changes<'b>(
        &self,
        _cursor: &ChangeCursor,
        _max_entries: usize,
        _budget: &'b QueryBudget,
    ) -> Result<(ChangeBatch, MemoryCharge<'b>), DatabaseError> {
        Err(DatabaseError::Unsupported(
            "reading a group's log needs the replication feature".into(),
        ))
    }
}
