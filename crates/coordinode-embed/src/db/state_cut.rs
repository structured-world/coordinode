//! State recovery for a change-stream consumer that lost its journal
//! window: a fixed cut of one history, over one row scope.
//!
//! [`Database::state_cut`] pins a read view of a scope below a bound and
//! starts the continuation that joins it. The consumer copies the view
//! ([`Database::state_cut_scan`]) into a replacement of its destination,
//! then reads the continuation ([`Database::state_cut_tail`]) and applies
//! the units it hands over, with the existing committed-effect semantics:
//! units the view already holds are left out, so a non-idempotent effect is
//! applied once. The replacement, the absence of every key the view lacks
//! and the consumer's progress are published together, in one destination
//! transaction, and only once the join is complete ([`StateCut::joined`]).
//!
//! A cut recovers the rows of its scope only: not the whole database, a
//! columnar table or an index. An effect inside the scope it cannot carry
//! is refused, and the continuation does not move past it. Every read is
//! bounded by a caller's budget, and what it hands over stays charged to
//! that budget while it is held.

use std::time::Duration;

use coordinode_core::budget::{MemoryCharge, QueryBudget};
use coordinode_storage::engine::fixed_cut::{CutRead, CutUnit, FixedCut};

use super::{Database, DatabaseError};

pub use coordinode_storage::engine::fixed_cut::{CutHistory, CutResume, CutScope};

/// A key of the view and its value.
pub type CutRow = (Vec<u8>, Vec<u8>);

/// A pinned read view of a scope below [`Self::bound`] and the position its
/// continuation has reached. Dropping it releases the view; a view lost
/// with the database is refused, not served again.
#[derive(Debug)]
pub struct StateCut {
    cut: FixedCut,
    next: u64,
}

impl StateCut {
    /// The view holds every commit stamped below this bound and none at or
    /// above it.
    #[must_use]
    pub fn bound(&self) -> u64 {
        self.cut.bound()
    }

    /// The first log position of the continuation.
    #[must_use]
    pub fn first(&self) -> u64 {
        self.cut.first()
    }

    /// The end of the join: the continuation is complete once read to here.
    #[must_use]
    pub fn join_end(&self) -> u64 {
        self.cut.join_end()
    }

    /// The position the next read of the continuation starts at.
    #[must_use]
    pub fn next(&self) -> u64 {
        self.next
    }

    /// The rows the cut hands over.
    #[must_use]
    pub fn scope(&self) -> &CutScope {
        self.cut.scope()
    }

    /// The history the cut is of.
    #[must_use]
    pub fn history(&self) -> CutHistory {
        self.cut.history()
    }

    /// Where reading this database's log resumes, once the join is complete;
    /// `None` before. Publishing the replacement earlier would certify a
    /// prefix the view already passed. The position names its history and
    /// scope: [`Database::resume_position`] gives it back only in the same
    /// history, for the same scope.
    #[must_use]
    pub fn joined(&self) -> Option<CutResume> {
        self.cut.resume_at(self.next)
    }
}

/// One page of a cut's view. Its rows stay charged to the budget the page
/// was read within until it is dropped.
#[derive(Debug)]
pub struct CutPage<'b> {
    rows: Vec<CutRow>,
    _charge: MemoryCharge<'b>,
}

impl CutPage<'_> {
    /// The rows, in key order.
    #[must_use]
    pub fn rows(&self) -> &[CutRow] {
        &self.rows
    }

    /// The last key handed over: where the next page goes on from.
    #[must_use]
    pub fn last_key(&self) -> Option<&[u8]> {
        self.rows.last().map(|(key, _)| key.as_slice())
    }
}

/// One read of a cut's continuation. Its units stay charged to the budget
/// the read was made within until it is dropped.
#[derive(Debug)]
pub struct CutBatch<'b> {
    units: Vec<CutUnit>,
    joined: Option<CutResume>,
    _charge: MemoryCharge<'b>,
}

impl CutBatch<'_> {
    /// The units to apply, in log order: those the view does not hold, that
    /// took effect and change something in the scope, each with only its
    /// changes in the scope.
    #[must_use]
    pub fn units(&self) -> &[CutUnit] {
        &self.units
    }

    /// See [`StateCut::joined`], after this read.
    #[must_use]
    pub fn joined(&self) -> Option<&CutResume> {
        self.joined.as_ref()
    }
}

impl Database {
    /// Take a fixed cut of this database's history over `scope`: a view
    /// pinned below a fresh bound, and the continuation that joins it.
    /// Gives the database its deployment identity first when it has none,
    /// so the cut names its history.
    ///
    /// # Errors
    ///
    /// [`DatabaseError::Storage`] carrying `CutNotReady` when a commit below
    /// the cut is still in flight or not durable after `timeout`, or the
    /// history of a member's log is not known yet; a database with neither a
    /// journal of its own nor its group's log.
    pub fn state_cut(&self, scope: CutScope, timeout: Duration) -> Result<StateCut, DatabaseError> {
        self.found_deployment()?;
        let cut = self.engine.fixed_cut(scope, timeout)?;
        Ok(StateCut {
            next: cut.first(),
            cut,
        })
    }

    /// One page of the cut's view: the scope's rows in key order, after
    /// `after` when given, at most `max_rows` of them, within `budget`. A
    /// page reads from `after` on, so a late page costs what an early one
    /// does.
    ///
    /// # Errors
    ///
    /// `CutLost` for a cut of a database closed since, or whose scope was
    /// repaired: its view is gone and a new cut is taken. `Budget` for a
    /// first row that does not fit the budget on its own. A read that fails.
    pub fn state_cut_scan<'b>(
        &self,
        cut: &StateCut,
        after: Option<&[u8]>,
        max_rows: usize,
        budget: &'b QueryBudget,
    ) -> Result<CutPage<'b>, DatabaseError> {
        let (rows, charge) = self.engine.cut_page(&cut.cut, after, max_rows, budget)?;
        Ok(CutPage {
            rows,
            _charge: charge,
        })
    }

    /// Read up to `max_entries` log entries of the cut's continuation, within
    /// `budget`, from where the last read ended. Only entries this database
    /// applied and made durable are read: one still undecided is not handed
    /// over until it is.
    ///
    /// # Errors
    ///
    /// [`DatabaseError::Storage`] carrying `RetentionLost` when the log no
    /// longer holds the next position: the copy has to start again from a
    /// new cut; `CutLost` for a cut of a database closed since;
    /// `CutUnsupported` for an effect inside the scope the cut cannot carry;
    /// `Budget` for a first entry that does not fit the budget on its own; a
    /// confirmed entry that cannot be read; a cut of another history. On
    /// each of them the position stays where it was.
    pub fn state_cut_tail<'b>(
        &self,
        cut: &mut StateCut,
        max_entries: usize,
        budget: &'b QueryBudget,
    ) -> Result<CutBatch<'b>, DatabaseError> {
        let (read, charge) = if self.engine.has_journal() {
            self.engine
                .cut_read(&cut.cut, cut.next, max_entries, budget)?
        } else {
            self.group_cut_read(&cut.cut, cut.next, max_entries, budget)?
        };
        cut.next = read.next;
        Ok(CutBatch {
            units: read
                .units
                .into_iter()
                .filter(|unit| !unit.in_view && !unit.refused && !unit.mutations.is_empty())
                .collect(),
            joined: cut.joined(),
            _charge: charge,
        })
    }

    /// Where this database's log is read on from for `scope`, after a cut's
    /// join reached `resume`.
    ///
    /// # Errors
    ///
    /// A position of another history, or reached for another scope: it
    /// counts entries of another log, or the consumer's state covers other
    /// rows.
    pub fn resume_position(
        &self,
        resume: &CutResume,
        scope: &CutScope,
    ) -> Result<u64, DatabaseError> {
        Ok(self.engine.resume_position(resume, scope)?)
    }

    /// [`Self::state_cut_tail`] over the log of the group this database
    /// applies, up to the last entry it applied: an entry past it may never
    /// commit.
    #[cfg(feature = "replication")]
    fn group_cut_read<'b>(
        &self,
        cut: &FixedCut,
        from: u64,
        max_entries: usize,
        budget: &'b QueryBudget,
    ) -> Result<(CutRead, MemoryCharge<'b>), DatabaseError> {
        // The log of the group this database is a member of: its shard.
        const GROUP_LOG_SHARD: u32 = 0;
        let dirs = coordinode_raft::storage::raft_oplog_dirs(&self.engine, GROUP_LOG_SHARD)
            .map_err(|e| DatabaseError::Other(format!("the group's log: {e}")))?
            .all;
        let until = self
            .engine
            .group_applied_through()
            .checked_add(1)
            .ok_or_else(|| DatabaseError::Other("the applied log index overflowed".into()))?;
        Ok(self
            .engine
            .cut_read_log(cut, &dirs, from, until, max_entries, budget)?)
    }

    /// Without replication a database applies no group's log.
    #[cfg(not(feature = "replication"))]
    fn group_cut_read<'b>(
        &self,
        _cut: &FixedCut,
        _from: u64,
        _max_entries: usize,
        _budget: &'b QueryBudget,
    ) -> Result<(CutRead, MemoryCharge<'b>), DatabaseError> {
        Err(DatabaseError::Unsupported(
            "reading a group's log needs the replication feature".into(),
        ))
    }
}
