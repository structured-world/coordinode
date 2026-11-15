//! State recovery for a change-stream consumer that lost its journal
//! window: a fixed cut of one history.
//!
//! [`Database::state_cut`] pins a read view below a bound and starts the
//! continuation that joins it. The consumer copies the selected scope of the
//! view ([`Database::state_cut_scan`]) into a replacement of its destination,
//! then reads the continuation ([`Database::state_cut_tail`]) and applies
//! the units it hands over, with the existing committed-effect semantics:
//! units the view already holds are left out, so a non-idempotent effect is
//! applied once. The replacement, the absence of every key the view lacks
//! and the consumer's progress are published together, in one destination
//! transaction, and only once the join is complete ([`StateCut::joined`]).
//! From there the consumer reads the journal as any reader does.

use std::time::Duration;

use coordinode_storage::engine::fixed_cut::{CutRead, CutUnit, FixedCut};
use coordinode_storage::engine::partition::Partition;

use super::{Database, DatabaseError};

/// A key of the view and its value.
pub type CutRow = (Vec<u8>, Vec<u8>);

/// The log of a group this database is a member of: its shard.
#[cfg(feature = "replication")]
const GROUP_LOG_SHARD: u32 = 0;

/// A pinned read view below [`Self::bound`] and the position its
/// continuation has reached. Dropping it releases the view; a view lost
/// with the database cannot be taken again from its positions alone.
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

    /// The first journal position of the continuation.
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

    /// Where ordinary change reading resumes, once the join is complete;
    /// `None` before. Publishing the replacement earlier would certify a
    /// prefix the view already passed.
    #[must_use]
    pub fn joined(&self) -> Option<u64> {
        (self.next >= self.cut.join_end()).then_some(self.next)
    }
}

/// One read of a cut's continuation.
#[derive(Debug, Clone, PartialEq)]
pub struct CutBatch {
    /// The units to apply, in log order: those the view does not hold and
    /// that took effect.
    pub units: Vec<CutUnit>,
    /// See [`StateCut::joined`], after this read.
    pub joined: Option<u64>,
}

impl Database {
    /// Take a fixed cut of this database's history: a view pinned below a
    /// fresh bound, and the continuation that joins it. Gives the database
    /// its deployment identity first when it has none, so the cut names its
    /// history.
    ///
    /// # Errors
    ///
    /// [`DatabaseError::Storage`] carrying `CutNotReady` when a commit below
    /// the cut is still in flight or not durable after `timeout`; a database
    /// with neither a journal of its own nor its group's log.
    pub fn state_cut(&self, timeout: Duration) -> Result<StateCut, DatabaseError> {
        self.found_deployment()?;
        let cut = self.engine.fixed_cut(timeout)?;
        Ok(StateCut {
            next: cut.first(),
            cut,
        })
    }

    /// Up to `limit` keys of `partition` under `prefix` in the cut's view,
    /// in key order, after `after` when given: one page of the copy.
    ///
    /// # Errors
    ///
    /// `CutLost` for a cut of a database closed since: its view is gone and
    /// a new cut is taken. A read that fails.
    pub fn state_cut_scan(
        &self,
        cut: &StateCut,
        partition: Partition,
        prefix: &[u8],
        after: Option<&[u8]>,
        limit: usize,
    ) -> Result<Vec<CutRow>, DatabaseError> {
        self.engine.held_cut(&cut.cut)?;
        let mut page = Vec::new();
        for guard in self
            .engine
            .snapshot_prefix_iter(&cut.bound(), partition, prefix)?
        {
            if page.len() >= limit {
                break;
            }
            let (key, value) = guard
                .into_owned()
                .map_err(coordinode_storage::error::StorageError::from)?;
            if after.is_some_and(|after| key.as_slice() <= after) {
                continue;
            }
            page.push((key, value.to_vec()));
        }
        Ok(page)
    }

    /// Read up to `entries` entries of the cut's continuation, from where
    /// the last read ended.
    ///
    /// # Errors
    ///
    /// [`DatabaseError::Storage`] carrying `RetentionLost` when the journal
    /// no longer holds the next position: the copy has to start again from a
    /// new cut; `CutLost` for a cut of a database closed since. A unit that
    /// cannot be read, or a cut of another history.
    pub fn state_cut_tail(
        &self,
        cut: &mut StateCut,
        entries: u64,
    ) -> Result<CutBatch, DatabaseError> {
        // Before reading: a lost view would otherwise surface as a lost
        // position or not at all.
        self.engine.held_cut(&cut.cut)?;
        let read = if self.engine.has_journal() {
            self.engine.fixed_cut_units(&cut.cut, cut.next, entries)?
        } else {
            self.group_cut_read(&cut.cut, cut.next, entries)?
        };
        cut.next = read.next;
        Ok(CutBatch {
            units: read
                .units
                .into_iter()
                .filter(|unit| !unit.in_view && !unit.refused)
                .collect(),
            joined: cut.joined(),
        })
    }

    /// [`Self::state_cut_tail`] over the log of the group this database
    /// applies, read in place with each unit frame as recorded.
    #[cfg(feature = "replication")]
    fn group_cut_read(
        &self,
        cut: &FixedCut,
        from: u64,
        entries: u64,
    ) -> Result<CutRead, DatabaseError> {
        use coordinode_storage::oplog::tailer::{CdcFilters, OplogTailer, ResumeToken};
        let dirs = coordinode_raft::storage::raft_oplog_dirs(&self.engine, GROUP_LOG_SHARD)
            .map_err(|e| DatabaseError::Other(format!("the group's log: {e}")))?
            .all;
        let mut tailer = OplogTailer::new(
            &dirs,
            ResumeToken {
                shard_id: 0,
                segment_id: from,
                entry_offset: 0,
            },
        )?
        .with_unit_frames();
        // A count past the address space reads as much as there is.
        let max = usize::try_from(entries).unwrap_or(usize::MAX);
        // Only what this member applied: an entry past it may never commit.
        let until = self
            .engine
            .group_applied_through()
            .checked_add(1)
            .ok_or_else(|| DatabaseError::Other("the applied log index overflowed".into()))?;
        let read: Vec<_> = tailer
            .read_next(max, &CdcFilters::default(), until)?
            .into_iter()
            .map(|(entry, _)| entry)
            .collect();
        Ok(CutRead {
            units: self.engine.cut_units_of(cut, &read)?,
            next: tailer.next_index(),
        })
    }

    /// Without replication a database applies no group's log.
    #[cfg(not(feature = "replication"))]
    fn group_cut_read(
        &self,
        _cut: &FixedCut,
        _from: u64,
        _entries: u64,
    ) -> Result<CutRead, DatabaseError> {
        Err(DatabaseError::Unsupported(
            "reading a group's log needs the replication feature".into(),
        ))
    }
}
