//! How far the Raft log applied here is closed, and the commit ceiling of
//! each of its prefixes.
//!
//! A read bounded by a log position must hold every commit of the log up to
//! that position: the prefix's commit ceiling, the highest commit timestamp
//! in it. The ledger keeps that ceiling for every prefix the closed bound
//! has not passed yet, as the positions where it rose, and for the prefix it
//! passed last. It is a function of the log alone, so every member applying
//! the log keeps the same ledger, and its record replicates and survives a
//! restart with the entries it describes.
//!
//! A position the closed bound passed gets the ceiling of the last closed
//! prefix: no lower than its own, and below the bound, so a read needs no
//! commit the log has not closed. A position still open gets the ceiling
//! where it last rose at or before the position: its own exactly. When more
//! open positions pile up than [`OPEN_CHECKPOINTS`], the two oldest merge
//! into the earlier position with the later ceiling, which is no lower than
//! any position between them needs.

// no-std: alloc::collections::VecDeque
use std::collections::VecDeque;

use crate::error::{StorageError, StorageResult};

/// The open positions a ledger keeps one by one. A closed bound that falls
/// this many ceiling rises behind lets the oldest merge, so the record an
/// entry writes stays a few kilobytes at most.
pub(crate) const OPEN_CHECKPOINTS: usize = 256;

/// The stored form's leading byte.
const FORMAT: u8 = 1;

/// A log position where the commit ceiling rose, and the ceiling there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Checkpoint {
    index: u64,
    ceiling: u64,
}

/// The closed bound of the log applied here and the commit ceilings of its
/// prefixes; see the module documentation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ClosureLedger {
    /// The closed bound: every commit below it is in an applied entry.
    below: u64,
    /// The last entry the ledger accounts for; an entry at or before it is
    /// already in, as when a restarted member applies its log again.
    last_index: u64,
    /// The last position whose prefix the bound passed, and that prefix's
    /// ceiling.
    closed_through: u64,
    closed_ceiling: u64,
    /// The positions after `closed_through` where the ceiling rose, in log
    /// order, ceilings rising with them.
    open: VecDeque<Checkpoint>,
}

impl ClosureLedger {
    /// The closed bound.
    pub(crate) fn below(&self) -> u64 {
        self.below
    }

    /// The commit ceiling of the whole log applied here.
    pub(crate) fn applied_ceiling(&self) -> u64 {
        self.open.back().map_or(self.closed_ceiling, |c| c.ceiling)
    }

    /// Account for a commit at `commit_ts` in entry `index`. Whether the
    /// ledger changed.
    pub(crate) fn commit(&mut self, index: u64, commit_ts: u64) -> bool {
        if index < self.last_index {
            return false;
        }
        self.last_index = index;
        if commit_ts <= self.applied_ceiling() {
            return false;
        }
        match self.open.back_mut() {
            Some(last) if last.index == index => last.ceiling = commit_ts,
            _ => {
                self.open.push_back(Checkpoint {
                    index,
                    ceiling: commit_ts,
                });
                if self.open.len() > OPEN_CHECKPOINTS
                    && let Some(oldest) = self.open.pop_front()
                    && let Some(next) = self.open.front_mut()
                {
                    next.index = oldest.index;
                }
            }
        }
        true
    }

    /// Account for the closed bound `below` that entry `index` carries.
    /// Whether the ledger changed.
    pub(crate) fn close(&mut self, below: u64, index: u64) -> bool {
        if index < self.last_index || below <= self.below {
            return false;
        }
        self.last_index = index;
        self.below = below;
        while let Some(&first) = self.open.front()
            && first.ceiling < below
        {
            self.closed_through = first.index;
            self.closed_ceiling = first.ceiling;
            self.open.pop_front();
        }
        if self.open.is_empty() {
            // Every entry accounted for is closed, those without commits too.
            self.closed_through = self.last_index;
        }
        true
    }

    /// The commit timestamp a read must reach to hold every commit of the
    /// log through `index`, a position this member applied.
    pub(crate) fn ceiling_through(&self, index: u64) -> u64 {
        if index <= self.closed_through {
            return self.closed_ceiling;
        }
        self.open
            .iter()
            .rev()
            .find(|c| c.index <= index)
            .map_or(self.closed_ceiling, |c| c.ceiling)
    }

    /// The stored form.
    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(37 + 16 * self.open.len());
        out.push(FORMAT);
        for field in [
            self.below,
            self.last_index,
            self.closed_through,
            self.closed_ceiling,
        ] {
            out.extend_from_slice(&field.to_be_bytes());
        }
        // At most OPEN_CHECKPOINTS, far below u32::MAX.
        let count = u32::try_from(self.open.len()).unwrap_or(u32::MAX);
        out.extend_from_slice(&count.to_be_bytes());
        for c in &self.open {
            out.extend_from_slice(&c.index.to_be_bytes());
            out.extend_from_slice(&c.ceiling.to_be_bytes());
        }
        out
    }

    /// Read the stored form back.
    ///
    /// # Errors
    ///
    /// The record is not one [`Self::encode`] wrote.
    pub(crate) fn decode(bytes: &[u8]) -> StorageResult<Self> {
        fn malformed(why: &str) -> StorageError {
            StorageError::Serialization(format!("the closed bound record {why}"))
        }
        fn take<const N: usize>(rest: &mut &[u8]) -> StorageResult<[u8; N]> {
            let (head, tail) = rest
                .split_first_chunk::<N>()
                .ok_or_else(|| malformed("is truncated"))?;
            *rest = tail;
            Ok(*head)
        }
        let (&format, mut rest) = bytes.split_first().ok_or_else(|| malformed("is empty"))?;
        if format != FORMAT {
            return Err(malformed(&format!("has format {format}, not {FORMAT}")));
        }
        let below = u64::from_be_bytes(take(&mut rest)?);
        let last_index = u64::from_be_bytes(take(&mut rest)?);
        let closed_through = u64::from_be_bytes(take(&mut rest)?);
        let closed_ceiling = u64::from_be_bytes(take(&mut rest)?);
        let count = usize::try_from(u32::from_be_bytes(take(&mut rest)?))
            .map_err(|_| malformed("counts more positions than fit"))?;
        if count > OPEN_CHECKPOINTS {
            return Err(malformed("keeps more open positions than a ledger does"));
        }
        let mut open = VecDeque::with_capacity(count);
        for _ in 0..count {
            let index = u64::from_be_bytes(take(&mut rest)?);
            let ceiling = u64::from_be_bytes(take(&mut rest)?);
            open.push_back(Checkpoint { index, ceiling });
        }
        if !rest.is_empty() {
            return Err(malformed("has bytes past its positions"));
        }
        Ok(Self {
            below,
            last_index,
            closed_through,
            closed_ceiling,
            open,
        })
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
