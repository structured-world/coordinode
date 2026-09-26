//! The transactions open on this engine, by the seqno they opened at.
//!
//! Some changes take effect for a writer only once it looks: a new index is
//! maintained by the transactions that see its definition, and a transaction
//! that decided what to maintain before the definition existed goes on
//! without it until it ends. Whoever makes such a change takes a
//! [`StorageEngine::snapshot_boundary`](crate::engine::core::StorageEngine::snapshot_boundary)
//! after it and waits for the transactions opened at or before it to end
//! ([`StorageEngine::await_transactions_through`](crate::engine::core::StorageEngine::await_transactions_through)),
//! the way PostgreSQL's concurrent index build waits for older snapshots.
//!
//! A snapshot pin does not mark a transaction's life: a statement repins at
//! its own snapshot, and a transaction can exist before it pins anything.
//! This table is entered when the transaction is created and left when its
//! last state is dropped, whatever it read in between.

use std::collections::BTreeMap;
use std::sync::Arc;

use parking_lot::Mutex;

/// Open transactions keyed by the seqno each opened at.
#[derive(Debug, Default)]
pub(crate) struct OpenTransactions {
    by_seqno: Mutex<BTreeMap<u64, usize>>,
}

impl OpenTransactions {
    /// Enter one transaction at the seqno `current` reads.
    ///
    /// The seqno is read under the table's lock, so a boundary allocated
    /// before a check that finds this transaction absent is below every
    /// seqno it can be entered at later.
    pub(crate) fn open(self: &Arc<Self>, current: impl FnOnce() -> u64) -> OpenTransaction {
        let mut table = self.by_seqno.lock();
        let seqno = current();
        // Pairs with the fence ahead of the boundary's allocation: a
        // transaction that reads a seqno past the boundary also sees what
        // was published before the boundary was taken.
        std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
        *table.entry(seqno).or_insert(0) += 1;
        OpenTransaction {
            table: Arc::clone(self),
            seqno,
        }
    }

    /// How many transactions opened at or before `boundary` are still open.
    pub(crate) fn open_through(&self, boundary: u64) -> usize {
        self.by_seqno
            .lock()
            .range(..=boundary)
            .map(|(_, n)| *n)
            .sum()
    }

    fn close(&self, seqno: u64) {
        let mut table = self.by_seqno.lock();
        if let Some(count) = table.get_mut(&seqno) {
            *count -= 1;
            if *count == 0 {
                table.remove(&seqno);
            }
        }
    }
}

/// One open transaction. Moves with the transaction's parked state between
/// the statements of an interactive transaction; leaves the table on drop.
#[derive(Debug)]
pub struct OpenTransaction {
    table: Arc<OpenTransactions>,
    seqno: u64,
}

impl Drop for OpenTransaction {
    fn drop(&mut self) {
        self.table.close(self.seqno);
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests;
