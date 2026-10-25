//! Row type: a single result tuple from query execution.

use std::collections::BTreeMap;

use coordinode_core::graph::types::Value;

/// A result row: ordered map of column name → value.
///
/// Uses BTreeMap for deterministic column ordering in output.
pub type Row = BTreeMap<String, Value>;

/// Hash of `values` under which equal value lists hash alike (see
/// [`Value::hash_equal`]).
pub fn values_hash<'a>(values: impl IntoIterator<Item = &'a Value>) -> u64 {
    use core::hash::Hasher as _;
    let mut state = rustc_hash::FxHasher::default();
    for value in values {
        value.hash_equal(&mut state);
    }
    state.finish()
}

/// Hash of `row` under which equal rows hash alike.
pub fn row_hash(row: &Row) -> u64 {
    use core::hash::{Hash as _, Hasher as _};
    let mut state = rustc_hash::FxHasher::default();
    for (column, value) in row {
        column.hash(&mut state);
        value.hash_equal(&mut state);
    }
    state.finish()
}

/// The positions of the distinct items kept so far, chained by hash: an
/// exact deduplication compares an item only with the kept items sharing
/// its hash, not with every kept item.
#[derive(Default)]
pub struct DistinctIndex {
    /// The latest kept position with each hash.
    heads: rustc_hash::FxHashMap<u64, usize>,
    /// For each kept position, the previous kept position with its hash.
    previous: Vec<usize>,
    /// The positions the charged memory covers: never kept past this.
    reserved: usize,
}

impl DistinctIndex {
    const NONE: usize = usize::MAX;

    /// An upper bound of the memory the index holds once it has room for
    /// `positions`: the hash table's buckets (one entry and one control byte
    /// each, a power of two at least 8/7 of the entries) with its trailing
    /// control group, and one link per position.
    fn bytes_for(positions: usize) -> u64 {
        const GROUP: usize = 16;
        let entry = core::mem::size_of::<(u64, usize)>() + 1;
        let buckets = (positions / 7 * 8 + positions % 7 * 8 / 7 + 1)
            .max(4)
            .next_power_of_two();
        (buckets * entry + GROUP + positions * core::mem::size_of::<usize>()) as u64
    }

    /// Makes room for `additional` more kept positions, charging `charge`
    /// with the memory before allocating it.
    pub fn reserve(
        &mut self,
        additional: usize,
        charge: &mut coordinode_core::budget::MemoryCharge<'_>,
    ) -> Result<(), coordinode_core::budget::BudgetStop> {
        let want = self.previous.len() + additional;
        if want <= self.reserved {
            return Ok(());
        }
        // Charged so far: the bound for the room already made, nothing for
        // an index that allocated nothing.
        let charged = if self.reserved == 0 {
            0
        } else {
            Self::bytes_for(self.reserved)
        };
        charge.grow(Self::bytes_for(want) - charged)?;
        self.heads.reserve(want - self.heads.len());
        self.previous.reserve_exact(want - self.previous.len());
        self.reserved = want;
        Ok(())
    }

    /// The kept position whose item `is_equal` accepts among those with
    /// `hash`, and the number of positions it was asked about.
    pub fn find(&self, hash: u64, mut is_equal: impl FnMut(usize) -> bool) -> (Option<usize>, u64) {
        let mut asked = 0;
        let mut at = self.heads.get(&hash).copied().unwrap_or(Self::NONE);
        while at != Self::NONE {
            asked += 1;
            if is_equal(at) {
                return (Some(at), asked);
            }
            at = self.previous[at];
        }
        (None, asked)
    }

    /// Records the next kept position (the count of positions kept before
    /// it) under `hash`, within the room [`Self::reserve`] made.
    pub fn keep(&mut self, hash: u64) -> usize {
        let at = self.previous.len();
        debug_assert!(at < self.reserved, "kept past the reserved room");
        let previous = self.heads.insert(hash, at).unwrap_or(Self::NONE);
        self.previous.push(previous);
        at
    }
}

/// Pushes `item`, charging `charge` before the push allocates: a full list
/// doubles (at least four slots) and the new slots are charged first.
pub fn push_charged<T>(
    items: &mut Vec<T>,
    item: T,
    charge: &mut coordinode_core::budget::MemoryCharge<'_>,
) -> Result<(), coordinode_core::budget::BudgetStop> {
    if items.len() == items.capacity() {
        let grow = items.capacity().max(4);
        charge.grow((grow * core::mem::size_of::<T>()) as u64)?;
        items.reserve_exact(grow);
    }
    items.push(item);
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;

/// Memory `row` holds: what a query budget charges for keeping it.
pub fn row_held_bytes(row: &Row) -> u64 {
    core::mem::size_of::<Row>() as u64
        + row
            .iter()
            .map(|(column, value)| {
                coordinode_core::graph::types::map_entry_bytes(column) + value.held_bytes()
            })
            .sum::<u64>()
}
