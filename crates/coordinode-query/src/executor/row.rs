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
}

impl DistinctIndex {
    /// Memory one indexed position holds, for a budget's charge.
    pub const ENTRY_BYTES: u64 =
        (core::mem::size_of::<(u64, usize)>() + core::mem::size_of::<usize>()) as u64;

    const NONE: usize = usize::MAX;

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
    /// it) under `hash`.
    pub fn keep(&mut self, hash: u64) -> usize {
        let at = self.previous.len();
        let previous = self.heads.insert(hash, at).unwrap_or(Self::NONE);
        self.previous.push(previous);
        at
    }
}

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
