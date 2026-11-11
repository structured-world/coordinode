//! Row type: a single result tuple from query execution.

use std::collections::BTreeMap;

use coordinode_core::graph::types::Value;

/// A result row: ordered map of column name → value.
///
/// Uses BTreeMap for deterministic column ordering in output.
pub type Row = BTreeMap<String, Value>;

/// The bytes a `var.property` column name is assembled in without the heap;
/// longer names fall back to an allocated one.
const COLUMN_NAME_INLINE: usize = 128;

/// What [`property`] allocates to find `var.property`: nothing for a name
/// the stack holds, the assembled name past that. A charged reader reserves
/// it before the lookup.
pub fn column_lookup_bytes(var: &str, property: &str) -> u64 {
    let len = var.len() + 1 + property.len();
    if len > COLUMN_NAME_INLINE {
        len as u64
    } else {
        0
    }
}

/// The value of column `var.property` in `row`, by one keyed lookup. A name
/// the stack holds is assembled there; a longer one allocates
/// [`column_lookup_bytes`] for the lookup's length.
pub fn property<'r>(row: &'r Row, var: &str, property: &str) -> Option<&'r Value> {
    let len = var.len() + 1 + property.len();
    if len > COLUMN_NAME_INLINE {
        return row.get(column_name(var, property).as_str());
    }
    let mut name = [0u8; COLUMN_NAME_INLINE];
    name[..var.len()].copy_from_slice(var.as_bytes());
    name[var.len()] = b'.';
    name[var.len() + 1..len].copy_from_slice(property.as_bytes());
    // Two UTF-8 strings joined by an ASCII dot are UTF-8.
    let name = core::str::from_utf8(&name[..len]).ok()?;
    row.get(name)
}

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
    reserve_one_charged(items, charge)?;
    items.push(item);
    Ok(())
}

/// Makes room in `items` for one more item, charging `charge` before the
/// room is allocated: a full list doubles (at least four slots).
pub fn reserve_one_charged<T>(
    items: &mut Vec<T>,
    charge: &mut coordinode_core::budget::MemoryCharge<'_>,
) -> Result<(), coordinode_core::budget::BudgetStop> {
    if items.len() == items.capacity() {
        let grow = items.capacity().max(4);
        charge.grow((grow * core::mem::size_of::<T>()) as u64)?;
        items.reserve_exact(grow);
    }
    Ok(())
}

/// The column `variable.name`, holding exactly its bytes, so that
/// [`column_entry_bytes`] is what a row keeping it holds for its key.
pub fn column_name(variable: &str, name: &str) -> String {
    let mut column = String::with_capacity(variable.len() + 1 + name.len());
    column.push_str(variable);
    column.push('.');
    column.push_str(name);
    column
}

/// What [`row_held_bytes`] counts for the key of an entry under
/// [`column_name`]`(variable, name)`, known before the key is made.
pub fn column_entry_bytes(variable: &str, name: &str) -> u64 {
    key_entry_bytes(variable.len() + 1 + name.len())
}

/// What [`row_held_bytes`] counts for an entry's key of exactly `len` bytes.
pub fn key_entry_bytes(len: usize) -> u64 {
    (core::mem::size_of::<String>() + core::mem::size_of::<usize>() + len) as u64
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
