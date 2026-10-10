//! Row type: a single result tuple from query execution.

use std::collections::BTreeMap;

use coordinode_core::graph::types::Value;

/// A result row: ordered map of column name → value.
///
/// Uses BTreeMap for deterministic column ordering in output.
pub type Row = BTreeMap<String, Value>;

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
