//! Series that disappear with what they describe.
//!
//! The `metrics` facade cannot remove a series: one set once is rendered
//! for the life of the process. The series here describe things that come
//! and go (a `STORAGE COLUMNAR` table, a label with nodes), so they are kept
//! by the engine and rendered beside the facade's output, and each goes when
//! its subject does.

use std::collections::BTreeMap;

/// The write-pressure series of each live `STORAGE COLUMNAR` table.
///
/// A table's series lives exactly as long as the table: the compaction
/// monitor sets it only while the table's life holds, and `DROP TABLE`
/// removes it under the same life it ends, so a table dropped or re-created
/// under its name never leaves a series behind or has one overwritten by the
/// monitor of the table it replaced.
#[derive(Debug, Default)]
pub struct TableMetrics {
    series: parking_lot::Mutex<BTreeMap<String, TableSeries>>,
}

#[derive(Debug, Clone, Copy)]
struct TableSeries {
    /// Write-pressure tier: 0 none, 1 slowdown, 2 stop.
    tier: u8,
    /// Bytes compaction still owes the table's tree.
    debt: u64,
}

impl TableMetrics {
    /// Record `table`'s verdict. Called under the table's life, never after
    /// its drop.
    pub(crate) fn set(&self, table: &str, tier: u8, debt: u64) {
        let mut series = self.series.lock();
        match series.get_mut(table) {
            Some(entry) => *entry = TableSeries { tier, debt },
            None => {
                series.insert(table.to_owned(), TableSeries { tier, debt });
            }
        }
    }

    /// Remove `table`'s series. Called by its drop under the life it ends.
    pub(crate) fn remove(&self, table: &str) {
        self.series.lock().remove(table);
    }

    /// Append the series in the Prometheus text exposition format: one
    /// `coordinode_storage_table_backpressure_tier` and one
    /// `coordinode_storage_table_compaction_debt_bytes` sample per live table.
    /// Nothing is written while no table exists.
    pub fn render(&self, out: &mut String) {
        let series = self.series.lock();
        push_family(
            out,
            "coordinode_storage_table_backpressure_tier",
            "Write-pressure tier of a STORAGE COLUMNAR table: 0 none, 1 slowdown, 2 stop.",
            "table",
            series.iter().map(|(t, s)| (t.as_str(), u64::from(s.tier))),
        );
        push_family(
            out,
            "coordinode_storage_table_compaction_debt_bytes",
            "Bytes compaction still owes a STORAGE COLUMNAR table's tree.",
            "table",
            series.iter().map(|(t, s)| (t.as_str(), s.debt)),
        );
    }
}

/// The stored nodes of each label that has any.
///
/// Replaced whole on every sample of the label counters, so a label whose
/// last node is gone loses its series with the next sample: the counters
/// themselves are kept forever and only move by deltas.
#[derive(Debug, Default)]
pub struct NodeCounts {
    series: parking_lot::Mutex<BTreeMap<String, u64>>,
}

impl NodeCounts {
    /// Publish `counts`, every label with nodes, in place of the last sample.
    pub(crate) fn replace(&self, counts: BTreeMap<String, u64>) {
        *self.series.lock() = counts;
    }

    /// Append `coordinode_graph_nodes_total`, one sample per label with
    /// nodes, in the Prometheus text exposition format. Nothing is written
    /// while no label has any.
    pub fn render(&self, out: &mut String) {
        let series = self.series.lock();
        push_family(
            out,
            "coordinode_graph_nodes_total",
            "Stored nodes per label.",
            "label",
            series.iter().map(|(label, count)| (label.as_str(), *count)),
        );
    }
}

/// Append one gauge family with one label: its `HELP` and `TYPE` lines and a
/// sample per `(label value, value)`. Nothing at all for no samples, so no
/// header is left without a sample.
fn push_family<'a>(
    out: &mut String,
    name: &str,
    help: &str,
    label: &str,
    samples: impl Iterator<Item = (&'a str, u64)>,
) {
    let mut samples = samples.peekable();
    if samples.peek().is_none() {
        return;
    }
    for (kind, text) in [("# HELP ", help), ("# TYPE ", "gauge")] {
        out.push_str(kind);
        out.push_str(name);
        out.push(' ');
        out.push_str(text);
        out.push('\n');
    }
    for (value_of_label, value) in samples {
        out.push_str(name);
        out.push('{');
        out.push_str(label);
        out.push_str("=\"");
        push_label_value(out, value_of_label);
        out.push_str("\"} ");
        out.push_str(&value.to_string());
        out.push('\n');
    }
}

/// Append a label value escaped as the exposition format requires:
/// backslash, double quote and line feed.
fn push_label_value(out: &mut String, value: &str) {
    for c in value.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            c => out.push(c),
        }
    }
}

#[cfg(test)]
mod tests;
