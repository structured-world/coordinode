//! What a store's node records hold now: per label, how many records and how
//! big, and which properties make them big. For comparing the data a store
//! keeps with the bytes it spends on it.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Context as _;
use coordinode_core::graph::node::NodeRecord;
use coordinode_storage::Guard as _;
use coordinode_storage::engine::partition::Partition;

/// Totals of one label's current records.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LabelSizes {
    /// Records whose primary label it is.
    pub records: u64,
    /// Bytes of those records as stored.
    pub bytes: u64,
    /// Bytes of the largest one.
    pub max_bytes: u64,
    /// Per property (field id, or the name of an overflow property): records
    /// holding it and the bytes of its encoded values.
    pub properties: Vec<(String, u64, u64)>,
}

/// Per property: records holding it and the bytes of its values.
type PropertyTotals = HashMap<String, (u64, u64)>;

/// The current node records of the store at `dir`, per primary label, the
/// largest labels first.
///
/// # Errors
///
/// The store cannot be opened, or a record cannot be read or decoded.
pub fn node_sizes(dir: &Path) -> anyhow::Result<Vec<(String, LabelSizes)>> {
    let engine = crate::open_store(dir)?;
    let mut labels: HashMap<String, (LabelSizes, PropertyTotals)> = HashMap::new();
    for row in engine
        .prefix_scan(Partition::Node, b"node:")
        .context("scan the node records")?
    {
        let (_, value) = row.into_inner().context("read a node record")?;
        let record = NodeRecord::from_msgpack(&value).context("decode a node record")?;
        let (sizes, properties) = labels
            .entry(record.primary_label().to_string())
            .or_default();
        let bytes = value.len() as u64;
        sizes.records += 1;
        sizes.bytes += bytes;
        sizes.max_bytes = sizes.max_bytes.max(bytes);
        let mut count = |name: String, value: &coordinode_core::graph::types::Value| {
            let size = rmp_serde::to_vec(value).map_or(0, |v| v.len() as u64);
            let entry = properties.entry(name).or_default();
            entry.0 += 1;
            entry.1 += size;
        };
        for (field, value) in &record.props {
            count(format!("#{field}"), value);
        }
        for (name, value) in record.extra.iter().flatten() {
            count(name.clone(), value);
        }
    }
    let mut out: Vec<(String, LabelSizes)> = labels
        .into_iter()
        .map(|(label, (mut sizes, properties))| {
            let mut properties: Vec<(String, u64, u64)> = properties
                .into_iter()
                .map(|(name, (records, bytes))| (name, records, bytes))
                .collect();
            properties.sort_by_key(|p| core::cmp::Reverse(p.2));
            sizes.properties = properties;
            (label, sizes)
        })
        .collect();
    out.sort_by_key(|(_, sizes)| core::cmp::Reverse(sizes.bytes));
    Ok(out)
}
