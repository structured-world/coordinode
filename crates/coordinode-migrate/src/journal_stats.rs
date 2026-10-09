//! What a store's journal holds: which writes, how big, how often the same
//! key is written again. For finding writes that cost more than the data
//! they carry.

use std::collections::HashMap;
use std::path::{Component, Path};

use anyhow::Context as _;
use coordinode_core::txn::frame::{DecodeLimits, decode_proposal};
use coordinode_core::txn::proposal::Mutation;
use coordinode_storage::oplog::OplogOp;

use crate::oplog;

/// Totals of one class of writes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ClassStats {
    /// Writes of the class.
    pub writes: u64,
    /// Bytes of their keys.
    pub key_bytes: u64,
    /// Bytes of their values or operands.
    pub value_bytes: u64,
    /// Distinct keys written.
    pub distinct_keys: u64,
    /// Most writes of one key.
    pub max_per_key: u64,
}

/// What the journal of a store holds.
#[derive(Debug, Clone, Default)]
pub struct JournalStats {
    /// Journal records read.
    pub records: u64,
    /// Bytes of their Raft envelopes.
    pub envelope_bytes: u64,
    /// Bytes of their proposal frames.
    pub frame_bytes: u64,
    /// Proposals decoded from the frames.
    pub proposals: u64,
    /// Lowest and highest record timestamp, HLC.
    pub span: Option<(u64, u64)>,
    /// Per class of write: `<kind> <partition> <key prefix>`.
    pub classes: Vec<(String, ClassStats)>,
    /// The keys written most often: class, printable key, writes.
    pub hottest: Vec<(String, String, u64)>,
    /// Node record rewrites per primary label, by how much of each record
    /// changed against the version before it.
    pub node_rewrites: Vec<(String, NodeRewrites)>,
}

/// How a label's node records were rewritten.
#[derive(Debug, Clone, Default)]
pub struct NodeRewrites {
    /// Puts of a record whose key was written before in the journal.
    pub rewrites: u64,
    /// Bytes of those records.
    pub bytes: u64,
    /// Bytes of the properties that differ from the version before.
    pub changed_bytes: u64,
    /// Per property (field id, or name for an undeclared one): times it
    /// changed, and its size in the last record.
    pub fields: Vec<(String, u64, u64)>,
}

/// The printable leading part of a key that names what it is: ASCII up to
/// the second ':' or the first byte that is not a name character.
fn key_class(key: &[u8]) -> String {
    let mut out = String::new();
    let mut colons = 0;
    for &b in key {
        if b == b':' {
            out.push(':');
            colons += 1;
            if colons == 2 {
                break;
            }
        } else if b.is_ascii_alphanumeric() || b == b'_' || b == b'-' {
            out.push(char::from(b));
        } else {
            break;
        }
    }
    if out.is_empty() {
        format!("0x{:02x}..", key.first().copied().unwrap_or(0))
    } else {
        out
    }
}

/// Read every segment of the store's own journal under `data` (the copies
/// inside checkpoints are left out) and total what it holds.
///
/// # Errors
///
/// A segment or a proposal frame does not read.
pub fn journal_stats(data: &Path, hottest: usize) -> anyhow::Result<JournalStats> {
    let mut stats = JournalStats::default();
    let mut classes: HashMap<String, ClassStats> = HashMap::new();
    let mut per_key: HashMap<(String, Vec<u8>), u64> = HashMap::new();
    let mut nodes = NodeTracker::default();
    for path in oplog::segment_files(data)? {
        let inside_checkpoint = path
            .strip_prefix(data)
            .map(|rel| {
                rel.components()
                    .any(|c| c == Component::Normal("checkpoints".as_ref()))
            })
            .unwrap_or(false);
        if inside_checkpoint {
            continue;
        }
        for entry in oplog::read(&path)?.entries {
            stats.records += 1;
            // Membership and blank entries carry no commit timestamp.
            if entry.ts != 0 {
                stats.span = Some(match stats.span {
                    None => (entry.ts, entry.ts),
                    Some((lo, hi)) => (lo.min(entry.ts), hi.max(entry.ts)),
                });
            }
            for op in &entry.ops {
                match op {
                    OplogOp::RaftEntry { data } => stats.envelope_bytes += data.len() as u64,
                    OplogOp::Unit { frame } => {
                        stats.frame_bytes += frame.len() as u64;
                        let proposal = decode_proposal(frame, &DecodeLimits::DEFAULT)
                            .with_context(|| {
                                format!("entry {} in {}", entry.index, path.display())
                            })?;
                        stats.proposals += 1;
                        for m in &proposal.mutations {
                            if let Mutation::Put {
                                partition: coordinode_core::txn::proposal::PartitionId::Node,
                                key,
                                value,
                            } = m
                            {
                                nodes.put(key, value);
                            }
                            let (class, key, value_len) = describe(m);
                            let c = classes.entry(class.clone()).or_default();
                            c.writes += 1;
                            c.key_bytes += key.len() as u64;
                            c.value_bytes += value_len;
                            *per_key.entry((class, key.to_vec())).or_insert(0) += 1;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    for ((class, _), n) in &per_key {
        if let Some(c) = classes.get_mut(class) {
            c.distinct_keys += 1;
            c.max_per_key = c.max_per_key.max(*n);
        }
    }
    let mut hot: Vec<_> = per_key.into_iter().collect();
    hot.sort_by_key(|a| core::cmp::Reverse(a.1));
    stats.hottest = hot
        .into_iter()
        .take(hottest)
        .map(|((class, key), n)| (class, coordinode_storage::error::printable_key(&key), n))
        .collect();
    let mut classes: Vec<_> = classes.into_iter().collect();
    classes.sort_by_key(|a| core::cmp::Reverse(a.1.key_bytes + a.1.value_bytes));
    stats.classes = classes;
    stats.node_rewrites = nodes.finish();
    Ok(stats)
}

/// Per property of a label: times it changed, and its size in the last
/// record.
type FieldChanges = HashMap<String, (u64, u64)>;

/// The last version of every node record seen, and what each rewrite
/// changed against it.
#[derive(Default)]
struct NodeTracker {
    /// Per key: the property encodings of the last version.
    last: HashMap<Vec<u8>, HashMap<String, Vec<u8>>>,
    /// Per label: totals, and per field changes.
    labels: HashMap<String, (NodeRewrites, FieldChanges)>,
}

impl NodeTracker {
    fn put(&mut self, key: &[u8], value: &[u8]) {
        let Some((label, props)) = node_props(value) else {
            return;
        };
        let (totals, fields) = self.labels.entry(label).or_default();
        let previous = self.last.insert(key.to_vec(), props);
        let Some(previous) = previous else {
            return;
        };
        let Some(props) = self.last.get(key) else {
            return;
        };
        totals.rewrites += 1;
        totals.bytes += value.len() as u64;
        for (field, bytes) in props {
            let entry = fields.entry(field.clone()).or_default();
            entry.1 = bytes.len() as u64;
            if previous.get(field) != Some(bytes) {
                entry.0 += 1;
                totals.changed_bytes += bytes.len() as u64;
            }
        }
    }

    fn finish(self) -> Vec<(String, NodeRewrites)> {
        let mut out: Vec<_> = self
            .labels
            .into_iter()
            .filter(|(_, (t, _))| t.rewrites > 0)
            .map(|(label, (mut totals, fields))| {
                let mut fields: Vec<_> = fields
                    .into_iter()
                    .map(|(f, (changes, size))| (f, changes, size))
                    .collect();
                fields.sort_by_key(|a| core::cmp::Reverse(a.2));
                totals.fields = fields;
                (label, totals)
            })
            .collect();
        out.sort_by_key(|a| core::cmp::Reverse(a.1.bytes));
        out
    }
}

/// A node record's primary label and the encoding of each property, keyed
/// by field id (declared) or name (undeclared). `None` for a value that is
/// not a node record.
fn node_props(value: &[u8]) -> Option<(String, HashMap<String, Vec<u8>>)> {
    let record = rmpv::decode::read_value(&mut &value[..]).ok()?;
    let rmpv::Value::Array(parts) = record else {
        return None;
    };
    let label = match parts.first()? {
        rmpv::Value::Array(labels) => labels.first()?.as_str()?.to_string(),
        _ => return None,
    };
    let mut props = HashMap::new();
    for (i, part) in parts.iter().enumerate().skip(1) {
        let rmpv::Value::Map(map) = part else {
            continue;
        };
        for (k, v) in map {
            let name = if i == 1 {
                format!("#{k}")
            } else {
                k.as_str().unwrap_or("?").to_string()
            };
            let mut bytes = Vec::new();
            rmpv::encode::write_value(&mut bytes, v).ok()?;
            props.insert(name, bytes);
        }
    }
    Some((label, props))
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;

/// A mutation's class, key and value size.
fn describe(m: &Mutation) -> (String, &[u8], u64) {
    match m {
        Mutation::Put {
            partition,
            key,
            value,
        } => (
            format!("put {partition:?} {}", key_class(key)),
            key,
            value.len() as u64,
        ),
        Mutation::Delete { partition, key } => {
            (format!("delete {partition:?} {}", key_class(key)), key, 0)
        }
        Mutation::Merge {
            partition,
            key,
            operand,
        } => (
            format!("merge {partition:?} {}", key_class(key)),
            key,
            operand.len() as u64,
        ),
        Mutation::RemoveRange {
            partition, start, ..
        } => (
            format!("remove-range {partition:?} {}", key_class(start)),
            start,
            0,
        ),
        Mutation::Command(_) => ("command".to_string(), &[], 0),
        Mutation::Derive(work) => (
            "derive".to_string(),
            &[],
            rmp_serde::to_vec(work).map_or(0, |b| b.len() as u64),
        ),
    }
}
