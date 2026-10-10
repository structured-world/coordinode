//! Every migration this build ships, oldest first. A migration names the
//! shape it was written for in its own module, with the evidence that
//! identifies it.

mod index_build_repair;
mod label_schema_shape;
mod raft_closed_bound;

use std::path::Path;

use anyhow::Context as _;
use coordinode_storage::engine::partition::Partition;

use crate::{Finding, Migration, open_store, stores};

/// The migrations, in the order a run applies them.
pub static MIGRATIONS: &[Migration] = &[
    raft_closed_bound::MIGRATION,
    index_build_repair::MIGRATION,
    label_schema_shape::MIGRATION,
];

/// A family of Schema records a migration rewrites: where they are, which
/// keys under the prefix are records, and how one is upgraded (`None` when
/// it is current already).
pub(crate) struct SchemaRecords {
    pub(crate) what: &'static str,
    pub(crate) prefix: &'static [u8],
    pub(crate) is_record: fn(&[u8]) -> bool,
    pub(crate) upgrade: fn(&[u8]) -> anyhow::Result<Option<Vec<u8>>>,
}

impl SchemaRecords {
    /// Every record of the family in the store at `dir` that needs a
    /// rewrite, with its new value. A record of no known shape stops the run.
    fn old(&self, dir: &Path) -> anyhow::Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let engine = open_store(dir)?;
        let mut out = Vec::new();
        for guard in engine
            .prefix_scan(Partition::Schema, self.prefix)
            .with_context(|| format!("scan {}", dir.display()))?
        {
            let (key, value) = guard
                .into_inner()
                .with_context(|| format!("scan {}", dir.display()))?;
            if !(self.is_record)(&key) {
                continue;
            }
            if let Some(new) = (self.upgrade)(&value).with_context(|| {
                format!(
                    "{} {} in {}",
                    self.what,
                    coordinode_storage::error::printable_key(&key),
                    dir.display()
                )
            })? {
                out.push((key.to_vec(), new));
            }
        }
        Ok(out)
    }

    /// One finding per store under `data` holding records to rewrite.
    pub(crate) fn survey(&self, data: &Path) -> anyhow::Result<Vec<Finding>> {
        let mut findings = Vec::new();
        for dir in stores(data)? {
            let n = self.old(&dir)?.len();
            if n > 0 {
                findings.push(Finding {
                    path: dir,
                    detail: format!("{n} {} records in an earlier shape", self.what),
                });
            }
        }
        Ok(findings)
    }

    /// Rewrite the records of the store a finding names, and flush.
    ///
    /// The rewrite is a new version of each key: the old one stays in the
    /// store's files until compaction drops it, so no file is replaced and
    /// nothing goes to the backup.
    pub(crate) fn apply(&self, finding: &Finding) -> anyhow::Result<()> {
        let records = self.old(&finding.path)?;
        let engine = open_store(&finding.path)?;
        for (key, value) in records {
            engine
                .put(Partition::Schema, &key, &value)
                .with_context(|| {
                    format!(
                        "rewrite {} in {}",
                        coordinode_storage::error::printable_key(&key),
                        finding.path.display()
                    )
                })?;
        }
        engine
            .persist()
            .with_context(|| format!("flush {}", finding.path.display()))
    }
}
