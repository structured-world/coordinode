//! Index build records written before a build could repair the duplicates
//! it meets.
//!
//! A record was a five-field array (generation, index, failure policy,
//! state, nodes indexed); the current one adds the duplicate repair policy
//! and the count of repairs made. The current server refuses the old record
//! when it takes up unfinished builds ("invalid length 5, expected struct
//! IndexBuildRecord with 7 elements"), retrying forever. The record gets no
//! repair policy and no repairs, which is what a build did before: fail on
//! the first duplicate.
//!
//! The records live in the Schema partition of the store and of every
//! checkpoint, which is opened here as a plain engine: no journal is
//! replayed, and the rewrite is flushed before the engine closes.

use std::path::Path;

use anyhow::{Context as _, bail};
use coordinode_modality::IndexBuildRecord;
use rmpv::Value;

use super::SchemaRecords;
use crate::{Backup, Finding, Migration};

pub(super) const MIGRATION: Migration = Migration {
    name: "index-build-repair",
    summary: "index build records without the duplicate repair fields get none",
    survey,
    apply,
};

const RECORDS: SchemaRecords = SchemaRecords {
    what: "index build",
    prefix: IndexBuildRecord::PREFIX,
    is_record: |_| true,
    upgrade: upgrade_record,
};

fn survey(data: &Path) -> anyhow::Result<Vec<Finding>> {
    RECORDS.survey(data)
}

fn apply(finding: &Finding, _backup: &Backup) -> anyhow::Result<()> {
    RECORDS.apply(finding)
}

/// The record `data` with no repair policy and no repairs, or `None` when it
/// already reads as a current record. Only a five-field record is the old
/// shape; the result must read back as a current record.
pub(crate) fn upgrade_record(data: &[u8]) -> anyhow::Result<Option<Vec<u8>>> {
    if rmp_serde::from_slice::<IndexBuildRecord>(data).is_ok() {
        return Ok(None);
    }
    let mut value = rmpv::decode::read_value(&mut &data[..]).context("not msgpack")?;
    let Value::Array(fields) = &mut value else {
        bail!("not an index build record: {value}");
    };
    if fields.len() != 5 {
        bail!(
            "record with {} fields, neither the old five nor the current seven",
            fields.len()
        );
    }
    fields.push(Value::Nil);
    fields.push(Value::from(0u64));
    let mut out = Vec::with_capacity(data.len() + 2);
    rmpv::encode::write_value(&mut out, &value).context("encode the rewritten record")?;
    rmp_serde::from_slice::<IndexBuildRecord>(&out)
        .context("the rewritten record does not read back")?;
    Ok(Some(out))
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
