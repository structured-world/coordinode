//! Raft log entries written before each entry carried the leader's closed
//! bound.
//!
//! A Normal entry's request was encoded as a one-field array, its proposals;
//! the current request has a second field, the closed bound. The current
//! engine refuses such an entry when it reads its log ("invalid length 1,
//! expected struct Request with 2 elements"). The entry is rewritten with a
//! closed bound of 0, which is what an entry carries when it vouches for
//! nothing: the bound only ever moves forward from entries that do.

use std::path::Path;

use anyhow::{Context as _, bail};
use coordinode_raft::storage::Entry;
use coordinode_storage::oplog::OplogOp;
use rmpv::Value;

use crate::oplog::{self, Segment};
use crate::{Backup, Finding, Migration};

pub(super) const MIGRATION: Migration = Migration {
    name: "raft-closed-bound",
    summary: "Raft log entries without the closed bound get one of 0",
    survey,
    apply,
};

fn survey(data: &Path) -> anyhow::Result<Vec<Finding>> {
    let mut findings = Vec::new();
    for path in oplog::segment_files(data)? {
        let segment = oplog::read(&path)?;
        let (upgraded, _) = upgrade_segment(&path, segment)?;
        if upgraded > 0 {
            findings.push(Finding {
                path,
                detail: format!("{upgraded} log entries without a closed bound"),
            });
        }
    }
    Ok(findings)
}

fn apply(finding: &Finding, backup: &Backup) -> anyhow::Result<()> {
    let segment = oplog::read(&finding.path)?;
    let (upgraded, segment) = upgrade_segment(&finding.path, segment)?;
    if upgraded == 0 {
        // Already rewritten by an earlier run.
        return Ok(());
    }
    oplog::replace(&finding.path, &segment, backup)
}

/// The segment with every entry in the old shape rewritten, and how many
/// were. An entry that is neither current nor the old shape stops the
/// migration: rewriting around it would hide what it is.
fn upgrade_segment(path: &Path, mut segment: Segment) -> anyhow::Result<(usize, Segment)> {
    let mut upgraded = 0;
    for entry in &mut segment.entries {
        let Some(OplogOp::RaftEntry { data }) = entry.ops.first_mut() else {
            // Not a Raft log entry: the embedded journal writes these.
            continue;
        };
        if let Some(new) = upgrade_entry(data)
            .with_context(|| format!("log entry {} in {}", entry.index, path.display()))?
        {
            *data = new;
            upgraded += 1;
        }
    }
    Ok((upgraded, segment))
}

/// The entry `data` rewritten with a closed bound of 0, or `None` when it
/// already reads as a current entry.
///
/// An entry is `[log_id, payload]`; a Normal payload is a one-key map whose
/// value is the request array. Only a request array of exactly one field is
/// the old shape; the result must read back as a current entry.
pub(crate) fn upgrade_entry(data: &[u8]) -> anyhow::Result<Option<Vec<u8>>> {
    if rmp_serde::from_slice::<Entry>(data).is_ok() {
        return Ok(None);
    }
    let mut value = rmpv::decode::read_value(&mut &data[..]).context("not msgpack")?;
    let Value::Array(fields) = &mut value else {
        bail!("not a log entry: {value}");
    };
    let Some(Value::Map(payload)) = fields.get_mut(1) else {
        bail!("not a Normal log entry in a known shape");
    };
    let [(_, Value::Array(request))] = payload.as_mut_slice() else {
        bail!("not a Normal log entry in a known shape");
    };
    if request.len() != 1 {
        bail!(
            "request with {} fields, neither the old one nor the current two",
            request.len()
        );
    }
    request.push(Value::from(0u64));
    let mut out = Vec::with_capacity(data.len() + 1);
    rmpv::encode::write_value(&mut out, &value).context("encode the rewritten entry")?;
    rmp_serde::from_slice::<Entry>(&out).context("the rewritten entry does not read back")?;
    Ok(Some(out))
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
