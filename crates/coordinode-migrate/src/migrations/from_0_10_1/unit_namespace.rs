//! Units of the journal name the namespace their data is written in.
//!
//! 0.10.1 wrote a unit as its operations alone. This build refuses a unit
//! that writes a namespace's data before a namespace marker names the
//! namespace, so the journal of a 0.10.1 store does not replay. Every other
//! byte of the frame is read the same by both releases: the marker is its
//! own operation, a tag and the namespace's number, which leaves the key
//! and literal references of the operations after it as they were. So the
//! marker is put in front of the operations, the operation count raised by
//! one and the checksum written again; the frame is then read back by this
//! build's decoder before anything is replaced.
//!
//! Units ride in the store's own journal and in its group's Raft log, as
//! unit operations of an entry, the same frame in both.

use std::path::Path;

use anyhow::{Context as _, bail};
use coordinode_core::txn::frame::{FRAME_VERSION, FrameError, decode_unit};
use coordinode_storage::oplog::OplogOp;

use super::NAMESPACE;
use crate::oplog::{self, Segment};
use crate::{Backup, Finding, Migration};

pub(super) const MIGRATION: Migration = Migration {
    name: "unit-namespace",
    summary: "journal units name the namespace their data is written in",
    survey,
    apply,
};

/// The tag of a namespace marker in a frame: the command kind (4) with a
/// partition number (1) no command carries.
const TAG_NAMESPACE: u8 = 4 | (1 << 3);

fn survey(data: &Path) -> anyhow::Result<Vec<Finding>> {
    let mut findings = Vec::new();
    for path in oplog::segment_files(data)? {
        let segment = oplog::read(&path)?;
        let (named, _) = name_segment(&path, segment)?;
        if named > 0 {
            findings.push(Finding {
                path,
                detail: format!("{named} units naming no namespace"),
            });
        }
    }
    Ok(findings)
}

fn apply(finding: &Finding, backup: &Backup) -> anyhow::Result<()> {
    let segment = oplog::read(&finding.path)?;
    let (named, segment) = name_segment(&finding.path, segment)?;
    if named == 0 {
        // Rewritten by an earlier run.
        return Ok(());
    }
    oplog::replace(&finding.path, &segment, backup)
}

/// The segment with every unit that names no namespace given the marker,
/// and how many were. A frame that neither reads now nor reads once named
/// stops the migration: rewriting around it would hide what it is.
fn name_segment(path: &Path, mut segment: Segment) -> anyhow::Result<(usize, Segment)> {
    let mut named = 0;
    for entry in &mut segment.entries {
        for op in &mut entry.ops {
            let OplogOp::Unit { frame } = op else {
                continue;
            };
            if let Some(new) = name_unit(frame)
                .with_context(|| format!("log entry {} in {}", entry.index, path.display()))?
            {
                *frame = new;
                named += 1;
            }
        }
    }
    Ok((named, segment))
}

/// `frame` with the namespace marker in front of its operations, or `None`
/// when it reads already.
pub(crate) fn name_unit(frame: &[u8]) -> anyhow::Result<Option<Vec<u8>>> {
    match decode_unit(frame) {
        Ok(_) => return Ok(None),
        Err(FrameError::Unscoped(_)) => {}
        Err(e) => bail!("a unit frame that does not read: {e}"),
    }
    let body = frame
        .get(..frame.len().saturating_sub(4))
        .context("a frame shorter than its checksum")?;
    let mut at = 0;
    if body.first() != Some(&FRAME_VERSION) {
        bail!("a unit frame of layout {:?}", body.first());
    }
    at += 1;
    // The unit's id, commit and start timestamps.
    for _ in 0..3 {
        varint(body, &mut at)?;
    }
    // The flags.
    at += 1;
    let count_at = at;
    let count = varint(body, &mut at)?;
    let mut new = Vec::with_capacity(frame.len() + 12);
    new.extend_from_slice(&body[..count_at]);
    put_varint(count + 1, &mut new);
    new.push(TAG_NAMESPACE);
    put_varint(NAMESPACE.as_raw(), &mut new);
    new.extend_from_slice(&body[at..]);
    let crc = crc32c::crc32c(&new);
    new.extend_from_slice(&crc.to_le_bytes());
    decode_unit(&new).context("the unit, once named, does not read")?;
    Ok(Some(new))
}

/// The varint at `*at` of `bytes`, `*at` moved past it.
fn varint(bytes: &[u8], at: &mut usize) -> anyhow::Result<u64> {
    let mut value = 0u64;
    for shift in (0..64).step_by(7) {
        let byte = *bytes.get(*at).context("a frame header cut short")?;
        *at += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    bail!("a varint longer than 64 bits in a frame header")
}

fn put_varint(mut value: u64, out: &mut Vec<u8>) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
