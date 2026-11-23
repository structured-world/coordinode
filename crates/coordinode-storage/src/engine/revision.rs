//! The revision a stored record carries: the commit timestamp of the write
//! that produced its value, kept in the value's bytes.
//!
//! A record's physical sequence number does not name its write once storage
//! has rewritten the record: a bottommost compaction settles old entries to
//! sequence number zero, and installing a copy of a partition writes its rows
//! at a sequence number of the install's choosing. So every value of a key
//! outside the reserved namespace, in a partition whose records are replaced
//! rather than folded, ends with the eight little-endian bytes of the
//! timestamp of the commit that wrote it. A copy of the row copies the stamp;
//! an ordinary write stamps its own commit. Readers of values see the payload
//! only; [`super::core::StorageEngine::record_version`] reads the stamp.

use std::borrow::Cow;

use super::coverage;
use super::partition::Partition;

/// The length of the stamp at the end of a stamped value.
pub(crate) const STAMP_LEN: usize = 8;

/// The revision of a record whose original commit timestamp is not known:
/// no commit is ever at timestamp zero, so it names none.
pub(crate) const UNKNOWN: u64 = 0;

/// Whether the value stored at `key` of `part` carries a stamp.
#[inline]
pub(crate) fn stamped(part: Partition, key: &[u8]) -> bool {
    part.carries_revision() && !coverage::is_reserved(key)
}

/// `payload` followed by the stamp of `revision`.
#[inline]
pub(crate) fn stamp(payload: &[u8], revision: u64) -> Vec<u8> {
    let mut value = Vec::with_capacity(payload.len() + STAMP_LEN);
    value.extend_from_slice(payload);
    append_stamp(&mut value, revision);
    value
}

/// Stamp the payload in `value` with `revision`, in place.
#[inline]
pub(crate) fn append_stamp(value: &mut Vec<u8>, revision: u64) {
    value.extend_from_slice(&revision.to_le_bytes());
}

/// The bytes an ordinary write of `payload` at `key` of `part` stores: the
/// payload stamped with `revision`, the write's commit timestamp, when the
/// key carries one, and the payload itself otherwise.
#[inline]
pub(crate) fn stored<'a>(
    part: Partition,
    key: &[u8],
    payload: &'a [u8],
    revision: u64,
) -> Cow<'a, [u8]> {
    if stamped(part, key) {
        Cow::Owned(stamp(payload, revision))
    } else {
        Cow::Borrowed(payload)
    }
}

/// A stamped value's payload and revision, or `None` for a value too short
/// to hold a stamp, which no stamped write produces.
#[inline]
pub(crate) fn split(value: &[u8]) -> Option<(&[u8], u64)> {
    let at = value.len().checked_sub(STAMP_LEN)?;
    let (payload, stamp) = value.split_at(at);
    let stamp: [u8; STAMP_LEN] = stamp.try_into().ok()?;
    Some((payload, u64::from_le_bytes(stamp)))
}

/// A stamped value's payload.
///
/// # Errors
///
/// The value is too short to hold a stamp: the stored bytes are not a
/// record this engine wrote.
#[inline]
pub(crate) fn payload_of(value: &[u8]) -> lsm_tree::Result<&[u8]> {
    split(value)
        .map(|(payload, _)| payload)
        .ok_or(lsm_tree::Error::InvalidHeader(
            "a stored record is shorter than its revision stamp",
        ))
}

/// The payload of `value` read at `key` of `part`, copied out of the tree's
/// buffer: what a point read hands its caller.
///
/// # Errors
///
/// As [`payload_of`].
#[inline]
pub(crate) fn user_bytes(
    part: Partition,
    key: &[u8],
    value: Option<lsm_tree::UserValue>,
) -> lsm_tree::Result<Option<bytes::Bytes>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let payload = if stamped(part, key) {
        payload_of(&value)?
    } else {
        &value
    };
    Ok(Some(bytes::Bytes::copy_from_slice(payload)))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
