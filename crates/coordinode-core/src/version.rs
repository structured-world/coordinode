//! The version a member runs, for matching the members of one group.
//!
//! Members of one consensus group never interoperate across versions: two
//! members match only when both parts of their [`VersionPair`] are equal.
//! The [`Handshake`] that compares them is the one record whose encoding
//! never changes, so that any two releases can still tell each other apart.

use alloc::string::String;
use alloc::vec::Vec;

use crate::group::GroupId;

/// The engine format version this build reads and writes.
///
/// Raised by a release that changes anything one member sends another (log
/// entry kinds and encoding, snapshot format, inter-node messages) or
/// anything in the directory (index key layouts, partition layout, persisted
/// consensus state). A directory of another format is refused, so a build
/// never reads keys laid out by another.
pub const ENGINE_FORMAT_VERSION: u32 = 2;

/// The engine format version this process runs: [`ENGINE_FORMAT_VERSION`],
/// raised by `COORDINODE_TEST_ENGINE_FORMAT_BUMP` in builds with the
/// `test-format-bump` feature, so one test suite can run two versions.
pub fn engine_format_version() -> u32 {
    ENGINE_FORMAT_VERSION + test_format_bump()
}

#[cfg(feature = "test-format-bump")]
fn test_format_bump() -> u32 {
    // no-std: the bump is a test-build knob; a no-std build has no environment.
    std::env::var("COORDINODE_TEST_ENGINE_FORMAT_BUMP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

#[cfg(not(feature = "test-format-bump"))]
const fn test_format_bump() -> u32 {
    0
}

/// What a member runs, for matching: equal pairs interoperate, any
/// difference in either part does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct VersionPair {
    /// The engine's format version, compiled in.
    pub engine: u32,
    /// The embedding application's format epoch, opaque to the engine and
    /// supplied where a member is opened. A server runs zero.
    pub host_epoch: u64,
}

impl VersionPair {
    /// The pair of this process for a host at `host_epoch`.
    pub fn current(host_epoch: u64) -> Self {
        Self {
            engine: engine_format_version(),
            host_epoch,
        }
    }

    /// Whether this pair is a later version than `other`. Versions only move
    /// forward: a release that changes the engine format is later whatever
    /// the host's epoch, and at one engine format the higher host epoch is.
    pub fn is_newer_than(&self, other: &Self) -> bool {
        (self.engine, self.host_epoch) > (other.engine, other.host_epoch)
    }
}

impl core::fmt::Display for VersionPair {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "engine format {} / host epoch {}",
            self.engine, self.host_epoch
        )
    }
}

/// The pair a group runs, as recorded by its log. Records are numbered in
/// log order from 1, the same on every member, so that of two reports the
/// later one wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RecordedPair {
    /// The pair the group runs from this record on.
    pub pair: VersionPair,
    /// The record's number.
    pub seq: u64,
}

impl RecordedPair {
    /// Whether this record was made after `other`.
    pub fn is_later_than(&self, other: &Self) -> bool {
        self.seq > other.seq
    }
}

/// Why a member is read-only, as every refusal names it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mismatch {
    /// The pair this member runs.
    pub own: VersionPair,
    /// The pair its group runs.
    pub group: VersionPair,
    /// The group moved past this member's pair (it is behind); otherwise
    /// this member runs a pair the group has not moved to yet (ahead).
    pub behind: bool,
    /// The group's leader as far as this member knows: id and address.
    pub leader: Option<(u64, String)>,
    /// The commit timestamp of the last entry this member applied, which
    /// every read it serves is as of.
    pub as_of: u64,
}

impl core::fmt::Display for Mismatch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "this member runs {}, its group runs {} ({}); it is read-only and serves reads \
             as of commit timestamp {}",
            self.own,
            self.group,
            if self.behind {
                "the group has moved on: update this member"
            } else {
                "the group has not moved yet"
            },
            self.as_of
        )?;
        if let Some((id, addr)) = &self.leader {
            write!(f, "; the leader is node {id} at {addr}")?;
        }
        Ok(())
    }
}

/// Schema-partition prefix of the group's pair records; the key ends with
/// the record number, big-endian, so the last record sorts last.
pub const GROUP_PAIR_KEY_PREFIX: &[u8] = b"meta:group_pair:";

/// The key of group pair record `seq`.
pub fn group_pair_key(seq: u64) -> Vec<u8> {
    let mut key = Vec::with_capacity(GROUP_PAIR_KEY_PREFIX.len() + 8);
    key.extend_from_slice(GROUP_PAIR_KEY_PREFIX);
    key.extend_from_slice(&seq.to_be_bytes());
    key
}

/// The record number a group pair key names.
pub fn decode_group_pair_key(key: &[u8]) -> Option<u64> {
    let seq = key.strip_prefix(GROUP_PAIR_KEY_PREFIX)?;
    Some(u64::from_be_bytes(seq.try_into().ok()?))
}

/// A group pair record's value: `engine u32`, `host_epoch u64`, little-endian.
pub fn encode_pair(pair: VersionPair) -> Vec<u8> {
    let mut out = Vec::with_capacity(12);
    put_pair(&mut out, pair);
    out
}

/// The pair a group pair record holds.
pub fn decode_pair(bytes: &[u8]) -> Option<VersionPair> {
    let mut r = Reader(bytes);
    let pair = r.pair().ok()?;
    r.0.is_empty().then_some(pair)
}

/// The first exchange between two members, and the only one when they do
/// not match. Its encoding ([`Handshake::encode`]) is frozen for the life of
/// the product.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Handshake {
    /// The sender's node id.
    pub node_id: u64,
    /// The consensus group the sender speaks for.
    pub group_id: GroupId,
    /// The pair the sender runs.
    pub pair: VersionPair,
    /// The pair its group runs, as far as the sender knows.
    pub group_pair: Option<RecordedPair>,
    /// The group's leader as far as the sender knows: its node id and the
    /// address members reach it at.
    pub leader: Option<(u64, String)>,
}

/// Leading bytes of an encoded [`Handshake`].
const HANDSHAKE_MAGIC: [u8; 4] = *b"CNVH";
/// Longest leader address an encoded handshake carries.
const MAX_LEADER_ADDR: usize = 1024;

/// Why bytes are not a [`Handshake`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum HandshakeError {
    /// The bytes end before the record does.
    #[error("handshake truncated")]
    Truncated,
    /// The bytes do not start with the handshake magic.
    #[error("not a version handshake")]
    BadMagic,
    /// A presence flag is neither 0 nor 1.
    #[error("handshake flag out of range")]
    BadFlag,
    /// The leader address is too long or not UTF-8.
    #[error("handshake leader address invalid")]
    BadAddress,
    /// Bytes follow the record.
    #[error("handshake has trailing bytes")]
    Trailing,
}

impl Handshake {
    /// The frozen encoding: magic `CNVH`, then little-endian `node_id u64`,
    /// `group_id u64`, `engine u32`, `host_epoch u64`; a flag byte and, when
    /// 1, the group's `engine u32`, `host_epoch u64` and record number `seq
    /// u64`; a flag byte and, when 1, the leader's `node_id u64`, address
    /// length `u16` and UTF-8 address.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(64);
        out.extend_from_slice(&HANDSHAKE_MAGIC);
        out.extend_from_slice(&self.node_id.to_le_bytes());
        out.extend_from_slice(&self.group_id.raw().to_le_bytes());
        put_pair(&mut out, self.pair);
        match &self.group_pair {
            Some(recorded) => {
                out.push(1);
                put_pair(&mut out, recorded.pair);
                out.extend_from_slice(&recorded.seq.to_le_bytes());
            }
            None => out.push(0),
        }
        match &self.leader {
            Some((id, addr)) => {
                // A longer address is not sent rather than cut mid-character.
                let addr = if addr.len() <= MAX_LEADER_ADDR {
                    addr.as_str()
                } else {
                    ""
                };
                out.push(1);
                out.extend_from_slice(&id.to_le_bytes());
                // At most MAX_LEADER_ADDR, which fits a u16.
                out.extend_from_slice(&(addr.len() as u16).to_le_bytes());
                out.extend_from_slice(addr.as_bytes());
            }
            None => out.push(0),
        }
        out
    }

    /// Decode a record made by [`Handshake::encode`] of any release.
    pub fn decode(bytes: &[u8]) -> Result<Self, HandshakeError> {
        let mut r = Reader(bytes);
        if r.take(4)? != HANDSHAKE_MAGIC {
            return Err(HandshakeError::BadMagic);
        }
        let node_id = r.u64()?;
        let group_id = GroupId(r.u64()?);
        let pair = r.pair()?;
        let group_pair = if r.flag()? {
            let pair = r.pair()?;
            Some(RecordedPair {
                pair,
                seq: r.u64()?,
            })
        } else {
            None
        };
        let leader = if r.flag()? {
            let id = r.u64()?;
            let len = usize::from(u16::from_le_bytes(r.array()?));
            if len > MAX_LEADER_ADDR {
                return Err(HandshakeError::BadAddress);
            }
            let addr =
                core::str::from_utf8(r.take(len)?).map_err(|_| HandshakeError::BadAddress)?;
            Some((id, String::from(addr)))
        } else {
            None
        };
        if !r.0.is_empty() {
            return Err(HandshakeError::Trailing);
        }
        Ok(Self {
            node_id,
            group_id,
            pair,
            group_pair,
            leader,
        })
    }
}

fn put_pair(out: &mut Vec<u8>, pair: VersionPair) {
    out.extend_from_slice(&pair.engine.to_le_bytes());
    out.extend_from_slice(&pair.host_epoch.to_le_bytes());
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], HandshakeError> {
        if self.0.len() < n {
            return Err(HandshakeError::Truncated);
        }
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        Ok(head)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], HandshakeError> {
        let mut a = [0u8; N];
        a.copy_from_slice(self.take(N)?);
        Ok(a)
    }

    fn u64(&mut self) -> Result<u64, HandshakeError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn pair(&mut self) -> Result<VersionPair, HandshakeError> {
        Ok(VersionPair {
            engine: u32::from_le_bytes(self.array()?),
            host_epoch: self.u64()?,
        })
    }

    fn flag(&mut self) -> Result<bool, HandshakeError> {
        match self.take(1)?[0] {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(HandshakeError::BadFlag),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
