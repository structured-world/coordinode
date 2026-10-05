//! Server-wide defaults for the per-statement consistency settings.
//!
//! A statement that names a read concern, read preference or write concern
//! uses it. One that leaves a setting out takes its session's value, and a
//! session that set nothing takes these. They come from the config file keys
//! `default_read_concern`, `default_read_preference` and
//! `default_write_concern`; a key left out keeps the built-in value: a local
//! read from the leader, and a write acknowledged by a majority with the
//! journal fsynced.

use coordinode_core::txn::read_concern::ReadConcernLevel;
use coordinode_core::txn::write_concern::{Journal, WriteAck, WriteConcern};
use coordinode_raft::read_fence::ReadPreference;
use serde::Deserialize;

/// The resolved defaults a statement falls back to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StatementDefaults {
    /// What a read may observe.
    pub read_concern: ReadConcernLevel,
    /// Which member may serve a read.
    pub read_preference: ReadPreference,
    /// When a write is acknowledged.
    pub write_concern: WriteConcern,
}

/// `default_read_concern` as written in the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadConcernSetting {
    /// Whatever the serving member has applied.
    Local,
    /// Only what a majority has committed.
    Majority,
    /// The leader confirms its lease before reading.
    Linearizable,
    /// A pinned MVCC snapshot.
    Snapshot,
}

impl From<ReadConcernSetting> for ReadConcernLevel {
    fn from(s: ReadConcernSetting) -> Self {
        match s {
            ReadConcernSetting::Local => Self::Local,
            ReadConcernSetting::Majority => Self::Majority,
            ReadConcernSetting::Linearizable => Self::Linearizable,
            ReadConcernSetting::Snapshot => Self::Snapshot,
        }
    }
}

/// `default_read_preference` as written in the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadPreferenceSetting {
    /// The leader only.
    Primary,
    /// The leader, or a follower while there is none.
    PrimaryPreferred,
    /// Followers only.
    Secondary,
    /// A follower, or the leader while there is none.
    SecondaryPreferred,
    /// The member the request reached.
    Nearest,
}

impl From<ReadPreferenceSetting> for ReadPreference {
    fn from(s: ReadPreferenceSetting) -> Self {
        match s {
            ReadPreferenceSetting::Primary => Self::Primary,
            ReadPreferenceSetting::PrimaryPreferred => Self::PrimaryPreferred,
            ReadPreferenceSetting::Secondary => Self::Secondary,
            ReadPreferenceSetting::SecondaryPreferred => Self::SecondaryPreferred,
            ReadPreferenceSetting::Nearest => Self::Nearest,
        }
    }
}

/// `default_write_concern`, checked when the file is read: a combination the
/// engine cannot honour fails the start instead of every write later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(try_from = "WriteConcernFile")]
pub struct WriteConcernSetting(pub WriteConcern);

/// `default_write_concern` as written in the file:
/// `{ w: majority | <members>, journal: journal | cache | memory, timeout_ms: <ms> }`.
#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
struct WriteConcernFile {
    w: AckFile,
    journal: JournalFile,
    timeout_ms: u32,
}

impl Default for WriteConcernFile {
    fn default() -> Self {
        Self {
            w: AckFile::Mode(AckMode::Majority),
            journal: JournalFile::Journal,
            timeout_ms: 0,
        }
    }
}

/// `w`: a member count, or `majority`.
#[derive(Deserialize)]
#[serde(untagged)]
enum AckFile {
    Members(u32),
    Mode(AckMode),
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum AckMode {
    Majority,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum JournalFile {
    Journal,
    Cache,
    Memory,
}

impl TryFrom<WriteConcernFile> for WriteConcernSetting {
    type Error = String;

    fn try_from(file: WriteConcernFile) -> Result<Self, String> {
        let concern = WriteConcern {
            w: match file.w {
                AckFile::Members(n) => WriteAck::Acks(n),
                AckFile::Mode(AckMode::Majority) => WriteAck::Majority,
            },
            journal: match file.journal {
                JournalFile::Journal => Journal::Journal,
                JournalFile::Cache => Journal::Cache,
                JournalFile::Memory => Journal::Memory,
            },
            timeout_ms: file.timeout_ms,
        };
        // The member count is checked per write against the group as it is
        // then; only the combination is fixed here.
        concern
            .validate(None)
            .map_err(|e| format!("default_write_concern: {e}"))?;
        Ok(Self(concern))
    }
}
