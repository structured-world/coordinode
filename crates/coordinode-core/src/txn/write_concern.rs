//! Write concern: when a write is acknowledged to the caller.
//!
//! A write concern is two independent parameters, as in MongoDB:
//!
//! - [`WriteAck`] (`w`): how many members must hold the write before the
//!   caller is answered. A number is always a count of members, the leader
//!   included; [`WriteAck::Majority`] is a rule, not a count.
//! - [`Journal`] (`j`): the state each of those members holds the write in
//!   when it counts: fsynced in its journal, in RAM plus the NVMe write cache,
//!   or in RAM alone.
//!
//! Neither axis expresses the other, and nothing is silently rewritten
//! between them. `w: 0` is answered at once whatever `j` says.
//!
//! A write concern decides when the caller is answered, never whether the
//! write is replicated: every write enters the group's log through the leader
//! and is applied on every member. The volatile journal levels ([`Journal::Memory`],
//! [`Journal::Cache`]) keep the write in a separate overlay until the drain
//! proposes it; a crash before the drain loses it, which is the contract the
//! caller accepted.

/// How many members must hold the write before the caller is answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum WriteAck {
    /// Exactly this many members, the leader included. `0` is fire-and-forget:
    /// the write still goes through the leader and the log, the caller is not
    /// told whether it committed. `1` is the leader alone.
    Acks(u32),
    /// A majority of the group's current membership. The default.
    #[default]
    Majority,
}

impl WriteAck {
    /// Fire-and-forget: no acknowledgement is promised.
    pub const NONE: Self = Self::Acks(0);
    /// The leader alone.
    pub const LEADER: Self = Self::Acks(1);

    /// Whether the caller waits for nothing at all.
    pub fn is_fire_and_forget(&self) -> bool {
        matches!(self, Self::Acks(0))
    }

    /// Whether more than one member must hold the write.
    pub fn needs_replication(&self) -> bool {
        match self {
            Self::Acks(n) => *n > 1,
            Self::Majority => true,
        }
    }
}

/// The state each acknowledging member holds the write in when it counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Journal {
    /// Fsynced in the member's journal (MongoDB `j: true`). The default.
    #[default]
    Journal,
    /// In RAM plus the member's NVMe write cache, drained to the log later.
    /// Survives a process crash, not a power failure before the drain.
    Cache,
    /// In RAM only, drained to the log later (MongoDB `j: false`). Lost on
    /// process crash before the drain.
    Memory,
}

impl Journal {
    /// Whether the write may be answered before it is in the log.
    pub fn is_volatile(&self) -> bool {
        !matches!(self, Self::Journal)
    }
}

/// Why a write concern is not accepted.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WriteConcernError {
    /// `w` names more members than the group has.
    #[error("write concern asks for {requested} members, the group has {members}")]
    TooManyAcks {
        /// The count the caller asked for.
        requested: u32,
        /// The group's member count.
        members: u32,
    },
    /// A volatile journal level with more than one acknowledging member: the
    /// log store acknowledges an append only after the batch fsync, so no
    /// member but the leader can hold a write "in memory" for the caller.
    #[error(
        "write concern {0} is not supported: a journal level below `journal` \
         is honoured for w:1 only"
    )]
    VolatileReplication(WriteConcern),
}

/// Write concern configuration for a write statement or a session default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WriteConcern {
    /// How many members must hold the write.
    pub w: WriteAck,
    /// The state each of them holds it in.
    pub journal: Journal,
    /// Time in milliseconds to wait for `w`; `0` = no timeout. On timeout the
    /// call fails, but the write is NOT rolled back: it may still commit.
    pub timeout_ms: u32,
}

impl WriteConcern {
    /// `w: 0`: fire-and-forget, no acknowledgement.
    pub fn w0() -> Self {
        Self {
            w: WriteAck::NONE,
            ..Self::default()
        }
    }

    /// `w: 1, j: journal`: the leader holds the write fsynced.
    pub fn w1() -> Self {
        Self {
            w: WriteAck::LEADER,
            ..Self::default()
        }
    }

    /// `w: N, j: journal`: N members hold the write fsynced.
    pub fn acks(n: u32) -> Self {
        Self {
            w: WriteAck::Acks(n),
            ..Self::default()
        }
    }

    /// `w: majority, j: journal`: the default.
    pub fn majority() -> Self {
        Self::default()
    }

    /// `w: 1, j: memory`: the leader holds the write in RAM, drained later.
    pub fn memory() -> Self {
        Self {
            w: WriteAck::LEADER,
            journal: Journal::Memory,
            timeout_ms: 0,
        }
    }

    /// `w: 1, j: cache`: the leader holds the write in RAM and its NVMe write
    /// cache, drained later.
    pub fn cache() -> Self {
        Self {
            w: WriteAck::LEADER,
            journal: Journal::Cache,
            timeout_ms: 0,
        }
    }

    /// `w: majority, j: journal` with a timeout.
    pub fn majority_with_timeout(timeout_ms: u32) -> Self {
        Self {
            timeout_ms,
            ..Self::default()
        }
    }

    /// Whether the write is answered before it is in the log and kept in the
    /// volatile overlay until the drain: a volatile journal level, and a caller
    /// that waits for something (`w: 0` takes the ordinary log path and is
    /// simply not waited for).
    pub fn is_volatile(&self) -> bool {
        self.journal.is_volatile() && !self.w.is_fire_and_forget()
    }

    /// Whether this concern is safe for a causal session: the write must be
    /// committed by a majority and fsynced before the caller learns its
    /// position, or the position is a promise that a failover can break.
    pub fn is_causal_safe(&self) -> bool {
        matches!(self.w, WriteAck::Majority) && matches!(self.journal, Journal::Journal)
    }

    /// Whether the write can still be lost after the caller is answered.
    pub fn can_rollback(&self) -> bool {
        !self.is_causal_safe()
    }

    /// Check the concern against the group.
    ///
    /// `members` is the group's member count when known; `None` skips the
    /// count check (a standalone or embedded caller that has no group). A
    /// volatile journal level is honoured for `w: 1` only, see
    /// [`WriteConcernError::VolatileReplication`].
    pub fn validate(&self, members: Option<u32>) -> Result<(), WriteConcernError> {
        if let (WriteAck::Acks(n), Some(members)) = (self.w, members) {
            if n > members {
                return Err(WriteConcernError::TooManyAcks {
                    requested: n,
                    members,
                });
            }
        }
        if self.journal.is_volatile() && self.w.needs_replication() {
            return Err(WriteConcernError::VolatileReplication(*self));
        }
        Ok(())
    }

    /// Check the concern for a causal session, with the message the caller
    /// sees.
    pub fn validate_for_causal_session(&self) -> Result<(), &'static str> {
        if !self.is_causal_safe() {
            return Err("causal sessions require w:majority with j:journal; \
                 use a non-causal session for weaker write concerns");
        }
        Ok(())
    }
}

impl core::fmt::Display for WriteAck {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Acks(n) => write!(f, "w:{n}"),
            Self::Majority => write!(f, "w:majority"),
        }
    }
}

impl core::fmt::Display for Journal {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Journal => write!(f, "j:journal"),
            Self::Cache => write!(f, "j:cache"),
            Self::Memory => write!(f, "j:memory"),
        }
    }
}

impl core::fmt::Display for WriteConcern {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{},{}", self.w, self.journal)?;
        if self.timeout_ms > 0 {
            write!(f, ",wtimeout:{}", self.timeout_ms)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
