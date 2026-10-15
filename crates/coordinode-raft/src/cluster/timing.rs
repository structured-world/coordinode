//! How long a member waits before it stands for election, and how often a
//! leader tells its group it is still there.
//!
//! Timing is a profile, not a constant: a rack of servers wants a failover in
//! well under a second, while a personal fleet of machines that sleep and
//! roam pays more for an unnecessary leadership move than for a short stall.

use std::time::Duration;

/// Election and heartbeat timing of one member.
///
/// A follower that hears nothing from its leader for a random time in
/// `[election_min, election_max)` stands for election; a leader sends a
/// heartbeat every `heartbeat` while its group is active.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ElectionTiming {
    heartbeat: Duration,
    election_min: Duration,
    election_max: Duration,
}

/// A timing that would let a group elect needlessly or never.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TimingError {
    /// A follower would stand for election before a leader's next heartbeat
    /// could reach it, so a healthy group would keep electing.
    #[error(
        "the heartbeat ({heartbeat:?}) must be shorter than half the shortest election \
         timeout ({election_min:?}), or a healthy group keeps electing"
    )]
    HeartbeatTooSlow {
        heartbeat: Duration,
        election_min: Duration,
    },
    /// The election window is empty or inverted: members would time out
    /// together and split their votes.
    #[error(
        "the election timeout range is empty ({election_min:?}..{election_max:?}): \
         members would stand together and split the vote"
    )]
    EmptyElectionWindow {
        election_min: Duration,
        election_max: Duration,
    },
    /// A value openraft cannot carry: it counts milliseconds in a `u64`.
    #[error("a timing of {0:?} is out of range")]
    OutOfRange(Duration),
}

impl ElectionTiming {
    /// Servers in one rack or data centre: a failover within a second.
    pub const RACK: Self = Self {
        heartbeat: Duration::from_millis(150),
        election_min: Duration::from_millis(300),
        election_max: Duration::from_millis(600),
    };

    /// Machines that sleep, wake and change networks: a leader is given tens
    /// of seconds before its group replaces it.
    pub const PERSONAL_FLEET: Self = Self {
        heartbeat: Duration::from_secs(2),
        election_min: Duration::from_secs(20),
        election_max: Duration::from_secs(40),
    };

    /// A timing of its own, checked: the heartbeat comes at least twice
    /// within the shortest election timeout, and the election window is not
    /// empty.
    ///
    /// # Errors
    ///
    /// [`TimingError`] naming the rule the values break.
    pub fn new(
        heartbeat: Duration,
        election_min: Duration,
        election_max: Duration,
    ) -> Result<Self, TimingError> {
        let fits = |value: Duration| u64::try_from(value.as_millis()).is_ok();
        for value in [heartbeat, election_min, election_max] {
            if value.is_zero() || !fits(value) {
                return Err(TimingError::OutOfRange(value));
            }
        }
        // A member that stands last runs up to three times the longest
        // timeout (`for_member`); that has to be representable too.
        if !election_max.checked_mul(3).is_some_and(fits) {
            return Err(TimingError::OutOfRange(election_max));
        }
        if election_min >= election_max {
            return Err(TimingError::EmptyElectionWindow {
                election_min,
                election_max,
            });
        }
        let twice = heartbeat
            .checked_mul(2)
            .ok_or(TimingError::OutOfRange(heartbeat))?;
        if twice > election_min {
            return Err(TimingError::HeartbeatTooSlow {
                heartbeat,
                election_min,
            });
        }
        Ok(Self {
            heartbeat,
            election_min,
            election_max,
        })
    }

    /// The heartbeat interval.
    pub fn heartbeat(&self) -> Duration {
        self.heartbeat
    }

    /// The shortest election timeout.
    pub fn election_min(&self) -> Duration {
        self.election_min
    }

    /// The longest election timeout.
    pub fn election_max(&self) -> Duration {
        self.election_max
    }

    /// This timing with its election window lengthened to at least
    /// `[min, max)`; the heartbeat is kept. Both are well inside the range
    /// `new` admits.
    pub(crate) fn at_least(self, min: Duration, max: Duration) -> Self {
        Self {
            heartbeat: self.heartbeat,
            election_min: self.election_min.max(min),
            election_max: self.election_max.max(max),
        }
    }

    /// Heartbeat, shortest and longest election timeout in milliseconds, as
    /// openraft counts them.
    pub(crate) fn millis(&self) -> (u64, u64, u64) {
        // `new` admits only values whose milliseconds fit in a u64 (three
        // times the longest included), and the two profiles are far inside
        // that, so none of these casts truncates.
        (
            self.heartbeat.as_millis() as u64,
            self.election_min.as_millis() as u64,
            self.election_max.as_millis() as u64,
        )
    }

    /// This timing as a member with `preference` runs it.
    ///
    /// A member that stands last waits from twice to three times the
    /// group's longest election timeout: past every other member's whole
    /// window, so it stands only once each of them has had its chance and
    /// failed, and still within a bound, so it wins when it holds the most
    /// complete log and no other member can.
    pub fn for_member(self, preference: ElectionPreference) -> Self {
        match preference {
            ElectionPreference::Ordinary => self,
            // In range: `new` admits only a longest timeout whose triple
            // still fits, and the two profiles are far inside it.
            ElectionPreference::Last => Self {
                heartbeat: self.heartbeat,
                election_min: self.election_max * 2,
                election_max: self.election_max * 3,
            },
        }
    }
}

impl Default for ElectionTiming {
    fn default() -> Self {
        Self::RACK
    }
}

/// When a member stands for election, relative to the others of its group.
///
/// There is no "never stands": when the preferred members lag and the leader
/// is away, the member holding the most complete log must be able to win,
/// replicate and hand leadership over, or a live majority is left without a
/// leader.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ElectionPreference {
    /// Stands as soon as its election timeout expires.
    #[default]
    Ordinary,
    /// Stands only after every other member of its group could have.
    Last,
}

#[cfg(test)]
mod tests;
