//! Consensus group identity.
//!
//! A server hosts replicas of several consensus groups: data groups and the
//! coordinator's metadata group. Every inter-node consensus message names its
//! group, and the receiving server dispatches it to its replica of that
//! group. The identity is logical: it does not name a server, and moving a
//! group's replicas between servers does not change it.

use core::fmt;

/// The identity of one consensus group.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Default,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct GroupId(pub u64);

impl GroupId {
    /// The group a deployment is formed with: the only group of a
    /// single-group deployment.
    pub const FORMING: Self = Self(0);

    /// The raw value carried on the wire.
    ///
    /// # Examples
    ///
    /// ```
    /// use coordinode_core::group::GroupId;
    /// assert_eq!(GroupId(7).raw(), 7);
    /// assert_eq!(GroupId::default(), GroupId::FORMING);
    /// ```
    pub fn raw(self) -> u64 {
        self.0
    }
}

impl fmt::Display for GroupId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "group {}", self.0)
    }
}
