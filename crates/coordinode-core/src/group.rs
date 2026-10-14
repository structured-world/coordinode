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

/// The identity of one deployment: chosen at random when its first group
/// is formed and kept for its life, through restarts, credential rotation
/// and growth. Two deployments never share one, so a member id, group id or
/// node identity valid in one grants nothing in the other.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct DeploymentId(pub [u8; 16]);

/// Digits of a deployment identity's text form.
const DEPLOYMENT_HEX_LEN: usize = 32;

impl DeploymentId {
    /// The raw bytes, as recorded and carried on the wire.
    pub fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    /// The identity `bytes` encode, `None` when they are not 16 bytes.
    pub fn from_slice(bytes: &[u8]) -> Option<Self> {
        bytes.try_into().ok().map(Self)
    }
}

impl fmt::Display for DeploymentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// Schema-partition key of the deployment's identity record.
pub const DEPLOYMENT_KEY: &[u8] = b"meta:deployment";

/// Schema-partition prefix of retired node identities; the key ends with
/// the identity's canonical encoding.
pub const RETIRED_IDENTITY_KEY_PREFIX: &[u8] = b"meta:retired_identity:";

/// The key recording that `identity` (canonically encoded) was retired.
pub fn retired_identity_key(identity: &[u8]) -> alloc::vec::Vec<u8> {
    let mut key =
        alloc::vec::Vec::with_capacity(RETIRED_IDENTITY_KEY_PREFIX.len() + identity.len());
    key.extend_from_slice(RETIRED_IDENTITY_KEY_PREFIX);
    key.extend_from_slice(identity);
    key
}

/// Why text is not a deployment identity.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("a deployment identity is 32 hexadecimal digits")]
pub struct DeploymentIdError;

impl core::str::FromStr for DeploymentId {
    type Err = DeploymentIdError;

    /// Parse the 32 hexadecimal digits [`Display`](fmt::Display) writes.
    ///
    /// # Examples
    ///
    /// ```
    /// use coordinode_core::group::DeploymentId;
    /// let id = DeploymentId([7; 16]);
    /// assert_eq!(id.to_string().parse::<DeploymentId>(), Ok(id));
    /// assert!("07".parse::<DeploymentId>().is_err());
    /// ```
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let digits = text.as_bytes();
        // from_str_radix takes a leading sign, which is not a digit here.
        if digits.len() != DEPLOYMENT_HEX_LEN || !digits.iter().all(u8::is_ascii_hexdigit) {
            return Err(DeploymentIdError);
        }
        let mut bytes = [0u8; 16];
        // The length is even, so no digit is left over.
        let (pairs, _) = digits.as_chunks::<2>();
        for (byte, pair) in bytes.iter_mut().zip(pairs) {
            let pair = core::str::from_utf8(pair).map_err(|_| DeploymentIdError)?;
            *byte = u8::from_str_radix(pair, 16).map_err(|_| DeploymentIdError)?;
        }
        Ok(Self(bytes))
    }
}
