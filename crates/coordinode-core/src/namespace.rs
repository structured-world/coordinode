//! The namespace a piece of data belongs to.
//!
//! A namespace is the tenancy unit of a deployment: its data lives in a set
//! of trees of its own, and keys carry no namespace. What tells a write's
//! namespace is the unit it travels in, which names it before the writes it
//! holds of that namespace.

use alloc::string::String;

/// A namespace incarnation of one deployment. Allocated by the deployment's
/// catalog and never reused within it: a namespace dropped and created again
/// under the same name is a new incarnation with a new id, so nothing bound
/// to the old one (a dictionary handle, a tree, a delayed worker) can reach
/// the new one.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct NamespaceId(u64);

impl NamespaceId {
    /// The id from its number.
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// The id's number.
    #[must_use]
    pub const fn as_raw(self) -> u64 {
        self.0
    }

    /// The name of the namespace's directory in a store: fixed width, so a
    /// listing sorts by id.
    #[must_use]
    pub fn dir_name(self) -> String {
        alloc::format!("{:016x}", self.0)
    }

    /// The id a namespace directory is named for, `None` for a name no
    /// namespace directory has.
    #[must_use]
    pub fn from_dir_name(name: &str) -> Option<Self> {
        (name.len() == 16 && name.bytes().all(|b| b.is_ascii_hexdigit()))
            .then(|| u64::from_str_radix(name, 16).ok().map(Self))
            .flatten()
    }
}

impl core::fmt::Display for NamespaceId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "namespace {}", self.0)
    }
}
