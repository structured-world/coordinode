//! Which set of trees a tree belongs to.
//!
//! A deployment keeps its own state (the consensus log, the consumer
//! registry, the catalog of namespaces and the records that hold for the
//! whole deployment) in one system set of trees, and every namespace's data
//! in a set of its own. Keys carry no namespace: the tree a key lives in is
//! what says whose it is, so a tree is addressed by its set and its
//! partition together.

use std::path::PathBuf;

use crate::engine::partition::Partition;

/// The directory, relative to each endpoint, holding the system set.
pub const SYSTEM_DIR: &str = "system";

/// The directory, relative to each endpoint, holding one directory per
/// namespace.
pub const NAMESPACES_DIR: &str = "ns";

/// A namespace incarnation of one deployment. Allocated by the deployment's
/// catalog and never reused within it: a namespace dropped and created again
/// under the same name is a new incarnation with a new id, so nothing bound
/// to the old one (a dictionary handle, a tree, a delayed worker) can reach
/// the new one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
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

    /// The name of the namespace's directory under [`NAMESPACES_DIR`]:
    /// fixed width, so a listing sorts by id.
    #[must_use]
    pub fn dir_name(self) -> String {
        format!("{:016x}", self.0)
    }

    /// The id a directory under [`NAMESPACES_DIR`] is named for, `None` for
    /// a name no namespace directory has.
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

/// The set of trees a tree belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Scope {
    /// The deployment's own state.
    System,
    /// One namespace's data.
    Namespace(NamespaceId),
}

impl Scope {
    /// The directory of the set, relative to each endpoint.
    #[must_use]
    pub fn dir(self) -> PathBuf {
        match self {
            Self::System => PathBuf::from(SYSTEM_DIR),
            Self::Namespace(ns) => PathBuf::from(NAMESPACES_DIR).join(ns.dir_name()),
        }
    }

    /// Whether `part` has a tree in a set of this kind: the consensus log
    /// and the consumer registry are the deployment's alone, every data
    /// partition is a namespace's alone, and each set has a Schema tree of
    /// its own.
    #[must_use]
    pub fn holds(self, part: Partition) -> bool {
        match self {
            Self::System => part.is_system() || part == Partition::Schema,
            Self::Namespace(_) => !part.is_system(),
        }
    }

    /// The partitions with a tree in a set of this kind, in
    /// [`Partition::all`] order.
    pub fn partitions(self) -> impl Iterator<Item = Partition> {
        Partition::all()
            .iter()
            .copied()
            .filter(move |&part| self.holds(part))
    }
}

/// One tree of a deployment: a partition in a set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TreeId {
    /// The set.
    pub scope: Scope,
    /// The partition.
    pub part: Partition,
}

impl TreeId {
    /// `part` in the system set.
    #[must_use]
    pub const fn system(part: Partition) -> Self {
        Self {
            scope: Scope::System,
            part,
        }
    }

    /// `part` in namespace `ns`.
    #[must_use]
    pub const fn namespace(ns: NamespaceId, part: Partition) -> Self {
        Self {
            scope: Scope::Namespace(ns),
            part,
        }
    }

    /// The tree's directory, relative to each endpoint.
    #[must_use]
    pub fn dir(self) -> PathBuf {
        self.scope.dir().join(self.part.name())
    }

    /// The tree's name as a copy of the store carries it: its directory with
    /// `/` between the parts, whatever the platform's separator.
    #[must_use]
    pub fn name(self) -> String {
        match self.scope {
            Scope::System => format!("{SYSTEM_DIR}/{}", self.part.name()),
            Scope::Namespace(ns) => {
                format!("{NAMESPACES_DIR}/{}/{}", ns.dir_name(), self.part.name())
            }
        }
    }

    /// The tree a name given by [`Self::name`] stands for; `None` for any
    /// other name, a bare partition name included, and for a partition the
    /// named set holds no tree of.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        let mut parts = name.split('/');
        let scope = match parts.next()? {
            SYSTEM_DIR => Scope::System,
            NAMESPACES_DIR => Scope::Namespace(NamespaceId::from_dir_name(parts.next()?)?),
            _ => return None,
        };
        let part_name = parts.next()?;
        if parts.next().is_some() {
            return None;
        }
        let part = Partition::all()
            .iter()
            .copied()
            .find(|part| part.name() == part_name)?;
        scope.holds(part).then_some(Self { scope, part })
    }
}

impl core::fmt::Display for TreeId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.scope {
            Scope::System => write!(f, "system {}", self.part.name()),
            Scope::Namespace(ns) => write!(f, "{ns} {}", self.part.name()),
        }
    }
}

/// The namespace an engine's data partitions open in while it holds one set
/// of them.
pub const STORE_NAMESPACE: NamespaceId = NamespaceId(1);

/// The tree an engine holding one data set keeps `part` in: the deployment's
/// own partitions in the system set, the rest in [`STORE_NAMESPACE`]. Its
/// [`TreeId::dir`] is where the tree lies under each endpoint.
#[must_use]
pub const fn store_tree(part: Partition) -> TreeId {
    if part.is_system() {
        TreeId::system(part)
    } else {
        TreeId::namespace(STORE_NAMESPACE, part)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
