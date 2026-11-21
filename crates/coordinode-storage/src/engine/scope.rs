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

pub use coordinode_core::namespace::NamespaceId;

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

    /// Whether `part` has a tree in a set of this kind: the consensus log,
    /// the consumer registry and the deployment's records are the
    /// deployment's alone ([`Partition::is_deployment`]), every other
    /// partition is a namespace's alone.
    #[must_use]
    pub fn holds(self, part: Partition) -> bool {
        match self {
            Self::System => part.is_deployment(),
            Self::Namespace(_) => !part.is_deployment(),
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

/// The namespace a fresh deployment's single store creates first, for tests
/// that build units and tree paths of the store they open before holding
/// its engine. Not built into a release: no serving path names a namespace
/// it was not bound to.
#[cfg(any(test, feature = "test-faults"))]
pub const STORE_NAMESPACE: NamespaceId = NamespaceId::from_raw(1);

/// The unit of `mutations` in [`STORE_NAMESPACE`], the marker naming it
/// first.
#[cfg(any(test, feature = "test-faults"))]
pub fn store_unit(
    mutations: impl IntoIterator<Item = coordinode_core::txn::proposal::Mutation>,
) -> Vec<coordinode_core::txn::proposal::Mutation> {
    coordinode_core::txn::proposal::in_namespace(STORE_NAMESPACE, mutations)
}

/// The tree a fresh single store keeps `part` in: the deployment's own
/// partitions in the system set, the rest in [`STORE_NAMESPACE`]. Its
/// [`TreeId::dir`] is where the tree lies under each endpoint.
#[cfg(any(test, feature = "test-faults"))]
#[must_use]
pub const fn store_tree(part: Partition) -> TreeId {
    if part.is_deployment() {
        TreeId::system(part)
    } else {
        TreeId::namespace(STORE_NAMESPACE, part)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
