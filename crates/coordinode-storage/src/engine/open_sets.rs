//! The trees an engine has open, by set.
//!
//! The system set opens with the engine and stays open for its life. A
//! namespace's set is added when the namespace's trees are opened and
//! removed when it is dropped, so everything that walks the trees (the
//! flush and compaction monitors, the capacity scanner, a checkpoint) lists
//! them afresh rather than keeping the list it saw at open.
//!
//! A reader or writer of one namespace holds that namespace's set
//! ([`SetTrees`]) and finds a tree in it by partition, an index into a fixed
//! array: the map of sets is consulted once per handle, never per key.

use std::collections::BTreeMap;
use std::sync::Arc;

use lsm_tree::AnyTree;
use parking_lot::RwLock;

use crate::engine::partition::Partition;
use crate::engine::scope::{NamespaceId, Scope, TreeId};

/// One set's open trees.
pub struct SetTrees {
    scope: Scope,
    trees: [Option<AnyTree>; Partition::COUNT],
    /// Which local installation holds each index generation in the set's
    /// index tree; every read and write of a generation key goes through
    /// it. `None` for a set without an index tree.
    installations: Option<crate::engine::installation::Installations>,
    /// Per node key of the set, the property deltas written since its last
    /// whole write: a read of a key with pending deltas folds every one of
    /// them, so a writer bounds the run by writing the record whole once it
    /// is long enough. A hint for the choice of write form only, so
    /// process-local and bounded: a lost or reset count costs one early
    /// whole write. Kept per set: the same key in two namespaces is two
    /// records. no-std: spin::Mutex
    node_delta_runs: parking_lot::Mutex<rustc_hash::FxHashMap<Vec<u8>, u32>>,
}

impl SetTrees {
    /// A set of `scope` with no tree open yet.
    #[must_use]
    pub(crate) fn new(scope: Scope) -> Self {
        Self {
            scope,
            trees: Default::default(),
            installations: None,
            node_delta_runs: parking_lot::Mutex::new(rustc_hash::FxHashMap::default()),
        }
    }

    /// Count one more property delta written to `node_key` and return the
    /// run it makes since the key's last whole write.
    pub(crate) fn note_node_delta(&self, node_key: &[u8]) -> u32 {
        /// Keys tracked at once; past it the table starts over, which only
        /// makes some runs look shorter than they are.
        const MAX_TRACKED: usize = 1 << 16;
        let mut runs = self.node_delta_runs.lock();
        if runs.len() >= MAX_TRACKED && !runs.contains_key(node_key) {
            runs.clear();
        }
        let run = runs.entry(node_key.to_vec()).or_insert(0);
        // Bounded far below u32::MAX: a writer ends the run with a whole
        // write once it reaches its limit.
        *run += 1;
        *run
    }

    /// The property deltas written to `node_key` since its last whole write.
    pub(crate) fn node_delta_run(&self, node_key: &[u8]) -> u32 {
        self.node_delta_runs
            .lock()
            .get(node_key)
            .copied()
            .unwrap_or(0)
    }

    /// `node_key` was written whole: its run of deltas is over.
    pub(crate) fn note_node_whole_write(&self, node_key: &[u8]) {
        self.node_delta_runs.lock().remove(node_key);
    }

    /// The set with its index tree's installation catalog.
    #[must_use]
    pub(crate) fn with_installations(
        mut self,
        installations: crate::engine::installation::Installations,
    ) -> Self {
        self.installations = Some(installations);
        self
    }

    /// The index tree's installation catalog, when `part` is the index
    /// partition and the set has one.
    #[inline]
    #[must_use]
    pub(crate) fn installations_of(
        &self,
        part: Partition,
    ) -> Option<&crate::engine::installation::Installations> {
        if part == Partition::Idx {
            self.installations.as_ref()
        } else {
            None
        }
    }

    /// The index tree's installation catalog, when the set has one.
    #[inline]
    #[must_use]
    pub(crate) fn installations(&self) -> Option<&crate::engine::installation::Installations> {
        self.installations.as_ref()
    }

    /// The set.
    #[must_use]
    pub fn scope(&self) -> Scope {
        self.scope
    }

    /// Put `tree` in as the set's `part`. A partition the set holds no tree
    /// of is a programming error the debug build catches.
    pub(crate) fn insert(&mut self, part: Partition, tree: AnyTree) {
        debug_assert!(
            self.scope.holds(part),
            "{:?} holds no {} tree",
            self.scope,
            part.name()
        );
        self.trees[part.position()] = Some(tree);
    }

    /// The set's `part` tree, when it is open.
    #[inline]
    #[must_use]
    pub fn get(&self, part: Partition) -> Option<&AnyTree> {
        self.trees[part.position()].as_ref()
    }

    /// Every open tree of the set, in [`Partition::all`] order.
    pub fn iter(&self) -> impl Iterator<Item = (TreeId, &AnyTree)> {
        let scope = self.scope;
        Partition::all()
            .iter()
            .filter_map(move |&part| self.get(part).map(|tree| (TreeId { scope, part }, tree)))
    }

    /// The number of open trees.
    #[must_use]
    pub fn len(&self) -> usize {
        self.trees.iter().flatten().count()
    }

    /// Whether no tree is open.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Every set an engine has open; see the [module docs](self).
pub struct OpenSets {
    system: Arc<SetTrees>,
    namespaces: RwLock<BTreeMap<NamespaceId, Arc<SetTrees>>>,
}

impl OpenSets {
    /// The system set and the namespace sets an engine opened.
    pub(crate) fn new(
        system: SetTrees,
        namespaces: impl IntoIterator<Item = Arc<SetTrees>>,
    ) -> Self {
        debug_assert_eq!(system.scope(), Scope::System);
        let namespaces = namespaces
            .into_iter()
            .filter_map(|set| match set.scope() {
                Scope::Namespace(ns) => Some((ns, set)),
                Scope::System => None,
            })
            .collect();
        Self {
            system: Arc::new(system),
            namespaces: RwLock::new(namespaces),
        }
    }

    /// The system set.
    #[inline]
    #[must_use]
    pub fn system(&self) -> &Arc<SetTrees> {
        &self.system
    }

    /// Namespace `ns`'s set, when it is open.
    #[must_use]
    pub fn namespace(&self, ns: NamespaceId) -> Option<Arc<SetTrees>> {
        self.namespaces.read().get(&ns).cloned()
    }

    /// The namespaces whose sets are open, in id order.
    #[must_use]
    pub fn namespace_ids(&self) -> Vec<NamespaceId> {
        self.namespaces.read().keys().copied().collect()
    }

    /// The set of `scope`, when it is open.
    #[must_use]
    pub fn set(&self, scope: Scope) -> Option<Arc<SetTrees>> {
        match scope {
            Scope::System => Some(Arc::clone(&self.system)),
            Scope::Namespace(ns) => self.namespace(ns),
        }
    }

    /// The tree `id`, when its set and the tree are open.
    #[must_use]
    pub fn tree(&self, id: TreeId) -> Option<AnyTree> {
        self.set(id.scope).and_then(|set| set.get(id.part).cloned())
    }

    /// Every open tree now: the system set's first, then each namespace's in
    /// id order. A snapshot: a namespace dropped since holds its trees open
    /// only as long as the caller keeps the list.
    #[must_use]
    pub fn list(&self) -> Vec<(TreeId, AnyTree)> {
        let namespaces: Vec<Arc<SetTrees>> = self.namespaces.read().values().cloned().collect();
        std::iter::once(&self.system)
            .chain(namespaces.iter())
            .flat_map(|set| set.iter().map(|(id, tree)| (id, tree.clone())))
            .collect()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
