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
    /// Set when the set's namespace was dropped: the directories its trees
    /// lie in, removed once the set itself goes. no-std: spin::Mutex
    retire: parking_lot::Mutex<Option<Retire>>,
    /// The namespace's `STORAGE COLUMNAR` table trees, under
    /// `<endpoint>/ns/<id>/tables`; `None` for the system set.
    #[cfg(feature = "columnar")]
    columnar: Option<Arc<crate::columnar::ColumnarTableRegistry>>,
    /// The namespace's tables whose tree the open repaired by block salvage
    /// and which still owe what it lost. no-std: spin::Mutex
    #[cfg(feature = "columnar")]
    table_repairs: parking_lot::Mutex<Vec<crate::columnar::TableRepair>>,
    /// The index layer's handling of transferred copies of the namespace's
    /// indexes: each namespace has an index catalog of its own.
    index_copy_qualifier: RwLock<Option<Arc<dyn crate::engine::index_copies::IndexCopyQualifier>>>,
    /// Set whenever the set is reached, cleared by each pass that looks for
    /// idle sets: a set found clear was reached by nothing for a whole pass.
    touched: std::sync::atomic::AtomicBool,
    /// The namespace's storage settings as the catalog holds them now: what
    /// a table tree opened later writes with. no-std: spin::RwLock
    storage: Arc<RwLock<coordinode_core::namespace::NamespaceStorage>>,
}

/// What a dropped namespace's set removes when it goes.
struct Retire {
    dirs: Vec<std::path::PathBuf>,
    footprints: crate::engine::footprint::Footprints,
}

impl Drop for SetTrees {
    fn drop(&mut self) {
        let Some(retire) = self.retire.get_mut().take() else {
            return;
        };
        // The set's own handles first, so nothing of it keeps a file open.
        self.trees = Default::default();
        self.installations = None;
        #[cfg(feature = "columnar")]
        {
            self.columnar = None;
        }
        crate::engine::core::remove_namespace_dirs(&retire.dirs, &retire.footprints);
    }
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
            retire: parking_lot::Mutex::new(None),
            #[cfg(feature = "columnar")]
            columnar: None,
            #[cfg(feature = "columnar")]
            table_repairs: parking_lot::Mutex::new(Vec::new()),
            index_copy_qualifier: RwLock::new(None),
            touched: std::sync::atomic::AtomicBool::new(true),
            storage: Arc::new(RwLock::new(
                coordinode_core::namespace::NamespaceStorage::default(),
            )),
        }
    }

    /// The set with its namespace's storage settings.
    #[must_use]
    pub(crate) fn with_storage(
        self,
        storage: coordinode_core::namespace::NamespaceStorage,
    ) -> Self {
        *self.storage.write() = storage;
        self
    }

    /// The cell holding the namespace's storage settings, shared with what
    /// opens its table trees.
    pub(crate) fn storage_cell(&self) -> Arc<RwLock<coordinode_core::namespace::NamespaceStorage>> {
        Arc::clone(&self.storage)
    }

    /// Record the namespace's storage settings the catalog now holds.
    pub(crate) fn set_storage(&self, storage: coordinode_core::namespace::NamespaceStorage) {
        *self.storage.write() = storage;
    }

    /// Note that the set was reached.
    #[inline]
    pub(crate) fn touch(&self) {
        // Relaxed: a hint for the idle pass, which decides under the open
        // lock and the set's handle count, not on this flag alone.
        self.touched
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether the set was reached since the last call, clearing the mark.
    pub(crate) fn take_touched(&self) -> bool {
        self.touched
            .swap(false, std::sync::atomic::Ordering::Relaxed)
    }

    /// Close every tree of the set, the namespace's tables' included,
    /// keeping their directories: the namespace goes idle and opens again
    /// from disk. The caller flushed the trees and holds the namespace closed
    /// meanwhile. The signals report when each tree no longer touches its
    /// directory, which a reopen waits for.
    pub(crate) fn close(mut self) -> Vec<lsm_tree::TreeClose> {
        let mut closing: Vec<lsm_tree::TreeClose> = self
            .trees
            .iter_mut()
            .filter_map(Option::take)
            .map(lsm_tree::AnyTree::close)
            .collect();
        self.installations = None;
        #[cfg(feature = "columnar")]
        if let Some(registry) = self.columnar.take() {
            closing.extend(registry.close_all());
        }
        closing
    }

    /// The index layer's handling of the namespace's transferred index
    /// copies, once it registered one.
    pub(crate) fn index_copy_qualifier(
        &self,
    ) -> Option<Arc<dyn crate::engine::index_copies::IndexCopyQualifier>> {
        self.index_copy_qualifier.read().clone()
    }

    /// Register the index layer's handling of the namespace's transferred
    /// index copies.
    pub(crate) fn set_index_copy_qualifier(
        &self,
        qualifier: Arc<dyn crate::engine::index_copies::IndexCopyQualifier>,
    ) {
        *self.index_copy_qualifier.write() = Some(qualifier);
    }

    /// The set with its namespace's table registry and the repairs its open
    /// made.
    #[cfg(feature = "columnar")]
    #[must_use]
    pub(crate) fn with_columnar(
        mut self,
        registry: Arc<crate::columnar::ColumnarTableRegistry>,
        repairs: Vec<crate::columnar::TableRepair>,
    ) -> Self {
        self.columnar = Some(registry);
        self.table_repairs = parking_lot::Mutex::new(repairs);
        self
    }

    /// The namespace's table registry, when the set has one.
    #[cfg(feature = "columnar")]
    #[must_use]
    pub(crate) fn columnar(&self) -> Option<&Arc<crate::columnar::ColumnarTableRegistry>> {
        self.columnar.as_ref()
    }

    /// The namespace's table repairs still owed.
    #[cfg(feature = "columnar")]
    pub(crate) fn table_repairs(&self) -> &parking_lot::Mutex<Vec<crate::columnar::TableRepair>> {
        &self.table_repairs
    }

    /// Have the set remove `dirs`, its namespace's directories, once it
    /// goes: its namespace was dropped.
    pub(crate) fn retire_on_drop(
        &self,
        dirs: Vec<std::path::PathBuf>,
        footprints: crate::engine::footprint::Footprints,
    ) {
        *self.retire.lock() = Some(Retire { dirs, footprints });
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

    /// Add namespace `set`, opened after the engine. A set of the system or
    /// of a namespace already open is a programming error the debug build
    /// catches; the release build keeps the set it has.
    pub(crate) fn insert(&self, set: Arc<SetTrees>) {
        let Scope::Namespace(namespace) = set.scope() else {
            debug_assert!(false, "the system set is not added after the open");
            return;
        };
        let mut namespaces = self.namespaces.write();
        debug_assert!(
            !namespaces.contains_key(&namespace),
            "{namespace} opened twice"
        );
        namespaces.entry(namespace).or_insert(set);
    }

    /// Take namespace `namespace`'s set out: nothing new reaches its trees
    /// through the engine. Returned, so the caller can see it through.
    pub(crate) fn remove(&self, namespace: NamespaceId) -> Option<Arc<SetTrees>> {
        self.namespaces.write().remove(&namespace)
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

    /// Every `STORAGE COLUMNAR` table tree of every open namespace now, for
    /// the flush and compaction monitors' passes.
    #[cfg(feature = "columnar")]
    #[must_use]
    pub(crate) fn table_trees(&self) -> Vec<crate::engine::flush::TableTree> {
        let namespaces: Vec<Arc<SetTrees>> = self.namespaces.read().values().cloned().collect();
        namespaces
            .iter()
            .filter_map(|set| set.columnar())
            .flat_map(|registry| registry.trees())
            .collect()
    }

    /// Every open namespace's set now, in id order.
    #[must_use]
    pub(crate) fn namespace_sets(&self) -> Vec<Arc<SetTrees>> {
        self.namespaces.read().values().cloned().collect()
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
