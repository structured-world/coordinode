//! Opening a namespace's set of trees, at the store's open for every
//! namespace its catalog holds live, and later for one created since.
//!
//! Every tree of every set shares what the open set up: the sequence line,
//! the block cache, the write buffer, the maintenance budget and the
//! compaction room. A namespace's data partitions take the per-level
//! routing the deployment holds for them.

use std::collections::HashMap;
use std::sync::Arc;

use coordinode_core::namespace::{
    NAMESPACE_RECORD_PREFIX, NamespaceId, NamespaceRecord, NamespaceState,
    decode_namespace_record_key,
};

use super::OpenRepair;
use crate::engine::config::StorageConfig;
use crate::engine::open_sets::SetTrees;
use crate::engine::partition::Partition;
use crate::engine::routing::PartitionRouting;
use crate::engine::scope::{Scope, TreeId};
use crate::error::{StorageError, StorageResult};

/// What opening a tree of a namespace takes, kept from the store's open.
pub(crate) struct TreeOpener {
    /// The configuration the store opened with, its filesystem included.
    pub(crate) config: StorageConfig,
    pub(crate) seqno: lsm_tree::SharedSequenceNumberGenerator,
    pub(crate) cache: Arc<lsm_tree::Cache>,
    pub(crate) write_buffer: Arc<lsm_tree::WriteBufferBudget>,
    pub(crate) maintenance: Arc<lsm_tree::rate_limiter::RateLimiter>,
    pub(crate) capacity: Arc<lsm_tree::CapacityPool>,
    pub(crate) repair_policy: lsm_tree::RepairPolicy,
    /// The per-level routing of each data partition that has one; a
    /// namespace's Schema tree stays single-tier. The node partition's also
    /// places the levels of the namespace's columnar tables.
    pub(crate) routing: HashMap<Partition, PartitionRouting>,
    /// Every columnar table's write-pressure series, of every namespace.
    pub(crate) table_metrics: Arc<crate::engine::owned_metrics::TableMetrics>,
}

impl TreeOpener {
    /// Open namespace `namespace`'s trees, one per partition a namespace
    /// holds, creating those that do not exist yet, each repaired in place
    /// when its structure is damaged; the repairs come back for the open to
    /// account for. The set reads its index tree through the installations
    /// that tree records.
    ///
    /// # Errors
    ///
    /// A tree cannot be opened or repaired, or its installations cannot be
    /// read.
    pub(crate) fn open_namespace(
        &self,
        namespace: NamespaceId,
    ) -> StorageResult<(SetTrees, Vec<OpenRepair>)> {
        let mut set = SetTrees::new(Scope::Namespace(namespace));
        let mut repairs = Vec::new();
        let thresholds = self.config.backpressure.to_thresholds();
        for part in Scope::Namespace(namespace).partitions() {
            let tree_config = self
                .config
                .to_tree_config_with_routing(
                    TreeId::namespace(namespace, part),
                    Arc::clone(&self.seqno),
                    self.routing.get(&part),
                )
                .use_cache(Arc::clone(&self.cache))
                .write_buffer_budget(Arc::clone(&self.write_buffer))
                .compaction_rate_limiter(Arc::clone(&self.maintenance))
                .compaction_capacity(Arc::clone(&self.capacity));
            let (tree, repair) = tree_config.open_or_repair(self.repair_policy)?;
            if let Some(report) = repair {
                tracing::warn!(
                    %namespace,
                    partition = part.name(),
                    recovered = report.recovered,
                    salvaged = report.salvaged,
                    scope = ?report.wal_replay_scope(),
                    "partition tree structurally repaired at open"
                );
                repairs.push(OpenRepair {
                    partition: part,
                    replay_scope: report.wal_replay_scope(),
                    report,
                });
            }
            // The monitor's per-tree verdict needs thresholds to compute
            // against: they default to off inside the tree.
            let inner = match &tree {
                lsm_tree::AnyTree::Standard(t) => t,
                lsm_tree::AnyTree::Blob(bt) => &bt.index,
            };
            inner.update_runtime_config(|cfg| cfg.backpressure = thresholds)?;
            set.insert(part, tree);
        }
        let idx = set
            .get(Partition::Idx)
            .ok_or_else(|| StorageError::PartitionNotFound {
                name: Partition::Idx.name().to_string(),
            })?;
        let installations = crate::engine::installation::Installations::load(idx)?;
        let set = set.with_installations(installations);
        #[cfg(feature = "columnar")]
        let set = {
            let (registry, table_repairs) = self.open_tables(namespace)?;
            set.with_columnar(registry, table_repairs)
        };
        Ok((set, repairs))
    }

    /// Open namespace `namespace`'s `STORAGE COLUMNAR` table registry, which
    /// re-opens every table tree of it already on disk, repairing one whose
    /// open fails structurally.
    ///
    /// # Errors
    ///
    /// The registry's directory cannot be made or read, or a table tree
    /// cannot be opened or repaired.
    #[cfg(feature = "columnar")]
    fn open_tables(
        &self,
        namespace: NamespaceId,
    ) -> StorageResult<(
        Arc<crate::columnar::ColumnarTableRegistry>,
        Vec<crate::columnar::TableRepair>,
    )> {
        // The store's filesystem (a memory one for a store kept in memory),
        // so no table tree touches the host's when the store is virtual.
        let fs: Arc<dyn lsm_tree::fs::Fs> = self
            .config
            .fs
            .clone()
            .unwrap_or_else(|| Arc::new(lsm_tree::fs::StdFs));
        let tree_config = {
            let config = self.config.clone();
            let seqno = Arc::clone(&self.seqno);
            let routing = self.routing.get(&Partition::Node).cloned();
            move |table_id: &str| {
                config.to_table_tree_config(
                    namespace,
                    table_id,
                    Arc::clone(&seqno),
                    routing.as_ref(),
                )
            }
        };
        let (registry, repairs) = crate::columnar::ColumnarTableRegistry::open(
            fs,
            crate::columnar::TableTreeSettings {
                config: Box::new(tree_config),
                roots: self
                    .config
                    .endpoints
                    .iter()
                    .map(|endpoint| {
                        endpoint
                            .path
                            .join(Scope::Namespace(namespace).dir())
                            .join(crate::columnar::TABLES_DIR)
                    })
                    .collect(),
                cache: Arc::clone(&self.cache),
                maintenance: Arc::clone(&self.maintenance),
                write_buffer: Arc::clone(&self.write_buffer),
                capacity: Arc::clone(&self.capacity),
                column_encoding: crate::columnar::TableTreeSettings::column_encoding_from(
                    self.config.columnar_encoded_from_level,
                ),
                backpressure: self.config.backpressure.to_thresholds(),
                metrics: Arc::clone(&self.table_metrics),
            },
        )?;
        Ok((Arc::new(registry), repairs))
    }
}

/// The namespaces whose directory lies under any of `endpoints`. A namespace's
/// directory is made only when its set opens, which only a namespace its
/// catalog created does: the directories name the namespaces whose trees the
/// store holds even while the catalog that says so is owed a rebuild. A
/// store kept in memory lists none.
pub(crate) fn namespaces_on_disk(
    endpoints: &[crate::engine::config::EndpointConfig],
) -> std::collections::BTreeSet<NamespaceId> {
    let mut found = std::collections::BTreeSet::new();
    for endpoint in endpoints {
        let Ok(entries) =
            std::fs::read_dir(endpoint.path.join(crate::engine::scope::NAMESPACES_DIR))
        else {
            continue;
        };
        for entry in entries.flatten() {
            if entry.file_type().is_ok_and(|kind| kind.is_dir())
                && let Some(id) = entry
                    .file_name()
                    .to_str()
                    .and_then(NamespaceId::from_dir_name)
            {
                found.insert(id);
            }
        }
    }
    found
}

/// The namespaces the catalog in `records`, the deployment's records tree,
/// holds at `seqno`, live and dropping, in id order.
///
/// # Errors
///
/// A record or its key does not decode, or the read fails.
pub(crate) fn catalog_at(
    records: &lsm_tree::AnyTree,
    seqno: lsm_tree::SeqNo,
) -> StorageResult<Vec<(NamespaceId, NamespaceRecord)>> {
    use lsm_tree::{AbstractTree as _, Guard as _};
    let mut out = Vec::new();
    for guard in records.prefix(NAMESPACE_RECORD_PREFIX, seqno, None) {
        let (key, value) = guard.into_inner()?;
        let malformed = || {
            StorageError::Serialization("a namespace record of the catalog does not decode".into())
        };
        let id = decode_namespace_record_key(&key).ok_or_else(malformed)?;
        out.push((id, NamespaceRecord::decode(&value).ok_or_else(malformed)?));
    }
    Ok(out)
}

/// Whether the catalog in `records` holds namespace `id` live at `seqno`.
///
/// # Errors
///
/// The record does not decode, or the read fails.
pub(crate) fn live_at(
    records: &lsm_tree::AnyTree,
    id: NamespaceId,
    seqno: lsm_tree::SeqNo,
) -> StorageResult<bool> {
    use lsm_tree::AbstractTree as _;
    let key = coordinode_core::namespace::namespace_record_key(id);
    match records.get(&key, seqno)? {
        None => Ok(false),
        Some(bytes) => NamespaceRecord::decode(&bytes)
            .map(|record| record.state == NamespaceState::Live)
            .ok_or_else(|| {
                StorageError::Serialization(format!("the catalog record of {id} does not decode"))
            }),
    }
}
