//! Index registry: the B-tree index definitions in force, and the
//! maintenance of their entries as nodes are created, changed and deleted.
//!
//! Every entry is staged in the writing statement's transaction (see
//! [`coordinode_modality::IndexStore`]), so it commits, replicates and rolls
//! back with the data it indexes.

use std::collections::HashMap;
use std::sync::RwLock;

use coordinode_core::graph::node::NodeId;
use coordinode_core::graph::types::Value;
use coordinode_modality::{IndexStore as _, LocalIndexStore, StoreError};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::transaction::Transaction;
use coordinode_storage::error::StorageError;

use super::definition::{IndexDefinition, IndexType};

/// Registry of active B-tree indexes.
///
/// Interior mutability lets DDL executed inside a statement update the live
/// registry through the shared reference the execution context holds.
pub struct IndexRegistry {
    /// Active indexes: name → definition.
    indexes: RwLock<HashMap<String, Registered>>,
}

/// A definition in force, with the version of its stored record when this
/// member read it; `None` for one registered without a stored record.
#[derive(Debug, Clone)]
struct Registered {
    def: IndexDefinition,
    version: Option<u64>,
}

/// A unique value a statement claimed. Kept until the statement commits, so
/// a claim that loses a race can name the node that won it.
#[derive(Debug, Clone)]
pub struct UniqueClaim {
    /// The unique index.
    pub index: IndexDefinition,
    /// The values claimed, one per indexed property.
    pub values: Vec<Value>,
    /// The node that claimed them.
    pub node_id: NodeId,
}

/// Properties of one node changing value: what the index maintenance of a
/// SET, REMOVE or node merge needs to know.
pub struct PropertyChange<'a> {
    /// The node.
    pub node_id: NodeId,
    /// Its primary label.
    pub label: &'a str,
    /// The properties that change.
    pub properties: &'a [&'a str],
    /// The node's properties before the change.
    pub before: &'a dyn Fn(&str) -> Option<Value>,
    /// The node's properties after the change.
    pub after: &'a dyn Fn(&str) -> Option<Value>,
}

/// A node as it is created, or as it was before it is deleted.
pub struct NodeState<'a> {
    /// The node.
    pub node_id: NodeId,
    /// Its primary label.
    pub label: &'a str,
    /// Its properties.
    pub value_of: &'a dyn Fn(&str) -> Option<Value>,
}

/// The field id a property name is bound to now, which a DERIVED effect is
/// sealed with.
pub type FieldOf<'a> = &'a dyn Fn(&str) -> Option<u32>;

/// A write that would give a unique index's value a second holder.
#[derive(Debug, Clone, thiserror::Error)]
#[error(
    "unique constraint violated on index `{index_name}`: property `{property}` already has \
     value {value:?} (node {})",
    .holder.to_element_id()
)]
pub struct UniqueViolation {
    /// The unique index.
    pub index_name: String,
    /// The indexed properties, comma-separated.
    pub property: String,
    /// The value, or the list of values of a compound index.
    pub value: Value,
    /// The node that holds the value.
    pub holder: NodeId,
}

/// Why maintaining an index failed.
#[derive(Debug, thiserror::Error)]
pub enum IndexWriteError {
    /// The write would break a unique index.
    #[error(transparent)]
    Unique(#[from] UniqueViolation),
    /// The index store failed.
    #[error(transparent)]
    Store(#[from] StoreError),
}

impl UniqueViolation {
    /// The violation of `index` by `values`, held by `holder`.
    pub fn new(index: &IndexDefinition, values: &[Value], holder: NodeId) -> Self {
        Self {
            index_name: index.name.clone(),
            property: index.properties.join(","),
            value: match values {
                [one] => one.clone(),
                many => Value::Array(many.to_vec()),
            },
            holder,
        }
    }
}

/// The values `index` holds for a node whose properties `value_of` answers,
/// or `None` when the node has no entry in it: a sparse index skips a node
/// missing any indexed property, a partial index one its filter rejects.
fn entry_values(
    index: &IndexDefinition,
    value_of: &dyn Fn(&str) -> Option<Value>,
) -> Option<Vec<Value>> {
    let values: Vec<Value> = index
        .properties
        .iter()
        .map(|p| value_of(p).unwrap_or(Value::Null))
        .collect();
    if index.sparse && values.iter().any(Value::is_null) {
        return None;
    }
    if let Some(filter) = &index.filter {
        let property = filter.property();
        let tested = [(
            property.to_string(),
            value_of(property).unwrap_or(Value::Null),
        )];
        if !filter.matches(&tested) {
            return None;
        }
    }
    Some(values)
}

/// A property lookup over a list of `(name, value)` pairs.
pub fn props_lookup(props: &[(String, Value)]) -> impl Fn(&str) -> Option<Value> + '_ {
    move |name| {
        props
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.clone())
    }
}

impl IndexRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            indexes: RwLock::new(HashMap::new()),
        }
    }

    /// Make `index` active in this process without a stored record to bind
    /// writers to: for a context that has none.
    pub fn register_in_memory(&self, index: IndexDefinition) {
        self.insert(index, None);
    }

    /// Make `index` active in this process as its stored record now stands,
    /// after the statement that published it: writers bind their effects to
    /// that record's version.
    ///
    /// # Errors
    ///
    /// A storage failure reading the record's version.
    pub fn register_published(
        &self,
        engine: &StorageEngine,
        index: IndexDefinition,
    ) -> Result<(), StorageError> {
        let version = engine.record_version(
            coordinode_storage::engine::partition::Partition::Schema,
            &index.schema_key(),
        )?;
        self.insert(index, version);
        Ok(())
    }

    fn insert(&self, def: IndexDefinition, version: Option<u64>) {
        self.indexes
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(def.name.clone(), Registered { def, version });
    }

    /// Stop maintaining the index `name` in this process.
    pub fn unregister(&self, name: &str) {
        self.indexes
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .remove(name);
    }

    /// Replace the active set with the definitions stored in the schema
    /// partition. A member that applied another member's CREATE or DROP INDEX
    /// picks it up here. Every type is listed, so planning and advice see
    /// every index; only B-tree indexes have entries maintained here.
    pub fn load_all(&self, engine: &StorageEngine) -> Result<(), StorageError> {
        let defs = super::ops::list_index_definitions(engine)?;
        let mut loaded = HashMap::with_capacity(defs.len());
        for def in defs {
            let version = engine.record_version(
                coordinode_storage::engine::partition::Partition::Schema,
                &def.schema_key(),
            )?;
            loaded.insert(def.name.clone(), Registered { def, version });
        }
        *self.indexes.write().unwrap_or_else(|e| e.into_inner()) = loaded;
        Ok(())
    }

    /// The B-tree indexes on `label`: the ones whose entries this registry
    /// maintains.
    fn btree_for_label(&self, label: &str) -> Vec<Registered> {
        self.indexes
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .filter(|r| r.def.label == label && r.def.index_type == IndexType::BTree)
            .cloned()
            .collect()
    }

    /// The indexes matching `keep` (owned clones).
    fn defs_where(&self, keep: impl Fn(&IndexDefinition) -> bool) -> Vec<IndexDefinition> {
        self.indexes
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .filter(|r| keep(&r.def))
            .map(|r| r.def.clone())
            .collect()
    }

    /// Get all indexes for a specific label (returns owned clones).
    pub fn indexes_for_label(&self, label: &str) -> Vec<IndexDefinition> {
        self.defs_where(|idx| idx.label == label)
    }

    /// Get all indexes that cover a specific label + property (returns owned clones).
    pub fn indexes_for_property(&self, label: &str, property: &str) -> Vec<IndexDefinition> {
        self.defs_where(|idx| idx.label == label && idx.properties.iter().any(|p| p == property))
    }

    /// Get an index by name (returns owned clone).
    pub fn get(&self, name: &str) -> Option<IndexDefinition> {
        self.indexes
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(name)
            .map(|r| r.def.clone())
    }

    /// Every active index (owned clones).
    pub fn all(&self) -> Vec<IndexDefinition> {
        self.defs_where(|_| true)
    }

    /// Whether any B-tree index on `label` reads `property`, as an indexed
    /// value or as its partial filter.
    pub fn reads_property(&self, label: &str, property: &str) -> bool {
        self.btree_for_label(label).iter().any(|r| {
            r.def.properties.iter().any(|p| p == property)
                || r.def
                    .filter
                    .as_ref()
                    .is_some_and(|f| f.property() == property)
        })
    }

    /// Check if any index exists for a label.
    pub fn has_indexes_for(&self, label: &str) -> bool {
        self.indexes
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .any(|r| r.def.label == label)
    }

    /// Whether any B-tree index, whose entries a write maintains, exists for
    /// a label. The check a write makes before any other index work.
    pub fn has_btree_for(&self, label: &str) -> bool {
        self.indexes
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .any(|r| r.def.label == label && r.def.index_type == IndexType::BTree)
    }

    /// Number of registered indexes.
    pub fn len(&self) -> usize {
        self.indexes.read().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.indexes
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty()
    }

    /// Stage the entries of a node being created. A unique value another
    /// node holds refuses the write; each unique value claimed is appended to
    /// `claims`.
    pub fn on_node_created(
        &self,
        engine: &StorageEngine,
        txn: &mut Transaction,
        node: &NodeState<'_>,
        field_of: FieldOf<'_>,
        claims: &mut Vec<UniqueClaim>,
    ) -> Result<(), IndexWriteError> {
        for Registered {
            def: index,
            version,
        } in self.btree_for_label(node.label)
        {
            if entry_values(&index, node.value_of).is_some() {
                bind(txn, &index, version)?;
            }
            stage_node_entry(
                engine,
                txn,
                &index,
                node.node_id,
                node.value_of,
                field_of,
                claims,
            )?;
        }
        Ok(())
    }

    /// Move the entries of a node whose property changes as `change`
    /// describes. Only the indexes that read the property are touched.
    pub fn on_property_changed(
        &self,
        engine: &StorageEngine,
        txn: &mut Transaction,
        change: &PropertyChange<'_>,
        field_of: FieldOf<'_>,
        claims: &mut Vec<UniqueClaim>,
    ) -> Result<(), IndexWriteError> {
        for Registered {
            def: index,
            version,
        } in self.btree_for_label(change.label)
        {
            let changed = |p: &str| change.properties.contains(&p);
            let reads = index.properties.iter().any(|p| changed(p))
                || index.filter.as_ref().is_some_and(|f| changed(f.property()));
            if !reads {
                continue;
            }
            let old = entry_values(&index, change.before);
            let new = entry_values(&index, change.after);
            if old == new {
                continue;
            }
            bind(txn, &index, version)?;
            stage(
                engine,
                txn,
                &index,
                field_of,
                change.node_id,
                old,
                new,
                claims,
            )?;
        }
        Ok(())
    }

    /// Stage the removal of the entries of a node being deleted.
    pub fn on_node_deleted(
        &self,
        engine: &StorageEngine,
        txn: &mut Transaction,
        node: &NodeState<'_>,
        field_of: FieldOf<'_>,
    ) -> Result<(), StoreError> {
        let store = LocalIndexStore::new(engine);
        for Registered {
            def: index,
            version,
        } in self.btree_for_label(node.label)
        {
            if let Some(values) = entry_values(&index, node.value_of) {
                bind(txn, &index, version)?;
                store.stage_membership(txn, &index, field_of, node.node_id, Some(&values), None)?;
            }
        }
        Ok(())
    }
}

/// Bind the writing transaction's effects in `index` to the definition
/// record this member read, when it read one: a transition, drop or rebuild
/// of the index before the commit refuses the write, to be retried under
/// the binding in force.
fn bind(
    txn: &mut Transaction,
    index: &IndexDefinition,
    version: Option<u64>,
) -> Result<(), StoreError> {
    if version.is_some() {
        txn.bind_index_definition(&index.schema_key(), version)?;
    }
    Ok(())
}

/// Stage the entry `index` holds for a node whose properties `value_of`
/// answers, if the node has one there, and say whether it did. A unique
/// value another node holds refuses the write; a unique value claimed is
/// appended to `claims`.
pub fn stage_node_entry(
    engine: &StorageEngine,
    txn: &mut Transaction,
    index: &IndexDefinition,
    node_id: NodeId,
    value_of: &dyn Fn(&str) -> Option<Value>,
    field_of: FieldOf<'_>,
    claims: &mut Vec<UniqueClaim>,
) -> Result<bool, IndexWriteError> {
    match entry_values(index, value_of) {
        Some(values) => stage(
            engine,
            txn,
            index,
            field_of,
            node_id,
            None,
            Some(values),
            claims,
        ),
        None => Ok(false),
    }
}

/// A property lookup over a stored node: declared properties through the
/// field dictionary, then the undeclared ones kept by name.
pub fn record_lookup<'r>(
    record: &'r coordinode_core::graph::node::NodeRecord,
    interner: &'r coordinode_core::graph::intern::FieldInterner,
) -> impl Fn(&str) -> Option<Value> + 'r {
    move |name| {
        interner
            .lookup(name)
            .and_then(|field| record.get(field).cloned())
            .or_else(|| record.get_extra(name).cloned())
    }
}

/// Stage one node's membership in `index` moving from `old` to `new`,
/// refusing a unique value another node holds, and say whether an entry was
/// put.
#[allow(clippy::too_many_arguments)]
fn stage(
    engine: &StorageEngine,
    txn: &mut Transaction,
    index: &IndexDefinition,
    field_of: FieldOf<'_>,
    node_id: NodeId,
    old: Option<Vec<Value>>,
    new: Option<Vec<Value>>,
    claims: &mut Vec<UniqueClaim>,
) -> Result<bool, IndexWriteError> {
    let store = LocalIndexStore::new(engine);
    if index.unique {
        if let Some(values) = &new {
            if let Some(holder) = store.unique_conflict(txn, index, values, node_id)? {
                return Err(UniqueViolation::new(index, values, holder).into());
            }
        }
    }
    let written = store.stage_membership(
        txn,
        index,
        field_of,
        node_id,
        old.as_deref(),
        new.as_deref(),
    )? > 0;
    if index.unique && written {
        if let Some(values) = new {
            claims.push(UniqueClaim {
                index: index.clone(),
                values,
                node_id,
            });
        }
    }
    Ok(written)
}

impl Default for IndexRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
