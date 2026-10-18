//! Constraint declarations checked and staged once, for every path that
//! creates a constraint: `CREATE CONSTRAINT`, `CREATE UNIQUE INDEX`, the
//! embedded call and a schema package.
//!
//! Planning checks a declaration against the label definition it will be
//! published under, which a package may be staging in the same commit.
//! Staging writes what a commit publishes; nothing here commits.

use coordinode_core::schema::definition::{
    ConstraintKind, ConstraintState, LabelSchema, NodeConstraint, PropertyType,
};
use coordinode_modality::{IndexStore as _, LocalIndexStore, StoreError};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::transaction::Transaction;

use super::runner::ExecutionError;
use crate::index::definition::PartialFilter;
use crate::index::{
    BuildFailure, DuplicateRepair, IndexBuildRecord, IndexDefinition, IndexDescriptor,
    IndexProfile, IndexState,
};

/// A constraint to create, before it is checked.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstraintRequest {
    /// Its name; `None` derives one from the label, properties and kind.
    pub name: Option<String>,
    /// The label whose nodes it constrains.
    pub label: String,
    /// The constrained properties, in order.
    pub properties: Vec<String>,
    /// What it requires.
    pub kind: ConstraintKind,
    /// For a uniqueness, the nodes it holds among.
    pub scope: Option<PartialFilter>,
    /// The string property a uniqueness build may rewrite to end a stored
    /// duplicate.
    pub on_duplicate_rename: Option<String>,
    /// Leave nodes missing a value out of the owned index; `None` takes the
    /// kind's own choice (a uniqueness skips them, a key requires them).
    pub sparse: Option<bool>,
    /// The owned index's maintenance profile; `None` takes the namespace
    /// default.
    pub maintenance: Option<IndexProfile>,
}

impl ConstraintRequest {
    /// The name the constraint is published under.
    pub fn resolved_name(&self) -> String {
        self.name
            .clone()
            .unwrap_or_else(|| derived_constraint_name(&self.label, &self.properties, &self.kind))
    }

    /// The requirement, as the catalog would hold it once active.
    pub fn requirement(&self) -> NodeConstraint {
        NodeConstraint {
            name: self.resolved_name(),
            properties: self.properties.clone(),
            kind: self.kind.clone(),
            state: ConstraintState::Active,
            scope: self.scope.clone(),
        }
    }
}

/// A checked constraint, ready to stage.
#[derive(Debug, Clone)]
pub struct PlannedConstraint {
    /// The label it is added to.
    pub label: String,
    /// The constraint as it is published: validating while it owns an index
    /// still to be built, active otherwise.
    pub constraint: NodeConstraint,
    /// The unique index it owns, published building with it.
    pub owned_index: Option<OwnedIndex>,
}

/// The index a uniqueness or key constraint owns, before publication.
#[derive(Debug, Clone)]
pub struct OwnedIndex {
    /// The index, named after the constraint.
    pub descriptor: IndexDescriptor,
    /// Its maintenance profile; `None` takes the namespace default.
    pub maintenance: Option<IndexProfile>,
    /// The repair its build may make of a stored duplicate.
    pub repair: Option<DuplicateRepair>,
}

/// Check `request` against `schema`, the definition of its label the
/// constraint is published under (`None`: the label has none). Whether the
/// name is free is the caller's to check, against the catalog it stages in.
///
/// # Errors
///
/// The declaration cannot be enforced as asked, or cannot hold under
/// `schema`.
pub fn plan_constraint(
    request: &ConstraintRequest,
    schema: Option<&LabelSchema>,
) -> Result<PlannedConstraint, ExecutionError> {
    let mut constraint = request.requirement();
    constraint.state = ConstraintState::Validating;
    let label = request.label.as_str();
    // A scope narrows which nodes are compared with each other; a key also
    // requires its values of every node, which a scope cannot narrow.
    if constraint.scope.is_some() && constraint.kind != ConstraintKind::Unique {
        return Err(ExecutionError::CatalogRefused(format!(
            "only a uniqueness can hold among the nodes a predicate admits, not an IS {} \
             constraint",
            constraint.kind
        )));
    }
    if request.on_duplicate_rename.is_some() && !constraint.owns_index() {
        return Err(ExecutionError::CatalogRefused(format!(
            "ON DUPLICATE RENAME repairs duplicates of a UNIQUE or NODE KEY constraint, not of \
             an IS {} one",
            constraint.kind
        )));
    }
    let repair = duplicate_repair(
        request.on_duplicate_rename.as_deref(),
        label,
        &request.properties,
        schema,
    )?;
    // The rows of a COLUMNAR table are written outside the transaction a
    // schema revision binds, so nothing could hold them to the constraint
    // while it is enabled.
    if schema.is_some_and(LabelSchema::is_columnar) {
        return Err(ExecutionError::CatalogRefused(format!(
            "constraints on the COLUMNAR table '{label}' are not supported"
        )));
    }
    if let Some(conflict) = schema.and_then(|s| s.constraint_conflict(&constraint)) {
        return Err(ExecutionError::CatalogRefused(format!(
            "constraint '{}' cannot hold: {conflict}",
            constraint.name
        )));
    }
    if !constraint.owns_index() {
        // Checked on each node in the commit publishing it: active there.
        constraint.state = ConstraintState::Active;
        return Ok(PlannedConstraint {
            label: request.label.clone(),
            constraint,
            owned_index: None,
        });
    }
    let mut descriptor =
        IndexDescriptor::compound(&constraint.name, label, request.properties.clone())
            .unique()
            .owned_by(&constraint.name);
    // A node missing a value is not constrained by uniqueness; a key
    // requires the values, which its per-node part enforces.
    if request
        .sparse
        .unwrap_or(constraint.kind == ConstraintKind::Unique)
    {
        descriptor = descriptor.sparse();
    }
    if let Some(scope) = &constraint.scope {
        descriptor = descriptor.with_filter(scope.clone());
    }
    Ok(PlannedConstraint {
        label: request.label.clone(),
        constraint,
        owned_index: Some(OwnedIndex {
            descriptor,
            maintenance: request.maintenance,
            repair,
        }),
    })
}

/// Stage in `txn` the publication of `owned` as building, with its build
/// admitted: the catalog gives it a new identity and generation and binds
/// its name, on the condition that no live index holds the name when the
/// commit lands. The build is the engine's to run once the commit is in.
///
/// # Errors
///
/// The index policy or the catalog could not be read or staged.
pub fn stage_owned_index(
    engine: &StorageEngine,
    txn: &mut Transaction<'_>,
    owned: OwnedIndex,
) -> Result<IndexDefinition, StoreError> {
    stage_index_build(
        engine,
        txn,
        owned.descriptor,
        owned.maintenance,
        owned.repair,
    )
}

/// Stage in `txn` the publication of B-tree index `descriptor` as building,
/// with its build admitted, as [`stage_owned_index`] does.
///
/// # Errors
///
/// The index policy or the catalog could not be read or staged.
pub fn stage_index_build(
    engine: &StorageEngine,
    txn: &mut Transaction<'_>,
    mut descriptor: IndexDescriptor,
    maintenance: Option<IndexProfile>,
    repair: Option<DuplicateRepair>,
) -> Result<IndexDefinition, StoreError> {
    let store = LocalIndexStore::new(engine);
    // The binding is resolved once, here, and recorded: a later change of
    // the namespace default does not reinterpret this index.
    let (policy, _) = store.index_policy()?;
    descriptor.maintenance = policy.resolve(maintenance, 1);
    descriptor.state = IndexState::Building {
        written: 0,
        estimated_total: 0,
    };
    let def = store.publish_definition_txn(txn, descriptor)?;
    store.put_build_txn(
        txn,
        &IndexBuildRecord::accepted(def.id, def.generation, BuildFailure::Withdraw)
            .repairing(repair),
        None,
    )?;
    Ok(def)
}

/// The name a constraint declared without one gets: its label, properties
/// and kind, so repeating the declaration names the same constraint.
pub fn derived_constraint_name(
    label: &str,
    properties: &[String],
    kind: &ConstraintKind,
) -> String {
    let suffix = match kind {
        ConstraintKind::Unique => "unique",
        ConstraintKind::NotNull => "not_null",
        ConstraintKind::NodeKey => "node_key",
        ConstraintKind::Type(_) => "type",
    };
    format!("{label}_{}_{suffix}", properties.join("_"))
}

/// The repair `ON DUPLICATE RENAME target` lets the build of a unique index
/// over `properties` of `label` make, refused before anything is published
/// unless the build can make it: the target is one of the properties, and a
/// string `schema` lets be rewritten in place.
fn duplicate_repair(
    target: Option<&str>,
    label: &str,
    properties: &[String],
    schema: Option<&LabelSchema>,
) -> Result<Option<DuplicateRepair>, ExecutionError> {
    let Some(target) = target else {
        return Ok(None);
    };
    let refused = |why: String| {
        Err(ExecutionError::CatalogRefused(format!(
            "ON DUPLICATE RENAME {target}: {why}"
        )))
    };
    if !properties.iter().any(|p| p == target) {
        return refused(format!(
            "the repaired property must be one the uniqueness covers ({})",
            properties.join(", ")
        ));
    }
    if let Some(schema) = schema {
        if schema.temporal {
            // Every version a temporal node held keeps its value, so a
            // rewrite of the current one leaves the duplicate in the history
            // the index covers.
            return refused(format!(
                ":{label} is temporal; a repair cannot rewrite the versions it keeps"
            ));
        }
        if schema.is_columnar() {
            return refused(format!(":{label} is a COLUMNAR table"));
        }
        match schema.properties.get(target).map(|p| &p.property_type) {
            None | Some(PropertyType::String) => {}
            Some(PropertyType::Computed(_)) => {
                return refused(format!("`{target}` of :{label} is computed and read-only"));
            }
            Some(other) => {
                return refused(format!(
                    "`{target}` of :{label} is declared {other}; a repair appends a suffix to \
                     a string"
                ));
            }
        }
    }
    Ok(Some(DuplicateRepair {
        property: target.to_string(),
    }))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
