//! The deployment's namespace catalog as statements reach it:
//! `CREATE NAMESPACE`, `DROP NAMESPACE` and `SHOW NAMESPACES`.
//!
//! A catalog change is the deployment's, not the namespace's a statement
//! runs in, so the executor does not make it itself: the host that runs the
//! statement commits it through its own catalog path ([`NamespaceCatalog`]),
//! the one its embedded API uses too.

use coordinode_core::access::PrincipalId;
use coordinode_core::graph::types::Value;
use coordinode_core::txn::proposal::AccessChange;

use super::row::Row;
use super::runner::ExecutionError;

/// A failure the host's catalog reports, carried through the executor
/// unchanged so the host reads back its own error.
pub type CatalogFailure = Box<dyn std::error::Error + Send + Sync>;

/// One namespace of the catalog, as `SHOW NAMESPACES` lists it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamespaceEntry {
    /// The name sessions name it by.
    pub name: String,
    /// Whether it is being dropped; its name is free again.
    pub dropping: bool,
    /// Its data-block storage settings; unset ones inherit the deployment's.
    pub storage: coordinode_core::namespace::NamespaceStorage,
}

/// What a namespace catalog statement found already so, which its existence
/// guard turns into success.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CatalogOutcome {
    /// The statement changed the catalog.
    Changed,
    /// The state it asked for already held: a creation of a live name, a
    /// drop of an unknown one.
    AlreadySo,
}

/// The host's namespace catalog: changed through the path every catalog
/// change of the deployment takes, so a member of a group changes it through
/// the group's log.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a namespace catalog",
    label = "needs `NamespaceCatalog`",
    note = "the embedded database implements it over its deployment's catalog"
)]
pub trait NamespaceCatalog: Send + Sync {
    /// Create namespace `name` with the data-block storage settings
    /// `storage`, acting for `actor`. [`CatalogOutcome::AlreadySo`] when a
    /// live namespace holds the name.
    ///
    /// # Errors
    ///
    /// The host's refusal: a name no namespace may have, a principal without
    /// the right to create namespaces, a write it could not commit.
    fn create_namespace(
        &self,
        actor: Option<&PrincipalId>,
        name: &str,
        storage: coordinode_core::namespace::NamespaceStorage,
    ) -> Result<CatalogOutcome, CatalogFailure>;

    /// Change the storage settings `change` names of the live namespace
    /// `name`, keeping the others, acting for `actor`.
    ///
    /// # Errors
    ///
    /// The host's refusal: no live namespace of the name, a principal
    /// without admin on it, a write it could not commit.
    fn set_namespace_storage(
        &self,
        actor: Option<&PrincipalId>,
        name: &str,
        change: coordinode_core::namespace::NamespaceStorage,
    ) -> Result<(), CatalogFailure>;

    /// Drop namespace `name`, acting for `actor`.
    /// [`CatalogOutcome::AlreadySo`] when no live namespace has the name.
    ///
    /// # Errors
    ///
    /// The host's refusal: the namespace the statement runs in, a principal
    /// without admin on it, a write it could not commit.
    fn drop_namespace(
        &self,
        actor: Option<&PrincipalId>,
        name: &str,
    ) -> Result<CatalogOutcome, CatalogFailure>;

    /// The catalog's namespaces, in creation order.
    ///
    /// # Errors
    ///
    /// The catalog cannot be read.
    fn namespaces(&self) -> Result<Vec<NamespaceEntry>, CatalogFailure>;

    /// Commit `change` to the deployment's grant catalog, acting for
    /// `actor`; decided where the change applies, like every catalog change.
    ///
    /// # Errors
    ///
    /// The host's refusal: a principal without the right the change needs,
    /// a write it could not commit.
    fn change_access(
        &self,
        actor: Option<&PrincipalId>,
        change: AccessChange,
    ) -> Result<(), CatalogFailure>;
}

/// The catalog a statement reaches, or the refusal of a host that gave none.
fn catalog(
    catalog: Option<&dyn NamespaceCatalog>,
) -> Result<&dyn NamespaceCatalog, ExecutionError> {
    catalog.ok_or_else(|| {
        ExecutionError::Unsupported("this host does not change its namespace catalog".into())
    })
}

/// `CREATE NAMESPACE name [IF NOT EXISTS]`: one row naming the namespace and
/// whether the statement created it.
pub(crate) fn create(
    host: Option<&dyn NamespaceCatalog>,
    actor: Option<&PrincipalId>,
    name: &str,
    if_not_exists: bool,
    storage: coordinode_core::namespace::NamespaceStorage,
) -> Result<Vec<Row>, ExecutionError> {
    let outcome = catalog(host)?
        .create_namespace(actor, name, storage)
        .map_err(ExecutionError::Catalog)?;
    if outcome == CatalogOutcome::AlreadySo && !if_not_exists {
        return Err(ExecutionError::NamespaceExists(name.to_owned()));
    }
    Ok(vec![changed_row(name, outcome)])
}

/// `DROP NAMESPACE name [IF EXISTS]`: one row naming the namespace and
/// whether the statement dropped it.
pub(crate) fn drop(
    host: Option<&dyn NamespaceCatalog>,
    actor: Option<&PrincipalId>,
    name: &str,
    if_exists: bool,
) -> Result<Vec<Row>, ExecutionError> {
    let outcome = catalog(host)?
        .drop_namespace(actor, name)
        .map_err(ExecutionError::Catalog)?;
    if outcome == CatalogOutcome::AlreadySo && !if_exists {
        return Err(ExecutionError::UnknownNamespace(name.to_owned()));
    }
    Ok(vec![changed_row(name, outcome)])
}

/// `ALTER NAMESPACE name SET STORAGE (...)`: one row naming the namespace.
pub(crate) fn set_storage(
    host: Option<&dyn NamespaceCatalog>,
    actor: Option<&PrincipalId>,
    name: &str,
    change: coordinode_core::namespace::NamespaceStorage,
) -> Result<Vec<Row>, ExecutionError> {
    catalog(host)?
        .set_namespace_storage(actor, name, change)
        .map_err(ExecutionError::Catalog)?;
    Ok(vec![changed_row(name, CatalogOutcome::Changed)])
}

/// `GRANT`, `REVOKE`, `TRANSFER`, `CREATE / DROP / ALTER ROLE`: the change
/// committed, with no rows. A change whose condition did not hold where it
/// applied wrote nothing; the catalog says what holds.
pub(crate) fn change_access(
    host: Option<&dyn NamespaceCatalog>,
    actor: Option<&PrincipalId>,
    change: &AccessChange,
) -> Result<Vec<Row>, ExecutionError> {
    catalog(host)?
        .change_access(actor, change.clone())
        .map_err(ExecutionError::Catalog)?;
    Ok(Vec::new())
}

/// `SHOW NAMESPACES`: a row per namespace, its name, state and the storage
/// settings it sets (null for one it inherits from the deployment).
pub(crate) fn show(host: Option<&dyn NamespaceCatalog>) -> Result<Vec<Row>, ExecutionError> {
    let codec = |codec: Option<coordinode_core::namespace::BlockCodec>| {
        codec.map_or(Value::Null, |codec| Value::String(codec.to_string()))
    };
    Ok(catalog(host)?
        .namespaces()
        .map_err(ExecutionError::Catalog)?
        .into_iter()
        .map(|entry| {
            let mut row = Row::new();
            row.insert("name".into(), Value::String(entry.name));
            row.insert(
                "state".into(),
                Value::String(if entry.dropping { "dropping" } else { "live" }.into()),
            );
            row.insert("compression".into(), codec(entry.storage.compression));
            row.insert(
                "cold_compression".into(),
                codec(entry.storage.cold_compression),
            );
            row.insert(
                "dictionaries".into(),
                entry.storage.dictionaries.map_or(Value::Null, Value::Bool),
            );
            row
        })
        .collect())
}

fn changed_row(name: &str, outcome: CatalogOutcome) -> Row {
    let mut row = Row::new();
    row.insert("namespace".into(), Value::String(name.to_owned()));
    row.insert(
        "changed".into(),
        Value::Bool(outcome == CatalogOutcome::Changed),
    );
    row
}
