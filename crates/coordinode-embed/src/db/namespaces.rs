//! The deployment's namespace catalog: creating, dropping and listing the
//! namespaces a deployment holds.
//!
//! A catalog change is a command of the deployment, not of the namespace the
//! handle works in: it commits through the same write path as every other
//! write, so a member of a group changes the catalog through the group's log
//! and every member applies it in the same order.

use coordinode_core::access::PrincipalId;
use coordinode_core::namespace::{NamespaceState, NamespaceStorage, validate_namespace_name};
use coordinode_core::txn::proposal::MetadataCommand;
use coordinode_query::executor::namespaces::{CatalogFailure, CatalogOutcome, NamespaceEntry};
use coordinode_storage::engine::metadata;

use super::{Database, DatabaseError};

/// A namespace of the deployment's catalog.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamespaceInfo {
    /// The namespace's name, as sessions name it.
    pub name: String,
    /// Whether it is live, or being dropped: a dropping namespace is named
    /// by no session, and its name is free again.
    pub dropping: bool,
    /// Its data-block storage settings; unset ones inherit the deployment's.
    pub storage: NamespaceStorage,
}

impl Database {
    /// Open the store `config` describes, bound to its namespace called
    /// `name`: what a host serving several namespaces opens one of, and what
    /// a restore into a named namespace writes. With `create`, a namespace
    /// of the name is created when the catalog holds none.
    ///
    /// # Errors
    ///
    /// [`DatabaseError::UnknownNamespace`] when the catalog holds no such
    /// namespace and `create` is off; [`DatabaseError::NamespaceRefused`]
    /// for a name no namespace may have; what opening the store refuses.
    pub fn open_namespace(
        config: coordinode_storage::engine::config::StorageConfig,
        name: &str,
        create: bool,
    ) -> Result<Self, DatabaseError> {
        use coordinode_storage::error::StorageError;
        let path = config.data_dir().to_path_buf();
        let oracle = std::sync::Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new());
        let deployment = coordinode_storage::engine::core::Deployment::open_embedded(
            &config,
            std::sync::Arc::clone(&oracle),
        )?;
        let engine = match deployment.engine_named(name) {
            Ok(engine) => engine,
            Err(StorageError::NoSuchNamespace { .. } | StorageError::UnknownNamespace { .. })
                if create =>
            {
                validate_namespace_name(name).map_err(|e| DatabaseError::NamespaceRefused {
                    name: name.to_owned(),
                    reason: e.to_string(),
                })?;
                deployment.engine(deployment.create_namespace(name)?)?
            }
            Err(StorageError::NoSuchNamespace { .. } | StorageError::UnknownNamespace { .. }) => {
                return Err(DatabaseError::UnknownNamespace(name.to_owned()));
            }
            Err(e) => return Err(e.into()),
        };
        let engine = std::sync::Arc::new(engine);
        let pipeline: std::sync::Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> =
            std::sync::Arc::new(
                coordinode_storage::engine::local_pipeline::OwnedLocalProposalPipeline::new(
                    &engine,
                ),
            );
        Self::finish_open(&path, oracle, engine, pipeline, false, None)
    }

    /// Create the namespace `name` in the deployment's catalog. It holds no
    /// data; a session naming it works in it from here on.
    ///
    /// # Errors
    ///
    /// [`DatabaseError::NamespaceRefused`] for a name no namespace may have;
    /// [`DatabaseError::NamespaceExists`] when a live namespace holds it,
    /// including one a concurrent creation took first; the write path's
    /// errors (a follower answers [`DatabaseError::NotLeader`]).
    pub fn create_namespace(&self, name: &str) -> Result<(), DatabaseError> {
        self.create_namespace_with(name, NamespaceStorage::default())
    }

    /// [`Self::create_namespace`] with the data-block storage settings
    /// `storage`: the codec of its hot and cold levels and whether it uses
    /// the per-label dictionaries. Each unset one inherits the deployment's.
    ///
    /// # Errors
    ///
    /// As [`Self::create_namespace`].
    pub fn create_namespace_with(
        &self,
        name: &str,
        storage: NamespaceStorage,
    ) -> Result<(), DatabaseError> {
        self.create_namespace_as(None, name, storage)
    }

    /// [`Self::create_namespace_with`] acting for `actor`: a principal
    /// creates only with the right to create namespaces, and becomes admin of
    /// what it created in the same change.
    pub(super) fn create_namespace_as(
        &self,
        actor: Option<PrincipalId>,
        name: &str,
        storage: NamespaceStorage,
    ) -> Result<(), DatabaseError> {
        validate_namespace_name(name).map_err(|e| DatabaseError::NamespaceRefused {
            name: name.to_owned(),
            reason: e.to_string(),
        })?;
        let deployment = self.engine.deployment();
        // A principal without the right learns nothing of which names are
        // taken: refused here as the apply would refuse it. The apply still
        // decides; this only answers early.
        if let Some(principal) = &actor {
            let rights =
                coordinode_storage::engine::access::deployment_rights(deployment, principal)?;
            if !(rights.create_namespace || rights.admin) {
                return Err(DatabaseError::AccessRefused(
                    coordinode_core::access::AccessRefusal {
                        principal: principal.clone(),
                        missing: coordinode_core::access::MissingRight::CreateNamespace,
                    },
                ));
            }
        }
        if metadata::namespace_by_name(deployment, name)?.is_some() {
            return Err(DatabaseError::NamespaceExists(name.to_owned()));
        }
        // The token tells this creation's record from a concurrent one of
        // the same name, which the log orders first and this one then loses.
        let token = metadata::namespace_token();
        self.propose_command_as(
            actor,
            MetadataCommand::CreateNamespace {
                name: name.to_owned(),
                token,
                storage,
            },
        )?;
        let id = metadata::namespace_by_name(deployment, name)?
            .ok_or_else(|| DatabaseError::NamespaceExists(name.to_owned()))?;
        match metadata::namespace_record(deployment, id)? {
            Some(record) if record.token == token => Ok(()),
            _ => Err(DatabaseError::NamespaceExists(name.to_owned())),
        }
    }

    /// Drop the live namespace `name` from the deployment's catalog: from
    /// here on no session names it, and its data goes with it.
    ///
    /// # Errors
    ///
    /// [`DatabaseError::UnknownNamespace`] when no live namespace has the
    /// name; [`DatabaseError::NamespaceInUse`] for the namespace this handle
    /// works in, which is dropped from another; the write path's errors.
    pub fn drop_namespace(&self, name: &str) -> Result<(), DatabaseError> {
        self.drop_namespace_as(None, name)
    }

    /// [`Self::drop_namespace`] acting for `actor`, which needs admin on the
    /// namespace.
    pub(super) fn drop_namespace_as(
        &self,
        actor: Option<PrincipalId>,
        name: &str,
    ) -> Result<(), DatabaseError> {
        let deployment = self.engine.deployment();
        let found = metadata::namespace_by_name(deployment, name)?;
        // A principal without admin on it learns nothing of whether it
        // exists: refused here as the apply would refuse it.
        if let Some(principal) = &actor {
            use coordinode_core::access::{AccessRefusal, MissingRight, NamespaceRole};
            use coordinode_storage::engine::access;
            let missing = match found {
                None => Some(MissingRight::DeploymentAdmin),
                Some(namespace) => {
                    let admin = access::deployment_rights(deployment, principal)?.admin
                        || access::effective_role(deployment, principal, namespace)?
                            == Some(NamespaceRole::Admin);
                    (!admin).then_some(MissingRight::Namespace {
                        namespace,
                        role: NamespaceRole::Admin,
                    })
                }
            };
            if let Some(missing) = missing {
                return Err(DatabaseError::AccessRefused(AccessRefusal {
                    principal: principal.clone(),
                    missing,
                }));
            }
        }
        let Some(id) = found else {
            return Err(DatabaseError::UnknownNamespace(name.to_owned()));
        };
        // PostgreSQL refuses to drop the database a connection is open on
        // (DROP DATABASE: "cannot drop the currently open database").
        if id == self.engine.namespace() {
            return Err(DatabaseError::NamespaceInUse(name.to_owned()));
        }
        self.propose_command_as(
            actor,
            MetadataCommand::DropNamespace {
                name: name.to_owned(),
            },
        )?;
        // A concurrent drop of the same namespace the log ordered first
        // leaves the same outcome: the namespace is gone.
        Ok(())
    }

    /// Change the storage settings `change` names of the live namespace
    /// `name`, keeping the others: its trees write with them from their next
    /// flush or compaction, on every member; nothing already written is
    /// rewritten for it.
    ///
    /// # Errors
    ///
    /// [`DatabaseError::UnknownNamespace`] when no live namespace has the
    /// name; the write path's errors.
    pub fn set_namespace_storage(
        &self,
        name: &str,
        change: NamespaceStorage,
    ) -> Result<(), DatabaseError> {
        self.set_namespace_storage_as(None, name, change)
    }

    /// [`Self::set_namespace_storage`] acting for `actor`, which needs admin
    /// on the namespace.
    fn set_namespace_storage_as(
        &self,
        actor: Option<PrincipalId>,
        name: &str,
        change: NamespaceStorage,
    ) -> Result<(), DatabaseError> {
        if metadata::namespace_by_name(self.engine.deployment(), name)?.is_none() {
            return Err(DatabaseError::UnknownNamespace(name.to_owned()));
        }
        self.propose_command_as(
            actor,
            MetadataCommand::SetNamespaceStorage {
                name: name.to_owned(),
                change,
            },
        )
    }

    /// Close the trees of every namespace of the deployment that no handle
    /// holds and nothing reached since the previous call, flushing them
    /// first; each opens again from disk on its next use. A replicated
    /// member runs this itself, on its session idle bound; a host keeping
    /// sessions of its own calls it as often as it wants namespaces closed.
    /// The number closed.
    ///
    /// # Errors
    ///
    /// A flush failure; the namespace it struck stays open.
    pub fn close_idle_namespaces(&self) -> Result<usize, DatabaseError> {
        Ok(self.engine.deployment().close_idle_namespaces()?)
    }

    /// The namespaces of the deployment's catalog, in creation order.
    ///
    /// # Errors
    ///
    /// The catalog cannot be read.
    pub fn namespaces(&self) -> Result<Vec<NamespaceInfo>, DatabaseError> {
        Ok(metadata::namespaces(self.engine.deployment())?
            .into_iter()
            .map(|(_, record)| NamespaceInfo {
                name: record.name,
                dropping: record.state == NamespaceState::Dropping,
                storage: record.storage,
            })
            .collect())
    }
}

/// The statements' path to the catalog is the embedded API's: one way a
/// namespace is created or dropped, whichever surface asks.
impl coordinode_query::executor::namespaces::NamespaceCatalog for Database {
    fn create_namespace(
        &self,
        actor: Option<&PrincipalId>,
        name: &str,
        storage: NamespaceStorage,
    ) -> Result<CatalogOutcome, CatalogFailure> {
        match self.create_namespace_as(actor.cloned(), name, storage) {
            Ok(()) => Ok(CatalogOutcome::Changed),
            Err(DatabaseError::NamespaceExists(_)) => Ok(CatalogOutcome::AlreadySo),
            Err(e) => Err(Box::new(e)),
        }
    }

    fn set_namespace_storage(
        &self,
        actor: Option<&PrincipalId>,
        name: &str,
        change: NamespaceStorage,
    ) -> Result<(), CatalogFailure> {
        Ok(self.set_namespace_storage_as(actor.cloned(), name, change)?)
    }

    fn drop_namespace(
        &self,
        actor: Option<&PrincipalId>,
        name: &str,
    ) -> Result<CatalogOutcome, CatalogFailure> {
        match self.drop_namespace_as(actor.cloned(), name) {
            Ok(()) => Ok(CatalogOutcome::Changed),
            Err(DatabaseError::UnknownNamespace(_)) => Ok(CatalogOutcome::AlreadySo),
            Err(e) => Err(Box::new(e)),
        }
    }

    fn change_access(
        &self,
        actor: Option<&PrincipalId>,
        change: coordinode_core::txn::proposal::AccessChange,
    ) -> Result<(), CatalogFailure> {
        Ok(self.propose_command_as(actor.cloned(), MetadataCommand::from(change))?)
    }

    fn namespaces(&self) -> Result<Vec<NamespaceEntry>, CatalogFailure> {
        Ok(Database::namespaces(self)?
            .into_iter()
            .map(|n| NamespaceEntry {
                name: n.name,
                dropping: n.dropping,
                storage: n.storage,
            })
            .collect())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
