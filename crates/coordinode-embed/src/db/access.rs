//! Changes to the deployment's grant catalog: who may read, write and
//! administer which namespace, and who holds which role.
//!
//! A change is a command of the deployment, committed through the same write
//! path as every other write and decided where it is applied: whether its
//! actor may make it, and whether its condition holds (a transfer from a
//! holder that no longer holds), is read from the catalog the log has
//! produced up to that entry, identically on every member.

use coordinode_core::access::{AccessRefusal, Grantee, MissingRight, NamespaceRole, PrincipalId};
use coordinode_core::namespace::NamespaceId;
use coordinode_core::txn::command::ActingContext;
use coordinode_core::txn::proposal::{AccessChange, GranteeName, MetadataCommand, Mutation};
use coordinode_storage::engine::{access, metadata};

use super::{Database, DatabaseError};

/// One grant on a namespace, as the catalog holds it now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrantView {
    /// Who holds it: a principal, or a role by its home and name.
    pub grantee: GranteeName,
    /// The right.
    pub role: NamespaceRole,
    /// The commit timestamp of the change that made the grant what it is.
    pub changed_ts: u64,
}

/// A role and who holds it, as the catalog holds it now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoleView {
    /// The principals holding it.
    pub holders: Vec<PrincipalId>,
    /// For an exclusive role, the principals that may hold it; `None` for a
    /// role any number of principals may hold.
    pub eligible: Option<Vec<PrincipalId>>,
    /// The commit timestamp of the change that last changed the role or who
    /// holds it: what a taker conditions a transfer on.
    pub changed_ts: u64,
}

/// A holder of the whole-deployment admin right, which reaches every
/// namespace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeploymentAdmin {
    /// The principal.
    pub principal: PrincipalId,
    /// The commit timestamp of the change that gave it the right.
    pub changed_ts: u64,
}

/// The deployment's catalog as one principal reaches it without a session
/// on any namespace's data: what a principal holding only the right to
/// create namespaces does before it has one. It runs no statement, reads
/// and writes no namespace's data and opens no cursor or transaction: it
/// has no way to.
pub struct CatalogScope<'d> {
    db: &'d Database,
    principal: PrincipalId,
}

impl core::fmt::Debug for CatalogScope<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CatalogScope")
            .field("principal", &self.principal)
            .finish_non_exhaustive()
    }
}

impl CatalogScope<'_> {
    /// The principal this scope acts for.
    #[must_use]
    pub fn principal(&self) -> &PrincipalId {
        &self.principal
    }

    /// Create the namespace `name`, with the right to create namespaces;
    /// the principal becomes its admin in the same change.
    ///
    /// # Errors
    ///
    /// As [`Database::create_namespace`];
    /// [`DatabaseError::AccessRefused`] without the right.
    pub fn create_namespace(&self, name: &str) -> Result<(), DatabaseError> {
        self.create_namespace_with(
            name,
            coordinode_core::namespace::NamespaceStorage::default(),
        )
    }

    /// [`Self::create_namespace`] with the data-block storage settings
    /// `storage`.
    ///
    /// # Errors
    ///
    /// As [`Self::create_namespace`].
    pub fn create_namespace_with(
        &self,
        name: &str,
        storage: coordinode_core::namespace::NamespaceStorage,
    ) -> Result<(), DatabaseError> {
        self.db
            .create_namespace_as(Some(self.principal.clone()), name, storage)
    }

    /// Drop the namespace `name`, as its admin.
    ///
    /// # Errors
    ///
    /// As [`Database::drop_namespace`];
    /// [`DatabaseError::AccessRefused`] without admin on it.
    pub fn drop_namespace(&self, name: &str) -> Result<(), DatabaseError> {
        self.db
            .drop_namespace_as(Some(self.principal.clone()), name)
    }

    /// The deployment's namespaces, for a principal that may create
    /// namespaces or holds the deployment admin right.
    ///
    /// # Errors
    ///
    /// [`DatabaseError::AccessRefused`] for any other principal; the
    /// catalog cannot be read.
    pub fn namespaces(&self) -> Result<Vec<super::NamespaceInfo>, DatabaseError> {
        let rights = access::deployment_rights(self.db.engine.deployment(), &self.principal)?;
        if !(rights.create_namespace || rights.admin) {
            return Err(DatabaseError::AccessRefused(AccessRefusal {
                principal: self.principal.clone(),
                missing: MissingRight::CreateNamespace,
            }));
        }
        self.db.namespaces()
    }

    /// Commit `change` to the grant catalog, acting for the principal: the
    /// rights it needs are decided where the change applies, as through
    /// [`Database::change_access`].
    ///
    /// # Errors
    ///
    /// As [`Database::change_access`].
    pub fn change_access(&self, change: AccessChange) -> Result<(), DatabaseError> {
        self.db
            .propose_command_as(Some(self.principal.clone()), MetadataCommand::from(change))
    }
}

impl Database {
    /// The deployment's catalog as `principal` reaches it with no session on
    /// a namespace's data. The host vouches for the principal, as it does
    /// for a session's.
    #[must_use]
    pub fn catalog_scope(&self, principal: PrincipalId) -> CatalogScope<'_> {
        CatalogScope {
            db: self,
            principal,
        }
    }

    /// The grants on `namespace`, read from the catalog now, for `actor`: a
    /// principal reads them while it holds a right on that namespace.
    ///
    /// # Errors
    ///
    /// [`DatabaseError::AccessRefused`] for a principal holding no right on
    /// it, or naming a namespace that is not live (the refusal does not tell
    /// whether it exists); [`DatabaseError::UnknownNamespace`] for the
    /// trusted host naming one that is not; the catalog cannot be read.
    pub fn namespace_grants(
        &self,
        actor: &ActingContext,
        namespace: &str,
    ) -> Result<Vec<GrantView>, DatabaseError> {
        let principal = self.statement_principal(Some(actor))?;
        let deployment = self.engine.deployment();
        let id = self.readable_namespace(principal.as_ref(), namespace)?;
        let mut out = Vec::new();
        for (grantee, record) in access::namespace_grants(deployment, id)? {
            let grantee = match grantee {
                Grantee::Principal(principal) => GranteeName::Principal(principal),
                Grantee::Role(role) => {
                    let Some(record) = access::role_record(deployment, role)? else {
                        continue;
                    };
                    let Some(home) = metadata::namespace_record(deployment, record.home)? else {
                        continue;
                    };
                    GranteeName::Role {
                        home: home.name,
                        name: record.name,
                    }
                }
            };
            out.push(GrantView {
                grantee,
                role: record.role,
                changed_ts: record.changed_ts,
            });
        }
        Ok(out)
    }

    /// The holders of the deployment admin right, which reaches every
    /// namespace, read from the catalog now: empty when none is issued. A
    /// principal reads them while it holds a right on `namespace`, so it can
    /// establish that nobody can write there in its name over its head.
    ///
    /// # Errors
    ///
    /// As [`Self::namespace_grants`].
    pub fn deployment_admins(
        &self,
        actor: &ActingContext,
        namespace: &str,
    ) -> Result<Vec<DeploymentAdmin>, DatabaseError> {
        let principal = self.statement_principal(Some(actor))?;
        self.readable_namespace(principal.as_ref(), namespace)?;
        Ok(access::deployment_admins(self.engine.deployment())?
            .into_iter()
            .map(|(principal, rights)| DeploymentAdmin {
                principal,
                changed_ts: rights.changed_ts,
            })
            .collect())
    }

    /// The role `name` of the namespace `home` and who holds it, read from
    /// the catalog now; `None` for the trusted host naming a role that is
    /// not live. A principal reads it as its home's admin, as a holder, or
    /// as one of an exclusive role's eligible principals, whatever rights it
    /// holds on any namespace's data: a taker conditions its transfer on the
    /// holder it read.
    ///
    /// # Errors
    ///
    /// [`DatabaseError::AccessRefused`] for any other principal, and for a
    /// principal naming a role that is not live (the refusal does not tell
    /// whether it exists); the catalog cannot be read.
    pub fn role(
        &self,
        actor: &ActingContext,
        home: &str,
        name: &str,
    ) -> Result<Option<RoleView>, DatabaseError> {
        let principal = self.statement_principal(Some(actor))?;
        let deployment = self.engine.deployment();
        let home_id = metadata::namespace_by_name(deployment, home)?;
        let found = access::role_by_name(deployment, home, name)?;
        let view = match &found {
            Some((role, record)) => Some(RoleView {
                holders: access::role_holders(deployment, *role)?,
                eligible: record.eligible.clone(),
                changed_ts: record.changed_ts,
            }),
            None => None,
        };
        let Some(principal) = principal else {
            return Ok(view);
        };
        let refused = || {
            DatabaseError::AccessRefused(AccessRefusal {
                principal: principal.clone(),
                missing: home_id.map_or(MissingRight::DeploymentAdmin, |namespace| {
                    MissingRight::Namespace {
                        namespace,
                        role: NamespaceRole::Admin,
                    }
                }),
            })
        };
        let Some(view) = view else {
            return Err(refused());
        };
        let allowed = view.holders.contains(&principal)
            || view
                .eligible
                .as_ref()
                .is_some_and(|eligible| eligible.contains(&principal))
            || access::deployment_rights(deployment, &principal)?.admin
            || match home_id {
                Some(home) => {
                    access::effective_role(deployment, &principal, home)?
                        == Some(NamespaceRole::Admin)
                }
                None => false,
            };
        if allowed {
            Ok(Some(view))
        } else {
            Err(refused())
        }
    }

    /// The id of `namespace` when `principal` may read its grants: any right
    /// on it, or the deployment admin. The trusted host reads every live one.
    fn readable_namespace(
        &self,
        principal: Option<&PrincipalId>,
        namespace: &str,
    ) -> Result<NamespaceId, DatabaseError> {
        let deployment = self.engine.deployment();
        let found = metadata::namespace_by_name(deployment, namespace)?;
        let Some(principal) = principal else {
            return found.ok_or_else(|| DatabaseError::UnknownNamespace(namespace.to_owned()));
        };
        let refusal = |missing| {
            DatabaseError::AccessRefused(AccessRefusal {
                principal: principal.clone(),
                missing,
            })
        };
        let Some(id) = found else {
            return Err(refusal(MissingRight::DeploymentAdmin));
        };
        match access::may_read(deployment, principal, id)? {
            Ok(()) => Ok(id),
            Err(refused) => Err(DatabaseError::AccessRefused(refused)),
        }
    }
    /// Commit `change` to the grant catalog, acting for `actor`.
    ///
    /// A principal makes the change only where the log applies it and only
    /// with the rights it holds there: admin on each namespace the change
    /// grants on or revokes from, admin on a role's home namespace for its
    /// eligible set, and the deployment admin for deployment rights. Any
    /// principal of an exclusive role's eligible set may transfer the role
    /// to itself. The explicitly trusted host is checked against nothing.
    ///
    /// # Errors
    ///
    /// [`DatabaseError::AccessRefused`] when the principal lacks a right the
    /// change needs where the log applied it: nothing changed.
    /// [`DatabaseError::NamespaceMismatch`] for a principal bound to another
    /// namespace than this handle's. The write path's errors.
    pub fn change_access(
        &self,
        actor: &ActingContext,
        change: AccessChange,
    ) -> Result<(), DatabaseError> {
        let principal = self.statement_principal(Some(actor))?;
        self.propose_command_as(principal, MetadataCommand::from(change))
    }

    /// Commit the command `command` on its own, acting for `principal` when
    /// one is named.
    pub(super) fn propose_command_as(
        &self,
        principal: Option<PrincipalId>,
        command: MetadataCommand,
    ) -> Result<(), DatabaseError> {
        let mut mutations = Vec::with_capacity(2);
        // The acting principal is the unit's first operation: the member
        // applying the entry checks every later one against its rights.
        if let Some(principal) = principal {
            mutations.push(Mutation::Actor(principal));
        }
        mutations.push(Mutation::Command(command));
        self.propose_mutations(mutations)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
