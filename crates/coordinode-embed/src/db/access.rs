//! Changes to the deployment's grant catalog: who may read, write and
//! administer which namespace, and who holds which role.
//!
//! A change is a command of the deployment, committed through the same write
//! path as every other write and decided where it is applied: whether its
//! actor may make it, and whether its condition holds (a transfer from a
//! holder that no longer holds), is read from the catalog the log has
//! produced up to that entry, identically on every member.

use coordinode_core::access::{NamespaceRole, PrincipalId};
use coordinode_core::txn::command::ActingContext;
use coordinode_core::txn::proposal::{GranteeName, MetadataCommand, Mutation};

use super::{Database, DatabaseError};

/// One change to the grant catalog.
///
/// A change whose condition does not hold where the log applies it (a
/// transfer from a principal that no longer holds the right, a grant on a
/// namespace or role that is not live) writes nothing; the catalog read
/// afterwards says what holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccessChange {
    /// Give `grantee` the right `role` on `namespace`, replacing a right it
    /// held there.
    Grant {
        /// The namespace's name.
        namespace: String,
        /// Who receives the right.
        grantee: GranteeName,
        /// The right.
        role: NamespaceRole,
    },
    /// Take every right `grantee` holds on each of `namespaces`, as one
    /// change: no state of the store has it holding some and not the others.
    Revoke {
        /// The namespaces' names.
        namespaces: Vec<String>,
        /// Who loses the rights.
        grantee: GranteeName,
    },
    /// Move the right `role` on `namespace` from `from` to `to`, only while
    /// `from` holds exactly that right there.
    Transfer {
        /// The namespace's name.
        namespace: String,
        /// The right moved.
        role: NamespaceRole,
        /// The principal that holds it now.
        from: PrincipalId,
        /// The principal that receives it.
        to: PrincipalId,
    },
    /// Set the deployment rights of `principal`; each one left `None` is
    /// kept.
    DeploymentRights {
        /// The principal.
        principal: PrincipalId,
        /// Whether it may create namespaces.
        create_namespace: Option<bool>,
        /// Whether it holds every right on the whole deployment.
        admin: Option<bool>,
    },
    /// Create the role `name` in the namespace `home`. An exclusive role
    /// names its eligible principals; `holder`, when named, holds it from
    /// the start and must be eligible.
    CreateRole {
        /// The home namespace's name.
        home: String,
        /// The role's name.
        name: String,
        /// For an exclusive role, its eligible principals.
        eligible: Option<Vec<PrincipalId>>,
        /// The principal that holds it from the start.
        holder: Option<PrincipalId>,
    },
    /// Drop a role with every grant it holds and everyone holding it.
    DropRole {
        /// The home namespace's name.
        home: String,
        /// The role's name.
        name: String,
    },
    /// Replace an exclusive role's eligible principals; a holder no longer
    /// among them loses the role in the same change.
    RoleEligible {
        /// The home namespace's name.
        home: String,
        /// The role's name.
        name: String,
        /// The principals that may hold it from here on.
        eligible: Vec<PrincipalId>,
    },
    /// Make `to` a holder of a role: of an exclusive one only while nobody
    /// holds it and `to` is eligible.
    GrantRole {
        /// The home namespace's name.
        home: String,
        /// The role's name.
        name: String,
        /// The principal that receives it.
        to: PrincipalId,
    },
    /// Take a role from `from`: every right it carries, on every namespace,
    /// in one change.
    RevokeRole {
        /// The home namespace's name.
        home: String,
        /// The role's name.
        name: String,
        /// The principal that loses it.
        from: PrincipalId,
    },
    /// Move a role from `from` to `to`, only while `from` holds it and, for
    /// an exclusive role, `to` is eligible. Of two concurrent takers exactly
    /// one succeeds.
    TransferRole {
        /// The home namespace's name.
        home: String,
        /// The role's name.
        name: String,
        /// The principal that holds it now.
        from: PrincipalId,
        /// The principal that receives it.
        to: PrincipalId,
    },
}

impl From<AccessChange> for MetadataCommand {
    fn from(change: AccessChange) -> Self {
        match change {
            AccessChange::Grant {
                namespace,
                grantee,
                role,
            } => Self::GrantAccess {
                namespace,
                grantee,
                role,
            },
            AccessChange::Revoke {
                namespaces,
                grantee,
            } => Self::RevokeAccess {
                namespaces,
                grantee,
            },
            AccessChange::Transfer {
                namespace,
                role,
                from,
                to,
            } => Self::TransferAccess {
                namespace,
                role,
                from,
                to,
            },
            AccessChange::DeploymentRights {
                principal,
                create_namespace,
                admin,
            } => Self::SetDeploymentRights {
                principal,
                create_namespace,
                admin,
            },
            AccessChange::CreateRole {
                home,
                name,
                eligible,
                holder,
            } => Self::CreateRole {
                home,
                name,
                eligible,
                holder,
            },
            AccessChange::DropRole { home, name } => Self::DropRole { home, name },
            AccessChange::RoleEligible {
                home,
                name,
                eligible,
            } => Self::SetRoleEligible {
                home,
                name,
                eligible,
            },
            AccessChange::GrantRole { home, name, to } => Self::GrantRole { home, name, to },
            AccessChange::RevokeRole { home, name, from } => Self::RevokeRole { home, name, from },
            AccessChange::TransferRole {
                home,
                name,
                from,
                to,
            } => Self::TransferRole {
                home,
                name,
                from,
                to,
            },
        }
    }
}

impl Database {
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
