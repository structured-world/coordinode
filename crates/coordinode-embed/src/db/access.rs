//! Changes to the deployment's grant catalog: who may read, write and
//! administer which namespace, and who holds which role.
//!
//! A change is a command of the deployment, committed through the same write
//! path as every other write and decided where it is applied: whether its
//! actor may make it, and whether its condition holds (a transfer from a
//! holder that no longer holds), is read from the catalog the log has
//! produced up to that entry, identically on every member.

use coordinode_core::access::PrincipalId;
use coordinode_core::txn::command::ActingContext;
use coordinode_core::txn::proposal::{AccessChange, MetadataCommand, Mutation};

use super::{Database, DatabaseError};

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
