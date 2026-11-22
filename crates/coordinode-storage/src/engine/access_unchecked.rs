//! A store built without `tenancy` decides access by who can open its files:
//! it holds no grant catalog and checks no principal. A unit naming one, as
//! a store written by a build with the feature may carry in its log, applies
//! as any other; a grant-catalog command applies nothing.

use coordinode_core::access::{AccessRefusal, PrincipalId};
use coordinode_core::txn::operation::LogPosition;
use coordinode_core::txn::proposal::{MetadataCommand, Mutation};

use crate::engine::core::Deployment;
use crate::error::StorageResult;

/// What applying a unit decided: here always to apply it.
#[derive(Debug)]
pub(crate) enum Checked<'m> {
    /// The unit's operations, its principal left out.
    Unchecked(&'m [Mutation]),
}

impl<'m> Checked<'m> {
    /// The operations that apply, acting for no principal.
    pub(crate) fn applying(
        self,
    ) -> Result<(Option<&'m PrincipalId>, &'m [Mutation]), AccessRefusal> {
        match self {
            Self::Unchecked(m) => Ok((None, m)),
        }
    }
}

/// `mutations` with the principal it names, if any, left out.
///
/// # Errors
///
/// Never; the signature is the checked build's.
#[allow(clippy::unnecessary_wraps)]
pub(crate) fn check_unit<'m>(
    _engine: &Deployment,
    mutations: &'m [Mutation],
) -> StorageResult<Checked<'m>> {
    Ok(Checked::Unchecked(match mutations.split_first() {
        Some((Mutation::Actor(_), rest)) => rest,
        _ => mutations,
    }))
}

/// Why the unit at `at` was applied as a refusal for its principal's
/// grants: never, as no principal is checked.
///
/// # Errors
///
/// Never; the signature is the checked build's.
#[allow(clippy::unnecessary_wraps)]
pub fn unit_refusal(
    _engine: &Deployment,
    _at: LogPosition,
) -> StorageResult<Option<AccessRefusal>> {
    Ok(None)
}

/// A grant-catalog command applies nothing; `None` for any other command.
///
/// # Errors
///
/// Never; the signature is the checked build's.
#[allow(clippy::unnecessary_wraps)]
pub(crate) fn decide(
    _engine: &Deployment,
    command: &MetadataCommand,
    _commit_ts: u64,
) -> StorageResult<Option<Vec<Mutation>>> {
    Ok(matches!(
        command,
        MetadataCommand::GrantAccess { .. }
            | MetadataCommand::RevokeAccess { .. }
            | MetadataCommand::RevokeGrant { .. }
            | MetadataCommand::TransferAccess { .. }
            | MetadataCommand::SetDeploymentRights { .. }
            | MetadataCommand::CreateRole { .. }
            | MetadataCommand::DropRole { .. }
            | MetadataCommand::SetRoleEligible { .. }
            | MetadataCommand::GrantRole { .. }
            | MetadataCommand::RevokeRole { .. }
            | MetadataCommand::TransferRole { .. }
    )
    .then(Vec::new))
}
