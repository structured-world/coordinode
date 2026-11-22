//! The grants a unit's principal holds, read where the unit applies.
//!
//! A unit that names an acting principal is checked against the grant
//! catalog the log has produced up to that unit, as one step with applying
//! it: every member applies the same log, so every member reaches the same
//! verdict. A unit whose principal lacks a right it needs applies as a
//! refusal that changes nothing. A unit with no principal is the store's own
//! work or a write of an explicitly trusted caller and is not checked here.

use coordinode_core::access::{
    AccessRefusal, DeploymentRights, GrantRecord, Grantee, MissingRight, NamespaceRole,
    PrincipalId, decode_held_role_key, deployment_rights_key, grant_key, held_roles_prefix,
};
use coordinode_core::namespace::NamespaceId;
use coordinode_core::txn::operation::LogPosition;
use coordinode_core::txn::proposal::{MetadataCommand, Mutation, PartitionId};

use crate::engine::core::Deployment;
use crate::engine::partition::Partition;
use crate::error::{StorageError, StorageResult};

/// What checking a unit's principal decided.
#[derive(Debug)]
pub(crate) enum Checked<'m> {
    /// The unit names no principal: it applies as it is.
    Unchecked(&'m [Mutation]),
    /// The principal holds every right the unit needs: these operations,
    /// the principal left out, apply.
    Granted(&'m [Mutation]),
    /// The principal lacks a right: nothing of the unit applies.
    Refused(AccessRefusal),
}

/// Check `mutations` against the grants `engine` holds now.
///
/// # Errors
///
/// A catalog record cannot be read or does not decode.
pub(crate) fn check_unit<'m>(
    engine: &Deployment,
    mutations: &'m [Mutation],
) -> StorageResult<Checked<'m>> {
    let Some((Mutation::Actor(principal), rest)) = mutations.split_first() else {
        return Ok(Checked::Unchecked(mutations));
    };
    let rights = deployment_rights(engine, principal)?;
    if rights.admin {
        return Ok(Checked::Granted(rest));
    }
    for need in needs(engine, rest)? {
        let held = match &need {
            MissingRight::Namespace { namespace, role } => {
                effective_role(engine, principal, *namespace)?.is_some_and(|held| held >= *role)
            }
            MissingRight::CreateNamespace => rights.create_namespace,
            MissingRight::DeploymentAdmin => false,
        };
        if !held {
            return Ok(Checked::Refused(AccessRefusal {
                principal: principal.clone(),
                missing: need,
            }));
        }
    }
    Ok(Checked::Granted(rest))
}

/// The rights `mutations` need, each namespace once at the highest role any
/// of its operations needs, in the order first needed.
fn needs(engine: &Deployment, mutations: &[Mutation]) -> StorageResult<Vec<MissingRight>> {
    let mut out: Vec<MissingRight> = Vec::new();
    let mut add = |need: MissingRight| match (&need, out.iter_mut().find(|n| same_scope(n, &need)))
    {
        (
            MissingRight::Namespace { role, .. },
            Some(MissingRight::Namespace { role: held, .. }),
        ) => {
            *held = (*held).max(*role);
        }
        (_, Some(_)) => {}
        (_, None) => out.push(need),
    };
    let mut namespace: Option<NamespaceId> = None;
    // A write of a namespace's data names the namespace before it; one with
    // no marker before it is refused by the apply as unscoped.
    let on = |namespace: Option<NamespaceId>, role| {
        namespace.map_or(MissingRight::DeploymentAdmin, |namespace| {
            MissingRight::Namespace { namespace, role }
        })
    };
    for mutation in mutations {
        match mutation {
            Mutation::Namespace(id) => namespace = Some(*id),
            Mutation::Put { partition, .. }
            | Mutation::Delete { partition, .. }
            | Mutation::Merge { partition, .. }
            | Mutation::RemoveRange { partition, .. } => add(match partition {
                // The deployment's own records are no tenant's to write.
                p if p.is_deployment() => MissingRight::DeploymentAdmin,
                // Label, index and constraint definitions.
                PartitionId::Schema => on(namespace, NamespaceRole::Admin),
                _ => on(namespace, NamespaceRole::Writer),
            }),
            Mutation::Command(command) => add(command_need(engine, command, namespace)?),
            // Index maintenance and the meaning of the unit's rows go with
            // the rows a writer writes.
            Mutation::Derive(_) | Mutation::Effects(_) => {
                add(on(namespace, NamespaceRole::Writer));
            }
            // Ownership of the operation is the issuer's, checked apart.
            Mutation::Operation(_) | Mutation::Actor(_) => {}
        }
    }
    Ok(out)
}

/// Whether two needs are of the same scope: the same namespace, or the same
/// deployment right.
fn same_scope(a: &MissingRight, b: &MissingRight) -> bool {
    match (a, b) {
        (
            MissingRight::Namespace { namespace: x, .. },
            MissingRight::Namespace { namespace: y, .. },
        ) => x == y,
        _ => a == b,
    }
}

/// The right a metadata command needs, under the namespace marker
/// `namespace`.
fn command_need(
    engine: &Deployment,
    command: &MetadataCommand,
    namespace: Option<NamespaceId>,
) -> StorageResult<MissingRight> {
    let on = |role| {
        namespace.map_or(MissingRight::DeploymentAdmin, |namespace| {
            MissingRight::Namespace { namespace, role }
        })
    };
    // A namespace named by name: its admin, or the deployment's admin when
    // no live namespace has the name, so a refusal does not tell a caller
    // without rights whether it exists.
    let admin_of = |name: &str| -> StorageResult<MissingRight> {
        Ok(
            match crate::engine::metadata::namespace_by_name(engine, name)? {
                Some(namespace) => MissingRight::Namespace {
                    namespace,
                    role: NamespaceRole::Admin,
                },
                None => MissingRight::DeploymentAdmin,
            },
        )
    };
    Ok(match command {
        // Names a write introduces, and the labels it uses first.
        MetadataCommand::RegisterFields { .. }
        | MetadataCommand::AdoptFields { .. }
        | MetadataCommand::BindLabels { .. } => on(NamespaceRole::Writer),
        MetadataCommand::CreateNamespace { .. } => MissingRight::CreateNamespace,
        MetadataCommand::DropNamespace { name }
        | MetadataCommand::SetNamespaceStorage { name, .. } => admin_of(name)?,
        // The store's own bookkeeping: proposed by the store, with no
        // principal.
        MetadataCommand::GrantNodeLease { .. }
        | MetadataCommand::RecordGroupPair { .. }
        | MetadataCommand::RecordDeployment { .. }
        | MetadataCommand::ConditionalRegistryWrite { .. } => MissingRight::DeploymentAdmin,
    })
}

/// The highest role `principal` holds on `namespace`: its own grant, or the
/// grant of a role it holds. `None` when it holds none.
///
/// # Errors
///
/// A catalog record cannot be read or does not decode.
pub fn effective_role(
    engine: &Deployment,
    principal: &PrincipalId,
    namespace: NamespaceId,
) -> StorageResult<Option<NamespaceRole>> {
    let mut best =
        grant(engine, namespace, &Grantee::Principal(principal.clone()))?.map(|grant| grant.role);
    if best == Some(NamespaceRole::Admin) {
        return Ok(best);
    }
    for guard in engine.system_prefix_scan(Partition::Deployment, &held_roles_prefix(principal))? {
        let (key, _) = guard.into_inner()?;
        let role = decode_held_role_key(principal, &key).ok_or_else(|| {
            StorageError::Serialization("a held-role record of the catalog does not decode".into())
        })?;
        if let Some(grant) = grant(engine, namespace, &Grantee::Role(role))? {
            best = best.max(Some(grant.role));
        }
    }
    Ok(best)
}

/// `grantee`'s grant on `namespace`, if it holds one.
///
/// # Errors
///
/// The record does not decode.
pub fn grant(
    engine: &Deployment,
    namespace: NamespaceId,
    grantee: &Grantee,
) -> StorageResult<Option<GrantRecord>> {
    engine
        .system_get(Partition::Deployment, &grant_key(namespace, grantee))?
        .map(|bytes| {
            GrantRecord::decode(&bytes).ok_or_else(|| {
                StorageError::Serialization(format!(
                    "the grant of {grantee:?} on {namespace} does not decode"
                ))
            })
        })
        .transpose()
}

/// `principal`'s rights over the whole deployment; none when it holds no
/// record.
///
/// # Errors
///
/// The record does not decode.
pub fn deployment_rights(
    engine: &Deployment,
    principal: &PrincipalId,
) -> StorageResult<DeploymentRights> {
    match engine.system_get(Partition::Deployment, &deployment_rights_key(principal))? {
        None => Ok(DeploymentRights::default()),
        Some(bytes) => DeploymentRights::decode(&bytes).ok_or_else(|| {
            StorageError::Serialization(format!(
                "the deployment rights of {principal:?} do not decode"
            ))
        }),
    }
}

/// Why the unit at log position `at` was applied as a refusal for its
/// principal's grants, if it was.
///
/// # Errors
///
/// The mark cannot be read, or holds bytes no refusal encodes.
pub fn unit_refusal(engine: &Deployment, at: LogPosition) -> StorageResult<Option<AccessRefusal>> {
    match engine.system_get(
        Partition::Deployment,
        &crate::engine::operations::skipped_unit_key(at),
    )? {
        // An empty mark is an operation binding's refusal.
        Some(bytes) if !bytes.is_empty() => {
            AccessRefusal::decode(&bytes).map(Some).ok_or_else(|| {
                StorageError::Serialization(format!("the refusal of unit {at:?} does not decode"))
            })
        }
        _ => Ok(None),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
