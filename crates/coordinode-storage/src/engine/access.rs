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
    PrincipalId, ROLE_ALLOCATOR_KEY, ROLE_GRANT_PREFIX, RoleId, RoleRecord, decode_grant_key,
    decode_held_role_key, decode_role_holder_key, deployment_rights_key, grant_key, held_role_key,
    held_roles_prefix, role_holder_key, role_holders_prefix, role_name_key, role_record_key,
};
use coordinode_core::namespace::NamespaceId;
use coordinode_core::txn::operation::LogPosition;
use coordinode_core::txn::proposal::{GranteeName, MetadataCommand, Mutation, PartitionId};

use crate::engine::core::Deployment;
use crate::engine::partition::Partition;
use crate::error::{StorageError, StorageResult};

/// What checking a unit's principal decided.
#[derive(Debug)]
pub(crate) enum Checked<'m> {
    /// The unit names no principal: it applies as it is.
    Unchecked(&'m [Mutation]),
    /// The principal holds every right the unit needs: these operations,
    /// the principal left out, apply, decided on its behalf.
    Granted {
        /// The principal the unit acts for.
        principal: &'m PrincipalId,
        /// The unit's other operations.
        rest: &'m [Mutation],
    },
    /// The principal lacks a right: nothing of the unit applies.
    Refused(AccessRefusal),
}

impl<'m> Checked<'m> {
    /// The operations that apply and the principal they act for, or the
    /// refusal.
    pub(crate) fn applying(
        self,
    ) -> Result<(Option<&'m PrincipalId>, &'m [Mutation]), AccessRefusal> {
        match self {
            Self::Unchecked(m) => Ok((None, m)),
            Self::Granted { principal, rest } => Ok((Some(principal), rest)),
            Self::Refused(refusal) => Err(refusal),
        }
    }
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
        return Ok(Checked::Granted { principal, rest });
    }
    for need in needs(engine, principal, rest)? {
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
    Ok(Checked::Granted { principal, rest })
}

/// The rights `mutations`, acting for `principal`, need: each namespace once
/// at the highest role any of its operations needs, in the order first
/// needed.
fn needs(
    engine: &Deployment,
    principal: &PrincipalId,
    mutations: &[Mutation],
) -> StorageResult<Vec<MissingRight>> {
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
            Mutation::Command(command) => {
                for need in command_needs(engine, principal, command, namespace)? {
                    add(need);
                }
            }
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

/// The rights a metadata command acting for `principal` needs, under the
/// namespace marker `namespace`.
fn command_needs(
    engine: &Deployment,
    principal: &PrincipalId,
    command: &MetadataCommand,
    namespace: Option<NamespaceId>,
) -> StorageResult<Vec<MissingRight>> {
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
        | MetadataCommand::BindLabels { .. } => vec![on(NamespaceRole::Writer)],
        MetadataCommand::CreateNamespace { .. } => vec![MissingRight::CreateNamespace],
        MetadataCommand::DropNamespace { name }
        | MetadataCommand::SetNamespaceStorage { name, .. }
        | MetadataCommand::TransferAccess {
            namespace: name, ..
        } => vec![admin_of(name)?],
        // A role's grant on a namespace is that namespace's admin's to give;
        // it gives no say over who holds the role.
        MetadataCommand::GrantAccess {
            namespace: name, ..
        }
        | MetadataCommand::RevokeGrant {
            namespace: name, ..
        } => vec![admin_of(name)?],
        MetadataCommand::RevokeAccess { namespaces, .. } => namespaces
            .iter()
            .map(|name| admin_of(name))
            .collect::<StorageResult<_>>()?,
        MetadataCommand::SetDeploymentRights { .. } => vec![MissingRight::DeploymentAdmin],
        // Who may hold a role is its home namespace's admin's alone.
        MetadataCommand::CreateRole { home, .. }
        | MetadataCommand::DropRole { home, .. }
        | MetadataCommand::SetRoleEligible { home, .. }
        | MetadataCommand::GrantRole { home, .. } => vec![admin_of(home)?],
        // A holder gives its role up without anyone's leave.
        MetadataCommand::RevokeRole { home, from, .. } => {
            if from == principal {
                Vec::new()
            } else {
                vec![admin_of(home)?]
            }
        }
        // A principal of an exclusive role's eligible set takes the role for
        // itself; whether the holder it names still holds it is decided as
        // the transfer applies. Anyone else needs the home's admin.
        MetadataCommand::TransferRole { home, name, to, .. } => {
            let eligible_self = to == principal
                && role_by_name(engine, home, name)?.is_some_and(|(_, record)| {
                    record
                        .eligible
                        .as_ref()
                        .is_some_and(|eligible| eligible.contains(principal))
                });
            if eligible_self {
                Vec::new()
            } else {
                vec![admin_of(home)?]
            }
        }
        // The store's own bookkeeping: proposed by the store, with no
        // principal.
        MetadataCommand::GrantNodeLease { .. }
        | MetadataCommand::RecordGroupPair { .. }
        | MetadataCommand::RecordDeployment { .. }
        | MetadataCommand::ConditionalRegistryWrite { .. } => vec![MissingRight::DeploymentAdmin],
    })
}

/// The live role called `name` in the live namespace called `home`, with
/// its record.
///
/// # Errors
///
/// A catalog record cannot be read or does not decode.
pub fn role_by_name(
    engine: &Deployment,
    home: &str,
    name: &str,
) -> StorageResult<Option<(RoleId, RoleRecord)>> {
    let Some(home) = crate::engine::metadata::namespace_by_name(engine, home)? else {
        return Ok(None);
    };
    let Some(bytes) = engine.system_get(Partition::Deployment, &role_name_key(home, name))? else {
        return Ok(None);
    };
    let id = <[u8; 8]>::try_from(bytes.as_ref())
        .map(|raw| RoleId::from_raw(u64::from_be_bytes(raw)))
        .map_err(|_| {
            StorageError::Serialization(format!("the role entry of {name:?} does not decode"))
        })?;
    let record = role_record(engine, id)?.ok_or_else(|| {
        StorageError::Serialization(format!("role {name:?} is named without a record"))
    })?;
    Ok(Some((id, record)))
}

/// The catalog's record of `role`, if it has one.
///
/// # Errors
///
/// The record does not decode.
pub fn role_record(engine: &Deployment, role: RoleId) -> StorageResult<Option<RoleRecord>> {
    engine
        .system_get(Partition::Deployment, &role_record_key(role))?
        .map(|bytes| {
            RoleRecord::decode(&bytes).ok_or_else(|| {
                StorageError::Serialization(format!(
                    "the record of role {} does not decode",
                    role.as_raw()
                ))
            })
        })
        .transpose()
}

/// Every principal holding `role`, in key order.
///
/// # Errors
///
/// A holder record does not decode.
pub fn role_holders(engine: &Deployment, role: RoleId) -> StorageResult<Vec<PrincipalId>> {
    let mut out = Vec::new();
    for guard in engine.system_prefix_scan(Partition::Deployment, &role_holders_prefix(role))? {
        let (key, _) = guard.into_inner()?;
        let (_, principal) = decode_role_holder_key(&key).ok_or_else(|| {
            StorageError::Serialization("a role holder record does not decode".into())
        })?;
        out.push(principal);
    }
    Ok(out)
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

/// Whether `principal` may read `namespace` by the catalog `engine` holds
/// now: a deployment admin, or a principal holding at least reader there.
/// The refusal names the reader role it lacks.
///
/// # Errors
///
/// A catalog record cannot be read or does not decode.
pub fn may_read(
    engine: &Deployment,
    principal: &PrincipalId,
    namespace: NamespaceId,
) -> StorageResult<Result<(), AccessRefusal>> {
    if deployment_rights(engine, principal)?.admin
        || effective_role(engine, principal, namespace)?.is_some()
    {
        return Ok(Ok(()));
    }
    Ok(Err(AccessRefusal {
        principal: principal.clone(),
        missing: MissingRight::Namespace {
            namespace,
            role: NamespaceRole::Reader,
        },
    }))
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

fn put(key: Vec<u8>, value: Vec<u8>) -> Mutation {
    Mutation::Put {
        partition: PartitionId::Deployment,
        key,
        value,
    }
}

fn delete(key: Vec<u8>) -> Mutation {
    Mutation::Delete {
        partition: PartitionId::Deployment,
        key,
    }
}

/// The grantee a statement names, resolved against the catalog now: `None`
/// for a role that is not live.
fn resolve_grantee(engine: &Deployment, grantee: &GranteeName) -> StorageResult<Option<Grantee>> {
    Ok(match grantee {
        GranteeName::Principal(principal) => Some(Grantee::Principal(principal.clone())),
        GranteeName::Role { home, name } => {
            role_by_name(engine, home, name)?.map(|(id, _)| Grantee::Role(id))
        }
    })
}

/// The effects of the grant-catalog command `command`, committing at
/// `commit_ts`, against the catalog `engine` holds now. An empty result is a
/// refusal: the command writes nothing. `None` for a command that is not a
/// grant-catalog command.
///
/// # Errors
///
/// A catalog record cannot be read or does not decode.
pub(crate) fn decide(
    engine: &Deployment,
    command: &MetadataCommand,
    commit_ts: u64,
) -> StorageResult<Option<Vec<Mutation>>> {
    use crate::engine::metadata::namespace_by_name;
    let live = |name: &str| namespace_by_name(engine, name);
    Ok(Some(match command {
        MetadataCommand::GrantAccess {
            namespace,
            grantee,
            role,
        } => {
            let (Some(namespace), Some(grantee)) =
                (live(namespace)?, resolve_grantee(engine, grantee)?)
            else {
                return Ok(Some(Vec::new()));
            };
            // Granting what is held already changes nothing: the grant keeps
            // the timestamp of the change that made it.
            if grant(engine, namespace, &grantee)?.is_some_and(|held| held.role == *role) {
                return Ok(Some(Vec::new()));
            }
            vec![put(
                grant_key(namespace, &grantee),
                GrantRecord {
                    role: *role,
                    changed_ts: commit_ts,
                }
                .encode(),
            )]
        }
        MetadataCommand::RevokeAccess {
            namespaces,
            grantee,
        } => {
            let Some(grantee) = resolve_grantee(engine, grantee)? else {
                return Ok(Some(Vec::new()));
            };
            let mut effects = Vec::with_capacity(namespaces.len());
            for name in namespaces {
                let Some(namespace) = live(name)? else {
                    // One namespace not live refuses the whole revocation:
                    // its rights go together or not at all.
                    return Ok(Some(Vec::new()));
                };
                if grant(engine, namespace, &grantee)?.is_some() {
                    effects.push(delete(grant_key(namespace, &grantee)));
                }
            }
            effects
        }
        MetadataCommand::RevokeGrant {
            namespace,
            grantee,
            role,
        } => {
            let (Some(namespace), Some(grantee)) =
                (live(namespace)?, resolve_grantee(engine, grantee)?)
            else {
                return Ok(Some(Vec::new()));
            };
            // The level named is matched here, where the change applies: a
            // grant an earlier change already replaced with another level is
            // not the one this revocation names, and survives it.
            if !grant(engine, namespace, &grantee)?.is_some_and(|held| held.role == *role) {
                return Ok(Some(Vec::new()));
            }
            vec![delete(grant_key(namespace, &grantee))]
        }
        MetadataCommand::TransferAccess {
            namespace,
            role,
            from,
            to,
        } => {
            let Some(namespace) = live(namespace)? else {
                return Ok(Some(Vec::new()));
            };
            let from = Grantee::Principal(from.clone());
            if !grant(engine, namespace, &from)?.is_some_and(|held| held.role == *role) {
                return Ok(Some(Vec::new()));
            }
            let to = Grantee::Principal(to.clone());
            let received =
                grant(engine, namespace, &to)?.map_or(*role, |held| held.role.max(*role));
            vec![
                delete(grant_key(namespace, &from)),
                put(
                    grant_key(namespace, &to),
                    GrantRecord {
                        role: received,
                        changed_ts: commit_ts,
                    }
                    .encode(),
                ),
            ]
        }
        MetadataCommand::SetDeploymentRights {
            principal,
            create_namespace,
            admin,
        } => {
            let held = deployment_rights(engine, principal)?;
            let rights = DeploymentRights {
                create_namespace: create_namespace.unwrap_or(held.create_namespace),
                admin: admin.unwrap_or(held.admin),
                changed_ts: commit_ts,
            };
            if (rights.create_namespace, rights.admin) == (held.create_namespace, held.admin) {
                return Ok(Some(Vec::new()));
            }
            let key = deployment_rights_key(principal);
            if !rights.create_namespace && !rights.admin {
                vec![delete(key)]
            } else {
                vec![put(key, rights.encode())]
            }
        }
        MetadataCommand::CreateRole {
            home,
            name,
            eligible,
            holder,
        } => decide_role_creation(
            engine,
            home,
            name,
            eligible.as_deref(),
            holder.as_ref(),
            commit_ts,
        )?,
        MetadataCommand::DropRole { home, name } => {
            let Some((role, record)) = role_by_name(engine, home, name)? else {
                return Ok(Some(Vec::new()));
            };
            let mut effects = vec![
                delete(role_record_key(role)),
                delete(role_name_key(record.home, &record.name)),
            ];
            for holder in role_holders(engine, role)? {
                effects.push(delete(role_holder_key(role, &holder)));
                effects.push(delete(held_role_key(&holder, role)));
            }
            for guard in engine.system_prefix_scan(Partition::Deployment, ROLE_GRANT_PREFIX)? {
                let (key, _) = guard.into_inner()?;
                if matches!(decode_grant_key(&key), Some((_, Grantee::Role(r))) if r == role) {
                    effects.push(delete(key.to_vec()));
                }
            }
            effects
        }
        MetadataCommand::SetRoleEligible {
            home,
            name,
            eligible,
        } => {
            let Some((role, record)) = role_by_name(engine, home, name)? else {
                return Ok(Some(Vec::new()));
            };
            if record.eligible.is_none() {
                return Ok(Some(Vec::new()));
            }
            let mut eligible = eligible.clone();
            eligible.sort();
            eligible.dedup();
            let mut effects = Vec::new();
            for holder in role_holders(engine, role)? {
                if !eligible.contains(&holder) {
                    effects.push(delete(role_holder_key(role, &holder)));
                    effects.push(delete(held_role_key(&holder, role)));
                }
            }
            effects.push(put(
                role_record_key(role),
                RoleRecord {
                    eligible: Some(eligible),
                    changed_ts: commit_ts,
                    ..record
                }
                .encode(),
            ));
            effects
        }
        MetadataCommand::GrantRole { home, name, to } => {
            let Some((role, record)) = role_by_name(engine, home, name)? else {
                return Ok(Some(Vec::new()));
            };
            let holders = role_holders(engine, role)?;
            if holders.contains(to) {
                return Ok(Some(Vec::new()));
            }
            if let Some(eligible) = &record.eligible
                && (!holders.is_empty() || !eligible.contains(to))
            {
                return Ok(Some(Vec::new()));
            }
            hold(role, record, to, commit_ts)
        }
        MetadataCommand::RevokeRole { home, name, from } => {
            let Some((role, record)) = role_by_name(engine, home, name)? else {
                return Ok(Some(Vec::new()));
            };
            if !role_holders(engine, role)?.contains(from) {
                return Ok(Some(Vec::new()));
            }
            vec![
                delete(role_holder_key(role, from)),
                delete(held_role_key(from, role)),
                put(
                    role_record_key(role),
                    RoleRecord {
                        changed_ts: commit_ts,
                        ..record
                    }
                    .encode(),
                ),
            ]
        }
        MetadataCommand::TransferRole {
            home,
            name,
            from,
            to,
        } => {
            let Some((role, record)) = role_by_name(engine, home, name)? else {
                return Ok(Some(Vec::new()));
            };
            let holders = role_holders(engine, role)?;
            // A transfer to the holder it names is not a change: a repeat
            // after an uncertain outcome finds itself the holder already.
            if from == to
                || !holders.contains(from)
                || holders.contains(to)
                || record
                    .eligible
                    .as_ref()
                    .is_some_and(|eligible| !eligible.contains(to))
            {
                return Ok(Some(Vec::new()));
            }
            let mut effects = vec![
                delete(role_holder_key(role, from)),
                delete(held_role_key(from, role)),
            ];
            effects.extend(hold(role, record, to, commit_ts));
            effects
        }
        _ => return Ok(None),
    }))
}

/// The records making `to` a holder of `role` at `commit_ts`.
fn hold(role: RoleId, record: RoleRecord, to: &PrincipalId, commit_ts: u64) -> Vec<Mutation> {
    let since = commit_ts.to_be_bytes().to_vec();
    vec![
        put(role_holder_key(role, to), since.clone()),
        put(held_role_key(to, role), since),
        put(
            role_record_key(role),
            RoleRecord {
                changed_ts: commit_ts,
                ..record
            }
            .encode(),
        ),
    ]
}

/// The records creating the role `name` in `home`.
fn decide_role_creation(
    engine: &Deployment,
    home: &str,
    name: &str,
    eligible: Option<&[PrincipalId]>,
    holder: Option<&PrincipalId>,
    commit_ts: u64,
) -> StorageResult<Vec<Mutation>> {
    let Some(home_id) = crate::engine::metadata::namespace_by_name(engine, home)? else {
        return Ok(Vec::new());
    };
    if coordinode_core::namespace::validate_namespace_name(name).is_err()
        || role_by_name(engine, home, name)?.is_some()
    {
        return Ok(Vec::new());
    }
    let eligible = eligible.map(|eligible| {
        let mut eligible = eligible.to_vec();
        eligible.sort();
        eligible.dedup();
        eligible
    });
    // An exclusive role has exactly one holder, out of its eligible set.
    if let Some(eligible) = &eligible
        && !holder.is_some_and(|holder| eligible.contains(holder))
    {
        return Ok(Vec::new());
    }
    let next = match engine.system_get(Partition::Deployment, ROLE_ALLOCATOR_KEY)? {
        None => 1,
        Some(bytes) => <[u8; 8]>::try_from(bytes.as_ref())
            .map(u64::from_be_bytes)
            .map_err(|_| {
                StorageError::Serialization("the role allocator does not decode".into())
            })?,
    };
    let Some(after) = next.checked_add(1) else {
        tracing::warn!(name, "role creation refused: the ids are exhausted");
        return Ok(Vec::new());
    };
    let role = RoleId::from_raw(next);
    let record = RoleRecord {
        name: name.to_owned(),
        home: home_id,
        eligible,
        changed_ts: commit_ts,
    };
    let mut effects = vec![
        put(ROLE_ALLOCATOR_KEY.to_vec(), after.to_be_bytes().to_vec()),
        put(role_name_key(home_id, name), next.to_be_bytes().to_vec()),
    ];
    match holder {
        Some(holder) => effects.extend(hold(role, record, holder, commit_ts)),
        None => effects.push(put(role_record_key(role), record.encode())),
    }
    Ok(effects)
}

/// The admin grant a namespace's creator receives on it, in the same change
/// that creates it.
pub(crate) fn creator_grant(
    namespace: NamespaceId,
    creator: &PrincipalId,
    commit_ts: u64,
) -> Mutation {
    put(
        grant_key(namespace, &Grantee::Principal(creator.clone())),
        GrantRecord {
            role: NamespaceRole::Admin,
            changed_ts: commit_ts,
        }
        .encode(),
    )
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
