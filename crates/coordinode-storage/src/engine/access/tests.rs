//! A unit's principal checked against the grant catalog where the unit
//! applies, on the log's path and on the local ones.

use std::sync::Arc;

use coordinode_core::access::{
    DeploymentRights, GrantRecord, Grantee, MissingRight, NamespaceRole, PrincipalId, RoleId,
    deployment_rights_key, grant_key, held_role_key,
};
use coordinode_core::txn::timestamp::TimestampOracle;
use tempfile::TempDir;

use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use crate::engine::core::StorageEngine;

fn config(dir: &TempDir) -> StorageConfig {
    StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )])
}

fn open(dir: &TempDir) -> (StorageEngine, Arc<TimestampOracle>) {
    let oracle = Arc::new(TimestampOracle::new());
    let engine = StorageEngine::open_with_oracle(&config(dir), Arc::clone(&oracle)).unwrap();
    (engine, oracle)
}

fn alice() -> PrincipalId {
    PrincipalId::new("alice", "issuer-a").unwrap()
}

const KEY: &[u8] = b"node:1:42";

/// One write of the store's node partition, acting for `principal`.
fn write_as(principal: &PrincipalId, partition: PartitionId) -> Vec<Mutation> {
    let mut unit = vec![Mutation::Actor(principal.clone())];
    unit.extend(crate::engine::scope::store_unit([Mutation::Put {
        partition,
        key: KEY.to_vec(),
        value: b"v".to_vec(),
    }]));
    unit
}

/// Give `grantee` `role` on the store's namespace, as a catalog change
/// would have.
fn grant_on(engine: &StorageEngine, grantee: &Grantee, role: NamespaceRole) {
    engine
        .system_put(
            Partition::Deployment,
            &grant_key(engine.namespace(), grantee),
            &GrantRecord {
                role,
                changed_ts: 1,
            }
            .encode(),
        )
        .unwrap();
}

fn holds(engine: &StorageEngine, partition: Partition) -> bool {
    engine.get(partition, KEY).unwrap().is_some()
}

/// A unit whose principal holds no grant on the namespace applies as a
/// refusal on the log's path: none of its writes land, and the mark at its
/// position says the unit took no effect and why, naming the principal, the
/// namespace and the least role that would have sufficed.
#[test]
fn a_unit_whose_principal_holds_no_grant_applies_nothing() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    let unit = write_as(&alice(), PartitionId::Node);
    engine
        .apply_raft_proposal(&unit, oracle.next().as_raw(), 5, 0, |_| false)
        .unwrap();
    assert!(!holds(&engine, Partition::Node));
    let at = LogPosition { index: 5, sub: 0 };
    assert!(crate::engine::operations::unit_skipped(&engine, at).unwrap());
    assert_eq!(
        unit_refusal(&engine, at).unwrap(),
        Some(AccessRefusal {
            principal: alice(),
            missing: MissingRight::Namespace {
                namespace: engine.namespace(),
                role: NamespaceRole::Writer,
            },
        })
    );
}

/// A writer's unit applies, the principal left out of what lands; a
/// reader's does not.
#[test]
fn a_writer_writes_and_a_reader_does_not() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    grant_on(&engine, &Grantee::Principal(alice()), NamespaceRole::Reader);
    engine
        .apply_raft_proposal(
            &write_as(&alice(), PartitionId::Node),
            oracle.next().as_raw(),
            1,
            0,
            |_| false,
        )
        .unwrap();
    assert!(!holds(&engine, Partition::Node), "a reader wrote");

    grant_on(&engine, &Grantee::Principal(alice()), NamespaceRole::Writer);
    engine
        .apply_raft_proposal(
            &write_as(&alice(), PartitionId::Node),
            oracle.next().as_raw(),
            2,
            0,
            |_| false,
        )
        .unwrap();
    assert!(holds(&engine, Partition::Node), "a writer did not write");
    let at = LogPosition { index: 2, sub: 0 };
    assert!(!crate::engine::operations::unit_skipped(&engine, at).unwrap());
}

/// A writer cannot change the namespace's schema; its admin can.
#[test]
fn a_schema_change_needs_admin() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    grant_on(&engine, &Grantee::Principal(alice()), NamespaceRole::Writer);
    let refused = engine.apply_proposal_at(
        &write_as(&alice(), PartitionId::Schema),
        oracle.next().as_raw(),
    );
    assert!(
        matches!(
            &refused,
            Err(StorageError::AccessRefused(AccessRefusal {
                missing: MissingRight::Namespace {
                    role: NamespaceRole::Admin,
                    ..
                },
                ..
            }))
        ),
        "{refused:?}"
    );
    assert!(!holds(&engine, Partition::Schema));
    grant_on(&engine, &Grantee::Principal(alice()), NamespaceRole::Admin);
    engine
        .apply_proposal_at(
            &write_as(&alice(), PartitionId::Schema),
            oracle.next().as_raw(),
        )
        .unwrap();
    assert!(holds(&engine, Partition::Schema));
}

/// A role a principal holds carries the role's grants; without holding it
/// the principal has none of them.
#[test]
fn a_held_role_carries_its_grants() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    let role = RoleId::from_raw(1);
    grant_on(&engine, &Grantee::Role(role), NamespaceRole::Writer);
    let refused = engine.apply_proposal_at(
        &write_as(&alice(), PartitionId::Node),
        oracle.next().as_raw(),
    );
    assert!(matches!(refused, Err(StorageError::AccessRefused(_))));
    engine
        .system_put(
            Partition::Deployment,
            &held_role_key(&alice(), role),
            &1u64.to_be_bytes(),
        )
        .unwrap();
    engine
        .apply_proposal_at(
            &write_as(&alice(), PartitionId::Node),
            oracle.next().as_raw(),
        )
        .unwrap();
    assert!(holds(&engine, Partition::Node));
}

/// The deployment's admin writes every namespace, and the deployment's own
/// records, with no grant on any namespace; the right to create namespaces
/// alone writes neither.
#[test]
fn the_deployment_admin_reaches_every_namespace() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    let set = |create_namespace, admin| {
        engine
            .system_put(
                Partition::Deployment,
                &deployment_rights_key(&alice()),
                &DeploymentRights {
                    create_namespace,
                    admin,
                    changed_ts: 1,
                }
                .encode(),
            )
            .unwrap();
    };
    set(true, false);
    let refused = engine.apply_proposal_at(
        &write_as(&alice(), PartitionId::Node),
        oracle.next().as_raw(),
    );
    assert!(matches!(refused, Err(StorageError::AccessRefused(_))));
    set(false, true);
    engine
        .apply_proposal_at(
            &write_as(&alice(), PartitionId::Node),
            oracle.next().as_raw(),
        )
        .unwrap();
    assert!(holds(&engine, Partition::Node));
}

/// On the journalled path a refused unit journals nothing: the store opened
/// again replays no write of it.
#[test]
fn a_refused_journalled_unit_is_not_replayed() {
    let dir = TempDir::new().unwrap();
    let oracle = Arc::new(TimestampOracle::new());
    {
        let engine = StorageEngine::open_embedded(&config(&dir), Arc::clone(&oracle)).unwrap();
        let refused = engine.commit_journaled(
            &write_as(&alice(), PartitionId::Node),
            oracle.next().as_raw(),
        );
        assert!(matches!(refused, Err(StorageError::AccessRefused(_))));
        assert!(!holds(&engine, Partition::Node));
    }
    let engine = StorageEngine::open_embedded(&config(&dir), Arc::clone(&oracle)).unwrap();
    assert!(!holds(&engine, Partition::Node));
}

/// A unit that names no principal is not checked: it applies as before,
/// with no grant anywhere.
#[test]
fn a_unit_without_a_principal_is_not_checked() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    engine
        .apply_proposal_at(
            &write_as(&alice(), PartitionId::Node)[1..],
            oracle.next().as_raw(),
        )
        .unwrap();
    assert!(holds(&engine, Partition::Node));
}
