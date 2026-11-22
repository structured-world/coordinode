//! A unit's principal checked against the grant catalog where the unit
//! applies, on the log's path and on the local ones.

use std::sync::Arc;

use coordinode_core::access::{
    DeploymentRights, GrantRecord, Grantee, MissingRight, NamespaceRole, PrincipalId, RoleId,
    deployment_rights_key, grant_key, held_role_key,
};
use coordinode_core::txn::proposal::{GranteeName, MetadataCommand};
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

fn principal(subject: &str) -> PrincipalId {
    PrincipalId::new(subject, "issuer-a").unwrap()
}

/// A replicated log over one store: each unit applies at the next index.
struct Log<'e> {
    engine: &'e StorageEngine,
    oracle: Arc<TimestampOracle>,
    index: u64,
}

impl Log<'_> {
    /// Apply `unit` at the next log position; its position and commit
    /// timestamp.
    fn apply(&mut self, unit: &[Mutation]) -> (LogPosition, u64) {
        self.index += 1;
        let ts = self.oracle.next().as_raw();
        self.engine
            .apply_raft_proposal(unit, ts, self.index, 0, |_| false)
            .unwrap();
        (
            LogPosition {
                index: self.index,
                sub: 0,
            },
            ts,
        )
    }

    /// Apply `command` acting for `actor`, or as the store's own when
    /// `None`; whether it was refused for the actor's grants.
    fn command(&mut self, actor: Option<&PrincipalId>, command: MetadataCommand) -> bool {
        let mut unit: Vec<Mutation> = actor
            .map(|p| Mutation::Actor(p.clone()))
            .into_iter()
            .collect();
        unit.push(Mutation::Command(command));
        let (at, _) = self.apply(&unit);
        unit_refusal(self.engine, at).unwrap().is_some()
    }

    /// Write `key` into `namespace` acting for `actor`; whether it landed.
    fn write(&mut self, actor: &PrincipalId, namespace: NamespaceId, key: &[u8]) -> bool {
        let unit = [
            Mutation::Actor(actor.clone()),
            Mutation::Namespace(namespace),
            Mutation::Put {
                partition: PartitionId::Node,
                key: key.to_vec(),
                value: b"v".to_vec(),
            },
        ];
        let (at, _) = self.apply(&unit);
        unit_refusal(self.engine, at).unwrap().is_none()
    }
}

/// Create a live namespace `name` as the store's own work; its id.
fn create(log: &mut Log<'_>, name: &str) -> NamespaceId {
    log.command(
        None,
        MetadataCommand::CreateNamespace {
            name: name.into(),
            token: crate::engine::metadata::namespace_token(),
            storage: Default::default(),
        },
    );
    crate::engine::metadata::namespace_by_name(log.engine, name)
        .unwrap()
        .expect("created")
}

fn grant_cmd(namespace: &str, grantee: GranteeName, role: NamespaceRole) -> MetadataCommand {
    MetadataCommand::GrantAccess {
        namespace: namespace.into(),
        grantee,
        role,
    }
}

/// A write issued before a revocation and ordered after it in the log is
/// refused: the grant is read where the write applies, never where it was
/// issued. A grant given back lets the next write through.
#[test]
fn a_revocation_earlier_in_the_log_refuses_every_later_write() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    let mut log = Log {
        engine: &engine,
        oracle,
        index: 0,
    };
    let a = create(&mut log, "a");
    let bob = principal("bob");
    let grantee = GranteeName::Principal(bob.clone());
    log.command(None, grant_cmd("a", grantee.clone(), NamespaceRole::Writer));
    assert!(log.write(&bob, a, b"node:1:1"));
    log.command(
        None,
        MetadataCommand::RevokeAccess {
            namespaces: vec!["a".into()],
            grantee: grantee.clone(),
        },
    );
    assert!(!log.write(&bob, a, b"node:1:2"), "a revoked writer wrote");
    let ns = engine.engine(a).unwrap();
    assert!(ns.get(Partition::Node, b"node:1:2").unwrap().is_none());
    log.command(None, grant_cmd("a", grantee, NamespaceRole::Writer));
    assert!(log.write(&bob, a, b"node:1:3"));
}

/// A revocation of a named level removes the direct grant only while it
/// still has that level where the revocation applies: a `REVOKE writer`
/// ordered after the grant became admin leaves the admin grant in place;
/// one ordered while it is writer removes it.
#[test]
fn a_named_revocation_spares_a_grant_replaced_before_it() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    let mut log = Log {
        engine: &engine,
        oracle,
        index: 0,
    };
    let a = create(&mut log, "a");
    let bob = principal("bob");
    let grantee = GranteeName::Principal(bob.clone());
    let revoke_writer = || MetadataCommand::RevokeGrant {
        namespace: "a".into(),
        grantee: GranteeName::Principal(principal("bob")),
        role: NamespaceRole::Writer,
    };

    log.command(None, grant_cmd("a", grantee.clone(), NamespaceRole::Writer));
    // An earlier change replaces the writer grant with admin; the revocation
    // naming writer applies after it.
    log.command(None, grant_cmd("a", grantee.clone(), NamespaceRole::Admin));
    log.command(None, revoke_writer());
    assert_eq!(
        effective_role(engine.deployment(), &bob, a).unwrap(),
        Some(NamespaceRole::Admin),
        "the admin grant the revocation did not name survives"
    );

    log.command(None, grant_cmd("a", grantee, NamespaceRole::Writer));
    log.command(None, revoke_writer());
    assert_eq!(
        effective_role(engine.deployment(), &bob, a).unwrap(),
        None,
        "the writer grant it named is gone"
    );
}

/// A principal holding only the right to create namespaces creates one and
/// becomes its admin in the same change, the grant stamped with the
/// change's commit timestamp; it has no right on a namespace it did not
/// create, nor may it drop one.
#[test]
fn the_creator_of_a_namespace_becomes_its_admin_at_once() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    let mut log = Log {
        engine: &engine,
        oracle,
        index: 0,
    };
    let other = create(&mut log, "other");
    let carol = principal("carol");
    log.command(
        None,
        MetadataCommand::SetDeploymentRights {
            principal: carol.clone(),
            create_namespace: Some(true),
            admin: None,
        },
    );
    let mut unit = vec![
        Mutation::Actor(carol.clone()),
        Mutation::Command(MetadataCommand::CreateNamespace {
            name: "mine".into(),
            token: crate::engine::metadata::namespace_token(),
            storage: Default::default(),
        }),
    ];
    let (at, ts) = log.apply(&unit);
    assert_eq!(unit_refusal(&engine, at).unwrap(), None);
    let mine = crate::engine::metadata::namespace_by_name(&engine, "mine")
        .unwrap()
        .expect("created");
    assert_eq!(
        grant(&engine, mine, &Grantee::Principal(carol.clone())).unwrap(),
        Some(GrantRecord {
            role: NamespaceRole::Admin,
            changed_ts: ts,
        })
    );
    assert!(log.write(&carol, mine, b"node:1:1"));
    assert!(
        !log.write(&carol, other, b"node:1:1"),
        "the creation right reached another namespace"
    );
    unit[1] = Mutation::Command(MetadataCommand::DropNamespace {
        name: "other".into(),
    });
    let (at, _) = log.apply(&unit);
    assert!(
        unit_refusal(&engine, at).unwrap().is_some(),
        "dropped a namespace it did not create"
    );
}

/// An exclusive role moves by transfer, every right it carries at once: of
/// two candidates taking it from the same holder, exactly one succeeds, the
/// other learns who holds it, and a taker repeating its transfer finds
/// itself the holder already. The old holder is refused from that position
/// on, and a principal outside the eligible set can neither take it nor
/// write.
#[test]
fn of_two_concurrent_takers_of_a_role_exactly_one_wins() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    let mut log = Log {
        engine: &engine,
        oracle,
        index: 0,
    };
    let home = create(&mut log, "home");
    let shared = create(&mut log, "shared");
    let (p, q, r, outsider) = (
        principal("p"),
        principal("q"),
        principal("r"),
        principal("x"),
    );
    log.command(
        None,
        MetadataCommand::CreateRole {
            home: "home".into(),
            name: "coordinator".into(),
            eligible: Some(vec![p.clone(), q.clone(), r.clone()]),
            holder: Some(p.clone()),
        },
    );
    let role = GranteeName::Role {
        home: "home".into(),
        name: "coordinator".into(),
    };
    for namespace in ["home", "shared"] {
        log.command(
            None,
            grant_cmd(namespace, role.clone(), NamespaceRole::Writer),
        );
    }
    assert!(log.write(&p, home, b"node:1:1") && log.write(&p, shared, b"node:1:1"));

    let take = |to: &PrincipalId| MetadataCommand::TransferRole {
        home: "home".into(),
        name: "coordinator".into(),
        from: p.clone(),
        to: to.clone(),
    };
    // Both read p as the holder; the log orders them.
    assert!(!log.command(Some(&q), take(&q)), "q may take it");
    assert!(!log.command(Some(&r), take(&r)), "r may attempt it");
    let (id, _) = role_by_name(&engine, "home", "coordinator")
        .unwrap()
        .unwrap();
    assert_eq!(role_holders(&engine, id).unwrap(), vec![q.clone()]);
    // A repeat after an uncertain outcome changes nothing.
    assert!(!log.command(Some(&q), take(&q)));
    assert_eq!(role_holders(&engine, id).unwrap(), vec![q.clone()]);

    for namespace in [home, shared] {
        assert!(log.write(&q, namespace, b"node:1:2"));
        assert!(
            !log.write(&p, namespace, b"node:1:3"),
            "the old holder wrote"
        );
        assert!(
            !log.write(&r, namespace, b"node:1:3"),
            "the losing taker wrote"
        );
    }
    // Outside the eligible set: no transfer to itself, no write.
    assert!(log.command(
        Some(&outsider),
        MetadataCommand::TransferRole {
            home: "home".into(),
            name: "coordinator".into(),
            from: q.clone(),
            to: outsider.clone(),
        }
    ));
    assert!(!log.write(&outsider, home, b"node:1:4"));
}

/// Revoking a principal's rights on two namespaces is one change: after it
/// neither namespace accepts its writes, and no position of the log has one
/// refusing while the other accepts.
#[test]
fn a_revocation_across_two_namespaces_is_one_change() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    let mut log = Log {
        engine: &engine,
        oracle,
        index: 0,
    };
    let a = create(&mut log, "a");
    let b = create(&mut log, "b");
    let dan = principal("dan");
    let grantee = GranteeName::Principal(dan.clone());
    for namespace in ["a", "b"] {
        log.command(
            None,
            grant_cmd(namespace, grantee.clone(), NamespaceRole::Writer),
        );
    }
    let before = log.index;
    log.command(
        None,
        MetadataCommand::RevokeAccess {
            namespaces: vec!["a".into(), "b".into()],
            grantee,
        },
    );
    assert_eq!(log.index, before + 1, "one log position");
    for namespace in [a, b] {
        assert_eq!(effective_role(&engine, &dan, namespace).unwrap(), None);
        assert!(!log.write(&dan, namespace, b"node:1:1"));
    }
}

/// A principal removed from a role's eligible set loses the role at that
/// position and cannot take it back; the set is changed by the home
/// namespace's admin alone; and the removal travels with the deletion of
/// what only that principal could read, in one change.
#[test]
fn removal_from_the_eligible_set_travels_with_the_data_it_closes() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    let mut log = Log {
        engine: &engine,
        oracle,
        index: 0,
    };
    let home = create(&mut log, "home");
    let (p, q) = (principal("p"), principal("q"));
    log.command(
        None,
        MetadataCommand::CreateRole {
            home: "home".into(),
            name: "reader-of-p".into(),
            eligible: Some(vec![p.clone(), q.clone()]),
            holder: Some(p.clone()),
        },
    );
    let ns = engine.engine(home).unwrap();
    log.apply(&[
        Mutation::Namespace(home),
        Mutation::Put {
            partition: PartitionId::Node,
            key: b"node:1:secret".to_vec(),
            value: b"v".to_vec(),
        },
    ]);
    // q is eligible but holds nothing: it cannot change the set.
    let set = |eligible: Vec<PrincipalId>| MetadataCommand::SetRoleEligible {
        home: "home".into(),
        name: "reader-of-p".into(),
        eligible,
    };
    assert!(
        log.command(Some(&q), set(vec![q.clone()])),
        "a candidate changed the set"
    );
    // The home's admin drops p and deletes p's data in one change.
    let admin = principal("admin");
    log.command(
        None,
        grant_cmd(
            "home",
            GranteeName::Principal(admin.clone()),
            NamespaceRole::Admin,
        ),
    );
    let (at, _) = log.apply(&[
        Mutation::Actor(admin.clone()),
        Mutation::Namespace(home),
        Mutation::Delete {
            partition: PartitionId::Node,
            key: b"node:1:secret".to_vec(),
        },
        Mutation::Command(set(vec![q.clone()])),
    ]);
    assert_eq!(unit_refusal(&engine, at).unwrap(), None);
    assert!(ns.get(Partition::Node, b"node:1:secret").unwrap().is_none());
    let (id, record) = role_by_name(&engine, "home", "reader-of-p")
        .unwrap()
        .unwrap();
    assert!(
        role_holders(&engine, id).unwrap().is_empty(),
        "p still holds the role"
    );
    assert_eq!(record.eligible, Some(vec![q.clone()]));
    // p cannot take the role back from the position of its removal.
    assert!(log.command(
        Some(&p),
        MetadataCommand::GrantRole {
            home: "home".into(),
            name: "reader-of-p".into(),
            to: p.clone(),
        }
    ));
}

/// A holder gives its role up without anyone's leave; a principal that
/// does not hold it cannot take it from another.
#[test]
fn a_holder_gives_its_role_up_on_its_own() {
    let dir = TempDir::new().unwrap();
    let (engine, oracle) = open(&dir);
    let mut log = Log {
        engine: &engine,
        oracle,
        index: 0,
    };
    let home = create(&mut log, "home");
    let (p, q) = (principal("p"), principal("q"));
    log.command(
        None,
        MetadataCommand::CreateRole {
            home: "home".into(),
            name: "worker".into(),
            eligible: None,
            holder: None,
        },
    );
    for to in [&p, &q] {
        log.command(
            None,
            MetadataCommand::GrantRole {
                home: "home".into(),
                name: "worker".into(),
                to: to.clone(),
            },
        );
    }
    log.command(
        None,
        grant_cmd(
            "home",
            GranteeName::Role {
                home: "home".into(),
                name: "worker".into(),
            },
            NamespaceRole::Writer,
        ),
    );
    let revoke = |from: &PrincipalId| MetadataCommand::RevokeRole {
        home: "home".into(),
        name: "worker".into(),
        from: from.clone(),
    };
    assert!(log.command(Some(&q), revoke(&p)), "q revoked p's role");
    assert!(log.write(&p, home, b"node:1:1"));
    assert!(
        !log.command(Some(&p), revoke(&p)),
        "p could not give its role up"
    );
    assert!(!log.write(&p, home, b"node:1:2"));
    assert!(log.write(&q, home, b"node:1:3"));
}
