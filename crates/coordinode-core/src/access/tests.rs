use super::*;

/// A principal is both of its parts: the same subject under another trust
/// scope is another principal, and neither part may be empty or longer than
/// a log entry may carry.
#[test]
fn a_principal_is_its_subject_and_trust_scope() {
    let alice = PrincipalId::new("alice", "issuer-a").expect("alice");
    assert_eq!(alice, PrincipalId::new("alice", "issuer-a").expect("again"));
    assert_ne!(
        alice,
        PrincipalId::new("alice", "issuer-b").expect("other scope")
    );
    assert_eq!(
        PrincipalId::new("", "issuer-a"),
        Err(BadPrincipal {
            part: "subject",
            len: 0
        })
    );
    let long = "x".repeat(MAX_PRINCIPAL_PART_BYTES + 1);
    assert_eq!(
        PrincipalId::new("alice", &long),
        Err(BadPrincipal {
            part: "trust scope",
            len: MAX_PRINCIPAL_PART_BYTES + 1
        })
    );
    let edge = "x".repeat(MAX_PRINCIPAL_PART_BYTES);
    assert!(PrincipalId::new(&edge, &edge).is_ok());
}

fn principal(subject: &str, trust_scope: &str) -> PrincipalId {
    PrincipalId::new(subject, trust_scope).expect("principal")
}

const NS: NamespaceId = NamespaceId::from_raw(7);

/// Two principals whose parts join to the same text have different keys,
/// and neither key prefixes the other: otherwise a scan of one principal's
/// grants or roles would find the other's.
#[test]
fn a_principal_key_never_prefixes_another() {
    let a = principal("ab", "c");
    let b = principal("a", "bc");
    for (ka, kb) in [
        (
            grant_key(NS, &Grantee::Principal(a.clone())),
            grant_key(NS, &Grantee::Principal(b.clone())),
        ),
        (deployment_rights_key(&a), deployment_rights_key(&b)),
        (held_roles_prefix(&a), held_roles_prefix(&b)),
    ] {
        assert_ne!(ka, kb);
        assert!(
            !ka.starts_with(&kb) && !kb.starts_with(&ka),
            "{ka:?} {kb:?}"
        );
    }
    let role = RoleId::from_raw(3);
    assert_eq!(
        decode_held_role_key(&a, &held_role_key(&a, role)),
        Some(role)
    );
    assert_eq!(decode_held_role_key(&b, &held_role_key(&a, role)), None);
}

/// Every catalog key reads back as what it names, and a key of another
/// record does not.
#[test]
fn catalog_keys_read_back() {
    let alice = principal("alice", "issuer-a");
    let role = RoleId::from_raw(u64::MAX);
    for grantee in [Grantee::Principal(alice.clone()), Grantee::Role(role)] {
        assert_eq!(
            decode_grant_key(&grant_key(NS, &grantee)),
            Some((NS, grantee))
        );
    }
    assert_eq!(
        decode_deployment_rights_key(&deployment_rights_key(&alice)),
        Some(alice.clone())
    );
    assert_eq!(
        decode_role_holder_key(&role_holder_key(role, &alice)),
        Some((role, alice.clone()))
    );
    assert_eq!(decode_grant_key(&deployment_rights_key(&alice)), None);
    assert_eq!(decode_deployment_rights_key(&role_record_key(role)), None);
    // A grant key with bytes past the principal is no grant key.
    let mut long = grant_key(NS, &Grantee::Principal(alice));
    long.push(0);
    assert_eq!(decode_grant_key(&long), None);
}

/// Every record reads back as it was written, and bytes no encoder wrote are
/// refused rather than read as some record.
#[test]
fn catalog_records_read_back() {
    for role in [
        NamespaceRole::Reader,
        NamespaceRole::Writer,
        NamespaceRole::Admin,
    ] {
        let grant = GrantRecord {
            role,
            changed_ts: 1_790_000_000_000_123,
        };
        assert_eq!(GrantRecord::decode(&grant.encode()), Some(grant));
        assert_eq!(
            NamespaceRole::parse(&role.name().to_uppercase()),
            Some(role)
        );
    }
    assert_eq!(GrantRecord::decode(&[4, 0, 0, 0, 0, 0, 0, 0, 0]), None);
    assert_eq!(GrantRecord::decode(&[1, 0]), None);

    for (create_namespace, admin) in [(false, false), (true, false), (false, true), (true, true)] {
        let rights = DeploymentRights {
            create_namespace,
            admin,
            changed_ts: 9,
        };
        assert_eq!(DeploymentRights::decode(&rights.encode()), Some(rights));
    }
    assert_eq!(DeploymentRights::decode(&[4, 0, 0, 0, 0, 0, 0, 0, 0]), None);

    for eligible in [
        None,
        Some(Vec::new()),
        Some(vec![principal("p", "s"), principal("q", "s")]),
    ] {
        let record = RoleRecord {
            name: "coordinator".into(),
            home: NS,
            eligible,
            changed_ts: 42,
        };
        assert_eq!(RoleRecord::decode(&record.encode()), Some(record));
    }
    let mut bad = RoleRecord {
        name: "r".into(),
        home: NS,
        eligible: None,
        changed_ts: 1,
    }
    .encode();
    bad[16] = 2;
    assert_eq!(
        RoleRecord::decode(&bad),
        None,
        "an unknown exclusivity byte"
    );
    // A count no remaining bytes could hold.
    let mut huge = RoleRecord {
        name: "r".into(),
        home: NS,
        eligible: Some(Vec::new()),
        changed_ts: 1,
    }
    .encode();
    huge[17..21].copy_from_slice(&u32::MAX.to_be_bytes());
    assert_eq!(RoleRecord::decode(&huge), None);

    let alice = principal("alice", "issuer-a");
    for missing in [
        MissingRight::Namespace {
            namespace: NS,
            role: NamespaceRole::Writer,
        },
        MissingRight::CreateNamespace,
        MissingRight::DeploymentAdmin,
    ] {
        let refusal = AccessRefusal {
            principal: alice.clone(),
            missing,
        };
        assert_eq!(AccessRefusal::decode(&refusal.encode()), Some(refusal));
    }
    assert_eq!(AccessRefusal::decode(&[0x12]), None);
}
